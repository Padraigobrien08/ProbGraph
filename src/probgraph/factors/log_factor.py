"""Factors stored as logarithms, so tiny and huge values stay representable.

The derivation (P9), including the max-shift lemma for log-sum-exp, is in
``docs/mathematics/log_space.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from probgraph._arrays import as_float_array
from probgraph.exceptions import NormalisationError, ValidationError
from probgraph.factors._named_table import _NamedTable, _validate_variables
from probgraph.factors.discrete_factor import DiscreteFactor
from probgraph.variables import DiscreteVariable

if TYPE_CHECKING:
    from probgraph.distributions import TabularCPD

#: Default absolute tolerance, in log space (a relative tolerance on values), for ``allclose``.
LOG_FACTOR_ATOL = 1e-12


class LogFactor(_NamedTable):
    """log phi(x_S): a factor stored as its logarithm.

    Products add log values, and marginalising uses log-sum-exp with the max
    shift, so nothing overflows (P9 Lemma 1). ``-inf`` encodes a structural zero.

    Invariants:
        L1  entries are finite or -inf; never NaN, never +inf (checked on every
            result, so overflow raises)
        L3  every operation matches its DiscreteFactor counterpart under exp
        L4  log-sum-exp never overflows, and an all -inf slice gives -inf, not NaN
    """

    __slots__ = ()
    _kind = "Log factor"

    def __init__(self, variables: Sequence[DiscreteVariable], log_values: ArrayLike) -> None:
        variables = _validate_variables(variables)
        names = tuple(v.name for v in variables)
        table = as_float_array(log_values, f"Log factor over {names}")
        self._check_shape(variables, table, "Log factor")
        if np.isnan(table).any():
            raise ValidationError(f"Log factor over {names} has a NaN entry.")
        if np.isposinf(table).any():
            raise ValidationError(f"Log factor over {names} has a +inf entry.")
        self._set(variables, table)

    @classmethod
    def unit(cls) -> LogFactor:
        """The scalar log factor 0 (that is, phi = 1), the identity for the product."""
        return cls._trusted((), np.array(0.0))

    @classmethod
    def from_factor(cls, phi: DiscreteFactor) -> LogFactor:
        """log phi, with log 0 = -inf (a structural zero)."""
        with np.errstate(divide="ignore"):
            return cls._trusted(phi.variables, np.log(phi.values))

    @classmethod
    def from_cpd(cls, cpd: TabularCPD) -> LogFactor:
        return cls.from_factor(DiscreteFactor.from_cpd(cpd))

    def to_factor(self) -> DiscreteFactor:
        """exp of this factor.

        Entries below about -745 underflow to 0. Entries above about 709
        overflow, which ``DiscreteFactor`` rejects with ``ValidationError``.
        """
        with np.errstate(over="ignore"):
            values = np.exp(self._values)
        return DiscreteFactor._trusted(self._variables, values)

    # -- accessors --------------------------------------------------------------

    @property
    def log_values(self) -> np.ndarray:
        """A read-only view of the log table."""
        return self._values.view()

    def log_value(self, assignment: Mapping[str, str]) -> float:
        """log phi(x) for an assignment that covers exactly the scope."""
        return self._entry(assignment)

    def log_total(self) -> float:
        """log Z = log Σ_x phi(x); -inf if every entry is -inf."""
        return float(_log_sum_exp(self._values, tuple(range(self._values.ndim))))

    def normalise(self) -> LogFactor:
        """Return log(phi / Z). Raises ``NormalisationError`` if Z = 0."""
        log_z = self.log_total()
        if log_z == -np.inf:
            raise NormalisationError(f"Cannot normalise log factor over {self.names}: total is 0.")
        return LogFactor._trusted(self._variables, self._values - log_z)

    def allclose(self, other: LogFactor, atol: float = LOG_FACTOR_ATOL) -> bool:
        """True if both have the same scope and definitions, and agree everywhere.

        ``-inf`` entries must coincide exactly; the rest must agree within ``atol``.

        ``atol`` is an absolute tolerance on log values, which is a relative
        tolerance on the values themselves (P9 §6).
        """
        return super().allclose(other, atol)

    # -- arithmetic -------------------------------------------------------------

    def _check_result(self, names: tuple[str, ...], table: np.ndarray) -> None:
        if np.isnan(table).any() or np.isposinf(table).any():
            raise ValidationError(
                f"Log factor over {names} has a NaN or +inf entry (overflow of a log value)."
            )

    def _combine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # -inf + finite = -inf; +inf is excluded by L1, so no NaN can arise.
        total: np.ndarray = a + b
        return total

    def _sum_out(self, values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        return _log_sum_exp(values, axes)

    def _close(self, a: np.ndarray, b: np.ndarray, atol: float) -> bool:
        a_zero, b_zero = np.isneginf(a), np.isneginf(b)
        if not np.array_equal(a_zero, b_zero):
            return False
        finite = ~a_zero
        return bool(np.allclose(a[finite], b[finite], atol=atol, rtol=0.0))


def _log_sum_exp(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """log Σ exp over ``axes``, computed as m + log Σ exp(a - m) (P9 Lemma 1)."""
    if not axes:
        return values
    m = np.max(values, axis=axes, keepdims=True)
    # An all -inf slice would give (-inf) - (-inf) = NaN. Shifting it by 0 instead
    # leaves every term exp(-inf) = 0, so the slice correctly sums to log 0 = -inf.
    shift = np.where(np.isneginf(m), 0.0, m)
    with np.errstate(divide="ignore"):
        result = shift + np.log(np.sum(np.exp(values - shift), axis=axes, keepdims=True))
    return np.squeeze(result, axis=axes)
