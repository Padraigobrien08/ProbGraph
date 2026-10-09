"""Discrete factors (potentials) and their algebra.

The definitions and the laws F1–F10 are in ``docs/mathematics/factor_algebra.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from probgraph._arrays import as_float_array
from probgraph.exceptions import NormalisationError, ValidationError
from probgraph.factors._named_table import _NamedTable, _validate_variables
from probgraph.variables import DiscreteVariable

if TYPE_CHECKING:
    from probgraph.distributions import TabularCPD

#: Default absolute tolerance for ``DiscreteFactor.allclose``.
FACTOR_ATOL = 1e-12


class DiscreteFactor(_NamedTable):
    """A nonnegative function phi(x_S) over a set S of named discrete variables.

    ``values`` has one axis per variable, in the order of ``variables``. The axis
    order is only a storage detail (F4): values are looked up by name, products
    align their inputs by name, and comparisons align by name before comparing.

    Invariants:
        F1  every entry is finite and >= 0 (this is also checked on every result,
            so overflow raises instead of producing inf)
        F2  the variable names are unique, and the shape matches their cardinalities
        F3  a variable that appears in two factors has the same definition in both

    Factors are immutable. Every operation returns a new factor.
    """

    __slots__ = ()

    def __init__(self, variables: Sequence[DiscreteVariable], values: ArrayLike) -> None:
        variables = _validate_variables(variables)
        names = tuple(v.name for v in variables)
        table = as_float_array(values, f"Factor over {names}")
        self._check_shape(variables, table, "Factor")
        if (table < 0).any():
            raise ValidationError(f"Factor over {names} has a negative entry.")
        self._set(variables, table)

    @classmethod
    def unit(cls) -> DiscreteFactor:
        """The scalar factor 1, which is the identity for the product."""
        return cls._trusted((), np.array(1.0))

    @classmethod
    def from_cpd(cls, cpd: TabularCPD) -> DiscreteFactor:
        """View p(X | U) as the factor phi(X, U), with axes (X, U_1, ..., U_k)."""
        return cls._trusted((cpd.variable, *cpd.parents), cpd.values)

    # -- accessors ------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """A read-only view of the table. It cannot be made writable again."""
        return self._values.view()

    def value(self, assignment: Mapping[str, str]) -> float:
        """Return phi(x) for an assignment that covers exactly the scope."""
        return self._entry(assignment)

    def total(self) -> float:
        """Z = sum over x of phi(x). For a scalar factor this is its value."""
        return float(self._values.sum())

    def normalise(self) -> DiscreteFactor:
        """Return phi / Z. Raises ``NormalisationError`` if Z = 0."""
        z = self.total()
        if z == 0.0:
            raise NormalisationError(f"Cannot normalise factor over {self.names}: total is 0.")
        return DiscreteFactor._trusted(self._variables, self._values / z)

    def allclose(self, other: DiscreteFactor, atol: float = FACTOR_ATOL) -> bool:
        """True if both factors have the same scope and definitions, and agree within ``atol``."""
        return super().allclose(other, atol)

    # -- arithmetic -------------------------------------------------------------

    def _check_result(self, names: tuple[str, ...], table: np.ndarray) -> None:
        if not np.isfinite(table).all():
            raise ValidationError(
                f"Factor over {names} has non-finite entries (NaN, inf, or overflow)."
            )

    def _combine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        product: np.ndarray = a * b
        return product

    def _sum_out(self, values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        return np.asarray(values.sum(axis=axes))

    def _close(self, a: np.ndarray, b: np.ndarray, atol: float) -> bool:
        return bool(np.allclose(a, b, atol=atol, rtol=0.0))
