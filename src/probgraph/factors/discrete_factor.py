"""Discrete factors (potentials) and their algebra.

The definitions and the laws F1–F10 are in ``docs/mathematics/factor_algebra.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from probgraph._arrays import as_float_array
from probgraph.exceptions import NormalisationError, ValidationError
from probgraph.variables import DiscreteVariable

if TYPE_CHECKING:
    from probgraph.distributions import TabularCPD

#: Default absolute tolerance for ``DiscreteFactor.allclose``.
FACTOR_ATOL = 1e-12


class DiscreteFactor:
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

    __slots__ = ("_axis", "_values", "_variables")

    _variables: tuple[DiscreteVariable, ...]
    _values: np.ndarray
    _axis: dict[str, int]

    def __init__(self, variables: Sequence[DiscreteVariable], values: ArrayLike) -> None:
        variables = _validate_variables(variables)
        names = tuple(v.name for v in variables)
        table = as_float_array(values, f"Factor over {names}")

        expected = tuple(v.cardinality for v in variables)
        if table.shape != expected:
            raise ValidationError(
                f"Factor over {names} has shape {table.shape}; expected {expected}."
            )
        if (table < 0).any():
            raise ValidationError(f"Factor over {names} has a negative entry.")
        self._set(variables, table)

    @classmethod
    def _trusted(
        cls, variables: tuple[DiscreteVariable, ...], values: np.ndarray
    ) -> DiscreteFactor:
        """Build a factor from the result of an operation on valid factors.

        Variable validation, the copy and the nonnegativity check are skipped:
        products and sums of nonnegative numbers stay nonnegative. Finiteness is
        still checked, because a product can overflow.
        """
        factor = cls.__new__(cls)
        factor._set(variables, np.asarray(values, dtype=np.float64))
        return factor

    def _set(self, variables: tuple[DiscreteVariable, ...], table: np.ndarray) -> None:
        if not np.isfinite(table).all():
            names = tuple(v.name for v in variables)
            raise ValidationError(
                f"Factor over {names} has non-finite entries (NaN, inf, or overflow)."
            )
        table.setflags(write=False)
        self._variables = variables
        self._values = table
        self._axis = {v.name: i for i, v in enumerate(variables)}

    @classmethod
    def unit(cls) -> DiscreteFactor:
        """The scalar factor 1, which is the identity for the product."""
        return cls._trusted((), np.array(1.0))

    @classmethod
    def from_cpd(cls, cpd: TabularCPD) -> DiscreteFactor:
        """View p(X | U) as the factor phi(X, U), with axes (X, U_1, ..., U_k)."""
        return cls._trusted((cpd.variable, *cpd.parents), cpd.values)

    # -- accessors ----------------------------------------------------------

    @property
    def variables(self) -> tuple[DiscreteVariable, ...]:
        """The scope's variables, in axis order."""
        return self._variables

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self._variables)

    @property
    def scope(self) -> frozenset[str]:
        return frozenset(self._axis)

    @property
    def values(self) -> np.ndarray:
        """A read-only view of the table. It cannot be made writable again."""
        return self._values.view()

    def value(self, assignment: Mapping[str, str]) -> float:
        """Return phi(x) for an assignment that covers exactly the scope."""
        _require_exact_names(assignment, self._axis, "Factor assignment")
        index = tuple(v.index_of(assignment[v.name]) for v in self._variables)
        return float(self._values[index])

    def total(self) -> float:
        """Z = sum over x of phi(x). For a scalar factor this is its value."""
        return float(self._values.sum())

    # -- algebra ------------------------------------------------------------

    def __mul__(self, other: object) -> DiscreteFactor:
        """Return (phi * psi)(x) = phi(x|S) psi(x|T), with scope S ∪ T."""
        if not isinstance(other, DiscreteFactor):
            return NotImplemented
        for v in other._variables:
            if v.name in self._axis:
                _check_same_definition(self._variables[self._axis[v.name]], v)
        variables = self._variables + tuple(v for v in other._variables if v.name not in self._axis)
        names = [v.name for v in variables]
        with np.errstate(over="ignore"):  # overflow is caught by the finiteness check
            product = self._extend(names) * other._extend(names)
        return DiscreteFactor._trusted(variables, product)

    def marginalise(self, names: Iterable[str]) -> DiscreteFactor:
        """Sum out the variables in ``names``. Each one must be in the scope."""
        drop = set(_as_name_collection(names, "marginalise"))
        unknown = sorted(drop - self._axis.keys())
        if unknown:
            raise ValidationError(f"Cannot marginalise {unknown}: not in scope {self.names}.")
        if not drop:
            return self
        axes = tuple(self._axis[n] for n in drop)
        kept = tuple(v for v in self._variables if v.name not in drop)
        return DiscreteFactor._trusted(kept, np.asarray(self._values.sum(axis=axes)))

    def reduce(self, evidence: Mapping[str, str]) -> DiscreteFactor:
        """Fix the observed variables in scope at their observed states, and drop those axes.

        Evidence on variables outside the scope is ignored, so the same evidence
        can be applied to every factor in a model.
        """
        if not isinstance(evidence, Mapping):
            raise ValidationError(f"Evidence must be a mapping, got {evidence!r}.")
        index: list[int | slice] = []
        kept: list[DiscreteVariable] = []
        for v in self._variables:
            if v.name in evidence:
                index.append(v.index_of(evidence[v.name]))
            else:
                index.append(slice(None))
                kept.append(v)
        if len(kept) == len(self._variables):
            return self
        return DiscreteFactor._trusted(tuple(kept), np.asarray(self._values[tuple(index)]))

    def normalise(self) -> DiscreteFactor:
        """Return phi / Z. Raises ``NormalisationError`` if Z = 0."""
        z = self.total()
        if z == 0.0:
            raise NormalisationError(f"Cannot normalise factor over {self.names}: total is 0.")
        return DiscreteFactor._trusted(self._variables, self._values / z)

    # -- axis order (F4) ----------------------------------------------------

    def aligned(self, order: Sequence[str]) -> DiscreteFactor:
        """Return the same function with its axes in ``order``.

        ``order`` must be a permutation of the scope.
        """
        order = _as_name_collection(order, "aligned")
        if len(order) != len(self._axis) or set(order) != self._axis.keys():
            raise ValidationError(f"{list(order)} is not a permutation of the scope {self.names}.")
        variables = tuple(self._variables[self._axis[n]] for n in order)
        return DiscreteFactor._trusted(
            variables, self._values.transpose([self._axis[n] for n in order])
        )

    def allclose(self, other: DiscreteFactor, atol: float = FACTOR_ATOL) -> bool:
        """True if both factors have the same scope and variable definitions.

        They must also agree at every assignment, within ``atol``.
        """
        if self.scope != other.scope:
            return False
        if any(other._variables[other._axis[v.name]] != v for v in self._variables):
            return False
        aligned = other.aligned(self.names)._values
        return bool(np.allclose(self._values, aligned, atol=atol, rtol=0.0))

    # -- internals ----------------------------------------------------------

    def _extend(self, names: Sequence[str]) -> np.ndarray:
        """Cylindrical extension: transpose to the order of ``names``.

        Size-1 axes are inserted for variables not in this factor's scope.
        """
        transposed = self._values.transpose([self._axis[n] for n in names if n in self._axis])
        shape = [self._values.shape[self._axis[n]] if n in self._axis else 1 for n in names]
        return transposed.reshape(shape)

    def __repr__(self) -> str:
        return f"DiscreteFactor(scope={self.names}, shape={self._values.shape})"


def _validate_variables(variables: Sequence[DiscreteVariable]) -> tuple[DiscreteVariable, ...]:
    if isinstance(variables, DiscreteVariable) or not isinstance(variables, Sequence):
        raise ValidationError(
            f"variables must be a sequence of DiscreteVariable, got {variables!r}."
        )
    variables = tuple(variables)
    for v in variables:
        if not isinstance(v, DiscreteVariable):
            raise ValidationError(f"Expected DiscreteVariable, got {v!r}.")
    names = [v.name for v in variables]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValidationError(f"Factor has duplicate variables: {duplicates}.")
    return variables


def _check_same_definition(mine: DiscreteVariable, theirs: DiscreteVariable) -> None:
    if mine != theirs:
        raise ValidationError(
            f"Variable {mine.name!r} has domain {mine.states} in one factor "
            f"and {theirs.states} in the other."
        )


def _as_name_collection(names: Iterable[str], operation: str) -> list[str]:
    if isinstance(names, str):
        raise ValidationError(
            f"{operation} expects a collection of names, not the string {names!r}."
        )
    return list(names)


def _require_exact_names(
    assignment: Mapping[str, str], expected: Mapping[str, int], what: str
) -> None:
    if not isinstance(assignment, Mapping):
        raise ValidationError(f"{what} must be a mapping, got {assignment!r}.")
    missing = [n for n in expected if n not in assignment]
    unexpected = [n for n in assignment if n not in expected]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValidationError(
            f"{what} must cover exactly {list(expected)}: " + "; ".join(details) + "."
        )
