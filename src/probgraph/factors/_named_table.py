"""The part of factor behaviour that does not depend on the arithmetic.

``DiscreteFactor`` (probability space, P5) and ``LogFactor`` (log space, P9)
share everything here: named axes, alignment by name, reduction, and value
lookup. Each subclass supplies the four things that differ: which entries are
valid, how two entries combine (× or +), how a variable is summed out (sum or
log-sum-exp), and how two tables are compared.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, Self

import numpy as np

from probgraph.exceptions import ValidationError
from probgraph.variables import DiscreteVariable


class _NamedTable:
    __slots__ = ("_axis", "_values", "_variables")

    #: Shown in error messages, e.g. "Factor" or "Log factor".
    _kind: ClassVar[str] = "Factor"

    _variables: tuple[DiscreteVariable, ...]
    _values: np.ndarray
    _axis: dict[str, int]

    # -- construction ---------------------------------------------------------

    @classmethod
    def _trusted(cls, variables: tuple[DiscreteVariable, ...], values: np.ndarray) -> Self:
        """Build a table from the result of an operation on valid tables.

        Variable validation and the defensive copy are skipped. The entries are
        still checked, because an operation can overflow.
        """
        table = cls.__new__(cls)
        table._set(variables, np.asarray(values, dtype=np.float64))
        return table

    def _set(self, variables: tuple[DiscreteVariable, ...], table: np.ndarray) -> None:
        self._check_result(tuple(v.name for v in variables), table)
        table.setflags(write=False)
        self._variables = variables
        self._values = table
        self._axis = {v.name: i for i, v in enumerate(variables)}

    @staticmethod
    def _check_shape(variables: tuple[DiscreteVariable, ...], table: np.ndarray, kind: str) -> None:
        names = tuple(v.name for v in variables)
        expected = tuple(v.cardinality for v in variables)
        if table.shape != expected:
            raise ValidationError(
                f"{kind} over {names} has shape {table.shape}; expected {expected}."
            )

    # -- arithmetic supplied by subclasses -----------------------------------

    def _check_result(self, names: tuple[str, ...], table: np.ndarray) -> None:
        raise NotImplementedError

    def _combine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _sum_out(self, values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def _close(self, a: np.ndarray, b: np.ndarray, atol: float) -> bool:
        raise NotImplementedError

    # -- accessors ------------------------------------------------------------

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

    def _entry(self, assignment: Mapping[str, str]) -> float:
        _require_exact_names(assignment, self._axis, f"{self._kind} assignment")
        index = tuple(v.index_of(assignment[v.name]) for v in self._variables)
        return float(self._values[index])

    # -- algebra --------------------------------------------------------------

    def __mul__(self, other: object) -> Self:
        """Combine two tables of the same kind, aligned by variable name."""
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, _NamedTable)
        for v in other._variables:
            if v.name in self._axis:
                _check_same_definition(self._variables[self._axis[v.name]], v)
        variables = self._variables + tuple(v for v in other._variables if v.name not in self._axis)
        names = [v.name for v in variables]
        with np.errstate(over="ignore"):  # overflow is caught by _check_result
            combined = self._combine(self._extend(names), other._extend(names))
        return self._trusted(variables, combined)

    def marginalise(self, names: Iterable[str]) -> Self:
        """Sum out the variables in ``names``. Each one must be in the scope."""
        drop = set(_as_name_collection(names, "marginalise"))
        unknown = sorted(drop - self._axis.keys())
        if unknown:
            raise ValidationError(f"Cannot marginalise {unknown}: not in scope {self.names}.")
        if not drop:
            return self
        axes = tuple(self._axis[n] for n in drop)
        kept = tuple(v for v in self._variables if v.name not in drop)
        return self._trusted(kept, np.asarray(self._sum_out(self._values, axes)))

    def reduce(self, evidence: Mapping[str, str]) -> Self:
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
        return self._trusted(tuple(kept), np.asarray(self._values[tuple(index)]))

    # -- axis order (F4) ------------------------------------------------------

    def aligned(self, order: Sequence[str]) -> Self:
        """Return the same function with its axes in ``order``.

        ``order`` must be a permutation of the scope.
        """
        order = _as_name_collection(order, "aligned")
        if len(order) != len(self._axis) or set(order) != self._axis.keys():
            raise ValidationError(f"{list(order)} is not a permutation of the scope {self.names}.")
        variables = tuple(self._variables[self._axis[n]] for n in order)
        return self._trusted(variables, self._values.transpose([self._axis[n] for n in order]))

    def allclose(self, other: Self, atol: float) -> bool:
        """True if both tables have the same scope and variable definitions.

        They must also agree at every assignment, in the subclass's sense of
        agreement.
        """
        if type(other) is not type(self) or self.scope != other.scope:
            return False
        if any(other._variables[other._axis[v.name]] != v for v in self._variables):
            return False
        return self._close(self._values, other.aligned(self.names)._values, atol)

    # -- internals ------------------------------------------------------------

    def _extend(self, names: Sequence[str]) -> np.ndarray:
        """Cylindrical extension: transpose to the order of ``names``.

        Size-1 axes are inserted for variables not in this table's scope.
        """
        transposed = self._values.transpose([self._axis[n] for n in names if n in self._axis])
        shape = [self._values.shape[self._axis[n]] if n in self._axis else 1 for n in names]
        return transposed.reshape(shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scope={self.names}, shape={self._values.shape})"


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
