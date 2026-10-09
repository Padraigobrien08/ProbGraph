"""Conditional probability tables for discrete variables.

The mathematics behind this representation is in
``docs/mathematics/conditional_probability_tables.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from probgraph._arrays import as_float_array
from probgraph.exceptions import ValidationError
from probgraph.variables import DiscreteVariable

#: Largest allowed absolute deviation of a column sum from 1 (invariant C2).
NORMALISATION_ATOL = 1e-10


class TabularCPD:
    """The conditional distribution p(X | U_1, ..., U_k), stored as a table.

    ``values`` has shape ``(|X|, |U_1|, ..., |U_k|)``. Axis 0 is the child
    variable, and the remaining axes are the parents in the order given.
    ``values[x, u1, ..., uk]`` is p(X = x | U_1 = u1, ..., U_k = uk), where each
    index is the state's position in its variable's domain.

    Invariants (checked when the object is built; it is immutable afterwards):
        C1  every entry is >= 0
        C2  every fibre ``values[:, u1, ..., uk]`` sums to 1, within ``NORMALISATION_ATOL``
        C3  the shape matches the declared cardinalities exactly
        C4  every entry is finite
    Every lookup checks:
        C5  the parent assignment covers exactly the parent set
        C6  every state belongs to its variable's domain
    """

    __slots__ = ("_parent_axis", "_parents", "_values", "_variable")

    def __init__(
        self,
        variable: DiscreteVariable,
        parents: Sequence[DiscreteVariable],
        values: ArrayLike,
    ) -> None:
        if not isinstance(variable, DiscreteVariable):
            raise ValidationError(f"variable must be a DiscreteVariable, got {variable!r}.")
        parents = _validate_parents(variable, parents)
        table = as_float_array(values, f"CPD for {variable.name!r}")

        expected_shape = (variable.cardinality, *(p.cardinality for p in parents))
        if table.shape != expected_shape:
            axes = ", ".join(f"|{v.name}|" for v in (variable, *parents))
            raise ValidationError(
                f"CPD for {variable.name!r} has shape {table.shape}; "
                f"expected ({axes}) = {expected_shape}."
            )
        if not np.isfinite(table).all():
            raise ValidationError(f"CPD for {variable.name!r} contains non-finite entries.")
        if (table < 0).any():
            where = _describe(variable, parents, [int(i) for i in np.argwhere(table < 0)[0]])
            raise ValidationError(f"CPD for {variable.name!r} has a negative entry at {where}.")

        column_sums = table.sum(axis=0)
        error = np.abs(column_sums - 1.0)
        if (error > NORMALISATION_ATOL).any():
            worst = tuple(int(i) for i in np.unravel_index(np.argmax(error), error.shape))
            config = _describe_parents(parents, worst)
            raise ValidationError(
                f"CPD for {variable.name!r} is not normalised: "
                f"sum over {variable.name} given {config} is {column_sums[worst]!r}, expected 1 "
                f"(tolerance {NORMALISATION_ATOL})."
            )

        table.setflags(write=False)
        self._variable = variable
        self._parents = parents
        self._values = table
        self._parent_axis = {p.name: axis for axis, p in enumerate(parents)}

    # -- accessors ----------------------------------------------------------

    @property
    def variable(self) -> DiscreteVariable:
        return self._variable

    @property
    def parents(self) -> tuple[DiscreteVariable, ...]:
        """The parent variables, in the same order as axes 1..k of ``values``."""
        return self._parents

    @property
    def parent_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._parents)

    @property
    def n_free_parameters(self) -> int:
        """(|X| - 1) * prod_j |U_j|: one point of the simplex per parent configuration (P4)."""
        return (self._variable.cardinality - 1) * int(
            np.prod([p.cardinality for p in self._parents])
        )

    @property
    def values(self) -> np.ndarray:
        """A read-only view of the table. It cannot be made writable again."""
        return self._values.view()

    # -- queries ------------------------------------------------------------

    def distribution(self, given: Mapping[str, str]) -> np.ndarray:
        """Return p(X | U = given) as a read-only vector ordered like ``variable.states``."""
        fibre: tuple[slice | int, ...] = (slice(None), *self._parent_indices(given))
        return self._values[fibre]

    def probability(self, state: str, given: Mapping[str, str]) -> float:
        """Return p(X = state | U = given)."""
        index = (self._variable.index_of(state), *self._parent_indices(given))
        return float(self._values[index])

    # -- internals ----------------------------------------------------------

    def _parent_indices(self, given: Mapping[str, str]) -> tuple[int, ...]:
        if not isinstance(given, Mapping):
            raise ValidationError(f"Parent assignment must be a mapping, got {given!r}.")
        missing = [name for name in self._parent_axis if name not in given]
        unexpected = [name for name in given if name not in self._parent_axis]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise ValidationError(
                f"Parent assignment for p({self._variable.name} | ...) must cover exactly "
                f"{list(self._parent_axis)}: " + "; ".join(details) + "."
            )
        return tuple(p.index_of(given[p.name]) for p in self._parents)

    def __repr__(self) -> str:
        given = ", ".join(self.parent_names)
        head = f"{self._variable.name} | {given}" if given else self._variable.name
        return f"TabularCPD(p({head}), shape={self._values.shape})"


def _validate_parents(
    variable: DiscreteVariable, parents: Sequence[DiscreteVariable]
) -> tuple[DiscreteVariable, ...]:
    if isinstance(parents, (DiscreteVariable, str)) or not isinstance(parents, Sequence):
        raise ValidationError(
            f"parents of {variable.name!r} must be a sequence of DiscreteVariable, got {parents!r}."
        )
    parents = tuple(parents)
    for p in parents:
        if not isinstance(p, DiscreteVariable):
            raise ValidationError(
                f"parents of {variable.name!r} must be DiscreteVariable, got {p!r}."
            )
    names = [p.name for p in parents]
    if len(set(names)) != len(names):
        raise ValidationError(f"CPD for {variable.name!r} has duplicate parents: {names}.")
    if variable.name in names:
        raise ValidationError(f"{variable.name!r} cannot be its own parent.")
    return parents


def _describe_parents(parents: Sequence[DiscreteVariable], config: Sequence[int]) -> str:
    if not parents:
        return "no parents"
    return ", ".join(f"{p.name}={p.states[i]}" for p, i in zip(parents, config, strict=True))


def _describe(
    variable: DiscreteVariable, parents: Sequence[DiscreteVariable], index: Sequence[int]
) -> str:
    head = f"{variable.name}={variable.states[index[0]]}"
    return f"p({head} | {_describe_parents(parents, index[1:])})"
