"""Datasets of discrete observations, with explicit missing values.

See ``docs/mathematics/likelihood.md`` (Part 1).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from probgraph.exceptions import UnknownNodeError, UnknownStateError, ValidationError
from probgraph.factors import DiscreteFactor
from probgraph.variables import DiscreteVariable

if TYPE_CHECKING:
    from probgraph.models import BayesianNetwork

#: The internal code for a missing value.
MISSING = -1

#: Separates with_missing's random stream from a sampler given the same seed.
_MCAR_STREAM = 0x4D434152  # "MCAR"


class Dataset:
    """An immutable table of observations: one row per case, one column per variable.

    Every row must name every variable. A value is either a state of that
    variable or ``None`` for missing; nothing else means missing.

    Invariants:
        D1  every observed value is a valid state; unknown and omitted variables are rejected
        D2  ``None`` is the only representation of missing
        D3  the dataset cannot be changed after construction
        D4  ``counts(S)`` sums to the number of rows in which all of S is observed
    """

    __slots__ = ("_codes", "_column", "_variables")

    def __init__(
        self,
        variables: Sequence[DiscreteVariable],
        rows: Iterable[Mapping[str, str | None]],
    ) -> None:
        if isinstance(variables, DiscreteVariable) or not isinstance(variables, Iterable):
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
            raise ValidationError(f"Dataset has duplicate variable names: {duplicates}.")

        codes = [self._encode(variables, row, m) for m, row in enumerate(rows)]
        table = np.array(codes, dtype=np.int64).reshape(len(codes), len(variables))
        self._set(variables, table)

    @classmethod
    def from_samples(cls, model: BayesianNetwork, samples: Iterable[Mapping[str, str]]) -> Dataset:
        """A complete dataset over the model's variables, e.g. from ``AncestralSampler``."""
        return cls(model.variables, samples)

    @classmethod
    def _from_codes(cls, variables: tuple[DiscreteVariable, ...], codes: np.ndarray) -> Dataset:
        data = cls.__new__(cls)
        data._set(variables, codes)
        return data

    def _set(self, variables: tuple[DiscreteVariable, ...], codes: np.ndarray) -> None:
        codes.setflags(write=False)
        self._variables = variables
        self._codes = codes
        self._column = {v.name: i for i, v in enumerate(variables)}

    # -- accessors ----------------------------------------------------------------

    @property
    def variables(self) -> tuple[DiscreteVariable, ...]:
        return self._variables

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self._variables)

    @property
    def n_rows(self) -> int:
        return int(self._codes.shape[0])

    def __len__(self) -> int:
        return self.n_rows

    @property
    def codes(self) -> np.ndarray:
        """A read-only (rows × variables) array of state indices; -1 means missing."""
        return self._codes.view()

    @property
    def missing_count(self) -> int:
        return int((self._codes == MISSING).sum())

    @property
    def is_complete(self) -> bool:
        return self.missing_count == 0

    def rows(self) -> Iterator[dict[str, str | None]]:
        """The rows as dictionaries, with ``None`` for missing values."""
        for codes in self._codes.tolist():
            yield {
                v.name: (None if c == MISSING else v.states[c])
                for v, c in zip(self._variables, codes, strict=True)
            }

    # -- sufficient statistics ----------------------------------------------------

    def counts(self, names: Sequence[str]) -> DiscreteFactor:
        """N(s) for every configuration s of ``names``, over rows where all of them are observed.

        The result is a factor with axes in the order given. For no names it is
        the scalar number of rows (likelihood.md, Lemma 1).
        """
        if isinstance(names, str):
            raise ValidationError(
                f"counts() expects a sequence of names, not the string {names!r}."
            )
        wanted = list(names)
        duplicates = sorted({n for n in wanted if wanted.count(n) > 1})
        if duplicates:
            raise ValidationError(f"counts() got duplicate names {duplicates}.")
        unknown = [n for n in wanted if n not in self._column]
        if unknown:
            raise UnknownNodeError(f"counts() names unknown variables {unknown}.")
        columns = [self._column[n] for n in wanted]
        variables = [self._variables[c] for c in columns]
        observed = self._codes[:, columns]
        complete_rows = observed[(observed != MISSING).all(axis=1)] if columns else observed
        table = np.zeros(tuple(v.cardinality for v in variables))
        if columns:
            np.add.at(table, tuple(complete_rows[:, j] for j in range(len(columns))), 1.0)
        else:
            table = np.array(float(self.n_rows))
        return DiscreteFactor(variables, table)

    # -- missing completely at random -------------------------------------------------

    def with_missing(
        self, fraction: float, seed: int, variables: Sequence[str] | None = None
    ) -> Dataset:
        """Hide each observed cell independently with probability ``fraction`` (MCAR).

        Only the columns named in ``variables`` are affected (default: all). The mask
        is independent of the data even when ``seed`` is also the seed of the
        sampler that produced it.
        Missing cells stay missing, and observed values are never changed.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValidationError(f"fraction must be in [0, 1], got {fraction}.")
        if variables is None:
            columns = list(range(len(self._variables)))
        else:
            unknown = [n for n in variables if n not in self._column]
            if unknown:
                raise UnknownNodeError(f"with_missing() names unknown variables {unknown}.")
            columns = [self._column[n] for n in variables]
        # A stream of its own: with default_rng(seed), the mask would reuse the very
        # uniforms that AncestralSampler(seed=seed) turned into the values, so hiding
        # would depend on the values, which is not MCAR.
        rng = np.random.default_rng(np.random.SeedSequence([seed, _MCAR_STREAM]))
        hide = np.zeros(self._codes.shape, dtype=bool)
        hide[:, columns] = rng.random((self.n_rows, len(columns))) < fraction
        codes = np.where(hide, MISSING, self._codes)
        return Dataset._from_codes(self._variables, codes)

    # -- internals ----------------------------------------------------------------------

    @staticmethod
    def _encode(
        variables: tuple[DiscreteVariable, ...], row: Mapping[str, str | None], m: int
    ) -> list[int]:
        if not isinstance(row, Mapping):
            raise ValidationError(f"Row {m} must be a mapping from variable names to states.")
        missing = [v.name for v in variables if v.name not in row]
        if missing:
            raise ValidationError(f"Row {m} is missing {missing}; write None for a missing value.")
        known = {v.name for v in variables}
        unexpected = [n for n in row if n not in known]
        if unexpected:
            raise UnknownNodeError(f"Row {m} has unexpected variables {unexpected}.")
        codes = []
        for v in variables:
            value = row[v.name]
            if value is None:
                codes.append(MISSING)
                continue
            try:
                codes.append(v.index_of(value))
            except UnknownStateError as exc:
                raise UnknownStateError(f"Row {m}: {exc}") from None
        return codes

    def __repr__(self) -> str:
        return (
            f"Dataset(variables={list(self.names)}, rows={self.n_rows}, "
            f"missing={self.missing_count})"
        )
