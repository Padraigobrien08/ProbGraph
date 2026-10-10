"""The output of a Markov chain Monte Carlo run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from probgraph._assignments import check_query
from probgraph.exceptions import UnknownNodeError, UnknownStateError
from probgraph.factors import DiscreteFactor
from probgraph.variables import DiscreteVariable


@dataclass(frozen=True)
class Chain:
    """Recorded states of an MCMC run, one row per kept sample.

    ``states[s, j]`` is the state index of ``variables[j]`` in sample ``s``. Only the
    unobserved variables appear; the evidence is fixed and not recorded.
    """

    variables: tuple[DiscreteVariable, ...]
    states: np.ndarray
    acceptance_rate: float

    def __post_init__(self) -> None:
        self.states.setflags(write=False)

    @property
    def n_samples(self) -> int:
        return int(self.states.shape[0])

    def indicator(self, variable: str, state: str) -> np.ndarray:
        """(n,) array: 1.0 where ``variable`` is in ``state``, else 0.0."""
        j = self._column(variable)
        v = self.variables[j]
        if state not in v.states:
            raise UnknownStateError(f"{state!r} is not a state of {variable!r} {v.states}.")
        result: np.ndarray = (self.states[:, j] == v.states.index(state)).astype(np.float64)
        return result

    def estimate(self, variables: Sequence[str]) -> DiscreteFactor:
        """The empirical joint distribution of ``variables`` over the kept samples."""
        names = check_query({v.name: v for v in self.variables}, variables)
        columns = [self._column(n) for n in names]
        chosen = [self.variables[j] for j in columns]
        shape = tuple(v.cardinality for v in chosen)
        flat = np.ravel_multi_index(tuple(self.states[:, j] for j in columns), shape)
        counts = np.bincount(flat, minlength=int(np.prod(shape, dtype=int))).astype(np.float64)
        return DiscreteFactor(chosen, counts.reshape(shape) / self.n_samples)

    def _column(self, variable: str) -> int:
        for j, v in enumerate(self.variables):
            if v.name == variable:
                return j
        raise UnknownNodeError(
            f"{variable!r} is not in the chain (it is observed, or not in the model)."
        )
