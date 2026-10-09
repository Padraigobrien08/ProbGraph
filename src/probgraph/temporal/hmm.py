"""Homogeneous discrete hidden Markov models.

The mathematics (P18) is in ``docs/mathematics/markov_chains.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from probgraph._arrays import as_float_array
from probgraph.distributions import TabularCPD
from probgraph.distributions.tabular_cpd import NORMALISATION_ATOL
from probgraph.exceptions import ValidationError
from probgraph.models import BayesianNetwork
from probgraph.sampling.ancestral import cumulative_table, inverse_cdf
from probgraph.variables import DiscreteVariable


class HiddenMarkovModel:
    """A hidden chain X_1, X_2, ... with one observation Y_t per step.

    ``initial[i] = P(X_1 = i)``, ``transition[i, j] = P(X_{t+1} = j | X_t = i)`` and
    ``emission[i, m] = P(Y_t = m | X_t = i)``, with indices in the variables' state
    order. The same tables are used at every step (the model is homogeneous).

    Invariants (markov_chains.md):
        H1  ``initial`` and every row of ``transition`` and ``emission`` lie on the
            simplex (within ``NORMALISATION_ATOL``); the tables are read-only copies
        H2  ``to_bayesian_network(T)`` has exactly the HMM's joint distribution
        H3  ``sample`` is ancestral sampling of that network
        H4  ``stationary_distribution`` is the unique μ with μA = μ, or it raises
    """

    __slots__ = ("_emission", "_hidden", "_initial", "_observed", "_transition")

    def __init__(
        self,
        hidden: DiscreteVariable,
        observed: DiscreteVariable,
        initial: ArrayLike,
        transition: ArrayLike,
        emission: ArrayLike,
    ) -> None:
        for v in (hidden, observed):
            if not isinstance(v, DiscreteVariable):
                raise ValidationError(f"Expected a DiscreteVariable, got {v!r}.")
        if hidden.name == observed.name:
            raise ValidationError(
                f"The hidden and observed variables need distinct names, got {hidden.name!r} twice."
            )
        k, m = hidden.cardinality, observed.cardinality
        self._hidden = hidden
        self._observed = observed
        self._initial = _distribution_table(initial, "initial", (k,), None)
        self._transition = _distribution_table(transition, "transition", (k, k), hidden)
        self._emission = _distribution_table(emission, "emission", (k, m), hidden)

    # -- accessors ----------------------------------------------------------------

    @property
    def hidden(self) -> DiscreteVariable:
        return self._hidden

    @property
    def observed(self) -> DiscreteVariable:
        return self._observed

    @property
    def initial(self) -> np.ndarray:
        return self._initial

    @property
    def transition(self) -> np.ndarray:
        return self._transition

    @property
    def emission(self) -> np.ndarray:
        return self._emission

    @property
    def n_states(self) -> int:
        return self._hidden.cardinality

    @property
    def n_symbols(self) -> int:
        return self._observed.cardinality

    @property
    def n_free_parameters(self) -> int:
        """(K - 1) + K(K - 1) + K(M - 1): the tables are tied across time (§2)."""
        k, m = self.n_states, self.n_symbols
        return (k - 1) + k * (k - 1) + k * (m - 1)

    # -- unrolling and sampling --------------------------------------------------------

    def to_bayesian_network(self, length: int) -> BayesianNetwork:
        """The unrolled network over ``{hidden}_1..T`` and ``{observed}_1..T`` (H2)."""
        length = _check_length(length)
        hidden, observed = self._hidden, self._observed
        xs = [DiscreteVariable(f"{hidden.name}_{t}", hidden.states) for t in range(1, length + 1)]
        ys = [
            DiscreteVariable(f"{observed.name}_{t}", observed.states) for t in range(1, length + 1)
        ]
        edges = [(xs[t].name, xs[t + 1].name) for t in range(length - 1)]
        edges += [(x.name, y.name) for x, y in zip(xs, ys, strict=True)]
        network = BayesianNetwork([*xs, *ys], edges)
        network.add_cpd(TabularCPD(xs[0], (), self._initial))
        for t in range(length):
            if t > 0:  # TabularCPD puts the child on axis 0, so the row-stochastic tables transpose
                network.add_cpd(TabularCPD(xs[t], (xs[t - 1],), self._transition.T))
            network.add_cpd(TabularCPD(ys[t], (xs[t],), self._emission.T))
        return network

    def sample(self, length: int, seed: int | None = None) -> tuple[list[str], list[str]]:
        """Draw one sequence ``(states, observations)`` by ancestral sampling (H3, §7).

        Row t of a (length, 2) block of uniforms gives X_t and then Y_t, so a shorter
        sequence with the same seed is a prefix of a longer one.
        """
        length = _check_length(length)
        uniforms = np.random.default_rng(seed).random((length, 2))
        start = cumulative_table(self._initial)
        moves = cumulative_table(self._transition.T)  # column i: the CDF of row i
        emits = cumulative_table(self._emission.T)
        xs = np.empty(length, dtype=np.intp)
        xs[0] = inverse_cdf(start, uniforms[:1, 0])[0]
        for t in range(1, length):
            xs[t] = inverse_cdf(moves[:, xs[t - 1]], uniforms[t : t + 1, 0])[0]
        ys = inverse_cdf(emits[:, xs], uniforms[:, 1])
        states, symbols = self._hidden.states, self._observed.states
        return [states[i] for i in xs], [symbols[i] for i in ys]

    # -- long-run behaviour --------------------------------------------------------------

    def stationary_distribution(self) -> np.ndarray:
        """The unique μ with μA = μ and Σμ = 1 (Theorem 1).

        It exists and is unique exactly when the chain has one closed communicating
        class; otherwise this raises, naming the closed classes. States outside the
        closed class get exactly 0.
        """
        closed = _closed_classes(self._transition)
        if len(closed) != 1:
            names = [[self._hidden.states[i] for i in c] for c in closed]
            raise ValidationError(
                f"The chain has {len(closed)} closed classes {names}, so its stationary "
                "distribution is not unique."
            )
        members = closed[0]
        block = self._transition[np.ix_(members, members)]
        system = block.T - np.eye(len(members))
        system[-1, :] = 1.0  # replace one balance equation by the normalisation
        rhs = np.zeros(len(members))
        rhs[-1] = 1.0
        mu = np.zeros(self.n_states)
        mu[members] = np.linalg.solve(system, rhs)
        mu.setflags(write=False)
        return mu

    def __repr__(self) -> str:
        return (
            f"HiddenMarkovModel(hidden={self._hidden.name!r} ({self.n_states} states), "
            f"observed={self._observed.name!r} ({self.n_symbols} symbols))"
        )


def _distribution_table(
    values: ArrayLike, name: str, shape: tuple[int, ...], rows_of: DiscreteVariable | None
) -> np.ndarray:
    """Validate a vector, or a matrix of row distributions, and return a read-only copy."""
    table = as_float_array(values, name)
    if table.shape != shape:
        raise ValidationError(f"{name} has shape {table.shape}; expected {shape}.")
    if not np.isfinite(table).all():
        raise ValidationError(f"{name} must be finite.")
    if (table < 0).any():
        raise ValidationError(f"{name} has negative entries.")
    totals = table.sum(axis=-1)
    if rows_of is None:
        if abs(float(totals) - 1.0) > NORMALISATION_ATOL:
            raise ValidationError(f"{name} sums to {float(totals):.12g}, not 1.")
    else:
        for i, total in enumerate(totals):
            if abs(float(total) - 1.0) > NORMALISATION_ATOL:
                raise ValidationError(
                    f"row {i} of {name} (from {rows_of.name}={rows_of.states[i]}) sums to "
                    f"{float(total):.12g}, not 1."
                )
    table.setflags(write=False)
    return table


def _closed_classes(transition: np.ndarray) -> list[list[int]]:
    """Closed communicating classes of the graph {(i, j): A_ij > 0}, by depth-first search."""
    k = len(transition)
    reachable = []
    for start in range(k):
        seen = {start}
        stack = [start]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(transition[i] > 0).tolist():
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        reachable.append(seen)
    classes = []
    for i in range(k):
        cls = sorted(j for j in reachable[i] if i in reachable[j])
        if cls[0] == i and reachable[i] == set(cls):  # first member, and nothing leaves it
            classes.append(cls)
    return classes


def _check_length(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValidationError(f"length must be a positive int, got {length!r}.")
    return length
