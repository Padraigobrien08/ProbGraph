"""Loopy belief propagation on the factor graph, in log space.

Exact on trees; approximate, slow, or non-convergent on graphs with cycles.
See ``docs/mathematics/loopy_bp.md`` (P13).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from probgraph._assignments import check_partial_assignment
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.factors import DiscreteFactor, LogFactor
from probgraph.models import BayesianNetwork, MarkovNetwork


@dataclass(frozen=True, eq=False)
class LoopyResult:
    """The outcome of a loopy BP run.

    Attributes:
        marginals: the beliefs b_x for every unobserved variable. Exact on trees;
            on cycles, approximate (and meaningless if ``converged`` is False).
        converged: True only if the final message change is below the tolerance (B4).
        iterations: the number of synchronous iterations performed.
        residual: the largest change of any normalised factor-to-variable message,
            in probability space, at the final iteration.
    """

    marginals: dict[str, DiscreteFactor]
    converged: bool
    iterations: int
    residual: float


class LoopyBeliefPropagation:
    """Synchronous sum-product on the factor graph of a Markov (or Bayesian) network.

    Single-variable factors are absorbed into their variable as a local potential.
    With ``damping`` α, each factor-to-variable message is updated to
    (1 - α)·new + α·old in probability space, which never changes the fixed
    points (P13, Proposition 2).
    """

    def __init__(
        self,
        model: BayesianNetwork | MarkovNetwork,
        damping: float = 0.0,
        max_iterations: int = 200,
        tolerance: float = 1e-10,
    ) -> None:
        if isinstance(model, BayesianNetwork):
            model = model.to_markov_network()
        if not isinstance(model, MarkovNetwork):
            raise ValidationError(
                f"Expected a BayesianNetwork or MarkovNetwork, got {type(model).__name__}."
            )
        if not 0.0 <= damping < 1.0:
            raise ValidationError(f"damping must be in [0, 1), got {damping}.")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValidationError(
                f"max_iterations must be a positive integer, got {max_iterations}."
            )
        if not tolerance >= 0.0:
            raise ValidationError(f"tolerance must be non-negative, got {tolerance}.")
        self._variables = {v.name: v for v in model.variables}
        self._factors = [
            phi if isinstance(phi, LogFactor) else LogFactor.from_factor(phi)
            for phi in model.factors
        ]
        self._damping = damping
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def run(
        self, evidence: Mapping[str, str] | None = None, seed: int | None = None
    ) -> LoopyResult:
        """Run until the messages change by less than the tolerance, or the iteration limit.

        ``seed`` starts the messages from random positive values instead of uniform
        ones, which shows that a tree's fixed point does not depend on the start.
        """
        observed = check_partial_assignment(self._variables, evidence or {})
        free = [name for name in self._variables if name not in observed]
        cards = {name: self._variables[name].cardinality for name in free}

        local = {name: np.zeros(cards[name]) for name in free}
        nodes: list[tuple[tuple[str, ...], np.ndarray]] = []  # (scope, log table)
        for phi in self._factors:
            reduced = phi.reduce(observed)
            if reduced.log_total() == -np.inf:
                raise ZeroProbabilityEvidenceError(
                    f"A factor over {phi.names} is zero for evidence {observed}."
                )
            if len(reduced.names) == 1:
                local[reduced.names[0]] = local[reduced.names[0]] + reduced.log_values
            elif len(reduced.names) >= 2:
                nodes.append((reduced.names, np.asarray(reduced.log_values)))

        rng = np.random.default_rng(seed) if seed is not None else None
        incoming: dict[str, list[tuple[int, str]]] = {name: [] for name in free}
        messages: dict[tuple[int, str], np.ndarray] = {}
        for f, (scope, _) in enumerate(nodes):
            for x in scope:
                incoming[x].append((f, x))
                start = rng.uniform(0.1, 1.0, cards[x]) if rng is not None else np.ones(cards[x])
                messages[(f, x)] = _normalise(np.log(start))

        log_keep, log_new = (
            (np.log(self._damping), np.log1p(-self._damping)) if self._damping > 0 else (None, None)
        )
        residual = 0.0
        iterations = 0
        converged = not messages  # nothing to pass: the beliefs are already exact
        while not converged and iterations < self._max_iterations:
            iterations += 1
            # Variable-to-factor messages from the previous iteration's factor messages.
            to_factor = {
                (f, x): _normalise(
                    local[x]
                    + sum(
                        (messages[key] for key in incoming[x] if key[0] != f),
                        start=np.zeros(cards[x]),
                    )
                )
                for (f, x) in messages
            }
            updated = {}
            for f, (scope, table) in enumerate(nodes):
                for axis, x in enumerate(scope):
                    total = table.copy()
                    for other_axis, y in enumerate(scope):
                        if y != x:
                            shape = [1] * len(scope)
                            shape[other_axis] = cards[y]
                            total = total + to_factor[(f, y)].reshape(shape)
                    others = tuple(a for a in range(len(scope)) if a != axis)
                    message = _normalise(_log_sum_exp(total, others))
                    if log_keep is not None:
                        message = _normalise(
                            np.logaddexp(log_new + message, log_keep + messages[(f, x)])
                        )
                    updated[(f, x)] = message
            residual = max(
                float(np.abs(np.exp(updated[key]) - np.exp(messages[key])).max())
                for key in messages
            )
            messages = updated
            converged = residual < self._tolerance

        marginals = {}
        for name in free:
            belief = local[name] + sum(
                (messages[key] for key in incoming[name]), start=np.zeros(cards[name])
            )
            marginals[name] = DiscreteFactor([self._variables[name]], np.exp(_normalise(belief)))
        return LoopyResult(marginals, converged, iterations, residual)


def _normalise(log_values: np.ndarray) -> np.ndarray:
    total = _log_sum_exp(log_values, (0,))
    if total == -np.inf:
        raise ZeroProbabilityEvidenceError(
            "Loopy BP found no consistent configuration: a message or belief is zero everywhere."
        )
    normalised: np.ndarray = log_values - total
    return normalised


def _log_sum_exp(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    m = np.max(values, axis=axes, keepdims=True)
    shift = np.where(np.isneginf(m), 0.0, m)
    with np.errstate(divide="ignore"):
        result = shift + np.log(np.sum(np.exp(values - shift), axis=axes, keepdims=True))
    squeezed: np.ndarray = np.squeeze(result, axis=axes)
    return squeezed
