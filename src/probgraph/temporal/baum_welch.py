"""Learning HMM parameters: Baum–Welch (EM) and the supervised count estimate.

The derivations (P21) are in ``docs/mathematics/baum_welch.md``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from probgraph.exceptions import UnknownStateError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.temporal.forward_backward import ForwardBackward
from probgraph.temporal.hmm import HiddenMarkovModel
from probgraph.variables import DiscreteVariable

Observations = Sequence[str | None]


@dataclass(frozen=True)
class HMMCounts:
    """Expected (or observed) counts: N_1(i), N(i -> j), N(i, m), and log P(y) under the model."""

    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    log_likelihood: float


@dataclass(frozen=True)
class BaumWelchResult:
    """The outcome of ``BaumWelch.run`` (the same layout as M4's ``EMResult``).

    ``log_likelihood[0]`` is Σ_s log P(y^(s)) under the initial model and entry t is
    the value after iteration t. ``log_objective`` is what never decreases: the
    log-likelihood, plus α Σ log θ over every parameter with a pseudocount α
    (baum_welch.md §3).
    """

    model: HiddenMarkovModel
    log_likelihood: tuple[float, ...]
    log_objective: tuple[float, ...]
    converged: bool
    iterations: int


class BaumWelch:
    """Fit an HMM to unlabelled observation sequences by EM with tied parameters (P21).

    Each sequence holds states of ``observed`` or ``None`` for missing values
    (assumed missing at random). The E-step runs forward–backward on every
    sequence; the M-step is the ratio of expected counts, plus ``pseudocount`` in
    every cell (a posterior mean). Iteration stops when ``log_objective`` changes by
    less than ``tolerance``, or after ``max_iterations``.
    """

    def __init__(
        self,
        hidden: DiscreteVariable,
        observed: DiscreteVariable,
        sequences: Sequence[Observations],
        pseudocount: float = 0.0,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
    ) -> None:
        self._hidden = hidden
        self._observed = observed
        self._sequences = _check_sequences(observed, sequences)
        self._pseudocount = _check_pseudocount(pseudocount)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ValidationError(f"max_iterations must be an int, got {max_iterations!r}.")
        if max_iterations < 1:
            raise ValidationError(f"max_iterations must be at least 1, got {max_iterations}.")
        if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance)) or tolerance < 0:
            raise ValidationError(f"tolerance must be finite and non-negative, got {tolerance!r}.")
        self._max_iterations = max_iterations
        self._tolerance = float(tolerance)

    def initial_model(self, seed: int | None = None) -> HiddenMarkovModel:
        """π and every row of A and B drawn independently from the uniform Dirichlet."""
        rng = np.random.default_rng(seed)
        k, m = self._hidden.cardinality, self._observed.cardinality
        return HiddenMarkovModel(
            self._hidden,
            self._observed,
            rng.dirichlet(np.ones(k)),
            rng.dirichlet(np.ones(k), size=k),
            rng.dirichlet(np.ones(m), size=k),
        )

    def expected_counts(self, model: HiddenMarkovModel) -> HMMCounts:
        """The E-step: counts pooled over time and sequences from γ and ξ (baum_welch.md §2)."""
        self._check_model(model)
        k, m = model.n_states, model.n_symbols
        initial, transition, emission = np.zeros(k), np.zeros((k, k)), np.zeros((k, m))
        log_likelihood = 0.0
        for s, ys in enumerate(self._sequences):
            fb = ForwardBackward(model, ys)
            if fb.log_likelihood == -math.inf:
                raise ZeroProbabilityEvidenceError(
                    f"The model gives probability 0 to sequence {s + 1}, so EM cannot condition "
                    "on it; start from a model that gives every sequence positive probability."
                )
            log_likelihood += fb.log_likelihood
            initial += fb.smoothed[0]
            transition += fb.pairwise.sum(axis=0)
            for t, y in enumerate(ys):
                if y is not None:
                    emission[:, self._observed.states.index(y)] += fb.smoothed[t]
        return HMMCounts(initial, transition, emission, log_likelihood)

    def step(self, model: HiddenMarkovModel) -> HiddenMarkovModel:
        """One iteration: the E-step under ``model``, then the M-step."""
        return _estimate(
            self._hidden, self._observed, self.expected_counts(model), self._pseudocount
        )

    def run(
        self, initial: HiddenMarkovModel | None = None, seed: int | None = None
    ) -> BaumWelchResult:
        """Iterate from ``initial``, or from ``initial_model(seed)`` when it is None."""
        model = self.initial_model(seed) if initial is None else initial
        counts = self.expected_counts(model)
        history = [counts.log_likelihood]
        objective = [counts.log_likelihood + self._log_prior(model)]
        converged = False
        iterations = 0
        while iterations < self._max_iterations:
            model = _estimate(self._hidden, self._observed, counts, self._pseudocount)
            counts = self.expected_counts(model)
            iterations += 1
            history.append(counts.log_likelihood)
            objective.append(counts.log_likelihood + self._log_prior(model))
            if abs(objective[-1] - objective[-2]) < self._tolerance:
                converged = True
                break
        return BaumWelchResult(model, tuple(history), tuple(objective), converged, iterations)

    def _log_prior(self, model: HiddenMarkovModel) -> float:
        """α Σ log θ: the posterior-mean M-step is the MAP under pseudocounts α + 1 (§3)."""
        if self._pseudocount == 0.0:
            return 0.0
        tables = (model.initial, model.transition, model.emission)
        return self._pseudocount * math.fsum(float(np.log(t).sum()) for t in tables)

    def _check_model(self, model: HiddenMarkovModel) -> None:
        if not isinstance(model, HiddenMarkovModel):
            raise ValidationError(f"Expected a HiddenMarkovModel, got {type(model).__name__}.")
        if model.hidden != self._hidden or model.observed != self._observed:
            raise ValidationError(
                "The model must have the same hidden and observed variables as the data."
            )


def supervised_estimate(
    hidden: DiscreteVariable,
    observed: DiscreteVariable,
    labelled: Sequence[tuple[Sequence[str], Observations]],
    pseudocount: float = 0.0,
) -> HiddenMarkovModel:
    """The MLE (or, with a pseudocount, posterior mean) from sequences with known states (§1).

    Each item is ``(states, observations)`` of equal length; observations may be
    ``None``. Counts are pooled over time and sequences.
    """
    pseudocount = _check_pseudocount(pseudocount)
    if isinstance(labelled, (str, bytes)) or len(labelled) == 0:
        raise ValidationError("labelled must contain at least one (states, observations) pair.")
    observation_lists = _check_sequences(observed, [ys for _, ys in labelled])
    k, m = hidden.cardinality, observed.cardinality
    initial, transition, emission = np.zeros(k), np.zeros((k, k)), np.zeros((k, m))
    for s, ((states, _), ys) in enumerate(zip(labelled, observation_lists, strict=True)):
        if len(states) != len(ys):
            raise ValidationError(
                f"Sequence {s + 1} has {len(states)} states but {len(ys)} observations."
            )
        xs = []
        for t, state in enumerate(states):
            if state not in hidden.states:
                raise UnknownStateError(
                    f"sequence {s + 1}, state {t + 1} is {state!r}, not a state of "
                    f"{hidden.name!r} {hidden.states}."
                )
            xs.append(hidden.states.index(state))
        initial[xs[0]] += 1
        for a, b in itertools.pairwise(xs):
            transition[a, b] += 1
        for x, y in zip(xs, ys, strict=True):
            if y is not None:
                emission[x, observed.states.index(y)] += 1
    return _estimate(hidden, observed, HMMCounts(initial, transition, emission, 0.0), pseudocount)


def _estimate(
    hidden: DiscreteVariable, observed: DiscreteVariable, counts: HMMCounts, pseudocount: float
) -> HiddenMarkovModel:
    """The M-step: (N + α) / (row total + α · width) for π and every row of A and B."""
    tables = []
    empty: list[str] = []
    for name, table in (
        ("initial", counts.initial),
        ("transitions", counts.transition),
        ("emissions", counts.emission),
    ):
        smoothed = table + pseudocount
        totals = smoothed.sum(axis=-1, keepdims=True)
        rows = np.atleast_1d(totals[..., 0] == 0)
        if rows.any():
            if name == "initial":
                empty.append("the initial state")
            else:
                empty.extend(
                    f"{name} from {hidden.name}={hidden.states[i]}" for i in np.flatnonzero(rows)
                )
            continue
        tables.append(smoothed / totals)
    if empty:
        raise ValidationError(
            f"No (expected) counts for {empty}, so those distributions are undetermined "
            "(baum_welch.md §3); use a pseudocount."
        )
    return HiddenMarkovModel(hidden, observed, *tables)


def _check_sequences(
    observed: DiscreteVariable, sequences: Sequence[Observations]
) -> list[list[str | None]]:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise ValidationError(f"sequences must be a sequence of sequences, got {sequences!r}.")
    if len(sequences) == 0:
        raise ValidationError("sequences must contain at least one sequence.")
    checked = []
    for s, ys in enumerate(sequences):
        if isinstance(ys, (str, bytes)) or not isinstance(ys, Sequence):
            raise ValidationError(
                f"sequence {s + 1} must be a sequence of states or None, got {ys!r}."
            )
        if len(ys) == 0:
            raise ValidationError(f"sequence {s + 1} is empty.")
        for t, y in enumerate(ys):
            if y is not None and y not in observed.states:
                raise UnknownStateError(
                    f"sequence {s + 1}, observation {t + 1} is {y!r}, not a state of "
                    f"{observed.name!r} {observed.states} or None."
                )
        checked.append(list(ys))
    return checked


def _check_pseudocount(pseudocount: float) -> float:
    if (
        isinstance(pseudocount, bool)
        or not isinstance(pseudocount, (int, float))
        or not math.isfinite(pseudocount)
        or pseudocount < 0
    ):
        raise ValidationError(f"pseudocount must be finite and >= 0, got {pseudocount!r}.")
    return float(pseudocount)
