"""Exact inference on a hidden Markov model by the forward–backward sweeps.

The derivations (P19) are in ``docs/mathematics/forward_backward.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from probgraph.exceptions import UnknownStateError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.temporal.hmm import HiddenMarkovModel

#: Below this, a step's scaled total is recomputed in log space (forward_backward.md §4).
_LOG_FALLBACK = 1e-280


class ForwardBackward:
    """Every posterior of an HMM given one observation sequence, from one pass.

    ``observations`` holds one entry per step: a state of ``model.observed``, or
    ``None`` for a missing observation. Everything is computed when the object is
    built, in O(T K^2).

    Invariants (forward_backward.md):
        F1  ``filtered[t]`` = P(X_t | y_1:t) and ``log_likelihood`` = log P(y_1:T)
        F2  ``step_log_likelihoods[t]`` = log P(y_t | y_1:t-1); they sum to ``log_likelihood``
        F4  a missing observation contributes a factor of 1
        F6  an impossible sequence has ``log_likelihood`` = -inf, and its beliefs raise
    """

    __slots__ = ("_filtered", "_impossible_at", "_log_likelihood", "_model", "_steps")

    def __init__(self, model: HiddenMarkovModel, observations: Sequence[str | None]) -> None:
        if not isinstance(model, HiddenMarkovModel):
            raise ValidationError(f"Expected a HiddenMarkovModel, got {type(model).__name__}.")
        self._model = model
        evidence = _evidence_vectors(model, observations)
        self._filtered, self._steps, self._impossible_at = _forward(model, evidence)
        self._log_likelihood = (
            -math.inf if self._impossible_at is not None else math.fsum(self._steps.tolist())
        )

    @property
    def model(self) -> HiddenMarkovModel:
        return self._model

    @property
    def length(self) -> int:
        return len(self._steps)

    @property
    def log_likelihood(self) -> float:
        """log P(y_1:T) = Σ_t log c_t (Proposition 2); -inf for an impossible sequence."""
        return self._log_likelihood

    @property
    def filtered(self) -> np.ndarray:
        """(T, K): row t is P(X_t | y_1:t)."""
        self._require_possible()
        return self._filtered

    @property
    def step_log_likelihoods(self) -> np.ndarray:
        """(T,): entry t is log c_t = log P(y_t | y_1:t-1)."""
        self._require_possible()
        return self._steps

    def _require_possible(self) -> None:
        if self._impossible_at is not None:
            raise ZeroProbabilityEvidenceError(
                f"The observations are impossible under the model (P(y_1:t) = 0 from step "
                f"{self._impossible_at + 1}), so the posteriors are undefined."
            )


def _evidence_vectors(
    model: HiddenMarkovModel, observations: Sequence[str | None]
) -> list[np.ndarray | None]:
    """e_t = B[:, y_t] for an observed step, None (a factor of 1) for a missing one."""
    if isinstance(observations, str) or not isinstance(observations, Sequence):
        raise ValidationError(
            f"observations must be a sequence of states or None, got {observations!r}."
        )
    if len(observations) == 0:
        raise ValidationError("observations must contain at least one step.")
    symbols = model.observed.states
    vectors: list[np.ndarray | None] = []
    for t, y in enumerate(observations):
        if y is None:
            vectors.append(None)
        elif y in symbols:
            vectors.append(model.emission[:, symbols.index(y)])
        else:
            raise UnknownStateError(
                f"observation {t + 1} is {y!r}, not a state of {model.observed.name!r} "
                f"{symbols} or None."
            )
    return vectors


def _forward(
    model: HiddenMarkovModel, evidence: list[np.ndarray | None]
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """The normalised forward sweep (Proposition 2), with the safeguards of §4."""
    length, k = len(evidence), model.n_states
    filtered = np.zeros((length, k))
    steps = np.zeros(length)
    predicted = model.initial
    for t, e in enumerate(evidence):
        if t > 0:
            predicted = filtered[t - 1] @ model.transition
        if e is None:
            filtered[t] = predicted / predicted.sum()  # sums to 1 up to rounding
            continue
        peak = float(e.max())
        if peak == 0.0:
            return _finish(filtered, steps, t)
        scaled = predicted * (e / peak)
        total = float(scaled.sum())
        if total >= _LOG_FALLBACK:
            filtered[t] = scaled / total
            steps[t] = math.log(peak) + math.log(total)
            continue
        with np.errstate(divide="ignore"):
            logs = np.log(predicted) + np.log(e / peak)
        top = float(logs.max())
        if top == -math.inf:
            return _finish(filtered, steps, t)
        shifted = np.exp(logs - top)
        log_total = top + math.log(float(shifted.sum()))
        filtered[t] = shifted / shifted.sum()
        steps[t] = math.log(peak) + log_total
    return _finish(filtered, steps, None)


def _finish(
    filtered: np.ndarray, steps: np.ndarray, impossible_at: int | None
) -> tuple[np.ndarray, np.ndarray, int | None]:
    filtered.setflags(write=False)
    steps.setflags(write=False)
    return filtered, steps, impossible_at
