"""Exact inference on a hidden Markov model by the forward–backward sweeps.

The derivations (P19) are in ``docs/mathematics/forward_backward.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from probgraph.exceptions import UnknownStateError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.temporal.hmm import HiddenMarkovModel


class ForwardBackward:
    """Every posterior of an HMM given one observation sequence, from one pass.

    ``observations`` holds one entry per step: a state of ``model.observed``, or
    ``None`` for a missing observation. Everything is computed when the object is
    built, in O(T K^2).

    Invariants (forward_backward.md):
        F1  ``filtered[t]`` = P(X_t | y_1:t), ``smoothed[t]`` = P(X_t | y_1:T),
            ``pairwise[t]`` = P(X_t, X_t+1 | y_1:T), ``log_likelihood`` = log P(y_1:T)
        F2  ``step_log_likelihoods[t]`` = log P(y_t | y_1:t-1); they sum to ``log_likelihood``
        F3  the pairwise posteriors' marginals are the smoothed beliefs
        F4  a missing observation contributes a factor of 1
        F5  ``predict(k)`` = P(X_T+k | y_1:T), which tends to the stationary distribution
        F6  an impossible sequence has ``log_likelihood`` = -inf, and its beliefs raise
    """

    __slots__ = (
        "_filtered",
        "_impossible_at",
        "_log_likelihood",
        "_model",
        "_pairwise",
        "_smoothed",
        "_steps",
    )

    def __init__(self, model: HiddenMarkovModel, observations: Sequence[str | None]) -> None:
        if not isinstance(model, HiddenMarkovModel):
            raise ValidationError(f"Expected a HiddenMarkovModel, got {type(model).__name__}.")
        self._model = model
        evidence = _evidence_vectors(model, observations)
        log_filtered, self._steps, self._impossible_at = _forward(model, evidence)
        filtered: np.ndarray = np.exp(log_filtered)
        filtered.setflags(write=False)
        self._filtered = filtered
        self._log_likelihood = (
            -math.inf if self._impossible_at is not None else math.fsum(self._steps.tolist())
        )
        if self._impossible_at is None:
            self._smoothed, self._pairwise = _backward(model, evidence, log_filtered)

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

    @property
    def smoothed(self) -> np.ndarray:
        """(T, K): row t is P(X_t | y_1:T) (Proposition 4)."""
        self._require_possible()
        return self._smoothed

    @property
    def pairwise(self) -> np.ndarray:
        """(T - 1, K, K): entry [t, i, j] is P(X_t = i, X_t+1 = j | y_1:T)."""
        self._require_possible()
        return self._pairwise

    def predict(self, steps: int) -> np.ndarray:
        """(steps, K): row k - 1 is P(X_T+k | y_1:T) = f_T A^k (forward_backward.md §10)."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValidationError(f"steps must be a positive int, got {steps!r}.")
        self._require_possible()
        predicted = np.empty((steps, self._model.n_states))
        belief = self._filtered[-1]
        for k in range(steps):
            belief = belief @ self._model.transition
            predicted[k] = belief
        predicted.setflags(write=False)
        return predicted

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
    """The normalised forward sweep (Proposition 2), carried as logarithms (§4).

    Returns the filtered beliefs as log-probabilities, log c_t for every step, and
    the first impossible step (or None).
    """
    length, k = len(evidence), model.n_states
    with np.errstate(divide="ignore"):
        log_a = np.log(model.transition)
        log_predicted = np.log(model.initial)
    log_filtered = np.full((length, k), -math.inf)
    steps = np.zeros(length)
    for t, e in enumerate(evidence):
        if t > 0:  # log P(X_t | y_1:t-1) = lse_i(log f_t-1(i) + log A_ij)
            log_predicted = _log_sum_exp_rows((log_filtered[t - 1][:, np.newaxis] + log_a).T)
        if e is None:  # a factor of 1: c_t = 1 exactly
            log_filtered[t] = log_predicted - _log_sum_exp(log_predicted)
            continue
        with np.errstate(divide="ignore"):
            joint = log_predicted + np.log(e)
        log_c = _log_sum_exp(joint)
        if log_c == -math.inf:
            return _finish(log_filtered, steps, t)
        log_filtered[t] = joint - log_c
        steps[t] = log_c
    return _finish(log_filtered, steps, None)


def _backward(
    model: HiddenMarkovModel, evidence: list[np.ndarray | None], log_f: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The backward sweep in log space (§6–8), and the smoothed and pairwise posteriors."""
    length, k = len(evidence), model.n_states
    with np.errstate(divide="ignore"):
        log_a = np.log(model.transition)
        log_e = [np.zeros(k) if e is None else np.log(e) for e in evidence]
    smoothed = np.empty((length, k))
    pairwise = np.empty((max(length - 1, 0), k, k))
    smoothed[-1] = np.exp(log_f[-1])  # β_T = 1
    log_b = np.zeros(k)
    for t in range(length - 2, -1, -1):
        incoming = log_e[t + 1] + log_b  # log of e_t+1 ∘ b_t+1
        pairwise[t] = _softmax(log_f[t][:, np.newaxis] + log_a + incoming[np.newaxis, :])
        log_b = _log_sum_exp_rows(log_a + incoming[np.newaxis, :])
        log_b = log_b - float(np.max(log_b))
        smoothed[t] = _softmax(log_f[t] + log_b)
    smoothed.setflags(write=False)
    pairwise.setflags(write=False)
    return smoothed, pairwise


def _log_sum_exp_rows(logs: np.ndarray) -> np.ndarray:
    """log Σ_j exp(logs[i, j]) for every row, with the max shift (P9); -inf rows stay -inf."""
    top = logs.max(axis=1)
    safe = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        result: np.ndarray = safe + np.log(np.exp(logs - safe[:, np.newaxis]).sum(axis=1))
    return result


def _log_sum_exp(logs: np.ndarray) -> float:
    top = float(logs.max())
    if top == -math.inf:
        return -math.inf
    return top + math.log(float(np.exp(logs - top).sum()))


def _softmax(logs: np.ndarray) -> np.ndarray:
    """exp(logs) / Σ exp(logs), shifted by the maximum so nothing overflows or underflows to 0/0."""
    weights = np.exp(logs - float(np.max(logs)))
    result: np.ndarray = weights / weights.sum()
    return result


def _finish(
    log_filtered: np.ndarray, steps: np.ndarray, impossible_at: int | None
) -> tuple[np.ndarray, np.ndarray, int | None]:
    log_filtered.setflags(write=False)
    steps.setflags(write=False)
    return log_filtered, steps, impossible_at
