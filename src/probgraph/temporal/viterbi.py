"""Decoding the hidden states of an HMM: Viterbi and posterior decoding.

The derivations (P20, Part 2) are in ``docs/mathematics/max_product.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.temporal.forward_backward import ForwardBackward, _evidence_vectors
from probgraph.temporal.hmm import HiddenMarkovModel


def viterbi(
    model: HiddenMarkovModel, observations: Sequence[str | None]
) -> tuple[list[str], float]:
    """The most probable hidden path, and log P(path, y_1:T) (max_product.md §8).

    Max-sum elimination along the chain with back-pointers, in O(T K^2) and in log
    space. A missing observation contributes a factor of 1. Among equally probable
    paths, each step of the traceback takes the first state. Raises
    ``ZeroProbabilityEvidenceError`` for an impossible observation sequence.
    """
    if not isinstance(model, HiddenMarkovModel):
        raise ValidationError(f"Expected a HiddenMarkovModel, got {type(model).__name__}.")
    evidence = _evidence_vectors(model, observations)
    length, k = len(evidence), model.n_states
    with np.errstate(divide="ignore"):
        log_a = np.log(model.transition)
        log_e = [np.zeros(k) if e is None else np.log(e) for e in evidence]
        delta = np.log(model.initial) + log_e[0]
    pointers = np.zeros((length, k), dtype=np.intp)
    for t in range(1, length):
        scores = delta[:, np.newaxis] + log_a  # scores[i, j]: best path ending i -> j
        pointers[t] = np.argmax(scores, axis=0)  # the first i attaining the maximum
        delta = scores[pointers[t], np.arange(k)] + log_e[t]
    log_p = float(np.max(delta))
    if log_p == -math.inf:
        raise ZeroProbabilityEvidenceError(
            "The observations are impossible under the model, so no path has positive probability."
        )
    path = [int(np.argmax(delta))]
    for t in range(length - 1, 0, -1):
        path.append(int(pointers[t, path[-1]]))
    states = model.hidden.states
    return [states[i] for i in reversed(path)], log_p


def posterior_decode(model: HiddenMarkovModel, observations: Sequence[str | None]) -> list[str]:
    """The most probable state at each step on its own: argmax_i P(X_t = i | y_1:T).

    This maximises the expected number of correctly decoded steps, not the
    probability of the whole path, and it can return a path of probability zero
    (max_product.md §9). Ties take the first state.
    """
    smoothed = ForwardBackward(model, observations).smoothed
    states = model.hidden.states
    return [states[int(i)] for i in np.argmax(smoothed, axis=1)]
