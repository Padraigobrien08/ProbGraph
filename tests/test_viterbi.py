"""M5.5: Viterbi and posterior decoding (V3–V5; F1, F2; max_product.md Part 2)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from test_forward_backward import (
    FIVE_DAYS,
    NONE,
    UMBRELLA,
    WEATHER,
    U,
    random_hmm,
    random_observations,
    umbrella_world,
)

from probgraph import DiscreteVariable, VariableElimination
from probgraph.exceptions import UnknownStateError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.temporal import ForwardBackward, HiddenMarkovModel, posterior_decode, viterbi


def path_probabilities(model: HiddenMarkovModel, ys) -> dict[tuple[int, ...], float]:
    """P(x_1:T, y_1:T) for every hidden path."""
    symbols = model.observed.states
    result = {}
    for path in itertools.product(range(model.n_states), repeat=len(ys)):
        p = model.initial[path[0]]
        for t in range(len(ys)):
            if t > 0:
                p *= model.transition[path[t - 1], path[t]]
            if ys[t] is not None:
                p *= model.emission[path[t], symbols.index(ys[t])]
        result[path] = float(p)
    return result


def as_indices(model: HiddenMarkovModel, states) -> tuple[int, ...]:
    return tuple(model.hidden.states.index(s) for s in states)


def f2_model() -> HiddenMarkovModel:
    hidden = DiscreteVariable("S", ("s0", "s1", "s2"))
    observed = DiscreteVariable("O", ("a", "b"))
    return HiddenMarkovModel(
        hidden,
        observed,
        [0.4, 0.3, 0.3],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.5]] * 3,
    )


# ---------------------------------------------------------------------------
# Fixtures F1 and F2
# ---------------------------------------------------------------------------


def test_f1_viterbi_five_days():
    path, log_p = viterbi(umbrella_world(), FIVE_DAYS)
    assert path == ["rain", "rain", "dry", "rain", "rain"]
    assert log_p == pytest.approx(math.log(Fraction(2893401, 250_000_000)), abs=1e-14)
    posterior = math.exp(log_p - ForwardBackward(umbrella_world(), FIVE_DAYS).log_likelihood)
    assert posterior == pytest.approx(float(Fraction(23147208, 68607401)), abs=1e-14)  # 0.337386


def test_f2_posterior_decoding_returns_an_impossible_path():
    model = f2_model()
    ys = ["a", "b"]
    decoded = posterior_decode(model, ys)
    assert decoded == ["s0", "s2"]
    assert path_probabilities(model, ys)[as_indices(model, decoded)] == 0.0  # A[s0, s2] = 0
    path, log_p = viterbi(model, ys)
    assert path == ["s0", "s1"]
    assert math.exp(log_p - ForwardBackward(model, ys).log_likelihood) == pytest.approx(0.4)


def expected_correct(smoothed: np.ndarray, path: tuple[int, ...]) -> float:
    return float(sum(smoothed[t, i] for t, i in enumerate(path)))


def test_f2_each_decoder_wins_its_own_criterion():
    model = f2_model()
    smoothed = ForwardBackward(model, ["a", "b"]).smoothed
    assert expected_correct(smoothed, (0, 2)) == pytest.approx(1.0)  # posterior decoding
    assert expected_correct(smoothed, (0, 1)) == pytest.approx(0.8)  # Viterbi


# ---------------------------------------------------------------------------
# V3: brute force, and the MPE of the unrolled network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_viterbi_matches_brute_force(seed):
    model = random_hmm(seed)
    ys = random_observations(model, seed, length=1 + seed % 6, missing=0.25)
    probabilities = path_probabilities(model, ys)
    ranked = sorted(probabilities.values(), reverse=True)
    path, log_p = viterbi(model, ys)
    assert log_p == pytest.approx(math.log(ranked[0]), rel=1e-12, abs=1e-12)
    assert probabilities[as_indices(model, path)] == pytest.approx(ranked[0], rel=1e-12)
    if len(ranked) == 1 or ranked[1] < ranked[0] * (1 - 1e-9):
        best = max(probabilities, key=probabilities.get)
        assert as_indices(model, path) == best


@pytest.mark.parametrize("seed", range(20))
def test_viterbi_is_the_mpe_of_the_unrolled_network(seed):
    """With every Y_t observed, the hidden states are all that the MPE maximises over."""
    model = random_hmm(seed + 100, k=3, m=3)
    length = 10
    ys = random_observations(model, seed, length, missing=0.0)
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys)}
    ve = VariableElimination(model.to_bayesian_network(length))
    assignment, mpe_log_p = ve.most_probable_explanation(evidence)
    path, log_p = viterbi(model, ys)
    assert log_p == pytest.approx(mpe_log_p, rel=1e-12)
    assert path == [assignment[f"X_{t + 1}"] for t in range(length)]  # unique for these seeds


@pytest.mark.parametrize("seed", range(10))
def test_with_missing_observations_the_unrolled_mpe_answers_a_different_question(seed):
    """Viterbi sums a missing Y_t out (a factor 1); the unrolled MPE maximises over it
    (a factor max_y B[x, y]). The barren-leaf lesson of max_product.md §5 again."""
    model = random_hmm(seed + 150, k=3, m=3)
    ys = random_observations(model, seed, 6, missing=0.0)
    ys[2] = None
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys) if y is not None}
    assignment, mpe_log_p = VariableElimination(
        model.to_bayesian_network(6)
    ).most_probable_explanation(evidence)
    assert "Y_3" in assignment
    _, log_p = viterbi(model, ys)
    assert mpe_log_p < log_p  # max_y B[x, y] < 1 = Σ_y B[x, y]
    brute = max(
        p * max(model.emission[path[2]]) for path, p in path_probabilities(model, ys).items()
    )
    assert mpe_log_p == pytest.approx(math.log(brute), rel=1e-12)


# ---------------------------------------------------------------------------
# Proposition 2: each decoder is optimal for its own criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_each_decoder_is_optimal_for_its_criterion(seed):
    model = random_hmm(seed + 200, k=3, m=2)
    ys = random_observations(model, seed, length=4, missing=0.2)
    probabilities = path_probabilities(model, ys)
    smoothed = ForwardBackward(model, ys).smoothed
    decoded = as_indices(model, posterior_decode(model, ys))
    best_path = as_indices(model, viterbi(model, ys)[0])
    for path, p in probabilities.items():
        assert expected_correct(smoothed, path) <= expected_correct(smoothed, decoded) + 1e-12
        assert p <= probabilities[best_path] * (1 + 1e-12)
    assert decoded == tuple(int(np.argmax(row)) for row in smoothed)


# ---------------------------------------------------------------------------
# Long sequences, missing observations, ties, impossibility
# ---------------------------------------------------------------------------


def test_five_thousand_umbrellas():
    path, log_p = viterbi(umbrella_world(), [U] * 5000)
    assert path == ["rain"] * 5000
    expected = math.log(0.5) + math.log(0.9) + 4999 * math.log(0.63)
    assert log_p == pytest.approx(expected, rel=1e-12)
    assert log_p < -745


def test_missing_observations_follow_the_chain():
    """With nothing observed, the best path is the most probable path of the chain itself."""
    model = random_hmm(5, k=3, m=2)
    ys = [None] * 4
    probabilities = path_probabilities(model, ys)
    path, log_p = viterbi(model, ys)
    assert log_p == pytest.approx(math.log(max(probabilities.values())), rel=1e-12)
    assert probabilities[as_indices(model, path)] == pytest.approx(max(probabilities.values()))


def test_ties_choose_the_first_states():
    hidden = DiscreteVariable("X", ("p", "q", "r"))
    observed = DiscreteVariable("Y", ("a",))
    model = HiddenMarkovModel(
        hidden, observed, np.full(3, 1 / 3), np.full((3, 3), 1 / 3), np.ones((3, 1))
    )
    assert viterbi(model, ["a"] * 4)[0] == ["p"] * 4
    assert posterior_decode(model, ["a"] * 4) == ["p"] * 4


def test_impossible_sequences_raise():
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, [1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]
    )
    with pytest.raises(ZeroProbabilityEvidenceError):
        viterbi(model, [U, NONE])
    with pytest.raises(ZeroProbabilityEvidenceError):
        posterior_decode(model, [U, NONE])


def test_validation():
    with pytest.raises(UnknownStateError, match="observation 1"):
        viterbi(umbrella_world(), ["snow"])
    with pytest.raises(ValidationError, match="HiddenMarkovModel"):
        viterbi("umbrella", [U])
