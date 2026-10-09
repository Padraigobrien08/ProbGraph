"""M5.3: smoothing, pairwise posteriors and prediction (F1, F3, F5; forward_backward.md §6–11)."""

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
    extreme_hmm,
    random_hmm,
    random_observations,
    umbrella_world,
)

from probgraph import VariableElimination
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import JunctionTree
from probgraph.temporal import ForwardBackward, HiddenMarkovModel


def brute_force_posteriors(model: HiddenMarkovModel, ys, exact: bool = False):
    """Smoothed (T, K) and pairwise (T-1, K, K) posteriors by enumerating every path."""
    convert = (lambda v: Fraction(v).limit_denominator(10**6)) if exact else float
    pi = [convert(v) for v in model.initial]
    a = [[convert(v) for v in row] for row in model.transition]
    b = [[convert(v) for v in row] for row in model.emission]
    symbols = model.observed.states
    k, length = model.n_states, len(ys)
    smoothed = [[0] * k for _ in range(length)]
    pairwise = [[[0] * k for _ in range(k)] for _ in range(length - 1)]
    total = 0
    for path in itertools.product(range(k), repeat=length):
        p = pi[path[0]]
        for t in range(length):
            if t > 0:
                p *= a[path[t - 1]][path[t]]
            if ys[t] is not None:
                p *= b[path[t]][symbols.index(ys[t])]
        total += p
        for t in range(length):
            smoothed[t][path[t]] += p
            if t < length - 1:
                pairwise[t][path[t]][path[t + 1]] += p
    smoothed = [[p / total for p in row] for row in smoothed]
    pairwise = [[[p / total for p in row] for row in table] for table in pairwise]
    return smoothed, pairwise


# ---------------------------------------------------------------------------
# Fixture F1, exactly
# ---------------------------------------------------------------------------


def test_f1_smoothing_two_days():
    fb = ForwardBackward(umbrella_world(), [U, U])
    assert fb.smoothed[0] == pytest.approx([621 / 703, 82 / 703], abs=1e-15)
    assert fb.smoothed[1] == pytest.approx(fb.filtered[1], abs=1e-15)


def test_f1_smoothing_five_days():
    fb = ForwardBackward(umbrella_world(), FIVE_DAYS)
    smoothed, pairwise = brute_force_posteriors(umbrella_world(), FIVE_DAYS, exact=True)
    for t in range(5):
        assert fb.smoothed[t] == pytest.approx([float(p) for p in smoothed[t]], abs=1e-15)
    expected = [0.8673388895754848, 0.8204190536236754, 0.3074835760066177]
    assert fb.smoothed[:3, 0] == pytest.approx(expected, abs=1e-15)
    assert fb.smoothed[3:, 0] == pytest.approx(expected[1::-1], abs=1e-15)  # symmetric sequence
    for t in range(4):
        assert fb.pairwise[t] == pytest.approx(np.array(pairwise[t], dtype=float), abs=1e-15)


# ---------------------------------------------------------------------------
# Brute force and the junction tree on the unrolled network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_smoothing_matches_brute_force(seed):
    model = random_hmm(seed)
    ys = random_observations(model, seed, length=1 + seed % 6, missing=0.25)
    fb = ForwardBackward(model, ys)
    smoothed, pairwise = brute_force_posteriors(model, ys)
    assert fb.smoothed == pytest.approx(np.array(smoothed), abs=1e-12)
    assert fb.pairwise.shape == (len(ys) - 1, model.n_states, model.n_states)
    if pairwise:
        assert fb.pairwise == pytest.approx(np.array(pairwise), abs=1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_smoothing_matches_the_junction_tree(seed):
    model = random_hmm(seed + 100, k=3, m=3)
    length = 12
    ys = random_observations(model, seed, length, missing=0.3)
    fb = ForwardBackward(model, ys)
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys) if y is not None}
    tree = JunctionTree(model.to_bayesian_network(length), evidence)
    for t in range(length):
        assert fb.smoothed[t] == pytest.approx(tree.marginal(f"X_{t + 1}").values, abs=1e-12)
        if t < length - 1:
            pair = tree.query([f"X_{t + 1}", f"X_{t + 2}"]).values
            assert fb.pairwise[t] == pytest.approx(pair, abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_pairwise_marginals_are_consistent(seed):
    model = random_hmm(seed + 200, k=3, m=2)
    ys = random_observations(model, seed, length=30, missing=0.2)
    fb = ForwardBackward(model, ys)
    assert fb.pairwise.sum(axis=2) == pytest.approx(fb.smoothed[:-1], abs=1e-12)
    assert fb.pairwise.sum(axis=1) == pytest.approx(fb.smoothed[1:], abs=1e-12)
    assert fb.smoothed[-1] == pytest.approx(fb.filtered[-1], abs=1e-15)
    assert fb.smoothed.sum(axis=1) == pytest.approx(np.ones(30), abs=1e-12)


def test_smoothing_long_sequences():
    ys = [U, NONE] * 2500
    fb = ForwardBackward(umbrella_world(), ys)
    assert np.isfinite(fb.smoothed).all() and np.isfinite(fb.pairwise).all()
    # Away from the ends the sequence is periodic, so the smoothed beliefs repeat.
    assert fb.smoothed[2000] == pytest.approx(fb.smoothed[2002], abs=1e-12)
    assert fb.smoothed[2001] == pytest.approx(fb.smoothed[2003], abs=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_extreme_parameters_smoothing_matches_log_space_elimination(seed):
    model = extreme_hmm(seed)
    rng = np.random.default_rng(seed + 1000)
    ys = [str(model.observed.states[int(i)]) for i in rng.integers(0, 3, size=6)]
    fb = ForwardBackward(model, ys)
    if fb.log_likelihood == -math.inf:
        return
    network = model.to_bayesian_network(len(ys))
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys)}
    ve = VariableElimination(network)
    for t in range(len(ys)):
        belief = ve.query([f"X_{t + 1}"], evidence).values
        assert fb.smoothed[t] == pytest.approx(belief, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# F5: prediction
# ---------------------------------------------------------------------------


def test_f5_umbrella_prediction_forgets_at_rate_four_tenths():
    fb = ForwardBackward(umbrella_world(), [U, U])
    predicted = fb.predict(6)
    assert predicted.shape == (6, 2)
    gap = Fraction(621, 703) - Fraction(1, 2)
    for k in range(1, 7):
        assert predicted[k - 1, 0] == pytest.approx(
            float(Fraction(1, 2) + Fraction(2, 5) ** k * gap), abs=1e-15
        )
    assert predicted[:3, 0] == pytest.approx(
        [0.6533428165007112, 0.5613371266002845, 0.5245348506401138], abs=1e-15
    )


@pytest.mark.parametrize("seed", range(10))
def test_prediction_matches_the_unrolled_network(seed):
    model = random_hmm(seed + 300, k=3, m=2)
    ys = random_observations(model, seed, length=5, missing=0.2)
    steps = 4
    network = model.to_bayesian_network(len(ys) + steps)
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys) if y is not None}
    tree = JunctionTree(network, evidence)
    predicted = ForwardBackward(model, ys).predict(steps)
    for k in range(steps):
        assert predicted[k] == pytest.approx(
            tree.marginal(f"X_{len(ys) + k + 1}").values, abs=1e-12
        )


@pytest.mark.parametrize("seed", range(10))
def test_prediction_converges_to_the_stationary_distribution(seed):
    model = random_hmm(seed + 400, k=4, m=3)
    ys = random_observations(model, seed, length=10, missing=0.0)
    far = ForwardBackward(model, ys).predict(400)[-1]
    assert far == pytest.approx(model.stationary_distribution(), abs=1e-10)


def test_predict_validation_and_impossible_sequences():
    fb = ForwardBackward(umbrella_world(), [U])
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValidationError, match="steps"):
            fb.predict(bad)
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, [1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]
    )
    impossible = ForwardBackward(model, [U, NONE])
    for read in (
        lambda: impossible.smoothed,
        lambda: impossible.pairwise,
        lambda: impossible.predict(1),
    ):
        with pytest.raises(ZeroProbabilityEvidenceError):
            read()


def test_single_step_sequences_have_no_pairwise_posteriors():
    fb = ForwardBackward(umbrella_world(), [U])
    assert fb.pairwise.shape == (0, 2, 2)
    assert fb.smoothed[0] == pytest.approx([9 / 11, 2 / 11], abs=1e-15)


def test_results_are_read_only():
    fb = ForwardBackward(umbrella_world(), FIVE_DAYS)
    for table in (fb.smoothed, fb.pairwise):
        with pytest.raises(ValueError):
            table[0] = 0.0


@pytest.mark.parametrize("seed", [112, 415, 577, 1146, 1527, 1916, 2054, 2082])
def test_posteriors_need_log_space(seed):
    """Found by search over 3,000 extreme models: linear arithmetic (a backward message scaled to
    max 1, or filtered beliefs stored as floats) flushes decisive small entries to 0 and gets these
    posteriors wrong by up to 1.0 (forward_backward.md §4, §8)."""
    model = extreme_hmm(seed)
    rng = np.random.default_rng(seed + 1000)
    ys = [str(model.observed.states[int(i)]) for i in rng.integers(0, 3, size=6)]
    fb = ForwardBackward(model, ys)
    network = model.to_bayesian_network(len(ys))
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys)}
    ve = VariableElimination(network)
    for t in range(len(ys)):
        belief = ve.query([f"X_{t + 1}"], evidence).values
        assert fb.smoothed[t] == pytest.approx(belief, rel=0, abs=1e-12)
        if t < len(ys) - 1:
            pair = ve.query([f"X_{t + 1}", f"X_{t + 2}"], evidence).values
            assert fb.pairwise[t] == pytest.approx(pair, rel=0, abs=1e-12)
