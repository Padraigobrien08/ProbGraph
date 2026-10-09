"""M5.7: Baum–Welch against brute force, recovery up to relabelling, symmetric fixed points
(W5, W6; baum_welch.md §4)."""

import itertools
import math

import numpy as np
import pytest
from test_baum_welch import brute_force_counts
from test_forward_backward import random_hmm, random_observations

from probgraph import DiscreteVariable
from probgraph.temporal import BaumWelch, ForwardBackward, HiddenMarkovModel

Z = 5.0

REGIME = DiscreteVariable("Regime", ("calm", "storm"))
WAVE = DiscreteVariable("Wave", ("low", "mid", "high"))


def bernstein(p: float, n: int) -> float:
    return Z * math.sqrt(p * (1 - p) / n) + Z**2 / (3 * n)


def sea() -> HiddenMarkovModel:
    return HiddenMarkovModel(
        REGIME,
        WAVE,
        [0.6, 0.4],
        [[0.9, 0.1], [0.2, 0.8]],
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
    )


def sea_data() -> list[list[str | None]]:
    """20 sequences of 100 steps, with 20% of observations missing at random."""
    rng = np.random.default_rng(0)
    sequences = []
    for s in range(20):
        _, ys = sea().sample(100, seed=s)
        sequences.append([None if rng.random() < 0.2 else y for y in ys])
    return sequences


def log_likelihood(model: HiddenMarkovModel, sequences) -> float:
    return math.fsum(ForwardBackward(model, ys).log_likelihood for ys in sequences)


def relabelled(model: HiddenMarkovModel, order) -> HiddenMarkovModel:
    """State i of the result is state order[i] of ``model``."""
    order = list(order)
    return HiddenMarkovModel(
        model.hidden,
        model.observed,
        model.initial[order],
        model.transition[np.ix_(order, order)],
        model.emission[order],
    )


def closest_relabelling(model: HiddenMarkovModel, target: HiddenMarkovModel):
    return min(
        itertools.permutations(range(model.n_states)),
        key=lambda order: np.abs(relabelled(model, order).emission - target.emission).max(),
    )


# ---------------------------------------------------------------------------
# Brute-force Baum–Welch, iteration by iteration
# ---------------------------------------------------------------------------


def brute_force_step(model: HiddenMarkovModel, sequences, alpha: float) -> HiddenMarkovModel:
    initial, transition, emission = brute_force_counts(model, sequences)
    tables = [t + alpha for t in (initial, transition, emission)]
    return HiddenMarkovModel(
        model.hidden, model.observed, *(t / t.sum(axis=-1, keepdims=True) for t in tables)
    )


@pytest.mark.parametrize("alpha", [0.0, 0.5])
@pytest.mark.parametrize("seed", range(12))
def test_agrees_with_brute_force_iteration_by_iteration(seed, alpha):
    model = random_hmm(seed + 600, k=2, m=3)
    sequences = [random_observations(model, seed + 4 * s, 5, missing=0.2) for s in range(3)]
    bw = BaumWelch(model.hidden, model.observed, sequences, pseudocount=alpha)
    ours = reference = bw.initial_model(seed)
    for _ in range(6):
        ours = bw.step(ours)
        reference = brute_force_step(reference, sequences, alpha)
        assert ours.initial == pytest.approx(reference.initial, abs=1e-12)
        assert ours.transition == pytest.approx(reference.transition, abs=1e-12)
        assert ours.emission == pytest.approx(reference.emission, abs=1e-12)


def test_expected_counts_add_over_sequences():
    model = random_hmm(5, k=3, m=2)
    a = random_observations(model, 1, 9, missing=0.2)
    b = random_observations(model, 2, 4, missing=0.2)
    both = BaumWelch(model.hidden, model.observed, [a, b]).expected_counts(model)
    first = BaumWelch(model.hidden, model.observed, [a]).expected_counts(model)
    second = BaumWelch(model.hidden, model.observed, [b]).expected_counts(model)
    for name in ("initial", "transition", "emission"):
        assert getattr(both, name) == pytest.approx(
            getattr(first, name) + getattr(second, name), abs=1e-13
        )
    assert both.log_likelihood == pytest.approx(first.log_likelihood + second.log_likelihood)


# ---------------------------------------------------------------------------
# W5: recovery up to relabelling
# ---------------------------------------------------------------------------


def test_relabelling_leaves_the_likelihood_unchanged():
    sequences = sea_data()
    assert log_likelihood(relabelled(sea(), (1, 0)), sequences) == pytest.approx(
        log_likelihood(sea(), sequences), rel=1e-13
    )


def test_recovers_the_sea_up_to_relabelling():
    sequences = sea_data()
    bw = BaumWelch(REGIME, WAVE, sequences, tolerance=1e-6, max_iterations=300)
    truth = sea()
    truth_log_l = log_likelihood(truth, sequences)
    fits = [bw.run(seed=seed).model for seed in (0, 1)]
    orders = [closest_relabelling(fit, truth) for fit in fits]
    assert set(orders) == {(0, 1), (1, 0)}  # the two starts chose opposite labellings
    for fit, order in zip(fits, orders, strict=True):
        fit_log_l = log_likelihood(fit, sequences)
        # The fit is at least as likely as the truth; the gain is about d/2 = 3.5 (Wilks).
        assert 0 <= fit_log_l - truth_log_l < 20
        aligned = relabelled(fit, order)
        # Loose, asymptotic yardsticks (not proven bounds): ~1,600 observed steps.
        assert np.abs(aligned.emission - truth.emission).max() < 0.1
        assert np.abs(aligned.transition - truth.transition).max() < 0.1
        # π is learned from only the 20 first steps.
        for i in range(2):
            p = float(truth.initial[i])
            assert abs(aligned.initial[i] - p) <= bernstein(p, 20)
    a, b = (relabelled(fit, order) for fit, order in zip(fits, orders, strict=True))
    assert np.abs(a.emission - b.emission).max() < 1e-3  # one optimum, two names


# ---------------------------------------------------------------------------
# W6: the symmetric fixed point (Theorem 1)
# ---------------------------------------------------------------------------


def symmetric_model(seed: int, stationary: bool) -> HiddenMarkovModel:
    rng = np.random.default_rng(seed)
    transition = rng.dirichlet(np.ones(2), size=2)
    chain = HiddenMarkovModel(REGIME, WAVE, [0.5, 0.5], transition, [[1 / 3] * 3] * 2)
    initial = chain.stationary_distribution() if stationary else np.array([0.95, 0.05])
    row = rng.dirichlet(np.ones(3))
    return HiddenMarkovModel(REGIME, WAVE, initial, transition, [row, row])


def observed_frequencies(sequences) -> np.ndarray:
    observed = [y for ys in sequences for y in ys if y is not None]
    return np.array([observed.count(s) / len(observed) for s in WAVE.states])


@pytest.mark.parametrize("seed", range(6))
def test_theorem_1_one_step_to_the_symmetric_fixed_point(seed):
    sequences = sea_data()
    start = symmetric_model(seed, stationary=True)
    bw = BaumWelch(REGIME, WAVE, sequences)
    model = bw.step(start)
    assert model.initial == pytest.approx(start.initial, abs=1e-12)
    assert model.transition == pytest.approx(start.transition, abs=1e-12)
    for i in range(2):
        assert model.emission[i] == pytest.approx(observed_frequencies(sequences), abs=1e-12)
    again = bw.step(model)
    assert again.emission == pytest.approx(model.emission, abs=1e-12)  # a fixed point
    # There the observations are i.i.d. with the observed frequencies.
    frequencies = observed_frequencies(sequences)
    iid = sum(
        math.log(frequencies[WAVE.states.index(y)]) for ys in sequences for y in ys if y is not None
    )
    assert log_likelihood(model, sequences) == pytest.approx(iid, rel=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_without_stationarity_the_emission_rows_separate(seed):
    model = BaumWelch(REGIME, WAVE, sea_data()).step(symmetric_model(seed, stationary=False))
    assert np.abs(model.emission[0] - model.emission[1]).max() > 1e-3


def test_near_the_symmetric_point_a_converged_run_can_be_far_from_optimal():
    """The symmetric point is a saddle, but a weakly repelling one: the first step pulls a
    nudged start back almost onto equal emission rows, after which l grows by ~1e-5 per
    iteration. A tolerance-based stop then reports convergence about 70 nats below what
    random starts reach. "Converged" is not "optimal" (baum_welch.md §4)."""
    sequences = sea_data()
    bw = BaumWelch(REGIME, WAVE, sequences, tolerance=1e-6, max_iterations=300)
    start = symmetric_model(0, stationary=True)
    stuck = bw.run(initial=start)
    assert stuck.converged and stuck.iterations <= 2
    emission = start.emission.copy()
    emission[0] += [1e-3, -1e-3, 0.0]
    nudged = bw.run(
        initial=HiddenMarkovModel(REGIME, WAVE, start.initial, start.transition, emission)
    )
    assert nudged.converged
    assert abs(nudged.log_likelihood[-1] - stuck.log_likelihood[-1]) < 1.0  # still on the plateau
    random_start = bw.run(seed=0)
    assert random_start.log_likelihood[-1] > nudged.log_likelihood[-1] + 50
