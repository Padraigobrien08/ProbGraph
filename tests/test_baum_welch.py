"""M5.6: Baum–Welch (W1–W4; F3; baum_welch.md)."""

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

from probgraph import BayesianNetwork, DiscreteVariable
from probgraph.exceptions import UnknownStateError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.learning import Dataset, expected_counts, maximum_likelihood
from probgraph.temporal import (
    BaumWelch,
    BaumWelchResult,
    ForwardBackward,
    HiddenMarkovModel,
    supervised_estimate,
)


def brute_force_counts(model: HiddenMarkovModel, sequences):
    """Expected initial, transition and emission counts by enumerating every path."""
    k, m = model.n_states, model.n_symbols
    symbols = model.observed.states
    initial, transition, emission = np.zeros(k), np.zeros((k, k)), np.zeros((k, m))
    for ys in sequences:
        weights = {}
        for path in itertools.product(range(k), repeat=len(ys)):
            p = model.initial[path[0]]
            for t in range(len(ys)):
                if t > 0:
                    p *= model.transition[path[t - 1], path[t]]
                if ys[t] is not None:
                    p *= model.emission[path[t], symbols.index(ys[t])]
            weights[path] = p
        total = sum(weights.values())
        for path, p in weights.items():
            w = p / total
            initial[path[0]] += w
            for t in range(len(ys)):
                if t > 0:
                    transition[path[t - 1], path[t]] += w
                if ys[t] is not None:
                    emission[path[t], symbols.index(ys[t])] += w
    return initial, transition, emission


# ---------------------------------------------------------------------------
# Fixture F3, exactly
# ---------------------------------------------------------------------------


def test_f3_one_iteration():
    bw = BaumWelch(WEATHER, UMBRELLA, [FIVE_DAYS])
    model = bw.step(umbrella_world())
    exact = {
        "pi": Fraction(59505867, 68607401),
        "rain->rain": Fraction(7928676, 10731953),
        "dry->rain": Fraction(25229493, 40627225),
        "u|rain": Fraction(25731708, 28075669),
        "u|dry": Fraction(5355529, 11294498),
    }
    assert model.initial[0] == pytest.approx(float(exact["pi"]), abs=1e-15)
    assert model.transition[0, 0] == pytest.approx(float(exact["rain->rain"]), abs=1e-15)
    assert model.transition[1, 0] == pytest.approx(float(exact["dry->rain"]), abs=1e-15)
    assert model.emission[0, 0] == pytest.approx(float(exact["u|rain"]), abs=1e-15)
    assert model.emission[1, 0] == pytest.approx(float(exact["u|dry"]), abs=1e-15)


def test_f3_history():
    result = BaumWelch(WEATHER, UMBRELLA, [FIVE_DAYS], max_iterations=1).run(
        initial=umbrella_world()
    )
    assert isinstance(result, BaumWelchResult)
    assert result.iterations == 1 and not result.converged
    assert result.log_likelihood == pytest.approx((-3.3725020443, -2.4583851294), abs=1e-9)
    assert result.log_objective == result.log_likelihood


# ---------------------------------------------------------------------------
# W1: expected counts, against brute force and against M4 on the unrolled network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_expected_counts_match_brute_force(seed):
    model = random_hmm(seed)
    sequences = [
        random_observations(model, seed + 10 * s, length=1 + (seed + s) % 5, missing=0.25)
        for s in range(3)
    ]
    counts = BaumWelch(model.hidden, model.observed, sequences).expected_counts(model)
    initial, transition, emission = brute_force_counts(model, sequences)
    assert counts.initial == pytest.approx(initial, abs=1e-12)
    assert counts.transition == pytest.approx(transition, abs=1e-12)
    assert counts.emission == pytest.approx(emission, abs=1e-12)
    assert counts.log_likelihood == pytest.approx(
        sum(ForwardBackward(model, ys).log_likelihood for ys in sequences), rel=1e-12
    )


@pytest.mark.parametrize("seed", range(15))
def test_expected_counts_are_m4_counts_summed_over_tied_families(seed):
    model = random_hmm(seed + 100, k=3, m=3)
    length = 7
    sequences = [random_observations(model, seed + 5 * s, length, missing=0.2) for s in range(4)]
    network = model.to_bayesian_network(length)
    rows = []
    for ys in sequences:
        row: dict[str, str | None] = {f"X_{t}": None for t in range(1, length + 1)}
        row.update({f"Y_{t + 1}": y for t, y in enumerate(ys)})
        rows.append(row)
    m4 = expected_counts(network, Dataset(network.variables, rows))
    counts = BaumWelch(model.hidden, model.observed, sequences).expected_counts(model)
    # Family tables have the child on axis 0, so transitions are (X_t, X_t-1): transpose.
    transition = sum(m4[f"X_{t}"].T for t in range(2, length + 1))
    emission = sum(m4[f"Y_{t}"].T for t in range(1, length + 1))
    assert counts.initial == pytest.approx(m4["X_1"], abs=1e-11)
    assert counts.transition == pytest.approx(transition, abs=1e-11)
    # M4 treats a missing Y_t as missing data and fills in γ_t(i) B[i, m]; Baum–Welch sums
    # it out and adds nothing (baum_welch.md §2). That is the only difference.
    filled_in = sum(
        ForwardBackward(model, ys).smoothed[t][:, np.newaxis] * model.emission
        for ys in sequences
        for t, y in enumerate(ys)
        if y is None
    )
    assert counts.emission + filled_in == pytest.approx(emission, abs=1e-11)


@pytest.mark.parametrize("seed", range(5))
def test_both_treatments_of_a_missing_observation_share_fixed_points(seed):
    """At a Baum–Welch fixed point, M4's fill-in M-step (N_obs + Σγ B)/(n_obs + Σγ) returns B."""
    truth = random_hmm(seed + 120, k=2, m=3)
    sequences = [random_observations(truth, seed + 9 * s, 20, missing=0.3) for s in range(3)]
    bw = BaumWelch(truth.hidden, truth.observed, sequences, tolerance=1e-13, max_iterations=5000)
    fixed = bw.run(seed=seed).model
    counts = bw.expected_counts(fixed)
    missing_weight = sum(
        ForwardBackward(fixed, ys).smoothed[t]
        for ys in sequences
        for t, y in enumerate(ys)
        if y is None
    )
    filled = counts.emission + missing_weight[:, np.newaxis] * fixed.emission
    assert filled / filled.sum(axis=1, keepdims=True) == pytest.approx(fixed.emission, abs=1e-7)


def test_missing_observations_add_no_emission_counts():
    counts = BaumWelch(WEATHER, UMBRELLA, [[U, None, None, U]]).expected_counts(umbrella_world())
    assert counts.emission.sum() == pytest.approx(2.0)
    assert counts.transition.sum() == pytest.approx(3.0)
    assert counts.initial.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# W3: monotonicity
# ---------------------------------------------------------------------------


def assert_non_decreasing(history, slack=1e-10):
    assert (np.diff(history) >= -slack).all(), np.diff(history).min()


@pytest.mark.parametrize("seed", range(20))
def test_log_likelihood_never_decreases(seed):
    truth = random_hmm(seed + 200, k=3, m=3)
    sequences = [random_observations(truth, seed + 7 * s, 30, missing=0.1) for s in range(5)]
    result = BaumWelch(truth.hidden, truth.observed, sequences, max_iterations=50).run(seed=seed)
    assert result.iterations >= 1
    assert len(result.log_likelihood) == result.iterations + 1
    assert_non_decreasing(result.log_likelihood)
    final = sum(ForwardBackward(result.model, ys).log_likelihood for ys in sequences)
    assert result.log_likelihood[-1] == pytest.approx(final, rel=1e-11)


def log_prior(model: HiddenMarkovModel, alpha: float) -> float:
    return alpha * float(
        np.log(model.initial).sum() + np.log(model.transition).sum() + np.log(model.emission).sum()
    )


@pytest.mark.parametrize("alpha", [0.3, 1.0, 4.0])
@pytest.mark.parametrize("seed", range(8))
def test_with_pseudocounts_the_objective_never_decreases(seed, alpha):
    truth = random_hmm(seed + 300, k=3, m=2)
    sequences = [random_observations(truth, seed + 3 * s, 12, missing=0.2) for s in range(3)]
    result = BaumWelch(
        truth.hidden, truth.observed, sequences, pseudocount=alpha, max_iterations=40
    ).run(seed=seed)
    assert_non_decreasing(result.log_objective)
    final = result.log_likelihood[-1] + log_prior(result.model, alpha)
    assert result.log_objective[-1] == pytest.approx(final, rel=1e-11)


def test_runs_are_reproducible():
    sequences = [umbrella_world().sample(50, seed=s)[1] for s in range(4)]
    bw = BaumWelch(WEATHER, UMBRELLA, sequences, max_iterations=30)
    first, second = bw.run(seed=1), bw.run(seed=1)
    assert first.log_likelihood == second.log_likelihood
    assert np.array_equal(first.model.transition, second.model.transition)
    assert bw.run(seed=2).log_likelihood[0] != first.log_likelihood[0]


def test_runs_converge_to_a_fixed_point():
    sequences = [umbrella_world().sample(50, seed=s)[1] for s in range(4)]
    bw = BaumWelch(WEATHER, UMBRELLA, sequences, tolerance=1e-6, max_iterations=2000)
    result = bw.run(seed=1)
    assert result.converged
    again = bw.step(result.model)
    assert np.abs(again.transition - result.model.transition).max() < 1e-2
    assert np.abs(again.emission - result.model.emission).max() < 1e-2


# ---------------------------------------------------------------------------
# W4: with the states labelled, the estimate is the pooled count ratio
# ---------------------------------------------------------------------------


def test_supervised_estimate_is_the_pooled_mle():
    truth = random_hmm(7, k=3, m=2)
    labelled = [truth.sample(400, seed=s) for s in range(5)]
    model = supervised_estimate(truth.hidden, truth.observed, labelled)
    # Oracle: M4's MLE for Prev -> Next on every transition pair, pooled over time and sequences.
    prev = DiscreteVariable("Prev", truth.hidden.states)
    nxt = DiscreteVariable("Next", truth.hidden.states)
    pairs = [
        {"Prev": a, "Next": b} for states, _ in labelled for a, b in itertools.pairwise(states)
    ]
    mle = maximum_likelihood(
        BayesianNetwork([prev, nxt], [("Prev", "Next")]), Dataset([prev, nxt], pairs)
    )
    assert model.transition == pytest.approx(mle.cpds["Next"].values.T, abs=1e-14)
    starts = [states[0] for states, _ in labelled]
    for i, s in enumerate(truth.hidden.states):
        assert model.initial[i] == pytest.approx(starts.count(s) / 5)


def test_supervised_estimate_skips_missing_observations_and_smooths():
    labelled = [(["rain", "rain", "dry"], ["umbrella", None, "none"])]
    model = supervised_estimate(WEATHER, UMBRELLA, labelled, pseudocount=1.0)
    assert model.initial == pytest.approx([2 / 3, 1 / 3])  # (1 + 1) / (1 + 2)
    assert model.transition[0] == pytest.approx([2 / 4, 2 / 4])  # rain->rain, rain->dry once each
    assert model.transition[1] == pytest.approx([1 / 2, 1 / 2])  # no transitions out of dry
    assert model.emission[0] == pytest.approx([2 / 3, 1 / 3])  # one 'umbrella' from rain
    with pytest.raises(ValidationError, match="transitions from Weather=dry"):
        supervised_estimate(WEATHER, UMBRELLA, labelled)


# ---------------------------------------------------------------------------
# Validation and failure modes
# ---------------------------------------------------------------------------


def test_impossible_sequences_under_the_initial_model():
    initial = HiddenMarkovModel(
        WEATHER, UMBRELLA, [1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]
    )
    with pytest.raises(ZeroProbabilityEvidenceError, match="sequence 1"):
        BaumWelch(WEATHER, UMBRELLA, [[U, NONE]]).run(initial=initial)


def test_a_state_never_left_has_an_undetermined_row():
    """Single-step sequences give no transitions at all: every row of A is undetermined."""
    with pytest.raises(ValidationError, match="transitions from"):
        BaumWelch(WEATHER, UMBRELLA, [[U], [NONE]]).run(seed=0)
    smoothed = BaumWelch(WEATHER, UMBRELLA, [[U], [NONE]], pseudocount=1.0).run(seed=0)
    assert smoothed.model.transition == pytest.approx(np.full((2, 2), 0.5))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sequences": []}, ValidationError, "at least one sequence"),
        ({"sequences": [[]]}, ValidationError, "sequence 1 is empty"),
        ({"sequences": [[U, "snow"]]}, UnknownStateError, "sequence 1, observation 2"),
        ({"sequences": ["umbrella"]}, ValidationError, "sequence"),
        ({"pseudocount": -1.0}, ValidationError, "pseudocount"),
        ({"max_iterations": 0}, ValidationError, "max_iterations"),
        ({"tolerance": math.nan}, ValidationError, "tolerance"),
    ],
)
def test_validation(kwargs, error, match):
    arguments = {"sequences": [[U, NONE]], **kwargs}
    with pytest.raises(error, match=match):
        BaumWelch(WEATHER, UMBRELLA, **arguments)


def test_initial_model_must_use_the_same_variables():
    other = DiscreteVariable("Other", ("rain", "dry"))
    model = HiddenMarkovModel(other, UMBRELLA, [0.5, 0.5], [[0.5, 0.5]] * 2, [[0.5, 0.5]] * 2)
    with pytest.raises(ValidationError, match="same hidden and observed variables"):
        BaumWelch(WEATHER, UMBRELLA, [[U]]).run(initial=model)


@pytest.mark.parametrize("seed", range(6))
def test_with_pseudocounts_convergence_is_judged_on_the_objective(seed):
    truth = random_hmm(seed + 400, k=3, m=2)
    sequences = [random_observations(truth, seed + 3 * s, 15, missing=0.2) for s in range(3)]
    result = BaumWelch(
        truth.hidden, truth.observed, sequences, pseudocount=2.0, tolerance=1e-4, max_iterations=500
    ).run(seed=seed)
    assert result.converged
    assert abs(result.log_objective[-1] - result.log_objective[-2]) < 1e-4
