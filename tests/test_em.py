"""M4.6: expectation–maximisation (spec §3.4; M1–M4; em.md)."""

import math
from fractions import Fraction

import numpy as np
import pytest
from support import brute_force_expected_counts, late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.learning import (
    Dataset,
    DirichletPrior,
    EMResult,
    ExpectationMaximisation,
    bayesian_estimate,
    expected_counts,
    log_likelihood,
    maximum_likelihood,
)
from probgraph.learning import em as em_module

RAIN = DiscreteVariable("Rain", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))
Y, N = "yes", "no"


def structure() -> BayesianNetwork:
    return BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])


def f1_rows() -> list[dict[str, str | None]]:
    return (
        [{"Rain": Y, "Traffic": Y}] * 3
        + [{"Rain": Y, "Traffic": N}] * 1
        + [{"Rain": N, "Traffic": Y}] * 1
        + [{"Rain": N, "Traffic": N}] * 5
    )


def f3() -> Dataset:
    return Dataset(
        [RAIN, TRAFFIC], [*f1_rows(), {"Rain": Y, "Traffic": None}, {"Rain": None, "Traffic": Y}]
    )


def f1_mle() -> BayesianNetwork:
    return maximum_likelihood(structure(), Dataset([RAIN, TRAFFIC], f1_rows()))


def sample(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


# ---------------------------------------------------------------------------
# Fixture F3 (milestone-4.md §4), exactly
# ---------------------------------------------------------------------------


def test_f3_expected_counts():
    """(R=yes, T=?) splits 3/4 : 1/4 over T; (R=?, T=yes) splits 3/4 : 1/4 over R."""
    counts = expected_counts(f1_mle(), f3())
    # Rain: axis (no, yes).
    assert counts["Rain"] == pytest.approx([6 + 1 / 4, 4 + 1 + 3 / 4], abs=1e-14)
    # Traffic: axes (Traffic, Rain), each (no, yes).
    expected = [[5, 1 + 1 / 4], [1 + 1 / 4, 3 + 3 / 4 + 3 / 4]]
    assert counts["Traffic"] == pytest.approx(np.array(expected), abs=1e-14)


def test_f3_one_iteration_gives_the_exact_fractions():
    em = ExpectationMaximisation(structure(), f3())
    model = em.step(f1_mle())
    rain, traffic = model.cpds["Rain"], model.cpds["Traffic"]
    assert rain.probability(Y, {}) == pytest.approx(float(Fraction(23, 48)), abs=1e-15)
    assert traffic.probability(Y, {"Rain": Y}) == pytest.approx(float(Fraction(18, 23)), abs=1e-15)
    assert traffic.probability(Y, {"Rain": N}) == pytest.approx(float(Fraction(1, 5)), abs=1e-15)


def test_f3_log_likelihood_history():
    result = ExpectationMaximisation(structure(), f3(), max_iterations=1).run(initial=f1_mle())
    assert result.iterations == 1
    assert not result.converged
    assert result.log_likelihood == pytest.approx((-13.515406, -13.314771), abs=5e-7)
    assert result.log_objective == result.log_likelihood  # no prior: EM ascends ℓ itself


def test_f3_runs_to_a_fixed_point():
    em = ExpectationMaximisation(structure(), f3(), tolerance=1e-13)
    result = em.run(initial=f1_mle())
    assert result.converged
    assert len(result.log_likelihood) == result.iterations + 1
    assert result.log_likelihood[-1] == pytest.approx(log_likelihood(result.model, f3()), abs=1e-12)
    again = em.step(result.model)
    for name, cpd in result.model.cpds.items():
        assert np.abs(again.cpds[name].values - cpd.values).max() < 1e-6


# ---------------------------------------------------------------------------
# M1: expected counts equal the brute-force expectation over completions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_expected_counts_match_brute_force(seed):
    truth = random_network(seed, n_vars=(1, 5), cards=(1, 3), edge_prob=0.5)
    data = sample(truth, 40, seed).with_missing(0.4, seed=seed + 1)
    counts = expected_counts(truth, data)
    oracle = brute_force_expected_counts(truth, data)
    assert set(counts) == set(oracle)
    for name in oracle:
        assert counts[name] == pytest.approx(oracle[name], abs=1e-11)
        # Every row contributes exactly 1 to each family (P(o) cancels).
        assert counts[name].sum() == pytest.approx(data.n_rows, abs=1e-10)


def test_rows_with_the_same_observations_share_one_calibration(monkeypatch):
    built = []
    original = em_module.JunctionTree

    def counting(*args, **kwargs):
        built.append(kwargs.get("evidence"))
        return original(*args, **kwargs)

    monkeypatch.setattr(em_module, "JunctionTree", counting)
    data = Dataset([RAIN, TRAFFIC], [*f1_rows()] + [{"Rain": Y, "Traffic": None}] * 7)
    expected_counts(f1_mle(), data)
    assert len(built) == 1  # complete rows need no calibration; seven identical rows need one


# ---------------------------------------------------------------------------
# M2: monotonicity
# ---------------------------------------------------------------------------


def assert_non_decreasing(history, slack=1e-10):
    steps = np.diff(history)
    assert (steps >= -slack).all(), steps.min()


@pytest.mark.parametrize("seed", range(20))
def test_log_likelihood_never_decreases(seed):
    truth = random_network(seed + 100, n_vars=(2, 5), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 150, seed).with_missing(0.35, seed=seed)
    result = ExpectationMaximisation(truth, data, max_iterations=60).run(seed=seed)
    assert result.iterations >= 1
    assert_non_decreasing(result.log_likelihood)
    assert result.log_likelihood[-1] == pytest.approx(
        log_likelihood(result.model, data), rel=1e-11, abs=1e-11
    )


def log_prior_term(model: BayesianNetwork, prior: DirichletPrior, exponent_shift: float) -> float:
    """Σ (α + shift - 1) log θ over every cell, with 0 · log 0 = 0 (em.md §7)."""
    total = 0.0
    for v in model.variables:
        cpd = model.cpds[v.name]
        parents = [p for p in model.variables if p.name in model.parents(v.name)]
        theta = np.transpose(
            cpd.values, [0, *(1 + cpd.parent_names.index(p.name) for p in parents)]
        )
        exponent = prior.pseudocounts(v, parents) + exponent_shift - 1
        mask = exponent != 0
        total += float(np.sum(exponent[mask] * np.log(theta[mask])))
    return total


@pytest.mark.parametrize(
    ("prior", "point", "shift"),
    [
        (DirichletPrior.uniform(1.0), "map", 0.0),
        (DirichletPrior.uniform(2.5), "map", 0.0),
        (DirichletPrior.uniform(0.5), "mean", 1.0),
        (DirichletPrior.bdeu(4.0), "mean", 1.0),
    ],
)
@pytest.mark.parametrize("seed", range(8))
def test_log_posterior_never_decreases(seed, prior, point, shift):
    """MAP-EM ascends ℓ + log Dir(α); mean-EM is MAP-EM for Dir(α + 1) (em.md §7)."""
    truth = random_network(seed + 200, n_vars=(2, 4), cards=(2, 3), edge_prob=0.6)
    data = sample(truth, 80, seed).with_missing(0.4, seed=seed)
    result = ExpectationMaximisation(truth, data, prior=prior, point=point, max_iterations=40).run(
        seed=seed
    )
    assert_non_decreasing(result.log_objective)
    final = log_likelihood(result.model, data) + log_prior_term(result.model, prior, shift)
    assert result.log_objective[-1] == pytest.approx(final, rel=1e-11)


# ---------------------------------------------------------------------------
# M3: complete data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_complete_data_reaches_the_mle_in_one_iteration(seed):
    truth = random_network(seed + 300, n_vars=(1, 5), cards=(2, 2))
    data = sample(truth, 3000, seed)
    mle = maximum_likelihood(truth, data)
    result = ExpectationMaximisation(truth, data).run(seed=seed)
    assert result.converged and result.iterations == 1
    for name, cpd in mle.cpds.items():
        assert result.model.cpds[name].values == pytest.approx(cpd.values, abs=1e-14)


@pytest.mark.parametrize("point", ["mean", "map"])
def test_complete_data_with_a_prior_gives_the_bayesian_estimate(point):
    truth = late_network()
    data = sample(truth, 500, seed=1)
    prior = DirichletPrior.uniform(2.0)
    result = ExpectationMaximisation(truth, data, prior=prior, point=point).run(seed=0)
    assert result.converged and result.iterations == 1
    reference = bayesian_estimate(truth, data, prior, point=point)
    for name, cpd in reference.cpds.items():
        assert result.model.cpds[name].values == pytest.approx(cpd.values, abs=1e-14)


# ---------------------------------------------------------------------------
# §6: a fixed point is a stationary point of ℓ
# ---------------------------------------------------------------------------


def test_converged_parameters_are_stationary():
    truth = late_network()
    data = sample(truth, 400, seed=3).with_missing(0.3, seed=4)
    em = ExpectationMaximisation(truth, data, tolerance=1e-14, max_iterations=2000)
    result = em.run(seed=5)
    assert result.converged
    start = em.initial_model(seed=5)

    def slope(model: BayesianNetwork, name: str, column: tuple[int, ...]) -> float:
        """dℓ/dε along θ(s0|u) += ε, θ(s1|u) -= ε (a direction inside the simplex)."""
        h = 1e-6
        values = []
        for sign in (1, -1):
            cpd = model.cpds[name]
            theta = cpd.values.copy()
            theta[(0, *column)] += sign * h
            theta[(1, *column)] -= sign * h
            moved = BayesianNetwork(model.variables, model.edges())
            for other in model.cpds.values():
                moved.add_cpd(
                    TabularCPD(cpd.variable, cpd.parents, theta) if other is cpd else other
                )
            values.append(log_likelihood(moved, data))
        return (values[0] - values[1]) / (2 * h)

    columns = [("Rain", ()), ("Late", (0,)), ("Umbrella", (1,)), ("Traffic", (1, 0))]
    at_start = [abs(slope(start, n, c)) for n, c in columns]
    at_end = [abs(slope(result.model, n, c)) for n, c in columns]
    assert max(at_start) > 1.0
    assert max(at_end) < 1e-3


# ---------------------------------------------------------------------------
# Runs, initialisation and validation
# ---------------------------------------------------------------------------


def test_random_initialisation_is_reproducible():
    truth = late_network()
    data = sample(truth, 200, seed=2).with_missing(0.3, seed=2)
    em = ExpectationMaximisation(truth, data, max_iterations=20)
    first, second = em.run(seed=9), em.run(seed=9)
    assert first.log_likelihood == second.log_likelihood
    assert isinstance(first, EMResult)
    for name in truth.cpds:
        assert np.array_equal(first.model.cpds[name].values, second.model.cpds[name].values)
    assert em.run(seed=10).log_likelihood[0] != first.log_likelihood[0]


def test_the_initial_model_is_used_as_given():
    result = ExpectationMaximisation(structure(), f3(), max_iterations=1).run(initial=f1_mle())
    assert result.log_likelihood[0] == pytest.approx(log_likelihood(f1_mle(), f3()), abs=1e-12)


def test_rejects_an_initial_model_that_rules_out_an_observed_row():
    initial = BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])
    initial.add_cpd(TabularCPD(RAIN, (), [1.0, 0.0]))
    initial.add_cpd(TabularCPD(TRAFFIC, (RAIN,), [[0.5, 0.5], [0.5, 0.5]]))
    with pytest.raises(ZeroProbabilityEvidenceError, match="probability 0"):
        ExpectationMaximisation(structure(), f3()).run(initial=initial)


def test_rejects_an_initial_model_with_a_different_graph():
    other = BayesianNetwork([RAIN, TRAFFIC], [("Traffic", "Rain")])
    other.add_cpd(TabularCPD(TRAFFIC, (), [0.5, 0.5]))
    other.add_cpd(TabularCPD(RAIN, (TRAFFIC,), [[0.5, 0.5], [0.5, 0.5]]))
    with pytest.raises(ValidationError, match="same variables and edges"):
        ExpectationMaximisation(structure(), f3()).run(initial=other)


def test_unseen_expected_column_without_a_prior_raises():
    """No row can have Rain = yes, so Traffic | Rain = yes gets no expected count at all."""
    data = Dataset([RAIN, TRAFFIC], [{"Rain": N, "Traffic": None}, {"Rain": N, "Traffic": Y}])
    with pytest.raises(ValidationError, match="Traffic \\| Rain=yes"):
        ExpectationMaximisation(structure(), data).run(seed=0)
    smoothed = ExpectationMaximisation(structure(), data, prior=DirichletPrior.uniform()).run(
        seed=0
    )
    assert smoothed.model.cpds["Traffic"].probability(Y, {"Rain": Y}) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"tolerance": math.nan}, "tolerance"),
        ({"point": "median"}, "point"),
        ({"point": "map", "prior": DirichletPrior.uniform(0.5)}, "MAP needs"),
    ],
)
def test_parameter_validation(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        ExpectationMaximisation(structure(), f3(), **kwargs).run(seed=0)


def test_rejects_mismatched_data():
    with pytest.raises(ValidationError, match="do not match"):
        ExpectationMaximisation(structure(), Dataset([RAIN], [{"Rain": Y}]))


def test_with_a_prior_the_likelihood_itself_may_decrease():
    """Why log_objective exists: mean-EM ascends ℓ + Σ α log θ, not ℓ (em.md §7)."""
    truth = random_network(211, n_vars=(2, 4), cards=(2, 3), edge_prob=0.6)
    data = sample(truth, 80, 11).with_missing(0.4, seed=11)
    result = ExpectationMaximisation(
        truth, data, prior=DirichletPrior.bdeu(4.0), max_iterations=40
    ).run(seed=11)
    assert np.diff(result.log_likelihood).min() < -1e-2
    assert_non_decreasing(result.log_objective)
