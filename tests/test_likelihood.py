"""M4.2: log-likelihood and maximum likelihood (spec §3.2; L1–L4, L6; likelihood.md Part 2)."""

import itertools
import math
from collections import Counter

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import ValidationError
from probgraph.learning import Dataset, log_likelihood, maximum_likelihood

RAIN = DiscreteVariable("Rain", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))
Y, N = "yes", "no"


def structure() -> BayesianNetwork:
    return BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])


def f1() -> Dataset:
    rows = (
        [{"Rain": Y, "Traffic": Y}] * 3
        + [{"Rain": Y, "Traffic": N}] * 1
        + [{"Rain": N, "Traffic": Y}] * 1
        + [{"Rain": N, "Traffic": N}] * 5
    )
    return Dataset([RAIN, TRAFFIC], rows)


def brute_force_log_likelihood(model: BayesianNetwork, data: Dataset) -> float:
    """Σ_m log Σ_completions P(x) by enumeration, using only M1's joint_probability."""
    total = 0.0
    for row in data.rows():
        hidden = [v for v in model.variables if row[v.name] is None]
        p = 0.0
        for states in itertools.product(*(v.states for v in hidden)):
            full = {**row, **dict(zip((v.name for v in hidden), states, strict=True))}
            p += model.joint_probability(full)
        total += math.log(p) if p > 0 else -math.inf
    return total


def sample_data(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


# ---------------------------------------------------------------------------
# F1: exact fractions (milestone-4.md §4)
# ---------------------------------------------------------------------------


def test_f1_maximum_likelihood():
    model = maximum_likelihood(structure(), f1())
    assert model.cpds["Rain"].probability(Y, {}) == pytest.approx(2 / 5, abs=1e-15)
    assert model.cpds["Traffic"].probability(Y, {"Rain": Y}) == pytest.approx(3 / 4, abs=1e-15)
    assert model.cpds["Traffic"].probability(Y, {"Rain": N}) == pytest.approx(1 / 6, abs=1e-15)
    model.validate()


def test_f1_log_likelihood():
    model = maximum_likelihood(structure(), f1())
    expected = (
        4 * math.log(2 / 5)
        + 6 * math.log(3 / 5)
        + 3 * math.log(3 / 4)
        + math.log(1 / 4)
        + math.log(1 / 6)
        + 5 * math.log(5 / 6)
    )
    assert expected == pytest.approx(-11.682825, abs=1e-6)
    assert log_likelihood(model, f1()) == pytest.approx(expected, abs=1e-12)


def test_structure_is_not_modified_and_its_cpds_are_ignored():
    s = structure()
    s.add_cpd(TabularCPD(RAIN, (), [0.5, 0.5]))
    model = maximum_likelihood(s, f1())
    assert s.cpds["Rain"].probability(Y, {}) == 0.5
    assert "Traffic" not in s.cpds
    assert model.cpds["Rain"].probability(Y, {}) == pytest.approx(0.4)
    assert model is not s


# ---------------------------------------------------------------------------
# L1: the log-likelihood equals brute force (complete and incomplete data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_complete_data_log_likelihood_matches_brute_force(seed):
    model = random_network(seed, n_vars=(1, 6))
    data = sample_data(model, 200, seed)
    assert log_likelihood(model, data) == pytest.approx(
        brute_force_log_likelihood(model, data), abs=1e-9
    )


@pytest.mark.parametrize("seed", range(30))
def test_observed_data_log_likelihood_matches_brute_force(seed):
    model = random_network(seed, n_vars=(2, 6))
    data = sample_data(model, 120, seed).with_missing(0.35, seed=seed)
    assert log_likelihood(model, data) == pytest.approx(
        brute_force_log_likelihood(model, data), abs=1e-9
    )


def test_f3_observed_log_likelihood_at_the_f1_mle():
    """milestone-4.md F3: F1 plus (R=yes, T=?) and (R=?, T=yes), at F1's MLE."""
    model = maximum_likelihood(structure(), f1())
    rows = [*f1().rows(), {"Rain": Y, "Traffic": None}, {"Rain": None, "Traffic": Y}]
    data = Dataset([RAIN, TRAFFIC], rows)
    assert log_likelihood(model, data) == pytest.approx(-13.515405965513933, abs=1e-12)


def test_impossible_row_gives_minus_infinity():
    model = structure()
    model.add_cpd(TabularCPD(RAIN, (), [1.0, 0.0]))  # it never rains
    model.add_cpd(TabularCPD(TRAFFIC, (RAIN,), [[0.5, 0.5], [0.5, 0.5]]))
    assert log_likelihood(model, f1()) == -math.inf
    no_rain = Dataset([RAIN, TRAFFIC], [{"Rain": N, "Traffic": Y}, {"Rain": None, "Traffic": N}])
    assert log_likelihood(model, no_rain) == pytest.approx(2 * math.log(0.5), abs=1e-12)


def test_variables_must_match_the_model():
    other = DiscreteVariable("Rain", ("dry", "wet"))
    data = Dataset([other, TRAFFIC], [{"Rain": "dry", "Traffic": Y}])
    with pytest.raises(ValidationError, match="Rain"):
        log_likelihood(maximum_likelihood(structure(), f1()), data)
    with pytest.raises(ValidationError, match="Rain"):
        maximum_likelihood(structure(), data)
    with pytest.raises(ValidationError, match="Late"):
        maximum_likelihood(late_network(), f1())


def test_row_order_does_not_matter():
    model = maximum_likelihood(structure(), f1())
    shuffled = Dataset([RAIN, TRAFFIC], list(reversed(list(f1().rows()))))
    assert log_likelihood(model, shuffled) == pytest.approx(log_likelihood(model, f1()), abs=1e-12)
    again = maximum_likelihood(structure(), shuffled)
    for name in ("Rain", "Traffic"):
        np.testing.assert_array_equal(again.cpds[name].values, model.cpds[name].values)


# ---------------------------------------------------------------------------
# L2: the closed form, against brute-force tallies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_mle_is_the_ratio_of_counts(seed):
    truth = random_network(seed, n_vars=(1, 5), cards=(1, 3))
    data = sample_data(truth, 3000, seed)
    try:
        model = maximum_likelihood(truth, data)
    except ValidationError:
        model = maximum_likelihood(truth, data, unseen="uniform")
    rows = list(data.rows())
    for name, cpd in model.cpds.items():
        parents = cpd.parent_names
        for config in itertools.product(*(p.states for p in cpd.parents)):
            given = dict(zip(parents, config, strict=True))
            selected = [r for r in rows if all(r[p] == s for p, s in given.items())]
            tally = Counter(r[name] for r in selected)
            for state in cpd.variable.states:
                expected = (
                    tally[state] / len(selected) if selected else 1 / cpd.variable.cardinality
                )
                assert cpd.probability(state, given) == pytest.approx(expected, abs=1e-15)


# ---------------------------------------------------------------------------
# L3: no valid perturbation increases the likelihood
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_perturbations_never_increase_the_likelihood(seed):
    truth = random_network(seed, n_vars=(2, 5), cards=(2, 3))
    data = sample_data(truth, 500, seed)
    model = maximum_likelihood(truth, data, unseen="uniform")
    best = log_likelihood(model, data)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        name = str(rng.choice(list(model.cpds)))
        cpd = model.cpds[name]
        q = rng.dirichlet(
            np.ones(cpd.variable.cardinality), size=int(np.prod(cpd.values.shape[1:]))
        ).T
        epsilon = float(rng.uniform(0.01, 0.5))
        mixed = (1 - epsilon) * cpd.values + epsilon * q.reshape(cpd.values.shape)
        perturbed = BayesianNetwork(model.variables, model.edges())
        for other in model.cpds.values():
            perturbed.add_cpd(
                other
                if other.variable.name != name
                else TabularCPD(cpd.variable, cpd.parents, mixed)
            )
        assert log_likelihood(perturbed, data) <= best + 1e-9


def test_the_mle_is_a_strict_maximum_on_f1():
    model = maximum_likelihood(structure(), f1())
    best = log_likelihood(model, f1())
    for p in (0.39, 0.41, 0.3, 0.5):
        nudged = structure()
        nudged.add_cpd(TabularCPD(RAIN, (), [1 - p, p]))
        nudged.add_cpd(model.cpds["Traffic"])
        assert log_likelihood(nudged, f1()) < best


# ---------------------------------------------------------------------------
# L4: unseen parent configurations; L6: incomplete data
# ---------------------------------------------------------------------------


def no_rain_data() -> Dataset:
    return Dataset(
        [RAIN, TRAFFIC],
        [{"Rain": N, "Traffic": Y}, {"Rain": N, "Traffic": N}, {"Rain": N, "Traffic": N}],
    )


def test_unseen_parent_configuration_raises_and_names_it():
    with pytest.raises(ValidationError, match=r"Traffic.*Rain=yes"):
        maximum_likelihood(structure(), no_rain_data())


def test_unseen_uniform_is_an_explicit_opt_in():
    model = maximum_likelihood(structure(), no_rain_data(), unseen="uniform")
    assert model.cpds["Traffic"].distribution({"Rain": Y}).tolist() == [0.5, 0.5]
    assert model.cpds["Traffic"].probability(Y, {"Rain": N}) == pytest.approx(1 / 3)
    assert model.cpds["Rain"].probability(Y, {}) == 0.0  # zero counts, but seen parents: MLE is 0
    with pytest.raises(ValidationError, match="unseen"):
        maximum_likelihood(structure(), no_rain_data(), unseen="ignore")  # type: ignore[arg-type]


def test_incomplete_data_points_to_em():
    data = f1().with_missing(0.2, seed=1)
    with pytest.raises(ValidationError, match="ExpectationMaximisation"):
        maximum_likelihood(structure(), data)


def test_empty_data_has_zero_log_likelihood_but_no_mle():
    empty = Dataset([RAIN, TRAFFIC], [])
    assert log_likelihood(maximum_likelihood(structure(), f1()), empty) == 0.0
    with pytest.raises(ValidationError, match="unseen"):
        maximum_likelihood(structure(), empty)
