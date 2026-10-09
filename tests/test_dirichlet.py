"""M4.4: Dirichlet priors and posterior point estimates (spec §3.3; B1–B3; dirichlet.md Part 1)."""

import itertools
from collections import Counter

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable
from probgraph.exceptions import ValidationError
from probgraph.learning import Dataset, DirichletPrior, bayesian_estimate, maximum_likelihood

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


def no_rain() -> Dataset:
    return Dataset(
        [RAIN, TRAFFIC],
        [{"Rain": N, "Traffic": Y}, {"Rain": N, "Traffic": N}, {"Rain": N, "Traffic": N}],
    )


# ---------------------------------------------------------------------------
# F1 (milestone-4.md §4)
# ---------------------------------------------------------------------------


def test_f1_laplace_posterior_means():
    model = bayesian_estimate(structure(), f1(), DirichletPrior.uniform(1.0))
    assert model.cpds["Rain"].probability(Y, {}) == pytest.approx(5 / 12, abs=1e-15)
    assert model.cpds["Traffic"].probability(Y, {"Rain": Y}) == pytest.approx(2 / 3, abs=1e-15)
    assert model.cpds["Traffic"].probability(Y, {"Rain": N}) == pytest.approx(1 / 4, abs=1e-15)


# ---------------------------------------------------------------------------
# B1: conjugacy, against numerical integration (no conjugacy used)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "alpha"), [((3, 1), (1.0, 1.0)), ((0, 7), (0.5, 2.0)), ((12, 30), (3.0, 0.7))]
)
def test_two_state_posterior_mean_and_mode_by_quadrature(counts, alpha):
    a = np.array(alpha) + np.array(counts)
    # Mean: substitute θ = s² (dθ = 2s ds). This removes the θ^(a-1) singularity at 0 when
    # a < 1, which a uniform grid in θ integrates badly (the first version of this oracle did).
    s = np.linspace(0.0, 1.0, 400_001)[1:-1]
    theta = s**2
    log_weight = (a[0] - 1) * np.log(theta) + (a[1] - 1) * np.log1p(-theta) + np.log(2 * s)
    weight = np.exp(log_weight - log_weight.max())
    mean = float((theta * weight).sum() / weight.sum())
    assert mean == pytest.approx(a[0] / a.sum(), abs=2e-6)
    # Mode: maximise the density itself on a uniform grid in θ.
    if (a >= 1).all() and a.sum() > 2:
        grid = np.linspace(0.0, 1.0, 400_001)[1:-1]
        log_post = (a[0] - 1) * np.log(grid) + (a[1] - 1) * np.log1p(-grid)
        assert float(grid[np.argmax(log_post)]) == pytest.approx(
            (a[0] - 1) / (a.sum() - 2), abs=1e-5
        )


def test_three_state_posterior_mean_by_quadrature():
    counts, alpha = np.array([4.0, 1.0, 6.0]), np.array([1.5, 0.8, 2.0])
    grid = np.linspace(0.0, 1.0, 1201)[1:-1]
    t1, t2 = np.meshgrid(grid, grid, indexing="ij")
    t3 = 1.0 - t1 - t2
    inside = t3 > 0
    a = counts + alpha
    log_post = np.where(
        inside,
        (a[0] - 1) * np.log(t1)
        + (a[1] - 1) * np.log(t2)
        + (a[2] - 1) * np.log(np.where(inside, t3, 1.0)),
        -np.inf,
    )
    w = np.exp(log_post - log_post[inside].max())
    means = [float((t * w).sum() / w.sum()) for t in (t1, t2, np.where(inside, t3, 0.0))]
    np.testing.assert_allclose(means, a / a.sum(), atol=2e-3)


# ---------------------------------------------------------------------------
# B2: closed forms on random networks, against tallies
# ---------------------------------------------------------------------------


def tallies(data: Dataset, child: str, parents, config):
    rows = [r for r in data.rows() if all(r[p] == s for p, s in zip(parents, config, strict=True))]
    return Counter(r[child] for r in rows), len(rows)


@pytest.mark.parametrize("seed", range(20))
def test_posterior_mean_matches_counts_plus_pseudocounts(seed):
    truth = random_network(seed, n_vars=(1, 5), cards=(2, 3))
    data = Dataset.from_samples(
        truth,
        AncestralSampler(truth, seed=seed).sample(int(np.random.default_rng(seed).integers(0, 60))),
    )
    alpha = float(np.random.default_rng(seed).uniform(0.2, 3.0))
    model = bayesian_estimate(truth, data, DirichletPrior.uniform(alpha))
    for name, cpd in model.cpds.items():
        k = cpd.variable.cardinality
        for config in itertools.product(*(p.states for p in cpd.parents)):
            counts, n_u = tallies(data, name, cpd.parent_names, config)
            given = dict(zip(cpd.parent_names, config, strict=True))
            for state in cpd.variable.states:
                expected = (counts[state] + alpha) / (n_u + k * alpha)
                assert cpd.probability(state, given) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("seed", range(15))
def test_posterior_mean_is_the_convex_combination(seed):
    truth = random_network(seed + 40, n_vars=(2, 5), cards=(2, 3))
    data = Dataset.from_samples(truth, AncestralSampler(truth, seed=seed).sample(400))
    prior = DirichletPrior.bdeu(4.0)
    bayes = bayesian_estimate(truth, data, prior)
    mle = maximum_likelihood(truth, data, unseen="uniform")
    for name, cpd in bayes.cpds.items():
        parents = list(cpd.parent_names)
        pseudo = prior.pseudocounts(cpd.variable, cpd.parents)
        counts = data.counts([name, *parents]).values
        n_u = counts.sum(axis=0)
        lam = n_u / (n_u + pseudo.sum(axis=0))
        prior_mean = pseudo / pseudo.sum(axis=0, keepdims=True)
        expected = lam * mle.cpds[name].values + (1 - lam) * prior_mean
        np.testing.assert_allclose(cpd.values, expected, atol=1e-13)


# ---------------------------------------------------------------------------
# B3: limits; unseen configurations need no special case
# ---------------------------------------------------------------------------


def test_vanishing_prior_recovers_the_mle():
    data = Dataset.from_samples(
        late_network(), AncestralSampler(late_network(), seed=1).sample(500)
    )
    bayes = bayesian_estimate(late_network(), data, DirichletPrior.uniform(1e-12))
    mle = maximum_likelihood(late_network(), data)
    for name in mle.cpds:
        np.testing.assert_allclose(bayes.cpds[name].values, mle.cpds[name].values, atol=1e-9)


def test_unseen_column_is_the_prior_mean():
    model = bayesian_estimate(structure(), no_rain(), DirichletPrior.uniform(2.0))
    assert model.cpds["Traffic"].distribution({"Rain": Y}).tolist() == [0.5, 0.5]
    assert model.cpds["Traffic"].probability(Y, {"Rain": N}) == pytest.approx((1 + 2) / (3 + 4))


def test_prior_washes_out_with_data():
    truth = late_network()
    data = Dataset.from_samples(truth, AncestralSampler(truth, seed=2).sample(50_000))
    bayes = bayesian_estimate(truth, data, DirichletPrior.uniform(1.0))
    mle = maximum_likelihood(truth, data)
    for name, cpd in bayes.cpds.items():
        parents = list(cpd.parent_names)
        n_u = data.counts([name, *parents]).values.sum(axis=0)
        bound = cpd.variable.cardinality / (n_u + cpd.variable.cardinality)  # α·/(N(u) + α·)
        assert (np.abs(cpd.values - mle.cpds[name].values) <= bound + 1e-15).all()


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------


def test_map_with_k2_equals_the_mle():
    model = bayesian_estimate(structure(), f1(), DirichletPrior.uniform(1.0), point="map")
    mle = maximum_likelihood(structure(), f1())
    for name in mle.cpds:
        np.testing.assert_allclose(model.cpds[name].values, mle.cpds[name].values, atol=1e-15)


def test_map_with_alpha_two():
    model = bayesian_estimate(structure(), f1(), DirichletPrior.uniform(2.0), point="map")
    assert model.cpds["Rain"].probability(Y, {}) == pytest.approx((4 + 1) / (10 + 2))
    assert model.cpds["Traffic"].probability(Y, {"Rain": N}) == pytest.approx((1 + 1) / (6 + 2))


def test_map_rejects_alpha_below_one_and_flat_columns():
    with pytest.raises(ValidationError, match="MAP"):
        bayesian_estimate(structure(), f1(), DirichletPrior.uniform(0.5), point="map")
    with pytest.raises(ValidationError, match=r"Traffic.*Rain=yes"):
        bayesian_estimate(structure(), no_rain(), DirichletPrior.uniform(1.0), point="map")
    model = bayesian_estimate(structure(), no_rain(), DirichletPrior.uniform(2.0), point="map")
    assert model.cpds["Traffic"].distribution({"Rain": Y}).tolist() == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Priors: BDeu's equivalent sample size; explicit pseudocounts; validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ess", [0.5, 1.0, 10.0])
def test_bdeu_family_totals_equal_the_equivalent_sample_size(ess):
    prior = DirichletPrior.bdeu(ess)
    for cpd in late_network().cpds.values():
        pseudo = prior.pseudocounts(cpd.variable, cpd.parents)
        assert pseudo.shape == cpd.values.shape
        assert pseudo.sum() == pytest.approx(ess)
        assert np.allclose(pseudo, pseudo.flat[0])  # spread uniformly over the family's cells


def test_explicit_prior():
    prior = DirichletPrior.explicit({"Rain": [1.0, 3.0], "Traffic": [[2.0, 1.0], [1.0, 2.0]]})
    model = bayesian_estimate(structure(), f1(), prior)
    assert model.cpds["Rain"].probability(Y, {}) == pytest.approx((4 + 3) / (10 + 4))
    assert model.cpds["Traffic"].probability(Y, {"Rain": Y}) == pytest.approx((3 + 2) / (4 + 3))


@pytest.mark.parametrize(
    "pseudocounts",
    [
        {"Rain": [1.0, 1.0]},  # Traffic missing
        {"Rain": [1.0, 1.0], "Traffic": [1.0, 1.0]},  # wrong shape
        {"Rain": [1.0, 0.0], "Traffic": [[1.0, 1.0], [1.0, 1.0]]},  # not positive
        {"Rain": [1.0, np.nan], "Traffic": [[1.0, 1.0], [1.0, 1.0]]},
    ],
)
def test_explicit_prior_validation(pseudocounts):
    with pytest.raises(ValidationError):
        bayesian_estimate(structure(), f1(), DirichletPrior.explicit(pseudocounts))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scalar_prior_validation(bad):
    with pytest.raises(ValidationError):
        DirichletPrior.uniform(bad)
    with pytest.raises(ValidationError):
        DirichletPrior.bdeu(bad)


def test_estimate_validation():
    with pytest.raises(ValidationError, match="ExpectationMaximisation"):
        bayesian_estimate(structure(), f1().with_missing(0.3, seed=1), DirichletPrior.uniform(1.0))
    with pytest.raises(ValidationError, match="point"):
        bayesian_estimate(structure(), f1(), DirichletPrior.uniform(1.0), point="median")  # type: ignore[arg-type]
