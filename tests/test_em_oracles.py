"""M4.7: EM against brute force, the symmetric fixed point, and recovery (M5–M7; em.md §2, §8)."""

import itertools
import math

import numpy as np
import pytest
from support import brute_force_expected_counts, joint_table, late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.learning import (
    Dataset,
    DirichletPrior,
    ExpectationMaximisation,
    maximum_likelihood,
)

Z = 5.0


def bernstein(p: float, n: int) -> float:
    return Z * math.sqrt(p * (1 - p) / n) + Z**2 / (3 * n)


def sample(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


def declared_parents(model: BayesianNetwork, name: str) -> list[DiscreteVariable]:
    return [p for p in model.variables if p.name in model.parents(name)]


# ---------------------------------------------------------------------------
# M7: agreement with a brute-force EM that only uses joint tables
# ---------------------------------------------------------------------------


def brute_force_log_likelihood(model: BayesianNetwork, data: Dataset) -> float:
    table = joint_table(model)
    column = [data.names.index(v.name) for v in model.variables]
    total = 0.0
    for codes in data.codes:
        index = tuple(slice(None) if codes[c] < 0 else int(codes[c]) for c in column)
        total += math.log(float(np.sum(table[index])))
    return total


def brute_force_em(
    initial: BayesianNetwork,
    data: Dataset,
    iterations: int,
    prior: DirichletPrior | None,
    point: str,
) -> tuple[BayesianNetwork, list[float]]:
    model = initial
    history = [brute_force_log_likelihood(model, data)]
    for _ in range(iterations):
        counts = brute_force_expected_counts(model, data)
        new = BayesianNetwork(model.variables, model.edges())
        for v in model.variables:
            parents = declared_parents(model, v.name)
            c = counts[v.name]
            if prior is not None:
                c = c + prior.pseudocounts(v, parents) - (1.0 if point == "map" else 0.0)
            new.add_cpd(TabularCPD(v, parents, c / c.sum(axis=0, keepdims=True)))
        model = new
        history.append(brute_force_log_likelihood(model, data))
    return model, history


@pytest.mark.parametrize(
    ("prior", "point"),
    [
        (None, "mean"),
        (DirichletPrior.uniform(0.7), "mean"),
        (DirichletPrior.bdeu(3.0), "mean"),
        (DirichletPrior.uniform(1.5), "map"),
    ],
)
@pytest.mark.parametrize("seed", range(10))
def test_em_agrees_with_brute_force_em(seed, prior, point):
    truth = random_network(seed + 700, n_vars=(2, 4), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 60, seed).with_missing(0.3, seed=seed)
    em = ExpectationMaximisation(
        truth, data, prior=prior, point=point, max_iterations=8, tolerance=0
    )
    initial = em.initial_model(seed)
    result = em.run(initial=initial)
    reference, history = brute_force_em(initial, data, 8, prior, point)
    assert result.iterations == 8
    assert result.log_likelihood == pytest.approx(history, rel=1e-11, abs=1e-11)
    for v in truth.variables:
        assert result.model.cpds[v.name].values == pytest.approx(
            reference.cpds[v.name].values, abs=1e-12
        )


# ---------------------------------------------------------------------------
# M5: the symmetric fixed point of a latent-class model (em.md §8, Theorem 2)
# ---------------------------------------------------------------------------

CLASS = DiscreteVariable("C", ("c0", "c1"))
FEATURES = [DiscreteVariable(f"F{j}", ("a", "b", "c")) for j in range(3)]


def latent_class_truth() -> BayesianNetwork:
    model = BayesianNetwork([CLASS, *FEATURES], [("C", f.name) for f in FEATURES])
    model.add_cpd(TabularCPD(CLASS, (), [0.4, 0.6]))
    tables = [
        [[0.7, 0.1], [0.2, 0.2], [0.1, 0.7]],
        [[0.6, 0.2], [0.3, 0.1], [0.1, 0.7]],
        [[0.1, 0.5], [0.8, 0.2], [0.1, 0.3]],
    ]
    for feature, table in zip(FEATURES, tables, strict=True):
        model.add_cpd(TabularCPD(feature, (CLASS,), table))
    return model


def latent_class_data(n: int = 1500) -> Dataset:
    """C is never observed; each feature is missing 10% of the time (MCAR)."""
    full = sample(latent_class_truth(), n, seed=1)
    return full.with_missing(1.0, seed=0, variables=["C"]).with_missing(
        0.1, seed=2, variables=[f.name for f in FEATURES]
    )


def symmetric_model(seed: int) -> BayesianNetwork:
    rng = np.random.default_rng(seed)
    model = BayesianNetwork([CLASS, *FEATURES], [("C", f.name) for f in FEATURES])
    model.add_cpd(TabularCPD(CLASS, (), rng.dirichlet([1.0, 1.0])))
    for feature in FEATURES:
        column = rng.dirichlet(np.ones(3))
        model.add_cpd(TabularCPD(feature, (CLASS,), np.stack([column, column], axis=1)))
    return model


def empirical_feature_marginals(data: Dataset) -> dict[str, np.ndarray]:
    """N_j(f) / N_j over the rows where F_j is observed."""
    return {
        f.name: data.counts([f.name]).values / data.counts([f.name]).values.sum() for f in FEATURES
    }


@pytest.mark.parametrize("seed", range(5))
def test_one_step_from_symmetry_gives_theorem_2(seed):
    """θ' = (N_j(f) + M_j θ) / N for every class, and P(C) is unchanged (em.md §8)."""
    data = latent_class_data()
    start = symmetric_model(seed)
    model = ExpectationMaximisation(latent_class_truth(), data).step(start)
    assert model.cpds["C"].values == pytest.approx(start.cpds["C"].values, abs=1e-12)
    for f in FEATURES:
        observed = data.counts([f.name]).values
        missing = data.n_rows - observed.sum()
        expected = (observed + missing * start.cpds[f.name].values[:, 0]) / data.n_rows
        for c in range(2):
            assert model.cpds[f.name].values[:, c] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_symmetric_start_never_breaks_symmetry(seed):
    data = latent_class_data()
    result = ExpectationMaximisation(latent_class_truth(), data, tolerance=1e-12).run(
        initial=symmetric_model(seed)
    )
    assert result.converged
    assert result.model.cpds["C"].values == pytest.approx(
        symmetric_model(seed).cpds["C"].values, abs=1e-12
    )
    for name, marginal in empirical_feature_marginals(data).items():
        values = result.model.cpds[name].values
        assert values[:, 0] == pytest.approx(values[:, 1], abs=1e-12)  # still symmetric
        assert values[:, 0] == pytest.approx(marginal, abs=1e-9)  # the fixed point N_j(f) / N_j
    # There, the model fits the features as if they were independent.
    independent = sum(
        float(np.sum(data.counts([name]).values * np.log(marginal)))
        for name, marginal in empirical_feature_marginals(data).items()
    )
    assert result.log_likelihood[-1] == pytest.approx(independent, rel=1e-11)


def test_with_only_the_class_missing_the_symmetric_point_is_reached_in_one_step():
    data = sample(latent_class_truth(), 500, seed=3).with_missing(1.0, seed=0, variables=["C"])
    model = ExpectationMaximisation(latent_class_truth(), data).step(symmetric_model(0))
    for name, marginal in empirical_feature_marginals(data).items():
        for c in range(2):
            assert model.cpds[name].values[:, c] == pytest.approx(marginal, abs=1e-12)


def test_a_small_perturbation_escapes_the_symmetric_point():
    """The symmetric point is stationary (em.md §6) but not a maximum: nudged, EM climbs away."""
    data = latent_class_data()
    em = ExpectationMaximisation(latent_class_truth(), data, tolerance=1e-9, max_iterations=1000)
    stuck = em.run(initial=symmetric_model(0))
    nudged = symmetric_model(0)
    feature = nudged.cpds["F0"]
    values = feature.values.copy()
    values[:, 0] += [1e-3, -1e-3, 0.0]
    nudged.add_cpd(TabularCPD(FEATURES[0], (CLASS,), values))
    escaped = em.run(initial=nudged)
    assert escaped.log_likelihood[-1] > stuck.log_likelihood[-1] + 100


def class_permutation_distance(a: BayesianNetwork, b: BayesianNetwork) -> float:
    """The smaller parameter distance over the two labellings of the hidden class."""
    best = math.inf
    for perm in itertools.permutations(range(2)):
        d = abs(a.cpds["C"].values - b.cpds["C"].values[list(perm)]).max()
        for f in FEATURES:
            d = max(d, abs(a.cpds[f.name].values - b.cpds[f.name].values[:, list(perm)]).max())
        best = min(best, d)
    return float(best)


def test_random_starts_recover_the_classes_up_to_relabelling():
    """Label switching: the likelihood cannot tell which class is called c0 (a symmetry, so
    every optimum comes in pairs). Up to relabelling, EM recovers the generating classes."""
    data = latent_class_data()
    em = ExpectationMaximisation(latent_class_truth(), data, tolerance=1e-8, max_iterations=1000)
    runs = [em.run(seed=s) for s in (0, 2)]  # two starts that land on opposite labellings
    best = max(r.log_likelihood[-1] for r in runs)
    labels = set()
    for r in runs:
        assert r.converged
        assert r.log_likelihood[-1] == pytest.approx(best, abs=1e-4)
        assert class_permutation_distance(r.model, runs[0].model) < 1e-3
        # A loose sanity bound: 1500 rows with a hidden class pin parameters down only to ~0.05.
        assert class_permutation_distance(r.model, latent_class_truth()) < 0.1
        labels.add(int(np.argmax(r.model.cpds["C"].values)))
    assert labels == {0, 1}  # the two runs really did pick different labellings


# ---------------------------------------------------------------------------
# M6: recovery under MCAR, and better than discarding incomplete rows
# ---------------------------------------------------------------------------


def columns(model: BayesianNetwork):
    for name, cpd in model.cpds.items():
        for config in itertools.product(*(p.states for p in cpd.parents)):
            yield name, dict(zip(cpd.parent_names, config, strict=True))


@pytest.mark.parametrize("seed", range(2))
def test_em_recovers_the_late_network_under_mcar(seed):
    truth = late_network()
    data = sample(truth, 20_000, seed).with_missing(0.3, seed=seed)
    result = ExpectationMaximisation(truth, data, tolerance=1e-7).run(seed=seed)
    assert result.converged
    complete_rows = Dataset(truth.variables, [r for r in data.rows() if None not in r.values()])
    complete_case = maximum_likelihood(truth, complete_rows)

    em_sse = cc_sse = 0.0
    for name, given in columns(truth):
        true = truth.cpds[name].distribution(given)
        em_estimate = result.model.cpds[name].distribution(given)
        cc_sse += float(((complete_case.cpds[name].distribution(given) - true) ** 2).sum())
        em_sse += float(((em_estimate - true) ** 2).sum())
        # EM uses at least the rows in which the whole family is observed, so the
        # Bernstein bound for that many rows is a (conservative) yardstick.
        family = [name, *given]
        n = sum(
            1
            for r in data.rows()
            if all(r[v] is not None for v in family) and all(r[p] == s for p, s in given.items())
        )
        for est, p in zip(em_estimate, true, strict=True):
            assert abs(est - p) <= bernstein(float(p), n), (name, given, est, p, n)
    # Only 0.7⁵ ≈ 17% of rows are complete; EM uses the rest too.
    assert complete_rows.n_rows < 0.2 * data.n_rows
    assert em_sse < 0.5 * cc_sse


# ---------------------------------------------------------------------------
# em.md §2: MAR is what makes ignoring the missingness valid
# ---------------------------------------------------------------------------

RAIN = DiscreteVariable("Rain", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))


def rain_traffic() -> BayesianNetwork:
    model = BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])
    model.add_cpd(TabularCPD(RAIN, (), [0.6, 0.4]))
    model.add_cpd(TabularCPD(TRAFFIC, (RAIN,), [[0.85, 0.25], [0.15, 0.75]]))
    return model


TRUE_P_TRAFFIC = 0.6 * 0.15 + 0.4 * 0.75  # 0.39


def hide_traffic(depends_on: str) -> Dataset:
    """Hide Traffic with probability 0.8 or 0.1, depending on Rain (MAR) or Traffic (MNAR)."""
    rows = AncestralSampler(rain_traffic(), seed=1).sample(20_000)
    rng = np.random.default_rng(2)
    out = []
    for r in rows:
        p = 0.8 if r[depends_on] == "yes" else 0.1
        out.append({"Rain": r["Rain"], "Traffic": None if rng.random() < p else r["Traffic"]})
    return Dataset([RAIN, TRAFFIC], out)


def em_p_traffic(data: Dataset) -> float:
    model = ExpectationMaximisation(rain_traffic(), data, tolerance=1e-10).run(seed=0).model
    return sum(
        model.cpds["Rain"].probability(s, {})
        * model.cpds["Traffic"].probability("yes", {"Rain": s})
        for s in RAIN.states
    )


def naive_p_traffic(data: Dataset) -> float:
    observed = [r["Traffic"] for r in data.rows() if r["Traffic"] is not None]
    return sum(t == "yes" for t in observed) / len(observed)


def test_em_is_unbiased_under_mar_while_the_observed_average_is_not():
    data = hide_traffic("Rain")
    n_observed = data.n_rows - data.missing_count
    assert abs(em_p_traffic(data) - TRUE_P_TRAFFIC) <= bernstein(TRUE_P_TRAFFIC, n_observed)
    # Rainy days (more traffic) are mostly hidden, so the raw average is far too low.
    assert naive_p_traffic(data) < TRUE_P_TRAFFIC - 0.1


def test_em_is_biased_under_mnar():
    """When the missing value itself drives the missingness, P(R | o, z) ≠ P(R | o)."""
    data = hide_traffic("Traffic")
    assert em_p_traffic(data) < TRUE_P_TRAFFIC - 0.1
