"""M4.5: the marginal likelihood, the Bayesian score (spec §3.3; B4, B5; dirichlet.md Part 2)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable
from probgraph.exceptions import ValidationError
from probgraph.learning import (
    Dataset,
    DirichletPrior,
    bayesian_estimate,
    log_likelihood,
    log_marginal_likelihood,
    maximum_likelihood,
)

RAIN = DiscreteVariable("Rain", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))
Y, N = "yes", "no"


def rain_to_traffic() -> BayesianNetwork:
    return BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])


def traffic_to_rain() -> BayesianNetwork:
    return BayesianNetwork([RAIN, TRAFFIC], [("Traffic", "Rain")])


def two_node_data(yy: int, yn: int, ny: int, nn: int) -> Dataset:
    rows = (
        [{"Rain": Y, "Traffic": Y}] * yy
        + [{"Rain": Y, "Traffic": N}] * yn
        + [{"Rain": N, "Traffic": Y}] * ny
        + [{"Rain": N, "Traffic": N}] * nn
    )
    return Dataset([RAIN, TRAFFIC], rows)


def sample(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


# ---------------------------------------------------------------------------
# Fixtures F5 and F1 (milestone-4.md §4)
# ---------------------------------------------------------------------------


def test_f5_bdeu_is_score_equivalent():
    data = two_node_data(5, 1, 2, 2)
    prior = DirichletPrior.bdeu(1.0)
    forward = log_marginal_likelihood(rain_to_traffic(), data, prior)
    backward = log_marginal_likelihood(traffic_to_rain(), data, prior)
    assert forward == pytest.approx(-16.5436551681, abs=1e-10)
    assert backward == pytest.approx(-16.5436551681, abs=1e-10)


def test_f5_k2_is_not_score_equivalent():
    data = two_node_data(5, 1, 2, 2)
    prior = DirichletPrior.uniform(1.0)
    assert log_marginal_likelihood(rain_to_traffic(), data, prior) == pytest.approx(
        -14.8838698035, abs=1e-10
    )
    assert log_marginal_likelihood(traffic_to_rain(), data, prior) == pytest.approx(
        -14.7942576448, abs=1e-10
    )


def test_f1_k2_exact_fraction():
    """Rain: 4 yes, 6 no. Traffic | yes: 3, 1. Traffic | no: 1, 5. All α = 1 (Theorem 2)."""
    f = math.factorial
    rain = Fraction(f(1) * f(4) * f(6), f(11))  # Γ(2)/Γ(12) · Γ(5)Γ(7)
    given_yes = Fraction(f(3) * f(1), f(5))  # Γ(2)/Γ(6) · Γ(4)Γ(2)
    given_no = Fraction(f(1) * f(5), f(7))  # Γ(2)/Γ(8) · Γ(2)Γ(6)
    exact = rain * given_yes * given_no
    assert exact == Fraction(1, 1_940_400)
    value = log_marginal_likelihood(
        rain_to_traffic(), two_node_data(3, 1, 1, 5), DirichletPrior.uniform()
    )
    assert value == pytest.approx(math.log(exact), abs=1e-12)


# ---------------------------------------------------------------------------
# Theorem 3: the product of sequential posterior predictives
# ---------------------------------------------------------------------------


def random_prior(structure: BayesianNetwork, rng: np.random.Generator) -> DirichletPrior:
    kind = int(rng.integers(3))
    if kind == 0:
        return DirichletPrior.uniform(float(rng.uniform(0.2, 3.0)))
    if kind == 1:
        return DirichletPrior.bdeu(float(rng.uniform(0.5, 10.0)))
    pseudocounts = {}
    for v in structure.variables:
        parents = [p for p in structure.variables if p.name in structure.parents(v.name)]
        shape = (v.cardinality, *(p.cardinality for p in parents))
        pseudocounts[v.name] = rng.uniform(0.1, 3.0, size=shape)
    return DirichletPrior.explicit(pseudocounts)


def sequential_log_score(structure: BayesianNetwork, rows, prior: DirichletPrior) -> float:
    """Σ_m log P(x^(m) | x^(1..m-1)), refitting the posterior mean on every prefix (§5, §9)."""
    total = 0.0
    for m, row in enumerate(rows):
        predictive = bayesian_estimate(structure, Dataset(structure.variables, rows[:m]), prior)
        total += log_likelihood(predictive, Dataset(structure.variables, [row]))
    return total


@pytest.mark.parametrize("seed", range(25))
def test_gamma_form_equals_sequential_prediction(seed):
    rng = np.random.default_rng(seed)
    truth = random_network(seed, n_vars=(1, 4), cards=(1, 3))
    rows = list(sample(truth, int(rng.integers(0, 25)), seed).rows())
    prior = random_prior(truth, rng)
    closed = log_marginal_likelihood(truth, Dataset(truth.variables, rows), prior)
    for _ in range(3):  # exchangeability: every ordering gives the same product
        order = rng.permutation(len(rows))
        shuffled = [rows[i] for i in order]
        assert sequential_log_score(truth, shuffled, prior) == pytest.approx(
            closed, rel=1e-10, abs=1e-10
        )


# ---------------------------------------------------------------------------
# Quadrature: integrate likelihood × prior directly for one two-state column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "alpha"),
    [((0, 0), (1.0, 1.0)), ((3, 1), (1.0, 1.0)), ((7, 2), (0.5, 2.0)), ((0, 4), (0.3, 1.5))],
)
def test_single_column_by_quadrature(counts, alpha):
    """∫ θ^a (1-θ)^b Beta(θ | α1, α2) dθ, with θ = s⁴ to tame the singularity at 0."""
    coin = DiscreteVariable("Coin", ("heads", "tails"))
    rows = [{"Coin": "heads"}] * counts[0] + [{"Coin": "tails"}] * counts[1]
    prior = DirichletPrior.explicit({"Coin": alpha})
    value = log_marginal_likelihood(BayesianNetwork([coin], []), Dataset([coin], rows), prior)

    a1, a2 = alpha
    log_beta = math.lgamma(a1) + math.lgamma(a2) - math.lgamma(a1 + a2)
    n = 400_000
    s = (np.arange(n) + 0.5) / n
    theta = s**4
    integrand = theta ** (counts[0] + a1 - 1) * (1 - theta) ** (counts[1] + a2 - 1) * 4 * s**3
    integral = float(integrand.sum() / n) / math.exp(log_beta)
    assert value == pytest.approx(math.log(integral), abs=1e-6)


# ---------------------------------------------------------------------------
# §10: summed over every dataset of size N, the marginal likelihood is 1
# ---------------------------------------------------------------------------


def all_datasets(variables, n_rows: int):
    cells = [
        dict(zip([v.name for v in variables], states, strict=True))
        for states in itertools.product(*(v.states for v in variables))
    ]
    for rows in itertools.product(cells, repeat=n_rows):
        yield Dataset(variables, list(rows))


@pytest.mark.parametrize(
    "prior", [DirichletPrior.uniform(1.0), DirichletPrior.uniform(0.4), DirichletPrior.bdeu(2.5)]
)
@pytest.mark.parametrize("structure", [rain_to_traffic, traffic_to_rain])
def test_sums_to_one_over_all_two_node_datasets(structure, prior):
    model = structure()
    total = sum(
        math.exp(log_marginal_likelihood(model, d, prior)) for d in all_datasets(model.variables, 3)
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_sums_to_one_with_parents_of_different_sizes():
    a = DiscreteVariable("A", ("a0", "a1", "a2"))
    b = DiscreteVariable("B", ("b0", "b1"))
    model = BayesianNetwork([a, b], [("B", "A")])
    prior = DirichletPrior.explicit({"A": [[0.5, 2.0], [1.0, 0.3], [1.5, 1.0]], "B": [0.7, 1.9]})
    total = sum(math.exp(log_marginal_likelihood(model, d, prior)) for d in all_datasets([a, b], 3))
    assert total == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# §11: score equivalence (B5)
# ---------------------------------------------------------------------------


def covered_edges(model: BayesianNetwork):
    """Edges X → Y with pa(Y) = pa(X) ∪ {X}: reversing one gives an equivalent DAG."""
    for x, y in model.edges():
        if set(model.parents(y)) == set(model.parents(x)) | {x}:
            yield x, y


def reverse(model: BayesianNetwork, edge) -> BayesianNetwork:
    x, y = edge
    edges = [e for e in model.edges() if e != edge] + [(y, x)]
    return BayesianNetwork(model.variables, edges)


def bde_prior(structure: BayesianNetwork, joint: np.ndarray, ess: float) -> DirichletPrior:
    """α(x | u) = ess · P'(x, u) for one joint P' over all variables (axes in declaration order)."""
    names = [v.name for v in structure.variables]
    pseudocounts = {}
    for v in structure.variables:
        family = [v.name] + [p for p in names if p in structure.parents(v.name)]
        others = tuple(i for i, name in enumerate(names) if name not in family)
        margin = joint.sum(axis=others)
        kept = [name for name in names if name in family]  # margin's axes, in declaration order
        pseudocounts[v.name] = ess * np.transpose(margin, [kept.index(f) for f in family])
    return DirichletPrior.explicit(pseudocounts)


@pytest.mark.parametrize("seed", range(30))
def test_covered_edge_reversal_preserves_bdeu_and_bde(seed):
    rng = np.random.default_rng(seed)
    truth = random_network(seed + 500, n_vars=(2, 5), cards=(2, 3), edge_prob=0.6)
    data = sample(truth, 40, seed)
    ess = float(rng.uniform(0.5, 8.0))
    joint = rng.dirichlet(np.ones(math.prod(v.cardinality for v in truth.variables)))
    joint = joint.reshape([v.cardinality for v in truth.variables])
    for edge in covered_edges(truth):
        reversed_model = reverse(truth, edge)
        assert log_marginal_likelihood(truth, data, DirichletPrior.bdeu(ess)) == pytest.approx(
            log_marginal_likelihood(reversed_model, data, DirichletPrior.bdeu(ess)), rel=1e-11
        )
        assert log_marginal_likelihood(truth, data, bde_prior(truth, joint, ess)) == pytest.approx(
            log_marginal_likelihood(reversed_model, data, bde_prior(reversed_model, joint, ess)),
            rel=1e-11,
        )


def test_covered_edge_reversals_are_actually_exercised():
    count = sum(
        len(list(covered_edges(random_network(s + 500, (2, 5), (2, 3), 0.6)))) for s in range(30)
    )
    assert count >= 20


def test_whole_three_node_equivalence_class_ties_under_bdeu():
    a, b, c = (DiscreteVariable(n, ("0", "1", "2")) for n in "ABC")
    truth = random_network(7, n_vars=(3, 3), cards=(3, 3), edge_prob=1.0)
    data = sample(truth, 60, seed=1)
    renamed = Dataset(
        [a, b, c],
        [{n: r[f"V{i}"].replace("s", "") for i, n in enumerate("ABC")} for r in data.rows()],
    )
    prior = DirichletPrior.bdeu(3.0)

    def score(edges):
        return log_marginal_likelihood(BayesianNetwork([a, b, c], edges), renamed, prior)

    chain = score([("A", "B"), ("B", "C")])
    assert score([("C", "B"), ("B", "A")]) == pytest.approx(chain, rel=1e-12)
    assert score([("B", "A"), ("B", "C")]) == pytest.approx(chain, rel=1e-12)
    # A v-structure has a different equivalence class, and BDeu tells them apart.
    assert abs(score([("A", "B"), ("C", "B")]) - chain) > 1e-3


# ---------------------------------------------------------------------------
# Behaviour: Occam's razor, and agreement with BIC to O(1) (Laplace)
# ---------------------------------------------------------------------------


def test_the_score_prefers_the_structure_that_generated_the_data():
    rng = np.random.default_rng(0)
    prior = DirichletPrior.bdeu(1.0)
    empty = BayesianNetwork([RAIN, TRAFFIC], [])
    # Independent coins: the extra edge buys no fit and costs prior mass.
    rows = [
        {"Rain": Y if rng.random() < 0.4 else N, "Traffic": Y if rng.random() < 0.3 else N}
        for _ in range(2000)
    ]
    data = Dataset([RAIN, TRAFFIC], rows)
    assert log_marginal_likelihood(empty, data, prior) > log_marginal_likelihood(
        rain_to_traffic(), data, prior
    )
    # Dependent data: the edge is worth its parameters.
    dependent = two_node_data(300, 100, 100, 500)
    assert log_marginal_likelihood(rain_to_traffic(), dependent, prior) > log_marginal_likelihood(
        empty, dependent, prior
    )


def test_log_marginal_likelihood_tracks_bic_to_constant_order():
    """log P(D | G) = log L(θ̂) - (d/2) log N + O(1) (P17). The penalty grows by ~23 here."""
    truth = late_network()
    d = truth.n_free_parameters
    gaps = []
    for n in (1_000, 10_000, 100_000):
        data = sample(truth, n, seed=5)
        bic = log_likelihood(maximum_likelihood(truth, data), data) - d / 2 * math.log(n)
        gaps.append(log_marginal_likelihood(truth, data, DirichletPrior.uniform()) - bic)
    assert d / 2 * math.log(100) > 20
    assert max(gaps) - min(gaps) < 1.0


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


def test_empty_data_scores_zero():
    assert (
        log_marginal_likelihood(
            rain_to_traffic(), Dataset([RAIN, TRAFFIC], []), DirichletPrior.bdeu(1)
        )
        == 0.0
    )


def test_unseen_columns_contribute_nothing():
    data = two_node_data(0, 0, 2, 3)  # Traffic | Rain=yes is never observed
    with_edge = log_marginal_likelihood(rain_to_traffic(), data, DirichletPrior.uniform())
    rain_only = log_marginal_likelihood(
        BayesianNetwork([RAIN], []), Dataset([RAIN], [{"Rain": N}] * 5), DirichletPrior.uniform()
    )
    traffic_given_no = math.lgamma(2) - math.lgamma(7) + math.lgamma(3) + math.lgamma(4)
    assert with_edge == pytest.approx(rain_only + traffic_given_no, abs=1e-12)


def test_rejects_incomplete_data():
    data = Dataset([RAIN, TRAFFIC], [{"Rain": Y, "Traffic": None}])
    with pytest.raises(ValidationError, match="complete data"):
        log_marginal_likelihood(rain_to_traffic(), data, DirichletPrior.uniform())


def test_rejects_mismatched_variables():
    with pytest.raises(ValidationError, match="do not match"):
        log_marginal_likelihood(
            rain_to_traffic(), Dataset([RAIN], [{"Rain": Y}]), DirichletPrior.uniform()
        )
