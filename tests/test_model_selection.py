"""M4.8: BIC and structure scoring (spec §3.5; S1–S4; model_selection.md)."""

import itertools
import math

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable
from probgraph.exceptions import ValidationError
from probgraph.learning import (
    Dataset,
    DirichletPrior,
    bic,
    family_scores,
    log_likelihood,
    log_marginal_likelihood,
    maximum_likelihood,
    score_structures,
)

RAIN = DiscreteVariable("Rain", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))
Y, N = "yes", "no"


def f1() -> Dataset:
    rows = (
        [{"Rain": Y, "Traffic": Y}] * 3
        + [{"Rain": Y, "Traffic": N}] * 1
        + [{"Rain": N, "Traffic": Y}] * 1
        + [{"Rain": N, "Traffic": N}] * 5
    )
    return Dataset([RAIN, TRAFFIC], rows)


def sample(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


def with_edges(model: BayesianNetwork, edges) -> BayesianNetwork:
    return BayesianNetwork(model.variables, list(edges))


# ---------------------------------------------------------------------------
# F1 and S1
# ---------------------------------------------------------------------------


def test_f1_bic_values():
    dependent = BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])
    independent = BayesianNetwork([RAIN, TRAFFIC], [])
    assert bic(dependent, f1()) == pytest.approx(-15.136702, abs=5e-7)
    assert bic(independent, f1()) == pytest.approx(-15.762818, abs=5e-7)
    ranked = score_structures([independent, dependent], f1())
    assert [model for _, model in ranked] == [dependent, independent]


def parameter_count(model: BayesianNetwork) -> int:
    """d counted from scratch: (|X| - 1) per parent configuration, per variable."""
    total = 0
    for v in model.variables:
        configurations = 1
        for p in model.variables:
            if p.name in model.parents(v.name):
                configurations *= p.cardinality
        total += (v.cardinality - 1) * configurations
    return total


@pytest.mark.parametrize("seed", range(20))
def test_bic_is_the_penalised_mle_log_likelihood(seed):
    truth = random_network(seed, n_vars=(1, 5), cards=(1, 3))
    data = sample(truth, 500, seed)
    fitted = maximum_likelihood(truth, data, unseen="uniform")
    expected = log_likelihood(fitted, data) - parameter_count(truth) / 2 * math.log(500)
    assert bic(truth, data) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_unseen_parent_configurations_cost_parameters_but_no_likelihood():
    data = Dataset([RAIN, TRAFFIC], [{"Rain": N, "Traffic": Y}, {"Rain": N, "Traffic": N}])
    dependent = BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])
    # log L: Rain is "no" twice (log 1 = 0); Traffic | no is 1:1 (2 log 1/2). d = 3, N = 2.
    assert bic(dependent, data) == pytest.approx(2 * math.log(0.5) - 1.5 * math.log(2))


def test_bdeu_score_is_the_log_marginal_likelihood():
    truth = late_network()
    data = sample(truth, 300, seed=2)
    total = sum(family_scores(truth, data, "bdeu", equivalent_sample_size=5.0).values())
    assert total == pytest.approx(
        log_marginal_likelihood(truth, data, DirichletPrior.bdeu(5.0)), rel=1e-13
    )


# ---------------------------------------------------------------------------
# S2: decomposability
# ---------------------------------------------------------------------------


def ancestors(model: BayesianNetwork, name: str) -> set[str]:
    found: set[str] = set()
    stack = list(model.parents(name))
    while stack:
        node = stack.pop()
        if node not in found:
            found.add(node)
            stack.extend(model.parents(node))
    return found


def addable_edges(model: BayesianNetwork):
    """Edges whose addition keeps the graph acyclic (child not an ancestor of the parent)."""
    names = [v.name for v in model.variables]
    for a, b in itertools.permutations(names, 2):
        if (a, b) in model.edges() or (b, a) in model.edges():
            continue
        if b in ancestors(model, a):
            continue
        yield a, b


@pytest.mark.parametrize("score", ["bic", "bdeu"])
@pytest.mark.parametrize("seed", range(15))
def test_one_edge_changes_one_family(seed, score):
    truth = random_network(seed + 50, n_vars=(3, 5), cards=(2, 3), edge_prob=0.4)
    data = sample(truth, 200, seed)
    before = family_scores(truth, data, score)
    assert sum(before.values()) == pytest.approx(
        score_structures([truth], data, score)[0][0], rel=1e-13
    )
    edges = list(addable_edges(truth))
    assert edges
    a, b = edges[seed % len(edges)]
    after = family_scores(with_edges(truth, [*truth.edges(), (a, b)]), data, score)
    for name in before:
        if name == b:
            assert after[name] != before[name]
        else:
            assert after[name] == before[name]  # bit for bit: same counts, same arithmetic


# ---------------------------------------------------------------------------
# S4: score equivalence
# ---------------------------------------------------------------------------


def covered_edges(model: BayesianNetwork):
    for x, y in model.edges():
        if set(model.parents(y)) == set(model.parents(x)) | {x}:
            yield x, y


@pytest.mark.parametrize("seed", range(30))
def test_covered_edge_reversal_preserves_bic(seed):
    truth = random_network(seed + 500, n_vars=(2, 5), cards=(2, 3), edge_prob=0.6)
    data = sample(truth, 100, seed)
    for x, y in covered_edges(truth):
        reversed_model = with_edges(truth, [e for e in truth.edges() if e != (x, y)] + [(y, x)])
        assert parameter_count(reversed_model) == parameter_count(truth)  # Theorem 1
        assert bic(reversed_model, data) == pytest.approx(bic(truth, data), rel=1e-12)


def test_a_non_covered_reversal_changes_bic():
    """Traffic → Late is not covered (Traffic has parents Late lacks); reversing it breaks the
    v-structure Rain → Traffic ← Accident, so the class changes and so does the score."""
    truth = late_network()
    data = sample(truth, 2000, seed=1)
    flipped = with_edges(
        truth, [e for e in truth.edges() if e != ("Traffic", "Late")] + [("Late", "Traffic")]
    )
    assert abs(bic(flipped, data) - bic(truth, data)) > 1.0


# ---------------------------------------------------------------------------
# S3: consistency
# ---------------------------------------------------------------------------


def late_candidates() -> dict[str, BayesianNetwork]:
    truth = late_network()
    edges = list(truth.edges())
    candidates = {
        "truth": with_edges(truth, edges),
        "equivalent": with_edges(
            truth, [e for e in edges if e != ("Rain", "Umbrella")] + [("Umbrella", "Rain")]
        ),
    }
    for e in edges:
        candidates[f"-{e}"] = with_edges(truth, [x for x in edges if x != e])
    for e in [
        ("Accident", "Late"),
        ("Rain", "Late"),
        ("Umbrella", "Late"),
        ("Accident", "Umbrella"),
        ("Rain", "Accident"),
    ]:
        candidates[f"+{e}"] = with_edges(truth, [*edges, e])
    return candidates


@pytest.mark.parametrize("score", ["bic", "bdeu"])
def test_large_samples_rank_the_true_class_first(score):
    data = sample(late_network(), 50_000, seed=1)
    candidates = late_candidates()
    ranked = score_structures(candidates.values(), data, score)
    name_of = {id(model): name for name, model in candidates.items()}
    top = [name_of[id(model)] for _, model in ranked[:2]]
    assert top == ["truth", "equivalent"]  # a tie, kept in candidate order
    assert ranked[0][0] == pytest.approx(ranked[1][0], rel=1e-12)
    assert ranked[1][0] - ranked[2][0] > 2.0


def test_superset_penalty_matches_wilks():
    """BIC(truth) - BIC(superset) = (Δd/2) log N - (Wilks' χ²_Δd)/2 (model_selection.md §5)."""
    n = 50_000
    data = sample(late_network(), n, seed=1)
    truth_bic = bic(late_network(), data)
    for name, model in late_candidates().items():
        if not name.startswith("+"):
            continue
        extra = parameter_count(model) - parameter_count(late_network())
        assert extra >= 1
        half_chi_square = extra / 2 * math.log(n) - (truth_bic - bic(model, data))
        # χ²_Δd / 2 ≥ 0 (the likelihood can only rise) and, for Δd ≤ 2, below 10 with
        # probability > 1 - 2e-9.
        assert 0 <= half_chi_square < 10


def test_small_samples_prefer_sparser_structures():
    """At N = 50 the penalty dominates: the best structure has fewer parameters than the truth."""
    data = sample(late_network(), 50, seed=1)
    candidates = late_candidates()
    best_score, best = score_structures(candidates.values(), data, "bic")[0]
    assert parameter_count(best) < parameter_count(late_network())
    assert best_score > bic(late_network(), data)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation():
    model = BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])
    with pytest.raises(ValidationError, match="score must be"):
        family_scores(model, f1(), "aic")
    with pytest.raises(ValidationError, match="complete data"):
        bic(model, Dataset([RAIN, TRAFFIC], [{"Rain": Y, "Traffic": None}]))
    with pytest.raises(ValidationError, match="at least one row"):
        bic(model, Dataset([RAIN, TRAFFIC], []))
    with pytest.raises(ValidationError, match="at least one candidate"):
        score_structures([], f1())
    with pytest.raises(ValidationError, match="BayesianNetwork"):
        score_structures(["Rain -> Traffic"], f1())
    with pytest.raises(ValidationError, match="do not match"):
        bic(BayesianNetwork([RAIN], []), f1())
    with pytest.raises(ValidationError, match="equivalent_sample_size"):
        family_scores(model, f1(), "bdeu", equivalent_sample_size=0)


def test_bdeu_on_empty_data_is_zero():
    model = BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic")])
    scores = family_scores(model, Dataset([RAIN, TRAFFIC], []), "bdeu")
    assert scores == {"Rain": 0.0, "Traffic": 0.0}
    assert np.isfinite(list(scores.values())).all()
