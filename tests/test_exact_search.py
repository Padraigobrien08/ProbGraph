"""M7.3: exact structure search by dynamic programming (S3, S5; structure_search.md §4)."""

import math

import pytest
from support import late_network, random_network
from test_hill_climb import sample, total
from test_pdag import all_dags

from probgraph import DiscreteVariable
from probgraph.exceptions import ValidationError
from probgraph.learning import Dataset
from probgraph.structure import (
    cpdag,
    exact_search,
    hill_climb,
    structural_hamming_distance,
)
from probgraph.structure.search import FamilyScorer

DAGS = {n: list(all_dags(n)) for n in (3, 4, 5)}


def exhaustive(data: Dataset, score="bic", max_parents=None) -> float:
    """The maximum over every DAG, scored with a fresh family cache."""
    scorer = FamilyScorer(data, score, 1.0)
    names = scorer.names
    best = -math.inf
    for dag in DAGS[len(names)]:
        parents = {names[int(c)]: [names[int(p)] for p in dag.parents(c)] for c in dag.nodes()}
        if max_parents is not None and any(len(p) > max_parents for p in parents.values()):
            continue
        best = max(best, math.fsum(scorer(c, p) for c, p in parents.items()))
    return best


# ---------------------------------------------------------------------------
# S3: the DP equals exhaustive search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", ["bic", "bdeu"])
@pytest.mark.parametrize("seed", range(15))
def test_four_nodes_match_exhaustive_search(seed, score):
    truth = random_network(seed, n_vars=(4, 4), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 300, seed)
    result = exact_search(data, score)
    assert result.score == pytest.approx(exhaustive(data, score), abs=1e-9)
    assert result.score == pytest.approx(total(data, result.structure.edges(), score), abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_five_nodes_match_exhaustive_search(seed):
    truth = random_network(seed + 20, n_vars=(5, 5), cards=(2, 2), edge_prob=0.5)
    data = sample(truth, 400, seed)
    assert exact_search(data).score == pytest.approx(exhaustive(data), abs=1e-9)


@pytest.mark.parametrize("max_parents", [0, 1, 2])
@pytest.mark.parametrize("seed", range(6))
def test_max_parents(seed, max_parents):
    truth = random_network(seed + 40, n_vars=(4, 4), cards=(2, 2), edge_prob=0.8)
    data = sample(truth, 500, seed)
    result = exact_search(data, max_parents=max_parents)
    assert result.score == pytest.approx(exhaustive(data, max_parents=max_parents), abs=1e-9)
    assert all(len(result.structure.parents(v.name)) <= max_parents for v in data.variables)


@pytest.mark.parametrize("seed", range(20))
def test_never_worse_than_hill_climbing(seed):
    truth = random_network(seed + 60, n_vars=(3, 5), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 300, seed)
    assert exact_search(data).score >= hill_climb(data).score - 1e-9


def test_three_nodes_exhaustively_and_evaluation_count():
    truth = random_network(1, n_vars=(3, 3), cards=(2, 3), edge_prob=0.7)
    data = sample(truth, 200, 1)
    result = exact_search(data)
    assert result.score == pytest.approx(exhaustive(data), abs=1e-9)
    assert result.evaluations == 3 * 2**2  # every family of every variable, once
    assert result.history == (result.score,)


# ---------------------------------------------------------------------------
# S5: consistency: large samples give the true equivalence class
# ---------------------------------------------------------------------------


def test_late_network_is_recovered_up_to_equivalence():
    data = sample(late_network(), 20_000, seed=1)
    learned = exact_search(data).structure
    assert structural_hamming_distance(cpdag(learned.graph), cpdag(late_network().graph)) == 0


def test_small_samples_underfit_large_samples_recover():
    distances = []
    for n in (50, 500, 20_000):
        learned = exact_search(sample(late_network(), n, seed=2)).structure
        distances.append(
            structural_hamming_distance(cpdag(learned.graph), cpdag(late_network().graph))
        )
    assert distances[0] > 0 and distances[-1] == 0
    assert distances == sorted(distances, reverse=True)


@pytest.mark.parametrize("score", ["bic", "bdeu"])
def test_equivalent_optima_score_equally(score):
    """BIC and BDeu are score-equivalent, so every member of the optimal class is optimal."""
    data = sample(late_network(), 5000, seed=3)
    result = exact_search(data, score)
    for member in cpdag(result.structure.graph).members():
        assert total(data, member.edges(), score) == pytest.approx(result.score, abs=1e-8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_too_many_variables():
    variables = [DiscreteVariable(f"V{i}", ("a", "b")) for i in range(13)]
    data = Dataset(variables, [{v.name: "a" for v in variables}])
    with pytest.raises(ValidationError, match="at most 12"):
        exact_search(data)


def test_max_parents_validated():
    with pytest.raises(ValidationError, match="max_parents"):
        exact_search(sample(late_network(), 20, seed=0), max_parents=-1)
