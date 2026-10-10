"""M7.2: hill climbing (S1, S2, S4, S6; structure_search.md §2–3)."""

import itertools
import math

import pytest
from support import late_network, random_network
from test_pdag import all_dags

from probgraph import AncestralSampler, BayesianNetwork
from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.learning import Dataset, family_scores
from probgraph.structure import SearchResult, hill_climb
from probgraph.structure.search import FamilyScorer, _Constraints, _delta, _Graph, _legal_moves


def sample(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


def total(data: Dataset, edges, score="bic") -> float:
    """The full score by M4's family_scores: independent of the search's cache."""
    graph = BayesianNetwork(data.variables, list(edges))
    return math.fsum(family_scores(graph, data, score).values())


def single_moves(variables, edges):
    """Every DAG one add, delete or reverse away, built directly."""
    names = [v.name for v in variables]
    edges = set(edges)
    for a, b in itertools.permutations(names, 2):
        if (a, b) in edges:
            candidates = [edges - {(a, b)}, (edges - {(a, b)}) | {(b, a)}]
        elif (b, a) not in edges:
            candidates = [edges | {(a, b)}]
        else:
            candidates = []
        for c in candidates:
            try:
                BayesianNetwork(variables, sorted(c))
            except ValidationError:
                continue
            yield c


# ---------------------------------------------------------------------------
# S1: deltas equal full rescoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", ["bic", "bdeu"])
@pytest.mark.parametrize("seed", range(12))
def test_every_delta_equals_rescoring(seed, score):
    truth = random_network(seed, n_vars=(3, 5), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 300, seed)
    scorer = FamilyScorer(data, score, 1.0)
    graph = _Graph(scorer.names, truth.edges())
    rules = _Constraints(scorer.names, None, (), ())
    before = total(data, truth.edges(), score)
    moves = _legal_moves(graph, rules)
    assert moves
    for move in moves:
        kind, a, b = move
        edges = set(truth.edges())
        if kind == "add":
            edges.add((a, b))
        elif kind == "delete":
            edges.discard((a, b))
        else:
            edges.discard((a, b))
            edges.add((b, a))
        assert _delta(graph, move, scorer) == pytest.approx(
            total(data, edges, score) - before, abs=1e-9
        )


# ---------------------------------------------------------------------------
# S2: plain hill climbing strictly improves and ends at a verified local optimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", ["bic", "bdeu"])
@pytest.mark.parametrize("seed", range(15))
def test_plain_hill_climbing_ends_at_a_local_optimum(seed, score):
    truth = random_network(seed + 30, n_vars=(3, 5), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 400, seed)
    result = hill_climb(data, score)
    assert isinstance(result, SearchResult)
    assert all(b > a for a, b in itertools.pairwise(result.history))
    assert result.score == pytest.approx(result.history[-1], abs=1e-9)
    assert result.score == pytest.approx(total(data, result.structure.edges(), score), abs=1e-9)
    for neighbour in single_moves(data.variables, result.structure.edges()):
        assert total(data, neighbour, score) <= result.score + 1e-9


def test_cache_bounds_the_evaluations():
    data = sample(late_network(), 2000, seed=1)
    result = hill_climb(data)
    n = len(data.variables)
    assert 0 < result.evaluations <= n * 2 ** (n - 1)


# ---------------------------------------------------------------------------
# S4 / F5: greedy traps, and what tabu and restarts do about them
# ---------------------------------------------------------------------------


def exhaustive_best(data: Dataset) -> float:
    names = [v.name for v in data.variables]
    best = -math.inf
    for dag in all_dags(len(names)):
        edges = [(names[int(a)], names[int(b)]) for a, b in dag.edges()]
        best = max(best, total(data, edges))
    return best


def test_greedy_traps_are_common_and_escapes_reduce_them():
    plain_trapped = escaped_trapped = 0
    for seed in range(30):
        truth = random_network(seed, n_vars=(4, 4), cards=(2, 3), edge_prob=0.6)
        data = sample(truth, 300, seed)
        best = exhaustive_best(data)
        plain = hill_climb(data).score
        escaped = hill_climb(data, tabu_length=10, restarts=5, seed=seed).score
        assert plain <= best + 1e-9 and escaped <= best + 1e-9
        assert escaped >= plain - 1e-9  # escapes never return worse than plain greedy
        plain_trapped += plain < best - 1e-6
        escaped_trapped += escaped < best - 1e-6
    assert plain_trapped >= 4  # F5: plain greedy is trapped in a good share of problems
    assert escaped_trapped < plain_trapped
    assert escaped_trapped <= 2


def test_tabu_returns_the_best_graph_seen():
    data = sample(late_network(), 1000, seed=2)
    result = hill_climb(data, tabu_length=5)
    assert result.score == pytest.approx(max(result.history), abs=1e-9)
    assert result.score == pytest.approx(total(data, result.structure.edges()), abs=1e-9)


def test_restarts_are_reproducible():
    data = sample(late_network(), 500, seed=3)
    a = hill_climb(data, restarts=3, seed=7)
    b = hill_climb(data, restarts=3, seed=7)
    assert a.score == b.score and set(a.structure.edges()) == set(b.structure.edges())
    assert a.history == b.history


# ---------------------------------------------------------------------------
# S6: constraints
# ---------------------------------------------------------------------------


def test_constraints_are_respected():
    data = sample(late_network(), 2000, seed=4)
    required = [("Umbrella", "Rain")]
    forbidden = [("Rain", "Traffic"), ("Traffic", "Rain")]
    result = hill_climb(
        data,
        max_parents=1,
        required=required,
        forbidden=forbidden,
        tabu_length=4,
        restarts=2,
        seed=0,
    )
    edges = set(result.structure.edges())
    assert ("Umbrella", "Rain") in edges
    assert not edges & set(forbidden)
    assert all(len(result.structure.parents(v.name)) <= 1 for v in data.variables)


def test_start_graph_is_used():
    data = sample(late_network(), 300, seed=5)
    start = BayesianNetwork(data.variables, [("Late", "Traffic")])
    result = hill_climb(data, start=start, max_iterations=0)
    assert set(result.structure.edges()) == {("Late", "Traffic")}
    assert result.history == (pytest.approx(total(data, [("Late", "Traffic")])),)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"score": "aic"}, ValidationError, "score"),
        ({"max_parents": -1}, ValidationError, "max_parents"),
        ({"tabu_length": -1}, ValidationError, "tabu_length"),
        ({"required": [("Rain", "Snow")]}, UnknownNodeError, "Snow"),
        ({"required": [("Rain", "Rain")]}, ValidationError, "self-loop"),
        (
            {"required": [("Rain", "Late")], "forbidden": [("Rain", "Late")]},
            ValidationError,
            "both required and forbidden",
        ),
        ({"required": [("Rain", "Late"), ("Late", "Rain")]}, ValidationError, "invalid"),
        ({"equivalent_sample_size": 0, "score": "bdeu"}, ValidationError, "equivalent_sample_size"),
    ],
)
def test_validation(kwargs, error, match):
    data = sample(late_network(), 50, seed=6)
    with pytest.raises(error, match=match):
        hill_climb(data, **kwargs)


def test_needs_complete_data():
    data = sample(late_network(), 50, seed=6).with_missing(0.1, seed=1)
    with pytest.raises(ValidationError, match="complete data"):
        hill_climb(data)


def test_restarts_alone_escape_traps():
    """Restarts without tabu: 6 of these 30 problems trap plain greedy; 10 restarts leave 2."""
    plain_trapped = restarted_trapped = 0
    for seed in range(30):
        truth = random_network(seed, n_vars=(4, 4), cards=(2, 3), edge_prob=0.6)
        data = sample(truth, 300, seed)
        best = exhaustive_best(data)
        plain_trapped += hill_climb(data).score < best - 1e-6
        restarted_trapped += hill_climb(data, restarts=10, seed=seed).score < best - 1e-6
    assert restarted_trapped < plain_trapped
    assert restarted_trapped <= 3


@pytest.mark.parametrize("seed", range(10))
def test_each_step_takes_the_best_move(seed):
    """Steepest ascent: the first step's gain equals the best single-move gain from the start."""
    truth = random_network(seed + 60, n_vars=(3, 5), cards=(2, 3), edge_prob=0.5)
    data = sample(truth, 400, seed)
    result = hill_climb(data, max_iterations=1)
    start = total(data, [])
    best_neighbour = max(total(data, n) for n in single_moves(data.variables, []))
    if best_neighbour > start + 1e-9:
        assert result.history[1] - result.history[0] == pytest.approx(
            best_neighbour - start, abs=1e-9
        )
