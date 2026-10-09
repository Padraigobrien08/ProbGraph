"""M2.5: elimination heuristics, traces and the cost of an order (spec task 5; V2, V8).

Propositions 3 and 4 of docs/mathematics/elimination_orders.md are tested
exactly. On general graphs the heuristics are only compared against a
brute-force optimum, as a consistency check.
"""

import itertools

import numpy as np
import pytest
from support import late_network, posterior, random_evidence, random_network

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs import UndirectedGraph, interaction_graph
from probgraph.inference import (
    HEURISTICS,
    EliminationStep,
    EliminationTrace,
    greedy_order,
    simulate_elimination,
)

# ---------------------------------------------------------------------------
# Graph helpers (test-local, independent of the implementation)
# ---------------------------------------------------------------------------


def fill_in(graph: UndirectedGraph, order) -> int:
    """Count the fill edges that eliminating ``order`` adds."""
    g = graph.copy()
    added = 0
    for z in order:
        for u, v in itertools.combinations(sorted(g.neighbours(z)), 2):
            if not g.has_edge(u, v):
                g.add_edge(u, v)
                added += 1
        g.remove_node(z)
    return added


def width_of(graph: UndirectedGraph, order) -> int:
    return simulate_elimination(graph, order, {n: 2 for n in graph.nodes()}).width


def optimal_width(graph: UndirectedGraph) -> int:
    return min(width_of(graph, p) for p in itertools.permutations(graph.nodes()))


def max_clique_size(graph: UndirectedGraph) -> int:
    nodes = graph.nodes()
    for k in range(len(nodes), 0, -1):
        for subset in itertools.combinations(nodes, k):
            if all(graph.has_edge(u, v) for u, v in itertools.combinations(subset, 2)):
                return k
    return 0


def random_forest(rng: np.random.Generator, n: int) -> UndirectedGraph:
    nodes = [f"N{i}" for i in range(n)]
    g = UndirectedGraph(nodes)
    for i in range(1, n):
        if rng.random() < 0.85:  # otherwise start a new tree
            g.add_edge(nodes[i], nodes[int(rng.integers(i))])
    return g


def random_chordal(rng: np.random.Generator, n: int) -> UndirectedGraph:
    """Each new vertex attaches to a subset of an existing clique, so it is simplicial
    when added. Reversing the insertion order is a perfect elimination ordering, so the
    graph is chordal."""
    nodes = [f"N{i}" for i in range(n)]
    g = UndirectedGraph(nodes)
    cliques: list[list[str]] = [[nodes[0]]]
    for v in nodes[1:]:
        base = cliques[int(rng.integers(len(cliques)))]
        attach = [u for u in base if rng.random() < 0.7]
        for u in attach:
            g.add_edge(v, u)
        cliques.append([*attach, v])
    # Shuffle insertion order so that tie-breaking cannot simply replay the construction.
    shuffled = UndirectedGraph(list(rng.permutation(nodes)), g.edges())
    return shuffled


def random_graph(rng: np.random.Generator, n: int, p: float) -> UndirectedGraph:
    nodes = [f"N{i}" for i in range(n)]
    return UndirectedGraph(
        nodes, [(u, v) for u, v in itertools.combinations(nodes, 2) if rng.random() < p]
    )


# ---------------------------------------------------------------------------
# EliminationTrace and simulate_elimination
# ---------------------------------------------------------------------------


def test_trace_of_a_path():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B"), ("B", "C")])
    trace = simulate_elimination(g, ["A", "B"], {"A": 2, "B": 3, "C": 2})
    assert trace.steps == (
        EliminationStep("A", frozenset("AB"), 6),
        EliminationStep("B", frozenset("BC"), 6),
    )
    assert trace.order == ("A", "B")
    assert trace.width == 1
    assert trace.max_size == 6
    assert trace.total_cost == 12


def test_eliminating_the_centre_of_a_star_creates_a_clique():
    g = UndirectedGraph(nodes="CWXYZ", edges=[("C", leaf) for leaf in "WXYZ"])
    trace = simulate_elimination(g, list("CWXYZ"), dict.fromkeys("CWXYZ", 2))
    assert trace.steps[0].scope == frozenset("CWXYZ")
    assert trace.width == 4
    assert fill_in(g, "CWXYZ") == 6  # C(4, 2) fill edges among the leaves


def test_empty_trace():
    trace = simulate_elimination(UndirectedGraph(nodes="AB"), [], {"A": 2, "B": 2})
    assert trace.steps == () and trace.width == 0 and trace.max_size == 0 and trace.total_cost == 0


def test_simulation_does_not_mutate_the_graph():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B"), ("A", "C")])
    simulate_elimination(g, ["A"], dict.fromkeys("ABC", 2))
    assert g.nodes() == ("A", "B", "C") and not g.has_edge("B", "C")


@pytest.mark.parametrize(
    ("order", "error"),
    [(["A", "A"], ValidationError), (["Z"], UnknownNodeError), ("AB", ValidationError)],
)
def test_simulation_validates_order(order, error):
    g = UndirectedGraph(nodes="AB", edges=[("A", "B")])
    with pytest.raises(error):
        simulate_elimination(g, order, {"A": 2, "B": 2})


def test_simulation_requires_cardinalities():
    g = UndirectedGraph(nodes="AB", edges=[("A", "B")])
    with pytest.raises(ValidationError, match="cardinalit"):
        simulate_elimination(g, ["A"], {"A": 2})


# ---------------------------------------------------------------------------
# Greedy heuristics
# ---------------------------------------------------------------------------


def test_heuristic_names():
    assert HEURISTICS == ("min_fill", "min_neighbours", "min_weight")


def test_min_fill_and_min_neighbours_disagree():
    """P has 3 neighbours forming a clique (fill 0). Q has 2 non-adjacent neighbours (fill 1)."""
    g = UndirectedGraph(
        nodes=["Q", "P", "a", "b", "c", "x", "y"],
        edges=[
            *[("P", "a"), ("P", "b"), ("P", "c"), ("a", "b"), ("b", "c"), ("a", "c")],
            *[("Q", "x"), ("Q", "y")],
        ],
    )
    assert greedy_order(g, ["Q", "P"], "min_neighbours")[0] == "Q"
    assert greedy_order(g, ["Q", "P"], "min_fill")[0] == "P"


def test_min_weight_uses_cardinalities():
    """U has 1 neighbour of cardinality 5. V has 2 neighbours of cardinality 2 (weight 4)."""
    g = UndirectedGraph(
        nodes=["U", "V", "big", "s1", "s2"], edges=[("U", "big"), ("V", "s1"), ("V", "s2")]
    )
    cards = {"U": 2, "V": 2, "big": 5, "s1": 2, "s2": 2}
    assert greedy_order(g, ["U", "V"], "min_neighbours")[0] == "U"
    assert greedy_order(g, ["U", "V"], "min_weight", cards)[0] == "V"
    with pytest.raises(ValidationError, match="cardinalit"):
        greedy_order(g, ["U", "V"], "min_weight")


def test_ties_break_by_insertion_order():
    g = UndirectedGraph(nodes=["D", "B", "A", "C"])
    for h in HEURISTICS:
        assert greedy_order(g, ["A", "B", "C", "D"], h, dict.fromkeys("ABCD", 2)) == [
            "D",
            "B",
            "A",
            "C",
        ]


def test_greedy_order_validates_inputs():
    g = UndirectedGraph(nodes="AB", edges=[("A", "B")])
    with pytest.raises(ValidationError, match="heuristic"):
        greedy_order(g, ["A"], "min_size")  # type: ignore[arg-type]
    with pytest.raises(UnknownNodeError):
        greedy_order(g, ["Z"], "min_fill")
    with pytest.raises(ValidationError, match="duplicate"):
        greedy_order(g, ["A", "A"], "min_fill")


@pytest.mark.parametrize("seed", range(40))
def test_proposition_3_forests_have_width_at_most_one(seed):
    rng = np.random.default_rng(seed)
    g = random_forest(rng, int(rng.integers(1, 12)))
    for heuristic in ("min_fill", "min_neighbours"):
        order = greedy_order(g, g.nodes(), heuristic)
        assert sorted(order) == sorted(g.nodes())
        assert width_of(g, order) <= 1
        assert fill_in(g, order) == 0


@pytest.mark.parametrize("seed", range(40))
def test_proposition_4_min_fill_is_optimal_on_chordal_graphs(seed):
    rng = np.random.default_rng(seed)
    g = random_chordal(rng, int(rng.integers(1, 9)))
    order = greedy_order(g, g.nodes(), "min_fill")
    assert fill_in(g, order) == 0
    assert width_of(g, order) == max_clique_size(g) - 1


@pytest.mark.parametrize("seed", range(30))
def test_heuristics_never_beat_the_brute_force_optimum(seed):
    rng = np.random.default_rng(seed)
    g = random_graph(rng, int(rng.integers(2, 7)), p=0.45)
    best = optimal_width(g)
    cards = {n: int(rng.integers(2, 4)) for n in g.nodes()}
    for heuristic in HEURISTICS:
        order = greedy_order(g, g.nodes(), heuristic, cards)
        assert sorted(order) == sorted(g.nodes())
        assert width_of(g, order) >= best


# ---------------------------------------------------------------------------
# V8: measured traces equal predicted traces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_measured_trace_matches_graph_prediction(seed):
    model = random_network(seed, n_vars=(2, 6))
    rng = np.random.default_rng(seed + 30_000)
    evidence = random_evidence(model, rng)
    unobserved = [v.name for v in model.variables if v.name not in evidence]
    k = int(rng.integers(0, len(unobserved) + 1))
    order = [str(n) for n in rng.permutation(unobserved)[:k]]  # a partial order is allowed

    ve = VariableElimination(model)
    measured = ve.elimination_cost(order, evidence)
    graph = interaction_graph(phi.reduce(evidence) for phi in model.factors())
    cards = {v.name: v.cardinality for v in model.variables}
    assert measured == simulate_elimination(graph, order, cards)


def sparse_problem(seed: int):
    """A sparser, larger network with a full random order. These usually need fill edges.

    At most one variable is observed (in half the cases): heavier evidence deletes so
    many vertices that fill becomes rare (measured: 5/60 cases with ~half observed).
    """
    model = random_network(seed, n_vars=(5, 8), cards=(2, 3), edge_prob=0.4)
    rng = np.random.default_rng(seed + 50_000)
    evidence = {}
    if rng.random() < 0.5:
        v = model.variables[int(rng.integers(len(model.variables)))]
        evidence[v.name] = v.states[int(rng.integers(v.cardinality))]
    unobserved = [v.name for v in model.variables if v.name not in evidence]
    order = [str(n) for n in rng.permutation(unobserved)]
    graph = interaction_graph(phi.reduce(evidence) for phi in model.factors())
    return model, evidence, order, graph


@pytest.mark.parametrize("seed", range(60))
def test_measured_trace_matches_prediction_when_fill_is_needed(seed):
    model, evidence, order, graph = sparse_problem(seed)
    cards = {v.name: v.cardinality for v in model.variables}
    measured = VariableElimination(model).elimination_cost(order, evidence)
    assert measured == simulate_elimination(graph, order, cards)


def test_sparse_problems_exercise_fill():
    """Guards the power of the test above: most of its cases must actually create fill."""
    problems = [sparse_problem(s) for s in range(60)]
    with_fill = sum(fill_in(graph, order) > 0 for _, _, order, graph in problems)
    assert with_fill >= 30, with_fill


def replay_score(graph: UndirectedGraph, node: str, heuristic: str, cards) -> int:
    neighbours = graph.neighbours(node)
    if heuristic == "min_neighbours":
        return len(neighbours)
    if heuristic == "min_weight":
        return int(np.prod([cards[v] for v in neighbours]))
    return sum(1 for u, v in itertools.combinations(neighbours, 2) if not graph.has_edge(u, v))


@pytest.mark.parametrize("seed", range(40))
def test_each_greedy_choice_is_best_in_the_current_graph(seed):
    """Replay oracle: at every step the chosen vertex has the lowest score in the graph
    *including earlier fill*, with ties going to the earliest-inserted vertex."""
    rng = np.random.default_rng(seed + 60_000)
    g = random_graph(rng, int(rng.integers(3, 9)), p=0.3)
    cards = {n: int(rng.integers(2, 5)) for n in g.nodes()}
    rank = {n: i for i, n in enumerate(g.nodes())}
    eliminate = [n for n in g.nodes() if rng.random() < 0.8]
    for heuristic in HEURISTICS:
        order = greedy_order(g, eliminate, heuristic, cards)
        current = g.copy()
        remaining = list(eliminate)
        for chosen in order:
            best = min(
                remaining, key=lambda n: (replay_score(current, n, heuristic, cards), rank[n])
            )
            assert chosen == best, (heuristic, order)
            remaining.remove(chosen)
            for u, v in itertools.combinations(sorted(current.neighbours(chosen)), 2):
                current.add_edge(u, v)
            current.remove_node(chosen)


@pytest.mark.parametrize("seed", range(30))
def test_heuristic_orders_are_valid_and_exact(seed):
    model = random_network(seed, n_vars=(2, 6))
    rng = np.random.default_rng(seed + 40_000)
    names = [v.name for v in model.variables]
    query = [str(rng.choice(names))]
    evidence = random_evidence(model, rng, exclude=query)
    eliminable = sorted(n for n in names if n not in query and n not in evidence)
    ve = VariableElimination(model)
    expected = posterior(model, query, evidence)
    barren = ve.barren_variables(query, evidence)
    for heuristic in HEURISTICS:
        full = ve.elimination_order(query, evidence, heuristic, prune_barren=False)
        pruned = ve.elimination_order(query, evidence, heuristic)
        assert sorted(full) == eliminable
        assert sorted(pruned) == sorted(set(eliminable) - barren)
        assert ve.query(query, evidence, elimination_order=heuristic).allclose(expected, atol=1e-12)


def test_elimination_cost_validates_order():
    ve = VariableElimination(late_network())
    with pytest.raises(ValidationError, match="duplicate"):
        ve.elimination_cost(["Rain", "Rain"])
    with pytest.raises(UnknownNodeError):
        ve.elimination_cost(["Snow"])
    with pytest.raises(ValidationError, match="observed"):
        ve.elimination_cost(["Late"], {"Late": "yes"})
    with pytest.raises(ValidationError, match="string"):
        ve.elimination_cost("Rain")


# ---------------------------------------------------------------------------
# The naive-Bayes demonstration: order changes cost, never the answer
# ---------------------------------------------------------------------------


def naive_bayes(n_features: int) -> BayesianNetwork:
    rng = np.random.default_rng(n_features)
    c = DiscreteVariable("C", ("0", "1"))
    features = [DiscreteVariable(f"F{i}", ("0", "1")) for i in range(1, n_features + 1)]
    model = BayesianNetwork([c, *features], [("C", f.name) for f in features])
    model.add_cpd(TabularCPD(c, (), [0.4, 0.6]))
    for f in features:
        model.add_cpd(TabularCPD(f, (c,), rng.dirichlet([1.0, 1.0], size=2).T))
    return model


def test_order_changes_cost_but_not_the_answer():
    n = 10
    model = naive_bayes(n)
    ve = VariableElimination(model)
    good = ve.elimination_order(["F1"], prune_barren=False)  # min_fill on the full graph
    bad = ["C", *(f"F{i}" for i in range(2, n + 1))]

    assert good[-1] == "C"  # every other feature is a fill-free leaf
    good_cost, bad_cost = ve.elimination_cost(good), ve.elimination_cost(bad)
    assert (good_cost.width, good_cost.max_size) == (1, 4)
    assert (bad_cost.width, bad_cost.max_size) == (n, 2 ** (n + 1))
    # good: 9 leaves × 4 + 4 = 40 cells; bad: 2^11 + 2^10 + ... + 2^2 = 4092 cells.
    assert (good_cost.total_cost, bad_cost.total_cost) == (40, 4092)

    # The measured traces equal the graph predictions, including the C(10, 2) = 45 fill
    # edges that eliminating C first adds among the features.
    graph = interaction_graph(model.factors())
    cards = {v.name: 2 for v in model.variables}
    assert good_cost == simulate_elimination(graph, good, cards)
    assert bad_cost == simulate_elimination(graph, bad, cards)
    assert fill_in(graph, bad) == 45

    expected = posterior(model, ["F1"], {})
    assert ve.query(["F1"], elimination_order=good).allclose(expected, atol=1e-12)
    assert ve.query(["F1"], elimination_order=bad).allclose(expected, atol=1e-12)


def test_default_order_is_min_fill():
    ve = VariableElimination(naive_bayes(6))
    assert ve.elimination_order(["F1"]) == ve.elimination_order(["F1"], heuristic="min_fill")
    full = ve.elimination_order(["F1"], prune_barren=False)
    assert ve.elimination_cost(full).width == 1
    # With pruning (the default), F2..F6 are barren and only C is left to eliminate.
    assert ve.elimination_order(["F1"]) == ["C"]
    assert ve.query_trace(["F1"]).total_cost == 4


def test_unknown_heuristic_in_query_is_rejected():
    with pytest.raises(ValidationError, match="heuristic"):
        VariableElimination(late_network()).query(["Rain"], elimination_order="min_size")


def test_trace_is_a_value_object():
    trace = EliminationTrace((EliminationStep("A", frozenset("AB"), 4),))
    assert trace == EliminationTrace((EliminationStep("A", frozenset("AB"), 4),))
    with pytest.raises(AttributeError):
        trace.steps = ()  # type: ignore[misc]
