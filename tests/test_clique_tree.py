"""M3.5: CliqueTree from an elimination order (spec task 5; J1–J3, J5, J6; clique_trees.md Part 2).

Independent checks: RIP by brute force, the closed form Σ|C_i| − n (Proposition 9),
and a Kruskal maximum-weight spanning tree computed here (the corollary).
"""

import itertools

import numpy as np
import pytest
from support import late_network, misconception_factors, random_network

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs import (
    UndirectedGraph,
    interaction_graph,
    maximal_cliques,
    moral_graph,
    triangulate,
)
from probgraph.inference import CliqueTree, greedy_order, simulate_elimination

# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def is_tree(k: int, edges) -> bool:
    if k == 0:
        return not edges
    if len(edges) != k - 1:
        return False
    adjacent = {i: set() for i in range(k)}
    for i, j in edges:
        adjacent[i].add(j)
        adjacent[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        for j in adjacent[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == k


def has_rip(cliques, edges) -> bool:
    """For every vertex, the clusters containing it are connected using only such clusters."""
    for x in set().union(*cliques) if cliques else set():
        holding = {i for i, c in enumerate(cliques) if x in c}
        start = min(holding)
        seen, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for a, b in edges:
                for j in (b,) if a == i else (a,) if b == i else ():
                    if j in holding and j not in seen:
                        seen.add(j)
                        stack.append(j)
        if seen != holding:
            return False
    return True


def kruskal_max_weight(cliques) -> int:
    """Weight of a maximum spanning tree of the clique graph (weights |C_i ∩ C_j|)."""
    parent = list(range(len(cliques)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = sorted(
        (
            (len(cliques[i] & cliques[j]), i, j)
            for i, j in itertools.combinations(range(len(cliques)), 2)
        ),
        reverse=True,
    )
    total = 0
    for w, i, j in pairs:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            total += w
    return total


def weight(tree: CliqueTree) -> int:
    return sum(len(tree.separator(i, j)) for i, j in tree.edges)


def check_junction_tree(graph: UndirectedGraph, order, tree: CliqueTree) -> None:
    cliques, edges = list(tree.cliques), list(tree.edges)
    assert is_tree(len(cliques), edges)  # J1
    assert has_rip(cliques, edges)  # J2
    triangulated = triangulate(graph, order)
    assert set(cliques) == set(maximal_cliques(triangulated))  # J3
    assert len(set(cliques)) == len(cliques)
    assert all(any({u, v} <= c for c in cliques) for u, v in graph.edges())
    if graph.nodes():
        assert weight(tree) == sum(len(c) for c in cliques) - len(graph)  # Proposition 9
        assert weight(tree) == kruskal_max_weight(cliques)  # Jensen & Jensen
    # J5: the width matches M2's simulation of the same order.
    cards = dict.fromkeys(graph.nodes(), 2)
    assert tree.width == simulate_elimination(graph, order, cards).width


# ---------------------------------------------------------------------------
# Fixtures (spec §4)
# ---------------------------------------------------------------------------


def test_late_network_clique_tree():
    model = late_network()
    graph = moral_graph(model.graph)
    order = greedy_order(graph, graph.nodes(), "min_fill")
    tree = CliqueTree.from_elimination(graph, order)
    expected = {
        frozenset({"Accident", "Rain", "Traffic"}),
        frozenset({"Late", "Traffic"}),
        frozenset({"Rain", "Umbrella"}),
    }
    assert set(tree.cliques) == expected
    separators = {tree.separator(i, j) for i, j in tree.edges}
    assert separators == {frozenset({"Traffic"}), frozenset({"Rain"})}
    check_junction_tree(graph, order, tree)


def test_misconception_clique_tree():
    graph = interaction_graph(misconception_factors())
    tree = CliqueTree.from_elimination(graph, list("ABCD"))
    assert set(tree.cliques) == {frozenset("ABD"), frozenset("BCD")}
    assert [tree.separator(i, j) for i, j in tree.edges] == [frozenset("BD")]
    assert tree.width == 2
    check_junction_tree(graph, list("ABCD"), tree)


# ---------------------------------------------------------------------------
# Random graphs and orders
# ---------------------------------------------------------------------------


def random_graph(rng, n: int, p: float) -> UndirectedGraph:
    nodes = [f"N{i}" for i in range(n)]
    return UndirectedGraph(
        nodes, [(u, v) for u, v in itertools.combinations(nodes, 2) if rng.random() < p]
    )


@pytest.mark.parametrize("seed", range(120))
def test_random_orders_give_junction_trees(seed):
    rng = np.random.default_rng(seed)
    g = random_graph(rng, int(rng.integers(1, 10)), p=float(rng.uniform(0.1, 0.6)))
    order = [str(v) for v in rng.permutation(list(g.nodes()))]
    check_junction_tree(g, order, CliqueTree.from_elimination(g, order))


@pytest.mark.parametrize("seed", range(40))
def test_heuristic_orders_on_moral_graphs(seed):
    model = random_network(seed, n_vars=(2, 9), edge_prob=0.35)
    graph = moral_graph(model.graph)
    for heuristic in ("min_fill", "min_neighbours"):
        order = greedy_order(graph, graph.nodes(), heuristic)
        check_junction_tree(graph, order, CliqueTree.from_elimination(graph, order))


def test_disconnected_graph_is_joined_by_empty_separators():
    g = UndirectedGraph("ABCDE", [("A", "B"), ("C", "D")])
    tree = CliqueTree.from_elimination(g, list("ABCDE"))
    assert set(tree.cliques) == {frozenset("AB"), frozenset("CD"), frozenset("E")}
    assert sorted(len(tree.separator(i, j)) for i, j in tree.edges) == [0, 0]
    check_junction_tree(g, list("ABCDE"), tree)


def test_empty_and_single_vertex_graphs():
    empty = CliqueTree.from_elimination(UndirectedGraph(), [])
    assert empty.cliques == () and empty.edges == ()
    single = CliqueTree.from_elimination(UndirectedGraph("A"), ["A"])
    assert single.cliques == (frozenset("A"),) and single.edges == () and single.width == 0


# ---------------------------------------------------------------------------
# J6: family preservation; API details
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_every_factor_scope_is_assigned_to_a_containing_clique(seed):
    model = random_network(seed, n_vars=(2, 8), edge_prob=0.4)
    factors = model.factors()
    graph = interaction_graph(factors)
    tree = CliqueTree.from_elimination(graph, greedy_order(graph, graph.nodes(), "min_fill"))
    assignment = tree.assign([phi.names for phi in factors])
    assert len(assignment) == len(factors)
    for phi, i in zip(factors, assignment, strict=True):
        assert set(phi.names) <= tree.cliques[i]


def test_assign_rejects_scopes_no_clique_contains():
    tree = CliqueTree.from_elimination(
        UndirectedGraph("ABC", [("A", "B"), ("B", "C")]), list("ABC")
    )
    with pytest.raises(ValidationError, match="no clique"):
        tree.assign([["A", "C"]])


def test_neighbours_and_separator_api():
    g = UndirectedGraph("ABCD", [("A", "B"), ("B", "C"), ("C", "D")])  # a path
    tree = CliqueTree.from_elimination(g, list("ABCD"))
    assert len(tree.cliques) == 3
    for i, j in tree.edges:
        assert j in tree.neighbours(i) and i in tree.neighbours(j)
        assert tree.separator(i, j) == tree.separator(j, i) == tree.cliques[i] & tree.cliques[j]
    with pytest.raises(ValidationError, match="not adjacent"):
        non_adjacent = next(
            (i, j) for i, j in itertools.combinations(range(3), 2) if (i, j) not in tree.edges
        )
        tree.separator(*non_adjacent)


@pytest.mark.parametrize(
    ("order", "error"),
    [(["A", "B"], ValidationError), (["A", "B", "Z"], UnknownNodeError), ("ABC", ValidationError)],
)
def test_order_validation(order, error):
    with pytest.raises(error):
        CliqueTree.from_elimination(UndirectedGraph("ABC", [("A", "B")]), order)


def test_construction_is_deterministic():
    rng = np.random.default_rng(11)
    g = random_graph(rng, 9, 0.35)
    order = [str(v) for v in rng.permutation(list(g.nodes()))]
    a, b = CliqueTree.from_elimination(g, order), CliqueTree.from_elimination(g.copy(), order)
    assert (a.cliques, a.edges) == (b.cliques, b.edges)
