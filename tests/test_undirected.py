"""M2.5: UndirectedGraph, moralisation and interaction graphs (spec §3.2).

Proposition 1 of docs/mathematics/elimination_orders.md is checked here: the
interaction graph of the CPD factors is the moral graph, and evidence removes
the observed vertices.
"""

import itertools

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import DAG, DiscreteFactor, DiscreteVariable
from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs import UndirectedGraph, interaction_graph, moral_graph


def edge_set(graph: UndirectedGraph) -> set[frozenset[str]]:
    return {frozenset(e) for e in graph.edges()}


# ---------------------------------------------------------------------------
# UndirectedGraph basics
# ---------------------------------------------------------------------------


def test_construction_and_queries():
    g = UndirectedGraph(nodes="ABCD", edges=[("A", "B"), ("B", "C")])
    assert g.nodes() == ("A", "B", "C", "D")
    assert g.edges() == (("A", "B"), ("B", "C"))
    assert g.neighbours("B") == {"A", "C"}
    assert g.neighbours("D") == set()
    assert g.has_edge("B", "A") and not g.has_edge("A", "C")
    assert "A" in g and "Z" not in g and len(g) == 4


def test_edges_are_symmetric_and_idempotent():
    g = UndirectedGraph(nodes="AB")
    g.add_edge("A", "B")
    g.add_edge("B", "A")
    assert g.edges() == (("A", "B"),)
    assert g.neighbours("A") == {"B"} and g.neighbours("B") == {"A"}


def test_self_loops_are_rejected():
    with pytest.raises(ValidationError, match="Self-loop"):
        UndirectedGraph(nodes="A", edges=[("A", "A")])


@pytest.mark.parametrize(
    "operation",
    [
        lambda g: g.add_edge("A", "Z"),
        lambda g: g.remove_edge("Z", "A"),
        lambda g: g.remove_node("Z"),
        lambda g: g.neighbours("Z"),
    ],
)
def test_unknown_nodes_are_rejected(operation):
    g = UndirectedGraph(nodes="AB", edges=[("A", "B")])
    with pytest.raises(UnknownNodeError):
        operation(g)


@pytest.mark.parametrize("bad", ["", None, 3])
def test_invalid_node_keys_are_rejected(bad):
    with pytest.raises(ValidationError):
        UndirectedGraph().add_node(bad)


def test_remove_node_removes_incident_edges():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B"), ("B", "C"), ("A", "C")])
    g.remove_node("B")
    assert g.nodes() == ("A", "C")
    assert edge_set(g) == {frozenset("AC")}
    assert g.neighbours("A") == {"C"}


def test_remove_missing_edge_is_noop():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B")])
    g.remove_edge("A", "C")
    assert g.edges() == (("A", "B"),)
    g.remove_edge("B", "A")
    assert g.edges() == ()


def test_copy_is_independent():
    g = UndirectedGraph(nodes="AB", edges=[("A", "B")])
    h = g.copy()
    h.add_node("C")
    h.remove_edge("A", "B")
    assert g.nodes() == ("A", "B") and g.edges() == (("A", "B"),)


def test_neighbours_returns_a_copy():
    g = UndirectedGraph(nodes="AB", edges=[("A", "B")])
    g.neighbours("A").clear()
    assert g.neighbours("A") == {"B"}


def test_equality_ignores_order():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B"), ("B", "C")])
    h = UndirectedGraph(nodes="CBA", edges=[("C", "B"), ("B", "A")])
    assert g == h
    h.add_edge("A", "C")
    assert g != h


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------


def test_separation_on_a_path():
    g = UndirectedGraph(nodes="ABCD", edges=[("A", "B"), ("B", "C"), ("C", "D")])
    assert not g.separated({"A"}, {"D"})
    assert g.separated({"A"}, {"D"}, given={"B"})
    assert g.separated({"A"}, {"D"}, given={"C"})
    assert not g.separated({"A", "C"}, {"D"}, given={"B"})


def test_separation_of_disconnected_components():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B")])
    assert g.separated({"A"}, {"C"})


def test_separation_requires_disjoint_known_sets():
    g = UndirectedGraph(nodes="ABC", edges=[("A", "B")])
    with pytest.raises(ValidationError, match="disjoint"):
        g.separated({"A"}, {"A", "B"})
    with pytest.raises(ValidationError, match="disjoint"):
        g.separated({"A"}, {"B"}, given={"A"})
    with pytest.raises(UnknownNodeError):
        g.separated({"A"}, {"Z"})
    with pytest.raises(ValidationError, match="empty"):
        g.separated(set(), {"B"})


@pytest.mark.parametrize("seed", range(30))
def test_separation_matches_reachability_oracle(seed):
    """Oracle: X and Y are separated by Z iff no X–Y path exists after deleting Z."""
    rng = np.random.default_rng(seed)
    nodes = [f"N{i}" for i in range(6)]
    edges = [(u, v) for u, v in itertools.combinations(nodes, 2) if rng.random() < 0.35]
    g = UndirectedGraph(nodes, edges)
    perm = list(rng.permutation(nodes))
    xs, ys, zs = {perm[0]}, {perm[1]}, set(perm[2 : 2 + int(rng.integers(0, 4))])

    # Independent check: connected components via repeated matrix squaring on G - Z.
    keep = [n for n in nodes if n not in zs]
    idx = {n: i for i, n in enumerate(keep)}
    a = np.eye(len(keep), dtype=bool)
    for u, v in edges:
        if u in idx and v in idx:
            a[idx[u], idx[v]] = a[idx[v], idx[u]] = True
    for _ in range(len(keep)):
        a = (a.astype(int) @ a.astype(int)) > 0
    connected = any(a[idx[x], idx[y]] for x in xs for y in ys)
    assert g.separated(xs, ys, zs) is (not connected)


# ---------------------------------------------------------------------------
# Moral graph and interaction graph (Proposition 1)
# ---------------------------------------------------------------------------


def test_moral_graph_marries_parents():
    dag = DAG(nodes="RATLU", edges=[("R", "T"), ("A", "T"), ("T", "L"), ("R", "U")])
    m = moral_graph(dag)
    assert m.nodes() == dag.nodes()
    assert edge_set(m) == {frozenset(e) for e in ["RT", "AT", "TL", "RU", "RA"]}


def test_moral_graph_of_v_structure_with_three_parents():
    dag = DAG(nodes="ABCX", edges=[("A", "X"), ("B", "X"), ("C", "X")])
    assert edge_set(moral_graph(dag)) == {
        frozenset(e) for e in ["AX", "BX", "CX", "AB", "AC", "BC"]
    }


def test_subgraph_keeps_induced_edges_only():
    g = UndirectedGraph(nodes="ABCD", edges=[("A", "B"), ("B", "C"), ("C", "D")])
    h = g.subgraph(["D", "B", "C"])
    assert h.nodes() == ("B", "C", "D")  # original insertion order
    assert edge_set(h) == {frozenset("BC"), frozenset("CD")}
    with pytest.raises(UnknownNodeError):
        g.subgraph(["A", "Z"])


def test_moralising_a_subgraph_differs_from_restricting_the_moral_graph():
    """A -> X <- B restricted to {A, B}, where the shared child X is outside the set.

    moral_graph(dag, restrict_to=S) moralises G[S] (needed for d-separation, M2.8):
    X is absent, so A and B are not married.
    moral_graph(dag).subgraph(S) restricts M(G) (what evidence on X produces):
    the marriage A–B survives.
    """
    dag = DAG(nodes="ABX", edges=[("A", "X"), ("B", "X")])
    assert edge_set(moral_graph(dag, restrict_to={"A", "B"})) == set()
    assert edge_set(moral_graph(dag).subgraph({"A", "B"})) == {frozenset("AB")}


def test_restrict_to_drops_parents_outside_the_set():
    # X is kept but its parent B is not: B must not appear, and nothing marries A to B.
    dag = DAG(nodes="ABX", edges=[("A", "X"), ("B", "X")])
    m = moral_graph(dag, restrict_to={"A", "X"})
    assert set(m.nodes()) == {"A", "X"}
    assert edge_set(m) == {frozenset("AX")}


def test_moral_graph_restricted_to_a_subset():
    dag = DAG(nodes="RATLU", edges=[("R", "T"), ("A", "T"), ("T", "L"), ("R", "U")])
    # Moralise the subgraph induced on {R, A, T}: the marriage R–A is kept.
    m = moral_graph(dag, restrict_to={"R", "A", "T"})
    assert set(m.nodes()) == {"R", "A", "T"}
    assert edge_set(m) == {frozenset(e) for e in ["RT", "AT", "RA"]}
    with pytest.raises(UnknownNodeError):
        moral_graph(dag, restrict_to={"R", "Z"})


def test_interaction_graph_of_hand_built_factors():
    a, b, c, d = (DiscreteVariable(n, ("0", "1")) for n in "ABCD")
    factors = [
        DiscreteFactor([a, b], np.ones((2, 2))),
        DiscreteFactor([b, c, d], np.ones((2, 2, 2))),
        DiscreteFactor([], 3.0),
    ]
    h = interaction_graph(factors)
    assert h.nodes() == ("A", "B", "C", "D")
    assert edge_set(h) == {frozenset(e) for e in ["AB", "BC", "BD", "CD"]}


def test_fixture_interaction_graph_is_moral_graph():
    model = late_network()
    assert interaction_graph(model.factors()) == moral_graph(model.graph)


@pytest.mark.parametrize("seed", range(40))
def test_interaction_graph_is_moral_graph_for_random_networks(seed):
    model = random_network(seed, n_vars=(1, 6))
    assert interaction_graph(model.factors()) == moral_graph(model.graph)


@pytest.mark.parametrize("seed", range(40))
def test_evidence_deletes_observed_vertices(seed):
    model = random_network(seed, n_vars=(2, 6))
    rng = np.random.default_rng(seed)
    observed = {v.name: v.states[0] for v in model.variables if rng.random() < 0.4}
    reduced = interaction_graph(phi.reduce(observed) for phi in model.factors())
    # H(Φ[e]) is the moral graph restricted to the unobserved vertices (Proposition 1).
    unobserved = [n for n in model.graph.nodes() if n not in observed]
    assert reduced == moral_graph(model.graph).subgraph(unobserved)
