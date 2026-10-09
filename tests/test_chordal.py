"""M3.4: triangulation, chordality and maximal cliques (spec task 4; J4; clique_trees.md Part 1).

Oracles:
  * the definition, by brute force: no induced cycle of length >= 4;
  * every clique, by brute force over vertex subsets;
  * OEIS A058862, the number of labelled chordal graphs: 1, 2, 8, 61, 822, 18154.
"""

import itertools

import numpy as np
import pytest

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs import (
    UndirectedGraph,
    is_chordal,
    is_perfect_elimination_ordering,
    maximal_cliques,
    perfect_elimination_ordering,
    triangulate,
)
from probgraph.inference import simulate_elimination

OEIS_A058862 = {1: 1, 2: 2, 3: 8, 4: 61, 5: 822, 6: 18154}

# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def has_chordless_cycle(g: UndirectedGraph) -> bool:
    """True if some vertex subset of size >= 4 induces a single cycle."""
    nodes = g.nodes()
    for k in range(4, len(nodes) + 1):
        for subset in itertools.combinations(nodes, k):
            s = set(subset)
            if all(len(g.neighbours(v) & s) == 2 for v in subset):
                seen, stack = {subset[0]}, [subset[0]]
                while stack:
                    v = stack.pop()
                    for w in g.neighbours(v) & s:
                        if w not in seen:
                            seen.add(w)
                            stack.append(w)
                if len(seen) == k:
                    return True
    return False


def brute_force_maximal_cliques(g: UndirectedGraph) -> set[frozenset[str]]:
    nodes = g.nodes()
    cliques = [
        frozenset(s)
        for k in range(1, len(nodes) + 1)
        for s in itertools.combinations(nodes, k)
        if all(g.has_edge(u, v) for u, v in itertools.combinations(s, 2))
    ]
    return {c for c in cliques if not any(c < d for d in cliques)}


def all_graphs(n: int):
    nodes = [f"N{i}" for i in range(n)]
    pairs = list(itertools.combinations(nodes, 2))
    for mask in range(1 << len(pairs)):
        yield UndirectedGraph(nodes, [p for k, p in enumerate(pairs) if mask >> k & 1])


def random_graph(rng, n: int, p: float) -> UndirectedGraph:
    nodes = [f"N{i}" for i in range(n)]
    return UndirectedGraph(
        nodes, [(u, v) for u, v in itertools.combinations(nodes, 2) if rng.random() < p]
    )


def cycle(n: int) -> UndirectedGraph:
    nodes = [f"C{i}" for i in range(n)]
    return UndirectedGraph(nodes, [(nodes[i], nodes[(i + 1) % n]) for i in range(n)])


def edge_set(g: UndirectedGraph) -> set[frozenset[str]]:
    return {frozenset(e) for e in g.edges()}


# ---------------------------------------------------------------------------
# Triangulation (Lemma 1)
# ---------------------------------------------------------------------------


def test_triangulating_the_four_cycle():
    g = UndirectedGraph("ABCD", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    t = triangulate(g, list("ABCD"))
    assert edge_set(t) - edge_set(g) == {frozenset("BD")}  # eliminating A first adds B–D
    assert is_chordal(t) and not is_chordal(g)
    assert edge_set(g) <= edge_set(t)
    assert g.nodes() == t.nodes()
    assert not g.has_edge("B", "D")  # the input graph is not modified


@pytest.mark.parametrize(
    ("order", "error"),
    [
        (["A", "B", "C"], ValidationError),
        (["A", "B", "C", "D", "A"], ValidationError),
        (["A", "B", "C", "Z"], UnknownNodeError),
        ("ABCD", ValidationError),
    ],
)
def test_triangulate_requires_a_full_ordering(order, error):
    g = UndirectedGraph("ABCD", [("A", "B")])
    with pytest.raises(error):
        triangulate(g, order)


@pytest.mark.parametrize("seed", range(60))
def test_lemma_1_the_order_is_a_peo_of_its_triangulation(seed):
    rng = np.random.default_rng(seed)
    g = random_graph(rng, int(rng.integers(1, 9)), p=0.35)
    order = [str(v) for v in rng.permutation(list(g.nodes()))]
    t = triangulate(g, order)
    assert edge_set(g) <= edge_set(t)
    assert is_perfect_elimination_ordering(t, order)
    assert is_chordal(t)
    if len(g) <= 7:
        assert not has_chordless_cycle(t)  # J4, by the definition
    # Width agrees with M2's simulation of the same order (J5, preview).
    cards = dict.fromkeys(g.nodes(), 2)
    later = {v: {u for u in t.neighbours(v) if order.index(u) > order.index(v)} for v in order}
    assert (
        max((len(s) for s in later.values()), default=0)
        == simulate_elimination(g, order, cards).width
    )


# ---------------------------------------------------------------------------
# Chordality (Theorems 3 and 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 7))
def test_is_chordal_agrees_with_the_definition_on_every_small_graph(n):
    count = 0
    for g in all_graphs(n):
        chordal = is_chordal(g)
        if n <= 5:
            assert chordal is (not has_chordless_cycle(g)), g
        count += chordal
    assert count == OEIS_A058862[n]


def test_known_families():
    assert not is_chordal(cycle(4)) and not is_chordal(cycle(7))
    assert is_chordal(cycle(3))
    chorded = cycle(5)
    chorded.add_edge("C0", "C2")
    chorded.add_edge("C0", "C3")
    assert is_chordal(chorded)  # a fan
    assert is_chordal(UndirectedGraph())  # the empty graph
    tree = UndirectedGraph("ABCDE", [("A", "B"), ("A", "C"), ("C", "D"), ("C", "E")])
    assert is_chordal(tree)


@pytest.mark.parametrize("seed", range(60))
def test_peo_is_returned_exactly_for_chordal_graphs(seed):
    rng = np.random.default_rng(seed + 1000)
    g = random_graph(rng, int(rng.integers(1, 9)), p=float(rng.uniform(0.2, 0.8)))
    peo = perfect_elimination_ordering(g)
    if peo is None:
        assert not is_chordal(g)
        if len(g) <= 7:
            assert has_chordless_cycle(g)
    else:
        assert sorted(peo) == sorted(g.nodes())
        assert is_perfect_elimination_ordering(g, peo)
        assert is_chordal(g)


def test_is_perfect_elimination_ordering():
    g = UndirectedGraph("ABCD", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("B", "D")])
    assert is_perfect_elimination_ordering(g, list("ACBD"))
    assert not is_perfect_elimination_ordering(
        g, list("BACD")
    )  # B's later neighbours A, C, D: A–C missing
    with pytest.raises(ValidationError):
        is_perfect_elimination_ordering(g, list("ABC"))


# ---------------------------------------------------------------------------
# Maximal cliques (Proposition 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 6))
def test_maximal_cliques_on_every_small_chordal_graph(n):
    for g in all_graphs(n):
        if not is_chordal(g):
            continue
        cliques = maximal_cliques(g)
        assert set(cliques) == brute_force_maximal_cliques(g)
        assert len(cliques) == len(set(cliques)) <= max(n, 1)


@pytest.mark.parametrize("seed", range(40))
def test_maximal_cliques_of_random_triangulations(seed):
    rng = np.random.default_rng(seed + 2000)
    g = random_graph(rng, int(rng.integers(2, 9)), p=0.3)
    t = triangulate(g, [str(v) for v in rng.permutation(list(g.nodes()))])
    cliques = maximal_cliques(t)
    assert set(cliques) == brute_force_maximal_cliques(t)
    # Every edge of the original graph lies inside some clique (J3, preview).
    assert all(any({u, v} <= c for c in cliques) for u, v in g.edges())


def test_maximal_cliques_examples():
    g = UndirectedGraph("ABCD", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("B", "D")])
    assert set(maximal_cliques(g)) == {frozenset("ABD"), frozenset("BCD")}
    complete = UndirectedGraph("ABCD", list(itertools.combinations("ABCD", 2)))
    assert maximal_cliques(complete) == [frozenset("ABCD")]
    lonely = UndirectedGraph("AB")
    assert set(maximal_cliques(lonely)) == {frozenset("A"), frozenset("B")}


def test_maximal_cliques_refuses_non_chordal_graphs():
    with pytest.raises(ValidationError, match="chordal"):
        maximal_cliques(cycle(4))


def test_results_are_deterministic():
    rng = np.random.default_rng(7)
    g = triangulate(random_graph(rng, 8, 0.3), [f"N{i}" for i in range(8)])
    assert maximal_cliques(g) == maximal_cliques(g.copy())
    assert perfect_elimination_ordering(g) == perfect_elimination_ordering(g.copy())
