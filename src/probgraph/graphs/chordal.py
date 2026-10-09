"""Chordal graphs: triangulation, recognition by maximum cardinality search, maximal cliques.

The proofs are in ``docs/mathematics/clique_trees.md`` (Part 1).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs.undirected import UndirectedGraph


def triangulate(graph: UndirectedGraph, order: Sequence[str]) -> UndirectedGraph:
    """Return ``graph`` plus every fill edge created by eliminating its vertices in ``order``.

    ``order`` must list every vertex exactly once. The result is chordal, and
    ``order`` is a perfect elimination ordering of it (Lemma 1). The input graph is
    not modified.
    """
    _check_full_order(graph, order)
    result = graph.copy()
    working = graph.copy()
    for z in order:
        for u, v in itertools.combinations(working.neighbours(z), 2):
            working.add_edge(u, v)
            result.add_edge(u, v)
        working.remove_node(z)
    return result


def is_perfect_elimination_ordering(graph: UndirectedGraph, order: Sequence[str]) -> bool:
    """True if, for every vertex, its neighbours later in ``order`` form a clique."""
    position = _check_full_order(graph, order)
    for v in order:
        later = [u for u in graph.neighbours(v) if position[u] > position[v]]
        if any(not graph.has_edge(a, b) for a, b in itertools.combinations(later, 2)):
            return False
    return True


def perfect_elimination_ordering(graph: UndirectedGraph) -> list[str] | None:
    """A perfect elimination ordering if ``graph`` is chordal, otherwise ``None``.

    Runs maximum cardinality search and checks its reversed visit order (Theorem 4).
    A returned order is always verified, so a non-None result proves chordality
    without relying on Theorem 4 (Lemma 2).
    """
    order = list(reversed(_maximum_cardinality_search(graph)))
    return order if is_perfect_elimination_ordering(graph, order) else None


def is_chordal(graph: UndirectedGraph) -> bool:
    """True if every cycle of length >= 4 has a chord."""
    return perfect_elimination_ordering(graph) is not None


def maximal_cliques(graph: UndirectedGraph) -> list[frozenset[str]]:
    """The maximal cliques of a chordal graph (Proposition 5); at most one per vertex.

    Cliques are listed in the order of their first vertex in the perfect
    elimination ordering. Raises ``ValidationError`` for a non-chordal graph, which
    can have exponentially many maximal cliques.
    """
    order = perfect_elimination_ordering(graph)
    if order is None:
        raise ValidationError("maximal_cliques() needs a chordal graph; triangulate it first.")
    position = {v: i for i, v in enumerate(order)}
    candidates = [
        frozenset({v} | {u for u in graph.neighbours(v) if position[u] > position[v]})
        for v in order
    ]
    cliques: list[frozenset[str]] = []
    for c in candidates:
        if not any(c < d for d in candidates) and c not in cliques:
            cliques.append(c)
    return cliques


def _maximum_cardinality_search(graph: UndirectedGraph) -> list[str]:
    """Visit vertices one by one, each time the one with the most visited neighbours.

    Ties go to the vertex inserted earliest. O(n^2 + m), which is plenty here; the
    original algorithm reaches O(n + m) with bucket queues.
    """
    nodes = graph.nodes()
    weight = dict.fromkeys(nodes, 0)
    rank = {v: i for i, v in enumerate(nodes)}
    unvisited = set(nodes)
    visit: list[str] = []
    while unvisited:
        v = max(unvisited, key=lambda u: (weight[u], -rank[u]))
        visit.append(v)
        unvisited.remove(v)
        for u in graph.neighbours(v) & unvisited:
            weight[u] += 1
    return visit


def _check_full_order(graph: UndirectedGraph, order: Sequence[str]) -> dict[str, int]:
    if isinstance(order, str):
        raise ValidationError(f"The order must be a sequence of names, not the string {order!r}.")
    names = list(order)
    unknown = [n for n in names if n not in graph]
    if unknown:
        raise UnknownNodeError(f"The order names unknown vertices {unknown}.")
    if len(names) != len(set(names)) or set(names) != set(graph.nodes()):
        raise ValidationError(
            f"The order must list every vertex exactly once: got {names}, "
            f"vertices are {list(graph.nodes())}."
        )
    return {v: i for i, v in enumerate(names)}
