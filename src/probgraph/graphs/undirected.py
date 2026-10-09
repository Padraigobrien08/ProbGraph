"""Undirected graphs, moralisation and interaction graphs.

The constructions and Proposition 1 are in ``docs/mathematics/elimination_orders.md``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs.dag import DAG

if TYPE_CHECKING:
    from probgraph.factors import DiscreteFactor


class UndirectedGraph:
    """A simple undirected graph over string-keyed nodes, with no self-loops.

    ``nodes()`` and ``edges()`` are returned in insertion order. Equality
    compares node and edge *sets*, so it ignores insertion order. Graphs are
    mutable, and therefore unhashable.
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, nodes: Iterable[str] = (), edges: Iterable[tuple[str, str]] = ()) -> None:
        self._adjacent: dict[str, set[str]] = {}
        # Keyed by the unordered pair; the value keeps the pair as first added.
        self._edges: dict[frozenset[str], tuple[str, str]] = {}
        for node in nodes:
            self.add_node(node)
        for u, v in edges:
            self.add_edge(u, v)

    # -- mutation -----------------------------------------------------------

    def add_node(self, node: str) -> None:
        if not isinstance(node, str) or not node:
            raise ValidationError(f"Node keys must be non-empty strings, got {node!r}.")
        self._adjacent.setdefault(node, set())

    def add_edge(self, u: str, v: str) -> None:
        self._require(u)
        self._require(v)
        if u == v:
            raise ValidationError(f"Self-loop {u} — {u} is not allowed.")
        key = frozenset((u, v))
        if key not in self._edges:
            self._edges[key] = (u, v)
            self._adjacent[u].add(v)
            self._adjacent[v].add(u)

    def remove_edge(self, u: str, v: str) -> None:
        """Remove the edge u — v. Does nothing if both nodes exist but the edge does not."""
        self._require(u)
        self._require(v)
        if self._edges.pop(frozenset((u, v)), None) is not None:
            self._adjacent[u].discard(v)
            self._adjacent[v].discard(u)

    def remove_node(self, node: str) -> None:
        """Remove ``node`` and every edge incident to it."""
        self._require(node)
        for other in self._adjacent.pop(node):
            self._adjacent[other].discard(node)
            del self._edges[frozenset((node, other))]

    # -- structure ----------------------------------------------------------

    def nodes(self) -> tuple[str, ...]:
        return tuple(self._adjacent)

    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._edges.values())

    def has_edge(self, u: str, v: str) -> bool:
        return frozenset((u, v)) in self._edges

    def neighbours(self, node: str) -> set[str]:
        self._require(node)
        return set(self._adjacent[node])

    def copy(self) -> UndirectedGraph:
        return UndirectedGraph(self.nodes(), self.edges())

    def subgraph(self, nodes: Iterable[str]) -> UndirectedGraph:
        """The induced subgraph on ``nodes``, keeping this graph's insertion order."""
        keep = set(nodes)
        for node in keep:
            self._require(node)
        return UndirectedGraph(
            [n for n in self._adjacent if n in keep],
            [(u, v) for u, v in self._edges.values() if u in keep and v in keep],
        )

    def separated(self, xs: Iterable[str], ys: Iterable[str], given: Iterable[str] = ()) -> bool:
        """True if every path from ``xs`` to ``ys`` passes through ``given``.

        The three sets must be known, pairwise disjoint, and ``xs`` and ``ys``
        must be non-empty.
        """
        x, y, z = (self._node_set(group) for group in (xs, ys, given))
        if not x or not y:
            raise ValidationError("separated() needs non-empty xs and ys.")
        if x & y or x & z or y & z:
            raise ValidationError("separated() needs disjoint xs, ys and given.")
        seen = set(x)
        frontier = list(x)
        while frontier:
            node = frontier.pop()
            for nxt in self._adjacent[node]:
                if nxt in y:
                    return False
                if nxt not in seen and nxt not in z:
                    seen.add(nxt)
                    frontier.append(nxt)
        return True

    # -- dunder -------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UndirectedGraph):
            return NotImplemented
        return self._adjacent.keys() == other._adjacent.keys() and (
            self._edges.keys() == other._edges.keys()
        )

    def __contains__(self, node: object) -> bool:
        return node in self._adjacent

    def __len__(self) -> int:
        return len(self._adjacent)

    def __repr__(self) -> str:
        return f"UndirectedGraph(nodes={list(self.nodes())!r}, edges={list(self.edges())!r})"

    # -- internals ----------------------------------------------------------

    def _require(self, node: str) -> None:
        if node not in self._adjacent:
            raise UnknownNodeError(f"Unknown node {node!r}.")

    def _node_set(self, nodes: Iterable[str]) -> set[str]:
        if isinstance(nodes, str):
            raise ValidationError(f"Expected a collection of node names, not the string {nodes!r}.")
        result = set(nodes)
        for node in result:
            self._require(node)
        return result


def moral_graph(dag: DAG, restrict_to: Iterable[str] | None = None) -> UndirectedGraph:
    """Moralise ``dag``: marry every pair of parents that share a child, then drop directions.

    With ``restrict_to``, moralise the subgraph *induced* on those nodes. Parents
    are married only through a child inside the set. This is generally not the
    same as ``moral_graph(dag).subgraph(restrict_to)``; see elimination_orders.md §1.
    """
    if restrict_to is None:
        keep = set(dag.nodes())
    else:
        keep = set(restrict_to)
        unknown = sorted(n for n in keep if n not in dag)
        if unknown:
            raise UnknownNodeError(f"Unknown nodes {unknown}.")
    order = [n for n in dag.nodes() if n in keep]
    position = {n: i for i, n in enumerate(order)}
    graph = UndirectedGraph(order)
    for child in order:
        parents = sorted((p for p in dag.parents(child) if p in keep), key=position.__getitem__)
        for parent in parents:
            graph.add_edge(parent, child)
        for a, b in itertools.combinations(parents, 2):
            graph.add_edge(a, b)
    return graph


def interaction_graph(factors: Iterable[DiscreteFactor]) -> UndirectedGraph:
    """Connect every two variables that appear together in some factor's scope.

    Nodes are added in order of first appearance.
    """
    graph = UndirectedGraph()
    for phi in factors:
        for name in phi.names:
            graph.add_node(name)
        for u, v in itertools.combinations(phi.names, 2):
            graph.add_edge(u, v)
    return graph
