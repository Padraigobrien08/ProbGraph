"""Directed acyclic graphs over string-keyed nodes.

Correctness arguments for each algorithm are in
``docs/mathematics/dag_algorithms.md``.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

from probgraph.exceptions import CycleError, UnknownNodeError, ValidationError


class DAG:
    """A mutable directed graph that is guaranteed to stay acyclic.

    Invariants:
        G1  No directed cycle exists. Self-loops are rejected.
        G2  ``topological_sort`` places every edge's source before its target.
        G3  ``u in parents(v)`` iff ``v in children(u)``.
        G4  ``u in ancestors(v)`` iff ``v in descendants(u)``.
        G5  An insertion that would create a cycle raises ``CycleError`` and
            leaves the graph unchanged.

    ``nodes()``, ``edges()`` and ``topological_sort()`` return results in a
    deterministic order based on insertion order.
    """

    def __init__(
        self,
        nodes: Iterable[str] = (),
        edges: Iterable[tuple[str, str]] = (),
    ) -> None:
        # Both directions are stored so that parent and child lookups are O(1).
        # Every mutation updates both maps together, which keeps G3 true.
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}
        # Dicts keep insertion order, so these double as the deterministic ordering.
        self._edges: dict[tuple[str, str], None] = {}

        for node in nodes:
            self.add_node(node)
        for source, target in edges:
            self.add_edge(source, target)

    # -- mutation -----------------------------------------------------------

    def add_node(self, node: str) -> None:
        """Add ``node``. Adding an existing node does nothing."""
        if not isinstance(node, str) or not node:
            raise ValidationError(f"Node keys must be non-empty strings, got {node!r}.")
        if node not in self._children:
            self._children[node] = set()
            self._parents[node] = set()

    def add_edge(self, source: str, target: str) -> None:
        """Add the edge ``source -> target``.

        Raises ``UnknownNodeError`` if either endpoint has not been added, and
        ``CycleError`` if the edge would close a directed cycle. In both cases
        the graph is left unchanged. Adding an existing edge does nothing.
        """
        self._require(source)
        self._require(target)
        if (source, target) in self._edges:
            return

        # Lemma 1: the new edge closes a cycle iff source == target or target ~> source.
        if source == target:
            raise CycleError(f"Self-loop {source} -> {source} is not allowed.")
        path_back = self._find_path(target, source)
        if path_back is not None:
            cycle = " -> ".join([source, *path_back])
            raise CycleError(f"Adding edge {source} -> {target} would create cycle {cycle}.")

        self._children[source].add(target)
        self._parents[target].add(source)
        self._edges[(source, target)] = None

    def remove_edge(self, source: str, target: str) -> None:
        """Remove the edge ``source -> target``.

        Does nothing if both nodes exist but the edge does not. Raises
        ``UnknownNodeError`` if either node is unknown.
        """
        self._require(source)
        self._require(target)
        self._children[source].discard(target)
        self._parents[target].discard(source)
        self._edges.pop((source, target), None)

    # -- structure ----------------------------------------------------------

    def nodes(self) -> tuple[str, ...]:
        return tuple(self._children)

    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._edges)

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edges

    def parents(self, node: str) -> set[str]:
        """pa(node): the direct predecessors of ``node``."""
        self._require(node)
        return set(self._parents[node])

    def children(self, node: str) -> set[str]:
        """The direct successors of ``node``."""
        self._require(node)
        return set(self._children[node])

    def ancestors(self, node: str) -> set[str]:
        """anc(node) = {u : u ~> node}. The node itself is not included."""
        self._require(node)
        return self._reachable(node, self._parents)

    def descendants(self, node: str) -> set[str]:
        """desc(node) = {v : node ~> v}. The node itself is not included."""
        self._require(node)
        return self._reachable(node, self._children)

    def ancestral_set(self, nodes: Iterable[str]) -> set[str]:
        """An*(nodes): the nodes themselves together with all of their ancestors.

        The result is closed under taking parents. See variable_elimination.md §5.
        """
        if isinstance(nodes, str):
            raise ValidationError(f"Expected a collection of node names, not the string {nodes!r}.")
        result: set[str] = set()
        stack = list(nodes)
        for node in stack:
            self._require(node)
        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(self._parents[node] - result)
        return result

    def topological_sort(self) -> list[str]:
        """Return a topological ordering, computed with Kahn's algorithm.

        When several nodes are ready at once, the one inserted earliest is
        taken first. The result is therefore deterministic. It is still only
        one of possibly many valid orderings.
        """
        order_of = {node: i for i, node in enumerate(self._children)}
        by_index = list(self._children)
        in_degree = {node: len(parents) for node, parents in self._parents.items()}

        ready = [order_of[n] for n, d in in_degree.items() if d == 0]
        heapq.heapify(ready)
        ordering: list[str] = []
        while ready:
            node = by_index[heapq.heappop(ready)]
            ordering.append(node)
            for child in self._children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    heapq.heappush(ready, order_of[child])

        if len(ordering) != len(by_index):
            # Insertion-time checks should make this unreachable (G1).
            raise CycleError("Internal error: graph contains a cycle.")
        return ordering

    # -- dunder -------------------------------------------------------------

    def __contains__(self, node: object) -> bool:
        return node in self._children

    def __len__(self) -> int:
        return len(self._children)

    def __repr__(self) -> str:
        return f"DAG(nodes={list(self.nodes())!r}, edges={list(self.edges())!r})"

    # -- internals ----------------------------------------------------------

    def _require(self, node: str) -> None:
        if node not in self._children:
            raise UnknownNodeError(f"Unknown node {node!r}.")

    @staticmethod
    def _reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
        """Return every node reachable from ``start`` by a path of length >= 1."""
        seen: set[str] = set()
        stack = list(adjacency[start])
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(adjacency[node] - seen)
        return seen

    def _find_path(self, start: str, goal: str) -> list[str] | None:
        """Find a directed path from ``start`` to ``goal`` with an iterative DFS.

        Returns the path as a list of nodes, or ``None`` if ``goal`` cannot be
        reached.
        """
        predecessor: dict[str, str | None] = {start: None}
        stack = [start]
        while stack:
            node = stack.pop()
            if node == goal:
                path = [node]
                while (prev := predecessor[path[-1]]) is not None:
                    path.append(prev)
                return path[::-1]
            for child in self._children[node]:
                if child not in predecessor:
                    predecessor[child] = node
                    stack.append(child)
        return None
