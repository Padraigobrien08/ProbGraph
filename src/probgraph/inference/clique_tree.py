"""Clique trees (junction trees) built from an elimination order.

The construction and its proofs (Lemmas 6–7, Theorem 8, Proposition 9) are in
``docs/mathematics/clique_trees.md`` (Part 2).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

from probgraph.exceptions import ValidationError
from probgraph.graphs import UndirectedGraph, triangulate
from probgraph.graphs.chordal import _check_full_order


class CliqueTree:
    """A tree of clusters with the running intersection property.

    Invariants:
        J1  it is a tree (clusters with no shared variables are joined by empty separators)
        J2  running intersection: each variable's clusters form a connected subtree
        J3  the clusters are exactly the maximal cliques of the triangulated graph
        J5  ``width`` = max cluster size - 1 = the elimination order's width
    """

    __slots__ = ("_adjacent", "_cliques", "_edges")

    def __init__(self, cliques: Sequence[frozenset[str]], edges: Sequence[tuple[int, int]]) -> None:
        self._cliques = tuple(cliques)
        self._edges = tuple(sorted((min(i, j), max(i, j)) for i, j in edges))
        self._adjacent: dict[int, list[int]] = {i: [] for i in range(len(self._cliques))}
        for i, j in self._edges:
            self._adjacent[i].append(j)
            self._adjacent[j].append(i)

    @classmethod
    def from_elimination(cls, graph: UndirectedGraph, order: Sequence[str]) -> CliqueTree:
        """Build the clique tree of ``graph`` triangulated along ``order`` (§6).

        Step 1: link each elimination cluster C_v = {v} ∪ later(v) to the cluster of
        its parent, the first vertex of later(v) in ``order`` (Lemma 6).
        Step 2: contract every cluster into a neighbour that contains it (Lemma 7).
        """
        position = _check_full_order(graph, order)
        triangulated = triangulate(graph, order)
        later = {
            v: {u for u in triangulated.neighbours(v) if position[u] > position[v]} for v in order
        }
        cluster = {v: frozenset({v} | later[v]) for v in order}
        adjacent: dict[str, set[str]] = {v: set() for v in order}
        roots = []
        for v in order:
            if later[v]:
                parent = min(later[v], key=position.__getitem__)
                adjacent[v].add(parent)
                adjacent[parent].add(v)
            else:
                roots.append(v)
        for a, b in itertools.pairwise(roots):  # one tree, even for a disconnected graph
            adjacent[a].add(b)
            adjacent[b].add(a)

        alive = list(order)
        changed = True
        while changed:
            changed = False
            for v in alive:
                host = next(
                    (
                        u
                        for u in sorted(adjacent[v], key=position.__getitem__)
                        if cluster[v] <= cluster[u]
                    ),
                    None,
                )
                if host is not None:
                    for w in adjacent.pop(v):
                        adjacent[w].discard(v)
                        if w != host:
                            adjacent[w].add(host)
                            adjacent[host].add(w)
                    alive.remove(v)
                    changed = True
                    break

        index = {v: i for i, v in enumerate(alive)}
        edges = {
            (min(index[v], index[u]), max(index[v], index[u])) for v in alive for u in adjacent[v]
        }
        return cls([cluster[v] for v in alive], sorted(edges))

    # -- structure ------------------------------------------------------------

    @property
    def cliques(self) -> tuple[frozenset[str], ...]:
        return self._cliques

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Tree edges as (i, j) with i < j, sorted."""
        return self._edges

    @property
    def width(self) -> int:
        """The largest clique size minus 1; 0 for an empty tree."""
        return max((len(c) - 1 for c in self._cliques), default=0)

    def neighbours(self, i: int) -> tuple[int, ...]:
        return tuple(self._adjacent[i])

    def separator(self, i: int, j: int) -> frozenset[str]:
        """C_i ∩ C_j for adjacent cliques i and j."""
        if (min(i, j), max(i, j)) not in self._edges:
            raise ValidationError(f"Cliques {i} and {j} are not adjacent in the tree.")
        return self._cliques[i] & self._cliques[j]

    def assign(self, scopes: Iterable[Iterable[str]]) -> list[int]:
        """For each scope, the index of the first clique that contains it (J6)."""
        result = []
        for scope in scopes:
            names = set(scope)
            home = next((i for i, c in enumerate(self._cliques) if names <= c), None)
            if home is None:
                raise ValidationError(f"Scope {sorted(names)} is contained in no clique.")
            result.append(home)
        return result

    def __repr__(self) -> str:
        cliques = [sorted(c) for c in self._cliques]
        return f"CliqueTree(cliques={cliques}, edges={list(self._edges)})"
