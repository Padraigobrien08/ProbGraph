"""Partially directed graphs and Markov equivalence classes of DAGs.

The mathematics (P28) is in ``docs/mathematics/equivalence_classes.md``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

from probgraph.exceptions import CycleError, ValidationError
from probgraph.graphs import DAG


@dataclass(frozen=True)
class PDAG:
    """A graph with directed edges ``(a, b)`` meaning a -> b, and undirected edges {a, b}.

    A CPDAG (from ``cpdag``) represents a Markov equivalence class: its directed edges
    are compelled, its undirected edges reversible (equivalence_classes.md §3).
    """

    nodes: tuple[str, ...]
    directed: frozenset[tuple[str, str]]
    undirected: frozenset[frozenset[str]]

    def __post_init__(self) -> None:
        known = set(self.nodes)
        if len(known) != len(self.nodes):
            raise ValidationError(f"PDAG has duplicate nodes {self.nodes}.")
        for a, b in self.directed:
            if a not in known or b not in known:
                raise ValidationError(f"Edge {a!r} -> {b!r} names an unknown node.")
            if a == b:
                raise ValidationError(f"Edge {a!r} -> {b!r} is a self-loop.")
            if (b, a) in self.directed:
                raise ValidationError(f"Edge {a!r} - {b!r} is directed in both directions.")
            if frozenset((a, b)) in self.undirected:
                raise ValidationError(f"Edge {a!r} - {b!r} is both directed and undirected.")
        for edge in self.undirected:
            if len(edge) != 2:
                raise ValidationError(f"Undirected edge {set(edge)} must join two distinct nodes.")
            if not edge <= known:
                raise ValidationError(f"Undirected edge {set(edge)} names an unknown node.")

    def adjacent(self, a: str, b: str) -> bool:
        return (
            (a, b) in self.directed
            or (b, a) in self.directed
            or frozenset((a, b)) in self.undirected
        )

    def consistent_extension(self) -> DAG:
        """A DAG with this skeleton, these directed edges and no new v-structures (§4).

        Dor–Tarsi: repeatedly remove a node with no outgoing directed edge whose
        undirected neighbours are adjacent to all its other adjacents, orienting its
        undirected edges into it. Raises ``ValidationError`` if none exists.
        """
        remaining = set(self.nodes)
        directed = set(self.directed)
        undirected = set(self.undirected)
        oriented: list[tuple[str, str]] = []
        while remaining:
            for x in sorted(remaining, key=self.nodes.index):
                if any(a == x and b in remaining for a, b in directed):
                    continue
                neighbours = {next(iter(e - {x})) for e in undirected if x in e and e <= remaining}
                parents = {a for a, b in directed if b == x and a in remaining}
                adjacent = neighbours | parents
                if all(self.adjacent(n, m) for n in neighbours for m in adjacent if m != n):
                    break
            else:
                raise ValidationError("This PDAG has no consistent extension (no member DAG).")
            for n in sorted({next(iter(e - {x})) for e in undirected if x in e and e <= remaining}):
                oriented.append((n, x))
            remaining.discard(x)
        return DAG(nodes=list(self.nodes), edges=sorted(directed | set(oriented)))

    def members(self, limit: int = 10_000) -> Iterator[DAG]:
        """Every DAG whose CPDAG is this one (small graphs: tries each orientation)."""
        target = cpdag(self.consistent_extension()) if self._extendable() else None
        if target is None or target != self:
            return
        pairs = sorted(tuple(sorted(e)) for e in self.undirected)
        count = 0
        for choice in itertools.product((False, True), repeat=len(pairs)):
            edges = set(self.directed) | {
                (b, a) if flip else (a, b) for (a, b), flip in zip(pairs, choice, strict=True)
            }
            try:
                dag = DAG(nodes=list(self.nodes), edges=sorted(edges))
            except CycleError:
                continue
            if cpdag(dag) == self:
                yield dag
                count += 1
                if count >= limit:
                    return

    def _extendable(self) -> bool:
        try:
            self.consistent_extension()
        except ValidationError:
            return False
        return True


def cpdag(dag: DAG) -> PDAG:
    """The CPDAG of ``dag``: its pattern closed under Meek's rules R1–R3 (§3)."""
    nodes = tuple(dag.nodes())
    edges = dag.edges()
    adjacent = {frozenset(e) for e in edges}
    directed: set[tuple[str, str]] = set()
    for c in nodes:
        for a, b in itertools.combinations(sorted(dag.parents(c)), 2):
            if frozenset((a, b)) not in adjacent:
                directed |= {(a, c), (b, c)}
    undirected = {frozenset(e) for e in edges if e not in directed}
    _apply_meek(nodes, directed, undirected, adjacent)
    return PDAG(nodes, frozenset(directed), frozenset(undirected))


def markov_equivalent(a: DAG, b: DAG) -> bool:
    """Same d-separation statements, i.e. the same CPDAG (Theorem 1)."""
    return cpdag(a) == cpdag(b)


def structural_hamming_distance(a: PDAG, b: PDAG) -> int:
    """The number of node pairs whose marks (absent, undirected, ->, <-) differ (§5)."""
    if set(a.nodes) != set(b.nodes):
        raise ValidationError("Structural Hamming distance needs PDAGs on the same nodes.")
    return sum(
        _mark(a, x, y) != _mark(b, x, y) for x, y in itertools.combinations(sorted(a.nodes), 2)
    )


def _mark(g: PDAG, x: str, y: str) -> str:
    if (x, y) in g.directed:
        return "->"
    if (y, x) in g.directed:
        return "<-"
    if frozenset((x, y)) in g.undirected:
        return "-"
    return ""


def _apply_meek(
    nodes: tuple[str, ...],
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    adjacent: set[frozenset[str]],
) -> None:
    """Orient undirected edges by R1–R3 until nothing changes (in place)."""

    def adj(x: str, y: str) -> bool:
        return frozenset((x, y)) in adjacent

    changed = True
    while changed:
        changed = False
        for edge in sorted(undirected, key=lambda e: tuple(sorted(e))):
            for b, c in itertools.permutations(sorted(edge), 2):
                # R1: a -> b, b - c, a and c not adjacent  =>  b -> c
                r1 = any((a, b) in directed and not adj(a, c) for a in nodes if a != c)
                # R2: b -> m -> c and b - c  =>  b -> c
                r2 = any((b, m) in directed and (m, c) in directed for m in nodes)
                # R3: b - m, b - n, m -> c, n -> c, m and n not adjacent  =>  b -> c
                r3 = any(
                    frozenset((b, m)) in undirected
                    and frozenset((b, n)) in undirected
                    and (m, c) in directed
                    and (n, c) in directed
                    and not adj(m, n)
                    for m, n in itertools.combinations(nodes, 2)
                )
                if r1 or r2 or r3:
                    undirected.discard(edge)
                    directed.add((b, c))
                    changed = True
                    break
