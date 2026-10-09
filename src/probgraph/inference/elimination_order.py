"""Elimination orders: greedy heuristics and cost traces.

Lemma 2 in ``docs/mathematics/elimination_orders.md`` shows that eliminating a
variable from the factors is the same as eliminating its vertex from the
interaction graph. So every cost here is computed on the graph alone.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.graphs import UndirectedGraph

Heuristic = Literal["min_fill", "min_neighbours", "min_weight"]
HEURISTICS: tuple[Heuristic, ...] = ("min_fill", "min_neighbours", "min_weight")


@dataclass(frozen=True)
class EliminationStep:
    """One elimination step.

    ``scope`` is the scope of the product ψ_z, so it includes ``variable``.
    ``size`` is the number of cells in that product.
    """

    variable: str
    scope: frozenset[str]
    size: int


@dataclass(frozen=True)
class EliminationTrace:
    """The sequence of intermediate tables that an elimination order creates."""

    steps: tuple[EliminationStep, ...]

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(step.variable for step in self.steps)

    @property
    def width(self) -> int:
        """max |scope(ψ_z)| - 1: the most neighbours any variable had when eliminated.

        0 if there are no steps.
        """
        return max((len(step.scope) - 1 for step in self.steps), default=0)

    @property
    def max_size(self) -> int:
        """The largest intermediate table, in cells. 0 if empty."""
        return max((step.size for step in self.steps), default=0)

    @property
    def total_cost(self) -> int:
        """The sum of the intermediate table sizes. This is proportional to the work done."""
        return sum(step.size for step in self.steps)


def greedy_order(
    graph: UndirectedGraph,
    eliminate: Iterable[str],
    heuristic: Heuristic = "min_fill",
    cardinalities: Mapping[str, int] | None = None,
) -> list[str]:
    """Order ``eliminate`` greedily: repeatedly take the eligible vertex with the lowest score.

    Scores are computed in the current graph, including fill edges added by
    earlier steps. Ties go to the vertex inserted earliest into ``graph``.
    ``min_weight`` needs ``cardinalities``.
    """
    if heuristic not in HEURISTICS:
        raise ValidationError(
            f"Unknown elimination heuristic {heuristic!r}; expected one of {HEURISTICS}."
        )
    remaining = _check_names(graph, eliminate, "eliminate")
    if heuristic == "min_weight":
        _check_cardinalities(graph, cardinalities)
    rank = {node: i for i, node in enumerate(graph.nodes())}
    working = graph.copy()
    order: list[str] = []
    while remaining:
        z = min(remaining, key=lambda n: (_score(working, n, heuristic, cardinalities), rank[n]))
        order.append(z)
        remaining.remove(z)
        _eliminate_vertex(working, z)
    return order


def simulate_elimination(
    graph: UndirectedGraph, order: Sequence[str], cardinalities: Mapping[str, int]
) -> EliminationTrace:
    """Predict the trace of eliminating ``order`` from the graph alone (Lemma 2)."""
    names = _check_names(graph, order, "Elimination order")
    _check_cardinalities(graph, cardinalities)
    working = graph.copy()
    steps = []
    for z in names:
        scope = frozenset({z} | working.neighbours(z))
        steps.append(EliminationStep(z, scope, math.prod(cardinalities[v] for v in scope)))
        _eliminate_vertex(working, z)
    return EliminationTrace(tuple(steps))


def _score(
    graph: UndirectedGraph,
    node: str,
    heuristic: Heuristic,
    cardinalities: Mapping[str, int] | None,
) -> int:
    neighbours = graph.neighbours(node)
    if heuristic == "min_neighbours":
        return len(neighbours)
    if heuristic == "min_weight":
        assert cardinalities is not None
        return math.prod(cardinalities[v] for v in neighbours)
    return sum(1 for u, v in itertools.combinations(neighbours, 2) if not graph.has_edge(u, v))


def _eliminate_vertex(graph: UndirectedGraph, z: str) -> None:
    """Connect all of z's neighbours (the fill edges), then delete z."""
    for u, v in itertools.combinations(graph.neighbours(z), 2):
        graph.add_edge(u, v)
    graph.remove_node(z)


def _check_names(graph: UndirectedGraph, names: Iterable[str], what: str) -> list[str]:
    if isinstance(names, str):
        raise ValidationError(f"{what} must be a sequence of names, not the string {names!r}.")
    result = list(names)
    duplicates = sorted({n for n in result if result.count(n) > 1})
    if duplicates:
        raise ValidationError(f"{what} has duplicate variables {duplicates}.")
    unknown = [n for n in result if n not in graph]
    if unknown:
        raise UnknownNodeError(f"{what} names unknown variables {unknown}.")
    return result


def _check_cardinalities(graph: UndirectedGraph, cardinalities: Mapping[str, int] | None) -> None:
    if cardinalities is None:
        raise ValidationError("This operation requires cardinalities for every graph node.")
    missing = [n for n in graph.nodes() if n not in cardinalities]
    if missing:
        raise ValidationError(f"Missing cardinalities for {missing}.")
