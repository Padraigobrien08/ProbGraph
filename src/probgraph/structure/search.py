"""Score-based structure search: hill climbing, and exact search by dynamic programming.

The derivations (P29) are in ``docs/mathematics/structure_search.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.learning.dataset import Dataset
from probgraph.learning.dirichlet import DirichletPrior
from probgraph.learning.model_selection import _family_score
from probgraph.models import BayesianNetwork

Score = Literal["bic", "bdeu"]
Edge = tuple[str, str]

#: The tolerance for "strictly better" when comparing floating-point scores.
IMPROVEMENT = 1e-10


@dataclass(frozen=True)
class SearchResult:
    """A learned graph (no CPDs: pass it to ``maximum_likelihood`` or ``bayesian_estimate``)."""

    structure: BayesianNetwork
    score: float
    history: tuple[float, ...]
    evaluations: int


class FamilyScorer:
    """s(X, U) for the data, cached by (child, parent set) (structure_search.md §2)."""

    def __init__(self, data: Dataset, score: Score, equivalent_sample_size: float) -> None:
        if not isinstance(data, Dataset):
            raise ValidationError(f"Expected a Dataset, got {type(data).__name__}.")
        if score not in ("bic", "bdeu"):
            raise ValidationError(f"score must be 'bic' or 'bdeu', got {score!r}.")
        if not data.is_complete:
            raise ValidationError(
                f"Structure search needs complete data; {data.missing_count} values are missing."
            )
        if data.n_rows == 0:
            raise ValidationError("Structure search needs at least one row.")
        if score == "bdeu":
            DirichletPrior.bdeu(equivalent_sample_size)  # validates
        self.data = data
        self.names = tuple(data.names)
        self._variables = {v.name: v for v in data.variables}
        self._score = score
        self._ess = float(equivalent_sample_size)
        self._cache: dict[tuple[str, frozenset[str]], float] = {}

    @property
    def evaluations(self) -> int:
        return len(self._cache)

    def __call__(self, child: str, parents: Iterable[str]) -> float:
        key = (child, frozenset(parents))
        if key not in self._cache:
            ordered = [n for n in self.names if n in key[1]]
            counts = self.data.counts([child, *ordered]).values
            self._cache[key] = _family_score(
                self._variables[child],
                [self._variables[n] for n in ordered],
                counts,
                self._score,
                self._ess,
                self.data.n_rows,
            )
        return self._cache[key]


class _Graph:
    """Parent sets with acyclicity checks, for the search."""

    def __init__(self, names: Sequence[str], edges: Iterable[Edge]) -> None:
        self.names = tuple(names)
        self.parents: dict[str, set[str]] = {n: set() for n in names}
        for a, b in edges:
            self.parents[b].add(a)

    def copy(self) -> _Graph:
        g = _Graph(self.names, ())
        g.parents = {n: set(p) for n, p in self.parents.items()}
        return g

    def edges(self) -> list[Edge]:
        return [(a, b) for b in self.names for a in self.names if a in self.parents[b]]

    def key(self) -> frozenset[Edge]:
        return frozenset(self.edges())

    def reaches(self, start: str, goal: str, skip: Edge | None = None) -> bool:
        """Is there a directed path start ⇝ goal (optionally ignoring one edge)?"""
        children: dict[str, list[str]] = {n: [] for n in self.names}
        for b, ps in self.parents.items():
            for a in ps:
                if (a, b) != skip:
                    children[a].append(b)
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            for c in children[node]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return False


Move = tuple[str, str, str]  # (kind, a, b) with kind in add, delete, reverse


class _Constraints:
    def __init__(
        self,
        names: Sequence[str],
        max_parents: int | None,
        required: Iterable[Edge],
        forbidden: Iterable[Edge],
    ) -> None:
        if max_parents is not None and (
            isinstance(max_parents, bool) or not isinstance(max_parents, int) or max_parents < 0
        ):
            raise ValidationError(f"max_parents must be a non-negative int, got {max_parents!r}.")
        self.max_parents = max_parents
        self.required = _edge_set(required, names, "required")
        self.forbidden = _edge_set(forbidden, names, "forbidden")
        clash = self.required & self.forbidden
        if clash:
            raise ValidationError(f"Edges {sorted(clash)} are both required and forbidden.")

    def room(self, graph: _Graph, child: str) -> bool:
        return self.max_parents is None or len(graph.parents[child]) < self.max_parents


def _legal_moves(graph: _Graph, rules: _Constraints) -> list[Move]:
    moves: list[Move] = []
    for a in graph.names:
        for b in graph.names:
            if a == b:
                continue
            if a in graph.parents[b]:
                if (a, b) in rules.required:
                    continue
                moves.append(("delete", a, b))
                if (
                    (b, a) not in rules.forbidden
                    and rules.room(graph, a)
                    and not graph.reaches(a, b, skip=(a, b))
                ):
                    moves.append(("reverse", a, b))
            elif b not in graph.parents[a]:
                if (
                    (a, b) not in rules.forbidden
                    and rules.room(graph, b)
                    and not graph.reaches(b, a)
                ):
                    moves.append(("add", a, b))
    order = {"add": 0, "delete": 1, "reverse": 2}
    names = {n: i for i, n in enumerate(graph.names)}
    return sorted(moves, key=lambda m: (names[m[1]], names[m[2]], order[m[0]]))


def _delta(graph: _Graph, move: Move, scorer: FamilyScorer) -> float:
    """Proposition 1: only the families whose parents change contribute."""
    kind, a, b = move
    pb = graph.parents[b]
    if kind == "add":
        return scorer(b, pb | {a}) - scorer(b, pb)
    gain = scorer(b, pb - {a}) - scorer(b, pb)
    if kind == "delete":
        return gain
    pa = graph.parents[a]
    return gain + scorer(a, pa | {b}) - scorer(a, pa)


def _apply(graph: _Graph, move: Move) -> _Graph:
    kind, a, b = move
    g = graph.copy()
    if kind == "add":
        g.parents[b].add(a)
    elif kind == "delete":
        g.parents[b].discard(a)
    else:
        g.parents[b].discard(a)
        g.parents[a].add(b)
    return g


def _total(graph: _Graph, scorer: FamilyScorer) -> float:
    return math.fsum(scorer(n, graph.parents[n]) for n in graph.names)


def _climb(
    graph: _Graph,
    scorer: FamilyScorer,
    rules: _Constraints,
    tabu_length: int,
    max_iterations: int,
    history: list[float],
) -> tuple[_Graph, float]:
    current, score = graph, _total(graph, scorer)
    best, best_score = current, score
    tabu: list[frozenset[Edge]] = [current.key()]
    stale = 0
    for _ in range(max_iterations):
        candidates = [(m, _delta(current, m, scorer)) for m in _legal_moves(current, rules)]
        if tabu_length == 0:
            improving = [(m, d) for m, d in candidates if d > IMPROVEMENT]
            if not improving:
                break
            move, delta = max(improving, key=lambda md: md[1])  # first of the best (stable)
        else:
            fresh = [(m, d) for m, d in candidates if _apply(current, m).key() not in tabu]
            if not fresh:
                break
            move, delta = max(fresh, key=lambda md: md[1])
        current = _apply(current, move)
        score += delta
        history.append(score)
        if tabu_length:
            tabu = [*tabu, current.key()][-tabu_length:]
        if score > best_score + IMPROVEMENT:
            best, best_score, stale = current, score, 0
        else:
            stale += 1
            if tabu_length and stale >= tabu_length:
                break
    return best, best_score


def hill_climb(
    data: Dataset,
    score: Score = "bic",
    equivalent_sample_size: float = 1.0,
    start: BayesianNetwork | None = None,
    max_parents: int | None = None,
    required: Iterable[Edge] = (),
    forbidden: Iterable[Edge] = (),
    tabu_length: int = 0,
    restarts: int = 0,
    seed: int | None = None,
    max_iterations: int = 1000,
) -> SearchResult:
    """Greedy search over DAGs with add, delete and reverse moves (structure_search.md §3).

    Plain (``tabu_length=0``, ``restarts=0``): take the best strictly improving move
    until none exists, which is a local optimum. ``tabu_length`` > 0 allows the best
    non-improving move to an unvisited graph; ``restarts`` perturbs the best graph by
    random legal moves and climbs again. The best graph found is returned.
    """
    scorer = FamilyScorer(data, score, equivalent_sample_size)
    names = scorer.names
    rules = _Constraints(names, max_parents, required, forbidden)
    for value, name in (
        (tabu_length, "tabu_length"),
        (restarts, "restarts"),
        (max_iterations, "max_iterations"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"{name} must be a non-negative int, got {value!r}.")
    edges: set[Edge] = set(rules.required)
    if start is not None:
        if not isinstance(start, BayesianNetwork) or {v.name for v in start.variables} != set(
            names
        ):
            raise ValidationError("start must be a BayesianNetwork over the data's variables.")
        edges |= set(start.edges())
    try:
        BayesianNetwork(data.variables, sorted(edges))
    except ValidationError as error:
        raise ValidationError(
            f"The start graph with the required edges is invalid: {error}"
        ) from None
    graph = _Graph(names, edges)
    if any(e in rules.forbidden for e in graph.edges()) or any(
        max_parents is not None and len(p) > max_parents for p in graph.parents.values()
    ):
        raise ValidationError("The start graph violates max_parents or the forbidden edges.")

    history: list[float] = [_total(graph, scorer)]
    best, best_score = _climb(graph, scorer, rules, tabu_length, max_iterations, history)
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        perturbed = best
        for _ in range(len(names)):
            moves = _legal_moves(perturbed, rules)
            if not moves:
                break
            perturbed = _apply(perturbed, moves[int(rng.integers(len(moves)))])
        found, found_score = _climb(perturbed, scorer, rules, tabu_length, max_iterations, history)
        if found_score > best_score + IMPROVEMENT:
            best, best_score = found, found_score
    structure = BayesianNetwork(data.variables, best.edges())
    return SearchResult(structure, best_score, tuple(history), scorer.evaluations)


def _edge_set(edges: Iterable[Edge], names: Sequence[str], what: str) -> frozenset[Edge]:
    result = set()
    for edge in edges:
        if not (isinstance(edge, tuple) and len(edge) == 2):
            raise ValidationError(f"{what} edges must be (parent, child) pairs, got {edge!r}.")
        a, b = edge
        for n in (a, b):
            if n not in names:
                raise UnknownNodeError(f"{what} edge {edge} names unknown variable {n!r}.")
        if a == b:
            raise ValidationError(f"{what} edge {edge} is a self-loop.")
        result.add((a, b))
    return frozenset(result)


#: The largest number of variables ``exact_search`` accepts (spec ⚑4).
EXACT_LIMIT = 12


def exact_search(
    data: Dataset,
    score: Score = "bic",
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
) -> SearchResult:
    """The highest-scoring DAG, by dynamic programming over subsets (Proposition 3).

    For each variable, the best parent set within every candidate set; then the best
    score F(S) of a DAG on each subset S, built by choosing S's sink. Needs at most
    n 2^(n-1) family scores, so it is limited to ``EXACT_LIMIT`` variables.
    """
    scorer = FamilyScorer(data, score, equivalent_sample_size)
    names = scorer.names
    n = len(names)
    if n > EXACT_LIMIT:
        raise ValidationError(f"exact_search handles at most {EXACT_LIMIT} variables, got {n}.")
    _Constraints(names, max_parents, (), ())  # validates max_parents
    limit = n if max_parents is None else max_parents

    # best[x][mask]: (score, parent mask) of the best parent set within `mask`, a subset of
    # the other variables expressed as a mask over all n bits (x's own bit never set).
    best: list[list[tuple[float, int]]] = []
    for x in range(n):
        table: list[tuple[float, int]] = [(-math.inf, 0)] * (1 << n)
        for mask in range(1 << n):
            if mask >> x & 1:
                continue
            members = [names[i] for i in range(n) if mask >> i & 1]
            own = (scorer(names[x], members), mask) if len(members) <= limit else (-math.inf, mask)
            candidate = own
            for i in range(n):
                if mask >> i & 1:
                    smaller = table[mask & ~(1 << i)]
                    if smaller[0] > candidate[0]:
                        candidate = smaller
            table[mask] = candidate
        best.append(table)

    # F[S]: the best score of a DAG on S (parents drawn from S), and the sink achieving it.
    total = [-math.inf] * (1 << n)
    sink = [-1] * (1 << n)
    total[0] = 0.0
    for subset in range(1, 1 << n):
        for x in range(n):
            if subset >> x & 1:
                rest = subset & ~(1 << x)
                value = total[rest] + best[x][rest][0]
                if value > total[subset]:
                    total[subset], sink[subset] = value, x
    edges: list[Edge] = []
    subset = (1 << n) - 1
    while subset:
        x = sink[subset]
        rest = subset & ~(1 << x)
        parents = best[x][rest][1]
        edges += [(names[i], names[x]) for i in range(n) if parents >> i & 1]
        subset = rest
    structure = BayesianNetwork(data.variables, edges)
    return SearchResult(structure, total[(1 << n) - 1], (total[(1 << n) - 1],), scorer.evaluations)
