"""Exact inference by sum-product variable elimination.

The correctness proof (P6) is in ``docs/mathematics/variable_elimination.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

from probgraph._assignments import check_partial_assignment
from probgraph.exceptions import (
    NormalisationError,
    UnknownNodeError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)
from probgraph.factors import DiscreteFactor
from probgraph.graphs import UndirectedGraph, interaction_graph
from probgraph.inference.elimination_order import (
    HEURISTICS,
    EliminationStep,
    EliminationTrace,
    Heuristic,
    greedy_order,
)

if TYPE_CHECKING:
    from probgraph.models import BayesianNetwork


class VariableElimination:
    """Answers P(Q | e) and P(e) exactly, without building the joint distribution.

    The model is validated, and its CPD factors are recorded, when the engine is
    created. Later ``add_cpd`` calls on the model do not affect an existing
    engine. This matches ``AncestralSampler``.

    ``elimination_order`` is either a heuristic name from ``HEURISTICS`` (the
    default is ``"min_fill"``), or an explicit list of exactly the variables
    outside the query and evidence, each once. The order changes the cost
    (``elimination_cost``) but never the answer.

    Invariants:
        V1  results equal brute-force enumeration
        V2  every valid elimination order gives the same result
        V3  ``query`` returns a normalised factor whose scope is exactly the
            query variables, in the order requested
        V4  query and evidence names are known, disjoint and have valid states
        V5  evidence with P(e) = 0 raises ``ZeroProbabilityEvidenceError``
        V6  with no evidence, ``query`` returns the prior marginal, and
            ``probability_of_evidence({})`` is 1
        V7  barren-node pruning (``prune_barren``, on by default) never changes
            an answer
    """

    def __init__(self, model: BayesianNetwork) -> None:
        self._factors = model.factors()  # validates the model
        self._graph = model.graph  # a copy; used for ancestral pruning
        self._variables = {v.name: v for v in model.variables}
        self._cardinalities = {v.name: v.cardinality for v in model.variables}

    # -- queries ------------------------------------------------------------

    def query(
        self,
        variables: Sequence[str],
        evidence: Mapping[str, str] | None = None,
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        prune_barren: bool = True,
    ) -> DiscreteFactor:
        """Return P(variables | evidence) as a normalised factor over ``variables``.

        With ``prune_barren`` (the default), only the CPDs of An*(query ∪ evidence)
        are used (Proposition 5, V7). The answer is identical either way.
        """
        query, observed = self._check_query_and_evidence(variables, evidence)
        factors, order = self._plan(query, observed, elimination_order, prune_barren)
        joint = _eliminate(factors, order)  # P(Q, e), with scope exactly Q
        try:
            return joint.normalise().aligned(query)
        except NormalisationError:
            raise ZeroProbabilityEvidenceError(
                f"P(e) = 0 for evidence {observed}; the posterior is undefined."
            ) from None

    def probability_of_evidence(
        self,
        evidence: Mapping[str, str],
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        prune_barren: bool = True,
    ) -> float:
        """Return P(e), eliminating every unobserved variable.

        Impossible evidence returns 0.0. Unlike ``query``, this does not raise.
        """
        observed = check_partial_assignment(self._variables, evidence)
        factors, order = self._plan([], observed, elimination_order, prune_barren)
        return _eliminate(factors, order).total()

    # -- inspection ---------------------------------------------------------

    def barren_variables(
        self, variables: Sequence[str], evidence: Mapping[str, str] | None = None
    ) -> set[str]:
        """The variables that pruning removes for this query: V minus An*(query ∪ evidence)."""
        query, observed = self._check_query_and_evidence(variables, evidence)
        return set(self._variables) - self._graph.ancestral_set([*query, *observed])

    def elimination_order(
        self,
        variables: Sequence[str],
        evidence: Mapping[str, str] | None = None,
        heuristic: Heuristic = "min_fill",
        prune_barren: bool = True,
    ) -> list[str]:
        """The order ``heuristic`` chooses for this query. Ties go to model declaration order."""
        query, observed = self._check_query_and_evidence(variables, evidence)
        if not isinstance(heuristic, str):
            raise ValidationError(f"heuristic must be one of {HEURISTICS}, got {heuristic!r}.")
        return self._plan(query, observed, heuristic, prune_barren)[1]

    def query_trace(
        self,
        variables: Sequence[str],
        evidence: Mapping[str, str] | None = None,
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        prune_barren: bool = True,
    ) -> EliminationTrace:
        """The intermediate tables that ``query`` with the same arguments would build."""
        query, observed = self._check_query_and_evidence(variables, evidence)
        factors, order = self._plan(query, observed, elimination_order, prune_barren)
        steps: list[EliminationStep] = []
        _eliminate(factors, order, steps)
        return EliminationTrace(tuple(steps))

    def elimination_cost(
        self, order: Sequence[str], evidence: Mapping[str, str] | None = None
    ) -> EliminationTrace:
        """Run the elimination of ``order`` on *all* CPD factors and record each table.

        ``order`` can be any sequence of distinct unobserved variables. Whatever
        is not eliminated stays in the final factor. No pruning is applied. The
        trace is *measured* from the factors themselves; ``simulate_elimination``
        predicts it from the graph (invariant V8).
        """
        observed = check_partial_assignment(self._variables, evidence or {})
        if isinstance(order, str):
            raise ValidationError(
                f"Elimination order must be a sequence of names, not the string {order!r}."
            )
        names = list(order)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValidationError(f"Elimination order has duplicate variables {duplicates}.")
        unknown = [n for n in names if n not in self._variables]
        if unknown:
            raise UnknownNodeError(f"Elimination order names unknown variables {unknown}.")
        clash = [n for n in names if n in observed]
        if clash:
            raise ValidationError(f"Cannot eliminate observed variables {clash}.")
        steps: list[EliminationStep] = []
        _eliminate([phi.reduce(observed) for phi in self._factors], names, steps)
        return EliminationTrace(tuple(steps))

    # -- internals ----------------------------------------------------------

    def _plan(
        self,
        query: Sequence[str],
        observed: Mapping[str, str],
        elimination_order: Sequence[str] | Heuristic,
        prune_barren: bool,
    ) -> tuple[list[DiscreteFactor], list[str]]:
        """Choose the factors to use and the order in which to eliminate."""
        if prune_barren:
            keep = self._graph.ancestral_set([*query, *observed])
        else:
            keep = set(self._variables)
        # Each factor is the CPD of its first variable; keep it if that variable is kept.
        factors = [phi.reduce(observed) for phi in self._factors if phi.names[0] in keep]

        if isinstance(elimination_order, str):
            if elimination_order not in HEURISTICS:
                raise ValidationError(
                    f"Unknown elimination heuristic {elimination_order!r}; "
                    f"expected one of {HEURISTICS}."
                )
            graph = interaction_graph(factors)
            graph = UndirectedGraph([n for n in self._variables if n in graph], graph.edges())
            eliminate = [n for n in graph.nodes() if n not in query]
            order = greedy_order(graph, eliminate, elimination_order, self._cardinalities)
        else:
            # An explicit order lists every unobserved non-query variable; pruned ones are skipped.
            full = self._check_order(elimination_order, exclude=[*query, *observed])
            order = [n for n in full if n in keep]
        return factors, order

    def _check_query_and_evidence(
        self, variables: Sequence[str], evidence: Mapping[str, str] | None
    ) -> tuple[list[str], dict[str, str]]:
        query = self._check_query(variables)
        observed = check_partial_assignment(self._variables, evidence or {})
        both = [name for name in query if name in observed]
        if both:
            raise ValidationError(f"Variables {both} are both queried and observed.")
        return query, observed

    def _check_query(self, variables: Sequence[str]) -> list[str]:
        if isinstance(variables, str):
            raise ValidationError(
                f"Query must be a sequence of variable names, not the string {variables!r}."
            )
        query = list(variables)
        if not query:
            raise ValidationError("Query must name at least one variable.")
        duplicates = sorted({n for n in query if query.count(n) > 1})
        if duplicates:
            raise ValidationError(f"Query has duplicate variables {duplicates}.")
        unknown = [n for n in query if n not in self._variables]
        if unknown:
            raise UnknownNodeError(f"Query names unknown variables {unknown}.")
        return query

    def _check_order(self, order: Sequence[str], exclude: Iterable[str]) -> list[str]:
        excluded = set(exclude)
        expected = [n for n in self._variables if n not in excluded]
        given = list(order)
        duplicates = sorted({n for n in given if given.count(n) > 1})
        if duplicates:
            raise ValidationError(f"Elimination order has duplicate variables {duplicates}.")
        missing = [n for n in expected if n not in given]
        unexpected = [n for n in given if n not in expected]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise ValidationError(
                "Elimination order must list each variable outside the query and evidence "
                "exactly once: " + "; ".join(details) + "."
            )
        return given


def _eliminate(
    factors: Iterable[DiscreteFactor],
    order: Sequence[str],
    trace: list[EliminationStep] | None = None,
) -> DiscreteFactor:
    """Return Σ_order Π factors, summing out one variable at a time (P6 §2)."""
    remaining = list(factors)
    for z in order:
        involved = [phi for phi in remaining if z in phi.scope]
        # Keep the factors that do not mention z: the lemma moves the sum past them.
        remaining = [phi for phi in remaining if z not in phi.scope]
        psi = _product(involved)
        if trace is not None:
            trace.append(EliminationStep(z, psi.scope, psi.values.size))
        remaining.append(psi.marginalise([z]))
    return _product(remaining)


def _product(factors: Iterable[DiscreteFactor]) -> DiscreteFactor:
    return reduce(mul, factors, DiscreteFactor.unit())
