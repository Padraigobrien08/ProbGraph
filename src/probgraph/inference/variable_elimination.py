"""Exact inference by variable elimination: sum-product, and max-product for the MPE.

The correctness proofs are in ``docs/mathematics/variable_elimination.md`` (P6) and
``docs/mathematics/max_product.md`` (P20).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from probgraph._assignments import check_partial_assignment, check_query_and_evidence
from probgraph.exceptions import (
    NormalisationError,
    UnknownNodeError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)
from probgraph.factors import DiscreteFactor, LogFactor
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
    from probgraph.variables import DiscreteVariable

Space = Literal["probability", "log"]
SPACES: tuple[Space, ...] = ("probability", "log")

_Table = TypeVar("_Table", DiscreteFactor, LogFactor)


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
        self._log_factors = tuple(LogFactor.from_factor(phi) for phi in self._factors)
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
        prune_evidence: bool = False,
        space: Space = "log",
    ) -> DiscreteFactor:
        """Return P(variables | evidence) as a normalised factor over ``variables``.

        By default (``space="log"``) every step works on log factors
        (log_space.md §8). This avoids underflow, so tiny-but-possible evidence is
        never mistaken for impossible evidence, and never silently corrupts the
        posterior. ``space="probability"`` is the M2 behaviour: slightly faster, and
        identical wherever it does not underflow.

        With ``prune_barren`` (the default), only the CPDs of An*(query ∪ evidence)
        are used (Proposition 5, V7). The answer is identical either way.

        With ``prune_evidence`` (opt-in), observations d-separated from the query
        are dropped first, which can enable more barren pruning (d_separation.md §9).
        Caveat: if the evidence is impossible only because of a dropped observation,
        the answer given the requisite evidence is returned instead of raising
        ``ZeroProbabilityEvidenceError``.
        """
        query, observed = self._check_query_and_evidence(variables, evidence)
        _check_space(space)
        order, kept, used = self._plan(
            query, observed, elimination_order, prune_barren, prune_evidence
        )
        try:
            if space == "log":
                log_factors = self._select(self._log_factors, kept, used)
                log_joint = _eliminate(log_factors, order, LogFactor.unit())
                return log_joint.normalise().to_factor().aligned(query)
            factors = self._select(self._factors, kept, used)
            joint = _eliminate(factors, order, DiscreteFactor.unit())  # P(Q, e)
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
        space: Space = "log",
    ) -> float:
        """Return P(e), eliminating every unobserved variable.

        Impossible evidence returns 0.0. Unlike ``query``, this does not raise.
        By default (``space="log"``) the result is exp(log P(e)), which is exact up to
        rounding but can underflow to 0.0 for tiny P(e). Use
        ``log_probability_of_evidence`` when that matters.
        """
        _check_space(space)
        if space == "log":
            return math.exp(
                self.log_probability_of_evidence(evidence, elimination_order, prune_barren)
            )
        observed = check_partial_assignment(self._variables, evidence)
        # P(e) depends on every observation, so evidence is never dropped here.
        order, kept, used = self._plan([], observed, elimination_order, prune_barren, False)
        factors = self._select(self._factors, kept, used)
        return _eliminate(factors, order, DiscreteFactor.unit()).total()

    def log_probability_of_evidence(
        self,
        evidence: Mapping[str, str],
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        prune_barren: bool = True,
    ) -> float:
        """Return log P(e), computed entirely in log space.

        Returns -inf exactly when the evidence is structurally impossible: in log
        space a -inf cannot arise from underflow (log_space.md §8).
        """
        observed = check_partial_assignment(self._variables, evidence)
        order, kept, used = self._plan([], observed, elimination_order, prune_barren, False)
        log_factors = self._select(self._log_factors, kept, used)
        return _eliminate(log_factors, order, LogFactor.unit()).log_total()

    def most_probable_explanation(
        self,
        evidence: Mapping[str, str] | None = None,
        elimination_order: Sequence[str] | Heuristic = "min_fill",
    ) -> tuple[dict[str, str], float]:
        """Return ``(x*, log P(x*, e))``, where x* maximises P(x, e) over every unobserved variable.

        Max-product (max-sum, in log space) variable elimination followed by a
        traceback (max_product.md §3–4). Every elimination order gives the same
        value. Among several maximisers, each traceback step takes the first state
        that attains the maximum. Barren variables are never pruned: unlike a sum,
        max_x P(x | u) depends on u (max_product.md §5). Raises
        ``ZeroProbabilityEvidenceError`` when P(e) = 0.
        """
        observed = check_partial_assignment(self._variables, evidence or {})
        order, kept, used = self._plan([], observed, elimination_order, False, False)
        remaining = self._select(self._log_factors, kept, used)
        records: list[tuple[str, tuple[DiscreteVariable, ...], np.ndarray]] = []
        for z in order:
            involved = [phi for phi in remaining if z in phi.scope]
            remaining = [phi for phi in remaining if z not in phi.scope]
            psi = _product(involved, LogFactor.unit())
            axis = psi.names.index(z)
            rest = tuple(v for v in psi.variables if v.name != z)
            records.append((z, rest, np.argmax(psi.log_values, axis=axis)))  # first maximiser
            remaining.append(LogFactor._trusted(rest, np.max(psi.log_values, axis=axis)))
        log_max = _product(remaining, LogFactor.unit()).log_total()
        if log_max == -math.inf:
            raise ZeroProbabilityEvidenceError(
                f"P(e) = 0 for evidence {observed}; every assignment is impossible."
            )
        chosen: dict[str, int] = {}
        for z, rest, best in reversed(records):  # Proposition 1: later choices come first
            chosen[z] = int(best[tuple(chosen[v.name] for v in rest)])
        assignment = {
            name: self._variables[name].states[chosen[name]]
            for name in self._variables
            if name in chosen
        }
        return assignment, log_max

    # -- inspection ---------------------------------------------------------

    def requisite_evidence(
        self, variables: Sequence[str], evidence: Mapping[str, str]
    ) -> dict[str, str]:
        """The observations that can affect P(variables | evidence) (Proposition 6)."""
        query, observed = self._check_query_and_evidence(variables, evidence)
        requisite = self._graph.requisite_evidence(query, observed)
        return {name: state for name, state in observed.items() if name in requisite}

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
        prune_evidence: bool = False,
    ) -> list[str]:
        """The order ``heuristic`` chooses for this query. Ties go to model declaration order."""
        query, observed = self._check_query_and_evidence(variables, evidence)
        if not isinstance(heuristic, str):
            raise ValidationError(f"heuristic must be one of {HEURISTICS}, got {heuristic!r}.")
        return self._plan(query, observed, heuristic, prune_barren, prune_evidence)[0]

    def query_trace(
        self,
        variables: Sequence[str],
        evidence: Mapping[str, str] | None = None,
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        prune_barren: bool = True,
        prune_evidence: bool = False,
        space: Space = "log",
    ) -> EliminationTrace:
        """The intermediate tables that ``query`` with the same arguments would build.

        The trace is the same in either space: the order and the scopes depend
        only on the graph.
        """
        query, observed = self._check_query_and_evidence(variables, evidence)
        _check_space(space)
        order, kept, used = self._plan(
            query, observed, elimination_order, prune_barren, prune_evidence
        )
        steps: list[EliminationStep] = []
        if space == "log":
            _eliminate(self._select(self._log_factors, kept, used), order, LogFactor.unit(), steps)
        else:
            _eliminate(self._select(self._factors, kept, used), order, DiscreteFactor.unit(), steps)
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
        reduced = [phi.reduce(observed) for phi in self._factors]
        _eliminate(reduced, names, DiscreteFactor.unit(), steps)
        return EliminationTrace(tuple(steps))

    # -- internals ----------------------------------------------------------

    def _plan(
        self,
        query: Sequence[str],
        observed: Mapping[str, str],
        elimination_order: Sequence[str] | Heuristic,
        prune_barren: bool,
        prune_evidence: bool,
    ) -> tuple[list[str], list[int], dict[str, str]]:
        """Choose the elimination order, the factors to keep, and the evidence to apply.

        Returns ``(order, kept, used)``: ``kept`` indexes into the model's factors
        (in either space), and ``used`` is the evidence to reduce them by.
        """
        anchor = [*query, *observed]
        if prune_evidence and query:
            # Proposition 6: only requisite observations need to hold the ancestral set open.
            anchor = [*query, *self._graph.requisite_evidence(query, observed)]
        keep = self._graph.ancestral_set(anchor) if prune_barren else set(self._variables)
        # A dropped observation inside `keep` stays observed: still exact (weak union +
        # decomposition) and cheaper than eliminating it. See d_separation.md §9.
        used = {name: state for name, state in observed.items() if name in keep}
        # Each factor is the CPD of its first variable; keep it if that variable is kept.
        kept = [i for i, phi in enumerate(self._factors) if phi.names[0] in keep]
        factors = self._select(self._factors, kept, used)

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
        return order, kept, used

    @staticmethod
    def _select(
        source: Sequence[_Table], kept: Sequence[int], used: Mapping[str, str]
    ) -> list[_Table]:
        return [source[i].reduce(used) for i in kept]

    def _check_query_and_evidence(
        self, variables: Sequence[str], evidence: Mapping[str, str] | None
    ) -> tuple[list[str], dict[str, str]]:
        return check_query_and_evidence(self._variables, variables, evidence)

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
    factors: Sequence[_Table],
    order: Sequence[str],
    unit: _Table,
    trace: list[EliminationStep] | None = None,
) -> _Table:
    """Return Σ_order Π factors, summing out one variable at a time (P6 §2).

    The same code runs in either space: for log factors, × is + and Σ is
    log-sum-exp (log_space.md §5).
    """
    remaining = list(factors)
    for z in order:
        involved = [phi for phi in remaining if z in phi.scope]
        # Keep the factors that do not mention z: the lemma moves the sum past them.
        remaining = [phi for phi in remaining if z not in phi.scope]
        psi = _product(involved, unit)
        if trace is not None:
            size = math.prod(v.cardinality for v in psi.variables)
            trace.append(EliminationStep(z, psi.scope, size))
        remaining.append(psi.marginalise([z]))
    return _product(remaining, unit)


def _product(factors: Iterable[_Table], unit: _Table) -> _Table:
    return reduce(mul, factors, unit)


def _check_space(space: object) -> None:
    if space not in SPACES:
        raise ValidationError(f"space must be one of {SPACES}, got {space!r}.")
