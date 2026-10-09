"""Markov networks: P(x) = (1/Z) Π_k φ_k(x_{C_k}).

The derivations (P10) are in ``docs/mathematics/markov_networks.md``. All
computation is in log space (P9), so Z may be far beyond float64.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from probgraph._assignments import check_partial_assignment, check_query_and_evidence
from probgraph.exceptions import NormalisationError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.factors import DiscreteFactor, LogFactor
from probgraph.graphs import UndirectedGraph
from probgraph.inference.elimination_order import greedy_order
from probgraph.inference.variable_elimination import _eliminate
from probgraph.variables import DiscreteVariable


class MarkovNetwork:
    """An undirected model: a product of nonnegative factors, normalised by Z.

    ``factors`` may be ``DiscreteFactor`` or ``LogFactor`` (for potentials beyond
    float64). A variable that appears in no factor gets the all-ones factor, so it
    is uniform and independent of everything else.

    Invariants:
        M1  every factor's variables are model variables with the model's definitions
        M2  Z > 0; otherwise ``ValidationError`` is raised when Z is first needed
        M3  probability(x) = Π φ(x) / Z, and it sums to 1
        M4  separation in ``graph`` implies conditional independence (P10 Theorem 1)
    """

    def __init__(
        self,
        variables: Sequence[DiscreteVariable],
        factors: Iterable[DiscreteFactor | LogFactor],
    ) -> None:
        if isinstance(variables, DiscreteVariable) or not isinstance(variables, Iterable):
            raise ValidationError(
                f"variables must be a sequence of DiscreteVariable, got {variables!r}."
            )
        variables = tuple(variables)
        for v in variables:
            if not isinstance(v, DiscreteVariable):
                raise ValidationError(f"Expected DiscreteVariable, got {v!r}.")
        names = [v.name for v in variables]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValidationError(f"Model has duplicate variable names: {duplicates}.")
        self._variables = {v.name: v for v in variables}

        given = list(factors)
        for phi in given:
            self._check_factor(phi)
        self._factors = tuple(given)

        logs = [phi if isinstance(phi, LogFactor) else LogFactor.from_factor(phi) for phi in given]
        covered = {name for phi in given for name in phi.names}
        for v in variables:
            if v.name not in covered:
                logs.append(LogFactor([v], np.zeros(v.cardinality)))  # log 1: uniform
        self._log_factors = tuple(logs)

        self._graph = UndirectedGraph(names)
        for phi in given:
            for u, w in itertools.combinations(phi.names, 2):
                self._graph.add_edge(u, w)
        self._log_z: float | None = None

    # -- structure ------------------------------------------------------------

    @property
    def variables(self) -> tuple[DiscreteVariable, ...]:
        return tuple(self._variables.values())

    @property
    def factors(self) -> tuple[DiscreteFactor | LogFactor, ...]:
        """The factors as given (not including the implicit uniform ones)."""
        return self._factors

    @property
    def graph(self) -> UndirectedGraph:
        """A copy of the interaction graph: u — v whenever u and v share a factor."""
        return self._graph.copy()

    def separated(self, xs: Iterable[str], ys: Iterable[str], given: Iterable[str] = ()) -> bool:
        """True if ``given`` separates ``xs`` from ``ys`` in the graph (so X ⊥ Y | Z, by M4)."""
        return self._graph.separated(xs, ys, given)

    # -- the partition function ----------------------------------------------

    def log_partition_function(self) -> float:
        """log Z, computed by log-space variable elimination over every variable."""
        if self._log_z is None:
            log_z = self._log_sum({}, keep=[]).log_total()
            if log_z == -math.inf:
                raise ValidationError(
                    "The partition function is 0: no assignment has positive weight."
                )
            self._log_z = log_z
        return self._log_z

    def partition_function(self) -> float:
        """Z. Raises ``ValidationError`` if Z overflows float64; use ``log_partition_function``."""
        log_z = self.log_partition_function()
        try:
            return math.exp(log_z)
        except OverflowError:
            raise ValidationError(
                f"Z = exp({log_z:.6g}) would overflow float64; use log_partition_function()."
            ) from None

    # -- probabilities ----------------------------------------------------------

    def log_probability(self, assignment: Mapping[str, str]) -> float:
        """log P(x) for a complete assignment x."""
        x = check_partial_assignment(self._variables, assignment, "Assignment")
        missing = [n for n in self._variables if n not in x]
        if missing:
            raise ValidationError(f"Assignment must cover every variable: missing {missing}.")
        log_weight = sum(phi.log_value({n: x[n] for n in phi.names}) for phi in self._log_factors)
        return log_weight - self.log_partition_function()

    def probability(self, assignment: Mapping[str, str]) -> float:
        """P(x). May underflow to 0 for very improbable x; ``log_probability`` does not."""
        return math.exp(self.log_probability(assignment))

    def query(
        self, variables: Sequence[str], evidence: Mapping[str, str] | None = None
    ) -> DiscreteFactor:
        """P(variables | evidence), by log-space variable elimination with min-fill."""
        query, observed = check_query_and_evidence(self._variables, variables, evidence)
        self.log_partition_function()  # M2: reject a model with Z = 0
        joint = self._log_sum(observed, keep=query)
        try:
            return joint.normalise().to_factor().aligned(query)
        except NormalisationError:
            raise ZeroProbabilityEvidenceError(
                f"P(e) = 0 for evidence {observed}; the posterior is undefined."
            ) from None

    def log_probability_of_evidence(self, evidence: Mapping[str, str]) -> float:
        """log P(e) = log Z(e) - log Z; -inf exactly when e is impossible."""
        observed = check_partial_assignment(self._variables, evidence)
        log_z = self.log_partition_function()
        return self._log_sum(observed, keep=[]).log_total() - log_z

    # -- internals --------------------------------------------------------------

    def _log_sum(self, evidence: Mapping[str, str], keep: Sequence[str]) -> LogFactor:
        """log Σ over every unobserved variable not in ``keep`` of Π φ[e]."""
        reduced = [phi.reduce(evidence) for phi in self._log_factors]
        graph = self._graph.subgraph(n for n in self._variables if n not in evidence)
        eliminate = [n for n in graph.nodes() if n not in keep]
        cards = {name: v.cardinality for name, v in self._variables.items()}
        order = greedy_order(graph, eliminate, "min_fill", cards)
        return _eliminate(reduced, order, LogFactor.unit())

    def _check_factor(self, phi: object) -> None:
        if not isinstance(phi, (DiscreteFactor, LogFactor)):
            raise ValidationError(f"Expected DiscreteFactor or LogFactor, got {phi!r}.")
        for v in phi.variables:
            if v.name not in self._variables:
                raise ValidationError(
                    f"Factor over {phi.names} uses {v.name!r}, which is not a model variable."
                )
            canonical = self._variables[v.name]
            if v != canonical:
                raise ValidationError(
                    f"Factor declares {v.name!r} with domain {v.states}, but the model's "
                    f"{v.name!r} has domain {canonical.states}."
                )

    def __repr__(self) -> str:
        return (
            f"MarkovNetwork(variables={list(self._variables)}, "
            f"factors={len(self._factors)}, edges={len(self._graph.edges())})"
        )
