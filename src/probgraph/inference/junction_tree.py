"""Exact inference by Shafer–Shenoy message passing on a clique tree, in log space.

The proof that beliefs equal marginals (P12) is in
``docs/mathematics/message_passing.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
from operator import mul
from typing import NamedTuple

import numpy as np

from probgraph._assignments import check_partial_assignment, check_query
from probgraph.exceptions import UnknownNodeError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.factors import DiscreteFactor, LogFactor
from probgraph.inference.clique_tree import CliqueTree
from probgraph.inference.elimination_order import HEURISTICS, Heuristic, greedy_order
from probgraph.models import BayesianNetwork, MarkovNetwork


class JunctionTree:
    """A calibrated clique tree: every marginal of the model from one pass of messages.

    The model (a ``BayesianNetwork`` or ``MarkovNetwork``) is recorded when the
    tree is built; later changes to it have no effect. Everything is computed in
    log space.

    Invariants (message_passing.md):
        C1  exactly 2(k - 1) messages are computed, each once
        C2  neighbouring beliefs agree on their separator
        C3  each normalised clique belief is the joint marginal of its clique
        C5  every single-variable marginal is exact
        C6  the root and the elimination order do not change any result
    """

    def __init__(
        self,
        model: BayesianNetwork | MarkovNetwork,
        evidence: Mapping[str, str] | None = None,
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        root: int = 0,
    ) -> None:
        is_bayesian_network = isinstance(model, BayesianNetwork)
        if isinstance(model, BayesianNetwork):
            model = model.to_markov_network()
        if not isinstance(model, MarkovNetwork):
            raise ValidationError(
                f"Expected a BayesianNetwork or MarkovNetwork, got {type(model).__name__}."
            )
        self._variables = {v.name: v for v in model.variables}
        self._observed = check_partial_assignment(self._variables, evidence or {})
        # The evidence-free log Z. For a Bayesian network it is exactly 0 (P2), and computing
        # it anyway would mean eliminating the whole model without evidence.
        self._log_z = 0.0 if is_bayesian_network else model.log_partition_function()

        # Reduce first: the observed variables then leave the graph (message_passing.md §8).
        factors = [
            (phi if isinstance(phi, LogFactor) else LogFactor.from_factor(phi)).reduce(
                self._observed
            )
            for phi in model.factors
        ]
        graph = model.graph.subgraph(n for n in self._variables if n not in self._observed)
        cards = {name: v.cardinality for name, v in self._variables.items()}
        if isinstance(elimination_order, str):
            if elimination_order not in HEURISTICS:
                raise ValidationError(
                    f"Unknown elimination heuristic {elimination_order!r}; "
                    f"expected one of {HEURISTICS}."
                )
            order = greedy_order(graph, graph.nodes(), elimination_order, cards)
        else:
            order = list(elimination_order)  # must list exactly the unobserved variables
        self._tree = CliqueTree.from_elimination(graph, order)

        cliques = self._tree.cliques
        if not 0 <= root < max(len(cliques), 1):
            raise ValidationError(f"root must index a clique (0..{len(cliques) - 1}), got {root}.")

        if not cliques:  # everything observed: Z(e) is the product of the scalars
            self._beliefs: list[LogFactor] = []
            self._stats = _CalibrationStats(0, 0)
            self._log_z_e = sum(phi.log_total() for phi in factors)
            return

        home = self._tree.assign([phi.names for phi in factors])
        potentials = []
        for i, clique in enumerate(cliques):
            # The all-ones factor gives ψ_i exactly the scope C_i (message_passing.md §1).
            members = [v for name, v in self._variables.items() if name in clique]
            ones = LogFactor(members, np.zeros(tuple(v.cardinality for v in members)))
            assigned = [phi for phi, h in zip(factors, home, strict=True) if h == i]
            potentials.append(reduce(mul, assigned, ones))

        self._beliefs, self._stats = _calibrate(self._tree, potentials, root)
        self._log_z_e = self._beliefs[0].log_total()  # C4: the same for every clique

    # -- structure ----------------------------------------------------------------

    @property
    def tree(self) -> CliqueTree:
        return self._tree

    def message_count(self) -> int:
        """The number of messages computed during calibration: 2(k - 1) for k cliques."""
        return self._stats.messages

    def calibration_cost(self) -> int:
        """Table cells processed: |C_i| per message from clique i, plus |C_i| per belief (§9)."""
        return self._stats.cells

    def clique_belief(self, i: int) -> LogFactor:
        """β_i, proportional to the joint marginal of clique i (C3)."""
        if not 0 <= i < len(self._beliefs):
            raise ValidationError(f"No clique {i}; the tree has {len(self._beliefs)}.")
        return self._beliefs[i]

    def log_partition_function(self) -> float:
        """log Z of the model without evidence: 0 for a Bayesian network."""
        return self._log_z

    def log_probability_of_evidence(self) -> float:
        """log P(e) = log Z(e) - log Z; -inf exactly when the evidence is impossible (C8)."""
        return self._log_z_e - self._log_z

    # -- queries --------------------------------------------------------------------

    def marginal(self, variable: str) -> DiscreteFactor:
        """P(variable | e), from the smallest clique that contains it.

        For an observed variable this is the point mass at its observed state.
        """
        if variable not in self._variables:
            raise UnknownNodeError(f"Unknown variable {variable!r}.")
        if variable in self._observed:
            self._require_possible_evidence()
            v = self._variables[variable]
            point = np.zeros(v.cardinality)
            point[v.index_of(self._observed[variable])] = 1.0
            return DiscreteFactor([v], point)
        return self.query([variable])

    def marginals(self) -> dict[str, DiscreteFactor]:
        """P(x | e) for every unobserved variable, all from the one calibration."""
        self._require_possible_evidence()
        return {name: self.query([name]) for name in self._variables if name not in self._observed}

    def query(self, variables: Sequence[str]) -> DiscreteFactor:
        """P(variables) for variables that lie together in one clique (spec ⚑6)."""
        query = check_query(self._variables, variables)
        both = [name for name in query if name in self._observed]
        if both:
            raise ValidationError(f"Variables {both} are both queried and observed.")
        self._require_possible_evidence()
        wanted = set(query)
        candidates = [i for i, c in enumerate(self._tree.cliques) if wanted <= c]
        if not candidates:
            raise ValidationError(
                f"{query} do not lie together in one clique of the junction tree; "
                "use VariableElimination for this query."
            )
        i = min(candidates, key=lambda j: len(self._tree.cliques[j]))
        belief = self._beliefs[i].marginalise(self._tree.cliques[i] - wanted)
        return belief.normalise().to_factor().aligned(query)

    def _require_possible_evidence(self) -> None:
        if self._log_z_e == -np.inf:
            raise ZeroProbabilityEvidenceError(
                f"P(e) = 0 for evidence {self._observed}; the posterior is undefined."
            )


class _CalibrationStats(NamedTuple):
    messages: int
    cells: int


def _calibrate(
    tree: CliqueTree, potentials: Sequence[LogFactor], root: int
) -> tuple[list[LogFactor], _CalibrationStats]:
    """Run Shafer–Shenoy: an inward sweep to ``root``, then an outward sweep.

    ``potentials[i]`` must have scope exactly ``tree.cliques[i]``. Returns the
    beliefs, and the number of messages and table cells processed (§9).
    """
    cliques = tree.cliques
    if len(potentials) != len(cliques) or any(
        psi.scope != c for psi, c in zip(potentials, cliques, strict=True)
    ):
        raise ValidationError("Each potential must have exactly its clique's scope.")
    if not cliques:
        return [], _CalibrationStats(0, 0)
    size = [int(np.prod([v.cardinality for v in psi.variables])) for psi in potentials]

    # Parent pointers and a preorder from the root.
    parent: dict[int, int | None] = {root: None}
    preorder = [root]
    for i in preorder:
        for j in tree.neighbours(i):
            if j not in parent:
                parent[j] = i
                preorder.append(j)

    messages: dict[tuple[int, int], LogFactor] = {}

    def send(i: int, j: int) -> None:
        incoming = [messages[(k, i)] for k in tree.neighbours(i) if k != j]
        product = reduce(mul, incoming, potentials[i])
        messages[(i, j)] = product.marginalise(cliques[i] - cliques[j])

    for i in reversed(preorder):  # inward: children before parents
        p = parent[i]
        if p is not None:
            send(i, p)
    for i in preorder:  # outward: parents before children
        for j in tree.neighbours(i):
            if parent.get(j) == i:
                send(i, j)

    beliefs = [
        reduce(mul, [messages[(k, i)] for k in tree.neighbours(i)], potentials[i])
        for i in range(len(cliques))
    ]
    cells = sum(size[i] for i, _ in messages) + sum(size)
    return beliefs, _CalibrationStats(len(messages), cells)
