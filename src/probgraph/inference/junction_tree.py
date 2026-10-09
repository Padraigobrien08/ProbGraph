"""Exact inference by Shafer–Shenoy message passing on a clique tree, in log space.

The proof that beliefs equal marginals (P12) is in
``docs/mathematics/message_passing.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul

import numpy as np

from probgraph._assignments import check_query
from probgraph.exceptions import UnknownNodeError, ValidationError
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
        elimination_order: Sequence[str] | Heuristic = "min_fill",
        root: int = 0,
    ) -> None:
        if isinstance(model, BayesianNetwork):
            model = model.to_markov_network()
        if not isinstance(model, MarkovNetwork):
            raise ValidationError(
                f"Expected a BayesianNetwork or MarkovNetwork, got {type(model).__name__}."
            )
        self._variables = {v.name: v for v in model.variables}
        graph = model.graph
        cards = {name: v.cardinality for name, v in self._variables.items()}

        if isinstance(elimination_order, str):
            if elimination_order not in HEURISTICS:
                raise ValidationError(
                    f"Unknown elimination heuristic {elimination_order!r}; "
                    f"expected one of {HEURISTICS}."
                )
            order = greedy_order(graph, graph.nodes(), elimination_order, cards)
        else:
            order = list(elimination_order)
        self._tree = CliqueTree.from_elimination(graph, order)

        cliques = self._tree.cliques
        if not 0 <= root < max(len(cliques), 1):
            raise ValidationError(f"root must index a clique (0..{len(cliques) - 1}), got {root}.")

        factors = [
            phi if isinstance(phi, LogFactor) else LogFactor.from_factor(phi)
            for phi in model.factors
        ]
        home = self._tree.assign([phi.names for phi in factors])
        potentials = []
        for i, clique in enumerate(cliques):
            # The all-ones factor gives ψ_i exactly the scope C_i (message_passing.md §1).
            ones = LogFactor(
                [v for name, v in self._variables.items() if name in clique],
                np.zeros(tuple(v.cardinality for n, v in self._variables.items() if n in clique)),
            )
            assigned = [phi for phi, h in zip(factors, home, strict=True) if h == i]
            potentials.append(reduce(mul, assigned, ones))

        self._beliefs, self._message_count = _calibrate(self._tree, potentials, root)
        self._log_z = self._beliefs[0].log_total() if self._beliefs else 0.0
        if self._log_z == -np.inf:
            raise ValidationError("The partition function is 0: no assignment has positive weight.")

    # -- structure ----------------------------------------------------------------

    @property
    def tree(self) -> CliqueTree:
        return self._tree

    def message_count(self) -> int:
        """The number of messages computed during calibration: 2(k - 1) for k cliques."""
        return self._message_count

    def clique_belief(self, i: int) -> LogFactor:
        """β_i, proportional to the joint marginal of clique i (C3)."""
        if not 0 <= i < len(self._beliefs):
            raise ValidationError(f"No clique {i}; the tree has {len(self._beliefs)}.")
        return self._beliefs[i]

    def log_partition_function(self) -> float:
        """log Z: 0 for a Bayesian network; log Σ β_i for any clique i (C4)."""
        return self._log_z

    # -- queries --------------------------------------------------------------------

    def marginal(self, variable: str) -> DiscreteFactor:
        """P(variable), from the smallest clique that contains it."""
        if variable not in self._variables:
            raise UnknownNodeError(f"Unknown variable {variable!r}.")
        return self.query([variable])

    def marginals(self) -> dict[str, DiscreteFactor]:
        """P(x) for every variable, all from the one calibration."""
        return {name: self.marginal(name) for name in self._variables}

    def query(self, variables: Sequence[str]) -> DiscreteFactor:
        """P(variables) for variables that lie together in one clique (spec ⚑6)."""
        query = check_query(self._variables, variables)
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


def _calibrate(
    tree: CliqueTree, potentials: Sequence[LogFactor], root: int
) -> tuple[list[LogFactor], int]:
    """Run Shafer–Shenoy: an inward sweep to ``root``, then an outward sweep.

    ``potentials[i]`` must have scope exactly ``tree.cliques[i]``. Returns the
    beliefs and the number of messages computed.
    """
    cliques = tree.cliques
    if len(potentials) != len(cliques) or any(
        psi.scope != c for psi, c in zip(potentials, cliques, strict=True)
    ):
        raise ValidationError("Each potential must have exactly its clique's scope.")
    if not cliques:
        return [], 0

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
    return beliefs, len(messages)
