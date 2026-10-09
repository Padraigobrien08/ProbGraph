"""Discrete Bayesian networks: a DAG plus one CPD per node.

The product of the CPDs is a valid joint distribution because of the result
proved in ``docs/mathematics/joint_normalisation.md`` (P2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from probgraph.distributions import TabularCPD
from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.factors import DiscreteFactor
from probgraph.graphs import DAG
from probgraph.variables import DiscreteVariable


class BayesianNetwork:
    """The joint distribution p(x) = prod_i p(x_i | pa(x_i)) over a fixed DAG.

    The variables and edges are fixed when the network is constructed. CPDs
    are attached afterwards with ``add_cpd``. Adding a CPD for a node that
    already has one replaces it. Any ``add_cpd`` call marks the model as
    unvalidated, and the next query validates it again.

    Invariants:
        B1  every node has exactly one CPD (checked by ``validate``)
        B2  the CPD's parent set equals the node's parent set in the graph.
            The CPD keeps its own axis order.
        B3  ``joint_probability`` is the product of the local conditionals
        B4  the joint sums to 1, and B5 every value is >= 0 (both follow from
            B1, B2, C1 and C2 by P2)
        B6  every variable in every CPD, child or parent, is identical to the
            model's own definition of that variable (same name, same ordered
            domain)
        B7  queries must assign exactly one valid state to every variable
    B2 and B6 are checked in ``add_cpd``, so an invalid CPD is never stored.
    """

    def __init__(
        self,
        variables: Sequence[DiscreteVariable],
        edges: Iterable[tuple[str, str]],
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

        self._variables: dict[str, DiscreteVariable] = {v.name: v for v in variables}
        self._dag = DAG(nodes=names, edges=edges)
        self._cpds: dict[str, TabularCPD] = {}
        # The topological order is cached by validate(). None means the model
        # has not been validated since it last changed.
        self._order: tuple[str, ...] | None = None

    # -- structure ----------------------------------------------------------

    @property
    def variables(self) -> tuple[DiscreteVariable, ...]:
        """The model's variables, in declaration order."""
        return tuple(self._variables.values())

    def variable(self, name: str) -> DiscreteVariable:
        try:
            return self._variables[name]
        except KeyError:
            raise UnknownNodeError(f"Model has no variable {name!r}.") from None

    def parents(self, name: str) -> set[str]:
        return self._dag.parents(name)

    def edges(self) -> tuple[tuple[str, str], ...]:
        return self._dag.edges()

    @property
    def graph(self) -> DAG:
        """A copy of the model's graph. Changing the copy does not affect the model."""
        return DAG(nodes=self._dag.nodes(), edges=self._dag.edges())

    @property
    def n_free_parameters(self) -> int:
        """sum_i (|X_i| - 1) * prod_{j in pa(i)} |X_j|. Depends only on the structure (P4)."""
        total = 0
        for name, variable in self._variables.items():
            columns = 1
            for parent in self._dag.parents(name):
                columns *= self._variables[parent].cardinality
            total += (variable.cardinality - 1) * columns
        return total

    @property
    def cpds(self) -> Mapping[str, TabularCPD]:
        """A read-only snapshot of the CPDs attached so far, keyed by variable name."""
        return MappingProxyType(dict(self._cpds))

    # -- construction -------------------------------------------------------

    def add_cpd(self, cpd: TabularCPD) -> None:
        """Attach ``cpd`` to its variable, replacing any existing CPD.

        Raises ``ValidationError`` if the CPD breaks B2 or B6. In that case the
        model is left unchanged.
        """
        if not isinstance(cpd, TabularCPD):
            raise ValidationError(f"Expected TabularCPD, got {cpd!r}.")
        self._check_cpd(cpd)
        self._cpds[cpd.variable.name] = cpd
        self._order = None

    def validate(self) -> None:
        """Check that the model is complete and consistent (B1, B2, B6)."""
        missing = [name for name in self._variables if name not in self._cpds]
        if missing:
            raise ValidationError(f"Model is incomplete: missing CPDs for {missing}.")
        for cpd in self._cpds.values():
            self._check_cpd(cpd)
        self._order = tuple(self._dag.topological_sort())

    # -- queries ------------------------------------------------------------

    def joint_probability(self, assignment: Mapping[str, str]) -> float:
        """Return p(x) = prod_i p(x_i | pa(x_i)) for a complete assignment x."""
        order = self._validated_order()
        self._check_assignment(assignment)

        probability = 1.0
        for name in order:
            cpd = self._cpds[name]
            given = {parent: assignment[parent] for parent in cpd.parent_names}
            probability *= cpd.probability(assignment[name], given)
        return probability

    def factors(self) -> tuple[DiscreteFactor, ...]:
        """Return the CPDs as factors, in topological order.

        By P1 their product is the joint distribution. Validates the model first.
        """
        order = self._validated_order()
        return tuple(DiscreteFactor.from_cpd(self._cpds[name]) for name in order)

    def check_evidence(self, evidence: Mapping[str, str]) -> dict[str, str]:
        """Validate a partial assignment and return it as a plain dict.

        Every name must be a model variable (``UnknownNodeError``), and every
        state must be in that variable's domain (``UnknownStateError``). Unlike
        ``joint_probability``, the assignment does not need to cover every
        variable.
        """
        if not isinstance(evidence, Mapping):
            raise ValidationError(f"Evidence must be a mapping, got {evidence!r}.")
        unknown = [name for name in evidence if name not in self._variables]
        if unknown:
            raise UnknownNodeError(f"Evidence names unknown variables {unknown}.")
        for name, state in evidence.items():
            self._variables[name].index_of(state)
        return dict(evidence)

    def sample(self, n: int, seed: int | None = None) -> list[dict[str, str]]:
        """Draw ``n`` i.i.d. joint samples by ancestral sampling.

        This is equivalent to ``AncestralSampler(self, seed).sample(n)``.
        """
        from probgraph.sampling import AncestralSampler  # local import avoids an import cycle

        return AncestralSampler(self, seed=seed).sample(n)

    # -- internals ----------------------------------------------------------

    def _validated_order(self) -> tuple[str, ...]:
        if self._order is None:
            self.validate()
        assert self._order is not None
        return self._order

    def _check_cpd(self, cpd: TabularCPD) -> None:
        name = cpd.variable.name
        if name not in self._variables:
            raise ValidationError(
                f"CPD is for {name!r}, which is not a variable of this model "
                f"(variables: {list(self._variables)})."
            )
        self._check_canonical(cpd.variable, f"CPD for {name!r}")

        # B2: compare as sets. The CPD's own parent order only fixes its axes.
        declared = set(cpd.parent_names)
        graph_parents = self._dag.parents(name)
        if declared != graph_parents:
            raise ValidationError(
                f"CPD for {name!r} has parents {sorted(declared)}, "
                f"but the graph's parents of {name!r} are {sorted(graph_parents)}."
            )
        for parent in cpd.parents:
            self._check_canonical(parent, f"CPD for {name!r} (parent {parent.name!r})")

    def _check_canonical(self, variable: DiscreteVariable, context: str) -> None:
        canonical = self._variables[variable.name]
        if variable != canonical:
            raise ValidationError(
                f"{context}: declares {variable.name!r} with domain {variable.states}, "
                f"but the model's {variable.name!r} has domain {canonical.states}."
            )

    def _check_assignment(self, assignment: Mapping[str, str]) -> None:
        if not isinstance(assignment, Mapping):
            raise ValidationError(f"Assignment must be a mapping, got {assignment!r}.")
        missing = [name for name in self._variables if name not in assignment]
        unexpected = [name for name in assignment if name not in self._variables]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise ValidationError(
                "Assignment must give exactly one state for every model variable: "
                + "; ".join(details)
                + "."
            )
        for name, variable in self._variables.items():
            variable.index_of(assignment[name])  # raises UnknownStateError (C6 / B7)

    def __repr__(self) -> str:
        return (
            f"BayesianNetwork(variables={list(self._variables)}, edges={list(self.edges())}, "
            f"cpds={len(self._cpds)}/{len(self._variables)})"
        )
