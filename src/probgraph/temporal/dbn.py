"""Dynamic Bayesian networks, specified as 2-TBNs and unrolled.

The mathematics (P22) is in ``docs/mathematics/dbn.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

from probgraph.distributions import TabularCPD
from probgraph.exceptions import ValidationError
from probgraph.graphs import DAG
from probgraph.models import BayesianNetwork
from probgraph.temporal.hmm import HiddenMarkovModel, _check_length
from probgraph.variables import DiscreteVariable

_PREVIOUS = "[t-1]"


def previous(variable: DiscreteVariable) -> DiscreteVariable:
    """The copy of ``variable`` in the previous slice, for use as a transition-CPD parent."""
    return DiscreteVariable(f"{variable.name}{_PREVIOUS}", variable.states)


class DynamicBayesianNetwork:
    """A 2-TBN: a network for slice 1, and one transition CPD per template variable (§1).

    A transition CPD's parents are template variables of the same slice or
    ``previous(v)`` copies from the slice before. The intra-slice graph must be
    acyclic. ``unroll(T)`` gives an ordinary ``BayesianNetwork`` over
    ``{name}_1 .. {name}_T``.

    Invariants (dbn.md):
        N1  an HMM written with ``from_hmm`` unrolls to ``HiddenMarkovModel.to_bayesian_network``
        P1  the unrolled joint is the product of the slice-1 and transition CPDs
        P2  the interface of slice t d-separates the past from the future
    """

    __slots__ = ("_initial", "_interface", "_transition", "_variables")

    def __init__(self, initial: BayesianNetwork, transition: Sequence[TabularCPD]) -> None:
        if not isinstance(initial, BayesianNetwork):
            raise ValidationError(
                f"initial must be a BayesianNetwork, got {type(initial).__name__}."
            )
        initial.validate()
        self._variables = initial.variables
        template = {v.name: v for v in self._variables}
        by_child: dict[str, TabularCPD] = {}
        for cpd in transition:
            if not isinstance(cpd, TabularCPD):
                raise ValidationError(f"Expected TabularCPD, got {type(cpd).__name__}.")
            name = cpd.variable.name
            if name.endswith(_PREVIOUS):
                raise ValidationError(
                    f"A transition CPD's child must be a template variable, not the "
                    f"previous-slice copy {name!r}."
                )
            if name not in template or cpd.variable != template[name]:
                raise ValidationError(f"Transition CPD child {name!r} is not a template variable.")
            if name in by_child:
                raise ValidationError(f"There are two transition CPDs for {name!r}.")
            for parent in cpd.parents:
                base = parent.name.removesuffix(_PREVIOUS)
                if base not in template or parent.states != template[base].states:
                    raise ValidationError(
                        f"Transition CPD for {name!r} has parent {parent.name!r}, which is neither "
                        "a template variable nor its previous-slice copy."
                    )
            by_child[name] = cpd
        missing = [n for n in template if n not in by_child]
        if missing:
            raise ValidationError(f"There is no transition CPD for {missing[0]!r}.")
        # The intra-slice graph must be acyclic: DAG rejects a cycle with CycleError.
        DAG(
            nodes=list(template),
            edges=[
                (p, c) for c, cpd in by_child.items() for p in cpd.parent_names if p in template
            ],
        )
        self._initial = initial
        self._transition = tuple(by_child[n] for n in template)
        linked = {
            p.removesuffix(_PREVIOUS)
            for cpd in self._transition
            for p in cpd.parent_names
            if p.endswith(_PREVIOUS)
        }
        self._interface = tuple(n for n in template if n in linked)

    @classmethod
    def from_hmm(cls, model: HiddenMarkovModel) -> DynamicBayesianNetwork:
        """An HMM as a 2-TBN: X -> Y in every slice, and X[t-1] -> X (N1)."""
        x, y = model.hidden, model.observed
        initial = BayesianNetwork([x, y], [(x.name, y.name)])
        initial.add_cpd(TabularCPD(x, (), model.initial))
        emission = TabularCPD(y, (x,), model.emission.T)
        initial.add_cpd(emission)
        return cls(initial, [TabularCPD(x, (previous(x),), model.transition.T), emission])

    @property
    def variables(self) -> tuple[DiscreteVariable, ...]:
        """The template variables of one slice."""
        return self._variables

    @property
    def initial(self) -> BayesianNetwork:
        return self._initial

    @property
    def transition(self) -> tuple[TabularCPD, ...]:
        """One CPD per template variable, in template order."""
        return self._transition

    @property
    def interface(self) -> tuple[str, ...]:
        """Template variables with a child in the next slice, in template order (§3)."""
        return self._interface

    def unroll(self, length: int) -> BayesianNetwork:
        """The Bayesian network over ``length`` slices (Proposition 1)."""
        length = _check_length(length)

        def at(variable: DiscreteVariable, t: int) -> DiscreteVariable:
            return DiscreteVariable(f"{variable.name}_{t}", variable.states)

        def resolve(parent: DiscreteVariable, t: int) -> DiscreteVariable:
            if parent.name.endswith(_PREVIOUS):
                base = parent.name.removesuffix(_PREVIOUS)
                return DiscreteVariable(f"{base}_{t - 1}", parent.states)
            return at(parent, t)

        variables = [at(v, t) for t in range(1, length + 1) for v in self._variables]
        cpds = [
            TabularCPD(at(cpd.variable, 1), tuple(at(p, 1) for p in cpd.parents), cpd.values)
            for cpd in (self._initial.cpds[v.name] for v in self._variables)
        ]
        for t in range(2, length + 1):
            cpds += [
                TabularCPD(
                    at(cpd.variable, t), tuple(resolve(p, t) for p in cpd.parents), cpd.values
                )
                for cpd in self._transition
            ]
        edges = [(p.name, cpd.variable.name) for cpd in cpds for p in cpd.parents]
        network = BayesianNetwork(variables, edges)
        for cpd in cpds:
            network.add_cpd(cpd)
        return network

    def __repr__(self) -> str:
        names = [v.name for v in self._variables]
        return f"DynamicBayesianNetwork(variables={names}, interface={list(self._interface)})"
