"""Log-likelihood and maximum-likelihood estimation for Bayesian networks.

The derivations (P14, Part 2) are in ``docs/mathematics/likelihood.md``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from probgraph.distributions import TabularCPD
from probgraph.exceptions import ValidationError
from probgraph.inference import VariableElimination
from probgraph.learning.dataset import MISSING, Dataset
from probgraph.models import BayesianNetwork
from probgraph.variables import DiscreteVariable

Unseen = Literal["raise", "uniform"]


def log_likelihood(model: BayesianNetwork, data: Dataset) -> float:
    """log L = Σ_m log P(x_obs^(m)), the observed-data log-likelihood (likelihood.md §7).

    Rows with identical values share one computation. Complete rows use the
    factorisation directly; rows with missing values use log-space variable
    elimination. A row that the model deems impossible gives -inf.
    """
    _check_variables(model.variables, data)
    if data.n_rows == 0:
        return 0.0
    model.validate()
    column = {name: i for i, name in enumerate(data.names)}
    patterns, multiplicity = np.unique(data.codes, axis=0, return_counts=True)
    engine: VariableElimination | None = None
    total = 0.0
    for codes, count in zip(patterns.tolist(), multiplicity.tolist(), strict=True):
        if MISSING not in codes:
            term = _complete_row_log_probability(model, codes, column)
        else:
            if engine is None:
                engine = VariableElimination(model)
            observed = {
                v.name: v.states[codes[column[v.name]]]
                for v in model.variables
                if codes[column[v.name]] != MISSING
            }
            term = engine.log_probability_of_evidence(observed)
        if term == -math.inf:
            return -math.inf
        total += count * term
    return total


def maximum_likelihood(
    structure: BayesianNetwork, data: Dataset, unseen: Unseen = "raise"
) -> BayesianNetwork:
    """The MLE of every CPD of ``structure``'s graph: θ̂(x | u) = N(x, u) / N(u) (Theorem 2).

    ``structure`` provides the variables and edges; any CPDs it has are ignored,
    and a new model is returned. The data must be complete (use
    ``ExpectationMaximisation`` otherwise). A parent configuration with N(u) = 0
    leaves its column undetermined: by default that raises, naming every such
    configuration; ``unseen="uniform"`` fills those columns with the uniform
    distribution instead.
    """
    if unseen not in ("raise", "uniform"):
        raise ValidationError(f"unseen must be 'raise' or 'uniform', got {unseen!r}.")
    _check_variables(structure.variables, data)
    if not data.is_complete:
        raise ValidationError(
            f"maximum_likelihood needs complete data, but {data.missing_count} values are "
            "missing; use ExpectationMaximisation (P16) instead."
        )
    model = BayesianNetwork(structure.variables, structure.edges())
    undetermined: list[str] = []
    for variable in structure.variables:
        parents = _parents_in_order(structure, variable.name)
        counts = data.counts([variable.name, *(p.name for p in parents)]).values
        n_u = counts.sum(axis=0, keepdims=True)
        empty = n_u == 0
        if empty.any() and unseen == "raise":
            undetermined.extend(_describe_columns(variable, parents, empty[0]))
            continue
        theta = np.where(empty, 1.0 / variable.cardinality, counts / np.where(empty, 1.0, n_u))
        model.add_cpd(TabularCPD(variable, parents, theta))
    if undetermined:
        raise ValidationError(
            f"No data for {undetermined}: the MLE of those CPD columns is undefined "
            "(likelihood.md §6). Use a Dirichlet prior, or pass unseen='uniform'."
        )
    return model


def _complete_row_log_probability(
    model: BayesianNetwork, codes: Sequence[int], column: dict[str, int]
) -> float:
    """Σ_i log θ(x_i | u_i): the factorisation (P1), summed in log space so it cannot underflow."""
    total = 0.0
    for cpd in model.cpds.values():
        index = (codes[column[cpd.variable.name]], *(codes[column[p]] for p in cpd.parent_names))
        p = float(cpd.values[index])
        if p == 0.0:
            return -math.inf
        total += math.log(p)
    return total


def _parents_in_order(structure: BayesianNetwork, name: str) -> list[DiscreteVariable]:
    parents = structure.parents(name)
    return [v for v in structure.variables if v.name in parents]


def _describe_columns(
    variable: DiscreteVariable, parents: Sequence[DiscreteVariable], empty: np.ndarray
) -> list[str]:
    if not parents:
        return [f"{variable.name} (no parents)"] if bool(empty.all()) else []
    descriptions = []
    for config in itertools.product(*(range(p.cardinality) for p in parents)):
        if empty[config]:
            given = ", ".join(
                f"{p.name}={p.states[i]}" for p, i in zip(parents, config, strict=True)
            )
            descriptions.append(f"{variable.name} | {given}")
    return descriptions


def _check_variables(variables: Sequence[DiscreteVariable], data: Dataset) -> None:
    expected = {v.name: v for v in variables}
    given = {v.name: v for v in data.variables}
    missing = sorted(set(expected) - set(given))
    extra = sorted(set(given) - set(expected))
    if missing or extra:
        raise ValidationError(
            f"The data's variables do not match the model's: missing {missing}, unexpected {extra}."
        )
    for name, v in expected.items():
        if given[name] != v:
            raise ValidationError(
                f"Variable {name!r} has domain {given[name].states} in the data but "
                f"{v.states} in the model."
            )
