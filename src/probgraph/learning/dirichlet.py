"""Dirichlet priors and Bayesian parameter estimation for Bayesian networks.

The derivations (P15, Parts 1 and 2) are in ``docs/mathematics/dirichlet.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from probgraph.distributions import TabularCPD
from probgraph.exceptions import ValidationError
from probgraph.learning.dataset import Dataset
from probgraph.learning.likelihood import _check_variables, _describe_columns, _parents_in_order
from probgraph.models import BayesianNetwork
from probgraph.variables import DiscreteVariable

Point = Literal["mean", "map"]


class DirichletPrior:
    """Independent Dirichlet pseudocounts for every CPD column (dirichlet.md §2).

    Build one with ``uniform(alpha)``, ``bdeu(equivalent_sample_size)`` or
    ``explicit({variable: pseudocounts})``. Explicit arrays have the CPD's shape,
    (|X|, |U_1|, ...), with parents in the model's declaration order.
    """

    __slots__ = ("_explicit", "_kind", "_value")

    def __init__(
        self, kind: str, value: float, explicit: Mapping[str, np.ndarray] | None = None
    ) -> None:
        self._kind = kind
        self._value = value
        self._explicit = dict(explicit or {})

    @classmethod
    def uniform(cls, alpha: float = 1.0) -> DirichletPrior:
        """α pseudocounts in every cell. α = 1 is the K2 / Laplace prior."""
        return cls("uniform", _positive(alpha, "alpha"))

    @classmethod
    def bdeu(cls, equivalent_sample_size: float) -> DirichletPrior:
        """ess / (|X| · q) in every cell, so each family's pseudocounts total ``ess``."""
        return cls("bdeu", _positive(equivalent_sample_size, "equivalent_sample_size"))

    @classmethod
    def explicit(cls, pseudocounts: Mapping[str, ArrayLike]) -> DirichletPrior:
        arrays = {}
        for name, values in pseudocounts.items():
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all() or (array <= 0).any():
                raise ValidationError(f"Pseudocounts for {name!r} must be finite and positive.")
            arrays[name] = array
        return cls("explicit", 0.0, arrays)

    def pseudocounts(
        self, variable: DiscreteVariable, parents: Sequence[DiscreteVariable]
    ) -> np.ndarray:
        """The pseudocount array α for ``variable``'s CPD, shape (|X|, |U_1|, ...)."""
        shape = (variable.cardinality, *(p.cardinality for p in parents))
        if self._kind == "uniform":
            return np.full(shape, self._value)
        if self._kind == "bdeu":
            return np.full(shape, self._value / math.prod(shape))
        if variable.name not in self._explicit:
            raise ValidationError(f"The explicit prior has no pseudocounts for {variable.name!r}.")
        array = self._explicit[variable.name]
        if array.shape != shape:
            raise ValidationError(
                f"Pseudocounts for {variable.name!r} have shape {array.shape}; expected {shape}."
            )
        return array

    def __repr__(self) -> str:
        if self._kind == "explicit":
            return f"DirichletPrior.explicit({sorted(self._explicit)})"
        return f"DirichletPrior.{self._kind}({self._value})"


def bayesian_estimate(
    structure: BayesianNetwork, data: Dataset, prior: DirichletPrior, point: Point = "mean"
) -> BayesianNetwork:
    """A point estimate of every CPD from its Dirichlet posterior, Dir(α + N) (Theorem 1).

    ``point="mean"`` (the default) gives (N + α) / (N(u) + α·), which is defined
    for every column, including unseen ones. ``point="map"`` gives the posterior
    mode (N + α - 1) / (N(u) + α· - |X|). It needs every prior α >= 1, and it
    raises for a column whose posterior is flat (no data and α = 1).
    """
    if point not in ("mean", "map"):
        raise ValidationError(f"point must be 'mean' or 'map', got {point!r}.")
    _check_variables(structure.variables, data)
    if not data.is_complete:
        raise ValidationError(
            f"bayesian_estimate needs complete data, but {data.missing_count} values are "
            "missing; use ExpectationMaximisation (P16) with a prior instead."
        )
    model = BayesianNetwork(structure.variables, structure.edges())
    flat: list[str] = []
    for variable in structure.variables:
        parents = _parents_in_order(structure, variable.name)
        counts = data.counts([variable.name, *(p.name for p in parents)]).values
        alpha = prior.pseudocounts(variable, parents)
        posterior = counts + alpha
        if point == "mean":
            theta = posterior / posterior.sum(axis=0, keepdims=True)
        else:
            if (alpha < 1).any():
                raise ValidationError(
                    f"MAP needs every pseudocount >= 1, but {variable.name!r} has some below 1 "
                    "(the posterior mode would sit on the boundary or not exist). Use point='mean'."
                )
            numerator = posterior - 1.0
            denominator = numerator.sum(axis=0, keepdims=True)
            empty = denominator == 0
            if empty.any():
                flat.extend(_describe_columns(variable, parents, empty[0]))
                continue
            theta = numerator / denominator
        model.add_cpd(TabularCPD(variable, parents, theta))
    if flat:
        raise ValidationError(
            f"The posterior is flat for {flat} (no data and α = 1), so its MAP is undefined; "
            "use point='mean' or a prior with α > 1."
        )
    return model


def log_marginal_likelihood(
    structure: BayesianNetwork, data: Dataset, prior: DirichletPrior
) -> float:
    """log P(D | G), the Bayesian score of ``structure``'s graph (Theorem 2, B4).

    The parameters are integrated out against ``prior``, giving one ratio of
    Dirichlet normalising constants per CPD column:
    Σ_{i,u} [log Γ(α·|u) - log Γ(α·|u + N(u)) + Σ_x (log Γ(α_x|u + N(x,u)) - log Γ(α_x|u))].
    ``D`` is the ordered sequence of rows. Any CPDs on ``structure`` are ignored,
    and the data must be complete.
    """
    _check_variables(structure.variables, data)
    if not data.is_complete:
        raise ValidationError(
            f"log_marginal_likelihood needs complete data, but {data.missing_count} values are "
            "missing."
        )
    total = 0.0
    for variable in structure.variables:
        parents = _parents_in_order(structure, variable.name)
        counts = data.counts([variable.name, *(p.name for p in parents)]).values
        alpha = prior.pseudocounts(variable, parents)
        alpha_total = alpha.sum(axis=0)
        total += float(
            (_lgamma(alpha_total) - _lgamma(alpha_total + counts.sum(axis=0))).sum()
            + (_lgamma(alpha + counts) - _lgamma(alpha)).sum()
        )
    return total


_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def _positive(value: float, name: str) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValidationError(f"{name} must be a finite positive number, got {value!r}.")
    return float(value)
