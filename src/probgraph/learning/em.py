"""Expectation–maximisation for Bayesian network parameters with missing values.

The derivations (P16) are in ``docs/mathematics/em.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from probgraph.distributions import TabularCPD
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import JunctionTree
from probgraph.learning.dataset import MISSING, Dataset
from probgraph.learning.dirichlet import (
    DirichletPrior,
    Point,
    _check_point,
    _posterior_from_counts,
)
from probgraph.learning.likelihood import (
    _check_variables,
    _complete_row_log_probability,
    _mle_from_counts,
    _parents_in_order,
)
from probgraph.models import BayesianNetwork
from probgraph.variables import DiscreteVariable


@dataclass(frozen=True)
class EMResult:
    """The outcome of ``ExpectationMaximisation.run``.

    ``log_likelihood[0]`` is the observed-data log L of the initial model, and
    ``log_likelihood[t]`` is the value after iteration t, so there are
    ``iterations + 1`` entries. ``log_objective`` is the quantity EM is guaranteed
    not to decrease (em.md §7): log L without a prior; with a prior,
    log L + Σ (α - 1) log θ for MAP and log L + Σ α log θ for the posterior mean
    (both without the prior's normalising constant).
    """

    model: BayesianNetwork
    log_likelihood: tuple[float, ...]
    log_objective: tuple[float, ...]
    converged: bool
    iterations: int


def expected_counts(model: BayesianNetwork, data: Dataset) -> dict[str, np.ndarray]:
    """N̄(x, u) = Σ_m P(X_i = x, Pa_i = u | o_m) for every family (em.md §4).

    Each array has shape (|X|, |U_1|, ...) with parents in the model's declaration
    order. Complete rows add their counts directly; rows with missing values are
    grouped by their observed values, and each group needs one junction tree.
    """
    _check_variables(model.variables, data)
    model.validate()
    counts, _ = _e_step(model, data)
    return counts


class ExpectationMaximisation:
    """Fit the CPDs of ``structure``'s graph to data with missing values (P16).

    Assumes the values are missing at random (em.md §2). Without a prior, the
    M-step is the MLE of the expected counts and the observed-data log-likelihood
    never decreases. With a ``prior``, the M-step is the posterior mean (default)
    or MAP of the expected counts, and the log-posterior never decreases.
    Iteration stops when ``log_objective`` changes by less than ``tolerance``, or
    after ``max_iterations``.
    """

    def __init__(
        self,
        structure: BayesianNetwork,
        data: Dataset,
        prior: DirichletPrior | None = None,
        point: Point = "mean",
        max_iterations: int = 200,
        tolerance: float = 1e-8,
    ) -> None:
        _check_variables(structure.variables, data)
        _check_point(point)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ValidationError(f"max_iterations must be an int, got {max_iterations!r}.")
        if max_iterations < 1:
            raise ValidationError(f"max_iterations must be at least 1, got {max_iterations}.")
        if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance)) or tolerance < 0:
            raise ValidationError(f"tolerance must be finite and non-negative, got {tolerance!r}.")
        self._structure = BayesianNetwork(structure.variables, structure.edges())
        self._data = data
        self._prior = prior
        self._point: Point = point
        self._max_iterations = max_iterations
        self._tolerance = float(tolerance)

    def initial_model(self, seed: int | None = None) -> BayesianNetwork:
        """Every CPD column drawn independently from the uniform Dirichlet."""
        rng = np.random.default_rng(seed)
        model = BayesianNetwork(self._structure.variables, self._structure.edges())
        for variable in self._structure.variables:
            parents = _parents_in_order(self._structure, variable.name)
            shape = (variable.cardinality, *(p.cardinality for p in parents))
            columns = rng.dirichlet(np.ones(variable.cardinality), size=math.prod(shape[1:]))
            model.add_cpd(TabularCPD(variable, parents, columns.T.reshape(shape)))
        return model

    def step(self, model: BayesianNetwork) -> BayesianNetwork:
        """One iteration: the E-step under ``model``, then the M-step."""
        self._check_initial(model)
        counts, _ = _e_step(model, self._data)
        return self._m_step(counts)

    def run(self, initial: BayesianNetwork | None = None, seed: int | None = None) -> EMResult:
        """Iterate from ``initial``, or from ``initial_model(seed)`` when it is None."""
        if initial is None:
            model = self.initial_model(seed)
        else:
            self._check_initial(initial)
            model = initial
        counts, log_l = _e_step(model, self._data)
        history = [log_l]
        objective = [log_l + self._log_prior_term(model)]
        complete = self._data.is_complete
        converged = False
        iterations = 0
        while iterations < self._max_iterations:
            model = self._m_step(counts)
            counts, log_l = _e_step(model, self._data)
            iterations += 1
            history.append(log_l)
            objective.append(log_l + self._log_prior_term(model))
            # With complete data the expected counts do not depend on the model, so
            # the first M-step is already the fixed point (em.md §7).
            if complete or abs(objective[-1] - objective[-2]) < self._tolerance:
                converged = True
                break
        return EMResult(model, tuple(history), tuple(objective), converged, iterations)

    # -- internals ----------------------------------------------------------------

    def _m_step(self, counts: dict[str, np.ndarray]) -> BayesianNetwork:
        if self._prior is None:
            return _mle_from_counts(
                self._structure, counts, remedy="Give ExpectationMaximisation a Dirichlet prior."
            )
        return _posterior_from_counts(self._structure, counts, self._prior, self._point)

    def _log_prior_term(self, model: BayesianNetwork) -> float:
        """Σ (α' - 1) log θ, with α' = α for MAP and α + 1 for the mean (em.md §7)."""
        if self._prior is None:
            return 0.0
        shift = 0.0 if self._point == "map" else 1.0
        total = 0.0
        for variable in self._structure.variables:
            parents = _parents_in_order(self._structure, variable.name)
            exponent = self._prior.pseudocounts(variable, parents) + shift - 1.0
            theta = _family_array(model, variable.name, parents)
            mask = exponent != 0
            with np.errstate(divide="ignore"):
                total += float(np.sum(exponent[mask] * np.log(theta[mask])))
        return total

    def _check_initial(self, model: BayesianNetwork) -> None:
        if not isinstance(model, BayesianNetwork):
            raise ValidationError(f"Expected a BayesianNetwork, got {type(model).__name__}.")
        if model.variables != self._structure.variables or set(model.edges()) != set(
            self._structure.edges()
        ):
            raise ValidationError(
                "The initial model must have the same variables and edges as the structure."
            )
        model.validate()


def _e_step(model: BayesianNetwork, data: Dataset) -> tuple[dict[str, np.ndarray], float]:
    """The expected counts under ``model``, and the observed-data log-likelihood with them."""
    column = {name: i for i, name in enumerate(data.names)}
    families = {
        v.name: [v.name, *(p.name for p in _parents_in_order(model, v.name))]
        for v in model.variables
    }
    counts = {
        name: np.zeros(tuple(model.variable(n).cardinality for n in family))
        for name, family in families.items()
    }
    log_l = 0.0
    if data.n_rows == 0:
        return counts, log_l
    patterns, multiplicity = np.unique(data.codes, axis=0, return_counts=True)
    for codes, count in zip(patterns.tolist(), multiplicity.tolist(), strict=True):
        if MISSING not in codes:
            log_p = _complete_row_log_probability(model, codes, column)
            if log_p == -math.inf:
                _raise_impossible(model, codes, column)
            log_l += count * log_p
            for name, family in families.items():
                counts[name][tuple(codes[column[n]] for n in family)] += count
            continue
        observed = {
            v.name: v.states[codes[column[v.name]]]
            for v in model.variables
            if codes[column[v.name]] != MISSING
        }
        tree = JunctionTree(model, evidence=observed)
        log_p = tree.log_probability_of_evidence()
        if log_p == -math.inf:
            _raise_impossible(model, codes, column)
        log_l += count * log_p
        for name, family in families.items():
            hidden = [n for n in family if n not in observed]
            index = tuple(slice(None) if n in hidden else codes[column[n]] for n in family)
            if hidden:
                counts[name][index] += count * tree.query(hidden).values
            else:
                counts[name][index] += count
    return counts, log_l


def _family_array(
    model: BayesianNetwork, name: str, parents: Sequence[DiscreteVariable]
) -> np.ndarray:
    """θ(x | u) with axes (X, parents in declaration order), whatever the CPD's axis order."""
    cpd = model.cpds[name]
    return np.transpose(cpd.values, [0, *(1 + cpd.parent_names.index(p.name) for p in parents)])


def _raise_impossible(model: BayesianNetwork, codes: list[int], column: dict[str, int]) -> None:
    row = {
        v.name: (None if codes[column[v.name]] == MISSING else v.states[codes[column[v.name]]])
        for v in model.variables
    }
    raise ZeroProbabilityEvidenceError(
        f"The model gives probability 0 to the observed row {row}, so EM cannot condition on it; "
        "start from a model that gives every observed row positive probability."
    )
