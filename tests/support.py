"""Shared fixtures and test oracles.

The enumeration oracle is exponential in the number of variables. It exists
only to check small networks, and is deliberately kept out of the library's
public API (spec §2).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping

import numpy as np

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD

# ---------------------------------------------------------------------------
# Canonical Rain / Accident / Traffic network (spec §4)
# ---------------------------------------------------------------------------

RAIN = DiscreteVariable("Rain", ("no", "yes"))
ACCIDENT = DiscreteVariable("Accident", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))

P_TRAFFIC_GIVEN_RAIN_ACCIDENT = np.array([[0.1, 0.7], [0.8, 0.95]])  # [rain, accident]


def rain_cpd() -> TabularCPD:
    return TabularCPD(RAIN, (), [0.7, 0.3])


def accident_cpd() -> TabularCPD:
    return TabularCPD(ACCIDENT, (), [0.9, 0.1])


def traffic_cpd() -> TabularCPD:
    t1 = P_TRAFFIC_GIVEN_RAIN_ACCIDENT
    return TabularCPD(TRAFFIC, (RAIN, ACCIDENT), np.stack([1 - t1, t1]))


def rain_network_structure() -> BayesianNetwork:
    """The fixture graph R -> T <- A, without any CPDs."""
    return BayesianNetwork(
        variables=[RAIN, ACCIDENT, TRAFFIC],
        edges=[("Rain", "Traffic"), ("Accident", "Traffic")],
    )


def rain_network() -> BayesianNetwork:
    model = rain_network_structure()
    for cpd in (rain_cpd(), accident_cpd(), traffic_cpd()):
        model.add_cpd(cpd)
    return model


# ---------------------------------------------------------------------------
# Enumeration oracle
# ---------------------------------------------------------------------------


def joint_table(model: BayesianNetwork) -> np.ndarray:
    """Evaluate ``joint_probability`` at every complete assignment.

    The result is an array with one axis per variable, in ``model.variables``
    order, indexed by state position.
    """
    variables = model.variables
    table = np.empty(tuple(v.cardinality for v in variables))
    for index in itertools.product(*(range(v.cardinality) for v in variables)):
        assignment = {v.name: v.states[i] for v, i in zip(variables, index, strict=True)}
        table[index] = model.joint_probability(assignment)
    return table


def probability_of(
    model: BayesianNetwork,
    event: Callable[[Mapping[str, str]], bool],
    given: Callable[[Mapping[str, str]], bool] = lambda _: True,
) -> float:
    """P(event | given), computed by brute-force summation over the joint."""
    variables = model.variables
    numerator = denominator = 0.0
    for states in itertools.product(*(v.states for v in variables)):
        assignment = dict(zip((v.name for v in variables), states, strict=True))
        if given(assignment):
            p = model.joint_probability(assignment)
            denominator += p
            if event(assignment):
                numerator += p
    return numerator / denominator


def marginal(table: np.ndarray, keep: list[int]) -> np.ndarray:
    """Sum out every axis not in ``keep``. Kept axes stay in their original order."""
    drop = tuple(i for i in range(table.ndim) if i not in keep)
    return table.sum(axis=drop)
