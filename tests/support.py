"""Shared fixtures and test oracles.

The enumeration oracle is exponential in the number of variables. It exists
only to check small networks, and is deliberately kept out of the library's
public API (spec §2).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from probgraph import BayesianNetwork, DiscreteFactor, DiscreteVariable, TabularCPD

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
# Extended fixture for Milestone 2 (docs/specs/milestone-2.md §4):
#   Umbrella <- Rain -> Traffic <- Accident,   Traffic -> Late
# ---------------------------------------------------------------------------

LATE = DiscreteVariable("Late", ("no", "yes"))
UMBRELLA = DiscreteVariable("Umbrella", ("no", "yes"))


def late_cpd() -> TabularCPD:
    return TabularCPD(LATE, (TRAFFIC,), [[0.9, 0.4], [0.1, 0.6]])


def umbrella_cpd() -> TabularCPD:
    return TabularCPD(UMBRELLA, (RAIN,), [[0.9, 0.15], [0.1, 0.85]])


def late_network() -> BayesianNetwork:
    model = BayesianNetwork(
        variables=[RAIN, ACCIDENT, TRAFFIC, LATE, UMBRELLA],
        edges=[
            ("Rain", "Traffic"),
            ("Accident", "Traffic"),
            ("Traffic", "Late"),
            ("Rain", "Umbrella"),
        ],
    )
    for cpd in (rain_cpd(), accident_cpd(), traffic_cpd(), late_cpd(), umbrella_cpd()):
        model.add_cpd(cpd)
    return model


# ---------------------------------------------------------------------------
# Random networks
# ---------------------------------------------------------------------------


def random_network(
    seed: int, n_vars: tuple[int, int] = (1, 5), cards: tuple[int, int] = (1, 3)
) -> BayesianNetwork:
    """A random DAG with random cardinalities and Dirichlet(1) CPDs.

    ``n_vars`` and ``cards`` are inclusive ranges. Parent axes are shuffled, so a
    CPD's axis order differs from the order in which the graph lists the parents.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_vars[0], n_vars[1] + 1))
    variables = [
        DiscreteVariable(
            f"V{i}", tuple(f"s{j}" for j in range(int(rng.integers(cards[0], cards[1] + 1))))
        )
        for i in range(n)
    ]
    hidden = rng.permutation(n)
    edges = [
        (f"V{hidden[i]}", f"V{hidden[j]}")
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < 0.5
    ]
    model = BayesianNetwork(variables, edges)
    by_name = {v.name: v for v in variables}
    for v in variables:
        parents = tuple(by_name[p] for p in rng.permutation(sorted(model.parents(v.name))))
        shape = (v.cardinality, *(p.cardinality for p in parents))
        columns = rng.dirichlet(np.ones(v.cardinality), size=int(np.prod(shape[1:], dtype=int))).T
        model.add_cpd(TabularCPD(v, parents, columns.reshape(shape)))
    return model


def random_evidence(
    model: BayesianNetwork, rng: np.random.Generator, exclude: Iterable[str] = ()
) -> dict[str, str]:
    """Observe each variable not in ``exclude`` with probability 1/2, at a random state."""
    excluded = set(exclude)
    return {
        v.name: v.states[int(rng.integers(v.cardinality))]
        for v in model.variables
        if v.name not in excluded and rng.random() < 0.5
    }


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


def posterior(
    model: BayesianNetwork, query: Sequence[str], evidence: Mapping[str, str]
) -> DiscreteFactor:
    """P(query | evidence) by brute-force enumeration.

    This uses only M1's ``joint_probability`` and numpy sums, never the factor
    algebra, so it is an independent oracle for factor-based inference. Raises
    ``ZeroDivisionError`` when P(e) = 0.
    """
    table = joint_table(model)
    names = [v.name for v in model.variables]
    index: list[int | slice] = [
        v.index_of(evidence[v.name]) if v.name in evidence else slice(None) for v in model.variables
    ]
    reduced = table[tuple(index)]  # axes: unobserved variables, in model order
    free = [n for n in names if n not in evidence]
    summed = reduced.sum(axis=tuple(i for i, n in enumerate(free) if n not in query))
    kept = [n for n in free if n in query]  # axes of `summed`, in model order
    total = summed.sum()
    if total == 0:
        raise ZeroDivisionError(f"P(evidence) = 0 for {dict(evidence)}")
    by_name = {v.name: v for v in model.variables}
    return DiscreteFactor([by_name[n] for n in kept], summed / total).aligned(list(query))


def evidence_probability(model: BayesianNetwork, evidence: Mapping[str, str]) -> float:
    """P(e) by brute-force enumeration (numpy only)."""
    table = joint_table(model)
    index = tuple(
        v.index_of(evidence[v.name]) if v.name in evidence else slice(None) for v in model.variables
    )
    return float(table[index].sum())


def conditionally_independent(
    model: BayesianNetwork,
    xs: Iterable[str],
    ys: Iterable[str],
    given: Iterable[str] = (),
    atol: float = 1e-12,
) -> bool:
    """Check X ⊥ Y | Z numerically: P(x,y,z) P(z) = P(x,z) P(y,z) for every cell."""
    table = joint_table(model)
    axis = {v.name: i for i, v in enumerate(model.variables)}
    x, y, z = ([axis[n] for n in group] for group in (xs, ys, given))

    def m(keep: list[int]) -> np.ndarray:
        drop = tuple(i for i in range(table.ndim) if i not in keep)
        return table.sum(axis=drop, keepdims=True)

    return bool(np.allclose(m(x + y + z) * m(z), m(x + z) * m(y + z), atol=atol, rtol=0.0))
