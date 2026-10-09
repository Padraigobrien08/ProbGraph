"""P4: parameter counts, and the Jacobian-rank experiment from parameter_reduction.md §5."""

import itertools

import numpy as np
import pytest
from support import TRAFFIC, joint_table, rain_network, rain_network_structure, traffic_cpd

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD


def binary(name: str) -> DiscreteVariable:
    return DiscreteVariable(name, ("0", "1"))


def variables_with(cards: list[int]) -> list[DiscreteVariable]:
    return [
        DiscreteVariable(f"V{i}", tuple(f"s{j}" for j in range(k))) for i, k in enumerate(cards)
    ]


def complete_dag(cards: list[int]) -> BayesianNetwork:
    vs = variables_with(cards)
    edges = [(vs[i].name, vs[j].name) for i, j in itertools.combinations(range(len(vs)), 2)]
    return BayesianNetwork(vs, edges)


def chain(d: int) -> BayesianNetwork:
    vs = [binary(f"X{i}") for i in range(d)]
    return BayesianNetwork(vs, list(itertools.pairwise(v.name for v in vs)))


def naive_bayes(n_features: int) -> BayesianNetwork:
    vs = [binary("C"), *(binary(f"F{i}") for i in range(n_features))]
    return BayesianNetwork(vs, [("C", v.name) for v in vs[1:]])


def random_dag(seed: int) -> BayesianNetwork:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    vs = variables_with([int(rng.integers(2, 4)) for _ in range(n)])
    order = rng.permutation(n)
    edges = [
        (vs[order[i]].name, vs[order[j]].name)
        for i, j in itertools.combinations(range(n), 2)
        if rng.random() < 0.5
    ]
    return BayesianNetwork(vs, edges)


def full_joint_dimension(model: BayesianNetwork) -> int:
    return int(np.prod([v.cardinality for v in model.variables])) - 1


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_cpd_parameter_count():
    assert traffic_cpd().n_free_parameters == 4  # (2 - 1) * 2 * 2
    x = DiscreteVariable("X", ("a", "b", "c"))
    a = DiscreteVariable("A", ("a0", "a1", "a2", "a3"))
    assert TabularCPD(x, (a,), np.full((3, 4), 1 / 3)).n_free_parameters == 8
    assert TabularCPD(x, (), [0.2, 0.3, 0.5]).n_free_parameters == 2


def test_fixture_saves_exactly_one_parameter():
    model = rain_network()
    assert model.n_free_parameters == 6
    assert full_joint_dimension(model) == 7
    # The count depends only on the structure, not on whether CPDs are attached.
    assert rain_network_structure().n_free_parameters == 6
    assert model.n_free_parameters == sum(c.n_free_parameters for c in model.cpds.values())


@pytest.mark.parametrize("cards", [[2], [2, 2], [2, 3], [3, 2, 4], [2, 2, 2, 2, 2], [4, 1, 3]])
def test_complete_dag_matches_full_joint(cards):
    # Telescoping: sum_i (k_i - 1) K_{i-1} = K_d - 1.
    model = complete_dag(cards)
    assert model.n_free_parameters == full_joint_dimension(model)


@pytest.mark.parametrize("d", range(1, 9))
def test_chain_is_linear(d):
    assert chain(d).n_free_parameters == 2 * d - 1


@pytest.mark.parametrize("n", range(1, 9))
def test_naive_bayes_is_linear(n):
    assert naive_bayes(n).n_free_parameters == 1 + 2 * n


def test_empty_graph_counts_marginals_only():
    model = BayesianNetwork(variables_with([2, 3, 4]), [])
    assert model.n_free_parameters == 1 + 2 + 3


@pytest.mark.parametrize("seed", range(10))
def test_removing_any_edge_strictly_reduces_parameters(seed):
    model = random_dag(seed)
    for removed in model.edges():
        kept = [e for e in model.edges() if e != removed]
        smaller = BayesianNetwork(model.variables, kept)
        assert smaller.n_free_parameters < model.n_free_parameters


# ---------------------------------------------------------------------------
# Jacobian-rank experiment (§4–§5): #θ is the true dimension of the model family
# ---------------------------------------------------------------------------


def build(structure: BayesianNetwork, theta: np.ndarray) -> BayesianNetwork:
    """Fill ``structure`` with CPDs taken from the flat parameter vector ``theta``.

    Each column is (θ_1, ..., θ_{k-1}, 1 - Σθ).
    """
    model = BayesianNetwork(structure.variables, structure.edges())
    by_name = {v.name: v for v in structure.variables}
    offset = 0
    for v in structure.variables:
        parents = tuple(by_name[p] for p in sorted(structure.parents(v.name)))
        n_columns = int(np.prod([p.cardinality for p in parents]))
        free = theta[offset : offset + (v.cardinality - 1) * n_columns]
        offset += free.size
        head = free.reshape(v.cardinality - 1, n_columns)
        table = np.vstack([head, 1.0 - head.sum(axis=0, keepdims=True)])
        model.add_cpd(
            TabularCPD(
                v, parents, table.reshape((v.cardinality, *(p.cardinality for p in parents)))
            )
        )
    assert offset == theta.size
    return model


def interior_theta(structure: BayesianNetwork, rng: np.random.Generator) -> np.ndarray:
    """A random strictly positive parameter vector, well away from the simplex boundary."""
    parts = []
    for v in structure.variables:
        n_columns = int(
            np.prod([structure.variable(p).cardinality for p in structure.parents(v.name)])
        )
        columns = rng.dirichlet(np.full(v.cardinality, 5.0), size=n_columns).T  # (k, n_columns)
        assert columns.min() > 1e-3
        parts.append(columns[:-1].ravel())
    return np.concatenate(parts)


def jacobian_rank(structure: BayesianNetwork, seed: int = 0, h: float = 1e-4) -> int:
    theta = interior_theta(structure, np.random.default_rng(seed))
    assert theta.size == structure.n_free_parameters
    columns = []
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = h
        # P is affine in each coordinate, so the central difference is exact up to rounding.
        diff = joint_table(build(structure, theta + step)) - joint_table(
            build(structure, theta - step)
        )
        columns.append(diff.ravel() / (2 * h))
    jac = np.column_stack(columns)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    return int((singular_values > 1e-8 * singular_values[0]).sum())


STRUCTURES = {
    "fixture": rain_network_structure,
    "chain4": lambda: chain(4),
    "naive_bayes3": lambda: naive_bayes(3),
    "complete_2_3_2": lambda: complete_dag([2, 3, 2]),
    **{f"random{s}": (lambda s=s: random_dag(s)) for s in range(6)},
}


@pytest.mark.parametrize("name", STRUCTURES)
def test_jacobian_rank_equals_parameter_count(name):
    structure = STRUCTURES[name]()
    rank = jacobian_rank(structure)
    assert rank == structure.n_free_parameters
    assert rank <= full_joint_dimension(structure)


def test_fixture_family_is_a_proper_subset_of_the_simplex():
    structure = rain_network_structure()
    assert jacobian_rank(structure) == 6 < 7 == full_joint_dimension(structure)


def test_build_round_trips_the_fixture():
    # Sanity check of the parameterisation: the fixture's own θ reproduces its joint.
    t1 = np.array([[0.1, 0.7], [0.8, 0.95]])  # P(T=yes | rain, accident), indexed [rain, accident]
    # build() orders parents alphabetically, (Accident, Rain), so Traffic's columns run over
    # [accident, rain]. The free entry in each column is P(T=no | ...).
    theta = np.concatenate([[0.7], [0.9], (1 - t1).T.ravel()])
    model = build(rain_network_structure(), theta)
    assert model.cpds["Traffic"].variable == TRAFFIC
    np.testing.assert_allclose(joint_table(model), joint_table(rain_network()), atol=1e-15)
