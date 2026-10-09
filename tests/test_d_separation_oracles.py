"""M2.8: independent oracles for d-separation (spec task 8; invariants D3–D6).

* Theorem 2: Bayes ball agrees with the moralised-ancestral criterion. Exhaustive
  over every DAG with 5 nodes up to relabelling, and randomised on larger ones.
* Theorem 4 (soundness) and Theorem 5 (generic completeness), numerically on
  random networks.
* §7: the XOR network is unfaithful, and composition fails for probabilities.
* §8: the graphoid axioms, plus intersection and composition, hold for d-separation.
"""

import itertools

import numpy as np
import pytest
from support import joint_table, late_network, random_network

from probgraph import DAG, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.graphs import moral_graph


def moral_ancestral_separated(dag: DAG, xs, ys, zs) -> bool:
    """Theorem 2's criterion: separation in the moral graph of G[An*(X ∪ Y ∪ Z)]."""
    ancestral = dag.ancestral_set({*xs, *ys, *zs})
    return moral_graph(dag, restrict_to=ancestral).separated(set(xs), set(ys), set(zs))


def ci_gap(table: np.ndarray, x: list[int], y: list[int], z: list[int]) -> float:
    """max |P(x,y,z) P(z) - P(x,z) P(y,z)|. It is 0 exactly when X ⊥ Y | Z."""

    def m(keep: list[int]) -> np.ndarray:
        return table.sum(axis=tuple(i for i in range(table.ndim) if i not in keep), keepdims=True)

    return float(np.abs(m(x + y + z) * m(z) - m(x + z) * m(y + z)).max())


# ---------------------------------------------------------------------------
# D3 / Theorem 2: agreement with the moralised-ancestral criterion
# ---------------------------------------------------------------------------


def upper_triangular_dags(nodes: str):
    """Every DAG whose topological order is ``nodes``. Up to relabelling, these are all DAGs."""
    pairs = list(itertools.combinations(nodes, 2))
    for mask in range(1 << len(pairs)):
        yield DAG(nodes=nodes, edges=[p for k, p in enumerate(pairs) if mask >> k & 1])


def test_theorem_2_exhaustive_on_all_five_node_dags():
    nodes = "ABCDE"
    checked = 0
    for dag in upper_triangular_dags(nodes):
        for x, y in itertools.combinations(nodes, 2):
            rest = [n for n in nodes if n not in (x, y)]
            for r in range(len(rest) + 1):
                for z in itertools.combinations(rest, r):
                    expected = moral_ancestral_separated(dag, {x}, {y}, z)
                    assert dag.d_separated({x}, {y}, set(z)) is expected, (dag, x, y, z)
                    checked += 1
    assert checked == 1024 * 10 * 8


@pytest.mark.parametrize("seed", range(80))
def test_theorem_2_on_random_larger_dags_with_sets(seed):
    rng = np.random.default_rng(seed + 90_000)
    dag = random_network(seed, n_vars=(6, 9), edge_prob=0.3).graph
    perm = [str(n) for n in rng.permutation(list(dag.nodes()))]
    nx, ny = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    xs, ys = perm[:nx], perm[nx : nx + ny]
    zs = [n for n in perm[nx + ny :] if rng.random() < 0.4]
    assert dag.d_separated(xs, ys, zs) is moral_ancestral_separated(dag, xs, ys, zs)


def test_fixture_facts_by_moral_criterion():
    dag = late_network().graph
    assert moral_ancestral_separated(dag, ["Accident"], ["Rain"], [])
    assert not moral_ancestral_separated(dag, ["Accident"], ["Rain"], ["Late"])
    assert moral_ancestral_separated(dag, ["Umbrella"], ["Accident"], ["Traffic", "Rain"])
    assert not moral_ancestral_separated(dag, ["Umbrella"], ["Accident"], ["Traffic"])


def test_ancestral_restriction_matters():
    """The A–B marriage via an unobserved, non-ancestral child must not connect them.

    A -> X <- B, and the query is A ⊥ B with nothing observed. X is outside An*({A, B}),
    so the moral graph of the restricted DAG has no A–B edge. Moralising the whole graph
    would wrongly connect A and B.
    """
    dag = DAG(nodes="ABX", edges=[("A", "X"), ("B", "X")])
    assert dag.d_separated({"A"}, {"B"})
    assert moral_ancestral_separated(dag, {"A"}, {"B"}, set())
    assert not moral_graph(dag).separated({"A"}, {"B"})  # without the ancestral restriction


# ---------------------------------------------------------------------------
# D4 / D5: soundness, and completeness for generic parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_soundness_and_generic_completeness(seed):
    # Cardinalities >= 2: a constant (cardinality 1) is independent of everything.
    model = random_network(seed, n_vars=(3, 6), cards=(2, 3), edge_prob=0.45)
    dag = model.graph
    table = joint_table(model)
    axis = {v.name: i for i, v in enumerate(model.variables)}
    names = list(dag.nodes())
    separated_seen = connected_seen = 0
    for x, y in itertools.combinations(names, 2):
        rest = [n for n in names if n not in (x, y)]
        for r in range(len(rest) + 1):
            for z in itertools.combinations(rest, r):
                gap = ci_gap(table, [axis[x]], [axis[y]], [axis[n] for n in z])
                if dag.d_separated({x}, {y}, set(z)):
                    assert gap <= 1e-12, ("soundness", x, y, z, gap)  # Theorem 4
                    separated_seen += 1
                else:
                    assert gap > 1e-9, ("generic completeness", x, y, z, gap)  # Theorem 5
                    connected_seen += 1
    # Guard the test's own power: any graph with an edge has a d-connected pair.
    assert connected_seen > 0 or not dag.edges()


# ---------------------------------------------------------------------------
# §7: the XOR network is unfaithful
# ---------------------------------------------------------------------------


def xor_network() -> BayesianNetwork:
    x, y, w = (DiscreteVariable(n, ("0", "1")) for n in "XYW")
    model = BayesianNetwork([x, y, w], [("X", "W"), ("Y", "W")])
    model.add_cpd(TabularCPD(x, (), [0.5, 0.5]))
    model.add_cpd(TabularCPD(y, (), [0.5, 0.5]))
    xor = np.zeros((2, 2, 2))
    for a, b in itertools.product((0, 1), repeat=2):
        xor[a ^ b, a, b] = 1.0  # values[w, x, y]
    model.add_cpd(TabularCPD(w, (x, y), xor))
    return model


def test_xor_is_unfaithful():
    model = xor_network()
    table = joint_table(model)
    x, w = 0, 2
    # X and W are adjacent, so they are d-connected...
    assert not model.graph.d_separated({"X"}, {"W"})
    # ...but numerically independent: a measure-zero, non-generic parameter choice.
    assert ci_gap(table, [x], [w], []) <= 1e-15


def test_composition_fails_for_probabilities_but_not_for_d_separation():
    table = joint_table(xor_network())
    x, y, w = 0, 1, 2
    assert ci_gap(table, [x], [y], []) <= 1e-15  # X ⊥ Y
    assert ci_gap(table, [x], [w], []) <= 1e-15  # X ⊥ W
    assert ci_gap(table, [x], [y, w], []) > 0.1  # but X is not independent of (Y, W)
    # d-separation does not claim X ⊥ W here, so it is not contradicted.
    dag = xor_network().graph
    assert dag.d_separated({"X"}, {"Y"}) and not dag.d_separated({"X"}, {"W"})


# ---------------------------------------------------------------------------
# D6 / §8: graphoid axioms, intersection and composition for d-separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(150))
def test_graphoid_axioms(seed):
    rng = np.random.default_rng(seed + 100_000)
    dag = random_network(seed, n_vars=(4, 8), edge_prob=0.35).graph
    perm = [str(n) for n in rng.permutation(list(dag.nodes()))]
    x, y, w = {perm[0]}, {perm[1]}, {perm[2]}
    z = {n for n in perm[3:] if rng.random() < 0.5}

    def ds(a, b, c):
        return dag.d_separated(a, b, c)

    assert ds(x, y, z) == ds(y, x, z)  # symmetry
    if ds(x, y | w, z):
        assert ds(x, y, z)  # decomposition
        assert ds(x, y, z | w)  # weak union
    if ds(x, y, z) and ds(x, w, z | y):
        assert ds(x, y | w, z)  # contraction
    if ds(x, y, z | w) and ds(x, w, z | y):
        assert ds(x, y | w, z)  # intersection
    if ds(x, y, z) and ds(x, w, z):
        assert ds(x, y | w, z)  # composition
