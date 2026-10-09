"""M3.3: MarkovNetwork and BayesianNetwork.to_markov_network() (spec §3.2; M1–M6; P10)."""

import itertools
import math

import numpy as np
import pytest
from support import (
    MISCONCEPTION_Z,
    STUDENTS,
    gibbs_table,
    joint_table,
    late_network,
    misconception_factors,
    rain_network,
    random_network,
)

from probgraph import DAG, DiscreteFactor, DiscreteVariable, MarkovNetwork, VariableElimination
from probgraph.exceptions import (
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)
from probgraph.factors import LogFactor
from probgraph.graphs import UndirectedGraph, moral_graph


def misconception() -> MarkovNetwork:
    return MarkovNetwork(STUDENTS, misconception_factors())


def ci_gap(table: np.ndarray, x: list[int], y: list[int], z: list[int]) -> float:
    def m(keep):
        return table.sum(axis=tuple(i for i in range(table.ndim) if i not in keep), keepdims=True)

    return float(np.abs(m(x + y + z) * m(z) - m(x + z) * m(y + z)).max())


# ---------------------------------------------------------------------------
# F2: the misconception network (spec §4)
# ---------------------------------------------------------------------------


def test_partition_function_matches_the_textbook():
    mn = misconception()
    assert mn.partition_function() == pytest.approx(MISCONCEPTION_Z, rel=1e-12)
    assert mn.log_partition_function() == pytest.approx(math.log(MISCONCEPTION_Z), abs=1e-12)


@pytest.mark.parametrize(
    ("var", "evidence", "expected"),
    [
        ("A", {}, 130031 / 720184),
        ("B", {}, 530151 / 720184),
        ("C", {}, 550073 / 720184),
        ("D", {}, 150113 / 720184),
        ("B", {"C": "1"}, 520050 / 550073),
        ("A", {"C": "1"}, 20020 / 550073),
    ],
)
def test_misconception_marginals_are_exact(var, evidence, expected):
    assert misconception().query([var], evidence).value({var: "1"}) == pytest.approx(
        expected, abs=1e-12
    )


def test_probability_of_an_assignment():
    # φ1(a1,b1) φ2(b1,c0) φ3(c0,d1) φ4(d1,a1) = 10 · 1 · 100 · 100
    mn = misconception()
    x = {"A": "1", "B": "1", "C": "0", "D": "1"}
    assert mn.probability(x) == pytest.approx(100_000 / MISCONCEPTION_Z, rel=1e-12)
    assert mn.log_probability(x) == pytest.approx(math.log(100_000 / MISCONCEPTION_Z), abs=1e-12)


def test_probabilities_sum_to_one():
    mn = misconception()
    total = sum(
        mn.probability(dict(zip("ABCD", states, strict=True)))
        for states in itertools.product("01", repeat=4)
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_graph_is_the_four_cycle():
    g = misconception().graph
    assert {frozenset(e) for e in g.edges()} == {frozenset(p) for p in ["AB", "BC", "CD", "DA"]}


def test_misconception_separation_facts():
    mn = misconception()
    table = gibbs_table(STUDENTS, misconception_factors())
    a, b, c, d = 0, 1, 2, 3
    assert mn.separated({"A"}, {"C"}, {"B", "D"})
    assert ci_gap(table / table.sum(), [a], [c], [b, d]) <= 1e-15
    assert mn.separated({"B"}, {"D"}, {"A", "C"})
    assert not mn.separated({"A"}, {"C"}, {"B"})
    assert ci_gap(table / table.sum(), [a], [c], [b]) > 1e-3


def test_evidence_probability_is_a_ratio_of_partition_functions():
    mn = misconception()
    table = gibbs_table(STUDENTS, misconception_factors())
    assert mn.log_probability_of_evidence({"C": "1"}) == pytest.approx(
        math.log(table[:, :, 1, :].sum() / table.sum()), abs=1e-12
    )
    assert mn.log_probability_of_evidence({}) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# M1 / M2: validation and Z > 0
# ---------------------------------------------------------------------------


def test_factor_with_unknown_variable_is_rejected():
    stranger = DiscreteVariable("E", ("0", "1"))
    with pytest.raises(ValidationError, match="E"):
        MarkovNetwork(STUDENTS, [DiscreteFactor([stranger], [1.0, 1.0])])


def test_factor_with_conflicting_domain_is_rejected():
    other_a = DiscreteVariable("A", ("0", "1", "2"))
    with pytest.raises(ValidationError, match="domain"):
        MarkovNetwork(STUDENTS, [DiscreteFactor([other_a], [1.0, 1.0, 1.0])])


def test_duplicate_variables_and_non_factors_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        MarkovNetwork([*STUDENTS, STUDENTS[0]], [])
    with pytest.raises(ValidationError):
        MarkovNetwork(STUDENTS, [np.ones(2)])  # type: ignore[list-item]


def test_all_zero_model_is_rejected_when_z_is_needed():
    a = STUDENTS[0]
    mn = MarkovNetwork([a], [DiscreteFactor([a], [0.0, 0.0])])
    with pytest.raises(ValidationError, match="partition function"):
        mn.log_partition_function()
    with pytest.raises(ValidationError, match="partition function"):
        mn.query(["A"])


def test_isolated_variable_is_uniform_and_independent():
    a, b = STUDENTS[:2]
    mn = MarkovNetwork([a, b], [DiscreteFactor([a], [1.0, 3.0])])
    assert "B" in mn.graph and mn.graph.neighbours("B") == set()
    assert mn.query(["B"]).values.tolist() == pytest.approx([0.5, 0.5])
    assert mn.query(["A"], {"B": "1"}).values.tolist() == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize(
    ("query", "evidence", "error"),
    [
        (["E"], {}, UnknownNodeError),
        ([], {}, ValidationError),
        (["A"], {"A": "1"}, ValidationError),
        (["A"], {"B": "7"}, UnknownStateError),
    ],
)
def test_query_validation(query, evidence, error):
    with pytest.raises(error):
        misconception().query(query, evidence)


def test_impossible_evidence_raises():
    a, b = STUDENTS[:2]
    mn = MarkovNetwork(
        [a, b], [DiscreteFactor([a, b], [[1.0, 0.0], [0.0, 1.0]]), DiscreteFactor([a], [1.0, 0.0])]
    )
    with pytest.raises(ZeroProbabilityEvidenceError):
        mn.query(["A"], {"B": "1"})
    assert mn.log_probability_of_evidence({"B": "1"}) == -math.inf


def test_probability_requires_a_complete_assignment():
    with pytest.raises(ValidationError, match="missing"):
        misconception().probability({"A": "1"})


def test_graph_is_a_copy():
    mn = misconception()
    mn.graph.add_edge("A", "C")
    assert not mn.graph.has_edge("A", "C")


# ---------------------------------------------------------------------------
# Log space: potentials beyond float64
# ---------------------------------------------------------------------------


def test_huge_partition_function_is_finite_in_log_space():
    variables = [DiscreteVariable(f"X{i}", ("0", "1")) for i in range(200)]
    factors = [DiscreteFactor([v], [100.0, 1.0]) for v in variables]
    mn = MarkovNetwork(variables, factors)
    assert mn.log_partition_function() == pytest.approx(200 * math.log(101.0), rel=1e-12)
    with pytest.raises(ValidationError, match="overflow"):
        mn.partition_function()
    assert mn.query(["X7"]).value({"X7": "0"}) == pytest.approx(100 / 101, abs=1e-12)


def test_log_factor_potentials_are_accepted():
    a = STUDENTS[0]
    mn = MarkovNetwork([a], [LogFactor([a], [1000.0, 1000.0 + math.log(3)])])
    assert mn.query(["A"]).values.tolist() == pytest.approx([0.25, 0.75])
    assert mn.log_partition_function() == pytest.approx(1000.0 + math.log(4), abs=1e-9)


# ---------------------------------------------------------------------------
# M3 / M4: random Markov networks against brute force
# ---------------------------------------------------------------------------


def random_markov_network(seed: int, positive: bool = False):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    variables = [
        DiscreteVariable(f"V{i}", tuple(f"s{j}" for j in range(int(rng.integers(2, 4)))))
        for i in range(n)
    ]
    factors = []
    for _ in range(int(rng.integers(1, 6))):
        k = int(rng.integers(1, min(3, n) + 1))
        scope = [variables[i] for i in rng.choice(n, size=k, replace=False)]
        table = rng.uniform(0.1, 3.0, size=tuple(v.cardinality for v in scope))
        if not positive:
            table[rng.random(table.shape) < 0.15] = 0.0
        factors.append(DiscreteFactor(scope, table))
    return variables, factors


@pytest.mark.parametrize("seed", range(40))
def test_random_networks_match_brute_force(seed):
    variables, factors = random_markov_network(seed)
    table = gibbs_table(variables, factors)
    if table.sum() == 0.0:
        pytest.skip("this random model has no positive assignment")
    mn = MarkovNetwork(variables, factors)
    assert mn.log_partition_function() == pytest.approx(math.log(table.sum()), abs=1e-10)
    joint = table / table.sum()
    for i, v in enumerate(variables):
        expected = joint.sum(axis=tuple(j for j in range(len(variables)) if j != i))
        np.testing.assert_allclose(mn.query([v.name]).values, expected, atol=1e-12)


@pytest.mark.parametrize("seed", range(30))
def test_global_markov_property_and_generic_completeness(seed):
    """Theorem 1, numerically; and, for random positive potentials, its converse holds too."""
    variables, factors = random_markov_network(seed, positive=True)
    mn = MarkovNetwork(variables, factors)
    table = gibbs_table(variables, factors)
    joint = table / table.sum()
    names = [v.name for v in variables]
    for x, y in itertools.combinations(range(len(names)), 2):
        rest = [i for i in range(len(names)) if i not in (x, y)]
        for r in range(len(rest) + 1):
            for z in itertools.combinations(rest, r):
                gap = ci_gap(joint, [x], [y], list(z))
                if mn.separated({names[x]}, {names[y]}, {names[i] for i in z}):
                    assert gap <= 1e-12
                else:
                    assert gap > 1e-9


# ---------------------------------------------------------------------------
# M5 / M6: from a Bayesian network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [rain_network, late_network])
def test_conversion_preserves_the_distribution(build):
    bn = build()
    mn = bn.to_markov_network()
    assert mn.graph == moral_graph(bn.graph)
    assert mn.log_partition_function() == pytest.approx(0.0, abs=1e-12)  # Z = 1
    names = [v.name for v in bn.variables]
    table = joint_table(bn)
    for index in itertools.product(*(range(v.cardinality) for v in bn.variables)):
        x = {v.name: v.states[i] for v, i in zip(bn.variables, index, strict=True)}
        assert mn.probability(x) == pytest.approx(table[index], abs=1e-12)
    ve = VariableElimination(bn)
    for name in names:
        assert mn.query([name], {}).allclose(ve.query([name]), atol=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_conversion_on_random_networks(seed):
    bn = random_network(seed, n_vars=(2, 6), cards=(2, 3))
    mn = bn.to_markov_network()
    assert mn.graph == moral_graph(bn.graph)
    assert mn.log_partition_function() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(gibbs_table(bn.variables, mn.factors), joint_table(bn), atol=1e-15)


def test_moralisation_loses_the_collider_independence():
    bn = rain_network()
    mn = bn.to_markov_network()
    assert bn.graph.d_separated({"Accident"}, {"Rain"})  # A ⊥ R in the BN graph
    assert not mn.separated({"Accident"}, {"Rain"})  # married in the moral graph
    table = joint_table(bn)
    assert ci_gap(table, [1], [0], []) <= 1e-15  # yet still true of the distribution


@pytest.mark.parametrize("seed", range(40))
def test_proposition_3_moral_separation_implies_d_separation(seed):
    dag = random_network(seed, n_vars=(3, 8), edge_prob=0.35).graph
    moral = moral_graph(dag)
    nodes = list(dag.nodes())
    rng = np.random.default_rng(seed)
    for _ in range(20):
        perm = [str(n) for n in rng.permutation(nodes)]
        x, y = {perm[0]}, {perm[1]}
        z = {n for n in perm[2:] if rng.random() < 0.5}
        if moral.separated(x, y, z):
            assert dag.d_separated(x, y, z)


def test_proposition_4_no_dag_captures_the_four_cycle():
    cycle = UndirectedGraph("ABCD", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    queries = [
        (x, y, frozenset(z))
        for x, y in itertools.combinations("ABCD", 2)
        for r in range(3)
        for z in itertools.combinations([n for n in "ABCD" if n not in (x, y)], r)
    ]
    assert len(queries) == 24
    target = {q: cycle.separated({q[0]}, {q[1]}, q[2]) for q in queries}

    pairs = list(itertools.combinations("ABCD", 2))
    matches = dags = 0
    for choice in itertools.product((None, 0, 1), repeat=len(pairs)):
        edges = [
            (u, v) if c == 0 else (v, u)
            for (u, v), c in zip(pairs, choice, strict=True)
            if c is not None
        ]
        try:
            dag = DAG(nodes="ABCD", edges=edges)
        except ValidationError:
            continue
        dags += 1
        if all(dag.d_separated({x}, {y}, z) == sep for (x, y, z), sep in target.items()):
            matches += 1
    assert dags == 543
    assert matches == 0
