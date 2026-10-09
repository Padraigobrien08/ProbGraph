"""M3.6: Shafer–Shenoy calibration (spec task 6; C1–C3, C5, C6; message_passing.md)."""

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
    random_network,
)

from probgraph import DiscreteFactor, DiscreteVariable, MarkovNetwork, VariableElimination
from probgraph.exceptions import ValidationError
from probgraph.factors import LogFactor
from probgraph.inference import CliqueTree, JunctionTree
from probgraph.inference.junction_tree import _calibrate

Y = "yes"


def misconception() -> MarkovNetwork:
    return MarkovNetwork(STUDENTS, misconception_factors())


def brute_force_marginals(variables, factors) -> dict[str, np.ndarray]:
    table = gibbs_table(variables, factors)
    joint = table / table.sum()
    return {
        v.name: joint.sum(axis=tuple(j for j in range(len(variables)) if j != i))
        for i, v in enumerate(variables)
    }


def assert_calibrated(jt: JunctionTree) -> None:
    """C2: neighbouring beliefs agree on their separator."""
    tree = jt.tree
    for i, j in tree.edges:
        s = tree.separator(i, j)
        left = jt.clique_belief(i).marginalise(tree.cliques[i] - s)
        right = jt.clique_belief(j).marginalise(tree.cliques[j] - s)
        assert left.allclose(right, atol=1e-9)


# ---------------------------------------------------------------------------
# F1: the Late network, all marginals from one calibration
# ---------------------------------------------------------------------------


def test_late_network_marginals_in_one_calibration():
    jt = JunctionTree(late_network())
    ve = VariableElimination(late_network())
    marginals = jt.marginals()
    assert set(marginals) == {"Rain", "Accident", "Traffic", "Late", "Umbrella"}
    for name, factor in marginals.items():
        assert factor.allclose(ve.query([name]), atol=1e-12)
    assert marginals["Late"].value({"Late": Y}) == pytest.approx(1113 / 4000, abs=1e-12)
    assert jt.message_count() == 2 * (len(jt.tree.cliques) - 1) == 4  # C1
    assert jt.log_partition_function() == pytest.approx(0.0, abs=1e-12)  # Z = 1 for a BN
    assert_calibrated(jt)


def test_query_within_one_clique():
    jt = JunctionTree(late_network())
    joint = jt.query(["Rain", "Accident", "Traffic"])
    expected = VariableElimination(late_network()).query(["Rain", "Accident", "Traffic"])
    assert joint.allclose(expected, atol=1e-12)
    assert jt.query(["Traffic", "Late"]).names == ("Traffic", "Late")


def test_query_across_cliques_is_refused():
    with pytest.raises(ValidationError, match="VariableElimination"):
        JunctionTree(late_network()).query(["Umbrella", "Late"])


# ---------------------------------------------------------------------------
# F2: the misconception network
# ---------------------------------------------------------------------------


def test_misconception_calibration():
    jt = JunctionTree(misconception())
    assert jt.log_partition_function() == pytest.approx(math.log(MISCONCEPTION_Z), abs=1e-12)
    m = jt.marginals()
    for name, numerator in [("A", 130031), ("B", 530151), ("C", 550073), ("D", 150113)]:
        assert m[name].value({name: "1"}) == pytest.approx(numerator / 720184, abs=1e-12)
    assert set(jt.tree.cliques) == {frozenset("ABD"), frozenset("BCD")}
    assert jt.query(["B", "D"]).allclose(misconception().query(["B", "D"]), atol=1e-12)
    assert_calibrated(jt)


def test_every_clique_has_the_same_total():
    """C4: log Σ β_i = log Z for every clique."""
    jt = JunctionTree(misconception())
    for i in range(len(jt.tree.cliques)):
        assert jt.clique_belief(i).log_total() == pytest.approx(math.log(MISCONCEPTION_Z), abs=1e-9)


# ---------------------------------------------------------------------------
# Random models against brute force (C2, C3, C5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_random_bayesian_networks(seed):
    bn = random_network(seed, n_vars=(1, 7), edge_prob=0.4)
    jt = JunctionTree(bn)
    table = joint_table(bn)
    for i, v in enumerate(bn.variables):
        expected = table.sum(axis=tuple(j for j in range(table.ndim) if j != i))
        np.testing.assert_allclose(jt.marginal(v.name).values, expected, atol=1e-12)
    assert jt.message_count() == 2 * max(len(jt.tree.cliques) - 1, 0)
    assert_calibrated(jt)


def random_markov_network(seed: int) -> MarkovNetwork:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    variables = [
        DiscreteVariable(f"V{i}", tuple(f"s{j}" for j in range(int(rng.integers(2, 4)))))
        for i in range(n)
    ]
    factors = []
    for _ in range(int(rng.integers(2, 8))):
        k = int(rng.integers(1, min(3, n) + 1))
        scope = [variables[i] for i in rng.choice(n, size=k, replace=False)]
        factors.append(
            DiscreteFactor(scope, rng.uniform(0.1, 3.0, size=tuple(v.cardinality for v in scope)))
        )
    return MarkovNetwork(variables, factors)


@pytest.mark.parametrize("seed", range(40))
def test_random_markov_networks(seed):
    mn = random_markov_network(seed)
    jt = JunctionTree(mn)
    expected = brute_force_marginals(mn.variables, mn.factors)
    for name, marginal in jt.marginals().items():
        np.testing.assert_allclose(marginal.values, expected[name], atol=1e-12)
    assert jt.log_partition_function() == pytest.approx(mn.log_partition_function(), abs=1e-10)
    # C3: each normalised clique belief is the joint marginal of its clique.
    for i, clique in enumerate(jt.tree.cliques):
        names = sorted(clique)
        belief = jt.clique_belief(i).normalise().to_factor()
        assert belief.allclose(mn.query(names), atol=1e-12)
    assert_calibrated(jt)


# ---------------------------------------------------------------------------
# C6: the schedule and the tree do not matter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_root_and_elimination_order_do_not_change_marginals(seed):
    mn = random_markov_network(seed + 500)
    reference = JunctionTree(mn).marginals()
    rng = np.random.default_rng(seed)
    names = [v.name for v in mn.variables]
    trees = [JunctionTree(mn, root=r) for r in range(len(JunctionTree(mn).tree.cliques))]
    trees += [JunctionTree(mn, elimination_order=h) for h in ("min_neighbours", "min_weight")]
    trees += [JunctionTree(mn, elimination_order=[str(v) for v in rng.permutation(names)])]
    for jt in trees:
        for name, marginal in jt.marginals().items():
            assert marginal.allclose(reference[name], atol=1e-12)


# ---------------------------------------------------------------------------
# Why RIP matters (message_passing.md §3)
# ---------------------------------------------------------------------------


def test_breaking_rip_gives_wrong_marginals():
    """Clusters {A,B}, {C}, {B,C} on a path: B is in clusters 0 and 2 but not 1, so RIP fails.

    The {A,B} side never learns that B also appears in φ_BC, so its belief is φ_AB alone.
    With Σ_c φ_BC(b, c) = (10, 2) depending on b, that is measurably wrong. The proper
    tree {A,B} – {B,C} gets the exact joint marginal of A and B.
    """
    a, b, c = (DiscreteVariable(n, ("0", "1")) for n in "ABC")
    phi_ab = DiscreteFactor([a, b], [[5.0, 1.0], [1.0, 5.0]])
    phi_bc = DiscreteFactor([b, c], [[9.0, 1.0], [1.0, 1.0]])
    logs = [LogFactor.from_factor(phi_ab), LogFactor.from_factor(phi_bc)]
    table = gibbs_table([a, b, c], [phi_ab, phi_bc])
    true_ab = table.sum(axis=2) / table.sum()

    good = CliqueTree([frozenset("AB"), frozenset("BC")], [(0, 1)])
    good_beliefs, _ = _calibrate(good, logs, root=0)
    np.testing.assert_allclose(
        good_beliefs[0].normalise().to_factor().aligned(["A", "B"]).values, true_ab, atol=1e-12
    )

    bad = CliqueTree([frozenset("AB"), frozenset("C"), frozenset("BC")], [(0, 1), (1, 2)])
    bad_beliefs, _ = _calibrate(bad, [logs[0], LogFactor([c], [0.0, 0.0]), logs[1]], root=0)
    bad_ab = bad_beliefs[0].normalise().to_factor().aligned(["A", "B"]).values
    assert np.abs(bad_ab - true_ab).max() > 0.1


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


def test_isolated_variables_and_single_clique():
    a, b = STUDENTS[:2]
    mn = MarkovNetwork([a, b], [DiscreteFactor([a], [1.0, 3.0])])
    jt = JunctionTree(mn)
    assert jt.marginal("B").values.tolist() == pytest.approx([0.5, 0.5])
    assert jt.marginal("A").values.tolist() == pytest.approx([0.25, 0.75])


def test_huge_potentials_stay_in_log_space():
    variables = [DiscreteVariable(f"X{i}", ("0", "1")) for i in range(200)]
    mn = MarkovNetwork(variables, [DiscreteFactor([v], [100.0, 1.0]) for v in variables])
    jt = JunctionTree(mn)
    assert jt.log_partition_function() == pytest.approx(200 * math.log(101.0), rel=1e-12)
    assert jt.marginal("X42").value({"X42": "0"}) == pytest.approx(100 / 101, abs=1e-12)


def test_zero_partition_function_is_rejected():
    a = STUDENTS[0]
    with pytest.raises(ValidationError, match="partition function"):
        JunctionTree(MarkovNetwork([a], [DiscreteFactor([a], [0.0, 0.0])]))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"elimination_order": "min_size"}, ValidationError),
        ({"elimination_order": ["A"]}, ValidationError),
        ({"root": 9}, ValidationError),
    ],
)
def test_validation(kwargs, error):
    with pytest.raises(error):
        JunctionTree(misconception(), **kwargs)


def test_unknown_variable():
    with pytest.raises(Exception, match="Z9"):
        JunctionTree(misconception()).marginal("Z9")


def test_snapshot_semantics():
    bn = late_network()
    jt = JunctionTree(bn)
    before = jt.marginal("Rain")
    from probgraph import TabularCPD

    bn.add_cpd(TabularCPD(bn.variable("Rain"), (), [0.0, 1.0]))
    assert jt.marginal("Rain").allclose(before)


def test_pairs_of_marginals_match_brute_force_for_all_cliques():
    mn = misconception()
    jt = JunctionTree(mn)
    table = gibbs_table(mn.variables, mn.factors)
    joint = table / table.sum()
    for clique in jt.tree.cliques:
        for pair in itertools.combinations(sorted(clique), 2):
            axes = tuple(i for i, v in enumerate("ABCD") if v not in pair)
            np.testing.assert_allclose(
                jt.query(list(pair)).values, joint.sum(axis=axes), atol=1e-12
            )
