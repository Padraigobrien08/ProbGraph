"""M3.7: evidence, P(e), zero detection and cost.

Spec task 7; invariants C4, C7 (as revised), C8; message_passing.md §8–9.
"""

import math

import numpy as np
import pytest
from support import (
    STUDENTS,
    evidence_probability,
    late_network,
    misconception_factors,
    posterior,
    random_evidence,
    random_network,
    underflow_evidence,
    underflow_network,
)

from probgraph import (
    BayesianNetwork,
    DiscreteFactor,
    DiscreteVariable,
    MarkovNetwork,
    TabularCPD,
    VariableElimination,
)
from probgraph.exceptions import (
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)
from probgraph.inference import JunctionTree

Y, N = "yes", "no"

# ---------------------------------------------------------------------------
# Fixtures with evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("evidence", "accident", "log_p_e"),
    [
        ({"Late": Y}, 65 / 371, math.log(1113 / 4000)),
        ({"Late": Y, "Umbrella": Y}, 475 / 3787, math.log(11361 / 80000)),
        ({"Late": Y, "Umbrella": N}, 275 / 1211, None),
        ({"Traffic": Y, "Umbrella": Y}, 1165 / 8761, None),
    ],
)
def test_late_network_with_evidence(evidence, accident, log_p_e):
    jt = JunctionTree(late_network(), evidence)
    assert jt.marginal("Accident").value({"Accident": Y}) == pytest.approx(accident, abs=1e-12)
    if log_p_e is not None:
        assert jt.log_probability_of_evidence() == pytest.approx(log_p_e, abs=1e-12)
    # Every unobserved marginal agrees with variable elimination.
    ve = VariableElimination(late_network())
    marginals = jt.marginals()
    assert set(marginals) == {v.name for v in late_network().variables} - set(evidence)
    for name, factor in marginals.items():
        assert factor.allclose(ve.query([name], evidence), atol=1e-12)


def test_observed_variables_leave_the_tree():
    jt = JunctionTree(late_network(), {"Traffic": Y})
    assert all("Traffic" not in c for c in jt.tree.cliques)
    # With Traffic observed, the moral graph minus Traffic splits into {A, R}, {R, U} and {L}.
    assert max(len(c) for c in jt.tree.cliques) == 2


def test_misconception_with_evidence():
    mn = MarkovNetwork(STUDENTS, misconception_factors())
    jt = JunctionTree(mn, {"C": "1"})
    assert jt.marginal("B").value({"B": "1"}) == pytest.approx(520050 / 550073, abs=1e-12)
    assert jt.marginal("A").value({"A": "1"}) == pytest.approx(20020 / 550073, abs=1e-12)
    assert jt.log_probability_of_evidence() == pytest.approx(math.log(550073 / 720184), abs=1e-12)
    assert jt.log_partition_function() == pytest.approx(math.log(7_201_840), abs=1e-12)


def test_observed_marginal_is_a_point_mass_and_joint_queries_reject_it():
    jt = JunctionTree(late_network(), {"Late": Y})
    assert jt.marginal("Late").values.tolist() == [0.0, 1.0]
    with pytest.raises(ValidationError, match="observed"):
        jt.query(["Late", "Traffic"])


# ---------------------------------------------------------------------------
# F3: the underflow regression, through the junction tree (C8)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def f3_model() -> BayesianNetwork:
    return underflow_network(1100)


def test_f3_balanced(f3_model):
    jt = JunctionTree(f3_model, underflow_evidence(550, 550))
    assert jt.marginal("C").values.tolist() == pytest.approx([0.5, 0.5], abs=1e-12)
    assert jt.log_probability_of_evidence() == pytest.approx(
        550 * math.log(0.6) + 550 * math.log(0.4), abs=1e-9
    )
    assert jt.tree.cliques == (frozenset({"C"}),)  # every feature observed


def test_f3_unbalanced(f3_model):
    jt = JunctionTree(f3_model, underflow_evidence(551, 549))
    assert jt.marginal("C").value({"C": "0"}) == pytest.approx(2.25 / 3.25, abs=1e-12)


# ---------------------------------------------------------------------------
# Impossible evidence (C8)
# ---------------------------------------------------------------------------


def switch_model() -> BayesianNetwork:
    switch = DiscreteVariable("Switch", ("off", "on"))
    light = DiscreteVariable("Light", ("off", "on"))
    model = BayesianNetwork([switch, light], [("Switch", "Light")])
    model.add_cpd(TabularCPD(switch, (), [1.0, 0.0]))
    model.add_cpd(TabularCPD(light, (switch,), np.eye(2)))
    return model


def test_impossible_evidence():
    jt = JunctionTree(switch_model(), {"Light": "on"})
    assert jt.log_probability_of_evidence() == -math.inf
    with pytest.raises(ZeroProbabilityEvidenceError):
        jt.marginal("Switch")
    with pytest.raises(ZeroProbabilityEvidenceError):
        jt.marginals()
    possible = JunctionTree(switch_model(), {"Light": "off"})
    assert possible.marginal("Switch").values.tolist() == [1.0, 0.0]


def test_everything_observed():
    jt = JunctionTree(
        late_network(), {"Rain": Y, "Accident": N, "Traffic": Y, "Late": Y, "Umbrella": Y}
    )
    assert jt.tree.cliques == ()
    p = 0.3 * 0.9 * 0.8 * 0.6 * 0.85
    assert jt.log_probability_of_evidence() == pytest.approx(math.log(p), abs=1e-12)
    assert jt.marginals() == {}


@pytest.mark.parametrize(
    ("evidence", "error"),
    [
        ({"Snow": Y}, UnknownNodeError),
        ({"Late": "very"}, UnknownStateError),
        ([("Late", Y)], ValidationError),
    ],
)
def test_evidence_validation(evidence, error):
    with pytest.raises(error):
        JunctionTree(late_network(), evidence)


# ---------------------------------------------------------------------------
# Random models with evidence (C4, C5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_random_bayesian_networks_with_evidence(seed):
    bn = random_network(seed, n_vars=(2, 7), edge_prob=0.4)
    rng = np.random.default_rng(seed + 150_000)
    evidence = random_evidence(bn, rng)
    jt = JunctionTree(bn, evidence)
    assert jt.log_probability_of_evidence() == pytest.approx(
        math.log(evidence_probability(bn, evidence)), abs=1e-10
    )
    for name, marginal in jt.marginals().items():
        assert marginal.allclose(posterior(bn, [name], evidence), atol=1e-12)
    # C4: every clique has the same total, log Z(e).
    for i in range(len(jt.tree.cliques)):
        assert jt.clique_belief(i).log_total() == pytest.approx(
            jt.log_probability_of_evidence(), abs=1e-9
        )


@pytest.mark.parametrize("seed", range(30))
def test_random_markov_networks_with_evidence(seed):
    rng = np.random.default_rng(seed + 160_000)
    n = int(rng.integers(2, 7))
    variables = [
        DiscreteVariable(f"V{i}", ("a", "b", "c")[: int(rng.integers(2, 4))]) for i in range(n)
    ]
    factors = []
    for _ in range(int(rng.integers(2, 8))):
        k = int(rng.integers(1, min(3, n) + 1))
        scope = [variables[i] for i in rng.choice(n, size=k, replace=False)]
        factors.append(
            DiscreteFactor(scope, rng.uniform(0.1, 3.0, size=tuple(v.cardinality for v in scope)))
        )
    mn = MarkovNetwork(variables, factors)
    evidence = {v.name: v.states[0] for v in variables if rng.random() < 0.4}
    jt = JunctionTree(mn, evidence)
    assert jt.log_probability_of_evidence() == pytest.approx(
        mn.log_probability_of_evidence(evidence), abs=1e-10
    )
    for name, marginal in jt.marginals().items():
        assert marginal.allclose(mn.query([name], evidence), atol=1e-12)


# ---------------------------------------------------------------------------
# C7: cost of all marginals, one calibration against n variable-elimination queries
# ---------------------------------------------------------------------------


def chain(n: int) -> BayesianNetwork:
    rng = np.random.default_rng(n)
    xs = [DiscreteVariable(f"X{i}", ("0", "1")) for i in range(n)]
    bn = BayesianNetwork(xs, [(xs[i].name, xs[i + 1].name) for i in range(n - 1)])
    bn.add_cpd(TabularCPD(xs[0], (), [0.5, 0.5]))
    for i in range(1, n):
        bn.add_cpd(TabularCPD(xs[i], (xs[i - 1],), rng.dirichlet([1.0, 1.0], size=2).T))
    return bn


def ve_cost_of_all_marginals(model, evidence=None) -> int:
    ve = VariableElimination(model)
    evidence = evidence or {}
    return sum(
        ve.query_trace([v.name], evidence).total_cost
        for v in model.variables
        if v.name not in evidence
    )


@pytest.mark.parametrize("n", [10, 30, 60])
def test_calibration_is_cheaper_than_repeated_elimination_on_chains(n):
    bn = chain(n)
    jt = JunctionTree(bn)
    ve_cost = ve_cost_of_all_marginals(bn)
    # message_passing.md §9: about 12n cells for calibration, about 2n² for n VE queries.
    assert jt.calibration_cost() == 2 * (n - 2) * 4 + (n - 1) * 4
    assert jt.calibration_cost() < ve_cost
    if n >= 30:
        assert ve_cost > 4 * jt.calibration_cost()


def test_cost_on_the_fixture_is_reported_honestly():
    """On a 5-variable model the two are comparable; the advantage is asymptotic (§9)."""
    bn = late_network()
    jt_cost, ve_cost = JunctionTree(bn).calibration_cost(), ve_cost_of_all_marginals(bn)
    assert jt_cost == 2 * 8 + 4 + 4 + (8 + 4 + 4)  # messages from {A,R,T} ×2, {L,T}, {R,U}; beliefs
    assert 0.5 < ve_cost / jt_cost < 2.0


def sparse_network(seed: int) -> BayesianNetwork:
    return random_network(seed, n_vars=(25, 30), cards=(2, 2), edge_prob=0.08)


def leaf_evidence(bn: BayesianNetwork) -> dict[str, str]:
    g = bn.graph
    return {n: "s0" for n in g.nodes() if not g.children(n) and g.parents(n)}


@pytest.mark.parametrize("seed", range(10))
def test_calibration_wins_when_evidence_is_downstream(seed):
    """Evidence at the leaves keeps every ancestor, so pruning cannot shrink VE's queries."""
    bn = sparse_network(seed)
    evidence = leaf_evidence(bn)
    jt_cost = JunctionTree(bn, evidence).calibration_cost()
    assert 3 * jt_cost < ve_cost_of_all_marginals(bn, evidence)


@pytest.mark.parametrize("seed", range(10))
def test_calibration_beats_unpruned_elimination(seed):
    bn = sparse_network(seed)
    ve = VariableElimination(bn)
    unpruned = sum(ve.query_trace([v.name], prune_barren=False).total_cost for v in bn.variables)
    assert 5 * JunctionTree(bn).calibration_cost() < unpruned


def test_without_evidence_pruned_elimination_can_be_cheaper():
    """The honest counterpart (message_passing.md §9): with no evidence, P(X) needs only
    X's ancestors, which are few in a sparse DAG, while calibration pays for the whole tree."""
    wins = sum(
        ve_cost_of_all_marginals(sparse_network(s))
        < JunctionTree(sparse_network(s)).calibration_cost()
        for s in range(10)
    )
    assert wins >= 5
