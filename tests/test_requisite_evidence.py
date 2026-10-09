"""M2.9: requisite evidence (spec task 9; d_separation.md §9, Proposition 6)."""

import numpy as np
import pytest
from support import late_network, posterior, random_evidence, random_network

from probgraph import DAG, BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import UnknownNodeError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import HEURISTICS

Y, N = "yes", "no"

# ---------------------------------------------------------------------------
# DAG.requisite_evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "evidence", "requisite"),
    [
        # Explaining away through two edges: both observations matter.
        (["Accident"], ["Late", "Umbrella"], {"Late", "Umbrella"}),
        # Without Late, the collider at Traffic is closed: Umbrella cannot matter.
        (["Accident"], ["Umbrella"], set()),
        # Observing Rain blocks the fork, so Late is irrelevant to Umbrella.
        (["Umbrella"], ["Rain", "Late"], {"Rain"}),
        # Traffic blocks the chain to Late, but opens the collider towards Rain.
        (["Accident"], ["Traffic", "Late", "Rain"], {"Traffic", "Rain"}),
        ([], [], set()),
    ],
)
def test_fixture_requisite_evidence(query, evidence, requisite):
    if not query:
        with pytest.raises(ValidationError, match="empty"):
            late_network().graph.requisite_evidence(query, evidence)
        return
    assert late_network().graph.requisite_evidence(query, evidence) == requisite


def test_requisite_evidence_validation():
    dag = DAG(nodes="ABC", edges=[("A", "B"), ("B", "C")])
    with pytest.raises(ValidationError, match="disjoint"):
        dag.requisite_evidence({"A"}, {"A", "C"})
    with pytest.raises(UnknownNodeError):
        dag.requisite_evidence({"A"}, {"Z"})
    with pytest.raises(ValidationError, match="string"):
        dag.requisite_evidence("A", {"C"})


@pytest.mark.parametrize("seed", range(100))
def test_proposition_6_dropped_evidence_is_d_separated(seed):
    rng = np.random.default_rng(seed + 110_000)
    dag = random_network(seed, n_vars=(3, 9), edge_prob=0.35).graph
    perm = [str(n) for n in rng.permutation(list(dag.nodes()))]
    query = perm[: int(rng.integers(1, 3))]
    evidence = {n for n in perm[len(query) :] if rng.random() < 0.5}
    requisite = dag.requisite_evidence(query, evidence)
    assert requisite <= evidence
    dropped = evidence - requisite
    if dropped:
        assert dag.d_separated(query, dropped, requisite)


# ---------------------------------------------------------------------------
# VariableElimination with prune_evidence=True
# ---------------------------------------------------------------------------


def test_fixture_irrelevant_umbrella_is_dropped():
    ve = VariableElimination(late_network())
    evidence = {"Umbrella": Y}
    assert ve.requisite_evidence(["Accident"], evidence) == {}
    pruned = ve.query(["Accident"], evidence, prune_evidence=True)
    assert pruned.value({"Accident": Y}) == pytest.approx(1 / 10, abs=1e-12)
    # With Umbrella dropped, everything except Accident is barren: nothing to eliminate.
    assert ve.query_trace(["Accident"], evidence, prune_evidence=True).steps == ()
    assert ve.query_trace(["Accident"], evidence).steps != ()


def test_dropped_ancestor_stays_observed():
    """W -> V -> Q with W and V observed: only V is requisite, but W is still an ancestor."""
    w, v, q = (DiscreteVariable(n, ("0", "1")) for n in "WVQ")
    model = BayesianNetwork([w, v, q], [("W", "V"), ("V", "Q")])
    model.add_cpd(TabularCPD(w, (), [0.3, 0.7]))
    model.add_cpd(TabularCPD(v, (w,), [[0.9, 0.2], [0.1, 0.8]]))
    model.add_cpd(TabularCPD(q, (v,), [[0.6, 0.25], [0.4, 0.75]]))
    ve = VariableElimination(model)
    evidence = {"W": "1", "V": "0"}
    assert ve.requisite_evidence(["Q"], evidence) == {"V": "0"}
    expected = posterior(model, ["Q"], evidence)
    for order in ("min_fill", []):  # nothing is left to eliminate
        result = ve.query(["Q"], evidence, elimination_order=order, prune_evidence=True)
        assert result.allclose(expected, atol=1e-12)
    assert result.value({"Q": "1"}) == pytest.approx(0.4)


def test_fixture_relevant_evidence_is_kept():
    ve = VariableElimination(late_network())
    evidence = {"Late": Y, "Umbrella": Y}
    assert ve.requisite_evidence(["Accident"], evidence) == evidence
    result = ve.query(["Accident"], evidence, prune_evidence=True)
    assert result.value({"Accident": Y}) == pytest.approx(475 / 3787, abs=1e-12)


@pytest.mark.parametrize("seed", range(100))
def test_pruned_answers_match_enumeration(seed):
    model = random_network(seed, n_vars=(2, 7), edge_prob=0.35)
    rng = np.random.default_rng(seed + 120_000)
    names = [v.name for v in model.variables]
    query = [str(rng.choice(names))]
    evidence = random_evidence(model, rng, exclude=query)
    expected = posterior(model, query, evidence)  # Dirichlet CPDs: P(e) > 0
    ve = VariableElimination(model)
    for heuristic in HEURISTICS:
        result = ve.query(query, evidence, elimination_order=heuristic, prune_evidence=True)
        assert result.allclose(expected, atol=1e-12)
    # With a fixed order, dropping evidence never makes a step larger (fill-path lemma).
    order = [
        str(n) for n in rng.permutation([n for n in names if n not in query and n not in evidence])
    ]
    explicit = ve.query(query, evidence, elimination_order=order, prune_evidence=True)
    assert explicit.allclose(expected, atol=1e-12)
    pruned = ve.query_trace(query, evidence, elimination_order=order, prune_evidence=True)
    full = ve.query_trace(query, evidence, elimination_order=order)
    assert pruned.total_cost <= full.total_cost


def test_explicit_order_refers_to_the_original_evidence():
    """An explicit order lists the variables outside the *original* query and evidence."""
    ve = VariableElimination(late_network())
    evidence = {"Umbrella": Y}
    order = ["Rain", "Traffic", "Late"]  # excludes Umbrella, which is observed
    result = ve.query(["Accident"], evidence, elimination_order=order, prune_evidence=True)
    assert result.value({"Accident": Y}) == pytest.approx(0.1, abs=1e-12)


# ---------------------------------------------------------------------------
# The zero-probability caveat (§9): why evidence pruning is opt-in
# ---------------------------------------------------------------------------


def impossible_but_irrelevant() -> BayesianNetwork:
    x = DiscreteVariable("X", ("a", "b"))
    switch = DiscreteVariable("Switch", ("off", "on"))
    model = BayesianNetwork([x, switch], [])
    model.add_cpd(TabularCPD(x, (), [0.25, 0.75]))
    model.add_cpd(TabularCPD(switch, (), [1.0, 0.0]))  # "on" is impossible
    return model


def test_default_still_raises_on_impossible_irrelevant_evidence():
    ve = VariableElimination(impossible_but_irrelevant())
    with pytest.raises(ZeroProbabilityEvidenceError):
        ve.query(["X"], {"Switch": "on"})
    assert ve.probability_of_evidence({"Switch": "on"}) == 0.0


def test_opt_in_pruning_returns_the_answer_given_the_requisite_evidence():
    ve = VariableElimination(impossible_but_irrelevant())
    result = ve.query(["X"], {"Switch": "on"}, prune_evidence=True)
    assert result.value({"X": "b"}) == pytest.approx(0.75)  # P(X), since Switch is dropped
