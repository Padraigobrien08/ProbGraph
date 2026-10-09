"""M2.6: barren-node pruning (spec task 6; invariant V7).

Proposition 5 of docs/mathematics/variable_elimination.md §5 is checked against
a test-local simulation of repeated leaf deletion, and V7 against the enumeration
oracle.
"""

import numpy as np
import pytest
from support import (
    evidence_probability,
    late_network,
    posterior,
    random_evidence,
    random_network,
)

from probgraph import DAG, VariableElimination
from probgraph.exceptions import UnknownNodeError
from probgraph.inference import HEURISTICS

Y = "yes"


def repeatedly_delete_barren_leaves(dag: DAG, keep: set[str]) -> set[str]:
    """Delete leaves outside ``keep`` until none remain, on a test-local copy of the graph."""
    children = {n: set(dag.children(n)) for n in dag.nodes()}
    parents = {n: set(dag.parents(n)) for n in dag.nodes()}
    alive = set(dag.nodes())
    changed = True
    while changed:
        changed = False
        for n in sorted(alive):
            if n not in keep and not (children[n] & alive):
                alive.remove(n)
                changed = True
    assert all(parents[n] <= alive for n in alive)  # the survivors form an ancestral set
    return alive


# ---------------------------------------------------------------------------
# DAG.ancestral_set
# ---------------------------------------------------------------------------


def test_ancestral_set_includes_the_nodes_themselves():
    dag = DAG(nodes="RATLU", edges=[("R", "T"), ("A", "T"), ("T", "L"), ("R", "U")])
    assert dag.ancestral_set({"L"}) == {"L", "T", "R", "A"}
    assert dag.ancestral_set({"A"}) == {"A"}
    assert dag.ancestral_set({"U", "A"}) == {"U", "R", "A"}
    assert dag.ancestral_set(set()) == set()


def test_ancestral_set_rejects_unknown_nodes_and_strings():
    dag = DAG(nodes="AB", edges=[("A", "B")])
    with pytest.raises(UnknownNodeError):
        dag.ancestral_set({"Z"})
    with pytest.raises(Exception, match="string"):
        dag.ancestral_set("AB")


@pytest.mark.parametrize("seed", range(40))
def test_proposition_5_pruning_removes_exactly_the_non_ancestors(seed):
    model = random_network(seed, n_vars=(1, 8), edge_prob=0.35)
    rng = np.random.default_rng(seed)
    dag = model.graph
    keep = {n for n in dag.nodes() if rng.random() < 0.3}
    assert repeatedly_delete_barren_leaves(dag, keep) == dag.ancestral_set(keep)


# ---------------------------------------------------------------------------
# VariableElimination.barren_variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "evidence", "barren"),
    [
        (["Accident"], {"Late": Y}, {"Umbrella"}),
        (["Accident"], {}, {"Rain", "Traffic", "Late", "Umbrella"}),
        (["Rain"], {"Umbrella": Y}, {"Accident", "Traffic", "Late"}),
        (["Late"], {}, {"Umbrella"}),
        (["Umbrella"], {"Late": Y}, set()),
    ],
)
def test_fixture_barren_variables(query, evidence, barren):
    assert VariableElimination(late_network()).barren_variables(query, evidence) == barren


# ---------------------------------------------------------------------------
# V7: pruning never changes an answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(80))
def test_pruned_and_unpruned_answers_agree(seed):
    model = random_network(seed, n_vars=(2, 7), edge_prob=0.4)
    rng = np.random.default_rng(seed + 70_000)
    names = [v.name for v in model.variables]
    query = [str(rng.choice(names))]
    evidence = random_evidence(model, rng, exclude=query)
    ve = VariableElimination(model)

    expected = posterior(model, query, evidence)
    p_e = evidence_probability(model, evidence)
    for heuristic in HEURISTICS:
        for prune in (True, False):
            result = ve.query(query, evidence, elimination_order=heuristic, prune_barren=prune)
            assert result.allclose(expected, atol=1e-12)
            assert ve.probability_of_evidence(
                evidence, elimination_order=heuristic, prune_barren=prune
            ) == pytest.approx(p_e, abs=1e-12)


def test_explicit_order_may_list_barren_variables():
    ve = VariableElimination(late_network())
    order = ["Umbrella", "Rain", "Traffic"]  # Umbrella is barren for this query
    expected = posterior(late_network(), ["Accident"], {"Late": Y})
    for prune in (True, False):
        result = ve.query(["Accident"], {"Late": Y}, elimination_order=order, prune_barren=prune)
        assert result.allclose(expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Cost: what query() actually does
# ---------------------------------------------------------------------------


def test_fixture_query_trace_skips_the_barren_umbrella():
    ve = VariableElimination(late_network())
    pruned = ve.query_trace(["Accident"], {"Late": Y})
    unpruned = ve.query_trace(["Accident"], {"Late": Y}, prune_barren=False)
    assert "Umbrella" not in pruned.order
    assert "Umbrella" in unpruned.order
    assert set(pruned.order) == {"Rain", "Traffic"}
    assert pruned.total_cost < unpruned.total_cost


def test_query_trace_follows_an_explicit_order():
    # Closes the M2.4 gap: query()'s use of an explicit order is now observable.
    ve = VariableElimination(late_network())
    order = ["Traffic", "Rain", "Umbrella"]
    trace = ve.query_trace(["Accident"], {"Late": Y}, elimination_order=order, prune_barren=False)
    assert trace.order == tuple(order)
    pruned = ve.query_trace(["Accident"], {"Late": Y}, elimination_order=order)
    assert pruned.order == ("Traffic", "Rain")


def test_root_query_needs_only_its_own_factor():
    ve = VariableElimination(late_network())
    assert ve.query_trace(["Accident"]).steps == ()
    assert ve.query(["Accident"]).value({"Accident": Y}) == pytest.approx(0.1)


@pytest.mark.parametrize("seed", range(60))
def test_pruning_never_enlarges_a_fixed_orders_steps(seed):
    """Fill-path lemma: with the same order, each pruned step's scope is a subset of the
    unpruned step's scope, and the pruned steps are the unpruned ones minus barren variables."""
    model = random_network(seed, n_vars=(3, 8), cards=(2, 3), edge_prob=0.4)
    rng = np.random.default_rng(seed + 80_000)
    names = [v.name for v in model.variables]
    query = [str(rng.choice(names))]
    evidence = {}
    if rng.random() < 0.5:
        other = [n for n in names if n not in query]
        v = model.variable(str(rng.choice(other)))
        evidence[v.name] = v.states[0]
    order = [
        str(n) for n in rng.permutation([n for n in names if n not in query and n not in evidence])
    ]

    ve = VariableElimination(model)
    barren = ve.barren_variables(query, evidence)
    full = ve.query_trace(query, evidence, elimination_order=order, prune_barren=False)
    pruned = ve.query_trace(query, evidence, elimination_order=order, prune_barren=True)

    assert pruned.order == tuple(n for n in order if n not in barren)
    by_variable = {step.variable: step for step in full.steps}
    for step in pruned.steps:
        assert step.scope <= by_variable[step.variable].scope
        assert step.size <= by_variable[step.variable].size
    assert pruned.total_cost <= full.total_cost
