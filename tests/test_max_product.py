"""M5.4: the most probable explanation by max-product VE (V1, V2, V5; max_product.md Part 1)."""

import itertools
import math

import numpy as np
import pytest
from support import (
    joint_table,
    late_network,
    random_evidence,
    random_network,
    underflow_network,
)

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import UnknownNodeError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import HEURISTICS


def brute_force_mpe(model: BayesianNetwork, evidence):
    """(sorted list of (P(x, e), assignment)) over every completion of the evidence."""
    table = joint_table(model)
    names = [v.name for v in model.variables]
    results = []
    for index in itertools.product(*(range(v.cardinality) for v in model.variables)):
        assignment = {v.name: v.states[i] for v, i in zip(model.variables, index, strict=True)}
        if all(assignment[k] == s for k, s in evidence.items()):
            free = {n: assignment[n] for n in names if n not in evidence}
            results.append((float(table[index]), free))
    results.sort(key=lambda pair: -pair[0])
    return results


# ---------------------------------------------------------------------------
# Fixture F5: the late network
# ---------------------------------------------------------------------------


def test_f5_without_evidence_nothing_happens():
    assignment, log_p = VariableElimination(late_network()).most_probable_explanation()
    assert assignment == {n: "no" for n in ("Rain", "Accident", "Traffic", "Late", "Umbrella")}
    assert math.exp(log_p) == pytest.approx(0.45927, abs=1e-14)


def test_f5_given_late():
    assignment, log_p = VariableElimination(late_network()).most_probable_explanation(
        {"Late": "yes"}
    )
    assert assignment == {"Rain": "yes", "Accident": "no", "Traffic": "yes", "Umbrella": "yes"}
    assert math.exp(log_p) == pytest.approx(0.3 * 0.9 * 0.8 * 0.6 * 0.85, abs=1e-15)  # 0.11016
    runner_up = brute_force_mpe(late_network(), {"Late": "yes"})[1][0]
    assert runner_up == pytest.approx(0.05103, abs=1e-15)


# ---------------------------------------------------------------------------
# V2: brute force on random networks; V1: every order gives the same value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(80))
def test_mpe_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    model = random_network(seed, n_vars=(1, 6), cards=(1, 3), edge_prob=0.5)
    evidence = random_evidence(model, rng)
    ranked = brute_force_mpe(model, evidence)
    best = ranked[0][0]
    ve = VariableElimination(model)
    if best == 0.0:
        with pytest.raises(ZeroProbabilityEvidenceError):
            ve.most_probable_explanation(evidence)
        return
    assignment, log_p = ve.most_probable_explanation(evidence)
    assert log_p == pytest.approx(math.log(best), rel=1e-12, abs=1e-12)
    assert set(assignment) == {v.name for v in model.variables} - set(evidence)
    attained = model.joint_probability({**assignment, **evidence})
    assert attained == pytest.approx(best, rel=1e-12)  # the assignment attains the maximum
    if len(ranked) == 1 or ranked[1][0] < best * (1 - 1e-9):
        assert assignment == ranked[0][1]  # unique maximiser: it is returned


@pytest.mark.parametrize("seed", range(30))
def test_every_elimination_order_gives_the_same_value(seed):
    rng = np.random.default_rng(seed + 500)
    model = random_network(seed + 500, n_vars=(2, 6), cards=(2, 3), edge_prob=0.5)
    evidence = random_evidence(model, rng)
    ve = VariableElimination(model)
    free = [v.name for v in model.variables if v.name not in evidence]
    values = [ve.most_probable_explanation(evidence, heuristic)[1] for heuristic in HEURISTICS]
    for _ in range(4):
        order = list(rng.permutation(free))
        values.append(ve.most_probable_explanation(evidence, order)[1])
    assert values == pytest.approx([values[0]] * len(values), rel=1e-12, abs=1e-12)


def test_all_variables_observed():
    model = late_network()
    evidence = {"Rain": "yes", "Accident": "no", "Traffic": "yes", "Late": "yes", "Umbrella": "no"}
    assignment, log_p = VariableElimination(model).most_probable_explanation(evidence)
    assert assignment == {}
    assert log_p == pytest.approx(math.log(model.joint_probability(evidence)), abs=1e-15)


# ---------------------------------------------------------------------------
# §5–6: barren leaves matter, and the MPE is not the marginal argmax
# ---------------------------------------------------------------------------


def barren_example() -> BayesianNetwork:
    x = DiscreteVariable("X", ("0", "1"))
    y = DiscreteVariable("Y", ("0", "1", "2"))
    model = BayesianNetwork([x, y], [("X", "Y")])
    model.add_cpd(TabularCPD(x, (), [0.4, 0.6]))
    model.add_cpd(TabularCPD(y, (x,), [[1.0, 1 / 3], [0.0, 1 / 3], [0.0, 1 / 3]]))
    return model


def test_a_barren_leaf_changes_the_mpe():
    ve = VariableElimination(barren_example())
    assignment, log_p = ve.most_probable_explanation()
    assert assignment == {"X": "0", "Y": "0"}
    assert math.exp(log_p) == pytest.approx(0.4, abs=1e-15)
    # Pruning Y would maximise P(X) alone, and pick X = 1: max_y P(y | x) depends on x.
    assert ve.query(["X"]).value({"X": "1"}) == pytest.approx(0.6)


def test_the_mpe_is_not_the_individually_most_probable_value():
    ve = VariableElimination(barren_example())
    marginal = ve.query(["X"]).values
    assert int(np.argmax(marginal)) == 1  # the most probable X on its own
    assert ve.most_probable_explanation()[0]["X"] == "0"  # but not in the best joint assignment


# ---------------------------------------------------------------------------
# V5: ties are broken deterministically
# ---------------------------------------------------------------------------


def test_ties_choose_the_first_states():
    a, b, c = (DiscreteVariable(n, ("p", "q", "r")) for n in "ABC")
    model = BayesianNetwork([a, b, c], [("A", "B"), ("B", "C")])
    model.add_cpd(TabularCPD(a, (), np.full(3, 1 / 3)))
    model.add_cpd(TabularCPD(b, (a,), np.full((3, 3), 1 / 3)))
    model.add_cpd(TabularCPD(c, (b,), np.full((3, 3), 1 / 3)))
    ve = VariableElimination(model)
    for order in itertools.permutations("ABC"):
        assignment, log_p = ve.most_probable_explanation(elimination_order=list(order))
        assert assignment == {"A": "p", "B": "p", "C": "p"}
        assert log_p == pytest.approx(3 * math.log(1 / 3))


def test_the_answer_is_reproducible():
    model = random_network(3, n_vars=(5, 5), cards=(2, 3))
    ve = VariableElimination(model)
    assert ve.most_probable_explanation() == ve.most_probable_explanation()


# ---------------------------------------------------------------------------
# Log space: far below float64's range
# ---------------------------------------------------------------------------


def test_underflow_network():
    """1,100 observed features and 100 unobserved ones: P(x*, e) is below float64's range."""
    model = underflow_network(1200)
    evidence = {f"F{i}": ("0" if i < 650 else "1") for i in range(1100)}
    assignment, log_p = VariableElimination(model).most_probable_explanation(evidence)
    # C = 0 explains the 650 zeros; each unobserved feature then takes its likelier state, "0".
    assert assignment["C"] == "0"
    assert all(assignment[f"F{i}"] == "0" for i in range(1100, 1200))
    expected = math.log(0.5) + 650 * math.log(0.6) + 450 * math.log(0.4) + 100 * math.log(0.6)
    assert log_p == pytest.approx(expected, rel=1e-12)
    assert log_p < -745  # P(x*, e) itself is not representable as a float


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_impossible_evidence_raises():
    with pytest.raises(ZeroProbabilityEvidenceError):
        VariableElimination(barren_example()).most_probable_explanation({"X": "0", "Y": "2"})


def test_validation():
    ve = VariableElimination(late_network())
    with pytest.raises(UnknownNodeError):
        ve.most_probable_explanation({"Snow": "yes"})
    with pytest.raises(ValidationError, match="Elimination order"):
        ve.most_probable_explanation({"Late": "yes"}, ["Rain", "Accident"])
    with pytest.raises(ValidationError, match="heuristic"):
        ve.most_probable_explanation({}, "max_fill")
