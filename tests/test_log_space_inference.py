"""M3.2: variable elimination in log space (spec task 2; invariant L5; log_space.md §8)."""

import math

import numpy as np
import pytest
from support import (
    evidence_probability,
    late_network,
    posterior,
    random_evidence,
    random_network,
    underflow_evidence,
    underflow_network,
)

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import HEURISTICS

LOG_06, LOG_04 = math.log(0.6), math.log(0.4)

# ---------------------------------------------------------------------------
# F3: the underflow regression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def f3() -> VariableElimination:
    return VariableElimination(underflow_network(1100))


def test_probability_space_cannot_be_trusted_on_f3(f3):
    """Pins the M2 limitation that log space fixes (log_space.md §1).

    The true P(e) ≈ 10^-341 is below float64's range. Depending on the order of
    the products, probability space ends at exactly 0, or at a subnormal
    leftover such as 5e-324. In the first case it raises
    ZeroProbabilityEvidenceError for possible evidence. In the second it
    *silently returns a confident, wrong posterior*.
    """
    alternating = {f"F{i}": ("1" if i % 2 else "0") for i in range(1100)}
    assert f3.probability_of_evidence(alternating, space="probability") == 0.0
    with pytest.raises(ZeroProbabilityEvidenceError):
        f3.query(["C"], alternating, space="probability")

    blocked = underflow_evidence(550, 550)  # the same counts, listed in a different order
    assert (
        0.0 < f3.probability_of_evidence(blocked, space="probability") < np.finfo(float).tiny
    )  # subnormal garbage
    wrong = f3.query(["C"], blocked, space="probability")
    assert wrong.value({"C": "0"}) == pytest.approx(0.0, abs=1e-12)  # truth is 0.5

    for evidence in (alternating, blocked):
        assert f3.query(["C"], evidence, space="log").values.tolist() == pytest.approx([0.5, 0.5])


def test_log_space_is_the_default(f3):
    """Since v0.3.0 the default call is safe on F3 (M3 spec ⚑2, revised)."""
    blocked = underflow_evidence(550, 550)
    assert f3.query(["C"], blocked).values.tolist() == pytest.approx([0.5, 0.5])
    assert f3.query_trace(["C"], blocked) == f3.query_trace(["C"], blocked, space="probability")


def test_log_space_fixes_f3_balanced(f3):
    evidence = underflow_evidence(550, 550)
    result = f3.query(["C"], evidence, space="log")
    assert result.values.tolist() == pytest.approx([0.5, 0.5], abs=1e-12)
    # P(e | c) is the same for both classes, so log P(e) = 550 log 0.6 + 550 log 0.4.
    expected = 550 * LOG_06 + 550 * LOG_04
    assert expected == pytest.approx(-784.91, abs=0.01)
    assert f3.log_probability_of_evidence(evidence) == pytest.approx(expected, abs=1e-9)


def test_log_space_fixes_f3_unbalanced(f3):
    """551 zeros and 549 ones: the likelihood ratio is (0.6/0.4)^2 = 2.25 in favour of C=0."""
    evidence = underflow_evidence(551, 549)
    result = f3.query(["C"], evidence, space="log")
    assert result.value({"C": "0"}) == pytest.approx(2.25 / 3.25, abs=1e-12)
    log_p_e = math.log(0.5) + 549 * LOG_06 + 549 * LOG_04 + math.log(0.6**2 + 0.4**2)
    assert f3.log_probability_of_evidence(evidence) == pytest.approx(log_p_e, abs=1e-9)


def test_exponentiated_evidence_probability_underflows_without_raising(f3):
    assert f3.probability_of_evidence(underflow_evidence(550, 550), space="log") == 0.0


# ---------------------------------------------------------------------------
# Agreement with probability space wherever it does not underflow
# ---------------------------------------------------------------------------

Y = "yes"


def test_fixture_posteriors_agree_across_spaces():
    ve = VariableElimination(late_network())
    for evidence, expected in [
        ({"Late": Y}, 65 / 371),
        ({"Late": Y, "Umbrella": Y}, 475 / 3787),
        ({"Traffic": Y, "Umbrella": Y}, 1165 / 8761),
    ]:
        result = ve.query(["Accident"], evidence, space="log")
        assert result.value({"Accident": Y}) == pytest.approx(expected, abs=1e-12)
    assert ve.log_probability_of_evidence({"Late": Y, "Umbrella": Y}) == pytest.approx(
        math.log(11361 / 80000), abs=1e-12
    )
    assert ve.log_probability_of_evidence({}) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(60))
def test_log_space_matches_enumeration_on_random_problems(seed):
    model = random_network(seed, n_vars=(2, 6), edge_prob=0.4)
    rng = np.random.default_rng(seed + 140_000)
    names = [v.name for v in model.variables]
    query = [str(rng.choice(names))]
    evidence = random_evidence(model, rng, exclude=query)
    ve = VariableElimination(model)
    expected = posterior(model, query, evidence)
    p_e = evidence_probability(model, evidence)
    for heuristic in HEURISTICS:
        for prune in (True, False):
            result = ve.query(
                query, evidence, elimination_order=heuristic, prune_barren=prune, space="log"
            )
            assert result.allclose(expected, atol=1e-12)
            assert ve.log_probability_of_evidence(
                evidence, elimination_order=heuristic, prune_barren=prune
            ) == pytest.approx(math.log(p_e), abs=1e-10)
    # The order used, and therefore the cost, does not depend on the space.
    assert ve.query_trace(query, evidence) == ve.query_trace(query, evidence, space="log")


# ---------------------------------------------------------------------------
# Exact zero detection
# ---------------------------------------------------------------------------


def switch_model() -> BayesianNetwork:
    switch = DiscreteVariable("Switch", ("off", "on"))
    light = DiscreteVariable("Light", ("off", "on"))
    model = BayesianNetwork([switch, light], [("Switch", "Light")])
    model.add_cpd(TabularCPD(switch, (), [1.0, 0.0]))
    model.add_cpd(TabularCPD(light, (switch,), np.eye(2)))
    return model


def test_structurally_impossible_evidence_is_minus_infinity_and_raises():
    ve = VariableElimination(switch_model())
    assert ve.log_probability_of_evidence({"Light": "on"}) == -math.inf
    with pytest.raises(ZeroProbabilityEvidenceError):
        ve.query(["Switch"], {"Light": "on"}, space="log")
    assert ve.query(["Switch"], {"Light": "off"}, space="log").value({"Switch": "off"}) == 1.0


def test_tiny_but_possible_is_never_minus_infinity(f3):
    # Even 1,100 observations far from the model's typical data stay finite in log space.
    evidence = underflow_evidence(1100, 0)
    log_p_e = f3.log_probability_of_evidence(evidence)
    assert math.isfinite(log_p_e)
    # P(e) = 0.5 (0.6^1100 + 0.4^1100) = 0.5 · 0.6^1100 · (1 + (0.4/0.6)^1100).
    expected = math.log(0.5) + 1100 * LOG_06 + math.log1p((0.4 / 0.6) ** 1100)
    assert log_p_e == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["logarithm", "", None])
def test_invalid_space_is_rejected(bad):
    ve = VariableElimination(late_network())
    with pytest.raises(ValidationError, match="space"):
        ve.query(["Rain"], space=bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="space"):
        ve.probability_of_evidence({}, space=bad)  # type: ignore[arg-type]
