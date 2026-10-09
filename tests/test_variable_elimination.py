"""M2.4: variable elimination with an explicit order (spec task 4; invariants V1–V6).

Every numerical check compares against the enumeration oracle in ``support``,
which shares no code with the factor algebra.
"""

import itertools

import numpy as np
import pytest
from support import (
    evidence_probability,
    late_network,
    posterior,
    rain_network_structure,
    random_evidence,
    random_network,
)

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.exceptions import (
    NormalisationError,
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)

Y, N = "yes", "no"


@pytest.fixture(scope="module")
def ve() -> VariableElimination:
    return VariableElimination(late_network())


# ---------------------------------------------------------------------------
# Fixture (docs/specs/milestone-2.md §4), exact values
# ---------------------------------------------------------------------------

FIXTURE_POSTERIORS = [
    ("Late", Y, {}, 1113 / 4000),
    ("Accident", Y, {}, 1 / 10),
    ("Accident", Y, {"Late": Y}, 65 / 371),
    ("Accident", Y, {"Late": Y, "Umbrella": Y}, 475 / 3787),
    ("Accident", Y, {"Late": Y, "Umbrella": N}, 275 / 1211),
    ("Accident", Y, {"Umbrella": Y}, 1 / 10),
    ("Rain", Y, {"Umbrella": Y}, 51 / 65),
    ("Accident", Y, {"Traffic": Y}, 5 / 23),
    ("Accident", Y, {"Traffic": Y, "Umbrella": Y}, 1165 / 8761),
]


@pytest.mark.parametrize(("var", "state", "evidence", "expected"), FIXTURE_POSTERIORS)
def test_fixture_posteriors(ve, var, state, evidence, expected):
    result = ve.query([var], evidence)
    assert result.value({var: state}) == pytest.approx(expected, abs=1e-12)
    assert result.total() == pytest.approx(1.0, abs=1e-12)


def test_fixture_evidence_probability(ve):
    assert ve.probability_of_evidence({"Late": Y, "Umbrella": Y}) == pytest.approx(
        11361 / 80000, abs=1e-12
    )
    assert ve.probability_of_evidence({"Late": Y}) == pytest.approx(1113 / 4000, abs=1e-12)


def test_every_order_gives_the_same_fixture_answer(ve):
    # Query A given L=1: eliminate R, T, U, in all 3! orders (V2).
    expected = posterior(late_network(), ["Accident"], {"Late": Y})
    for order in itertools.permutations(["Rain", "Traffic", "Umbrella"]):
        result = ve.query(["Accident"], {"Late": Y}, elimination_order=order)
        assert result.allclose(expected, atol=1e-12), order


# ---------------------------------------------------------------------------
# V3: result scope and order
# ---------------------------------------------------------------------------


def test_joint_query_has_requested_scope_and_order(ve):
    result = ve.query(["Umbrella", "Accident"], {"Late": Y})
    assert result.names == ("Umbrella", "Accident")
    assert result.allclose(posterior(late_network(), ["Umbrella", "Accident"], {"Late": Y}))


def test_query_of_every_unobserved_variable_needs_no_elimination(ve):
    query = ["Rain", "Accident", "Traffic", "Umbrella"]
    result = ve.query(query, {"Late": Y}, elimination_order=[])
    assert result.names == tuple(query)
    assert result.allclose(posterior(late_network(), query, {"Late": Y}))


# ---------------------------------------------------------------------------
# V4: validation of queries, evidence and orders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "evidence", "error", "match"),
    [
        (["Snow"], {}, UnknownNodeError, "Snow"),
        ([], {}, ValidationError, "at least one"),
        (["Rain", "Rain"], {}, ValidationError, "duplicate"),
        ("Rain", {}, ValidationError, "string"),
        (["Rain"], {"Rain": Y}, ValidationError, "both"),
        (["Rain"], {"Snow": Y}, UnknownNodeError, "Snow"),
        (["Rain"], {"Late": "very"}, UnknownStateError, "very"),
    ],
)
def test_invalid_queries_are_rejected(ve, query, evidence, error, match):
    with pytest.raises(error, match=match):
        ve.query(query, evidence)


@pytest.mark.parametrize(
    ("order", "match"),
    [
        (["Rain", "Traffic"], r"missing.*Umbrella"),  # incomplete
        (["Rain", "Traffic", "Umbrella", "Snow"], r"unexpected.*Snow"),  # unknown variable
        (["Rain", "Traffic", "Umbrella", "Accident"], r"unexpected.*Accident"),  # query variable
        (["Rain", "Traffic", "Umbrella", "Late"], r"unexpected.*Late"),  # evidence variable
        (["Rain", "Traffic", "Umbrella", "Rain"], "duplicate"),
        ("RTU", "string"),
    ],
)
def test_invalid_elimination_orders_are_rejected(ve, order, match):
    with pytest.raises(ValidationError, match=match):
        ve.query(["Accident"], {"Late": Y}, elimination_order=order)


def test_probability_of_evidence_validates_its_order(ve):
    with pytest.raises(ValidationError, match=r"missing"):
        ve.probability_of_evidence({"Late": Y}, elimination_order=["Rain"])


def test_incomplete_model_is_rejected_at_construction():
    with pytest.raises(ValidationError, match="missing CPDs"):
        VariableElimination(rain_network_structure())


# ---------------------------------------------------------------------------
# V5: zero-probability evidence
# ---------------------------------------------------------------------------


def switch_model() -> BayesianNetwork:
    switch = DiscreteVariable("Switch", ("off", "on"))
    light = DiscreteVariable("Light", ("off", "on"))
    model = BayesianNetwork([switch, light], [("Switch", "Light")])
    model.add_cpd(TabularCPD(switch, (), [1.0, 0.0]))  # the switch is always off
    model.add_cpd(TabularCPD(light, (switch,), np.eye(2)))  # the light mirrors the switch
    return model


def test_zero_probability_evidence_raises():
    ve = VariableElimination(switch_model())
    with pytest.raises(ZeroProbabilityEvidenceError, match="P\\(e\\) = 0"):
        ve.query(["Switch"], {"Light": "on"})
    # It is also a NormalisationError, so callers can catch the general case.
    with pytest.raises(NormalisationError):
        ve.query(["Switch"], {"Light": "on"})


def test_zero_probability_evidence_has_probability_zero():
    assert VariableElimination(switch_model()).probability_of_evidence({"Light": "on"}) == 0.0


def test_possible_evidence_in_a_deterministic_model():
    ve = VariableElimination(switch_model())
    assert ve.query(["Switch"], {"Light": "off"}).value({"Switch": "off"}) == 1.0


# ---------------------------------------------------------------------------
# V6: no evidence
# ---------------------------------------------------------------------------


def test_no_evidence_gives_the_prior_marginal(ve):
    assert ve.query(["Late"]).allclose(posterior(late_network(), ["Late"], {}))
    assert ve.query(["Late"], {}).allclose(ve.query(["Late"]))


def test_probability_of_no_evidence_is_one(ve):
    assert ve.probability_of_evidence({}) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Snapshot semantics (as for AncestralSampler)
# ---------------------------------------------------------------------------


def test_engine_snapshots_the_model():
    model = late_network()
    engine = VariableElimination(model)
    before = engine.query(["Rain"])
    model.add_cpd(TabularCPD(model.variable("Rain"), (), [0.0, 1.0]))
    assert engine.query(["Rain"]).allclose(before)
    assert VariableElimination(model).query(["Rain"]).value({"Rain": Y}) == 1.0


def test_single_variable_model():
    x = DiscreteVariable("X", ("a", "b", "c"))
    model = BayesianNetwork([x], [])
    model.add_cpd(TabularCPD(x, (), [0.2, 0.3, 0.5]))
    ve = VariableElimination(model)
    assert ve.query(["X"]).values.tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert ve.probability_of_evidence({"X": "b"}) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# V1, V2: random networks × queries × evidence × orders against enumeration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(100))
def test_matches_enumeration_on_random_problems(seed):
    model = random_network(seed, n_vars=(2, 6))
    rng = np.random.default_rng(seed + 20_000)
    names = [v.name for v in model.variables]
    size = int(rng.integers(1, min(3, len(names)) + 1))
    query = [str(n) for n in rng.choice(names, size=size, replace=False)]
    evidence = random_evidence(model, rng, exclude=query)
    eliminable = [n for n in names if n not in query and n not in evidence]

    ve = VariableElimination(model)
    p_e = evidence_probability(model, evidence)
    orders = [None] + [list(rng.permutation(eliminable)) for _ in range(3)]
    for order in orders:
        # P(e) also eliminates the query variables.
        evidence_order = None if order is None else [*order, *query]
        assert ve.probability_of_evidence(evidence, evidence_order) == pytest.approx(p_e, abs=1e-12)
        if p_e == 0.0:
            with pytest.raises(ZeroProbabilityEvidenceError):
                ve.query(query, evidence, elimination_order=order)
            continue
        result = ve.query(query, evidence, elimination_order=order)
        assert result.names == tuple(query)
        assert result.allclose(posterior(model, query, evidence), atol=1e-12)
