"""M2.2: CPDs as factors, evidence, and the posterior oracle (spec tasks 2 and 3).

Two independent routes to every posterior are compared here:
  * the enumeration oracle in ``support`` (M1's joint_probability and numpy sums), and
  * brute force in the factor algebra: normalise( Σ_rest ( Π φ_i )[e] ).
Variable elimination (M2.4) will be checked against both.
"""

from functools import reduce
from operator import mul

import numpy as np
import pytest
from support import (
    ACCIDENT,
    RAIN,
    conditionally_independent,
    evidence_probability,
    joint_table,
    late_network,
    posterior,
    rain_network,
    rain_network_structure,
    random_evidence,
    random_network,
)

from probgraph import BayesianNetwork, DiscreteFactor, DiscreteVariable, TabularCPD
from probgraph.exceptions import (
    NormalisationError,
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
)


def product(factors) -> DiscreteFactor:
    return reduce(mul, factors, DiscreteFactor.unit())


def factor_posterior(model: BayesianNetwork, query, evidence) -> DiscreteFactor:
    """Brute force in the factor algebra (factor_algebra.md §4, second form)."""
    reduced = product(phi.reduce(evidence) for phi in model.factors())
    rest = [n for n in reduced.names if n not in query]
    return reduced.marginalise(rest).normalise().aligned(list(query))


# ---------------------------------------------------------------------------
# Task 2: the CPDs as factors reproduce the M1 joint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", [late_network(), *(random_network(s, n_vars=(3, 5)) for s in range(10))]
)
def test_factors_are_the_cpds_in_topological_order(model):
    # Random networks declare variables in an order unrelated to the graph's.
    factors = model.factors()
    assert isinstance(factors, tuple)
    heads = [phi.names[0] for phi in factors]
    assert heads == model.graph.topological_sort()
    for phi in factors:
        assert phi.allclose(DiscreteFactor.from_cpd(model.cpds[phi.names[0]]))


def test_factors_require_a_complete_model():
    model = rain_network_structure()
    with pytest.raises(ValidationError, match="missing CPDs"):
        model.factors()


@pytest.mark.parametrize("build", [rain_network, late_network])
def test_product_of_factors_is_the_joint_for_fixtures(build):
    model = build()
    joint = DiscreteFactor(model.variables, joint_table(model))
    assert product(model.factors()).allclose(joint)


@pytest.mark.parametrize("seed", range(40))
def test_product_of_factors_is_the_joint_for_random_networks(seed):
    model = random_network(seed)
    joint = DiscreteFactor(model.variables, joint_table(model))
    result = product(model.factors())
    assert result.allclose(joint)
    assert result.total() == pytest.approx(1.0, abs=1e-12)  # P2, through the factor algebra


# ---------------------------------------------------------------------------
# Task 3: evidence validation
# ---------------------------------------------------------------------------


def test_check_evidence_returns_a_plain_copy():
    model = late_network()
    evidence = {"Late": "yes", "Umbrella": "no"}
    checked = model.check_evidence(evidence)
    assert checked == evidence and checked is not evidence
    assert model.check_evidence({}) == {}


def test_check_evidence_rejects_unknown_variables():
    with pytest.raises(UnknownNodeError, match="Snow"):
        late_network().check_evidence({"Late": "yes", "Snow": "yes"})


def test_check_evidence_rejects_unknown_states():
    with pytest.raises(UnknownStateError):
        late_network().check_evidence({"Late": "very"})


def test_check_evidence_rejects_non_mappings():
    with pytest.raises(ValidationError):
        late_network().check_evidence([("Late", "yes")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Task 3: the fixture table in docs/specs/milestone-2.md §4
# ---------------------------------------------------------------------------

Y, N = "yes", "no"

# Exact values: every CPD entry is a short decimal, so each posterior is a rational number.
FIXTURE_POSTERIORS = [
    # (query variable, state, evidence, exact value)
    ("Late", Y, {}, 1113 / 4000),  # 0.27825
    ("Accident", Y, {}, 1 / 10),
    ("Accident", Y, {"Late": Y}, 65 / 371),  # 0.175202...
    ("Accident", Y, {"Late": Y, "Umbrella": Y}, 475 / 3787),  # 0.125429...
    ("Accident", Y, {"Late": Y, "Umbrella": N}, 275 / 1211),  # 0.227085...
    ("Accident", Y, {"Umbrella": Y}, 1 / 10),
    ("Rain", Y, {"Umbrella": Y}, 51 / 65),  # 0.784615...
    ("Accident", Y, {"Traffic": Y}, 5 / 23),  # 0.217391...
    ("Accident", Y, {"Traffic": Y, "Umbrella": Y}, 1165 / 8761),  # 0.132976...
]


@pytest.mark.parametrize(("var", "state", "evidence", "expected"), FIXTURE_POSTERIORS)
def test_fixture_posteriors_by_enumeration(var, state, evidence, expected):
    p = posterior(late_network(), [var], evidence)
    assert p.value({var: state}) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(("var", "state", "evidence", "expected"), FIXTURE_POSTERIORS)
def test_fixture_posteriors_by_factor_algebra(var, state, evidence, expected):
    p = factor_posterior(late_network(), [var], evidence)
    assert p.value({var: state}) == pytest.approx(expected, abs=1e-12)


def test_fixture_evidence_probability():
    model = late_network()
    evidence = {"Late": Y, "Umbrella": Y}
    assert evidence_probability(model, evidence) == pytest.approx(11361 / 80000, abs=1e-12)
    reduced = product(phi.reduce(evidence) for phi in model.factors())
    assert reduced.total() == pytest.approx(11361 / 80000, abs=1e-12)


def test_marginal_independence_is_exact_in_the_fixture():
    # A ⊥ U, so observing the umbrella leaves P(A) unchanged (up to rounding).
    model = late_network()
    assert posterior(model, ["Accident"], {"Umbrella": Y}).allclose(
        posterior(model, ["Accident"], {})
    )


INDEPENDENCE_FACTS = [
    # (X, Y, Z, holds?) from docs/specs/milestone-2.md §4
    (["Accident"], ["Rain"], [], True),
    (["Accident"], ["Rain"], ["Traffic"], False),
    (["Accident"], ["Rain"], ["Late"], False),
    (["Accident"], ["Umbrella"], ["Late"], False),
    (["Accident"], ["Late"], ["Traffic"], True),
    (["Umbrella"], ["Late"], ["Rain"], True),
    (["Umbrella"], ["Late"], ["Traffic"], True),
    (["Umbrella"], ["Accident"], ["Traffic"], False),
    (["Umbrella"], ["Accident"], ["Traffic", "Rain"], True),
]


@pytest.mark.parametrize(("xs", "ys", "zs", "holds"), INDEPENDENCE_FACTS)
def test_fixture_independence_facts_hold_numerically(xs, ys, zs, holds):
    assert conditionally_independent(late_network(), xs, ys, zs) is holds


def test_hand_derivation_of_first_elimination():
    """The derivation in milestone-2.md §4 and §10, step by step in the factor algebra.

    P(A, L=1) = P(A) Σ_r P(r) [Σ_u P(u|r)] Σ_t P(t|r,A) P(L=1|t)
    """
    model = late_network()
    phi = {f.names[0]: f for f in model.factors()}
    evidence = {"Late": Y}

    # The bracket: U is a barren node, so summing it out leaves the all-ones factor over R.
    ones = phi["Umbrella"].marginalise(["Umbrella"])
    assert ones.allclose(DiscreteFactor([RAIN], [1.0, 1.0]))

    # τ1(r, a) = Σ_t P(t | r, a) P(L=1 | t)
    tau1 = (phi["Traffic"] * phi["Late"].reduce(evidence)).marginalise(["Traffic"])
    assert tau1.allclose(DiscreteFactor([RAIN, ACCIDENT], [[0.15, 0.45], [0.5, 0.575]]))

    # τ2(a) = Σ_r P(r) τ1(r, a)
    tau2 = (phi["Rain"] * tau1).marginalise(["Rain"])
    unnormalised = phi["Accident"] * tau2  # P(A, L=1)

    assert unnormalised.total() == pytest.approx(1113 / 4000, abs=1e-12)  # P(L=1)
    assert unnormalised.normalise().value({"Accident": Y}) == pytest.approx(65 / 371, abs=1e-12)


# ---------------------------------------------------------------------------
# The two routes agree on random networks, queries and evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_factor_route_matches_enumeration_oracle(seed):
    model = random_network(seed, n_vars=(2, 5))
    rng = np.random.default_rng(seed + 10_000)
    names = [v.name for v in model.variables]
    size = int(rng.integers(1, min(3, len(names)) + 1))
    query = [str(n) for n in rng.choice(names, size=size, replace=False)]
    evidence = random_evidence(model, rng, exclude=query)

    expected = posterior(model, query, evidence)
    assert factor_posterior(model, query, evidence).allclose(expected, atol=1e-12)

    reduced = product(phi.reduce(evidence) for phi in model.factors())
    assert reduced.total() == pytest.approx(evidence_probability(model, evidence), abs=1e-12)


def test_impossible_evidence_has_zero_probability_in_both_routes():
    switch = DiscreteVariable("Switch", ("off", "on"))
    light = DiscreteVariable("Light", ("off", "on"))
    model = BayesianNetwork([switch, light], [("Switch", "Light")])
    model.add_cpd(TabularCPD(switch, (), [1.0, 0.0]))  # the switch is always off
    model.add_cpd(TabularCPD(light, (switch,), np.eye(2)))

    impossible = {"Light": "on"}
    assert evidence_probability(model, impossible) == 0.0
    with pytest.raises(ZeroDivisionError):
        posterior(model, ["Switch"], impossible)
    with pytest.raises(NormalisationError):
        factor_posterior(model, ["Switch"], impossible)
