"""M3.1: LogFactor (spec §3.1; invariants L1–L4; docs/mathematics/log_space.md).

Two routes are checked:
  * through the isomorphism: every operation on random DiscreteFactors agrees with
    the same operation on their LogFactors, compared under log;
  * directly: the laws hold for log factors with magnitudes around ±500, where the
    probability-space counterpart cannot be represented at all.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support import POOL, evidence_for, factors, traffic_cpd

from probgraph import DiscreteFactor, DiscreteVariable
from probgraph.exceptions import NormalisationError, UnknownStateError, ValidationError
from probgraph.factors import LogFactor

A = DiscreteVariable("A", ("a0", "a1"))
B = DiscreteVariable("B", ("b0", "b1", "b2"))
C = DiscreteVariable("C", ("c0", "c1"))
NEG_INF = -np.inf

# ---------------------------------------------------------------------------
# Construction (L1)
# ---------------------------------------------------------------------------


def test_valid_log_factor_allows_minus_infinity():
    f = LogFactor([A, C], [[0.0, NEG_INF], [-2.5, 700.0]])
    assert f.names == ("A", "C")
    assert f.log_value({"A": "a0", "C": "c1"}) == NEG_INF
    assert f.log_value({"A": "a1", "C": "c1"}) == 700.0


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([0.0, np.nan], "NaN"),
        ([0.0, np.inf], r"\+inf"),
        ([[0.0], [1.0]], "shape"),
        ([True, False], "real"),
        ([[0.0, 1.0], [2.0]], "regular"),
    ],
)
def test_invalid_log_factors_are_rejected(values, match):
    with pytest.raises(ValidationError, match=match):
        LogFactor([A], values)


def test_storage_is_copied_and_read_only():
    source = np.array([0.0, -1.0])
    f = LogFactor([A], source)
    source[0] = 5.0
    assert f.log_value({"A": "a0"}) == 0.0
    with pytest.raises(ValueError):
        f.log_values[0] = 1.0


def test_unit_and_scalar():
    assert LogFactor.unit().log_value({}) == 0.0
    assert LogFactor([], -3.0).log_total() == -3.0


def test_value_lookup_validation():
    f = LogFactor([A], [0.0, -1.0])
    with pytest.raises(ValidationError, match="missing"):
        f.log_value({})
    with pytest.raises(UnknownStateError):
        f.log_value({"A": "a9"})


# ---------------------------------------------------------------------------
# Conversions (L2)
# ---------------------------------------------------------------------------


def test_round_trip_with_zeros():
    phi = DiscreteFactor([A, C], [[0.25, 0.0], [3.0, 1.0]])
    ell = LogFactor.from_factor(phi)
    assert ell.log_value({"A": "a0", "C": "c1"}) == NEG_INF
    assert ell.log_value({"A": "a1", "C": "c0"}) == pytest.approx(math.log(3.0))
    assert ell.to_factor().allclose(phi)


def test_from_cpd():
    cpd = traffic_cpd()
    assert LogFactor.from_cpd(cpd).to_factor().allclose(DiscreteFactor.from_cpd(cpd))


def test_to_factor_underflows_to_zero_as_documented():
    tiny = LogFactor([A], [-800.0, 0.0])
    assert tiny.to_factor().value({"A": "a0"}) == 0.0


def test_to_factor_rejects_overflow():
    with pytest.raises(ValidationError, match="finite"):
        LogFactor([A], [800.0, 0.0]).to_factor()


# ---------------------------------------------------------------------------
# Log-sum-exp (L4): Lemma 1 and the all -inf slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("log_values", "expected"),
    [
        ([1000.0, 1000.0], 1000.0 + math.log(2)),  # would overflow without the max-shift
        ([-1000.0, -1000.0], -1000.0 + math.log(2)),  # would underflow to log 0 without it
        ([0.0, NEG_INF], 0.0),
        ([NEG_INF, NEG_INF], NEG_INF),  # the all -inf slice: -inf, never NaN
        ([-745.0, 709.0], 709.0),
    ],
)
def test_log_sum_exp(log_values, expected):
    total = LogFactor([A], log_values).marginalise(["A"]).log_value({})
    if expected == NEG_INF:
        assert total == NEG_INF
    else:
        assert total == pytest.approx(expected, abs=1e-12)


def test_all_minus_infinity_slices_stay_minus_infinity():
    f = LogFactor([A, C], [[NEG_INF, NEG_INF], [0.0, NEG_INF]])
    m = f.marginalise(["A"])  # column c0: log(0 + 1) = 0; column c1: log(0 + 0) = -inf
    assert m.log_value({"C": "c0"}) == pytest.approx(0.0)
    assert m.log_value({"C": "c1"}) == NEG_INF
    assert not np.isnan(m.log_values).any()


def test_normalise_and_zero_total():
    f = LogFactor([A], [1000.0, 1000.0 + math.log(3)])
    p = f.normalise()
    assert p.to_factor().values.tolist() == pytest.approx([0.25, 0.75])
    assert p.log_total() == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(NormalisationError):
        LogFactor([A], [NEG_INF, NEG_INF]).normalise()


def test_product_overflow_is_detected():
    with pytest.raises(ValidationError, match="overflow"):
        LogFactor([A], [1e308, 0.0]) * LogFactor([C], [1e308, 0.0])


# ---------------------------------------------------------------------------
# Representing what probability space cannot (§1)
# ---------------------------------------------------------------------------


def test_underflowing_product_is_exact_in_log_space():
    """1,100 factors of 0.5: 2^-1100 underflows to 0.0 in float64."""
    half = DiscreteFactor([], 0.5)
    prob = half
    log = LogFactor.from_factor(half)
    for _ in range(1099):
        prob = prob * half
        log = log * LogFactor.from_factor(half)
    assert prob.total() == 0.0
    assert log.log_total() == pytest.approx(1100 * math.log(0.5), rel=1e-12)


def test_overflowing_partition_function_is_finite_in_log_space():
    """200 factors with entries up to 100: Z ≈ 10^400 cannot be a float64."""
    variables = [DiscreteVariable(f"X{i}", ("0", "1")) for i in range(200)]
    log = LogFactor.unit()
    for v in variables:
        log = (log * LogFactor([v], np.log([100.0, 1.0]))).marginalise([v.name])
    assert log.log_total() == pytest.approx(200 * math.log(101.0), rel=1e-12)
    with pytest.raises(ValidationError):
        log.to_factor()


# ---------------------------------------------------------------------------
# allclose semantics (§6)
# ---------------------------------------------------------------------------


def test_allclose_requires_exact_minus_infinity_positions():
    f = LogFactor([A], [0.0, NEG_INF])
    assert f.allclose(LogFactor([A], [1e-13, NEG_INF]))
    assert not f.allclose(LogFactor([A], [0.0, -800.0]))  # tiny is not impossible
    assert not f.allclose(LogFactor([C], [0.0, NEG_INF]))  # different scope


def test_aligned_preserves_the_function():
    f = LogFactor([A, B], np.log(np.arange(1.0, 7.0)).reshape(2, 3))
    g = f.aligned(["B", "A"])
    assert g.names == ("B", "A")
    assert g.allclose(f)


# ---------------------------------------------------------------------------
# L3: every operation commutes with the isomorphism
# ---------------------------------------------------------------------------

SETTINGS = settings(max_examples=150, deadline=None)


@SETTINGS
@given(factors(), factors())
def test_product_matches_probability_space(phi, psi):
    lhs = LogFactor.from_factor(phi) * LogFactor.from_factor(psi)
    assert lhs.allclose(LogFactor.from_factor(phi * psi), atol=1e-12)


@SETTINGS
@given(factors(), st.data())
def test_marginal_matches_probability_space(phi, data):
    names = data.draw(st.sets(st.sampled_from(sorted(phi.scope)))) if phi.scope else set()
    lhs = LogFactor.from_factor(phi).marginalise(names)
    assert lhs.allclose(LogFactor.from_factor(phi.marginalise(names)), atol=1e-12)


@SETTINGS
@given(factors(), evidence_for())
def test_reduce_matches_probability_space(phi, evidence):
    lhs = LogFactor.from_factor(phi).reduce(evidence)
    assert lhs.allclose(LogFactor.from_factor(phi.reduce(evidence)), atol=0.0)


@SETTINGS
@given(factors())
def test_normalise_and_total_match_probability_space(phi):
    ell = LogFactor.from_factor(phi)
    if phi.total() == 0.0:
        assert ell.log_total() == NEG_INF
        with pytest.raises(NormalisationError):
            ell.normalise()
    else:
        assert ell.log_total() == pytest.approx(math.log(phi.total()), abs=1e-12)
        assert ell.normalise().allclose(LogFactor.from_factor(phi.normalise()), atol=1e-12)


# ---------------------------------------------------------------------------
# The laws, directly, at magnitudes probability space cannot hold
# ---------------------------------------------------------------------------


@st.composite
def big_log_factors(draw, pool=tuple(POOL)):
    scope = draw(st.lists(st.sampled_from(pool), unique_by=lambda v: v.name, max_size=4))
    scope = draw(st.permutations(scope))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    values = rng.uniform(-500.0, 500.0, size=tuple(v.cardinality for v in scope))
    if draw(st.booleans()):
        values[rng.random(values.shape) < 0.3] = NEG_INF
    return LogFactor(scope, values)


@SETTINGS
@given(big_log_factors(), big_log_factors(), big_log_factors())
def test_laws_hold_directly_at_large_magnitudes(f, g, h):
    # Commutativity, associativity, identity.
    assert (f * g).allclose(g * f, atol=1e-9)
    assert ((f * g) * h).allclose(f * (g * h), atol=1e-9)
    assert (LogFactor.unit() * f).allclose(f, atol=0.0)
    # Distributivity: LSE_x (f + g) = f + LSE_x g when x is not in f's scope.
    for x in sorted(g.scope - f.scope):
        assert (f * g).marginalise([x]).allclose(f * g.marginalise([x]), atol=1e-9)
    # Log-sum-exps over different variables commute.
    if len(f.scope) >= 2:
        x, y = sorted(f.scope)[:2]
        assert f.marginalise([x]).marginalise([y]).allclose(f.marginalise([x, y]), atol=1e-9)


def test_mixing_spaces_is_a_type_error():
    """A log value times a probability is meaningless; conversions must be explicit."""
    phi = DiscreteFactor([A], [0.5, 0.5])
    with pytest.raises(TypeError):
        phi * LogFactor.from_factor(phi)  # type: ignore[operator]
    with pytest.raises(TypeError):
        LogFactor.from_factor(phi) * phi  # type: ignore[operator]
    assert not LogFactor.from_factor(phi).allclose(phi)  # type: ignore[arg-type]
