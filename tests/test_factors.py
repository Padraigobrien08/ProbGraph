"""M2.1: DiscreteFactor. Invariants F1–F10 from docs/specs/milestone-2.md §3.1.

The *definitional oracles* below compute each operation one cell at a time with
``value()`` and explicit loops, following docs/mathematics/factor_algebra.md §1.
They share no code path with the broadcasting implementation.
"""

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support import RAIN, TRAFFIC, traffic_cpd

from probgraph import DiscreteFactor, DiscreteVariable
from probgraph.exceptions import NormalisationError, UnknownStateError, ValidationError

A = DiscreteVariable("A", ("a0", "a1"))
B = DiscreteVariable("B", ("b0", "b1", "b2"))
C = DiscreteVariable("C", ("c0", "c1"))

# ---------------------------------------------------------------------------
# Definitional oracles
# ---------------------------------------------------------------------------


def assignments(variables: Sequence[DiscreteVariable]):
    for states in itertools.product(*(v.states for v in variables)):
        yield dict(zip((v.name for v in variables), states, strict=True))


def union(*factors: DiscreteFactor) -> list[DiscreteVariable]:
    seen: dict[str, DiscreteVariable] = {}
    for f in factors:
        for v in f.variables:
            seen.setdefault(v.name, v)
    return list(seen.values())


def restrict(x: Mapping[str, str], f: DiscreteFactor) -> dict[str, str]:
    return {name: x[name] for name in f.scope}


def assert_product_matches_definition(phi, psi, result):
    assert result.scope == phi.scope | psi.scope
    for x in assignments(union(phi, psi)):
        expected = phi.value(restrict(x, phi)) * psi.value(restrict(x, psi))
        assert result.value(x) == pytest.approx(expected, abs=1e-15)


def assert_marginal_matches_definition(phi, names, result):
    keep = [v for v in phi.variables if v.name not in names]
    summed = [v for v in phi.variables if v.name in names]
    assert result.scope == frozenset(v.name for v in keep)
    for x in assignments(keep):
        expected = sum(phi.value({**x, **y}) for y in assignments(summed))
        assert result.value(x) == pytest.approx(expected, abs=1e-12)


def assert_reduction_matches_definition(phi, evidence, result):
    relevant = {k: v for k, v in evidence.items() if k in phi.scope}
    keep = [v for v in phi.variables if v.name not in relevant]
    assert result.scope == frozenset(v.name for v in keep)
    for x in assignments(keep):
        assert result.value(x) == phi.value({**x, **relevant})


# ---------------------------------------------------------------------------
# Construction (F1, F2)
# ---------------------------------------------------------------------------


def test_valid_factor_need_not_be_normalised():
    f = DiscreteFactor([A, C], [[3.0, 0.0], [5.0, 2.5]])
    assert f.names == ("A", "C")
    assert f.scope == frozenset({"A", "C"})
    assert f.values.shape == (2, 2)
    assert f.value({"A": "a1", "C": "c0"}) == 5.0
    assert f.total() == pytest.approx(10.5)


def test_scalar_factor():
    s = DiscreteFactor([], 2.5)
    assert s.scope == frozenset()
    assert s.values.shape == ()
    assert s.value({}) == 2.5
    assert s.total() == 2.5
    assert DiscreteFactor.unit().value({}) == 1.0


@pytest.mark.parametrize(
    ("variables", "values", "match"),
    [
        ([A, C], [[1.0, 2.0]], "shape"),
        ([A, C], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "shape"),
        ([A], [[1.0], [2.0]], "shape"),
        ([], [1.0], "shape"),
        ([A, A], np.ones((2, 2)), "duplicate"),
        ([A], [1.0, -0.5], "negative"),
        ([A], [1.0, np.nan], "finite"),
        ([A], [np.inf, 1.0], "finite"),
        ([A], [True, False], "real"),
        ([A], ["1", "2"], "real"),
        ([A], [[1.0, 2.0], [3.0]], "regular"),
    ],
)
def test_invalid_factors_are_rejected(variables, values, match):
    with pytest.raises(ValidationError, match=match):
        DiscreteFactor(variables, values)


def test_variables_must_be_discrete_variables():
    with pytest.raises(ValidationError):
        DiscreteFactor(["A"], [1.0, 2.0])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        DiscreteFactor(A, [1.0, 2.0])  # type: ignore[arg-type]


def test_storage_is_copied_and_read_only():
    source = np.array([1.0, 2.0])
    f = DiscreteFactor([A], source)
    source[0] = 99.0
    assert f.value({"A": "a0"}) == 1.0
    with pytest.raises(ValueError):
        f.values[0] = 0.0
    with pytest.raises(ValueError):
        f.values.setflags(write=True)
    with pytest.raises(AttributeError):
        f.variables = (C,)  # type: ignore[misc]


def test_value_requires_exact_scope_and_valid_states():
    f = DiscreteFactor([A, C], np.ones((2, 2)))
    with pytest.raises(ValidationError, match=r"missing.*C"):
        f.value({"A": "a0"})
    with pytest.raises(ValidationError, match=r"unexpected.*B"):
        f.value({"A": "a0", "C": "c0", "B": "b0"})
    with pytest.raises(UnknownStateError):
        f.value({"A": "a9", "C": "c0"})


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


def test_product_hand_example():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    psi = DiscreteFactor([B, C], [[1, 10], [2, 20], [3, 30]])
    result = phi * psi
    assert result.scope == {"A", "B", "C"}
    # φ(a1, b2) ψ(b2, c1) = 6 · 30
    assert result.value({"A": "a1", "B": "b2", "C": "c1"}) == 180.0
    assert result.value({"A": "a0", "B": "b1", "C": "c0"}) == 4.0
    assert_product_matches_definition(phi, psi, result)


def test_product_aligns_by_name_not_position():
    # ψ stores the same variables in the opposite axis order. Aligning by position would
    # pair A with C and fail the definitional check.
    phi = DiscreteFactor([A, C], [[1.0, 2.0], [3.0, 4.0]])
    psi = DiscreteFactor([C, A], [[10.0, 20.0], [30.0, 40.0]])
    result = phi * psi
    assert result.value({"A": "a0", "C": "c1"}) == 2.0 * 30.0
    assert result.value({"A": "a1", "C": "c0"}) == 3.0 * 20.0
    assert_product_matches_definition(phi, psi, result)


def test_disjoint_product_is_outer_product():
    phi = DiscreteFactor([A], [2.0, 3.0])
    psi = DiscreteFactor([C], [5.0, 7.0])
    assert (phi * psi).aligned(["A", "C"]).values.tolist() == [[10.0, 14.0], [15.0, 21.0]]


def test_scalar_scales_a_factor():
    phi = DiscreteFactor([A], [2.0, 3.0])
    assert (DiscreteFactor([], 0.5) * phi).allclose(DiscreteFactor([A], [1.0, 1.5]))
    assert (phi * DiscreteFactor.unit()).allclose(phi)


def test_product_with_conflicting_domain_is_rejected():
    other_a = DiscreteVariable("A", ("a0", "a1", "a2"))
    with pytest.raises(ValidationError, match="domain"):
        DiscreteFactor([A], [1.0, 1.0]) * DiscreteFactor([other_a], [1.0, 1.0, 1.0])
    reordered = DiscreteVariable("A", ("a1", "a0"))
    with pytest.raises(ValidationError, match="domain"):
        DiscreteFactor([A], [1.0, 2.0]) * DiscreteFactor([reordered], [1.0, 2.0])


def test_product_with_non_factor_is_not_supported():
    with pytest.raises(TypeError):
        DiscreteFactor([A], [1.0, 2.0]) * 2.0  # type: ignore[operator]


def test_overflow_is_detected():
    big = DiscreteFactor([A], [1e200, 1.0])
    with pytest.raises(ValidationError, match="finite"):
        big * DiscreteFactor([C], [1e200, 1.0])


# ---------------------------------------------------------------------------
# Marginalise
# ---------------------------------------------------------------------------


def test_marginalise_hand_example():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    over_b = phi.marginalise(["B"])
    assert over_b.names == ("A",)
    assert over_b.values.tolist() == [6.0, 15.0]
    assert phi.marginalise(["A"]).values.tolist() == [5.0, 7.0, 9.0]
    assert_marginal_matches_definition(phi, {"B"}, over_b)


def test_marginalising_everything_gives_the_total():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    scalar = phi.marginalise(["A", "B"])
    assert scalar.scope == frozenset()
    assert scalar.value({}) == 21.0 == phi.total()


def test_marginalise_nothing_is_identity():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    assert phi.marginalise([]).allclose(phi)


def test_marginalise_unknown_variable_is_rejected():
    with pytest.raises(ValidationError, match="not in scope"):
        DiscreteFactor([A], [1.0, 2.0]).marginalise(["C"])


def test_marginalise_rejects_bare_string():
    # "AB" would otherwise be read as the two names "A" and "B".
    phi = DiscreteFactor([A, B], np.ones((2, 3)))
    with pytest.raises(ValidationError):
        phi.marginalise("AB")  # type: ignore[arg-type]


def test_distributivity_requires_its_side_condition():
    # F7 counterexample: X is in the scope of φ, so the sum cannot move past it.
    x = DiscreteVariable("X", ("0", "1"))
    phi = DiscreteFactor([x], [0.0, 1.0])
    lhs = (phi * phi).marginalise(["X"])
    rhs = phi * phi.marginalise(["X"])
    assert lhs.scope == frozenset() and lhs.value({}) == 1.0
    assert rhs.scope == {"X"}


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


def test_reduce_hand_example():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    reduced = phi.reduce({"B": "b2"})
    assert reduced.names == ("A",)
    assert reduced.values.tolist() == [3.0, 6.0]
    assert_reduction_matches_definition(phi, {"B": "b2"}, reduced)


def test_reduce_ignores_out_of_scope_evidence():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    reduced = phi.reduce({"B": "b0", "C": "c1", "Z": "anything"})
    assert reduced.names == ("A",)
    assert reduced.values.tolist() == [1.0, 4.0]
    assert phi.reduce({"C": "c1"}).allclose(phi)


def test_reduce_everything_gives_a_scalar():
    phi = DiscreteFactor([A, B], [[1, 2, 3], [4, 5, 6]])
    assert phi.reduce({"A": "a1", "B": "b1"}).value({}) == 5.0


def test_reduce_rejects_invalid_state_in_scope():
    with pytest.raises(UnknownStateError):
        DiscreteFactor([A], [1.0, 2.0]).reduce({"A": "a9"})


# ---------------------------------------------------------------------------
# Normalise (F9)
# ---------------------------------------------------------------------------


def test_normalise():
    phi = DiscreteFactor([A, C], [[1.0, 3.0], [0.0, 4.0]])
    p = phi.normalise()
    assert p.total() == pytest.approx(1.0)
    assert p.value({"A": "a0", "C": "c1"}) == pytest.approx(3.0 / 8.0)


def test_normalising_zero_factor_raises():
    with pytest.raises(NormalisationError):
        DiscreteFactor([A], [0.0, 0.0]).normalise()
    with pytest.raises(NormalisationError):
        DiscreteFactor([], 0.0).normalise()


def test_normalised_scalar_is_one():
    assert DiscreteFactor([], 7.0).normalise().value({}) == 1.0


# ---------------------------------------------------------------------------
# Axis order (F4): aligned and allclose
# ---------------------------------------------------------------------------


def test_aligned_permutes_axes_without_changing_the_function():
    phi = DiscreteFactor([A, B, C], np.arange(12, dtype=float).reshape(2, 3, 2))
    psi = phi.aligned(["C", "A", "B"])
    assert psi.names == ("C", "A", "B")
    assert psi.values.shape == (2, 2, 3)
    for x in assignments([A, B, C]):
        assert psi.value(x) == phi.value(x)
    assert psi.allclose(phi) and phi.allclose(psi)


def test_aligned_requires_a_permutation_of_the_scope():
    phi = DiscreteFactor([A, C], np.ones((2, 2)))
    # ["A", "C", "A"] has the right *set* of names but repeats one; it must not reach NumPy.
    for bad in (["A"], ["A", "C", "B"], ["A", "A"], ["A", "C", "A"]):
        with pytest.raises(ValidationError):
            phi.aligned(bad)


def test_allclose_compares_functions():
    phi = DiscreteFactor([A, C], [[1.0, 2.0], [3.0, 4.0]])
    assert phi.allclose(DiscreteFactor([C, A], [[1.0, 3.0], [2.0, 4.0]]))
    assert not phi.allclose(DiscreteFactor([C, A], [[1.0, 2.0], [3.0, 4.0]]))
    assert not phi.allclose(DiscreteFactor([A], [1.0, 2.0]))  # different scope
    assert not phi.allclose(phi.marginalise(["C"]))


# ---------------------------------------------------------------------------
# CPDs as factors (F10)
# ---------------------------------------------------------------------------


def test_from_cpd_preserves_values_by_name():
    cpd = traffic_cpd()
    phi = DiscreteFactor.from_cpd(cpd)
    assert phi.names == ("Traffic", "Rain", "Accident")
    for x in assignments([TRAFFIC, RAIN, cpd.parents[1]]):
        given = {"Rain": x["Rain"], "Accident": x["Accident"]}
        assert phi.value(x) == cpd.probability(x["Traffic"], given)


def test_cpd_factor_sums_to_one_over_child():
    phi = DiscreteFactor.from_cpd(traffic_cpd())
    ones = phi.marginalise(["Traffic"])
    assert ones.scope == {"Rain", "Accident"}
    np.testing.assert_allclose(ones.values, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Property-based tests on random factors
# ---------------------------------------------------------------------------

POOL = [
    DiscreteVariable("V0", ("s0", "s1")),
    DiscreteVariable("V1", ("s0", "s1", "s2")),
    DiscreteVariable("V2", ("s0",)),
    DiscreteVariable("V3", ("s0", "s1")),
    DiscreteVariable("V4", ("s0", "s1", "s2")),
]


@st.composite
def factors(draw, pool=tuple(POOL)):
    scope = draw(st.lists(st.sampled_from(pool), unique_by=lambda v: v.name, max_size=4))
    scope = draw(st.permutations(scope))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    values = rng.random(tuple(v.cardinality for v in scope))
    if draw(st.booleans()):
        values[rng.random(values.shape) < 0.3] = 0.0  # include exact zeros
    return DiscreteFactor(scope, values)


@st.composite
def evidence_for(draw, pool=tuple(POOL)):
    chosen = draw(st.lists(st.sampled_from(pool), unique_by=lambda v: v.name, max_size=3))
    return {v.name: draw(st.sampled_from(v.states)) for v in chosen}


SETTINGS = settings(max_examples=150, deadline=None)


@SETTINGS
@given(factors(), factors())
def test_product_matches_definition(phi, psi):
    assert_product_matches_definition(phi, psi, phi * psi)


@SETTINGS
@given(factors(), st.data())
def test_marginal_matches_definition(phi, data):
    names = data.draw(st.sets(st.sampled_from(sorted(phi.scope)))) if phi.scope else set()
    assert_marginal_matches_definition(phi, names, phi.marginalise(names))


@SETTINGS
@given(factors(), evidence_for())
def test_reduction_matches_definition(phi, evidence):
    assert_reduction_matches_definition(phi, evidence, phi.reduce(evidence))


@SETTINGS
@given(factors(), st.data())
def test_f4_axis_order_is_invisible(phi, data):
    order = data.draw(st.permutations(list(phi.names)))
    psi = phi.aligned(order)
    for x in assignments(phi.variables):
        assert psi.value(x) == phi.value(x)
    assert psi.allclose(phi)


@SETTINGS
@given(factors(), factors(), factors())
def test_f5_commutative_monoid(phi, psi, chi):
    assert (phi * psi).allclose(psi * phi)
    assert ((phi * psi) * chi).allclose(phi * (psi * chi))
    assert (DiscreteFactor.unit() * phi).allclose(phi)


@SETTINGS
@given(factors(), st.data())
def test_f6_sums_commute(phi, data):
    if len(phi.scope) < 2:
        return
    x, y = data.draw(st.permutations(sorted(phi.scope)))[:2]
    one_then_other = phi.marginalise([x]).marginalise([y])
    other_then_one = phi.marginalise([y]).marginalise([x])
    together = phi.marginalise([x, y])
    assert one_then_other.allclose(other_then_one)
    assert one_then_other.allclose(together)


@SETTINGS
@given(factors(), factors(), st.data())
def test_f7_distributivity(phi, psi, data):
    candidates = sorted(psi.scope - phi.scope)
    if not candidates:
        return
    x = data.draw(st.sampled_from(candidates))
    assert (phi * psi).marginalise([x]).allclose(phi * psi.marginalise([x]))


@SETTINGS
@given(factors(), factors(), evidence_for())
def test_f8_reduction_commutes_with_product(phi, psi, evidence):
    assert (phi * psi).reduce(evidence).allclose(phi.reduce(evidence) * psi.reduce(evidence))


@SETTINGS
@given(factors(), evidence_for(), st.data())
def test_f8_reduction_commutes_with_marginalisation(phi, evidence, data):
    free = sorted(phi.scope - set(evidence))
    names = data.draw(st.sets(st.sampled_from(free))) if free else set()
    lhs = phi.marginalise(names).reduce(evidence)
    rhs = phi.reduce(evidence).marginalise(names)
    assert lhs.allclose(rhs)


@SETTINGS
@given(factors(), evidence_for())
def test_reduction_is_product_with_indicator_then_sum(phi, evidence):
    """φ[e] = Σ_{E∩S} φ · δ_e (factor_algebra.md §3)."""
    by_name = {v.name: v for v in phi.variables}
    observed = [by_name[n] for n in evidence if n in phi.scope]
    indicator = DiscreteFactor.unit()
    for v in observed:
        delta = np.zeros(v.cardinality)
        delta[v.index_of(evidence[v.name])] = 1.0
        indicator = indicator * DiscreteFactor([v], delta)
    via_indicator = (phi * indicator).marginalise([v.name for v in observed])
    assert phi.reduce(evidence).allclose(via_indicator)


@SETTINGS
@given(factors())
def test_f9_normalise(phi):
    if phi.total() == 0.0:
        with pytest.raises(NormalisationError):
            phi.normalise()
    else:
        p = phi.normalise()
        assert p.total() == pytest.approx(1.0, abs=1e-12)
        assert p.scope == phi.scope
