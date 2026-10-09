import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probgraph import DiscreteVariable, TabularCPD
from probgraph.exceptions import UnknownStateError, ValidationError

# ---------------------------------------------------------------------------
# Fixture variables
# ---------------------------------------------------------------------------

RAIN = DiscreteVariable("Rain", ("no", "yes"))
ACCIDENT = DiscreteVariable("Accident", ("no", "yes"))
TRAFFIC = DiscreteVariable("Traffic", ("no", "yes"))

X = DiscreteVariable("X", ("x0", "x1"))
A = DiscreteVariable("A", ("a0", "a1", "a2"))
B = DiscreteVariable("B", ("b0", "b1"))


def traffic_given_rain() -> TabularCPD:
    return TabularCPD(TRAFFIC, (RAIN,), np.array([[0.8, 0.2], [0.2, 0.8]]))


def traffic_given_rain_accident() -> TabularCPD:
    # P(T=1 | R, A) from the canonical fixture; axes are (Traffic, Rain, Accident).
    t1 = np.array([[0.1, 0.7], [0.8, 0.95]])
    return TabularCPD(TRAFFIC, (RAIN, ACCIDENT), np.stack([1 - t1, t1]))


# ---------------------------------------------------------------------------
# Specification table (§3.3)
# ---------------------------------------------------------------------------


def test_valid_conditional_table_is_accepted():
    cpd = traffic_given_rain()
    assert cpd.variable == TRAFFIC
    assert cpd.parents == (RAIN,)
    assert cpd.values.shape == (2, 2)


def test_column_summing_to_point_nine_is_rejected():
    with pytest.raises(ValidationError, match="Rain=yes"):
        TabularCPD(TRAFFIC, (RAIN,), [[0.8, 0.1], [0.2, 0.8]])


def test_negative_probability_is_rejected():
    with pytest.raises(ValidationError, match="negative"):
        TabularCPD(TRAFFIC, (RAIN,), [[1.2, 0.2], [-0.2, 0.8]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_entries_are_rejected(bad):
    with pytest.raises(ValidationError, match="finite"):
        TabularCPD(TRAFFIC, (RAIN,), [[bad, 0.2], [0.2, 0.8]])


@pytest.mark.parametrize(
    "values",
    [
        [0.5, 0.5],  # missing parent axis
        [[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]],  # parent axis too long
        [[[0.5, 0.5]], [[0.5, 0.5]]],  # extra axis
        [[0.8, 0.2]],  # child axis too short
    ],
)
def test_wrong_tensor_dimensions_are_rejected(values):
    with pytest.raises(ValidationError, match="shape"):
        TabularCPD(TRAFFIC, (RAIN,), values)


def test_transposed_table_is_rejected_when_shape_differs():
    # P(X | A) must have shape (2, 3), not (3, 2).
    good = np.full((2, 3), 0.5)
    TabularCPD(X, (A,), good)
    with pytest.raises(ValidationError, match="shape"):
        TabularCPD(X, (A,), good.T)


def test_correct_conditional_lookup():
    cpd = traffic_given_rain()
    assert cpd.probability("yes", {"Rain": "no"}) == pytest.approx(0.2)
    assert cpd.probability("yes", {"Rain": "yes"}) == pytest.approx(0.8)
    assert cpd.probability("no", {"Rain": "yes"}) == pytest.approx(0.2)
    assert isinstance(cpd.probability("yes", {"Rain": "no"}), float)


def test_parent_assignment_order_does_not_matter():
    cpd = traffic_given_rain_accident()
    forward = cpd.probability("yes", {"Rain": "yes", "Accident": "no"})
    backward = cpd.probability("yes", {"Accident": "no", "Rain": "yes"})
    assert forward == backward == pytest.approx(0.8)


def test_fixture_lookups():
    cpd = traffic_given_rain_accident()
    expected = {("no", "no"): 0.1, ("no", "yes"): 0.7, ("yes", "no"): 0.8, ("yes", "yes"): 0.95}
    for (r, a), p in expected.items():
        given_ = {"Rain": r, "Accident": a}
        assert cpd.probability("yes", given_) == pytest.approx(p)
        assert cpd.probability("no", given_) == pytest.approx(1 - p)


def test_missing_parent_assignment_is_rejected():
    cpd = traffic_given_rain_accident()
    with pytest.raises(ValidationError, match=r"missing.*Accident"):
        cpd.probability("yes", {"Rain": "yes"})


def test_unknown_parent_in_assignment_is_rejected():
    cpd = traffic_given_rain()
    with pytest.raises(ValidationError, match=r"unexpected.*Weekend"):
        cpd.probability("yes", {"Rain": "yes", "Weekend": "no"})


def test_unknown_states_are_rejected():
    cpd = traffic_given_rain()
    with pytest.raises(UnknownStateError):
        cpd.probability("jammed", {"Rain": "yes"})
    with pytest.raises(UnknownStateError):
        cpd.probability("yes", {"Rain": "drizzle"})


def test_parentless_variable_uses_1d_array():
    cpd = TabularCPD(RAIN, (), np.array([0.7, 0.3]))
    assert cpd.parents == ()
    assert cpd.probability("yes", {}) == pytest.approx(0.3)
    np.testing.assert_allclose(cpd.distribution({}), [0.7, 0.3])


def test_parentless_variable_rejects_2d_array():
    with pytest.raises(ValidationError, match="shape"):
        TabularCPD(RAIN, (), [[0.7], [0.3]])


def test_two_parents_with_different_cardinalities_index_correctly():
    # values[x, a, b] encodes P(X=x | A=a, B=b). Each (a, b) fibre is distinct,
    # so an axis mix-up selects the wrong number.
    x1 = np.array(
        [
            # b0    b1
            [0.10, 0.20],  # a0
            [0.30, 0.40],  # a1
            [0.50, 0.60],  # a2
        ]
    )
    cpd = TabularCPD(X, (A, B), np.stack([1 - x1, x1]))
    assert cpd.values.shape == (2, 3, 2)
    for (ia, a), (ib, b) in itertools.product(enumerate(A.states), enumerate(B.states)):
        assert cpd.probability("x1", {"A": a, "B": b}) == pytest.approx(x1[ia, ib])
    assert cpd.probability("x1", {"B": "b0", "A": "a2"}) == pytest.approx(0.5)
    assert cpd.probability("x0", {"A": "a1", "B": "b1"}) == pytest.approx(0.6)


def test_distribution_returns_the_fibre():
    cpd = traffic_given_rain_accident()
    np.testing.assert_allclose(cpd.distribution({"Rain": "no", "Accident": "yes"}), [0.3, 0.7])


# ---------------------------------------------------------------------------
# Numerical policy
# ---------------------------------------------------------------------------


def test_normalisation_within_tolerance_is_accepted():
    TabularCPD(RAIN, (), [0.7, 0.3 + 1e-11])


def test_normalisation_outside_tolerance_is_rejected():
    with pytest.raises(ValidationError, match="sum"):
        TabularCPD(RAIN, (), [0.7, 0.3 + 1e-9])


def test_decimal_rounding_is_accepted():
    TabularCPD(DiscreteVariable("W", ("a", "b", "c")), (), [0.1, 0.2, 0.7])


def test_table_is_not_silently_renormalised():
    with pytest.raises(ValidationError):
        TabularCPD(RAIN, (), [7, 3])


def test_whole_table_summing_to_one_is_not_enough():
    # Sums to 1 overall, but not per column: that is a joint, not a conditional.
    with pytest.raises(ValidationError):
        TabularCPD(TRAFFIC, (RAIN,), [[0.4, 0.1], [0.1, 0.4]])


def test_zero_probabilities_and_integer_tables_are_accepted():
    cpd = TabularCPD(TRAFFIC, (RAIN,), [[1, 0], [0, 1]])
    assert cpd.values.dtype == np.float64
    assert cpd.probability("yes", {"Rain": "no"}) == 0.0


def test_singleton_child_domain():
    certain = DiscreteVariable("Sun", ("rises",))
    cpd = TabularCPD(certain, (RAIN,), [[1.0, 1.0]])
    assert cpd.probability("rises", {"Rain": "yes"}) == 1.0


@pytest.mark.parametrize(
    "values",
    [
        [[True, False], [False, True]],  # boolean
        [["0.8", "0.2"], ["0.2", "0.8"]],  # strings
        [[0.8 + 0j, 0.2], [0.2, 0.8]],  # complex
        [[0.8, 0.2], [0.2]],  # ragged
    ],
)
def test_non_real_or_ragged_input_is_rejected(values):
    with pytest.raises(ValidationError):
        TabularCPD(TRAFFIC, (RAIN,), values)


# ---------------------------------------------------------------------------
# Structural validation of the variable and parent declarations
# ---------------------------------------------------------------------------


def test_duplicate_parent_names_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        TabularCPD(X, (B, B), np.full((2, 2, 2), 0.5))


def test_variable_cannot_be_its_own_parent():
    with pytest.raises(ValidationError, match="own parent"):
        TabularCPD(X, (X,), np.eye(2))


def test_parents_must_be_a_sequence_of_variables():
    with pytest.raises(ValidationError):
        TabularCPD(TRAFFIC, RAIN, [[0.8, 0.2], [0.2, 0.8]])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TabularCPD(TRAFFIC, ("Rain",), [[0.8, 0.2], [0.2, 0.8]])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TabularCPD("Traffic", (RAIN,), [[0.8, 0.2], [0.2, 0.8]])  # type: ignore[arg-type]


def test_parent_list_is_frozen_into_tuple():
    cpd = TabularCPD(TRAFFIC, [RAIN], [[0.8, 0.2], [0.2, 0.8]])
    assert cpd.parents == (RAIN,)
    assert cpd.parent_names == ("Rain",)


# ---------------------------------------------------------------------------
# Immutability (spec §8: defensive copy, read-only storage)
# ---------------------------------------------------------------------------


def test_input_array_is_defensively_copied():
    source = np.array([[0.8, 0.2], [0.2, 0.8]])
    cpd = TabularCPD(TRAFFIC, (RAIN,), source)
    source[0, 0] = 0.0
    assert cpd.probability("no", {"Rain": "no"}) == pytest.approx(0.8)


def test_exposed_arrays_are_read_only():
    cpd = traffic_given_rain()
    with pytest.raises(ValueError):
        cpd.values[0, 0] = 0.0
    with pytest.raises(ValueError):
        cpd.distribution({"Rain": "no"})[0] = 0.0
    with pytest.raises(ValueError):
        cpd.values.setflags(write=True)
    assert cpd.probability("no", {"Rain": "no"}) == pytest.approx(0.8)


def test_attributes_cannot_be_reassigned():
    cpd = traffic_given_rain()
    with pytest.raises(AttributeError):
        cpd.variable = RAIN  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cpd.values = np.eye(2)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property-based tests: random cardinalities and random valid tables
# ---------------------------------------------------------------------------


@st.composite
def random_cpds(draw):
    child_card = draw(st.integers(1, 4))
    parent_cards = draw(st.lists(st.integers(1, 4), max_size=3))
    child = DiscreteVariable("C", tuple(f"c{i}" for i in range(child_card)))
    parents = tuple(
        DiscreteVariable(f"P{j}", tuple(f"p{j}_{i}" for i in range(card)))
        for j, card in enumerate(parent_cards)
    )
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    # Dirichlet(1) draws are uniform on the simplex; columns are stacked along axis 0.
    n_configs = int(np.prod(parent_cards, dtype=int))
    columns = rng.dirichlet(np.ones(child_card), size=n_configs).T
    values = columns.reshape((child_card, *parent_cards))
    return child, parents, values


@settings(max_examples=200, deadline=None)
@given(random_cpds())
def test_random_valid_tables_round_trip_through_named_lookup(case):
    child, parents, values = case
    cpd = TabularCPD(child, parents, values)

    for config in itertools.product(*(range(p.cardinality) for p in parents)):
        assignment = {p.name: p.states[i] for p, i in zip(parents, config, strict=True)}
        dist = cpd.distribution(assignment)
        # C1, C2 on every fibre.
        assert (dist >= 0).all()
        assert dist.sum() == pytest.approx(1.0, abs=1e-10)
        for ix, state in enumerate(child.states):
            assert cpd.probability(state, assignment) == values[(ix, *config)]
        # Name-based lookup ignores mapping order.
        reversed_assignment = dict(reversed(list(assignment.items())))
        np.testing.assert_array_equal(cpd.distribution(reversed_assignment), dist)


@settings(max_examples=200, deadline=None)
@given(random_cpds(), st.data())
def test_perturbing_any_entry_breaks_normalisation(case, data):
    child, parents, values = case
    index = tuple(data.draw(st.integers(0, n - 1)) for n in values.shape)
    perturbed = values.copy()
    perturbed[index] += 1e-6
    with pytest.raises(ValidationError):
        TabularCPD(child, parents, perturbed)
