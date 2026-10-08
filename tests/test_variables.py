import dataclasses

import pytest

from probgraph import DiscreteVariable
from probgraph.exceptions import ProbGraphError, UnknownStateError, ValidationError


def test_two_distinct_states_have_cardinality_two():
    rain = DiscreteVariable("Rain", ("no", "yes"))
    assert rain.cardinality == 2


def test_singleton_domain_is_valid():
    assert DiscreteVariable("Constant", ("on",)).cardinality == 1


def test_index_of_follows_declared_order():
    weather = DiscreteVariable("Weather", ("sun", "cloud", "rain"))
    assert [weather.index_of(s) for s in weather.states] == [0, 1, 2]


@pytest.mark.parametrize(
    "states",
    [
        ("no", "no"),  # duplicate
        (),  # empty domain
        ("no", ""),  # empty label
        ("no", 1),  # non-string label
    ],
)
def test_invalid_domains_are_rejected(states):
    with pytest.raises(ValidationError):
        DiscreteVariable("X", states)


@pytest.mark.parametrize("name", ["", None, 3])
def test_invalid_names_are_rejected(name):
    with pytest.raises(ValidationError):
        DiscreteVariable(name, ("a", "b"))


def test_bare_string_is_not_silently_split_into_characters():
    with pytest.raises(ValidationError):
        DiscreteVariable("X", "ab")


@pytest.mark.parametrize("states", [{"a", "b"}, frozenset({"a", "b"})])
def test_unordered_domains_are_rejected(states):
    # Domain order defines CPD axis indexing, so it must be deterministic.
    with pytest.raises(ValidationError):
        DiscreteVariable("X", states)


def test_list_domain_is_frozen_into_tuple():
    source = ["no", "yes"]
    v = DiscreteVariable("Rain", source)
    source.append("maybe")
    assert v.states == ("no", "yes")
    assert isinstance(v.states, tuple)


def test_unknown_state_lookup_raises():
    rain = DiscreteVariable("Rain", ("no", "yes"))
    with pytest.raises(UnknownStateError):
        rain.index_of("maybe")
    with pytest.raises(LookupError):  # also catchable as a standard lookup failure
        rain.index_of("maybe")


def test_attributes_cannot_be_reassigned():
    rain = DiscreteVariable("Rain", ("no", "yes"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        rain.states = ("a", "b")  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rain.name = "Snow"  # type: ignore[misc]


def test_states_tuple_cannot_be_mutated():
    rain = DiscreteVariable("Rain", ("no", "yes"))
    with pytest.raises(TypeError):
        rain.states[0] = "maybe"  # type: ignore[index]


def test_value_equality_and_hashing():
    a = DiscreteVariable("Rain", ("no", "yes"))
    b = DiscreteVariable("Rain", ["no", "yes"])
    c = DiscreteVariable("Rain", ("yes", "no"))  # same labels, different axis order
    assert a == b and hash(a) == hash(b)
    assert a != c


def test_validation_errors_share_library_base_class():
    with pytest.raises(ProbGraphError):
        DiscreteVariable("X", ())
