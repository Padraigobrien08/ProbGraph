import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support import (
    ACCIDENT,
    RAIN,
    TRAFFIC,
    accident_cpd,
    joint_table,
    marginal,
    probability_of,
    rain_cpd,
    rain_network,
    rain_network_structure,
)

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import (
    CycleError,
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_structure_is_exposed_read_only():
    model = rain_network_structure()
    assert [v.name for v in model.variables] == ["Rain", "Accident", "Traffic"]
    assert model.variable("Traffic") == TRAFFIC
    assert model.parents("Traffic") == {"Rain", "Accident"}
    assert model.edges() == (("Rain", "Traffic"), ("Accident", "Traffic"))

    copy = model.graph
    copy.add_edge("Rain", "Accident")  # mutating the copy...
    assert model.parents("Accident") == set()  # ...does not touch the model


def test_duplicate_variable_names_are_rejected():
    clash = DiscreteVariable("Rain", ("dry", "wet"))
    with pytest.raises(ValidationError, match="duplicate"):
        BayesianNetwork([RAIN, clash], [])


def test_non_variable_is_rejected():
    with pytest.raises(ValidationError):
        BayesianNetwork(["Rain"], [])  # type: ignore[list-item]


def test_edge_to_undeclared_variable_is_rejected():
    with pytest.raises(UnknownNodeError):
        BayesianNetwork([RAIN], [("Rain", "Traffic")])


def test_cyclic_structure_is_rejected():
    with pytest.raises(CycleError):
        BayesianNetwork([RAIN, TRAFFIC], [("Rain", "Traffic"), ("Traffic", "Rain")])


def test_unknown_variable_lookup():
    with pytest.raises(UnknownNodeError):
        rain_network_structure().variable("Snow")


# ---------------------------------------------------------------------------
# Structural validation (B1, B2, B6)
# ---------------------------------------------------------------------------


def test_complete_valid_model_passes_validation():
    rain_network().validate()


def test_missing_cpd_raises():
    model = rain_network_structure()
    model.add_cpd(rain_cpd())
    with pytest.raises(ValidationError, match=r"missing CPDs.*Accident.*Traffic"):
        model.validate()


def test_query_on_incomplete_model_raises():
    model = rain_network_structure()
    model.add_cpd(rain_cpd())
    with pytest.raises(ValidationError, match="missing CPDs"):
        model.joint_probability({"Rain": "no", "Accident": "no", "Traffic": "no"})


def test_cpd_with_too_few_parents_is_rejected():
    model = rain_network_structure()
    bad = TabularCPD(TRAFFIC, (RAIN,), [[0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValidationError, match="parents"):
        model.add_cpd(bad)


def test_cpd_with_extra_parent_is_rejected():
    model = rain_network_structure()
    # Accident has no parents in the graph, but this CPD conditions on Rain.
    bad = TabularCPD(ACCIDENT, (RAIN,), [[0.9, 0.8], [0.1, 0.2]])
    with pytest.raises(ValidationError, match="parents"):
        model.add_cpd(bad)


def test_cpd_conditioning_on_a_child_is_rejected():
    # Rain -> Traffic in the graph; p(Rain | Traffic) reverses the edge (the P2 counterexample).
    model = rain_network_structure()
    bad = TabularCPD(RAIN, (TRAFFIC,), [[0.9, 0.4], [0.1, 0.6]])
    with pytest.raises(ValidationError, match="parents"):
        model.add_cpd(bad)


def test_cpd_for_unknown_node_is_rejected():
    model = rain_network_structure()
    snow = DiscreteVariable("Snow", ("no", "yes"))
    with pytest.raises(ValidationError, match="Snow"):
        model.add_cpd(TabularCPD(snow, (), [0.5, 0.5]))


def test_cpd_with_conflicting_variable_domain_is_rejected():
    model = rain_network_structure()
    wet = DiscreteVariable("Rain", ("dry", "wet"))
    with pytest.raises(ValidationError, match="domain"):
        model.add_cpd(TabularCPD(wet, (), [0.7, 0.3]))


def test_cpd_with_reordered_variable_domain_is_rejected():
    # Same labels in a different order would silently swap probabilities.
    model = rain_network_structure()
    flipped = DiscreteVariable("Rain", ("yes", "no"))
    with pytest.raises(ValidationError, match="domain"):
        model.add_cpd(TabularCPD(flipped, (), [0.3, 0.7]))


def test_cpd_with_conflicting_parent_domain_is_rejected():
    model = rain_network_structure()
    rain3 = DiscreteVariable("Rain", ("no", "light", "heavy"))
    values = np.full((2, 3, 2), 0.5)
    with pytest.raises(ValidationError, match="domain"):
        model.add_cpd(TabularCPD(TRAFFIC, (rain3, ACCIDENT), values))


def test_cpd_parent_order_may_differ_from_graph():
    # B2 is set equality: the CPD keeps its own axis order.
    t1 = np.array([[0.1, 0.7], [0.8, 0.95]])  # [rain, accident]
    swapped = TabularCPD(TRAFFIC, (ACCIDENT, RAIN), np.stack([1 - t1, t1]).transpose(0, 2, 1))
    model = rain_network_structure()
    model.add_cpd(rain_cpd())
    model.add_cpd(accident_cpd())
    model.add_cpd(swapped)
    assert model.joint_probability({"Rain": "yes", "Accident": "no", "Traffic": "yes"}) == (
        pytest.approx(0.216)
    )


def test_rejected_cpd_leaves_model_unchanged():
    model = rain_network()
    with pytest.raises(ValidationError):
        model.add_cpd(TabularCPD(TRAFFIC, (RAIN,), [[0.8, 0.2], [0.2, 0.8]]))
    assert model.cpds["Traffic"].parent_names == ("Rain", "Accident")
    model.validate()


def test_replacing_a_cpd_changes_subsequent_queries():
    model = rain_network()
    query = {"Rain": "yes", "Accident": "no", "Traffic": "yes"}
    assert model.joint_probability(query) == pytest.approx(0.216)

    model.add_cpd(TabularCPD(RAIN, (), [0.5, 0.5]))
    assert model.joint_probability(query) == pytest.approx(0.5 * 0.9 * 0.8)


def test_cpd_registry_is_read_only():
    model = rain_network()
    with pytest.raises(TypeError):
        model.cpds["Rain"] = rain_cpd()  # type: ignore[index]


# ---------------------------------------------------------------------------
# Joint probability: canonical fixture (spec §4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rain", "accident", "traffic", "expected"),
    [
        ("yes", "no", "yes", 0.216),
        ("yes", "yes", "yes", 0.0285),
        ("no", "no", "no", 0.567),
        ("no", "yes", "no", 0.021),
    ],
)
def test_joint_probability_matches_hand_calculation(rain, accident, traffic, expected):
    model = rain_network()
    result = model.joint_probability({"Rain": rain, "Accident": accident, "Traffic": traffic})
    assert result == pytest.approx(expected, abs=1e-12)
    assert isinstance(result, float)


def test_joint_assignment_order_does_not_matter():
    model = rain_network()
    a = model.joint_probability({"Rain": "yes", "Accident": "no", "Traffic": "yes"})
    b = model.joint_probability({"Traffic": "yes", "Accident": "no", "Rain": "yes"})
    assert a == b


def test_fixture_joint_sums_to_one():
    table = joint_table(rain_network())
    assert table.shape == (2, 2, 2)
    assert table.sum() == pytest.approx(1.0, abs=1e-12)
    assert (table >= 0).all() and (table <= 1).all()


def test_traffic_marginal():
    # Spec v0.3 lists 0.3615; the correct value is 0.063 + 0.049 + 0.216 + 0.0285 = 0.3565.
    model = rain_network()
    assert probability_of(model, lambda x: x["Traffic"] == "yes") == pytest.approx(
        0.3565, abs=1e-12
    )


def test_collider_independence_and_explaining_away():
    """The worked numbers in docs/mathematics/factorisation.md §2."""
    model = rain_network()
    accident = lambda x: x["Accident"] == "yes"  # noqa: E731
    p = lambda given: probability_of(model, accident, given)  # noqa: E731

    assert p(lambda x: True) == pytest.approx(0.1)
    # Marginal independence: A ⊥ R.
    assert p(lambda x: x["Rain"] == "yes") == pytest.approx(0.1)
    assert p(lambda x: x["Rain"] == "no") == pytest.approx(0.1)
    # Conditional dependence given the collider.
    assert p(lambda x: x["Traffic"] == "yes") == pytest.approx(0.0775 / 0.3565)
    assert p(lambda x: x["Traffic"] == "yes" and x["Rain"] == "yes") == pytest.approx(0.095 / 0.815)
    assert p(lambda x: x["Traffic"] == "yes" and x["Rain"] == "no") == pytest.approx(0.4375)


def test_eliminating_leaf_recovers_parent_network():
    # Corollary C-a: summing out Traffic leaves P(R) P(A).
    table = joint_table(rain_network())
    np.testing.assert_allclose(marginal(table, [0, 1]), np.outer([0.7, 0.3], [0.9, 0.1]))


def test_zero_probability_event_returns_zero():
    on_off = DiscreteVariable("Switch", ("off", "on"))
    light = DiscreteVariable("Light", ("off", "on"))
    model = BayesianNetwork([on_off, light], [("Switch", "Light")])
    model.add_cpd(TabularCPD(on_off, (), [0.5, 0.5]))
    model.add_cpd(TabularCPD(light, (on_off,), np.eye(2)))  # light mirrors switch exactly
    assert model.joint_probability({"Switch": "off", "Light": "on"}) == 0.0
    assert model.joint_probability({"Switch": "on", "Light": "on"}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Complete queries (B7)
# ---------------------------------------------------------------------------


def test_missing_variable_in_query_is_rejected():
    with pytest.raises(ValidationError, match=r"missing.*Traffic"):
        rain_network().joint_probability({"Rain": "yes", "Accident": "no"})


def test_extra_variable_in_query_is_rejected():
    with pytest.raises(ValidationError, match=r"unexpected.*Snow"):
        rain_network().joint_probability(
            {"Rain": "yes", "Accident": "no", "Traffic": "yes", "Snow": "no"}
        )


def test_unknown_state_in_query_is_rejected():
    with pytest.raises(UnknownStateError):
        rain_network().joint_probability({"Rain": "drizzle", "Accident": "no", "Traffic": "yes"})


# ---------------------------------------------------------------------------
# Property-based tests on random networks (B3–B5, P2 corollaries C-b, C-c)
# ---------------------------------------------------------------------------


@st.composite
def random_networks(draw):
    n = draw(st.integers(1, 5))
    cards = draw(st.lists(st.integers(1, 3), min_size=n, max_size=n))
    variables = [
        DiscreteVariable(f"V{i}", tuple(f"s{j}" for j in range(c))) for i, c in enumerate(cards)
    ]
    # Pick a hidden ordering, then only allow edges that go forward in it, so the graph is acyclic.
    hidden = draw(st.permutations(range(n)))
    edges = [
        (f"V{hidden[i]}", f"V{hidden[j]}")
        for i in range(n)
        for j in range(i + 1, n)
        if draw(st.booleans())
    ]
    model = BayesianNetwork(variables, edges)

    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    by_name = {v.name: v for v in variables}
    for v in variables:
        # Shuffle the parent axis order, so the CPD's axes differ from graph insertion order.
        parents = tuple(by_name[p] for p in draw(st.permutations(sorted(model.parents(v.name)))))
        parent_cards = [p.cardinality for p in parents]
        n_configs = int(np.prod(parent_cards, dtype=int))
        columns = rng.dirichlet(np.ones(v.cardinality), size=n_configs).T
        model.add_cpd(TabularCPD(v, parents, columns.reshape((v.cardinality, *parent_cards))))
    return model


def _axis(model: BayesianNetwork, name: str) -> int:
    return [v.name for v in model.variables].index(name)


@settings(max_examples=150, deadline=None)
@given(random_networks())
def test_random_networks_define_valid_joint_distributions(model):
    table = joint_table(model)
    # B5 and B4.
    assert (table >= 0).all()
    assert table.sum() == pytest.approx(1.0, abs=1e-10)


@settings(max_examples=150, deadline=None)
@given(random_networks())
def test_cpds_are_the_conditionals_of_the_joint(model):
    """Corollary C-b: P(x_i | pa_i), recomputed from the joint, equals the input CPD."""
    table = joint_table(model)
    for name, cpd in model.cpds.items():
        child = _axis(model, name)
        parent_axes = [_axis(model, p) for p in cpd.parent_names]
        keep = sorted([child, *parent_axes])
        m = marginal(table, keep)
        # Reorder the axes to (child, *CPD parent order) to match cpd.values.
        m = np.moveaxis(m, [keep.index(a) for a in [child, *parent_axes]], range(len(keep)))
        conditional = m / m.sum(axis=0, keepdims=True)  # Dirichlet draws make p(pa) > 0
        np.testing.assert_allclose(conditional, cpd.values, atol=1e-9)


@settings(max_examples=150, deadline=None)
@given(random_networks())
def test_local_markov_property_holds(model):
    """Corollary C-c: x_i ⊥ nd(x_i) \\ pa(x_i) | pa(x_i), checked as a tensor identity.

    With S = nd \\ pa, independence given pa means
        P(x, pa, s) · P(pa) = P(x, pa) · P(pa, s)   for all x, pa, s.
    """
    table = joint_table(model)
    graph = model.graph
    for name in graph.nodes():
        x = [_axis(model, name)]
        pa = [_axis(model, p) for p in graph.parents(name)]
        excluded = {name} | graph.descendants(name) | graph.parents(name)
        s = [_axis(model, v) for v in graph.nodes() if v not in excluded]
        if not s:
            continue

        def m(axes):
            # Marginal over `axes`, keeping size-1 axes elsewhere so broadcasting lines up.
            return table.sum(
                axis=tuple(i for i in range(table.ndim) if i not in axes), keepdims=True
            )

        lhs = m(x + pa + s) * m(pa)
        rhs = m(x + pa) * m(pa + s)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)
