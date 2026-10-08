import itertools
from collections import Counter

import numpy as np
import pytest
from support import RAIN, joint_table, rain_cpd, rain_network, rain_network_structure

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import ValidationError
from probgraph.sampling import cumulative_table, inverse_cdf

N_LARGE = 200_000
Z = 5.0  # tolerance in standard errors; see docs/mathematics/ancestral_sampling.md §4


def assert_frequency(count: int, n: int, p: float, label: str = "") -> None:
    """Check that count/n is within Z·SE + Z²/(3n) of p (Bernstein bound, valid for any p).

    A correct sampler fails this with probability at most 2·exp(-Z²/2) ≈ 7.5e-6, with no
    normal approximation. When p is 0 or 1 the tolerance is exactly zero.
    """
    p_hat = count / n
    se = np.sqrt(p * (1 - p) / n)
    tol = Z * se + Z**2 / (3 * n) if 0 < p < 1 else 0.0
    assert abs(p_hat - p) <= tol, f"{label}: p̂={p_hat:.6f}, p={p:.6f}, tol={tol:.6f}"


# ---------------------------------------------------------------------------
# Lemma: inverse-CDF categorical draws
# ---------------------------------------------------------------------------


def test_inverse_cdf_partitions_unit_interval():
    cdf = cumulative_table(np.array([0.2, 0.5, 0.3]))
    np.testing.assert_allclose(cdf, [0.2, 0.7, 1.0])
    u = np.array([0.0, 0.1999, 0.2, 0.6999, 0.7, np.nextafter(1.0, 0.0)])
    # Half-open intervals [F_{k-1}, F_k): a boundary value belongs to the next state.
    np.testing.assert_array_equal(inverse_cdf(cdf, u), [0, 0, 1, 1, 2, 2])


def test_zero_mass_states_are_never_selected_at_any_position():
    cdf = cumulative_table(np.array([0.0, 0.5, 0.0, 0.5, 0.0]))
    u = np.array([0.0, 0.25, 0.5, 0.75, np.nextafter(1.0, 0.0)])
    np.testing.assert_array_equal(inverse_cdf(cdf, u), [1, 1, 3, 3, 3])


def test_cumulative_table_ends_at_exactly_one_despite_rounding():
    # The column sums to 1 - 5e-11, which C2 accepts. The last CDF entry must still be exactly 1.0.
    values = np.array([[0.3, 1.0], [0.7 - 5e-11, 0.0]])
    cdf = cumulative_table(values)
    assert (cdf[-1] == 1.0).all()
    u = np.full(2, np.nextafter(1.0, 0.0))
    # Each column is selected by fancy indexing; the result is always a valid state index.
    np.testing.assert_array_equal(inverse_cdf(cdf, u), [1, 0])


def test_cumulative_table_is_read_only():
    cdf = cumulative_table(np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        cdf[0] = 0.0


# ---------------------------------------------------------------------------
# S1 valid samples, S2 parent-first ordering
# ---------------------------------------------------------------------------


def test_samples_are_complete_valid_assignments():
    model = rain_network()
    samples = model.sample(1_000, seed=0)
    assert len(samples) == 1_000
    for s in samples:
        assert set(s) == {"Rain", "Accident", "Traffic"}
        for v in model.variables:
            assert s[v.name] in v.states


def test_sampler_order_is_topological():
    model = rain_network()
    order = AncestralSampler(model, seed=0).order
    position = {name: i for i, name in enumerate(order)}
    assert set(order) == {"Rain", "Accident", "Traffic"}
    for u, v in model.edges():
        assert position[u] < position[v]


def copy_chain(declared_order: list[str]) -> BayesianNetwork:
    """A -> B -> C, where B and C copy their parent exactly and A is uniform on 3 states."""
    v = {n: DiscreteVariable(n, ("r", "g", "b")) for n in "ABC"}
    model = BayesianNetwork([v[n] for n in declared_order], [("A", "B"), ("B", "C")])
    model.add_cpd(TabularCPD(v["A"], (), np.full(3, 1 / 3)))
    model.add_cpd(TabularCPD(v["B"], (v["A"],), np.eye(3)))
    model.add_cpd(TabularCPD(v["C"], (v["B"],), np.eye(3)))
    return model


def test_parents_are_sampled_before_children():
    # Variables are declared child-first. Sampling in declaration order would
    # read B and C's parent values before they are set.
    samples = copy_chain(["C", "B", "A"]).sample(3_000, seed=1)
    assert all(s["A"] == s["B"] == s["C"] for s in samples)
    assert {s["A"] for s in samples} == {"r", "g", "b"}


def test_zero_probability_states_never_appear():
    x = DiscreteVariable("X", ("never_first", "a", "never_middle", "b", "never_last"))
    model = BayesianNetwork([x], [])
    model.add_cpd(TabularCPD(x, (), [0.0, 0.5, 0.0, 0.5, 0.0]))
    seen = Counter(s["X"] for s in model.sample(20_000, seed=2))
    assert set(seen) == {"a", "b"}


def test_zero_probability_joint_events_never_appear():
    model = copy_chain(["A", "B", "C"])
    # The enumerated joint and the sampler agree on which events are impossible.
    table = joint_table(model)
    impossible = {
        idx for idx in itertools.product(range(3), repeat=3) if table[idx] == 0.0
    }
    for s in model.sample(5_000, seed=3):
        idx = tuple("rgb".index(s[n]) for n in "ABC")
        assert idx not in impossible


def test_singleton_domain():
    on = DiscreteVariable("Power", ("on",))
    model = BayesianNetwork([RAIN, on], [("Rain", "Power")])
    model.add_cpd(rain_cpd())
    model.add_cpd(TabularCPD(on, (RAIN,), [[1.0, 1.0]]))
    assert all(s["Power"] == "on" for s in model.sample(100, seed=0))


# ---------------------------------------------------------------------------
# S3 reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_samples():
    model = rain_network()
    assert AncestralSampler(model, seed=42).sample(500) == AncestralSampler(model, seed=42).sample(500)
    assert model.sample(500, seed=42) == AncestralSampler(model, seed=42).sample(500)


def test_different_seeds_differ():
    model = rain_network()
    assert model.sample(500, seed=1) != model.sample(500, seed=2)


def test_stream_consistency_across_batches():
    model = rain_network()
    sampler = AncestralSampler(model, seed=7)
    batched = sampler.sample(3) + sampler.sample(0) + sampler.sample(4)
    assert batched == AncestralSampler(model, seed=7).sample(7)


def test_unseeded_sampler_runs():
    assert len(rain_network().sample(10)) == 10


@pytest.mark.parametrize("n", [-1, 2.5, "10", True])
def test_invalid_sample_size_is_rejected(n):
    with pytest.raises(ValidationError):
        AncestralSampler(rain_network(), seed=0).sample(n)


def test_numpy_integer_sample_size_is_accepted():
    assert len(rain_network().sample(np.int64(5), seed=0)) == 5


def test_zero_samples():
    assert rain_network().sample(0, seed=0) == []


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------


def test_incomplete_model_cannot_be_sampled():
    model = rain_network_structure()
    model.add_cpd(rain_cpd())
    with pytest.raises(ValidationError, match="missing CPDs"):
        AncestralSampler(model, seed=0)
    with pytest.raises(ValidationError, match="missing CPDs"):
        model.sample(10, seed=0)


def test_sampler_snapshots_model_at_construction():
    model = rain_network()
    sampler = AncestralSampler(model, seed=0)
    model.add_cpd(TabularCPD(RAIN, (), [0.0, 1.0]))  # it now always rains
    samples = sampler.sample(2_000)
    assert any(s["Rain"] == "no" for s in samples)  # the sampler still uses the old CPD
    assert all(s["Rain"] == "yes" for s in model.sample(200, seed=0))


# ---------------------------------------------------------------------------
# S4 distributional correctness
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_samples():
    return rain_network().sample(N_LARGE, seed=20261008)


def test_every_joint_cell_matches_theory(fixture_samples):
    model = rain_network()
    counts = Counter((s["Rain"], s["Accident"], s["Traffic"]) for s in fixture_samples)
    for r, a, t in itertools.product(("no", "yes"), repeat=3):
        p = model.joint_probability({"Rain": r, "Accident": a, "Traffic": t})
        assert_frequency(counts[(r, a, t)], N_LARGE, p, f"P(R={r},A={a},T={t})")


def test_marginal_frequencies_match_theory(fixture_samples):
    def freq(event):
        return sum(1 for s in fixture_samples if event(s))

    assert_frequency(freq(lambda s: s["Rain"] == "yes"), N_LARGE, 0.3, "P(R=1)")
    assert_frequency(freq(lambda s: s["Accident"] == "yes"), N_LARGE, 0.1, "P(A=1)")
    assert_frequency(freq(lambda s: s["Traffic"] == "yes"), N_LARGE, 0.3565, "P(T=1)")


def test_explaining_away_is_visible_in_samples(fixture_samples):
    jam_and_rain = [s for s in fixture_samples if s["Traffic"] == "yes" and s["Rain"] == "yes"]
    jam_no_rain = [s for s in fixture_samples if s["Traffic"] == "yes" and s["Rain"] == "no"]
    for subset, p, label in [
        (jam_and_rain, 0.0285 / 0.2445, "P(A=1|T=1,R=1)"),
        (jam_no_rain, 0.4375, "P(A=1|T=1,R=0)"),
    ]:
        count = sum(1 for s in subset if s["Accident"] == "yes")
        assert_frequency(count, len(subset), p, label)  # SE uses the conditional count N_C


def test_consecutive_samples_are_independent(fixture_samples):
    # S5: use disjoint pairs (2k, 2k+1). If nothing carries over between samples, each pair is independent.
    pairs = list(zip(fixture_samples[0::2], fixture_samples[1::2]))
    for var, state, p in [("Rain", "yes", 0.3), ("Traffic", "yes", 0.3565)]:
        both = sum(1 for a, b in pairs if a[var] == state and b[var] == state)
        assert_frequency(both, len(pairs), p * p, f"P({var}_k={state}, {var}_k+1={state})")


@pytest.mark.parametrize("seed", range(20))
def test_random_networks_match_enumerated_joint(seed):
    """Random structure, cardinalities and CPDs; every joint cell is checked against the oracle."""
    rng = np.random.default_rng(seed)
    n_vars = int(rng.integers(2, 5))
    variables = [
        DiscreteVariable(f"V{i}", tuple(f"s{j}" for j in range(int(rng.integers(1, 4)))))
        for i in range(n_vars)
    ]
    hidden = rng.permutation(n_vars)
    edges = [
        (f"V{hidden[i]}", f"V{hidden[j]}")
        for i in range(n_vars)
        for j in range(i + 1, n_vars)
        if rng.random() < 0.5
    ]
    model = BayesianNetwork(variables, edges)
    by_name = {v.name: v for v in variables}
    for v in variables:
        parents = tuple(by_name[p] for p in rng.permutation(sorted(model.parents(v.name))))
        shape = (v.cardinality, *(p.cardinality for p in parents))
        columns = rng.dirichlet(np.ones(v.cardinality), size=int(np.prod(shape[1:], dtype=int))).T
        model.add_cpd(TabularCPD(v, parents, columns.reshape(shape)))

    n = 50_000
    table = joint_table(model)
    counts = Counter(
        tuple(v.index_of(s[v.name]) for v in variables) for s in model.sample(n, seed=seed)
    )
    for idx in itertools.product(*(range(v.cardinality) for v in variables)):
        assert_frequency(counts[idx], n, table[idx], f"seed={seed} cell={idx}")
