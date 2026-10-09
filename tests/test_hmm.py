"""M5.1: hidden Markov models: validation, unrolling, sampling, stationarity (H1–H4; P18)."""

import itertools
import math
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest
from support import joint_table

from probgraph import DiscreteVariable, VariableElimination
from probgraph.exceptions import ValidationError
from probgraph.temporal import HiddenMarkovModel

WEATHER = DiscreteVariable("Weather", ("rain", "dry"))
UMBRELLA = DiscreteVariable("Umbrella", ("umbrella", "none"))
Z = 5.0


def umbrella_world() -> HiddenMarkovModel:
    return HiddenMarkovModel(
        WEATHER,
        UMBRELLA,
        initial=[0.5, 0.5],
        transition=[[0.7, 0.3], [0.3, 0.7]],
        emission=[[0.9, 0.1], [0.2, 0.8]],
    )


def random_hmm(seed: int, k: int | None = None, m: int | None = None) -> HiddenMarkovModel:
    rng = np.random.default_rng(seed)
    k = k or int(rng.integers(1, 4))
    m = m or int(rng.integers(1, 4))
    hidden = DiscreteVariable("X", tuple(f"x{i}" for i in range(k)))
    observed = DiscreteVariable("Y", tuple(f"y{i}" for i in range(m)))
    return HiddenMarkovModel(
        hidden,
        observed,
        rng.dirichlet(np.ones(k)),
        rng.dirichlet(np.ones(k), size=k),
        rng.dirichlet(np.ones(m), size=k),
    )


def bernstein(p: float, n: int) -> float:
    return Z * math.sqrt(p * (1 - p) / n) + Z**2 / (3 * n)


# ---------------------------------------------------------------------------
# H1: validation and storage
# ---------------------------------------------------------------------------


def test_construction_and_read_only_storage():
    initial = np.array([0.5, 0.5])
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, initial, [[0.7, 0.3], [0.3, 0.7]], [[0.9, 0.1], [0.2, 0.8]]
    )
    initial[0] = 0.9  # the model keeps its own copy
    assert model.initial.tolist() == [0.5, 0.5]
    assert model.transition[0, 1] == 0.3 and model.emission[1, 0] == 0.2
    assert (model.n_states, model.n_symbols) == (2, 2)
    assert model.hidden is WEATHER and model.observed is UMBRELLA
    for table in (model.initial, model.transition, model.emission):
        with pytest.raises(ValueError):
            table[0] = 0.0
    assert "Weather" in repr(model) and "Umbrella" in repr(model)


GOOD = {
    "initial": [0.5, 0.5],
    "transition": [[0.7, 0.3], [0.3, 0.7]],
    "emission": [[0.9, 0.1], [0.2, 0.8]],
}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"initial": [0.5, 0.4]}, "initial sums to 0.9"),
        ({"initial": [1.0]}, r"initial has shape \(1,\); expected \(2,\)"),
        ({"transition": [[0.7, 0.3], [0.3, 0.6]]}, r"row 1 of transition \(from Weather=dry\)"),
        ({"transition": [[0.7, 0.3, 0.0], [0.3, 0.7, 0.0]]}, r"transition has shape \(2, 3\)"),
        ({"emission": [[1.1, -0.1], [0.2, 0.8]]}, "negative"),
        ({"emission": [[0.9, 0.1]]}, r"emission has shape \(1, 2\); expected \(2, 2\)"),
        ({"emission": [[np.nan, 0.1], [0.2, 0.8]]}, "finite"),
        ({"initial": ["a", "b"]}, "real numbers"),
    ],
)
def test_invalid_parameters_are_rejected(override, match):
    with pytest.raises(ValidationError, match=match):
        HiddenMarkovModel(WEATHER, UMBRELLA, **{**GOOD, **override})


def test_variables_must_be_distinct_discrete_variables():
    with pytest.raises(ValidationError, match="distinct names"):
        HiddenMarkovModel(WEATHER, WEATHER, GOOD["initial"], GOOD["transition"], GOOD["transition"])
    with pytest.raises(ValidationError, match="DiscreteVariable"):
        HiddenMarkovModel("Weather", UMBRELLA, **GOOD)


def test_free_parameters_are_tied_across_time():
    """d = (K-1) + K(K-1) + K(M-1), whatever T is (markov_chains.md §2)."""
    for seed in range(10):
        model = random_hmm(seed)
        k, m = model.n_states, model.n_symbols
        assert model.n_free_parameters == (k - 1) + k * (k - 1) + k * (m - 1)
        # The unrolled T = 2 network has one extra emission CPD's worth of parameters.
        unrolled = model.to_bayesian_network(2).n_free_parameters
        assert model.n_free_parameters == unrolled - k * (m - 1)


# ---------------------------------------------------------------------------
# H2: unrolling is exact
# ---------------------------------------------------------------------------


def hmm_joint(model: HiddenMarkovModel, xs, ys) -> float:
    p = model.initial[xs[0]] * model.emission[xs[0], ys[0]]
    for t in range(1, len(xs)):
        p *= model.transition[xs[t - 1], xs[t]] * model.emission[xs[t], ys[t]]
    return float(p)


@pytest.mark.parametrize("seed", range(20))
def test_unrolled_joint_matches_the_hmm_factorisation(seed):
    model = random_hmm(seed)
    length = 1 + seed % 4
    network = model.to_bayesian_network(length)
    network.validate()
    hidden = [f"X_{t}" for t in range(1, length + 1)]
    observed = [f"Y_{t}" for t in range(1, length + 1)]
    assert {v.name for v in network.variables} == set(hidden) | set(observed)
    table = joint_table(network)
    assert table.sum() == pytest.approx(1.0, abs=1e-12)
    position = {v.name: i for i, v in enumerate(network.variables)}
    k, m = model.n_states, model.n_symbols
    for xs in itertools.product(range(k), repeat=length):
        for ys in itertools.product(range(m), repeat=length):
            index = [0] * len(position)
            for t in range(length):
                index[position[hidden[t]]] = xs[t]
                index[position[observed[t]]] = ys[t]
            assert table[tuple(index)] == pytest.approx(hmm_joint(model, xs, ys), abs=1e-15)


def test_unrolled_structure():
    network = umbrella_world().to_bayesian_network(3)
    assert set(network.edges()) == {
        ("Weather_1", "Weather_2"),
        ("Weather_2", "Weather_3"),
        ("Weather_1", "Umbrella_1"),
        ("Weather_2", "Umbrella_2"),
        ("Weather_3", "Umbrella_3"),
    }
    assert network.variable("Weather_2").states == WEATHER.states
    assert network.variable("Umbrella_3").states == UMBRELLA.states


def test_where_to_begin_umbrella_by_enumeration():
    """P(rain_2 | u_1, u_2) = 621/703 on the unrolled network (spec §10, fixture F1)."""
    ve = VariableElimination(umbrella_world().to_bayesian_network(2))
    evidence = {"Umbrella_1": "umbrella", "Umbrella_2": "umbrella"}
    assert ve.query(["Weather_2"], evidence).value({"Weather_2": "rain"}) == pytest.approx(
        float(Fraction(621, 703)), abs=1e-14
    )
    assert ve.probability_of_evidence(evidence) == pytest.approx(703 / 2000, abs=1e-15)


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_length_must_be_a_positive_int(length):
    with pytest.raises(ValidationError, match="length"):
        umbrella_world().to_bayesian_network(length)
    with pytest.raises(ValidationError, match="length"):
        umbrella_world().sample(length, seed=0)


# ---------------------------------------------------------------------------
# H3: sampling
# ---------------------------------------------------------------------------


def test_sampling_is_reproducible_and_prefix_consistent():
    model = umbrella_world()
    states, observations = model.sample(50, seed=3)
    assert len(states) == len(observations) == 50
    assert set(states) <= set(WEATHER.states) and set(observations) <= set(UMBRELLA.states)
    assert model.sample(50, seed=3) == (states, observations)
    short = model.sample(20, seed=3)
    assert short == (states[:20], observations[:20])
    assert model.sample(50, seed=4) != (states, observations)


def test_transition_and_emission_frequencies_are_within_bernstein_bounds():
    """Given the number of visits to i, the moves out of i are i.i.d. draws from row i of A."""
    model = random_hmm(7, k=3, m=3)
    states, observations = model.sample(200_000, seed=1)
    index = {s: i for i, s in enumerate(model.hidden.states)}
    symbol = {s: i for i, s in enumerate(model.observed.states)}
    xs = [index[s] for s in states]
    ys = [symbol[s] for s in observations]
    moves = Counter(itertools.pairwise(xs))
    emitted = Counter(zip(xs, ys, strict=True))
    visits = Counter(xs[:-1])
    totals = Counter(xs)
    for i in range(3):
        for j in range(3):
            p = float(model.transition[i, j])
            assert abs(moves[(i, j)] / visits[i] - p) <= bernstein(p, visits[i])
            q = float(model.emission[i, j])
            assert abs(emitted[(i, j)] / totals[i] - q) <= bernstein(q, totals[i])


def test_short_sequences_follow_the_unrolled_joint():
    """Every (x1, y1, x2, y2) cell over 20,000 seeds, against the unrolled network's joint."""
    model = random_hmm(11, k=2, m=2)
    n = 20_000
    cells = Counter(
        tuple(itertools.chain.from_iterable(zip(*model.sample(2, seed=s), strict=True)))
        for s in range(n)
    )
    states, symbols = model.hidden.states, model.observed.states
    for x1, y1, x2, y2 in itertools.product(states, symbols, states, symbols):
        xs = (states.index(x1), states.index(x2))
        ys = (symbols.index(y1), symbols.index(y2))
        p = hmm_joint(model, xs, ys)
        assert abs(cells[(x1, y1, x2, y2)] / n - p) <= bernstein(p, n)


def test_zero_probability_states_are_never_sampled():
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, [1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.5, 0.5]]
    )
    states, observations = model.sample(1000, seed=0)
    assert set(states) == {"rain"} and set(observations) == {"none"}


# ---------------------------------------------------------------------------
# H4: the stationary distribution
# ---------------------------------------------------------------------------


def chain(transition) -> HiddenMarkovModel:
    k = len(transition)
    hidden = DiscreteVariable("X", tuple(f"x{i}" for i in range(k)))
    observed = DiscreteVariable("Y", ("y",))
    return HiddenMarkovModel(hidden, observed, np.full(k, 1 / k), transition, np.ones((k, 1)))


def test_umbrella_world_is_stationary_at_one_half():
    assert umbrella_world().stationary_distribution() == pytest.approx([0.5, 0.5], abs=1e-15)


@pytest.mark.parametrize("seed", range(20))
def test_two_state_closed_form(seed):
    """μ = (b, a)/(a + b), and p_{t+k} - μ = (1 - a - b)^k (p_t - μ) (markov_chains.md §6)."""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.01, 1.0, size=2)
    model = chain([[1 - a, a], [b, 1 - b]])
    mu = np.array([b, a]) / (a + b)
    assert model.stationary_distribution() == pytest.approx(mu, abs=1e-14)
    p = rng.dirichlet([1.0, 1.0])
    for k in range(10):
        moved = p @ np.linalg.matrix_power(model.transition, k)
        assert moved - mu == pytest.approx((1 - a - b) ** k * (p - mu), abs=1e-13)


def closed_classes(transition: np.ndarray) -> list[set[int]]:
    """An independent oracle: reachability by boolean matrix powers, then closed classes."""
    k = len(transition)
    reach = np.eye(k, dtype=bool) | (transition > 0)
    for _ in range(k):
        reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
    classes = {frozenset(j for j in range(k) if reach[i, j] and reach[j, i]) for i in range(k)}
    return [set(c) for c in classes if all(not reach[i, j] or j in c for i in c for j in range(k))]


@pytest.mark.parametrize("seed", range(60))
def test_stationary_distribution_on_random_sparse_chains(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 6))
    transition = rng.dirichlet(np.ones(k), size=k) * (rng.random((k, k)) < rng.uniform(0.1, 0.6))
    transition[np.arange(k), rng.integers(0, k, size=k)] += 0.2  # every row nonzero
    transition /= transition.sum(axis=1, keepdims=True)
    model = chain(transition)
    closed = closed_classes(transition)
    if len(closed) != 1:
        with pytest.raises(ValidationError, match="closed"):
            model.stationary_distribution()
        return
    mu = model.stationary_distribution()
    assert mu.sum() == pytest.approx(1.0, abs=1e-12)
    assert mu @ model.transition == pytest.approx(mu, abs=1e-12)
    for i in range(k):
        if i in closed[0]:
            assert mu[i] > 0
        else:
            assert mu[i] == 0.0  # transient states carry no mass (Theorem 1)


@pytest.mark.parametrize("seed", range(20))
def test_several_closed_classes_always_raise(seed):
    """Block chains: 2-3 closed blocks plus transient states feeding them, shuffled."""
    rng = np.random.default_rng(seed)
    sizes = list(rng.integers(1, 3, size=int(rng.integers(2, 4))))
    n_transient = int(rng.integers(0, 3))
    k = sum(sizes) + n_transient
    transition = np.zeros((k, k))
    start = 0
    for size in sizes:
        block = slice(start, start + size)
        transition[block, block] = rng.dirichlet(np.ones(size), size=size)
        start += size
    transition[start:, :] = rng.dirichlet(np.ones(k), size=n_transient)
    order = rng.permutation(k)
    shuffled = transition[np.ix_(order, order)]
    assert len(closed_classes(shuffled)) == len(sizes)
    with pytest.raises(ValidationError, match=f"{len(sizes)} closed classes"):
        chain(shuffled).stationary_distribution()


def test_transient_state_and_closed_class_by_hand():
    mu = chain([[0.5, 0.5, 0.0], [0.0, 0.2, 0.8], [0.0, 0.6, 0.4]]).stationary_distribution()
    assert mu == pytest.approx([0.0, 3 / 7, 4 / 7], abs=1e-15)


def test_two_closed_classes_are_named():
    with pytest.raises(ValidationError, match=r"\['x0'\].*\['x1'\]"):
        chain([[1.0, 0.0], [0.0, 1.0]]).stationary_distribution()


def test_periodic_chain_is_unique_but_does_not_converge():
    model = chain([[0.0, 1.0], [1.0, 0.0]])
    assert model.stationary_distribution() == pytest.approx([0.5, 0.5])
    p = np.array([1.0, 0.0])
    assert p @ np.linalg.matrix_power(model.transition, 101) == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize("seed", range(10))
def test_aperiodic_chains_converge_at_the_second_eigenvalue(seed):
    """‖p A^k - μ‖^(1/k) → |λ₂| (markov_chains.md §6, stated general case)."""
    model = random_hmm(seed + 30, k=4, m=1)
    mu = model.stationary_distribution()
    rate = sorted(np.abs(np.linalg.eigvals(model.transition)))[-2]
    p = np.array([1.0, 0.0, 0.0, 0.0])
    errors = []
    for _ in range(10_000):
        errors.append(np.abs(p - mu).max())
        if errors[-1] <= 1e-10:  # stop well above rounding error
            break
        p = p @ model.transition
    assert errors[-1] <= 1e-10, "p A^k does not approach μ"
    late = len(errors) - 1
    early = late // 2
    assert late - early >= 4
    observed_rate = (errors[late] / errors[early]) ** (1 / (late - early))
    assert observed_rate == pytest.approx(rate, abs=0.06)
