"""M5.2: the forward sweep: filtering and likelihood (F1, F2, F4, F6; forward_backward.md §1–5)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from probgraph import DiscreteVariable, VariableElimination
from probgraph.exceptions import UnknownStateError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import JunctionTree
from probgraph.temporal import ForwardBackward, HiddenMarkovModel

WEATHER = DiscreteVariable("Weather", ("rain", "dry"))
UMBRELLA = DiscreteVariable("Umbrella", ("umbrella", "none"))
U, NONE = "umbrella", "none"
FIVE_DAYS = [U, U, NONE, U, U]


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
    return HiddenMarkovModel(
        DiscreteVariable("X", tuple(f"x{i}" for i in range(k))),
        DiscreteVariable("Y", tuple(f"y{i}" for i in range(m))),
        rng.dirichlet(np.ones(k)),
        rng.dirichlet(np.ones(k), size=k),
        rng.dirichlet(np.ones(m), size=k),
    )


def random_observations(model: HiddenMarkovModel, seed: int, length: int, missing: float):
    rng = np.random.default_rng(seed)
    _, ys = model.sample(length, seed=seed)
    return [None if rng.random() < missing else y for y in ys]


def brute_force(model: HiddenMarkovModel, ys, exact: bool = False):
    """Enumerate all K^T paths: returns (P(y), [P(X_t | y_1:t) for each t])."""
    convert = (lambda v: Fraction(v).limit_denominator(10**6)) if exact else float
    pi = [convert(v) for v in model.initial]
    a = [[convert(v) for v in row] for row in model.transition]
    b = [[convert(v) for v in row] for row in model.emission]
    symbols = model.observed.states
    k = model.n_states

    def joint(path, t):
        p = pi[path[0]]
        for s in range(t + 1):
            if s > 0:
                p *= a[path[s - 1]][path[s]]
            if ys[s] is not None:
                p *= b[path[s]][symbols.index(ys[s])]
        return p

    filtered = []
    total = None
    for t in range(len(ys)):
        weights = [0] * k
        for path in itertools.product(range(k), repeat=t + 1):
            weights[path[-1]] += joint(path, t)
        total = sum(weights)
        filtered.append([w / total for w in weights] if total else None)
    return total, filtered


# ---------------------------------------------------------------------------
# Fixture F1, exactly
# ---------------------------------------------------------------------------


def test_f1_filtering_and_likelihood():
    fb = ForwardBackward(umbrella_world(), [U, U])
    assert fb.filtered[0] == pytest.approx([9 / 11, 2 / 11], abs=1e-15)
    assert fb.filtered[1] == pytest.approx([621 / 703, 82 / 703], abs=1e-15)
    assert fb.log_likelihood == pytest.approx(math.log(703 / 2000), abs=1e-15)


def test_f1_five_days_against_exact_fractions():
    fb = ForwardBackward(umbrella_world(), FIVE_DAYS)
    total, filtered = brute_force(umbrella_world(), FIVE_DAYS, exact=True)
    assert total == Fraction(68607401, 2_000_000_000)
    assert fb.log_likelihood == pytest.approx(math.log(total), abs=1e-14)
    for t in range(5):
        assert fb.filtered[t] == pytest.approx([float(p) for p in filtered[t]], abs=1e-15)
    assert fb.filtered[2][0] == pytest.approx(0.19066793972352525, abs=1e-15)


# ---------------------------------------------------------------------------
# F1: brute force and the general algorithms on the unrolled network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_filtering_matches_brute_force(seed):
    model = random_hmm(seed)
    ys = random_observations(model, seed, length=1 + seed % 6, missing=0.25)
    fb = ForwardBackward(model, ys)
    total, filtered = brute_force(model, ys)
    assert fb.log_likelihood == pytest.approx(math.log(total), rel=1e-12, abs=1e-12)
    for t, belief in enumerate(filtered):
        assert fb.filtered[t] == pytest.approx(belief, abs=1e-12)
    # Step likelihoods are the one-step predictive probabilities (Proposition 2).
    prefix = [brute_force(model, ys[: t + 1])[0] for t in range(len(ys))]
    expected = [math.log(prefix[0])] + [
        math.log(prefix[t] / prefix[t - 1]) for t in range(1, len(ys))
    ]
    assert fb.step_log_likelihoods == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_filtering_matches_the_unrolled_network(seed):
    model = random_hmm(seed + 100, k=3, m=3)
    length = 12
    ys = random_observations(model, seed, length, missing=0.3)
    fb = ForwardBackward(model, ys)
    network = model.to_bayesian_network(length)
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys) if y is not None}
    assert fb.log_likelihood == pytest.approx(
        VariableElimination(network).log_probability_of_evidence(evidence), rel=1e-12
    )
    for t in range(length):
        seen = {k: v for k, v in evidence.items() if int(k.split("_")[1]) <= t + 1}
        marginal = JunctionTree(network, seen).marginal(f"X_{t + 1}")
        assert fb.filtered[t] == pytest.approx(marginal.values, abs=1e-12)


def test_all_missing_is_prediction_without_evidence():
    """No observations: log P = 0 and f_t = π A^(t-1) (P18 §4)."""
    model = random_hmm(3, k=3, m=2)
    fb = ForwardBackward(model, [None] * 8)
    assert fb.log_likelihood == 0.0
    for t in range(8):
        expected = model.initial @ np.linalg.matrix_power(model.transition, t)
        assert fb.filtered[t] == pytest.approx(expected, abs=1e-14)


# ---------------------------------------------------------------------------
# F2 and F4: normalisation and underflow
# ---------------------------------------------------------------------------


def unnormalised_likelihood(model: HiddenMarkovModel, ys) -> float:
    """Proposition 1 without normalising: what goes wrong for long sequences."""
    symbols = model.observed.states
    alpha = model.initial * model.emission[:, symbols.index(ys[0])]
    for y in ys[1:]:
        alpha = (alpha @ model.transition) * model.emission[:, symbols.index(y)]
    return float(alpha.sum())


@pytest.mark.parametrize(
    ("ys", "exact"),
    [
        ([U] * 1000, -414.0917995185),
        ([U] * 5000, -2069.560503770881),
        ([U, NONE] * 500, -868.4784295542072),
    ],
)
def test_f4_long_sequences(ys, exact):
    fb = ForwardBackward(umbrella_world(), ys)
    assert fb.log_likelihood == pytest.approx(exact, rel=1e-9)
    assert np.isfinite(fb.filtered).all()
    assert fb.filtered.sum(axis=1) == pytest.approx(np.ones(len(ys)), abs=1e-12)
    if exact < -745:
        # Without normalising, the answer is either 0 or stuck at the smallest subnormal,
        # 5e-324: 0.63 x 5e-324 rounds back up to 5e-324. Either way it is wrong by > 1000 nats.
        naive = unnormalised_likelihood(umbrella_world(), ys)
        assert naive in (0.0, 5e-324)
        assert naive == 0.0 or math.log(naive) > exact + 1000


def test_long_sequence_with_umbrellas_settles_to_the_fixed_point():
    """With u every day, f_t converges to the fixed point of the update map."""
    fb = ForwardBackward(umbrella_world(), [U] * 200)
    f = fb.filtered[-1]
    predicted = f @ umbrella_world().transition
    updated = predicted * umbrella_world().emission[:, 0]
    assert updated / updated.sum() == pytest.approx(f, abs=1e-14)


def test_subnormal_emissions_are_handled_exactly():
    """Emission probabilities of 1e-320 (subnormal) against log-space VE (§4)."""
    tiny = 1e-320
    model = HiddenMarkovModel(
        WEATHER,
        UMBRELLA,
        [0.5, 0.5],
        [[0.7, 0.3], [0.3, 0.7]],
        [[1.0, tiny], [1.0 - 0.5, 0.5]],
    )
    ys = [NONE, NONE, U, NONE, U, U, NONE]
    fb = ForwardBackward(model, ys)
    network = model.to_bayesian_network(len(ys))
    evidence = {f"Umbrella_{t + 1}": y for t, y in enumerate(ys)}
    ve = VariableElimination(network)
    assert fb.log_likelihood == pytest.approx(ve.log_probability_of_evidence(evidence), rel=1e-12)
    last = ve.query([f"Weather_{len(ys)}"], evidence).values
    assert fb.filtered[-1] == pytest.approx(last, abs=1e-12)

    # Both states emitting 'none' with tiny probability: every step is ~1e-320.
    both = HiddenMarkovModel(
        WEATHER, UMBRELLA, [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]], [[1.0, tiny], [1.0, tiny / 4]]
    )
    fb = ForwardBackward(both, [NONE] * 3)
    total, _ = brute_force(both, [NONE] * 3, exact=False)
    log_tiny = math.log(tiny)  # math.log of a subnormal is exact enough; total itself underflows
    assert total == 0.0
    assert fb.log_likelihood == pytest.approx(
        VariableElimination(both.to_bayesian_network(3)).log_probability_of_evidence(
            {f"Umbrella_{t}": NONE for t in (1, 2, 3)}
        ),
        rel=1e-12,
    )
    assert fb.log_likelihood < 3 * log_tiny + 1


# ---------------------------------------------------------------------------
# F6: impossible sequences
# ---------------------------------------------------------------------------


def test_impossible_sequence():
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, [1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]
    )
    fb = ForwardBackward(model, [U, U, NONE, U])  # rain forever never emits 'none'
    assert fb.log_likelihood == -math.inf
    with pytest.raises(ZeroProbabilityEvidenceError, match="step 3"):
        _ = fb.filtered
    with pytest.raises(ZeroProbabilityEvidenceError):
        _ = fb.step_log_likelihoods


def test_impossible_through_the_transition():
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, [0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert ForwardBackward(model, [NONE, NONE, U]).log_likelihood == -math.inf


# ---------------------------------------------------------------------------
# Validation and storage
# ---------------------------------------------------------------------------


def test_results_are_read_only_and_shaped():
    fb = ForwardBackward(umbrella_world(), FIVE_DAYS)
    assert fb.filtered.shape == (5, 2) and fb.step_log_likelihoods.shape == (5,)
    assert fb.length == 5 and fb.model is not None
    with pytest.raises(ValueError):
        fb.filtered[0, 0] = 1.0


def test_validation():
    with pytest.raises(UnknownStateError, match="observation 2"):
        ForwardBackward(umbrella_world(), [U, "snow"])
    with pytest.raises(ValidationError, match="at least one"):
        ForwardBackward(umbrella_world(), [])
    with pytest.raises(ValidationError, match="HiddenMarkovModel"):
        ForwardBackward("umbrella", [U])
    with pytest.raises(ValidationError, match="sequence"):
        ForwardBackward(umbrella_world(), "umbrella")


def extreme_hmm(seed: int) -> HiddenMarkovModel:
    """Entries spread over exp(-U(0, 760)): many are subnormal, some underflow to 0."""
    rng = np.random.default_rng(seed)

    def rows(shape):
        table = np.exp(-rng.uniform(0, 760, size=shape))
        table[..., int(rng.integers(shape[-1]))] = 1.0  # keep every row normalisable
        return table / table.sum(axis=-1, keepdims=True)

    return HiddenMarkovModel(
        DiscreteVariable("X", ("x0", "x1", "x2")),
        DiscreteVariable("Y", ("y0", "y1", "y2")),
        rows((3,)),
        rows((3, 3)),
        rows((3, 3)),
    )


@pytest.mark.parametrize("seed", range(40))
def test_extreme_parameters_match_log_space_elimination(seed):
    model = extreme_hmm(seed)
    rng = np.random.default_rng(seed + 1000)
    ys = [str(model.observed.states[int(i)]) for i in rng.integers(0, 3, size=6)]
    fb = ForwardBackward(model, ys)
    network = model.to_bayesian_network(len(ys))
    evidence = {f"Y_{t + 1}": y for t, y in enumerate(ys)}
    expected = VariableElimination(network).log_probability_of_evidence(evidence)
    if expected == -math.inf:
        assert fb.log_likelihood == -math.inf
        return
    assert fb.log_likelihood == pytest.approx(expected, rel=1e-9)
    last = VariableElimination(network).query([f"X_{len(ys)}"], evidence).values
    assert fb.filtered[-1] == pytest.approx(last, abs=1e-9)


@pytest.mark.parametrize("seed", [112, 706, 2173])
def test_subnormal_step_totals_are_exact_in_log_space(seed):
    """Found by search: one step's linear total is subnormal (1e-308 to 1e-314), where a float
    carries only a few significant digits; linear arithmetic filtered seed 2173 to the wrong
    state. The log-space recursion is exact to 1e-13 (forward_backward.md §4)."""
    model = extreme_hmm(seed)
    rng = np.random.default_rng(seed + 1000)
    ys = [str(model.observed.states[int(i)]) for i in rng.integers(0, 3, size=6)]
    fb = ForwardBackward(model, ys)
    network = model.to_bayesian_network(len(ys))
    for t in range(len(ys)):
        evidence = {f"Y_{s + 1}": ys[s] for s in range(t + 1)}
        belief = VariableElimination(network).query([f"X_{t + 1}"], evidence).values
        assert fb.filtered[t] == pytest.approx(belief, rel=0, abs=1e-13)


def test_a_symbol_no_state_emits_is_impossible():
    model = HiddenMarkovModel(
        WEATHER, UMBRELLA, [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]], [[1.0, 0.0], [1.0, 0.0]]
    )
    fb = ForwardBackward(model, [U, NONE, U])
    assert fb.log_likelihood == -math.inf
    with pytest.raises(ZeroProbabilityEvidenceError, match="step 2"):
        _ = fb.filtered
