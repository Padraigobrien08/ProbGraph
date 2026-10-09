"""Task 8 (optional): loopy belief propagation (B1–B4; docs/mathematics/loopy_bp.md).

The exact junction tree (M3.6/M3.7) is the oracle. The pathological examples were
found by a prototype search during planning and are written out as constants here.
"""

import numpy as np
import pytest
from support import STUDENTS, misconception_factors

from probgraph import DiscreteFactor, DiscreteVariable, MarkovNetwork
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.inference import JunctionTree, LoopyBeliefPropagation


def binary(n: int) -> list[DiscreteVariable]:
    return [DiscreteVariable(f"X{i}", ("0", "1")) for i in range(n)]


def coupling(a, b, j: float) -> DiscreteFactor:
    """exp(j) when the two agree, exp(-j) otherwise (attractive for j > 0)."""
    return DiscreteFactor([a, b], np.exp(j * np.array([[1.0, -1.0], [-1.0, 1.0]])))


def max_error(result, model, evidence=None) -> float:
    exact = JunctionTree(model, evidence).marginals()
    return max(float(np.abs(result.marginals[n].values - exact[n].values).max()) for n in exact)


# ---------------------------------------------------------------------------
# B1: exact on trees
# ---------------------------------------------------------------------------


def random_tree_model(seed: int) -> MarkovNetwork:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 10))
    xs = [DiscreteVariable(f"X{i}", tuple("abc"[: int(rng.integers(2, 4))])) for i in range(n)]
    factors = []
    for i in range(1, n):  # a random spanning tree
        j = int(rng.integers(i))
        a, b = xs[i], xs[j]
        factors.append(
            DiscreteFactor([a, b], rng.uniform(0.05, 4.0, size=(a.cardinality, b.cardinality)))
        )
    for x in xs:
        if rng.random() < 0.6:
            factors.append(DiscreteFactor([x], rng.uniform(0.1, 3.0, size=x.cardinality)))
    return MarkovNetwork(xs, factors)


@pytest.mark.parametrize("seed", range(40))
def test_exact_on_trees(seed):
    model = random_tree_model(seed)
    result = LoopyBeliefPropagation(model).run()
    assert result.converged
    assert result.iterations <= len(model.variables) + 2  # at most about the diameter (Theorem 1)
    assert max_error(result, model) < 1e-9


@pytest.mark.parametrize("seed", range(20))
def test_exact_on_trees_with_evidence_and_random_initial_messages(seed):
    model = random_tree_model(seed + 100)
    rng = np.random.default_rng(seed)
    evidence = {v.name: v.states[0] for v in model.variables if rng.random() < 0.3}
    if len(evidence) == len(model.variables):
        evidence.pop(next(iter(evidence)))
    for init_seed in (None, 1, 2):  # the fixed point on a tree is unique
        result = LoopyBeliefPropagation(model).run(evidence, seed=init_seed)
        assert result.converged
        assert max_error(result, model, evidence) < 1e-9


def test_exact_on_an_acyclic_factor_graph_with_a_three_way_factor():
    a, b, c, d = binary(4)
    rng = np.random.default_rng(3)
    model = MarkovNetwork(
        [a, b, c, d],
        [
            DiscreteFactor([a, b, c], rng.uniform(0.1, 3.0, size=(2, 2, 2))),
            DiscreteFactor([c, d], rng.uniform(0.1, 3.0, size=(2, 2))),
        ],
    )
    result = LoopyBeliefPropagation(model).run()
    assert result.converged and max_error(result, model) < 1e-9


@pytest.mark.parametrize("seed", range(20))
def test_damping_does_not_change_the_answer_on_trees(seed):
    """Proposition 2: damping keeps the fixed points."""
    model = random_tree_model(seed + 700)
    for damping in (0.3, 0.7):
        result = LoopyBeliefPropagation(model, damping=damping, max_iterations=2000).run()
        assert result.converged and max_error(result, model) < 1e-9


@pytest.mark.parametrize("damping", [0.2, 0.5, 0.9])
def test_damping_is_a_convex_combination_in_probability_space(damping):
    """One iteration from uniform messages: ν = (1 - α) F + α · uniform, exactly (§4).

    With X0 – X1 and local potentials u0, u1, the belief of X1 after one iteration is
    u1 · ν (normalised), so F can be recovered from an undamped one-iteration run.
    """
    x0, x1 = binary(2)
    u0, u1 = np.array([2.0, 1.0]), np.array([1.0, 3.0])
    model = MarkovNetwork(
        [x0, x1],
        [
            DiscreteFactor([x0, x1], [[4.0, 1.0], [1.0, 2.0]]),
            DiscreteFactor([x0], u0),
            DiscreteFactor([x1], u1),
        ],
    )

    def belief_after_one(alpha: float) -> np.ndarray:
        return (
            LoopyBeliefPropagation(model, damping=alpha, max_iterations=1)
            .run()
            .marginals["X1"]
            .values
        )

    undamped = belief_after_one(0.0) / u1
    f = undamped / undamped.sum()
    expected_message = (1 - damping) * f + damping * np.array([0.5, 0.5])
    expected = u1 * expected_message
    np.testing.assert_allclose(belief_after_one(damping), expected / expected.sum(), atol=1e-12)


# ---------------------------------------------------------------------------
# B2: approximate on cycles
# ---------------------------------------------------------------------------


def test_misconception_converges_to_approximate_beliefs():
    model = MarkovNetwork(STUDENTS, misconception_factors())
    result = LoopyBeliefPropagation(model, max_iterations=1000).run()
    assert result.converged and 200 < result.iterations < 300  # about 210, past the default
    # Measured and reported, not hidden: P(A=1) ≈ 0.43 against an exact 130031/720184 ≈ 0.18.
    assert result.marginals["A"].value({"A": "1"}) == pytest.approx(0.4344, abs=1e-3)
    assert 0.25 < max_error(result, model) < 0.26


def four_cycle_with_conflicting_evidence(j: float = 3.0, h: float = 1.0) -> MarkovNetwork:
    xs = binary(4)
    factors = [coupling(xs[a], xs[b], j) for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)]]
    factors += [DiscreteFactor([xs[0]], np.exp([h, -h])), DiscreteFactor([xs[2]], np.exp([-h, h]))]
    return MarkovNetwork(xs, factors)


def test_strong_coupling_converges_slowly_without_damping():
    model = four_cycle_with_conflicting_evidence()
    undamped = LoopyBeliefPropagation(model, max_iterations=1000).run()
    damped = LoopyBeliefPropagation(model, damping=0.5, max_iterations=1000).run()
    assert not undamped.converged
    assert damped.converged and damped.iterations < 100
    # By symmetry the exact marginals of X1 and X3 are 1/2; the damped fixed point agrees.
    assert damped.marginals["X1"].values.tolist() == pytest.approx([0.5, 0.5], abs=1e-6)


# ---------------------------------------------------------------------------
# B3: a genuine period-2 oscillation, and what damping does to it
# ---------------------------------------------------------------------------

GRID_J = 4.547846749285817
GRID_EDGES = [
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
]
GRID_SIGNS = [1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
GRID_LOG_UNARIES = [
    [0.4735404815646211, -0.3518676179034963],
    [-0.6327107355230263, -0.3116372312686761],
    [0.0206629896736218, -1.1625153873194172],
    [-0.10939583196627287, -0.6229554736265326],
    [-0.3661336773517258, -0.27212949142865495],
    [-0.15815007818457727, 0.2058152681870664],
    [0.5212566847213388, -0.06426733147201713],
    [0.6832317352748429, -0.33259733674330677],
    [0.17575503504650986, 0.4517350908259043],
]


def oscillating_grid() -> MarkovNetwork:
    xs = binary(9)
    factors = [
        coupling(xs[a], xs[b], s * GRID_J) for (a, b), s in zip(GRID_EDGES, GRID_SIGNS, strict=True)
    ]
    factors += [DiscreteFactor([x], np.exp(u)) for x, u in zip(xs, GRID_LOG_UNARIES, strict=True)]
    return MarkovNetwork(xs, factors)


def test_undamped_grid_oscillates_with_period_two():
    model = oscillating_grid()
    beliefs = [
        LoopyBeliefPropagation(model, max_iterations=t, tolerance=0.0)
        .run()
        .marginals["X4"]
        .values[0]
        for t in (2000, 2001, 2002)
    ]
    assert beliefs[0] == pytest.approx(beliefs[2], abs=1e-9)  # period 2...
    assert abs(beliefs[0] - beliefs[1]) > 0.99  # ...flipping between 0 and 1
    result = LoopyBeliefPropagation(model, max_iterations=2000).run()
    assert not result.converged and result.residual > 0.9


def test_damping_converges_but_to_confidently_wrong_beliefs():
    model = oscillating_grid()
    result = LoopyBeliefPropagation(model, damping=0.5, max_iterations=2000).run()
    assert result.converged and result.iterations < 200
    assert max_error(result, model) > 0.4  # convergence is not accuracy (loopy_bp.md §3)


# ---------------------------------------------------------------------------
# B4: the convergence flag is honest
# ---------------------------------------------------------------------------


def test_converged_is_reported_only_below_tolerance():
    model = MarkovNetwork(STUDENTS, misconception_factors())
    short = LoopyBeliefPropagation(model, max_iterations=1).run()
    assert not short.converged and short.iterations == 1 and short.residual > 0
    full = LoopyBeliefPropagation(model, max_iterations=1000, tolerance=1e-12).run()
    assert full.converged and full.residual < 1e-12


@pytest.mark.parametrize("seed", range(20))
def test_residual_matches_the_flag(seed):
    model = random_tree_model(seed + 300)
    for iterations in (1, 2, 50):
        result = LoopyBeliefPropagation(model, max_iterations=iterations, tolerance=1e-10).run()
        assert result.converged == (result.residual < 1e-10)


# ---------------------------------------------------------------------------
# Validation and edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"damping": -0.1}, {"damping": 1.0}, {"max_iterations": 0}, {"tolerance": -1.0}],
)
def test_parameter_validation(kwargs):
    with pytest.raises(ValidationError):
        LoopyBeliefPropagation(MarkovNetwork(STUDENTS, misconception_factors()), **kwargs)


def test_accepts_a_bayesian_network():
    from support import late_network

    result = LoopyBeliefPropagation(late_network()).run({"Late": "yes"})
    assert result.converged
    assert set(result.marginals) == {"Rain", "Accident", "Traffic", "Umbrella"}


def test_contradictory_evidence_raises():
    a, b = binary(2)
    model = MarkovNetwork([a, b], [DiscreteFactor([a, b], np.eye(2))])  # a must equal b
    with pytest.raises(ZeroProbabilityEvidenceError):
        LoopyBeliefPropagation(model).run({"X0": "0", "X1": "1"})


def test_isolated_variable_and_unary_only_model():
    a, b = binary(2)
    model = MarkovNetwork([a, b], [DiscreteFactor([a], [1.0, 3.0])])
    result = LoopyBeliefPropagation(model).run()
    assert result.converged
    assert result.marginals["X0"].values.tolist() == pytest.approx([0.25, 0.75])
    assert result.marginals["X1"].values.tolist() == pytest.approx([0.5, 0.5])


def test_deterministic():
    model = oscillating_grid()
    a = LoopyBeliefPropagation(model, damping=0.5).run()
    b = LoopyBeliefPropagation(model, damping=0.5).run()
    assert a.iterations == b.iterations
    for name in a.marginals:
        assert np.array_equal(a.marginals[name].values, b.marginals[name].values)


def test_all_pairs_of_grid_variables_are_reported():
    result = LoopyBeliefPropagation(oscillating_grid(), damping=0.5).run()
    assert sorted(result.marginals) == sorted(f"X{i}" for i in range(9))
    for marginal in result.marginals.values():
        assert marginal.total() == pytest.approx(1.0)
