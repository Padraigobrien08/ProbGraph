"""M6.1: the Gibbs sampler: local conditionals, scans, chains, initialisation (G1, G4, G5)."""

import numpy as np
import pytest
from support import (
    STUDENTS,
    gibbs_table,
    joint_table,
    late_network,
    misconception_factors,
    random_network,
)

from probgraph import (
    BayesianNetwork,
    DiscreteFactor,
    DiscreteVariable,
    MarkovNetwork,
    TabularCPD,
    VariableElimination,
)
from probgraph.exceptions import (
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)
from probgraph.mcmc import Chain, GibbsSampler


def f1() -> BayesianNetwork:
    """Fixture F1: X -> Y, P(X=1) = 3/10, P(Y=1 | X=0) = 1/5, P(Y=1 | X=1) = 9/10."""
    x, y = DiscreteVariable("X", ("0", "1")), DiscreteVariable("Y", ("0", "1"))
    model = BayesianNetwork([x, y], [("X", "Y")])
    model.add_cpd(TabularCPD(x, (), [0.7, 0.3]))
    model.add_cpd(TabularCPD(y, (x,), [[0.8, 0.1], [0.2, 0.9]]))
    return model


def copy_network() -> BayesianNetwork:
    """Fixture F3: Y = X exactly."""
    x, y = DiscreteVariable("X", ("0", "1")), DiscreteVariable("Y", ("0", "1"))
    model = BayesianNetwork([x, y], [("X", "Y")])
    model.add_cpd(TabularCPD(x, (), [0.5, 0.5]))
    model.add_cpd(TabularCPD(y, (x,), [[1.0, 0.0], [0.0, 1.0]]))
    return model


def random_markov(seed: int) -> MarkovNetwork:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    variables = [
        DiscreteVariable(f"V{i}", tuple(f"s{j}" for j in range(int(rng.integers(2, 4)))))
        for i in range(n)
    ]
    factors = []
    for _ in range(int(rng.integers(1, 6))):
        k = int(rng.integers(1, min(3, n) + 1))
        scope = [variables[i] for i in rng.choice(n, size=k, replace=False)]
        factors.append(
            DiscreteFactor(scope, rng.uniform(0.1, 3.0, size=tuple(v.cardinality for v in scope)))
        )
    return MarkovNetwork(variables, factors)


def brute_conditional(table: np.ndarray, variables, name: str, state: dict[str, str]) -> np.ndarray:
    """P(name | everything else) from the full (unnormalised) joint table."""
    index = []
    for v in variables:
        index.append(slice(None) if v.name == name else v.states.index(state[v.name]))
    column = table[tuple(index)]
    return column / column.sum()


def random_state(model, rng, evidence) -> dict[str, str]:
    return {
        v.name: evidence.get(v.name, v.states[int(rng.integers(v.cardinality))])
        for v in model.variables
    }


# ---------------------------------------------------------------------------
# G1: the full conditional is the brute-force conditional, and it is local
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_bayesian_full_conditionals_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    model = random_network(seed, n_vars=(1, 5), cards=(2, 3), edge_prob=0.5)
    names = [v.name for v in model.variables]
    evidence = {n: model.variable(n).states[0] for n in names[1:] if rng.random() < 0.3}
    sampler = GibbsSampler(model, evidence, seed=seed)
    table = joint_table(model)
    for _ in range(5):
        state = random_state(model, rng, evidence)
        if table[tuple(model.variable(n).states.index(state[n]) for n in names)] == 0:
            continue
        for name in names:
            if name in evidence:
                continue
            column = brute_conditional(table, model.variables, name, state)
            assert sampler.full_conditional(name, state) == pytest.approx(column, abs=1e-12)


@pytest.mark.parametrize("seed", range(30))
def test_markov_full_conditionals_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    model = random_markov(seed)
    sampler = GibbsSampler(model, seed=seed)
    table = gibbs_table(model.variables, model.factors)
    for _ in range(5):
        state = random_state(model, rng, {})
        for v in model.variables:
            column = brute_conditional(table, model.variables, v.name, state)
            assert sampler.full_conditional(v.name, state) == pytest.approx(column, abs=1e-12)


def test_misconception_cycle_conditional_by_hand():
    """P(A | B=1, D=0) ∝ φ(A, B=1) φ(D=0, A) = (5·100, 10·1) for A = 0, 1."""
    sampler = GibbsSampler(MarkovNetwork(list(STUDENTS), misconception_factors()), seed=0)
    conditional = sampler.full_conditional("A", {"B": "1", "C": "0", "D": "0"})
    assert conditional == pytest.approx([500 / 510, 10 / 510], abs=1e-15)


def bayesian_blanket(model: BayesianNetwork, name: str) -> set[str]:
    children = {c for c in (v.name for v in model.variables) if name in model.parents(c)}
    co_parents = set().union(*(model.parents(c) for c in children)) if children else set()
    return (set(model.parents(name)) | children | co_parents) - {name}


@pytest.mark.parametrize("seed", range(30))
def test_the_conditional_depends_only_on_the_markov_blanket(seed):
    rng = np.random.default_rng(seed)
    model = random_network(seed + 100, n_vars=(3, 6), cards=(2, 3), edge_prob=0.4)
    sampler = GibbsSampler(model, seed=seed)
    for v in model.variables:
        blanket = bayesian_blanket(model, v.name)
        assert sampler.markov_blanket(v.name) == blanket
        outside = {u.name for u in model.variables} - blanket - {v.name}
        if outside:
            assert model.graph.d_separated({v.name}, outside, blanket)
        state = random_state(model, rng, {})
        before = sampler.full_conditional(v.name, state)
        changed = dict(state)
        for name in outside:
            states = model.variable(name).states
            changed[name] = states[int(rng.integers(len(states)))]
        assert sampler.full_conditional(v.name, changed) == pytest.approx(before, abs=0)


def test_extreme_conditionals_do_not_underflow():
    """Both states' factor products underflow (1e-400 and 1e-360), so a naive product gives
    0/0. As a softmax of logs the conditional is exact: P(X=0 | ...) = 1e-40 / (1 + 1e-40)."""
    x = DiscreteVariable("X", ("0", "1"))
    kids = [DiscreteVariable(f"C{i}", ("0", "1")) for i in range(2)]
    model = BayesianNetwork([x, *kids], [("X", k.name) for k in kids])
    model.add_cpd(TabularCPD(x, (), [0.5, 0.5]))
    for k in kids:
        model.add_cpd(TabularCPD(k, (x,), [[1 - 1e-200, 1 - 1e-180], [1e-200, 1e-180]]))
    conditional = GibbsSampler(model, seed=0).full_conditional("X", {"C0": "1", "C1": "1"})
    assert conditional[0] == pytest.approx(1e-40, rel=1e-9)
    assert conditional[1] == pytest.approx(1.0, rel=1e-15)
    naive = np.array([0.5 * 1e-200 * 1e-200, 0.5 * 1e-180 * 1e-180])
    assert naive.sum() == 0.0  # the naive computation has nothing left to normalise


# ---------------------------------------------------------------------------
# G4: chains, streams, evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scan", ["systematic", "random"])
def test_runs_are_reproducible_and_burn_in_and_thinning_are_slices(scan):
    model = late_network()
    evidence = {"Late": "yes"}
    plain = GibbsSampler(model, evidence, scan=scan, seed=7).run(60, initial=None)
    again = GibbsSampler(model, evidence, scan=scan, seed=7).run(60)
    assert np.array_equal(plain.states, again.states)
    long = GibbsSampler(model, evidence, scan=scan, seed=7).run(5 + 3 * 10)
    sliced = GibbsSampler(model, evidence, scan=scan, seed=7).run(10, burn_in=5, thin=3)
    assert np.array_equal(sliced.states, long.states[5 + 2 :: 3][:10])
    assert not np.array_equal(
        GibbsSampler(model, evidence, scan=scan, seed=8).run(60).states, plain.states
    )


def test_chain_layout_and_estimates():
    chain = GibbsSampler(late_network(), {"Late": "yes"}, seed=1).run(2000)
    assert isinstance(chain, Chain)
    assert [v.name for v in chain.variables] == ["Rain", "Accident", "Traffic", "Umbrella"]
    assert chain.states.shape == (2000, 4) and chain.acceptance_rate == 1.0
    with pytest.raises(ValueError):
        chain.states[0, 0] = 1
    rain = chain.indicator("Rain", "yes")
    assert set(np.unique(rain)) <= {0.0, 1.0}
    estimate = chain.estimate(["Rain"])
    assert estimate.value({"Rain": "yes"}) == pytest.approx(rain.mean())
    joint = chain.estimate(["Rain", "Traffic"])
    assert joint.values.sum() == pytest.approx(1.0) and joint.names == ("Rain", "Traffic")
    with pytest.raises(UnknownNodeError):
        chain.indicator("Late", "yes")  # observed, so not in the chain
    with pytest.raises(UnknownStateError):
        chain.indicator("Rain", "maybe")


def test_random_scan_changes_at_most_one_variable_per_sample():
    chain = GibbsSampler(late_network(), scan="random", seed=2).run(500)
    changes = (np.diff(chain.states, axis=0) != 0).sum(axis=1)
    assert changes.max() <= 1


def test_random_scan_updates_every_variable():
    """Each variable changes value at some point; with 5 variables, about 1/5 of steps pick each."""
    chain = GibbsSampler(late_network(), scan="random", seed=2).run(5000)
    changed = (np.diff(chain.states, axis=0) != 0).sum(axis=0)
    assert (changed > 0).all()
    # Changes are spread over the variables, not concentrated on one.
    assert changed.max() < 0.8 * changed.sum()


def test_estimates_are_roughly_right():
    """A sanity check only; M6.2 sets the exact CLT tolerances. P(Rain=yes | Late=yes) = 29/53."""
    chain = GibbsSampler(late_network(), {"Late": "yes"}, seed=3).run(20_000, burn_in=100)
    exact = (
        VariableElimination(late_network()).query(["Rain"], {"Late": "yes"}).value({"Rain": "yes"})
    )
    assert exact == pytest.approx(29 / 53)
    assert chain.estimate(["Rain"]).value({"Rain": "yes"}) == pytest.approx(exact, abs=0.03)


# ---------------------------------------------------------------------------
# G5: determinism and initialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", ["0", "1"])
def test_f3_a_deterministic_copy_never_moves(start):
    chain = GibbsSampler(copy_network(), seed=0).run(500, initial={"X": start, "Y": start})
    assert (chain.states == int(start)).all()
    assert chain.estimate(["X"]).value({"X": "1"}) == float(
        start
    )  # confidently wrong: truth is 1/2


def test_initialisation_finds_a_possible_state():
    """Forward sampling with the evidence clamped; Y = 1 forces X = 1 under the copy model."""
    chain = GibbsSampler(copy_network(), {"Y": "1"}, seed=0).run(10)
    assert (chain.states[:, 0] == 1).all()


def test_impossible_evidence_raises():
    model = f1()
    impossible = BayesianNetwork(model.variables, model.edges())
    impossible.add_cpd(TabularCPD(model.variable("X"), (), [1.0, 0.0]))
    impossible.add_cpd(model.cpds["Y"])
    with pytest.raises(ZeroProbabilityEvidenceError, match="1000"):
        GibbsSampler(impossible, {"X": "1"}, seed=0).run(5)


def test_impossible_initial_state_raises():
    with pytest.raises(ZeroProbabilityEvidenceError, match="initial"):
        GibbsSampler(copy_network(), seed=0).run(5, initial={"X": "0", "Y": "1"})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 5, "burn_in": -1}, "burn_in"),
        ({"n_samples": 5, "thin": 0}, "thin"),
        ({"n_samples": 5, "initial": {"X": "0"}}, "initial"),
    ],
)
def test_run_validation(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        GibbsSampler(f1(), seed=0).run(**kwargs)


def test_constructor_validation():
    with pytest.raises(ValidationError, match="scan"):
        GibbsSampler(f1(), scan="diagonal")
    with pytest.raises(ValidationError, match="BayesianNetwork or MarkovNetwork"):
        GibbsSampler("model")
    with pytest.raises(UnknownNodeError):
        GibbsSampler(f1(), {"Z": "0"})
    with pytest.raises(ValidationError, match="every variable is observed"):
        GibbsSampler(f1(), {"X": "0", "Y": "1"}).run(3)
    with pytest.raises(ValidationError, match="observed"):
        GibbsSampler(f1(), {"X": "0"}).full_conditional("X", {"Y": "1"})
