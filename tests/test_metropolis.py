"""M6.3: single-site Metropolis–Hastings (H1–H3; F4; mcmc.md §3, §7)."""

import math

import numpy as np
import pytest
from kernels import ExactChain, asymptotic_variance
from support import STUDENTS, late_network, misconception_factors, random_network
from test_gibbs import copy_network, f1, random_markov

from probgraph import MarkovNetwork
from probgraph.mcmc import Chain, MetropolisHastings

Z = 5.0


def random_evidence_for(model, rng):
    names = [v.name for v in model.variables]
    evidence = {n: model.variable(n).states[0] for n in names[1:] if rng.random() < 0.3}
    if len(evidence) == len(names):
        evidence.pop(names[-1])
    return evidence


# ---------------------------------------------------------------------------
# H1: detailed balance and stationarity of the exact kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_site_kernels_satisfy_detailed_balance(seed):
    rng = np.random.default_rng(seed)
    model = random_network(seed, n_vars=(2, 4), cards=(1, 3), edge_prob=0.5)
    exact = ExactChain(model, random_evidence_for(model, rng))
    for j in range(len(exact.free)):
        kernel = exact.metropolis_site_kernel(j)
        assert kernel.sum(axis=1) == pytest.approx(np.ones(exact.size), abs=1e-12)
        flow = exact.pi[:, np.newaxis] * kernel
        assert flow == pytest.approx(flow.T, abs=1e-15)
    for kernel in (exact.systematic(metropolis=True), exact.random_scan(metropolis=True)):
        assert exact.pi @ kernel == pytest.approx(exact.pi, abs=1e-14)


@pytest.mark.parametrize("seed", range(10))
def test_markov_network_kernels_are_stationary(seed):
    exact = ExactChain(random_markov(seed))
    assert exact.pi @ exact.random_scan(metropolis=True) == pytest.approx(exact.pi, abs=1e-14)


# ---------------------------------------------------------------------------
# Estimates within exact CLT bounds, and the acceptance rate (H2)
# ---------------------------------------------------------------------------


def check(model, evidence, scan, seed, n=6000, burn_in=300):
    exact = ExactChain(model, evidence)
    kernel = exact.systematic(True) if scan == "systematic" else exact.random_scan(True)
    chain = MetropolisHastings(model, evidence, scan=scan, seed=seed).run(n, burn_in=burn_in)
    assert isinstance(chain, Chain)
    checked = 0
    for v in exact.free:
        for state in v.states:
            f = exact.indicator(v.name, state)
            sigma2 = asymptotic_variance(kernel, exact.pi, f)
            if sigma2 <= 1e-12:
                continue
            estimate = float(chain.indicator(v.name, state).mean())
            assert abs(estimate - float(exact.pi @ f)) <= Z * math.sqrt(sigma2 / n)
            checked += 1
    return checked, chain, exact


@pytest.mark.parametrize("scan", ["random", "systematic"])
@pytest.mark.parametrize("seed", range(12))
def test_estimates_within_exact_clt_bounds(seed, scan):
    rng = np.random.default_rng(seed + 70)
    model = random_network(seed + 70, n_vars=(2, 4), cards=(2, 3), edge_prob=0.5)
    assert check(model, random_evidence_for(model, rng), scan, seed)[0] > 0


def test_late_network_and_misconception():
    assert check(late_network(), {"Late": "yes"}, "random", seed=1)[0] > 0
    misconception = MarkovNetwork(list(STUDENTS), misconception_factors())
    assert check(misconception, {}, "random", seed=2, n=30_000)[0] > 0


@pytest.mark.parametrize("seed", range(6))
def test_acceptance_rate_matches_its_exact_expectation(seed):
    """Over a long random-scan run the acceptance rate tends to E_π[α]; 50k proposals give a
    standard error below ~0.005 even with strong correlation, so 0.03 is generous."""
    rng = np.random.default_rng(seed + 90)
    model = random_network(seed + 90, n_vars=(2, 4), cards=(2, 3), edge_prob=0.5)
    evidence = random_evidence_for(model, rng)
    exact = ExactChain(model, evidence)
    chain = MetropolisHastings(model, evidence, seed=seed).run(50_000, burn_in=500)
    assert 0 < chain.acceptance_rate <= 1
    assert chain.acceptance_rate == pytest.approx(exact.acceptance_probability(), abs=0.03)


# ---------------------------------------------------------------------------
# H3: Peskun's ordering (fixture F4)
# ---------------------------------------------------------------------------


def test_f4_peskun_on_f1():
    exact = ExactChain(f1())
    gibbs, mh = exact.random_scan(), exact.random_scan(metropolis=True)
    for name, g, m in (
        ("Y", 2.158304892086331, 1.6449858823529417),
        ("X", 1.87368345323741, 1.344),
    ):
        f = exact.indicator(name, "1")
        assert asymptotic_variance(gibbs, exact.pi, f) == pytest.approx(g, rel=1e-12)
        assert asymptotic_variance(mh, exact.pi, f) == pytest.approx(m, rel=1e-12)


@pytest.mark.parametrize("seed", range(25))
def test_peskun_on_random_binary_models(seed):
    rng = np.random.default_rng(seed + 300)
    model = random_network(seed + 300, n_vars=(2, 4), cards=(2, 2), edge_prob=0.5)
    exact = ExactChain(model, random_evidence_for(model, rng))
    gibbs, mh = exact.random_scan(), exact.random_scan(metropolis=True)
    off = ~np.eye(exact.size, dtype=bool)
    assert (mh[off] >= gibbs[off] - 1e-15).all()  # off-diagonal dominance
    functions = [exact.indicator(v.name, "s1") for v in exact.free]
    functions += [rng.normal(size=exact.size) for _ in range(5)]
    for f in functions:
        assert (
            asymptotic_variance(mh, exact.pi, f) <= asymptotic_variance(gibbs, exact.pi, f) + 1e-10
        )


def test_without_two_states_peskun_dominance_can_fail():
    """Three states: MH proposes uniformly, Gibbs moves towards likely states (mcmc.md §7)."""
    found = False
    for seed in range(40):
        model = random_network(seed + 400, n_vars=(1, 2), cards=(3, 3), edge_prob=0.5)
        exact = ExactChain(model)
        off = ~np.eye(exact.size, dtype=bool)
        if (exact.random_scan(True)[off] < exact.random_scan()[off] - 1e-9).any():
            found = True
            break
    assert found


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_reproducible_and_stuck_on_a_deterministic_copy():
    a = MetropolisHastings(late_network(), {"Late": "yes"}, seed=4).run(300)
    b = MetropolisHastings(late_network(), {"Late": "yes"}, seed=4).run(300)
    assert np.array_equal(a.states, b.states)
    stuck = MetropolisHastings(copy_network(), seed=0).run(400, initial={"X": "0", "Y": "0"})
    assert (stuck.states == 0).all() and stuck.acceptance_rate == 0.0


def test_proposals_are_always_a_different_state():
    """With random scan, every accepted move changes exactly one variable."""
    chain = MetropolisHastings(late_network(), seed=5).run(3000)
    changes = (np.diff(chain.states, axis=0) != 0).sum(axis=1)
    assert set(np.unique(changes)) <= {0, 1}
    moved = (changes == 1).mean()
    assert moved == pytest.approx(chain.acceptance_rate, abs=0.01)
