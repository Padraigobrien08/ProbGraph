"""M6.2: exact Gibbs kernels: stationarity, F1–F3, and exact CLT tolerances (G2, G3; mcmc.md)."""

import math
from fractions import Fraction

import numpy as np
import pytest
from kernels import ExactChain, asymptotic_variance, second_eigenvalue
from support import STUDENTS, late_network, misconception_factors, random_network
from test_gibbs import copy_network, f1, random_markov

from probgraph import BayesianNetwork, DiscreteVariable, MarkovNetwork, TabularCPD
from probgraph.mcmc import GibbsSampler

Z = 5.0


def near_copy(eps: float) -> BayesianNetwork:
    """Fixture F2: X uniform, P(Y = X) = 1 - eps."""
    x, y = DiscreteVariable("X", ("0", "1")), DiscreteVariable("Y", ("0", "1"))
    model = BayesianNetwork([x, y], [("X", "Y")])
    model.add_cpd(TabularCPD(x, (), [0.5, 0.5]))
    model.add_cpd(TabularCPD(y, (x,), [[1 - eps, eps], [eps, 1 - eps]]))
    return model


# ---------------------------------------------------------------------------
# G2: the exact kernels leave the posterior invariant
# ---------------------------------------------------------------------------


def test_f1_systematic_kernel_is_stationary_in_exact_fractions():
    """Update X then Y, with every probability a fraction (mcmc.md §2)."""
    px = {0: Fraction(7, 10), 1: Fraction(3, 10)}
    py = {
        (0, 0): Fraction(4, 5),
        (0, 1): Fraction(1, 5),
        (1, 0): Fraction(1, 10),
        (1, 1): Fraction(9, 10),
    }
    joint = {(x, y): px[x] * py[(x, y)] for x in (0, 1) for y in (0, 1)}
    assert joint == {
        (0, 0): Fraction(14, 25),
        (0, 1): Fraction(7, 50),
        (1, 0): Fraction(3, 100),
        (1, 1): Fraction(27, 100),
    }

    def x_given_y(x, y):
        return joint[(x, y)] / (joint[(0, y)] + joint[(1, y)])

    states = list(joint)
    kernel = {(s, t): x_given_y(t[0], s[1]) * py[(t[0], t[1])] for s in states for t in states}
    for t in states:
        assert sum(joint[s] * kernel[(s, t)] for s in states) == joint[t]
    assert x_given_y(1, 1) == Fraction(27, 41)


@pytest.mark.parametrize("scan", ["systematic", "random"])
@pytest.mark.parametrize("seed", range(25))
def test_bayesian_kernels_are_stationary(seed, scan):
    rng = np.random.default_rng(seed)
    model = random_network(seed, n_vars=(2, 4), cards=(2, 3), edge_prob=0.5)
    names = [v.name for v in model.variables]
    evidence = {n: model.variable(n).states[0] for n in names[1:] if rng.random() < 0.3}
    exact = ExactChain(model, evidence)
    kernel = exact.systematic() if scan == "systematic" else exact.random_scan()
    assert kernel.sum(axis=1) == pytest.approx(np.ones(exact.size), abs=1e-12)
    assert exact.pi @ kernel == pytest.approx(exact.pi, abs=1e-14)
    if scan == "random":  # a mixture of reversible kernels is reversible
        flow = exact.pi[:, np.newaxis] * kernel
        assert flow == pytest.approx(flow.T, abs=1e-14)


@pytest.mark.parametrize("seed", range(15))
def test_markov_kernels_are_stationary(seed):
    exact = ExactChain(random_markov(seed))
    for kernel in (exact.systematic(), exact.random_scan()):
        assert exact.pi @ kernel == pytest.approx(exact.pi, abs=1e-14)


def test_the_systematic_sweep_is_not_reversible_in_general():
    exact = ExactChain(f1())
    flow = exact.pi[:, np.newaxis] * exact.systematic()
    assert np.abs(flow - flow.T).max() > 1e-3


# ---------------------------------------------------------------------------
# F1–F3: the spectra
# ---------------------------------------------------------------------------


def test_f1_second_eigenvalue():
    assert second_eigenvalue(ExactChain(f1()).systematic()) == pytest.approx(1029 / 2419, abs=1e-13)


@pytest.mark.parametrize("eps", [0.1, 0.01, 0.001])
def test_f2_second_eigenvalue_is_one_minus_two_eps_squared(eps):
    exact = ExactChain(near_copy(eps))
    lam = second_eigenvalue(exact.systematic())
    assert lam == pytest.approx((1 - 2 * eps) ** 2, abs=1e-12)
    tau = asymptotic_variance(exact.systematic(), exact.pi, exact.indicator("Y", "1")) / 0.25
    assert tau == pytest.approx((1 + lam) / (1 - lam), rel=1e-9)  # ≈ 1/(2 eps)


def test_f3_the_deterministic_copy_has_the_identity_kernel():
    exact = ExactChain(copy_network())
    assert exact.states == [(0, 0), (1, 1)]
    assert exact.systematic() == pytest.approx(np.eye(2))
    assert exact.random_scan() == pytest.approx(np.eye(2))  # reducible: two closed classes


# ---------------------------------------------------------------------------
# G3: estimates within exact CLT tolerances
# ---------------------------------------------------------------------------


def check_estimate(model, evidence, scan, seed, n=4000, burn_in=200):
    exact = ExactChain(model, evidence)
    kernel = exact.systematic() if scan == "systematic" else exact.random_scan()
    chain = GibbsSampler(model, evidence, scan=scan, seed=seed).run(n, burn_in=burn_in)
    checked = 0
    for v in exact.free:
        for state in v.states:
            f = exact.indicator(v.name, state)
            truth = float(exact.pi @ f)
            sigma2 = asymptotic_variance(kernel, exact.pi, f)
            if sigma2 <= 1e-12:
                continue
            estimate = float(chain.indicator(v.name, state).mean())
            assert abs(estimate - truth) <= Z * math.sqrt(sigma2 / n), (
                v.name,
                state,
                estimate,
                truth,
            )
            checked += 1
    return checked


@pytest.mark.parametrize("scan", ["systematic", "random"])
@pytest.mark.parametrize("seed", range(15))
def test_bayesian_estimates_within_exact_clt_bounds(seed, scan):
    rng = np.random.default_rng(seed + 50)
    model = random_network(seed + 50, n_vars=(2, 4), cards=(2, 3), edge_prob=0.5)
    names = [v.name for v in model.variables]
    evidence = {n: model.variable(n).states[0] for n in names[1:] if rng.random() < 0.3}
    if len(evidence) == len(names):
        evidence.pop(names[-1])
    assert check_estimate(model, evidence, scan, seed) > 0


@pytest.mark.parametrize("scan", ["systematic", "random"])
def test_late_network_and_misconception_within_exact_clt_bounds(scan):
    assert check_estimate(late_network(), {"Late": "yes"}, scan, seed=1) > 0
    misconception = MarkovNetwork(list(STUDENTS), misconception_factors())
    assert check_estimate(misconception, {}, scan, seed=2, n=20_000) > 0


def test_correlation_inflates_the_variance_by_tau():
    """Across 200 replicate chains on F1, Var(μ̂) ≈ σ²_asym / n, which is τ ≈ 2.48 times the
    naive Var(f) / n. The sample variance of 200 means has relative sd √(2/199) ≈ 0.1, so
    ±40% is a 4-sd band."""
    exact = ExactChain(f1())
    f = exact.indicator("Y", "1")
    sigma2 = asymptotic_variance(exact.systematic(), exact.pi, f)
    naive = float(exact.pi @ f - (exact.pi @ f) ** 2)
    assert sigma2 / naive == pytest.approx(1724 / 695, rel=1e-9)  # τ for F1
    n, replicates = 1000, 200
    means = [
        GibbsSampler(f1(), seed=r).run(n, burn_in=50).indicator("Y", "1").mean()
        for r in range(replicates)
    ]
    observed = float(np.var(means, ddof=1))
    assert 0.6 * sigma2 / n < observed < 1.4 * sigma2 / n
    assert observed > 1.5 * naive / n  # independent-sample error bars would be far too narrow
