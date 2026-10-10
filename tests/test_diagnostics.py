"""M6.4: MCMC diagnostics (D1–D4; F2, F3, F6; diagnostics.md)."""

import math

import numpy as np
import pytest
from test_gibbs import copy_network, f1
from test_mcmc_kernels import near_copy

from probgraph.exceptions import ValidationError
from probgraph.mcmc import (
    GibbsSampler,
    autocorrelation,
    effective_sample_size,
    integrated_autocorrelation_time,
    monte_carlo_standard_error,
    split_r_hat,
)


def two_state_chain(p_stay: float, n: int, seed: int) -> np.ndarray:
    """A 0/1 chain that keeps its state with probability p_stay, started from stationarity."""
    rng = np.random.default_rng(seed)
    flips = rng.random(n) >= p_stay
    start = int(rng.random() < 0.5)
    return ((start + np.cumsum(flips)) % 2).astype(float)


# ---------------------------------------------------------------------------
# §1: the autocorrelation estimator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_autocorrelation_matches_the_direct_sum(seed):
    x = np.random.default_rng(seed).normal(size=300).cumsum()
    centred = x - x.mean()
    gamma = [float(np.dot(centred[: len(x) - k], centred[k:])) / len(x) for k in range(40)]
    assert autocorrelation(x, 39) == pytest.approx(np.array(gamma) / gamma[0], abs=1e-12)


def test_constant_series_are_undefined():
    x = np.ones(50)
    assert np.isnan(autocorrelation(x, 5)).all()
    assert math.isnan(integrated_autocorrelation_time(x))
    assert math.isnan(effective_sample_size(x))


# ---------------------------------------------------------------------------
# D1, F6: the two-state chain, exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("p", "tau"), [(0.9, 9.0), (0.7, 7 / 3), (0.5, 1.0), (0.25, 1 / 3)])
def test_two_state_chain_tau_is_p_over_one_minus_p(p, tau):
    x = two_state_chain(p, 200_000, seed=1)
    lam = 2 * p - 1
    assert autocorrelation(x, 4)[1:] == pytest.approx([lam**k for k in range(1, 5)], abs=0.02)
    assert integrated_autocorrelation_time(x) == pytest.approx(tau, rel=0.1)
    assert effective_sample_size(x) == pytest.approx(len(x) / tau, rel=0.1)


def test_independent_draws_have_tau_one():
    x = np.random.default_rng(3).random(100_000)
    assert integrated_autocorrelation_time(x) == pytest.approx(1.0, abs=0.05)
    assert effective_sample_size(x) == pytest.approx(len(x), rel=0.05)


def test_gibbs_on_f1_has_the_exact_tau():
    chain = GibbsSampler(f1(), seed=4).run(60_000, burn_in=100)
    tau = integrated_autocorrelation_time(chain.indicator("Y", "1"))
    assert tau == pytest.approx(1724 / 695, rel=0.1)  # 2.4806 (mcmc.md §6)


def test_gibbs_on_f2_has_the_exact_tau():
    eps = 0.1
    lam = (1 - 2 * eps) ** 2
    chain = GibbsSampler(near_copy(eps), seed=5).run(80_000, burn_in=100)
    tau = integrated_autocorrelation_time(chain.indicator("Y", "1"))
    assert tau == pytest.approx((1 + lam) / (1 - lam), rel=0.15)  # 4.56


# ---------------------------------------------------------------------------
# D3: the MCSE is calibrated
# ---------------------------------------------------------------------------


def test_mcse_is_calibrated():
    """|μ̂ - μ| <= 2 MCSE should hold about 95% of the time; 400 replicates give a standard
    error of about 1.1%, so [0.88, 0.99] is generous."""
    hits = 0
    replicates = 400
    for r in range(replicates):
        x = two_state_chain(0.9, 3000, seed=100 + r)
        hits += abs(x.mean() - 0.5) <= 2 * monte_carlo_standard_error(x)
    assert 0.88 <= hits / replicates <= 0.99


def test_mcse_exceeds_the_naive_error_by_sqrt_tau():
    x = two_state_chain(0.9, 100_000, seed=6)
    naive = math.sqrt(x.var() / len(x))
    assert monte_carlo_standard_error(x) / naive == pytest.approx(3.0, rel=0.1)  # sqrt(9)


# ---------------------------------------------------------------------------
# D4: R̂ and what it can and cannot see
# ---------------------------------------------------------------------------


def test_mixing_chains_have_r_hat_near_one():
    chains = [
        GibbsSampler(f1(), seed=s).run(4000, initial={"X": x, "Y": x}).indicator("Y", "1")
        for s, x in enumerate(["0", "1", "0", "1"])
    ]
    assert split_r_hat(chains) < 1.01


def test_f2_one_chain_often_looks_converged_but_several_chains_do_not():
    """F2 with ε = 0.005 (τ ≈ 100) and 500 sweeps, over 40 replicates: about half the single
    chains pass split-R̂ < 1.1 while their estimate of P(Y=1) = 1/2 is off by ~0.2; four
    chains from opposite modes flag R̂ > 1.1 in about 90% of replicates (diagnostics.md §5)."""
    model = near_copy(0.005)
    fooled, errors, flagged = 0, [], 0
    for r in range(40):
        chains = [
            GibbsSampler(model, seed=1000 * r + s)
            .run(500, initial={"X": x, "Y": x})
            .indicator("Y", "1")
            for s, x in enumerate(["0", "0", "1", "1"])
        ]
        single = split_r_hat([chains[0]])
        if single < 1.1:
            fooled += 1
            errors.append(abs(chains[0].mean() - 0.5))
        flagged += split_r_hat(chains) > 1.1
    assert fooled >= 12  # a single chain often looks converged...
    assert np.mean(errors) > 0.1  # ...while being badly wrong
    assert flagged >= 32  # several chains from different starts usually expose it


def test_f3_stuck_chains_give_infinite_r_hat():
    chains = [
        GibbsSampler(copy_network(), seed=s).run(200, initial={"X": x, "Y": x}).indicator("X", "1")
        for s, x in enumerate(["0", "1"])
    ]
    assert split_r_hat(chains) == math.inf
    assert math.isnan(effective_sample_size(chains[0]))  # each chain alone is constant
    assert math.isnan(split_r_hat([np.zeros(10), np.zeros(10)]))


def test_split_r_hat_catches_drift_within_one_chain():
    drifting = np.concatenate([np.zeros(500), np.ones(500)]) + np.random.default_rng(0).normal(
        0, 0.1, 1000
    )
    assert split_r_hat([drifting]) > 1.5


def test_split_r_hat_by_hand():
    """Two chains of length 4: halves [0,1],[2,3] and [1,1],[3,5]."""
    halves = [np.array(h, dtype=float) for h in ([0, 1], [2, 3], [1, 1], [3, 5])]
    n = 2
    means = np.array([h.mean() for h in halves])
    w = np.mean([h.var(ddof=1) for h in halves])
    b = n * means.var(ddof=1)
    expected = math.sqrt(((n - 1) / n * w + b / n) / w)
    assert split_r_hat([[0, 1, 2, 3], [1, 1, 3, 5]]) == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: autocorrelation([1.0, 2.0, 3.0], 3), "max_lag"),
        (lambda: autocorrelation([[1.0, 2.0]], 0), "one-dimensional"),
        (lambda: integrated_autocorrelation_time([1.0]), "at least 2"),
        (lambda: integrated_autocorrelation_time([1.0, math.nan]), "finite"),
        (lambda: split_r_hat([[1, 2, 3, 4], [1, 2, 3]]), "same length"),
        (lambda: split_r_hat([[1, 2, 3]]), "at least 4"),
        (lambda: split_r_hat([]), "at least one"),
    ],
)
def test_validation(call, match):
    with pytest.raises(ValidationError, match=match):
        call()


def geyer_by_definition(x: np.ndarray) -> tuple[float, bool]:
    """τ̂ from autocorrelation(), by diagnostics.md §2 steps 1–3; also whether step 2 mattered."""
    rho = autocorrelation(x, len(x) - 1)
    pairs = []
    for m in range(len(rho) // 2):
        pair = rho[2 * m] + rho[2 * m + 1]
        if pair <= 0:
            break
        pairs.append(pair)
    monotone = np.minimum.accumulate(pairs)
    return -1 + 2 * float(np.sum(monotone)), bool(np.any(monotone < np.array(pairs)))


def test_the_initial_monotone_sequence_step_is_applied():
    """Short noisy chains: the raw pair sums sometimes rise, and step 2 must cap them."""
    mattered = 0
    for seed in range(60):
        x = two_state_chain(0.8, 120, seed=seed)
        if x.var() == 0:
            continue
        expected, step_two = geyer_by_definition(x)
        assert integrated_autocorrelation_time(x) == pytest.approx(expected, abs=1e-12)
        mattered += step_two
    assert mattered >= 10
