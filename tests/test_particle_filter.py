"""M6.5: the particle filter (P1, P2, P4, P5; F5; particle_filtering.md)."""

import math
from fractions import Fraction

import numpy as np
import pytest
from test_forward_backward import FIVE_DAYS, NONE, U, umbrella_world

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import (
    UnknownNodeError,
    UnknownStateError,
    ValidationError,
    ZeroProbabilityEvidenceError,
)
from probgraph.temporal import DynamicBayesianNetwork, ForwardBackward, ParticleFilter, previous
from probgraph.temporal.particle import resample

Z = 5.0
TWO_DAYS = float(Fraction(703, 2000))
FIVE = float(Fraction(68607401, 2_000_000_000))


def likelihoods(observations, n_particles, resampling, replicates, offset=0) -> np.ndarray:
    return np.array(
        [
            math.exp(
                ParticleFilter.for_hmm(
                    umbrella_world(),
                    observations,
                    n_particles,
                    resampling=resampling,
                    seed=offset + r,
                ).log_likelihood
            )
            for r in range(replicates)
        ]
    )


# ---------------------------------------------------------------------------
# P1: E[P̂] = P exactly, for every N and every scheme (fixture F5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resampling", ["systematic", "multinomial", "adaptive", "none"])
@pytest.mark.parametrize(("observations", "exact"), [([U, U], TWO_DAYS), (FIVE_DAYS, FIVE)])
def test_the_likelihood_estimate_is_unbiased(observations, exact, resampling):
    values = likelihoods(observations, 10, resampling, replicates=3000)
    error = Z * values.std(ddof=1) / math.sqrt(len(values))
    assert abs(values.mean() - exact) <= error


def test_unbiased_even_with_one_particle():
    values = likelihoods(FIVE_DAYS, 1, "systematic", replicates=20_000)
    assert abs(values.mean() - FIVE) <= Z * values.std(ddof=1) / math.sqrt(len(values))
    assert values.std() > values.mean()  # unbiased, but one particle is very noisy


# ---------------------------------------------------------------------------
# P2: Jensen: log P̂ is biased low, by about 1/N
# ---------------------------------------------------------------------------


def test_log_likelihood_is_biased_low_by_about_one_over_n():
    exact = math.log(FIVE)
    gaps = []
    for n in (4, 16, 64):
        logs = np.log(likelihoods(FIVE_DAYS, n, "systematic", replicates=3000, offset=10_000 * n))
        gaps.append(exact - logs.mean())
    assert all(g > 0 for g in gaps)
    assert gaps[0] > gaps[1] > gaps[2]
    assert 2.0 < gaps[0] / gaps[1] < 8.0 and 2.0 < gaps[1] / gaps[2] < 8.0  # about 4 each time


# ---------------------------------------------------------------------------
# P4: degeneracy without resampling
# ---------------------------------------------------------------------------


def test_without_resampling_the_weights_degenerate():
    observations = umbrella_world().sample(200, seed=3)[1]
    none = ParticleFilter.for_hmm(umbrella_world(), observations, 500, resampling="none", seed=1)
    systematic = ParticleFilter.for_hmm(umbrella_world(), observations, 500, seed=1)
    assert none.effective_sample_sizes[-1] < 5
    assert systematic.effective_sample_sizes.min() > 150
    adaptive = ParticleFilter.for_hmm(
        umbrella_world(), observations, 500, resampling="adaptive", seed=1
    )
    assert adaptive.effective_sample_sizes.min() >= 150 / 2


# ---------------------------------------------------------------------------
# P5: resampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_systematic_offspring_are_floor_or_ceiling(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 40))
    weights = rng.dirichlet(np.full(n, 0.5))
    counts = np.bincount(resample(weights, rng, "systematic"), minlength=n)
    assert counts.sum() == n
    assert ((counts == np.floor(n * weights)) | (counts == np.ceil(n * weights))).all()


@pytest.mark.parametrize("scheme", ["systematic", "multinomial"])
def test_offspring_are_unbiased(scheme):
    rng = np.random.default_rng(0)
    weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    replicates = 20_000
    counts = np.array(
        [np.bincount(resample(weights, rng, scheme), minlength=5) for _ in range(replicates)]
    )
    expected = 5 * weights
    for i in range(5):
        sd = counts[:, i].std(ddof=1) / math.sqrt(replicates)
        assert abs(counts[:, i].mean() - expected[i]) <= Z * max(sd, 1e-12)
    if scheme == "systematic":
        assert (counts.var(axis=0) <= 5 * weights * (1 - weights) + 1e-12).all()


def test_zero_weight_particles_are_never_resampled():
    rng = np.random.default_rng(1)
    chosen = resample(np.array([0.0, 0.5, 0.0, 0.5]), rng, "multinomial")
    assert set(chosen.tolist()) <= {1, 3}


# ---------------------------------------------------------------------------
# Behaviour, DBNs, and failure modes
# ---------------------------------------------------------------------------


def test_filtered_estimates_are_distributions_and_roughly_right():
    pf = ParticleFilter.for_hmm(umbrella_world(), FIVE_DAYS, 20_000, seed=2)
    estimate = pf.filtered("Weather")
    assert estimate.shape == (5, 2)
    assert estimate.sum(axis=1) == pytest.approx(np.ones(5))
    exact = ForwardBackward(umbrella_world(), FIVE_DAYS).filtered
    assert estimate == pytest.approx(exact, abs=0.02)  # M6.6 sets the 1/sqrt(N) tolerance
    assert pf.filtered("Umbrella")[:, 0] == pytest.approx([1, 1, 0, 1, 1])  # observed: certain


def test_reproducible():
    a = ParticleFilter.for_hmm(umbrella_world(), FIVE_DAYS, 50, seed=4)
    b = ParticleFilter.for_hmm(umbrella_world(), FIVE_DAYS, 50, seed=4)
    assert a.log_likelihood == b.log_likelihood
    assert np.array_equal(a.filtered("Weather"), b.filtered("Weather"))


def test_missing_observations():
    pf = ParticleFilter.for_hmm(umbrella_world(), [U, None, None, U], 5000, seed=5)
    exact = ForwardBackward(umbrella_world(), [U, None, None, U])
    assert math.exp(pf.log_likelihood) == pytest.approx(math.exp(exact.log_likelihood), rel=0.05)


def test_all_particles_dead():
    """Rain forever never emits 'none': every particle gets weight 0 at slice 2."""
    weather = DiscreteVariable("W", ("rain", "dry"))
    umbrella = DiscreteVariable("U", ("umbrella", "none"))
    initial = BayesianNetwork([weather, umbrella], [("W", "U")])
    initial.add_cpd(TabularCPD(weather, (), [1.0, 0.0]))
    emit = TabularCPD(umbrella, (weather,), [[1.0, 0.5], [0.0, 0.5]])
    initial.add_cpd(emit)
    dbn = DynamicBayesianNetwork(
        initial, [TabularCPD(weather, (previous(weather),), [[1.0, 0.5], [0.0, 0.5]]), emit]
    )
    pf = ParticleFilter(dbn, [{"U": "umbrella"}, {"U": "none"}], 50, seed=0)
    assert pf.log_likelihood == -math.inf
    with pytest.raises(ZeroProbabilityEvidenceError, match="slice 2"):
        pf.filtered("W")


def test_validation():
    dbn = DynamicBayesianNetwork.from_hmm(umbrella_world())
    with pytest.raises(ValidationError, match="n_particles"):
        ParticleFilter(dbn, [{}], 0)
    with pytest.raises(ValidationError, match="resampling"):
        ParticleFilter(dbn, [{}], 10, resampling="stratified")
    with pytest.raises(ValidationError, match="at least one slice"):
        ParticleFilter(dbn, [], 10)
    with pytest.raises(ValidationError, match="sequence of mappings"):
        ParticleFilter(dbn, {"Umbrella": "umbrella"}, 10)
    with pytest.raises(UnknownNodeError):
        ParticleFilter(dbn, [{"Snow": "yes"}], 10)
    with pytest.raises(UnknownStateError):
        ParticleFilter(dbn, [{"Umbrella": "parasol"}], 10)
    with pytest.raises(ValidationError, match="for_hmm"):
        ParticleFilter(umbrella_world(), [{}], 10)
    with pytest.raises(UnknownNodeError):
        ParticleFilter(dbn, [{}], 10).filtered("Snow")
    assert NONE == "none"
