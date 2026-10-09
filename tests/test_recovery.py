"""M4.3: learning recovers what generated the data (spec task 3; L5; likelihood.md §8–10).

Every statistical check uses a bound with a stated failure probability, and all
seeds are fixed, so each test is deterministic.
"""

import itertools
import math

import numpy as np
import pytest
from support import late_network, random_network

from probgraph import AncestralSampler, BayesianNetwork
from probgraph.learning import Dataset, log_likelihood, maximum_likelihood

Z = 5.0


def bernstein(p: float, n: int) -> float:
    """M1's tolerance for a binomial proportion from n trials (failure probability ≤ 7.5e-6)."""
    return Z * math.sqrt(p * (1 - p) / n) + Z**2 / (3 * n)


def sample(model: BayesianNetwork, n: int, seed: int) -> Dataset:
    return Dataset.from_samples(model, AncestralSampler(model, seed=seed).sample(n))


def columns(learned: BayesianNetwork, truth: BayesianNetwork, data: Dataset):
    """Yield (θ̂ column, θ column, N(u)) for every CPD column of the learned model."""
    for name, cpd in learned.cpds.items():
        true_cpd = truth.cpds[name]
        n_u = data.counts(list(cpd.parent_names)).values if cpd.parents else np.array(data.n_rows)
        for config in itertools.product(*(range(p.cardinality) for p in cpd.parents)):
            given = {p.name: p.states[i] for p, i in zip(cpd.parents, config, strict=True)}
            count = int(n_u[config]) if cpd.parents else int(n_u)
            yield cpd.distribution(given), true_cpd.distribution(given), count


# ---------------------------------------------------------------------------
# Fixture F2 (milestone-4.md §4), reproduced exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "max_error", "smallest_count"),
    [(1_000, 0.0481, 32), (10_000, 0.0108, 259), (100_000, 0.0052, 2999)],
)
def test_f2_recovery_of_the_late_network(n, max_error, smallest_count):
    truth = late_network()
    data = sample(truth, n, seed=2026)
    learned = maximum_likelihood(truth, data)
    errors, counts = [], []
    for estimate, true, count in columns(learned, truth, data):
        errors.append(float(np.abs(estimate - true).max()))
        counts.append(count)
    assert max(errors) == pytest.approx(max_error, abs=5e-5)
    assert min(counts) == smallest_count


# ---------------------------------------------------------------------------
# L5: every estimate within the per-column Bernstein bound (§8)
# ---------------------------------------------------------------------------


def assert_within_bounds(truth: BayesianNetwork, data: Dataset) -> int:
    learned = maximum_likelihood(truth, data, unseen="uniform")
    checked = 0
    for estimate, true, count in columns(learned, truth, data):
        if count == 0:
            continue
        for est, p in zip(estimate, true, strict=True):
            assert abs(est - p) <= bernstein(float(p), count), (est, p, count)
            checked += 1
    return checked


def test_late_network_every_entry_within_bounds():
    assert assert_within_bounds(late_network(), sample(late_network(), 100_000, seed=7)) == 20


@pytest.mark.parametrize("seed", range(20))
def test_random_networks_every_entry_within_bounds(seed):
    truth = random_network(seed, n_vars=(2, 6), cards=(2, 3), edge_prob=0.4)
    assert assert_within_bounds(truth, sample(truth, 20_000, seed=seed + 1)) > 0


# ---------------------------------------------------------------------------
# Rate: errors shrink like 1/√N(u)
# ---------------------------------------------------------------------------


def test_standardised_errors_have_the_normal_scale():
    """|θ̂ - θ| · √(N(u) / θ(1 - θ)) is about |N(0, 1)|, whose mean is √(2/π) ≈ 0.798.

    One entry per column (columns use disjoint rows, so they are independent given
    their counts), and only where the normal approximation is good: N(u) ≥ 200 and
    θ in [0.05, 0.95]. Over ~250 columns the mean's standard deviation is about
    0.04, so [0.65, 0.95] is a generous band. Consistency alone would not pass this:
    it pins down the 1/√N rate.
    """
    standardised = []
    for seed in range(40):
        truth = random_network(seed + 100, n_vars=(3, 6), cards=(2, 3), edge_prob=0.4)
        data = sample(truth, 20_000, seed=seed)
        learned = maximum_likelihood(truth, data, unseen="uniform")
        for estimate, true, count in columns(learned, truth, data):
            p = float(true[0])
            if count >= 200 and 0.05 <= p <= 0.95:
                standardised.append(abs(float(estimate[0]) - p) * math.sqrt(count / (p * (1 - p))))
    assert len(standardised) > 150
    assert 0.65 < float(np.mean(standardised)) < 0.95


def test_errors_shrink_as_data_grows():
    truth = late_network()
    worst = []
    for n in (1_000, 10_000, 100_000):
        data = sample(truth, n, seed=3)
        learned = maximum_likelihood(truth, data)
        worst.append(max(float(np.abs(e - t).max()) for e, t, _ in columns(learned, truth, data)))
    assert worst[0] > worst[1] > worst[2]


# ---------------------------------------------------------------------------
# §9: a wrong structure converges to the projection
# ---------------------------------------------------------------------------


def test_misspecified_structure_converges_to_the_true_conditional():
    truth = late_network()
    data = sample(truth, 200_000, seed=11)
    wrong = BayesianNetwork(
        truth.variables, [e for e in truth.edges() if e != ("Accident", "Traffic")]
    )
    learned = maximum_likelihood(wrong, data)
    for rain, target in (("yes", 0.815), ("no", 0.16)):
        n_u = int(data.counts(["Rain"]).value({"Rain": rain}))
        estimate = learned.cpds["Traffic"].probability("yes", {"Rain": rain})
        assert abs(estimate - target) <= bernstein(target, n_u)
    # And it fits worse than the correct structure, as it must (it is a restriction).
    assert log_likelihood(learned, data) < log_likelihood(maximum_likelihood(truth, data), data)


# ---------------------------------------------------------------------------
# §10: Wilks — the MLE overfits by about d/2 in log-likelihood
# ---------------------------------------------------------------------------


def test_wilks_overfitting_matches_the_parameter_count():
    truth = late_network()
    d = truth.n_free_parameters
    assert d == 10
    replicates = 200
    gaps = []
    for r in range(replicates):
        data = sample(truth, 2000, seed=10_000 + r)
        gaps.append(
            2
            * (log_likelihood(maximum_likelihood(truth, data), data) - log_likelihood(truth, data))
        )
    gaps = np.array(gaps)
    assert (gaps >= -1e-9).all()  # the MLE never scores worse than the truth on its own data
    # χ²_d has mean d and variance 2d; the mean of 200 replicates has sd √(2d/200) ≈ 0.32.
    assert abs(gaps.mean() - d) <= Z * math.sqrt(2 * d / replicates)
    assert 2 * d * 0.6 < gaps.var() < 2 * d * 1.6
