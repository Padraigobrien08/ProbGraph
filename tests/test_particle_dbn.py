"""M6.6: particle filtering on DBNs, its 1/√N accuracy, and entanglement (P3, P6)."""

import math

import numpy as np
import pytest
from test_dbn import factorial_hmm, largest_clique, observe_y
from test_forward_backward import random_hmm, random_observations

from probgraph import VariableElimination
from probgraph.temporal import ForwardBackward, ParticleFilter


def rms_error(model, observations, n, replicates, offset=0) -> float:
    exact = ForwardBackward(model, observations).filtered
    errors = [
        float(
            np.mean(
                (
                    ParticleFilter.for_hmm(model, observations, n, seed=offset + r).filtered("X")
                    - exact
                )
                ** 2
            )
        )
        for r in range(replicates)
    ]
    return math.sqrt(float(np.mean(errors)))


# ---------------------------------------------------------------------------
# P3: the error shrinks like 1/sqrt(N)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_filtered_error_shrinks_like_one_over_root_n(seed):
    model = random_hmm(seed + 500, k=3, m=3)
    observations = random_observations(model, seed, 15, missing=0.1)
    errors = [
        rms_error(model, observations, n, replicates=40, offset=1000 * n) for n in (100, 400, 1600)
    ]
    # Quadrupling N should halve the RMS error: allow [1.5, 2.7] for Monte Carlo noise.
    assert 1.5 < errors[0] / errors[1] < 2.7
    assert 1.5 < errors[1] / errors[2] < 2.7


def test_large_n_is_close_to_forward_backward():
    model = random_hmm(42, k=3, m=3)
    observations = random_observations(model, 1, 20, missing=0.1)
    estimate = ParticleFilter.for_hmm(model, observations, 50_000, seed=3).filtered("X")
    assert estimate == pytest.approx(ForwardBackward(model, observations).filtered, abs=0.02)


# ---------------------------------------------------------------------------
# P6: entangled DBNs: accurate joints, at a cost linear in the chains
# ---------------------------------------------------------------------------


def exact_filtered_joint(dbn, evidence, t: int, names) -> np.ndarray:
    """P(names at slice t | evidence up to t) by variable elimination on the unrolled network."""
    network = dbn.unroll(t)
    seen = {f"{k}_{s + 1}": v for s in range(t) for k, v in evidence[s].items()}
    return VariableElimination(network).query([f"{n}_{t}" for n in names], seen).values


@pytest.mark.parametrize("n_chains", [2, 3])
def test_factorial_hmm_marginals_and_joints_match_exact_filtering(n_chains):
    dbn = factorial_hmm(n_chains, seed=7)
    length = 6
    evidence = [{"Y": y} for y in observe_y(length).values()]
    pf = ParticleFilter(dbn, evidence, 30_000, seed=1)
    for t in (1, 3, length):
        for k in range(n_chains):
            exact = exact_filtered_joint(dbn, evidence, t, [f"C{k}"])
            assert pf.filtered(f"C{k}")[t - 1] == pytest.approx(exact, abs=0.02)
        joint = pf.filtered_joint(["C0", "C1"])[t - 1]
        assert joint == pytest.approx(
            exact_filtered_joint(dbn, evidence, t, ["C0", "C1"]), abs=0.02
        )


def test_particles_represent_the_entanglement():
    """The particle joint is far from the product of its marginals, as the exact belief is."""
    dbn = factorial_hmm(2, seed=3)
    evidence = [{"Y": y} for y in observe_y(5).values()]
    joint = ParticleFilter(dbn, evidence, 30_000, seed=2).filtered_joint(["C0", "C1"])[-1]
    exact = exact_filtered_joint(dbn, evidence, 5, ["C0", "C1"])
    exact_gap = np.abs(exact - np.outer(exact.sum(1), exact.sum(0))).max()
    particle_gap = np.abs(joint - np.outer(joint.sum(1), joint.sum(0))).max()
    assert exact_gap > 1e-2
    assert particle_gap == pytest.approx(exact_gap, abs=0.015)


def test_ten_chains_exact_clique_grows_while_particles_stay_accurate():
    """Exact filtering needs a clique over all 10 chains (2^11 cells); particles do not."""
    dbn = factorial_hmm(10, seed=5)
    assert largest_clique(dbn.unroll(4), observe_y(4)) >= 11
    evidence = [{"Y": y} for y in observe_y(4).values()]
    pf = ParticleFilter(dbn, evidence, 20_000, seed=4)
    for k in (0, 5, 9):
        exact = exact_filtered_joint(dbn, evidence, 4, [f"C{k}"])
        assert pf.filtered(f"C{k}")[-1] == pytest.approx(exact, abs=0.03)


def test_filtered_joint_layout():
    dbn = factorial_hmm(2, seed=1)
    pf = ParticleFilter(dbn, [{"Y": "0"}, {}], 200, seed=0)
    joint = pf.filtered_joint(["C1", "Y"])
    assert joint.shape == (2, 2, 3)
    assert joint.sum(axis=(1, 2)) == pytest.approx([1.0, 1.0])
    assert joint[:, :, :].sum(axis=1) == pytest.approx(pf.filtered("Y"))
    assert pf.filtered("Y")[0] == pytest.approx([1.0, 0.0, 0.0])
