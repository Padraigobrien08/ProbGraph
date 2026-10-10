"""M6.7: blocked Gibbs sampling (B1, B2; blocked_gibbs.md)."""

import math

import numpy as np
import pytest
from kernels import ExactChain, asymptotic_variance, second_eigenvalue
from support import gibbs_table, joint_table, random_network
from test_gibbs import copy_network, random_markov, random_state
from test_mcmc_kernels import near_copy

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.mcmc import (
    BlockedGibbsSampler,
    GibbsSampler,
    integrated_autocorrelation_time,
    split_r_hat,
)

Z = 5.0


def brute_block(table, variables, block, state) -> np.ndarray:
    """P(x_B | x_-B) from the full joint table, with axes in ``block`` order."""
    index = [slice(None) if v.name in block else v.states.index(state[v.name]) for v in variables]
    sub = table[tuple(index)]  # axes: block variables in model order
    in_model = [v.name for v in variables if v.name in block]
    sub = np.transpose(sub, [in_model.index(n) for n in block])
    return sub / sub.sum()


# ---------------------------------------------------------------------------
# B1: exact block conditionals and kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_block_conditionals_match_the_joint_table(seed):
    rng = np.random.default_rng(seed)
    model = random_network(seed, n_vars=(2, 5), cards=(2, 3), edge_prob=0.5)
    names = [v.name for v in model.variables]
    size = int(rng.integers(2, len(names) + 1))
    block = [str(n) for n in rng.choice(names, size=size, replace=False)]
    sampler = BlockedGibbsSampler(model, [block], seed=seed)
    table = joint_table(model)
    state = random_state(model, rng, {})
    rest = {n: s for n, s in state.items() if n not in block}
    if table[tuple(model.variable(n).states.index(state[n]) for n in names)] == 0:
        return
    expected = brute_block(table, model.variables, block, state)
    assert sampler.block_conditional(block, rest) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_markov_block_conditionals(seed):
    rng = np.random.default_rng(seed)
    model = random_markov(seed)
    names = [v.name for v in model.variables]
    block = names[: max(2, len(names) - 1)]
    table = gibbs_table(model.variables, model.factors)
    state = random_state(model, rng, {})
    expected = brute_block(table, model.variables, block, state)
    got = BlockedGibbsSampler(model, [block], seed=0).block_conditional(
        block, {n: s for n, s in state.items() if n not in block}
    )
    assert got == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_block_kernels_are_stationary_and_reversible(seed):
    rng = np.random.default_rng(seed + 20)
    model = random_network(seed + 20, n_vars=(3, 4), cards=(2, 2), edge_prob=0.5)
    exact = ExactChain(model)
    columns = tuple(sorted(rng.choice(len(exact.free), size=2, replace=False).tolist()))
    kernel = exact.block_kernel(columns)
    assert exact.pi @ kernel == pytest.approx(exact.pi, abs=1e-14)
    flow = exact.pi[:, np.newaxis] * kernel
    assert flow == pytest.approx(flow.T, abs=1e-15)


@pytest.mark.parametrize("seed", range(10))
def test_estimates_within_exact_clt_bounds(seed):
    """The sweep is the block kernel then the remaining singletons, in model order."""
    model = random_network(seed + 40, n_vars=(3, 3), cards=(2, 3), edge_prob=0.6)
    exact = ExactChain(model)
    names = [v.name for v in exact.free]
    kernel = exact.block_kernel((0, 1)) @ exact.site_kernel(2)
    chain = BlockedGibbsSampler(model, [names[:2]], seed=seed).run(5000, burn_in=100)
    for v in exact.free:
        for state in v.states:
            f = exact.indicator(v.name, state)
            sigma2 = asymptotic_variance(kernel, exact.pi, f)
            if sigma2 > 1e-12:
                estimate = chain.indicator(v.name, state).mean()
                assert abs(estimate - exact.pi @ f) <= Z * math.sqrt(sigma2 / 5000)


# ---------------------------------------------------------------------------
# B2: blocking cures F2 and F3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.005, 0.001])
def test_f2_blocking_makes_samples_independent(eps):
    exact = ExactChain(near_copy(eps))
    assert second_eigenvalue(exact.block_kernel((0, 1))) == pytest.approx(0.0, abs=1e-12)
    assert second_eigenvalue(exact.systematic()) == pytest.approx((1 - 2 * eps) ** 2, abs=1e-12)
    chain = BlockedGibbsSampler(near_copy(eps), [["X", "Y"]], seed=1).run(20_000)
    assert integrated_autocorrelation_time(chain.indicator("Y", "1")) == pytest.approx(1.0, abs=0.1)


def test_f2_blocked_chains_agree_where_single_site_chains_do_not():
    model = near_copy(0.005)
    starts = ["0", "0", "1", "1"]
    single = [
        GibbsSampler(model, seed=s).run(500, initial={"X": x, "Y": x}).indicator("Y", "1")
        for s, x in enumerate(starts)
    ]
    blocked = [
        BlockedGibbsSampler(model, [["X", "Y"]], seed=s)
        .run(500, initial={"X": x, "Y": x})
        .indicator("Y", "1")
        for s, x in enumerate(starts)
    ]
    assert split_r_hat(single) > 1.1
    assert split_r_hat(blocked) < 1.01


def test_f3_blocking_restores_irreducibility():
    n = 4000
    chain = BlockedGibbsSampler(copy_network(), [["X", "Y"]], seed=2).run(
        n, initial={"X": "0", "Y": "0"}
    )
    assert set(map(tuple, chain.states.tolist())) == {(0, 0), (1, 1)}
    estimate = chain.indicator("X", "1").mean()
    assert abs(estimate - 0.5) <= Z * math.sqrt(0.25 / n)  # independent draws (λ2 = 0)


# ---------------------------------------------------------------------------
# Behaviour and validation
# ---------------------------------------------------------------------------


def test_unblocked_variables_are_updated_singly_and_blocks_jointly():
    model = random_network(3, n_vars=(4, 4), cards=(2, 2), edge_prob=0.5)
    names = [v.name for v in model.variables]
    sampler = BlockedGibbsSampler(model, [[names[2], names[0]]], seed=0)
    assert sampler._units == [(0, 2), (1,), (3,)]
    chain = sampler.run(200)
    assert chain.states.shape == (200, 4)


def test_without_blocks_it_is_plain_gibbs():
    model = random_network(5, n_vars=(4, 4), cards=(2, 3), edge_prob=0.5)
    a = BlockedGibbsSampler(model, [], seed=9).run(300)
    b = GibbsSampler(model, seed=9).run(300)
    assert np.array_equal(a.states, b.states)


@pytest.mark.parametrize(
    ("blocks", "error", "match"),
    [
        ([["X", "Z"]], UnknownNodeError, "Z"),
        ([["X"], ["X", "Y"]], ValidationError, "more than one block"),
        ([[]], ValidationError, "non-empty"),
        ("XY", ValidationError, "sequence"),
    ],
)
def test_validation(blocks, error, match):
    with pytest.raises(error, match=match):
        BlockedGibbsSampler(copy_network(), blocks)


def test_observed_variables_cannot_be_blocked():
    with pytest.raises(ValidationError, match="observed"):
        BlockedGibbsSampler(copy_network(), [["X", "Y"]], evidence={"Y": "1"})
