"""Milestone 6 walkthrough: approximate inference by sampling.

Run with:  python examples/sampling.py

Shows the ideas of Milestone 6:
  1. Gibbs sampling, with honest error bars (ESS, MCSE) and R-hat over several chains (P24, P25),
  2. a chain that looks converged but is not, caught by R-hat and cured by blocking (P25, P27),
  3. Metropolis-Hastings beating Gibbs on binary variables (Peskun, P23),
  4. particle filtering: an unbiased likelihood, and entangled chains that exact filtering
     cannot afford (P26).
"""

import math

import numpy as np

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.mcmc import (
    BlockedGibbsSampler,
    GibbsSampler,
    MetropolisHastings,
    effective_sample_size,
    integrated_autocorrelation_time,
    monte_carlo_standard_error,
    split_r_hat,
)
from probgraph.temporal import (
    DynamicBayesianNetwork,
    ForwardBackward,
    HiddenMarkovModel,
    ParticleFilter,
    previous,
)

rain, accident, traffic, late, umbrella = (
    DiscreteVariable(name, ("no", "yes"))
    for name in ("Rain", "Accident", "Traffic", "Late", "Umbrella")
)
network = BayesianNetwork(
    [rain, accident, traffic, late, umbrella],
    [("Rain", "Traffic"), ("Accident", "Traffic"), ("Traffic", "Late"), ("Rain", "Umbrella")],
)
t_yes = np.array([[0.10, 0.70], [0.80, 0.95]])
network.add_cpd(TabularCPD(rain, (), [0.7, 0.3]))
network.add_cpd(TabularCPD(accident, (), [0.9, 0.1]))
network.add_cpd(TabularCPD(traffic, (rain, accident), np.stack([1 - t_yes, t_yes])))
network.add_cpd(TabularCPD(late, (traffic,), [[0.9, 0.4], [0.1, 0.6]]))
network.add_cpd(TabularCPD(umbrella, (rain,), [[0.9, 0.15], [0.1, 0.85]]))

# -- 1. Gibbs with error bars ---------------------------------------------------------------

print("1. Gibbs sampling P(Rain | Late = yes) on the late-for-work network")
exact = VariableElimination(network).query(["Rain"], {"Late": "yes"}).value({"Rain": "yes"})
chains = [GibbsSampler(network, {"Late": "yes"}, seed=s).run(5000, burn_in=200) for s in range(4)]
series = [c.indicator("Rain", "yes") for c in chains]
x = series[0]
print(
    f"   exact {exact:.4f} (29/53); one chain {x.mean():.4f} ± {monte_carlo_standard_error(x):.4f}"
)
tau, ess = integrated_autocorrelation_time(x), effective_sample_size(x)
print(f"   tau = {tau:.2f}: 5,000 sweeps are worth {ess:.0f} independent samples")
print(f"   R-hat over 4 chains = {split_r_hat(series):.4f}  (about 1: they agree)")

# -- 2. A chain that lies -----------------------------------------------------------------------

print("\n2. Near-copy: X uniform, Y = X except with probability 0.005")
x_, y_ = DiscreteVariable("X", ("0", "1")), DiscreteVariable("Y", ("0", "1"))
copy = BayesianNetwork([x_, y_], [("X", "Y")])
copy.add_cpd(TabularCPD(x_, (), [0.5, 0.5]))
copy.add_cpd(TabularCPD(y_, (x_,), [[0.995, 0.005], [0.005, 0.995]]))
starts = ["0", "0", "1", "1"]
single = [
    GibbsSampler(copy, seed=s).run(500, initial={"X": v, "Y": v}).indicator("Y", "1")
    for s, v in enumerate(starts)
]
alone = split_r_hat([single[0]])
print(f"   one Gibbs chain: P(Y=1) = {single[0].mean():.3f} (truth 0.5), its own R-hat {alone:.3f}")
print(f"   four chains from both modes: R-hat = {split_r_hat(single):.2f}  -> not converged")
blocked = [
    BlockedGibbsSampler(copy, [["X", "Y"]], seed=s)
    .run(500, initial={"X": v, "Y": v})
    .indicator("Y", "1")
    for s, v in enumerate(starts)
]
pooled = np.concatenate(blocked).mean()
r_hat = split_r_hat(blocked)
print(f"   blocked Gibbs, (X, Y) drawn together: R-hat = {r_hat:.4f}, P(Y=1) = {pooled:.3f}")

# -- 3. Peskun ------------------------------------------------------------------------------

print("\n3. Metropolis-Hastings vs Gibbs on binary variables (random scan)")
for label, sampler in (
    ("Gibbs", GibbsSampler(network, {"Late": "yes"}, scan="random", seed=1)),
    ("Metropolis-Hastings", MetropolisHastings(network, {"Late": "yes"}, seed=1)),
):
    chain = sampler.run(40_000, burn_in=500)
    tau = integrated_autocorrelation_time(chain.indicator("Rain", "yes"))
    print(f"   {label:<20} tau = {tau:6.2f}, acceptance {chain.acceptance_rate:.2f}")
print("   Proposing the other state always (MH) moves more often, so it mixes faster.")

# -- 4. Particle filtering -------------------------------------------------------------------

print("\n4. Particle filtering")
weather = DiscreteVariable("Weather", ("rain", "dry"))
brolly = DiscreteVariable("Umbrella", ("umbrella", "none"))
world = HiddenMarkovModel(
    weather, brolly, [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]], [[0.9, 0.1], [0.2, 0.8]]
)
days = ["umbrella", "umbrella", "none", "umbrella", "umbrella"]
estimates = [
    math.exp(ParticleFilter.for_hmm(world, days, 10, seed=s).log_likelihood) for s in range(4000)
]
print(
    f"   P(days) = {math.exp(ForwardBackward(world, days).log_likelihood):.6f};"
    f" mean of 4,000 ten-particle estimates = {np.mean(estimates):.6f} (unbiased)"
)

chains_ = [DiscreteVariable(f"C{k}", ("0", "1")) for k in range(10)]
signal = DiscreteVariable("Signal", ("low", "high"))
initial = BayesianNetwork([*chains_, signal], [(c.name, "Signal") for c in chains_])
for c in chains_:
    initial.add_cpd(TabularCPD(c, (), [0.5, 0.5]))
# P(high | chains) rises with the number of chains that are on.
on = np.indices((2,) * 10).sum(axis=0) / 10
high = 0.05 + 0.9 * on
emission = TabularCPD(signal, tuple(chains_), np.stack([1 - high, high]))
initial.add_cpd(emission)
stay = [[0.9, 0.2], [0.1, 0.8]]
dbn = DynamicBayesianNetwork(
    initial, [TabularCPD(c, (previous(c),), stay) for c in chains_] + [emission]
)
readings = [{"Signal": s} for s in ("high", "high", "low", "high")]
pf = ParticleFilter(dbn, readings, 20_000, seed=3)
unrolled = dbn.unroll(4)
seen = {f"Signal_{t + 1}": r["Signal"] for t, r in enumerate(readings)}
truth = VariableElimination(unrolled).query(["C0_4"], seen).value({"C0_4": "1"})
particles = pf.filtered("C0")[-1, 1]
print(f"   10 entangled chains: exact P(C0 on) = {truth:.4f}, particles {particles:.4f}")
print(
    "   Exact filtering needs a clique over all 10 chains; particles cost 11 lookups each per step."
)
