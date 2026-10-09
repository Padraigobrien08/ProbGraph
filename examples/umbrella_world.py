"""Milestone 5 walkthrough: the umbrella world (Russell & Norvig §14.2).

Run with:  python examples/umbrella_world.py

A security guard never sees outside. Each day the director either brings an umbrella or does
not, and the guard wants to know whether it is raining.

    Weather_1 -> Weather_2 -> Weather_3 -> ...      P(rain -> rain) = 0.7
        |            |            |
    Umbrella_1   Umbrella_2   Umbrella_3            P(umbrella | rain) = 0.9, | dry) = 0.2

Shows the ideas of Milestone 5:
  1. filtering, smoothing and prediction by forward-backward (P18, P19),
  2. why the sweeps run in log space,
  3. the most probable explanation: Viterbi, and why decoding each day separately is not it (P20),
  4. learning the model back from unlabelled sequences with Baum-Welch (P21),
  5. a dynamic Bayesian network whose chains become entangled (P22).
"""

import itertools
import math

import numpy as np

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.temporal import (
    BaumWelch,
    DynamicBayesianNetwork,
    ForwardBackward,
    HiddenMarkovModel,
    posterior_decode,
    previous,
    viterbi,
)

weather = DiscreteVariable("Weather", ("rain", "dry"))
umbrella = DiscreteVariable("Umbrella", ("umbrella", "none"))
world = HiddenMarkovModel(
    weather,
    umbrella,
    initial=[0.5, 0.5],
    transition=[[0.7, 0.3], [0.3, 0.7]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
)
days = ["umbrella", "umbrella", "none", "umbrella", "umbrella"]

# -- 1. Filtering, smoothing, prediction -------------------------------------------------

fb = ForwardBackward(world, days)
print("1. P(rain) on each day, given the umbrellas")
print(f"   {'day':>3}  {'seen':>8}  {'filtered':>9}  {'smoothed':>9}")
for t, seen in enumerate(days):
    print(f"   {t + 1:>3}  {seen:>8}  {fb.filtered[t, 0]:9.4f}  {fb.smoothed[t, 0]:9.4f}")
print("   Day 1 filtered is 9/11 = 0.8182; smoothing also uses the later days.")
ahead = ForwardBackward(world, days[:2]).predict(4)[:, 0]
print(f"   Forecast after two umbrellas: {', '.join(f'{p:.4f}' for p in ahead)}")
print("   The gap to 1/2 shrinks by exactly 0.4 a day: the chain forgets.")
print(f"   log P(days) = {fb.log_likelihood:.6f}   (P = 68607401 / 2e9)")

# -- 2. Log space ----------------------------------------------------------------------------

print("\n2. Long sequences: 5,000 days with an umbrella every day")
long = ["umbrella"] * 5000
naive = world.initial * world.emission[:, 0]
for _ in long[1:]:
    naive = (naive @ world.transition) * world.emission[:, 0]
print(
    f"   forward-backward in log space: log P = {ForwardBackward(world, long).log_likelihood:.4f}"
)
stuck = float(naive.sum())
print(f"   the textbook recursion in floats:  P = {stuck:.3g}, log P = {math.log(stuck):.1f}")
print(
    "   It is stuck at the smallest subnormal (0.63 x 5e-324 rounds back up): wrong by 1,300 nats."
)

# -- 3. The most probable explanation ----------------------------------------------------------

path, log_p = viterbi(world, days)
print("\n3. Decoding the weather")
print(f"   Viterbi (the most probable week): {path}")
print(f"   P(that week | umbrellas) = {math.exp(log_p - fb.log_likelihood):.4f}")
print(f"   Most probable weather day by day: {posterior_decode(world, days)}")
s = DiscreteVariable("S", ("s0", "s1", "s2"))
o = DiscreteVariable("O", ("a", "b"))
trap = HiddenMarkovModel(s, o, [0.4, 0.3, 0.3], [[0, 1, 0], [0, 0, 1], [0, 0, 1]], [[0.5, 0.5]] * 3)
print("   They can disagree badly. In a chain where s0 always moves to s1:")
print(f"     day by day: {posterior_decode(trap, ['a', 'b'])}  (s0 -> s2 is impossible)")
print(f"     Viterbi:    {viterbi(trap, ['a', 'b'])[0]}")
mpe, _ = VariableElimination(world.to_bayesian_network(5)).most_probable_explanation(
    {f"Umbrella_{t + 1}": y for t, y in enumerate(days)}
)
agreed = [mpe[f"Weather_{t}"] for t in range(1, 6)]
print(f"   Max-product VE on the unrolled network agrees: {agreed}")

# -- 4. Baum-Welch ---------------------------------------------------------------------------------

print("\n4. Learning the umbrella world back from 30 unlabelled sequences of 60 days")
sequences = [world.sample(60, seed=s)[1] for s in range(30)]
fit = BaumWelch(weather, umbrella, sequences, tolerance=1e-4, max_iterations=1000).run(seed=0)
learned = fit.model
order = min(
    itertools.permutations(range(2)),
    key=lambda p: np.abs(learned.emission[list(p)] - world.emission).max(),
)
p = list(order)
first, last = fit.log_likelihood[0], fit.log_likelihood[-1]
print(f"   {fit.iterations} iterations; log L {first:.1f} -> {last:.1f}")
if order != (0, 1):
    print("   (the learned states came out in the other order: label switching)")
transition = np.round(learned.transition[np.ix_(p, p)], 3).tolist()
print(f"   transition: {transition}   truth [[0.7, 0.3], [0.3, 0.7]]")
print(
    f"   emission:   {np.round(learned.emission[p], 3).tolist()}   truth [[0.9, 0.1], [0.2, 0.8]]"
)

# -- 5. A dynamic Bayesian network --------------------------------------------------------

print("\n5. Two independent weather systems, one umbrella (a factorial HMM as a DBN)")
north, south = (DiscreteVariable(n, ("rain", "dry")) for n in ("North", "South"))
brolly = DiscreteVariable("Brolly", ("yes", "no"))
initial = BayesianNetwork([north, south, brolly], [("North", "Brolly"), ("South", "Brolly")])
initial.add_cpd(TabularCPD(north, (), [0.5, 0.5]))
initial.add_cpd(TabularCPD(south, (), [0.5, 0.5]))
# P(Brolly = yes | North, South): 0.95 if both rain, 0.8 if one does, 0.1 if neither.
yes = np.array([[0.95, 0.8], [0.8, 0.1]])
either = TabularCPD(brolly, (north, south), np.stack([yes, 1 - yes]))
initial.add_cpd(either)
stay = [[0.8, 0.3], [0.2, 0.7]]
dbn = DynamicBayesianNetwork(
    initial,
    [
        TabularCPD(north, (previous(north),), stay),
        TabularCPD(south, (previous(south),), stay),
        either,
    ],
)
network = dbn.unroll(4)
ve = VariableElimination(network)
joint = ve.query(["North_4", "South_4"], {f"Brolly_{t}": "yes" for t in range(1, 5)}).values
product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
print(f"   interface: {list(dbn.interface)}")
gap = np.abs(joint - product).max()
print(f"   P(North_4, South_4 | brollies) is {gap:.3f} away from the product of its marginals.")
print("   The systems are independent a priori, but the shared umbrella entangles them:")
print("   exact inference must track both together, so its cost grows with the number of chains.")
