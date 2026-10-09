"""Milestone 1 walkthrough: the Rain / Accident / Traffic network.

Run with:  python examples/rain_accident_traffic.py

Shows the four ideas of Milestone 1 end to end:
  1. a DAG and its topological ordering,
  2. CPDs as one probability distribution per parent configuration,
  3. the joint as a product of local conditionals (P1, P2),
  4. ancestral sampling and its agreement with the exact joint (P3).
"""

import itertools

import numpy as np

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD

# -- 1. Variables and structure: R -> T <- A (a collider) --------------------

rain = DiscreteVariable("Rain", ("no", "yes"))
accident = DiscreteVariable("Accident", ("no", "yes"))
traffic = DiscreteVariable("Traffic", ("no", "yes"))

model = BayesianNetwork(
    variables=[rain, accident, traffic],
    edges=[("Rain", "Traffic"), ("Accident", "Traffic")],
)

# -- 2. CPDs. Axis 0 is the child; the remaining axes are the parents in the order declared.

p_traffic_yes = np.array(
    [
        # Accident=no  Accident=yes
        [0.10, 0.70],  # Rain=no
        [0.80, 0.95],  # Rain=yes
    ]
)
model.add_cpd(TabularCPD(rain, (), [0.7, 0.3]))
model.add_cpd(TabularCPD(accident, (), [0.9, 0.1]))
model.add_cpd(TabularCPD(traffic, (rain, accident), np.stack([1 - p_traffic_yes, p_traffic_yes])))
model.validate()

print("Topological order:", model.graph.topological_sort())
print(
    f"Free parameters: {model.n_free_parameters} (a full joint over 3 binary variables needs 7)\n"
)

# -- 3. The exact joint, P(r, a, t) = P(r) P(a) P(t | r, a) -----------------

names = [v.name for v in model.variables]
joint = {}
print(f"{'Rain':>5} {'Accident':>9} {'Traffic':>8}   P")
for states in itertools.product(*(v.states for v in model.variables)):
    assignment = dict(zip(names, states, strict=True))
    joint[states] = model.joint_probability(assignment)
    print(f"{states[0]:>5} {states[1]:>9} {states[2]:>8}   {joint[states]:.4f}")
print(f"{'':>25}  Σ = {sum(joint.values()):.4f}\n")


def exact(event, given=lambda s: True):
    """P(event | given), computed by summing the exact joint. Feasible only for tiny networks."""
    cells = [(dict(zip(names, s, strict=True)), p) for s, p in joint.items()]
    den = sum(p for s, p in cells if given(s))
    return sum(p for s, p in cells if given(s) and event(s)) / den


# -- 4. Ancestral sampling --------------------------------------------------

n = 200_000
samples = AncestralSampler(model, seed=2026).sample(n)


def empirical(event, given=lambda s: True):
    """P(event | given) as a frequency in the samples, with its standard error."""
    subset = [s for s in samples if given(s)]
    p_hat = sum(1 for s in subset if event(s)) / len(subset)
    return p_hat, np.sqrt(p_hat * (1 - p_hat) / len(subset))


queries = [
    ("P(Traffic=yes)", lambda s: s["Traffic"] == "yes", lambda s: True),
    ("P(Accident=yes)", lambda s: s["Accident"] == "yes", lambda s: True),
    ("P(Accident=yes | Rain=yes)", lambda s: s["Accident"] == "yes", lambda s: s["Rain"] == "yes"),
    (
        "P(Accident=yes | Traffic=yes)",
        lambda s: s["Accident"] == "yes",
        lambda s: s["Traffic"] == "yes",
    ),
    (
        "P(Accident=yes | Traffic=yes, Rain=yes)",
        lambda s: s["Accident"] == "yes",
        lambda s: s["Traffic"] == "yes" and s["Rain"] == "yes",
    ),
]

print(f"Ancestral sampling, N = {n:,}")
print(f"{'query':<42} {'exact':>7} {'sampled':>8} {'± SE':>7}")
for label, event, given in queries:
    p_hat, se = empirical(event, given)
    print(f"{label:<42} {exact(event, given):7.4f} {p_hat:8.4f} {se:7.4f}")

print(
    "\nRain and Accident are independent (rows 2 and 3 match). Once Traffic is observed,\n"
    "learning that it rained makes an accident less likely: the rain explains the\n"
    "traffic. This is 'explaining away' at a collider."
)
