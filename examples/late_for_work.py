"""Milestone 2 walkthrough: why am I late for work?

Run with:  python examples/late_for_work.py

    Umbrella <- Rain -> Traffic <- Accident
                          |
                          v
                         Late

Shows the four ideas of Milestone 2 end to end:
  1. d-separation: which independences the graph guarantees (P7),
  2. exact inference by variable elimination, and explaining away (P5, P6),
  3. what elimination costs, and how pruning reduces it,
  4. approximate inference by sampling, checked against the exact answers (P8).
"""

import numpy as np

from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD, VariableElimination
from probgraph.sampling import LikelihoodWeighting, RejectionSampler

rain, accident, traffic, late, umbrella = (
    DiscreteVariable(name, ("no", "yes"))
    for name in ("Rain", "Accident", "Traffic", "Late", "Umbrella")
)
model = BayesianNetwork(
    [rain, accident, traffic, late, umbrella],
    edges=[("Rain", "Traffic"), ("Accident", "Traffic"), ("Traffic", "Late"), ("Rain", "Umbrella")],
)
t_yes = np.array([[0.10, 0.70], [0.80, 0.95]])  # P(Traffic=yes | Rain, Accident)
model.add_cpd(TabularCPD(rain, (), [0.7, 0.3]))
model.add_cpd(TabularCPD(accident, (), [0.9, 0.1]))
model.add_cpd(TabularCPD(traffic, (rain, accident), np.stack([1 - t_yes, t_yes])))
model.add_cpd(TabularCPD(late, (traffic,), [[0.9, 0.4], [0.1, 0.6]]))
model.add_cpd(TabularCPD(umbrella, (rain,), [[0.9, 0.15], [0.1, 0.85]]))

# -- 1. d-separation: read independences off the graph ----------------------

dag = model.graph
print("1. What the graph guarantees (d-separation)")
for xs, ys, zs in [
    (["Accident"], ["Rain"], []),
    (["Accident"], ["Rain"], ["Late"]),
    (["Accident"], ["Late"], ["Traffic"]),
    (["Umbrella"], ["Accident"], ["Traffic"]),
    (["Umbrella"], ["Accident"], ["Traffic", "Rain"]),
]:
    verdict = "independent" if dag.d_separated(xs, ys, zs) else "may depend"
    given = f" | {', '.join(zs)}" if zs else ""
    print(f"   {xs[0]:>8} vs {ys[0]:<8}{given:<18} {verdict}")

# -- 2. Exact inference and explaining away ----------------------------------

ve = VariableElimination(model)


def p_accident(evidence: dict[str, str]) -> float:
    return ve.query(["Accident"], evidence).value({"Accident": "yes"})


print("\n2. Why am I late? Exact posteriors by variable elimination")
for label, evidence in [
    ("prior", {}),
    ("I am late", {"Late": "yes"}),
    ("late, and colleagues carry umbrellas", {"Late": "yes", "Umbrella": "yes"}),
    ("late, and no umbrellas anywhere", {"Late": "yes", "Umbrella": "no"}),
]:
    print(f"   P(Accident=yes | {label:<38}) = {p_accident(evidence):.4f}")
print("   Umbrellas point to rain; rain explains the lateness, so an accident becomes less likely.")

# -- 3. What elimination costs ------------------------------------------------

print("\n3. Cost of P(Accident | Late=yes)")
query, evidence = ["Accident"], {"Late": "yes"}
print(f"   barren variables pruned:   {sorted(ve.barren_variables(query, evidence))}")
print(f"   min-fill elimination order: {ve.elimination_order(query, evidence)}")
for prune in (False, True):
    trace = ve.query_trace(query, evidence, prune_barren=prune)
    print(
        f"   prune_barren={prune!s:<5}  steps={len(trace.steps)}  width={trace.width}"
        f"  total cells={trace.total_cost}"
    )
umbrella_only = {"Umbrella": "yes"}
print(
    f"   P(Accident | Umbrella=yes): requisite evidence = "
    f"{ve.requisite_evidence(query, umbrella_only)} (the umbrella cannot matter)"
)

# -- 4. Approximate inference, checked against the exact answer ---------------

print("\n4. Sampling estimates of P(Accident=yes | Late=yes, Umbrella=yes), n = 100,000")
evidence = {"Late": "yes", "Umbrella": "yes"}
exact = p_accident(evidence)
p_e = ve.probability_of_evidence(evidence)
print(f"   exact (variable elimination): {exact:.4f}    P(e) = {p_e:.4f}")
for name, sampler in [
    ("rejection", RejectionSampler(model)),
    ("likelihood weighting", LikelihoodWeighting(model)),
]:
    result = sampler.query(["Accident"], evidence, n=100_000, seed=2026)
    print(
        f"   {name:<21} {result.estimate.value({'Accident': 'yes'}):.4f}"
        f"    P̂(e) = {result.evidence_estimate:.4f}"
        f"    effective samples = {result.effective_sample_size:,.0f}"
    )
print("   Rejection keeps only the ~14% of samples that match the evidence;")
print("   likelihood weighting keeps every sample but gives them unequal weights.")
