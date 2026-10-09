"""Milestone 3 walkthrough: the misconception network (Koller & Friedman §4.1).

Run with:  python examples/misconception.py

Four students study in pairs around a cycle, A – B – C – D – A. Each pairwise
factor says how much two study partners tend to agree about a misconception.

Shows the four ideas of Milestone 3:
  1. a Markov network and its partition function Z (P10),
  2. triangulation and the clique tree that the 4-cycle needs (P11),
  3. one calibration giving every marginal, with and without evidence (P12),
  4. loopy BP on the raw cycle: it converges, but to the wrong answer (P13),
and why everything is computed in log space (P9).
"""

import math

from probgraph import (
    BayesianNetwork,
    DiscreteFactor,
    DiscreteVariable,
    MarkovNetwork,
    TabularCPD,
    VariableElimination,
)
from probgraph.inference import JunctionTree, LoopyBeliefPropagation

a, b, c, d = (DiscreteVariable(name, ("0", "1")) for name in "ABCD")
model = MarkovNetwork(
    [a, b, c, d],
    [
        DiscreteFactor([a, b], [[30, 5], [1, 10]]),
        DiscreteFactor([b, c], [[100, 1], [1, 100]]),
        DiscreteFactor([c, d], [[1, 100], [100, 1]]),
        DiscreteFactor([d, a], [[100, 1], [1, 100]]),
    ],
)

# -- 1. The partition function ----------------------------------------------------

print("1. A Markov network over the cycle A – B – C – D – A")
print(f"   Z = {model.partition_function():,.0f}   (the textbook value is 7,201,840)")
x = {"A": "1", "B": "1", "C": "0", "D": "1"}
print(f"   P(a1, b1, c0, d1) = 10·1·100·100 / Z = {model.probability(x):.6f}")
print(
    f"   A ⊥ C | B, D ? {model.separated({'A'}, {'C'}, {'B', 'D'})}    "
    f"A ⊥ C | B ? {model.separated({'A'}, {'C'}, {'B'})}"
)

# -- 2. Triangulation and the clique tree ----------------------------------------

jt = JunctionTree(model)
print("\n2. The 4-cycle is not chordal, so min-fill adds the edge B – D")
for i, clique in enumerate(jt.tree.cliques):
    print(f"   clique {i}: {sorted(clique)}")
for i, j in jt.tree.edges:
    print(f"   separator {i} – {j}: {sorted(jt.tree.separator(i, j))}")

# -- 3. One calibration, every marginal --------------------------------------------

print(
    f"\n3. Calibrated with {jt.message_count()} messages; log Z = {jt.log_partition_function():.4f}"
)
for name, marginal in jt.marginals().items():
    print(f"   P({name}=1) = {marginal.value({name: '1'}):.4f}")
given = JunctionTree(model, {"C": "1"})
print(
    f"   Given C=1:  P(B=1 | C=1) = {given.marginal('B').value({'B': '1'}):.4f}   "
    f"P(A=1 | C=1) = {given.marginal('A').value({'A': '1'}):.4f}   "
    f"P(C=1) = {math.exp(given.log_probability_of_evidence()):.4f}"
)

# -- 4. Loopy BP on the raw cycle ---------------------------------------------------

result = LoopyBeliefPropagation(model, max_iterations=1000).run()
print(
    f"\n4. Loopy BP on the cycle: converged={result.converged} after {result.iterations} iterations"
)
for name in "ABCD":
    exact = jt.marginal(name).value({name: "1"})
    approx = result.marginals[name].value({name: "1"})
    print(
        f"   P({name}=1): loopy {approx:.4f}   exact {exact:.4f}   error {abs(approx - exact):.4f}"
    )
print(
    "   Converging is not the same as being right: messages around the cycle count evidence twice."
)

# -- Why log space ------------------------------------------------------------------

print("\n5. Why everything is computed in log space")
c_var = DiscreteVariable("Class", ("0", "1"))
features = [DiscreteVariable(f"F{i}", ("0", "1")) for i in range(1100)]
nb = BayesianNetwork([c_var, *features], [("Class", f.name) for f in features])
nb.add_cpd(TabularCPD(c_var, (), [0.5, 0.5]))
for f in features:
    nb.add_cpd(TabularCPD(f, (c_var,), [[0.6, 0.4], [0.4, 0.6]]))
evidence = {f.name: ("0" if i < 550 else "1") for i, f in enumerate(features)}
ve = VariableElimination(nb)
log_p_e = ve.log_probability_of_evidence(evidence)
wrong = ve.query(["Class"], evidence, space="probability").values.round(4).tolist()
right = ve.query(["Class"], evidence).values.round(4).tolist()
print(f"   1,100 observations: log P(e) = {log_p_e:.2f}, so P(e) ≈ 10^{log_p_e / math.log(10):.0f}")
print("   (below float64's smallest positive number, about 10^-324)")
print(f"   probability space: P(Class | e) = {wrong}  (wrong)")
print(f"   log space:         P(Class | e) = {right}  (correct, by symmetry)")
