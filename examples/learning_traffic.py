"""Milestone 4 walkthrough: learning the late-for-work network back from data.

Run with:  python examples/learning_traffic.py

    Umbrella <- Rain -> Traffic <- Accident
                          |
                          v
                         Late

Sample data from a known network, forget its parameters, and learn them back:
  1. maximum likelihood: counts, and how close they get (P14),
  2. Dirichlet priors: smoothing small samples, and the Bayesian score (P15),
  3. EM when values are missing, against discarding incomplete rows (P16),
  4. choosing the structure with BIC and BDeu (P17).
"""

import itertools

import numpy as np

from probgraph import AncestralSampler, BayesianNetwork, DiscreteVariable, TabularCPD
from probgraph.exceptions import ValidationError
from probgraph.learning import (
    Dataset,
    DirichletPrior,
    ExpectationMaximisation,
    bayesian_estimate,
    log_marginal_likelihood,
    maximum_likelihood,
    score_structures,
)

rain, accident, traffic, late, umbrella = (
    DiscreteVariable(name, ("no", "yes"))
    for name in ("Rain", "Accident", "Traffic", "Late", "Umbrella")
)
EDGES = [("Rain", "Traffic"), ("Accident", "Traffic"), ("Traffic", "Late"), ("Rain", "Umbrella")]
truth = BayesianNetwork([rain, accident, traffic, late, umbrella], EDGES)
t_yes = np.array([[0.10, 0.70], [0.80, 0.95]])  # P(Traffic=yes | Rain, Accident)
truth.add_cpd(TabularCPD(rain, (), [0.7, 0.3]))
truth.add_cpd(TabularCPD(accident, (), [0.9, 0.1]))
truth.add_cpd(TabularCPD(traffic, (rain, accident), np.stack([1 - t_yes, t_yes])))
truth.add_cpd(TabularCPD(late, (traffic,), [[0.9, 0.4], [0.1, 0.6]]))
truth.add_cpd(TabularCPD(umbrella, (rain,), [[0.9, 0.15], [0.1, 0.85]]))

structure = BayesianNetwork(truth.variables, EDGES)  # the graph only: no parameters


def sample(n: int, seed: int) -> Dataset:
    return Dataset.from_samples(truth, AncestralSampler(truth, seed=seed).sample(n))


def worst_error(model: BayesianNetwork) -> float:
    """The largest |θ̂ - θ| over every entry of every CPD."""
    worst = 0.0
    for name, cpd in truth.cpds.items():
        for config in itertools.product(*(p.states for p in cpd.parents)):
            given = dict(zip(cpd.parent_names, config, strict=True))
            error = np.abs(model.cpds[name].distribution(given) - cpd.distribution(given)).max()
            worst = max(worst, float(error))
    return worst


# -- 1. Maximum likelihood ------------------------------------------------------------

print("1. Maximum likelihood: θ̂(x | u) = N(x, u) / N(u)")
for n in (1_000, 10_000, 100_000):
    error = worst_error(maximum_likelihood(structure, sample(n, 2026)))
    print(f"   N = {n:>7,}: worst error {error:.4f}")
print(
    "   The worst entry is P(Traffic | Rain=yes, Accident=yes): that configuration is 3% of rows."
)

small = sample(20, seed=3)
try:
    maximum_likelihood(structure, small)
except ValidationError as error:
    print(f"\n   With 20 rows the MLE is undefined somewhere:\n     {error}")

# -- 2. Dirichlet priors ------------------------------------------------------------------

print("\n2. A Dirichlet prior adds pseudocounts: (N(x, u) + α) / (N(u) + α·)")
smoothed = bayesian_estimate(structure, small, DirichletPrior.uniform(1.0))
p = smoothed.cpds["Traffic"].probability("yes", {"Rain": "yes", "Accident": "yes"})
print(f"   Laplace (α = 1) gives P(Traffic=yes | Rain=yes, Accident=yes) = {p:.3f} (truth 0.95)")
print("   An unseen column gets the prior mean; data overrule the prior as N(u) grows.")
larger = bayesian_estimate(structure, sample(10_000, 2026), DirichletPrior.uniform())
print(
    f"   worst error with 20 rows: {worst_error(smoothed):.3f}; "
    f"with the 10,000 rows of part 1: {worst_error(larger):.4f}"
)

data = sample(500, seed=5)
equivalent = BayesianNetwork(
    truth.variables, [e for e in EDGES if e != ("Rain", "Umbrella")] + [("Umbrella", "Rain")]
)
for name, prior in [("BDeu(1)", DirichletPrior.bdeu(1.0)), ("K2", DirichletPrior.uniform(1.0))]:
    a = log_marginal_likelihood(structure, data, prior)
    b = log_marginal_likelihood(equivalent, data, prior)
    print(f"   log P(D | G), {name:<7}: Rain→Umbrella {a:.4f}, Umbrella→Rain {b:.4f}")
print("   BDeu scores the two equivalent graphs equally; K2 does not.")

# -- 3. EM with missing values ---------------------------------------------------------------

print("\n3. EM: 30% of all values hidden at random")
full = sample(20_000, seed=6)
masked = full.with_missing(0.3, seed=6)
complete_rows = Dataset(truth.variables, [r for r in masked.rows() if None not in r.values()])
print(f"   rows with nothing missing: {complete_rows.n_rows:,} of {masked.n_rows:,}")
result = ExpectationMaximisation(structure, masked, tolerance=1e-7).run(seed=0)
history = result.log_likelihood
print(
    f"   EM: {result.iterations} iterations; log L {history[0]:,.1f} → {history[-1]:,.1f}, "
    f"never decreasing: {bool(np.all(np.diff(history) >= -1e-10))}"
)
for label, model in [
    ("EM on all rows", result.model),
    ("MLE on complete rows only", maximum_likelihood(structure, complete_rows)),
    ("MLE before hiding anything", maximum_likelihood(structure, full)),
]:
    print(f"   worst error, {label:<27} {worst_error(model):.4f}")

# -- 4. Structure scores ------------------------------------------------------------------------

print("\n4. Which graph? BIC = log L(θ̂) - (d/2) log N, relative to the best candidate")
candidates = {
    "truth": structure,
    "Umbrella→Rain (equivalent)": equivalent,
    "no Accident→Traffic": BayesianNetwork(
        truth.variables, [e for e in EDGES if e != ("Accident", "Traffic")]
    ),
    "+ Rain→Late": BayesianNetwork(truth.variables, [*EDGES, ("Rain", "Late")]),
    "+ Accident→Umbrella": BayesianNetwork(truth.variables, [*EDGES, ("Accident", "Umbrella")]),
}
names = {id(g): name for name, g in candidates.items()}
for n in (50, 50_000):
    ranked = score_structures(candidates.values(), sample(n, seed=1), "bic")
    best = ranked[0][0]
    print(f"   N = {n:,}:")
    for score, g in ranked:
        print(f"     {names[id(g)]:<28} {score - best:9.2f}")
print("   With little data the penalty wins and a sparser graph ranks first;")
print("   with plenty, the true graph and its equivalent tie at the top.")
