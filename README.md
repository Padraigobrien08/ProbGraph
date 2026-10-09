# ProbGraph

Discrete Bayesian networks built from first principles: DAGs, conditional
probability tables, joint factorisation and ancestral sampling. No graph or
Bayesian-network library is used, and NumPy is used only for array storage and
arithmetic. Every algorithm comes with a written mathematical justification and
with tests designed to fail if the implementation is subtly wrong.

**Status:** `v0.1.0`, which completes Milestone 1: representation, factorisation
and sampling.

## Install

```bash
git clone <this repository> && cd ProbGraph
uv venv && uv pip install -e ".[dev]"   # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Requires Python ≥ 3.11 and NumPy ≥ 1.26.

## Quick start

```python
import numpy as np
from probgraph import BayesianNetwork, DiscreteVariable, TabularCPD

rain = DiscreteVariable("Rain", ("no", "yes"))
accident = DiscreteVariable("Accident", ("no", "yes"))
traffic = DiscreteVariable("Traffic", ("no", "yes"))

model = BayesianNetwork(
    [rain, accident, traffic],
    edges=[("Rain", "Traffic"), ("Accident", "Traffic")],
)

# values.shape == (|child|, |parent_1|, ..., |parent_k|)
# P(Traffic=yes | Rain, Accident): rows are Rain=no/yes, columns are Accident=no/yes.
t_yes = np.array([[0.10, 0.70], [0.80, 0.95]])
model.add_cpd(TabularCPD(rain, (), [0.7, 0.3]))
model.add_cpd(TabularCPD(accident, (), [0.9, 0.1]))
model.add_cpd(TabularCPD(traffic, (rain, accident), np.stack([1 - t_yes, t_yes])))

model.joint_probability({"Rain": "yes", "Accident": "no", "Traffic": "yes"})  # 0.216
model.sample(3, seed=0)  # [{'Rain': 'no', 'Accident': 'no', 'Traffic': 'no'}, ...]
model.n_free_parameters  # 6, compared with 7 for an unrestricted joint
```

[`examples/rain_accident_traffic.py`](examples/rain_accident_traffic.py) is a
complete walkthrough: the exact joint table, sampled frequencies compared with
exact probabilities, and explaining away.

## API

| Class | Represents | Key guarantees |
|---|---|---|
| `DiscreteVariable(name, states)` | a finite random variable | immutable; states are ordered and unique; at least one state |
| `DAG(nodes, edges)` | directed acyclic graph | an edge that would create a cycle is rejected and the graph is left unchanged; parents/children and ancestors/descendants always agree; topological order is deterministic |
| `TabularCPD(variable, parents, values)` | $p(X\mid U_1..U_k)$ | shape, finiteness, nonnegativity and per-column normalisation (`atol=1e-10`) are checked; storage is read-only; lookups are by parent name |
| `BayesianNetwork(variables, edges)` | $\prod_i p(x_i\mid\mathrm{pa}_i)$ | CPD parents must equal the graph's parents; variables must match the model's definitions; queries re-validate after any change and require complete assignments |
| `AncestralSampler(model, seed)` | i.i.d. draws from the joint | samples in topological order; inverse-CDF draws by hand; reproducible; `sample(a) + sample(b) == sample(a+b)` |

All library errors derive from `probgraph.exceptions.ProbGraphError`.

## Mathematics

Each implementation step is justified in [`docs/mathematics/`](docs/mathematics/):

| Document | Content |
|---|---|
| [factorisation.md](docs/mathematics/factorisation.md) | **P1**: chain rule plus local Markov property gives the DAG factorisation; worked collider example; explaining away |
| [dag_algorithms.md](docs/mathematics/dag_algorithms.md) | correctness of cycle rejection, ancestor/descendant search and Kahn's algorithm; the nilpotent-adjacency test oracle |
| [conditional_probability_tables.md](docs/mathematics/conditional_probability_tables.md) | CPDs as one simplex point per parent configuration; axis convention; tolerance policy |
| [joint_normalisation.md](docs/mathematics/joint_normalisation.md) | **P2**: locally normalised CPDs give a globally normalised joint; corollaries used as tests |
| [ancestral_sampling.md](docs/mathematics/ancestral_sampling.md) | **P3**: inverse-CDF lemma and proof that ancestral sampling is exact; Bernstein-based test tolerances |
| [parameter_reduction.md](docs/mathematics/parameter_reduction.md) | **P4**: parameter counting, telescoping for complete DAGs, Jacobian-rank experiment |

## Milestone 1 acceptance

| Criterion | Evidence |
|---|---|
| Every valid DAG has a valid topological ordering | `test_dag.py::test_invariants_hold_under_random_insertions` (G2 on random graphs) |
| Cycle insertion fails without modifying the graph | `test_dag.py` (G5 snapshot comparison on every rejected insertion) |
| Every CPD validates its domain, shape and normalisation | `test_cpd.py` (spec table, plus randomised perturbation tests) |
| A complete model passes graph/CPD consistency checks | `test_bayesian_network.py` (B1, B2, B6 tests) |
| Joint probabilities agree with hand-calculated examples | `test_bayesian_network.py::test_joint_probability_matches_hand_calculation` |
| Exhaustive probabilities sum to one on small valid networks | `test_bayesian_network.py::test_random_networks_define_valid_joint_distributions` |
| Fixed-seed ancestral sampling is reproducible | `test_ancestral_sampling.py` (S3 tests, including stream consistency) |
| Empirical frequencies agree statistically with theory | `test_ancestral_sampling.py` (Bernstein-bounded frequency tests, 20 random networks) |
| Mathematical proofs and algorithm explanations are documented | `docs/mathematics/` (P1–P4) |
| The package installs and passes all CI checks | `.github/workflows/ci.yml` (lint, types, tests on 3.11–3.13; wheel and sdist installed fresh; oldest dependency versions) |

The Milestone 1 spec (v0.3) gives $P(T=1)=0.3615$ for the fixture. The correct
value is $0.3565$, and the tests use the corrected value.

## Development

```bash
.venv/bin/pytest                      # 221 tests, about 2 s
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                        # strict mode, src/ only
.venv/bin/python examples/rain_accident_traffic.py
```

## Scope and limitations

These are deliberately out of scope for Milestone 1: continuous variables,
undirected models, d-separation queries, and posterior inference (variable
elimination, message passing). Exhaustive enumeration appears only as a test
oracle. `joint_probability` multiplies probabilities directly, so it can
underflow for networks with many hundreds of low-probability factors. A
log-space variant is a candidate for a later milestone.

## Licence

[MIT](LICENSE)
