# ProbGraph

Discrete Bayesian networks built from first principles: DAGs, conditional
probability tables, factorisation, sampling, d-separation, and exact and
approximate inference. No graph or
Bayesian-network library is used, and NumPy is used only for array storage and
arithmetic. Every algorithm comes with a written mathematical justification and
with tests designed to fail if the implementation is subtly wrong.

**Status:** `v0.2.0`, which completes Milestone 2: conditional independence, evidence
and exact inference. (`v0.1.0` was Milestone 1: representation, factorisation and
sampling.)

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

Inference on the same model:

```python
from probgraph import VariableElimination

ve = VariableElimination(model)
ve.query(["Accident"], {"Traffic": "yes"}).value({"Accident": "yes"})  # 5/23 ≈ 0.217
ve.probability_of_evidence({"Traffic": "yes"})  # 0.3565
model.graph.d_separated({"Accident"}, {"Rain"})  # True: the collider is unobserved
model.graph.d_separated({"Accident"}, {"Rain"}, given={"Traffic"})  # False
```

Two complete walkthroughs:

- [`examples/rain_accident_traffic.py`](examples/rain_accident_traffic.py) (M1): the
  exact joint table, sampled frequencies compared with exact probabilities, and
  explaining away.
- [`examples/late_for_work.py`](examples/late_for_work.py) (M2): d-separation, exact
  posteriors, elimination cost and pruning, and sampling estimates checked against
  the exact answers.

## API

| Class | Represents | Key guarantees |
|---|---|---|
| `DiscreteVariable(name, states)` | a finite random variable | immutable; states are ordered and unique; at least one state |
| `DAG(nodes, edges)` | directed acyclic graph | an edge that would create a cycle is rejected and the graph is left unchanged; parents/children and ancestors/descendants always agree; topological order is deterministic |
| `TabularCPD(variable, parents, values)` | $p(X\mid U_1..U_k)$ | shape, finiteness, nonnegativity and per-column normalisation (`atol=1e-10`) are checked; storage is read-only; lookups are by parent name |
| `BayesianNetwork(variables, edges)` | $\prod_i p(x_i\mid\mathrm{pa}_i)$ | CPD parents must equal the graph's parents; variables must match the model's definitions; queries re-validate after any change and require complete assignments |
| `AncestralSampler(model, seed)` | i.i.d. draws from the joint | samples in topological order; inverse-CDF draws by hand; reproducible; `sample(a) + sample(b) == sample(a+b)` |
| `DiscreteFactor(variables, values)` | a nonnegative table $\phi(x_S)$ | product aligns by *name*, never by axis position; `marginalise`, `reduce`, `normalise`; immutable; overflow raises |
| `VariableElimination(model)` | exact $P(Q\mid e)$ and $P(e)$ | equals enumeration for every elimination order; min-fill by default; barren pruning by default, evidence pruning opt-in; `query_trace` shows the cost; impossible evidence raises |
| `DAG.d_separated(xs, ys, given)` | $X\perp_G Y\mid Z$ | Bayes ball in $O(\lvert V\rvert+\lvert E\rvert)$; agrees with the moralised-ancestral criterion |
| `RejectionSampler`, `LikelihoodWeighting` | sampling estimates of $P(Q\mid e)$ | return the estimate, $\hat P(e)$ and the effective sample size; raise when no sample carries information |

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
| [factor_algebra.md](docs/mathematics/factor_algebra.md) | **P5**: factors as functions; the product as broadcasting; laws F5–F10, including distributivity |
| [variable_elimination.md](docs/mathematics/variable_elimination.md) | **P6**: correctness and order independence; cost in terms of width; barren nodes and ancestral pruning |
| [elimination_orders.md](docs/mathematics/elimination_orders.md) | interaction graph = moral graph; factor elimination = vertex elimination; heuristics optimal on forests and chordal graphs |
| [d_separation.md](docs/mathematics/d_separation.md) | **P7**: Bayes ball and the walk lemma; the moralised-ancestral criterion; soundness; generic completeness (XOR); graphoid axioms; requisite evidence |
| [sampling_inference.md](docs/mathematics/sampling_inference.md) | **P8**: rejection sampling; likelihood weighting as importance sampling; bias of the ratio estimator; effective sample size |

The Milestone 2 specification is in [`docs/specs/milestone-2.md`](docs/specs/milestone-2.md).

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

## Milestone 2 acceptance

| Criterion | Evidence |
|---|---|
| The factor algebra laws (F5–F8) hold on random factors | `test_factors.py` (each operation checked cell by cell against its definition, and each law property-tested) |
| The product of CPD factors reproduces the M1 joint | `test_model_factors.py` (fixtures and 40 random networks) |
| VE matches enumeration on every random query/evidence/order tested | `test_variable_elimination.py` (100 random problems × 6 orders), `test_elimination_order.py`, `test_pruning.py` |
| The elimination order changes cost but never answers | `test_elimination_order.py::test_order_changes_cost_but_not_the_answer` (naive Bayes: 40 vs 4092 cells, identical posteriors); measured traces equal graph predictions (V8) |
| Zero-probability evidence fails explicitly | `ZeroProbabilityEvidenceError` from `query`, `InsufficientSamplesError` from the samplers. The one documented exception is the opt-in `prune_evidence` (`d_separation.md` §9) |
| d-separation agrees with the moral-ancestral criterion on every DAG with ≤ 5 nodes | `test_d_separation_oracles.py` (all 1,024 five-node DAGs up to relabelling, 81,920 queries), plus all 543 four-node labelled DAGs against literal trail enumeration |
| d-separation is numerically sound and generically complete | `test_d_separation_oracles.py::test_soundness_and_generic_completeness`; the XOR network shows that completeness is only generic |
| Pruning never changes an answer | `test_pruning.py` (barren: V7 on 80 random problems); `test_requisite_evidence.py` (requisite evidence, whenever $P(e)>0$) |
| Proofs P5–P8 are documented | `docs/mathematics/` (the table above) |
| `v0.2.0` installs fresh and passes CI | `.github/workflows/ci.yml` (unchanged gates, plus both examples) |

The Milestone 2 spec (v1.0) printed $P(L{=}1,U{=}1)$ as 0.142012. The exact value is
$11361/80000=0.1420125$, and the tests compare against exact fractions.

## Development

```bash
.venv/bin/pytest                      # 1858 tests, about 6 s
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                        # strict mode, src/ only
.venv/bin/python examples/rain_accident_traffic.py
.venv/bin/python examples/late_for_work.py
```

## Scope and limitations

Out of scope so far:
- continuous variables;
- undirected models (Markov networks) as a model class;
- junction trees and message passing;
- MAP inference;
- MCMC;
- learning parameters from data.

Exhaustive enumeration appears only as a test oracle. All computation is in
probability space, so products of very many small probabilities can underflow;
log-space factors are a candidate for Milestone 3. Variable elimination recomputes
everything for each query, and caching intermediate messages across queries is
what junction trees add.

## Licence

[MIT](LICENSE)
