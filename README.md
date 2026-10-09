# ProbGraph

Discrete probabilistic graphical models built from first principles: Bayesian
networks and Markov networks, factorisation, sampling, d-separation, variable
elimination, clique trees and message passing, all computed in log space. No graph or
graphical-model library is used, and NumPy is used only for array storage and
arithmetic. Every algorithm comes with a written mathematical justification and
with tests designed to fail if the implementation is subtly wrong.

**Status:** `v0.3.0`, which completes Milestone 3: message passing (Markov networks,
clique trees, belief propagation, log-space numerics). Earlier releases: `v0.2.0`
(Milestone 2: conditional independence, evidence and exact inference) and `v0.1.0`
(Milestone 1: representation, factorisation and sampling).

> **Changed in v0.3.0:** `VariableElimination` now computes in log space by default. In
> v0.2.0, enough evidence to push P(e) below float64's range (about 10⁻³²⁴) could make it
> raise an error for possible evidence, or silently return a wrong posterior. Results that
> did not underflow are unchanged. `space="probability"` restores the old behaviour.

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

Every marginal at once with a junction tree, and an undirected model:

```python
from probgraph import DiscreteFactor, MarkovNetwork
from probgraph.inference import JunctionTree

JunctionTree(model, {"Traffic": "yes"}).marginals()  # P(Rain | e) and P(Accident | e) together

a, b = DiscreteVariable("A", ("0", "1")), DiscreteVariable("B", ("0", "1"))
mn = MarkovNetwork([a, b], [DiscreteFactor([a, b], [[30, 5], [1, 10]])])
mn.partition_function()  # 46.0
mn.query(["A"]).value({"A": "1"})  # 11/46 ≈ 0.239
```

Three complete walkthroughs:

- [`examples/rain_accident_traffic.py`](examples/rain_accident_traffic.py) (M1): the
  exact joint table, sampled frequencies compared with exact probabilities, and
  explaining away.
- [`examples/late_for_work.py`](examples/late_for_work.py) (M2): d-separation, exact
  posteriors, elimination cost and pruning, and sampling estimates checked against
  the exact answers.
- [`examples/misconception.py`](examples/misconception.py) (M3): a Markov network's
  partition function, triangulating a 4-cycle, one calibration for every marginal,
  loopy BP converging to the wrong answer, and why log space matters.

## API

| Class | Represents | Key guarantees |
|---|---|---|
| `DiscreteVariable(name, states)` | a finite random variable | immutable; states are ordered and unique; at least one state |
| `DAG(nodes, edges)` | directed acyclic graph | an edge that would create a cycle is rejected and the graph is left unchanged; parents/children and ancestors/descendants always agree; topological order is deterministic |
| `TabularCPD(variable, parents, values)` | $p(X\mid U_1..U_k)$ | shape, finiteness, nonnegativity and per-column normalisation (`atol=1e-10`) are checked; storage is read-only; lookups are by parent name |
| `BayesianNetwork(variables, edges)` | $\prod_i p(x_i\mid\mathrm{pa}_i)$ | CPD parents must equal the graph's parents; variables must match the model's definitions; queries re-validate after any change and require complete assignments |
| `AncestralSampler(model, seed)` | i.i.d. draws from the joint | samples in topological order; inverse-CDF draws by hand; reproducible; `sample(a) + sample(b) == sample(a+b)` |
| `DiscreteFactor(variables, values)` | a nonnegative table $\phi(x_S)$ | product aligns by *name*, never by axis position; `marginalise`, `reduce`, `normalise`; immutable; overflow raises |
| `VariableElimination(model)` | exact $P(Q\mid e)$ and $P(e)$ | equals enumeration for every elimination order; log space and min-fill by default; barren pruning by default, evidence pruning opt-in; `query_trace` shows the cost; impossible evidence raises, tiny evidence does not |
| `DAG.d_separated(xs, ys, given)` | $X\perp_G Y\mid Z$ | Bayes ball in $O(\lvert V\rvert+\lvert E\rvert)$; agrees with the moralised-ancestral criterion |
| `RejectionSampler`, `LikelihoodWeighting` | sampling estimates of $P(Q\mid e)$ | return the estimate, $\hat P(e)$ and the effective sample size; raise when no sample carries information |
| `LogFactor(variables, log_values)` | $\log\phi(x_S)$ | products add; log-sum-exp never overflows or produces NaN; `-inf` is a structural zero; never mixes with `DiscreteFactor` |
| `MarkovNetwork(variables, factors)` | $\frac1Z\prod_k\phi_k$ | $\log Z$ in log space, so $Z\approx10^{400}$ is fine; separation implies independence; `BayesianNetwork.to_markov_network()` keeps the distribution |
| `triangulate`, `is_chordal`, `maximal_cliques` | chordal graphs | recognition by maximum cardinality search, with every returned order verified; agrees with the definition on all graphs up to 6 vertices |
| `CliqueTree.from_elimination(graph, order)` | a junction tree | running intersection property; the cliques are exactly the maximal cliques; maximum-weight spanning tree |
| `JunctionTree(model, evidence)` | every $P(x\mid e)$ at once | Shafer–Shenoy in log space; $2(k-1)$ messages; exact; $\log P(e)$ for free; cross-clique queries refused |
| `LoopyBeliefPropagation(model)` | approximate marginals | exact on trees; on cycles it may converge to wrong beliefs or oscillate; `converged` is reported honestly |

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
| [log_space.md](docs/mathematics/log_space.md) | **P9**: the max-shift lemma for log-sum-exp; the factor laws transfer through the exp isomorphism; exact zero detection |
| [markov_networks.md](docs/mathematics/markov_networks.md) | **P10**: Gibbs distributions; the global Markov property; moralisation keeps the distribution but loses collider independences; no DAG captures the 4-cycle |
| [clique_trees.md](docs/mathematics/clique_trees.md) | **P11**: triangulation, perfect elimination orderings, maximum cardinality search, maximal cliques; clique trees exist exactly for chordal graphs; junction trees are the maximum-weight spanning trees |
| [message_passing.md](docs/mathematics/message_passing.md) | **P12**: Shafer–Shenoy messages as partial eliminations; beliefs are marginals; evidence; measured cost against variable elimination |
| [loopy_bp.md](docs/mathematics/loopy_bp.md) | **P13**: exact on trees; double counting on cycles; damping keeps the fixed points; convergence is not accuracy |

The specifications are in [`docs/specs/`](docs/specs/): [Milestone 2](docs/specs/milestone-2.md) and
[Milestone 3](docs/specs/milestone-3.md).

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

## Milestone 3 acceptance

| Criterion | Evidence |
|---|---|
| `LogFactor` matches `DiscreteFactor` under exp, and never overflows or produces NaN | `test_log_factor.py` (every operation against probability space; laws directly at magnitudes around ±500; 2⁻¹¹⁰⁰ and $Z\approx10^{400}$) |
| The F3 underflow regression is fixed | `test_log_space_inference.py` and `test_junction_tree_evidence.py` (P(C \| e) = [0.5, 0.5], log P(e) ≈ −784.91; probability space pinned as unreliable) |
| Markov networks: $Z$ for F2 is 7,201,840; separation implies independence | `test_markov_network.py` (exact fractions; the global Markov property and generic completeness on random models) |
| BN → MN keeps the distribution; the lost collider independence is shown | `test_markov_network.py` (Proposition 3 on random DAGs; Proposition 4 over all 543 four-node DAGs) |
| Clique trees satisfy the running intersection property | `test_clique_tree.py` (brute-force RIP, the closed form $\sum\lvert C_i\rvert-n$, and an independent Kruskal maximum spanning tree); `test_chordal.py` (OEIS A058862 counts on all graphs up to 6 vertices) |
| One calibration reproduces the M2 posteriors and F2's marginals as exact fractions | `test_junction_tree.py`, `test_junction_tree_evidence.py` |
| Calibrated marginals equal VE on random BNs and MNs, with and without evidence | the same files (180+ random models, every root and elimination order) |
| Calibration is cheaper than repeated VE when evidence is downstream | `test_junction_tree_evidence.py` (4–10×). The no-evidence counter-case, where pruned VE wins, is also pinned. |
| Proofs P9–P13 are documented | `docs/mathematics/` (the table above) |
| `v0.3.0` installs fresh and passes CI | `.github/workflows/ci.yml` (unchanged gates, plus all three examples) |

Measuring and testing during M3 changed the spec three times (now v1.3). Probability space
was found to *silently* return wrong posteriors, not merely raise. Variable elimination
therefore defaults to log space. And the cost claim C7 holds only when evidence is
downstream.

## Development

```bash
.venv/bin/pytest                      # 2853 tests, about 15 s
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                        # strict mode, src/ only
.venv/bin/python examples/rain_accident_traffic.py
.venv/bin/python examples/late_for_work.py
.venv/bin/python examples/misconception.py
```

## Scope and limitations

Out of scope so far:
- continuous variables;
- MAP / max-product inference;
- MCMC (Gibbs sampling);
- learning parameters or structure from data.

Known limitations:
- `JunctionTree.query` answers joint queries only over variables that share a clique. Use
  `VariableElimination` for the rest.
- Loopy belief propagation has no accuracy guarantee on graphs with cycles. The tests show it
  converging to badly wrong beliefs, and oscillating forever without damping.
- Choosing a greedy elimination order is slow for graphs with a very high-degree variable
  (min-fill scoring is quadratic in the degree). Bayesian networks avoid the worst case.
- Exhaustive enumeration appears only as a test oracle.

## Licence

[MIT](LICENSE)
