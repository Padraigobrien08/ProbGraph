# ProbGraph

Discrete probabilistic graphical models built from first principles: Bayesian
networks and Markov networks, factorisation, sampling, d-separation, variable
elimination, clique trees and message passing, all computed in log space; learning
their parameters and structure from data, including data with missing values; and hidden
Markov models and dynamic Bayesian networks over time. No graph or
graphical-model library is used, and NumPy is used only for array storage and
arithmetic. Every algorithm comes with a written mathematical justification and
with tests designed to fail if the implementation is subtly wrong.

**Status:** `v0.5.0`, which completes Milestone 5: temporal models (hidden Markov models,
forward–backward, Viterbi and the most probable explanation, Baum–Welch, and dynamic
Bayesian networks). Earlier releases: `v0.4.0` (Milestone 4: learning from data), `v0.3.0`
(Milestone 3: message passing), `v0.2.0` (Milestone 2: conditional independence, evidence and
exact inference) and `v0.1.0` (Milestone 1: representation, factorisation and sampling).

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

Learning the parameters back from data, with or without missing values:

```python
from probgraph.learning import Dataset, DirichletPrior, ExpectationMaximisation
from probgraph.learning import bayesian_estimate, bic, maximum_likelihood

data = Dataset.from_samples(model, model.sample(5000, seed=1))
learned = maximum_likelihood(model, data)  # uses the graph only; the CPDs come from counts
learned.cpds["Traffic"].probability("yes", {"Rain": "no", "Accident": "yes"})  # 0.684 (truth 0.70)

small = Dataset.from_samples(model, model.sample(30, seed=1))
# Rain=yes with Accident=yes never occurs in 30 rows: the MLE raises, and Laplace gives the prior mean.
bayesian_estimate(model, small, DirichletPrior.uniform(1.0))  # that column becomes [0.5, 0.5]

result = ExpectationMaximisation(model, data.with_missing(0.3, seed=2)).run(seed=0)
result.converged, result.iterations  # (True, 23); result.log_likelihood never decreases

bic(model, data) > bic(BayesianNetwork(model.variables, [("Rain", "Traffic")]), data)  # True
```

Sequences with a hidden Markov model:

```python
from probgraph.temporal import BaumWelch, ForwardBackward, HiddenMarkovModel, viterbi

weather = DiscreteVariable("Weather", ("rain", "dry"))
umbrella = DiscreteVariable("Umbrella", ("umbrella", "none"))
hmm = HiddenMarkovModel(
    weather,
    umbrella,
    initial=[0.5, 0.5],
    transition=[[0.7, 0.3], [0.3, 0.7]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
)
fb = ForwardBackward(hmm, ["umbrella", "umbrella"])
fb.filtered[1, 0], fb.smoothed[0, 0]  # P(rain_2 | u1, u2) = P(rain_1 | u1, u2) = 621/703
fb.predict(1)[0, 0]  # 0.6533: tomorrow, forgetting at rate 0.4
viterbi(hmm, ["umbrella", "umbrella", "none", "umbrella", "umbrella"])[0]
# ['rain', 'rain', 'dry', 'rain', 'rain']

sequences = [hmm.sample(60, seed=s)[1] for s in range(30)]
bw = BaumWelch(weather, umbrella, sequences, tolerance=1e-4, max_iterations=1000)
fit = bw.run(seed=0)  # converges after 326 iterations; compare states up to relabelling
```

Five complete walkthroughs:

- [`examples/rain_accident_traffic.py`](examples/rain_accident_traffic.py) (M1): the
  exact joint table, sampled frequencies compared with exact probabilities, and
  explaining away.
- [`examples/late_for_work.py`](examples/late_for_work.py) (M2): d-separation, exact
  posteriors, elimination cost and pruning, and sampling estimates checked against
  the exact answers.
- [`examples/misconception.py`](examples/misconception.py) (M3): a Markov network's
  partition function, triangulating a 4-cycle, one calibration for every marginal,
  loopy BP converging to the wrong answer, and why log space matters.
- [`examples/learning_traffic.py`](examples/learning_traffic.py) (M4): learning the
  late-for-work network back from samples, smoothing small samples, EM with 30% of values
  missing, and ranking structures by BIC.
- [`examples/umbrella_world.py`](examples/umbrella_world.py) (M5): filtering, smoothing and
  prediction, why the sweeps run in log space, Viterbi against day-by-day decoding,
  Baum–Welch, and two weather systems entangled by one umbrella.

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
| `learning.Dataset(variables, rows)` | observations, `None` for missing | validated once; immutable; exact `counts`; `with_missing` is MCAR, even with the sampler's seed |
| `learning.maximum_likelihood(structure, data)` | $\hat\theta=N(x,u)/N(u)$ | exact count ratios; unseen parent configurations raise unless `unseen="uniform"`; complete data only |
| `learning.bayesian_estimate(structure, data, prior)` | the Dirichlet posterior mean or MAP | `DirichletPrior.uniform`, `bdeu` or `explicit`; MAP refused when any $\alpha<1$ |
| `learning.log_marginal_likelihood(structure, data, prior)` | $\log P(D\mid G)$ | the gamma closed form, equal to sequential prediction; BDeu is score-equivalent |
| `learning.ExpectationMaximisation(structure, data, prior)` | parameters from incomplete data | E-step by junction tree, one per distinct observed pattern; the objective never decreases; the full history is reported |
| `learning.bic`, `family_scores`, `score_structures` | structure scores | BIC and BDeu; one term per family; equivalent structures tie |
| `VariableElimination.most_probable_explanation(e)` | $\arg\max_xP(x,e)$ | max-sum elimination and traceback; exact in every order; never prunes barren nodes |
| `temporal.HiddenMarkovModel(...)` | a homogeneous discrete HMM | validated tables; exact unrolling; sampling; the stationary distribution when it is unique |
| `temporal.ForwardBackward(model, ys)` | filtered, smoothed, pairwise, predicted beliefs | $O(TK^2)$ in log space; exact for 5,000+ steps; missing observations are `None` |
| `temporal.viterbi`, `posterior_decode` | decoding | Viterbi is the unrolled MPE; posterior decoding maximises expected correct steps |
| `temporal.BaumWelch(...)` | HMM parameters from sequences | tied EM; the objective never decreases; many sequences; pseudocounts |
| `temporal.DynamicBayesianNetwork(initial, transition)` | a 2-TBN | derived interface that d-separates past and future; exact unrolling |

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
| [likelihood.md](docs/mathematics/likelihood.md) | **P14**: counts are sufficient; the likelihood decomposes by column; the MLE by Gibbs' inequality; Bernstein bounds; misspecification; Wilks |
| [dirichlet.md](docs/mathematics/dirichlet.md) | **P15**: conjugacy; posterior mean and MAP; the marginal likelihood as a ratio of normalising constants, equal to sequential prediction; BDeu score equivalence |
| [em.md](docs/mathematics/em.md) | **P16**: MAR; Jensen's bound; expected counts; monotonicity; fixed points are stationary; MAP-EM; the symmetric latent-class fixed point |
| [model_selection.md](docs/mathematics/model_selection.md) | **P17**: BIC as a Laplace approximation; decomposability; score equivalence by the entropy chain rule; consistency via Wilks |
| [markov_chains.md](docs/mathematics/markov_chains.md) | **P18**: HMMs as unrolled networks; tied parameters; stationary distributions and closed classes; geometric forgetting |
| [forward_backward.md](docs/mathematics/forward_backward.md) | **P19**: filtering, smoothing and prediction; why the recursion runs in log space; equivalence with Shafer–Shenoy |
| [max_product.md](docs/mathematics/max_product.md) | **P20**: semirings; max-product VE and traceback; why barren pruning fails for MPE; Viterbi; posterior decoding versus MAP |
| [baum_welch.md](docs/mathematics/baum_welch.md) | **P21**: tied EM; two valid treatments of missing observations; label switching; the symmetric saddle |
| [dbn.md](docs/mathematics/dbn.md) | **P22**: 2-TBNs; the interface d-separates past and future; entanglement |

The specifications are in [`docs/specs/`](docs/specs/): [Milestone 2](docs/specs/milestone-2.md),
[Milestone 3](docs/specs/milestone-3.md), [Milestone 4](docs/specs/milestone-4.md) and
[Milestone 5](docs/specs/milestone-5.md).

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

## Milestone 4 acceptance

| Criterion | Evidence |
|---|---|
| `Dataset` validates states, uses only `None` for missing, and counts exactly | `test_dataset.py` (counts against brute force on random data; MCAR rates within Bernstein bounds, independent of the values) |
| The MLE matches F1 exactly, and no valid perturbation increases the likelihood | `test_likelihood.py` (exact fractions; random perturbations inside the simplex) |
| Learning from sampled data recovers the generating parameters within proven bounds | `test_recovery.py` (F2; every entry within its Bernstein bound; the $1/\sqrt{N(u)}$ rate; Wilks' $\chi^2_d$) |
| Dirichlet posterior means and MAP match their closed forms; $\alpha\to0$ gives the MLE | `test_dirichlet.py` (quadrature oracles; tallies; the convex combination) |
| The gamma formula equals sequential prediction; BDeu is score-equivalent | `test_marginal_likelihood.py` (F5; random orderings; summing to 1 over all datasets; covered-edge reversals under BDeu and BDe) |
| EM's first iteration on F3 is exact; the likelihood never decreases | `test_em.py` (23/48, 18/23, 1/5; expected counts against enumeration; monotone objectives with and without priors) |
| EM agrees with brute-force EM; the symmetric fixed point and its escape are shown | `test_em_oracles.py` (iteration by iteration; Theorem 2's formula; label switching; MCAR recovery; MAR vs MNAR) |
| Proofs P14–P17 are documented | `docs/mathematics/` (the table above) |
| `v0.4.0` installs fresh and passes CI | `.github/workflows/ci.yml` (unchanged gates, plus all four examples) |

Testing during M4 corrected the spec twice (now v1.2): with a prior, EM's posterior-mean
step does not make the log-likelihood monotone (it makes a shifted log-posterior monotone),
and BIC's equal-parameter-count condition always holds for equivalent structures. It also
found that `with_missing` and `AncestralSampler` with the same seed shared their uniforms,
so the "MCAR" mask depended on the values. That is fixed.

## Milestone 5 acceptance

| Criterion | Evidence |
|---|---|
| Unrolling is exact, and sampled sequences pass statistical tests | `test_hmm.py` (the joint against the HMM formula for every assignment; transition and emission frequencies within Bernstein bounds) |
| Filtering, smoothing, pairwise posteriors and the likelihood match brute force and the unrolled junction tree | `test_forward_backward.py`, `test_smoothing.py` (F1 as exact fractions; all $K^T$ paths; `JunctionTree` on the unrolled network) |
| 5,000-step sequences are exact where the textbook recursion fails | `test_forward_backward.py` (F4 against exact rationals; the stuck subnormal); extreme models found by search, against log-space VE |
| Prediction converges at the proven rate | `test_smoothing.py` (exactly $\frac12+0.4^k(\ldots)$; convergence to the stationary distribution) |
| MPE by max-product VE; Viterbi equals the unrolled MPE and brute force | `test_max_product.py` (F5; 80 random networks), `test_viterbi.py` |
| Posterior decoding can return an impossible path | `test_viterbi.py` (F2; each decoder optimal for its own criterion) |
| Baum–Welch: F3 exactly; the objective never decreases; counts match M4 on the unrolled network | `test_baum_welch.py`, `test_baum_welch_oracles.py` (brute-force EM step by step; relabelling; Theorem 1) |
| Proofs P18–P22 are documented | `docs/mathematics/` (the table above) |
| `v0.5.0` installs fresh and passes CI | `.github/workflows/ci.yml` (unchanged gates, plus all five examples) |

Testing during M5 changed the spec three times (now v1.3):
- Linear scaling in forward–backward was replaced by log space, after a search of extreme
  models found it wrong by up to 1.0.
- The symmetric Baum–Welch fixed point also needs a stationary π.
- M4's EM fills in missing observations where Baum–Welch sums them out. Both are valid, with
  the same fixed points.

## Development

```bash
.venv/bin/pytest                      # 4514 tests, about 70 s
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                        # strict mode, src/ only
.venv/bin/python examples/rain_accident_traffic.py
.venv/bin/python examples/late_for_work.py
.venv/bin/python examples/misconception.py
.venv/bin/python examples/learning_traffic.py
.venv/bin/python examples/umbrella_world.py
```

CI runs on Linux (Python 3.11–3.13, the oldest supported dependencies, and fresh wheel and
sdist installs) for every push and pull request. The full Linux and macOS matrix runs on
demand, with `gh workflow run CI`, on each release commit before it is tagged.

## Scope and limitations

Out of scope so far:
- continuous variables (Gaussian HMMs, Kalman filters);
- marginal MAP (maximising some variables while summing out others);
- MCMC (Gibbs sampling), particle filtering and approximate DBN inference;
- structure *search* (the scores are here; searching over graphs is not);
- learning Markov network parameters;
- data missing not at random (EM assumes MAR).

Known limitations:
- `JunctionTree.query` answers joint queries only over variables that share a clique. Use
  `VariableElimination` for the rest.
- Loopy belief propagation has no accuracy guarantee on graphs with cycles. The tests show it
  converging to badly wrong beliefs, and oscillating forever without damping.
- EM finds a local optimum that depends on its start; for latent variables, the classes are
  identified only up to relabelling. Each iteration builds one junction tree per distinct
  pattern of observed values, so it is slow when almost every row is different.
- Baum–Welch shares EM's local optima and label switching, and converges slowly when the
  observations carry little information about the states. Near its symmetric saddle a
  tolerance-based stop can report convergence far below the optimum: use several starts.
- Exact DBN inference costs grow exponentially with the interface (entanglement).
- Exhaustive enumeration appears only as a test oracle.

## Licence

[MIT](LICENSE)
