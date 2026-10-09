# Changelog

## 0.5.0 — Milestone 5: temporal models

### Added
- `probgraph.temporal`, a new subpackage.
- `HiddenMarkovModel`: validated, homogeneous discrete HMMs, with `sample`,
  `to_bayesian_network(T)` (exact unrolling), `stationary_distribution` (which names
  the closed classes when it is not unique) and `n_free_parameters`.
- `ForwardBackward`: `filtered`, `smoothed`, `pairwise`, `predict(k)`,
  `log_likelihood` and `step_log_likelihoods` in O(TK²). Missing observations are
  `None`. Impossible sequences give −∞ and raise on access.
- `VariableElimination.most_probable_explanation`: the MPE by max-product (max-sum)
  elimination with traceback, for any Bayesian network.
- `viterbi` and `posterior_decode`.
- `BaumWelch`, `BaumWelchResult`, `HMMCounts` and `supervised_estimate`: EM with
  parameters tied across time, for many sequences, with optional pseudocounts and a
  `log_objective` history.
- `DynamicBayesianNetwork` (2-TBNs) and `previous`, with a derived `interface`,
  `from_hmm` and `unroll(T)`.
- Proofs P18–P22 in `docs/mathematics/`, and `examples/umbrella_world.py`.

### Notes
- Forward–backward runs the normalised recursion **in log space**. The spec first
  chose linear scaling, but a search of extreme models found it wrong by up to 1.0. The
  textbook unnormalised recursion is worse: for 5,000 steps it sticks at the smallest
  subnormal, 5e-324, and reports log P ≈ −744 for a true −2069.6, with no error.
- Barren pruning is invalid for MPE: max_x P(x | u) depends on u. So the MPE never
  prunes, and with a missing observation the unrolled MPE differs from Viterbi.
- Posterior decoding can return a path of probability zero; Viterbi cannot.
- Baum–Welch sums missing observations out, whereas M4's EM fills them in. Both are
  valid EM with the same fixed points.
- Near its symmetric saddle, Baum–Welch can report convergence 70 nats below the
  optimum. Use several random starts.

## 0.4.0 — Milestone 4: learning from data

### Added
- `probgraph.learning`, a new subpackage.
- `Dataset`: validated rows with `None` as the only missing value, stored as integer
  codes. Exact `counts`, `from_samples`, and `with_missing` for MCAR experiments.
- `log_likelihood`: the observed-data log-likelihood, using log-space variable elimination
  for incomplete rows (rows with the same values share one computation).
- `maximum_likelihood`: count ratios. Unseen parent configurations raise and are named;
  `unseen="uniform"` is an explicit opt-in.
- `DirichletPrior` (`uniform`, `bdeu`, `explicit`) and `bayesian_estimate` (posterior mean
  by default, or MAP when every pseudocount is at least 1).
- `log_marginal_likelihood`: the Bayesian score of a structure, computed with `lgamma`.
- `ExpectationMaximisation` and `EMResult`. The E-step uses one junction tree per
  distinct pattern of observed values. The M-step is the MLE, or the posterior mean or
  MAP with a prior. Every run reports its full `log_likelihood` and `log_objective`
  history. Also `expected_counts`.
- `bic`, `family_scores` and `score_structures` (BIC and BDeu).
- Proofs P14–P17 in `docs/mathematics/`, and `examples/learning_traffic.py`.

### Notes
- With a prior, EM's posterior-mean M-step is MAP-EM for $\mathrm{Dir}(\alpha+1)$. The
  log-likelihood itself may then decrease. `log_objective` is the quantity guaranteed
  not to (`em.md` §7).
- `with_missing(seed=s)` draws from its own random stream. If it used the same stream as
  `AncestralSampler(seed=s)`, which draws the same uniforms, the mask would depend on
  the values and would not be MCAR.
- EM assumes values are missing at random. The tests show it unbiased under MAR and
  biased under MNAR.

## 0.3.0 — Milestone 3: message passing

### Added
- `LogFactor`: factors stored as logarithms. Products add, and marginalising uses
  log-sum-exp with the max shift, so nothing overflows, and an all `-inf` slice gives
  `-inf`, not NaN. `-inf` encodes a structural zero. `DiscreteFactor` and `LogFactor`
  now share a `_NamedTable` base class, and mixing the two is a `TypeError`.
- `VariableElimination`: `space="log" | "probability"`, and `log_probability_of_evidence`.
- `MarkovNetwork`: Gibbs distributions with `log_partition_function`,
  `partition_function`, `probability`, `log_probability`, `query`,
  `log_probability_of_evidence` and `separated`. Also
  `BayesianNetwork.to_markov_network()`.
- Chordal graphs: `triangulate`, `is_chordal` and `perfect_elimination_ordering`
  (maximum cardinality search, with every result verified),
  `is_perfect_elimination_ordering`, and `maximal_cliques`.
- `CliqueTree.from_elimination`: clique trees with the running intersection property,
  and family-preserving `assign`.
- `JunctionTree`: Shafer–Shenoy calibration in log space. Every marginal comes from one
  pass of messages, with evidence, `log_probability_of_evidence`, `clique_belief`,
  `message_count` and `calibration_cost`.
- `LoopyBeliefPropagation`: synchronous sum-product on the factor graph, with damping and
  an honest `converged` / `residual` report.
- Proofs P9–P13 in `docs/mathematics/`, and `examples/misconception.py`.

### Changed
- **`VariableElimination` now computes in log space by default** (`space="log"`). In
  probability space, 1,100 observations push P(e) below float64's range. Depending on the
  order of the evidence, v0.2.0 then either raised `ZeroProbabilityEvidenceError` for
  possible evidence or **silently returned a confident, wrong posterior** ([0, 1] where
  the truth is [0.5, 0.5]). Results that did not underflow are unchanged to about 1e-12.
  Pass `space="probability"` for the previous behaviour.

### Fixed
- The strict mypy check under NumPy 2.4, the newest version supporting Python 3.11.

### Notes
- A junction tree is cheaper than repeated variable elimination when evidence is
  downstream (4–10× in the tests). Without evidence, pruned variable elimination is often
  cheaper on sparse models (see `message_passing.md` §9).
- Loopy belief propagation can converge to badly wrong beliefs (the misconception cycle),
  or oscillate forever without damping (a 3×3 grid). Convergence is not accuracy.

## 0.2.0 — Milestone 2: conditional independence, evidence and exact inference

### Added
- `DiscreteFactor`: nonnegative factors over named variables, with product (aligned by
  name), `marginalise`, `reduce`, `normalise`, `total`, `value`, `aligned`, `allclose`,
  `unit()` and `from_cpd()`. Overflow raises instead of producing `inf`.
- `BayesianNetwork.factors()` and `check_evidence()`.
- `VariableElimination`: exact `query` (P(Q | e)) and `probability_of_evidence` (P(e)),
  with `min_fill` / `min_neighbours` / `min_weight` heuristics or an explicit order.
  Barren-node pruning is on by default (`prune_barren`), and requisite-evidence pruning
  is opt-in (`prune_evidence`). Also `elimination_order`, `query_trace`,
  `elimination_cost`, `barren_variables` and `requisite_evidence`.
- `UndirectedGraph` (with `subgraph` and `separated`), `moral_graph` and
  `interaction_graph`; `greedy_order`, `simulate_elimination`, and the
  `EliminationTrace` / `EliminationStep` value objects.
- `DAG.d_separated`, `d_connected_nodes` (Bayes ball), `ancestral_set` and
  `requisite_evidence`.
- `RejectionSampler` and `LikelihoodWeighting`, which return an `ApproximatePosterior`
  (estimate, n, P(e) estimate, effective sample size).
- New errors: `NormalisationError`, `ZeroProbabilityEvidenceError` and
  `InsufficientSamplesError`.
- Proofs P5–P8 in `docs/mathematics/` (factor algebra, variable elimination, elimination
  orders, d-separation, sampling-based inference), and `examples/late_for_work.py`.

### Changed
- `AncestralSampler` now shares its forward-sampling loop with likelihood weighting.
  The random stream, and therefore every seeded result, is unchanged.

### Notes
- With `prune_evidence=True`, evidence that is impossible only because of an irrelevant
  observation returns P(Q | requisite evidence) instead of raising; see
  `d_separation.md` §9. That is why the option is off by default.

## 0.1.0 — Milestone 1: representation, factorisation and sampling

### Added
- `DiscreteVariable`: immutable finite variables with ordered, unique states.
- `DAG`: an acyclic directed graph. Insertions that would create a cycle are rejected
  without changing the graph, and the error reports the cycle. Parent/child and
  ancestor/descendant queries, and a deterministic topological sort (Kahn's algorithm).
- `TabularCPD`: conditional probability tables with the axis convention
  `(|X|, |U_1|, ..., |U_k|)`. Checks shape, finiteness, nonnegativity and per-column
  normalisation (`atol=1e-10`). Storage is read-only and lookups are by parent name.
  `n_free_parameters`.
- `BayesianNetwork`: graph/CPD consistency checks, joint probability by factorisation,
  `sample`, and `n_free_parameters`.
- `AncestralSampler`: exact ancestral sampling with explicit inverse-CDF draws.
  Reproducible and consistent across batches.
- Proofs P1–P4 and algorithm notes in `docs/mathematics/`.
- An example walkthrough in `examples/rain_accident_traffic.py`.
- CI covering lint, strict type checks, tests on Python 3.11–3.13 (Linux and macOS),
  fresh installs of the wheel and sdist, and the oldest supported dependency versions.

### Notes
- Corrects the Milestone 1 spec (v0.3) fixture value: P(Traffic=yes) is 0.3565, not 0.3615.
