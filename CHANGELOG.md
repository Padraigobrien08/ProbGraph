# Changelog

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
