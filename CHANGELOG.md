# Changelog

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
