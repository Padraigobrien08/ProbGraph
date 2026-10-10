"""Particle filtering (sequential importance sampling with resampling) for DBNs and HMMs.

The derivations (P26) are in ``docs/mathematics/particle_filtering.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from probgraph._assignments import check_partial_assignment, check_query
from probgraph.distributions import TabularCPD
from probgraph.exceptions import ValidationError, ZeroProbabilityEvidenceError
from probgraph.graphs import DAG
from probgraph.sampling.ancestral import cumulative_table, inverse_cdf
from probgraph.temporal.dbn import _PREVIOUS, DynamicBayesianNetwork
from probgraph.temporal.hmm import HiddenMarkovModel

Resampling = Literal["systematic", "multinomial", "adaptive", "none"]
RESAMPLING: tuple[Resampling, ...] = ("systematic", "multinomial", "adaptive", "none")


class ParticleFilter:
    """The bootstrap particle filter on a 2-TBN, run over all the evidence when built.

    ``evidence[t]`` maps template variable names to observed states in slice t + 1
    (an empty mapping means nothing is observed). Particles are propagated through
    the transition model, weighted by the probability of the evidence, and
    resampled by ``resampling`` (default: systematic, every step; "adaptive"
    resamples only when the ESS falls below N/2).

    Invariants (particle_filtering.md):
        P1  exp(log_likelihood) is an unbiased estimate of P(e_1:T), for every N
        P4  ``effective_sample_sizes`` records the ESS of the weights before resampling
        P5  resampling gives each particle N W̄_i offspring in expectation
    """

    __slots__ = (
        "_dead_at",
        "_effective",
        "_log_likelihood",
        "_model",
        "_names",
        "_particles",
        "_template",
    )

    def __init__(
        self,
        model: DynamicBayesianNetwork,
        evidence: Sequence[Mapping[str, str]],
        n_particles: int,
        resampling: Resampling = "systematic",
        seed: int | None = None,
    ) -> None:
        if not isinstance(model, DynamicBayesianNetwork):
            raise ValidationError(
                f"Expected a DynamicBayesianNetwork, got {type(model).__name__}; "
                "use ParticleFilter.for_hmm for a HiddenMarkovModel."
            )
        if isinstance(n_particles, bool) or not isinstance(n_particles, int) or n_particles < 1:
            raise ValidationError(f"n_particles must be a positive int, got {n_particles!r}.")
        if resampling not in RESAMPLING:
            raise ValidationError(f"resampling must be one of {RESAMPLING}, got {resampling!r}.")
        if isinstance(evidence, (str, bytes, Mapping)) or not isinstance(evidence, Sequence):
            raise ValidationError("evidence must be a sequence of mappings, one per slice.")
        if len(evidence) == 0:
            raise ValidationError("evidence must have at least one slice.")
        template = {v.name: v for v in model.variables}
        slices = [
            check_partial_assignment(template, e, f"Evidence for slice {t + 1}")
            for t, e in enumerate(evidence)
        ]
        self._model = model
        self._names = tuple(template)
        self._run(model, slices, n_particles, resampling, np.random.default_rng(seed))

    @classmethod
    def for_hmm(
        cls,
        model: HiddenMarkovModel,
        observations: Sequence[str | None],
        n_particles: int,
        **kwargs: Any,
    ) -> ParticleFilter:
        """Filter an HMM written as a 2-TBN (``DynamicBayesianNetwork.from_hmm``)."""
        evidence = [{} if y is None else {model.observed.name: y} for y in observations]
        return cls(DynamicBayesianNetwork.from_hmm(model), evidence, n_particles, **kwargs)

    # -- results ------------------------------------------------------------------------

    @property
    def log_likelihood(self) -> float:
        """log P̂(e_1:T) = Σ_t log ĉ_t; -inf if every particle got weight 0 at some step."""
        return self._log_likelihood

    @property
    def effective_sample_sizes(self) -> np.ndarray:
        """(T,): 1 / Σ W̄² of the weights after weighting, before resampling."""
        return self._effective

    def filtered(self, variable: str) -> np.ndarray:
        """(T, |X|): the weighted particle estimate of P(X_t | e_1:t)."""
        result: np.ndarray = self.filtered_joint([variable])
        return result

    def filtered_joint(self, variables: Sequence[str]) -> np.ndarray:
        """(T, |X_1|, ..., |X_k|): the weighted particle estimate of P(X_1t, ..., X_kt | e_1:t).

        Particles are joint samples of the whole slice, so correlations between
        variables (entanglement, dbn.md §4) are represented, not just marginals.
        """
        names = check_query(self._template, variables)
        if self._dead_at is not None:
            raise ZeroProbabilityEvidenceError(
                f"Every particle had weight 0 at slice {self._dead_at + 1}, so the filtered "
                "estimates are undefined from there; use more particles."
            )
        columns = [self._names.index(n) for n in names]
        shape = tuple(self._template[n].cardinality for n in names)
        result = np.zeros((len(self._particles), *shape))
        for t, (states, weights) in enumerate(self._particles):
            flat = np.ravel_multi_index(tuple(states[:, j] for j in columns), shape)
            counts = np.bincount(flat, weights=weights, minlength=int(np.prod(shape, dtype=int)))
            result[t] = counts.reshape(shape)
        result.setflags(write=False)
        return result

    # -- the algorithm --------------------------------------------------------------------

    def _run(
        self,
        model: DynamicBayesianNetwork,
        slices: list[dict[str, str]],
        n: int,
        resampling: Resampling,
        rng: np.random.Generator,
    ) -> None:
        template = {v.name: v for v in model.variables}
        column = {name: j for j, name in enumerate(self._names)}
        initial = [model.initial.cpds[name] for name in model.initial.graph.topological_sort()]
        by_child = {cpd.variable.name: cpd for cpd in model.transition}
        intra = DAG(
            nodes=list(self._names),
            edges=[
                (p, c) for c, cpd in by_child.items() for p in cpd.parent_names if p in template
            ],
        )
        transition = [by_child[name] for name in intra.topological_sort()]

        length = len(slices)
        self._template = template
        self._particles: list[tuple[np.ndarray, np.ndarray]] = []
        self._effective = np.zeros(length)
        self._dead_at: int | None = None
        log_likelihood = 0.0
        weights = np.full(n, 1.0 / n)  # W̄_t-1, normalised
        previous = np.zeros((n, len(self._names)), dtype=np.intp)
        for t, observed in enumerate(slices):
            current = np.zeros_like(previous)
            log_w = np.zeros(n)
            for cpd in initial if t == 0 else transition:
                name = cpd.variable.name
                parents = tuple(
                    _parent_values(p, current, previous, column) for p in cpd.parent_names
                )
                if name in observed:
                    k = template[name].states.index(observed[name])
                    current[:, column[name]] = k
                    with np.errstate(divide="ignore"):
                        log_w += np.log(_column_values(cpd, parents, n)[k])
                else:
                    columns = cumulative_table(_column_values(cpd, parents, n))
                    current[:, column[name]] = inverse_cdf(columns, rng.random(n))
            top = float(np.max(log_w))
            if top == -math.inf:
                self._dead_at = t
                self._log_likelihood = -math.inf
                self._finish()
                return
            scaled = weights * np.exp(log_w - top)  # W̄_t-1 w_t, up to the factor e^top
            total = float(scaled.sum())
            log_likelihood += top + math.log(total)  # log ĉ_t
            weights = scaled / total
            self._effective[t] = 1.0 / float(np.sum(weights**2))
            self._particles.append((current, weights))
            if resampling == "none" or (resampling == "adaptive" and self._effective[t] >= n / 2):
                previous = current
                continue
            chosen = resample(
                weights, rng, "multinomial" if resampling == "multinomial" else "systematic"
            )
            previous = current[chosen]
            weights = np.full(n, 1.0 / n)
        self._log_likelihood = log_likelihood
        self._finish()

    def _finish(self) -> None:
        self._effective.setflags(write=False)


def resample(
    weights: np.ndarray, rng: np.random.Generator, scheme: Literal["systematic", "multinomial"]
) -> np.ndarray:
    """Indices of N resampled particles; particle i is chosen N W̄_i times in expectation (§3).

    Systematic: one uniform u in [0, 1/N) and the points u + k/N, so each particle
    gets ⌊N W̄_i⌋ or ⌈N W̄_i⌉ copies. Multinomial: N independent draws.
    """
    n = len(weights)
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    if scheme == "systematic":
        points = (rng.random() + np.arange(n)) / n
    elif scheme == "multinomial":
        points = rng.random(n)
    else:
        raise ValidationError(f"scheme must be 'systematic' or 'multinomial', got {scheme!r}.")
    chosen: np.ndarray = np.minimum(np.searchsorted(cumulative, points, side="right"), n - 1)
    return chosen


def _parent_values(
    name: str, current: np.ndarray, previous: np.ndarray, column: dict[str, int]
) -> np.ndarray:
    if name.endswith(_PREVIOUS):
        return previous[:, column[name.removesuffix(_PREVIOUS)]]
    return current[:, column[name]]


def _column_values(cpd: TabularCPD, parents: tuple[np.ndarray, ...], n: int) -> np.ndarray:
    """(|X|, N): each particle's CPD column, given its parents' values."""
    if not parents:
        return np.repeat(cpd.values[:, np.newaxis], n, axis=1)
    index: tuple[slice | np.ndarray, ...] = (slice(None), *parents)
    values: np.ndarray = cpd.values[index]
    return values
