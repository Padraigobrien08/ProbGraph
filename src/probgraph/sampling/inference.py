"""Approximate inference by sampling: rejection sampling and likelihood weighting.

The proofs (P8) are in ``docs/mathematics/sampling_inference.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from probgraph._assignments import check_query_and_evidence
from probgraph.exceptions import InsufficientSamplesError, ValidationError
from probgraph.factors import DiscreteFactor
from probgraph.sampling.ancestral import _as_count, _ForwardPlan

if TYPE_CHECKING:
    from probgraph.models import BayesianNetwork


@dataclass(frozen=True, eq=False)
class ApproximatePosterior:
    """A sampling estimate of P(query | evidence).

    Attributes:
        estimate: the estimated posterior, a normalised factor over the query
            variables in the order requested.
        n: the number of samples drawn.
        evidence_estimate: an unbiased estimate of P(e). For rejection sampling it
            is the acceptance rate; for likelihood weighting, the mean weight.
        effective_sample_size: for rejection sampling, the number of accepted
            samples; for likelihood weighting, Kish's (Σw)² / Σw².
    """

    estimate: DiscreteFactor
    n: int
    evidence_estimate: float
    effective_sample_size: float


class _SamplingInference:
    def __init__(self, model: BayesianNetwork) -> None:
        self._plan = _ForwardPlan(model)  # validates and snapshots the model
        self._variables = {v.name: v for v in self._plan.variables}

    def _prepare(
        self, variables: Sequence[str], evidence: Mapping[str, str] | None, n: int
    ) -> tuple[list[str], dict[str, int], int]:
        query, observed = check_query_and_evidence(self._variables, variables, evidence)
        count = _as_count(n)
        if count == 0:
            raise ValidationError("At least one sample is needed to estimate a posterior.")
        clamp = {name: self._variables[name].index_of(state) for name, state in observed.items()}
        return query, clamp, count

    def _histogram(self, query: list[str], indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The weighted counts of each query configuration, with axes in query order."""
        shape = tuple(self._variables[name].cardinality for name in query)
        totals = np.zeros(shape)
        np.add.at(totals, tuple(indices[:, self._plan.column[name]] for name in query), weights)
        return totals

    def _factor(self, query: list[str], table: np.ndarray) -> DiscreteFactor:
        return DiscreteFactor([self._variables[name] for name in query], table)


class RejectionSampler(_SamplingInference):
    """Estimates P(Q | e) from the ancestral samples that agree with the evidence.

    The accepted samples are exact i.i.d. draws from P(· | e) (A1). The expected
    number accepted is n·P(e), so rare evidence makes this method wasteful.
    """

    def query(
        self,
        variables: Sequence[str],
        evidence: Mapping[str, str] | None = None,
        n: int = 10_000,
        seed: int | None = None,
    ) -> ApproximatePosterior:
        query, clamp, count = self._prepare(variables, evidence, n)
        indices, _ = self._plan.run(count, np.random.default_rng(seed))
        accept = np.ones(count, dtype=bool)
        for name, k in clamp.items():
            accept &= indices[:, self._plan.column[name]] == k
        accepted = int(accept.sum())
        if accepted == 0:
            raise InsufficientSamplesError(
                f"No samples were accepted out of {count}; the evidence may be rare or impossible."
            )
        counts = self._histogram(query, indices[accept], np.ones(accepted))
        return ApproximatePosterior(
            estimate=self._factor(query, counts / accepted),
            n=count,
            evidence_estimate=accepted / count,
            effective_sample_size=float(accepted),
        )


class LikelihoodWeighting(_SamplingInference):
    """Estimates P(Q | e) by clamping the evidence and weighting each sample.

    Each weight is w = Π_{i∈E} p(e_i | pa_i), in [0, 1]. The mean weight is an
    unbiased estimate of P(e) (A2). The posterior is a ratio of weighted sums:
    consistent, but biased for finite n (A3).
    """

    def query(
        self,
        variables: Sequence[str],
        evidence: Mapping[str, str] | None = None,
        n: int = 10_000,
        seed: int | None = None,
    ) -> ApproximatePosterior:
        query, clamp, count = self._prepare(variables, evidence, n)
        indices, weights = self._plan.run(count, np.random.default_rng(seed), clamp)
        total = float(weights.sum())
        if total == 0.0:
            raise InsufficientSamplesError(
                f"Every sample has weight 0 out of {count}; the evidence may be impossible."
            )
        weighted = self._histogram(query, indices, weights)
        return ApproximatePosterior(
            estimate=self._factor(query, weighted / total),
            n=count,
            evidence_estimate=total / count,
            effective_sample_size=total**2 / float((weights**2).sum()),
        )
