"""Structure scores: BIC and the BDeu Bayesian score.

The derivations (P17) are in ``docs/mathematics/model_selection.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import numpy as np

from probgraph.exceptions import ValidationError
from probgraph.learning.dataset import Dataset
from probgraph.learning.dirichlet import DirichletPrior, _family_log_marginal_likelihood
from probgraph.learning.likelihood import _check_variables, _family_counts, _parents_in_order
from probgraph.models import BayesianNetwork

Score = Literal["bic", "bdeu"]


def bic(structure: BayesianNetwork, data: Dataset) -> float:
    """log L(θ̂) - (d/2) log N, with θ̂ the MLE and d the free parameters (S1).

    ``structure`` provides the graph; the MLE is fitted here and any CPDs on
    ``structure`` are ignored. Unseen parent configurations contribute nothing to
    log L, so they need no special treatment. The data must be complete.
    """
    return math.fsum(family_scores(structure, data, "bic").values())


def family_scores(
    structure: BayesianNetwork,
    data: Dataset,
    score: Score = "bic",
    equivalent_sample_size: float = 1.0,
) -> dict[str, float]:
    """The score's term for each variable's family; they sum to the total (S2).

    Each term depends only on the variable and its parents, so changing one
    variable's parent set changes only that variable's term.
    ``equivalent_sample_size`` is used by ``score="bdeu"`` only.
    """
    if score not in ("bic", "bdeu"):
        raise ValidationError(f"score must be 'bic' or 'bdeu', got {score!r}.")
    _check_variables(structure.variables, data)
    if not data.is_complete:
        raise ValidationError(
            f"Structure scores need complete data, but {data.missing_count} values are missing."
        )
    if score == "bic" and data.n_rows == 0:
        raise ValidationError("BIC needs at least one row (its penalty involves log N).")
    prior = DirichletPrior.bdeu(equivalent_sample_size) if score == "bdeu" else None
    family_counts = _family_counts(structure, data)
    scores = {}
    for variable in structure.variables:
        parents = _parents_in_order(structure, variable.name)
        counts = family_counts[variable.name]
        if prior is not None:
            scores[variable.name] = _family_log_marginal_likelihood(
                prior.pseudocounts(variable, parents), counts
            )
            continue
        n_u = np.broadcast_to(counts.sum(axis=0, keepdims=True), counts.shape)
        seen = counts > 0
        log_l = float(np.sum(counts[seen] * np.log(counts[seen] / n_u[seen])))
        d = (variable.cardinality - 1) * math.prod(p.cardinality for p in parents)
        scores[variable.name] = log_l - d / 2 * math.log(data.n_rows)
    return scores


def score_structures(
    candidates: Iterable[BayesianNetwork],
    data: Dataset,
    score: Score = "bic",
    equivalent_sample_size: float = 1.0,
) -> list[tuple[float, BayesianNetwork]]:
    """Every candidate with its score, best first; ties keep the candidates' order."""
    candidates = list(candidates)
    if not candidates:
        raise ValidationError("score_structures needs at least one candidate.")
    for candidate in candidates:
        if not isinstance(candidate, BayesianNetwork):
            raise ValidationError(f"Expected a BayesianNetwork, got {type(candidate).__name__}.")
    scored = [
        (math.fsum(family_scores(c, data, score, equivalent_sample_size).values()), c)
        for c in candidates
    ]
    return sorted(scored, key=lambda pair: -pair[0])
