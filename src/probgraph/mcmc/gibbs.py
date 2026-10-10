"""Gibbs sampling for Bayesian and Markov networks.

The derivations (P24) are in ``docs/mathematics/gibbs.md``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from probgraph.mcmc._sampler import Scan, _SiteSampler
from probgraph.models import BayesianNetwork, MarkovNetwork


class GibbsSampler(_SiteSampler):
    """Samples P(x | e) by redrawing one variable at a time from its full conditional.

    The full conditional uses only the factors that mention the variable, i.e. its
    Markov blanket (gibbs.md §2). ``scan="systematic"`` records one sample per sweep
    over the unobserved variables in model order; ``scan="random"`` records one
    sample per single-variable update at a uniformly chosen variable.

    Invariants (gibbs.md):
        G1  ``full_conditional`` is P(X | x_-X, e) and depends only on the Markov blanket
        G4  evidence never changes; runs are reproducible; burn-in and thinning slice a run
        G5  a chain starts only from a state with positive probability
    """

    def __init__(
        self,
        model: BayesianNetwork | MarkovNetwork,
        evidence: Mapping[str, str] | None = None,
        scan: Scan = "systematic",
        seed: int | None = None,
    ) -> None:
        super().__init__(model, evidence, scan, seed)

    def _update(self, j: int, x: np.ndarray, u: np.ndarray) -> int:
        x[j] = self._draw(self._local_log_scores(j, x), float(u[0]))
        self._attempts += 1
        return 1
