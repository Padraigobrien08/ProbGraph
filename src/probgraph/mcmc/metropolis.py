"""Single-site Metropolis–Hastings for Bayesian and Markov networks.

The derivations (P23 §3, §7) are in ``docs/mathematics/mcmc.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from probgraph.mcmc._sampler import Scan, _SiteSampler
from probgraph.models import BayesianNetwork, MarkovNetwork


class MetropolisHastings(_SiteSampler):
    """Propose a different state for one variable uniformly; accept with min(1, π(x')/π(x)).

    The proposal is symmetric, so only the ratio of target probabilities enters,
    and it involves only the factors that mention the variable (its Markov
    blanket). A variable with a single state is never changed. ``scan`` chooses
    the variable as for ``GibbsSampler``; random scan is the default (spec ⚑3).

    Invariants (mcmc.md):
        H1  each site kernel satisfies detailed balance with respect to P(x | e)
        H2  the acceptance probability is min(1, π(x')/π(x)), from the Markov blanket
        H3  on binary variables, its asymptotic variance never exceeds Gibbs's (Peskun)
    """

    _uniforms_per_update = 2

    def __init__(
        self,
        model: BayesianNetwork | MarkovNetwork,
        evidence: Mapping[str, str] | None = None,
        scan: Scan = "random",
        seed: int | None = None,
    ) -> None:
        super().__init__(model, evidence, scan, seed)

    def _update(self, j: int, x: np.ndarray, u: np.ndarray) -> int:
        k = self._free[j].cardinality
        if k == 1:
            return 0
        current = int(x[j])
        proposal = min(int(float(u[0]) * (k - 1)), k - 2)
        proposal += proposal >= current  # skip the current state: uniform over the others
        scores = self._local_log_scores(j, x)
        self._attempts += 1
        log_ratio = float(scores[proposal] - scores[current])
        if log_ratio >= 0 or math.log(max(float(u[1]), 1e-300)) < log_ratio:
            x[j] = proposal
            return 1
        return 0
