"""Ancestral (forward) sampling from a Bayesian network.

The correctness proof (P3) is in ``docs/mathematics/ancestral_sampling.md``.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from probgraph.exceptions import ValidationError

if TYPE_CHECKING:
    from probgraph.models import BayesianNetwork


def cumulative_table(values: np.ndarray) -> np.ndarray:
    """Return the CDF of every column of a CPD table, taken along axis 0.

    Each column is divided by its own total, so the last entry is exactly 1.0.
    This keeps the inverse-CDF lookup in range even when a column sums to
    1 minus a rounding error.
    """
    cumulative = np.cumsum(values, axis=0)
    cumulative /= cumulative[-1]
    cumulative.setflags(write=False)
    return cumulative


def inverse_cdf(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return the state index whose interval [F_{k-1}, F_k) contains each uniform draw ``u``.

    ``cdf`` has shape ``(K,)`` or ``(K, n)``. In the second case column ``j``
    is the CDF used for ``u[j]``. The index is the number of CDF entries that
    are <= u. States with zero probability have empty intervals and are never
    returned.
    """
    cdf = cdf if cdf.ndim > 1 else cdf[:, np.newaxis]
    counts: np.ndarray = np.count_nonzero(cdf <= np.asarray(u)[np.newaxis, ...], axis=0)
    return counts


class _ForwardPlan:
    """A snapshot of a validated model, used to sample forward in topological order.

    ``run`` implements ancestral sampling (P3). With ``clamp`` it implements the
    clamped sampling of likelihood weighting (sampling_inference.md §2).
    """

    def __init__(self, model: BayesianNetwork) -> None:
        model.validate()
        self.variables = model.variables
        self.column = {v.name: i for i, v in enumerate(self.variables)}
        self.order = tuple(model.graph.topological_sort())
        cpds = model.cpds
        self.parents = {name: cpds[name].parent_names for name in self.order}
        self.cdfs = {name: cumulative_table(cpds[name].values) for name in self.order}
        self.tables = {name: cpds[name].values for name in self.order}

    def run(
        self, n: int, rng: np.random.Generator, clamp: Mapping[str, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, weights)``.

        ``indices`` is an ``(n, d)`` array of state indices, with columns in model
        variable order. A clamped variable is fixed at its given index, and each
        sample's weight is multiplied by p(clamped value | sampled parents). Without
        clamping, every weight is 1. The full ``(n, d)`` block of uniforms is always
        drawn, so the random stream does not depend on what is clamped.
        """
        clamp = clamp or {}
        uniforms = rng.random((n, len(self.order)))
        indices = np.zeros((n, len(self.variables)), dtype=np.intp)
        weights = np.ones(n)
        for step, name in enumerate(self.order):
            # Parent columns were filled in by earlier steps (G2 / S2).
            parent_indices = [indices[:, self.column[p]] for p in self.parents[name]]
            if name in clamp:
                k = clamp[name]
                indices[:, self.column[name]] = k
                weights *= self.tables[name][(k, *parent_indices)]
            else:
                cdf = self.cdfs[name]
                selection: tuple[slice | np.ndarray, ...] = (slice(None), *parent_indices)
                columns = cdf[selection] if parent_indices else cdf
                indices[:, self.column[name]] = inverse_cdf(columns, uniforms[:, step])
        return indices, weights


class AncestralSampler:
    """Draws i.i.d. joint samples from a validated ``BayesianNetwork``.

    The network is validated, and its CPDs and topological order are recorded,
    when the sampler is created. Later ``add_cpd`` calls on the model do not
    affect an existing sampler.

    Each call to ``sample(n)`` draws an ``(n, d)`` block of uniforms from a
    dedicated PCG64 generator. Row ``s`` is used only by sample ``s`` (S5), and
    column ``i`` only by the ``i``-th node in topological order. As a result,
    ``sample(a) + sample(b)`` equals ``sample(a + b)`` for the same seed.
    """

    def __init__(self, model: BayesianNetwork, seed: int | None = None) -> None:
        self._plan = _ForwardPlan(model)
        self._rng = np.random.default_rng(seed)

    @property
    def order(self) -> tuple[str, ...]:
        """The topological order in which nodes are sampled."""
        return self._plan.order

    def sample(self, n: int) -> list[dict[str, str]]:
        """Draw ``n`` independent joint samples, each a ``{variable: state}`` dict."""
        indices = self._sample_indices(_as_count(n))
        names = [v.name for v in self._plan.variables]
        states = [v.states for v in self._plan.variables]
        return [
            {name: domain[i] for name, domain, i in zip(names, states, row, strict=True)}
            for row in indices.tolist()
        ]

    def _sample_indices(self, n: int) -> np.ndarray:
        """Return an ``(n, d)`` array of state indices, with columns in model variable order."""
        return self._plan.run(n, self._rng)[0]


def _as_count(n: int) -> int:
    if isinstance(n, bool):
        raise ValidationError(f"Sample size must be a non-negative integer, got {n!r}.")
    try:
        count = operator.index(n)
    except TypeError:
        raise ValidationError(f"Sample size must be a non-negative integer, got {n!r}.") from None
    if count < 0:
        raise ValidationError(f"Sample size must be non-negative, got {count}.")
    return count
