"""Exact MCMC transition matrices and asymptotic variances: a test oracle (mcmc.md §5).

Everything here is built from the full joint table of a small model: the sampler's own code
is never used. States are the assignments of the unobserved variables with π > 0.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping

import numpy as np
from support import gibbs_table, joint_table

from probgraph import BayesianNetwork, MarkovNetwork


class ExactChain:
    """The support of π(x) = P(x | e), and functions to build kernels on it."""

    def __init__(
        self, model: BayesianNetwork | MarkovNetwork, evidence: Mapping[str, str] | None = None
    ):
        evidence = dict(evidence or {})
        variables = model.variables
        table = (
            joint_table(model)
            if isinstance(model, BayesianNetwork)
            else gibbs_table(variables, model.factors)
        )
        index = tuple(
            v.states.index(evidence[v.name]) if v.name in evidence else slice(None)
            for v in variables
        )
        self.free = [v for v in variables if v.name not in evidence]
        reduced = table[index]  # axes: the free variables, in model order
        cells = list(itertools.product(*(range(v.cardinality) for v in self.free)))
        weights = np.array([reduced[c] for c in cells], dtype=float)
        self.states = [c for c, w in zip(cells, weights, strict=True) if w > 0]
        self.position = {s: i for i, s in enumerate(self.states)}
        positive = weights[weights > 0]
        self.pi = positive / positive.sum()
        self._weight = {s: w for s, w in zip(self.states, positive, strict=True)}

    @property
    def size(self) -> int:
        return len(self.states)

    def conditional(self, state: tuple[int, ...], j: int) -> np.ndarray:
        """π(x_j = k | x_-j) for every k, from the joint table."""
        k = self.free[j].cardinality
        values = np.array(
            [self._weight.get((*state[:j], v, *state[j + 1 :]), 0.0) for v in range(k)]
        )
        return values / values.sum()

    def site_kernel(self, j: int) -> np.ndarray:
        """K_j: redraw variable j from its full conditional (mcmc.md §2)."""
        kernel = np.zeros((self.size, self.size))
        for s in self.states:
            for v, p in enumerate(self.conditional(s, j)):
                if p > 0:
                    kernel[self.position[s], self.position[(*s[:j], v, *s[j + 1 :])]] += p
        return kernel

    def metropolis_site_kernel(self, j: int) -> np.ndarray:
        """Propose each other state of variable j with probability 1/(K-1); accept with
        min(1, π(x')/π(x)); otherwise stay (mcmc.md §3, §7)."""
        k = self.free[j].cardinality
        kernel = np.zeros((self.size, self.size))
        for s in self.states:
            here = self.position[s]
            if k == 1:
                kernel[here, here] = 1.0
                continue
            for v in range(k):
                if v == s[j]:
                    continue
                target = (*s[:j], v, *s[j + 1 :])
                accept = min(1.0, self._weight.get(target, 0.0) / self._weight[s])
                kernel[here, here] += (1 - accept) / (k - 1)
                if accept > 0:
                    kernel[here, self.position[target]] += accept / (k - 1)
        return kernel

    def block_kernel(self, columns: tuple[int, ...]) -> np.ndarray:
        """K_B: redraw the variables in ``columns`` jointly from π(x_B | x_-B) (P27)."""
        kernel = np.zeros((self.size, self.size))
        for s in self.states:
            rest = [v for j, v in enumerate(s) if j not in columns]
            matches = [
                t for t in self.states if [v for j, v in enumerate(t) if j not in columns] == rest
            ]
            weights = np.array([self._weight[t] for t in matches])
            for t, w in zip(matches, weights / weights.sum(), strict=True):
                kernel[self.position[s], self.position[t]] += w
        return kernel

    def systematic(self, metropolis: bool = False) -> np.ndarray:
        site = self.metropolis_site_kernel if metropolis else self.site_kernel
        kernel = np.eye(self.size)
        for j in range(len(self.free)):
            kernel = kernel @ site(j)
        return kernel

    def random_scan(self, metropolis: bool = False) -> np.ndarray:
        site = self.metropolis_site_kernel if metropolis else self.site_kernel
        return sum(site(j) for j in range(len(self.free))) / len(self.free)

    def acceptance_probability(self) -> float:
        """E_π of the random-scan MH acceptance probability, over proposals that exist."""
        total, weight = 0.0, 0.0
        for s, p in zip(self.states, self.pi, strict=True):
            for j, v in enumerate(self.free):
                if v.cardinality == 1:
                    continue
                for value in range(v.cardinality):
                    if value != s[j]:
                        target = (*s[:j], value, *s[j + 1 :])
                        a = min(1.0, self._weight.get(target, 0.0) / self._weight[s])
                        total += p * a / (v.cardinality - 1)
                weight += p
        return total / weight

    def function(self, f: Callable[[dict[str, int]], float]) -> np.ndarray:
        """f evaluated at every support state (given as {name: state index})."""
        return np.array(
            [f({v.name: s[j] for j, v in enumerate(self.free)}) for s in self.states], dtype=float
        )

    def indicator(self, name: str, state: str) -> np.ndarray:
        v = next(u for u in self.free if u.name == name)
        k = v.states.index(state)
        return self.function(lambda x: float(x[name] == k))


def asymptotic_variance(kernel: np.ndarray, pi: np.ndarray, f: np.ndarray) -> float:
    """σ²_asym = 2 <f̄, Z f̄>_π - <f̄, f̄>_π with Z = (I - K + Π)^-1 (Proposition 3)."""
    centred = f - pi @ f
    fundamental = np.linalg.solve(
        np.eye(len(pi)) - kernel + np.outer(np.ones(len(pi)), pi), centred
    )
    return float(2 * np.sum(pi * centred * fundamental) - np.sum(pi * centred * centred))


def second_eigenvalue(kernel: np.ndarray) -> float:
    """The second largest eigenvalue modulus."""
    moduli = sorted(np.abs(np.linalg.eigvals(kernel)), reverse=True)
    return float(moduli[1]) if len(moduli) > 1 else 0.0
