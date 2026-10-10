"""Machinery shared by the single-site MCMC samplers: factors, local scores, runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np

from probgraph._assignments import check_partial_assignment
from probgraph.exceptions import UnknownNodeError, ValidationError, ZeroProbabilityEvidenceError
from probgraph.factors import LogFactor
from probgraph.mcmc.chain import Chain
from probgraph.models import BayesianNetwork, MarkovNetwork
from probgraph.sampling.ancestral import cumulative_table, inverse_cdf

Scan = Literal["systematic", "random"]
SCANS: tuple[Scan, ...] = ("systematic", "random")

#: How many forward-sampling attempts initialisation makes (spec ⚑4).
INITIALISATION_ATTEMPTS = 1000


class _SiteSampler:
    """A Markov chain over the unobserved variables that updates one variable at a time.

    The target is π(x) ∝ Π_k φ_k(x, e) with every factor reduced by the evidence
    (gibbs.md §1). Subclasses define one single-site update in ``_update``, which
    consumes ``_uniforms_per_update`` uniforms.
    """

    _uniforms_per_update = 1

    def __init__(
        self,
        model: BayesianNetwork | MarkovNetwork,
        evidence: Mapping[str, str] | None,
        scan: Scan,
        seed: int | None,
    ) -> None:
        if isinstance(model, BayesianNetwork):
            factors = [LogFactor.from_factor(phi) for phi in model.factors()]  # validates
        elif isinstance(model, MarkovNetwork):
            factors = [
                phi if isinstance(phi, LogFactor) else LogFactor.from_factor(phi)
                for phi in model.factors
            ]
        else:
            raise ValidationError(
                f"Expected a BayesianNetwork or MarkovNetwork, got {type(model).__name__}."
            )
        if scan not in SCANS:
            raise ValidationError(f"scan must be one of {SCANS}, got {scan!r}.")
        self._model = model
        self._scan: Scan = scan
        self._rng = np.random.default_rng(seed)
        self._all = {v.name: v for v in model.variables}
        self._evidence = check_partial_assignment(self._all, evidence or {})
        self._free = tuple(v for v in model.variables if v.name not in self._evidence)
        self._column = {v.name: j for j, v in enumerate(self._free)}
        # The update units: one variable each, unless a subclass groups them into blocks.
        self._units: list[tuple[int, ...]] = [(j,) for j in range(len(self._free))]
        self._blanket: dict[str, set[str]] = {v.name: set() for v in model.variables}
        for phi in factors:
            for name in phi.names:
                self._blanket[name] |= set(phi.names) - {name}
        # Each reduced factor: its log table, and for each axis the column of its variable.
        self._factors: list[tuple[np.ndarray, tuple[int, ...]]] = []
        self._touching: list[list[int]] = [[] for _ in self._free]
        self._constant = 0.0  # the factors that the evidence reduces to constants
        self._attempts = 0
        for phi in factors:
            reduced = phi.reduce({n: s for n, s in self._evidence.items() if n in phi.scope})
            axes = tuple(self._column[n] for n in reduced.names)
            if not axes:
                # A constant cancels from every ratio, but -inf means the evidence is impossible.
                self._constant += float(reduced.log_values)
                continue
            k = len(self._factors)
            self._factors.append((np.asarray(reduced.log_values), axes))
            for j in set(axes):
                self._touching[j].append(k)

    # -- public helpers -------------------------------------------------------------

    def markov_blanket(self, variable: str) -> set[str]:
        """Every other variable that shares a factor with ``variable`` (gibbs.md §2)."""
        if variable not in self._blanket:
            raise UnknownNodeError(f"Unknown variable {variable!r}.")
        return set(self._blanket[variable])

    def full_conditional(self, variable: str, state: Mapping[str, str]) -> np.ndarray:
        """P(variable | every other unobserved variable at ``state``, evidence) (Proposition 1)."""
        if variable not in self._all:
            raise UnknownNodeError(f"Unknown variable {variable!r}.")
        if variable in self._evidence:
            raise ValidationError(f"{variable!r} is observed; it has no full conditional.")
        x = self._state_array(state, skip=variable)
        return _softmax(self._local_log_scores(self._column[variable], x))

    # -- running --------------------------------------------------------------------------

    def run(
        self,
        n_samples: int,
        burn_in: int = 0,
        thin: int = 1,
        initial: Mapping[str, str] | None = None,
    ) -> Chain:
        """Apply the kernel ``burn_in + n_samples * thin`` times; keep every ``thin``-th state."""
        _check_count(n_samples, "n_samples", 1)
        _check_count(burn_in, "burn_in", 0)
        _check_count(thin, "thin", 1)
        if not self._free:
            raise ValidationError("Nothing to sample: every variable is observed.")
        x = self._initial_state(initial)
        total = burn_in + n_samples * thin
        per = self._uniforms_per_update
        units = len(self._units)
        width = units * per if self._scan == "systematic" else 1 + per
        uniforms = self._rng.random((total, width))
        kept = np.empty((n_samples, len(self._free)), dtype=np.intp)
        accepted = 0
        for step in range(total):
            u = uniforms[step]
            if self._scan == "systematic":
                for j in range(units):
                    accepted += self._update(j, x, u[j * per : (j + 1) * per])
            else:
                j = min(int(u[0] * units), units - 1)
                accepted += self._update(j, x, u[1:])
            if step >= burn_in and (step - burn_in + 1) % thin == 0:
                kept[(step - burn_in) // thin] = x
        attempts = self._attempts
        self._attempts = 0
        return Chain(self._free, kept, accepted / attempts if attempts else 1.0)

    def _update(self, j: int, x: np.ndarray, u: np.ndarray) -> int:
        """Update unit ``j`` of ``x`` in place using uniforms ``u``; return 1 if accepted.

        Each call that makes a proposal adds 1 to ``self._attempts``.
        """
        raise NotImplementedError

    # -- internals ----------------------------------------------------------------------

    def _local_log_scores(self, j: int, x: np.ndarray) -> np.ndarray:
        """log Π_{φ ∋ X_j} φ(x_j = k, x_-j) for every state k: the unnormalised log conditional."""
        scores = np.zeros(self._free[j].cardinality)
        for k in self._touching[j]:
            table, axes = self._factors[k]
            index = tuple(slice(None) if a == j else int(x[a]) for a in axes)
            scores = scores + table[index]
        return scores

    def _log_score(self, x: np.ndarray) -> float:
        """log π̃(x) = Σ_k log φ_k(x): -inf exactly when x is impossible."""
        terms = [float(table[tuple(int(x[a]) for a in axes)]) for table, axes in self._factors]
        return math.fsum([self._constant, *terms])

    def _draw(self, scores: np.ndarray, u: float) -> int:
        """Inverse-CDF draw from softmax(``scores``); zero-probability states are never drawn."""
        weights = np.exp(scores - np.max(scores))
        return int(inverse_cdf(cumulative_table(weights), np.array([u]))[0])

    def _state_array(self, state: Mapping[str, str], skip: str | None = None) -> np.ndarray:
        needed = [v.name for v in self._free if v.name != skip]
        missing = [n for n in needed if n not in state]
        if missing:
            raise ValidationError(
                f"The state must give every unobserved variable; missing {missing}."
            )
        x = np.zeros(len(self._free), dtype=np.intp)
        for v in self._free:
            if v.name == skip:
                continue
            value = state[v.name]
            if value not in v.states:
                raise ValidationError(f"{value!r} is not a state of {v.name!r} {v.states}.")
            x[self._column[v.name]] = v.states.index(value)
        return x

    def _initial_state(self, initial: Mapping[str, str] | None) -> np.ndarray:
        if initial is not None:
            names = [v.name for v in self._free]
            missing = [n for n in names if n not in initial]
            if missing:
                raise ValidationError(
                    f"initial must give every unobserved variable; missing {missing}."
                )
            x = self._state_array(initial)
            if self._log_score(x) == -math.inf:
                raise ZeroProbabilityEvidenceError(
                    f"The initial state {dict(initial)} has probability 0 given the evidence."
                )
            return x
        for _ in range(INITIALISATION_ATTEMPTS):
            x = self._forward_sample()
            if self._log_score(x) > -math.inf:
                return x
        raise ZeroProbabilityEvidenceError(
            f"No state with positive probability given evidence {self._evidence} was found in "
            f"{INITIALISATION_ATTEMPTS} attempts; the evidence may be impossible. Pass `initial`."
        )

    def _forward_sample(self) -> np.ndarray:
        """Forward sampling with the evidence clamped (BN), or uniform states (MN) (gibbs.md §4)."""
        x = np.zeros(len(self._free), dtype=np.intp)
        if not isinstance(self._model, BayesianNetwork):
            for j, v in enumerate(self._free):
                x[j] = int(self._rng.integers(v.cardinality))
            return x
        values = {n: self._all[n].states.index(s) for n, s in self._evidence.items()}
        for name in self._model.graph.topological_sort():
            if name in self._evidence:
                continue
            cpd = self._model.cpds[name]
            index: tuple[slice | int, ...] = (slice(None), *(values[p] for p in cpd.parent_names))
            column = cpd.values[index]
            values[name] = int(inverse_cdf(cumulative_table(column), self._rng.random(1))[0])
            x[self._column[name]] = values[name]
        return x


def _softmax(scores: np.ndarray) -> np.ndarray:
    weights = np.exp(scores - np.max(scores))
    result: np.ndarray = weights / weights.sum()
    return result


def _check_count(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{name} must be an int >= {minimum}, got {value!r}.")
