"""Blocked Gibbs sampling: draw groups of variables jointly from their exact conditional.

The derivations (P27) are in ``docs/mathematics/blocked_gibbs.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.mcmc._sampler import Scan, _softmax
from probgraph.mcmc.gibbs import GibbsSampler
from probgraph.models import BayesianNetwork, MarkovNetwork


class BlockedGibbsSampler(GibbsSampler):
    """Gibbs sampling in which each block of variables is drawn jointly (P27).

    ``blocks`` are disjoint groups of unobserved variables; every other unobserved
    variable forms a block of its own. A block's joint conditional is the product
    of the factors that touch it, tabulated over the block's states, so a block of
    sizes k_1..k_m costs k_1 ... k_m cells per update.

    Invariants (blocked_gibbs.md):
        B1  each block is drawn exactly from P(x_B | x_-B, e), and the kernel is stationary
        B2  blocking the coupled pair of F2 or F3 makes every sweep an independent draw
    """

    def __init__(
        self,
        model: BayesianNetwork | MarkovNetwork,
        blocks: Sequence[Sequence[str]],
        evidence: Mapping[str, str] | None = None,
        scan: Scan = "systematic",
        seed: int | None = None,
    ) -> None:
        super().__init__(model, evidence, scan, seed)
        if isinstance(blocks, str) or not isinstance(blocks, Sequence):
            raise ValidationError(
                f"blocks must be a sequence of variable-name sequences, got {blocks!r}."
            )
        seen: set[str] = set()
        grouped: list[tuple[int, ...]] = []
        for block in blocks:
            if isinstance(block, str) or not isinstance(block, Sequence) or len(block) == 0:
                raise ValidationError(
                    f"Each block must be a non-empty sequence of names, got {block!r}."
                )
            for name in block:
                if name not in self._all:
                    raise UnknownNodeError(f"Block names unknown variable {name!r}.")
                if name in self._evidence:
                    raise ValidationError(f"Block variable {name!r} is observed.")
                if name in seen:
                    raise ValidationError(f"Variable {name!r} appears in more than one block.")
                seen.add(name)
            grouped.append(tuple(sorted(self._column[n] for n in block)))
        singles = [(j,) for j, v in enumerate(self._free) if v.name not in seen]
        self._units = sorted(grouped + singles, key=lambda unit: unit[0])

    def block_conditional(self, block: Sequence[str], state: Mapping[str, str]) -> np.ndarray:
        """P(x_B | x_-B, e) as a table with one axis per block variable, in the order given."""
        for name in block:
            if name not in self._column:
                raise ValidationError(f"{name!r} is not an unobserved variable.")
        columns = tuple(self._column[n] for n in block)
        x = self._state_array(
            {**state, **{n: self._free[self._column[n]].states[0] for n in block}}
        )
        return _softmax(self._block_log_table(columns, x))

    def _update(self, j: int, x: np.ndarray, u: np.ndarray) -> int:
        unit = self._units[j]
        if len(unit) == 1:
            return super()._update(unit[0], x, u)
        table = self._block_log_table(unit, x)
        flat = self._draw(table.ravel(), float(u[0]))
        x[list(unit)] = np.unravel_index(flat, table.shape)
        self._attempts += 1
        return 1

    def _block_log_table(self, columns: tuple[int, ...], x: np.ndarray) -> np.ndarray:
        """Σ of every touching factor's log table, reduced by x outside the block and
        broadcast over the block's joint states (axes in ``columns`` order)."""
        shape = tuple(self._free[c].cardinality for c in columns)
        total = np.zeros(shape)
        touching = sorted({k for c in columns for k in self._touching[c]})
        for k in touching:
            table, axes = self._factors[k]
            index = tuple(slice(None) if a in columns else int(x[a]) for a in axes)
            reduced = table[index]
            kept = [a for a in axes if a in columns]
            order = sorted(range(len(kept)), key=lambda i: columns.index(kept[i]))
            reduced = np.transpose(reduced, order)
            expanded = [shape[i] if columns[i] in kept else 1 for i in range(len(columns))]
            total = total + reduced.reshape(expanded)
        return total
