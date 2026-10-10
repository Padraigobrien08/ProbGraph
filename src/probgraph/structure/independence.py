"""Conditional-independence tests: G² with exact chi-square tails.

The derivations (P30, Part 1) are in ``docs/mathematics/pc_algorithm.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from probgraph.exceptions import UnknownNodeError, ValidationError
from probgraph.learning.dataset import Dataset


@dataclass(frozen=True)
class IndependenceTest:
    """The outcome of a test of X ⊥ Y | Z."""

    statistic: float  # G² = 2 N Î(X; Y | Z), in nats
    dof: int
    p_value: float


def chi_square_survival(x: float, dof: int) -> float:
    """P(χ²_dof > x) exactly, for integer dof >= 1 (pc_algorithm.md §3)."""
    if isinstance(dof, bool) or not isinstance(dof, int) or dof < 1:
        raise ValidationError(f"dof must be a positive int, got {dof!r}.")
    if not math.isfinite(x) and x != math.inf:
        raise ValidationError(f"x must be a number, got {x!r}.")
    if x <= 0:
        return 1.0
    if x == math.inf:
        return 0.0
    return math.exp(_log_survival(float(x), dof))


def g_squared_test(data: Dataset, x: str, y: str, given: Sequence[str] = ()) -> IndependenceTest:
    """The G² test of X ⊥ Y | given, with stratum-adjusted degrees of freedom (§1–2).

    Uses the rows in which X, Y and every conditioning variable are observed. With no
    degrees of freedom left (for example, a constant variable), returns p = 1.
    """
    if not isinstance(data, Dataset):
        raise ValidationError(f"Expected a Dataset, got {type(data).__name__}.")
    given = list(given)
    names = [x, y, *given]
    for name in names:
        if name not in data.names:
            raise UnknownNodeError(f"Unknown variable {name!r}.")
    if len(set(names)) != len(names):
        raise ValidationError(f"The test variables must be distinct, got {names}.")
    counts = data.counts(names).values
    table = counts.reshape(counts.shape[0], counts.shape[1], -1)  # (|X|, |Y|, strata)
    statistic = 0.0
    dof = 0
    for k in range(table.shape[2]):
        stratum = table[:, :, k]
        total = float(stratum.sum())
        if total == 0:
            continue
        rows, columns = stratum.sum(axis=1), stratum.sum(axis=0)
        dof += (int(np.count_nonzero(rows)) - 1) * (int(np.count_nonzero(columns)) - 1)
        seen = stratum > 0
        expected = np.outer(rows, columns)[seen] / total
        statistic += 2.0 * float(np.sum(stratum[seen] * np.log(stratum[seen] / expected)))
    statistic = max(statistic, 0.0)  # rounding can leave -1e-15 for a factorising table
    p_value = 1.0 if dof == 0 else chi_square_survival(statistic, dof)
    return IndependenceTest(statistic, dof, p_value)


def _log_survival(x: float, dof: int) -> float:
    y = x / 2
    log_y = math.log(y)
    if dof % 2 == 0:
        terms = [-y + j * log_y - math.lgamma(j + 1) for j in range(dof // 2)]
    else:
        terms = [_log_erfc(math.sqrt(y))]
        terms += [-y + (j + 0.5) * log_y - math.lgamma(j + 1.5) for j in range((dof - 1) // 2)]
    top = max(terms)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(t - top) for t in terms))


def _log_erfc(z: float) -> float:
    """log erfc(z), with the asymptotic expansion where erfc underflows (z > 25)."""
    if z <= 25.0:
        return math.log(math.erfc(z))
    inverse = 1.0 / (z * z)
    series = 1 - inverse / 2 + 3 * inverse**2 / 4 - 15 * inverse**3 / 8
    return -z * z - math.log(z * math.sqrt(math.pi)) + math.log(series)
