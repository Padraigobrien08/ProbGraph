"""Convergence and efficiency diagnostics for MCMC output.

The derivations (P25) are in ``docs/mathematics/diagnostics.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from probgraph.exceptions import ValidationError


def autocorrelation(x: ArrayLike, max_lag: int) -> np.ndarray:
    """ρ̂_0..ρ̂_max_lag, with γ̂_k = (1/n) Σ (x_i - x̄)(x_i+k - x̄) (diagnostics.md §1).

    Computed with the FFT. A constant series gives ``nan`` at every lag.
    """
    series = _series(x)
    n = len(series)
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not 0 <= max_lag < n:
        raise ValidationError(f"max_lag must be an int in [0, {n - 1}], got {max_lag!r}.")
    autocovariance = _autocovariance(series)
    if autocovariance[0] == 0.0:
        return np.full(max_lag + 1, math.nan)
    result: np.ndarray = autocovariance[: max_lag + 1] / autocovariance[0]
    return result


def integrated_autocorrelation_time(x: ArrayLike) -> float:
    """τ̂ = 1 + 2 Σ ρ̂_k by Geyer's initial monotone sequence (diagnostics.md §2).

    ``nan`` for a constant series.
    """
    series = _series(x)
    autocovariance = _autocovariance(series)
    if autocovariance[0] == 0.0:
        return math.nan
    rho = autocovariance / autocovariance[0]
    total = 0.0
    previous = math.inf
    for m in range(len(rho) // 2):
        pair = float(rho[2 * m] + rho[2 * m + 1])
        if pair <= 0:
            break
        pair = min(pair, previous)  # the initial monotone sequence
        total += pair
        previous = pair
    return -1.0 + 2.0 * total


def effective_sample_size(x: ArrayLike) -> float:
    """n / τ̂; ``nan`` for a constant series."""
    series = _series(x)
    return len(series) / integrated_autocorrelation_time(series)


def monte_carlo_standard_error(x: ArrayLike) -> float:
    """sqrt(γ̂_0 τ̂ / n): the standard error of the sample mean, allowing for correlation."""
    series = _series(x)
    tau = integrated_autocorrelation_time(series)
    return math.sqrt(float(np.var(series)) * tau / len(series))


def split_r_hat(chains: Sequence[ArrayLike]) -> float:
    """Gelman–Rubin R̂ on the halves of every chain (diagnostics.md §4).

    All chains must have the same length (at least 4). Returns ``inf`` when the
    halves are each constant but disagree, and ``nan`` when every sample is equal.
    """
    if isinstance(chains, np.ndarray) and chains.ndim == 1:
        chains = [chains]
    arrays = [_series(c) for c in chains]
    if not arrays:
        raise ValidationError("split_r_hat needs at least one chain.")
    length = len(arrays[0])
    if any(len(a) != length for a in arrays):
        raise ValidationError(
            f"All chains must have the same length, got {[len(a) for a in arrays]}."
        )
    if length < 4:
        raise ValidationError(f"Each chain needs at least 4 samples, got {length}.")
    half = length // 2
    halves = [part for a in arrays for part in (a[:half], a[length - half :])]
    means = np.array([h.mean() for h in halves])
    within = float(np.mean([h.var(ddof=1) for h in halves]))
    between = half * float(means.var(ddof=1))
    if within == 0.0:
        return math.inf if between > 0 else math.nan
    pooled = (half - 1) / half * within + between / half
    return math.sqrt(pooled / within)


def _series(x: ArrayLike) -> np.ndarray:
    try:
        series = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValidationError(f"Expected a one-dimensional numeric series, got {x!r}.") from None
    if series.ndim != 1:
        raise ValidationError(f"Expected a one-dimensional series, got shape {series.shape}.")
    if len(series) < 2:
        raise ValidationError(f"A series needs at least 2 values, got {len(series)}.")
    if not np.isfinite(series).all():
        raise ValidationError("The series must be finite.")
    return series


def _autocovariance(series: np.ndarray) -> np.ndarray:
    """γ̂_k for k = 0..n-1 via the FFT, zero-padded so that no lag wraps around."""
    n = len(series)
    centred = series - series.mean()
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    result: np.ndarray = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n] / n
    if float(np.max(np.abs(centred))) == 0.0:
        result[:] = 0.0
    return result
