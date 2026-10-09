"""Shared validation of numeric table input."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from probgraph.exceptions import ValidationError


def as_float_array(values: ArrayLike, context: str) -> np.ndarray:
    """Return a private float64 copy of ``values``.

    Raises ``ValidationError`` for ragged or non-real input (bool, str, complex,
    object). ``context`` prefixes the error message, e.g. ``"CPD for 'Rain'"``.
    """
    try:
        raw = np.asarray(values)
    except ValueError as exc:  # ragged nested sequences
        raise ValidationError(f"{context}: values are not a regular array ({exc}).") from None
    if raw.dtype.kind not in "iuf":
        raise ValidationError(f"{context}: values must be real numbers, got dtype {raw.dtype}.")
    # Always copy, so that later changes to the caller's array cannot reach the table.
    return np.array(raw, dtype=np.float64, copy=True)
