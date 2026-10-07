"""Count validation and the default likelihood confidence threshold."""

from __future__ import annotations

import numpy as np

PROFILE_LIKELIHOOD_DROP_95 = 1.920729410347062


def validate_counts(n_frozen: np.ndarray, n_total: np.ndarray) -> None:
    if np.any(~np.isfinite(n_frozen)) or np.any(~np.isfinite(n_total)):
        raise ValueError("n_frozen and n_total must be finite")
    if np.any(n_total <= 0):
        raise ValueError("n_total must be positive")
    if np.any(n_frozen < 0):
        raise ValueError("n_frozen cannot be negative")
    if np.any(n_frozen > n_total):
        raise ValueError("n_frozen cannot exceed n_total")
