"""Task 4 - Adstock & saturation transformations (Member 5)."""
import numpy as np



def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock to monthly advertising spend.

    Parameters
    ----------
    x : array-like
        Monthly advertising spend in chronological order.
    decay : float
        Carryover rate between 0 (inclusive) and 1 (exclusive).

    Returns
    -------
    np.ndarray
        Adstock-adjusted advertising spend.

    Assumption
    ----------
    Initial carryover before the first observation is zero.
    """
    x = np.asarray(x, dtype=float)

    # Validate advertising spend
    if x.ndim != 1:
        raise ValueError("Advertising spend must be a 1D array.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Advertising spend contains invalid values.")

    if np.any(x < 0):
        raise ValueError("Advertising spend cannot be negative.")

    # Validate decay parameter
    if not np.isfinite(decay) or not (0 <= decay < 1):
        raise ValueError("Decay must be between 0 and 1.")

    # Calculate adstock
    out = np.zeros(len(x), dtype=float)
    carry = 0.0

    for t, spend in enumerate(x):
        carry = spend + decay * carry
        out[t] = carry

    return out


def hill_saturation(x: np.ndarray, half_sat: float, slope: float = 1.0) -> np.ndarray:
    """Diminishing returns: x^s / (x^s + K^s), output in [0, 1)."""
    x = np.asarray(x, dtype=float)
    return x ** slope / (x ** slope + half_sat ** slope)


def transform_channel(x, decay, half_sat, slope=1.0):
    """Apply adstock then saturation."""
    return hill_saturation(geometric_adstock(np.asarray(x), decay), half_sat, slope)
