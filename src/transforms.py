"""Task 4 - Adstock & saturation transformations (Member 5)."""
import numpy as np


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Carryover effect: a_t = x_t + decay * a_{t-1}."""
    out = np.zeros(len(x), dtype=float)
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + decay * carry
        out[t] = carry
    return out


def hill_saturation(x: np.ndarray, half_sat: float, slope: float = 1.0) -> np.ndarray:
    """Diminishing returns: x^s / (x^s + K^s), output in [0, 1)."""
    x = np.asarray(x, dtype=float)
    return x ** slope / (x ** slope + half_sat ** slope)


def transform_channel(x, decay, half_sat, slope=1.0):
    """Apply adstock then saturation."""
    return hill_saturation(geometric_adstock(np.asarray(x), decay), half_sat, slope)
