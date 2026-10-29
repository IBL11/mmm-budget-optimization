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



def hill_saturation(
    x: np.ndarray,
    half_sat: float,
    slope: float = 1.0
) -> np.ndarray:
    """
    Apply Hill saturation to non-negative Adstock values.

    S(x) = x^s / (x^s + K^s)

    Parameters
    ----------
    x : array-like
        Adstock-adjusted advertising spend.
    half_sat : float
        K: Adstock value where saturation equals 0.5.
        Must be positive.
    slope : float
        s: Controls the shape of the saturation curve.
        Must be positive.

    Returns
    -------
    np.ndarray
        Saturation values between 0 and 1.
    """
    x = np.asarray(x, dtype=float)

    # Validate input shape and values
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError(
            "Input must contain finite, non-negative values."
        )

    # Validate parameters
    try:
        half_sat = float(half_sat)
        slope = float(slope)
    except (TypeError, ValueError):
        raise ValueError(
            "half_sat and slope must be numeric."
        )

    if not np.isfinite(half_sat) or half_sat <= 0:
        raise ValueError("half_sat must be greater than zero.")

    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("slope must be greater than zero.")

    # Stable calculation of the Hill function
    result = np.zeros_like(x, dtype=float)

    positive = x > 0

    log_ratio = slope * (
        np.log(x[positive]) - np.log(half_sat)
    )

    result[positive] = np.exp(
        -np.logaddexp(0.0, -log_ratio)
    )

    return result


def transform_channel_with_training_k(
    x,
    decay: float,
    train_end: int,
    k_multiplier: float = 1.0,
    slope: float = 1.0
):
    """
    Fit half-saturation K on the training portion only,
    then transform the full chronological spend series.

    Returns
    -------
    saturated : np.ndarray
        Adstock + Hill saturation features.
    half_sat : float
        Derived half-saturation parameter (LKR).
    """
    x = np.asarray(x, dtype=float)

    if (
        not isinstance(train_end, (int, np.integer))
        or not 1 <= train_end <= len(x)
    ):
        raise ValueError("Invalid training window.")

    try:
        k_multiplier = float(k_multiplier)
    except (TypeError, ValueError):
        raise ValueError("k_multiplier must be numeric.")

    if not np.isfinite(k_multiplier) or k_multiplier <= 0:
        raise ValueError("k_multiplier must be positive.")

    # Step 1: Adstock with selected decay
    adstock = geometric_adstock(x, decay)

    # Step 2: Estimate K from training observations only
    train_adstock = adstock[:train_end]
    positive = train_adstock[train_adstock > 0]

    if len(positive) == 0:
        raise ValueError("No positive training Adstock values.")

    half_sat = float(np.median(positive) * k_multiplier)

    if not np.isfinite(half_sat) or half_sat <= 0:
        raise ValueError("Invalid half-saturation value.")

    # Step 3: Apply Hill saturation
    saturated = hill_saturation(
        adstock,
        half_sat=half_sat,
        slope=slope
    )

    return saturated, half_sat
