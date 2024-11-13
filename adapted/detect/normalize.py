"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import logging
from typing import Tuple

import numpy as np


def med_mad(signal: np.ndarray, with_nan: bool = False) -> Tuple[float, float]:
    if with_nan:
        med = np.nanmedian(signal)
        mad = np.nanmedian(np.abs(signal - med))
    else:
        med = np.median(signal)
        mad = np.median(np.abs(signal - med))
    return float(med), float(mad)


def clip_signal(
    signal: np.ndarray, outlier_thresh: float, med: float, mad: float
) -> np.ndarray:
    return np.clip(signal, med - (mad * outlier_thresh), med + (mad * outlier_thresh))


def normalize_signal(
    signal: np.ndarray,
    outlier_thresh: float = 5.0,
    with_nan: bool = False,
) -> np.ndarray:
    """
    Normalize a signal by applying MAD-based winsorization followed by MAD normalization.

    Parameters
    ----------
    signal : np.ndarray
        The signal to normalize.
    outlier_thresh : float, optional
        Threshold for winsorizing outliers, by default 5.0.

    Returns
    -------
    np.ndarray
        The normalized signal array as a NumPy array with dtype np.float64.
    """
    if len(signal) == 0:
        return np.array([], dtype=np.float64)

    med, mad = med_mad(signal, with_nan=with_nan)

    if mad == 0:
        msg = "MAD normalization failed: scale is 0"
        logging.error(msg)
        raise ValueError(msg)

    norm_sig = clip_signal(signal, outlier_thresh, med, mad)

    return (norm_sig - med) / mad
