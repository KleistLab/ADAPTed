"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import numpy as np


def impute_window_median(
    signal: np.ndarray, indices: np.ndarray, window_size: int = 5
) -> np.ndarray:
    """
    Impute the specified indices of the signal with the median of a surrounding window.

    Parameters
    ----------
    signal : np.ndarray
        The input signal array.
    indices : np.ndarray
        Indices of the signal array that require imputation.
    window_size : int, optional
        The size of the window used for calculating the median for imputation.
        Must be at least 3. The default is 5.

    Returns
    -------
    np.ndarray
        The signal array with values at specified indices imputed with window medians.

    Raises
    ------
    ValueError
        If the window_size is less than 3.
    """

    # check if window size is at least 3
    if window_size < 3:
        raise ValueError("window_size should be at least 3")

    half_window = window_size // 2

    signal_copy = signal.copy()
    for index in indices:
        if index < half_window:
            signal_copy[index] = np.median(signal_copy[: index + window_size])
        elif index > (len(signal_copy) - (half_window + 1)):
            signal_copy[index] = np.median(signal_copy[index - window_size :])
        else:
            signal_copy[index] = np.median(
                signal_copy[index - half_window : index + half_window]
            )
    return signal_copy


def mad_outlier_indices(signal: np.ndarray, outlier_thresh: float = 5.0) -> np.ndarray:
    """
    Identify indices of outlier values in a signal using Median Absolute Deviation (MAD).

    Parameters
    ----------
    signal : np.ndarray
        The input signal array.
    outlier_thresh : float, optional
        The threshold multiplier for MAD to determine outliers. Default is 5.0.

    Returns
    -------
    np.ndarray
        Array of indices that correspond to outlier values in the signal.
    """

    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    lower_lim = med - (mad * outlier_thresh)
    upper_lim = med + (mad * outlier_thresh)

    return ((signal < lower_lim) | (signal > upper_lim)).nonzero()[0]


def mad_normalize(signal: np.ndarray) -> np.ndarray:
    """ "
    Normalize a signal array using Median Absolute Deviation (MAD).

    Parameters
    ----------
    signal : np.ndarray
        The input signal array to be normalized.

    Returns
    -------
    np.ndarray
        The normalized signal array, where the median is shifted to 0 and scaled by MAD.
    """

    shift = np.median(signal)
    scale = np.median(np.abs(signal - shift))

    if scale == 0:
        raise ValueError("MAD normalization failed: scale is 0")

    norm_signal = (signal - shift) / scale
    return norm_signal


def mad_winsor(
    signal: np.ndarray, outlier_thresh: float = 5.0, window_size: int = 5
) -> np.ndarray:
    """
    Apply winsorization to a signal using MAD to limit the effect of outliers.

    Parameters
    ----------
    signal : np.ndarray
        The input signal array to be winsorized.
    outlier_thresh : float, optional
        The threshold multiplier for MAD to determine outliers. Default is 5.
    window_size : int, optional
        The size of the window used for imputation of outliers. Default is 5.

    Returns
    -------
    np.ndarray
        The winsorized signal array with outliers handled according to MAD.
    """

    outlier_indices = mad_outlier_indices(signal, outlier_thresh)
    wind_signal = impute_window_median(signal, outlier_indices, window_size)
    return wind_signal


def normalize_signal(
    signal: np.ndarray, outlier_thresh: float = 5.0, window_size: int = 5
) -> np.ndarray:
    """
    Normalize a signal by applying MAD-based winsorization followed by MAD normalization.

    Parameters
    ----------
    signal : np.ndarray
        The signal to normalize.
    outlier_thresh : float, optional
        Threshold for winsorizing outliers, by default 5.0.
    window_size : int, optional
        The window size for imputation during winsorizing, by default 5.

    Returns
    -------
    np.ndarray
        The normalized signal array as a NumPy array with dtype np.float64.
    """
    if len(signal) == 0:
        return np.array([], dtype=np.float64)

    return mad_normalize(mad_winsor(signal, outlier_thresh, window_size))
