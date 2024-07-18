"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from typing import Literal, Tuple, overload

import numpy as np

from adapted.config.sig_proc import RealRangeConfig
from adapted.detect.utils import LOCAL_RANGE_PCTLS, in_range


@overload
def real_range_check(
    calibrated_signal: np.ndarray,
    params: RealRangeConfig,
    return_values: Literal[True] = True,
) -> Tuple[bool, float, float, float]: ...


@overload
def real_range_check(
    calibrated_signal: np.ndarray,
    params: RealRangeConfig,
    return_values: Literal[False] = False,
) -> bool: ...


# ensure that enough signal is present before running this function
def real_range_check(
    calibrated_signal: np.ndarray,
    params: RealRangeConfig,
    return_values=False,
):
    """Checks that signal start with adapter, ends with polyA tail and has a sufficient local range."""

    if len(calibrated_signal) < 2 * params.mean_window:
        return (False, None, None, None) if return_values else False

    mean_start = np.mean(calibrated_signal[: params.mean_window])
    mean_end = np.mean(calibrated_signal[-params.mean_window :])
    vals = [mean_start, mean_end, None]  # mean_start, mean_end, local_range

    if in_range(float(mean_start), *params.mean_start_range) and in_range(
        float(mean_end), *params.mean_end_range
    ):
        local_range_ = np.subtract(
            *np.percentile(
                calibrated_signal[
                    -min(params.max_obs_local_range, len(calibrated_signal)) :
                ],
                LOCAL_RANGE_PCTLS,
            )
        )
        vals[2] = local_range_
        range_check = in_range(local_range_, *params.local_range)

        return (range_check, *vals) if return_values else range_check

    return (False, *vals) if return_values else False
