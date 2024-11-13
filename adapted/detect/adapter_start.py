"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import bottleneck
import numpy as np
from adapted.config.sig_proc import MMAdapterStartConfig


def moving_mean_adapter_start_detect(
    calibrated_signal: np.ndarray, params: MMAdapterStartConfig
):
    """
    calibrated_siganl: normalized, calibrated signal up to the detected adapter end
    """
    sig_rev = calibrated_signal[::-1]
    moving_mean_rev = bottleneck.move_mean(
        sig_rev[params.min_obs_adapter :],
        window=params.window,
    )[
        params.window :
    ]  # remove edge effects

    L = len(moving_mean_rev)
    moving_mean_fwd = moving_mean_rev[::-1]

    forward_cumsum = np.cumsum(moving_mean_fwd)
    forward_mean = forward_cumsum / np.arange(1, L + 1)

    backward_cumsum = np.cumsum(moving_mean_rev)[::-1]
    backward_mean = backward_cumsum / np.arange(L, 0, -1)

    difference = backward_mean - forward_mean
    if not difference.size:
        return 0

    difference[: params.window] = difference.max()  # remove edge effects

    cand = np.argmin(difference)
    if difference[cand] < -params.min_shift and any(
        moving_mean_fwd[
            max(0, int(cand) - params.window // 2) : int(cand) + params.window // 2
        ]
        > params.min_pA_current
    ):
        adapter_start = cand + params.window // 2
    else:
        adapter_start = 0

    return adapter_start
