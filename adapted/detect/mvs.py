"""
ADAPTed (Adapter and poly(A) Detection And Profiling Toolfiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

#########################################
# Mean Var Shift based adapter detection
#########################################

from typing import Literal, Tuple, Union, overload

import bottleneck
import numpy as np

from adapted.config.sig_proc import MVSPolyAConfig, StreamingConfig
from adapted.detect.utils import LOCAL_RANGE_PCTLS, in_range


@overload
def mean_var_shift_polyA_check(
    calibrated_signal: np.ndarray,
    adapter_end: int,
    polya_end: int,
    params: MVSPolyAConfig,
    return_values: Literal[True] = True,
    less_signal_ok: bool = False,
    windowed_stats: bool = True,
) -> Tuple[bool, np.ndarray, float, float, float, float, float]: ...


@overload
def mean_var_shift_polyA_check(
    calibrated_signal: np.ndarray,
    adapter_end: int,
    polya_end: int,
    params: MVSPolyAConfig,
    return_values: Literal[False] = False,
    less_signal_ok: bool = False,
    windowed_stats: bool = True,
) -> bool: ...


def mean_var_shift_polyA_check(
    calibrated_signal: np.ndarray,
    adapter_end: int,
    polya_end: int,
    params: MVSPolyAConfig,
    return_values: bool = False,
    less_signal_ok: bool = False,
    windowed_stats: bool = True,
) -> Union[bool, Tuple[bool, np.ndarray, float, float, float, float, float]]:
    return_mean = 0.0
    return_var = 0.0
    return_polya_med = 0.0
    return_polya_local_range_ = 0.0
    return_med_shift = 0.0

    sig_size = calibrated_signal.size

    # not enough signal after loc to execute checks
    if not less_signal_ok and sig_size < (adapter_end + params.median_shift_window):
        return (False, np.zeros(5).astype(bool), return_mean, return_var, return_polya_med, return_polya_local_range_, return_med_shift) if return_values else False  # type: ignore

    if windowed_stats:
        if polya_end - adapter_end <= params.pA_var_window:
            polya_var = np.var(calibrated_signal[adapter_end:polya_end])
        else:
            polya_var = np.nanmedian(
                bottleneck.move_var(
                    calibrated_signal[adapter_end:polya_end],
                    window=params.pA_var_window,
                )
            )

        if polya_end - adapter_end <= params.pA_mean_window:
            polya_mean = np.mean(calibrated_signal[adapter_end:polya_end])
        else:
            polya_mean = np.nanmedian(
                bottleneck.move_mean(
                    calibrated_signal[adapter_end:polya_end],
                    window=params.pA_mean_window,
                )
            )
    else:
        polya_mean = np.mean(calibrated_signal[adapter_end:polya_end])
        polya_var = np.var(calibrated_signal[adapter_end:polya_end])

    polya_med = np.median(calibrated_signal[adapter_end:polya_end])
    polya_local_range_ = np.subtract(
        *np.percentile(
            calibrated_signal[adapter_end:polya_end],
            LOCAL_RANGE_PCTLS,
        )
    )

    med_shift = np.median(
        calibrated_signal[
            adapter_end : min(adapter_end + params.median_shift_window, sig_size)
        ]
    ) - np.median(
        calibrated_signal[
            max(adapter_end - params.median_shift_window, 0) : adapter_end
        ]
    )

    return_mean = float(polya_mean)
    return_var = float(polya_var)
    return_polya_med = float(polya_med)
    return_polya_local_range_ = float(polya_local_range_)
    return_med_shift = float(med_shift)

    check_vector = np.array(
        [
            in_range(return_mean, *params.pA_mean_range),
            in_range(return_var, *params.pA_var_range),
            in_range(return_polya_med, *params.polyA_med_range),
            in_range(return_polya_local_range_, *params.polyA_local_range),
            in_range(return_med_shift, *params.median_shift_range),
        ]
    )

    return (
        (
            check_vector.all(),
            check_vector,
            return_mean,
            return_var,
            return_polya_med,
            return_polya_local_range_,
            return_med_shift,
        )
        if return_values
        else check_vector.all()
    )


@overload
def mean_var_shift_polyA_detect_at_loc(
    calibrated_signal: np.ndarray,
    loc: int,
    params: MVSPolyAConfig,
    return_values: Literal[True] = True,
    less_signal_ok: bool = True,
) -> Tuple[bool, int, float, float, float, float, float]: ...


@overload
def mean_var_shift_polyA_detect_at_loc(
    calibrated_signal: np.ndarray,
    loc: int,
    params: MVSPolyAConfig,
    return_values: Literal[False] = False,
    less_signal_ok: bool = True,
) -> bool: ...


def mean_var_shift_polyA_detect_at_loc(
    calibrated_signal: np.ndarray,
    loc=0,
    params: MVSPolyAConfig = MVSPolyAConfig(),
    return_values=False,
    less_signal_ok=True,
) -> Union[bool, Tuple[bool, int, float, float, float, float, float]]:
    """_summary_

    Parameters
    ----------
    calibrated_signal : np.ndarray
         normalized, calibrated signal, expected to contain adapter, polyA and RNA.
    loc : int, optional
         location to check for start of polyA tail, by default 0
    params : MVSPolyAConfig
         object containing parameters for polyA detection
    return_values : bool, optional
         return values if True, by default False
    less_signal_ok : bool, optional
        allow for less signal after loc than needed for median_shift_window and polyA_window, by default True

    Returns
    -------
    _type_
        _description_
    """

    return_idx = 0
    return_mean = 0.0
    return_var = 0.0
    return_polya_med = 0.0
    return_polya_local_range_ = 0.0
    return_med_shift = 0.0

    sig_size = calibrated_signal.size

    # not enough signal after loc to execute checks
    if not less_signal_ok and sig_size < (
        loc
        + params.search_window
        + np.max([params.median_shift_window, params.polyA_window])
    ):
        return (
            (
                False,
                return_idx,
                return_mean,
                return_var,
                return_polya_med,
                return_polya_local_range_,
                return_med_shift,
            )
            if return_values
            else False
        )

    # not enough signal before loc to compensate for moving mean and var offset
    if loc < max([params.pA_mean_window, params.pA_var_window]):
        return (
            (
                False,
                return_idx,
                return_mean,
                return_var,
                return_polya_med,
                return_polya_local_range_,
                return_med_shift,
            )
            if return_values
            else False
        )

    offset = max(params.pA_mean_window, params.pA_var_window)
    moving_mean = bottleneck.move_mean(
        calibrated_signal[loc - offset : loc + params.search_window],
        window=params.pA_mean_window,
    )

    moving_var = bottleneck.move_var(
        calibrated_signal[loc - offset : loc + params.search_window],
        window=params.pA_var_window,
    )

    idx = np.argmax(
        in_range(moving_mean, *params.pA_mean_range)
        * in_range(moving_var, *params.pA_var_range)
    )  # returns index of first true, or 0 if all are false

    if idx > 0:
        mean = moving_mean[idx]
        var = moving_var[idx]
        idx += loc - offset

    else:
        # if loc is the LLR detected boundary, the mvs detected boundary is expected at loc+offset due to the running window lag
        mean = moving_mean[2 * offset]
        var = moving_var[2 * offset]

    # if found, mean and var reflect signal features at found location
    # else (idx==0), mean and var reflect signal features at loc
    return_idx = int(idx)
    return_mean = float(mean)
    return_var = float(var)

    loc_ = max(loc, int(idx))

    polya_med = np.median(
        calibrated_signal[loc_ : min(loc_ + params.polyA_window, sig_size)]
    )
    polya_local_range_ = np.subtract(
        *np.percentile(
            calibrated_signal[loc_ : min(loc_ + params.polyA_window, sig_size)],
            LOCAL_RANGE_PCTLS,
        )
    )

    med_shift = np.median(
        calibrated_signal[loc_ : min(loc_ + params.median_shift_window, sig_size)]
    ) - np.median(calibrated_signal[:loc_])

    return_polya_med = float(polya_med)
    return_polya_local_range_ = float(polya_local_range_)
    return_med_shift = float(med_shift)

    if (
        (idx > 0)
        and in_range(return_polya_med, *params.polyA_med_range)
        and in_range(return_polya_local_range_, *params.polyA_local_range)
        and in_range(return_med_shift, *params.median_shift_range)
    ):
        return (
            (
                True,
                return_idx,
                return_mean,
                return_var,
                return_polya_med,
                return_polya_local_range_,
                return_med_shift,
            )
            if return_values
            else True
        )

    return (
        (
            False,
            return_idx,
            return_mean,
            return_var,
            return_polya_med,
            return_polya_local_range_,
            return_med_shift,
        )
        if return_values
        else False
    )


def mean_var_shift_polyA_detect(
    calibrated_signal: np.ndarray,
    params: StreamingConfig = StreamingConfig(),
):
    """
    Intended for streaming polyA detection with an accumulating readuntil cache.

    When using this function on a full-length read, rather than a read chunk, it may be wise to use signal[:int(.75*len(signal))] as the input.
    Since the adapter should not take up more than 75% of the read.
    Default parameters are optimized for live data.
    """
    sig_size = calibrated_signal.size

    if sig_size < (
        params.min_obs_adapter
        + np.max(
            [
                params.pA_mean_window,
                params.pA_var_window,
                params.min_obs_post_loc,
                params.polyA_window,
            ]
        )
    ):
        return 0

    moving_mean = bottleneck.move_mean(
        calibrated_signal[params.min_obs_adapter :], window=params.pA_mean_window
    )
    moving_var = bottleneck.move_var(
        calibrated_signal[params.min_obs_adapter :], window=params.pA_var_window
    )

    signal_match = np.asarray(
        in_range(moving_mean, *params.pA_mean_range)
        & in_range(moving_var, *params.pA_var_range)
    )

    # slicing off increasing amount of the left part of signal to ignore abberant pA matches
    offset = max(params.pA_mean_window, params.pA_var_window)
    while offset < sig_size - params.min_obs_adapter:
        idx = np.argmax(
            signal_match[offset:]
        )  # returns index of first true, or 0 if all are false

        if idx > 0 or (
            # catching case where all conditions are true at position 0
            signal_match[offset]
        ):
            # adjusting idx position to raw signal reference
            idx += params.min_obs_adapter + offset

            # not enough signal left to check for median shift
            # for the smallest polyA tails we expect 10nt - 4 flanking= 6nt *14ms (RNA002)=84 ms =~250 obs
            if sig_size - idx < params.min_obs_post_loc:
                return 0

            polya = calibrated_signal[
                idx : min(int(idx + params.polyA_window), sig_size)
            ]
            median_shift = np.median(
                calibrated_signal[
                    idx : min(int(idx + params.median_shift_window), sig_size)
                ]
            ) - np.median(
                calibrated_signal[max(int(idx - params.median_shift_window), 0) : idx]
            )

            if (
                in_range(np.median(polya), *params.polyA_med_range)
                and in_range(
                    np.subtract(*np.percentile(polya, LOCAL_RANGE_PCTLS)),
                    *params.polyA_local_range,
                )
                and in_range(float(median_shift), *params.median_shift_range)
            ):
                return idx

            # if potential poly A pos doesn't match median shift criterion next offset
            else:
                offset = idx - params.min_obs_adapter + params.search_increment_step
                continue

        # didn't find a match, return 0
        return 0
    return 0
