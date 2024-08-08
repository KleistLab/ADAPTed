"""
ADAPTed (Adapter and poly(A) Detection And Profiling Toolfiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from scipy.stats import zscore

from adapted.config.sig_proc import LLRBoundariesConfig
from adapted.detect.llr_helpers import (
    LLRTrace,
    adapter_end_from_trace,
    calc_adapter_trace,
    calc_polya_trace,
    find_peaks_in_trace,
)
from adapted.detect.normalize import normalize_signal

##############################
# LLR-based adapter detection
##############################


@dataclass
class Boundaries:
    adapter_start: int
    adapter_end: int
    polya_end: int
    adapter_end_adjust: Optional[int] = None
    polya_end_adjust: Optional[int] = None
    trace: Optional[np.ndarray] = None
    logstr: Optional[str] = None
    polya_truncated: Optional[bool] = None
    polya_trace_early_stop_pos: Optional[int] = None


class LLRBoundariesLog:
    too_little_signal: bool = False
    no_adapter_end_found: bool = False
    truncated_polya: bool = False
    adapter_end_too_close_to_trace_end: bool = False
    no_polya_end_found: bool = False
    refine_too_few_extrema: bool = False
    refine_adapter_end_adjusted_within_atol: bool = False
    refine_polya_not_poi: bool = False
    refine_region_too_short: bool = False
    refine_possible_noisy_polya: bool = False
    refine_adapter_end_adjusted: bool = False

    def to_string(self):
        return " ".join([f"{k}" for k, v in self.__dict__.items() if v])


def detect_boundaries(
    signal_pa: np.ndarray,
    params: LLRBoundariesConfig,
    return_trace: bool = False,
) -> Boundaries:
    """signal should be clipped at max length"""

    logger = LLRBoundariesLog()

    results = Boundaries(
        adapter_start=0,
        adapter_end=0,
        polya_end=0,
        trace=np.array([]),
        logstr="",
    )

    if (
        signal_pa.size
        <= params.min_obs_adapter
        + params.adapter_trace_tail_trim
        + params.min_obs_polya
    ):
        logger.too_little_signal = True
        results.logstr = logger.to_string()
        return results

    norm_signal = normalize_signal(
        signal_pa[: params.max_obs_trace],
        outlier_thresh=params.sig_norm_outlier_thresh,
        window_size=params.sig_norm_winsor_window,
    )

    # 1. calc adapter trace
    trace: "LLRTrace" = calc_adapter_trace(
        signal=norm_signal,
        offset_head=params.min_obs_adapter,
        offset_tail=params.adapter_trace_tail_trim,
        stride=params.adapter_trace_stride,
        early_stop1_window=params.adapter_trace_early_stop_window,
        early_stop1_stride=params.adapter_trace_early_stop_stride,
        early_stop2_window=params.polya_trace_early_stop_window,
        early_stop2_stride=params.polya_trace_early_stop_stride,
        return_c_c2=True,
    )
    results.trace = trace.signal if return_trace else np.array([])
    trace.interp_start()

    # 2. find adapter end
    adapter_end_cands = adapter_end_from_trace(
        trace,
        prominence=params.adapter_peak_prominence,
        rel_height=params.adapter_peak_rel_height,
        width=params.adapter_peak_width,
        fix_plateau=True,
        correct_for_split_peaks=True,
    )

    if adapter_end_cands.size == 0:
        logger.no_adapter_end_found = True
        # give another try
        adapter_end_cands = [np.argmax(trace.signal)]

    results.adapter_end = int(adapter_end_cands[0])
    results.polya_end = int(trace.max_len_no_early_stop)

    # 3. if not early stop, return
    results.polya_truncated = not trace.early_stop
    results.polya_trace_early_stop_pos = trace.end

    if results.polya_truncated:
        logger.truncated_polya = True
        results.logstr = logger.to_string()
        return results

    # 4. find polya_end

    # 4.1 get trace

    if trace.end - results.adapter_end < params.min_obs_polya:
        logger.adapter_end_too_close_to_trace_end = True
        results.logstr = logger.to_string()
        return results

    assert trace.c is not None and trace.c2 is not None  # make type checker happy

    polya_trace = calc_polya_trace(
        c=trace.c,
        c2=trace.c2,
        adapter_end=results.adapter_end,
        trace_early_stop_end=trace.end,  # where the trace stops
        min_obs_polya=params.polya_trace_start,
        stride=params.polya_trace_stride,
    )
    polya_trace.interp_end()

    # 4.2 find polya end
    polya_end_cands = find_peaks_in_trace(
        polya_trace,
        width=params.polya_peak_width,
        prominence=params.polya_peak_prominence,
        rel_height=params.polya_peak_rel_height,
    )

    if polya_end_cands.size == 0:
        logger.no_polya_end_found = True
        # none found, use max likelihood position
        polya_end_cands = [np.argmax(polya_trace.signal)]

    results.polya_end = int(polya_end_cands[0])

    # 5.refine adapter end and polya end
    prev_adapter_end = results.adapter_end
    prev_polya_end = results.polya_end

    results.adapter_end, results.polya_end, logger = refine_boundaries(
        norm_signal, results.adapter_end, results.polya_end, params, logger
    )  # TODO: test if using norm_signal here is a problem
    results.adapter_end_adjust = results.adapter_end - prev_adapter_end
    results.polya_end_adjust = results.polya_end - prev_polya_end

    # 6. return adapter end, polya end
    results.logstr = logger.to_string()

    return results


def refine_boundaries(
    signal_pa: np.ndarray,
    adapter_end: int,
    polya_end: int,
    params: LLRBoundariesConfig,
    logger: LLRBoundariesLog,
) -> Tuple[int, int, LLRBoundariesLog]:
    smooth_pa = gaussian_filter(
        signal_pa[adapter_end : polya_end + params.refine_polya_atol],
        sigma=params.refine_smooth_sigma,
    )
    smooth_pa_diff = np.diff(smooth_pa)
    local_min_idx = argrelextrema(
        smooth_pa_diff,
        np.less,
    )[0]
    local_max_idx = argrelextrema(
        smooth_pa_diff,
        np.greater,
    )[0]

    extrema_idx = np.sort(np.concatenate([local_min_idx, local_max_idx]))
    extrema_val = smooth_pa_diff[extrema_idx]

    if extrema_val.size < 10:
        logger.refine_too_few_extrema = True
        return adapter_end, polya_end, logger

    zscores = abs(zscore(extrema_val))
    count, bins = np.histogram(zscores, bins=10)
    thr = bins[
        np.argmax(count == 0)
    ]  # left most bin_edge if no gap is found --> all extrema are poi, and read is filtered out below as noisey read
    thr = min(thr, 3)

    extrema_poi_idx = np.argwhere(zscores > thr).ravel()  # points of interest
    poi_idx = extrema_idx[extrema_poi_idx]

    local_min_poi_idx = np.array([x for x in poi_idx if x in local_min_idx])
    local_max_poi_idx = np.array([x for x in poi_idx if x in local_max_idx])

    polya_poi_mask = np.isclose(
        polya_end - adapter_end, poi_idx, atol=params.refine_polya_atol
    )
    polya_poi = polya_poi_mask.any()
    polya_poi_idx = poi_idx[polya_poi_mask]

    if not polya_poi:
        logger.refine_polya_not_poi = True
        return adapter_end, polya_end, logger

    # adjust adapter start within atol window
    if local_max_poi_idx.size and local_max_poi_idx[0] < params.refine_polya_atol:
        logger.refine_adapter_end_adjusted_within_atol = True
        adapter_end = adapter_end + local_max_poi_idx[0]
        # remove from index arrays
        poi_idx = poi_idx[poi_idx > local_max_poi_idx[0]]
        polya_poi_idx = polya_poi_idx - local_max_poi_idx[0]
        local_min_poi_idx = local_min_poi_idx - local_max_poi_idx[0]
        local_max_poi_idx = local_max_poi_idx - local_max_poi_idx[0]
        local_max_poi_idx = local_max_poi_idx[local_max_poi_idx > local_max_poi_idx[0]]

    if polya_end - adapter_end != polya_poi_idx[0]:
        polya_end = (
            adapter_end + polya_poi_idx[0]
        )  # adjust polya_end within atol window

    if (
        polya_end - adapter_end < params.min_obs_polya
    ):  # TODO: document that 100 is a good param for this, too short and stats become unreliable
        logger.refine_region_too_short = True
        return adapter_end, polya_end, logger

    n_smaller_pois = (poi_idx < (polya_end - adapter_end)).sum()
    if n_smaller_pois > 2:
        logger.refine_possible_noisy_polya = True  # might be noisy
        return adapter_end, polya_end, logger

    # find the left-most local min before the polyA end
    if local_min_poi_idx.size > 0:
        first_local_min_poi_idx = local_min_poi_idx[
            np.argmax(local_min_poi_idx < polya_end - adapter_end)
        ]
        # check if any was found, argmax will return 0 if no min was found
        if first_local_min_poi_idx < polya_end - adapter_end:
            # check if there are any extrema in between
            # if not, and the range between first_local_min_poi_idx and polya_end
            # is greater than the distance between adapter end and  first_local_min_poi_idx
            # and the variance of first_local_min_poi_idx to polya_end is
            # less than the variance of adapter_end to first_local_min_poi_idx
            # then don't accept it.

            var1 = np.var(smooth_pa_diff[:first_local_min_poi_idx])
            var2 = np.var(
                smooth_pa_diff[first_local_min_poi_idx : polya_end - adapter_end]
            )
            # print("var1", var1, "var2", var2)

            if (
                first_local_min_poi_idx
                > first_local_min_poi_idx - (polya_end - adapter_end)
                and var1 < var2
            ):
                polya_end = adapter_end + first_local_min_poi_idx

            else:
                first_local_min_poi_idx = (
                    polya_end - adapter_end
                )  # reset for next part!

        # find if there is any local max before last_min_idx
        if local_max_poi_idx.size > 0:
            local_max_poi_before_first_local_min_poi_idx = local_max_poi_idx[
                np.argmax(local_max_poi_idx < first_local_min_poi_idx)
            ]
            # check if any was found, argmax will return 0 if no min was found
            if (
                local_max_poi_before_first_local_min_poi_idx < first_local_min_poi_idx
                and first_local_min_poi_idx
                - local_max_poi_before_first_local_min_poi_idx
                >= params.min_obs_polya
            ):  # min length polya
                logger.refine_adapter_end_adjusted = True
                adapter_end = adapter_end + local_max_poi_before_first_local_min_poi_idx

    return adapter_end, polya_end, logger
