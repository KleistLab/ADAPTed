"""
ADAPTed (Adapter and poly(A) Detection And Profiling Toolfiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from adapted.config.sig_proc import SigProcConfig
from adapted.partition.signal_partitions import Partitions, calc_partitions_from_vals
from adapted.detect.anomalies import find_open_pores
from adapted.detect.llr import detect_boundaries
from adapted.detect.mvs import (
    mean_var_shift_polyA_check,
    mean_var_shift_polyA_detect_at_loc,
)
from adapted.detect.normalize import mad_winsor
from adapted.detect.real_range import real_range_check
from adapted.detect.utils import in_range, range_is_empty
from copy import deepcopy


@dataclass
class DetectResults:
    success: bool

    adapter_start: Optional[int] = None
    adapter_end: Optional[int] = None
    adapter_len: Optional[int] = None
    adapter_mean: Optional[float] = None
    adapter_std: Optional[float] = None
    adapter_med: Optional[float] = None
    adapter_mad: Optional[float] = None

    polya_start: Optional[int] = None
    polya_end: Optional[int] = None
    polya_len: Optional[int] = None
    polya_mean: Optional[float] = None
    polya_std: Optional[float] = None
    polya_med: Optional[float] = None
    polya_mad: Optional[float] = None
    polya_truncated: Optional[bool] = None

    rna_start: Optional[int] = None
    rna_len: Optional[int] = None
    rna_mean: Optional[float] = None
    rna_std: Optional[float] = None
    rna_med: Optional[float] = None
    rna_mad: Optional[float] = None

    llr_adapter_end: Optional[int] = None
    llr_rel_adapter_end: Optional[float] = None
    llr_trace: Optional[np.ndarray] = None
    llr_adapter_end_adjust: Optional[int] = None
    llr_polya_end_adjust: Optional[int] = None

    mvs_llr_polya_end_adjust_ignored: Optional[bool] = None
    mvs_llr_polya_end_to_early_stop: Optional[bool] = None

    mvs_adapter_end: Optional[int] = None
    mvs_detect_mean_at_loc: Optional[float] = None
    mvs_detect_var_at_loc: Optional[float] = None
    mvs_detect_polya_med: Optional[float] = None
    mvs_detect_polya_local_range: Optional[float] = None
    mvs_detect_med_shift: Optional[float] = None

    real_adapter_mean_start: Optional[float] = None
    real_adapter_mean_end: Optional[float] = None
    real_adapter_local_range: Optional[float] = None

    open_pores: Optional[np.ndarray] = None

    fail_reason: Optional[str] = None
    llr_detect_log: Optional[str] = None

    def to_dict(self):
        return {
            **self.__dict__,
        }

    def update(self, partitions: Partitions):
        self.__dict__.update(partitions.to_dict())


##############################
# Combined
##############################


def combined_detect(
    calibrated_signal: np.ndarray,
    full_signal_len: int,
    spc: SigProcConfig,
    llr_return_trace: bool = False,
) -> DetectResults:

    spc = deepcopy(
        spc
    )  # copy to avoid changing original config when setting pa_mean_range

    # full_signal_len is can be smaller than config.llr_boundaries.max_obs_adapter, truncate before detection
    boundaries = detect_boundaries(
        calibrated_signal[:full_signal_len], spc.llr_boundaries, llr_return_trace
    )
    adapter_start = 0
    polya_end = boundaries.polya_end
    llr_adapter_end = boundaries.adapter_end

    llr_rel_adapter_end = (
        llr_adapter_end / full_signal_len
        if (full_signal_len is not None and full_signal_len > 0)
        else 0
    )
    llr_trace = boundaries.trace  # for debug
    llr_adapter_end_adjust = boundaries.adapter_end_adjust
    llr_polya_end_adjust = boundaries.polya_end_adjust

    polya_truncated = boundaries.polya_truncated
    success = True
    mvs_adapter_end = None
    adapter_end = None
    fail_reason = None

    mvs_detect_mean_at_loc = None
    mvs_detect_var_at_loc = None
    mvs_detect_polya_med = None
    mvs_detect_polya_local_range = None
    mvs_detect_med_shift = None

    mvs_llr_polya_end_adjust_ignored = False
    mvs_llr_polya_end_to_early_stop = False

    real_adapter_mean_start = None
    real_adapter_mean_end = None
    real_adapter_local_range = None

    adapter_mad = None
    adapter_med = None

    open_pores = None

    if llr_adapter_end > 0:
        adapter_med = float(np.median(calibrated_signal[adapter_start:llr_adapter_end]))
        deviations = np.abs(
            calibrated_signal[adapter_start:llr_adapter_end] - adapter_med
        )
        adapter_mad = float(np.median(deviations))  # make type checker happy

    if llr_adapter_end == 0:
        success = False
        fail_reason = "No adapter detected (ADAPT)"

    elif polya_end == 0:
        success = False
        fail_reason = "No polya detected (ADAPT)"

    # NOTE: makes sense to have in when adapter signal normalization becomes important
    # elif llr_rel_adapter_end > max_rel_adapter_pos:
    #     success = False
    #     fail_reason = "Detected boundary too close to end of read"

    # catch llr_detect failure cases: initial stall and unexpected signal (e.g. high variance noise)
    if adapter_mad and not in_range(adapter_mad, *spc.real_range.adapter_mad_range):
        success = False
        fail_reason = "adapter MAD check failed"

    if spc.real_range.detect_open_pores:
        open_pores = find_open_pores(
            calibrated_signal[adapter_start:llr_adapter_end],
        ).ravel()

        if open_pores.size > 0:
            open_pores = open_pores + adapter_start
            # use the last open pore in the adapter as the adapter start
            adapter_start = open_pores[-1]

            if llr_adapter_end - adapter_start < spc.llr_boundaries.min_obs_adapter:
                success = False
                fail_reason = "Open pore too close to boundary"

    if success:
        adapter_end = llr_adapter_end

        if spc.real_range.real_signal_check or spc.mvs_polya.mvs_detect_check:
            norm_signal = mad_winsor(
                calibrated_signal[:full_signal_len],
                spc.llr_boundaries.sig_norm_outlier_thresh,
                spc.llr_boundaries.sig_norm_winsor_window,
            )  # use same params as in llr_detect for signal normalization

            if range_is_empty(spc.mvs_polya.pA_mean_range) and not range_is_empty(
                spc.mvs_polya.pA_mean_adapter_med_scale_range
            ):
                # if pA mean range is not set, use adapter med scale range
                mvs_pA_mean_range = (
                    np.array(spc.mvs_polya.pA_mean_adapter_med_scale_range)
                    * adapter_med
                )
                spc.mvs_polya.pA_mean_range = (
                    mvs_pA_mean_range[0],
                    mvs_pA_mean_range[1],
                )

            # TODO: add to config parser, check valid param config for mvs check
            elif range_is_empty(spc.mvs_polya.pA_mean_range):
                success = False
                fail_reason = "pA mean range not set"

            if (
                spc.mvs_polya.mvs_detect_check
                and not spc.mvs_polya.mvs_detect_overwrite
            ):
                if polya_end == 0 or polya_end == llr_adapter_end:
                    success = False
                    fail_reason = "MVS polya check: no polyA detected"
                else:
                    assert (
                        llr_adapter_end is not None and polya_end is not None
                    )  # make type checker happy
                    (
                        mvs_success,
                        mvs_check_vector,
                        mvs_detect_mean_at_loc,  # TODO change names to reflect this is not loc
                        mvs_detect_var_at_loc,
                        mvs_detect_polya_med,
                        mvs_detect_polya_local_range,
                        mvs_detect_med_shift,
                    ) = mean_var_shift_polyA_check(
                        norm_signal,
                        adapter_end=int(llr_adapter_end),
                        polya_end=int(polya_end),
                        params=spc.mvs_polya,
                        return_values=True,
                        less_signal_ok=False,
                        windowed_stats=True,
                    )

                    if not mvs_success:
                        success = False
                        if mvs_detect_mean_at_loc == 0:  # too less signal check failed!
                            fail_reason = "MVS polya check failed: too less signal"
                        else:
                            failed_checks_str = ""
                            failed_checks_str += (
                                "mean " if not mvs_check_vector[0] else ""
                            )
                            # if not mvs_check_vector[0]:
                            #     print(
                            #         spc.mvs_polya.pA_mean_range,
                            #         mvs_detect_mean_at_loc,
                            #         adapter_med,
                            #         mvs_detect_mean_at_loc / adapter_med,
                            #     )
                            failed_checks_str += (
                                "var " if not mvs_check_vector[1] else ""
                            )
                            failed_checks_str += (
                                "med " if not mvs_check_vector[2] else ""
                            )
                            failed_checks_str += (
                                "range " if not mvs_check_vector[3] else ""
                            )
                            failed_checks_str += (
                                "shift" if not mvs_check_vector[4] else ""
                            )
                            # remove trailing space
                            failed_checks_str = failed_checks_str.rstrip()
                            fail_reason = f"MVS polya check failed: {failed_checks_str}"

            elif spc.mvs_polya.mvs_detect_check and spc.mvs_polya.mvs_detect_overwrite:
                # look for adapter in [loc,loc+mvs_llr_max_offset] using MVS method

                assert (
                    llr_adapter_end is not None and adapter_end is not None
                )  # make type checker happy

                (
                    mvs_success,
                    mvs_adapter_end,
                    mvs_detect_mean_at_loc,
                    mvs_detect_var_at_loc,
                    mvs_detect_polya_med,
                    mvs_detect_polya_local_range,
                    mvs_detect_med_shift,
                ) = mean_var_shift_polyA_detect_at_loc(
                    norm_signal,
                    loc=llr_adapter_end,
                    params=spc.mvs_polya,
                    return_values=True,
                    less_signal_ok=False,
                )

                if not mvs_success:
                    success = False
                    fail_reason = "No adapter detected in range (mvs_detect)"

                elif mvs_adapter_end - adapter_end > 0:
                    adapter_end = mvs_adapter_end

                    # now, adapter_end might be > polya_end, especially if the polya was (incorrectly) refined
                    if adapter_end > polya_end:
                        polya_end = adapter_end
                        if (
                            (boundaries.polya_end_adjust is not None)
                            and (boundaries.polya_end_adjust < 0)
                            and (
                                -boundaries.polya_end_adjust > (adapter_end - polya_end)
                            )
                        ):
                            polya_end = polya_end - boundaries.polya_end_adjust
                            mvs_llr_polya_end_adjust_ignored = True
                        elif not boundaries.polya_truncated:
                            polya_end = boundaries.polya_trace_early_stop_pos
                            mvs_llr_polya_end_to_early_stop = True

            if spc.real_range.real_signal_check and success:
                (
                    real_adapter_succes,
                    real_adapter_mean_start,
                    real_adapter_mean_end,
                    real_adapter_local_range,
                ) = real_range_check(
                    norm_signal[adapter_start:adapter_end],
                    params=spc.real_range,
                    return_values=True,
                )

                if not real_adapter_succes:
                    success = False
                    fail_reason = "Real signal check failed"

    partitions = calc_partitions_from_vals(
        calibrated_signal[:full_signal_len], adapter_start, adapter_end, polya_end
    )

    detect_res = DetectResults(
        success=success,
        adapter_start=adapter_start,
        adapter_end=adapter_end,
        polya_end=polya_end,
        polya_truncated=polya_truncated,
        llr_adapter_end=llr_adapter_end,
        llr_rel_adapter_end=llr_rel_adapter_end,
        llr_trace=llr_trace,
        llr_adapter_end_adjust=llr_adapter_end_adjust,
        llr_polya_end_adjust=llr_polya_end_adjust,
        mvs_llr_polya_end_adjust_ignored=mvs_llr_polya_end_adjust_ignored,
        mvs_llr_polya_end_to_early_stop=mvs_llr_polya_end_to_early_stop,
        mvs_adapter_end=mvs_adapter_end,
        mvs_detect_mean_at_loc=mvs_detect_mean_at_loc,
        mvs_detect_var_at_loc=mvs_detect_var_at_loc,
        mvs_detect_polya_med=mvs_detect_polya_med,
        mvs_detect_polya_local_range=mvs_detect_polya_local_range,
        mvs_detect_med_shift=mvs_detect_med_shift,
        real_adapter_mean_start=real_adapter_mean_start,
        real_adapter_mean_end=real_adapter_mean_end,
        real_adapter_local_range=real_adapter_local_range,
        adapter_med=adapter_med,
        adapter_mad=adapter_mad,
        open_pores=open_pores,
        fail_reason=fail_reason,
        llr_detect_log=boundaries.logstr,
    )

    detect_res.update(partitions)
    return detect_res
