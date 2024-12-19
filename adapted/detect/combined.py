"""
ADAPTed (Adapter and poly(A) Detection And Profiling Toolfiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import warnings
from copy import deepcopy
from typing import List, Union

import numpy as np
from adapted.config.sig_proc import SigProcConfig
from adapted.container_types import Boundaries, DetectResults
from adapted.detect.anomalies import find_open_pores
from adapted.detect.cnn import BoundariesCNN, cnn_detect_boundaries
from adapted.detect.downscale import downscale_signal
from adapted.detect.llr import (
    adapter_end_from_trace,
    calc_adapter_trace,
    detect_full_polya_trace_peak_with_spike,
)
from adapted.detect.mvs import (
    mean_var_shift_polyA_check,
    mean_var_shift_polyA_detect_at_loc,
)
from adapted.detect.normalize import normalize_signal
from adapted.detect.real_range import real_range_check
from adapted.detect.start_peak import detect_rna_start_peak
from adapted.detect.utils import in_range, range_is_empty
from adapted.partition.signal_partitions import calc_partitions_from_vals

##############################
# Combined
##############################


def detect_llr_on_downscaled_signal(
    ds_signal: np.ndarray,
    spc: SigProcConfig,
) -> Boundaries:
    """
    Detect adapter and poly(A) boundaries using LLR on a downscaled signal.

    Parameters:
        ds_signal: np.ndarray
            The downscaled signal to detect boundaries on. normalized, downscaled and without nan values.
        spc: SigProcConfig
            The signal processing configuration.

    Returns:
        Boundaries:
            The detected boundaries.
    """
    boundaries = Boundaries(
        adapter_start=0,
        adapter_end=0,
        polya_end=0,
        trace=np.array([]),
    )

    trace = calc_adapter_trace(
        signal=ds_signal,
        offset_head=1 + spc.core.min_obs_adapter // spc.core.downscale_factor,
        offset_tail=1,
        stride=1,
        early_stop1_window=0,
        early_stop1_stride=0,
        early_stop2_window=0,
        early_stop2_stride=0,
        return_c_c2=True,
        trace_start=0,
        adapter_early_stopping=0,
        polya_early_stopping=0,
        c=None,
        c2=None,
    )
    with warnings.catch_warnings():  # TODO: figure out source of warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        adapter_end_cands = adapter_end_from_trace(
            trace,
            prominence=spc.llr_boundaries.adapter_peak_prominence,
            rel_height=spc.llr_boundaries.adapter_peak_rel_height,
            width=spc.llr_boundaries.adapter_peak_width // spc.core.downscale_factor,
            fix_plateau=True,
            correct_for_split_peaks=True,
        )
        if len(adapter_end_cands) > 0:
            adapter_end = adapter_end_cands[0]
            if adapter_end > 0:
                boundaries.adapter_end = adapter_end * spc.core.downscale_factor
                trace = calc_adapter_trace(
                    signal=ds_signal,
                    offset_head=1,
                    offset_tail=1,
                    stride=1,
                    early_stop1_window=0,
                    early_stop1_stride=0,
                    early_stop2_window=0,
                    early_stop2_stride=0,
                    return_c_c2=False,
                    trace_start=adapter_end,
                    adapter_early_stopping=0,
                    polya_early_stopping=0,
                    c=trace.c,
                    c2=trace.c2,
                )
                polya_end = detect_full_polya_trace_peak_with_spike(trace.signal)
                if polya_end > 0:
                    boundaries.polya_end = polya_end * spc.core.downscale_factor
                    boundaries.polya_end_topk = np.array([boundaries.polya_end])

        return boundaries


def downscale_single_read_excl_nan(
    signal: np.ndarray, spc: SigProcConfig
) -> np.ndarray:
    ds = downscale_signal(
        signal.reshape(1, -1),
        spc.core.downscale_factor,
    ).ravel()
    m_down = ds.size
    n_nan = np.isnan(ds).sum()
    s = ds[: m_down - n_nan]
    return s


def combined_detect_llr(
    calibrated_signal: np.ndarray,
    full_signal_len: int,
    spc: SigProcConfig,
) -> DetectResults:

    norm_signal = normalize_signal(
        calibrated_signal[: min(spc.core.max_obs_trace, full_signal_len)],
        outlier_thresh=spc.core.sig_norm_outlier_thresh,
        with_nan=True,
    )
    s = downscale_single_read_excl_nan(norm_signal, spc)

    boundaries = detect_llr_on_downscaled_signal(s, spc)
    return validate_boundaries(
        calibrated_signal[:full_signal_len],
        boundaries,
        spc,
        full_signal_len,
    )


def combined_detect_llr2(
    batch_of_signals: np.ndarray,
    full_signal_lens: np.ndarray,
    spc: SigProcConfig,
) -> List[DetectResults]:

    norm_signal = normalize_signal(
        batch_of_signals[:, : spc.core.max_obs_trace],
        outlier_thresh=spc.core.sig_norm_outlier_thresh,
        with_nan=True,
    )  # batch normalized
    downscaled = downscale_signal(
        norm_signal,
        spc.core.downscale_factor,
    )

    list_of_boundaries = []
    m_down = downscaled.shape[1]
    n_nan = np.isnan(downscaled).sum(axis=1)

    for s, n in zip(downscaled, n_nan):
        s_ = s[: m_down - n]
        boundaries = detect_llr_on_downscaled_signal(s_, spc)
        list_of_boundaries.append(boundaries)

    del downscaled, n_nan

    list_of_detect_res = []
    for signal, boundaries, full_signal_len in zip(
        batch_of_signals, list_of_boundaries, full_signal_lens
    ):
        try:
            list_of_detect_res.append(
                validate_boundaries(
                    signal[:full_signal_len],
                    boundaries,
                    spc,
                    full_signal_len,
                )
            )
        except Exception as e:
            list_of_detect_res.append(DetectResults(success=False, fail_reason=str(e)))
    return list_of_detect_res


def combined_detect_cnn(
    batch_of_signals: np.ndarray,
    full_signal_lens: np.ndarray,
    model: BoundariesCNN,
    spc: SigProcConfig,
) -> Union[List[DetectResults], DetectResults]:

    list_of_boundaries = cnn_detect_boundaries(
        batch_of_signals, model, spc.cnn_boundaries, spc.core
    )
    assert isinstance(list_of_boundaries, list)

    res = []
    for signal, boundaries, full_signal_len in zip(
        batch_of_signals, list_of_boundaries, full_signal_lens
    ):
        try:

            validated = validate_boundaries(
                signal[:full_signal_len],
                boundaries,
                spc,
                full_signal_len,
            )
            if not validated.success:
                # TODO: norm downscaled signal for speed?
                norm_signal = normalize_signal(
                    signal[: min(spc.core.max_obs_trace, full_signal_len)],
                    outlier_thresh=spc.core.sig_norm_outlier_thresh,
                    with_nan=True,
                )
                spc_copy = deepcopy(spc)
                spc_copy.primary_method = "llr"
                # short reads for which polya boundary was likely overshoot
                if (
                    boundaries.adapter_end > 0
                    and boundaries.polya_end > 0
                    and boundaries.polya_end - boundaries.adapter_end > 1000
                    and full_signal_len < 2 * spc_copy.core.max_obs_adapter
                    and spc_copy.cnn_boundaries.fallback_to_llr_short_reads
                ):

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        # hail mary
                        s = downscale_single_read_excl_nan(
                            norm_signal[
                                int(boundaries.adapter_end) : int(boundaries.polya_end)
                            ],
                            spc_copy,
                        )

                        trace = calc_adapter_trace(
                            signal=s,
                            offset_head=5,
                            offset_tail=5,
                            stride=1,
                            early_stop1_window=0,
                            early_stop1_stride=0,
                            early_stop2_window=0,
                            early_stop2_stride=0,
                            return_c_c2=True,
                            adapter_early_stopping=0,
                            polya_early_stopping=0,
                            c=None,
                            c2=None,
                        )
                        polya_end = detect_full_polya_trace_peak_with_spike(
                            trace.signal
                        )
                        if polya_end > 0:
                            boundaries.polya_end = int(
                                polya_end * spc_copy.core.downscale_factor
                                + boundaries.adapter_end
                            )
                            boundaries.polya_end_topk = np.array([boundaries.polya_end])
                            validated = validate_boundaries(
                                signal[:full_signal_len],
                                boundaries,
                                spc_copy,
                                full_signal_len,
                            )
                if (
                    not validated.success and spc_copy.cnn_boundaries.fallback_to_llr
                ):  # still no success, retry with full LLR on downscaled signal
                    s = downscale_single_read_excl_nan(
                        norm_signal[
                            : min(spc_copy.core.max_obs_trace, full_signal_len)
                        ],
                        spc_copy,
                    )
                    boundaries = detect_llr_on_downscaled_signal(s, spc_copy)
                    detect_res = validate_boundaries(
                        signal[:full_signal_len],
                        boundaries,
                        spc_copy,
                        full_signal_len,
                    )
                    if detect_res.success:
                        validated = detect_res

            res.append(validated)
        except Exception as e:
            res.append(DetectResults(success=False, fail_reason=str(e)))

    del batch_of_signals, full_signal_lens, list_of_boundaries

    return res if len(res) > 1 else res[0]


def combined_detect_start_peak(
    batch_of_signals: np.ndarray,
    full_signal_lens: np.ndarray,
    spc: SigProcConfig,
) -> List[DetectResults]:

    df_res = detect_rna_start_peak(batch_of_signals, full_signal_lens, spc)

    list_of_detect_res = []
    read_i = 0
    for signal, full_signal_len in zip(batch_of_signals, full_signal_lens):
        res = df_res.iloc[read_i]
        boundaries = Boundaries(
            adapter_start=0,
            adapter_end=res.next_greater_idx,
            polya_end=res.next_greater_idx,
        )
        try:
            detect_res = validate_boundaries(
                signal[:full_signal_len],
                boundaries,
                spc,
                full_signal_len,
            )
            detect_res.start_peak_idx = res.start_peak_idx
            detect_res.start_peak_pa = res.start_peak_pa
            detect_res.start_peak_next_max_idx = res.next_greater_idx
            detect_res.start_peak_next_max_pa = res.next_greater_pa
            detect_res.start_peak_open_pore_idx = res.open_pore_idx
            detect_res.start_peak_open_pore_type = res.flagged_type

            flagged = res.flagged_type is not None
            false_before = not detect_res.success
            detect_res.success = detect_res.success and not flagged
            detect_res.fail_reason = (
                detect_res.fail_reason + ("+" + res.flagged_type)
                if false_before and flagged
                else detect_res.fail_reason
            )

            # retry failed reads using LLR on downscaled signal
            if detect_res.success is False and spc.cnn_boundaries.fallback_to_llr:
                spc_copy = deepcopy(spc)
                spc_copy.primary_method = "llr"
                s = downscale_single_read_excl_nan(signal[:full_signal_len], spc_copy)
                new_boundaries = detect_llr_on_downscaled_signal(s, spc_copy)
                new_detect_res = validate_boundaries(
                    signal[:full_signal_len],
                    new_boundaries,
                    spc_copy,
                    full_signal_len,
                )
                if new_detect_res.success:
                    detect_res = new_detect_res

            list_of_detect_res.append(detect_res)

        except Exception as e:
            list_of_detect_res.append(DetectResults(success=False, fail_reason=str(e)))

        read_i += 1

    return list_of_detect_res


def validate_boundaries(signal, boundaries, spc, full_signal_len) -> DetectResults:
    spc = deepcopy(
        spc
    )  # copy to avoid changing original config when setting pa_mean_range

    adapter_start = boundaries.adapter_start  # might be updated
    adapter_end = (
        boundaries.adapter_end
    )  # might be updated TODO: check if default != None causes issues here

    polya_end_best = boundaries.polya_end

    success = True
    mvs_adapter_end = None
    fail_reason = None

    mvs_detect_mean_at_loc = None
    mvs_detect_var_at_loc = None
    mvs_detect_polya_med = None
    mvs_detect_polya_local_range = None
    mvs_detect_med_shift = None

    real_adapter_mean_start = None
    real_adapter_mean_end = None
    real_adapter_local_range = None

    adapter_rna_median_shift = None

    adapter_mad = None
    adapter_med = None

    open_pores = None

    if adapter_end == 0 or adapter_end is None:
        success = False
        fail_reason = "No adapter detected (primary)"
    else:
        adapter_med = float(np.median(signal[adapter_start:adapter_end]))
        deviations = np.abs(signal[adapter_start:adapter_end] - adapter_med)
        adapter_mad = float(np.median(deviations))  # make type checker happy

    # catch llr_detect failure cases: initial stall and unexpected signal (e.g. high variance noise)
    if (
        success
        and adapter_mad
        and not in_range(adapter_mad, *spc.real_range.adapter_mad_range)
    ):
        success = False
        fail_reason = "adapter MAD check failed"

    if success and spc.real_range.detect_open_pores:
        open_pores = find_open_pores(
            signal[adapter_start:adapter_end],
        ).ravel()

        if open_pores.size > 0:
            open_pores = open_pores + adapter_start
            # use the last open pore in the adapter as the adapter start
            adapter_start = open_pores[-1]

            if adapter_end - adapter_start < spc.core.min_obs_adapter:
                success = False
                fail_reason = "Open pore too close to boundary"

    if success and spc.real_range.real_signal_check:
        (
            real_adapter_succes,
            real_adapter_mean_start,
            real_adapter_mean_end,
            real_adapter_local_range,
        ) = real_range_check(
            signal[adapter_start:adapter_end],
            params=spc.real_range,
            return_values=True,
        )

        if not real_adapter_succes:
            success = False
            fail_reason = "Real signal check failed"

    if success and spc.mvs_polya.mvs_detect_check:
        if polya_end_best == 0 or polya_end_best is None:
            success = False
            fail_reason = "No polya detected (primary)"

        else:
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
                raise ValueError("pA_mean_range is not specified")

            for polya_end in boundaries.polya_end_topk:
                if polya_end == 0 or polya_end is None:
                    break
                if not spc.mvs_polya.mvs_detect_overwrite:
                    assert adapter_end not in [None, 0] and polya_end not in [
                        None,
                        0,
                    ]  # make type checker happy
                    (
                        mvs_success,
                        mvs_check_vector,
                        mvs_detect_mean_at_loc,  # TODO change names to reflect this is not loc
                        mvs_detect_var_at_loc,
                        mvs_detect_polya_med,
                        mvs_detect_polya_local_range,
                        mvs_detect_med_shift,
                    ) = mean_var_shift_polyA_check(
                        signal,
                        adapter_end=int(adapter_end),
                        polya_end=int(polya_end),
                        params=spc.mvs_polya,
                        return_values=True,
                        less_signal_ok=False,
                        windowed_stats=True,
                    )

                    if not mvs_success:
                        success = False
                        if (
                            mvs_detect_mean_at_loc == 0
                        ):  # signal size or polya size check failed
                            fail_reason = "MVS polya check failed: not enough signal"
                        else:
                            failed_checks_str = ""
                            failed_checks_str += (
                                "mean " if not mvs_check_vector[0] else ""
                            )
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

                else:  # spc.mvs_polya.mvs_detect_overwrite = True
                    # look for adapter in [loc,loc+mvs_llr_max_offset] using MVS method

                    assert (
                        adapter_end is not None and adapter_end is not None
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
                        signal,
                        loc=adapter_end,
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
                            success = False
                            fail_reason = (
                                "Adapter end > polya end (mvs_detect overwrite)"
                            )

                if success:
                    polya_end_best = polya_end
                    break

    if success and spc.med_shift.detect_med_shift:
        adapter_rna_median_shift = np.median(
            signal[
                adapter_end : min(
                    adapter_end + spc.med_shift.med_shift_window, full_signal_len
                )
            ]
        ) - np.median(
            signal[max(adapter_end - spc.med_shift.med_shift_window, 0) : adapter_end]
        )
        if not in_range(adapter_rna_median_shift, *spc.med_shift.med_shift_range):
            success = False
            fail_reason = "Median shift check failed"

    partitions = calc_partitions_from_vals(
        signal[:full_signal_len],
        adapter_start,
        adapter_end,
        polya_end_best,
    )

    primary_section = {
        f"{spc.primary_method}_adapter_end": boundaries.adapter_end,
        f"{spc.primary_method}_polya_end": boundaries.polya_end,
    }

    detect_res = DetectResults(
        success=success,
        signal_len=full_signal_len,
        preloaded=(
            min(full_signal_len, signal.size)
            if full_signal_len is not None
            else signal.size
        ),
        adapter_end=adapter_end,
        polya_end=polya_end_best,
        polya_candidates=boundaries.polya_end_topk,
        **primary_section,
        mvs_adapter_end=mvs_adapter_end,
        mvs_detect_mean_at_loc=mvs_detect_mean_at_loc,
        mvs_detect_var_at_loc=mvs_detect_var_at_loc,
        mvs_detect_polya_med=mvs_detect_polya_med,
        mvs_detect_polya_local_range=mvs_detect_polya_local_range,
        mvs_detect_med_shift=mvs_detect_med_shift,
        adapter_rna_median_shift=adapter_rna_median_shift,
        real_adapter_mean_start=real_adapter_mean_start,
        real_adapter_mean_end=real_adapter_mean_end,
        real_adapter_local_range=real_adapter_local_range,
        open_pores=open_pores,
        fail_reason=fail_reason,
        **partitions.adapter.to_dict("adapter"),  # start,len,mean,std,med,mad
        **partitions.polya.to_dict("polya"),  # start,len,mean,std,med,mad
        **partitions.rna.to_dict("rna_preloaded"),  # start,len,mean,std,med,mad
    )

    return detect_res
