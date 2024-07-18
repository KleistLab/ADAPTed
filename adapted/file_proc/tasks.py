"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import traceback
from typing import List

from adapted.config.sig_proc import SigProcConfig
from adapted.detect.combined import combined_detect
from adapted.io_utils import construct_filename
from adapted.output import save_detected_boundaries, save_traces
from adapted.file_proc.file_proc import ReadResult


def process_preloaded_signal(
    signal,
    signal_len: int,
    full_sig_len: int,
    read_id: str,
    spc: SigProcConfig,
    llr_return_trace: bool = False,
) -> ReadResult:
    try:
        detect_results = combined_detect(
            calibrated_signal=signal[:signal_len],
            full_signal_len=full_sig_len,
            spc=spc,
            llr_return_trace=llr_return_trace,
        )
        return ReadResult(
            read_id=read_id,
            success=detect_results.success,
            fail_reason=detect_results.fail_reason,
            detect_results=detect_results,
        )

    except:
        print(f"Failed on read {read_id}")
        traceback.print_exc()
        return ReadResult(
            read_id=read_id,
            success=False,
            fail_reason="Unknown error",
            detect_results=None,
        )


def save_results_batch(
    pass_or_fail: str,
    results: List[ReadResult],
    batch_idx: int,
    output_dir: str,
    save_llr_trace: bool = False,
) -> None:
    if pass_or_fail == "pass":
        fn = "detected_boundaries"
    elif pass_or_fail == "fail":
        fn = "failed_reads"
    else:
        raise ValueError(
            f"Invalid pass_or_fail: {pass_or_fail}. Must be 'pass' or 'fail'."
        )

    save_detected_boundaries(
        results,
        construct_filename(output_dir, f"{fn}_{batch_idx}.csv"),
        save_fail_reasons=pass_or_fail == "fail",
    )

    if save_llr_trace:
        save_traces(
            results,
            construct_filename(output_dir, f"{fn}_{batch_idx}_llr_trace.npz"),
        )
