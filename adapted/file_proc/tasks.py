"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import logging
import traceback
from typing import List, Optional

import numpy as np
from adapted.config.sig_proc import SigProcConfig
from adapted.container_types import ReadResult
from adapted.detect.cnn import BoundariesCNN, load_cnn_model
from adapted.detect.combined import combined_detect_cnn, combined_detect_llr
from adapted.io_utils import construct_filename
from adapted.output import save_detected_boundaries, save_traces


def process_preloaded_signal_combined_detect_llr(
    signal: np.ndarray,
    full_sig_len: int,
    read_id: str,
    spc: SigProcConfig,
) -> ReadResult:
    try:

        detect_results = combined_detect_llr(
            calibrated_signal=signal,
            full_signal_len=full_sig_len,
            spc=spc,
        )
        return ReadResult(
            read_id=read_id,
            success=detect_results.success,
            fail_reason=detect_results.fail_reason,
            detect_results=detect_results,
        )

    except:
        logging.error(f"Failed on read {read_id}")
        logging.error(traceback.format_exc())
        return ReadResult(
            read_id=read_id,
            success=False,
            fail_reason="Unknown error",
            detect_results=None,
        )


def process_preloaded_signal_combined_detect_cnn(
    batch_of_signals: np.ndarray,
    full_lengths: np.ndarray,
    read_ids: np.ndarray,
    spc: SigProcConfig,
    model: Optional[BoundariesCNN] = None,
) -> List[ReadResult]:
    if model is None:
        model = load_cnn_model(spc.cnn_boundaries.model_name)

    return [
        ReadResult(
            read_id=read_id,
            success=detect_results.success,
            fail_reason=detect_results.fail_reason,
            detect_results=detect_results,
        )
        for read_id, detect_results in zip(
            read_ids,
            combined_detect_cnn(
                batch_of_signals=batch_of_signals,
                full_signal_lens=full_lengths,
                model=model,
                spc=spc,
            ),
        )
    ]


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
        msg = f"Invalid pass_or_fail: {pass_or_fail}. Must be 'pass' or 'fail'."
        logging.error(msg)
        raise ValueError(msg)

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
