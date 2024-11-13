"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from typing import List

import numpy as np
import pandas as pd
from adapted.file_proc.file_proc import ReadResult


def save_traces(results: List[ReadResult], filename: str) -> None:
    traces = {
        str(res.read_id): res.detect_results.llr_trace
        for res in results
        if res.detect_results is not None and res.detect_results.llr_trace is not None
    }
    np.savez(filename, **traces)


def save_detected_boundaries(
    processing_results: List[ReadResult],
    filename: str,
    save_fail_reasons: bool = False,
):
    """Save detected boundaries and read ids to a csv file."""

    df = pd.DataFrame(
        [pr.to_summary_dict() for pr in processing_results],
    )

    if not df.empty:
        drop_cols = ["success", "llr_trace"]
        if not save_fail_reasons:
            drop_cols.append("fail_reason")

        for col in drop_cols:
            try:
                df.drop(columns=col, inplace=True)
            except:
                pass

    df.round(3).to_csv(
        filename,
        index=False,
    )
