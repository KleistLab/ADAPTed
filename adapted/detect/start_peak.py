import numpy as np
import pandas as pd
from adapted.config.sig_proc import SigProcConfig
from adapted.detect.downscale import downscale_signal


def detect_rna_start_peak(
    batch_of_signals: np.ndarray, full_signal_lens: np.ndarray, spc: SigProcConfig
) -> pd.DataFrame:

    n, m = batch_of_signals.shape

    downscale_factor = spc.rna_start_peak.downscale_factor
    open_pore_pa = spc.rna_start_peak.open_pore_pa
    offset1 = spc.rna_start_peak.offset1
    start_peak_max_idx = spc.rna_start_peak.start_peak_max_idx
    offset2 = spc.rna_start_peak.offset2

    end_idx = np.minimum(full_signal_lens, m)
    end_idx = end_idx // downscale_factor

    signals_downsampled = downscale_signal(batch_of_signals, downscale_factor)

    open_pore_ids = {}
    for i in range(n):
        open_pore_idx = (
            np.argmax(batch_of_signals[i, : end_idx[i]] > open_pore_pa)
            // downscale_factor
        )
        if open_pore_idx > 0:
            open_pore_ids[i] = open_pore_idx

    res = []
    for i in range(n):
        try:
            max_ = signals_downsampled[i, offset1:start_peak_max_idx].max()
            max_idx = (
                np.argmax(signals_downsampled[i, offset1:start_peak_max_idx] == max_)
                + offset1
            )

            next_max_idx = (
                np.argmax(
                    signals_downsampled[i, start_peak_max_idx + offset2 : end_idx[i]]
                    > max_
                )
                + start_peak_max_idx
                + offset2
            )
            next_max_ = signals_downsampled[i, next_max_idx]

            open_pore_idx = open_pore_ids.get(i, None)
            if open_pore_idx is not None and np.isclose(
                next_max_idx, open_pore_idx, atol=2, rtol=0.01
            ):
                res.append(
                    (
                        max_idx,
                        max_,
                        next_max_idx,
                        next_max_,
                        open_pore_idx,
                        "open pore in adapter",
                    )
                )

            # open pore removed in downsampled signal
            elif open_pore_idx is not None and max_idx < open_pore_idx < next_max_idx:
                # if np.isclose(max_, next_max_, atol=1, rtol=0.05):
                res.append(
                    (
                        max_idx,
                        max_,
                        next_max_idx,
                        next_max_,
                        open_pore_idx,
                        "potential concatemer adapter-only read",
                    )
                )

            else:
                res.append((max_idx, max_, next_max_idx, next_max_, None, None))
        except Exception as e:
            res.append((None, None, None, None, None, None))

    res = pd.DataFrame(
        res,
        columns=[
            "start_peak_idx",
            "start_peak_pa",
            "next_greater_idx",
            "next_greater_pa",
            "open_pore_idx",
            "flagged_type",
        ],
    )
    # Calculate scale only where both values are not None
    mask = (res.next_greater_pa.notna()) & (res.start_peak_pa.notna())
    res.loc[mask, "scale"] = (
        res.loc[mask, "next_greater_pa"] / res.loc[mask, "start_peak_pa"]
    )

    # Scale indices only where they are not None
    mask = res.start_peak_idx.notna()
    res.loc[mask, "start_peak_idx"] = (
        res.loc[mask, "start_peak_idx"] * downscale_factor
    ).astype(int)

    mask = res.next_greater_idx.notna()
    res.loc[mask, "next_greater_idx"] = (
        res.loc[mask, "next_greater_idx"] * downscale_factor
    ).astype(int)

    mask = res.open_pore_idx.notna()
    res.loc[mask, "open_pore_idx"] = (
        res.loc[mask, "open_pore_idx"] * downscale_factor
    ).astype(int)

    return res
