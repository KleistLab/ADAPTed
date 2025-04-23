import numpy as np
import pandas as pd
import logging
from adapted.config.sig_proc import SigProcConfig
from adapted.detect.downscale import downscale_signal

logging.getLogger().setLevel(logging.DEBUG)

def detect_polya(
    signal_downsampled: np.ndarray,
    adapter_end_idx: int,
    min_len: int = 10,
    zscore: float = 3,
    max_gap: int = 5,
    min_stretch_post_gap: int = 5,
    min_std: float = 1.0,
    max_std: float = 3.0,
    update_std: bool = True,
) -> int:
    """Detects the end of a polyA tail sequence in a downsampled signal.

    This function identifies where the polyA tail ends by looking for a continuous segment
    of signal that stays within a standard deviation threshold of the mean polyA signal.
    Gaps between polyA segments are allowed, but the length of the gap is limited by
    max_gap and at least min_stretch_post_gap polyA matches must follow the gap.

    Args:
        signal_downsampled: Downsampled signal array without NaN values
        adapter_end_idx: Index where the adapter sequence ends
        min_len: Minimum length of polyA sequence to consider
        zscore: Number of standard deviations to use for threshold
        max_gap: Maximum gap allowed between polyA segments
        min_stretch_post_gap: Minimum stretch of polyA segments after a gap
        min_std: Minimum standard deviation of polyA segments
        max_std: Maximum standard deviation of polyA segments
        update_std: Whether to update the standard deviation of polyA segments
        after the first False position

    Returns:
        int: Index where the polyA tail ends. If no valid polyA is found,
             returns the adapter_end_idx
    """
    assert np.all(~np.isnan(signal_downsampled)), "signal downsampled contains nan"

    # Calculate baseline statistics from initial portion of potential polyA region
    assert type(adapter_end_idx) == np.int64, f"adapter_end_idx must be an integer, but is {adapter_end_idx} ({type(adapter_end_idx)}))"
    assert type(min_len) == int, f"min_len must be an integer, but is {min_len} ({type(min_len)}))"
    
    signal_post_adapter = signal_downsampled[adapter_end_idx:]
    polya_mean = signal_post_adapter[:min_len].mean()
    polya_std = signal_post_adapter[:min_len].std()
    polya_std = min(max(polya_std, min_std), max_std)
    

    within_bounds = abs(signal_post_adapter - polya_mean) <= zscore * polya_std

    # Find the first False position, update statistics
    first_false_idx = np.where(~within_bounds)[0]
    if len(first_false_idx) > 0 and update_std:
        if first_false_idx[0] > min_len:
            first_false_idx = first_false_idx[0]

            polya_mean = signal_post_adapter[:first_false_idx].mean()
            polya_std = signal_post_adapter[:first_false_idx].std()
            polya_std = min(max(polya_std, min_std), max_std)

            within_bounds = abs(signal_post_adapter - polya_mean) <= zscore * polya_std

    # Find valid polyA end allowing for small gaps
    if len(within_bounds) > min_len:
        # Convert to int array and get the portion after minimum length
        signal_after_min = within_bounds.astype(int)[min_len:]

        # Find runs of True and False values
        run_starts = np.where(np.diff(np.hstack(([0], signal_after_min))))[0]
        run_values = signal_after_min[run_starts]

        if len(run_starts) > 0:
            run_lengths = np.diff(np.hstack((run_starts, [len(signal_after_min)])))

            for i in range(len(run_starts)):
                if run_values[i] == 0:  # Found a gap
                    gap_length = run_lengths[i]

                    if gap_length > max_gap:
                        break

                    next_true_idx = i + 1
                    if (
                        next_true_idx >= len(run_values)
                        or run_lengths[next_true_idx] < min_stretch_post_gap
                    ):
                        break
            else:
                # No break, no invalid gaps found, use the entire sequence
                i = len(run_starts)

            if i > 0:
                end_pos = (
                    run_starts[i - 1]
                    if run_values[i - 1] == 0
                    else run_starts[i - 1] + run_lengths[i - 1]
                )
                return adapter_end_idx + end_pos + min_len

    return adapter_end_idx  # no valid polyA tail detected


def detect_rna_start_peak(
    batch_of_signals: np.ndarray, full_signal_lens: np.ndarray, spc: SigProcConfig
) -> pd.DataFrame:
    """Detects RNA start peaks and polyA tails in a batch of nanopore signals.

    This function processes multiple signals to identify for each signal:
    1. The initial adapter peak
    2. The adapter end position
    3. [Optional] The polyA tail region
    4. Any open pore signals that might indicate read issues

    The polyA tail region is only detected if the spc.rna_start_peak.detect_polya
    attribute is set to True in the config.

    Args:
        batch_of_signals: 2D array of raw signals [n_signals, signal_length]
        full_signal_lens: Array of actual lengths for each signal
        spc: Signal processing configuration object

    Returns:
        DataFrame containing detection results with columns:
        - success: Whether valid adapter and optionally polya were found
        - start_peak_idx: Position of initial peak
        - start_peak_pa: Signal value at start peak
        - next_greater_idx: Position of adapter end
        - next_greater_pa: Signal value at adapter end
        - polya_end_idx: Position where polyA tail ends
        - open_pore_idx: Position of any open pore signal
        - fail_reason: Description if detection failed
    """
    n, m = batch_of_signals.shape

    # Extract config parameters
    downscale_factor = spc.rna_start_peak.downscale_factor
    open_pore_pa = spc.rna_start_peak.open_pore_pa
    offset1 = spc.rna_start_peak.offset1
    start_peak_max_idx = spc.rna_start_peak.start_peak_max_idx
    offset2 = spc.rna_start_peak.offset2
    min_length = spc.core.min_obs_adapter // downscale_factor
    # Adjust signal lengths for downscaling
    end_idx = np.minimum(full_signal_lens, m)
    end_idx = end_idx // downscale_factor
    
    # Downsample signals for faster processing
    signals_downsampled = downscale_signal(batch_of_signals, downscale_factor)

    # Detect open pore signals (high current regions that might indicate read issues)
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
            success = True
            fail_reason = ""
            
            if end_idx[i] < min_length:
                success = False
                fail_reason = "read too short"
                res.append((success, None, None, None, None, None, None, fail_reason))
                continue

            # Find initial peak in the signal
            max_ = signals_downsampled[i, offset1:start_peak_max_idx].max()
            max_idx = (
                np.argmax(signals_downsampled[i, offset1:start_peak_max_idx] == max_)
                + offset1
            )
            max_ = max(spc.rna_start_peak.min_start_peak_pa, max_)

            # Find the next peak that exceeds the initial peak (potential adapter end)
            if max_idx + offset2 > end_idx[i]:
                success = False
                fail_reason = "read too short"
                res.append((success, None, None, None, None, None, None, fail_reason))
                continue

            next_max_idx = (
                np.argmax(signals_downsampled[i, max_idx + offset2 : end_idx[i]] > max_)
                + max_idx
                + offset2
            )
            next_max_ = signals_downsampled[i, next_max_idx]

            # Validate potential polyA region
            polya_mean_cand = signals_downsampled[
                i, next_max_idx : next_max_idx + spc.rna_start_peak.min_len_polya
            ].mean()
            adapter_med = np.median(signals_downsampled[i, :next_max_idx])

            # Check if polyA signal is too low compared to adapter
            if (
                polya_mean_cand
                < spc.rna_start_peak.adapter_med_polya_mean_scale * adapter_med
            ):
                success = False
                fail_reason = "adapter med candidate polya mean ratio too low"

            polya_end_idx = next_max_idx
            if spc.rna_start_peak.detect_polya and success:
                try:
                    polya_end_idx = detect_polya(
                        signal_downsampled=signals_downsampled[i, : end_idx[i]],
                        adapter_end_idx=next_max_idx,
                        min_len=spc.rna_start_peak.detect_polya_min_len,
                        zscore=spc.rna_start_peak.detect_polya_zscore,
                        max_gap=spc.rna_start_peak.detect_polya_max_gap_len,
                        min_stretch_post_gap=spc.rna_start_peak.detect_polya_min_stretch_post_gap,
                        min_std=spc.rna_start_peak.detect_polya_min_std,
                        max_std=spc.rna_start_peak.detect_polya_max_std,
                        update_std=spc.rna_start_peak.detect_polya_update_std,
                    )
                    assert type(polya_end_idx) == np.int64, f"detected polya_end_idx must be an integer, but is {polya_end_idx} ({type(polya_end_idx)}))"
                except Exception as e:
                    logging.debug(f"Error detecting polya: {e}")

                # logging.debug(f"polya_end_idx: {polya_end_idx}")
                
                if (polya_end_idx > next_max_idx) and (
                    signals_downsampled[i, next_max_idx:polya_end_idx].mean()
                    < spc.rna_start_peak.adapter_med_polya_mean_scale * adapter_med
                ):
                    success = False
                    fail_reason = "adapter med polya mean ratio too low"

                if polya_end_idx == next_max_idx:
                    fail_reason = "no polya detected"
                    success = False

            open_pore_idx = open_pore_ids.get(i, None)

            if open_pore_idx is not None and np.isclose(
                next_max_idx, open_pore_idx, atol=2, rtol=0.01
            ):
                success = False
                fail_reason = "open pore in adapter"

                res.append(
                    (
                        success,
                        max_idx,
                        max_,
                        next_max_idx,
                        next_max_,
                        polya_end_idx,
                        open_pore_idx,
                        fail_reason,
                    )
                )

            # open pore removed in downsampled signal
            elif open_pore_idx is not None and max_idx < open_pore_idx < next_max_idx:
                success = False
                fail_reason = "potential concatemer adapter-only read"
                res.append(
                    (
                        success,
                        max_idx,
                        max_,
                        next_max_idx,
                        next_max_,
                        polya_end_idx,
                        open_pore_idx,
                        fail_reason,
                    )
                )

            else:
                # Validate adapter length
                if next_max_idx < spc.core.min_obs_adapter // downscale_factor:
                    success = False
                    fail_reason = "start_peak adapter too short"
                elif next_max_idx > spc.core.max_obs_adapter // downscale_factor:
                    success = False
                    fail_reason = "start_peak adapter too long"

                res.append(
                    (
                        success,
                        max_idx,
                        max_,
                        next_max_idx,
                        next_max_,
                        polya_end_idx,
                        open_pore_idx,
                        fail_reason,
                    )
                )
        except Exception as e:
            print(e)
            res.append((None, None, None, None, None, None, None, None))

    res = pd.DataFrame(
        res,
        columns=[
            "success",
            "start_peak_idx",
            "start_peak_pa",
            "next_greater_idx",
            "next_greater_pa",
            "polya_end_idx",
            "open_pore_idx",
            "fail_reason",
        ],
    )

    # Scale indices back to original resolution only where they are not None
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

    mask = res.polya_end_idx.notna()
    res.loc[mask, "polya_end_idx"] = (
        res.loc[mask, "polya_end_idx"] * downscale_factor
    ).astype(int)

    return res
