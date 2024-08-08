"""
ADAPTed (Adapter and poly(A) Detection And Profiling Toolfiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de


"""

from typing import Optional, Tuple

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from attrs import define, field
from scipy.signal import find_peaks

from adapted.detect._c_llr import _gains, c_llr_trace, c_llr_trace_gains


@define
class LLRTrace:
    signal: np.ndarray = field(kw_only=True)
    c: Optional[np.ndarray] = field(default=None, kw_only=True)
    c2: Optional[np.ndarray] = field(default=None, kw_only=True)
    trace_start: int = field(default=0, kw_only=True)

    stride: int = field(kw_only=True)
    min_obs: int = field(kw_only=True)
    tail_trim: int = field(kw_only=True)

    start: int = field(default=None, init=False)
    end: int = field(default=None, init=False)

    # states
    early_stop: bool = field(default=None, init=False)
    stride_interp: bool = field(default=None, init=False)
    start_interp: bool = field(default=None, init=False)
    end_interp: bool = field(default=None, init=False)

    @property
    def max_len_no_early_stop(self):
        try:
            return np.arange(
                self.min_obs,
                self.signal.size - 1 - self.tail_trim,
                self.stride,
            )[-1]
        except IndexError:
            print("ERROR", self.min_obs, self.signal.size, self.tail_trim, self.stride)
            return self.signal.size - 1 - self.tail_trim

    @property
    def zero_tail_length(self):
        return self.signal.size - self.end

    def __attrs_post_init__(self):
        if self.signal is None:
            raise ValueError("signal is None")

        self.start, self.end = self._trace_start_end()
        self.early_stop = self.end < self.max_len_no_early_stop

        if self.stride > 1:
            self.interp_stride()

    def interp_start(self):
        y_val = self.signal[self.start]

        self.signal[np.arange(self.start)] = np.interp(
            np.arange(self.start), [0, self.start], [0, y_val], left=0
        )
        self.start_interp = True

    def interp_end(self):
        y_val = self.signal[self.end]

        self.signal[-1 * np.arange(self.zero_tail_length)] = np.interp(
            np.arange(self.zero_tail_length),
            [self.zero_tail_length, 0],
            [y_val, 0],
            left=0,
        )
        self.end_interp = True

    def interp_stride(self):
        # find non-zero values in llr_trace
        non_zero_indices = np.where(self.signal[self.start : self.end] != 0)[0]

        self.signal = np.interp(
            np.arange(self.signal.size),
            non_zero_indices + self.start,
            self.signal[non_zero_indices + self.start],
            left=0,  # tails remain 0 if not previously interpolated
            right=0,  # tails remain 0 if not previously interpolated
        )
        self.stride_interp = True

    def _trace_start_end(self) -> Tuple[int, int]:
        trace_start = np.argmin(self.signal <= 0)
        # if self.signal[trace_start] <= 0:
        #     print("ERROR", self.signal[trace_start], trace_start)
        trace_end = self.signal.size - np.argmin(self.signal[::-1] <= 0) - 1
        # if self.signal[trace_end] <= 0:
        #     print("ERROR", self.signal[trace_end], trace_end)
        return int(trace_start), int(trace_end)


def correct_for_plateau(
    trace_sig: np.ndarray,
    peak: int,
    s: int = 10,
    t: float = 0.9,
    window: int = 500,
    verbose: bool = False,
):
    """
    Check whether the signal chunk of size `window` in the trace after the `peak`
    contains an increase or plateau of at least lenght s, where the end of the plateau is at least of value t*trace[peak].

    Returns:
    int: The end index of the last plateau. -1 if no such plateau exists.
    """

    trace_ = trace_sig[peak : min(peak + window, trace_sig.size)]
    plateau_end = -1

    changes = np.diff(trace_)
    n = len(changes)
    for i in range(n - s, -1, -1):
        if (changes[i : i + (s - 1)] >= 0).all() and trace_[i + (s - 1)] > t * trace_[
            0
        ]:
            plateau_end = i + (s - 1)
            break

    if plateau_end > 0:
        if verbose:
            print(f"plateau end found! {peak+ plateau_end}")
        peak = peak + plateau_end
    return peak


def correct_for_split_peak(
    trace_sig: np.ndarray,
    peak: int,
    s: int = 10,
    t: float = 0.9,
    window: int = 500,
    prominence: float = 1.0,
    verbose: bool = False,
):
    peaks, _ = find_peaks(
        trace_sig[peak : min(peak + window, trace_sig.size)],
        width=s,
        prominence=prominence,
    )
    if verbose and peaks.size > 0:
        print(f"found split peaks! {peaks}")
        print(trace_sig[peaks[0] + peak], trace_sig[peak], t * trace_sig[peak])
    if peaks.size > 0 and trace_sig[peaks[0] + peak] >= t * trace_sig[peak]:
        if verbose:
            print(f"split peak accepted! {peaks[0]}")
        return peaks[0] + peak
    return peak


def find_peaks_in_trace(
    trace: LLRTrace,
    width: int = 100,
    prominence: float = 1.0,
    rel_height=0.5,
) -> np.ndarray:
    # ommit start and end chunks of 0-valued trace resulting from border trimming
    # helps to not select first nonzero pos as peak
    trace_sig_clip = trace.signal[
        trace.start if not trace.start_interp else 0 : (
            trace.end if not trace.end_interp else -1
        )
    ]

    peaks, _ = find_peaks(
        trace_sig_clip,
        width=width,
        prominence=prominence * np.nanstd(trace_sig_clip),
        rel_height=rel_height,
    )
    return peaks + (trace.start if not trace.start_interp else 0)


def adapter_end_from_trace(
    trace: LLRTrace,
    prominence: float = 1.0,
    rel_height: float = 1.0,
    width: int = 2000,
    fix_plateau: bool = True,
    correct_for_split_peaks: bool = True,
) -> np.ndarray:
    peaks = find_peaks_in_trace(trace, width, prominence, rel_height)

    # check and correct for (semi) plateau
    if fix_plateau:
        fixed_peaks = []
        for peak in peaks:
            fixed_peaks.append(
                correct_for_plateau(
                    trace.signal,
                    peak,
                )
            )
        peaks = np.array(fixed_peaks)

    if correct_for_split_peaks:
        fixed_peaks = []
        for peak in peaks:
            fixed_peaks.append(
                correct_for_split_peak(
                    trace.signal,
                    peak,
                )
            )
        peaks = np.array(fixed_peaks)
    return peaks


def calc_adapter_trace(
    signal: np.ndarray,
    offset_head: int,
    offset_tail: int,
    stride: int,
    early_stop1_window: int,
    early_stop1_stride: int,
    early_stop2_window: int,
    early_stop2_stride: int,
    return_c_c2: bool,
    trace_start: int = 0,
    c: Optional[np.ndarray] = None,
    c2: Optional[np.ndarray] = None,
) -> LLRTrace:
    # check if c and c2 are provided, they either need to be both there or not
    if (c is not None) != (c2 is not None):
        raise ValueError("c and c2 need to be both provided or not provided")
    # if c and c2 are provided, check if they have the correct size
    if c is not None and c2 is not None:
        if c.size != c2.size:
            raise ValueError("c and c2 need to have the same size")
        if c.size != signal.size:
            raise ValueError("c and c2 need to have the same size as signal")

        llr_trace = c_llr_trace_gains(
            c=c.astype(np.float64),
            c2=c2.astype(np.float64),
            start=trace_start,
            end=signal.size - 1,
            min_obs=offset_head,
            border_trim=offset_tail,
            stride=stride,
            adapter_early_stopping=0,
            adapter_early_stop_window=early_stop1_window,
            adapter_early_stop_stride=early_stop1_stride,
            polya_early_stopping=1,
            polya_early_stop_window=early_stop2_window,
            polya_early_stop_stride=early_stop2_stride,
        )
        if not return_c_c2:
            c, c2 = None, None

    else:
        res = c_llr_trace(
            signal.astype(np.float64),
            trace_start,
            signal.size - 1,
            offset_head,
            offset_tail,
            stride,
            0,
            early_stop1_window,
            early_stop1_stride,
            1,
            early_stop2_window,
            early_stop2_stride,
            int(return_c_c2),
        )
        if return_c_c2:
            llr_trace, c, c2 = res
        else:
            llr_trace = res
            c, c2 = None, None

    return LLRTrace(
        signal=llr_trace,
        c=c,
        c2=c2,
        trace_start=trace_start,
        stride=stride,
        min_obs=offset_head,
        tail_trim=offset_tail,
    )


def calc_polya_trace(
    c: np.ndarray,
    c2: np.ndarray,
    adapter_end: int,
    trace_early_stop_end: int,
    min_obs_polya: int,
    stride: int = 1,
) -> LLRTrace:
    trace_sig = _gains(
        adapter_end,  # start
        c.size - 1,  # end
        c.astype(np.float64),
        c2.astype(np.float64),
        min_obs_polya,  # offset head
        c.size - trace_early_stop_end,  # offset tail
        stride,
    )

    return LLRTrace(
        signal=trace_sig,
        c=c,
        c2=c2,
        stride=stride,
        min_obs=min_obs_polya + adapter_end,
        tail_trim=c.size - 1 - trace_early_stop_end,
    )
