from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Boundaries:
    adapter_start: int
    adapter_end: int
    polya_end: int
    polya_end_topk: Optional[np.ndarray] = None
    trace: Optional[np.ndarray] = None


@dataclass
class DetectResults:
    success: bool

    signal_len: Optional[int] = None
    preloaded: Optional[int] = None

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
    polya_candidates: Optional[np.ndarray] = None

    rna_preloaded_start: Optional[int] = None
    rna_preloaded_len: Optional[int] = None
    rna_preloaded_mean: Optional[float] = None
    rna_preloaded_std: Optional[float] = None
    rna_preloaded_med: Optional[float] = None
    rna_preloaded_mad: Optional[float] = None

    start_peak_idx: Optional[int] = None
    start_peak_pa: Optional[float] = None
    start_peak_next_max_idx: Optional[int] = None
    start_peak_next_max_pa: Optional[float] = None
    start_peak_open_pore_idx: Optional[int] = None
    start_peak_open_pore_type: Optional[str] = None

    adapter_rna_median_shift: Optional[float] = None

    llr_adapter_end: Optional[int] = None
    llr_polya_end: Optional[int] = None

    cnn_adapter_end: Optional[int] = None
    cnn_polya_end: Optional[int] = None

    start_peak_adapter_end: Optional[int] = None
    start_peak_polya_end: Optional[int] = None  # NOTE: not used

    llr_trace: Optional[np.ndarray] = None

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

    def to_dict(self):
        return {
            **self.__dict__,
        }

    def update(self, d: dict):
        self.__dict__.update(d)


@dataclass
class ReadResult:
    read_id: Optional[str] = None
    success: bool = True
    fail_reason: Optional[str] = None

    detect_results: Optional[DetectResults] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        detect_dict = self.detect_results.to_dict() if self.detect_results else {}
        detect_dict.pop("fail_reason", None)
        return {
            "read_id": self.read_id,
            **detect_dict,
            "fail_reason": self.fail_reason,
        }
