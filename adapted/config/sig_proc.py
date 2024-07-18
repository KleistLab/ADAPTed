"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Optional, Tuple

from adapted.config import config_files
from adapted.config.base import BaseConfig, NestedConfig, load_nested_config_from_file


@dataclass
class LLRBoundariesConfig(BaseConfig):
    # default values RNA004

    sig_norm_winsor_window: int = 5
    sig_norm_outlier_thresh: float = 5.0

    min_obs_adapter: int = 1500
    max_obs_trace: int = 6500

    # in case of polya determination based on early stopping, this should be a small value
    adapter_trace_tail_trim: int = 5
    adapter_trace_stride: int = 20

    min_obs_polya: int = 100

    # polya_trace_start < min_obs_polya helps with polyA detection
    polya_trace_start: int = 40
    polya_trace_stride: int = 5

    adapter_trace_early_stop_window: int = 1000
    adapter_trace_early_stop_stride: int = 500
    polya_trace_early_stop_window: int = 100
    polya_trace_early_stop_stride: int = 20

    adapter_peak_prominence: float = 1.0
    adapter_peak_rel_height: float = 1.0
    adapter_peak_width: int = 1000

    polya_peak_prominence: float = 1.0
    polya_peak_rel_height: float = 0.5
    polya_peak_width: int = 50

    refine_polya_atol: int = 20
    refine_smooth_sigma: int = 10


@dataclass
class MVSPolyAConfig(BaseConfig):
    mvs_detect_check: bool = True
    mvs_detect_overwrite: bool = False

    search_window: int = 500
    pA_mean_window: int = 20
    pA_mean_range: Tuple[Optional[float], Optional[float]] = (None, None)
    pA_var_window: int = 100
    pA_var_range: Tuple[Optional[float], Optional[float]] = (None, 20.0)
    median_shift_range: Tuple[Optional[float], Optional[float]] = (20.0, None)
    median_shift_window: int = 2000
    polyA_window: int = 300
    polyA_med_range: Tuple[Optional[float], Optional[float]] = (90.0, 130.0)
    polyA_local_range: Tuple[Optional[float], Optional[float]] = (0.0, 15.0)

    pA_mean_adapter_med_scale_range: Tuple[Optional[float], Optional[float]] = (
        1.3,
        None,
    )


@dataclass
class RealRangeConfig(BaseConfig):
    detect_open_pores: bool = True
    real_signal_check: bool = True

    mean_window: int = 300
    mean_start_range: Tuple[Optional[float], Optional[float]] = (50.0, 100.0)
    mean_end_range: Tuple[Optional[float], Optional[float]] = (75.0, 120.0)
    max_obs_local_range: int = 5000
    local_range: Tuple[Optional[float], Optional[float]] = (10.0, 30.0)
    adapter_mad_range: Tuple[Optional[float], Optional[float]] = (3.0, 12.0)


# NOTE: not used
@dataclass
class MMAdapterStartConfig(BaseConfig):
    detect_adapter_start: bool = False
    window: int = 100
    min_obs_adapter: int = 2500
    min_shift: float = 20.0
    min_pA_current: float = 90.0


@dataclass
class StreamingConfig(BaseConfig):
    # default values RNA002

    min_obs_adapter: int = 2500
    min_obs_post_loc: int = (
        300  # used for median shift calculation and polyA statistics; determines min offset between polyA and chunk_end
    )
    search_increment_step: int = 100

    pA_mean_window: int = 20
    pA_mean_range: Tuple[Optional[float], Optional[float]] = (90.0, 130.0)
    pA_var_window: int = 100
    pA_var_range: Tuple[Optional[float], Optional[float]] = (None, 20.0)

    median_shift_window: int = 2000
    median_shift_range: Tuple[Optional[float], Optional[float]] = (20.0, None)

    polyA_window: int = 300
    polyA_med_range: Tuple[Optional[float], Optional[float]] = (90.0, 130.0)
    polyA_local_range: Tuple[Optional[float], Optional[float]] = (0.0, 10.0)


@dataclass
class SigProcConfig(NestedConfig):
    llr_boundaries: LLRBoundariesConfig = LLRBoundariesConfig()
    mvs_polya: MVSPolyAConfig = MVSPolyAConfig()
    real_range: RealRangeConfig = RealRangeConfig()
    streaming: Optional[StreamingConfig] = None

    def __post_init__(self):
        self.sig_preload_size = (
            self.llr_boundaries.max_obs_trace
            + (
                self.mvs_polya.search_window
                + max(self.mvs_polya.median_shift_window, self.mvs_polya.polyA_window)
            )
            if self.mvs_polya.mvs_detect_check
            else 0
        )


def get_config(config_name: str) -> SigProcConfig:
    with pkg_resources.path(config_files, f"{config_name}.toml") as config_path:
        return load_nested_config_from_file(config_path, SigProcConfig)
