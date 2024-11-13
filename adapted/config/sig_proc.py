"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import importlib.resources as pkg_resources
import logging
import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional, Tuple, Union

import toml
from adapted import models
from adapted._version import __version__
from adapted.config import config_files
from adapted.config.base import BaseConfig, NestedConfig, load_nested_config_from_file


@dataclass
class CoreConfig(BaseConfig):
    min_obs_adapter: int = 1000
    max_obs_adapter: int = 6500
    min_obs_polya: int = 100
    downscale_factor: int = 10
    max_obs_trace: int = 16000

    sig_norm_outlier_thresh: float = 5.0


@dataclass
class CNNBoundariesConfig(BaseConfig):
    cnn_detect: bool = True
    model_name: str = "rna004_130bps@v0.2.3.pth"
    polya_cand_k: int = 15
    fallback_to_llr_short_reads: bool = True

    def __post_init__(self):
        if self.cnn_detect:
            if self.model_name == "" or self.model_name is None:
                raise ValueError("model_name is required")
            elif not os.path.exists(self.model_name):
                try:
                    with pkg_resources.path(models, self.model_name) as repo_path:
                        if os.path.exists(repo_path):
                            pass
                        else:
                            raise FileNotFoundError(
                                f"model_name does not exist in package resources: {self.model_name}"
                            )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"model_name does not exist: {self.model_name}"
                    )


@dataclass
class CNNModelConfig(BaseConfig):
    downscale_factor: int = 10
    model_name: str = "rna004_130bps@v0.2.3.pth"


@dataclass
class LLRBoundariesConfig(BaseConfig):
    llr_detect: bool = False

    adapter_peak_prominence: float = 1.0
    adapter_peak_rel_height: float = 1.0
    adapter_peak_width: int = 1000

    polya_peak_prominence: float = 1.0
    polya_peak_rel_height: float = 0.5
    polya_peak_width: int = 50


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
    min_obs_post_loc: int = 300
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
    core: CoreConfig = CoreConfig()
    llr_boundaries: LLRBoundariesConfig = LLRBoundariesConfig()
    mvs_polya: MVSPolyAConfig = MVSPolyAConfig()
    real_range: RealRangeConfig = RealRangeConfig()
    streaming: Optional[StreamingConfig] = None
    cnn_boundaries: CNNBoundariesConfig = CNNBoundariesConfig()

    primary_method: Optional[str] = None
    primary_config: Optional[Union[LLRBoundariesConfig, CNNBoundariesConfig]] = None

    def __post_init__(self):
        self.update_primary_method()
        self.update_sig_preload_size()

    def update_sig_preload_size(self):
        self.sig_preload_size = self.core.max_obs_trace + (  # type: ignore
            (
                self.mvs_polya.search_window
                + max(self.mvs_polya.median_shift_window, self.mvs_polya.polyA_window)
            )
            if (self.mvs_polya.mvs_detect_check)
            else 0
        )

    def update_primary_method(self):
        llr_detect = self.llr_boundaries.llr_detect
        cnn_detect = self.cnn_boundaries.cnn_detect
        if llr_detect and cnn_detect:
            raise ValueError("Both LLR and CNN are enabled, please choose one")
        elif llr_detect:
            self.primary_method = "llr"
            self.primary_config = self.llr_boundaries
        elif cnn_detect:
            self.primary_method = "cnn"
            self.primary_config = self.cnn_boundaries
        else:
            raise ValueError("No primary method is enabled")

        self.check_cnn_downscale_factor()

    def check_cnn_downscale_factor(self):
        if self.primary_method != "cnn":
            return
        with pkg_resources.path(models, "config.toml") as config_path:
            model_config = toml.load(config_path)[
                self.cnn_boundaries.model_name.replace("@", "_").replace(".", "_")
            ]

        if model_config["downscale_factor"] != self.core.downscale_factor:
            msg = "CNN downscale factor and core downscale factor do not match"
            logging.error(msg)
            raise ValueError(msg)


def config_name_to_dict(config_name: str) -> Union[dict, MutableMapping[str, Any]]:
    with pkg_resources.path(config_files, f"{config_name}.toml") as config_path:
        return toml.load(config_path)


def get_config(config_name: str) -> SigProcConfig:
    with pkg_resources.path(config_files, f"{config_name}.toml") as config_path:
        return load_nested_config_from_file(config_path, SigProcConfig)


def chemistry_specific_config_name(
    chemistry: str, version: Optional[str] = None
) -> str:
    if version is None:
        version = __version__
    speed = config_files.speeds[chemistry.lower()]
    return f"{chemistry.lower()}_{speed}@v{version}"


def get_chemistry_specific_config(
    chemistry: str, version: Optional[str] = None
) -> SigProcConfig:
    if chemistry.lower() not in ["rna002", "rna004"]:
        msg = f"Unknown chemistry: {chemistry}"
        logging.error(msg)
        raise ValueError(msg)
    if version is None:
        version = __version__

    return get_config(chemistry_specific_config_name(chemistry, version))
