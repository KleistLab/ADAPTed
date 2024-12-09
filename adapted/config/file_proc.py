"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import os
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from adapted.config.base import BaseConfig


@dataclass
class BatchConfig(BaseConfig):
    num_proc: int = -1  # default to number of cores
    batch_size_output: int = 4000
    minibatch_size: int = 1000
    bidx_pass: int = 0
    bidx_fail: int = 0


@dataclass
class OutputConfig(BaseConfig):
    output_dir: str = field(default="")

    output_subdir_fail: str = "failed_reads"
    output_subdir_boundaries: str = "boundaries"

    def __post_init__(self):
        self.output_dir_fail = os.path.join(self.output_dir, self.output_subdir_fail)
        self.output_dir_boundaries = os.path.join(
            self.output_dir, self.output_subdir_boundaries
        )

        # create output directories if valid
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir_fail, exist_ok=True)
            os.makedirs(self.output_dir_boundaries, exist_ok=True)


@dataclass
class InputConfig(BaseConfig):
    files: List[str] = field(default_factory=list)
    read_ids: Union[List[str], np.ndarray] = field(default_factory=list)
    continue_from: str = field(default="")
    n_reads: int = -1
