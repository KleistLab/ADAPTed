"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from adapted.config.base import BaseConfig


@dataclass
class BatchConfig(BaseConfig):
    num_proc: int = -1  # default to number of cores
    batch_size: int = 4000
    minibatch_size: int = 50


@dataclass
class OutputConfig(BaseConfig):
    output_dir: str = field(default="")

    save_llr_trace: bool = False


@dataclass
class InputConfig(BaseConfig):
    files: List[str] = field(default_factory=list)
    read_ids: Union[List[str], np.ndarray] = field(default_factory=list)


@dataclass
class TaskConfig(BaseConfig):
    llr_return_trace: bool = False
