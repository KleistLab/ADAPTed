"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from dataclasses import dataclass

from adapted.config.base import NestedConfig
from adapted.config.file_proc import BatchConfig, InputConfig, OutputConfig
from adapted.config.sig_proc import SigProcConfig


@dataclass
class Config(NestedConfig):
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    batch: BatchConfig = BatchConfig()
    sig_proc: SigProcConfig = SigProcConfig()
