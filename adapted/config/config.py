"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from dataclasses import dataclass, field

from adapted.config.base import NestedConfig
from adapted.config.file_proc import BatchConfig, InputConfig, OutputConfig
from adapted.config.sig_proc import SigProcConfig


@dataclass
class Config(NestedConfig):
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    sig_proc: SigProcConfig = field(default_factory=SigProcConfig)
