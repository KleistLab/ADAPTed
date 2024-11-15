"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adapted.container_types import Boundaries


@dataclass
class Partition:
    start: Optional[int]
    len: Optional[int]
    mean: Optional[float]
    std: Optional[float]
    med: Optional[float]
    mad: Optional[float]

    def to_dict(self, name: str = ""):
        prefix = name + "_" if name else ""
        return {
            f"{prefix}start": self.start,
            f"{prefix}len": self.len,
            f"{prefix}mean": self.mean,
            f"{prefix}std": self.std,
            f"{prefix}med": self.med,
            f"{prefix}mad": self.mad,
        }


@dataclass
class Partitions:
    adapter: Partition
    polya: Partition
    rna: Partition

    def to_dict(self, name: str = ""):
        prefix = name + "_" if name else ""
        return {
            **self.adapter.to_dict(name=prefix + "adapter"),
            **self.polya.to_dict(name=prefix + "polya"),
            **self.rna.to_dict(name=prefix + "rna"),
        }


def calc_partitions(
    signal: np.ndarray,
    boundaries: Boundaries,
) -> Partitions:
    adapter = calc_partition_stats(
        signal, boundaries.adapter_start, boundaries.adapter_end
    )
    polya = calc_partition_stats(signal, boundaries.adapter_end, boundaries.polya_end)
    rna = calc_partition_stats(signal, boundaries.polya_end, signal.size)
    return Partitions(adapter, polya, rna)


def calc_partitions_from_vals(
    signal: np.ndarray,
    adapter_start: Optional[int],
    adapter_end: Optional[int],
    polya_end: Optional[int],
    polya_truncated: Optional[bool] = False,
) -> Partitions:
    adapter = calc_partition_stats(signal, adapter_start, adapter_end)
    polya = calc_partition_stats(signal, adapter_end, polya_end)
    if not polya_truncated:
        rna = calc_partition_stats(signal, polya_end, signal.size)
    else:
        rna = Partition(None, None, None, None, None, None)
    return Partitions(adapter, polya, rna)


def calc_partition_stats(
    signal: np.ndarray, start: Optional[int], end: Optional[int]
) -> Partition:
    if start is None or end is None or end <= start:
        return Partition(start, None, None, None, None, None)

    length = end - start
    if length == 0:
        return Partition(start, 0, None, None, None, None)
    sig = signal[start:end]
    signal_mean = float(np.mean(sig))
    signal_std = float(np.std(sig))
    signal_med = float(np.median(sig))
    signal_mad = float(np.median(np.abs(sig - signal_med)))

    return Partition(start, length, signal_mean, signal_std, signal_med, signal_mad)
