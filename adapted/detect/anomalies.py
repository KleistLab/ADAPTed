"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from typing import Optional, Tuple

import numpy as np
from adapted.detect.utils import in_range


def find_open_pores(
    signal: np.ndarray,
    sig_range: Tuple[Optional[float], Optional[float]] = (200.0, None),
    min_obs_diff: int = 10,
):
    min, max = sig_range
    pos = np.argwhere(in_range(signal, min, max))

    if pos.size > 1:
        valid_pos = []
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_obs_diff:
                continue
            else:
                valid_pos.append(pos[i])
        if len(valid_pos) == 0:
            valid_pos = pos[-1]

        return np.array(valid_pos)

    return pos
