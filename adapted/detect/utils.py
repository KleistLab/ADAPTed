"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

from typing import Any, Optional, Tuple, Union

import numpy as np

LOCAL_RANGE_PCTLS = (85, 15)


def in_range(
    val: Union[float, np.ndarray],
    min: Union[Optional[float], float, np.ndarray],
    max: Union[Optional[float], float, np.ndarray],
) -> Union[bool, np.ndarray]:  # type: ignore
    min_ = -np.inf if min is None else min
    max_ = np.inf if max is None else max
    if np.ndim(val) == 0:
        return bool(min_ <= val <= max_)
    else:
        return np.asarray((min_ <= val) & (val <= max_))


def range_is_empty(
    range: Union[Tuple[Optional[float], Optional[float]], None],
) -> bool:
    if range is None:
        return True
    return (range[0] == -np.inf and range[1] == np.inf) or (
        range[0] == None and range[1] == None
    )
