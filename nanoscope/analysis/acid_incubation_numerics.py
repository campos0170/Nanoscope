"""Small, explicitly checked numerical primitives for acid incubation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def median_filter_1d(values, size, mode='reflect'):
    """Exact odd-window median with SciPy's half-sample reflection semantics.

    SciPy 1.15.1's 1-D ndimage median filter can return incorrect values for
    quantized input (scipy/scipy#22333). Explicit symmetric padding and the
    rolling order statistic preserve the specified mathematics on all supported
    runtimes. Casting back preserves the notebook's float32 baseline contract.
    """
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        raise ValueError('Median filtering requires a nonempty 1-D signal')
    if isinstance(size, bool) or int(size) != size or size < 1 or size % 2 != 1:
        raise ValueError('Median window must be a positive odd integer')
    if mode not in ('reflect', 'nearest'):
        raise ValueError('Supported median padding modes are reflect and nearest')
    if not np.isfinite(values).all():
        raise ValueError('Median filtering requires finite input')
    radius = int(size) // 2
    padded = np.pad(values, (radius, radius), mode='symmetric' if mode == 'reflect' else 'edge')
    result = pd.Series(padded).rolling(int(size), center=True, min_periods=int(size)).median().to_numpy()
    return result[radius:radius + len(values)].astype(values.dtype, copy=False)
