"""Small compatibility shims for the ported DSP code.

The target Anaconda environment pins numpy 1.26.4, which predates
``np.trapezoid`` (added in numpy 2.0). The SOURCE ``stats.py`` calls
``np.trapezoid``; bind the name here so the port is byte-faithful and a numpy
upgrade later is transparent. Do NOT upgrade numpy to get ``np.trapezoid`` —
that risks the neo/h5py/nptdms reader stack the app depends on (PORT_PLAN D2/R1).
"""

from __future__ import annotations

import numpy as np

# numpy>=2.0 exposes ``trapezoid``; 1.26.x only has the deprecated ``trapz``.
# This must be a LAZY fallback: the default argument to ``getattr`` is evaluated
# before the call, and numpy 2.5 removed ``np.trapz`` altogether.
try:
    trapezoid = np.trapezoid
except AttributeError:  # numpy 1.26
    trapezoid = np.trapz

__all__ = ["trapezoid"]
