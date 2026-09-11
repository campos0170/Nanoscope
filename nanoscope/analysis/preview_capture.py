"""Ephemeral, worker-local access to the exact arrays used by a detector.

This is an output-only preview seam: no arrays enter parameters, provenance or a
saved result. Ordinary detection neither copies nor retains anything here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionTraceCapture:
    current: np.ndarray | None = None
    baseline: np.ndarray | None = None
    count: int = 0

    def require_single(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count != 1 or self.current is None or self.baseline is None:
            raise ValueError("Reference preview requires exactly one detector trace; no substitute was computed.")
        return self.current, self.baseline


_capture: ContextVar[DetectionTraceCapture | None] = ContextVar("detection_trace_preview", default=None)


@contextmanager
def capture_detection_trace():
    capture = DetectionTraceCapture()
    token = _capture.set(capture)
    try:
        yield capture
    finally:
        _capture.reset(token)


def publish_detection_trace(current: np.ndarray, baseline: np.ndarray) -> None:
    capture = _capture.get()
    if capture is None:
        return
    capture.count += 1
    # Read-only views protect consumers without changing the detector arrays'
    # write flags or copying a full recording. Capture scope never crosses jobs.
    capture.current = np.asarray(current).view()
    capture.baseline = np.asarray(baseline).view()
    capture.current.setflags(write=False)
    capture.baseline.setflags(write=False)
