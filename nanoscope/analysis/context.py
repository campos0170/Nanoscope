"""Concrete :class:`AnalysisContext` (PORT_PLAN §4, gap #1).

A module receives calibrated data through an :class:`AnalysisContext` and nothing
else -- it never imports Qt, never touches :class:`~nanoscope.core.session.Session`,
never draws. :class:`SimpleAnalysisContext` is the plain, GUI-free implementation
of that Protocol (see :mod:`nanoscope.core.analysis`): the analysis worker builds
one wired to its progress/cancel callbacks, and tests build one around a synthetic
:class:`~nanoscope.core.document.CalibratedChannel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.document import CalibratedChannel
from ..core.types import RecordingMeta

__all__ = ["SimpleAnalysisContext"]


def _noop_progress(frac: float, msg: str = "") -> None:  # default progress sink
    return None


def _never_cancelled() -> bool:  # default cancel probe
    return False


@dataclass
class SimpleAnalysisContext:
    """A concrete :class:`~nanoscope.core.analysis.AnalysisContext`.

    ``channel``/``meta``/``params`` are the three things a module reads;
    ``progress_cb`` and ``cancel_flag`` are how it talks back to whoever is
    running it (the worker wires them to ``AnalysisWorker.progress.emit`` and its
    cancel ``threading.Event.is_set``). Both default to inert callables so a bare
    ``SimpleAnalysisContext(channel, meta, params)`` runs fine off the GUI, e.g.
    in tests or a notebook.
    """

    channel: CalibratedChannel
    meta: RecordingMeta
    params: Mapping[str, Any]
    progress_cb: Callable[[float, str], None] = field(default=_noop_progress)
    cancel_flag: Callable[[], bool] = field(default=_never_cancelled)

    def report_progress(self, frac: float, msg: str = "") -> None:
        self.progress_cb(frac, msg)

    def cancelled(self) -> bool:
        return bool(self.cancel_flag())
