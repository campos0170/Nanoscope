"""Shared builders for the change-C tests: the Event inspector's two hosts, the Window menu
and remembered child-window geometry. Not a test module (no ``test_`` prefix)."""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  (Qt bootstrap before any Qt import)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc

import numpy as np
import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1_000.0
SPANS = ((5_000, 5_450), (15_000, 15_550), (28_000, 28_700), (43_000, 43_500),
         (55_000, 55_400), (66_000, 66_600))
N = 80_000
#: The owner's 14" work area (1512x900 less the menu bar), and its window frame's content.
WORK_AREA = (1512, 875)
TITLE_BAR = 28


def signal() -> np.ndarray:
    rng = np.random.default_rng(91)
    data = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 7.0, N)
    for start, end in SPANS:
        data[start:end] -= 300.0
    return np.ascontiguousarray(data, dtype=np.float32)


@pytest.fixture
def private_settings(qapp):
    """A settings namespace of this module's own, so a dock placed here (it rides the saved
    windowState) never decides where another test's dock lands."""
    org, name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("InspectorHosting")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(name)


@pytest.fixture
def real_font(qapp):
    """The owner's 13 pt system font; widgets read it when they are built."""
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        yield
    finally:
        qapp.setFont(old)


def settle(ms: int = 30) -> None:
    QTest.qWait(ms)
    QtWidgets.QApplication.processEvents()


def build_window(qtbot, *, size=(1512, 847), show: bool = True,
                 detection: bool = True, name: str = "hosting.dat") -> MainWindow:
    """A wired MainWindow, optionally shown at ``size``, with a six-event detection."""
    import tempfile
    from pathlib import Path

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(*size)
    if show:
        window.show()
        settle()
        window.resize(*size)
        settle()
    if detection:
        data = signal()
        info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
        meta = RecordingMeta(
            path=Path(tempfile.mkdtemp(prefix="ns-hosting-")) / name, format_name="synthetic",
            n_samples=N, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
            channels=(info,))
        doc = TraceDocument(meta=meta, channels=(data,))
        doc_id = window._session.add_document(doc)
        window._create_default_layers(doc)
        window._on_selection(doc_id, 0)
        frame, provenance = detect_events(data, FS, baseline=I0, min_dwell_s=1e-4)
        assert len(frame) == len(SPANS), frame
        window._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={"min_dwell_s": 1e-4},
            calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
            payload=EventSet(frame, provenance)))
        settle()
    return window


def close_window(window: MainWindow) -> None:
    """A quit (so _save_settings runs) that never raises the unsaved-work modal."""
    window._dirty = False
    for name in ("_controller", "_analysis_controller", "_batch_controller"):
        controller = getattr(window, name, None)
        if controller is not None:
            controller.abort()
    window.close()
    settle(10)


def docked(window: MainWindow):
    """Dock the inspector (opening it) and return (inspector, dock)."""
    window.set_curation_host("event_inspector", "dock", present=True)
    settle()
    return window._event_inspector, window._inspector_dock


def accepted(window: MainWindow, event_id: int) -> bool:
    df = window._analysis_panel.current_detection().payload.df
    return bool(df.loc[df["event_id"] == event_id, "accepted"].iloc[0])


def dead_wrappers() -> int:
    """QObject wrappers whose C++ object is gone -- the zombies this PySide6 leaks."""
    n = 0
    for obj in gc.get_objects():
        # type(), not isinstance(): a lazy module proxy answers __class__ by importing.
        if issubclass(type(obj), QtCore.QObject):
            try:
                if not shiboken6.isValid(obj):
                    n += 1
            except Exception:  # noqa: BLE001
                pass
    return n
