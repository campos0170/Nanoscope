"""Regression tests for EXP-02: a failed measurement-CSV export must be REPORTED.

``MeasurePanel.export_csv`` used to call ``write_measurements_csv`` with no try/except —
a PermissionError escaped the clicked-signal slot: PySide printed a traceback to the
console, the user saw no dialog, no status change and no file, and could close the app
believing the hand-measured table was saved. The fix reports the failure the way every
MainWindow export does ("Could not write the export:" + the real cause) and says nothing
succeeded. Driven through the real widgets offscreen, dialog included (patched to capture,
never to show — the static ``QMessageBox.warning`` builds its own modal in C++, so an
unpatched regression would HANG here rather than fail).
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

import nanoscope.gui.measure_panel as measure_panel_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_BLOCKADE = (40_000, 60_000)


# ================================================================================================
# builders
# ================================================================================================

def _square_trace() -> np.ndarray:
    x = np.full(N, 100.0, dtype=np.float32)
    x[_BLOCKADE[0]:_BLOCKADE[1]] = 70.0
    return x


def _doc(tmp_path, *, name: str) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(_square_trace(),))


def _panel_with_one_row(window: MainWindow, tmp_path):
    """A MeasurePanel holding one real hand measurement, via the real cursor path."""
    session = window._session
    doc = _doc(tmp_path, name="m.dat")
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[_BLOCKADE[0]]),
                                          float(vals[_BLOCKADE[1]]))
    assert mp.add_current() is True
    return mp


def _readonly_dir(tmp_path, request):
    """A directory the test cannot write into — restored writable at teardown so pytest
    can clean the tmp tree up."""
    ro = tmp_path / "readonly"
    ro.mkdir()
    os.chmod(ro, 0o555)
    request.addfinalizer(lambda: os.chmod(ro, 0o755))
    return ro


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def warnings_shown(monkeypatch):
    """Capture (title, text) of every QMessageBox.warning instead of showing it."""
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda _parent, title, text, *a, **k: shown.append((title, text))))
    return shown


# ================================================================================================
# the failure is reported, and nothing claims success
# ================================================================================================

def test_export_csv_permission_error_reports_and_returns_none(
        window, tmp_path, request, warnings_shown):
    mp = _panel_with_one_row(window, tmp_path)
    ro = _readonly_dir(tmp_path, request)

    out = mp.export_csv(str(ro / "measurements.csv"))

    assert out is None, "a failed export must not return a path as if written"
    assert not (ro / "measurements.csv").exists()
    # The dialog names the real cause, in the same words the MainWindow exports use.
    assert len(warnings_shown) == 1
    title, text = warnings_shown[0]
    assert title == "Export measurements"
    assert "Could not write the export:" in text
    assert "Permission denied" in text          # the OS's own words reach the user
    # The status line says nothing succeeded — no "Wrote N measurement(s)".
    status = mp._status.text()
    assert "Wrote" not in status
    assert "failed" in status.lower()


def test_export_csv_button_click_into_readonly_dir_reports(
        window, tmp_path, request, warnings_shown, monkeypatch):
    """The real user gesture: the Export CSV… button with the file dialog answering a
    path inside an unwritable folder. No exception may escape the slot."""
    mp = _panel_with_one_row(window, tmp_path)
    ro = _readonly_dir(tmp_path, request)
    monkeypatch.setattr(measure_panel_mod, "ask_export_path",
                        lambda *a, **k: ro / "measurements.csv")

    mp._export_button.click()

    assert not (ro / "measurements.csv").exists()
    assert len(warnings_shown) == 1
    assert "Could not write the export:" in warnings_shown[0][1]
    assert "Wrote" not in mp._status.text()


def test_export_csv_success_still_reports_success(window, tmp_path, warnings_shown):
    """The guard must not dim a genuine success: file written, status says so, no dialog."""
    mp = _panel_with_one_row(window, tmp_path)

    out = mp.export_csv(str(tmp_path / "measurements.csv"))

    assert out is not None and out.exists()
    assert warnings_shown == []
    assert "Wrote 1 measurement(s)" in mp._status.text()
