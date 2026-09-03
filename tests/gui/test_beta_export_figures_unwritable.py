"""Regression tests for EXP-01: figure export names a write failure as a write failure.

Exporting figures into a folder the user cannot write raised PermissionError at
staging for EVERY figure; each one was swallowed by the per-figure
``except Exception: failed.append(name)`` and the summary said "No figures could be
exported from the current detection result." — blaming the user's data for a folder
permission. A scientist reads that and re-runs detection instead of fixing the folder.

The fix treats an ``OSError`` from the save as what it is — environmental, hitting
every figure identically — and reports it in the same "Could not write the export"
words the events export uses, after abandoning the staged set (so the folder is
untouched, and the dialog says so). Genuinely per-figure failures (a figure that
cannot be BUILT) keep today's behaviour.

The dialog is patched to capture, never to show — an unpatched static
``QMessageBox.warning`` builds a real modal and would hang offscreen.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 40_000


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    """A quiet baseline with three unmistakable blockades — enough for detection."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path: Path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, qtbot, tmp_path):
    """The window with one synthetic recording detected on its current channel."""
    doc = _synth_doc(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    window._analysis_panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert window._analysis_panel.current_detection() is not None
    return window


@pytest.fixture
def warnings_shown(monkeypatch):
    """Capture (title, text) of every QMessageBox.warning instead of showing it."""
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda _parent, title, text, *a, **k: shown.append((title, text))))
    return shown


@pytest.fixture
def readonly_dir(tmp_path, request):
    """A directory the test cannot write into — restored writable at teardown so
    pytest can clean the tmp tree up."""
    ro = tmp_path / "readonly"
    ro.mkdir()
    os.chmod(ro, 0o555)
    request.addfinalizer(lambda: os.chmod(ro, 0o755))
    return ro


def _export_to(window, base: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(base), "PDF (*.pdf)")))
    window._export_figures()


# ================================================================================================
# an unwritable folder is reported as a write failure, never as bad data
# ================================================================================================

def test_export_into_readonly_dir_names_the_write_failure(
        detected, monkeypatch, readonly_dir, warnings_shown):
    _export_to(detected, readonly_dir / "figures.pdf", monkeypatch)

    assert len(warnings_shown) == 1
    _title, text = warnings_shown[0]
    # The real cause, in the events export's own words...
    assert "Could not write the export" in text
    assert "Permission denied" in text
    # ...never the old sentence that blamed the detection result.
    assert "detection result" not in text
    # The staged set was abandoned: the folder is exactly as it was.
    assert list(readonly_dir.iterdir()) == []


# ================================================================================================
# the fix must not touch the healthy paths
# ================================================================================================

def test_export_into_writable_dir_still_writes_and_says_so(
        detected, monkeypatch, tmp_path, warnings_shown):
    out = tmp_path / "figs"
    out.mkdir()
    _export_to(detected, out / "figures.pdf", monkeypatch)

    assert warnings_shown == []
    for name in ("dwell", "depth", "scatter"):
        assert (out / f"figures_{name}.pdf").exists()
        assert (out / f"figures_{name}.png").exists()
    assert "Wrote" in detected.statusBar().currentMessage()


def test_a_figure_that_cannot_be_built_stays_per_figure(
        detected, monkeypatch, tmp_path, warnings_shown):
    """A genuinely per-figure failure (the figure cannot be BUILT) must keep today's
    behaviour: the rest of the export still ships, and the write-failure dialog does
    not fire — that sentence is reserved for the environment."""
    from nanoscope.plot import figures

    def _boom(*a, **k):
        raise RuntimeError("scatter cannot be built")
    monkeypatch.setattr(figures, "scatter_dwell_depth", _boom)

    out = tmp_path / "figs2"
    out.mkdir()
    _export_to(detected, out / "figures.pdf", monkeypatch)

    assert not any("Could not write the export" in t for _ti, t in warnings_shown)
    assert (out / "figures_dwell.pdf").exists()
    assert (out / "figures_depth.pdf").exists()
    assert not (out / "figures_scatter.pdf").exists()
