"""Regression tests for EXP-03: a failed "Save image…" must be reported, never celebrated.

pyqtgraph's ``ImageExporter.export`` ends in ``QImage.save``, which reports failure by
RETURNING False — it raises nothing. All five plot panels ignored that return: an
export into a read-only folder wrote no file while four of them appended
"Saved image → <name>" to their details pane and every one of them handed the path
back as if written. The fix checks the export result in each panel; on failure the
user gets a dialog naming the write failure (in the "Could not write" words the
MainWindow exports use) and ``save_image`` returns None.

The dialog is patched to capture, never to show — the static ``QMessageBox.warning``
builds its own modal in C++, so an unpatched regression would HANG here rather than
fail.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui.event_plot_panel import EventPlotWindow
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.gui.psd_panel import PSDWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

#: Every panel with a ``save_image`` — the defect was copied into all five.
_PANELS = (PlotsWindow, PSDWindow, IVWindow, VDepWindow, EventPlotWindow)
#: The four with a Details pane (a text box) that records "Saved image → <name>".
#: EventPlotWindow has no such pane: its ``details_text()`` (779d4ec) is computed
#: figure-bundle provenance, so only the negative check below applies to it.
_DETAILS_PANE_PANELS = (PlotsWindow, PSDWindow, IVWindow, VDepWindow)


# ================================================================================================
# fixtures
# ================================================================================================

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


def _window(cls, qtbot):
    w = cls(Session())
    qtbot.addWidget(w)
    return w


# ================================================================================================
# the failure is reported, and nothing claims success
# ================================================================================================

@pytest.mark.parametrize("cls", _PANELS, ids=lambda c: c.__name__)
def test_save_image_into_readonly_dir_reports_and_returns_none(
        cls, qapp, qtbot, readonly_dir, warnings_shown):
    w = _window(cls, qtbot)

    out = w.save_image(str(readonly_dir / "plot.png"))

    assert out is None, "a failed write must not return a path as if written"
    assert not (readonly_dir / "plot.png").exists()
    # The dialog says the WRITE failed (not the data), and points at the folder.
    assert len(warnings_shown) == 1
    _title, text = warnings_shown[0]
    assert "Could not write the image" in text
    assert "plot.png" in text
    # Nothing that reports Details — a pane, or EventPlotWindow's provenance text — may
    # carry the success line.
    if hasattr(w, "details_text"):
        assert "Saved image" not in w.details_text()


@pytest.mark.parametrize("cls", _PANELS, ids=lambda c: c.__name__)
def test_save_image_success_still_returns_path_and_says_so(
        cls, qapp, qtbot, tmp_path, warnings_shown):
    """The fix must not turn a good write into a refusal: the path comes back, the
    file exists, no warning fires, and the details line (where there is one) appears
    only now — when the file actually landed."""
    w = _window(cls, qtbot)
    target = tmp_path / "plot.png"

    out = w.save_image(str(target))

    assert out is not None and out == target
    assert target.exists() and target.stat().st_size > 0
    assert warnings_shown == []
    if cls in _DETAILS_PANE_PANELS:
        assert "Saved image → plot.png" in w.details_text()
