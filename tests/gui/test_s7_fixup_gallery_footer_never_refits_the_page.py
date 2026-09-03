"""Sweep-7 fix-up B1 / CW-3 — the gallery's footer has a FIXED height, so no sentence on the
status line can re-fit the "Fit" page or re-cut the representative sample.

Measured on the owner's -1000 mV recording (110 events, gallery at 1000x700, page size Fit)
after the S7-POP1 fix: one Preprocess knob on the MAIN window pushed the banner's ~720-char
sentence onto the word-wrapped status QLabel, the footer grew 45 -> 90 px, the viewport lost
45 px, the viewport's Resize reached ``_apply_fit_page_size`` and the page went 24 -> 18
tiles / 5 -> 7 pages; with "Representative sample" ticked the dwell ladder was re-cut and 18
of the 24 events on screen were replaced -- exactly what the page-size tooltip forbids ("a
sample quietly cut to fit a window is not a sample"). The knob turned back re-cut it again.

Fixed at the invariant, not at the sentence: the status label reserves a constant number of
lines (``_StatusLine``), so the tile grid's geometry depends on the window and on nothing the
line says. A line longer than the budget is elided on the glass with the whole of it in
``text()`` and in the tooltip -- never silently cut. Every test below drives the real window.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N_EVENTS = 60
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000

#: The banner's sentence is ~720 characters on the owner's project; this one is longer still,
#: so the test cannot pass by the sentence happening to fit.
_LONG_NOTE = ("Depth, frac_depth, ECD, mean/max deviation and SNR are stale — measured with "
              "an open-pore estimator (rough 0.5 s / base 0.1 s) that is no longer the "
              "configured one (rough 2.0 s / base 0.1 s); " * 5).strip()


def _signal() -> np.ndarray:
    rng = np.random.default_rng(23)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "footer.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov),
        baseline_fingerprint=w._analysis_panel.baseline_panel.config().fingerprint(FS)))
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.show()
    gal.activateWindow()
    gal.resize(1000, 700)
    qtbot.wait(80)
    assert gal.page_size_box.currentText() == "Fit"
    assert gal.page_count() >= 2, "the fixture must span several Fit pages"
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _geometry(gal) -> tuple:
    """Everything the sentence must not move: the grid, the footer, the page."""
    return (gal.view.geometry().getRect(), gal.view.viewport().height(),
            gal.status.geometry().getRect(), gal._model.page_size(), gal.page_count())


def _painted(label: QtWidgets.QLabel) -> str:
    """What is on the glass, as opposed to what ``text()`` reports."""
    return QtWidgets.QLabel.text(label)


# ================================================================================================
def test_the_stale_sentence_moves_neither_the_tile_grid_nor_the_fit_page(wired, qtbot):
    """THE MEASURED DEFECT: the sentence arrives, the footer grows, the grid shrinks, Fit
    re-pages. Now the grid and the page are what they were, and the sentence is still on
    the line (``text()``) rather than dropped."""
    w, doc_id, gal = wired
    gal.set_sort("Depth (deepest first)")
    qtbot.wait(30)
    before = _geometry(gal)
    gal.set_stale_columns(REMEASURED_COLUMNS, _LONG_NOTE)
    qtbot.wait(120)
    assert _LONG_NOTE in gal.status.text(), "the disclosure must stay on the line"
    assert _geometry(gal) == before, (
        f"the sentence moved the grid / re-fitted the page: {before} -> {_geometry(gal)}")

    gal.set_stale_columns(())
    qtbot.wait(120)
    assert _geometry(gal) == before


def test_the_representative_sample_is_not_recut_by_a_sentence(wired, qtbot):
    """With the sample ticked the page size IS the sample size; the sentence must not touch
    either. Measured before: 24 -> 18 ladder positions and 18 of 24 events replaced."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_representative(True)
    qtbot.wait(60)
    assert gal.is_representative()
    sample = [int(df.iloc[r]["event_id"]) for r in gal._model.df_rows()]
    assert 0 < len(sample) < N_EVENTS, "the sample must be a proper thinning of the subset"

    gal.set_stale_columns(REMEASURED_COLUMNS, _LONG_NOTE)
    qtbot.wait(120)
    assert [int(df.iloc[r]["event_id"]) for r in gal._model.df_rows()] == sample, (
        "the sentence re-cut the representative sample")
    gal.set_stale_columns(())
    qtbot.wait(120)
    assert [int(df.iloc[r]["event_id"]) for r in gal._model.df_rows()] == sample


def test_every_note_the_line_can_carry_leaves_the_footer_height_alone(wired, qtbot):
    """The invariant, not the one sentence: the hotkey note, the missed-event note, the
    off-page note and the stale sentence all land on the same line, and none of them
    changes the footer's height by a pixel."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    h0 = gal.status.height()
    assert h0 > 0
    lines = h0 // gal.status.fontMetrics().lineSpacing()
    assert lines >= 2, f"a {h0} px footer is not a multi-line budget"

    # A digit with no palette behind it: the hotkey note.
    idx = gal._model.index(0, 0)
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     gal.view.visualRect(idx).center())
    QTest.keyClick(gal.view, Qt.Key_1)
    qtbot.wait(40)
    assert "is not armed" in gal.status.text()
    assert gal.status.height() == h0

    # An inbound show_event the filter excludes: the missed note.
    victim = int(df.iloc[5]["event_id"])
    df.iloc[5, df.columns.get_loc("accepted")] = False
    gal.refresh()
    gal.set_filter("Accepted only")
    qtbot.wait(40)
    gal.show_event(victim)
    qtbot.wait(40)
    assert f"Event {victim} is filtered out" in gal.status.text(), gal.status.text()
    assert gal.status.height() == h0

    # A sweep rejected under Accepted only: the off-page note.
    first = gal._model.index(0, 0)
    last = gal._model.index(3, 0)
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     gal.view.visualRect(first).center())
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, Qt.ShiftModifier,
                     gal.view.visualRect(last).center())
    qtbot.wait(20)
    assert len(gal._selected_rows()) == 4
    QTest.keyClick(gal, Qt.Key_R)
    qtbot.wait(60)
    assert "4 selected tiles left this page" in gal.status.text(), gal.status.text()
    assert gal.status.height() == h0

    # ...and the banner's sentence on top of all of it.
    gal.set_stale_columns(REMEASURED_COLUMNS, _LONG_NOTE)
    qtbot.wait(120)
    assert gal.status.height() == h0
    assert gal.view.viewport().height() > 0


def test_a_line_longer_than_the_footer_is_elided_on_the_glass_and_whole_in_text_and_tooltip(
        wired, qtbot):
    """A fixed budget must never become a silent cut: what does not fit is elided with an
    ellipsis on the glass, ``text()`` still returns the whole line (every assertion in this
    suite reads it), and the tooltip carries it in full. When the line fits, nothing is
    elided and there is no tooltip repeating the visible text."""
    w, doc_id, gal = wired
    gal.set_stale_columns(REMEASURED_COLUMNS, _LONG_NOTE)
    qtbot.wait(120)
    full = gal.status.text()
    shown = _painted(gal.status)
    assert _LONG_NOTE in full
    assert shown != full and shown.endswith("…"), (
        f"a {len(full)}-char line in a {gal.status.height()} px footer was not elided: "
        f"{shown[-60:]!r}")
    assert gal.status.toolTip() == full, "the whole line must be reachable"
    assert shown.rstrip("…") == full[:len(shown) - 1], "the visible part is a prefix"

    gal.set_stale_columns(())
    qtbot.wait(120)
    full = gal.status.text()
    shown = _painted(gal.status)
    clipped = gal.status.clipped()
    assert (shown != full) is clipped
    assert gal.status.toolTip() == (full if clipped else "")
