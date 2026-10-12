"""Staleness disclosures must not move the waveform or its review controls.

The scrollable details sidebar now lets the complete warning wrap without stealing
height from the plot. Empty warnings reserve no space. The legacy fixed-line note
helper retains its separate elision contract, but no longer constrains this sidebar.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.baseline import BaselineConfig
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_inspector import EventInspectorWindow

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]

# The three sentences the analysis panel's banner actually raises (analysis_panel.py,
# _baseline_stale_message / _check_staleness), in their real shapes and lengths -- the
# open-pore one names BOTH estimators through BaselineConfig.describe().
_OLD = BaselineConfig().describe()
NOTE_BASELINE = (
    f"The open-pore estimator changed since this detection ran: these events were measured "
    f"with {_OLD}, and {_OLD} is configured now. Depth, frac_depth, ECD, mean/max deviation "
    "and SNR are stale.")
NOTE_PREDATES = (
    "These events were measured with block_median @ 10 ms (this table predates the baseline "
    "stamp) — this table predates the open-pore estimator now in use. Depth, frac_depth, ECD, "
    "mean/max deviation and SNR can change by more than 10x on recordings whose events are "
    "long compared with half that window. Nothing has been re-measured: use Analysis ▸ "
    "Compare baseline estimators… to see the size of the change on this recording first.")
NOTE_CALIBRATION = (
    "Calibration changed since this detection ran — the blockade depths and ECDs are "
    "computed against the old gain/offset and are now stale.")


# ================================================================================================
# builders (a Session + the real detector; no MainWindow -- the seam is the inspector's own)
# ================================================================================================

def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _session_with_doc(tmp_path) -> tuple[Session, str, int]:
    arr = _event_trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_p1.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    return session, doc_id, 0


def _shown_inspector(qtbot, tmp_path, size: tuple[int, int] | None) -> EventInspectorWindow:
    session, doc_id, ch = _session_with_doc(tmp_path)
    cal = session.calibrated(doc_id, ch)
    df, prov = detect_events(cal.data, cal.sample_rate_hz, min_dwell_s=5e-4)
    assert len(df) == len(_EVENTS)
    insp = EventInspectorWindow(session)
    qtbot.addWidget(insp)
    insp.set_events(doc_id, ch, EventSet(df, prov).df)
    if size is not None:
        insp.resize(*size)
    insp.show()
    qtbot.wait(60)
    insp.show_event(int(df.iloc[0]["event_id"]))
    qtbot.wait(60)
    assert insp.current_event_id() is not None
    return insp


def _geometry(insp: EventInspectorWindow) -> dict:
    """The stable work area; sidebar contents may grow and scroll independently."""
    plot = insp._plot_widget
    vb = insp._plot.getViewBox().geometry()
    return {
        "plot": (plot.mapTo(insp, QtCore.QPoint(0, 0)).toTuple(), plot.size().toTuple()),
        "viewbox": (round(vb.x()), round(vb.y()), round(vb.width()), round(vb.height())),
        "review_controls": tuple(
            (widget.mapTo(insp, QtCore.QPoint()).toTuple(), widget.size().toTuple())
            for widget in (insp._prev_btn, insp._index_spin, insp._next_btn,
                           insp._accept_btn, insp._reject_btn, insp._accept_check)),
        "window": insp.size().toTuple(),
        "min_width": insp.minimumSizeHint().width(),
    }


def _flip(insp, qtbot, columns, note) -> None:
    insp.set_stale_columns(columns, note)
    qtbot.wait(80)          # a layout request is a posted event; let it land (or not)


# ================================================================================================
# the invariant: the badge changes text, and nothing else about the window's geometry
# ================================================================================================

@pytest.mark.parametrize("note", [NOTE_BASELINE, NOTE_PREDATES, NOTE_CALIBRATION],
                         ids=["open-pore changed", "predates the stamp", "calibration"])
def test_the_staleness_flip_leaves_the_plot_review_controls_and_window_where_they_were(
        qapp, qtbot, tmp_path, note):
    insp = _shown_inspector(qtbot, tmp_path, (1000, 800))
    clean = _geometry(insp)
    clean_lines = insp._readout.text().count("\n") + 1
    assert clean["window"] == (1000, 800)

    _flip(insp, qtbot, REMEASURED_COLUMNS, note)
    stale = _geometry(insp)
    assert stale == clean, f"the badge moved the window:\n  {clean}\n  {stale}"
    assert clean_lines == 10, clean_lines
    # ...while the disclosure is fully there: the sentence in the note, whole and shown
    # whole (nothing elided at this width), and the per-line marks in the readout.
    assert insp.stale_note() == note
    assert QtWidgets.QLabel.text(insp._stale_label) == f"⚠ {note}"
    assert insp._stale_label.toolTip() == ""
    assert insp._stale_label.height() >= insp._stale_label.heightForWidth(
        insp._stale_label.width())
    assert insp._readout.text().count("\n") + 1 == clean_lines, insp._readout.text()
    assert "⚠ " not in insp._readout.text(), "the sentence is a note, not readout lines"
    depth_line = next(l for l in insp._readout.text().splitlines() if l.startswith("median depth:"))
    assert depth_line.endswith(" ⚠"), depth_line
    assert "⚠" not in insp._plot.titleLabel.text

    # The CLEAR direction (the banner's "Re-measure depths") moved the plot back before.
    _flip(insp, qtbot, (), "")
    assert _geometry(insp) == clean
    assert insp.stale_note() == "" and QtWidgets.QLabel.text(insp._stale_label) == ""
    assert "⚠" not in insp._readout.text()


def test_at_the_default_size_the_window_itself_does_not_grow_when_the_badge_arrives(
        qapp, qtbot, tmp_path):
    """CW-2's second measurement: at the window's default size the top-level window grew
    640 -> 653 px when the sentence arrived -- the readout's sizeHint pushed the frame."""
    insp = _shown_inspector(qtbot, tmp_path, None)
    clean = _geometry(insp)
    _flip(insp, qtbot, REMEASURED_COLUMNS, NOTE_BASELINE)
    assert _geometry(insp) == clean
    _flip(insp, qtbot, (), "")
    assert _geometry(insp) == clean


def test_the_note_uses_no_empty_gap_and_wraps_completely_in_the_scrollable_sidebar(
        qapp, qtbot, tmp_path):
    insp = _shown_inspector(qtbot, tmp_path, (1000, 800))
    note = insp._stale_label
    assert QtWidgets.QLabel.text(note) == ""
    assert note.isHidden()
    assert note.parentWidget().layout().itemAt(0).isEmpty()
    clean = _geometry(insp)
    _flip(insp, qtbot, REMEASURED_COLUMNS, NOTE_PREDATES)
    assert not note.isHidden() and note.wordWrap()
    assert note.text() == f"⚠ {NOTE_PREDATES}"
    assert note.height() >= note.heightForWidth(note.width())
    assert note.maximumHeight() > note.minimumHeight()
    insp._review_sidebar.ensureWidgetVisible(note)
    qtbot.wait(30)
    assert note.visibleRegion().contains(note.rect().center())
    assert _geometry(insp) == clean
    _flip(insp, qtbot, (), "")
    assert note.isHidden()
    assert note.parentWidget().layout().itemAt(0).isEmpty()
    assert _geometry(insp) == clean


# ================================================================================================
# the backstop: a sentence too long for the reserved lines shows its head and says so
# ================================================================================================

def test_a_sentence_longer_than_the_reserved_lines_is_elided_from_the_head_with_the_whole_in_the_tooltip(
        qapp, qtbot):
    from nanoscope.gui.event_inspector import _STALE_NOTE_LINES, _FixedLinesNote
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    lay = QtWidgets.QVBoxLayout(host)
    note = _FixedLinesNote(_STALE_NOTE_LINES)
    lay.addWidget(note)
    host.resize(320, 120)
    host.show()
    qtbot.wait(30)
    reserved = note.height()

    note.setText(f"⚠ {NOTE_PREDATES}")
    qtbot.wait(30)
    shown = QtWidgets.QLabel.text(note)
    assert note.text() == f"⚠ {NOTE_PREDATES}", "text() is the whole sentence"
    assert shown.endswith(" …") and len(shown) < len(note.text()), shown
    assert shown.startswith("⚠ These events were measured"), shown
    assert note.toolTip() == f"⚠ {NOTE_PREDATES}"
    assert note.height() == reserved

    # Widened, the same sentence fits and is shown whole; the tooltip stands down.
    host.resize(1400, 120)
    qtbot.wait(30)
    assert QtWidgets.QLabel.text(note) == f"⚠ {NOTE_PREDATES}"
    assert note.toolTip() == ""
    assert note.height() == reserved

    # Cleared: nothing shown, nothing hinted, the same height.
    note.setText("")
    qtbot.wait(10)
    assert QtWidgets.QLabel.text(note) == "" and note.toolTip() == ""
    assert note.height() == reserved


def test_the_note_never_sets_the_windows_minimum_width(qapp, qtbot, tmp_path):
    """A word-wrapping label's natural width is its whole sentence; the box must take the
    window's width, not dictate it (the curation bar was measured against a 1512 px
    display and the window must keep fitting under 1100 px)."""
    insp = _shown_inspector(qtbot, tmp_path, (1000, 800))
    before = insp.minimumSizeHint().width()
    _flip(insp, qtbot, REMEASURED_COLUMNS, NOTE_PREDATES)
    assert insp.minimumSizeHint().width() == before <= 1100
