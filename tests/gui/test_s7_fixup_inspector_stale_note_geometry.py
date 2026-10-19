"""Sweep-7 fix-up B3 / CW-2 -- the staleness sentence must not move the Event Inspector.

The S7-POP1 fix spliced the banner's sentence into the mono readout as up to six wrapped
lines. The readout's group box shares one QVBoxLayout column with the waveform plot (which
carries the stretch), so every line the readout gained was height the plot lost: one
Preprocess knob on the MAIN window -- or the "Re-measure depths" that clears the badge --
resized the waveform, its axes, the region and the level lines under the pointer.

Measured before this fix, on the owner's -1000 mV recording: at 1000x800 the plot went
531 -> 447 -> 531 px and the readout 140 -> 224 px on one knob and back; at the window's
default size the top-level window itself grew 640 -> 653 px. The subpopulation report's
S7-POP4 note was built as an always-present fixed-height label precisely so that nothing
moves; this window now follows it (``_FixedLinesNote``, ``_STALE_NOTE_LINES``).
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
    """Everything a user looking at the waveform would see move: the plot widget's place
    and size in the window, the viewbox's rectangle, the readout box's place and size,
    the window's own size and the width it insists on."""
    plot = insp._plot_widget
    box = insp._readout.parentWidget()
    vb = insp._plot.getViewBox().geometry()
    return {
        "plot": (plot.mapTo(insp, QtCore.QPoint(0, 0)).toTuple(), plot.size().toTuple()),
        "viewbox": (round(vb.x()), round(vb.y()), round(vb.width()), round(vb.height())),
        "readout_box": (box.mapTo(insp, QtCore.QPoint(0, 0)).toTuple(), box.size().toTuple()),
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
def test_the_staleness_flip_leaves_the_plot_the_readout_box_and_the_window_where_they_were(
        qapp, qtbot, tmp_path, note):
    insp = _shown_inspector(qtbot, tmp_path, (1000, 800))
    clean = _geometry(insp)
    clean_lines = insp._readout.text().count("\n") + 1
    assert clean["window"] == (1000, 800)

    _flip(insp, qtbot, REMEASURED_COLUMNS, note)
    stale = _geometry(insp)
    assert stale == clean, f"the badge moved the window:\n  {clean}\n  {stale}"
    assert clean_lines == 10, clean_lines
    # ...while the disclosure is fully there: the sentence in the note, and the per-line
    # marks in the readout. Updated (change C): the note is a one-line chip, so a sentence
    # longer than the line shows its head and carries the whole in its tooltip.
    assert insp.stale_note() == note
    shown = QtWidgets.QLabel.text(insp._stale_label)
    if shown == f"⚠ {note}":
        assert insp._stale_label.toolTip() == ""
    else:
        assert shown.endswith(" …") and f"⚠ {note}".startswith(shown[:-2]), shown
        assert insp._stale_label.toolTip() == f"⚠ {note}"
    assert insp._details_btn.text().startswith("⚠ ") and note in insp._details_btn.toolTip()
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
    640 -> 653 px when the sentence arrived -- the readout's sizeHint pushed the frame.
    (Updated, change C: the default size is now 1100x860; the invariant is the same.)"""
    insp = _shown_inspector(qtbot, tmp_path, None)
    clean = _geometry(insp)
    _flip(insp, qtbot, REMEASURED_COLUMNS, NOTE_BASELINE)
    assert _geometry(insp) == clean
    _flip(insp, qtbot, (), "")
    assert _geometry(insp) == clean


def test_the_note_reserves_its_lines_while_clean_and_the_flip_does_not_change_its_height(
        qapp, qtbot, tmp_path):
    from nanoscope.gui.event_inspector import _STALE_NOTE_LINES
    insp = _shown_inspector(qtbot, tmp_path, (1000, 800))
    note = insp._stale_label
    fm = note.fontMetrics()
    assert QtWidgets.QLabel.text(note) == ""
    reserved = note.height()
    assert reserved >= fm.lineSpacing() * _STALE_NOTE_LINES, (reserved, fm.lineSpacing())
    assert note.isVisible() and note.minimumHeight() == note.maximumHeight() == reserved
    _flip(insp, qtbot, REMEASURED_COLUMNS, NOTE_PREDATES)
    assert note.height() == reserved
    _flip(insp, qtbot, (), "")
    assert note.height() == reserved


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
    # (Updated, change C: one reserved line, so "wide enough" is the sentence's own width.)
    host.resize(note.fontMetrics().horizontalAdvance(f"⚠ {NOTE_PREDATES}") + 40, 120)
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
