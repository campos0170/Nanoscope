"""EP2-01 / EXP2-02 — A NO-WRITE REFUSAL IS DISCLOSED ON ALL THREE SURFACES, NOT ONE.

The shared refusal ledger (BL-1/BL-2) made the three "kept rejected" notes read one
record — but only a refusal made AT THE TABLE had a repaint hop
(``acceptanceRefused`` → ``_on_table_acceptance_refused``). A refusal recorded by a
no-write gesture in the gallery or the inspector — the A key / Accepted checkbox on a
row the detector flagged unmeasurable, or "Accept all" pressed straight after detection
when every measurable row is already accepted — updated the shared ledger and rebuilt
only the gesturing window's own note. Measured: the gallery said "3 tiles kept
rejected" while the events table's label and an ALREADY-OPEN inspector said nothing
about the same frame; worse, a table label already showing "1 event kept rejected"
kept displaying 1 while the shared record held 2. The hidden note then popped up on
any unrelated repaint, attributed by position to whatever the user did last.

THE FIX THESE TESTS PIN. Both windows now declare ``acceptanceRefused`` — emitted
exactly when an accept recorded refusals on the ledger and WROTE nothing (the written
paths already repaint every surface through their data signals) — and MainWindow wires
it to a repaint-only hop (``_on_gallery_acceptance_refused`` /
``_on_inspector_acceptance_refused``) that rebuilds the OTHER two surfaces' notes,
mirroring the table's own hop. The dirty flag and the undo recorder still never hear
about a no-write: the new signal is not ``curationBatchChanged``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

import nanoscope.gui.event_gallery as gal_mod
from nanoscope.analysis.events import mark_reviewed
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
#: The BL layout: a clipped opening + two over-dwell events (the detector's three
#: refusals) + five clean events.
_LONG = [(10_000, 20_000), (30_000, 40_000)]
_CLEAN = [(50_000, 50_300), (60_000, 60_250), (70_000, 70_300),
          (90_000, 90_250), (110_000, 110_300)]
_MAX_DWELL_S = 5e-3


def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0                        # the recording OPENS mid-event -> clipped row 0
    for s, e in _LONG + _CLEAN:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_s2ep.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _drain(qtbot, ms: int = 120) -> None:
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _doc(tmp_path)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    panel._detection.max_dwell_s.setValue(_MAX_DWELL_S)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    _drain(qtbot)
    df = panel.current_detection().payload.df
    unmeasurable = [i for i in range(len(df))
                    if not gal_mod._derived_accepted(df["flag"].iloc[i])]
    assert len(unmeasurable) == 3, (
        "the fixture needs three of the detector's own rejections; got "
        f"{df['flag'].tolist()}")
    try:
        yield win, df, unmeasurable
    finally:
        win._analysis_controller.abort()


def _gallery(win, qtbot):
    win._act_event_gallery.trigger()
    _drain(qtbot)
    return win._event_gallery


def _inspector(win, qtbot):
    win._act_event_inspector.trigger()
    _drain(qtbot)
    return win._event_inspector


def _table(win):
    return win._analysis_panel._table


# ================================================================================================
# EXP2-02 — the most ordinary accept-all there is: confirming a fresh detection
# ================================================================================================
def test_no_change_gallery_accept_all_is_one_voice_on_all_three_surfaces(wired, qtbot):
    """Right after detection every measurable row is already accepted, so the very first
    "Accept all shown" click writes NOTHING — and used to disclose its three refusals
    on the gallery's status line only, with the table label and an ALREADY-OPEN
    inspector silent about the identical frame."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)          # open BEFORE the gesture — the measured hole
    assert df["accepted"].to_numpy(dtype=bool).sum() == len(df) - 3, \
        "the fixture must start with every measurable row accepted (a no-change accept)"

    gal.accept_all_btn.click()
    _drain(qtbot)

    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()
    assert "3 events kept rejected" in _table(win).refusal_label.text(), (
        "the events table stayed silent about a refusal the gallery disclosed:\n"
        + repr(_table(win).refusal_label.text()))
    assert "3 events kept rejected" in insp._batch_note.text(), (
        "an open inspector stayed silent about a refusal the gallery disclosed:\n"
        + repr(insp._batch_note.text()))
    assert not df["accepted"].to_numpy(dtype=bool)[unmeasurable].any(), \
        "sanity: the refused rows must still be rejected"


# ================================================================================================
# EP2-01 (a) — the inspector's per-event refusal, by the real A-key shortcut
# ================================================================================================
def test_inspector_akey_refusal_reaches_table_and_gallery(wired, qtbot):
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)              # opened LAST, so it holds activation —
    refused_id = int(df["event_id"].iloc[unmeasurable[0]])
    insp.show_event(refused_id)
    insp.activateWindow()                      # — the A shortcut is a WindowShortcut
    _drain(qtbot)

    QTest.keyClick(insp, QtCore.Qt.Key_A)      # the real shortcut, not a slot call
    _drain(qtbot)

    assert "1 event kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert "1 event kept rejected" in _table(win).refusal_label.text(), (
        "the table label never learned of the inspector's refusal:\n"
        + repr(_table(win).refusal_label.text()))
    assert "1 tile kept rejected" in gal.status.text(), (
        "the gallery status never learned of the inspector's refusal:\n"
        + repr(gal.status.text()))


# ================================================================================================
# EP2-01 (s2b) — a VISIBLE stale count is actively wrong, not merely silent
# ================================================================================================
def test_a_second_refusal_moves_the_visible_note_everywhere_at_once(wired, qtbot):
    """The worst form: the table already shows "1 event kept rejected"; a second
    inspector refusal moved the shared record to 2 while the table's visible note kept
    displaying 1 — two surfaces stating different counts about one record at the same
    moment."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)

    insp.show_event(int(df["event_id"].iloc[unmeasurable[0]]))
    _drain(qtbot)
    insp._accept_check.setChecked(True)        # the Accepted checkbox — refused, no write
    _drain(qtbot)
    assert "1 event kept rejected" in _table(win).refusal_label.text(), \
        _table(win).refusal_label.text()

    insp.show_event(int(df["event_id"].iloc[unmeasurable[1]]))
    _drain(qtbot)
    insp._accept_check.setChecked(True)
    _drain(qtbot)

    assert "2 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert "2 events kept rejected" in _table(win).refusal_label.text(), (
        "the table's visible note kept displaying the OLD count while the shared "
        "record held 2:\n" + repr(_table(win).refusal_label.text()))


# ================================================================================================
# the no-write signal must not masquerade as a data change
# ================================================================================================
def test_a_no_write_refusal_reaches_neither_the_dirty_flag_nor_the_undo_recorder(
        wired, qtbot):
    """The reason the windows stayed silent in the first place was the right one: a
    signal with no change behind it reaches the dirty flag and the undo recorder. The
    fix is a DEDICATED signal, so the repaint must not have re-opened that hole."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    # An explicit Accept now records human review even when every acceptance boolean
    # already has its requested value; that is persistent labeling work and correctly
    # creates Undo. Pre-review this scope so the gesture below is genuinely write-free and
    # this older refusal-signal contract still tests what its name says it tests.
    mark_reviewed(df, rows=np.arange(len(df), dtype=np.int64))
    dirty_before = win._dirty
    undo_enabled_before = win._act_undo.isEnabled()

    gal.accept_all_btn.click()                 # writes nothing; records 3 refusals
    _drain(qtbot)

    assert "3 tiles kept rejected" in gal.status.text()
    assert win._dirty == dirty_before, \
        "a gesture that wrote nothing marked the project dirty"
    assert win._act_undo.isEnabled() == undo_enabled_before, \
        "a gesture that wrote nothing landed on the undo stack"
