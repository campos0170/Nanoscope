"""BL-2 — THE REFUSAL RECORD RIDES THE DETECTION'S LINEAGE, NOT ONE WINDOW'S FRAME SLOT.

tests/test_fup_refusal_note_wired.py made the refusal note STATE rebuilt on every
repaint, and tests/test_rp_refusal_clear_wired.py made it survive every second gesture.
Three paths still silenced it while the rows stayed refused, because the record lived in
each window and died whenever that window was handed a different ``df`` object:

* **Edit ▸ Undo** of the bulk accept — the history installs a deep-copied snapshot, a new
  object, so the windows dropped the record while the restored frame still held every
  refused row;
* **a merge** — ``add_result(replace(res, payload=…))`` installs a rewritten frame, same
  detection, new object; the refresh dropped the note;
* **a second window** — a gallery or inspector opened AFTER the gesture got a fresh
  private ledger and never showed the standing disclosure at all.

THE FIX THESE TESTS PIN
-----------------------
MainWindow keeps ONE ``_RefusalLedger`` per detection LINEAGE (``_event_refusals``,
keyed on doc/channel/module and the detector's own ``detected_at`` run stamp — copied
intact by the undo snapshot and by a merge, NEW on every re-run) and hands it to the
gallery, the inspector AND the events table with every re-feed
(``set_events(…, refusals=…)`` / the table's resolver). Every surface rebuilds its note
from that one record, so:

1. the note survives Edit ▸ Undo / Redo of the bulk accept;
2. the note survives a merge (the frame is a new object; the lineage is not);
3. a window opened after the gesture shows the standing note — either direction — and
   the table's label (BL-1's disclosure) reads from the same record;
4. a NEW detection still takes every surface's note down at once;
5. the note dies exactly when the refused set truly empties, one row at a time.

Two assertions in tests/test_rk_refusal_counter_wired.py pinned the per-window silence
("the inspector never made a gesture to disclose") and are edited there, with the
justification written beside them.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

import nanoscope.gui.event_gallery as gal_mod
import nanoscope.gui.event_inspector as insp_mod
import nanoscope.gui.main_window as mw_mod
import nanoscope.gui.panels as panels_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
#: Two mergeable pairs (5 ms gaps: separate at detection, merged by a 6 ms 'Merge with
#: next'), placed so a merge can be made either side of a refused row: PAIR_EARLY sits
#: BEFORE the last long event, so merging it renumbers a refused row's ``event_id`` —
#: the drift the owner's remap must survive — while PAIR_LATE sits after everything.
_LONG_EARLY = [(10_000, 20_000)]
_PAIR_EARLY = [(50_000, 50_250), (50_750, 51_000)]
_CLEAN = [(70_000, 70_300), (90_000, 90_250), (110_000, 110_300)]
_LONG_LATE = [(130_000, 140_000)]
_PAIR_LATE = [(160_000, 160_250), (160_750, 161_000)]
_ALL = _LONG_EARLY + _PAIR_EARLY + _CLEAN + _LONG_LATE + _PAIR_LATE
_MAX_DWELL_S = 5e-3


# ================================================================================================
# builders — a real recording, a real MainWindow, the real detector
# ================================================================================================
def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0                        # the recording OPENS mid-event -> clipped row 0
    for s, e in _ALL:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_bl2.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def settings_ns(qapp):
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("NanoScopeBL2NoteLineage")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app)


def _drain(qtbot, ms: int = 120) -> None:
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


@pytest.fixture
def wired(qapp, qtbot, tmp_path, settings_ns):
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
    assert len(df) == len(_ALL) + 1, f"unexpected detection: {len(df)} events"
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


def _really_refused(df) -> list[int]:
    return [i for i in range(len(df))
            if not bool(df["accepted"].iloc[i])
            and not gal_mod._derived_accepted(df["flag"].iloc[i])]


def _current_df(win):
    return win._analysis_panel.current_detection().payload.df


# ================================================================================================
# 0. provenance — which tree, and the lineage token really exists to ride on
# ================================================================================================
def test_the_package_under_test_is_the_one_beside_this_file():
    here = Path(__file__).resolve().parents[2]
    for mod in (nanoscope, gal_mod, insp_mod, mw_mod, panels_mod):
        assert Path(mod.__file__).resolve().is_relative_to(here), \
            f"{mod.__name__} resolved from {mod.__file__}, not from {here}"
    # The doors this file drives, with the real signatures.
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        assert "refusals" in inspect.signature(cls.set_events).parameters, \
            f"{cls.__name__}.set_events takes no shared ledger"
    assert callable(getattr(mw_mod.MainWindow, "_refusal_ledger_for", None))
    assert callable(getattr(panels_mod.EventTable, "set_refusals_source", None))


def test_the_detection_carries_the_run_stamp_the_lineage_rides_on(wired, qtbot):
    """No ``detected_at``, no way to tell an undo's restored snapshot from a new
    detection — the fallback is then frame-object identity, i.e. the old behaviour. The
    Schmitt path this app detects with must keep stamping."""
    win, _df, _u = wired
    prov = win._analysis_panel.current_detection().payload.provenance or {}
    assert prov.get("detected_at"), "the detector no longer stamps detected_at"


# ================================================================================================
# 1. Edit ▸ Undo — the note survives the restored snapshot (a different df object)
# ================================================================================================
def test_the_note_survives_undo_and_redo_of_the_bulk_accept(wired, qtbot):
    """Measured defect (a): reject-all, accept-all ("3 tiles kept rejected"), Edit ▸ Undo
    — the note vanished though the restored frame still held all three refused rows."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    win._act_undo.trigger()
    _drain(qtbot)
    restored = _current_df(win)
    assert restored is not df, \
        "undo did not install a snapshot; this test is no longer driving the eraser"
    assert not restored["accepted"].to_numpy(dtype=bool).any(), \
        "the undo did not restore the all-rejected frame"
    assert len(_really_refused(restored)) == 3
    assert "3 tiles kept rejected" in gal.status.text(), (
        "Edit ▸ Undo silenced a disclosure about rows that are still refused:\n"
        + gal.status.text())
    assert "3 events kept rejected" in _table(win).refusal_label.text(), \
        _table(win).refusal_label.text()

    win._act_redo.trigger()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()


# ================================================================================================
# 2. a merge — the note survives the rewritten frame (same detection, new df object)
# ================================================================================================
def test_the_note_survives_a_merge(wired, qtbot):
    """Measured defect (b): a merge installs a NEW EventSet through ``add_result``; the
    refresh handed the windows a different frame and the record died with it, while the
    merged frame still held every refused row (the merge touches only its own run)."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    order = np.argsort(df["start_idx"].to_numpy())
    pair_first = int(df["event_id"].to_numpy()[order][-2])   # _PAIR_LATE sits last
    report = win._analysis_panel.merge_event_with_next(pair_first, 0.006)
    _drain(qtbot)
    assert report.status == "ok", f"the merge did not run: {report}"

    merged = _current_df(win)
    assert merged is not df and len(merged) == len(df) - 1, \
        "the merge did not install a rewritten frame; the eraser was never driven"
    assert len(_really_refused(merged)) >= 3, "the merge touched the refused rows"
    assert "3 tiles kept rejected" in gal.status.text(), (
        "a merge silenced a disclosure about rows it never touched:\n"
        + gal.status.text())
    assert "3 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert "3 events kept rejected" in _table(win).refusal_label.text()


def test_the_count_survives_a_merge_that_renumbers_a_refused_rows_id(wired, qtbot):
    """The half of the merge story the last test cannot see. ``merge_by_gap`` renumbers
    ``event_id`` with ``np.arange`` over the survivors, so a merge BEFORE a refused row
    shifts that row's id — and a record that survives the merge but keeps the OLD ids
    counts somebody else's event (measured on the owner's -600 mV recording: three rows
    refused, one early merge, and the note said 2). The record is re-pointed by
    ``start_idx``, the identity a merge leaves a refused row."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    # _PAIR_EARLY's head: the event starting at ~50_000 — BEFORE the late long event.
    starts = df["start_idx"].to_numpy()
    pair_first = int(df["event_id"].to_numpy()[
        int(np.argmin(np.abs(starts - _PAIR_EARLY[0][0])))])
    report = win._analysis_panel.merge_event_with_next(pair_first, 0.006)
    _drain(qtbot)
    assert report.status == "ok", f"the merge did not run: {report}"

    merged = _current_df(win)
    assert len(merged) == len(df) - 1
    still = _really_refused(merged)
    # FOUR rows now stand refused: the original three, plus the merged pair itself —
    # its 10 ms span is over the dwell limit, so the rebuild rejected and flagged it.
    # Only the three the accept was REFUSED OVER may be counted: the merged row was
    # never asked to be accepted, so a note reading 4 would invent a refusal.
    assert len(still) == 4, f"unexpected refused set after the merge: {still}"

    # The test is only about renumbering if renumbering happened: the late long event
    # (start ~130 000) must carry a DIFFERENT id than it did before the merge.
    def _id_at(frame, start):
        s = frame["start_idx"].to_numpy()
        return int(frame["event_id"].to_numpy()[int(np.argmin(np.abs(s - start)))])
    late_start = _LONG_LATE[0][0]
    assert _id_at(merged, late_start) != _id_at(df, late_start), \
        "the merge did not renumber the refused row; this test is not testing the remap"
    assert "3 tiles kept rejected" in gal.status.text(), (
        "the record kept the pre-merge ids and counted somebody else's events:\n"
        + gal.status.text())
    assert "3 events kept rejected" in _table(win).refusal_label.text(), \
        _table(win).refusal_label.text()


# ================================================================================================
# 3. a second window — opened AFTER the gesture, it shows the standing note
# ================================================================================================
def test_an_inspector_opened_after_the_gallery_gesture_shows_the_standing_note(
        wired, qtbot):
    """Measured defect (c), one direction: the disclosure is about the FRAME, and a
    window built after the gesture is showing that same frame."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert win._event_inspector is None, "the inspector existed early; wrong scenario"

    insp = _inspector(win, qtbot)                 # FIRST build, after the gesture
    assert "3 events kept rejected" in insp._batch_note.text(), (
        "a window opened after the gesture never learned of the standing refusals:\n"
        + repr(insp._batch_note.text()))
    assert not insp._batch_note.isHidden()


def test_a_gallery_opened_after_the_inspector_gesture_shows_the_standing_note(
        wired, qtbot):
    """...and the other direction, plus the table's label (BL-1's disclosure) reading
    from the same record."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    insp._reject_all_btn.click()
    _drain(qtbot)
    insp._accept_all_btn.click()
    _drain(qtbot)
    assert "3 events kept rejected" in insp._batch_note.text()
    assert win._event_gallery is None, "the gallery existed early; wrong scenario"

    gal = _gallery(win, qtbot)                    # FIRST build, after the gesture
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()
    assert "3 events kept rejected" in _table(win).refusal_label.text(), \
        _table(win).refusal_label.text()


# ================================================================================================
# 4. the one thing that still clears it everywhere at once — a NEW detection
# ================================================================================================
def test_a_new_detection_takes_the_note_down_on_every_surface(wired, qtbot):
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in gal.status.text()

    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._analysis_panel.trigger_detection()
    _drain(qtbot)
    assert "kept rejected" not in gal.status.text(), gal.status.text()
    assert insp.refusal_note() == "", insp.refusal_note()
    assert _table(win).refusal_note() == "", _table(win).refusal_note()
    assert _table(win).refusal_label.isHidden()


# ================================================================================================
# 5. ...and otherwise it dies only when the refused set truly empties
# ================================================================================================
def test_the_note_dies_exactly_when_the_refused_set_empties(wired, qtbot):
    """The invariant stated end to end: repair each refused row's FLAG (what a mask edit
    that re-characterizes the span does), announce the edit, and watch the count step
    down on every surface — dying only with the last row."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    flag_col = df.columns.get_loc("flag")
    for k, row in enumerate(unmeasurable, start=1):
        df.iat[row, flag_col] = ""                # the detector's objection, repaired
        win._analysis_panel.resync_curation()
        win._refresh_surface_windows()
        _drain(qtbot)
        left = 3 - k
        if left:
            assert f"{left} tile" in gal.status.text() and \
                "kept rejected" in gal.status.text(), gal.status.text()
            assert f"{left} event" in _table(win).refusal_note(), \
                _table(win).refusal_note()
        else:
            assert "kept rejected" not in gal.status.text(), (
                "the note outlived the last row it was about:\n" + gal.status.text())
            assert _table(win).refusal_note() == ""
