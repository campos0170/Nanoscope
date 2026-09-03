"""M6c — THE REFUSAL COUNTER IS A COUNT OF CURRENTLY-REFUSED ROWS, NOT A LATCH.

MEASURED BY ME, in a real ``MainWindow``, on the owner's
``…_0006_Process_-600 mV.abf`` (ch 2 ``IN15``, 50 kHz, 317 events, ``max_dwell_s`` 26 ms so
the detector refuses three rows: one ``clipped_by_boundary``, two ``dwell>max_dwell_s``),
event loop drained after every gesture. Left column is an out-of-tree revert of this
change, right column is this tree::

                                         REVERTED                 THIS TREE
  the accept button, before any click    Accept all 317           Accept 314 of 317
  status after clicking it               3 tiles kept rejected    3 tiles kept rejected
  …then tick ONE of the three in the
  events table                           (no clause at all)       2 tiles kept rejected
  rows genuinely still refused           2                        2
  inspector, A on an unmeasurable row    accepted := True         accepted stays False,
                                         silently                 "1 event kept rejected"

The middle row is the defect: **two rows were still refused and nothing on either surface
said so.** The staleness guard zeroed the entire counter on the first refutation instead of
recounting, and ``test_the_note_comes_down_when_the_table_accepts_the_row_it_counted`` in
``tests/test_fup_refusal_note_wired.py`` pins that zeroing as correct — so the suite
defended the defect.

THE INVARIANT, AND WHY IT IS NOT AN ORDERING TRICK
--------------------------------------------------
Ordering tricks were the last batch's fix and they produced this defect: the counter was a
snapshot taken at the moment of the gesture, and one privileged call site had to keep it
true forever after. It is now a **recount**: ``_refused`` is a property that asks each
remembered row whether it is STILL rejected and STILL unmeasurable
(``_still_refused``), and every rebuild of the status line runs that recount. Nothing
"decrements". One row leaving the refused set is simply one row that answers ``False``,
and ``reflect_acceptance`` no longer touches the note on either window.

Consequences the tests below pin, which a decrementing latch would fail:

* the count goes back UP when the refuted row is rejected again (a latch is one-way);
* the count is right after ANY rebuild — a page turn, a filter change, ``refresh()``;
* the note disappears only when the LAST refused row stops being refused.

THE HUNKS THIS FILE PINS
------------------------
1. gallery + inspector — ``_refused`` recounted from ``_refused_kept`` /
   ``_refused_repaired`` (row indices) rather than stored as three integers;
2. gallery + inspector — ``reflect_acceptance`` no longer zeroes anything;
3. gallery — the accept button promises what it will do (``Accept 314 of 317``);
4. inspector — the PER-ROW write applies the same AND the bulk write does, so the
   Accepted checkbox / the A key can no longer accept a row the detector could not
   measure, and it says so on the same note widget.

WHAT WAS NOT FIXED HERE, AND SINCE HAS BEEN: the EventTable's own checkbox
(``_EventTableModel.setData`` / ``set_accepted``, ``nanoscope/gui/panels.py``) was still
unguarded when this file was written (``panels.py`` was outside that change's fence, so
the finding was reported rather than patched). BL-1 closed it — the table's per-row write
now applies the same derived-AND and discloses its refusals on the shared ledger; see
tests/gui/test_bl_table_accept_guard.py. The "tick one row" tests below therefore
install the refuted state OUT-OF-BAND (see ``_table_tick``), which is how it remains
reachable in the field: a table written before the guard.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # CPU before any TF op can be reached

import inspect
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

import nanoscope.gui.event_gallery as gal_mod
import nanoscope.gui.event_inspector as insp_mod
import nanoscope.gui.main_window as mw_mod
from nanoscope.analysis.events import AUTOMATIC_REJECTION_FLAGS, mark_reviewed
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
_CLEAN = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400), (70_000, 70_200)]
_LONG = (90_000, 170_000)
_MAX_DWELL_S = 5e-3


# ================================================================================================
# builders — a real recording, a real MainWindow, the real detector
# ================================================================================================

def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0                        # the recording OPENS mid-event -> clipped row 0
    for s, e in _CLEAN:
        x[s:e] -= 30.0
    x[_LONG[0]:_LONG[1]] -= 30.0            # ...and one event over the dwell limit
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_m6c.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def settings_ns(qapp):
    """A PRIVATE QSettings namespace: this file opens standalone windows and must not read
    (or leave behind) the developer's own saved layout."""
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("NanoScopeM6cRefusalCounter")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app)


def _drain(qtbot, ms: int = 120) -> None:
    """Let every queued connection run. The refresher that used to erase the note, and the
    one that now RE-RUNS the recount, are both queued work."""
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


@pytest.fixture
def wired(qapp, qtbot, tmp_path, settings_ns):
    """A real MainWindow with a real detection — the only configuration in which the
    surface refresher, the events table and both windows are connected to each other."""
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
        "this file needs THREE of the detector's own rejections — with one, a latch that "
        f"zeroes on the first refutation is indistinguishable from a recount; got "
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


def _accepted(df) -> list[bool]:
    return [bool(v) for v in df["accepted"].to_numpy(dtype=bool)]


def _really_refused(df) -> list[int]:
    """The rows that are, RIGHT NOW, rejected and unmeasurable — computed from the frame and
    not from either window, so a test cannot pass by agreeing with the thing under test."""
    return [i for i in range(len(df))
            if not bool(df["accepted"].iloc[i])
            and not gal_mod._derived_accepted(df["flag"].iloc[i])]


def _table_tick(win, df, row: int, accepted: bool, qtbot) -> None:
    """Put ONE row's ``accepted`` into the given state OUT-OF-BAND, and announce it the
    way any externally edited frame is announced (the owner's resync + refreshers).

    [EDITED with BL-1] This helper used to drive ``set_event_accepted`` — the events
    table's own per-row write — and the tests asserted the accept LANDED on unmeasurable
    rows, i.e. they exercised exactly the unguarded door BL-1 names a defect and closes
    (the table now applies the same derived-AND as every other surface; see
    tests/gui/test_bl_table_accept_guard.py). The state the recount tests need — an
    unmeasurable row inside the accepted set — is still reachable in the field, from a
    table written before the guard (an old project file), which is what this direct write
    stands for. Nothing here records on the refusal ledger: an out-of-band write is not
    an accept gesture, so it can refute a counted row but never create a refusal."""
    df.iat[row, df.columns.get_loc("accepted")] = bool(accepted)
    win._analysis_panel.resync_curation()
    win._refresh_surface_windows()
    _drain(qtbot)


# ================================================================================================
# 0. provenance — which tree, and are these the real methods
# ================================================================================================

def test_the_package_under_test_is_the_one_beside_this_file():
    """A revert run from inside the repo silently tests the LIVE tree: cwd beats PYTHONPATH
    when cwd holds the package. Every module asserted about below must resolve from the tree
    this file lives in."""
    here = Path(__file__).resolve().parents[1]
    for mod in (nanoscope, gal_mod, insp_mod, mw_mod):
        assert Path(mod.__file__).resolve().is_relative_to(here), \
            f"{mod.__name__} resolved from {mod.__file__}, not from {here}"

    # ...and the methods are the real ones with the real signatures — a revert stub with a
    # wrong signature has manufactured fake failures in this project before.
    assert list(inspect.signature(gal_mod.EventGalleryWindow._set_rows_accepted)
                .parameters) == ["self", "rows", "accepted"]
    assert list(inspect.signature(insp_mod.EventInspectorWindow._set_all_accepted)
                .parameters) == ["self", "accepted"]
    assert list(inspect.signature(insp_mod.EventInspectorWindow._set_df_accepted)
                .parameters) == ["self", "row", "accepted"]
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        assert list(inspect.signature(cls.refusal_note).parameters) == ["self"]
    assert "curationBatchChanged" in mw_mod.SURFACE_MUTATION_SIGNALS["event_gallery"]
    assert "curationBatchChanged" in mw_mod.SURFACE_MUTATION_SIGNALS["event_inspector"]


def test_the_counter_is_recomputed_and_not_stored(wired, qtbot):
    """HUNK 1, structurally: ``_refused`` is derived, so there is no snapshot for a later
    gesture to have to keep true. Asserted as well as the behaviour because the behaviour
    can be faked by decrementing, and a decrement is the same class of bug one step later
    (see ``…goes_back_up…`` below, which is what a decrement fails)."""
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        member = cls.__dict__.get("_refused")
        assert isinstance(member, property), \
            f"{cls.__name__}._refused is stored state, not a recount: {member!r}"
        assert callable(cls.__dict__.get("_still_refused")), \
            f"{cls.__name__} has no per-row 'is it still refused' question"
    # ...and reflect_acceptance no longer decides anything about the note.
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        src = inspect.getsource(cls.reflect_acceptance)
        assert "_refused" not in src.split('"""')[-1], \
            f"{cls.__name__}.reflect_acceptance still writes the counter"


# ================================================================================================
# 1. THE HEADLINE — the disclosure that disappeared while refusals still stood
# ================================================================================================

def test_wired_gallery_still_discloses_the_refusals_that_still_stand(wired, qtbot):
    """THE measured defect. Three rows refused, tick ONE in the events table, and the whole
    clause vanished while two rows were still refused."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    _table_tick(win, df, unmeasurable[0], True, qtbot)
    assert bool(df["accepted"].iloc[unmeasurable[0]]), \
        "the out-of-band write did not land; this test has stopped testing the recount"
    assert len(_really_refused(df)) == 2, "the frame itself must still hold two refusals"
    assert "2 tiles kept rejected: flagged by the detector as not measurable" \
        in gal.status.text(), (
            "the disclosure went down while two rows were still refused:\n"
            + gal.status.text())
    assert "3 tiles" not in gal.status.text(), "the counter is stale, not recounted"

    _table_tick(win, df, unmeasurable[1], True, qtbot)
    assert len(_really_refused(df)) == 1
    assert "1 tile kept rejected" in gal.status.text(), gal.status.text()   # singular

    _table_tick(win, df, unmeasurable[2], True, qtbot)
    assert _really_refused(df) == []
    assert "kept rejected" not in gal.status.text(), (
        "the note outlived the last row it was about:\n" + gal.status.text())


def test_wired_inspector_still_discloses_the_refusals_that_still_stand(wired, qtbot):
    """The same defect on the other surface, on its own note widget."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)

    insp._accept_all_btn.click()
    _drain(qtbot)
    assert "3 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()

    _table_tick(win, df, unmeasurable[0], True, qtbot)
    assert len(_really_refused(df)) == 2
    assert "2 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert insp._batch_note.isVisible(), "the note is set but not shown"

    _table_tick(win, df, unmeasurable[1], True, qtbot)
    assert "1 event kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    _table_tick(win, df, unmeasurable[2], True, qtbot)
    assert insp._batch_note.text() == ""
    assert not insp._batch_note.isVisible()


@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_the_number_on_screen_equals_the_number_of_rows_actually_refused(
        wired, qtbot, surface):
    """The invariant stated as an assertion, at every point on the way down. The count on
    screen is compared against a count computed from the FRAME, so agreeing with the window
    under test cannot make this pass."""
    win, df, unmeasurable = wired
    w = _gallery(win, qtbot) if surface == "gallery" else _inspector(win, qtbot)
    (w.accept_all_btn if surface == "gallery" else w._accept_all_btn).click()
    _drain(qtbot)
    noun = "tile" if surface == "gallery" else "event"

    for k, row in enumerate([None] + list(unmeasurable)):
        if row is not None:
            _table_tick(win, df, row, True, qtbot)
        n = len(_really_refused(df))
        note = w.refusal_note()
        assert n == 3 - k, "the fixture drifted"
        if n == 0:
            assert note == "", note
        else:
            assert note.startswith(f"{n} {noun}" + ("" if n == 1 else "s") +
                                   " kept rejected"), (n, note)


def test_the_count_goes_back_up_when_the_refuted_row_is_rejected_again(wired, qtbot):
    """WHAT A DECREMENT CANNOT DO. A latch that subtracts one on refutation looks identical
    to a recount until the user changes their mind: rejecting the row again puts it back in
    the refused set, and the disclosure must go back to three."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    _table_tick(win, df, unmeasurable[0], True, qtbot)
    assert "2 tiles kept rejected" in gal.status.text()

    _table_tick(win, df, unmeasurable[0], False, qtbot)     # the user changes their mind
    assert len(_really_refused(df)) == 3
    assert "3 tiles kept rejected" in gal.status.text(), (
        "the counter only counts downwards — it is a decrementing latch, not a count:\n"
        + gal.status.text())
    # [EDITED with BL-2] This used to assert the inspector said NOTHING ("the inspector
    # never made a gesture to disclose") — the per-window silence BL-2 names a defect:
    # the refused-row record is a fact about the FRAME, held once and spoken by every
    # surface, so a window that made no gesture still shows the standing disclosure.
    assert "3 events kept rejected" in insp.refusal_note(), insp.refusal_note()


def test_the_recount_survives_every_independent_rebuild_of_the_line(wired, qtbot):
    """Not text written over output, and not right only at the moment of a gesture: a page
    turn, a filter change, a bare repaint and the refresher's own door all rebuild the line,
    and each must print the CURRENT number."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    _table_tick(win, df, unmeasurable[0], True, qtbot)

    for gesture in (gal._sync_controls, gal.refresh):
        gesture()
        assert "2 tiles kept rejected" in gal.status.text(), (gesture, gal.status.text())
        assert gal.status.text().count("kept rejected") == 1, gal.status.text()

    gal.set_filter("Rejected only")
    _drain(qtbot)
    assert "2 tiles kept rejected" in gal.status.text(), gal.status.text()
    gal.set_filter("All")
    _drain(qtbot)
    assert "2 tiles kept rejected" in gal.status.text(), gal.status.text()


def test_an_unrelated_verdict_still_does_not_take_down_an_accurate_note(wired, qtbot):
    """The rule M6b installed, kept: rejecting an ORDINARY event says nothing about what the
    detector could measure, so all three refusals stay disclosed."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    _table_tick(win, df, ordinary, False, qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()


# ================================================================================================
# 2. THE BUTTON — it must promise what it will do
# ================================================================================================

def test_the_accept_button_names_what_it_will_actually_accept(wired, qtbot):
    """HUNK 3. Measured on the owner's -600 mV recording: "Accept all 317", 313 accepted.
    The status line's disclosure only exists AFTER the click; this is the one the user reads
    before it."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    n = len(df)

    assert gal.accept_all_btn.text() == f"Accept {n - 3} of {n}", gal.accept_all_btn.text()
    assert "not measurable" in gal.accept_all_btn.toolTip(), gal.accept_all_btn.toolTip()
    # and the promise is kept, exactly
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert _accepted(df).count(True) == n - 3
    assert gal.accept_all_btn.text() == f"Accept {n - 3} of {n}", \
        "the promise moved after the click that kept it"
    # Rejection is never blocked, so that button was already honest and is left alone.
    assert gal.reject_all_btn.text() == f"Reject all {n}"


def test_the_button_says_all_only_when_it_means_all(wired, qtbot):
    """The false-positive direction: with the unmeasurable rows filtered out of the subset,
    the button really would accept every tile shown and must say so."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()          # measurable rows accepted, the three are not
    _drain(qtbot)
    gal.set_filter("Accepted only")
    _drain(qtbot)
    shown = len(gal._model.df_rows())
    assert shown == len(df) - 3, shown
    assert gal.accept_all_btn.text() == f"Accept all {shown}", gal.accept_all_btn.text()


# ================================================================================================
# 3. THE PER-ROW WRITE — ``accepted`` is an AND everywhere, not only where a bulk gesture
#    wrote it
# ================================================================================================

def test_the_inspector_checkbox_cannot_accept_a_row_the_detector_could_not_measure(
        wired, qtbot):
    """HUNK 4, through the real checkbox. Before it, one click put ``accepted=True`` beside
    ``dwell>max_dwell_s`` — the forbidden state, one row at a time, past the very rule
    "Accept all" had just enforced over the whole frame."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    gal = _gallery(win, qtbot)
    row = unmeasurable[0]
    eid = int(df["event_id"].iloc[row])
    insp.show_event(eid)
    _drain(qtbot)
    assert insp.current_event_id() == eid
    assert insp._accept_check.isChecked() is False

    # Accept is also an explicit review gesture now.  Pre-review this row so the click
    # below remains the genuinely write-free refusal this older signal-contract test is
    # intended to isolate; a fresh row correctly emits a data signal for its new review
    # value even though the derived acceptance guard keeps it rejected.
    mark_reviewed(df, rows=np.array([row], dtype=np.int64))
    seen: list[tuple[int, bool]] = []
    insp.acceptanceChanged.connect(lambda i, a: seen.append((int(i), bool(a))))
    insp._accept_check.click()                    # the user ticks the box
    _drain(qtbot)

    assert bool(df["accepted"].iloc[row]) is False, \
        "one click put a row the detector could not measure into the accepted set"
    assert insp._accept_check.isChecked() is False, \
        "the box is ticked while the row underneath it says False"
    assert "1 event kept rejected: flagged by the detector as not measurable" \
        in insp._batch_note.text(), insp._batch_note.text()
    assert seen == [], (
        "nothing was written, so no DATA signal may be announced: a signal with no change "
        f"behind it reaches the dirty flag and the undo recorder. {seen}")
    # EDITED for EP2-01 (was: assert "kept rejected" NOT in gal.status.text(), "the
    # gallery reported a gesture it did not make"). That assertion pinned the pre-fix
    # silence, and the silence was the defect: the three surfaces read ONE shared
    # refusal ledger ("one voice" — tests/gui/test_bl_table_accept_guard.py), so the
    # gallery's status line is a disclosure of the shared record, not a report of its
    # own gestures — exactly as it already was for a refusal made at the TABLE. The
    # refusal now travels by the windows' dedicated ``acceptanceRefused`` (repaint-only;
    # the ``seen == []`` guard above still proves no data signal fired), and the gallery
    # must say what the record says.
    assert "1 tile kept rejected" in gal.status.text(), (
        "the gallery stayed silent about the shared record (EP2-01):\n"
        + gal.status.text())


def test_the_inspector_a_key_is_the_same_write_and_the_same_guard(wired, qtbot):
    """The keyboard reaches the same choke point — a guard installed in one gesture is a
    guard missing from the others."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    row = unmeasurable[1]
    insp.show_event(int(df["event_id"].iloc[row]))
    _drain(qtbot)
    insp.set_accepted_current(True)               # what the A key calls
    _drain(qtbot)
    assert bool(df["accepted"].iloc[row]) is False
    assert "1 event kept rejected" in insp._batch_note.text()


def test_per_row_rejection_is_still_never_blocked(wired, qtbot):
    """The other half of the rule: a person may always reject an event the app was willing
    to keep, and that says nothing worth disclosing."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    assert bool(df["accepted"].iloc[ordinary]) is True
    insp.show_event(int(df["event_id"].iloc[ordinary]))
    _drain(qtbot)
    insp._accept_check.click()                    # untick == reject
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is False
    assert insp._accept_check.isChecked() is False
    assert insp.refusal_note() == "", insp.refusal_note()


def test_a_hand_rejected_unflagged_row_is_still_re_acceptable_one_row_at_a_time(
        wired, qtbot):
    """MUST NOT MOVE. ``flagged``/``molecule=apo``/``zero_baseline`` reject nothing, and a
    curator's own earlier rejection is the curator's to undo."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    insp.show_event(int(df["event_id"].iloc[ordinary]))
    _drain(qtbot)
    insp.set_accepted_current(False)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is False

    insp.set_accepted_current(True)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is True, \
        "a hand-rejected row with no rejecting flag became un-acceptable"
    assert insp.refusal_note() == "", insp.refusal_note()


def test_the_per_row_write_fails_closed_on_a_frame_with_no_flag_column(wired, qtbot):
    """Same rule as the bulk path: nothing vouches for the row, so the accept is refused,
    the cell is left alone, and the user is told why. Rejection still lands."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    flagless = df.drop(columns=["flag"])
    insp.set_events(insp._doc_id, insp._channel, flagless)
    _drain(qtbot)
    row = unmeasurable[0]
    before = _accepted(flagless)
    insp.show_event(int(flagless["event_id"].iloc[row]))
    _drain(qtbot)

    insp.set_accepted_current(True)
    _drain(qtbot)
    assert _accepted(flagless) == before, "the guard failed OPEN on a frame it cannot read"
    text = insp._batch_note.text()
    assert "NOT accepted" in text and "'flag' column" in text, text

    ordinary = next(r for r in range(len(flagless)) if r not in unmeasurable)
    insp.show_event(int(flagless["event_id"].iloc[ordinary]))
    _drain(qtbot)
    insp.set_accepted_current(False)
    _drain(qtbot)
    assert bool(flagless["accepted"].iloc[ordinary]) is False, "rejection was blocked"


def test_a_row_repaired_out_of_the_forbidden_state_one_row_at_a_time_is_not_kept(
        wired, qtbot):
    """``accepted=True`` beside a rejecting flag is reachable from any table written before
    M6 — and from the events table today. Accepting there TAKES AN ACCEPTANCE AWAY, and the
    per-row path owes the same honest verb the bulk path does."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    row = unmeasurable[0]
    _table_tick(win, df, row, True, qtbot)          # the forbidden state, via the table
    assert bool(df["accepted"].iloc[row]) is True

    insp.show_event(int(df["event_id"].iloc[row]))
    _drain(qtbot)
    seen: list[tuple[int, bool]] = []
    insp.acceptanceChanged.connect(lambda i, a: seen.append((int(i), bool(a))))
    insp.set_accepted_current(True)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[row]) is False, "the repair did not happen"
    assert "1 event corrected back to rejected" in insp._batch_note.text(), \
        insp._batch_note.text()
    assert "kept rejected" not in insp._batch_note.text()
    # Something DID move, so it is announced — and announced as what was WRITTEN, not as
    # what was asked for: the owner mirrors this straight onto the table's checkbox.
    assert seen == [(int(df["event_id"].iloc[row]), False)], seen


# ================================================================================================
# 4. collateral that must not move
# ================================================================================================

def test_the_rejecting_flags_are_still_the_three_the_schema_names():
    assert AUTOMATIC_REJECTION_FLAGS == {
        "clipped_by_boundary", "dwell>max_dwell_s", "non_finite"}


def test_a_new_detection_still_takes_the_note_and_the_counted_rows_with_it(wired, qtbot):
    """The remembered rows are POSITIONS in a frame; a different frame makes them somebody
    else's events. They are dropped with the frame, not re-indexed."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in gal.status.text()

    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._analysis_panel.trigger_detection()
    _drain(qtbot)
    assert gal._refused_kept == [] and gal._refused_repaired == []
    assert "kept rejected" not in gal.status.text(), gal.status.text()
