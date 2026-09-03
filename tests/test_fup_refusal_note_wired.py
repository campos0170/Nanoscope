"""M6b — A REFUSAL THE USER CANNOT SEE IS NOT A REFUSAL.

M6 stopped "Accept all" overriding the detector. In the WIRED application it then said so
nowhere. Measured by me on the owner's ``…_0006_Process_-600 mV.abf`` (ch 2, 317 events,
one ``clipped_by_boundary``) through a real ``MainWindow``, Analysis ▸ Event gallery,
one click on the real button, event loop drained::

    button                                Accept all 317
    status, immediately after the click   …   316 of 317 accepted   ·   drawn from: …
    status, after draining                …   316 of 317 accepted   ·   drawn from: …
    Event inspector, "Accept all"         (no note of any kind exists on that window)

— no "kept rejected" clause on either surface, on all three gallery paths (the no-change
path, the 316-rows-changed path, and the repair path). So an honest refusal and a broken
button read identically.

WHY IT WAS INVISIBLE, AND WHY THE EXISTING TEST COULD NOT SEE IT
---------------------------------------------------------------
``_note_refused`` APPENDED to ``self.status.text()`` after ``_sync_controls`` had rebuilt
it. Two things then erase it, and both are ordinary:

* ``curationBatchChanged`` is in ``main_window.SURFACE_MUTATION_SIGNALS['event_gallery']``,
  so it reaches ``_refresh_surface_windows`` -> ``_refresh_event_gallery`` -> ``refresh()``
  -> ``_sync_controls()``, which rebuilds the line from scratch;
* even the path that emits NOTHING loses it, because the click itself provokes a relayout
  (the page size went 12 -> 40 tiles in the measurement above) and that too calls
  ``_sync_controls``.

``test_gallery_says_on_the_status_line_how_many_tiles_it_kept_rejected`` builds
``EventGalleryWindow(session)`` with no MainWindow, so neither eraser is connected and the
test cannot fail on this. Every test here drives a real wired ``MainWindow`` and asserts
the string a person reads.

THE FIX, AND THE HUNKS THESE TESTS PIN
--------------------------------------
1. the gallery keeps ``_refused`` as STATE and ``_sync_controls`` rebuilds the note FROM
   it — so every rebuild puts it back instead of wiping it;
2. the inspector, which had no disclosure at all, grows one (``_batch_note``), rendered
   from the same state by its own ``_sync_controls`` — and it is its own widget, not part
   of the per-event readout, which is rewritten on every "Next";
3. "kept rejected" is only said of a row that was ALREADY rejected. A row sitting
   ``accepted=True`` beside ``clipped_by_boundary`` (the forbidden state, reachable from
   any table written before M6) is *corrected back to rejected* — the accepted count goes
   DOWN on a click that said "accept", and the old wording denied that;
4. + 5. both windows used to FAIL OPEN on a frame with no ``flag`` column
   (``np.ones(n, bool)`` / ``flag_col is None`` -> broadcast ``requested``), i.e. the guard
   switched itself off and restored the exact leak M6 closed. Both now refuse the accept
   outright, write nothing, and say why. Rejection is still never blocked.

Collateral this file also holds down: ``Edit ▸ Undo`` must keep restoring the frame
exactly (``atol=0``, NaN-equal).
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
import pandas as pd
import pytest
from PySide6 import QtCore

import nanoscope.gui.event_gallery as gal_mod
import nanoscope.gui.event_inspector as insp_mod
import nanoscope.gui.main_window as mw_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
#: Four ordinary events, plus one 80 ms event over the 5 ms limit the form is set to.
_CLEAN = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400), (70_000, 70_200)]
_LONG = (90_000, 170_000)
_MAX_DWELL_S = 5e-3


# ================================================================================================
# builders — a real recording, opened in a real MainWindow, detected by the real detector
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
    meta = RecordingMeta(path=tmp_path / "pore_m6b.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def settings_ns(qapp):
    """A PRIVATE QSettings namespace: this file opens two standalone windows and must not
    read (or leave behind) the developer's own saved layout."""
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("NanoScopeM6bRefusalNote")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app)


def _drain(qtbot, ms: int = 120) -> None:
    """Let every queued connection run. The erasers are queued work, so a test that does
    not drain cannot observe them — which is the whole point of this file."""
    qtbot.wait(ms)
    qapp = QtCore.QCoreApplication.instance()
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


@pytest.fixture
def wired(qapp, qtbot, tmp_path, settings_ns):
    """A real MainWindow with a real detection: the ONLY configuration in which the
    erasing signal chain is connected at all."""
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
    # The fixture's whole point: the DETECTOR rejected two rows on its own grounds.
    unmeasurable = [i for i in range(len(df))
                    if not gal_mod._derived_accepted(df["flag"].iloc[i])]
    assert len(unmeasurable) == 3, (
        f"the fixture must give the detector three of its own rejections; got "
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


# ================================================================================================
# 0. provenance — which package is under test, and are these the real methods
# ================================================================================================

def test_the_package_under_test_is_the_one_beside_this_file():
    """``nanoscope`` is installed editable on this machine AND a revert run from inside the
    repo silently tests the live tree (cwd beats PYTHONPATH when cwd holds the package). So
    every module this file asserts about must resolve from the tree this file lives in, and
    the methods must be the real ones with the real signatures — a revert stub with a wrong
    signature has manufactured four fake failures in this project before."""
    here = Path(__file__).resolve().parents[1]
    for mod in (nanoscope, gal_mod, insp_mod, mw_mod):
        assert Path(mod.__file__).resolve().is_relative_to(here), \
            f"{mod.__name__} resolved from {mod.__file__}, not from {here}"

    assert list(inspect.signature(gal_mod.EventGalleryWindow._set_rows_accepted)
                .parameters) == ["self", "rows", "accepted"]
    assert list(inspect.signature(insp_mod.EventInspectorWindow._set_all_accepted)
                .parameters) == ["self", "accepted"]
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        fn = getattr(cls, "refusal_note", None)
        assert fn is not None, f"{cls.__name__} has no refusal_note()"
        assert list(inspect.signature(fn).parameters) == ["self"]
    # ...and MainWindow really is wired to the refresher that erased the note, so the
    # tests below are exercising the eraser and not an empty signal.
    assert "curationBatchChanged" in mw_mod.SURFACE_MUTATION_SIGNALS["event_gallery"]
    assert "curationBatchChanged" in mw_mod.SURFACE_MUTATION_SIGNALS["event_inspector"]


# ================================================================================================
# 1. HUNK 1 — the gallery's note survives the refresh that used to erase it
# ================================================================================================

def test_wired_gallery_accept_all_says_what_it_kept_rejected(wired, qtbot):
    """THE headline, on the no-change path (the state the app is in the moment a detection
    finishes: every measurable row already accepted). Before: the status line read
    "N-2 of N accepted" and nothing else, both immediately and after draining."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)

    before = gal.status.text()
    assert "kept rejected" not in before

    gal.accept_all_btn.click()
    assert "3 tiles kept rejected" in gal.status.text(), \
        f"the note is not even up before the event loop runs:\n{gal.status.text()}"
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), (
        "the note was erased by the refresh — this is the measured defect:\n"
        f"{gal.status.text()}")
    assert "flagged by the detector as not measurable" in gal.status.text()
    assert _accepted(df).count(False) == 3


def test_wired_gallery_note_survives_the_batch_signal_it_emits(wired, qtbot):
    """The path that ACTUALLY emits: reject everything first, so the accept moves rows and
    ``curationBatchChanged`` fires. That signal is the eraser (SURFACE_MUTATION_SIGNALS ->
    _refresh_surface_windows -> _refresh_event_gallery -> refresh -> _sync_controls), so
    this test is worthless unless the signal really fired — which it asserts."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)

    gal.reject_all_btn.click()
    _drain(qtbot)
    assert _accepted(df) == [False] * len(df)

    fired: list[int] = []
    gal.curationBatchChanged.connect(lambda: fired.append(1))
    gal.accept_all_btn.click()
    _drain(qtbot)

    assert fired, "curationBatchChanged did not fire; the eraser was never exercised"
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()
    assert _accepted(df).count(True) == len(df) - 3


def test_the_gallery_note_is_state_the_line_is_rebuilt_from_not_text_written_over_it(
        wired, qtbot):
    """The invariant, not the symptom. ANY independent rebuild of the status line must put
    the note back — the reported one-line fix (move ``_note_refused`` after the emits) does
    not survive this, because a resize, a page turn or a filter change calls
    ``_sync_controls`` with no curation anywhere near it."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in gal.status.text()

    gal._sync_controls()                       # a bare repaint of the line
    assert "3 tiles kept rejected" in gal.status.text(), \
        "the note does not survive a plain rebuild — it is still text written over output"

    gal.refresh()                              # the refresher's own door
    assert "3 tiles kept rejected" in gal.status.text()
    assert gal.status.text().count("kept rejected") == 1, \
        f"the note doubled on rebuild:\n{gal.status.text()}"


def test_the_gallery_note_survives_a_reject_and_dies_with_the_detection(
        wired, qtbot):
    """It describes THE FRAME, not one gesture (M6d). A reject refuses nothing, so it
    retires no refusal — and the identical whole-frame reject arriving from the inspector
    never cleared this note, so the app said two different things about one frame
    depending on which window the click landed in. What DOES take it down is a different
    detection."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in gal.status.text()

    gal.reject_all_btn.click()                 # rejection refuses nothing, and retires none
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in gal.status.text()
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._analysis_panel.trigger_detection()   # a DIFFERENT frame
    _drain(qtbot)
    assert "kept rejected" not in gal.status.text(), (
        "the note outlived the detection it counted:\n" + gal.status.text())


# ================================================================================================
# 2. HUNK 2 — the inspector had no disclosure at all
# ================================================================================================

def test_wired_inspector_accept_all_tells_the_user_what_it_refused(wired, qtbot):
    """Before this hunk the inspector's "Accept all" refused silently: nothing on the
    window said so, and the implementer disclosed it."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)

    assert insp.refusal_note() == ""
    insp._reject_all_btn.click()
    _drain(qtbot)

    fired: list[int] = []
    insp.curationBatchChanged.connect(lambda: fired.append(1))
    insp._accept_all_btn.click()
    _drain(qtbot)

    assert fired, "curationBatchChanged did not fire; the eraser was never exercised"
    assert "3 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert insp._batch_note.isVisible(), "the note is set but not shown"
    assert _accepted(df).count(True) == len(df) - 3


def test_the_inspector_note_survives_pressing_next(wired, qtbot):
    """It is its OWN widget for this reason: the per-event readout is rewritten on every
    navigation, so a disclosure living there dies on the user's next keystroke."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    insp._accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in insp._batch_note.text()

    insp.next_event()
    insp.next_event()
    _drain(qtbot)
    assert "3 events kept rejected" in insp._batch_note.text(), \
        "the note died on 'Next' — it is riding the per-event readout"
    assert "kept rejected" not in insp._readout.text()


def test_the_inspector_note_goes_away_with_the_detection(wired, qtbot):
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    insp._accept_all_btn.click()
    _drain(qtbot)
    assert insp.refusal_note()

    insp.clear()
    _drain(qtbot)
    assert insp.refusal_note() == ""
    assert not insp._batch_note.isVisible()


# ================================================================================================
# 3. HUNK 3 — "kept rejected" is the wrong verb for the repair case
# ================================================================================================

@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_a_row_repaired_out_of_the_forbidden_state_is_not_described_as_kept(
        wired, qtbot, surface):
    """``accepted=True`` beside ``clipped_by_boundary`` is the state every table written
    before M6 can be in. Accepting there does not KEEP it rejected — it takes an acceptance
    AWAY, and the user's accepted count goes DOWN on a click that said "accept". Saying
    "kept rejected" about that is a false statement about a destructive edit."""
    win, df, unmeasurable = wired
    for r in unmeasurable:                      # put the frame in the forbidden state
        df.iat[r, df.columns.get_loc("accepted")] = True
    win._refresh_surface_windows()
    _drain(qtbot)

    if surface == "gallery":
        gal = _gallery(win, qtbot)
        gal.accept_all_btn.click()
        _drain(qtbot)
        text, noun = gal.status.text(), "tiles"
    else:
        insp = _inspector(win, qtbot)
        insp._accept_all_btn.click()
        _drain(qtbot)
        text, noun = insp._batch_note.text(), "events"

    assert f"3 {noun} corrected back to rejected" in text, text
    assert "kept rejected" not in text, \
        "a row taken OUT of the accepted set is described as having been kept out of it"
    assert _accepted(df).count(False) == 3, "the repair itself did not happen"


# ================================================================================================
# 4. HUNKS 4 + 5 — the guard must not switch itself off on a frame with no 'flag' column
# ================================================================================================

def _feed_flagless(win, qtbot, surface, df):
    """Hand the WIRED window a frame with no ``flag`` column, through its own public door.

    Why not mutate the session's frame: ``EventSet.__post_init__`` names ``flag`` a required
    column, so a flag-less frame cannot travel the ``current_detection()`` path at all —
    ``_gate_event_plots`` and the undo snapshotter both raise on it. The window's
    ``set_events`` takes any DataFrame, which is exactly how such a frame reaches it (a
    project file, a hand-built table, a future column drop), so this is the real door and
    the window is still the wired one, with every erasing signal connected.
    """
    flagless = df.drop(columns=["flag"])
    if surface == "gallery":
        w = _gallery(win, qtbot)
        w.set_events(w._doc_id, w._channel, flagless)
    else:
        w = _inspector(win, qtbot)
        w.set_events(w._doc_id, w._channel, flagless)
    _drain(qtbot)
    assert "flag" not in flagless.columns
    return w, flagless


@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_a_frame_with_no_flag_column_refuses_the_accept_instead_of_broadcasting(
        wired, qtbot, surface):
    """FAIL CLOSED. Both windows used to answer "is this row measurable?" with ``True`` for
    every row when the frame carried no ``flag`` column — which is the pre-fix broadcast,
    reinstated by the guard itself. Nothing may be accepted, nothing may be newly rejected
    (that would be a second, destructive answer), and the user must be told why."""
    win, df, unmeasurable = wired
    w, flagless = _feed_flagless(win, qtbot, surface, df)
    before = _accepted(flagless)
    assert before.count(False) == 3, "the pre-state must have rejections to be broadcast over"

    if surface == "gallery":
        w.accept_all_btn.click()
        _drain(qtbot)
        text = w.status.text()
    else:
        w._accept_all_btn.click()
        _drain(qtbot)
        text = w._batch_note.text()

    assert _accepted(flagless) == before, \
        "the guard failed OPEN: it wrote a column it cannot vouch for"
    assert "NOT accepted" in text and "'flag' column" in text, text


@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_rejection_is_still_never_blocked_on_a_frame_with_no_flag_column(
        wired, qtbot, surface):
    """The other half of the rule, and the one a careless fail-closed breaks: a person may
    always reject an event the app was willing to keep."""
    win, df, unmeasurable = wired
    w, flagless = _feed_flagless(win, qtbot, surface, df)

    if surface == "gallery":
        w.reject_all_btn.click()
    else:
        w._reject_all_btn.click()
    _drain(qtbot)
    assert _accepted(flagless) == [False] * len(flagless)


# ================================================================================================
# 4b. the note must come down the moment it stops being true — and not one gesture earlier
# ================================================================================================

@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_the_note_comes_down_when_the_table_accepts_the_row_it_counted(
        wired, qtbot, surface):
    """A note that survives its own refutation is the defect this file fixes, mirrored —
    but it must come down BY THE ONE ROW THAT MOVED, not all at once.

    ``reflect_acceptance`` is DELIBERATELY unguarded on both windows (a mirror that
    disagreed with the frame it mirrors is worse than the leak). The EventTable's checkbox
    used to be the door that put an unmeasurable row into the accepted set; BL-1 closed it
    (the table's per-row write applies the same derived-AND as every other surface — see
    tests/gui/test_bl_table_accept_guard.py), so the refuted state is installed the one
    way it is still reachable in the field: a table written before the guard (an old
    project file) carries ``accepted=True`` beside a rejecting flag, i.e. an out-of-band
    write on the shared frame, announced through the owner's refreshers. The recount must
    still drop exactly the row that moved. [EDITED with BL-1: the previous version drove
    ``set_event_accepted(…, True)`` and asserted the write LANDED — a direct assertion of
    the unguarded per-row accept BL-1 names a defect.]

    M6c: this test used to assert the WHOLE disclosure vanished on the first such tick, and
    that is what it was pinning as correct. On the owner's -600 mV recording the counter
    held four rows; ticking one erased all four and three stayed refused with nothing on
    either surface saying so. The counter is a count of currently-refused rows, so it goes
    down by one. See tests/test_rk_refusal_counter_wired.py."""
    win, df, unmeasurable = wired
    w = _gallery(win, qtbot) if surface == "gallery" else _inspector(win, qtbot)
    (w.accept_all_btn if surface == "gallery" else w._accept_all_btn).click()
    _drain(qtbot)
    assert w.refusal_note(), "nothing to refute"
    noun = "tile" if surface == "gallery" else "event"

    for k, row in enumerate(unmeasurable, start=1):
        df.iat[row, df.columns.get_loc("accepted")] = True   # out-of-band (pre-guard table)
        win._analysis_panel.resync_curation()
        win._refresh_surface_windows()
        _drain(qtbot)
        assert bool(df["accepted"].iloc[row]), "the out-of-band write did not land"
        left = len(unmeasurable) - k
        if left:
            assert w.refusal_note().startswith(
                f"{left} {noun}" + ("" if left == 1 else "s") + " kept rejected"), (
                    "the note came down by more than the one row that was refuted: "
                    + w.refusal_note())
        else:
            assert w.refusal_note() == "", \
                "the note outlived the last row it was about"


@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_an_unrelated_verdict_does_not_take_down_an_accurate_note(wired, qtbot, surface):
    """The other side of the same rule. Rejecting an ORDINARY event changes nothing about
    what the detector could measure, so the disclosure is still true and must stay up — a
    clear-on-any-mirrored-verdict would also let this window's own mixed gesture erase its
    own note through the echo the panel sends straight back."""
    win, df, unmeasurable = wired
    w = _gallery(win, qtbot) if surface == "gallery" else _inspector(win, qtbot)
    (w.accept_all_btn if surface == "gallery" else w._accept_all_btn).click()
    _drain(qtbot)
    note = w.refusal_note()
    assert note

    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    win._analysis_panel.set_event_accepted(int(df["event_id"].iloc[ordinary]), False)
    _drain(qtbot)
    assert w.refusal_note() == note, "an accurate disclosure was taken down by an unrelated edit"


# ================================================================================================
# 5. collateral that must not regress
# ================================================================================================

def test_undo_after_the_bulk_accept_restores_the_frame_exactly(wired, qtbot):
    """Known collateral of M6: Edit ▸ Undo now reads "Undo Curate N events" rather than
    "Undo Accept all events". Fine — what is NOT fine is the frame coming back different.
    ``atol=0``, NaN-equal, all 20 columns."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.reject_all_btn.click()
    _drain(qtbot)
    snapshot = df.copy(deep=True)

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert _accepted(df).count(True) == len(df) - 3

    win._act_undo.trigger()
    _drain(qtbot)
    restored = win._analysis_panel.current_detection().payload.df
    assert list(restored.columns) == list(snapshot.columns)
    for col in snapshot.columns:
        a, b = snapshot[col], restored[col]
        if pd.api.types.is_float_dtype(a):
            assert np.allclose(a.to_numpy(), b.to_numpy(), atol=0, rtol=0, equal_nan=True), col
        else:
            assert [x if pd.notna(x) else "<NA>" for x in a.tolist()] == \
                [x if pd.notna(x) else "<NA>" for x in b.tolist()], col


def test_the_note_never_claims_a_refusal_when_there_was_none(wired, qtbot):
    """The false-positive direction. A gesture that accepts only measurable rows must say
    nothing extra — on either surface."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)

    ok = [r for r in range(len(df)) if r not in unmeasurable]
    gal.set_events(gal._doc_id, gal._channel, df)      # a clean start on the same frame
    _drain(qtbot)
    for r in ok:
        win._analysis_panel.set_event_accepted(int(df["event_id"].iloc[r]), False)
    _drain(qtbot)

    from PySide6.QtCore import QItemSelection, QItemSelectionModel
    sm = gal.view.selectionModel()
    sm.clearSelection()
    for row in range(gal._model.rowCount()):
        idx = gal._model.index(row, 0)
        if gal._model.df_row(idx) in ok:
            sm.select(QItemSelection(idx, idx), QItemSelectionModel.Select)
    gal.accept_selected_btn.click()
    _drain(qtbot)

    assert gal.refusal_note() == "", gal.refusal_note()
    assert "kept rejected" not in gal.status.text()
    assert insp.refusal_note() == ""
