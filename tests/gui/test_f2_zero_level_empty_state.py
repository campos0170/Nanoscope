"""F2 — A MASK EDIT THAT PRODUCES ZERO SUB-LEVELS, AS THE USER EXPERIENCES IT.

Round 4 gave the mask drag a second half: ``MainWindow._refit_levels_for_event`` re-measures
the edited event over its new span and splices the group back (see
``tests/gui/test_level_fit_reconcile.py``). It also has five ways of ending with NO levels —
a span too short to resolve a boundary, a rejected row, a cold re-fit cache, a fit that
raises, and a fit that comes back empty — and every one of them is correct in the RESULT
STORE and was invisible or wrong in the window the user was looking at.

TWO DEFECTS, BOTH MEASURED ON THE FIXTURE IN THIS FILE BEFORE THE FIX.

**F2-a — the prescribed recovery did not work.** The rejected route and the empty-fit route
returned in silence, so the only sentence left on screen was ``_reconcile_level_fit``'s
generic one::

    Sub-state levels dropped for 1 event(s) whose span changed — re-run Analysis ▸ Fit
    sub-state levels (CUSUM)… to measure the new spans.

Perform that re-run on a rejected event and it comes back with **0 level rows** for it:
``modules.fit_levels`` runs ``accepted_only=True``, so a rejected event has no levels by
construction and no number of re-runs will produce any. The user was handed the one gesture
that cannot help. The gesture that does help is the ACCEPT — measured here: accept + re-run
gives 3 levels back, accept + one more mask edit gives 3 levels back.

The short-span sentence was wrong in the same direction for a different reason: it named no
remedy at all, and the obvious guess ("re-run the fit") is actively harmful, because
``_fit_levels`` applies no minimum and writes the single meaningless "level" spanning the
whole 9-sample event that ``MIN_SAMPLES_FOR_LEVELS`` exists to refuse. The remedy is the
mask, and widening it back re-fits on release (measured: 7 levels).

**F2-b — the Inspector said nothing at all.** Drag event 0 down to 9 samples and its
staircase, its whole CUSUM readout block and every ``_cusum_items`` entry vanished together,
with ``_batch_note`` empty and the always-on ``levels: 1`` line still printing a number. The
window whose mask had just been dragged was the one surface with no explanation on it.

WHAT IS PINNED HERE
  * each route produces its OWN sentence, on the status bar AND in the Inspector;
  * every sentence names a remedy that was performed against the app in this file;
  * the note CLEARS on recovery — widen the mask back and the staircase returns with no
    stale "no levels" sentence over it, which would be its own bug;
  * the C-5 invariant stays green through all of it: no reader ever holds a group that does
    not tile its parent.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.cusum import MIN_SAMPLES_FOR_LEVELS
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
#: Five events, each with two clean sub-states, exactly as in the C-5 file — the same shape
#: the round-4 numbers quoted above were measured on.
_SPANS = [(5_000, 5_600), (15_000, 15_600), (25_000, 25_600),
          (35_000, 35_600), (45_000, 45_600)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 5.0, N)
    for s, e in _SPANS:
        mid = (s + e) // 2
        x[s:mid] -= 300.0
        x[mid:e] -= 500.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, signal, name: str = "f2_levels.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(signal.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def fitted(window, qtbot, tmp_path):
    """A window with one detection, the Inspector open and CUSUM levels fitted."""
    signal = _signal()
    session = window._session
    doc_id = session.add_document(_doc(tmp_path, signal))
    window._create_default_layers(session.document(doc_id))
    window._on_selection(doc_id, 0)

    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4},
        calibration_fingerprint=session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    qtbot.wait(10)
    window._open_event_inspector()
    window._fit_levels()
    qtbot.wait(10)
    assert window._level_fit is not None and not window._level_fit.payload.empty
    return window, doc_id, df


def _drag_mask(window, doc_id, df, row: int, new_end: int) -> None:
    """Resize event ``row``'s mask to end at ``new_end`` as a user drag would."""
    insp = window._event_inspector
    ev_id = int(df["event_id"].iloc[row])
    insp.show_event(ev_id)
    start = int(df.iloc[row]["start_idx"])
    xv = window._session.x_axis(doc_id).values
    insp._region.setRegion((float(xv[start]), float(xv[new_end])))
    insp._region.sigRegionChangeFinished.emit(insp._region)


def _levels_for(fit, ev_id: int) -> pd.DataFrame:
    if fit is None or getattr(fit, "payload", None) is None:
        return pd.DataFrame({"event_id": []})
    return fit.payload[fit.payload["event_id"] == int(ev_id)]


def _status(window) -> str:
    return window.statusBar().currentMessage()


def _assert_no_stale_group(window, df) -> None:
    """The C-5 invariant, restated for this file: every group the export can reach tiles the
    event it names. It is asserted after EVERY route below, because a message about missing
    levels is worthless if the table it describes is meanwhile holding the old ones."""
    for holder in (window._level_fit,
                   window._session.results.get(
                       window._level_fit.doc_id, window._level_fit.channel, "fit.levels")
                   if window._level_fit is not None else None,
                   type("_L", (), {"payload": window._event_inspector._levels})()):
        payload = getattr(holder, "payload", None) if holder is not None else None
        if payload is None or getattr(payload, "empty", True):
            continue
        spans = df.set_index(df["event_id"].astype("int64"))[["start_idx", "end_idx"]]
        for ev_id, grp in payload.groupby("event_id"):
            want = spans.loc[int(ev_id)]
            assert int(grp["start_idx"].min()) == int(want["start_idx"])
            assert int(grp["end_idx"].max()) == int(want["end_idx"])


# ================================================================================================
# F2-a — THE SENTENCE WHOSE REMEDY DID NOT WORK
# ================================================================================================

def test_the_old_prescribed_recovery_provably_does_not_work_for_a_rejected_event(fitted,
                                                                                 qtbot):
    """THE MEASUREMENT F2-a RESTS ON, run as a test rather than quoted as a claim.

    The sentence this route used to leave standing was "re-run Analysis ▸ Fit sub-state
    levels (CUSUM)… to measure the new spans". Perform exactly that on a rejected event:
    the menu action is live, the fit runs, and the event still has zero level rows. The
    remedy was not merely unhelpful, it was structurally incapable of working — the fit is
    ``accepted_only=True``.

    THIS TEST DOES NOT CHECK THE WORDING. It checks the fact the wording has to respect, so
    it keeps meaning something if the sentence is rewritten again.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    df.iat[0, df.columns.get_loc("accepted")] = False
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)
    assert _levels_for(window._level_fit, ev_id).empty

    assert window._act_fit_levels.isEnabled(), "the prescribed gesture is not even available"
    window._fit_levels()                      # ...perform it, exactly as the sentence said
    qtbot.wait(10)
    assert _levels_for(window._level_fit, ev_id).empty, (
        "re-running the fit DID restore a rejected event's levels — if that is now true, "
        "the accepted_only rule changed and the sentence naming the accept must be revisited")


def test_the_remedy_the_rejected_message_now_names_is_one_the_user_can_perform(fitted, qtbot):
    """...and the replacement is checked the same way: perform what it says, get levels.

    Both halves, because the sentence names both depending on whether a fit is still
    standing: accept the event and edit the mask (the splice path), or accept it and re-run
    the fit.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    acc = df.columns.get_loc("accepted")
    df.iat[0, acc] = False
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)
    note = window._event_inspector.level_note()
    assert "Accept it again" in note, note

    df.iat[0, acc] = True                     # THE REMEDY, performed
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2 - 5)
    qtbot.wait(10)
    assert not _levels_for(window._level_fit, ev_id).empty, (
        "the message told the user to accept the event and edit the mask; doing exactly "
        "that produced no levels")
    _assert_no_stale_group(window, df)
    assert window._event_inspector.level_note() == "", (
        "the levels came back and the window is still explaining their absence")


def test_re_running_the_fit_on_a_too_short_span_writes_the_row_the_guard_refuses(fitted,
                                                                                 qtbot):
    """Why the short-span sentence names the MASK and not the menu.

    ``_refit_levels_for_event`` declines below ``MIN_SAMPLES_FOR_LEVELS`` because the fit
    would return one "level" that is just a copy of the event. ``_fit_levels`` applies no
    such minimum, so "re-run the fit" — the obvious guess when a message names no remedy —
    puts that very row into the table and the export. Measured: 1 row spanning the whole
    9-sample event.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)
    assert _levels_for(window._level_fit, ev_id).empty
    # The sentence sends the user to the MASK, and the rest of this test is why.
    assert "Widen the mask" in window._event_inspector.level_note(), \
        window._event_inspector.level_note()

    window._fit_levels()
    qtbot.wait(10)
    degenerate = _levels_for(window._level_fit, ev_id)
    assert len(degenerate) == 1 and int(degenerate["n_levels"].iloc[0]) == 1, (
        "the whole-detection fit no longer writes a single degenerate level for a "
        "sub-minimum span; the short-span sentence may now name the menu after all")
    assert int(degenerate["end_idx"].iloc[0]) - int(degenerate["start_idx"].iloc[0]) \
        == MIN_SAMPLES_FOR_LEVELS - 1, "the 'level' is not simply a copy of the event"


# ================================================================================================
# F2-b — EVERY ROUTE HAS ITS OWN SENTENCE, ON THE SURFACE THE USER IS LOOKING AT
# ================================================================================================

def _drive_short(window, doc_id, df, qtbot):
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)


def _drive_rejected(window, doc_id, df, qtbot):
    df.iat[0, df.columns.get_loc("accepted")] = False
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)


def _drive_cold(window, doc_id, df, qtbot):
    # Precisely the state `_sync_export_action`'s adoption branch leaves after a reopen: the
    # fit is back from the project file, the inputs behind it are not.
    window._level_fit_inputs = None
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)


def _drive_raise(window, doc_id, df, qtbot, monkeypatch):
    import nanoscope.analysis.cusum as cusum

    def boom(*a, **kw):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cusum, "fit_event_levels", boom)
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)


def _drive_empty(window, doc_id, df, qtbot, monkeypatch):
    import nanoscope.analysis.cusum as cusum
    from nanoscope.analysis.cusum import empty_levels

    monkeypatch.setattr(cusum, "fit_event_levels", lambda *a, **k: empty_levels())
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)


#: (route name, driver, a phrase only THAT route's sentence contains). The phrases are the
#: REASON each sentence gives, not decoration: the point of the parametrisation is that five
#: different causes do not collapse into one "the levels are gone" message.
_ROUTES = [
    ("too_short", "below the 10 a level fit needs"),
    ("rejected", "it is rejected, and the fit measures accepted events only"),
    ("cold", "not in memory after reopening a project"),
    ("raised", "synthetic failure"),
    ("empty", "the re-fit resolved no levels over its new span"),
]


def _drive(name, window, doc_id, df, qtbot, monkeypatch):
    if name == "too_short":
        _drive_short(window, doc_id, df, qtbot)
    elif name == "rejected":
        _drive_rejected(window, doc_id, df, qtbot)
    elif name == "cold":
        _drive_cold(window, doc_id, df, qtbot)
    elif name == "raised":
        _drive_raise(window, doc_id, df, qtbot, monkeypatch)
    elif name == "empty":
        _drive_empty(window, doc_id, df, qtbot, monkeypatch)
    else:                                                # pragma: no cover - typo guard
        raise AssertionError(name)


@pytest.mark.parametrize("route,phrase", _ROUTES)
def test_every_zero_level_route_names_its_own_reason_in_the_inspector(
        fitted, qtbot, monkeypatch, route, phrase):
    """THE F2-b HEADLINE. Each route explains itself IN THE WINDOW, not only behind it.

    Before this, three of the five said nothing anywhere near the Inspector and two said
    nothing at all: the staircase went, the readout's CUSUM block went, and the only
    remaining sign was a status-bar line that expired after eight seconds.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    _drive(route, window, doc_id, df, qtbot, monkeypatch)

    assert _levels_for(window._level_fit, ev_id).empty, "this route did not drop the levels"
    insp = window._event_inspector
    note = insp.level_note()
    assert note, f"the {route} route left the Inspector silent about the missing levels"
    assert phrase in note, f"{route}: {note!r}"
    assert insp._level_note.isVisible(), "the note is set but not shown"
    # ...and in the readout too, where the per-level breakdown used to be.
    assert "CUSUM levels:    none" in insp._readout.text(), insp._readout.text()
    assert phrase in insp._readout.text()
    _assert_no_stale_group(window, df)


@pytest.mark.parametrize("route,phrase", _ROUTES)
def test_every_zero_level_route_says_the_same_thing_on_both_surfaces(
        fitted, qtbot, monkeypatch, route, phrase):
    """ONE STRING, two surfaces. Two wordings for one route is how the status bar came to
    prescribe a re-run while the fit's own accepted-only rule made it impossible."""
    window, doc_id, df = fitted
    _drive(route, window, doc_id, df, qtbot, monkeypatch)
    assert _status(window) == window._event_inspector.level_note(), (
        f"{route}: status bar says {_status(window)!r} and the Inspector says "
        f"{window._event_inspector.level_note()!r}")


def test_the_five_routes_do_not_collapse_into_one_message():
    """A shared "the levels are gone" sentence would satisfy every test above one at a time
    and still leave the user unable to tell a rejected row from a cold cache — which is the
    difference between clicking Accept and re-running a menu item. Asserted against the
    builders directly, so it holds without five windows.

    The REMEDIES are compared as well as the sentences: two routes wearing different
    reasons and prescribing the same gesture is the shape F2-a had.
    """
    from nanoscope.gui.event_inspector import (
        level_gap_cold,
        level_gap_empty,
        level_gap_failed,
        level_gap_rejected,
        level_gap_too_short,
    )

    built = [
        level_gap_too_short(0, n_samples=9, min_samples=MIN_SAMPLES_FOR_LEVELS),
        level_gap_rejected(0),
        level_gap_cold(0),
        level_gap_failed(0, detail="synthetic failure"),
        level_gap_empty(0),
    ]
    assert len(set(built)) == len(built), built
    for text in built:
        assert text.startswith("Sub-state levels dropped for event 0:"), text
        assert text.rstrip().endswith("."), f"a sentence without a full stop: {text!r}"
        # Every one of them names a gesture, and it is a gesture that exists in the app.
        assert ("mask" in text or "Accept it again" in text
                or "Re-run Analysis ▸ Fit sub-state levels (CUSUM)…" in text), text


# ================================================================================================
# THE ROUND TRIP — WIDEN THE MASK BACK AND THE STAIRCASE RETURNS WITH NO NOTE OVER IT
# ================================================================================================

def test_widening_the_mask_again_brings_the_levels_back_and_clears_the_message(fitted,
                                                                               qtbot):
    """THE PROPERTY A STALE NOTE WOULD DESTROY, and the reason the note is rendered from
    state rather than latched on the edit that produced it.

    Shrink event 0 below the minimum (message, no levels), then widen it back. Measured: 7
    levels re-fitted on release. A window still saying "its new span is 9 sample(s)" over
    that staircase would be a worse defect than the silence F2-b replaced — it would be
    actively false about the picture directly beneath it.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    insp = window._event_inspector
    start = int(df.iloc[0]["start_idx"])

    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)
    assert insp.level_note(), "no message on the way down"
    assert not insp._cusum_items, "the staircase is still drawn on a 9-sample event"

    _drag_mask(window, doc_id, df, 0, start + 600)
    qtbot.wait(10)

    assert not _levels_for(window._level_fit, ev_id).empty, "the levels did not come back"
    assert insp._levels is not None and not insp._levels[
        insp._levels["event_id"] == ev_id].empty
    assert insp._cusum_items, "the table has the group back but the waveform does not"
    assert insp.level_note() == "", (
        "a 'no levels' message is standing over a restored staircase: " + insp.level_note())
    assert not insp._level_note.isVisible()
    assert "CUSUM levels:    none" not in insp._readout.text()
    assert "  L0:" in insp._readout.text(), "the per-level breakdown did not come back"
    assert ev_id not in insp._level_gaps, "the recorded reason outlived the disappearance"
    _assert_no_stale_group(window, df)


def test_the_note_cannot_be_rendered_over_a_live_group_even_if_one_is_recorded(fitted,
                                                                               qtbot):
    """The guard stated on its own, without the re-fit in the way.

    Record a reason by hand and then hand the window a table that DOES cover the event. The
    note must not appear, because the question it answers ("why is there no group?") has a
    visible answer: there is one. This is what makes every other clearing rule in the
    window belt-and-braces rather than load-bearing.
    """
    window, doc_id, df = fitted
    insp = window._event_inspector
    ev_id = int(df["event_id"].iloc[0])
    insp.note_level_gap(ev_id, "a reason nobody re-checked")
    assert insp.level_note() == "", (
        "a recorded reason was rendered over an event that has its levels")
    assert ev_id not in insp._level_gaps, (
        "and it was not even recorded: a reason the table in hand contradicts is refused "
        "at the door, so the dict cannot hold a sentence that could never be shown")


def test_the_note_follows_the_browsed_event(fitted, qtbot):
    """It is a statement about ONE event, so navigating to a neighbour whose levels are
    intact must take it down — and coming back must bring it back. A note that survived
    Next would explain event 1's healthy staircase with event 0's edit."""
    window, doc_id, df = fitted
    insp = window._event_inspector
    ev0 = int(df["event_id"].iloc[0])
    ev1 = int(df["event_id"].iloc[1])
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)
    assert insp.level_note()

    insp.show_event(ev1)
    qtbot.wait(5)
    assert insp.level_note() == "", "event 0's message is explaining event 1"
    assert insp._cusum_items, "event 1 lost its own staircase"

    insp.show_event(ev0)
    qtbot.wait(5)
    assert insp.level_note(), "the message did not come back with the event it is about"


def test_a_new_detection_takes_the_message_with_it(fitted, qtbot, tmp_path):
    """Event ids mean nothing across detections, so a reason naming one must not outlive
    the frame it was recorded on — the same rule ``set_events`` already applies to the
    level table itself."""
    window, doc_id, df = fitted
    insp = window._event_inspector
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)
    assert insp.level_note()

    fresh, prov = detect_events(_signal(seed=3), FS, baseline=I0, min_dwell_s=1e-4)
    insp.set_events(doc_id, 0, fresh, provenance=dict(prov))
    qtbot.wait(5)
    assert insp.level_note() == "", "a reason recorded against the old frame is still shown"
    assert not insp._level_gaps


# ================================================================================================
# THE SINGLE-EVENT CASE — THE ONE PLACE "WIDEN IT AGAIN" WOULD HAVE BEEN A LIE
# ================================================================================================

def test_when_the_whole_fit_retires_the_message_names_the_menu_instead_of_the_mask(fitted,
                                                                                   qtbot):
    """MEASURED, and the reason the sentences branch on whether a fit is still standing.

    Reject every other event so event 0 carries the only group. Shrinking it takes
    ``reconcile_levels`` to ``kept is None``, i.e. ``_invalidate_level_fit`` — the fit out
    of the window's hands, out of the store and out of the menu. ``_refit_levels_for_event``
    then returns at its first guard, so widening the mask back restores NOTHING: measured,
    the fit is still None afterwards. "Widen the mask and they are re-fitted on release" is
    true of the ordinary case and false of this one, and the message has to know which it
    is in.
    """
    window, doc_id, df = fitted
    insp = window._event_inspector
    for r in range(1, len(df)):
        df.iat[r, df.columns.get_loc("accepted")] = False
    window._fit_levels()
    qtbot.wait(10)
    assert set(window._level_fit.payload["event_id"]) == {int(df["event_id"].iloc[0])}

    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)

    assert window._level_fit is None, "this fixture no longer reaches the whole-fit retire"
    note = insp.level_note()
    assert "below the 10 a level fit needs" in note, note
    assert "re-run Analysis ▸ Fit sub-state levels (CUSUM)…" in note, (
        "with no standing fit a mask edit re-fits nothing, so the message must send the "
        "user to the menu: " + note)

    # ...and the claim is the measured one: widening alone really does not bring them back.
    _drag_mask(window, doc_id, df, 0, start + 600)
    qtbot.wait(10)
    assert window._level_fit is None, (
        "a mask edit now rebuilds a retired fit — if that is deliberate, this message "
        "should go back to naming the mask alone")
    # THE SENTENCE MUST NOT SURVIVE ITS OWN SPAN. It quotes a sample count, the mask has
    # since been dragged back out to 600 samples, and nothing re-fitted — so the reason is
    # retired by its stamp rather than left standing as a false statement about the event
    # currently on screen. (Silence is what is left: from the window alone, "the fit
    # retired a moment ago" and "no fit was ever run" are the same state, and the second is
    # every session that never ran the fit.)
    note = insp.level_note()
    assert "9 sample(s)" not in note, (
        "the message is still quoting the span the event had two drags ago: " + note)
    assert insp.level_note() == "", note


# ================================================================================================
# WHAT THE WINDOW DERIVES ON ITS OWN
# ================================================================================================

def test_an_event_the_fit_never_covered_explains_itself_without_being_told(fitted, qtbot):
    """No mask edit, no announcement: reject an event and re-run the fit, and it simply has
    no group. The window reads that off the frame it already holds rather than waiting to
    be told, because "rejected" and "re-run the fit" is the one pairing that is always
    wrong."""
    window, doc_id, df = fitted
    insp = window._event_inspector
    ev1 = int(df["event_id"].iloc[1])
    df.iat[1, df.columns.get_loc("accepted")] = False
    window._fit_levels()
    qtbot.wait(10)
    assert _levels_for(window._level_fit, ev1).empty
    assert ev1 not in insp._level_gaps, "nothing announced this; it must be derived"

    insp.show_event(ev1)
    qtbot.wait(5)
    note = insp.level_note()
    assert "it is rejected, and the fit measures accepted events only" in note, note


def test_with_no_fit_standing_at_all_the_window_says_nothing(fitted, qtbot):
    """The common case, and it must look exactly as it did before any of this existed: a
    level fit is on demand, so most sessions never run one and a window explaining the
    absence of a fit nobody asked for is noise on every event."""
    window, doc_id, df = fitted
    insp = window._event_inspector
    window._invalidate_level_fit()
    qtbot.wait(5)
    insp.show_event(int(df["event_id"].iloc[0]))
    qtbot.wait(5)
    assert insp._levels is None
    assert insp.level_note() == "", insp.level_note()
    assert "CUSUM levels" not in insp._readout.text()


def test_the_empty_state_does_not_stack_a_second_message_under_its_own(fitted, qtbot):
    """"The recording for this detection was closed." is the whole story; a sentence about
    a level fit that also has nothing to draw is noise about the smaller problem."""
    window, doc_id, df = fitted
    insp = window._event_inspector
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)
    assert insp.level_note()

    insp.clear()
    qtbot.wait(5)
    assert insp.level_note() == ""
    assert not insp._level_note.isVisible()


# ================================================================================================
# THE INVARIANT THAT WAS ALREADY THERE
# ================================================================================================

@pytest.mark.parametrize("route,_phrase", _ROUTES)
def test_no_route_leaves_a_stale_group_in_the_table_or_the_export(fitted, qtbot,
                                                                  monkeypatch,
                                                                  route, _phrase):
    """C-5, re-asserted through the new messages: whatever a route says, the table it is
    talking about must never hold a group describing a span the event no longer has. The
    message is the surface; this is the guarantee underneath it."""
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    _drive(route, window, doc_id, df, qtbot, monkeypatch)

    assert _levels_for(window._level_fit, ev_id).empty
    _assert_no_stale_group(window, df)
    # ...and the other four events are untouched, exactly as in the drop-only world.
    if window._level_fit is not None:
        for r in range(1, len(df)):
            grp = _levels_for(window._level_fit, int(df["event_id"].iloc[r]))
            assert not grp.empty, f"route {route} dropped an event it never touched"


def test_the_export_is_never_enabled_over_a_group_the_message_says_is_gone(fitted, qtbot):
    """The signal the user has must not point the other way from the file they would get —
    the C-5 signature. If the window is explaining that event 0 has no levels, event 0's
    rows must not be sitting in the exportable table."""
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)

    assert window._event_inspector.level_note()
    stored = window._session.results.get(doc_id, 0, "fit.levels")
    if stored is not None:
        assert _levels_for(stored, ev_id).empty, (
            "the window says these levels are gone and the export still writes them")
    assert window._act_export_levels.isEnabled() == (window._level_fit is not None)
