"""C-5 — EXPORTED SUB-STATE LEVELS DESCRIBED SPANS THAT NO LONGER EXISTED, AND SURVIVED
SAVE/REOPEN.

Fit CUSUM levels, then hand-resize an event's mask in the Event Inspector. Measured on 5
two-sub-level events, event 1 dragged from 6.000 ms to 4.000 ms:

* level 0 started 0.800 ms **before** the event now started,
* level 1 ended 1.200 ms **after** it ended,
* summed sub-state dwell **6.0 ms inside a 4.0 ms event** — a 50% overstatement.

The Inspector dropped the levels from its OWN private ``self._levels``, so the window drew
no CUSUM line; ``MainWindow._level_fit`` and ``Session.results['fit.levels']`` kept them,
Export stayed enabled, and the stale table round-tripped through the project file and came
back exportable. The UI showed no levels while the CSV wrote them — the user's only signal
pointed the wrong way.

THE FIX IS NOT A FOURTH ANNOUNCEMENT. Routing ``eventEdited`` to the level table would fix
the one path that announces and leave every other span-mover (a merge, an undo, a
re-measure, a restore, and whatever is written next month) exactly as broken. Staleness
here is DERIVABLE, because :func:`~nanoscope.analysis.cusum.fit_event_levels` tiles each
parent event exactly — it slices ``[start_idx, end_idx)`` and forces ``starts[0] = 0`` /
``stops[-1] = dev.size`` before offsetting — so for any level table still describing its
events::

    levels.start_idx.min() == event.start_idx   and   levels.end_idx.max() == event.end_idx

``reconcile_levels`` is that comparison, and it is applied by the object that OWNS the
table (MainWindow, feeding Export and the project file) as well as by the Inspector's own
view. One rule, one implementation, no announcement required.

``test_the_tiling_invariant_the_whole_fix_rests_on`` is deliberately the first test in the
file: if ``fit_event_levels`` ever stops tiling, the reconciliation stops meaning anything,
and that must fail loudly here rather than quietly in an export.

ROUND 4 — THE DROP GREW A SECOND HALF, AND FOUR EXPECTATIONS IN THIS FILE CHANGED.
----------------------------------------------------------------------------------
Shipped as above, the drop is all there was, and the owner read it as the feature being
broken: "when I change the window manually, the sublevels disappear. When I change the
window size, the app should automatically recalculate the sublevels." Measured on this
file's own fixture, event 0 spanning (5000, 5600): 14 CUSUM plot items and 32 level rows
before the drag, then 0 items and 25 rows after dragging to (5000, 5300) — the group
deleted from the Inspector, the store, the export and the project file, with nothing
putting it back.

``MainWindow._refit_levels_for_event`` is the missing half: it re-measures the edited
event over its NEW span and splices the group in. **The drop still runs first, always**,
so the invariant above can never be violated even when the re-fit declines (a cold cache,
a rejected row, a span too short) or raises. What changed is the END STATE of a mask
edit — a group that tiles the NEW span, where there used to be no group — and these four
tests, which pinned the deletion itself, now pin that instead:

* ``test_resizing_a_mask_refits_the_levels_the_table_export_actually_reads``
  (was ``..._drops_the_levels_...``)
* ``test_the_other_events_keep_their_levels``
* ``test_the_summary_scalars_follow_the_surviving_rows``
* ``test_the_stale_levels_do_not_round_trip_through_a_project``

The C-5 property they exist for is UNCHANGED and is what they still assert, in the form
that survives the re-fit: **no reader ever holds a group that does not tile its parent.**
``test_a_span_change_that_announces_NOTHING_is_still_caught`` is untouched on purpose — a
span moved by a path that emits no signal gets the drop and no re-fit, because the re-fit
is wired to ``eventEdited`` while the drop is derived from the tables themselves.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.cusum import empty_levels
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
#: Five events, each with two clean sub-states (a 300 pA step then a 500 pA one) so CUSUM
#: has something real to resolve.
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


def _doc(tmp_path, signal, name: str = "levels.dat") -> TraceDocument:
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
    return fit.payload[fit.payload["event_id"] == int(ev_id)]


def _assert_tiles(levels: pd.DataFrame, df: pd.DataFrame, row: int) -> None:
    """The C-5 invariant for one event: its level group spans exactly its current bounds.

    THE assertion this whole file is about, and the one that had to survive round 4: before
    the re-fit existed the only way to satisfy it after an edit was to hold no rows at all,
    so the tests said "empty". Now there are two ways to satisfy it and only one wrong
    answer -- a group describing the OLD span.
    """
    want_lo = int(df.iloc[row]["start_idx"])
    want_hi = int(df.iloc[row]["end_idx"])
    ev_id = int(df.iloc[row]["event_id"])
    grp = levels[levels["event_id"] == ev_id]
    assert not grp.empty, f"event {ev_id} has no levels at all"
    assert (int(grp["start_idx"].min()), int(grp["end_idx"].max())) == (want_lo, want_hi), (
        f"event {ev_id}'s levels span "
        f"[{int(grp['start_idx'].min())}, {int(grp['end_idx'].max())}) but the event now "
        f"spans [{want_lo}, {want_hi}) -- a reader holding this exports sub-state that is "
        f"not inside the event")
    # ...and contiguously inside it, which is what makes the summed sub-state dwell equal
    # the event's dwell rather than the 50% overstatement C-5 measured.
    g = grp.sort_values("level_idx")
    assert list(g["start_idx"])[1:] == list(g["end_idx"])[:-1]


# ================================================================================================
# the property the whole fix rests on
# ================================================================================================

def test_the_tiling_invariant_the_whole_fix_rests_on(fitted):
    """Every event's levels tile that event EXACTLY — first level starts where the event
    starts, last ends where it ends.

    Not a convention: ``fit_event_levels`` restores the trimmed edges with
    ``starts[0] = 0`` / ``stops[-1] = dev.size`` before offsetting by the event's own
    ``start_idx``. The reconciliation is only meaningful while this holds, so it is
    asserted against a REAL fit rather than assumed.

    THIS ONE PASSES BEFORE AND AFTER, deliberately: it pins the PREMISE, not the fix. If a
    change to ``fit_event_levels`` ever stops the levels tiling their parent, every
    reconciliation in this file silently starts answering a different question, and that
    must fail loudly here rather than quietly inside an export.
    """
    window, _doc_id, df = fitted
    levels = window._level_fit.payload
    assert len(levels) > len(df), "the fit resolved no sub-structure to reconcile against"

    spans = df.set_index(df["event_id"].astype("int64"))[["start_idx", "end_idx"]]
    for ev_id, grp in levels.groupby("event_id"):
        want = spans.loc[int(ev_id)]
        assert int(grp["start_idx"].min()) == int(want["start_idx"])
        assert int(grp["end_idx"].max()) == int(want["end_idx"])
        # ...and the levels are contiguous inside it: no gap, no overlap.
        g = grp.sort_values("level_idx")
        assert list(g["start_idx"])[1:] == list(g["end_idx"])[:-1]


# ================================================================================================
# the measured defect
# ================================================================================================

def test_resizing_a_mask_refits_the_levels_the_table_export_actually_reads(fitted, qtbot):
    """THE headline, in its round-4 form: after the drag, every reader holds a group that
    describes the NEW span — never the old one, and (this is the part that changed) never
    nothing at all.

    Was ``test_resizing_a_mask_drops_the_levels_from_the_table_export_actually_reads``,
    which asserted the group was gone. The C-5 property it existed for is untouched: no
    reader may hold levels describing a span the event no longer has. What changed is that
    "gone" is no longer the only way to satisfy it — the owner asked for the second half
    ("the app should automatically recalculate the sublevels"), and a re-measured group
    satisfies the invariant while a deleted one merely avoids violating it.

    Measured before the re-fit existed, this fixture, event 0 (5000, 5600) dragged to
    (5000, 5300): the Inspector went from 14 CUSUM plot items to 0 and the store from 32
    level rows to 25.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    before = _levels_for(window._level_fit, ev_id)
    assert len(before) >= 2, "need a multi-level event for this to be the measured case"
    start = int(df.iloc[0]["start_idx"])
    end = int(df.iloc[0]["end_idx"])

    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    assert int(df.iloc[0]["end_idx"]) < end, "the drag did not actually move the mask"
    fit = window._level_fit
    assert fit is not None
    # THE STALE ROWS ARE GONE: not one boundary of the old fit survives past the new end.
    assert int(_levels_for(fit, ev_id)["end_idx"].max()) == int(df.iloc[0]["end_idx"])
    _assert_tiles(fit.payload, df, 0)

    # ...in the table Export and project save actually read, not merely in this window.
    stored = window._session.results.get(doc_id, 0, "fit.levels")
    assert stored is not None
    _assert_tiles(stored.payload, df, 0)

    # ...and in the Inspector's own view, which is where the owner reported the loss.
    insp_levels = window._event_inspector._levels
    assert insp_levels is not None
    _assert_tiles(insp_levels, df, 0)
    assert window._event_inspector._cusum_items, (
        "the waveform draws no level segments — the fit is back in the table but the "
        "window still shows the disappearance the owner reported")


def test_the_other_events_keep_their_levels(fitted, qtbot):
    """Only the edited event loses its fit. Dropping the whole table would cost the user
    four correct measurements to punish one edit."""
    window, doc_id, df = fitted
    survivors = [int(v) for v in df["event_id"].iloc[1:]]
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    fit = window._level_fit
    assert fit is not None
    # The edited event is re-measured (round 4), NOT left describing its old span...
    _assert_tiles(fit.payload, df, 0)
    # ...and every untouched event is left exactly as it was fitted: same row count, same
    # boundaries. A splice that re-numbered or re-ordered its neighbours would be a
    # different defect wearing this fix's clothes.
    for ev in survivors:
        after = _levels_for(fit, ev)
        assert not after.empty, f"event {ev}'s levels were dropped too"
        row = int(np.flatnonzero(df["event_id"].to_numpy() == ev)[0])
        _assert_tiles(fit.payload, df, row)
    assert list(fit.payload["event_id"]) == sorted(fit.payload["event_id"]), (
        "the spliced table is no longer ordered by event_id, which every groupby reader "
        "and the export's row order depend on")
    assert window._act_export_levels.isEnabled(), (
        "there are still valid levels; the export must stay available")


def test_the_summary_scalars_follow_the_surviving_rows(fitted, qtbot):
    """``scalars`` describes the fit, so they are RE-DERIVED from whatever the table holds
    after the edit — never patched, never left over.

    Round 4 changed the expected numbers, not the rule. The drop alone took
    ``n_events_fitted`` down by one; the drop-then-splice keeps every event fitted and moves
    ``n_levels_total`` instead, because the shorter span resolves fewer sub-states. Both are
    ``levels_summary`` of the table actually in hand, which is the property worth pinning:
    a scalar that does not follow the rows is what let ``n_events_fitted`` count an event
    whose levels were gone.
    """
    from nanoscope.analysis.cusum import levels_summary

    window, doc_id, df = fitted
    before_events = int(window._level_fit.scalars["n_events_fitted"])
    before_levels = int(window._level_fit.scalars["n_levels_total"])
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    fit = window._level_fit
    assert fit.scalars == levels_summary(fit.payload), (
        "the summary does not describe the table it sits on")
    assert int(fit.scalars["n_events_fitted"]) == before_events, (
        "the edited event was re-fitted, so it is still a fitted event")
    assert int(fit.scalars["n_levels_total"]) < before_levels, (
        "half the event was cut away and the level count did not move — the re-fit is "
        "recycling the old rows rather than measuring the new span")


def test_a_span_change_that_announces_NOTHING_is_still_caught(fitted, qtbot):
    """The point of deriving instead of announcing.

    A merge rewrites spans through the panel's merge engine, which has no idea a level fit
    exists and emits no level-related signal at all. The levels of every event it touched
    must still go.
    """
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    # Move the span the way any un-announced writer would: on the shared frame.
    col = df.columns.get_loc("end_idx")
    df.iat[0, col] = int(df.iat[0, col]) - 40
    window._refresh_surface_windows()          # the one fan-out every mutation reaches
    qtbot.wait(5)

    assert window._level_fit is not None
    assert _levels_for(window._level_fit, ev_id).empty, (
        "a span moved by a path that announces nothing left its levels in the export")


def test_the_whole_fit_retiring_clears_the_menu_and_the_store(fitted, qtbot):
    """When nothing survives, absence must be complete: no export, no store entry, no
    Inspector readout. A half-retired fit is what round-tripped through the project."""
    window, doc_id, df = fitted
    col = df.columns.get_loc("end_idx")
    for r in range(len(df)):
        df.iat[r, col] = int(df.iat[r, col]) - 40
    window._refresh_surface_windows()
    qtbot.wait(5)

    assert window._level_fit is None
    assert not window._act_export_levels.isEnabled()
    assert window._session.results.get(doc_id, 0, "fit.levels") is None
    assert window._event_inspector._levels is None


def test_an_untouched_fit_is_left_exactly_alone(fitted, qtbot):
    """The idempotence claim, and the "must not move a number" side of it: refreshing
    repeatedly with nothing edited returns the very same frame object."""
    window, _doc_id, _df = fitted
    first = window._level_fit
    for _ in range(5):
        window._reconcile_level_fit()      # the method under test, called directly
        window._refresh_surface_windows()
        qtbot.wait(2)
    assert window._level_fit is first, "an untouched level fit was rebuilt"
    assert window._level_fit.payload is first.payload


def test_the_stale_levels_do_not_round_trip_through_a_project(fitted, qtbot, tmp_path):
    """The property that made this survive a restart: the stale table was persisted and
    came back exportable."""
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    proj = tmp_path / "levels.nanoscope"
    window._write_project(str(proj))
    saved = window._session.results.get(doc_id, 0, "fit.levels")
    assert saved is not None
    # Round 4: the store now holds a RE-FITTED group for this event rather than none, so
    # the property is stated as the invariant itself. Either answer is acceptable; a group
    # describing the pre-drag span is not, and that is the one that round-tripped.
    _assert_tiles(saved.payload, df, 0)
    assert int(saved.payload[saved.payload["event_id"] == ev_id]["end_idx"].max()) \
        == int(df.iloc[0]["end_idx"])


# ================================================================================================
# the rule itself, as a pure function
# ================================================================================================

def reconcile_levels(*a, **k):
    """The rule under test, looked up LATE — see the note in ``_tables`` of the sibling
    C-1 file: against a reverted package a module-level import would turn every test in
    this file into a collection error instead of a failed assertion."""
    from nanoscope.analysis import cusum

    fn = getattr(cusum, "reconcile_levels", None)
    assert fn is not None, (
        "nanoscope.analysis.cusum declares no reconcile_levels: nothing derives whether a "
        "level table still describes its events")
    return fn(*a, **k)


def _levels(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["event_id", "start_idx", "end_idx"])


def _events(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["event_id", "start_idx", "end_idx"])


def test_reconcile_keeps_a_table_that_still_describes_its_events():
    lv = _levels([(1, 10, 15), (1, 15, 20), (2, 30, 40)])
    ev = _events([(1, 10, 20), (2, 30, 40)])
    kept, dropped = reconcile_levels(lv, ev)
    assert dropped == [] and kept is lv, "an untouched table must be returned unchanged"


def test_reconcile_drops_only_the_group_whose_span_moved():
    lv = _levels([(1, 10, 15), (1, 15, 20), (2, 30, 40)])
    ev = _events([(1, 10, 18), (2, 30, 40)])          # event 1 was shortened
    kept, dropped = reconcile_levels(lv, ev)
    assert dropped == [1]
    assert list(kept["event_id"]) == [2]


def test_reconcile_drops_an_event_that_left_the_table():
    lv = _levels([(1, 10, 20), (2, 30, 40)])
    kept, dropped = reconcile_levels(lv, _events([(2, 30, 40)]))
    assert dropped == [1] and list(kept["event_id"]) == [2]


def test_reconcile_returns_none_when_nothing_survives():
    lv = _levels([(1, 10, 20)])
    kept, dropped = reconcile_levels(lv, _events([(1, 10, 21)]))
    assert kept is None and dropped == [1], (
        "absence is how the readout and the export say 'no level fit'")


def test_reconcile_is_total():
    """PIN UPDATED DELIBERATELY (empty input). It used to assert that an EMPTY levels
    table comes back as ``kept=None``; it now comes back as the very frame it was given.

    The reason is the divergence that assertion was hiding. ``reconcile_levels`` also
    returns ``(None, [])`` on the three inputs it REFUSES to certify — a table with no
    ``start_idx``/``end_idx``, and an ``event_id`` column that will not cast — where it
    cannot name the ids it is discarding. With ``None`` ALSO meaning "there was nothing
    here", ``dropped`` was the only signal left, and the two callers read it in opposite
    directions: the Inspector took ``kept=None`` and drew no CUSUM line while MainWindow
    saw ``dropped == []``, kept the uncertified table and left Export enabled. That is
    the C-5 symptom the function exists to end.

    So ``kept is levels`` is now the "nothing moved" signal, total across every branch,
    and it is what both callers branch on. Nothing about a real reconciliation moved: an
    empty level table still exports nothing (``_export_levels`` gates on ``.empty``) and
    still draws nothing.
    """
    ev = _events([(1, 10, 20)])
    assert reconcile_levels(None, ev) == (None, [])

    empty = empty_levels()
    kept, dropped = reconcile_levels(empty, ev)
    assert kept is empty and dropped == []

    assert reconcile_levels(_levels([(1, 10, 20)]), None) == (None, [1])
    assert reconcile_levels(_levels([(1, 10, 20)]), _events([])) == (None, [1])


def test_nothing_moved_is_identity_on_every_branch():
    """The contract the two callers depend on: ``kept is levels`` iff nothing was dropped.

    Fails against the pre-fix function, where a refusal and a clean table were told apart
    only by ``dropped`` — which a refusal cannot always fill in.
    """
    ev = _events([(1, 10, 20), (2, 30, 40)])
    lv = _levels([(1, 10, 15), (1, 15, 20), (2, 30, 40)])

    # unchanged -> identity
    assert reconcile_levels(lv, ev)[0] is lv
    assert reconcile_levels(empty_levels(), ev)[0] is not None

    # every refusal -> NOT identity, so a caller branching on identity acts on it
    no_span = lv.drop(columns=["start_idx"])
    assert reconcile_levels(no_span, ev)[0] is not no_span
    unjoinable = lv.copy()
    unjoinable["event_id"] = ["a", "b", "c"]
    assert reconcile_levels(unjoinable, ev)[0] is not unjoinable
    # ...and each of those refusals names no ids, which is exactly why `dropped`
    # cannot be the signal.
    assert reconcile_levels(no_span, ev)[1] == []
    assert reconcile_levels(unjoinable, ev)[1] == []


def test_reconcile_cannot_certify_a_table_it_cannot_join_on():
    """An unusable ``event_id`` means nothing in the table can be shown to be current, and
    a level table that cannot be certified must not be exported."""
    lv = _levels([(1, 10, 20)])
    ev = pd.DataFrame({"event_id": [np.nan], "start_idx": [10], "end_idx": [20]})
    assert reconcile_levels(lv, ev)[0] is None
    assert reconcile_levels(pd.DataFrame({"event_id": [1]}), ev)[0] is None


def test_both_callers_use_the_same_function():
    """One rule, one implementation. Two copies of this question is what produced a window
    drawing no levels while the export wrote them."""
    import inspect

    from nanoscope.gui import event_inspector, main_window

    assert "reconcile_levels" in inspect.getsource(
        main_window.MainWindow._reconcile_level_fit)
    assert "reconcile_levels" in inspect.getsource(
        event_inspector.EventInspectorWindow._reconcile_levels)
    assert not hasattr(event_inspector.EventInspectorWindow,
                       "_invalidate_levels_for_row"), (
        "the Inspector's private by-event_id rule is back")


# ================================================================================================
# C-5 REOPENED AT THE ONE INPUT THE SHARED RULE DECLINES TO ANSWER
# ================================================================================================
# `reconcile_levels` returns `(None, [])` on the inputs it REFUSES to certify, because it
# cannot read the ids it is discarding. With both callers branching on `dropped`, that
# refusal was read in opposite directions: the Inspector assigned `self._levels = None`
# and drew no CUSUM line, while MainWindow saw `dropped == []`, kept the uncertified
# table in `Session.results['fit.levels']` and left Export enabled. A window showing
# nothing while the export writes the full table is precisely the C-5 signature.
#
# The fix is at the function: `kept is levels` is the "nothing moved" signal on every
# branch, and it is what both callers now read.

def _uncertifiable(fit):
    """The level table with its ``start_idx`` column removed — a table that can no longer
    state which spans it describes. Reachable from a project written by another build, an
    imported table, or a schema change; `fit_event_levels` does not produce it today."""
    return fit.payload.drop(columns=["start_idx"])


def test_a_level_table_that_cannot_be_certified_is_not_left_exportable(fitted, qtbot):
    """THE REPRO. Before the fix MainWindow returned early on ``dropped == []`` and the
    uncertifiable table stayed in the store with Export enabled."""
    from dataclasses import replace

    window, doc_id, df = fitted
    window._level_fit = replace(window._level_fit, payload=_uncertifiable(window._level_fit))
    window._session.results.put(window._level_fit)

    window._reconcile_level_fit()
    qtbot.wait(5)

    assert window._level_fit is None or window._level_fit.payload is None, (
        "the window is still holding a level table it cannot show describes the events "
        "on screen")
    assert not window._act_export_levels.isEnabled(), (
        "Export sub-state levels is still enabled on a table nothing certified")


def test_the_two_callers_do_not_disagree_about_an_uncertifiable_table(fitted, qtbot):
    """The Inspector and MainWindow must reach the SAME verdict. Before the fix the
    Inspector dropped it and MainWindow kept it."""
    from dataclasses import replace

    window, doc_id, df = fitted
    bad = _uncertifiable(window._level_fit)
    insp = window._event_inspector
    # Assigned rather than pushed through `set_levels`, which REDRAWS: a table with no
    # `start_idx` also breaks `_draw_levels` (AttributeError on the itertuples row). Same
    # unreachable-today input, a different consequence, and not what this test is about.
    insp._levels = bad
    insp._reconcile_levels()
    inspector_kept = insp._levels is not None

    window._level_fit = replace(window._level_fit, payload=bad)
    window._session.results.put(window._level_fit)
    window._reconcile_level_fit()
    main_kept = window._level_fit is not None and window._level_fit.payload is not None

    assert inspector_kept == main_kept, (
        f"the Inspector {'kept' if inspector_kept else 'dropped'} the table and the "
        f"export path {'kept' if main_kept else 'dropped'} it")


def test_the_export_reconciles_rather_than_reading_around_the_reconciler(fitted, qtbot,
                                                                        monkeypatch):
    """The guarantee must not rest on the refresher having fired. Every path that moves a
    span emits one of ``SURFACE_MUTATION_SIGNALS`` today, so this is hardening — but the
    batch's own rule is that a second reader becomes a passer-through."""
    import inspect as _inspect

    from nanoscope.gui.main_window import MainWindow as _MW

    src = _inspect.getsource(_MW._export_levels)
    assert "_reconcile_level_fit" in src, (
        "_export_levels reads self._level_fit directly; it must pass through the "
        "reconciler like every other reader")

    window, doc_id, df = fitted
    calls = {"n": 0}
    real = _MW._reconcile_level_fit
    monkeypatch.setattr(_MW, "_reconcile_level_fit",
                        lambda self: (calls.__setitem__("n", calls["n"] + 1),
                                      real(self))[1])
    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path",
                        lambda *a, **k: None)
    window._export_levels()
    assert calls["n"] == 1


def test_reconciling_a_clean_table_before_export_changes_nothing(fitted, qtbot):
    """PASSES BOTH — the non-regression the pass-through could plausibly have broken.
    An ordinary export must write byte-identical rows."""
    window, doc_id, df = fitted
    before = window._level_fit.payload.copy(deep=True)
    window._reconcile_level_fit()
    pd.testing.assert_frame_equal(window._level_fit.payload, before, check_dtype=True)
    assert window._act_export_levels.isEnabled()


# ================================================================================================
# ROUND 4 — THE SECOND HALF: RE-FIT THE EDITED EVENT OVER ITS NEW SPAN
# ================================================================================================
# The drop above is correct and stays. What the owner reported is that it was the WHOLE
# behaviour: "when I change the window manually, the sublevels disappear. When I change the
# window size, the app should automatically recalculate the sublevels."
#
# `MainWindow._refit_levels_for_event` re-measures the one edited event and splices its group
# back in. The tests below pin the properties that make that safe rather than merely nice:
# the drop runs FIRST (so a declined or failed re-fit degrades to today's behaviour and never
# to a stale group), and the re-fit is the SAME fit its neighbours got.

def _status(window) -> str:
    return window.statusBar().currentMessage()


def test_the_refit_reuses_the_recorded_sigma_and_tau_not_a_fresh_estimate(fitted, qtbot,
                                                                          monkeypatch):
    """The re-fit must segment at the significance scale the ORIGINAL fit resolved.

    Not a performance note (though it is that too — measured 3.4 ms against 26.3 ms for one
    event on this fixture, because a fresh tau re-walks every event's residual in the
    recording). It is a CORRECTNESS one: `tau` is estimated from the open-pore residual
    around the events, so an estimate taken after the edit is taken from a different event
    set. Two neighbouring groups in one exported table would then answer to two different
    z-test scales with nothing on screen saying so.

    Pinned at the call, because "we passed the cached numbers" is exactly the kind of claim
    that survives a refactor as a comment while the argument quietly goes missing.
    """
    import nanoscope.analysis.cusum as cusum

    window, doc_id, df = fitted
    params = window._level_fit.params
    assert "sigma_pA" in params and "tau_samples" in params, (
        "modules.fit_levels did not record what the fit ran with, so no re-fit can reuse it")

    seen: list[dict] = []
    real = cusum.fit_event_levels

    def spy(*a, **kw):
        seen.append(dict(kw))
        return real(*a, **kw)

    monkeypatch.setattr(cusum, "fit_event_levels", spy)
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    assert seen, "the mask edit did not re-fit anything at all"
    call = seen[-1]
    assert call["sigma"] == params["sigma_pA"]
    assert call["tau"] == params["tau_samples"]
    assert call["tau"] is not None, (
        "tau was left to be re-measured, which is both the slow path and a different "
        "estimator from the one the neighbouring groups were fitted with")


def test_a_cold_cache_keeps_the_drop_and_says_why(fitted, qtbot):
    """A fit RESTORED FROM A PROJECT has no inputs behind it, and re-deriving them means a
    644 ms open-pore baseline estimate — on the GUI thread, inside a mouse-release handler.

    So the cold case keeps exactly the shipped behaviour (drop) and tells the user the one
    thing that fixes it. Simulated by clearing the cache, which is precisely the state
    `_sync_export_action`'s adoption branch leaves behind after a reopen.
    """
    window, doc_id, df = fitted
    window._level_fit_inputs = None
    ev_id = int(df["event_id"].iloc[0])
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])

    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    assert _levels_for(window._level_fit, ev_id).empty, (
        "levels were spliced back without the inputs that measure them")
    message = _status(window)
    assert "Fit sub-state levels" in message, (
        f"the status bar does not tell the user how to recover: {message!r}")
    # ...and the OTHER events are untouched, exactly as in the drop-only world.
    for r in range(1, len(df)):
        _assert_tiles(window._level_fit.payload, df, r)


def test_a_rejected_event_is_not_refitted(fitted, qtbot):
    """The fit ran ``accepted_only=True``, so a rejected event has no levels BY
    CONSTRUCTION. Inventing some on an edit would put rows in the export that re-running
    the fit would not produce — a table that disagrees with its own recipe."""
    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    df.iat[0, df.columns.get_loc("accepted")] = False
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])

    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    assert _levels_for(window._level_fit, ev_id).empty


def test_a_span_too_short_to_resolve_a_boundary_is_not_refitted(fitted, qtbot):
    """Below ``MIN_SAMPLES_FOR_LEVELS`` the fit does not fail — it returns ONE "level"
    spanning the whole event, which says nothing and would sit in the export looking like a
    measurement. Decline, keep the drop, and say which."""
    from nanoscope.analysis.cusum import MIN_SAMPLES_FOR_LEVELS

    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])
    start = int(df.iloc[0]["start_idx"])
    _drag_mask(window, doc_id, df, 0, start + MIN_SAMPLES_FOR_LEVELS - 1)
    qtbot.wait(10)

    assert int(df.iloc[0]["end_idx"]) - start < MIN_SAMPLES_FOR_LEVELS
    assert _levels_for(window._level_fit, ev_id).empty
    assert "below the" in _status(window)


def test_a_refit_that_raises_leaves_the_drop_and_never_a_stale_group(fitted, qtbot,
                                                                     monkeypatch):
    """THE ORDERING GUARANTEE, stated as a test: drop first, splice second. However the
    re-fit fails, the worst reachable outcome is the behaviour that shipped."""
    import nanoscope.analysis.cusum as cusum

    window, doc_id, df = fitted
    ev_id = int(df["event_id"].iloc[0])

    def boom(*a, **kw):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cusum, "fit_event_levels", boom)
    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    assert _levels_for(window._level_fit, ev_id).empty, (
        "a failed re-fit left the pre-drag group standing")
    for r in range(1, len(df)):
        _assert_tiles(window._level_fit.payload, df, r)
    assert "synthetic failure" in _status(window)


def test_the_last_surviving_group_comes_back_after_its_own_edit(fitted, qtbot):
    """The splice must work when the reconcile has just retired the WHOLE fit.

    Reject every other event first, so event 0 is the only one carrying levels; editing it
    then takes `reconcile_levels` down to ``kept is None``, which is
    `_invalidate_level_fit`'s branch — the fit out of this window's hands, out of the store
    and out of the menu. Splicing has to rebuild from that, or the one-event case (which is
    every single-event detection) silently loses its fit on the first drag.
    """
    window, doc_id, df = fitted
    for r in range(1, len(df)):
        df.iat[r, df.columns.get_loc("accepted")] = False
    window._fit_levels()               # re-fit so only event 0 has levels
    qtbot.wait(10)
    assert set(window._level_fit.payload["event_id"]) == {int(df["event_id"].iloc[0])}

    start, end = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    _drag_mask(window, doc_id, df, 0, start + (end - start) // 2)
    qtbot.wait(10)

    assert window._level_fit is not None, (
        "the whole fit retired and nothing rebuilt it — a single-event detection loses its "
        "levels on the first mask drag")
    _assert_tiles(window._level_fit.payload, df, 0)
    assert window._session.results.get(doc_id, 0, "fit.levels") is not None
    assert window._act_export_levels.isEnabled()
