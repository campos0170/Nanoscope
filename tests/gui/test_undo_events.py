"""THE DESTRUCTIVE-ACTION TIER — undo over every EventSet mutation, and the one
confirmation undo cannot replace.

Three measured defects, one guarantee. Each was reproduced through a real MainWindow
on the current tree before this file existed:

* **Re-running detection replaced the EventSet outright.** Six curated events, two
  hand rejections, one hand-added row -> ``rejections 2 -> 0``, ``dialogs shown: []``,
  and ``[a for a in dir(win) if 'undo' in a.lower()] == []``: there was no undo of any
  kind in the application.
* **An Event-Inspector mask resize rewrote 14 columns** — ``start_idx``, ``end_idx``,
  ``t_start_s``, ``dwell_s``, ``baseline_pA``, ``depth_pA``, ``frac_depth``,
  ``ecd_pA_s``, ``max_dev_pA``, ``mean_dev_pA``, ``rise_s``, ``fall_s``, ``snr``,
  ``n_levels`` — and left ``dirty=False`` with the title still ``'NanoScope'``.
  ``eventEdited`` was connected to ``refresh_after_edit`` and to nothing else.
* **Closing a document took its results with it, unguarded.** A detected file:
  ``results 1 -> 0, prompts: []``. A SEGMENT carrying 1 rejected + 1 hand-added event:
  ``result -> None, prompts: []`` — and a segment is in-memory only, with no reopen
  path at all.

The fix is :mod:`nanoscope.core.history`: one snapshot stack that every mutation is
measured against, whether or not it announces itself. These tests drive the real
widgets (the table model, the Event Inspector's region drag, the merge engine, the
threaded detector) rather than the stack directly, because the stack being correct in
isolation is exactly what would not have caught any of the three.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import SCHEMA, EventSet
from nanoscope.analysis.scoring import ADDED_COLUMN, ensure_added_column
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import MAX_DEPTH, describe_change
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
#: Six well-separated events, plus a close pair the merge engine can collapse.
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400), (50_000, 50_700)]
_PAIR = [(5_000, 5_400), (5_450, 5_900), (20_000, 20_600), (40_000, 40_400)]


# ================================================================================================
# builders
# ================================================================================================
def _signal(spans=_SPANS, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in spans:
        x[s:e] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, signal, *, name="undo.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(signal.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels if c.kind is ChannelKind.CURRENT)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _detect_into(window: MainWindow, doc_id: str, signal: np.ndarray,
                 channel: int = 0) -> EventSet:
    """Store a detection the way the analysis controller does (main thread, one
    ``add_result``), so the test does not depend on thread timing for the SETUP."""
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    events = EventSet(df, prov)
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=window._session.calibration(doc_id,
                                                                      channel).fingerprint(),
        payload=events))
    return events


def _curate(window: MainWindow, rows=(1, 3)) -> pd.DataFrame:
    """Reject ``rows`` THROUGH the table (the real curation path), add a curator's own
    flag token and a hand-added marker, and return a detached copy of the result."""
    df = window._analysis_panel.current_detection().payload.df
    ensure_added_column(df)
    for r in rows:
        window._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[r]), False)
    df.iat[rows[-1], df.columns.get_loc("flag")] = "operator_note"
    df.iat[0, df.columns.get_loc(ADDED_COLUMN)] = True
    window._analysis_panel.eventAcceptanceChanged.emit(int(df["event_id"].iloc[0]), True)
    return df.copy(deep=True)


def _differing_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Every column whose values differ, including nullable values and dtypes.

    ``Series.equals`` treats NaN/``pd.NA`` in the same positions as equal without
    asking pandas' three-valued boolean result to become a Python ``bool``.  The old
    ``bool((x == y).all())`` became ambiguous as soon as the event table gained a
    nullable per-event provenance column, so the history assertion crashed before it
    could say whether undo had actually restored the frame.
    """
    if list(a.columns) != list(b.columns):
        return [f"columns {list(a.columns)} != {list(b.columns)}"]
    if len(a) != len(b):
        return [f"len {len(a)} != {len(b)}"]
    return [col for col in a.columns if not a[col].equals(b[col])]


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, tmp_path):
    """A window with one recording, one detection and the Event Inspector open."""
    signal = _signal()
    doc_id, channel = _install(window, _doc(tmp_path, signal))
    events = _detect_into(window, doc_id, signal, channel)
    window._open_event_inspector()
    return window, doc_id, channel, events, signal


# ================================================================================================
# C14 — re-running detection destroyed the curation, silently and irrecoverably
# ================================================================================================

def test_re_detection_replaces_curation_and_undo_puts_every_column_back(detected, qtbot):
    """THE headline. Measured before: 2 rejections -> 0, dialogs shown ``[]``, no undo.

    The restored table is compared COLUMN BY COLUMN, including ``added`` and the
    hand-written ``flag`` token — the non-schema state the frozen 20-column schema does
    not carry and that a naive "keep the EventSet" snapshot would drop.
    """
    window, doc_id, channel, _events, _signal_ = detected
    curated = _curate(window)
    n_rejected = int((~curated["accepted"].to_numpy(bool)).sum())
    assert n_rejected == 2
    assert curated[ADDED_COLUMN].to_numpy(bool).sum() == 1
    assert "operator_note" in set(curated["flag"])

    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(30)

    after = panel.current_detection().payload.df
    assert int((~after["accepted"].to_numpy(bool)).sum()) == 0, (
        "the re-detection is still expected to replace the table — undo is the remedy, "
        "not a merge of the old curation into the new rows")

    assert window._act_undo.isEnabled()
    assert window._act_undo.text() == "Undo Re-run detection"
    window._undo_events()
    qtbot.wait(10)

    restored = panel.current_detection().payload.df
    assert _differing_columns(curated, restored) == []
    assert restored is panel._table.dataframe, (
        "the table must be looking at the RESTORED frame, not at a detached copy")


def test_redo_returns_to_the_new_detection_and_undo_comes_back_again(detected, qtbot):
    window, *_ = detected
    curated = _curate(window)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(30)
    fresh = panel.current_detection().payload.df.copy(deep=True)

    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(curated, panel.current_detection().payload.df) == []
    assert window._act_redo.isEnabled()
    assert window._act_redo.text() == "Redo Re-run detection"

    window._redo_events()
    qtbot.wait(5)
    assert _differing_columns(fresh, panel.current_detection().payload.df) == []

    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(curated, panel.current_detection().payload.df) == []


def test_a_re_detection_over_curated_events_says_so_and_names_the_remedy(detected, qtbot):
    """A modal would be wrong here — re-detecting is the RECOMMENDED move after the
    baseline change — but silence is what made the loss invisible. Measured before:
    nothing in the status bar, nothing in a dialog, nothing anywhere."""
    window, *_ = detected
    _curate(window)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(30)
    msg = window.statusBar().currentMessage()
    # The 2 hand rejections are what this re-run destroys. The hand-ADDED event is not: it is
    # carried into the new table, and the same line says so.
    assert "2 rejected" in msg and "1 hand-added" not in msg
    assert "Kept 1 hand-made event" in msg
    assert "Undo" in msg


def test_accept_all_is_one_undoable_step(detected, qtbot):
    """``_set_all_accepted`` rewrites the whole ``accepted`` column on one click."""
    window, *_ = detected
    before = _curate(window)
    insp = window._event_inspector
    insp.accept_all()
    qtbot.wait(5)
    live = window._analysis_panel.current_detection().payload.df
    assert bool(live["accepted"].to_numpy(bool).all())
    assert window._act_undo.text() == "Undo Accept all events"

    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(
        before, window._analysis_panel.current_detection().payload.df) == []


def test_a_merge_is_one_undoable_step(window, tmp_path, qtbot):
    signal = _signal(_PAIR)
    doc_id, channel = _install(window, _doc(tmp_path, signal, name="merge.dat"))
    _detect_into(window, doc_id, signal, channel)
    before = window._analysis_panel.current_detection().payload.df.copy(deep=True)

    # 'Merge all' runs on the analysis controller now (it costs 3.3-8.7 s on the owner's
    # real recordings, and used to freeze the window for all of it), so the table changes
    # when the job lands rather than inside the click.
    with qtbot.waitSignal(window._analysis_panel.mergeFinished, timeout=30_000):
        window._on_merge_all_requested(0.005)
    qtbot.wait(20)
    after = window._analysis_panel.current_detection().payload.df
    assert len(after) < len(before)
    assert window._act_undo.text().startswith("Undo Merge events")

    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(
        before, window._analysis_panel.current_detection().payload.df) == []


# ================================================================================================
# C16 — a mask resize rewrote nine columns and never marked the project dirty
# ================================================================================================

def _drag_mask(window, qtbot, *, row: int = 2, pad: int = 300) -> int:
    """Widen event ``row``'s mask by ``pad`` samples each side through the REAL region
    drag, exactly as releasing the Event Inspector's region handle does."""
    insp = window._event_inspector
    df = window._analysis_panel.current_detection().payload.df
    ev_id = int(df["event_id"].iloc[row])
    insp.show_event(ev_id)
    r = insp._row_for_id(ev_id)
    xv = window._session.x_axis(window._current_selection[0]).values
    s = int(df.iat[r, df.columns.get_loc("start_idx")])
    e = int(df.iat[r, df.columns.get_loc("end_idx")])
    insp._region.setRegion((float(xv[s - pad]), float(xv[e + pad])))
    insp._on_region_edited()
    qtbot.wait(5)
    return ev_id


def test_a_mask_resize_marks_the_project_dirty(detected, qtbot):
    """Measured before: 14 columns rewritten, ``dirty=False``, title ``'NanoScope'``.
    Open a saved project, resize masks all afternoon, quit — no prompt, all gone."""
    window, *_ = detected
    window._mark_clean()
    assert window._dirty is False
    _drag_mask(window, qtbot)
    assert window._dirty is True, "a mask resize is unsaved work a project save would persist"
    assert window.windowTitle().startswith("•")


def test_a_mask_resize_is_undoable_column_by_column(detected, qtbot):
    window, *_ = detected
    before = window._analysis_panel.current_detection().payload.df.copy(deep=True)
    _drag_mask(window, qtbot)
    live = window._analysis_panel.current_detection().payload.df
    moved = _differing_columns(before, live)
    assert "start_idx" in moved and "dwell_s" in moved and "depth_pA" in moved
    assert window._act_undo.text() == "Undo Resize event mask"

    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(
        before, window._analysis_panel.current_detection().payload.df) == []


def test_an_edit_that_announces_nothing_is_still_undoable(detected, qtbot):
    """The safety net. A mutation path that forgets to tell MainWindow must still land
    on the stack — with a derived label instead of a declared one. A missed door costs
    a word, never a row."""
    window, *_ = detected
    before = window._analysis_panel.current_detection().payload.df.copy(deep=True)
    df = window._analysis_panel.current_detection().payload.df
    df.iat[2, df.columns.get_loc("accepted")] = not bool(df.iat[2, df.columns.get_loc("accepted")])
    # No signal, no checkpoint call: the frame was written behind everyone's back.
    assert not window._history.can_undo(window._undo_key())

    window._on_edit_menu_about_to_show()
    assert window._act_undo.isEnabled()
    assert window._act_undo.text() == "Undo Curate event"
    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(
        before, window._analysis_panel.current_detection().payload.df) == []


# ================================================================================================
# C17 — closing a document (or a segment) took its results with it, unguarded
# ================================================================================================

def test_closing_a_detected_recording_confirms_and_names_the_events(detected, qtbot):
    """Measured before: ``results 1 -> 0, prompts: []``."""
    window, doc_id, *_ = detected
    _curate(window)
    asked: list = []
    window._confirm_close_discarding_events = lambda ids: (
        asked.append(window._describe_close_stake(ids)) or False)

    window._close_document(doc_id)
    assert window._session.has_document(doc_id), "refusing the prompt must keep it open"
    assert len(window._session.results.for_doc(doc_id)) == 1
    (sentence, is_segment), = asked
    assert "1 detection" in sentence
    assert "2 rejected" in sentence and "1 hand-added" in sentence
    assert is_segment is False


def test_closing_a_curated_segment_says_it_cannot_be_reopened(window, tmp_path, qtbot):
    """A segment is in-memory only: there is no file to reopen and no way to detect
    these events again. Measured before: ``result -> None, prompts: []``."""
    signal = _signal()
    parent_id, _ = _install(window, _doc(tmp_path, signal, name="parent.dat"))
    seg_id = window._session.create_segment(parent_id, 1_000, 45_000, name="seg A")
    window._on_selection(seg_id, 0)
    seg_signal = np.asarray(window._session.calibrated(seg_id, 0).data)
    _detect_into(window, seg_id, seg_signal, 0)
    _curate(window)

    asked: list = []
    window._confirm_close_discarding_events = lambda ids: (
        asked.append(window._describe_close_stake(ids)) or False)
    window._close_document(seg_id)

    assert window._session.has_document(seg_id)
    (sentence, is_segment), = asked
    assert is_segment is True, "the prompt must know a segment has no reopen path"
    assert "hand-added" in sentence


def test_closing_a_recording_with_no_events_asks_nothing(window, tmp_path):
    """The mirror mistake this window has already paid for once: nagging about an
    ordinary action. Open a recording, look at it, close it — no dialog."""
    doc_id, _ = _install(window, _doc(tmp_path, _signal(), name="quiet.dat"))
    assert window._describe_close_stake([doc_id]) == ("", False)
    asked: list = []
    window._confirm_close_discarding_events = lambda ids: (asked.append(ids) or True)
    window._close_document(doc_id)
    assert not window._session.has_document(doc_id)


def test_confirming_the_close_actually_closes_it(detected):
    window, doc_id, *_ = detected
    _curate(window)
    window._confirm_close_discarding_events = lambda ids: True
    window._close_document(doc_id)
    assert not window._session.has_document(doc_id)


def test_a_closed_recording_takes_its_history_with_it(detected, qtbot):
    """Which is exactly why the close has to confirm: undo cannot reach past it."""
    window, doc_id, *_ = detected
    _curate(window)
    assert window._history.has_any()
    window._confirm_close_discarding_events = lambda ids: True
    window._close_document(doc_id)
    qtbot.wait(5)
    assert not window._history.has_any()
    assert not window._act_undo.isEnabled()


# ================================================================================================
# the stack's own contract: per-target, bounded, and silent during a restore
# ================================================================================================

def test_undo_is_offered_per_event_table_not_globally(window, tmp_path, qtbot):
    """Undoing a merge on a recording that is not on screen would repair a table the
    user cannot see. The action names the table it would act on instead."""
    sig_a, sig_b = _signal(), _signal(seed=7)
    doc_a, _ = _install(window, _doc(tmp_path, sig_a, name="a.dat"))
    _detect_into(window, doc_a, sig_a)
    _curate(window)
    assert window._act_undo.isEnabled()

    doc_b, _ = _install(window, _doc(tmp_path, sig_b, name="b.dat"))
    _detect_into(window, doc_b, sig_b)
    qtbot.wait(5)
    assert not window._act_undo.isEnabled()
    assert window._act_undo.text().endswith("on this event table")

    window._on_selection(doc_a, 0)
    assert window._act_undo.isEnabled()
    assert window._act_undo.text().startswith("Undo Curate")


def test_the_stack_is_bounded(detected, qtbot):
    window, *_ = detected
    df = window._analysis_panel.current_detection().payload.df
    col = df.columns.get_loc("accepted")
    key = window._undo_key()
    for i in range(MAX_DEPTH + 12):
        df.iat[i % len(df), col] = not bool(df.iat[i % len(df), col])
        window._history.checkpoint(key)
    assert window._history.depth() == MAX_DEPTH
    assert window._history.nbytes() > 0


def test_a_project_restore_creates_no_undo_steps(detected, qtbot):
    """A restore rebuilds every table from disk. An undo stack that offered to walk
    back through the reconstruction would offer to destroy the project just opened."""
    window, doc_id, channel, _events, signal = detected
    window._history.clear()
    window._restoring = True
    window._history.set_suspended(True)
    try:
        _detect_into(window, doc_id, signal, channel)   # a "restored" result landing
        qtbot.wait(5)
        assert not window._history.has_any()
    finally:
        window._restoring = False
        window._history.set_suspended(False)
    # ...and ordinary editing resumes recording immediately afterwards.
    _curate(window)
    assert window._history.has_any()


def test_checkpointing_twice_invents_no_step(detected, qtbot):
    """MainWindow wires the recorder to four partly-redundant signals on purpose. A
    checkpoint that finds nothing changed must push nothing, or every redundant
    connection would become a phantom undo step."""
    window, *_ = detected
    _curate(window)
    depth = window._history.depth()
    key = window._undo_key()
    for _ in range(5):
        window._history.checkpoint(key)
    assert window._history.depth() == depth


def test_undo_of_a_hand_added_event_removes_exactly_that_row(detected, qtbot):
    window, doc_id, channel, *_ = detected
    before = window._analysis_panel.current_detection().payload.df.copy(deep=True)
    window._analysis_panel.measure_panel.set_active(True)
    window._trace_view.set_measure_region(25_000 / FS, 25_400 / FS)
    new_id = window._analysis_panel.add_current_selection_as_event()
    qtbot.wait(5)
    assert new_id is not None
    live = window._analysis_panel.current_detection().payload.df
    assert len(live) == len(before) + 1
    assert window._act_undo.text() == "Undo Add event"

    window._undo_events()
    qtbot.wait(5)
    assert _differing_columns(
        before, window._analysis_panel.current_detection().payload.df) == []


# ================================================================================================
# label derivation — the labels are DERIVED from the diff, never declared
# ================================================================================================

def _tiny(n=4, *, accepted=True) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p",
            "start_idx": 100 * i, "end_idx": 100 * i + 50, "t_start_s": 0.002 * i,
            "dwell_s": 0.001, "baseline_pA": -100.0, "depth_pA": 20.0,
            "frac_depth": 0.2, "ecd_pA_s": 0.02, "max_dev_pA": 25.0,
            "mean_dev_pA": 20.0, "rise_s": 1e-5, "fall_s": 1e-5, "snr": 5.0,
            "n_levels": 1, "filter_limited": False, "accepted": accepted, "flag": "",
        })
    df = pd.DataFrame(rows).astype({c: d for c, (d, _, _) in SCHEMA.items()})
    return ensure_added_column(df)


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: d.iloc[:3].reset_index(drop=True), "Merge events"),
    (lambda d: d.iloc[:2].reset_index(drop=True), "Merge events (4 → 2)"),
    (lambda d: d.assign(accepted=[True, False, True, True]), "Curate event"),
    (lambda d: d.assign(accepted=False), "Reject all events"),
    (lambda d: d.assign(end_idx=[50, 199, 250, 350]), "Resize event mask"),
    (lambda d: d.assign(depth_pA=[20.0, 33.0, 20.0, 20.0]), "Re-measure depths"),
    (lambda d: d.assign(start_idx=[1, 2, 3, 4], end_idx=[9, 19, 29, 39]),
     "Re-run detection"),
])
def test_the_label_is_derived_from_what_actually_changed(mutate, expected):
    before = _tiny()
    assert describe_change(before, mutate(before.copy(deep=True))) == expected


def test_a_growing_table_with_untouched_rows_is_an_addition():
    before = _tiny()
    row = before.iloc[[0]].copy()
    row["event_id"] = 99
    after = pd.concat([before, row], ignore_index=True)
    assert describe_change(before, after) == "Add event"


def test_accept_all_and_reject_all_are_named_apart():
    before = _tiny(accepted=False)
    assert describe_change(before, before.assign(accepted=True)) == "Accept all events"


def test_a_re_measure_and_a_re_detection_that_land_on_the_same_spans_are_named_apart(
        window, tmp_path):
    """The one case the frame diff cannot settle on its own.

    A re-measure keeps every span and moves the numbers ("it re-measures, it does not
    re-detect, so the curated event set survives"); a re-detection that happens to find
    the same spans looks identical in the frame. The RESULT separates them: a re-measure
    is the only operation whose parameter change is confined to the ``baseline_*`` block.

    Verified end to end on the owner's ``_0006_`` recording as well — a real
    "Re-measure depths" at rough 2.0 / base 0.5 s came back labelled "Re-measure depths",
    and undo put back the frame, the ``baseline_fingerprint`` and the params.
    """
    from dataclasses import replace as dc_replace

    signal = _signal()
    doc_id, channel = _install(window, _doc(tmp_path, signal, name="stamp.dat"))
    events = _detect_into(window, doc_id, signal, channel)
    base = window._session.results.get(doc_id, channel, "detect.schmitt")
    base = dc_replace(base, params={"threshold_sigma": 5.0, "baseline_rough_s": 0.5},
                      baseline_fingerprint="sha256:aaaa")
    window._session.add_result(base)
    key = window._undo_key()

    moved = events.df.copy(deep=True)
    moved["depth_pA"] = moved["depth_pA"].to_numpy() + 1.0
    window._session.add_result(dc_replace(
        base, payload=EventSet(moved, dict(events.provenance or {})),
        params={"threshold_sigma": 5.0, "baseline_rough_s": 2.0},
        baseline_fingerprint="sha256:bbbb"))
    assert window._history.undo_label(key) == "Re-measure depths"

    again = moved.copy(deep=True)
    again["depth_pA"] = again["depth_pA"].to_numpy() + 1.0
    window._session.add_result(dc_replace(
        base, payload=EventSet(again, dict(events.provenance or {})),
        params={"threshold_sigma": 6.0, "baseline_rough_s": 2.0},
        baseline_fingerprint="sha256:cccc"))
    assert window._history.undo_label(key) == "Re-run detection"


def test_a_table_holding_nan_and_na_compares_equal_to_itself():
    """A rejected ``non_finite`` row keeps NaN in two columns, and a flag cell can be
    ``pd.NA``. Under a plain ``==`` every one of those reads as a change, so a table
    holding a single NaN would push a phantom undo step on every checkpoint — on every
    redundant signal, forever. The comparator has to be NaN-tolerant or the stack fills
    with steps that undo nothing."""
    from nanoscope.core.history import _frames_equal

    df = _tiny()
    df.loc[1, "depth_pA"] = np.nan
    df.loc[1, "ecd_pA_s"] = np.inf
    df["flag"] = pd.array(["", pd.NA, "operator_note", ""], dtype="string")
    assert _frames_equal(df, df.copy(deep=True))
    moved = df.copy(deep=True)
    moved.loc[2, "depth_pA"] = 21.0
    assert not _frames_equal(df, moved)


def test_the_undo_shortcut_does_not_steal_ctrl_z_from_a_text_field(window, qtbot):
    """Adding a global Ctrl+Z is how an application breaks typing in its own forms.

    The detection panel is full of spin boxes and line edits, and undoing the EVENT
    TABLE when the user meant to undo three characters of a threshold would be a new
    data-loss path introduced by the fix for one. Qt's ShortcutOverride is what prevents
    it, so the property is pinned rather than assumed."""
    from PySide6 import QtCore, QtGui, QtWidgets

    field = QtWidgets.QLineEdit(window)
    qtbot.addWidget(field)
    field.insert("hello")
    seq = QtGui.QKeySequence(QtGui.QKeySequence.Undo)
    event = QtGui.QKeyEvent(QtCore.QEvent.ShortcutOverride,
                            QtCore.Qt.Key(seq[0].key()), seq[0].keyboardModifiers())
    field.event(event)
    assert event.isAccepted(), (
        "the focused text field must claim the undo sequence before the menu action")
    assert window._act_undo.shortcutContext() == QtCore.Qt.WindowShortcut


# ================================================================================================
# the structural guards — one door, one wirer
# ================================================================================================
def _relative_imports(module) -> list[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    return [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]


def test_the_history_module_imports_no_gui():
    """The recorder is session state, not a panel. If it could import a panel, a panel
    could import it back — and the next mutation site would push its own undo step
    instead of passing the one door."""
    import nanoscope.core.history as hist

    for mod in _relative_imports(hist):
        assert "gui" not in mod.split("."), f"core/history.py must not import gui: {mod}"


def test_main_window_is_the_only_gui_module_that_touches_the_history():
    """MainWindow is the sole wirer, for undo exactly as for everything else. It is the
    one object that already receives all four mutation announcements, so nothing else
    needs to know the stack exists — and a panel that talked to it directly would be
    the second recorder this design exists to prevent."""
    gui_dir = pathlib.Path(nanoscope.__file__).parent / "gui"
    offenders = []
    for path in sorted(gui_dir.glob("*.py")):
        if path.name == "main_window.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "history" in (node.module or ""):
                offenders.append(f"{path.name}: from {node.module} import ...")
            if isinstance(node, ast.Attribute) and node.attr in (
                    "checkpoint", "checkpoint_all"):
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr == "_history":
                    offenders.append(f"{path.name}: ...._history.{node.attr}(...)")
    assert offenders == [], offenders


def test_every_eventset_mutation_announcement_reaches_the_recorder():
    """The four signals an EventSet mutation can arrive on, and the fact that MainWindow
    connects every one of them to a checkpoint. This is the guard that would have caught
    ``eventEdited`` reaching nothing: it was connected to ``refresh_after_edit`` alone.
    """
    path = pathlib.Path(nanoscope.__file__).parent / "gui" / "main_window.py"
    tree = ast.parse(path.read_text())
    reached: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "connect"):
            continue
        emitter = node.func.value
        if not isinstance(emitter, ast.Attribute):
            continue
        # The slot may be a bound method, a lambda, or a lambda wrapping a call — walk
        # the whole argument subtree rather than matching a line of source.
        names = {n.attr for a in node.args for n in ast.walk(a)
                 if isinstance(n, ast.Attribute)}
        if any(name.startswith("_checkpoint") for name in names):
            reached.add(emitter.attr)
    for signal in ("resultsChanged", "eventAcceptanceChanged",
                   "curationBatchChanged", "eventEdited"):
        assert signal in reached, f"{signal} does not reach the undo recorder"


# ================================================================================================
# THE RECONCILIATION PASS — four defects the tier's own verification found in it
#
# All four were measured on the shipped tier, through a real MainWindow, before this
# section existed. They are grouped here because they are one shape: the history was
# right about the ROWS and wrong about everything it derived around them.
# ================================================================================================

def _second_recording(window, tmp_path, *, name="hidden.dat", seed=3):
    """A recording that is OPEN but not the one the panel is showing — what every
    recording except one is during a batch detect."""
    signal = _signal(seed=seed)
    doc_id, channel = _install(window, _doc(tmp_path, signal, name=name))
    return doc_id, channel, signal


def test_a_detection_the_panel_never_showed_invents_no_undo_steps(window, tmp_path, qtbot):
    """MEASURED BEFORE (four recordings, Analysis ▸ Batch detect): three of the four live
    tables had no ``added`` column, because it is materialised only when a detection is
    DISPLAYED, while every snapshot has it. ``_frames_equal`` compared column LISTS, so
    each ``checkpoint_all()`` reported three untouched tables as mutated and pushed three
    phantom "Edit events" steps — measured: depth 0 → 3 → 6 → 9 on three consecutive
    calls, with ``MAX_DEPTH`` then evicting the user's real curation.
    """
    hidden_id, hidden_ch, hidden_signal = _second_recording(window, tmp_path)
    # A second recording, selected LAST, so it is the one the panel displays. The first
    # then receives its detection the way every recording but one does in a batch run:
    # into the session, never through _show_result.
    shown_signal = _signal(seed=4)
    shown_id, shown_ch = _install(window, _doc(tmp_path, shown_signal, name="shown.dat"))
    _detect_into(window, shown_id, shown_signal, shown_ch)
    _detect_into(window, hidden_id, hidden_signal, hidden_ch)
    qtbot.wait(10)

    hidden = window._session.results.get(hidden_id, hidden_ch, "detect.schmitt")
    assert ADDED_COLUMN not in hidden.payload.df.columns, (
        "the premise of this test: a table the panel never showed has no 'added' column")

    window._history.checkpoint_all()          # establish the pristine states
    depth = window._history.depth()
    for _ in range(3):
        assert window._history.checkpoint_all() is False
    assert window._history.depth() == depth


def test_opening_the_edit_menu_does_not_re_dirty_a_saved_project(window, tmp_path, qtbot):
    """The second consequence of the same defect, aimed at the one dialog that protects
    unsaved work: ``history.changed`` drives ``_mark_dirty``, so a phantom step made a
    just-saved session dirty again. Measured: dirty False → True and the title regrew its
    ``•`` on the first Edit-menu open, with nothing changed."""
    hidden_id, hidden_ch, hidden_signal = _second_recording(window, tmp_path)
    shown_signal = _signal(seed=4)
    shown_id, shown_ch = _install(window, _doc(tmp_path, shown_signal, name="shown.dat"))
    _detect_into(window, shown_id, shown_signal, shown_ch)
    _detect_into(window, hidden_id, hidden_signal, hidden_ch)
    qtbot.wait(10)
    assert window._write_project(str(tmp_path / "p.nanoscope"))
    assert window._dirty is False
    title = window.windowTitle()

    for _ in range(3):
        window._on_edit_menu_about_to_show()
    assert window._dirty is False
    assert window.windowTitle() == title


def test_pressing_detect_again_with_the_same_settings_is_named_a_re_detection(
        window, tmp_path, qtbot):
    """MEASURED BEFORE: a second Detect with UNCHANGED settings returns a byte-identical
    table whose accept/reject column has been reset. ``describe_change`` sees only the two
    frames, so it called that "Curate 4 events" — Edit ▸ Undo offered to undo the opposite
    of what happened, and the replacement warning (which is gated on the label) never
    appeared: status bar ``''``, dialogs ``[]``. That is C14's whole symptom, back on the
    commonest path.

    The discriminator is the RESULT, not the frame: a re-detection installs a NEW EventSet
    object and carries none of the previous table's hand curation, while a merge, a resize
    and a hand-added event all rebuild rows and inherit it (``inherited_curation``).
    """
    signal = _signal()
    doc_id, channel = _install(window, _doc(tmp_path, signal, name="again.dat"))
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(30)
    n_first = len(panel.current_detection().payload.df)
    curated = _curate(window)
    window.statusBar().clearMessage()

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()             # SAME settings — no parameter changed
    qtbot.wait(30)

    assert len(panel.current_detection().payload.df) == n_first, (
        "the premise: unchanged settings give the same rows, so only the curation differs")
    assert window._act_undo.text() == "Undo Re-run detection"
    msg = window.statusBar().currentMessage()
    assert "Detection replaced the previous table" in msg and "Undo" in msg

    window._undo_events()
    qtbot.wait(10)
    assert _differing_columns(curated, panel.current_detection().payload.df) == []


def test_the_replacement_warning_counts_only_the_curation_a_person_made(detected, qtbot):
    """An automatic rejection is RE-DERIVED by the next run (that is the frozen
    ``inherited_curation`` rule), so counting it beside a hand rejection overstates what a
    re-detection destroys — and a warning that overstates is one the user learns to click
    past. Measured before: "5 rejected" where 4 were the user's and 1 was
    ``clipped_by_boundary``."""
    from nanoscope.analysis.events import FLAG_CLIPPED

    window, *_ = detected
    df = window._analysis_panel.current_detection().payload.df
    ensure_added_column(df)
    df.iat[0, df.columns.get_loc("accepted")] = False          # the detector's own reason
    df.iat[0, df.columns.get_loc("flag")] = FLAG_CLIPPED
    df.iat[2, df.columns.get_loc("accepted")] = False          # a person's
    window._on_edit_menu_about_to_show()                       # commit that state

    seen: list[tuple[str, dict]] = []
    window._history.stepRecorded.connect(lambda label, s: seen.append((label, dict(s))))
    # Anything at all now, so the step's summary describes the table above — which is what
    # a replacement warning quotes: what the table it replaced HELD.
    df.iat[3, df.columns.get_loc("accepted")] = False
    window._on_edit_menu_about_to_show()
    assert seen, "the in-place edit must still be committed"
    summary = seen[-1][1]
    assert summary["n_rejected"] == 2                          # what the table holds
    assert summary["n_hand_rejected"] == 1                     # what a re-run would destroy


def test_an_unannounced_edit_reaches_the_quit_prompt_and_the_project_save(
        detected, tmp_path, qtbot, monkeypatch):
    """``core.history`` documents that a forgotten announcement "still lands on the stack
    the next time anything asks (an undo, a project save, a quit)". MEASURED BEFORE:
    ``checkpoint_all()`` had exactly two callers and neither was a save or a quit, so an
    edit through the one writer that announces nothing left ``dirty=False``, the title
    without its ``•``, and ``closeEvent`` asking nothing — the edit went silently."""
    from PySide6 import QtGui

    window, *_ = detected
    assert window._write_project(str(tmp_path / "q.nanoscope"))
    assert window._dirty is False
    df = window._analysis_panel.current_detection().payload.df
    df.iat[0, df.columns.get_loc("flag")] = "operator_note"     # emits nothing at all

    from nanoscope.gui import main_window as mw_module

    asked: list[str] = []
    monkeypatch.setattr(type(window), "_confirm_close_choice",
                        lambda self: (asked.append("asked"), mw_module._CLOSE_CANCEL)[1])
    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert asked == ["asked"], "the quit prompt must be shown for an unannounced edit"
    assert not event.isAccepted()
    assert window._history.can_undo(window._undo_key())

    # ...and a save commits it too, so the file and the stack agree about what happened.
    window._history.undo(window._undo_key())
    df = window._analysis_panel.current_detection().payload.df
    df.iat[1, df.columns.get_loc("flag")] = "operator_note"
    depth = window._history.depth()
    assert window._write_project(str(tmp_path / "q.nanoscope"))
    assert window._history.depth() == depth + 1


def test_a_snapshot_shares_no_mutable_provenance_with_the_live_result(detected, qtbot):
    """The frame is deep-copied because the Inspector writes it in place; the provenance
    was not, so ``exclude_regions``/``polarity_evidence``/``baseline_warnings`` were the
    SAME objects in the snapshot and in the live result. An undo would then put the rows
    back under a provenance that had moved on."""
    window, *_ = detected
    window._history.checkpoint_all()
    key = window._undo_key()
    live = window._session.results.get(*key)
    pristine = window._history._pristine[key]
    shared = [k for k, v in live.payload.provenance.items()
              if isinstance(v, (list, dict, set)) and pristine.payload.provenance.get(k) is v]
    assert shared == [], f"provenance values shared with the live result: {shared}"
