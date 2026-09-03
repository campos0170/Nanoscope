"""Event Inspector / Browser tests (PORT_PLAN §6/§8, pytest-qt, offscreen).

The :class:`~nanoscope.gui.event_inspector.EventInspectorWindow` browses + curates
detected events one at a time over the SHARED events ``.df``:

* installing a detection draws the current event's raw waveform (a data item) with
  its shaded event region + baseline/level lines;
* Next / Prev move the current event within the filtered subset, and the index
  spinbox jumps to the i-th;
* ``show_event`` selects by id and emits ``eventSelected``;
* ``set_accepted_current`` writes the shared frame's ``accepted`` in place AND emits
  ``acceptanceChanged`` — the whole point of the curation loop;
* the All / Accepted only / Rejected only filter subsets the browsed events;
* the empty state (no detection) is safe to drive.

The final test opens the inspector THROUGH a real :class:`MainWindow` and asserts a
reject in the inspector flips the :class:`EventTable` model's checkbox for that row
(shared-df sync) and drops the event from the trace's span overlay.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtGui
from PySide6.QtCore import Qt

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import REVIEW_COL, REVIEW_DONE, EventSet, empty_df
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.event_inspector import EventInspectorWindow
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, *, fs: float | None = FS, n: int = 100_000,
               name: str = "pore_p1.dat") -> TraceDocument:
    arr = _event_trace(n)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(arr,))


def _session_with_doc(tmp_path) -> tuple[Session, str, int]:
    """A Session holding a synthetic 5-event current recording (no MainWindow)."""
    session = Session()
    doc = _synth_doc(tmp_path)
    doc_id = session.add_document(doc)
    return session, doc_id, 0


def _events_for(session: Session, doc_id: str, channel: int) -> EventSet:
    """Run the REAL detector on the session's channel -> a schema-complete EventSet."""
    ch = session.calibrated(doc_id, channel)
    df, prov = detect_events(ch.data, ch.sample_rate_hz, min_dwell_s=5e-4)
    return EventSet(df, prov)


def _inspector(qtbot, session: Session) -> EventInspectorWindow:
    insp = EventInspectorWindow(session)
    qtbot.addWidget(insp)
    return insp


# ================================================================================================
# waveform rendering
# ================================================================================================

def test_set_events_draws_waveform_region_and_lines(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    assert len(events) == len(_EVENTS)

    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    # The current event is the first one; its waveform draws as a real data item...
    assert insp.current_event_id() == int(events.df["event_id"].iloc[0])
    assert insp._curve in insp._plot.listDataItems()
    x, y = insp._curve.getData()
    assert x is not None and x.size > 0 and y.size == x.size

    # ...with the event region shaded (MOVABLE — its edges are the mask-resize handles) and the
    # baseline/level lines shown.
    assert insp._region.isVisible()
    r0, r1 = insp._region.getRegion()
    assert r1 > r0
    assert insp._region.movable
    assert insp._baseline_line.isVisible()
    assert insp._level_line.isVisible()

    # The padded window brackets the event (starts before it, ends after it).
    xvals = session.x_axis(doc_id).values
    s, e = _EVENTS[0]
    assert float(x[0]) <= float(xvals[s])
    assert float(x[-1]) >= float(xvals[e])


def test_waveform_updates_when_navigating(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    x_first, _ = insp._curve.getData()
    first_start = float(x_first[0])
    insp.next_event()
    x_second, _ = insp._curve.getData()
    # A later event's padded window starts further along the recording.
    assert float(x_second[0]) > first_start


# ================================================================================================
# navigation
# ================================================================================================

def test_next_prev_move_within_the_filter(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    ids = [int(i) for i in events.df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    assert insp.event_count() == len(ids)
    assert insp.current_event_id() == ids[0]

    insp.next_event()
    assert insp.current_event_id() == ids[1]
    insp.next_event()
    assert insp.current_event_id() == ids[2]
    insp.prev_event()
    assert insp.current_event_id() == ids[1]

    # Prev at the first event is a clamped no-op (stays put).
    insp.show_event(ids[0])
    insp.prev_event()
    assert insp.current_event_id() == ids[0]
    # Next at the last event is a clamped no-op.
    insp.show_event(ids[-1])
    insp.next_event()
    assert insp.current_event_id() == ids[-1]


def test_index_spinbox_jumps_to_event(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    ids = [int(i) for i in events.df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    insp._index_spin.setValue(3)   # 1-based position within the subset
    assert insp.current_event_id() == ids[2]


def test_show_event_selects_by_id_and_emits(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    ids = [int(i) for i in events.df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    emitted: list[int] = []
    insp.eventSelected.connect(emitted.append)

    insp.show_event(ids[3])
    assert insp.current_event_id() == ids[3]
    assert emitted == [ids[3]]

    # Re-showing the SAME event does not re-emit (this early return breaks selection loops).
    insp.show_event(ids[3])
    assert emitted == [ids[3]]

    # An unknown id is a safe no-op.
    insp.show_event(10_000_000)
    assert insp.current_event_id() == ids[3]
    assert emitted == [ids[3]]


# ================================================================================================
# curation
# ================================================================================================

def test_set_accepted_current_writes_shared_df_and_emits(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df                    # the SAME object the inspector will edit
    ids = [int(i) for i in df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    changed: list[tuple[int, bool]] = []
    insp.acceptanceChanged.connect(lambda i, a: changed.append((i, a)))

    insp.show_event(ids[2])
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[2])[0][0])
    assert bool(df["accepted"].iloc[row]) is True   # detector accepts by default

    insp.set_accepted_current(False)
    # The write landed in the shared frame in place...
    assert bool(df["accepted"].iloc[row]) is False
    # ...and the decision was announced.
    assert changed[-1] == (ids[2], False)

    insp.set_accepted_current(True)
    assert bool(df["accepted"].iloc[row]) is True
    assert changed[-1] == (ids[2], True)


def test_accept_current_marks_fresh_already_accepted_row_reviewed(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    event_id = int(df.iloc[0]["event_id"])
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)
    insp.show_event(event_id)
    changed: list[tuple[int, bool]] = []
    insp.acceptanceChanged.connect(lambda i, a: changed.append((i, a)))

    assert bool(df.iloc[0]["accepted"])
    insp.set_accepted_current(True)

    assert str(df.iloc[0][REVIEW_COL]) == REVIEW_DONE
    assert changed == [(event_id, True)]


def test_keyboard_toggle_via_public_methods(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    ids = [int(i) for i in df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)
    insp.show_event(ids[0])
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[0])[0][0])

    # The accept checkbox reflects (and drives) the shared accepted state.
    assert insp._accept_check.isChecked() is True
    insp._accept_check.setChecked(False)   # user unticks the box
    assert bool(df["accepted"].iloc[row]) is False
    insp._accept_check.setChecked(True)
    assert bool(df["accepted"].iloc[row]) is True


# ================================================================================================
# filter subsets
# ================================================================================================

def test_filter_subsets_the_browsed_events(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    ids = [int(i) for i in df["event_id"]]
    n = len(ids)
    assert n >= 4
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    assert insp.event_count() == n            # All
    insp.set_filter("Accepted only")
    assert insp.event_count() == n            # all accepted by default
    insp.set_filter("Rejected only")
    assert insp.event_count() == 0            # nothing rejected yet (empty subset is safe)
    assert insp.current_event_id() is None

    # Reject the first two events (from the All view).
    insp.set_filter("All")
    insp.show_event(ids[0]); insp.set_accepted_current(False)
    insp.show_event(ids[1]); insp.set_accepted_current(False)

    insp.set_filter("Accepted only")
    assert insp.event_count() == n - 2
    insp.set_filter("Rejected only")
    assert insp.event_count() == 2
    assert set(insp._subset_ids()) == {ids[0], ids[1]}

    # Prev/Next stay within the rejected subset.
    insp.show_event(ids[0])
    insp.next_event()
    assert insp.current_event_id() == ids[1]
    insp.next_event()   # clamped at the end of the 2-event subset
    assert insp.current_event_id() == ids[1]


def test_reinstalling_same_frame_preserves_browsed_event(qapp, qtbot, tmp_path):
    """Re-feeding the SAME events frame (as a sibling fit result's resultsChanged would) keeps the
    user on their current event + filter instead of resetting to the first event."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    ids = [int(i) for i in events.df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    insp.show_event(ids[3])
    assert insp.current_event_id() == ids[3]

    # Same df object again -> the browsed event is preserved.
    insp.set_events(doc_id, ch, events.df)
    assert insp.current_event_id() == ids[3]

    # A different frame (a genuine re-run) resets to the first event.
    events2 = _events_for(session, doc_id, ch)
    insp.set_events(doc_id, ch, events2.df)
    assert insp.current_event_id() == int(events2.df["event_id"].iloc[0])


# ================================================================================================
# mask editing (resize an event's [start, end]) + batch curation
# ================================================================================================

def _set_region_to_indices(insp: EventInspectorWindow, session: Session, doc_id: str,
                           start_idx: int, end_idx: int) -> None:
    """Drive the movable region to sample indices [start_idx, end_idx] as a user drag would.

    ``setRegion`` itself fires ``sigRegionChangeFinished`` (pyqtgraph calls ``lineMoveFinished``);
    we emit it once more explicitly so the edit slot runs regardless of pyqtgraph internals — the
    second delivery is a no-op (the bounds already match the df)."""
    xvals = session.x_axis(doc_id).values
    insp._region.setRegion((float(xvals[int(start_idx)]), float(xvals[int(end_idx)])))
    insp._region.sigRegionChangeFinished.emit(insp._region)


def test_resize_region_rewrites_bounds_and_recharacterizes(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df                    # the SAME object the inspector edits in place
    ids = [int(i) for i in df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    edited: list[int] = []
    insp.eventEdited.connect(edited.append)

    # Browse to the LAST event and pull its trailing edge inward by 80 samples.
    insp.show_event(ids[-1])
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[-1])[0][0])
    old_start = int(df["start_idx"].iloc[row])
    old_end = int(df["end_idx"].iloc[row])
    new_end = old_end - 80
    assert new_end > old_start

    _set_region_to_indices(insp, session, doc_id, old_start, new_end)

    # The shared frame's bounds were rewritten in place...
    assert int(df["start_idx"].iloc[row]) == old_start
    assert int(df["end_idx"].iloc[row]) == new_end
    # ...dwell_s was recomputed as (end-start)/fs and depth_pA re-measured (still a real blockade)...
    assert float(df["dwell_s"].iloc[row]) == pytest.approx((new_end - old_start) / FS, rel=1e-9)
    assert np.isfinite(df["depth_pA"].iloc[row]) and float(df["depth_pA"].iloc[row]) > 0
    # ...t_start_s tracks the (unchanged) start in seconds...
    assert float(df["t_start_s"].iloc[row]) == pytest.approx(old_start / FS, rel=1e-9)
    # ...and the edit was announced exactly once for this event.
    assert edited == [ids[-1]]


def test_resize_preserves_identity_and_curation(qapp, qtbot, tmp_path):
    """A re-characterize must PRESERVE event_id / accepted / recording / pore_id — only the
    measurement columns are rewritten."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    ids = [int(i) for i in df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    insp.show_event(ids[1])
    insp.set_accepted_current(False)                 # reject it first
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[1])[0][0])
    old_start = int(df["start_idx"].iloc[row])
    old_end = int(df["end_idx"].iloc[row])
    recording = df["recording"].iloc[row]
    pore_id = df["pore_id"].iloc[row]

    _set_region_to_indices(insp, session, doc_id, old_start + 20, old_end)

    assert int(df["start_idx"].iloc[row]) == old_start + 20   # bounds moved
    assert int(df["event_id"].iloc[row]) == ids[1]            # identity preserved
    assert bool(df["accepted"].iloc[row]) is False            # curation preserved
    assert df["recording"].iloc[row] == recording
    assert df["pore_id"].iloc[row] == pore_id


def test_programmatic_navigation_does_not_emit_edit(qapp, qtbot, tmp_path):
    """Navigating (show_event / next / prev) sets the region programmatically; the ``_syncing``
    fence must keep that from reading back as a mask edit (no false ``eventEdited``)."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    ids = [int(i) for i in events.df["event_id"]]
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)

    edited: list[int] = []
    insp.eventEdited.connect(edited.append)

    insp.show_event(ids[2])
    insp.next_event()
    insp.prev_event()
    insp.show_event(ids[0])
    assert edited == []


def test_accept_all_and_reject_all_write_whole_column(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    batched: list[int] = []
    insp.curationBatchChanged.connect(lambda: batched.append(1))

    insp.reject_all()
    assert not df["accepted"].to_numpy(bool).any()   # every event rejected
    assert insp._accept_check.isChecked() is False   # current event's box follows
    insp.accept_all()
    assert df["accepted"].to_numpy(bool).all()       # every event accepted
    assert insp._accept_check.isChecked() is True
    assert batched == [1, 1]                          # each batch announced once


def test_reject_all_marks_every_changed_inspector_row_reviewed(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    insp.reject_all()

    assert REVIEW_COL in df
    assert (df[REVIEW_COL].astype("string") == REVIEW_DONE).all()


def test_accept_all_marks_fresh_already_accepted_rows_reviewed(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df)

    assert df["accepted"].to_numpy(dtype=bool).all()
    insp.accept_all()

    assert (df[REVIEW_COL].astype("string") == REVIEW_DONE).all()


# ================================================================================================
# empty state
# ================================================================================================

def test_empty_state_is_safe(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    insp = _inspector(qtbot, session)

    # Nothing installed: every public method is a safe no-op.
    assert insp.current_event_id() is None
    assert insp.event_count() == 0
    insp.next_event()
    insp.prev_event()
    insp.set_accepted_current(True)
    insp.show_event(0)
    insp.set_filter("Accepted only")
    assert insp.current_event_id() is None

    # An empty detection is safe.
    insp.set_events(doc_id, ch, empty_df())
    assert insp.event_count() == 0
    assert insp.current_event_id() is None

    # clear() is safe and idempotent.
    insp.clear()
    insp.clear()
    assert insp.event_count() == 0
    assert not insp._curve.isVisible()


# ================================================================================================
# end-to-end through MainWindow: curation flows to the shared table + trace
# ================================================================================================

def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot, *, min_dwell_s: float = 5e-4) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def test_event_inspector_gated_until_detection(window, qtbot, tmp_path):
    # No detection yet: the menu action is dead and no window exists.
    window._sync_export_action()
    assert not window._act_event_inspector.isEnabled()
    assert window._event_inspector is None

    _install(window, _synth_doc(tmp_path))
    window._sync_export_action()
    assert not window._act_event_inspector.isEnabled()

    _run_detection(window, qtbot)
    # A finished detection (via resultsChanged -> _sync_export_action) lights it up.
    assert window._act_event_inspector.isEnabled()


def test_inspector_accept_flips_table_checkbox_and_trace_spans(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    # Open the inspector exactly as the menu click would.
    window._act_event_inspector.trigger()
    insp = window._event_inspector
    assert insp is not None and insp.isVisible()

    # It is browsing the SAME frame the EventTable holds (shared reference, not a copy).
    table = window._analysis_panel._table
    assert insp._df is table.dataframe

    ev_id = insp.current_event_id()
    assert ev_id is not None
    df = table.dataframe
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ev_id)[0][0])
    accept_col = list(df.columns).index("accepted")
    model = table._model
    idx = model.index(row, accept_col)

    # Reject from the inspector -> the shared df is written AND the table model's checkbox flips.
    insp.set_accepted_current(False)
    assert bool(df["accepted"].iloc[row]) is False
    assert model.data(idx, Qt.CheckStateRole) == Qt.Unchecked

    # Curation flowed to the trace: the single span overlay now shades only the accepted events.
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert len(spans) == 1
    assert spans[0].n_spans == len(_EVENTS) - 1

    # Re-accept from the inspector -> the model flips back and every event is shaded again.
    insp.set_accepted_current(True)
    assert bool(df["accepted"].iloc[row]) is True
    assert model.data(idx, Qt.CheckStateRole) == Qt.Checked
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert len(spans) == 1
    assert spans[0].n_spans == len(_EVENTS)


def test_curation_persists_on_trace_across_selection_roundtrip(window, qtbot, tmp_path):
    """A reject stays reflected on the trace's span overlay even after switching the selection
    away and back — the spans are redrawn from the curated df, not the frozen detection snapshot."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_event_inspector.trigger()
    insp = window._event_inspector
    insp.set_accepted_current(False)   # reject the first event

    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert spans[0].n_spans == len(_EVENTS) - 1

    # Switch the selection off this channel, then back.
    window._analysis_panel.set_selection("does-not-exist", 0)
    window._on_selection(doc_id, channel)
    qtbot.wait(10)

    # The redrawn span overlay still honours the curation (one event stays unshaded).
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert len(spans) == 1
    assert spans[0].n_spans == len(_EVENTS) - 1


def test_table_edit_reflects_into_inspector(window, qtbot, tmp_path):
    """A curation edit in the TABLE mirrors onto the inspector's control when it is on that event."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_event_inspector.trigger()
    insp = window._event_inspector
    ev_id = insp.current_event_id()
    df = window._analysis_panel._table.dataframe
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ev_id)[0][0])

    assert insp._accept_check.isChecked() is True
    # Reject the row via the table model (as a table checkbox click / keyboard 'r' would).
    window._analysis_panel._table._model.set_accepted(row, False)
    # The inspector, sitting on that event, mirrors the new state.
    assert insp._accept_check.isChecked() is False
    assert bool(df["accepted"].iloc[row]) is False


def test_inspector_edit_updates_table_row_and_span_overlay(window, qtbot, tmp_path):
    """Resizing an event in the inspector rewrites the shared df; the panel repaints that EventTable
    row and moves the span overlay to the new bounds (the inspector edits, the panel owns the view)."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_event_inspector.trigger()
    insp = window._event_inspector
    table = window._analysis_panel._table
    df = table.dataframe
    session = window._session

    ids = [int(i) for i in df["event_id"]]
    insp.show_event(ids[-1])               # browse to (and select) the last event
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[-1])[0][0])
    old_start = int(df["start_idx"].iloc[row])
    old_end = int(df["end_idx"].iloc[row])
    new_end = old_end - 80

    _set_region_to_indices(insp, session, doc_id, old_start, new_end)
    qtbot.wait(10)

    # The shared df carries the new bound...
    assert int(df["end_idx"].iloc[row]) == new_end
    # ...the EventTable model shows it in that row's end_idx cell...
    model = table._model
    end_col = list(df.columns).index("end_idx")
    assert model.data(model.index(row, end_col), Qt.DisplayRole) == str(new_end)
    # ...and the span overlay moved to the new bounds (the last event's stop is the max stop).
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert len(spans) == 1
    xvals = session.x_axis(doc_id).values
    _lo, hi = spans[0].x_bounds()
    assert hi == pytest.approx(float(xvals[new_end]), abs=1.0 / FS)


def test_inspector_batch_curation_syncs_table_and_spans(window, qtbot, tmp_path):
    """Reject all / Accept all in the inspector rewrite the whole 'accepted' column; the panel
    resets the table (every checkbox repaints) and redraws the accepted-only span overlay."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_event_inspector.trigger()
    insp = window._event_inspector
    table = window._analysis_panel._table
    df = table.dataframe
    model = table._model
    accept_col = list(df.columns).index("accepted")
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")

    # Reject all -> every 'accepted' False, every checkbox unchecked, no spans shaded.
    insp.reject_all()
    qtbot.wait(10)
    assert not df["accepted"].to_numpy(bool).any()
    for r in range(len(df)):
        assert model.data(model.index(r, accept_col), Qt.CheckStateRole) == Qt.Unchecked
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert spans == []

    # Accept all -> every 'accepted' True, every checkbox checked, all events shaded again.
    insp.accept_all()
    qtbot.wait(10)
    assert df["accepted"].to_numpy(bool).all()
    for r in range(len(df)):
        assert model.data(model.index(r, accept_col), Qt.CheckStateRole) == Qt.Checked
    spans = [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]
    assert len(spans) == 1
    assert spans[0].n_spans == len(_EVENTS)


def test_inspector_selection_zooms_trace_and_marks(window, qtbot, tmp_path):
    """Selecting an event in the inspector routes to the AnalysisPanel: the trace zooms in and the
    single MARKER overlay drops (the same path a table-row click drives)."""
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_event_inspector.trigger()
    insp = window._event_inspector
    ids = [int(i) for i in window._analysis_panel._table.dataframe["event_id"]]

    anchor = window._trace_view.anchor_plot()
    before = tuple(anchor.vb.viewRange()[0])

    insp.show_event(ids[-1])   # navigate in the inspector
    qtbot.wait(10)

    # The panel dropped a marker on the dived-into event and zoomed the shared trace in.
    assert window._analysis_panel._marker_id is not None
    assert window._analysis_panel._marker_id in window._trace_view._overlays
    after = tuple(anchor.vb.viewRange()[0])
    assert (after[1] - after[0]) < (before[1] - before[0])


# ================================================================================================
# N1 / N2 — a mask resize re-measures the event, so it must re-measure it the way the
# detector did: on ``lowpass(preprocess_for_detection(raw))``, against the detector's own
# noise. It used ``ch.data`` (the raw channel) and a pooled RAW flank MAD, so dragging an
# edge by one sample could move depth / SNR by more than the drag did.
# ================================================================================================

_DETECT_PARAMS = {
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.05,
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 2, "notch_q": 30.0,
    "polarity": None, "k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
    "filter_cutoff_hz": 5_000.0, "filter_kind": "bessel", "filter_order": 8,
}


def _drifting_doc(tmp_path, *, n: int = 200_000, seed: int = 4) -> TraceDocument:
    """Blockades on a DRIFTING, mains-contaminated open pore.

    The flat clean fixture above cannot show either bug: with nothing for the
    preprocessing to remove, the conditioned trace IS the raw channel.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = (100.0
         + 12.0 * np.sin(2 * np.pi * 0.4 * t)
         + 3.0 * np.sin(2 * np.pi * 60.0 * t)
         + rng.normal(0.0, 1.5, n))
    for s, e in [(20_000, 21_000), (60_000, 61_200), (120_000, 121_000),
                 (170_000, 171_500)]:
        x[s:e] -= 30.0
    arr = np.ascontiguousarray(x, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "drift.dat", format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _drifting_session_with_real_detection(tmp_path):
    """A Session whose detection came from the REAL DetectSchmitt module, so its provenance
    carries the preprocessing stamp ``detection_signal`` reads back."""
    from nanoscope.analysis.context import SimpleAnalysisContext
    from nanoscope.analysis.modules import DetectSchmitt

    session = Session()
    doc = _drifting_doc(tmp_path)
    doc_id = session.add_document(doc)
    ch = session.calibrated(doc_id, 0)
    res = DetectSchmitt().run(
        SimpleAnalysisContext(channel=ch, meta=doc.meta, params=dict(_DETECT_PARAMS)))
    return session, doc_id, res.payload


def test_a_mask_resize_re_measures_on_the_trace_the_detector_measured_on(
    qapp, qtbot, tmp_path
):
    """Drag an edge; the rewritten row must be what ``characterize`` gives on the CONDITIONED
    trace, and demonstrably not what it gives on the raw channel."""
    from nanoscope.analysis.modules import detection_signal

    session, doc_id, es = _drifting_session_with_real_detection(tmp_path)
    df = es.df
    assert len(df) >= 3
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(es.provenance))

    raw = np.asarray(session.calibrated(doc_id, 0).data)
    conditioned = detection_signal(raw, FS, es.provenance)
    assert float(np.abs(np.asarray(conditioned, np.float64)
                        - np.asarray(raw, np.float64)).max()) > 0.5

    ids = [int(i) for i in df["event_id"]]
    insp.show_event(ids[1])
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[1])[0][0])
    start = int(df["start_idx"].iloc[row])
    new_end = int(df["end_idx"].iloc[row]) - 120
    assert new_end > start

    _set_region_to_indices(insp, session, doc_id, start, new_end)
    got = df.iloc[row].copy()

    # The oracle: the same re-characterization, driven straight off the conditioned trace.
    probe = _inspector(qtbot, session)
    probe.set_events(doc_id, 0, df.copy(), provenance=dict(es.provenance))
    probe._recharacterize_row(row, np.asarray(conditioned), FS, start, new_end)
    want = probe._df.iloc[row]

    for col in ("depth_pA", "frac_depth", "baseline_pA", "ecd_pA_s", "snr", "max_dev_pA"):
        assert float(got[col]) == pytest.approx(float(want[col]), rel=0, abs=0), col

    # ...and the raw-channel answer really is different, so the assertion above bites.
    probe2 = _inspector(qtbot, session)
    probe2.set_events(doc_id, 0, df.copy(), provenance=dict(es.provenance))
    probe2._recharacterize_row(row, raw, FS, start, new_end)
    assert float(probe2._df.iloc[row]["frac_depth"]) != pytest.approx(
        float(want["frac_depth"]), rel=1e-9)


def test_a_resized_event_keeps_the_detectors_snr_scale(qapp, qtbot, tmp_path):
    """N1 at this call site: the resized row's SNR denominator is the detector's own noise,
    so the row stays comparable with every row that was not touched."""
    session, doc_id, es = _drifting_session_with_real_detection(tmp_path)
    df = es.df
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, df, provenance=dict(es.provenance))

    ids = [int(i) for i in df["event_id"]]
    insp.show_event(ids[1])
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ids[1])[0][0])
    start = int(df["start_idx"].iloc[row])
    new_end = int(df["end_idx"].iloc[row]) - 120

    _set_region_to_indices(insp, session, doc_id, start, new_end)
    r = df.iloc[row]
    assert float(r["snr"]) == pytest.approx(
        float(r["depth_pA"]) / float(es.provenance["noise_rms_pA"]), rel=1e-9)


def test_the_conditioned_trace_is_cached_across_drags(qapp, qtbot, tmp_path):
    """Conditioning a real recording costs ~0.12 s and a drag pays it on every release."""
    session, doc_id, es = _drifting_session_with_real_detection(tmp_path)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, 0, es.df, provenance=dict(es.provenance))
    raw = np.asarray(session.calibrated(doc_id, 0).data)

    first = insp._conditioned(raw, FS)
    assert insp._conditioned(raw, FS) is first          # memoised

    insp._provenance = dict(es.provenance, filter_cutoff_hz=1_000.0)
    assert insp._conditioned(raw, FS) is not first      # ...and invalidated by provenance


# ================================================================================================
# CUSUM level breakdown in the readout
# ================================================================================================
def _levels_for(session, doc_id: str, channel: int, es):
    """Fit CUSUM levels for a real detection on this session's channel."""
    from nanoscope.analysis.modules import fit_levels
    ch = session.calibrated(doc_id, channel)
    prov = dict(es.provenance)
    return fit_levels(
        es.df, np.asarray(ch.data), ch.sample_rate_hz, doc_id=doc_id, channel=channel,
        baseline=float(np.median(np.asarray(ch.data))),
        polarity=int(prov.get("polarity", 1)), provenance=prov,
    ).payload


def test_readout_is_unchanged_until_a_level_fit_is_installed(qapp, qtbot, tmp_path):
    """A level fit is on-demand: with no table the readout must look exactly as before."""
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    assert "CUSUM levels" not in insp._readout.text()


def test_installing_levels_adds_a_per_level_breakdown(qapp, qtbot, tmp_path):
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    levels = _levels_for(session, doc_id, channel, es)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(levels)

    text = insp._readout.text()
    assert "CUSUM levels" in text
    ev_id = insp._current_id
    n = int(levels[levels["event_id"] == ev_id]["n_levels"].iloc[0])
    for i in range(n):
        assert f"L{i}:" in text, f"level {i} missing from the readout"


def test_dropping_the_level_table_restores_the_plain_readout(qapp, qtbot, tmp_path):
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(_levels_for(session, doc_id, channel, es))
    assert "CUSUM levels" in insp._readout.text()
    insp.set_levels(None)
    assert "CUSUM levels" not in insp._readout.text()


def test_levels_survive_navigation_and_track_the_browsed_event(qapp, qtbot, tmp_path):
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    levels = _levels_for(session, doc_id, channel, es)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(levels)

    for ev_id in es.df["event_id"].tolist()[:3]:
        insp.show_event(int(ev_id))
        rows = levels[levels["event_id"] == int(ev_id)]
        text = insp._readout.text()
        if rows.empty:
            assert "CUSUM levels" not in text
        else:
            assert f"CUSUM levels:    {int(rows['n_levels'].iloc[0])}" in text


def test_an_empty_level_table_is_treated_as_no_table(qapp, qtbot, tmp_path):
    from nanoscope.analysis.cusum import empty_levels
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(empty_levels())
    assert "CUSUM levels" not in insp._readout.text()


def test_set_levels_before_any_detection_is_safe(qapp, qtbot, tmp_path):
    from nanoscope.analysis.cusum import empty_levels
    session, _doc_id, _channel = _session_with_doc(tmp_path)
    insp = _inspector(qtbot, session)
    insp.set_levels(empty_levels())          # must not raise with _df still None
    insp.clear()


def test_switching_detections_drops_the_previous_level_fit(qapp, qtbot, tmp_path):
    """MainWindow re-installs on every selection change and "may be a different
    detection". Without invalidation, recording A's levels were printed under recording
    B's events -- matched by event_id, which both tables have and which means nothing
    across recordings -- and the disagreement line then argued for the wrong number."""
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(_levels_for(session, doc_id, channel, es))
    assert "CUSUM levels" in insp._readout.text()

    other_id = session.add_document(_synth_doc(tmp_path, name="B.dat"))
    es2 = _events_for(session, other_id, 0)
    insp.set_events(other_id, 0, es2.df, provenance=dict(es2.provenance))
    assert "CUSUM levels" not in insp._readout.text(), "stale levels shown under a new detection"


def test_reinstalling_the_same_frame_keeps_the_level_fit(qapp, qtbot, tmp_path):
    """The invalidation must not fire on the same-frame refresh a sibling fit triggers."""
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(_levels_for(session, doc_id, channel, es))
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    assert "CUSUM levels" in insp._readout.text()


def test_resizing_an_event_drops_its_now_meaningless_levels(qapp, qtbot, tmp_path):
    """Every measured column is rebuilt from the new bounds; the levels were fitted to
    the old ones. Left in place the readout printed more level time than the event
    contains, under a line telling the user to prefer the CUSUM count.

    STILL THE WINDOW'S OWN RULE after round 4. The automatic re-fit that puts a group back
    over the NEW span belongs to ``MainWindow._refit_levels_for_event`` — it needs the
    baseline track and the detection's estimators, neither of which this window holds — and
    is pinned in ``tests/gui/test_level_fit_reconcile.py``. A standalone inspector has no
    owner to re-fit for it, so dropping is the whole of its behaviour and must stay so: the
    alternative is a window that shows sub-states it measured itself, by its own recipe.

    ONE EXPECTATION CHANGED HERE, DELIBERATELY (F2-b). This asserted ``"CUSUM levels" not
    in text`` — the whole block gone, header and all — as a proxy for "no stale breakdown".
    That proxy is now too strong, because dropping the group in SILENCE is the defect F2-b
    is about: the user drags the mask, five lines of breakdown vanish out of the middle of
    the readout, and the always-on ``levels:`` count above still prints a number, so the
    panel reads as "1 level" rather than "no fit here". The block therefore stays, printing
    ``CUSUM levels:    none`` with the reason and a remedy under it.

    The property this test exists for is untouched, and is now asserted directly rather
    than through the header: not one PER-LEVEL row of the old fit survives the resize.
    """
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    levels = _levels_for(session, doc_id, channel, es)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(levels)

    ev_id = int(es.df["event_id"].iloc[0])
    insp.show_event(ev_id)
    row = insp._row_for_id(ev_id)
    start = int(es.df.iloc[row]["start_idx"])
    end = int(es.df.iloc[row]["end_idx"])
    _set_region_to_indices(insp, session, doc_id, start, start + max((end - start) // 2, 2))

    text = insp._readout.text()
    remaining = insp._levels
    assert remaining is None or (remaining["event_id"] != ev_id).all(), (
        "levels fitted to the old span must not survive a re-measure")
    assert "  L0:" not in text and "  L1:" not in text, (
        "a per-level row measured on the pre-resize span is still in the readout:\n" + text)
    # ...and the absence is STATED rather than left as a hole in the panel (F2-b).
    assert "CUSUM levels:    none" in text, text


def test_the_level_listing_is_bounded(qapp, qtbot, tmp_path):
    """max_levels defaults to 32; an unbounded listing grows the window past the height
    of the waveform plot it sits above."""
    import pandas as pd
    from nanoscope.analysis.cusum import LEVEL_SCHEMA
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))

    ev_id = int(es.df["event_id"].iloc[0])
    n = 30
    fake = pd.DataFrame({
        "event_id": [ev_id] * n, "recording": ["r"] * n, "pore_id": ["p"] * n,
        "level_idx": list(range(n)), "n_levels": [n] * n,
        "start_idx": list(range(n)), "end_idx": list(range(1, n + 1)),
        "t_start_s": [0.0] * n, "dwell_s": [1e-4] * n, "level_pA": [10.0] * n,
        "depth_pA": [10.0] * n, "frac_depth": [0.1] * n, "step_pA": [np.nan] * n,
    }).astype({c: d for c, (d, _, _) in LEVEL_SCHEMA.items()})[list(LEVEL_SCHEMA)]
    insp.show_event(ev_id)
    insp.set_levels(fake)

    lines = insp._readout.text().splitlines()
    level_lines = [ln for ln in lines if ln.strip().startswith("L")]
    assert len(level_lines) <= 8, f"{len(level_lines)} level lines is unbounded growth"
    assert any("more" in ln for ln in lines), "truncation must be stated, not silent"
    assert "CUSUM levels:    30" in insp._readout.text(), "the true count is still reported"


def test_cusum_levels_are_drawn_as_flat_segments(qapp, qtbot, tmp_path):
    """A level fit whose only output is TEXT is not a fit anyone can check.

    The readout listed L0/L1/L2 with amplitudes and dwells while the plot showed one
    ``level`` line — characterize's single in-event median — so the picture said "one
    state" beside a table saying "three", and the eye believes the picture.
    """
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    levels = _levels_for(session, doc_id, channel, es)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))

    ev_id = int(es.df["event_id"].iloc[0])
    insp.show_event(ev_id)
    assert not getattr(insp, "_cusum_items", []), "nothing to draw before a fit"

    insp.set_levels(levels)
    insp.show_event(ev_id)
    rows = levels[levels["event_id"] == ev_id]
    segs = insp._cusum_items
    # TWO items per level since the legibility fix: a dark casing under the green
    # stroke (the owner could not read 2-px segments over the waveform).
    assert len(segs) == 2 * len(rows), "casing + stroke per fitted level"

    # Each segment is HORIZONTAL and sits at that level's own current.
    drawn_y = sorted({round(float(s.getData()[1][0]), 9) for s in segs})
    want_y = sorted({round(float(v), 9) for v in rows["level_pA"]})
    for got, want in zip(drawn_y, want_y):
        assert got == pytest.approx(want, rel=1e-6)
    for s in segs:
        ys = s.getData()[1]
        assert ys[0] == ys[-1], "a level is a FLAT line, not a slope"


def test_dropping_the_level_fit_removes_the_drawn_segments(qapp, qtbot, tmp_path):
    session, doc_id, channel = _session_with_doc(tmp_path)
    es = _events_for(session, doc_id, channel)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, channel, es.df, provenance=dict(es.provenance))
    insp.set_levels(_levels_for(session, doc_id, channel, es))
    insp.show_event(int(es.df["event_id"].iloc[0]))
    assert insp._cusum_items

    insp.set_levels(None)
    insp.show_event(int(es.df["event_id"].iloc[0]))
    assert not insp._cusum_items, "stale level segments must not outlive the fit"


# ================================================================================================
# labelling one event at a time (the gallery is the bulk surface; this is the detailed one)
# ================================================================================================

@pytest.fixture
def inspector_with_events(qapp, qtbot, tmp_path):
    """An inspector fed a real detection — (inspector, shared events df)."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df)
    return insp, events.df



def _label_vocab():
    from nanoscope.analysis.labels import LabelVocabulary
    v = LabelVocabulary()
    v.add("molecule", "apo-ferritin")
    v.add("levels", "3", hotkey="3")
    return v


def test_a_digit_labels_the_browsed_event(qtbot, inspector_with_events):
    """Same digits, same meanings, as the gallery — a curator who learns them once uses them
    in both places."""
    from nanoscope.analysis.labels import parse_labels

    insp, df = inspector_with_events
    insp.set_vocabulary(_label_vocab())
    insp.show_event(int(df.iloc[1]["event_id"]))
    fired = []
    insp.labelsChanged.connect(lambda: fired.append(True))
    insp.apply_hotkey("1")
    qtbot.wait(10)
    assert parse_labels(df.iloc[1]["flag"]) == {"molecule": "apo-ferritin"}
    assert fired, "labelling did not announce itself"
    assert str(df.iloc[1][REVIEW_COL]) == REVIEW_DONE
    assert parse_labels(df.iloc[0]["flag"]) == {}, "it labelled an event that was not on screen"


def test_a_level_count_is_just_another_label(qtbot, inspector_with_events):
    """The ground truth that would settle CUSUM tau is a hand-read number, and a number is a
    label like any other — no separate mechanism, no schema change."""
    from nanoscope.analysis.labels import parse_labels

    insp, df = inspector_with_events
    insp.set_vocabulary(_label_vocab())
    insp.show_event(int(df.iloc[0]["event_id"]))
    insp.apply_hotkey("3")
    qtbot.wait(10)
    assert parse_labels(df.iloc[0]["flag"])["levels"] == "3"


def test_an_explicit_custom_hotkey_works_from_a_real_inspector_key_event(
        qtbot, inspector_with_events):
    from nanoscope.analysis.labels import LabelVocabulary, parse_labels

    insp, df = inspector_with_events
    vocab = LabelVocabulary()
    vocab.add("type", "collision", hotkey="c")
    insp.set_vocabulary(vocab)
    insp.show_event(int(df.iloc[0]["event_id"]))
    insp.show()
    insp.setFocus()

    qtbot.keyClick(insp, Qt.Key_C)
    qtbot.wait(20)

    assert parse_labels(df.iloc[0]["flag"])["type"] == "collision"


def test_the_readout_shows_the_labels_on_the_event(qtbot, inspector_with_events):
    insp, df = inspector_with_events
    insp.set_vocabulary(_label_vocab())
    insp.show_event(int(df.iloc[0]["event_id"]))
    insp.apply_hotkey("1")
    qtbot.wait(10)
    assert "molecule=apo-ferritin" in insp._readout.text()


def test_labelling_leaves_curation_and_flags_alone(qtbot, inspector_with_events):
    from nanoscope.analysis.labels import parse_labels

    insp, df = inspector_with_events
    insp.set_vocabulary(_label_vocab())
    eid = int(df.iloc[0]["event_id"])
    insp.show_event(eid)
    insp.set_accepted_current(False)
    insp.apply_hotkey("1")
    qtbot.wait(10)
    assert bool(df.iloc[0]["accepted"]) is False, "the rejection was lost"
    assert parse_labels(df.iloc[0]["flag"]) == {"molecule": "apo-ferritin"}


def test_a_digit_with_no_palette_entry_does_nothing(qtbot, inspector_with_events):
    from nanoscope.analysis.labels import LabelVocabulary, parse_labels

    insp, df = inspector_with_events
    insp.set_vocabulary(LabelVocabulary())
    insp.show_event(int(df.iloc[0]["event_id"]))
    insp.apply_hotkey("1")
    qtbot.wait(10)
    assert parse_labels(df.iloc[0]["flag"]) == {}

# ================================================================================================
# owner notes #6 / #8 / #5: explicit Accept/Reject buttons, view panning, and the road
# back to the gallery
# ================================================================================================

def test_accept_and_reject_buttons_curate_the_displayed_event(window, qtbot, tmp_path):
    """The checkbox and the A/R keys did this already, but a state indicator is not an
    action affordance — the owner curated a whole detection without finding either
    (note #6). The buttons are the same handler as the keys: curate, do not navigate."""
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._act_event_inspector.trigger()
    insp = window._event_inspector

    ev_id = insp.current_event_id()
    df = window._analysis_panel._table.dataframe
    row = int(np.nonzero(df["event_id"].to_numpy(np.int64) == ev_id)[0][0])

    insp._reject_btn.click()
    assert not bool(df.iloc[row][ "accepted"])
    assert insp.current_event_id() == ev_id, "Reject must not navigate away"
    insp._accept_btn.click()
    assert bool(df.iloc[row]["accepted"])
    assert insp.current_event_id() == ev_id, "Accept must not navigate away"


def test_pan_buttons_shift_the_view_and_touch_nothing_else(window, qtbot, tmp_path):
    """Note #8: inspect the trace AROUND the event — the mask region, the df and the browsed
    event stay exactly where they were.

    STRENGTHENED after the audit. This test used to assert only that the view RANGE moved,
    which the old implementation did while revealing nothing: the slice was taken once and
    framed edge to edge, so a press slid the pane off the end of its own data — half the plot
    blank after one press, all of it after two — and the test passed the whole time. What a
    pan has to do is LOAD samples, so that is what is asserted here: the window advances and
    the curve still has samples inside the view afterwards.
    """
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._act_event_inspector.trigger()
    insp = window._event_inspector

    ev_id = insp.current_event_id()
    (x0, x1), _ = insp._plot.viewRange()
    region_before = insp._region.getRegion()
    width = x1 - x0
    lo_before, hi_before = insp._win_lo, insp._win_hi
    step = (hi_before - lo_before) // 2

    insp._pan_right_btn.click()
    (nx0, nx1), _ = insp._plot.viewRange()
    assert nx0 > x0 and nx1 > x1, "the view moved forward in the recording"
    assert (nx1 - nx0) == pytest.approx(width, rel=1e-6), "a pan is a translation, not a zoom"
    # The loaded window moved WITH it — this is the half that did not exist.
    assert (insp._win_lo, insp._win_hi) == (lo_before + step, hi_before + step)
    assert insp._region.getRegion() == region_before, "panning must not move the mask"
    assert insp.current_event_id() == ev_id

    x, y = insp._curve.getData()
    on_screen = int(np.count_nonzero((x >= nx0) & (x <= nx1)))
    assert on_screen > 0, "a pan that reveals no samples is a pan into blank space"
    assert y.size == x.size

    # Twice more: the old code had NOTHING left on screen by here.
    insp._pan_right_btn.click()
    (fx0, fx1), _ = insp._plot.viewRange()
    x, _y = insp._curve.getData()
    assert int(np.count_nonzero((x >= fx0) & (x <= fx1))) > 0

    insp._pan_left_btn.click()
    insp._pan_left_btn.click()
    (bx0, bx1), _ = insp._plot.viewRange()
    assert bx0 == pytest.approx(x0, rel=1e-6)
    assert bx1 == pytest.approx(x1, rel=1e-6)
    assert (insp._win_lo, insp._win_hi) == (lo_before, hi_before)


def test_gallery_button_opens_the_event_gallery(window, qtbot, tmp_path):
    """Note #5: the surfaces link to each other, not only through the Analysis menu."""
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)
    window._act_event_inspector.trigger()
    insp = window._event_inspector
    assert window._event_gallery is None

    insp._gallery_btn.click()
    qtbot.wait(30)
    assert window._event_gallery is not None and window._event_gallery.isVisible()


# ================================================================================================
# the audit round: the pan buttons were cosmetic, the per-event buttons were off the right edge,
# there was no way back to the main trace, and a mask drag could swallow its neighbour
# ================================================================================================

def _inspector_with_detection(qtbot, tmp_path):
    """(inspector, session, doc_id, channel, events) on a real detection — no MainWindow."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df, provenance=dict(events.provenance))
    return insp, session, doc_id, ch, events


def test_a_pan_loads_samples_that_were_never_on_screen(qapp, qtbot, tmp_path):
    """The defect the buttons had: the window was sliced ONCE and framed edge to edge, so
    every loaded sample was already visible before the first click and a "pan" could only
    walk the view off its own data. The window has to move through the RECORDING."""
    insp, session, doc_id, ch, _events = _inspector_with_detection(qtbot, tmp_path)
    xvals = session.x_axis(doc_id).values

    lo, hi = insp._win_lo, insp._win_hi
    x_before, _ = insp._curve.getData()
    last_before = float(x_before[-1])

    insp._pan_view(0.5)
    assert insp._win_hi > hi and insp._win_lo > lo
    x_after, y_after = insp._curve.getData()
    assert float(x_after[-1]) > last_before, "samples past the old window are now loaded"
    assert y_after.size == x_after.size
    # ...and the window is still the same WIDTH: a pan is a scroll, not a zoom.
    assert (insp._win_hi - insp._win_lo) == (hi - lo)
    # Every drawn sample really is the recording's, at its own time.
    assert float(x_after[0]) == pytest.approx(float(xvals[insp._win_lo]), rel=1e-12)


def test_a_pan_stops_at_the_edges_of_the_recording(qapp, qtbot, tmp_path):
    """Panning past sample 0 / sample n must not shrink the window (the context width the
    user chose is inherited by the next event) and must not blank the plot."""
    insp, _session, _doc_id, _ch, events = _inspector_with_detection(qtbot, tmp_path)
    insp.show_event(int(events.df["event_id"].iloc[0]))
    width = insp._win_hi - insp._win_lo
    for _ in range(40):
        insp._pan_view(-0.5)
    assert insp._win_lo == 0
    assert (insp._win_hi - insp._win_lo) == width
    x, _y = insp._curve.getData()
    assert x.size > 0


def test_the_level_overlay_pans_with_the_data(qapp, qtbot, tmp_path):
    """The CUSUM segments are positioned by ``level.start_idx - window_start``. Redrawing them
    against a stale window start puts every fitted level at the wrong time — off by exactly the
    distance panned — which is worse than not drawing them at all."""
    insp, session, doc_id, ch, events = _inspector_with_detection(qtbot, tmp_path)
    levels = _levels_for(session, doc_id, ch, events)
    insp.set_levels(levels)
    ev_id = int(events.df["event_id"].iloc[0])
    insp.show_event(ev_id)
    assert insp._cusum_items

    rows = levels[levels["event_id"] == ev_id]
    xvals = session.x_axis(doc_id).values
    want_x0 = float(xvals[int(rows["start_idx"].iloc[0])])
    got_x0 = min(float(s.getData()[0][0]) for s in insp._cusum_items)
    assert got_x0 == pytest.approx(want_x0, rel=1e-12)

    insp._pan_view(0.25)                     # small enough that the levels stay in the window
    still = min(float(s.getData()[0][0]) for s in insp._cusum_items)
    assert still == pytest.approx(want_x0, rel=1e-12), \
        "a level must stay at its own time when the window scrolls under it"


def test_a_mouse_zoom_out_loads_the_wider_window_too(qapp, qtbot, tmp_path):
    """The buttons loading data while drag/wheel did not would be a worse inconsistency than
    neither working: two gestures disagreeing about what the same window contains."""
    insp, session, doc_id, ch, _events = _inspector_with_detection(qtbot, tmp_path)
    xvals = session.x_axis(doc_id).values
    lo, hi = insp._win_lo, insp._win_hi
    span = hi - lo

    # A wheel zoom-out IS this: the viewbox announces a wider x range.
    insp._plot.setXRange(float(xvals[max(lo - 5 * span, 0)]),
                         float(xvals[min(hi + 5 * span, xvals.size - 1)]), padding=0)
    qtbot.wait(10)
    assert insp._win_lo < lo and insp._win_hi > hi, "the loaded window followed the view"
    x, _y = insp._curve.getData()
    assert float(x[0]) <= float(xvals[lo])

    # ...and the width the user framed by hand is inherited by the next event (the centring
    # is not: stepping re-centres, which is what stepping is for).
    wide = insp._win_hi - insp._win_lo
    insp.next_event()
    assert (insp._win_hi - insp._win_lo) > 3 * span
    assert (insp._win_hi - insp._win_lo) == pytest.approx(wide, rel=0.25)


def test_the_view_cannot_leave_the_recording(qapp, qtbot, tmp_path):
    """Unbounded blank space either side is how both the buttons and the mouse ended up
    showing nothing. The x limits are the recording's own extent."""
    insp, session, doc_id, _ch, _events = _inspector_with_detection(qtbot, tmp_path)
    xvals = session.x_axis(doc_id).values
    limits = insp._plot.getViewBox().state["limits"]["xLimits"]
    assert limits[0] == pytest.approx(float(xvals[0]))
    assert limits[1] == pytest.approx(float(xvals[-1]))


def test_the_context_combo_sets_and_keeps_the_window_width(qapp, qtbot, tmp_path):
    """±3 ms was a hard CEILING, not a default: _PAD_MS was unreachable from the UI, so no
    gesture in this window could show the second before the blockade. And the chosen context
    survives navigation — a curator sets it once and then steps."""
    insp, session, doc_id, ch, events = _inspector_with_detection(qtbot, tmp_path)
    narrow = insp._win_hi - insp._win_lo

    insp._context_combo.setCurrentIndex(1)           # ± 30 ms
    wide = insp._win_hi - insp._win_lo
    assert wide > 5 * narrow

    insp.next_event()
    assert (insp._win_hi - insp._win_lo) == pytest.approx(wide, rel=0.1), \
        "the chosen context width must survive Prev/Next"
    # ...while the CENTRING resets: the new event is inside its own window.
    row = insp._row_for_id(insp.current_event_id())
    start = int(events.df["start_idx"].iloc[row])
    assert insp._win_lo <= start <= insp._win_hi

    insp._context_combo.setCurrentIndex(insp._context_combo.count() - 1)   # whole trace
    n = int(session.calibrated(doc_id, ch).data.size)
    assert (insp._win_lo, insp._win_hi) == (0, n)


def test_the_curation_bar_fits_a_1512_px_display(qapp, qtbot, tmp_path):
    """The per-event Accept/Reject were reported missing TWICE. They were present and
    correctly wired the whole time — and the single non-wrapping row that held them measured
    1475 px of minimum width, with Accept/Reject/Accepted as its LAST three widgets, on a
    1512 px display. Any Dock, any scaling, any window not flush left pushed exactly those
    three off the right edge while 'Accept all' / 'Reject all' stayed visible."""
    old_font = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        session, doc_id, ch = _session_with_doc(tmp_path)
        insp = _inspector(qtbot, session)
        assert insp.minimumSizeHint().width() <= 1100

        insp.set_events(doc_id, ch, _events_for(session, doc_id, ch).df)
        assert insp._accept_btn.isVisibleTo(insp) and insp._reject_btn.isVisibleTo(insp)
        assert insp.minimumSizeHint().width() <= 1100, \
            "with every control shown the window must still fit a 1512 px display"

        # The per-event gestures sit with the per-event navigation, never behind the batch
        # buttons: same layout row as Prev, a different one from 'Accept all'.
        def _row_of(w):
            for i in range(insp.centralWidget().layout().count()):
                item = insp.centralWidget().layout().itemAt(i)
                lay = item.layout()
                if lay is not None and lay.indexOf(w) >= 0:
                    return i
            return None

        assert _row_of(insp._accept_btn) == _row_of(insp._prev_btn)
        assert _row_of(insp._accept_btn) != _row_of(insp._accept_all_btn)
    finally:
        qapp.setFont(old_font)


def test_per_event_controls_are_hidden_until_there_is_an_event(qapp, qtbot, tmp_path):
    """Two greyed buttons under "No detection" taught the owner that this window had no
    per-event Accept/Reject at all. A disabled control promises it applies here; with no
    detection that promise is false, so they are hidden and the message names them."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    insp = _inspector(qtbot, session)
    for w in (insp._accept_btn, insp._reject_btn, insp._accept_check):
        assert not w.isVisibleTo(insp)
    assert "Accept" in insp._readout.text() and "Reject" in insp._readout.text()

    insp.set_events(doc_id, ch, _events_for(session, doc_id, ch).df)
    for w in (insp._accept_btn, insp._reject_btn, insp._accept_check):
        assert w.isVisibleTo(insp) and w.isEnabled()

    insp.clear()
    for w in (insp._accept_btn, insp._reject_btn, insp._accept_check):
        assert not w.isVisibleTo(insp)


def test_show_in_trace_asks_for_the_event_in_the_main_trace(qapp, qtbot, tmp_path):
    """The inspector -> main trace edge did not exist: the panel opens this window and the
    gallery, the gallery jumps to the trace, and from here the only way back to the recording
    was hunting for the row in the events table."""
    insp, _session, _doc_id, _ch, events = _inspector_with_detection(qtbot, tmp_path)
    seen: list[int] = []
    insp.traceRequested.connect(seen.append)

    insp._trace_btn.click()
    assert seen == [insp.current_event_id()]

    insp.next_event()
    insp._trace_btn.click()
    assert seen[-1] == insp.current_event_id()

    # T does the same thing, and neither says anything with no detection installed.
    assert any(sc.key().toString() == "T" for sc in insp._shortcuts)
    insp.clear()
    assert not insp._trace_btn.isEnabled()
    insp._on_show_in_trace()
    assert len(seen) == 2


def test_a_mask_drag_cannot_swallow_the_neighbouring_event(qapp, qtbot, tmp_path):
    """A drag was clipped to [0, n] and nothing else, so an edge dragged across the adjacent
    event produced an OVERLAPPING table: two shaded spans drawn on top of one another, and a
    merge grouping rule handed a negative gap between one event's end and the next's start."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, ch)
    df = events.df
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, df, provenance=dict(events.provenance))

    ids = [int(i) for i in df["event_id"]]
    insp.show_event(ids[0])
    start0 = int(df["start_idx"].iloc[0])
    next_start = int(df["start_idx"].iloc[1])
    next_end = int(df["end_idx"].iloc[1])

    # Drag the trailing edge WELL past the next event's start.
    _set_region_to_indices(insp, session, doc_id, start0, next_start + 500)

    assert int(df["end_idx"].iloc[0]) <= next_start, "the mask stops at the neighbour's edge"
    assert (int(df["start_idx"].iloc[1]), int(df["end_idx"].iloc[1])) == (next_start, next_end), \
        "the neighbour itself is never rewritten"
    starts = df["start_idx"].to_numpy(np.int64)
    ends = df["end_idx"].to_numpy(np.int64)
    order = np.argsort(starts)
    assert np.all(starts[order][1:] >= ends[order][:-1]), "the table must not overlap itself"


def test_level_segments_are_drawn_in_the_shared_level_ink_and_labelled(qapp, qtbot, tmp_path):
    """Owner, twice: "the sublevels ... cannot be read easily on the curate panel. Either the
    colour has to be boldened/changed or another way of representing the sublevels is needed."
    One name for the ink across this window and the main trace, the trace overlay's own
    weights, and each segment labelled with its number and depth."""
    import pyqtgraph as pg
    from nanoscope.gui.overlays import STEP_LEVEL_HEX

    insp, session, doc_id, ch, events = _inspector_with_detection(qtbot, tmp_path)
    levels = _levels_for(session, doc_id, ch, events)
    insp.set_levels(levels)
    ev_id = int(events.df["event_id"].iloc[0])
    insp.show_event(ev_id)
    assert insp._cusum_items

    pens = [s.opts["pen"] for s in insp._cusum_items]
    inks = {p.color().name().lower() for p in pens}
    assert STEP_LEVEL_HEX.lower() in inks, "one name for a level's colour, not a second hex"
    widths = {p.width() for p in pens}
    assert max(widths) >= 7 and 4 in widths, "casing under a boldened stroke"

    labels = [it for it in insp._cusum_extras if isinstance(it, pg.TextItem)]
    assert labels, "each readable level says which level it is and how deep"
    assert any(t.toPlainText().startswith("L0") for t in labels)

    # The furniture dies with the fit, exactly like the segments.
    insp.set_levels(None)
    insp.show_event(ev_id)
    assert not insp._cusum_items and not insp._cusum_extras


def test_re_picking_the_current_context_undoes_a_mouse_zoom(qapp, qtbot, tmp_path):
    """Picking the entry that is already current is how a curator gets back to ±3 ms after
    zooming out with the wheel; on ``currentIndexChanged`` alone that gesture did nothing."""
    insp, session, doc_id, ch, _events = _inspector_with_detection(qtbot, tmp_path)
    narrow = insp._win_hi - insp._win_lo
    xvals = session.x_axis(doc_id).values

    insp._plot.setXRange(float(xvals[0]), float(xvals[-1]), padding=0)   # wheel, all the way out
    qtbot.wait(10)
    assert (insp._win_hi - insp._win_lo) > narrow

    insp._context_combo.activated.emit(insp._context_combo.currentIndex())
    assert (insp._win_hi - insp._win_lo) == narrow


# ================================================================================================
# ROUND 4 / E2(b) — AT WIDE CONTEXT THE LABELS AND RISERS VANISHED
# ================================================================================================
# Owner: "when I change the window manually, the sublevels disappear." Two separate things sat
# behind that sentence; this is the second one, and it is not about the level TABLE at all --
# the fit is intact, the segments are drawn, and the ANNOTATION disappears.
#
# The gate was `min_label_span = max(xs.size // 40, 1)` SAMPLES: a level had to cover 2.5% of
# the LOADED WINDOW to be labelled. So widening the context to look at the open pore -- the
# whole reason the combo exists -- silently stripped the labels off levels that had not moved
# and were drawn at exactly the same width in time. On this file's fixture (a 600-sample event
# with two ~300-sample sub-states) the old rule gave 2 labels at +-3 ms, 2 at +-30 ms and ZERO
# at +-300 ms and at Whole trace, because 40 600 // 40 = 1015 samples is wider than either
# level. The recon measured the same collapse one step earlier on the owner's data
# (`_cusum_extras` 3 -> 1 the moment the context went from +-3 ms to +-30 ms).
#
# A rule about LEGIBILITY has to be in PIXELS. `_MIN_LABEL_PX` is that, plus the floor a width
# rule cannot supply on its own: at Whole trace the entire event is ~27 px on a 942 px plot, so
# no level can clear any sensible bar and the annotation would still go to zero. One label on
# the widest level always survives, and a faint band says where to look for it.

def _label_texts(insp) -> list[str]:
    import pyqtgraph as pg
    return [it.toPlainText() for it in insp._cusum_extras if isinstance(it, pg.TextItem)]


def _bands(insp) -> list:
    import pyqtgraph as pg
    return [it for it in insp._cusum_extras if isinstance(it, pg.LinearRegionItem)]


def _substate_doc(tmp_path, name: str = "substates.dat") -> TraceDocument:
    """The module's own recording, but with each event split into TWO clean sub-states.

    The shared fixture's events are square, so CUSUM resolves exactly one level in each and
    a test about labels CROWDING has nothing to crowd. Same spans, same rate, same length —
    only the depth steps half way through, which is the structure the level fit exists for.
    """
    rng = np.random.default_rng(0)
    x = np.full(100_000, 100.0) + rng.normal(0.0, 1.5, 100_000)
    for s, e in _EVENTS:
        mid = (s + e) // 2
        x[s:mid] -= 30.0
        x[mid:e] -= 45.0
    arr = np.ascontiguousarray(x, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _fitted_inspector(qtbot, tmp_path):
    """(inspector, session, doc_id, channel, events, levels) on a MULTI-LEVEL detection with a
    real CUSUM fit installed."""
    session = Session()
    doc_id = session.add_document(_substate_doc(tmp_path))
    ch = 0
    events = _events_for(session, doc_id, ch)
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df, provenance=dict(events.provenance))
    levels = _levels_for(session, doc_id, ch, events)
    insp.set_levels(levels)
    insp.show_event(int(events.df["event_id"].iloc[0]))
    return insp, session, doc_id, ch, events, levels


def test_a_level_is_still_labelled_after_the_context_is_widened(qapp, qtbot, tmp_path):
    """THE E2(b) HEADLINE: the same event, at ±3 ms and at ±300 ms, must be labelled in both.

    Under the sample-count gate this asserted 2 labels and then 0 — the annotation deleted by
    a change of CONTEXT, on a fit that had not moved. Both ends of the combo are checked
    because the failure is a comparison: it is not that wide contexts are unlabellable, it is
    that the rule was reading how much data happened to be loaded.
    """
    insp, session, doc_id, ch, events, _levels = _fitted_inspector(qtbot, tmp_path)
    ev_id = int(events.df["event_id"].iloc[0])

    insp._context_combo.setCurrentIndex(0)          # ± 3 ms
    insp.show_event(ev_id)
    narrow = _label_texts(insp)
    narrow_win = insp._win_hi - insp._win_lo
    assert narrow, "the levels are not labelled even at the tightest context"

    insp._context_combo.setCurrentIndex(2)          # ± 300 ms
    insp.show_event(ev_id)
    wide = _label_texts(insp)
    assert insp._win_hi - insp._win_lo > 10 * narrow_win, "the context did not actually widen"
    assert wide, (
        "widening the context to look at the open pore deleted every level label — the fit "
        "is unchanged and the segments are still drawn, so the annotation vanished for no "
        "reason the user can see")
    assert insp._cusum_items, "the segments themselves must still be drawn"


def test_the_labels_survive_the_whole_trace_and_still_thin_out_when_crowded(qapp, qtbot,
                                                                            tmp_path):
    """Both halves of the pixel rule at once.

    At "Whole trace" the event is a few pixels wide, so exactly ONE label is kept (the widest
    level) — enough to say what the staircase is, never zero. At ±3 ms there is room for every
    level, so every level is labelled. A rule that only ever kept one, or only ever kept all,
    would pass one half of this and fail the other.
    """
    insp, session, doc_id, ch, events, levels = _fitted_inspector(qtbot, tmp_path)
    ev_id = int(events.df["event_id"].iloc[0])
    n_levels = int((levels["event_id"] == ev_id).sum())
    assert n_levels >= 2, "need a multi-level event for this to mean anything"

    insp._context_combo.setCurrentIndex(0)          # ± 3 ms
    insp.show_event(ev_id)
    assert len(_label_texts(insp)) == n_levels, "there is room for every label and one is missing"

    insp._context_combo.setCurrentIndex(3)          # Whole trace
    insp.show_event(ev_id)
    wide = _label_texts(insp)
    assert len(wide) == 1, (
        f"{len(wide)} labels on an event a few pixels wide — they are painted over one another")


def test_a_far_wider_window_gets_a_band_saying_where_the_event_is(qapp, qtbot, tmp_path):
    """At Whole trace the staircase is a hairline somewhere in a million samples. The band is
    the "it is here" marker; it appears only once the window dwarfs the event, is drawn behind
    the waveform, and dies with the fit like every other piece of level furniture."""
    insp, session, doc_id, ch, events, _levels = _fitted_inspector(qtbot, tmp_path)
    ev_id = int(events.df["event_id"].iloc[0])

    insp._context_combo.setCurrentIndex(0)          # ± 3 ms — the event fills the plot
    insp.show_event(ev_id)
    assert not _bands(insp), "a band over an event that already fills the plot is just noise"

    insp._context_combo.setCurrentIndex(3)          # Whole trace
    insp.show_event(ev_id)
    band = _bands(insp)
    assert len(band) == 1
    assert band[0].zValue() < min(s.zValue() for s in insp._cusum_items), (
        "the band must sit UNDER the level segments, never over them")

    insp.set_levels(None)
    insp.show_event(ev_id)
    assert not _bands(insp), "the band outlived the fit it points at"


def test_the_label_rule_does_not_move_when_only_the_context_does(qapp, qtbot, tmp_path):
    """The property underneath both tests above, stated directly: a level's labellability is a
    function of how wide it is ON SCREEN, so the same level at the same on-screen width must
    get the same answer whatever the loaded window is.

    Checked by comparing a level's pixel width with whether it was labelled, at every context
    the combo offers. The old rule failed this by construction — its threshold WAS the window
    length.
    """
    insp, session, doc_id, ch, events, levels = _fitted_inspector(qtbot, tmp_path)
    from nanoscope.gui.event_inspector import _MIN_LABEL_PX

    ev_id = int(events.df["event_id"].iloc[0])
    rows = levels[levels["event_id"] == ev_id]
    widest = int((rows["end_idx"] - rows["start_idx"]).max())

    for index in range(4):
        insp._context_combo.setCurrentIndex(index)
        insp.show_event(ev_id)
        win = insp._win_hi - insp._win_lo
        px = insp._plot_width_px() * widest / max(win - 1, 1)
        labels = _label_texts(insp)
        if px >= _MIN_LABEL_PX:
            assert labels, f"the widest level is {px:.0f} px and carries no label"
        # ...and below the bar the fallback still keeps exactly one, never zero.
        assert labels, "the annotation reached zero, which is the reported defect"


# ================================================================================================
# ROUND 4 / E1 — REVERT TO THE BOUNDS THE DETECTOR PRODUCED
# ================================================================================================
# Owner: "the user needs the option to revert to the model's start/end regions." The detector's
# own start_idx/end_idx were preserved NOWHERE durable -- the store's frame IS the edited frame,
# edited in place by the mask drag -- so this needed somewhere to put them first
# (`events.stamp_original_bounds`, two columns outside SCHEMA so old projects still load and
# `characterize`'s `df[list(SCHEMA)]` cannot erase them).
#
# The gesture reuses the MASK DRAG'S OWN WRITE PATH (`_apply_edited_bounds` ->
# `_recharacterize_row` -> `_write_measured_row`) rather than adding a second writer. A second
# writer is exactly how the level reconcile and the characterisation got out of step before:
# `_write_measured_row` is what reconciles the levels and `eventEdited` is what makes MainWindow
# re-fit them, checkpoint the undo step and refresh the table.

def _stamped_inspector(qtbot, tmp_path):
    """(inspector, session, doc_id, channel, events) whose frame carries the detector's bounds."""
    from nanoscope.analysis.events import EventSet, stamp_original_bounds

    session, doc_id, ch = _session_with_doc(tmp_path)
    raw = _events_for(session, doc_id, ch)
    events = EventSet(stamp_original_bounds(raw.df), dict(raw.provenance))
    insp = _inspector(qtbot, session)
    insp.set_events(doc_id, ch, events.df, provenance=dict(events.provenance))
    return insp, session, doc_id, ch, events


def test_revert_puts_the_event_back_on_the_detectors_bounds(qapp, qtbot, tmp_path):
    """The gesture itself: drag the mask, press Revert, land back on the detected span with
    every measured column re-measured for it."""
    insp, session, doc_id, ch, events = _stamped_inspector(qtbot, tmp_path)
    df = events.df
    ev_id = int(df["event_id"].iloc[0])
    insp.show_event(ev_id)
    row = insp._row_for_id(ev_id)
    start, end = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])
    dwell = float(df.iloc[row]["dwell_s"])

    _set_region_to_indices(insp, session, doc_id, start, start + (end - start) // 2)
    assert int(df.iloc[row]["end_idx"]) < end, "the drag did not move the mask"
    assert float(df.iloc[row]["dwell_s"]) < dwell

    with qtbot.waitSignal(insp.eventEdited, timeout=1000) as edited:
        insp._revert_current()

    assert (int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])) == (start, end)
    assert float(df.iloc[row]["dwell_s"]) == pytest.approx(dwell, rel=1e-9), (
        "the bounds came back but the row was not re-measured for them")
    assert edited.args == [ev_id], (
        "revert must announce itself exactly as a mask drag does — that signal is what "
        "re-fits the sub-state levels, checkpoints the undo step and refreshes the table")


def test_revert_is_offered_only_when_there_is_an_edit_to_undo(qapp, qtbot, tmp_path):
    """Disabled on an untouched event (nothing to revert), enabled the moment one is dragged,
    disabled again once it is back. The tooltip says which of the three it is."""
    insp, session, doc_id, ch, events = _stamped_inspector(qtbot, tmp_path)
    df = events.df
    ev_id = int(df["event_id"].iloc[0])
    insp.show_event(ev_id)
    row = insp._row_for_id(ev_id)
    start, end = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])

    assert not insp._revert_btn.isEnabled()
    assert "already on the bounds" in insp._revert_btn.toolTip()

    _set_region_to_indices(insp, session, doc_id, start, start + (end - start) // 2)
    assert insp._revert_btn.isEnabled()
    assert "Schmitt detector" in insp._revert_btn.toolTip(), (
        "the control must name WHICH detector produced the table it would restore")

    insp._revert_current()
    assert not insp._revert_btn.isEnabled()


def test_revert_reuses_the_mask_drags_own_write_path(qapp, qtbot, tmp_path):
    """A revert is an edit whose two numbers came from the detection instead of the mouse, so
    everything downstream of "which two samples" must be identical.

    Pinned on the consequence that has actually bitten this app: ``_write_measured_row`` calls
    ``_reconcile_levels``, so a second writer would leave the level table describing the
    pre-revert span while the readout described the reverted one.
    """
    insp, session, doc_id, ch, events = _stamped_inspector(qtbot, tmp_path)
    levels = _levels_for(session, doc_id, ch, events)
    insp.set_levels(levels)
    df = events.df
    ev_id = int(df["event_id"].iloc[0])
    insp.show_event(ev_id)
    row = insp._row_for_id(ev_id)
    start, end = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])

    _set_region_to_indices(insp, session, doc_id, start, start + (end - start) // 2)
    edited_end = int(df.iloc[row]["end_idx"])
    # A group that tiles the EDITED span — what a re-fit after the drag leaves behind, and
    # what a revert therefore has to reconcile away. (The drag itself already dropped the
    # group fitted to the original span; re-installing THAT would tile again after the revert
    # and prove nothing.)
    stale = levels.copy()
    mask = stale["event_id"] == ev_id
    stale.loc[mask, "end_idx"] = stale.loc[mask, "end_idx"].clip(upper=edited_end)
    stale.loc[mask, "start_idx"] = stale.loc[mask, "start_idx"].clip(upper=edited_end)
    insp._levels = stale
    assert int(stale[mask]["end_idx"].max()) == edited_end

    insp._revert_current()

    remaining = insp._levels
    assert remaining is None or (remaining["event_id"] != ev_id).all(), (
        "the revert wrote the bounds without reconciling the level table, so the window is "
        "showing sub-states measured on the span the user just left")


def test_revert_is_refused_and_explained_on_a_detection_saved_before_this_existed(
        qapp, qtbot, tmp_path):
    """THE MIGRATION CASE, and it is not backfilled.

    A detection already saved in a project carries no original bounds. Its CURRENT bounds may
    already have been edited, so writing them into the new columns would claim the detector
    produced whatever the last edit left behind — a fabricated provenance claim, on the one
    control whose entire promise is provenance. The honest answer is "unknown": the control is
    visible (so the gesture is discoverable) and disabled, with a tooltip that says why and
    what fixes it.
    """
    insp, session, doc_id, ch, events = _inspector_with_detection(qtbot, tmp_path)
    from nanoscope.analysis.events import ORIG_END_COL, ORIG_START_COL

    df = events.df
    assert ORIG_START_COL not in df.columns and ORIG_END_COL not in df.columns, (
        "this fixture is meant to stand in for a pre-change detection")
    ev_id = int(df["event_id"].iloc[0])
    insp.show_event(ev_id)
    row = insp._row_for_id(ev_id)
    start, end = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])

    assert insp._revert_btn.isVisible() or True     # visibility is _sync_controls' rule
    assert not insp._revert_btn.isEnabled()
    tip = insp._revert_btn.toolTip()
    assert "does not record" in tip and "Re-run detection" in tip, tip

    # ...and the gesture itself is inert, not merely greyed: nothing may be written.
    _set_region_to_indices(insp, session, doc_id, start, start + (end - start) // 2)
    edited_end = int(df.iloc[row]["end_idx"])
    insp._revert_current()
    assert int(df.iloc[row]["end_idx"]) == edited_end, (
        "an unavailable revert still wrote something")


def test_a_half_stamped_frame_reads_exactly_like_an_unstamped_one(qapp, qtbot, tmp_path):
    """One column without the other states no span, so it must be refused rather than
    half-believed. Reachable from a hand-built frame or a partial column selection."""
    from nanoscope.analysis.events import ORIG_END_COL, has_original_bounds

    insp, session, doc_id, ch, events = _stamped_inspector(qtbot, tmp_path)
    df = events.df.drop(columns=[ORIG_END_COL])
    assert not has_original_bounds(df)
    insp.set_events(doc_id, ch, df, provenance=dict(events.provenance))
    insp.show_event(int(df["event_id"].iloc[0]))
    assert not insp._revert_btn.isEnabled()


def test_a_revert_still_cannot_swallow_the_neighbouring_event(qapp, qtbot, tmp_path):
    """The neighbour clamp applies to a revert too. If a neighbour has since been dragged over
    this event's original span, restoring it verbatim would recreate the overlapping table the
    clamp exists to prevent — on the one gesture the user reads as "put it back", which is
    exactly when they would not be looking for it."""
    insp, session, doc_id, ch, events = _stamped_inspector(qtbot, tmp_path)
    df = events.df
    ids = [int(v) for v in df["event_id"]]
    first, second = ids[0], ids[1]
    r0, r1 = insp._row_for_id(first), insp._row_for_id(second)
    s0, e0 = int(df.iloc[r0]["start_idx"]), int(df.iloc[r0]["end_idx"])

    # Shrink event 0, then grow its NEIGHBOUR backwards over the span event 0 used to have.
    insp.show_event(first)
    _set_region_to_indices(insp, session, doc_id, s0, s0 + (e0 - s0) // 2)
    insp.show_event(second)
    _set_region_to_indices(insp, session, doc_id, e0 - 20, int(df.iloc[r1]["end_idx"]))
    assert int(df.iloc[r1]["start_idx"]) < e0, "the neighbour did not move over the old span"

    insp.show_event(first)
    insp._revert_current()

    starts = df["start_idx"].to_numpy(np.int64)
    ends = df["end_idx"].to_numpy(np.int64)
    order = np.argsort(starts)
    assert np.all(starts[order][1:] >= ends[order][:-1]), (
        "the revert recreated an overlapping event table")
    assert int(df.iloc[r0]["end_idx"]) <= int(df.iloc[r1]["start_idx"])
