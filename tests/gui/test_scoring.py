"""Detector-scoring + add-missed-event + training-export GUI tests (pytest-qt, offscreen).

Driven THROUGH THE APP PATH — a real :class:`MainWindow`, its :class:`AnalysisPanel`,
the owned :class:`MeasurePanel` + :class:`ScorePanel`, and the shared
:class:`TraceView` — exactly as a click would. The scenario closes the ML loop:

  1. run detection on a synthetic trace (the detector finds 5 events and MISSES one
     short blockade deliberately below the dwell floor);
  2. reject 2 of the detected events (they become spurious / FP);
  3. bracket the missed blockade with the Measure cursors and press *Mark selection as
     missed event* — it is characterized, appended as ``accepted + added`` (FN), and the
     result re-installed;
  4. the ScorePanel then reads tp=3 / fp=2 / fn=1 with precision 0.6, recall 0.75, F1 ⅔;
  5. the added event is in the table, shaded on the trace, and carries valid dwell/depth;
  6. the training export writes a CSV whose accepted rows are exactly the curated set —
     including the hand-added one.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.scoring import ADDED_COLUMN, build_training_frame, write_training_export
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem

FS = 100_000.0
N = 100_000
# Five clean 30 pA-deep squares the detector finds (all >= 200 samples = 2 ms dwell)...
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]
# ...plus ONE short 30-sample (0.3 ms) blockade the detector MISSES at min_dwell = 1 ms.
_MISSED = (60_000, 60_030)
_MIN_DWELL_S = 1e-3


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    x[_MISSED[0]:_MISSED[1]] -= 30.0          # the short one the detector will drop
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, *, name: str = "pore_p1.dat") -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(_MIN_DWELL_S)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)   # let the queued add_result -> resultsChanged -> table/overlay settle


def _bracket_missed(window: MainWindow, doc_id: str) -> None:
    """Turn on the Measure cursors and drop them exactly on the missed blockade."""
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[_MISSED[0]]), float(vals[_MISSED[1]]))


def _span_items(window: MainWindow, doc_id: str, channel: int) -> list[SpanOverlayItem]:
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    return [it for it in pane.plot.items if isinstance(it, SpanOverlayItem)]


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# the score readout, end to end
# ================================================================================================

def test_score_reflects_reject_two_and_add_one_missed(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    score = panel._score

    # The detector found exactly the five long events (the 0.3 ms one was dropped).
    df0 = panel.current_events_df()
    assert len(df0) == len(_EVENTS)

    # A fresh, all-accepted detection matches all 5 of its own detections -- and that is
    # NOT a score. Nothing has been rejected and nothing hand-added, so the truth set is
    # the detector's own output and the three ratios would be 1.000 by construction. The
    # counts are real and are shown; the ratios are withheld and say why.
    # (This block used to assert precision/recall/F1 == "1.000" -- it pinned the defect
    # of Medium-17: three 1.000s straight after detection, the most convincing numbers on
    # the screen, measuring nothing.)
    assert score.value_text("n_detected") == "5"
    assert score.value_text("tp") == "5"
    assert score.value_text("fp") == "0"
    assert score.value_text("fn") == "0"
    for key in ("precision", "recall", "f1"):
        assert score.value_text(key) == "not scored yet"
    assert not score.tautology_label.isHidden()
    assert "by construction" in score.tautology_label.text()

    # Reject two detected events -> they become spurious (FP).
    ids = [int(i) for i in df0["event_id"].to_numpy()]
    panel.set_event_accepted(ids[0], False)
    panel.set_event_accepted(ids[1], False)
    assert score.value_text("tp") == "3"
    assert score.value_text("fp") == "2"
    assert score.value_text("fn") == "0"

    # Bracket the missed blockade and press the Mark-missed button (the full wiring:
    # markMissedRequested -> AnalysisPanel.add_current_selection_as_event).
    _bracket_missed(window, doc_id)
    mp = panel.measure_panel
    assert mp._mark_missed_btn.isEnabled(), "button must arm with a detection + live cursors"
    with qtbot.waitSignal(window._session.resultsChanged, timeout=5_000):
        mp._mark_missed_btn.click()
    qtbot.wait(20)

    # Final score: 3 matched, 2 spurious, 1 missed -> P=0.6, R=0.75, F1=2/3.
    assert score.value_text("n_detected") == "5"     # the added event is NOT a detection
    assert score.value_text("n_truth") == "4"        # 3 kept + 1 added
    assert score.value_text("tp") == "3"
    assert score.value_text("fp") == "2"
    assert score.value_text("fn") == "1"
    assert score.value_text("precision") == "0.600"
    assert score.value_text("recall") == "0.750"
    assert score.value_text("f1") == "0.667"

    # And the pure score API agrees (used by tests / MainWindow).
    s = panel.current_score()
    assert (s.tp, s.fp, s.fn) == (3, 2, 1)
    assert s.missed_ids == [ids[-1] + 1]             # the fresh id handed to the added event


# ================================================================================================
# the added event: in the table, on the trace, with valid stats
# ================================================================================================

def test_added_missed_event_lands_in_table_overlay_and_has_valid_stats(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel

    before = len(panel.current_events_df())
    _bracket_missed(window, doc_id)
    new_id = panel.add_current_selection_as_event()
    qtbot.wait(20)
    assert new_id is not None

    df = panel.current_events_df()
    assert len(df) == before + 1

    # The row exists with the added/accepted markers set.
    row = df[df["event_id"].to_numpy() == new_id].iloc[0]
    assert bool(row[ADDED_COLUMN]) is True
    assert bool(row["accepted"]) is True

    # Its bounds are the bracketed span and its dwell/depth are real, finite numbers.
    assert int(row["start_idx"]) == _MISSED[0]
    assert int(row["end_idx"]) == _MISSED[1]
    assert np.isfinite(row["dwell_s"]) and float(row["dwell_s"]) == pytest.approx(30 / FS, rel=0.2)
    assert np.isfinite(row["depth_pA"]) and float(row["depth_pA"]) > 10.0

    # It shows in the live table model...
    table_df = panel._table.dataframe
    assert new_id in set(int(i) for i in table_df["event_id"].to_numpy())

    # ...and is shaded on the trace: the accepted-span overlay now covers all 6 events
    # (5 detected + the 1 added, none rejected here).
    spans = _span_items(window, doc_id, channel)
    assert len(spans) == 1
    assert spans[0].n_spans == before + 1


def test_added_event_persists_the_added_marker_through_save_load(window, qtbot, tmp_path):
    # The 'added' column lives in the df, so it must survive a project round-trip.
    from nanoscope.core.project import load_project, save_project

    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    _bracket_missed(window, doc_id)
    new_id = panel.add_current_selection_as_event()
    qtbot.wait(20)

    state = window._gather_project_state()
    out = tmp_path / "p.nanoscope"
    save_project(out, state)
    reloaded = load_project(out)
    ev = reloaded.results[0].events
    assert ADDED_COLUMN in ev.columns
    assert bool(ev.loc[ev["event_id"] == new_id, ADDED_COLUMN].iloc[0]) is True


def test_mark_missed_shortcut_action_works_with_measure_tab_hidden_and_keeps_view(
        window, qtbot, tmp_path):
    """The trace-first labeling loop must not depend on the Manual measurement page being open."""
    doc_id, _channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    _bracket_missed(window, doc_id)
    # Put another mode page on screen; the cursor deliberately remains live behind it.
    panel.detection_panel.mode_tabs.setCurrentIndex(0)
    assert panel.detection_panel.mode_tabs.currentIndex() == 0
    before_view = tuple(window._trace_view.anchor_plot().vb.viewRange()[0])
    before_bounds = window._trace_view.measure_bounds()
    before_n = len(panel.current_events_df())

    assert window._act_mark_missed.shortcut().toString() == "Ctrl+Shift+M"
    with qtbot.waitSignal(window._session.resultsChanged, timeout=5_000):
        window._act_mark_missed.trigger()
    qtbot.wait(20)

    assert len(panel.current_events_df()) == before_n + 1
    assert tuple(window._trace_view.anchor_plot().vb.viewRange()[0]) == pytest.approx(before_view)
    assert window._trace_view.measure_bounds() == before_bounds
    assert "reviewed training label" in window.statusBar().currentMessage()


def test_mark_missed_shortcut_without_detector_refuses_out_loud_and_writes_nothing(
        window, tmp_path):
    doc_id, _channel = _install(window, _doc(tmp_path))
    _bracket_missed(window, doc_id)
    assert window._act_mark_missed.isEnabled()

    window._act_mark_missed.trigger()

    assert window._analysis_panel.current_detection() is None
    assert "no source detector table" in window.statusBar().currentMessage().lower()


# ================================================================================================
# training export
# ================================================================================================

def test_training_export_accepted_rows_match_curation(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel

    df0 = panel.current_events_df()
    ids = [int(i) for i in df0["event_id"].to_numpy()]
    # Curate: reject two, add one missed. Curated truth = 3 kept + 1 added = 4 events.
    panel.set_event_accepted(ids[0], False)
    panel.set_event_accepted(ids[1], False)
    _bracket_missed(window, doc_id)
    new_id = panel.add_current_selection_as_event()
    qtbot.wait(20)

    res = panel.current_detection()
    out = write_training_export(
        res.payload.df, tmp_path / "train.csv",
        provenance=res.payload.provenance, source_file=doc_id, detector=res.module_id)

    loaded = pd.read_csv(out["data"])
    # Every exported row is accepted, and the exported set is exactly the curated truth.
    assert bool(loaded["accepted"].all())
    assert len(loaded) == 4
    # The hand-added missed event is in the export and flagged added.
    added_rows = loaded[loaded["added"] == True]                                  # noqa: E712
    assert len(added_rows) == 1
    assert int(added_rows.iloc[0]["start_idx"]) == _MISSED[0]
    assert new_id is not None

    # The menu action is gated on a detection and is live here.
    assert window._act_export_training.isEnabled()


def test_training_export_include_all_keeps_rejected(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    ids = [int(i) for i in panel.current_events_df()["event_id"].to_numpy()]
    panel.set_event_accepted(ids[0], False)

    res = panel.current_detection()
    frame_all = build_training_frame(res.payload.df, include_all=True)
    frame_acc = build_training_frame(res.payload.df, include_all=False)
    assert len(frame_all) == 5           # all detected events kept
    assert len(frame_acc) == 4           # the rejected one dropped


# ================================================================================================
# the Mark-missed button gate
# ================================================================================================

def test_mark_missed_button_requires_detection_and_cursors(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path))
    mp = window._analysis_panel.measure_panel

    # No detection yet: even with cursors up, the button is dead.
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[_MISSED[0]]), float(vals[_MISSED[1]]))
    assert not mp._mark_missed_btn.isEnabled()

    # After a detection lands, the same live selection arms the button.
    _run_detection(window, qtbot)
    _bracket_missed(window, doc_id)
    assert mp._mark_missed_btn.isEnabled()

    # Turning the cursors off disarms it again.
    mp.set_active(False)
    assert not mp._mark_missed_btn.isEnabled()
