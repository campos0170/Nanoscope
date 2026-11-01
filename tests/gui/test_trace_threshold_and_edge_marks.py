"""View > Show detection thresholds, and the marks on edges a person moved or drew.

Driven through a real :class:`MainWindow`: a synthetic recording is installed the way the load
pipeline would, detection runs through the window's own controller, and the assertions read
the live TraceView pane. SYNTHETIC ONLY -- no recording, model or project is opened from disk.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui

from nanoscope.analysis.detection_thresholds import RAIL_DRIFT, RAIL_ENTER, RAIL_EXIT
from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.analysis import OverlayHint, OverlayKind
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import (EDGE_CORRECTED, EDGE_HAND, THRESHOLD_TRACK_HEX,
                                    EdgeOverlayItem, SpanOverlayItem)

FS = 50_000.0
EVENTS = [(15_000 + i * 15_000, 15_000 + i * 15_000 + 300 + 50 * i) for i in range(6)]
N = 110_000


def _signal(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = -800.0 + 15.0 * np.sin(2 * np.pi * t / (N / FS)) + rng.normal(0.0, 4.0, N)
    for s, e in EVENTS:
        x[s:e] += 120.0          # blockades reduce |I|: the signed current moves UP at -800 pA
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str = "pore.dat", data: np.ndarray | None = None) -> TraceDocument:
    data = _signal() if data is None else data
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=data.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(data,))


@pytest.fixture
def win(qapp, qtbot, monkeypatch, tmp_path):
    # A leftover recovery snapshot would open a modal on construction.
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self, *a, **k: None,
                        raising=False)
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    doc = _doc(tmp_path)
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    qtbot.wait(20)
    yield w, doc_id
    w._analysis_controller.abort()


def _detect(w, qtbot, **form):
    panel = w._analysis_panel
    if form:
        panel.detection_panel.set_params(form)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(60)
    return panel._stored_detection()


def _pane(w, doc_id):
    return w._trace_view.pane_for_layer(f"{doc_id}#0")


def _rails(w, doc_id):
    """The threshold rails' INK curves on the pane (not their casings), by label."""
    view = w._trace_view
    out = {}
    for oid, label, *_rest in view._threshold_tracks.get(f"{doc_id}#0", []):
        items = view._overlays.get(oid, [])
        ink = [it for it in items if isinstance(it, pg.PlotDataItem)
               and it.opts["pen"].color().name() == QtGui.QColor(THRESHOLD_TRACK_HEX).name()]
        if ink:
            out[label] = ink[0]
    return out


def _edges(w, doc_id):
    return {it.label: it for it in _pane(w, doc_id).plot.items if isinstance(it, EdgeOverlayItem)}


def _install(w, qtbot, doc) -> str:
    """A second synthetic recording, installed and selected the way the fixture's is."""
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    qtbot.wait(20)
    return doc_id


# ================================================================================================
# B2a. The threshold lines.
# ================================================================================================
def test_off_by_default_and_toggling_draws_the_detectors_tracks(win, qtbot):
    w, doc_id = win
    res = _detect(w, qtbot)
    assert res is not None and len(res.payload.df) == len(EVENTS)
    assert not w._act_thresholds.isChecked()
    assert _rails(w, doc_id) == {}
    vb = _pane(w, doc_id).plot.vb
    before = [tuple(r) for r in vb.viewRange()]

    w._act_thresholds.setChecked(True)
    rails = _rails(w, doc_id)
    assert set(rails) == {RAIL_ENTER, RAIL_EXIT}
    prov = res.payload.provenance
    pol = int(prov["polarity"])
    assert pol == +1
    _x, enter = rails[RAIL_ENTER].getData()
    _x, exit_ = rails[RAIL_EXIT].getData()
    # The two rails are the same moving baseline, offset by the detector's own thresholds.
    np.testing.assert_allclose(enter - exit_, pol * (prov["threshold_in_pA"] - prov["threshold_out_pA"]),
                               atol=1e-6)
    assert float(enter.max() - enter.min()) > 10.0, "a flat line where the baseline sags"
    note = w._analysis_panel.detection_thresholds_note()
    assert note.startswith("Detection thresholds drawn")
    assert w._act_thresholds.statusTip() == note
    assert w.statusBar().currentMessage() == note
    # Appearance only: the user's zoom is untouched, and so is the table.
    assert [tuple(r) for r in vb.viewRange()] == before
    assert len(res.payload.df) == len(EVENTS)
    rail_pens = {lbl: it.opts["pen"].style() for lbl, it in rails.items()}
    assert rail_pens == {RAIL_ENTER: QtCore.Qt.DashLine, RAIL_EXIT: QtCore.Qt.DotLine}

    w._act_thresholds.setChecked(False)
    assert _rails(w, doc_id) == {}
    assert not w._trace_view.has_threshold_tracks(doc_id, 0)


def test_the_rails_follow_the_baseline_subtracted_view(win, qtbot):
    w, doc_id = win
    _detect(w, qtbot)
    w._act_thresholds.setChecked(True)
    x0, raw_enter = _rails(w, doc_id)[RAIL_ENTER].getData()
    base = np.full(N, -800.0, dtype=np.float64)
    w._trace_view.set_baseline_subtracted(doc_id, 0, base)
    x1, sub_enter = _rails(w, doc_id)[RAIL_ENTER].getData()
    np.testing.assert_allclose(sub_enter, raw_enter + 800.0, atol=1e-6)
    w._trace_view.set_baseline_subtracted(doc_id, 0, None)
    _x, back = _rails(w, doc_id)[RAIL_ENTER].getData()
    np.testing.assert_allclose(back, raw_enter, atol=1e-9)


def test_curation_does_not_rebuild_the_rails(win, qtbot, monkeypatch):
    w, doc_id = win
    res = _detect(w, qtbot)
    w._act_thresholds.setChecked(True)
    import nanoscope.analysis.detection_thresholds as dt
    calls = []
    real = dt.detection_threshold_tracks
    monkeypatch.setattr(dt, "detection_threshold_tracks",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    first = int(res.payload.df["event_id"].iloc[0])
    w._analysis_panel.set_event_accepted(first, False)
    w._analysis_panel._check_staleness(res)
    assert calls == []
    assert set(_rails(w, doc_id)) == {RAIL_ENTER, RAIL_EXIT}


def test_a_calibration_change_takes_the_rails_down_and_says_stale(win, qtbot):
    w, doc_id = win
    _detect(w, qtbot)
    w._act_thresholds.setChecked(True)
    assert _rails(w, doc_id)
    # Through the session's own signal, as the Calibration panel's Apply does: no direct call.
    w._session.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0, offset=0.0),
                             write_sidecar=False)
    qtbot.wait(20)
    assert _rails(w, doc_id) == {}
    assert "stale" in w._analysis_panel.detection_thresholds_note()


def test_a_high_pass_table_is_refused_with_its_reason(win, qtbot):
    w, doc_id = win
    _detect(w, qtbot, hp_enabled=True, hp_cutoff_hz=5.0)
    w._act_thresholds.setChecked(True)
    assert _rails(w, doc_id) == {}
    note = w._analysis_panel.detection_thresholds_note()
    assert note.startswith("Detection thresholds are not drawn") and "high-passed" in note


def test_no_table_says_so(win, qtbot):
    w, doc_id = win
    w._act_thresholds.setChecked(True)
    assert "no event table" in w._analysis_panel.detection_thresholds_note()


def test_a_low_passed_table_draws_its_rails_and_names_the_low_pass(win, qtbot):
    """The default form has no low-pass, so the panel's own rebuild of a LOW-PASSED detection
    signal (`_active_signal` under the table's provenance) is only reached from here."""
    w, doc_id = win
    res = _detect(w, qtbot, filter_cutoff_hz=5_000.0)
    assert float(res.payload.provenance["filter_cutoff_hz"]) == 5_000.0
    w._act_thresholds.setChecked(True)
    assert set(_rails(w, doc_id)) == {RAIL_ENTER, RAIL_EXIT}
    note = w._analysis_panel.detection_thresholds_note()
    assert "5 kHz low-pass" in note and "noisier" in note


def test_a_cusum_table_draws_one_dash_dot_drift_rail_and_no_h_line(win, qtbot):
    w, doc_id = win
    w._analysis_panel.detection_panel.set_detector("detect.cusum")
    res = _detect(w, qtbot)
    assert res.module_id == "detect.cusum" and len(res.payload.df) >= 1
    w._act_thresholds.setChecked(True)
    rails = _rails(w, doc_id)
    assert set(rails) == {RAIL_DRIFT}
    assert rails[RAIL_DRIFT].opts["pen"].style() == QtCore.Qt.DashDotLine
    prov = res.payload.provenance
    _x, drift = rails[RAIL_DRIFT].getData()
    # The drift level is the detector's baseline + polarity * k * sigma; its median sits there.
    assert float(np.median(drift)) - float(prov["baseline_median_pA"]) == pytest.approx(
        int(prov["polarity"]) * float(prov["cusum_k"]) * float(prov["noise_rms_pA"]), abs=2.0)
    assert "not a current" in w._analysis_panel.detection_thresholds_note()


def test_re_detecting_with_the_lines_on_draws_the_new_tables_thresholds(win, qtbot):
    """The cache is keyed on the table's provenance: a new run must not be served the old rails."""
    w, doc_id = win
    first = _detect(w, qtbot)
    w._act_thresholds.setChecked(True)
    assert set(_rails(w, doc_id)) == {RAIL_ENTER, RAIL_EXIT}
    second = _detect(w, qtbot, k_in=8.0)
    p1, p2 = first.payload.provenance, second.payload.provenance
    assert float(p2["threshold_in_pA"]) > float(p1["threshold_in_pA"])
    rails = _rails(w, doc_id)
    assert set(rails) == {RAIL_ENTER, RAIL_EXIT}
    _x, enter = rails[RAIL_ENTER].getData()
    _x, exit_ = rails[RAIL_EXIT].getData()
    np.testing.assert_allclose(
        enter - exit_, int(p2["polarity"]) * (p2["threshold_in_pA"] - p2["threshold_out_pA"]),
        atol=1e-6)
    assert "8 sigma" in w._analysis_panel.detection_thresholds_note()


def test_a_sample_rate_change_takes_the_rails_down_and_says_stale(win, qtbot):
    w, doc_id = win
    _detect(w, qtbot)
    w._act_thresholds.setChecked(True)
    assert _rails(w, doc_id)
    w._session.set_sample_rate_declaration(
        doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=25_000.0))
    qtbot.wait(20)
    assert _rails(w, doc_id) == {}
    assert "stale" in w._analysis_panel.detection_thresholds_note()


def test_closing_the_recording_takes_the_rails_down(win, qtbot):
    w, doc_id = win
    _detect(w, qtbot)
    w._act_thresholds.setChecked(True)
    assert w._trace_view.has_threshold_tracks(doc_id, 0)
    w._session.close_document(doc_id)
    qtbot.wait(20)
    assert not w._trace_view._threshold_tracks
    assert w._analysis_panel.detection_thresholds_note().startswith(
        "Detection thresholds are not drawn")


# ================================================================================================
# B2b. Corrected and hand-made edges.
# ================================================================================================
def test_a_fresh_detection_has_no_edge_marks_and_a_mask_drag_adds_one_revert_removes_it(win, qtbot):
    w, doc_id = win
    res = _detect(w, qtbot)
    assert _edges(w, doc_id) == {}
    df = res.payload.df
    w._open_event_inspector()
    insp = w._event_inspector
    ev = int(df["event_id"].iloc[2])
    insp.show_event(ev)
    row = insp._row_for_id(ev)
    s, e = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])
    raw = w._session.calibrated(doc_id, 0).data
    insp._apply_edited_bounds(row, s, e - 40, raw)
    qtbot.wait(20)
    marks = _edges(w, doc_id)
    assert set(marks) == {EDGE_CORRECTED}
    xs = w._session.x_axis(doc_id).values
    assert marks[EDGE_CORRECTED].n_edges == 1
    assert marks[EDGE_CORRECTED]._x[0] == pytest.approx(xs[e - 40])
    assert marks[EDGE_CORRECTED].pen().style() == QtCore.Qt.DashLine
    insp._revert_current()
    qtbot.wait(20)
    assert _edges(w, doc_id) == {}


def test_edit_undo_takes_the_mark_away_with_the_edit(win, qtbot):
    w, doc_id = win
    res = _detect(w, qtbot)
    df = res.payload.df
    w._open_event_inspector()
    insp = w._event_inspector
    ev = int(df["event_id"].iloc[1])
    insp.show_event(ev)
    row = insp._row_for_id(ev)
    s, e = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])
    insp._apply_edited_bounds(row, s + 30, e, w._session.calibrated(doc_id, 0).data)
    qtbot.wait(20)
    assert set(_edges(w, doc_id)) == {EDGE_CORRECTED}
    w._act_undo.trigger()
    qtbot.wait(50)
    assert _edges(w, doc_id) == {}


def test_a_hand_made_event_gets_dotted_marks_on_both_edges(win, qtbot):
    w, doc_id = win
    _detect(w, qtbot)
    # Through the one door every manual gesture uses (Ctrl+M, Measure's Add, the cursor keys).
    out = w._analysis_panel.add_manual_event(doc_id, 0, 2_000, 2_300)
    assert out.ok, out.message
    qtbot.wait(20)
    marks = _edges(w, doc_id)
    assert set(marks) == {EDGE_HAND}
    xs = w._session.x_axis(doc_id).values
    assert marks[EDGE_HAND].n_edges == 2
    assert marks[EDGE_HAND]._x.tolist() == pytest.approx([xs[2_000], xs[2_300]])
    assert marks[EDGE_HAND].pen().style() == QtCore.Qt.DotLine
    # The span shade is still ONE batched item: the marks add an item per style, not per row.
    assert sum(isinstance(it, SpanOverlayItem) for it in _pane(w, doc_id).plot.items) == 1


def test_rejecting_a_row_with_a_moved_edge_takes_its_mark_away(win, qtbot):
    """The marks qualify the ACCEPTED spans the shade draws; a rejected row has no shade."""
    w, doc_id = win
    res = _detect(w, qtbot)
    df = res.payload.df
    w._open_event_inspector()
    insp = w._event_inspector
    ev = int(df["event_id"].iloc[3])
    insp.show_event(ev)
    row = insp._row_for_id(ev)
    s, e = int(df.iloc[row]["start_idx"]), int(df.iloc[row]["end_idx"])
    insp._apply_edited_bounds(row, s + 25, e, w._session.calibrated(doc_id, 0).data)
    qtbot.wait(20)
    assert set(_edges(w, doc_id)) == {EDGE_CORRECTED}
    w._analysis_panel.set_event_accepted(ev, False)
    qtbot.wait(20)
    assert _edges(w, doc_id) == {}
    w._analysis_panel.set_event_accepted(ev, True)
    qtbot.wait(20)
    assert set(_edges(w, doc_id)) == {EDGE_CORRECTED}


def test_merge_with_next_marks_the_far_edge_it_moved_and_not_the_near_one(win, qtbot, tmp_path):
    """Through the Event Inspector's Merge with next and the window's own route to the panel."""
    w, _first = win
    data = _signal().astype(np.float64)
    data[100_000:100_300] += 120.0           # one passage, broken by a 40-sample return to the
    data[100_340:100_640] += 120.0           # open pore: detected as two, merged back by hand
    doc_id = _install(w, qtbot, _doc(tmp_path, "pair.dat", data.astype(np.float32)))
    res = _detect(w, qtbot)
    df = res.payload.df
    starts = df["start_idx"].to_numpy()
    assert len(df) == len(EVENTS) + 2
    ev = int(df["event_id"].iloc[int(np.argmin(np.abs(starts - 100_000)))])
    assert _edges(w, doc_id) == {}
    w._analysis_panel._merge.set_window_s(0.006)
    w._open_event_inspector()
    insp = w._event_inspector
    insp.show_event(ev)
    insp._on_merge_with_next()
    qtbot.wait(50)
    merged = w._analysis_panel._stored_detection().payload.df
    assert len(merged) == len(df) - 1, "the two fragments were not merged"
    row = merged[(merged["start_idx"] <= 100_010) & (merged["end_idx"] >= 100_630)]
    assert len(row) == 1
    marks = _edges(w, doc_id)
    assert set(marks) == {EDGE_CORRECTED} and marks[EDGE_CORRECTED].n_edges == 1
    xs = w._session.x_axis(doc_id).values
    assert marks[EDGE_CORRECTED]._x[0] == pytest.approx(xs[int(row["end_idx"].iloc[0])])


def test_an_event_caught_and_fitted_with_the_trace_keys_is_a_hand_made_table(win, qtbot):
    """The trace keys' edge fits place a NEW event's edges (C catches and fits both); they never
    move a detected row's. With no detection the row files into the Manual table: no threshold
    to draw, said in a sentence, and dotted hand-made marks at the fitted edges."""
    w, doc_id = win
    s0, e0 = EVENTS[0]
    w._trace_view.pointer_sample = lambda: (doc_id, 0, float((s0 + e0) // 2), 10.0)
    w._trace_keys.catch()
    assert w._trace_view.measure_bounds() is not None
    w._trace_keys.add()
    qtbot.wait(20)
    res = w._analysis_panel._stored_detection()
    assert res is not None and res.module_id == MANUAL_MODULE_ID
    row = res.payload.df.iloc[0]
    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    marks = _edges(w, doc_id)
    assert set(marks) == {EDGE_HAND}
    xs = w._session.x_axis(doc_id).values
    assert marks[EDGE_HAND]._x.tolist() == pytest.approx(
        [xs[int(row["start_idx"])], xs[int(row["end_idx"])]])
    w._act_thresholds.setChecked(True)
    assert _rails(w, doc_id) == {}
    note = w._analysis_panel.detection_thresholds_note()
    assert note.startswith("Detection thresholds are not drawn") and "hand-made" in note


def test_the_edge_item_paints_one_mark_per_pixel_column_and_only_what_is_visible():
    item = EdgeOverlayItem(np.array([5.0, 0.0, 1.0, 1.0004, 1.0008, 3.0, 9.0]),
                           label=EDGE_CORRECTED)
    assert item.visible_marks((0.5, 4.0), 0.001).tolist() == [1.0, 3.0]
    assert item.visible_marks((0.5, 4.0), 1e-5).tolist() == [1.0, 1.0004, 1.0008, 3.0]
    assert item.visible_marks((10.0, 11.0), 0.001).size == 0


def test_the_events_tab_says_what_the_marks_mean(win):
    w, _doc_id = win
    tip = w._analysis_panel.results_tabs.tabToolTip(0)
    assert "dashed edge" in tip and "dotted" in tip
    # ...and what a table saved before the detector's own bounds were recorded cannot show.
    assert "saved before" in tip and "edge fit" not in tip
