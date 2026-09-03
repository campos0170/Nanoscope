"""GUI tests for post-detection sub-event merge, wired through a real MainWindow (pytest-qt).

Three entry points, all mutating the ACTIVE detection's EventSet and flowing through the
EXISTING curation refresh (table / overlays / summary / inspector rebuild):

* WHOLE TRACE — the AnalysisPanel's "Merge all" collapses every run of fragments within the
  window; the event table + summary row counts drop.
* REGION — "Merge events in selection" collapses only the events under the on-trace Measure
  selection, leaving the rest.
* INSPECTOR — the Event Inspector's "Merge with next" stitches the browsed event to its
  forward run and advances.

Each path must leave a schema-valid EventSet and refresh the overlays/summary; all three are
gated when there is no active detection. A final AST guard pins that the new merge widget
imports no sibling gui panel — MainWindow stays the sole wirer.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import SCHEMA, EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 50_000
I0 = -1000.0
DEPTH = 300.0

# Three fragments ~20 samples apart (one translocation the detector split) + a far event.
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000)]
_FAR = (30_000, 30_300)
_PAIR = [(10_000, 10_300), (10_320, 10_600)]   # a single close pair, for the inspector test


# ================================================================================================
# builders
# ================================================================================================
def _signal(spans, *, n=N, depth=DEPTH, polarity=-1, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] += polarity * depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name="frag.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _install_detection(session: Session, doc_id: str, signal: np.ndarray) -> EventSet:
    """Detect on the flat-baseline signal (no detect-time merge) and store it as the active
    detection for (doc, channel 0), under the live calibration fingerprint (no stale banner)."""
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    es = EventSet(df, prov)
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=session.calibration(doc_id, 0).fingerprint(),
        payload=es))
    return es


def _open_with_detection(win: MainWindow, tmp_path, spans):
    signal = _signal(spans)
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    es = _install_detection(win._session, doc_id, signal)
    return doc_id, es


def _active_df(win: MainWindow):
    return win._analysis_panel.current_detection().payload.df


def _summary_n_events(win: MainWindow) -> int:
    return int(win._analysis_panel._stats.summary_table.item(0, 0).text())


def _assert_schema_valid(df) -> None:
    assert set(SCHEMA).issubset(df.columns)
    for col, (dtype, _u, _d) in SCHEMA.items():
        assert df[col].dtype == np.dtype(dtype) if dtype != "string" else True, col
    # event_id contiguous + unique
    ids = df["event_id"].to_numpy()
    assert list(ids) == list(range(len(df)))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _merge_all(win, qtbot, window_s: float):
    """Press 'Merge all' and wait for it to land.

    IT IS NO LONGER SYNCHRONOUS, and that is the fix these tests came with: the whole-
    trace merge costs 3.3-8.7 s on the owner's real recordings (it re-characterizes
    every surviving span), and it used to run on the GUI thread with no progress bar,
    no busy cursor and no status message. It now goes through the shared
    AnalysisController -- which is what puts the status-bar progress and the Cancel
    button in front of it -- so the outcome arrives on ``mergeFinished`` instead of as
    the return value of the click.
    """
    win._analysis_panel.merge_controls.set_window_s(window_s)
    with qtbot.waitSignal(win._analysis_panel.mergeFinished, timeout=30_000) as blocker:
        win._analysis_panel.merge_controls.merge_all_btn.click()
    qtbot.wait(20)
    return blocker.args[0]


# ================================================================================================
# (a) whole-trace merge
# ================================================================================================
def test_merge_all_collapses_fragments_and_counts_drop(win, qtbot, tmp_path):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    assert len(_active_df(win)) == 4
    assert _summary_n_events(win) == 4
    n_rows_before = win._analysis_panel._table._proxy.rowCount()
    assert n_rows_before == 4

    # Window of 1 ms (50 samples) bridges the ~20-sample gaps; the click drives the real path.
    report = _merge_all(win, qtbot, 1e-3)
    assert report.status == "ok"

    df = _active_df(win)
    assert len(df) == 2                                   # 3 fragments -> 1, far stays
    assert _summary_n_events(win) == 2                    # summary refreshed
    assert win._analysis_panel._table._proxy.rowCount() == 2   # table refreshed
    _assert_schema_valid(df)
    # the status line reports the before -> after collapse
    assert "→ 2 events" in win.statusBar().currentMessage()


def test_merge_all_is_a_noop_when_nothing_is_within_the_window(win, qtbot, tmp_path):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    # A tiny window (2 µs = 0.1 samples) merges nothing.
    assert _merge_all(win, qtbot, 2e-6).status == "noop"
    assert len(_active_df(win)) == 4
    assert "No sub-events" in win.statusBar().currentMessage()


# ================================================================================================
# (b) region merge — only the selected events
# ================================================================================================
def test_merge_region_collapses_only_the_selected_events(win, tmp_path):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    assert len(_active_df(win)) == 4

    # Turn on the Measure tool and drop a region over the three fragments (not the far event).
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(9_500 / FS, 11_500 / FS)
    assert win._trace_view.measure_bounds() is not None

    win._analysis_panel.merge_controls.merge_region_btn.click()

    df = _active_df(win)
    assert len(df) == 2                                   # 3 selected -> 1, far untouched
    # the far event survived with its original bounds
    far = df[df["start_idx"] > 20_000]
    assert len(far) == 1
    assert int(far["end_idx"].iloc[0]) == int(_FAR[1])
    _assert_schema_valid(df)
    assert win._analysis_panel.current_detection().payload.provenance["merged"]["rule"] == "region"


def test_merge_region_without_a_selection_tells_the_user_to_select(win, tmp_path):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    # Measure tool OFF -> no selection region -> the button reports "select a region first".
    assert win._trace_view.measure_bounds() is None
    win._analysis_panel.merge_controls.merge_region_btn.click()
    assert len(_active_df(win)) == 4                      # unchanged
    assert "Select a region" in win.statusBar().currentMessage()


# ================================================================================================
# (c) inspector "Merge with next"
# ================================================================================================
def test_inspector_merge_with_next_merges_the_current_pair(win, tmp_path):
    _open_with_detection(win, tmp_path, _PAIR + [_FAR])
    assert len(_active_df(win)) == 3

    win._open_event_inspector()
    insp = win._event_inspector
    first_id = insp.current_event_id()
    assert first_id is not None

    win._analysis_panel.merge_controls.set_window_s(1e-3)   # 50-sample window bridges the pair
    insp._merge_next_btn.click()

    df = _active_df(win)
    assert len(df) == 2                                   # the close pair collapsed; far stays
    _assert_schema_valid(df)
    # the browser advanced off the merged event to the following (far) event
    assert insp.current_event_id() is not None
    assert insp.current_event_id() != first_id
    # the inspector's shared frame is the SAME (new) df the panel now holds
    assert insp._df is df


def test_inspector_merge_with_next_on_a_lone_event_is_a_noop(win, tmp_path):
    _open_with_detection(win, tmp_path, _PAIR + [_FAR])
    win._open_event_inspector()
    insp = win._event_inspector
    # Browse to the FAR event (nothing within the window ahead of it) and merge-with-next.
    far_id = int(_active_df(win).sort_values("start_idx")["event_id"].iloc[-1])
    insp.show_event(far_id)
    win._analysis_panel.merge_controls.set_window_s(1e-3)
    insp._merge_next_btn.click()
    assert len(_active_df(win)) == 3                      # unchanged
    assert "No sub-events" in win.statusBar().currentMessage()


# ================================================================================================
# I10 — the occupancy floor, wired through the real widgets.
#
# The user's report was "the merging is not giving good results", and this is the shape of
# it: at the 10-15 ms window the merge panel's own text recommends, an ordinary train of
# well-separated events collapsed into ONE row (dwell 312.40 ms, depth 0.70 pA against a
# 300 pA ground truth, accepted, flag ''), and one drag over two events 600 ms apart did the
# same thing on the region path. Every entry point now refuses a span that is mostly open
# pore, and REPORTS the refusal — a merge that silently declines is the same complaint again.
# ================================================================================================

#: 40 clean 0.4 ms events spaced 8 ms apart. Nothing here is shattered; there is nothing to
#: stitch. A 10 ms window groups all 40 transitively.
_TRAIN = [(500 + i * 400, 520 + i * 400) for i in range(40)]


def test_merge_all_refuses_a_train_of_clean_events(win, qtbot, tmp_path):
    _open_with_detection(win, tmp_path, _TRAIN)
    assert len(_active_df(win)) == 40

    report = _merge_all(win, qtbot, 1e-2)          # the RECOMMENDED window
    assert report.status == "low_occupancy"
    assert report.n_refused_low_occupancy == 1
    df = _active_df(win)
    assert len(df) == 40                           # the table is untouched
    assert float(df["dwell_s"].max()) < 1e-3       # no 312 ms row
    assert float(df["depth_pA"].median()) == pytest.approx(DEPTH, abs=15.0)
    _assert_schema_valid(df)
    # The status LINE for this status is MainWindow's `_post_merge_status` branch (that file
    # is owned elsewhere); the report is the contract this panel is responsible for.


def test_a_refusal_is_not_reported_as_a_no_op(win, qtbot, tmp_path):
    """"No sub-events within 10 ms to merge" would be false about a table where the runs
    WERE within the window and were declined — and it is the sentence that made the user
    conclude the merge does nothing."""
    _open_with_detection(win, tmp_path, _TRAIN)
    refused = _merge_all(win, qtbot, 1e-2)
    assert refused.status == "low_occupancy" and refused.n_refused_low_occupancy > 0

    # ...while a genuinely empty window is still a plain no-op.
    noop = _merge_all(win, qtbot, 2e-6)
    assert noop.status == "noop" and noop.n_refused_low_occupancy == 0


def test_merge_all_still_merges_real_fragments_and_counts_the_refusals(win, qtbot, tmp_path):
    """A shattered event and a train in the same table: the fragments merge, the train is
    refused, and the report carries both halves."""
    train = [(20_000 + i * 400, 20_020 + i * 400) for i in range(20)]
    _open_with_detection(win, tmp_path, _FRAGMENTS + train)
    assert len(_active_df(win)) == 23

    report = _merge_all(win, qtbot, 1e-2)
    assert report.status == "ok"
    assert (report.n_before, report.n_after) == (23, 21)     # 3 fragments -> 1
    assert report.n_refused_low_occupancy == 1
    _assert_schema_valid(_active_df(win))


def test_merge_region_refuses_a_mostly_open_pore_selection(win, tmp_path):
    """Two 4 ms events 600 ms apart, one drag over both: it used to become a single 604 ms
    event of depth ~0.05 pA."""
    _open_with_detection(win, tmp_path, [(5_000, 5_200), (35_000, 35_200)])
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(4_000 / FS, 36_000 / FS)
    assert win._trace_view.measure_bounds() is not None

    report = win._analysis_panel.merge_selection_subevents()
    assert report.status == "low_occupancy"
    assert report.n_refused_low_occupancy == 1
    df = _active_df(win)
    assert len(df) == 2                                      # untouched
    assert float(df["dwell_s"].max()) < 1e-2


def test_inspector_merge_with_next_stops_at_the_occupancy_floor(win, tmp_path):
    """The forward-run walk was unbounded: on a train, one click ran to the end of the
    recording. It now takes only the events that are really one translocation."""
    _open_with_detection(win, tmp_path, _TRAIN)
    win._open_event_inspector()
    insp = win._event_inspector
    first_id = int(_active_df(win).sort_values("start_idx")["event_id"].iloc[0])
    insp.show_event(first_id)

    report = win._analysis_panel.merge_event_with_next(first_id, 1e-2)
    assert report.status == "low_occupancy"
    assert report.n_refused_low_occupancy == 1
    assert len(_active_df(win)) == 40                        # nothing swallowed

    # ...and the same button still merges a genuinely shattered pair.
    _open_with_detection(win, tmp_path, _PAIR + [_FAR])
    pair_id = int(_active_df(win).sort_values("start_idx")["event_id"].iloc[0])
    ok = win._analysis_panel.merge_event_with_next(pair_id, 1e-3)
    assert ok.status == "ok" and len(_active_df(win)) == 2


# ================================================================================================
# I10 (5) — a Measure region on ANOTHER recording must not be used on this one.
#
# `measure_bounds()` returns SAMPLE INDICES with no recording attached, and the Measure
# region does not have to be on the app's selected recording. MainWindow's "spectrum of the
# selection" action already guards this with `measure_target()`; the two panel entry points
# that consume the same bounds did not.
# ================================================================================================
def test_merge_in_selection_ignores_a_region_on_another_recording(win, tmp_path, monkeypatch):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(9_500 / FS, 11_500 / FS)
    assert win._trace_view.measure_bounds() is not None

    # The bounds stay; only the recording they are indices INTO changes.
    monkeypatch.setattr(win._trace_view, "measure_target", lambda: ("some-other-doc", 0))
    report = win._analysis_panel.merge_selection_subevents()
    assert report.status == "no_selection"
    assert len(_active_df(win)) == 4                          # nothing merged


def test_adding_a_missed_event_ignores_a_region_on_another_recording(win, tmp_path, monkeypatch):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(20_000 / FS, 20_400 / FS)
    assert win._analysis_panel.add_current_selection_as_event() is not None   # same target: adds
    assert len(_active_df(win)) == 5

    monkeypatch.setattr(win._trace_view, "measure_target", lambda: ("some-other-doc", 0))
    assert win._analysis_panel.add_current_selection_as_event() is None
    assert len(_active_df(win)) == 5                          # and nothing was appended


# ================================================================================================
# I9 residual — the window spinbox has no fake "off"
# ================================================================================================
def test_the_merge_window_cannot_be_set_to_a_fake_off(win, tmp_path):
    """0.0 ms looked like the sibling DetectionPanel's "0 = off" sentinel and was not one:
    at a 0-sample window ``gap <= 0`` still merged exactly-touching rows. The spinbox now has
    a positive minimum, so the control cannot express an "off" that does not exist."""
    from nanoscope.gui.merge_panel import MIN_WINDOW_MS

    mc = win._analysis_panel.merge_controls
    assert MIN_WINDOW_MS > 0.0
    assert mc.window_ms.minimum() == pytest.approx(MIN_WINDOW_MS)
    mc.set_window_s(0.0)
    assert mc.window_s() == pytest.approx(MIN_WINDOW_MS * 1e-3)
    assert mc.window_s() > 0.0


# ================================================================================================
# gating: no active detection
# ================================================================================================
def test_merge_controls_disabled_without_a_detection(win, tmp_path):
    signal = _signal(_FRAGMENTS)
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)                          # a current channel, but NO detection
    mc = win._analysis_panel.merge_controls
    assert not mc.merge_all_btn.isEnabled()
    assert not mc.merge_region_btn.isEnabled()
    # the engine methods refuse honestly too
    assert win._analysis_panel.merge_all_subevents(1e-3).status == "no_detection"
    assert win._analysis_panel.merge_selection_subevents().status == "no_detection"


def test_merge_controls_enable_once_a_detection_is_shown(win, tmp_path):
    _open_with_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    mc = win._analysis_panel.merge_controls
    assert mc.merge_all_btn.isEnabled()
    assert mc.merge_region_btn.isEnabled()


# ================================================================================================
# the no-cross-panel-import invariant, for the NEW merge widget (main_window is the sole wirer)
# ================================================================================================
def _relative_sibling_imports(module) -> list[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    return [n.module for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module]


def test_merge_panel_imports_no_sibling_gui_panel():
    """The MergePanel is a pure control group: it may import Qt only, never another gui panel.
    MainWindow stays the sole importer that wires the merge signals to the session + refresh."""
    import nanoscope.gui.merge_panel as merge_mod

    assert _relative_sibling_imports(merge_mod) == []


def test_merge_panel_stays_qt_only():
    import nanoscope.gui.merge_panel as merge_mod

    tree = ast.parse(pathlib.Path(merge_mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "gui" not in mod.split("."), f"merge_panel must not import gui: {mod}"
            assert not mod.startswith("nanoscope.analysis"), "merge_panel stays view-only (no analysis)"


# ================================================================================================
# N2 — the merge must re-characterize on the trace the DETECTOR measured on.
#
# ``DetectSchmitt`` conditions the signal (``preprocess_for_detection``) and ``detect_events``
# then applies its own low-pass, so every depth / baseline / rise in the table was read off
# ``lowpass(preprocess_for_detection(raw))``. ``AnalysisPanel._active_signal`` handed the merge
# ``ch.data`` — the RAW channel — so a merged row's numbers came from a different trace than the
# rows either side of it. Measured on the user's -600 mV recording over 349 merged spans:
# ``frac_depth`` off by 3.3% at the median at the default 10 kHz Bessel, 10.7% at 2 kHz, and one
# row changed sign.
# ================================================================================================

def _drifting_signal(spans, *, n=N, depth=DEPTH, seed=3) -> np.ndarray:
    """Fragments on a DRIFTING, mains-contaminated open pore.

    A flat, clean baseline cannot show N2: with nothing for the preprocessing to remove and a
    cutoff above the noise, the conditioned trace and the raw channel are the same array. That
    is exactly why the tests above stayed green.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = (I0
         + 120.0 * np.sin(2 * np.pi * 0.35 * t)          # slow open-pore wander
         + 25.0 * np.sin(2 * np.pi * 60.0 * t)           # mains hum
         + rng.normal(0.0, 8.0, n))
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


#: Fragments separated widely enough that the 5 kHz low-pass does NOT bridge them, so the
#: detector still reports them as separate rows and the merge has real work to do.
_WIDE_FRAGMENTS = [(10_000, 10_600), (10_900, 11_500), (11_800, 12_400)]
_WIDE_WINDOW_S = 1e-2

_DETECT_PARAMS = {
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.05,
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 2, "notch_q": 30.0,
    "polarity": None, "k_in": 5.0, "k_out": 2.0, "min_dwell_s": 1e-4,
    "filter_cutoff_hz": 5_000.0, "filter_kind": "bessel", "filter_order": 8,
}


def _open_with_real_detection(win: MainWindow, tmp_path, spans):
    """Open a drifting recording and run the REAL DetectSchmitt module on it, so the stored
    result carries the preprocessing provenance ``detection_signal`` reads back."""
    from nanoscope.analysis.context import SimpleAnalysisContext
    from nanoscope.analysis.modules import DetectSchmitt

    signal = _drifting_signal(spans)
    doc = _doc(tmp_path, signal, name="drift.dat")
    doc_id = win._session.add_document(doc)
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)

    ch = win._session.calibrated(doc_id, 0)
    res = DetectSchmitt().run(
        SimpleAnalysisContext(channel=ch, meta=doc.meta, params=dict(_DETECT_PARAMS)))
    win._session.add_result(res.__class__(
        module_id=res.module_id, module_version=res.module_version, doc_id=doc_id,
        channel=0, params=dict(_DETECT_PARAMS),
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=res.payload))
    return doc_id, signal, res.payload


def test_merge_re_characterizes_on_the_trace_the_detector_measured_on(win, qtbot, tmp_path):
    """The merged row must equal a merge over ``detection_signal``, and must NOT equal a
    merge over the raw channel."""
    from nanoscope.analysis.merge import merge_by_gap
    from nanoscope.analysis.modules import detection_signal

    doc_id, signal, es = _open_with_real_detection(
        win, tmp_path, _WIDE_FRAGMENTS + [_FAR])
    spans_before = set(zip(es.df["start_idx"], es.df["end_idx"]))
    assert len(es.df) >= 4

    conditioned = detection_signal(signal, FS, es.provenance)
    assert float(np.abs(np.asarray(conditioned, np.float64)
                        - np.asarray(signal, np.float64)).max()) > 1.0

    want, n_want = merge_by_gap(es, conditioned, FS, max_gap_s=_WIDE_WINDOW_S)
    raw_way, _ = merge_by_gap(es, signal, FS, max_gap_s=_WIDE_WINDOW_S)
    assert n_want >= 1, "fixture must actually merge something"

    _merge_all(win, qtbot, _WIDE_WINDOW_S)
    got = _active_df(win)

    assert len(got) == len(want)
    cols = ["depth_pA", "frac_depth", "baseline_pA", "ecd_pA_s", "snr"]
    np.testing.assert_allclose(got[cols].to_numpy(), want[cols].to_numpy(), rtol=0, atol=0)

    # ...and on the MERGED rows -- the only ones re-characterized -- the raw-channel
    # answer is a DIFFERENT measurement, so this is not vacuous.
    merged = np.array([(int(s), int(e)) not in spans_before
                       for s, e in zip(want["start_idx"], want["end_idx"])])
    assert merged.any()
    assert not np.allclose(want["frac_depth"].to_numpy()[merged],
                           raw_way["frac_depth"].to_numpy()[merged], rtol=1e-6, atol=0)


def test_the_conditioned_trace_is_cached_between_merges(win, tmp_path):
    """The merge entry points run on the GUI thread and conditioning a real recording costs
    ~0.12 s; the memo must survive a second merge and invalidate on a new detection."""
    doc_id, signal, es = _open_with_real_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    panel = win._analysis_panel

    first, _ = panel._active_signal(es.provenance)
    again, _ = panel._active_signal(es.provenance)
    assert again is first                                   # same object: memoised

    moved = dict(es.provenance)
    moved["filter_cutoff_hz"] = 2_000.0
    other, _ = panel._active_signal(moved)
    assert other is not first
    assert not np.array_equal(other, first)


def test_a_hand_added_event_is_measured_on_the_same_trace_as_the_detected_ones(win, tmp_path):
    """A missed event added by hand lands in the same table; measuring it on ``ch.data``
    would put a differently-derived row next to the detected ones."""
    from nanoscope.analysis.modules import detection_signal

    doc_id, signal, es = _open_with_real_detection(win, tmp_path, _FRAGMENTS + [_FAR])
    panel = win._analysis_panel
    conditioned = detection_signal(signal, FS, es.provenance)

    start, end = 20_000, 20_400
    panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(start / FS, end / FS)
    bounds = win._trace_view.measure_bounds()
    assert bounds is not None
    start, end = int(bounds[0]), int(bounds[1])
    new_id = panel.add_current_selection_as_event()
    assert new_id is not None

    row = _active_df(win).set_index("event_id").loc[new_id]
    want = panel._characterize_missed(
        np.asarray(conditioned), FS, start, end, dict(es.provenance), "pA", "r", "p")
    assert float(row["depth_pA"]) == pytest.approx(float(want["depth_pA"]), rel=0, abs=0)
    assert float(row["snr"]) == pytest.approx(float(want["snr"]), rel=0, abs=0)

    raw_way = panel._characterize_missed(
        np.asarray(signal), FS, start, end, dict(es.provenance), "pA", "r", "p")
    assert float(want["frac_depth"]) != pytest.approx(float(raw_way["frac_depth"]), rel=1e-6)
