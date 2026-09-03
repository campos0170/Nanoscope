"""ADVERSARIAL, independent verification of Session save/load through the REAL app.

Assumes the feature is overstated and tries to catch it lying. Differences from the
authors' own test_project_io.py (so this is a genuinely independent check):

  * the event-boundary EDIT goes through the REAL EventInspectorWindow resize path
    (open the inspector, drag its LinearRegion, let sigRegionChangeFinished fire),
    not a hand-written df.loc mutation;
  * TWO events are rejected (not one);
  * a non-identity calibration with a NON-ZERO offset is used and BOTH gain AND
    offset are asserted to round-trip exactly;
  * the .nanoscope zip is cracked open and its manifest is proven to hold PATH
    REFERENCES with no raw signal, and the file is asserted small;
  * NO re-detection is proven behaviourally (curation + edited bounds survive, which
    a fresh detection could never reproduce), and the analysis controller is asserted
    idle across the whole restore;
  * a real ABF is round-tripped and the project size reported.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from conftest import live_doc_id
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem

FS = 100_000.0
N = 100_000
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]

_REAL_ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_"
                 "process _ -1000 mV_ 385-400 s time trace.abf")


def _event_trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(win: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = win._session
    doc_id = session.add_document(doc)
    win._create_default_layers(doc)
    ch = next(c.index for c in doc.meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    win._on_selection(doc_id, ch)
    return doc_id, ch


def _detect(win: MainWindow, qtbot, min_dwell_s: float = 5e-4) -> None:
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _span_items(pane) -> list[SpanOverlayItem]:
    """The DETECTION span overlay on a pane.

    Filtered by the hint's label: the manual-measurement panel now shades its own saved
    measurements with a second (differently coloured) SpanOverlayItem on the same pane, so
    "the only SpanOverlayItem" is no longer the same statement as "the detection's spans".
    """
    return [it for it in pane.plot.items
            if isinstance(it, SpanOverlayItem) and it.label == "events"]


def _fresh_window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _edit_event0_via_inspector(win: MainWindow, qtbot, doc_id: str, channel: int,
                               new_start: int, new_end: int) -> None:
    """Edit event 0's [start,end] through the REAL inspector resize path: open the
    window, browse to event 0, and set the LinearRegion — letting the widget's own
    sigRegionChangeFinished -> _on_region_edited -> eventEdited chain fire."""
    win._open_event_inspector()
    insp = win._event_inspector
    assert insp is not None
    insp.show_event(0)
    assert insp.current_event_id() == 0
    meta = win._session.document(doc_id).meta
    t0, fs = meta.t0_s, meta.sample_rate_hz
    x0 = t0 + new_start / fs
    x1 = t0 + new_end / fs
    # A real, non-programmatic region change (outside _render_event's _syncing fence).
    with qtbot.waitSignal(insp.eventEdited, timeout=5_000):
        insp._region.setRegion([x0, x1])
    qtbot.wait(10)


def _make_source_project(qtbot, tmp_path) -> dict:
    csv_a, csv_b = tmp_path / "pore_a.csv", tmp_path / "pore_b.csv"
    arr_a, arr_b = _event_trace(seed=0), _event_trace(seed=1)
    _write_csv(csv_a, arr_a)
    _write_csv(csv_b, arr_b)

    win = _fresh_window(qtbot)
    doc_a, ch_a = _install(win, _doc(csv_a, arr_a))
    doc_b, ch_b = _install(win, _doc(csv_b, arr_b))
    sess = win._session

    # Non-identity calibration WITH non-zero offset on the current channel.
    sess.set_calibration(doc_a, ch_a, ChannelCalibration(0.5, 3.0, "nA", Provenance.MEASURED))
    sess.set_calibration(doc_b, ch_b, ChannelCalibration(2.0, 0.0, "pA", Provenance.MEASURED))

    # Detection A, then curate through the real widgets.
    win._on_selection(doc_a, ch_a)
    _detect(win, qtbot)
    panel = win._analysis_panel
    df = sess.results.get(doc_a, ch_a, "detect.schmitt").payload.df
    assert len(df) == len(_EVENTS)

    # Reject TWO events (1 and 3) through the table model.
    panel.set_event_accepted(1, False)
    panel.set_event_accepted(3, False)

    # Edit event 0's bounds through the REAL inspector resize path.
    new_start, new_end = _EVENTS[0][0] + 17, _EVENTS[0][1] - 9
    _edit_event0_via_inspector(win, qtbot, doc_a, ch_a, new_start, new_end)

    i0 = df.index[df["event_id"].to_numpy() == 0][0]
    assert int(df.loc[i0, "start_idx"]) == new_start, "inspector did not rewrite start_idx"
    assert int(df.loc[i0, "end_idx"]) == new_end, "inspector did not rewrite end_idx"
    edited_dwell = float(df.loc[i0, "dwell_s"])
    df_a = df.copy()

    # Detection B (uncurated).
    win._on_selection(doc_b, ch_b)
    _detect(win, qtbot)
    df_b = sess.results.get(doc_b, ch_b, "detect.schmitt").payload.df.copy()
    assert len(df_b) == len(_EVENTS)

    # Layer view state + non-default view mode.
    sess.set_layer_color(f"{doc_b}#{ch_b}", "#E57373")
    sess.set_layer_visible(f"{doc_b}#{ch_b}", False)
    sess.set_layer_time_offset(f"{doc_b}#{ch_b}", 0.25)
    sess.set_layer_time_offset(f"{doc_a}#{ch_a}", 0.1)
    sess.set_view_mode(ViewMode.STACKED)

    # Two manual measurements through the real cursor tool.
    win._on_selection(doc_a, ch_a)
    mp = panel.measure_panel
    mp.set_active(True)
    vals = sess.x_axis(doc_a).values
    for s, e in _EVENTS[:2]:
        win._trace_view.set_measure_region(float(vals[s]), float(vals[e]))
        qtbot.wait(5)
        assert mp.current_measurement() is not None
        assert mp.add_current() is True
    meas = mp.measurements_df.copy()
    assert len(meas) == 2

    proj = tmp_path / "session.nanoscope"
    win._write_project(str(proj))
    assert proj.exists()

    win._analysis_controller.abort()
    win._controller.abort()

    return {
        "proj": str(proj), "csv_a": str(csv_a), "csv_b": str(csv_b),
        "doc_a": doc_a, "doc_b": doc_b, "ch_a": ch_a, "ch_b": ch_b,
        "df_a": df_a, "df_b": df_b, "meas": meas,
        "n_acc_a": int(df_a["accepted"].to_numpy().sum()),
        "new_start": new_start, "new_end": new_end, "edited_dwell": edited_dwell,
    }


def test_zip_holds_only_path_references_and_is_small(qapp, qtbot, tmp_path):
    ctx = _make_source_project(qtbot, tmp_path)
    proj = ctx["proj"]

    assert zipfile.is_zipfile(proj)
    with zipfile.ZipFile(proj) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        parquets = [n for n in names if n.endswith(".parquet")]
        assert any(n.startswith("results/") for n in parquets), parquets
        manifest = json.loads(zf.read("manifest.json"))

    # Documents are PATH REFERENCES — the abspath + loaded range, no signal payload.
    docs = manifest["documents"]
    assert {d["path"] for d in docs} == {ctx["csv_a"], ctx["csv_b"]}
    for d in docs:
        # `source` (v4) is a FINGERPRINT of the file — four scalars — not its signal: what
        # the path CONTAINED when the project was saved, so a re-export over the same
        # filename cannot pass for the original. The size assertions below are what keep it
        # honest about staying a reference.
        assert set(d.keys()) <= {"path", "loaded_range", "source"}
        assert set(d["source"]) == {"n_samples", "size", "digest", "mtime_ns"}

    # No raw signal anywhere: the two source arrays are ~100k float samples; the whole
    # project must be far smaller than even one uncompressed channel (~400 KB) + the CSV.
    size = os.path.getsize(proj)
    csv_size = os.path.getsize(ctx["csv_a"])
    assert size < csv_size, f"project {size} B not smaller than one source CSV {csv_size} B"
    assert size < 200_000, f"project unexpectedly large: {size} B"

    # The manifest JSON must contain no numpy arrays / giant lists (provenance coerced).
    raw = json.dumps(manifest)
    assert '"current"' not in raw or len(raw) < 50_000
    print(f"\n[zip] members={sorted(names)} size={size}B csv={csv_size}B")


def test_full_roundtrip_reuses_stored_events(qapp, qtbot, tmp_path):
    ctx = _make_source_project(qtbot, tmp_path)
    doc_a, doc_b, ch_a, ch_b = ctx["doc_a"], ctx["doc_b"], ctx["ch_a"], ctx["ch_b"]

    # Delete calibration sidecars: a restored calibration can ONLY come from the project.
    for sidecar in tmp_path.glob("*.npcal.json"):
        sidecar.unlink()

    w2 = _fresh_window(qtbot)

    # Prove NO detection re-runs: spy the analysis controller's finished signal.
    detections_ran = []
    w2._analysis_controller.finished.connect(lambda *_: detections_ran.append(1))

    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(ctx["proj"])
    qtbot.wait(50)
    sess = w2._session

    # documents reopened by path — as NEW LOADS with new ids (core/identity.py), which is
    # why they are re-resolved through the SOURCE rather than carried across the boundary.
    doc_a, doc_b = live_doc_id(sess, doc_a), live_doc_id(sess, doc_b)
    assert doc_a is not None
    assert doc_b is not None

    # calibration gain + OFFSET + units restored exactly
    ca = sess.calibration(doc_a, ch_a)
    assert ca.gain == 0.5 and ca.offset == 3.0 and ca.target_units == "nA"
    cb = sess.calibration(doc_b, ch_b)
    assert cb.gain == 2.0 and cb.offset == 0.0 and cb.target_units == "pA"

    # layers + view mode
    assert sess.view_mode is ViewMode.STACKED
    lb = sess.layer(f"{doc_b}#{ch_b}")
    assert lb is not None and lb.color == "#E57373" and lb.visible is False
    assert lb.time_offset_s == pytest.approx(0.25)
    la = sess.layer(f"{doc_a}#{ch_a}")
    assert la is not None and la.time_offset_s == pytest.approx(0.1)

    # EventSet A: same rows, same accepted vector, edited bounds + dwell preserved
    ra = sess.results.get(doc_a, ch_a, "detect.schmitt")
    assert ra is not None
    assert len(ra.payload.df) == len(_EVENTS)
    assert_frame_equal(ra.payload.df, ctx["df_a"])
    assert list(ra.payload.df["accepted"]) == list(ctx["df_a"]["accepted"])
    i0 = ctx["df_a"].index[0]
    assert int(ra.payload.df.loc[i0, "start_idx"]) == ctx["new_start"]
    assert int(ra.payload.df.loc[i0, "end_idx"]) == ctx["new_end"]
    assert float(ra.payload.df.loc[i0, "dwell_s"]) == pytest.approx(ctx["edited_dwell"])

    rb = sess.results.get(doc_b, ch_b, "detect.schmitt")
    assert rb is not None
    assert_frame_equal(rb.payload.df, ctx["df_b"])

    # event table + batched span overlay repopulated; n_spans == accepted count
    panel = w2._analysis_panel
    assert panel._sel == (doc_a, ch_a)
    assert panel._table._proxy.rowCount() == len(_EVENTS)
    pane = w2._trace_view.pane_for_layer(f"{doc_a}#{ch_a}")
    assert pane is not None
    spans = _span_items(pane)
    assert len(spans) == 1
    assert spans[0].n_spans == ctx["n_acc_a"] == 3

    # not stale (fingerprint matches restored calibration)
    assert panel._stale.isHidden()

    # measurements
    assert_frame_equal(panel.measure_panel.measurements_df, ctx["meas"])

    # NO detection re-ran during the entire restore
    assert detections_ran == [], f"detection re-ran {len(detections_ran)} time(s) on restore"

    w2._analysis_controller.abort()
    w2._controller.abort()


def test_missing_source_restores_the_rest_with_warning(qapp, qtbot, tmp_path):
    ctx = _make_source_project(qtbot, tmp_path)

    # RENAME (not delete) one source file so it can no longer be found by path.
    renamed = Path(ctx["csv_b"]).with_name("pore_b_moved.csv")
    Path(ctx["csv_b"]).rename(renamed)

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(ctx["proj"])
    qtbot.wait(50)
    sess = w2._session

    doc_a = live_doc_id(sess, ctx["doc_a"])
    assert doc_a is not None
    assert sess.results.get(doc_a, ctx["ch_a"], "detect.schmitt") is not None
    assert w2._analysis_panel._table._proxy.rowCount() == len(_EVENTS)

    assert live_doc_id(sess, ctx["doc_b"]) is None
    assert sess.results.get(ctx["doc_b"], ctx["ch_b"], "detect.schmitt") is None

    assert not w2._info_bar.isHidden()
    assert "pore_b" in w2._info_bar._label.text()

    w2._analysis_controller.abort()
    w2._controller.abort()


@pytest.mark.skipif(not _REAL_ABF.exists(), reason="real ABF not present")
def test_real_abf_roundtrip_reports_size(qapp, qtbot, tmp_path, capsys):
    from nanoscope.io.base import open_document

    win = _fresh_window(qtbot)
    doc = open_document(_REAL_ABF, None)
    doc_id, ch = _install(win, doc)
    assert doc.meta.channels[ch].kind is ChannelKind.CURRENT

    _detect(win, qtbot, min_dwell_s=5e-5)
    res = win._session.results.get(doc_id, ch, "detect.schmitt")
    assert res is not None
    n_events = len(res.payload)
    assert n_events > 0
    df_saved = res.payload.df.copy()

    proj = tmp_path / "real.nanoscope"
    win._write_project(str(proj))
    assert proj.exists()
    size = os.path.getsize(proj)
    abf_size = os.path.getsize(_REAL_ABF)

    win._analysis_controller.abort()
    win._controller.abort()

    # Reopen in a FRESH window; events must come back WITHOUT re-detecting.
    w2 = _fresh_window(qtbot)
    reran = []
    w2._analysis_controller.finished.connect(lambda *_: reran.append(1))
    with qtbot.waitSignal(w2.projectRestored, timeout=60_000):
        w2._open_project(str(proj))
    qtbot.wait(50)

    res2 = w2._session.results.get(live_doc_id(w2._session, doc_id), ch, "detect.schmitt")
    assert res2 is not None
    assert len(res2.payload) == n_events
    assert_frame_equal(res2.payload.df, df_saved)
    assert reran == [], "detection re-ran on real-ABF restore"

    with capsys.disabled():
        print(f"\n[real-abf] events={n_events} project={size}B abf={abf_size}B "
              f"ratio={size / abf_size:.4f}")

    w2._analysis_controller.abort()
    w2._controller.abort()
