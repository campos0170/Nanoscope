"""Sweep 7, S7-POP1 (minor), the MainWindow half: the events table's stale badge set and the
banner's sentence reach the Event Gallery and the Event Inspector — the SAME tuple, on
every re-feed and on the two paths that reach no re-feed.

Measured on the pre-fix tree (docs/sweep7/repros/S7-POP1_stale_depth_unmarked_on_
curation_surfaces.py): one Preprocess knob after a detection badged nine table headers
and raised "Depth, frac_depth, ECD, mean/max deviation and SNR are stale", while the
gallery went on sorting and captioning by ``depth_pA`` unmarked and the inspector printed
"depth: 122.37 pA" bare — both separate top-level windows sitting OVER the banner. Agent B
gave both windows a ``set_stale_columns`` seam and agent C2 gave the panel a
``stalenessChanged`` signal; this file pins the wiring between them
(``MainWindow._push_stale_columns``): the knob path, the calibration path, every re-feed,
and a restored project whose detections predate the stamp — the owner's own project
opens badged on all three recordings, so the gallery and inspector must open badged too.

Real widgets throughout: a real Schmitt detection through the panel (so the stamps
compared are the engine's), the real gallery and inspector opened through their menu
actions, the real Preprocess knob, the real calibration door, a real project round-trip.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import register_all
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as MW
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans=_FRAGMENTS, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


def _doc(path: Path, signal: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._batch_controller.abort()


@pytest.fixture
def detected(window, qtbot, tmp_path):
    """A real Schmitt detection on screen, its stamps written by the engine, with BOTH
    curation windows open (through their menu actions) and the gallery sorted by depth."""
    did = window._session.add_document(_doc(tmp_path / "a.dat", _signal()))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    qtbot.wait(20)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    assert panel.current_detection() is not None
    assert panel._table.stale_columns == ()
    window._act_event_gallery.trigger()
    qtbot.wait(30)
    window._act_event_inspector.trigger()
    qtbot.wait(30)
    gal, insp = window._event_gallery, window._event_inspector
    assert gal is not None and insp is not None
    gal.set_sort("Depth (deepest first)")
    df = panel.current_detection().payload.df
    insp.show_event(int(df.iloc[0]["event_id"]))
    qtbot.wait(20)
    assert gal.stale_columns() == () and insp.stale_columns() == ()
    return window, did, gal, insp


def _captions(gal, df, n: int = 4) -> list[str]:
    return [gal.caption_for(df.iloc[r]) for r in gal._model.page_df_rows()[:n]]


def _stale_body(insp) -> str:
    """The staleness sentence as the inspector's Event box carries it -- its fixed-height
    note (``stale_note()``), not readout lines (sweep-7 fix-up B3 / CW-2: as readout
    lines it resized the waveform plot under the pointer)."""
    return insp.stale_note()


def _assert_both_badged(window, gal, insp) -> str:
    panel = window._analysis_panel
    cols = tuple(panel._table.stale_columns)
    assert cols == tuple(REMEASURED_COLUMNS), cols
    note = panel._stale.message()
    assert note
    df = panel.current_detection().payload.df
    assert gal.stale_columns() == cols
    assert all("⚠" not in c for c in _captions(gal, df)), _captions(gal, df)
    assert note in gal.status.text(), gal.status.text()
    assert insp.stale_columns() == cols
    assert _stale_body(insp) == note, insp._readout.text().splitlines()[:3]
    return note


def _assert_both_clean(window, gal, insp) -> None:
    panel = window._analysis_panel
    assert panel._table.stale_columns == ()
    df = panel.current_detection().payload.df
    assert gal.stale_columns() == () and insp.stale_columns() == ()
    assert not any(c.endswith("⚠") for c in _captions(gal, df))
    assert "⚠" not in gal.status.text(), gal.status.text()
    assert _stale_body(insp) == ""


# ================================================================================================
# the two paths that reach no surface refresh: the knob and the gain edit
# ================================================================================================

def test_a_preprocess_knob_warns_in_gallery_status_and_the_inspector_readout(
        detected, qtbot):
    window, _did, gal, insp = detected
    bl = window._analysis_panel.baseline_panel
    original = bl.rough_s.value()

    bl.rough_s.setValue(original * 4)
    qtbot.wait(30)
    note = _assert_both_badged(window, gal, insp)
    assert "stale" in note or "predates" in note, note

    # ...and the CLEAR path reaches both windows too.
    bl.rough_s.setValue(original)
    qtbot.wait(30)
    _assert_both_clean(window, gal, insp)


def test_a_calibration_edit_badges_both_windows_with_the_calibration_sentence(
        detected, qtbot):
    window, did, gal, insp = detected
    window._session.set_calibration(did, 0, ChannelCalibration(2.0, 0.0, "pA"),
                                    write_sidecar=False)
    qtbot.wait(30)
    note = _assert_both_badged(window, gal, insp)
    assert "Calibration changed" in note, note
    assert window._analysis_panel._stale.kind == "calibration"


# ================================================================================================
# the re-feeds: a window opened AFTER the knob is fed its badges with the frame
# ================================================================================================

def test_a_window_opened_after_the_knob_is_fed_badged(window, qtbot, tmp_path):
    did = window._session.add_document(_doc(tmp_path / "b.dat", _signal()))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    bl = panel.baseline_panel
    bl.rough_s.setValue(bl.rough_s.value() * 4)
    qtbot.wait(30)
    assert panel._table.stale_columns == tuple(REMEASURED_COLUMNS)
    window._act_event_gallery.trigger()
    window._act_event_inspector.trigger()
    qtbot.wait(40)
    gal, insp = window._event_gallery, window._event_inspector
    gal.set_sort("Depth (deepest first)")
    insp.show_event(int(panel.current_detection().payload.df.iloc[0]["event_id"]))
    qtbot.wait(20)
    _assert_both_badged(window, gal, insp)


# ================================================================================================
# a restored project whose detections predate the stamp: badged on open, on every recording
# ================================================================================================

def _csv_doc(tmp_path, signal, *, name) -> tuple[TraceDocument, Path]:
    path = tmp_path / name
    t = np.arange(signal.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(signal, dtype=np.float64)}).to_csv(
        path, index=False)
    return _doc(path, signal), path


def test_a_restored_project_whose_detections_predate_the_stamp_opens_badged_everywhere(
        qapp, qtbot, tmp_path):
    """The owner's saved project: three recordings whose detections carry no baseline
    stamp (they predate it), which the panel judges as "this table predates the open-pore
    estimator now in use". The gallery and the inspector must open badged on each of them,
    not only the table."""
    w1 = MainWindow()
    qtbot.addWidget(w1)
    paths = []
    for i in range(3):
        doc, path = _csv_doc(tmp_path, _signal(seed=i), name=f"r{i}.csv")
        did = w1._session.add_document(doc)
        w1._create_default_layers(doc)
        df, prov = detect_events(_signal(seed=i), FS, baseline=I0, min_dwell_s=1e-4)
        assert len(df) == len(_FRAGMENTS)
        w1._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=did, channel=0,
            params={}, calibration_fingerprint=w1._session.calibration(did, 0).fingerprint(),
            payload=EventSet(df, prov)))          # baseline_fingerprint: None — predates it
        paths.append(path)
    proj = tmp_path / "old.nanoscope"
    w1._write_project(str(proj))
    w1._analysis_controller.abort()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(80)
    w2._act_event_gallery.trigger()
    w2._act_event_inspector.trigger()
    qtbot.wait(40)
    gal, insp = w2._event_gallery, w2._event_inspector
    gal.set_sort("Depth (deepest first)")
    for path in paths:
        live = w2._session.doc_id_for_source(str(path.resolve()))
        assert live is not None
        w2._on_selection(live, 0)
        qtbot.wait(40)
        panel = w2._analysis_panel
        res = panel.current_detection()
        assert res is not None and res.doc_id == live
        insp.show_event(int(res.payload.df.iloc[0]["event_id"]))
        qtbot.wait(20)
        note = _assert_both_badged(w2, gal, insp)
        assert "predates" in note, note
    w2._analysis_controller.abort()


# ================================================================================================
# the wiring is declared, and the declared route is the one that fires
# ================================================================================================

def test_the_signal_is_classified_neutral_and_routed_to_the_push(detected, qtbot):
    key = "analysis_panel.stalenessChanged"
    assert key in MW.SURFACE_NEUTRAL_SIGNALS
    assert "_push_stale_columns" in MW.SURFACE_NEUTRAL_SIGNALS[key]
    window, _did, gal, insp = detected
    hits: list[int] = []
    real = window._push_stale_columns
    window._push_stale_columns = lambda: (hits.append(1), real())[1]
    window._analysis_panel.stalenessChanged.emit()
    qtbot.wait(5)
    assert hits == [1]
