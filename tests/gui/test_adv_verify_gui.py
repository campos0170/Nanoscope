"""ADVERSARIAL VERIFICATION of the three fix tiers -- written from the DEFECT
DESCRIPTIONS, not from the fixes.

Every test here reproduces an original user-visible failure and asserts the user's
outcome (her calibration is there, her segment is there, the number on screen is the
number that was measured), never that a particular new code path was entered. Each is
run against a pre-fix tree as well, so a test that cannot fail is caught.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, must precede any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from conftest import live_doc_id
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 40_000
_MODULE = "detect.schmitt"
_EVENTS = [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]


# =====================================================================
# builders
# =====================================================================
def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _win(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _install(win, doc):
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    return doc_id, 0


def _detect(win, qtbot, min_dwell_s: float = 5e-4):
    p = win._analysis_panel
    p._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        p.trigger_detection()
    qtbot.wait(20)


def _quit(win):
    win._analysis_controller.abort()
    win._controller.abort()


def _events(n: int = 3) -> EventSet:
    from nanoscope.analysis import events as ev

    seed = {"event_id": lambda i: i, "recording": lambda i: "rec",
            "pore_id": lambda i: "p", "start_idx": lambda i: 1000 + 1000 * i,
            "end_idx": lambda i: 1100 + 1000 * i,
            "t_start_s": lambda i: (1000 + 1000 * i) / FS,
            "dwell_s": lambda i: 100.0 / FS, "baseline_pA": lambda i: 100.0,
            "depth_pA": lambda i: 30.0, "frac_depth": lambda i: 0.3,
            "ecd_pA_s": lambda i: 0.03, "max_dev_pA": lambda i: 32.0,
            "mean_dev_pA": lambda i: 30.0, "rise_s": lambda i: 1e-5,
            "fall_s": lambda i: 1e-5, "snr": lambda i: 20.0, "n_levels": lambda i: 1,
            "filter_limited": lambda i: False, "accepted": lambda i: True,
            "flag": lambda i: ""}
    rows = [{c: seed[c](i) for c in ev.REQUIRED} for i in range(n)]
    df = pd.DataFrame(rows)[ev.REQUIRED].astype(
        {c: d for c, (d, _, _) in ev.SCHEMA.items()})
    return EventSet(df, {"module": _MODULE, "analyzed_duration_s": N / FS})


def _result(win, doc_id, ch=0, payload=None):
    return AnalysisResult(
        module_id=_MODULE, module_version="1", doc_id=doc_id, channel=ch, params={},
        calibration_fingerprint=win._session.calibration(doc_id, ch).fingerprint(),
        payload=payload or _events(), overlays=(), scalars={}, warnings=())


def _measure(win, qtbot, doc_id) -> int:
    """Add one hand measurement through the real MeasurePanel path."""
    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    vals = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(vals[2_000]), float(vals[5_000]))
    qtbot.wait(5)
    assert mp.add_current() is True
    return len(mp.measurements_df)


def _project_with_work(qtbot, tmp_path, name, *, gain, seed):
    """A saved project holding a real detection AND a non-default calibration."""
    csv = tmp_path / f"{name}.csv"
    arr = _trace(seed)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(gain, 0.0, "pA", Provenance.UNKNOWN))
    _detect(w, qtbot)
    n = len(w._session.results.get(doc_id, ch, _MODULE).payload)
    assert n > 0
    proj = tmp_path / f"{name}.nanoscope"
    assert w._write_project(str(proj)) is True
    _quit(w)
    for sc in tmp_path.glob("*.npcal.json"):
        sc.unlink()   # a restored calibration can now only have come from the project
    return {"proj": str(proj), "csv": csv, "arr": arr, "doc_id": doc_id, "ch": ch,
            "n": n, "gain": gain}


# =====================================================================
# C1 -- Open Project destroyed saved work and then lied
# =====================================================================
def test_C1_reopening_a_project_over_the_same_open_file_restores_the_saved_work(
        qapp, qtbot, tmp_path):
    """ORIGINAL: the user opens her recording, then File>Open Project on the project she
    saved for it. The open pipeline skipped the already-open document, so the project's
    calibration and detection were discarded and the empty in-memory state was shown AS
    the project."""
    ctx = _project_with_work(qtbot, tmp_path, "pore", gain=0.5, seed=0)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(ctx["csv"], ctx["arr"]))
    assert w._session.calibration(doc_id, ch).gain == 1.0
    assert w._session.results.get(doc_id, ch, _MODULE) is None

    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(ctx["proj"])
    qtbot.wait(50)

    # Re-resolved by SOURCE: the restore closed the in-memory load and reopened the file, so
    # the id moved (core/identity.py). The state under test — the project's calibration and
    # detection, not the id — is what has to be there.
    live = live_doc_id(w._session, ctx["doc_id"])
    assert live is not None
    assert w._session.calibration(live, ctx["ch"]).gain == pytest.approx(0.5)
    got = w._session.results.get(live, ctx["ch"], _MODULE)
    assert got is not None, "the project's detection was discarded"
    assert len(got.payload) == ctx["n"]
    _quit(w)


def test_C1_saving_after_reopening_does_not_overwrite_the_project_with_nothing(
        qapp, qtbot, tmp_path):
    """ORIGINAL, the damaging half: because the skipped restore still marked the session
    clean, the next Ctrl+S wrote the STALE state over the good file."""
    ctx = _project_with_work(qtbot, tmp_path, "pore", gain=0.5, seed=1)
    w = _win(qtbot)
    _install(w, _doc(ctx["csv"], ctx["arr"]))
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(ctx["proj"])
    qtbot.wait(50)
    assert w._write_project(ctx["proj"]) is True
    _quit(w)

    from nanoscope.core.project import load_project
    st = load_project(ctx["proj"])
    assert len(st.results) >= 1, "re-saving wiped the detection out of the project"
    assert any(abs(c.gain - 0.5) < 1e-9 for c in st.calibrations), \
        "re-saving wiped the calibration out of the project"


def test_C1_opening_a_second_project_replaces_the_first_rather_than_merging(
        qapp, qtbot, tmp_path):
    a = _project_with_work(qtbot, tmp_path, "a", gain=0.5, seed=2)
    b = _project_with_work(qtbot, tmp_path, "b", gain=0.25, seed=3)
    w = _win(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(a["proj"])
    qtbot.wait(50)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(b["proj"])
    qtbot.wait(50)
    assert live_doc_id(w._session, b["doc_id"]) is not None
    assert live_doc_id(w._session, a["doc_id"]) is None, \
        "the first project's documents were merged in"
    _quit(w)


def test_C1_cancelling_the_unsaved_work_prompt_aborts_the_open_cleanly(
        qapp, qtbot, tmp_path):
    """NEW-DAMAGE CHECK: Cancel must leave the session EXACTLY as it was -- not half
    cleared, not partly restored."""
    ctx = _project_with_work(qtbot, tmp_path, "pore", gain=0.5, seed=4)
    other = tmp_path / "other.csv"
    arr = _trace(9)
    _write_csv(other, arr)

    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(other, arr))
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(3.0, 1.0, "pA", Provenance.UNKNOWN))
    w._session.add_result(_result(w, doc_id, ch))
    w._mark_dirty()

    before_docs = [d.meta.path for d in w._session.documents()]
    before_cal = w._session.calibration(doc_id, ch)
    before_res = w._session.results.get(doc_id, ch, _MODULE)
    assert before_res is not None

    from nanoscope.gui.main_window import _CLOSE_CANCEL
    w._confirm_close_choice = lambda *a, **k: _CLOSE_CANCEL
    w._open_project(ctx["proj"])
    qtbot.wait(50)

    assert [d.meta.path for d in w._session.documents()] == before_docs, \
        "Cancel still disturbed the session"
    assert w._session.calibration(doc_id, ch).gain == before_cal.gain
    assert w._session.results.get(doc_id, ch, _MODULE) is not None, \
        "Cancel destroyed the detection it promised to keep"
    assert not w._session.has_document(ctx["doc_id"]) or ctx["doc_id"] == doc_id, \
        "Cancel still went on to open the project"
    _quit(w)


def test_C1_a_full_open_work_save_reopen_cycle_round_trips_everything(
        qapp, qtbot, tmp_path):
    """NEW-DAMAGE CHECK: the ordinary cycle must still carry documents, calibrations,
    detections, segments and measurements across a save/reopen."""
    csv = tmp_path / "full.csv"
    arr = _trace(5)
    _write_csv(csv, arr)

    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(0.75, 2.0, "pA", Provenance.UNKNOWN))
    _detect(w, qtbot)
    n_events = len(w._session.results.get(doc_id, ch, _MODULE).payload)

    # a measurement row
    n_meas = _measure(w, qtbot, doc_id)
    assert n_meas >= 1

    # a hand-cut segment
    seg_id = w._session.create_segment(doc_id, 1_000, 20_000, name="cut")
    assert seg_id is not None

    proj = tmp_path / "full.nanoscope"
    assert w._write_project(str(proj)) is True
    _quit(w)

    w2 = _win(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(80)

    s = w2._session
    live = live_doc_id(s, doc_id)
    assert live is not None, "the recording did not come back"
    assert live != doc_id, "a reopened file must be a new LOAD (core/identity.py)"
    assert s.calibration(live, ch).gain == pytest.approx(0.75)
    assert s.calibration(live, ch).offset == pytest.approx(2.0)
    r = s.results.get(live, ch, _MODULE)
    assert r is not None and len(r.payload) == n_events, "the detection did not come back"
    assert len(w2._analysis_panel.measure_panel.measurements_df) == n_meas, \
        "the measurements did not come back"
    assert len(s.segment_documents()) >= 1, "the segment did not come back"
    _quit(w2)


# =====================================================================
# C2 -- restoring a project destroyed the sidecar beside the RAW DATA
# =====================================================================
def test_C2_restoring_a_project_leaves_the_users_sidecar_byte_identical(
        qapp, qtbot, tmp_path):
    """ORIGINAL: the restore called set_calibration, which wrote (or, at default values,
    UNLINKED) <datafile>.npcal.json next to the user's raw data -- destroying a note the
    project file cannot carry. Destruction OUTSIDE the app."""
    csv = tmp_path / "sc.csv"
    arr = _trace(6)
    _write_csv(csv, arr)

    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.add_result(_result(w, doc_id, ch))
    proj = tmp_path / "sc.nanoscope"
    assert w._write_project(str(proj)) is True
    _quit(w)

    # The user's own sidecar, with a note only IT can hold.
    sidecar = csv.with_suffix(csv.suffix + ".npcal.json")
    sidecar.write_text(json.dumps(
        {"version": 1, "channels": {"0": {"gain": 1.0, "offset": 0.0, "units": "pA",
                                          "note": "pore 3, do not lose me"}}}, indent=2))
    before = sidecar.read_bytes()

    w2 = _win(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(60)
    _quit(w2)

    assert sidecar.exists(), "restoring the project DELETED the sidecar by the raw data"
    assert sidecar.read_bytes() == before, "restoring the project rewrote the sidecar"
    assert "do not lose me" in sidecar.read_text()


def test_C2_a_real_calibration_edit_still_writes_the_sidecar(qapp, qtbot, tmp_path):
    """CONTROL: the fix must not have switched sidecar writing off altogether."""
    csv = tmp_path / "edit.csv"
    arr = _trace(7)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(2.5, 0.0, "pA", Provenance.UNKNOWN))
    _quit(w)
    sidecar = csv.with_suffix(csv.suffix + ".npcal.json")
    assert sidecar.exists(), "a genuine calibration edit no longer persists a sidecar"
    assert abs(json.loads(sidecar.read_text())["channels"]["0"]["gain"] - 2.5) < 1e-9


# =====================================================================
# C4 -- a measurements-only session closed with no prompt and no dirty mark
# =====================================================================
def test_C4_a_measurements_only_session_is_work_worth_prompting_for(qapp, qtbot, tmp_path):
    """ORIGINAL: measure a region, close the recording -> the table still holds the rows,
    but the app had no documents, so it showed no dot and quit without asking."""
    csv = tmp_path / "m.csv"
    arr = _trace(8)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    assert _measure(w, qtbot, doc_id) >= 1
    w._mark_dirty()

    w._close_document(doc_id)
    qtbot.wait(20)
    assert not w._session.documents()
    assert len(w._analysis_panel.measure_panel.measurements_df) >= 1, \
        "fixture broken: the rows went too"

    assert w._has_work_to_lose() is True, "measurements alone did not count as work"
    assert "•" in w.windowTitle(), f"no dirty marker: {w.windowTitle()!r}"

    asked = []
    from nanoscope.gui.main_window import _CLOSE_CANCEL
    w._confirm_close_choice = lambda *a, **k: (asked.append(1), _CLOSE_CANCEL)[1]
    from PySide6 import QtGui
    ev = QtGui.QCloseEvent()
    w.closeEvent(ev)
    assert asked, "closing a measurements-only session did not ask"
    _quit(w)


# =====================================================================
# C10 -- a detection finishing after its document closed was installed anyway
# =====================================================================
def test_C10_reopening_a_file_shows_no_phantom_detection(qapp, qtbot, tmp_path):
    """ORIGINAL: start a detection, close the recording before it lands. The result was
    installed under the doc_id anyway -- and a file's doc_id is its path, so reopening the
    same file showed a detection that never ran on it."""
    csv = tmp_path / "ph.csv"
    arr = _trace(10)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))

    w._close_document(doc_id)
    qtbot.wait(10)
    assert doc_id not in w._session.documents()

    # the in-flight answer arrives now, for a document that is gone
    w._analysis_panel._on_controller_finished(_result(w, doc_id, ch))
    qtbot.wait(10)

    doc_id2, ch2 = _install(w, _doc(csv, arr))
    # A reopen is a DIFFERENT LOAD and says so — which is now the mechanism, not merely an
    # observation: the stale result's id can no longer match any open document, so the
    # phantom is impossible rather than merely guarded against at one call site.
    assert doc_id2 != doc_id, "a reopened file must be a new load (core/identity.py)"
    assert w._session.results.get(doc_id, ch, _MODULE) is None, \
        "the reopened file shows a phantom detection"
    assert w._session.results.get(doc_id2, ch2, _MODULE) is None, \
        "the reopened file shows a phantom detection"
    _quit(w)


# =====================================================================
# C11 -- the Plots window was left off resultsChanged
# =====================================================================
def test_C11_a_re_detection_reaches_an_open_plots_window(qapp, qtbot, tmp_path):
    """ORIGINAL: curation appeared to flow only because the panel mutates the shared frame
    in place. A re-detection REPLACES the frame, and the window never heard about it."""
    csv = tmp_path / "pl.csv"
    arr = _trace(11)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.add_result(_result(w, doc_id, ch, payload=_events(3)))

    w._open_plots()
    qtbot.wait(20)
    seen = []
    orig = w._refresh_plots
    w._refresh_plots = lambda *a, **k: (seen.append(1), orig(*a, **k))[1]

    w._session.add_result(_result(w, doc_id, ch, payload=_events(7)))
    qtbot.wait(30)
    assert seen, "the Plots window was never told the results changed"
    _quit(w)


# =====================================================================
# B3 -- opening a recording armed the quit guard
# =====================================================================
def test_B3_opening_a_recording_alone_is_not_unsaved_work(qapp, qtbot, tmp_path):
    """ORIGINAL: merely opening a file marked the session dirty, so quitting nagged about
    'unsaved work' that a path on the Recent list fully describes."""
    csv = tmp_path / "b3.csv"
    arr = _trace(12)
    _write_csv(csv, arr)
    w = _win(qtbot)
    _install(w, _doc(csv, arr))
    qtbot.wait(10)
    assert "•" not in w.windowTitle(), f"opening alone armed the guard: {w.windowTitle()!r}"

    asked = []
    from nanoscope.gui.main_window import _CLOSE_DISCARD
    w._confirm_close_choice = lambda *a, **k: (asked.append(1), _CLOSE_DISCARD)[1]
    from PySide6 import QtGui
    w.closeEvent(QtGui.QCloseEvent())
    assert not asked, "quitting after merely opening a file still nagged"
    _quit(w)


def test_B3_real_work_on_an_opened_recording_still_marks_dirty(qapp, qtbot, tmp_path):
    """CONTROL: the fix must not have disarmed the guard for genuine work."""
    csv = tmp_path / "b3b.csv"
    arr = _trace(13)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(2.0, 0.0, "pA", Provenance.UNKNOWN))
    w._mark_dirty()
    qtbot.wait(10)
    assert w._has_work_to_lose() is True
    assert "•" in w.windowTitle()
    _quit(w)


# =====================================================================
# C3 -- closing a segment's parent then saving silently dropped the segment
# =====================================================================
def test_C3_closing_a_parent_asks_before_taking_its_segments(qapp, qtbot, tmp_path):
    """ORIGINAL: a segment is stored as a crop of its parent and re-derived on load, so
    closing the parent and saving wrote a project that reloads WITHOUT the segment -- while
    reporting 'Saved project (1 result(s))' and clearing the dirty dot."""
    csv = tmp_path / "seg.csv"
    arr = _trace(20)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    seg_id = w._session.create_segment(doc_id, 1_000, 20_000, name="cut")
    assert w._session.has_document(seg_id)

    asked = []
    from nanoscope.gui.main_window import _CLOSE_CANCEL
    w._confirm_close_with_segments = lambda name, n: (asked.append((name, n)),
                                                      _CLOSE_CANCEL)[1]

    # Drive the two paths the USER has, not the internal handler: the File menu action
    # and the Files dock button. Both must land on the same guard.
    w._on_selection(doc_id, ch)
    w._close_current()                       # File > Close document
    qtbot.wait(10)
    assert asked, "File > Close on a parent with a segment did not warn"
    assert asked[0][1] == 1
    assert w._session.has_document(doc_id), "Cancel still closed the parent"
    assert w._session.has_document(seg_id), "Cancel still dropped the segment"

    asked.clear()
    w._file_dock.closeDocumentRequested.emit(doc_id)   # the dock's close button
    qtbot.wait(10)
    assert asked, "the Files dock close button bypassed the guard"
    assert w._session.has_document(doc_id) and w._session.has_document(seg_id)
    _quit(w)


def test_C3_saving_an_orphaned_segment_refuses_rather_than_writing_a_lie(
        qapp, qtbot, tmp_path):
    """ORIGINAL: with the parent closed the segment cannot be re-derived, yet the save
    claimed success. It must now ask, and a decline must write NOTHING and stay dirty."""
    csv = tmp_path / "orph.csv"
    arr = _trace(21)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    seg_id = w._session.create_segment(doc_id, 1_000, 20_000, name="cut")
    w._session.add_result(_result(w, seg_id, 0))

    from nanoscope.gui.main_window import _CLOSE_PARENT_ONLY
    w._confirm_close_with_segments = lambda name, n: _CLOSE_PARENT_ONLY
    w._close_document(doc_id)
    qtbot.wait(10)
    assert w._session.has_document(seg_id)
    assert not w._session.has_document(doc_id)

    asked = []
    w._confirm_save_dropping_segments = lambda ns, nr: (asked.append((ns, nr)), False)[1]
    proj = tmp_path / "orph.nanoscope"
    ok = w._write_project(str(proj))
    assert asked, "saving an unrestorable segment did not warn"
    assert ok is False, "the save reported success while dropping the segment"
    assert not proj.exists(), "a declined save still wrote the file"
    assert w._dirty is True, "a declined save cleared the dirty flag"
    _quit(w)


# =====================================================================
# C5 -- a detection did not remember the settings it ran under
# =====================================================================
def test_C5_a_reopened_detection_carries_the_settings_it_ran_under(
        qapp, qtbot, tmp_path):
    """ORIGINAL: params were dropped on save AND replaced with {} on restore, so a
    reopened detection could not say what produced it -- and 'Re-run' re-detected with the
    panel DEFAULTS, replacing the table (and every hand curation on it)."""
    csv = tmp_path / "c5.csv"
    arr = _trace(22)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    _detect(w, qtbot, min_dwell_s=2.2e-3)          # deliberately NOT the panel default
    res = w._session.results.get(doc_id, ch, _MODULE)
    assert res.params.get("min_dwell_s") == pytest.approx(2.2e-3)
    n_events = len(res.payload)

    proj = tmp_path / "c5.nanoscope"
    assert w._write_project(str(proj)) is True
    _quit(w)

    w2 = _win(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(80)
    got = w2._session.results.get(live_doc_id(w2._session, doc_id), ch, _MODULE)
    assert got is not None
    assert got.params.get("min_dwell_s") == pytest.approx(2.2e-3), \
        f"the reopened detection forgot its settings: {got.params!r}"
    assert len(got.payload) == n_events
    # and the form on screen must describe the table on screen
    assert w2._analysis_panel._detection.params()["min_dwell_s"] == pytest.approx(2.2e-3), \
        "the detection form was not re-seeded from the reopened run"
    _quit(w2)


def test_C5_a_rerun_that_cannot_run_does_not_dismiss_the_staleness_banner(
        qapp, qtbot, tmp_path):
    """ORIGINAL sub-bug: the banner was hidden BEFORE the guard, so every silent refusal
    removed the only warning that the depths on screen were computed under the old gain."""
    csv = tmp_path / "c5b.csv"
    arr = _trace(23)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    _detect(w, qtbot)
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(2.0, 0.0, "pA", Provenance.UNKNOWN))
    qtbot.wait(20)
    panel = w._analysis_panel
    assert not panel._stale.isHidden(), "fixture broken: the banner should be up"

    panel._run_detection = lambda params: False      # a gate that has since closed
    panel._launch_ml = lambda params: False
    panel._on_rerun()
    qtbot.wait(10)
    assert not panel._stale.isHidden(), \
        "a re-run that never started still dismissed the staleness warning"
    _quit(w)


# =====================================================================
# C6 -- the project round-trip cleared a warning that was still true
# =====================================================================
def test_C6_a_stale_detection_is_still_stale_after_a_project_round_trip(
        qapp, qtbot, tmp_path):
    """ORIGINAL: the restore stamped the result with the LIVE calibration, so a detection
    computed under the old gain reopened looking current -- warning gone, depths still
    wrong."""
    csv = tmp_path / "c6.csv"
    arr = _trace(24)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    _detect(w, qtbot)
    depth_before = float(
        w._session.results.get(doc_id, ch, _MODULE).payload.df["depth_pA"].iloc[0])
    # the user recalibrates AFTER detecting: the result is now stale, and stays stale
    w._session.set_calibration(doc_id, ch,
                               ChannelCalibration(2.0, 0.0, "pA", Provenance.UNKNOWN))
    qtbot.wait(20)
    proj = tmp_path / "c6.nanoscope"
    assert w._write_project(str(proj)) is True
    _quit(w)

    w2 = _win(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(80)
    live_id = live_doc_id(w2._session, doc_id)
    got = w2._session.results.get(live_id, ch, _MODULE)
    live = w2._session.calibration(live_id, ch).fingerprint()
    assert tuple(got.calibration_fingerprint) != tuple(live), \
        "the round-trip re-stamped the result with the live calibration"
    assert float(got.payload.df["depth_pA"].iloc[0]) == pytest.approx(depth_before)
    assert not w2._analysis_panel._stale.isHidden(), \
        "the staleness warning was cleared by the round-trip, but it is still true"
    _quit(w2)


# =====================================================================
# C7 -- eta(t) and the FDR followed the user to another recording
# =====================================================================
def test_C7_the_fdr_readout_does_not_follow_you_to_another_recording(
        qapp, qtbot, tmp_path):
    """ORIGINAL: the FDR label and the recovery map were panel-wide, so a percentage
    measured on recording A stayed on screen -- reading as A's -- after selecting B."""
    a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    a_arr, b_arr = _trace(30), _trace(31)
    _write_csv(a_csv, a_arr)
    _write_csv(b_csv, b_arr)
    w = _win(qtbot)
    a_id, _ = _install(w, _doc(a_csv, a_arr))
    b_id, _ = _install(w, _doc(b_csv, b_arr))

    w._on_selection(a_id, 0)
    qtbot.wait(10)
    panel = w._analysis_panel
    panel._fdr[(a_id, 0)] = {"estimated_fdr": 0.42, "n_surrogates": 10}
    panel._recovery[(a_id, 0)] = object()
    panel._sync_surrogate_readouts()
    qtbot.wait(5)
    a_text = panel._stats.fdr_label.text()
    assert "42" in a_text, f"fixture broken: {a_text!r}"
    assert panel._stats.use_efficiency.isEnabled()

    w._on_selection(b_id, 0)
    qtbot.wait(20)
    b_text = panel._stats.fdr_label.text()
    assert "42" not in b_text, \
        f"A's false-positive rate is still on screen for B: {b_text!r}"
    assert not panel._stats.use_efficiency.isEnabled(), \
        "B offers 'use measured efficiency' from A's recovery map"
    _quit(w)


def test_C7_a_surrogate_answer_for_a_closed_recording_is_dropped(qapp, qtbot, tmp_path):
    csv = tmp_path / "c7b.csv"
    arr = _trace(32)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    panel = w._analysis_panel
    panel._pending, panel._pending_sel = "null", (doc_id, ch)
    w._close_document(doc_id)
    qtbot.wait(10)
    panel._on_controller_finished({"estimated_fdr": 0.9, "n_surrogates": 5})
    qtbot.wait(5)
    assert (doc_id, ch) not in panel._fdr, \
        "an answer for a closed recording was stored anyway"
    _quit(w)


# =====================================================================
# C8 -- an on-screen x is the document's axis PLUS the layer's offset
# =====================================================================
def test_C8_the_visible_range_of_an_aligned_layer_is_what_the_user_framed(
        qapp, qtbot, tmp_path):
    """ORIGINAL: the visible-range lookup searched the DOCUMENT's axis with SCREEN x. On a
    nudged layer the two searchsorted results collapsed to the same end of the array and
    the degenerate fallback returned THE WHOLE RECORDING -- so 'Detect over the visible
    range', the default ML mode, ran over everything."""
    csv = tmp_path / "c8.csv"
    arr = _trace(33)
    _write_csv(csv, arr)
    w = _win(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    panel = w._analysis_panel

    # frame a narrow window with NO offset -> the honest answer
    anchor = w._trace_view.anchor_plot()
    assert anchor is not None
    vals = w._session.x_axis(doc_id).values
    lo, hi = float(vals[10_000]), float(vals[15_000])
    anchor.vb.setXRange(lo, hi, padding=0)
    qtbot.wait(20)
    i0, i1 = panel._visible_sample_range(doc_id, ch, arr.size)
    assert (i1 - i0) < arr.size * 0.6, f"fixture broken: {(i0, i1)}"

    # now nudge the layer and frame the SAME samples on screen
    off = 0.5 * arr.size / FS
    w._session.set_layer_time_offset(f"{doc_id}#{ch}", off)
    qtbot.wait(20)
    anchor = w._trace_view.anchor_plot()
    anchor.vb.setXRange(lo + off, hi + off, padding=0)
    qtbot.wait(20)
    j0, j1 = panel._visible_sample_range(doc_id, ch, arr.size)
    assert (j1 - j0) < arr.size * 0.6, \
        f"an aligned layer's visible range blew up to the whole recording: {(j0, j1)}"
    assert abs(j0 - i0) <= 3 and abs(j1 - i1) <= 3, \
        f"aligned {(j0, j1)} does not match unaligned {(i0, i1)}"
    _quit(w)
