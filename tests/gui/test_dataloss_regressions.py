"""Regressions for the DATA-LOSS tier: work the app used to destroy, and then lie about.

Every test here reproduces the ORIGINAL FAILURE — the user's calibration is gone, the
sidecar next to her raw data is gone, the reloaded project is empty, a detection appears on
a file it never ran on — rather than asserting that some new code path was entered.

Covered:

* **C1** ``File ▸ Open Project`` restored ON TOP of the live session. Documents that happened
  to be open were skipped, so the project's saved calibration + detections were discarded, the
  stale in-memory state was presented AS the project, and the session was then marked clean —
  so the next Ctrl+S overwrote the good file with the stale state. Two projects opened in a row
  merged. Unsaved work was discarded without a word.
* **C2** restoring a project rewrote — or, at default values, ``unlink``ed — the
  ``<datafile>.npcal.json`` sidecar next to the user's RAW DATA, destroying its ``note``
  (which a project file cannot carry). Destruction outside the app.
* **C3** closing a segment's parent and saving reported ``Saved project (1 result(s))``,
  cleared the dirty flag, and reloaded to nothing: a segment is persisted as a crop of its
  parent and re-derived from it on load. Plus the re-derive warning that called an unrebuilt
  segment a missing source file.
* **C10** a detection finishing AFTER its document was closed was installed anyway, so a
  freshly reopened file showed a detection it never ran.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import live_doc_id
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import (
    CalibrationRec,
    DocumentRef,
    ProjectState,
    ResultRec,
    SegmentRec,
    load_project,
    save_project,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import (
    _CLOSE_CANCEL,
    _CLOSE_DISCARD,
    _CLOSE_PARENT_ONLY,
    _CLOSE_SEGMENTS_TOO,
    MainWindow,
)

FS = 100_000.0
N = 40_000
_EVENTS = [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]
_MODULE = "detect.schmitt"


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _fresh_window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _install(win: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    return doc_id, 0


def _detect(win: MainWindow, qtbot, min_dwell_s: float = 5e-4) -> None:
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _fake_events(n: int = 3) -> EventSet:
    """A minimal, schema-valid EventSet — enough to be a persistable detection result.

    Built from the FROZEN 20-column ``events.SCHEMA`` itself, so this never drifts from it
    and never has to restate it."""
    from nanoscope.analysis import events as ev_mod

    df = ev_mod.empty_df()
    seed = {
        "event_id": lambda i: i, "recording": lambda i: "rec", "pore_id": lambda i: "pore",
        "start_idx": lambda i: 1_000 + 1_000 * i, "end_idx": lambda i: 1_100 + 1_000 * i,
        "t_start_s": lambda i: (1_000 + 1_000 * i) / FS, "dwell_s": lambda i: 100.0 / FS,
        "baseline_pA": lambda i: 100.0, "depth_pA": lambda i: 30.0,
        "frac_depth": lambda i: 0.3, "ecd_pA_s": lambda i: 0.03,
        "max_dev_pA": lambda i: 32.0, "mean_dev_pA": lambda i: 30.0,
        "rise_s": lambda i: 1e-5, "fall_s": lambda i: 1e-5, "snr": lambda i: 20.0,
        "n_levels": lambda i: 1, "filter_limited": lambda i: False,
        "accepted": lambda i: True, "flag": lambda i: "",
    }
    rows = [{col: seed[col](i) for col in ev_mod.REQUIRED} for i in range(n)]
    df = pd.DataFrame(rows)[ev_mod.REQUIRED].astype(
        {c: d for c, (d, _, _) in ev_mod.SCHEMA.items()})
    return EventSet(df, {"module": _MODULE, "analyzed_duration_s": N / FS})


def _install_fake_detection(win: MainWindow, doc_id: str, channel: int = 0) -> EventSet:
    ev = _fake_events()
    win._session.add_result(AnalysisResult(
        module_id=_MODULE, module_version="1", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, channel).fingerprint(),
        payload=ev, overlays=(), scalars={}, warnings=()))
    return ev


def _make_project_with_detection(qtbot, tmp_path, name: str, *, gain: float, seed: int) -> dict:
    """Save a one-document project holding a real detection and a non-default calibration."""
    csv = tmp_path / f"{name}.csv"
    arr = _event_trace(seed=seed)
    _write_csv(csv, arr)

    win = _fresh_window(qtbot)
    doc_id, ch = _install(win, _doc(csv, arr))
    win._session.set_calibration(
        doc_id, ch, ChannelCalibration(gain, 0.0, "pA", Provenance.UNKNOWN))
    _detect(win, qtbot)
    n_events = len(win._session.results.get(doc_id, ch, _MODULE).payload)
    assert n_events > 0

    proj = tmp_path / f"{name}.nanoscope"
    assert win._write_project(str(proj)) is True
    win._analysis_controller.abort()
    win._controller.abort()
    # The source window wrote a sidecar next to the CSV; drop it so a restored calibration can
    # only have come from the project file.
    for sc in tmp_path.glob("*.npcal.json"):
        sc.unlink()
    return {"proj": str(proj), "csv": csv, "arr": arr, "doc_id": doc_id, "ch": ch,
            "n_events": n_events, "gain": gain}


# ================================================================================================
# C1 — Open Project discards saved work and then lies about it
# ================================================================================================

def test_open_project_applies_saved_state_to_an_already_open_document(qapp, qtbot, tmp_path):
    """REGRESSION: the open pipeline SKIPPED any document already in the session, so the
    project's saved calibration and detection were silently thrown away and the stale
    in-memory state was presented as "the project"."""
    ctx = _make_project_with_detection(qtbot, tmp_path, "pore", gain=0.5, seed=0)

    w = _fresh_window(qtbot)
    # The very same recording is already open — with DEFAULT calibration and no detection.
    doc_id, ch = _install(w, _doc(ctx["csv"], ctx["arr"]))
    assert w._session.calibration(doc_id, ch).gain == 1.0
    assert w._session.results.get(doc_id, ch, _MODULE) is None

    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(ctx["proj"])
    qtbot.wait(50)

    sess = w._session
    # By SOURCE: the restore closed the in-memory load and reopened the file, so the id moved
    # (core/identity.py). The project's state is what has to survive, not the id.
    live = live_doc_id(sess, ctx["doc_id"])
    assert live is not None
    assert sess.calibration(live, ctx["ch"]).gain == pytest.approx(0.5)
    restored = sess.results.get(live, ctx["ch"], _MODULE)
    assert restored is not None
    assert len(restored.payload) == ctx["n_events"]

    w._analysis_controller.abort()
    w._controller.abort()


def test_resaving_a_reopened_project_does_not_overwrite_it_with_stale_state(
        qapp, qtbot, tmp_path):
    """REGRESSION — the whole C1 chain, end to end: open the file, open its project on top,
    Ctrl+S. The skipped restore left the session without the detection but marked it CLEAN, so
    the save happily wrote an empty project over the good one."""
    ctx = _make_project_with_detection(qtbot, tmp_path, "pore", gain=0.5, seed=1)

    w = _fresh_window(qtbot)
    _install(w, _doc(ctx["csv"], ctx["arr"]))          # same file, no detection
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(ctx["proj"])
    qtbot.wait(50)
    assert w._save_project() is True                    # Ctrl+S — re-saves in place
    w._analysis_controller.abort()
    w._controller.abort()

    # The project on disk must still hold the detection and the calibration.
    state = load_project(ctx["proj"])
    assert len(state.results) == 1
    assert len(state.results[0].events) == ctx["n_events"]
    assert any(c.gain == pytest.approx(0.5) for c in state.calibrations)


def test_opening_a_second_project_replaces_the_first_instead_of_merging(qapp, qtbot, tmp_path):
    """REGRESSION: opening project B over project A left A's documents in the session, so the
    two projects merged — and the next save wrote the merger to B's file."""
    a = _make_project_with_detection(qtbot, tmp_path, "pore_a", gain=0.5, seed=2)
    b = _make_project_with_detection(qtbot, tmp_path, "pore_b", gain=2.0, seed=3)

    w = _fresh_window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(a["proj"])
    qtbot.wait(30)
    assert live_doc_id(w._session, a["doc_id"]) is not None

    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(b["proj"])
    qtbot.wait(50)

    sess = w._session
    live_b = live_doc_id(sess, b["doc_id"])
    assert live_b is not None
    assert live_doc_id(sess, a["doc_id"]) is None, "project A survived into project B"
    assert {d.doc_id for d in sess.documents()} == {live_b}
    assert sess.results.get(a["doc_id"], a["ch"], _MODULE) is None

    w._analysis_controller.abort()
    w._controller.abort()


def test_opening_a_project_asks_before_discarding_unsaved_work(qapp, qtbot, tmp_path,
                                                               monkeypatch):
    """REGRESSION: unsaved analysis was discarded with no prompt at all. Cancel must leave the
    session EXACTLY as it was — nothing closed, no project adopted, still dirty."""
    ctx = _make_project_with_detection(qtbot, tmp_path, "pore", gain=0.5, seed=4)

    w = _fresh_window(qtbot)
    scratch = tmp_path / "scratch.csv"
    scratch_arr = _event_trace(seed=9)
    _write_csv(scratch, scratch_arr)
    scratch_id, _ch = _install(w, _doc(scratch, scratch_arr))
    _install_fake_detection(w, scratch_id)
    assert w._dirty is True

    prompts: list = []
    monkeypatch.setattr(w, "_confirm_close_choice",
                        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])
    w._open_project(ctx["proj"])
    qtbot.wait(30)

    assert len(prompts) == 1                                   # it asked
    assert w._session.has_document(scratch_id)                 # ...and Cancel changed nothing
    assert w._session.results.get(scratch_id, 0, _MODULE) is not None
    assert not w._session.has_document(ctx["doc_id"])
    assert w._project_path is None
    assert w._dirty is True

    w._analysis_controller.abort()
    w._controller.abort()


def test_discarding_at_the_open_project_prompt_starts_from_a_clean_session(
        qapp, qtbot, tmp_path, monkeypatch):
    """Discard is the explicit choice: the scratch work goes, the project comes back whole."""
    ctx = _make_project_with_detection(qtbot, tmp_path, "pore", gain=0.5, seed=5)

    w = _fresh_window(qtbot)
    scratch = tmp_path / "scratch.csv"
    scratch_arr = _event_trace(seed=11)
    _write_csv(scratch, scratch_arr)
    scratch_id, _ch = _install(w, _doc(scratch, scratch_arr))
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_DISCARD)

    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(ctx["proj"])
    qtbot.wait(50)

    assert not w._session.has_document(scratch_id)
    assert live_doc_id(w._session, ctx["doc_id"]) is not None
    assert w._dirty is False

    w._analysis_controller.abort()
    w._controller.abort()


def test_opening_a_project_clears_the_previous_measurements(qapp, qtbot, tmp_path, monkeypatch):
    """Measurements live outside the Session, so clearing the documents is not enough: the
    previous session's hand measurements must not be re-attributed to the incoming project."""
    ctx = _make_project_with_detection(qtbot, tmp_path, "pore", gain=0.5, seed=6)

    w = _fresh_window(qtbot)
    scratch = tmp_path / "scratch.csv"
    scratch_arr = _event_trace(seed=13)
    _write_csv(scratch, scratch_arr)
    scratch_id, _ch = _install(w, _doc(scratch, scratch_arr))
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    vals = w._session.x_axis(scratch_id).values
    w._trace_view.set_measure_region(float(vals[2_000]), float(vals[5_000]))
    qtbot.wait(5)
    assert mp.add_current() is True
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_DISCARD)

    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(ctx["proj"])
    qtbot.wait(50)
    assert len(mp.measurements_df) == 0

    w._analysis_controller.abort()
    w._controller.abort()


# ================================================================================================
# C2 — restoring a project destroys the .npcal.json sidecar next to the user's RAW DATA
# ================================================================================================

_NOTE = "electrode replaced 2026-07-01 — gain measured against the 1 MΩ standard"


def _write_sidecar(csv: Path, *, gain: float, target_units: str, note: str) -> Path:
    """A sidecar exactly as the app writes them — including the ``note`` a project cannot carry."""
    sc = csv.with_name(csv.name + ".npcal.json")
    sc.write_text(json.dumps({
        "version": 1,
        "channels": {"0": {"gain": gain, "offset": 0.0,
                           "target_units": target_units, "note": note}},
    }, indent=2))
    return sc


def _project_referencing(csv: Path, tmp_path: Path, *, cal: CalibrationRec) -> str:
    """A minimal project referencing ``csv`` with exactly one calibration record."""
    proj = tmp_path / "sidecar.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(path=str(csv.resolve()))],
        calibrations=[cal]))
    return str(proj)


def test_restoring_a_project_does_not_delete_the_users_sidecar(qapp, qtbot, tmp_path):
    """REGRESSION — the destructive case. The project's stored calibration is the DEFAULT, so
    the restore's write-through hit ``save_sidecar``'s "nothing differs from default" branch
    and ``unlink``ed the user's file. A file outside the app, gone, on open."""
    csv = tmp_path / "pore.csv"
    _write_csv(csv, _event_trace())
    sidecar = _write_sidecar(csv, gain=0.5, target_units="pA", note=_NOTE)
    before = sidecar.read_text()

    proj = _project_referencing(csv, tmp_path, cal=CalibrationRec(
        doc_id=str(csv.resolve()), channel=0, gain=1.0, offset=0.0,
        target_units="", provenance="unknown"))

    w = _fresh_window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(proj)
    qtbot.wait(50)

    assert sidecar.exists(), "restoring a project DELETED the sidecar next to the raw data"
    assert sidecar.read_text() == before
    w._controller.abort()


def test_restoring_a_project_does_not_rewrite_the_users_sidecar_note(qapp, qtbot, tmp_path):
    """REGRESSION — the lossy case. A non-default restored calibration rewrote the sidecar
    through ``save_sidecar``, which serialises ``ChannelCalibration.note`` — and a project
    carries no note, so the restored value's note was ``""``. The user's note was overwritten
    with nothing."""
    csv = tmp_path / "pore.csv"
    _write_csv(csv, _event_trace())
    sidecar = _write_sidecar(csv, gain=0.5, target_units="pA", note=_NOTE)
    before = sidecar.read_text()

    proj = _project_referencing(csv, tmp_path, cal=CalibrationRec(
        doc_id=str(csv.resolve()), channel=0, gain=2500.0, offset=0.0,
        target_units="nS", provenance="measured"))

    w = _fresh_window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(proj)
    qtbot.wait(50)

    assert sidecar.exists()
    assert sidecar.read_text() == before, "the restore rewrote the sidecar"
    # read through json (write_text escapes non-ASCII), so this asserts the NOTE, not the bytes
    assert json.loads(sidecar.read_text())["channels"]["0"]["note"] == _NOTE
    # ...and the project's calibration still won in memory, which is the point of restoring.
    live = live_doc_id(w._session, csv)
    assert live is not None
    assert w._session.calibration(live, 0).gain == pytest.approx(2500.0)
    w._controller.abort()


def test_a_real_calibration_edit_still_writes_the_sidecar(qapp, qtbot, tmp_path):
    """The suppression is scoped to reconstruction: a genuine edit must still persist."""
    csv = tmp_path / "pore.csv"
    arr = _event_trace()
    _write_csv(csv, arr)

    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    w._session.set_calibration(
        doc_id, ch, ChannelCalibration(2500.0, 0.0, "nS", Provenance.MEASURED))
    sidecar = csv.with_name(csv.name + ".npcal.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["channels"]["0"]["gain"] == 2500.0
    w._controller.abort()


# ================================================================================================
# C3 — closing a segment's parent then saving silently drops the segment
# ================================================================================================

def _window_with_segment(qtbot, tmp_path, monkeypatch) -> dict:
    """A window holding a file recording, a segment cut from it, and a detection ON the segment.
    The parent is then closed with the explicit "close only this recording" answer."""
    csv = tmp_path / "parent.csv"
    arr = _event_trace()
    _write_csv(csv, arr)

    w = _fresh_window(qtbot)
    parent_id, _ch = _install(w, _doc(csv, arr))
    seg_id = w._session.create_segment(parent_id, 1_000, 20_000, name="crop A")
    w._create_default_layers(w._session.document(seg_id))
    _install_fake_detection(w, seg_id)
    assert w._session.results.get(seg_id, 0, _MODULE) is not None
    return {"win": w, "parent_id": parent_id, "seg_id": seg_id, "csv": csv}


def test_closing_a_parent_with_segments_asks_first(qapp, qtbot, tmp_path, monkeypatch):
    """REGRESSION: closing the parent went straight through, quietly making every segment cut
    from it unsaveable. Cancel must leave both open."""
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]

    asked: list = []
    monkeypatch.setattr(w, "_confirm_close_with_segments",
                        lambda name, n: (asked.append((name, n)), _CLOSE_CANCEL)[1],
                        raising=False)
    w._close_current()   # File ▸ Close document, on the segment's PARENT

    assert len(asked) == 1
    assert asked[0][1] == 1                       # it counted the one dependent segment
    assert w._session.has_document(ctx["parent_id"])
    assert w._session.has_document(ctx["seg_id"])


def test_closing_a_parent_can_take_its_segments_with_it(qapp, qtbot, tmp_path, monkeypatch):
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    monkeypatch.setattr(w, "_confirm_close_with_segments",
                        lambda name, n: _CLOSE_SEGMENTS_TOO, raising=False)
    w._close_current()
    assert not w._session.has_document(ctx["parent_id"])
    assert not w._session.has_document(ctx["seg_id"])


def test_the_files_dock_close_button_goes_through_the_same_guard(qapp, qtbot, tmp_path,
                                                                 monkeypatch):
    """REGRESSION: the dock called ``Session.close_document`` itself, so the guard could be
    bypassed simply by using the button instead of the menu."""
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    asked: list = []
    monkeypatch.setattr(w, "_confirm_close_with_segments",
                        lambda name, n: (asked.append(n), _CLOSE_CANCEL)[1], raising=False)

    # Select the PARENT's row in the tree, then click the dock's own Close button.
    assert w._file_dock._focus(("doc", ctx["parent_id"])) is True
    w._file_dock._close_selected()

    assert asked == [1]
    assert w._session.has_document(ctx["parent_id"])
    assert w._session.has_document(ctx["seg_id"])


def test_nested_segments_are_all_counted_when_closing_the_root(qapp, qtbot, tmp_path,
                                                               monkeypatch):
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    nested = w._session.create_segment(ctx["seg_id"], 100, 5_000, name="crop A.1")
    assert w._session.has_document(nested)

    asked: list = []
    monkeypatch.setattr(w, "_confirm_close_with_segments",
                        lambda name, n: (asked.append(n), _CLOSE_SEGMENTS_TOO)[1], raising=False)
    w._close_current()
    assert asked == [2]                       # the segment AND the segment cut from it
    assert w._session.documents() == ()


def test_saving_an_orphaned_segment_refuses_rather_than_writing_a_lie(qapp, qtbot, tmp_path,
                                                                     monkeypatch):
    """REGRESSION: the save wrote the file, reported ``Saved project (1 result(s))`` and
    cleared the dirty flag while the segment and its detection were being dropped on the
    floor. Declining the warning must write NOTHING and keep the work flagged unsaved."""
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    monkeypatch.setattr(w, "_confirm_close_with_segments", lambda name, n: _CLOSE_PARENT_ONLY,
                        raising=False)
    w._close_current()
    assert not w._session.has_document(ctx["parent_id"])
    assert w._session.has_document(ctx["seg_id"])     # the segment outlives its parent in RAM

    warned: list = []
    monkeypatch.setattr(w, "_confirm_save_dropping_segments",
                        lambda n_seg, n_res: (warned.append((n_seg, n_res)), False)[1],
                        raising=False)

    proj = tmp_path / "orphan.nanoscope"
    assert w._write_project(str(proj)) is False
    assert warned == [(1, 1)]                        # one segment, one result on it
    assert not proj.exists(), "a project that would drop the segment was written anyway"
    assert w._dirty is True                          # ...and the work is still flagged unsaved


def test_saving_an_orphaned_segment_anyway_never_claims_it_saved_the_result(
        qapp, qtbot, tmp_path, monkeypatch):
    """REGRESSION: "Saved project (1 result(s))" was a lie — that one result was the segment's,
    and the reload came back with ``results=[]``. Saving anyway must state what was left out
    and must not count it."""
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    monkeypatch.setattr(w, "_confirm_close_with_segments", lambda name, n: _CLOSE_PARENT_ONLY,
                        raising=False)
    w._close_current()

    monkeypatch.setattr(w, "_confirm_save_dropping_segments", lambda n_seg, n_res: True,
                        raising=False)
    proj = tmp_path / "orphan.nanoscope"
    assert w._write_project(str(proj)) is True

    msg = w.statusBar().currentMessage()
    assert "Saved project (0 result(s))" in msg, msg
    assert "NOT saved" in msg, msg
    # ...and the claim is true: the reload really does come back without them.
    state = load_project(str(proj))
    assert state.documents == []


def test_unrebuildable_segment_warning_does_not_call_it_a_missing_source_file(
        qapp, qtbot, tmp_path):
    """REGRESSION: a segment whose parent is not in the project was funnelled into the
    "N source file(s) could not be found" banner — which then ran ``Path(...).name`` over a
    sentence. It was never a file, and nothing was missing from disk."""
    proj = tmp_path / "lonely_segment.nanoscope"
    save_project(proj, ProjectState(
        segments=[SegmentRec(segment_uid="seg-abcd1234-0-100", parent_key="/gone/parent.csv",
                             i0=0, i1=100, name="crop A")],
        results=[ResultRec(doc_id="seg-abcd1234-0-100", channel=0, module_id=_MODULE,
                           events=_fake_events().df, provenance={})]))

    w = _fresh_window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(30)

    assert not w._info_bar.isHidden()
    text = w._info_bar._label.text()
    assert "segment" in text.lower(), text
    assert "crop A" in text, text
    assert "source file" not in text.lower(), text
    w._controller.abort()


def test_a_segment_saved_with_its_parent_still_round_trips(qapp, qtbot, tmp_path, monkeypatch):
    """The guard must not have broken the case that always worked."""
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    proj = tmp_path / "with_parent.nanoscope"
    assert w._write_project(str(proj)) is True
    assert "NOT saved" not in w.statusBar().currentMessage()
    w._controller.abort()

    w2 = _fresh_window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    segs = w2._session.segment_documents()
    assert len(segs) == 1
    assert segs[0].display_name == "crop A"
    assert w2._session.results.get(segs[0].doc_id, 0, _MODULE) is not None
    w2._controller.abort()


# ================================================================================================
# C10 — a detection that finishes AFTER its document is closed is installed anyway
# ================================================================================================

def test_detection_finishing_after_its_document_closed_is_dropped(qapp, qtbot, tmp_path):
    """REGRESSION: ``add_result`` ran unconditionally even though ``close_document`` had
    already run ``drop_doc``, so the result was stored under a dead doc_id and the session was
    marked dirty over a result about nothing."""
    csv = tmp_path / "pore.csv"
    arr = _event_trace()
    _write_csv(csv, arr)

    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    late = AnalysisResult(
        module_id=_MODULE, module_version="1", doc_id=doc_id, channel=ch, params={},
        calibration_fingerprint=(1.0, 0.0), payload=_fake_events(), overlays=(),
        scalars={}, warnings=())

    w._session.close_document(doc_id)          # the user closes it while the job is in flight
    w._mark_clean()
    w._analysis_panel._on_controller_finished(late)   # ...and only THEN does it land

    assert w._session.results.get(doc_id, ch, _MODULE) is None
    assert w._dirty is False
    w._controller.abort()


def test_reopening_the_file_shows_no_phantom_detection(qapp, qtbot, tmp_path):
    """REGRESSION, the user-visible half: a file's doc_id IS its path, so the orphaned result
    re-attached itself the moment the same file was opened again — a detection on a freshly
    opened recording that was never run on it."""
    csv = tmp_path / "pore.csv"
    arr = _event_trace()
    _write_csv(csv, arr)

    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    late = AnalysisResult(
        module_id=_MODULE, module_version="1", doc_id=doc_id, channel=ch, params={},
        calibration_fingerprint=(1.0, 0.0), payload=_fake_events(), overlays=(),
        scalars={}, warnings=())
    w._session.close_document(doc_id)
    w._analysis_panel._on_controller_finished(late)

    _install(w, _doc(csv, arr))                       # reopen the very same recording
    assert w._session.results.get(doc_id, ch, _MODULE) is None
    assert w._analysis_panel.current_detection() is None
    assert w._act_export_events.isEnabled() is False
    w._controller.abort()


def test_batch_results_for_closed_documents_are_dropped_and_reported(qapp, qtbot, tmp_path):
    """The batch sink has exactly the same race, and a batch run is long enough to lose it."""
    csv_a, csv_b = tmp_path / "a.csv", tmp_path / "b.csv"
    arr = _event_trace()
    _write_csv(csv_a, arr)
    _write_csv(csv_b, arr)

    w = _fresh_window(qtbot)
    doc_a, _ = _install(w, _doc(csv_a, arr))
    doc_b, _ = _install(w, _doc(csv_b, arr))

    def _res(doc_id):
        return AnalysisResult(
            module_id=_MODULE, module_version="1", doc_id=doc_id, channel=0, params={},
            calibration_fingerprint=(1.0, 0.0), payload=_fake_events(), overlays=(),
            scalars={}, warnings=())

    outcome = type("Outcome", (), {"results": [_res(doc_a), _res(doc_b)], "errors": ()})()
    w._session.close_document(doc_b)
    w._on_batch_finished(outcome)

    assert w._session.results.get(doc_a, 0, _MODULE) is not None
    assert w._session.results.get(doc_b, 0, _MODULE) is None
    msg = w.statusBar().currentMessage()
    assert "Batch detect: 1 recording(s)" in msg, msg
    assert "Dropped 1" in msg, msg
    w._controller.abort()


def test_a_detection_for_a_still_open_document_is_installed_as_before(qapp, qtbot, tmp_path):
    """The guard must not have broken the ordinary path."""
    csv = tmp_path / "pore.csv"
    arr = _event_trace()
    _write_csv(csv, arr)

    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(csv, arr))
    res = AnalysisResult(
        module_id=_MODULE, module_version="1", doc_id=doc_id, channel=ch, params={},
        calibration_fingerprint=(1.0, 0.0), payload=_fake_events(), overlays=(),
        scalars={}, warnings=())
    w._analysis_panel._on_controller_finished(res)
    assert w._session.results.get(doc_id, ch, _MODULE) is not None
    w._controller.abort()


def test_the_drop_warning_counts_derived_fits_not_just_detections(qapp, qtbot, tmp_path,
                                                                  monkeypatch):
    """The one dialog whose entire purpose is to name the work being abandoned must name
    ALL of it.

    MEASURED BEFORE: ``n_dropped_results`` counted only ``state.results`` (EventSet
    payloads), so once derived fits became persistable they were left out of exactly this
    count. A user who had fitted a dwell distribution and a CUSUM level table on a segment
    was told "1 result(s)" would be lost, clicked through, and lost three. The post-save
    status line under-reported in the same direction.
    """
    ctx = _window_with_segment(qtbot, tmp_path, monkeypatch)
    w = ctx["win"]
    seg_id = ctx["seg_id"]

    # A derived fit ON THE SEGMENT, alongside the detection the fixture already made.
    import pandas as pd

    from nanoscope.core.analysis import AnalysisResult
    w._session.add_result(AnalysisResult(
        module_id="fit.levels", module_version="", doc_id=seg_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0),
        payload=pd.DataFrame({"event_id": [0], "level_idx": [0], "level_pA": [1.0]})))

    monkeypatch.setattr(w, "_confirm_close_with_segments", lambda name, n: _CLOSE_PARENT_ONLY,
                        raising=False)
    w._close_current()

    warned: list = []
    monkeypatch.setattr(w, "_confirm_save_dropping_segments",
                        lambda n_seg, n_res: (warned.append((n_seg, n_res)), False)[1],
                        raising=False)
    assert w._write_project(str(tmp_path / "orphan.nanoscope")) is False
    assert warned == [(1, 2)], (
        f"the dialog named {warned} — it must count the derived fit as work being dropped")
