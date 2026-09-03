"""H-1 + H-2 + C-6 — what a restore could not re-attach must be REPORTED and KEPT, and a
record must land on the channel it was measured on.

Three defects, one restore. They are tested together because fixing either of the first two
alone converts it into the other:

* **H-2** — a segment whose parent was re-exported SHORTER was silently rebuilt at a
  different window. Measured: ``[40 000, 60 000)`` came back as ``[40 000, 45 000)`` — a
  quarter of the analysed window — with ``meta.issues == ()``, a note stating only the new
  window, all 4 saved events reattached and 3 of them lying entirely past the end of the
  segment they were attached to; every dwell/depth/ECD in that table had been measured on
  data no longer on disk, and Ctrl+S rewrote ``SegmentRec(i1=45000)``. On a project written
  without a source fingerprint — an older project, which ``MIN_READ_VERSION = 1``
  guarantees still loads — there was no banner at all.
* **H-1** — the save then erased every record the restore could not re-attach.
  ``_capture_unrestorable`` carried records for exactly ONE enumerated reason (the file was
  not on disk) while the restore has at least seven exits. Ctrl+S wrote ``segments=[]``,
  ``results=[]`` and reported *"Saved project (0 result(s))"*.
* **C-6** — records were filed under a bare channel INDEX, so a re-export that reordered
  the signal columns put a ``gain=0.5, 'pA'`` calibration on the VOLTAGE channel (the
  -600 mV trace read -301.0 pA), left the real current channel at gain 1.0, reattached the
  detection to the voltage trace — and, because the stored fingerprint matched the
  misplaced live calibration, positively CERTIFIED those events as current-valid.

The pure halves are pinned in ``tests/core/test_channel_identity.py`` and
``tests/core/test_segment_spec_precondition.py``. This file drives a real MainWindow.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.layer import ViewMode
from nanoscope.core.project import (
    CalibrationRec, DocumentRef, MANIFEST_NAME, ProjectState, ResultRec, SegmentRec,
    load_project, save_project,
)
from nanoscope.core.types import ChannelKind, Provenance
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
REAL_ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")


# ================================================================================================
# builders — a real CSV on disk, so the restore genuinely reopens by path
# ================================================================================================

def _write_csv(path: Path, n: int, cols: tuple[str, ...] = ("voltage", "current")) -> None:
    rng = np.random.default_rng(0)
    cur = np.full(n, 100.0) + rng.normal(0, 1.0, n)
    for s in range(1_000, n - 600, 6_000):
        cur[s:s + 300] -= 30.0
    data = {"time": np.arange(n, dtype=np.float64) / FS}
    for c in cols:
        data[c] = np.full(n, -600.0) if c == "voltage" else cur
    pd.DataFrame(data).to_csv(path, index=False)


def _win(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _open(w: MainWindow, qtbot, path: Path) -> str:
    with qtbot.waitSignal(w._session.documentOpened, timeout=30_000):
        w.open_paths([str(path)])
    qtbot.wait(30)
    doc_id = w._session.doc_id_for_source(str(Path(path).resolve()))
    assert doc_id is not None
    return doc_id


def _reopen(qtbot, proj: Path) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.projectRestored, timeout=60_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    return w


def _detect(w: MainWindow, qtbot, min_dwell_s: float = 5e-4) -> None:
    w._analysis_panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=90_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)


def _current_channel(w: MainWindow, doc_id: str) -> int:
    return next(c.index for c in w._session.document(doc_id).meta.channels
                if c.kind is ChannelKind.CURRENT)


def _manifest(p: Path) -> dict:
    with zipfile.ZipFile(p) as zf:
        return json.loads(zf.read(MANIFEST_NAME))


def _banner(w: MainWindow) -> str:
    """Everything the restore said, as one string. Read off the spy the info bar keeps."""
    label = getattr(w._info_bar, "_label", None)
    return "" if label is None else label.text()


def _segments(w: MainWindow) -> list:
    return [d for d in w._session.documents() if hasattr(d, "origin")]


def _carried(w: MainWindow) -> dict[str, int]:
    return {k: len(v) for k, v in (w._carried_over or {}).items()}


# ================================================================================================
# H-2 — a segment that no longer fits refuses, is reported, and its definition survives
# ================================================================================================

@pytest.fixture
def shortened_parent(qapp, qtbot, tmp_path):
    """A project holding a curated segment [40 000, 60 000) whose parent is then re-exported
    at 45 000 samples over the same name — the owner's own hazard (the codebase's comment
    at ``_begin_restore`` names re-exporting from Clampex over the same filename)."""
    rec = tmp_path / "rec.csv"
    _write_csv(rec, 100_000, cols=("current",))
    w = _win(qtbot)
    doc = _open(w, qtbot, rec)
    seg = w._session.create_segment(doc, 40_000, 60_000, name="cut")
    w._create_default_layers(w._session.document(seg))
    w._on_selection(seg, 0)
    _detect(w, qtbot)
    res = list(w._session.results.for_doc(seg))
    assert res and len(res[0].payload.df) >= 3
    n_events = len(res[0].payload.df)
    proj = tmp_path / "p.nanoscope"
    assert w._write_project(str(proj))
    w._analysis_controller.abort()
    _write_csv(rec, 45_000, cols=("current",))       # THE TRIGGER
    return proj, rec, n_events


def test_a_segment_that_no_longer_fits_is_refused_not_quietly_rebuilt(shortened_parent, qtbot):
    proj, _rec, _n = shortened_parent
    w = _reopen(qtbot, proj)
    assert _segments(w) == [], (
        "the segment was rebuilt at a DIFFERENT window and its events reattached to it")
    text = _banner(w)
    assert "could not be rebuilt" in text and "cut" in text
    assert "60,000" in text and "45,000" in text, text
    w._controller.abort()


def test_the_refused_segment_and_its_results_survive_the_next_save(shortened_parent, qtbot,
                                                                   tmp_path):
    """H-1, reached through H-2 — which is why the order matters. Before: Ctrl+S wrote
    ``segments=[]``, ``results=[]`` and said *"Saved project (0 result(s))"*; the segment's
    definition and its curated events were gone from the file permanently, unrecoverable
    even by restoring the original recording, because the manifest no longer described
    them."""
    proj, _rec, n_events = shortened_parent
    w = _reopen(qtbot, proj)
    assert _carried(w).get("segments") == 1, _carried(w)

    again = tmp_path / "again.nanoscope"
    assert w._write_project(str(again))
    m = _manifest(again)
    assert len(m["segments"]) == 1
    spec = m["segments"][0]
    assert (spec["i0"], spec["i1"]) == (40_000, 60_000), (
        "the save rewrote the analysed window to the one that happened to fit")
    assert len(m["results"]) == 1
    assert len(load_project(again).results[0].events) == n_events
    assert "kept in it, not dropped" in w.statusBar().currentMessage()
    w._controller.abort()


def test_the_refusal_survives_a_project_with_no_source_fingerprint(qapp, qtbot, tmp_path):
    """The verifier's stronger case, and the reason this is not merely a banner bug: on a
    project written without a ``SourceFingerprint`` — an older project, which frozen
    contract 8 guarantees still loads — ``_note_source_change`` returns early and the whole
    truncation used to be unannounced end to end."""
    rec = tmp_path / "old.csv"
    _write_csv(rec, 100_000, cols=("current",))
    proj = tmp_path / "old.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(path=str(rec.resolve()), loaded_range=None, source=None)],
        segments=[SegmentRec(segment_uid="seg-x", parent_key=str(rec.resolve()),
                             i0=40_000, i1=60_000, name="cut", t0_abs_s=0.8)]))
    _write_csv(rec, 45_000, cols=("current",))

    w = _reopen(qtbot, proj)
    assert _segments(w) == []
    assert "could not be rebuilt" in _banner(w), _banner(w)
    assert _carried(w).get("segments") == 1
    w._controller.abort()


def test_a_segment_whose_parent_is_unchanged_still_rebuilds(qapp, qtbot, tmp_path):
    """CONTROL — the precondition must cost nothing on an ordinary reopen."""
    rec = tmp_path / "fine.csv"
    _write_csv(rec, 100_000, cols=("current",))
    w = _win(qtbot)
    doc = _open(w, qtbot, rec)
    w._session.create_segment(doc, 40_000, 60_000, name="cut")
    proj = tmp_path / "fine.nanoscope"
    assert w._write_project(str(proj))
    w._controller.abort()

    w2 = _reopen(qtbot, proj)
    segs = _segments(w2)
    assert len(segs) == 1
    assert (segs[0].origin.i0, segs[0].origin.i1) == (40_000, 60_000)
    assert segs[0].meta.n_samples == 20_000
    assert w2._carried_over is None, "a healthy restore must carry nothing"
    w2._controller.abort()


# ================================================================================================
# H-1 — the carry is a DIFF, so every exit is covered, including ones nobody enumerated
# ================================================================================================

def test_a_healthy_round_trip_carries_nothing_and_duplicates_nothing(qapp, qtbot, tmp_path):
    """The control the diff formulation has to pass first: everything installs, so nothing
    is carried, so a re-save writes exactly what the first one did."""
    rec = tmp_path / "clean.csv"
    _write_csv(rec, 60_000)
    w = _win(qtbot)
    doc = _open(w, qtbot, rec)
    ch = _current_channel(w, doc)
    w._session.set_calibration(doc, ch, ChannelCalibration(0.5, 0.0, "pA", Provenance.UNKNOWN,
                                                           "a note"), write_sidecar=False)
    w._on_selection(doc, ch)
    _detect(w, qtbot)
    proj = tmp_path / "clean.nanoscope"
    assert w._write_project(str(proj))
    first = _manifest(proj)
    w._analysis_controller.abort()

    w2 = _reopen(qtbot, proj)
    assert w2._carried_over is None, _carried(w2)
    again = tmp_path / "clean2.nanoscope"
    assert w2._write_project(str(again))
    second = _manifest(again)
    for key in ("documents", "segments", "calibrations", "layers", "results", "fits"):
        assert len(second[key]) == len(first[key]), (
            f"{key}: {len(second[key])} vs {len(first[key])}")
    w2._controller.abort()


def test_a_record_whose_channel_is_gone_is_kept_rather_than_erased(qapp, qtbot, tmp_path):
    """Trigger 2, measured before: the voltage column dropped on re-export, the calibration
    and an 11-event detection ``continue``d in SILENCE, the banner mentioned only the byte
    size, and the next save wrote ``results=0`` with the calibration gone."""
    m = tmp_path / "m.csv"
    _write_csv(m, 60_000)
    w = _win(qtbot)
    doc = _open(w, qtbot, m)
    ch = _current_channel(w, doc)
    w._session.set_calibration(doc, ch, ChannelCalibration(
        0.5, -1.0, "pA", Provenance.UNKNOWN, "cal on CURRENT"), write_sidecar=False)
    w._on_selection(doc, ch)
    _detect(w, qtbot)
    proj = tmp_path / "m.nanoscope"
    assert w._write_project(str(proj))
    n_cal = len(_manifest(proj)["calibrations"])
    w._analysis_controller.abort()

    # Drop BOTH signal columns' worth of identity: re-export with only the voltage channel,
    # so the record's own channel genuinely no longer exists anywhere.
    _write_csv(m, 60_000, cols=("voltage",))
    for sc in tmp_path.glob("*.npcal.json"):
        sc.unlink()

    w2 = _reopen(qtbot, proj)
    assert "no longer has and were NOT applied" in _banner(w2), _banner(w2)
    carried = _carried(w2)
    assert carried.get("results") == 1, carried
    assert carried.get("calibrations", 0) >= 1, carried

    again = tmp_path / "m2.nanoscope"
    assert w2._write_project(str(again))
    back = load_project(again)
    assert len(back.results) == 1, "the curated detection was erased by the save"
    kept = [c for c in back.calibrations if c.note == "cal on CURRENT"]
    assert kept and kept[0].gain == 0.5, "the measured calibration was erased by the save"
    assert len(back.calibrations) >= n_cal
    w2._controller.abort()


def test_a_malformed_result_table_is_kept_rather_than_erased(qapp, qtbot, tmp_path):
    """A skip path nobody enumerated: ``EventSet(...)`` raising on a damaged frame was a
    silent ``continue``. Under a diff it needs no rule of its own."""
    rec = tmp_path / "bad.csv"
    _write_csv(rec, 40_000, cols=("current",))
    proj = tmp_path / "bad.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(path=str(rec.resolve()))],
        results=[ResultRec(doc_id=str(rec.resolve()), channel=0, module_id="detect.schmitt",
                           events=pd.DataFrame({"not_an_event_column": [1, 2, 3]}))]))

    w = _reopen(qtbot, proj)
    live = w._session.doc_id_for_source(str(rec.resolve()))
    assert list(w._session.results.for_doc(live)) == [], "premise: the frame is unusable"
    assert _carried(w).get("results") == 1, _carried(w)

    again = tmp_path / "bad2.nanoscope"
    assert w._write_project(str(again))
    assert len(load_project(again).results) == 1, "an unreadable table was erased by the save"
    w._controller.abort()


def test_a_missing_source_file_is_still_carried(qapp, qtbot, tmp_path):
    """CONTROL / PASSES BOTH — the one reason the old enumeration DID cover. The diff must
    not have lost it while gaining the other six."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(a, 40_000, cols=("current",))
    _write_csv(b, 40_000, cols=("current",))
    w = _win(qtbot)
    doc_a = _open(w, qtbot, a)
    _open(w, qtbot, b)
    w._on_selection(doc_a, 0)
    _detect(w, qtbot)
    proj = tmp_path / "two.nanoscope"
    assert w._write_project(str(proj))
    w._analysis_controller.abort()
    a.unlink()

    w2 = _reopen(qtbot, proj)
    assert _carried(w2).get("documents") == 1
    assert _carried(w2).get("results") == 1
    again = tmp_path / "two2.nanoscope"
    assert w2._write_project(str(again))
    back = load_project(again)
    assert len(back.documents) == 2 and len(back.results) == 1
    assert "kept in the project, not dropped" in w2.statusBar().currentMessage()
    w2._controller.abort()


def test_the_carry_is_computed_from_the_state_not_from_a_list_of_reasons():
    """THE INVARIANT, asserted structurally. The old code was
    ``lost = set(self._restore_missing)`` — one remembered reason for seven exits. What
    makes the fix durable is that the carry walks ProjectState's own fields and subtracts
    what was installed, so a record TYPE nobody here remembered is carried on the day it is
    added, and so is a skip path written next year."""
    import inspect

    src = inspect.getsource(MainWindow._capture_unrestorable)
    assert "_restored_records" in src
    assert "fields" in src and "is_dataclass" in src, (
        "the carry must be derived from ProjectState's fields, not from a hand-written list")
    for name in ("documents", "segments", "calibrations", "layers", "results", "fits"):
        assert f'"{name}"' not in src, (
            f"{name!r} is enumerated in _capture_unrestorable — an enumeration silently "
            f"drops any record type nobody remembered")
    merge = inspect.getsource(MainWindow._gather_project_state)
    assert "_carried_over" in merge and "getattr(state, name" in merge, (
        "the write-back must walk the same fields the carry does")


# ================================================================================================
# C-6 — a record lands on the channel it was measured on
# ================================================================================================

@pytest.fixture
def reordered(qapp, qtbot, tmp_path):
    """A project whose source is then re-exported with its signal columns SWAPPED — same
    row count, same channel count, byte-identical size; only the order moved."""
    o = tmp_path / "o.csv"
    _write_csv(o, 40_000, cols=("voltage", "current"))
    size_before = o.stat().st_size
    w = _win(qtbot)
    doc = _open(w, qtbot, o)
    ch = _current_channel(w, doc)
    assert ch == 1, "premise: current is the second column"
    w._session.set_calibration(doc, ch, ChannelCalibration(
        0.5, -1.0, "pA", Provenance.UNKNOWN, "cal on CURRENT"), write_sidecar=False)
    w._on_selection(doc, ch)
    _detect(w, qtbot)
    n_events = len(w._session.results.get(doc, ch, "detect.schmitt").payload.df)
    proj = tmp_path / "o.nanoscope"
    assert w._write_project(str(proj))
    w._analysis_controller.abort()

    _write_csv(o, 40_000, cols=("current", "voltage"))
    assert o.stat().st_size == size_before, "premise: the file is the same size"
    for sc in tmp_path.glob("*.npcal.json"):
        sc.unlink()
    return proj, o, n_events


def test_a_pA_calibration_no_longer_lands_on_the_voltage_channel(reordered, qtbot):
    """THE MEASURED FAILURE. Before: ch1 (now voltage) held gain 0.5 / 'pA' /
    'cal on CURRENT' and read -301.0 pA off the -600 mV trace, while the real current
    channel sat at gain 1.0 with units ''."""
    proj, _o, _n = reordered
    w = _reopen(qtbot, proj)
    live = w._session.doc_id_for_source(str(_o.resolve()))
    chans = {c.index: (c.name, c.kind) for c in w._session.document(live).meta.channels}
    assert chans[0][1] is ChannelKind.CURRENT and chans[1][1] is ChannelKind.VOLTAGE

    on_current = w._session.calibration(live, 0)
    on_voltage = w._session.calibration(live, 1)
    assert (on_current.gain, on_current.target_units) == (0.5, "pA")
    assert on_current.note == "cal on CURRENT"
    assert on_voltage.gain == 1.0 and on_voltage.target_units == ""
    first = np.asarray(w._session.calibrated_array(live, 1))[:3]
    assert np.all(np.abs(first + 600.0) < 1e-3), (
        f"the voltage trace is being scaled by a pA calibration: {first}")
    w._controller.abort()


def test_the_detection_follows_its_own_channel_and_is_not_certified_on_the_other(reordered,
                                                                                 qtbot):
    """The half that makes this High rather than Medium: the restored result's fingerprint
    MATCHED the misplaced live calibration, so the staleness system affirmatively declared
    events measured on the current channel to be valid against the voltage trace."""
    proj, o, n_events = reordered
    w = _reopen(qtbot, proj)
    live = w._session.doc_id_for_source(str(o.resolve()))
    results = list(w._session.results.for_doc(live))
    assert len(results) == 1
    assert results[0].channel == 0, "the detection stayed on the index, not on its channel"
    assert len(results[0].payload.df) == n_events
    assert results[0].calibration_fingerprint == w._session.calibration(live, 0).fingerprint()
    assert "re-attached by channel name" in _banner(w), _banner(w)
    w._controller.abort()


def test_a_reordered_export_is_saved_back_with_the_new_identity(reordered, qtbot, tmp_path):
    import dataclasses

    from nanoscope.core.project import CalibrationRec
    _fields = {f.name for f in dataclasses.fields(CalibrationRec)}
    assert {"channel_name", "channel_kind"} <= _fields, (
        "a saved calibration still records only WHICH POSITION it came from")
    proj, o, _n = reordered
    w = _reopen(qtbot, proj)
    again = tmp_path / "o2.nanoscope"
    assert w._write_project(str(again))
    back = load_project(again)
    cal = next(c for c in back.calibrations if c.note == "cal on CURRENT")
    assert (cal.channel, cal.channel_name, cal.channel_kind) == (0, "current", "current")
    assert back.results[0].channel == 0
    w._controller.abort()


def test_a_pre_v7_project_still_restores_by_index_and_says_nothing(qapp, qtbot, tmp_path):
    """CONTROL / PASSES BOTH — frozen contract 8. A project written before the identity
    existed carries none, and
    must restore exactly as it always did — no relocation, no refusal, no banner. A false
    alarm on every old project is how a real one gets ignored."""
    rec = tmp_path / "old.csv"
    _write_csv(rec, 40_000)
    proj = tmp_path / "v6.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(path=str(rec.resolve()))],
        calibrations=[CalibrationRec(doc_id=str(rec.resolve()), channel=1, gain=0.5,
                                     offset=-1.0, target_units="pA", provenance="unknown",
                                     note="pre-v7")]))
    with zipfile.ZipFile(proj) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME))
    manifest["version"] = 6
    for entry in manifest["calibrations"]:
        entry.pop("channel_name", None)
        entry.pop("channel_kind", None)
    with zipfile.ZipFile(proj, "w") as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest))

    w = _reopen(qtbot, proj)
    live = w._session.doc_id_for_source(str(rec.resolve()))
    assert w._session.calibration(live, 1).gain == 0.5
    assert w._session.calibration(live, 1).note == "pre-v7"
    assert "channel" not in _banner(w).lower() or "re-attached" not in _banner(w)
    assert w._carried_over is None
    w._controller.abort()


# ================================================================================================
# the whole round trip — every kind of state this project can hold
# ================================================================================================

def _populate(w: MainWindow, qtbot, tmp_path, rec: Path) -> dict:
    """Detections, curation with a hand-written flag token, NESTED segments, calibrations
    with notes, manual measurements, layer view state, an ML result and a baseline
    fingerprint. Returns everything the reopen has to reproduce."""
    from nanoscope.analysis.events import EventSet
    from nanoscope.analysis.stats import FitResult, fit_to_json  # noqa: F401
    from nanoscope.core.analysis import AnalysisResult
    from nanoscope.gui.main_window import _stored_baseline_fingerprint

    doc = _open(w, qtbot, rec)
    ch = _current_channel(w, doc)
    w._session.set_calibration(doc, ch, ChannelCalibration(
        0.5, -1.0, "pA", Provenance.MEASURED, "electrode replaced — 1 MΩ standard"),
        write_sidecar=False)
    w._on_selection(doc, ch)
    _detect(w, qtbot)

    res = w._session.results.get(doc, ch, "detect.schmitt")
    df = res.payload.df
    assert len(df) >= 3
    ids = df["event_id"].tolist()
    w._analysis_panel.set_event_accepted(ids[1], False)
    df.iat[0, df.columns.get_loc("flag")] = "manual;résumé_flag;molecule=apo"
    w._analysis_panel.refresh_after_edit(ids[0])

    # An ML result on the same channel — a second EventSet under a different module id.
    ml = EventSet(df.head(2).copy(deep=True), dict(res.payload.provenance or {}))
    w._session.add_result(AnalysisResult(
        module_id="detect.keras", module_version="1.0", doc_id=doc, channel=ch,
        params={"model_path": "/models/apo.keras", "threshold": 0.7},
        calibration_fingerprint=w._session.calibration(doc, ch).fingerprint(),
        payload=ml, warnings=("the open-pore level was never probed",),
        # The open-pore stamp an EventSet result carries is DERIVED from its own provenance
        # (the project persists provenance verbatim, so no record field holds it). Stamping
        # it here with the same helper the restore uses is what makes the comparison below a
        # round-trip test rather than a test of the helper.
        baseline_fingerprint=_stored_baseline_fingerprint(ml.provenance)))

    # A DERIVED table result (fit.levels) with its own baseline fingerprint. The rows must
    # TILE their event exactly — `cusum.reconcile_levels` drops a level group that does not,
    # and a fixture that ignored that would be testing the reconciler, not the round trip.
    e0, e1 = int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"])
    mid = (e0 + e1) // 2
    levels = pd.DataFrame({"event_id": [ids[0], ids[0]], "level": [0, 1],
                           "start_idx": [e0, mid], "end_idx": [mid, e1],
                           "mean_pA": [70.0, 65.5]})
    w._session.add_result(AnalysisResult(
        module_id="fit.levels", module_version="", doc_id=doc, channel=ch, params={},
        calibration_fingerprint=w._session.calibration(doc, ch).fingerprint(),
        payload=levels, baseline_fingerprint="sha256:0a7dc4d5a803"))

    # Manual measurements, through the real cursor tool. Taken BEFORE the segments
    # exist, so the tool is measuring the recording the selection is on.
    w._on_selection(doc, ch)
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    vals = w._session.x_axis(doc).values
    for a, b in ((1_000, 1_300), (7_000, 7_300)):
        w._trace_view.set_measure_region(float(vals[a]), float(vals[b]))
        qtbot.wait(5)
        assert mp.add_current() is True
    assert len(mp.measurements_df) == 2

    # Nested segments, each with its own calibration edit.
    seg = w._session.create_segment(doc, 5_000, 25_000, name="cut")
    w._create_default_layers(w._session.document(seg))
    nested = w._session.create_segment(seg, 2_000, 9_000, name="nested")
    w._create_default_layers(w._session.document(nested))
    w._session.set_calibration(nested, 0, ChannelCalibration(
        0.037, 0.0, "nS", Provenance.UNKNOWN, "segment-local"), write_sidecar=False)

    # Layer view state + a non-default view mode.
    w._session.set_layer_color(f"{doc}#{ch}", "#E57373")
    w._session.set_layer_time_offset(f"{doc}#{ch}", 0.25)
    w._session.set_view_mode(ViewMode.STACKED)
    w._on_selection(doc, ch)

    return {
        "doc": doc, "ch": ch,
        "events": df.copy(deep=True),
        "ml": ml.df.copy(deep=True),
        # An EventSet result's baseline stamp travels INSIDE its provenance (the project
        # persists provenance verbatim), not as a record field -- so what the round trip has
        # to reproduce is the value the live session derived, whatever it is.
        "ml_baseline": w._session.results.get(doc, ch, "detect.keras").baseline_fingerprint,
        "levels": levels.copy(deep=True),
        "meas": mp.measurements_df.copy(deep=True),
        "seg_uid": w._session.document(seg).origin.segment_uid,
        "nested_uid": w._session.document(nested).origin.segment_uid,
    }


def _assert_round_trip(w2: MainWindow, rec: Path, want: dict) -> None:
    live = w2._session.doc_id_for_source(str(rec.resolve()))
    assert live is not None
    ch = want["ch"]

    cal = w2._session.calibration(live, ch)
    assert (cal.gain, cal.offset, cal.target_units) == (0.5, -1.0, "pA")
    assert cal.note == "electrode replaced — 1 MΩ standard"
    assert cal.provenance is Provenance.MEASURED

    got = w2._session.results.get(live, ch, "detect.schmitt")
    assert got is not None
    assert_frame_equal(got.payload.df, want["events"])
    assert got.payload.df.iloc[0]["flag"] == "manual;résumé_flag;molecule=apo"

    ml = w2._session.results.get(live, ch, "detect.keras")
    assert ml is not None
    assert_frame_equal(ml.payload.df, want["ml"])
    assert ml.params["model_path"] == "/models/apo.keras"
    assert ml.warnings == ("the open-pore level was never probed",)
    assert ml.baseline_fingerprint == want["ml_baseline"]
    assert ml.baseline_fingerprint, "the open-pore stamp was dropped on reopen"

    lv = w2._session.results.get(live, ch, "fit.levels")
    assert lv is not None
    assert_frame_equal(lv.payload, want["levels"])
    assert lv.baseline_fingerprint == "sha256:0a7dc4d5a803"

    segs = {d.origin.segment_uid: d for d in _segments(w2)}
    assert want["seg_uid"] in segs and want["nested_uid"] in segs
    assert segs[want["seg_uid"]].meta.n_samples == 20_000
    assert segs[want["nested_uid"]].meta.n_samples == 7_000
    nested_live = segs[want["nested_uid"]].doc_id
    ncal = w2._session.calibration(nested_live, 0)
    assert (ncal.gain, ncal.target_units, ncal.note) == (0.037, "nS", "segment-local")

    layer = next(l for l in w2._session.layers() if l.doc_id == live and l.channel == ch)
    assert layer.color == "#E57373"
    assert layer.time_offset_s == pytest.approx(0.25)
    assert w2._session.view_mode is ViewMode.STACKED

    assert_frame_equal(w2._analysis_panel.measure_panel.measurements_df, want["meas"])
    assert w2._carried_over is None, _carried(w2)


def test_the_whole_project_round_trips_field_by_field(qapp, qtbot, tmp_path):
    """CONTROL / PASSES BOTH. Not a repro — the demonstration the brief asks for, that this
    batch moved nothing on a HEALTHY round trip. It has to pass against the pre-fix package
    too; the day it stops, one of the four fixes above has cost something."""
    rec = tmp_path / "full.csv"
    _write_csv(rec, 60_000)
    w = _win(qtbot)
    want = _populate(w, qtbot, tmp_path, rec)
    proj = tmp_path / "full.nanoscope"
    assert w._write_project(str(proj))
    w._analysis_controller.abort()
    w._controller.abort()

    w2 = _reopen(qtbot, proj)
    _assert_round_trip(w2, rec, want)
    w2._controller.abort()


@pytest.mark.corpus
def test_the_whole_project_round_trips_on_the_real_recording(qapp, qtbot, tmp_path):
    """CONTROL / PASSES BOTH. The same proof on the owner's own -600 mV apo-ferritin file
    (2,993,798 samples, 50 kHz, 3 channels, current at index 2). Worked on a COPY — the
    recording itself is read-only, and every calibration here is set with
    ``write_sidecar=False`` so nothing is written next to his data."""
    if not REAL_ABF.exists():
        pytest.skip(f"{REAL_ABF.name} is not on this machine")
    rec = tmp_path / REAL_ABF.name
    shutil.copy2(REAL_ABF, rec)

    w = _win(qtbot)
    want = _populate(w, qtbot, tmp_path, rec)
    assert w._session.document(want["doc"]).meta.n_samples == 2_993_798
    assert want["ch"] == 2, "current is channel 2 of 3 on this recording"
    proj = tmp_path / "real.nanoscope"
    assert w._write_project(str(proj))
    w._analysis_controller.abort()
    w._controller.abort()

    w2 = _reopen(qtbot, proj)
    _assert_round_trip(w2, rec, want)
    w2._controller.abort()
    assert not list(tmp_path.glob("*.npcal.json")), "the round trip wrote a sidecar"
