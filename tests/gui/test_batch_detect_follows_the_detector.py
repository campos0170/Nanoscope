"""Analysis ▸ Batch detect runs the detector the Detection panel has selected.

Batch detect used to run Schmitt whatever the detector box showed. It now runs whichever of
Schmitt and CUSUM the FORM names (never ML), and everything in the flow that asked about
"the Schmitt table" asks about that detector's table.

What is pinned here:

* the menu tooltip;
* a Schmitt batch is what it was: the same params dict reaches the controller, the same
  module, the same prompt, progress and summary texts;
* a CUSUM batch gives every recording a ``detect.cusum`` table equal, row for row, to what a
  single CUSUM Detect with the same form gives that recording, with the opposite-direction
  provenance keys, and no Schmitt table;
* the batch baseline rule holds on the CUSUM path (``'batch'`` on the recordings the form was
  not filled in for);
* a recording with only a Schmitt table is "missing" for a CUSUM batch, and its Schmitt table
  and curation are left alone (and the reverse);
* the "already have a CUSUM detection" prompt, "Only missing", and the curation at stake
  counted on the selected detector's tables;
* the user open-pore reference refusal still fires, and the form's detector decides with
  another kind of table on screen;
* the closing summary counts the recordings that raised the opposite-direction advisory,
  with the function that decides the advisory, on a real CUSUM run.

And, from the review of that change (the same for either detector unless it says otherwise):

* a batch's table becomes the table its recording shows: after a pass every recording would
  show that pass's table, the detector box reads its detector there, and the multi-recording
  density figure reads one detector's tables;
* the voltage series pools the pass just made when both detectors are on every recording,
  and its provenance block always names the detector;
* a recording confirmed under one detector keeps its confirmed baseline stamp in the other
  detector's batch;
* Undo of a batch re-run brings back hand rejections nobody had committed;
* a batch-made table records the acquisition bandwidth a single Detect records, and its
  whole provenance equals the single run's apart from the run stamp and the baseline source;
* a maximum dwell not above the minimum is refused before the prompt, in the single run's
  sentence; a second batch is refused while one runs; Cancel says what Cancel says, installs
  nothing and forgets the cancelled pass.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import voltage_dep
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import (
    DetectCusum, DetectSchmitt, raises_opposite_direction_advisory, register_all,
)
from nanoscope.analysis.truth import synth_trace
from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_worker import BatchOutcome
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

S, C = "detect.schmitt", "detect.cusum"
FS = 50_000.0
WAIT_MS = 60_000

TOOLTIP = (
    "Run the selected detector (Schmitt or CUSUM) with its current parameters on every open "
    "recording that has a current channel and a known sample rate, in the background. Detect "
    "the whole -1000..-200 mV series in one pass, then plot the voltage dependence. (ML "
    "detection is not offered here — it is too slow to batch over many full recordings.)")


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._batch_controller.abort()


# ================================================================================================
# builders
# ================================================================================================

def _signal(seed: int, *, n_events: int = 12, i0_pA: float | None = None) -> np.ndarray:
    kw = {} if i0_pA is None else {"i0_pA": i0_pA}
    rec, _truth = synth_trace(fs=FS, duration_s=1.2, n_events=n_events, noise_pA=8.0,
                              dwell_range_s=(4e-4, 3e-3), depth_frac_range=(0.2, 0.4),
                              seed=seed, **kw)
    return np.ascontiguousarray(rec.data, dtype=np.float32)


def _add(win: MainWindow, tmp_path, name: str, seed: int, *, volts: float | None = None,
         **kw) -> str:
    signal = _signal(seed, **kw)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=volts)
    doc = TraceDocument(meta=meta, channels=(signal,))
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    return doc_id


def _look(win: MainWindow, doc_id: str) -> None:
    win._on_selection(doc_id, 0)
    QTest.qWait(20)


def _combo(win: MainWindow, module_id: str) -> None:
    """The detector combo, as the user moves it."""
    combo = win._analysis_panel.detection_panel.detector_combo
    combo.setCurrentIndex(combo.findData(module_id))
    QTest.qWait(20)
    assert win._analysis_panel.detection_panel.detector_choice() == module_id


def _single(win: MainWindow, doc_id: str, module_id: str):
    """What a single Detect of ``module_id`` computes on ``doc_id`` with the form as it
    stands: the module run over the form's own single-run params, with the ``pore_id`` the
    single-run controller hands it (the recording's stem). Stored nowhere."""
    params = {**win._analysis_panel._form_params(), "detector": module_id,
              "pore_id": win._session.document(doc_id).meta.stem_key}
    module = DetectSchmitt() if module_id == S else DetectCusum()
    return module.run(SimpleAnalysisContext(
        win._session.calibrated(doc_id, 0), win._session.document(doc_id).meta, params))


def _land(win: MainWindow, doc_id: str, module_id: str):
    """A stored table of ``module_id`` on ``doc_id``, made from the form as it stands."""
    res = _single(win, doc_id, module_id)
    win._session.add_result(res)
    QTest.qWait(10)
    return win._session.results.get(doc_id, 0, module_id)


def _answer(monkeypatch, label: str | None) -> list[tuple[str, str]]:
    """Answer the batch's "Only missing / Re-run all" box by button text (``None`` leaves it
    unanswered, which is Cancel). Returns ``(text, informative text)`` of each box shown."""
    seen: list[tuple[str, str]] = []

    def _exec(self):
        seen.append((self.text(), self.informativeText()))
        self._answer = next(
            (b for b in self.buttons() if b.text().replace("&", "") == label), None)
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton",
                        lambda self: getattr(self, "_answer", None))
    return seen


def _batch(win: MainWindow, qtbot) -> None:
    with qtbot.waitSignal(win._batch_controller.finished, timeout=WAIT_MS):
        win._batch_detect()
    qtbot.wait(50)


def _spans(res) -> list[tuple[int, int]]:
    df = res.payload.df
    return list(zip(df["start_idx"].astype(int).tolist(), df["end_idx"].astype(int).tolist()))


def _reject_by_hand(res, n: int) -> None:
    """``n`` rows of a stored table rejected with no reason: a person's decision. In place,
    and announced to nobody: no undo step holds it until something commits everything."""
    df = res.payload.df
    assert len(df) >= n, "setup: too few events to reject"
    df.loc[df.index[:n], "accepted"] = False


def _accepted(res) -> list[bool]:
    return [bool(v) for v in res.payload.df["accepted"].tolist()]


def _plain(v):
    """A provenance value in a form ``==`` can compare (NaN equals NaN, arrays are lists)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return repr(v)


def _provenance(res, *, without=("detected_at", "baseline_param_source")) -> dict:
    """A result's whole provenance, apart from the run stamp and the baseline source: the two
    things a batch-made table may record differently from a single Detect's."""
    return {k: _plain(v) for k, v in dict(res.payload.provenance).items() if k not in without}


def _hold(monkeypatch, module_id: str) -> threading.Event:
    """Make ``module_id``'s runs wait, cancellably, until the returned event is set: a batch
    that is still running when the next thing is asked of the window."""
    gate = threading.Event()
    cls = DetectCusum if module_id == C else DetectSchmitt
    real = cls.run

    def held(self, ctx):
        while not gate.is_set():
            check_cancelled()
            time.sleep(0.002)
        return real(self, ctx)

    monkeypatch.setattr(cls, "run", held)
    return gate


def _detector_lines(win: MainWindow, module_id: str) -> list[str]:
    """The ``Detector:`` line(s) of the Voltage dependence window's 'Pooled provenance' block
    for a series pooling ``module_id``, built by the window's own method."""
    series = voltage_dep.vdep_series(win._session, metric="n_events", module_id=module_id)
    lines = VDepWindow._provenance_lines(SimpleNamespace(_series_module_id=module_id), series)
    return [line for line in lines if "Detector:" in line]


# ================================================================================================
# the menu
# ================================================================================================

def test_the_menu_tooltip_names_both_detectors(win):
    assert win._act_batch_detect.toolTip() == TOOLTIP


# ================================================================================================
# Schmitt: unchanged
# ================================================================================================

def test_a_schmitt_batch_hands_the_controller_what_it_always_did(win, tmp_path, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    panel = win._analysis_panel
    assert panel.detection_panel.detector_choice() == S and panel.batch_detector() == S
    expected = {**panel.detection_panel.params(), **panel.baseline_panel.batch_params()}

    calls = []
    monkeypatch.setattr(
        win._batch_controller, "start",
        lambda targets, params=None, **kw: calls.append((list(targets), dict(params), kw)))
    win._batch_detect()

    assert len(calls) == 1
    targets, params, kw = calls[0]
    assert targets == [(a, 0), (b, 0)]
    assert params == expected and params["detector"] == S
    assert kw.get("module", S) == S


def test_a_schmitt_batch_says_what_it_always_said(win, tmp_path, qtbot, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    busy: list[str] = []
    win._batch_controller.started.connect(lambda _n: busy.append(win._busy_job_format))

    _batch(win, qtbot)
    n = sum(len(win._session.results.get(d, 0, S).payload.df) for d in (a, b))
    assert busy == ["Batch detecting 2 recording(s)…"]
    assert win.statusBar().currentMessage() == f"Batch detect: 2 recording(s), {n} event(s)."
    assert all(win._session.results.get(d, 0, C) is None for d in (a, b))

    # A third recording, then the prompt: two of three have a Schmitt table.
    c = _add(win, tmp_path, "c.dat", 3)
    seen = _answer(monkeypatch, None)                       # Cancel
    win._batch_detect()
    assert seen == [("2 of 3 recording(s) already have a Schmitt detection.",
                     "Re-run all of them, or only the ones that are missing one?")]
    assert not win._batch_controller.is_busy()
    assert win._session.results.get(c, 0, S) is None

    # Every recording detected, "Only missing": nothing to do, in the old words.
    kept = _land(win, c, S)
    seen.clear()
    _answer(monkeypatch, "Only missing")
    win._batch_detect()
    assert win.statusBar().currentMessage() == (
        "Batch detect: nothing to do — every recording already has a detection.")
    assert win._session.results.get(c, 0, S) is kept


# ================================================================================================
# CUSUM: the same run a single CUSUM Detect makes
# ================================================================================================

def test_a_cusum_batch_gives_each_recording_a_single_cusum_detects_table(win, tmp_path, qtbot):
    ids = [_add(win, tmp_path, f"r{k}.dat", 10 + k, n_events=8 + 3 * k) for k in range(3)]
    # The user declared an acquisition low-pass on the last one: metadata a single run stamps.
    declared = AcquisitionBandwidth(cutoff_hz=10_000.0, status="reported", source="user",
                                    filter_type="Bessel", filter_order=4)
    win._session.set_acquisition_bandwidth(ids[2], 0, declared)
    _look(win, ids[0])
    _combo(win, C)
    panel = win._analysis_panel
    assert panel.batch_detector() == C
    single = {d: _single(win, d, C) for d in ids}
    assert all(len(r.payload.df) > 0 for r in single.values()), "setup: CUSUM found nothing"
    assert len({len(r.payload.df) for r in single.values()}) > 1, "setup: identical recordings"
    busy: list[str] = []
    win._batch_controller.started.connect(lambda _n: busy.append(win._busy_job_format))

    _batch(win, qtbot)

    total = 0
    for d in ids:
        res = win._session.results.get(d, 0, C)
        assert res is not None and res.module_id == C and isinstance(res.payload, EventSet)
        assert win._session.results.get(d, 0, S) is None, "a CUSUM batch made a Schmitt table"
        assert len(res.payload.df) == len(single[d].payload.df)
        assert _spans(res) == _spans(single[d])
        np.testing.assert_allclose(res.payload.df["depth_pA"].to_numpy(),
                                   single[d].payload.df["depth_pA"].to_numpy())
        assert res.params["detector"] == C
        for key in ("cusum_k", "cusum_h", "min_dwell_s", "merge_gap_s", "polarity"):
            assert res.params[key] == single[d].params[key], key
        prov = res.payload.provenance
        for key in ("n_opposite_events", "n_opposite_isolated", "n_opposite_reference"):
            assert key in prov and prov[key] == single[d].payload.provenance[key], key
        assert res.warnings == single[d].warnings
        # ...and everything else the run recorded about itself, key for key.
        assert _provenance(res) == _provenance(single[d]), d
        total += len(res.payload.df)
    effective = win._session.results.get(ids[2], 0, C).payload.provenance[
        "acquisition_bandwidth"]["effective"]
    assert effective == declared.to_dict() and effective["source"] == "user", effective
    assert busy == ["Batch detecting 3 recording(s) with CUSUM…"]
    assert win.statusBar().currentMessage().startswith(
        f"Batch detect (CUSUM): 3 recording(s), {total} event(s).")
    # The recording on screen shows its CUSUM table, and the form still names CUSUM.
    assert panel.current_detection().module_id == C
    assert panel.detection_panel.detector_choice() == C
    assert win._dirty, "the batch's tables did not mark the session unsaved"

    # ...and the panel's own single Detect, pressed on a batch-made table, finds the same rows.
    _look(win, ids[1])
    assert panel.detection_panel.detector_choice() == C
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=WAIT_MS):
        panel.trigger_detection()
    qtbot.wait(20)
    assert _spans(win._session.results.get(ids[1], 0, C)) == _spans(single[ids[1]])


def test_a_cusum_batch_stamps_batch_on_the_recordings_the_form_is_not_about(
        win, tmp_path, qtbot):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _combo(win, C)
    bl = win._analysis_panel.baseline_panel
    bl.set_params({"baseline_rough_s": 0.2, "baseline_base_s": 0.05,
                   "baseline_param_source": "proposed"})
    assert bl.param_source() == "proposed"
    assert win._analysis_panel.batch_detection_params()[B.BATCH_SCOPE_KEY] == (a, 0)

    _batch(win, qtbot)

    ra, rb = (win._session.results.get(d, 0, C) for d in (a, b))
    assert ra is not None and rb is not None
    assert ra.payload.provenance["baseline_param_source"] == "proposed"
    assert rb.payload.provenance["baseline_param_source"] == "batch"
    assert rb.payload.provenance["baseline_rough_s"] == pytest.approx(0.2)
    assert any("COPIED FROM THE BATCH FORM" in w for w in rb.warnings)
    assert not any("COPIED FROM THE BATCH FORM" in w for w in ra.warnings)
    assert ra.params["baseline_param_source"] == "proposed"
    assert rb.params["baseline_param_source"] == "batch"
    assert B.BATCH_SCOPE_KEY not in ra.params and B.BATCH_SCOPE_KEY not in rb.params


# ================================================================================================
# "missing" is asked of the selected detector's table
# ================================================================================================

def test_a_schmitt_only_recording_is_missing_for_a_cusum_batch_and_keeps_its_schmitt_table(
        win, tmp_path, qtbot, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    schmitt = _land(win, a, S)
    _reject_by_hand(schmitt, 2)
    before = schmitt.payload.df.copy(deep=True)
    _combo(win, C)
    seen = _answer(monkeypatch, None)        # any prompt would be cancelled, and recorded

    _batch(win, qtbot)

    assert seen == [], "a CUSUM batch asked about Schmitt tables"
    assert all(win._session.results.get(d, 0, C) is not None for d in (a, b))
    after = win._session.results.get(a, 0, S)
    assert after is schmitt, "the Schmitt table was replaced"
    assert after.payload.df.equals(before), "the Schmitt table's rows or curation changed"
    assert win._session.results.get(b, 0, S) is None


def test_a_cusum_only_recording_is_missing_for_a_schmitt_batch_and_keeps_its_cusum_table(
        win, tmp_path, qtbot, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _combo(win, C)
    cusum = _land(win, a, C)
    _reject_by_hand(cusum, 1)
    before = cusum.payload.df.copy(deep=True)
    _combo(win, S)
    seen = _answer(monkeypatch, None)

    _batch(win, qtbot)

    assert seen == [], "a Schmitt batch asked about CUSUM tables"
    assert all(win._session.results.get(d, 0, S) is not None for d in (a, b))
    after = win._session.results.get(a, 0, C)
    assert after is cusum and after.payload.df.equals(before)
    assert win._session.results.get(b, 0, C) is None


def test_the_prompt_names_cusum_and_only_missing_runs_the_recordings_without_one(
        win, tmp_path, qtbot, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    b_schmitt = _land(win, b, S)                 # b: a Schmitt table only
    _combo(win, C)
    a_cusum = _land(win, a, C)                   # a: a CUSUM table
    seen = _answer(monkeypatch, "Only missing")

    _batch(win, qtbot)

    assert seen == [("1 of 2 recording(s) already have a CUSUM detection.",
                     "Re-run all of them, or only the ones that are missing one?")]
    assert win._session.results.get(a, 0, C) is a_cusum, "Only missing re-ran a's CUSUM table"
    assert win._session.results.get(b, 0, C) is not None
    assert win._session.results.get(b, 0, S) is b_schmitt
    assert win.statusBar().currentMessage().startswith("Batch detect (CUSUM): 1 recording(s), ")

    # Now every recording has one: "Only missing" has nothing to do, and says which detector.
    seen.clear()
    win._batch_detect()
    assert seen and seen[0][0] == "2 of 2 recording(s) already have a CUSUM detection."
    assert not win._batch_controller.is_busy()
    assert win.statusBar().currentMessage() == (
        "Batch detect (CUSUM): nothing to do — every recording already has a CUSUM detection.")


@pytest.mark.parametrize("ran, other", [(C, S), (S, C)])
def test_re_run_all_replaces_the_selected_detectors_table_and_leaves_the_other_alone(
        win, tmp_path, qtbot, monkeypatch, ran, other):
    """The same for either detector: the table re-run is replaced, and Undo brings back the
    table it replaced WITH the hand rejections nobody had committed -- the ones the prompt
    counted; the other detector's table is the same object throughout."""
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _combo(win, other)
    kept = _land(win, a, other)
    _combo(win, ran)
    old = {a: _land(win, a, ran), b: _land(win, b, ran)}
    _reject_by_hand(old[a], 2)                   # in place: no checkpoint, no announcement
    _reject_by_hand(old[b], 1)
    before = {d: _accepted(old[d]) for d in (a, b)}
    assert before[a].count(False) == 2 and before[b].count(False) == 1
    seen = _answer(monkeypatch, "Re-run all")

    _batch(win, qtbot)

    name = "CUSUM" if ran == C else "Schmitt"
    assert seen and seen[0][0] == f"2 of 2 recording(s) already have a {name} detection."
    assert "2 of them carry human curation (3 hand rejections)." in seen[0][1]
    for d in (a, b):
        rerun = win._session.results.get(d, 0, ran)
        assert rerun is not old[d] and _spans(rerun) == _spans(old[d])
        assert all(_accepted(rerun)), "the re-run kept a rejection it should have re-made"
    assert win._session.results.get(a, 0, other) is kept
    assert win._analysis_panel.current_detection() is win._session.results.get(a, 0, ran)

    # Undo on the recording on screen, then on the other one: each gets its table back as
    # it stood when the batch was asked for, rejections included.
    for d in (a, b):
        _look(win, d)
        assert win._analysis_panel.current_detection().module_id == ran
        win._on_edit_menu_about_to_show()
        assert win._act_undo.isEnabled(), "the batch's re-run left no undo step"
        assert win._act_undo.text() == "Undo Re-run detection"
        win._act_undo.trigger()
        QTest.qWait(40)
        restored = win._session.results.get(d, 0, ran)
        assert _spans(restored) == _spans(old[d])
        assert _accepted(restored) == before[d], (
            "Undo of the batch re-run lost hand rejections that were never committed")
    assert win._session.results.get(a, 0, other) is kept


def test_the_curation_at_stake_is_counted_on_the_selected_detectors_tables(
        win, tmp_path, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _reject_by_hand(_land(win, a, S), 2)         # a: Schmitt, two hand rejections
    _land(win, b, S)
    _combo(win, C)
    _reject_by_hand(_land(win, a, C), 1)         # a: CUSUM, one hand rejection
    _land(win, b, C)
    seen = _answer(monkeypatch, None)            # Cancel both times

    win._batch_detect()
    assert seen[-1][0] == "2 of 2 recording(s) already have a CUSUM detection."
    assert "1 of them carry human curation (1 hand rejection)." in seen[-1][1]

    _combo(win, S)
    win._batch_detect()
    assert seen[-1][0] == "2 of 2 recording(s) already have a Schmitt detection."
    assert "1 of them carry human curation (2 hand rejections)." in seen[-1][1]
    assert len(seen) == 2 and not win._batch_controller.is_busy()


# ================================================================================================
# what is refused or left out stays so
# ================================================================================================

def test_the_user_reference_refusal_still_fires_with_cusum_selected(
        win, tmp_path, monkeypatch):
    a = _add(win, tmp_path, "a.dat", 1)
    _look(win, a)
    _combo(win, C)
    mode = win._analysis_panel.baseline_panel.reference_editor.mode
    other = next(i for i in range(mode.count()) if str(mode.itemData(i)) != "automatic")
    mode.setCurrentIndex(other)
    assert win._analysis_panel.baseline_panel.reference_mode() != "automatic"
    calls = []
    monkeypatch.setattr(win._batch_controller, "start",
                        lambda *args, **kw: calls.append((args, kw)))

    win._batch_detect()

    assert calls == []
    assert win.statusBar().currentMessage() == (
        "Batch detect refused: this open-pore reference belongs to the selected "
        "recording and channel. Run Detect there, or explicitly choose Automatic reference.")
    assert win._session.results.get(a, 0, C) is None


def test_the_forms_detector_decides_with_another_table_on_screen(win, tmp_path, qtbot):
    """A Manual table on screen: the detector box still names CUSUM, and CUSUM is what runs."""
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    panel = win._analysis_panel
    added = panel.add_manual_event(a, 0, 20_000, 20_300)
    assert added.ok and panel.current_detection().module_id == "detect.manual", added.message
    manual = win._session.results.get(a, 0, "detect.manual")
    _combo(win, C)
    assert panel.current_detection().module_id == "detect.manual", "setup: Manual left the screen"
    assert panel.batch_detector() == C

    _batch(win, qtbot)

    assert all(win._session.results.get(d, 0, C) is not None for d in (a, b))
    assert all(win._session.results.get(d, 0, S) is None for d in (a, b))
    assert len(win._session.results.get(a, 0, "detect.manual").payload.df) == len(
        manual.payload.df)


# ================================================================================================
# the closing summary
# ================================================================================================

def test_the_summary_counts_the_recordings_whose_cusum_run_raised_the_advisory(
        win, tmp_path, qtbot):
    """A real CUSUM batch, with a polarity that is wrong for one recording: the advisory comes
    from the runs themselves, and the count is the advisory's own decision about each."""
    ids = [_add(win, tmp_path, "a.dat", 31),
           _add(win, tmp_path, "b.dat", 32, i0_pA=+1000.0),      # its events run the other way
           _add(win, tmp_path, "c.dat", 33)]
    _look(win, ids[0])
    _combo(win, C)
    polarity = win._analysis_panel.detection_panel.polarity_override
    polarity.setCurrentIndex(2)                                 # +1 for every recording
    QTest.qWait(20)

    _batch(win, qtbot)

    results = [win._session.results.get(d, 0, C) for d in ids]
    assert all(r is not None for r in results)
    raised = [raises_opposite_direction_advisory(r.payload.provenance) for r in results]
    assert raised[1], "setup: the wrong-polarity recording raised no advisory"
    assert not all(raised), "setup: every recording raised the advisory"
    for res, did in zip(results, raised):
        assert any("opposite-direction detection(s)" in w for w in res.warnings) == did
    n = sum(len(r.payload.df) for r in results)
    assert win.statusBar().currentMessage() == (
        f"Batch detect (CUSUM): 3 recording(s), {n} event(s). "
        f"{sum(raised)} recording(s) raised the opposite-direction advisory.")

    # The same polarity under Schmitt: no such count is stamped, and no clause is added.
    _combo(win, S)
    _batch(win, qtbot)
    n = sum(len(win._session.results.get(d, 0, S).payload.df) for d in ids)
    assert win.statusBar().currentMessage() == f"Batch detect: 3 recording(s), {n} event(s)."


def test_the_advisory_count_follows_the_decision_not_the_sentence(win, tmp_path):
    """The count reads the run's stamped numbers through the function that decides the
    advisory: a warning that merely contains the words is not counted, and a run whose
    numbers meet the trigger is, whatever its warnings say."""
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _combo(win, C)
    quiet = _single(win, a, C)
    plain = _single(win, b, C)
    assert not raises_opposite_direction_advisory(quiet.payload.provenance)
    assert not raises_opposite_direction_advisory(plain.payload.provenance)
    n = len(quiet.payload.df) + len(plain.payload.df)

    # The words alone, on a run whose numbers did not meet the trigger: no clause.
    worded = replace(plain, warnings=(
        *plain.warnings, "40 opposite-direction detection(s) against 12 in the event direction."))
    win._batch_module = C
    win._on_batch_finished(BatchOutcome(results=(quiet, worded)))
    assert win.statusBar().currentMessage() == (
        f"Batch detect (CUSUM): 2 recording(s), {n} event(s).")

    # The trigger met on the stamped numbers, with no such sentence among the warnings.
    stamped = replace(plain, payload=EventSet(plain.payload.df, {
        **plain.payload.provenance,
        "n_opposite_isolated": 40, "n_opposite_events": 40, "n_opposite_reference": 12}))
    assert raises_opposite_direction_advisory(stamped.payload.provenance)
    assert not any("opposite-direction" in w for w in stamped.warnings)
    win._batch_module = C
    win._on_batch_finished(BatchOutcome(results=(quiet, stamped)))
    assert win.statusBar().currentMessage() == (
        f"Batch detect (CUSUM): 2 recording(s), {n} event(s). "
        "1 recording(s) raised the opposite-direction advisory.")


# ================================================================================================
# a batch's table is the table its recording shows
# ================================================================================================

@pytest.mark.parametrize("first, then", [(S, C), (C, S)])
def test_a_batchs_tables_become_the_tables_the_recordings_show(
        win, tmp_path, qtbot, first, then):
    """A pass with one detector, a look at each recording (each now remembers that table),
    then a pass with the other: every recording would show the second pass's table, the
    detector box reads that detector on each, and the density figure reads one detector."""
    ids = [_add(win, tmp_path, f"r{k}.dat", 131 + k) for k in range(3)]
    panel = win._analysis_panel
    _look(win, ids[0])
    _combo(win, first)
    _batch(win, qtbot)
    for d in (ids[1], ids[2], ids[0]):
        _look(win, d)
        assert panel.current_detection().module_id == first
    assert {d: panel.chosen_tables().get((d, 0)) for d in ids} == {d: first for d in ids}, (
        "setup: a recording does not remember the first pass's table")

    _combo(win, then)
    _batch(win, qtbot)

    made = {d: win._session.results.get(d, 0, then) for d in ids}
    assert all(r is not None and int(r.payload.df["accepted"].sum()) >= 5
               for r in made.values()), "setup: too few events for the density figure"
    assert all(win._session.results.get(d, 0, first) is not None for d in ids)
    assert {d: panel.stored_to_show(d, 0).module_id for d in ids} == {d: then for d in ids}
    assert sorted((r.doc_id, r.module_id) for r in win._density_results()) == sorted(
        (d, then) for d in ids), "the density figure mixes detectors"
    for d in (ids[1], ids[2], ids[0]):
        _look(win, d)
        assert panel.current_detection() is made[d]
        assert panel.detection_panel.detector_choice() == then
        assert panel.batch_detector() == then, "a Batch detect from here would run the other one"


# ================================================================================================
# the voltage series follows the pass
# ================================================================================================

def test_the_voltage_series_pools_the_pass_just_made_and_names_its_detector(
        win, tmp_path, qtbot, monkeypatch):
    ids = [_add(win, tmp_path, f"v{k}.dat", 60 + k, volts=-200.0 * (k + 1)) for k in range(3)]
    session = win._session
    _look(win, ids[0])
    _batch(win, qtbot)                                   # Schmitt on every recording
    placeable = voltage_dep.placeable_doc_ids(session)
    assert sorted(placeable) == sorted(ids), "setup: a recording has no applied voltage"
    assert voltage_dep.series_module_id(session, placeable) == S

    _combo(win, C)
    _batch(win, qtbot)                                   # CUSUM on every recording: a tie
    assert all(session.results.get(d, 0, m) is not None for d in ids for m in (S, C))
    assert voltage_dep.series_module_id(session, placeable) == C, (
        "the voltage series still pools the Schmitt tables after a CUSUM pass")
    assert _detector_lines(win, C) == ["  Detector: CUSUM"]

    _combo(win, S)
    seen = _answer(monkeypatch, "Re-run all")
    _batch(win, qtbot)                                   # Schmitt again, every table re-made
    assert seen and seen[0][0] == "3 of 3 recording(s) already have a Schmitt detection."
    assert voltage_dep.series_module_id(session, placeable) == S
    assert _detector_lines(win, S) == ["  Detector: Schmitt"]


# ================================================================================================
# the confirmed baseline stamp belongs to the recording, not to one detector's table
# ================================================================================================

def test_a_recording_confirmed_under_schmitt_keeps_its_stamp_in_a_cusum_batch_from_another(
        win, tmp_path, qtbot):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    bl = win._analysis_panel.baseline_panel
    bl.set_params({"baseline_rough_s": 0.2, "baseline_base_s": 0.05,
                   "baseline_param_source": "proposed"})
    QTest.qWait(20)
    _batch(win, qtbot)                                   # Schmitt, launched from a
    assert [win._session.results.get(d, 0, S).payload.provenance["baseline_param_source"]
            for d in (a, b)] == ["proposed", "batch"]

    _look(win, b)                                        # the form is b's now: copied windows
    _combo(win, C)
    assert bl.param_source() == "batch"
    _batch(win, qtbot)                                   # CUSUM, launched from b

    ra, rb = (win._session.results.get(d, 0, C) for d in (a, b))
    assert ra is not None and rb is not None
    assert ra.payload.provenance["baseline_param_source"] == "proposed", (
        "a's first CUSUM table says its windows were never measured on it")
    assert ra.params["baseline_param_source"] == "proposed"
    assert ra.payload.provenance["baseline_rough_s"] == pytest.approx(0.2)
    assert ra.payload.provenance["baseline_base_s"] == pytest.approx(0.05)
    assert not any("COPIED FROM THE BATCH FORM" in w for w in ra.warnings)
    assert rb.payload.provenance["baseline_param_source"] == "batch"
    assert B.BATCH_SCOPE_KEY not in ra.params and B.BATCH_SCOPE_KEY not in rb.params


# ================================================================================================
# the acquisition bandwidth a single Detect records
# ================================================================================================

@pytest.mark.parametrize("module_id", [S, C])
def test_a_batch_made_table_records_the_declared_acquisition_bandwidth(
        win, tmp_path, qtbot, module_id):
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    declared = AcquisitionBandwidth(cutoff_hz=10_000.0, status="reported", source="user",
                                    filter_type="Bessel", filter_order=4)
    win._session.set_acquisition_bandwidth(b, 0, declared)
    _look(win, a)
    _combo(win, module_id)
    single = {d: _single(win, d, module_id) for d in (a, b)}

    _batch(win, qtbot)

    for d in (a, b):
        res = win._session.results.get(d, 0, module_id)
        assert (res.payload.provenance["acquisition_bandwidth"]
                == single[d].payload.provenance["acquisition_bandwidth"]), d
    recorded = win._session.results.get(b, 0, module_id).payload.provenance[
        "acquisition_bandwidth"]
    assert recorded["effective"] == declared.to_dict()
    assert recorded["file_header"] == AcquisitionBandwidth().to_dict()


# ================================================================================================
# refused before anything is asked or started
# ================================================================================================

@pytest.mark.parametrize("module_id", [S, C])
def test_a_max_dwell_not_above_the_min_dwell_is_refused_before_the_prompt(
        win, tmp_path, monkeypatch, module_id):
    a = _add(win, tmp_path, "a.dat", 1)
    _look(win, a)
    _combo(win, module_id)
    table = _land(win, a, module_id)             # a table of this detector: a prompt is due
    det = win._analysis_panel.detection_panel
    det.min_dwell_s.setValue(1e-3)
    det.max_dwell_s.setValue(5e-4)
    QTest.qWait(20)
    refused: list[str] = []
    win._analysis_panel.runRefused.connect(refused.append)
    win._analysis_panel.trigger_detection()      # what a single Detect says
    assert refused == [
        "Detection refused — maximum dwell (0.5 ms) must exceed minimum dwell (1 ms). "
        "Increase the maximum or set it to 0 (off)."]
    seen = _answer(monkeypatch, "Re-run all")
    calls = []
    monkeypatch.setattr(win._batch_controller, "start",
                        lambda *args, **kw: calls.append((args, kw)))

    win._batch_detect()

    assert seen == [], "the batch asked its question about a run it then refused"
    assert calls == []
    assert win.statusBar().currentMessage() == (
        "Batch detect refused — maximum dwell (0.5 ms) must exceed minimum dwell (1 ms). "
        "Increase the maximum or set it to 0 (off).")
    assert win._session.results.get(a, 0, module_id) is table

    # Maximum dwell off (0): nothing to refuse, the question is asked.
    det.max_dwell_s.setValue(0.0)
    QTest.qWait(20)
    win._batch_detect()
    assert len(seen) == 1 and len(calls) == 1


def test_a_second_batch_is_refused_while_one_is_running(win, tmp_path, qtbot, monkeypatch):
    """A CUSUM pass held open, the box moved to Schmitt, Batch detect again: refused in the run
    gate's own sentence, and the running pass finishes as the CUSUM pass it was."""
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _combo(win, C)
    gate = _hold(monkeypatch, C)
    started: list[int] = []
    win._batch_controller.started.connect(started.append)
    win._batch_detect()
    QTest.qWait(50)
    assert win._batch_controller.is_busy() and started == [2]

    _combo(win, S)
    seen = _answer(monkeypatch, "Re-run all")
    win._batch_detect()

    assert win.statusBar().currentMessage() == (
        "A batch detection is already running. This app runs one analysis job at a time, "
        "so the request for a batch detection was refused; nothing was started. Use Cancel "
        "in the window running that job, or wait for it to finish.")
    assert started == [2] and seen == []
    assert win._batch_controller.is_busy() and win._batch_module == C

    with qtbot.waitSignal(win._batch_controller.finished, timeout=WAIT_MS):
        gate.set()
    qtbot.wait(50)
    assert all(win._session.results.get(d, 0, C) is not None for d in (a, b))
    assert all(win._session.results.get(d, 0, S) is None for d in (a, b))
    assert win.statusBar().currentMessage().startswith("Batch detect (CUSUM): 2 recording(s), ")


def test_cancel_stops_a_cusum_batch_installs_nothing_and_forgets_it(
        win, tmp_path, qtbot, monkeypatch):
    """The Cancel button in the middle of a CUSUM "Re-run all". What the user is shown is
    Cancel's own line: the controller announces no cancel of its own (its worker is detached
    as it is told to stop), so the window's batch-cancelled slot is never reached."""
    a = _add(win, tmp_path, "a.dat", 1)
    b = _add(win, tmp_path, "b.dat", 2)
    _look(win, a)
    _combo(win, C)
    old = {d: _land(win, d, C) for d in (a, b)}
    gate = _hold(monkeypatch, C)
    _answer(monkeypatch, "Re-run all")
    announced: list[str] = []
    win._batch_controller.cancelled.connect(lambda: announced.append("cancelled"))
    win._batch_controller.finished.connect(lambda _o: announced.append("finished"))
    win._batch_detect()
    QTest.qWait(50)
    assert win._batch_controller.is_busy()
    assert win._batch_module == C and win._batch_form is not None

    win._cancel_load()
    QTest.qWait(200)

    # Cancel's two lines: the worker stopped inside the join, or it is still winding down.
    assert win.statusBar().currentMessage() in (
        "Cancelled.",
        "Cancelling — the run is finishing in the background. Its result is discarded.")
    assert not win._batch_controller.is_busy()
    assert not win._progress.isVisible()
    assert win._batch_module == S and win._batch_form is None and win._batch_skipped == 0

    gate.set()                                   # whatever the detached worker does now
    QTest.qWait(300)
    assert all(win._session.results.get(d, 0, C) is old[d] for d in (a, b)), (
        "a cancelled batch installed a table")
    assert all(win._session.results.get(d, 0, S) is None for d in (a, b))
    assert not win.statusBar().currentMessage().startswith("Batch detect")
    assert announced == [], "the cancelled pass announced itself after Cancel"
