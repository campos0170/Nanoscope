"""S9-19 — A run that lands on another channel of the shown recording changes nothing on the
channel shown.

MEASURED DEFECT (on e2c92cb as a screen flip; S9-01 made it stick). ``resultsChanged`` carries
``(doc_id, module_id)`` but no channel, and ``AnalysisPanel._on_results_changed`` fetched the
module's table on the SELECTED channel. A Batch detect, which runs each recording's first current
channel, or a single Run on channel 0 that finished after the person moved to channel 1, therefore
put channel 1's own older Schmitt table on screen over the CUSUM table the person was working in.
On e2c92cb the remembered choice stayed CUSUM, so the add went to CUSUM, which was not on screen
(finding 1's pattern). With S9-01 the remembered table follows the screen, so the flip also stuck:
a round trip came back on Schmitt and a save wrote Schmitt into ``event_table_choices_v1``.

THE FIX. ``Session.add_result`` says which ``(doc_id, channel, module_id)`` it is announcing for as
long as ``resultsChanged`` runs (``Session.announcing``), and the panel ignores a write to another
channel of the recording it shows.

CUSUM is reached the natural way: Schmitt ran on channel 1, then the Detection tab's detector combo
was switched to CUSUM and CUSUM ran there. Synthetic recordings only.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
SCHMITT = "detect.schmitt"
CUSUM = "detect.cusum"
B_DETECTED = [(15_000, 15_300), (40_000, 40_300)]
B_MISSED = (25_000, 25_300)          # a real event on channel 1 no table holds


def _signal(spans, seed=0):
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in spans:
        x[s:e] += 300.0
    return x.astype(np.float32)


def _two_channel_doc(tmp_path):
    a = _signal([(10_000, 10_300), (30_000, 30_300)], seed=1)
    b = _signal(B_DETECTED + [B_MISSED], seed=2)
    infos = (ChannelInfo(index=0, name="pore 1", raw_units="pA", kind=ChannelKind.CURRENT),
             ChannelInfo(index=1, name="pore 2", raw_units="pA", kind=ChannelKind.CURRENT))
    meta = RecordingMeta(
        path=tmp_path / "rig.dat", format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=infos)
    return TraceDocument(meta=meta, channels=(a, b))


def _result(win, doc_id, channel, module_id, seed):
    df, prov = detect_events(_signal(B_DETECTED, seed=seed), FS, baseline=I0, min_dwell_s=1e-4)
    return AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id=doc_id, channel=channel, params={},
        calibration_fingerprint=win._session.calibration(doc_id, channel).fingerprint(),
        payload=EventSet(df, prov))


def _settle():
    QtWidgets.QApplication.processEvents()


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _channel_1_on_cusum(win, tmp_path):
    """Channel 1: an older Schmitt run, then the combo switched to CUSUM and CUSUM run there."""
    doc_id = win._session.add_document(_two_channel_doc(tmp_path))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 1)
    _settle()
    panel = win._analysis_panel
    assert panel.current_target() == (doc_id, 1)
    win._session.add_result(_result(win, doc_id, 1, SCHMITT, 3))
    _settle()
    combo = panel._detection.detector_combo
    combo.setCurrentIndex(combo.findData(CUSUM))
    _settle()
    win._session.add_result(_result(win, doc_id, 1, CUSUM, 4))
    _settle()
    assert panel.current_detection().module_id == CUSUM
    assert panel.chosen_tables().get((doc_id, 1)) == CUSUM
    return doc_id


def _state(win, doc_id):
    panel = win._analysis_panel
    shown = panel.current_detection()
    return {"shown": None if shown is None else shown.module_id,
            "remembered": panel.chosen_tables().get((doc_id, 1)),
            "add_files_into": panel.target_module_id(doc_id, 1),
            "trace_label": panel.target_table_name(doc_id, 1)}


def _batch_lands_on_channel_0(win, doc_id):
    win._on_batch_finished(SimpleNamespace(
        results=(_result(win, doc_id, 0, SCHMITT, 5),), errors=()))
    _settle()


def _round_trip(win, doc_id):
    win._on_selection(doc_id, 0)
    _settle()
    win._on_selection(doc_id, 1)
    _settle()
    return win._analysis_panel.current_detection().module_id


def _saved_choice(win, tmp_path, channel=1):
    from nanoscope.core.project import load_project, save_project
    path = tmp_path / "p.nanoscope"
    save_project(path, win._gather_project_state())
    saved = load_project(path).workspace.extra.get("event_table_choices_v1") or []
    return {int(r["channel"]): r["module_id"] for r in saved}.get(channel)


# ================================================================================================
# a Batch detect answer for channel 0 while channel 1 is shown
# ================================================================================================
def test_a_batch_answer_for_another_channel_does_not_change_the_table_on_screen(win, tmp_path):
    doc_id = _channel_1_on_cusum(win, tmp_path)
    _batch_lands_on_channel_0(win, doc_id)
    st = _state(win, doc_id)
    assert st["shown"] == CUSUM, f"the screen flipped: {st}"


def test_the_remembered_table_the_round_trip_and_the_save_keep_cusum(win, tmp_path):
    doc_id = _channel_1_on_cusum(win, tmp_path)
    _batch_lands_on_channel_0(win, doc_id)
    st = _state(win, doc_id)
    saved = _saved_choice(win, tmp_path)
    back = _round_trip(win, doc_id)
    assert (st["remembered"], saved, back) == (CUSUM, CUSUM, CUSUM), (st, saved, back)


def test_an_add_after_it_goes_to_the_table_on_screen_and_undo_takes_it_back(win, tmp_path):
    from nanoscope.analysis.manual_events import hand_made_mask
    doc_id = _channel_1_on_cusum(win, tmp_path)
    _batch_lands_on_channel_0(win, doc_id)
    st = _state(win, doc_id)
    assert st["add_files_into"] == st["shown"] == CUSUM, st

    out = win._analysis_panel.add_manual_event(doc_id, 1, *B_MISSED)
    _settle()
    assert out.ok and out.module_id == CUSUM, (out.module_id, out.message, st)

    def hand():
        r = win._session.results.get(doc_id, 1, CUSUM)
        return int(hand_made_mask(r.payload.df).sum())

    before = hand()
    win._act_undo.trigger()
    _settle()
    assert hand() == before - 1, (win._act_undo.text(), before, hand())


# ================================================================================================
# a single Run on channel 0 that finishes after the person moved to channel 1
# ================================================================================================
def test_a_run_on_channel_0_finishing_after_a_move_to_channel_1_leaves_it_on_cusum(win, tmp_path):
    doc_id = _channel_1_on_cusum(win, tmp_path)
    win._analysis_panel._on_controller_finished(_result(win, doc_id, 0, SCHMITT, 6))
    _settle()
    st = _state(win, doc_id)
    back = _round_trip(win, doc_id)
    assert (st["shown"], st["remembered"], back) == (CUSUM, CUSUM, CUSUM), (st, back)


def test_a_run_on_the_channel_shown_still_takes_the_screen(win, tmp_path):
    """The control: a write to the channel shown is still this channel's own news."""
    doc_id = _channel_1_on_cusum(win, tmp_path)
    win._session.add_result(_result(win, doc_id, 1, SCHMITT, 7))
    _settle()
    st = _state(win, doc_id)
    assert (st["shown"], st["remembered"]) == (SCHMITT, SCHMITT), st
