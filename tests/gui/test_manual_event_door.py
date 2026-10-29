"""The one add door for hand-made events (docs/MANUAL_EVENT_TABLE_DESIGN.md, sections 3-5).

`AnalysisPanel.add_manual_event(doc_id, channel, start, end)` files a span as an event row a
person drew: into the channel's chosen table (Schmitt, CUSUM, a study table...) or, when there
is none, into the channel's Manual table, which it creates. Wired through a real MainWindow.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import MANUAL_MODULE_ID, SCHEMA, EventSet
from nanoscope.analysis.manual_events import hand_made_mask
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0

_DETECTED = [(10_000, 10_300), (30_000, 30_300)]
_MISSED = (20_000, 20_300)            # a real event the detector was not given


def _signal(spans, *, n=N, depth=DEPTH, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] += depth                       # I0 < 0: events run toward zero
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name="door.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _open(win: MainWindow, tmp_path, *, detect=True):
    signal = _signal(_DETECTED + [_MISSED])
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    if detect:
        hole = _signal(_DETECTED, seed=1)         # the detector sees only two of the three
        df, prov = detect_events(hole, FS, baseline=I0, min_dwell_s=1e-4)
        win._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
            payload=EventSet(df, prov)))
    return doc_id


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _table(win, doc_id, module_id):
    res = win._session.results.get(doc_id, 0, module_id)
    return None if res is None else res.payload.df


# ------------------------------------------------------------------------------------------------
# a table is shown: the event lands in it
# ------------------------------------------------------------------------------------------------
def test_the_door_adds_a_measured_hand_made_row_to_the_shown_table(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    before = _table(win, doc_id, "detect.schmitt")
    assert len(before) == 2

    out = panel.add_manual_event(doc_id, 0, *_MISSED)

    assert out.ok and out.module_id == "detect.schmitt" and not out.created_table
    df = _table(win, doc_id, "detect.schmitt")
    assert len(df) == 3
    row = df.loc[df["event_id"] == out.event_id].iloc[0]
    assert bool(row["added"]) and row["source"] == "manual" and row["review"] == "reviewed"
    assert bool(row["accepted"])
    assert (int(row["start_idx"]), int(row["end_idx"])) == _MISSED
    assert row["depth_pA"] == pytest.approx(DEPTH, rel=0.1)
    assert row["bounds_source"] == "start=drawn;end=drawn"
    assert hand_made_mask(df).sum() == 1
    # time order, not append order
    assert list(df["start_idx"]) == sorted(df["start_idx"])
    assert "Schmitt" in out.message and str(out.event_id) in out.message
    # no Manual table was made behind the user's back
    assert win._session.results.get(doc_id, 0, MANUAL_MODULE_ID) is None


def test_the_door_is_one_undo_step(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    assert len(_table(win, doc_id, "detect.schmitt")) == 3
    win._act_undo.trigger()
    assert len(_table(win, doc_id, "detect.schmitt")) == 2


@pytest.mark.parametrize("span, fragment", [
    ((10_100, 10_200), "overlaps event"),
    ((500, 500), "empty"),
])
def test_the_door_refuses_and_says_why(win, tmp_path, span, fragment):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    seen = []
    panel.runRefused.connect(seen.append)
    out = panel.add_manual_event(doc_id, 0, *span)
    assert not out.ok and fragment in out.message
    assert panel.last_missed_refusal == out.message and seen == [out.message]
    assert len(_table(win, doc_id, "detect.schmitt")) == 2


def test_the_door_refuses_a_span_that_is_not_on_the_shown_channel(win, tmp_path):
    doc_id = _open(win, tmp_path)
    out = win._analysis_panel.add_manual_event("some-other-doc", 0, *_MISSED)
    assert not out.ok and "not on the channel" in out.message


# ------------------------------------------------------------------------------------------------
# no table: the Manual table is created
# ------------------------------------------------------------------------------------------------
def test_with_no_table_the_door_creates_the_manual_table(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert win._session.results.get(doc_id, 0, MANUAL_MODULE_ID) is None

    out = panel.add_manual_event(doc_id, 0, *_MISSED)

    assert out.ok and out.created_table and out.module_id == MANUAL_MODULE_ID
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    df = res.payload.df
    assert len(df) == 1 and df["source"].iloc[0] == "manual"
    assert df["recording"].iloc[0] == "door.dat"            # the file name, not the doc_id
    assert df["pore_id"].iloc[0] == win._session.document(doc_id).meta.stem_key
    prov = res.payload.provenance
    assert prov["signal_basis"] == "raw" and prov["noise_rms_pA"] > 0
    assert prov["baseline_fingerprint"] == res.baseline_fingerprint
    assert panel.current_detection().module_id == MANUAL_MODULE_ID      # shown, and chosen
    assert "analyzed_duration_s" not in prov              # nobody knows how much was searched
    assert df["depth_pA"].iloc[0] == pytest.approx(DEPTH, rel=0.1)


def test_a_second_manual_add_reuses_the_table(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    out = panel.add_manual_event(doc_id, 0, *_DETECTED[0])
    assert out.ok and not out.created_table
    df = _table(win, doc_id, MANUAL_MODULE_ID)
    assert list(df["start_idx"]) == [10_000, 20_000]


def test_undoing_the_first_manual_add_leaves_an_empty_manual_table(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    win._act_undo.trigger()
    df = _table(win, doc_id, MANUAL_MODULE_ID)
    assert df is not None and len(df) == 0


def test_a_refused_first_add_leaves_no_table_behind(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    out = win._analysis_panel.add_manual_event(doc_id, 0, 500, 500)
    assert not out.ok
    assert win._session.results.get(doc_id, 0, MANUAL_MODULE_ID) is None


# ------------------------------------------------------------------------------------------------
# the Table selector offers Manual and remembers a choice per channel
# ------------------------------------------------------------------------------------------------
def _selector_labels(panel):
    sel = panel._result_selector
    return [(sel.itemText(i), sel.itemData(i)) for i in range(sel.count())]


def test_the_selector_always_offers_manual(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    labels = _selector_labels(panel)
    assert ("Manual — no events yet", MANUAL_MODULE_ID) in labels


def test_choosing_manual_creates_it_empty_and_shows_it(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(MANUAL_MODULE_ID))
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    assert res is not None and len(res.payload.df) == 0
    assert panel.current_detection().module_id == MANUAL_MODULE_ID
    # Ctrl+M now files into Manual even though Schmitt has rows
    out = panel.add_manual_event(doc_id, 0, *_MISSED)
    assert out.ok and out.module_id == MANUAL_MODULE_ID and not out.created_table
    assert len(_table(win, doc_id, "detect.schmitt")) == 2
    assert [f for f in _selector_labels(panel) if f[1] == MANUAL_MODULE_ID][0][0] == "Manual — 1 event"


def test_a_detector_finishing_does_not_move_a_user_working_in_manual(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    messages = []
    panel.statusMessage.connect(messages.append)
    hole = _signal(_DETECTED, seed=1)
    df, prov = detect_events(hole, FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    assert panel.current_detection().module_id == MANUAL_MODULE_ID
    assert any("Schmitt finished" in m for m in messages)
    assert any(code == "detect.schmitt" for _label, code in _selector_labels(panel))


def test_the_target_is_remembered_per_channel_and_names_the_table(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.target_module_id(doc_id, 0) == "detect.schmitt"
    assert panel.target_table_name(doc_id, 0) == "Schmitt"
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(MANUAL_MODULE_ID))
    assert panel.target_module_id(doc_id, 0) == MANUAL_MODULE_ID
    assert panel.target_table_name(doc_id, 0) == "Manual"


def test_reconciled_refuses_a_hand_add(win, tmp_path):
    from nanoscope.analysis.reconcile import CONSENSUS_MODULE_ID

    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    base = win._session.results.get(doc_id, 0, "detect.schmitt")
    win._session.add_result(AnalysisResult(
        module_id=CONSENSUS_MODULE_ID, module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=base.calibration_fingerprint,
        payload=EventSet(base.payload.df.copy(), dict(base.payload.provenance))))
    assert panel.target_module_id(doc_id, 0) == CONSENSUS_MODULE_ID
    out = panel.add_manual_event(doc_id, 0, *_MISSED)
    assert not out.ok and "Reconciled is derived" in out.message


# ================================================================================================
# the window: Ctrl+M is the door, and the curation windows land on the new event
# ================================================================================================
from ._inspector_hosting import (  # noqa: E402
    FS as HFS, build_window, close_window, private_settings, settle)


def _hframe(window):
    return window._analysis_panel.current_detection().payload.df


def _hbracket(window, start: int, end: int) -> None:
    window._analysis_panel.measure_panel.set_active(True)
    window._trace_view.set_measure_region(start / HFS, end / HFS)
    settle()


def _hx_range(window):
    return [round(float(v), 6) for v in window._trace_view.anchor_plot().vb.viewRange()[0]]


def test_ctrl_m_files_an_event_in_the_shown_table_and_says_where(private_settings, qtbot):
    window = build_window(qtbot)
    n = len(_hframe(window))
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    frame = _hframe(window)
    assert len(frame) == n + 1
    assert "Added event" in window.statusBar().currentMessage()
    assert "Schmitt" in window.statusBar().currentMessage()
    # and no row went to the old measurement list
    assert window._analysis_panel.measure_panel.measurements_df is None \
        or len(window._analysis_panel.measure_panel.measurements_df) == 0
    close_window(window)


def test_ctrl_m_with_measure_off_says_how_to_turn_it_on(private_settings, qtbot):
    window = build_window(qtbot)
    window._act_add_event.trigger()
    assert "Measure is off" in window.statusBar().currentMessage()
    assert len(_hframe(window)) == 6
    close_window(window)


def test_the_curation_windows_keep_their_place_and_land_on_the_new_event(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_gallery()
    window._open_event_inspector()
    settle()
    gal, insp = window._event_gallery, window._event_inspector
    window._trace_view.set_x_range(0.40, 0.50)
    settle()
    before = _hx_range(window)
    announced: list[int] = []
    gal.eventSelected.connect(announced.append)
    insp.eventSelected.connect(announced.append)
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    frame = _hframe(window)
    new_id = int(frame.loc[frame["source"] == "manual", "event_id"].iloc[0])
    assert gal._current_event_id() == new_id
    assert insp._current_id == new_id
    assert announced == [] and _hx_range(window) == before        # the view is the curator's
    # one undo takes it back out, still without moving anything
    window._act_undo.trigger()
    settle()
    assert len(_hframe(window)) == len(frame) - 1
    assert announced == [] and _hx_range(window) == before
    close_window(window)


def test_the_gallery_filter_survives_an_add(private_settings, qtbot):
    from nanoscope.analysis.curation_view import FILTER_ACCEPTED

    window = build_window(qtbot)
    window._open_event_gallery()
    settle()
    gal = window._event_gallery
    ix = gal.filter_options().index(FILTER_ACCEPTED)
    gal._filter_combo.setCurrentIndex(ix) if hasattr(gal, "_filter_combo") else None
    chosen = gal.current_filter() if hasattr(gal, "current_filter") else None
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    if chosen is not None:
        assert gal.current_filter() == chosen
    close_window(window)


# ================================================================================================
# undo labels, Delete, Rebuild and the Manual table's staleness
# ================================================================================================
def _undo_text(win) -> str:
    return win._act_undo.text()


def test_the_undo_menu_names_a_hand_add_and_a_delete(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    assert "Add event" in _undo_text(win)
    out = panel.add_manual_event(doc_id, 0, *_DETECTED[0])
    assert out.ok and "Add event" in _undo_text(win)
    assert panel.delete_hand_made_events([out.event_id]) == [out.event_id]
    assert "Delete event" in _undo_text(win)
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 1
    win._act_undo.trigger()
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 2


def test_only_hand_made_events_can_be_deleted(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    refusals = []
    panel.runRefused.connect(refusals.append)
    first = int(_table(win, doc_id, "detect.schmitt")["event_id"].iloc[0])
    assert panel.delete_hand_made_events([first]) == []
    assert refusals and "Only hand-made events can be deleted" in refusals[-1]
    assert len(_table(win, doc_id, "detect.schmitt")) == 2
    out = panel.add_manual_event(doc_id, 0, *_MISSED)
    assert panel.delete_hand_made_events([out.event_id]) == [out.event_id]
    assert len(_table(win, doc_id, "detect.schmitt")) == 2
    # the span is free again
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok


def test_a_manual_table_goes_stale_with_the_rate_and_rebuild_repairs_it(win, qtbot, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    before = _table(win, doc_id, MANUAL_MODULE_ID).iloc[0]
    # the user declares the recording was really sampled at 100 kHz
    from nanoscope.core.sample_rate import SampleRateDeclaration
    win._session.set_sample_rate_declaration(
        doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=2 * FS))
    assert panel._stale.kind == "sample_rate" and panel._stale.action_text() == "Rebuild"
    # adding is refused until it is rebuilt, and the sentence says Rebuild
    out = panel.add_manual_event(doc_id, 0, *_DETECTED[0])
    assert not out.ok and "Rebuild the Manual table" in out.message
    # Re-run is never offered for it
    seen = []
    panel.runRefused.connect(seen.append)
    panel._on_rerun()
    assert seen and "never re-run" in seen[-1]
    # rebuild: off-thread, one undo step, times move onto the new clock
    with qtbot.waitSignal(win._session.resultsChanged, timeout=30_000):
        assert panel.rebuild_manual_table() == "started"
    after = _table(win, doc_id, MANUAL_MODULE_ID).iloc[0]
    assert after["dwell_s"] == pytest.approx(before["dwell_s"] / 2, rel=1e-6)
    assert int(after["start_idx"]) == int(before["start_idx"])
    assert panel._stale.isHidden()
    assert panel.add_manual_event(doc_id, 0, *_DETECTED[0]).ok


# ================================================================================================
# a re-run keeps what people drew
# ================================================================================================
def _fresh_run(win, doc_id, spans=_DETECTED, seed=2):
    sig = _signal(spans, seed=seed)
    df, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4)
    return AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov))


def test_a_rerun_keeps_the_hand_made_events_and_stays_one_rerun_step(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    messages = []
    panel.statusMessage.connect(messages.append)

    report = panel.install_detection_result(_fresh_run(win, doc_id))

    assert report is not None and len(report.kept) == 1
    df = _table(win, doc_id, "detect.schmitt")
    assert len(df) == 3 and hand_made_mask(df).sum() == 1
    hand = df.loc[hand_made_mask(df)].iloc[0]
    assert (int(hand["start_idx"]), int(hand["end_idx"])) == _MISSED
    assert hand["origin"].startswith("detect.schmitt:")
    assert any("Kept 1 hand-made event" in m for m in messages)
    assert "Re-run detection" in _undo_text(win)
    win._act_undo.trigger()
    assert len(_table(win, doc_id, "detect.schmitt")) == 3         # the previous table is back
    assert hand_made_mask(_table(win, doc_id, "detect.schmitt")).sum() == 1


def test_the_replace_prompt_does_not_count_hand_made_events_as_at_stake(win, tmp_path):
    from nanoscope.gui.analysis_panel import AnalysisPanel

    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    work = AnalysisPanel.hand_work_at_stake(_table(win, doc_id, "detect.schmitt"))
    assert not any(work.values())                  # nothing a re-run destroys: no prompt
    first = int(_table(win, doc_id, "detect.schmitt")["event_id"].iloc[0])
    panel.set_event_accepted(first, False)         # a rejected DETECTED row still is
    work = AnalysisPanel.hand_work_at_stake(_table(win, doc_id, "detect.schmitt"))
    assert work["n_hand_rejected"] == 1 and work["n_added"] == 0


def test_a_rerun_with_no_hand_made_events_is_exactly_what_it_was(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.install_detection_result(_fresh_run(win, doc_id)) is None
    assert panel.last_carry_report is None
    assert len(_table(win, doc_id, "detect.schmitt")) == 2


# ================================================================================================
# a project keeps the Manual table and the table each channel files into
# ================================================================================================
def test_the_manual_table_and_the_chosen_table_survive_save_and_reopen(qapp, qtbot, tmp_path):
    from nanoscope.core.project import load_project, save_project

    sig = _signal(_DETECTED + [_MISSED])
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = win._session.add_document(_doc(tmp_path, sig))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED, note="a slow one").ok
    assert panel.add_manual_event(doc_id, 0, *_DETECTED[0]).ok
    assert panel.target_module_id(doc_id, 0) == MANUAL_MODULE_ID
    before = _table(win, doc_id, MANUAL_MODULE_ID).copy()

    path = tmp_path / "p.nanoscope"
    save_project(path, win._gather_project_state())
    state = load_project(path)
    assert any(r.module_id == MANUAL_MODULE_ID for r in state.results)
    assert any(isinstance(v, list) and v and v[0]["module_id"] == MANUAL_MODULE_ID
               for k, v in state.workspace.extra.items() if k == "event_table_choices_v1")

    win2 = MainWindow()
    qtbot.addWidget(win2)
    live_id = win2._session.add_document(_doc(tmp_path, sig))
    win2._create_default_layers(win2._session.document(live_id))
    win2._restore_state = state
    win2._apply_restore_for_doc(live_id)
    win2._on_selection(live_id, 0)

    res = win2._session.results.get(live_id, 0, MANUAL_MODULE_ID)
    assert res is not None
    after = res.payload.df
    assert list(after["start_idx"]) == list(before["start_idx"])
    assert after["note"].tolist()[after["start_idx"].tolist().index(_MISSED[0])] == "a slow one"
    assert (after["source"] == "manual").all() and after["added"].all()
    assert res.baseline_fingerprint == _table_res(win, doc_id).baseline_fingerprint
    # and where the next Ctrl+M goes is still the Manual table, not whichever table came first
    assert win2._analysis_panel.target_module_id(live_id, 0) == MANUAL_MODULE_ID
    assert win2._analysis_panel.current_detection().module_id == MANUAL_MODULE_ID


def _table_res(win, doc_id):
    return win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)


# ================================================================================================
# Delete hand-made event, from the three surfaces
# ================================================================================================
def test_the_inspector_shows_delete_only_for_a_hand_made_event(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_inspector()
    settle()
    insp = window._event_inspector
    insp.show_event_quietly(int(_hframe(window)["event_id"].iloc[0]))
    assert insp._delete_btn.isHidden()                          # a detected event
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    new_id = int(_hframe(window).loc[_hframe(window)["source"] == "manual", "event_id"].iloc[0])
    assert insp._current_id == new_id and not insp._delete_btn.isHidden()
    n = len(_hframe(window))
    insp._delete_btn.click()
    settle()
    assert len(_hframe(window)) == n - 1
    assert insp._delete_btn.isHidden()                           # it landed on a detected one
    assert "Delete event" in window._act_undo.text()
    close_window(window)


def test_the_results_table_menu_offers_delete_for_hand_made_rows_only(private_settings, qtbot,
                                                                        monkeypatch):
    from nanoscope.gui.panels import EventTable

    window = build_window(qtbot)
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    table = window._analysis_panel._table
    df = _hframe(window)
    offered: list[list[str]] = []

    def fake_exec(menu, _pos):
        offered.append([a.text().replace("&", "") for a in menu.actions() if a.text()])
        return next((a for a in menu.actions() if "Delete" in a.text()), None)

    monkeypatch.setattr(EventTable, "_exec_menu", staticmethod(fake_exec))
    hand_row = int(df.index[df["source"] == "manual"][0])
    detected_row = int(df.index[df["source"] != "manual"][0])
    for row, expect in ((detected_row, False), (hand_row, True)):
        proxy = table._proxy.mapFromSource(table._model.index(row, 0))
        table.view.selectRow(proxy.row())
        rect = table.view.visualRect(proxy)
        table._on_context_menu(rect.center())
        assert any("Delete hand-made" in t for t in offered[-1]) is expect
    assert len(_hframe(window)) == len(df) - 1        # the second menu chose Delete
    close_window(window)


# ================================================================================================
# Export event statistics
# ================================================================================================
def test_the_statistics_export_covers_hand_made_and_detected_events(private_settings, qtbot,
                                                                      tmp_path):
    from nanoscope.analysis.measure import MEASUREMENT_SCHEMA

    window = build_window(qtbot)
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    assert window._act_export_event_statistics.isEnabled()
    out = window._analysis_panel.export_event_statistics(tmp_path / "stats.csv", scope="shown")
    assert out is not None and out.exists()
    frame = pd.read_csv(out)
    assert list(frame.columns)[:len(MEASUREMENT_SCHEMA)] == list(MEASUREMENT_SCHEMA)
    assert list(frame.columns)[len(MEASUREMENT_SCHEMA):] == ["signal_basis", "table", "event_id"]
    assert len(frame) == len(_hframe(window)) == 7
    hand = frame.loc[frame["start_idx"] == 21_000].iloc[0]
    assert hand["table"] == "Schmitt"
    detected = frame.loc[frame["start_idx"] == 5_000].iloc[0]
    assert detected["amplitude"] == pytest.approx(300.0, rel=0.1)       # a real blockade
    assert abs(hand["amplitude"]) < 30.0                              # the bracket was baseline
    # the table's own baseline and polarity, not a second estimate off a local flank
    row = _hframe(window).loc[_hframe(window)["start_idx"] == 21_000].iloc[0]
    assert hand["baseline"] == pytest.approx(row["baseline_pA"])
    for col in ("mean", "median", "std", "var", "min", "max", "ptp", "rms", "skewness",
                "kurtosis", "area_ecd", "snr"):
        assert np.isfinite(hand[col]), col
    close_window(window)


def test_the_statistics_of_every_manual_table_land_in_one_file(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED, note="slow").ok
    assert panel.add_manual_event(doc_id, 0, *_DETECTED[0]).ok
    out = panel.export_event_statistics(tmp_path / "manual.csv", scope="manual")
    frame = pd.read_csv(out)
    assert len(frame) == 2 and set(frame["table"]) == {"Manual"}
    assert set(frame["signal_basis"]) == {"raw"}
    assert "slow" in set(frame["label"].fillna(""))


def test_a_table_on_another_sample_rate_is_left_out_and_said(win, tmp_path):
    from nanoscope.core.sample_rate import SampleRateDeclaration

    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    win._session.set_sample_rate_declaration(
        doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=2 * FS))
    refused = []
    panel.runRefused.connect(refused.append)
    assert panel.export_event_statistics(tmp_path / "s.csv", scope="shown") is None
    assert refused and "another sample rate" in refused[-1]
    assert not (tmp_path / "s.csv").exists()


# ================================================================================================
# the source filter, the M mark and the inspector's source line
# ================================================================================================
def test_the_source_filters_appear_only_once_a_table_holds_a_hand_made_event(
        private_settings, qtbot):
    from nanoscope.analysis.curation_view import FILTER_DETECTED, FILTER_HAND_MADE

    window = build_window(qtbot)
    window._open_event_gallery()
    window._open_event_inspector()
    settle()
    gal, insp = window._event_gallery, window._event_inspector
    assert FILTER_HAND_MADE not in gal.filter_options()
    assert FILTER_HAND_MADE not in insp.filter_options()
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    for opts in (gal.filter_options(), insp.filter_options()):
        assert FILTER_HAND_MADE in opts and FILTER_DETECTED in opts
    gal.set_filter(FILTER_HAND_MADE)
    settle()
    assert gal._model.rowCount() == 1
    gal.set_filter(FILTER_DETECTED)
    settle()
    assert gal._model.rowCount() == 6
    close_window(window)


def test_a_hand_made_event_is_marked_in_the_gallery_and_described_in_the_inspector(
        private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_gallery()
    window._open_event_inspector()
    settle()
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    frame = _hframe(window)
    gal, insp = window._event_gallery, window._event_inspector
    hand = int(frame.index[frame["source"] == "manual"][0])
    detected = int(frame.index[frame["source"] != "manual"][0])
    assert gal.is_hand_made(hand) and not gal.is_hand_made(detected)
    insp.show_event_quietly(int(frame["event_id"].iloc[hand]))
    text = insp._readout.text()
    assert "source:" in text and "drawn by hand" in text and "bounds start=drawn" in text
    insp.show_event_quietly(int(frame["event_id"].iloc[detected]))
    assert "drawn by hand" not in insp._readout.text()
    close_window(window)


def test_the_include_switch_governs_the_events_export(private_settings, qtbot, tmp_path,
                                                        monkeypatch):
    import json
    from nanoscope.gui import main_window as mw

    window = build_window(qtbot)
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    assert window._act_include_hand_made.isChecked() and window._act_include_hand_made.isEnabled()
    target = tmp_path / "events.csv"
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: target)

    window._export_events()
    assert len(pd.read_csv(target)) == 7
    window._act_include_hand_made.setChecked(False)
    target2 = tmp_path / "events2.csv"
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: target2)
    window._export_events()
    assert len(pd.read_csv(target2)) == 6
    meta = json.loads((tmp_path / "events2.meta.json").read_text())["curation"]
    assert meta["include_added"] is False and meta["n_added_excluded"] == 1
    assert "left out" in window.statusBar().currentMessage()
    # it is a preference of the app, not of one table: and it greys out where it means nothing
    window._act_include_hand_made.setChecked(True)
    close_window(window)


# ================================================================================================
# Copy hand-made events to another table
# ================================================================================================
def test_copy_puts_hand_made_events_in_the_manual_table_once(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    a = panel.add_manual_event(doc_id, 0, *_MISSED, note="slow one")
    assert a.ok and panel.copy_targets() == [MANUAL_MODULE_ID]

    report = panel.copy_hand_made_events(MANUAL_MODULE_ID)

    assert report is not None and len(report.copied) == 1
    manual = _table(win, doc_id, MANUAL_MODULE_ID)
    assert len(manual) == 1 and manual["origin"].iloc[0] == f"detect.schmitt:{a.event_id}"
    assert manual["note"].iloc[0] == "slow one" and manual["source"].iloc[0] == "manual"
    assert len(_table(win, doc_id, "detect.schmitt")) == 3         # the source is untouched
    again = panel.copy_hand_made_events(MANUAL_MODULE_ID)
    assert again.copied == () and again.already_copied == (a.event_id,)
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 1          # nothing copied twice


def test_copy_is_one_undo_step_on_the_target_and_measures_under_its_provenance(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    messages = []
    panel.statusMessage.connect(messages.append)
    panel.copy_hand_made_events(MANUAL_MODULE_ID)
    assert any(m.startswith("Copied 1 hand-made event from Schmitt to Manual") for m in messages)
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(MANUAL_MODULE_ID))
    assert "Add event" in _undo_text(win)                       # the target's own history
    win._act_undo.trigger()
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 0
    assert len(_table(win, doc_id, "detect.schmitt")) == 3


def test_copy_refuses_when_there_is_nothing_to_copy_or_nowhere_to_put_it(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    refused = []
    panel.runRefused.connect(refused.append)
    assert panel.copy_hand_made_events(MANUAL_MODULE_ID) is None
    assert refused and "holds no hand-made events" in refused[-1]
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    assert panel.copy_hand_made_events("detect.schmitt") is None
    assert "Choose another table" in refused[-1]


def test_copy_skips_a_span_the_target_already_holds(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(MANUAL_MODULE_ID))
    assert panel.add_manual_event(doc_id, 0, 20_050, 20_250).ok        # the same event, drawn again
    sel.setCurrentIndex(sel.findData("detect.schmitt"))
    report = panel.copy_hand_made_events(MANUAL_MODULE_ID)
    assert report.copied == () and len(report.overlapping) == 1
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 1


def test_the_copy_menu_item_follows_the_table_on_screen(private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    assert not window._act_copy_hand_made.isEnabled()                 # nothing drawn yet
    _hbracket(window, 21_000, 21_400)
    window._act_add_event.trigger()
    settle()
    assert window._act_copy_hand_made.isEnabled()
    monkeypatch.setattr(window, "_choose_copy_target", lambda: "detect.manual")
    window._act_copy_hand_made.trigger()
    settle()
    assert window._session.results.get(
        window._analysis_panel.current_target()[0], 0, "detect.manual") is not None
    close_window(window)


# ================================================================================================
# the panel's edge fit and catch (the analysis behind the cursor keys)
# ================================================================================================
def test_the_panel_fits_a_roughly_placed_span_onto_the_step(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    fit, why = panel.fit_span_edges(doc_id, 0, 10_030, 10_270, px_samples=10.0)
    assert why == "" and fit is not None and fit.complete
    assert abs(fit.start.position - 10_000) <= 3 and abs(fit.end.position - 10_300) <= 3
    assert '"method":"step"' in fit.to_json()


def test_a_fit_far_from_any_step_is_left_unfitted_and_says_why(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    fit, _ = win._analysis_panel.fit_span_edges(doc_id, 0, 40_000, 40_300, px_samples=10.0)
    assert fit is not None and not fit.start.fitted and not fit.end.fitted
    assert (fit.start.position, fit.end.position) == (40_000, 40_300)
    assert fit.start.reason


def test_catch_finds_the_event_under_the_pointer_and_fits_both_edges(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    catch, why = win._analysis_panel.catch_event_at(
        doc_id, 0, 20_150.0, px_samples=10.0, view_lo=18_000, view_hi=23_000)
    assert why == "" and catch.ok, catch.reason
    assert abs(catch.fit.start.position - 20_000) <= 3 and abs(catch.fit.end.position - 20_300) <= 3
    nothing, _ = win._analysis_panel.catch_event_at(
        doc_id, 0, 45_000.0, px_samples=10.0, view_lo=43_000, view_hi=48_000)
    assert not nothing.ok and "No event under the pointer" in nothing.reason


def test_catch_refuses_a_pointer_on_an_event_the_table_already_holds(win, tmp_path):
    doc_id = _open(win, tmp_path)
    catch, _ = win._analysis_panel.catch_event_at(
        doc_id, 0, 10_150.0, px_samples=10.0, view_lo=9_000, view_hi=12_000)
    assert not catch.ok and "already in the table" in catch.reason


def test_the_manual_contract_carries_the_open_pore_tau(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    contract = win._analysis_panel._manual_contract(doc_id, 0)
    assert contract.open_pore_tau is not None and contract.open_pore_tau >= 1.0
    assert contract.provenance["open_pore_tau"] == contract.open_pore_tau


def test_the_fit_has_a_sentence_when_the_table_cannot_take_one(win, tmp_path):
    doc_id = _open(win, tmp_path)
    fit, why = win._analysis_panel.fit_span_edges("some-other-doc", 0, 10, 20)
    assert fit is None and "not on the channel" in why


def test_the_density_figure_draws_one_table_per_recording_and_channel(private_settings, qtbot):
    window = build_window(qtbot)
    panel = window._analysis_panel
    doc_id = panel.current_target()[0]
    assert len(window._density_groups()) == 1
    # a Manual table beside Schmitt must not become a second, colliding population
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(MANUAL_MODULE_ID))
    for i in range(6):
        start = 1_000 + i * 4_000
        window._analysis_panel.add_manual_event(doc_id, 0, start, start + 300)
    assert len(window._density_groups()) == 1
    sel.setCurrentIndex(sel.findData("detect.schmitt"))
    assert len(window._density_groups()) == 1
    close_window(window)


# ================================================================================================
# converting the old measurement list
# ================================================================================================
def _seed_measurements(win, doc_id, spans_labels):
    """Put rows in the legacy list the way a project restore does: ``set_measurements_df``."""
    from nanoscope.analysis.measure import coerce_measurements
    from nanoscope.core.segment import persistent_key

    key = persistent_key(win._session.document(doc_id))
    recs = [{"measurement_id": i, "recording": "door.dat", "doc_key": key, "channel": "current",
             "channel_index": 0, "start_idx": s, "end_idx": e, "label": label, "units": "pA",
             "cal_gain": 1.0, "cal_offset": 0.0}
            for i, (s, e, label) in enumerate(spans_labels)]
    win._analysis_panel.measure_panel.set_measurements_df(coerce_measurements(pd.DataFrame(recs)))


def test_old_measurements_become_events_in_one_undoable_step(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    _seed_measurements(win, doc_id, [(*_MISSED, "slow one"), (*_DETECTED[0], ""),
                                     (500, 500, "empty")])
    plan = panel.conversion_plan()
    assert [r.placeable for r in plan] == [True, True, False]

    report = panel.convert_measurements([0, 1, 2])

    assert sorted(m for m, _e in report.converted) == [0, 1] and report.skipped[0][0] == 2
    manual = _table(win, doc_id, MANUAL_MODULE_ID)
    assert len(manual) == 2 and set(manual["legacy_measurement_ids"]) == {"0", "1"}
    row = manual.loc[manual["start_idx"] == _MISSED[0]].iloc[0]
    assert row["note"] == "slow one" and row["source"] == "manual" and bool(row["accepted"])
    assert panel.current_detection().module_id == MANUAL_MODULE_ID
    assert len(panel._measure.measurements_df) == 3                   # the list itself is untouched
    assert "Add 2 events" in _undo_text(win)
    # offered once: the converted rows say so
    assert [r.already_converted for r in panel.conversion_plan()] == [True, True, False]
    win._act_undo.trigger()
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 0
    assert panel.converted_measurement_ids() == set()


def test_conversion_leaves_out_a_row_that_overlaps_an_event_already_there(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    _seed_measurements(win, doc_id, [(10_050, 10_250, "over a detection"), (*_MISSED, "free")])
    # the Manual table is its own table: a detected event in Schmitt is not in the way...
    report = panel.convert_measurements([0, 1])
    assert len(report.converted) == 2
    # ...but converting the same span twice is: the second pass finds the first conversion
    again = panel.convert_measurements([0, 1])
    assert again.converted == () and again.skipped


def test_the_convert_menu_item_follows_the_measurement_list(private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    assert not window._act_convert_measurements.isEnabled()
    doc_id = window._analysis_panel.current_target()[0]
    _seed_measurements(window, doc_id, [(21_000, 21_400, "x")])
    window._sync_export_action()
    assert window._act_convert_measurements.isEnabled()
    monkeypatch.setattr(window, "_choose_measurements_to_convert", lambda plan: [0])
    window._act_convert_measurements.trigger()
    settle()
    assert window._session.results.get(doc_id, 0, MANUAL_MODULE_ID) is not None
    close_window(window)


def test_the_conversion_dialog_ticks_what_can_be_placed(qapp):
    from nanoscope.analysis.measurement_conversion import ConversionRow
    from nanoscope.gui.convert_measurements_dialog import ConvertMeasurementsDialog

    dialog = ConvertMeasurementsDialog([
        ConversionRow(0, "r", "current", 10, 20, "a", True, ""),
        ConversionRow(1, "r", "current", 30, 40, "b", False, "that recording is not open"),
        ConversionRow(2, "r", "current", 50, 60, "c", True, "", numbers_change=True)])
    assert dialog.selected_ids() == [0, 2]
    dialog.table.item(0, 0).setCheckState(QtCore.Qt.Unchecked)
    assert dialog.selected_ids() == [2]
    assert "numbers will change" in dialog.table.item(2, 6).text()
    assert "not open" in dialog.table.item(1, 6).text()


def test_a_converted_measurement_is_not_offered_again_as_a_plot_source(win, tmp_path):
    from nanoscope.core.sources import MANUAL_SOURCE_KEY

    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    _seed_measurements(win, doc_id, [(*_MISSED, "a"), (*_DETECTED[0], "b")])
    assert any(s.key == MANUAL_SOURCE_KEY for s in win._plots_sources())
    panel.convert_measurements([0, 1])
    assert not any(s.key == MANUAL_SOURCE_KEY for s in win._plots_sources())
    keys = {s.key for s in win._plots_sources()}
    assert any(k.endswith("@detect.manual") or "manual" in k for k in keys)
