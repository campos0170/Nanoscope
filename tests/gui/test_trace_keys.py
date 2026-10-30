"""The trace keys: catch, place an edge, add, step, discard, draw mode (docs/CURSOR_ERGONOMICS_DESIGN.md).

Driven through a real MainWindow. The pointer is stood in for by a patched
`TraceView.pointer_sample` (a real pointer is a screen coordinate), everything else is the
shipped path: the controller's slots, the panel's fit and catch, the add door.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
EVENT = (20_000, 20_400)            # a clean blockade the detector was never asked about
SECOND = (60_000, 60_250)


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in (EVENT, SECOND):
        x[s:e] += DEPTH
    return x.astype(np.float32)


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "keys.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return win, doc_id


def _point(win, doc_id, sample, px=10.0):
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _table(win, doc_id):
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return None if res is None else res.payload.df


def test_the_keys_are_trace_scoped_shortcuts_with_one_exception_each(rig):
    win, _ = rig
    seqs = {s.key().toString() for s in win._trace_keys.shortcuts}
    assert {"C", "S", "F", "D", "G", "X", "E", "Shift+S", "Shift+F", "Shift+D", "Shift+G",
            "Return", "Enter", "Shift+Return", "Shift+Enter", "Esc"} <= seqs
    for s in win._trace_keys.shortcuts:
        assert s.parent() is win._trace_view
        assert s.context() == QtCore.Qt.WidgetWithChildrenShortcut
        assert s.autoRepeat() == (s.key().toString() in ("G", "Shift+G"))


def test_catch_then_add_files_a_fitted_event_and_leaves_measure_on_with_no_mask(rig):
    win, doc_id = rig
    tv = win._trace_view
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    assert tv.measure_bounds() is not None
    lo, hi = tv.measure_bounds()
    assert abs(lo - EVENT[0]) <= 3 and abs(hi - EVENT[1]) <= 3
    view_before = tv.anchor_plot().vb.viewRange()[0]

    win._trace_keys.add()

    df = _table(win, doc_id)
    assert df is not None and len(df) == 1
    row = df.iloc[0]
    assert abs(int(row["start_idx"]) - EVENT[0]) <= 3 and abs(int(row["end_idx"]) - EVENT[1]) <= 3
    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    assert json.loads(row["fit_json"])["method"] == "step"
    assert int(row["drawn_start_idx"]) > 0 and row["source"] == "manual"
    assert tv.measure_empty() and tv.measure_bounds() is None       # the next event starts clean
    assert win._analysis_panel.measure_panel.is_active()
    assert list(tv.anchor_plot().vb.viewRange()[0]) == pytest.approx(list(view_before))
    assert "Added event" in win.statusBar().currentMessage()


def test_catch_refuses_on_baseline_and_says_so(rig):
    win, doc_id = rig
    _point(win, doc_id, 90_000)
    win._trace_keys.catch()
    assert "No event under the pointer" in win.statusBar().currentMessage()
    assert win._trace_view.measure_bounds() is None


def test_s_then_f_builds_a_mask_and_fits_each_edge(rig):
    win, doc_id = rig
    tv = win._trace_view
    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)
    assert "Point at the finish" in win.statusBar().currentMessage()
    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)
    lo, hi = tv.measure_bounds()
    assert abs(lo - EVENT[0]) <= 3 and abs(hi - EVENT[1]) <= 3
    win._trace_keys.add()
    row = _table(win, doc_id).iloc[0]
    assert row["bounds_source"] == "start=fit:step;end=fit:step"


def test_shift_s_and_shift_f_set_an_edge_exactly_and_record_it_as_drawn(rig):
    win, doc_id = rig
    _point(win, doc_id, 30_000)
    win._trace_keys.place_edge("start", fit=False)
    _point(win, doc_id, 30_333)
    win._trace_keys.place_edge("end", fit=False)
    assert win._trace_view.measure_bounds() == (30_000, 30_333)
    win._trace_keys.add()
    row = _table(win, doc_id).iloc[0]
    assert (int(row["start_idx"]), int(row["end_idx"])) == (30_000, 30_333)
    assert row["bounds_source"] == "start=drawn;end=drawn"
    assert pd_isna(row["fit_json"])


def pd_isna(v) -> bool:
    import pandas as pd
    return bool(pd.isna(v))


def test_an_edge_that_cannot_be_fitted_stays_where_pointed_and_d_asks_twice(rig):
    win, doc_id = rig
    _point(win, doc_id, 90_000)
    win._trace_keys.place_edge("start", fit=True)         # nothing to fit on baseline
    assert "could not be fitted" in win.statusBar().currentMessage()
    _point(win, doc_id, 90_300)
    win._trace_keys.place_edge("end", fit=True)
    win._trace_keys.add()
    assert _table(win, doc_id) is None                    # the first D only asked
    assert "Press again" in win.statusBar().currentMessage()   # Ctrl+M and Add ask it too
    win._trace_keys.add()
    row = _table(win, doc_id).iloc[0]
    assert row["bounds_source"] == "start=placed;end=placed"


def test_d_without_a_mask_says_what_to_do(rig):
    win, doc_id = rig
    win._trace_keys.add()
    assert "No mask: press C" in win.statusBar().currentMessage()
    assert _table(win, doc_id) is None


def test_x_discards_the_mask_and_keeps_measure_on(rig):
    win, doc_id = rig
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    win._trace_keys.discard()
    assert win._trace_view.measure_empty()
    assert win._analysis_panel.measure_panel.is_active()


def test_return_adds_only_after_the_user_pointed_at_the_trace(rig):
    win, doc_id = rig
    tv = win._trace_view
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    tv._last_focus_reason = QtCore.Qt.FocusReason.OtherFocusReason     # the window parked focus
    win._trace_keys._alias(win._trace_keys.add)
    assert _table(win, doc_id) is None and tv.measure_bounds() is not None
    tv._last_focus_reason = QtCore.Qt.FocusReason.MouseFocusReason
    win._trace_keys._alias(win._trace_keys.add)
    assert len(_table(win, doc_id)) == 1


def test_esc_discards_but_never_leaves_draw_mode(rig):
    win, doc_id = rig
    tv = win._trace_view
    win._trace_keys.set_draw_mode(True)
    assert tv.draw_mode() and win._act_draw_mode.isChecked()
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    win._trace_keys.escape()
    assert tv.measure_empty() and tv.draw_mode()


def test_e_toggles_draw_mode_and_the_menu_item_follows(rig):
    win, _ = rig
    win._trace_keys.toggle_draw_mode()
    assert win._trace_view.draw_mode() and win._act_draw_mode.isChecked()
    win._act_draw_mode.setChecked(False)
    assert not win._trace_view.draw_mode()
    win._act_draw_mode.setChecked(True)
    assert win._trace_view.draw_mode()
    assert win._analysis_panel.measure_panel.is_active()       # draw mode turned Measure on


def test_a_drawn_span_is_fitted_on_release(rig):
    win, doc_id = rig
    tv = win._trace_view
    win._trace_keys.set_draw_mode(True)
    lo, hi = EVENT[0] + 25, EVENT[1] - 30                      # drawn a little inside the step
    tv.set_measure_region(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, hi))
    tv.sigDrawReleased.emit(doc_id, 0, lo, hi, 10.0)
    b = tv.measure_bounds()
    assert abs(b[0] - EVENT[0]) <= 3 and abs(b[1] - EVENT[1]) <= 3
    win._trace_keys.add()
    row = _table(win, doc_id).iloc[0]
    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    assert int(row["drawn_start_idx"]) == lo and int(row["drawn_end_idx"]) == hi


def test_a_double_click_in_draw_mode_catches_the_event_there(rig):
    win, doc_id = rig
    win._trace_keys.set_draw_mode(True)
    win._trace_view.sigCatchRequested.emit(doc_id, 0, 20_200, 10.0)
    b = win._trace_view.measure_bounds()
    assert b is not None and abs(b[0] - EVENT[0]) <= 3


def test_a_mask_moved_by_hand_after_the_fit_is_recorded_as_drawn(rig):
    win, doc_id = rig
    tv = win._trace_view
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    lo, hi = tv.measure_bounds()
    tv.set_measure_region(tv.x_for_index(doc_id, 0, lo - 50), tv.x_for_index(doc_id, 0, hi))
    win._trace_keys.add()
    row = _table(win, doc_id).iloc[0]
    assert row["bounds_source"].startswith("start=drawn;")
    assert "end=fit:step" in row["bounds_source"]


def test_g_steps_the_window_without_carrying_a_mask_that_is_not_there(rig):
    win, doc_id = rig
    tv = win._trace_view
    tv.set_x_range(0.0, 0.5)
    before = tv.anchor_plot().vb.viewRange()[0]
    win._trace_keys.step_window(+1)
    after = tv.anchor_plot().vb.viewRange()[0]
    assert after[0] > before[0] and (after[1] - after[0]) == pytest.approx(before[1] - before[0])
    assert tv.measure_bounds() is None


def test_shift_d_adds_and_steps_the_window(rig):
    win, doc_id = rig
    tv = win._trace_view
    tv.set_x_range(0.38, 0.42)
    _point(win, doc_id, 20_200, px=1.0)
    win._trace_keys.catch()
    before = tv.anchor_plot().vb.viewRange()[0]
    win._trace_keys.add_and_next()
    after = tv.anchor_plot().vb.viewRange()[0]
    assert len(_table(win, doc_id)) == 1 and after[0] > before[0]


def test_the_label_names_the_target_and_the_counts(rig, qtbot):
    win, doc_id = rig
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    win._trace_keys.add()
    qtbot.wait(200)
    text = win._trace_view.target_label_text()
    assert text.startswith("→ Manual") and "1 hand-made" in text
    assert win._analysis_panel.measure_panel.add_target_text() == "→ Manual"
    win._trace_keys.set_draw_mode(True)
    qtbot.wait(200)
    assert "draw" in win._trace_view.target_label_text()


def test_a_real_key_press_on_a_focused_trace_catches(rig, qtbot):
    win, doc_id = rig
    win.resize(1_200, 800)
    win.show()
    win.activateWindow()
    qtbot.waitUntil(win.isActiveWindow, timeout=5_000)
    tv = win._trace_view
    _point(win, doc_id, 20_200)
    tv.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    assert tv.hasFocus()
    QTest.keyClick(tv, QtCore.Qt.Key_C)
    qtbot.wait(50)
    assert tv.measure_bounds() is not None
    QTest.keyClick(tv, QtCore.Qt.Key_D)
    qtbot.wait(50)
    assert len(_table(win, doc_id)) == 1


def test_a_key_does_nothing_when_the_results_table_has_focus(rig, qtbot):
    win, doc_id = rig
    win.resize(1_200, 800)
    win.show()
    win.activateWindow()
    qtbot.waitUntil(win.isActiveWindow, timeout=5_000)
    _point(win, doc_id, 20_200)
    table = win._analysis_panel._table.view
    table.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    QTest.keyClick(table, QtCore.Qt.Key_C)
    qtbot.wait(30)
    assert win._trace_view.measure_bounds() is None
