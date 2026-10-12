"""Long annotation sessions need stable, reachable controls beside a usable waveform."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_inspector import EventInspectorWindow


@pytest.fixture(params=[(1000, 700), (1160, 760)], ids=["compact", "default"])
def inspector(request, qapp, qtbot, tmp_path):
    raw = 100 + np.random.default_rng(5).normal(0, 1, 20000)
    raw[4000:4300] -= 30
    raw[11000:11400] -= 25
    raw = np.asarray(raw, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "layout.dat", format_name="synthetic",
                         n_samples=raw.size, sample_rate_hz=100000,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(raw,)))
    frame, provenance = detect_events(raw, 100000, min_dwell_s=.0005)
    assert len(frame) == 2
    win = EventInspectorWindow(session)
    qtbot.addWidget(win)
    win.set_events(doc_id, 0, frame, provenance=provenance)
    win.resize(*request.param)
    win.show()
    qtbot.wait(60)
    assert win.size().toTuple() == request.param
    return win


def _rect(widget, window):
    return QtCore.QRect(widget.mapTo(window, QtCore.QPoint()), widget.size())


def _reachable(widget, window):
    assert widget.isVisible(), widget.objectName() or type(widget).__name__
    assert widget.visibleRegion().contains(widget.rect().adjusted(1, 1, -1, -1))
    assert window.rect().contains(_rect(widget, window))


def _work_geometry(win):
    return tuple(_rect(widget, win) for widget in (
        win._plot_widget, win._prev_btn, win._index_spin, win._next_btn,
        win._accept_btn, win._reject_btn, win._accept_check))


def test_review_actions_are_reachable_and_clustered_below_a_usable_plot(inspector):
    win = inspector
    plot = _rect(win._plot_widget, win)
    assert plot.height() >= 450
    assert plot.width() >= 480
    for widget in (win._prev_btn, win._index_spin, win._next_btn,
                   win._accept_btn, win._reject_btn, win._accept_check,
                   win._context_combo, win._filter_combo, win._gallery_btn,
                   win._trace_btn, win._project_save_button):
        _reachable(widget, win)
    actions = [_rect(widget, win) for widget in (
        win._prev_btn, win._next_btn, win._accept_btn, win._reject_btn)]
    assert len({rect.top() for rect in actions}) == 1
    assert all(rect.top() > plot.bottom() for rect in actions)
    assert 0 <= actions[2].left() - actions[1].right() <= 32
    assert 0 <= actions[3].left() - actions[2].right() <= 16
    assert not win._review_sidebar.isAncestorOf(win._accept_btn)
    assert win._review_sidebar.isAncestorOf(win._accept_all_btn)
    assert win._review_sidebar.isAncestorOf(win._reject_all_btn)


@pytest.mark.parametrize("method", ["selected_median", "manual"])
def test_amplitude_modes_and_sidebar_scrolling_keep_review_controls_fixed(
        inspector, qtbot, method):
    win = inspector
    before = win._df.copy(deep=True)
    geometry = _work_geometry(win)
    controls = win._amplitude
    controls.method.setCurrentIndex(controls.method.findData(method))
    qtbot.wait(50)
    relevant = (controls.method, controls.apply_button, controls.reset_button,
                controls.start, controls.end, controls.state) if method == "selected_median" else (
                    controls.method, controls.apply_button, controls.level, controls.edit_level)
    for widget in (*relevant, win._revert_btn, win._merge_next_btn,
                   win._accept_all_btn, win._reject_all_btn):
        win._review_sidebar.ensureWidgetVisible(widget)
        qtbot.wait(15)
        _reachable(widget, win)
        assert _work_geometry(win) == geometry
    assert win._review_sidebar.horizontalScrollBar().maximum() == 0
    for widget in (win._prev_btn, win._next_btn, win._accept_btn, win._reject_btn):
        _reachable(widget, win)
    pd.testing.assert_frame_equal(win._df, before, check_exact=True)


def test_complete_warning_remains_readable_without_moving_waveform_or_footer(inspector, qtbot):
    win = inspector
    geometry = _work_geometry(win)
    before = win._df.copy(deep=True)
    warning = (
        "These events were measured with the previous open-pore estimator. Depth, "
        "fractional depth, ECD, mean/max deviation and SNR are stale. Long events can "
        "change by more than tenfold when the reference window is too short. Nothing "
        "has been re-measured: compare baseline estimators on this recording before "
        "using these measurements, then re-measure the event depths with the intended "
        "reference and inspect the resulting waveform and summary.")
    win.set_stale_columns(REMEASURED_COLUMNS, warning)
    qtbot.wait(50)
    label = win._stale_label
    assert label.text() == "⚠ " + warning
    assert label.wordWrap() and not label.toolTip()
    assert label.height() >= label.heightForWidth(label.width())
    win._review_sidebar.ensureWidgetVisible(label)
    qtbot.wait(30)
    _reachable(label, win)
    assert _work_geometry(win) == geometry
    win.set_stale_columns(())
    qtbot.wait(30)
    assert label.isHidden()
    assert _work_geometry(win) == geometry
    pd.testing.assert_frame_equal(win._df, before, check_exact=True)
