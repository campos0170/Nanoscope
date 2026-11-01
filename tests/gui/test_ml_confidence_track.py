"""View > ML confidence track: the ML table's kept probability, drawn under the trace.

Offscreen, synthetic, and with a FAKE model: ``load_keras_model`` is monkeypatched and
``TENSORFLOW_AVAILABLE`` forced on (the pattern of ``test_ml_detection.py``), so no model
file is read and TensorFlow is never imported. What is pinned:

* the pane is opt-in, sits under the trace panes, shares their x axis, spans y 0-1, draws
  the run's enter / exit thresholds, and is built by the four-step peak-decimating curve;
* the time axis moves down to it and back, and toggling it never moves the user's zoom;
* it follows the table on screen (an ML table only) and survives a rebuild;
* an ML table with no kept probability -- the state after reopening a project -- says
  plainly that it is not saved and needs a re-run, through rebuilds and toggles too;
* a Visible-view run lands on the document's own samples, the crosshair reads p there,
  a hover over the strip moves the shared cursor to a sample, and a layer nudge moves it;
* Edit > Undo of a re-run puts the earlier run's track and thresholds back;
* turning it on while the ML table's channel is hidden says so, and the pane comes back
  with the channel.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore

from nanoscope.analysis import ml_detect
from nanoscope.core.confidence import NOT_KEPT_NOTE, ConfidenceTrack
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
FAKE_W = 1024
N = 60_000
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (45_000, 45_400)]


def _event_trace(n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the text reader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


class _FakeEventModel:
    """Fires on the deep, normalised event samples (graded |x|, clipped to 1)."""

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)
        return np.clip(np.abs(x), 0.0, 1.0).astype(np.float16)


@pytest.fixture
def fake_model(monkeypatch):
    wrapper = ml_detect.KerasEventModel(
        model=_FakeEventModel(), window=FAKE_W, path="<fake>", mtime=0.0,
        input_shape=(None, FAKE_W, 1), output_shape=(None, FAKE_W, 1))
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)
    return wrapper


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _run_ml(window: MainWindow, qtbot, *, visible_view: bool = False) -> None:
    panel = window._analysis_panel
    panel._detection.set_model_path("/fake/model.keras")
    # 0 = "Visible view", 1 = "Whole recording".
    panel._detection.ml_range.setCurrentIndex(0 if visible_view else 1)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._run_ml_detection()
    qtbot.wait(20)


def _run_schmitt(window: MainWindow, qtbot) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _show_table(window: MainWindow, module_id: str) -> None:
    sel = window._analysis_panel._result_selector
    idx = sel.findData(module_id)
    assert idx >= 0, module_id
    sel.setCurrentIndex(idx)


def _layout_row(view, item) -> int:
    rows = [r for r, cols in view.ci.rows.items() for c in cols.values() if c is item]
    assert rows, "item is not in the trace layout"
    return rows[0]


# ------------------------------------------------------------------------------------------

def test_the_view_option_is_off_by_default_and_draws_nothing(window, qtbot, tmp_path,
                                                             fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    assert window._act_confidence_track.isCheckable()
    assert not window._act_confidence_track.isChecked()
    assert window._trace_view.confidence_plot() is None
    # ...though the probability IS kept with the table.
    res = window._analysis_panel.current_detection()
    assert res.module_id == "detect.keras" and isinstance(res.confidence, ConfidenceTrack)


def test_the_pane_sits_under_the_trace_and_shows_the_run(window, qtbot, tmp_path, fake_model):
    doc_id, ch = _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    window._analysis_panel._detection.ml_enter.setValue(0.6)
    window._analysis_panel._detection.ml_exit.setValue(0.3)
    _run_ml(window, qtbot)
    view = window._trace_view
    label_before = view.x_axis_label()
    bottom = view.panes()[-1].plot

    window._act_confidence_track.setChecked(True)
    plot = view.confidence_plot()
    assert plot is not None
    # UNDER the trace, x-linked to the anchor, y the probability's whole range.
    assert _layout_row(view, plot) > _layout_row(view, bottom)
    assert plot.vb.linkedView(pg.ViewBox.XAxis) is view.anchor_plot().vb
    y0, y1 = plot.vb.viewRange()[1]
    assert y0 < 0.0 < 1.0 < y1 and y1 - y0 < 1.2
    assert plot.maximumHeight() <= 90

    # The thresholds this run decoded with, enter first.
    track = window._analysis_panel.current_detection().confidence
    assert (track.enter, track.exit) == (pytest.approx(0.6), pytest.approx(0.3))
    lines = view.confidence_threshold_lines()
    assert [ln.value() for ln in lines] == [pytest.approx(0.6), pytest.approx(0.3)]

    # The curve: peak decimation and clip-to-view (ChannelPane.make_curve), on the
    # document's own x samples, high on the events.
    curve = view.confidence_curve()
    assert curve.opts["autoDownsample"] is True
    assert curve.opts["downsampleMethod"] == "peak"
    assert curve.opts["clipToView"] is True
    xs = window._session.x_axis(doc_id, 0.0).values
    assert curve.xData.size == track.covered
    assert curve.xData[0] == xs[track.start]
    for s, e in _EVENTS:
        assert curve.yData[s + 50:e - 50].mean() > 0.5
    assert curve.yData[20_000:25_000].max() < 0.3

    # One time axis, moved to the bottom of the stack and handed back.
    assert view.x_axis_label() == label_before
    assert bottom.getAxis("bottom").style["showValues"] is False
    assert plot.getAxis("bottom").style["showValues"] is True
    assert view.confidence_note().startswith("ML probability")

    window._act_confidence_track.setChecked(False)
    assert view.confidence_plot() is None
    assert bottom.getAxis("bottom").style["showValues"] is True
    assert view.x_axis_label() == label_before


def test_toggling_the_pane_keeps_the_users_zoom(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    view = window._trace_view
    pane = view.panes()[0]
    view.set_x_range(0.29, 0.31)
    pane.plot.vb.setYRange(60.0, 110.0, padding=0)
    qtbot.wait(10)
    x_before = tuple(pane.plot.vb.viewRange()[0])
    y_before = tuple(pane.plot.vb.viewRange()[1])

    window._act_confidence_track.setChecked(True)
    qtbot.wait(10)
    assert view.panes()[0] is pane                         # no rebuild
    assert tuple(pane.plot.vb.viewRange()[0]) == pytest.approx(x_before)
    assert tuple(pane.plot.vb.viewRange()[1]) == pytest.approx(y_before)
    window._act_confidence_track.setChecked(False)
    qtbot.wait(10)
    assert tuple(pane.plot.vb.viewRange()[0]) == pytest.approx(x_before)
    assert tuple(pane.plot.vb.viewRange()[1]) == pytest.approx(y_before)


def test_the_pane_follows_the_table_on_screen(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_schmitt(window, qtbot)
    window._act_confidence_track.setChecked(True)
    assert window._trace_view.confidence_plot() is None     # a Schmitt table: nothing to draw
    _run_ml(window, qtbot)
    assert window._trace_view.confidence_plot() is not None

    _show_table(window, "detect.schmitt")
    assert window._trace_view.confidence_plot() is None
    _show_table(window, "detect.keras")
    assert window._trace_view.confidence_plot() is not None


def test_turning_it_on_with_no_ml_table_says_so(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    messages: list[str] = []
    window._analysis_panel.statusMessage.connect(messages.append)
    window._act_confidence_track.setChecked(True)
    assert window._trace_view.confidence_plot() is None
    assert any("no ML table" in m for m in messages)


def test_the_pane_survives_a_rebuild(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    window._act_confidence_track.setChecked(True)
    view = window._trace_view
    first = view.confidence_plot()
    for mode in (ViewMode.STACKED, ViewMode.OVERLAY):
        window._choose_view_mode(mode)
        qtbot.wait(10)
        plot = view.confidence_plot()
        assert plot is not None and plot is not first
        assert plot.vb.linkedView(pg.ViewBox.XAxis) is view.anchor_plot().vb
        assert view.panes()[-1].plot.getAxis("bottom").style["showValues"] is False
        first = plot


def test_equal_thresholds_draw_one_line(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    window._analysis_panel._detection.ml_enter.setValue(0.4)
    window._analysis_panel._detection.ml_exit.setValue(0.4)
    _run_ml(window, qtbot)
    window._act_confidence_track.setChecked(True)
    lines = window._trace_view.confidence_threshold_lines()
    assert [ln.value() for ln in lines] == [pytest.approx(0.4)]


def test_curation_edits_keep_the_same_track_and_pane(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    window._act_confidence_track.setChecked(True)
    view = window._trace_view
    panel = window._analysis_panel
    track = panel.current_detection().confidence
    plot = view.confidence_plot()
    first_id = int(panel.current_detection().payload.df["event_id"].iloc[0])
    panel.set_event_accepted(first_id, False)
    qtbot.wait(10)
    assert panel.current_detection().confidence is track
    assert view.confidence_plot() is plot                  # the same request: not redrawn


def test_a_reopened_project_says_the_probability_is_not_saved(qapp, qtbot, tmp_path,
                                                              fake_model):
    csv = tmp_path / "pore.csv"
    arr = _event_trace()
    _write_csv(csv, arr)
    src = MainWindow()
    qtbot.addWidget(src)
    _install(src, _doc(csv, arr))
    _run_ml(src, qtbot)
    assert isinstance(src._analysis_panel.current_detection().confidence, ConfidenceTrack)
    proj = tmp_path / "ml.nanoscope"
    assert src._write_project(str(proj)) is True
    src._analysis_controller.abort()

    w = MainWindow()
    qtbot.addWidget(w)
    try:
        with qtbot.waitSignal(w.projectRestored, timeout=30_000):
            w._open_project(str(proj))
        qtbot.wait(20)
        _show_table(w, "detect.keras")
        res = w._analysis_panel.current_detection()
        assert res is not None and res.module_id == "detect.keras"
        assert res.confidence is None                      # not saved in the project
        w._act_confidence_track.setChecked(True)
        view = w._trace_view
        assert view.confidence_plot() is not None
        assert view.confidence_curve() is None
        assert view.confidence_note() == NOT_KEPT_NOTE
        assert "re-run" in view.confidence_note()

        # The note pane is a standing request like the curve pane: it survives a rebuild
        # (its boxed caption is detached and rebuilt with it), and turning the option off
        # hands the time axis back.
        for mode in (ViewMode.OVERLAY, ViewMode.STACKED):
            w._choose_view_mode(mode)
            qtbot.wait(10)
            assert view.confidence_plot() is not None
            assert view.confidence_note() == NOT_KEPT_NOTE
        w._act_confidence_track.setChecked(False)
        assert view.confidence_plot() is None and view.confidence_note() == ""
        assert view.panes()[-1].plot.getAxis("bottom").style["showValues"] is True
        assert view.x_axis_label() != ""

        # ...and a re-run in the reopened session brings the curve back.
        w._act_confidence_track.setChecked(True)
        _run_ml(w, qtbot)
        assert view.confidence_curve() is not None
        assert view.confidence_note().startswith("ML probability")
    finally:
        w._analysis_controller.abort()


# ------------------------------------------------------------------------------------------
# where the curve lands, what the crosshair reads, and what moves it
# ------------------------------------------------------------------------------------------

def test_a_visible_view_run_lands_on_the_documents_samples(window, qtbot, tmp_path,
                                                           fake_model):
    """A sub-range track starts at the slice offset, so the curve's first x is the x of
    document sample ``start`` -- not of sample 0 -- and the crosshair reads p there."""
    doc_id, _ = _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    view = window._trace_view
    view.set_x_range(0.25, 0.50)
    qtbot.wait(20)
    _run_ml(window, qtbot, visible_view=True)
    track = window._analysis_panel.current_detection().confidence
    assert 0 < track.start and track.stop < N              # a genuine sub-range

    window._act_confidence_track.setChecked(True)
    curve = view.confidence_curve()
    xs = window._session.x_axis(doc_id, 0.0).values
    assert curve.xData[0] == xs[track.start]
    assert curve.xData.size == track.covered
    assert curve.yData[30_100 - track.start] > 0.5          # inside the 30,000 event

    _vline, readout = view._confidence_cursor
    window._session.set_cursor(float(xs[30_100]))
    qtbot.wait(5)
    assert readout.textItem.toPlainText() == f"p = {track.at(30_100):.2f}"
    # Outside the analysed range there is no p to read, so the readout is empty, not 0.
    window._session.set_cursor(float(xs[1_000]))
    qtbot.wait(5)
    assert readout.textItem.toPlainText() == ""


def test_a_hover_over_the_strip_moves_the_shared_cursor_to_a_sample(window, qtbot,
                                                                    tmp_path, fake_model):
    doc_id, _ = _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    window._act_confidence_track.setChecked(True)
    window.resize(1_200, 800)
    window.show()
    view = window._trace_view
    view.set_x_range(0.29, 0.31)
    qtbot.wait(30)
    plot = view.confidence_plot()
    assert plot is not None
    target = 0.300_123_4                                    # between two samples (dt 10 us)
    scene_pos = plot.vb.mapViewToScene(QtCore.QPointF(target, 0.5))
    assert plot.sceneBoundingRect().contains(scene_pos)
    view.scene().sigMouseMoved.emit(scene_pos)
    qtbot.wait(5)
    xs = window._session.x_axis(doc_id, 0.0).values
    cx = window._session.cursor_x
    nearest = int(np.argmin(np.abs(xs - target)))
    assert cx == xs[nearest]                                 # snapped to a true sample
    _vline, readout = view._confidence_cursor
    assert readout.textItem.toPlainText().startswith("p = ")


def test_the_curve_follows_a_layer_time_offset(window, qtbot, tmp_path, fake_model):
    doc_id, _ = _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    window._act_confidence_track.setChecked(True)
    window._session.set_layer_time_offset(f"{doc_id}#0", 0.01)
    qtbot.wait(10)
    track = window._analysis_panel.current_detection().confidence
    xs = window._session.x_axis(doc_id, 0.01).values
    assert window._trace_view.confidence_curve().xData[0] == pytest.approx(xs[track.start])


# ------------------------------------------------------------------------------------------
# undo, and a channel that is not on screen
# ------------------------------------------------------------------------------------------

def test_undo_of_a_rerun_redraws_the_earlier_track(window, qtbot, tmp_path, fake_model):
    _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    det = window._analysis_panel._detection
    det.ml_enter.setValue(0.6)
    det.ml_exit.setValue(0.3)
    _run_ml(window, qtbot)
    first = window._analysis_panel.current_detection().confidence
    window._act_confidence_track.setChecked(True)
    det.ml_enter.setValue(0.4)
    det.ml_exit.setValue(0.2)
    _run_ml(window, qtbot)
    second = window._analysis_panel.current_detection().confidence
    assert second is not first
    view = window._trace_view
    assert [ln.value() for ln in view.confidence_threshold_lines()] == [
        pytest.approx(0.4), pytest.approx(0.2)]

    window._act_undo.trigger()
    qtbot.wait(20)
    assert window._analysis_panel.current_detection().confidence is first
    assert [ln.value() for ln in view.confidence_threshold_lines()] == [
        pytest.approx(0.6), pytest.approx(0.3)]


def test_a_hidden_channel_says_so_and_the_pane_comes_back_with_it(window, qtbot, tmp_path,
                                                                  fake_model):
    doc_id, _ = _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    window._session.set_layer_visible(f"{doc_id}#0", False)
    qtbot.wait(10)
    messages: list[str] = []
    window._analysis_panel.statusMessage.connect(messages.append)
    window._act_confidence_track.setChecked(True)
    view = window._trace_view
    assert view.confidence_plot() is None
    assert any("hidden" in m for m in messages), messages
    assert not any("no ML table" in m for m in messages)
    window._session.set_layer_visible(f"{doc_id}#0", True)
    qtbot.wait(10)
    assert view.confidence_plot() is not None               # the request stood


def test_the_strip_readout_is_the_trace_panes_own_readout(window, qtbot, tmp_path, fake_model):
    """One crosshair readout across the stack: the strip uses the trace panes' readout class,
    so it draws nothing before the first hover, flips to the line's left in the right half and
    survives a rebuild without leaving a second readout behind."""
    from nanoscope.gui.trace_view import _CrosshairReadout
    doc_id, _ = _install(window, _doc(tmp_path / "pore.dat", _event_trace()))
    _run_ml(window, qtbot)
    window.resize(1_200, 800)
    window.show()
    window._act_confidence_track.setChecked(True)
    view = window._trace_view
    _vline, readout = view._confidence_cursor
    assert isinstance(readout, _CrosshairReadout)
    assert not readout.isVisible(), "no readout at x = 0 before anyone hovered"
    xs = window._session.x_axis(doc_id, 0.0).values
    view.set_x_range(0.0, 0.6)
    qtbot.wait(30)
    window._session.set_cursor(float(xs[45_100]))           # right half of the view
    qtbot.wait(5)
    assert readout.isVisible() and readout.anchor.x() > 0.0
    for mode in (ViewMode.OVERLAY, ViewMode.STACKED):
        window._choose_view_mode(mode)
        qtbot.wait(10)
        plot = view.confidence_plot()
        assert sum(isinstance(i, _CrosshairReadout) for i in plot.items) == 1
        assert all(sum(isinstance(i, _CrosshairReadout) for i in p.plot.items) == 1
                   for p in view.panes())
