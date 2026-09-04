"""Focused contracts for the adaptive navigator and manual scan strip."""
from __future__ import annotations

import os
from pathlib import Path

import nanoscope  # noqa: F401  -- bootstrap Qt before importing PySide

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.navigator import LOCAL_CONTEXT_FACTOR, Navigator
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.pan_scrollbar import LOCAL_ZOOM_FACTOR, SCAN_STEP_FRACTION, PanScrollBar
from nanoscope.gui.trace_view import TraceView


FS = 10.0
N = 3_001  # exactly 300 seconds from the first through the last sample


def _document(tmp_path: Path) -> TraceDocument:
    voltage = np.full(N, -700.0, dtype=np.float32)
    current = np.linspace(-60.0, -40.0, N, dtype=np.float32)
    channels = (
        ChannelInfo(index=0, name="command", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
    )
    meta = RecordingMeta(
        path=tmp_path / "manual_scan_300s.dat",
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=channels,
    )
    return TraceDocument(meta=meta, channels=(voltage, current))


def _one_channel_document(tmp_path: Path, name: str, n_samples: int) -> TraceDocument:
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=n_samples,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(ChannelInfo(index=0, name="current", raw_units="pA",
                              kind=ChannelKind.CURRENT),),
    )
    return TraceDocument(meta=meta, channels=(np.zeros(n_samples, dtype=np.float32),))


@pytest.fixture
def scan_rig(qapp, qtbot, tmp_path):
    session = Session()
    doc = _document(tmp_path)
    session.add_document(doc)
    session.add_layer(doc.doc_id, 0)
    session.add_layer(doc.doc_id, 1)

    view = TraceView(session)
    view.resize(1_200, 600)
    navigator = Navigator()
    navigator.resize(1_200, 80)
    navigator.bind(session, view)
    scan = PanScrollBar()
    scan.bind(session, view)
    for widget in (view, navigator, scan):
        qtbot.addWidget(widget)

    navigator.set_target(doc.doc_id, 1)
    scan.set_target(doc.doc_id, 1)
    qapp.processEvents()
    return session, doc.doc_id, view, navigator, scan


def _xrange(view: TraceView) -> tuple[float, float]:
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    return float(lo), float(hi)


def _assert_range(actual, expected, *, abs_tol: float = 1e-9) -> None:
    assert actual == pytest.approx(expected, abs=abs_tol)


def test_two_millisecond_view_gets_a_nine_view_local_lens(scan_rig, qapp):
    _session, _doc_id, view, navigator, _scan = scan_rig

    view.zoom_x_to(149.999, 150.001)
    qapp.processEvents()

    visible = _xrange(view)
    context = navigator.context_range()
    assert navigator.is_local_mode() is True
    assert context is not None
    assert visible[1] - visible[0] == pytest.approx(0.002, abs=1e-10)
    assert (context[1] - context[0]) / (visible[1] - visible[0]) \
        == pytest.approx(LOCAL_CONTEXT_FACTOR, rel=1e-8)
    assert 0.5 * (context[0] + context[1]) == pytest.approx(150.0, abs=1e-9)
    assert navigator._mode_label.text() == "Local context · 9× view"


def test_reset_returns_the_adaptive_lens_to_the_full_overview(scan_rig, qapp):
    _session, _doc_id, view, navigator, _scan = scan_rig
    view.zoom_x_to(149.999, 150.001)
    assert navigator.is_local_mode()

    view.reset_zoom()
    qapp.processEvents()

    assert navigator.is_local_mode() is False
    _assert_range(navigator.context_range(), view.full_x_extent())
    assert navigator._mode_label.text() == "Full recording"


def test_navigator_exactly_mirrors_padded_end_views_then_contracts_after_a_drag(
        scan_rig, qapp):
    _session, _doc_id, view, navigator, _scan = scan_rig

    # Symmetric framing around the last event legitimately contains blank space after the data.
    view.zoom_x_to(299.5, 300.5)
    qapp.processEvents()
    _assert_range(navigator._region.getRegion(), (299.5, 300.5))
    assert navigator.context_range()[1] == pytest.approx(300.5)

    # A body move back onto data is allowed, mirrors exactly, and withdraws the temporary 0.5 s
    # drag allowance. A subsequent edge gesture still works within the ordinary target bounds.
    navigator._region.setRegion((299.0, 300.0))
    qapp.processEvents()
    _assert_range(_xrange(view), (299.0, 300.0))
    _assert_range(navigator._region.getRegion(), (299.0, 300.0))
    assert navigator.context_range()[1] == pytest.approx(300.0)

    navigator._region.setRegion((299.0, 299.8))
    qapp.processEvents()
    _assert_range(_xrange(view), (299.0, 299.8))
    _assert_range(navigator._region.getRegion(), (299.0, 299.8))

    # The first-event case is the exact mirror image and must not be clipped either.
    view.zoom_x_to(-0.5, 0.5)
    qapp.processEvents()
    _assert_range(navigator._region.getRegion(), (-0.5, 0.5))
    assert navigator.context_range()[0] == pytest.approx(-0.5)


def test_explicit_target_selects_that_layer_for_local_context(scan_rig, qapp):
    session, doc_id, _view, navigator, scan = scan_rig
    current = np.asarray(session.calibrated(doc_id, 1).data)
    assert np.array_equal(np.asarray(navigator._curve.yData), current)

    navigator.set_target(doc_id, 0)
    scan.set_target(doc_id, 0)
    qapp.processEvents()

    voltage = np.asarray(session.calibrated(doc_id, 0).data)
    assert np.array_equal(np.asarray(navigator._curve.yData), voltage)
    assert navigator._overview_layer().channel == 0
    assert scan._movement_extent() == pytest.approx((0.0, 300.0))


def test_short_overlay_target_bounds_navigator_and_scan_to_the_same_extent(
        qapp, qtbot, tmp_path):
    session = Session()
    short = _one_channel_document(tmp_path, "short_10s.dat", 101)
    long = _one_channel_document(tmp_path, "long_30s.dat", 301)
    for document in (short, long):
        session.add_document(document)
        session.add_layer(document.doc_id, 0)
    view = TraceView(session)
    navigator = Navigator()
    navigator.bind(session, view)
    scan = PanScrollBar()
    scan.bind(session, view)
    for widget in (view, navigator, scan):
        qtbot.addWidget(widget)
    navigator.set_target(short.doc_id, 0)
    scan.set_target(short.doc_id, 0)
    view.zoom_x_to(2.0, 4.0)
    qapp.processEvents()

    assert navigator._extent == pytest.approx((0.0, 10.0))
    assert scan._extent == pytest.approx((0.0, 10.0))
    # A Navigator proposal beyond the longer neighbour cannot drive the view off the target.
    navigator._region.setRegion((20.0, 22.0))
    qapp.processEvents()
    _assert_range(_xrange(view), (2.0, 4.0))
    assert navigator._region.getRegion()[1] <= 10.0

    while scan.next_view():
        pass
    assert _xrange(view)[1] <= 10.0 + 1e-12


def test_clearing_old_measure_refreshes_scan_to_the_new_target_extent(qapp, qtbot, tmp_path):
    session = Session()
    old = _one_channel_document(tmp_path, "old_10s.dat", 101)
    new = _one_channel_document(tmp_path, "new_30s.dat", 301)
    for document in (old, new):
        session.add_document(document)
        session.add_layer(document.doc_id, 0)
    view = TraceView(session)
    scan = PanScrollBar()
    scan.bind(session, view)
    qtbot.addWidget(view)
    qtbot.addWidget(scan)

    view.set_measure_active(old.doc_id, 0, True)
    scan.set_target(new.doc_id, 0)
    assert scan._extent == pytest.approx((0.0, 10.0))  # live cursor target takes precedence

    view.set_measure_active(old.doc_id, 0, False)
    qapp.processEvents()
    assert scan._extent == pytest.approx((0.0, 30.0))


def test_next_and_previous_move_ninety_percent_without_width_drift(scan_rig):
    _session, _doc_id, view, _navigator, scan = scan_rig
    view.zoom_x_to(100.0, 110.0)
    width = _xrange(view)[1] - _xrange(view)[0]

    assert scan.next_view() is True
    _assert_range(_xrange(view), (100.0 + SCAN_STEP_FRACTION * width,
                                  110.0 + SCAN_STEP_FRACTION * width))
    assert _xrange(view)[1] - _xrange(view)[0] == pytest.approx(width, rel=1e-12)

    assert scan.previous_view() is True
    _assert_range(_xrange(view), (100.0, 110.0))
    assert _xrange(view)[1] - _xrange(view)[0] == pytest.approx(width, rel=1e-12)


def test_scan_steps_clamp_at_both_target_boundaries(scan_rig):
    _session, _doc_id, view, _navigator, scan = scan_rig

    view.zoom_x_to(285.0, 295.0)
    assert scan.next_view() is True
    _assert_range(_xrange(view), (290.0, 300.0))
    assert scan.next_view() is False
    _assert_range(_xrange(view), (290.0, 300.0))

    view.zoom_x_to(5.0, 15.0)
    assert scan.previous_view() is True
    _assert_range(_xrange(view), (0.0, 10.0))
    assert scan.previous_view() is False
    _assert_range(_xrange(view), (0.0, 10.0))


def test_scan_never_reverses_direction_from_a_padded_boundary_view(scan_rig):
    """Event jumps pad symmetrically and may put part of the view beyond the target extent."""
    _session, _doc_id, view, _navigator, scan = scan_rig

    view.zoom_x_to(299.5, 300.5)
    before = _xrange(view)
    assert scan._next_button.isEnabled() is False
    assert scan.next_view() is False
    _assert_range(_xrange(view), before)

    view.zoom_x_to(-0.5, 0.5)
    before = _xrange(view)
    assert scan._previous_button.isEnabled() is False
    assert scan.previous_view() is False
    _assert_range(_xrange(view), before)


def test_clamped_scan_translates_live_measure_by_the_actual_view_delta(scan_rig, qapp):
    _session, doc_id, view, navigator, scan = scan_rig
    view.zoom_x_to(285.0, 295.0)
    view.set_measure_active(doc_id, 1, True)
    view.set_measure_region(286.0, 287.0)
    qapp.processEvents()
    before = view.measure_x_range()
    assert before == pytest.approx((286.0, 287.0))

    # The requested 90% step is 9 s, but the recording boundary permits only 5 s.
    assert scan.next_view() is True
    qapp.processEvents()

    _assert_range(_xrange(view), (290.0, 300.0))
    _assert_range(view.measure_x_range(), (291.0, 292.0))
    _assert_range(navigator._measure_band.getRegion(), (291.0, 292.0))


@pytest.mark.parametrize("offset", (0.0, 0.25))
def test_scan_cursor_domain_includes_the_half_open_trace_end(scan_rig, qapp, offset):
    session, doc_id, view, navigator, scan = scan_rig
    session.set_layer_time_offset(f"{doc_id}#1", offset)
    qapp.processEvents()
    view.zoom_x_to(298.0 + offset, 299.0 + offset)
    view.set_measure_active(doc_id, 1, True)
    view.set_measure_region(298.4 + offset, 299.4 + offset)
    assert view.measure_bounds() == (2_984, 2_994)

    assert scan.next_view() is True
    qapp.processEvents()

    _assert_range(view.measure_x_range(), (299.1 + offset, 300.1 + offset))
    assert view.measure_bounds() == (2_991, 3_001)
    _assert_range(navigator._measure_band.getRegion(),
                  (299.1 + offset, 300.1 + offset))


def test_measure_translation_refuses_an_overwide_region(scan_rig):
    _session, doc_id, view, _navigator, _scan = scan_rig
    view.set_measure_active(doc_id, 1, True)
    view.set_measure_region(-1.0, 301.0)
    before = view.measure_x_range()

    assert view.translate_measure_region(1.0) is False
    _assert_range(view.measure_x_range(), before)


def test_measure_seed_is_target_local_when_overlay_view_is_disjoint(qapp, qtbot, tmp_path):
    session = Session()
    short = _one_channel_document(tmp_path, "seed_short_10s.dat", 101)
    long = _one_channel_document(tmp_path, "seed_long_300s.dat", 3_001)
    for document in (short, long):
        session.add_document(document)
        session.add_layer(document.doc_id, 0)
    view = TraceView(session)
    scan = PanScrollBar()
    scan.bind(session, view)
    qtbot.addWidget(view)
    qtbot.addWidget(scan)
    scan.set_target(short.doc_id, 0)
    view.zoom_x_to(120.0, 180.0)
    qapp.processEvents()

    view.set_measure_active(short.doc_id, 0, True)
    qapp.processEvents()
    lo, hi = view.measure_x_range()
    assert 0.0 <= lo < hi <= 10.1
    assert 0.5 * (lo + hi) == pytest.approx(5.05)
    start, stop = view.measure_bounds()
    assert 0 <= start < stop <= 101

    # This old overlay window cannot be translated into the shorter target without changing zoom.
    before = _xrange(view)
    assert scan._previous_button.isEnabled() is False
    assert scan._next_button.isEnabled() is False
    assert scan._zoom_in_button.isEnabled() is True
    assert scan.previous_view() is False
    assert scan.next_view() is False
    _assert_range(_xrange(view), before)
    assert scan.zoom_out() is False
    _assert_range(_xrange(view), before)

    from nanoscope.gui.measure_panel import MeasurePanel
    measure = MeasurePanel(session, view)
    qtbot.addWidget(measure)
    measure.set_selection(short.doc_id, 0)
    measure.set_active(True)
    valid_seed = view.measure_x_range()
    measure._snap_to_view()
    _assert_range(view.measure_x_range(), valid_seed)
    assert "does not overlap" in measure._status.text().lower()

    view.zoom_x_to(2.0, 4.0)
    qapp.processEvents()
    measure._snap_to_view()
    _assert_range(view.measure_x_range(), (2.8, 3.2))
    assert "middle 20%" in measure._status.text().lower()
    assert "does not overlap" not in measure._status.text().lower()

    view.zoom_x_to(8.0, 12.0)
    qapp.processEvents()
    measure._snap_to_view()
    _assert_range(view.measure_x_range(), (8.65, 9.45))
    assert "visible part" in measure._status.text().lower()


def test_trace_double_click_places_a_capped_mask_without_resetting_view(
        scan_rig, qapp, qtbot):
    """The old reset gesture is now direct, click-centred annotation placement."""
    session, doc_id, view, _navigator, _scan = scan_rig
    from nanoscope.gui.measure_panel import MeasurePanel

    measure = MeasurePanel(session, view)
    qtbot.addWidget(measure)
    measure.set_selection(doc_id, 1)
    view.show()
    view.zoom_x_to(100.0, 110.0)
    qapp.processEvents()
    pane = view.pane_for_layer(f"{doc_id}#1")
    assert pane is not None

    class _DoubleClick:
        def __init__(self, scene_pos):
            self._scene_pos = scene_pos
            self.accepted = False

        def double(self): return True
        def button(self): return QtCore.Qt.LeftButton
        def scenePos(self): return self._scene_pos
        def accept(self): self.accepted = True

    def double_click_at(x: float) -> _DoubleClick:
        y0, y1 = pane.plot.vb.viewRange()[1]
        scene_pos = pane.plot.vb.mapViewToScene(QtCore.QPointF(x, 0.5 * (y0 + y1)))
        event = _DoubleClick(scene_pos)
        pane.plot.vb.mouseClickEvent(event)
        qapp.processEvents()
        return event

    before = _xrange(view)
    event = double_click_at(107.0)
    assert event.accepted is True
    assert measure.is_active() is True
    _assert_range(_xrange(view), before)
    lo, hi = view.measure_x_range()
    assert hi - lo == pytest.approx(0.20 * (before[1] - before[0]))
    assert 0.5 * (lo + hi) == pytest.approx(107.0)
    assert before[0] <= lo < hi <= before[1]

    # Near the left edge the same-size mask clamps wholly into view instead of spilling out.
    double_click_at(100.1)
    _assert_range(_xrange(view), before)
    _assert_range(view.measure_x_range(), (100.0, 102.0))


def test_full_button_restores_complete_trace_without_moving_live_mask(scan_rig, qapp):
    _session, doc_id, view, navigator, scan = scan_rig
    view.zoom_x_to(100.0, 110.0)
    view.set_measure_active(doc_id, 1, True)
    view.set_measure_region(103.0, 104.0)
    qapp.processEvents()
    assert scan._full_trace_button.isEnabled() is True

    scan._full_trace_button.click()
    qapp.processEvents()

    _assert_range(_xrange(view), view.full_x_extent())
    _assert_range(view.measure_x_range(), (103.0, 104.0))
    assert navigator._mode_label.text() == "Full recording"
    assert scan._full_trace_button.isEnabled() is False


def test_hidden_aligned_target_uses_the_visible_union_for_both_controllers(
        qapp, qtbot, tmp_path):
    session = Session()
    channels = (
        ChannelInfo(index=0, name="hidden", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="shown", raw_units="pA", kind=ChannelKind.CURRENT),
    )
    meta = RecordingMeta(
        path=tmp_path / "hidden_aligned_target.dat",
        format_name="synthetic",
        n_samples=101,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=channels,
    )
    document = TraceDocument(
        meta=meta,
        channels=(np.zeros(101, dtype=np.float32), np.ones(101, dtype=np.float32)),
    )
    session.add_document(document)
    hidden = session.add_layer(document.doc_id, 0)
    session.add_layer(document.doc_id, 1)
    session.set_layer_time_offset(hidden.layer_id, 5.0)
    view = TraceView(session)
    navigator = Navigator()
    navigator.bind(session, view)
    scan = PanScrollBar()
    scan.bind(session, view)
    for widget in (view, navigator, scan):
        qtbot.addWidget(widget)
    navigator.set_target(document.doc_id, 0)
    scan.set_target(document.doc_id, 0)

    session.set_layer_excluded(hidden.layer_id, True)
    qapp.processEvents()

    assert view.pane_for_layer(hidden.layer_id) is None
    assert navigator._extent == pytest.approx((0.0, 10.0))
    assert scan._movement_extent() == pytest.approx((0.0, 10.0))
    view.zoom_x_to(8.0, 10.0)
    assert scan.next_view() is False
    _assert_range(_xrange(view), (8.0, 10.0))


def test_voltage_row_navigation_uses_the_resolved_current_annotation_target(
        qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    channels = (
        ChannelInfo(index=0, name="command", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
    )
    meta = RecordingMeta(
        path=tmp_path / "resolved_navigation_target.dat",
        format_name="synthetic",
        n_samples=101,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=channels,
    )
    document = TraceDocument(
        meta=meta,
        channels=(np.full(101, -700.0, dtype=np.float32),
                  np.linspace(-60.0, -40.0, 101, dtype=np.float32)),
    )
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._session.set_layer_time_offset(f"{doc_id}#0", -5.0)
    window._session.set_layer_time_offset(f"{doc_id}#1", 5.0)

    window._on_selection(doc_id, 0)  # raw selection is voltage; Analysis resolves current ch1
    qapp.processEvents()

    target = (doc_id, 1)
    assert window._analysis_panel.current_target() == target
    assert window._navigator._target == target
    assert window._pan_scrollbar._target == target
    assert window._navigator._extent == pytest.approx((5.0, 15.0))
    assert window._pan_scrollbar._movement_extent() == pytest.approx((5.0, 15.0))
    assert np.array_equal(np.asarray(window._navigator._curve.yData),
                          np.asarray(document.channels[1]))

    window._analysis_panel.measure_panel.set_active(True)
    qapp.processEvents()
    assert window._trace_view.measure_target() == target
    assert window._navigator._target == target
    assert window._pan_scrollbar._movement_extent() == pytest.approx((5.0, 15.0))

    active_changes: list[bool] = []
    window._trace_view.sigMeasureActiveChanged.connect(active_changes.append)
    window._session.set_layer_excluded(f"{doc_id}#1", True)
    qapp.processEvents()

    assert active_changes and active_changes[-1] is False
    assert window._trace_view.measure_x_range() is None
    assert window._analysis_panel.measure_panel.is_active() is False
    assert window._analysis_panel.measure_panel._toggle.isEnabled() is False
    assert window._act_measure.isChecked() is False
    assert window._act_measure.isEnabled() is False
    # The save/label shortcuts remain reachable so their handlers can explain the hidden channel.
    assert window._act_add_measurement.isEnabled() is True
    assert window._act_mark_missed.isEnabled() is True


def test_hidden_annotation_target_refuses_measure_arm_and_shortcuts_out_loud(
        qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    document = _document(tmp_path)
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._on_selection(doc_id, 1)
    window._session.set_layer_excluded(f"{doc_id}#1", True)
    qapp.processEvents()
    panel = window._analysis_panel.measure_panel

    assert panel.has_open_target() is True
    assert panel.can_measure() is False
    assert panel._toggle.isEnabled() is False
    panel.set_active(True)
    assert panel.is_active() is False
    assert window._trace_view.measure_x_range() is None

    assert window._act_add_measurement.isEnabled() is True
    window._act_add_measurement.trigger()
    assert "not shown" in window.statusBar().currentMessage().lower()
    assert "file manager" in window.statusBar().currentMessage().lower()

    assert window._act_mark_missed.isEnabled() is True
    window._act_mark_missed.trigger()
    assert "not shown" in window.statusBar().currentMessage().lower()
    assert "file manager" in window.statusBar().currentMessage().lower()


def test_shift_down_cannot_zoom_in_a_union_view_larger_than_the_target(
        qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    short = _one_channel_document(tmp_path, "shortcut_short_10s.dat", 101)
    long = _one_channel_document(tmp_path, "shortcut_long_30s.dat", 301)
    for document in (short, long):
        window._session.add_document(document)
        window._create_default_layers(document)
    window._on_selection(short.doc_id, 0)
    window._session.set_layer_visible(f"{long.doc_id}#0", True)
    window._trace_view.reset_zoom()
    window.resize(1_200, 800)
    window.show()
    window._trace_view.setFocus()
    qapp.processEvents()
    before = _xrange(window._trace_view)
    assert before[1] - before[0] > 10.0

    QTest.keyClick(window._trace_view, QtCore.Qt.Key_Down, QtCore.Qt.ShiftModifier)
    qapp.processEvents()

    _assert_range(_xrange(window._trace_view), before)


@pytest.mark.parametrize("region", ((100.034, 100.287), (299.3, 300.1)))
def test_centre_action_uses_exact_live_cursor_midpoint(qapp, qtbot, tmp_path,
                                                       monkeypatch, region):
    window = MainWindow()
    qtbot.addWidget(window)
    document = _document(tmp_path)
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._on_selection(doc_id, 1)
    window._trace_view.set_measure_active(doc_id, 1, True)
    window._trace_view.set_measure_region(*region)
    qapp.processEvents()
    centred: list[float] = []
    monkeypatch.setattr(window._pan_scrollbar, "centre_on",
                        lambda x: centred.append(float(x)) or True)

    window._centre_selection_in_view()

    assert centred == [pytest.approx(0.5 * (region[0] + region[1]))]


def test_zoom_buttons_change_one_octave_about_the_same_centre(scan_rig, qapp):
    _session, _doc_id, view, _navigator, scan = scan_rig
    view.zoom_x_to(100.0, 110.0)

    scan._zoom_in_button.click()
    qapp.processEvents()
    _assert_range(_xrange(view), (102.5, 107.5))
    assert _xrange(view)[1] - _xrange(view)[0] == pytest.approx(10.0 / LOCAL_ZOOM_FACTOR)

    scan._zoom_out_button.click()
    qapp.processEvents()
    _assert_range(_xrange(view), (100.0, 110.0))


def test_zoom_in_button_disables_at_the_two_sample_floor(scan_rig):
    _session, _doc_id, view, _navigator, scan = scan_rig
    view.zoom_x_to(100.0, 100.0 + 2.0 / FS)
    before = _xrange(view)

    assert scan._zoom_in_button.isEnabled() is False
    assert scan.zoom_in() is False
    _assert_range(_xrange(view), before)


def test_measure_band_is_a_read_only_mirror(scan_rig, qapp):
    _session, doc_id, view, navigator, _scan = scan_rig
    view.zoom_x_to(100.0, 110.0)
    view.set_measure_active(doc_id, 1, True)
    view.set_measure_region(103.0, 104.0)
    qapp.processEvents()

    assert navigator._measure_band.isVisible()
    assert navigator._measure_band.movable is False
    _assert_range(navigator._measure_band.getRegion(), (103.0, 104.0))

    # Even a programmatic band move cannot write through to the working Measure region.
    navigator._measure_band.setRegion((120.0, 121.0))
    _assert_range(view.measure_x_range(), (103.0, 104.0))
    navigator._sync_measure_band()
    _assert_range(navigator._measure_band.getRegion(), (103.0, 104.0))


def test_shift_arrow_scan_shortcuts_are_scoped_to_the_trace(qapp, qtbot):
    """The annotation keys must not steal Shift+Arrow from a table or editor in a dock."""
    window = MainWindow()
    qtbot.addWidget(window)
    shortcuts = window._annotation_navigation_shortcuts
    assert {shortcut.key().toString() for shortcut in shortcuts} == {
        "Shift+Left", "Shift+Right", "Shift+Up", "Shift+Down",
    }
    assert all(shortcut.parent() is window._trace_view for shortcut in shortcuts)
    assert all(shortcut.context() == QtCore.Qt.WidgetWithChildrenShortcut
               for shortcut in shortcuts)


def test_shift_right_scans_when_trace_focused_but_not_when_event_table_focused(
        qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    document = _document(tmp_path)
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._on_selection(doc_id, 1)
    window.resize(1_200, 800)
    window.show()
    view = window._trace_view
    view.zoom_x_to(100.0, 110.0)
    qapp.processEvents()

    view.setFocus()
    QTest.keyClick(view, QtCore.Qt.Key_Right, QtCore.Qt.ShiftModifier)
    qapp.processEvents()
    _assert_range(_xrange(view), (109.0, 119.0))

    table = window._analysis_panel._table
    table.setFocus()
    before = _xrange(view)
    QTest.keyClick(table, QtCore.Qt.Key_Right, QtCore.Qt.ShiftModifier)
    qapp.processEvents()
    _assert_range(_xrange(view), before)


def test_shift_scan_shortcuts_do_not_reverse_at_padded_recording_ends(
        qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    document = _document(tmp_path)
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._on_selection(doc_id, 1)
    window.resize(1_200, 800)
    window.show()
    view = window._trace_view
    view.setFocus()

    view.zoom_x_to(299.5, 300.5)
    qapp.processEvents()
    before = _xrange(view)
    QTest.keyClick(view, QtCore.Qt.Key_Right, QtCore.Qt.ShiftModifier)
    qapp.processEvents()
    _assert_range(_xrange(view), before)

    view.zoom_x_to(-0.5, 0.5)
    qapp.processEvents()
    before = _xrange(view)
    QTest.keyClick(view, QtCore.Qt.Key_Left, QtCore.Qt.ShiftModifier)
    qapp.processEvents()
    _assert_range(_xrange(view), before)
