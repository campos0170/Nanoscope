"""The other plot windows after the build-5 review, through the routes a user takes.

* FIG-2: Refresh (and an Explorer tab switch, which refreshes) put a range pinned in Axis
  properties back in view in I-V and Voltage dependence; an appearance step still keeps a
  manual zoom.
* A pin never hides the I-V / Voltage dependence empty-state message.
* FIG-4: ⌘Z / Ctrl+Z takes back an edit made in the figure whatever the page's Section
  shows, in all six windows that edit text in the figure, and only the page in front of
  the Analysis Explorer answers.
* The Voltage dependence caption: a drag is one Undo step, saved with the project.
* Distribution summaries: a legend rename is one Undo step.
* "Add title…" in every window.

Every gesture is a real one: clicks, double-clicks and keys go through the QWINDOW
(``widget.window().windowHandle()``), menu entries and buttons are clicked, a drag is a
paced run of mouse events, and a save is a real ``_write_project`` reopened by
``_open_project``. Synthetic data only (in-memory documents, CSV traces written to
``tmp_path``); offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.key_sequences import undo_sequences
from nanoscope.gui.plot_appearance import PlotAppearance

from .test_iv import _ohmic_session
from .test_voltage_dep import _constant_df, _install_detection, _ladder_session

L = QtCore.Qt.MouseButton.LeftButton
R = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
CAPTION = "annotation.provenance"


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# -- real gestures --------------------------------------------------------------------------------

def window_handle(widget):
    """The QWindow of ``widget``'s top-level window: where a real click, drag or key is sent.

    PySide keeps ONE Python wrapper per C++ address, and a wrapper of an object that Qt deleted
    underneath it (a QLayoutItem a layout dropped) is never told. When a window is later created
    at that address, ``windowHandle()`` hands back the old wrapper, of another type, and
    ``QTest.mouseMove(QWidgetItem, QPoint)`` raises a TypeError -- once in a while, under load.
    The stale wrapper is dropped (``invalidate`` forgets the wrapper and deletes nothing) and the
    window asked again, so the gesture gets the QWindow it was after."""
    top = widget.window()
    handle = top.windowHandle()
    for _ in range(3):
        if handle is None or isinstance(handle, QtGui.QWindow):
            break
        shiboken6.invalidate(handle)
        handle = top.windowHandle()
    return handle


def _window_pos(plot, scene_pt):
    view = plot.scene().views()[0]
    return view.viewport().mapTo(view.window(), view.mapFromScene(scene_pt))


def _click(plot, scene_pt, button=R, double=False):
    handle = window_handle(plot.scene().views()[0])
    pos = _window_pos(plot, scene_pt)
    QtTest.QTest.mouseMove(handle, pos)
    if double:
        QtTest.QTest.mouseDClick(handle, button, NOMOD, pos)
    else:
        QtTest.QTest.mouseClick(handle, button, NOMOD, pos)
    QtWidgets.QApplication.processEvents()


def _drag(plot, start, end, steps=6):
    """Press, paced moves, release, through the QWindow: pyqtgraph's scene drops moves
    closer together than its rate limit, so each one waits ~30 ms."""
    handle = window_handle(plot.scene().views()[0])
    a, b = _window_pos(plot, start), _window_pos(plot, end)
    QtTest.QTest.mouseMove(handle, a)
    QtTest.QTest.qWait(30)
    QtTest.QTest.mousePress(handle, L, NOMOD, a)
    for i in range(1, steps + 1):
        QtTest.QTest.qWait(30)
        QtTest.QTest.mouseMove(handle, a + (b - a) * i / steps)
    QtTest.QTest.qWait(30)
    QtTest.QTest.mouseRelease(handle, L, NOMOD, b)
    QtWidgets.QApplication.processEvents()


def _names(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def _choose(menu, text):
    action = next(a for a in menu.actions() if a.text() == text)
    QtTest.QTest.mouseClick(menu, L, NOMOD, menu.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()


def _press_undo(page):
    """The platform's Undo key (⌘Z / Ctrl+Z), pressed on the page's window."""
    page.window().activateWindow()
    QtWidgets.QApplication.processEvents()
    combo = undo_sequences()[0][0]
    QtTest.QTest.keyClick(window_handle(page), combo.key(), combo.keyboardModifiers())
    QtWidgets.QApplication.processEvents()


def _type_return(editor, text):
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, text)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()


def _background(plot):
    vb = plot.getViewBox().sceneBoundingRect()
    return QtCore.QPointF(vb.left() + 0.55 * vb.width(), vb.top() + 0.45 * vb.height())


def _add_title(plot, text):
    """Right-click the plot itself, Add title…, type, Return."""
    controller = figure_text_for(plot)
    controller.last_menu = None
    _click(plot, _background(plot))
    menu = controller.last_menu
    assert menu is not None, "a right-click on the plot raised no menu"
    entry = next((name for name in _names(menu) if name.startswith("Add title")), None)
    assert entry is not None, _names(menu)
    _choose(menu, entry)
    editor = controller.editor
    assert editor is not None and editor.isVisible() and editor.text() == ""
    _type_return(editor, text)


def _edit_double_click(plot, item, text):
    # A new pair of clicks: QTest's synthetic clock can put it inside the last pair's
    # double-click interval, which the gate would take as that pair again.
    figure_text.GATE.reset()
    controller = figure_text_for(plot)
    _click(plot, item.sceneBoundingRect().center(), button=L, double=True)
    editor = controller.editor
    assert editor is not None, "a double-click opened no editor"
    _type_return(editor, text)


def _pin_through_axis_properties(plot, side, lo, hi):
    """Right-click the axis label, Axis properties…, untick Auto, type, OK."""
    controller = figure_text_for(plot)
    controller.last_menu = None
    label = plot.getAxis(side).label
    _click(plot, label.sceneBoundingRect().center())
    menu = controller.last_menu
    assert menu is not None and "Axis properties…" in _names(menu), menu and _names(menu)
    _choose(menu, "Axis properties…")
    popover = controller.popover
    assert popover is not None and popover.side == side
    if popover.auto.isChecked():
        QtTest.QTest.mouseClick(popover.auto, L)
    for box, value in ((popover.lo, lo), (popover.hi, hi)):
        box.selectAll()
        QtTest.QTest.keyClicks(box, f"{value:g}")
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()


def _shown(plot, side="bottom"):
    """The view's range in the units the axis shows (mV on a voltage axis)."""
    index = 0 if side == "bottom" else 1
    scale = plot.getAxis(side).scale
    lo, hi = plot.getViewBox().viewRange()[index]
    return lo * scale, hi * scale


def _refresh(page):
    """The page's "Analyze / refresh" button, clicked."""
    button = page._workflow.primary_actions[0]
    assert button.text() == "Analyze / refresh" and button.isVisible()
    QtTest.QTest.mouseClick(button, L)
    QtWidgets.QApplication.processEvents()


def _show(page, qtbot, size=(1000, 820)):
    page.resize(*size)
    page.show()
    qtbot.waitExposed(page)
    page.activateWindow()
    QtWidgets.QApplication.processEvents()
    return page


@pytest.fixture
def iv(qapp, qtbot, tmp_path):
    from nanoscope.gui.iv_panel import IVWindow

    page = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(page)
    return _show(page, qtbot)


@pytest.fixture
def vdep(qapp, qtbot, tmp_path):
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    page = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(page)
    page.set_fit("exponential")
    return _show(page, qtbot)


# -- FIG-2: a data redraw goes back to the pin; an appearance step keeps a manual zoom -------------

@pytest.mark.parametrize("make, pin", [("iv", (-700.0, -300.0)), ("vdep", (-650.0, -150.0))])
def test_refresh_puts_a_pinned_range_back_in_view(make, pin, request):
    """Review FIG-2. Before: Refresh left the data's own framing (I-V -1045..+37 mV, V-dep
    -1028..-172 mV) while Axis properties still said the axis was pinned."""
    page = request.getfixturevalue(make)
    plot = page.plot_item()
    _pin_through_axis_properties(plot, "bottom", *pin)
    assert _shown(plot) == pytest.approx(pin)
    assert page.appearance_state()["axes"]["bottom"] == {"min": pin[0], "max": pin[1]}
    _refresh(page)
    assert _shown(plot) == pytest.approx(pin)
    # An appearance step keeps a manual zoom, before and after its Undo.
    plot.getViewBox().setXRange(-0.6, -0.4, padding=0)
    _add_title(plot, "Pore 3")
    assert plot.titleLabel.isVisible()
    assert _shown(plot) == pytest.approx((-600.0, -400.0))
    assert page._appearance_widget.undo()
    assert not plot.titleLabel.isVisible()
    assert _shown(plot) == pytest.approx((-600.0, -400.0))
    _refresh(page)
    assert _shown(plot) == pytest.approx(pin)


@pytest.mark.parametrize("kind", ["noise", "events"])
def test_a_data_redraw_puts_a_pinned_range_back_in_noise_and_event_waveforms(
        kind, qapp, qtbot, tmp_path):
    """FIG-2 in the multi-view windows: each draw styled its view, THEN auto-ranged it, so a
    redraw (Refresh in Noise, a new Points count in Event waveforms' ensemble) dropped a
    range typed in Axis properties while Axis properties still showed it (Noise 10-1000 Hz
    came back as 1.9-39 844 Hz)."""
    page, plot, _edit, _shown_text = _build(kind, qtbot, tmp_path)
    if kind == "events":
        page.set_view("Ensemble")
        QtWidgets.QApplication.processEvents()
        plot = page._ensemble_plot
    host = figure_text_for(plot).host
    lo, hi = host.axis_state("bottom").shown_range
    pin = (10.0, 1000.0) if kind == "noise" else (lo + 0.25 * (hi - lo), lo + 0.5 * (hi - lo))
    _pin_through_axis_properties(plot, "bottom", *pin)
    assert host.axis_state("bottom").shown_range == pytest.approx(pin, rel=1e-3)
    if kind == "noise":
        _refresh(page)
    else:
        page._npoints.setValue(page._npoints.value() + 10)
        QtWidgets.QApplication.processEvents()
    assert host.axis_state("bottom").range == pytest.approx(pin, rel=1e-3)
    assert host.axis_state("bottom").shown_range == pytest.approx(pin, rel=1e-3)
    # An appearance step alone still keeps a zoom made by hand.
    view = plot.getViewBox()
    x0, x1 = view.viewRange()[0]
    view.setXRange(x0, x0 + 0.5 * (x1 - x0), padding=0)
    zoomed = view.viewRange()[0]
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), title_size=17.0))
    assert view.viewRange()[0] == pytest.approx(zoomed)


def _choose_in_combo(combo, data):
    """Click the combo, then arrow keys and Return to the entry, as on a keyboard."""
    combo.setFocus()
    QtTest.QTest.mouseClick(combo, L)
    QtWidgets.QApplication.processEvents()
    target = next(i for i in range(combo.count()) if combo.itemData(i) == data)
    popup = combo.view()
    while popup.currentIndex().row() != target:
        down = popup.currentIndex().row() < target
        QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Down if down else QtCore.Qt.Key.Key_Up)
    QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert combo.currentData() == data


def _frames(plot, lo, hi):
    """The view shows [lo, hi] and that span fills at least half of it."""
    a, b = plot.getViewBox().viewRange()[0]
    return a <= lo and b >= hi and (hi - lo) >= 0.5 * (b - a)


def _allpoints_framed(page):
    hist = page._hist_plot
    lo, hi = page._hist_item.dataBounds(0)
    return _frames(hist, lo, hi)


def test_a_range_pinned_on_one_noise_view_frames_only_that_view(qapp, qtbot, tmp_path):
    """figures2-1. The four Noise views share one appearance, but a pin is typed on ONE axis:
    10-1000 Hz on the PSD made All-points (pA) and the Spectrogram (s) draw 10-1000 too, so
    the spectrogram was blank, and their Axis properties showed a pin never typed there."""
    from nanoscope.gui.psd_panel import VIEW_ALLPOINTS, VIEW_PSD, VIEW_SPECTROGRAM

    page, plot, _edit, _shown_text = _build("noise", qtbot, tmp_path)
    _pin_through_axis_properties(plot, "bottom", 10.0, 1000.0)
    assert page.appearance_state()["axes"] == {"psd.bottom": {"min": 10.0, "max": 1000.0}}
    _choose_in_combo(page._view_combo, VIEW_ALLPOINTS)
    assert _allpoints_framed(page)
    assert figure_text_for(page._hist_plot).host.axis_state("bottom").range is None
    _refresh(page)
    assert _allpoints_framed(page)
    _choose_in_combo(page._view_combo, VIEW_SPECTROGRAM)
    image = page._spec_image.mapRectToParent(page._spec_image.boundingRect())
    assert _frames(page._spec_plot, image.left(), image.right())
    _choose_in_combo(page._view_combo, VIEW_PSD)
    host = figure_text_for(plot).host
    assert host.axis_state("bottom").range == pytest.approx((10.0, 1000.0))
    assert host.axis_state("bottom").shown_range == pytest.approx((10.0, 1000.0), rel=1e-3)
    assert page._appearance_widget.undo_button.toolTip() == "Undo: x axis range 10–1000"


def test_a_range_pinned_on_the_ensemble_leaves_the_event_count_map_framed(
        qapp, qtbot, tmp_path):
    """figures2-1. A time range pinned on the ensemble became the Event-count map's x range
    (log10 dwell), a blank map, and stayed through a Bins redraw."""
    page, _plot, _edit, _shown_text = _build("events", qtbot, tmp_path)
    page.set_view("Ensemble")
    QtWidgets.QApplication.processEvents()
    ensemble = page._ensemble_plot
    lo, hi = figure_text_for(ensemble).host.axis_state("bottom").shown_range
    pin = (lo + 0.25 * (hi - lo), lo + 0.5 * (hi - lo))
    _pin_through_axis_properties(ensemble, "bottom", *pin)
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())
    QtWidgets.QApplication.processEvents()
    image = page._density_image.mapRectToParent(page._density_image.boundingRect())
    assert _frames(page._density_main, image.left(), image.right())
    assert figure_text_for(page._density_main).host.axis_state("bottom").range is None
    page._bins.setValue(page._bins.value() + 4)        # a map redraw
    QtWidgets.QApplication.processEvents()
    image = page._density_image.mapRectToParent(page._density_image.boundingRect())
    assert _frames(page._density_main, image.left(), image.right())
    assert figure_text_for(ensemble).host.axis_state("bottom").shown_range == pytest.approx(
        pin, rel=1e-3)


def test_a_range_pinned_on_the_event_count_map_stays_the_maps_through_the_ensemble(
        qapp, qtbot, tmp_path):
    """figures2-1, the other way round: a range typed on the map (Log x off) is the map's
    own, kept as ``map.bottom``, drawn on the map again after a Points redraw and a view
    switch, and never the ensemble's."""
    page, mp, _edit, _shown_text = _build("events", qtbot, tmp_path)
    sections = page._workflow.sections
    sections.setFocus()
    qtbot.wait(20)       # let the focus change land before the shortcut map is asked
    QtTest.QTest.keyClick(sections, QtCore.Qt.Key.Key_2,
                          QtCore.Qt.KeyboardModifier.ControlModifier)     # Ctrl+2: Analysis
    QtWidgets.QApplication.processEvents()
    assert sections.currentText() == "Analysis" and page._logx.isVisible()
    if page._logx.isChecked():
        QtTest.QTest.mouseClick(page._logx, L)          # the map in linear dwell
        QtWidgets.QApplication.processEvents()
    assert not page._logx.isChecked()
    lo, hi = mp.getViewBox().viewRange()[0]
    pin = (float(f"{lo + 0.2 * (hi - lo):g}"), float(f"{lo + 0.7 * (hi - lo):g}"))
    # ViewBox coordinates are seconds; the visible dwell range editor uses ms.
    display_pin = tuple(value * 1000.0 for value in pin)
    _pin_through_axis_properties(mp, "bottom", *display_pin)
    assert set(page.appearance_state()["axes"]) == {"map.bottom"}
    assert mp.getViewBox().viewRange()[0] == pytest.approx(pin, rel=1e-4)
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())
    QtWidgets.QApplication.processEvents()
    assert page.current_view() == "Ensemble"
    ensemble = page._ensemble_plot
    assert figure_text_for(ensemble).host.axis_state("bottom").range is None
    assert page._npoints.isVisible()
    points = page._npoints.value()
    page._npoints.setFocus()
    QtTest.QTest.keyClick(page._npoints, QtCore.Qt.Key.Key_PageUp)     # an ensemble redraw
    QtWidgets.QApplication.processEvents()
    assert page._npoints.value() != points
    assert figure_text_for(ensemble).host.axis_state("bottom").range is None
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())
    QtWidgets.QApplication.processEvents()
    assert mp.getViewBox().viewRange()[0] == pytest.approx(pin, rel=1e-4)
    assert figure_text_for(mp).host.axis_state("bottom").range == pytest.approx(display_pin)
    assert set(page.appearance_state()["axes"]) == {"map.bottom"}


def test_a_range_pinned_on_the_subpopulation_scatter_leaves_the_waveforms_framed(
        qapp, qtbot):
    """figures2-1 in Subpopulations: its two figures are on screen together on one
    appearance, so a range typed on the scatter (embedding units) became the waveforms'
    (ms): they were drawn almost blank, and their Axis properties showed a range never
    typed there. Each figure keeps its own pins; a bare one is the scatter's."""
    from nanoscope.analysis import subpop_report
    from nanoscope.gui.subpop_window import SubpopReportWindow

    from .test_lay_subpop_window_fits_the_screen import FS, N_SAMPLES, _synthetic_events

    arr, events = _synthetic_events()
    config = subpop_report.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    report = subpop_report.run_subpopulations(arr, arr, FS, events, recording="r",
                                              duration_s=N_SAMPLES / FS, config=config)
    page = SubpopReportWindow()
    qtbot.addWidget(page)
    _show(page, qtbot, size=(1400, 880))
    page.set_report(report, subtitle="synthetic")
    QtWidgets.QApplication.processEvents()
    scatter = page._embed_view._plot.getPlotItem()
    waves = page._wave_view._plot.getPlotItem()
    framed = waves.getViewBox().viewRange()[0]
    lo, hi = scatter.getViewBox().viewRange()[0]
    pin = (float(f"{lo + 0.25 * (hi - lo):g}"), float(f"{lo + 0.5 * (hi - lo):g}"))
    _pin_through_axis_properties(scatter, "bottom", *pin)
    assert set(page.appearance_state()["axes"]) == {"embedding.bottom"}
    assert scatter.getViewBox().viewRange()[0] == pytest.approx(pin, rel=1e-4)
    assert waves.getViewBox().viewRange()[0] == pytest.approx(framed, rel=1e-6)
    assert figure_text_for(waves).host.axis_state("bottom").range is None
    # A bare pin (saved before pins were kept per figure) is the scatter's alone.
    page.set_appearance_state({"axes": {"bottom": {"min": pin[0], "max": pin[1]}}})
    QtWidgets.QApplication.processEvents()
    assert figure_text_for(scatter).host.axis_state("bottom").range == pytest.approx(pin)
    assert figure_text_for(waves).host.axis_state("bottom").range is None
    assert waves.getViewBox().viewRange()[0] == pytest.approx(framed, rel=1e-6)


def _noisy_csv(path, n=20_000, seed=3):
    rng = np.random.default_rng(seed)
    pd.DataFrame({"time": np.arange(n) / 50000.0,
                  "current": (100.0 + rng.normal(0.0, 4.0, n)).astype(np.float32)}
                 ).to_csv(path, index=False)


def test_a_noise_range_saved_before_pins_were_per_view_frames_only_the_psd(
        qapp, qtbot, tmp_path, monkeypatch):
    """figures2-1, an older project: its range was saved as a bare ``bottom`` pin, with no
    view. It is the PSD's (the view Noise opens on) and no other view's, so All-points
    reopens on its own data instead of 10-1000 pA."""
    import zipfile

    from nanoscope.core.project import DocumentRef, ProjectState
    from nanoscope.gui.main_window import MainWindow
    from nanoscope.gui.psd_panel import VIEW_ALLPOINTS

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    recording = tmp_path / "noise.csv"
    _noisy_csv(recording)
    window = _main_window(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(recording))]))
    window._open_psd()
    window._analysis_explorer.update_current()
    QtWidgets.QApplication.processEvents()
    page = window._psd_window
    assert page.has_data()
    figure_id = window._figure_workspace.current_id()
    _pin_through_axis_properties(page._plot, "bottom", 10.0, 1000.0)
    assert page.appearance_state()["axes"] == {"psd.bottom": {"min": 10.0, "max": 1000.0}}
    saved = tmp_path / "pinned.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    # The same project as a build that kept one pin per side wrote it.
    older = tmp_path / "older.nanoscope"
    with zipfile.ZipFile(saved) as src, zipfile.ZipFile(older, "w", zipfile.ZIP_DEFLATED) as dst:
        renamed = 0
        for info in src.infolist():
            data = src.read(info)
            if b'"psd.bottom"' in data:
                renamed += data.count(b'"psd.bottom"')
                data = data.replace(b'"psd.bottom"', b'"bottom"')
            dst.writestr(info, data)
    # The native binding ledger also records the axis key in v13. Both copies
    # must describe the same legacy bare pin; the assertions below check its
    # restored meaning, independent of how many serialization witnesses exist.
    assert renamed >= 1

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(older))
    assert reopened._figure_workspace.activate(figure_id)
    QtWidgets.QApplication.processEvents()
    page = reopened._figure_workspace.page(figure_id)
    assert page.appearance_state()["axes"] == {"bottom": {"min": 10.0, "max": 1000.0}}
    host = figure_text_for(page._plot).host
    assert host.axis_state("bottom").range == pytest.approx((10.0, 1000.0))
    assert host.axis_state("bottom").shown_range == pytest.approx((10.0, 1000.0), rel=1e-3)
    _choose_in_combo(page._view_combo, VIEW_ALLPOINTS)
    reopened._analysis_explorer.update_current()
    assert _allpoints_framed(page)
    assert figure_text_for(page._hist_plot).host.axis_state("bottom").range is None
    # Typed again on the PSD, the range moves to the PSD's own key, as one Undo step that
    # names the axis it changed.
    _choose_in_combo(page._view_combo, "psd")
    reopened._analysis_explorer.update_current()
    _pin_through_axis_properties(page._plot, "bottom", 20.0, 500.0)
    assert page.appearance_state()["axes"] == {"psd.bottom": {"min": 20.0, "max": 500.0}}
    assert page._appearance_widget.undo_button.toolTip() == "Undo: x axis range 20–500"
    reopened._dirty = False


@pytest.mark.parametrize("make", ["iv", "vdep"])
def test_a_saved_view_restored_into_the_open_page_goes_back_to_its_pin(make, request):
    """The project-restore draw (``_render_workspace_snapshot``) frames the saved points
    itself, as a refresh does; into a page already showing that pin, it kept the data's
    framing."""
    from nanoscope.core import analysis_workspace as workspace

    page = request.getfixturevalue(make)
    capture, restore = {"iv": (workspace.capture_iv, workspace.restore_iv),
                        "vdep": (workspace.capture_voltage_dependence,
                                 workspace.restore_voltage_dependence)}[make]
    plot = page.plot_item()
    _pin_through_axis_properties(plot, "bottom", -700.0, -300.0)
    artifact = capture(page)
    plot.getViewBox().setXRange(-0.6, -0.4, padding=0)
    assert restore(page, artifact).restored
    assert _shown(plot) == pytest.approx((-700.0, -300.0))


def test_refresh_keeps_a_pinned_y_range_too(iv):
    plot = iv.plot_item()
    _pin_through_axis_properties(plot, "left", -80000.0, -20000.0)
    assert _shown(plot, "left") == pytest.approx((-80000.0, -20000.0))
    _refresh(iv)
    assert _shown(plot, "left") == pytest.approx((-80000.0, -20000.0))


# -- the empty-state message ----------------------------------------------------------------------

def _message_in_view(page):
    item = page._message_item
    assert item is not None and item.isVisible()
    (x0, x1), (y0, y1) = page.plot_item().getViewBox().viewRange()
    return x0 <= item.pos().x() <= x1 and y0 <= item.pos().y() <= y1


@pytest.mark.parametrize("make", ["iv", "vdep"])
def test_a_pin_never_hides_the_empty_state_message(make, request):
    """INTEGRATE §3: with the x axis pinned to -900..-100 mV and no recordings left, the
    'No usable recordings…' text sat at x = 0.5, outside the pinned view: a blank plot."""
    page = request.getfixturevalue(make)
    plot = page.plot_item()
    _pin_through_axis_properties(plot, "bottom", -900.0, -100.0)
    _pin_through_axis_properties(plot, "left", 5.0, 50.0)
    session = page._session
    docs = list(session.documents())
    for doc in docs:
        session.close_document(doc.doc_id)
    _refresh(page)
    assert page.points() == []
    assert "No usable recordings" in page._message_item.toPlainText()
    assert _message_in_view(page)
    # An appearance step (the title) leaves it in view too; the pins are still the figure's.
    _add_title(plot, "Pore 3")
    assert _message_in_view(page)
    assert set(page.appearance_state()["axes"]) == {"bottom", "left"}
    if make == "iv":
        # The data come back: the pinned range is the view again.
        for doc in docs:
            session.add_document(doc)
        _refresh(page)
        assert page._message_item is None
        assert _shown(plot) == pytest.approx((-900.0, -100.0))


# -- FIG-4: the Undo key in every Section, in every window that edits text in the figure ----------

def _build(kind, qtbot, tmp_path):
    """(page, plot, edit(text), shown()) for one window, with synthetic data."""
    if kind == "iv":
        from nanoscope.gui.iv_panel import IVWindow
        page = IVWindow(_ohmic_session(tmp_path))
    elif kind == "vdep":
        from nanoscope.gui.voltage_dep_panel import VDepWindow
        page = VDepWindow(_ladder_session(tmp_path))
    elif kind == "noise":
        from .test_psd import _fed_window
        page, _doc = _fed_window(qtbot, tmp_path)
    elif kind == "events":
        from .test_event_plot import _fed_window
        page = _fed_window(qtbot, tmp_path)
    elif kind == "summaries":
        from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
        from .test_distribution_summary_panel import _source
        page = DistributionSummaryWindow()
        page.set_sources([_source([1.0, 2.0, 3.0, 4.0])])
        page.analyze()
    else:
        from nanoscope.gui.subpop_window import SubpopReportWindow
        page = SubpopReportWindow(None)
    qtbot.addWidget(page)
    _show(page, qtbot)
    plot = {"iv": lambda: page.plot_item(), "vdep": lambda: page.plot_item(),
            "noise": lambda: page._plot, "events": lambda: page._density_main,
            "summaries": lambda: page._plot,
            "subpop": lambda: page._embed_view._plot.getPlotItem()}[kind]()
    if kind in ("iv", "vdep"):
        # The x label: these windows' own, typed on the figure.
        axis = plot.getAxis("bottom")
        return (page, plot, lambda text: _edit_double_click(plot, axis.label, text),
                lambda: axis.label.toPlainText().strip())
    if kind == "subpop":
        # Its figures have no title until one is added on the figure.
        return (page, plot, lambda text: _add_title(plot, text),
                lambda: plot.titleLabel.text if plot.titleLabel.isVisible() else "")
    return (page, plot, lambda text: _edit_double_click(plot, plot.titleLabel, text),
            lambda: plot.titleLabel.text)


@pytest.mark.parametrize("kind", ["iv", "vdep", "noise", "events", "summaries", "subpop"])
def test_the_undo_key_takes_back_an_edit_made_in_the_figure_in_any_section(
        kind, qapp, qtbot, tmp_path):
    """Review FIG-4. Before: the Undo/Redo keys lived on the Appearance controls, which
    every Section but Appearance hides, so ⌘Z after an edit made in the figure did nothing
    at the default Section (Data)."""
    page, plot, edit, shown = _build(kind, qtbot, tmp_path)
    widget = page._appearance_widget
    assert page._workflow.sections.currentText() == "Data"
    assert not widget.isVisible(), "the precondition: the Appearance controls are folded away"
    before, depth = shown(), widget._history.depth()
    edit("Pore 3 edit")
    assert shown() == "Pore 3 edit"
    assert widget._history.depth() == (depth[0] + 1, 0)
    _press_undo(page)
    assert shown() == before
    assert widget._history.depth() == (depth[0], 1)


def _csv(path, value, n=4000):
    x = np.full(n, value, dtype=np.float32)
    pd.DataFrame({"time": np.arange(n) / 50000.0, "current": x}).to_csv(path, index=False)


def _main_window(qtbot, monkeypatch=None):
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _restored_main_window(qtbot, tmp_path, monkeypatch):
    from nanoscope.core.project import DocumentRef, ProjectState
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    paths = []
    for mv in (-200, -500, -1000):
        path = tmp_path / f"pore_{mv}mV.csv"
        _csv(path, 0.1 * mv - 10.0)
        paths.append(path)
    window = _main_window(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(p)) for p in paths]))
    return window


def _click_tab(explorer, page):
    tabs = explorer._tabs
    owner = explorer.figure_workspace()
    if owner is not None:
        listing = explorer.figures_list()
        explorer.refresh_figures()
        item = listing._items[owner.id_for_page(page)]
        listing.tree.scrollToItem(item)
        QtTest.QTest.mouseClick(listing.tree.viewport(), L, NOMOD,
                               listing.tree.visualItemRect(item).center())
    else:
        bar = tabs.tabBar()
        QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(tabs.indexOf(page)).center())
    QtWidgets.QApplication.processEvents()
    assert tabs.currentWidget() is page


def test_in_the_explorer_only_the_page_in_front_answers_the_undo_key(
        qapp, qtbot, tmp_path, monkeypatch):
    """FIG-4 inside the real Analysis Explorer, where the pages share one window: the key
    reaches the page in front, never one behind it, and never Edit ▸ Undo (another window).
    Figure selection preserves the committed result and its pinned range (FIG-2)."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    window._open_iv()
    iv = window._iv_window
    iv.inspector_adapter().update()
    window._open_psd()
    noise = window._psd_window
    noise.inspector_adapter().update()
    explorer = window._analysis_explorer
    _click_tab(explorer, iv)
    plot = iv.plot_item()
    _pin_through_axis_properties(plot, "bottom", -700.0, -300.0)
    axis = plot.getAxis("bottom")
    automatic = axis.label.toPlainText().strip()
    _edit_double_click(plot, axis.label, "Bias")
    assert axis.label.toPlainText().strip() == "Bias"
    iv_history, noise_history = iv._appearance_widget._history, noise._appearance_widget._history
    iv_depth, noise_depth = iv_history.depth(), noise_history.depth()
    _click_tab(explorer, noise)
    _press_undo(noise)
    assert axis.label.toPlainText().strip() == "Bias", "the page behind answered"
    assert iv_history.depth() == iv_depth and noise_history.depth() == noise_depth
    _click_tab(explorer, iv)           # selecting the figure does not recompute it
    assert _shown(plot) == pytest.approx((-700.0, -300.0))
    _press_undo(iv)
    assert axis.label.toPlainText().strip() == automatic
    assert iv_history.depth() == (iv_depth[0] - 1, 1)
    window._dirty = False


# -- the Voltage dependence caption: a drag is a step, saved with the project ---------------------

def _caption_drag(page, dx=120.0, dy=-80.0):
    plot = page.plot_item()
    caption = page._caption
    start = caption.sceneBoundingRect().center()
    _drag(plot, start, start + QtCore.QPointF(dx, dy))


def test_a_caption_drag_is_one_undo_step(vdep):
    """INTEGRATE §3 (G4 major): the drag moved the caption but was neither an Undo step
    nor part of the appearance, so Undo, templates and projects lost it."""
    page = vdep
    widget = page._appearance_widget
    caption = page._caption
    default = caption.offset()
    view = page.plot_item().getViewBox().viewRange()
    depth = widget._history.depth()
    _caption_drag(page)
    moved = caption.offset()
    assert moved[0] == pytest.approx(default[0] + 120.0, abs=6.0)
    assert moved[1] == pytest.approx(default[1] - 80.0, abs=6.0)
    assert page.plot_item().getViewBox().viewRange() == view, "the drag panned the view"
    assert widget._history.depth() == (depth[0] + 1, 0)
    assert widget.undo_button.toolTip() == "Undo: caption moved"
    saved = page.appearance_state()["annotations"][CAPTION]["offset"]
    assert saved == pytest.approx(list(moved))
    page.refresh()                                   # a redraw leaves it where it was put
    assert caption.offset() == pytest.approx(moved)
    _press_undo(page)
    assert caption.offset() == pytest.approx(default)
    assert "annotations" not in page.appearance_state()
    assert widget.redo()
    assert caption.offset() == pytest.approx(moved)


def _install_series(window, paths):
    """A detection on every open recording, so Voltage dependence has a series."""
    session = window._session
    for doc in session.documents():
        _install_detection(session, doc.doc_id, _constant_df(
            n=20, dwell_s=1e-3, depth_pA=20.0, frac_depth=0.02), duration_s=2.0)
    window._sync_vdep_action()


def test_a_dragged_caption_is_saved_with_the_project_and_reopens_there(
        qapp, qtbot, tmp_path, monkeypatch):
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    window._open_vdep()
    page = window._vdep_window
    page.inspector_adapter().update()
    QtWidgets.QApplication.processEvents()
    assert page.caption_text(), "the series has a caption to drag"
    _caption_drag(page)
    moved = page._caption.offset()
    saved = page.appearance_state()
    assert saved["annotations"][CAPTION]["offset"] == pytest.approx(list(moved))
    assert window._dirty, "a drag is an edit of the project"
    project = tmp_path / "caption.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_vdep()
    QtWidgets.QApplication.processEvents()
    page = reopened._vdep_window
    assert page.appearance_state()["annotations"] == saved["annotations"]
    assert page._caption.offset() == pytest.approx(moved)
    reopened._dirty = False


# -- Distribution summaries: a legend rename is one Undo step -------------------------------------

@pytest.fixture
def summaries(qapp, qtbot):
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
    from .test_distribution_summary_panel import _source

    page = DistributionSummaryWindow()
    qtbot.addWidget(page)
    page.set_sources([_source([1.0, 2.0, 3.0, 4.0])])
    page.analyze()
    return _show(page, qtbot)


def test_a_summaries_generate_puts_a_pinned_range_back_in_view(summaries):
    """FIG-2's defect in Distribution summaries too: Generate framed the new figure
    (auto-range) after styling it, so a pin typed in Axis properties left the view."""
    page = summaries
    plot = page._plot
    _pin_through_axis_properties(plot, "bottom", 1.5, 3.0)
    assert _shown(plot) == pytest.approx((1.5, 3.0))
    button = page._workflow.primary_actions[0]
    assert button.text().startswith("Generate")
    QtTest.QTest.mouseClick(button, L)
    QtWidgets.QApplication.processEvents()
    assert _shown(plot) == pytest.approx((1.5, 3.0))
    scale = plot.getAxis('bottom').scale
    plot.getViewBox().setXRange(2.0 / scale, 2.5 / scale, padding=0)  # zoom to 2–2.5 displayed ms
    assert _shown(plot) == pytest.approx((2.0, 2.5))
    _edit_double_click(plot, plot.titleLabel, "Dwell ECDF")
    assert _shown(plot) == pytest.approx((2.0, 2.5))        # …kept by an appearance step


def _generate(page, kind):
    _choose_in_combo(page._view_combo, kind)
    button = page._workflow.primary_actions[0]
    assert button.text().startswith("Generate")
    QtTest.QTest.mouseClick(button, L)
    QtWidgets.QApplication.processEvents()


def _reset_view(page):
    button = page.findChild(QtWidgets.QAbstractButton, "plotResetViewButton")
    assert button is not None and button.isVisible()
    QtTest.QTest.mouseClick(button, L)
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("kind", ["box", "violin"])
@pytest.mark.parametrize("side, pin", [("bottom", (1.5, 3.0)), ("left", (0.2, 0.8))])
def test_a_summaries_pin_does_not_frame_a_type_with_other_axes(summaries, kind, side, pin):
    """figures2-2. An x range pinned on the ECDF (the value) became Box/Violin's x range (the
    sources) after Type → Generate, so nothing was drawn in view, and Reset view went back
    there. A pin belongs to what the axes show: Box/Violin start on their own framing, and
    the ECDF's range comes back with the ECDF."""
    page = summaries
    plot = page._plot
    _pin_through_axis_properties(plot, side, *pin)
    assert _shown(plot, side) == pytest.approx(pin)
    steps = page._appearance_widget._history.depth()
    _generate(page, kind)
    assert plot.getAxis("bottom").labelText == "Source"
    for check in (lambda: None, lambda: _reset_view(page)):
        check()
        assert _shown(plot, "bottom") == pytest.approx((-0.6, 0.6))
        lo, hi = _shown(plot, "left")
        scale = plot.getAxis('left').scale
        assert scale == 1000, 'Dwell seconds are displayed in milliseconds'
        assert lo <= 1.0 * scale and hi >= 4.0 * scale, (lo, hi)
    assert figure_text_for(plot).host.axis_state(side).range is None
    assert page._appearance_widget._history.depth() == steps, "a Type change is no appearance step"
    _generate(page, "survival" if side == "bottom" else "ecdf")
    assert figure_text_for(plot).host.axis_state(side).range == pytest.approx(pin)
    assert _shown(plot, side) == pytest.approx(pin)


def test_a_summaries_pin_stays_through_a_type_with_the_same_axes(summaries):
    """ECDF and Survival both draw the value on x: the range stays, after Reset view too."""
    page = summaries
    plot = page._plot
    _pin_through_axis_properties(plot, "bottom", 1.5, 3.0)
    _generate(page, "survival")
    assert _shown(plot) == pytest.approx((1.5, 3.0))
    _reset_view(page)
    assert _shown(plot) == pytest.approx((1.5, 3.0))
    # A range typed on the Box is the Box's and Violin's, and never the curves'.
    _generate(page, "box")
    _pin_through_axis_properties(plot, "left", 1.5, 2.5)
    assert page.appearance_state()["axes"] == {
        "categories.left": {"min": 1.5, "max": 2.5}, "curves.bottom": {"min": 1.5, "max": 3.0}}
    _generate(page, "violin")
    assert _shown(plot, "left") == pytest.approx((1.5, 2.5))
    _generate(page, "ecdf")
    assert _shown(plot) == pytest.approx((1.5, 3.0))
    lo, hi = _shown(plot, "left")
    assert lo <= 0.25 and hi >= 1.0, (lo, hi)


def _legend_row(page):
    return page._plot.legend.items[0][1]


def _legend_text(page):
    return _legend_row(page).item.toPlainText()


def test_a_summaries_legend_rename_is_one_undo_step(summaries):
    """INTEGRATE §3 (G4): a rename typed in the legend wrote the Source names store but
    was no Undo step (the Undo button stayed disabled)."""
    page = summaries
    widget = page._appearance_widget
    depth = widget._history.depth()
    _edit_double_click(page._plot, _legend_row(page), "Apo <1>")
    assert _legend_text(page).startswith("Apo <1> · n=4")
    assert page.source_appearance_state()["measurements"]["label"] == "Apo <1>"
    assert widget._history.depth() == (depth[0] + 1, 0)
    assert widget.undo_button.isEnabled()
    assert widget.undo_button.toolTip() == "Undo: source names and colors"
    assert page._source_appearance_editor.state()["measurements"]["label"] == "Apo <1>"
    _press_undo(page)
    assert _legend_text(page).startswith("Measured events · n=4")
    assert "measurements" not in page.source_appearance_state()
    assert widget.redo()
    assert page.source_appearance_state()["measurements"]["label"] == "Apo <1>"
    # A second rename steps back to the first, not past it.
    _edit_double_click(page._plot, _legend_row(page), "Holo")
    assert widget.undo()
    assert page.source_appearance_state()["measurements"]["label"] == "Apo <1>"
    # The name is saved once, in the Source names store; Reset figure keeps it.
    assert "text" not in page.appearance_state()
    QtTest.QTest.mouseClick(widget.reset_button, L)
    assert page.source_appearance_state()["measurements"]["label"] == "Apo <1>"


def test_a_name_set_in_source_names_is_where_undo_and_redo_land(summaries):
    """Canvas and source-editor renames are consecutive steps in the same history."""
    page = summaries
    widget = page._appearance_widget
    _edit_double_click(page._plot, _legend_row(page), "A")
    editor = page._source_appearance_editor
    editor.label.setText("B")
    editor.apply.click()  # hidden compatibility route; visible renames use Format/legend
    assert page.source_appearance_state()["measurements"]["label"] == "B"
    assert widget.undo()
    assert page.source_appearance_state()["measurements"]["label"] == "A"
    assert widget.undo()
    assert "measurements" not in page.source_appearance_state()
    assert widget.redo()
    assert page.source_appearance_state()["measurements"]["label"] == "A"
    assert widget.redo()
    assert page.source_appearance_state()["measurements"]["label"] == "B"


def _editor_rename(page, text):
    """Exercise the retained legacy source-edit API beside visible canvas editing."""
    editor = page._source_appearance_editor
    editor.label.setText(text)
    editor.apply.click()
    QtWidgets.QApplication.processEvents()


def test_reset_figure_after_a_legend_rename_records_no_empty_step(summaries):
    """The legend name lives in Source names and colors, which Reset figure leaves
    alone. Reset dropped the name's carrier all the same, as a step that changed nothing on
    screen, so the next ⌘Z seemed to do nothing."""
    page = summaries
    widget = page._appearance_widget
    _edit_double_click(page._plot, _legend_row(page), "Apo")
    depth = widget._history.depth()
    QtTest.QTest.mouseClick(widget.reset_button, L)
    QtWidgets.QApplication.processEvents()
    assert widget._history.depth() == depth
    assert page.source_appearance_state()["measurements"]["label"] == "Apo"
    _press_undo(page)                       # the rename, in one press
    assert "measurements" not in page.source_appearance_state()
    assert _legend_text(page).startswith("Measured events · n=4")


def test_source_rename_and_template_recall_undo_in_edit_order(summaries):
    """A recipe preserves source identity, and the later source edit has its own Undo."""
    page = summaries
    _edit_double_click(page._plot, _legend_row(page), "Apo")
    settings, _appearance = page.template_settings()
    page.apply_template_settings(settings, PlotAppearance(title="From template").to_dict())
    QtWidgets.QApplication.processEvents()
    assert page.source_appearance_state()["measurements"]["label"] == "Apo"
    _editor_rename(page, "Typed in editor")
    _press_undo(page)                       # takes back the later source edit
    assert page._appearance_widget.state().title == "From template"
    assert page.source_appearance_state()["measurements"]["label"] == "Apo"
    _press_undo(page)                       # then takes back the template
    assert page._appearance_widget.state().title != "From template"
    assert page.source_appearance_state()["measurements"]["label"] == "Apo"


# -- Add title… in every window ------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["iv", "vdep", "noise", "events", "summaries", "subpop"])
def test_add_title_from_the_plot_menu_in_every_window(kind, qapp, qtbot, tmp_path):
    """FB1's Add title… (a plain right-click on a plot whose title draws nothing) in each
    window that edits text in the figure. Windows whose title has automatic text offer it
    once that title is hidden, with Show title beside it."""
    page, plot, _edit, _shown_text = _build(kind, qtbot, tmp_path)
    widget = page._appearance_widget
    controller = figure_text_for(plot)
    # The views of Noise, Event waveforms and Subpopulations share one title, and the entry says
    # what a title typed there covers (the figure or figures on screen).
    add = {"noise": "Add title (this view)…", "events": "Add title (this view)…",
           "subpop": "Add title (both figures)…"}.get(kind, "Add title…")
    if plot.titleLabel.isVisible():
        controller.last_menu = None
        _click(plot, plot.titleLabel.sceneBoundingRect().center())
        _choose(controller.last_menu, "Hide title")
        assert not plot.titleLabel.isVisible()
        controller.last_menu = None
        _click(plot, _background(plot))
        assert _names(controller.last_menu)[:2] == [add, "Show title"]
        controller.last_menu.close()
    depth = widget._history.depth()
    _add_title(plot, "Pore 3: t_{1}")
    assert plot.titleLabel.isVisible()
    assert plot.titleLabel.item.toPlainText() == "Pore 3: t1"
    assert widget.state().title == "Pore 3: t_{1}"
    assert widget._history.depth() == (depth[0] + 1, 0)
    controller.last_menu = None
    _click(plot, _background(plot))
    assert add not in _names(controller.last_menu)
    controller.last_menu.close()
    assert widget.undo()
    assert not plot.titleLabel.isVisible()


def test_noise_add_title_on_a_view_says_what_it_titles(qapp, qtbot, tmp_path):
    """Noise's views share ONE title ("Title (this view)" in Appearance). "Add title…" on
    one view retitled every other view too, and nothing on the view said so; the entry now
    names what it covers (the view on screen, until the view changes)."""
    page, _plot, _edit, _shown = _build("noise", qtbot, tmp_path)
    psd = page._plot
    page.set_view("fft")
    QtWidgets.QApplication.processEvents()
    fft = page._fft_plot
    fft = fft.getPlotItem() if hasattr(fft, "getPlotItem") else fft
    controller = figure_text_for(fft)
    controller.last_menu = None
    _click(fft, fft.titleLabel.sceneBoundingRect().center())
    _choose(controller.last_menu, "Hide title")
    assert not fft.titleLabel.isVisible()
    controller.last_menu = None
    _click(fft, _background(fft))
    names = _names(controller.last_menu)
    assert names[:2] == ["Add title (this view)…", "Show title"], names
    _choose(controller.last_menu, "Add title (this view)…")
    editor = controller.editor
    assert editor.parentWidget().rect().contains(editor.geometry())
    _type_return(editor, "My FFT")
    assert fft.titleLabel.isVisible() and fft.titleLabel.item.toPlainText() == "My FFT"
    assert psd.titleLabel.item.toPlainText() == "My FFT"
