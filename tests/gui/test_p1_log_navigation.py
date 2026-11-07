"""Format links reveal the real page control, without applying a numerical change."""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.event_plot_panel import DENSITY, ENSEMBLE, EventPlotWindow
from nanoscope.gui.format_inspector import ElementRef
from nanoscope.gui.plots_panel import HEATMAP, HISTOGRAM, PCA, SCATTER, PlotsWindow
from nanoscope.gui.psd_panel import (VIEW_ALLPOINTS, VIEW_FFT, VIEW_PSD, VIEW_SPECTROGRAM,
                                     PSDWindow)
from nanoscope.gui.voltage_dep_panel import VDepWindow

from . import p1_hosting as H


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _host(qtbot, key, view=None):
    session = Session()
    if key == "voltage_series":
        page = VDepWindow(session, defer_initial_refresh=True)
    elif key == "event_waveforms":
        page = EventPlotWindow(session)
        page.set_view(view or DENSITY)
    elif key == "noise":
        page = PSDWindow(session, defer_initial_refresh=True)
        page.set_view(view or VIEW_PSD)
    else:
        page = PlotsWindow(session)
        page.auto_plot = False
        page.set_plot_type(view or HISTOGRAM)
    explorer = H.hosted(qtbot, page, key)
    return explorer, page, explorer.inspector().pane(key)


def _axis(pane, side):
    pane.format_pane.refresh()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    refs = [ref for ref in pane.format_pane.elements() if ref.key == f"axis.{side}"]
    assert len(refs) == 1
    pane.format_pane.show_element(refs[0])
    QtWidgets.QApplication.processEvents()
    return pane.format_pane.current_panel()


@pytest.mark.parametrize("key,view,side,group,control", [
    ("voltage_series", None, "left", "plot", "_log_y"),
    ("event_waveforms", DENSITY, "bottom", "plot_map", "_logx"),
    ("event_waveforms", DENSITY, "left", "plot_map", "_logy"),
    ("noise", VIEW_PSD, "bottom", "view", "_loglog_check"),
    ("noise", VIEW_FFT, "left", "view", "_loglog_check"),
    ("distributions", HISTOGRAM, "bottom", "plot", "_hist_logx"),
    ("distributions", HEATMAP, "left", "plot", "_heatmap_logy"),
])
def test_axis_log_link_expands_the_owning_group_and_focuses_its_control(
        qtbot, key, view, side, group, control):
    explorer, page, pane = _host(qtbot, key, view)
    box = getattr(page, control)
    frame = pane.data_pane.group(group)
    frame.set_expanded(False)
    H.show_format(explorer)
    panel = _axis(pane, side)
    assert panel.log_link.isVisible() and not box.isVisible()
    before = box.isChecked(), explorer.adapter(key).history(), explorer.adapter(key).pending()
    changed = []
    box.toggled.connect(changed.append)
    H.click(panel.log_link)
    assert explorer.inspector().current_tab() == 0
    assert frame.is_expanded() and box.isVisible() and box.isEnabled() and box.hasFocus()
    assert pane.data_pane.viewport().rect().intersects(
        QtCore.QRect(box.mapTo(pane.data_pane.viewport(), QtCore.QPoint()), box.size()))
    assert (box.isChecked(), explorer.adapter(key).history(), explorer.adapter(key).pending()) == before
    assert changed == []


def test_a_disabled_log_control_is_revealed_without_receiving_focus_or_changing(qtbot):
    explorer, page, pane = _host(qtbot, "voltage_series")
    page._log_y.setEnabled(False)
    pane.data_pane.group("plot").set_expanded(False)
    H.show_format(explorer)
    panel = _axis(pane, "left")
    before = page._log_y.isChecked()
    H.click(panel.log_link)
    assert pane.data_pane.group("plot").is_expanded() and page._log_y.isVisible()
    assert not page._log_y.isEnabled() and not page._log_y.hasFocus()
    assert page._log_y.isChecked() == before


def test_spectrogram_log_link_reveals_its_existing_colour_map_control(qtbot):
    explorer, page, pane = _host(qtbot, "noise", VIEW_SPECTROGRAM)
    H.show_format(explorer)
    panel = _axis(pane, "left")
    assert panel.log_link.text() == "Change Log in Colour map"
    before = page._spec_logf_check.isChecked()
    H.click(panel.log_link)
    assert explorer.inspector().current_tab() == 1
    assert pane.format_pane.current_panel().isAncestorOf(page._spec_logf_check)
    assert page._spec_logf_check.isVisible() and page._spec_logf_check.hasFocus()
    assert page._spec_logf_check.isChecked() == before


@pytest.mark.parametrize("key,view,side", [
    ("voltage_series", None, "bottom"),
    ("event_waveforms", ENSEMBLE, "bottom"),
    ("noise", VIEW_SPECTROGRAM, "bottom"),
    ("noise", VIEW_ALLPOINTS, "left"),
])
def test_an_axis_without_an_editable_log_control_offers_no_navigation_link(qtbot, key, view, side):
    explorer, _page, pane = _host(qtbot, key, view)
    H.show_format(explorer)
    panel = _axis(pane, side)
    assert not panel.log_link.isVisibleTo(panel)


def test_distributions_colour_map_is_heatmap_only_and_survives_plot_type_changes(qtbot):
    explorer, page, pane = _host(qtbot, "distributions")
    H.show_format(explorer)
    ref = ElementRef("", "colour_map")
    assert ref not in pane.format_pane.elements()
    page.set_plot_type(HEATMAP)
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert ref in pane.format_pane.elements()
    pane.format_pane.show_element(ref)
    panel = pane.format_pane.current_panel()
    control = page._heatmap_cmap
    assert panel.isAncestorOf(control) and control.isVisible()
    chosen = control.currentText()
    for kind in (HISTOGRAM, SCATTER, PCA):
        page.set_plot_type(kind)
        for _ in range(3):
            QtWidgets.QApplication.processEvents()
        assert ref not in pane.format_pane.elements()
        assert pane.format_pane.current_panel() is not panel
    page.set_plot_type(HEATMAP)
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    pane.format_pane.show_element(ref)
    assert pane.format_pane.current_panel() is panel
    assert panel.isAncestorOf(control) and control.currentText() == chosen
