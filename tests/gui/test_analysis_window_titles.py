"""Standalone analysis windows use the same names as the Explorer catalog."""
import os

import nanoscope  # noqa: F401 - initializes the Qt plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib

import pytest
from PySide6 import QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui.analysis_catalog import ANALYSIS_LABELS


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("module,class_name,route,needs_session", [
    ("plots_panel", "PlotsWindow", "distributions", True),
    ("distribution_summary_panel", "DistributionSummaryWindow", "distribution_summaries", True),
    ("comparison_figures_panel", "ComparisonFiguresWindow", "comparison_figures", True),
    ("event_plot_panel", "EventPlotWindow", "event_waveforms", True),
    ("psd_panel", "PSDWindow", "noise", True),
    ("iv_panel", "IVWindow", "iv", True),
    ("voltage_dep_panel", "VDepWindow", "voltage_series", True),
    ("subpop_window", "SubpopReportWindow", "subpopulations", False),
])
def test_analysis_window_title_matches_catalog(app, module, class_name, route, needs_session):
    factory = getattr(importlib.import_module(f"nanoscope.gui.{module}"), class_name)
    window = factory(Session()) if needs_session else factory()
    try:
        assert window.windowTitle() == f"NanoScope — {ANALYSIS_LABELS[route]}"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
