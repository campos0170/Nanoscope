"""Population-plots organization and plot-to-event handoff."""
from __future__ import annotations

import os

import nanoscope  # noqa: F401 -- Qt bootstrap before PySide
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow


def _source(key: str, frame: pd.DataFrame) -> PlotSource:
    return PlotSource(
        key=key, label=key, df=frame, provenance={}, colour="#0072B2",
        doc_id=f"doc-{key}", channel=0, module_id="detect.threshold",
        calibration_fingerprint=(1.0, 0.0),
    )


class _Click:
    def __init__(self, modifiers=QtCore.Qt.NoModifier) -> None:
        self._modifiers = modifiers

    def modifiers(self):
        return self._modifiers


def test_compact_workflow_keeps_navigation_run_and_export_reachable(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1120, 720)
    w.show()
    qapp.processEvents()

    assert [w._section_combo.itemText(i) for i in range(w._section_combo.count())] == [
        "Data", "Analysis", "Appearance", "Export"]
    assert w._workflow_toolbar.isVisible()
    assert w._run_action.text() == "Plot / Apply"
    assert w._export_menu_btn.isVisible()
    assert [action.text() for action in w._export_menu.actions()] == [
        "Save image…", "Export plot data…", "Export configuration JSON…"]

    # The persistent strip does not replace the established, resizable controls column.
    assert isinstance(w.centralWidget(), QtWidgets.QSplitter)
    assert isinstance(w.centralWidget().widget(0), QtWidgets.QScrollArea)
    assert w.centralWidget().widget(0).widget() is w._controls_host

    assert w._source_group.isVisible() and w._plot_group.isVisible()
    assert not w._style_group.isVisible() and not w._config_group.isVisible()

    w._show_control_section("Appearance")
    qapp.processEvents()
    assert w._style_group.isVisible()
    assert not w._source_group.isVisible() and not w._cluster_group.isVisible()
    assert w._run_action.isVisible() and w._export_menu_btn.isVisible()
    assert w._section_combo.accessibleName()
    assert w._plot_widget.accessibleDescription()


def test_result_summary_is_concise_and_full_details_are_collapsible(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.show()
    qapp.processEvents()

    report = (
        "Scatter depth_pA vs dwell_s   n = 91\n"
        "! Detector bandwidths are NOT COMPARABLE across these sources.\n"
        "method: a deliberately long provenance record stays here in full"
    )
    w._set_details(report)

    assert w._result_summary.text() == "Scatter depth_pA vs dwell_s   n = 91"
    assert "NOT COMPARABLE" in w._result_warning.text()
    assert w._result_warning.isVisible()
    assert w.details_text() == report
    assert not w._details.isVisible()

    w._details_box.setChecked(True)
    qapp.processEvents()
    assert w._details.isVisible()
    assert w._details.accessibleName() == "Full plot details"


def test_uncertified_baseline_warning_stays_visible_when_details_are_collapsed(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.show()
    qapp.processEvents()

    warning = (
        "The open pore here is NOT CERTIFIED, only chosen. Compare the baseline "
        "with an independently known open-pore current."
    )
    w._set_details(f"Histogram dwell_s   n = 91\n{warning}")

    assert w._result_warning.isVisible()
    assert w._result_warning.text() == warning
    assert not w._details.isVisible()


def test_scatter_click_emits_exact_filtered_source_rows_and_supports_addition(qapp, qtbot):
    frame = pd.DataFrame({
        "dwell_s": [1.0, -2.0, 3.0, 4.0, 5.0],
        "depth_pA": [10.0, 20.0, np.nan, 40.0, 50.0],
        "accepted": [True, True, True, True, False],
    }, index=pd.Index(["first", "log-invalid", "nonfinite", "second", "rejected"],
                    name="event_id"))
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("apo-700", frame)])
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_log(True, False)

    points = list(w._scatter_item.points())
    assert [point.data() for point in points] == ["first", "second"]
    received = []
    w.eventRowsSelected.connect(lambda source, rows: received.append((source, rows)))

    w._on_event_points_clicked("apo-700", w._scatter_item, [points[0]], _Click())
    w._on_event_points_clicked(
        "apo-700", w._scatter_item, [points[1]], _Click(QtCore.Qt.ControlModifier))
    assert received[-1] == ("apo-700", ("first", "second"))

    # Additive clicks toggle an already-selected row without changing its identity.
    w._on_event_points_clicked(
        "apo-700", w._scatter_item, [points[0]], _Click(QtCore.Qt.ControlModifier))
    assert received[-1] == ("apo-700", ("second",))


def test_pca_click_uses_kept_index_not_positional_rows(qapp, qtbot):
    frame = pd.DataFrame({
        "dwell_s": [1.0, 2.0, np.nan, 4.0],
        "depth_pA": [10.0, 11.0, 12.0, 14.0],
        "accepted": [True, True, True, True],
    }, index=pd.Index([101, 205, 999, 410], name="event_id"))
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("apo-700", frame)])
    w.set_feature_selection(["dwell_s", "depth_pA"])
    result = w.run_pca()

    assert result is not None
    assert list(result.kept_index) == [101, 205, 410]
    assert [point.data() for point in w._scatter_item.points()] == [101, 205, 410]
    received = []
    w.eventRowsSelected.connect(lambda source, rows: received.append((source, rows)))
    point = w._scatter_item.points()[2]
    w._on_event_points_clicked("apo-700", w._scatter_item, [point], _Click())
    assert received == [("apo-700", (410,))]


def test_detector_points_use_event_id_when_dataframe_index_is_not_unique(qapp, qtbot):
    frame = pd.DataFrame({
        "event_id": [17, 23, 41],
        "dwell_s": [1.0, 2.0, 3.0],
        "depth_pA": [10.0, 20.0, 30.0],
        "accepted": [True, True, True],
    }, index=pd.Index([0.0, 0.0, np.nan]))
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("duplicates", frame)])
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")

    points = list(w._scatter_item.points())
    assert [point.data() for point in points] == [17, 23, 41]
    received = []
    w.eventRowsSelected.connect(lambda source, rows: received.append((source, rows)))
    w._on_event_points_clicked("duplicates", w._scatter_item, [points[1]], _Click())
    w._on_event_points_clicked("duplicates", w._scatter_item, [points[2]], _Click())
    assert received == [("duplicates", (23,)), ("duplicates", (41,))]


def test_additive_point_selection_moves_between_sources_instead_of_leaving_stale_ring(
        qapp, qtbot):
    first = pd.DataFrame({
        "event_id": [1], "dwell_s": [1.0], "depth_pA": [10.0], "accepted": [True],
    })
    second = pd.DataFrame({
        "event_id": [2], "dwell_s": [2.0], "depth_pA": [20.0], "accepted": [True],
    })
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("A", first), _source("B", second)])
    w.set_sources(["A", "B"])
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    a = w._event_point_items["A"][0].points()[0]
    b = w._event_point_items["B"][0].points()[0]

    received = []
    w.eventRowsSelected.connect(lambda source, rows: received.append((source, rows)))
    w._on_event_points_clicked("A", w._event_point_items["A"][0], [a], _Click())
    w._on_event_points_clicked(
        "B", w._event_point_items["B"][0], [b], _Click(QtCore.Qt.ControlModifier))

    assert w._selected_event_rows == {"B": [2]}
    assert received[-1] == ("B", (2,))
    assert a.pen().style() == QtCore.Qt.NoPen
    assert b.pen().style() != QtCore.Qt.NoPen
