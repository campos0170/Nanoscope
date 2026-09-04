"""Regression tests for the unified analytical-window shell."""

from __future__ import annotations

from dataclasses import replace
import os

import nanoscope  # noqa: F401 -- Qt bootstrap must precede PySide
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from PySide6 import QtWidgets

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_explorer import (
    AnalysisExplorer,
    PAGE_DISTRIBUTIONS,
    PAGE_IV,
)
from nanoscope.gui.main_window import MainWindow


def test_shell_routes_tabs_and_refresh_to_only_the_current_page(qapp, qtbot):
    calls: list[str] = []
    shell = AnalysisExplorer()
    qtbot.addWidget(shell)
    distributions = QtWidgets.QMainWindow()
    iv = QtWidgets.QMainWindow()

    shell.show_page(
        PAGE_DISTRIBUTIONS, distributions,
        refresh=lambda: calls.append("refresh distributions"),
        activate=lambda: calls.append("activate distributions"),
    )
    shell.show_page(
        PAGE_IV, iv,
        refresh=lambda: calls.append("refresh iv"),
        activate=lambda: calls.append("activate iv"),
    )
    assert shell.page_ids() == (PAGE_DISTRIBUTIONS, PAGE_IV)
    assert shell.current_page_id() == PAGE_IV

    shell.refresh_current()
    assert calls == ["refresh iv"]

    # A tab-bar change (not a menu opener) runs the live-page activator.
    shell._tabs.setCurrentWidget(distributions)
    assert calls[-1] == "activate distributions"
    assert shell.current_page_id() == PAGE_DISTRIBUTIONS


def test_removing_current_page_does_not_activate_next_outgoing_page(qapp, qtbot):
    activated: list[str] = []
    shell = AnalysisExplorer()
    qtbot.addWidget(shell)
    distributions = QtWidgets.QWidget()
    iv = QtWidgets.QWidget()
    shell.show_page(
        PAGE_DISTRIBUTIONS, distributions,
        activate=lambda: activated.append("distributions"))
    shell.show_page(PAGE_IV, iv, activate=lambda: activated.append("iv"))
    assert shell.current_page_id() == PAGE_IV

    shell.remove_page(PAGE_IV)

    assert shell.current_page_id() == PAGE_DISTRIBUTIONS
    assert activated == []


def test_main_window_uses_one_shell_and_never_duplicates_pages(qapp, qtbot):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)

    window._open_plots()
    explorer = window._analysis_explorer
    plots = window._plots_window
    assert explorer is not None and explorer.isVisible()
    assert explorer.current_page_id() == PAGE_DISTRIBUTIONS

    window._open_iv()
    iv = window._iv_window
    assert window._analysis_explorer is explorer
    assert explorer.page_ids() == (PAGE_DISTRIBUTIONS, PAGE_IV)
    assert explorer.current_page_id() == PAGE_IV

    window._open_plots()
    window._open_iv()
    assert window._analysis_explorer is explorer
    assert window._plots_window is plots
    assert window._iv_window is iv
    assert explorer.page_count() == 2


def test_closing_shell_closes_pages_and_the_same_objects_can_reopen(qapp, qtbot):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window._open_plots()
    window._open_iv()
    explorer, plots, iv = (
        window._analysis_explorer, window._plots_window, window._iv_window)

    explorer.close()
    qapp.processEvents()
    assert not explorer.isVisible()
    assert not plots.isVisible()
    assert not iv.isVisible()

    window._open_plots()
    assert window._analysis_explorer is explorer
    assert window._plots_window is plots
    assert explorer.isVisible() and plots.isVisible()


def test_project_reset_does_not_leave_an_empty_explorer(qapp, qtbot):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window._open_iv()
    explorer = window._analysis_explorer
    assert explorer.isVisible() and explorer.page_count() == 1

    window._reset_analytical_workspace()
    qapp.processEvents()

    assert window._iv_window is None
    assert explorer.page_count() == 0
    assert not explorer.isVisible()


def test_project_reset_never_refreshes_the_next_outgoing_analysis_page(
        qapp, qtbot, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window._open_plots()
    window._open_iv()       # current; removing it would normally activate Distributions
    explorer = window._analysis_explorer
    refreshed: list[str] = []
    monkeypatch.setattr(window, "_refresh_plots",
                        lambda: refreshed.append("distributions"))
    monkeypatch.setattr(window._iv_window, "refresh",
                        lambda: refreshed.append("iv"))

    window._reset_analytical_workspace()
    qapp.processEvents()

    assert refreshed == []
    assert explorer.page_ids() == (PAGE_DISTRIBUTIONS,)


def _install_events(window: MainWindow, tmp_path) -> tuple[str, EventSet]:
    signal = np.full(4000, 100.0, dtype=np.float32)
    info = ChannelInfo(
        index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "linked.dat", format_name="synthetic",
        n_samples=signal.size, sample_rate_hz=10_000.0,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    document = TraceDocument(meta=meta, channels=(signal,))
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)

    frame = empty_df()
    for index, event_id, start in ((101, 7, 500), (205, 11, 1800)):
        frame.loc[index] = {
            "event_id": event_id, "recording": "linked.dat", "pore_id": "pore",
            "start_idx": start, "end_idx": start + 100,
            "t_start_s": start / 10_000.0, "dwell_s": 0.01,
            "baseline_pA": 100.0, "depth_pA": 20.0 + event_id,
            "frac_depth": 0.2, "ecd_pA_s": 0.2, "max_dev_pA": 25.0,
            "mean_dev_pA": 20.0, "rise_s": 0.001, "fall_s": 0.001,
            "snr": 10.0, "n_levels": 1, "filter_limited": False,
            "accepted": True, "flag": "",
        }
    frame = frame.astype({column: spec[0] for column, spec in SCHEMA.items()})
    events = EventSet(frame, {"recording": "linked.dat", "analyzed_duration_s": 0.4})
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events))
    window._on_selection(doc_id, 0)
    return doc_id, events


def test_plot_row_selection_reaches_table_and_trace_focus(qapp, qtbot, tmp_path):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    doc_id, _events = _install_events(window, tmp_path)
    window._open_plots()
    source = next(item for item in window._plots_window.sources()
                  if item.doc_id == doc_id and item.module_id == "detect.schmitt")

    # Stable event ids from two plotted points become the two selected rows; the last
    # point is the trace/table focus without collapsing the range.
    window._plots_window.eventRowsSelected.emit(source.key, (7, 11))
    assert window._analysis_panel._table.selected_event_ids() == [7, 11]
    assert window._analysis_panel._selected_id == 11
    assert "Selected 2 events from the plot" in window.statusBar().currentMessage()


def test_frozen_plot_cannot_navigate_into_a_new_detector_run_with_reused_ids(
        qapp, qtbot, tmp_path):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    doc_id, events = _install_events(window, tmp_path)
    window._open_plots()
    source = next(item for item in window._plots_window.sources()
                  if item.doc_id == doc_id and item.module_id == "detect.schmitt")

    # Replace the stored run without refreshing the plot, as a frozen/restored surface
    # can observe. Event ids are deliberately reused by the new detector run.
    newer_frame = events.df.copy(deep=True)
    newer_frame["start_idx"] += 1000
    newer = EventSet(newer_frame, dict(events.provenance))
    result = window._session.results.get(doc_id, 0, "detect.schmitt")
    window._session.results.put(replace(result, payload=newer))

    window._plots_window.eventRowsSelected.emit(source.key, (7,))

    assert "earlier detector run" in window.statusBar().currentMessage()
    assert window._analysis_panel._selected_id is None


def test_in_place_curation_keeps_plot_navigation_on_the_same_detector_generation(
        qapp, qtbot, tmp_path):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    doc_id, events = _install_events(window, tmp_path)
    window._open_plots()
    source = next(item for item in window._plots_window.sources()
                  if item.doc_id == doc_id and item.module_id == "detect.schmitt")

    # Curation deliberately mutates the EventSet frame in place; it must not look like
    # a detector rerun to the object-generation guard.
    events.df.loc[events.df["event_id"] == 7, "accepted"] = False
    window._plots_window.eventRowsSelected.emit(source.key, (11,))

    assert window._analysis_panel._selected_id == 11
    assert "Selected 1 event from the plot" in window.statusBar().currentMessage()
