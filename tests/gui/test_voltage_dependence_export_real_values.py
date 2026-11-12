"""Voltage dependence: Export data writes the fit line's real values, whatever Log y says.

With the Log y box ticked the plot is in log mode, and ``PlotDataItem.getData()`` answers in
DISPLAY coordinates, so the ``part == "fit"`` rows of Export data held the log10 of the fitted
value under ``metric_value`` (a fitted depth of 100 pA was written as 2.0) beside point rows that
held the real values. ``metric_value_display`` was worked out from that exponent (a dwell of
0.2 ms came out as -3698.97 ms). The figure bundle and the Figure Library snapshot store the same
table, so they carried the exponents too.

The rows must hold the fit evaluated at their voltages whatever the box says, and nothing else in
the export may change. Synthetic data only; offscreen.
"""
from __future__ import annotations

import io
import os
import zipfile
from types import SimpleNamespace

import nanoscope  # noqa: F401  -- Qt bootstrap before the panel imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import voltage_dep
from nanoscope.core.analysis_workspace import (
    VOLTAGE_DEPENDENCE_ARTIFACT_ID,
    capture_voltage_dependence,
    restore_voltage_dependence,
)
from nanoscope.core.figure_library import figures, make_figure
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.quantity_display import quantity_for, to_display
from nanoscope.core.session import Session
from nanoscope.gui.figure_bundle import FigureSnapshot, write_figure_bundle
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

from .test_voltage_dep import _LADDER_MV, _constant_df, _current_doc, _install_detection, _ladder_session

# (metric, fit model): a rate that is exponential in V, a depth and a dwell that are linear in V.
# The dwell is shown in ms, so its display column makes a wrong exponent obvious.
_FITS = [
    pytest.param("capture_rate_hz", "exponential", id="capture rate, exponential"),
    pytest.param("median_depth", "linear", id="depth, linear"),
    pytest.param("median_dwell_s", "linear", id="dwell, linear"),
]


# -- builders ---------------------------------------------------------------------------------------

def _window(qtbot, session, *, metric="capture_rate_hz", model="exponential", absolute=False):
    """A Voltage dependence window on ``session`` with ``model`` drawn over ``metric``."""
    window = VDepWindow(session)
    qtbot.addWidget(window)
    window.set_metric(metric)
    window.set_abs_voltage(absolute)
    window.set_fit(model)
    window.show()
    return window


def _depth_session(tmp_path, depths_pA, millivolts):
    """A session whose recordings sit at ``millivolts`` and whose median depth is ``depths_pA``."""
    session = Session()
    for mV, depth in zip(millivolts, depths_pA):
        doc_id = session.add_document(
            _current_doc(tmp_path, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
        _install_detection(session, doc_id, _constant_df(
            n=50, dwell_s=1e-3, depth_pA=depth, frac_depth=0.05), duration_s=1.0)
    return session


def _export(window, tmp_path, name):
    """Export data, read back as a person's spreadsheet would hold it (every digit kept)."""
    out = window.export_data(str(tmp_path / name))
    assert out is not None, "the export was refused"
    return pd.read_csv(out, float_precision="round_trip", low_memory=False)


def _part(frame, part):
    return frame[frame["part"] == part]


def _plotted_voltages(window, absolute):
    return np.array([abs(p.v_volts) if absolute else p.v_volts for p in window.points()])


def _assert_the_fit(rows, fit, metric, *, where):
    """``rows`` hold ``fit`` evaluated at their own voltages, and the display column follows."""
    volts = rows["voltage_V"].to_numpy()
    expected = voltage_dep.vdep_predict(fit, volts)
    np.testing.assert_allclose(rows["metric_value"].to_numpy(), expected, rtol=1e-12,
                               err_msg=f"{where}: the fit rows are not the fit's values")
    np.testing.assert_allclose(
        rows["metric_value_display"].to_numpy(),
        to_display(expected, quantity_for(metric)), rtol=1e-12,
        err_msg=f"{where}: the display column is not the fit's value in its display unit")


# -- the file holds the fit's real values -----------------------------------------------------------

@pytest.mark.parametrize("absolute", [False, True], ids=["signed V", "|V|"])
@pytest.mark.parametrize("metric, model", _FITS)
def test_fit_rows_are_the_fit_at_their_voltages_whatever_log_y_says(
        qapp, qtbot, tmp_path, metric, model, absolute):
    window = _window(qtbot, _ladder_session(tmp_path), metric=metric, model=model,
                     absolute=absolute)
    plotted = _plotted_voltages(window, absolute)
    frames = {}
    for log_y in (True, False):
        window.set_log_y(log_y)
        assert window.plot_item().getAxis("left").logMode is log_y
        fit = window.fit()
        assert fit is not None and fit["model"] == model
        rows = frames[log_y] = _part(_export(window, tmp_path, f"log_{log_y}.csv"), "fit")
        assert len(rows) > 1
        volts = rows["voltage_V"].to_numpy()
        # The line spans the plotted voltages (x was never the problem: it is not log here).
        assert volts[0] == pytest.approx(plotted.min()) and volts[-1] == pytest.approx(plotted.max())
        _assert_the_fit(rows, fit, metric, where=f"Log y {'ticked' if log_y else 'unticked'}")
    # One set of numbers: nothing in the fit rows depends on the box (but the box's own column).
    assert frames[True]["log_y"].all() and not frames[False]["log_y"].any()
    for column in ("voltage_V", "metric_value", "metric_value_display", "metric_value_display_unit",
                   "fit_model", "fit_params_json", "fit_r2", "fit_n"):
        pd.testing.assert_series_equal(
            frames[True][column].reset_index(drop=True),
            frames[False][column].reset_index(drop=True), check_exact=True, check_names=False,
            obj=f"fit rows, column {column}, ticked against unticked")


def test_the_fit_rows_agree_with_the_points_they_were_fitted_to(qapp, qtbot, tmp_path):
    """The ladder is exactly linear in V, so the line must pass through every point: a log10
    row (2.0 for a depth of 100 pA) is nowhere near the point rows beside it."""
    window = _window(qtbot, _ladder_session(tmp_path), metric="median_depth", model="linear")
    for log_y in (True, False):
        window.set_log_y(log_y)
        frame = _export(window, tmp_path, f"agree_{log_y}.csv")
        points, line = _part(frame, "point"), _part(frame, "fit")
        # The rows are a straight line in V, so reading it between two rows is exact.
        on_the_line = np.interp(points["voltage_V"], line["voltage_V"], line["metric_value"])
        np.testing.assert_allclose(
            on_the_line, points["metric_value"], rtol=1e-9,
            err_msg=f"Log y {log_y}: the fit rows do not run through the points beside them")


def test_point_rows_are_the_recordings_whatever_log_y_says(qapp, qtbot, tmp_path):
    window = _window(qtbot, _ladder_session(tmp_path), metric="median_depth", model="linear")
    points = window.points()
    assert len(points) == len(_LADDER_MV)
    frames = {}
    for log_y in (True, False):
        window.set_log_y(log_y)
        rows = frames[log_y] = _part(_export(window, tmp_path, f"points_{log_y}.csv"), "point")
        assert rows["source_recording"].tolist() == [p.recording for p in points]
        np.testing.assert_array_equal(rows["voltage_V"].to_numpy(), [p.v_volts for p in points])
        np.testing.assert_array_equal(rows["metric_value"].to_numpy(), [p.value for p in points])
        np.testing.assert_allclose(rows["metric_value_display"].to_numpy(),
                                   to_display(np.array([p.value for p in points]),
                                              quantity_for("median_depth")), rtol=1e-12)
    for column in ("source_recording", "source_doc_id", "voltage_V", "metric_value",
                   "metric_value_display", "voltage_source", "warning"):
        pd.testing.assert_series_equal(
            frames[True][column].reset_index(drop=True),
            frames[False][column].reset_index(drop=True), check_exact=True, check_names=False,
            obj=f"point rows, column {column}, ticked against unticked")


def test_a_point_left_out_of_a_log_plot_keeps_its_real_value_and_the_fit_rows_stay_real(
        qapp, qtbot, tmp_path):
    """A depth of 0 has no logarithm, so Log y leaves it off the plot and out of the fit. Its row
    says so and keeps the number; the fit rows are the fit over the points that are drawn."""
    session = _depth_session(tmp_path, (20.0, 0.0, 70.0, 100.0), _LADDER_MV)
    window = _window(qtbot, session, metric="median_depth", model="linear")
    window.set_log_y(True)
    frame = _export(window, tmp_path, "left_out.csv")
    left_out = _part(frame, "excluded_from_view")
    assert len(left_out) == 1
    assert left_out["metric_value"].iloc[0] == 0.0
    assert left_out["warning"].iloc[0] == "non-positive for log y"
    assert len(_part(frame, "point")) == 3
    fit = window.fit()
    assert fit is not None and fit["n"] == 3
    _assert_the_fit(_part(frame, "fit"), fit, "median_depth", where="Log y ticked, a point left out")


def test_a_fit_that_runs_below_zero_exports_its_real_values_in_both_states(qapp, qtbot, tmp_path):
    """A straight line through a hockey stick is negative at the low end: Log y cannot draw
    that stretch, but the rows hold what the fit says there, as they do with the box unticked."""
    millivolts = (-100.0, -200.0, -300.0, -400.0, -500.0)
    window = _window(qtbot, _depth_session(tmp_path, (0.1, 0.1, 0.1, 0.1, 100.0), millivolts),
                     metric="median_depth", model="linear")
    frames = {}
    for log_y in (True, False):
        window.set_log_y(log_y)
        fit = window.fit()
        assert fit is not None and fit["n"] == len(millivolts)
        rows = frames[log_y] = _part(_export(window, tmp_path, f"below_zero_{log_y}.csv"), "fit")
        assert (rows["metric_value"] < 0).any(), "the scenario: the line goes below zero"
        _assert_the_fit(rows, fit, "median_depth", where=f"Log y {log_y}")
    np.testing.assert_array_equal(frames[True]["metric_value"].to_numpy(),
                                  frames[False]["metric_value"].to_numpy())


# -- the figure bundle and the Figure Library snapshot hold the same table --------------------------

def _capture(window):
    """The figure as Save to Figure Library and Save figure bundle freeze it: the page's own
    ``_plot_data`` table, no analysis re-run (the route ``test_figure_capture_pages`` uses)."""
    owner = SimpleNamespace(
        _session=window._session,
        _figure_workspace=None,
        _figure_page=lambda _page_id: (VOLTAGE_DEPENDENCE_ARTIFACT_ID, window),
        _workspace_kwarg=MainWindow._workspace_kwarg,
        _analysis_explorer_status=lambda _page_id, _panel: ("CURRENT", "current"))
    return MainWindow._capture_figure(owner, "voltage")


def test_the_bundle_and_the_library_snapshot_hold_the_same_real_fit_rows(qapp, qtbot, tmp_path):
    metric = "median_dwell_s"
    window = _window(qtbot, _ladder_session(tmp_path), metric=metric, model="linear")
    tables = {}
    for log_y in (True, False):
        window.set_log_y(log_y)
        qtbot.wait(10)
        fit = window.fit()
        exported = _export(window, tmp_path, f"export_{log_y}.csv")

        analysis, snapshot, source, live, status = _capture(window)
        # The Figure Library: the snapshot goes into the project and comes back out of it.
        metadata, blobs = snapshot.to_storage()
        entry = make_figure(name="vdep", page_id="voltage", source_label=source, analysis=analysis,
                            snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=live,
                            capture_status=status)
        project = save_project(tmp_path / f"library_{log_y}.nanoscope", ProjectState(
            workspace=AnalyticalWorkspace(artifacts=[entry])))
        back = figures(load_project(project).workspace)[0]
        library = FigureSnapshot.from_storage(back.payload["snapshot"], back.blobs).dataframe()
        # The bundle: data.csv (17-digit floats) and data.parquet (native) of the same table.
        bundle = write_figure_bundle(tmp_path / f"bundle_{log_y}.zip", snapshot)
        with zipfile.ZipFile(bundle) as archive:
            csv = pd.read_csv(io.BytesIO(archive.read("data.csv")), float_precision="round_trip",
                              low_memory=False)
            parquet = pd.read_parquet(io.BytesIO(archive.read("data.parquet")),
                                      columns=["voltage_V", "metric_value", "metric_value_display"])
        is_fit = (csv["part"] == "fit").to_numpy()
        assert is_fit.any()
        parquet_fit = parquet[is_fit]

        sources = {"Export data": _part(exported, "fit"),
                   "snapshot": _part(snapshot.dataframe(), "fit"),
                   "library entry": _part(library, "fit"),
                   "bundle data.csv": _part(csv, "fit"),
                   "bundle data.parquet": parquet_fit}
        for name, rows in sources.items():
            _assert_the_fit(rows, fit, metric, where=f"{name}, Log y {log_y}")
        tables[log_y] = sources

    for name in tables[True]:
        for column in ("voltage_V", "metric_value", "metric_value_display"):
            np.testing.assert_array_equal(
                tables[True][name][column].to_numpy(), tables[False][name][column].to_numpy(),
                err_msg=f"{name}: {column} depends on the Log y box")


# -- a reopened project ------------------------------------------------------------------------------

def test_a_reopened_project_exports_the_real_fit_rows(qapp, qtbot, tmp_path):
    """The saved result is drawn again without refitting (``_render_workspace_snapshot``), on the
    same fit item, in log mode when the project was saved with Log y ticked."""
    window = _window(qtbot, _ladder_session(tmp_path), metric="median_dwell_s", model="linear")
    window.set_log_y(True)
    live = _part(_export(window, tmp_path, "live.csv"), "fit")
    saved = save_project(tmp_path / "vdep.nanoscope", ProjectState(
        workspace=AnalyticalWorkspace(artifacts=[capture_voltage_dependence(window)])))
    artifact, = load_project(saved).workspace.artifacts

    reopened = VDepWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(reopened)
    report = restore_voltage_dependence(reopened, artifact)
    assert report.skipped == ()
    assert reopened._log_y.isChecked() and reopened.plot_item().getAxis("left").logMode
    fit = reopened.fit()
    assert fit is not None
    rows = _part(_export(reopened, tmp_path, "reopened.csv"), "fit")
    _assert_the_fit(rows, fit, "median_dwell_s", where="reopened project, Log y ticked")
    np.testing.assert_array_equal(rows["voltage_V"].to_numpy(), live["voltage_V"].to_numpy())
    for column in ("metric_value", "metric_value_display"):
        np.testing.assert_allclose(rows[column].to_numpy(), live[column].to_numpy(), rtol=1e-12,
                                   err_msg=f"{column} changed across Save Project and reopen")
