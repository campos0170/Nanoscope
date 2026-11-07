"""Generate Report figures draws a saved fit only where it describes the histogram (S10-LIN3).

THE GAP. The Plots window files a fit of any depth column -- ``frac_depth``, ``depth_pA``,
``max_dev_pA``, ``mean_dev_pA`` -- under ``fit.depth`` when it closes. The figure export asked
only ``stored_fit_problem`` (calibration and accepted count), so a Gaussian fit of ``depth_pA``
was drawn over the report's ``frac_depth`` (dI/I0) histogram: the status line said "depth: used
your saved fit (k=1)", the curve was flat zero and the annotation printed mu in pA on a fractional
axis. The export never asked which table a fit was made on either: the Fits tab does
(``judge_stored_fit``), and a Schmitt fit could be drawn on a CUSUM table whose count matched.

THE FIX. ``modules.report_fit_problem`` refuses a fit of another column than the figure draws,
and a fit that records another table; the histogram is refit afresh and the status line says
why, in the words of the existing refusals. A fit that records NO table (an earlier build's, the
Plots window's) is still judged by calibration and count alone and drawn, as the owner decided in
S8-RD1 (owner note #16).

These tests drive the real MainWindow on synthetic recordings under ``tmp_path``: Analysis ▸
Plots, the Fits tab's Fit button, the Results toolbar's Table selector and Generate Report
figures. They assert what the user reads: the status line and the figure's source-data CSV.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.analysis.modules import FIT_DETECTION_KEY, FIT_INPUT_COLUMN_KEY, fit_depth
from nanoscope.core.analysis import AnalysisResult
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow

from . import test_s8_rd1_fits_tab_follows_the_shown_table as R
from .test_editing_dependencies import _open, _plots, _press_fit

REFIT = "the figure was refit fresh at k=1"


@pytest.fixture
def window(qapp, qtbot, monkeypatch, tmp_path):
    """Every export dialog is answered with ``tmp_path / 'figs.pdf'``."""
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: tmp_path / "figs.pdf")
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


def _export(w: MainWindow, tmp_path: Path) -> tuple[str, pd.DataFrame]:
    """Analysis ▸ Generate Report figures…: the status line and the depth figure's CSV."""
    w._export_figures()
    QtWidgets.QApplication.processEvents()
    return w.statusBar().currentMessage(), pd.read_csv(tmp_path / "figs_depth.csv")


def _plots_fit(w: MainWindow, column: str) -> None:
    """Analysis ▸ Plots, a Gaussian mixture on ``column``, close the window."""
    plots = _plots(w)
    plots.set_histogram_column(column)
    plots.set_fit("gaussian-mixture")
    QtWidgets.QApplication.processEvents()
    plots._run_plot()
    plots.close()
    QtWidgets.QApplication.processEvents()


# ================================================================================================
# the column
# ================================================================================================

def test_a_plots_fit_of_depth_pA_is_not_drawn_on_the_dI_I0_report_figure(window, tmp_path):
    """The finder's route. The pA fit stays in the store; the figure is refit on frac_depth and
    the status line names the column, not "used your saved fit"."""
    doc_id = _open(window, tmp_path)
    _plots_fit(window, "depth_pA")
    stored = window._session.results.get(doc_id, 0, "fit.depth")
    assert stored is not None and stored.params["column"] == "depth_pA", "precondition"
    assert stored.payload.params["mu_1"] > 100, "precondition: a fit in pA"

    said, csv = _export(window, tmp_path)

    assert "depth: used your saved fit" not in said, said
    assert (f"depth: saved fit is of depth_pA, not the frac_depth this figure draws — {REFIT}"
            in said), said
    assert csv["bin_center"].max() < 1.0, "the report histogram is frac_depth"
    assert csv["fit_density"].max() > 1.0, (
        "the curve on the dI/I0 histogram is the pA fit, flat along zero")
    assert not np.allclose(csv["fit_density"],
                           stored.payload.predict(csv["bin_center"].to_numpy()))


def test_a_frac_depth_fit_pressed_in_the_fits_box_is_still_drawn(window, tmp_path):
    """The fit that describes the histogram is drawn as before, and the curve in the CSV is
    the stored one. (24 events: the mixture refuses k = 2 below 40.)"""
    doc_id = _open(window, tmp_path)
    _press_fit(window)
    stored = window._session.results.get(doc_id, 0, "fit.depth")
    assert stored is not None, "precondition: the Fit button filed a depth fit"
    assert stored.payload.meta[FIT_INPUT_COLUMN_KEY] == "frac_depth", "precondition"

    said, csv = _export(window, tmp_path)

    assert "depth: used your saved fit (k=1)" in said, said
    assert "dwell: used your saved fit (k=1)" in said, said
    np.testing.assert_allclose(csv["fit_density"].to_numpy(),
                               stored.payload.predict(csv["bin_center"].to_numpy()),
                               rtol=1e-12)


def test_a_fit_that_recorded_another_column_is_refused_whatever_its_table():
    """``input_column`` is read as well as the Plots window's ``params['column']``; a fit that
    recorded neither (an earlier build's Fits box) is judged as before."""
    from nanoscope.analysis.modules import report_fit_problem

    table = _table()
    fit = fit_depth(table, doc_id="doc", channel=0, n_boot=20,
                    detection_module_id=table.module_id)
    assert report_fit_problem(fit, table) == ""
    fit.payload.meta[FIT_INPUT_COLUMN_KEY] = "max_dev_pA"
    assert report_fit_problem(fit, table) == (
        "saved fit is of max_dev_pA, not the frac_depth this figure draws")
    del fit.payload.meta[FIT_INPUT_COLUMN_KEY]
    assert report_fit_problem(fit, table) == "", "no column recorded: the Fits box's own"


def test_a_dwell_fit_under_an_alias_of_dwell_s_is_drawn_and_another_duration_is_not():
    """The Plots window fits ``dwell`` and ``dwell_time_s`` as dwells and files them under
    ``fit.dwell`` (``stats.DWELL_QUANTITY_NAMES``): the same quantity the figure draws, so they
    are not refused. ``rise_s`` is seconds too, but not a dwell."""
    from nanoscope.analysis.modules import fit_dwell, report_fit_problem

    table = _table()
    fit = fit_dwell(table, doc_id="doc", channel=0, n_boot=20,
                    detection_module_id=table.module_id)
    for alias in ("dwell", "dwell_time_s"):
        fit.payload.meta[FIT_INPUT_COLUMN_KEY] = alias
        assert report_fit_problem(fit, table) == "", alias
    fit.payload.meta[FIT_INPUT_COLUMN_KEY] = "rise_s"
    assert report_fit_problem(fit, table) == (
        "saved fit is of rise_s, not the dwell_s this figure draws")


# ================================================================================================
# the table
# ================================================================================================

def test_a_fit_made_on_another_table_is_refused(window, qtbot, tmp_path):
    """Fit pressed on the CUSUM table, Schmitt on screen, Generate Report figures. The Fits tab
    shows no fit beside Schmitt; the export draws none either and names the table."""
    R._two_tables(window, qtbot, tmp_path)
    R._pick_table(window, qtbot, "CUSUM")
    R._press_fit(window, qtbot)
    R._pick_table(window, qtbot, "Schmitt")

    said, _ = _export(window, tmp_path)

    assert "used your saved fit" not in said, said
    for kind in ("dwell", "depth"):
        assert f"{kind}: saved fit belongs to the CUSUM table — {REFIT}" in said, said


def test_a_fit_of_another_table_at_the_same_count_is_refused():
    """The count the export trusted can coincide across tables; the recorded table cannot."""
    from nanoscope.analysis.modules import report_fit_problem, stored_fit_problem

    schmitt, cusum = _table("detect.schmitt"), _table("detect.cusum")
    fit = fit_depth(cusum, doc_id="doc", channel=0, n_boot=20,
                    detection_module_id=cusum.module_id)
    assert stored_fit_problem(fit, schmitt) == "", "precondition: count and calibration agree"
    assert report_fit_problem(fit, schmitt) == "saved fit belongs to the CUSUM table"
    assert report_fit_problem(fit, cusum) == ""


def test_a_plots_fit_of_frac_depth_records_no_table_and_is_still_drawn(window, tmp_path):
    """The skeptic's control. The Plots window names no table; the export keeps drawing such a
    fit, judged by calibration and count, as the owner decided (S8-RD1, owner note #16)."""
    doc_id = _open(window, tmp_path)
    _plots_fit(window, "frac_depth")
    stored = window._session.results.get(doc_id, 0, "fit.depth")
    assert stored is not None and stored.params["column"] == "frac_depth", "precondition"
    assert FIT_DETECTION_KEY not in stored.payload.meta, "precondition: no table recorded"

    said, csv = _export(window, tmp_path)

    assert "depth: used your saved fit (k=1)" in said, said
    np.testing.assert_allclose(csv["fit_density"].to_numpy(),
                               stored.payload.predict(csv["bin_center"].to_numpy()),
                               rtol=1e-12)


def test_a_fit_that_records_no_table_is_judged_by_the_two_guards_alone():
    """No table recorded: neither the column check (it is the right one) nor the table refuses
    it, and the export's old guards still do."""
    from nanoscope.analysis.modules import report_fit_problem

    table = _table()
    fit = fit_depth(table, doc_id="doc", channel=0, n_boot=20)
    assert FIT_DETECTION_KEY not in fit.payload.meta, "precondition"
    assert report_fit_problem(fit, table) == ""
    assert report_fit_problem(fit, _table(accepted=39)) == (
        "saved fit is stale (the accepted events changed since it ran)")


# ================================================================================================
# builders
# ================================================================================================

def _table(module_id: str = "detect.schmitt", *, accepted: int = 40) -> AnalysisResult:
    """A stored detection of 40 events, the first ``accepted`` of them accepted, as the S8-RD1
    unit tests build one."""
    rng = np.random.default_rng(0)
    df = empty_df()
    df["start_idx"] = np.arange(40, dtype=np.int64) * 1_000
    df["dwell_s"] = 3e-5 + rng.exponential(4e-4, 40)
    df["frac_depth"] = np.abs(rng.normal(0.3, 0.03, 40))
    df["accepted"] = np.arange(40) < accepted
    es = EventSet(df, {"min_dwell_s": 3e-5, "fs": 50_000.0, "analyzed_duration_s": 1.0})
    return AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id="doc", channel=0, params={},
        calibration_fingerprint=(1.0, 0.0), payload=es)
