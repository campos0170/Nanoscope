"""Subpopulations: the waveform rows of the figure's table are the curves as they were GIVEN,
however far the view is zoomed in.

The ``part == "waveform"`` rows (the table a Snapshot, a Figure Library entry and the figure bundle
store) were read from ``PlotDataItem.getData()``. That answers in DISPLAY coordinates. These
figures are drawn on linear axes, so in every ordinary state those are the numbers the curve was
given; but pyqtgraph clips the y values ``getData()`` answers once the view is zoomed in far enough
(its dynamic range limit: a curve more than 6,000,000 times taller than the view), and the rows
then stopped being the median waveform. They are read from what the curve was given now. Nothing
else in the table, and nothing in any ordinary state, may change. A real report; offscreen.
"""
from __future__ import annotations

import copy
import os

import nanoscope  # noqa: F401  -- Qt bootstrap before the panel imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.gui.subpop_window import SubpopReportWindow

from .test_subpop_views import report  # noqa: F401  -- one small real report fixture

# How tall the view is, as a fraction of the shortest waveform's height. pyqtgraph clips once a
# curve is more than 6,000,000 times the view's height, so 1e-4 is still the ordinary state and
# 1e-9 is far past the limit for every curve.
_ORDINARY_ZOOMS = [
    pytest.param(1e-2, id="y a hundredth of the waveform"),
    pytest.param(1e-4, id="y a ten-thousandth of the waveform"),
]
_FAR_ZOOM = 1e-9


# -- builders ---------------------------------------------------------------------------------------

def _page(qtbot, report):
    """The report's page, with the waveform figure framed on its data as the window frames it."""
    page = SubpopReportWindow()
    qtbot.addWidget(page)
    page.set_report(report)
    page.show()
    page.waveform_view()._plot.getPlotItem().autoRange()
    return page


def _curves(page):
    """The median-waveform curves the page drew (one per class whose average exists)."""
    return [item for item in page.waveform_view()._plot.getPlotItem().items
            if hasattr(item, "_nanoscope_waveform_series")]


def _set_y_height(page, fraction):
    """Make the waveform plot's y range ``fraction`` of the shortest curve's height, centred where
    every curve has data (a view that overlaps no data is never clipped)."""
    arrays = [curve.getOriginalDataset()[1] for curve in _curves(page)]
    lowest_top = min(float(np.nanmax(y)) for y in arrays)
    highest_bottom = max(float(np.nanmin(y)) for y in arrays)
    assert highest_bottom < lowest_top, "the curves share no current range to zoom into"
    middle = (highest_bottom + lowest_top) / 2
    height = min(float(np.nanmax(y) - np.nanmin(y)) for y in arrays) * fraction
    page.waveform_view()._plot.getPlotItem().getViewBox().setYRange(
        middle, middle + height, padding=0)


def _pyqtgraph_answers_what_the_curves_were_given(page):
    return all(np.array_equal(a, b, equal_nan=True)
               for curve in _curves(page)
               for a, b in zip(curve.getOriginalDataset(), curve.getData()))


def _waveform_rows(frame):
    return frame[frame["part"] == "waveform"]


def _assert_the_rows_are_the_waveforms(frame, report, *, where):
    """Every drawn class has its own rows, and they are ``report.waveforms`` as it stands."""
    table = report.waveforms
    drawn = [column for column in table.columns
             if column != "T" and np.any(np.isfinite(table[column].to_numpy(dtype=float)))]
    assert drawn, "the report draws no class at all"
    rows = _waveform_rows(frame)
    assert list(dict.fromkeys(rows["series"])) == drawn, f"{where}: the classes are not the drawn ones"
    counts = table.attrs.get("n_events", {}) or {}
    for column in drawn:
        samples = rows[rows["series"] == column]
        assert len(samples) == len(table), f"{where}: {column} has the wrong number of rows"
        np.testing.assert_array_equal(samples["point_index"].to_numpy(), np.arange(len(table)))
        np.testing.assert_array_equal(
            samples["x"].to_numpy(), table["T"].to_numpy(dtype=float),
            err_msg=f"{where}: {column}: the normalised times are not the table's")
        np.testing.assert_array_equal(
            samples["y"].to_numpy(), table[column].to_numpy(dtype=float),
            err_msg=f"{where}: {column}: the currents are not the median waveform")
        assert set(samples["n_events"]) == {int(counts.get(column, 0))}


def _with_a_gap(report):
    """A copy of ``report`` whose first class has no average over part of the event (NaN there)."""
    altered = copy.deepcopy(report)
    column = next(c for c in altered.waveforms.columns if c != "T")
    values = altered.waveforms[column].to_numpy(dtype=float).copy()
    values[60:80] = np.nan
    altered.waveforms[column] = values
    assert int(np.isnan(altered.waveforms[column].to_numpy(dtype=float)).sum()) == 20
    return altered


# -- ordinary states: the rows are the waveforms, and nothing moved ---------------------------------

def test_waveform_rows_are_the_median_waveforms_in_the_ordinary_view(qapp, qtbot, report):
    page = _page(qtbot, report)
    assert len(_curves(page)) >= 2
    assert _pyqtgraph_answers_what_the_curves_were_given(page)
    _assert_the_rows_are_the_waveforms(page._plot_data(), report, where="ordinary view")


@pytest.mark.parametrize("fraction", _ORDINARY_ZOOMS)
def test_waveform_rows_are_the_waveforms_in_every_ordinary_view(qapp, qtbot, report, fraction):
    page = _page(qtbot, report)
    _set_y_height(page, fraction)
    # An ordinary state is one where pyqtgraph still answers what the curve was given: that is why
    # the export never showed the problem here, and why it must not change here.
    assert _pyqtgraph_answers_what_the_curves_were_given(page), "this view is not an ordinary one"
    _assert_the_rows_are_the_waveforms(page._plot_data(), report, where="ordinary view")


def test_waveform_rows_are_the_waveforms_whatever_the_x_range_is(qapp, qtbot, report):
    page = _page(qtbot, report)
    page.waveform_view()._plot.getPlotItem().getViewBox().setXRange(0.4, 0.4001, padding=0)
    assert _pyqtgraph_answers_what_the_curves_were_given(page)
    _assert_the_rows_are_the_waveforms(page._plot_data(), report, where="x zoomed in")


def test_a_gap_in_a_waveform_is_exported_as_a_gap_zoomed_in_or_not(qapp, qtbot, report):
    """NaN where a class has no average is part of what the curve was given."""
    altered = _with_a_gap(report)
    page = _page(qtbot, altered)
    _assert_the_rows_are_the_waveforms(page._plot_data(), altered, where="ordinary view")
    _set_y_height(page, _FAR_ZOOM)
    _assert_the_rows_are_the_waveforms(page._plot_data(), altered, where="zoomed in far")


def test_a_class_that_is_not_drawn_has_no_waveform_rows(qapp, qtbot, report):
    """A class with no average anywhere is not a curve (the caption names it), so it is not a row."""
    altered = copy.deepcopy(report)
    altered.waveforms[[c for c in altered.waveforms.columns if c != "T"][0]] = np.nan
    page = _page(qtbot, altered)
    assert len(_curves(page)) == len(report.waveforms.columns) - 2    # T and the class left out
    assert _pyqtgraph_answers_what_the_curves_were_given(page)
    _assert_the_rows_are_the_waveforms(page._plot_data(), altered, where="ordinary view")


# -- the view zoomed in far: pyqtgraph's own answer is no longer what the curves were given ---------

def test_waveform_rows_are_the_waveforms_when_the_view_is_zoomed_in_far(qapp, qtbot, report):
    page = _page(qtbot, report)
    _set_y_height(page, _FAR_ZOOM)
    # The state this test is about: pyqtgraph has clipped what getData() answers for every curve,
    # so reading it back would hand the table different waveforms from the ones drawn.
    for curve in _curves(page):
        shown_y = curve.getData()[1]
        given_y = curve.getOriginalDataset()[1]
        assert not np.array_equal(shown_y, given_y), "the zoom did not make pyqtgraph clip getData()"
    _assert_the_rows_are_the_waveforms(page._plot_data(), report, where="zoomed in far")


def test_nothing_else_in_the_table_depends_on_the_zoom(qapp, qtbot, report):
    """The embedding rows, every column and the row order are the same zoomed in far as in the
    ordinary view, and again once the view is back."""
    page = _page(qtbot, report)
    ordinary = page._plot_data().copy(deep=True)
    assert {"embedding", "waveform"} <= set(ordinary["part"])
    _set_y_height(page, _FAR_ZOOM)
    assert not _pyqtgraph_answers_what_the_curves_were_given(page)
    pd.testing.assert_frame_equal(page._plot_data(), ordinary, check_exact=True)
    page.waveform_view()._plot.getPlotItem().autoRange()
    assert _pyqtgraph_answers_what_the_curves_were_given(page)
    pd.testing.assert_frame_equal(page._plot_data(), ordinary, check_exact=True)
