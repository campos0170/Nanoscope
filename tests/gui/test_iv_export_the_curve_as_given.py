"""I-V: Export data writes the fit line the curve was GIVEN, however far the view is zoomed in.

The ``part == "fit"`` rows of the table behind Export data (and behind a Figure Library snapshot and
the figure bundle, which store the same table) were read from ``PlotDataItem.getData()``. That
answers in DISPLAY coordinates. This page has no log mode, so in every ordinary state those are the
numbers the line was given; but pyqtgraph clips the y values ``getData()`` answers once the view is
zoomed in far enough (its dynamic range limit: a line more than 6,000,000 times taller than the
view), and the rows then stopped being the fitted line. They are read from what the curve was given
now. Nothing else in the table, and nothing in any ordinary state, may change. Synthetic data;
offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -- Qt bootstrap before the panel imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import iv
from nanoscope.core.session import Session
from nanoscope.gui.iv_panel import IVWindow

from .test_iv import _current_doc, _ohmic_session

# How tall the view is, as a fraction of the fitted line's height. pyqtgraph clips once the line is
# more than 6,000,000 times the view's height, so a ten-thousandth is still the ordinary state and
# a billionth is far past the limit.
_ORDINARY_ZOOMS = [
    pytest.param(1.0, id="auto range"),
    pytest.param(1e-2, id="y a hundredth of the line"),
    pytest.param(1e-4, id="y a ten-thousandth of the line"),
]
_FAR_ZOOM = 1e-9


# -- builders ---------------------------------------------------------------------------------------

def _window(qtbot, tmp_path, *, G_nS=100.0, leak_pA=-10.0):
    """An I-V window on three recordings that lie on a known line, with its fit drawn."""
    window = IVWindow(_ohmic_session(tmp_path, G_nS=G_nS, leak_pA=leak_pA))
    qtbot.addWidget(window)
    window.show()
    assert window.fit() is not None and window._fit_item is not None
    return window


def _two_recordings(qtbot, tmp_path):
    """The least a fit can use: two recordings."""
    session = Session()
    for mV in (-300.0, -900.0):
        session.add_document(_current_doc(
            tmp_path, i_open_pA=80.0 * mV + 5.0, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
    window = IVWindow(session)
    qtbot.addWidget(window)
    window.show()
    assert window.fit() is not None and window.fit().n == 2
    return window


def _recordings_in_units(qtbot, units):
    """Four recordings handed to the window directly, one current unit each (the window converts
    to a shared unit when they differ, and draws its fit in that)."""
    window = IVWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(window)
    window._points = [iv.IVPoint(v, 100 * v + 2, units[i], "measured", str(i), f"run {i}")
                      for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    window._redraw()
    window.show()
    assert window.fit() is not None and window.fit().n == 4
    return window


_SESSIONS = [
    pytest.param(lambda qtbot, tmp_path: _window(qtbot, tmp_path), id="ohmic, leak below zero"),
    pytest.param(lambda qtbot, tmp_path: _window(qtbot, tmp_path, G_nS=50.0, leak_pA=25.0),
                 id="ohmic, leak above zero"),
    pytest.param(_two_recordings, id="two recordings"),
    pytest.param(lambda qtbot, tmp_path: _recordings_in_units(qtbot, ("pA", "nA", "pA", "nA")),
                 id="mixed current units"),
    pytest.param(lambda qtbot, tmp_path: _recordings_in_units(qtbot, ("", "", "", "")),
                 id="current units unknown"),
]


def _the_line(window):
    """The fit line, worked out from the fit alone: 200 points from the smaller of 0 V and the
    lowest plotted voltage to the larger of 0 V and the highest, with the fit evaluated there."""
    volts = np.array([p.v_volts for p in window.points()])
    xs = np.linspace(min(volts.min(), 0.0), max(volts.max(), 0.0), 200)
    return xs, window.fit().predict(xs)


def _set_y_height(window, fraction):
    """Make the view's y range ``fraction`` of the fit line's height, centred on the line."""
    _x, y = window._fit_item.getOriginalDataset()
    middle, height = float(y.min() + y.max()) / 2, float(y.max() - y.min()) * fraction
    window.plot_item().getViewBox().setYRange(middle, middle + height, padding=0)


def _pyqtgraph_answers_the_line_it_was_given(window):
    given = window._fit_item.getOriginalDataset()
    shown = window._fit_item.getData()
    return all(np.array_equal(a, b) for a, b in zip(given, shown))


def _fit_rows(frame):
    return frame[frame["part"] == "fit"]


def _assert_the_rows_are_the_line(rows, window, *, where):
    xs, ys = _the_line(window)
    assert len(rows) == len(xs) == 200, f"{where}: the fit line has the wrong number of rows"
    np.testing.assert_array_equal(
        rows["voltage_V"].to_numpy(), xs, err_msg=f"{where}: the voltages are not the line's")
    np.testing.assert_array_equal(
        rows["current"].to_numpy(), ys, err_msg=f"{where}: the currents are not the fitted line")


# -- the reference is what the curve was given -----------------------------------------------------

def test_the_curve_is_given_the_fit_evaluated_over_the_plotted_voltages(qapp, qtbot, tmp_path):
    """The reference the other tests use is not circular: it is worked out from the fit alone, and
    it is what pyqtgraph itself records as the data given to the curve."""
    window = _window(qtbot, tmp_path)
    xs, ys = _the_line(window)
    given_x, given_y = window._fit_item.getOriginalDataset()
    np.testing.assert_array_equal(given_x, xs)
    np.testing.assert_array_equal(given_y, ys)
    # The ohmic line of the fixture: 100 nS through -10 pA, so the fit reads it back.
    assert window.fit().conductance_nS == pytest.approx(100.0, rel=1e-3)
    assert ys[0] == pytest.approx(-100_010.0, rel=1e-6) and ys[-1] == pytest.approx(-10.0, abs=1e-3)


# -- ordinary states: the rows are the line, and nothing moved --------------------------------------

@pytest.mark.parametrize("make", _SESSIONS)
def test_fit_rows_are_the_line_for_every_kind_of_session(qapp, qtbot, tmp_path, make):
    window = make(qtbot, tmp_path)
    # An ordinary state is one where pyqtgraph still answers the line it was given: that is why the
    # export never showed the problem here, and why it must not change here.
    assert _pyqtgraph_answers_the_line_it_was_given(window), "this view is not an ordinary one"
    _assert_the_rows_are_the_line(_fit_rows(window._plot_data()), window, where="ordinary view")


@pytest.mark.parametrize("fraction", _ORDINARY_ZOOMS)
def test_fit_rows_are_the_line_in_every_ordinary_view(qapp, qtbot, tmp_path, fraction):
    window = _window(qtbot, tmp_path)
    if fraction != 1.0:
        _set_y_height(window, fraction)
    assert _pyqtgraph_answers_the_line_it_was_given(window), "this view is not an ordinary one"
    _assert_the_rows_are_the_line(_fit_rows(window._plot_data()), window, where="ordinary view")


def test_fit_rows_are_the_line_whatever_the_x_range_is(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    window.plot_item().getViewBox().setXRange(-0.30001, -0.3, padding=0)   # a sliver of the voltages
    assert _pyqtgraph_answers_the_line_it_was_given(window)
    _assert_the_rows_are_the_line(_fit_rows(window._plot_data()), window, where="x zoomed in")


# -- the view zoomed in far: pyqtgraph's own answer is no longer the line ---------------------------

def test_fit_rows_are_the_line_when_the_view_is_zoomed_in_far(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    _set_y_height(window, _FAR_ZOOM)
    # The state this test is about: pyqtgraph has clipped what getData() answers, so reading it
    # back would hand the file a different line from the fitted one.
    shown_y = window._fit_item.getData()[1]
    given_y = window._fit_item.getOriginalDataset()[1]
    assert not np.array_equal(shown_y, given_y), "the zoom did not make pyqtgraph clip getData()"
    assert np.abs(shown_y - given_y).max() > 1.0
    _assert_the_rows_are_the_line(_fit_rows(window._plot_data()), window, where="zoomed in far")


def test_export_data_file_holds_the_line_when_the_view_is_zoomed_in_far(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    _set_y_height(window, _FAR_ZOOM)
    assert not _pyqtgraph_answers_the_line_it_was_given(window)
    out = window.export_data(str(tmp_path / "iv.csv"))
    assert out is not None, "the export was refused"
    # read back as a spreadsheet would hold it, every digit kept
    frame = pd.read_csv(out, float_precision="round_trip", low_memory=False)
    _assert_the_rows_are_the_line(_fit_rows(frame), window, where="the exported file")


def test_nothing_else_in_the_table_depends_on_the_zoom(qapp, qtbot, tmp_path):
    """The point rows, every column and the row order are the same zoomed in far as in the
    ordinary view, and again once the view is back."""
    window = _window(qtbot, tmp_path)
    ordinary = window._plot_data().copy(deep=True)
    _set_y_height(window, _FAR_ZOOM)
    assert not _pyqtgraph_answers_the_line_it_was_given(window)
    pd.testing.assert_frame_equal(window._plot_data(), ordinary, check_exact=True)
    window.plot_item().autoRange()
    assert _pyqtgraph_answers_the_line_it_was_given(window)
    pd.testing.assert_frame_equal(window._plot_data(), ordinary, check_exact=True)


# -- no fit line, no fit rows -----------------------------------------------------------------------

def test_without_a_fit_line_there_are_no_fit_rows(qapp, qtbot, tmp_path):
    """One recording cannot fit a line: the page draws its point alone, and the table says so."""
    session = Session()
    session.add_document(_current_doc(tmp_path, i_open_pA=-20000.0, nominal_mV=-200.0, name="a.dat"))
    window = IVWindow(session)
    qtbot.addWidget(window)
    window.show()
    assert window.fit() is None and window._fit_item is None
    frame = window._plot_data()
    assert set(frame["part"]) == {"point"} and len(frame) == 1
    assert frame["current"].iloc[0] == -20000.0
