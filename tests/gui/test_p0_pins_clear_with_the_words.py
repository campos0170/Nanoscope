"""Phase 0, group G5: a pinned axis range clears with the words when its axis changes quantity.

The owner's decision: when a view's quantity changes, the pin on each axis whose quantity moved
(the range and the major tick spacing typed in Axis properties) goes back to automatic in the same
Undo step and the same status sentence as the typed words (``gui.quantity_reset``): *Title, Y label
and Y range reset: the plot now shows median depth.* An axis whose quantity did not move keeps its
pin, and so does another view's unless its axis shows the same quantity (Distribution summaries'
two Type pairs share their measurement).

Every gesture is a real one. A range is pinned by right-clicking the axis label, choosing
**Axis properties…**, unticking Auto and typing into the popover (``_pin``); the quantity is changed
with the real control (the combo's popup, the tab bar, a check box, the Generate button); Undo is
the Undo key on the page's window; a save is a real ``_write_project`` reopened by
``_open_project``. What is asserted is what is drawn (the range the view shows, whether the axis is
auto-ranging again, what Axis properties would say) and not only the state behind it. Synthetic
data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisPin, TextTarget, figure_text_for
from nanoscope.gui.key_sequences import redo_sequences
from nanoscope.gui.plot_appearance import PlotAppearance

from .test_p0_reset_on_quantity_change import (  # noqa: F401  (fixtures are used by name)
    _boxes, _clear_status, _click_button, _drawn_title, _explorer_of, _generate, _label, _pick,
    _said, _section, _series_window, _standing_line, _type_comparison_words, _type_in_box,
    _user_picks, comparison, dist, events, noise, subpop, subpop_report, summaries, vdep)
from .test_plot_window_routes import (
    _add_title, _choose, _click, _edit_double_click, _install_series, _main_window, _names,
    _press_undo, _restored_main_window, window_handle)

L = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# -- real gestures --------------------------------------------------------------------------------

def _host(plot):
    return figure_text_for(plot).host


def _axis(plot, side):
    """What Axis properties would show for ``side``: the pin (``range``, None is Auto) and the
    range the view draws, both in the units the axis shows."""
    return _host(plot).axis_state(side)


def _pin(plot, side, lo, hi, major=None, log=None):
    """Right-click the axis label, **Axis properties…**, untick Auto, type the range (and a
    major tick spacing), OK. ``lo`` None leaves the range on Auto. ``log`` True or False ticks or
    unticks Log scale in the same dialog (None leaves it as it is). The offscreen screen is
    800 px square and the popover opens under the label, so the window is moved to keep the
    whole popover on it for the click, and put back after."""
    controller = figure_text_for(plot)
    view = plot.scene().views()[0]
    window = view.window()
    home = window.pos()
    rect = controller._anchor_rect(TextTarget(item=plot.getAxis(side)))
    below = view.mapToGlobal(rect.bottomLeft())
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    window.move(home.x() - max(0, below.x() + 520 - screen.right()),
                home.y() - max(0, below.y() + 280 - screen.bottom()))
    QtWidgets.QApplication.processEvents()
    try:
        controller.last_menu = None
        _click(plot, plot.getAxis(side).label.sceneBoundingRect().center())
        menu = controller.last_menu
        assert menu is not None and "Axis properties…" in _names(menu), menu and _names(menu)
        _choose(menu, "Axis properties…")
        popover = controller.popover
        assert popover is not None and popover.side == side
        if lo is None:
            if not popover.auto.isChecked():
                QtTest.QTest.mouseClick(popover.auto, L)
        else:
            if popover.auto.isChecked():
                QtTest.QTest.mouseClick(popover.auto, L)
            for box, value in ((popover.lo, lo), (popover.hi, hi)):
                box.selectAll()
                QtTest.QTest.keyClicks(box, f"{value:g}")
        if log is not None and popover.log.isChecked() != log:
            # The box is as wide as the form, but only its indicator and its text take a click.
            indicator = QtCore.QPoint(8, popover.log.height() // 2)
            QtTest.QTest.mouseClick(popover.log, L, NOMOD, indicator)
            assert popover.log.isChecked() == log
        if major is not None:
            popover.major.selectAll()
            QtTest.QTest.keyClicks(popover.major, f"{major:g}")
        ok = popover.findChild(QtWidgets.QDialogButtonBox).button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        QtTest.QTest.mouseClick(ok, L)
        QtWidgets.QApplication.processEvents()
        assert not popover.isVisible(), "OK was refused: " + popover.error.text()
    finally:
        window.move(home)
        QtWidgets.QApplication.processEvents()


def _pin_inside(plot, side, *, span=(0.2, 0.7), major=None):
    """Pin ``side`` to a window inside what it shows now (3 significant figures, as a person
    types it); returns the range typed."""
    a, b = _axis(plot, side).shown_range
    rng = (float(f"{a + span[0] * (b - a):.3g}"), float(f"{a + span[1] * (b - a):.3g}"))
    _pin(plot, side, *rng, major=major)
    return rng


def _press_redo(page):
    """The platform's Redo key (⇧⌘Z / Ctrl+Y), pressed on the page's window."""
    page.window().activateWindow()
    QtWidgets.QApplication.processEvents()
    combo = redo_sequences()[0][0]
    QtTest.QTest.keyClick(window_handle(page), combo.key(), combo.keyboardModifiers())
    QtWidgets.QApplication.processEvents()


def _reset_view(page):
    """Click the page's Reset view button: the zoom goes back to the page's own framing, which is
    the pin of an axis that still has one (``_nanoscope_axis_pins``) and the data otherwise."""
    button = page.findChild(QtWidgets.QAbstractButton, "plotResetViewButton")
    assert button is not None and button.isVisible()
    QtTest.QTest.mouseClick(button, L)
    QtWidgets.QApplication.processEvents()


def _stays_pinned(plot, side, rng):
    """The axis is still pinned to ``rng``: Axis properties says so and the view draws it."""
    QtWidgets.QApplication.processEvents()          # a paint settles pyqtgraph's pending auto-range
    state = _axis(plot, side)
    assert state.range == pytest.approx(rng), (side, state.range, rng)
    assert state.shown_range == pytest.approx(rng, rel=1e-3), (side, state.shown_range, rng)


def _is_automatic(plot, side, was=None, auto_ranging=True):
    """The axis is back on automatic: Axis properties says Auto, pyqtgraph is auto-ranging it
    again (``auto_ranging``; Distributions frames its data itself), and the view draws all the
    data (so not the range that was pinned)."""
    QtWidgets.QApplication.processEvents()          # a paint settles pyqtgraph's pending auto-range
    state = _axis(plot, side)
    assert state.range is None, (side, state.range)
    view = plot.getViewBox()
    index = 0 if side == "bottom" else 1
    assert not auto_ranging or view.autoRangeEnabled()[index], (side, view.autoRangeEnabled())
    lo, hi = view.viewRange()[index]
    bounds = view.childrenBounds()[index]
    assert bounds is not None
    slack = 1e-6 * max(abs(bounds[1] - bounds[0]), 1e-12)
    assert lo <= bounds[0] + slack and hi >= bounds[1] - slack, ((lo, hi), bounds)
    if bounds[1] - bounds[0] > 1e-9 * max(abs(bounds[0]), abs(bounds[1]), 1e-12):
        assert hi - lo <= 1.3 * (bounds[1] - bounds[0]) + slack, ((lo, hi), bounds)   # not a degenerate span
    if was is not None:
        assert state.shown_range != pytest.approx(was, rel=1e-3), (state.shown_range, was)


def _axes_of(page):
    """The pins the page keeps, as saved: ``{key: AxisPin dict}``."""
    return page.appearance_state().get("axes", {})


def _depth(widget):
    return widget._history.depth()


# -- the pure parts ---------------------------------------------------------------------------------

def _pinned(**pins):
    out = PlotAppearance()
    for key, pin in pins.items():
        out = out.with_axis_pin(key.replace("__", "."), pin)
    return out


def test_without_axis_pin_takes_the_range_and_tick_off_the_axis_a_view_draws():
    look = _pinned(left=AxisPin(min=1.0, max=2.0, tick_major=0.5, log=True),
                   psd__left=AxisPin(min=3.0, max=4.0), psd__bottom=AxisPin(min=5.0, max=6.0),
                   fft__left=AxisPin(min=7.0, max=8.0))
    # A one-view window: the bare key. The range and the tick go; log is how the axis is drawn.
    one = look.without_axis_pin("", "left")
    assert one.axis_pin("left") == AxisPin(log=True)
    assert one.axis_pin("psd.left") == look.axis_pin("psd.left")
    # The primary view of a window with several: its own key AND the bare key an older project saved.
    primary = look.without_axis_pin("psd", "left", primary="psd")
    assert primary.axis_pin("psd.left") == AxisPin() and primary.axis_pin("left") == AxisPin(log=True)
    assert primary.axis_pin("psd.bottom") == look.axis_pin("psd.bottom")
    assert primary.axis_pin("fft.left") == look.axis_pin("fft.left")
    older = _pinned(left=AxisPin(min=1.0, max=2.0), bottom=AxisPin(min=0.5, max=0.7))
    gone = older.without_axis_pin("psd", "left", primary="psd")
    assert gone.axes == older.axes[:1] and gone.axis_pin("bottom") == older.axis_pin("bottom")
    # Another view: only its own key; the bare key is the primary view's.
    other = look.without_axis_pin("fft", "left", primary="psd")
    assert other.axis_pin("fft.left") == AxisPin() and other.axis_pin("left") == look.axis_pin("left")
    # Nothing pinned there: the same appearance back.
    assert PlotAppearance().without_axis_pin("psd", "left", primary="psd") == PlotAppearance()
    # Only the pin changes.
    titled = replace(look, title="Mine", font_family="Arial")
    assert replace(titled.without_axis_pin("", "left"), axes=look.axes) == titled


def test_pinned_axes_and_cleared_pins_name_each_pin_once_in_the_order_given():
    from nanoscope.gui.quantity_reset import cleared_pins, pinned_axes

    look = _pinned(bottom=AxisPin(min=1.0, max=2.0), left=AxisPin(min=0.0, max=9.0, tick_major=3.0),
                   curves__bottom=AxisPin(min=1.0, max=3.0),
                   categories__left=AxisPin(min=2.0, max=4.0), curves__left=AxisPin(min=0.0, max=1.0))
    assert pinned_axes(look, [("", "left"), ("", "bottom")]) == ("Y range", "Y major tick", "X range")
    assert pinned_axes(look, [("", "bottom")]) == ("X range",)
    # A tick alone is named for itself; an axis with no pin is not named.
    assert pinned_axes(_pinned(left=AxisPin(tick_major=2.0)), [("", "left"), ("", "bottom")]) == (
        "Y major tick",)
    assert pinned_axes(PlotAppearance(), [("", "left")]) == ()
    # Two axes that carry one name (Distribution summaries' value axis) are said once.
    value = [("curves", "bottom", "Value"), ("categories", "left", "Value")]
    assert pinned_axes(look, value, primary="curves") == ("Value range",)
    # What a window draws is what counts: another view's pin is not this view's.
    assert pinned_axes(look, [("psd", "left")], primary="psd") == ("Y range", "Y major tick")  # bare = primary's
    assert pinned_axes(look, [("fft", "left")], primary="psd") == ()
    after, names = cleared_pins(look, [("", "left")])
    assert names == ("Y range", "Y major tick") and after.axes == look.axes[:4]   # 'left' sorts last
    again, none = cleared_pins(after, [("", "left")])
    assert none == () and again == after
    after, names = cleared_pins(look, value, primary="curves")
    assert names == ("Value range",)
    assert after.axis_pin("curves.bottom") == AxisPin() and after.axis_pin("bottom") == AxisPin()
    assert after.axis_pin("categories.left") == AxisPin()
    assert after.axis_pin("curves.left") == look.axis_pin("curves.left")


def test_the_sentence_names_the_ranges_after_the_words():
    from nanoscope.gui.quantity_reset import reset_sentence

    assert reset_sentence(("Title", "Y label", "Y range"), "median depth") == (
        "Title, Y label and Y range reset: the plot now shows median depth.")
    assert reset_sentence(("Y range",), "median depth") == (
        "Y range reset: the plot now shows median depth.")
    assert reset_sentence(("X range", "Y range"), "the UMAP embedding") == (
        "X range and Y range reset: the plot now shows the UMAP embedding.")
    assert reset_sentence(("Value range",), "the ECDF of median depth") == (
        "Value range reset: the plot now shows the ECDF of median depth.")
    assert reset_sentence(("Title", "Value range"), "the ECDF of median depth") == (
        "Title and value range reset: the plot now shows the ECDF of median depth.")
    assert reset_sentence(("Title", "X label", "Y label", "Y range", "Y major tick"), "x") == (
        "Title, X label, Y label, Y range and Y major tick reset: the plot now shows x.")


def test_the_widget_takes_words_and_pins_in_one_step_and_undo_brings_both_back(qapp):
    """The shared method: words and pins are one step; nothing to clear is no step; ``only`` limits
    the words and leaves every pin; a pin on an axis that is not named stays."""
    from nanoscope.gui.plot_appearance import PlotAppearanceWidget

    class Bar:
        def __init__(self):
            self.text = ""

        def showMessage(self, text, msecs):
            self.text = text

        def currentMessage(self):
            return self.text

        def clearMessage(self):
            self.text = ""

    pins = _pinned(left=AxisPin(min=1.0, max=2.0), bottom=AxisPin(min=3.0, max=4.0))
    start = replace(pins, title="Mine", y_label="Rate")
    widget = PlotAppearanceWidget(start)
    bar = Bar()
    axes = (("", "left"),)
    said = widget.reset_typed_words("median depth", bar, axes=axes)
    assert said == bar.text == "Title, Y label and Y range reset: the plot now shows median depth."
    state = widget.state()
    assert (state.title, state.y_label) == (None, None)
    assert state.axis_pin("left") == AxisPin() and state.axis_pin("bottom") == pins.axis_pin("bottom")
    assert _depth(widget) == (1, 0)
    assert widget.undo() and widget.state() == start and _depth(widget) == (0, 1)
    assert widget.redo() and widget.state() == state
    # Nothing typed and no pin on the axes named: no step, no message of its own.
    assert widget.reset_typed_words("event count", bar, axes=axes) == "" and bar.text == ""
    assert _depth(widget) == (1, 0)
    # A pin alone is a step; the words stay out of the sentence.
    widget.set_state(pins)
    widget.clear_history()
    assert widget.reset_typed_words("median depth", axes=axes) == (
        "Y range reset: the plot now shows median depth.")
    assert widget.state().axis_pin("left") == AxisPin() and _depth(widget) == (1, 0)
    # ``only`` is the words; the pins named are still taken, unless none are named.
    widget.set_state(start)
    widget.clear_history()
    assert widget.reset_typed_words("the linear fit", only=("fit_label",)) == ""
    assert widget.state() == start and _depth(widget) == (0, 0)
    # And nothing is cleared that was not named: the x pin stayed through all of it.
    assert widget.reset_typed_words("x", axes=axes) != ""
    assert widget.state().axis_pin("bottom") == pins.axis_pin("bottom")


# -- Voltage dependence ---------------------------------------------------------------------------

def test_voltage_dependence_a_new_metric_takes_the_y_range_with_the_words_and_keeps_the_x_range(vdep):
    """Before: a Y range pinned for the rate in Hz still framed median depths in pA, over a
    typed 'Rate (Hz)'. The metric is the Y axis; the X axis is voltage whatever the metric."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    assert set(_axes_of(page)) == {"bottom", "left"}
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)
    steps = _depth(widget)
    _section(page, "Analysis")

    _pick(page._metric_combo, text="Median depth (pA)")

    # One step, and it took the words and the Y pin together.
    assert _depth(widget) == (steps[0] + 1, 0)
    assert _drawn_title(plot)[0] == "" and _label(plot, "left") == "Median depth (pA)"
    _is_automatic(plot, "left", was=y_pin)                    # what is DRAWN: the data again
    assert "left" not in _axes_of(page)
    _stays_pinned(plot, "bottom", x_pin)                      # the other axis keeps its pin
    assert _axes_of(page)["bottom"] == {"min": x_pin[0], "max": x_pin[1]}
    plot.getViewBox().setYRange(0.0, 1.0, padding=0)          # a zoom by hand ...
    _reset_view(page)                                         # ... and Reset view: the data, not the old pin
    _is_automatic(plot, "left", was=y_pin, auto_ranging=False)
    _stays_pinned(plot, "bottom", x_pin)
    assert _said(page, explorer) == (
        "Title, Y label and Y range reset: the plot now shows median depth.", True)

    _press_undo(page)                                         # ONE Undo: the words and the pin
    assert _depth(widget) == (steps[0], 1)
    assert (_drawn_title(plot)[0], _label(plot, "left")) == ("Capture vs voltage", "Rate (Hz)")
    _stays_pinned(plot, "left", y_pin)
    _stays_pinned(plot, "bottom", x_pin)
    _press_redo(page)                                         # and Redo takes them again
    assert _depth(widget) == (steps[0] + 1, 0) and _drawn_title(plot)[0] == ""
    _is_automatic(plot, "left", was=y_pin)
    _stays_pinned(plot, "bottom", x_pin)
    # Automatic is what a fresh draw of this metric frames: the same range to the last digit.
    cleared = plot.getViewBox().viewRange()[1]
    _pick(page._metric_combo, text="Capture rate (Hz)")       # nothing typed, nothing pinned on Y
    _pick(page._metric_combo, text="Median depth (pA)")
    assert _depth(widget) == (steps[0] + 1, 0)
    assert plot.getViewBox().viewRange()[1] == pytest.approx(cleared, rel=1e-12)


@pytest.mark.parametrize("typed, sentence", [
    ("range", "Y range reset: the plot now shows median depth."),
    ("tick", "Y major tick reset: the plot now shows median depth."),
    ("both", "Y range and Y major tick reset: the plot now shows median depth.")])
def test_voltage_dependence_a_pin_alone_is_a_step_and_a_sentence_and_a_tick_is_part_of_it(
        vdep, typed, sentence):
    """Nothing typed but a pin on the axis that moved: still one step, and the sentence names
    exactly what went (a major tick spacing is typed in the numbers of the old metric too)."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    if typed == "tick":
        _pin(plot, "left", None, None, major=10.0)
        assert _axes_of(page) == {"left": {"tick_major": 10.0}}
    else:
        y_pin = _pin_inside(plot, "left", major=10.0 if typed == "both" else None)
        _stays_pinned(plot, "left", y_pin)
    assert _drawn_title(plot)[0] == "" and _boxes(widget) == ("", "", "")
    steps = _depth(widget)
    _section(page, "Analysis")

    _pick(page._metric_combo, text="Median depth (pA)")

    assert _depth(widget) == (steps[0] + 1, 0)
    assert _axes_of(page) == {}
    assert _axis(plot, "left").range is None and _axis(plot, "left").tick_major is None
    if typed != "tick":
        _is_automatic(plot, "left", was=y_pin)
    assert _said(page, explorer) == (sentence, True)
    _press_undo(page)
    assert _depth(widget) == (steps[0], 1) and "left" in _axes_of(page)
    if typed != "tick":
        _stays_pinned(plot, "left", y_pin)
    if typed != "range":
        assert _axis(plot, "left").tick_major == 10.0


def test_voltage_dependence_a_pin_on_the_axis_that_did_not_move_records_nothing(vdep):
    """Only the X axis is pinned and the metric changes: X is still voltage, so there is nothing
    to take back, no step and no sentence."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    x_pin = _pin_inside(plot, "bottom", major=100.0)
    before = _axes_of(page)
    steps = _depth(widget)
    _section(page, "Analysis")
    _clear_status(page)

    _pick(page._metric_combo, text="Median depth (pA)")

    assert _depth(widget) == steps and _said(page, explorer)[0] == ""
    assert _axes_of(page) == before
    _stays_pinned(plot, "bottom", x_pin)
    assert _axis(plot, "bottom").tick_major == 100.0
    assert _label(plot, "left") == "Median depth (pA)"


def test_voltage_dependence_another_trend_takes_the_fit_wording_and_no_pin(vdep):
    """Applying another trend moves no axis: the wording goes (G1), both pins stay."""
    page, explorer = vdep                                      # the linear trend is applied
    plot, widget = page.plot_item(), page._appearance_widget
    _edit_double_click(plot, page._fit_label, "Linear fit: {equation}")
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    steps = _depth(widget)
    _section(page, "Analysis")
    _user_picks(page._fit_combo, "exponential")
    _click_button(page._fit_apply_btn)

    assert page._applied_fit == "exponential"
    assert widget.state().fit_label_state().template == ""
    assert _depth(widget) == (steps[0] + 1, 0)
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot now shows the exponential fit.", True)
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)
    assert set(_axes_of(page)) == {"bottom", "left"}
    assert widget.undo()                                       # the wording, over the new trend
    _stays_pinned(plot, "left", y_pin)


def test_voltage_dependence_the_setter_is_the_combo_and_takes_the_pin_too(vdep):
    page, explorer = vdep
    plot = page.plot_item()
    y_pin = _pin_inside(plot, "left")
    page.set_metric("median_depth")
    assert "left" not in _axes_of(page)
    _is_automatic(plot, "left", was=y_pin)
    assert _said(page, explorer)[0] == "Y range reset: the plot now shows median depth."


def test_voltage_dependence_abs_v_takes_the_x_range_and_leaves_the_words_and_the_y_range(vdep):
    """|V| turns the X axis from the signed voltage into its magnitude, so a range pinned in signed
    mV (here about -867 to -422) frames nothing over points at 199 to 1001 mV: the trap a Y range
    was over another metric. It goes back to automatic in one step with a sentence. The words are
    not among what |V| puts back (quantity_reset), and the Y range is another axis."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    view = plot.getViewBox()
    _add_title(plot, "Capture vs voltage")
    x_pin = _pin_inside(plot, "bottom", major=100.0)
    y_pin = _pin_inside(plot, "left")
    assert x_pin[1] < 0, "the series is at negative bias"
    steps = _depth(widget)
    _section(page, "Analysis")

    _click_button(page._abs_v)                                  # |V| on

    assert page._abs_v.isChecked() and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"left"}
    assert _label(plot, "bottom") == "Applied |voltage| (mV)"
    _is_automatic(plot, "bottom", was=x_pin)                    # what is DRAWN: every magnitude
    assert view.viewRange()[0][0] > 0 and _axis(plot, "bottom").tick_major is None
    _stays_pinned(plot, "left", y_pin)
    assert _drawn_title(plot)[0] == "Capture vs voltage"        # the words stay
    assert _said(page, explorer) == (
        "X range and X major tick reset: the plot now shows capture rate against |V|.", True)
    cleared = view.viewRange()[0]
    # To return to the signed voltage with its range: clear |V| first (nothing is pinned any more,
    # so that records no step and takes the sentence down), then Undo.
    _click_button(page._abs_v)
    assert _depth(widget) == (steps[0] + 1, 0) and _said(page, explorer)[0] == ""
    _press_undo(page)
    assert _depth(widget) == (steps[0], 1)
    _stays_pinned(plot, "bottom", x_pin)                        # it frames the signed points again
    assert _axis(plot, "bottom").tick_major == 100.0
    _stays_pinned(plot, "left", y_pin)
    _press_redo(page)
    _is_automatic(plot, "bottom", was=x_pin)
    # Automatic is what a fresh draw of the magnitudes frames: the same range to the last digit.
    _click_button(page._abs_v)
    assert _depth(widget) == (steps[0] + 1, 0)
    assert view.viewRange()[0] == pytest.approx(cleared, rel=1e-12)


def test_voltage_dependence_clearing_abs_v_takes_a_range_pinned_over_the_magnitudes(vdep):
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    view = plot.getViewBox()
    _section(page, "Analysis")
    _click_button(page._abs_v)                                  # nothing pinned yet: no step
    x_pin = _pin_inside(plot, "bottom")
    assert x_pin[0] > 0
    steps = _depth(widget)

    _click_button(page._abs_v)                                  # back to the signed voltage

    assert not page._abs_v.isChecked() and _depth(widget) == (steps[0] + 1, 0)
    assert _axes_of(page) == {}
    _is_automatic(plot, "bottom", was=x_pin)
    assert view.viewRange()[0][1] < 0                           # the signed (negative) points
    assert _said(page, explorer) == (
        "X range reset: the plot now shows capture rate against the signed voltage.", True)


def test_voltage_dependence_abs_v_with_no_x_range_records_nothing_and_the_y_range_stays(vdep):
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    y_pin = _pin_inside(plot, "left", major=10.0)
    pins, steps = _axes_of(page), _depth(widget)
    _section(page, "Analysis")
    _clear_status(page)

    _click_button(page._abs_v)
    _click_button(page._abs_v)

    assert _depth(widget) == steps and _said(page, explorer)[0] == "" and _axes_of(page) == pins
    _stays_pinned(plot, "left", y_pin)
    assert _axis(plot, "left").tick_major == 10.0


def test_voltage_dependence_a_template_and_a_restore_move_abs_v_without_taking_the_x_range(vdep):
    """Both block the check box's signal and draw what they hold: the rule does not run, and the
    next tick of the box is measured from what they drew."""
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _section(page, "Analysis")
    _click_button(page._abs_v)                                  # |V| on, then a range over it
    x_pin = _pin_inside(plot, "bottom")
    artifact = workspace.capture_voltage_dependence(page)
    settings, appearance = page.template_settings()
    _click_button(page._abs_v)                                  # signed again: the range went
    assert _axes_of(page) == {}
    _clear_status(page)

    pinned = {"bottom": {"min": x_pin[0], "max": x_pin[1]}}
    page.apply_template_settings(settings, appearance)          # a template: |V| on, with its range
    assert page._abs_v.isChecked() and _axes_of(page) == pinned
    _stays_pinned(plot, "bottom", x_pin)
    assert _said(page, explorer)[0] == ""
    _click_button(page._abs_v)                                  # the next tick is a change
    assert _axes_of(page) == {} and "against the signed voltage" in _said(page, explorer)[0]
    _clear_status(page)

    assert workspace.restore_voltage_dependence(page, artifact).restored      # a saved page
    assert page._abs_v.isChecked() and _axes_of(page) == pinned
    _stays_pinned(plot, "bottom", x_pin)
    assert _said(page, explorer)[0] == "" and _depth(widget) == (0, 0)
    _click_button(page._abs_v)                                  # measured from the restore
    assert _axes_of(page) == {}
    _is_automatic(plot, "bottom", was=x_pin)
    assert _said(page, explorer)[0].startswith("X range reset:")


def test_voltage_dependence_the_setter_is_the_check_box_and_takes_the_x_range_too(vdep):
    page, explorer = vdep
    plot = page.plot_item()
    x_pin = _pin_inside(plot, "bottom")
    page.set_abs_voltage(True)
    assert "bottom" not in _axes_of(page)
    _is_automatic(plot, "bottom", was=x_pin)
    assert _said(page, explorer)[0] == "X range reset: the plot now shows capture rate against |V|."


def test_voltage_dependence_a_page_with_nothing_drawn_has_no_x_axis_to_leave(vdep):
    """With every recording closed the plot shows a message and no axis: ticking |V| leaves no
    quantity, so the range pinned earlier is not the rule's to take (the empty plot of the metric's
    own test)."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _pin_inside(plot, "bottom")
    for document in list(page._session.documents()):
        page._session.close_document(document.doc_id)
    page.refresh()
    assert page.points() == [] and page._message_item is not None
    pins, steps = _axes_of(page), _depth(widget)
    _section(page, "Analysis")
    _clear_status(page)
    _click_button(page._abs_v)
    assert _depth(widget) == steps and _said(page, explorer)[0] == "" and _axes_of(page) == pins


def test_voltage_dependence_routes_that_are_not_a_change_of_metric_keep_both_pins(vdep):
    """A refresh, the same metric again, Analyze / refresh, the Explorer's Refresh current,
    Accepted only, Log y (a real log axis, whose range is typed in the units it shows), a closed
    recording, a preset, a template of the same metric, and Undo/Redo of something else: none of
    them is the Y axis changing quantity. (|V| moves the X axis and has its own tests.)"""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    _add_title(plot, "Capture vs voltage")
    pins, words, steps = _axes_of(page), widget.state(), _depth(widget)
    _section(page, "Analysis")
    _clear_status(page)

    _pick(page._metric_combo, text="Capture rate (Hz)")         # the same metric again
    _click_button(page._workflow.primary_actions[0])            # Analyze / refresh
    explorer.refresh_current()                                  # the Explorer's Refresh current
    _click_button(page._log_y)
    _stays_pinned(plot, "left", y_pin)                          # on the log axis, in its units
    _stays_pinned(plot, "bottom", x_pin)
    _click_button(page._log_y)
    _section(page, "Data")
    _click_button(page._accepted_only)
    _click_button(page._accepted_only)
    _section(page, "Analysis")
    session = page._session                                     # another recording: one is closed
    doc_id = list(session.documents())[0].doc_id
    session.close_document(doc_id)
    page.refresh()
    settings, appearance = page.template_settings()
    page.apply_template_settings(settings, appearance)          # a template of this metric
    widget._apply_preset(1)                                     # a preset carries the look only
    assert widget.undo()
    assert _depth(widget)[0] == steps[0] and widget.state() == words    # the preset: undone
    assert _axes_of(page) == pins and _said(page, explorer)[0] == ""
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)
    _edit_double_click(plot, plot.getAxis("bottom").label, "Bias (mV)")
    assert widget.undo() and widget.redo() and widget.undo()    # Undo/Redo of something else
    assert _axes_of(page) == pins and _said(page, explorer)[0] == ""
    _stays_pinned(plot, "left", y_pin)


def test_voltage_dependence_a_template_of_another_metric_installs_its_own_pins(vdep):
    """The rule does not run when a template moves the metric: the template's appearance is
    installed whole, its pins included, and nothing is said."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    y_pin = _pin_inside(plot, "left")
    settings, appearance = page.template_settings()
    assert appearance["axes"]["left"] == {"min": y_pin[0], "max": y_pin[1]}
    _clear_status(page)
    steps = _depth(widget)

    page.apply_template_settings({**settings, "metric": "median_depth"}, appearance)

    assert page._metric_combo.currentText() == "Median depth (pA)"
    assert _axes_of(page)["left"] == {"min": y_pin[0], "max": y_pin[1]}
    _stays_pinned(plot, "left", y_pin)
    assert _depth(widget) == steps and _said(page, explorer)[0] == ""
    # The next pick is a change of metric, and takes the pin.
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Event count")
    assert "left" not in _axes_of(page)
    _is_automatic(plot, "left")


def test_voltage_dependence_pins_saved_in_a_project_open_as_saved_and_the_next_metric_takes_one(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save while both axes are pinned, reopened by ``_open_project``: the page opens
    with both ranges drawn and says nothing; the next metric takes the Y pin and not the X one."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    window._open_vdep()
    page = window._vdep_window
    explorer = window._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot = page.plot_item()
    _add_title(plot, "Capture vs voltage")
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    saved = _axes_of(page)
    assert set(saved) == {"bottom", "left"}
    project = tmp_path / "pins.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_vdep()
    QtWidgets.QApplication.processEvents()
    page = reopened._vdep_window
    explorer = reopened._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot = page.plot_item()
    assert _axes_of(page) == saved
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)
    assert _drawn_title(plot)[0] == "Capture vs voltage" and _said(page, explorer)[0] == ""
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Capture rate (Hz)")        # opening drew it: the same metric
    assert _axes_of(page) == saved
    _pick(page._metric_combo, text="Median depth (pA)")        # the next one is a change
    assert set(_axes_of(page)) == {"bottom"}
    _is_automatic(plot, "left", was=y_pin)
    _stays_pinned(plot, "bottom", x_pin)
    assert _said(page, explorer) == (
        "Title and Y range reset: the plot now shows median depth.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


def test_voltage_dependence_abs_v_range_saved_in_a_project_opens_as_saved_and_clearing_takes_it(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save with |V| ticked and a range pinned over the magnitudes, reopened by
    ``_open_project``: the page opens on it, drawn, and says nothing; the first tick of the box
    after opening is a change (opening drew the saved state, so it is measured from that)."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    window._open_vdep()
    page = window._vdep_window
    explorer = window._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot = page.plot_item()
    _section(page, "Analysis")
    _click_button(page._abs_v)
    x_pin = _pin_inside(plot, "bottom")
    saved = _axes_of(page)
    assert set(saved) == {"bottom"} and x_pin[0] > 0
    project = tmp_path / "abs-pins.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_vdep()
    QtWidgets.QApplication.processEvents()
    page = reopened._vdep_window
    explorer = reopened._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot = page.plot_item()
    assert page._abs_v.isChecked() and _axes_of(page) == saved
    _stays_pinned(plot, "bottom", x_pin)
    assert _said(page, explorer)[0] == ""
    _section(page, "Analysis")
    _click_button(page._abs_v)
    assert _axes_of(page) == {}
    _is_automatic(plot, "bottom", was=x_pin)
    assert _said(page, explorer) == (
        "X range reset: the plot now shows capture rate against the signed voltage.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


# -- Distribution summaries -----------------------------------------------------------------------

def test_summaries_a_new_measurement_takes_the_value_range_of_both_pairs_and_says_so_once(summaries):
    """The measurement is the value axis, which is x in ECDF and Survival and y in Box and
    Violin, so one new measurement takes the ranges of BOTH pairs' value axes (the Box range
    typed earlier would otherwise frame depths in pA with a dwell range) and names them once.
    The ECDF's fraction axis is not the measurement: its pin stays."""
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    value = _pin_inside(plot, "bottom")                         # the ECDF's value axis (x)
    fraction = _pin_inside(plot, "left")                        # the ECDF's fraction axis (y)
    _section(page, "Analysis")
    _pick(page._view_combo, data="box")
    _generate(page)
    box_value = _pin_inside(plot, "left")                       # Box: the value is y
    assert set(_axes_of(page)) == {"curves.bottom", "curves.left", "categories.left"}
    _pick(page._view_combo, data="ecdf")
    _generate(page)
    _edit_double_click(plot, plot.titleLabel, "Dwell ECDF")
    _stays_pinned(plot, "bottom", value)
    _stays_pinned(plot, "left", fraction)
    steps = _depth(widget)
    _section(page, "Data")

    _pick(page._metric_combo, data="depth_pA")                  # a draft: nothing moves yet
    assert page._stale and _depth(widget) == steps and _said(page, explorer)[0] == ""
    _stays_pinned(plot, "bottom", value)
    _generate(page)                                             # the commit

    assert _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"curves.left"}               # both value pins went, the fraction stayed
    _is_automatic(plot, "bottom", was=value)
    _stays_pinned(plot, "left", fraction)
    assert _drawn_title(plot)[0] == "ECDF"
    sentence, on_screen = _said(page, explorer)
    assert sentence == "Title and value range reset: the plot now shows the ECDF of median depth."
    assert on_screen and sentence.count("value range") == 1
    _section(page, "Analysis")                                  # the Box pair's pin went too
    _pick(page._view_combo, data="box")
    _generate(page)
    _is_automatic(plot, "left", was=box_value)
    assert plot.getAxis("bottom").labelText == "Source"
    _pick(page._view_combo, data="ecdf")
    _generate(page)
    _stays_pinned(plot, "left", fraction)

    # No value axis is pinned any more: the next measurement takes the title alone (G1).
    _edit_double_click(plot, plot.titleLabel, "Depth ECDF")
    _section(page, "Data")
    _pick(page._metric_combo, data="dwell_s")
    _generate(page)
    assert _said(page, explorer)[0] == "Title reset: the plot now shows the ECDF of dwell time."
    assert _axes_of(page).keys() == {"curves.left"}             # the fraction pin is still there
    _press_undo(page)                                           # takes the title back
    assert _drawn_title(plot)[0] == "Depth ECDF"


@pytest.mark.parametrize("kind, phrase", [("box", "a box plot"), ("violin", "a violin plot")])
def test_summaries_a_new_measurement_leaves_the_sources_axis_of_box_and_violin_pinned(
        summaries, kind, phrase):
    """The sources are the other axis of the Box and Violin pair, as the fraction is of the ECDF
    and Survival pair: a new measurement is the value axis changing, and takes only that."""
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    _section(page, "Analysis")
    _pick(page._view_combo, data=kind)
    _generate(page)
    sources = _pin_inside(plot, "bottom")
    value = _pin_inside(plot, "left")
    assert set(_axes_of(page)) == {"categories.bottom", "categories.left"}
    steps = _depth(widget)
    _section(page, "Data")

    _pick(page._metric_combo, data="depth_pA")
    _generate(page)

    assert page._applied_config["metric"] == "depth_pA" and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"categories.bottom"}         # the sources axis kept its pin
    _stays_pinned(plot, "bottom", sources)
    _is_automatic(plot, "left", was=value)
    assert _said(page, explorer) == (
        f"Value range reset: the plot now shows {phrase} of median depth.", True)
    _press_undo(page)
    _stays_pinned(plot, "bottom", sources)
    _stays_pinned(plot, "left", value)


def test_summaries_undo_brings_back_the_words_and_every_pair_s_value_pin_in_one_step(summaries):
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    value = _pin_inside(plot, "bottom")
    _section(page, "Analysis")
    _pick(page._view_combo, data="violin")
    _generate(page)
    box_value = _pin_inside(plot, "left")
    _pick(page._view_combo, data="ecdf")
    _generate(page)
    _edit_double_click(plot, plot.titleLabel, "Dwell ECDF")
    pins, steps = _axes_of(page), _depth(widget)
    _section(page, "Data")
    _pick(page._metric_combo, data="frac_depth")
    _generate(page)
    assert _axes_of(page) == {} and _depth(widget) == (steps[0] + 1, 0)

    _press_undo(page)

    assert _depth(widget) == (steps[0], 1) and _axes_of(page) == pins
    assert _drawn_title(plot)[0] == "Dwell ECDF"
    _stays_pinned(plot, "bottom", value)
    _press_redo(page)                                           # Redo takes the same three again
    assert _depth(widget) == (steps[0] + 1, 0) and _axes_of(page) == {}
    _is_automatic(plot, "bottom", was=value)
    _press_undo(page)
    _section(page, "Analysis")
    _pick(page._view_combo, data="violin")
    _generate(page)
    _stays_pinned(plot, "left", box_value)                      # the other pair's pin is back as well


def test_summaries_a_summary_type_moves_no_axis_so_every_pin_stays_while_the_words_go(summaries):
    """The pins are kept per pair (ECDF and Survival share theirs, as Box and Violin do) and a
    pair's own wait for its return. Changing only the Type therefore takes the words (G1) and no
    pin, whichever Type it goes to; a pair's fraction axis is a fraction in both of its Types."""
    window, page, explorer = summaries
    plot = page._plot
    value = _pin_inside(plot, "bottom")
    fraction = _pin_inside(plot, "left")
    pins = {"curves.bottom": _axes_of(page)["curves.bottom"], "curves.left": _axes_of(page)["curves.left"]}
    _section(page, "Analysis")
    for kind, side, rng in (("survival", "bottom", value), ("survival", "left", fraction)):
        _edit_double_click(plot, plot.titleLabel, f"Typed for {kind}")
        _pick(page._view_combo, data=kind)
        _generate(page)
        assert _said(page, explorer)[0].startswith("Title reset:")      # the words, and only the words
        assert _axes_of(page) == pins
        _stays_pinned(plot, side, rng)
        _pick(page._view_combo, data="ecdf")
        _generate(page)
    _pick(page._view_combo, data="box")
    _generate(page)
    box_value = _pin_inside(plot, "left")
    pins["categories.left"] = _axes_of(page)["categories.left"]
    assert _axes_of(page) == pins
    for kind in ("violin", "ecdf", "box"):
        _edit_double_click(plot, plot.titleLabel, f"Typed for {kind}")
        _pick(page._view_combo, data=kind)
        _generate(page)
        assert _axes_of(page) == pins, kind
        assert _said(page, explorer)[0].startswith("Title reset:")
    _pick(page._view_combo, data="violin")
    _generate(page)
    _stays_pinned(plot, "left", box_value)                       # Box and Violin share theirs
    _pick(page._view_combo, data="ecdf")
    _generate(page)
    _stays_pinned(plot, "bottom", value)                         # and the ECDF's came back with it
    _stays_pinned(plot, "left", fraction)


def test_summaries_routes_that_are_not_a_new_measurement_keep_every_pin(summaries):
    """Generating the same figure again, the Explorer's Refresh current, ticking another source,
    a template that carries another measurement, and the project: none is the value axis moving
    under the page's feet (the recording is no change of quantity, owner decision 2)."""
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    value = _pin_inside(plot, "bottom")
    fraction = _pin_inside(plot, "left")
    _edit_double_click(plot, plot.titleLabel, "Dwell ECDF")
    pins, words, steps = _axes_of(page), widget.state(), _depth(widget)
    _clear_status(page)

    _generate(page)
    explorer.refresh_current()
    _section(page, "Data")
    other = next(i for i in range(page._source_list.count())
                 if page._source_list.item(i).checkState() != QtCore.Qt.CheckState.Checked)
    page._source_list.item(other).setCheckState(QtCore.Qt.CheckState.Checked)       # another pore
    _generate(page)
    assert len(page.selected_sources()) == 2
    # A template with another measurement: the page installs the template's appearance (the same
    # pins here) and the Generate that commits its measurement keeps what came with it.
    settings, appearance = page.template_settings()
    page.apply_template_settings({**settings, "metric": "frac_depth"}, page.appearance_state())
    _generate(page)
    assert page._applied_config["metric"] == "frac_depth"
    assert _axes_of(page) == pins and widget.state() == words and _depth(widget) == steps
    assert _said(page, explorer)[0] == ""
    _stays_pinned(plot, "bottom", value)
    _stays_pinned(plot, "left", fraction)
    # The next measurement that is not a template's takes the value pin.
    _pick(page._metric_combo, data="dwell_s")
    _generate(page)
    assert set(_axes_of(page)) == {"curves.left"}


def test_summaries_pins_saved_in_a_project_open_as_saved_and_a_new_measurement_takes_the_value_pin(
        summaries, qtbot, tmp_path):
    """One real save with the ECDF pinned on both axes and the Box pinned on its value, reopened
    by ``_open_project``: every pin is back, drawn, and a Generate of the same figure keeps them."""
    window, page, explorer = summaries
    plot = page._plot
    value = _pin_inside(plot, "bottom")
    fraction = _pin_inside(plot, "left")
    _section(page, "Analysis")
    _pick(page._view_combo, data="box")
    _generate(page)
    _pin_inside(plot, "left")
    _pick(page._view_combo, data="ecdf")
    _generate(page)
    saved = _axes_of(page)
    assert set(saved) == {"curves.bottom", "curves.left", "categories.left"}
    project = tmp_path / "summaries-pins.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_distribution_summaries()
    page = reopened._distribution_summary_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    plot = page._plot
    assert page.has_data() and _axes_of(page) == saved
    _stays_pinned(plot, "bottom", value)
    _stays_pinned(plot, "left", fraction)
    assert _said(page, explorer)[0] == ""
    _generate(page)                                             # the same figure again
    assert _axes_of(page) == saved
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    assert set(_axes_of(page)) == {"curves.left"}
    _is_automatic(plot, "bottom", was=value)
    _stays_pinned(plot, "left", fraction)
    assert _said(page, explorer) == ("Value range reset: the plot now shows the ECDF of median depth.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


# -- Noise ----------------------------------------------------------------------------------------

def test_noise_the_asd_takes_the_psd_s_y_range_with_the_words_and_keeps_every_other_pin(noise):
    """PSD against ASD is the PSD view's y axis (units²/Hz against units/√Hz): a range pinned
    for the one frames nothing of the other. Frequency stays frequency, and the other views'
    pins are their own."""
    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    _section(page, "Data")
    _pick(page._view_combo, data="spectrogram")                 # a pin on another view's axis
    spec_pin = _pin_inside(page._spec_plot, "left")
    _pick(page._view_combo, data="psd")
    _edit_double_click(plot, plot.titleLabel, "PSD, pore A")
    freq_pin = _pin_inside(plot, "bottom")
    psd_pin = _pin_inside(plot, "left")
    assert set(_axes_of(page)) == {"psd.bottom", "psd.left", "spectrogram.left"}
    steps = _depth(widget)
    _section(page, "Analysis")

    _click_button(page._asd_check)                              # PSD -> ASD

    assert page.is_asd() and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"psd.bottom", "spectrogram.left"}
    _is_automatic(plot, "left", was=psd_pin)
    _stays_pinned(plot, "bottom", freq_pin)                     # frequency did not move
    assert _label(plot, "left").startswith("ASD (") and _drawn_title(plot)[0].startswith("ASD — ")
    assert _said(page, explorer) == (
        "Title and Y range reset: the plot now shows the amplitude spectral density.", True)
    assert _axis(page._spec_plot, "left").range == pytest.approx(spec_pin)    # the spectrogram's own
    _press_undo(page)                                           # one Undo: the title and the PSD range
    assert _depth(widget) == (steps[0], 1)
    assert _drawn_title(plot)[0] == "PSD, pore A"
    _stays_pinned(plot, "left", psd_pin)
    _stays_pinned(plot, "bottom", freq_pin)
    assert set(_axes_of(page)) == {"psd.bottom", "psd.left", "spectrogram.left"}
    _press_redo(page)
    _is_automatic(plot, "left", was=psd_pin)
    cleared = plot.getViewBox().viewRange()[1]                  # what a fresh draw of the ASD frames
    _click_button(page._asd_check)
    _click_button(page._asd_check)
    assert page.is_asd() and plot.getViewBox().viewRange()[1] == pytest.approx(cleared, rel=1e-9)

    # The same axis moving back is a change too: pin again, tick the box off.
    psd_pin = _pin_inside(plot, "left")
    _click_button(page._asd_check)
    assert not page.is_asd() and "psd.left" not in _axes_of(page)
    _is_automatic(plot, "left", was=psd_pin)
    assert _said(page, explorer)[0] == "Y range reset: the plot now shows the power spectral density."
    _stays_pinned(plot, "bottom", freq_pin)


def test_noise_going_to_another_view_and_back_keeps_the_psd_range_and_every_view_its_own(noise):
    """A pin belongs to one view's axis, so leaving the view takes none off: PSD, FFT,
    Spectrogram, All-points, PSD brings the PSD's range back as it was, and takes the words (G1)."""
    page, explorer = noise
    plot = page._plot
    freq_pin = _pin_inside(plot, "bottom")
    psd_pin = _pin_inside(plot, "left")
    _section(page, "Data")
    _pick(page._view_combo, data="fft")
    fft_pin = _pin_inside(page._fft_plot, "bottom")
    pins = _axes_of(page)
    assert set(pins) == {"psd.bottom", "psd.left", "fft.bottom"}
    for view in ("spectrogram", "allpoints", "fft", "psd"):
        _edit_double_click(page.active_plot_item(), page.active_plot_item().titleLabel, "Typed here")
        _pick(page._view_combo, data=view)
        assert _axes_of(page) == pins, view
        assert _said(page, explorer)[0].startswith("Title reset:"), view
    _stays_pinned(plot, "bottom", freq_pin)
    _stays_pinned(plot, "left", psd_pin)
    assert _axis(page._fft_plot, "bottom").range == pytest.approx(fft_pin)
    _pick(page._view_combo, data="allpoints")
    assert _axis(page._hist_plot, "bottom").range is None                # never typed there
    _pick(page._view_combo, data="psd")


def test_noise_routes_that_do_not_change_the_psd_keep_both_of_its_pins(noise, tmp_path):
    """Log-log, the Welch settings, a refresh, another recording and a template of the same
    spectrum are not the PSD's y axis moving: they change how the spectrum is estimated or
    drawn, not what the axis measures."""
    from .test_psd import _current_doc

    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    freq_pin = _pin_inside(plot, "bottom")
    psd_pin = _pin_inside(plot, "left")
    pins, steps = _axes_of(page), _depth(widget)
    _section(page, "Analysis")
    _clear_status(page)
    _click_button(page._loglog_check)
    _click_button(page._loglog_check)
    page.set_nperseg(2048)
    page.set_overlap_percent(25)
    page.set_window("flattop")
    page.refresh()
    explorer.refresh_current()
    settings, appearance = page.template_settings()
    page.apply_template_settings(settings, appearance)
    _section(page, "Data")
    page._session.add_document(_current_doc(tmp_path, name="second.dat"))
    page.refresh()
    _pick(page._rec_combo, text=page._rec_combo.itemText(1))     # another recording
    assert _axes_of(page) == pins and _depth(widget) == steps and _said(page, explorer)[0] == ""
    _stays_pinned(plot, "bottom", freq_pin)
    _stays_pinned(plot, "left", psd_pin)


def test_noise_log_frequency_is_a_real_log_axis_and_the_spectrograms_range_stays(noise):
    """Log frequency draws the spectrogram's rows on an axis in log mode, so a range pinned in Hz
    is the same range on either scale: it stays, drawn, when the box is ticked and cleared."""
    page, explorer = noise
    widget = page._appearance_widget
    _section(page, "Data")
    _pick(page._view_combo, data="spectrogram")
    plot = page._spec_plot
    rng = _pin_inside(plot, "left")
    assert rng[0] > 0
    pins, steps = _axes_of(page), _depth(widget)
    _clear_status(page)

    for on in (True, False):
        page._spec_logf_check.setChecked(on)
        QtWidgets.QApplication.processEvents()
        assert plot.getAxis("left").logMode is on
        assert _axes_of(page) == pins and _depth(widget) == steps
        _stays_pinned(plot, "left", rng)
    assert _said(page, explorer)[0] == ""


def test_noise_a_template_that_switches_to_the_asd_installs_its_own_pins(noise):
    """The rule does not run when a template moves the spectrum to the ASD: the template's
    appearance is installed whole, its PSD range included, and nothing is said. The next flip of
    the check box is a change, and takes the range."""
    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    freq_pin = _pin_inside(plot, "bottom")
    level = _pin_inside(plot, "left")
    settings, appearance = page.template_settings()
    pins, steps = _axes_of(page), _depth(widget)
    _clear_status(page)

    page.apply_template_settings({**settings, "asd": True}, appearance)

    assert page.is_asd() and _axes_of(page) == pins and _depth(widget) == steps
    assert _said(page, explorer)[0] == ""
    _stays_pinned(plot, "bottom", freq_pin)
    _stays_pinned(plot, "left", level)
    _section(page, "Analysis")
    _click_button(page._asd_check)                                  # ASD -> PSD: the Y axis moves
    assert set(_axes_of(page)) == {"psd.bottom"}
    _is_automatic(plot, "left", was=level)
    assert _said(page, explorer)[0] == "Y range reset: the plot now shows the power spectral density."


def test_noise_pins_saved_in_a_project_open_as_saved_and_the_asd_takes_the_psd_one(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save while the PSD is pinned on both axes, reopened by ``_open_project``."""
    from nanoscope.core.project import DocumentRef, ProjectState

    from .test_plot_window_routes import _noisy_csv

    recording = tmp_path / "noise.csv"
    _noisy_csv(recording)
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    window = _main_window(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(recording))]))
    window._open_psd()
    page = window._psd_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.has_data()
    plot = page._plot
    _edit_double_click(plot, plot.titleLabel, "PSD of the pore")
    freq_pin = _pin_inside(plot, "bottom")
    psd_pin = _pin_inside(plot, "left")
    saved = _axes_of(page)
    assert set(saved) == {"psd.bottom", "psd.left"}
    project = tmp_path / "noise-pins.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_psd()
    page = reopened._psd_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    plot = page._plot
    assert _axes_of(page) == saved and _drawn_title(plot)[0] == "PSD of the pore"
    _stays_pinned(plot, "bottom", freq_pin)
    _stays_pinned(plot, "left", psd_pin)
    assert _said(page, explorer)[0] == ""
    _section(page, "Analysis")
    _click_button(page._loglog_check)                            # the same spectrum, other axes
    _click_button(page._loglog_check)
    assert _axes_of(page) == saved
    _click_button(page._asd_check)
    assert set(_axes_of(page)) == {"psd.bottom"}
    _is_automatic(plot, "left", was=psd_pin)
    _stays_pinned(plot, "bottom", freq_pin)
    assert _said(page, explorer) == (
        "Title and Y range reset: the plot now shows the amplitude spectral density.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


# -- Event waveforms ------------------------------------------------------------------------------

def _tab(page, index):
    """Click a tab of the page: 0 the Event-count heatmap, 1 the Ensemble."""
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(index).center())
    QtWidgets.QApplication.processEvents()


def _pin_the_event_figures(page):
    """Linear dwell on the map (Log x off, as a person types a range in seconds), the map's x and
    y (with a major tick on y) and the ensemble's x and y pinned through Axis properties."""
    _section(page, "Analysis")
    if page._logx.isChecked():
        _click_button(page._logx)
    if page._logy.isChecked():
        _click_button(page._logy)
    mp, ens = page._density_main, page._ensemble_plot
    pins = {"map.bottom": _pin_inside(mp, "bottom"), "map.left": _pin_inside(mp, "left", major=20.0)}
    _tab(page, 1)
    pins["ensemble.bottom"] = _pin_inside(ens, "bottom")
    pins["ensemble.left"] = _pin_inside(ens, "left")
    _tab(page, 0)
    return pins


def _type_event_labels(page):
    """The shared title on the map, and both shared axis labels in the Appearance boxes."""
    _edit_double_click(page._density_main, page._density_main.titleLabel, "Dwell vs depth")
    _section(page, "Appearance")
    box = page._appearance_widget
    _type_in_box(box.x_label, "Dwell")
    _type_in_box(box.y_label, "Depth")
    QtTest.QTest.qWait(800)                      # the run of typing closes: one step
    _section(page, "Analysis")


def test_event_waveforms_the_x_metric_takes_the_maps_x_range_with_the_words_and_nothing_else(events):
    page, explorer = events
    mp, ens, widget = page._density_main, page._ensemble_plot, page._appearance_widget
    pins = _pin_the_event_figures(page)
    _type_event_labels(page)
    assert set(_axes_of(page)) == set(pins)
    steps = _depth(widget)
    before = _axes_of(page)

    _pick(page._x_combo, data="frac_depth")

    assert _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"map.left", "ensemble.bottom", "ensemble.left"}
    _is_automatic(mp, "bottom", was=pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])                  # Y did not move, nor did its tick
    assert _axis(mp, "left").tick_major == 20.0
    assert _said(page, explorer) == (
        "Title, X label, Y label and X range reset: the plot now shows median depth against dI/I₀.",
        True)
    _tab(page, 1)                                                # the ensemble keeps its own pins
    _stays_pinned(ens, "bottom", pins["ensemble.bottom"])
    _stays_pinned(ens, "left", pins["ensemble.left"])
    _tab(page, 0)
    _press_undo(page)                                            # ONE Undo
    assert _depth(widget) == (steps[0], 1) and _axes_of(page) == before
    assert (_drawn_title(mp)[0], _label(mp, "bottom")) == ("Dwell vs depth", "Dwell")
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _press_redo(page)
    _is_automatic(mp, "bottom", was=pins["map.bottom"])


def test_event_waveforms_the_y_metric_takes_the_maps_y_range_and_tick_and_nothing_else(events):
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    pins = _pin_the_event_figures(page)
    steps, before = _depth(widget), _axes_of(page)

    _pick(page._y_combo, data="frac_depth")                      # nothing typed: the pins alone

    assert _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"map.bottom", "ensemble.bottom", "ensemble.left"}
    _is_automatic(mp, "left", was=pins["map.left"])
    assert _axis(mp, "left").tick_major is None
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    assert _said(page, explorer) == (
        "Y range and Y major tick reset: the plot now shows dI/I₀ against dwell.", True)
    _press_undo(page)
    assert _axes_of(page) == before and _depth(widget) == (steps[0], 1)
    _stays_pinned(mp, "left", pins["map.left"])
    assert _axis(mp, "left").tick_major == 20.0


def test_event_waveforms_normalize_time_takes_the_ensembles_x_range_and_the_map_keeps_its_pins(events):
    page, explorer = events
    mp, ens, widget = page._density_main, page._ensemble_plot, page._appearance_widget
    pins = _pin_the_event_figures(page)
    _tab(page, 1)
    _edit_double_click(ens, ens.titleLabel, "Mean event")
    steps, before = _depth(widget), _axes_of(page)
    _section(page, "Analysis")

    _click_button(page._normalize)                               # normalised time -> time in seconds

    assert not page._normalize.isChecked() and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"map.bottom", "map.left", "ensemble.left"}
    _is_automatic(ens, "bottom", was=pins["ensemble.bottom"])
    _stays_pinned(ens, "left", pins["ensemble.left"])            # current stays current
    assert _label(ens, "bottom") == "Time (s)"
    assert _said(page, explorer) == (
        "Title and X range reset: the plot now shows the ensemble against time in seconds.", True)
    _tab(page, 0)
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])
    _tab(page, 1)
    _press_undo(page)
    assert _axes_of(page) == before and _drawn_title(ens)[0] == "Mean event"
    _stays_pinned(ens, "bottom", pins["ensemble.bottom"])


def test_event_waveforms_log_x_takes_the_maps_x_range_and_not_the_words(events):
    """With Log x ticked the map draws log10 values on a linear axis, so a range pinned in seconds
    (about 0.0014 to 0.0029) frames nothing over log10 dwell (about -3.04 to -2.43): a blank map.
    The range goes in one step with a sentence. The words are not among what Log x puts back
    (quantity_reset), and the map's Y and the ensemble's axes are other axes."""
    page, explorer = events
    mp, ens, widget = page._density_main, page._ensemble_plot, page._appearance_widget
    pins = _pin_the_event_figures(page)
    _type_event_labels(page)
    steps, before = _depth(widget), _axes_of(page)
    _section(page, "Analysis")

    _click_button(page._logx)

    assert page._logx.isChecked() and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == set(before) - {"map.bottom"}
    _is_automatic(mp, "bottom", was=pins["map.bottom"])         # what is DRAWN: every log10 dwell
    assert mp.getViewBox().viewRange()[0][1] < 0
    _stays_pinned(mp, "left", pins["map.left"])                 # Y did not move, nor did its tick
    assert _axis(mp, "left").tick_major == 20.0
    assert (_drawn_title(mp)[0], _label(mp, "bottom"), _label(mp, "left")) == (
        "Dwell vs depth", "Dwell", "Depth")                     # the words stay
    assert _said(page, explorer) == (
        "X range reset: the plot now shows median depth against log10 of dwell.", True)
    cleared = mp.getViewBox().viewRange()[0]
    # To return to the linear map with its range: clear Log x first (nothing is pinned on X now,
    # so that records no step and takes the sentence down), then Undo.
    _click_button(page._logx)
    assert _depth(widget) == (steps[0] + 1, 0) and _said(page, explorer)[0] == ""
    _press_undo(page)
    assert _depth(widget) == (steps[0], 1) and _axes_of(page) == before
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _press_redo(page)
    _is_automatic(mp, "bottom", was=pins["map.bottom"])
    # Automatic is what a fresh draw of the log map frames: the same range to the last digit.
    _click_button(page._logx)
    assert page._logx.isChecked() and _depth(widget) == (steps[0] + 1, 0)
    assert mp.getViewBox().viewRange()[0] == pytest.approx(cleared, rel=1e-12)
    _tab(page, 1)                                                # the ensemble kept its own pins
    _stays_pinned(ens, "bottom", pins["ensemble.bottom"])
    _stays_pinned(ens, "left", pins["ensemble.left"])


def test_event_waveforms_log_y_takes_the_maps_y_range_and_tick_and_nothing_else(events):
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    pins = _pin_the_event_figures(page)
    _edit_double_click(mp, mp.titleLabel, "Dwell vs depth")
    steps, before = _depth(widget), _axes_of(page)

    _click_button(page._logy)

    assert page._logy.isChecked() and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == set(before) - {"map.left"}
    _is_automatic(mp, "left", was=pins["map.left"])
    assert _axis(mp, "left").tick_major is None
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    assert _drawn_title(mp)[0] == "Dwell vs depth"
    assert _said(page, explorer) == (
        "Y range and Y major tick reset: the plot now shows log10 of median depth against dwell.",
        True)
    _click_button(page._logx)                                    # and both log: both axes named
    assert _said(page, explorer)[0] == (
        "X range reset: the plot now shows log10 of median depth against log10 of dwell.")
    _click_button(page._logy)                                    # nothing pinned on Y any more
    assert _said(page, explorer)[0] == ""
    assert set(_axes_of(page)) == {"ensemble.bottom", "ensemble.left"}


def test_event_waveforms_log_scale_in_axis_properties_keeps_the_range_typed_beside_it(events):
    """The map opens on a log x axis (dwell). Its Axis properties show log10 values with Log scale
    ticked and refuse a range at or below zero, so a range in seconds is typed by unticking Log
    scale in the same dialog. That Log change takes no pin off, or the range typed with it would go
    at once; ticking the box in Analysis afterwards is the change that takes it."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    assert page._logx.isChecked() and page._density_logs == (True, False)
    steps = _depth(widget)

    _pin(mp, "bottom", 0.0012, 0.003, log=False)

    assert not page._logx.isChecked()
    assert _axes_of(page) == {"map.bottom": {"min": 0.0012, "max": 0.003}}
    _stays_pinned(mp, "bottom", (0.0012, 0.003))                 # drawn over the linear dwell
    assert mp.getViewBox().childrenBounds()[0][0] > 0
    assert _depth(widget) == (steps[0] + 1, 0) and _said(page, explorer)[0] == ""
    assert page._density_logs == (False, False)
    _section(page, "Analysis")
    _click_button(page._logx)                                    # the box: the change
    assert _axes_of(page) == {}
    _is_automatic(mp, "bottom", was=(0.0012, 0.003))
    assert _said(page, explorer)[0] == (
        "X range reset: the plot now shows median depth against log10 of dwell.")
    # Ticking Log scale in the dialog is the same route: nothing is taken, nothing is said.
    _clear_status(page)
    _pin(mp, "bottom", None, None, log=True)
    assert page._logx.isChecked() and _said(page, explorer)[0] == ""


def test_event_waveforms_the_log_setter_is_the_check_boxes_and_takes_the_ranges_too(events):
    """``set_density_log`` sets both boxes at once and redraws as a click on them does, so the
    range (and tick) of each axis whose box moved goes with it, in one step."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    pins = _pin_the_event_figures(page)                          # Log x and Log y off, four pins
    steps = _depth(widget)

    page.set_density_log(True, True)

    assert page._logx.isChecked() and page._logy.isChecked()
    assert _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"ensemble.bottom", "ensemble.left"}
    _is_automatic(mp, "bottom", was=pins["map.bottom"])
    _is_automatic(mp, "left", was=pins["map.left"])
    assert _said(page, explorer) == (
        "X range, Y range and Y major tick reset: the plot now shows log10 of median depth "
        "against log10 of dwell.", True)
    page.set_density_log(True, True)                             # nothing moved: nothing to take
    assert _depth(widget) == (steps[0] + 1, 0)


def test_event_waveforms_an_empty_map_has_no_log_state_to_leave(events):
    """With no events to draw the map shows a message and no axis, so Log x leaves no quantity and
    the range pinned earlier is not the rule's to take: not on a live window that lost its source,
    nor on one that restored a snapshot saved empty."""
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    doc_id, channel, eventset = page._doc_id, page._channel, page._eventset
    _section(page, "Analysis")
    if page._logx.isChecked():
        _click_button(page._logx)
    x_pin = _pin_inside(mp, "bottom")
    page.set_source(None, None, None)
    assert page.density_counts() is None
    pins, steps = _axes_of(page), _depth(widget)
    _clear_status(page)
    _click_button(page._logx)
    assert _depth(widget) == steps and _axes_of(page) == pins and _said(page, explorer)[0] == ""
    _click_button(page._logx)

    empty = workspace.capture_event_plots(page)                  # saved with nothing to show
    page.set_source(doc_id, channel, eventset)
    assert page.density_counts() is not None
    assert workspace.restore_event_plots(page, empty).restored
    assert page.density_counts() is None and _axes_of(page) == pins
    steps = _depth(widget)
    _click_button(page._logx)
    assert _depth(widget) == steps and _axes_of(page) == pins and _said(page, explorer)[0] == ""
    assert _axes_of(page)["map.bottom"] == {"min": x_pin[0], "max": x_pin[1]}


def test_event_waveforms_switching_the_view_takes_the_words_and_no_pin(events):
    page, explorer = events
    mp, ens = page._density_main, page._ensemble_plot
    pins = _pin_the_event_figures(page)
    before = _axes_of(page)
    for index in (1, 0, 1, 0):
        plot = page._density_main if index == 1 else page._ensemble_plot      # the one being left
        _edit_double_click(plot, plot.titleLabel, "Typed here")
        _tab(page, index)
        assert _axes_of(page) == before
        assert _said(page, explorer)[0].startswith("Title reset:")
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])
    _tab(page, 1)
    _stays_pinned(ens, "bottom", pins["ensemble.bottom"])
    _stays_pinned(ens, "left", pins["ensemble.left"])


def test_event_waveforms_routes_that_are_not_a_new_metric_keep_every_pin(events, tmp_path):
    """Bins, the colormap, the band, the alignment while time is normalised, the number of points,
    another recording, a template that carries another metric, and a workspace restore are not the
    axes changing quantity. (Log x and Log y change the numbers on the map's axes, and the alignment
    of a real-time ensemble what t = 0 is: they have their own tests.)"""
    from nanoscope.core import analysis_workspace as workspace

    from .test_event_plot import _doc_and_events, _install_detection

    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    pins = _pin_the_event_figures(page)
    before, steps = _axes_of(page), _depth(widget)
    _clear_status(page)
    page.set_bins(page._bins.value() + 4)
    page.set_colormap("magma")
    page.set_band("sd")
    page.set_align("center")
    page.set_n_points(page._npoints.value() + 4)
    doc, ev = _doc_and_events(tmp_path, name="other.dat", n_events=12)
    doc_id = page._session.add_document(doc)
    _install_detection(page._session, doc_id, ev)
    page.set_source(doc_id, 0, ev)                               # another recording
    assert page.density_counts().sum() == 12
    assert _axes_of(page) == before and _depth(widget) == steps and _said(page, explorer)[0] == ""
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])
    settings, appearance = page.template_settings()
    page.apply_template_settings({**settings, "x_metric": "frac_depth"}, page.appearance_state())
    assert page._x_combo.currentData() == "frac_depth"           # the template's own pins stay
    assert _axes_of(page) == before and _said(page, explorer)[0] == ""
    artifact = workspace.capture_event_plots(page)
    assert workspace.restore_event_plots(page, artifact).restored
    assert _axes_of(page) == before and _depth(widget) == (0, 0)   # a restore starts the history over
    assert _said(page, explorer)[0] == ""
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    # The next change is measured from what the restore drew: Log y first, then the metric.
    _click_button(page._logy)
    assert "map.left" not in _axes_of(page) and "map.bottom" in _axes_of(page)
    _pick(page._x_combo, data="dwell_s")
    assert "map.bottom" not in _axes_of(page)


def test_event_waveforms_pins_saved_in_a_project_open_as_saved_and_the_next_metric_takes_one(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save with the map and the ensemble pinned, reopened by ``_open_project``."""
    window = _series_window(qtbot, tmp_path, monkeypatch)
    first = next(iter(window._session.documents())).doc_id
    window._on_selection(first, 0)
    window._open_event_plots()
    page = window._event_plot_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.density_counts() is not None
    pins = _pin_the_event_figures(page)
    _edit_double_click(page._density_main, page._density_main.titleLabel, "Dwell vs depth")
    saved = _axes_of(page)
    assert set(saved) == set(pins)
    project = tmp_path / "events-pins.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_event_plots()
    page = reopened._event_plot_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    mp, ens = page._density_main, page._ensemble_plot
    assert _axes_of(page) == saved and _drawn_title(mp)[0] == "Dwell vs depth"
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])
    assert _axis(mp, "left").tick_major == 20.0
    _tab(page, 1)                                                # the tab switch takes the title (G1)
    _stays_pinned(ens, "bottom", pins["ensemble.bottom"])
    _tab(page, 0)
    assert _axes_of(page) == saved and _said(page, explorer)[0] == ""
    _edit_double_click(mp, mp.titleLabel, "Dwell vs depth")
    _clear_status(page)
    _section(page, "Analysis")
    _pick(page._x_combo, data="dwell_s")                         # the same metric: nothing
    assert _axes_of(page) == saved
    _pick(page._x_combo, data="frac_depth")                      # the next one is a change
    assert set(_axes_of(page)) == set(saved) - {"map.bottom"}
    _is_automatic(mp, "bottom", was=pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])
    assert _said(page, explorer) == (
        "Title and X range reset: the plot now shows median depth against dI/I₀.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


# -- Subpopulations -------------------------------------------------------------------------------

def test_subpopulations_another_embedding_takes_both_scatter_ranges_and_leaves_the_waveforms(subpop):
    """Both axes of the scatter are the embedding (t-SNE 1 and 2 against UMAP 1 and 2); the
    waveforms beside it are time and current whichever embedding is drawn."""
    page, explorer = subpop
    scatter = page.embedding_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    waves = page._wave_view._plot.getPlotItem()
    _add_title(plot, "Pore A in t-SNE")
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    w_pin = _pin_inside(waves, "bottom")
    assert set(_axes_of(page)) == {"embedding.bottom", "embedding.left", "waveforms.bottom"}
    steps = _depth(widget)
    _section(page, "Analysis")

    _pick(scatter._embed_combo, text="UMAP")

    assert scatter.embedding() == "UMAP" and _depth(widget) == (steps[0] + 1, 0)
    assert set(_axes_of(page)) == {"waveforms.bottom"}
    _is_automatic(plot, "bottom", was=x_pin)
    _is_automatic(plot, "left", was=y_pin)
    _stays_pinned(waves, "bottom", w_pin)
    assert _label(plot, "bottom") == "UMAP 1 (arb. units)"
    assert _standing_line(page, explorer) == (
        "Title, X range and Y range reset: the plot now shows the UMAP embedding.", True)
    _press_undo(page)                                            # ONE Undo: the title and both ranges
    assert _depth(widget) == (steps[0], 1)
    assert _drawn_title(plot)[0] == "Pore A in t-SNE"
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)
    _stays_pinned(waves, "bottom", w_pin)
    _press_redo(page)
    _is_automatic(plot, "bottom", was=x_pin)
    # Back to t-SNE with nothing typed and nothing pinned on the scatter: no step, no sentence.
    steps = _depth(widget)
    _pick(scatter._embed_combo, text="t-SNE")
    assert _depth(widget) == steps and _standing_line(page, explorer)[0] == ""
    assert set(_axes_of(page)) == {"waveforms.bottom"}


def test_subpopulations_colouring_a_new_report_and_a_restore_keep_the_pins(subpop, subpop_report):
    """Colouring is how the same points are drawn, a re-run of the same embedding is no new
    quantity, and a restore ends with the pins it saved (it moves the embedding through the
    public setter, which resets the live page for a moment and is then replaced)."""
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = subpop
    scatter = page.embedding_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    waves = page._wave_view._plot.getPlotItem()
    x_pin = _pin_inside(plot, "bottom")
    w_pin = _pin_inside(waves, "bottom")
    pins, steps = _axes_of(page), _depth(widget)
    _section(page, "Analysis")
    _pick(scatter._colour_combo, text=scatter.colour_modes()[1])
    page.set_report(subpop_report, subtitle="a second run")
    assert _axes_of(page) == pins and _depth(widget) == steps
    _stays_pinned(plot, "bottom", x_pin)
    assert _standing_line(page, explorer)[0] == ""

    _pick(scatter._embed_combo, text="UMAP")                     # pin the UMAP, then save the page
    y_pin = _pin_inside(plot, "left")
    umap_x = _pin_inside(plot, "bottom")
    artifact = workspace.capture_subpopulation(page)
    _pick(scatter._embed_combo, text="t-SNE")                    # leaves the UMAP pins behind
    assert set(_axes_of(page)) == {"waveforms.bottom"}
    assert workspace.restore_subpopulation(page, artifact).restored

    assert scatter.embedding() == "UMAP"
    assert _axes_of(page)["embedding.bottom"] == {"min": umap_x[0], "max": umap_x[1]}
    assert _axes_of(page)["embedding.left"] == {"min": y_pin[0], "max": y_pin[1]}
    _stays_pinned(plot, "bottom", umap_x)
    _stays_pinned(plot, "left", y_pin)
    _stays_pinned(waves, "bottom", w_pin)
    assert _standing_line(page, explorer)[0] == "" and _depth(widget) == (0, 0)
    _pick(scatter._embed_combo, text="t-SNE")                    # the next embedding is a change
    assert set(_axes_of(page)) == {"waveforms.bottom"}
    assert _standing_line(page, explorer)[0] == "X range and Y range reset: the plot now shows the t-SNE embedding."


# -- Distributions: the range it already took, now named ------------------------------------------

def test_distributions_a_new_plot_type_takes_the_ranges_and_says_so(dist):
    page, explorer = dist
    plot = page._plot
    _add_title(plot, "Dwell, pore A")
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    page._history_touch()
    assert page.config().axes.manual_x is not None and page.config().axes.manual_y is not None
    depth = page._history.depth()[0]

    _pick(page._plot_type_combo, text="Scatter")

    assert page._history.depth()[0] == depth + 1
    assert page.config().axes.ranges() == ()
    assert page.config().axes.manual_x is None and page.config().axes.manual_y is None
    assert _said(page, explorer) == (
        "Title, X range and Y range reset: the plot is now set to show the scatter of ycol "
        "against xcol.", True)
    _click_button(page._plot_btn)                                # the picture follows at Plot
    _is_automatic(plot, "bottom", was=x_pin, auto_ranging=False)
    _is_automatic(plot, "left", was=y_pin, auto_ranging=False)
    assert page.undo_config()                                    # one Configuration step
    while page._plot_type_combo.currentText() != "Histogram" and page.undo_config():
        pass
    assert page._plot_type_combo.currentText() == "Histogram"
    assert page.config().axes.ranges() == ("X range", "Y range")
    assert _drawn_title(plot)[0] == "Dwell, pore A"
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)


def test_distributions_a_column_takes_the_range_of_the_axis_that_moved_and_names_it(dist):
    page, explorer = dist
    plot = page._plot
    page.set_plot_type("Scatter")
    _click_button(page._plot_btn)
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    _clear_status(page)
    columns = [page._scatter_x.itemText(i) for i in range(page._scatter_x.count())]
    other = next(name for name in columns if name not in ("xcol", "ycol"))

    _user_picks(page._scatter_x, other)                          # only the X column moved

    assert page.config().axes.ranges() == ("Y range",)
    assert _said(page, explorer)[0] == (
        f"X range reset: the plot is now set to show the scatter of ycol against {other}.")
    _click_button(page._plot_btn)
    _is_automatic(plot, "bottom", was=x_pin, auto_ranging=False)
    _stays_pinned(plot, "left", y_pin)
    _user_picks(page._scatter_y, other)                          # and now the Y column
    assert page.config().axes.ranges() == ()
    assert _said(page, explorer)[0].startswith("Y range reset: the plot is now set to show")


def test_distributions_the_time_unit_and_another_source_keep_the_ranges(dist):
    """The unit rescales a typed range and a second pore changes no quantity (owner decision 2):
    neither takes it."""
    page, explorer = dist
    plot = page._plot
    _pin_inside(plot, "bottom")
    pinned = page.config().axes
    _clear_status(page)
    _user_picks(page._time_unit, "µs")
    assert page.config().axes.manual_x == pinned.manual_x and page.config().axes.x_min != pinned.x_min
    other = next(i for i in range(page._source_list.count())
                 if page._source_list.item(i).checkState() != QtCore.Qt.CheckState.Checked)
    page._source_list.item(other).setCheckState(QtCore.Qt.CheckState.Checked)
    _click_button(page._plot_btn)
    assert page.config().axes.ranges() == ("X range",) and _said(page, explorer)[0] == ""


# -- nothing typed and no pin on an axis that moved: no step and no sentence ---------------------

def test_noise_the_asd_with_only_the_frequency_pinned_records_nothing(noise):
    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    freq_pin = _pin_inside(plot, "bottom")
    steps, pins = _depth(widget), _axes_of(page)
    _section(page, "Analysis")
    _clear_status(page)

    _click_button(page._asd_check)

    assert page.is_asd() and _depth(widget) == steps and _said(page, explorer)[0] == ""
    assert _axes_of(page) == pins
    _stays_pinned(plot, "bottom", freq_pin)


def test_event_waveforms_a_metric_on_an_axis_with_no_pin_records_nothing(events):
    """The map's x is not pinned (only its y and the ensemble's x are): a new X metric moves the
    one axis nothing is pinned on, so there is nothing to take."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    _section(page, "Analysis")
    if page._logy.isChecked():
        _click_button(page._logy)
    y_pin = _pin_inside(mp, "left")
    _tab(page, 1)
    _pin_inside(page._ensemble_plot, "bottom")
    _tab(page, 0)
    steps, pins = _depth(widget), _axes_of(page)
    _section(page, "Analysis")
    _clear_status(page)

    _pick(page._x_combo, data="frac_depth")

    assert _depth(widget) == steps and _said(page, explorer)[0] == "" and _axes_of(page) == pins
    _stays_pinned(mp, "left", y_pin)


def test_summaries_a_new_measurement_with_only_the_fraction_pinned_records_nothing(summaries):
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    fraction = _pin_inside(plot, "left")
    steps, pins = _depth(widget), _axes_of(page)
    _section(page, "Data")
    _clear_status(page)
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    assert page._applied_config["metric"] == "depth_pA"
    assert _depth(widget) == steps and _said(page, explorer)[0] == "" and _axes_of(page) == pins
    _stays_pinned(plot, "left", fraction)


def test_subpopulations_only_the_waveforms_pinned_records_nothing_for_another_embedding(subpop):
    page, explorer = subpop
    scatter = page.embedding_view()
    waves = page._wave_view._plot.getPlotItem()
    w_pin = _pin_inside(waves, "bottom")
    steps, pins = _depth(page._appearance_widget), _axes_of(page)
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")
    assert scatter.embedding() == "UMAP" and _depth(page._appearance_widget) == steps
    assert _standing_line(page, explorer)[0] == "" and _axes_of(page) == pins
    _stays_pinned(waves, "bottom", w_pin)


# -- the windows with no pin to take ------------------------------------------------------------------

def test_comparison_figures_draw_a_picture_with_no_axis_properties_and_take_no_pin(comparison):
    """Generate takes the words of a Comparison figure (G1) and no range: the figure is a
    matplotlib picture in a canvas, with no Axis properties to pin anything on. A pin that a file
    carried in its appearance (nothing in the page can make one) is not the rule's to take."""
    import pyqtgraph as pg

    page, explorer, drawn = comparison
    widget = page._appearance_widget
    assert not page.findChildren(pg.GraphicsView) and not page.findChildren(pg.PlotWidget)
    _type_comparison_words(page)
    widget.apply_appearance(widget.state().with_axis_pin("left", AxisPin(min=0.0, max=1.0)))
    pins = _axes_of(page)
    assert pins == {"left": {"min": 0.0, "max": 1.0}}
    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA")
    _generate(page)
    assert widget.state().title is None and _axes_of(page) == pins
    assert page._status.text().startswith("Title, X label and Y label reset:")
    assert "range" not in page._status.text().split("the plot now shows")[0]


def test_i_v_has_no_quantity_control_so_its_pins_have_nothing_to_move_for(qapp, qtbot, tmp_path):
    """G1's guard, for pins: I-V has no control that changes what an axis shows, so a Refresh
    keeps a pinned range (the redraw of the same quantity)."""
    from nanoscope.gui.analysis_explorer import PAGE_IV
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session
    from .test_p0_reset_on_quantity_change import _in_explorer

    page = IVWindow(_ohmic_session(tmp_path))
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_IV)
    page.activateWindow()
    plot = page.plot_item()
    x_pin = _pin_inside(plot, "bottom")
    y_pin = _pin_inside(plot, "left")
    _click_button(page._workflow.primary_actions[0])            # Analyze / refresh
    explorer.refresh_current()
    assert set(_axes_of(page)) == {"bottom", "left"} and _said(page, explorer)[0] == ""
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)


# -- the guide and the inventory ----------------------------------------------------------------------

def test_the_guide_and_the_inventory_say_what_the_rule_does_with_the_pins():
    """Each sentence is pinned by a behaviour test above: the Y range with the metric, the X range
    with |V| and the Y range kept (Voltage dependence), the value range of both pairs said once and
    the Type that takes none (Distribution summaries), the map's x and y, Log x and Log y, and the
    ensemble's x (Event waveforms), the PSD's y and no range for a view (Noise), both axes of the
    scatter (Subpopulations), the range named in Distributions' sentence, Undo and the project."""
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    flat = " ".join((root / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8").split())
    for sentence in (
            "So does the range pinned in **Axis properties** on each axis whose quantity moved, "
            "with the major tick spacing typed beside it (*range* below means both)",
            "(*Title, Y label and Y range reset: the plot now shows median depth.*); the range of "
            "an axis whose quantity did not move stays, and so does another view's unless its axis "
            "shows the same quantity (the two pairs of Distribution summaries, below)",
            "The sentence names a major tick spacing apart from the range (*Y range and Y major "
            "tick reset: …*, or *Y major tick reset: …* when only a spacing was typed)",
            "A figure with no typed words records no step and says nothing, unless a range is "
            "pinned on an axis that moved (it takes that range alone, and says so)",
            "**Voltage dependence:** the **Metric**, which is the Y axis (the X axis is voltage "
            "whatever the metric, so its range stays)",
            "no axis moves with it, so both ranges stay",
            "it takes the value range pinned in both pairs (the sentence says *value range* once), "
            "and leaves the ECDF's and Survival's fraction axis alone",
            "A new summary type takes no range: each pair keeps its own, and the one you leave is "
            "waiting when you come back",
            "A Comparison figure is a picture without Axis properties, so it has no range to take",
            "The heatmap's **X** takes the map's X range and its **Y** the map's Y range, "
            "**Normalize time** takes the ensemble's X range (current stays current), and switching "
            "tabs takes none: each view keeps its own",
            "The view takes no range: each view keeps its own, so Spectrogram and back brings the "
            "PSD's range back",
            "so it takes the PSD's Y range and leaves its frequency range",
            "which is what both axes of the scatter show, so both of its ranges go with the words "
            "(the waveforms' ranges stay)",
            "the typed range of an axis whose column moved goes with the first two, and the "
            "sentence names it",
            "(*Title, X range and Y range reset: the plot is now set to show the scatter of ycol "
            "against xcol.*)",
            "finds nothing typed and no range set takes that sentence off the status line",
            "The look stays: fonts, sizes, colours, text styles, the legend, the ranges pinned on "
            "axes whose quantity did not move",
            "**Undo** brings the words and the ranges back over the quantity that is on screen",
            "a project opens with the words and ranges it was saved with",
            "and the ranges it was saved with in place of the ones pinned",
            "A new measurement also takes the range pinned on the value axis, in both pairs; a new "
            "summary takes no range",
            "The figure is a picture without Axis properties, so there is no pinned range to take",
            "until the axis comes to show another quantity",
            # What the first pass left unpinned (each is true of the code).
            "an axis whose column moved loses its typed range (a change of plot type moves "
            "both axes)",
            "and leaves the Title, axis labels and ranges as they are",
            "(a new **Measurement** is the value axis changing, and takes the value range of both "
            "pairs)",
            "a Y range pinned for the capture rate in Hz frames nothing over median depths in "
            "pA. It goes in the same Undo step, and the sentence names it after the words",
            "In Distribution summaries a new measurement is the value axis changing, which is X in "
            "ECDF and Survival and Y in Box and Violin:",
            "A range pinned in Axis properties on an axis that moved goes in the same step: the "
            "heatmap's **X** takes the map's X range and its **Y** the map's Y range, "
            "**Normalize time** takes the ensemble's X range, and switching tabs takes none "
            "(each view keeps its own)",
            "A range pinned in Axis properties belongs to one view, so switching **Show** takes "
            "none (Spectrogram and back keeps the PSD's); **Amplitude spectral density** changes "
            "what the PSD's Y axis shows, so it takes the PSD's Y range in the same step and "
            "leaves its frequency range",
            # |V|, Log x and Log y: the controls that leave the words and still take a range.
            "Ticking or clearing **|V|** changes the X axis between the signed voltage and its "
            "magnitude, so it takes the X range (a range pinned in signed millivolts frames "
            "nothing over magnitudes), in the same step and with a sentence (*X range reset: the "
            "plot now shows capture rate against |V|.*); the words and the Y range stay",
            "**Log y** is a real log axis, whose range is typed in the units it shows, so it keeps "
            "both ranges",
            "**Log x** and **Log y** take the range of their own axis of the map and leave the "
            "words: with one ticked the map draws log10 values on a linear axis, so a range pinned "
            "over the other state frames nothing",
            "Of these, **|V|** and the Event-count heatmap's **Log x** and **Log y** still take a "
            "range, because they change the numbers on the axis it was typed in (both above); "
            "the words stay",
            "On the Event-count map, whose axes hold log10 values while Log x or Log y is "
            "ticked, a range typed in the same dialog as the Log scale change is kept, and "
            "ticking the Analysis box takes the range of its axis",
            "**Log x** and **Log y** take the range of their own axis of the map in the same way, "
            "with a sentence (*X range reset: the plot now shows median depth against log10 of "
            "dwell.*) and without the words, because the map draws log10 values on a linear axis "
            "while one is ticked",
            "Changing **Log scale** in the map's Axis properties takes no range: that dialog "
            "commits the range typed beside it, and unticking Log scale there to type a range in "
            "seconds is how the map's X axis, which opens on a log axis, is pinned",
            "Bins, the colormap, the band, the subset and the number of points keep both the "
            "words and the ranges"):
        assert sentence in flat, sentence
    # What the guide said before: pinned ranges stay through a change of quantity.
    for gone in ("text styles, axis ranges typed in Axis properties, the legend",
                 "(Distributions also clears the typed range of an axis whose column moved)",
                 "Log axes, bins, the colormap, the band, the subset and the alignment keep them."):
        assert gone not in flat, gone
    inventory = json.loads((root / "docs" / "inventory.json").read_text(encoding="utf-8"))
    said = [g for page in inventory for g in page.get("gotchas", ())
            if g.startswith("Changing what a figure shows puts the words you typed back to automatic")]
    assert len(said) == 2 and all("range" in g for g in said), said
    assert any("the PSD's Y range" in g and "(the map's X range)" in g for g in said)
    assert any("value axis in both Type pairs" in g for g in said)
    assert any("in Subpopulations the Embedding (both axes of the scatter)" in g for g in said)
    assert any("Switching the Event waveforms tab or Noise's Show view takes the words and no "
               "range (each view keeps its own)" in g for g in said)
    assert not any("but not the tab" in g or "but not the Show view" in g for g in said)
    assert any("a typed major tick is named apart from the range: Y range and Y major tick reset"
               in g for g in said)
    assert any("Two controls that leave the words still take the range of the axis whose "
               "numbers they change: |V| in Voltage dependence (the X range, signed voltage "
               "against its magnitude) and Log x / Log y on the Event-count map (log10 values on "
               "a linear axis), except that Log scale changed in Axis properties keeps the range "
               "typed beside it" in g for g in said)
    controls = {(c["label"], c.get("group")): c for area in inventory for c in area["controls"]}
    assert "X range reset: the plot now shows capture rate against |V|." in controls[
        ("|V|", "Voltage dependence ▸ controls row")]["notes"]
    for label in ("Log x", "Log y"):
        assert "takes a range (and major tick) pinned on the map's" in controls[
            (label, "Event waveforms ▸ Event-count heatmap")]["notes"]
    assert "Changing Log scale in Axis properties keeps the range typed in that dialog." in (
        controls[("Log x", "Event waveforms ▸ Event-count heatmap")]["notes"])
    # The ASD box takes the words and the PSD's Y range, so its note can no longer say "only".
    asd = controls[("Amplitude spectral density (√PSD)", "Noise spectrum ▸ Display")]["notes"]
    assert "Changes only" not in asd, asd
    assert ("Ticking or clearing it also puts the words you typed (Title, X label, Y label and the "
            "fit label's wording) back to automatic and takes a range (and major tick) pinned on "
            "the PSD's Y axis in Axis properties") in asd, asd
    assert ("'Title and Y range reset: the plot now shows the amplitude spectral density.'"
            ) in asd and "clearing the box says 'the power spectral density'" in asd, asd
    assert asd.endswith("the frequency range stays."), asd


# -- a project saved before pins were kept per view: its bare pin is the primary view's -------------

def _inside(plot, side, span=(0.2, 0.7)):
    a, b = _axis(plot, side).shown_range
    return float(f"{a + span[0] * (b - a):.3g}"), float(f"{a + span[1] * (b - a):.3g}")


def _bare(**sides):
    """The ``axes`` of an appearance saved by a build that kept one pin per side: ``bottom=(lo, hi)``."""
    return {"axes": {side: {"min": lo, "max": hi} for side, (lo, hi) in sides.items()}}


def test_noise_a_bare_pin_from_an_older_project_is_the_psds_and_goes_with_the_asd(noise):
    page, explorer = noise
    plot = page._plot
    freq, level = _inside(plot, "bottom"), _inside(plot, "left")
    page.set_appearance_state(_bare(bottom=freq, left=level))
    _stays_pinned(plot, "bottom", freq)
    _stays_pinned(plot, "left", level)
    _section(page, "Analysis")

    _click_button(page._asd_check)

    assert _axes_of(page) == {"bottom": {"min": freq[0], "max": freq[1]}}     # the bare Y pin went
    _is_automatic(plot, "left", was=level)
    _stays_pinned(plot, "bottom", freq)
    assert _said(page, explorer)[0] == "Y range reset: the plot now shows the amplitude spectral density."


def test_event_waveforms_a_bare_pin_from_an_older_project_is_the_maps_and_the_ensemble_leaves_it(events):
    page, explorer = events
    mp, ens = page._density_main, page._ensemble_plot
    rng = _inside(mp, "bottom")
    _tab(page, 1)
    e_rng = _inside(ens, "bottom")
    _tab(page, 0)
    page.set_appearance_state({"axes": {"bottom": {"min": rng[0], "max": rng[1]},          # the map's
                                        "ensemble.bottom": {"min": e_rng[0], "max": e_rng[1]}}})
    _stays_pinned(mp, "bottom", rng)
    _tab(page, 1)
    _stays_pinned(ens, "bottom", e_rng)
    _section(page, "Analysis")
    _click_button(page._normalize)                               # the ensemble's x moves, not the map's
    assert _axes_of(page) == {"bottom": {"min": rng[0], "max": rng[1]}}      # only the ensemble's went
    _is_automatic(ens, "bottom", was=e_rng)
    assert _said(page, explorer)[0] == "X range reset: the plot now shows the ensemble against time in seconds."
    _tab(page, 0)
    _stays_pinned(mp, "bottom", rng)
    _pick(page._x_combo, data="frac_depth")                      # the map's x moves: its bare pin goes
    assert _axes_of(page) == {}
    _is_automatic(mp, "bottom", was=rng)
    assert _said(page, explorer)[0] == "X range reset: the plot now shows median depth against dI/I₀."


def test_summaries_a_bare_pin_from_an_older_project_is_the_curves_value_axis(summaries):
    window, page, explorer = summaries
    plot = page._plot
    value, fraction = _inside(plot, "bottom"), _inside(plot, "left")
    page.set_appearance_state(_bare(bottom=value, left=fraction))
    _stays_pinned(plot, "bottom", value)
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    assert _axes_of(page) == {"left": {"min": fraction[0], "max": fraction[1]}}
    _is_automatic(plot, "bottom", was=value)
    _stays_pinned(plot, "left", fraction)
    assert _said(page, explorer)[0] == "Value range reset: the plot now shows the ECDF of median depth."


def test_subpopulations_a_bare_pin_from_an_older_project_is_the_scatters(subpop):
    page, explorer = subpop
    scatter = page.embedding_view()
    plot = scatter._plot.getPlotItem()
    waves = page._wave_view._plot.getPlotItem()
    x, y = _inside(plot, "bottom"), _inside(plot, "left")
    page.set_appearance_state(_bare(bottom=x, left=y))
    _stays_pinned(plot, "bottom", x)
    assert _axis(waves, "bottom").range is None                  # a bare pin is never the waveforms'
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")
    assert _axes_of(page) == {}
    _is_automatic(plot, "bottom", was=x)
    _is_automatic(plot, "left", was=y)
    assert _standing_line(page, explorer)[0] == (
        "X range and Y range reset: the plot now shows the UMAP embedding.")


def test_distributions_names_the_axes_that_hold_a_typed_limit_or_a_pin():
    from nanoscope.gui.plots_panel import AxesConfig

    assert AxesConfig().ranges() == ()
    assert AxesConfig(x_min="1").ranges() == ("X range",)
    assert AxesConfig(x_max="2").ranges() == ("X range",)
    assert AxesConfig(manual_x=(1.0, 2.0)).ranges() == ("X range",)
    assert AxesConfig(y_min="1").ranges() == ("Y range",)
    assert AxesConfig(manual_y=(1.0, 2.0)).ranges() == ("Y range",)
    assert AxesConfig(x_min="1", manual_y=(1.0, 2.0)).ranges() == ("X range", "Y range")
    assert AxesConfig(time_unit="µs").ranges() == ()                      # the unit is no range
