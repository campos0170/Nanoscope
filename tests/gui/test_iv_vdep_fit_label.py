"""I-V and Voltage dependence: the fit is never the title, and the voltage axis reads mV.

Owner, 2026-11-01 (plot UX round, group G4): a fit's numbers go in the fit label (on by
default, draggable, editable with live tokens), Voltage dependence's pooled provenance
goes in a caption on the plot (Q-F3), and "Applied voltage (V) (x0.001)" becomes
"Applied voltage (mV)" with ticks that mean millivolts. Text in both windows is editable
in the figure through figure_text, and Axis properties' Log follows each window's own
control (greyed with the reason in I-V).

Clicks go through the QWINDOW (``window().windowHandle()``): a click sent to the viewport
widget never reaches a pyqtgraph item.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisState, TextStyle, figure_text_for
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow
from nanoscope.plot.pg_svg import export_svg

from .test_iv import _ohmic_session
from .test_voltage_dep import _ladder_session

L = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def _fresh_gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


@pytest.fixture
def iv(qapp, qtbot, tmp_path):
    w = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(w)
    w.resize(900, 760)
    w.show()
    qtbot.waitExposed(w)
    return w


@pytest.fixture
def vdep(qapp, qtbot, tmp_path):
    w = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(w)
    w.set_fit("exponential")
    return w


def _double_click(widget, item) -> None:
    view = widget.plot_item().scene().views()[0]
    vp = view.viewport()
    pos = view.mapFromScene(item.sceneBoundingRect().center())
    QtTest.QTest.mouseMove(vp, pos)
    QtTest.QTest.mouseDClick(view.window().windowHandle(), L, NOMOD, vp.mapTo(view.window(), pos))
    QtWidgets.QApplication.processEvents()


def test_an_iv_fit_goes_to_the_fit_label_and_a_typed_title_keeps_g(iv, tmp_path):
    """Plan G4 test 1. Before: the title WAS 'G (I-V slope) = … nS · R² = …', so a typed
    title lost G from the exported figure."""
    assert iv.plot_item().titleLabel.text == ""
    label = iv.fit_label_text()
    assert "G (I-V slope) = 100 nS" in label and "R² = " in label and "n = 3" in label
    widget = iv._appearance_widget
    widget.apply_appearance(replace(widget.state(), title="Pore 3 < 5 nm"))
    assert iv.plot_item().titleLabel.item.toPlainText() == "Pore 3 < 5 nm"
    out = tmp_path / "iv.svg"
    export_svg(iv.plot_item(), out)
    svg = out.read_text(encoding="utf-8")
    assert "G (I-V slope)" in svg and "Pore 3 &lt; 5 nm" in svg


def test_a_two_recording_fit_keeps_its_dof_tag_whatever_the_words(qapp, qtbot, tmp_path):
    """The R² dof tag is the label's refusal: a typed template cannot drop it."""
    from .test_iv import _current_doc
    from nanoscope.core.session import Session

    session = Session()
    for mV in (-200.0, -500.0):
        session.add_document(_current_doc(tmp_path, i_open_pA=100.0 * mV, nominal_mV=mV,
                                          name=f"p{int(-mV)}.dat"))
    w = IVWindow(session)
    qtbot.addWidget(w)
    assert "exact by construction" in w.fit_label_text()
    widget = w._appearance_widget
    widget.apply_appearance(widget.state().with_text_style(
        "annotation.fit", TextStyle(text="Pore G: {equation}")))
    assert w.fit_label_text().startswith("Pore G: G (I-V slope)")
    assert "exact by construction" in w.fit_label_text()


@pytest.mark.parametrize("make", ["iv", "vdep"])
def test_the_voltage_axis_reads_millivolts(make, request):
    """Plan G4 test 3. Before: 'Applied voltage (V) (x0.001)' over ticks of 200-1000."""
    w = request.getfixturevalue(make)
    axis = w.plot_item().getAxis("bottom")
    text = axis.labelString()
    assert "mV" in text and "x0.001" not in text and "(V)" not in text
    assert axis.scale == 1000.0 and not axis.autoSIPrefix
    lo, hi = w.plot_item().getViewBox().viewRange()[0]
    assert lo * axis.scale <= -500 <= hi * axis.scale     # ticks in mV, data still in V


def test_vdep_numbers_go_to_the_fit_label_and_the_provenance_to_the_caption(vdep):
    """Plan G4 test 2 and Q-F3. Before: '_set_title(lead + prov)' put V₀ and R² and the
    pooled method in the title, so a typed title dropped both."""
    title = vdep.plot_item().titleLabel.text
    assert title == "" and "R²" not in title and "V₀" not in title
    label = vdep.fit_label_text()
    assert "V₀ = " in label and "R² = " in label
    assert vdep._caption_segments, "the ladder has a pooled method to state"
    assert vdep.caption_text() == "   ·   ".join(vdep._caption_segments)
    widget = vdep._appearance_widget
    widget.apply_appearance(replace(widget.state(), title="My series"))
    assert "V₀ = " in vdep.fit_label_text() and vdep.caption_text()


def test_vdep_caption_is_editable_draggable_and_hides_with_a_space(vdep, qtbot):
    vdep.resize(900, 760)
    vdep.show()
    qtbot.waitExposed(vdep)
    controller = figure_text_for(vdep.plot_item())
    assert "annotation.provenance" in controller.targets()
    widget = vdep._appearance_widget
    before = widget.state()
    widget.apply_appearance(before.with_text_style("annotation.provenance",
                                                   TextStyle(text="Ladder A_{1}")))
    assert vdep.caption_text() == "Ladder A1"
    widget.undo()
    assert vdep.caption_text() == "   ·   ".join(vdep._caption_segments)
    widget.apply_appearance(before.with_text_style("annotation.provenance",
                                                   TextStyle(text=" ")))
    assert vdep.caption_text() == ""
    widget.undo()
    vdep._caption.moved.emit((40.0, 60.0))
    vdep.refresh()
    assert vdep._caption.offset() == pytest.approx((40.0, 60.0))


def test_a_title_double_click_edits_it_in_one_undo_step(iv):
    """Plan G4 test 4: double-click the title, type, Return: one appearance step."""
    controller = figure_text_for(iv.plot_item())
    opened = []
    controller.opened.connect(lambda kind, key: opened.append((kind, key)))
    widget = iv._appearance_widget
    # An empty automatic title draws nothing to click; the Appearance box gives one.
    widget.apply_appearance(replace(widget.state(), title="Pore 2"))
    QtWidgets.QApplication.processEvents()
    depth = widget._history.depth()
    _double_click(iv, iv.plot_item().titleLabel.item)
    assert opened == [("inline", "title")]
    editor = controller.editor
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, "Pore 3")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert iv.plot_item().titleLabel.item.toPlainText() == "Pore 3"
    assert widget._history.depth()[0] == depth[0] + 1
    assert iv.appearance_state()["title"] == "Pore 3"
    widget.undo()
    assert iv.plot_item().titleLabel.item.toPlainText() == "Pore 2"


def test_iv_log_is_greyed_with_the_reason_and_range_pins_the_view(iv):
    """Plan G4 test 5 (I-V): no log control, so Log says why; a typed range pins the
    view and Reset view returns to it."""
    controller = figure_text_for(iv.plot_item())
    popover = controller.open_axis_properties("bottom")
    assert not popover.log.isEnabled() and "I-V has no log scale" in popover.log.toolTip()
    controller.close_editors()
    state = iv._text_host.axis_state("bottom")
    assert state.units == "" and "mV" in state.label_auto
    iv._text_host.apply_axis("bottom", replace(state, range=(-900.0, -100.0)))
    lo, hi = iv.plot_item().getViewBox().viewRange()[0]
    assert (lo * 1000, hi * 1000) == pytest.approx((-900.0, -100.0))
    iv.plot_item().getViewBox().setXRange(-0.3, -0.2, padding=0)
    iv._plot_nav.reset_view()
    lo, hi = iv.plot_item().getViewBox().viewRange()[0]
    assert (lo * 1000, hi * 1000) == pytest.approx((-900.0, -100.0))


def test_vdep_log_is_the_windows_own_log_y_and_unticking_it_turns_it_off(vdep):
    """Plan G4 test 5 (Voltage dependence): Log on the y axis toggles Log y.

    Log y is an analysis control, not an appearance: with it on, a non-positive value
    leaves the plot AND the fit, so it is saved with the window's settings and is not an
    appearance Undo step. Unticking Log (here or in Analysis) is how it goes back."""
    host = vdep._text_host
    state = host.axis_state("left")
    assert state.log_available and state.log is False
    depth = vdep._appearance_widget._history.depth()
    host.apply_axis("left", replace(state, log=True))
    assert vdep._log_y.isChecked() and vdep.plot_item().getAxis("left").logMode
    assert host.axis_state("left").log is True
    assert vdep._appearance_widget._history.depth() == depth, "not an appearance step"
    assert vdep.template_settings()[0]["log_y"] is True
    host.apply_axis("left", replace(host.axis_state("left"), log=False))
    assert not vdep._log_y.isChecked()
    # The x axis has no log scale: asking for it changes nothing.
    host.apply_axis("bottom", replace(host.axis_state("bottom"), log=True))
    assert not vdep.plot_item().getAxis("bottom").logMode
    assert isinstance(state, AxisState)


# -- the fit's numbers in the axis's own frame (review intent-3, INTEGRATE §3 "V₀ in V") ---------

def _number(pattern: str, text: str) -> float:
    import re

    match = re.search(pattern, text)
    assert match, f"{pattern!r} not in {text!r}"
    return float(match.group(1).replace("−", "-"))


def _plotted_in_axis_frame(w):
    """The points as the axis shows them: x in mV (the axis scale), y as plotted."""
    x, y = w._scatter_item.getData()
    return np.asarray(x) * w.plot_item().getAxis("bottom").scale, np.asarray(y)


def test_vdep_linear_slope_is_per_mv_like_its_axis(qapp, qtbot, tmp_path):
    """Before: 'slope = -73.31' beside 'Applied voltage (mV)', a slope per V with no unit:
    read off the axis it was 1000x too large."""
    w = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(w)
    w.set_fit("linear")
    x_mV, y = _plotted_in_axis_frame(w)
    per_mV = np.polyfit(x_mV, y, 1)[0]                  # Hz per mV, off the axis itself
    pattern = r"slope = (\S+) Hz/mV"
    for text in (w.fit_label_text(), w._headline.text(), w.details_text()):
        assert _number(pattern, text) == pytest.approx(per_mV, rel=1e-3)
    assert "Applied voltage (mV)" in w.plot_item().getAxis("bottom").labelString()
    out = tmp_path / "vdep.svg"
    export_svg(w.plot_item(), out)
    assert "Hz/mV" in out.read_text(encoding="utf-8")
    # The CSV keeps the fit's own numbers in the frame of its voltage_V column, labelled.
    data = w._plot_data()
    params = json.loads(data["fit_params_json"].iloc[0])
    units = json.loads(data["fit_params_units_json"].iloc[0])
    assert units == {"slope": "Hz/V", "intercept": "Hz"}
    assert params["slope"] == pytest.approx(per_mV * 1000.0, rel=1e-6)
    # A metric with no unit says "per mV".
    w.set_metric("median_frac_depth")
    assert "per mV" in w.fit_label_text() and "per mV" in w._headline.text()


def test_vdep_v0_is_in_mv_like_its_axis(qapp, qtbot, tmp_path):
    """INTEGRATE §3: 'V₀ = -0.5 V' beside an axis in mV."""
    w = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(w)
    w.set_fit("exponential")
    x_mV, y = _plotted_in_axis_frame(w)
    v0_mV = 1.0 / np.polyfit(x_mV, np.log(y), 1)[0]    # value = A·exp(V / V₀), V in mV
    pattern = r"V₀ = (\S+) mV"
    for text in (w.fit_label_text(), w._headline.text(), w.details_text()):
        assert _number(pattern, text) == pytest.approx(v0_mV, rel=1e-3)
    assert v0_mV == pytest.approx(-500.0, rel=1e-6)
    data = w._plot_data()
    assert json.loads(data["fit_params_units_json"].iloc[0]) == {"A": "Hz", "V0": "V"}
    assert json.loads(data["fit_params_json"].iloc[0])["V0"] == pytest.approx(-0.5)
    # |V| plots the magnitude, in mV too.
    w.set_abs_voltage(True)
    x_mV, y = _plotted_in_axis_frame(w)
    assert _number(pattern, w.fit_label_text()) == pytest.approx(
        1.0 / np.polyfit(x_mV, np.log(y), 1)[0], rel=1e-3)


def test_iv_slope_and_conductance_agree_with_the_mv_axis(qapp, qtbot, tmp_path):
    """I-V's own numbers against its mV axis: G in nS is the slope in pA per mV, and a
    slope without nS says per V explicitly (1000x the axis-frame slope)."""
    from .test_s3_units_iv_window import _two_pore_session

    w = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(w)
    x_mV, y_pA = _plotted_in_axis_frame(w)
    per_mV, at_zero = np.polyfit(x_mV, y_pA, 1)
    assert _number(r"G \(I-V slope\) = (\S+) nS", w.fit_label_text()) == pytest.approx(
        per_mV, rel=1e-3)                                # pA/mV is nS
    assert _number(r"Leak \(intercept\):\s+(\S+) pA at 0 V", w.details_text()) == \
        pytest.approx(at_zero, abs=1e-3)
    data = w._plot_data()
    assert set(data["fit_slope_units"]) == {"pA/V"} and set(data["fit_intercept_units"]) == {"pA"}
    assert data["fit_slope"].iloc[0] == pytest.approx(per_mV * 1000.0, rel=1e-6)

    session, _ids = _two_pore_session(tmp_path)          # current units undeclared
    w = IVWindow(session)
    qtbot.addWidget(w)
    x_mV, y = _plotted_in_axis_frame(w)
    per_mV = np.polyfit(x_mV, y, 1)[0]
    for text in (w.fit_label_text(), w._headline.text()):
        assert _number(r"slope = (\S+) \(units unknown\)/V", text) == pytest.approx(
            per_mV * 1000.0, rel=1e-3)
    assert set(w._plot_data()["fit_slope_units"]) == {"(units unknown)/V"}
