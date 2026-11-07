"""In-figure text editing in the other live plot windows (plan G4, owner 2026-11-01).

Noise, Event waveforms, Distribution summaries and Subpopulations each install
figure_text's controller through ``install_plot_interaction(…, text=host)``: an edit is
one appearance Undo step and is saved by ``appearance_state()``. The window-specific
targets are the colour-bar and marginal labels (Event waveforms), the 1/f fit label of a
single curve (Noise) and legend names owned by the Source names editor (Distribution
summaries).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import TextStyle, figure_text_for

from .test_distribution_summary_panel import _source


@pytest.fixture(autouse=True)
def _fresh_gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _commit_inline(controller, key: str, text: str) -> None:
    editor = controller.edit_inline(key)
    assert editor is not None, key
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, text)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()


def test_event_waveforms_colour_bar_and_marginal_labels_are_editable(qapp, qtbot):
    """Before: the colour bar's 'Event count' and the marginals' 'count' had no store."""
    from nanoscope.gui.event_plot_panel import EventPlotWindow

    w = EventPlotWindow(Session())
    qtbot.addWidget(w)
    w.resize(900, 800)
    w.show()
    qtbot.waitExposed(w)
    controller = figure_text_for(w._density_main)
    keys = controller.targets()
    assert {"colorbar.label", "marginal.left.label", "marginal.bottom.label"} <= set(keys)
    assert figure_text_for(w._density_top) is None      # one controller, on the map
    assert figure_text_for(w._ensemble_plot) is not None
    widget = w._appearance_widget
    depth = widget._history.depth()[0]
    _commit_inline(controller, "colorbar.label", "Events per bin")
    assert w.label_text("colorbar.label") == "Events per bin"
    # On the label the user SEES: use the horizontal color scale's active axis.
    seen = w._colorbar.axis
    assert seen.label.isVisible() and seen.label.toPlainText().strip() == "Events per bin"
    assert w.appearance_state()["text"]["colorbar.label"]["text"] == "Events per bin"
    assert widget._history.depth()[0] == depth + 1
    widget.apply_appearance(widget.state().with_text_style(
        "marginal.left.label", TextStyle(text="n", bold=True)))
    assert w.label_text("marginal.left.label") == "n"
    assert w._density_top.getAxis("left").labelStyle.get("font-weight") == "bold"
    widget.apply_appearance(widget.state().with_text_style("marginal.left.label", TextStyle()))
    assert w.label_text("marginal.left.label") == "count"
    assert "font-weight" not in w._density_top.getAxis("left").labelStyle
    widget.undo()
    widget.undo()
    widget.undo()
    assert w.label_text("colorbar.label") == "Event count"
    # The map's own axis pins never reach the marginals' count axes.
    widget.apply_appearance(widget.state().with_axis_pin(
        "left", figure_text.AxisPin(min=0.0, max=1.0)))
    assert not getattr(w._density_top, "_nanoscope_axis_pins", {})


def test_distribution_summary_legend_renames_go_to_the_source_names_store(qapp, qtbot):
    """Plan G4: legend names stay owned by the Source names editor (one store)."""
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow

    page = DistributionSummaryWindow()
    qtbot.addWidget(page)
    page.set_sources([_source([1., 2., 3., 4.])])
    page.analyze()
    page.resize(900, 760)
    page.show()
    qtbot.waitExposed(page)
    controller = figure_text_for(page._plot)
    key = "legend.entry.measurements"
    assert key in controller.targets()
    assert controller.targets()[key].auto == "Measured events"
    _commit_inline(controller, key, "Apo <1>")
    assert page.source_appearance_state()["measurements"]["label"] == "Apo <1>"
    assert page._plot.legend.items[0][1].text.startswith("Apo &lt;1&gt;")
    # The appearance only carries the name through Undo; what is saved and drawn is the
    # Source names store's (build 5: the rename is one Undo step).
    assert "text" not in page.appearance_state(), "no second legend-entry store"
    controller._set(key, TextStyle())                   # Reset name
    assert "label" not in page.source_appearance_state().get("measurements", {})


def test_noise_single_curve_alpha_is_on_the_figure(qapp, qtbot, tmp_path):
    """Before: a single curve's 1/f exponent was printed in Details only, because the
    legend row that names it is drawn for an overlay alone."""
    from .test_psd import _fed_window

    w, _doc = _fed_window(qtbot, tmp_path)
    if w._fit_item is None:
        pytest.skip("the synthetic recording gives no 1/f fit")
    assert len(w._curves) == 1
    assert "α = " in w.fit_label_text()
    assert not any("α" in text for text in w.legend_labels())
    widget = w._appearance_widget
    widget.apply_appearance(replace(widget.state(), title="Noise of pore 3"))
    assert "α = " in w.fit_label_text()


def test_subpopulation_figures_take_in_figure_edits(qapp, qtbot):
    from nanoscope.gui.subpop_window import SubpopReportWindow

    w = SubpopReportWindow(None)
    qtbot.addWidget(w)
    for view in (w._embed_view, w._wave_view):
        controller = figure_text_for(view._plot.getPlotItem())
        assert controller is not None and "title" in controller.targets()
    host = figure_text_for(w._embed_view._plot.getPlotItem()).host
    state = host.axis_state("bottom")
    assert not state.log_available and "linear axes" in state.log_reason
    host.apply_text("axis.left.label", TextStyle(text="PC 2"))
    assert w.appearance_state()["y_label"] == "PC 2"
