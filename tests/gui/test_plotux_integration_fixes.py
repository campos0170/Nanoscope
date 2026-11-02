"""The plot UX round, integrated (2026-11-01): what the per-group reviews found once the
groups met, each pinned where a user meets it.

Presets carried by an older project, a pinned axis under a manual zoom, the fit label's
size through a project, Distributions' legend rows, Undo grain and Axis properties, and
the windows whose views share one X and one Y label. Synthetic data only; offscreen.
Clicks go through the QWINDOW (``window().windowHandle()``), as a user's do.
"""
from __future__ import annotations

import json
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisState, TextStyle, figure_text_for
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plots_panel import PlotsWindow

LEFT = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _df(seed: int = 0, n: int = 300, slope: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({
        "xcol": x, "ycol": slope * x + 5.0 + rng.normal(0.0, 0.3, n),
        "dwell_s": rng.exponential(0.002, n) + 1e-4,
        "depth_pA": rng.normal(50.0, 5.0, n),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


def _window(qtbot, frames=None) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1200, 820)
    w.show()
    qtbot.waitExposed(w)
    w.set_data(frames or {"probe": _df()})
    if frames:
        w.set_sources(list(frames))
    return w


def _scatter(qtbot, frames=None) -> PlotsWindow:
    w = _window(qtbot, frames)
    w.set_plot_type("Scatter")
    w.set_scatter_columns("xcol", "ycol")
    w.set_scatter_fit("linear")
    return w


def _at(w, item):
    view = w._plot_widget
    pos = view.mapFromScene(item.sceneBoundingRect().center())
    return view.viewport().mapTo(view.window(), pos)


def _depth(w) -> int:
    return w._history.depth()[0]


# -- presets and the fit label's model -----------------------------------------------------------

def test_an_older_projects_preset_with_a_title_is_not_added_twice(qapp):
    """An older build saved presets with their wording. Presets are style only now, so the
    carried copy must be read style only too, or the same preset arrives as a second
    "(from project)" one with a banner saying they differ."""
    from nanoscope.core.plot_preset_state import APPEARANCE_PRESETS, WORKSPACE_KEY
    from nanoscope.gui.plot_appearance import PlotAppearanceWidget
    from nanoscope.gui.plot_preset_sync import install_project_presets

    old = {"title": "IV of pore 3", "title_size": 14.0, "background": "light"}
    settings = QtCore.QSettings()
    settings.setValue(PlotAppearanceWidget._PRESETS_KEY, json.dumps({"look": old}))
    settings.sync()
    try:
        notes = install_project_presets({WORKSPACE_KEY: {APPEARANCE_PRESETS: {"look": dict(old)}}})
        stored = json.loads(QtCore.QSettings().value(PlotAppearanceWidget._PRESETS_KEY))
        assert list(stored) == ["look"], stored
        assert not notes, notes
    finally:
        QtCore.QSettings().remove(PlotAppearanceWidget._PRESETS_KEY)


def test_a_fit_label_size_from_text_style_survives_a_project_round_trip():
    """Text style… offers 4–72 pt; the label kept only 6–30 when a project reopened."""
    for size in (5.0, 40.0):
        state = PlotAppearance().with_text_style("annotation.fit", TextStyle(size_pt=size))
        assert state.fit_label_state().size_pt == size
        assert PlotAppearance.from_mapping(state.to_dict()) == state


# -- Distributions -------------------------------------------------------------------------------

def test_a_legend_row_is_renamed_by_a_double_click_on_the_figure(qapp, qtbot):
    """The legend is created after the text controller; its rows must still be targets."""
    w = _scatter(qtbot, {"Apo": _df(0), "Holo": _df(1, slope=3.0)})
    controller = figure_text_for(w._plot)
    key = next(k for k in controller.targets() if k.startswith("legend.entry."))
    row = controller.targets()[key].item
    back = _depth(w)
    QTest.mouseDClick(w._plot_widget.window().windowHandle(), LEFT, NOMOD, _at(w, row))
    qapp.processEvents()
    assert controller.editor is not None, "a double-click on a legend row opened nothing"
    controller.editor.selectAll()
    QTest.keyClicks(controller.editor, "A & B")
    QTest.keyClick(controller.editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    rows = [label.item.toPlainText() for _s, label in w._plot.legend.items]
    assert any(t.startswith("A & B") for t in rows), rows
    assert _depth(w) == back + 1


@pytest.mark.parametrize("box", ["title", "fit_label_template"])
def test_typing_in_a_text_box_then_return_is_one_undo_step(qapp, qtbot, box):
    """D13: the first keystroke used to be a step of its own (the widget spells colours
    '#0072b2', the window '#0072B2'), and every keystroke in the fit-label template was."""
    w = _scatter(qtbot)
    edit = getattr(w._appearance_widget, box)
    edit.setFocus()
    back = _depth(w)
    QTest.keyClicks(edit, "Fig {n}")
    QTest.keyClick(edit, QtCore.Qt.Key.Key_Return)
    qtbot.wait(600)                  # past the coalescing timer
    assert _depth(w) == back + 1
    w._history_undo() if hasattr(w, "_history_undo") else w._undo_btn.click()
    assert _depth(w) == back


def test_the_numbers_row_still_adds_bic_after_the_label_is_saved(qapp, qtbot):
    w = _window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    w._appearance_widget.apply_appearance(w._appearance_widget.state().with_text_style(
        "annotation.fit", TextStyle(bold=True)))      # any Fit label edit saves the label
    assert w.config().style.fit_label
    assert "BIC" not in w._fit_label_item.plain_text()
    bic = w._appearance_widget.fit_label_fields["bic"]
    bic.click()
    assert "BIC" in w._fit_label_item.plain_text()
    bic.click()
    assert "BIC" not in w._fit_label_item.plain_text()


def test_axis_properties_range_and_log_in_one_ok_are_one_undo_step(qapp, qtbot):
    w = _scatter(qtbot)
    controller = figure_text_for(w._plot)
    back = _depth(w)
    pop = controller.open_axis_properties("bottom")
    assert not pop.major.isVisibleTo(pop), "Distributions keeps no major tick spacing"
    pop.auto.setChecked(False)
    pop.lo.setText("1.5")
    pop.hi.setText("4.5")
    pop.log.setChecked(True)
    pop.accept()
    qapp.processEvents()
    assert w._scatter_logx.isChecked() and w.config().axes.manual_x is not None
    assert _depth(w) == back + 1


def test_axis_properties_ok_with_nothing_changed_keeps_the_pin_exact(qapp, qtbot):
    w = _scatter(qtbot)
    w._range_edits["xmin"].setText("1.23456789")
    w._range_edits["xmax"].setText("4.87654321")
    w._apply_axis_range()
    w._history_touch()
    pin = w.config().axes.manual_x
    back = _depth(w)
    figure_text_for(w._plot).open_axis_properties("bottom").accept()
    qapp.processEvents()
    assert w.config().axes.manual_x == pin
    assert _depth(w) == back


# -- the PlotAppearance windows ------------------------------------------------------------------

def test_a_pinned_axis_keeps_a_manual_zoom_through_an_unrelated_appearance_step(qapp, qtbot,
                                                                                 tmp_path):
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session

    w = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w._text_host.apply_axis("bottom", AxisState(range=(-900.0, -100.0)))
    view = w.plot_item().getViewBox()
    assert view.viewRange()[0] == pytest.approx([-0.9, -0.1])
    view.setXRange(-0.6, -0.4, padding=0)                  # the user zooms in
    widget = w._appearance_widget
    widget.apply_appearance(PlotAppearance.from_mapping({**widget.state().to_dict(),
                                                         "title_size": 16}))
    assert view.viewRange()[0] == pytest.approx([-0.6, -0.4])
    widget.undo()
    assert view.viewRange()[0] == pytest.approx([-0.6, -0.4])


def test_in_legend_instead_is_not_offered_where_the_legend_cannot_carry_the_numbers(
        qapp, qtbot, tmp_path):
    """I-V's legend row is "I-V fit": with the label in the legend, G, R² and the dof tag
    left the figure. The box is not offered, and a state that asks for it still draws."""
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session

    w = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    widget = w._appearance_widget
    assert widget.fit_label_group.isVisibleTo(widget)
    assert not widget.fit_label_in_legend.isVisibleTo(widget)
    text = w.fit_label_text()
    assert "G (I-V slope)" in text
    state = widget.state()
    widget.apply_appearance(PlotAppearance.from_mapping(
        {**state.to_dict(), "fit_label": {**state.fit_label_state().to_dict(), "in_legend": True}}))
    assert w.fit_label_text() == text


def test_a_shared_axis_label_is_not_renamed_from_one_view(qapp, qtbot):
    """Noise's views share one X and one Y label: renaming the spectrogram's y label on the
    figure relabelled the PSD's 'PSD (pA²/Hz)' too. The label's text is renamed in
    Appearance (this view); on a figure it offers Text style only, and Axis properties
    shows it read-only with the reason."""
    from nanoscope.gui.psd_panel import PSDWindow

    w = PSDWindow(Session())
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    for plot in (w._plot, w._spec_plot):
        controller = figure_text_for(plot)
        target = controller.targets()["axis.left.label"]
        assert target.capabilities == frozenset({"style"})
        titles = [title for title, _cb in controller.menu_entries("axis.left.label")]
        assert "Edit label…" not in titles and "Text style…" in titles
        assert controller.host.axis_state("left").label_locked
    before = w._appearance_widget.state().y_label
    pop = figure_text_for(w._spec_plot).open_axis_properties("left")
    assert pop.label.isReadOnly() and "share one X and one Y label" in pop.label.toolTip()
    pop.label.setText("f (Hz)")
    pop.accept()
    assert w._appearance_widget.state().y_label == before
