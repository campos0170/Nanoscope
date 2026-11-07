"""The fit label (``gui.fit_label``): honest text, live numbers, drag, old files.
Synthetic data only."""
from __future__ import annotations

import json

import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import appearance_text_host
from nanoscope.gui.fit_label import FitLabelItem, FitLabelState, FitReadout, format_fit_label
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_interaction import install_plot_interaction

ZERO_DOF = "R² undefined: zero degrees of freedom"


def _plain(lines):
    doc = QtGui.QTextDocument()
    out = []
    for text, _colour in lines:
        doc.setHtml(text)
        out.append(doc.toPlainText())
    return out


def test_no_template_drops_the_r2_refusal_tag():
    """Plan G2 test 10."""
    readout = FitReadout({"equation": "y = 2·x + 1", "n": "n = 2"}, refusals=(ZERO_DOF,))
    for template in ("", "My fit", "{equation}", " "):
        text = _plain(format_fit_label([readout], FitLabelState(template=template)))
        assert text and ZERO_DOF in text[0], template
    # {r2} of a fit with no R² prints why, once
    assert _plain(format_fit_label([readout], FitLabelState(template="Fit: {r2}"))) == [
        f"Fit: {ZERO_DOF}"]
    # R² not asked for at all: no tag (nothing claims an R²)
    assert ZERO_DOF not in _plain(format_fit_label(
        [readout], FitLabelState(fields=("equation", "n"))))[0]


def test_a_refit_keeps_the_users_words_with_new_numbers():
    """Plan G2 test 11."""
    state = FitLabelState(template="Capture rate {equation} ({r2}) & more")
    first = _plain(format_fit_label([FitReadout({"equation": "y = 2·x", "r2": "R² = 0.9"})], state))
    second = _plain(format_fit_label([FitReadout({"equation": "y = 3·x", "r2": "R² = 0.8"})], state))
    assert first == ["Capture rate y = 2·x (R² = 0.9) & more"]
    assert second == ["Capture rate y = 3·x (R² = 0.8) & more"]


def test_two_sources_give_two_lines_in_their_colours_prefixed_by_name():
    lines = format_fit_label([FitReadout({"equation": "a"}, source="Apo", colour="#0072b2"),
                              FitReadout({"equation": "b"}, source="Holo", colour="#009e73")],
                             FitLabelState(fields=("equation",)))
    assert _plain(lines) == ["Apo: a", "Holo: b"]
    assert [colour for _text, colour in lines] == ["#0072b2", "#009e73"]


def test_a_source_name_draws_its_sub_and_superscripts_as_the_legend_does():
    """Review FIG-3: the legend drew 'τ_{1}' as τ with a subscript 1, the fit label's
    prefix and its {source} token printed the raw braces. Numbers stay escaped text."""
    readouts = [FitReadout({"equation": "y = x", "n": "n = <3> & 4"}, source="a_{1} <b> & c^{2}"),
                FitReadout({"equation": "y = 2x"}, source="b")]
    lines = format_fit_label(readouts, FitLabelState(fields=("equation", "n")))
    assert lines[0][0] == ("a<sub>1</sub> &lt;b&gt; &amp; c<sup>2</sup>: "
                           "y = x · n = &lt;3&gt; &amp; 4")
    assert _plain(lines) == ["a1 <b> & c2: y = x · n = <3> & 4", "b: y = 2x"]
    token = format_fit_label(readouts, FitLabelState(template="{source}: {equation} {n}"))
    assert token[0][0] == "a<sub>1</sub> &lt;b&gt; &amp; c<sup>2</sup>: y = x n = &lt;3&gt; &amp; 4"
    # A refusal is still the fit's own words, escaped, never markup.
    refused = FitReadout({"equation": "y_{0}"}, refusals=("R² undefined: n_{0} < 3",), source="s")
    assert _plain(format_fit_label([refused], FitLabelState(template="{source} {r2}"))) == [
        "s R² undefined: n_{0} < 3"]


def test_old_files_load_the_label_visible_with_the_mapped_fields():
    """Plan G2 test 13: before the label the fit was in the title, so it stays on screen."""
    state = FitLabelState.from_mapping(None, legacy={"fit_values_on_plot": False, "show_r2": True})
    assert state.visible and state.fields == ("equation", "r2", "n")
    assert FitLabelState.from_mapping(None, legacy={"show_r2": False, "show_gof": True}).fields == (
        "equation", "n", "ks")
    assert PlotAppearance.from_mapping({"title": "x"}).fit_label_state() == FitLabelState()
    saved = FitLabelState(template="Fit {r2}", placement="custom", offset=(40.0, 30.0))
    assert FitLabelState.from_mapping(saved.to_dict()) == saved


def _drag(view, start, end):
    vp = view.viewport()
    a, b = view.mapFromScene(start), view.mapFromScene(end)
    no = QtCore.Qt.MouseButton.NoButton
    left = QtCore.Qt.MouseButton.LeftButton
    mod = QtCore.Qt.KeyboardModifier.NoModifier

    def send(kind, pos, button, held):
        # pyqtgraph's scene drops mouse moves closer together than its rate limit.
        QtTest.QTest.qWait(25)
        p = QtCore.QPointF(pos)
        QtWidgets.QApplication.sendEvent(vp, QtGui.QMouseEvent(
            kind, p, p, QtCore.QPointF(vp.mapToGlobal(pos)), button, held, mod))
        QtWidgets.QApplication.processEvents()

    send(QtCore.QEvent.Type.MouseMove, a, no, no)
    send(QtCore.QEvent.Type.MouseButtonPress, a, left, left)
    for i in range(1, 6):
        send(QtCore.QEvent.Type.MouseMove, a + (b - a) * i / 5, no, left)
    send(QtCore.QEvent.Type.MouseButtonRelease, b, left, no)


def test_dragging_moves_the_label_not_the_view_and_saves_one_step(qtbot):
    """Plan G2 test 12, and the saved position through the generic host (Q-T5)."""
    figure_text.GATE.reset()
    widget = pg.PlotWidget(background="w")
    appearance = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    qtbot.addWidget(appearance)
    widget.resize(700, 450)
    plot = widget.getPlotItem()
    plot.plot([0, 1, 2], [0, 1, 4])
    host = appearance_text_host(plot, appearance)
    install_plot_interaction(plot, text=host)
    label = FitLabelItem(plot.getViewBox())
    label.apply_state(FitLabelState(), [FitReadout({"equation": "y = x²", "n": "n = 3"})])
    host.register_fit_label(label)
    widget.show()
    qtbot.waitExposed(widget)
    QtWidgets.QApplication.processEvents()
    moved = []
    label.moved.connect(moved.append)
    plot.getViewBox().setRange(xRange=(0, 2), yRange=(0, 4), padding=0)   # no late auto-range
    qtbot.wait(50)
    before_range = plot.getViewBox().viewRange()
    start = label.sceneBoundingRect().center()
    before = label.offset()
    _drag(widget, start, start + QtCore.QPointF(120, 60))
    assert len(moved) == 1
    after = label.offset()
    assert after[0] - before[0] == pytest.approx(120, abs=2)
    assert after[1] - before[1] == pytest.approx(60, abs=2)
    assert plot.getViewBox().viewRange() == before_range
    state = appearance.state().fit_label_state()
    assert state.placement == "custom" and state.offset == pytest.approx(moved[0])
    assert appearance._history.depth() == (1, 0)
    assert appearance.undo_button.toolTip() == "Undo: fit label moved"


@pytest.mark.parametrize("placement", ["top-left", "top-right", "bottom-left", "bottom-right", "custom"])
def test_fit_readout_wraps_to_narrow_plot_without_losing_words_or_saved_position(qtbot, placement):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 450)
    widget.show()
    qtbot.waitExposed(widget)
    view = widget.getPlotItem().getViewBox()
    state = FitLabelState(placement=placement, offset=(30.0, 25.0) if placement == "custom" else None)
    readouts = [FitReadout({"equation": "1/f fit: alpha = -0.026; descriptive model over the measured spectrum",
                           "r2": "R² = 0.000; descriptive fit only", "n": "81 bins; 1.22–100 Hz"},
                          ("R² is descriptive, not a mechanistic validation",))]
    label = FitLabelItem(view)
    label.apply_state(state, readouts)
    QtWidgets.QApplication.processEvents()
    narrow_height = label.boundingRect().height()
    text = label.plain_text()
    assert "81 bins" in text and "not a mechanistic validation" in text
    assert view.boundingRect().adjusted(-1, -1, 1, 1).contains(label.mapRectToParent(label.boundingRect()))
    if placement == "custom":
        assert label.offset() == pytest.approx(state.offset)
    widget.resize(1000, 450)
    QtWidgets.QApplication.processEvents()
    assert label.boundingRect().height() < narrow_height
    assert label.plain_text() == text and label.state() == state
    widget.resize(320, 450)
    QtWidgets.QApplication.processEvents()
    assert view.boundingRect().adjusted(-1, -1, 1, 1).contains(label.mapRectToParent(label.boundingRect()))
    assert label.plain_text() == text and label.state() == state


# -- other dragged annotations (build 5: the provenance caption's saved place) --------------

CAPTION = "annotation.provenance"


class _Caption(FitLabelItem):
    """A window's own annotation with the fit label's drag (as Voltage dependence's
    provenance caption is): drawn at the appearance's offset, else bottom-left."""

    def show_caption(self, text: str, offset) -> None:
        self._state = (FitLabelState(placement="custom", offset=offset) if offset is not None
                       else FitLabelState(placement="bottom-left"))
        self.setText(text, justify="left")
        self.setVisible(True)
        self._place()


class _CaptionWindow(QtWidgets.QWidget):
    def __init__(self, state=None):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.view = pg.PlotWidget(background="w")
        self.plot = self.view.getPlotItem()
        self.plot.plot([0, 1, 2], [0, 1, 4])
        self.appearance = PlotAppearanceWidget(state)
        self.appearance.changed.connect(self.redraw)
        layout.addWidget(self.view, 3)
        layout.addWidget(self.appearance, 1)
        self.host = appearance_text_host(self.plot, self.appearance)
        install_plot_interaction(self.plot, text=self.host)
        self.caption = _Caption(self.plot.getViewBox())
        self.host.register_annotation(self.caption, CAPTION)
        self.redraw(self.appearance.state())
        self.resize(900, 450)

    def redraw(self, state):
        self.caption.show_caption("Pooled: 3 recordings, merged", state.annotation_offset(CAPTION))


def test_a_dragged_caption_is_one_undo_step_and_reopens_where_it_was_left(qtbot):
    """INTEGRATE §3 (G4, major): a drag of V-dep's provenance caption was neither saved
    with the project nor an Undo step. The place is ``PlotAppearance.annotations``."""
    figure_text.GATE.reset()
    w = _CaptionWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.plot.getViewBox().setRange(xRange=(0, 2), yRange=(0, 4), padding=0)
    qtbot.wait(50)
    before_range = w.plot.getViewBox().viewRange()
    home = w.caption.offset()
    start = w.caption.sceneBoundingRect().center()
    _drag(w.view, start, start + QtCore.QPointF(120, -80))
    dragged = w.appearance.state().annotation_offset(CAPTION)
    assert dragged is not None
    assert dragged[0] - home[0] == pytest.approx(120, abs=2)
    assert dragged[1] - home[1] == pytest.approx(-80, abs=2)
    assert w.caption.offset() == pytest.approx(dragged)
    assert w.plot.getViewBox().viewRange() == before_range, "the view did not pan"
    assert w.appearance._history.depth() == (1, 0)
    assert w.appearance.undo_button.toolTip() == "Undo: caption moved"
    saved = json.loads(json.dumps(w.appearance.state().to_dict()))
    assert saved == {"annotations": {CAPTION: {"offset": list(dragged)}}}
    assert w.appearance.undo()
    assert w.appearance.state().annotation_offset(CAPTION) is None
    assert w.caption.offset() == pytest.approx(home), "back at the window's own place"
    assert w.appearance.redo()
    assert w.caption.offset() == pytest.approx(dragged)
    # A project reopens it where it was left.
    again = _CaptionWindow(PlotAppearance.from_mapping(saved))
    qtbot.addWidget(again)
    again.show()
    qtbot.waitExposed(again)
    assert again.caption.offset() == pytest.approx(dragged)


def test_an_annotation_place_is_kept_by_templates_and_dropped_by_presets():
    from nanoscope.gui.plot_templates import decode_window_template, encode_window_template

    state = (PlotAppearance(title="Fig 2", title_size=14)
             .with_annotation_offset(CAPTION, (130.0, 228.5))
             .with_annotation_offset("annotation.fit", (40.0, 30.0)))
    assert state.annotation_offset(CAPTION) == (130.0, 228.5)
    assert state.annotation_offset("annotation.fit") == (40.0, 30.0)
    assert state.fit_label_state().placement == "custom", "the fit label keeps its own store"
    assert dict(state.annotations) == {CAPTION: (130.0, 228.5)}
    _settings, appearance = decode_window_template(
        "voltage-dependence", encode_window_template("voltage-dependence", {}, state.to_dict()))
    assert PlotAppearance.from_mapping(appearance).annotation_offset(CAPTION) == (130.0, 228.5)
    look = state.style_only()
    assert look.annotations is None and look.annotation_offset("annotation.fit") is None
    assert look.title_size == 14
    # A preset applied to a figure keeps that figure's places.
    assert state.restyled(PlotAppearance(title_size=10)).annotation_offset(CAPTION) == (130.0, 228.5)
    # Back to the window's own place; bad keys and offsets change nothing.
    assert state.with_annotation_offset(CAPTION, None).annotations is None
    assert state.with_annotation_offset("annotation.fit", None).fit_label_state().placement == "top-left"
    assert PlotAppearance().with_annotation_offset("annotation.fit", None) == PlotAppearance()
    assert state.with_annotation_offset("title", (1.0, 2.0)) == state
    assert state.with_annotation_offset(CAPTION, (float("nan"), 2.0)) == state
    loaded = PlotAppearance.from_mapping({"annotations": {
        CAPTION: {"offset": [10, 20]}, "annotation.fit": {"offset": [1, 2]}, "title": {"offset": [1, 2]},
        "annotation.x": {"offset": [1e9, 2]}, "annotation.y": [1, 2]}})
    assert loaded.annotations == ((CAPTION, (10.0, 20.0)),)
