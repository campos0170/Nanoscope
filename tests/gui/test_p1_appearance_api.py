"""Phase 1 Foundation, F2: the additive appearance API the toolbar, Style ▾ and Format use.

``historyChanged``, ``restored``, ``history_mark``, ``apply_look``, ``reset_figure``,
``edit_text_style``, ``restyled(keep_hand=)``, the four journal looks, the row-label registry.
Synthetic; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import fields

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.plot_appearance import (_BUILTIN_PRESETS, APPEARANCE_COALESCE_MS,
                                           PlotAppearance, PlotAppearanceWidget)

NATURE = "7 pt Arial · Nature, Elsevier"
JOURNAL = (NATURE, "8 pt Arial · PLOS, Springer, AIP", "9 pt Times New Roman · IEEE",
           "Slides · 18 pt Arial")
PAUSE_MS = APPEARANCE_COALESCE_MS + 150


@pytest.fixture
def widget(qtbot):
    w = PlotAppearanceWidget()
    qtbot.addWidget(w)
    return w


def _counter(signal):
    box = {"n": 0}
    signal.connect(lambda *_a: box.__setitem__("n", box["n"] + 1))
    return box


def test_history_changed_fires_on_edit_undo_redo_and_reset(widget):
    seen = _counter(widget.historyChanged)
    widget.title.textEdited.emit("T")
    assert seen["n"] >= 1
    for act in (widget.undo, widget.redo, widget.reset_figure):
        before = seen["n"]
        act()
        assert seen["n"] > before, act.__name__
    # the buttons had already moved when it fired
    fresh = PlotAppearanceWidget()
    fresh.title.textEdited.emit("T")
    fresh._close_burst()
    states = []
    fresh.historyChanged.connect(
        lambda: states.append((fresh.undo_button.isEnabled(), fresh.redo_button.isEnabled())))
    fresh.undo()
    assert states and states[-1] == (False, True)


def test_restored_fires_for_an_install_from_outside_and_not_for_the_widgets_own_announce(widget):
    restored = _counter(widget.restored)
    widget.title.textEdited.emit("typed")                 # the widget's own announce
    widget.apply_look(PlotAppearance(font_family="Arial"))
    widget.undo()
    assert restored["n"] == 0
    widget.set_state(PlotAppearance(title="from outside"))
    assert restored["n"] == 1
    # a host that normalises inside ``changed`` is the widget's own announce too
    widget.changed.connect(lambda s: widget.set_state(s))
    widget.title.textEdited.emit("again")
    assert restored["n"] == 1


def test_history_mark_counts_steps_an_open_run_and_the_ceiling(widget):
    assert widget.history_mark() == 0
    widget.apply_look(PlotAppearance(font_family="Arial"))
    assert widget.history_mark() == 1
    widget.title.textEdited.emit("x")                     # an open run counts, as Undo does
    assert widget.history_mark() == 2
    widget.undo()
    assert widget.history_mark() == 1
    bare = PlotAppearanceWidget(history=False)
    assert bare.history_mark() == 0 and bare.undo_button is None


def test_history_mark_keeps_growing_past_the_ceiling(widget):
    from nanoscope.gui.plot_appearance import APPEARANCE_HISTORY_DEPTH
    for i in range(APPEARANCE_HISTORY_DEPTH + 5):
        widget.apply_look(PlotAppearance(line_width=0.25 + 0.01 * i))
        assert widget.history_mark() == i + 1, i
    assert widget._history.depth()[0] == APPEARANCE_HISTORY_DEPTH      # the stack itself is capped


def test_apply_look_is_one_step_and_one_undo_restores(widget):
    widget.title.textEdited.emit("My title")
    widget._close_burst()
    before = widget.state()
    steps = widget._history.depth()[0]
    look = _BUILTIN_PRESETS[NATURE]
    assert widget.apply_look(look) is True
    assert widget._history.depth()[0] == steps + 1
    assert widget.state().title_size == 7 and widget.state().title == "My title"
    assert "preset" in widget.undo_button.toolTip()
    assert widget.undo() is True and widget.state() == before
    assert widget.apply_look(look) is True
    assert widget.apply_look(look) is False               # already that look: no state moved


def test_apply_look_announces_even_when_the_controls_already_show_the_state(widget):
    heard = []
    widget.changed.connect(heard.append)
    look = _BUILTIN_PRESETS[NATURE]
    widget.apply_look(look)
    widget.apply_look(look)                  # the host may lag the controls: it must still hear it
    assert len(heard) == 2 and heard[0] == heard[1]
    assert widget._history.depth()[0] == 1   # ...and the second records no step


def test_keep_hand_keeps_a_hand_set_title_style_and_off_replaces_it():
    hand = PlotAppearance().with_text_style("title", TextStyle(bold=True, size_pt=16))
    look = _BUILTIN_PRESETS[NATURE].with_text_style("title", TextStyle(size_pt=9, italic=True))
    kept = hand.restyled(look, keep_hand=True).text_style("title")
    # field by field: the hand's bold and 16 pt win, the look's italic fills the gap
    assert (kept.bold, kept.size_pt, kept.italic) == (True, 16.0, True)
    replaced = hand.restyled(look).text_style("title")
    assert (replaced.bold, replaced.size_pt, replaced.italic) == (None, 9.0, True)  # today's


def test_keep_hand_default_matches_todays_restyled_exactly():
    hand = PlotAppearance(title="t").with_text_style("axis.left.label", TextStyle(bold=True))
    for look in _BUILTIN_PRESETS.values():
        assert hand.restyled(look) == hand.restyled(look, keep_hand=False)


def test_keep_hand_does_not_touch_legend_entries_or_words():
    hand = (PlotAppearance(title="Words", x_label="X")
            .with_text_style("legend.entry.a", TextStyle(bold=True)))
    out = hand.restyled(_BUILTIN_PRESETS[NATURE], keep_hand=True)
    assert out.title == "Words" and out.x_label == "X"
    assert out.text_style("legend.entry.a").bold is True


def test_the_four_journal_looks_follow_the_two_shipped_ones(widget):
    names = list(_BUILTIN_PRESETS)
    assert names[:2] == ["Publication · light", "Presentation · dark"]
    assert names[2:] == list(JOURNAL)
    # the combo is the same list behind its placeholder, so indexes 1 and 2 did not move
    assert [widget.preset.itemText(i) for i in range(widget.preset.count())] == \
        ["Choose appearance preset…", *names]
    sizes = {n: (p.title_size, p.axis_size, p.tick_size, p.legend_size)
             for n, p in _BUILTIN_PRESETS.items() if n in JOURNAL}
    assert sizes[NATURE] == (7, 7, 6, 6) and sizes["Slides · 18 pt Arial"] == (18, 18, 16, 16)
    assert _BUILTIN_PRESETS["9 pt Times New Roman · IEEE"].font_family == "Times New Roman"


def test_a_journal_look_has_no_fit_label_and_no_element_style_outside_the_figure_wide_fields():
    for name in JOURNAL:
        look = _BUILTIN_PRESETS[name]
        assert look.fit_label is None and look.text is None, name
        assert look.axes is None and look.annotations is None, name
        assert look.title is None and look.x_label is None and look.y_label is None
        assert look.style_only() == look, name         # it IS a look: nothing a look drops
        for f in fields(PlotAppearance):
            if f.name not in ("font_family", "title_size", "axis_size", "tick_size",
                              "legend_size", "grid", "background", "line_width", "marker_size"):
                assert getattr(look, f.name) is None, (name, f.name)


def test_choosing_a_journal_look_in_the_combo_is_one_named_step(widget):
    widget.preset.activated.emit(widget.preset.findText(NATURE))
    assert widget.state().title_size == 7
    assert widget.undo_button.toolTip() == f"Undo: {NATURE} preset"


def test_a_figure_with_a_look_applied_still_composes_live(qtbot):
    import pyqtgraph as pg
    from nanoscope.gui.figure_composition import capture_cartesian_model
    from nanoscope.gui.plot_appearance import apply_plot_appearance
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    plot.plot([0, 1, 2], [1.0, 3.0, 2.0])
    assert capture_cartesian_model(plot) is not None
    for name in JOURNAL:
        apply_plot_appearance(plot, _BUILTIN_PRESETS[name], plot_widget=widget)
        assert capture_cartesian_model(plot) is not None, name


def test_edit_text_style_coalesces_a_run_into_one_step_and_one_undo_takes_it_back(qtbot, widget):
    steps = widget._history.depth()[0]
    for size in (8, 9, 10, 11):
        widget.edit_text_style("title", TextStyle(size_pt=size, bold=True))
    assert widget.state().text_style("title").size_pt == 11.0
    assert widget._history.depth()[0] == steps          # the run is still open
    qtbot.wait(PAUSE_MS)
    assert widget._history.depth()[0] == steps + 1      # ...and closed as ONE step
    assert widget.undo() is True
    assert widget.state().text_style("title") == TextStyle()
    # words typed through the same route reach the Title box
    widget.edit_text_style("title", TextStyle(text="Typed"))
    assert widget.title.text() == "Typed"
    # a no-op edit and a bad key record nothing
    widget._close_burst()
    depth = widget._history.depth()
    widget.edit_text_style("title", TextStyle(text="Typed"))
    widget.edit_text_style("nonsense", TextStyle(bold=True))
    assert widget._history.depth() == depth


def test_reset_figure_is_the_buttons_own_handler_and_one_step(widget):
    widget.apply_look(_BUILTIN_PRESETS[NATURE])
    steps = widget._history.depth()[0]
    widget.reset_figure()
    assert widget._history.depth()[0] == steps + 1 and widget.state().to_dict() == {}
    widget.apply_look(_BUILTIN_PRESETS[NATURE])
    widget.reset_button.click()
    assert widget.state().to_dict() == {}


def test_every_row_control_names_its_label(widget):
    for name in ("font_family", "background", "title_size", "axis_size", "tick_size",
                 "line_width", "marker_size", "grid", "data_color", "fit_color",
                 "title", "x_label", "y_label", "legend_visible", "legend_size"):
        control = getattr(widget, name)
        label = control._nanoscope_row_label
        assert isinstance(label, QtWidgets.QLabel) and label.text(), name
    assert widget.title._nanoscope_row_label.text() == "Title"
    assert widget.marker_size._nanoscope_row_label.text() == "Marker size"
    assert widget.legend_size._nanoscope_row_label.text() == "Legend text (pt)"


# -- the Format group's additions (G2, post-F ownership of this file) ---------------------------------------

def test_a_font_picker_with_an_inherit_label_has_exactly_one_more_row_and_the_rest_untouched(qtbot):
    from nanoscope.gui.font_picker import FontPicker
    from nanoscope.plot import figure_fonts as ff

    plain, inheriting = FontPicker(), FontPicker(inherit_label="Figure font")
    qtbot.addWidget(plain)
    qtbot.addWidget(inheriting)
    assert inheriting.count() == plain.count() + 1
    assert inheriting.itemText(0) == "Figure font" and inheriting.itemData(0) == ""
    # the four main rows follow; Arial is an explicit family here, not "unset"
    assert [inheriting.itemData(i) for i in range(1, 5)] == list(ff.FIGURE_FAMILIES)
    assert [plain.itemData(i) for i in range(4)] == ["", *ff.FIGURE_FAMILIES[1:]]     # untouched
    assert plain.itemText(0).startswith("Arial (default)")


def test_the_inherit_row_and_an_explicit_arial_are_two_different_values(qtbot):
    from nanoscope.gui.font_picker import FontPicker

    picker = FontPicker(inherit_label="Figure font")
    qtbot.addWidget(picker)
    seen = []
    picker.changed.connect(seen.append)
    picker.set_value(None)                                  # silent, and the inherit row
    assert picker.currentIndex() == 0 and picker.value() == "" and seen == []
    picker.set_value("Arial")
    assert picker.currentIndex() == 1 and picker.value() == "Arial" and seen == []
    picker.setCurrentIndex(3)                               # Times New Roman
    assert seen == ["Times New Roman"]
    picker.setCurrentIndex(0)
    assert seen == ["Times New Roman", ""] and picker.value() == ""
    picker.set_value("Menlo")                               # one extra row, under the four main ones
    assert picker.has_extra_row() and picker.currentIndex() == 5 and picker.value() == "Menlo"
    picker.set_value("Helvetica")
    assert picker.currentIndex() == 2 and picker.value() == "Helvetica"


def test_a_run_write_through_the_controller_is_one_step_and_a_plain_write_is_one_each(qtbot):
    from dataclasses import replace

    from nanoscope.gui.figure_text import appearance_text_host, attach_figure_text
    import pyqtgraph as pg

    widget = PlotAppearanceWidget(PlotAppearance(title="T"))
    qtbot.addWidget(widget)
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    controller = attach_figure_text(plot, appearance_text_host(plot, widget))
    mark = widget.history_mark()
    for size in (5.0, 6.0, 7.0, 8.0):
        controller.set_style_property("title", "size_pt", size, run=True)
    assert widget.history_mark() == mark + 1 and widget.state().text_style("title").size_pt == 8.0
    assert widget.undo() and widget.state().text_style("title").size_pt is None
    controller.set_style_property("title", "bold", True)
    controller.set_style_property("title", "italic", True)
    assert widget.history_mark() == mark + 2                    # each plain write closes its own step
    # an empty family is "follow the figure", and the words are never touched
    controller.set_style_property("title", "family", "")
    style = widget.state().text_style("title")
    assert style.family is None and style.text == "T" and style.bold and style.italic


def test_a_property_write_starts_from_the_current_style_so_the_fit_labels_wording_survives(qtbot):
    from nanoscope.gui.figure_text import appearance_text_host, attach_figure_text
    from nanoscope.gui.fit_label import FitLabelItem, FitLabelState
    import pyqtgraph as pg

    widget = PlotAppearanceWidget(PlotAppearance(fit_label=FitLabelState(
        template="{equation} fitted", size_pt=12.0, color="#d55e00")))
    qtbot.addWidget(widget)
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    host = appearance_text_host(plot, widget)
    host.register_fit_label(FitLabelItem(plot.getPlotItem().getViewBox()))
    controller = attach_figure_text(plot, host)
    controller.set_style_property("annotation.fit", "italic", True)
    label = widget.state().fit_label_state()
    assert (label.template, label.size_pt, label.color) == ("{equation} fitted", 12.0, "#d55e00")
    assert widget.state().text_style("annotation.fit").italic is True
    # the trap itself: a bare TextStyle through the same host DOES clear them
    host.apply_text("annotation.fit", TextStyle(italic=True))
    assert widget.state().fit_label_state().template == ""


def test_inheriting_picker_preserves_an_old_compositions_private_family(qtbot):
    from nanoscope.gui.font_picker import FontPicker, PRIVATE_ROW_LABEL

    picker = FontPicker(inherit_label="Figure font")
    qtbot.addWidget(picker)
    picker.set_value(".AppleSystemUIFont")
    assert picker.currentIndex() == 5
    assert picker.currentText() == PRIVATE_ROW_LABEL
    assert picker.value() == ".AppleSystemUIFont"
    picker.set_value(None)
    assert picker.currentIndex() == 0 and picker.value() == ""
