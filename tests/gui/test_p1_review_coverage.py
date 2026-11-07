"""Phase 1 Foundation, review round: behaviours the first test files left to chance.

Each test names a hunk that could be deleted or flipped with every earlier test staying
green: the native adapters' history wiring on the six workflow-bar pages, the Explorer's one refresh
route, the Format tab's text-style route, the design's look values, the reflowed controls, the host's retire/reveal guards, ``form_rows``, explicit export kinds and
the toolbar's focus and sync. Real clicks and keys; synthetic data; offscreen.
"""
from __future__ import annotations

import os
import sys

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui.analysis_inspector import GroupFrame, InspectorHost, InspectorSettings
from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.format_inspector import ElementRef
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS, PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_workflow import InspectorGroup, form_rows

from . import p1_hosting as H

BAR_PAGES = ("distribution_summaries", "comparison_figures", "event_waveforms", "noise",
             "voltage_series", "subpopulations")
SLIDES = "Slides · 18 pt Arial"
RETURN = QtCore.Qt.Key.Key_Return


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


# -- the journal looks are design 6.4's table -------------------------------------------------------

#: name -> (font, title, axis, ticks, legend, line width, marker, grid, background)
LOOK_TABLE = {
    "7 pt Arial · Nature, Elsevier": ("Arial", 7, 7, 6, 6, 0.75, 3, False, "light"),
    "8 pt Arial · PLOS, Springer, AIP": ("Arial", 8, 8, 7, 7, 1, 4, False, "light"),
    "9 pt Times New Roman · IEEE": ("Times New Roman", 9, 9, 8, 8, 1, 4, False, "light"),
    "Slides · 18 pt Arial": ("Arial", 18, 18, 16, 16, 3, 12, False, "light"),
}


@pytest.mark.parametrize("name", list(LOOK_TABLE))
def test_each_journal_look_is_its_row_of_the_design_table(name):
    look = _BUILTIN_PRESETS[name]
    got = (look.font_family, look.title_size, look.axis_size, look.tick_size, look.legend_size,
           look.line_width, look.marker_size, look.grid, look.background)
    assert got == LOOK_TABLE[name]
    plain = PlotAppearance()          # colours are the figure's own: a look never sets them
    assert (look.data_color, look.fit_color) == (plain.data_color, plain.fit_color)


# -- the fallback's history, look and reset wiring on the six workflow-bar pages -----------------------

@pytest.mark.parametrize("key", BAR_PAGES)
def test_a_bar_pages_look_undo_redo_and_reset_figure_reach_its_appearance_history(qtbot, builders, key):
    page = builders[key]()
    explorer = H.hosted(qtbot, page, key)
    widget, bar = page._appearance_widget, explorer.toolbar()
    start = widget.history_mark()
    result = explorer.apply_look(_BUILTIN_PRESETS[SLIDES])
    assert result.applied == (key,) and result.failed == ()
    assert widget.state().title_size == 18 and widget.history_mark() == start + 1
    assert bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    H.press_undo(explorer)
    assert widget.state().title_size is None and widget.history_mark() == start
    assert bar.redo_action.isEnabled()
    H.press_redo(explorer)
    assert widget.state().title_size == 18 and widget.history_mark() == start + 1
    styled = widget.state().to_dict()
    # Reset removes presentation overrides while retaining the quantities/units
    # and source identities needed to interpret this page's scientific content.
    retained = {name: styled[name] for name in ("quantity_display", "source_appearance")
                if name in styled}
    H.press_reset_figure(explorer)
    assert widget.state().to_dict() == retained and widget.history_mark() == start + 2
    H.press_undo(explorer)
    assert widget.state().to_dict() == styled and widget.history_mark() == start + 1
    H.press_redo(explorer)
    assert widget.state().to_dict() == retained and widget.history_mark() == start + 2


# -- the Explorer's one refresh route -----------------------------------------------------------------

def test_refresh_header_re_reads_the_front_pages_groups_and_format_tab(qtbot):
    flag = {"on": True}
    page = H.SyntheticPlotPage("iv")
    adapter = page.inspector_adapter()
    adapter.groups[:] = [InspectorGroup("estimator", "Estimator", [page._estimator]),
                         InspectorGroup("only_when", "Only when", [QtWidgets.QLabel("x")],
                                        visible_when=lambda: flag["on"])]
    explorer = H.hosted(qtbot, page, "iv")
    frame = explorer.inspector().pane("iv").data_pane.group("only_when")
    assert frame.isVisible()
    flag["on"] = False                       # a silent change: a project open emits nothing
    assert frame.isVisible()
    explorer.refresh_header()                # MainWindow calls this after a restore
    assert not frame.isVisible()


# -- the Format tab's text-style route -----------------------------------------------------------------

def test_edit_text_style_is_heard_by_the_host_and_moves_the_history_buttons(qtbot):
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    heard = []
    widget.changed.connect(heard.append)
    big = TextStyle(bold=True, size_pt=16)
    assert not widget.undo_button.isEnabled()
    widget.edit_text_style("title", big)
    assert len(heard) == 1 and heard[0].text_style("title").size_pt == 16.0
    assert widget.undo_button.isEnabled()                      # an open run is offered to Undo
    widget.edit_text_style("title", big)                       # the same value again: nothing
    assert len(heard) == 1
    widget._loading = True                                     # while the controls are being loaded
    widget.edit_text_style("title", TextStyle(italic=True))
    widget._loading = False
    assert len(heard) == 1 and widget.state().text_style("title").italic is None


def test_edit_text_style_on_a_widget_without_history_still_announces(qtbot):
    widget = PlotAppearanceWidget(history=False)
    qtbot.addWidget(widget)
    heard = []
    widget.changed.connect(heard.append)
    widget.edit_text_style("title", TextStyle(size_pt=12))
    assert len(heard) == 1 and widget.state().text_style("title").size_pt == 12.0
    assert widget.undo_button is None


# -- the reflowed controls ----------------------------------------------------------------

@pytest.mark.parametrize("key", ("voltage_series", "iv"))
def test_native_adapter_controls_fit_without_a_sideways_scrollbar(qtbot, builders, key):
    explorer = H.hosted(qtbot, builders[key](), key, size=(840, 560))
    scroll = explorer.inspector().pane(key).data_pane
    for group in explorer.adapter(key).groups:
        explorer.inspector().reveal(group.key)
    QtWidgets.QApplication.processEvents()
    assert scroll.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert scroll.horizontalScrollBar().maximum() == 0
    assert not scroll.horizontalScrollBar().isVisible()


# -- the host's retire / reveal / tab guards -----------------------------------------------------------

def _two_pages(qtbot):
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    for page in (a, b):
        qtbot.addWidget(page)
    host = InspectorHost()
    qtbot.addWidget(host)
    host.resize(360, 600)
    host.adopt("iv", a.inspector_adapter())
    host.adopt("voltage_series", b.inspector_adapter())
    host.show_page("voltage_series")
    host.show()
    qtbot.waitExposed(host)
    return a, b, host


def test_retiring_a_page_behind_the_front_one_leaves_the_front_page_showing(qtbot):
    _a, _b, host = _two_pages(qtbot)
    host.retire("iv")
    assert host.current_key() == "voltage_series" and host.isVisible()
    assert host.pane("iv") is None and host.pane("voltage_series") is not None


def test_retiring_the_front_page_clears_its_selection_and_hides_the_column(qtbot):
    _a, _b, host = _two_pages(qtbot)
    hub = host.selection_hub("voltage_series")
    hub.select(ElementRef("", "figure"))
    assert hub.current() is not None
    host.retire("voltage_series")
    assert hub.current() is None                 # nothing stays selected on a page that left
    assert host.current_key() is None and not host.isVisible()


def test_set_tab_with_no_page_in_front_is_remembered_and_announced(qtbot):
    host = InspectorHost()
    qtbot.addWidget(host)
    heard = []
    host.tabChanged.connect(heard.append)
    host.set_tab(1)
    assert host.current_tab() == 1 and heard == [1]
    host.set_tab(0)
    assert heard == [1, 0]


def test_revealing_a_data_group_on_a_page_behind_prepares_it_and_leaves_the_front_tab(qtbot):
    a, _b, host = _two_pages(qtbot)
    host.set_tab(1)
    assert host.reveal("@format", key="iv") is True
    assert host.pane("iv").currentIndex() == 1
    assert host.reveal("estimator", key="iv") is True        # a Data group, page behind
    assert host.pane("iv").currentIndex() == 0               # that page is on its Data tab ...
    assert host.current_tab() == 1                           # ... the visible one did not move
    assert host.pane("iv").data_pane.group("estimator").is_expanded()


def test_a_group_hidden_by_the_page_can_be_shown_again(qtbot):
    _a, _b, host = _two_pages(qtbot)
    data = host.pane("voltage_series").data_pane
    frame = data.group("estimator")
    assert frame.isVisible()
    data.set_group_visible("estimator", False)
    assert not frame.isVisible()
    data.set_group_visible("estimator", True)
    assert frame.isVisible()


def test_adopting_a_key_twice_tells_the_first_adapter_it_was_retired(qtbot):
    page = H.SyntheticPlotPage()
    qtbot.addWidget(page)
    host = InspectorHost()
    qtbot.addWidget(host)
    first = page.inspector_adapter()
    told = []
    first.on_retire = lambda: told.append("first")
    host.adopt("synthetic", first)
    second = H.SyntheticPlotPage().inspector_adapter()
    host.adopt("synthetic", second)
    assert told == ["first"] and host.adapter("synthetic") is second


# -- group headers -------------------------------------------------------------------------------------

def test_a_group_title_with_an_ampersand_shows_it_and_makes_no_mnemonic(qtbot):
    frame = GroupFrame("recipes", "Export & recipes")
    qtbot.addWidget(frame)
    header = frame.findChild(QtWidgets.QToolButton, "inspectorGroupHeader_recipes")
    assert header is not None
    assert header.accessibleName() == "Export & recipes group"
    assert header.text() == "Export && recipes"
    assert QtGui.QKeySequence.mnemonic(header.text()).isEmpty()


# -- form_rows -----------------------------------------------------------------------------------------

def test_form_rows_lets_a_wide_widget_shrink_and_wraps_long_rows(qtbot):
    combo = QtWidgets.QComboBox()
    combo.setMinimumWidth(400)
    edit = QtWidgets.QLineEdit()
    edit.hide()
    box = form_rows([("Fit model", combo), ("Name", edit)])
    qtbot.addWidget(box)
    form = box.layout()
    assert box.minimumWidth() == 0 and combo.minimumWidth() == 0 and edit.minimumWidth() == 0
    assert form.rowWrapPolicy() == QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
    assert combo.parentWidget() is box and not edit.isHidden()      # adopted, and shown


# -- explicit export kinds --------------------------------------------------------------

@pytest.mark.parametrize("key", ("iv", "voltage_series", "distribution_summaries",
                                 "comparison_figures", "event_waveforms", "noise",
                                 "subpopulations", "distributions"))
def test_each_page_declares_export_kinds_without_guessing_from_words(qtbot, builders, key):
    explorer = H.hosted(qtbot, builders[key](), key)
    entries = explorer.adapter(key).exports()
    expected = (["figure", "data", "recipe"] if key == "distributions" else
                ["figure", "page"] if key == "subpopulations" else ["figure", "data"])
    assert [entry.kind for entry in entries] == expected
    menu = H.export_menu(explorer)
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions[:len(entries)]] == [entry.text for entry in entries]
    assert [a.isEnabled() for a in actions[:len(entries)]] == [True if e.enabled is None else bool(e.enabled()) for e in entries]
    menu.close()


def test_the_export_menu_has_no_separator_when_the_page_lists_nothing(qtbot):
    page = H.SyntheticPlotPage("iv")
    page.inspector_adapter().exports = lambda: []
    explorer = H.hosted(qtbot, page, "iv")
    menu = H.export_menu(explorer)
    shown = [a.isSeparator() for a in menu.actions()]
    menu.close()
    assert shown and not shown[0]
    assert [a.text() for a in menu.actions()] == ["Save to Figure Library…", "Save figure bundle…",
                                                  "Open Figure Library"]


# -- the toolbar: focus and sync -----------------------------------------------------------------------

def test_update_plot_clears_focus_only_inside_the_inspector(qtbot):
    page = H.SyntheticPlotPage("iv")
    outside = QtWidgets.QLineEdit(page.centralWidget())
    explorer = H.hosted(qtbot, page, "iv")
    outside.show()
    outside.setFocus()
    QtWidgets.QApplication.processEvents()
    assert outside.hasFocus()
    H.press_update(explorer)
    outside.setFocus()
    explorer.update_current()
    assert outside.hasFocus()                                   # the page's own widget keeps it
    explorer.inspector().set_tab(1)              # the appearance boxes live on the Format tab
    inside = page._appearance_widget.title_size      # a box of the Format tab's Figure panel
    inside.setFocus()
    QtWidgets.QApplication.processEvents()
    assert inside.hasFocus()
    explorer.update_current()
    assert not inside.hasFocus()


def test_the_toolbar_re_reads_the_page_after_a_verb_that_emits_nothing(qtbot):
    page = H.SyntheticPlotPage("iv")
    adapter = page.inspector_adapter()
    undo, redo, reset = adapter.undo, adapter.redo, adapter.reset_figure
    silent = lambda: setattr(page, "pending_count", page.pending_count + 1)    # noqa: E731
    adapter.update = silent
    adapter.undo = lambda: (undo(), silent())[0]
    adapter.redo = lambda: (redo(), silent())[0]
    adapter.reset_figure = lambda: (reset(), silent())[0]
    explorer = H.hosted(qtbot, page, "iv")
    bar = explorer.toolbar()
    widget = page._appearance_widget
    widget.title.textEdited.emit("Hello")
    widget._close_burst()
    assert bar.update_button.text() == "Update plot"
    H.press_update(explorer)
    assert bar.update_button.text() == "Update plot (1)"
    H.press_undo(explorer)
    assert bar.update_button.text() == "Update plot (2)"
    H.press_redo(explorer)
    assert bar.update_button.text() == "Update plot (3)"
    H.press_reset_figure(explorer)
    assert bar.update_button.text() == "Update plot (4)"


# -- the Explorer's inspector column -------------------------------------------------------------------

def test_a_hidden_inspector_starts_hidden_and_comes_back_at_its_remembered_width(qtbot):
    settings = InspectorSettings()
    settings.width = 430
    settings.visible = False
    page = H.SyntheticPlotPage("iv")
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    explorer.show_page("iv", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    bar, host = explorer.toolbar(), explorer.inspector()
    assert not bar.inspector_toggle.isChecked() and not host.isVisible()
    explorer.resize(1000, 700)                       # the window moves while the column is away
    QtWidgets.QApplication.processEvents()
    H.click(bar.inspector_toggle)
    QtWidgets.QApplication.processEvents()
    assert host.isVisible() and abs(host.width() - 430) <= 2
    assert InspectorSettings().visible is True


def test_the_inspector_builds_one_qsettings_not_one_per_read(qtbot, monkeypatch):
    """Every read of a group's flag used to build a QSettings: a Qt object per group per page,
    constructed on the hot path that the crash-rate loops stress."""
    made = []
    real = QtCore.QSettings

    class Counting(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(sys._getframe(1).f_globals.get("__name__", ""))
    monkeypatch.setattr(QtCore, "QSettings", Counting)
    settings = InspectorSettings()
    page = H.SyntheticPlotPage("iv")
    qtbot.addWidget(page)
    adapter = page.inspector_adapter()
    adapter.groups[:] = [InspectorGroup(f"g{n}", f"Group {n}", [QtWidgets.QLabel(str(n))])
                         for n in range(6)]
    host = InspectorHost(settings=settings)
    qtbot.addWidget(host)
    host.adopt("iv", adapter)
    host.show_page("iv")
    for n in range(6):
        settings.set_expanded("iv", f"g{n}", n % 2 == 0)
    assert settings.expanded("iv", "g1", True) is False and settings.expanded("iv", "g2", False) is True
    assert (settings.tab, settings.width, settings.visible) == (
        0, InspectorSettings.DEFAULT_WIDTH, True)
    assert [name for name in made if name.endswith("analysis_inspector")] == [
        "nanoscope.gui.analysis_inspector"]
