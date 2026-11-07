"""Phase 1 Foundation, F3 and F6: the inspector host, its groups, its settings, its lifetime,
and the two controls that were re-flowed to fit its column.

Driven on ``SyntheticPlotPage`` (the interface with no real page). Synthetic; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_inspector import (DataPane, GroupFrame, InspectorHost,
                                              InspectorSettings, PagePane)
from nanoscope.gui.format_inspector import FormatPane, SelectionHub
from nanoscope.gui.plot_templates import WindowTemplateControls
from nanoscope.gui.plot_workflow import FormatSpec, InspectorGroup, PageAdapter
from nanoscope.gui.source_appearance import SourceAppearanceEditor

from . import p1_hosting as H


def _host(qtbot, groups=None, *, key="synthetic"):
    page = H.SyntheticPlotPage(key)
    qtbot.addWidget(page)
    adapter = page.inspector_adapter()
    if groups is not None:
        adapter.groups[:] = groups(page)
    host = InspectorHost()
    qtbot.addWidget(host)
    host.resize(360, 600)
    pane = host.adopt(key, adapter)
    host.show_page(key)
    host.show()
    qtbot.waitExposed(host)
    return page, adapter, host, pane


def test_a_group_holds_the_same_widget_instances_the_page_has(qtbot):
    page, adapter, host, pane = _host(qtbot)
    frame = pane.data_pane.group("estimator")
    assert isinstance(frame, GroupFrame) and frame.objectName() == "inspectorGroup_estimator"
    assert page._estimator.parentWidget() is frame.body
    assert page._estimator.isVisible()
    assert isinstance(pane, PagePane) and isinstance(pane.data_pane, DataPane)
    assert isinstance(pane.format_pane, FormatPane) and isinstance(pane.hub, SelectionHub)
    assert host.pane("synthetic") is pane and host.adapter("synthetic") is adapter
    assert host.selection_hub("synthetic") is pane.hub and host.current_key() == "synthetic"
    assert host.accessibleName() == "Plot inspector" and host.objectName() == "analysisInspector"


def test_collapse_hides_the_body_only_and_moved_widgets_keep_their_own_flags(qtbot):
    page, _adapter, host, pane = _host(qtbot)
    frame = pane.data_pane.group("estimator")
    assert frame.is_expanded() and page._estimator.isVisible()
    frame.set_expanded(False)
    assert not frame.is_expanded() and not frame.body.isVisible()
    assert not page._estimator.isVisible() and not page._estimator.isHidden()   # its own flag
    page._estimator.hide()                    # hidden for another reason
    frame.set_expanded(True)
    assert frame.body.isVisible() and page._estimator.isHidden()                # stays hidden
    assert not page._estimator.isVisible()


def test_a_real_click_on_a_group_header_collapses_it_and_the_choice_is_kept(qtbot):
    page, _adapter, host, pane = _host(qtbot)
    frame = pane.data_pane.group("estimator")
    header = frame.findChild(QtWidgets.QToolButton, "inspectorGroupHeader_estimator")
    assert header.accessibleName() == "Estimator group"
    H.click(header)
    assert not frame.is_expanded() and not page._estimator.isVisible()
    assert InspectorSettings().expanded("synthetic", "estimator", True) is False
    H.click(header)
    assert frame.is_expanded() and InspectorSettings().expanded("synthetic", "estimator", False)


def test_reveal_selects_the_tab_expands_the_group_and_reports_what_it_found(qtbot):
    def groups(page):
        return [InspectorGroup("estimator", "Estimator", [page._estimator]),
                InspectorGroup("recipes", "Recipes", [QtWidgets.QLabel("recipe")], expanded=False)]
    page, adapter, host, pane = _host(qtbot, groups)
    host.set_tab(1)
    assert host.current_tab() == 1 and pane.currentIndex() == 1
    recipes = pane.data_pane.group("recipes")
    assert not recipes.is_expanded()
    assert host.reveal("recipes") is True
    assert host.current_tab() == 0 and pane.currentIndex() == 0 and recipes.is_expanded()
    # an old Section name goes through the adapter's map; "@format" is the Format tab
    assert host.reveal("Appearance") is True and host.current_tab() == 1
    assert host.reveal("Export") is True and host.current_tab() == 0     # "recipes"
    # an unknown key falls back to the first visible group
    pane.data_pane.group("estimator").set_expanded(False)
    assert host.reveal("no-such-group") is True and pane.data_pane.group("estimator").is_expanded()
    asked = []
    pane.format_pane.show_element = asked.append
    assert host.reveal("@format:title") is True and host.current_tab() == 1
    assert asked == ["title"]
    assert host.reveal("@format") is True and asked == ["title"]      # no element: just the tab
    host.show_page(None)
    assert host.reveal("recipes") is False


def test_reveal_scrolls_the_frame_to_the_top_only_inside_reveal(qtbot):
    def groups(page):
        return [InspectorGroup(f"g{i}", f"G{i}", [QtWidgets.QLabel("x\n" * 12)]) for i in range(8)]
    page, adapter, host, pane = _host(qtbot, groups)
    bar = pane.data_pane.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == 0
    host.reveal("g3")
    frame = pane.data_pane.group("g3")
    top = frame.mapTo(pane.data_pane.widget(), QtCore.QPoint(0, 0)).y()
    assert bar.value() > 0 and abs(min(top, bar.maximum()) - bar.value()) <= 1
    host.reveal("g7")                      # the last group cannot reach the top: it is clamped
    assert bar.value() == bar.maximum()
    host.reveal("g3")
    # nothing else scrolls it: switching tab, refreshing, expanding another group
    value = bar.value()
    host.set_tab(1)
    host.set_tab(0)
    host.refresh_current()
    pane.data_pane.group("g1").set_expanded(False)
    pane.data_pane.group("g1").set_expanded(True)
    assert bar.value() == value


def test_visible_when_follows_a_stub_signal(qtbot):
    class Stub(QtCore.QObject):
        changed = QtCore.Signal()

    stub = Stub()
    view = {"name": "a"}

    def groups(page):
        return [InspectorGroup("estimator", "Estimator", [page._estimator]),
                InspectorGroup("only_a", "Only A", [QtWidgets.QLabel("a")],
                               visible_when=lambda: view["name"] == "a")]
    page = H.SyntheticPlotPage()
    qtbot.addWidget(page)
    adapter = page.inspector_adapter()
    adapter.groups[:] = groups(page)
    adapter.format.views_changed = stub.changed
    host = InspectorHost()
    qtbot.addWidget(host)
    pane = host.adopt("synthetic", adapter)
    host.show_page("synthetic")
    host.show()
    qtbot.waitExposed(host)
    frame = pane.data_pane.group("only_a")
    assert frame.isVisible()
    view["name"] = "b"
    stub.changed.emit()
    assert not frame.isVisible()
    view["name"] = "a"
    host.refresh_current()                       # the silent route re-reads it too
    assert frame.isVisible()
    pane.data_pane.set_group_visible("only_a", False)
    assert not frame.isVisible()
    pane.data_pane.sync_visibility()
    assert not frame.isVisible()                 # a forced hide survives the predicate


def test_settings_round_trip_through_a_fresh_host(qtbot):
    page, adapter, host, pane = _host(qtbot)
    host.set_tab(1)
    host.settings().width = 411
    host.settings().keep_hand = False
    host.set_user_visible(False)
    pane.data_pane.group("estimator").userToggled.emit(False)
    fresh = InspectorHost()
    qtbot.addWidget(fresh)
    assert fresh.current_tab() == 1
    assert fresh.settings().width == 411 and fresh.settings().keep_hand is False
    assert fresh.settings().visible is False
    assert fresh.settings().expanded("synthetic", "estimator", True) is False


def test_settings_are_validated_and_clamped():
    s = InspectorSettings()
    s.width = 10_000
    assert s.width == InspectorSettings.MAX_WIDTH
    s.width = 5
    assert s.width == InspectorSettings.MIN_WIDTH
    settings = QtCore.QSettings()
    for key, hostile in (("tab", "nonsense"), ("width", [1]), ("visible", "maybe"),
                         ("style/keep_hand", 7)):
        settings.setValue(f"windows/analysis_explorer/inspector/{key}", hostile)
    assert (s.tab, s.width, s.visible, s.keep_hand) == (0, InspectorSettings.DEFAULT_WIDTH, True, True)
    settings.setValue("windows/analysis_explorer/inspector/tab", 9)
    assert s.tab == 1
    # an INI store hands booleans back as text: "false" is False, not the default
    for text, value in (("false", False), ("False", False), ("true", True), (" TRUE ", True)):
        settings.setValue("windows/analysis_explorer/inspector/visible", text)
        settings.setValue("windows/analysis_explorer/inspector/groups/k/g", text)
        assert s.visible is value and s.expanded("k", "g", not value) is value
    s.set_expanded("a/b", "c\\d", False)            # a slash cannot make a path
    assert s.expanded("a/b", "c\\d", True) is False


def test_the_pane_dies_after_its_page_and_nothing_raises(qtbot):
    page = H.SyntheticPlotPage()
    host = InspectorHost()
    qtbot.addWidget(host)
    host.adopt("synthetic", page.inspector_adapter())
    host.show_page("synthetic")
    estimator = page._estimator
    # No reference to the pane is kept: the host's own bookkeeping is all there is, as in
    # AnalysisExplorer.remove_page. (A local ``pane`` would hide a pane that died early.)
    assert len(host._stack.findChildren(PagePane)) == 1
    host.retire("synthetic")
    assert host._stack.count() == 0
    assert host.pane("synthetic") is None and host.adapter("synthetic") is None
    assert host.current_key() is None and not host.isVisible()
    assert len(host._stack.findChildren(PagePane)) == 1               # hidden, still the stack's
    assert not host._stack.findChildren(PagePane)[0].isVisible()
    assert shiboken6.isValid(estimator) and estimator.isHidden() is False   # the page still reads it
    page.deleteLater()
    for _ in range(3):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()
    assert not shiboken6.isValid(page) and not shiboken6.isValid(estimator)
    assert host._stack.findChildren(PagePane) == []
    host.retire("synthetic")                                          # twice is safe


def test_a_page_deleted_without_retire_takes_its_pane_with_it(qtbot):
    page = H.SyntheticPlotPage()
    host = InspectorHost()
    qtbot.addWidget(host)
    pane = host.adopt("synthetic", page.inspector_adapter())
    page.deleteLater()
    for _ in range(3):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()
    assert not shiboken6.isValid(pane)


def test_adopting_hands_the_page_its_reveal_and_the_bar_its_own_chrome(qtbot):
    page, adapter, host, _pane = _host(qtbot)
    assert page.property("nanoscopeInspectorHosted") is True
    assert not page._workflow.isVisible() and page._workflow.parentWidget().isHidden()
    page._workflow.set_section("Appearance")           # the shim: a Section name reveals
    assert host.current_tab() == 1
    page._workflow.set_section("Data")
    assert host.current_tab() == 0
    page._inspector_reveal("Appearance")
    assert host.current_tab() == 1


def test_a_page_with_nothing_to_host_gets_an_empty_pane_and_no_inspector(qtbot):
    page = QtWidgets.QWidget()
    qtbot.addWidget(page)
    from nanoscope.gui.analysis_explorer import AnalysisExplorer
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.add_page("iv", page)
    adapter = explorer.adapter("iv")
    assert not adapter.fallback and adapter.groups == []
    host = explorer.inspector()
    host.show_page("iv")
    assert not host.has_content("iv") and not host.isVisible()
    host.set_user_visible(True)
    assert not host.isVisible()


def test_the_format_tab_is_built_on_first_use_and_holds_the_appearance_widgets_rows(qtbot):
    """Was the Foundation's stub (the appearance widget whole); the Format group moved its rows
    into element panels (``test_p1_format_panels`` has the detail)."""
    page, adapter, host, pane = _host(qtbot)
    widget = page._appearance_widget
    assert pane.format_pane._body is None and not widget.isVisible()   # a skeleton
    host.set_tab(1)
    assert pane.format_pane._body is not None
    body = pane.format_pane.panel_for("figure")
    assert body.isAncestorOf(widget.font_family) and widget.font_family.isVisible()
    assert not widget.isVisible()                                      # the husk
    assert pane.format_pane.elements()


# -- F6: the two controls that had to fit the column -------------------------------------------

def test_the_window_template_group_fits_the_inspector_column(qtbot):
    controls = WindowTemplateControls(
        "noise-spectrum", capture=lambda: ({}, {}), recall=lambda s, a: None,
        note="A template keeps the settings and the appearance. " * 6)
    qtbot.addWidget(controls)
    assert controls.minimumWidth() == 0
    assert controls.minimumSizeHint().width() <= 290
    assert controls.note.wordWrap()
    # the buttons are still the same objects, in a group the tests find by name
    assert controls.save_button.text() == "Save as template…"
    assert {b.text() for b in (controls.rename_button, controls.delete_button)} == {"Rename", "Delete"}


def test_the_source_names_editor_is_a_wrapping_form_that_fits_the_column(qtbot):
    editor = SourceAppearanceEditor()
    qtbot.addWidget(editor)
    assert editor.minimumSizeHint().width() <= 290
    layout = editor.layout()
    assert isinstance(layout, QtWidgets.QFormLayout)
    assert layout.rowWrapPolicy() == QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
    assert layout.rowCount() == 6
    assert layout.labelForField(editor.source).text() == "Source"
    assert layout.labelForField(editor.label).text() == "Display label"
    for row, field in enumerate((editor.source, editor.label)):
        assert layout.getWidgetPosition(field) == (row, QtWidgets.QFormLayout.ItemRole.FieldRole)
    for name in ("source", "label", "color", "reset_color", "apply", "project"):
        assert editor.isAncestorOf(getattr(editor, name))


def test_group_roots_the_old_section_hid_are_shown_when_they_move(qtbot):
    page = H.SyntheticPlotPage()
    qtbot.addWidget(page)
    page._estimator.hide()                          # what the Section logic did to it
    host = InspectorHost()
    qtbot.addWidget(host)
    host.adopt("synthetic", page.inspector_adapter())
    host.show_page("synthetic")
    host.show()
    qtbot.waitExposed(host)
    assert page._estimator.isVisible()


def test_retire_tells_the_adapter_and_a_second_retire_does_not(qtbot):
    page = H.SyntheticPlotPage()
    qtbot.addWidget(page)
    adapter = page.inspector_adapter()
    told = []
    adapter.on_retire = lambda: told.append(1)
    host = InspectorHost()
    qtbot.addWidget(host)
    host.adopt("synthetic", adapter)
    host.retire("synthetic")
    host.retire("synthetic")
    assert told == [1]


def test_a_page_behind_another_tab_is_revealed_without_touching_the_visible_tab(qtbot):
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    for page in (a, b):
        qtbot.addWidget(page)
    host = InspectorHost()
    qtbot.addWidget(host)
    host.adopt("iv", a.inspector_adapter())
    host.adopt("voltage_series", b.inspector_adapter())
    host.show_page("voltage_series")
    host.show()
    host.set_tab(0)
    assert host.reveal("@format", key="iv") is True        # a Section name on the page behind
    assert host.current_tab() == 0 and host.pane("iv").currentIndex() == 1
    host.show_page("iv")                                   # coming to the front takes the host's tab
    assert host.pane("iv").currentIndex() == 0


def test_the_comparison_adapter_lowers_its_old_minimum_and_releases_hidden_sections(qtbot):
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
    from nanoscope.core.session import Session
    page = ComparisonFiguresWindow(Session())
    qtbot.addWidget(page)
    assert page.minimumWidth() >= 700 and page._workflow._sections["Analysis"][0].maximumHeight() < 16777215
    explorer = H.hosted(qtbot, page, "comparison_figures")
    assert page.minimumWidth() <= 480 and page.minimumHeight() <= 360
    assert not explorer.adapter("comparison_figures").fallback
    assert page._analysis_section.isHidden()
    pane = explorer.inspector().pane("comparison_figures")
    for group in explorer.adapter("comparison_figures").groups:
        for widget in group.widgets:
            assert pane.data_pane.group(group.key).body.isAncestorOf(widget)


def test_retiring_a_page_disconnects_format_before_clearing_its_selection(qtbot):
    page, adapter, host, pane = _host(qtbot, key="iv")
    assert adapter.key == host.current_key() == "iv"
    host.set_tab(1)
    QtWidgets.QApplication.processEvents()
    offered = pane.format_pane.elements()
    selected = next(ref for ref in offered if ref.key != "figure")
    pane.format_pane.select(selected)
    QtWidgets.QApplication.processEvents()
    assert pane.hub.current() == selected and selected is not None
    retired_when_cleared = []
    pane.hub.selectionChanged.connect(
        lambda ref, _canvas: retired_when_cleared.append(pane.format_pane._retired)
        if ref is None else None)
    host.retire(adapter.key)
    assert retired_when_cleared == [True]
    assert pane.format_pane._retired and pane.page_connections == []
    assert pane.hub.current() is None
    host.retire(adapter.key)                 # idempotent, including the Format cleanup


def test_the_generic_analytical_fallback_has_been_removed():
    from nanoscope.gui import analysis_inspector
    for name in ("fallback_adapter", "_bar_page_adapter", "_distributions_adapter"):
        assert not hasattr(analysis_inspector, name)
