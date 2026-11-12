"""Phase 1, shell group (G1): the one-line header, the bands that went, Home, Export ▾'s shell
items, the settings, and the Explorer at its smallest.

A real Explorer; clicks and keys go through the QWindow and menus are real. Synthetic pages
only (``SyntheticPlotPage``, and the real ``MainWindow`` on a synthetic CSV for the checkpoint
test). Offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_catalog import STATUS_HELP
from nanoscope.gui.analysis_inspector import InspectorSettings

from . import p1_hosting as H

KEY_1, KEY_2 = QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2


def _explorer(qtbot, *, keys=("iv",), state=("Current", "current"), context="pore A · channel 1",
              size=(1280, 820), availability=None):
    explorer = H.AnalysisExplorer(
        context_provider=lambda: context,
        status_provider=lambda _key, _page: state() if callable(state) else state,
        availability_provider=availability)
    qtbot.addWidget(explorer)
    explorer.resize(*size)
    pages = {}
    for key in keys:
        pages[key] = H.SyntheticPlotPage(key)
        explorer.add_page(key, pages[key])
    explorer.show()
    explorer.activateWindow()
    qtbot.waitExposed(explorer)
    if keys:
        explorer.raise_page(keys[0])
    QtWidgets.QApplication.processEvents()
    return explorer, pages


def _visible_buttons(root) -> list[str]:
    return [b.text() for b in root.findChildren(QtWidgets.QPushButton) if b.isVisible()]


# -- the header ----------------------------------------------------------------------------------

def test_the_header_is_one_24_px_line_above_the_tab_bar_in_the_figure_column(qtbot):
    explorer, _pages = _explorer(qtbot)
    header = explorer.findChild(QtWidgets.QFrame, "AnalysisExplorerHeader")
    assert header is not None and header.height() == 24 and header.isVisible()
    tabs = explorer._tabs
    left = explorer.findChild(QtWidgets.QWidget, "explorerFigureColumn")
    assert left.isAncestorOf(header) and left.isAncestorOf(tabs)
    assert header.mapTo(explorer, QtCore.QPoint(0, 0)).y() < tabs.mapTo(explorer, QtCore.QPoint(0, 0)).y()
    assert header.width() == tabs.width(), "the header is the figure column's, not the window's"
    assert not explorer.inspector().isAncestorOf(header)
    # three things on the line: the Analysis home button, the context and the chip
    assert explorer._home_btn.isVisible() and explorer._home_btn.isFlat()
    assert explorer._context_label.isVisible() and explorer._status_label.isVisible()
    assert explorer._home_btn.text() == "Analysis home" and explorer._status_label.text() == "CURRENT"
    assert explorer._context_label.text() == "pore A · channel 1"


def test_the_headers_home_button_goes_to_home_with_a_real_click(qtbot):
    explorer, _pages = _explorer(qtbot)
    assert explorer.current_page_id() == "iv"
    H.click(explorer._home_btn)
    assert explorer.current_page_id() == "analysis_home"
    assert explorer._status_label.text() == "ANALYSIS HOME"


def test_the_chip_and_context_come_from_the_unchanged_providers_and_follow_them(qtbot):
    box = {"state": ("Stale", "stale"), "context": "pore A"}
    explorer = H.AnalysisExplorer(context_provider=lambda: box["context"],
                                  status_provider=lambda _k, _p: box["state"])
    qtbot.addWidget(explorer)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    assert explorer._status_label.text() == "STALE" and explorer._context_label.text() == "pore A"
    box["state"], box["context"] = ("Current", "current"), "pore B · 2 recordings"
    explorer.refresh_header()
    assert explorer._status_label.text() == "CURRENT"
    assert explorer._context_label.text() == "pore B · 2 recordings"
    assert explorer._context_label.toolTip() == "pore B · 2 recordings"


def test_the_chip_tooltip_says_why_while_the_count_is_above_zero(qtbot):
    explorer, pages = _explorer(qtbot, state=("Stale", "stale"))
    page = pages["iv"]
    assert explorer._status_label.toolTip() == STATUS_HELP
    page.set_pending(2, "the metric", "a fit model")
    tip = explorer._status_label.toolTip()
    assert tip.startswith(STATUS_HELP) and "Out of date: the metric; a fit model." in tip
    assert explorer._status_label.text() == "STALE"
    page.set_pending(0)
    assert explorer._status_label.toolTip() == STATUS_HELP
    page.set_pending(1)                                      # a count with no reason still says so
    assert "Out of date:" in explorer._status_label.toolTip()


def test_a_long_context_is_elided_in_the_line_and_whole_in_the_tooltip_at_840_by_560(qtbot):
    long = "pore A · channel 1 · " + "a very long recording name " * 8
    explorer, _pages = _explorer(qtbot, context=long, size=(840, 560))
    label = explorer._context_label
    assert label.text() == long and label.toolTip() == long
    assert label.width() < label.fontMetrics().horizontalAdvance(long)
    assert label.minimumSizeHint().width() == 0, "it can shrink: the text is elided, not the window grown"
    elided = label.elided_text()
    assert elided.endswith("…") and elided != long
    assert label.fontMetrics().horizontalAdvance(elided) <= label.contentsRect().width()
    drawn = []
    label.elided_text = lambda: drawn.append(1) or elided        # what the painter asks for
    label.grab()
    assert drawn, "the line is painted from the elided text"
    del label.elided_text
    assert explorer.minimumSizeHint().width() <= 840
    header = explorer.findChild(QtWidgets.QFrame, "AnalysisExplorerHeader")
    assert header.width() == explorer._tabs.width(), "the long text did not widen the column"
    assert explorer._status_label.isVisible()
    assert explorer.width() == 840, "nothing in the header forced the window wider"


# -- the bands that went ---------------------------------------------------------------------------

def test_band_one_and_band_two_are_gone_and_their_buttons_are_hidden_objects(qtbot):
    explorer, _pages = _explorer(qtbot)
    for text in ("Refresh current", "Refresh availability", "Save to Figure Library…",
                 "Save figure bundle…", "Figure Library"):
        assert text not in _visible_buttons(explorer), text
    for button in (explorer._refresh_btn, explorer._save_figure_btn, explorer._save_bundle_btn,
                   explorer._figure_library_btn):
        assert button.isHidden() and explorer.isAncestorOf(button)
    explorer.show_home()
    assert "Refresh availability" in _visible_buttons(explorer)          # Home's own button
    assert not explorer._refresh_btn.isVisible()
    # the chrome above the canvas is the toolbar, the 24 px header and the tab bar: nothing else
    assert explorer._tabs.mapTo(explorer, QtCore.QPoint(0, 0)).y() <= (
        explorer.toolbar().height() + 6 + 24 + 4 + 10)


def test_the_retired_buttons_still_work_and_follow_the_state_as_the_old_ones_did(qtbot):
    explorer, pages = _explorer(qtbot)
    seen = []
    explorer.saveFigureRequested.connect(lambda key: seen.append(("figure", key)))
    explorer.saveBundleRequested.connect(lambda key: seen.append(("bundle", key)))
    explorer.figureLibraryRequested.connect(lambda: seen.append(("library", None)))
    explorer.add_page("iv", pages["iv"], refresh=lambda: seen.append(("refresh", None)))
    explorer.refresh_header()                 # a registered refresher enables the hidden button
    explorer._save_figure_btn.click()
    explorer._save_bundle_btn.click()
    explorer._figure_library_btn.click()
    explorer._refresh_btn.click()
    assert seen == [("figure", "iv"), ("bundle", "iv"), ("library", None), ("refresh", None)]
    assert explorer._save_figure_btn.isEnabled() and explorer._save_bundle_btn.isEnabled()
    explorer.show_home()
    assert not explorer._save_figure_btn.isEnabled() and not explorer._save_bundle_btn.isEnabled()
    assert explorer._refresh_btn.text() == "Refresh availability" and explorer._refresh_btn.isEnabled()


# -- Home ----------------------------------------------------------------------------------------

def test_home_has_its_own_refresh_availability_button_and_a_real_click_re_reads_the_gates(qtbot):
    calls = []

    def availability(route):
        calls.append(route)
        return True, "ready"
    explorer, _pages = _explorer(qtbot, keys=(), availability=availability)
    home = explorer.home()
    button = home.findChild(QtWidgets.QPushButton, "analysisHomeRefresh")
    assert button is not None and button.text() == "Refresh availability" and button.isVisible()
    assert explorer.current_page_id() == "analysis_home"
    calls.clear()
    H.click(button)
    assert calls, "a click re-read every gate"
    assert len(set(calls)) == len(home.route_ids())
    # refresh_header() still re-reads them on Home (MainWindow calls it after a restore)
    calls.clear()
    explorer.refresh_header()
    assert calls


def test_on_home_the_toolbar_is_off_except_style_and_the_inspector_is_hidden(qtbot):
    explorer, _pages = _explorer(qtbot)
    bar = explorer.toolbar()
    assert bar.update_button.isEnabled() and explorer.inspector().isVisible()
    explorer.show_home()
    assert not explorer.inspector().isVisible()
    for widget in (bar.update_button, bar.reset_figure_button, bar.export_button,
                   bar.inspector_toggle):
        assert not widget.isEnabled()
    assert not bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    assert bar.style_button.isEnabled()


# -- Export ▾'s shell items --------------------------------------------------------------------

def test_the_shell_items_are_enabled_exactly_as_the_old_buttons_were_and_emit_the_signals(qtbot):
    box = {"state": ("Current", "current")}
    explorer, _pages = _explorer(qtbot, state=lambda: box["state"])
    seen = []
    explorer.saveFigureRequested.connect(lambda key: seen.append(("figure", key)))
    explorer.saveBundleRequested.connect(lambda key: seen.append(("bundle", key)))
    explorer.figureLibraryRequested.connect(lambda: seen.append(("library", None)))

    def entries():
        menu = H.export_menu(explorer)
        found = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
        menu.hide()
        return found
    on = entries()
    assert on["Save to Figure Library…"] and on["Save figure bundle…"] and on["Open Figure Library"]
    assert explorer._save_figure_btn.isEnabled() and explorer._save_bundle_btn.isEnabled()
    H.choose_export(explorer, "Save to Figure Library…")
    H.choose_export(explorer, "Save figure bundle…")
    H.choose_export(explorer, "Open Figure Library")
    assert seen == [("figure", "iv"), ("bundle", "iv"), ("library", None)]
    # a page with nothing to save yet ("empty"): the two save items are off, the library opens
    box["state"] = ("No data", "empty")
    explorer.refresh_header()
    off = entries()
    assert not off["Save to Figure Library…"] and not off["Save figure bundle…"]
    assert off["Open Figure Library"]
    assert not explorer._save_figure_btn.isEnabled() and not explorer._save_bundle_btn.isEnabled()
    # the page's own entries come first, then the three shell items, in this order
    texts = [a.text() for a in H.export_menu(explorer).actions() if a.text()]
    assert texts[-3:] == ["Save to Figure Library…", "Save figure bundle…", "Open Figure Library"]
    assert texts[:2] == ["Export current figure…", "Export data (CSV)…"]


# -- settings ------------------------------------------------------------------------------------

def test_the_inspector_state_round_trips_through_a_fresh_explorer(qtbot):
    explorer, pages = _explorer(qtbot, keys=("iv", "voltage_series"))
    H.key(explorer, KEY_2, H.CTRL)                           # Format
    assert explorer.inspector().current_tab() == 1
    explorer.inspector().set_tab(0)
    explorer.reveal("estimator")
    pane = explorer.inspector().pane("iv")
    header = pane.data_pane.group("estimator").findChild(
        QtWidgets.QToolButton, "inspectorGroupHeader_estimator")
    H.click(header)                                          # collapse Estimator
    explorer.inspector().set_tab(1)
    explorer.inspector().settings().keep_hand = False
    H.click(explorer.toolbar().inspector_toggle)             # hide
    assert not explorer.inspector().isVisible()
    explorer.close()

    fresh, fresh_pages = _explorer(qtbot, keys=("iv", "voltage_series"))
    host = fresh.inspector()
    assert host.current_tab() == 1
    assert not fresh.toolbar().inspector_toggle.isChecked() and not host.isVisible()
    assert host.settings().keep_hand is False
    # expanded flags are per page KIND: the other page with the same kind id starts collapsed
    assert not host.pane("iv").data_pane.group("estimator").is_expanded()
    assert not host.pane("iv").data_pane.group("estimator").isVisibleTo(fresh)
    H.click(fresh.toolbar().inspector_toggle)
    assert host.isVisible()


def test_a_dragged_width_is_clamped_to_320_and_560_and_remembered(qtbot):
    explorer, _pages = _explorer(qtbot)
    host = explorer.inspector()
    for asked, expected in ((900, 560), (40, 320), (480, 480)):
        explorer._body.setSizes([explorer.width() - asked, asked])
        explorer._body.splitterMoved.emit(explorer.width() - asked, 1)
        QtWidgets.QApplication.processEvents()
        assert host.width() == expected, (asked, host.width())
        assert InspectorSettings().width == expected
    H.click(explorer.toolbar().inspector_toggle)
    H.click(explorer.toolbar().inspector_toggle)
    assert host.width() == 480, "the remembered width comes back with the column"


# -- the Explorer at its smallest ------------------------------------------------------------------

def test_at_840_by_560_the_toolbar_does_not_overflow_and_the_page_gets_room(qtbot):
    explorer, pages = _explorer(qtbot, size=(840, 560))
    bar = explorer.toolbar()
    extension = bar.findChild(QtWidgets.QToolButton, "qt_toolbar_ext_button")
    assert extension is None or not extension.isVisible()
    for widget in (bar.update_button, bar.style_button, bar.reset_figure_button,
                   bar.export_button, bar.inspector_toggle):
        assert widget.isVisible()
        assert widget.mapTo(explorer, QtCore.QPoint(widget.width(), 0)).x() <= 840
    assert bar.sizeHint().width() <= 650, "the design's own budget for the row"
    assert pages["iv"].minimumSizeHint().width() <= 500
    assert explorer.inspector().width() >= 320
    assert pages["iv"]._plot_widget.width() >= 420 and pages["iv"]._plot_widget.height() >= 300


# -- the checkpoint logic of MainWindow ------------------------------------------------------------

def test_a_tab_switch_does_not_mark_any_workspace_surface_pending(
        qapp, qtbot, tmp_path, monkeypatch):
    """``MainWindow.eventFilter`` reads Hide as a tab switch (the property the Explorer sets on
    its pages) and must not mark a page pending: the host, the toolbar and the inspector that
    bind to a page when its tab comes forward add no Hide to a page that stays alive."""
    from .test_plot_window_routes import _click_tab, _restored_main_window
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    pages = []
    ids = []
    for opener, kind in ((window._open_iv, "iv"), (window._open_psd, "noise"),
                         (window._open_plots, "distributions")):
        opener()
        figure_id = window._figure_workspace.current_id()
        assert window._figure_workspace.get(figure_id).spec.kind == kind
        ids.append(figure_id)
        pages.append(window._figure_workspace.page(figure_id))
    explorer = window._analysis_explorer
    assert len(set(ids)) == 3
    assert [explorer.page(key) for key in ids] == pages
    assert all(page is not None for page in pages)
    for _round in range(3):
        for page in pages:
            _click_tab(explorer, page)
            assert explorer.inspector().current_key() == explorer.current_page_id()
            assert window._workspace_pending == set(), "a tab switch is not a close"
    assert all(page.property("nanoscopeAnalysisExplorerPage") for page in pages)
    assert not any(page.property("nanoscopeAnalysisExplorerClosing") for page in pages)
    window._dirty = False


# -- guards over the shell files this group edits ---------------------------------------------------

def _calls(path, names):
    import ast
    tree = ast.parse(path.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in names:
                found[name] = found.get(name, 0) + 1
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found["def " + node.name] = found.get("def " + node.name, 0) + 1
    return found


def test_the_explorer_and_home_add_no_shortcut_no_event_filter_and_no_page_import():
    """Phase 1 design 12.3 for the two files the shell group edits beyond its own modules:
    the Explorer keeps its ONE ``QShortcut`` site (⌘1 and ⌘2, built in a loop) and its one baseline
    page filter and ``eventFilter``. Its one exact-target ``itemAt`` invalidates a tab's
    wrapper before native removal; it never traverses the layout. Home has none of these.
    Nothing here imports a page module."""
    import ast
    import pathlib
    gui = pathlib.Path(__import__("nanoscope.gui", fromlist=["x"]).__file__).parent
    explorer, home = gui / "analysis_explorer.py", gui / "analysis_home.py"
    found = _calls(explorer, {"QShortcut", "setShortcut", "setShortcuts", "installEventFilter",
                              "eventFilter", "itemAt", "windowHandle"})
    assert found == {"QShortcut": 1, "installEventFilter": 1, "def eventFilter": 1,
                     "eventFilter": 1, "itemAt": 1}, found  # eventFilter calls ``super()``
    keys = [n.value for n in ast.walk(ast.parse(explorer.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("Ctrl+")]
    assert sorted(keys) == ["Ctrl+1", "Ctrl+2"]
    assert _calls(home, {"QShortcut", "setShortcut", "installEventFilter", "itemAt"}) == {}
    pages = {"iv_panel", "voltage_dep_panel", "distribution_summary_panel", "comparison_figures_panel",
             "event_plot_panel", "psd_panel", "subpop_window", "plots_panel", "main_window"}
    # The files main_window.py is split into (main_window_<part>.py, _shared, _seams) are the main
    # window too, and hold what a name imported from it used to resolve to.
    pages |= {p.stem for p in gui.glob("main_window_*.py")}
    assert {"main_window_shared", "main_window_seams"} <= pages, pages
    # ... and so are the files plots_panel.py is split into (plots_constants.py and the
    # plots_<part>_mixin.py parts), for the same reason. Not ``plots_series``: a shared helper that
    # predates the split.
    pages |= {"plots_constants"} | {p.stem for p in gui.glob("plots_*_mixin.py")}
    assert {"plots_constants", "plots_fit_mixin", "plots_sources_mixin"} <= pages, pages
    assert "plots_series" not in pages, pages
    for path in (explorer, home):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[-1] not in pages, (path.name, node.module)
