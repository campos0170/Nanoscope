"""Phase 1: the page contract, one parametrized test per page id.

Every analytical page supplies its own adapter and satisfies the full contract at both
1280x820 and 840x560. There is no generic analytical-page fallback.

The full contract here: the keys, the verb, the export list, one step per look/Reset/Undo, Reset
view on the status line, Update plot's count, a look on a page behind another tab, the widget
budget across tab changes. Not here (a real save and reopen needs a ``MainWindow``): it is
``test_p1_iv_hosted``'s for I-V, and each page group adds its own the same way.

Synthetic only (an empty ``Session`` per page); offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets


from . import p1_hosting as H

PAGE_IDS = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
            "noise", "iv", "voltage_series", "subpopulations")

SIZES = ((1280, 820), (840, 560))
RETURN = QtCore.Qt.Key.Key_Return


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


def _host(qtbot, builders, key, size):
    page = builders[key]()
    explorer = H.hosted(qtbot, page, key, size=size)
    return explorer, page, explorer.adapter(key)


def _reachable(pane_scroll: QtWidgets.QScrollArea) -> bool:
    """Every part of the column can be brought into view: it fits, or it scrolls sideways."""
    bar = pane_scroll.horizontalScrollBar()
    content = pane_scroll.widget()
    need = content.minimumSizeHint().width() - pane_scroll.viewport().width()
    return need <= 0 or bar.maximum() >= need - 1


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("key", PAGE_IDS)
def test_the_page_is_hosted_and_every_control_is_reachable(qtbot, builders, key, size):
    explorer, page, adapter = _host(qtbot, builders, key, size)
    host, bar = explorer.inspector(), explorer.toolbar()
    # generic: hosted, adopted, the only visible page, a verb on the toolbar
    assert adapter is not None and adapter.page is page and host.pane(key) is not None
    assert page.property("nanoscopeInspectorHosted") is True and page.isVisible()
    assert host.isVisible() and host.current_key() == key and bar.bound_key() == key
    assert bar.update_button.isEnabled() and bar.export_button.isEnabled()
    assert not adapter.fallback and adapter is page.inspector_adapter()
    for tab in (0, 1):
        host.set_tab(tab)
        QtWidgets.QApplication.processEvents()
        scroll = host.pane(key).data_pane if tab == 0 else host.pane(key).format_pane._scroll
        assert _reachable(scroll), (key, tab)
    # every group's widgets are the page's own instances, inside the group body
    for group in adapter.groups:
        frame = host.pane(key).data_pane.group(group.key)
        assert frame is not None
        for widget in group.widgets:
            assert frame.body.isAncestorOf(widget) or widget is frame.body, (key, group.key)
    assert not host.pane(key).format_pane.is_legacy()
    explorer.resize(*size)
    QtWidgets.QApplication.processEvents()
    if size == (840, 560):
        assert page.minimumSizeHint().width() <= 500
        for group in adapter.groups:
            budget = 296 if key == "distributions" and group.key in ("groups", "plot") else 290
            assert host.pane(key).data_pane.group(group.key).minimumSizeHint().width() <= budget
        if key == "comparison_figures":
            # The magnification row and methods summary leave a 240 px canvas budget.
            assert page._view.width() >= 420 and page._view.height() >= 240
        elif key == "subpopulations":
            # Two figures and a report share this page, with independent minimum sizes.
            for view in (page._embed_view, page._wave_view):
                canvas = view._plot
                assert canvas.width() >= canvas.minimumWidth()
                assert canvas.height() >= canvas.minimumHeight()
                assert canvas.isVisible()
            assert page._readout.isVisible()
        else:
            canvas = page._density_glw if key == "event_waveforms" else page._plot_widget
            assert canvas.width() >= 420 and canvas.height() >= 300
        host.set_tab(0)
        assert host.pane(key).data_pane.horizontalScrollBar().maximum() == 0, key
    host.set_tab(0)
    # keys: ⌘2 and ⌘1 flip the tabs, ⌘Return runs the verb once
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    assert host.current_tab() == 1
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)
    assert host.current_tab() == 0
    ran = []
    original = adapter.update
    adapter.update = lambda: ran.append(1)
    H.key(explorer, RETURN, H.CTRL)
    assert ran == [1]
    adapter.update = original
    # the page's export list is the right-click menu's
    menu = H.export_menu(explorer)
    shell = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}
    entries = [a.text() for a in menu.actions() if not a.isSeparator() and a.text() not in shell]
    menu.close()
    original_menu = page._export_menu if key == "distributions" else page._workflow.export_menu
    original_entries = [a.text() for a in original_menu.actions() if not a.isSeparator()]
    assert entries == original_entries
    assert entries == [entry.text for entry in adapter.exports()]
    # Undo, Redo and Reset figure are one step each; a look is one step
    widget = adapter.format.appearance
    steps = adapter.history().mark
    from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
    adapter.apply_look(_BUILTIN_PRESETS["8 pt Arial · PLOS, Springer, AIP"], True)
    assert adapter.history().mark == steps + 1
    H.press_reset_figure(explorer)
    assert adapter.history().mark == steps + 2
    H.press_undo(explorer)
    H.press_undo(explorer)
    assert adapter.history().mark == steps
    # Navigation is page-specific: six pyqtgraph pages use the status line; Comparison
    # keeps its magnifier and Subpopulations keeps both per-figure navigation menus.
    status = adapter.status_bar()
    if adapter.navigation() is not None:
        assert status is not None
        visible = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
        assert len(visible) == 1 and visible[0].isVisible()
    elif key == "comparison_figures":
        assert page._fit_button.isVisible() and page._zoom_in.isVisible()
    else:
        assert key == "subpopulations" and status is None and page._readout.isVisible()
    old_bar = page._workflow_toolbar if key == "distributions" else page._workflow
    assert not old_bar.isVisible()
    if key != "distributions":
        assert old_bar.primary_actions                       # compatibility objects stay
    # Update plot's count is the page's pending state, in the words of the design
    count = adapter.pending().count
    assert bar.update_button.text() == (f"Update plot ({count})" if count else "Update plot")
    # a look reaches a page BEHIND another tab, as one step on its own history
    other = "voltage_series" if key != "voltage_series" else "iv"
    explorer.show_page(other, H.SyntheticPlotPage(other))
    explorer.raise_page(other)
    QtWidgets.QApplication.processEvents()
    assert not page.isVisible() and host.current_key() == other
    behind = adapter.history().mark
    result = explorer.apply_look(_BUILTIN_PRESETS["Slides · 18 pt Arial"], scope="all")
    assert key in result.applied and not result.failed
    assert adapter.history().mark == behind + 1 and widget.state().title_size == 18
    explorer.raise_page(key)
    QtWidgets.QApplication.processEvents()
    host.set_tab(1)
    QtWidgets.QApplication.processEvents()
    # the widget budget: cycling the tabs adds no widget (the Format tab was built just now)
    settled = H.count_widgets(host)
    for tab in (0, 1, 0, 1):
        host.set_tab(tab)
        QtWidgets.QApplication.processEvents()
    assert H.count_widgets(host) == settled, key


@pytest.mark.parametrize("key", PAGE_IDS)
def test_removing_the_hosted_page_then_clicking_every_button_raises_nothing(qtbot, builders, key):
    explorer, page, adapter = _host(qtbot, builders, key, (1280, 820))
    bar = explorer.toolbar()
    removed = explorer.remove_page(key)
    assert removed is page and bar.bound_key() is None and explorer.inspector().pane(key) is None
    removed.hide()
    removed.deleteLater()
    for _ in range(3):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()
    for button in (bar.update_button, bar.reset_figure_button, bar.export_button,
                   bar.inspector_toggle):
        H.click(button)
    H.press_undo(explorer)
    H.key(explorer, RETURN, H.CTRL)
    assert not shiboken6.isValid(page)


def test_the_contract_covers_every_registered_analytical_page():
    from nanoscope.gui.analysis_catalog import PAGE_IDS as registered
    assert set(PAGE_IDS) == set(registered)
