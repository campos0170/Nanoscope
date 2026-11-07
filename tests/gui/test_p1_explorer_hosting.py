"""Phase 1 Foundation, F5 and F9: the Explorer hosts all eight pages through their own adapters, moves nothing on a tab switch, and stays inside the
widget-count ceilings the Foundation measured.

Synthetic data only (an empty ``Session`` per page); offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math

import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.plot_workflow import PageAdapter

from . import p1_hosting as H

PAGE_IDS = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
            "noise", "iv", "voltage_series", "subpopulations")

#: Measured on the integrated tree 310e79c, 2026-11-03, offscreen, by this exact
#: before/after counting algorithm. ``adopt`` excludes the page's pre-existing widgets;
#: ``format`` counts the lazy element list, footer and first property panel. Ceilings add
#: 10%, rounded up. Comparison, Events and V-dep include their completed form reflows.
#: Noise's -5 is a tree-count artefact: its colour controls move into a FormatSpec-held
#: container that is attached on first use. Keep its earlier 24-widget adoption budget;
#: a negative net count is not a useful allowance for future construction.
FORMAT_FIRST_SHOW = 43
MEASURED = {"distributions": (27, FORMAT_FIRST_SHOW), "distribution_summaries": (24, FORMAT_FIRST_SHOW),
            "comparison_figures": (34, FORMAT_FIRST_SHOW), "event_waveforms": (34, FORMAT_FIRST_SHOW),
            "noise": (-5, FORMAT_FIRST_SHOW), "iv": (14, FORMAT_FIRST_SHOW),
            "voltage_series": (29, FORMAT_FIRST_SHOW), "subpopulations": (21, FORMAT_FIRST_SHOW)}
CEILING = {key: (math.ceil((24 if key == "noise" else adopt) * 1.1),
                 math.ceil(first_format * 1.1))
           for key, (adopt, first_format) in MEASURED.items()}
TOOLBAR_CEILING = math.ceil(12 * 1.1)
INSPECTOR_SHELL_CEILING = math.ceil(5 * 1.1)


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


def _explorer(qtbot, size=(1280, 820)):
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(*size)
    return explorer


def _all_pages(qtbot, builders, size=(1280, 820)):
    explorer = _explorer(qtbot, size)
    pages = {}
    for key in PAGE_IDS:
        pages[key] = builders[key]()
        explorer.add_page(key, pages[key])
    explorer.show()
    qtbot.waitExposed(explorer)
    return explorer, pages


def test_all_eight_pages_are_adopted_and_only_the_one_in_front_is_visible(qtbot, builders):
    explorer, pages = _all_pages(qtbot, builders)
    for key in PAGE_IDS:
        adapter = explorer.adapter(key)
        assert isinstance(adapter, PageAdapter) and adapter.key in (key, "iv")
        assert explorer.inspector().pane(key) is not None
        assert adapter is pages[key].inspector_adapter() and not adapter.fallback
        assert pages[key].property("nanoscopeAnalysisExplorerPage") is True
        assert pages[key].property("nanoscopeInspectorHosted") is True
    for key in PAGE_IDS:
        assert explorer.raise_page(key)
        QtWidgets.QApplication.processEvents()
        visible = [k for k in PAGE_IDS if pages[k].isVisible()]
        assert visible == [key], (key, visible)
        assert explorer.inspector().current_key() == key and explorer.inspector().isVisible()
        assert explorer.toolbar().bound_key() == key
        assert pages[key].property("nanoscopeAnalysisExplorerClosing") is False
    assert explorer.page_count() == 8


def test_a_tab_switch_moves_nothing(qtbot, builders):
    explorer, pages = _all_pages(qtbot, builders)
    host = explorer.inspector()
    counters = [H.child_added_counter(host._stack)]
    homes = {}
    for key in PAGE_IDS:
        pane = host.pane(key)
        counters.append(H.child_added_counter(pane))
        counters.append(H.child_added_counter(pane.data_pane.widget()))
        for gkey in pane.data_pane.group_keys():
            frame = pane.data_pane.group(gkey)
            counters.append(H.child_added_counter(frame.body))
            homes[(key, gkey)] = [(w, w.parentWidget())
                                  for w in frame.body.findChildren(
                                      QtWidgets.QWidget,
                                      options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)]
    total = H.count_widgets(explorer)
    update_actions = len(explorer.toolbar().update_button.actions())
    for _ in range(50):
        for key in PAGE_IDS:
            explorer.raise_page(key)
    QtWidgets.QApplication.processEvents()
    assert [c.count for c in counters] == [0] * len(counters)
    assert H.count_widgets(explorer) == total
    for (key, gkey), pairs in homes.items():
        frame = host.pane(key).data_pane.group(gkey)
        assert all(w.parentWidget() is parent is frame.body for w, parent in pairs)
    assert len(explorer.toolbar().update_button.actions()) == update_actions     # three or fifty


def test_adding_the_same_page_twice_adopts_once(qtbot, builders):
    explorer = _explorer(qtbot)
    page = builders["iv"]()
    explorer.add_page("iv", page)
    pane = explorer.inspector().pane("iv")
    explorer.add_page("iv", page, refresh=lambda: None)
    assert explorer.inspector().pane("iv") is pane and explorer.page_count() == 1
    # a different widget under the same id retires the first one's pane
    other = builders["iv"]()
    explorer.add_page("iv", other)
    assert explorer.inspector().pane("iv") is not pane and explorer.page_count() == 1
    assert explorer.adapter("iv").page is other


def test_refresh_current_calls_only_the_front_pages_refresher(qtbot):
    calls = []
    explorer = _explorer(qtbot)
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    explorer.add_page("iv", a, refresh=lambda: calls.append("iv"))
    explorer.add_page("voltage_series", b, refresh=lambda: calls.append("voltage_series"))
    explorer.show()
    explorer.raise_page("iv")
    explorer.refresh_current()
    explorer.raise_page("voltage_series")
    explorer.refresh_current()
    assert calls == ["iv", "voltage_series"]


def test_removing_the_bound_page_then_clicking_every_button_raises_nothing(qtbot):
    explorer = _explorer(qtbot)
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    explorer.show_page("iv", a)
    explorer.show_page("voltage_series", b)
    explorer.show()
    qtbot.waitExposed(explorer)
    bar = explorer.toolbar()
    assert bar.bound_key() == "voltage_series"
    removed = explorer.remove_page("voltage_series")
    assert removed is b and bar.bound_key() == "iv"            # the page behind took over
    removed = explorer.remove_page("iv")
    assert bar.bound_key() is None and explorer.inspector().current_key() is None
    removed.hide()
    b.hide()
    a.deleteLater()
    b.deleteLater()
    for _ in range(3):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()
    updates = a.updates
    for button in (bar.update_button, bar.reset_figure_button, bar.export_button,
                   bar.inspector_toggle, bar.style_button):
        H.click(button)
    H.press_undo(explorer)
    H.press_redo(explorer)
    for keycode, mods in ((QtCore.Qt.Key.Key_Return, H.CTRL), (QtCore.Qt.Key.Key_1, H.CTRL),
                          (QtCore.Qt.Key.Key_2, H.CTRL), (QtCore.Qt.Key.Key_Z, H.CTRL)):
        H.key(explorer, keycode, mods)
    assert explorer.inspector().pane("iv") is None and explorer.page_count() == 0
    assert not shiboken6.isValid(a) and not shiboken6.isValid(b)


def test_a_page_deleted_while_bound_is_a_noop_for_the_toolbar(qtbot):
    explorer = _explorer(qtbot)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    bar = explorer.toolbar()
    page.hide()
    explorer.remove_page("iv")                                 # forgets the adapter first
    page.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    bar.sync()
    bar.run_update()                                           # nothing bound: nothing happens


def test_home_hides_the_inspector_and_the_keys_do_nothing_there(qtbot):
    explorer = _explorer(qtbot)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    bar, host = explorer.toolbar(), explorer.inspector()
    assert host.isVisible() and bar.update_button.isEnabled()
    explorer.show_home()
    QtWidgets.QApplication.processEvents()
    assert not host.isVisible() and bar.bound_key() is None
    for w in (bar.update_button, bar.reset_figure_button, bar.export_button, bar.inspector_toggle):
        assert not w.isEnabled()
    assert bar.style_button.isEnabled()
    H.key(explorer, QtCore.Qt.Key.Key_Return, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    assert page.updates == 0 and host.current_tab() == 0
    explorer.raise_page("iv")
    assert host.isVisible() and bar.bound_key() == "iv"


def test_command_1_and_2_are_the_inspector_tabs_on_a_hosted_page_only(qtbot):
    explorer = _explorer(qtbot)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    explorer.activateWindow()
    qtbot.waitExposed(explorer)
    host = explorer.inspector()
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    assert host.current_tab() == 1
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)
    assert host.current_tab() == 0
    H.key(explorer, QtCore.Qt.Key.Key_3, H.CTRL)               # unbound
    H.key(explorer, QtCore.Qt.Key.Key_4, H.CTRL)
    assert host.current_tab() == 0 and page._workflow.sections.currentText() == "Data"


def test_the_page_stand_ins_that_other_tests_use_still_work(qtbot):
    explorer = _explorer(qtbot)
    for key in ("distributions", "iv"):
        explorer.show_page(key, QtWidgets.QMainWindow())
    explorer.show()
    assert explorer.page_ids() == ("distributions", "iv") and explorer.current_page_id() == "iv"
    assert not explorer.inspector().isVisible() and explorer.toolbar().bound_key() is None
    explorer.remove_page("iv")
    assert explorer.current_page_id() == "distributions"


def test_closing_the_explorer_with_pages_hosted_and_reopening_it(qtbot, builders):
    explorer = _explorer(qtbot)
    page = builders["noise"]()
    explorer.show_page("noise", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    explorer.close()
    QtWidgets.QApplication.processEvents()
    assert not explorer.isVisible()
    explorer.show_page("noise", page)
    QtWidgets.QApplication.processEvents()
    assert explorer.isVisible() and page.isVisible() and explorer.inspector().isVisible()


def test_widget_counts_stay_inside_the_ceilings_the_foundation_measured(qtbot, builders):
    explorer = _explorer(qtbot)
    explorer.show()
    qtbot.waitExposed(explorer)
    base = H.count_widgets(explorer)
    assert H.count_widgets(explorer.toolbar()) <= TOOLBAR_CEILING + 1      # + the toolbar itself
    assert H.count_widgets(explorer.inspector()) <= INSPECTOR_SHELL_CEILING + 1
    for key in PAGE_IDS:
        page = builders[key]()
        own = H.count_widgets(page)
        before = H.count_widgets(explorer)
        explorer.add_page(key, page)
        QtWidgets.QApplication.processEvents()
        added = H.count_widgets(explorer) - before - own
        explorer.raise_page(key)
        QtWidgets.QApplication.processEvents()
        mid = H.count_widgets(explorer)
        explorer.inspector().set_tab(1)
        QtWidgets.QApplication.processEvents()
        first_format = H.count_widgets(explorer) - mid
        explorer.inspector().set_tab(0)
        # the second show, and a hundred selections' worth of switching, add nothing
        mid = H.count_widgets(explorer)
        for _ in range(10):
            explorer.inspector().set_tab(1)
            explorer.inspector().set_tab(0)
        assert H.count_widgets(explorer) == mid
        assert added <= CEILING[key][0], (key, added, CEILING[key])
        assert first_format <= CEILING[key][1], (key, first_format, CEILING[key])
    assert H.count_widgets(explorer) > base


def test_the_inspector_opens_at_its_remembered_width_and_a_real_drag_is_remembered(qtbot):
    from nanoscope.gui.analysis_inspector import InspectorSettings
    InspectorSettings().width = 420
    explorer = _explorer(qtbot)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    host = explorer.inspector()
    assert abs(host.width() - 420) <= 2
    handle = explorer._body.handle(1)
    start = handle.mapTo(explorer, handle.rect().center())
    window = H.window_handle(explorer)
    from PySide6 import QtTest
    QtTest.QTest.mousePress(window, H.L, H.NOMOD, start)
    for step in range(1, 7):
        QtTest.QTest.mouseMove(window, start - QtCore.QPoint(15 * step, 0))
    QtTest.QTest.mouseRelease(window, H.L, H.NOMOD, start - QtCore.QPoint(90, 0))
    QtWidgets.QApplication.processEvents()
    assert 500 <= host.width() <= 560 and InspectorSettings().width == host.width()
    # the width is clamped, whatever the drag asks
    assert host.minimumWidth() == 320 and host.maximumWidth() == 560
    explorer.show_home()                      # hidden on Home: the width is not "saved" as 0
    assert InspectorSettings().width >= 320


def test_update_current_commits_focus_then_calls_the_front_pages_verb(qtbot):
    explorer = _explorer(qtbot)
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    explorer.show_page("iv", a)
    explorer.show_page("voltage_series", b)
    explorer.show()
    explorer.update_current()
    assert (a.updates, b.updates) == (0, 1)               # the page behind is not asked
    explorer.raise_page("iv")
    explorer.update_current()
    assert (a.updates, b.updates) == (1, 1)
    explorer.show_home()
    explorer.update_current()
    assert (a.updates, b.updates) == (1, 1)


def test_reveal_on_the_explorer_is_the_front_pages(qtbot):
    explorer = _explorer(qtbot)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    assert explorer.reveal("@format") is True and explorer.inspector().current_tab() == 1
    H.show_data(explorer, "estimator")
    assert explorer.inspector().current_tab() == 0
    H.show_format(explorer, "title")
    assert explorer.inspector().current_tab() == 1
    explorer.show_home()
    assert explorer.reveal("estimator") is False
