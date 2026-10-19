"""Change C -- the Event inspector docks beside the trace (opt-in), as one reparented window.

The inspector used to be a separate top-level window only; at its 1105x807 minimum it
covered the trace it curates. It can now be docked as a tab beside the Inspector panel
(Left/Right areas only), and moves between the two hosts without ever being rebuilt.
These tests pin the hosting contract: the default, the placement, the re-feeds, the
dock's own close, focus, the saved layout's placeholders and the main window's minimum.
"""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    WORK_AREA, TITLE_BAR, accepted, build_window, close_window, dead_wrappers, docked,
    private_settings, real_font, settle)

import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult


def _current_ids(window) -> list[int]:
    return [int(i) for i in window._analysis_panel.current_detection().payload.df["event_id"]]


def test_the_inspector_opens_as_its_own_window_by_default(private_settings, qtbot):
    window = build_window(qtbot)
    window._act_event_inspector.trigger()
    settle()
    insp = window._event_inspector
    assert insp.isWindow() and insp.isVisible()
    assert window.curation_host("event_inspector") == "window"
    assert window._inspector_dock is None, "no dock is built until the user docks"
    assert not window.findChildren(QtWidgets.QDockWidget, "EventInspectorDock")
    assert insp._host_btn.isVisible() and insp._host_btn.text() == "Dock beside trace"
    close_window(window)


def test_docking_makes_it_a_tab_beside_the_inspector_panel(private_settings, qtbot):
    window = build_window(qtbot)
    column = window._inspector.width()
    insp, dock = docked(window)
    assert window.findChildren(QtWidgets.QDockWidget, "EventInspectorDock") == [dock]
    assert dock.widget() is insp and not insp.isWindow()
    assert window.dockWidgetArea(dock) == Qt.RightDockWidgetArea
    assert window._inspector in window.tabifiedDockWidgets(dock)
    assert abs(window._inspector.width() - column) <= 6, (column, window._inspector.width())
    features = dock.features()
    assert features & QtWidgets.QDockWidget.DockWidgetClosable
    assert features & QtWidgets.QDockWidget.DockWidgetMovable
    assert not features & QtWidgets.QDockWidget.DockWidgetFloatable
    assert dock.allowedAreas() == Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
    assert window.curation_surface_on_screen("event_inspector")
    assert insp._host_btn.text() == "Own window"
    close_window(window)


def test_moving_between_hosts_never_rebuilds_it(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_inspector()
    insp = window._event_inspector
    for _ in range(3):
        insp._host_btn.click()
        settle()
        assert not insp.isWindow() and window._inspector_dock.widget() is insp
        insp._host_btn.click()
        settle()
        assert insp.isWindow() and insp.isVisible()
    assert window._event_inspector is insp and shiboken6.isValid(insp)
    assert len(window.findChildren(QtWidgets.QDockWidget, "EventInspectorDock")) == 1
    # ...and still wired: a decision here lands on the shared table.
    eid = insp.current_event_id()
    insp.set_accepted_current(False)
    assert not accepted(window, eid)
    close_window(window)


def test_a_host_change_keeps_the_browsed_event_filter_and_context(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_inspector()
    insp = window._event_inspector
    ids = _current_ids(window)
    insp.show_event(ids[3])
    insp._context_combo.setCurrentIndex(1)
    insp.set_filter("Accepted only")
    for host in ("dock", "window", "dock"):
        window.set_curation_host("event_inspector", host)
        settle()
        assert insp.current_event_id() == ids[3]
        assert insp.current_filter() == "Accepted only"
        assert insp._context_combo.currentIndex() == 1
    close_window(window)


def test_a_docked_inspector_behind_its_tab_is_still_refed(private_settings, qtbot):
    window = build_window(qtbot)
    insp, dock = docked(window)
    window._inspector.raise_()                 # the parameter panel comes forward
    settle()
    assert insp.visibleRegion().isEmpty()
    assert window.curation_surface_on_screen("event_inspector")
    eid = insp.current_event_id()
    window._analysis_panel.set_event_accepted(eid, False)
    settle()
    assert not insp._accept_check.isChecked(), "the table's decision reached the hidden tab"
    close_window(window)


def _new_detection(window) -> object:
    """A second detection on the same recording: a new frame for the windows to be fed."""
    res = window._analysis_panel.current_detection()
    data = window._session.calibrated(res.doc_id, 0).data
    frame, provenance = detect_events(np.asarray(data), 50_000.0, baseline=-1_000.0,
                                      min_dwell_s=1e-4)
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=res.doc_id, channel=0,
        params={"min_dwell_s": 1e-4, "run": 2},
        calibration_fingerprint=window._session.calibration(res.doc_id, 0).fingerprint(),
        payload=EventSet(frame, provenance)))
    settle()
    return window._analysis_panel.current_detection().payload.df


def test_closing_the_dock_closes_the_inspector_and_reopening_refeeds_it(private_settings,
                                                                          qtbot):
    window = build_window(qtbot)
    insp, dock = docked(window)
    dock.close()
    settle()
    assert dock.isHidden() and not window.curation_surface_on_screen("event_inspector")
    old = insp._df
    fresh = _new_detection(window)
    assert fresh is not old and insp._df is old, "a closed surface is not re-fed"
    window._act_event_inspector.trigger()
    settle()
    assert not dock.isHidden() and dock.widget() is insp
    assert insp._df is fresh
    assert insp.focusWidget() is insp._plot_widget
    close_window(window)


def test_showing_the_dock_by_any_other_route_refeeds_it(private_settings, qtbot):
    window = build_window(qtbot)
    insp, dock = docked(window)
    dock.close()
    settle()
    fresh = _new_detection(window)
    dock.show()                                # not the opener: e.g. a restored layout
    settle()
    assert insp._df is fresh


def test_close_on_the_docked_inspector_hides_its_dock(private_settings, qtbot):
    window = build_window(qtbot)
    insp, dock = docked(window)
    insp.close()
    settle()
    assert dock.isHidden(), "close() must never leave an empty dock behind"
    assert not window.curation_surface_on_screen("event_inspector")
    close_window(window)


def _right_click_a_dock_title(window) -> list[str] | None:
    """What the dock-area context menu offers, from a real right-click on a dock's title
    bar: the menu QMainWindow builds and deletes itself, never one made from Python."""
    pos = window._file_dock.mapTo(window, QtCore.QPoint(20, 6))
    QtWidgets.QApplication.sendEvent(window, QtGui.QContextMenuEvent(
        QtGui.QContextMenuEvent.Mouse, pos, window.mapToGlobal(pos)))
    popup = QtWidgets.QApplication.activePopupWidget()
    if popup is None:
        return None
    offered = [a.text() for a in popup.actions() if a.isVisible() and not a.isSeparator()]
    popup.close()
    return offered


def _drain_deletes() -> None:
    settle(5)
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_the_dock_area_context_menu_never_lists_the_inspector(private_settings, qtbot):
    window = build_window(qtbot)
    _insp, dock = docked(window)
    offered = _right_click_a_dock_title(window)
    assert offered is not None and "Inspector" in offered, offered
    assert dock.toggleViewAction().text() not in offered, offered
    # Updated (review C-ZOMBIE-1): no Python override builds that menu any more, so a
    # right-click leaves no dead wrapper behind.
    _drain_deletes()
    before = dead_wrappers()
    for _ in range(5):
        _right_click_a_dock_title(window)
        _drain_deletes()
    assert dead_wrappers() <= before
    close_window(window)


def test_show_in_trace_from_the_docked_inspector_lowers_nothing(private_settings, qtbot,
                                                                monkeypatch):
    window = build_window(qtbot)
    insp, _dock = docked(window)
    ids = _current_ids(window)
    insp.show_event(ids[2])
    lowered: list[object] = []
    monkeypatch.setattr(type(insp), "lower", lambda self: lowered.append(self))
    insp.focus_curation_surface()
    insp._on_show_in_trace()
    settle()
    assert window._analysis_panel.selected_event_id() == ids[2]
    assert lowered == []
    close_window(window)


def test_opening_the_docked_inspector_lowers_a_gallery_window_in_front_of_it(
        private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    window.set_curation_host("event_inspector", "dock")
    window._open_event_gallery()
    settle()
    gal = window._event_gallery
    gal.setGeometry(window.frameGeometry())    # a gallery covering the whole main window
    settle()
    lowered: list[object] = []
    monkeypatch.setattr(type(gal), "lower", lambda self: lowered.append(self))
    window._open_inspector_at_event(_current_ids(window)[1])
    settle()
    assert lowered == [gal]
    assert not window._inspector_dock.isHidden()
    gal.hide()
    close_window(window)


def test_a_hidden_tab_never_keeps_the_keyboard(private_settings, qtbot):
    window = build_window(qtbot)
    insp, dock = docked(window)
    window.activateWindow()
    insp.focus_curation_surface()
    settle()
    assert QtWidgets.QApplication.focusWidget() is insp._plot_widget
    window._inspector.raise_()
    settle()
    focus = QtWidgets.QApplication.focusWidget()
    assert focus is not None and not (focus is insp or insp.isAncestorOf(focus)), focus
    before = list(window._analysis_panel.current_detection().payload.df["accepted"])
    QTest.keyClick(focus, Qt.Key_R)
    settle()
    assert list(window._analysis_panel.current_detection().payload.df["accepted"]) == before
    close_window(window)


def test_closing_the_docked_tab_hands_the_keyboard_to_the_trace(private_settings, qtbot):
    window = build_window(qtbot)
    insp, dock = docked(window)
    window.activateWindow()
    insp.focus_curation_surface()
    settle()
    dock.close()
    settle()
    assert QtWidgets.QApplication.focusWidget() is window._trace_view
    close_window(window)


def _saved_state_with(qtbot, place) -> None:
    """Quit a window whose inspector dock ``place`` put somewhere, so the next window
    restores that layout."""
    window = build_window(qtbot)
    _insp, dock = docked(window)
    place(window, dock)
    settle()
    close_window(window)


def test_a_lazily_built_dock_lands_where_the_saved_layout_put_it(private_settings, qtbot):
    _saved_state_with(qtbot, lambda w, d: w.addDockWidget(Qt.LeftDockWidgetArea, d))
    window = build_window(qtbot)
    assert window.curation_host("event_inspector") == "dock"
    window._open_event_inspector()
    settle()
    assert window.dockWidgetArea(window._inspector_dock) == Qt.LeftDockWidgetArea
    close_window(window)


def _float(window, dock) -> None:
    dock.setFeatures(dock.features() | QtWidgets.QDockWidget.DockWidgetFloatable)
    dock.setFloating(True)


def _top(window, dock) -> None:
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    window.addDockWidget(Qt.TopDockWidgetArea, dock)


def test_a_saved_layout_that_floats_or_misplaces_the_dock_is_redocked(private_settings, qtbot):
    for place in (_float, _top):
        _saved_state_with(qtbot, place)
        window = build_window(qtbot)
        window._open_event_inspector()
        settle()
        dock = window._inspector_dock
        assert not dock.isFloating()
        assert window.dockWidgetArea(dock) == Qt.RightDockWidgetArea
        assert window._inspector in window.tabifiedDockWidgets(dock)
        close_window(window)


#: A 1280x800 screen's content area: the smallest the owner's windows are used on.
SMALL_WORK_AREA = (1280, 775 - TITLE_BAR)


def test_the_main_windows_minimum_fits_the_work_area_wherever_it_docks(private_settings,
                                                                       real_font, qtbot):
    """Review C-R3/C-R4: under the File manager, with the interval method and Details, the
    docked inspector forced the main window 20-98 px past a 1512x900 screen's work area."""
    window = build_window(qtbot)
    insp, dock = docked(window)
    amp = insp._amplitude
    # Right: the default tab beside the Inspector panel; Left: under the File manager.
    for area in (Qt.RightDockWidgetArea, Qt.LeftDockWidgetArea):
        if window.dockWidgetArea(dock) != area:
            window.addDockWidget(area, dock)
        for method in ("median", "selected_median", "manual"):
            amp.method.setCurrentIndex(amp.method.findData(method))
            for note in ("", "No sub-state levels for this event: the fit could not run on "
                             "this mask (412 samples)."):
                insp._level_note.setText(note)
                insp._level_note.setVisible(bool(note))
                for shown in (False, True):
                    insp.set_details_shown(shown)
                    settle(5)
                    hint = window.minimumSizeHint()
                    assert (hint.width() <= SMALL_WORK_AREA[0]
                            and hint.height() <= SMALL_WORK_AREA[1]), (area, method, note,
                                                                       shown, hint)
    # Below its content's own minimum the docked body scrolls; the waveform keeps its floor.
    assert insp._plot_widget.height() >= insp._plot_widget.minimumHeight()
    close_window(window)


def test_view_panels_inspector_brings_the_panel_forward_over_the_docked_tab(private_settings,
                                                                            qtbot):
    """Review C-R7: re-shown as a tab behind the docked Event inspector, the toggle seemed
    to do nothing."""
    window = build_window(qtbot)
    toggle = window._inspector.toggleViewAction()
    toggle.trigger()                           # View ▸ Panels ▸ Inspector off
    settle()
    assert window._inspector.isHidden()
    _insp, dock = docked(window)               # the inspector takes the right column alone
    toggle.trigger()                           # ...and on again
    settle(60)
    assert not window._inspector.widget().visibleRegion().isEmpty()
    assert dock.widget().visibleRegion().isEmpty()
    close_window(window)


def test_hosting_leaves_no_dead_wrappers_behind(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_inspector()
    insp = window._event_inspector
    insp._host_btn.click()
    insp._host_btn.click()
    window._window_menu.aboutToShow.emit()
    settle()
    before = dead_wrappers()
    for _ in range(10):
        insp._host_btn.click()
        window._window_menu.aboutToShow.emit()
        settle(5)
    assert dead_wrappers() <= before
    close_window(window)
