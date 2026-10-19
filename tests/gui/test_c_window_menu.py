"""Change C -- a Window menu: every window this app opens, where the Event inspector lives,
and a way back to the designed layout.

Its actions are built once and re-read on every open (destroyed Qt wrappers leak in this
PySide6, so nothing is created per open). Its entries are the windows' own openers, so a
window reached from here is re-fed exactly as Ctrl+I / Ctrl+G would.
"""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, dead_wrappers, docked, private_settings, settle)

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.gui import main_window as mw_mod
from nanoscope.gui.main_window import MainWindow


def _menu_titles(window) -> list[str]:
    return [a.text().replace("&", "") for a in window.menuBar().actions() if a.text()]


def _entries(window) -> dict[str, tuple[bool, bool]]:
    window._window_menu.aboutToShow.emit()
    return {a.text(): (a.isChecked(), a.isEnabled())
            for a in window._window_menu.actions() if a.isVisible() and not a.isSeparator()}


def test_the_window_menu_sits_before_help(private_settings, qtbot):
    window = build_window(qtbot, show=False, detection=False)
    assert _menu_titles(window) == ["File", "Edit", "View", "Analysis", "Window", "Help"]
    # Its entries are built on the first open; until then it is one action, never empty
    # (macOS leaves an empty menu out of the menu bar).
    assert [a.text() for a in window._window_menu.actions()] == ["Bring All to Front"]
    window._window_menu.aboutToShow.emit()
    assert len(window._window_menu.actions()) > 10
    close_window(window)


def test_it_lists_the_standard_windows_and_checks_the_open_ones(private_settings, qtbot):
    window = build_window(qtbot)
    entries = _entries(window)
    for name in ("Event inspector", "Event gallery", "Analysis Explorer", "Figure Library",
                 "Bring All to Front", "Dock Event inspector beside the trace",
                 "Reset Window Layout…"):
        assert name in entries, (name, entries)
    assert not entries["Event inspector"][0] and not entries["Event gallery"][0]
    assert not any(text.startswith("    ") for text in entries), "no Explorer page yet"
    docked(window)
    entries = _entries(window)
    assert entries["Event inspector — docked"] == (True, True)
    assert entries["Dock Event inspector beside the trace"][0]
    assert not entries["Event gallery"][0]
    window._open_event_gallery()
    settle()
    assert _entries(window)["Event gallery"][0]
    window._event_gallery.close()
    settle()
    assert not _entries(window)["Event gallery"][0]
    close_window(window)


def test_choosing_an_entry_runs_the_windows_own_opener(private_settings, qtbot, monkeypatch):
    opened: list[str] = []
    real = MainWindow._open_event_inspector
    monkeypatch.setattr(MainWindow, "_open_event_inspector",
                        lambda self: (opened.append("inspector"), real(self))[1])
    window = build_window(qtbot)
    window._window_menu.aboutToShow.emit()
    window._window_menu_entries["event_inspector"].trigger()
    settle()
    assert opened == ["inspector"]
    assert window.curation_surface_on_screen("event_inspector")
    close_window(window)


class _Page(QtWidgets.QWidget):
    """Stands in for an analytical page: what the Explorer's status line asks of one."""

    def has_data(self) -> bool:
        return False

    def points(self) -> list:
        return []


def test_an_explorer_page_entry_raises_that_page_as_a_tab_click_would(private_settings, qtbot):
    window = build_window(qtbot)
    explorer = window._ensure_analysis_explorer()
    pages = {pid: _Page() for pid in ("noise", "iv")}
    activated: list[str] = []
    for pid, page in pages.items():
        explorer.add_page(pid, page, activate=lambda p=pid: activated.append(p))
    explorer.show_home()
    settle()
    entries = _entries(window)
    shown = [text.strip() for text in entries if text.startswith("    ")]
    assert shown == ["Noise spectrum", "I-V / conductance"], shown
    page_actions = dict(window._window_menu_pages)
    page_actions["iv"].trigger()
    settle()
    assert explorer.current_page_id() == "iv" and activated == ["iv"]
    assert _entries(window)["    I-V / conductance"][0]
    explorer.close()
    close_window(window)


def test_other_windows_are_listed_once_built(private_settings, qtbot):
    window = build_window(qtbot, detection=False)
    assert "About NanoScope" not in _entries(window)
    window._open_build_info()
    settle()
    assert _entries(window)["About NanoScope"][0]
    window._build_info_dialog.close()
    settle()
    assert _entries(window)["About NanoScope"] == (False, True)
    close_window(window)


def test_the_window_menu_reuses_its_actions(private_settings, qtbot):
    window = build_window(qtbot)
    docked(window)
    menu = window._window_menu
    menu.aboutToShow.emit()
    first = [id(a) for a in menu.actions()]
    before = dead_wrappers()
    for _ in range(25):
        menu.aboutToShow.emit()
    assert [id(a) for a in menu.actions()] == first
    assert dead_wrappers() <= before
    close_window(window)


def test_bring_all_to_front_raises_every_open_window_and_keeps_the_active_one(
        private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    window._open_event_gallery()
    window._open_figure_library()
    settle()
    raised: list[str] = []
    for win in (window, window._event_gallery, window._figure_library):
        monkeypatch.setattr(win, "raise_", lambda w=win: raised.append(w.windowTitle()))
    window._event_gallery.activateWindow()
    settle()
    active = QtWidgets.QApplication.activeWindow()
    window._act_bring_all_to_front.trigger()
    settle()
    assert QtWidgets.QApplication.activeWindow() is active
    assert {window.windowTitle(), window._figure_library.windowTitle()} <= set(raised)
    window._event_gallery.hide()
    window._figure_library.hide()
    close_window(window)


def test_the_dock_entry_reflects_and_changes_the_host(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_inspector()
    settle()
    window._window_menu.aboutToShow.emit()
    toggle = window._act_dock_inspector
    assert not toggle.isChecked()
    toggle.trigger()
    settle()
    insp = window._event_inspector
    assert window.curation_host("event_inspector") == "dock" and not insp.isWindow()
    assert window.curation_surface_on_screen("event_inspector"), "an open window stays open"
    window._window_menu.aboutToShow.emit()
    assert toggle.isChecked()
    toggle.trigger()
    settle()
    assert insp.isWindow() and insp.isVisible()
    close_window(window)


def test_reset_window_layout_asks_first(private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    window._file_dock.setFloating(True)
    settle()
    state = window.saveState()
    monkeypatch.setattr(MainWindow, "_confirm_reset_window_layout", lambda self: False)
    window._window_menu.aboutToShow.emit()
    window._act_reset_window_layout.trigger()
    settle()
    assert window.saveState() == state and window._file_dock.isFloating()
    window._file_dock.setFloating(False)
    close_window(window)


def test_reset_window_layout_restores_the_designed_layout(private_settings, qtbot,
                                                          monkeypatch):
    window = build_window(qtbot)
    insp, dock = docked(window)
    window.addDockWidget(Qt.LeftDockWidgetArea, dock)
    window._file_dock.setFloating(True)
    window.addDockWidget(Qt.LeftDockWidgetArea, window._inspector)
    window._results_dock.hide()
    window._open_event_gallery()
    window._event_gallery.move(30, 40)
    window._event_gallery.hide()                     # its geometry is remembered on hide
    settle()
    s = QtCore.QSettings()
    assert s.contains("windows/event_gallery/geometry")
    monkeypatch.setattr(MainWindow, "_confirm_reset_window_layout", lambda self: True)
    window._window_menu.aboutToShow.emit()
    window._act_reset_window_layout.trigger()
    settle()
    assert not window._file_dock.isFloating()
    assert window.dockWidgetArea(window._file_dock) == Qt.LeftDockWidgetArea
    assert window.dockWidgetArea(window._inspector) == Qt.RightDockWidgetArea
    assert abs(window._inspector.width() - window._inspector_default_width()) <= 6
    assert window.dockWidgetArea(window._results_dock) == Qt.BottomDockWidgetArea
    assert not window._results_dock.isHidden()
    assert window.curation_host("event_inspector") == "window"
    assert insp.isWindow() and insp.isVisible(), "it was open, so it stays open"
    assert not any(s.contains(f"windows/{key}/geometry") for key in mw_mod._CHILD_WINDOW_KEYS)
    assert s.value("windows/event_inspector/host") == "window"
    close_window(window)
