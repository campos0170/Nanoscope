"""Change C -- child windows keep their size and place, and the inspector its host.

Only the main window used to remember where it was. Now every window MainWindow opens
keeps its geometry under ``windows/<key>/geometry`` (written when it hides and at quit),
and the Event inspector its host and Details. Everything read back is external input:
validated like the main window's own geometry, a value Qt refuses is said and dropped,
and a remembered frame is clamped to the screen it now has.
"""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, docked, private_settings, settle)

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt


def _avail(window) -> QtCore.QRect:
    return (window.screen() or QtGui.QGuiApplication.primaryScreen()).availableGeometry()


def _near(a: QtCore.QRect, b: QtCore.QRect, tol: int = 2) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a.getRect(), b.getRect()))


def test_a_child_windows_geometry_round_trips(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_gallery()
    gal = window._event_gallery
    # Inside the offscreen suite's 800x800 screen, so no clamp moves it.
    gal.setGeometry(12, 80, gal.minimumSizeHint().width() + 8, 520)
    settle()
    want = gal.geometry()
    gal.close()
    close_window(window)

    again = build_window(qtbot)
    again._open_event_gallery()
    settle()
    assert _near(again._event_gallery.geometry(), want), (again._event_gallery.geometry(), want)
    again._event_gallery.hide()
    close_window(again)


def test_the_analysis_explorer_keeps_its_place_too(private_settings, qtbot):
    window = build_window(qtbot, detection=False)
    window._open_analysis_home()
    explorer = window._analysis_explorer
    explorer.setGeometry(0, 30, 840, 600)
    settle()
    want = explorer.geometry()
    close_window(window)

    again = build_window(qtbot, detection=False)
    again._open_analysis_home()
    settle()
    assert _near(again._analysis_explorer.geometry(), want)
    again._analysis_explorer.close()
    close_window(again)


def test_the_inspectors_host_and_details_round_trip(private_settings, qtbot):
    window = build_window(qtbot)
    insp, _dock = docked(window)
    insp._details_btn.click()                       # shown while docked
    close_window(window)

    again = build_window(qtbot)
    assert again.curation_host("event_inspector") == "dock"
    again._open_event_inspector()
    settle()
    insp = again._event_inspector
    assert not insp.isWindow() and again._inspector_dock.widget() is insp
    assert again.dockWidgetArea(again._inspector_dock) == Qt.RightDockWidgetArea
    assert insp.details_state() == {"window": True, "dock": True}
    insp._host_btn.click()                          # back to its own window
    settle()
    close_window(again)

    third = build_window(qtbot)
    assert third.curation_host("event_inspector") == "window"
    close_window(third)


def test_wrong_typed_window_preferences_are_ignored_and_named(private_settings, qtbot):
    s = QtCore.QSettings()
    s.setValue("windows/event_inspector/host", "sideways")
    s.setValue("windows/event_inspector/details_dock", "maybe")
    s.setValue("windows/event_gallery/geometry", "not a geometry")
    s.sync()
    window = build_window(qtbot)
    text = window._info_bar._label.text()
    for key in ("windows/event_inspector/host", "windows/event_inspector/details_dock",
                "windows/event_gallery/geometry"):
        assert key in text, text
    assert window.curation_host("event_inspector") == "window"
    window._open_event_inspector()
    assert window._event_inspector.details_state() == {"window": True, "dock": False}
    window._open_event_gallery()
    settle()
    assert window._event_gallery.isVisible()
    window._event_gallery.hide()
    close_window(window)
    s = QtCore.QSettings()
    assert s.value("windows/event_inspector/host") == "window"
    assert s.value("windows/event_inspector/details_dock") in ("false", False)
    assert isinstance(s.value("windows/event_gallery/geometry"), QtCore.QByteArray)


def test_a_geometry_qt_refuses_is_reported_and_dropped(private_settings, qtbot):
    s = QtCore.QSettings()
    s.setValue("windows/event_gallery/geometry", QtCore.QByteArray(b"garbage"))
    s.sync()
    window = build_window(qtbot)
    said: list[str] = []
    window.statusBar().messageChanged.connect(said.append)
    window._open_event_gallery()
    settle()
    assert any("could not be used" in m for m in said), said
    assert not QtCore.QSettings().contains("windows/event_gallery/geometry")
    assert window._event_gallery.isVisible()
    window._event_gallery.hide()
    close_window(window)


def _saved_gallery_geometry(qtbot, rect: QtCore.QRect) -> None:
    window = build_window(qtbot)
    window._open_event_gallery()
    window._event_gallery.setGeometry(rect)
    settle()
    window._event_gallery.hide()
    close_window(window)


def test_a_window_saved_off_screen_comes_back_on_it(private_settings, qtbot):
    window = build_window(qtbot, detection=False)
    avail = _avail(window)
    close_window(window)
    _saved_gallery_geometry(qtbot, QtCore.QRect(avail.right() + 4000, 40, 500, 400))
    again = build_window(qtbot)
    again._open_event_gallery()
    settle()
    frame = again._event_gallery.frameGeometry()
    assert avail.contains(frame), (frame, avail)
    assert again._event_gallery.isVisible()
    again._event_gallery.hide()
    close_window(again)


def test_an_oversized_saved_window_is_clamped_to_the_screen(private_settings, qtbot):
    window = build_window(qtbot, detection=False)
    avail = _avail(window)
    close_window(window)
    _saved_gallery_geometry(qtbot, QtCore.QRect(0, 0, avail.width() + 2000,
                                                avail.height() + 1500))
    again = build_window(qtbot)
    again._open_event_gallery()
    settle()
    size = again._event_gallery.frameGeometry().size()
    assert size.width() <= avail.width() and size.height() <= avail.height(), (size, avail)
    again._event_gallery.hide()
    close_window(again)


def test_a_window_with_nothing_remembered_opens_where_qt_puts_it(private_settings, qtbot):
    """Review C-PLACE-1: moving a never-shown window onto the screen stopped Qt placing it,
    so every first open landed in the screen's top-left corner, not over this window."""
    from nanoscope.gui.build_info_dialog import BuildInfoDialog

    window = build_window(qtbot, detection=False)
    plain = BuildInfoDialog(window)              # the same dialog, never adopted
    plain.show()
    settle()
    expected = plain.frameGeometry().topLeft()
    plain.close()
    assert expected != _avail(window).topLeft(), "Qt centres it on the main window"
    window._open_build_info()
    settle()
    about = window._build_info_dialog
    assert about.frameGeometry().topLeft() == expected
    about.close()
    close_window(window)


def test_defaults_apply_only_to_a_main_window_on_screen(private_settings, qtbot):
    hidden = build_window(qtbot, show=False)
    hidden._open_event_gallery()
    assert hidden._event_gallery.size() == QtCore.QSize(940, 700), "constructor size kept"
    hidden._event_gallery.hide()
    hidden._dirty = False
    hidden._analysis_controller.abort()


def test_a_docked_inspector_never_writes_window_geometry(private_settings, qtbot):
    window = build_window(qtbot)
    window._open_event_inspector()
    insp = window._event_inspector
    insp.setGeometry(20, 60, 700, 600)
    settle()
    want = insp.geometry()
    docked(window)                                  # its window geometry is kept, not lost
    close_window(window)
    s = QtCore.QSettings()
    assert s.contains("windows/event_inspector/geometry")

    again = build_window(qtbot)
    again.set_curation_host("event_inspector", "window")
    again._open_event_inspector()
    settle()
    assert _near(again._event_inspector.geometry(), want), again._event_inspector.geometry()
    again._event_inspector.hide()
    close_window(again)
