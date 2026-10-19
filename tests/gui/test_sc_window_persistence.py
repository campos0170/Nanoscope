"""Beta-sweep SC group: a saved window geometry from a differently sized display is adopted
(not called corrupt), Signal Conditioning is listed in the Window menu, and Reset Window
Layout puts a window that is hidden back at its default size."""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, private_settings, settle)

import struct

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.gui.main_window import MainWindow, _saved_normal_rect

# Offsets in QWidget.saveGeometry()'s blob: the maximized flag and the screen's width.
_MAXIMIZED, _SCREEN_WIDTH = 44, 46


def _blob(qtbot, rect: QtCore.QRect, *, screen_width: int = 2560,
          maximized: bool = False) -> QtCore.QByteArray:
    """A sound geometry blob as a 2560-wide display would have saved it. The offscreen
    screen is 800 wide, so Qt itself declines it (checked here, or the test tests nothing)."""
    probe = QtWidgets.QMainWindow()
    qtbot.addWidget(probe)
    probe.setGeometry(rect)
    probe.show()
    settle()
    raw = bytearray(bytes(probe.saveGeometry()))
    struct.pack_into(">i", raw, _SCREEN_WIDTH, screen_width)
    raw[_MAXIMIZED] = int(maximized)
    blob = QtCore.QByteArray(bytes(raw))
    other = QtWidgets.QMainWindow()
    qtbot.addWidget(other)
    assert not other.restoreGeometry(blob), "precondition: Qt declines this display change"
    return blob


def _saved(key: str, blob) -> None:
    s = QtCore.QSettings()
    s.setValue(key, blob)
    s.sync()


# -- X1 ------------------------------------------------------------------------------------
def test_a_geometry_blob_is_read_for_its_normal_rect(qtbot):
    rect = QtCore.QRect(12, 80, 500, 400)
    blob = _blob(qtbot, rect)
    assert _saved_normal_rect(blob) == (rect, False)
    assert _saved_normal_rect(_blob(qtbot, rect, maximized=True)) == (rect, True)


def test_only_bytes_qt_could_never_restore_are_unusable(qtbot):
    good = bytes(_blob(qtbot, QtCore.QRect(12, 80, 500, 400)))
    assert _saved_normal_rect(good) is not None
    for bad in (b"", b"garbage", good[:20],
                b"\x00\x00\x00\x00" + good[4:],                      # other magic
                good[:4] + b"\x00\x09" + good[6:],                   # a newer major version
                good[:24] + struct.pack(">4i", 0, 0, -1, -1) + good[40:]):    # an empty rect
        assert _saved_normal_rect(bad) is None, bad


def test_a_main_geometry_from_another_display_is_adopted_not_called_corrupt(
        private_settings, qtbot):
    _saved("geometry", _blob(qtbot, QtCore.QRect(10, 20, 760, 700)))
    window = MainWindow()
    qtbot.addWidget(window)
    assert "geometry" not in window._info_bar._label.text()
    assert (window.width(), window.height()) == (max(760, window.minimumWidth()),
                                                 max(700, window.minimumHeight()))


def test_a_maximized_main_window_stays_maximized_on_another_display(private_settings, qtbot):
    _saved("geometry", _blob(qtbot, QtCore.QRect(10, 20, 760, 700), maximized=True))
    window = MainWindow()
    qtbot.addWidget(window)
    assert "geometry" not in window._info_bar._label.text()
    assert window.windowState() & Qt.WindowMaximized


def test_a_main_geometry_bigger_than_this_screen_is_fitted_to_it(private_settings, qtbot):
    _saved("geometry", _blob(qtbot, QtCore.QRect(-300, -200, 2200, 1500)))
    window = MainWindow()
    qtbot.addWidget(window)
    avail = window._available_screen_geometry()
    assert "geometry" not in window._info_bar._label.text()
    assert window.width() <= avail.width() and window.height() <= avail.height()
    assert window.x() >= avail.left() and window.y() >= avail.top()


def test_an_unusable_main_geometry_is_still_named(private_settings, qtbot):
    _saved("geometry", QtCore.QByteArray(b"garbage"))
    window = MainWindow()
    qtbot.addWidget(window)
    assert "geometry" in window._info_bar._label.text()


def test_a_child_geometry_from_another_display_is_kept_and_used(private_settings, qtbot):
    _saved("windows/event_gallery/geometry", _blob(qtbot, QtCore.QRect(12, 80, 780, 520)))
    window = build_window(qtbot)
    said: list[str] = []
    window.statusBar().messageChanged.connect(said.append)
    window._open_event_gallery()
    settle()
    gal = window._event_gallery
    assert not any("could not be used" in m for m in said), said
    assert QtCore.QSettings().contains("windows/event_gallery/geometry")
    assert (gal.width(), gal.height()) == (780, 520), gal.geometry()
    gal.hide()
    close_window(window)


def test_a_signal_conditioning_geometry_from_another_display_is_used(private_settings, qtbot):
    rect = QtCore.QRect(20, 40, 780, 700)
    _saved("signalConditioning/geometry", _blob(qtbot, rect))
    window = build_window(qtbot, detection=False)
    window._open_signal_conditioning()
    settle()
    ws = window._signal_conditioning
    assert "Ignored saved Signal Conditioning" not in window.statusBar().currentMessage()
    assert (ws.width(), ws.height()) == (780, 700), ws.geometry()
    ws.hide()
    close_window(window)


def test_an_unusable_signal_conditioning_geometry_is_still_named(private_settings, qtbot):
    _saved("signalConditioning/geometry", QtCore.QByteArray(b"garbage"))
    window = build_window(qtbot, detection=False)
    window._open_signal_conditioning()
    settle()
    assert "signalConditioning/geometry" in window.statusBar().currentMessage()
    window._signal_conditioning.hide()
    close_window(window)


# -- X5 ------------------------------------------------------------------------------------
def _entries(window) -> dict[str, tuple[bool, bool]]:
    window._window_menu.aboutToShow.emit()
    return {a.text(): (a.isChecked(), a.isEnabled())
            for a in window._window_menu.actions() if a.isVisible() and not a.isSeparator()}


def test_signal_conditioning_is_in_the_window_menu(private_settings, qtbot, monkeypatch):
    opened: list[str] = []
    real = MainWindow._open_signal_conditioning
    monkeypatch.setattr(MainWindow, "_open_signal_conditioning",
                        lambda self, *a: (opened.append("sc"), real(self, *a))[1])
    window = build_window(qtbot, detection=False)
    assert _entries(window)["Signal Conditioning"] == (False, True)
    window._window_menu_entries["signal_conditioning"].trigger()
    settle()
    assert opened == ["sc"]
    assert _entries(window)["Signal Conditioning"][0]
    window._signal_conditioning.hide()
    settle()
    assert not _entries(window)["Signal Conditioning"][0]
    close_window(window)


# -- LC-2 ----------------------------------------------------------------------------------
def _reset(window, monkeypatch) -> None:
    monkeypatch.setattr(MainWindow, "_confirm_reset_window_layout", lambda self: True)
    window._reset_window_layout()
    settle()


def test_a_window_hidden_before_a_reset_reopens_at_its_default_size(private_settings, qtbot,
                                                                     monkeypatch):
    window = build_window(qtbot)
    window._open_event_gallery()
    gal = window._event_gallery
    settle()
    default = gal.size()
    gal.resize(default.width() - 60, default.height() - 90)
    settle()
    assert gal.size() != default
    gal.hide()
    settle()
    _reset(window, monkeypatch)
    window._open_event_gallery()
    settle()
    assert gal.size() == default
    gal.hide()
    close_window(window)
