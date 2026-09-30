"""Native cursor crash guard and ownership of the application override stack."""
from types import SimpleNamespace

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui import busy_cursor


def test_macos_busy_cursor_never_enters_native_conversion(qapp, monkeypatch):
    monkeypatch.setattr(busy_cursor, "sys", SimpleNamespace(platform="darwin"))
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
    try:
        def forbidden(*args):
            pytest.fail("macOS busy feedback must not change the native cursor")

        with monkeypatch.context() as patch:
            patch.setattr(QtWidgets.QApplication, "setOverrideCursor", forbidden)
            patch.setattr(QtWidgets.QApplication, "restoreOverrideCursor", forbidden)
            active = busy_cursor.begin_wait_cursor()
            assert active is False
            busy_cursor.end_wait_cursor(active)
        assert QtWidgets.QApplication.overrideCursor().shape() == QtCore.Qt.CrossCursor
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_supported_cursor_restores_preexisting_override(qapp, monkeypatch, platform):
    monkeypatch.setattr(busy_cursor, "sys", SimpleNamespace(platform=platform))
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
    try:
        active = busy_cursor.begin_wait_cursor()
        assert active is True
        try:
            assert QtWidgets.QApplication.overrideCursor().shape() == QtCore.Qt.WaitCursor
        finally:
            busy_cursor.end_wait_cursor(active)
        assert QtWidgets.QApplication.overrideCursor().shape() == QtCore.Qt.CrossCursor
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
