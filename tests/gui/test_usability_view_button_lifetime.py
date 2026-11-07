"""Plot navigation can outlive its view menu during native window teardown."""
import nanoscope  # noqa: F401
import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets
from nanoscope.gui.plot_interaction import view_button
from .test_plot_interaction import bare


@pytest.mark.parametrize('target', ['button', 'menu', 'back', 'forward'])
def test_navigation_does_not_resolve_or_touch_destroyed_view_controls(bare, qtbot, target):
    widget, plot, nav = bare
    resolves = []
    def resolve():
        resolves.append(True)
        return nav
    baseline_receivers = nav.receivers(QtCore.SIGNAL('historyChanged()'))
    button = view_button(resolve, (nav,), parent=widget)
    assert nav.receivers(QtCore.SIGNAL('historyChanged()')) == baseline_receivers + 1
    menu = button.menu()  # Keep the wrapper while its native parent owns it.
    back, forward = button._nanoscope_back, button._nanoscope_forward
    callback = button._nanoscope_sync
    assert all(shiboken6.isValid(obj) for obj in (button,menu,back,forward,nav))
    receivers = nav.receivers(QtCore.SIGNAL('historyChanged()'))
    obj = {'button':button, 'menu':menu, 'back':back, 'forward':forward}[target]
    try:
        obj.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert not shiboken6.isValid(obj)
        if target in ('button','menu'):
            assert nav.receivers(QtCore.SIGNAL('historyChanged()')) == receivers - 1
        assert shiboken6.isValid(nav) and shiboken6.isValid(widget)
        if target != 'button':
            assert shiboken6.isValid(button), 'match partially destroyed native window state'
        count = len(resolves)
        with qtbot.captureExceptions() as exceptions:
            nav.reset_view()  # Real application action emits normal history changes.
            QtWidgets.QApplication.processEvents()
        assert exceptions == []
        callback()  # A caller may still hold the Python callback after its receiver dies.
        assert len(resolves) == count, 'dead controls must not resolve a closing page'
    finally:
        # Delete only this test's control. On the unfixed baseline, retire its exact
        # dangling closure as well so a red witness cannot pollute later tests.
        if shiboken6.isValid(button):
            button.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        if shiboken6.isValid(nav) and nav.receivers(QtCore.SIGNAL('historyChanged()')) > baseline_receivers:
            nav.historyChanged.disconnect(callback)



def test_a_previously_queued_view_callback_is_harmless_after_menu_teardown(bare, qtbot):
    widget, plot, nav = bare
    resolves = []
    baseline_receivers = nav.receivers(QtCore.SIGNAL('historyChanged()'))
    button = view_button(lambda: resolves.append(True) or nav, (nav,), parent=widget)
    menu = button.menu()
    callback = button._nanoscope_sync
    count = len(resolves)
    completed = []
    # This models a callback already posted before the menu's destruction. It must
    # be safe even though QObject-scoped connections cannot revoke a direct caller.
    def queued():
        try:
            callback()
        finally:
            completed.append(True)
    QtCore.QTimer.singleShot(0, queued)
    try:
        menu.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert not shiboken6.isValid(menu) and shiboken6.isValid(button)
        with qtbot.captureExceptions() as exceptions:
            qtbot.waitUntil(lambda: bool(completed), timeout=1000)
        assert exceptions == [] and len(resolves) == count
        assert shiboken6.isValid(nav) and shiboken6.isValid(widget)
    finally:
        if shiboken6.isValid(button):
            button.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        if shiboken6.isValid(nav) and nav.receivers(QtCore.SIGNAL('historyChanged()')) > baseline_receivers:
            nav.historyChanged.disconnect(callback)
