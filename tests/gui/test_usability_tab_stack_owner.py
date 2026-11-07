"""Native actions must survive page removal without a caller retaining Qt's stack."""
import gc
import weakref

import nanoscope  # noqa: F401
import shiboken6
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_explorer import AnalysisExplorer
from . import p1_hosting as H


def _drain():
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()


def test_removed_page_native_actions_survive_gc_close_and_die_with_the_explorer(qapp):
    explorer = AnalysisExplorer()
    page = H.page_builders()['distributions']()
    explorer.show_page('distributions', page)
    QtWidgets.QApplication.processEvents()
    actions = [page._run_action, page._lasso_action, *page._export_menu.actions()]
    destroyed = []
    for index, action in enumerate(actions):
        action.destroyed.connect(lambda *_, index=index: destroyed.append(index))
    try:
        # No local reference to page.parentWidget() or its layout: retaining it in
        # the test would mask the production temporary-parent-wrapper defect.
        assert explorer.remove_page('distributions') is page
        assert all(shiboken6.isValid(action) for action in actions)
        gc.collect()
        _drain()
        assert all(shiboken6.isValid(action) for action in actions)
        assert not destroyed
        stack_ref = weakref.ref(explorer._tab_stack_owner)
        assert stack_ref() is not None and shiboken6.isValid(stack_ref())
        page.close()  # Native closeEvent reads and updates the retained QActions.
        _drain()
        assert all(shiboken6.isValid(action) for action in actions)
        assert not destroyed

        explorer.close()
        explorer.deleteLater()
        _drain()
        assert not shiboken6.isValid(explorer)
        assert not shiboken6.isValid(page)
        assert all(not shiboken6.isValid(action) for action in actions)
        assert sorted(destroyed) == list(range(len(actions)))
        assert stack_ref() is None or not shiboken6.isValid(stack_ref())
        # Native destruction is proved above. Python callbacks can retain an
        # already-invalid wrapper; this regression does not require collecting
        # every callback cycle belonging to the complete plot-page fixture.
        del page, action
        actions.clear()
        del explorer
        gc.collect()
        assert stack_ref() is None or not shiboken6.isValid(stack_ref())
    finally:
        if 'explorer' in locals() and shiboken6.isValid(explorer):
            explorer.deleteLater()
            _drain()
