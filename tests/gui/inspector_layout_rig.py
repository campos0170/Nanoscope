"""Geometry witnesses for the current page-scrolled Inspector: every mode page scrolls in an area
of its own and pins its actions above it, except the hosted Preprocess overview, whose door
scrolls with it."""
from __future__ import annotations

import nanoscope  # noqa: F401 -- Qt bootstrap
from PySide6 import QtCore, QtWidgets


def settle(window):
    for _ in range(4):
        QtWidgets.QApplication.processEvents()
        for widget in window.findChildren(QtWidgets.QWidget):
            layout = widget.layout()
            if layout is not None:
                layout.activate()
        if window.layout() is not None:
            window.layout().activate()
        QtWidgets.QApplication.processEvents()


def build(qtbot, size=(1512, 900), width=None):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    tabs = window._inspector.widget()
    tabs.setCurrentWidget(window._analysis_panel)
    window._analysis_panel.detection_panel.set_control_level('advanced')
    if width is not None:
        window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
    settle(window)
    return window


def show_page(window, title):
    """Raise the page and return the scroll area a user sees on it.

    2026-11-11: the shipped window hosts the Preprocess page as the workspace overview, which
    scrolls in its own area (``preprocess_overview_scroll``, 8661d35). The page's own scroller
    (``page_scrolls``) is hidden then and keeps a stale size (its viewport stays 638 px wide at
    every dock width, and "horizontal scrollbar not visible" is true of any hidden widget), so
    measuring it would prove nothing about the page on screen."""
    detection = window._analysis_panel.detection_panel
    title = {'ML detector': 'ML', 'Manual measurement': 'Measure'}.get(title, title)
    index = next(i for i in range(detection.mode_tabs.count())
                 if detection.mode_tabs.tabText(i) == title)
    detection.mode_tabs.setCurrentIndex(index)
    if title == 'Preprocess':
        detection.show_signal_chain()
    settle(window)
    if title == 'Preprocess' and getattr(detection, '_preprocess_workspace_hosted', False):
        return detection.preprocess_overview_scroll
    return detection.page_scrolls[detection.mode_tabs.currentWidget()]


def owning_scroll(widget):
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QtWidgets.QScrollArea):
            return parent
        parent = parent.parentWidget()
    raise AssertionError(f'{widget.objectName() or type(widget).__name__} has no scroll owner')


def rectangle(widget, ancestor):
    return QtCore.QRect(widget.mapTo(ancestor, QtCore.QPoint()), widget.size())


def assert_whole(widget, ancestor=None):
    assert widget.isVisible(), getattr(widget, 'text', lambda: type(widget).__name__)()
    # QAbstractScrollArea paints through its viewport; its frame's visibleRegion
    # can legitimately be empty because the opaque viewport covers it completely.
    painted = widget.viewport() if isinstance(widget, QtWidgets.QAbstractScrollArea) else widget
    visible = painted.visibleRegion().boundingRect()
    assert visible.width() >= painted.width() - 2, (visible, painted.rect())
    assert visible.height() >= painted.height() - 2, (visible, painted.rect())
    assert painted.visibleRegion().contains(painted.rect().center())
    if ancestor is not None:
        box = rectangle(widget, ancestor)
        assert ancestor.rect().adjusted(-1, -1, 1, 1).contains(box), (box, ancestor.rect())


def reveal(window, widget):
    area = owning_scroll(widget)
    area.ensureWidgetVisible(widget, 0, 0)
    settle(window)
    assert area.horizontalScrollBar().maximum() == 0
    assert_whole(widget, area.viewport())
    return area


def close_window(window):
    window._analysis_controller.abort()
    window._teardown()
    window.close()
