"""Release the old container's ownership before moving an existing inspector control."""
from __future__ import annotations

import shiboken6
from PySide6 import QtWidgets


def take_from_scroll_area(widget: QtWidgets.QWidget) -> None:
    """Clear a scroll area's content pointer before its content is reparented."""
    viewport = widget.parentWidget()
    area = viewport.parentWidget() if viewport is not None else None
    if isinstance(area, QtWidgets.QScrollArea) and area.widget() is widget:
        area.takeWidget()


def prepare_widget_move(widget: QtWidgets.QWidget) -> None:
    """Delete only the widget's detached layout item through its Python binding.

    Implicit reparenting lets Qt delete the old QWidgetItem without invalidating an
    existing PySide wrapper. Its stale native address can then alias another object.
    takeAt transfers ownership to Python; deleting that exact item invalidates its
    wrapper while leaving the widget, its children and connections alive.

    Search only layouts owned by the widget's parent, using indexOf(widget), so the
    search does not create or retain wrappers for unrelated layout items.
    """
    take_from_scroll_area(widget)
    parent = widget.parentWidget()
    root = parent.layout() if parent is not None else None
    if root is None:
        return
    for layout in [root, *root.findChildren(QtWidgets.QLayout)]:
        index = layout.indexOf(widget)
        if index < 0:
            continue
        item = layout.takeAt(index)
        if not isinstance(item, QtWidgets.QWidgetItem) or item.widget() is not widget:
            raise RuntimeError("The source layout did not return the widget's layout item")
        shiboken6.delete(item)
        return
