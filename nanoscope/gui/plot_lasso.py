"""Selection-only lasso for a live pyqtgraph canvas.

The temporary outline is a viewport overlay, not a scientific graphics item. It
never enters a figure export or changes the ViewBox, plotted arrays, or fit state.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from PySide6 import QtCore, QtGui, QtWidgets


class RowIdentityIndex:
    """Fast membership for large event selections, using the host's exact equality.

    Hashes only partition candidates; they never replace identity comparison. The
    fallback retains support for unusual unhashable dataframe index labels.
    """
    def __init__(self, rows: Iterable, equal: Callable):
        self.equal = equal
        self.buckets = defaultdict(list)
        self.unhashable = []
        for row in rows:
            self.add(row)

    def add(self, row):
        try:
            key = hash(row)
        except TypeError:
            self.unhashable.append(row)
        else:
            self.buckets[key].append(row)

    def contains(self, row) -> bool:
        try:
            candidates = self.buckets.get(hash(row), ())
        except TypeError:
            candidates = (value for values in self.buckets.values() for value in values)
        return (any(self.equal(row, old) for old in candidates)
                or any(self.equal(row, old) for old in self.unhashable))


def unique_rows(rows: Iterable, equal: Callable) -> list:
    index = RowIdentityIndex((), equal)
    result = []
    for row in rows:
        if not index.contains(row):
            result.append(row)
            index.add(row)
    return result


def toggled_rows(previous: Iterable, incoming: Iterable, equal: Callable) -> list:
    """Toggle an outlined set once per identity, retaining stable display order."""
    old, new = unique_rows(previous, equal), unique_rows(incoming, equal)
    old_index, new_index = RowIdentityIndex(old, equal), RowIdentityIndex(new, equal)
    return ([row for row in old if not new_index.contains(row)]
            + [row for row in new if not old_index.contains(row)])


class _LassoOverlay(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.polygon = QtGui.QPolygonF()
        self.clip = QtCore.QRectF()
        self.hide()

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.clip)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e69f00"), 2,
                                  QtCore.Qt.PenStyle.DashLine))
        painter.setBrush(QtGui.QColor(230, 159, 0, 32))
        painter.drawPolygon(self.polygon, QtCore.Qt.FillRule.OddEvenFill)


class PlotLasso(QtCore.QObject):
    """Intercept only a lasso gesture; ordinary pan/zoom works when disabled.

    Emits a closed polygon in SCENE coordinates and the modifiers held at press.
    Esc/right-click cancel and restore pan/zoom; resize, hide, or a genuine range
    change cancel an incomplete outline so it cannot select a different picture.
    """
    completed = QtCore.Signal(object, object)
    enabledChanged = QtCore.Signal(bool)
    cancelled = QtCore.Signal(str)
    MAX_POINTS = 4096

    def __init__(self, plot_widget, parent=None):
        super().__init__(parent or plot_widget)
        self.widget = plot_widget
        self.viewport = plot_widget.viewport()
        self.view = plot_widget.getPlotItem().getViewBox()
        self.overlay = _LassoOverlay(self.viewport)
        self.enabled = False
        self._drawing = False
        self._points = []
        self._modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        self._consume_left_release = False
        self._consume_right_release = False
        self._suppress_context_menu = False
        self._cursor = None
        self.viewport.installEventFilter(self)
        self.widget.installEventFilter(self)
        self.view.sigRangeChanged.connect(self._range_changed)

    @property
    def drawing(self):
        return self._drawing

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        if not enabled:
            self.cancel()
        self.enabled = enabled
        if enabled:
            self._cursor = self.viewport.cursor()
            self.viewport.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        elif self._cursor is not None:
            self.viewport.setCursor(self._cursor)
            self._cursor = None
        self.enabledChanged.emit(enabled)

    def cancel(self, reason: str = "") -> None:
        was_drawing = self._drawing
        self._drawing = False
        self._points = []
        self.overlay.hide()
        self.overlay.polygon = QtGui.QPolygonF()
        if reason and was_drawing:
            self.cancelled.emit(reason)

    def _range_changed(self, *_args):
        if self._drawing:
            self.cancel("Lasso cancelled because the visible range changed.")

    def _data_rect(self):
        return QtCore.QRectF(self.widget.mapFromScene(self.view.sceneBoundingRect()).boundingRect())

    def _position(self, event):
        point = QtCore.QPointF(event.position())
        rect = self._data_rect()
        return QtCore.QPointF(min(max(point.x(), rect.left()), rect.right()),
                             min(max(point.y(), rect.top()), rect.bottom()))

    def _extend(self, event, *, force=False):
        point = self._position(event)
        if self._points and not force:
            delta = point - self._points[-1]
            if delta.x() ** 2 + delta.y() ** 2 < 4:
                return
        if len(self._points) >= self.MAX_POINTS:
            self.cancel("Lasso outline is too long; draw a shorter outline. Selection is unchanged.")
            return
        self._points.append(point)
        self.overlay.polygon = QtGui.QPolygonF(self._points)
        self.overlay.update()

    def eventFilter(self, watched, event):  # noqa: N802
        kind = event.type()
        if watched is self.viewport:
            if kind == QtCore.QEvent.Type.ContextMenu and self._suppress_context_menu:
                self._suppress_context_menu = False
                return True
            if kind == QtCore.QEvent.Type.Resize:
                self.cancel("Lasso cancelled because the canvas was resized.")
                self.overlay.setGeometry(self.viewport.rect())
            elif kind == QtCore.QEvent.Type.MouseButtonRelease:
                if event.button() == QtCore.Qt.MouseButton.RightButton and self._consume_right_release:
                    self._consume_right_release = False
                    return True
                if event.button() == QtCore.Qt.MouseButton.LeftButton and self._consume_left_release:
                    self._consume_left_release = False
                    if self._drawing:
                        self._extend(event, force=True)
                        points, modifiers = list(self._points), self._modifiers
                        self.cancel()
                        polygon = QtGui.QPolygonF(points)
                        if (len(points) >= 3 and polygon.boundingRect().width() >= 4
                                and polygon.boundingRect().height() >= 4):
                            scene = QtGui.QPolygonF([
                                self.widget.mapToScene(point.toPoint()) for point in points])
                            self.completed.emit(scene, modifiers)
                    return True
            elif kind == QtCore.QEvent.Type.MouseMove and self._drawing:
                self._extend(event)
                return True
            elif kind == QtCore.QEvent.Type.Wheel and self._drawing:
                return True
        if kind in (QtCore.QEvent.Type.Hide, QtCore.QEvent.Type.WindowDeactivate):
            self.cancel("Lasso cancelled because the canvas lost focus.")
        if not self.enabled:
            return super().eventFilter(watched, event)
        if kind == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel("Lasso cancelled; pan/zoom restored.")
            self.set_enabled(False)
            return True
        if watched is self.viewport:
            if kind == QtCore.QEvent.Type.ContextMenu:
                return True
            if kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonDblClick):
                if event.button() == QtCore.Qt.MouseButton.RightButton:
                    self._consume_right_release = True
                    # Qt can deliver ContextMenu after the release, when this
                    # action has already restored pan/zoom. Do not open it then.
                    self._suppress_context_menu = True
                    self.cancel("Lasso cancelled; pan/zoom restored.")
                    self.set_enabled(False)
                    return True
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._suppress_context_menu = False
                    if not self._data_rect().contains(event.position()):
                        return super().eventFilter(watched, event)
                    self._consume_left_release = True
                    self._drawing = True
                    self._points = []
                    self._modifiers = event.modifiers()
                    self.overlay.setGeometry(self.viewport.rect())
                    self.overlay.clip = self._data_rect()
                    self._extend(event, force=True)
                    self.overlay.show()
                    self.overlay.raise_()
                    self.widget.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                    return True
        return super().eventFilter(watched, event)
