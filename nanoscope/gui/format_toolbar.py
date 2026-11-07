"""The floating text toolbar and the selection outline of the Format tab.

Both are CHILDREN OF A PLOT VIEW'S VIEWPORT (the pattern of ``plot_lasso._LassoOverlay`` and of
``figure_text.InlineTextEditor``), never scene items: a scene item reaches exports and freezes
live figure composition, a child widget of the viewport is in neither. Each is built LAZILY,
once per controller, at the first selection that needs it, and then only shown, hidden and
moved. Neither is a ``Qt.Popup`` (it would swallow the next canvas click, and the selection
could not follow it) nor a top-level window. No ``WA_DeleteOnClose``: they are retired with
``hide()`` then ``deleteLater()`` (:func:`retire`), and every later use goes through ``_alive``.

No Python event filter anywhere in this module (a filter on a view raised the segfault rate
about fourfold); positions follow signals (``ViewBox.sigResized``, the annotation's ``moved``,
the Format tab's own refresh).

The toolbar writes through ``FigureTextController.set_style_property``: it starts from the
style the element has NOW and changes the one property, so one gesture is one Undo step and a
run of size clicks is one step (``run=True``).
"""
from __future__ import annotations

import weakref

from PySide6 import QtCore, QtGui, QtWidgets

from .figure_text import TextStyleFields, _alive

__all__ = ["MiniTextToolbar", "SelectionOutline", "retire"]

_GAP = 4          # px between the toolbar and the element it floats over


def _viewport_rect(controller) -> QtCore.QRect | None:
    view = controller.view()
    return view.viewport().rect() if view is not None else None


class SelectionOutline(QtWidgets.QWidget):
    """A one-pixel outline around the selected element, painted in ``paintEvent`` and
    transparent to the mouse. One per controller, a child of its view's viewport."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("formatSelectionOutline")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._controller = None
        self._key: str | None = None
        self._hooked: set[int] = set()
        self.hide()

    # -- wiring -------------------------------------------------------------------------------
    def attach(self, controller) -> None:
        """Follow ``controller``'s plot (held weakly): its ViewBox resizing moves the outline."""
        self._controller = weakref.ref(controller)
        plot = controller.plot
        if plot is not None:
            plot.getViewBox().sigResized.connect(self.reposition)
            plot.getViewBox().sigRangeChanged.connect(self.reposition)
        else:
            geometry = getattr(controller, "geometryChanged", None)
            if geometry is not None:
                geometry.connect(self.reposition)

    def key(self) -> str | None:
        return self._key

    def show_key(self, key: str | None) -> None:
        """Outline element ``key`` (None hides it)."""
        self._key = key
        if key is not None:
            self._hook_item(key)
        self.reposition()

    def _hook_item(self, key: str) -> None:
        """An annotation that is dragged says where it ended (``moved``): follow it, once."""
        controller = self._controller() if self._controller is not None else None
        target = controller.target(key) if controller is not None else None
        item = target.item if target is not None else None
        if item is not None and _alive(item) and hasattr(item, "moved") \
                and id(item) not in self._hooked:
            self._hooked.add(id(item))
            item.moved.connect(self.reposition)

    def reposition(self, *_args) -> None:
        controller = self._controller() if self._controller is not None else None
        if self._key is None or controller is None or not _alive(controller):
            self.hide()
            return
        if self._key.startswith("series."):
            plot = getattr(controller, "plot", None)
            target = getattr(plot, "_nanoscope_series_targets", {}).get(self._key)
            view = controller.view()
            scene_rect = None
            bounds_for = getattr(plot, '_nanoscope_series_bounds', None)
            if callable(bounds_for):
                # Masked series can share one native graphics item. The owner
                # supplies only this series' visible footprint in scene units.
                scene_rect = bounds_for(self._key)
            else:
                for reference in target[1] if target else ():
                    item = reference()
                    if item is not None and _alive(item) and item.isVisible():
                        bounds = item.sceneBoundingRect()
                        # PlotDataItem is a container with an empty own rectangle;
                        # its curve/scatter children carry the visible geometry.
                        children = item.childrenBoundingRect()
                        if not children.isEmpty():
                            bounds = bounds.united(item.mapRectToScene(children))
                        scene_rect = bounds if scene_rect is None else scene_rect.united(bounds)
                if scene_rect is not None and plot is not None:
                    scene_rect = scene_rect.intersected(plot.getViewBox().sceneBoundingRect())
            rect = view.mapFromScene(scene_rect).boundingRect() if view is not None and scene_rect is not None else None
        else:
            rect = controller.anchor_rect(self._key)
        room = _viewport_rect(controller)
        if rect is None or room is None or rect.isEmpty():
            self.hide()
            return
        self.setGeometry(rect.adjusted(-2, -2, 2, 2).intersected(room))
        self.show()
        self.raise_()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.ColorRole.Highlight), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class MiniTextToolbar(QtWidgets.QFrame):
    """Font, size, B, I, colour, alignment and Edit text over a selected text: the same
    :class:`~nanoscope.gui.figure_text.TextStyleFields` the Format tab's panel holds, in one
    compact row. One per controller, built lazily at the first selection of a text element,
    a child of the view's viewport.

    Shown above the element, never over it; hidden when the selection clears, while the inline
    editor is open (``controller.editor``), and when the view hides (it is a child of the
    viewport). It stays open while the user works."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("formatMiniToolbar")
        self.setAccessibleName("Text format")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self._controller = None
        self._key: str | None = None
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(4, 3, 4, 3)
        row.setSpacing(4)
        self.fields = TextStyleFields(live=True, inherit_label="Figure font", compact=True)
        self.fields.changed.connect(self._written)
        self.fields.editingFinished.connect(self._finish_run)
        row.addWidget(self.fields)
        self.edit_button = QtWidgets.QToolButton()
        self.edit_button.setText("Edit text")
        self.edit_button.setAccessibleName("Edit text")
        self.edit_button.clicked.connect(self._edit)
        row.addWidget(self.edit_button)
        self.hide()

    # -- wiring -------------------------------------------------------------------------------
    def attach(self, controller) -> None:
        self._controller = weakref.ref(controller)
        plot = controller.plot
        if plot is not None:
            plot.getViewBox().sigResized.connect(self.reposition)
        else:
            geometry = getattr(controller, "geometryChanged", None)
            if geometry is not None:
                geometry.connect(self.reposition)
        controller.editingChanged.connect(self.reposition)

    def key(self) -> str | None:
        return self._key

    def show_key(self, key: str | None) -> None:
        """Float over element ``key`` (None hides the toolbar)."""
        self._key = key
        self.refresh()

    def refresh(self) -> None:
        """Re-read the element's style into the controls, then place the toolbar."""
        controller = self._controller() if self._controller is not None else None
        target = controller.target(self._key) if controller is not None and self._key else None
        caps = target.capabilities if target is not None else frozenset()
        if target is None or not ({"style", "edit"} & caps):
            self.hide()
            return
        self.fields.set_target(self._key, target)
        self.fields.setVisible("style" in caps)
        self.edit_button.setVisible("edit" in caps)
        self.reposition()

    def reposition(self, *_args) -> None:
        controller = self._controller() if self._controller is not None else None
        if self._key is None or controller is None or not _alive(controller):
            self.hide()
            return
        if _alive(controller.editor):           # the inline editor is open over the element
            self.hide()
            return
        rect = controller.anchor_rect(self._key)
        room = _viewport_rect(controller)
        if rect is None or room is None or rect.isEmpty():
            self.hide()
            return
        margin = self.layout().contentsMargins()
        edit_width = self.edit_button.sizeHint().width() if not self.edit_button.isHidden() else 0
        self.fields.fit_compact_width(max(
            1, room.width() - margin.left() - margin.right() - edit_width
            - self.layout().spacing() - 2 * self.frameWidth()))
        self.adjustSize()
        width, height = self.width(), self.height()
        # Keep the next editable label reachable too. A title spans the canvas; a
        # left-aligned bar below it used to cover Noise's entire fit label. Try both
        # alignments before moving farther away, then minimise overlap with other text.
        # Axis tick targets use a bounding rect extending across the plot, rather
        # than just their painted glyphs. Treating that whole region as text can
        # favour covering an actual label along the edge of the view.
        others = [anchor for key in controller.targets() if key != self._key
                  and not key.endswith(".ticks")
                  if (anchor := controller.anchor_rect(key)) is not None
                  and not anchor.isEmpty()]
        # Samples toggle series visibility but are intentionally not editable text
        # targets. Keep the whole legend reachable, including those sample glyphs.
        legend = controller.anchor_rect("legend")
        if legend is not None and not legend.isEmpty():
            others.append(legend)
        preferred = [(rect.left(), rect.top() - height - _GAP),
                     (rect.right() - width + 1, rect.top() - height - _GAP),
                     (rect.left(), rect.bottom() + _GAP),
                     (rect.right() - width + 1, rect.bottom() + _GAP),
                     (rect.right() + _GAP, rect.top()),
                     (rect.left() - width - _GAP, rect.top())]
        obstacles = [rect, *others]
        # A free rectangular placement, if one exists, can be slid to a room or
        # obstacle edge. Include those interior bands instead of trading a legend
        # sample for an axis label when the few near-title positions are occupied.
        xs = {0, room.width() - width}
        ys = {0, room.height() - height}
        for obstacle in obstacles:
            xs.update((obstacle.left() - width - _GAP, obstacle.right() + _GAP))
            ys.update((obstacle.top() - height - _GAP, obstacle.bottom() + _GAP))
        edges = sorted(((x, y) for x in xs for y in ys),
                       key=lambda pos: abs(pos[0] - rect.left()) + abs(pos[1] - rect.top()))
        seen = set()
        chosen = None
        for x, y in [*preferred, *edges]:
            x = max(0, min(x, room.width() - width))
            y = max(0, min(y, room.height() - height))
            if (x, y) in seen:
                continue
            seen.add((x, y))
            candidate = QtCore.QRect(x, y, width, height)
            if room.contains(candidate) and not any(candidate.intersects(anchor)
                                                    for anchor in obstacles):
                chosen = candidate
                break
        if chosen is None:
            # Format remains available when a tiny viewport has no unobstructed room.
            self.hide()
            return
        self.move(chosen.topLeft())
        self.show()
        self.raise_()

    # -- writes -------------------------------------------------------------------------------
    def _written(self, prop: str, value) -> None:
        controller = self._controller() if self._controller is not None else None
        if controller is None or not _alive(controller) or self._key is None:
            return
        controller.set_style_property(self._key, prop, value, run=(prop == "size_pt"))

    def _edit(self, _checked: bool = False) -> None:
        controller = self._controller() if self._controller is not None else None
        if controller is not None and _alive(controller) and self._key is not None:
            controller.edit_inline(self._key)

    def _finish_run(self) -> None:
        controller = self._controller() if self._controller is not None else None
        if controller is not None and _alive(controller):
            controller.finish_style_run()


def retire(widget: QtWidgets.QWidget | None) -> None:
    """Hide ``widget`` and hand it to ``deleteLater`` (never ``WA_DeleteOnClose``)."""
    if widget is not None and _alive(widget):
        widget.hide()
        widget.deleteLater()
