"""The shared text editors over a frozen Matplotlib canvas, with page-owned state."""
from __future__ import annotations

from dataclasses import replace
import weakref

from PySide6 import QtCore

from ..plot.matplotlib_targets import panel_view, storage_key
from .figure_text import (AxisPropertiesPopover, AxisState, FigureTextController, TextTarget,
                          _alive)


class _TargetHandle(QtCore.QObject):
    """A Qt lifetime handle, never an artist or an exported graphics item."""
    def __init__(self, key, parent):
        super().__init__(parent)
        self.key = key


class MatplotlibAppearanceHost:
    independent_title = True
    independent_legend = True
    independent_text = True

    def __init__(self, widget, panel, label):
        self._widget = weakref.ref(widget)
        self.panel = panel
        self.pin_view = panel_view(panel)
        self.panel_label = label
        self._controller = lambda: None

    def storage_key(self, key):
        return storage_key(self.panel, key)

    def text_targets(self):
        controller, widget = self._controller(), self._widget()
        if controller is None or widget is None or not _alive(widget):
            return {}
        state = widget.state()
        return {entry.key: TextTarget(controller.handle(entry.key), entry.automatic,
                    state.text_style(self.storage_key(entry.key)), entry.noun, entry.capabilities)
                for entry in controller.descriptors()}

    def apply_text(self, key, style):
        widget = self._widget()
        target = self.text_targets().get(key)
        if widget is None or target is None:
            return
        if "edit" not in target.capabilities:
            style = replace(style, text=target.style.text)
        widget.apply_appearance(widget.state().with_text_style(self.storage_key(key), style))

    def apply_text_run(self, key, style):
        widget = self._widget()
        if widget is not None:
            widget.edit_text_style(self.storage_key(key), style)

    def commit_text_run(self):
        widget = self._widget()
        if widget is not None:
            widget._close_burst()

    def apply_format_styles(self, styles, *, auto_range_of=None):
        widget = self._widget()
        if widget is None:
            return
        state = widget.state()
        for key, style in styles.items():
            state = state.with_text_style(self.storage_key(key), style)
        if auto_range_of:
            state = state.without_axis_pin(self.pin_view, auto_range_of)
        widget.apply_appearance(state)

    def axis_state(self, side):
        controller, widget = self._controller(), self._widget()
        entry = next((a for a in controller.axes() if a.side == side), None) if controller else None
        if entry is None or widget is None:
            raise NotImplementedError("This target has no numerical axis")
        state = widget.state()
        state = self._axis_display_state(state, entry)
        pin = state.axis_pin(self.pin_view + "." + side)
        return AxisState(label=state.text_style(self.storage_key(f"axis.{side}.label")).text,
                         label_auto=entry.label, range=pin.range(), shown_range=entry.shown_range,
                         log=entry.log, log_available=False,
                         log_reason="The committed figure determines this axis scale; change it through the page's analysis controls.",
                         tick_pt=state.text_style(self.storage_key(f"axis.{side}.ticks")).size_pt,
                         tick_major=pin.tick_major, tick_major_available=entry.major_available,
                         label_locked=entry.label_locked)

    def _axis_display_state(self, state, entry):
        if entry.basis is None:
            return state
        from ..core.figure_quantity_basis import migrate_appearance_basis
        from ..core.quantity_display import AxisDisplayBasis
        from .plot_appearance import PlotAppearance
        key, basis = self.pin_view + "." + entry.side, entry.basis
        legacy = AxisDisplayBasis(replace(basis.quantity, factor=1., offset=0.,
                                         display_unit=basis.quantity.stored_unit), basis.coordinate_frame)
        return PlotAppearance.from_mapping(migrate_appearance_basis(state.to_dict(),
            {key: basis}, legacy_axes={key: legacy}))

    def apply_axis(self, side, values):
        current = self.axis_state(side)
        widget = self._widget()
        state = widget.state()
        entry = next((a for a in self._controller().axes() if a.side == side), None)
        if entry is not None:
            state = self._axis_display_state(state, entry)
        if values.range is not None and current.log and values.range[0] <= 0:
            raise ValueError("A logarithmic axis range must be positive")
        pin_key = self.pin_view + "." + side
        pin = state.axis_pin(pin_key)
        state = state.with_axis_pin(pin_key, replace(pin,
            min=values.range[0] if values.range else None,
            max=values.range[1] if values.range else None,
            tick_major=values.tick_major if current.tick_major_available else pin.tick_major))
        if not current.label_locked:
            key = self.storage_key(f"axis.{side}.label")
            state = state.with_text_style(key, replace(state.text_style(key), text=values.label))
        key = self.storage_key(f"axis.{side}.ticks")
        state = state.with_text_style(key, replace(state.text_style(key), size_pt=values.tick_pt))
        widget.apply_appearance(state)


class MatplotlibTextController(FigureTextController):
    """Same Format/inline/popover contract, mapped through the current canvas view."""
    geometryChanged = QtCore.Signal()

    def __init__(self, widget, panel, label, view, parent):
        QtCore.QObject.__init__(self, parent)
        self.panel = panel
        self.host = MatplotlibAppearanceHost(widget, panel, label)
        self.host._controller = weakref.ref(self)
        self._view_ref = weakref.ref(view)
        self._item_ref = lambda: None
        self._handles = {}
        self._retired = False
        self.editor = self.popover = self.last_menu = None
        self._legend = None
        view.viewChanged.connect(self.geometryChanged)
        view.horizontalScrollBar().valueChanged.connect(self.geometryChanged)
        view.verticalScrollBar().valueChanged.connect(self.geometryChanged)
        geometry = getattr(view, "geometryChanged", None)
        if geometry is not None:
            geometry.connect(self.geometryChanged)

    @property
    def plot(self):
        return None

    def _view(self):
        view = self._view_ref()
        return view if view is not None and _alive(view) and not self._retired else None

    def item(self):
        item = self._item_ref()
        return item if item is not None and _alive(item) else None

    def set_item(self, item):
        self._item_ref = weakref.ref(item)
        self.geometryChanged.emit()

    def descriptors(self):
        item = self.item()
        return tuple(t for t in item.text_targets() if t.panel == self.panel) if item else ()

    def axes(self):
        item = self.item()
        return tuple(a for a in item.axis_targets() if a.panel == self.panel) if item else ()

    def handle(self, key):
        if key not in self._handles:
            self._handles[key] = _TargetHandle(key, self)
        return self._handles[key]

    def anchor_rect(self, key):
        item, view = self.item(), self._view()
        if item is None or view is None:
            return None
        entries = [t for t in self.descriptors() if t.key == key or
                   (key == "legend" and t.key.startswith("legend.entry.")) or
                   (key.startswith("axis.") and key.count(".") == 1 and t.key.startswith(key + "."))]
        rect = QtCore.QRectF()
        for entry in entries:
            candidate = QtCore.QRectF(*entry.rect)
            rect = rect.united(candidate) if not rect.isEmpty() else candidate
        if key == "legend":
            frame = item.legend_rect(self.panel)
            if frame is not None:
                rect = QtCore.QRectF(*frame)
        if rect.isEmpty():
            return None
        return view.mapFromScene(item.mapRectToScene(rect)).boundingRect()

    def _anchor_rect(self, target):
        return self.anchor_rect(target.item.key) if target is not None and _alive(target.item) else None

    def open_axis_properties(self, side):
        view = self._view()
        if view is None:
            return None
        try:
            state = self.host.axis_state(side)
        except NotImplementedError:
            return None
        self.close_editors()
        popover = AxisPropertiesPopover(side, state, view.window())
        ref = weakref.ref(self)
        popover.committed.connect(lambda s, value, r=ref: r() and r()._apply_axis(s, value))
        self._place(popover, self.target(f"axis.{side}.ticks"))
        self.popover = popover
        self.opened.emit("axis", f"axis.{side}.ticks")
        return popover

    def retire(self):
        if self._retired:
            return
        self.close_editors()
        from .format_toolbar import retire
        for name in ("_nanoscope_outline", "_nanoscope_minibar"):
            retire(getattr(self, name, None))
            setattr(self, name, None)
        self._retired = True
        self._item_ref = lambda: None


def synchronize_controllers(owner, item, widget, view, previous=()):
    """Reuse panel controllers across redraws; old frozen items can be deleted safely."""
    existing = {c.panel: c for c in previous}
    panels = dict.fromkeys(t.panel for t in item.text_targets())
    result = []
    for panel in panels:
        label = next(t.panel_label for t in item.text_targets() if t.panel == panel)
        controller = existing.pop(panel, None)
        if controller is None:
            controller = MatplotlibTextController(widget, panel, label, view, owner)
        controller.set_item(item)
        result.append(controller)
    for controller in existing.values():
        controller.retire()
        controller.deleteLater()
    item.set_text_controllers(result)
    return result
