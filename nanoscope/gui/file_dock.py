from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.document import Document
from ..core.layer import ViewMode
from ..core.segment import SegmentDocument
from ..core.session import Session
from ..core.types import ChannelKind, Provenance

__all__ = ["FileDock"]

_COL_NAME = 0
_COL_UNITS = 1
_COL_COLOR = 2
_COL_DT = 3

# Qt.UserRole payloads distinguish the three tree levels.
_ROLE = QtCore.Qt.UserRole

# On every channel row. The tick box is the ONLY way to put two recordings on the same axes now
# that opening one no longer does it behind the user's back, so it says so where it is used.
_CHANNEL_TIP = (
    "Tick to draw this channel on the trace.\n"
    "Ticking channels from more than one recording draws them together — that is how you "
    "compare two runs.\nUntick to take the channel off the trace.")

# On every document row, under its path/provenance. States both halves of the new rule: clicking
# switches (it does not accumulate), and the recordings you switch away from are still open.
_DOC_TIP = ("Click to show this recording — the others stay open and listed.\n"
            "Right-click ▸ “Add to view” (or tick a channel) to draw it alongside for comparison.")


def _default_channel(doc: Document) -> int:
    """The channel to select when a document (not a specific channel) is chosen. Prefer the first
    live CURRENT channel — it is what you analyse and calibrate, and it lets detection run out of the
    box — then any live channel, falling back to the first channel only if every channel is dead."""
    channels = doc.meta.channels
    for info in channels:
        if info.kind is ChannelKind.CURRENT and not info.is_dead:
            return info.index
    for info in channels:
        if not info.is_dead:
            return info.index
    return channels[0].index


class _HintLabel(QtWidgets.QLabel):
    """A one-line hint that ELIDES instead of wrapping.

    Both halves matter for this dock. Wrapping would let a squeezed dock grow the hint to three
    lines and take that height out of the file tree; and a plain non-wrapping QLabel reports its
    whole text width as a minimum, which is exactly the floor on ``QDockWidget::minimumSize()``
    the toolbar had to be rescued from. Ignored horizontal policy contributes 0 to that floor,
    Fixed vertical pins it to one line, and the full sentence stays available as the tooltip.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self._full = text
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(
            self._full, QtCore.Qt.ElideRight, max(0, self.width())))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._elide()


class _ColorSwatch(QtWidgets.QFrame):
    """A small clickable colour chip showing (and letting the user change) a Layer's colour —
    essential for telling overlaid comparison traces apart."""

    clicked = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(26, 14)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Click to change this layer's colour")

    def set_color(self, color: str) -> None:
        self.setStyleSheet(
            f"background:{color}; border:1px solid rgba(0,0,0,0.5); border-radius:3px;")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class FileDock(QtWidgets.QDockWidget):
    """Documents grouped by nominal voltage; each channel is a checkbox meaning IS IT DRAWN.

    The checkbox is ticked when the channel has a Layer *and* that Layer is visible, because
    "ticked" has to mean the one thing the user can see: this channel is on the trace. Ticking
    adds the layer (if it is new) and makes it visible — that is the explicit opt-in for
    comparing two recordings. Unticking REMOVES the layer, because unticking is a decision about
    the channel ("not this one"), not about which recording is on screen; switching recordings is
    MainWindow's job and it only moves visibility, so a recording switched away from keeps its
    colours and its alignment.

    Subscribes to Session only; never imports another panel. Exposes `selectionChanged(doc_id, ch)`
    so MainWindow can drive the calibration/info panels — that is panel→window wiring, not
    panel→panel.
    """

    selectionChanged = QtCore.Signal(str, int)   # doc_id, channel
    # Right-click ▸ "Add to view" — draw every non-dead channel of this recording ALONGSIDE what
    # is already shown. ANNOUNCED, not performed: what "shown" means is MainWindow's policy (see
    # THE SHOWN-RECORDING RULE), and this dock does not get to hold a second copy of it.
    addToViewRequested = QtCore.Signal(str)       # doc_id
    showOnlyRequested = QtCore.Signal(str)        # doc_id  — right-click ▸ "Show only this"
    createSegmentRequested = QtCore.Signal()      # "Segment" clicked — MainWindow reads the region
    # "Close" clicked — ANNOUNCED, not performed. Closing a recording that has segments cut from
    # it is a data-loss decision (a segment can only be persisted while its parent is open), and
    # that policy belongs to MainWindow, the sole wirer — exactly like createSegmentRequested.
    closeDocumentRequested = QtCore.Signal(str)   # doc_id
    # The user EDITED a layer's view state — ticked or unticked a channel, dragged its Δt, picked
    # its colour. All three are written into the project file, so losing one silently is the same
    # class of loss as an unsaved detection; MainWindow arms the quit guard on this.
    #
    # It is deliberately NOT `Session.layersChanged`, which cannot tell an edit from the layer
    # churn of opening, closing or switching between recordings — connecting the quit guard to
    # that signal would rebuild, by a third door, the exact "open a file, look at it, quit,
    # answer a modal" nag that B3 and C29 removed. The line drawn here is *edits are work,
    # navigation is not*: what the user did to a layer arms the guard, what merely changed which
    # layer is on screen does not.
    viewStateEdited = QtCore.Signal()

    def __init__(self, session: Session, parent=None) -> None:
        # Titled for what it is — the left column's file manager (recordings, their segments
        # and their channels). The objectName is unchanged, so saveState/restoreState still
        # match this dock across the rename.
        super().__init__("File Manager", parent)
        self.setObjectName("FileDock")
        self._session = session
        self._updating = False
        # layer_id -> per-row widgets, for in-place updates without a full rebuild.
        self._swatches: dict[str, _ColorSwatch] = {}
        self._spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        #: Every widget handed to ``QTreeWidget.setItemWidget`` — the CENTERING WRAPPER
        #: for the colour column, and the spin box itself for Δt. Kept because
        #: ``QTreeWidget.clear()`` deletes the ITEMS and leaves these alive, parented to
        #: the viewport, where nothing will ever reach them again. Filled by
        #: :meth:`_install_row_widget` (the only installer); drained by :meth:`_rebuild`.
        self._row_widgets: list[QtWidgets.QWidget] = []

        container = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # The toolbar is a ROW OF BUTTONS, and a row of buttons never wraps: ``qSmartMinSize``
        # will not take a push button below its own text, so the row's ~293 px width propagated
        # straight into ``QDockWidget::minimumSize()`` — the exact bound ``QDockAreaLayout``
        # refuses to drag a separator past. That is the same floor the Inspector was fixed for,
        # and once this dock was given the LEFT COLUMN it became the floor on how much width the
        # user can hand back to the signal viewer. Same remedy: park the row in a
        # ``widgetResizable`` QScrollArea with an explicit zero minimum, so it keeps its natural
        # width and scrolls horizontally instead of pushing the dock back open. The tree below
        # scrolls on its own and asks for very little, so the dock now floors near it.
        bar = self._build_toolbar()
        bar.setContentsMargins(0, 0, 0, 0)
        self._toolbar_host = QtWidgets.QWidget()
        self._toolbar_host.setLayout(bar)
        self._toolbar_scroll = QtWidgets.QScrollArea(container)
        self._toolbar_scroll.setWidgetResizable(True)
        self._toolbar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._toolbar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._toolbar_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._toolbar_scroll.setMinimumWidth(0)
        self._toolbar_scroll.setWidget(self._toolbar_host)
        # Exactly one row high, growing by the scrollbar's own height only while the row is
        # actually being scrolled — so a roomy dock spends no pixels on the affordance and a
        # squeezed one never steals the tree's height to show it.
        self._toolbar_scroll.horizontalScrollBar().rangeChanged.connect(
            lambda *_: self._sync_toolbar_height())
        self._sync_toolbar_height()
        outer.addWidget(self._toolbar_scroll)

        self._tree = QtWidgets.QTreeWidget(container)
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Channel", "Units", "Colour", "Δt (s)"])
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(_COL_NAME, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_UNITS, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_COLOR, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(_COL_DT, QtWidgets.QHeaderView.ResizeToContents)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        outer.addWidget(self._tree, 1)

        # The compare affordance, said out loud. Clicking a recording now shows THAT recording,
        # so the way to put two on the same axes is the tick box — and nobody guesses that from
        # an unlabelled checkbox. One line, under the tree, wrapped, dimmed.
        self._hint = _HintLabel(
            "Click a recording to show it · tick channels to draw them together")
        self._hint.setStyleSheet("color:#7a828b; font-size:10px;")
        self._hint.setToolTip(
            "Opening or clicking a recording displays it on its own; the others stay open.\n"
            "To compare recordings, tick the channel of each one you want drawn — or right-click "
            "a recording and choose “Add to view”.\nOverlay/Stacked then decide whether the "
            "ticked channels share a pane or get one each.")
        outer.addWidget(self._hint)

        self.setWidget(container)

        session.documentOpened.connect(lambda _d: self._rebuild())
        session.documentClosed.connect(lambda _d: self._rebuild())
        session.layersChanged.connect(self._sync_layers)
        session.viewModeChanged.connect(self._sync_view_mode)

        self._rebuild()

    # -- toolbar ---------------------------------------------------------------------------------
    def _sync_toolbar_height(self) -> None:
        """Keep the toolbar strip one row tall, plus the scrollbar only while it is showing."""
        sb = self._toolbar_scroll.horizontalScrollBar()
        extra = sb.sizeHint().height() if sb.maximum() > 0 else 0
        self._toolbar_scroll.setFixedHeight(self._toolbar_host.sizeHint().height() + extra)

    def _build_toolbar(self) -> QtWidgets.QHBoxLayout:
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)

        self._btn_overlay = QtWidgets.QPushButton("Overlay")
        self._btn_stacked = QtWidgets.QPushButton("Stacked")
        for btn in (self._btn_overlay, self._btn_stacked):
            btn.setCheckable(True)
            btn.setToolTip("Overlay: group comparable channels into one x-linked pane.\n"
                           "Stacked: one pane per channel.")
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._btn_overlay)
        self._mode_group.addButton(self._btn_stacked)
        self._btn_overlay.clicked.connect(lambda: self._session.set_view_mode(ViewMode.OVERLAY))
        self._btn_stacked.clicked.connect(lambda: self._session.set_view_mode(ViewMode.STACKED))

        bar.addWidget(self._btn_overlay)
        bar.addWidget(self._btn_stacked)
        bar.addStretch(1)

        # "Segment" — crop the current on-trace selection into a new derived recording. Gated by
        # MainWindow (via set_segment_enabled) on an active recording + a live Measure region; the
        # click only announces intent (MainWindow reads the region + creates the segment), so this
        # dock imports no sibling panel and MainWindow stays the sole wirer.
        self._btn_segment = QtWidgets.QToolButton()
        self._btn_segment.setText("Segment")
        self._btn_segment.setToolTip(
            "Create a new derived recording from the current on-trace selection.\n"
            "Turn on Measure (Analysis ▸ Measure event) and drag a region first. The "
            "segment is analysed independently of the parent.")
        self._btn_segment.setEnabled(False)
        self._btn_segment.clicked.connect(self.createSegmentRequested)
        bar.addWidget(self._btn_segment)

        self._btn_close = QtWidgets.QToolButton()
        self._btn_close.setText("Close")
        self._btn_close.setToolTip("Close the selected document")
        self._btn_close.clicked.connect(self._close_selected)
        bar.addWidget(self._btn_close)

        self._sync_view_mode()
        return bar

    def _sync_view_mode(self) -> None:
        is_overlay = self._session.view_mode is ViewMode.OVERLAY
        self._btn_overlay.setChecked(is_overlay)
        self._btn_stacked.setChecked(not is_overlay)

    # -- tree construction -----------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Rebuild the whole tree from the session's documents.

        ``QTreeWidget.clear()`` deletes the items; it does NOT delete the widgets those
        items carried. A widget installed with ``setItemWidget`` is re-parented to the
        tree's VIEWPORT, and it stays there, alive and invisible, after the item that
        owned it is gone. This method runs on every ``documentOpened`` AND every
        ``documentClosed``, so the residue accumulated on the most ordinary loop in the
        app: measured at +6 colour swatches and +6 Δt spin boxes per open/close cycle, and
        74 swatches and 105 spin boxes still parented to the viewport after closing EVERY
        document — a dock listing nothing, holding 179 live widgets.

        So the widgets are tracked as they are installed and destroyed here, before the
        items that reference them go away.
        """
        self._updating = True
        try:
            for w in self._row_widgets:
                w.setParent(None)
                w.deleteLater()
            self._row_widgets.clear()
            self._tree.clear()
            self._swatches.clear()
            self._spins.clear()

            groups: dict[float | None, list[Document]] = {}
            for doc in self._session.documents():
                groups.setdefault(doc.meta.voltage_group_mV, []).append(doc)

            known = sorted((g for g in groups if g is not None), reverse=True)
            ordered: list[float | None] = list(known)
            if None in groups:
                ordered.append(None)

            for gv in ordered:
                group_item = QtWidgets.QTreeWidgetItem([self._group_label(gv)])
                group_item.setFirstColumnSpanned(True)
                group_item.setData(_COL_NAME, _ROLE, ("group",))
                font = group_item.font(_COL_NAME)
                font.setBold(True)
                group_item.setFont(_COL_NAME, font)
                group_item.setFlags(QtCore.Qt.ItemIsEnabled)
                self._tree.addTopLevelItem(group_item)
                for doc in groups[gv]:
                    self._add_document(group_item, doc)
                group_item.setExpanded(True)
        finally:
            self._updating = False

        self._sync_layers()

    @staticmethod
    def _group_label(gv: float | None) -> str:
        return "Unknown voltage" if gv is None else f"{gv:.0f} mV"

    def _add_document(self, parent: QtWidgets.QTreeWidgetItem, doc: Document) -> None:
        meta = doc.meta
        label = doc.display_name
        if meta.is_partial and meta.loaded_range is not None:
            label += f"  (partial: samples {meta.loaded_range.start:,}–{meta.loaded_range.stop:,})"

        doc_item = QtWidgets.QTreeWidgetItem([label])
        doc_item.setData(_COL_NAME, _ROLE, ("doc", doc.doc_id))
        doc_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        doc_item.setToolTip(_COL_NAME, f"{meta.path}\n\n{_DOC_TIP}")

        if isinstance(doc, SegmentDocument):
            # A derived recording: mark it distinctly (its name already encodes the range) and
            # give it a tooltip tracing it back to the parent + window it was cut from.
            o = doc.origin
            parent_name = Path(o.parent_key).name or o.parent_key
            doc_item.setText(_COL_NAME, "✂ " + label)
            font = doc_item.font(_COL_NAME)
            font.setItalic(True)
            doc_item.setFont(_COL_NAME, font)
            doc_item.setForeground(_COL_NAME, QtGui.QBrush(QtGui.QColor("#4DB6AC")))
            when = f" · t₀={o.t0_abs_s:.6g}s" if meta.sample_rate_hz is not None else ""
            doc_item.setToolTip(
                _COL_NAME,
                f"segment of {parent_name}\nsamples {o.i0:,}–{o.i1:,}{when}\n\n{_DOC_TIP}")
        # Voltage inferred from the filename is UNRELIABLE — render it visibly differently.
        elif meta.voltage_provenance is Provenance.FILENAME:
            font = doc_item.font(_COL_NAME)
            font.setItalic(True)
            doc_item.setFont(_COL_NAME, font)
            doc_item.setToolTip(
                _COL_NAME,
                f"{meta.path}\nvoltage inferred from filename (unreliable)\n\n{_DOC_TIP}")

        parent.addChild(doc_item)

        for info in meta.channels:
            self._add_channel(doc_item, doc, info)
        doc_item.setExpanded(True)

    def _add_channel(self, parent: QtWidgets.QTreeWidgetItem, doc: Document, info) -> None:
        units = info.raw_units or "—"
        item = QtWidgets.QTreeWidgetItem([info.name, units, "", ""])
        item.setData(_COL_NAME, _ROLE, ("channel", doc.doc_id, info.index))
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                      | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(_COL_NAME, QtCore.Qt.Checked
                           if self._is_shown(doc.doc_id, info.index) else QtCore.Qt.Unchecked)
        item.setToolTip(_COL_NAME, _CHANNEL_TIP)

        if info.is_dead:
            grey = QtGui.QBrush(QtGui.QColor("#7a828b"))
            for col in (_COL_NAME, _COL_UNITS):
                item.setForeground(col, grey)
            tip = info.dead_reason or "flagged dead (no signal)"
            item.setToolTip(_COL_NAME, f"{info.name} — {tip}\n\n{_CHANNEL_TIP}")
            item.setText(_COL_UNITS, units + "  ·  dead")

        parent.addChild(item)

        layer_id = f"{doc.doc_id}#{info.index}"

        swatch = _ColorSwatch()
        swatch.clicked.connect(lambda lid=layer_id: self._pick_color(lid))
        wrap = self._center(swatch)
        self._install_row_widget(item, _COL_COLOR, wrap)   # the WRAPPER is what the item holds
        self._swatches[layer_id] = swatch

        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e6, 1e6)
        spin.setSingleStep(0.001)
        spin.setSuffix(" s")
        spin.setKeyboardTracking(False)
        spin.setToolTip("Shift this layer along the time axis to align it with others")
        spin.valueChanged.connect(lambda val, lid=layer_id: self._on_offset_changed(lid, val))
        self._install_row_widget(item, _COL_DT, spin)
        self._spins[layer_id] = spin

    def _install_row_widget(self, item: QtWidgets.QTreeWidgetItem, column: int,
                            widget: QtWidgets.QWidget) -> None:
        """The ONE place a widget is handed to ``setItemWidget`` — so the two rules that apply
        to every embedded row widget are stated once.

        1. Track it. ``QTreeWidget.clear()`` deletes the items and leaves these widgets alive on
           the viewport; :meth:`_rebuild` destroys what this list holds.
        2. Take it OUT OF THE TAB CHAIN (click focus only). ``QAbstractItemView::eventFilter``
           turns a *focus-in* on a persistent index widget into ``setCurrentIndex(...)`` — it
           cannot tell a click from a Tab. So Tab out of any text field elsewhere in the window
           (measured: the Preprocess ▸ notch "lines" box, the natural way to commit that field)
           walked into a row's Δt spin, and the tree then SELECTED that row: on the owner's
           −1000 mV recording the app selection moved IN15 → IN14, the voltage channel, taking
           the Calibration panel — the panel whose Apply writes a ``.npcal.json`` beside the
           user's recording — with it, while the Analysis panel stayed on IN15 because it
           resolves through ``Session.resolve_current_channel``. A keystroke every form user
           makes must not repoint the app, so Tab skips these; a click still focuses and edits
           them, and clicking a row's own control legitimately selects that row.

        The Δt spin's Qt default was ``WheelFocus`` (15), so a wheel over it could grab focus —
        and therefore the selection — as well; ``ClickFocus`` (2) drops that too. Measured on
        both builds, a wheel delivered to the unfocused spin changes neither its value nor the
        selection, so this costs the user nothing.
        """
        if widget.focusPolicy() != QtCore.Qt.NoFocus:
            widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        for child in widget.findChildren(QtWidgets.QWidget):
            if child.focusPolicy() != QtCore.Qt.NoFocus:
                child.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._tree.setItemWidget(item, column, widget)
        self._row_widgets.append(widget)

    @staticmethod
    def _center(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(widget)
        lay.addStretch(1)
        return wrap

    # -- syncing (in place, no rebuild — so an open spinbox edit is never destroyed) -------------
    def _is_shown(self, doc_id: str, ch: int) -> bool:
        """Is this channel actually ON THE TRACE — i.e. does it have a Layer that is visible?
        The check box, the colour swatch and the Δt spin all answer to this one predicate, so a
        tick always means "you can see it" and never "there is a hidden layer for it"."""
        layer = self._session.layer(f"{doc_id}#{ch}")
        return layer is not None and layer.visible

    def _sync_layers(self) -> None:
        self._updating = True
        try:
            it = QtWidgets.QTreeWidgetItemIterator(self._tree)
            while it.value():
                item = it.value()
                payload = item.data(_COL_NAME, _ROLE)
                if payload and payload[0] == "channel":
                    _, doc_id, ch = payload
                    layer = self._session.layer(f"{doc_id}#{ch}")
                    shown = layer is not None and layer.visible
                    item.setCheckState(_COL_NAME,
                                       QtCore.Qt.Checked if shown else QtCore.Qt.Unchecked)
                    self._update_row_widgets(doc_id, ch, layer if shown else None)
                it += 1
        finally:
            self._updating = False

    def _update_row_widgets(self, doc_id: str, ch: int, layer) -> None:
        layer_id = f"{doc_id}#{ch}"
        swatch = self._swatches.get(layer_id)
        if swatch is not None:
            if layer is not None:
                swatch.setVisible(True)
                swatch.set_color(layer.color)
            else:
                swatch.setVisible(False)

        spin = self._spins.get(layer_id)
        if spin is not None:
            rate_known = self._session.document(doc_id).meta.sample_rate_hz is not None
            spin.setEnabled(layer is not None and rate_known)
            if layer is not None:
                blocked = spin.blockSignals(True)
                spin.setValue(layer.time_offset_s)
                spin.blockSignals(blocked)
            if not rate_known:
                spin.setToolTip("This file declares no sample rate; alignment is unavailable.")

    # -- interaction -----------------------------------------------------------------------------
    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if self._updating or column != _COL_NAME:
            return
        payload = item.data(_COL_NAME, _ROLE)
        if not payload or payload[0] != "channel":
            return
        _, doc_id, ch = payload
        checked = item.checkState(_COL_NAME) == QtCore.Qt.Checked
        if checked:
            # THE EXPLICIT OPT-IN TO COMPARING. Ticking a channel of a recording that is not the
            # one on screen draws it ALONGSIDE — it never switches to it. (Qt sends a click on a
            # check indicator to the delegate without moving the current item, so this cannot be
            # mistaken by MainWindow for "select that recording", which would switch away from
            # whatever we are trying to overlay onto.)
            layer = self._session.add_layer(doc_id, ch)     # no-op if it already exists
            self._session.set_layer_excluded(layer.layer_id, False)
            self.viewStateEdited.emit()
        elif self._session.has_layer(doc_id, ch):
            # Unticking is a statement about the CHANNEL, not about which recording is on
            # screen — and it is now RECORDED as one rather than enacted by deletion. The
            # layer keeps its colour and its Δt alignment and simply stops being drawn, and
            # `show_only` will not resurrect it when the user switches back, which is the
            # only thing deleting it ever bought. Re-ticking restores what was there.
            self._session.set_layer_excluded(f"{doc_id}#{ch}", True)
            self.viewStateEdited.emit()

    def _on_offset_changed(self, layer_id: str, value: float) -> None:
        if self._updating:
            return
        if self._session.layer(layer_id) is not None:
            self._session.set_layer_time_offset(layer_id, float(value))
            self.viewStateEdited.emit()

    def _pick_color(self, layer_id: str) -> None:
        layer = self._session.layer(layer_id)
        if layer is None:
            return
        chosen = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(layer.color), self, "Layer colour")
        if chosen.isValid():
            self._session.set_layer_color(layer_id, chosen.name())
            self.viewStateEdited.emit()

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        payload = items[0].data(_COL_NAME, _ROLE)
        if not payload:
            return
        if payload[0] == "channel":
            self.selectionChanged.emit(payload[1], payload[2])
        elif payload[0] == "doc":
            doc = self._session.document(payload[1])
            if doc.meta.channels:
                self.selectionChanged.emit(payload[1], _default_channel(doc))

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        """Right-click a recording: the compare affordance, spelled out in words.

        The checkbox is the fine-grained control (one channel), but a user who wants "show me
        both runs" should not have to work out that a tick box is how you say that. Both entries
        are ANNOUNCED to MainWindow rather than performed here — "shown" is one policy and it
        lives in one place."""
        item = self._tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(_COL_NAME, _ROLE)
        if not payload or payload[0] == "group":
            return
        doc_id = str(payload[1])
        doc = self._session.document(doc_id) if self._session.has_document(doc_id) else None
        if doc is None:
            return

        # PARENTED, THEN FREED. The parent is the dock so the popup inherits its palette,
        # font and screen -- but a parent in Qt is an OWNER, and an owner keeps what it is
        # given for as long as it lives. Measured before the ``finally`` below existed: 400
        # right-clicks on one recording row left 400 QMenu widgets and 2 000 QActions alive
        # under the File Manager, +1 menu and +5 actions per click with no plateau, because
        # right-clicking the list is the dock's own advertised way to say "show only this
        # recording" (the hint label points at it) and a long comparison session does it
        # over and over. ``deleteLater`` and not ``WA_DeleteOnClose``: the menu closes
        # BEFORE ``exec`` returns, and the QAction ``exec`` hands back is owned by the menu
        # -- deleting on close would leave ``chosen`` pointing at a freed object. The
        # ``finally`` runs on the exception path too, which is the only way the rule cannot
        # be lost the next time a branch is added between here and the return.
        menu = QtWidgets.QMenu(self)
        try:
            act_only = menu.addAction("Show only this recording")
            act_only.setToolTip("Draw this recording on its own; the others stay open.")
            act_add = menu.addAction("Add to view (compare)")
            act_add.setToolTip("Draw this recording ALONGSIDE what is already on the trace.")
            # Adding what is already the only thing shown says nothing; adding a recording
            # already drawn says nothing either.
            shown = self._session.shown_doc_ids()
            act_add.setEnabled(doc_id not in shown or len(shown) > 1)
            act_only.setEnabled(shown != frozenset({doc_id}))
            menu.addSeparator()
            act_close = menu.addAction("Close recording")

            chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
            if chosen is act_only:
                self.showOnlyRequested.emit(doc_id)
            elif chosen is act_add:
                self.addToViewRequested.emit(doc_id)
            elif chosen is act_close:
                self.closeDocumentRequested.emit(doc_id)
        finally:
            # Off the dock's child list NOW (so a second right-click one event-loop turn
            # later cannot find yesterday's menu still hanging there), and destroyed on the
            # next turn -- never inside the signal handlers above, which may still be on the
            # stack when Qt gets round to it.
            menu.setParent(None)
            menu.deleteLater()

    def _close_selected(self) -> None:
        """Announce the intent to close the selected recording. The dock does NOT call
        ``Session.close_document`` itself: closing a parent that has live segments is a
        data-loss decision, so MainWindow (the sole wirer) owns it and routes the File-menu
        twin through the same handler."""
        items = self._tree.selectedItems()
        if not items:
            return
        payload = items[0].data(_COL_NAME, _ROLE)
        if not payload or payload[0] == "group":
            return
        self.closeDocumentRequested.emit(str(payload[1]))

    # -- external API (MainWindow) ---------------------------------------------------------------
    def set_segment_enabled(self, on: bool) -> None:
        """Enable/disable the *Segment* button. MainWindow drives this from the live Measure
        selection so the button and its File-menu twin never disagree."""
        self._btn_segment.setEnabled(bool(on))

    def focus_document(self, doc_id: str) -> None:
        """Select and reveal a document (used when the user re-opens an already-open file)."""
        self._focus(("doc", doc_id))

    def focus_channel(self, doc_id: str, channel: int) -> bool:
        """Select and reveal ONE channel row, so the app selection lands on exactly that
        channel rather than the document's default.

        Used when something outside the dock knows the channel it needs — e.g. returning to a
        saved manual measurement taken on a specific channel. Returns False (selecting
        nothing) when that channel row is not in the tree."""
        return self._focus(("channel", doc_id, int(channel)))

    def _focus(self, payload: tuple) -> bool:
        """Select + scroll to the tree row carrying exactly ``payload``. Selecting emits the
        tree's own itemSelectionChanged, so ``selectionChanged`` fires as if clicked."""
        it = QtWidgets.QTreeWidgetItemIterator(self._tree)
        while it.value():
            item = it.value()
            data = item.data(_COL_NAME, _ROLE)
            # tuple() so a payload round-tripped through QVariant as a list still matches.
            if data and tuple(data) == payload:
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                return True
            it += 1
        return False
