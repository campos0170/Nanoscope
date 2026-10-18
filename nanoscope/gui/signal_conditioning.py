"""The Signal Conditioning window: the classical detectors' chain editor, in execution order.

ONE top-level window, built once by MainWindow (parented to it, never deleted). It hosts
the SAME stage-editor widgets the DetectionPanel builds and owns — reparented through
``DetectionPanel.signal_chain_editor_parts()``, never copied — so the panel's
``params()`` / ``set_params()`` / setters stay the only state. On top of them it adds:

* a stage list in EXECUTION order (notch → … → low-pass "applied inside detection"),
  rendered from ``conditioning_recipes.stage_rows``; ticking a row writes the stage's own
  enable checkbox through MainWindow, selecting a row raises that stage's form;
* the chain-versus-detection status with Revert / Run detection…;
* a toolbar: Follow main trace, Show main trace, Copy as Python, ML chain… and protocol
  Import / Export (JSON);
* the frequency-response view (Response tab).

This module wires nothing outside itself: it emits intent and MainWindow, the sole
wirer, routes it. It imports no gui sibling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.conditioning_recipes import SCOPE_SENTENCE, STAGE_KEYS, ChainStatus, stage_rows

__all__ = ["SignalConditioningWindow", "clamp_rect", "default_window_rect"]

#: The stage rows' tooltip: what the list is, and what the chain does NOT feed.
_STAGE_LIST_TIP = (
    "The signal chain in the order it runs. Tick a stage to enable it; select it to edit "
    "it below. The low-pass is applied inside detection, after every other stage.\n\n"
    + SCOPE_SENTENCE)

_FOLLOW_TIP = (
    "On: the window this workspace describes (Copy as Python's START/STOP) follows the "
    "main trace's visible range. Off: it stays on the range it last followed.")


def clamp_rect(rect: QtCore.QRect, avail: QtCore.QRect, title: int = 0) -> QtCore.QRect:
    """``rect`` (a client rect) moved/shrunk to lie inside ``avail``, leaving ``title`` px
    above it for the window frame."""
    w = min(rect.width(), avail.width())
    h = min(rect.height(), max(0, avail.height() - title))
    x = min(max(rect.x(), avail.left()), avail.right() - w + 1)
    y = min(max(rect.y(), avail.top() + title), avail.bottom() - h + 1)
    return QtCore.QRect(x, y, w, h)


def default_window_rect(main_frame: QtCore.QRect, main_client: QtCore.QRect,
                        avail: QtCore.QRect, *, width: int) -> QtCore.QRect:
    """Right-anchored to the main window's right edge, as tall as the main window.

    It covers the Inspector column and leaves the main trace to its left visible.
    ``main_frame`` / ``main_client`` give the title-bar height to reserve.

    Pixel budget with a fresh profile (the window cannot be narrower than its 760 px
    minimum): of the trace view, about 416 of 694 px stay uncovered at 1512x900, 824 of
    1102 px at 1920x1080, and only about 184 of 476 px at 1280x800. That last case is
    accepted: anchoring anywhere else covers more of the trace, and "Show main trace"
    lowers the window in one click.
    """
    title = max(0, main_client.top() - main_frame.top())
    w = min(int(width), avail.width())
    h = max(0, main_frame.height() - title)
    rect = QtCore.QRect(main_frame.right() - w + 1, main_frame.top() + title, w, h)
    return clamp_rect(rect, avail, title)


class _ElidedLine(QtWidgets.QLabel):
    """One line that elides on the right; full text in ``text()`` and the tooltip."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = str(text or "")
        self.setToolTip(self._full)
        self._elide()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = max(0, self.width() - 2)
        shown = (self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, width)
                 if width > 0 else self._full)
        if QtWidgets.QLabel.text(self) != shown:
            QtWidgets.QLabel.setText(self, shown)


class _FormHost(QtWidgets.QWidget):
    """The stage form's scroll content, which never gets SHORTER while the same stage
    stays shown at the same width.

    A form that shrank under a scrolled viewport (a detrend method with fewer knobs, a
    notice going away) clamped the scroll value and slid every control down under the
    pointer. The spare height left below is harmless; a stage switch or a new width
    starts again from the content's own height."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._stack: QtWidgets.QWidget | None = None
        self._held_for: tuple[int, int] | None = None
        self._held = 0

    def set_stage_stack(self, stack: QtWidgets.QWidget) -> None:
        self._stack = stack

    def _hold(self, width: int, natural: int) -> int:
        key = (self._stack.currentIndex() if self._stack is not None else -1, int(width))
        if key != self._held_for:
            self._held_for, self._held = key, natural
        else:
            self._held = max(self._held, natural)
        return self._held

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        hint = super().minimumSizeHint()
        return QtCore.QSize(hint.width(), self._hold(self.width(), hint.height()))

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        natural = super().heightForWidth(width)
        return natural if natural < 0 else self._hold(width, natural)


class SignalConditioningWindow(QtWidgets.QWidget):
    """The classical detectors' signal-chain workspace. See the module docstring.

    Emits (MainWindow routes every one):
        stageSelected(str): a stage row was selected — raise that stage's form.
        stageEnableRequested(str, bool): a stage row's tick changed.
        revertRequested(): put the shown detection's chain back on the form.
        runDetectionRequested(): Analysis ▸ Run detection (the prompted menu door).
        exportProtocolsRequested() / importProtocolsRequested(): protocol JSON exchange.
        copyPythonRequested(): put a script rebuilding the processed trace on the clipboard.
        mlChainRequested(): show the ML detector's own chain (ML tab).
        showMainTraceRequested(): bring the main trace in front of this window.
        followMainChanged(bool): "Follow main trace" toggled.
        saveProjectRequested() / saveProjectAsRequested(): ⌘S / ⌘⇧S while this window is
            the active one — MainWindow runs its own project save.
    """

    stageSelected = QtCore.Signal(str)
    stageEnableRequested = QtCore.Signal(str, bool)
    revertRequested = QtCore.Signal()
    runDetectionRequested = QtCore.Signal()
    exportProtocolsRequested = QtCore.Signal()
    importProtocolsRequested = QtCore.Signal()
    copyPythonRequested = QtCore.Signal()
    mlChainRequested = QtCore.Signal()
    showMainTraceRequested = QtCore.Signal()
    followMainChanged = QtCore.Signal(bool)
    saveProjectRequested = QtCore.Signal()
    saveProjectAsRequested = QtCore.Signal()

    MIN_WIDTH = 760
    MIN_HEIGHT = 560
    DEFAULT_WIDTH = 760
    LEFT_WIDTH = 350
    LEFT_MIN_WIDTH = 300
    STATUS_ROW_HEIGHT = 26

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setObjectName("SignalConditioningWindow")
        self.setWindowTitle("Signal Conditioning")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self._installed = False
        self._rendering = False
        self._params: dict[str, Any] = {}
        self._fs: float | None = None
        self._lowpass_design_hz: float | None = None
        self._rows: tuple = ()
        self._window_range: tuple[int, int] | None = None
        self._status = ChainStatus.none()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # -- toolbar: overflows into its extension menu, never floors the width -------------
        self.toolbar = QtWidgets.QToolBar("Signal conditioning")
        self.toolbar.setObjectName("signalConditioningToolbar")
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.toolbar.setMovable(False)
        self.toolbar.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        # The stateful toggle comes FIRST: at the 760 px default width the last actions
        # overflow into the » menu, and the one that decides what Copy as Python slices
        # must stay on screen.
        self.act_follow = self.toolbar.addAction("Follow main trace")
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(True)
        self.act_follow.setToolTip(_FOLLOW_TIP)
        self.act_follow.toggled.connect(self.followMainChanged.emit)
        self.act_show_main = self.toolbar.addAction("Show main trace")
        self.act_show_main.setToolTip("Bring the main window's trace in front of this window.")
        self.act_show_main.triggered.connect(lambda *_: self.showMainTraceRequested.emit())
        self.toolbar.addSeparator()
        self.act_copy_python = self.toolbar.addAction("Copy as Python")
        self.act_copy_python.setToolTip(
            "Copy a script that reads this recording, applies this calibration and this "
            "chain with nanoscope.analysis.conditioning.conditioned_trace, and slices the "
            "window — the same samples NanoScope shows.")
        self.act_copy_python.triggered.connect(lambda *_: self.copyPythonRequested.emit())
        self.act_ml_chain = self.toolbar.addAction("ML chain…")
        self.act_ml_chain.setToolTip(
            "The ML detector does not use this chain: it runs its own model preprocessing. "
            "Show it (Inspector ▸ Analysis ▸ ML).")
        self.act_ml_chain.triggered.connect(lambda *_: self.mlChainRequested.emit())
        self.toolbar.addSeparator()
        self.act_import = self.toolbar.addAction("Import protocols…")
        self.act_import.setToolTip(
            "Add protocols from a NanoScope conditioning-protocol file (.json). Nothing "
            "is overwritten: a name already in use is imported as “name (imported)”.")
        self.act_import.triggered.connect(lambda *_: self.importProtocolsRequested.emit())
        self.act_export = self.toolbar.addAction("Export protocols…")
        self.act_export.setToolTip(
            "Write your saved protocols (every stage of each) to a .json file you can "
            "import on another computer.")
        self.act_export.triggered.connect(lambda *_: self.exportProtocolsRequested.emit())
        root.addWidget(self.toolbar)

        # ⌘S / ⌘⇧S: the main window's File actions do not reach a separate top-level
        # window, so this one forwards them (as the Event Gallery and Inspector do).
        self.act_save_project = QtGui.QAction("Save Project", self)
        self.act_save_project.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.act_save_project.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.act_save_project.triggered.connect(lambda *_: self.saveProjectRequested.emit())
        self.addAction(self.act_save_project)
        self.act_save_project_as = QtGui.QAction("Save Project As…", self)
        self.act_save_project_as.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.act_save_project_as.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.act_save_project_as.triggered.connect(
            lambda *_: self.saveProjectAsRequested.emit())
        self.addAction(self.act_save_project_as)

        self.scope_line = _ElidedLine(SCOPE_SENTENCE)
        self.scope_line.setObjectName("signalConditioningScope")
        self.scope_line.setStyleSheet("color:#8b9199;")
        root.addWidget(self.scope_line)

        # -- status row: fixed height, both buttons always present (enabled/disabled only) --
        status_row = QtWidgets.QWidget()
        status_row.setObjectName("signalConditioningStatusRow")
        status_row.setFixedHeight(self.STATUS_ROW_HEIGHT)
        srl = QtWidgets.QHBoxLayout(status_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(6)
        self.status_line = _ElidedLine("")
        self.status_line.setObjectName("signalConditioningStatus")
        srl.addWidget(self.status_line, 1)
        self.revert_button = QtWidgets.QPushButton("Revert to detection's chain")
        self.revert_button.setObjectName("signalConditioningRevert")
        self.revert_button.setToolTip(
            "Put the chain the detection on screen ran with back on the form.")
        self.revert_button.setEnabled(False)
        # Neither status button may be where focus lands on open, nor keep it after a
        # click: a stray Space would revert the form or start a run.
        self.revert_button.setFocusPolicy(QtCore.Qt.TabFocus)
        self.revert_button.clicked.connect(lambda *_: self.revertRequested.emit())
        srl.addWidget(self.revert_button)
        self.run_button = QtWidgets.QPushButton("Run detection…")
        self.run_button.setObjectName("signalConditioningRun")
        self.run_button.setToolTip(
            "Analysis ▸ Run detection with the form's chain (asks first when the table "
            "on screen carries hand curation).")
        self.run_button.setEnabled(False)
        self.run_button.setFocusPolicy(QtCore.Qt.TabFocus)
        self.run_button.clicked.connect(lambda *_: self.runDetectionRequested.emit())
        srl.addWidget(self.run_button)
        root.addWidget(status_row)
        self._status_row = status_row

        # -- splitter: editor column | preview tabs ---------------------------------------
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("signalConditioningSplitter")
        self.splitter.setChildrenCollapsible(False)

        left = QtWidgets.QWidget()
        left.setObjectName("signalConditioningEditor")
        left.setMinimumWidth(self.LEFT_MIN_WIDTH)
        self._left_layout = QtWidgets.QVBoxLayout(left)
        self._left_layout.setContentsMargins(0, 0, 4, 0)
        self._left_layout.setSpacing(4)
        self._protocol_slot = QtWidgets.QVBoxLayout()
        self._protocol_slot.setContentsMargins(0, 0, 0, 0)
        self._left_layout.addLayout(self._protocol_slot)
        caption = QtWidgets.QLabel("Stages, in execution order")
        caption.setStyleSheet("font-weight:600;")
        self._left_layout.addWidget(caption)

        self.stage_list = QtWidgets.QListWidget()
        self.stage_list.setObjectName("signalConditioningStages")
        self.stage_list.setToolTip(_STAGE_LIST_TIP)
        self.stage_list.setUniformItemSizes(True)
        self.stage_list.setWordWrap(False)
        self.stage_list.setTextElideMode(QtCore.Qt.ElideRight)
        self.stage_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.stage_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.stage_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.stage_list.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        for key in STAGE_KEYS:
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                          | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, key)
            self.stage_list.addItem(item)
        self._fit_stage_list()
        self.stage_list.currentRowChanged.connect(self._on_row_selected)
        self.stage_list.itemChanged.connect(self._on_item_changed)
        self._left_layout.addWidget(self.stage_list)

        self.form_area = QtWidgets.QScrollArea()
        self.form_area.setObjectName("signalConditioningForm")
        self.form_area.setWidgetResizable(True)
        self.form_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Always on: a scrollbar that came and went with the page's height would narrow
        # every control of the form by its width.
        self.form_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.form_area.setFocusPolicy(QtCore.Qt.NoFocus)   # not a Tab stop of its own
        self.form_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._form_host = _FormHost()
        self._form_layout = QtWidgets.QVBoxLayout(self._form_host)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(6)
        self._form_layout.addStretch(1)
        self.form_area.setWidget(self._form_host)
        self._left_layout.addWidget(self.form_area, 1)

        self.preview_tabs = QtWidgets.QTabWidget()
        self.preview_tabs.setObjectName("signalConditioningPreview")
        self._response_page = QtWidgets.QWidget()
        self._response_layout = QtWidgets.QVBoxLayout(self._response_page)
        self._response_layout.setContentsMargins(4, 4, 4, 4)
        self.preview_tabs.addTab(self._response_page, "Response")
        self.preview_tabs.setTabToolTip(
            0, "The applied |H(f)| of the selected stage, or of every enabled linear stage.")

        self.splitter.addWidget(left)
        self.splitter.addWidget(self.preview_tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self.LEFT_WIDTH, max(1, self.DEFAULT_WIDTH - self.LEFT_WIDTH)])
        root.addWidget(self.splitter, 1)

        # Edits arrive ~40 per set_params; the rows are rendered once per burst.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render_rows)
        self.set_chain_status(None)

    # -- installation -----------------------------------------------------------------------
    def install_editor(self, parts) -> None:
        """Reparent the DetectionPanel's editor parts into this window (once)."""
        if self._installed:
            return
        self._installed = True
        self._protocol_slot.addWidget(parts.protocol_bar)
        self._form_layout.insertWidget(0, parts.stage_stack)
        self._form_layout.insertWidget(1, parts.notices)
        self._form_host.set_stage_stack(parts.stage_stack)
        self._response_layout.addWidget(parts.response_view, 1)
        self._parts = parts
        self._order_tab_chain()

    def _order_tab_chain(self) -> None:
        """Tab from the stage list goes straight into the shown stage's form, then the
        response view, the protocol bar and the status buttons.

        Reparented widgets join the focus chain AFTER the window's own, which put the
        tab bar and the protocol bar between the list and the form."""
        chain: list[QtWidgets.QWidget] = []
        w = self.nextInFocusChain()
        while w is not None and w is not self and len(chain) < 20_000:
            chain.append(w)
            w = w.nextInFocusChain()

        def stops(container: QtWidgets.QWidget) -> list[QtWidgets.QWidget]:
            return [x for x in chain if container.isAncestorOf(x) and x.focusProxy() is None
                    and x.focusPolicy() & QtCore.Qt.TabFocus]

        parts = self._parts
        # The tab widget goes in itself (it proxies focus to its bar): left behind, it
        # would be the stop Tab wraps to instead of the stage list.
        order = ([self.stage_list] + stops(parts.stage_stack) + stops(parts.notices)
                 + [self.preview_tabs] + stops(parts.response_view)
                 + stops(parts.protocol_bar) + [self.revert_button, self.run_button])
        for first, second in zip(order, order[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

    def focus_stage_list(self) -> None:
        """Put keyboard focus on the stage list (where every open lands)."""
        self.stage_list.setFocus(QtCore.Qt.ActiveWindowFocusReason)

    def is_installed(self) -> bool:
        return self._installed

    # -- pushes from MainWindow --------------------------------------------------------------
    def set_chain(self, params: Mapping[str, Any], fs: float | None,
                  lowpass_design_hz: float | None = None) -> None:
        """The form's current chain (coalesced: rendered once per event-loop pass)."""
        self._params = dict(params or {})
        self._fs = fs
        self._lowpass_design_hz = lowpass_design_hz
        self._render_timer.start()

    def flush(self) -> None:
        """Render a pending :meth:`set_chain` now."""
        if self._render_timer.isActive():
            self._render_timer.stop()
            self._render_rows()

    def _render_rows(self) -> None:
        rows = stage_rows(self._params, self._fs, lowpass_design_hz=self._lowpass_design_hz)
        self._rows = rows
        self._rendering = True
        was = self.stage_list.blockSignals(True)
        try:
            for i, row in enumerate(rows):
                item = self.stage_list.item(i)
                if item.text() != row.text:
                    item.setText(row.text)
                state = QtCore.Qt.Checked if row.enabled else QtCore.Qt.Unchecked
                if item.checkState() != state:
                    item.setCheckState(state)
                tip = (f"{row.title}: {row.summary}"
                       + (f" ({row.detail})" if row.detail else "")
                       + (f" — {row.role}" if row.role else ""))
                if not row.linear:
                    tip += "\nNot a linear filter: it has no fixed frequency response."
                if item.toolTip() != tip:
                    item.setToolTip(tip)
        finally:
            self.stage_list.blockSignals(was)
            self._rendering = False
        self.stage_list.viewport().update()

    def stage_rows(self) -> tuple:
        """The rows last rendered (StageRow tuples), after any pending render."""
        self.flush()
        return self._rows

    def show_stage(self, key: str) -> None:
        """Select ``key``'s row without echoing a selection back."""
        if key not in STAGE_KEYS:
            return
        row = STAGE_KEYS.index(key)
        if self.stage_list.currentRow() == row:
            return
        was = self.stage_list.blockSignals(True)
        try:
            self.stage_list.setCurrentRow(row)
        finally:
            self.stage_list.blockSignals(was)

    def set_chain_status(self, status: ChainStatus | None) -> None:
        """The chain-versus-detection verdict on the status row."""
        status = status or ChainStatus.none()
        self._status = status
        text = status.headline or (
            "No Schmitt/CUSUM detection is shown, so the chain has nothing to be compared "
            "with." if status.kind == "none" else "")
        self.status_line.setText(text)
        # One line at the default width: the differences lead, the sentence is the tip.
        self.status_line.setToolTip(status.sentence or text)
        warn = status.kind == "differs"
        style = "color:#FFB74D; font-weight:600;" if warn else "color:#8b9199;"
        if self.status_line.styleSheet() != style:
            self.status_line.setStyleSheet(style)
        self.revert_button.setEnabled(warn and bool(status.revert_params))

    def chain_status(self) -> ChainStatus:
        return self._status

    def set_run_enabled(self, enabled: bool) -> None:
        self.run_button.setEnabled(bool(enabled))

    def set_source_label(self, label: str) -> None:
        self.setWindowTitle(f"Signal Conditioning — {label}" if label else "Signal Conditioning")

    def set_view_window(self, start: int, stop: int) -> None:
        """The main trace's visible sample range; kept only while following."""
        if self.follows_main():
            self._window_range = (int(start), int(stop))

    def clear_view_window(self) -> None:
        self._window_range = None

    def current_window(self) -> tuple[int, int] | None:
        return self._window_range

    def follows_main(self) -> bool:
        return self.act_follow.isChecked()

    def set_follow_main(self, on: bool) -> None:
        was = self.act_follow.blockSignals(True)
        try:
            self.act_follow.setChecked(bool(on))
        finally:
            self.act_follow.blockSignals(was)

    def splitter_state(self) -> QtCore.QByteArray:
        return self.splitter.saveState()

    def restore_splitter_state(self, state) -> bool:
        return bool(self.splitter.restoreState(QtCore.QByteArray(state)))

    def abort(self) -> None:
        """Stop pending work (teardown protocol). The window runs no worker of its own."""
        self._render_timer.stop()

    # -- stage list --------------------------------------------------------------------------
    def _fit_stage_list(self) -> None:
        rows = self.stage_list.count()
        row_h = max(1, self.stage_list.sizeHintForRow(0))
        frame = 2 * self.stage_list.frameWidth()
        self.stage_list.setFixedHeight(rows * row_h + frame + 2)

    def _row_key(self, row: int) -> str:
        return STAGE_KEYS[row] if 0 <= row < len(STAGE_KEYS) else ""

    def _on_row_selected(self, row: int) -> None:
        key = self._row_key(row)
        if key and not self._rendering:
            self.stageSelected.emit(key)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._rendering:
            return
        key = str(item.data(QtCore.Qt.UserRole) or "")
        if not key:
            return
        on = item.checkState() == QtCore.Qt.Checked
        rendered = next((r.enabled for r in self._rows if r.key == key), None)
        if rendered is None or rendered != on:
            self.stageEnableRequested.emit(key, on)

    # -- events ------------------------------------------------------------------------------
    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self._fit_stage_list()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._fit_stage_list()
        self.flush()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Closing HIDES the window (it is built once and kept); nothing is deleted."""
        self.abort()
        super().closeEvent(event)
