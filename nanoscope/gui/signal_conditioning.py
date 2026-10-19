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
* previews of the VISIBLE window — Time (raw against processed), Spectrum (before/after,
  with a line finder that sends lines to the notch) and Response (magnitude, phase, group
  delay, step) — computed by ONE long-lived worker thread, debounced, coalesced to the
  latest request and cancellable (``analysis.conditioning_preview``).

This module wires nothing outside itself: it emits intent and MainWindow, the sole
wirer, routes it. Of gui/ it reaches only the worker-thread seam, the readable axes and
the trace view's curve factory.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis import filters
from ..analysis.cancel import OperationCancelled, cancel_scope
from ..analysis.conditioning import conditioning_enabled, conditioning_params
from ..analysis.conditioning_preview import (
    MAX_PREVIEW_WINDOW_SAMPLES,
    ConditionedTraceMemo,
    ConditioningSource,
    PreviewRequest,
    PreviewResult,
    clamp_window,
    compute_exact,
    compute_preview,
    estimate_stage_titles,
    line_is_placeable,
    memo_key,
    notch_covers,
    pad_limited_stages,
    preview_pad_samples,
    units_after_chain,
    window_for_resolution_s,
    window_is_exact,
)
from ..analysis.conditioning_recipes import (
    SCOPE_SENTENCE,
    STAGE_ENABLE_KEY,
    STAGE_KEYS,
    STAGE_TITLES,
    ChainStatus,
    chain_signature,
    stage_rows,
)
from ..analysis.filter_views import (
    MAX_STEP_SIDE_SAMPLES,
    linear_stage_specs,
    stage_title,
    step_span_s,
)
from ..analysis.modules import notch_freqs_param
from ..core import clock
from .axes import ReadableLinearAxis, ReadableLogAxis
from .trace_view import ChannelPane
from .worker_thread import DEFAULT_JOIN_MS, JOB_NONE, retire_worker_thread, start_worker_thread

__all__ = ["SignalConditioningWindow", "clamp_rect", "default_window_rect"]

#: The stage rows' tooltip: what the list is, and what the chain does NOT feed.
_STAGE_LIST_TIP = (
    "The signal chain in the order it runs. Tick a stage to enable it; select it to edit "
    "it below. The low-pass is applied inside detection, after every other stage.\n\n"
    + SCOPE_SENTENCE)

_FOLLOW_TIP = (
    "On: the previews (and Copy as Python's START/STOP) follow the main trace's visible "
    "range. Panning or zooming a preview turns this off and previews that range instead.")

_EXACT_TIP = (
    "Condition the WHOLE recording off the GUI thread (seconds on a long file) and preview "
    "its exact samples. Baseline drift and Normalize use whole-recording statistics, and a "
    "very slow high-pass or notch settles longer than the window's padding, so a window of "
    "them is only an estimate. Runs only when pressed; an edit of the chain cancels it.")

#: The exact button's three states (it is as wide as the widest, so it never reflows).
_EXACT_TEXTS = ("Compute exact", "Computing…", "Retry exact")

#: A step response probe up to this many samples is computed on the GUI thread (a few
#: ms; a 1 Hz low-pass's is ~800k samples and goes to the worker).
_GUI_STEP_SAMPLES = 400_000

_RAW_INK = "#8b9199"
_PROCESSED_INK = "#56B4E9"
_DIFF_INK = "#CC79A7"
_LINE_INK = "#F0E442"
_LEGEND_TEXT = "#d7dbe0"
_LEGEND_ROW_PX = 22
_SPLIT_TICK_TEXT_PX = 56

#: Tab keys, in tab order (also the persisted values).
TAB_KEYS = ("time", "spectrum", "response")
TIME_MODES = ("overlay", "split", "difference")
RESPONSE_VIEWS = ("magnitude", "phase", "group_delay", "step")

_MAX_LINE_ROWS = 24
_DEBOUNCE_MS = 250
_OUTCOME_POLL_MS = 15


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
    """``lines`` lines (word-wrapped) that elide at the end of the last one, at a fixed
    height; full text in ``text()`` and the tooltip."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None, *,
                 lines: int = 1) -> None:
        super().__init__(parent)
        self._full = ""
        self._lines = max(1, int(lines))
        self.setTextFormat(QtCore.Qt.PlainText)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        if self._lines > 1:
            self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.setFixedHeight(self._lines * self.fontMetrics().lineSpacing() + 2)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = str(text or "")
        if self.toolTip() != self._full:
            self.setToolTip(self._full)
        self._elide()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._elide()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().changeEvent(event)
        if self._lines > 1 and event.type() == QtCore.QEvent.FontChange:
            self.setFixedHeight(self._lines * self.fontMetrics().lineSpacing() + 2)
            self._elide()

    def _elide(self) -> None:
        width = max(0, self.width() - 2)
        if width <= 0:
            shown = self._full
        elif self._lines == 1:
            shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, width)
        else:
            shown = _elide_lines(self.font(), self._full, width, self._lines)
        if QtWidgets.QLabel.text(self) != shown:
            QtWidgets.QLabel.setText(self, shown)


def _elide_lines(font: QtGui.QFont, text: str, width: int, lines: int) -> str:
    """``text`` word-wrapped to ``width`` px, the last of ``lines`` lines elided."""
    fm = QtGui.QFontMetrics(font)
    layout = QtGui.QTextLayout(text, font)
    out: list[str] = []
    layout.beginLayout()
    try:
        for i in range(lines):
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            start = line.textStart()
            if i == lines - 1:
                out.append(fm.elidedText(text[start:].strip(), QtCore.Qt.ElideRight, width))
            else:
                out.append(text[start:start + line.textLength()].rstrip())
    finally:
        layout.endLayout()
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# the one long-lived preview worker
# ---------------------------------------------------------------------------------------------
class _PreviewOutcome:
    """A terminal answer for one request: ``status`` is 'ok', 'cancelled' or 'error'."""

    __slots__ = ("token", "kind", "status", "result", "message")

    def __init__(self, token: int, kind: str, status: str,
                 result: PreviewResult | None = None, message: str = "") -> None:
        self.token, self.kind, self.status = token, kind, status
        self.result, self.message = result, message


class _PreviewMailbox:
    """The latest request of each kind, shared by the GUI thread and the worker.

    Posting REPLACES a waiting request of the same kind (coalescing) and sets the shared
    cancel event when the request now running is superseded by it. Both happen under one
    lock with the worker's take, so a cancel can never land on the request after it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting: dict[str, PreviewRequest] = {}
        self._running: PreviewRequest | None = None
        self._outbox: list[_PreviewOutcome] = []
        self.cancel = threading.Event()
        self._stopped = False

    @staticmethod
    def _supersedes(new: PreviewRequest, running: PreviewRequest) -> bool:
        if new.kind == running.kind:
            return True
        if running.kind == "exact":
            # A window request for the same chain does not need the exact pass stopped.
            return (new.signature, new.source_key) != (running.signature, running.source_key)
        return False

    def post(self, req: PreviewRequest) -> None:
        with self._lock:
            if self._stopped:
                return
            self._waiting[req.kind] = req
            if self._running is not None and self._supersedes(req, self._running):
                self.cancel.set()

    def withdraw(self, kind: str | None = None) -> None:
        """Drop a waiting request of ``kind`` (None: every kind) and cancel a running one."""
        with self._lock:
            if kind is None:
                self._waiting.clear()
            else:
                self._waiting.pop(kind, None)
            if self._running is not None and kind in (None, self._running.kind):
                self.cancel.set()

    def take(self) -> PreviewRequest | None:
        with self._lock:
            if self._stopped:
                return None
            for kind in ("window", "exact"):
                req = self._waiting.pop(kind, None)
                if req is not None:
                    self._running = req
                    self.cancel.clear()
                    return req
            return None

    def finish(self, outcome: _PreviewOutcome) -> None:
        """The running request ended: its answer joins the outbox in the same step, so
        the GUI never sees an idle mailbox with an answer still to come."""
        with self._lock:
            self._running = None
            self._outbox.append(outcome)

    def collect(self) -> list[_PreviewOutcome]:
        with self._lock:
            out, self._outbox = self._outbox, []
            return out

    def idle(self) -> bool:
        with self._lock:
            return self._running is None and not self._waiting and not self._outbox

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._waiting.clear()
            self._outbox = []
            self.cancel.set()


class _PreviewWorker(QtCore.QObject):
    """Lives on the window's ONE worker thread, from the first preview until MainWindow's
    teardown (closing the window only cancels what it was doing).

    Requests arrive through the mailbox; ``drain`` is invoked by a queued signal and
    computes until nothing is waiting. Answers go back through the mailbox's outbox, which
    the window polls on the GUI thread while work is outstanding, so this thread runs no
    Qt code of its own. No ``finished``/``failed``/``cancelled`` signals: those would quit
    the thread after the first request (``start_worker_thread``)."""

    def __init__(self, mailbox: _PreviewMailbox) -> None:
        super().__init__()
        self._mailbox = mailbox

    @QtCore.Slot()
    def run(self) -> None:
        self.drain()

    @QtCore.Slot()
    def drain(self) -> None:
        while True:
            req = self._mailbox.take()
            if req is None:
                return
            outcome = _PreviewOutcome(req.token, req.kind, "error", message="the preview stopped")
            try:
                with cancel_scope(self._mailbox.cancel.is_set):
                    result = compute_exact(req) if req.kind == "exact" else compute_preview(req)
                outcome = _PreviewOutcome(req.token, req.kind, "ok", result)
            except OperationCancelled:
                outcome = _PreviewOutcome(req.token, req.kind, "cancelled")
            except Exception as exc:  # noqa: BLE001 - a preview must never kill its thread
                outcome = _PreviewOutcome(req.token, req.kind, "error", message=str(exc))
            finally:
                self._mailbox.finish(outcome)


def _segmented(labels: Sequence[tuple[str, str]], tips: Mapping[str, str] | None = None
               ) -> tuple[QtWidgets.QWidget, QtWidgets.QButtonGroup,
                          dict[str, QtWidgets.QToolButton]]:
    """A row of exclusive, checkable text buttons (no menus, no popups)."""
    host = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    group = QtWidgets.QButtonGroup(host)
    group.setExclusive(True)
    buttons: dict[str, QtWidgets.QToolButton] = {}
    for key, text in labels:
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setCheckable(True)
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        b.setFocusPolicy(QtCore.Qt.TabFocus)
        if tips and key in tips:
            b.setToolTip(tips[key])
        group.addButton(b)
        row.addWidget(b)
        buttons[key] = b
    host.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    return host, group, buttons


def _legend(item: pg.PlotItem) -> pg.LegendItem:
    """A one-row legend right-aligned in the plot's own title row: inside the data area it
    hid the baseline band of a trace (and the plateau of a step) under its box."""
    item.setTitle(" ")
    item.layout.setRowFixedHeight(0, _LEGEND_ROW_PX)
    item.titleLabel.setMaximumHeight(_LEGEND_ROW_PX)
    legend = pg.LegendItem(offset=None, colCount=2, horSpacing=4, pen=None, brush=None,
                           labelTextColor=_LEGEND_TEXT)
    legend.setParentItem(item)
    legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-4, 0))
    legend.layout.setContentsMargins(0, 0, 0, 0)
    return legend


def _set_legend_text(legend: pg.LegendItem, index: int, text: str) -> None:
    label = legend.items[index][1]
    if label.text != text:
        label.setText(text)
        # A longer label otherwise squeezes the sample column to nothing.
        legend.updateSize()


def _fit_axis_label(vb: pg.ViewBox, axis: pg.AxisItem, side: str, full: str,
                    shorter: Sequence[str]) -> None:
    # A label longer than its axis is centred past both ends and clipped there. The axis
    # spans the view box's side; the view box is the one already resized when this runs.
    # The first of ``full`` and ``shorter`` that fits is shown; the last is elided.
    rect = vb.geometry()
    length = (rect.width() if side in ("bottom", "top") else rect.height()) - 6

    def shows(text: str) -> bool:
        if axis.labelText != text:
            axis.setLabel(text)
        return length <= 0 or axis.label.boundingRect().width() <= length

    candidates = [full] + [s for s in shorter if s and s != full]
    if not any(shows(text) for text in candidates):
        last = candidates[-1]
        fm = QtGui.QFontMetrics(axis.label.font())
        drawn = max(1.0, axis.label.boundingRect().width())
        scale = drawn / max(1, fm.horizontalAdvance(last))
        shows(fm.elidedText(last, QtCore.Qt.ElideRight, max(8, int(length / scale))))
    if axis.toolTip() != full:
        axis.setToolTip(full)


def _duration_text(seconds: float | None) -> str:
    """'51.0 µs', '3.48 ms', '1.2 s' (or '—')."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds:.3g} s"


def _seconds_text(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds >= 10 else f"{seconds:.2g} s"


def _settles_text(title: str, seconds: float, pad_s: float) -> str:
    """'High-pass settles over 400 s, longer than the ±40 s padding' — or, when the two
    print alike (2,000,001 against 2,000,000 samples), '… settles over more than …'."""
    own, pad = _seconds_text(seconds), _seconds_text(pad_s)
    if own == pad:
        return f"{title} settles over more than the ±{pad} padding"
    return f"{title} settles over {own}, longer than the ±{pad} padding"


class _LinesList(QtWidgets.QListWidget):
    """The spectrum's lines. Return / Enter sends the current line: on macOS the item
    views route those keys to editing, never to ``activated``."""

    returnPressed = QtCore.Signal()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt override)
        if (event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)
                and event.modifiers() in (QtCore.Qt.NoModifier, QtCore.Qt.KeypadModifier)):
            event.accept()
            self.returnPressed.emit()
            return
        super().keyPressEvent(event)


def _scaled_seconds(values: np.ndarray, span_s: float) -> tuple[np.ndarray, str]:
    """``values`` in µs or ms (whichever reads better for ``span_s``), and the unit."""
    if span_s < 2e-3:
        return np.asarray(values) * 1e6, "µs"
    if span_s < 2.0:
        return np.asarray(values) * 1e3, "ms"
    return np.asarray(values), "s"


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
        followMainChanged(bool): "Follow main trace" toggled (also by panning a preview).
        notchLinesRequested(object): spectrum lines (a list of Hz) to add to the notch.
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
    notchLinesRequested = QtCore.Signal(object)
    saveProjectRequested = QtCore.Signal()
    saveProjectAsRequested = QtCore.Signal()
    #: GUI thread -> the worker (queued): a request is waiting in the mailbox.
    _previewPosted = QtCore.Signal()

    MIN_WIDTH = 760
    MIN_HEIGHT = 560
    DEFAULT_WIDTH = 760
    LEFT_WIDTH = 350
    LEFT_MIN_WIDTH = 300
    STATUS_ROW_HEIGHT = 26
    DEBOUNCE_MS = _DEBOUNCE_MS

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setObjectName("SignalConditioningWindow")
        self.setWindowTitle("Signal Conditioning")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self._installed = False
        self._rendering = False
        self._params: dict[str, Any] = {}
        self._chain: dict[str, Any] = {}
        self._fs: float | None = None
        self._lowpass_design_hz: float | None = None
        self._rows: tuple = ()
        self._window_range: tuple[int, int] | None = None
        self._status = ChainStatus.none()
        self._stage_key = ""
        # Preview state.
        self._source: ConditioningSource | None = None
        self._memo = ConditionedTraceMemo(capacity=2)
        self._exact_elapsed: dict[tuple, float] = {}
        self._mailbox: _PreviewMailbox | None = None
        self._worker: _PreviewWorker | None = None
        self._thread: QtCore.QThread | None = None
        # The thread is started on the first preview and stopped only by abort()
        # (MainWindow's teardown); a closed window keeps it, idle.
        self._worker_retired = False
        self._token = 0
        self._window_token = 0          # the latest window request posted
        self._window_answered = 0       # the latest window request answered
        self._response_posted = False   # a slow step response is on the worker
        self._exact_token: int | None = None
        self._exact_key: tuple | None = None
        #: (memo key, message) of the last failed exact pass: shown only for that chain.
        self._exact_error: tuple[tuple | None, str] = (None, "")
        self._last: PreviewResult | None = None
        self._last_response: PreviewResult | None = None
        self._last_error = ""
        self._dirty = True
        self._time_choice = "overlay"
        self._line_rows: list[Any] = []
        self._axis_labels: dict[int, tuple[pg.ViewBox, pg.AxisItem, str, str, str]] = {}

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
        # overflow into the » menu, and the one that decides what is previewed must stay
        # on screen.
        self.act_follow = self.toolbar.addAction("Follow main trace")
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(True)
        self.act_follow.setToolTip(_FOLLOW_TIP)
        self.act_follow.toggled.connect(self._on_follow_toggled)
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
        self._build_time_tab()
        self._build_spectrum_tab()
        self._build_response_tab()
        self.preview_tabs.currentChanged.connect(lambda *_: self._on_view_changed())

        self.splitter.addWidget(left)
        self.splitter.addWidget(self.preview_tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self.LEFT_WIDTH, max(1, self.DEFAULT_WIDTH - self.LEFT_WIDTH)])
        root.addWidget(self.splitter, 1)

        # Answers are collected HERE, on the GUI thread, while work is outstanding: the
        # worker emitting a signal itself would run PySide's signal machinery on its
        # thread, and a wrapped or patched handler wired to it would draw off this one.
        self._outcome_timer = QtCore.QTimer(self)
        self._outcome_timer.setInterval(_OUTCOME_POLL_MS)
        self._outcome_timer.timeout.connect(self._collect_outcomes)
        # Edits arrive ~40 per set_params; the rows are rendered once per burst.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render_pending)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self._post_preview)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(0)
        self._status_timer.timeout.connect(self._refresh_status_texts)
        for i, seq in enumerate(("Ctrl+1", "Ctrl+2", "Ctrl+3")):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda i=i: self.preview_tabs.setCurrentIndex(i))
        self.set_chain_status(None)
        self._refresh_time_controls()
        self._refresh_status_texts()

    # -- the Time tab ---------------------------------------------------------------------
    def _plot(self, *, log_x: bool = False, log_y: bool = False) -> pg.PlotWidget:
        axis = ReadableLogAxis if (log_x or log_y) else ReadableLinearAxis
        w = pg.PlotWidget(axisItems={"bottom": axis(orientation="bottom"),
                                     "left": axis(orientation="left")})
        w.setMenuEnabled(False)
        w.hideButtons()
        w.showGrid(x=True, y=True, alpha=0.15)
        w.setMinimumSize(120, 80)
        if log_x or log_y:
            w.setLogMode(x=log_x, y=log_y)
        return w

    def _time_plot_item(self, item: pg.PlotItem) -> None:
        vb = item.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        vb.setAutoVisible(y=True)
        # Only a mouse drag or wheel turns Follow off: a programmatic range change, a
        # resize or the linked partner re-emitting never reaches this signal.
        vb.sigRangeChangedManually.connect(lambda mask, vb=vb: self._on_user_range(vb, mask))
        vb.sigResized.connect(lambda *_: self._refit_axis_labels())

    def _build_time_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        controls = QtWidgets.QWidget()
        controls.setFixedHeight(self.STATUS_ROW_HEIGHT)
        cl = QtWidgets.QHBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        host, self._time_group, self.time_mode_buttons = _segmented(
            (("overlay", "Overlay"), ("split", "Split"), ("difference", "Raw − processed")))
        self.time_mode_buttons["overlay"].setChecked(True)
        self._time_group.buttonClicked.connect(self._on_time_mode_clicked)
        cl.addWidget(host)
        cl.addStretch(1)
        lay.addWidget(controls)
        # The status gets fixed lines that lead with what is happening (computing, failed,
        # updating) and then the verdict, so neither is elided away and a longer sentence
        # never pushes the plot down. It shares its row with the exact button: on the mode
        # row the two would floor the pane's width and squeeze the stage form beside it.
        status_row = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(status_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        self.time_status = _ElidedLine("", lines=3)
        self.time_status.setObjectName("signalConditioningTimeStatus")
        self.time_status.setStyleSheet("color:#8b9199;")
        sl.addWidget(self.time_status, 1)
        self.exact_button = QtWidgets.QPushButton(_EXACT_TEXTS[0])
        self.exact_button.setObjectName("signalConditioningComputeExact")
        self.exact_button.setFocusPolicy(QtCore.Qt.TabFocus)
        self.exact_button.setToolTip(_EXACT_TIP)
        self.exact_button.clicked.connect(lambda *_: self.compute_exact())
        sl.addWidget(self.exact_button, 0, QtCore.Qt.AlignTop)
        self._fit_exact_button()
        status_row.setFixedHeight(max(self.time_status.height(),
                                      self.exact_button.sizeHint().height()))
        self._time_status_row = status_row
        lay.addWidget(status_row)

        self.time_stack = QtWidgets.QStackedWidget()
        self.overlay_plot = self._plot()
        oi = self.overlay_plot.getPlotItem()
        self._time_plot_item(oi)
        self.overlay_legend = _legend(oi)
        self.raw_curve = ChannelPane.make_curve(oi, _RAW_INK)
        self.raw_curve.setPen(pg.mkPen(QtGui.QColor(139, 145, 153, 200), width=1))
        self.processed_curve = ChannelPane.make_curve(oi, _PROCESSED_INK)
        self.overlay_legend.addItem(self.raw_curve, "raw (calibrated)")
        self.overlay_legend.addItem(self.processed_curve, "processed — window estimate")
        self.time_stack.addWidget(self.overlay_plot)

        self.split_view = pg.GraphicsLayoutWidget()
        self.split_view.setMinimumSize(120, 120)
        self.split_raw_plot = self.split_view.addPlot(
            row=0, col=0, axisItems={"bottom": ReadableLinearAxis(orientation="bottom"),
                                     "left": ReadableLinearAxis(orientation="left")})
        self.split_processed_plot = self.split_view.addPlot(
            row=1, col=0, axisItems={"bottom": ReadableLinearAxis(orientation="bottom"),
                                     "left": ReadableLinearAxis(orientation="left")})
        for item in (self.split_raw_plot, self.split_processed_plot):
            item.setMenuEnabled(False)
            item.hideButtons()
            item.showGrid(x=True, y=True, alpha=0.15)
            self._time_plot_item(item)
        self.split_processed_plot.setXLink(self.split_raw_plot)
        # One tick-text width for both: a link lines the two up by SCREEN position, so
        # left axes of different widths showed different x ranges and unaligned ticks.
        for item in (self.split_raw_plot, self.split_processed_plot):
            item.getAxis("left").setStyle(autoExpandTextSpace=False,
                                          tickTextWidth=_SPLIT_TICK_TEXT_PX)
        self.split_raw_legend = _legend(self.split_raw_plot)
        self.split_processed_legend = _legend(self.split_processed_plot)
        self.split_raw_curve = ChannelPane.make_curve(self.split_raw_plot, _RAW_INK)
        self.split_raw_curve.setPen(pg.mkPen(QtGui.QColor(139, 145, 153, 200), width=1))
        self.split_processed_curve = ChannelPane.make_curve(self.split_processed_plot,
                                                            _PROCESSED_INK)
        self.split_raw_legend.addItem(self.split_raw_curve, "raw (calibrated)")
        self.split_processed_legend.addItem(self.split_processed_curve,
                                            "processed — window estimate")
        self.time_stack.addWidget(self.split_view)

        self.difference_plot = self._plot()
        di = self.difference_plot.getPlotItem()
        self._time_plot_item(di)
        self.difference_legend = _legend(di)
        self.difference_curve = ChannelPane.make_curve(di, _DIFF_INK)
        self.difference_legend.addItem(self.difference_curve, "raw − processed")
        self.time_stack.addWidget(self.difference_plot)
        lay.addWidget(self.time_stack, 1)
        self.preview_tabs.addTab(page, "Time")
        self.preview_tabs.setTabToolTip(
            0, "The visible window: raw (grey) against processed (blue), split onto their "
               "own axes when the chain changes the unit or removes DC (Ctrl+1).")
        self._time_page = page

    # -- the Spectrum tab -----------------------------------------------------------------
    def _build_spectrum_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.spectrum_note = _ElidedLine("")
        self.spectrum_note.setObjectName("signalConditioningSpectrumNote")
        self.spectrum_note.setStyleSheet("color:#8b9199;")
        lay.addWidget(self.spectrum_note)
        # The plot is built on the tab's first visit (a pyqtgraph plot is ~80 QObjects).
        self.spectrum_plot: pg.PlotWidget | None = None
        self._line_markers: list[pg.InfiniteLine] = []
        self._spectrum_slot = QtWidgets.QVBoxLayout()
        self._spectrum_slot.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self._spectrum_slot, 1)
        caption = QtWidgets.QLabel("Narrow lines in the raw spectrum (≥ 10 dB above the local "
                                   "floor) — tick and send them to the notch")
        caption.setWordWrap(True)
        caption.setStyleSheet("color:#8b9199;")
        lay.addWidget(caption)
        self.lines_list = _LinesList()
        self.lines_list.setObjectName("signalConditioningLines")
        self.lines_list.setUniformItemSizes(True)
        self.lines_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lines_list.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                      QtWidgets.QSizePolicy.Fixed)
        for _ in range(_MAX_LINE_ROWS):
            item = QtWidgets.QListWidgetItem("")
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                          | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setHidden(True)
            self.lines_list.addItem(item)
        row_h = max(1, self.lines_list.sizeHintForRow(0))
        self.lines_list.setFixedHeight(5 * row_h + 2 * self.lines_list.frameWidth() + 2)
        # A double-click also ACTIVATES the row; connecting both would send it twice.
        self.lines_list.itemActivated.connect(self._send_one_line)
        self.lines_list.returnPressed.connect(
            lambda: self._send_one_line(self.lines_list.currentItem()))
        lay.addWidget(self.lines_list)
        row = QtWidgets.QWidget()
        row.setFixedHeight(self.STATUS_ROW_HEIGHT)
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.send_to_notch_button = QtWidgets.QPushButton("Send to notch")
        self.send_to_notch_button.setObjectName("signalConditioningSendToNotch")
        self.send_to_notch_button.setFocusPolicy(QtCore.Qt.TabFocus)
        self.send_to_notch_button.setToolTip(
            "Add the ticked lines (or the selected one) to the notch's explicit line list "
            "and turn the notch on (an off notch that keeps a list of its own stays off, "
            "and the lines join that list). Lines already removed are skipped; the list "
            "holds at most 32 lines. Double-click or Return sends one line.")
        self.send_to_notch_button.clicked.connect(lambda *_: self.send_checked_lines())
        rl.addWidget(self.send_to_notch_button)
        rl.addStretch(1)
        lay.addWidget(row)
        # Its own two full-width lines: beside the button the outcome was cut to its
        # first words, which hid what was refused or kept off.
        self.lines_note = _ElidedLine("", lines=2)
        self.lines_note.setObjectName("signalConditioningLinesNote")
        self.lines_note.setStyleSheet("color:#8b9199;")
        lay.addWidget(self.lines_note)
        self.preview_tabs.addTab(page, "Spectrum")
        self.preview_tabs.setTabToolTip(
            1, "Welch spectra of the same window before and after the chain, and the "
               "narrow lines a notch could remove (Ctrl+2).")
        self._spectrum_page = page

    def _ensure_spectrum_plot(self) -> None:
        if self.spectrum_plot is not None:
            return
        self.spectrum_plot = self._plot(log_x=True, log_y=True)
        si = self.spectrum_plot.getPlotItem()
        si.getViewBox().setMouseEnabled(x=False, y=False)
        self.spectrum_plot.setLabel("bottom", "frequency (Hz)")
        self.spectrum_legend = _legend(si)
        self.spectrum_raw_curve = si.plot(pen=pg.mkPen(QtGui.QColor(139, 145, 153, 220), width=1))
        self.spectrum_processed_curve = si.plot(pen=pg.mkPen(_PROCESSED_INK, width=1))
        self.spectrum_legend.addItem(self.spectrum_raw_curve, "raw (calibrated)")
        self.spectrum_legend.addItem(self.spectrum_processed_curve, "processed")
        for _ in range(_MAX_LINE_ROWS):
            marker = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen(_LINE_INK, width=1, style=QtCore.Qt.DashLine))
            marker.setVisible(False)
            si.addItem(marker, ignoreBounds=True)
            self._line_markers.append(marker)
        self._spectrum_slot.addWidget(self.spectrum_plot, 1)

    # -- the Response tab -----------------------------------------------------------------
    def _build_response_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        host, self._response_group, self.response_view_buttons = _segmented(
            (("magnitude", "Magnitude"), ("phase", "Phase"), ("group_delay", "Group delay"),
             ("step", "Step")),
            {"magnitude": "|H(f)| as applied (zero-phase IIR stages are squared).",
             "phase": "The applied phase (zero-phase) and the single-pass causal design.",
             "group_delay": "The applied group delay (0) and the single-pass causal design's.",
             "step": "A unit step through the linear stages, as applied and single-pass."})
        self.response_view_buttons["magnitude"].setChecked(True)
        self._response_group.buttonClicked.connect(lambda *_: self._on_view_changed())
        lay.addWidget(host)
        self._toggle_slot = QtWidgets.QVBoxLayout()
        self._toggle_slot.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self._toggle_slot)
        self.response_stack = QtWidgets.QStackedWidget()
        self._response_page = QtWidgets.QWidget()
        self._response_layout = QtWidgets.QVBoxLayout(self._response_page)
        self._response_layout.setContentsMargins(0, 0, 0, 0)
        self.response_stack.addWidget(self._response_page)

        # The three plots are built on the first visit to a non-magnitude view.
        self._response_views: dict[str, dict[str, Any]] = {}
        self._response_slots: dict[str, QtWidgets.QVBoxLayout] = {}
        for key in ("phase", "group_delay", "step"):
            w = QtWidgets.QWidget()
            slot = QtWidgets.QVBoxLayout(w)
            slot.setContentsMargins(0, 0, 0, 0)
            slot.setSpacing(4)
            self.response_stack.addWidget(w)
            self._response_slots[key] = slot
        lay.addWidget(self.response_stack, 1)
        self.preview_tabs.addTab(page, "Response")
        self.preview_tabs.setTabToolTip(
            2, "The selected stage (or every enabled linear stage): magnitude, phase, group "
               "delay and step response (Ctrl+3).")
        self._response_tab_page = page

    def _ensure_response_plots(self) -> None:
        if self._response_views:
            return
        for key, y_label in (("phase", "phase (°)"), ("group_delay", "group delay"),
                             ("step", "step response")):
            plot = self._plot(log_x=key != "step")
            item = plot.getPlotItem()
            item.getViewBox().setMouseEnabled(x=key == "step", y=False)
            plot.setLabel("left", y_label)
            plot.setLabel("bottom", "time from the step" if key == "step" else "frequency (Hz)")
            legend = _legend(item)
            applied = item.plot(pen=pg.mkPen(_PROCESSED_INK, width=2))
            reference = item.plot(pen=pg.mkPen(_RAW_INK, width=1, style=QtCore.Qt.DashLine))
            for curve in (applied, reference):
                # A 1 Hz low-pass's step spans 400k samples: undecimated, each repaint
                # held the GUI thread for ~100 ms. Peak keeps the overshoot.
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
            legend.addItem(applied, "applied (zero-phase)")
            legend.addItem(reference, "single pass, causal (reference)")
            title = _ElidedLine("")
            title.setStyleSheet("font-weight:600;")
            # Top-aligned and elided on its last line: a centred overflow cut the readout.
            note = _ElidedLine("", lines=5)
            note.setStyleSheet("color:#8b9199;")
            slot = self._response_slots[key]
            slot.addWidget(title)
            slot.addWidget(plot, 1)
            slot.addWidget(note)
            self._response_views[key] = {"plot": plot, "applied": applied,
                                         "reference": reference, "note": note, "title": title}

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
        # The scope toggle serves all four response views, so it sits above them.
        old = parts.response_view.layout()
        if old is not None:
            old.removeWidget(parts.chain_response_toggle)
        self._toggle_slot.addWidget(parts.chain_response_toggle)
        parts.chain_response_toggle.toggled.connect(lambda *_: self._on_view_changed())
        self._response_layout.addWidget(parts.response_view, 1)
        self._parts = parts
        self._order_tab_chain()

    def _order_tab_chain(self) -> None:
        """Tab from the stage list goes straight into the shown stage's form, then the
        previews, the protocol bar and the status buttons.

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
                 + [self.preview_tabs] + stops(self._time_page) + stops(self._spectrum_page)
                 + stops(self._response_tab_page)
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
        chain = conditioning_params(self._params)
        changed = (chain != self._chain or fs != self._fs
                   or lowpass_design_hz != self._lowpass_design_hz)
        self._fs = fs
        self._lowpass_design_hz = lowpass_design_hz
        self._render_timer.start()
        if changed:
            self._chain = chain
            if self._exact_key is not None and self._exact_key != self._current_memo_key():
                self._cancel_exact()
            self._refresh_time_controls()
            self._schedule()

    def flush(self) -> None:
        """Render a pending :meth:`set_chain` now."""
        if self._render_timer.isActive():
            self._render_timer.stop()
            self._render_pending()

    def _render_pending(self) -> None:
        self._render_rows()
        # The status and legends say "updating" from the edit on, not from the debounce.
        self._refresh_status_texts()

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
        if key != self._stage_key:
            self._stage_key = key
            if self.preview_tab() == "response" and not self._combined_response():
                self._on_view_changed()
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

    def set_source(self, source: ConditioningSource | None) -> None:
        """The selected channel's calibrated samples (None: nothing to preview)."""
        previous = self._source
        old = previous.key if previous is not None else None
        new = source.key if source is not None else None
        self._source = source
        if old == new:
            # The same samples (the key holds the calibration): a re-cached array, a moved
            # clock or a renamed unit needs no recompute, only a redraw.
            if source is not None and (source.x_offset_s, source.units) != (
                    previous.x_offset_s, previous.units):
                self._refresh_time_controls()
                self._render_time()
            return
        if self._exact_key is not None:
            self._cancel_exact()
        self._last = None
        self._last_error = ""
        self._exact_error = (None, "")
        self.lines_note.setText("")
        self._clear_plots()
        self._refresh_time_controls()
        self._refresh_status_texts()
        self._schedule()

    def drop_document(self, doc_id: str) -> None:
        """A document closed: forget its exact passes and stop previewing it."""
        self._memo.drop_doc(doc_id)
        if self._source is not None and self._source.doc_id == doc_id:
            self.set_source(None)

    def set_view_window(self, start: int, stop: int) -> None:
        """The main trace's visible sample range; kept only while following."""
        if self.follows_main():
            rng = (int(start), int(stop))
            if rng != self._window_range:
                self._window_range = rng
                self._schedule()

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

    def reset_splitter(self) -> None:
        total = max(self.splitter.width(), self.DEFAULT_WIDTH)
        self.splitter.setSizes([self.LEFT_WIDTH, max(1, total - self.LEFT_WIDTH)])

    def restore_splitter_state(self, state) -> bool:
        return bool(self.splitter.restoreState(QtCore.QByteArray(state)))

    def show_notch_outcome(self, text: str) -> None:
        """What "Send to notch" did (MainWindow passes the panel's sentence back)."""
        self.lines_note.setText(text)

    # -- view state (persisted by MainWindow) ------------------------------------------------
    def preview_tab(self) -> str:
        return TAB_KEYS[max(0, self.preview_tabs.currentIndex())]

    def set_preview_tab(self, key: str) -> bool:
        if key not in TAB_KEYS:
            return False
        self.preview_tabs.setCurrentIndex(TAB_KEYS.index(key))
        return True

    def time_mode(self) -> str:
        """The Time tab's mode as shown (Split when the chain forces it)."""
        forced = self._forced_split()
        if forced and self._time_choice == "overlay":
            return "split"
        if self._units().changes_units and self._time_choice == "difference":
            return "split"
        return self._time_choice

    def time_mode_choice(self) -> str:
        """The mode the user chose (what is saved), whether or not the chain overrides it."""
        return self._time_choice

    def set_time_mode(self, mode: str) -> bool:
        if mode not in TIME_MODES:
            return False
        self._time_choice = mode
        self._refresh_time_controls()
        self._render_time()
        return True

    def response_view(self) -> str:
        for key, button in self.response_view_buttons.items():
            if button.isChecked():
                return key
        return "magnitude"

    def set_response_view(self, view: str) -> bool:
        if view not in RESPONSE_VIEWS:
            return False
        self.response_view_buttons[view].setChecked(True)
        self._on_view_changed()
        return True

    # -- test and status accessors -----------------------------------------------------------
    def preview_state(self) -> str:
        """'idle' | 'pending' | 'window' | 'exact' | 'error' for the Time/Spectrum preview
        ('pending' from the edit on: through the debounce as well as the computation)."""
        if self._window_pending():
            return "pending"
        if self._last_error or (self._last is not None and (
                self._last.error or self._last.degenerate)):
            return "error"
        if self._last is None or self._last.raw_window is None:
            return "idle"
        return "exact" if self._last.exact else "window"

    def last_result(self) -> PreviewResult | None:
        return self._last

    def exact_pending(self) -> bool:
        return self._exact_token is not None

    def worker_thread(self) -> QtCore.QThread | None:
        return self._thread

    def processed_axis_label(self) -> str:
        return self._units().label

    def spectral_line_texts(self) -> list[str]:
        return [self.lines_list.item(i).text() for i in range(self.lines_list.count())
                if not self.lines_list.item(i).isHidden()]

    def legend_texts(self) -> list[str]:
        return [label.text for legend in (self.overlay_legend, self.split_raw_legend,
                                          self.split_processed_legend, self.difference_legend)
                for _sample, label in legend.items]

    def response_view_widgets(self, view: str) -> dict[str, Any]:
        """The plot, curves, title and note of a Phase / Group delay / Step view."""
        self._ensure_response_plots()
        return self._response_views[view]

    # -- lifetime ----------------------------------------------------------------------------
    def abort(self) -> None:
        """MainWindow's teardown: stop the preview thread for good (a bounded join).

        The window's one thread lives as long as the window; closing the window only
        cancels its work (:meth:`closeEvent`). Nothing restarts it after this."""
        self._render_timer.stop()
        self._debounce.stop()
        self._worker_retired = True
        self._stop_worker()
        self._last = self._last_response = None
        self._clear_plots()
        self._dirty = True

    def _ensure_worker(self) -> bool:
        if self._thread is not None:
            return True
        if self._worker_retired:
            return False
        self._mailbox = _PreviewMailbox()
        self._worker = _PreviewWorker(self._mailbox)
        self._previewPosted.connect(self._worker.drain)
        self._thread = start_worker_thread(self._worker, domain=JOB_NONE,
                                           label="signal-conditioning preview", owner=self)
        self._thread.setPriority(QtCore.QThread.LowestPriority)
        return True

    def _clear_pending(self) -> None:
        """Every terminal path ends here: nothing posted is still awaited."""
        self._window_answered = self._window_token
        self._response_posted = False
        self._exact_token = self._exact_key = None
        self._refresh_exact_button()

    def _cancel_work(self) -> None:
        """Drop every waiting request and cancel the running one; the thread stays."""
        self._render_timer.stop()
        self._debounce.stop()
        if self._mailbox is not None:
            self._mailbox.withdraw()
        self._clear_pending()

    def _stop_worker(self) -> None:
        self._outcome_timer.stop()
        if self._thread is not None:
            mailbox, worker, thread = self._mailbox, self._worker, self._thread
            self._mailbox = self._worker = self._thread = None
            mailbox.stop()
            try:
                self._previewPosted.disconnect(worker.drain)
            except (RuntimeError, TypeError):
                pass
            retire_worker_thread(worker, thread, wait_ms=DEFAULT_JOIN_MS)
        self._clear_pending()

    def _post(self, req: PreviewRequest) -> bool:
        if not self._ensure_worker():
            return False
        self._mailbox.post(req)
        self._previewPosted.emit()
        if not self._outcome_timer.isActive():
            self._outcome_timer.start()
        return True

    def _collect_outcomes(self) -> None:
        mailbox = self._mailbox
        if mailbox is None:
            self._outcome_timer.stop()
            return
        for outcome in mailbox.collect():
            self._on_outcome(outcome)
        if self._mailbox is None or self._mailbox.idle():
            self._outcome_timer.stop()

    def worker_idle(self) -> bool:
        """Nothing is waiting for, running on or coming back from the worker."""
        return self._mailbox is None or self._mailbox.idle()

    # -- scheduling ---------------------------------------------------------------------------
    def _schedule(self, *, immediate: bool = False) -> None:
        """Restart the 250 ms debounce; a hidden window only marks itself dirty."""
        if not self.isVisible():
            self._dirty = True
            self._debounce.stop()
            return
        self._debounce.start(0 if immediate else self.DEBOUNCE_MS)
        self._status_timer.start()

    def _on_view_changed(self) -> None:
        if self.preview_tab() == "spectrum":
            self._ensure_spectrum_plot()
        index = RESPONSE_VIEWS.index(self.response_view())
        if self.response_stack.currentIndex() != index:
            self.response_stack.setCurrentIndex(index)
        self._render_response()
        self._schedule()

    def _on_time_mode_clicked(self, button) -> None:
        for key, b in self.time_mode_buttons.items():
            if b is button:
                self.set_time_mode(key)
                return

    def _on_follow_toggled(self, on: bool) -> None:
        self.followMainChanged.emit(bool(on))

    def _on_user_range(self, vb: pg.ViewBox, mask) -> None:
        """The user panned or zoomed a Time plot: stop following and preview that range."""
        if isinstance(mask, (list, tuple)) and mask and not mask[0]:
            return
        src = self._source
        if src is None or not src.fs or src.x_offset_s is None:
            return
        x0, x1 = vb.viewRange()[0]
        i0 = int(np.floor((x0 - src.x_offset_s) * src.fs))
        i1 = int(np.ceil((x1 - src.x_offset_s) * src.fs))
        i0, i1 = max(0, min(i0, src.n)), max(0, min(i1, src.n))
        if i1 - i0 < 16:
            return
        if self.follows_main():
            self.act_follow.setChecked(False)
        if (i0, i1) != self._window_range:
            self._window_range = (i0, i1)
            self._schedule()

    def _signature(self) -> str:
        src = self._source
        return chain_signature(self._chain, src.fs if src is not None and src.fs else self._fs)

    def _current_memo_key(self) -> tuple | None:
        src = self._source
        if src is None:
            return None
        return memo_key(src.key, self._chain, src.fs)

    def _units(self):
        units = self._source.units if self._source is not None else ""
        return units_after_chain(self._chain, units)

    def _forced_split(self) -> bool:
        u = self._units()
        return u.changes_units or u.removes_dc

    def _combined_response(self) -> bool:
        parts = getattr(self, "_parts", None)
        return bool(parts is not None and parts.chain_response_toggle.isChecked())

    def _response_stages(self) -> tuple[list[dict[str, Any]], list[str], str]:
        """``(stages, omitted, title)`` for the phase / group-delay / step views."""
        if self._combined_response():
            stages, omitted = linear_stage_specs(self._params)
            names = " → ".join(stage_title(s) for s in stages) or "no linear stage on"
            return stages, omitted, f"Enabled linear chain — {names}"
        key = self._stage_key or "lowpass"
        stages, omitted = linear_stage_specs(self._params, only=key,
                                             lowpass_design_hz=self._lowpass_design_hz)
        flag = STAGE_ENABLE_KEY.get(key)
        enabled = (self._params.get("filter_cutoff_hz") is not None if flag is None
                   else bool(self._params.get(flag)))
        state = "ON" if enabled else "OFF · design preview"
        return stages, omitted, f"{STAGE_TITLES.get(key, key)} — {state}"

    def _post_preview(self) -> None:
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        tab = self.preview_tab()
        src = self._source
        want_time = tab in ("time", "spectrum") and src is not None and bool(src.fs)
        view = self.response_view() if tab == "response" else "magnitude"
        response_view = view if view != "magnitude" else None
        stages: tuple[dict[str, Any], ...] = ()
        self._response_posted = False
        if response_view is not None:
            stages = tuple(self._response_stages()[0])
            if self._response_is_quick(response_view, stages):
                # Data-free and a few ms: never queued behind a seconds-long exact pass.
                self._compute_response_here(response_view, stages)
                response_view = None
        self._response_posted = response_view is not None
        if not want_time and response_view is None:
            # Nothing left to wait for; a slow step still queued is no longer wanted.
            if self._mailbox is not None:
                self._mailbox.withdraw("window")
            self._token += 1
            self._window_token = self._window_answered = self._token
            self._refresh_status_texts()
            return
        self._token += 1
        token = self._token
        start = stop = 0
        memo = None
        if want_time:
            rng = self._window_range or (0, src.n)
            start, stop, _clamped = clamp_window(rng[0], rng[1], src.n)
            memo = self._memo.get(self._current_memo_key())
        req = PreviewRequest(
            token, "window", dict(self._chain), src.fs if want_time else self._fs,
            raw=src.data if want_time else None, start=start, stop=stop,
            signature=self._signature(),
            source_key=src.key if src is not None else None, memo=memo,
            want_time=want_time, want_spectrum=want_time and tab == "spectrum",
            response_view=response_view, response_stages=stages)
        self._window_token = token
        if not self._post(req):
            self._clear_pending()
        if self._response_posted:
            self._render_response()
        self._refresh_status_texts()

    def _response_is_quick(self, view: str, stages: Sequence[Mapping[str, Any]]) -> bool:
        if view != "step" or not self._fs:
            return True
        try:
            side = int(np.ceil(step_span_s(stages, float(self._fs)) * float(self._fs)))
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            return True                     # the error is reported by the computation
        return 4 * min(side, MAX_STEP_SIDE_SAMPLES) <= _GUI_STEP_SAMPLES

    def _compute_response_here(self, view: str, stages: Sequence[Mapping[str, Any]]) -> None:
        self._token += 1
        self._last_response = compute_preview(PreviewRequest(
            self._token, "window", dict(self._chain), self._fs, want_time=False,
            response_view=view, response_stages=tuple(stages)))
        self._render_response()

    def compute_exact(self) -> bool:
        """Run the whole recording through the chain off the GUI thread (on demand only)."""
        src = self._source
        key = self._current_memo_key()
        if (src is None or not src.fs or key is None or key in self._memo
                or self._exact_token is not None or window_is_exact(self._chain, src.fs)):
            return False
        self._token += 1
        self._exact_token = self._token
        self._exact_key = key
        self._exact_error = (None, "")
        if not self._post(PreviewRequest(
                self._token, "exact", dict(self._chain), src.fs, raw=src.data, start=0,
                stop=src.n, signature=self._signature(), source_key=src.key)):
            self._exact_token = self._exact_key = None
            return False
        self._refresh_exact_button()
        self._refresh_status_texts()
        return True

    def _cancel_exact(self) -> None:
        if self._mailbox is not None:
            self._mailbox.withdraw("exact")
        self._exact_token = self._exact_key = None
        self._refresh_exact_button()

    def _on_outcome(self, outcome: _PreviewOutcome) -> None:
        if self._worker is None:
            return          # the thread was stopped (teardown): nothing is awaited
        if outcome.kind == "exact":
            if outcome.token != self._exact_token:
                return
            key = self._exact_key
            self._exact_token = self._exact_key = None
            res = outcome.result
            if outcome.status == "ok" and res is not None and res.full is not None:
                self._memo.put(key, *res.full)
                self._exact_elapsed = {k: v for k, v in self._exact_elapsed.items()
                                       if k in self._memo}
                self._exact_elapsed[key] = res.elapsed_s
                self._schedule(immediate=True)
            elif outcome.status == "error" or (res is not None and res.error):
                self._exact_error = (key, outcome.message or (res.error if res else ""))
            self._refresh_exact_button()
            self._refresh_status_texts()
            return
        if outcome.token != self._window_token:
            return          # superseded: a newer request is on its way
        if outcome.status == "cancelled":
            self._window_answered = outcome.token     # nothing newer is waiting on it
            self._response_posted = False
            self._refresh_status_texts()
            return
        res = outcome.result
        current_source = self._source.key if self._source is not None else None
        if res is not None and (res.signature, res.source_key) != (self._signature(),
                                                                   current_source):
            return          # the form or the recording moved on; its request is coming
        self._window_answered = outcome.token
        posted, self._response_posted = self._response_posted, False
        if outcome.status == "ok" and res is not None:
            if res.raw_window is not None:
                self._last = res
                self._last_error = ""
                self._render_time()
                self._render_spectrum()
            if posted and (res.phase is not None or res.step is not None
                           or res.response_error):
                self._last_response = res
            if posted:
                self._render_response()
        elif outcome.status == "error":
            self._last_error = outcome.message or "the preview failed"
            if posted:
                self._render_response()
        self._refresh_exact_button()
        self._refresh_status_texts()

    # -- drawing -------------------------------------------------------------------------------
    def _clear_plots(self) -> None:
        curves = [self.raw_curve, self.processed_curve, self.split_raw_curve,
                  self.split_processed_curve, self.difference_curve]
        if self.spectrum_plot is not None:
            curves += [self.spectrum_raw_curve, self.spectrum_processed_curve]
        for curve in curves:
            curve.setData([], [])
        for marker in self._line_markers:
            marker.setVisible(False)
        self._line_rows = []
        for i in range(self.lines_list.count()):
            self.lines_list.item(i).setHidden(True)

    def _refresh_time_controls(self) -> None:
        u = self._units()
        overlay = self.time_mode_buttons["overlay"]
        diff = self.time_mode_buttons["difference"]
        overlay.setEnabled(not (u.changes_units or u.removes_dc))
        diff.setEnabled(not u.changes_units)
        overlay.setToolTip(
            "Not available: Normalize changes the unit, so raw and processed need their "
            "own axes." if u.changes_units else
            "Not available: the high-pass removes DC, so raw and processed need their own "
            "axes." if u.removes_dc else "Raw and processed on one axis.")
        diff.setToolTip(
            "Not available: Normalize changes the unit, so raw − processed has no meaning."
            if u.changes_units else "What the chain removed: raw minus processed.")
        mode = self.time_mode()
        if not self.time_mode_buttons[mode].isChecked():
            self.time_mode_buttons[mode].setChecked(True)
        self._show_time_page(mode)
        self._refresh_exact_button()

    def _time_page_item(self, mode: str) -> pg.PlotItem:
        return {"overlay": self.overlay_plot.getPlotItem(), "split": self.split_raw_plot,
                "difference": self.difference_plot.getPlotItem()}[mode]

    def _show_time_page(self, mode: str) -> None:
        """Raise ``mode``'s page at the x range the page it replaces shows: a range the user
        panned to must not revert to a stale one when Normalize or a high-pass forces Split."""
        index = TIME_MODES.index(mode)
        current = self.time_stack.currentIndex()
        if current == index:
            return
        if 0 <= current < len(TIME_MODES):
            x0, x1 = self._time_page_item(TIME_MODES[current]).getViewBox().viewRange()[0]
            self._time_page_item(mode).setXRange(x0, x1, padding=0.0)
        self.time_stack.setCurrentIndex(index)

    def _x(self, start: int, stop: int) -> np.ndarray:
        src = self._source
        return (np.arange(start, stop, dtype=np.float64) / float(src.fs)
                + float(src.x_offset_s or 0.0))

    def _render_time(self) -> None:
        mode = self.time_mode()
        self._show_time_page(mode)
        res = self._last
        src = self._source
        if res is None or res.raw_window is None or src is None or not src.fs:
            for curve in (self.raw_curve, self.processed_curve, self.split_raw_curve,
                          self.split_processed_curve, self.difference_curve):
                curve.setData([], [])
            return
        x = self._x(res.start, res.stop)
        raw = res.raw_window
        proc = res.processed_window
        units = src.units or "units"
        u = self._units()
        self._refresh_legends()
        x_label = clock.axis_label("s", clock.INSTRUMENT, src.x_offset_s)
        x_short = (f"s ({clock.CLOCK_SHORT.get(clock.INSTRUMENT, clock.INSTRUMENT)})"
                   if x_label.startswith("s (") else x_label)
        pages = {"overlay": (self.raw_curve, self.processed_curve, None),
                 "split": (self.split_raw_curve, self.split_processed_curve, None),
                 "difference": (None, None, self.difference_curve)}
        for key, (rc, pc, dc) in pages.items():
            if key != mode:
                for curve in (rc, pc, dc):
                    if curve is not None:
                        curve.setData([], [])
        rc, pc, dc = pages[mode]
        if rc is not None:
            rc.setData(x, raw)
        if pc is not None:
            if proc is not None:
                pc.setData(x, proc)
            else:
                pc.setData([], [])
        if dc is not None:
            if proc is not None and not u.changes_units:
                dc.setData(x, np.asarray(raw, dtype=np.float64) - np.asarray(proc, np.float64))
            else:
                dc.setData([], [])
        if mode == "overlay":
            item = self.overlay_plot.getPlotItem()
            self._set_axis_label(item, "left", f"raw and processed ({units})")
            self._set_axis_label(item, "bottom", x_label, x_short)
            items = [item]
        elif mode == "split":
            self._set_axis_label(self.split_raw_plot, "left", f"raw ({units})", units)
            self._set_axis_label(self.split_processed_plot, "left", u.label, u.short_label,
                                 u.unit_label)
            self._set_axis_label(self.split_processed_plot, "bottom", x_label, x_short)
            items = [self.split_raw_plot]
        else:
            item = self.difference_plot.getPlotItem()
            self._set_axis_label(item, "left", f"raw − processed ({units})")
            self._set_axis_label(item, "bottom", x_label, x_short)
            items = [item]
        lo, hi = float(x[0]), float(x[-1])
        for item in items:
            v0, v1 = item.getViewBox().viewRange()[0]
            # A range the user panned to stays theirs; only Follow (or a view that no
            # longer overlaps the data) moves it.
            if self.follows_main() or v1 < lo or v0 > hi:
                item.setXRange(lo, hi, padding=0.0)

    def _set_axis_label(self, item: pg.PlotItem, side: str, full: str, *shorter: str) -> None:
        """``full`` on the axis when it fits its length, else the first of ``shorter``
        that does (the last elided if need be); the full text is always the axis's
        tooltip. Re-fitted whenever the plot resizes."""
        axis = item.getAxis(side)
        vb = item.getViewBox()
        self._axis_labels[id(axis)] = (vb, axis, side, full, tuple(shorter))
        _fit_axis_label(vb, axis, side, full, tuple(shorter))

    def _refit_axis_labels(self) -> None:
        for entry in self._axis_labels.values():
            _fit_axis_label(*entry)

    def axis_label_text(self, item: pg.PlotItem, side: str) -> str:
        """The FULL label an axis stands for (what its tooltip says), for tests."""
        entry = self._axis_labels.get(id(item.getAxis(side)))
        return entry[3] if entry is not None else str(item.getAxis(side).labelText or "")

    def _active_notch_lines(self) -> list[float]:
        p = self._chain
        if not p.get("notch_enabled") or not self._fs:
            return []
        try:
            return filters.notch_frequencies(
                float(self._fs), float(p.get("notch_freq_hz") or 60.0),
                harmonics=int(p.get("notch_harmonics") or 3), freqs=notch_freqs_param(p))
        except (TypeError, ValueError):
            return []

    def _render_spectrum(self) -> None:
        res = self._last
        sp = res.spectrum if res is not None else None
        if sp is None:
            return
        self._ensure_spectrum_plot()
        self.spectrum_raw_curve.setData(sp.f_hz, sp.raw_psd)
        if sp.processed_psd is not None:
            self.spectrum_processed_curve.setData(sp.f_hz, sp.processed_psd)
        else:
            self.spectrum_processed_curve.setData([], [])
        # Scaled to the raw spectrum: a low-pass's stop band otherwise pulls the axis
        # down ten decades and flattens everything worth reading.
        good = sp.raw_psd[np.isfinite(sp.raw_psd) & (sp.raw_psd > 0)]
        if good.size:
            self.spectrum_plot.setYRange(float(np.log10(np.percentile(good, 0.5))) - 2.0,
                                         float(np.log10(good.max())) + 0.5, padding=0.0)
            self.spectrum_plot.setXRange(float(np.log10(sp.f_hz[0])),
                                         float(np.log10(sp.f_hz[-1])), padding=0.0)
        units = self._source.units if self._source is not None else ""
        self.spectrum_plot.setLabel("left", f"PSD ({units}²/Hz)" if units else "PSD")
        # Short: a legend wider than the plot is clipped from the left, samples first.
        _set_legend_text(self.spectrum_legend, 1,
                         "processed (no normalize)" if sp.normalize_omitted else "processed")
        note = f"Δf {sp.df_hz:.2f} Hz (nperseg {sp.nperseg:,}) on a {sp.window_s:.2f} s window"
        if sp.normalize_omitted:
            note += " · the processed curve leaves Normalize out (it only rescales)"
        if sp.df_hz > 1.0 and self._source is not None and self._source.fs:
            need = window_for_resolution_s(float(self._source.fs), 1.0)
            note += (" — resolving mains needs Δf ≤ 1 Hz: widen the window to ≥ "
                     f"{need:.0f} s" if need is not None else
                     " — Δf ≤ 1 Hz is not reachable at this sample rate")
        self.spectrum_note.setText(note)
        q = float(self._chain.get("notch_q") or 30.0)
        active = self._active_notch_lines()
        checked = {line.freq_hz for line, on in self._line_choices() if on}
        self._line_rows = []
        lines = list(res.lines)[:_MAX_LINE_ROWS]
        for i, marker in enumerate(self._line_markers):
            if i < len(lines):
                marker.setPos(float(np.log10(lines[i].freq_hz)))
                marker.setVisible(True)
            else:
                marker.setVisible(False)
        was = self.lines_list.blockSignals(True)
        try:
            for i in range(self.lines_list.count()):
                item = self.lines_list.item(i)
                if i >= len(lines):
                    item.setHidden(True)
                    item.setCheckState(QtCore.Qt.Unchecked)
                    continue
                line = lines[i]
                # Judged at the MEASURED peak: a notch on the harmonic beside it is not one
                # that removes it.
                notched = any(notch_covers(line.peak_hz, f, q) for f in active)
                placeable = line_is_placeable(line, sp.df_hz, q)
                text = line.text + (" · notched" if notched else "") + (
                    "" if placeable else " · Δf too coarse")
                item.setText(text)
                usable = placeable and not notched
                flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable
                item.setFlags(flags | QtCore.Qt.ItemIsEnabled if usable else flags)
                on = usable and line.freq_hz in checked
                item.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
                item.setToolTip(
                    "Already removed by the notch." if notched else
                    f"Δf {sp.df_hz:.2f} Hz is wider than this line's notch (f/Q = "
                    f"{line.freq_hz / q:.2f} Hz): its frequency is not known well enough to "
                    "place a notch on it. Widen the window." if not placeable else
                    "Tick to send to the notch, or double-click (or Return) to send this "
                    "line now.")
                item.setHidden(False)
                self._line_rows.append((line, on, usable))
        finally:
            self.lines_list.blockSignals(was)
        lo, hi = res.lines_band_hz
        band = f"{lo:g} Hz – {hi / 1e3:g} kHz" if hi >= 1e3 else f"{lo:g} – {hi:g} Hz"
        if not lines:
            self.lines_note.setText(f"No narrow line stands ≥ 10 dB above the local floor "
                                    f"in {band}.")
        elif not self.lines_note.text().startswith(("Notch", "Refused", "Tick")):
            self.lines_note.setText(
                f"{len(lines)} line{'s' if len(lines) != 1 else ''} in {band}.")

    def _line_choices(self) -> list[tuple[Any, bool]]:
        out = []
        for i, row in enumerate(self._line_rows):
            item = self.lines_list.item(i)
            out.append((row[0], bool(row[2]) and item.checkState() == QtCore.Qt.Checked))
        return out

    def send_checked_lines(self) -> list[float]:
        """Emit the ticked usable lines (or the selected usable one) for the notch."""
        freqs = [line.freq_hz for line, on in self._line_choices() if on]
        if not freqs:
            row = self.lines_list.currentRow()
            if 0 <= row < len(self._line_rows) and self._line_rows[row][2]:
                freqs = [self._line_rows[row][0].freq_hz]
        if not freqs:
            self.lines_note.setText("Tick the lines to send (or double-click one).")
            return []
        self.notchLinesRequested.emit(list(freqs))
        return freqs

    def _send_one_line(self, item: QtWidgets.QListWidgetItem) -> None:
        row = self.lines_list.row(item)
        if 0 <= row < len(self._line_rows) and self._line_rows[row][2]:
            self.notchLinesRequested.emit([self._line_rows[row][0].freq_hz])

    def _render_response(self) -> None:
        view = self.response_view()
        if view == "magnitude":
            return
        self._ensure_response_plots()
        # The note's height is fixed; its full sentence is always one hover away.
        self._draw_response(view, self._response_views[view])

    def _draw_response(self, view: str, widgets: dict[str, Any]) -> None:
        stages, omitted, title = self._response_stages()
        widgets["title"].setText(title)
        res = self._last_response
        applied, reference = widgets["applied"], widgets["reference"]
        note_tail = (" Not representable as a linear response: " + "; ".join(omitted) + "."
                     if omitted else "")
        # The readout and the omissions lead; the legend's meaning comes last (the note's
        # height is fixed and its tooltip holds the rest).
        omitted_lead = note_tail.strip() + " " if note_tail else ""
        if not stages:
            applied.setData([], [])
            reference.setData([], [])
            widgets["note"].setText(
                ("No linear stage is enabled." if self._combined_response()
                 else "This stage has no linear response to draw.") + note_tail)
            return
        if self._fs is None:
            applied.setData([], [])
            reference.setData([], [])
            widgets["note"].setText("Needs a channel with a known sample rate." + note_tail)
            return
        if self._response_posted:
            # A slow stage's step goes to the worker, which may be busy with the exact pass.
            widgets["note"].setText(
                ("Waiting for the exact pass to finish before computing this step "
                 "response…" if self._exact_token is not None else
                 "Computing this step response (a slow stage, off the GUI thread)…")
                + note_tail)
            return
        if res is None or (res.phase is None and res.step is None):
            if res is not None and res.response_error:
                applied.setData([], [])
                reference.setData([], [])
                widgets["note"].setText(f"Response unavailable ({res.response_error})."
                                        + note_tail)
            return
        plot = widgets["plot"]
        zero_phase = ("Solid: as applied — zero-phase, so edges stay put. Dashed: the same "
                      "design run once, causally (a hardware filter).")
        if view in ("phase", "group_delay") and res.phase is not None:
            ph = res.phase
            if view == "phase":
                applied.setData(ph.f_hz, np.degrees(ph.applied_phase_rad))
                reference.setData(ph.f_hz, np.degrees(ph.reference_phase_rad)
                                  if ph.has_reference else np.zeros_like(ph.f_hz))
                plot.setLabel("left", "phase (°)")
                extra = " A 180° step in the solid line is a smoother's sign flip, not a delay."
            else:
                gd = ph.reference_group_delay_s if ph.has_reference else ph.applied_group_delay_s
                finite = np.abs(gd[np.isfinite(gd)])
                span = float(finite.max()) if finite.size else 0.0
                a, unit = _scaled_seconds(ph.applied_group_delay_s, span or 1e-6)
                r, _u = _scaled_seconds(ph.reference_group_delay_s, span or 1e-6)
                applied.setData(ph.f_hz, a)
                reference.setData(ph.f_hz, r, connect="finite")
                plot.setLabel("left", f"group delay ({unit})")
                extra = ""
            widgets["note"].setText(omitted_lead + zero_phase + extra)
        elif view == "step" and res.step is not None:
            st = res.step
            span = float(st.t_s[-1] - st.t_s[0]) / 2.0
            t, unit = _scaled_seconds(st.t_s, span)
            applied.setData(t, st.applied)
            if st.reference is not None:
                reference.setData(t, st.reference)
            else:
                reference.setData([], [])
            plot.setLabel("bottom", f"time from the step ({unit})")
            plot.setLabel("left", "step response (unit step)")
            if st.removes_dc:
                text = ("The high-pass removes DC, so the plateau droops back toward 0: "
                        "rise and overshoot do not apply. ")
            else:
                text = (f"Applied: 10–90 % rise {_duration_text(st.applied_rise_s)} · "
                        f"overshoot {100 * st.applied_overshoot:.1f} % · pre-shoot "
                        f"{100 * st.applied_preshoot:.1f} %."
                        + (f" Single pass: rise {_duration_text(st.reference_rise_s)} · "
                           f"overshoot {100 * st.reference_overshoot:.1f} %."
                           if st.reference is not None else "") + " ")
            if st.truncated:
                text += "The span is truncated at 10 s each side. "
            widgets["note"].setText(text + omitted_lead + zero_phase)

    def _refresh_exact_button(self) -> None:
        src = self._source
        key = self._current_memo_key()
        usable = (src is not None and bool(src.fs) and key is not None
                  and not window_is_exact(self._chain, src.fs))
        last = self._last
        # A window that already reached both ends of the recording IS the exact trace.
        shown_exact = (usable and last is not None and last.exact
                       and last.signature == self._signature()
                       and last.source_key == src.key)
        enabled = bool(usable and not shown_exact and key not in self._memo
                       and self._exact_token is None)
        self.exact_button.setEnabled(enabled)
        failed = self._exact_failure()
        text = (_EXACT_TEXTS[1] if self._exact_token is not None
                else _EXACT_TEXTS[2] if failed and enabled else _EXACT_TEXTS[0])
        if self.exact_button.text() != text:
            self.exact_button.setText(text)
        if failed and enabled:
            tip = f"The last exact pass of this chain failed ({failed}). Press to retry."
        elif enabled or src is None or not src.fs:
            tip = _EXACT_TIP
        elif self._exact_token is not None:
            tip = "Conditioning the whole recording… (an edit of the chain cancels it)."
        elif not usable:
            tip = ("Not needed: every enabled stage is exact on the padded window, so the "
                   "preview already shows the whole-recording samples.")
        else:
            tip = "Done: the preview shows this chain's whole-recording (exact) samples."
        if self.exact_button.toolTip() != tip:
            self.exact_button.setToolTip(tip)

    def _exact_failure(self) -> str:
        """Why the last exact pass of the CURRENT chain failed ('' if it did not)."""
        key, message = self._exact_error
        return message if key is not None and key == self._current_memo_key() else ""

    def _fit_exact_button(self) -> None:
        """As wide as its widest state, so a state change never moves the status beside it."""
        button = self.exact_button
        shown = button.text()
        widths = []
        for text in _EXACT_TEXTS:
            button.setText(text)
            widths.append(button.sizeHint().width())
        button.setText(shown)
        button.setFixedWidth(max(widths))

    def _refresh_status_texts(self) -> None:
        self.time_status.setText(self._time_status_text())
        self._refresh_legends()
        if self._last is None or self._last.spectrum is None:
            src = self._source
            note = ("No recording selected." if src is None else
                    "Spectrum unavailable: sample rate unknown." if not src.fs else "")
            if note or self._last is None:          # never another recording's Δf
                self.spectrum_note.setText(note)

    def _is_stale(self, res: PreviewResult) -> bool:
        """``res`` was computed for another chain (or recording) than the form's now."""
        current = self._source.key if self._source is not None else None
        return (res.signature, res.source_key) != (self._signature(), current)

    def _window_pending(self) -> bool:
        return self._window_token != self._window_answered or self._debounce.isActive()

    def _refresh_legends(self) -> None:
        res = self._last
        if res is None or res.raw_window is None:
            text = "processed — window estimate"
        elif self._is_stale(res):
            text = "processed — previous chain"
        else:
            text = "processed — exact" if res.exact else "processed — window estimate"
        _set_legend_text(self.overlay_legend, 1, text)
        _set_legend_text(self.split_processed_legend, 0, text)
        if self.spectrum_plot is not None and res is not None and res.spectrum is not None:
            _set_legend_text(self.spectrum_legend, 1,
                             "processed — previous chain" if self._is_stale(res)
                             else "processed (no normalize)" if res.spectrum.normalize_omitted
                             else "processed")

    def _time_status_text(self) -> str:
        """What is happening now (computing, failed, updating) first, then the verdict on
        the CURRENT chain (exact or window estimate), the clamp, then the details: the
        widget shows three lines and elides the rest into its tooltip."""
        src = self._source
        if src is None:
            return ("No recording selected — select a channel to preview the chain on its "
                    "visible window.")
        if not src.fs:
            return ("Preview unavailable: sample rate unknown — the filter stages are "
                    "designed in Hz and seconds.")
        fs = float(src.fs)
        res = self._last
        shown = res is not None and res.raw_window is not None
        pending = self._window_pending()
        stale = shown and self._is_stale(res)
        failed = self._exact_failure()
        lead = []
        if self._exact_token is not None:
            lead.append("Computing exact (whole recording)…"
                        + (" — the preview waits for it" if pending else ""))
        elif failed:
            lead.append(f"Exact pass failed: {failed}")
        if not shown:
            if self._exact_token is not None:
                return lead[0]
            if self._last_error and not pending:
                return " · ".join(lead + [f"Preview unavailable: {self._last_error}"])
            return " · ".join(lead + ["Computing the preview…" if pending else "Preview idle."])
        if pending and self._exact_token is None:
            lead.append("Updating…")
        rng = self._window_range or (0, src.n)
        if stale:
            a, b, _ = clamp_window(rng[0], rng[1], src.n)
        else:
            a, b = res.start, res.stop
        dur = (b - a) / fs
        # The clamp comes straight after the verdict, ahead of any advice or timing.
        clamp = (f" · centre {dur:.1f} s of the {(rng[1] - rng[0]) / fs:.1f} s view (preview "
                 f"limit)" if rng[1] - rng[0] > MAX_PREVIEW_WINDOW_SAMPLES else "")
        advice = (" · press Compute exact for the true trace"
                  if self._exact_token is None and not failed else "")
        verdict = None
        if not stale:
            if res.window_too_short:
                verdict = (f"Window too short for Baseline drift "
                           f"({res.error or res.degenerate}){clamp}" + advice)
            elif res.error:
                verdict = f"Preview unavailable: {res.error}"
            elif res.degenerate:
                verdict = f"Preview unavailable: {res.degenerate}"
            elif res.from_exact_pass:
                elapsed = self._exact_elapsed.get(self._current_memo_key())
                took = ("" if elapsed is None else f" · the pass took {elapsed:.1f} s"
                        if elapsed >= 0.1 else f" · the pass took {elapsed * 1e3:.0f} ms")
                verdict = (f"Whole recording (exact){clamp or f' · showing {dur:.2f} s'}"
                           + took)
        if verdict is None:
            chain = self._chain
            pad = res.pad if not stale else preview_pad_samples(chain, fs)
            exact = res.exact if not stale else (
                window_is_exact(chain, fs) or (a - pad <= 0 and b + pad >= src.n))
            if not conditioning_enabled(chain):
                verdict = f"No stage is on, so processed = raw{clamp} · window {dur:.2f} s"
            elif exact:
                took = "" if stale else f" · {res.elapsed_s * 1e3:.0f} ms"
                verdict = (f"Exact for these stages{clamp} · window {dur:.2f} s, padded "
                           f"±{pad / fs:.2f} s" + took)
            else:
                why = []
                estimators = estimate_stage_titles(chain)
                if estimators:
                    verb = "use" if len(estimators) > 1 else "uses"
                    why.append(f"{'/'.join(estimators)} {verb} whole-recording statistics")
                for title, seconds in pad_limited_stages(chain, fs):
                    why.append(_settles_text(title, seconds, pad / fs))
                verdict = (f"Window estimate — {'; '.join(why) or 'the window is not exact'}"
                           + clamp + advice)
        return " · ".join(lead + [verdict])

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
            self._fit_exact_button()
            self._time_status_row.setFixedHeight(max(
                self.time_status.height(), self.exact_button.sizeHint().height()))

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._fit_stage_list()
        self.flush()
        if self._dirty or self._last is None:
            self._schedule(immediate=True)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Closing HIDES the window (it is built once and kept); nothing is deleted.

        Its work is cancelled and its preview arrays freed, but the one worker thread is
        kept for the next open: a QThread built and destroyed per open is the zombie
        shape this codebase avoids. MainWindow's teardown stops it (:meth:`abort`)."""
        self._cancel_work()
        self._last = self._last_response = None
        self._clear_plots()
        self._dirty = True
        self._refresh_status_texts()
        super().closeEvent(event)
