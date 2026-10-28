"""Analysis control panels (ported from SOURCE ``gui/panels.py``, re-seamed).

These are pure view widgets: they collect parameters and emit signals. All
business logic (running detection, fitting, sweeping) lives in
:class:`~nanoscope.gui.analysis_panel.AnalysisPanel`, which wires these signals
to the :class:`~nanoscope.gui.analysis_worker.AnalysisController`. The one
exception is the live rise-time readout, which calls
:func:`nanoscope.analysis.filters.measure_rise_time` directly because it is a
cheap, pure computation whose whole point is to update *as the user types*.

Adaptation from SOURCE (PORT_PLAN §6):

* ``DetectionPanel.params()`` returns EXACTLY the keyword arguments the target
  :func:`~nanoscope.analysis.detect.detect_events` accepts (``polarity``,
  ``k_in``, ``k_out``, ``min_dwell_s``, ``max_dwell_s``, ``filter_cutoff_hz``,
  ``filter_order``, ``filter_kind``, ``merge_gap_s``) with the ``None``
  off-sentinels, plus the preprocessing keys the module consumes before detection.
  The filter order/kind controls are BACK: ``detect_events`` now takes
  ``filter_order`` / ``filter_kind`` and forwards them to both ``lowpass`` and
  ``measure_rise_time``, so the form cannot lie about what detection does — the
  rise-time readout is measured through the exact design the user selected, and
  the design is stamped into provenance. Defaults are unchanged (8-pole Bessel
  low-pass, 4-pole Butterworth high-pass), so an untouched panel behaves exactly
  as before.
* ``set_recording`` also re-ranges every frequency spin against the recording's
  Nyquist, so a cutoff the kernel would reject is not reachable.
* A live frequency-response preview draws :func:`filters.response` for the
  selected stage. All the numerics live in the kernel; the panel only plots.
* ``set_recording`` TOLERATES ``fs is None`` (legacy TDMS): the readout says
  "rate unknown" instead of raising, and :meth:`DetectionPanel.set_runnable`
  disables the action buttons with an honest reason (``requires_time_base``).
* A "show on trace" filter-preview toggle drives the managed
  :meth:`TraceView.set_filter_preview` curve (PORT_PLAN D4).

Sign convention note: depth / frac_depth shown anywhere are positive magnitudes
(see :data:`nanoscope.analysis.events.SCHEMA`). Direction is a polarity readout,
never a signed feature.
"""

from __future__ import annotations

import json

import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractProxyModel,
    QAbstractTableModel,
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSettings,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QTabWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from .inspector_widgets import Disclosure, InspectorScrollArea, PinnedActions, host_layout
from .numeric_text import parse_strict_number

from ..core import clock
from ..analysis import events as events_mod
from ..analysis import baseline as baseline_mod
from ..analysis import filters, ml_detect
from ..analysis.conditioning import chain_description
from ..analysis.conditioning_preview import merge_notch_lines
from ..analysis.conditioning_recipes import (
    BUILTIN_RECIPES,
    CHAIN_KEYS,
    DETREND_METHOD_LABELS,
    MANUAL_PROTOCOL,
    SCOPE_SENTENCE,
    STAGE_KEYS,
    MergeReport as ProtocolMergeReport,
    chain_differences,
    complete_recipe,
    describe_differences,
    merge_imported_protocols,
    recipe_key_differences,
    recipes_equal,
    validate_recipe,
)
from ..analysis.modules import DETECTOR_PRIVATE_KEYS
from ..analysis.detect_cusum import (
    DEFAULT_CUSUM_H,
    DEFAULT_CUSUM_K,
    DEFAULT_CUSUM_MERGE_GAP_S,
    DEFAULT_CUSUM_MIN_DWELL_S,
)
from ..analysis.events import (
    AUTOMATIC_REJECTION_FLAGS,
    FLAG_CURATED,
    SCHEMA,
    combine_flags,
    flag_tokens,
    mark_reviewed,
)
from ..analysis.ml_preprocess import (
    BASELINE_METHODS,
    DEFAULT_NORMALIZE_CLIP_SIGMA,
    NORMALIZE_METHODS,
    DECODE_PRESETS,
    DEFAULT_PRESET,
    NOTCH_FREQS,
    PREPROCESS_PRESETS,
    PRESET_ALIASES,
    PRESET_LABELS,
    PRESET_MOLECULE,
    PRESET_NAMES,
    MLDecodeConfig,
    MLPreprocessConfig,
    preset_for_window,
    resolve_preset_name,
)
from ..analysis.reconcile import clear_curation_conflicts
from ..analysis.remeasure import EstimatorComparison
from ..analysis.scoring import ScoreResult

__all__ = [
    "DetectionPanel", "StatsPanel", "ScorePanel", "EventTable", "SweepDialog",
    "EstimatorCompareDialog",
]

#: The model input widths that are EVIDENCE ABOUT THE MOLECULE, not merely a hint about
#: which preset to default to.
#:
#: :func:`~nanoscope.analysis.ml_preprocess.preset_for_window` maps both 2048 and 3072 to
#: a preset, and its docstring is explicit that only one of them settles the molecule:
#: 3072 is BOTH apo lineages and nothing else, while at 2048 there are the three ferritin
#: notebooks AND an apo student (``ApoStudentModelMerged``) -- "2048 is not even proof of
#: the molecule". So a 3072 model under a ferritin preset is a contradiction worth saying
#: out loud, and a 2048 model under an apo preset is not evidence of anything. Stated once,
#: here, rather than inferred at the two places that would each have to get it right.
_WIDTH_SETTLES_MOLECULE = frozenset({3072})

#: How a channel whose file declared no units — and whose Calibration has not set any —
#: is named in a read-out here. THE SAME WORDING the Calibration panel (``_fmt_voltage``),
#: the I-V window (``_fmt_q``) and the PSD window (``_UNKNOWN_UNITS``) use, because a
#: fourth wording for one state is a fourth thing to keep in step. The Summary used to
#: caption its two amplitude rows "(pA)" unconditionally: on the owner's electroporation
#: HDF5, whose own loader note says "Units are not declared in this file. Magnitudes
#: suggest nA", it read "median depth (pA) 0.07274" beside a PSD read-out of "0.004204
#: (units unknown)" — one session, one channel, two answers — and it went on saying "(pA)"
#: even after the user declared the channel nA in Calibration (FE-04).
_UNKNOWN_UNITS = "units unknown"

#: Default reason on a badged column. Named once so the header, every cell and the
#: staleness banner cannot end up telling the user three different stories.
_STALE_COLUMN_TIP = (
    "Measured against an open-pore baseline that is no longer the configured one. "
    "These values are shown exactly as they were stored -- nothing has been "
    "re-measured. Use the Re-measure depths button on the banner above, or "
    "Analysis > Compare baseline estimators... to see how much they would change."
)

#: The RAW, comparable cell value — what a sorted view must order on. Sorting on the
#: DISPLAY string is a text sort: ``f"{v:.4g}"`` puts "99.49" above "150.2" (a descending
#: depth_pA sort then shows none of the ten deepest events), ".6g" emits exponents so a
#: 9.9e-05 s dwell sorts as the LONGEST, and an integer ``event_id`` runs 0,1,10,11,2.
#: Same constant, same value and the same reason as the measurements table's
#: ``measure_panel._SORT_ROLE`` — the two sortable tables in this app answer "which row
#: is bigger" the one way.
_SORT_ROLE = Qt.UserRole + 1

#: Event-table columns that are an INSTANT in seconds, and so are on a clock. Every
#: one of them is on the RECORDING clock (0 = first loaded sample) while the trace
#: x-axis beside them is on the instrument clock -- see :mod:`nanoscope.core.clock`.
#: Durations (``dwell_s``, ``rise_s``, ``fall_s``) are deliberately NOT here: a
#: length of time is on no clock and annotating it would be noise.
_CLOCK_COLUMNS = frozenset({"t_start_s", "t_end_s"})

#: Per-column explanations for the columns whose NAME is not the whole story.
_COLUMN_TIPS = {
    "depth_pA": (
        "Median depth: the median polarity-adjusted excursion from the analysis "
        "baseline over the whole detected event. This original measurement is not "
        "replaced when you choose another amplitude estimator in Curate."),
    "amplitude_pA": (
        "Selected amplitude: the optional measurement explicitly applied in Curate. "
        "Read amplitude_method, amplitude_units and amplitude_status with this value; "
        "the legacy column suffix does not override the recorded signal units."),
    "amplitude_method": "The estimator used for the selected amplitude, not for depth_pA.",
    "amplitude_status": (
        "current: measured for these event bounds and processing. stale: retained "
        "for audit after its inputs changed; reapply the measurement before using it."),
    "frac_depth": (
        "depth_pA / |baseline_pA| — the depth as a fraction of the open-pore current "
        "it was measured against.\n\n"
        "A blockade is between 0 and 1. A value ABOVE 1 means the current magnitude "
        "GREW during the event and this is not a blockade fraction."),
    "baseline_pA": (
        "The open-pore current STAMPED ON THIS EVENT by the characterisation "
        "baseline — the reference depth_pA was subtracted from, measured locally.\n\n"
        "The I-V / Conductance window's I₀ is a different estimate of the open pore "
        "(one value for the whole recording, its own estimator) and the two do not "
        "have to agree; on the owner's apoferritin recordings they differ by up to "
        "5.1%."),
    "recording": "Constant for every row of one detection — shown once above the table.",
    "pore_id": "Constant for every row of one detection — shown once above the table.",
}


# ---------------------------------------------------------------------------
# DetectionPanel
# ---------------------------------------------------------------------------
#: Baseline-drift estimator labels, keyed by the method string
#: :func:`nanoscope.analysis.baseline.remove_baseline_wander` accepts.
#:
#: THE SAVITZKY-GOLAY LABEL SAYS "(subtract)" FOR A REASON. There are two
#: Savitzky-Golay stages in the Preprocess tab and they are OPPOSITE operations
#: sharing a name: the smoother on the Savitzky-Golay page RETURNS the polynomial fit
#: (it keeps the smooth trace), while this estimator SUBTRACTS it (it keeps the
#: residual). Labelling both "Savitzky-Golay" is how a user ends up smoothing when
#: they meant to detrend.
_DETREND_LABELS: dict[str, str] = dict(DETREND_METHOD_LABELS)

#: Baseline-drift estimators NOT offered in the Preprocess tab's method combo.
#: 'constant' is a no-op detrend (a flat level; subtracting it changes nothing but the
#: DC offset), so offering it would only invite an enabled stage that does nothing.
_DETREND_OMITTED: frozenset[str] = frozenset({"constant"})


def _near(value: float, target: float | None, *, rtol: float = 1e-9) -> bool:
    """True when a spin still holds ``target`` (or ``target`` is not a thing).

    Used only by `DetectionPanel._on_detector_changed` to tell "the widget is still at
    the other detector's default" from "the user typed this". A tolerance rather than
    ``==`` because the value has been through a spin's decimals and, for the merge gap,
    a seconds-to-milliseconds round trip.
    """
    if target is None:
        return False
    return abs(float(value) - float(target)) <= max(1e-12, abs(float(target)) * rtol)


def _notch_lines_text(lines: Sequence[float]) -> str:
    """'60, 120, 180 Hz', or '12 lines: 60, 120, 180 … 4570.8 Hz' for a long list."""
    spelled = [f"{float(f):g}" for f in lines]
    if len(spelled) <= 6:
        return ", ".join(spelled) + " Hz"
    return f"{len(spelled)} lines: " + ", ".join(spelled[:3]) + f" … {spelled[-1]} Hz"


def _detrend_method_choices() -> tuple[tuple[str, str], ...]:
    """``(label, method)`` pairs for the detrend combo, in combo-index order.

    A module-level function purely because a class body is not a closure: a
    comprehension inside one cannot see the class's own attributes, so the labels and
    the omission set have to live out here for the list to be built from
    ``baseline.METHODS`` at all.
    """
    return tuple(
        (_DETREND_LABELS[m], m)
        for m in baseline_mod.METHODS
        if m not in _DETREND_OMITTED
    )


def _narrow_combo(minimum_chars: int = 18) -> QComboBox:
    """A combo box that does NOT take its width from its longest item.

    Qt's default ``SizeAdjustPolicy`` is ``AdjustToContentsOnFirstShow``: the combo sizes
    itself to the LONGEST string it holds. Three combos in this panel carry sentence-length
    labels — the ML preset names run to 115 characters — and each became a hard width floor
    for its form, its group, its page, the tab widget, the panel, and finally the Analysis
    scroll area's content. On the owner's 1512x900 screen that content is shown through a
    420 px viewport, so the surplus turned into a horizontal scrollbar and everything to the
    right of it went off-screen: 706 px of preset combo is how the ``Save as…``/``Delete``
    pair immediately below it stopped being clickable.

    Nothing is lost by capping: the POPUP still sizes itself to the longest item, so the
    full text is there to read when it matters, and the closed combo elides. The policy has
    to be set BEFORE ``addItem`` — ``QComboBox`` caches ``minimumSizeHint`` on first show,
    and lowering the policy afterwards does not lower the cached floor.
    """
    combo = QComboBox()
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(minimum_chars)
    return combo


class _ElidingCheckBox(QCheckBox):
    """A checkbox whose LABEL may elide instead of flooring the panel's width.

    WHY THIS EXISTS. The Inspector dock ships 460 px wide (a 420 px viewport) on the
    owner's 1512x900 screen. ``QCheckBox`` has no shrink flag: ``qSmartMinSize`` takes
    ``max(sizeHint, minimumSizeHint)`` for it, so one long label — 'enable
    (peak-preserving polynomial smooth — KEEPS the fit)' measured 387 px — becomes a
    hard minimum for its form, its group box, its stack page, the whole tab and finally
    the scroll area's content width. The content then exceeds the viewport, a horizontal
    scrollbar appears, and every control to the right of it goes off-screen. Three of
    those labels were between them worth ~130 px of the Preprocess page's floor and
    ~155 px of ``StatsPanel``'s.

    WHAT IT CHANGES. The text is untouched — ``text()`` still returns the full label and
    every ``setChecked``/``toggled`` behaviour is the base class's. Only the PAINT elides,
    and only when the width actually on offer is too small; at a wide dock the label draws
    in full, exactly as before. The full text is also put in the tooltip when the call site
    has not supplied one of its own.

    NOTHING MOVES. The minimum is a constant of the widget, not of its state, so the
    checkbox neither grows nor shrinks when it is ticked, and no neighbour is displaced.
    """

    #: How much label a squeezed checkbox must still show, in 'x'-widths. Twelve
    #: characters is enough for 'enable (peak…' — the reader can always widen the dock,
    #: which is precisely the thing this class restores.
    _MIN_LABEL_CHARS = 12

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        # Preferred is the policy that carries QSizePolicy.ShrinkFlag; without it
        # qSmartMinSize ignores minimumSizeHint() below and the floor comes back.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        if text and not self.toolTip():
            self.setToolTip(text)

    def _chrome_width(self) -> int:
        """Width of everything that is not the label: indicator, spacing, margins."""
        return max(
            0,
            super().minimumSizeHint().width()
            - self.fontMetrics().horizontalAdvance(self.text()),
        )

    def minimumSizeHint(self) -> QSize:  # noqa: D102 - Qt override
        full = super().minimumSizeHint()
        if not self.text():
            return full
        stub = self.fontMetrics().horizontalAdvance("x" * self._MIN_LABEL_CHARS + "…")
        return QSize(min(full.width(), self._chrome_width() + stub), full.height())

    def paintEvent(self, event) -> None:  # noqa: D102 - Qt override
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        available = self.width() - self._chrome_width()
        if self.text() and available < self.fontMetrics().horizontalAdvance(self.text()):
            opt.text = self.fontMetrics().elidedText(
                self.text(), Qt.ElideRight, max(0, available)
            )
        QStylePainter(self).drawControl(QStyle.ControlElement.CE_CheckBox, opt)


class _WrappedLabel(QLabel):
    """A word-wrapped label never shorter than its text at its own width.

    QFormLayout gives a spanning wrapped label its size-hint height even when its
    height-for-width is more (measured: 60 of the 80 px the rise-time readout's four
    lines need at 310 px, so the last line was cut); a minimum height makes it honour it."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        super().setText(text)
        self._fit_height()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit_height()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self._fit_height()

    def _fit_height(self) -> None:
        need = self.heightForWidth(self.width()) if self.width() > 0 else -1
        if need > 0 and need != self.minimumHeight():
            self.setMinimumHeight(need)


class _CurrentPageStackedWidget(QWidget):
    """Stack-compatible layout whose hidden pages consume no vertical space.

    QStackedLayout's height-for-width calculation includes inactive long captions.
    A normal box layout excludes hidden pages, including their height-for-width,
    while the small public stack API keeps navigation independent of parameter state.
    """

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pages = []
        self._current = -1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def addWidget(self, widget) -> int:  # noqa: N802
        self._pages.append(widget)
        self.layout().addWidget(widget)
        widget.hide()
        if self._current < 0:
            self.setCurrentIndex(0)
        return len(self._pages) - 1

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int):
        return self._pages[index] if 0 <= index < self.count() else None

    def currentIndex(self) -> int:  # noqa: N802
        return self._current

    def currentWidget(self):  # noqa: N802
        return self.widget(self._current)

    def indexOf(self, widget) -> int:  # noqa: N802 - QStackedWidget parity
        return self._pages.index(widget) if widget in self._pages else -1

    def setCurrentWidget(self, widget) -> None:  # noqa: N802
        if widget in self._pages:
            self.setCurrentIndex(self._pages.index(widget))

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if index == self._current or not 0 <= index < self.count():
            return
        if self.currentWidget() is not None:
            self.currentWidget().hide()
        self._current = index
        self.currentWidget().show()
        self.layout().invalidate()
        self.updateGeometry()
        self.currentChanged.emit(index)

    def sizeHint(self) -> QSize:  # noqa: N802
        current = self.currentWidget()
        if current is None:
            return QSize(0, 0)
        return QSize(max(p.sizeHint().width() for p in self._pages), current.sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        current = self.currentWidget()
        if current is None:
            return QSize(0, 0)
        return QSize(max(p.minimumSizeHint().width() for p in self._pages),
                     current.minimumSizeHint().height())


class SignalChainEditorParts(NamedTuple):
    """The live stage-editor widgets the Signal Conditioning window hosts (reparented)."""

    protocol_bar: QWidget
    stage_stack: QWidget
    notices: QWidget
    response_view: QWidget
    chain_response_toggle: QCheckBox


class DetectionPanel(QWidget):
    """Filter, threshold, dwell and polarity controls for event detection.

    Its body is a four-tab ``mode_tabs``: **Preprocess | Detection | ML detector | Manual
    measurement**. The first three are this panel's own controls. The fourth is an empty
    page it merely HOSTS — the owning
    :class:`~nanoscope.gui.analysis_panel.AnalysisPanel` installs the cursor-measurement
    tool into it via :meth:`set_measure_widget`, because that tool needs the Session and
    the shared TraceView and this module imports no sibling gui panel. All four are ways of
    producing or measuring the events that land in the one shared event table below.

    Emits:
        detectRequested(dict): run detection with these detect_events kwargs.
        sweepRequested(dict): run sweep_threshold() with these detect kwargs.
        nullControlRequested(dict): run null_control() with these detect kwargs.
        recoveryRequested(dict): run recovery_map() with these detect kwargs.
        paramsChanged(dict): any parameter changed (for the live filter preview).
        previewFilterToggled(bool): the "show on trace" filter preview toggle.
        mlDetectRequested(dict): run the ML detector with these params
            (model_path, enter, exit, range_source).
        mlModelChanged(str): a .keras model was loaded (its path), or "" when cleared.
        notchLinesRefused(str): the Preprocess tab's explicit notch-line box refused a
            commit or a stored list -- the sentence the orange note under the box
            carries, so the owner can put it on the status line too.
    """

    detectRequested = Signal(dict)
    sweepRequested = Signal(dict)
    nullControlRequested = Signal(dict)
    recoveryRequested = Signal(dict)
    paramsChanged = Signal(dict)
    previewFilterToggled = Signal(bool)
    mlDetectRequested = Signal(dict)
    controlLevelChanged = Signal(str)  # presentation only; never paramsChanged
    _CONTROL_LEVEL_SETTINGS_KEY = "inspector/detection_control_level"
    mlModelChanged = Signal(str)
    notchLinesRefused = Signal(str)
    #: "Open Signal Conditioning…" on the card (and, while hosted, the baseline
    #: workspace's Advanced signal-chain settings): the stage key to show, or "" for the
    #: current one.
    openSignalConditioningRequested = Signal(str)
    #: The stage page now raised in the editor (a STAGE_KEYS key), whatever raised it.
    stageShown = Signal(str)
    #: A sentence about a protocol recall for the status line (e.g. what the form clamped).
    protocolNotice = Signal(str)
    #: The Detect gate moved (``set_runnable``): True when detection can run now.
    runGateChanged = Signal(bool)

    #: Stage key per editor page, in filter_selector / filter_stack order.
    _EDITOR_STAGE_KEYS: tuple[str, ...] = (
        "lowpass", "highpass", "notch", "median", "detrend", "rmean", "savgol", "normalize")

    # Baseline-drift estimators offered in the combo, in combo-index order, BUILT FROM
    # ``baseline.METHODS`` so the menu cannot drift from the set the kernel implements
    # -- the same discipline the low-pass/high-pass combos already follow via
    # ``filters.KINDS``. It drifted once already: three of the seven methods were
    # unreachable from this combo while ``set_detrend_method`` silently ignored their
    # names, so a protocol recalling ``masked_median`` ran the BLOCK MEDIAN with no
    # warning, no visible difference and provenance that read correct.
    #
    # 'constant' is the one deliberate omission: it is a no-op detrend (a flat level,
    # subtracting which changes nothing but the DC offset) and offering it would only
    # invite an enabled stage that does nothing.
    #: The detectors the Detection tab offers, as ``(label, ANALYSIS_REGISTRY id)``
    #: pairs in combo-index order. The id is what :meth:`params` emits under
    #: ``"detector"``, so the owning panel starts a module by NAME rather than by
    #: guessing from a boolean -- a third detector is one row here plus one page on
    #: ``detector_stack``.
    #:
    #: The Keras ML detector is deliberately absent: it does not run on this tab's
    #: preprocessed trace (it has its own `ml_preprocess` chain, its own model and its
    #: own analysed-range selector), so it keeps its own tab.
    _DETECTORS: tuple[tuple[str, str], ...] = (
        ("Schmitt trigger (hysteresis)", "detect.schmitt"),
        ("Sequential CUSUM", "detect.cusum"),
    )

    #: The DEFAULTS each detector wants for the two knobs they SHARE but disagree
    #: about. Seeded into the widgets when the user switches detector and the widget
    #: still holds the other detector's default -- never over a value the user chose.
    #:
    #: They differ by 25x on the merge gap: the Schmitt path merges at 0.02 ms (it
    #: fragments on noise crossings), while the CUSUM path's own calibration used
    #: 0.50 ms and a 1.00 ms minimum event. Inheriting the Schmitt numbers would run
    #: the owner's algorithm at settings his calibration never saw.
    _DETECTOR_DEFAULTS: dict[str, dict[str, float]] = {
        "detect.schmitt": {"merge_gap_s": 0.0, "min_dwell_s": 2e-5},
        "detect.cusum": {
            "merge_gap_s": float(DEFAULT_CUSUM_MERGE_GAP_S),
            "min_dwell_s": float(DEFAULT_CUSUM_MIN_DWELL_S),
        },
    }

    _DETREND_LABELS: dict[str, str] = _DETREND_LABELS
    _DETREND_OMITTED: frozenset[str] = _DETREND_OMITTED
    _DETREND_METHODS: tuple[tuple[str, str], ...] = _detrend_method_choices()

    #: The methods for which the ``window`` spinbox above the sub-stack is a NO-OP.
    #:
    #: `MASKED_METHODS` read ``rough_s``/``base_s`` and never look at ``window_s``;
    #: it rides on ``BaselineConfig`` only because the seven methods share one object.
    #: PINNED BY MEASUREMENT: two masked_median runs at window_s=0.05 and 5.0 are
    #: byte-identical (tests/analysis/test_detrend_methods_reachable.py). A knob the
    #: user turns while the numbers do not move, with nothing saying why, is worse
    #: than no knob -- so the spin is disabled and says so.
    _DETREND_WINDOW_IGNORED: frozenset[str] = frozenset({"masked_median"})

    # Display names for the filter kinds :mod:`nanoscope.analysis.filters` accepts. The
    # combos are BUILT from ``filters.KINDS`` / ``filters.HP_KINDS``, so the choices the
    # user sees can never drift from the ones the kernels implement; this maps each
    # kernel string to its label.
    _KIND_LABELS: dict[str, str] = {
        "bessel": "Bessel",
        "butter": "Butterworth",
        "gaussian": "Gaussian",
    }

    # Display names for the ML detector's baseline estimators. The combo is BUILT from
    # ``ml_preprocess.BASELINE_METHODS`` (as the filter combos are built from
    # ``filters.KINDS``), so what the user can pick can never drift from what
    # ``apply_ml_preprocess`` implements; this only maps each kernel string to a label.
    _BASELINE_LABELS: dict[str, str] = {
        "robust_median": "robust median",
        "masked_median": "event-masked median",
        "savgol": "Savitzky-Golay",
        "none": "none",
    }

    # Fallback spin ranges used until a recording's sample rate is known. Once
    # ``set_recording`` supplies fs these are replaced by Nyquist-derived maxima.
    _STATIC_LP_RANGE = (1.0, 5_000_000.0)
    _STATIC_HP_RANGE = (0.1, 5_000_000.0)
    _STATIC_NOTCH_RANGE = (1.0, 100_000.0)

    #: *** THE SCHMITT NOTCH-LINE CAP. *** The most lines the Preprocess tab's explicit
    #: "lines" box may commit — the ML tab's ``_ML_NOTCH_MAX_LINES`` (32) by value: the
    #: two boxes are siblings on this one panel, and this one was left without the guard
    #: the other's docstring says became necessary after "ONE accidental clipboard paste
    #: of a spreadsheet column". Measured 2026-09-01 on the owner's -600 mV recording
    #: (2,993,798 samples, 50 kHz): ``filters.notch`` costs ~10 ms per line (100 lines
    #: 0.97 s, 300 lines 3.01 s), so the 4,155 lines one truncated paste left in this box
    #: cost 42 s per detection there and ~174 s on the 12.5 M-sample -500 mV file. 32
    #: bounds that at ~0.3 s (~1.6 s at the 15 M-sample ceiling) and still spells a 50 Hz
    #: mains comb past 1.5 kHz — more harmonics than any recording here has needed. A
    #: test pins this to the ML constant so the two boxes cannot drift apart.
    NOTCH_MAX_LINES = 32
    #: How much text the "lines" box HOLDS. Qt's default is 32,767 characters, and a
    #: QLineEdit truncates a longer paste silently — which is how a 10,923-value column
    #: became 4,155 lines whose last token, '54.153999…' cut to '54.', parsed to a 54.0 Hz
    #: notch nobody typed (a duplicate of line #4000, which the kernel then refused by
    #: name, or — without the collision — a notch applied and stamped into provenance).
    #: The box now holds 250,000 characters (~30,000 lines; ``QLineEdit.setText``
    #: measured 67 ms at 100k characters, 856 ms at 1 M) so a real spreadsheet-column
    #: paste is refused BY ITS TRUE COUNT; a paste longer than even this is refused as
    #: "at least N lines" with the limit named, never parsed as if it were the paste.
    NOTCH_BOX_MAX_CHARS = 250_000
    #: How many wrapped lines the note under the "lines" box OWNS, whatever it says.
    #: A disclosure must not move the surface it discloses on: the note used to be a
    #: bare word-wrapping QLabel, so painting a refusal grew it (measured 2026-09-02 on
    #: the owner's -1000 mV excerpt: 15 -> 45 px, the filter stack 227 -> 240 px) and
    #: everything under it moved before the mouse RELEASE of the very click that
    #: committed the box -- the "show processed trace" checkbox moved +13 px and the
    #: release landed at local y = -3, so the click was lost. In the Inspector dock the
    #: note is 772 px wide (~110 characters a line at 15 px a line); the longest
    #: refusal with its restore sentence is 297 characters and a two-line list's
    #: description 110, so four lines hold every refusal whole at that width. Longer
    #: (a 32-line list under a refusal, a narrower dock) is shown from the head with
    #: the refusal kept whole and the rest in the tooltip -- see `_FixedLinesNote`.
    NOTCH_NOTE_LINES = 4

    # `measure_rise_time` pushes a step of ~40*fs/cutoff samples through the real filter,
    # so its cost explodes as the cutoff falls (at 250 kHz a 1 Hz cutoff is a 20 M-sample
    # probe and a ~640 MB peak). A live readout must never do that on the GUI thread; past
    # this many samples it says so instead. Detection itself is unaffected — it runs off
    # the GUI thread and measures whatever the user asked for.
    _MAX_RISE_PROBE_SAMPLES = filters.MAX_RISE_TIME_PROBE_SAMPLES

    #: The loaded-models combo's empty-state row. It carries NO itemData, which is what
    #: makes it inert (`_on_model_combo_changed` only acts on an entry with a path) and
    #: what `_sync_model_combo` tests for when it needs to add or drop it.
    _ML_NO_MODEL_ITEM = "none loaded"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Set once a channel is selected so the rise-time readout can convert the
        # filter cutoff into a real resolution floor for this fs. None == the
        # channel's sample rate is genuinely unknown (legacy TDMS).
        self._fs: float | None = None

        # ML detector state. The panel holds only the model PATH (a string) -- it
        # never loads the TensorFlow model on the GUI thread; the worker does, off
        # the main thread. _ml_channel_ok mirrors the Schmitt gate (CURRENT channel
        # with a known rate); the Run button ANDs it with "a model is loaded" and
        # "TensorFlow is available".
        self._model_path: str | None = None
        self._ml_channel_ok: bool = False
        self._ml_channel_reason: str = ""

        # ML preprocessing state. ``_ml_loading`` suppresses the deviation indicator
        # while a preset is being pushed into the widgets; ``_ml_preset_chosen`` records
        # that the USER picked a preset, which permanently stops the model-width
        # heuristic from overriding it. ``_ml_model_window`` is the loaded model's input
        # width when it could be read WITHOUT importing TensorFlow.
        self._ml_loading: bool = False
        self._ml_preset_chosen: bool = False
        self._ml_model_window: int | None = None

        # Signal-chain hosting: see set_signal_chain_hosted / signal_chain_editor_parts.
        self._chain_editor_hosted = False
        self._editor_parts: SignalChainEditorParts | None = None
        self._stage_focus_partner: QWidget | None = None
        self._chain_status_text = ""
        self._chain_status_warn = False

        root = QVBoxLayout(self)

        # -- gate banner (requires_time_base / wrong channel kind) --------
        # A panel-wide banner (the Schmitt run gate), kept ABOVE the tabs so its
        # reason is visible no matter which tab is raised.
        # THE ROW IS RESERVED, NOT CONJURED. This banner used to be a bare word-wrapping
        # QLabel that was added to and removed from the layout by show/hide, and the whole
        # panel below it moved 21 px every time the gate answered differently — measured:
        # selecting a recording (runnable) hid it and pulled `Save as…` / `Rename` /
        # `Delete` from y=477 to y=456, i.e. nearly a full button-row, and a calibration
        # edit does the same. Those three protocol verbs only became clickable this round,
        # so a shift that used to displace nothing a user could hit now dodges three fresh
        # controls. It is also the shape three earlier rounds of this project were lost to.
        #
        # Two parts, and both are needed:
        #   * ``_FixedLinesNote(2)`` — the codebase's reserved-note twin (see its docstring
        #     and ``event_inspector._FixedLinesNote``): it OWNS two lines whether it says
        #     anything or not, so the LONG refusal ("Detection runs on CURRENT channels;
        #     this is a voltage channel (your Calibration-panel override; …)", 122 chars,
        #     two lines at the shipped 520 px and three at the narrowest 420 px) no longer
        #     grows the banner past the short one either. Over-long text is shown from the
        #     head with "…" and the whole of it goes to the tooltip; ``text()`` still
        #     returns the full string, which is what the user-guide test reads.
        #   * ``_gate_host`` — a fixed-height carrier. The note keeps its own show/hide
        #     (``isHidden()`` is how five test modules and ``analysis_panel`` ask "is the
        #     gate up?", and an always-visible empty banner would answer that wrongly), so
        #     the reservation has to live one level out, where hiding the child cannot
        #     collapse it. Height is the note's own fixed height: reserving exactly what
        #     the banner takes when it speaks, so no page loses a pixel it was using.
        self._gate_label = _FixedLinesNote(2)
        self._gate_label.setStyleSheet("color:#FFB74D;")
        self._gate_label.setVisible(False)
        self._gate_host = QWidget()
        _gate_host_layout = QVBoxLayout(self._gate_host)
        _gate_host_layout.setContentsMargins(0, 0, 0, 0)
        _gate_host_layout.addWidget(self._gate_label)
        self._gate_host.setFixedHeight(self._gate_label.minimumHeight())
        root.addWidget(self._gate_host)

        # -- four-tab control: Preprocess / Detection / ML detector / Manual measurement --
        # Preprocessing (the signal-filter chain) is SEPARATE from detection, and each
        # detector gets its own tab. The three groups below are the SAME widgets as
        # before — just reparented into tab pages instead of stacked in one scroller.
        # A QTabWidget keeps every page alive, so params()/ml_params() read every
        # control regardless of the active tab, and a hidden stage stays enabled. This
        # is LAYOUT ONLY: the widgets, params(), ml_params(), setters, gating and every
        # emitted signal are byte-for-byte as before. Preprocess is index 0 (the default
        # active tab) so its filter widgets are the ones shown when the panel opens.
        #
        # The FOURTH page is a deliberately EMPTY container. Manual measurement is the
        # fourth way a user produces events (cursors instead of a threshold or a model),
        # so the owner asked for it to sit beside the other three as a peer rather than
        # at the bottom of a long scroll. This panel does not own that tool and must not:
        # the measurement widget needs the Session and the shared TraceView, and panels.py
        # imports no sibling gui module (MainWindow / the owning AnalysisPanel is the only
        # wirer). So the page is created and titled here — the tab exists and is
        # reorder-proof whether or not anything is installed in it — and the owner drops
        # its widget in through :meth:`set_measure_widget`. A bare ``DetectionPanel()``
        # therefore still has four tabs, the last one simply blank.
        self.mode_tabs = QTabWidget()
        self._preprocess_page = QWidget()
        self._detection_page = QWidget()
        self._ml_page = QWidget()
        self._measure_page = QWidget()
        preprocess_layout = QVBoxLayout(self._preprocess_page)
        preprocess_layout.setContentsMargins(0, 0, 0, 0)
        detection_layout = QVBoxLayout(self._detection_page)
        detection_layout.setContentsMargins(0, 0, 0, 0)
        ml_layout = QVBoxLayout(self._ml_page)
        ml_layout.setContentsMargins(0, 0, 0, 0)
        self._measure_layout = QVBoxLayout(self._measure_page)
        self._measure_layout.setContentsMargins(0, 0, 0, 0)
        # Preprocess contains two sizeable, independent jobs. Keep only the selected
        # one open so the signal-chain controls and their preview are reachable without
        # first scrolling through the open-pore characterisation form.
        preprocess_section_row = QHBoxLayout()
        preprocess_section_row.addWidget(QLabel("Preprocess section:"))
        self.preprocess_section_selector = QComboBox()
        self.preprocess_section_selector.addItems(
            ["Signal chain", "Open-pore baseline"])
        self.preprocess_section_selector.setToolTip(
            "Signal chain edits the ordered filters used for detection. Open-pore "
            "baseline configures the reference used to measure event depths.")
        preprocess_section_row.addWidget(self.preprocess_section_selector, 1)
        preprocess_layout.addLayout(preprocess_section_row)

        self.preprocess_sections = _CurrentPageStackedWidget()
        self._signal_chain_page = QWidget()
        self._baseline_page = QWidget()
        signal_chain_layout = QVBoxLayout(self._signal_chain_page)
        signal_chain_layout.setContentsMargins(0, 0, 0, 0)
        # This module HOSTS the baseline page and never builds, wires or reads the
        # widget. That control group talks to `analysis.baseline_probe` and an
        # off-thread job, so building it here would break the view-only seam.
        self._baseline_slot = QVBoxLayout(self._baseline_page)
        self._baseline_slot.setContentsMargins(0, 0, 0, 0)
        self.preprocess_sections.addWidget(self._signal_chain_page)
        self.preprocess_sections.addWidget(self._baseline_page)
        self.preprocess_section_selector.currentIndexChanged.connect(
            self.preprocess_sections.setCurrentIndex)
        self.preprocess_section_selector.setCurrentIndex(0)
        self.preprocess_sections.setCurrentIndex(0)
        preprocess_layout.addWidget(self.preprocess_sections)
        self.mode_tabs.addTab(self._preprocess_page, "Preprocess")
        self.mode_tabs.addTab(self._detection_page, "Detection")
        self.mode_tabs.addTab(self._ml_page, "ML detector")
        self.mode_tabs.addTab(self._measure_page, "Manual measurement")
        root.addWidget(self.mode_tabs)

        # -- Signal filters (dropdown + stacked pages) --------------------
        # The eight pre-detection filter stages share ONE compact group instead of
        # eight stacked group boxes: a combo picks which stage's controls are shown, a
        # QStackedWidget holds one page per stage, and an active-filters indicator keeps
        # the whole chain visible without clicking through. CRITICAL: the stack keeps
        # every page constructed, so params() still reads all eight regardless of which
        # page is raised, and a stage stays enabled while hidden. This is layout only —
        # the widgets, params(), setters and emitted signals are byte-for-byte as before,
        # just re-parented into pages.
        filters_box = QGroupBox("Signal filters")
        filters_layout = QVBoxLayout(filters_box)

        # The selected-stage response is a DESIGN preview even while that stage is off.
        # Keep that distinction in the above-fold caption; the plot title and verdict
        # below state the current ON/OFF execution status explicitly. An eliding caption
        # avoids imposing a minimum width on the Inspector dock.
        self.response_hint = _ElidingCaption(
            "↓ selected-stage |H(f)| design preview",
            elide=Qt.ElideRight)
        self.response_hint.setStyleSheet("color:#8b9199;")

        self.filter_selector = QComboBox()
        self.filter_selector.setToolTip(
            "Pick a filter stage to edit. Enabled stages are marked ● here and listed "
            "in the summary below; every stage stays active whether or not its page is "
            "the one currently shown."
        )
        self.active_filters_label = QLabel(
            "Active execution order: raw calibrated signal (no stages enabled)")
        self.active_filters_label.setWordWrap(True)
        self.active_filters_label.setStyleSheet("color:#8b9199;")
        self.filter_stack = QStackedWidget()

        # -- page: Low-pass filter ---------------------------------------
        # Type and order are the user's, not the app's. The kernel has always accepted
        # bessel/butter/gaussian at any order; the panel used to pin 8-pole Bessel in a
        # checkbox label, so the two facts a reader needs to reproduce a measurement were
        # unreachable AND unrecorded. Defaults are unchanged (8-pole Bessel), so an
        # untouched panel detects byte-for-byte as before.
        lp_page = QWidget()
        filt_form = QFormLayout(lp_page)
        self.filter_enabled = _ElidingCheckBox("enable Low-pass")
        self.filter_kind = QComboBox()
        self.filter_kind.addItems([self._KIND_LABELS[k] for k in filters.KINDS])
        self.filter_kind.setCurrentIndex(filters.KINDS.index("bessel"))
        self.filter_kind.setToolTip(
            "Bessel (the electrophysiology convention) has a maximally flat group delay: "
            "it preserves pulse shape and does not ring on a step. Butterworth rolls off "
            "harder for the same order but overshoots a square blockade. Gaussian has no "
            "overshoot at all and no order (it is a symmetric smoothing kernel).\n\n"
            "All three run zero-phase (forward+backward), so edges do not shift — but the "
            "magnitude is SQUARED: the response is -6 dB at the nominal cutoff, not -3 dB. "
            "The plot below shows what is actually applied."
        )
        self.filter_order = QSpinBox()
        self.filter_order.setRange(1, 10)
        self.filter_order.setValue(8)
        self.filter_order.setKeyboardTracking(False)
        self.filter_order.setToolTip(
            "Number of poles. Higher is a steeper roll-off and a longer rise time — and "
            "zero-phase filtering DOUBLES the effective order on top of this. Gaussian "
            "ignores it. 8 is the hardware convention for an Axopatch-style Bessel."
        )
        self.filter_cutoff_hz = QDoubleSpinBox()
        self.filter_cutoff_hz.setDecimals(1)
        self.filter_cutoff_hz.setRange(*self._STATIC_LP_RANGE)
        self.filter_cutoff_hz.setSingleStep(1000.0)
        self.filter_cutoff_hz.setValue(10_000.0)
        self.filter_cutoff_hz.setSuffix(" Hz")
        self.filter_cutoff_hz.setKeyboardTracking(False)
        self.filter_cutoff_hz.setToolTip(
            "Nominal -3 dB design cutoff. Once a channel is selected the maximum is "
            "clamped to just below that recording's Nyquist frequency, so a cutoff the "
            "kernel would reject cannot be dialled in."
        )
        filt_form.addRow("", self.filter_enabled)
        filt_form.addRow("type", self.filter_kind)
        filt_form.addRow("order", self.filter_order)
        filt_form.addRow("cutoff", self.filter_cutoff_hz)

        # THE readout. Bold, wraps, lives on the Low-pass page.
        self.rise_time_label = _WrappedLabel("rise time: — (select a channel)")
        f = self.rise_time_label.font()
        f.setBold(True)
        self.rise_time_label.setFont(f)
        filt_form.addRow(self.rise_time_label)
        self.filter_stack.addWidget(lp_page)

        # -- page: High-pass filter --------------------------------------
        # A zero-phase high-pass for DC / slow-drift removal in the classical
        # CANDIDATE-LOCALIZATION path. It distorts square translocation events, so the
        # quantitative path deliberately omits it; the tooltip names the metrics that
        # remain boundary-dependent. Applied before the baseline-drift stage in
        # preprocess_for_detection.
        hp_page = QWidget()
        hp_form = QFormLayout(hp_page)
        self.hp_enabled = _ElidingCheckBox("enable High-pass")
        self.hp_kind = QComboBox()
        self.hp_kind.addItems([self._KIND_LABELS[k] for k in filters.HP_KINDS])
        self.hp_kind.setCurrentIndex(filters.HP_KINDS.index("butter"))
        self.hp_kind.setToolTip(
            "Butterworth has a maximally flat pass-band above the cutoff. Bessel "
            "trades some roll-off sharpness for a flatter group delay. "
            "(Gaussian is a low-pass kernel and has no high-pass counterpart.)"
        )
        self.hp_order = QSpinBox()
        self.hp_order.setRange(1, 10)
        self.hp_order.setValue(4)
        self.hp_order.setKeyboardTracking(False)
        self.hp_order.setToolTip(
            "Number of poles. A steeper high-pass droops and rings a square blockade "
            "harder, not less — keep it low and keep the cutoff far below the event band."
        )
        self.hp_cutoff_hz = QDoubleSpinBox()
        self.hp_cutoff_hz.setDecimals(2)
        self.hp_cutoff_hz.setRange(*self._STATIC_HP_RANGE)
        self.hp_cutoff_hz.setSingleStep(10.0)
        self.hp_cutoff_hz.setValue(100.0)
        self.hp_cutoff_hz.setSuffix(" Hz")
        self.hp_cutoff_hz.setKeyboardTracking(False)
        _hp_warning = (
            "High-pass filtering removes slow drift/DC and distorts square-event shapes "
            "on the candidate-localization trace. "
            "Used by classical Schmitt/CUSUM for CANDIDATE LOCALIZATION only. The "
            "orange preview and candidate-response plot include it; quantitative "
            "baseline, depth, fractional blockade, mean/max amplitude and SNR are "
            "measured on the equivalent chain with high-pass omitted. Because the event "
            "boundaries still come from this filtered trace, start/end, dwell, rise/fall "
            "and ECD integration limits can change. Keep the cutoff well below the event "
            "bandwidth."
        )
        self.hp_enabled.setToolTip(_hp_warning)
        self.hp_cutoff_hz.setToolTip(_hp_warning)
        hp_form.addRow("", self.hp_enabled)
        hp_form.addRow("type", self.hp_kind)
        hp_form.addRow("order", self.hp_order)
        hp_form.addRow("cutoff", self.hp_cutoff_hz)
        self.filter_stack.addWidget(hp_page)

        # -- page: Notch filter ------------------------------------------
        # Zero-phase notch at the mains line and its harmonics. Nearly free for fast
        # (sub-ms) events; distorts slow (~10 ms) events whose own band overlaps the line.
        notch_page = QWidget()
        notch_form = QFormLayout(notch_page)
        self.notch_enabled = _ElidingCheckBox("enable (mains hum + harmonics, zero-phase)")
        self.notch_freq_hz = QDoubleSpinBox()
        self.notch_freq_hz.setDecimals(1)
        self.notch_freq_hz.setRange(*self._STATIC_NOTCH_RANGE)
        self.notch_freq_hz.setSingleStep(10.0)
        self.notch_freq_hz.setValue(60.0)
        self.notch_freq_hz.setSuffix(" Hz")
        self.notch_freq_hz.setKeyboardTracking(False)
        self.notch_freq_hz.setToolTip(
            "Mains line frequency: 60 Hz in the Americas, 50 Hz in most of the world."
        )
        self.notch_harmonics = QSpinBox()
        self.notch_harmonics.setRange(1, 20)
        self.notch_harmonics.setValue(3)
        self.notch_harmonics.setToolTip(
            "How many multiples of the line to notch (60/120/180 …). Harmonics at or "
            "above Nyquist are skipped automatically."
        )
        self.notch_q = QDoubleSpinBox()
        self.notch_q.setDecimals(1)
        self.notch_q.setRange(1.0, 500.0)
        self.notch_q.setSingleStep(5.0)
        self.notch_q.setValue(30.0)
        self.notch_q.setToolTip(
            "Quality factor: higher Q is a narrower notch (Q=30 at 60 Hz is ~2 Hz wide)."
        )
        # -- THE TWO SPELLINGS, and why the panel makes you choose one --------
        # A (fundamental, harmonics) SERIES cannot express a line set with GAPS, and
        # the owner's own Welch censuses found exactly that: host-guest hums at
        # 60/120/180/300/420/540 with 240, 360 and 480 quiet, and the Zn/Cu rig at
        # 60/120/180/240/300/360/420/540/660 with 480 absent. Reaching 540 with
        # ``harmonics=9`` would notch three quiet bands -- three unnecessary bites out
        # of a real signal.
        #
        # The two spellings are EXCLUSIVE, not merged, in the kernel (`filters.notch`
        # ignores freq_hz/harmonics outright once ``freqs`` is given), so the panel
        # makes the exclusivity visible instead of leaving two live-looking controls
        # that only one of which is read: ticking the box disables the series spins.
        self.notch_use_lines = _ElidingCheckBox("use an explicit line list instead")
        self.notch_use_lines.setToolTip(
            "Notch a named set of lines rather than a fundamental and its first N "
            "multiples. This is the only way to express a set with GAPS — e.g. "
            "60, 120, 180, 300, 420, 540 with nothing at 240/360/480.\n"
            "While this is on, the frequency and harmonics above are NOT read at all.")
        #: The last LEGAL contents of the line box — what a refused commit is rolled back
        #: to, so the box never displays a list the note just said may not run (two
        #: surfaces disagreeing) — and that text parsed. ONE writer,
        #: :meth:`_accept_notch_text`, for a commit and for :meth:`set_notch_lines` alike.
        self._notch_ok_text: str = ""
        self._notch_ok_lines: list[float] = []
        #: The standing refusal, shown in orange under the box, and the explicit list
        #: that was IN FORCE when it was raised (None: the series spelling was). ONE
        #: writer, :meth:`_refuse_notch_lines`; cleared only by :meth:`_accept_notch_text`
        #: when the list in force changes -- see that method for why.
        self._notch_refusal: str | None = None
        self._notch_refusal_under: list[float] | None = None
        self.notch_freqs = QLineEdit("")
        self.notch_freqs.setMaxLength(self.NOTCH_BOX_MAX_CHARS)
        self.notch_freqs.setPlaceholderText("60, 120, 180, 300, 420, 540")
        self.notch_freqs.setToolTip(
            "Line frequencies in Hz separated by commas, semicolons, spaces or line "
            "breaks (a spreadsheet column or row pastes as it is), in any order (the "
            "order you type is the order provenance records). Lines at or above Nyquist "
            "are dropped with a warning. Duplicates are refused.\n"
            f"At most {self.NOTCH_MAX_LINES} lines, and every entry must be a positive "
            "number: a longer paste or a mistyped entry is REFUSED when you commit "
            "(Enter, or leaving the box) — the note below names what was refused and "
            "the box goes back to its previous list. Nothing is silently dropped or "
            "truncated.")
        # Owns its height (NOTCH_NOTE_LINES) so a refusal painted by the click that
        # committed the box cannot move the control under that click.
        self.notch_lines_note = _FixedLinesNote(self.NOTCH_NOTE_LINES)
        self.notch_lines_note.setStyleSheet("color:#8b9199;")

        notch_form.addRow("", self.notch_enabled)
        notch_form.addRow("frequency", self.notch_freq_hz)
        notch_form.addRow("harmonics", self.notch_harmonics)
        notch_form.addRow("Q", self.notch_q)
        notch_form.addRow("", self.notch_use_lines)
        notch_form.addRow("lines", self.notch_freqs)
        notch_form.addRow(self.notch_lines_note)
        self.filter_stack.addWidget(notch_page)
        self._notch_form = notch_form

        # -- page: Median filter -----------------------------------------
        # A sliding-median despiker: it deletes single-sample impulses/glitches that a
        # linear smoother would smear, while PRESERVING square edges (the median of a window
        # straddling a step is the step). Applied EARLY in the chain (right after the notch),
        # so spikes are gone before any smoothing spreads them.
        median_page = QWidget()
        median_form = QFormLayout(median_page)
        self.median_enabled = _ElidingCheckBox("enable (sliding median, despikes edges intact)")
        self.median_window_s = QDoubleSpinBox()
        self.median_window_s.setDecimals(5)
        self.median_window_s.setRange(0.00001, 1.0)
        self.median_window_s.setSingleStep(0.00005)
        self.median_window_s.setValue(0.0001)
        self.median_window_s.setSuffix(" s")
        self.median_window_s.setToolTip(
            "Median window in seconds. Rejects spikes up to about half the window wide, so "
            "keep it a few samples — MUCH shorter than the shortest real event, or the "
            "median eats the events along with the spikes. Clamped up to an odd sample count."
        )
        median_form.addRow("", self.median_enabled)
        median_form.addRow("window", self.median_window_s)
        self.filter_stack.addWidget(median_page)

        # -- page: Baseline drift ----------------------------------------
        # A separate PRE-detection transform that flattens slow open-pore wander
        # (electrode/temperature/clogging drift) while preserving event depths. It is
        # applied to the signal BEFORE detect_events; the detector still low-passes.
        detrend_page = QWidget()
        detrend_form = QFormLayout(detrend_page)
        self.detrend_enabled = _ElidingCheckBox("remove baseline wander (flatten slow drift)")
        self.detrend_method = QComboBox()
        self.detrend_method.addItems([label for label, _ in self._DETREND_METHODS])
        self.detrend_window_s = QDoubleSpinBox()
        self.detrend_window_s.setDecimals(3)
        self.detrend_window_s.setRange(0.001, 60.0)
        self.detrend_window_s.setSingleStep(0.1)
        self.detrend_window_s.setValue(0.5)
        self.detrend_window_s.setSuffix(" s")
        self.detrend_window_s.setToolTip(
            "Smoothing window for the open-pore estimate. Must be MUCH longer than the "
            "longest event, or the baseline dives into events and subtracts them "
            "(depths come out shallow, short events vanish)."
        )
        # THE STRIDE THE TWO MEDIAN ESTIMATORS SHARE, and the reason it is not on
        # either sub-page: `masked_median` and `robust_median` both feed the ONE
        # ``detrend_decim`` param key, and one key with two widgets is two values for
        # one number. So it lives on the outer form and its row is hidden for the four
        # methods that do not read it.
        #
        # 0 IS THE DEFAULT AND IT MEANS "AUTO" -- `modules._DECIM_AUTO`, which reaches
        # `estimate_baseline` as ``decim=None`` and lets the kernel derive the stride
        # from the tracking window. That is what keeps both estimators byte-identical
        # to the kernel until the user pins one, and it is the same "0 == the off/auto
        # state" spelling max dwell, merge gap and the low-pass cutoff already use.
        self.detrend_decim = QSpinBox()
        self.detrend_decim.setRange(0, 100000)
        self.detrend_decim.setValue(0)
        self.detrend_decim.setSpecialValueText("auto (from the window)")
        self.detrend_decim.setKeyboardTracking(False)
        self.detrend_decim.setToolTip(
            "Estimate the median on every Nth sample and interpolate back. 0 lets the "
            "kernel derive the stride from the tracking window, which is what it did "
            "before this knob existed. The owner's own recordings pin it at 10.")

        # Each estimator's own SHAPE parameters, on a sub-stack that follows the method
        # combo. They were frozen at estimate_baseline's defaults, which made two of the
        # three methods dead knobs: 'percentile' is documented for >30% occupancy but its
        # percentile could not be moved off 80, and ASLS's smoothness `lam` is measured in
        # BLOCK units — it rescales with the window — so ASLS at a 5 s window silently
        # meant something different than at 0.01 s, with no way to retune. Every default
        # below is exactly the kernel's, so nothing changes until the user moves one.
        # 'block median' genuinely has no shape parameters; it gets a note, not an empty box.
        self.detrend_params_stack = QStackedWidget()

        blockmed_page = QWidget()
        blockmed_form = QFormLayout(blockmed_page)
        blockmed_form.setContentsMargins(0, 0, 0, 0)
        _blockmed_note = QLabel(
            "Block median has no shape parameters — the window above is all it takes.")
        _blockmed_note.setWordWrap(True)
        _blockmed_note.setStyleSheet("color:#8b9199;")
        blockmed_form.addRow(_blockmed_note)
        self.detrend_params_stack.addWidget(blockmed_page)

        pct_page = QWidget()
        pct_form = QFormLayout(pct_page)
        pct_form.setContentsMargins(0, 0, 0, 0)
        self.detrend_percentile = QDoubleSpinBox()
        self.detrend_percentile.setDecimals(1)
        self.detrend_percentile.setRange(0.1, 99.9)
        self.detrend_percentile.setSingleStep(5.0)
        self.detrend_percentile.setValue(80.0)
        self.detrend_percentile.setKeyboardTracking(False)
        self.detrend_percentile.setToolTip(
            "How far toward the OPEN-PORE side to take the level, already sign-corrected "
            "for polarity — 80 means 'the level 80% of the way toward open pore', not a "
            "raw numpy percentile. Raise it when events are dense enough to drag a median "
            "(the case this method exists for); lower it when the trace is mostly open pore."
        )
        pct_form.addRow("percentile", self.detrend_percentile)
        self.detrend_params_stack.addWidget(pct_page)

        asls_page = QWidget()
        asls_form = QFormLayout(asls_page)
        asls_form.setContentsMargins(0, 0, 0, 0)
        self.detrend_asls_lam = QDoubleSpinBox()
        self.detrend_asls_lam.setDecimals(0)
        self.detrend_asls_lam.setRange(1.0, 1e9)
        self.detrend_asls_lam.setSingleStep(1e4)
        self.detrend_asls_lam.setValue(6e4)
        self.detrend_asls_lam.setKeyboardTracking(False)
        self.detrend_asls_lam.setToolTip(
            "Smoothness (Eilers-Boelens lambda). Higher is a stiffer, flatter baseline. "
            "It is measured in BLOCK units, so its meaning rescales with the window and "
            "the knot spacing below — changing the window without revisiting this changes "
            "what ASLS does."
        )
        self.detrend_asls_p = QDoubleSpinBox()
        self.detrend_asls_p.setDecimals(5)
        self.detrend_asls_p.setRange(1e-5, 0.5)
        self.detrend_asls_p.setSingleStep(1e-3)
        self.detrend_asls_p.setValue(1e-3)
        self.detrend_asls_p.setKeyboardTracking(False)
        self.detrend_asls_p.setToolTip(
            "Asymmetry. Small means heavily one-sided, which is what lets the fit ignore "
            "the events and follow the open pore. Raising it toward 0.5 makes the fit "
            "symmetric — it will start tracking events."
        )
        self.detrend_asls_niter = QSpinBox()
        self.detrend_asls_niter.setRange(1, 100)
        self.detrend_asls_niter.setValue(10)
        self.detrend_asls_niter.setKeyboardTracking(False)
        self.detrend_asls_niter.setToolTip(
            "Reweighting iterations. More costs time and converges the asymmetric fit "
            "further onto the open-pore side.")
        self.detrend_asls_decim = QDoubleSpinBox()
        self.detrend_asls_decim.setDecimals(3)
        self.detrend_asls_decim.setRange(0.001, 1.0)
        self.detrend_asls_decim.setSingleStep(0.05)
        self.detrend_asls_decim.setValue(0.1)
        self.detrend_asls_decim.setKeyboardTracking(False)
        self.detrend_asls_decim.setToolTip(
            "Knot spacing as a fraction of the window. Finer than 1 so some knots fall "
            "between dense events; the smoothness comes from lambda, not from this.")
        asls_form.addRow("smoothness (lam)", self.detrend_asls_lam)
        asls_form.addRow("asymmetry (p)", self.detrend_asls_p)
        asls_form.addRow("iterations", self.detrend_asls_niter)
        asls_form.addRow("knot spacing", self.detrend_asls_decim)
        self.detrend_params_stack.addWidget(asls_page)

        # -- event-masked median -------------------------------------------
        # The estimator the host-guest and Zn2 pipelines use. It runs a ROUGH window
        # first to expose the events, masks everything more than `mask_sigma` away
        # from that rough level (dilated by `dilate_ms` so an event's shoulders go
        # too), then re-estimates on the survivors with a SHORT window that can
        # actually track drift. That is why it has two windows and why the single
        # `window` spin above is inert for it.
        masked_page = QWidget()
        masked_form = QFormLayout(masked_page)
        masked_form.setContentsMargins(0, 0, 0, 0)
        self.detrend_rough_s = QDoubleSpinBox()
        self.detrend_rough_s.setDecimals(3)
        self.detrend_rough_s.setRange(0.001, 60.0)
        self.detrend_rough_s.setSingleStep(0.1)
        self.detrend_rough_s.setValue(0.5)
        self.detrend_rough_s.setSuffix(" s")
        self.detrend_rough_s.setKeyboardTracking(False)
        self.detrend_rough_s.setToolTip(
            "ROUGH window: the first, event-exposing pass. Must be MUCH longer than "
            "the longest event so the events stand out against it — this is the pass "
            "whose job is to be wrong inside events.")
        self.detrend_base_s = QDoubleSpinBox()
        self.detrend_base_s.setDecimals(4)
        self.detrend_base_s.setRange(0.0001, 60.0)
        self.detrend_base_s.setSingleStep(0.01)
        self.detrend_base_s.setValue(0.1)
        self.detrend_base_s.setSuffix(" s")
        self.detrend_base_s.setKeyboardTracking(False)
        self.detrend_base_s.setToolTip(
            "TRACKING window: the second pass, run on the masked trace. It may be far "
            "shorter than the rough window (the events are already excluded), which is "
            "what lets this method follow drift the block median cannot.")
        self.detrend_mask_sigma = QDoubleSpinBox()
        self.detrend_mask_sigma.setDecimals(2)
        self.detrend_mask_sigma.setRange(0.5, 20.0)
        self.detrend_mask_sigma.setSingleStep(0.5)
        self.detrend_mask_sigma.setValue(3.5)
        self.detrend_mask_sigma.setKeyboardTracking(False)
        self.detrend_mask_sigma.setToolTip(
            "How far from the rough level a sample must be to be masked out, in noise "
            "sigmas. Lower masks more aggressively (safer against shallow events, but "
            "it starts eating the noise the tracking pass needs).")
        self.detrend_dilate_ms = QDoubleSpinBox()
        self.detrend_dilate_ms.setDecimals(3)
        self.detrend_dilate_ms.setRange(0.0, 1000.0)
        self.detrend_dilate_ms.setSingleStep(0.1)
        self.detrend_dilate_ms.setValue(1.0)
        self.detrend_dilate_ms.setSuffix(" ms")
        self.detrend_dilate_ms.setKeyboardTracking(False)
        self.detrend_dilate_ms.setToolTip(
            "Grow each masked run by this much on both sides, so an event's filtered "
            "shoulders are excluded along with its body. Too small and the tracking "
            "pass is dragged into the edges of every event.")
        masked_form.addRow("rough window", self.detrend_rough_s)
        masked_form.addRow("tracking window", self.detrend_base_s)
        masked_form.addRow("mask threshold", self.detrend_mask_sigma)
        masked_form.addRow("mask dilation", self.detrend_dilate_ms)
        self.detrend_params_stack.addWidget(masked_page)

        # -- robust median --------------------------------------------------
        # One decimated running median over the whole trace. It reads the `window`
        # spin above (unlike masked median) and has only a stride to set.
        robmed_page = QWidget()
        robmed_form = QFormLayout(robmed_page)
        robmed_form.setContentsMargins(0, 0, 0, 0)
        _robmed_note = QLabel(
            "A decimated running median over the whole trace. It uses the window "
            "above, and the decimation above is its only other knob.")
        _robmed_note.setWordWrap(True)
        _robmed_note.setStyleSheet("color:#8b9199;")
        robmed_form.addRow(_robmed_note)
        self.detrend_params_stack.addWidget(robmed_page)

        # -- Savitzky-Golay BASELINE (subtract) ------------------------------
        # NOT the Savitzky-Golay SMOOTHER two pages further down. That one returns the
        # polynomial fit; this one subtracts it. Same kernel, opposite output.
        sgbase_page = QWidget()
        sgbase_form = QFormLayout(sgbase_page)
        sgbase_form.setContentsMargins(0, 0, 0, 0)
        self.detrend_sg_poly = QSpinBox()
        self.detrend_sg_poly.setRange(1, 5)
        self.detrend_sg_poly.setValue(3)
        self.detrend_sg_poly.setKeyboardTracking(False)
        self.detrend_sg_poly.setToolTip(
            "Order of the polynomial fitted in each window to form the BASELINE that "
            "is then subtracted. 3 is what the owner's Cu2-insulin pipeline uses.")
        self.detrend_sg_decim = QSpinBox()
        self.detrend_sg_decim.setRange(1, 10000)
        self.detrend_sg_decim.setValue(50)
        self.detrend_sg_decim.setKeyboardTracking(False)
        self.detrend_sg_decim.setToolTip(
            "Fit the polynomial on every Nth sample and interpolate back. 50 at 50 kHz "
            "turns a 0.2 s window into 200 fitted points — the shape that reproduced "
            "the owner's written cu_* baseline column bit-identically. 1 fits every "
            "sample and is much slower.")
        _sgbase_note = QLabel(
            "SUBTRACTS the Savitzky-Golay fit (it keeps the residual). The "
            "Savitzky-Golay page further down the filter list is the opposite stage — "
            "it KEEPS the fit. The window above is this baseline's window.")
        _sgbase_note.setWordWrap(True)
        _sgbase_note.setStyleSheet("color:#8b9199;")
        sgbase_form.addRow(_sgbase_note)
        sgbase_form.addRow("poly order", self.detrend_sg_poly)
        sgbase_form.addRow("decimation", self.detrend_sg_decim)
        self.detrend_params_stack.addWidget(sgbase_page)

        # The window spin is a genuine NO-OP for the masked estimator, and this label
        # is what says so on screen instead of leaving a dead knob to be turned.
        self.detrend_window_note = QLabel("")
        self.detrend_window_note.setWordWrap(True)
        self.detrend_window_note.setStyleSheet("color:#8b9199;")
        self.detrend_window_note.setVisible(False)

        detrend_form.addRow("", self.detrend_enabled)
        detrend_form.addRow("method", self.detrend_method)
        detrend_form.addRow("window", self.detrend_window_s)
        detrend_form.addRow(self.detrend_window_note)
        detrend_form.addRow("decimation", self.detrend_decim)
        detrend_form.addRow(self.detrend_params_stack)
        self._detrend_form = detrend_form
        self.filter_stack.addWidget(detrend_page)

        # Combo index == sub-stack page index BY CONSTRUCTION: both are built from the
        # same `_DETREND_METHODS` order (which is `baseline.METHODS` order), and the
        # pages above are added in exactly that sequence. `_on_detrend_method_changed`
        # re-derives the page from the METHOD rather than trusting the index, so an
        # inserted kernel method cannot silently raise the wrong page.
        self.detrend_method.currentIndexChanged.connect(self._on_detrend_method_changed)
        self.detrend_params_stack.setCurrentIndex(0)
        self._on_detrend_method_changed()

        # -- page: Running mean ------------------------------------------
        # A centred (zero-phase) box moving average: the plain smoother Savitzky-Golay is
        # contrasted against. It drops white noise by ~sqrt(N) fast and unbiased, but rounds
        # off peaks and smears steps, so keep the window short. Applied on the flattened
        # trace, just before the polynomial smooth.
        rmean_page = QWidget()
        rmean_form = QFormLayout(rmean_page)
        self.rmean_enabled = _ElidingCheckBox("enable (centred moving average, zero-phase)")
        self.rmean_window_s = QDoubleSpinBox()
        self.rmean_window_s.setDecimals(5)
        self.rmean_window_s.setRange(0.00001, 1.0)
        self.rmean_window_s.setSingleStep(0.0001)
        self.rmean_window_s.setValue(0.0005)
        self.rmean_window_s.setSuffix(" s")
        self.rmean_window_s.setToolTip(
            "Averaging window in seconds. Suppresses noise by about sqrt(window·fs), but a "
            "box average rounds peaks and smears steps — keep it SHORTER than the shortest "
            "event you care about. Centred, so it does not shift features in time."
        )
        rmean_form.addRow("", self.rmean_enabled)
        rmean_form.addRow("window", self.rmean_window_s)
        self.filter_stack.addWidget(rmean_page)

        # -- page: Savitzky-Golay smooth ---------------------------------
        # A polynomial-in-a-window smoother that suppresses noise while preserving peak
        # height and edges far better than a moving average. Applied LAST in the chain,
        # on the already-flattened trace.
        savgol_page = QWidget()
        savgol_form = QFormLayout(savgol_page)
        # NAME THE COUNTERPART. There are now two Savitzky-Golay stages in this panel
        # and they are OPPOSITE operations: this one KEEPS the polynomial fit (it
        # smooths), while the "Savitzky-Golay baseline (subtract)" detrend method on the
        # Baseline drift page keeps the RESIDUAL. The combo item above is titled from a
        # list other surfaces pin by text, so the disambiguation lives here and on the
        # detrend page rather than in the item label.
        self.savgol_enabled = _ElidingCheckBox(
            "enable (peak-preserving polynomial smooth — KEEPS the fit)")
        self.savgol_window_s = QDoubleSpinBox()
        self.savgol_window_s.setDecimals(4)
        self.savgol_window_s.setRange(0.0001, 1.0)
        self.savgol_window_s.setSingleStep(0.0005)
        self.savgol_window_s.setValue(0.0010)
        self.savgol_window_s.setSuffix(" s")
        self.savgol_window_s.setToolTip(
            "Smoothing window in seconds. Keep it SHORTER than the shortest event you "
            "care about, or the smoother rounds the event off along with the noise. It is "
            "clamped up to an odd sample count with more points than the polynomial order."
        )
        self.savgol_polyorder = QSpinBox()
        self.savgol_polyorder.setRange(2, 6)
        self.savgol_polyorder.setValue(3)
        self.savgol_polyorder.setToolTip(
            "Order of the polynomial fitted in each window. 2–3 is typical; a higher order "
            "follows sharper curvature but smooths less."
        )
        self.savgol_enabled.setToolTip(
            "Smooths the trace by keeping the Savitzky-Golay fit.\n\n"
            "NOT the same stage as “Savitzky-Golay baseline (subtract)” on the Baseline "
            "drift page: that one SUBTRACTS the same fit to flatten drift. Same kernel, "
            "opposite output — enabling the wrong one is a smooth where you wanted a "
            "detrend.")
        savgol_form.addRow("", self.savgol_enabled)
        savgol_form.addRow("window", self.savgol_window_s)
        savgol_form.addRow("poly order", self.savgol_polyorder)
        self.filter_stack.addWidget(savgol_page)

        # -- page: Normalize ---------------------------------------------
        # THE ONE STAGE IN THIS CHAIN THAT CHANGES THE UNIT, and the whole design of the
        # page follows from that. Everything downstream of detection — depth_pA, the
        # characterisation baseline, current-derived CSV columns, the Event Inspector
        # amplitude readout, and current/depth figure axes — is LABELLED pA, and those
        # labels are written from the channel's units, not measured. A z-score or min-max rescale divides the picoamps out of
        # the trace the detector measures on, so with this stage on, a "depth_pA" of 10
        # means ten sigma (or 0.3 of the recording's full span), not ten picoamps.
        #
        # The two honest options were: (a) keep the stage out of the path that feeds
        # measured depths — which in this app means it does nothing at all, since that
        # path IS the preprocessing chain; or (b) offer it and make the consequence
        # impossible to miss. This is (b), because a control that silently does nothing
        # is its own kind of lie. So the stage is real, it is captured by protocols, and
        # it pays for itself with a warning that is visible from EVERY page of the stack
        # (not just this one) for exactly as long as it is enabled — see
        # `normalize_units_label` and `_refresh_active_filters`. It also defaults OFF and
        # sits LAST in the chain. Last is the only defensible position: the linear stages
        # commute with a rescale so putting it earlier would buy nothing, while the
        # stages that do NOT commute would be actively harmed — ASLS's smoothness lambda
        # and the detrend percentile are quoted against the signal's own scale, so
        # normalising ahead of them silently redefines what those numbers mean.
        normalize_page = QWidget()
        normalize_form = QFormLayout(normalize_page)
        self.normalize_enabled = _ElidingCheckBox("enable (rescale the whole trace — NOT pA)")
        self.normalize_enabled.setToolTip(
            "Rescale the conditioned trace onto a dimensionless scale. Applied LAST, "
            "after every other stage.\n\n"
            "WARNING: this divides the picoamps out. Current-derived amplitudes, "
            "baselines, thresholds and pA/pA·s export fields downstream retain their "
            "legacy labels but are no longer in physical units. Centering also makes "
            "fractional blockade (dI/I0) invalid.")
        # Sentence-length mode labels; capped so they cannot floor the page (_narrow_combo).
        self.normalize_mode = _narrow_combo()
        # Combo data == the kernel's mode strings, taken from
        # ``filters.NORMALIZE_MODES_ALL`` -- the set the TWO kernels between them accept
        # -- so the menu cannot drift from what `modules._apply_normalize` can dispatch
        # to. NORMALIZE_MODES (the narrower tuple) is what `filters.normalize` alone
        # takes, and it still REFUSES 'model_input': that fourth mode is a different
        # function with a different contract, not a fourth branch of the same one.
        _NORMALIZE_MODE_LABELS = {
            "zscore": "z-score  (x − mean) / std",
            "robust": "robust z  (x − median) / 1.4826·MAD",
            "minmax": "min–max  → [−0.5, +0.5]",
            filters.MODEL_INPUT_MODE: "model input  (trimmed σ, clipped) → [−1, +1]",
        }
        for _mode in filters.NORMALIZE_MODES_ALL:
            self.normalize_mode.addItem(_NORMALIZE_MODE_LABELS[_mode], _mode)
        self.normalize_mode.setToolTip(
            "z-score standardises on the WHOLE trace, events included, so a busy "
            "recording inflates the divisor and shrinks every depth in sigma.\n"
            "robust z divides by the same 1.4826·MAD noise sigma the detector "
            "thresholds on, so 1.0 means one baseline-noise sigma.\n"
            "min–max is set by two order statistics — one digitiser glitch rescales "
            "the entire recording; despike first.\n"
            "model input is the normalization the trained student models were fed: a "
            "TWICE-TRIMMED median/MAD (so events are excluded from the divisor), then "
            "clip to ±σ and divide by it — bounded to [−1, +1], with the noise floor "
            "landing at 1/clip in every recording. It REPLACED x/max|x|, whose divisor "
            "was one extreme sample ranging 8.8–42.6 pA across the owner's recordings.")
        self.normalize_clip_sigma = QDoubleSpinBox()
        self.normalize_clip_sigma.setDecimals(1)
        self.normalize_clip_sigma.setRange(0.1, 1000.0)
        self.normalize_clip_sigma.setSingleStep(5.0)
        self.normalize_clip_sigma.setValue(float(filters.MODEL_INPUT_CLIP_SIGMA))
        self.normalize_clip_sigma.setKeyboardTracking(False)
        self.normalize_clip_sigma.setToolTip(
            "BOTH the clip bound and the divisor — the two cannot be set apart, and "
            "that is deliberate: clipping at ±30σ and then dividing by 30 is what puts "
            "the noise floor at exactly 1/30 in every recording, which is the one "
            "number the models were trained against. Anything deeper than this many "
            "sigmas is SATURATED, so depth stops being monotone in the original pA "
            "above it. 30 is the notebooks' value; only change it to match a model "
            "that was trained differently.")
        normalize_form.addRow("", self.normalize_enabled)
        normalize_form.addRow("mode", self.normalize_mode)
        normalize_form.addRow("clip (σ)", self.normalize_clip_sigma)
        # The page's own restatement of the cost, always visible (not only when on) so
        # someone reading the menu before ticking it learns what it does first.
        normalize_note = QLabel(
            "Current-derived amplitudes, baselines and pA/pA·s export fields downstream "
            "of detection retain labels written from the channel, not from the transformed "
            "data. This stage removes the picoamps, and nothing downstream can put them "
            "back — a "
            "depth reads in sigma (z-score / robust) or in fractions of the trace's "
            "full span (min–max); fractional blockade (dI/I0) is invalid after centering. "
            "Use it to feed a model or to compare recordings on a "
            "common scale, NOT to measure a conductance.")
        normalize_note.setWordWrap(True)
        normalize_note.setStyleSheet("color:#FFB74D;")
        normalize_form.addRow(normalize_note)
        self.filter_stack.addWidget(normalize_page)

        # -- frequency response of the SELECTED stage ---------------------
        # Choosing a filter parameter used to be blind: the panel showed a cutoff and a
        # rise time and nothing else, so the two things that actually bite — that
        # zero-phase filtering makes the nominal cutoff a -6 dB point, and that the
        # window smoothers band-limit the trace far harder than the low-pass does — were
        # invisible. This draws |H(f)| for the CURRENT parameters of the stage on screen,
        # as it is really applied (squared for the two-pass IIR kinds), with the nominal
        # cutoff marked so the gap between the label and the reality is on the plot.
        # The curve comes from filters.response(); this widget only draws it.
        self.response_plot = pg.PlotWidget()
        self.response_plot.setMinimumHeight(150)
        self.response_plot.setMaximumHeight(190)
        self.response_plot.setLogMode(x=True, y=False)
        self.response_plot.setLabel("bottom", "frequency", units="Hz")
        self.response_plot.setLabel("left", "|H|", units="dB")
        self.response_plot.showGrid(x=True, y=True, alpha=0.25)
        self.response_plot.setMenuEnabled(False)
        self.response_plot.setMouseEnabled(x=False, y=False)
        self.response_plot.hideButtons()
        self.response_curve = self.response_plot.plot(pen=pg.mkPen("#56B4E9", width=2))
        # Nominal cutoff (vertical) and the -3 dB line the label implies (horizontal).
        # Together they show at a glance that the applied curve is already through -3 dB
        # well before the number printed on the control.
        self.response_cutoff_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("#E69F00", width=1, style=Qt.DashLine))
        self.response_3db_line = pg.InfiniteLine(
            angle=0, pos=-3.0, pen=pg.mkPen("#8b9199", width=1, style=Qt.DotLine))
        self.response_plot.addItem(self.response_cutoff_line)
        self.response_plot.addItem(self.response_3db_line)

        # The one-line verdict under the plot: what the applied response really does at
        # the nominal cutoff, and where its true -3 dB corner sits. For a stage with no
        # transfer function (the median) this carries the explanation instead.
        self.response_note = QLabel("")
        self.response_note.setWordWrap(True)
        self.response_note.setStyleSheet("color:#8b9199;")

        self.response_chain_check = _ElidingCheckBox(
            "show combined response of enabled linear stages")
        self.response_chain_check.setToolTip(
            "Multiply every enabled linear stage on one shared frequency grid. "
            "Median, baseline estimation and normalization are signal-dependent or "
            "nonlinear; the note names them instead of fabricating a transfer function. "
            "When high-pass is enabled, this is explicitly the candidate-localization "
            "response; quantitative measurement omits high-pass.")

        # The "show processed trace" preview reflects the WHOLE chain (every enabled
        # stage, then the low-pass), not one filter — so it lives at the group level,
        # directly below the execution summary rather than on any single stage page.
        self.filter_preview = _ElidingCheckBox("show processed trace on the plot")
        self.filter_preview.setToolTip(
            "Show the full candidate-localization chain as the orange trace. When "
            "high-pass is enabled, the orange trace includes it; classical detector "
            "event-table amplitudes are measured on the equivalent chain without "
            "high-pass. Manual measurement of ‘Processed signal’ follows the orange "
            "trace, including high-pass.")
        self.filter_preview.setEnabled(False)   # only meaningful once a preprocessing stage is on

        # When the chain raises, the preview curve simply disappeared and the checkbox
        # stayed ticked — an over-Nyquist cutoff, a detrend window longer than the trace
        # and a segment too short for the filter's padding all looked identical (and like
        # nothing). The kernel's own message goes here instead.
        self.preview_error_label = QLabel("")
        self.preview_error_label.setWordWrap(True)
        self.preview_error_label.setStyleSheet("color:#FFB74D;")
        self.preview_error_label.setVisible(False)

        # -- the compact Signal chain card, used while the editor is HOSTED in the Signal
        # Conditioning window. Every row owns a fixed height, so no state moves anything.
        self.open_conditioning_button = QPushButton("Open Signal Conditioning…")
        self.open_conditioning_button.setObjectName("openSignalConditioningButton")
        self.open_conditioning_button.setToolTip(
            "Edit the classical detectors' signal chain in execution order, with its "
            "frequency response, protocols and Copy as Python (Ctrl+Shift+F).")
        self.open_conditioning_button.clicked.connect(
            lambda: self.openSignalConditioningRequested.emit(self.current_stage_key()))
        self.open_conditioning_button.setVisible(False)
        # Unselectable notes: a selectable QLabel carries a text control and document,
        # and every QObject here is one more wrapper per MainWindow teardown.
        self.chain_status_note = _FixedLinesNote(2, selectable=False)
        self.chain_status_note.setObjectName("chainStatusNote")
        self.chain_warning_note = _FixedLinesNote(2, selectable=False)
        self.chain_warning_note.setObjectName("chainWarningNote")
        self.chain_warning_note.setStyleSheet("color:#FFB74D; font-weight:bold;")
        self.chain_protocol_note = _ElidingCaption(elide=Qt.ElideRight)
        self.chain_protocol_note.setObjectName("chainProtocolNote")
        self.chain_scope_note = _ElidingCaption(SCOPE_SENTENCE, elide=Qt.ElideRight)
        self.chain_scope_note.setObjectName("chainScopeNote")
        self.chain_scope_note.setStyleSheet("color:#8b9199;")
        # The card's copy of active_filters_label (which stays in the in-page editor).
        self.chain_order_note = _FixedLinesNote(3, selectable=False)
        self.chain_order_note.setObjectName("chainOrderNote")
        self.chain_order_note.setStyleSheet("color:#8b9199;")
        self.signal_chain_summary = QWidget()
        self.signal_chain_summary.setObjectName("signalChainSummary")
        _card = QVBoxLayout(self.signal_chain_summary)
        _card.setContentsMargins(0, 0, 0, 0)
        _card.setSpacing(4)
        for _note in (self.chain_status_note, self.chain_warning_note,
                      self.chain_protocol_note, self.chain_scope_note,
                      self.chain_order_note):
            _card.addWidget(_note)
        self.signal_chain_summary.setVisible(False)

        # Combo order == page order; default selection is Low-pass (index 0). Each label
        # is paired with its enable checkbox so the active-filters indicator can mark and
        # summarise the enabled stages.
        self._filter_specs: tuple[tuple[str, QCheckBox], ...] = (
            ("Low-pass", self.filter_enabled),
            ("High-pass", self.hp_enabled),
            ("Notch", self.notch_enabled),
            ("Median", self.median_enabled),
            ("Baseline drift", self.detrend_enabled),
            ("Running mean", self.rmean_enabled),
            ("Savitzky-Golay", self.savgol_enabled),
            ("Normalize", self.normalize_enabled),
        )
        self.filter_selector.addItems([label for label, _ in self._filter_specs])

        # The unit banner for the Normalize stage. It lives at the GROUP level, beside
        # the execution summary, rather than on the Normalize page: the page is hidden
        # the moment the user pages away, and "the trace the detector measures is no
        # longer in pA" is not a fact that may go off screen while it is true. Hidden
        # whenever the stage is off, so it costs nothing in the ordinary case.
        self.normalize_units_label = QLabel("")
        self.normalize_units_label.setWordWrap(True)
        self.normalize_units_label.setStyleSheet(
            "color:#FFB74D; font-weight:bold;")
        self.normalize_units_label.setVisible(False)

        self.highpass_measurement_label = QLabel("")
        self.highpass_measurement_label.setWordWrap(True)
        self.highpass_measurement_label.setStyleSheet(
            "color:#FFB74D; font-weight:bold;")
        self.highpass_measurement_label.setVisible(False)

        # THE REFUSAL BANNER, and it lives here for the same reason the unit banner
        # does. `set_detrend_method` refuses a method this combo does not offer instead
        # of silently leaving a different estimator selected; the user who triggered
        # that was recalling a protocol or reopening a project, not reading the Baseline
        # drift page, so a note on that page would never be seen. It sits at the group
        # level and stays up until the next successful method change.
        self.detrend_refusal_label = QLabel("")
        self.detrend_refusal_label.setWordWrap(True)
        self.detrend_refusal_label.setStyleSheet("color:#EF5350; font-weight:bold;")
        self.detrend_refusal_label.setVisible(False)

        # -- named protocols (owner request): save the WHOLE chain under a name -------
        # A protocol is every preprocessing stage's enabled-state + parameters,
        # captured with params()/set_params() — the same round-trip a project restore
        # uses, so a recalled protocol reproduces the chain exactly. Stored in
        # QSettings as JSON, so protocols survive restarts and travel per user, not
        # per project. Editing any stage afterwards drops the combo back to
        # "— manual —": a protocol name over a hand-edited chain would be a lie.
        proto_row = QHBoxLayout()
        self.protocol_combo = QComboBox()
        self.protocol_combo.setToolTip(
            "Your saved preprocessing protocols. Picking one applies every stage "
            "(enabled flags and parameters) at once; '— manual —' means the chain is "
            "whatever the pages below say.")
        self.protocol_save_btn = QPushButton("Save as…")
        self.protocol_save_btn.setToolTip(
            "Save the current chain — every stage's enable flag and parameters — "
            "under a name, for reuse on any recording.")
        # Rename is not a convenience. Without it the only way to fix a typo in a
        # protocol's name was Save-as under the new name and Delete the old one — which
        # silently re-captures the chain that happens to be on screen NOW, not the one
        # that was saved. Renaming moves the stored settings untouched.
        self.protocol_rename_btn = QPushButton("Rename")
        self.protocol_rename_btn.setToolTip(
            "Rename the selected protocol. The saved settings are moved unchanged — "
            "unlike Save-as, which captures whatever the pages currently say.")
        self.protocol_delete_btn = QPushButton("Delete")
        self.protocol_delete_btn.setToolTip("Forget the selected protocol.")
        # TWO ROWS, NOT ONE. All five widgets used to share a single QHBoxLayout, with the
        # combo taking the stretch and the three verbs pinned to the right end of it. The
        # row is laid out across the scroll area's CONTENT width, which another page of this
        # panel had pushed to 836 px against a 420 px viewport, so the buttons landed at
        # x=561/647/733 — every one of them off-screen, with no menu item and no shortcut to
        # reach them by. Splitting the verbs onto their own left-aligned row makes their
        # position independent of the panel's width: 3x80 + 2x6 = 252 px, always visible.
        # It is a static reflow — the rows do not rearrange with state or with dock width,
        # so nothing ever moves out from under a pointer that is aiming at it.
        self._protocol_label = QLabel("Protocol:")
        proto_row.addWidget(self._protocol_label)
        proto_row.addWidget(self.protocol_combo, 1)
        proto_btn_row = QHBoxLayout()
        proto_btn_row.addWidget(self.protocol_save_btn)
        proto_btn_row.addWidget(self.protocol_rename_btn)
        proto_btn_row.addWidget(self.protocol_delete_btn)
        proto_btn_row.addStretch(1)
        self._protocol_updating = False
        self._load_protocols_into_combo()
        self.protocol_combo.currentTextChanged.connect(self._on_protocol_selected)
        self.protocol_save_btn.clicked.connect(self._save_protocol)
        self.protocol_rename_btn.clicked.connect(self._rename_protocol)
        self.protocol_delete_btn.clicked.connect(self._delete_protocol)
        self.paramsChanged.connect(self._protocol_drop_to_manual)
        self.protocol_combo.currentTextChanged.connect(self._refresh_chain_card)

        # The stage selector is the head of this group, and the response hint rides its
        # row instead of spending another vertical row. The execution summary and
        # whole-chain preview follow immediately; protocol management and the selected
        # stage's controls come afterwards.
        stage_row = QHBoxLayout()
        stage_row.setContentsMargins(0, 0, 0, 0)
        stage_row.addWidget(self.filter_selector, 0)
        stage_row.addWidget(self.response_hint, 1)
        filters_layout.addLayout(stage_row)
        filters_layout.addWidget(self.active_filters_label)
        # This checkbox previews the WHOLE chain, so keep it beside the execution
        # summary rather than below the selected-stage editor and response graph.
        filters_layout.addWidget(self.filter_preview)
        filters_layout.addWidget(self.preview_error_label)
        filters_layout.addLayout(proto_row)
        filters_layout.addLayout(proto_btn_row)
        filters_layout.addWidget(self.highpass_measurement_label)
        filters_layout.addWidget(self.normalize_units_label)
        filters_layout.addWidget(self.detrend_refusal_label)
        filters_layout.addWidget(self.filter_stack)
        filters_layout.addWidget(self.response_chain_check)
        filters_layout.addWidget(self.response_plot)
        filters_layout.addWidget(self.response_note)
        signal_chain_layout.addWidget(self.signal_chain_summary)
        signal_chain_layout.addWidget(filters_box)
        signal_chain_layout.addStretch(1)
        self._signal_filters_box = filters_box
        self._filters_layout = filters_layout

        # Selecting a filter in the combo raises its page (index-based on purpose:
        # rewriting an item's TEXT for the ● marker fires currentTextChanged, NOT
        # currentIndexChanged, so the marker refresh never switches pages underfoot).
        # Through `_raise_filter_page`, NOT straight into `filter_stack.setCurrentIndex`:
        # the bare connection let Qt move focus onto the incoming page and the enclosing
        # QScrollArea followed it, scrolling this whole column 334 px out from under the
        # pointer. See that method for the measurement.
        self.filter_selector.currentIndexChanged.connect(self._raise_filter_page)
        self.filter_selector.currentIndexChanged.connect(self._update_response_plot)
        self.response_chain_check.toggled.connect(self._update_response_plot)
        self.filter_selector.setCurrentIndex(0)
        self.filter_stack.setCurrentIndex(0)

        # -- Detection thresholds ----------------------------------------
        det_box = QGroupBox("Detection")
        det_form = QFormLayout(det_box)

        # -- WHICH detector, and why it is a combo rather than a fourth tab ----
        # Both detectors run on the SAME conditioned trace (the Preprocess tab's
        # chain), resolve the SAME characterisation baseline, and return the same
        # 20-column table. What differs is one small group of thresholds. A tab per
        # detector would duplicate polarity, dwell limits, merge gap, the low-pass and
        # the range note four times over and invite them to drift apart; a combo with a
        # swapping threshold sub-stack keeps one copy of everything they share.
        #
        # The ML detector keeps its own tab because it does NOT share that chain: it
        # runs its own notch/baseline/normalize (`ml_preprocess`), loads a model, and
        # has a range selector these two do not.
        self.detector_combo = QComboBox()
        for _label, _mid in self._DETECTORS:
            self.detector_combo.addItem(_label, _mid)
        self.detector_combo.setToolTip(
            "Schmitt: a two-threshold hysteresis trigger — an event starts when the "
            "deviation crosses k_in noise sigmas and ends when it falls back through "
            "k_out. Fast, and the app's long-standing default.\n\n"
            "CUSUM: a one-sided sequential test — it accumulates evidence sample by "
            "sample and fires when the running sum crosses h, then closes the event "
            "with a second CUSUM referenced to the level it just measured. It finds "
            "shallow events a fixed threshold misses, and its onset estimate is the "
            "changepoint rather than the crossing sample.\n\n"
            "Both run on the same preprocessed trace and the same baseline, so their "
            "depths are directly comparable.")
        self.detector_note = QLabel("")
        self.detector_note.setWordWrap(True)
        self.detector_note.setStyleSheet("color:#8b9199;")

        # The two detectors' PRIVATE thresholds, on a sub-stack. Everything below the
        # stack (dwell limits, merge gap, range, polarity) is shared and appears once.
        self.detector_stack = _CurrentPageStackedWidget()

        schmitt_page = QWidget()
        schmitt_form = QFormLayout(schmitt_page)
        schmitt_form.setContentsMargins(0, 0, 0, 0)
        self.k_in = QDoubleSpinBox()
        self.k_in.setDecimals(2)
        self.k_in.setRange(0.5, 50.0)
        self.k_in.setSingleStep(0.5)
        self.k_in.setValue(5.0)
        self.k_out = QDoubleSpinBox()
        self.k_out.setDecimals(2)
        self.k_out.setRange(0.1, 49.5)
        self.k_out.setSingleStep(0.5)
        self.k_out.setValue(2.0)
        self.min_dwell_s = QDoubleSpinBox()
        self.min_dwell_s.setDecimals(7)
        self.min_dwell_s.setRange(0.0, 1.0)
        self.min_dwell_s.setSingleStep(1e-5)
        self.min_dwell_s.setValue(2e-5)
        self.min_dwell_s.setSuffix(" s")
        self.max_dwell_s = QDoubleSpinBox()
        self.max_dwell_s.setDecimals(6)
        self.max_dwell_s.setRange(0.0, 100.0)
        self.max_dwell_s.setSingleStep(1e-3)
        self.max_dwell_s.setValue(0.0)
        self.max_dwell_s.setSuffix(" s  (0 = off)")
        self.max_dwell_s.setToolTip(
            "Events longer than this are KEPT but marked rejected (a clog is real signal "
            "you must see, not something to disappear).\n"
            "It also truncates the dwell sample on the right: the fits read this limit from "
            "provenance and condition the likelihood on it, because a left-truncation-only "
            "fit of a right-cut sample biases tau SHORT (-24% when the limit sits at 2.5 tau)."
        )
        # -- merge gap: MILLISECONDS, like the Merge panel ------------------
        #
        # This spin used to read SECONDS while :class:`~nanoscope.gui.merge_panel.MergePanel`
        # read MILLISECONDS, so the same physical window was two different numbers on two
        # panels (15 ms was "15" there and "0.015" here). That unit split is exactly what
        # already cost a real run: the first hands-on replication of the tutorial typed 15
        # into a µs-era field meaning 15 ms and merged at 15 µs — a 1000x silent under-merge.
        # The fix was applied to the Merge panel only, so half the confusion survived. Both
        # merge windows now speak milliseconds, the unit this bench's fragment gaps actually
        # live in (the walkthroughs merge at 10-15 ms).
        #
        # Range/decimals/step are the OLD seconds values scaled by 1000, so nothing a saved
        # project could hold got narrower: 0-1000 ms == the old 0-1 s, 4 decimals == the old
        # 7 (both resolve 1e-7 s), 0.01 ms step == the old 1e-5 s step.
        #
        # WHY 0 STILL MEANS "OFF" HERE AND NOT ON THE MERGE PANEL. That asymmetry is
        # deliberate, not a leftover. Merging is one optional STEP inside a detection run,
        # and params() genuinely maps 0 -> None, the detector's documented off-sentinel: a
        # run with no merge stage at all is a real, common thing to ask for. The Merge panel
        # has no such state to express — its spin sits next to a button whose only meaning is
        # "merge", so a window of "don't" would be a click that does nothing, and it floors
        # at MIN_WINDOW_MS instead.
        self.merge_gap_ms = QDoubleSpinBox()
        self.merge_gap_ms.setDecimals(4)
        self.merge_gap_ms.setRange(0.0, 1000.0)
        self.merge_gap_ms.setSingleStep(0.01)
        self.merge_gap_ms.setValue(0.0)
        self.merge_gap_ms.setSuffix(" ms  (0 = off)")
        self.merge_gap_ms.setToolTip(
            "Merge gap: while detecting, events separated by less than this gap "
            "(next start − current end) are stitched into one event, transitively along "
            "a chain. Same rule and same unit as the Merge panel's window.\n"
            "0 turns the merge stage off entirely — unlike the Merge panel, which always "
            "merges when you press its button and so has no 'off'.\n"
            "A run whose merged span would be mostly open pore is REFUSED rather than "
            "merged: 40 clean 0.4 ms events 8 ms apart are not one 312 ms translocation, "
            "so they are left as they are and the provenance records how many were skipped."
        )
        schmitt_form.addRow("k_in (enter)", self.k_in)
        schmitt_form.addRow("k_out (exit)", self.k_out)
        self.detector_stack.addWidget(schmitt_page)

        # -- CUSUM page ----------------------------------------------------
        # BOTH NUMBERS ARE DIMENSIONLESS, and the labels say so rather than leaving a
        # bare "h = 28" to be read as picoamps. They are in units of the MAD noise
        # sigma the detector measures on the baseline-subtracted residual — the SAME
        # sigma the table's snr column is divided by — so "k = 0.75" is 0.75 of the
        # noise floor whatever the recording's gain.
        cusum_page = QWidget()
        cusum_form = QFormLayout(cusum_page)
        cusum_form.setContentsMargins(0, 0, 0, 0)
        self.cusum_k = QDoubleSpinBox()
        self.cusum_k.setDecimals(3)
        self.cusum_k.setRange(0.001, 50.0)
        self.cusum_k.setSingleStep(0.25)
        self.cusum_k.setValue(float(DEFAULT_CUSUM_K))
        self.cusum_k.setSuffix(" σ")
        self.cusum_k.setKeyboardTracking(False)
        self.cusum_k.setToolTip(
            "Drift (slack), in NOISE SIGMAS — not picoamps. The statistic grows by "
            "(deviation − k) per sample, so a blockade shallower than k can never "
            "accumulate and is invisible to this detector at any duration. Raising it "
            "is how you buy quiet; lowering it is how you reach shallow events.\n"
            "0.75 is the calibrated value (exact labels, apo-ferritin at −600 mV, "
            "480 events: precision 0.854, recall 0.993, F1 0.918).")
        self.cusum_h = QDoubleSpinBox()
        self.cusum_h.setDecimals(2)
        self.cusum_h.setRange(0.01, 500.0)
        self.cusum_h.setSingleStep(1.0)
        self.cusum_h.setValue(float(DEFAULT_CUSUM_H))
        self.cusum_h.setSuffix(" σ·samples")
        self.cusum_h.setKeyboardTracking(False)
        self.cusum_h.setToolTip(
            "Decision threshold: how much accumulated evidence fires an event. Also "
            "DIMENSIONLESS — it is a sum of (deviation − k) over samples, so it trades "
            "sensitivity against detection delay: a blockade of depth d is detected "
            "about h/(d − k) samples after it starts. The reported onset is still the "
            "changepoint estimate, not the crossing sample, so the delay does not bias "
            "the dwell.\n"
            "28 is the calibrated value. It is NOT set by an average-run-length "
            "design — there is none anywhere in the source — it was grid-searched for "
            "event F1 at IoU ≥ 0.30.")
        _cusum_note = QLabel(
            "Calibrated on apo-ferritin at −600 mV against EXACT labels and "
            "transferred unchanged to −500/−700 mV. There are NO per-project values "
            "for the host-guest, Zn2 or Cu2 recordings — those pipelines never ran "
            "this detector — so treat 0.75 / 28 as a starting point there, not a "
            "contract. The merge gap and minimum dwell below default to this "
            "detector's own 0.50 ms / 1.00 ms rather than the Schmitt values.")
        _cusum_note.setWordWrap(True)
        _cusum_note.setStyleSheet("color:#8b9199;")
        cusum_form.addRow("k (drift)", self.cusum_k)
        cusum_form.addRow("h (threshold)", self.cusum_h)
        cusum_form.addRow(_cusum_note)
        self.detector_stack.addWidget(cusum_page)

        det_form.addRow("detector", self.detector_combo)
        det_form.addRow(self.detector_note)
        det_form.addRow(self.detector_stack)
        det_form.addRow("min dwell", self.min_dwell_s)
        det_form.addRow("max dwell", self.max_dwell_s)
        det_form.addRow("merge gap", self.merge_gap_ms)

        # -- range: the ASYMMETRY WITH THE ML TAB, said out loud ------------
        #
        # The ML tab has a "range" combo (Visible view / Whole recording) and this one
        # has nothing. Whole-recording is the right default here — the Schmitt detector
        # is ~1 s per million samples against the ML detector's ~0.24x realtime, so
        # there is no cost to pay for it — and the stored provenance is honest about
        # what was analysed. But an unexplained absence reads as an oversight, and a
        # user who genuinely wants a zoomed region was left with no answer at all.
        #
        # A range flag is NOT the answer for this detector, and that is the sentence
        # that was missing: the open-pore baseline windows are sized against the trace
        # they are measured on, so detecting a 2 s excerpt with the whole recording's
        # windows measures those depths against a DIFFERENT open pore than the table
        # says. Cutting a Segment gives the excerpt its own document, its own probe and
        # its own fingerprint — which is the thing the user actually wanted.
        self.range_readout = QLabel("whole recording")
        self.range_readout.setWordWrap(True)
        self.range_readout.setToolTip(
            "The Schmitt detector always runs over the WHOLE recording, and the "
            "detection's provenance records that.\n\n"
            "It is fast enough that there is nothing to save by running it over less "
            "(the ML tab offers a range because inference is ~0.24x realtime).\n\n"
            "To detect over a zoomed region, cut it out first: turn on the Measure "
            "tool, drag over the region, then Segment in the Files dock. The segment "
            "is its own recording with its own baseline probe and its own "
            "baseline_fingerprint — which is what makes its depths meaningful. "
            "Detecting a sub-range under the parent's baseline windows would measure "
            "it against a different open pore than the one it reports.")
        det_form.addRow("range", self.range_readout)
        detection_layout.addWidget(det_box)

        # -- Polarity -----------------------------------------------------
        pol_box = QGroupBox("Polarity")
        pol_form = QFormLayout(pol_box)
        self.polarity_readout = QLabel("not yet measured")
        self.polarity_readout.setWordWrap(True)
        self.polarity_override = QComboBox()
        # 'auto' -> measure it from the data; the others force it.
        self.polarity_override.addItems(
            ["auto (measure)", "blockade (down, -1)", "enhancement (up, +1)"]
        )
        pol_form.addRow("measured", self.polarity_readout)
        pol_form.addRow("override", self.polarity_override)
        detection_layout.addWidget(pol_box)

        # -- Buttons ------------------------------------------------------
        # A 2x2 GRID, NOT A FOUR-WIDE ROW. Four QPushButtons side by side demanded
        # 80+126+96+111 + 3x6 = 431 px as a hard floor (a push button has no shrink flag),
        # and when the row was stretched across this panel's 836 px content width the last
        # two landed at x=421..620 and x=626..825 — outside the 420 px viewport by a single
        # pixel and then some. Neither 'Null control…' nor 'Recovery map…' has a menu item
        # or a shortcut, and StatsPanel's own 'false-positive rate: not measured (run Null
        # control)' points at a button that could not be seen. Two rows of two floor at
        # 2x126 + 6 = 258 px. The grid is FIXED at two columns on purpose: a flow layout
        # that reflowed between one and two columns as the dock is dragged would move a
        # button out from under the pointer aiming at it.
        btn_row = QGridLayout()
        self.detect_button = QPushButton("Detect")
        self.detect_button.setDefault(True)
        self.sweep_button = QPushButton("Sweep threshold…")
        self.null_button = QPushButton("Null control…")
        self.recovery_button = QPushButton("Recovery map…")
        btn_row.addWidget(self.detect_button, 0, 0)
        btn_row.addWidget(self.sweep_button, 0, 1)
        btn_row.addWidget(self.null_button, 1, 0)
        btn_row.addWidget(self.recovery_button, 1, 1)
        btn_row.setColumnStretch(0, 1)
        btn_row.setColumnStretch(1, 1)
        # ABOVE THE PARAMETERS, NOT BELOW THEM — the page's verbs come first.
        #
        # Fixing x left y wrong. With the grid at the END of this page the four buttons sat
        # at content y=507..563 inside a viewport that shows y=0..466 at 1512x900 (and
        # y=0..405 at 1280x800): ``visibleRegion()`` empty for all four, a click at their
        # centre landing on the panel below, and the app's primary verb simply not on the
        # page a user arrives at. ``Sweep threshold…``, ``Null control…`` and
        # ``Recovery map…`` have no menu item and no shortcut, so scrolling was their only
        # door, while StatsPanel prints "false-positive rate: not measured (run Null
        # control)" and points at one of them.
        #
        # WHY MOVING THEM UP IS THE FIX AND NOT A NUDGE. The controls column is ~2.3k px of
        # content in a ~470 px viewport: scrolling is inevitable, so the question is only
        # WHAT is allowed to be below the fold. Parameters are — you go looking for the one
        # you want to change — and the actions that consume them are not. Putting the grid
        # first also makes its position CONSTANT: the only things above it are the tab bar
        # and the gate banner, whose row is now reserved (see ``_gate_host``), so no state
        # change moves these four buttons at all. Insert rather than re-order the source so
        # the creation order — and therefore the tab-focus order — is byte-for-byte as it
        # was; this is placement only, no signal, no default, no enable rule changes.
        detection_layout.insertLayout(0, btn_row)
        detection_layout.addStretch(1)

        # -- ML detector --------------------------------------------------
        # A parallel detector: a runtime-loaded Keras model run on the current
        # channel. Its events flow into the SAME pipeline as the Schmitt ones (event
        # table, span overlays, inspector, plots, exports). Inference is slow, so it
        # runs off-thread with progress + cancel and defaults to the visible view.
        ml_box = QGroupBox("ML detector")
        ml_form = QFormLayout(ml_box)

        model_row = QHBoxLayout()
        self.ml_load_button = QPushButton("Load model…")
        self.ml_load_button.setToolTip(
            "Load a .keras event-detector model. It is run on the current channel; "
            "the model is loaded off the GUI thread when you run detection.")
        self.ml_model_label = QLabel("none")
        self.ml_model_label.setStyleSheet("color:#8b9199;")
        self.ml_model_label.setWordWrap(True)
        model_row.addWidget(self.ml_load_button)
        model_row.addWidget(self.ml_model_label, 1)
        ml_form.addRow("model", model_row)

        # Every model loaded THIS SESSION, switchable without re-browsing (owner note
        # #7: comparing two students meant a file dialog per switch). The combo holds
        # basenames with the full path as item data — set_model_path is the single
        # writer, so the browse dialog, the menu action and a project restore all land
        # here. Session-scoped on purpose: a persisted list would accumulate dead paths.
        self.ml_model_combo = QComboBox()
        self.ml_model_combo.setToolTip(
            "Switch between the models loaded this session. Loading a model adds it "
            "here; picking one makes it the active detector.")
        self.ml_model_combo.setEnabled(False)
        self.ml_model_combo.addItem(self._ML_NO_MODEL_ITEM)
        self._ml_combo_updating = False
        self.ml_model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        ml_form.addRow("loaded", self.ml_model_combo)

        self.ml_range = QComboBox()
        self.ml_range.addItems(["Visible view", "Whole recording"])
        self.ml_range.setToolTip(
            "Where to run the model. Inference is slow (~0.24x realtime on CPU), so "
            "the visible view is the default; whole recording can take minutes.\n\n"
            "The normalization scale is a SEPARATE choice (see Model preprocessing): "
            "with scope 'whole recording' the model input is scaled the way training "
            "scaled it, even when you only infer over the visible window.")
        ml_form.addRow("range", self.ml_range)

        self.ml_detect_button = QPushButton("Run ML detection")
        ml_form.addRow(self.ml_detect_button)

        # THE SECOND GATE BANNER, AND ITS ROW IS RESERVED TOO. The panel-wide Schmitt gate
        # above (``_gate_host``) was given a fixed row last round; this sibling — same
        # panel, same amber, same show/hide-with-state — was not, so the anti-pattern
        # simply moved one widget along. Measured on a real MainWindow at 1512x900, ML
        # detector page, BEFORE the row was reserved: the whole "Model preprocessing" group
        # — preset combo, its Save/Delete verbs, the baseline rows — sat at content
        # y=475..500 under the one-line gate ("Load a .keras model to run ML detection."),
        # at 490..515 under the two-line one ("Detection runs on CURRENT channels; this is
        # a voltage channel …"), and at 454..479 with the banner hidden (TensorFlow present
        # AND a model loaded). A 36 px range on a 25 px button. Those controls only became
        # reachable at all this round, which is what turns that jump into a lost click.
        #
        # WHAT MOVES THE BANNER BETWEEN THOSE THREE IS NOT A CHANNEL CHANGE — an earlier
        # version of this comment said it was, and a skeptic measured otherwise. Selecting
        # a different channel does not repoint this panel: ``AnalysisPanel.set_selection``
        # sticks to the recording's current channel through
        # ``Session.resolve_current_channel`` precisely so that clicking the command
        # voltage to calibrate it cannot grey out detection. Driven on the apo ABF (IN13
        # raw_voltage, IN14 voltage, IN15 current), selecting each of the three in turn
        # left the target on IN15 and the gate reading "Load a .keras model to run ML
        # detection." all three times, unchanged.
        # The doors that DO swap these strings are: loading or clearing a model; a
        # Calibration-panel "treat this channel as" override on the target itself, which
        # is what produces the long refusal (measured: marking IN15 voltage gives
        # "…this is a voltage channel (your Calibration-panel override; the file says
        # current)." and clearing it returns the one-line gate); a recording with no live
        # current channel or no known sample rate; and TensorFlow's absence, which speaks
        # over all of them (``_refresh_ml_enabled`` names the FIRST unmet condition, so the
        # channel refusal outranks "load a model" — the two never show at once).
        #
        # Reserved EXACTLY the way the panel-wide banner is, for the reasons written there:
        #   * ``_FixedLinesNote(2)`` owns two lines whether it speaks or not, so the long
        #     wrong-channel refusal cannot grow the row past the short "load a model" one
        #     either (over-long text is painted from the head with "…" and kept whole in
        #     ``text()`` and the tooltip, which is what the ML tests read);
        #   * a fixed-height ``_ml_gate_host`` carries it, because the note keeps its own
        #     show/hide — ``ml_gate_label.isHidden()`` is how the ML tests and the owner
        #     ask "is the gate up?", and an always-visible empty banner answers wrongly.
        self.ml_gate_label = _FixedLinesNote(2)
        self.ml_gate_label.setStyleSheet("color:#FFB74D;")
        self.ml_gate_label.setVisible(False)
        self._ml_gate_host = QWidget()
        _ml_gate_host_layout = QVBoxLayout(self._ml_gate_host)
        _ml_gate_host_layout.setContentsMargins(0, 0, 0, 0)
        _ml_gate_host_layout.addWidget(self.ml_gate_label)
        self._ml_gate_host.setFixedHeight(self.ml_gate_label.minimumHeight())
        ml_form.addRow(self._ml_gate_host)
        ml_layout.addWidget(ml_box)

        self._build_ml_preprocess_group(ml_layout)
        self._build_ml_decode_group(ml_layout)
        ml_layout.addStretch(1)

        root.addStretch(1)

        # -- Wiring -------------------------------------------------------
        # Commit-on-finish for every spin that drives the live previews. With keyboard
        # tracking ON (Qt's default) each intermediate DIGIT is a value change, so typing
        # "5000" into the cutoff walks through 5, 50 and 500 — and each of those triggers
        # a rise-time measurement, a response redraw AND a full-channel refilter of the
        # processed-trace preview. Typing a Savitzky-Golay window that way is minutes of
        # wasted work for values the user never meant. The idiom is already used in
        # file_dock.py; the values themselves and every emitted signal are unchanged.
        for _spin in (
            self.filter_cutoff_hz, self.filter_order,
            self.hp_cutoff_hz, self.hp_order,
            self.notch_freq_hz, self.notch_q, self.notch_harmonics,
            self.median_window_s, self.detrend_window_s,
            self.rmean_window_s, self.savgol_window_s, self.savgol_polyorder,
            self.detrend_percentile, self.detrend_asls_lam, self.detrend_asls_p,
            self.detrend_asls_niter, self.detrend_asls_decim,
            self.cusum_k, self.cusum_h,
        ):
            _spin.setKeyboardTracking(False)

        # -- detector selection --------------------------------------------
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        self.detector_combo.currentIndexChanged.connect(self._emit_params)
        self._on_detector_changed()

        # -- notch spelling ------------------------------------------------
        self.notch_use_lines.toggled.connect(self._refresh_notch_mode)
        self.notch_use_lines.toggled.connect(self._emit_params)
        self.notch_use_lines.toggled.connect(self._update_response_plot)
        self.notch_freqs.editingFinished.connect(self._refresh_notch_mode)
        self.notch_freqs.editingFinished.connect(self._emit_params)
        self.notch_freqs.editingFinished.connect(self._update_response_plot)
        self.notch_harmonics.valueChanged.connect(self._refresh_notch_mode)
        self._refresh_notch_mode()

        self.detect_button.clicked.connect(lambda: self.detectRequested.emit(self.params()))
        # THE THREE SURROGATE STUDIES ARE SCHMITT-ONLY and get a narrowed dict. They
        # sweep k_in, and their receivers hand what they are given straight to
        # `detect_events`, so a CUSUM threshold riding along would reach it as an
        # unexpected keyword and kill the run. Sending the whole form would also be
        # wrong on its own terms: a sweep of the Schmitt enter threshold has no business
        # carrying another detector's knobs.
        self.sweep_button.clicked.connect(
            lambda: self.sweepRequested.emit(self.surrogate_params()))
        self.null_button.clicked.connect(
            lambda: self.nullControlRequested.emit(self.surrogate_params()))
        self.recovery_button.clicked.connect(
            lambda: self.recoveryRequested.emit(self.surrogate_params()))

        # k_out < k_in enforced live in the UI (not just at detect time).
        self.k_in.valueChanged.connect(self._enforce_k_constraint)
        self.k_out.valueChanged.connect(self._enforce_k_constraint)

        # Rise-time readout updates on any change to the low-pass DESIGN, not just its
        # cutoff: a 2-pole Butterworth and an 8-pole Bessel at the same cutoff have very
        # different rise times, so "dwells below X are filter-limited" has to follow the
        # type and order too.
        self.filter_cutoff_hz.valueChanged.connect(self._update_rise_time)
        self.filter_order.valueChanged.connect(self._update_rise_time)
        self.filter_kind.currentIndexChanged.connect(self._update_rise_time)
        self.filter_kind.currentIndexChanged.connect(self._on_filter_kind_changed)
        self.filter_enabled.toggled.connect(self._on_filter_enabled)
        self.filter_preview.toggled.connect(self.previewFilterToggled.emit)

        # The response plot follows every parameter of every stage that has one, so the
        # curve on screen always belongs to the values in the boxes above it.
        for _w in (
            self.filter_cutoff_hz, self.filter_order,
            self.hp_cutoff_hz, self.hp_order,
            self.notch_freq_hz, self.notch_q, self.notch_harmonics,
            self.rmean_window_s,
            self.savgol_window_s, self.savgol_polyorder,
            self.median_window_s,
        ):
            _w.valueChanged.connect(self._update_response_plot)
        self.filter_kind.currentIndexChanged.connect(self._update_response_plot)
        self.hp_kind.currentIndexChanged.connect(self._update_response_plot)

        # The preview curve reflects the FULL preprocessing chain (notch / high-pass /
        # detrend / savgol, then low-pass), so enabling ANY stage makes the "show on
        # trace" toggle meaningful.
        self.detrend_enabled.toggled.connect(self._on_detrend_enabled)
        for cb in (self.notch_enabled, self.median_enabled, self.hp_enabled,
                   self.rmean_enabled, self.savgol_enabled, self.normalize_enabled):
            cb.toggled.connect(self._on_preprocess_stage_toggled)

        # Active-filters indicator: whenever ANY stage's enable toggles, refresh the
        # ● markers in the combo and the execution-order summary, so the whole chain is
        # visible without paging through the stack (independent of the shown page).
        for _label, _cb in self._filter_specs:
            _cb.toggled.connect(self._refresh_active_filters)
            _cb.toggled.connect(self._update_response_plot)

        # Every widget that changes a detection parameter re-emits paramsChanged
        # (the AnalysisPanel recomputes the live preview from it).
        self.polarity_override.currentTextChanged.connect(self._emit_params)
        for w in (
            self.filter_cutoff_hz,
            self.filter_order,
            self.k_in,
            self.k_out,
            self.min_dwell_s,
            self.max_dwell_s,
            self.merge_gap_ms,
            self.detrend_window_s,
            self.detrend_percentile,
            self.detrend_asls_lam,
            self.detrend_asls_p,
            self.detrend_asls_niter,
            self.detrend_asls_decim,
            self.hp_cutoff_hz,
            self.hp_order,
            self.notch_freq_hz,
            self.notch_harmonics,
            self.notch_q,
            self.median_window_s,
            self.rmean_window_s,
            self.savgol_window_s,
            self.savgol_polyorder,
            self.cusum_k,
            self.cusum_h,
            self.detrend_rough_s,
            self.detrend_base_s,
            self.detrend_mask_sigma,
            self.detrend_dilate_ms,
            self.detrend_decim,
            self.detrend_sg_poly,
            self.detrend_sg_decim,
            self.normalize_clip_sigma,
        ):
            w.valueChanged.connect(self._emit_params)
        self.filter_enabled.toggled.connect(self._emit_params)
        self.filter_kind.currentIndexChanged.connect(self._emit_params)
        self.hp_kind.currentIndexChanged.connect(self._emit_params)
        self.detrend_enabled.toggled.connect(self._emit_params)
        self.detrend_method.currentTextChanged.connect(self._emit_params)
        self.normalize_mode.currentIndexChanged.connect(self._emit_params)
        # The banner names the mode, so it has to follow the mode, not only the toggle.
        self.normalize_mode.currentIndexChanged.connect(
            self._refresh_normalize_units_note)
        # The banner QUOTES the clip bound, so it has to follow that spin as well —
        # a banner naming 30 sigma over a run clipping at 10 is a disclosure that is
        # itself wrong.
        self.normalize_clip_sigma.valueChanged.connect(
            self._refresh_normalize_units_note)
        for cb in (self.notch_enabled, self.median_enabled, self.hp_enabled,
                   self.rmean_enabled, self.savgol_enabled, self.normalize_enabled):
            cb.toggled.connect(self._emit_params)

        # ``active_filters_label`` is the reader-facing rendering of ``params()``, not
        # merely of the on/off checkboxes.  A later change to, say, the event-masked
        # median method must replace the old ``block_median`` wording immediately:
        # otherwise the visible recipe and the provenance a new detection records
        # describe different chains.  Listening to the one already-complete form
        # signal covers every stage parameter (including future ones) without copying
        # a second, inevitably stale list of widgets here.
        self.paramsChanged.connect(self._refresh_active_filters)

        # ML detector wiring.
        self.ml_load_button.clicked.connect(self.browse_model)
        self.ml_detect_button.clicked.connect(
            lambda: self.mlDetectRequested.emit(self.ml_params()))
        # exit <= enter enforced live (hysteresis; mirrors the k_out < k_in clamp).
        self.ml_enter.valueChanged.connect(self._enforce_ml_threshold_constraint)
        self.ml_exit.valueChanged.connect(self._enforce_ml_threshold_constraint)

        # Changing the preset repopulates every ML widget; ``activated`` (user-driven
        # only) also latches "the user chose this", so the model-width heuristic never
        # overrules a deliberate pick.
        self.ml_preset.currentIndexChanged.connect(self._load_ml_preset)
        self.ml_preset.activated.connect(self._on_ml_preset_activated)
        self.ml_baseline_method.currentIndexChanged.connect(self._refresh_ml_baseline_rows)
        for _w in (
            self.ml_notch_q, self.ml_baseline_window_s, self.ml_baseline_poly,
            self.ml_baseline_decim, self.ml_masked_rough_s, self.ml_masked_base_s,
            self.ml_mask_sigma, self.ml_dilate_ms, self.ml_despike,
            self.ml_smoothing, self.ml_enter, self.ml_exit, self.ml_merge_ms,
            self.ml_min_event_ms, self.ml_onset_sigma, self.ml_onset_max_ms,
            self.ml_onset_gap_tol, self.ml_offset_sigma, self.ml_norm_clip,
            self.ml_min_depth_sigma, self.ml_min_depth_pa, self.ml_min_fracblock,
        ):
            _w.setKeyboardTracking(False)
            _w.valueChanged.connect(self._on_ml_param_changed)
        for _cb in (self.ml_notch_enabled, self.ml_notch_float32, self.ml_norm_enabled,
                    self.ml_norm_center, self.ml_anchor_tail,
                    self.ml_last_start_inclusive):
            _cb.toggled.connect(self._on_ml_param_changed)
        for _combo in (self.ml_baseline_method, self.ml_norm_scope,
                       self.ml_norm_method):
            _combo.currentIndexChanged.connect(self._on_ml_param_changed)
        self._on_ml_norm_method_changed()
        self.ml_notch_freqs.editingFinished.connect(self._on_ml_param_changed)

        # Start on the apo notebook's own reference bias -- the pipeline that was
        # actually executed and validated. Loading a model re-defaults the preset from
        # the model's input width unless the user has already chosen.
        self.apply_ml_preset(DEFAULT_PRESET)
        self._refresh_ml_enabled()
        self._refresh_active_filters()   # initial combo markers + summary (all off)
        self._on_filter_kind_changed()   # Gaussian has no order -> gate the spin
        self._update_response_plot()     # initial curve (no fs yet -> the "select a channel" note)
        self._organize_inspector_pages()
        self._install_control_levels()

    def _organize_inspector_pages(self) -> None:
        """Pin navigation/actions; move only the settings into independent scrollers.

        Every original parameter widget stays alive and connected. Disclosure and
        page changes cannot apply presets, toggle filters, or emit parameter edits.
        """
        self.setMinimumSize(0, 0)
        root = self.layout()
        root.setContentsMargins(0, 0, 0, 0)
        # The old trailing stretch competed with the active page for space.
        for i in reversed(range(root.count())):
            if root.itemAt(i).spacerItem() is not None:
                root.takeAt(i)
        root.removeWidget(self._gate_host)
        root.addWidget(self._gate_host)
        self._gate_host.setVisible(not self._gate_label.isHidden())
        root.setStretch(root.indexOf(self.mode_tabs), 1)
        self.mode_tabs.setTabText(2, 'ML')
        self.mode_tabs.setTabText(3, 'Measure')
        self.mode_tabs.setTabToolTip(2, 'Machine-learning detection and its model preprocessing')
        self.mode_tabs.setTabToolTip(3, 'Manual measurement with trace cursors')
        self.mode_tabs.setElideMode(Qt.ElideRight)
        self.mode_tabs.setMinimumSize(0, 0)
        self.page_scrolls = {}
        self.page_actions = {}
        for page in (self._preprocess_page, self._detection_page, self._ml_page, self._measure_page):
            old_layout = page.layout()
            # Measure is filled by AnalysisPanel after construction.
            if page is self._measure_page:
                self._measure_actions = QWidget()
                self._measure_actions.setLayout(QVBoxLayout())
                self._measure_actions.layout().setContentsMargins(0, 0, 0, 0)
                self._measure_action_scroll = PinnedActions(self._measure_actions)
                old_layout.addWidget(self._measure_action_scroll)
                continue
            content = host_layout(old_layout)
            old_layout.setAlignment(Qt.AlignTop)
            for i in reversed(range(old_layout.count())):
                if old_layout.itemAt(i).spacerItem() is not None:
                    old_layout.takeAt(i)
            layout = QVBoxLayout(page)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(6)
            actions = QWidget()
            action_layout = QVBoxLayout(actions)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(4)
            action_scroll = PinnedActions(actions)
            layout.addWidget(action_scroll)
            self.page_actions[page] = actions
            if page is self._preprocess_page:
                # Keep the subsection selector beside the persistent navigation.
                item = old_layout.takeAt(0)
                action_layout.addLayout(item.layout())
                self._preprocess_actions = _CurrentPageStackedWidget()
                chain_actions = QWidget()
                chain_layout = QVBoxLayout(chain_actions)
                chain_layout.setContentsMargins(0, 0, 0, 0)
                chain_layout.addWidget(self.open_conditioning_button)
                chain_layout.addWidget(self.filter_preview)
                baseline_actions = QWidget()
                baseline_actions.setLayout(QVBoxLayout())
                baseline_actions.layout().setContentsMargins(0, 0, 0, 0)
                self._baseline_actions_layout = baseline_actions.layout()
                self._preprocess_actions.addWidget(chain_actions)
                self._preprocess_actions.addWidget(baseline_actions)
                self.preprocess_section_selector.currentIndexChanged.connect(
                    self._preprocess_actions.setCurrentIndex)
                self._preprocess_actions.setCurrentIndex(self.preprocess_section_selector.currentIndex())
                action_layout.addWidget(self._preprocess_actions)
            elif page is self._detection_page:
                item = old_layout.takeAt(0)  # the existing 2x2 action grid
                action_layout.addLayout(item.layout())
                # Advanced options retain their exact objects, values and signals.
                advanced = QWidget()
                form = QFormLayout(advanced)
                form.setContentsMargins(0, 0, 0, 0)
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                for control in (self.max_dwell_s, self.merge_gap_ms):
                    source = control.parentWidget().layout()
                    row = source.takeRow(control)
                    form.addRow(row.labelItem.widget(), row.fieldItem.widget())
                # Direction is a primary scientific choice, not an expert-only
                # detail. Keep its original controls outside the disclosure.
                self.detection_advanced = Disclosure('Advanced detection settings', advanced)
                old_layout.addWidget(self.detection_advanced)
                self.detection_advanced_summary = QLabel()
                self.detection_advanced_summary.setWordWrap(True)
                self.detection_advanced_summary.setStyleSheet('color:#8b9199;')
                old_layout.addWidget(self.detection_advanced_summary)
                self.paramsChanged.connect(self._refresh_detection_advanced_summary)
                self._refresh_detection_advanced_summary()
            else:
                # Run always stays in view; the model gate stays next to its action.
                source = self.ml_detect_button.parentWidget().layout()
                source.takeRow(self.ml_detect_button)
                source.takeRow(self._ml_gate_host)
                action_layout.addWidget(self.ml_detect_button)
                action_layout.addWidget(self._ml_gate_host)
                # Decoding is a specialist form. Keep the model/preset and its
                # pipeline warnings outside the disclosure.
                decode = next((old_layout.itemAt(i).widget() for i in range(old_layout.count())
                               if isinstance(old_layout.itemAt(i).widget(), QGroupBox)
                               and 'decod' in old_layout.itemAt(i).widget().title().lower()), None)
                if decode is not None:
                    old_layout.removeWidget(decode)
                    self.ml_advanced = Disclosure('Advanced ML decoding', decode)
                    old_layout.addWidget(self.ml_advanced)
            for form in content.findChildren(QFormLayout):
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            old_layout.addStretch(1)
            action_scroll._sync_height()
            scroll = InspectorScrollArea(content)
            layout.addWidget(scroll, 1)
            self.page_scrolls[page] = scroll

    def _refresh_detection_advanced_summary(self, *_args) -> None:
        self.detection_advanced_summary.setText(
            f'Max dwell: {self.max_dwell_s.value():g} s (0 = off) · '
            f'Merge gap: {self.merge_gap_ms.value():g} ms (0 = off) · '
            f'Polarity: {self.polarity_override.currentText()}')

    def _install_control_levels(self) -> None:
        """Add view complexity choices without owning a scientific parameter."""
        self._control_level = "advanced"
        self._advanced_disclosure_state = {}
        self._essential_summary_text = ""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(4, 0, 4, 0)
        row.addWidget(QLabel("Controls"))
        self.control_level_selector = QComboBox()
        self.control_level_selector.addItem("Essential", "essential")
        self.control_level_selector.addItem("Advanced", "advanced")
        self.control_level_selector.setCurrentIndex(1)
        self.control_level_selector.setAccessibleName("Inspector control level")
        self.control_level_selector.setToolTip(
            "Changes visibility only. No filters, thresholds, presets, results or review history are changed.")
        row.addWidget(self.control_level_selector, 1)
        self.baseline_reference_button = QPushButton("Baseline reference…")
        self.baseline_reference_button.setToolTip(
            "Inspect and confirm the open-pore reference used to measure event amplitudes.")
        self.baseline_reference_button.clicked.connect(self.show_open_pore_baseline)
        row.addWidget(self.baseline_reference_button)
        self.layout().insertWidget(0, bar)
        self.essential_settings_summary = _FixedLinesNote(3)
        self.essential_settings_summary.setAccessibleName("Essential view retained settings")
        self.essential_settings_summary.setStyleSheet("color:#8b9199;")
        self.layout().insertWidget(1, self.essential_settings_summary)
        self.advanced_tools_button = QPushButton("Show advanced tools…")
        self.advanced_tools_button.setToolTip("Reveal threshold sweep, null control, recovery studies, ML and response design.")
        self.advanced_tools_button.clicked.connect(lambda: self.set_control_level("advanced", persist=True))
        self.page_actions[self._detection_page].layout().addWidget(self.advanced_tools_button)
        self.control_level_selector.currentIndexChanged.connect(
            lambda: self.set_control_level(str(self.control_level_selector.currentData())))
        self.control_level_selector.activated.connect(
            lambda: self.set_control_level(str(self.control_level_selector.currentData()), persist=True))
        self.paramsChanged.connect(self._refresh_control_level_summary)
        self.mlModelChanged.connect(self._refresh_control_level_summary)
        self.ml_range.currentIndexChanged.connect(self._refresh_control_level_summary)
        # A first-use guide choice changes visibility only. Explicit existing
        # inspector preferences always win; simply opening the guide writes neither.
        settings = QSettings()
        saved = str(settings.value(self._CONTROL_LEVEL_SETTINGS_KEY,
                    settings.value("onboarding/control_level_choice_v1", "advanced")))
        self.set_control_level(saved if saved in {"essential", "advanced"} else "advanced")

    def control_level(self) -> str:
        """Current visibility preference, separate from params()/ml_params()."""
        return self._control_level

    def set_control_level(self, level: str, *, persist: bool = False) -> None:
        """Choose Essential/Advanced without changing any scientific widget.

        Programmatic Guide navigation is non-persistent by default. User selection
        saves only this UI preference. Enabled filters and non-default expert
        settings remain disclosed; hidden controls are never disabled or reset.
        """
        if level not in {"essential", "advanced"}:
            raise ValueError("Control level must be 'essential' or 'advanced'.")
        previous = self._control_level
        if previous != level:
            disclosures = [self.detection_advanced, *([self.ml_advanced] if hasattr(self, "ml_advanced") else [])]
            if level == "essential":
                self._advanced_disclosure_state = {widget: widget.toggle.isChecked() for widget in disclosures}
                for widget in disclosures:
                    widget.set_expanded(False)
            else:
                for widget, expanded in self._advanced_disclosure_state.items():
                    widget.set_expanded(expanded)
            self._control_level = level
        # Block only this presentation selector. Scientific controls are untouched.
        was_blocked = self.control_level_selector.blockSignals(True)
        self.control_level_selector.setCurrentIndex(self.control_level_selector.findData(level))
        self.control_level_selector.blockSignals(was_blocked)
        essential = level == "essential"
        for widget in (self.sweep_button, self.null_button, self.recovery_button,
                       self.response_hint):
            widget.setVisible(not essential)
        # The hosted response view lives in the expert Signal Conditioning window, which
        # the Inspector's control level does not govern.
        for widget in (self.response_plot, self.response_note, self.response_chain_check):
            widget.setVisible(self._chain_editor_hosted or not essential)
        self.advanced_tools_button.setVisible(essential)
        self.essential_settings_summary.setVisible(essential)
        self._refresh_control_level_summary()
        for actions in self.findChildren(PinnedActions):
            actions._sync_height()
        if persist:
            QSettings().setValue(self._CONTROL_LEVEL_SETTINGS_KEY, level)
        if previous != level:
            self.controlLevelChanged.emit(level)

    _NORMALIZE_CRITICAL = "NORMALIZE ON: current-derived values are dimensionless"
    _HIGHPASS_CRITICAL = "high-pass affects localization and boundaries"

    def _refresh_control_level_summary(self, *_args) -> None:
        if not hasattr(self, "essential_settings_summary"):
            return
        active = [label for label, check in self._filter_specs if check.isChecked()]
        retained = []
        if self.max_dwell_s.value() > 0:
            retained.append(f"max dwell {self.max_dwell_s.value():g} s")
        if self.merge_gap_ms.value() > 0:
            retained.append(f"merge {self.merge_gap_ms.value():g} ms")
        if self.normalize_enabled.isChecked():
            retained.append(self._NORMALIZE_CRITICAL)
        if self.hp_enabled.isChecked():
            retained.append(self._HIGHPASS_CRITICAL)
        ml_configured = bool(self._model_path or self._ml_preset_chosen
                             or getattr(self, "_nanoscope_unresolved_model_reference", None))
        if ml_configured:
            retained.append("ML setup retained (separate model pipeline)")
        critical = [note for note in retained if note.startswith(("NORMALIZE", "high-pass"))]
        others = [note for note in retained if note not in critical]
        text = (("; ".join(critical) + ". " if critical else "")
                + "Enabled filters: " + (", ".join(active) if active else "none") + ". "
                + ("Retained settings: " + "; ".join(others) + ". " if others else "")
                + "Control level changes visibility only. Expert options remain available under Advanced.")
        self._essential_summary_text = text
        self.essential_settings_summary.setText(text)
        # A loaded/customized/unresolved model must not disappear from navigation.
        # Also keep the current ML page visible when switching views while reading it.
        ml_visible = (self._control_level == "advanced" or ml_configured
                      or self.mode_tabs.currentWidget() is self._ml_page)
        self.mode_tabs.setTabVisible(self.mode_tabs.indexOf(self._ml_page), ml_visible)

    def control_level_summary(self) -> str:
        """Full unelided disclosure for accessibility, tests and Guide details."""
        return self._essential_summary_text

    # -- ML tab: the model's OWN preprocessing + decode ---------------------
    # These two groups are the ML detector's whole signal chain. They exist because a
    # learned model only behaves as it was measured to when it is fed the pipeline it
    # was TRAINED on — which is NOT the general-purpose Preprocess tab chain. The
    # presets are the pipelines the shipped models were trained/validated under
    # (ml_preprocess.PREPROCESS_PRESETS / DECODE_PRESETS), across TWO MOLECULES —
    # ferritin and apo-ferritin; every parameter stays editable, because a user with
    # their own model needs their own numbers, exactly as on the general filters.
    # Editing one is flagged as a deviation from the preset — consistency with
    # training is a correctness property, not a taste.
    def _build_ml_preprocess_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("Model preprocessing")
        outer = QVBoxLayout(box)

        note = QLabel(self._PIPELINE_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b9199;")
        outer.addWidget(note)
        self.ml_pipeline_note = note

        form = QFormLayout()
        outer.addLayout(form)
        self._ml_preprocess_form = form

        # One combo, but the two MOLECULES are separated so the list cannot read as a
        # flat menu of settings. Every label is molecule-first, and a separator sits
        # on each molecule boundary.
        # The 115-character preset labels measured 706 px here and were the single widest
        # thing in the Inspector; see :func:`_narrow_combo`.
        self.ml_preset = _narrow_combo()
        prev_molecule: str | None = None
        for name in PRESET_NAMES:
            molecule = PRESET_MOLECULE[name]
            if prev_molecule is not None and molecule != prev_molecule:
                self.ml_preset.insertSeparator(self.ml_preset.count())
            self.ml_preset.addItem(PRESET_LABELS[name], name)
            prev_molecule = molecule
        self.ml_preset.setToolTip(
            "TWO MOLECULES, not two parameter sets — a ferritin-trained model on apo "
            "data (or the reverse) is a category error.\n"
            "Ferritin nb1/2/3 are the exact chains dataLoader.ipynb built the three "
            "StudentModel training sets with (window 2048); they are three different "
            "RECORDINGS, not three revisions.\n"
            "Apo-ferritin −500/−600/−700 mV are the ApoFerritin_LSTM_Cluster_Validation "
            "notebook (window 3072): one shared preprocessing chain, and a single "
            "hand-tuned threshold per bias with NO probability smoothing and NO "
            "hysteresis.\n"
            "Apo-ferritin raw-global is the older student_raw_global_raw_best chain "
            "(also window 3072). Its preprocessing is that model's real training "
            "contract; its decode numbers are function-signature defaults that no "
            "notebook ever ran — treat them as a starting point."
        )
        # Everything up to here is a SHIPPED preset; anything the combo gains below this
        # count is one the user saved. `_reload_ml_user_presets` truncates back to this
        # index before re-adding, so the built-in block can never be damaged by an edit
        # to the user store.
        self._ml_builtin_item_count = self.ml_preset.count()
        form.addRow("preset", self.ml_preset)

        # -- user-defined model presets (owner request) --------------------------------
        # The shipped presets are the training contracts of the models WE ship. A user
        # running their own model has their own contract, and until now the only way to
        # keep it was to re-dial ~20 fields from memory every session — which is exactly
        # how a run ends up silently deviating from the pipeline its model was trained
        # on. Saving stores this form's `MLPreprocessConfig` + `MLDecodeConfig` verbatim
        # under a name, in QSettings (per user, like the preprocessing protocols), and
        # recalling one pushes both back through the same `_push_ml_*` writers a shipped
        # preset uses — so a user preset IS a preset, deviations indicator included, not
        # a second-class copy of one.
        # LEFT-ALIGNED, DELIBERATELY. This row used to open with addStretch(1), which pins
        # both buttons to the RIGHT edge of whatever width the form is given. The form was
        # 814 px wide (see the combo above), the viewport is 420, so 'Save as…' sat at
        # x=647..727 and 'Delete' at x=733..813 — not one pixel of either on screen, and
        # neither has a menu item or a shortcut. A trailing stretch instead puts them at
        # x≈107..273, inside the viewport at ANY dock width this app can be dragged to.
        ml_preset_row = QHBoxLayout()
        self.ml_preset_save_btn = QPushButton("Save as…")
        self.ml_preset_save_btn.setToolTip(
            "Save the model preprocessing + decode settings currently on this tab "
            "under a name, for reuse with your own model.")
        self.ml_preset_delete_btn = QPushButton("Delete")
        self.ml_preset_delete_btn.setToolTip(
            "Forget the selected saved preset. The shipped presets cannot be deleted.")
        ml_preset_row.addWidget(self.ml_preset_save_btn)
        ml_preset_row.addWidget(self.ml_preset_delete_btn)
        ml_preset_row.addStretch(1)
        form.addRow("", ml_preset_row)
        self.ml_preset_save_btn.clicked.connect(self._save_ml_preset_as)
        self.ml_preset_delete_btn.clicked.connect(self._delete_ml_preset)
        self._reload_ml_user_presets()

        # The molecule, said outright rather than left to be read off the label.
        self.ml_preset_molecule = QLabel("")
        self.ml_preset_molecule.setWordWrap(True)
        form.addRow("molecule", self.ml_preset_molecule)

        self.ml_preset_status = QLabel("")
        self.ml_preset_status.setWordWrap(True)
        form.addRow("", self.ml_preset_status)

        self.ml_notch_enabled = _ElidingCheckBox("enable")
        self.ml_notch_enabled.setToolTip(
            "Mains notch, applied as one zero-phase filtfilt pass PER LINE in the order "
            "listed (dataLoader.ipynb's apply_notch_filters). This is deliberately not "
            "the Preprocess tab's notch, which cascades every line into one SOS and "
            "filters once — a numerically different signal at the window edges."
        )
        self.ml_notch_freqs = QLineEdit(
            ", ".join(f"{f:g}" for f in NOTCH_FREQS))
        self.ml_notch_freqs.setToolTip(
            "Comma-separated notch lines in Hz, applied in this order. Training used "
            "60, 120, 180.")
        self.ml_notch_q = QDoubleSpinBox()
        self.ml_notch_q.setDecimals(1)
        self.ml_notch_q.setRange(0.1, 500.0)
        self.ml_notch_q.setSingleStep(1.0)
        self.ml_notch_q.setValue(30.0)
        self.ml_notch_q.setKeyboardTracking(False)
        self.ml_notch_float32 = _ElidingCheckBox("round to float32 after notch")
        self.ml_notch_float32.setToolTip(
            "filtfilt returns float64, so the question is only what the BASELINE stage "
            "is handed. The apo notebook's apply_notch_filter_harmonics ends in "
            "'.astype(np.float32)', so its SavGol runs and accumulates in float32; "
            "dataLoader.ipynb (ferritin) has no such cast and keeps float64. Measured "
            "difference on the real −600 mV apo recording: 3e-05 pA — five orders below "
            "the ADC step, but it is the difference between bit-exact and not."
        )
        form.addRow("notch", self.ml_notch_enabled)
        form.addRow("notch lines", self.ml_notch_freqs)
        form.addRow("notch Q", self.ml_notch_q)
        form.addRow("notch dtype", self.ml_notch_float32)

        # BUILT from ml_preprocess.BASELINE_METHODS, so the estimators offered here can
        # never drift from the ones apply_ml_preprocess actually implements.
        self.ml_baseline_method = QComboBox()
        for key in BASELINE_METHODS:
            self.ml_baseline_method.addItem(self._BASELINE_LABELS[key], key)
        self.ml_baseline_method.setToolTip(
            "robust median — outlier-resistant rolling median (notebook 1).\n"
            "event-masked median — rough median exposes events, they are masked and "
            "interpolated over, then a SHORT median tracks drift (notebooks 2 and 3).\n"
            "Savitzky-Golay — polynomial envelope (apo legacy only; it overshoots "
            "around deep/long events)."
        )
        form.addRow("baseline", self.ml_baseline_method)

        self.ml_baseline_window_s = self._ml_seconds_spin(
            0.5, "Rolling-median / SavGol window. The median tap count is computed on "
                 "the DECIMATED grid: window x fs / decim, forced odd (0.5 s at 50 kHz "
                 "with decim 50 -> 501 taps).")
        self.ml_baseline_poly = QSpinBox()
        self.ml_baseline_poly.setRange(1, 9)
        self.ml_baseline_poly.setValue(3)
        self.ml_baseline_poly.setKeyboardTracking(False)
        self.ml_baseline_poly.setToolTip("Savitzky-Golay polynomial order.")
        self.ml_baseline_decim = QSpinBox()
        self.ml_baseline_decim.setRange(1, 10_000)
        self.ml_baseline_decim.setValue(50)
        self.ml_baseline_decim.setKeyboardTracking(False)
        self.ml_baseline_decim.setToolTip(
            "Decimation for the median estimate (the baseline is < 1 Hz, so this is "
            "lossless). It also sets the resolution floor: decim 50 at 50 kHz is "
            "1 ms/sample and cannot resolve a 10 ms window — notebook 3 drops it to 10.")
        self.ml_masked_rough_s = self._ml_seconds_spin(
            0.5, "Masked median: the ROUGH window that exposes events before masking.")
        self.ml_masked_base_s = self._ml_seconds_spin(
            0.1, "Masked median: the SHORT window that tracks drift after the events "
                 "have been masked and interpolated over.")
        self.ml_mask_sigma = QDoubleSpinBox()
        self.ml_mask_sigma.setDecimals(2)
        self.ml_mask_sigma.setRange(0.1, 50.0)
        self.ml_mask_sigma.setSingleStep(0.5)
        self.ml_mask_sigma.setValue(3.5)
        self.ml_mask_sigma.setKeyboardTracking(False)
        self.ml_mask_sigma.setToolTip(
            "Samples below median - sigma x MAD noise are masked as events. The MAD is "
            "GLOBAL over the preprocessed span.")
        self.ml_dilate_ms = QDoubleSpinBox()
        self.ml_dilate_ms.setDecimals(3)
        self.ml_dilate_ms.setRange(0.0, 1000.0)
        self.ml_dilate_ms.setSingleStep(0.5)
        self.ml_dilate_ms.setValue(1.0)
        self.ml_dilate_ms.setSuffix(" ms")
        self.ml_dilate_ms.setKeyboardTracking(False)
        self.ml_dilate_ms.setToolTip(
            "Binary dilation of the event mask, in ms (int(ms x 1e-3 x fs) iterations, "
            "so 50 at 50 kHz for 1 ms).")
        form.addRow("window", self.ml_baseline_window_s)
        form.addRow("poly order", self.ml_baseline_poly)
        form.addRow("decimation", self.ml_baseline_decim)
        form.addRow("rough window", self.ml_masked_rough_s)
        form.addRow("base window", self.ml_masked_base_s)
        form.addRow("mask sigma", self.ml_mask_sigma)
        form.addRow("mask dilate", self.ml_dilate_ms)

        self.ml_despike = QSpinBox()
        self.ml_despike.setRange(0, 99)
        self.ml_despike.setValue(0)
        self.ml_despike.setKeyboardTracking(False)
        self.ml_despike.setSpecialValueText("off")
        self.ml_despike.setToolTip(
            "Post-detrend median filter that removes 1-2 sample spike artifacts "
            "(0 = off). Only notebook 2 uses it, at size 3.")
        form.addRow("de-spike", self.ml_despike)

        self.ml_norm_enabled = _ElidingCheckBox("normalize the model input")
        self.ml_norm_enabled.setChecked(True)
        self.ml_norm_enabled.setToolTip(
            "Global normalization applied BEFORE windowing. (The 12-feature teacher "
            "contract standardizes internally and never uses this.)")
        # WHICH normalization. Two kernels, not two settings of one: max-abs is a single
        # affine map, robust sigma is a twice-trimmed re-estimation followed by a clip.
        # The combo is BUILT from ml_preprocess.NORMALIZE_METHODS so it cannot offer one
        # `model_input_normalization` would not dispatch to.
        # Sentence-length method labels (342 px); capped so they cannot floor the ML
        # page, which in turn floored the whole Inspector column (_narrow_combo).
        self.ml_norm_method = _narrow_combo()
        _ML_NORM_LABELS = {
            "maxabs": "max|x|  —  x / (max|x| + 1e-8)",
            "robust_sigma": "robust σ  —  trimmed median/MAD, clipped → [−1, +1]",
        }
        for _m in NORMALIZE_METHODS:
            self.ml_norm_method.addItem(_ML_NORM_LABELS[_m], _m)
        self.ml_norm_method.setToolTip(
            "max|x| is the original: the divisor is a SINGLE EXTREME SAMPLE of the "
            "recording. Measured across the owner's host-guest files that divisor "
            "ranged 8.8–42.6 pA, so ordinary noise reached the model at 0.049 for the "
            "training domain but 0.146 for the host — host NOISE arriving at the "
            "amplitude the model was taught to call an EVENT.\n\n"
            "robust σ is what replaced it in all six of his student notebooks: a "
            "twice-trimmed median/MAD (so events are excluded from the divisor), then "
            "clip to ±σ and divide by it. The noise floor lands at 1/σ in every "
            "recording by construction, so a model transfers between recordings.\n\n"
            "Mean-centring does not apply to robust σ — that kernel estimates its own "
            "centre.")
        self.ml_norm_clip = QDoubleSpinBox()
        self.ml_norm_clip.setDecimals(1)
        self.ml_norm_clip.setRange(0.1, 1000.0)
        self.ml_norm_clip.setSingleStep(5.0)
        self.ml_norm_clip.setValue(float(DEFAULT_NORMALIZE_CLIP_SIGMA))
        self.ml_norm_clip.setKeyboardTracking(False)
        self.ml_norm_clip.setToolTip(
            "BOTH the clip bound and the divisor, for robust σ only. Clipping at ±30σ "
            "and dividing by 30 is what puts the noise floor at exactly 1/30 — the one "
            "number these models were trained against. Anything deeper than this is "
            "SATURATED at the bound.")
        self.ml_norm_center = _ElidingCheckBox("subtract the mean first")
        self.ml_norm_center.setToolTip(
            "(x − mean) / (max|x − mean| + 1e-8) — the APO notebook's normalize_signal. "
            "The ferritin chain (student_predict) does NOT subtract the mean; it is the "
            "pure rescale x / (max|x| + 1e-8). After a SavGol detrend the residual mean "
            "is tiny (−0.0031 pA on the real −600 mV recording, and it moved 0 of 236 "
            "events), but it is what the apo code executes.")
        self.ml_norm_scope = QComboBox()
        self.ml_norm_scope.addItem("Whole recording", "recording")
        self.ml_norm_scope.addItem("Analysed range", "range")
        self.ml_norm_scope.setToolTip(
            "Where the mean and max|x| are measured. Both notebooks measured them over "
            "the WHOLE recording, so that is the default and the model's input scale "
            "then does not depend on your zoom. 'Analysed range' reproduces the older "
            "behaviour, where a sub-range run silently rescaled the input.")
        form.addRow("normalize", self.ml_norm_enabled)
        form.addRow("norm method", self.ml_norm_method)
        form.addRow("clip (σ)", self.ml_norm_clip)
        form.addRow("mean-centre", self.ml_norm_center)
        form.addRow("norm scope", self.ml_norm_scope)
        self._ml_preprocess_form = form
        self.ml_norm_method.currentIndexChanged.connect(self._on_ml_norm_method_changed)

        self.ml_anchor_tail = _ElidingCheckBox("cover the tail")
        self.ml_anchor_tail.setToolTip(
            "Add a final window anchored at n − W so the last partial stride is covered. "
            "Neither notebook does this, so their tails sit at probability 0; the "
            "raw-global apo path does.")
        self.ml_last_start_inclusive = _ElidingCheckBox("starts may reach n − W")
        self.ml_last_start_inclusive.setToolTip(
            "Which notebook's window-start loop to reproduce when the tail is not "
            "anchored: off = ferritin's range(0, n − W, step); on = apo's "
            "np.arange(0, n − W + 1, step). They agree on every recording length EXCEPT "
            "those where (n − W) is an exact multiple of the stride, where the ferritin "
            "loop stops one window early and leaves a whole stride blind. Ignored when "
            "the tail is anchored.")
        form.addRow("tail window", self.ml_anchor_tail)
        form.addRow("start bound", self.ml_last_start_inclusive)

        parent_layout.addWidget(box)
        self.ml_preprocess_box = box

    def _build_ml_decode_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("Model decode")
        form = QFormLayout(box)
        self._ml_decode_form = form

        self.ml_smoothing = QSpinBox()
        self.ml_smoothing.setRange(1, 999)
        self.ml_smoothing.setSingleStep(2)
        self.ml_smoothing.setValue(9)
        self.ml_smoothing.setKeyboardTracking(False)
        self.ml_smoothing.setToolTip(
            "Median-filter kernel applied to the per-sample probability trace before "
            "thresholding (forced odd). 1 = no smoothing at all, which is what the apo "
            "notebook does — it thresholds the raw overlap-averaged probability.")
        form.addRow("smoothing", self.ml_smoothing)

        # enter/exit live HERE — they are the decode's headline thresholds. Same
        # widgets as before (same attribute names, same signals), just grouped with
        # the rest of the decode contract instead of with the model loader.
        self.ml_enter = QDoubleSpinBox()
        self.ml_enter.setDecimals(2)
        self.ml_enter.setRange(0.01, 1.0)
        self.ml_enter.setSingleStep(0.05)
        self.ml_enter.setValue(ml_detect.DEFAULT_ENTER)
        self.ml_enter.setToolTip(
            "Enter threshold on the per-sample event probability (hysteresis). A run "
            "of samples is an event only if the probability reaches this.")
        self.ml_exit = QDoubleSpinBox()
        self.ml_exit.setDecimals(2)
        self.ml_exit.setRange(0.0, 1.0)
        self.ml_exit.setSingleStep(0.05)
        self.ml_exit.setValue(ml_detect.DEFAULT_EXIT)
        self.ml_exit.setToolTip(
            "Exit threshold: the event stays alive until the probability falls below "
            "this. Must be <= enter (hysteresis, so the boundary does not chatter).")
        form.addRow("enter", self.ml_enter)
        form.addRow("exit", self.ml_exit)

        self.ml_merge_ms = QDoubleSpinBox()
        self.ml_merge_ms.setDecimals(3)
        self.ml_merge_ms.setRange(0.0, 10_000.0)
        self.ml_merge_ms.setSingleStep(0.1)
        self.ml_merge_ms.setValue(0.1)
        self.ml_merge_ms.setSuffix(" ms")
        self.ml_merge_ms.setKeyboardTracking(False)
        self.ml_merge_ms.setToolTip("Join regions separated by at most this gap.")
        self.ml_min_event_ms = QDoubleSpinBox()
        self.ml_min_event_ms.setDecimals(3)
        self.ml_min_event_ms.setRange(0.0, 10_000.0)
        self.ml_min_event_ms.setSingleStep(0.1)
        self.ml_min_event_ms.setValue(0.1)
        self.ml_min_event_ms.setSuffix(" ms")
        self.ml_min_event_ms.setKeyboardTracking(False)
        self.ml_min_event_ms.setToolTip(
            "Drop events shorter than this. Applied LAST, so the test uses the final "
            "merged + trimmed extent.")
        form.addRow("merge gap", self.ml_merge_ms)
        form.addRow("min duration", self.ml_min_event_ms)

        self.ml_onset_sigma = QDoubleSpinBox()
        self.ml_onset_sigma.setDecimals(2)
        self.ml_onset_sigma.setRange(0.0, 50.0)
        self.ml_onset_sigma.setSingleStep(0.5)
        self.ml_onset_sigma.setValue(0.0)
        self.ml_onset_sigma.setSpecialValueText("off")
        self.ml_onset_sigma.setKeyboardTracking(False)
        self.ml_onset_sigma.setToolTip(
            "Onset snap: walk each onset back to where the current actually entered the "
            "blockade (below median - sigma x MAD noise), correcting model onset latency. "
            "0 = off. Notebook 3 uses 2.0; notebooks 1 and 2 leave it off.")
        self.ml_onset_max_ms = QDoubleSpinBox()
        self.ml_onset_max_ms.setDecimals(3)
        self.ml_onset_max_ms.setRange(0.0, 1000.0)
        self.ml_onset_max_ms.setSingleStep(0.5)
        self.ml_onset_max_ms.setValue(5.0)
        self.ml_onset_max_ms.setSuffix(" ms")
        self.ml_onset_max_ms.setKeyboardTracking(False)
        self.ml_onset_max_ms.setToolTip("How far back the onset snap may walk.")
        self.ml_onset_gap_tol = QSpinBox()
        self.ml_onset_gap_tol.setRange(0, 1000)
        self.ml_onset_gap_tol.setValue(5)
        self.ml_onset_gap_tol.setKeyboardTracking(False)
        self.ml_onset_gap_tol.setToolTip(
            "How many consecutive shallow samples the backward walk tolerates before "
            "it stops.")
        self.ml_offset_sigma = QDoubleSpinBox()
        self.ml_offset_sigma.setDecimals(2)
        self.ml_offset_sigma.setRange(0.0, 50.0)
        self.ml_offset_sigma.setSingleStep(0.5)
        self.ml_offset_sigma.setValue(0.0)
        self.ml_offset_sigma.setSpecialValueText("off")
        self.ml_offset_sigma.setKeyboardTracking(False)
        self.ml_offset_sigma.setToolTip(
            "Offset trim: snap each offset to the last sample deeper than "
            "median - sigma x MAD noise, removing shallow recovery tails. 0 = off. "
            "Notebook 3 uses 3.0.")
        form.addRow("onset snap", self.ml_onset_sigma)
        form.addRow("onset lookback", self.ml_onset_max_ms)
        form.addRow("onset gap tol", self.ml_onset_gap_tol)
        form.addRow("offset trim", self.ml_offset_sigma)

        # -- the three AMPLITUDE gates ------------------------------------------
        # They live at the BOTTOM of this form because that is where they live in
        # the pipeline: after the merge, after the min-duration filter, measured on
        # the final merged extent. A form that lists them next to "enter" would
        # invite reading them as another probability threshold, which they are not
        # — they never look at the probability trace at all.
        #
        # All three default to 0 = off so an untouched form reproduces the decode
        # this app has always run.
        self.ml_min_depth_sigma = QDoubleSpinBox()
        self.ml_min_depth_sigma.setDecimals(2)
        self.ml_min_depth_sigma.setRange(0.0, 100.0)
        self.ml_min_depth_sigma.setSingleStep(0.5)
        self.ml_min_depth_sigma.setValue(0.0)
        self.ml_min_depth_sigma.setSpecialValueText("off")
        self.ml_min_depth_sigma.setSuffix(" σ")
        self.ml_min_depth_sigma.setKeyboardTracking(False)
        self.ml_min_depth_sigma.setToolTip(
            "Sustained-depth gate: keep an event only if its MEDIAN depth over the "
            "whole event is at least this many event-excluded MADs. The median, not "
            "the deepest sample — a one-sample noise spike clears a min-based gate, "
            "a real event holds its level. The three student presets use 4.0; it "
            "does most of their selection work. 0 = off.")
        self.ml_min_depth_pa = QDoubleSpinBox()
        self.ml_min_depth_pa.setDecimals(2)
        self.ml_min_depth_pa.setRange(0.0, 100_000.0)
        self.ml_min_depth_pa.setSingleStep(1.0)
        self.ml_min_depth_pa.setValue(0.0)
        self.ml_min_depth_pa.setSpecialValueText("off")
        self.ml_min_depth_pa.setSuffix(" pA")
        self.ml_min_depth_pa.setKeyboardTracking(False)
        self.ml_min_depth_pa.setToolTip(
            "ABSOLUTE sustained-depth floor, in picoamps. The σ gate above scales "
            "with the recording's own noise, so where the baseline wanders it widens "
            "and a fluctuation still clears it; this one does not move. Blunt and "
            "pore-independent. Cu2-insulin uses 10 pA. 0 = off.")
        self.ml_min_fracblock = QDoubleSpinBox()
        self.ml_min_fracblock.setDecimals(4)
        self.ml_min_fracblock.setRange(0.0, 1.0)
        self.ml_min_fracblock.setSingleStep(0.01)
        self.ml_min_fracblock.setValue(0.0)
        self.ml_min_fracblock.setSpecialValueText("off")
        self.ml_min_fracblock.setKeyboardTracking(False)
        self.ml_min_fracblock.setToolTip(
            "ABSOLUTE sustained ΔI/I0 floor, measured against the TIME-LOCAL open-pore "
            "current (5 s rolling event-excluded median of the raw trace), so it "
            "follows I0 drift instead of using one global number. This is the gate "
            "that partially normalizes pore size, and it bites hardest exactly where "
            "the σ gate is weakest — on a 316 pA pore a 4σ gate is only ΔI/I0 ≈ 0.023. "
            "Cu2-insulin uses 0.05. 0 = off.")
        form.addRow("min depth (σ)", self.ml_min_depth_sigma)
        form.addRow("min depth (pA)", self.ml_min_depth_pa)
        form.addRow("min ΔI/I0", self.ml_min_fracblock)

        parent_layout.addWidget(box)
        self.ml_decode_box = box

    @staticmethod
    def _ml_seconds_spin(value: float, tip: str) -> QDoubleSpinBox:
        """A seconds-valued window spin with enough decimals for notebook 3's 0.01 s."""
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(0.0001, 3600.0)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        spin.setSuffix(" s")
        spin.setKeyboardTracking(False)
        spin.setToolTip(tip)
        return spin

    # -- user-defined ML presets ------------------------------------------------------
    #: QSettings key for the user's own model-preprocessing presets. A SECOND store,
    #: separate from `_PROTO_SETTINGS_KEY`: the two menus describe different pipelines
    #: (the general Preprocess chain vs the model's own training contract) and mixing
    #: them would let a name from one be recalled into the other.
    _ML_PRESET_SETTINGS_KEY = "detection/ml_user_presets"

    @staticmethod
    def _load_ml_user_presets() -> dict[str, dict]:
        """The user's saved model presets: ``{name: {'preprocess': …, 'decode': …}}``.

        A corrupt store means no user presets, never a crash — the same contract as
        :meth:`_load_protocols`, and for the same reason: this runs during ``__init__``.
        """
        import json
        raw = QSettings().value(DetectionPanel._ML_PRESET_SETTINGS_KEY, "")
        if not raw:
            return {}
        try:
            got = json.loads(raw)
            return got if isinstance(got, dict) else {}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def _store_ml_user_presets(presets: dict[str, dict]) -> None:
        import json
        QSettings().setValue(
            DetectionPanel._ML_PRESET_SETTINGS_KEY, json.dumps(presets))

    def _reload_ml_user_presets(self) -> None:
        """Rebuild the combo's user-preset block from the store, keeping the selection.

        Truncates to `_ml_builtin_item_count` and re-adds, so the shipped block is never
        touched. Guarded with ``_ml_loading`` because removing/adding items moves the
        current index, which would otherwise fire ``_load_ml_preset`` and overwrite the
        form the user is in the middle of saving."""
        combo = self.ml_preset
        wanted = combo.currentData()
        was_loading = self._ml_loading
        self._ml_loading = True
        combo.blockSignals(True)
        try:
            while combo.count() > self._ml_builtin_item_count:
                combo.removeItem(combo.count() - 1)
            names = sorted(self._load_ml_user_presets())
            if names:
                combo.insertSeparator(combo.count())
                for name in names:
                    combo.addItem(f"Yours — {name}", name)
            idx = combo.findData(wanted)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
            self._ml_loading = was_loading
        self._refresh_ml_preset_buttons()

    def _refresh_ml_preset_buttons(self) -> None:
        """Delete is only meaningful on a preset the user owns — the shipped ones are
        the models' training contracts and are not the user's to remove."""
        name = str(self.ml_preset.currentData() or "")
        self.ml_preset_delete_btn.setEnabled(
            bool(name) and name in self._load_ml_user_presets())

    def _ml_preset_configs(
        self, name: str
    ) -> tuple[MLPreprocessConfig | None, MLDecodeConfig | None]:
        """``(preprocess, decode)`` for a preset name — SHIPPED OR USER-SAVED.

        The one resolver behind every place that needs a preset's numbers. Four call
        sites used to index ``PREPROCESS_PRESETS`` / ``DECODE_PRESETS`` directly, which
        is fine for the shipped names and a ``KeyError`` for anything the user saved;
        `_refresh_ml_preset_status` and :meth:`ml_params` both did it on every keystroke
        and every run. Unknown in both stores -> ``(None, None)``, never a substituted
        preset with different numbers.
        """
        pre = PREPROCESS_PRESETS.get(name)
        dec = DECODE_PRESETS.get(name)
        if pre is not None and dec is not None:
            return pre, dec
        stored = self._load_ml_user_presets().get(name)
        if not isinstance(stored, dict):
            return None, None
        # A hand-edited or out-of-date store can hold values the dataclasses reject
        # (their __post_init__ validates); a bad entry is treated as absent rather than
        # allowed to raise out of a combo change.
        try:
            return (MLPreprocessConfig.from_dict(stored.get("preprocess")),
                    MLDecodeConfig.from_dict(stored.get("decode")))
        except (TypeError, ValueError):
            return None, None

    def _ml_preset_label(self, name: str) -> str:
        """The user-facing name of a preset, shipped or user-saved."""
        if name in PRESET_LABELS:
            return PRESET_LABELS[name]
        return f"your preset “{name}”"

    def _save_ml_preset_as(self) -> None:
        """Store the ML tab's current preprocessing + decode as a named user preset."""
        name, ok = QInputDialog.getText(
            self, "Save model preprocessing preset",
            "Name for this model's pipeline (preprocessing + decode):")
        name = (name or "").strip()
        if not ok or not name:
            return
        # A shipped name is not available: those keys are what a stored run resolves
        # through `resolve_preset_name`, so shadowing one would make an old run replay
        # under different numbers than it ran with.
        if name in PREPROCESS_PRESETS or name in PRESET_ALIASES:
            QMessageBox.information(
                self, "Save model preset",
                f"“{name}” is a shipped preset name. Pick another — a saved run that "
                f"names it must keep resolving to the pipeline it actually ran.")
            return
        try:
            pre, dec = self.ml_preprocess_config(), self.ml_decode_config()
        except ValueError as exc:
            QMessageBox.information(
                self, "Save model preset",
                f"These settings are not a valid pipeline yet, so there is nothing "
                f"safe to store:\n\n{exc}")
            return
        presets = self._load_ml_user_presets()
        if name in presets:
            answer = QMessageBox.question(
                self, "Overwrite preset?",
                f"A saved preset named “{name}” already exists.\n\n"
                f"Overwriting replaces its stored pipeline with the settings currently "
                f"on this tab. This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        # Stamp the NEW name into both configs' `preset` field before storing. Both were
        # built with `preset=ml_preset_name()`, i.e. whatever was selected while the user
        # edited — so without this a saved preset would claim to be the shipped one it
        # was derived from, and a run stored under it would resolve back to that one's
        # numbers. (`deviations_from` ignores `preset`, so the indicator is unaffected.)
        presets[name] = {"preprocess": pre.replace(preset=name).to_dict(),
                         "decode": dec.replace(preset=name).to_dict()}
        self._store_ml_user_presets(presets)
        self._reload_ml_user_presets()
        idx = self.ml_preset.findData(name)
        if idx >= 0:
            self.ml_preset.setCurrentIndex(idx)   # fires _load_ml_preset
        self._ml_preset_chosen = True             # an explicit choice, like picking one

    def _delete_ml_preset(self) -> None:
        """Forget the selected USER preset (the shipped ones are not deletable)."""
        name = str(self.ml_preset.currentData() or "")
        presets = self._load_ml_user_presets()
        if name not in presets:
            return
        presets.pop(name, None)
        self._store_ml_user_presets(presets)
        # The deleted entry was selected, so fall back to the app default rather than
        # leave the form describing a preset that no longer exists.
        self._reload_ml_user_presets()
        if self.ml_preset.findData(name) < 0:
            self.apply_ml_preset(DEFAULT_PRESET)
        self._refresh_ml_preset_buttons()

    # -- ML preset <-> widgets ---------------------------------------------
    def ml_preset_name(self) -> str:
        """The selected preprocessing/decode preset key."""
        return str(self.ml_preset.currentData())

    def apply_ml_preset(self, name: str, *, user_choice: bool = False) -> None:
        """Select ``name`` and repopulate every ML parameter widget from it.

        Superseded keys are accepted and resolved (``"apo_legacy"`` ->
        ``"apo_raw_global"``, the same numbers under a name that no longer implies
        it is the apo pipeline), so restoring an older run does not raise.

        A name the user saved themselves is accepted too: it is not in
        :data:`PRESET_NAMES`, so ``resolve_preset_name`` returns None for it, and the
        user store is consulted before the name is rejected as unknown.

        ``user_choice=True`` records that the user picked this deliberately, which
        stops :meth:`set_model_input_width` from ever overriding it later.
        """
        resolved = resolve_preset_name(name)
        if resolved is None and name in self._load_ml_user_presets():
            resolved = str(name)
        if resolved is None:
            raise ValueError(f"unknown ML preset {name!r}")
        name = resolved
        idx = self.ml_preset.findData(name)
        if idx >= 0 and self.ml_preset.currentIndex() != idx:
            self.ml_preset.setCurrentIndex(idx)   # fires _load_ml_preset
        else:
            self._load_ml_preset()
        if user_choice:
            self._ml_preset_chosen = True

    def _load_ml_preset(self) -> None:
        """Push the selected preset's values into every widget (no deviations left)."""
        name = self.ml_preset_name()
        pre, dec = self._ml_preset_configs(name)
        if pre is None or dec is None:
            return
        self._ml_loading = True
        try:
            # ONE writer for "put a config into these widgets", shared with
            # set_ml_params. A second copy of this list is how a restored run would
            # silently stop restoring the field somebody forgot to add to it.
            self._push_ml_preprocess(pre)
            self._push_ml_decode(dec)
        finally:
            self._ml_loading = False
        self._refresh_ml_baseline_rows()
        self._refresh_ml_preset_status()
        self._refresh_ml_preset_buttons()

    def _refresh_ml_baseline_rows(self) -> None:
        """Show only the baseline parameters the selected estimator actually reads."""
        method = str(self.ml_baseline_method.currentData())
        form = self._ml_preprocess_form
        wanted = {
            "robust_median": {self.ml_baseline_window_s, self.ml_baseline_decim},
            "masked_median": {self.ml_masked_rough_s, self.ml_masked_base_s,
                              self.ml_baseline_decim, self.ml_mask_sigma,
                              self.ml_dilate_ms},
            "savgol": {self.ml_baseline_window_s, self.ml_baseline_poly},
            "none": set(),
        }.get(method, set())
        for w in (self.ml_baseline_window_s, self.ml_baseline_poly,
                  self.ml_baseline_decim, self.ml_masked_rough_s,
                  self.ml_masked_base_s, self.ml_mask_sigma, self.ml_dilate_ms):
            idx, _role = form.getWidgetPosition(w)
            if idx >= 0:
                form.setRowVisible(idx, w in wanted)

    def _refresh_ml_preset_status(self) -> None:
        """Update the 'matches the preset' / 'N deviations' indicator.

        A deviation is not an error, but it IS a change to the contract the model was
        trained under, so it is named field by field rather than silently absorbed.
        """
        name = self.ml_preset_name()
        # Shipped OR user-saved: indexing PREPROCESS_PRESETS here raised on every
        # keystroke once a user preset could be selected.
        base_pre, base_dec = self._ml_preset_configs(name)
        if base_pre is None or base_dec is None:
            return
        molecule = PRESET_MOLECULE.get(name, "unknown")
        if name in PRESET_MOLECULE:
            self.ml_preset_molecule.setText(
                f"{molecule.upper()} — a model trained on the other molecule will not "
                f"behave as measured on this one."
            )
        else:
            # A user preset carries no molecule: this app cannot know what the user's
            # own model was trained on, and printing "UNKNOWN" as if it were a measured
            # fact would be worse than saying who is responsible for the answer.
            self.ml_preset_molecule.setText(
                "not recorded — this is your own saved pipeline, so the molecule it "
                "was built for is yours to track."
            )
        try:
            devs = list(self.ml_preprocess_config().deviations_from(base_pre))
            devs += list(self.ml_decode_config().deviations_from(base_dec))
        except ValueError:
            # A momentarily inconsistent form (e.g. exit above enter mid-edit).
            self.ml_preset_status.setText("custom (values not yet valid)")
            self.ml_preset_status.setStyleSheet("color:#FFB74D;")
            return
        if devs:
            shown = ", ".join(devs[:4]) + (", …" if len(devs) > 4 else "")
            self.ml_preset_status.setText(
                f"CUSTOM — {len(devs)} change(s) from "
                f"{self._ml_preset_label(name)}: {shown}")
            self.ml_preset_status.setStyleSheet("color:#FFB74D;")
        else:
            self.ml_preset_status.setText(
                f"matches {self._ml_preset_label(name)}")
            self.ml_preset_status.setStyleSheet("color:#8b9199;")

    def _on_ml_preset_activated(self) -> None:
        """The user picked a preset by hand — never auto-override it after this."""
        self._ml_preset_chosen = True
        self._refresh_control_level_summary()

    def _on_ml_norm_method_changed(self, *_a) -> None:
        """Gate the two knobs that belong to ONE of the two normalizations.

        ``clip (σ)`` is read only by robust σ; ``mean-centre`` only by max-abs (robust σ
        estimates its own trimmed centre, so leaving the checkbox live would offer a
        switch that does nothing). Disabled rather than hidden, so a stored value stays
        visible.
        """
        robust = str(self.ml_norm_method.currentData()) == "robust_sigma"
        self.ml_norm_clip.setEnabled(robust)
        self.ml_norm_center.setEnabled(not robust)
        self.ml_norm_center.setToolTip(
            "Not used by robust σ — that kernel re-estimates its own trimmed centre."
            if robust else
            "(x − mean) / (max|x − mean| + 1e-8) — the APO notebook's normalize_signal. "
            "The ferritin chain (student_predict) does NOT subtract the mean; it is the "
            "pure rescale x / (max|x| + 1e-8). After a SavGol detrend the residual mean "
            "is tiny (−0.0031 pA on the real −600 mV recording, and it moved 0 of 236 "
            "events), but it is what the apo code executes.")

    def _on_ml_param_changed(self) -> None:
        """A parameter moved. If a PERSON moved it, that is a choice — latch it.

        ``_ml_loading`` is true for exactly one thing: this panel pushing values into its
        own widgets (a preset load, or :meth:`set_ml_params` replaying a stored run). Every
        other emission of these signals came from the user, and a hand-dialled value is as
        deliberate a configuration as a preset picked from the combo — more so, since the
        status line then reads CUSTOM.

        This was the missing half of :attr:`_ml_preset_chosen`. The latch existed, was
        documented as "the user chose this, never override it", and NOTHING set it except
        picking a preset by name — so hand-dialling ``enter`` to 0.35 and then loading a
        2048-wide model rewrote seven parameter groups back to a ferritin preset, replaced
        the CUSTOM status with "matches Ferritin — nb1", and said nothing. The owner has
        twelve 2048-wide models, and 2048 does not even settle the molecule
        (:meth:`set_model_input_width` says so itself), so the guess was overriding a
        measured choice with something that is not evidence.
        """
        if self._ml_loading:
            return
        self._ml_preset_chosen = True
        self._refresh_ml_preset_status()
        self._refresh_control_level_summary()

    @staticmethod
    def _parse_notch_freqs(text: str) -> tuple[float, ...]:
        """Parse the comma/space-separated notch-line box.

        Unparseable or non-positive entries are dropped; an empty result falls back
        to the trained 60/120/180 so the model is never handed an un-notched signal
        because of a typo.
        """
        out: list[float] = []
        for tok in str(text).replace(";", ",").replace("/", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                val = float(tok)
            except ValueError:
                continue
            if val > 0:
                out.append(val)
        return tuple(out) if out else tuple(NOTCH_FREQS)

    def ml_preprocess_config(self) -> MLPreprocessConfig:
        """The ML detector's preprocessing chain as it is currently dialled in."""
        return MLPreprocessConfig(
            preset=self.ml_preset_name(),
            notch_enabled=self.ml_notch_enabled.isChecked(),
            notch_freqs=self._parse_notch_freqs(self.ml_notch_freqs.text()),
            notch_q=float(self.ml_notch_q.value()),
            notch_float32=self.ml_notch_float32.isChecked(),
            baseline_method=str(self.ml_baseline_method.currentData()),
            baseline_window_s=float(self.ml_baseline_window_s.value()),
            baseline_poly=int(self.ml_baseline_poly.value()),
            baseline_decim=int(self.ml_baseline_decim.value()),
            masked_rough_s=float(self.ml_masked_rough_s.value()),
            masked_base_s=float(self.ml_masked_base_s.value()),
            masked_mask_sigma=float(self.ml_mask_sigma.value()),
            masked_dilate_ms=float(self.ml_dilate_ms.value()),
            despike_size=int(self.ml_despike.value()),
            normalize=self.ml_norm_enabled.isChecked(),
            normalize_scope=str(self.ml_norm_scope.currentData()),
            normalize_method=str(self.ml_norm_method.currentData()),
            normalize_clip_sigma=float(self.ml_norm_clip.value()),
            normalize_center=self.ml_norm_center.isChecked(),
            anchor_tail=self.ml_anchor_tail.isChecked(),
            last_start_inclusive=self.ml_last_start_inclusive.isChecked(),
        )

    def ml_decode_config(self) -> MLDecodeConfig:
        """The ML detector's decode contract as it is currently dialled in."""
        return MLDecodeConfig(
            preset=self.ml_preset_name(),
            smoothing_kernel=int(self.ml_smoothing.value()),
            enter_threshold=float(self.ml_enter.value()),
            exit_threshold=float(self.ml_exit.value()),
            merge_radius_ms=float(self.ml_merge_ms.value()),
            min_event_ms=float(self.ml_min_event_ms.value()),
            onset_sigma=float(self.ml_onset_sigma.value()),
            onset_max_ms=float(self.ml_onset_max_ms.value()),
            onset_gap_tol=int(self.ml_onset_gap_tol.value()),
            offset_sigma=float(self.ml_offset_sigma.value()),
            min_depth_sigma=float(self.ml_min_depth_sigma.value()),
            min_depth_pa=float(self.ml_min_depth_pa.value()),
            min_fracblock=float(self.ml_min_fracblock.value()),
        )

    _PIPELINE_NOTE = (
        "The ML detector runs THIS pipeline, not the Preprocess tab. "
        "Pick the preset the model was trained with — the Ferritin presets and the "
        "Apo-ferritin ones are different MOLECULES, not just different settings."
    )

    def set_model_input_width(self, window: int | None) -> None:
        """Default the preset from the loaded model's input width, if it is known.

        2048 is the three ferritin StudentModel notebooks' ``STUDENT_WINDOW``; 3072
        is BOTH apo models. An explicit user choice is NEVER overridden, and an
        unknown or unrecognised width changes nothing.

        A width can only ever NARROW the choice, never identify a pipeline, and the
        note says exactly that rather than implying the guess is a match:

        * 2048 -- all three ferritin presets, and an apo student exists at that width
          too (``ApoStudentModelMerged``), so 2048 is not even proof of the molecule;
        * 3072 -- both apo lineages, which need DIFFERENT preprocessing (the notebook
          detector mean-centres and leaves the tail uncovered; the raw-global one does
          neither) and are indistinguishable by width.

        "An explicit user choice is NEVER overridden" is now true of a hand-dialled
        parameter as well as of a preset picked by name (see
        :meth:`_on_ml_param_changed`), and when the guess IS applied it says so in the
        note instead of leaving the user to notice that seven groups moved.

        NOT OVERRULING A CHOICE IS NOT THE SAME AS SAYING NOTHING. Refusing to overrule
        creates a state that could not persist before: a preset the user chose, and a
        model whose width says it belongs to the OTHER MOLECULE. That is the one thing
        this panel's own note calls out — "the Ferritin presets and the Apo-ferritin ones
        are different MOLECULES" — so when the width is evidence about the molecule and it
        disagrees with the preset in force, the note says so and leaves the choice alone.
        See :data:`_WIDTH_SETTLES_MOLECULE` for when a width counts as evidence.
        """
        self._ml_model_window = int(window) if window else None
        guess = preset_for_window(self._ml_model_window)
        applied_from = ""
        if guess and not self._ml_preset_chosen and guess != self.ml_preset_name():
            # `_ml_preset_label`, not PRESET_LABELS[...]: what is being replaced may be
            # one of the user's own saved presets, which has no shipped label.
            applied_from = self._ml_preset_label(self.ml_preset_name())
            self.apply_ml_preset(guess)

        # The disagreement the latch now lets stand. Read AFTER the auto-apply above, so
        # a width that was allowed to set the preset can never contradict it.
        in_force = self.ml_preset_name()
        # A USER PRESET HAS NO RECORDED MOLECULE, and "no answer" is not "a different
        # answer". Requiring `in_force` to be a shipped preset keeps this a claim about
        # two things the app actually knows; without it, every user preset tripped the
        # mismatch branch — which then indexed PRESET_LABELS / PRESET_MOLECULE by that
        # name and raised KeyError out of set_model_path.
        mismatch = (
            guess is not None
            and self._ml_model_window in _WIDTH_SETTLES_MOLECULE
            and in_force in PRESET_MOLECULE
            and PRESET_MOLECULE.get(guess) != PRESET_MOLECULE.get(in_force)
        )

        note = self._PIPELINE_NOTE
        if mismatch:
            note = (
                f"THIS MODEL AND THIS PRESET DISAGREE ABOUT THE MOLECULE. Every pipeline "
                f"this app knows at input width {self._ml_model_window} is "
                f"{PRESET_MOLECULE[guess].upper()}, and the one about to run is "
                f"“{PRESET_LABELS[in_force]}” — {PRESET_MOLECULE[in_force].upper()}. "
                f"Nothing has been changed: the setting on this tab came from you, or "
                f"from the run being restored, and the width does not overrule it. But if "
                f"this is not deliberate, pick an {PRESET_MOLECULE[guess]} preset before "
                f"running — otherwise the model is fed a signal preprocessed for a "
                f"different molecule, and every event it reports is measured under a "
                f"contract it was not trained on.  " + note
            )
        if applied_from:
            note = (
                f"The preset was changed from “{applied_from}” to "
                f"“{PRESET_LABELS[guess]}” because this model is {self._ml_model_window} "
                f"samples wide and nothing had been dialled in yet. Every ML parameter "
                f"below now holds that preset's value — change any of them and the width "
                f"will not overrule you again.  " + note
            )
        if self._ml_model_window:
            note += f"  (model input width {self._ml_model_window}"
            if self._ml_model_window == 2048:
                note += (
                    " — all three ferritin presets use 2048, so confirm which "
                    "recording yours resembles; an apo student also exists at 2048, "
                    "so the width does not even settle the molecule)"
                )
            elif self._ml_model_window == 3072:
                note += (
                    " — BOTH apo pipelines are 3072, so the width cannot tell them "
                    "apart: the −500/−600/−700 mV presets are the notebook detector "
                    "(student_global_norm_bias_invariant_v0_5ish, mean-centred), "
                    "raw-global is student_raw_global_raw_best. Pick by model file, "
                    "not by width)"
                )
            else:
                note += ")"
        self.ml_pipeline_note.setText(note)

    def model_input_width(self) -> int | None:
        """The loaded model's input window length, when it could be read."""
        return self._ml_model_window

    # -- public API -------------------------------------------------------
    def show_preprocess_tab(self) -> None:
        """Raise Preprocess without changing its current inner section."""
        self.mode_tabs.setCurrentWidget(self._preprocess_page)

    def show_signal_chain(self) -> None:
        """Raise Preprocess and reveal the ordered signal-chain controls."""
        self.show_preprocess_tab()
        self.preprocess_section_selector.setCurrentIndex(0)

    def baseline_drift_method_choices(self) -> tuple[tuple[str, str], ...]:
        """The stable ``(label, method)`` choices for baseline-drift controls.

        A compact, task-focused workspace may mirror the *selection* of the
        Baseline drift stage, but it must not grow a second hand-maintained list of
        kernel methods.  Publishing this immutable tuple keeps that workspace's
        labels in lockstep with the actual signal-chain combo while leaving this
        panel the sole owner of the processing parameters.
        """
        return tuple(self._DETREND_METHODS)

    def show_baseline_drift(self) -> None:
        """Raise Preprocess and reveal the Baseline drift stage's advanced controls.

        While hosted those controls live in the Signal Conditioning window, so this asks
        for that window on the Baseline drift stage and leaves the dock where it is (the
        user is in the baseline workspace)."""
        if not self._chain_editor_hosted:
            self.show_signal_chain()
        self.select_stage("detrend")
        if self._chain_editor_hosted:
            self.openSignalConditioningRequested.emit("detrend")

    # -- the Signal Conditioning window's seam ---------------------------
    def set_signal_chain_hosted(self, hosted: bool = True) -> None:
        """Move the stage editor out of the Preprocess page (it is hosted elsewhere).

        The group box is hidden and the compact card plus a pinned "Open Signal
        Conditioning…" button are shown instead. Every stage widget stays owned here, so
        :meth:`params`, :meth:`set_params` and the setters are unchanged. A standalone
        panel (never hosted) keeps the in-page editor exactly as before.
        """
        hosted = bool(hosted)
        if hosted == self._chain_editor_hosted:
            return
        self._chain_editor_hosted = hosted
        self._signal_filters_box.setVisible(not hosted)
        self.signal_chain_summary.setVisible(hosted)
        self.open_conditioning_button.setVisible(hosted)
        # The main-trace preview's error line stays readable in the dock: last on the
        # card, right under the execution order, so nothing below it can move.
        card = self.signal_chain_summary.layout()
        if hosted:
            card.addWidget(self.preview_error_label)
        else:
            self._filters_layout.insertWidget(
                self._filters_layout.indexOf(self.active_filters_label) + 1,
                self.preview_error_label)
        self.response_plot.setMaximumHeight(16_777_215 if hosted else 190)
        self.set_control_level(self._control_level)
        self._refresh_chain_card()
        for actions in self.findChildren(PinnedActions):
            actions._sync_height()

    def is_signal_chain_hosted(self) -> bool:
        return self._chain_editor_hosted

    def signal_chain_editor_parts(self) -> SignalChainEditorParts:
        """The live editor widgets, grouped for a host to reparent (built once).

        The hosted copy gets a stack whose pages are each only as tall as themselves, so
        the notices placed below it sit right under the shown stage.
        """
        if self._editor_parts is not None:
            return self._editor_parts
        protocol_bar = QWidget()
        protocol_bar.setObjectName("signalChainProtocolBar")
        bar = QVBoxLayout(protocol_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(self._protocol_label)
        row.addWidget(self.protocol_combo, 1)
        verbs = QHBoxLayout()
        for button in (self.protocol_save_btn, self.protocol_rename_btn,
                       self.protocol_delete_btn):
            verbs.addWidget(button)
        verbs.addStretch(1)
        bar.addLayout(row)
        bar.addLayout(verbs)

        old_stack = self.filter_stack
        current = old_stack.currentIndex()
        stack = _CurrentPageStackedWidget()
        stack.setObjectName("signalChainStageStack")
        for page in [old_stack.widget(i) for i in range(old_stack.count())]:
            old_stack.removeWidget(page)
            stack.addWidget(page)
        stack.setCurrentIndex(max(current, 0))
        old_stack.setVisible(False)
        self.filter_stack = stack

        notices = QWidget()
        notices.setObjectName("signalChainNotices")
        nl = QVBoxLayout(notices)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(4)
        for label in (self.highpass_measurement_label, self.normalize_units_label,
                      self.detrend_refusal_label):
            nl.addWidget(label)

        response_view = QWidget()
        response_view.setObjectName("signalChainResponseView")
        rl = QVBoxLayout(response_view)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        rl.addWidget(self.response_chain_check)
        rl.addWidget(self.response_plot, 1)
        rl.addWidget(self.response_note)
        self.response_plot.setMaximumHeight(16_777_215)
        self._set_response_title(self.response_plot.getPlotItem().titleLabel.text)
        self._editor_parts = SignalChainEditorParts(
            protocol_bar, stack, notices, response_view, self.response_chain_check)
        return self._editor_parts

    def set_stage_focus_partner(self, widget: QWidget | None) -> None:
        """The control that takes focus when a stage switch would hide the focused one."""
        self._stage_focus_partner = widget

    @staticmethod
    def stage_keys() -> tuple[str, ...]:
        """The stages in EXECUTION order (``conditioning_recipes.STAGE_KEYS``)."""
        return STAGE_KEYS

    def current_stage_key(self) -> str:
        idx = self.filter_selector.currentIndex()
        return self._EDITOR_STAGE_KEYS[idx] if 0 <= idx < len(self._EDITOR_STAGE_KEYS) else ""

    def select_stage(self, key: str) -> None:
        """Raise ``key``'s stage page through the stage selector (unknown keys ignored)."""
        if key in self._EDITOR_STAGE_KEYS:
            self.filter_selector.setCurrentIndex(self._EDITOR_STAGE_KEYS.index(key))

    def _stage_checkbox(self, key: str) -> QCheckBox | None:
        if key not in self._EDITOR_STAGE_KEYS:
            return None
        return self._filter_specs[self._EDITOR_STAGE_KEYS.index(key)][1]

    def set_stage_enabled(self, key: str, on: bool) -> None:
        """Tick or untick ``key``'s own enable checkbox (the only state there is)."""
        box = self._stage_checkbox(key)
        if box is not None and box.isChecked() != bool(on):
            box.setChecked(bool(on))

    def sample_rate(self) -> float | None:
        return self._fs

    def add_notch_lines(self, freqs: Sequence[float]) -> str:
        """Add spectrum lines to the notch through the form's own setters; the outcome.

        On: the lines IN FORCE (a live series spelled out, so nothing it removed is lost)
        gain the new ones. Off with an explicit list kept: the new lines join that list
        and the notch STAYS off — switching it on would bring back lines the user
        switched off, and replacing the list would lose them; the outcome says so. Off
        with the series spelling, nothing is kept: the new lines become the list and the
        notch is turned on (the series spins keep their values). A line the notch already
        removes is skipped; a list past :attr:`NOTCH_MAX_LINES` is refused and the form
        is left as it was."""
        on = self.notch_enabled.isChecked()
        current = self.notch_lines_choice()
        keep_off = not on and current is not None
        if not on and current is None:
            current = []
        elif current is None:
            f0, n = float(self.notch_freq_hz.value()), int(self.notch_harmonics.value())
            try:
                current = (filters.notch_frequencies(float(self._fs), f0, harmonics=n)
                           if self._fs else [f0 * h for h in range(1, n + 1)])
            except ValueError:          # every line at or above Nyquist: none in force
                current = []
        merge = merge_notch_lines(current, freqs, q=float(self.notch_q.value()),
                                  max_lines=self.NOTCH_MAX_LINES)
        if merge.lines is None:
            return merge.refusal
        if not merge.added:
            listed = ", ".join(f"{f:g}" for f in merge.skipped)
            verb = "is" if len(merge.skipped) == 1 else "are"
            if keep_off:
                return (f"Notch still OFF: {listed} Hz {verb} already in the list it keeps "
                        "— tick Notch to apply it.")
            return f"Notch: {listed} Hz {verb} already removed — nothing was added."
        self.set_notch_lines(merge.lines)
        if self.notch_lines_choice() != merge.lines:
            return self._notch_refusal or "Refused: the notch list was not accepted."
        if not keep_off:
            self.set_notch_enabled(True)
        self._emit_params()
        added = ", ".join(f"{f:g}" for f in merge.added)
        if keep_off:
            return (f"Notch still OFF: added {added} Hz to the list it keeps, now "
                    f"{_notch_lines_text(merge.lines)} — tick Notch to apply them.")
        return (f"Notch: added {added} Hz — the explicit line list is now "
                f"{_notch_lines_text(merge.lines)}.")

    def lowpass_design_hz(self) -> float:
        """The low-pass cutoff box, which :meth:`params` reports as None while it is off."""
        return float(self.filter_cutoff_hz.value())

    @classmethod
    def _stored_recipe(cls, raw: Any) -> dict[str, Any] | None:
        """A stored protocol as a complete recipe, or None when it cannot be one (a
        corrupt store entry means that protocol is unreadable, never a crash)."""
        if not isinstance(raw, Mapping):
            return None
        recipe, _why = validate_recipe({k: v for k, v in raw.items() if k in CHAIN_KEYS},
                                       max_notch_lines=cls.NOTCH_MAX_LINES, ranges=False)
        return recipe

    def protocol_library_report(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """``(library, unreadable names)``: the user's own protocols as complete recipes
        (the shipped three excluded), and the stored entries that could not be read."""
        library: dict[str, dict[str, Any]] = {}
        unreadable: list[str] = []
        for name, raw in sorted(self._load_protocols().items()):
            if name in self._BUILTIN_PROTOCOLS:
                continue
            recipe = self._stored_recipe(raw)
            if recipe is None:
                unreadable.append(str(name))
            else:
                library[name] = recipe
        return library, unreadable

    def protocol_library(self) -> dict[str, dict[str, Any]]:
        """The user's own readable protocols as complete recipes (shipped three excluded)."""
        return self.protocol_library_report()[0]

    def merge_protocols(self, imported: Mapping[str, Mapping[str, Any]], *,
                        refused: Sequence[str] = ()) -> ProtocolMergeReport:
        """Add imported protocols to the user's library, never overwriting one (not even
        a stored entry that cannot be read)."""
        library, unreadable = self.protocol_library_report()
        reserved = {**self._BUILTIN_PROTOCOLS, self._PROTO_MANUAL: None,
                    **{name: None for name in unreadable}}
        merged, report = merge_imported_protocols(
            library, imported, reserved=reserved, refused=refused)
        if report.added or report.renamed:
            current = self.protocol_combo.currentText()
            self._store_protocols({**self._load_protocols(), **merged})
            self._load_protocols_into_combo()
            self._select_protocol(current if self.protocol_combo.findText(current) >= 0
                                  else self._PROTO_MANUAL)
        return report

    def current_protocol(self) -> tuple[str | None, dict[str, Any] | None]:
        """``(name, complete recipe)`` of the protocol the combo names, or ``(None, None)``."""
        name = self.protocol_combo.currentText()
        recipe = self._protocol_recipe(name)
        return (name, recipe) if recipe is not None else (None, None)

    def _protocol_recipe(self, name: str) -> dict[str, Any] | None:
        if not name or name == self._PROTO_MANUAL:
            return None
        if name in self._BUILTIN_PROTOCOLS:
            return complete_recipe(self._BUILTIN_PROTOCOLS[name])
        return self._stored_recipe(self._load_protocols().get(name))

    def set_chain_status(self, text: str, warn: bool = False) -> None:
        """The chain-versus-detection sentence the card's first line carries."""
        self._chain_status_text = str(text or "")
        self._chain_status_warn = bool(warn)
        self._refresh_chain_card()

    def chain_card_text(self) -> dict[str, str]:
        """The card's full texts, for tests and for the window's status line."""
        return {
            "status": self.chain_status_note.text(),
            "warning": self.chain_warning_note.text(),
            "protocol": self.chain_protocol_note.text(),
            "scope": self.chain_scope_note.text(),
            "order": self.chain_order_note.text(),
        }

    def _refresh_chain_card(self, *_a) -> None:
        if not hasattr(self, "signal_chain_summary"):
            return
        critical = []
        if self.normalize_enabled.isChecked():
            critical.append(self._NORMALIZE_CRITICAL)
        if self.hp_enabled.isChecked():
            critical.append(self._HIGHPASS_CRITICAL)
        if self.detrend_refusal_label.text() and not self.detrend_refusal_label.isHidden():
            critical.append(self.detrend_refusal_label.text().split(". ")[0])
        name = self.protocol_combo.currentText() or self._PROTO_MANUAL
        texts = (
            (self.chain_status_note, self._chain_status_text),
            (self.chain_warning_note, " · ".join(critical)),
            (self.chain_protocol_note, f"Protocol: {name}"),
            (self.chain_order_note, self.active_filters_label.text()),
        )
        for note, text in texts:
            if note.text() != text:
                note.setText(text)
        style = "color:#FFB74D;" if self._chain_status_warn else "color:#8b9199;"
        if self.chain_status_note.styleSheet() != style:
            self.chain_status_note.setStyleSheet(style)

    def show_open_pore_baseline(self) -> None:
        """Raise Preprocess and reveal the open-pore baseline controls."""
        self.show_preprocess_tab()
        self.preprocess_section_selector.setCurrentIndex(1)

    def show_detection_tab(self) -> None:
        """Raise the Detection (Schmitt threshold / polarity / run) tab."""
        self.mode_tabs.setCurrentWidget(self._detection_page)

    def show_ml_tab(self) -> None:
        """Raise the ML detector tab."""
        if hasattr(self, "control_level_selector"):
            self.set_control_level("advanced")
        self.mode_tabs.setCurrentWidget(self._ml_page)

    def show_measure_tab(self) -> None:
        """Raise the Manual measurement (cursors) tab.

        Mirrors the other three so ``Analysis ▸ Measure event`` can REVEAL what it just
        turned on. Raising a tab is pure layout — it never touches the measurement tool's
        active state (the cursors live on the shared TraceView and outlive any tab switch).
        """
        self.mode_tabs.setCurrentWidget(self._measure_page)

    def set_measure_widget(self, widget: QWidget) -> None:
        """Install the owner's manual-measurement widget as the Manual measurement page.

        The panel takes the widget as a CHILD (Qt reparents it) but is not its owner in any
        other sense: it never connects to it, never reads it and never gates it. Everything
        the tool does still flows through :class:`~nanoscope.gui.analysis_panel.AnalysisPanel`,
        which built it and holds the only reference that matters.

        Callers pass the SCROLL-WRAPPED widget, not the panel itself — see
        ``analysis_panel._scroll_wrap``: the measurement tool's three-button row has a hard
        ~438 px layout minimum and a page's minimum width propagates through the tab widget
        into ``QDockWidget::minimumSize()``, which is the bound the Inspector dock cannot be
        dragged past.
        """
        # A caller may pass the older scroll wrapper; preserve the live panel and
        # provide the same bounded viewport used by the other modes.
        from PySide6.QtWidgets import QScrollArea
        panel = widget.takeWidget() if isinstance(widget, QScrollArea) else widget
        if panel is not widget:
            widget.deleteLater()
        actions = getattr(panel, 'action_widget', None)
        if actions is not None:
            self._measure_actions.layout().addWidget(actions)
            self._measure_action_scroll._sync_height()
        scroll = InspectorScrollArea(panel)
        self._measure_layout.addWidget(scroll, 1)
        self.page_scrolls[self._measure_page] = scroll

    def set_baseline_widget(self, widget: QWidget) -> None:
        """Install the owner's controls in the Open-pore baseline subsection.

        Install-only, exactly like :meth:`set_measure_widget`: the panel reparents the
        widget and never connects to it, reads it or gates it. The subsection label
        keeps this reference estimator distinct from the Baseline drift filter stage.
        """
        self._baseline_slot.addWidget(widget)
        actions = getattr(widget, 'action_widget', None)
        if actions is not None:
            self._baseline_actions_layout.addWidget(actions)
            for scroll in self._preprocess_page.findChildren(PinnedActions):
                scroll._sync_height()

    def set_recording(self, fs: float | None) -> None:
        """Tell the panel the sampling rate so the rise-time readout is real.

        Tolerates ``fs is None`` (the legacy-TDMS rate-unknown case): the readout
        says so rather than raising. Button enablement is a separate concern —
        :meth:`set_runnable` — because it also depends on the channel kind.

        This also re-ranges every frequency control against the new Nyquist, so a value
        the kernel would reject cannot be dialled in at all.
        """
        self._fs = float(fs) if fs else None
        self._apply_nyquist_ranges()
        self._refresh_notch_mode()
        self._update_rise_time()
        self._update_response_plot()

    # -- fs-aware spin ranges ---------------------------------------------
    @staticmethod
    def _nyquist_cap(fs: float, decimals: int) -> float:
        """The largest value a spin with ``decimals`` can hold that is still < fs/2.

        The kernels raise at ``cutoff >= fs/2`` (there is nothing to filter there), so the
        cap has to be strictly below Nyquist AND representable at the widget's precision —
        otherwise the spin's own maximum is a value that fails.
        """
        step = 10.0**-decimals
        cap = round(fs / 2.0 - step, decimals)
        # Defensive: rounding could land back on Nyquist. Bounded so a pathological rate
        # can never spin here.
        for _ in range(4):
            if cap < fs / 2.0:
                break
            cap -= step
        return min(cap, np.nextafter(fs / 2.0, 0.0))

    def _apply_nyquist_ranges(self) -> None:
        """Clamp the frequency spins to the active recording's Nyquist.

        The three ceilings used to be static and mutually inconsistent (5 MHz, 5 MHz,
        100 kHz) regardless of the recording: on a 50 kHz trace a user could dial a
        200x-Nyquist cutoff, and the only feedback was a failed run, a silently vanished
        preview, or nothing at all. With fs known, each maximum becomes the real one.
        Qt clamps an out-of-range CURRENT value as a side effect of setRange, which is the
        intent: a stale 100 kHz cutoff carried over from a faster recording is pulled down
        to something legal instead of poisoning the next run.

        With no rate (legacy TDMS) the static fallbacks stay — detection is gated off
        anyway, and inventing a ceiling would be a lie.
        """
        fs = self._fs
        if fs is None:
            self.filter_cutoff_hz.setRange(*self._STATIC_LP_RANGE)
            self.hp_cutoff_hz.setRange(*self._STATIC_HP_RANGE)
            self.notch_freq_hz.setRange(*self._STATIC_NOTCH_RANGE)
            return
        self.filter_cutoff_hz.setRange(
            min(self._STATIC_LP_RANGE[0], self._nyquist_cap(fs, 1)),
            self._nyquist_cap(fs, self.filter_cutoff_hz.decimals()),
        )
        self.hp_cutoff_hz.setRange(
            min(self._STATIC_HP_RANGE[0], self._nyquist_cap(fs, 2)),
            self._nyquist_cap(fs, self.hp_cutoff_hz.decimals()),
        )
        self.notch_freq_hz.setRange(
            min(self._STATIC_NOTCH_RANGE[0], self._nyquist_cap(fs, 1)),
            self._nyquist_cap(fs, self.notch_freq_hz.decimals()),
        )

    def set_preview_error(self, message: str) -> None:
        """Show why the processed-trace preview could not be computed (``""`` clears it).

        The owner (AnalysisPanel) pushes the kernel's own ValueError text in, so the user
        learns *which* stage is misconfigured instead of watching the curve vanish."""
        self.preview_error_label.setText(
            f"processed-trace preview unavailable: {message}" if message else "")
        self.preview_error_label.setVisible(bool(message))

    def set_runnable(self, runnable: bool, reason: str = "") -> None:
        """Enable/disable the action buttons, showing an honest reason when not.

        The AnalysisPanel computes ``runnable`` from ``accepts_kinds`` (channel is
        CURRENT) AND ``requires_time_base`` (``sample_rate_hz`` is not None). When
        a detector cannot run, the buttons go grey and ``reason`` explains why,
        rather than fabricating a dwell time from a nonexistent rate.
        """
        moved = self.detect_button.isEnabled() != bool(runnable)
        for btn in (self.detect_button, self.sweep_button,
                    self.null_button, self.recovery_button):
            btn.setEnabled(bool(runnable))
        self._gate_label.setText("" if runnable else reason)
        self._gate_label.setVisible(bool(reason) and not runnable)
        self._gate_host.setVisible(bool(reason) and not runnable)
        if moved:
            self.runGateChanged.emit(bool(runnable))

    # -- ML detector ------------------------------------------------------
    def browse_model(self) -> None:
        """Open a file dialog to pick a ``.keras`` model (default dir ~/Downloads).

        This is a pure view action -- it only records the chosen PATH and updates the
        label; the TensorFlow model is not loaded here (that happens off-thread at run
        time). Both the panel's own button and MainWindow's ``Load ML model…`` menu
        action call this."""
        start_dir = str(Path.home() / "Downloads")
        if not os.path.isdir(start_dir):
            start_dir = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Keras model", start_dir,
            "Keras model (*.keras);;All files (*)")
        if not path:
            return
        # Refuse a teacher HERE, at pick time. The authoritative check lives in
        # load_keras_model, but that runs off-thread at RUN time -- so without this the
        # user picks a teacher, watches it accepted into the label, chooses a range,
        # waits out the inference and only then learns it was never going to run. This
        # gate is pure zipfile+json (no TensorFlow on the GUI thread) and errs toward
        # allowing: a config it cannot read falls through to the real check.
        if ml_detect.is_teacher_model(path):
            QMessageBox.information(
                self, "Student models only",
                f"{os.path.basename(path)} is a TEACHER model "
                f"({ml_detect.TEACHER_CHANNELS}-feature input).\n\n"
                f"This app runs STUDENT models only — a teacher is the distillation "
                f"source that labels training data, not the detector meant to be "
                f"deployed. Load the matching student export instead.")
            return
        self.set_model_path(path)

    def _on_model_combo_changed(self, index: int) -> None:
        """A pick in the loaded-models combo IS a model switch — same door as browsing."""
        if self._ml_combo_updating or index < 0:
            return
        path = self.ml_model_combo.itemData(index)
        if path and path != self._model_path:
            self.set_model_path(str(path))

    def _sync_model_combo(self) -> None:
        """Reconcile the loaded-models combo with the active path: add it if new,
        select it, and keep the empty state honest. Guarded so selecting an entry
        (which calls set_model_path, which calls back here) cannot loop."""
        self._ml_combo_updating = True
        try:
            combo = self.ml_model_combo
            if self._model_path is None:
                # TWO READOUTS, ONE ANSWER. Doing nothing here left the last model
                # SELECTED in the combo while `ml_model_label` read "none": the panel
                # then showed two different answers to "which model is active?", and
                # nothing on screen said which one a Run would use. The fix is a
                # non-data sentinel at index 0 — itemData None, so the
                # `_on_model_combo_changed` handler ignores a pick of it — selected
                # whenever there is no model. The loaded history stays BELOW it, so
                # clearing a model does not throw the session's list away.
                if combo.count() == 0 or combo.itemData(0) is not None:
                    combo.insertItem(0, self._ML_NO_MODEL_ITEM)
                combo.setCurrentIndex(0)
                # Only worth enabling while there is something to switch TO.
                combo.setEnabled(combo.count() > 1)
                return
            idx = next((i for i in range(combo.count())
                        if combo.itemData(i) == self._model_path), -1)
            if idx < 0:
                combo.addItem(os.path.basename(self._model_path), self._model_path)
                combo.setItemData(combo.count() - 1, self._model_path,
                                  Qt.ToolTipRole)
                idx = combo.count() - 1
            # A model IS active, so the empty-state sentinel is no longer true; drop it
            # (and shift the index it was occupying) rather than leave a "none loaded"
            # row sitting above a live selection.
            if combo.count() > 1 and combo.itemData(0) is None:
                combo.removeItem(0)
                idx -= 1
            combo.setEnabled(True)
            combo.setCurrentIndex(idx)
        finally:
            self._ml_combo_updating = False

    def set_model_path(self, path: str | None) -> None:
        """Record the loaded model's PATH (or None to clear) and update the label +
        Run enablement. Emits ``mlModelChanged`` so the owner can re-gate its menu."""
        # A missing/changed model restored from a project is carried in this sidecar so
        # unrelated form edits do not erase its audit trail. Any ordinary model selection
        # or clear replaces that reference. The restore adapter holds the narrow guard
        # while clearing only the live runtime path.
        if (not getattr(self, "_nanoscope_preserve_unresolved_model_reference", False)
                and hasattr(self, "_nanoscope_unresolved_model_reference")):
            delattr(self, "_nanoscope_unresolved_model_reference")
        self._model_path = str(path) if path else None
        self.ml_model_label.setText(
            os.path.basename(self._model_path) if self._model_path else "none")
        self._sync_model_combo()
        # Read the model's input width from the .keras zip WITHOUT importing
        # TensorFlow (peek_model_window is pure zipfile+json, safe on the GUI thread)
        # so the preprocessing preset can default sensibly. Unknown -> nothing changes.
        self.set_model_input_width(
            ml_detect.peek_model_window(self._model_path) if self._model_path else None)
        self._refresh_ml_enabled()
        self.mlModelChanged.emit(self._model_path or "")

    def model_path(self) -> str | None:
        """The loaded model's path, or None when no model is loaded."""
        return self._model_path

    def set_ml_runnable(self, runnable: bool, reason: str = "") -> None:
        """Tell the panel whether the CURRENT channel can be ML-detected (a CURRENT
        channel with a known sample rate -- the same gate the Schmitt buttons use).
        The Run button ANDs this with 'a model is loaded' and 'TensorFlow available'."""
        self._ml_channel_ok = bool(runnable)
        self._ml_channel_reason = reason
        self._refresh_ml_enabled()

    def ml_params(self) -> dict[str, Any]:
        """The ML detector's current form parameters as a plain dict.

        ``model_path`` is the loaded model (or None); ``enter``/``exit`` are the
        hysteresis thresholds; ``range_source`` is ``'visible'`` or ``'whole'``. The
        AnalysisPanel resolves ``range_source`` into concrete sample indices before
        the run (it owns the trace view that knows the visible window).

        ``ml_preprocess`` / ``ml_decode`` carry the model's OWN pipeline as JSON-safe
        dicts (so a stored result stays readable), plus the preset they came from and
        whether the user deviated from it. The general Preprocess tab's FILTER chain is
        not involved in an ML run — but its accepted open-pore windows ARE: the owning
        AnalysisPanel merges ``BaselinePanel.params()`` into the run before launch, the
        same way it does for the Schmitt detector, because MLDetect reads ``baseline_*``
        for the measurement baseline every depth is referenced to. This dict carries no
        ``baseline_*`` key precisely so that merge is the single author of them."""
        pre = self.ml_preprocess_config()
        dec = self.ml_decode_config()
        name = self.ml_preset_name()
        # Shipped OR user-saved. This used to index PREPROCESS_PRESETS directly, which
        # KeyErrors on a user preset — and it is on the path of every ML run, so the
        # failure would have been "the Run button raises", not a cosmetic one. A preset
        # neither store knows leaves the deviation list empty rather than raising: there
        # is nothing to compare against, and "0 deviations" is then honest.
        base_pre, base_dec = self._ml_preset_configs(name)
        devs = list(pre.deviations_from(base_pre)) if base_pre is not None else []
        if base_dec is not None:
            devs += list(dec.deviations_from(base_dec))
        return {
            "model_path": self._model_path,
            "enter": float(self.ml_enter.value()),
            "exit": float(self.ml_exit.value()),
            "range_source": "visible" if self.ml_range.currentIndex() == 0 else "whole",
            "ml_preset": name,
            "ml_preprocess": pre.to_dict(),
            "ml_decode": dec.to_dict(),
            "ml_preset_customized": bool(devs),
            "ml_preset_deviations": devs,
        }

    def set_ml_params(self, params: Mapping[str, Any]) -> None:
        """Seed the whole ML tab from a stored ML run — :meth:`ml_params`' inverse.

        The exact counterpart of :meth:`set_params` for the other detector, and it exists
        for the exact same reason: a stored detection carries the settings it ran under,
        and without a way to put them back, reopening a project showed the panel's
        DEFAULTS over a table produced by completely different numbers. Measured before
        this existed: a result stored under ``notebook3`` / ``enter=0.9`` came back onto a
        panel reading ``notebook1`` / ``enter=0.5`` / ``smoothing=9`` — and Run ML from
        that form would have detected something else and replaced the table with it.

        ``ml_preprocess`` / ``ml_decode`` are the authority; ``ml_preset`` only chooses
        the baseline the deviations are reported against. So the preset is applied FIRST
        (which fills every widget) and the stored configs are laid over it, which
        reproduces a customised run exactly rather than snapping it back to its preset.

        Absent keys are left alone, as in :meth:`set_params`. ``{}`` — a v1/v2 project
        that stored no params — changes nothing, because a form showing values that at
        least came from the user beats one showing values invented here.

        Seeding is NOT a user choice, so it runs with the deviation indicator suppressed;
        but the result IS a deliberate configuration, so ``_ml_preset_chosen`` is latched
        afterwards: a model loaded next must not overrule the run that is on screen.
        """
        if not params:
            return
        pre = MLPreprocessConfig.from_dict(params.get("ml_preprocess"))
        dec = MLDecodeConfig.from_dict(params.get("ml_decode"))
        name = resolve_preset_name(params.get("ml_preset"))
        if name is None and pre is not None:
            name = resolve_preset_name(pre.preset)
        if name is None and dec is not None:
            name = resolve_preset_name(dec.preset)
        # A run stored under one of the USER's own presets resolves through neither
        # PRESET_NAMES nor the aliases; consult their store before giving up on naming
        # it. The parameters below are restored either way — this only decides whether
        # the combo can say which preset the run belonged to.
        if name is None:
            stored = self._load_ml_user_presets()
            for candidate in (params.get("ml_preset"),
                              pre.preset if pre is not None else None,
                              dec.preset if dec is not None else None):
                if candidate and str(candidate) in stored:
                    name = str(candidate)
                    break

        if name is not None:
            self.apply_ml_preset(name)

        self._ml_loading = True
        try:
            if pre is not None:
                self._push_ml_preprocess(pre)
            if dec is not None:
                self._push_ml_decode(dec)
            # The two thresholds live in the decode config; the top-level keys are the
            # fallback for a run stored before it existed. enter first — the live clamp
            # keeps exit <= enter, exactly as in _load_ml_preset.
            if dec is None:
                if "enter" in params:
                    self.ml_enter.setValue(float(params["enter"]))
                if "exit" in params:
                    self.ml_exit.setValue(float(params["exit"]))
            if "range_source" in params:
                self.ml_range.setCurrentIndex(
                    0 if str(params["range_source"]) == "visible" else 1)
        finally:
            self._ml_loading = False
        self._refresh_ml_baseline_rows()
        self._refresh_ml_preset_status()
        self._ml_preset_chosen = True
        self._refresh_control_level_summary()

    def _push_ml_preprocess(self, pre: MLPreprocessConfig) -> None:
        """Put one preprocessing config into the widgets. Caller owns ``_ml_loading``."""
        self.ml_notch_enabled.setChecked(pre.notch_enabled)
        self.ml_notch_freqs.setText(", ".join(f"{f:g}" for f in pre.notch_freqs))
        self.ml_notch_q.setValue(float(pre.notch_q))
        self.ml_notch_float32.setChecked(pre.notch_float32)
        bidx = self.ml_baseline_method.findData(pre.baseline_method)
        if bidx >= 0:
            self.ml_baseline_method.setCurrentIndex(bidx)
        self.ml_baseline_window_s.setValue(float(pre.baseline_window_s))
        self.ml_baseline_poly.setValue(int(pre.baseline_poly))
        self.ml_baseline_decim.setValue(int(pre.baseline_decim))
        self.ml_masked_rough_s.setValue(float(pre.masked_rough_s))
        self.ml_masked_base_s.setValue(float(pre.masked_base_s))
        self.ml_mask_sigma.setValue(float(pre.masked_mask_sigma))
        self.ml_dilate_ms.setValue(float(pre.masked_dilate_ms))
        self.ml_despike.setValue(int(pre.despike_size))
        self.ml_norm_enabled.setChecked(pre.normalize)
        midx = self.ml_norm_method.findData(pre.normalize_method)
        if midx >= 0:
            self.ml_norm_method.setCurrentIndex(midx)
        self.ml_norm_clip.setValue(float(pre.normalize_clip_sigma))
        self.ml_norm_center.setChecked(pre.normalize_center)
        sidx = self.ml_norm_scope.findData(pre.normalize_scope)
        if sidx >= 0:
            self.ml_norm_scope.setCurrentIndex(sidx)
        self.ml_anchor_tail.setChecked(pre.anchor_tail)
        self.ml_last_start_inclusive.setChecked(pre.last_start_inclusive)

    def _push_ml_decode(self, dec: MLDecodeConfig) -> None:
        """Put one decode config into the widgets. Caller owns ``_ml_loading``."""
        self.ml_smoothing.setValue(int(dec.smoothing_kernel))
        # enter first, then exit — the live clamp keeps exit <= enter.
        self.ml_enter.setValue(float(dec.enter_threshold))
        self.ml_exit.setValue(float(dec.exit_threshold))
        self.ml_merge_ms.setValue(float(dec.merge_radius_ms))
        self.ml_min_event_ms.setValue(float(dec.min_event_ms))
        self.ml_onset_sigma.setValue(float(dec.onset_sigma))
        self.ml_onset_max_ms.setValue(float(dec.onset_max_ms))
        self.ml_onset_gap_tol.setValue(int(dec.onset_gap_tol))
        self.ml_offset_sigma.setValue(float(dec.offset_sigma))
        self.ml_min_depth_sigma.setValue(float(dec.min_depth_sigma))
        self.ml_min_depth_pa.setValue(float(dec.min_depth_pa))
        self.ml_min_fracblock.setValue(float(dec.min_fracblock))

    def _refresh_ml_enabled(self) -> None:
        """Recompute the Run-ML button enablement + gate message.

        Enabled only when TensorFlow is importable AND a model is loaded AND the
        channel gate is satisfied. The message names the FIRST unmet condition so the
        user knows what to do next; TensorFlow-absent disables loading too, since
        there is nothing a loaded model could run on."""
        tf_ok = ml_detect.TENSORFLOW_AVAILABLE
        has_model = self._model_path is not None
        self.ml_load_button.setEnabled(tf_ok)
        self.ml_detect_button.setEnabled(tf_ok and has_model and self._ml_channel_ok)

        if not tf_ok:
            msg = ("TensorFlow is not installed, so ML detection is unavailable. "
                   "Install tensorflow (2.15) to enable it.")
        elif not self._ml_channel_ok:
            msg = self._ml_channel_reason or (
                "Select a CURRENT channel with a known sample rate to run ML detection.")
        elif not has_model:
            msg = "Load a .keras model to run ML detection."
        else:
            msg = ""
        self.ml_gate_label.setText(msg)
        self.ml_gate_label.setVisible(bool(msg))
        self._ml_gate_host.setVisible(bool(msg))

    def set_polarity_evidence(self, evidence: dict[str, Any]) -> None:
        """Show the measured polarity evidence from ``detect_polarity`` (carried in
        the detection result's provenance under ``polarity_evidence``)."""
        n_down = evidence.get("n_down", 0)
        n_up = evidence.get("n_up", 0)
        verdict = evidence.get("verdict", "undetermined")
        physical = evidence.get("physical", "unknown")
        direction = evidence.get("direction", "?")
        conf = evidence.get("confidence", 0.0)
        if verdict == "undetermined":
            self.polarity_readout.setText(
                f"undetermined: {n_down} down / {n_up} up — no clear excursions"
            )
        else:
            self.polarity_readout.setText(
                f"{physical} ({direction}): {n_down} down / {n_up} up "
                f"(confidence {conf:.2f}, {verdict})"
            )

    def polarity_choice(self) -> int | None:
        """The user's polarity override, or None for 'auto' (measure it)."""
        idx = self.polarity_override.currentIndex()
        if idx == 0:
            return None
        return -1 if idx == 1 else 1

    # -- low-pass / high-pass design --------------------------------------
    def filter_kind_choice(self) -> str:
        """The low-pass kind string :func:`~nanoscope.analysis.filters.lowpass` expects
        (``'bessel'`` / ``'butter'`` / ``'gaussian'``)."""
        return filters.KINDS[self.filter_kind.currentIndex()]

    def set_filter_kind(self, kind: str) -> None:
        """Select the low-pass kind by its kernel string. Unknown strings are ignored."""
        if kind in filters.KINDS:
            self.filter_kind.setCurrentIndex(filters.KINDS.index(kind))

    def set_filter_order(self, order: int) -> None:
        """Set the low-pass order (ignored by the Gaussian kind, which has none)."""
        self.filter_order.setValue(int(order))

    def hp_kind_choice(self) -> str:
        """The high-pass kind string :func:`~nanoscope.analysis.filters.highpass`
        expects (``'butter'`` / ``'bessel'``)."""
        return filters.HP_KINDS[self.hp_kind.currentIndex()]

    def set_hp_kind(self, kind: str) -> None:
        """Select the high-pass kind by its kernel string. Unknown strings are ignored."""
        if kind in filters.HP_KINDS:
            self.hp_kind.setCurrentIndex(filters.HP_KINDS.index(kind))

    def set_hp_order(self, order: int) -> None:
        """Set the high-pass order."""
        self.hp_order.setValue(int(order))

    # -- detector selection ------------------------------------------------
    def detector_choice(self) -> str:
        """The selected detector's ANALYSIS_REGISTRY module id.

        ``"detect.schmitt"`` or ``"detect.cusum"``. This is what :meth:`params` emits
        under ``"detector"`` and what the owning panel must start -- it is a module id
        rather than a nickname precisely so the panel and the registry cannot disagree
        about which detector "cusum" means.
        """
        return str(self.detector_combo.currentData())

    def set_detector(self, module_id: str) -> None:
        """Select a detector by its module id. Unknown ids are ignored.

        Ignoring is safe HERE (unlike :meth:`set_detrend_method`) because the id is not
        a parameter of a run: a restored project that names a detector this build does
        not have already cannot re-run it, and the events it produced are on screen with
        their own provenance either way. What must not happen -- and does not -- is the
        OTHER detector's thresholds silently reaching the selected one; that is prevented
        structurally by `modules._DETECTOR_PRIVATE_KEYS`, not by this setter.
        """
        idx = self.detector_combo.findData(str(module_id))
        if idx >= 0:
            self.detector_combo.setCurrentIndex(idx)

    def _on_detector_changed(self, *_a) -> None:
        """Raise the selected detector's threshold page and re-seed the shared knobs.

        THE RE-SEED IS THE PART WORTH READING. ``min dwell`` and ``merge gap`` are shared
        widgets whose sensible values differ by 25-50x between the two detectors (the
        Schmitt path merges at 0.02 ms and keeps 0.02 ms events; the CUSUM calibration
        used 0.50 ms and 1.00 ms). Leaving the Schmitt numbers in place when the user
        switches would run the owner's algorithm at settings his calibration never saw,
        and silently. So the CURRENT value is compared against the OTHER detector's
        default and replaced only when it still IS that default -- a number the user
        moved is theirs and is never overwritten.
        """
        module_id = self.detector_choice()
        pages = [m for _l, m in self._DETECTORS]
        self.detector_stack.setCurrentIndex(pages.index(module_id))

        mine = self._DETECTOR_DEFAULTS.get(module_id, {})
        for other_id, theirs in self._DETECTOR_DEFAULTS.items():
            if other_id == module_id:
                continue
            if "min_dwell_s" in mine and _near(
                    self.min_dwell_s.value(), theirs.get("min_dwell_s")):
                self.min_dwell_s.setValue(float(mine["min_dwell_s"]))
            if "merge_gap_s" in mine and _near(
                    self.merge_gap_ms.value(), theirs.get("merge_gap_s", 0.0) * 1e3):
                self.merge_gap_ms.setValue(float(mine["merge_gap_s"]) * 1e3)

        if module_id == "detect.cusum":
            self.detector_note.setText(
                "k and h are DIMENSIONLESS — multiples of the MAD noise sigma, the same "
                "sigma the table's snr column is divided by. They are not picoamps and "
                "they do not rescale with the recording's gain.")
        else:
            self.detector_note.setText(
                "k_in and k_out are DIMENSIONLESS — multiples of the MAD noise sigma.")

    def detrend_method_choice(self) -> str:
        """The baseline-drift estimator method string, as
        :func:`~nanoscope.analysis.baseline.remove_baseline_wander` expects it
        (one of :data:`nanoscope.analysis.baseline.METHODS` except ``'constant'``)."""
        return self._DETREND_METHODS[self.detrend_method.currentIndex()][1]

    def set_detrend_method(self, method: str) -> str | None:
        """Select the baseline-drift estimator by its method string.

        Returns ``None`` on success, or a one-sentence complaint when ``method`` is not
        one this combo offers -- in which case the current choice is LEFT ALONE and the
        panel raises a visible refusal banner (see :meth:`_detrend_refusal`).

        IT MUST NOT SILENTLY IGNORE AN UNKNOWN NAME, and that is the difference between
        this and its sibling :meth:`set_normalize_mode`. Ignoring a normalize mode keeps
        the SAME estimator running; ignoring a detrend method runs a DIFFERENT ONE. Three
        of the seven kernel methods used to be missing from this combo, so a protocol or
        a restored project that named ``masked_median`` left ``detrend_enabled`` True and
        ``detrend_method`` on ``block_median`` -- the wrong estimator, with no warning,
        no visible difference on the form and provenance that read correct. A protocol
        that cannot be honoured has to say so.
        """
        for i, (_, value) in enumerate(self._DETREND_METHODS):
            if value == method:
                self.detrend_method.setCurrentIndex(i)
                self._detrend_refusal("")
                return None
        offered = ", ".join(v for _, v in self._DETREND_METHODS)
        known = method in baseline_mod.METHODS
        why = (
            f"“{method}” is a real estimator but this panel does not offer it"
            if known else f"“{method}” is not a baseline estimator this app knows"
        )
        complaint = (
            f"Baseline-drift method not applied: {why}. The chain on screen still uses "
            f"“{self.detrend_method_choice()}”, which is a DIFFERENT estimator — do not "
            f"treat this run as the one that was stored. Offered: {offered}."
        )
        self._detrend_refusal(complaint)
        return complaint

    def _detrend_refusal(self, text: str) -> None:
        """Show (or clear) the "that detrend method was not applied" banner.

        It sits with the normalize units banner, ABOVE the filter stack, because the
        refusal has to be visible from whichever page the user is on -- the whole point
        is that they may never open the Baseline drift page and so would never see a
        note that lived there. While hosted, the Preprocess card's warning line mirrors
        it, so the card is refreshed on every write here: a refusal set or cleared with
        no widget value changing emits no ``paramsChanged``.
        """
        self.detrend_refusal_label.setText(text)
        self.detrend_refusal_label.setVisible(bool(text))
        self._refresh_chain_card()

    def _on_detrend_method_changed(self, *_a) -> None:
        """Raise the selected estimator's sub-page and gate the shared knobs above it.

        The page is derived from the METHOD, never from the combo index, so inserting a
        method into ``baseline.METHODS`` cannot silently raise the wrong parameters.
        """
        method = self.detrend_method_choice()
        pages = [m for _l, m in self._DETREND_METHODS]
        self.detrend_params_stack.setCurrentIndex(pages.index(method))

        # The window spin: real for five methods, a dead knob for the masked one.
        ignored = method in self._DETREND_WINDOW_IGNORED
        self.detrend_window_s.setEnabled(not ignored)
        self.detrend_window_note.setVisible(ignored)
        if ignored:
            self.detrend_window_note.setText(
                "The event-masked median never reads this window — it uses the rough "
                "and tracking windows below. (Measured: two runs at 0.05 s and 5 s are "
                "byte-identical.) It is disabled rather than hidden so a stored value "
                "is still visible.")

        # The shared decimation row: only the two median estimators read it.
        self._detrend_form.setRowVisible(
            self.detrend_decim, method in ("masked_median", "robust_median"))

    def normalize_mode_choice(self) -> str:
        """The rescaling mode string the normalize stage runs under: one of
        :data:`~nanoscope.analysis.filters.NORMALIZE_MODES_ALL`. The first three go to
        :func:`~nanoscope.analysis.filters.normalize`; ``'model_input'`` goes to
        :func:`~nanoscope.analysis.filters.robust_normalize`, which that function
        REFUSES -- `modules._apply_normalize` is the one place they are told apart."""
        return str(self.normalize_mode.currentData())

    def set_normalize_mode(self, mode: str) -> None:
        """Select the normalisation mode by its kernel string. Unknown strings are
        ignored, leaving the current choice — never silently swapped for another mode,
        which would rescale by a different divisor than the one that was stored."""
        idx = self.normalize_mode.findData(mode)
        if idx >= 0:
            self.normalize_mode.setCurrentIndex(idx)

    # -- high-pass / notch / savgol setters (mirror set_detrend_*) ---------
    def set_hp_enabled(self, enabled: bool) -> None:
        """Enable/disable the high-pass stage (DC / slow-drift removal)."""
        self.hp_enabled.setChecked(bool(enabled))

    def set_hp_cutoff_hz(self, cutoff_hz: float) -> None:
        """Set the high-pass cutoff in Hz."""
        self.hp_cutoff_hz.setValue(float(cutoff_hz))

    def set_notch_enabled(self, enabled: bool) -> None:
        """Enable/disable the mains-hum notch stage."""
        self.notch_enabled.setChecked(bool(enabled))

    def set_notch_freq_hz(self, freq_hz: float) -> None:
        """Set the notch fundamental line frequency in Hz (60 in the Americas, 50 else)."""
        self.notch_freq_hz.setValue(float(freq_hz))

    def set_notch_harmonics(self, harmonics: int) -> None:
        """Set how many harmonics of the mains line the notch removes."""
        self.notch_harmonics.setValue(int(harmonics))

    def set_notch_q(self, q: float) -> None:
        """Set the notch quality factor (higher Q -> a narrower notch)."""
        self.notch_q.setValue(float(q))

    #: What separates two entries in the "lines" box: ',', ';', '/' and ANY whitespace,
    #: including line breaks and tabs. A spreadsheet COLUMN pastes newline-separated and
    #: a ROW tab-separated; with only the punctuation accepted, each was ONE token, and
    #: the refusal read "1 entry is not a positive frequency in Hz — ‘60\n120\n180…’"
    #: for a paste whose every entry was one (the 10,923-value column that motivated the
    #: cap was refused as a single bad entry, so the cap sentence never appeared for the
    #: shape it exists for). No legal token holds whitespace (``float`` refuses it), so
    #: every list the box accepted before tokenizes identically.
    _NOTCH_SEPARATORS = re.compile(r"[,;/\s]+")

    @classmethod
    def _notch_line_tokens(cls, text: str) -> list[str]:
        """The box's non-empty tokens, in typed order, split on :attr:`_NOTCH_SEPARATORS`.
        A token can therefore never carry a line break into a refusal sentence."""
        return [tok for tok in cls._NOTCH_SEPARATORS.split(str(text)) if tok]

    @classmethod
    def parse_notch_lines(cls, text: str) -> list[float]:
        """Parse the explicit notch-line box into positive floats, IN TYPED ORDER.

        Deliberately NOT :meth:`_parse_notch_freqs` (the ML tab's parser): that one
        falls back to the trained 60/120/180 when the box is empty, because the ML
        chain must never hand a model an un-notched signal. Here an empty box means
        EMPTY -- the panel says so out loud and reverts to the series spelling rather
        than inventing three lines the user did not ask for. Duplicates are kept so the
        kernel can refuse them by name instead of the panel quietly de-duplicating a
        typo.

        A token that is not a positive, finite number is REFUSED — ``ValueError``
        naming every such token — never dropped. It used to be dropped: '60, -120,
        abc, 180' parsed to [60, 180], the note said "2 line(s)", provenance stamped
        [60, 180], and the box visibly held four entries, so a '-120' mistyped for
        '120' ran a different filter than the one on screen. That is the same class of
        failure the kernel refuses duplicates BY NAME to prevent, and a nine-line list
        earns the same policy as a two-line one. (``1e400`` is ``inf`` and used to pass
        the ``> 0`` test; it is refused too.) Ask :meth:`notch_lines_refusal` the
        question "may this text be committed?" — it folds this and the line cap into
        one sentence for the note.
        """
        out: list[float] = []
        bad: list[str] = []
        for tok in cls._notch_line_tokens(text):
            # Shared strict ASCII parser (beta-sweep B27, 2026-09-12): bare float() also
            # accepted non-ASCII digit scripts (e.g. Arabic-Indic), not only the
            # underscore this loop already guarded by name.
            val = parse_strict_number(tok)
            if val is None or not val > 0:
                bad.append(tok)
                continue
            out.append(val)
        if bad:
            shown = ", ".join(f"‘{t[:24]}’" for t in bad[:6])
            more = f" and {len(bad) - 6:,} more" if len(bad) > 6 else ""
            raise ValueError(
                f"{len(bad)} entr{'y is' if len(bad) == 1 else 'ies are'} not a positive "
                f"frequency in Hz — {shown}{more}")
        return out

    @classmethod
    def _notch_refusal_parts(cls, text: str, *,
                             hit_limit: bool = False) -> tuple[str, str] | None:
        """``(verdict, rationale)`` for a text that may NOT be committed — ``None`` if it may.

        Two refusals, the line cap first (a flood makes any token complaint noise):
        more than :attr:`NOTCH_MAX_LINES` tokens, or any token that is not a positive
        finite number. ``hit_limit`` is the caller's word that the box holds exactly
        its :attr:`NOTCH_BOX_MAX_CHARS`, i.e. Qt truncated a longer paste: the count is
        then a floor ("at least"), never presented as the paste's length. Both name the
        cap and the count / the tokens, because a refusal with no magnitude is the
        on-screen silence this guard exists to end.

        Split in two because the note under the box owns a fixed height and shows a
        long sentence from its head: the verdict (what was refused, and how much) is
        what must stay painted, and the callers put the fact the user needs next --
        what is in force instead -- BEFORE the rationale, which may be the part that
        ends up in the tooltip. :meth:`notch_lines_refusal` is the two joined.
        """
        tokens = cls._notch_line_tokens(text)
        if hit_limit or len(tokens) > cls.NOTCH_MAX_LINES:
            held = (f"at least {len(tokens):,} lines — the paste was longer than the box "
                    f"holds ({cls.NOTCH_BOX_MAX_CHARS:,} characters), so its true length "
                    f"is unknown —" if hit_limit else f"{len(tokens):,} lines")
            return (f"Refused: the list held {held} and the box accepts at most "
                    f"{cls.NOTCH_MAX_LINES}.",
                    "Each line is one more second-order section in the notch cascade over "
                    "the whole recording on every detection (~10 ms per line on a "
                    "3 M-sample file: 4,155 lines cost 42 s a run).")
        try:
            cls.parse_notch_lines(text)
        except ValueError as e:
            return f"Refused: {e}.", "Nothing was dropped and nothing ran."
        return None

    @classmethod
    def notch_lines_refusal(cls, text: str, *, hit_limit: bool = False) -> str | None:
        """Why ``text`` may NOT be committed as the explicit line list — ``None`` if it may.
        The one sentence :meth:`_notch_refusal_parts` describes; the launcher-side replay
        guard (``AnalysisPanel._notch_replay_refusal``) asks this same question."""
        parts = cls._notch_refusal_parts(text, hit_limit=hit_limit)
        return None if parts is None else " ".join(parts)

    def _accept_notch_text(self, text: str) -> None:
        """THE writer of the explicit list in force: a legal ``text`` becomes the rollback
        target, the box is spelled as that list, and a standing refusal is cleared
        exactly when the list in force is no longer the one it was raised under.

        A commit and a replay (:meth:`set_notch_lines`) both land here because they
        used to clear the refusal by two different rules, and the second erased what
        the first had just painted: a refused commit rolled the box back to the list
        in force with the refusal under it, the run that followed read that list,
        and when its result landed the form was re-seeded from it -- ``set_notch_lines``
        with the very same list -- which cleared the refusal unconditionally. Measured
        2026-09-02 on the owner's -1000 mV excerpt: type '60, abc', press Ctrl+D, and the
        end state was box '60, 120', a note with no refusal, an empty status bar and a
        detection stamped [60, 120] -- the user's list discarded with no surviving
        disclosure. The rule here is the one the commit already followed ("a refusal
        outlives the rollback it describes; a NEW legal list clears it"), judged on
        the PARSED list rather than the text so a respelling ('60,120', '60.0, 120.0',
        a column paste) is not mistaken for a new list, and so a stored-list refusal --
        raised with the series spelling in force, claiming it is live instead -- is
        cleared the moment any explicit list goes live, because its claim is then false.

        The box is re-spelled as ``", ".join(tokens)`` -- the user's own spellings, so
        no value moves -- only when that differs from what it holds: a pasted column
        then reads as a list, and the rollback target never carries a line break.
        """
        tokens = self._notch_line_tokens(text)
        lines = self.parse_notch_lines(text)    # legal by contract: callers judged it
        if self._notch_refusal is not None and lines != self._notch_refusal_under:
            self._notch_refusal = None
            self._notch_refusal_under = None
        self._notch_ok_text, self._notch_ok_lines = ", ".join(tokens), lines
        if self.notch_freqs.text() != self._notch_ok_text:
            self.notch_freqs.setText(self._notch_ok_text)

    def _refuse_notch_lines(self, why: str, *, under: list[float] | None) -> None:
        """THE writer of the standing refusal: ``why`` goes under the box (the next
        :meth:`_refresh_notch_mode` paints it) and out on ``notchLinesRefused`` for the
        status line, and ``under`` records the explicit list in force as it is raised
        (``None``: the series spelling) -- what :meth:`_accept_notch_text` judges a
        later legal list against."""
        self._notch_refusal = why
        self._notch_refusal_under = under
        self.notchLinesRefused.emit(why)

    def _commit_notch_lines(self) -> bool:
        """Commit the line box: keep a legal text, refuse an illegal one by name and roll
        the box back. True when a rollback happened (the note then needs re-drawing).

        The ML tab's box has this guard (``AnalysisPanel._on_ml_notch_committed``); this
        is the same policy on its sibling, hosted HERE because this panel owns the
        widget and its note, and because the launcher of Schmitt runs reads
        :meth:`params` — so the invariant "the box never holds a list that may not
        run" is what gates the run, with no launcher-side check to forget. Rolling back
        rather than keeping the refused text on screen for editing is deliberate: a box
        displaying '60, -120, abc, 180' beside a note saying [60, 180] is in force is
        the two-surfaces-disagree state this exists to end, and a paste of thousands
        of lines is not something anyone edits in place. The refusal quotes the
        restored list so the user knows exactly what is in force -- right after the
        verdict, ahead of the rationale, because the fixed-height note shows a long
        refusal from its head.
        """
        text = self.notch_freqs.text()
        parts = self._notch_refusal_parts(
            text, hit_limit=len(text) >= self.notch_freqs.maxLength())
        if parts is None:
            # A standing refusal outlives the rollback it describes: re-committing the
            # rolled-back list (which every note refresh and every params() does) keeps
            # it; only a legal DIFFERENT list clears it (`_accept_notch_text`).
            self._accept_notch_text(text)
            return False
        verdict, rationale = parts
        restored = self._notch_ok_text.strip()
        self._refuse_notch_lines(
            f"{verdict} The box was restored to its previous contents "
            + (f"({restored})." if restored else "(empty).") + f" {rationale}",
            under=list(self._notch_ok_lines))
        self.notch_freqs.setText(self._notch_ok_text)
        return True

    def notch_lines_choice(self) -> list[float] | None:
        """The EXPLICIT line list in force, or ``None`` for the series spelling.

        ``None`` is what :func:`~nanoscope.analysis.modules.notch_freqs_param` reads as
        "use ``notch_freq_hz`` / ``notch_harmonics``". An empty box under a ticked
        checkbox also answers ``None`` -- the kernel REFUSES an empty list, and a
        refusal at run time would be a detection that simply failed. It is not silent:
        :meth:`_refresh_notch_mode` puts the reason under the box in orange.

        The box is committed first (:meth:`_commit_notch_lines`, a no-op on a legal
        text): a run can be requested while the box still holds an uncommitted edit —
        a Detect click that did not move focus, a programmatic ``setText`` — and what a
        run reads must be what the note describes, never a list with a refused token
        dropped out of it or a truncated paste.
        """
        if not self.notch_use_lines.isChecked():
            return None
        if self._commit_notch_lines():
            self._refresh_notch_mode()
        return self.parse_notch_lines(self.notch_freqs.text()) or None

    def set_notch_lines(self, freqs: Sequence[float] | None) -> None:
        """Set the explicit line list (``None``/empty -> back to the series spelling).

        The inverse of :meth:`notch_lines_choice`. It sets BOTH the checkbox and the
        text, because those two are one setting: a stored list with the box unticked
        would be a chain that reads right and runs differently.

        A stored list the box may not hold — more than :attr:`NOTCH_MAX_LINES` lines,
        or a non-positive entry — is refused by name, exactly as a paste is: the box
        keeps its last legal contents, the SERIES spelling is what runs, and the note
        says both. Writing 10,923 stored lines into the box and letting the next
        commit roll them back would leave a panel that reads right and runs for 42 s.

        A replay of the list ALREADY in force -- what re-seeding the form from the run
        it just gated does -- is not a new list and keeps a standing refusal on screen;
        :meth:`_accept_notch_text` is the one place that rule is written.
        """
        lines = [float(f) for f in (freqs or [])]
        # Results re-seed this form before the next run. Six significant digits
        # can move a requested line across Nyquist, so replay must round-trip.
        text = ", ".join(repr(f).removesuffix(".0") for f in lines)
        parts = self._notch_refusal_parts(text) if lines else None
        if parts is not None:
            verdict, rationale = parts
            self.notch_use_lines.setChecked(False)
            self._refuse_notch_lines(
                f"A stored line list was refused — {verdict[len('Refused: '):]} The "
                "frequency and harmonics above (the series spelling) are live instead. "
                f"{rationale}",
                under=None)
            self._refresh_notch_mode()
            return
        self.notch_use_lines.setChecked(bool(lines))
        if lines:
            self._accept_notch_text(text)
        self._refresh_notch_mode()

    def _refresh_notch_mode(self, *_a) -> None:
        """Gate the series spins against the line box and explain which one is live.

        The kernel does not merge the two spellings, so neither does the form: while an
        explicit list is in force the frequency and harmonics spins are disabled (they
        are not read), and while it is not, the line box is.

        This is the box's commit point (it runs on the box's ``editingFinished``, ahead
        of ``_emit_params`` and the response plot): with the explicit spelling on, the
        text is committed through :meth:`_commit_notch_lines` before it is described,
        so the note never lists a truncated or typo-dropped version of what the box
        shows. A standing refusal is appended in orange under whichever spelling is
        live, until a legal commit or replay of a DIFFERENT list clears it
        (:meth:`_accept_notch_text`). The note owns its height, so this repaint can
        never move the control whose click committed the box.
        """
        explicit = self.notch_use_lines.isChecked()
        self.notch_freq_hz.setEnabled(not explicit)
        self.notch_harmonics.setEnabled(not explicit)
        self.notch_freqs.setEnabled(explicit)

        def show(text: str, warn: bool) -> None:
            if self._notch_refusal:
                text = f"{text}\n⚠ {self._notch_refusal}"
                warn = True
            self.notch_lines_note.setStyleSheet(
                "color:#FFB74D; font-weight:bold;" if warn else "color:#8b9199;")
            self.notch_lines_note.setText(text)

        if not explicit:
            show("Series spelling: the fundamental above and its first "
                 f"{int(self.notch_harmonics.value())} multiples.", False)
            return
        self._commit_notch_lines()
        lines = self.parse_notch_lines(self.notch_freqs.text())
        if not lines:
            show("⚠ The line list is EMPTY, so the notch falls back to the frequency "
                 "and harmonics above. Type at least one frequency, or untick the box.",
                 True)
            return
        spelling = ", ".join(repr(f).removesuffix(".0") for f in lines)
        if self._fs is not None and np.isfinite(self._fs) and self._fs > 0:
            nyquist = self._fs / 2.0
            dropped = [f for f in lines if f >= nyquist]
            if dropped:
                applied = [f for f in lines if f < nyquist]
                applied_text = ", ".join(repr(f).removesuffix(".0") for f in applied)
                dropped_text = ", ".join(repr(f).removesuffix(".0") for f in dropped)
                outcome = (f"Will notch {len(applied)} line(s): {applied_text} Hz."
                           if applied else "No usable notch lines; detection will be refused.")
                show(f"⚠ {outcome} Skipping {dropped_text} Hz at or above Nyquist "
                     f"({nyquist!r} Hz). Requested: {spelling} Hz.", True)
                return
        show(f"Explicit spelling: {len(lines)} line(s) — {spelling} Hz. "
             "The frequency and harmonics above are NOT read while this is on.", False)

    def set_savgol_enabled(self, enabled: bool) -> None:
        """Enable/disable the Savitzky-Golay smoothing stage."""
        self.savgol_enabled.setChecked(bool(enabled))

    def set_savgol_window_s(self, window_s: float) -> None:
        """Set the Savitzky-Golay window length in seconds."""
        self.savgol_window_s.setValue(float(window_s))

    def set_savgol_polyorder(self, polyorder: int) -> None:
        """Set the Savitzky-Golay polynomial order."""
        self.savgol_polyorder.setValue(int(polyorder))

    def set_median_enabled(self, enabled: bool) -> None:
        """Enable/disable the median despiking stage."""
        self.median_enabled.setChecked(bool(enabled))

    def set_median_window_s(self, window_s: float) -> None:
        """Set the median-filter window length in seconds."""
        self.median_window_s.setValue(float(window_s))

    def set_rmean_enabled(self, enabled: bool) -> None:
        """Enable/disable the running-mean (moving average) stage."""
        self.rmean_enabled.setChecked(bool(enabled))

    def set_rmean_window_s(self, window_s: float) -> None:
        """Set the running-mean window length in seconds."""
        self.rmean_window_s.setValue(float(window_s))

    def set_normalize_enabled(self, enabled: bool) -> None:
        """Enable/disable the whole-trace normalisation stage.

        Enabling this makes the detector's input DIMENSIONLESS; the panel's unit banner
        says so for as long as it is on (see :meth:`_refresh_normalize_units_note`)."""
        self.normalize_enabled.setChecked(bool(enabled))

    def params(self) -> dict[str, Any]:
        """Current detection + preprocessing parameters as a plain dict.

        Most keys are EXACTLY the ``detect_events`` kwargs the module forwards:
        ``merge_gap_s`` and ``max_dwell_s`` are None when 0, and ``filter_cutoff_hz``
        is None when the filter is disabled — the 'off' sentinels the detector expects.

        UNITS: the ``merge_gap_s`` key is in SECONDS even though the widget behind it
        (``merge_gap_ms``) reads milliseconds. detect_events' kwarg is seconds, and every
        saved project / protocol / session written before the widget changed unit stored
        this key in seconds; keeping the key and its unit fixed is what makes those files
        round-trip unchanged. The ms -> s conversion happens here and NOWHERE else (its
        mirror, s -> ms, is the single line in :meth:`set_params`).
        ``filter_order`` / ``filter_kind`` are detect_events kwargs too (the detector owns
        the low-pass) and are always present: the detector ignores them when
        ``filter_cutoff_hz`` is None.

        BOTH DETECTORS' THRESHOLDS ARE ALWAYS PRESENT, and ``"detector"`` says which of
        them is live. That is deliberate. Emitting only the selected detector's keys
        would mean switching detectors silently discarded whatever was dialled into the
        other, and a protocol or a restored project would round-trip only half the form.
        Instead the form carries both and `modules._detect_kwargs` strips the foreign
        pair (`modules._DETECTOR_PRIVATE_KEYS`) on the way into the run, so a CUSUM run
        never receives ``k_in`` and a Schmitt run never receives ``cusum_h`` -- the
        leak is prevented structurally, in one place, rather than by hoping every
        producer of a params dict remembers.

        ``"detector"`` is a MODULE ID (``"detect.schmitt"`` / ``"detect.cusum"``), not a
        nickname, so the panel and ``ANALYSIS_REGISTRY`` cannot disagree about what it
        names. It is not a detector kwarg either; it is consumed like ``pore_id``.

        The ``detrend_*`` keys and the notch / median / high-pass / running-mean / savgol
        keys are NOT detect_events kwargs; they steer the pre-detection conditioning that
        :func:`preprocess_for_detection` applies. The module consumes them (they are in
        ``DETREND_PARAM_KEYS`` / ``FILTER_PARAM_KEYS``) and strips them before calling the
        detector, so they never reach it. The enable flags and their values are always
        present; preprocessing simply ignores a stage whose flag is False.
        """
        cutoff = float(self.filter_cutoff_hz.value()) if self.filter_enabled.isChecked() else None
        # The ONE ms -> s conversion. The widget reads ms; the key stays seconds.
        merge_s = float(self.merge_gap_ms.value()) * 1e-3
        max_dwell = float(self.max_dwell_s.value())
        return {
            # WHICH detector to run. Not a detector kwarg -- it is in
            # `modules._CONSUMED_PARAM_KEYS`, and the owning panel reads it to pick a
            # module id. See the docstring for why BOTH detectors' thresholds ship
            # alongside it.
            "detector": self.detector_choice(),
            "polarity": self.polarity_choice(),
            "k_in": float(self.k_in.value()),
            "k_out": float(self.k_out.value()),
            "cusum_k": float(self.cusum_k.value()),
            "cusum_h": float(self.cusum_h.value()),
            "min_dwell_s": float(self.min_dwell_s.value()),
            "max_dwell_s": max_dwell if max_dwell > 0 else None,
            "filter_cutoff_hz": cutoff,
            # The low-pass DESIGN travels with its cutoff: unlike every other filter key
            # these two are real detect_events kwargs (the low-pass is applied inside the
            # detector, so it can measure the rise time of the filter it actually used).
            "filter_order": int(self.filter_order.value()),
            "filter_kind": self.filter_kind_choice(),
            "merge_gap_s": merge_s if merge_s > 0 else None,
            "detrend_enabled": bool(self.detrend_enabled.isChecked()),
            "detrend_method": self.detrend_method_choice(),
            "detrend_window_s": float(self.detrend_window_s.value()),
            "detrend_percentile": float(self.detrend_percentile.value()),
            "detrend_asls_lam": float(self.detrend_asls_lam.value()),
            "detrend_asls_p": float(self.detrend_asls_p.value()),
            "detrend_asls_niter": int(self.detrend_asls_niter.value()),
            "detrend_asls_decim": float(self.detrend_asls_decim.value()),
            "detrend_rough_s": float(self.detrend_rough_s.value()),
            "detrend_base_s": float(self.detrend_base_s.value()),
            "detrend_mask_sigma": float(self.detrend_mask_sigma.value()),
            "detrend_dilate_ms": float(self.detrend_dilate_ms.value()),
            # 0 == `modules._DECIM_AUTO`, which reaches the kernel as ``decim=None``.
            "detrend_decim": int(self.detrend_decim.value()),
            "detrend_sg_poly": int(self.detrend_sg_poly.value()),
            "detrend_sg_decim": int(self.detrend_sg_decim.value()),
            "notch_enabled": bool(self.notch_enabled.isChecked()),
            "notch_freq_hz": float(self.notch_freq_hz.value()),
            "notch_harmonics": int(self.notch_harmonics.value()),
            "notch_q": float(self.notch_q.value()),
            # THE SPELLING SWITCH, and it is always present so the contract is never
            # ambiguous: a LIST means the explicit lines are in force and freq/harmonics
            # are not read at all; None means the series above is. Never ``[]`` -- the
            # kernel refuses that by design, and `notch_lines_choice` folds an empty box
            # to None with a visible note rather than sending a run to its death.
            "notch_freqs": self.notch_lines_choice(),
            "median_enabled": bool(self.median_enabled.isChecked()),
            "median_window_s": float(self.median_window_s.value()),
            "hp_enabled": bool(self.hp_enabled.isChecked()),
            "hp_cutoff_hz": float(self.hp_cutoff_hz.value()),
            "hp_order": int(self.hp_order.value()),
            "hp_kind": self.hp_kind_choice(),
            "rmean_enabled": bool(self.rmean_enabled.isChecked()),
            "rmean_window_s": float(self.rmean_window_s.value()),
            "savgol_enabled": bool(self.savgol_enabled.isChecked()),
            "savgol_window_s": float(self.savgol_window_s.value()),
            "savgol_polyorder": int(self.savgol_polyorder.value()),
            # The unit-changing stage. Like the other pre-detection keys these are
            # consumed by preprocess_for_detection and stripped before detect_events;
            # unlike the others, a run with `normalize_enabled` True produces depths
            # that are NOT in pA, which is why the panel keeps a banner on screen for
            # as long as it is set (see _refresh_normalize_units_note).
            "normalize_enabled": bool(self.normalize_enabled.isChecked()),
            "normalize_mode": self.normalize_mode_choice(),
            # Read only by the 'model_input' mode; always present so a protocol that
            # later switches to that mode recalls the bound it was saved with.
            "normalize_clip_sigma": float(self.normalize_clip_sigma.value()),
        }

    def surrogate_params(self) -> dict[str, Any]:
        """:meth:`params` for the three SCHMITT-ONLY studies (sweep / null / recovery).

        Identical to ``params()`` minus ``"detector"`` and every OTHER detector's private
        thresholds, taken from `modules.DETECTOR_PRIVATE_KEYS` rather than re-spelled --
        those three receivers forward what they are handed to `detect_events` after
        stripping only the preprocessing keys, so a stray ``cusum_h`` arrives as an
        unexpected keyword argument and the study dies. It is also the honest dict: these
        studies sweep ``k_in``, and none of them has a CUSUM analogue.
        """
        foreign = frozenset().union(
            *(v for k, v in DETECTOR_PRIVATE_KEYS.items() if k != "detect.schmitt"))
        return {k: v for k, v in self.params().items()
                if k != "detector" and k not in foreign}

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Seed the whole Schmitt + preprocessing form from a params dict — :meth:`params`'
        inverse.

        This is what makes the form agree with the event table beside it. A stored detection
        carries the settings it ran under; without a way to put them back, reopening a project
        showed the DEFAULT form over a table produced by completely different numbers
        (``min_dwell_s`` reading 2e-05 above events detected at 0.0022), and anything that
        re-ran "the current detection" from the form re-detected something else entirely.

        A key that is ABSENT is left alone rather than defaulted, so a partial dict (an old
        provenance that never recorded a stage) is a partial update. ``None`` is the
        detector's off-sentinel for ``max_dwell_s`` / ``merge_gap_s`` / ``filter_cutoff_hz``
        and is translated back into this form's 0-means-off / unchecked spelling.

        Ordering is load-bearing in one place: ``k_in`` is applied before ``k_out``, because
        the panel clamps ``k_out``'s maximum to just under ``k_in`` and applying them the
        other way round would silently truncate a stored ``k_out``.
        """
        def _apply(key, setter, cast=None):
            if key in params:
                value = params[key]
                setter(value if cast is None else cast(value))

        # -- WHICH detector, BEFORE its thresholds -------------------------
        # `_on_detector_changed` re-seeds min dwell / merge gap when they still hold
        # the other detector's default, so the selection has to land first or that
        # re-seed would overwrite the stored values applied just below.
        _apply("detector", self.set_detector, str)

        # -- thresholds (k_in first; see the docstring) --------------------
        _apply("k_in", self.k_in.setValue, float)
        _apply("k_out", self.k_out.setValue, float)
        _apply("cusum_k", self.cusum_k.setValue, float)
        _apply("cusum_h", self.cusum_h.setValue, float)
        _apply("min_dwell_s", self.min_dwell_s.setValue, float)
        # 0 IS the off state for these two spins, which is exactly what None means upstream.
        _apply("max_dwell_s", self.max_dwell_s.setValue, lambda v: float(v or 0.0))
        # ``merge_gap_s`` is stored in SECONDS (see params()) and displayed in MILLISECONDS,
        # so it needs the s -> ms leg that ``max_dwell_s`` does not. This scaling is what
        # keeps files written before the widget changed unit reading correctly: a stored
        # 0.015 s is 15 ms on the form, never 0.015 ms.
        _apply("merge_gap_s", self.merge_gap_ms.setValue, lambda v: float(v or 0.0) * 1e3)

        # -- polarity ------------------------------------------------------
        if "polarity" in params:
            self.set_polarity(params["polarity"])

        # -- low-pass (cutoff None == the stage was off) --------------------
        if "filter_cutoff_hz" in params:
            cutoff = params["filter_cutoff_hz"]
            self.filter_enabled.setChecked(cutoff is not None)
            if cutoff is not None:
                self.filter_cutoff_hz.setValue(float(cutoff))
        _apply("filter_order", self.set_filter_order, int)
        _apply("filter_kind", self.set_filter_kind, str)

        # -- baseline-drift removal ----------------------------------------
        _apply("detrend_enabled", self.detrend_enabled.setChecked, bool)
        _apply("detrend_method", self.set_detrend_method, str)
        _apply("detrend_window_s", self.detrend_window_s.setValue, float)
        _apply("detrend_percentile", self.detrend_percentile.setValue, float)
        _apply("detrend_asls_lam", self.detrend_asls_lam.setValue, float)
        _apply("detrend_asls_p", self.detrend_asls_p.setValue, float)
        _apply("detrend_asls_niter", self.detrend_asls_niter.setValue, int)
        _apply("detrend_asls_decim", self.detrend_asls_decim.setValue, float)
        _apply("detrend_rough_s", self.detrend_rough_s.setValue, float)
        _apply("detrend_base_s", self.detrend_base_s.setValue, float)
        _apply("detrend_mask_sigma", self.detrend_mask_sigma.setValue, float)
        _apply("detrend_dilate_ms", self.detrend_dilate_ms.setValue, float)
        _apply("detrend_decim", self.detrend_decim.setValue,
               lambda v: int(v or 0))   # None/0 == the kernel's auto stride
        _apply("detrend_sg_poly", self.detrend_sg_poly.setValue, int)
        _apply("detrend_sg_decim", self.detrend_sg_decim.setValue, int)

        # -- the other pre-detection stages --------------------------------
        _apply("notch_enabled", self.set_notch_enabled, bool)
        _apply("notch_freq_hz", self.set_notch_freq_hz, float)
        _apply("notch_harmonics", self.set_notch_harmonics, int)
        _apply("notch_q", self.set_notch_q, float)
        # LAST of the notch keys, because it decides which of the other two are live.
        # A None here is a real value -- "the series spelling" -- not an absent key,
        # so it goes through the setter rather than being skipped.
        _apply("notch_freqs", self.set_notch_lines)
        _apply("median_enabled", self.set_median_enabled, bool)
        _apply("median_window_s", self.set_median_window_s, float)
        _apply("hp_enabled", self.set_hp_enabled, bool)
        _apply("hp_cutoff_hz", self.set_hp_cutoff_hz, float)
        _apply("hp_order", self.set_hp_order, int)
        _apply("hp_kind", self.set_hp_kind, str)
        _apply("rmean_enabled", self.set_rmean_enabled, bool)
        _apply("rmean_window_s", self.set_rmean_window_s, float)
        _apply("savgol_enabled", self.set_savgol_enabled, bool)
        _apply("savgol_window_s", self.set_savgol_window_s, float)
        _apply("savgol_polyorder", self.set_savgol_polyorder, int)
        # Mode BEFORE the enable flag, so the unit banner the flag raises already names
        # the mode that is about to run rather than the one left over from before.
        # Clip BEFORE the mode, and the mode before the enable flag, so the banner the
        # flag raises already quotes the bound this chain will actually run at.
        _apply("normalize_clip_sigma", self.normalize_clip_sigma.setValue, float)
        _apply("normalize_mode", self.set_normalize_mode, str)
        _apply("normalize_enabled", self.set_normalize_enabled, bool)

    def set_polarity(self, polarity: int | None) -> None:
        """Select the polarity override: None -> 'auto (measure)', -1 -> blockade, +1 ->
        enhancement. The inverse of :meth:`polarity_choice`; anything else is ignored."""
        if polarity is None:
            self.polarity_override.setCurrentIndex(0)
        elif int(polarity) == -1:
            self.polarity_override.setCurrentIndex(1)
        elif int(polarity) == 1:
            self.polarity_override.setCurrentIndex(2)

    # -- internal ---------------------------------------------------------
    def _enforce_k_constraint(self) -> None:
        """Keep k_out strictly below k_in in the UI itself.

        The Schmitt trigger needs ``k_out < k_in`` or it degenerates into a
        single-threshold detector that chatters and shatters events; clamp the
        widgets rather than let ``detect_events`` raise after the user committed.
        """
        self.k_out.setMaximum(max(self.k_out.minimum(), self.k_in.value() - 0.01))
        if self.k_out.value() >= self.k_in.value():
            self.k_out.setValue(max(self.k_out.minimum(), self.k_in.value() - 0.5))

    def _enforce_ml_threshold_constraint(self) -> None:
        """Keep the ML exit threshold at or below enter (hysteresis).

        ``decode_regions`` requires ``0 <= exit <= enter`` or it raises; clamp the
        widget rather than let the run fail after the user committed. Equal is allowed
        (a degenerate single-threshold decode), unlike the strict k_out < k_in."""
        self.ml_exit.setMaximum(self.ml_enter.value())
        if self.ml_exit.value() > self.ml_enter.value():
            self.ml_exit.setValue(self.ml_enter.value())

    def _on_filter_enabled(self, on: bool) -> None:
        self._refresh_preview_enabled()
        self._update_rise_time()

    def _on_detrend_enabled(self, on: bool) -> None:
        self._refresh_preview_enabled()

    def _on_preprocess_stage_toggled(self, _on: bool) -> None:
        """A notch / high-pass / savgol enable toggled: re-evaluate the preview toggle."""
        self._refresh_preview_enabled()

    def _refresh_preview_enabled(self) -> None:
        """The "show on trace" preview is meaningful whenever ANY stage is on — the
        low-pass, or any pre-detection filter (notch / high-pass / baseline-drift /
        savgol) — because the preview draws the fully-preprocessed signal the detector
        sees. Disabling every stage clears a live preview (nothing left to show)."""
        on = (
            self.filter_enabled.isChecked()
            or self.detrend_enabled.isChecked()
            or self.notch_enabled.isChecked()
            or self.median_enabled.isChecked()
            or self.hp_enabled.isChecked()
            or self.rmean_enabled.isChecked()
            or self.savgol_enabled.isChecked()
            or self.normalize_enabled.isChecked()
        )
        self.filter_preview.setEnabled(on)
        if not on and self.filter_preview.isChecked():
            self.filter_preview.setChecked(False)

    # -- named preprocessing protocols -----------------------------------------------------------
    #: Sentinel entry: the chain is whatever the pages say, no protocol's name over it.
    _PROTO_MANUAL = MANUAL_PROTOCOL
    #: The keys a protocol captures — every PREPROCESSING stage, and nothing else:
    #: thresholds (k_in/k_out/min_dwell…) belong to the detection, not the conditioning,
    #: and a "protocol" that silently rewrote them would change what an event IS.
    #: (``conditioning_recipes.CHAIN_KEYS``, which documents each group.)
    _PROTOCOL_KEYS = CHAIN_KEYS
    _PROTO_SETTINGS_KEY = "detection/filter_protocols"

    #: SHIPPED preprocessing protocols, listed above the user's own and not editable.
    #: COMPLETE recipes (every stage): the owner's three pipelines and their
    #: provenance notes live in ``conditioning_recipes._BUILTIN_PARTIALS``.
    _BUILTIN_PROTOCOLS: Mapping[str, dict[str, Any]] = BUILTIN_RECIPES

    @staticmethod
    def _load_protocols() -> dict[str, dict]:
        import json
        raw = QSettings().value(DetectionPanel._PROTO_SETTINGS_KEY, "")
        if not raw:
            return {}
        try:
            got = json.loads(raw)
            return got if isinstance(got, dict) else {}
        except (ValueError, TypeError):
            return {}   # a corrupt store means no protocols, never a crash

    @staticmethod
    def _store_protocols(protocols: dict[str, dict]) -> None:
        import json
        QSettings().setValue(DetectionPanel._PROTO_SETTINGS_KEY, json.dumps(protocols))

    def _load_protocols_into_combo(self) -> None:
        self._protocol_updating = True
        try:
            self.protocol_combo.clear()
            self.protocol_combo.addItem(self._PROTO_MANUAL)
            # Shipped first, in their declared order (not sorted -- the order is the
            # order the owner's three projects are usually reached for), then the
            # user's own, sorted. A user name that collides with a shipped one is
            # refused at save time, so a name identifies exactly one protocol.
            for name in self._BUILTIN_PROTOCOLS:
                self.protocol_combo.addItem(name)
            for name in sorted(self._load_protocols()):
                if name in self._BUILTIN_PROTOCOLS:
                    continue    # a store written before the shipped set existed
                self.protocol_combo.addItem(name)
            self._set_protocol_buttons_enabled(False)
        finally:
            self._protocol_updating = False

    def _set_protocol_buttons_enabled(self, on: bool) -> None:
        """Rename and Delete both act on the SELECTED protocol, so they are enabled and
        disabled together — there is no state where one is meaningful and the other is
        not, and two independent setEnabled calls is how that would drift.

        A SHIPPED protocol is never editable: renaming or deleting one would leave the
        combo naming settings the app can no longer reproduce, and the whole value of
        the three built-ins is that "Cu2-insulin" means one specific verified chain."""
        editable = bool(on) and self.protocol_combo.currentText() not in (
            self._BUILTIN_PROTOCOLS)
        self.protocol_rename_btn.setEnabled(editable)
        self.protocol_delete_btn.setEnabled(editable)

    def _on_protocol_selected(self, name: str) -> None:
        if self._protocol_updating or name == self._PROTO_MANUAL or not name:
            self._set_protocol_buttons_enabled(
                bool(name) and name != self._PROTO_MANUAL)
            return
        # Shipped protocols win over a same-named stored one. They cannot be created by
        # Save-as (the name is refused), so a collision can only come from a store
        # written before the shipped set existed -- and the shipped numbers are the
        # verified ones.
        proto = self._protocol_recipe(name)
        if proto is None:
            self._select_protocol(self._PROTO_MANUAL)
            self.protocolNotice.emit(
                f"Protocol “{name}” could not be read (its stored settings are not a valid "
                f"chain); the form was not changed.")
            return
        # set_params is the SAME round-trip a project restore uses, so the recalled
        # chain is exactly the saved one. It emits paramsChanged, which would drop the
        # combo back to manual — the guard keeps the just-picked name in place.
        # A protocol is a COMPLETE chain: recalling a partial dict left unrelated stages
        # (and the low-pass) on under the protocol's name (SP-04).
        self._protocol_updating = True
        try:
            self.set_params(complete_recipe(proto))
        finally:
            self._protocol_updating = False
        form = self.params()
        if not recipes_equal(form, proto):
            # The form clamped a value (a cutoff above this recording's Nyquist, a stored
            # value outside a box's range): the chain on screen is not the protocol.
            changed = (describe_differences(chain_differences(form, proto, fs=self._fs))
                       or "; ".join(recipe_key_differences(proto, form)))
            self._select_protocol(self._PROTO_MANUAL)
            self.protocolNotice.emit(
                f"Protocol “{name}” could not be applied exactly on this recording, so the "
                f"form shows “{self._PROTO_MANUAL}”: {changed}.")
            self._refresh_chain_card()
            return
        self._set_protocol_buttons_enabled(True)
        self._refresh_chain_card()

    def _protocol_dialog_parent(self) -> QWidget:
        """The window the protocol verbs are clicked in (the Signal Conditioning window
        while hosted), so their prompts open over it."""
        return self.protocol_combo.window() or self

    def _protocol_name_ok(self, name: str, *, title: str) -> bool:
        """Reject the reserved sentinel and the shipped names. Reports, returns False.

        A saved protocol may not shadow a shipped one. Allowing it would put two
        different chains behind one name in one combo -- and the shipped three carry
        numbers verified against the owner's own written metadata, which is exactly the
        thing that must not become ambiguous.
        """
        if name == self._PROTO_MANUAL:
            QMessageBox.information(self._protocol_dialog_parent(), title,
                                    "That name is reserved for the manual state.")
            return False
        if name in self._BUILTIN_PROTOCOLS:
            QMessageBox.information(
                self._protocol_dialog_parent(), title,
                f"“{name}” is a built-in protocol and cannot be replaced. Save your "
                f"chain under a different name — the built-in one stays available, so "
                f"you can always get back to the verified settings.")
            return False
        return True

    def _select_protocol(self, name: str) -> None:
        """Put ``name`` in the combo without the selection re-applying its settings or
        the paramsChanged echo knocking the combo back to '— manual —'."""
        self._protocol_updating = True
        try:
            self.protocol_combo.setCurrentText(name)
        finally:
            self._protocol_updating = False
        self._set_protocol_buttons_enabled(name != self._PROTO_MANUAL)

    def _save_protocol(self) -> None:
        name, ok = QInputDialog.getText(
            self._protocol_dialog_parent(), "Save preprocessing protocol",
            "Name for this chain (all stages, enabled or not):")
        name = (name or "").strip()
        if not ok or not name:
            return
        if not self._protocol_name_ok(name, title="Save protocol"):
            return
        protocols = self._load_protocols()
        # SAVING ONTO AN EXISTING NAME IS A DESTRUCTIVE EDIT, and it used to happen in
        # silence: typing a name that already existed replaced that protocol's whole
        # chain with whatever the pages happened to say, with no undo and nothing on
        # screen to notice. The names are typed from memory, so a collision is a typo
        # at least as often as it is an intent to update.
        if name in protocols:
            answer = QMessageBox.question(
                self._protocol_dialog_parent(), "Overwrite protocol?",
                f"A protocol named “{name}” already exists.\n\n"
                f"Overwriting replaces its saved chain with the settings currently on "
                f"screen. This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        params = self.params()
        protocols[name] = {k: params[k] for k in self._PROTOCOL_KEYS if k in params}
        self._store_protocols(protocols)
        self._load_protocols_into_combo()
        self._select_protocol(name)

    def _rename_protocol(self) -> None:
        """Rename the selected protocol, MOVING its stored settings unchanged.

        Deliberately not "save under the new name and delete the old": that would
        capture the chain on screen, which after any hand edit is not the chain that
        was stored under the old name."""
        old = self.protocol_combo.currentText()
        if old == self._PROTO_MANUAL or not old:
            return
        protocols = self._load_protocols()
        if old not in protocols:
            return
        new, ok = QInputDialog.getText(
            self._protocol_dialog_parent(), "Rename preprocessing protocol",
            f"New name for “{old}”:", text=old)
        new = (new or "").strip()
        if not ok or not new or new == old:
            return
        if not self._protocol_name_ok(new, title="Rename protocol"):
            return
        # A rename onto an occupied name destroys the occupant just as surely as a
        # Save-as would, so it asks the same question.
        if new in protocols:
            answer = QMessageBox.question(
                self._protocol_dialog_parent(), "Overwrite protocol?",
                f"A protocol named “{new}” already exists.\n\n"
                f"Renaming “{old}” onto it replaces its saved chain. This cannot be "
                f"undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        protocols[new] = protocols.pop(old)
        self._store_protocols(protocols)
        self._load_protocols_into_combo()
        self._select_protocol(new)

    def _delete_protocol(self) -> None:
        name = self.protocol_combo.currentText()
        if name == self._PROTO_MANUAL:
            return
        protocols = self._load_protocols()
        protocols.pop(name, None)
        self._store_protocols(protocols)
        self._load_protocols_into_combo()

    def _protocol_drop_to_manual(self, params: Mapping[str, Any] | None = None, *_a) -> None:
        """The combo names a protocol only while the form still IS that protocol.

        Any edit that makes the form differ from the protocol's complete recipe takes the
        name off — the combo says what produced the settings on screen, and after such an
        edit that is the user, not the protocol. A set_params that lands on the protocol's
        exact recipe keeps the name.
        """
        if self._protocol_updating:
            return
        name = self.protocol_combo.currentText()
        if name == self._PROTO_MANUAL:
            return
        recipe = self._protocol_recipe(name)
        form = params if isinstance(params, Mapping) else self.params()
        if recipe is None or not recipes_equal(form, recipe):
            self._protocol_updating = True
            try:
                self.protocol_combo.setCurrentText(self._PROTO_MANUAL)
                self._set_protocol_buttons_enabled(False)
            finally:
                self._protocol_updating = False

    def _refresh_active_filters(self, *_a) -> None:
        """Keep stage markers and the actual execution-order summary in sync.

        Combo markers stay in editor order, while :func:`chain_description` supplies
        the summary in the kernel's execution order. Thus the visible recipe cannot
        silently diverge when a stage is reordered in the processing implementation.
        """
        for i, (label, cb) in enumerate(self._filter_specs):
            on = cb.isChecked()
            self.filter_selector.setItemText(i, f"● {label}" if on else label)
        params = self.params()
        description = chain_description(params)
        if self.hp_enabled.isChecked():
            self.active_filters_label.setText(
                "Candidate-localization order: "
                + (description or "raw calibrated signal")
                + ". Quantitative measurement: same order with high-pass omitted.")
            self.filter_preview.setText(
                "show candidate-localization trace on the plot")
        else:
            self.active_filters_label.setText(
                "Active execution order: "
                + (description or "raw calibrated signal (no stages enabled)"))
            self.filter_preview.setText("show processed trace on the plot")
        self._refresh_highpass_measurement_note()
        self._refresh_normalize_units_note()
        self._refresh_chain_card()

    def _refresh_highpass_measurement_note(self) -> None:
        """Keep the classical dual-signal high-pass contract beside the chain."""
        if not self.hp_enabled.isChecked():
            self.highpass_measurement_label.setText("")
            self.highpass_measurement_label.setVisible(False)
            return
        self.highpass_measurement_label.setText(
            "HIGH-PASS IS FOR CANDIDATE LOCALIZATION ONLY: classical baseline, depth, "
            "fractional blockade, mean/max amplitude and SNR are measured on the same "
            "chain with high-pass omitted. Start/end, dwell, rise/fall and ECD "
            "integration limits still inherit the high-pass-derived boundaries.")
        self.highpass_measurement_label.setVisible(True)

    def _refresh_normalize_units_note(self, *_a) -> None:
        """Show/hide the group-level "this trace is no longer in pA" banner.

        The Normalize stage is the only one in the chain that changes the UNIT of the
        signal the detector measures on, and every current-derived readout downstream
        (depth_pA, the baseline, pA/pA·s export fields, the Inspector amplitude) writes
        its label from the channel's units rather than measuring it. So the disclosure cannot live on the
        Normalize page — it has to be visible from whichever page the user is on, for as
        long as the stage is enabled. That is what this label is."""
        if not self.normalize_enabled.isChecked():
            self.normalize_units_label.setVisible(False)
            self.normalize_units_label.setText("")
            self.normalize_clip_sigma.setEnabled(
                self.normalize_mode_choice() == filters.MODEL_INPUT_MODE)
            return
        mode = self.normalize_mode_choice()
        # 'model_input' is enabled-only-for-its-own-mode, like the detrend sub-pages:
        # the other three kernels have no clip and would ignore it.
        self.normalize_clip_sigma.setEnabled(mode == filters.MODEL_INPUT_MODE)
        if mode == filters.MODEL_INPUT_MODE:
            # THIS MODE'S SENTENCE IS DIFFERENT FROM THE OTHER THREE, and the extra
            # clause is the one that can silently corrupt an analysis: the transform is
            # BOUNDED, so every event deeper than the clip comes out at exactly the
            # bound. Depth is no longer monotone in the original picoamps above it, which
            # no downstream surface can detect — a depth histogram simply grows a spike
            # at 1.0 that looks like a population.
            clip = float(self.normalize_clip_sigma.value())
            self.normalize_units_label.setText(
                f"⚠ NORMALIZE IS ON (model input): the detector measures a "
                f"DIMENSIONLESS trace. Every current-derived depth, baseline and threshold "
                f"below — and pA/pA·s export fields that retain those labels — is in "
                f"event-excluded noise sigmas divided by {clip:g}, not picoamps, and is "
                f"BOUNDED to [−1, +1]. Anything deeper than {clip:g}σ is CLIPPED: two "
                f"events of different real depth come out identical above that, so a "
                f"depth histogram, a fit or a subpopulation split across the bound is "
                f"not measuring what it says. Fractional blockade (dI/I0) is invalid "
                f"because normalization recenters the baseline toward zero; clipping can "
                f"also change detected boundaries and counts.")
            self.normalize_units_label.setVisible(True)
            return
        unit = {
            "zscore": "standard deviations of the whole trace",
            "robust": "baseline-noise sigmas (1.4826·MAD)",
            "minmax": "fractions of the trace's full peak-to-peak span",
        }.get(mode, "dimensionless units")
        self.normalize_units_label.setText(
            f"⚠ NORMALIZE IS ON ({mode}): the detector measures a DIMENSIONLESS trace. "
            f"Current-derived depths, baselines and thresholds below — and pA/pA·s "
            f"export fields that retain those labels — are in {unit}, not picoamps. "
            f"Fractional blockade (dI/I0) is invalid because normalization recenters "
            f"the baseline toward zero.")
        self.normalize_units_label.setVisible(True)

    def _on_filter_kind_changed(self, *_a) -> None:
        """Gaussian is a symmetric smoothing kernel, not a pole cascade — it has no
        order. Disable the spin rather than let it imply a parameter the kernel ignores."""
        has_order = self.filter_kind_choice() not in filters.KINDS_WITHOUT_ORDER
        self.filter_order.setEnabled(has_order)
        self.filter_order.setToolTip(
            "" if has_order else "Gaussian is a symmetric kernel and has no order.")

    def _update_rise_time(self) -> None:
        if self._fs is None:
            self.rise_time_label.setText("rise time: — (sample rate unknown)")
            return
        if not self.filter_enabled.isChecked():
            self.rise_time_label.setText("rise time: — (filter disabled; full bandwidth)")
            return
        cutoff = float(self.filter_cutoff_hz.value())
        kind = self.filter_kind_choice()
        order = int(self.filter_order.value())
        # Guard the COST: the probe is ~40*fs/cutoff samples, so a very low cutoff at a
        # high rate would freeze the GUI. The kernel applies this same limit to an actual
        # run; moving the work off-thread must not turn an unsafe allocation into a safe one.
        if filters.rise_time_probe_samples(self._fs, cutoff) > self._MAX_RISE_PROBE_SAMPLES:
            self.rise_time_label.setText(
                f"rise time: — (cutoff {cutoff:g} Hz is too low to measure live at "
                f"{self._fs:g} Hz; this configuration will be refused at run time)")
            return
        try:
            # Measured through the EXACT kind/order/cutoff the user selected — the same
            # three values params() hands detect_events — so the readout can never
            # diverge from the real resolution floor the detector will apply.
            # The fs/2.5 RuntimeWarning is caught rather than left on stderr: it is
            # appended to this very label below, in words, which is where the user can
            # actually see it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rise_s = filters.measure_rise_time(
                    self._fs, cutoff, order=order, kind=kind)
        except Exception as exc:  # noqa: BLE001 — a readout must never crash the panel
            self.rise_time_label.setText(f"rise time: unavailable ({exc})")
            return
        rise_us = rise_s * 1e6
        floor_us = 2.0 * rise_us
        design = self._KIND_LABELS.get(kind, kind)
        design = design if kind in filters.KINDS_WITHOUT_ORDER else f"{order}-pole {design}"
        text = (
            f"rise time: {rise_us:.0f} µs ({design}, zero-phase)  →  dwells below "
            f"~{floor_us:.0f} µs are filter-limited (depth attenuated, dwell biased long)."
        )
        # The kernel warns above fs/2.5 that the transition band is truncated and the
        # nominal order is meaningless — a stderr RuntimeWarning nothing in the GUI ever
        # showed. Without this the readout prints a flattering 3 µs floor for a filter
        # that is barely attenuating anything.
        if cutoff > self._fs / 2.5:
            text += (
                f"  ⚠ above fs/2.5 ({self._fs / 2.5:g} Hz): the transition band is "
                "truncated by Nyquist, so this filter barely attenuates anything and "
                "the order is meaningless."
            )
        self.rise_time_label.setText(text)

    def _raise_filter_page(self, idx: int) -> None:
        """Raise stage ``idx``'s page WITHOUT moving the Preprocess column.

        ``QStackedWidget.setCurrentIndex`` hides the outgoing page, and when the widget
        that had focus is ON that page Qt hands focus to the incoming page's counterpart
        (``QStackedLayout::setCurrentIndex``: "try to move focus onto the incoming widget
        if focus was somewhere on the outgoing widget"). Inside a ``QScrollArea`` that
        second half is a scroll: measured on a real MainWindow at 1512x900 with his apo
        ABF loaded, clicking the Low-pass page's ``enable`` box and then picking Median
        from the selector directly above it took the column from 0 to 334 of 2020 in a
        457 px viewport — the stage combo the pointer was resting on went from window
        y=467 to y=133, both baseline view boxes from 347/377 to 13/43, and "Back to
        default" / "Use these values" to y=-22, i.e. off the top. The same switch with
        focus off the page moved 0 px, which is what isolates the cause to the focus
        transfer rather than to the layout: in this in-page layout the pages are all the
        same height and the stack's own size hint does not change when one is raised.
        (The hosted stack in the Signal Conditioning window sizes to the shown page; it
        sits in its own scroll area below the stage list, so the rule is the same.)

        THE ANSWER IS THE COMBO THAT RAISED THE PAGE, the same partner-widget shape
        ``plots_panel._retire_history_button`` uses for its Undo/Redo pair: focus is
        moved to ``filter_selector`` BEFORE the swap, so ``prev->isAncestorOf(fw)`` is
        false and Qt transfers nothing. The combo is the control the user just used, so
        it is on screen by construction, and a plain ``setFocus`` does not route through
        ``QScrollArea.focusNextPrevChild``, so it scrolls nothing itself — measured 0 px
        across all 8 stages in both directions with focus parked on a page each time.
        Focus is not simply dropped: the keyboard user who tabbed into the group keeps a
        place in the chain, one row above the page they are editing.

        Costs one ``focusWidget()`` call per stage change and does nothing at all unless
        focus is on the page being hidden — the only case Qt would otherwise resolve by
        scrolling the column. The page swap itself is unchanged.
        """
        stack = self.filter_stack
        if not 0 <= idx < stack.count():
            return
        # The STACK's window: while hosted it is the Signal Conditioning window, and the
        # partner (its stage list) is the control that raised the page there.
        window = stack.window()
        focused = window.focusWidget() if window is not None else None
        if focused is not None and stack.isAncestorOf(focused):
            (self._stage_focus_partner or self.filter_selector).setFocus(Qt.OtherFocusReason)
        stack.setCurrentIndex(idx)
        self.stageShown.emit(self._EDITOR_STAGE_KEYS[idx])

    # -- frequency-response preview ---------------------------------------
    def _response_args(self) -> tuple[str, dict[str, Any]] | None:
        """``(stage, kwargs)`` for :func:`filters.response` describing the SELECTED
        filter page, or ``None`` for a stage with no transfer function (the median).

        Keyed off the filter combo's index, which is the same index the page stack uses,
        so the curve always belongs to the controls on screen.
        """
        idx = self.filter_selector.currentIndex()
        if idx == 0:
            return "lowpass", {
                "cutoff_hz": float(self.filter_cutoff_hz.value()),
                "order": int(self.filter_order.value()),
                "kind": self.filter_kind_choice(),
            }
        if idx == 1:
            return "highpass", {
                "cutoff_hz": float(self.hp_cutoff_hz.value()),
                "order": int(self.hp_order.value()),
                "kind": self.hp_kind_choice(),
            }
        if idx == 2:
            return "notch", {
                "freq_hz": float(self.notch_freq_hz.value()),
                "q": float(self.notch_q.value()),
                "harmonics": int(self.notch_harmonics.value()),
                # WITHOUT THIS THE PLOTTED COMB IS NOT THE FILTER. `filters.response`
                # designs from the same resolver `notch` does, so handing it the line
                # list is what keeps the preview showing the six or nine lines that
                # will actually run rather than the three the series spins still hold.
                "freqs": self.notch_lines_choice(),
                # A Q=30 notch at 60 Hz is ~2 Hz wide, five decades below Nyquist: on the
                # default grid the trough falls between samples and the notch looks far
                # shallower than it is. Extra points are microseconds here.
                "n": 4096,
            }
        if idx == 3:
            return None                       # median: non-linear, no |H(f)|
        if idx == 4:
            return None                       # baseline drift: an estimator, not a filter
        if idx == 5:
            return "running_mean", {"window_s": float(self.rmean_window_s.value())}
        if idx == 6:
            return "savgol", {
                "window_s": float(self.savgol_window_s.value()),
                "polyorder": int(self.savgol_polyorder.value()),
            }
        return None                           # normalize: a scalar rescale, no |H(f)|

    def _set_response_title(self, title: str) -> None:
        self.response_plot.setTitle(title)
        if self._chain_editor_hosted:
            # pyqtgraph makes a title's text width the plot's minimum width; in the
            # workspace's narrower pane that clipped the plot itself. Let it elide instead.
            self.response_plot.getPlotItem().titleLabel.setMinimumWidth(0)

    def _clear_response(self, note: str) -> None:
        self.response_curve.setData([], [])
        self.response_cutoff_line.setVisible(False)
        self.response_note.setText(note)

    @staticmethod
    def _response_state_note(label: str, enabled: bool) -> str:
        """Plain-language status for the selected stage's design preview."""
        if enabled:
            if label == "High-pass":
                return (
                    "High-pass is ON for candidate localization only and is omitted "
                    "from quantitative measurement."
                )
            return f"{label} is ON and included in the execution chain."
        return f"{label} is OFF; this is a design preview only and is not applied."

    def _active_linear_response_stages(
        self,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """The enabled chain split into plottable LTI stages and honest omissions.

        Order mirrors :func:`analysis.modules.preprocess_for_detection`, with the
        detector-owned low-pass last. Magnitude multiplication is commutative, but
        preserving execution order keeps this recipe useful beside the visible chain
        description and prevents a future stateful stage from being casually reordered.
        """
        stages: list[dict[str, Any]] = []
        omitted: list[str] = []
        if self.notch_enabled.isChecked():
            stages.append({
                "stage": "notch",
                "freq_hz": float(self.notch_freq_hz.value()),
                "q": float(self.notch_q.value()),
                "harmonics": int(self.notch_harmonics.value()),
                "freqs": self.notch_lines_choice(),
            })
        if self.median_enabled.isChecked():
            omitted.append("Median (nonlinear rank filter)")
        if self.hp_enabled.isChecked():
            stages.append({
                "stage": "highpass",
                "cutoff_hz": float(self.hp_cutoff_hz.value()),
                "order": int(self.hp_order.value()),
                "kind": self.hp_kind_choice(),
            })
        if self.detrend_enabled.isChecked():
            omitted.append("Baseline drift (signal-dependent estimator)")
        if self.rmean_enabled.isChecked():
            stages.append({
                "stage": "running_mean",
                "window_s": float(self.rmean_window_s.value()),
            })
        if self.savgol_enabled.isChecked():
            stages.append({
                "stage": "savgol",
                "window_s": float(self.savgol_window_s.value()),
                "polyorder": int(self.savgol_polyorder.value()),
            })
        if self.normalize_enabled.isChecked():
            omitted.append("Normalize (data-dependent affine/nonlinear rescale)")
        if self.filter_enabled.isChecked():
            stages.append({
                "stage": "lowpass",
                "cutoff_hz": float(self.filter_cutoff_hz.value()),
                "order": int(self.filter_order.value()),
                "kind": self.filter_kind_choice(),
            })
        return stages, omitted

    def _update_combined_response_plot(self) -> None:
        """Draw the enabled linear cascade, naming the dual path when HP is on."""
        stages, omitted = self._active_linear_response_stages()
        highpass_localization = self.hp_enabled.isChecked()
        response_scope = (
            "Candidate-localization" if highpass_localization else "Active-chain")
        self._set_response_title(
            f"{response_scope} response — {len(stages)} linear stage(s) ON"
            if not self._chain_editor_hosted else
            f"{response_scope} response — {len(stages)} stage{'s' * (len(stages) != 1)}")
        self.response_curve.setPen(pg.mkPen("#E69F00", width=2, style=Qt.SolidLine))
        self.response_cutoff_line.setVisible(False)
        if self._fs is None:
            self._clear_response(
                f"{response_scope} response needs a channel with a known sample rate.")
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f, mag = filters.combined_response(stages, self._fs, n=2048)
        except Exception as exc:  # noqa: BLE001 — a preview must never crash the panel
            self._clear_response(f"{response_scope} response unavailable ({exc})")
            return

        db = filters.response_db(mag)
        self.response_curve.setData(f, db)
        self.response_plot.setYRange(max(-80.0, float(np.min(db)) - 3.0),
                                     max(3.0, float(np.max(db)) + 1.0))
        if stages:
            names = " → ".join(str(stage["stage"]).replace("_", " ") for stage in stages)
            note = (
                f"{response_scope} linear cascade: {names}. This curve is the "
                "pointwise product "
                "on one shared grid and includes the steady-state zero-phase |H|² "
                "semantics. Finite-record end padding is reported by the relevant stage."
            )
        else:
            note = "No enabled linear stages: the active linear response is unity (0 dB)."
        if highpass_localization:
            note += (
                " This curve includes high-pass for candidate localization. The "
                "classical event-table measurement cascade omits high-pass; its baseline, depth, "
                "fractional blockade, mean/max amplitude and SNR come from that companion "
                "branch, while boundary-dependent timing and ECD limits inherit the "
                "localized candidates. Every other enabled stage still applies, including "
                "Normalize when selected."
            )
        if omitted:
            note += (
                " Not included in |H(f)| because no honest fixed transfer function "
                "exists: " + "; ".join(omitted) + "."
            )
        self.response_note.setText(note)

    def _update_response_plot(self, *_a) -> None:
        """Redraw |H(f)| for the selected stage at its current parameters.

        Everything numeric here comes from :func:`filters.response`; this method only
        decides what to draw and what to say about it. The curve shows the selected
        design's response (squared for zero-phase IIR stages), while the title and note
        explicitly distinguish an enabled stage from an OFF design preview.
        """
        if self.response_chain_check.isChecked():
            self._update_combined_response_plot()
            return

        idx = self.filter_selector.currentIndex()
        label = self._filter_specs[idx][0] if idx < len(self._filter_specs) else ""
        enabled = bool(
            idx < len(self._filter_specs) and self._filter_specs[idx][1].isChecked())
        state = (
            "ON · candidate localization only"
            if enabled and label == "High-pass"
            else "ON · included in execution chain"
            if enabled else "OFF · design preview only")
        if self._chain_editor_hosted:
            # The workspace's preview pane is ~380 px: the long form was clipped mid-word.
            state = ("ON · localization only" if enabled and label == "High-pass"
                     else "ON" if enabled else "OFF · design preview")
        self._set_response_title(f"{label} response — {state}")
        self.response_curve.setPen(pg.mkPen(
            "#56B4E9" if enabled else "#8b9199", width=2,
            style=Qt.SolidLine if enabled else Qt.DashLine))
        state_note = self._response_state_note(label, enabled)
        spec = self._response_args()
        if spec is None:
            if idx == 3:
                self._clear_response(
                    f"{state_note} A sliding median is a NON-LINEAR rank filter: it has "
                    "no transfer "
                    "function, so there is no |H(f)| to draw. It deletes spikes shorter "
                    "than about half its window while leaving a square edge intact — that "
                    "behaviour cannot be expressed as a frequency response.")
            elif idx == 7:
                # |H(f)| would be a flat line at 1/scale, which is true and useless.
                # The thing worth saying about this stage is what it costs, so the
                # space the curve would have occupied says that instead.
                self._clear_response(
                    f"{state_note} Normalize is data-dependent, and model-input mode "
                    "also clips, so there is no fixed |H(f)| to draw. It changes the "
                    "UNIT: the detector then measures a dimensionless trace. "
                    "Current-derived depth/baseline readouts and pA/pA·s export fields "
                    "retain their legacy labels while no longer being in physical units; "
                    "fractional blockade (dI/I0) is invalid after centering.")
            else:
                self._clear_response(
                    f"{state_note} Baseline drift removal is an ESTIMATOR (it fits and "
                    "subtracts a slow "
                    "open-pore envelope), not a linear filter, so it has no single |H(f)|. "
                    "Its effect depends on the trace it is fitted to.")
            return
        if self._fs is None:
            self._clear_response(
                f"{state_note} Select a channel with a known sample rate to see its "
                "frequency response.")
            return

        stage, kwargs = spec
        try:
            # The kernel's fs/2.5 RuntimeWarning is real but belongs in the rise-time
            # readout (where _update_rise_time states it in words), not on stderr once
            # per redraw — and the truncated transition band is visible in the very
            # curve being drawn.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f, mag = filters.response(stage, self._fs, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a preview must never crash the panel
            self._clear_response(f"{state_note} Response unavailable ({exc})")
            return

        db = filters.response_db(mag)
        self.response_curve.setData(f, db)
        # Log x mode wants log10 coordinates for the marker line; y is already dB.
        self.response_plot.setYRange(max(-80.0, float(np.min(db)) - 3.0),
                                     max(3.0, float(np.max(db)) + 1.0))

        corner = kwargs.get("cutoff_hz") or kwargs.get("freq_hz")
        if corner:
            self.response_cutoff_line.setPos(np.log10(float(corner)))
            self.response_cutoff_line.setVisible(True)
        else:
            self.response_cutoff_line.setVisible(False)
        self.response_note.setText(
            f"{state_note} {self._response_note_text(stage, kwargs, f, mag)}")

    def _response_note_text(
        self, stage: str, kwargs: dict[str, Any], f: np.ndarray, mag: np.ndarray
    ) -> str:
        """The one-line verdict under the curve: what the selected design does.

        Only reached from :meth:`_update_response_plot` after it has established that the
        rate is known, so ``self._fs`` is never None here."""
        fs = float(self._fs or 0.0)
        if stage in ("lowpass", "highpass"):
            stage_label = "Low-pass" if stage == "lowpass" else "High-pass"
            nominal = float(kwargs["cutoff_hz"])
            at_nominal = float(np.interp(nominal, f, filters.response_db(mag)))
            true_corner = filters.response_crossing_hz(f, mag)
            kind = kwargs["kind"]
            zero_phase_note = (
                "single-pass symmetric kernel"
                if kind in filters.KINDS_WITHOUT_ORDER
                else "zero-phase, so |H| is SQUARED"
            )
            text = (
                f"{stage_label}: at the nominal {nominal:g} Hz the designed response is "
                f"{at_nominal:.1f} dB — {zero_phase_note}."
            )
            if true_corner:
                text += (
                    f" The true -3 dB corner is {true_corner:.0f} Hz "
                    f"({true_corner / nominal:.2f}× the number in the box)."
                )
            return text
        if stage == "notch":
            explicit = kwargs.get("freqs")
            lines = filters.notch_frequencies(
                fs, float(kwargs["freq_hz"]), harmonics=int(kwargs["harmonics"]),
                freqs=explicit)
            deepest = float(filters.response_db(mag).min())
            narrowest = min(lines) if lines else float(kwargs["freq_hz"])
            spelling = "explicit line list" if explicit else "fundamental + harmonics"
            settle_s = filters.notch_settle_s(
                fs, float(kwargs["freq_hz"]), q=float(kwargs["q"]),
                harmonics=int(kwargs["harmonics"]), freqs=explicit)
            settle_samples = int(np.ceil(settle_s * fs))
            cost = (
                f" Required settling pad: {settle_s:.3g} s "
                f"({settle_samples:,} samples per side)."
            )
            if settle_samples > filters.MAX_NOTCH_PAD_SAMPLES:
                cost += (
                    f" UNSAFE on a long trace: this exceeds the "
                    f"{filters.MAX_NOTCH_PAD_SAMPLES:,}-sample kernel limit and the run "
                    "will be refused; remove or raise the lowest-frequency line, or "
                    "lower Q."
                )
            return (
                f"Notching {len(lines)} line(s) ({spelling}): "
                f"{', '.join(f'{x:g}' for x in lines)} Hz, deepest {deepest:.0f} dB. "
                "Lines at or above Nyquist are skipped. A Q of "
                f"{float(kwargs['q']):g} at {narrowest:g} Hz is a "
                f"~{narrowest / float(kwargs['q']):.2g} Hz-wide notch, "
                "and it rings at each end of the trace." + cost
            )
        # The two window smoothers: their -3 dB point is the number that matters, because
        # it is routinely far BELOW the low-pass cutoff and nothing else in the app says so.
        window_s = float(kwargs["window_s"])
        n_samp = (
            filters.running_mean_samples(fs, window_s)
            if stage == "running_mean"
            else filters.savgol_window_samples(fs, window_s, int(kwargs["polyorder"]))
        )
        if stage == "running_mean" and n_samp == 1:
            return (
                f"A {window_s * 1e6:g} µs window is 1 sample at this rate: this stage is "
                "the IDENTITY — it is enabled but changes nothing. Widen the window.")
        corner = filters.response_crossing_hz(f, mag)
        text = f"{n_samp} samples wide."
        if corner:
            text += (
                f" Effective -3 dB at {corner:.0f} Hz — compare this with the low-pass "
                "cutoff: whichever is LOWER sets the real bandwidth of the trace you "
                "detect on, and the rise-time readout only knows about the low-pass."
            )
        if stage == "savgol":
            peak = float(filters.response_db(mag).max())
            if peak > 0.1:
                text += (
                    f" Pass-band peaking of +{peak:.1f} dB: a Savitzky-Golay overshoots "
                    "the plateau of a square blockade and leaves an opposite-sign lobe "
                    "flanking each edge.")
        return text

    def _emit_params(self, *_a) -> None:
        self.paramsChanged.emit(self.params())


# ---------------------------------------------------------------------------
# StatsPanel
# ---------------------------------------------------------------------------
class StatsPanel(QWidget):
    """Summary table, FDR readout, fit controls and a FitResult text area.

    Emits:
        fitRequested(dict): {'n_components_dwell', 'n_components_depth',
                             'use_efficiency', 'use_filter_limited'}.
    """

    fitRequested = Signal(dict)

    #: (row label, stats key). Order = the rows on screen.
    #:
    #: THE THREE ROWS THAT WERE MISSING. The app's own rule is that a depth does not
    #: travel alone: it is a difference from an open-pore reference measured under a
    #: named window, and two depths measured under different windows are not
    #: comparable (`nanoscope.analysis.baseline`, and the export sidecar carries
    #: `baseline_fingerprint` for exactly this reason). This table showed a
    #: *dimensionless ratio* and nothing else -- no depth in pA, no open-pore
    #: current, no fingerprint -- so the one surface a user reads after every
    #: detection was the one surface that could not answer "measured against what?".
    #: All three were reachable only by scrolling the event table sideways or by
    #: exporting. They are now here, beside the ratio they explain.
    #:
    #: EVERY ROW NAMES THE ROWS IT COUNTED, and every amplitude row names the unit it is
    #: in. The label is a TEMPLATE: ``{units}`` is filled from the channel the detection
    #: ran on (see :meth:`set_units`), because this panel is not entitled to assert "pA"
    #: on a channel whose file declared no units -- the PSD window, the I-V window and
    #: the Calibration panel all say "units unknown" there, and this one used to print
    #: "median depth (pA) 0.07274" on the same recording, in the same session, whose own
    #: loader note says the magnitudes suggest nA (FE-04). And the four medians below the
    #: rate are taken over the ACCEPTED rows, like the rate above them and like
    #: :attr:`EventSet.accepted` ("what figures and fits should use"), so the Summary
    #: cannot present "n accepted 85" directly above a median of 115 rows (FE-02).
    _SUMMARY_ROWS = [
        ("n events", "n_events"),
        ("n accepted", "n_accepted"),
        ("rate (Hz, accepted)", "rate_accepted_hz"),
        ("median dwell (µs, accepted)", "median_dwell_us"),
        ("median depth ({units}, accepted)", "median_depth_pA"),
        ("median dI/I0 (accepted)", "median_frac_depth"),
        ("open pore (median stamped, {units}, accepted)", "median_baseline_pA"),
        ("baseline window", "baseline_window"),
        ("baseline fingerprint", "baseline_fingerprint"),
        ("n filter-limited", "n_filter_limited"),
        ("n pores", "n_pores"),
    ]

    #: Fixed, per-row tooltips: what the number IS, in the terms the rest of the app
    #: uses for it. Keyed by stats key so a row can be moved without losing its
    #: explanation.
    _ROW_TIPS = {
        "rate_accepted_hz": (
            "Accepted events divided by the analysed duration from provenance — the "
            "same basis as the Voltage-dependence panel's capture-rate points and the "
            "same number the apo-ferritin report's master table quotes.\n\n"
            "It is NOT n events / duration. Those two differ whenever a row has been "
            "rejected or auto-flagged: on the -600 mV merged detection, 240 rows of "
            "which 239 are accepted, they read 3.992 and 4.008 Hz. A rate has to name "
            "the rows it counted, so this one does, in its own label."),
        "median_dwell_us": (
            "Median of the event table's dwell_s over the ACCEPTED rows of this "
            "detection — the same rows the three medians below it are taken over, and "
            "the same rows the rate above it counts."),
        "median_depth_pA": (
            "Median of the event table's depth_pA over the ACCEPTED rows of this "
            "detection — the same column, the same rows, as the dI/I0 below it.\n\n"
            "Rejecting rows moves it: it used to be taken over ALL rows, so it sat "
            "unchanged directly beneath an 'n accepted' that had moved, and disagreed "
            "with the Plots window (which filters to accepted events by default) about "
            "the same column of the same detection.\n\n"
            "This is an INSTRUMENT number, not a calibrated current: it is only as "
            "physical as the gain in Calibration. The unit in the row label is the "
            "channel's own (Calibration's target units, else the file's) — it reads "
            "'units unknown' when neither has declared one, because this panel does "
            "not get to invent a unit the rest of the app refuses to name."),
        "median_frac_depth": (
            "Median of frac_depth = depth_pA / |baseline_pA| over the ACCEPTED rows — "
            "the depth as a fraction of the open-pore current it was measured "
            "against."),
        "median_baseline_pA": (
            "Median over the ACCEPTED rows of the baseline_pA STAMPED ON EACH EVENT by "
            "the characterisation baseline: the open-pore reference every depth in "
            "this table was subtracted from, measured locally per event under the "
            "window named below.\n\n"
            "The unit in the row label is the channel's own, and reads 'units unknown' "
            "when neither the file nor Calibration has declared one.\n\n"
            "The I-V / Conductance window reports an open-pore current too, and it is "
            "NOT this number: it takes one I₀ for the whole recording by its own "
            "estimator. On the owner's apoferritin recordings the two differ by up to "
            "5.1% (−49.13 vs −51.78 pA at −700 mV). Quote this one beside a depth "
            "from this table; quote that one beside a conductance."),
        "baseline_window": (
            "The open-pore estimator's window pair (rough / base) this detection was "
            "measured under, from its provenance."),
        "baseline_fingerprint": (
            "The identity of the open-pore estimator configuration these depths were "
            "measured under.\n\n"
            "Depths carrying DIFFERENT fingerprints are not poolable. This is the "
            "value to quote beside any depth that leaves the app, and the one the "
            "export sidecar records."),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        #: The units the amplitude rows are captioned in — the channel's own, pushed in
        #: by the owner (:meth:`set_units`). ``""`` means NOBODY has declared one, and
        #: that is what a fresh HDF5/CSV channel is in; the labels then say so rather
        #: than filling the blank with "pA".
        self._units = ""

        # -- summary table ------------------------------------------------
        self.summary_table = QTableWidget(len(self._SUMMARY_ROWS), 1)
        self.summary_table.setHorizontalHeaderLabels(["value"])
        self.summary_table.setVerticalHeaderLabels(self._row_labels())
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i, (_, key) in enumerate(self._SUMMARY_ROWS):
            item = QTableWidgetItem("—")
            tip = self._ROW_TIPS.get(key, "")
            if tip:
                item.setToolTip(tip)
                header = self.summary_table.verticalHeaderItem(i)
                if header is not None:
                    header.setToolTip(tip)
            self.summary_table.setItem(i, 0, item)
        # EVERY ROW VISIBLE AT THE DOCK'S NATURAL WIDTH. A QTableWidget defaults to a
        # size hint of a few rows and scrolls the rest: at the Inspector dock's own
        # width this showed 5.5 of its 7 rows, with "n pores" off the bottom behind a
        # scrollbar nobody looks for inside a panel that is already in a scroll area.
        # A fixed-row readout must size to its content and let the ONE outer scroll
        # area do the scrolling — two nested scrollbars is how a row disappears.
        self.summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.summary_table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._fit_summary_height()
        root.addWidget(QLabel("Summary"))
        root.addWidget(self.summary_table)

        # -- the caveats those numbers were produced under ----------------
        #
        # THE CHANNEL HAD NO SURFACE. ``AnalysisResult.warnings`` is the general way a
        # module says what its numbers must not be trusted for -- an undetermined
        # polarity, a filter-limited fraction, a boundary-clipped row, a baseline that
        # fell back, a health check that fired, and above all a detection run at an
        # open-pore window NOBODY CHOSE. Four call sites in ``main_window`` persist the
        # tuple into the project file and restore it, and every one of them was writing
        # to a channel that no widget in the app read: a sweep of every text-bearing
        # widget in a live MainWindow found zero occurrences of the unprobed-default
        # advisory's own phrases. On the owner's -600 mV recording that silence is worth
        # 317 events / 85.68 pA / dI/I0 2.423 at the untouched default against 276 /
        # 87.59 / 2.523 after measuring the recording -- shown with no caveat anywhere.
        #
        # It goes HERE, under the Summary table, because this is the surface a user
        # reads after a detection and because the sentence qualifies the two medians
        # directly above it. Rendering the WHOLE tuple rather than one known string is
        # the actual fix: the next warning anyone appends is visible by construction,
        # which is the invariant ``test_warnings_surface.py`` pins.
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # Same amber as the staleness banner and the estimator-comparison dialog's
        # health block -- this is an advisory, not a failure, and it fires on every
        # unprobed detection. A GLYPH as well as the colour, for the reason the
        # dI/I0 > 1 marker carries one: a screenshot is how these numbers leave the app.
        self.warnings_label.setStyleSheet("color:#ffd479;")
        self.warnings_label.setVisible(False)
        root.addWidget(self.warnings_label)

        # -- FDR readout --------------------------------------------------
        self.fdr_label = QLabel(self._NO_FDR_TEXT)
        self.fdr_label.setWordWrap(True)
        root.addWidget(self.fdr_label)

        # -- fit controls -------------------------------------------------
        fit_box = QGroupBox("Fits")
        fit_form = QFormLayout(fit_box)
        self.n_components_dwell = QSpinBox()
        self.n_components_dwell.setRange(1, 4)
        self.n_components_dwell.setValue(1)
        self.n_components_depth = QSpinBox()
        self.n_components_depth.setRange(1, 4)
        self.n_components_depth.setValue(1)
        self.use_efficiency = _ElidingCheckBox("use measured efficiency (needs recovery map)")
        self.use_efficiency.setEnabled(False)  # enabled once a recovery map exists

        # Filter-limited events are EXCLUDED from the distribution fits by default.
        # Their dwell is within ~2x the filter rise time, so the low-pass never let
        # the current reach the true blockade level: depth attenuated, dwell biased
        # long. The scatter figure already draws them as open markers over a shaded
        # band; the fits are at least as honest about them.
        self.use_filter_limited = _ElidingCheckBox("include filter-limited events")
        self.use_filter_limited.setChecked(False)
        self.use_filter_limited.setToolTip(
            "Off (default): events whose dwell is within ~2x the filter rise time are "
            "excluded from the dwell and depth fits — their depth is attenuated and "
            "their dwell biased long, and the count excluded is reported with the fit.\n"
            "On: they are fit alongside the resolved events."
        )
        self.filter_limited_label = QLabel("")
        self.filter_limited_label.setWordWrap(True)
        self._n_filter_limited = 0
        self.use_filter_limited.toggled.connect(
            lambda _on: self._refresh_filter_limited_label())

        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip(
            "Fit the dwell and depth distributions of the ACCEPTED events of the current "
            "detection (exponential / log-normal mixtures), and store each fit with its "
            "provenance.")
        # The Fit gate. This button used to have NO gate at all: with no detection it accepted
        # the click and then wrote "Run detection first…" into the results box — a failure
        # discovered only after acting, in a place the user was not looking. It is now disabled
        # up front with the reason spelled out beside it, like every other gated control here.
        self._fit_gate_label = QLabel("")
        self._fit_gate_label.setWordWrap(True)
        self._fit_gate_label.setStyleSheet("color:#c9a227;")
        fit_form.addRow("dwell components", self.n_components_dwell)
        fit_form.addRow("depth components", self.n_components_depth)
        fit_form.addRow("", self.use_efficiency)
        fit_form.addRow("", self.use_filter_limited)
        fit_form.addRow("", self.filter_limited_label)
        fit_form.addRow(self.fit_button)
        fit_form.addRow("", self._fit_gate_label)
        self.fits_page = QWidget()
        fits_layout = QVBoxLayout(self.fits_page)
        fits_layout.setContentsMargins(0, 0, 0, 0)
        fits_layout.addWidget(fit_box)
        root.addWidget(self.fits_page)
        self.set_detection_present(False)

        # -- results text -------------------------------------------------
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        mono = QFont(self.results.font())
        mono.setStyleHint(QFont.Monospace)
        mono.setFamily("monospace")
        self.results.setFont(mono)
        self.results.setPlaceholderText("Fit results appear here.")
        fits_layout.addWidget(self.results, 1)

        self.fit_button.clicked.connect(
            lambda: self.fitRequested.emit(
                {
                    "n_components_dwell": int(self.n_components_dwell.value()),
                    "n_components_depth": int(self.n_components_depth.value()),
                    "use_efficiency": bool(
                        self.use_efficiency.isEnabled() and self.use_efficiency.isChecked()
                    ),
                    "use_filter_limited": bool(self.use_filter_limited.isChecked()),
                }
            )
        )

    #: dI/I0 above this is not a blockade fraction — see :meth:`frac_depth_note`.
    _FRAC_DEPTH_ENHANCEMENT = events_mod.FRAC_DEPTH_ENHANCEMENT

    @classmethod
    def frac_depth_note(cls, value: Any) -> str:
        """The warning that must travel with a ``dI/I0`` above 1, or ``""``.

        THE SENTENCE MOVED TO :func:`nanoscope.analysis.events.frac_depth_note`, beside
        the column it is about. Here it had two consumers -- this Summary row and the
        CSV sidecar -- and could have no more: the cross-panel import guard means no
        other panel can reach a classmethod on this one, so the Event Inspector, the
        Measure read-out and the publication depth figure all went on printing
        ``dI/I0 2.36`` bare. This delegate stays so the two existing call sites and
        their tests keep working, and so there is still exactly one sentence.
        """
        return events_mod.frac_depth_note(value)

    # -- row labels (they carry the units and the row set) ----------------
    def _row_labels(self) -> list[str]:
        """The vertical header texts, with ``{units}`` resolved for this channel."""
        units = self._units or _UNKNOWN_UNITS
        return [label.format(units=units) for label, _key in self._SUMMARY_ROWS]

    def set_units(self, units: str | None) -> None:
        """Caption the amplitude rows in the units of the channel the detection ran on.

        ``""``/``None`` is the state a fresh HDF5 or CSV channel is in — the file
        declared none and Calibration has not set any — and it is rendered as
        :data:`_UNKNOWN_UNITS`, never as "pA". The DataFrame columns keep their frozen
        ``_pA`` names (``analysis/characterize.py``: "the SCHEMA column names keep their
        `_pA` suffix as the frozen cross-module contract regardless"); this is the free
        text on screen, and a column name is not a unit.

        Idempotent and safe to call before any summary exists: the labels are the only
        thing it touches, so the numbers on screen are never re-described without being
        re-computed.
        """
        text = str(units or "")
        if text == self._units:
            return
        self._units = text
        self.summary_table.setVerticalHeaderLabels(self._row_labels())
        # setVerticalHeaderLabels REPLACES the header items, taking the tooltips with
        # them — re-attach them, or hovering the row name stops explaining the row.
        for i, (_, key) in enumerate(self._SUMMARY_ROWS):
            tip = self._ROW_TIPS.get(key, "")
            header = self.summary_table.verticalHeaderItem(i)
            if header is not None and tip:
                header.setToolTip(tip)

    def summary_label(self, key: str) -> str:
        """The row label one summary key is shown under — what the user reads beside the
        number (used by tests and by the owner, like :meth:`summary_text`)."""
        for i, (_, k) in enumerate(self._SUMMARY_ROWS):
            if k == key:
                header = self.summary_table.verticalHeaderItem(i)
                return header.text() if header is not None else ""
        return ""

    # -- public API -------------------------------------------------------
    def set_summary(self, stats: dict[str, Any]) -> None:
        """Fill the summary table from a plain stats dict (AnalysisPanel builds it)."""
        for i, (_, key) in enumerate(self._SUMMARY_ROWS):
            val = stats.get(key)
            item = self.summary_table.item(i, 0)
            text = self._fmt(val)
            note = self.frac_depth_note(val) if key == "median_frac_depth" else ""
            item.setText(f"{text}  ⚠ >1" if note else text)
            # The marker is a GLYPH plus a colour, never colour alone: a screenshot is
            # exactly how this number leaves the app, and a screenshot of an orange cell
            # read by someone else is a screenshot of a plain number.
            item.setForeground(QBrush(QColor("#ffb454")) if note else QBrush())
            item.setToolTip(
                (note + "\n\n" + self._ROW_TIPS.get(key, "")).strip()
                if note else self._ROW_TIPS.get(key, ""))
        self._fit_summary_height()

    #: Heads the advisory block. A glyph AND a colour (see the label's construction),
    #: and a heading that says what the sentences below are ABOUT rather than shouting
    #: a severity the tuple does not carry.
    WARNINGS_HEADING = "⚠ About these numbers"

    def set_warnings(self, warnings: Any) -> None:
        """Show every caveat on the result whose summary is above, or hide the block.

        Takes ``AnalysisResult.warnings`` WHOLE and unfiltered. It deliberately knows
        nothing about which warning it is rendering: the defect this repairs was a
        single advisory with no surface, and a fix that special-cased that one string
        would leave the next one just as invisible.

        There is no "don't show again". The block is dismissible only by REMOVING THE
        CONDITION -- measure the recording on the Preprocess tab, confirm the windows,
        detect again, and the re-run's warnings tuple no longer carries the sentence,
        so this clears itself. A dismiss button would let a user silence an advisory
        while the wrong number stayed on screen, which is the state the advisory exists
        to prevent.
        """
        items = [t for t in (str(w).strip() for w in (warnings or ())) if t]
        if not items:
            self.warnings_label.clear()
            self.warnings_label.setVisible(False)
            return
        self.warnings_label.setText(
            f"{self.WARNINGS_HEADING}\n• " + "\n• ".join(items))
        # The whole block is the tooltip too: the label word-wraps inside a dock that
        # can be dragged narrow, and a hover must be able to recover the full sentence.
        self.warnings_label.setToolTip("\n\n".join(items))
        self.warnings_label.setVisible(True)

    def warnings_text(self) -> str:
        """What the advisory block currently READS, or ``""`` when nothing is shown.

        Hidden means empty, so a caller cannot mistake a stale hidden label for a
        warning that is on screen.

        ``isVisibleTo(self)`` rather than ``isVisible()``: the question is whether the
        block was hidden by :meth:`set_warnings`, not whether the dock happens to be
        realised. ``isVisible()`` is False for every child of an unshown window, which
        would make this read "" for a panel under test that has warnings up.
        """
        return (self.warnings_label.text()
                if self.warnings_label.isVisibleTo(self) else "")

    def summary_text(self, key: str) -> str:
        """The displayed text of one summary row (used by tests and by the owner)."""
        for i, (_, k) in enumerate(self._SUMMARY_ROWS):
            if k == key:
                return self.summary_table.item(i, 0).text()
        return ""

    def summary_tooltip(self, key: str) -> str:
        """The tooltip of one summary row — where the row's warning actually reads."""
        for i, (_, k) in enumerate(self._SUMMARY_ROWS):
            if k == key:
                return self.summary_table.item(i, 0).toolTip()
        return ""

    def _fit_summary_height(self) -> None:
        """Height = every row + the header + the frame. No internal scrollbar, ever."""
        table = self.summary_table
        total = table.horizontalHeader().height() + 2 * table.frameWidth()
        for i in range(table.rowCount()):
            total += table.rowHeight(i)
        table.setFixedHeight(int(total))

    def clear_summary(self) -> None:
        for i, (_, key) in enumerate(self._SUMMARY_ROWS):
            item = self.summary_table.item(i, 0)
            item.setText("—")
            item.setForeground(QBrush())
            item.setToolTip(self._ROW_TIPS.get(key, ""))
        self.set_filter_limited_count(0)
        # No numbers on screen, so nothing to caveat. Leaving the block up would state
        # a caveat about a table that is no longer there.
        self.set_warnings(())

    _NO_FDR_TEXT = "false-positive rate: not measured (run Null control)"

    def _set_fdr_readout(self, text: str, tip: str = "") -> None:
        """THE ONLY WRITER OF THE FDR READOUT. Text and provenance, together, always.

        A printed number's provenance must be owned by the same lifecycle as the number,
        and the only way to make that true of a QLabel is to leave no way to move one
        without the other. :meth:`set_fdr` has four exits — reset, refusal-with-no-count,
        refusal-with-a-count, and the branch that prints a percentage — and until this
        method existed each set the text itself and remembered the tooltip separately.
        Three remembered; the reset exit did not.

        MEASURED, wired ``MainWindow``, two recordings, real event loop: run *Null
        control…* on recording A, then click recording B (no null control of its own).
        The label correctly became ``false-positive rate: not measured (run Null
        control)`` and the tooltip beneath it still read *"The mean of 5 independent AAFT
        surrogates, over 4 real detections. Per-surrogate counts: … seed = 2029049619 —
        pressing “Null control…” again on this recording with these settings reproduces
        these counts exactly."* Every "this recording" in that sentence was recording A,
        which was no longer on screen — a tooltip that outlives its label is a claim
        about a recording the user is not looking at.

        The default is ``""`` rather than "leave it alone" on purpose: a caller that says
        nothing about provenance is a caller that HAS none, and the honest rendering of
        no provenance is an empty tooltip, never the previous recording's.
        """
        self.fdr_label.setText(text)
        self.fdr_label.setToolTip(tip)

    def set_fdr(self, null_result: dict[str, Any] | None) -> None:
        """Show the FDR from a ``null_control()`` result dict.

        ``None`` resets the readout to "not measured". The FDR belongs to the recording it
        was measured on, so the owner pushes None whenever it repoints at a target that has
        no null control of its own — leaving the previous recording's percentage on screen
        states a measurement that was never made about this pore.

        A REFUSAL IS NOT A NUMBER. ``null_control`` returns ``nan`` when it will not stand
        behind a rate — because the AAFT surrogates are not the same noise as this
        recording, or because there are no real detections to divide by — and puts the
        reason in ``notes`` alongside ``n_event_free``, the false-positive count measured
        on the recording's own event-free noise with no surrogate at all. This tested
        ``fdr is None``, which ``nan`` is not, so it formatted it: the readout on the
        owner's own recordings read **"estimated false-positive rate: nan%"**, and the
        honest number the refusal exists to produce never reached the screen at all. (The
        `nan` path pre-dated the refusal; the refusal is what made it the common case.)
        """
        if not null_result:
            # THE EXIT THAT LEAKED. `None` is pushed by `AnalysisPanel._sync_surrogate_
            # readouts` on every selection change, so this is the path a user reaches by
            # clicking another recording -- and a stale provenance tooltip is at its most
            # convincing precisely here, where the label says the number was never
            # measured. Both halves go, together, through the one writer.
            self._set_fdr_readout(self._NO_FDR_TEXT)
            return
        fdr = null_result.get("estimated_fdr")
        n_sur = null_result.get("per_surrogate")
        n = len(n_sur) if n_sur is not None else null_result.get("n_surrogates", "?")
        if fdr is None or not np.isfinite(float(fdr)):
            notes = tuple(null_result.get("notes") or ())
            n_free = null_result.get("n_event_free")
            if n_free is None:
                text = self._NO_FDR_TEXT
            else:
                text = (
                    f"false-positive rate NOT estimated — {n_free} detection(s) on this "
                    f"recording's own event-free noise (a lower bound, measured with no "
                    f"surrogate)."
                )
            self._set_fdr_readout(text, notes[0] if notes else "")
            return
        # THE BRANCH THAT PRINTS A NUMBER IS THE BRANCH THAT OWES AN EXPLANATION.
        #
        # The refusal above says where its number came from; this one used to clear the
        # tooltip outright -- so the single path that puts a percentage on screen was the
        # single path that said nothing about it. That mattered here more than anywhere,
        # because the percentage is a MEAN OF INDEPENDENT DRAWS and the draws disagree
        # violently: measured through this panel on the owner's -700 mV apo recording,
        # five clicks against the same 1164 real detections, the five surrogate counts
        # inside a single run ran 42 to 643 (15x) and the printed rate ran 13.8 % to
        # 24.0 %.
        #
        # `AnalysisPanel._on_null_requested` now seeds the run and stamps the seed in, so
        # the number reproduces. Reproducibility is not honesty: a repeatable draw from a
        # 15x-wide distribution is a repeatable meaningless number if it is printed bare.
        # So the readout carries the RANGE the surrogates actually spanned beside the
        # mean, and the tooltip carries the counts, the seed that reproduces them, and the
        # event-free count the refusal branch would have quoted instead.
        #
        # A result with no seed in it (an older stored run, or any caller that did not
        # seed) is still shown -- and is labelled NOT REPRODUCIBLE, because the one thing
        # this panel must never do is print an unrepeatable number as though it were a
        # measurement.
        #
        # `is not None`, NOT truthiness: the `n` line above already tolerates any sequence
        # here, and `n_sur or ()` would raise on a numpy array ("truth value of an array
        # with more than one element is ambiguous") — turning a readout into a traceback
        # for a stored result that the pre-fix code displayed fine.
        per = [int(v) for v in n_sur] if n_sur is not None else []
        n_real = null_result.get("n_real")
        spread = self._fdr_spread(per, n_real)
        self._set_fdr_readout(
            f"estimated false-positive rate: {100.0 * fdr:.1f}% (AAFT, {n} surrogates)"
            f"{spread}",
            self._fdr_provenance(null_result, per, n_real),
        )

    @staticmethod
    def _fdr_spread(per: list[int], n_real: Any) -> str:
        """The trailing clause naming the range the surrogates spanned, or "".

        Reads ``— individual surrogates gave 5.6–28.4%``.

        Empty when the run kept no per-surrogate counts, when there is no real count to
        divide by, or when every surrogate lands on the same printed percentage — in which
        case the mean already IS the range and a repeated number reads as a second,
        different claim. Clipped at 100 % exactly where ``estimated_fdr`` is.
        """
        try:
            denom = float(n_real)
        except (TypeError, ValueError):
            return ""
        if not per or not denom > 0:
            return ""
        lo = 100.0 * min(1.0, min(per) / denom)
        hi = 100.0 * min(1.0, max(per) / denom)
        if f"{lo:.1f}" == f"{hi:.1f}":
            return ""
        return f" — individual surrogates gave {lo:.1f}–{hi:.1f}%"

    @staticmethod
    def _fdr_provenance(null_result: Mapping[str, Any], per: list[int],
                        n_real: Any) -> str:
        """Where the printed percentage came from, and whether pressing again repeats it."""
        lines: list[str] = []
        if per and n_real:
            lines.append(
                f"The mean of {len(per)} independent AAFT surrogates, over {n_real} real "
                f"detections. Per-surrogate counts: {', '.join(str(v) for v in per)}.")
            if min(per) > 0 and max(per) >= 2 * min(per):
                # `:.0f` ROUNDED THE ONE NUMBER ON THIS LINE WHOSE JOB IS DISPERSION,
                # and in BOTH directions. Measured on the owner's own -700 mV counts
                # (172, 237, 94, 16, 301): the true ratio is 301/16 = 18.8x and the line
                # read "18.8x rounded to 19x" -- an overstatement. Going the other way,
                # 2.49x reads "2x" (a fifth of the spread rounded away) and 2.5x also
                # reads "2x", because `format` rounds halves to even -- so the one figure
                # that exists to caveat the headline was itself rounded toward reassurance.
                # One decimal, with a bare ".0" dropped so the common wide case still
                # reads "19x" rather than "19.0x".
                ratio = max(per) / min(per)
                span = f"{ratio:.1f}".removesuffix(".0")
                lines.append(
                    f"Those counts span {span}x. The headline figure is "
                    f"one point on a distribution that wide — quote the range, not the point.")
        elif per:
            lines.append(
                f"Per-surrogate detection counts: {', '.join(str(v) for v in per)}.")

        seed = null_result.get("seed")
        if seed is None:
            lines.append(
                "NOT REPRODUCIBLE: this run recorded no seed, so pressing “Null control…” "
                "again will draw different surrogates and print a different percentage.")
        else:
            lines.append(
                f"seed = {seed} — pressing “Null control…” again on this recording with "
                f"these settings reproduces these counts exactly.")

        n_free = null_result.get("n_event_free")
        if n_free is not None:
            lines.append(
                f"For comparison, {n_free} detection(s) on this recording's own event-free "
                f"noise, measured with no surrogate at all.")
        return "\n\n".join(lines)

    _NO_DETECTION_REASON = ("Run detection first — the fits are computed from the accepted "
                            "events of the current detection.")

    def set_detection_present(self, present: bool, reason: str = "") -> None:
        """Gate the Fit button on there being a detection to fit, showing why when there is not.

        Pushed by the owner (:class:`~nanoscope.gui.analysis_panel.AnalysisPanel`) on every
        result / selection change — the same call it already makes on the merge controls, from
        the same state. The spinboxes and toggles stay live so the user can dial a fit in ahead
        of running detection; only the action is gated.
        """
        on = bool(present)
        self.fit_button.setEnabled(on)
        text = "" if on else (reason or self._NO_DETECTION_REASON)
        self._fit_gate_label.setText(text)
        self._fit_gate_label.setVisible(bool(text))
        self.fit_button.setToolTip(
            self.fit_button.toolTip().split("\n\n")[0] + ("" if on else "\n\n" + text))

    def fit_gate_reason(self) -> str:
        """The reason the Fit button is disabled ("" when it is live) — what the user reads."""
        return self._fit_gate_label.text()

    def set_recovery_available(self, available: bool) -> None:
        """Enable the 'use measured efficiency' checkbox once a recovery map exists."""
        self.use_efficiency.setEnabled(bool(available))
        if not available:
            self.use_efficiency.setChecked(False)

    def set_filter_limited_count(self, n: int) -> None:
        """Show how many accepted events the fits will drop as filter-limited.

        The number must be visible BEFORE the fit runs, not only afterwards in the
        result text: it is the difference between the n the summary table shows and
        the n the fit will report."""
        self._n_filter_limited = max(0, int(n))
        self._refresh_filter_limited_label()

    def _refresh_filter_limited_label(self) -> None:
        n = getattr(self, "_n_filter_limited", 0)
        if n <= 0:
            self.filter_limited_label.setText("")
            return
        verb = "INCLUDED in" if self.use_filter_limited.isChecked() else "excluded from"
        self.filter_limited_label.setText(
            f"{n} accepted event(s) are filter-limited — {verb} the fits."
        )

    def set_fit_text(self, text: str) -> None:
        """Show a FitResult.summary() (or any text) in the monospace area."""
        self.results.setPlainText(text)

    # -- internal ---------------------------------------------------------
    @staticmethod
    def _fmt(val: Any) -> str:
        if val is None:
            return "—"
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        if isinstance(val, (float, np.floating)):
            if not np.isfinite(val):
                return "—"
            return f"{val:.4g}"
        return str(val)


# ---------------------------------------------------------------------------
# ScorePanel
# ---------------------------------------------------------------------------
class ScorePanel(QWidget):
    """A compact readout that scores the DETECTOR against the user's curation.

    This is not a summary of the events — it is the model's report card. For the
    CURRENT detection it shows how many of the detector's outputs matched a kept
    event (TP), were rejected as false (FP), and how many real events the user had to
    add by hand because the detector missed them (FN), plus the precision / recall /
    F1 those imply. The AnalysisPanel recomputes it on every curation / edit / add /
    result change and pushes a :class:`~nanoscope.analysis.scoring.ScoreResult` (or
    ``None`` to blank it) through :meth:`set_score`.

    An undefined ratio (``None`` — no detections, nothing real, both zero) renders as
    ``n/a``, never a fabricated ``0``.
    """

    # (row label, ScoreResult field). Order = the readout's rows. Counts first, then
    # the three ratios the counts imply.
    _ROWS: tuple[tuple[str, str], ...] = (
        ("detected", "n_detected"),
        ("truth (kept)", "n_truth"),
        ("matched (TP)", "tp"),
        ("spurious (FP)", "fp"),
        ("missed (FN)", "fn"),
        ("auto-rejected", "n_auto_rejected"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("F1", "f1"),
    )
    _RATIO_KEYS = frozenset({"precision", "recall", "f1"})

    #: What the three ratios read while the truth set is still the detector's own
    #: output. Not a number, because there is no measurement here to round.
    _TAUTOLOGY_TEXT = "not scored yet"

    _TAUTOLOGY_TIP = (
        "Nothing has been rejected BY YOU and nothing hand-added, so the truth these "
        "are scored against IS the detector's own output: precision, recall and F1 "
        "are 1.000 by construction and measure nothing. Rows the detector rejected on "
        "its own (boundary-clipped, over the max dwell, non-finite) are its flags, not "
        "your verdict, and are counted under 'auto-rejected' rather than as spurious.\n\n"
        "Reject a spurious detection, or mark a missed event with the Measure tool, "
        "and the ratios start describing the detector."
    )

    @classmethod
    def is_tautological(cls, score: ScoreResult | None) -> bool:
        """True when the ratios are 1.000 BY CONSTRUCTION rather than by measurement.

        THE THREE MOST CONVINCING NUMBERS ON SCREEN. Straight after a detection the
        readout showed precision / recall / F1 = 1.000 — scored against an uncurated
        truth set, which is to say against the detector's own output. The group title
        says "vs your curation" and is honest, but a screenshot of three 1.000s is not,
        and on the owner's apoferritin run those were the numbers a reader's eye went
        to first.

        THE QUESTION IS "WERE THERE ANY USER GESTURES", asked of the score's own
        ``fp`` and ``fn`` — which :func:`nanoscope.analysis.scoring.score_detection`
        now defines as the USER's rejections and the USER's hand-added events, with
        the detector's own automatic rejections (``clipped_by_boundary``,
        ``dwell>max_dwell_s``, ``non_finite``) counted apart as ``n_auto_rejected``.
        The earlier test read ``fp == 0 and fn == 0`` off a ``fp`` that INCLUDED those
        rows, so a single boundary-clipped event defeated the guard: sweep 7 measured
        the owner's -600 mV file rendering precision 0.997 / F1 0.998 straight after
        Detect, and with a 5 ms max dwell "truth (kept) 77, spurious (FP) 240,
        precision 0.243, F1 0.391" — zero curation gestures, caption hidden, the
        detector's honest self-flagging published as its own false-positive rate
        (S7-DET2). The guard fired on the recordings that needed it least and went
        silent on the ones that needed it most.

        The sentence this puts on screen stays true even for a user who reviewed
        every event and rejected none — the ratios still follow from the construction
        rather than from a disagreement that was measured. A table that exists only
        as auto-rejections (everything over the max dwell) is still an uncurated
        table, so it is tautological too; an EMPTY score (nothing detected, nothing
        auto-rejected) is not — there is nothing to withhold.
        """
        if score is None:
            return False
        n_rows = (int(getattr(score, "n_detected", 0))
                  + int(getattr(score, "n_auto_rejected", 0)))
        gestures = int(getattr(score, "fp", 0)) + int(getattr(score, "fn", 0))
        return n_rows > 0 and gestures == 0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Detector score (vs your curation)")
        box.setToolTip(
            "Scores the DETECTOR against your curated truth: matched (kept), spurious "
            "(rejected) and missed (hand-added) events, and the precision/recall/F1 "
            "they imply. A ratio with a zero denominator shows as n/a.")
        form = QFormLayout(box)
        form.setContentsMargins(8, 6, 8, 8)
        self._values: dict[str, QLabel] = {}
        for label, key in self._ROWS:
            val = QLabel("—")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._values[key] = val
            form.addRow(label, val)
        # The reason the ratios are withheld, where the ratios are — not in a tooltip
        # only, because the defect was that the numbers were BELIEVED, and a caption
        # nobody hovers is a caption nobody reads.
        self.tautology_label = QLabel("")
        self.tautology_label.setWordWrap(True)
        self.tautology_label.setStyleSheet("color:#c9a227;")
        self.tautology_label.setVisible(False)
        form.addRow("", self.tautology_label)
        root.addWidget(box)

    # -- public API -------------------------------------------------------
    def set_score(self, score: ScoreResult | None) -> None:
        """Fill the readout from a :class:`ScoreResult`, or blank it with ``None``."""
        if score is None:
            self.clear()
            return
        tautological = self.is_tautological(score)
        for key, label in self._values.items():
            value = getattr(score, key, None)
            if tautological and key in self._RATIO_KEYS:
                label.setText(self._TAUTOLOGY_TEXT)
                label.setToolTip(self._TAUTOLOGY_TIP)
            else:
                label.setText(self._fmt(key, value))
                label.setToolTip("")
        n_auto = int(getattr(score, "n_auto_rejected", 0) or 0)
        auto_note = (
            f" ({n_auto} row(s) the detector rejected on its own are its flags, not "
            f"your verdict, and are not counted as spurious.)" if n_auto else "")
        self.tautology_label.setText(
            "Nothing rejected or hand-added yet — the truth is the detector's own "
            "output, so precision / recall / F1 would be 1.000 by construction."
            + auto_note
            if tautological else "")
        self.tautology_label.setVisible(tautological)

    def clear(self) -> None:
        for label in self._values.values():
            label.setText("—")
            label.setToolTip("")
        self.tautology_label.setText("")
        self.tautology_label.setVisible(False)

    def value_text(self, key: str) -> str:
        """The displayed text for a ScoreResult field (used by tests)."""
        lbl = self._values.get(key)
        return lbl.text() if lbl is not None else ""

    # -- internal ---------------------------------------------------------
    def _fmt(self, key: str, value: Any) -> str:
        if key in self._RATIO_KEYS:
            if value is None:
                return "n/a"
            return f"{float(value):.3f}"
        if value is None:
            return "—"
        return str(int(value))


# ---------------------------------------------------------------------------
# EventTable
# ---------------------------------------------------------------------------
# THE DERIVED HALF OF ``accepted``, THE REFUSAL SENTENCE AND THE REFUSAL LEDGER --
# byte-identical copies of the ones in event_gallery.py / event_inspector.py.
# MainWindow is the only gui/ module permitted to import other panels, so the three
# surfaces cannot share these by import; what they DO share is the rule's single
# source (:data:`~nanoscope.analysis.events.AUTOMATIC_REJECTION_FLAGS`, named inside,
# never restated) and -- in the wired application -- the LEDGER INSTANCE itself, which
# MainWindow hands to all three so an accept refused on any surface is disclosed by
# every surface, in the same sentence, until the refused set truly empties.

def _derived_accepted(flag: Any) -> bool:
    """``accepted`` as the SCHEMA derives it for a row carrying ``flag``.

    :func:`~nanoscope.analysis.characterize.characterize` owns the rule, and it is an AND,
    never an OR::

        accepted = ~(too_long | clipped | non_finite) & inherit_accepted

    The left half is a statement about the MEASUREMENT -- this row's numbers are not
    measurements of an event -- and the right half is the curator's own decision. A bulk
    "Accept all" is a curator's decision: it sets the right half, and has no standing over
    the left. This function reads the left half back off the row so the GUI can apply the
    same AND instead of writing ``True`` over both halves, which is how "Accept all" came
    to accept a row whose own flag says the detector could not measure it.

    It is read from ``flag`` and not re-derived from the samples because the flag column IS
    the record of that judgement: ``characterize`` writes the token and the ``False``
    together. :data:`~nanoscope.analysis.events.AUTOMATIC_REJECTION_FLAGS` is the ONE place
    that says which tokens are rejections, and naming it here rather than listing
    ``dwell>max_dwell_s`` / ``clipped_by_boundary`` / ``non_finite`` is the point: a fourth
    rejecting rule reaches this window with no second edit, while a derived token that
    rejects nothing (``zero_baseline`` -- a caveat about ``frac_depth``) and a curator's own
    token (``flagged``, ``molecule=apo``) never block an acceptance.

    Rejection is NOT guarded: ``accepted=False`` is always expressible, because a person may
    always reject an event the app was willing to keep.
    """
    return not any(t in AUTOMATIC_REJECTION_FLAGS for t in flag_tokens(flag))


def _refusal_note(kept: int, repaired: int, blocked: int, noun: str) -> str:
    """The sentence a bulk "accept" owes the user when the derived half refused rows.

    THREE OUTCOMES, THREE VERBS, because they are three different things and the user has
    to be able to tell them apart:

    * ``kept``      -- the row was already rejected and stays rejected. Nothing moved.
    * ``repaired``  -- the row was sitting ``accepted=True`` BESIDE a rejecting flag (the
      forbidden state, reachable from any table written before the derived-column fix) and
      this gesture put it back. Something moved, and "kept rejected" would be a lie about
      it: the user's accepted count went DOWN on a click that said "accept".
    * ``blocked``   -- the frame carries no ``flag`` column at all, so the derived half
      cannot be read and NOTHING was accepted. Fail closed: the guard turning itself off
      silently is the pre-fix broadcast wearing the fix's name.

    Returns ``""`` for a gesture with nothing to disclose, so the caller can test the
    string itself rather than remember a sentinel.
    """
    def _n(k: int) -> str:
        return f"{k} {noun}" + ("" if k == 1 else "s")

    if blocked > 0:
        return (f"{_n(blocked)} NOT accepted: this table has no 'flag' column, so the "
                "detector's own rejections cannot be read — refusing rather than guessing")
    parts: list[str] = []
    if kept > 0:
        parts.append(f"{_n(kept)} kept rejected")
    if repaired > 0:
        parts.append(f"{_n(repaired)} corrected back to rejected")
    if not parts:
        return ""
    return " and ".join(parts) + ": flagged by the detector as not measurable"


class _RefusalLedger:
    """The events this window has DECLINED to accept, on the frame it currently holds.

    A MEMBERSHIP RECORD THAT ONLY A NEW FRAME CLEARS. What it replaces was cleared at the
    top of every curation write, which made the disclosure a per-gesture artefact instead
    of a statement about the data. Measured in the wired app on the owner's -600 mV
    recording (317 events, ``max_dwell_s`` 26 ms, rows 0 / 4 / 260 unmeasurable):
    "Accept all" left ``314 of 317 accepted   ·   3 tiles kept rejected``, and then
    pressing R on ONE ORDINARY TILE left ``313 of 317 accepted`` with no clause at all --
    four rows unaccepted, three of them because the detector could not measure them, and
    nothing saying so. The IDENTICAL whole-frame reject made from the OTHER window left
    the note standing (that one arrives through ``refresh()``, which never cleared), so
    the app said two different things about one identical frame depending on which window
    the click landed in.

    THE RULE, in one place so it cannot be half-applied:

    * an ACCEPT re-decides the events IN ITS OWN SCOPE and no others. It is the only
      gesture that can create or retire a refusal, because a refusal is what an accept
      turned into.
    * a REJECT records nothing and forgets nothing. Rejection is never refused, so it has
      no refusal of its own to disclose, and the events an EARLIER accept was refused over
      are still refused. That is exactly what ``reflect_acceptance`` -- the same decision
      arriving from the events table -- has always done, and it is why the window's own
      gesture and the mirrored one now agree.
    * a NEW FRAME clears it, because the events it names no longer exist.

    THE COUNTS ARE NEVER STORED. :meth:`counts` re-reads the frame every time the line is
    rebuilt, so a row the events table has since accepted leaves the count on its own and
    takes nothing with it -- and comes BACK if it is rejected again, which a decrementing
    latch cannot do.

    IT REMEMBERS ``event_id``, NOT ROW POSITION. A row index survives an append and not a
    merge, and this record now outlives many more gestures than the per-gesture counter
    did. The window already re-finds the user's selection by ``event_id`` across a
    refilter for precisely that reason; this follows it rather than inventing a second
    policy.
    """

    __slots__ = ("_kept", "_repaired", "_blocked")

    def __init__(self) -> None:
        self._kept: set[int] = set()
        self._repaired: set[int] = set()
        # The fail-closed refusal is about the FRAME (no ``flag`` column at all) and not
        # about particular events, so it is a count of what one accept asked for and got
        # none of -- there are no event ids to remember when nothing vouches for any row.
        self._blocked: int = 0

    def clear(self) -> None:
        """Forget everything. A different frame is a different detection."""
        self._kept.clear()
        self._repaired.clear()
        self._blocked = 0

    def forget(self, event_ids) -> None:
        """This accept is about to re-decide ``event_ids``, so drop what an earlier one
        recorded about them. The frame-level refusal goes too: this gesture re-decides
        that as well, and on a frame with no ``flag`` column it reaches the same answer."""
        gone = {int(e) for e in event_ids}
        self._kept -= gone
        self._repaired -= gone
        self._blocked = 0

    def record(self, kept=(), repaired=()) -> None:
        """Two buckets, decided by where the row STARTED -- see :func:`_refusal_note`.
        WHICH events, never how many: the counts are recomputed on every rebuild of the
        line, and which bucket an event is in is history that cannot be re-derived."""
        self._kept.update(int(e) for e in kept)
        self._repaired.update(int(e) for e in repaired)

    def record_blocked(self, n: int) -> None:
        """``n`` rows asked for and none accepted: the frame carries no ``flag`` column."""
        self._blocked = int(n)

    def counts(self, df) -> tuple[int, int, int]:
        """``(kept, repaired, blocked)`` AS OF NOW -- recounted from ``df``, never stored.

        The frame-level refusal wins when it stands, because on a frame with no ``flag``
        column the other two questions cannot be asked at all.
        """
        if df is None or "flag" not in df.columns:
            return (0, 0, self._blocked)
        return (self.refused_in(df, self._kept), self.refused_in(df, self._repaired), 0)

    @staticmethod
    def refused_in(df, event_ids) -> int:
        """How many of ``event_ids`` are STILL refused -- rejected, AND rejected because
        the detector could not measure them. Both halves are re-read from the frame, so
        this is a question about NOW and not about the gesture that recorded them. An
        event that has left the frame entirely simply is not counted.

        VECTORISED, and counted PER DISTINCT FLAG STRING -- the same trick, for the same
        reason, as the gallery's accept-button promise. This runs on every rebuild of the
        status line, i.e. on every page turn, filter change, curation click and resize;
        asking :func:`_derived_accepted` once per row cost 833 ms on a 100 000-event
        detection with 90 % unmeasurable rows, which would have made an honest disclosure
        the slowest thing in the window.
        """
        if df is None or not len(event_ids) or "flag" not in df.columns:
            return 0
        if "event_id" not in df.columns:
            # A frame off the frozen schema has no identity to answer with. The window
            # cannot work at all without this column, but a REPAINT must not be the thing
            # that raises: the status line is rebuilt on every resize.
            return 0
        wanted = np.fromiter((int(e) for e in event_ids), dtype=np.int64,
                             count=len(event_ids))
        mask = np.isin(df["event_id"].to_numpy(), wanted)
        mask &= ~df["accepted"].to_numpy(dtype=bool)
        if not mask.any():
            return 0
        return int(sum(int(k) for value, k
                       in pd.Series(df["flag"].to_numpy()[mask])
                       .value_counts(dropna=False).items()
                       if not _derived_accepted(value)))

    @property
    def kept(self) -> set[int]:
        """The event ids in the ``kept rejected`` bucket. Read-only; the note counts them
        against the frame rather than trusting the size of this set."""
        return self._kept

    @property
    def repaired(self) -> set[int]:
        """The event ids in the ``corrected back to rejected`` bucket."""
        return self._repaired


# ---------------------------------------------------------------------------
# EventTable
# ---------------------------------------------------------------------------
class _FixedLinesNote(QLabel):
    """A word-wrapping note that OWNS a fixed number of lines whether it says anything or not.

    The twin of ``event_inspector._FixedLinesNote`` (not imported: this module imports
    no sibling gui panel, and the inspector pulls in the trace panes), for the same
    reason: a disclosure must not move the surface it discloses on. The note under the
    Preprocess tab's "lines" box was a bare word-wrapping QLabel, so the click that
    committed a refused list -- a press on a control further down the tab -- grew the
    note by two lines before its own release, moved that control +13 px, and lost the
    click (the release landed at local y = -3 of a 20 px checkbox). With the height
    reserved here, a repaint changes text and nothing else.

    Longer than the reserved lines at the current width, the text is shown from the
    head with "…" for what was withheld and the whole of it in the tooltip -- with one
    priority the inspector's twin does not need: the LAST segment after
    :attr:`MARKER` is a standing refusal, and it is kept whole while the description
    ahead of it is shortened; only when the refusal alone does not fit is it shortened
    too. A refusal pushed off the note by the list it refused would be the on-screen
    silence the note exists to end.

    :meth:`text` returns the FULL text; what is painted is what fitted.
    """

    #: What precedes the segment that must stay painted (see :meth:`_fit`).
    MARKER = "\n⚠ "

    def __init__(self, lines: int, parent: QWidget | None = None, *,
                 selectable: bool = True) -> None:
        super().__init__("", parent)
        self._full = ""
        flags = Qt.TextSelectableByMouse if selectable else Qt.NoTextInteraction
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setTextInteractionFlags(flags)
        # The width is the form's, and a word-wrapping label must never set the dock's
        # minimum width; the height is decided here, once.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # The measuring twin: the same font, wrap and interaction flags, so what it says
        # fits at a width is what this label paints at that width (a selectable QLabel
        # lays its text out through a text document, not the bare font metrics).
        self._probe = QLabel(self)
        self._probe.setWordWrap(True)
        self._probe.setTextFormat(Qt.PlainText)
        self._probe.setTextInteractionFlags(flags)
        self._probe.hide()
        self._probe.setText("\n".join(["x"] * max(int(lines), 1)))
        self.setFixedHeight(self._probe.sizeHint().height() + 4)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = str(text)
        self._fit()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # The note is restyled bold for a warning; bold text wraps differently, so a
        # font or style change re-fits with the font the label now paints in.
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._fit()

    def _fits(self, text: str) -> bool:
        self._probe.setFont(self.font())
        self._probe.setText(text)
        return self._probe.heightForWidth(self.width()) <= self.height()

    def _shortened(self, text: str, frame) -> str | None:
        """The longest head of ``text`` (whole words, then " …") for which
        ``frame(head)`` fits, or ``None`` when not even "…" does. Height is monotonic
        in the number of words kept, so this bisects."""
        words = text.split()
        best, lo, hi = None, 0, len(words) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = (" ".join(words[:mid]) + " …") if mid else "…"
            if self._fits(frame(cand)):
                best, lo = cand, mid + 1
            else:
                hi = mid - 1
        return best

    def _fit(self) -> None:
        full = self._full
        shown, elided = full, False
        if full and not self._fits(full):
            elided = True
            head, marker, refusal = full.rpartition(self.MARKER)
            if marker:
                cut = self._shortened(head, lambda h: f"{h}{marker}{refusal}")
                if cut is not None:
                    shown = f"{cut}{marker}{refusal}"
                else:
                    cut = self._shortened(refusal, lambda r: f"…{marker}{r}")
                    shown = f"…{marker}{cut}" if cut is not None else "…"
            else:
                cut = self._shortened(full, lambda t: t)
                shown = cut if cut is not None else "…"
        if shown != QLabel.text(self):
            QLabel.setText(self, shown)
        self.setToolTip(self._full if elided else "")


class _ElidingCaption(QLabel):
    """A one-line caption that elides to the width it is GIVEN.

    THE DOCK-WIDTH LESSON, AGAIN. A plain QLabel reports its FULL text width as
    ``minimumSizeHint()``, and that number propagates into
    ``QDockWidget::minimumSize()`` — so a caption naming a long file path would become
    a floor on how narrow the Inspector dock can be dragged, exactly as this panel's
    title and the Measure tool's button row already did. Here the full string stays
    ``text()`` (callers and tests read it unchanged) and only the painted copy is
    elided; the horizontal minimum is zero.

    WHERE THE CUT FALLS IS THE CALLER'S CHOICE. ``Qt.ElideMiddle`` is right for a file
    path — head and tail identify it and the middle is filler — and wrong for a sentence,
    whose informative half is the tail. Measured in the shipped Inspector dock at
    1512x900: this label is **422 px** wide, the events table's six-row caption needs
    **453 px**, and ElideMiddle painted
    ``'6 of 115 rows selected  ·  A accepts …ts all 6  ·  right-click for the same'`` —
    the run it swallowed was ``'/ R rejec'``, the only mention of the R key on that
    surface, and what survived reads as garbage. Captions that are sentences pass
    ``elide=Qt.ElideRight``; paths keep the default.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None, *,
                 elide=Qt.ElideMiddle) -> None:
        super().__init__(parent)
        self._full = ""
        self._elide = elide
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = text
        self.setToolTip(text)
        self._apply_elide()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def minimumSizeHint(self):  # noqa: N802 (Qt override)
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        from PySide6.QtGui import QFontMetrics

        width = max(self.width() - 2, 0)
        painted = (QFontMetrics(self.font()).elidedText(self._full, self._elide, width)
                   if width > 0 else self._full)
        QLabel.setText(self, painted)


class _NoticeChip(QPushButton):
    """A small push button whose label elides instead of flooring its row's width.

    A plain button, NOT a QToolButton InstantPopup: its menu opens through
    ``EventTable._exec_menu``, the one seam tests can answer (an InstantPopup menu is a
    C++ ``QMenu::exec`` the suite's modal guard cannot see).
    """

    _MIN_WIDTH = 28

    def __init__(self, slot_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._slot_text = str(slot_text)
        # The slot is kept while hidden, so the row never reflows when the chip appears,
        # goes or changes its text (a longer text elides). Its host gives it the same
        # stretch as its neighbour (the table selector), capped at the slot text's width:
        # a narrow row then takes width from the empty slot as much as from the selector.
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(self._MIN_WIDTH)
        self._fit_slot()

    def _fit_slot(self) -> None:
        width = max(self._MIN_WIDTH, self._size_for(self._slot_text).width())
        if self.maximumWidth() != width:
            self.setMaximumWidth(width)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._fit_slot()

    def _size_for(self, text: str) -> QSize:
        """QPushButton.sizeHint's own computation, for ``text`` instead of text()."""
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = text
        size = QSize(self.fontMetrics().horizontalAdvance(text),
                     max(self.fontMetrics().height(), 14))
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_PushButton, opt, size, self)

    def _chrome_width(self) -> int:
        return max(0, self._size_for("").width())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self._size_for(self._slot_text)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        full = self.sizeHint()
        return QSize(min(full.width(), self._MIN_WIDTH), full.height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        available = self.width() - self._chrome_width()
        if self.text() and available < self.fontMetrics().horizontalAdvance(self.text()):
            opt.text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight,
                                                     max(0, available))
        QStylePainter(self).drawControl(QStyle.ControlElement.CE_PushButton, opt)


class _BatchCuration(NamedTuple):
    """What ONE batch curation gesture wrote, and what it refused.

    The two refusal buckets are the per-row writer's own (:meth:`_EventTableModel._curate`)
    counted over a scope instead of over a row, so the ledger — and therefore the sentence
    on screen — says the same thing whether the user pressed R once or over 300 selected
    rows. ``blocked`` is the frame-level refusal (no ``flag`` column, so nothing vouches
    for any row); it is a COUNT because there are no event ids to remember.
    """

    changed: np.ndarray      # SOURCE rows whose ``accepted`` cell actually moved
    kept: tuple              # event_ids: already rejected, and the accept was refused
    repaired: tuple          # event_ids: forbidden accepted=True, corrected back to False
    blocked: int             # rows asked for on a frame with no ``flag`` column
    review_changed: int      # source/review cells changed by this human ruling
    conflict_changed: int    # reconciliation conflict cells resolved by this ruling


class _EventTableModel(QAbstractTableModel):
    """A QAbstractTableModel view over an EventSet's DataFrame.

    The whole point of a model (vs. QTableWidgetItems) is that curating 5000
    events never materialises 5000 widgets. The DataFrame is the source of
    truth; the 'accepted' column is editable via a checkbox and edits write
    straight back to the frame.
    """

    acceptanceChanged = Signal(int, bool)  # event_id, accepted — WHAT WAS WRITTEN
    flagChanged = Signal(int, str)         # event_id, the row's FULL new flag set
    #: An accept the derived half refused: event_id, and which bucket the refusal is
    #: (``"kept"`` — the row was already rejected and stays so; ``"repaired"`` — it sat
    #: ``accepted=True`` beside a rejecting flag and was corrected BACK to rejected;
    #: ``"blocked"`` — the frame has no ``flag`` column, so nothing vouches for any row).
    #: Emitted so the owning :class:`EventTable` can record it on the refusal ledger and
    #: say so — a checkbox that snaps back with no sentence anywhere is a silent no-op.
    acceptanceRefused = Signal(int, str)

    def __init__(self, df: pd.DataFrame | None = None, parent=None) -> None:
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame(
            {c: pd.Series(dtype=d) for c, (d, _, _) in SCHEMA.items()}
        )
        self._columns = list(self._df.columns)
        #: Columns whose values were measured by an estimator that is no longer the
        #: configured one. Empty is the normal case. Held on the MODEL, not painted
        #: by a delegate over 5000 rows: the badge is a role the model answers, so
        #: marking a table stale costs one repaint and no widgets.
        self._stale_columns: tuple[str, ...] = ()
        self._stale_tip = ""
        #: ``instrument_seconds - recording_seconds`` for the recording this table
        #: belongs to; None when unknown (no sample rate, or nothing pushed yet). The
        #: t-columns are on the RECORDING clock and the trace x-axis is not, and the
        #: header is where that stops being a trap. See :mod:`nanoscope.core.clock`.
        self._clock_offset_s: float | None = None

    def set_df(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self._columns = list(df.columns)
        self.endResetModel()

    def set_clock_offset(self, offset_s: float | None) -> None:
        """The recording's ``t₀`` (+ layer nudge), so the time headers can name their clock."""
        if offset_s == self._clock_offset_s:
            return
        self._clock_offset_s = offset_s
        self.headerDataChanged.emit(Qt.Horizontal, 0, max(len(self._columns) - 1, 0))

    def set_stale_columns(self, names: Iterable[str], tooltip: str = "") -> None:
        """Badge ``names`` as measured by a superseded estimator (or clear with ``()``).

        §4.2.5: an old project must stay readable, plottable and exportable, marked
        stale, **with the affected columns badged**. The badge is the honest half of
        "never silently re-measure": the numbers on screen are the numbers that were
        stored, and the table says which of them are no longer measured the way a
        fresh run would measure them.
        """
        cols = tuple(str(n) for n in names)
        if "depth_pA" in cols and "amplitude_pA" in self._df:
            cols = tuple(dict.fromkeys((*cols, "amplitude_pA", "amplitude_level_pA",
                                        "amplitude_baseline_pA")))
        if cols == self._stale_columns and tooltip == self._stale_tip:
            return
        self._stale_columns = cols
        self._stale_tip = str(tooltip)
        self.beginResetModel()   # header + every cell in those columns repaints
        self.endResetModel()

    @property
    def stale_columns(self) -> tuple[str, ...]:
        return self._stale_columns

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _accept_col(self) -> int:
        return self._columns.index("accepted") if "accepted" in self._columns else -1

    # -- Qt model protocol ------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return section if role == Qt.DisplayRole else None
        if not (0 <= section < len(self._columns)):
            return None
        name = self._columns[section]
        stale = name in self._stale_columns
        if role == Qt.DisplayRole:
            # A visible marker in the header, not colour alone: colour is the first
            # thing lost to a colour-blind reader, a screenshot, or a printed figure,
            # and this badge is the only thing distinguishing a number that is right
            # from one that is 15x wrong.
            label = "median depth (depth_pA)" if name == "depth_pA" else name
            text = f"{label} ⚠" if stale else label
            # ...and the same rule for the CLOCK. `t_start_s` is seconds from the start
            # of the recording; the trace x-axis beside it is seconds on the instrument
            # clock, and both used to be spelled "s". The header is the only place a
            # reader of this column ever looks.
            if name in _CLOCK_COLUMNS and not clock.clocks_agree(self._clock_offset_s) \
                    and self._clock_offset_s is not None:
                text = f"{text} (t₀=0)"
            return text
        if role == Qt.ToolTipRole:
            tips = []
            if stale:
                tips.append(self._stale_tip or _STALE_COLUMN_TIP)
            if name in _CLOCK_COLUMNS:
                tips.append(clock.field_note(clock.RECORDING, self._clock_offset_s))
            elif name in _COLUMN_TIPS:
                tips.append(_COLUMN_TIPS[name])
            return "\n\n".join(t for t in tips if t) or None
        if role == Qt.ForegroundRole and stale:
            return QBrush(QColor("#ffd479"))
        return None

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == self._accept_col():
            base |= Qt.ItemIsUserCheckable
        return base

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        colname = self._columns[col]
        value = self._df.iat[row, col]

        # THE SORT ROLE COMES FIRST, and covers EVERY column including the checkbox one.
        # A sort must order the numbers, not their captions, so this serves the raw cell
        # value; :data:`_SORT_ROLE` says what the display string does to an ordering.
        # NA sorts as an invalid value (together, at one end) rather than comparing
        # against a number.
        if role == _SORT_ROLE:
            if isinstance(value, (str, bytes)):
                return str(value)
            if value is None or pd.isna(value):
                return None
            if isinstance(value, (bool, np.bool_)):
                # ``accepted`` too: clicking that header groups the curated rows, which
                # is the only ordering it can honestly have.
                return float(bool(value))
            if isinstance(value, (int, np.integer, float, np.floating)):
                return float(value)
            return str(value)

        if col == self._accept_col():
            if role == Qt.CheckStateRole:
                return Qt.Checked if bool(value) else Qt.Unchecked
            return None  # no text next to the checkbox

        if role in (Qt.DisplayRole, Qt.EditRole):
            if isinstance(value, (float, np.floating)):
                if colname in ("dwell_s", "rise_s", "fall_s", "ecd_pA_s"):
                    return f"{value:.6g}"
                return f"{value:.4g}"
            return str(value)
        if colname in self._stale_columns:
            # Every CELL of a badged column carries the reason too. The header alone
            # scrolls out of sight sideways, and the number the owner is reading off
            # a row is exactly where the warning has to be reachable.
            if role == Qt.ToolTipRole:
                return self._stale_tip or _STALE_COLUMN_TIP
            if role == Qt.ForegroundRole:
                return QBrush(QColor("#ffd479"))
        if colname.startswith("amplitude_") and "amplitude_status" in self._columns:
            status = str(self._df.iat[row, self._columns.index("amplitude_status")])
            if status == "stale":
                if role == Qt.ToolTipRole:
                    return _COLUMN_TIPS["amplitude_status"]
                if role == Qt.ForegroundRole:
                    return QBrush(QColor("#ffd479"))
        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        if not index.isValid() or index.column() != self._accept_col():
            return False
        if role != Qt.CheckStateRole:
            return False
        return self._curate(index.row(), Qt.CheckState(value) == Qt.Checked)

    def _curate(self, row: int, requested: bool) -> bool:
        """THE one write of ``accepted`` this model has — the checkbox (:meth:`setData`),
        the a/r keys and :meth:`set_accepted` all land here. Returns True when a value
        was written.

        What lands in the cell is ``derived AND requested``, not ``requested`` — the same
        AND :func:`~nanoscope.analysis.characterize.characterize` applies and both bulk
        surfaces already enforce (see :func:`_derived_accepted`). This model was the LAST
        unguarded writer: one tick on the table's checkbox put ``accepted=True`` beside
        ``clipped_by_boundary`` — the forbidden state — past the very rule "Accept all"
        had just applied over the whole frame, and the gallery tile beside it stayed
        painted, and captioned, rejected.

        A refused accept writes nothing and ANNOUNCES nothing on ``acceptanceChanged``
        (a signal with no change behind it reaches the dirty flag and the undo recorder);
        it emits :attr:`acceptanceRefused` instead, so the owner discloses it — the
        checkbox snapping back with no sentence anywhere would be a silent no-op. The
        repair case (the forbidden state, reachable from any table written before the
        guard) DOES move: the cell goes back to False, and what is announced is what was
        written. Rejection is never blocked — ``requested=False`` is the curator's own
        half going to False, which ANDs to False whatever the row says.
        """
        col = self._accept_col()
        idx = self.index(row, col)
        target = bool(requested)
        if target:
            if "flag" not in self._columns:
                # FAIL CLOSED: with no ``flag`` column the derived half cannot be read,
                # and answering it True would be the broadcast the guard exists to close.
                self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])   # snap the box back
                self.acceptanceRefused.emit(self._event_id(row), "blocked")
                return False
            target = _derived_accepted(self._df.iat[row, self._columns.index("flag")])
        before = bool(self._df.iat[row, col])
        if bool(requested) and not target:
            # Which bucket depends on where the row STARTED — see :func:`_refusal_note`.
            if before:
                conflict_changed = clear_curation_conflicts(self._df, (row,))
                self._df.iat[row, col] = False
                self.dataChanged.emit(
                    self.index(row, 0) if conflict_changed else idx,
                    self.index(row, max(self.columnCount() - 1, 0))
                    if conflict_changed else idx,
                    [Qt.DisplayRole, Qt.CheckStateRole]
                    if conflict_changed else [Qt.CheckStateRole],
                )
                self.acceptanceChanged.emit(self._event_id(row), False)
                self.acceptanceRefused.emit(self._event_id(row), "repaired")
                return True
            self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])       # visible snap-back
            self.acceptanceRefused.emit(self._event_id(row), "kept")
            return False
        # Pushed through unconditionally — even when the frame already holds the value —
        # so a mirror write (set_accepted_by_id) always repaints the view.
        conflict_changed = clear_curation_conflicts(self._df, (row,))
        self._df.iat[row, col] = target
        self.dataChanged.emit(
            self.index(row, 0) if conflict_changed else idx,
            self.index(row, max(self.columnCount() - 1, 0))
            if conflict_changed else idx,
            [Qt.DisplayRole, Qt.CheckStateRole]
            if conflict_changed else [Qt.CheckStateRole],
        )
        self.acceptanceChanged.emit(self._event_id(row), target)
        return True

    # -- helpers ----------------------------------------------------------
    def _event_id(self, row: int) -> int:
        if "event_id" in self._columns:
            return int(self._df.iat[row, self._columns.index("event_id")])
        return int(row)

    def set_accepted(self, row: int, accepted: bool) -> None:
        """Curate one row programmatically (the a/r keys, and the mirror door
        ``EventTable.set_accepted_by_id``) — through :meth:`_curate`, i.e. under the SAME
        ``derived AND requested`` the checkbox applies. The mirror only ever carries what
        another surface actually WROTE (the windows announce written values, never
        requests), and a written True implies the derived half already said yes — so the
        guard re-derives to the same answer there and costs the mirror nothing."""
        col = self._accept_col()
        if col < 0 or not (0 <= row < len(self._df)):
            return
        self._curate(row, bool(accepted))

    def set_accepted_rows(self, rows, requested: bool) -> "_BatchCuration":
        """:meth:`_curate` over MANY rows in ONE pass — the same rule, one write, one repaint.

        THE RULE IS NOT RE-STATED HERE. What lands in every cell is still
        ``derived AND requested`` (:func:`_derived_accepted` off the row's own ``flag``),
        and the fail-closed answer on a frame with no ``flag`` column is still "accept
        nothing" — so a batch cannot accept a row the per-row writer would refuse, and a
        fifth token added to ``AUTOMATIC_REJECTION_FLAGS`` reaches this path with no
        second edit. This is the SAME writer widened, not a second one: the batch is the
        gallery's ``_set_rows_accepted`` idiom (vectorised derive, one ``iloc`` write, one
        announcement), moved to the surface the owner actually curates on.

        WHY NOT A LOOP OVER :meth:`_curate`. Measured in the wired app on the owner's
        -600 mV apoferritin recording (317 events): announcing one ``acceptanceChanged``
        per row for 300 rows cost **3.72 s** (12.4 ms/row) — the summary, the span
        overlays and the detector score are each rebuilt from the whole frame by every
        one of those signals, and the history records 300 undo steps for one gesture.
        The same 300 rows through here are one write, one ``dataChanged`` and one
        announcement by the caller.

        Emits ONE ``dataChanged`` spanning the changed rows (the proxy already collapses a
        multi-row span into a single whole-column repaint) and NO per-row
        ``acceptanceChanged`` / ``acceptanceRefused``: the caller owns the announcement,
        because only it knows whether the gesture moved one row or three hundred.

        Returns a :class:`_BatchCuration` naming what was WRITTEN and what was refused —
        the two refusal buckets are decided by where each row STARTED, exactly as in
        :meth:`_curate` (``kept`` = already rejected and left so; ``repaired`` = the
        forbidden ``accepted=True`` beside a rejecting flag, corrected back to False).
        """
        col = self._accept_col()
        rows_arr = np.unique(np.asarray(list(rows), dtype=np.int64)) if len(rows) else \
            np.zeros(0, dtype=np.int64)
        rows_arr = rows_arr[(rows_arr >= 0) & (rows_arr < len(self._df))]
        if col < 0 or rows_arr.size == 0:
            return _BatchCuration(np.zeros(0, dtype=np.int64), (), (), 0, 0, 0)

        ids = self._df.iloc[rows_arr, self._columns.index("event_id")].to_numpy(
            dtype=np.int64) if "event_id" in self._columns else rows_arr
        before = self._df.iloc[rows_arr, col].to_numpy(dtype=bool)
        if requested:
            if "flag" not in self._columns:
                # FAIL CLOSED, the same direction :meth:`_curate` fails: with no ``flag``
                # column nothing vouches for any row, so nothing is written and the whole
                # ask is one frame-level refusal.  It is not a resolved human ruling, so
                # it must not stamp ``reviewed`` or clear a reconciliation conflict.
                return _BatchCuration(np.zeros(0, dtype=np.int64), (), (),
                                      int(rows_arr.size), 0, 0)
        # The gesture's SCOPE is the review scope, even when the requested boolean is
        # already present.  "Accept all" on a fresh detector table is a human ruling on
        # every candidate, not a no-op merely because measurable rows start accepted.
        # Valid rows refused by automatic rejection flags still reach this stamp: the
        # curator did inspect those candidates even though the derived guard kept them
        # negative.
        conflict_changed = clear_curation_conflicts(self._df, rows_arr)
        review_changed = mark_reviewed(self._df, rows=rows_arr)
        if requested:
            target = self._derived_accepted_rows(rows_arr)
            refused = ~target
            kept = tuple(int(e) for e in ids[refused & ~before])
            repaired = tuple(int(e) for e in ids[refused & before])
        else:
            target = np.zeros(rows_arr.size, dtype=bool)
            kept = repaired = ()

        changed_mask = before != target
        changed = rows_arr[changed_mask]
        if changed.size:
            # In place on the SHARED frame — the same cells the ``iat`` writes hit.
            self._df.iloc[changed, col] = target[changed_mask]
            top, bottom = int(changed.min()), int(changed.max())
            self.dataChanged.emit(self.index(top, col), self.index(bottom, col),
                                  [Qt.CheckStateRole])
        elif review_changed or conflict_changed:
            # ``review`` changed while ``accepted`` did not. Repaint the gesture scope;
            # the caller emits one trailing curation signal so this write is dirty and
            # undoable exactly like a boolean-changing batch.
            top, bottom = int(rows_arr.min()), int(rows_arr.max())
            self.dataChanged.emit(self.index(top, 0),
                                  self.index(bottom, max(self.columnCount() - 1, 0)),
                                  [Qt.DisplayRole])
        elif kept or repaired:
            # Nothing moved, but the checkboxes the user just ticked must snap back.
            self.dataChanged.emit(self.index(int(rows_arr.min()), col),
                                  self.index(int(rows_arr.max()), col),
                                  [Qt.CheckStateRole])
        return _BatchCuration(
            changed, kept, repaired, 0, int(review_changed), int(conflict_changed))

    def _derived_accepted_rows(self, rows: np.ndarray) -> np.ndarray:
        """:func:`_derived_accepted` over many rows in ONE pass.

        The RULE stays the single per-value function — evaluated once per UNIQUE flag
        string and broadcast back over the rows, so a fifth rejecting token added to
        ``AUTOMATIC_REJECTION_FLAGS`` reaches this path with no second edit, exactly as it
        reaches :meth:`_curate`. Unique flag values are a handful while rows are 1e5,
        which is the whole speed-up. Same trick, same reason, as
        ``EventGalleryWindow._derived_accepted_rows`` and ``_RefusalLedger.refused_in``.
        """
        flags = self._df.iloc[:, self._columns.index("flag")].to_numpy()[rows]
        codes, uniques = pd.factorize(flags, use_na_sentinel=False)
        per_unique = np.fromiter((_derived_accepted(u) for u in uniques),
                                 dtype=bool, count=len(uniques))
        return per_unique[codes]

    def set_flag(self, row: int, flag: str) -> None:
        """ADD ``flag`` to this row's reasons. It is a set, not a slot.

        This wrote the value straight over the cell. The flag column is a ';'-joined SET
        of reasons (``analysis/events.py``), and the one token the GUI writes is the
        curator's own — so marking an event that the detector had already flagged
        ``clipped_by_boundary`` would have erased that reason while leaving the row
        rejected: C21's "rejected with NO REASON", invisible to the very filter meant to
        find it, at the one call site C21's fix did not reach. It survived review because
        it was unreachable (see :meth:`EventTable.keyPressEvent`), which is not a defence
        of the writer — it is why the writer had to be fixed at the same time as the keys.
        """
        if "flag" not in self._columns or not (0 <= row < len(self._df)):
            return
        col = self._columns.index("flag")
        merged = combine_flags(self._df.iat[row, col], flag)
        if merged == self._df.iat[row, col]:
            return                       # already carries it; do not repaint
        self._df.iat[row, col] = merged
        idx = self.index(row, col)
        self.dataChanged.emit(idx, idx, [Qt.DisplayRole])
        self.flagChanged.emit(self._event_id(row), merged)

    def refresh_row(self, row: int) -> None:
        """Repaint every cell of ``row`` — the frame was mutated in place elsewhere (an Event
        Inspector mask edit rewrote this event's bounds + stats). Emits ``dataChanged`` across the
        whole row so the view (and the sort proxy) picks up the new values."""
        if not (0 <= row < len(self._df)):
            return
        left = self.index(row, 0)
        right = self.index(row, max(self.columnCount() - 1, 0))
        self.dataChanged.emit(left, right, [Qt.DisplayRole, Qt.CheckStateRole])

    def refresh_all(self) -> None:
        """Reset the model so the whole view repaints from the current frame — for a BULK change
        (Accept all / Reject all rewrote the entire ``accepted`` column), where a per-cell signal
        storm would be wasteful."""
        self.beginResetModel()
        self._columns = list(self._df.columns)
        self.endResetModel()


class _VectorSortProxy(QAbstractProxyModel):
    """Sorting proxy over :class:`_EventTableModel` whose row order is computed in ONE
    vectorised pass over the DataFrame, not comparison-by-comparison.

    WHY NOT ``QSortFilterProxyModel``: its sort calls the source model's Python
    ``data()`` twice per comparison, so a sort is O(n log n) *Python round-trips* —
    measured on a 100,000-event table, ONE header click blocked the GUI thread 48.7 s
    (7.8 s on every re-sort), while a numpy argsort of the same column takes ~7 ms.
    This proxy asks the frame for the whole column once and lets numpy order it:
    O(n log n) *vectorised* comparisons plus O(n) Python to remap, which is ~60 ms at
    100k rows. Same view API surface the table already used: ``mapToSource`` /
    ``mapFromSource`` / ``rowCount`` / ``setData`` all behave as the
    ``QSortFilterProxyModel`` they replace.

    THE ORDERING CONTRACT IS UNCHANGED ON THE SHIPPED SURFACE — it is
    :data:`_SORT_ROLE`'s ordering, stated once at :meth:`_sort_key`: numeric (and
    bool) columns order as numbers, every other column as its ``str()`` caption.
    Placement matches what the replaced ``QSortFilterProxyModel`` MEASURABLY did
    when driven the way the app drives it — a shown ``QTableView`` whose mapping is
    materialised, header clicks via ``sortByColumn`` (differentially probed against
    the pristine widget, not inferred from Qt's docs): BOTH directions are stable
    sorts over the standing view order (so ties chain through sequential clicks and
    a repeated click changes nothing), and NA rows travel together in standing
    order — LAST on an ascending click, FIRST on a descending one. (Qt's proxy also
    had a second ordering buried in its lazy ``create_mapping`` path — descending
    ties reversed — but it is reachable only on a proxy that has never been queried,
    which a table view's proxy never is by the time a user can click or re-feed:
    every shipped gesture, re-feeds under a remembered sort included, measured
    identical. This proxy has ONE rule, the measured one, for every path — so a
    headless harness that sorts before ever querying sees the click ordering here
    where Qt would have served its lazy one.) The tests that pinned the FE-01
    numeric-truth fix (``test_s4_fe_table_numeric_sort.py``) run against this proxy
    unchanged.

    Sorting is DYNAMIC the way the old proxy's ``dynamicSortFilter`` was: an edit to a
    cell of the column currently sorted on re-sorts the view (that is how a toggled
    ``accepted`` checkbox moves its row while the table is grouped by acceptance);
    edits to other columns move nothing.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        #: proxy row -> source row, and its inverse. Identity until someone sorts.
        self._order = np.arange(0, dtype=np.intp)
        self._inverse = np.arange(0, dtype=np.intp)
        self._sort_column = -1                      # -1 = unsorted (source order)
        self._sort_qorder = Qt.AscendingOrder

    # -- wiring ------------------------------------------------------------
    def setSourceModel(self, model) -> None:  # noqa: N802 (Qt override)
        old = self.sourceModel()
        if old is not None:
            old.modelAboutToBeReset.disconnect(self._on_source_about_to_reset)
            old.modelReset.disconnect(self._on_source_reset)
            old.dataChanged.disconnect(self._on_source_data_changed)
            old.headerDataChanged.disconnect(self.headerDataChanged)
        super().setSourceModel(model)
        if model is not None:
            model.modelAboutToBeReset.connect(self._on_source_about_to_reset)
            model.modelReset.connect(self._on_source_reset)
            model.dataChanged.connect(self._on_source_data_changed)
            # Sections pass through unmapped: columns are never permuted, and the source
            # only announces HORIZONTAL header changes (the clock-suffix repaint).
            model.headerDataChanged.connect(self.headerDataChanged)
            self._set_order(self._compute_order())

    # -- Qt model shape ----------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else int(self._order.size)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        src = self.sourceModel()
        return 0 if (src is None or parent.isValid()) else src.columnCount()

    def index(self, row: int, column: int,
              parent: QModelIndex = QModelIndex()) -> QModelIndex:  # noqa: B008
        if parent.isValid() or not (0 <= row < self._order.size) \
                or not (0 <= column < self.columnCount()):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, index: QModelIndex = QModelIndex()) -> QModelIndex:  # noqa: B008
        return QModelIndex()

    def mapToSource(self, proxy_index: QModelIndex) -> QModelIndex:  # noqa: N802
        src = self.sourceModel()
        if src is None or not proxy_index.isValid() \
                or not (0 <= proxy_index.row() < self._order.size):
            return QModelIndex()
        return src.index(int(self._order[proxy_index.row()]), proxy_index.column())

    def mapFromSource(self, source_index: QModelIndex) -> QModelIndex:  # noqa: N802
        if not source_index.isValid() \
                or not (0 <= source_index.row() < self._inverse.size):
            return QModelIndex()
        return self.index(int(self._inverse[source_index.row()]), source_index.column())

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):  # noqa: N802
        src = self.sourceModel()
        if src is None:
            return None
        if orientation == Qt.Horizontal or not (0 <= section < self._order.size):
            return src.headerData(section, orientation, role)
        # Vertical sections carry the SOURCE row number, as the old proxy's did — the
        # label follows the row through a sort instead of always counting 0..n.
        return src.headerData(int(self._order[section]), orientation, role)

    # -- the sort itself ---------------------------------------------------
    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        """The header click lands here (``QTableView.sortByColumn`` calls it)."""
        self._sort_column = int(column)
        self._sort_qorder = order
        self._resort(base=self._order)

    def _source_frame(self) -> pd.DataFrame | None:
        src = self.sourceModel()
        return getattr(src, "df", None) if src is not None else None

    @staticmethod
    def _sort_key(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """One column's whole sort key: (float key, NA mask) — the vectorised statement
        of what ``_EventTableModel.data(_SORT_ROLE)`` serves per cell. Numbers and bools
        order as numbers; everything else as its ``str()`` caption (ranked, so equal
        captions stay tied and the stable sort keeps their source order)."""
        na = pd.isna(series).to_numpy(dtype=bool)
        if pd.api.types.is_numeric_dtype(series):    # bool counts as numeric
            key = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=np.float64, copy=True)
        else:
            texts = np.asarray(
                ["" if m else str(v) for v, m in zip(series.tolist(), na.tolist())],
                dtype=object)
            # Equal strings share a rank; unique captions are FEW next to 1e5 rows.
            _uniq, inv = np.unique(texts.astype(str), return_inverse=True)
            key = inv.astype(np.float64)
        key[na] = 0.0   # neutral — the NA half of the lexsort key decides their end
        return key, na

    def _compute_order(self, base: np.ndarray | None = None) -> np.ndarray:
        """The new proxy->source permutation. ``base`` is the order TIES keep: the
        current view order for a live sort (so clicking dwell after depth leaves equal
        dwells in depth order — the sequential multi-level sort the old proxy's
        sort-the-existing-mapping behaviour gave), the source order after a reset
        (where the old proxy rebuilt its mapping first, exactly as here)."""
        src = self.sourceModel()
        n = int(src.rowCount()) if src is not None else 0
        df = self._source_frame()
        col = self._sort_column
        if base is None or base.size != n:
            base = np.arange(n, dtype=np.intp)
        if n == 0 or df is None or len(df) != n or not (0 <= col < len(df.columns)):
            # No sort column = source order — ``sort(-1)`` RESTORES the source order,
            # as the proxy this replaces did.
            return np.arange(n, dtype=np.intp)
        key, na = self._sort_key(df.iloc[:, col])
        key, na = key[base], na[base]
        # lexsort is stable and sorts by the LAST key first. Both directions are a
        # STABLE sort over ``base`` — ties keep the standing order, which is what
        # makes sequential clicks chain and repeated clicks idempotent. NA rows
        # travel TOGETHER, in standing order: LAST ascending, FIRST descending.
        # This is byte-for-byte what the replaced ``QSortFilterProxyModel`` did on
        # the shipped surface — measured through a real ``QTableView.sortByColumn``
        # against the pristine widget, NOT inferred from Qt's docs. (Qt's proxy has
        # a second, lazier path — ``create_mapping`` on a never-queried proxy —
        # that reverses descending ties; a live table view always materialises the
        # mapping before the first click, so that path never reaches the user.)
        if self._sort_qorder == Qt.AscendingOrder:
            perm = np.lexsort((key, na))     # non-NA by value, NA rows last
        else:
            # ``key`` is 0.0 on NA rows (see ``_sort_key``), so ``-key`` stays tied
            # there and the stable sort keeps NA rows in standing order too.
            perm = np.lexsort((-key, ~na))   # NA rows first, then key descending
        return base[perm].astype(np.intp, copy=False)

    def _set_order(self, order: np.ndarray) -> None:
        self._order = order
        inv = np.empty(order.size, dtype=np.intp)
        inv[order] = np.arange(order.size, dtype=np.intp)
        self._inverse = inv

    def _resort(self, base: np.ndarray | None = None) -> None:
        """Re-order live, announced as a layout change so the view's selection and
        current row FOLLOW their events (persistent indexes are remapped)."""
        new_order = self._compute_order(base)
        if np.array_equal(new_order, self._order):
            return                                   # nothing would move
        self.layoutAboutToBeChanged.emit()
        stale = self.persistentIndexList()
        old_order = self._order
        self._set_order(new_order)
        fresh = []
        for idx in stale:
            r = idx.row()
            if 0 <= r < old_order.size:
                fresh.append(self.index(int(self._inverse[int(old_order[r])]),
                                        idx.column()))
            else:
                fresh.append(QModelIndex())
        self.changePersistentIndexList(stale, fresh)
        self.layoutChanged.emit()

    # -- source-change plumbing --------------------------------------------
    def _on_source_about_to_reset(self) -> None:
        self.beginResetModel()

    def _on_source_reset(self) -> None:
        # The frame changed wholesale (set_df / refresh_all). Re-apply the remembered
        # sort over the NEW rows — the old proxy re-sorted on reset too, which is how
        # a bulk curation pass keeps the user's chosen order on screen.
        self._set_order(self._compute_order())
        self.endResetModel()

    def _on_source_data_changed(self, top_left: QModelIndex,
                                bottom_right: QModelIndex, roles=()) -> None:
        if not top_left.isValid() or not bottom_right.isValid():
            return
        r0, r1 = top_left.row(), bottom_right.row()
        c0, c1 = top_left.column(), bottom_right.column()
        if r0 == r1:
            if 0 <= r0 < self._inverse.size:
                row = int(self._inverse[r0])
                self.dataChanged.emit(self.index(row, c0), self.index(row, c1),
                                      roles)
        elif self.rowCount():
            # A multi-row span lands on rows that are scattered in view order; one
            # whole-column repaint says it without a per-row signal storm.
            self.dataChanged.emit(self.index(0, c0),
                                  self.index(self.rowCount() - 1, c1), roles)
        if c0 <= self._sort_column <= c1:
            # Dynamic sort: the sorted-on column just changed. Ties keep the standing
            # view order, as the old proxy's dynamicSortFilter re-sort did.
            self._resort(base=self._order)


class EventTable(QWidget):
    """Sortable, curatable table over an EventSet.

    Keyboard curation: click a row, then j/k move the current row, 'a' accepts, 'r'
    rejects and 'f' marks it with the curator's own flag — the BARE keys only: a chord
    (Cmd+A, Cmd+R, Option+F, …) is not curation, and Cmd+A selects every row
    (:meth:`_handle_curation_event`). Selecting a row emits eventSelected so the trace
    view can jump to it.

    THOSE KEYS WERE DOCUMENTED AND DEAD. The handler was ``keyPressEvent`` on THIS
    widget, which is a plain container: the focus goes to the ``QTableView`` inside it,
    and a QWidget's default focus policy is ``NoFocus``, so this override could never
    be reached and the shipped app's j/k/a/r/f did nothing at all. It is now an event
    filter on the view — where the keystrokes actually arrive — and ``keyPressEvent``
    routes to the same handler so a caller that synthesises a key on the container
    still works.

    Enabling them without fixing :meth:`_EventTableModel.set_flag` would have been
    worse than leaving them dead: 'f' wrote the curator's token straight OVER the flag
    cell, so flagging a boundary-clipped event erased ``clipped_by_boundary`` and left
    the row rejected with no reason. Both halves land together.

    Emits:
        eventSelected(int): event_id of the current row.
        acceptanceChanged(int, bool): event_id, new accepted state — what was WRITTEN.
        flagChanged(int, str): event_id, the row's full new flag set.
        acceptanceRefused(int): an accept the derived half refused (event_id). Nothing was
            written and nothing is announced on ``acceptanceChanged``; the refusal is
            recorded on the ledger and stated on :attr:`refusal_label`, and this signal
            lets the owner repaint the other surfaces that read the same ledger.
        inspectRequested(int): right-click ▸ "Open in Event inspector" on that event_id.
        galleryRequested(int): right-click ▸ "Open in Event gallery" on that event_id.
            Both are INTENT — this widget owns no window and opens nothing; MainWindow
            routes them to the same two doors the gallery's own context menu uses.
    THERE IS NO ``curationBatchChanged`` HERE, AND THAT IS THE MEASURED ANSWER. This class
    declared one, copied from the gallery and the inspector, which do need theirs. Nothing
    ever connected it: measured in the wired app with both curation windows open,
    ``EventTable.receivers("curationBatchChanged()") == 0``. The question is whether that
    was a hole or a redundancy, so it was measured rather than argued — MainWindow at
    1512x900, the owner's 15 s -1000 mV run (115 events), gallery open on "Accepted only"
    and the inspector open, six rows rejected in one 'R':

        accepted 115 -> 109; gallery tiles 24 and status "1–24 of 109 (accepted only) ·
        109 of 115 accepted" — byte-identical to what an explicit
        ``_refresh_event_gallery()`` + ``_refresh_event_inspector()`` produced immediately
        afterwards. Nothing was stale on either window.

    It is a redundancy. The trailing ``acceptanceChanged`` already reaches both mirrors,
    and neither trusts the id it carries: ``EventGalleryWindow.reflect_acceptance`` ends in
    ``_refilter_keeping_selection`` and the inspector's in ``_sync_controls``, both
    whole-frame re-reads over the SHARED df — which is this app's own stated rule ("every
    listener re-reads the shared frame instead"). So one hop repaints all six rows, and a
    second announcement would only add a second reason to do the same work. A signal with
    no receiver and no information of its own is a dead announcement that reads, to the
    next person, like a wire someone forgot; it is gone rather than left to be trusted.

    BATCH CURATION IS THE SAME WRITER, WIDENED. Accept/Reject over a selection lands in
    :meth:`_EventTableModel.set_accepted_rows`, which applies the identical
    ``derived AND requested`` rule :meth:`~_EventTableModel._curate` applies per row and
    records its refusals on the SAME shared ledger — so the frame, the note and the undo
    stack cannot tell a batch from a run of single clicks except by being faster.
    """

    eventSelected = Signal(int)
    acceptanceChanged = Signal(int, bool)
    flagChanged = Signal(int, str)
    acceptanceRefused = Signal(int)
    inspectRequested = Signal(int)
    galleryRequested = Signal(int)
    #: One of the notice chip's menu entries was chosen (its key, see _NOTICE_ACTIONS).
    noticeActionRequested = Signal(str)

    #: The notice the chip's reserved slot is sized for.
    _NOTICE_SLOT_TEXT = "⚠ Chain changed"
    #: The notice chip's menu, built ONCE (key, label).
    _NOTICE_ACTIONS: tuple[tuple[str, str], ...] = (
        ("revert", "Revert form to this detection's chain"),
        ("run", "Run detection with the form's chain…"),
        ("open", "Open Signal Conditioning…"),
    )

    _COLUMN_SETTINGS_KEY = 'inspector/event_columns_v1'
    _SCIENTIFIC_ORDER = ('accepted', 't_start_s', 'dwell_s', 'depth_pA',
                         'amplitude_pA', 'amplitude_method', 'amplitude_status',
                         'frac_depth', 'flag', 'baseline_pA', 'snr',
                         'filter_limited', 'event_id')
    _DEFAULT_HIDDEN = frozenset((
        'start_idx', 'end_idx', 'amplitude_level_pA', 'amplitude_baseline_pA',
        'amplitude_start_idx', 'amplitude_end_idx', 'amplitude_units',
        'amplitude_provenance_json'))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self._applying_columns = False
        self._refreshing_selection = False
        try:
            saved = json.loads(str(QSettings().value(self._COLUMN_SETTINGS_KEY, '{}')))
            self._column_preferences = saved if isinstance(saved, dict) else {}
        except (ValueError, TypeError):
            self._column_preferences = {}
        toolbar = QHBoxLayout()
        self.column_toolbar = toolbar
        toolbar.setContentsMargins(0, 0, 0, 0)
        # Stretch 0: a widget the host inserts with a stretch (the table selector) takes
        # the spare width; with none, this expanding spacer does.
        toolbar.addStretch(0)
        # A notice about the table on screen (e.g. "⚠ Chain changed"), horizontal only.
        # Its slot is reserved even while hidden, so it never moves Columns…, the table
        # selector or the rows below; stretch 1 shares a narrow row with the selector.
        self.notice_chip = _NoticeChip(self._NOTICE_SLOT_TEXT)
        self.notice_chip.setObjectName("eventTableNoticeChip")
        self.notice_chip.setStyleSheet("QPushButton { color:#FFB74D; font-weight:600; }")
        self.notice_chip.ensurePolished()
        self.notice_chip._fit_slot()       # the bold stylesheet font sets the slot width
        self.notice_chip.setVisible(False)
        self.notice_menu = QMenu(self.notice_chip)
        self._notice_actions = {}
        for key, label in self._NOTICE_ACTIONS:
            action = self.notice_menu.addAction(label)
            action.setData(key)
            self._notice_actions[key] = action
        self.notice_chip.clicked.connect(self._open_notice_menu)
        toolbar.addWidget(self.notice_chip, 1)
        self.columns_button = QPushButton('Columns…')
        self.columns_button.setToolTip('Choose visible columns. Drag column headers to reorder; the view is remembered.')
        self.columns_menu = QMenu(self.columns_button)
        self.columns_menu.aboutToShow.connect(self._build_columns_menu)
        self.columns_button.setMenu(self.columns_menu)
        toolbar.addWidget(self.columns_button)
        root.addLayout(toolbar)

        # THE CONSTANT COLUMNS, SAID ONCE. See :meth:`_demote_constant_columns` — this
        # is where their single value goes so that demoting them loses nothing.
        self.constants_label = _ElidingCaption()
        self.constants_label.setStyleSheet("color:#8b9199;")
        self.constants_label.setVisible(False)
        root.addWidget(self.constants_label)

        self.view = QTableView()
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        # EXTENDED, because this is the surface a detection actually lands on and
        # curation on it was genuinely one event at a time. Measured on the pre-fix tree
        # with real mouse events on the owner's -600 mV run (317 rows): a Shift-click five
        # rows down left 1 row selected, a Ctrl-click left 1, Ctrl+A left 1, and 'r' with a
        # six-row range attempted changed exactly 1 row. The gallery — a separate window
        # he has to know to open — was where every batch gesture lived.
        #
        # The idiom is the manual-measurement table's in this same app
        # (``measure_panel.py``: ExtendedSelection + SelectRows + a ``mapToSource``
        # discipline for every row number that leaves the view), copied rather than
        # invented — including its measured trap: under ExtendedSelection
        # ``QTableView.selectRow`` is NOT "select this row", it asks ``selectionCommand``,
        # which consults ``QGuiApplication.keyboardModifiers()``, so with Ctrl still down
        # it EXTENDS. Every programmatic selection here therefore states its command
        # (see :meth:`_select_proxy_row`).
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setAlternatingRowColors(True)
        # The keys are curation, so they belong where the focus is.
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.installEventFilter(self)
        # THE DETECTION LANDS HERE AND THE ROAD STOPPED HERE. This table had no
        # double-click handler and no context menu, so from the surface a detection
        # actually appears on there was NO route to either curation window — the only
        # doors were items 5 and 13 of a native ~14-item menu that on macOS lives at the
        # top of the screen. Right-click is where a user looks for "do something with
        # this row", and the gallery already answers there.
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_menu)

        self._model = _EventTableModel()
        # A proxy gives us sortable columns without disturbing the source frame.
        # ON THE RAW VALUE, not on the formatted string the cell paints — comparing
        # QStrings put "99.49" above "150.2" in a descending depth sort and walked the
        # j/k/a/r curation keys over the wrong rows (see :data:`_SORT_ROLE`) — and
        # VECTORISED, never per-comparison: ``QSortFilterProxyModel`` round-trips into
        # Python ``data()`` twice per comparison, which at 100,000 events froze the GUI
        # thread 48.7 s for ONE header click. :class:`_VectorSortProxy` keeps the same
        # ordering contract and answers in milliseconds.
        self._proxy = _VectorSortProxy(self)
        self._proxy.setSourceModel(self._model)
        self.view.setModel(self._proxy)
        self.view.horizontalHeader().setSectionsMovable(True)
        self.view.horizontalHeader().sectionMoved.connect(self._save_column_preferences)

        root.addWidget(self.view)

        # THE TABLE'S OWN DISCLOSURE CHANNEL — one voice with the gallery's status line
        # and the inspector's batch note. A checkbox the guard snaps back is only honest
        # if a sentence somewhere says why; this label is where the table says it.
        # Rebuilt from the refusal LEDGER on every path that repaints the table (never
        # text written over output), so it survives a model reset, a re-feed and a merge
        # exactly as the windows' notes do — and dies only when the refused set empties.
        self.refusal_label = QLabel("")
        self.refusal_label.setWordWrap(True)
        self.refusal_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.refusal_label.setStyleSheet("color: #8b9199;")
        self.refusal_label.setVisible(False)
        root.addWidget(self.refusal_label)

        # HOW MANY ROWS ARE SELECTED, SAID OUT LOUD — and, in the same sentence, that a
        # selection is a thing you can curate. A range selection that no widget counts is
        # a gesture the user has to trust; and with the batch keys and the context menu
        # both hanging off the selection, the count IS the scope of the next A/R press.
        # An _ElidingCaption for the reason that class exists: a plain QLabel makes its
        # full text width a floor on how narrow the Inspector dock can be dragged (this
        # table's minimumSizeHint().width() is 108 px and must stay there).
        #
        # ELIDE RIGHT, NOT MIDDLE. This is a sentence, not a path: measured at 1512x900 the
        # label is 422 px and the six-row sentence needed 453, so ElideMiddle ate '/ R
        # rejec' — the only naming of the R key on this surface (see _ElidingCaption).
        #
        # AND IT IS NEVER HIDDEN. Showing/hiding it re-laid the column out under the
        # pointer: measured in the shipped Inspector dock at 1512x900, the view's viewport
        # went 134 px -> 113 px the instant a row was clicked and back again when the
        # selection cleared — one whole row of the five appearing and vanishing while the
        # mouse sat over the table, which is the exact "something moved under the cursor"
        # shape three rounds of a previous campaign were lost to. The line is a permanent
        # part of the column and only its TEXT comes and goes; an empty _ElidingCaption is
        # one line high whatever it says, so the geometry is now constant (measured after:
        # 113 px selected and unselected alike). A standing "0 selected" would be chrome,
        # so the reserved line stays blank until it has something to say — and it is where
        # :meth:`_handle_curation_key` says "nothing is selected" when A/R is pressed into
        # an empty selection.
        self.selection_label = _ElidingCaption(elide=Qt.ElideRight)
        self.selection_label.setStyleSheet("color:#8b9199;")
        root.addWidget(self.selection_label)

        #: WHICH EVENTS the table declined to accept — the standalone fallback. In the
        #: wired application MainWindow installs a SOURCE (see :meth:`set_refusals_source`)
        #: that resolves the ledger SHARED with the gallery and the inspector, so all three
        #: surfaces disclose one record; this instance only serves a table nobody wired.
        self._refusals = _RefusalLedger()
        self._refusals_source = None   # callable -> the shared ledger, or None

        #: What the caption says when a curation key was pressed into an EMPTY selection.
        #: Cleared by the next selection change (:meth:`_on_selection_changed`), so it is a
        #: reply to one gesture and never a standing sentence.
        self._armed_note = ""

        self._model.acceptanceChanged.connect(self.acceptanceChanged.emit)
        self._model.acceptanceChanged.connect(self._on_acceptance_written)
        self._model.acceptanceRefused.connect(self._on_acceptance_refused)
        self._model.flagChanged.connect(self.flagChanged.emit)
        # A flag edit can change which remembered rows still count as refused (the token
        # set is half of the recount), so the note follows it too.
        self._model.flagChanged.connect(lambda *_: self._sync_refusal_note())
        self.view.selectionModel().currentRowChanged.connect(self._on_current_changed)
        self.view.selectionModel().selectionChanged.connect(
            lambda *_: self._on_selection_changed())
        # A re-feed replaces the model's rows under the selection model, so the count has
        # to be re-read there too (a reset empties the selection and nothing else says so).
        self._model.modelReset.connect(lambda: self._on_selection_changed())
        self._sync_selection_note()

    # -- public API -------------------------------------------------------
    def set_events(self, events) -> None:
        """Load an EventSet (or a raw DataFrame) into the table."""
        df = getattr(events, "df", events)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected an EventSet or DataFrame, got {type(df)!r}")
        if df is not self._model.df:
            # A DIFFERENT FRAME IS A DIFFERENT DETECTION for the standalone fallback
            # ledger — the events it names no longer exist. DETACHED, never ``.clear()``ed:
            # when the wired source has been feeding this table a SHARED ledger, that
            # record belongs to every surface and to the frame's lineage, and whether it
            # survives this re-feed is the owner's call (it survives an undo and a merge,
            # and dies with a new detection), not this widget's.
            self._refusals = _RefusalLedger()
        self._applying_columns = True
        try:
            self._model.set_df(df)
            self._autosize_columns()
            self._demote_constant_columns(df)
            self._apply_column_preferences()
        finally:
            self._applying_columns = False
        self._sync_refusal_note()

    def _apply_column_preferences(self) -> None:
        """Change header order/visibility only; the source schema is untouched."""
        columns = list(self._model.df.columns)
        preferred = self._column_preferences.get('order', list(self._SCIENTIFIC_ORDER))
        if not isinstance(preferred, list):
            preferred = list(self._SCIENTIFIC_ORDER)
        order = list(dict.fromkeys(name for name in preferred if name in columns))
        # Optional measurements materialize only after the first explicit Apply.
        # Insert only NEW fields beside depth; do not rearrange a saved custom view.
        amplitude_columns = ('amplitude_pA', 'amplitude_method', 'amplitude_status')
        anchor = order.index('depth_pA') + 1 if 'depth_pA' in order else len(order)
        for name in amplitude_columns:
            if name in columns and name not in order:
                order.insert(anchor, name)
                anchor += 1
        order.extend(name for name in columns if name not in order)
        hidden = self._column_preferences.get('hidden', list(self._DEFAULT_HIDDEN))
        if not isinstance(hidden, list):
            hidden = list(self._DEFAULT_HIDDEN)
        # An old preference cannot express visibility for fields that did not exist.
        known = set(preferred)
        hidden = set(hidden) | {c for c in self._DEFAULT_HIDDEN if c not in known}
        header = self.view.horizontalHeader()
        self._applying_columns = True
        try:
            for visual, name in enumerate(order):
                logical = columns.index(name)
                header.moveSection(header.visualIndex(logical), visual)
                self.view.setColumnHidden(logical, name in hidden)
        finally:
            self._applying_columns = False

    def _save_column_preferences(self, *_args) -> None:
        if self._applying_columns or self._model.df.empty and not len(self._model.df.columns):
            return
        columns = list(self._model.df.columns)
        header = self.view.horizontalHeader()
        if header.count() != len(columns):
            return
        order = [columns[header.logicalIndex(i)] for i in range(header.count())]
        hidden = [name for i, name in enumerate(columns) if self.view.isColumnHidden(i)]
        self._column_preferences = {'order': order, 'hidden': hidden}
        QSettings().setValue(self._COLUMN_SETTINGS_KEY, json.dumps(self._column_preferences))

    def set_column_visible(self, name: str, visible: bool) -> None:
        columns = list(self._model.df.columns)
        if name in columns:
            self.view.setColumnHidden(columns.index(name), not visible)
            self._save_column_preferences()

    def reset_column_view(self) -> None:
        self._column_preferences = {}
        QSettings().remove(self._COLUMN_SETTINGS_KEY)
        self._apply_column_preferences()

    def _build_columns_menu(self) -> None:
        self.columns_menu.clear()
        reset = self.columns_menu.addAction('Restore scientific column order')
        reset.triggered.connect(self.reset_column_view)
        self.columns_menu.addSeparator()
        for i, name in enumerate(self._model.df.columns):
            action = self.columns_menu.addAction(str(name))
            action.setCheckable(True)
            action.setChecked(not self.view.isColumnHidden(i))
            action.toggled.connect(lambda checked, column=name: self.set_column_visible(column, checked))

    #: How many rows :meth:`_autosize_columns` measures — spread over the WHOLE frame,
    #: so a wide value in the tail still widens its column (Qt's own pass read only the
    #: first 1000 rows, so anything past them was already unmeasured).
    _AUTOSIZE_SAMPLE_ROWS = 256

    def _autosize_columns(self) -> None:
        """Size every column to its contents from a BOUNDED row sample — O(sample),
        never O(rows).

        ``QTableView.resizeColumnsToContents`` asks the delegate for a size hint cell
        by cell, and each hint is several ``data()`` round-trips into Python — measured
        at 3.6 s per call on a 100,000-event table, and it ran TWICE inside every
        undo/redo (the re-feed, then the stale-badge pass). A column width is a
        question a few hundred formatted values answer as well as a few million: the
        cells are fixed-precision numbers (``.4g``/``.6g``), so widths saturate after a
        handful of rows. The sample is spaced over the whole frame; widths are measured
        with the view's own font metrics on the same display strings the cells paint.
        """
        model = self._model
        df = model.df
        n = len(df)
        fm = QFontMetrics(self.view.font())
        header = self.view.horizontalHeader()
        hfm = QFontMetrics(header.font())
        rows = (np.unique(np.rint(
                    np.linspace(0, n - 1, min(n, self._AUTOSIZE_SAMPLE_ROWS))
                ).astype(np.int64))
                if n else np.zeros(0, dtype=np.int64))
        #: cell text padding / header padding+sort-indicator room / narrowest column
        #: (the checkbox column shows no text and must still fit its indicator).
        cell_pad, header_pad, floor = 12, 26, 28
        for col in range(model.columnCount()):
            head = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
            width = max(floor, hfm.horizontalAdvance(str(head or "")) + header_pad)
            for r in rows:
                text = model.data(model.index(int(r), col), Qt.DisplayRole)
                if text:
                    width = max(width, fm.horizontalAdvance(str(text)) + cell_pad)
            self.view.setColumnWidth(col, width)

    def set_clock_offset(self, offset_s: float | None) -> None:
        """Tell the table which clock its ``t_*`` columns are on relative to the trace.

        Pushed by the owner from the shared TraceView, because the offset is
        ``meta.t0_s + layer.time_offset_s`` — a property of what is DRAWN, not of the
        event frame. See :mod:`nanoscope.core.clock`.
        """
        self._model.set_clock_offset(offset_s)

    #: Text columns that identify the run rather than the row. Demoted when constant.
    _IDENTITY_COLUMNS = ("recording", "pore_id")

    def _demote_constant_columns(self, df: pd.DataFrame) -> None:
        """Move any leading column that is IDENTICAL on every row to the far right,
        and state its single value once, above the table.

        A column whose value never changes carries no per-row information, and these
        two are full-width strings (a file stem and a pore id) sitting in visual
        positions 1 and 2 of twenty. At the Inspector dock's real width that meant
        NO MEASURED COLUMN was on screen without scrolling sideways — on the surface
        the whole detection lands in. Measured on the owner's apoferritin run: the
        first visible measured column was off-screen at every dock width he uses.

        Demoted, not hidden, and only while actually constant: a table pooling two
        pores has a ``pore_id`` that varies, and there it is a real column and stays
        where the schema puts it. The move is VISUAL only —
        ``QHeaderView.moveSection`` reorders what is painted; the model, the frame,
        the 20-column schema, sorting and every export are untouched.
        """
        header = self.view.horizontalHeader()
        cols = list(df.columns)
        shown: list[str] = []
        # IN DECLARATION ORDER. Each move lands its column at the far right, so the LAST
        # one moved ends up rightmost — iterating backwards would reverse them relative
        # to the schema, which is a small thing that reads as a bug the first time
        # someone compares the table with a CSV.
        for name in self._IDENTITY_COLUMNS:
            if name not in cols:
                continue
            series = df[name]
            if len(series) and series.nunique(dropna=False) > 1:
                continue
            logical = cols.index(name)
            visual = header.visualIndex(logical)
            # The scientific default / remembered header layout owns ordering.
            # This method now only presents the constant identity values once.
            value = "" if not len(series) else str(series.iloc[0] or "")
            # An EMPTY constant carries nothing to state. It is still demoted (a blank
            # full-width string column is the worst of both) but it earns no caption:
            # "recording = " on screen is noise pretending to be provenance.
            if value:
                shown.append(f"{name} = {value}")
        # ...and the value they carried is not lost: it is stated once, here, which is
        # the only reason demoting them is honest rather than merely tidy.
        self.constants_label.setText("  ·  ".join(shown))
        self.constants_label.setVisible(bool(shown))

    @property
    def dataframe(self) -> pd.DataFrame:
        """The live DataFrame, with any curation edits applied in place."""
        return self._model.df

    def set_stale_columns(self, names: Iterable[str] = (), tooltip: str = "") -> None:
        """Badge the columns measured by a superseded open-pore estimator.

        Forwarded to the model rather than painted here, so the badge survives
        sorting through the proxy and costs nothing per row. Columns are re-measured
        only when the badge set actually CHANGED: the ⚠ suffix is the only width this
        call can move, and the staleness check runs on every re-feed (an undo, a merge,
        a re-detection) — re-measuring every column for an unchanged answer was half
        the cost of every undo at 100k events."""
        before = self._model.stale_columns
        self._model.set_stale_columns(names, tooltip)
        if self._model.stale_columns != before:
            self._autosize_columns()

    @property
    def stale_columns(self) -> tuple[str, ...]:
        return self._model.stale_columns

    def set_accepted_by_id(self, event_id: int, accepted: bool) -> bool:
        """Set the ``accepted`` state of the row carrying this ``event_id``, writing THROUGH the
        model so the checkbox repaints and ``acceptanceChanged`` fires (summary + span overlays
        refresh downstream). Used by the Event Inspector to curate the shared frame from outside
        the table. Returns True if a row matched — matched, not written: the model's write is
        ``derived AND requested`` (see :meth:`_EventTableModel._curate`), so an accept of a row
        the detector could not measure is refused here exactly as it is at every other surface,
        and the refusal is disclosed on :attr:`refusal_label`. The mirror is unaffected in
        practice: the windows announce what they WROTE, never what was asked, and a written True
        re-derives to True. An allowed value is pushed through unconditionally — even when the
        shared frame already holds it (the inspector may have written it first) — so the view is
        always told to repaint. Selecting a non-existent id is a harmless no-op."""
        df = self._model.df
        if "event_id" not in df.columns:
            return False
        matches = np.nonzero(df["event_id"].to_numpy(dtype=np.int64) == int(event_id))[0]
        if matches.size == 0:
            return False
        self._model.set_accepted(int(matches[0]), bool(accepted))
        return True

    def refresh_event(self, event_id: int) -> None:
        """Repaint the row carrying ``event_id`` after its cells changed in the shared df (an Event
        Inspector mask edit rewrote the bounds + stats). A no-op when no row matches."""
        df = self._model.df
        if list(df.columns) != self._model._columns:
            self.refresh_all()
            return
        if "event_id" not in df.columns:
            return
        matches = np.nonzero(df["event_id"].to_numpy(dtype=np.int64) == int(event_id))[0]
        if matches.size:
            self._model.refresh_row(int(matches[0]))
        # A mask edit can rewrite the row's flag — half of the refused-row recount.
        self._sync_refusal_note()

    def refresh_all(self) -> None:
        """Reset the model so every cell repaints from the shared df — for a bulk curation change
        (Accept all / Reject all), so every checkbox reflects the rewritten ``accepted`` column."""
        keep_ids, current_id = self.selected_event_ids(), self._current_event_id()
        old_columns = list(self._model._columns)
        widths = {name: self.view.columnWidth(i) for i, name in enumerate(old_columns)}
        schema_changed = old_columns != list(self._model.df.columns)
        self._applying_columns = True
        self._refreshing_selection = True
        try:
            self._model.refresh_all()
            if schema_changed:
                self._autosize_columns()
                self._demote_constant_columns(self._model.df)
                self._apply_column_preferences()
                for name, width in widths.items():
                    if name in self._model._columns:
                        self.view.setColumnWidth(self._model._columns.index(name), width)
            self._restore_selection(keep_ids, current_id)
        finally:
            self._applying_columns = False
            self._refreshing_selection = False
        # ...and the note repaints WITH the table: a bulk gesture elsewhere may have
        # recorded refusals on the shared ledger, or retired remembered rows.
        self._sync_refusal_note()

    # -- the refusal ledger, and the note rebuilt from it -----------------
    def set_refusals_source(self, source) -> None:
        """Install the wired application's ledger resolver — a callable returning the
        refusal ledger SHARED with the gallery and the inspector for the CURRENT
        detection, or None when there is none.

        A resolver rather than a ledger instance, because the right ledger changes under
        this widget without notice: the panel re-feeds the table on every resultsChanged
        (an undo, a merge, a new detection), and only the owner knows which of those kept
        the detection's lineage — an undo restores it, a merge edits it, a re-detection
        replaces it. Asking at read time makes the note follow the owner's answer with no
        re-install choreography to forget. When the resolver answers None (no detection),
        the table falls back to its own private ledger, which is all a standalone table
        ever has.
        """
        self._refusals_source = source
        self._sync_refusal_note()

    def refusal_note(self) -> str:
        """The standing disclosure — same sentence, same recount as the gallery's status
        line and the inspector's batch note (see :func:`_refusal_note`): a count of
        CURRENTLY-refused rows re-read from the frame, never a stored number. ``""`` when
        the refused set is empty."""
        return _refusal_note(*self._refusal_ledger().counts(self._model.df), noun="event")

    def _refusal_ledger(self):
        """The ledger this table records to and recounts from: the shared one when the
        owner installed a source and it answers, this table's own otherwise."""
        if self._refusals_source is not None:
            shared = self._refusals_source()
            if shared is not None:
                return shared
        return self._refusals

    def _sync_refusal_note(self) -> None:
        note = self.refusal_note()
        self.refusal_label.setText(note)
        self.refusal_label.setVisible(bool(note))

    def _on_acceptance_written(self, event_id: int, accepted: bool) -> None:
        """A value LANDED. An accept re-decides this event and no other, so it retires any
        refusal an earlier accept recorded about it (the flag may since have been edited
        clean); a reject records nothing and forgets nothing — see ``_RefusalLedger``."""
        if bool(accepted):
            self._refusal_ledger().forget((int(event_id),))
        self._sync_refusal_note()

    def _on_acceptance_refused(self, event_id: int, bucket: str) -> None:
        """The model refused an accept: record WHICH event and which bucket (that part is
        history — see ``_RefusalLedger.record``), rebuild the note, and tell the owner so
        the other surfaces reading the same ledger can repaint theirs. ``forget`` first:
        the bucket is re-decided by every accept over the same row."""
        led = self._refusal_ledger()
        if bucket == "blocked":
            led.record_blocked(1)
        else:
            led.forget((int(event_id),))
            led.record(**{bucket: (int(event_id),)})
        self._sync_refusal_note()
        self.acceptanceRefused.emit(int(event_id))

    def select_event(self, event_id: int) -> None:
        """Select (highlight) the row carrying this ``event_id`` via the sort proxy, which emits
        ``eventSelected``. Used by the Event Inspector so an inspector navigation highlights the
        matching table row. A no-op when no row matches."""
        df = self._model.df
        if "event_id" not in df.columns or len(df) == 0:
            return
        matches = np.nonzero(df["event_id"].to_numpy(dtype=np.int64) == int(event_id))[0]
        if matches.size == 0:
            return
        proxy_index = self._proxy.mapFromSource(self._model.index(int(matches[0]), 0))
        if proxy_index.isValid():
            self._select_proxy_row(proxy_index.row())

    # -- the selection: view rows are never event rows --------------------
    def selected_source_rows(self) -> list[int]:
        """The SOURCE rows behind the current selection, de-duplicated and ascending.

        THE PROXY IS THE WHOLE POINT. This table is sorted through
        :class:`_VectorSortProxy`, so view row N is not event N — a previous bug in this
        app shipped exactly that mistake in a different table, and the manual-measurement
        table's ``_selected_source_rows`` carries the same warning. Every row number that
        leaves this view goes through ``mapToSource``.

        Reads the SELECTION, never ``currentIndex()``: the current index stays valid after
        the visual selection is gone, which is how you curate a row you did not think was
        selected.
        """
        sel = self.view.selectionModel()
        if sel is None:
            return []
        rows = {self._proxy.mapToSource(i).row() for i in sel.selectedRows() if i.isValid()}
        return sorted(r for r in rows if r >= 0)

    def armed_rows(self) -> list[int]:
        """THE row set A / R / F are about to act on, and the one the caption counts.

        There is exactly one answer to "what is the next key going to touch?", and it is
        the selection — what the view paints highlighted. This method exists so that no
        second answer can be invented: the keys, the caption, the context menu and
        :meth:`curate_selected` all arrive here (or at :meth:`selected_row_count`, which
        counts the same selection model without materialising an index per row).

        Empty is a real answer, not a cue to look somewhere else. The old handler read
        ``currentIndex()`` when this was empty or held one row, and the current index
        outlives the highlight — see :meth:`_handle_curation_key` for the two measured
        gestures that made it curate a row nothing on screen showed as chosen.
        """
        return self.selected_source_rows()

    def selected_event_ids(self) -> list[int]:
        """The ``event_id`` of every selected row, in SOURCE order. The identity the
        selection is remembered by across a model reset — a reset is exactly the moment
        row numbers stop meaning anything (see :meth:`curate_selected`)."""
        df = self._model.df
        if "event_id" not in df.columns:
            return []
        ids = df["event_id"].to_numpy(dtype=np.int64)
        return [int(ids[r]) for r in self.selected_source_rows() if r < ids.size]

    def selected_row_count(self) -> int:
        """How many rows are selected, counted from the selection RANGES.

        ``len(selectionModel().selectedRows())`` materialises one QModelIndex per row and
        this runs on every ``selectionChanged`` — i.e. on every step of a Shift-drag and on
        Ctrl+A over the whole detection. The ranges are a handful of objects whatever the
        row count, and the single-range case (a plain click, a Shift-click, Ctrl+A) needs
        no set at all.
        """
        sel = self.view.selectionModel()
        if sel is None:
            return 0
        ranges = sel.selection()
        if len(ranges) == 0:
            return 0
        if len(ranges) == 1:
            return int(ranges[0].bottom() - ranges[0].top() + 1)
        rows: set[int] = set()
        for r in ranges:
            rows.update(range(r.top(), r.bottom() + 1))
        return len(rows)

    def curate_selected(self, accepted: bool) -> int:
        """Accept (or reject) EVERY selected row. Returns how many rows actually moved.

        THE ONE BATCH DOOR — the context menu, the A/R keys over a range and any future
        button all land here, and here lands on :meth:`_EventTableModel.set_accepted_rows`,
        which is :meth:`~_EventTableModel._curate`'s rule applied over a scope. There is no
        second curation writer and no second refusal record: the same shared ledger this
        table already keeps with the gallery and the inspector is updated in ONE call, so
        the sentence under the table reads the same after a batch as after a click.

        WHAT IS ANNOUNCED, and why it is not one signal per row. Measured in the wired app
        on the owner's -600 mV apoferritin recording (317 events): emitting
        ``acceptanceChanged`` 300 times cost **3.72 s** — the accepted summary, the span
        overlays and the detector score are each rebuilt from the whole frame by every one
        of them — and left 300 separate steps on the undo stack for one gesture. So:

        * one row moved -> ``acceptanceChanged`` alone, byte-identical to a checkbox click;
        * more moved -> ONE trailing ``acceptanceChanged`` (the last written row: a true
          statement about that row, and the hop every already-wired listener needs, since
          all of them — summary, spans, score, workflow chip, dirty flag, the history
          checkpoint, the analysis-surface gates — re-read the whole frame and so land
          correct for all N). That includes the two per-row mirrors: neither
          ``reflect_acceptance`` trusts the id it is handed — the gallery's ends in
          ``_refilter_keeping_selection`` and the inspector's in ``_sync_controls``, both
          whole-frame re-reads over the SHARED df — so one hop repaints all N rows.
          Measured with both windows open on the 15 s -1000 mV run: after a six-row 'R'
          the gallery's tiles and status line were byte-identical to an explicit refresh.
          Redundant checkpoints are free by this app's own rule: "a checkpoint that finds
          nothing changed pushes nothing", so the gesture is ONE undoable step, labelled
          by the history's own diff.

        There is NO second, batch-shaped announcement — see the class docstring for the
        measurement that retired it.

        THE SELECTION SURVIVES ITS OWN GESTURE. Anything that re-feeds this table resets
        the model, and a reset clears the selection — the same defect the gallery fixed
        with ``_refilter_keeping_selection``. Remembered by ``event_id`` and not by row (a
        reset is exactly when rows move), restored with an explicit selection command, and
        nothing is scrolled: the rows stay where the user left them.
        """
        rows = self.selected_source_rows()
        if not rows:
            return 0
        keep_ids = self.selected_event_ids()
        current = self._current_source_row()
        df = self._model.df
        current_id = (int(df["event_id"].to_numpy(dtype=np.int64)[current])
                      if current is not None and "event_id" in df.columns
                      and current < len(df) else None)

        result = self._model.set_accepted_rows(rows, bool(accepted))

        # The ledger, in ONE call — an accept re-decides its own scope and no other rows
        # (``_RefusalLedger``), a reject records and forgets nothing.
        led = self._refusal_ledger()
        if accepted:
            led.forget(keep_ids)
            if result.blocked:
                led.record_blocked(result.blocked)
            else:
                led.record(kept=result.kept, repaired=result.repaired)
        self._sync_refusal_note()

        n = int(result.changed.size)
        if n or result.review_changed or result.conflict_changed:
            # ONE announcement, whatever moved: if accepted changed, name its last row;
            # otherwise name the last row in the gesture scope.  The latter is the
            # important fresh-table case — Accept all often changes no booleans, but it
            # still changed ``review`` and must reach the dirty flag/history/export gates.
            row = int(result.changed[-1]) if n else int(rows[-1])
            self.acceptanceChanged.emit(self._model._event_id(row),
                                        bool(df.iat[row, self._model._accept_col()]))
        if result.kept or result.repaired or result.blocked:
            # Nothing more is written by a refusal, so no data signal will repaint the
            # other surfaces' notes — the same repaint-only hop the per-row path uses. The
            # id names ONE refused event because the signal carries one; the receiver
            # (``MainWindow._on_table_acceptance_refused``) ignores it and re-reads the
            # shared ledger, which is where the whole scope already is.
            refused = result.kept or result.repaired or tuple(keep_ids)
            self.acceptanceRefused.emit(int(refused[0]))
        self._restore_selection(keep_ids, current_id)
        return n

    def _restore_selection(self, event_ids, current_id) -> None:
        """Put the selection back on ``event_ids`` after something reset the model.

        By id, never by row. A no-op when the selection is already exactly this set, so an
        UNWIRED table (nothing reset anything) is not made to emit a spurious
        ``selectionChanged``. Never scrolls: moving the view under the pointer is the class
        of change three rounds of a previous campaign were lost to.
        """
        if not event_ids:
            return
        wanted = [int(e) for e in event_ids]
        if self.selected_event_ids() == wanted and (
                current_id is None or self._current_event_id() == current_id):
            return
        df = self._model.df
        if "event_id" not in df.columns:
            return
        ids = df["event_id"].to_numpy(dtype=np.int64)
        sel = self.view.selectionModel()
        if sel is None:
            return
        last_col = max(self._proxy.columnCount() - 1, 0)
        selection = QItemSelection()
        for event_id in wanted:
            matches = np.nonzero(ids == event_id)[0]
            if matches.size == 0:
                continue
            top = self._proxy.mapFromSource(self._model.index(int(matches[0]), 0))
            if not top.isValid():
                continue
            selection.select(top, self._proxy.index(top.row(), last_col))
        if selection.isEmpty():
            return
        sel.select(selection, QItemSelectionModel.ClearAndSelect
                   | QItemSelectionModel.Rows)
        if current_id is not None:
            matches = np.nonzero(ids == int(current_id))[0]
            if matches.size:
                idx = self._proxy.mapFromSource(self._model.index(int(matches[0]), 0))
                if idx.isValid():
                    sel.setCurrentIndex(idx, QItemSelectionModel.NoUpdate)

    def _current_event_id(self) -> int | None:
        row = self._current_source_row()
        df = self._model.df
        if row is None or "event_id" not in df.columns or not (0 <= row < len(df)):
            return None
        return int(df["event_id"].to_numpy(dtype=np.int64)[row])

    #: What the caption says when A / R is pressed into an EMPTY selection — the gallery's
    #: idiom for the same miss ("R labels the SELECTED tiles and none is selected — click
    #: or drag over tiles, then press R"), in the table's own words. See
    #: :meth:`_handle_curation_key`: the keys act on the selection and on nothing else, so
    #: the honest answer to "R with nothing highlighted" is this sentence, not a write.
    _NOTHING_ARMED_NOTE = "Nothing is selected  ·  click a row, then A accepts / R rejects"

    def selection_note(self) -> str:
        """The sentence under the table: how many rows are selected, and what the next
        A / R press would therefore do. ``""`` when nothing is selected and nothing has
        been asked of the empty selection — a permanent caption saying "0 selected" is
        chrome, not disclosure.

        IT FITS THE DOCK IT SHIPS IN. The multi-row sentence used to end "· right-click for
        the same" and needed 453 px in a 422 px label at the owner's 1512x900, so it was
        elided — and with ElideMiddle what vanished was "/ R rejec". The trailing clause is
        now the label's TOOLTIP (see :meth:`_sync_selection_note`) and the painted sentence
        measures 303 px at six rows and 315 px at sixty-one: it fits whole, keys and all.
        """
        n = self.selected_row_count()
        if n <= 0:
            return self._armed_note
        total = len(self._model.df)
        if n == 1:
            return f"1 of {total} rows selected  ·  Shift-click or ⌘-click to select more"
        return f"{n} of {total} rows selected  ·  A accepts / R rejects all {n}"

    def _sync_selection_note(self) -> None:
        """Repaint the caption. NEVER hides it: the line is a permanent part of the column
        (see the constructor for the 134 px -> 113 px measurement) and only its text moves.

        The tooltip carries the clause the label no longer has room to paint — the
        right-click route to the same two commands — so nothing was deleted from the
        disclosure, only moved off the one line that has to fit."""
        note = self.selection_note()
        self.selection_label.setText(note)
        if self.selected_row_count() > 1:
            self.selection_label.setToolTip(f"{note}  ·  right-click for the same")

    def _on_selection_changed(self) -> None:
        """The selection moved, so whatever the keys last said about an empty one is
        stale. One writer for the caption, and the selection is what drives it."""
        self._armed_note = ""
        self._sync_selection_note()

    def _select_proxy_row(self, proxy_row: int) -> None:
        """Make ``proxy_row`` THE selection, stating the command.

        ``QAbstractItemView.selectRow`` is not "select this row": it asks
        ``selectionCommand``, which consults ``QGuiApplication.keyboardModifiers()`` — so
        under ExtendedSelection, with Ctrl or Shift still down, it EXTENDS instead of
        replacing. Reachable by hand (j/k pressed while a modifier is held) and, as
        ``measure_panel.py`` measured for the same call, reproducible in a test suite,
        where a synthetic modified key leaves ``keyboardModifiers()`` latched for the whole
        process. Both callers here mean "this row is now the one".
        """
        idx = self._proxy.index(int(proxy_row), 0)
        if not idx.isValid():
            return
        sel = self.view.selectionModel()
        if sel is None:
            return
        last_col = max(self._proxy.columnCount() - 1, 0)
        sel.select(QItemSelection(idx, self._proxy.index(int(proxy_row), last_col)),
                   QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        sel.setCurrentIndex(idx, QItemSelectionModel.NoUpdate)

    # -- right-click: the two curation windows, on THIS row ---------------
    def _on_context_menu(self, pos) -> None:
        """Per-row context menu: open the event under the cursor on a curation surface.

        Emits INTENT and nothing else — this widget holds no session, no window and no
        other panel, exactly as the gallery's own context menu does; MainWindow routes
        both signals to the doors it already owns (``_open_inspector_at_event`` and
        ``_open_event_gallery``). The menu is built per invocation because its subject is
        whichever row is under the cursor, and a menu that outlives the model it indexes
        into is a stale-row bug waiting to happen.

        The ROW UNDER THE CURSOR decides, not the current row: that is what the user
        pointed at. (Qt has already made it current by the time the menu is requested, so
        in practice the two agree — but only the first of those is the user's statement.)
        """
        index = self.view.indexAt(pos)
        if not index.isValid():
            return
        src_row = self._proxy.mapToSource(index).row()
        df = self._model.df
        if not (0 <= src_row < len(df)):
            return
        event_id = int(self._model._event_id(src_row))

        # A RIGHT-CLICK OUTSIDE THE SELECTION RE-AIMS IT. Qt's own press handler leaves a
        # multi-row selection standing wherever the right button lands, which would let
        # "Reject selected (6)" fire on six rows the user is no longer pointing at. Inside
        # the selection the selection is the subject and is left alone — the platform
        # convention, and the only way a batch gesture is reachable by mouse at all.
        if src_row not in set(self.selected_source_rows()):
            self._select_proxy_row(index.row())

        menu = QMenu(self.view)
        n = self.selected_row_count()
        if n:
            scope = "selected event" if n == 1 else f"{n} selected events"
            act_accept = menu.addAction(f"&Accept {scope}")
            act_accept.setToolTip(
                "Accept every selected row at once — the same decision, and the same "
                "guard, as ticking each checkbox: a row the detector could not measure "
                "stays rejected and is counted in the note under the table.")
            act_reject = menu.addAction(f"&Reject {scope}")
            act_reject.setToolTip(
                "Reject every selected row at once. One step on Edit ▸ Undo, not one "
                "per row.")
            menu.addSeparator()
        else:
            act_accept = act_reject = None
        act_inspect = menu.addAction("Open in Event &inspector")
        act_inspect.setToolTip(
            "Step through this event one at a time: its raw waveform with the baseline, "
            "the level and the draggable mask.")
        act_gallery = menu.addAction("Open in Event &gallery")
        act_gallery.setToolTip(
            "See every event at once as a grid of waveform tiles, with this one focused.")
        chosen = self._exec_menu(menu, self.view.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_accept:
            self.curate_selected(True)
        elif chosen is act_reject:
            self.curate_selected(False)
        elif chosen is act_inspect:
            self.inspectRequested.emit(event_id)
        elif chosen is act_gallery:
            self.galleryRequested.emit(event_id)

    def set_notice(self, text: str, tooltip: str = "",
                   actions: Sequence[str] | None = None) -> None:
        """Show (``text``) or hide (``""``) the toolbar notice chip.

        Only text, tooltip and visibility change, and only when they differ: the chip and
        its menu are built once, because this is called on every form edit.
        """
        text = str(text or "")
        if text and self.notice_chip.text() != text:
            self.notice_chip.setText(text)      # hidden, it keeps its last text
        if self.notice_chip.toolTip() != tooltip:
            self.notice_chip.setToolTip(tooltip)
        wanted = set(self._notice_actions) if actions is None else set(actions)
        for key, action in self._notice_actions.items():
            if action.isVisible() != (key in wanted):
                action.setVisible(key in wanted)
        if self.notice_chip.isHidden() == bool(text):
            self.notice_chip.setVisible(bool(text))

    def notice_text(self) -> str:
        return self.notice_chip.text() if not self.notice_chip.isHidden() else ""

    def _open_notice_menu(self) -> None:
        chip = self.notice_chip
        chosen = self._exec_menu(self.notice_menu, chip.mapToGlobal(chip.rect().bottomLeft()))
        if chosen is not None and chosen.data():
            self.noticeActionRequested.emit(str(chosen.data()))

    @staticmethod
    def _exec_menu(menu, global_pos):
        """The one blocking line, behind a Python seam. ``QMenu.exec`` is a Shiboken
        built-in that cannot be reliably monkeypatched, and offscreen it blocks a test
        forever — so the tests patch THIS and the handler above stays real. Same seam,
        same reason, as :meth:`nanoscope.gui.event_gallery.EventGalleryWindow._exec_menu`.
        """
        return menu.exec(global_pos)

    # -- keyboard curation ------------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        """Curation keys, caught ON THE VIEW — the widget that actually has the focus.

        Returning True consumes the key, which also stops QTableView's built-in
        type-to-search from stealing a/r/f/j/k to jump to a row whose text starts with
        that letter."""
        if obj is self.view and event.type() == QEvent.KeyPress:
            if self._handle_curation_event(event):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        """The same keys if they reach the container (a synthesised event, a future layout
        in which this widget takes focus, or any key a child control ignores and Qt
        therefore climbs to its parent)."""
        if not self._handle_curation_event(event):
            super().keyPressEvent(event)

    def _handle_curation_event(self, event) -> bool:
        """The one gate in front of :meth:`_handle_curation_key`: ONLY AN UNMODIFIED KEY
        CURATES. Returns True if the key was ours.

        THE CHORDS THAT WERE READ AS THE BARE LETTER. Both doors used to hand the handler
        ``event.key()`` and nothing else, so a chord on a curation letter WAS the letter.
        Measured in the wired app on the owner's -600 mV run (317 events) with the keyboard
        in the Results table — which Shift+Cmd+R puts it in on purpose, "so a/r/f keep
        working": Cmd+A (Qt: Ctrl+A) accepted the selected row, Cmd+R rejected it, Cmd+F
        flagged it, and Ctrl+A, Option+R, Option+F and Cmd+Shift+A did the same — each a
        silent write to curation, with Edit ▸ Undo then reading "Undo Curate event" — while
        Select All, the gesture ExtendedSelection was given to this table for, selected
        nothing (one row of 317 stayed selected). The container door had it too: Cmd+R
        pressed on the focused "Columns…" button, which ignores it, climbed here and
        rejected the row.

        So a key with any Ctrl, Meta, Alt or Shift held is not ours: it is left UNHANDLED
        and goes on to QTableView — Cmd+A selects every row, the other chords are ignored
        and keep climbing to the window, where Shift+Cmd+F and Shift+Cmd+R are menu actions
        that win before the table is asked. That is the gallery's rule for the same keys
        (``_handle_own_key`` compares the whole key combination, not the key). The keypad
        flag is the one modifier that does not count — macOS raises it on the arrow keys,
        and it is the idiom the gallery's own arrow handling already uses — and Caps Lock is
        not a modifier to Qt at all, so a letter typed with it on still curates.

        Qt names the Mac keys the other way round (Cmd is ``ControlModifier``, Ctrl is
        ``MetaModifier``), which is why this asks for none of them rather than for one.
        """
        if event.modifiers() & ~Qt.KeypadModifier:
            return False
        return self._handle_curation_key(event.key())

    def _handle_curation_key(self, key) -> bool:
        """Apply one curation key. Returns True if it was ours. THE one handler.

        A / R / F ACT ON THE SELECTION AND ON NOTHING ELSE — the same row set the caption
        counts and the highlight paints. One question, one answer: :meth:`armed_rows`.

        THE GHOST ROW THIS CLOSES. The handler used to fall through to ``currentIndex()``
        whenever the selection held one row or none, and under the new ExtendedSelection
        "none" is a gesture the caption itself teaches ("⌘-click to select more" — the
        same click takes one away). Measured in the wired app at 1512x900 on the owner's
        -1000 mV run, 115 events: click source row 4, ⌘-click it OFF, and the selection is
        empty, the caption is blank and no row is painted — but the current index was
        still parked on row 4 and a single 'R' rejected it, with Edit ▸ Undo then reading
        "Undo Curate event". Worse without the count reaching zero: click row 2, ⌘-click
        row 9 on and off again, and the selection is [2] while the current index is 9 —
        'R' rejected **row 9**, a row that is not highlighted, while row 2 sat selected
        and untouched. Both are silent writes to curation on a project whose whole premise
        is that curation decides the final dataset.

        So an empty selection now means NOTHING IS ARMED. The key is still consumed (it is
        ours, and letting it through would hand a bare letter to QTableView's
        type-to-search) and the caption says why, in the gallery's own idiom for the same
        miss. A one-row selection takes the unchanged per-row write, so an ordinary
        click-then-'r' is byte-identical to before; more than one takes the batch door.
        """
        if key == Qt.Key_J:
            self._move(1)
            return True
        if key == Qt.Key_K:
            self._move(-1)
            return True
        if key not in (Qt.Key_A, Qt.Key_R, Qt.Key_F):
            return False
        rows = self.armed_rows()
        if not rows:
            # Ours, and deliberately inert. The sentence is the whole response.
            self._armed_note = self._NOTHING_ARMED_NOTE
            self._sync_selection_note()
            return True
        if key in (Qt.Key_A, Qt.Key_R):
            if len(rows) > 1:
                self.curate_selected(key == Qt.Key_A)
            else:
                self._model.set_accepted(rows[0], key == Qt.Key_A)
            return True
        # 'f' still marks ONE row — the current one when it is part of what is armed,
        # else the single armed row. ADDED to that row's reasons, never written over
        # them: the token comes from the frozen vocabulary and the union is
        # `combine_flags` (see `_EventTableModel.set_flag`).
        current = self._current_source_row()
        self._model.set_flag(current if current in set(rows) else rows[0], FLAG_CURATED)
        return True

    # -- internal ---------------------------------------------------------
    def _current_source_row(self) -> int | None:
        idx = self.view.currentIndex()
        if not idx.isValid():
            return None
        return self._proxy.mapToSource(idx).row()

    def _move(self, delta: int) -> None:
        n = self._proxy.rowCount()
        if n == 0:
            return
        idx = self.view.currentIndex()
        cur = idx.row() if idx.isValid() else -1
        nxt = int(np.clip(cur + delta, 0, n - 1))
        # NOT ``selectRow``: under ExtendedSelection that consults the live keyboard
        # modifiers and extends instead of replacing — see :meth:`_select_proxy_row`.
        self._select_proxy_row(nxt)

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        # A curation repaint restores the same current event after the model
        # reset. Echoing that restoration as navigation collapses the gallery's
        # multi-selection while it is applying successive labels.
        if self._refreshing_selection or not current.isValid():
            return
        src_row = self._proxy.mapToSource(current).row()
        self.eventSelected.emit(self._model._event_id(src_row))


# ---------------------------------------------------------------------------
# SweepDialog
# ---------------------------------------------------------------------------
class SweepDialog(QDialog):
    """Show a sweep_threshold() result and let the user commit a k.

    The plot is n_events vs k with the plateau region shaded and the surrogate
    (false-positive) count overlaid as a second series. This is the evidence
    that replaces a naked threshold slider.

    Emits:
        kChosen(float): the k the user committed.
    """

    kChosen = Signal(float)

    def __init__(self, sweep_df: pd.DataFrame, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Threshold sweep")
        self._df = sweep_df.reset_index(drop=True)
        self._k_recommended = sweep_df.attrs.get("k_recommended")
        # `sweep_threshold` decides for itself whether its AAFT surrogate is a null for
        # THIS recording and writes the verdict beside `k_recommended` in the same attrs
        # dict. This dialog read one of the two. On the owner's own apo file the surrogate
        # curve sits ~88x ABOVE the real one and flat in k -- and it was drawn in red,
        # labelled "surrogate (false +)", under a sentence promising that the surrogate
        # count "stays low". `reset_index` does not carry attrs, so both are read off the
        # frame as it arrived.
        self._surrogate_is_null = bool(sweep_df.attrs.get("surrogate_is_a_null", True))
        self._surrogate_note = str(sweep_df.attrs.get("surrogate_note", "") or "")

        root = QVBoxLayout(self)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "k (enter threshold, noise RMS)")
        self.plot.setLabel("left", "n events")
        self.plot.addLegend()
        root.addWidget(self.plot)
        self._build_plot()

        info = QLabel(self._recommendation_text())
        info.setWordWrap(True)
        root.addWidget(info)

        buttons = QDialogButtonBox()
        self.use_button = buttons.addButton("Use recommended k", QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        if self._k_recommended is None:
            self.use_button.setEnabled(False)
        root.addWidget(buttons)

        self.use_button.clicked.connect(self._accept_k)
        buttons.rejected.connect(self.reject)

    # -- internal ---------------------------------------------------------
    def _build_plot(self) -> None:
        df = self._df
        k = df["k"].to_numpy(dtype=float)
        n_events = df["n_events"].to_numpy(dtype=float)

        # Shade the plateau region if the sweep marked one.
        if "in_plateau" in df and bool(df["in_plateau"].any()):
            plat_k = k[df["in_plateau"].to_numpy(dtype=bool)]
            region = pg.LinearRegionItem(
                values=(float(plat_k.min()), float(plat_k.max())),
                movable=False,
                brush=(120, 170, 120, 60),
            )
            region.setZValue(-10)
            self.plot.addItem(region)

        self.plot.plot(
            k, n_events, pen=pg.mkPen((31, 119, 180), width=2),
            symbol="o", symbolSize=6, name="events",
        )

        if "n_surrogate" in df:
            n_sur = df["n_surrogate"].to_numpy(dtype=float)
            # The curve is still drawn -- it is real data and hiding it would be its own
            # kind of lie -- but it is named for what it IS on this recording. Calling a
            # curve that runs 88x above the real one a false-positive count is worse than
            # not plotting it.
            name = ("surrogate (false +)" if self._surrogate_is_null
                    else "surrogate — NOT a null here (see below)")
            colour = (214, 39, 40) if self._surrogate_is_null else (140, 140, 140)
            self.plot.plot(
                k, n_sur, pen=pg.mkPen(colour, width=2, style=Qt.DashLine),
                symbol="t", symbolSize=6, name=name,
            )

        if self._k_recommended is not None:
            line = pg.InfiniteLine(
                pos=float(self._k_recommended), angle=90,
                pen=pg.mkPen((200, 200, 200), width=1, style=Qt.DotLine),
                label=f"k*={self._k_recommended:g}",
            )
            self.plot.addItem(line)

    def _recommendation_text(self) -> str:
        if self._k_recommended is None:
            base = "No plateau found — the event count never stabilised. Widen the k range."
        elif self._surrogate_is_null:
            base = (
                f"Recommended k = {self._k_recommended:g} (centre of the plateau where the "
                f"event count is insensitive to threshold, while the surrogate count stays "
                f"low)."
            )
        else:
            # The plateau is still the recommendation -- it is read off `n_events` and does
            # not depend on the surrogate at all. What changes is that the sentence stops
            # citing a curve that is not measuring what it claims to.
            base = (
                f"Recommended k = {self._k_recommended:g} (centre of the plateau where the "
                f"event count is insensitive to threshold)."
            )
        if not self._surrogate_is_null and self._surrogate_note:
            base += "\n\n" + self._surrogate_note
        return base

    def _accept_k(self) -> None:
        if self._k_recommended is None:
            return
        self.kChosen.emit(float(self._k_recommended))
        self.accept()


# ---------------------------------------------------------------------------
# EstimatorCompareDialog
# ---------------------------------------------------------------------------
class EstimatorCompareDialog(QDialog):
    """§4.2.7's report: what re-measuring THIS recording would change, before it does.

    "Give him a way to see the size of the problem before committing." The whole
    reason this dialog exists is that the app cannot tell a recording with the
    block-median pathology from one without it by looking at a stored row -- the
    factor is 1.0x on a clean file and more than 10x on the owner's own -600 mV
    apo recording, and the row looks identical either way. So the evidence is put
    on screen and the decision is left to him.

    Three things are shown, in the order he needs them:

    1. **the sentence** -- median depth before and after, the factor, and how many
       events have a stamped baseline that moves more than 10 pA;
    2. **old vs new depth, per event**, against the y = x line, so a uniform
       rescaling and a per-event mess look different (they are: a clean file sits
       ON the line, the apo file fans out above it);
    3. **the ratio distribution**, because the median alone hides the spread that
       decides whether a population fit merely shifts or changes shape.

    Nothing is installed by opening this. ``exec()`` returns ``Accepted`` only if
    the user pressed *Re-measure now*, and the caller does the installing.
    """

    def __init__(self, comparison: EstimatorComparison,
                 parent: QWidget | None = None, *, can_remeasure: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare baseline estimators")
        self.setMinimumWidth(620)
        self._cmp = comparison

        root = QVBoxLayout(self)

        self.summary_label = QLabel(comparison.summary())
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.summary_label)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 4)
        self.old_label = QLabel(comparison.old_description)
        self.new_label = QLabel(comparison.new_description or "(the configured estimator)")
        for lbl in (self.old_label, self.new_label):
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Measured with", self.old_label)
        form.addRow("Would be measured with", self.new_label)
        root.addLayout(form)

        plots = QHBoxLayout()
        self.scatter_plot = pg.PlotWidget()
        self.scatter_plot.setLabel("bottom", "depth_pA as stored")
        self.scatter_plot.setLabel("left", "depth_pA re-measured")
        self.scatter_plot.setToolTip(
            "One point per event. Points ON the dashed y = x line are unchanged; "
            "points above it are events whose depth was understated by the stored "
            "baseline.")
        self.ratio_plot = pg.PlotWidget()
        self.ratio_plot.setLabel("bottom", "depth ratio (re-measured / stored)")
        self.ratio_plot.setLabel("left", "events")
        self.ratio_plot.setToolTip(
            "The distribution behind the median. A narrow spike at 1.0 means the "
            "estimator change is a no-op on this recording; a broad distribution "
            "away from 1.0 means it changes the shape of every population fit, not "
            "just its scale.")
        plots.addWidget(self.scatter_plot, 1)
        plots.addWidget(self.ratio_plot, 1)
        root.addLayout(plots)
        self._build_plots()

        if comparison.warnings:
            warn = QLabel("Health warnings from the new baseline:\n• "
                          + "\n• ".join(comparison.warnings))
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#ffd479;")
            root.addWidget(warn)

        buttons = QDialogButtonBox()
        self.remeasure_button = buttons.addButton(
            "Re-measure now", QDialogButtonBox.AcceptRole)
        buttons.addButton("Close", QDialogButtonBox.RejectRole)
        # Enabled even when nothing changes -- re-measuring a clean file is a
        # harmless no-op that stamps the new fingerprint and clears the banner, and
        # a disabled button with no explanation is the failure mode this project
        # already ruled against (buttons stay enabled and post the reason).
        self.remeasure_button.setEnabled(bool(can_remeasure))
        self.remeasure_button.setToolTip(
            "Re-measure these events against the configured estimator. Events are "
            "not re-detected: every span and every accept/reject survives."
            if can_remeasure else
            "There is no live recording to re-measure against.")
        root.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    # -- internal ---------------------------------------------------------
    def _build_plots(self) -> None:
        c = self._cmp
        old = np.asarray(c.old_depth_pA, dtype=float)
        new = np.asarray(c.new_depth_pA, dtype=float)
        ok = np.isfinite(old) & np.isfinite(new)
        if ok.any():
            self.scatter_plot.plot(
                old[ok], new[ok], pen=None, symbol="o", symbolSize=5,
                symbolBrush=(31, 119, 180, 140), symbolPen=None, name="events")
            lo = float(min(old[ok].min(), new[ok].min()))
            hi = float(max(old[ok].max(), new[ok].max()))
            self.scatter_plot.plot(
                [lo, hi], [lo, hi],
                pen=pg.mkPen((200, 200, 200), width=1, style=Qt.DashLine))

        ratio = np.asarray(c.depth_ratio, dtype=float)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            # Fixed bin count rather than a rule-of-thumb: the shape being read here
            # is "one spike at 1.0" vs "a broad smear", and a data-dependent bin
            # width makes those two look alike on small tables.
            counts, edges = np.histogram(ratio, bins=min(40, max(8, ratio.size // 4)))
            self.ratio_plot.plot(
                edges, counts, stepMode="center", fillLevel=0,
                brush=(31, 119, 180, 120), pen=pg.mkPen((31, 119, 180), width=1))
            self.ratio_plot.addItem(pg.InfiniteLine(
                pos=1.0, angle=90, label="1.0x",
                pen=pg.mkPen((200, 200, 200), width=1, style=Qt.DotLine)))

    def summary_text(self) -> str:
        """The rendered headline (tests read this)."""
        return self.summary_label.text()
