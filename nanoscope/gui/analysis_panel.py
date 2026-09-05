"""The analysis inspector tab: params form + Run + results + navigation (PORT_PLAN §6, gap #5).

``AnalysisPanel`` is the one place the ported DSP meets the live viewer. Like
:class:`~nanoscope.gui.calibration_panel.CalibrationPanel` it is constructed with
the :class:`~nanoscope.core.session.Session` and driven by
``MainWindow.set_selection``; it never talks to another top-level panel, only to
Session (via signals) and to the :class:`~nanoscope.gui.trace_view.TraceView` it
was handed (the documented viewer seam — ``add_overlay`` / ``zoom_x_to`` /
``set_filter_preview``).

What it does:

* **Run** — :class:`DetectionPanel` emits ``detectRequested(params)``; the panel
  gates on ``accepts_kinds`` (channel is CURRENT) and ``requires_time_base``
  (``sample_rate_hz`` is not None) and launches ``detect.schmitt`` on the shared
  :class:`~nanoscope.gui.analysis_worker.AnalysisController`.
* **Results** — on ``session.resultsChanged`` for the current selection it fills
  the :class:`EventTable` + :class:`StatsPanel`, shows the polarity evidence, and
  redraws the span overlays (removing the previous ones first).
* **Navigation** — clicking an event row zooms the trace to it
  (``TraceView.zoom_x_to``) and drops a single MARKER overlay on it.
* **Staleness** — every result carries the ``calibration_fingerprint`` it was
  computed under; when the live calibration diverges (the user's deferred
  current-gain workflow guarantees this fires) a banner offers a re-run.
* **Filter preview (D4)** — the DetectionPanel's "show on trace" toggle drives the
  managed low-pass preview curve on the pane.
* **Fits / sweep / null / recovery** — fits run in-process on the small event
  table; the three surrogate studies run on the controller's generic callable
  path (they re-detect on full-trace surrogates, so they must be off-thread).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..analysis import ml_detect
from ..analysis.baseline import (
    FALLBACK_CLAMPED,
    ConstantBaselineWarning,
    adapt_to_length,
    baseline_subtracted,
    characterisation_baseline,
    collect_baseline_warnings,
    estimated_wander,
    remove_baseline_wander,
)
from ..analysis.baseline_probe import BaselineProposal
from ..analysis.characterize import recharacterize_span
from ..analysis.detect import gap_window_samples
from ..analysis.conditioning import (
    BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
    BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
    baseline_correction_export_paths,
    characterization_trace,
    conditioned_trace,
    write_baseline_correction,
)
from ..analysis.events import EventSet, mark_reviewed
from ..analysis.merge import (
    MIN_MERGE_OCCUPANCY,
    merge_by_gap,
    merge_region,
    span_occupancy,
)
from ..analysis.modules import (
    DETREND_PARAM_KEYS,
    FILTER_PARAM_KEYS,
    any_preprocess_enabled,
    detection_params_from_provenance,
    detection_signal,
    detection_signal_key,
    detrend_method_kwargs,
    fit_depth,
    fit_dwell,
    preprocess_for_detection,
    reconcile_result,
)
from ..analysis.remeasure import (
    REMEASURED_COLUMNS,
    RemeasureResult,
    curation_snapshot,
    describe_stored_baseline,
    frozen_for_dispatch,
    merge_guard_snapshot,
    merge_late_curation,
)
from ..analysis.labels import label_conflicts
from ..analysis.reconcile import (
    CONSENSUS_LABEL_CONFLICTS_COL,
    CONSENSUS_MODULE_ID,
    clear_curation_conflicts,
    reconciliation_source_fingerprint,
)
from ..analysis.scoring import ADDED_COLUMN, ensure_added_column, score_detection
from ..core.analysis import AnalysisResult, OverlayHint, OverlayKind
from ..core import clock
from ..core.segment import persistent_key
from ..core.session import Session
from ..core.sources import (
    DETECTION_MODULE_IDS as STORED_DETECTION_MODULE_IDS,
    detector_name,
)
from ..core.types import ChannelKind
from ..plot.style import PALETTE
from .analysis_worker import AnalysisController
from .baseline_panel import BaselinePanel
from .baseline_workspace import BaselineCorrectionWorkspace
from .categorical import category_colour
from .export_paths import ask_export_path
from .measure_panel import MeasurePanel
from .merge_panel import MergePanel
from .panels import DetectionPanel, EventTable, ScorePanel, StatsPanel, SweepDialog
from .overlays import STEP_LEVEL_HEX
from .trace_view import TraceView

__all__ = ["AnalysisPanel", "MergeReport"]


class MergeReport(NamedTuple):
    """Outcome of a post-detection merge, so MainWindow can post an honest status line.

    ``status`` is one of ``"ok"`` (a merge happened), ``"noop"`` (nothing was within the
    window / region), ``"low_occupancy"`` (something WAS within the window / region and
    every candidate was refused — see :data:`~nanoscope.analysis.merge.MIN_MERGE_OCCUPANCY`),
    ``"no_detection"`` (no active detection to merge), ``"reconciled"`` (the active
    derived table is curation-only), or ``"no_selection"`` (a region merge with no
    usable on-trace Measure selection). ``focus_id`` is the ``event_id``
    the Event Inspector should advance to after a "Merge with next" (``-1`` when not
    applicable)."""

    status: str
    n_before: int
    n_after: int
    focus_id: int = -1
    #: The window the merge was asked for, in seconds, so a report that arrives LATE
    #: (the "Merge all" job runs off-thread — see
    #: :meth:`AnalysisPanel.merge_all_subevents`) can still say what it was merging
    #: within. MainWindow used to hold this in a local, which only works while the
    #: answer comes back inside the call.
    window_s: float | None = None
    #: How many candidate groups the occupancy floor DECLINED to merge. Non-zero with
    #: ``status == "low_occupancy"`` (nothing merged at all) and also with
    #: ``status == "ok"`` (some runs merged, others were mostly open pore). A refusal
    #: nobody prints is indistinguishable from the merge silently not working — which is
    #: the complaint the floor exists to answer — so MainWindow's ``_post_merge_status``
    #: says so in both cases.
    n_refused_low_occupancy: int = 0

_MODULE_ID = "detect.schmitt"
_ML_MODULE_ID = "detect.keras"
#: The owner's one-sided sequential CUSUM detector, selected by the Detection tab's own
#: detector combo (``params()["detector"]``). NOT to be confused with
#: :mod:`nanoscope.analysis.cusum`, which fits sub-state LEVELS inside events a detector
#: has already found -- a different algorithm at a different stage.
_CUSUM_MODULE_ID = "detect.cusum"
# Every detector (plus the derived reconciled table) produces the SAME EventSet shape and
# feeds the SAME table / overlays /
# inspector / plots / exports. The panel tracks WHICH one is currently displayed in
# ``_active_detection``; everything downstream is module-id-agnostic.
_DETECTION_MODULE_IDS = tuple(STORED_DETECTION_MODULE_IDS)

_RECONCILED_EDIT_REFUSAL = (
    "The Reconciled table is curation-only. Switch to a source detector table, "
    "add or merge there, then reconcile again."
)
_RECONCILED_BASELINE_REFUSAL = (
    "The Reconciled table can contain rows from different detector conditioning and "
    "cannot be re-measured against one baseline. Switch to a source detector table, "
    "re-measure there, then reconcile again."
)
_RECONCILED_STALE_REFUSAL = (
    "A Reconciled table cannot be refreshed directly because its rows can come from "
    "different source detectors and preprocessing chains. Switch to each source detector "
    "table, refresh the stale source results, then reconcile again."
)
_RECONCILED_SOURCE_CHANGE_REFUSAL = (
    "This Reconciled table is a preserved snapshot, not a result that can be refreshed "
    "in place. Review or restore its changed or missing source tables, then reconcile "
    "again."
)

# The ML detector's OWN preprocessing/decode contract, carried verbatim from the panel
# into the module's params (and therefore into the stored result, so a run is
# reproducible). Deliberately NOT the general Preprocess-tab keys: an ML run never
# touches preprocess_for_detection.
_ML_CONFIG_KEYS = (
    "ml_preset", "ml_preprocess", "ml_decode",
    "ml_preset_customized", "ml_preset_deviations",
)

# The comparison is a reading aid, not a second full-resolution trace viewer.  A few
# thousand evenly-spaced samples preserve slow baseline shape and event density while
# keeping a 15 M-sample recording responsive when a worker result crosses back to Qt.
_BASELINE_WORKSPACE_PREVIEW_POINTS = 6_000
_BASELINE_WORKSPACE_ZERO = BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY
_BASELINE_WORKSPACE_RETAINED = BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME


def _baseline_workspace_arrays(
    raw: np.ndarray,
    fs: float,
    *,
    reference_config: Any,
    reference_source: str,
    runtime_params: Mapping[str, Any],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], Mapping[str, Any] | None]:
    """Return fitted baseline, correction, advisories and actual fit provenance.

    This is deliberately narrower than :func:`conditioned_trace`: the workspace
    compares one baseline operation at a time.  Its retained-offset branch is the
    Baseline drift *stage* itself, byte-for-byte in formula with
    ``remove_baseline_wander``; any other enabled preprocessing stage remains a
    separate, explicitly named part of the Signal chain.  Collapsing them into this
    graph would make an orange low-pass or high-pass output look like a baseline
    correction, which is exactly the ambiguity this workspace exists to remove.
    """
    x = np.asarray(raw)
    polarity_value = runtime_params.get("polarity")
    polarity = None if polarity_value is None else int(polarity_value)
    result_warnings: tuple[str, ...] = ()
    fit_provenance: Mapping[str, Any] | None = None
    with collect_baseline_warnings() as kernel_warnings:
        if mode == _BASELINE_WORKSPACE_ZERO:
            result = characterisation_baseline(
                x, float(fs), config=reference_config, polarity=polarity,
                param_source=str(reference_source), health=False, cache=False,
            )
            baseline = np.asarray(result.baseline)
            corrected = baseline_subtracted(x, baseline)
            # Short recordings can clamp the requested windows or fall back to a
            # constant fit.  The result, not the form, is the authority on what
            # actually ran and which advisories belong in an exported record.
            result_warnings = tuple(str(item) for item in result.warnings if str(item))
            fit_provenance = result.provenance()
        elif mode == _BASELINE_WORKSPACE_RETAINED:
            method = str(runtime_params.get("detrend_method", "block_median"))
            window_s = float(runtime_params.get("detrend_window_s", 0.5))
            method_kwargs = detrend_method_kwargs(runtime_params)
            # The display rail and the retained-offset output are deliberately
            # obtained through the same two public analysis doors that power the
            # actual stage.  ``remove_baseline_wander`` also contributes its own
            # amplified-wander advisory, which a hand-written ``x - b + median(b)``
            # preview would silently omit.
            baseline = estimated_wander(
                x, float(fs), method=method, window_s=window_s,
                polarity=polarity, **method_kwargs,
            )
            corrected = remove_baseline_wander(
                x, float(fs), method=method, window_s=window_s,
                polarity=polarity, **method_kwargs,
            )
        else:
            raise ValueError(f"Unknown baseline workspace mode: {mode!r}")
    warnings = tuple(dict.fromkeys((
        *result_warnings,
        *(str(item) for item in kernel_warnings if str(item)),
    )))
    return (np.asarray(baseline), np.asarray(corrected), warnings, fit_provenance)


_BASELINE_WORKSPACE_OTHER_STAGES: tuple[tuple[str, str], ...] = (
    ("notch_enabled", "notch"),
    ("median_enabled", "median"),
    ("hp_enabled", "high_pass"),
    ("rmean_enabled", "running_mean"),
    ("savgol_enabled", "savgol"),
    ("normalize_enabled", "normalize"),
)


def _baseline_workspace_excluded_stages(params: Mapping[str, Any]) -> tuple[str, ...]:
    """Enabled processing stages deliberately outside a baseline-only comparison."""
    stages = [name for flag, name in _BASELINE_WORKSPACE_OTHER_STAGES if params.get(flag)]
    if params.get("filter_cutoff_hz") is not None:
        stages.append("low_pass")
    return tuple(stages)


def _baseline_workspace_recipe_signature(
    *,
    mode: str,
    runtime_params: Mapping[str, Any],
    reference_params: Mapping[str, Any],
    reference_source: str,
) -> tuple[Any, ...]:
    """Immutable signature for every control that can change a job or its label."""
    if mode == _BASELINE_WORKSPACE_ZERO:
        recipe = tuple(sorted((str(key), repr(value))
                              for key, value in reference_params.items()))
        return (mode, str(reference_source), repr(runtime_params.get("polarity")), recipe)
    runtime = tuple((key, repr(runtime_params.get(key)))
                    for key in ("polarity", *DETREND_PARAM_KEYS))
    return (mode, runtime, _baseline_workspace_excluded_stages(runtime_params))


def _baseline_workspace_preview(
    raw: np.ndarray,
    baseline: np.ndarray,
    corrected: np.ndarray,
    fs: float,
    *,
    max_points: int = _BASELINE_WORKSPACE_PREVIEW_POINTS,
) -> dict[str, np.ndarray]:
    """Evenly decimate a complete baseline comparison for the small workspace plot."""
    x = np.asarray(raw)
    b = np.asarray(baseline)
    out = np.asarray(corrected)
    if x.ndim != 1 or b.shape != x.shape or out.shape != x.shape:
        raise ValueError("baseline workspace arrays must be one-dimensional and equally long")
    n = int(x.size)
    if n == 0:
        return {
            "time_s": np.empty(0, dtype=np.float64),
            "raw": np.empty(0, dtype=np.float32),
            "baseline": np.empty(0, dtype=np.float32),
            "corrected": np.empty(0, dtype=np.float32),
        }
    count = min(n, max(2, int(max_points)))
    indices = np.arange(n, dtype=np.int64) if count == n else np.linspace(
        0, n - 1, num=count, dtype=np.int64)
    return {
        "time_s": indices.astype(np.float64) / float(fs),
        "raw": np.asarray(x[indices], dtype=np.float32),
        "baseline": np.asarray(b[indices], dtype=np.float32),
        "corrected": np.asarray(out[indices], dtype=np.float32),
    }

#: *** THE NOTCH-LINE CAP. *** The most notch lines an ML preprocess run may be configured
#: with through this panel. ``notch_sequential`` runs one zero-phase ``filtfilt`` over the
#: WHOLE analysed range PER LINE by contract (see its docstring — the passes are
#: deliberately sequential, not one SOS), measured at 0.8-0.9 ms/line on 60k samples,
#: i.e. ~200 ms/line at the app's own 15 M-sample SOFT_SAMPLE_CEILING. The trained
#: presets use 3 lines (60/120/180); 32 is enough for a 50 Hz mains comb past 1.5 kHz —
#: more harmonics than any recording here has ever needed — while bounding the worst
#: configurable cost at ~6 s per run at the ceiling. Without the cap, ONE accidental
#: clipboard paste of a spreadsheet column into the freeform "notch lines" box committed
#: every subsequent ML preprocess run to 10,923 identical passes (~35-40 min at the
#: ceiling), with the only on-screen trace being a preset-status line naming the field.
#: The refusal names this limit and the count it refused — never a silent truncation,
#: which would run a different filter than the one the box displays.
#:
#: ONE constant, not two. The Preprocess tab's explicit "lines" box grew the same cap in
#: sweep 7 (S7-NOTCH1: a truncated 10,923-value paste cost 42 s per detection on the
#: -600 mV file), and it lives on the DetectionPanel because that panel owns the box;
#: this name is kept for the ML launcher and its tests, as an ALIAS, so the two boxes
#: on one panel can never be capped differently. (``panels`` cannot import this module
#: back — it would be circular — so the definition sits there and the alias here.)
_ML_NOTCH_MAX_LINES = DetectionPanel.NOTCH_MAX_LINES

# The two staleness stamps a stored result carries, and the repair each one wants.
# They are NOT interchangeable and the difference is the whole of §4.2.3: a
# calibration change invalidates the detection (the thresholds were in the old
# gain), so it re-DETECTS; a baseline change invalidates only the amplitudes, so it
# RE-MEASURES and the curated event set survives. Wiring one button to
# "re-run the detection" for both would answer a baseline change by destroying every
# accept/reject the owner had made.
#: The identity :meth:`AnalysisPanel._show_empty` points the owned per-recording panels
#: at. NOT ``None``: ``BaselinePanel.set_recording(fs, target=None)`` documents ``None``
#: as "do not judge", so the panel kept the previous recording's windows, its
#: ``'measured on this recording, and confirmed'`` source line AND its standing
#: ACCEPTED-OVER-FAILED-HEALTH-CHECKS mark on screen after that recording was CLOSED —
#: a readout describing a file that is no longer open, which is the defect
#: :meth:`AnalysisPanel._clear_results` exists to prevent, on a panel it owns rather than
#: on one of its own labels. "Nothing selected" is a target like any other; it is one
#: value so re-entering it twice is navigation and costs nothing. It carries the same
#: ``(doc_id, channel)`` shape as a real one and can never collide with a real doc_id
#: (a document's id is derived from a path; see :mod:`nanoscope.core.identity`).
_NOTHING_SELECTED = ("<no recording selected>", -1)

_STALE_CALIBRATION = "calibration"
_STALE_BASELINE = "baseline"
_STALE_RECONCILIATION_SOURCE = "reconciliation_source"
_STALE_ACTION_TIPS = {
    _STALE_CALIBRATION: (
        "Run this SAME detection again under the new gain/offset. The event table is "
        "replaced, so curation on the current rows does not survive."),
    _STALE_BASELINE: (
        "Re-measure the amplitude columns of these events against the open-pore "
        "estimator now configured. Events are NOT re-detected: every span, and every "
        "accept/reject and hand-written flag on it, is kept."),
}


def schmitt_params_from_result(res: AnalysisResult | None) -> dict[str, Any]:
    """The settings a stored Schmitt detection ran under, or ``{}`` if they cannot be known.

    Two sources, in order of fidelity:

    1. ``res.params`` — the exact dict the module was handed. Present on every live run and,
       since project v3, on every restored one.
    2. the EventSet provenance — the detector's own stamp of its thresholds and its
       preprocessing chain. The fallback for a v1/v2 project, which stored no params at all
       and therefore came back with ``params={}``.

    ``{}`` means "the run cannot be reproduced": the caller must fall back to the live form
    rather than pretend, because a detection re-run under settings that are not the ones on
    screen replaces the event table — and every hand curation on it — with something else.
    """
    if res is None:
        return {}
    if res.params:
        return dict(res.params)
    events = res.payload
    if isinstance(events, EventSet) and events.provenance:
        return detection_params_from_provenance(events.provenance)
    return {}


def _baseline_summary_rows(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The open-pore reference a detection's depths were measured against, for display.

    Read from the STORED detection's provenance, never from the live baseline form:
    the form holds what the next run would use, and after a parameter edit (or a
    project reopened under a different default) those are two different windows.
    Showing the form's numbers beside the stored table's depths would state a
    provenance that is simply not the one the numbers have — the same class of
    mistake the stale-column badge exists to prevent.

    Returns ``{}``-safe placeholders rather than raising: an old project, or an ML
    detection stamped before the baseline keys existed, still has a table worth
    showing and must say "not recorded" rather than nothing.
    """
    rough = provenance.get("baseline_rough_s")
    base = provenance.get("baseline_base_s")
    fingerprint = provenance.get("baseline_fingerprint")
    if rough is None or base is None:
        window = "not recorded"
    else:
        window = f"rough {float(rough):g} s / base {float(base):g} s"
    return {
        "baseline_window": window,
        "baseline_fingerprint": str(fingerprint) if fingerprint else "not recorded",
    }


def _scroll_wrap(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    """Put ``widget`` inside a resizable :class:`QScrollArea` that imposes NO width floor.

    This is the one pattern that keeps the Inspector dock draggable in both directions. A
    ``QScrollArea`` reports its own tiny ``minimumSizeHint`` instead of its child's, so a
    control stack that cannot physically shrink (a row of push buttons never wraps and
    ``qSmartMinSize`` will not take a button below its text) stops dictating how narrow the
    dock may be. The child keeps its natural width; ``Qt.ScrollBarAsNeeded`` — the default,
    pinned here so it survives a restyle — hands the user a horizontal scrollbar the moment
    the viewport is narrower than that.
    """
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    area.setMinimumWidth(0)
    area.setWidget(widget)
    return area


def _pin_scroll_height(area: QtWidgets.QScrollArea, content: QtWidgets.QWidget) -> None:
    """Pin ``area`` at ``content``'s natural height PLUS the room a horizontal scrollbar
    would need — reserving that room where the LAYOUT can actually hand it over, and
    without imposing a height floor.

    Two things are being bought here, and both are about the strip not moving.

    * A ``widgetResizable`` scroll area is ``Expanding`` in both directions, so a short
      fixed-height control group dropped into a tall pane would be stretched to fill it.
      The cap keeps the strip the height it asked for and leaves the rest of the pane to
      the widget below.
    * ``Qt.ScrollBarAsNeeded`` makes a horizontal scrollbar APPEAR when the dock is dragged
      narrower than the content. If the area were capped at exactly the content height that
      scrollbar would have to take its 15-18 px out of the content — the group would
      re-flow, and a button the user was aiming at would move.

    A MAXIMUM IS NOT A GRANT, and pretending otherwise is what this function got wrong the
    first time. It set ``setMaximumHeight(natural + frame + bar)`` and left the reserved
    room to "the caller's trailing stretch"; but the enclosing ``QVBoxLayout`` hands a
    non-stretch child its ``sizeHint``, and a ``QScrollArea``'s sizeHint is ``frame +
    content.sizeHint()`` — a trailing ``addStretch`` contributes 0 to that. Measured on the
    owner's 1512x900 with a fresh profile: maximumHeight 115, actual height 97 at EVERY
    Inspector width from 860 down to 140, i.e. the 18 px was never taken up. So when the
    scrollbar was finally needed it did the one thing the paragraph above forbids — at dock
    260 the viewport went 258x95 -> 220x77, the 95 px of content no longer fitted, a
    VERTICAL scrollbar appeared too (max 18, stealing another 18 px of width), and "Merge
    events in selection" (content y=59..84) was bottom-clipped by 7 px. That width is
    reachable without any dragging: ``test_window_layout`` saves an Inspector width of 180.

    So the reservation is made in the one place the layout reads — ``content``'s own
    ``sizeHint`` — from the SAME ``bar`` that sizes the cap: a trailing spacer of exactly
    ``bar`` pixels whose *minimum* is zero. Now the granted height and the reserved height
    are one quantity: the area's sizeHint is ``natural + bar + frame``, which is also its
    maximum, so the layout grants exactly that. Wide, the spacer holds the empty 18 px
    under the group; narrow, the horizontal scrollbar takes the spacer's place and the
    viewport is still ``natural`` tall, so the content fits, no vertical scrollbar appears
    and nothing above it shifts by a pixel.

    Only the MAXIMUM is set. A minimum would propagate into ``QDockWidget::minimumSize``
    the same way a width floor does, and a short window (the suite measures at 775x450)
    could no longer make the dock shorter than this strip — which is also why the spacer's
    minimum has to be zero rather than ``bar``.
    """
    natural = max(content.sizeHint().height(), content.minimumSizeHint().height())
    frame = 2 * area.frameWidth()
    bar = area.horizontalScrollBar().sizeHint().height()
    layout = content.layout()
    if layout is not None:
        # sizeHint ``bar``, minimum 0, and it absorbs any surplus the pane hands over
        # instead of letting the control group above it stretch.
        layout.addItem(QtWidgets.QSpacerItem(
            0, bar, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))
    area.setMaximumHeight(natural + frame + bar)
    # ...and the floor is taken away, for the reason the docstring gives. A bare
    # ``QScrollArea`` reports ``minimumSizeHint() == 90x90`` whatever it holds, and that 90
    # would ADD to the event table's own 129 inside this pane: measured, it pushed
    # ``MainWindow.minimumSizeHint().height()`` from 363 px to 459 px, so a window the user
    # (or the suite, which measures at 775x450) asked to be 450 px tall came back 459.
    # ``qSmartMinSize`` only honours an explicit minimum that is greater than zero, so 1 px
    # is how "no floor" is spelled; the strip still takes its full ``sizeHint`` height
    # wherever the pane has the room, which at 1512x900 it does with 187 px left for the
    # table.
    area.setMinimumHeight(1)


class _ElidedLabel(QtWidgets.QLabel):
    """A QLabel that elides to the width it is GIVEN instead of demanding the width its
    text wants.

    A plain QLabel reports ``minimumSizeHint().width() == the full text width``, and that
    number propagates all the way up to ``QDockWidget::minimumSize()`` — so this panel's
    long "recording · channel · kind · units" title silently became a floor on how narrow
    the user could drag the Inspector dock. Here the full string stays ``text()`` (callers
    and tests are unaffected), only the painted copy is elided, and the horizontal minimum
    is zero: the dock can be dragged narrow and the title still reads.

    PLAIN TEXT, EXPLICITLY. The one string this label carries is
    "recording · channel · kind · units", and every one of those four pieces comes
    verbatim out of the FILE — the path the user chose, ``ChannelInfo.name`` and
    ``raw_units`` off the header (``io/text.py`` takes the CSV column header verbatim as
    the channel name). A QLabel left at ``Qt::AutoText`` runs ``mightBeRichText()`` over
    what it is handed and parses it as HTML the moment a '<' is followed by a recognised
    tag name, so a channel really named ``I<sub>0</sub>`` was painted ``I0`` and one named
    ``Ch <A>`` was painted ``Ch`` — the part of the name that distinguishes it deleted on
    its way to the screen, while ``text()`` still returned the whole thing, which is why
    the one test on this title (asserting on the source) went green over it. Same rule and
    same remedy as ``file_info_panel``/``status_chip``/``event_gallery``; the reason is
    spelled out at length in ``baseline_panel._check_html``.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setTextFormat(QtCore.Qt.PlainText)
        self._full = ""
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full = text
        self._apply_elide()

    def text(self) -> str:  # noqa: N802 (Qt override)
        return self._full

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        # Preferred width tracks the FULL text (so a wide dock shows all of it); deriving it
        # from the elided copy instead would let the label ratchet itself narrower.
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full),
                            super().sizeHint().height())

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        elided = self.fontMetrics().elidedText(
            self._full, QtCore.Qt.ElideRight, max(0, self.width()))
        if elided != QtWidgets.QLabel.text(self):
            QtWidgets.QLabel.setText(self, elided)


class _StaleBanner(QtWidgets.QFrame):
    """A thin banner shown when a stored result no longer matches the live settings.

    TWO stamps can go stale, and they want DIFFERENT repairs, so the banner carries
    one action button whose label and meaning are set with the message:

    * the ``calibration_fingerprint`` -- the gain/offset moved, so the whole
      detection has to run again ("Re-run");
    * the ``baseline_fingerprint`` -- the open-pore estimator moved, so the
      amplitudes have to be measured again but the events themselves, and every
      hand curation on them, must survive ("Re-measure depths").

    A single hard-wired handler was fine while there was one stamp. With two it
    would silently re-DETECT for a baseline change -- replacing the event table,
    and with it every accept/reject the owner had made -- which is precisely the
    outcome §4.2.3 exists to prevent. So the button emits an intent and the panel
    dispatches on the reason it last showed.

    THE BUTTON IS GATED, AND THE GATE IS SHOWN. The panel's Detect / ML / probe buttons
    take the detection gate through `AnalysisPanel._regate`; this button used to take
    nothing, so after the reverse "Treat this channel as: voltage" override the Detect
    button greyed out with the override named while the banner above it kept a live
    "Re-run" whose click landed on `_run_detection`'s silent gate return -- no run, no
    status line, banner unchanged (sweep-7 fix-up D2, measured on the owner's -1000 mV
    ABF). And a stored run whose notch list the box would refuse (S7-NOTCH1's replay
    guard) was refused on every press with advice -- "fix the list on the Preprocess
    tab" -- that this button can never take, because Re-run replays the result's STORED
    dict, never the box (CW-1). `set_action_gate` is the one door the panel's resolver
    (`AnalysisPanel._stale_action_refusal`) pushes through: the button is disabled and
    the reason is printed under the message, the way the Detect button's gate label
    prints its own, so a refusal is on screen before any click and never only in a
    tooltip.
    """

    actionRequested = QtCore.Signal(str)   # one of _STALE_* below

    def __init__(self, on_action, parent=None) -> None:
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(
            "QFrame{background:#5a3d00; border:1px solid #7a5510; border-radius:4px;}"
            "QLabel{border:none; color:#ffd479;}")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 6, 6)
        outer.setSpacing(4)
        lay = QtWidgets.QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel()
        self._label.setWordWrap(True)
        lay.addWidget(self._label, 1)
        self._kind = _STALE_CALIBRATION
        self._btn = QtWidgets.QPushButton("Re-run")
        self._btn.clicked.connect(lambda: self.actionRequested.emit(self._kind))
        self.actionRequested.connect(on_action)
        lay.addWidget(self._btn)
        outer.addLayout(lay)
        # Why the offered action cannot run right now -- shown only while it cannot, in
        # the banner itself rather than a hover-only tooltip.
        self._why = QtWidgets.QLabel()
        self._why.setWordWrap(True)
        self._why.setVisible(False)
        outer.addWidget(self._why)

    def show_message(self, text: str, kind: str = _STALE_CALIBRATION,
                     action_label: str = "Re-run") -> None:
        self._label.setText(text)
        self._kind = str(kind)
        self._btn.setText(action_label)
        # A fresh message starts with the action offered; the panel pushes the gate's
        # real answer right after (`_judge_staleness` -> `_sync_stale_action`).
        self.set_action_gate(True)
        self.setVisible(True)

    def set_action_gate(self, ok: bool, why: str = "") -> None:
        """Enable the offered action, or disable it and say why -- under the message
        AND in the button's tooltip. ``why`` is the panel's sentence (the detection
        gate's reason, or the replay refusal); this only frames it with the action's
        own label so "Re-run is unavailable: ..." reads about the button beside it."""
        ok = bool(ok)
        self._btn.setEnabled(ok)
        if ok:
            self._btn.setToolTip(_STALE_ACTION_TIPS.get(self._kind, ""))
            self._why.clear()
            self._why.setVisible(False)
            return
        text = f"{self._btn.text()} is unavailable: {why}"
        self._btn.setToolTip(text)
        self._why.setText(text)
        self._why.setVisible(True)

    def action_refusal(self) -> str:
        """Why the action button is disabled, or ``""`` while it is offered (tests and
        the status line read this)."""
        return "" if self._btn.isEnabled() else self._why.text()

    @property
    def kind(self) -> str:
        """Which staleness the banner is currently reporting (tests read this)."""
        return self._kind

    def action_text(self) -> str:
        """The action button's label (tests read this)."""
        return self._btn.text()

    def message(self) -> str:
        return self._label.text()


class AnalysisPanel(QtWidgets.QWidget):
    """Detection params + run + results table + event navigation + staleness.

    Two signals FORWARD the owned :class:`EventTable`'s row selection and curation
    so MainWindow can keep the standalone Event Inspector in sync WITHOUT reaching
    into the table (the panel stays the owner of the table + overlays):

        eventSelected(int): a table row was selected (event_id).
        eventAcceptanceChanged(int, bool): an event's accepted state changed.
        eventFlagChanged(int, str): an event gained the curator's own flag.
    """

    eventSelected = QtCore.Signal(int)
    eventAcceptanceChanged = QtCore.Signal(int, bool)
    #: The 'f' key on the event table added ``FLAG_CURATED`` to a row. Carries the row's
    #: FULL new flag set, because the cell is a set of reasons and the interesting fact is
    #: what it now says, not what was added to it.
    eventFlagChanged = QtCore.Signal(int, str)
    #: A baseline re-measure / comparison refused, started, or finished. One token
    #: per outcome, so MainWindow (the sole wirer) can post ONE honest status line
    #: whether the job was launched from the menu or from the stale banner's button.
    #: Without it the banner's click could refuse in silence -- and a click that
    #: does nothing and says nothing is the failure mode this project has already
    #: ruled against for every gated button.
    baselineJobStatus = QtCore.Signal(str, str)   # token, human message
    #: A run this panel would have started, or an answer it received, that could not be
    #: used — and WHY, in one sentence for the status bar. MainWindow (the sole wirer)
    #: posts it. Two things travel here and both used to happen in total silence: a
    #: "Visible view" run whose target is not on screen, and a detection that came back
    #: for a load of the recording that no longer exists (it was closed and reopened while
    #: the detector ran). A discarded answer that says nothing is indistinguishable from a
    #: detector that found nothing.
    runRefused = QtCore.Signal(str)               # human message
    #: A 'Merge all' finished (or was refused / dropped) — carrying its
    #: :class:`MergeReport`. That merge runs OFF-THREAD (it costs 3.3-8.7 s on the
    #: owner's recordings), so its outcome cannot be the return value of the click;
    #: MainWindow, the sole wirer, posts this with the same ``_post_merge_status`` it
    #: uses for the two synchronous merge entry points.
    mergeFinished = QtCore.Signal(object)         # MergeReport
    #: The shown result was just re-judged for staleness -- the badge set on the events
    #: table (``_table.stale_columns``) and the banner were (re)written, whether to a
    #: stale state or back to clean. Carries nothing: the receiver reads the tuple and
    #: the banner's sentence off this panel, so the SAME set reaches every surface.
    #: MainWindow (the sole wirer) pushes them onto the Event Gallery and the Event
    #: Inspector (S7-POP1): those two windows repeated a badged depth with no mark, and
    #: the Preprocess-knob and calibration paths into `_check_staleness` never reached a
    #: surface refresh -- the badge could move while the gallery sat on top of the banner.
    stalenessChanged = QtCore.Signal()
    #: A detection run is ABOUT to be started that will replace a stored event table, and
    #: whatever curation is on that table has to be recoverable when it lands.
    #:
    #: The panel does not (and must not) know :class:`~nanoscope.core.history.EventHistory`
    #: exists — MainWindow is the sole wirer of undo, exactly as it is of everything else —
    #: so this is the seam: MainWindow connects it to its own ``_checkpoint_everything``.
    #: Why BEFORE rather than relying on the checkpoint the arriving result triggers:
    #: ``checkpoint`` pushes the state it last COMMITTED, so an unannounced in-place
    #: curation edit (the class of door ``checkpoint_all`` exists to catch) would otherwise
    #: be skipped over and Ctrl+Z would restore the table as it stood BEFORE that curation.
    #: Free when nothing has changed, by the same contract that lets four redundant signals
    #: drive a checkpoint.
    willReplaceEvents = QtCore.Signal()

    def __init__(self, session: Session, controller: AnalysisController,
                 trace_view: TraceView, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self._controller = controller
        self._trace_view = trace_view

        self._sel: tuple[str, int] | None = None
        #: The (runnable, reason) last pushed to the owned panels by `_regate`, so a
        #: calibration edit that leaves the gate where it was pushes nothing.
        self._gate_shown: tuple[bool, str] | None = None
        self._overlay_ids: list[str] = []      # span overlays for the current result
        self._marker_id: str | None = None      # the single selected-event marker
        #: event_id -> subpopulation id, or None. A VIEW of a finished report, installed by
        #: MainWindow (see set_subpop_labels); never read from, and never written to, the
        #: events frame.
        self._subpop_labels: dict[int, int] | None = None
        self._selected_id: int | None = None    # event_id under the marker (for edit re-zoom)
        self._pending: str | None = None        # in-flight controller job kind
        #: The (doc_id, channel) the in-flight job was LAUNCHED on. A surrogate study runs
        #: off-thread for many seconds, and the user is free to select another recording while
        #: it does; without this the answer lands on whatever happens to be selected when it
        #: returns.
        self._pending_sel: tuple[str, int] | None = None
        #: Revision + target + calibration fingerprint + units + complete baseline
        #: recipe signature captured when the compact
        #: Baseline Correction comparison was launched.  A baseline fit can take
        #: long enough for a user to select another recording, recalibrate it, or
        #: edit the baseline controls; its graph must then be marked stale rather
        #: than presented as the current answer.
        self._baseline_workspace_job: tuple[
            int, tuple[str, int], tuple[float, float], str, tuple[Any, ...]
        ] | None = None
        self._baseline_workspace_revision = 0
        self._baseline_workspace_syncing = False
        self._baseline_workspace_runtime_signature: tuple[tuple[str, str], ...] | None = None
        #: What an in-flight 'Merge all' will be checked against when it lands:
        #: ``(doc_id, channel, module_id, n_before, window_s, event_ids, geometry,
        #: curation)``. See :meth:`_finish_merge_all`.
        self._merge_job: tuple[Any, ...] | None = None
        #: Surrogate-study state, keyed by (doc_id, channel) — NOT panel-global.
        #: ``_recovery`` holds each target's ``recovery_map`` frame (the eta(t) a dwell fit
        #: divides by) and ``_fdr`` each target's ``null_control`` dict (the false-positive
        #: rate on the readout). Both were single slots keyed by nothing, so selecting
        #: another recording silently carried them over: B's dwell fit ran with A's eta(t)
        #: and printed "measured eta(t) on [...]" as if it had been measured on B, and the
        #: FDR label kept showing A's percentage under B's name.
        self._recovery: dict[tuple[str, int], Any] = {}
        self._fdr: dict[tuple[str, int], dict] = {}
        #: THE STANDING "ACCEPTED OVER … FAILED HEALTH CHECKS" MARK, keyed by
        #: ``(doc_id, channel, rough_s, base_s)`` -> the ``(proposal, proposal_health)``
        #: that was refused and confirmed anyway.
        #:
        #: Per-target like ``_fdr`` / ``_recovery`` above, and per-VALUE as well, because
        #: that is precisely what `BaselinePanel` says the mark is: it stands "for as long
        #: as the form holds these numbers". Its own two clears are written for the LEAVING
        #: direction only (``set_recording`` on a target change, ``set_params`` on a seed),
        #: and navigating away and back re-seeds the very same condemned windows out of the
        #: stored detection's params — so the form held the numbers while the sentence
        #: written to say "these depths are not certifiable" had reverted to the panel's
        #: most reassuring one. Keying on the WINDOWS rather than on the click means the
        #: mark comes back exactly when they do and never otherwise: an edit, a Reset or a
        #: clean re-proposal moves the form off those values and the mark stays down.
        #: Written by `_remember_baseline_mark`, replayed by `_restore_baseline_mark`.
        self._baseline_marks: dict[
            tuple[str, int, float, float], tuple[BaselineProposal, Any]] = {}
        #: ``(target, proposal, health)`` currently ON OFFER. At most one exists — a
        #: proposal belongs to the recording it was measured on and
        #: `BaselinePanel.set_recording` drops it on every target change — and it is kept
        #: so an acceptance is remembered with the PROPOSAL-SCOPE health that condemned
        #: it, not with whatever health block happens to be on screen when the user clicks
        #: (a Detect between the probe and the Apply replaces it with a detection-scope
        #: one, and the two are not interchangeable: see `BaselinePanel.set_health`).
        self._baseline_offer: tuple[tuple[str, int], BaselineProposal, Any] | None = None
        #: One-entry memo for :meth:`_active_signal`'s conditioned trace: (key, array).
        self._cond_cache: tuple[tuple[Any, ...], np.ndarray] | None = None
        #: ``(doc_id, channel, module_id)`` an in-flight re-measure was launched for.
        #: Keyed by the RESULT, not by the selection: a re-measure takes seconds and a
        #: table installed under whichever detection happens to be shown when it lands
        #: would silently replace a different recording's events with these.
        self._remeasure_target: tuple[str, int, str] | None = None
        #: Set by `_say_baseline`; `_on_remeasure_finished` uses it to guarantee
        #: exactly one terminal message per started job.
        self._baseline_said = False
        #: The last `EstimatorComparison` computed, for the status line and the tests.
        self._last_comparison: Any = None
        #: The curator-visible state of the table when an in-flight re-measure was
        #: dispatched, and the baseline params it is measuring with. Both are
        #: captured at dispatch because both can change under a job that takes
        #: seconds while the table stays live.
        self._remeasure_curation: Any = None
        self._remeasure_params: dict[str, Any] | None = None
        # Which detector's result is currently shown (Schmitt or Keras ML). Every
        # "fetch the current detection" read goes through this, so ML events use the
        # identical table/overlay/inspector/plot/export path as Schmitt events.
        self._active_detection: str = _MODULE_ID
        #: What the Detection tab's detector combo said the last time this panel looked.
        #: ``paramsChanged`` carries every knob on the tab, so this is how a change of
        #: DETECTOR is told from a change of threshold — see :meth:`_on_params_changed`.
        self._combo_detector: str = _MODULE_ID
        #: Re-entrancy guard for :meth:`_show_detector_choice`: showing the chosen
        #: detector's stored run re-seeds the form, which emits ``paramsChanged`` again.
        self._switching_detector = False
        #: Re-entrancy guard for the result selector above the shared event table. Unlike
        #: the detector form's combo (which can name only Schmitt/CUSUM), this selector
        #: names every stored EventSet, including ML and the derived reconciled table.
        self._switching_result = False
        #: The stored result the form was last SEEDED from. The form must agree with the
        #: table on screen (see :meth:`_seed_form_from`), and it already does whenever this
        #: is the result being shown — so a selection re-emitted for the same target and
        #: the same table has nothing to do, and doing it anyway is how a user's edits got
        #: thrown away (C24). A different object here means a genuinely different table:
        #: a project restore installing the saved run, a detector switch, a re-detection.
        self._seeded_from: AnalysisResult | None = None

        self._build_ui()
        self._wire_session()
        self._wire_trace_view()
        self._wire_controller()
        self._show_empty()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._title = _ElidedLabel("No channel selected")
        self._title.setStyleSheet("font-weight:600; font-size:13px;")
        root.addWidget(self._title)

        self._stale = _StaleBanner(self._on_stale_action)
        root.addWidget(self._stale)

        self._detection = DetectionPanel()
        #: What the Detection tab's "measured" polarity line reads before anything has been
        #: measured, taken from the widget itself instead of re-typed here so the two files
        #: cannot drift apart. It is what that line must read again the moment this panel is
        #: pointed at a recording with no measurement of its own — see
        #: :meth:`_show_polarity_evidence`, which is where both halves of that readout live.
        self._polarity_unmeasured = self._detection.polarity_readout.text()
        self._stats = StatsPanel()
        # Scores the DETECTOR against the user's curation of the current detection
        # (matched / spurious / missed + precision / recall / F1). Sits with the summary.
        self._score = ScorePanel()
        # Post-detection sub-event merge (a curation-style edit): its two buttons emit
        # intent only; MainWindow (the sole wirer) routes them onto the active detection
        # via this panel's merge_* engine methods. Gated on there being a detection.
        self._result_selector = QtWidgets.QComboBox()
        self._result_selector.setObjectName("eventResultSelector")
        self._result_selector.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._result_selector.setMinimumContentsLength(12)
        # The strip must still fit without scrolling at a 320 px Inspector. Its popup
        # retains the full item text and the tooltip explains the control, so the closed
        # box may elide instead of making its 150 px size hint a dock-width floor.
        self._result_selector.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self._result_selector.setToolTip(
            "Choose which stored detector result is shown below. Switching only changes "
            "the view; every source table and its curation stays stored."
        )
        self._merge = MergePanel(table_selector=self._result_selector)
        self._table = EventTable()
        # The manual event-measurement tool (cursor selection + live stats + saveable
        # table). AnalysisPanel OWNS it and hands it the SAME shared TraceView it got,
        # so the Measure region lands on the real panes; it talks only to Session +
        # that TraceView, never to the sibling panels above it.
        self._measure = MeasurePanel(self._session, self._trace_view)

        # The measure tool is the FOURTH mode tab, beside Preprocess / Detection / ML
        # detector: all four are ways of producing (or measuring) the events that land in
        # the ONE shared event table below, so they belong in one tab strip rather than
        # with the tool buried at the bottom of a scroll under the table. This panel still
        # owns the widget — DetectionPanel only hosts the page (it never wires it), and
        # every signal below is connected here exactly as before.
        #
        # It is wrapped in a scroll area first, and that is load-bearing, not cosmetic: its
        # un-wrappable button row ("Measure event (cursors)" + "Snap to view" + "Snap to
        # selected event") has a hard 438 px layout minimum, a tab widget's minimum width is
        # the MAX of its pages', and that number propagates straight into
        # QDockWidget::minimumSize() — the bound QDockAreaLayout refuses to drag a separator
        # past. Unwrapped, it pinned the Inspector at ~458 px and the signal viewer could
        # never be made wider. Wrapped, the panel keeps its natural width and a narrow dock
        # simply scrolls it horizontally.
        measure_scroll = _scroll_wrap(self._measure)
        self._detection.set_measure_widget(measure_scroll)

        # The open-pore baseline controls, owned here for exactly the reasons
        # MeasurePanel is: the widget must reach an off-thread job and this panel is
        # the only launcher of analysis jobs, while ``panels.py`` imports no sibling
        # gui module and must stay that way. DetectionPanel only HOSTS the page.
        self._baseline = BaselinePanel()
        self._baseline_workspace = BaselineCorrectionWorkspace(
            self._baseline,
            method_choices=self._detection.baseline_drift_method_choices(),
        )
        self._detection.set_baseline_widget(self._baseline_workspace)
        # Opening Analysis starts on the baseline workspace rather than leaving its
        # familiar overlay/residual controls hidden behind the Preprocess-section
        # switcher.  Signal-chain controls remain one explicit selection away.
        # This is especially important in a compact Inspector, where an invisible
        # overlay checkbox is indistinguishable from a feature that no longer exists.
        self._detection.show_open_pore_baseline()

        # Controls up top (scrollable so a small dock still reaches the buttons);
        # the event table gets the remaining vertical space and scrolls natively.
        controls = QtWidgets.QWidget()
        clay = QtWidgets.QVBoxLayout(controls)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.addWidget(self._detection)
        clay.addWidget(self._stats)
        clay.addWidget(self._score)
        controls_scroll = _scroll_wrap(controls)

        # THE MERGE CONTROLS DO NOT LIVE IN THAT COLUMN. Measured on the shipped default
        # layout — a fresh QSettings profile, MainWindow at the owner's 1512x900, Inspector
        # 460 px — the controls column is a 420x511 viewport onto 836x2508 of content, and
        # ``MergePanel`` was its LAST child: "Merge all" sat at x=745..825, y=2440..2465,
        # i.e. 1997 px below the fold AND 405 px right of it. Reaching the only batch merge
        # in the application meant dragging the horizontal scrollbar to 393/416 and the
        # vertical one to its end at the same time; nothing of it was on screen otherwise,
        # and there is no menu item or shortcut for it. That is not a merge panel a user
        # finds. (The width came from a sibling three tabs away — a preset combo sized to a
        # 115-character item stretched the whole column to 836 px — so no edit inside
        # ``merge_panel.py`` could have brought it back on screen either.)
        #
        # It goes with the EVENT TABLE instead, which is the thing it edits: a fixed strip
        # directly above the table, outside the deep scroll, at a screen position that does
        # not move. Measured after: 421x95 at y=0 of its own pane, "Merge all" fully inside
        # the viewport, both scrollbar maxima 0.
        #
        # THE WRAP IS LOAD-BEARING, for the same reason ``measure_scroll`` above is. Taken
        # out of the scrolled column, ``MergePanel``'s own 252 px layout minimum (an
        # un-shrinkable "Merge events in selection" button) would become a floor under this
        # panel, the Analysis page, the tab widget and finally ``QDockWidget::minimumSize``
        # — the bound ``QDockAreaLayout`` will not drag a separator past — and the Inspector
        # would stop shrinking below ~270 px. Wrapped, it demands nothing and simply scrolls
        # if the dock is dragged narrower than it.
        #
        # THE HEIGHT IS PINNED AND THE SCROLLBAR'S ROOM IS RESERVED. ``_pin_scroll_height``
        # gives the strip its natural height plus the height of a horizontal scrollbar that
        # is not there yet, as ONE quantity: it is both the area's maximum and (via a
        # trailing spacer of exactly that height, appended to this layout) the sizeHint the
        # pane's QVBoxLayout grants. So dragging the dock narrow enough to need that
        # scrollbar spends the reserved space instead of re-flowing the group — a button
        # never moves out from under the pointer. There is deliberately no ``addStretch``
        # here: a stretch contributes 0 to the sizeHint, which is exactly how the room came
        # to be reserved on paper and never granted (see the helper's docstring).
        merge_host = QtWidgets.QWidget()
        mlay = QtWidgets.QVBoxLayout(merge_host)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.setSpacing(0)
        mlay.addWidget(self._merge)
        merge_scroll = _scroll_wrap(merge_host)
        _pin_scroll_height(merge_scroll, merge_host)

        # The event table STAYS below the tabs: it is the shared result surface every
        # detector — Schmitt, ML, and a hand-marked missed event — feeds. The merge strip
        # rides with it, above it, in the same splitter pane.
        results = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(results)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(6)
        rlay.addWidget(merge_scroll)
        rlay.addWidget(self._table, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(controls_scroll)
        split.addWidget(results)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        # -- panel signals ---------------------------------------------------------------------
        self._detection.detectRequested.connect(self._on_detect_requested)
        self._detection.sweepRequested.connect(self._on_sweep_requested)
        self._detection.nullControlRequested.connect(self._on_null_requested)
        self._detection.recoveryRequested.connect(self._on_recovery_requested)
        self._detection.paramsChanged.connect(self._on_params_changed)
        self._detection.previewFilterToggled.connect(self._on_preview_toggled)
        self._detection.mlDetectRequested.connect(self._on_ml_detect_requested)
        self._result_selector.currentIndexChanged.connect(self._show_result_choice)
        # The notch-line flood guard (see _ML_NOTCH_MAX_LINES). The box itself lives on the
        # DetectionPanel, but the DetectionPanel deliberately wires nothing and launches
        # nothing — this panel is the single launcher of analysis jobs, so the bound on what
        # a config may commit a run to is enforced here, at the same seam that starts the
        # runs. Two layers, both honest: the commit-time guard below refuses an over-cap
        # EDIT the moment it is committed (editingFinished — the panel's own
        # commit-on-finish idiom) and restores the box, and _launch_ml refuses an over-cap
        # RUN however its params arrived (a stored result replayed by the stale banner's
        # Re-run, a programmatic caller), so no path around the box exists.
        #: The last committed "notch lines" text that was UNDER the cap — what an over-cap
        #: commit is rolled back to, so a refusal never leaves the box holding a config
        #: the refusal just said may not run (two surfaces disagreeing). Seeded from the
        #: widget's initial (preset) text; tracked on textChanged so preset loads and
        #: set_ml_params replays keep it current without needing a commit of their own.
        self._ml_notch_ok_text: str = self._detection.ml_notch_freqs.text()
        self._ml_notch_reverting = False   # re-entrancy guard: the rollback below re-commits
        self._detection.ml_notch_freqs.textChanged.connect(self._on_ml_notch_text_changed)
        self._detection.ml_notch_freqs.editingFinished.connect(self._on_ml_notch_committed)
        # The SIBLING box on the Preprocess tab (S7-NOTCH1). The DetectionPanel refuses an
        # over-cap or illegal list itself (its `_commit_notch_lines`: rollback + the orange
        # note under the box) because it owns that widget; what it cannot do is reach the
        # status bar, and the ML box's refusal does (through ``runRefused``, above). Same
        # sentence on both surfaces. The latch on ``textEdited`` -- user edits only, never
        # the panel's own rollback ``setText`` -- is what makes the forward fire once per
        # refused EDIT: a Tab out of the untouched, rolled-back box re-commits legally and
        # keeps the standing note (by the DetectionPanel's design) but must not re-post it.
        self._notch_lines_edited = False
        self._detection.notch_freqs.textEdited.connect(self._on_notch_lines_edited)
        self._detection.notch_freqs.editingFinished.connect(self._on_notch_lines_committed)
        self._stats.fitRequested.connect(self._on_fit_requested)
        self._table.eventSelected.connect(self._on_event_selected)
        self._table.eventSelected.connect(self.eventSelected)   # forward for the Event Inspector
        # A curation edit (table checkbox, table keyboard, or a routed Event Inspector decision on
        # the shared df) refreshes the summary AND the span overlays, then forwards for the inspector.
        self._table.acceptanceChanged.connect(self._on_acceptance_changed)
        self._table.acceptanceChanged.connect(self.eventAcceptanceChanged)
        # The 'f' key writes the curator's own token into the shared frame. It is a
        # mutation of the event table exactly as an accept/reject is, so it announces
        # itself the same way — otherwise the only thing that would notice is the
        # history's next comparison, and the dirty flag would wait for it.
        self._table.flagChanged.connect(self.eventFlagChanged)
        # The measure tool's "Mark selection as missed event" — the owner reads the
        # cursor bounds and appends the characterized span to the active detection.
        self._measure.markMissedRequested.connect(self.add_current_selection_as_event)
        # The baseline controls emit intent only; this panel runs the probe off-thread
        # (the whole job -- conditioning + probe + health pass -- measured 2 s on the
        # owner's 3 M-sample recording and 6 s on 9 M samples; see _on_probe_requested)
        # and pushes the proposal back. Nothing is applied without the user's
        # confirming click.
        self._baseline.probeRequested.connect(self._on_probe_requested)
        # ...and a change to those controls re-judges the detection on screen at
        # once. Without this the estimator can be replaced while a table measured by
        # the old one is displayed, with nothing saying so until the next selection
        # change -- the silent 15x the whole migration exists to make impossible.
        self._baseline.paramsChanged.connect(self._on_baseline_params_changed)
        # ...and the two VIEW boxes, which ask for a drawing rather than a measurement.
        # Both land on one slot because the two compose into ONE picture (a subtracted
        # trace with its own rail on it is a flat line at zero), and the panel emits both
        # on either box for exactly that reason.
        self._baseline.showBaselineToggled.connect(self._on_baseline_view_changed)
        self._baseline.subtractBaselineToggled.connect(self._on_baseline_view_changed)
        # The workspace is a presentation/co-ordination layer only.  It emits a
        # user's intent; the legacy BaselinePanel remains the owner of I0 settings,
        # DetectionPanel remains the owner of the runtime detrend parameters, and
        # this AnalysisPanel remains the only launcher of a background calculation.
        self._baseline_workspace.refreshRequested.connect(self._on_baseline_workspace_refresh)
        self._baseline_workspace.exportRequested.connect(self._on_baseline_workspace_export)
        self._baseline_workspace.advancedRequested.connect(
            self._detection.show_baseline_drift)
        self._baseline_workspace.runtimeEnabledRequested.connect(
            self._on_baseline_workspace_runtime_enabled)
        self._baseline_workspace.runtimeMethodRequested.connect(
            self._on_baseline_workspace_runtime_method)
        self._baseline_workspace.modeChanged.connect(self._on_baseline_workspace_mode_changed)
        self._sync_baseline_workspace()

    def _wire_session(self) -> None:
        self._session.resultsChanged.connect(self._on_results_changed)
        self._session.calibrationChanged.connect(self._on_calibration_changed)
        self._session.documentClosed.connect(self._on_document_closed)

    def _wire_trace_view(self) -> None:
        """The plot tells this panel what became of the preview curve it asked for.

        The panel owns the REQUEST (the checkbox) and the error line; the view owns the
        CURVE, and it is the view that destroys and re-applies the curve on gestures this
        panel never hears — View ▸ Stacked/Overlay and every channel show/hide both land
        on `TraceView.rebuild`. So the one fact the two must agree on ("is the processed
        trace actually on the plot?") travels on one signal, and this is the only place it
        is subscribed to.
        """
        self._trace_view.sigFilterPreviewChanged.connect(self._on_preview_state_changed)

    def _wire_controller(self) -> None:
        # The controller is shared; only THIS panel launches analysis jobs, so every
        # terminal signal here is ours. failed/cancelled are surfaced by MainWindow's
        # status chrome; the panel just resets its own pending state + handles results.
        self._controller.finished.connect(self._on_controller_finished)
        self._controller.failed.connect(self._on_controller_failed)
        self._controller.cancelled.connect(self._on_controller_cancelled)

    # -- selection -------------------------------------------------------------------------------
    def set_selection(self, doc_id: str, channel: int) -> None:
        """Pushed in by MainWindow. Repoints the panel at (doc_id, channel), gates the
        detector honestly, and loads any stored result for it.

        Detection is inherently a CURRENT-channel operation, so the panel STICKS to the document's
        current channel: selecting the voltage channel (e.g. to calibrate it) must not abandon the
        analyte channel and grey out Detect. Only if the user explicitly selects a current channel do
        we honour that exact channel (supports a rig with more than one).

        RE-POINTING AT THE TARGET ALREADY SHOWN IS NAVIGATION, AND NAVIGATION COSTS NOTHING.
        This method drops overlays, clears the marker and — through
        :meth:`_load_stored_detection` — RE-SEEDS the whole form from the stored run. All of
        that is right when the target changes and wrong when it does not, because the second
        half silently discards whatever the user has dialled in since. Measured: edit ``k_in``
        5.0 -> 7.0, then click the VOLTAGE row of the same recording (which resolves back to
        this same current channel, deliberately) and ``k_in`` was 5.0 again, the whole
        preprocessing chain re-applied, with no prompt and nothing in the status bar.

        The identity used is ``doc_id``, which is per-LOAD (see :mod:`nanoscope.core.identity`),
        so closing a recording and reopening the same file is correctly a different target and
        does re-seed. ``MeasurePanel.set_selection`` and ``BaselinePanel.set_recording`` already
        held their own halves of this rule; this is the same rule at the panel that owns them.
        """
        if not self._session.has_document(doc_id):
            self._show_empty()
            return
        meta = self._session.document(doc_id).meta
        channel = self._resolve_analysis_channel(doc_id, channel)
        if self._sel == (doc_id, channel):
            # Same target. Re-gate (a calibration or a rate could have changed under us)
            # and let the owned panels decide for themselves — MeasurePanel and
            # BaselinePanel both no-op on an unchanged target — but touch neither the
            # form nor the overlays.
            self._regate()
            self._measure.set_selection(doc_id, channel)
            self._push_measure_conditioning()
            self._sync_surrogate_readouts()
            self._sync_baseline_workspace()
            # ...unless the TABLE changed underneath the unchanged target. A project
            # restore installs the saved run into a session already pointed here, and the
            # form has to follow it or the user is shown app defaults over somebody else's
            # numbers — which is the entire reason `_seed_form_from` exists. "Same target"
            # is not "same table"; only the second one means there is nothing to do.
            if self._stored_to_show(doc_id, channel) is not self._seeded_from:
                self._load_stored_detection(doc_id, channel)
            return

        # Leaving the previous selection: drop its overlays + any live preview.
        self._clear_overlays()
        self._clear_marker()
        self._selected_id = None
        self._trace_view.clear_filter_preview()
        self._baseline_workspace.clear_preview(
            "Recording changed — refresh the comparison for this trace.")
        # ...and the preview's error line with them. The message names the PREVIOUS
        # recording's numbers ("window_s=60s is longer than the recording (0.5s)"),
        # which is simply false about the recording now arriving — and the preview
        # toggle is dropped a few lines down, so there is no live request left for
        # the message to describe.
        self._detection.set_preview_error("")

        self._sel = (doc_id, channel)
        # The measure tool follows the SAME resolved CURRENT channel; a target change
        # clears its cursor region (it belonged to the previous channel's pane).
        self._measure.set_selection(doc_id, channel)
        # ...and the chain the form is asking for, which a target change does not carry:
        # a restored Preprocess form may never emit `paramsChanged` at all.
        self._push_measure_conditioning()
        fs = meta.sample_rate_hz

        # Gate: detection runs on CURRENT channels with a known sample rate. The
        # Schmitt buttons and the ML detector share this gate (ML also needs a model
        # loaded + TensorFlow, which the panel ANDs in itself). ``_regate`` also writes
        # the title -- what the channel is TREATED AS, and what the file says when the
        # two differ -- from the same resolver the gate reads.
        self._detection.set_recording(fs)
        runnable, reason = self._regate()
        # ``set_recording`` drops any proposal from the previous file. ``target`` as well
        # as the rate: it is what ends a CONFIRMED per-recording configuration (see
        # `BaselinePanel.set_recording`). Re-selecting the same channel is navigation and
        # must cost the user nothing; moving to another recording must not carry his
        # 2.0 / 0.5 s over onto it. The gate itself was set by ``_regate`` above; it is
        # re-asserted here because ``set_recording`` is the call that resets the panel.
        self._clear_baseline_notice()
        self._baseline.set_recording(fs, (doc_id, channel))
        self._baseline.set_gate(runnable, reason)
        # The two view boxes are a STANDING preference and the trace under them just
        # changed: re-apply them to the new selection, and (through the clear inside)
        # take the previous recording's rail down. Without this a rail ticked on the
        # -600 mV run stayed up over the -1000 mV one, claimed by nothing on screen.
        self._apply_baseline_views()
        self._sync_baseline_workspace()

        # Reflect the preview toggle onto the new pane (usually off after a switch).
        if self._detection.filter_preview.isChecked():
            self._detection.filter_preview.setChecked(False)

        self._sync_surrogate_readouts()
        self._load_stored_detection(doc_id, channel)

    def _sync_surrogate_readouts(self) -> None:
        """Point the FDR label and the 'use measured efficiency' toggle at the CURRENT target.

        Both readouts describe a surrogate study run on one (doc_id, channel); neither is a
        property of the panel. Repointing the panel therefore has to repoint them — showing
        the new target's own numbers, or none at all — rather than leave the previous
        recording's on screen where they read as this one's."""
        self._stats.set_fdr(self._fdr.get(self._sel) if self._sel is not None else None)
        self._stats.set_recovery_available(
            self._sel is not None and self._sel in self._recovery)

    def _load_stored_detection(self, doc_id: str, channel: int) -> None:
        """Show the stored detection for this selection, if any. Prefers the
        previously-active detector, then Schmitt, then ML, so a selection round-trip
        keeps showing the same detector's events (and updates ``_active_detection``).

        Repointing at a target also RE-SEEDS the detection form from that target's stored
        Schmitt run, so the settings on the tab are the settings that produced the table on
        screen. Without it a reopened project showed the panel's defaults over restored
        events — and Detect / Re-run then quietly detected something else."""
        res = self._stored_to_show(doc_id, channel)
        if res is not None:
            self._set_active_detection(res.module_id)
            self._seeded_from = res
            self._seed_form_from(res)
            self._show_result(res)
            return
        self._seeded_from = None
        self._clear_results()

    def _set_active_detection(self, module_id: str) -> None:
        """THE ONE WRITER of ``_active_detection`` — and the one place the detector combo
        is made to agree with the table on screen.

        THE MEASURED DEFECT: the combo lied. Detect with Schmitt (115 events), switch the
        combo to CUSUM and run it (213 events), then set the combo back to "Schmitt trigger
        (hysteresis)" — and ``_active_detection`` was still ``detect.cusum``. The table kept
        showing CUSUM's 213 rows under a combo reading Schmitt, and a channel round-trip did
        not fix it because :meth:`_stored_to_show` prefers ``_active_detection``. The stored
        Schmitt run was genuinely stranded: present in memory, invisible, and reachable only
        by pressing Run — the one action that destroys it.

        Two facts must not drift apart, so they are written together: what
        ``_stored_detection`` fetches, and what the combo names. The combo covers only the
        two threshold detectors; an ML result keeps its own tab and leaves the combo alone
        (a combo cannot name ``detect.keras``, so writing to it would be the same lie in
        the other direction).
        """
        self._active_detection = module_id
        if module_id in (_MODULE_ID, _CUSUM_MODULE_ID):
            self._combo_detector = module_id
            self._detection.set_detector(module_id)

    def _available_detection_results(self) -> tuple[AnalysisResult, ...]:
        """Every stored EventSet for the resolved selection, in the shared source order."""
        if self._sel is None or not self._session.has_document(self._sel[0]):
            return ()
        doc_id, channel = self._sel
        out: list[AnalysisResult] = []
        for module_id in _DETECTION_MODULE_IDS:
            res = self._session.results.get(doc_id, channel, module_id)
            if res is not None and isinstance(res.payload, EventSet):
                out.append(res)
        return tuple(out)

    def _sync_result_selector(self) -> None:
        """Make the table selector name what is actually shown and everything recoverable.

        The threshold-detector combo owns runnable detector parameters and therefore cannot
        name an ML table or the reconciled derived table.  This combo owns only display.
        Its entries are rebuilt from ResultStore, so switching away never strands a result.
        """
        results = self._available_detection_results()
        active_present = any(r.module_id == self._active_detection for r in results)
        self._result_selector.blockSignals(True)
        try:
            self._result_selector.clear()
            if not active_present and self._sel is not None:
                self._result_selector.addItem(
                    f"{detector_name(self._active_detection)} — no stored result", None)
            for res in results:
                n = len(res.payload.df)
                suffix = "candidate" if n == 1 else "candidates"
                self._result_selector.addItem(
                    f"{detector_name(res.module_id)} — {n:,} {suffix}", res.module_id)
            if not results and self._sel is None:
                self._result_selector.addItem("No event table", None)
            idx = self._result_selector.findData(self._active_detection)
            self._result_selector.setCurrentIndex(idx if idx >= 0 else 0)
            self._result_selector.setEnabled(bool(results))
        finally:
            self._result_selector.blockSignals(False)

    @QtCore.Slot(int)
    def _show_result_choice(self, index: int) -> None:
        """Show one stored result selected above the table; mutate no result."""
        if self._sel is None or self._switching_result:
            return
        module_id = self._result_selector.itemData(index)
        if not isinstance(module_id, str) or not module_id:
            return
        doc_id, channel = self._sel
        res = self._session.results.get(doc_id, channel, module_id)
        if res is None or not isinstance(res.payload, EventSet):
            self._sync_result_selector()
            return
        if module_id == self._active_detection and res is self._seeded_from:
            return
        self._switching_result = True
        try:
            self._set_active_detection(module_id)
            self._seeded_from = res
            self._seed_form_from(res)
            self._show_result(res)
        finally:
            self._switching_result = False

    def _show_detector_choice(self, module_id: str) -> None:
        """The user moved the detector combo: show THAT detector's stored run, or nothing.

        This is the door the audit found missing. With two detectors on one channel the
        only way back to the first one's table used to be Run — which replaces it, with
        every hand rejection on it (see :meth:`confirm_replacing_curation`). Both runs are
        already stored: ``ResultStore`` is keyed ``(doc_id, channel, module_id)`` and holds
        them side by side. Selecting a detector now simply SHOWS its table.

        A detector that has not run here shows nothing, through the same
        :meth:`_clear_results` every other "this target has no result" path uses. Nothing is
        destroyed either way — the other detector's run stays in the store and comes back
        the moment the combo does.

        Left alone when the table on screen belongs to the ML detector: that result lives on
        its own tab and this combo does not name it.
        """
        if self._sel is None or self._switching_detector or self._switching_result:
            return
        if self._active_detection not in (_MODULE_ID, _CUSUM_MODULE_ID):
            return
        if module_id == self._active_detection:
            return
        doc_id, channel = self._sel
        res = self._session.results.get(doc_id, channel, module_id)
        self._switching_detector = True
        try:
            self._active_detection = module_id
            if res is not None:
                self._seeded_from = res
                self._seed_form_from(res)
                self._show_result(res)
            else:
                self._seeded_from = None
                self._clear_results()
        finally:
            self._switching_detector = False

    def _stored_to_show(self, doc_id: str, channel: int) -> AnalysisResult | None:
        """WHICH stored detection this target shows: the previously-active detector, then
        Schmitt, then ML. The one place that order is written down — :meth:`set_selection`
        asks it to decide whether anything has changed, and :meth:`_load_stored_detection`
        asks it to decide what to display, and the two must never disagree."""
        for mid in dict.fromkeys((self._active_detection, *_DETECTION_MODULE_IDS)):
            res = self._session.results.get(doc_id, channel, mid)
            if res is not None:
                return res
        return None

    def _seed_form_from(self, res: AnalysisResult) -> None:
        """Put a stored detection's settings back into the form THAT DETECTOR uses.

        The rule is one sentence and it is the same for both detectors: **the settings on
        the tab are the settings that produced the table on screen.** Otherwise Run
        re-detects something else and replaces the table — with every hand curation on it —
        without the user having changed anything.

        Each detector seeds its OWN tab. The two do not share a form and never write over
        each other: a Schmitt result leaves the ML tab alone and an ML result leaves the
        Schmitt tab alone, so switching detectors on one channel is lossless in both
        directions.

        This used to return early for anything that was not Schmitt, justified by "an ML
        result's params share no keys with this form". That was true of the Schmitt form
        and irrelevant: the ML tab is a form too, and it was the one being left stale.
        Measured: a result stored under ``notebook3`` / ``enter=0.9`` came back onto a panel
        reading ``notebook1`` / ``enter=0.5`` / ``smoothing=9`` — verbatim the failure this
        method exists to prevent, on the detector whose preprocessing chain IS its contract
        with the trained model.

        Silent when the settings are unknowable (a pre-v3 project whose provenance recorded
        nothing) — the form then keeps showing values that at least came from the user."""
        # Both threshold detectors seed the same form: their params share every key the
        # form holds except the two CUSUM thresholds, and `schmitt_params_from_result`
        # reads only what the result actually recorded.
        if res.module_id in (_MODULE_ID, _CUSUM_MODULE_ID):
            params = schmitt_params_from_result(res)
            if params:
                self._detection.set_params(params)
                # ...and the baseline windows with them. Without this the form showed the
                # app default over a table measured with something else -- the same defect
                # the Schmitt seeding above exists to fix, on the parameter that decides
                # every depth in that table.
                self._baseline.set_params(params)
        elif res.module_id == _ML_MODULE_ID:
            # The model PATH is deliberately not restored: the file it names may be gone,
            # renamed or retrained, and a panel claiming a model is loaded when it is not
            # would arm a Run button that cannot run. The pipeline is what a stored run
            # can honestly put back, and it is what decides the numbers.
            self._detection.set_ml_params(dict(res.params or {}))

    def _stored_detection(self) -> AnalysisResult | None:
        """The currently-active detection result for the selection (Schmitt or ML), or
        None. The single 'fetch the current detection' accessor -- every consumer
        (summary, spans, navigation, staleness, fits) reads through it, so switching
        detectors is just a change of ``_active_detection``."""
        if self._sel is None:
            return None
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return None
        return self._session.results.get(doc_id, channel, self._active_detection)

    def can_run(self) -> bool:
        """Whether detection can run on the panel's CURRENT (resolved) target — the single source of
        truth for both the panel's Detect button and MainWindow's Run-detection menu action, so the
        two never disagree (the menu must not stay grey while the button is live, or vice-versa).

        DELIBERATELY BLIND TO WHETHER SOMETHING ELSE IS RUNNING. A batch pass in flight
        does not grey this button out, and that is a decision rather than an omission: Qt
        never delivers a click to a disabled widget, so a Detect greyed out by a batch
        could not tell the user it was the batch — the user would be left with a dead
        button and a guess. The click is allowed to land and the run gate answers it by
        name (:mod:`nanoscope.gui.worker_thread`). What this method reports is the
        PERMANENT gate — is this even a current channel, does it have a rate — which is a
        property of the target the user cannot resolve by waiting."""
        if self._sel is None:
            return False
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return False
        runnable, _ = self._gate_for(doc_id, channel)
        return runnable

    def can_run_ml(self) -> bool:
        """Whether ML detection can run: the Schmitt channel gate (CURRENT channel,
        known rate) AND TensorFlow is importable AND a model is loaded. The single
        source of truth for both the panel's Run-ML button and MainWindow's
        Run-ML-detection menu action, so the two never disagree."""
        if not self.can_run():
            return False
        if not ml_detect.TENSORFLOW_AVAILABLE:
            return False
        return self._detection.model_path() is not None

    def trigger_ml_detection(self) -> None:
        """Public entry point for MainWindow's Analysis ▸ Run ML detection action.
        Runs with the current ML form parameters (gating is enforced downstream)."""
        self._on_ml_detect_requested(self._detection.ml_params())

    def load_ml_model(self) -> None:
        """Public entry point for MainWindow's Analysis ▸ Load ML model action —
        opens the panel's own file dialog so the model state lives in one place."""
        self._detection.browse_model()

    def _resolve_analysis_channel(self, doc_id: str, channel: int) -> int:
        """Map the selected channel to the one detection should run on -- through the SESSION.

        Keep the selection if the session says it is a live current channel (its EFFECTIVE
        kind: the user's Calibration-panel override, else the header), else the
        recording's first live current channel so analysis stays put when the user clicks
        voltage, else the selection itself so the gate can explain honestly why detection
        is off. The policy is :meth:`Session.resolve_current_channel` -- the same one the
        workflow strip and the calibration chip resolve with -- and it is deliberately not
        re-derived here.

        Until sweep 7 this read ``meta.channels[ch].kind``, the header, while Batch detect
        one menu item over read the override (S7-CAL1). Measured on a 2 s CSV of the
        owner's own -700 mV recording with neutral column headers (both channels
        ``unknown``), after "Treat this channel as: current" + pA: Run detection stayed
        disabled with "Detection runs on CURRENT channels; this is a unknown channel." --
        quoting the header the user had just corrected -- while Batch detect ran on that
        exact channel and stored 15 events. The reverse override (IN15, header current,
        marked VOLTAGE on the -1000 mV ABF) left Run detection ENABLED on a channel the
        user had just called a command voltage. One resolver ends both.
        """
        resolved = self._session.resolve_current_channel(doc_id, channel)
        return channel if resolved is None else resolved

    def _gate_for(self, doc_id: str, channel: int) -> tuple[bool, str]:
        """The detection gate for ONE target, from the session's effective kind -- THE call
        every gate in this panel goes through (`set_selection`, `can_run`, `_run_detection`,
        `_regate`), so no path can read the header again. The header kind rides along so a
        refusal caused by the user's own override says so."""
        meta = self._session.document(doc_id).meta
        return self._gate(self._session.effective_kind(doc_id, channel),
                          meta.sample_rate_hz, header_kind=meta.channels[channel].kind)

    @staticmethod
    def _gate(kind: ChannelKind, fs: float | None, *,
              header_kind: ChannelKind | None = None) -> tuple[bool, str]:
        """Runnable, and if not, why -- in the user's own terms.

        ``kind`` is what the channel is TREATED AS (the session's effective kind);
        ``header_kind`` is what the file declared. When they differ, the refusal must
        name the override rather than the header: "this is a voltage channel" over IN15,
        whose header the user can see says current, reads as the app contradicting the
        file, when it is honouring the user's own Calibration-panel choice.
        """
        if kind is not ChannelKind.CURRENT:
            why = (f"{kind.value} channel" if header_kind in (None, kind) else
                   f"{kind.value} channel (your Calibration-panel override; the file says "
                   f"{header_kind.value})")
            return False, f"Detection runs on CURRENT channels; this is a {why}."
        if fs is None:
            return False, ("Sample rate unknown — dwell time is undefined without a "
                           "time base, so detection is disabled for this recording.")
        return True, ""

    def _kind_label(self, doc_id: str, channel: int) -> str:
        """What the channel is treated as, for the title -- and what the file says when the
        user's override differs from it: ``current (file says unknown)``. The same shape the
        Calibration panel's title took in sweep 7, so the two panels read alike."""
        header = self._session.document(doc_id).meta.channels[channel].kind
        kind = self._session.effective_kind(doc_id, channel)
        if kind is header:
            return kind.value
        return f"{kind.value} (file says {header.value})"

    def _regate(self, *, reassert: bool = True) -> tuple[bool, str]:
        """Re-judge the current target: the gate on the Schmitt buttons, the ML button and
        the baseline probe, and the title -- all from the session's effective kind, in one
        place. Called when the target is (re)pointed and when a calibration edit lands on
        it, because a "Treat this channel as" override IS a calibration edit and it moves
        the gate: without this the panel's Detect button kept the answer from the last
        repoint while the Analysis menu (which asks `can_run` afresh) had moved on -- two
        surfaces disagreeing about one fact. Returns the gate so `set_selection` can
        thread it on.

        ``reassert=False`` (the calibration-edit path) pushes the gate to the owned panels
        only when the ANSWER moved: ``BaselinePanel.set_gate(ok=True)`` hides that panel's
        gate label, which is also where a REFUSED-proposal notice is posted, and a gain
        edit is no reason to take a standing refusal off the screen. A repoint always
        reasserts, exactly as it did before this method existed.
        """
        assert self._sel is not None
        doc_id, channel = self._sel
        meta = self._session.document(doc_id).meta
        info = meta.channels[channel]
        self._title.setText(
            f"{meta.path.name}  ·  {info.name}  ·  {self._kind_label(doc_id, channel)}"
            + (f"  ·  {info.raw_units}" if info.raw_units else ""))
        runnable, reason = self._gate_for(doc_id, channel)
        if reassert or (runnable, reason) != self._gate_shown:
            self._detection.set_runnable(runnable, reason)
            self._detection.set_ml_runnable(runnable, reason)
            # The baseline probe rides the SAME gate as detection (a CURRENT channel with
            # a known rate), so the two can never disagree about whether this recording
            # can be measured.
            self._baseline.set_gate(runnable, reason)
        self._gate_shown = (runnable, reason)
        # ...and the stale banner's button, which is the one other door into a run on
        # this target. Always, not only when the answer moved: the banner's own gate also
        # depends on the stored result (its replayable notch list), and this is cheap.
        self._sync_stale_action()
        return runnable, reason

    def _rerun_params(self, res: AnalysisResult | None) -> dict[str, Any]:
        """The Schmitt dict the stale banner's Re-run replays: the RESULT's stored settings,
        or the live form only when the run cannot be known (see `schmitt_params_from_result`).
        One spelling, read by the launcher (`_on_rerun`) and by the gate that decides whether
        the button may be offered (`_stale_action_refusal`), so the two judge one dict."""
        return schmitt_params_from_result(res) or self._detection.params()

    def _stale_action_refusal(self, res: AnalysisResult | None) -> str | None:
        """Why the stale banner's offered action could not run now, or ``None`` if it can.

        THE ONE RESOLVER for that button. Two answers are folded in, in the order the
        launchers apply them:

        * the detection gate (`_gate_for`) -- both actions run on the selected channel,
          and a re-measure is a baseline job, which rides the same gate as the probe;
          so a channel the user's override has made a voltage refuses Re-run and
          Re-measure in the gate's own words (D2);
        * for Re-run, whether the dict it would replay may run at all: an ML result
          with no model path cannot be run again, and a Schmitt result whose stored
          notch list the Preprocess box would refuse is refused here BEFORE any click,
          with the sentence naming the door that does run the box (CW-1) -- the
          replay-time refusal in `_run_detection` stays as the belt-and-braces for a
          programmatic caller.
        """
        if res is None or self._sel is None:
            return None
        # A reconciled table is a derived collection of complete canonical source rows,
        # not one detector run whose parameters can be replayed.  In particular, falling
        # through the ordinary non-ML branch below would silently run Schmitt and replace
        # the derived table with an unrelated detector result.  Keep the refusal ahead of
        # every ordinary run/baseline gate so the banner never offers the wrong repair.
        if res.module_id == CONSENSUS_MODULE_ID:
            return (
                _RECONCILED_SOURCE_CHANGE_REFUSAL
                if self._stale.kind == _STALE_RECONCILIATION_SOURCE
                else _RECONCILED_STALE_REFUSAL
            )
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return None
        runnable, reason = self._gate_for(doc_id, channel)
        if not runnable:
            return reason
        if self._stale.kind != _STALE_CALIBRATION:
            return None
        if self._active_detection == _ML_MODULE_ID:
            if not res.params.get("model_path"):
                return ("this ML result records no model path, so the same model cannot "
                        "be run again; load a model on the Detection tab's ML page and "
                        "press Run.")
            return None
        return self._notch_replay_why(self._rerun_params(res))

    def _sync_stale_action(self, res: AnalysisResult | None = None) -> None:
        """Push the resolver's answer onto the banner's button. ``res`` is the result
        being judged (staleness paths); ``None`` reads the stored one (the gate paths)."""
        if res is None:
            res = self._stored_detection()
        why = self._stale_action_refusal(res)
        self._stale.set_action_gate(why is None, why or "")

    @property
    def measure_panel(self) -> MeasurePanel:
        """The owned manual-measurement panel. Exposed so MainWindow can wire its
        ``Analysis ▸ Measure event`` menu toggle to the same widget — the panel stays
        the owner; MainWindow only drives it, exactly as it does with detection."""
        return self._measure

    @property
    def detection_panel(self) -> DetectionPanel:
        """The owned DetectionPanel. Exposed (read-only) so MainWindow's workflow-rail
        reveal can raise the panel's inner Preprocess / Detection tab — the panel stays
        the sole owner of the detection form; MainWindow only drives which tab shows."""
        return self._detection

    def current_detection(self) -> AnalysisResult | None:
        """The stored detection :class:`AnalysisResult` for the current selection, or None.

        Public accessor so MainWindow's *Export figure…* action can pull the current
        analysis's events without reaching into this panel's private state — the panel
        stays the single owner of the results/selection. Returns whichever detector's
        result is active (Schmitt or ML), so exports/plots/inspector follow it.
        """
        return self._stored_detection()

    def current_target(self) -> tuple[str, int] | None:
        """The resolved ``(document, current-channel)`` this panel is showing."""
        return self._sel

    def current_events(self) -> EventSet | None:
        """The current detection's :class:`EventSet` — frame AND provenance — or None.

        Public accessor so MainWindow can push the detector's provenance
        (``min_dwell_s``, ``max_dwell_s``, ``filter_rise_s``, ``threshold_out_pA``)
        into the Plots window alongside the frame. Without it that window has to
        guess the detection floor, and a guessed floor is a biased tau.
        """
        res = self.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return None
        return res.payload

    def current_events_df(self):
        """The current detection's per-event ``DataFrame`` (the FULL, uncurated
        frame), or None when there is no detection for the selection.

        Public accessor so MainWindow's *Plots…* action can hand the whole event
        table to the standalone :class:`~nanoscope.gui.plots_panel.PlotsWindow`
        without reaching into this panel's private state — the panel stays the
        single owner of the results/selection. The window itself does the
        accepted-only filtering, so this deliberately returns the full frame.
        """
        res = self.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return None
        return res.payload.df

    # -- run -------------------------------------------------------------------------------------
    def detection_params(self) -> dict[str, Any]:
        """The DetectionPanel's current Schmitt + preprocessing parameters.

        Public accessor so MainWindow's *Batch detect* action can run the SAME
        params the interactive Detect button would, without reaching into this
        panel's private DetectionPanel — the panel stays the single owner of the
        detection form."""
        return self._form_params()

    def batch_detection_params(self) -> dict[str, Any]:
        """:meth:`detection_params` for a run over MANY recordings: the same form, plus
        the recording its ``baseline_param_source`` is ABOUT (`BaselinePanel.batch_params`).

        The interactive Detect button runs on the recording the form was filled in
        for, so its source is the truth. A batch runs the same form on recordings it
        was NOT filled in for, and until sweep 7 stamped the selected recording's
        ``'proposed'`` on all of them (S7-DET1). `baseline.resolve_param_source`
        reads the scope inside each detector run and stamps every other recording
        ``'batch'`` -- with its own never-measured advisory, its own panel label and
        the truthful source in the exported ``.meta.json``."""
        return {**self._detection.params(), **self._baseline.batch_params()}

    def _form_params(self) -> dict[str, Any]:
        """The detection form's params WITH the open-pore baseline block merged in.

        One function, so a batch run, the Detect button, the menu action and a
        surrogate study cannot disagree about which baseline measured the depths.
        The baseline keys are ``baseline_*`` and are consumed by
        ``modules._CONSUMED_PARAM_KEYS`` before ``detect_events`` is called, so they
        never reach the detector as loose kwargs — they reach it as one
        `BaselineConfig` resolved by ``baseline.baseline_config_from_params``."""
        return {**self._detection.params(), **self._baseline.params()}

    @property
    def baseline_panel(self) -> BaselinePanel:
        """The owned open-pore baseline controls. Exposed for MainWindow exactly as
        ``measure_panel`` and ``merge_controls`` are — the panel stays the owner."""
        return self._baseline

    @property
    def baseline_workspace(self) -> BaselineCorrectionWorkspace:
        """The compact Baseline Correction comparison and export workspace.

        ``baseline_panel`` remains the compatibility/public seam for the established
        I0 estimator controls.  This separate accessor gives new callers the
        workspace without inviting them to reach through it for those controls.
        """
        return self._baseline_workspace

    def trigger_detection(self) -> None:
        """Run with the current form parameters (gating is enforced downstream).

        DELIBERATELY UNPROMPTED. This is the programmatic entry point — a script, the
        tests — and a modal on it would be a modal nobody clicked for. The curation
        question belongs to a GESTURE, and there are exactly THREE of them; the list is
        written out so the next person does not have to re-derive it:

        * the Detection tab's **Detect** button → :meth:`_on_detect_requested`;
        * the Detection tab's ML **Run** button → :meth:`_on_ml_detect_requested`;
        * the stale banner's **Re-run** button → :meth:`_on_stale_action`.

        Plus MainWindow's Analysis ▸ Run detection (⌘D), which is a gesture in another
        module and therefore gets its own gated door here,
        :meth:`trigger_detection_from_menu`. Every one of those four asks
        :meth:`confirm_replacing_curation` first. NOTE that the banner's Re-run does not
        come through this method at all — it replays the RESULT's stored params through
        :meth:`_on_rerun`, which is why gating this method would not have caught it.
        """
        self._run_detection(self._form_params())

    def trigger_detection_from_menu(self) -> None:
        """MainWindow's Analysis ▸ Run detection action, curation prompt and all.

        The menu item is a person's click exactly as the Detect button is, so it asks the
        same question (:meth:`confirm_replacing_curation`) — and it is a separate method
        rather than a flag on :meth:`trigger_detection` because a default decides
        behaviour, and the default for a programmatic run must be "do what you were told".
        """
        params = self._form_params()
        if not self.confirm_replacing_curation(str(params.get("detector") or _MODULE_ID)):
            return
        self._run_detection(params)

    def _on_detect_requested(self, params: dict[str, Any]) -> None:
        # The DetectionPanel emits its OWN params; the baseline block lives beside it
        # and has to be merged in here, or pressing Detect would silently run the
        # app default over whatever the user just confirmed.
        merged = {**params, **self._baseline.params()}
        if not self.confirm_replacing_curation(str(merged.get("detector") or _MODULE_ID)):
            return
        self._run_detection(merged)

    @staticmethod
    def hand_work_at_stake(df: pd.DataFrame) -> dict[str, int]:
        """Counts of human work a wholesale result replacement would destroy.

        This is the GUI's one pure counter for the single-run prompt, reconciled-result
        prompt and batch-run prompt. ``n_notes`` follows the history stack's canonical
        distinction exactly: every flag token outside ``DERIVED_FLAGS`` is a curator
        note/label (including the built-in ``flagged`` token), while automatic detector
        reasons are reproducible. ``n_reviewed`` protects an affirmative review even when
        it did not change the default accepted value.
        """
        from ..analysis.events import (
            DERIVED_FLAGS,
            REVIEW_COL,
            REVIEW_DONE,
            flag_tokens,
            is_automatic_rejection,
        )

        out = {
            "n_hand_rejected": 0,
            "n_added": 0,
            "n_notes": 0,
            "n_reviewed": 0,
        }
        if len(df) == 0:
            return out
        if "flag" in df.columns:
            cells = df["flag"].astype("string").fillna("")
            parsed = {value: set(flag_tokens(value)) for value in cells.unique()}
            notes = {value: bool(tokens - DERIVED_FLAGS)
                     for value, tokens in parsed.items()}
            automatic = {value: is_automatic_rejection(value) for value in parsed}
            out["n_notes"] = int(cells.map(notes).to_numpy(dtype=bool).sum())
            derived_only = cells.map(automatic).to_numpy(dtype=bool)
        else:
            derived_only = np.zeros(len(df), dtype=bool)
        accepted = (df["accepted"].to_numpy(dtype=bool)
                    if "accepted" in df.columns else np.ones(len(df), dtype=bool))
        out["n_hand_rejected"] = int((~accepted & ~derived_only).sum())
        if ADDED_COLUMN in df.columns:
            out["n_added"] = int(df[ADDED_COLUMN].to_numpy(dtype=bool).sum())
        if REVIEW_COL in df.columns:
            reviewed = (df[REVIEW_COL].astype("string") == REVIEW_DONE).fillna(False)
            out["n_reviewed"] = int(reviewed.sum())
        return out

    @staticmethod
    def hand_curation_at_stake(df: pd.DataFrame) -> tuple[int, int]:
        """Compatibility pair ``(n_hand_rejected, n_added)`` for older callers/tests.

        The counts a person can recognise. A rejection with NO reason, or with a reason
        outside the derivable vocabulary, is a person's decision and a new run cannot
        recover it; one carrying only a derivable token (``clipped_by_boundary``,
        ``non_finite``, …) is re-derived from the new span and costs nothing. The question
        is put to :func:`~nanoscope.analysis.events.is_automatic_rejection`, which is the
        one place that distinction is defined — the same call
        ``core.history._hand_curation`` makes for the Edit menu's warning, so the prompt
        and the warning cannot disagree about the number.
        """
        counts = AnalysisPanel.hand_work_at_stake(df)
        return counts["n_hand_rejected"], counts["n_added"]

    def confirm_replacing_curation(self, module_id: str) -> bool:
        """Ask before a re-run throws away work a PERSON did. True = go ahead.

        THE MEASURED DEFECT. Hand-reject 10 events on the -1000 mV recording (105 accepted
        of 115), press Run again, and 115 came back accepted with no prompt, no dialog and
        nothing in the status bar until after the fact. It is reachable by accident the
        moment two detectors exist: switch to CUSUM to peek, switch back, and Run used to
        be the only door back to the Schmitt table — the door that destroys it.

        WHAT COUNTS AS WORK A PERSON DID is not "any rejected row": a rejection the next
        run will simply re-derive (``clipped_by_boundary``, ``non_finite``, …) costs
        nothing and must not raise a prompt, or the prompt becomes noise and is clicked
        through. That distinction is
        :func:`~nanoscope.analysis.events.is_automatic_rejection`'s — the SAME authority
        ``core.history._hand_curation`` asks for the Edit menu's post-hoc warning and
        ``remeasure.merge_late_curation`` asks for its carry-over — so this prompt and that
        warning cannot quote different numbers for the same loss. It is counted here rather
        than imported because no GUI module may import the undo stack's module (MainWindow
        is its sole wirer, and ``test_undo_events`` enforces it); the two are held to the
        same answer by ``tests/gui/test_wa_undo_rerun_prompt.py`` instead.

        CANCEL IS THE DEFAULT AND ESCAPE IS CANCEL **BY IDENTITY**. The answer is read as
        ``clickedButton() is replace``, never off a standard-button enum and never off
        ``buttons()`` order — a previous bug in this codebase made the destructive button
        the Enter default because ``buttons()`` returns LAYOUT order. A dialog that is
        dismissed rather than answered (Escape, the window close box, a suppressed modal in
        the test harness) therefore returns False and the table survives.

        THIS IS THE GESTURE'S GATE, NOT THE RUN'S. It sits on EVERY door a user presses,
        and the whole list is here so a fourth one cannot be added by accident:
        the Detect button (:meth:`_on_detect_requested`), the ML Run button
        (:meth:`_on_ml_detect_requested`), Analysis ▸ Run detection
        (:meth:`trigger_detection_from_menu`) and the stale banner's Re-run
        (:meth:`_on_stale_action`) — because it is a question about intent, and only a
        person can answer it. The banner was the one that got away: it is a button, but it
        launched through the programmatic replay ``_on_rerun``, so it destroyed 10 hand
        rejections with no prompt for as long as the other three were gated (measured,
        ``wave-a/verify-does-it-hold/v10_stale_rerun.py``). The RECOVERABILITY half is not
        gated at all: every launch, gesture or programmatic, emits
        :attr:`willReplaceEvents` from :meth:`_run_detection`, so an undo step exists
        either way.

        NOT A DOOR: the banner's other face, "Re-measure" (a stale BASELINE stamp), goes to
        :meth:`remeasure_active_detection`, which keeps the events and every accept/reject
        on them. Prompting there would be the noise this docstring's second paragraph
        warns about — there is nothing at stake to lose.
        """
        if self._sel is None:
            return True
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return True
        prior = self._session.results.get(doc_id, channel, module_id)
        if prior is None or not isinstance(prior.payload, EventSet):
            return True
        work = self.hand_work_at_stake(prior.payload.df)
        n_rejected = work["n_hand_rejected"]
        n_added = work["n_added"]
        n_notes = work["n_notes"]
        n_reviewed = work["n_reviewed"]
        if not any(work.values()):
            return True

        def _plural(n: int, noun: str) -> str:
            return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

        parts = []
        if n_rejected:
            parts.append(_plural(n_rejected, "event") + " you rejected by hand")
        if n_added:
            parts.append(_plural(n_added, "event") + " you added by hand")
        if n_notes:
            parts.append(
                f"{n_notes} event{'s' if n_notes != 1 else ''} with labels/notes")
        if n_reviewed:
            parts.append(_plural(n_reviewed, "reviewed event"))
        stake = " and ".join(parts)
        reconciling = module_id == CONSENSUS_MODULE_ID
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle(
            "Replace reconciled table" if reconciling else "Re-run detection")
        box.setText(f"This recording has {stake}.")
        if reconciling:
            box.setInformativeText(
                "Creating a new reconciled result replaces the previous reconciled table "
                "and its curation. The original Schmitt, CUSUM and ML source tables are "
                "not changed.\n\nEdit ▸ Undo (⌘Z) puts this curated reconciled table back "
                "if you go ahead.")
        else:
            box.setInformativeText(
                "Re-running the detector replaces the whole table, and the new rows carry "
                "none of that curation — the detector has never seen it.\n\n"
                "Edit ▸ Undo (⌘Z) puts the curated table back if you go ahead.")
        replace = box.addButton("Replace the table", QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        if box.clickedButton() is replace:
            return True
        self.runRefused.emit(
            ("Reconciliation cancelled" if reconciling else "Detection cancelled")
            + f" — the table keeps {stake}.")
        return False

    def _run_detection(self, params: dict[str, Any]) -> bool:
        """Launch a Schmitt run; True when a job actually started.

        The bool is what lets the stale banner be dismissed by a RUN rather than by a CLICK
        (see :meth:`_on_rerun`) — both early returns here are silent refusals."""
        if self._sel is None:
            return False
        doc_id, channel = self._sel
        meta = self._session.document(doc_id).meta
        runnable, reason = self._gate_for(doc_id, channel)
        if not runnable:
            # Belt-and-braces, OUT LOUD. The gated buttons cannot reach here, but two
            # doors could and did: the stale banner's Re-run before it took the gate,
            # and MainWindow's Run detection action, which syncs its enabled state only
            # on a selection change -- so under the reverse override it stayed live and
            # this return was a click that did nothing and said nothing (D2).
            self.runRefused.emit(f"Detection refused — {reason}")
            return False
        # The Schmitt half of the notch-line cap (S7-NOTCH1; mirror of `_launch_ml`).
        # The DetectionPanel's box refuses an over-cap or illegal list itself, so
        # ``params()`` can never carry one -- but the stale banner's Re-run replays the
        # STORED dict of a result (`schmitt_params_from_result`) and a project written
        # before the cap can hold 10,923 lines that never went through the box. Refuse by
        # name, never trim: running 32 of a stored run's 10,923 lines would be a different
        # filter than the one the provenance would then claim.
        why = self._notch_replay_refusal(params)
        if why is not None:
            self.runRefused.emit(why)
            return False
        # ``start`` consults the run gate and may refuse (a batch pass already holds the
        # analysis domain), announcing it itself. Record the pending job only if a job
        # actually exists: a ``_pending`` naming a run that never started makes the next
        # controller signal — from something else entirely — look like this one's answer.
        # WHICH detector the user picked in the tab's detector combo. Defaulting to
        # Schmitt keeps every caller that predates the combo (and every stored params dict
        # written before it existed) running exactly the detector it always ran.
        if not self._controller.start(str(params.get("detector") or _MODULE_ID),
                                      doc_id, channel, params=dict(params),
                                      pore_id=meta.stem_key):
            return False
        # A run HAS started and it will replace whatever table is stored under this
        # detector. Commit the current one NOW rather than trusting the checkpoint the
        # arriving result triggers: that one pushes the last COMMITTED state, so an
        # unannounced in-place curation edit would be jumped past and Ctrl+Z would restore
        # a table from before it. See :attr:`willReplaceEvents`.
        self.willReplaceEvents.emit()
        self._pending, self._pending_sel = "detect", self._sel
        return True

    @staticmethod
    def _notch_replay_refusal(params: Mapping[str, Any]) -> str | None:
        """Why a Schmitt param dict may NOT run its explicit notch list, or ``None``.

        Judged by the DetectionPanel's own :meth:`~DetectionPanel.notch_lines_refusal`
        -- the ONE definition of a list the box may hold (the line cap, and every entry a
        positive finite number) -- over the list spelled back into the box's text form,
        so a stored dict is held to exactly the standard a paste is, in the same words.
        Only an ENABLED notch with an explicit list is judged: with the stage off no pass
        runs, and ``None`` is the series spelling (``notch_freq_hz`` / ``notch_harmonics``),
        which is bounded by its own spin boxes.
        """
        why = AnalysisPanel._notch_replay_why(params)
        return None if why is None else f"Detection refused — {why}"

    @staticmethod
    def _notch_replay_why(params: Mapping[str, Any]) -> str | None:
        """The body of `_notch_replay_refusal`, without the "Detection refused" frame, so
        the stale banner can print the same sentence under a disabled Re-run.

        The advice names the door that runs the box. It used to say "fix the list on
        the Preprocess tab and run again", which the banner's Re-run can never honour:
        Re-run replays the result's STORED dict (`_rerun_params`), and the stored dict
        is not editable from the UI -- measured on the owner's 0004 ABF, a legal list
        typed into the box and a second press brought back the identical 400-character
        refusal (CW-1). The Detect button is the door that reads the box."""
        if not params.get("notch_enabled"):
            return None
        freqs = params.get("notch_freqs")
        if freqs is None:
            return None
        try:
            tokens = [str(f) for f in freqs]
        except TypeError:
            tokens = [str(freqs)]
        why = DetectionPanel.notch_lines_refusal(", ".join(tokens))
        if why is None:
            return None
        body = why[len("Refused: "):] if why.startswith("Refused: ") else why
        return (f"the notch list this run would replay was not accepted: {body} Re-run "
                f"replays this run's STORED list, never the box: fix the list on the "
                f"Preprocess tab (“use an explicit line list”) and press Detect.")

    def _on_notch_lines_edited(self, _text: str) -> None:
        """A USER edit of the Preprocess tab's "lines" box (never the panel's rollback)."""
        self._notch_lines_edited = True

    def _on_notch_lines_committed(self) -> None:
        """Forward the DetectionPanel's standing notch-list refusal to the status line, once
        per refused edit -- the same ``runRefused`` road the ML box's refusal takes, so both
        boxes on the panel announce a refusal the same way. The sentence is read off the
        note the panel just painted under the box (its ``⚠`` line), so the status bar can
        never say something the panel does not; the DetectionPanel has no refusal signal
        of its own (it deliberately wires nothing). Runs AFTER the panel's own commit
        handler (connected first, in its ``__init__``)."""
        if not self._notch_lines_edited:
            return
        self._notch_lines_edited = False
        note = self._detection.notch_lines_note.text()
        marker = "\n⚠ "
        if marker not in note:
            return
        self.runRefused.emit(
            f"“lines” (Preprocess ▸ notch): {note.split(marker, 1)[1].strip()}")

    # -- ML detection (off-thread through the same controller) ------------------------------------
    def _on_ml_notch_text_changed(self, text: str) -> None:
        """Track the last under-cap "notch lines" contents (see the wiring comment).

        textChanged, not editingFinished, so the panel's OWN programmatic writes — a
        preset load, :meth:`DetectionPanel.set_ml_params` replaying a stored run — keep
        the rollback target current: those never emit editingFinished, and rolling an
        over-cap paste back to a value from before a replay would restore a config the
        user was no longer looking at. An over-cap value is simply never recorded, which
        is the entire mechanism: the rollback target is by construction always legal."""
        if len(DetectionPanel._parse_notch_freqs(text)) <= _ML_NOTCH_MAX_LINES:
            self._ml_notch_ok_text = text

    def _on_ml_notch_committed(self) -> None:
        """Refuse an over-cap "notch lines" commit, out loud, and roll the box back.

        Runs on the same editingFinished the DetectionPanel commits the field on (its
        handler is connected first, so the CUSTOM latch has already fired — harmless: a
        hand edit did happen, it was just refused). The refusal goes through
        ``runRefused`` — the same status-bar line every other "the panel declined to do
        what the gesture asked" message uses — and NAMES the cap and the count, because
        the failure this guards against is precisely a config change with no on-screen
        magnitude. The rollback then re-commits the restored text through the widget's
        own editingFinished, so the DetectionPanel re-derives its preset-status line from
        what the box now actually holds instead of leaving it describing the refused
        paste (two surfaces must not disagree)."""
        if self._ml_notch_reverting:
            return
        n_lines = len(DetectionPanel._parse_notch_freqs(self._detection.ml_notch_freqs.text()))
        if n_lines <= _ML_NOTCH_MAX_LINES:
            return
        self._ml_notch_reverting = True
        try:
            self._detection.ml_notch_freqs.setText(self._ml_notch_ok_text)
            self._detection.ml_notch_freqs.editingFinished.emit()
        finally:
            self._ml_notch_reverting = False
        self.runRefused.emit(
            f"“notch lines” accepts at most {_ML_NOTCH_MAX_LINES} lines — this edit held "
            f"{n_lines:,}. Each line is one zero-phase filtfilt pass over the whole "
            f"analysed range on every ML preprocess run, so {n_lines:,} lines would be "
            f"minutes of filtering. The box was restored to its previous contents.")

    def _on_ml_detect_requested(self, ui_params: dict[str, Any]) -> None:
        """Launch an ML run: resolve the 'Visible view / Whole recording' choice into
        concrete sample indices (the panel owns the trace view that knows the visible
        window), then start ``detect.keras`` on the shared controller. Gating is
        enforced here AND the button is disabled anyway (belt-and-braces).

        The curation prompt sits here, on the GESTURE, for the same reason it sits on
        ``_on_detect_requested`` rather than on ``_launch_ml``: it is a question about
        intent (see :meth:`confirm_replacing_curation`), and an ML re-run destroys hand
        curation exactly as a Schmitt one does."""
        if self._sel is None or not self.can_run_ml():
            return
        if not self.confirm_replacing_curation(_ML_MODULE_ID):
            return
        doc_id, channel = self._sel
        n = self._session.document(doc_id).meta.n_samples
        if ui_params.get("range_source") == "whole":
            i0, i1 = 0, n
        else:
            rng = self._visible_sample_range(doc_id, channel, n)
            if rng is None:
                # "Visible view" over a channel that is not visible. Substituting the
                # whole recording — which is what happened for as long as this returned a
                # tuple unconditionally — runs a slow detection over 10x the data the user
                # asked for and labels the result with a range they never chose. Say what
                # is wrong and run nothing; the gesture is one tick of a checkbox away
                # from working.
                doc = self._session.document(doc_id)
                name = doc.meta.channels[channel].name if channel < len(
                    doc.meta.channels) else f"channel {channel}"
                # THE FIRST REAL BLOCKER, not a downstream one. Two different things reach
                # here and they need different instructions: the channel is not drawn at all,
                # or it is drawn and the view has been panned off the end of it.
                if self._trace_view.visible_x_range_for(doc_id, channel) is None:
                    self.runRefused.emit(
                        f"“{doc.display_name} — {name}” is not on screen, so there is no "
                        f"visible view of it to detect over. Tick that channel in the File "
                        f"Manager, or choose “Whole recording”.")
                else:
                    axis = self._trace_view.x_axis_for(doc_id, channel)
                    unit = axis.unit
                    lo = float(axis.values[0]) if axis.values.size else 0.0
                    hi = float(axis.values[-1]) if axis.values.size else 0.0
                    self.runRefused.emit(
                        f"The view is panned off “{doc.display_name} — {name}”: it holds no "
                        f"samples at all, so there is nothing visible to detect over. This "
                        f"recording runs {lo:,.6g}–{hi:,.6g} {unit}. Zoom back onto it, or "
                        f"choose “Whole recording”.")
                return
            i0, i1 = rng
        full: dict[str, Any] = {
            "model_path": ui_params.get("model_path"),
            "enter": float(ui_params.get("enter", ml_detect.DEFAULT_ENTER)),
            "exit": float(ui_params.get("exit", ml_detect.DEFAULT_EXIT)),
            "range_source": ui_params.get("range_source", "visible"),
            "range_start": int(i0),
            "range_stop": int(i1),
        }
        # The model's OWN preprocessing/decode configs, passed straight through as the
        # JSON-safe dicts the panel built. Absent (an older stored param set) means the
        # module keeps its pre-existing apo-legacy behaviour.
        for key in _ML_CONFIG_KEYS:
            if key in ui_params:
                full[key] = ui_params[key]
        # ...AND THE CHARACTERISATION BASELINE, exactly as the Schmitt run gets it
        # (:meth:`_run_detection` merges ``self._baseline.params()`` the same way).
        # MLDetect reads ``baseline_*`` off its params for the event-masked baseline
        # every depth is referenced to (ml_modules.py, `baseline_config_from_params`)
        # — but ml_params() deliberately carries none, its docstring even saying the
        # Preprocess tab "is not involved in an ML run at all". The consequence,
        # measured on the owner's three apo recordings: press "Measure this
        # recording", accept the proposal, Run ML — and every depth in the table was
        # still measured at the DEFAULT 0.5 s / 0.1 s window with
        # baseline_param_source='default', while the standing advisory told the user
        # to do the very thing they had just done. The ML preprocessing chain (the
        # tab's masked_rough_s etc.) is the MODEL's normalisation and is a different
        # object; this merge touches only the measurement baseline. Provenance now
        # records the accepted windows and 'proposed', so the advisory can finally
        # be satisfied on the ML path too.
        full.update(self._baseline.params())
        self._launch_ml(full)

    def _launch_ml(self, full_params: dict[str, Any]) -> bool:
        """Start the ML module on the controller with a fully-resolved param dict
        (model_path, enter/exit, and CONCRETE range_start/range_stop indices). True when a
        job actually started — see :meth:`_run_detection` for why the bool matters."""
        if self._sel is None:
            return False
        # The other half of the notch-line cap (see _ML_NOTCH_MAX_LINES and the commit-time
        # guard): EVERY ML start funnels through here — the Run button, the menu action,
        # the stale banner's Re-run replaying a STORED param dict — so a flooded config
        # that never went through the box (an old project, a programmatic caller) is still
        # refused before it costs anything, by the same name. Refusing (False) rather than
        # trimming: running 32 of a stored run's 10,923 lines would be a different filter
        # than the one the provenance would then claim. Only an ENABLED notch is gated —
        # with the stage off, no pass runs and there is no cost to refuse.
        pre = full_params.get("ml_preprocess")
        if isinstance(pre, Mapping) and pre.get("notch_enabled", True):
            n_lines = len(pre.get("notch_freqs") or ())
            if n_lines > _ML_NOTCH_MAX_LINES:
                self.runRefused.emit(
                    f"ML detection refused: this configuration's notch list holds "
                    f"{n_lines:,} lines and “notch lines” is capped at "
                    f"{_ML_NOTCH_MAX_LINES} — each line is one zero-phase filtfilt pass "
                    f"over the analysed range, so {n_lines:,} lines would be minutes of "
                    f"filtering per run. Trim the notch list on the Detection tab’s ML "
                    f"page and run again.")
                return False
        doc_id, channel = self._sel
        meta = self._session.document(doc_id).meta
        if not self._controller.start(_ML_MODULE_ID, doc_id, channel,
                                      params=dict(full_params), pore_id=meta.stem_key):
            return False   # refused by the run gate; see _run_detection
        # Same reason as the Schmitt/CUSUM launch: the table this run will replace has to
        # be committed BEFORE it is replaced, or an unannounced curation edit is jumped
        # past. See :attr:`willReplaceEvents`.
        self.willReplaceEvents.emit()
        self._pending, self._pending_sel = "ml_detect", self._sel
        return True

    def _visible_sample_range(self, doc_id: str, channel: int,
                              n: int) -> tuple[int, int] | None:
        """Map the visible x-window OF (doc_id, channel) to ``[i_start, i_stop)`` sample
        indices, or None when there is no visible view of that channel to map.

        ``searchsorted`` on the (ascending) x-axis values works for both a time axis
        (seconds) and a rate-unknown sample axis.

        TWO different questions, and conflating them is what made "Visible view" lie twice.

        *Which x window?* — :meth:`TraceView.visible_x_range_for` asks the pane that DRAWS
        this channel, and returns None when nothing does. This used to read
        ``anchor_plot()``, i.e. whichever pane happens to be first, and then fall back to
        ``0, n`` — the whole recording — when the numbers came out degenerate. So a channel
        that was not on screen at all got another recording's window searched into its axis
        and, when that collapsed, silently detected over EVERYTHING while the control said
        "Visible view". Measured: (0, 4000) of 40000 after a project restore.

        *Which axis?* — ``x_axis_for`` is the axis the curve is actually drawn against
        (the document's, shifted by the layer's ``time_offset_s``). Searching the native
        axis instead answers about a different stretch of the recording on any aligned
        layer.

        None is a REFUSAL, not an error: the caller says what is wrong and runs nothing.
        The one remaining fallback (a DEGENERATE window — zero-width, but sitting on the
        data — on a channel that IS drawn) still returns the whole recording, because there
        the user is looking at the recording and the whole of it is the honest superset of a
        zero-width view.

        A window that lies ENTIRELY OFF the data is a different thing and is refused. It is
        not degenerate, it is disjoint, and the whole recording is not a superset of nothing.
        [MEASURED] on the owner's own ABF, whose ``t0_s`` is 61.303 s so its x-axis runs
        61.30–121.18 s: panning to 10–20 s and pressing Run with "Visible view" selected gave
        ``i0 = i1 = 0``, took the fallback, and analysed all 2,993,798 samples — 59.876 s —
        with ``analyzed_duration_s`` recording that in provenance while the control still read
        "Visible view".
        """
        rng = self._trace_view.visible_x_range_for(doc_id, channel)
        if rng is None:
            return None
        xmin, xmax = rng
        vals = self._trace_view.x_axis_for(doc_id, channel).values
        if vals.size == 0:
            return 0, n
        i0 = int(np.clip(np.searchsorted(vals, xmin, side="left"), 0, n))
        i1 = int(np.clip(np.searchsorted(vals, xmax, side="right"), 0, n))
        if i1 <= i0:
            if float(xmax) < float(vals[0]) or float(xmin) > float(vals[-1]):
                return None
            return 0, n
        return i0, i1

    def _on_stale_action(self, kind: str) -> None:
        """The stale banner's button — A GESTURE. Dispatches on WHICH stamp went stale.

        The two repairs are not interchangeable and must never be confused: a
        calibration change re-DETECTS (the thresholds were in the old gain, so the
        events themselves are in question, and the table is replaced); a baseline
        change RE-MEASURES (the events are right, only their amplitudes were read
        against the wrong open-pore reference, and every accept/reject on them has
        to survive). Sending a baseline change down the re-detect path would answer
        it by destroying the owner's curation -- which is exactly what §4.2.3 says
        the offered button must not do.

        AND THE RE-DETECT HALF ASKS FIRST, because this is a button a PERSON CLICKS.
        Measured through a real MainWindow at 1512x900 on the 749k-sample -1000 mV apo
        recording (`wave-a/verify-does-it-hold/v10_stale_rerun.py`): hand-reject 10 of
        115 events (105 accepted), nudge the gain by 5%, click the banner's "Re-run" --
        `accepted 105 -> 115`, `*** NO MODAL APPEARED ***`. The Detect button on the very
        same state asked (`['Cancel', 'Replace the table']`). The curation prompt had been
        put on the two doors that were then known to be gestures (`_on_detect_requested`,
        `trigger_detection_from_menu`) while this one kept going through `_on_rerun`, the
        DELIBERATELY unprompted launcher -- so the loss the prompt exists to prevent was
        still one click away, through a third door.

        THE GATE BELONGS HERE, NOT IN :meth:`_on_rerun`, for the same reason it is not in
        :meth:`_run_detection`: `_on_rerun` is also the programmatic replay (scripts, and
        `test_staleness_regressions` / `test_s7_fixup_stale_banner_action_gate` call it
        directly), and a modal nobody clicked for would block them. A gesture asks; a
        caller that was told what to do does what it was told.

        The banner is NOT dismissed on a cancel: nothing was recomputed, so the warning
        that the depths on screen were read under the old gain is still true -- the same
        invariant `_on_rerun` records for its own early returns.
        """
        res = self._stored_detection()
        if res is not None and res.module_id == CONSENSUS_MODULE_ID:
            # Defensive even though `_stale_action_refusal` disables the visible button:
            # tests, scripts and queued Qt signals can invoke this slot directly.
            self.runRefused.emit(
                _RECONCILED_SOURCE_CHANGE_REFUSAL
                if kind == _STALE_RECONCILIATION_SOURCE
                else _RECONCILED_STALE_REFUSAL
            )
            return
        if kind == _STALE_BASELINE:
            self.remeasure_active_detection()
            return
        # `_active_detection` is what `_stored_detection` fetches and what the re-run
        # replaces, so the prompt counts the stake on the table the click would destroy
        # (an ML result when the ML tab is showing, the Schmitt one otherwise).
        if not self.confirm_replacing_curation(self._active_detection):
            return
        self._on_rerun()

    def _on_rerun(self) -> None:
        """Recompute THIS detection under the new calibration. THE LAUNCHER, NOT THE DOOR.

        Reached from the stale banner's Re-run *through* :meth:`_on_stale_action`, which is
        where the curation prompt lives — and reached bare by scripts and by the staleness
        tests, which is why the prompt is NOT here. Anything added to this method runs
        without a person having asked for it.

        Two things have to be true, and neither was.

        The banner is dismissed only once a run is actually launched. Hiding it first made
        every path that returns early — an ML result with no model path, no selection, a
        detector the gate has since disabled — a banner-DISMISSING NO-OP: the click removed
        the only warning that the depths on screen were computed under the old gain, and
        nothing recomputed them. The result then read as current for the rest of the session,
        and a fresh warning could only come from ANOTHER calibration edit.

        And it re-runs the params the RESULT was computed with, not whatever the form
        happens to show. "Re-run" here means "the same detection, new calibration" (the ML
        branch always meant that); taking the live form instead silently re-detected with
        different settings — after a reload, with the panel's DEFAULTS — and every hand
        curation went with the replaced event table.
        """
        res = self._stored_detection()
        if res is not None and res.module_id == CONSENSUS_MODULE_ID:
            # Never reinterpret the derived table as the panel's ordinary Schmitt result.
            # This is the programmatic backstop for callers that bypass the stale banner.
            self.runRefused.emit(
                _RECONCILED_SOURCE_CHANGE_REFUSAL
                if self._stale.kind == _STALE_RECONCILIATION_SOURCE
                else _RECONCILED_STALE_REFUSAL
            )
            return
        if self._active_detection == _ML_MODULE_ID:
            if res is None or not res.params.get("model_path"):
                # Nothing to re-run -- and the banner stays up, because it is still true.
                # Said, not swallowed: the button is gated off for this (see
                # `_stale_action_refusal`), so only a programmatic caller lands here.
                self.runRefused.emit(
                    "Re-run refused — this ML result records no model path, so the same "
                    "model cannot be run again.")
                return
            # Re-run the SAME model over the SAME range/thresholds under the new
            # calibration (the stored params already carry resolved indices).
            launched = self._launch_ml({
                k: res.params[k]
                for k in ("model_path", "enter", "exit",
                          "range_source", "range_start", "range_stop",
                          *_ML_CONFIG_KEYS)
                if k in res.params
            })
        else:
            launched = self._run_detection(self._rerun_params(res))
        if launched:
            self._stale.setVisible(False)

    # -- surrogate studies (off-thread; they re-detect on full-trace surrogates) -----------------
    def _surrogate_builder(self, detect_params: dict[str, Any]):
        """A zero-arg callable that builds the study's ``truth.SynthRecording``
        ON THE WORKER THREAD.

        The SAME baseline-drift removal detection applies is baked into the wrapped
        signal (preprocess_for_detection), so the surrogate studies detect on the
        identical preprocessed trace. The low-pass is NOT baked in: detect_events still
        owns it via ``filter_cutoff_hz`` in the detect kwargs (the detrend keys are
        stripped from those kwargs so the detector never sees them). Polarity is the
        user's override, else measured from the data.

        WHY A BUILDER RATHER THAN THE RECORDING. This used to run the whole conditioning
        chain inline, on the GUI thread, before the job reached the controller. Measured
        on the owner's 2,993,798-sample -600 mV apo ABF, interleaved min-of-5: **81 ms**
        with no stage enabled, **130 ms** with the notch on, **361 ms** with all six —
        and it is linear in n, so his 50,495,999-sample ATP recording pays roughly
        seventeen times that. A third of a second of dead window on his smallest file,
        and seconds on his largest, is the freeze a user answers by clicking again.
        Inside the job it costs nothing visible, it polls the cancel token between
        stages (:func:`~nanoscope.analysis.modules.preprocess_for_detection`), and a
        parameter the kernels refuse now produces the SAME "Analysis failed" box that
        the identical parameter produces on Detect — which is the consistency the
        previous fix here was reaching for and could not have while the two errors came
        out of two different places.

        The Session lookups stay on the GUI thread, as everywhere else in this app: the
        closure captures plain value objects (an array, a rate, a name, a units string)
        and never touches the Session, a widget or Qt.
        """
        doc_id, channel = self._sel  # type: ignore[misc]
        ch = self._session.calibrated(doc_id, channel)
        raw = ch.data
        fs = ch.sample_rate_hz
        units = ch.units
        name = self._session.document(doc_id).display_name
        params = dict(detect_params)

        def build():
            from ..analysis.baseline import derive_polarity
            from ..analysis.truth import SynthRecording

            data = np.ascontiguousarray(raw, dtype=np.float32)
            data = preprocess_for_detection(data, fs, params)
            pol = params.get("polarity")
            if pol is None:
                pol = int(derive_polarity(data))
            return SynthRecording(data=data, sample_rate_hz=float(fs),
                                  polarity=int(pol), name=name, units=units)

        return build

    @staticmethod
    def _detect_kwargs_for_truth(params: dict[str, Any]) -> dict[str, Any]:
        """The truth machinery passes polarity itself (from the SynthRecording); strip it
        and any None off-sentinels the sweep grid would otherwise fight with k_in. The
        detrend and filter keys are already applied to the wrapped signal, so strip them
        too — they are not detect_events kwargs."""
        drop = {"polarity", "k_in", *DETREND_PARAM_KEYS, *FILTER_PARAM_KEYS}
        out = {k: v for k, v in params.items() if k not in drop}
        return {k: v for k, v in out.items() if v is not None}

    def _on_sweep_requested(self, params: dict[str, Any]) -> None:
        if self._sel is None:
            return
        build = self._surrogate_builder(params)
        kw = self._detect_kwargs_for_truth(params)

        def job():
            from ..analysis.truth import sweep_threshold
            return sweep_threshold(build(), detect_kwargs=kw)

        # ``_pending`` is set only once a job exists: the run gate can refuse (a batch
        # pass holds the analysis domain) and announces that itself, and a ``_pending``
        # naming a run that never started makes the NEXT controller signal look like
        # this one's answer.
        if self._controller.run_callable(job, label="sweep_threshold"):
            self._pending, self._pending_sel = "sweep", self._sel

    # -- the baseline probe (off-thread, through the same controller) -----------------------------
    def _on_baseline_view_changed(self, _on: bool) -> None:
        """Either view box moved. Both are redrawn, for the current selection only."""
        self._apply_baseline_views()
        self._sync_baseline_workspace()

    def _apply_baseline_views(self) -> None:
        """Draw what the two boxes on `BaselinePanel` ask for, and nothing else.

        CLEAR-THEN-APPLY, always. These are per-LAYER requests behind a per-SELECTION
        checkbox: tick it on the -600 mV run, select the -1000 mV run, untick, and
        without the clear the first run's rail stays up claimed by nothing on screen.
        One writer, so the trace can only ever show what the boxes currently say.

        ON THE GUI THREAD, DELIBERATELY. The estimate is 70 ms on the owner's 749k-sample
        recording and ~0.3 s at 3 M (`baseline_track`'s docstring carries the numbers);
        the probe beside it is 0.56-3.8 s and is correctly off-thread. A worker here
        would buy nothing and cost a cancellation path.

        The polarity is PASSED, for the reason the door's docstring gives: derive_polarity
        reads +1 on the owner's own drifting raw traces and moves the baseline by up to
        312 pA, and a drawn baseline wrong by 312 pA is worse than none -- it looks like
        an answer.
        """
        from ..analysis.baseline import characterisation_baseline, decimate_track

        self._trace_view.clear_baseline_views()
        panel = self._baseline
        panel.say_view_problem("")
        if self._sel is None or not (panel.show_baseline() or panel.subtract_baseline()):
            return
        doc_id, channel = self._sel
        chan = self._session.calibrated(doc_id, channel)
        fs = chan.sample_rate_hz
        if fs is None:
            panel.say_view_problem(
                "This recording has no sample rate, so a baseline in seconds has no "
                "meaning here and nothing can be drawn.")
            return
        pol = self._detection.params().get("polarity")
        try:
            result = characterisation_baseline(
                np.asarray(chan.data), float(fs), config=panel.config(),
                polarity=None if pol is None else int(pol),
                param_source=panel.param_source(), health=False)
        except Exception as exc:                      # noqa: BLE001
            # A refusal must not take the window down AND must not go silent: the box
            # is still ticked, so something has to say why nothing appeared.
            panel.say_view_problem(f"The baseline could not be drawn here: {exc}")
            return
        if panel.subtract_baseline():
            self._trace_view.set_baseline_subtracted(doc_id, channel, result.baseline)
        if panel.show_baseline():
            self._trace_view.set_baseline_track(doc_id, channel,
                                                *decimate_track(result.baseline))

    # -- unified baseline-correction workspace --------------------------------------------------
    def _sync_baseline_workspace(self, params: Mapping[str, Any] | None = None) -> None:
        """Reflect the two existing baseline contracts in their shared workspace.

        This only copies state *out* of the owner widgets.  In particular, it does
        not calculate a baseline, enable a filter, or replace a stored detection.
        The small guard prevents a direct legacy checkbox click from being echoed back
        as a new workspace mode change.
        """
        if not hasattr(self, "_baseline_workspace"):
            return
        runtime = dict(params) if params is not None else self._detection.params()
        self._baseline_workspace_syncing = True
        try:
            self._baseline_workspace.sync_runtime(runtime)
            self._baseline_workspace.sync_display_state(
                self._baseline.show_baseline(), self._baseline.subtract_baseline())
        finally:
            self._baseline_workspace_syncing = False

        # A newly selected/restored form may never emit paramsChanged.  Record its
        # baseline-specific signature here so the first subsequent drift edit can
        # still invalidate a preview rather than being mistaken for initialization.
        self._baseline_workspace_runtime_signature = self._baseline_workspace_signature(runtime)

    @staticmethod
    def _baseline_workspace_signature(params: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        """The subset of a Detection form capable of changing drift comparison bytes."""
        return tuple((key, repr(params.get(key))) for key in ("polarity", *DETREND_PARAM_KEYS))

    def _baseline_workspace_current_recipe_signature(self) -> tuple[Any, ...]:
        """The currently displayed controls in the same form captured by a job."""
        return _baseline_workspace_recipe_signature(
            mode=self._baseline_workspace.output_mode(),
            runtime_params=self._detection.params(),
            reference_params=self._baseline.params(),
            reference_source=self._baseline.param_source(),
        )

    def _mark_baseline_workspace_runtime_change(self, params: Mapping[str, Any]) -> None:
        """Invalidate a comparison only when its own runtime recipe moved."""
        signature = self._baseline_workspace_signature(params)
        old, self._baseline_workspace_runtime_signature = (
            self._baseline_workspace_runtime_signature, signature)
        if old is not None and old != signature:
            self._baseline_workspace.mark_preview_stale(
                "Baseline-drift settings changed — refresh the comparison before relying on it.")

    def _on_baseline_workspace_mode_changed(self, mode: str) -> None:
        """Make the explicit workspace output choice visible on the main trace too."""
        if self._baseline_workspace_syncing:
            return
        if mode == _BASELINE_WORKSPACE_ZERO:
            show, subtract = True, True
        elif mode == _BASELINE_WORKSPACE_RETAINED:
            # The retained-offset comparison is a signal-chain result, not an
            # alternative zeroed TraceView mode.  Leaving an old zero-residual view
            # under it would make the two plots tell different stories.
            show, subtract = False, False
        else:
            self._baseline_workspace.set_status(
                f"Unknown baseline output mode {mode!r}; no display state changed.",
                warning=True)
            return

        # The legacy checks still own what TraceView draws.  Block their individual
        # signals so one deliberate mode choice performs one redraw rather than an
        # intermediate residual/rail state followed by the final one.
        blockers = (QtCore.QSignalBlocker(self._baseline.show_track),
                    QtCore.QSignalBlocker(self._baseline.subtract_view))
        try:
            self._baseline.show_track.setChecked(show)
            self._baseline.subtract_view.setChecked(subtract)
        finally:
            del blockers
        self._apply_baseline_views()
        self._sync_baseline_workspace()
        self._baseline_workspace.mark_preview_stale(
            "Output mode changed — refresh the comparison to calculate this mode.")

    def _on_baseline_workspace_runtime_enabled(self, enabled: bool) -> None:
        """Proxy the explicit runtime opt-in to DetectionPanel, its sole owner."""
        self._detection.set_params({"detrend_enabled": bool(enabled)})
        self._sync_baseline_workspace()
        self._baseline_workspace.mark_preview_stale(
            "Runtime correction changed — refresh the comparison before relying on it.")

    def _on_baseline_workspace_runtime_method(self, method: str) -> None:
        """Proxy the estimator selection without duplicating DetectionPanel validation."""
        complaint = self._detection.set_detrend_method(str(method))
        self._sync_baseline_workspace()
        if complaint:
            self._baseline_workspace.set_status(complaint, warning=True)
            return
        self._baseline_workspace.mark_preview_stale(
            "Baseline-drift method changed — refresh the comparison before relying on it.")

    def _baseline_workspace_target(
        self,
    ) -> tuple[tuple[str, int], Any, float, str, tuple[float, float]] | None:
        """Resolve an honest current/rate/calibration snapshot for a workspace job."""
        if self._sel is None:
            self._baseline_workspace.clear_preview("Select a current recording first.")
            return None
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            self._baseline_workspace.clear_preview("That recording is no longer open.")
            return None
        runnable, reason = self._gate_for(doc_id, channel)
        if not runnable:
            self._baseline_workspace.clear_preview(reason)
            return None
        chan = self._session.calibrated(doc_id, channel)
        if chan.sample_rate_hz is None:
            # Defensive duplication of the gate: a caller cannot produce a
            # seconds-based baseline just because a stale gate happened to say yes.
            self._baseline_workspace.clear_preview(
                "Sample rate unknown — a baseline in seconds cannot be calculated.")
            return None
        fingerprint = tuple(self._session.calibration(doc_id, channel).fingerprint())
        return ((doc_id, channel), chan, float(chan.sample_rate_hz),
                self._selection_units(), fingerprint)

    @staticmethod
    def _baseline_workspace_note(mode: str, runtime_enabled: bool,
                                 other_stages: bool, warnings: tuple[str, ...]) -> str:
        """One honest, compact status sentence for a completed comparison."""
        if mode == _BASELINE_WORKSPACE_ZERO:
            text = ("Visual residual only — it is not used for detection or the "
                    "processed-trace export.")
        else:
            text = (
                "Runtime Baseline drift is enabled for detection and processed export."
                if runtime_enabled else
                "Preview only — enable runtime correction to use it for detection or export.")
            if other_stages:
                text += " Other enabled Signal chain stages are not included here."
        if warnings:
            text += " Advisory: " + " ".join(warnings)
        return text

    def _on_baseline_workspace_refresh(self) -> None:
        """Calculate the selected baseline comparison off the GUI thread."""
        target_info = self._baseline_workspace_target()
        if target_info is None:
            return
        target, chan, fs, units, fingerprint = target_info
        mode = self._baseline_workspace.output_mode()
        if mode not in (_BASELINE_WORKSPACE_ZERO, _BASELINE_WORKSPACE_RETAINED):
            self._baseline_workspace.clear_preview("Choose a baseline output mode first.")
            return
        params = dict(self._detection.params())
        reference_config = self._baseline.config()
        reference_source = self._baseline.param_source()
        reference_params = dict(self._baseline.params())
        recipe_signature = _baseline_workspace_recipe_signature(
            mode=mode, runtime_params=params, reference_params=reference_params,
            reference_source=reference_source)
        # Calibrated channel arrays are immutable snapshots in the session; making a
        # contiguous view avoids a needless full-recording allocation while the worker
        # reads exactly the calibration/fingerprint captured above.
        raw = np.ascontiguousarray(np.asarray(chan.data))
        runtime_enabled = bool(params.get("detrend_enabled"))
        excluded_stages = _baseline_workspace_excluded_stages(params)
        other_stages = bool(excluded_stages)

        self._baseline_workspace_revision += 1
        revision = self._baseline_workspace_revision

        def job() -> dict[str, Any]:
            baseline, corrected, warnings, _fit_provenance = _baseline_workspace_arrays(
                raw, fs, reference_config=reference_config,
                reference_source=reference_source, runtime_params=params, mode=mode)
            return {
                "revision": revision,
                "mode": mode,
                "preview": _baseline_workspace_preview(raw, baseline, corrected, fs),
                "units": units,
                "runtime_enabled": runtime_enabled,
                "other_stages": other_stages,
                "warnings": warnings,
            }

        self._baseline_workspace.set_busy(True)
        if self._controller.run_callable(job, label="baseline_correction_preview"):
            self._pending, self._pending_sel = "baseline_workspace_preview", target
            self._baseline_workspace_job = (
                revision, target, fingerprint, units, recipe_signature)
            return
        self._baseline_workspace.set_busy(False)
        self._baseline_workspace.set_status(
            "The comparison could not start because another analysis is running.",
            warning=True)

    def _on_baseline_workspace_export(self) -> None:
        """Handle the workspace button; the public method keeps tests/menu callers simple."""
        self.export_baseline_correction()

    def export_baseline_correction(self, path: str | Path | None = None) -> bool:
        """Export raw, fitted baseline and selected correction as one atomic CSV set.

        The baseline is always fitted over the entire recording, and so is the
        exported array.  Slicing a cursor selection first would change a global
        baseline fit near the selection edges; a smaller selection can still be made
        after export in the user's analysis tool, with the fitted baseline retained
        beside it.  Calculation and writing run through the shared analysis controller
        so a multi-million sample CSV never freezes the inspector.
        """
        target_info = self._baseline_workspace_target()
        if target_info is None:
            return False
        target, chan, fs, units, fingerprint = target_info
        mode = self._baseline_workspace.output_mode()
        if mode not in (_BASELINE_WORKSPACE_ZERO, _BASELINE_WORKSPACE_RETAINED):
            self._baseline_workspace.set_status(
                "Choose a baseline output mode before exporting.", warning=True)
            return False
        doc_id, channel = target
        doc = self._session.document(doc_id)
        if path is None:
            chosen = ask_export_path(
                self, title="Export baseline correction",
                suggestion=f"{Path(doc.display_name).stem}_baseline_correction.csv",
                filters="CSV (*.csv);;All files (*)",
                outputs=baseline_correction_export_paths)
            if chosen is None:
                return False
            path = chosen

        params = dict(self._detection.params())
        reference_config = self._baseline.config()
        reference_source = self._baseline.param_source()
        reference_params = dict(self._baseline.params())
        excluded_stages = _baseline_workspace_excluded_stages(params)
        recipe_signature = _baseline_workspace_recipe_signature(
            mode=mode, runtime_params=params, reference_params=reference_params,
            reference_source=reference_source)
        fitted_recipe: Mapping[str, Any] | None
        fitted_source: str
        runtime_recipe: Mapping[str, Any] | None
        if mode == _BASELINE_WORKSPACE_ZERO:
            # The worker replaces this launch-time form snapshot with the actual
            # BaselineResult provenance.  That records any short-trace clamp or
            # fallback instead of claiming the requested windows ran unchanged.
            fitted_recipe = reference_params
            fitted_source = reference_source
            # A visual residual must never be stamped with a processing recipe.
            runtime_recipe = None
        else:
            # Here the fitted rail comes from the signal-chain drift estimator, not
            # from the separate open-pore reference form.  Only the drift-stage
            # recipe is stamped: other enabled stages are explicitly listed as
            # excluded because they did not produce these arrays.
            fitted_recipe = {
                "baseline_estimator": "signal_chain_baseline_drift",
                "polarity": params.get("polarity"),
                **{key: params.get(key) for key in DETREND_PARAM_KEYS if key in params},
            }
            fitted_source = "baseline_drift_estimate"
            runtime_recipe = {
                key: params.get(key)
                for key in ("polarity", *DETREND_PARAM_KEYS)
                if key in params
            }
        raw = np.ascontiguousarray(np.asarray(chan.data))
        channel_name = doc.meta.channels[channel].name
        cal_gain, cal_offset = fingerprint
        out_path = Path(path)
        self._baseline_workspace_revision += 1
        revision = self._baseline_workspace_revision

        def job() -> dict[str, Any]:
            baseline, corrected, warnings, actual_fit_provenance = _baseline_workspace_arrays(
                raw, fs, reference_config=reference_config,
                reference_source=reference_source, runtime_params=params, mode=mode)
            written_fit_recipe = (
                actual_fit_provenance
                if mode == _BASELINE_WORKSPACE_ZERO and actual_fit_provenance is not None
                else fitted_recipe
            )
            out = write_baseline_correction(
                out_path, raw, baseline, corrected, fs, mode=mode, units=units,
                start_idx=0, t0_s=clock.instrument_origin_s(doc.meta),
                baseline_params=written_fit_recipe, runtime_params=runtime_recipe,
                baseline_source=fitted_source, warnings=warnings,
                comparison_scope=(
                    "open_pore_reference_display_only"
                    if mode == _BASELINE_WORKSPACE_ZERO
                    else "baseline_drift_stage_only"),
                excluded_runtime_stages=(
                    () if mode == _BASELINE_WORKSPACE_ZERO else excluded_stages),
                recording=doc.display_name, doc_key=persistent_key(doc),
                channel=channel_name, channel_index=int(channel),
                span="whole recording (baseline fitted on whole recording)",
                cal_gain=cal_gain, cal_offset=cal_offset)
            return {
                "revision": revision,
                "out": out,
                "mode": mode,
                "n_samples": int(raw.size),
            }

        self._baseline_workspace.set_busy(True)
        if self._controller.run_callable(job, label="baseline_correction_export"):
            self._pending, self._pending_sel = "baseline_workspace_export", target
            self._baseline_workspace_job = (
                revision, target, fingerprint, units, recipe_signature)
            return True
        self._baseline_workspace.set_busy(False)
        self._baseline_workspace.set_status(
            "The export could not start because another analysis is running.", warning=True)
        return False

    def _finish_baseline_workspace_export(
        self, obj: object, target: tuple[str, int] | None
    ) -> None:
        """Report an export without pretending it belongs to a later selection."""
        job, self._baseline_workspace_job = self._baseline_workspace_job, None
        self._baseline_workspace.set_busy(False)
        if not isinstance(obj, Mapping) or job is None:
            self._baseline_workspace.set_status(
                "Baseline correction export returned an unexpected result.", warning=True)
            return
        out = obj.get("out")
        if not isinstance(out, Mapping) or "data" not in out or "meta" not in out:
            self._baseline_workspace.set_status(
                "Baseline correction export did not produce its CSV and sidecar.", warning=True)
            return
        data, meta = Path(out["data"]), Path(out["meta"])
        message = (
            f"Wrote {int(obj.get('n_samples', 0)):,} baseline-correction samples to "
            f"{data.name}, with provenance in {meta.name}.")
        _revision, launched_target, launched_fp, launched_units, launched_recipe = job
        snapshot_matches = (
            target == launched_target and self._sel == launched_target and
            self._session.has_document(launched_target[0]) and
            tuple(self._session.calibration(*launched_target).fingerprint()) == tuple(launched_fp))
        if snapshot_matches:
            settings_match = (
                self._selection_units() == launched_units and
                self._baseline_workspace_current_recipe_signature() == launched_recipe)
            if settings_match:
                self._baseline_workspace.set_status(message)
            else:
                # The file is valid and its sidecar records the launch-time recipe.
                # Say explicitly that the controls now describe a different output,
                # rather than allowing a late completion to look current.
                self._baseline_workspace.set_status(
                    message + " It uses the settings captured when export started; "
                    "the current baseline settings changed while it ran.",
                    warning=True)
        else:
            # The file is still valid — its sidecar names the launch-time calibration
            # snapshot — but do not overwrite the new target's workspace status with a
            # completion that it did not request.
            self.runRefused.emit(message + " It belongs to the recording/calibration snapshot at launch.")

    def _finish_baseline_workspace_preview(
        self, obj: object, target: tuple[str, int] | None
    ) -> None:
        """Install a completed compact comparison only on its original snapshot."""
        job, self._baseline_workspace_job = self._baseline_workspace_job, None
        self._baseline_workspace.set_busy(False)
        if not isinstance(obj, Mapping) or job is None:
            self._baseline_workspace.mark_preview_stale(
                "The comparison returned an unexpected result; refresh it again.")
            return
        revision, launched_target, launched_fp, launched_units, launched_recipe = job
        if (target != launched_target or self._sel != launched_target or
                int(obj.get("revision", -1)) != revision or
                not self._session.has_document(launched_target[0])):
            self._baseline_workspace.mark_preview_stale(
                "The recording changed while this comparison ran; refresh it for the current view.")
            return
        current_fp = tuple(self._session.calibration(*launched_target).fingerprint())
        if tuple(current_fp) != tuple(launched_fp):
            self._baseline_workspace.mark_preview_stale(
                "Calibration changed while this comparison ran; refresh it before using it.")
            return
        if (self._selection_units() != launched_units or
                self._baseline_workspace_current_recipe_signature() != launched_recipe):
            self._baseline_workspace.mark_preview_stale(
                "Baseline settings changed while this comparison ran; refresh it for "
                "the controls now on screen.")
            return
        preview = obj.get("preview")
        if not isinstance(preview, Mapping):
            self._baseline_workspace.mark_preview_stale(
                "The comparison did not contain plottable samples; refresh it again.")
            return
        warnings = tuple(str(item) for item in obj.get("warnings", ()) if str(item))
        mode = str(obj.get("mode", ""))
        self._baseline_workspace.set_preview(
            np.asarray(preview.get("time_s", ())), np.asarray(preview.get("raw", ())),
            np.asarray(preview.get("baseline", ())), np.asarray(preview.get("corrected", ())),
            units=str(obj.get("units", "")), mode=mode,
            message=self._baseline_workspace_note(
                mode, bool(obj.get("runtime_enabled")), bool(obj.get("other_stages")), warnings),
        )

    def _on_probe_requested(self) -> None:
        """Measure this recording's timescales off-thread and offer a proposal.

        NEVER on the GUI thread. The whole job this method dispatches -- conditioning,
        the probe, and the health second pass on what it proposes -- was measured at
        1.97 s (median of three presses) on the owner's 3 M-sample -600 mV apo file
        and 6.38 s on the 9 M-sample -700 mV file, and click-to-proposal wall time is
        within 60 ms of the callable, so that is the job, not the pump. (An earlier
        version of this docstring said "230 ms on the 3 M-sample apo file"; that
        number was ``ProbeResult.elapsed_s``, the PROBE STAGE alone, and it is what
        the ``baseline_probe`` docstring's table describes -- it was never the
        button.) A probe is exactly the kind of thing a user presses twice if the
        window stops repainting for two seconds.

        The probe runs on the quantitative CHARACTERIZATION trace rebuilt from the live
        form. That is the same chain classical detection uses for baseline and amplitude
        measurement, with high-pass omitted even when the orange candidate-localization
        preview includes it. A proposal measured on raw data would still size its
        windows against mains hum and other enabled conditioning the measurement path
        does not see. It is handed the polarity the detection form has settled on,
        because everything the probe measures is one-sided: `derive_polarity` reads +1
        on the owner's own drifting raw traces, and a probe pointed the wrong way
        measures the drift's dwell distribution instead of the events'.
        """
        if self._sel is None:
            self._baseline.set_busy(False)
            return
        doc_id, channel = self._sel
        ch = self._session.calibrated(doc_id, channel)
        fs = ch.sample_rate_hz
        if fs is None:
            self._baseline.set_busy(False)
            self._baseline.set_gate(
                False, "This recording has no sample rate, so a window in seconds "
                       "has no meaning here.")
            return
        params = self._detection.params()
        pol = params.get("polarity")
        raw = np.asarray(ch.data)
        # Bound as a plain pure function, not through ``self``: the closure runs on the
        # worker thread and must not hold a widget. Unlike the orange preview's
        # candidate-localization trace, this quantitative path omits high-pass.
        condition = characterization_trace

        def job():
            from ..analysis.baseline import characterisation_baseline
            from ..analysis.baseline_probe import probe_and_propose
            # Conditioned INSIDE the job: the chain itself is ~120 ms on a 3 M-sample
            # recording, which belongs off the GUI thread with the probe rather than
            # in front of it.
            data = condition(raw, float(fs), params)
            proposal = probe_and_propose(data, float(fs),
                                         polarity=None if pol is None else int(pol))
            # ...AND THE VERDICT ON WHAT IT PROPOSES, in the same job.
            #
            # The six health checks used to be computed only by a DETECTION and read back
            # out of its provenance, so the block that justifies these windows appeared
            # only after the run they were supposed to inform — and on the Preprocess tab,
            # which the user has left by then. The confirmation click, which decides every
            # depth_pA in the table, was made blind.
            #
            # It is the same estimator, at the PROPOSED config, on the SAME quantitative
            # characterization trace, at the polarity the probe settled on, so the
            # checks shown here are the checks the detection will stamp — not an
            # approximation of them.
            # Measured cost of this second pass on the owner's own recordings: 560 ms
            # (-600 mV, 3.0 M samples), 2.6 s (-500 mV, 12.5 M), 3.8 s (-700 mV, 9.1 M),
            # roughly the probe again. It is off-thread and cancellable like everything
            # else on this controller, and it buys the one click in the app that must not
            # be guessed at.
            health = None
            try:
                health = characterisation_baseline(
                    data, float(fs), config=proposal.config,
                    polarity=(int(proposal.probe.polarity)
                              if proposal.probe is not None else None),
                    health=True, cache=False).health
            except Exception:  # noqa: BLE001
                # A refusal here must not cost the user the proposal: an estimator that
                # will not run on these windows is exactly what the panel then says,
                # rather than swallowing the whole probe.
                health = None
            return (proposal, health)

        if self._controller.run_callable(job, label="baseline_probe"):
            self._pending, self._pending_sel = "probe", self._sel
        else:
            self._baseline.set_busy(False)   # refused: the spinner must not outlive the run

    @staticmethod
    def _recording_identity(source_key: str) -> str:
        """What this recording IS, for seeding: its bytes when they can be read.

        A PATH IS A NAME, NOT A RECORDING -- ``core.identity``'s opening line, and the
        reason :class:`~nanoscope.core.identity.SourceFingerprint` exists. Seeding off
        the source key made the published percentage a property of where the file sat.
        MEASURED, wired ``MainWindow``, the owner's -700 mV apo recording copied
        byte-for-byte to three names (all md5 ``1a4d0c62d443fd4fd235f72fd48c0356``),
        opened, channel 2 selected, nothing else touched: seeds
        **1592734707 / 1091228298 / 1722390092** -- three "reproducible" numbers for one
        recording. The tooltip's promise that pressing the button again reproduces the
        counts held only for a fixed path, so the figure published off a scratch copy is
        not the figure the owner sees when he opens his own file.

        SIZE + DIGEST, NOT ``mtime_ns``, and not ``n_samples``. ``mtime_ns`` is carried by
        the fingerprint for the MESSAGE only and is explicitly never a mismatch on its own
        -- copying a file preserves it and re-saving one in a sync folder changes it -- so
        letting it into the seed would re-seed a recording that had merely been touched.
        ``n_samples`` is not passed because it is the count a *document* loaded, and this
        method must answer the same way for the same bytes however they were opened.

        ``None`` back from :func:`fingerprint_source` means UNKNOWN, not "changed": a
        segment's persistent key is a ``segment_uid`` and not a file at all, and a source
        on a disconnected volume is unreadable. Both fall back to the source key, which is
        the pre-fix behaviour and is still stable for the whole life of that identity --
        so a segment reproduces too, it just reproduces off its uid instead of off bytes
        it does not have.

        TWO THINGS THIS IS NOT, both of which the reader should know:

        * **It runs on the GUI thread**, inside ``_on_null_requested``, before the job
          exists. Measured on the owner's own files, min of 7: **0.33 ms** on the 108 MB
          -700 mV recording and **0.33 ms** on the 150 MB -500 mV one -- the fingerprint
          reads three 256 KiB blocks regardless of size, so it is flat. That is a
          non-issue for a local disk and it is NOT a guarantee: a source on a sleeping
          external drive or an unresponsive mount can stall ``open``/``read`` for
          seconds, and :func:`fingerprint_source` catches ``OSError``, not slowness.
        * **It fingerprints the bytes on disk NOW, while the analysis runs on the bytes
          loaded THEN.** Re-export over the same filename while the recording is open and
          the next click seeds differently against unchanged in-memory data, so the
          tooltip's "pressing it again reproduces these counts" quietly stops being true
          until the file is reloaded. The load-time fingerprint the app already keeps
          (``MainWindow._restore_sources``) is the right input and is not reachable from
          a staticmethod on this panel.
        """
        from ..core.identity import fingerprint_source

        fp = fingerprint_source(source_key)
        if fp is None:
            return f"key:{source_key}"
        return f"bytes:{fp.size}:{fp.digest}"

    @staticmethod
    def _null_seed(target: tuple[str, int], settings: Mapping[str, Any]) -> int:
        """The seed the Null control runs under, derived from WHAT IS BEING MEASURED.

        `truth.null_control` draws its noise refill and every one of its AAFT surrogates
        from a single Generator, and handed ``rng=None`` that Generator is seeded from OS
        entropy. So the percentage this panel printed was a different percentage every
        time the button was pressed. Measured through this panel on the owner's -700 mV
        apo recording -- five clicks against the same 1164 real detections, nothing
        else touched:
        **17.0 / 24.0 / 20.3 / 13.8 / 19.0 %**, a 10.2-point spread on a number the reader
        takes for a property of the pore. (The battery of 2026-08-22 measured the same
        thing on the same file: 23.7 / 18.2 / 15.6 / 13.2 / 20.6 %.)

        Seeding is HALF the answer and this docstring is the other half's signpost: one
        seeded draw out of a distribution whose surrogate counts span 42 to 643 is a
        *reproducible* meaningless number unless the spread travels with it. That is
        :meth:`StatsPanel.set_fdr`'s job, and it needs the seed to do it -- so the seed
        is stamped into the result dict the job returns (``out["seed"]``), additively,
        the way every other provenance in this app travels with its number rather than
        beside it.

        WHY THIS SEED AND NOT A CONSTANT, AND NOT A COUNTER:

        * It is a function of the recording and of the settings, so **pressing the
          button again reproduces the number exactly** -- which is the whole claim -- while
          changing a setting draws a genuinely independent null instead of replaying one
          phase randomisation over every recording in the app.
        * It is derived from ``source_of(doc_id)``, not from ``doc_id`` itself: a doc_id
          carries a per-LOAD suffix, so closing and reopening the same file would
          otherwise re-seed it and the same recording would print a different rate
          tomorrow.
        * It is blake2b, not :func:`hash`, which is salted per process -- an app restart
          must not change a published number either.

        AND THE TWO HALVES THAT WERE WRONG THE FIRST TIME, both of which made this seed a
        function of something OTHER than what was measured:

        * **The recording is its bytes, not its path.** See :meth:`_recording_identity`:
          the source key is reduced to a content fingerprint, so three copies of one file
          under three names seed identically and a re-export over the same filename does
          not silently keep the old draw.
        * **``settings`` is EVERY setting that reaches the surrogate**, which is the whole
          detection form -- not the ``detect_events`` kwargs. The 8 ``DETREND_PARAM_KEYS``
          and 15 ``FILTER_PARAM_KEYS`` are stripped before the DETECTOR sees them, because
          they are not its kwargs; they are not stripped before the SURROGATE sees them,
          because :meth:`_surrogate_builder` runs ``preprocess_for_detection`` with the
          full form and the AAFT surrogates are drawn from *that* conditioned trace.
          Seeding off the stripped dict meant toggling the notch, the median, the
          high-pass or the Savitzky-Golay replayed the same phase randomisation while this
          docstring claimed a changed setting "draws a genuinely independent null".
          MEASURED, wired ``MainWindow``, -700 mV, channel 2: notch toggled -> seed
          **1592734707**, unchanged; median + high-pass + Savitzky-Golay + detrend all
          toggled on top of it -> **1592734707**, still unchanged. With surrogate counts
          spanning 19x inside a single run, a difference read as "the notch changed the
          false-positive rate" was mostly the draw the two runs shared.
        """
        from ..core.identity import source_of

        payload = repr((AnalysisPanel._recording_identity(source_of(str(target[0]))),
                        int(target[1]),
                        sorted((str(k), repr(v)) for k, v in settings.items())))
        return int.from_bytes(
            hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest(), "big")

    def _on_null_requested(self, params: dict[str, Any]) -> None:
        if self._sel is None:
            return
        build = self._surrogate_builder(params)
        kw = {k: v for k, v in params.items()
              if k != "polarity" and k not in DETREND_PARAM_KEYS
              and k not in FILTER_PARAM_KEYS and v is not None}
        # SEED FROM ``params``, THE WHOLE FORM -- not from ``kw``. ``kw`` is what the
        # DETECTOR takes; ``build`` closes over ``params`` and conditions the trace the
        # surrogates are drawn from with all of it. See :meth:`_null_seed`.
        seed = self._null_seed(self._sel, params)

        def job():
            from ..analysis.truth import null_control
            # A dict copy so the seed stamp is this panel's annotation and never a
            # mutation of what `null_control` handed back.
            out = dict(null_control(build(), detect_kwargs=kw, rng=seed))
            out["seed"] = int(seed)
            return out

        if self._controller.run_callable(job, label="null_control"):
            self._pending, self._pending_sel = "null", self._sel

    def _on_recovery_requested(self, params: dict[str, Any]) -> None:
        if self._sel is None:
            return
        build = self._surrogate_builder(params)
        kw = {k: v for k, v in params.items()
              if k != "polarity" and k not in DETREND_PARAM_KEYS
              and k not in FILTER_PARAM_KEYS and v is not None}

        def job():
            from ..analysis.truth import recovery_map
            return recovery_map(build(), detect_kwargs=kw)

        if self._controller.run_callable(job, label="recovery_map"):
            self._pending, self._pending_sel = "recovery", self._sel

    def _announce_discarded(self, res: AnalysisResult) -> None:
        """Say that an answer arrived for a recording that is no longer there.

        The two cases read differently and the user's next move differs with them, so they
        are two sentences: the recording was CLOSED (nothing to do), or it is still open but
        this answer belongs to an earlier LOAD of it — which is the case that used to install
        silently and wrongly, and the one where "run it again" is the actual remedy.
        """
        from ..core.identity import source_of

        source = source_of(res.doc_id)
        live = self._session.doc_id_for_source(source)
        # The reopened document's own display name when there is one (a segment's is the
        # user's, not its synthetic path); otherwise the file name, which is all that is
        # left once the recording is gone.
        name = (self._session.document(live).display_name if live is not None
                else Path(source).name) or "that recording"
        if live is not None:
            self.runRefused.emit(
                f"“{name}” was closed and reopened while that run was in flight, so its "
                f"answer describes a recording that is no longer loaded. It was discarded "
                f"— run it again on the recording that is open now.")
        else:
            self.runRefused.emit(
                f"“{name}” was closed while that run was in flight; its answer was "
                f"discarded.")

    # -- controller results ----------------------------------------------------------------------
    def _on_controller_finished(self, obj: object) -> None:
        pending, self._pending = self._pending, None
        # The target the job was LAUNCHED on — a surrogate study takes seconds and the user
        # may have moved on. Its answer belongs to that recording, not to the current one.
        target, self._pending_sel = self._pending_sel, None
        if isinstance(obj, AnalysisResult):
            # Detection (and any AnalysisResult) is stored on the MAIN THREAD here; the
            # resulting resultsChanged drives _on_results_changed to refresh + redraw.
            #
            # ...but ONLY if THE LOAD it was computed on is still open. A detection that
            # lands after its document was closed is a result about nothing: close_document
            # has already run ResultStore.drop_doc, so storing it would re-populate the store
            # for a dead document and mark the session dirty.
            #
            # ``doc_id`` now names the LOAD, not the path (core.identity), which is what
            # makes this check true rather than merely usually true. It used to pass for a
            # DIFFERENT load of the same file: close the recording mid-detection, reopen it —
            # at another range, or after re-exporting it — and the stale answer installed
            # itself onto the new document. Measured: a 100,000-sample detection landing on a
            # re-opened 4,000-sample load, all five events past the end of the file and
            # ``analyzed_duration_s`` 1.0 s against a true 0.04 s, so the capture rate was
            # wrong by 25x with nothing on screen to say so.
            if self._session.has_document(obj.doc_id):
                self._session.add_result(obj)
            else:
                self._announce_discarded(obj)
            return
        # ...and only if that recording is still open. Same rule as the detection sink above:
        # a file's doc_id is its path, so an answer stored under a closed document would
        # re-attach to it on its next open and present a stale eta(t) / FDR as fresh.
        if target is not None and not self._session.has_document(target[0]):
            target = None
        if pending == "baseline_workspace_preview":
            self._finish_baseline_workspace_preview(obj, target)
        elif pending == "baseline_workspace_export":
            self._finish_baseline_workspace_export(obj, target)
        elif pending == "probe":
            # A probe belongs to the recording it was measured on. The user is free to
            # select another one while it runs, and a proposal that landed on the wrong
            # file would be applied to it with a confirming click -- the same class of
            # defect the surrogate studies below are keyed against.
            self._baseline.set_busy(False)
            # The probe job returns (proposal, health): the windows it offers AND the
            # estimator's verdict on them, so the confirmation click is not blind (see
            # the job in :meth:`_on_probe_requested`). A bare BaselineProposal is still
            # accepted — a project restored from a session that predates this, and every
            # test that hands the panel a proposal directly, must keep working.
            proposal, health = obj if isinstance(obj, tuple) and len(obj) == 2 \
                else (obj, None)
            if target is not None and self._sel == target and isinstance(
                    proposal, BaselineProposal):
                self._baseline.set_proposal(proposal)
                self._baseline.set_health(health, measured_for="proposal")
                # Remember the OFFER beside showing it. If the user accepts it over the
                # refusal, the standing mark that acceptance raises has to survive the
                # user navigating to another recording and back, and the only honest way
                # to raise that mark again is to replay this exact pair — the proposal
                # and the health that condemned it. See `_restore_baseline_mark`.
                self._baseline_offer = (target, proposal, health)
        elif pending in ("remeasure", "compare"):
            # NOT gated on ``target``/``self._sel``: a re-measure belongs to the
            # RESULT it was launched for, which is tracked separately. Gating it on
            # the selection would silently drop the answer when the user clicked
            # another recording during the seconds it takes -- leaving the table
            # marked stale, the click apparently ignored, and no way to tell that
            # from a failure.
            if isinstance(obj, RemeasureResult):
                self._on_remeasure_finished(obj, install=(pending == "remeasure"))
            else:
                self._remeasure_target = None
                self._remeasure_curation = None
                self._remeasure_params = None
        elif pending == "merge_all":
            # NOT gated on ``target``/``self._sel``, for the reason the re-measure
            # branch above is not: the merge belongs to the RESULT it was launched for,
            # which `_merge_job` tracks. Gating it on the selection would drop the whole
            # job in silence whenever the user clicked another recording during it.
            self._finish_merge_all(obj)
        elif pending == "sweep":
            self._show_sweep(obj)
        elif pending == "null" and isinstance(obj, dict):
            if target is not None:
                self._fdr[target] = obj
                if self._sel == target:
                    self._stats.set_fdr(obj)
        elif pending == "recovery":
            if target is not None:
                self._recovery[target] = obj
                if self._sel == target:
                    self._stats.set_recovery_available(True)

    def _on_controller_failed(self, _msg: str) -> None:
        pending = self._pending
        self._pending = None   # MainWindow shows the message box
        self._pending_sel = None
        if pending in ("remeasure", "compare") and self._remeasure_target is not None:
            # ONE TERMINAL MESSAGE PER STARTED JOB, on the failure path too. Every
            # baseline entry point emits ``('started', "Re-measuring N events…")`` and
            # MainWindow leaves it in the status bar until something replaces it, so a
            # raise inside the worker used to leave that sentence standing for the rest
            # of the session -- a job that had been discarded, still reading as running,
            # beside a modal the user has already dismissed. `_on_remeasure_finished`
            # guarantees this for every way of FINISHING; this is the way of failing.
            self._say_baseline(
                "failed",
                "Re-measure discarded: it could not finish. The table is unchanged.")
        # A re-measure that raised must not leave its target armed: the next
        # finished job of ANY kind would otherwise be able to reach the install
        # path with an answer that was never computed for it.
        self._remeasure_target = None
        self._remeasure_curation = None
        self._remeasure_params = None
        # A 'Merge all' that raised says so ONCE, on this path as on the finishing one:
        # MainWindow leaves the "Merging N events…" line in the status bar until
        # something replaces it, so a silent exit reads as a job still running.
        if pending == "merge_all" and self._merge_job is not None:
            job, self._merge_job = self._merge_job, None
            self.mergeFinished.emit(
                MergeReport("failed", int(job[3]), int(job[3]), window_s=float(job[4])))
        self._merge_job = None
        # A probe that raised (an unrunnable filter cutoff, a detrend window longer
        # than the trace) must not leave the button reading "Measuring…" for ever --
        # a busy widget that never comes back is indistinguishable from a hang, and
        # the panel refuses a second probe while it thinks one is in flight.
        self._baseline.set_busy(False)
        if pending in ("baseline_workspace_preview", "baseline_workspace_export"):
            self._baseline_workspace_job = None
            self._baseline_workspace.set_busy(False)
            self._baseline_workspace.set_status(
                "Baseline correction could not finish; no comparison or export was applied.",
                warning=True)

    def _on_controller_cancelled(self) -> None:
        self.forget_pending_job()

    def forget_pending_job(self) -> None:
        """Drop the record of the in-flight controller job. Public because
        ``AnalysisController.abort()`` deliberately emits NOTHING, so a Cancel driven
        from the status bar (or a session clear) never reaches
        :meth:`_on_controller_cancelled` — and a ``_pending`` left pointing at an
        abandoned sweep is exactly the state in which a later answer gets routed to the
        wrong study. MainWindow calls this from its cancel path; the panel does not
        reach for the window."""
        self._pending = None
        self._pending_sel = None
        self._remeasure_target = None
        self._remeasure_curation = None
        self._remeasure_params = None
        # An abandoned 'Merge all' must not leave a check-list behind for a LATER answer
        # to be validated against — the same rule, for the same reason, as the
        # re-measure snapshots above it.
        self._merge_job = None
        # Same reason as in :meth:`_on_controller_failed`: an abandoned probe must
        # release the button, or Cancel leaves it saying "Measuring…" indefinitely.
        self._baseline.set_busy(False)
        self._baseline_workspace_job = None
        if hasattr(self, "_baseline_workspace"):
            self._baseline_workspace.set_busy(False)

    def _show_sweep(self, sweep_df) -> None:
        try:
            dlg = SweepDialog(sweep_df, self)
        except Exception:  # a malformed sweep frame must not crash the panel
            return
        dlg.kChosen.connect(self._detection.k_in.setValue)
        dlg.exec()

    # -- results display -------------------------------------------------------------------------
    def _on_results_changed(self, doc_id: str, module_id: str) -> None:
        # resultsChanged carries (doc_id, module_id) but not the channel; fetch by the
        # selected channel, so a sibling channel's run is a harmless idempotent redraw.
        if self._sel is None or doc_id != self._sel[0]:
            return
        # A source table can be replaced while its derived Reconciled snapshot is on
        # screen (Undo/Redo, a scripted install, or another detector run). Preserve and
        # re-judge that snapshot instead of silently switching away from the warning it
        # now needs to show. The updated source remains available in the result selector.
        shown = self._stored_detection()
        if (shown is not None
                and shown.module_id == CONSENSUS_MODULE_ID
                and module_id != CONSENSUS_MODULE_ID
                and module_id in self._reconciliation_input_fingerprints(shown)):
            self._check_staleness(shown)
            return
        if module_id in _DETECTION_MODULE_IDS:
            res = self._session.results.get(doc_id, self._sel[1], module_id)
            if res is not None:
                # A freshly (re-)computed detection becomes the active one shown, so
                # its events populate the table/overlays regardless of which detector
                # produced them.
                self._set_active_detection(module_id)
                # ...and the form follows the table, here as on selection change and
                # project reopen: "the settings on the tab are the settings that
                # produced the table on screen" (`_seed_form_from`). This signal also
                # carries Edit ▸ Undo/Redo (`EventHistory.undo` installs the restored
                # run through `session.add_result`, the same door as everything else),
                # which used to leave the NEWER run's stages ticked over the restored
                # table — Detect then re-ran with filters the shown events never had.
                # Guarded by identity, exactly like `set_selection`'s same-target
                # branch: a redraw of the result the form was already seeded from must
                # not stamp over what the user has dialled in since.
                if res is not self._seeded_from:
                    self._seeded_from = res
                    self._seed_form_from(res)
                self._show_result(res)
        elif module_id in ("fit.dwell", "fit.depth"):
            self._refresh_fit_text()

    def _show_result(self, res: AnalysisResult) -> None:
        events = res.payload
        if not isinstance(events, EventSet):
            return
        # Materialise the 'added' marker (default False) on the shared df the moment a
        # detection is shown, so it round-trips through save/load + exports and the
        # score can read it even before the user hand-adds a missed event.
        ensure_added_column(events.df)
        self._sync_result_selector()
        self._table.set_events(events)
        # WHICH CLOCK the table's t_start_s is on, relative to the trace beside it. The
        # offset belongs to what is DRAWN (meta.t0_s + the layer's alignment nudge), so
        # it comes from the shared TraceView, and it is pushed on every result rather
        # than read once: opening another recording changes it (10.25 / 61.30 / 0 s on
        # the owner's three files) and a stale offset is a wrong label, not a missing one.
        self._table.set_clock_offset(
            self._trace_view.clock_offset_for(res.doc_id, res.channel))
        self._fill_summary(events)
        # ...AND THE CAVEATS THOSE NUMBERS WERE PRODUCED UNDER, beside them.
        #
        # ``res.warnings`` is the module's own list of what its table must not be
        # trusted for. `main_window` persists it (save) and restores it (open) at four
        # call sites, `remeasure` recomputes it, and until now NOTHING RENDERED IT: the
        # unprobed-open-pore advisory -- the one warning that fires on a run whose
        # windows nobody chose, worth 317 events / 85.68 pA against 276 / 87.59 on the
        # owner's own -600 mV file -- was built, appended, carried, saved and shown to
        # no one.
        #
        # Pushed WHOLE, on every result shown, so this is the surface for warnings in
        # general and not for one string. It is pushed here rather than inside
        # `_fill_summary` because a warning belongs to the RESULT, while `_fill_summary`
        # is also the curation-edit refresh path (`_refresh_summary_from_table`), which
        # re-derives the counts from the live df on every accept/reject click and must
        # leave the caveats exactly where they were, because rejecting a row does not
        # change which open-pore window the depths were measured under. (That path does
        # hold the result -- it fetches one via `_stored_detection()` to reach the df --
        # so the reason to keep the push here is what a warning is ABOUT, not what is
        # reachable from there. An earlier version of this comment claimed the latter.)
        self._stats.set_warnings(res.warnings)
        prov = events.provenance or {}
        # Pushed on EVERY result, present or absent (see `_show_polarity_evidence`) —
        # a detection whose provenance recorded no measurement is not a reason to keep
        # showing the last recording that did.
        self._show_polarity_evidence(prov.get("polarity_evidence"))
        # The baseline's own verdict on the run that produced this table -- a mask that
        # found nothing, a mask that swallowed the recording, a rough pass measuring
        # drift instead of events, a trace too short for the two-pass structure. Every
        # one of those regimes returns a finite, plausible, WRONG baseline with no
        # exception anywhere, so the checks are the only thing standing between the
        # user and a silently mis-measured depth column. Shown for the detection on
        # screen, Schmitt or ML alike, and read from provenance so a reopened project
        # still shows it.
        # ...and, BEFORE that block is pushed, whether the windows it describes were ones
        # the app itself refused and the user accepted anyway. The re-seed that puts those
        # windows back (`_seed_form_from`) has already run by the time a result is shown,
        # and it clears the mark that says so; this is where the mark comes back with them.
        # It must run FIRST: the replay puts the accepted proposal briefly back on offer,
        # which is a PROPOSAL-scope health block, and the detection-scope one below is the
        # verdict on the table on screen — the two are never interchangeable, so the last
        # word here has to be the detection's.
        self._restore_baseline_mark()
        self._baseline.set_health(prov.get("baseline_health"),
                                  measured_for="detection")
        # ...and the windows the run ACTUALLY used, when they are not the form's: a
        # clamped run on a short cut kept showing the asked-for 0.5 / 0.1 s (S7-DET3).
        self._baseline.set_run_windows(prov)

        # Overlays: remove the previous, then draw this result on the pane. The event SPANS are
        # drawn from the LIVE (curated) df so a reject persists on the trace across selection
        # round-trips (not just until the next edit); any NON-span overlay the module declared is
        # drawn from res.overlays as-is. _redraw_spans_from_df clears the tracked overlays first.
        self._clear_marker()
        self._redraw_spans_from_df()
        doc_id, channel = res.doc_id, res.channel
        for hint in res.overlays:
            if hint.kind is OverlayKind.SPAN:
                continue   # spans come from the curated df above, not the frozen detection snapshot
            self._overlay_ids.append(self._trace_view.add_overlay(doc_id, channel, hint))

        self._check_staleness(res)
        self._refresh_fit_text()
        self._refresh_score()
        curation_only = res.module_id == CONSENSUS_MODULE_ID
        # A reconciled table is a derived curation surface. Its rows can come from
        # different conditioning chains and its membership provenance is cluster-shaped,
        # so geometry edits belong on a source table followed by a new reconciliation.
        self._measure.set_detection_present(
            not curation_only,
            _RECONCILED_EDIT_REFUSAL if curation_only else "",
        )
        self._merge.set_detection_present(
            not curation_only,
            _RECONCILED_EDIT_REFUSAL if curation_only else "",
        )
        # The Fit button is gated on exactly the same state as the merge controls: there is a
        # detection on this target to fit. It used to have no gate at all and reported the
        # failure into its own results box AFTER the click.
        self._stats.set_detection_present(True)

    def _show_polarity_evidence(self, evidence: Any) -> None:
        """THE ONLY WRITER of the Detection tab's "measured" polarity line — both answers.

        Both, in one method, because the defect was that only one of them had a writer.
        :meth:`_show_result` pushed the evidence when a detection's provenance carried a
        dict and NOTHING ever put the line back, so one recording's measurement —
        "blockade (down): 18 down / 0 up (confidence 1.00, measured)" — stayed on screen
        for a recording whose own events go the other way, with an empty event table
        beside it, and stayed after that recording was CLOSED, under a title reading "No
        channel selected".

        It is not a decorative readout. Polarity is the direction the whole depth column
        is measured against, this line is what a user consults before setting the
        override for a recording he has not detected yet, and it carries the word
        "measured" and a confidence of 1.00 — so a claim about the wrong file is read as
        a decisive statement about this one, and the wrong override "does not raise an
        error, it simply detects almost nothing" (the app's own manual).

        ``DetectionPanel.set_polarity_evidence`` stays the only writer of a MEASUREMENT;
        this is the only writer of its ABSENCE, and the text it restores is the widget's
        own initial "not yet measured", captured at construction (``_polarity_unmeasured``).
        The reset would sit better in ``panels.py`` as ``set_polarity_evidence(None)``;
        that file is not this change's to edit, and keeping both halves in this one method
        is what stops the pair from drifting in the meantime.
        """
        if isinstance(evidence, dict):
            self._detection.set_polarity_evidence(evidence)
            return
        self._detection.polarity_readout.setText(self._polarity_unmeasured)

    def _fill_summary(self, events: EventSet) -> None:
        df = events.df
        n = len(df)
        summary: dict[str, Any] = {
            "n_events": n,
            "n_accepted": int(df["accepted"].to_numpy(dtype=bool).sum()) if n else 0,
            "n_filter_limited": int(df["filter_limited"].to_numpy(dtype=bool).sum()) if n else 0,
            "n_pores": events.n_pores if n else 0,
        }
        # The fits drop accepted-AND-filter-limited events by default; show that
        # count next to the toggle so the user knows the fit's n before running it.
        if n:
            both = (df["accepted"].to_numpy(dtype=bool)
                    & df["filter_limited"].to_numpy(dtype=bool))
            self._stats.set_filter_limited_count(int(both.sum()))
        else:
            self._stats.set_filter_limited_count(0)
        try:
            summary["rate_hz"] = events.rate_hz()
        except ValueError:
            summary["rate_hz"] = None
        # A RATE MUST NAME THE ROWS IT COUNTED. `EventSet.rate_hz` divides ALL rows by
        # the analysed duration, and this table shows "n accepted" one row above it, so
        # the two disagreed whenever anything was curated out: -600 mV merged has 240
        # rows of which 239 are accepted (one `clipped_by_boundary`), and the Summary
        # read 4.008 Hz where the report's master table and the Voltage-dependence
        # panel both quote 3.992. Three surfaces, two denominators, nothing on screen
        # saying which. The panel now shows the accepted-only rate -- the same basis as
        # the other two surfaces -- and `rate_hz` stays as it was for the module scalar
        # that stamps it into provenance.
        summary["rate_accepted_hz"] = None
        dur = events.provenance.get("analyzed_duration_s")
        if n and dur:
            summary["rate_accepted_hz"] = float(
                df["accepted"].to_numpy(dtype=bool).sum()) / float(dur)
        # A MEDIAN MUST NAME THE ROWS IT COUNTED, exactly like the rate above it. These
        # four were taken over the WHOLE frame while "n accepted" and the rate beside
        # them were taken over the accepted rows, so rejecting the 30 deepest of 115
        # events moved "n accepted" 115 -> 85 and left "median depth (pA) 90.33" — a
        # number describing rows the curator had just thrown away — sitting directly
        # underneath it, byte-identical, with nothing on screen saying the two rows
        # count different sets. The Plots window opened on the same detection filters to
        # accepted events by default, so the two surfaces also disagreed about the same
        # column of the same table (FE-02).
        #
        # `EventSet.accepted` is the app's own answer to which rows a summary is over
        # ("Only curated-in events. This is what figures and fits should use"), and it
        # is the basis the Voltage-dependence panel's medians already use
        # (`voltage_dep._metric_value`, accepted_only=True). The labels in
        # `StatsPanel._SUMMARY_ROWS` say "accepted" so the basis is READ, not inferred.
        acc = df["accepted"].to_numpy(dtype=bool) if n else np.zeros(0, dtype=bool)
        adf = df[acc]
        if len(adf):
            summary["median_dwell_us"] = float(np.median(adf["dwell_s"].to_numpy())) * 1e6
            summary["median_frac_depth"] = float(np.nanmedian(adf["frac_depth"].to_numpy()))
            # A DEPTH DOES NOT TRAVEL ALONE. `frac_depth` is a ratio, and a ratio with
            # no numerator and no denominator on screen is a number the reader cannot
            # check: the summary showed dI/I0 and nothing else, so "measured against
            # what?" was answerable only by scrolling the event table sideways past two
            # constant text columns, or by exporting the provenance sidecar. These are
            # the SAME rows the two medians above are taken over, so the three numbers
            # are arithmetically consistent by construction.
            summary["median_depth_pA"] = float(np.nanmedian(adf["depth_pA"].to_numpy()))
            summary["median_baseline_pA"] = float(
                np.nanmedian(adf["baseline_pA"].to_numpy()))
        else:
            # Every row rejected (or nothing detected) is not a median of zero rows: the
            # rows read "—", the same as an empty detection, because there is no
            # accepted event to take a median of and inventing one over the rejected
            # rows is the defect this branch exists to avoid.
            summary["median_dwell_us"] = None
            summary["median_frac_depth"] = None
            summary["median_depth_pA"] = None
            summary["median_baseline_pA"] = None
        # ...and WHICH open-pore estimator produced them, read from the detection's own
        # provenance rather than from the live form: the form is what the NEXT run would
        # use, and after a parameter edit those are different windows. The export
        # sidecar already carries the fingerprint for exactly this reason; the surface
        # the user actually reads did not.
        summary.update(_baseline_summary_rows(events.provenance or {}))
        # ...and in WHAT UNIT. The amplitude rows used to be captioned "(pA)" no matter
        # what channel produced them — on an HDF5 whose loader note reads "Units are not
        # declared in this file. Magnitudes suggest nA" the Summary asserted picoamps
        # while the PSD window on the same channel printed "(units unknown)". The panel
        # is handed the channel's own units and says "units unknown" when there are
        # none, which is what every other read-out in the app already does (FE-04).
        self._stats.set_units(self._selection_units())
        self._stats.set_summary(summary)

    def _selection_units(self) -> str:
        """The units the selected channel's numbers are in, or ``""`` when undeclared.

        ``calibration.target_units`` when the user has set one, else the file's
        ``raw_units`` — the same two-step, in the same order, as
        ``trace_view._effective_units``, so the y-axis of the trace and the caption on
        the Summary cannot name one channel two ways. Never raises and never guesses: a
        panel with no selection, a document that has closed under it, or a channel index
        the meta does not carry all read as "undeclared", which the labels state.
        """
        if self._sel is None:
            return ""
        doc_id, channel = self._sel
        try:
            cal = self._session.calibration(doc_id, channel)
            info = self._session.document(doc_id).meta.channels[channel]
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            return ""
        return str(getattr(cal, "target_units", "") or getattr(info, "raw_units", "") or "")

    def _refresh_summary_from_table(self) -> None:
        """Curation edits the acceptance in place; recompute the accepted count."""
        if self._sel is None:
            return
        res = self._stored_detection()
        if res is not None and isinstance(res.payload, EventSet):
            self._fill_summary(res.payload)

    def _on_acceptance_changed(self, *args) -> None:
        """Any curation edit (from the table or a routed Event Inspector decision on the shared df)
        refreshes the accepted summary AND redraws the span overlays so only the accepted events
        stay shaded — curation flows to the trace, exactly as it flows to exports/plots. It also
        re-scores the detector, since accepting/rejecting a detection moves it between TP and FP.

        AND IT RECORDS THAT A PERSON RULED. ``accepted`` is only half a curation: the
        status a row reports is derived from (``review``, ``accepted``), so an event a
        curator accepted read 'unreviewed' — indistinguishable from one nobody had ever
        looked at — until this call landed. THE ONE SITE, deliberately: every single-row
        accept/reject in this panel goes through ``EventTable``'s model, whose write
        announces ``acceptanceChanged``, so the Event Inspector's routed decision
        (:meth:`set_event_accepted`) and a click on the table's own checkbox are the same
        door. A second call inside ``set_event_accepted`` would be a second writer for
        one gesture.

        BEFORE THE FORWARD, so the ruling reaches the undo stack. ``eventAcceptanceChanged``
        is what MainWindow checkpoints on (it is connected to this signal AFTER this slot,
        and Qt delivers direct connections in connection order), so the review write is in
        the frame the checkpoint snapshots. Recorded beside it instead, Ctrl+Z would walk
        the acceptance back and leave the reviewedness standing.
        """
        self._mark_rows_reviewed_by_id(args[0] if args else None)
        self._refresh_summary_from_table()
        self._redraw_spans_from_df()
        self._refresh_score()

    def _mark_rows_reviewed_by_id(self, event_id: object) -> None:
        """Record that a person ruled on ``event_id``, on the LIVE frame. Best effort.

        ``mark_reviewed`` takes POSITIONS, not labels (an event table's index is not
        guaranteed unique), so the id is resolved to positions here — every row carrying
        it, which is one row on any frame this app builds.

        Never raises into a curation gesture: a checkbox that throws because the frame is
        mid-rebuild would lose the accept the user actually asked for, and the reviewed
        half is the smaller loss of the two. ``mark_reviewed`` returns 0 and touches
        nothing when there is nothing to mark, so a no-op costs an empty checkpoint at
        worst — never a phantom undo step.
        """
        if event_id is None or self._sel is None:
            return
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return
        df = res.payload.df
        if "event_id" not in df.columns or len(df) == 0:
            return
        try:
            rows = np.flatnonzero(
                df["event_id"].to_numpy(dtype=np.int64) == int(event_id))
            if rows.size:
                clear_curation_conflicts(df, rows)
                mark_reviewed(df, rows=rows)
        except Exception:                                   # noqa: BLE001
            return

    # -- item 13: the full-recording map, cut by subpopulation -----------------------------------
    def set_subpop_labels(self, assignment: Mapping[int, int] | None) -> None:
        """Colour the event spans by subpopulation, or (``None``) go back to one amber shade.

        ``assignment`` maps ``event_id`` -> subpopulation id and is READ FROM a finished
        :class:`~nanoscope.analysis.subpop_report.SubpopReport` by ``MainWindow``, which is the
        only thing here that knows what a report is. Nothing is computed on this side and
        nothing is stored on the events frame: the colouring is a VIEW of an analysis result,
        and writing it into the shared df would make a provisional mixture assignment look like
        a measured column that survives to the exports.

        An id this map does not name (an event the gate dropped, or one added since the run)
        keeps the "in no class" grey :mod:`~nanoscope.gui.categorical` gives every negative id,
        rather than silently borrowing class 0's colour.
        """
        self._subpop_labels = None if assignment is None else {
            int(k): int(v) for k, v in dict(assignment).items()}
        self._redraw_spans_from_df()

    def subpop_labels(self) -> dict[int, int] | None:
        """The installed event_id -> subpopulation map, or ``None``."""
        return None if self._subpop_labels is None else dict(self._subpop_labels)

    def _redraw_spans_from_df(self) -> None:
        """Rebuild the SPAN overlay from the ACCEPTED events in the shared df.

        ``detect.schmitt`` emits exactly one overlay (a SPAN over every event), so replacing the
        panel's tracked span overlay with an accepted-only span is lossless. The single selected
        MARKER is tracked separately (``_marker_id``) and untouched here.

        WITH A SUBPOPULATION MAP INSTALLED THIS BECOMES ONE HINT PER CLASS, NOT ONE PER EVENT.
        That is the whole reason the colouring is affordable: the anti-``10k-LinearRegionItem``
        design gives one batched graphics item per hint, so four subpopulations over eleven
        hundred events cost four items and four paints -- the same order as the single amber
        overlay it replaces, and not the 1076 the obvious per-event colouring would cost. Every
        hint keeps ``label="events"``, because every one of them IS the detection's span overlay;
        the class travels in ``color`` and in ``name`` (see :class:`OverlayHint`).
        """
        if self._sel is None:
            return
        doc_id, channel = self._sel
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return
        df = res.payload.df
        self._clear_overlays()
        if len(df) == 0:
            return
        accepted = df["accepted"].to_numpy(dtype=bool)
        starts = df["start_idx"].to_numpy(dtype=np.int64)[accepted]
        stops = df["end_idx"].to_numpy(dtype=np.int64)[accepted]
        if starts.size == 0:
            return   # nothing accepted -> no shaded spans (an honest empty overlay)
        for hint in self._span_hints(df, accepted, starts, stops):
            self._overlay_ids.append(self._trace_view.add_overlay(doc_id, channel, hint))
        # ...and the fitted sub-state levels ON TOP of those spans, whenever a level fit
        # is stored for this selection. Tracked in the same _overlay_ids list, so every
        # path that clears/rebuilds the spans clears/rebuilds the levels with them.
        step = self._level_step_hint(df)
        if step is not None:
            self._overlay_ids.append(self._trace_view.add_overlay(doc_id, channel, step))

    #: The level ink's ONE name (overlays.STEP_LEVEL_HEX — the inspector's _CUSUM_HEX
    #: is the same import). render_step overrides the hint's colour with it anyway
    #: (how a level reads is the SURFACE's call, and that renderer says why), so this
    #: value no longer decides any paint — but a hint declaring the retired green
    #: would be a false record of what is drawn, and the one-name rule is exactly how
    #: the inspector and the trace were stopped from drifting apart about level colour.
    _LEVEL_HEX = STEP_LEVEL_HEX

    def _level_step_hint(self, df) -> OverlayHint | None:
        """One STEP hint for the stored CUSUM level fit, or ``None`` when there is none.

        Read from the SAME store entry the inspector and the export read
        (``"fit.levels"``), passed through the SAME reconciler the inspector and the
        export pass it through (a level group whose parent span moved is dropped, not
        drawn wrong), and filtered to the ACCEPTED events only — the exact set whose
        spans are shaded above. Until this producer existed, ``OverlayKind.STEP`` was
        declared in the enum and emitted by nothing: sub-state levels were visible in
        the Event Inspector and invisible on the main trace (owner note #18).
        """
        if self._sel is None:
            return None
        doc_id, channel = self._sel
        res = self._session.results.get(doc_id, channel, "fit.levels")
        if res is None:
            return None
        levels = res.payload
        if levels is None or getattr(levels, "empty", True):
            return None
        from ..analysis.cusum import reconcile_levels
        kept, _dropped = reconcile_levels(levels, df)
        if kept is None or kept.empty:
            return None
        acc_ids = df.loc[df["accepted"].astype(bool), "event_id"].astype(np.int64)
        kept = kept[kept["event_id"].astype(np.int64).isin(set(acc_ids.tolist()))]
        if kept.empty:
            return None
        return OverlayHint(
            OverlayKind.STEP,
            starts=kept["start_idx"].to_numpy(dtype=np.int64),
            stops=kept["end_idx"].to_numpy(dtype=np.int64),
            values=kept["level_pA"].to_numpy(dtype=np.float64),
            label="levels", color=self._LEVEL_HEX, name="sub-state levels",
        )

    def redraw_overlays(self) -> None:
        """Public entry point (MainWindow) after the stored level fit changed — fitted
        ("fit.levels" added) or retired (dropped). Needed because resultsChanged only
        triggers a redraw for detection modules, and a store ``drop()`` emits no signal
        at all — without this the STEP overlay would appear/vanish only on the next
        curation edit."""
        self._redraw_spans_from_df()

    def _span_hints(self, df, accepted: np.ndarray, starts: np.ndarray,
                    stops: np.ndarray) -> list[OverlayHint]:
        """One SPAN hint for the whole detection, or one per subpopulation when coloured.

        The groups are emitted in ASCENDING CLASS ORDER with the unassigned group last, the
        same order :func:`~nanoscope.gui.categorical.category_counts` lists a legend in -- so
        the trace and the report window enumerate the classes the same way.
        """
        if not self._subpop_labels:
            return [OverlayHint(OverlayKind.SPAN, starts=starts, stops=stops, label="events")]
        ids = df["event_id"].to_numpy(dtype=np.int64)[accepted]
        # -1 is categorical's "in no class" sentinel and is what an event the report never
        # saw must be drawn as; it is NOT a class the mixture found.
        classes = np.array([self._subpop_labels.get(int(e), -1) for e in ids], dtype=np.int64)
        out: list[OverlayHint] = []
        for cid in sorted({int(c) for c in classes.tolist()}, key=lambda c: (c < 0, c)):
            mask = classes == cid
            out.append(OverlayHint(
                OverlayKind.SPAN,
                starts=starts[mask], stops=stops[mask], label="events",
                color=category_colour(cid).name(),
                name=(f"subpop {cid}" if cid >= 0 else "not in the report"),
            ))
        return out

    # -- detector score (vs the user's curation) -------------------------------------------------
    def _refresh_score(self) -> None:
        """Recompute the detector-vs-curation score for the CURRENT detection and push it
        to the ScorePanel (or blank it when there is no detection). Called on every result
        change, curation edit, mask edit, batch curation and hand-added missed event."""
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            self._score.clear()
            return
        self._score.set_score(score_detection(res.payload.df))

    def current_score(self):
        """The current detection's :class:`ScoreResult`, or None. Public accessor for
        MainWindow / tests; the panel stays the owner of the results/selection."""
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return None
        return score_detection(res.payload.df)

    # -- add a missed event (the detector missed it; the user brackets it with cursors) ----------
    def add_current_selection_as_event(self) -> int | None:
        """Append the MeasurePanel's current cursor selection to the active detection as an
        event the detector MISSED.

        The bracketed span ``[start, end)`` on the active (doc_id, channel) is characterized
        exactly like a detected event — a local open-pore baseline (median of the flanks) and
        flank noise, with the detector's polarity / filter-rise taken from the result
        provenance — then appended to the detection's ``EventSet.df`` with ``accepted=True``,
        ``added=True`` and a fresh unique ``event_id``. The result is re-installed (so the
        table, span overlays, score, inspector and exports all pick up the new row) and the new
        ``event_id`` is returned, or ``None`` when there is nothing to add.
        """
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return None
        if res.module_id == CONSENSUS_MODULE_ID:
            self.runRefused.emit(_RECONCILED_EDIT_REFUSAL)
            return None
        if self._sel is None:
            return None
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return None
        # The cursors have to be ON THIS RECORDING. ``measure_bounds`` hands back sample
        # indices with no recording attached, and the Measure region may be sitting on
        # another pane — appending those indices here would add an event bracketing
        # whatever happens to live at those sample numbers in THIS trace. Same guard, same
        # reason, as MainWindow's "spectrum of the selection" action.
        target = self._trace_view.measure_target()
        bounds = self._trace_view.measure_bounds()
        if bounds is None or target is None or target != (doc_id, channel):
            return None

        ch = self._session.calibrated(doc_id, channel)
        data = np.asarray(ch.data)
        n = int(data.size)
        fs = ch.sample_rate_hz
        if fs is None or n == 0:
            return None            # dwell is undefined without a time base
        start = int(np.clip(int(bounds[0]), 0, n))
        end = int(np.clip(int(bounds[1]), 0, n))
        if end <= start:
            return None

        events = res.payload
        df = events.df
        ensure_added_column(df)
        recording, pore_id = self._missed_identity(df, doc_id)
        # Measure the hand-added event on the trace the DETECTOR measured on, so its row
        # is comparable with the detected rows it is being appended to (see
        # :meth:`_active_signal`). Falls back to the raw channel when the conditioning
        # cannot be rebuilt.
        conditioned, _ = self._active_signal(events.provenance or {})
        crow = self._characterize_missed(
            conditioned if conditioned is not None else data,
            float(fs), start, end, events.provenance or {},
            ch.units or "pA", recording, pore_id)
        if crow is None:
            return None

        new_id = (int(df["event_id"].max()) + 1) if len(df) else 0
        row = crow.to_dict()
        row["event_id"] = int(new_id)
        row[ADDED_COLUMN] = True                      # this is a detector MISS
        # ``accepted`` / ``flag`` come from the rebuild, NOT from a blanket
        # ``accepted = True`` here. A hand-added event is real by definition, and it
        # arrives with ``accepted=True`` for exactly that reason -- but if the user's
        # selection runs off the end of the recording, the row is boundary-clipped and
        # its dwell is a lower bound, and asserting it is fine would be the same lie
        # the merge used to tell.

        # reindex, NOT `[df.columns]`: the detection frame now carries the detector's
        # ORIGINAL bounds (`orig_start_idx`/`orig_end_idx`, stamped at detection so the
        # inspector's Revert has something to revert TO), and `row` comes from
        # `characterize`, whose output is `df[list(SCHEMA)]` and so has neither column.
        # A column SELECTION raises KeyError on them; reindex fills NA, which is the right
        # answer twice over -- a hand-added event HAS no detector bounds, and
        # `events.original_bounds` reads NA as "unknown" and disables Revert for that row.
        combined = pd.concat(
            [df, pd.DataFrame([row]).reindex(columns=df.columns)], ignore_index=True)
        combined = combined.astype({c: df[c].dtype for c in df.columns})
        # A HAND-ADDED EVENT IS A RULING BY DEFINITION -- somebody looked at the trace,
        # drew a span and said "the detector missed this". `ensure_source_status` stopped
        # inferring reviewedness from ``added`` (it may not promote a merged row), so
        # without this the row the user just created reads status 'unknown'. Marked on
        # the frame that is about to be installed, so the ONE ``add_result`` below carries
        # both writes and the checkpoint it drives sees the row whole.
        mark_reviewed(combined, rows=np.array([len(combined) - 1], dtype=np.int64))
        new_events = EventSet(combined, dict(events.provenance or {}))
        # Re-install under the SAME (doc, channel, module) key: resultsChanged drives
        # _show_result (table + spans + score) and MainWindow re-feeds the inspector.
        # Through `reconcile_result`, not a bare `replace`: the table just grew a row,
        # so every row-derived count and sentence on the result is re-read from it.
        self._session.add_result(reconcile_result(res, new_events))
        return int(new_id)

    @staticmethod
    def _missed_identity(df: pd.DataFrame, doc_id: str) -> tuple[str, str]:
        """The recording + pore_id to stamp on a hand-added event: copy the detection's own
        (from any existing row) so the added event is never a lone stranger; fall back to the
        doc_id when the detection is empty."""
        if len(df):
            return str(df["recording"].iloc[0]), str(df["pore_id"].iloc[0])
        return doc_id, doc_id

    def _characterize_missed(self, data: np.ndarray, fs: float, start: int, end: int,
                             provenance: dict[str, Any], units: str,
                             recording: str, pore_id: str) -> pd.Series | None:
        """Characterize a single hand-added event at ``[start, end)``.

        Delegates to :func:`~nanoscope.analysis.characterize.recharacterize_span`, the ONE
        row-rebuild path — the same call the gap merge and the Event Inspector's mask resize
        make, so a hand-added event is measured on exactly the same basis as a merged or a
        resized one. This method was a third private copy of that padded-window measurement;
        the copies were what let the three paths disagree about ``flag`` and ``accepted``.

        ``prior=None`` says this row replaces nothing, so nothing is inherited: the automatic
        rules are judged purely on the new span. Returns the single SCHEMA row as a Series with
        ABSOLUTE indices, or None if the window is degenerate.

        ``data`` is expected to be the CONDITIONED trace (``_active_signal``), not ``ch.data``."""
        try:
            out = recharacterize_span(
                data, float(fs), start, end,
                provenance=provenance,
                prior=None,
                recording=recording,
                pore_id=pore_id,
                units=units or "pA",
            )
        except (ValueError, IndexError):
            return None
        if len(out) == 0:
            return None
        return out.iloc[0]

    # -- post-detection sub-event merge (curation-style; mutates the active EventSet) ------------
    @property
    def merge_controls(self) -> MergePanel:
        """The owned sub-event merge controls. Exposed so MainWindow (the sole wirer) can route the
        whole-trace / region merge signals onto this panel's ``merge_*`` engine methods — the panel
        stays the owner of the results/selection + curation refresh (exactly as with ``measure_panel``)."""
        return self._merge

    def merge_window_s(self) -> float:
        """The merge window (seconds) set on the controls, so the Event Inspector's 'Merge with
        next' stitches with the SAME window as the whole-trace 'Merge all'."""
        return self._merge.window_s()

    def _active_signal(
        self, provenance: dict[str, Any] | None = None
    ) -> tuple[np.ndarray | None, float | None]:
        """The trace a re-characterization must measure on, + the sample rate.

        NOT the raw channel. ``detect_events`` measured every reported ``depth_pA`` /
        ``baseline_pA`` / ``rise_s`` on ``lowpass(preprocess_for_detection(raw))``, so a
        merge that re-characterizes the union span on ``ch.data`` reports a different
        quantity under the same column names as the rows it sits next to. Measured on the
        user's -600 mV recording at the default 10 kHz Bessel, merged ``frac_depth`` came
        out 2.6% off at the median (and further with a slower filter, up to a sign flip).
        :func:`~nanoscope.analysis.modules.detection_signal` rebuilds the conditioning
        from the detection's own provenance; a detection that recorded no preprocessing
        is left alone.

        ``(None, None)`` when there is no live target or the rate is unknown (dwell —
        hence a merge — is undefined then). Cached: ~0.12 s for a 3 M-sample recording,
        and the merge entry points are on the GUI thread.
        """
        if self._sel is None:
            return None, None
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return None, None
        ch = self._session.calibrated(doc_id, channel)
        data = np.asarray(ch.data)
        fs = ch.sample_rate_hz
        if fs is None or data.size == 0:
            return None, None
        if provenance is None:
            return data, float(fs)

        try:
            calib = self._session.calibration(doc_id, channel).fingerprint()
        except Exception:  # noqa: BLE001 — a missing calibration must not break a merge
            calib = None
        key = (doc_id, channel, int(data.size), float(fs), calib,
               detection_signal_key(provenance))
        if self._cond_cache is not None and self._cond_cache[0] == key:
            return self._cond_cache[1], float(fs)
        try:
            out = detection_signal(data, float(fs), provenance)
        except Exception:  # noqa: BLE001 — a bad stored param must not break a merge
            out = data
        self._cond_cache = (key, out)
        return out, float(fs)

    def _install_merged(self, res: AnalysisResult, new_events: EventSet) -> None:
        """Re-install a merged EventSet under the SAME (doc, channel, module) key. The resulting
        ``resultsChanged`` drives ``_show_result`` (table + spans + summary + score) and MainWindow
        re-feeds the inspector — the identical refresh path accept/reject and hand-added events use,
        so a merge is just another curation edit downstream.

        Through `modules.reconcile_result`, never a bare ``replace(res, payload=...)``: a
        merge shrinks the frame, and a result whose ``provenance['n_events']`` and
        composed warnings still describe the pre-merge table is what put "1 of 317
        event(s)" over 247 rows and "71 ... curated OUT" into the subpopulation report
        (S7-DET4). All three merge entry points land here, so all three are covered."""
        self._session.add_result(reconcile_result(res, new_events))

    def merge_all_subevents(self, window_s: float) -> MergeReport:
        """Merge sub-events across the WHOLE active detection within ``window_s`` — the
        'Merge all' entry point. **OFF THE GUI THREAD.**

        MEASURED 3.3-8.7 s ON THE OWNER'S OWN RECORDINGS -- comparable to the
        detector's own runtime on 9 M samples, because it re-characterizes every
        surviving span against the conditioned trace. It used to run right here, on
        the GUI thread, with no progress bar, no busy cursor and no status message:
        the window simply stopped repainting for up to nine seconds after a click. A
        frozen window is a window the user clicks again, and this app has already
        written that lesson down twice (see :meth:`remeasure_active_detection`, whose
        docstring says it in as many words about a 3.3 s job).

        It now goes through the same controller everything else expensive uses, which
        is what the finding meant by "the app has a cancellation system it is not
        using here": MainWindow's status-bar progress and its Cancel button are wired
        to ``AnalysisController.started`` / ``finished``, so moving the work here buys
        all three missing affordances at once, plus the ambient cancel token the
        kernels already poll.

        Returns ``'started'`` (or a refusal) IMMEDIATELY; the real outcome arrives on
        :attr:`mergeFinished` and MainWindow posts it with the same
        ``_post_merge_status`` it always did.

        The other two merge entry points stay synchronous ON PURPOSE. Their cost is
        bounded by a selection or by one forward run, not by the table: merging a
        region of a handful of events is milliseconds, and putting a controller job
        (and its refusal-when-busy) in front of it would buy nothing and cost the
        Event Inspector's "merge, then advance" its immediacy.
        """
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return MergeReport("no_detection", 0, 0, window_s=float(window_s))
        if res.module_id == CONSENSUS_MODULE_ID:
            n = len(res.payload.df)
            return MergeReport("reconciled", n, n, window_s=float(window_s))
        signal, fs = self._active_signal(res.payload.provenance)
        if signal is None or fs is None:
            return MergeReport("no_detection", 0, 0, window_s=float(window_s))
        before = len(res.payload.df)
        if self._pending is not None:
            return MergeReport("busy", before, before, window_s=float(window_s))
        # FROZEN AT DISPATCH, for the reason `frozen_for_dispatch` exists: the table,
        # the Event Inspector and Undo/Redo all curate `res.payload.df` IN PLACE and it
        # stays live for the seconds this runs. Handing the worker that frame would put
        # a pandas frame under two threads.
        events = frozen_for_dispatch(res.payload)

        def job():
            return merge_by_gap(events, signal, fs, max_gap_s=float(window_s))

        if not self._controller.run_callable(job, label="merge_all"):
            return MergeReport("busy", before, before, window_s=float(window_s))
        self._pending, self._pending_sel = "merge_all", self._sel
        # What the answer will be checked against on the way in. Geometry AND curation:
        # the merged table is built from the dispatch-time frame, so installing it over
        # a frame the user has edited since would silently undo the edit.
        self._merge_job = (
            res.doc_id, res.channel, res.module_id, before, float(window_s),
            events.df["event_id"].to_numpy(dtype=np.int64).copy(),
            events.df[["start_idx", "end_idx"]].to_numpy(dtype=np.int64).copy(),
            # ``merge_guard_snapshot``, NOT ``curation_snapshot``: the guard asks "did
            # the user rule on a row while this ran", and the answer is ``accepted`` +
            # ``flag``. ``curation_snapshot`` widened to include ``review`` for the
            # re-measure install (which CAN carry a late review forward), and read here
            # a single "mark reviewed" click during a Merge all made the guard say the
            # rows moved and threw a finished merge away. Must change together with the
            # comparison in `_merge_outcome` -- the snapshot and the diff are one shape.
            merge_guard_snapshot(events.df),
        )
        return MergeReport("started", before, before, window_s=float(window_s))

    def _finish_merge_all(self, obj: object) -> None:
        """Install a finished 'Merge all', or drop it and say so. Emits exactly one
        :attr:`mergeFinished`, on every path.

        THREE THINGS CAN HAVE CHANGED under a job that takes seconds, and the answer
        was computed from none of them:

        * the recording or the detection is gone — drop;
        * the rows moved (an Undo, a Redo, a mask drag, a hand-added event, another
          merge) — drop, because the merged table's rows are not those rows;
        * **the user curated** — drop. A merge is built from the dispatch-time frame
          and installing it would replace an accept/reject the user made while it ran.
          The re-measure path can carry late curation forward because it keeps every
          row; a merge deletes rows, so there is no honest row to carry a decision
          onto, and the only truthful options are "drop" and "say so".
        """
        job, self._merge_job = self._merge_job, None
        if job is None:
            return
        doc_id, channel, module_id, before, window_s, ids, geom, curation = job
        report = self._merge_outcome(
            obj, doc_id, channel, module_id, before, window_s, ids, geom, curation)
        self.mergeFinished.emit(report)

    def _merge_outcome(self, obj: object, doc_id: str, channel: int, module_id: str,
                       before: int, window_s: float, ids, geom, curation) -> MergeReport:
        if not (isinstance(obj, tuple) and len(obj) == 2):
            return MergeReport("dropped", before, before, window_s=window_s)
        if not self._session.has_document(doc_id):
            return MergeReport("dropped", before, before, window_s=window_s)
        res = self._session.results.get(doc_id, channel, module_id)
        if res is None or not isinstance(res.payload, EventSet):
            return MergeReport("dropped", before, before, window_s=window_s)
        live = res.payload.df
        moved = (len(live) != ids.size
                 or not np.array_equal(live["event_id"].to_numpy(dtype=np.int64), ids)
                 or not np.array_equal(
                     live[["start_idx", "end_idx"]].to_numpy(dtype=np.int64), geom)
                 or not merge_guard_snapshot(live).equals(curation))
        if moved:
            return MergeReport("dropped", before, len(live), window_s=window_s)
        new_events, n = obj
        # ``merge_by_gap`` returns a MergeOutcome — still the (eventset, n_merged) pair,
        # with the occupancy refusals attached. A plain tuple (an older cached job, a
        # test double) reads as zero refusals rather than raising.
        refused = int(getattr(obj, "n_refused", 0))
        if n == 0:
            # Refused is NOT a no-op: "No sub-events within 10 ms to merge" would be a
            # lie about a table where ten runs were within the window and every one of
            # them was mostly open pore.
            status = "low_occupancy" if refused else "noop"
            return MergeReport(status, before, before, window_s=window_s,
                               n_refused_low_occupancy=refused)
        self._install_merged(res, new_events)
        return MergeReport("ok", before, len(new_events.df), window_s=window_s,
                           n_refused_low_occupancy=refused)

    def merge_selection_subevents(self) -> MergeReport:
        """Merge every detected event overlapping the on-trace Measure selection into one (the
        'Merge events in selection' entry point; an explicit selection overrides the window). Needs
        the Measure region active ON THIS TARGET — returns ``no_selection`` when there is none.

        The selection has to be on the recording being merged, and that is not implied by there
        being a selection at all: :meth:`TraceView.measure_bounds` returns SAMPLE INDICES, and the
        Measure region may well be sitting on another recording's pane (MainWindow's PSD action
        guards the identical pattern with :meth:`TraceView.measure_target`, for the identical
        reason). Merging one recording's indices into another recording's event table would
        collapse whichever rows happen to live at those sample numbers — a silent, wrong,
        undo-able-but-invisible edit."""
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return MergeReport("no_detection", 0, 0)
        if res.module_id == CONSENSUS_MODULE_ID:
            n = len(res.payload.df)
            return MergeReport("reconciled", n, n)
        target = self._trace_view.measure_target()
        bounds = self._trace_view.measure_bounds()
        if bounds is None or target is None or self._sel is None or target != self._sel:
            # A region on ANOTHER recording is reported as "no selection" deliberately:
            # from this target's point of view there is none, and the status line's
            # "drag a region on the trace first" is the action that fixes it either way.
            return MergeReport("no_selection", 0, 0)
        signal, fs = self._active_signal(res.payload.provenance)
        if signal is None or fs is None:
            return MergeReport("no_detection", 0, 0)
        i0, i1 = int(bounds[0]), int(bounds[1])
        before = len(res.payload.df)
        out = merge_region(res.payload, signal, fs, t0_s=i0 / fs, t1_s=i1 / fs)
        new_events, n = out
        refused = int(getattr(out, "n_refused", 0))
        if n == 0:
            # The drag covered two or more events and they were mostly open pore: say
            # that, rather than "fewer than two events overlap the selection".
            status = "low_occupancy" if refused else "noop"
            return MergeReport(status, before, before, n_refused_low_occupancy=refused)
        self._install_merged(res, new_events)
        return MergeReport("ok", before, len(new_events.df))

    def merge_event_with_next(self, event_id: int, window_s: float) -> MergeReport:
        """Merge the given event with the following event(s) within ``window_s`` (the Event
        Inspector's 'Merge with next'): walk the forward run under the SAME gap rule, merge just
        that local run (``index_range`` keeps the rest untouched) and report the ``event_id`` to
        advance to. A no-op when nothing sits within the window ahead.

        THE WALK IS BOUNDED BY OCCUPANCY, not only by the gap. "Merge with next" is one click on
        one event, but the run it walks is transitive like every other merge path: on a regularly
        spaced train the loop ran to the end of the recording, and the single click produced one
        row spanning hundreds of milliseconds. The run therefore stops growing as soon as taking
        the next event would drop it below
        :data:`~nanoscope.analysis.merge.MIN_MERGE_OCCUPANCY` — the user gets the fragments that
        really are one translocation and keeps the rest, instead of an all-or-nothing refusal on a
        button whose whole point is to work event by event."""
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return MergeReport("no_detection", 0, 0)
        if res.module_id == CONSENSUS_MODULE_ID:
            n = len(res.payload.df)
            return MergeReport("reconciled", n, n, window_s=float(window_s))
        signal, fs = self._active_signal(res.payload.provenance)
        if signal is None or fs is None:
            return MergeReport("no_detection", 0, 0)
        df = res.payload.df
        before = len(df)
        order = np.argsort(df["start_idx"].to_numpy(dtype=np.int64), kind="stable")
        starts = df["start_idx"].to_numpy(dtype=np.int64)[order]
        ends = df["end_idx"].to_numpy(dtype=np.int64)[order]
        ids = df["event_id"].to_numpy(dtype=np.int64)[order]
        pos = np.flatnonzero(ids == int(event_id))
        if pos.size == 0:
            return MergeReport("no_detection", before, before)
        p = int(pos[0])
        max_gap = gap_window_samples(window_s, fs)
        run_end = p
        # ``cur_end`` is the running MAXIMUM end of the run, matching
        # ``group_starts_by_gap``: a curated table can hold a short event nested inside a
        # long one, and comparing against the previous row's end alone would read a huge
        # gap after the nested one and stop the walk early.
        cur_end = int(ends[p])
        refused = 0
        while run_end + 1 < starts.size and (starts[run_end + 1] - cur_end) <= max_gap:
            if (span_occupancy(starts[p:run_end + 2], ends[p:run_end + 2])
                    < MIN_MERGE_OCCUPANCY):
                refused = 1
                break
            run_end += 1
            cur_end = max(cur_end, int(ends[run_end]))
        if run_end == p:
            # Nothing was taken: either nothing was within the window (noop) or the one
            # candidate ahead would have made a span that is mostly open pore.
            status = "low_occupancy" if refused else "noop"
            return MergeReport(status, before, before, int(event_id),
                               n_refused_low_occupancy=refused)
        i0, i1 = int(starts[p]), cur_end
        out = merge_by_gap(
            res.payload, signal, fs, max_gap_s=float(window_s), index_range=(i0, i1))
        new_events, n = out
        refused += int(getattr(out, "n_refused", 0))
        if n == 0:
            status = "low_occupancy" if refused else "noop"
            return MergeReport(status, before, before, int(event_id),
                               n_refused_low_occupancy=refused)
        self._install_merged(res, new_events)
        return MergeReport("ok", before, len(new_events.df), self._event_after(new_events.df, i0),
                           n_refused_low_occupancy=refused)

    @staticmethod
    def _event_after(new_df: pd.DataFrame, merged_start: int) -> int:
        """The ``event_id`` to browse to after a 'Merge with next': the first event starting at or
        after the merged event's end, else the merged event itself (it was the last)."""
        s = new_df["start_idx"].to_numpy(dtype=np.int64)
        e = new_df["end_idx"].to_numpy(dtype=np.int64)
        ids = new_df["event_id"].to_numpy(dtype=np.int64)
        if ids.size == 0:
            return -1
        mpos = np.flatnonzero(s == int(merged_start))
        if mpos.size == 0:
            return int(ids[0])
        m = int(mpos[0])
        after = np.flatnonzero(s >= int(e[m]))
        return int(ids[after[0]]) if after.size else int(ids[m])

    # -- migration: re-measure against the configured estimator ----------------------------------
    def remeasure_active_detection(self) -> str:
        """Re-measure the shown detection's amplitudes against the live estimator.

        THE OFFERED BUTTON of §4.2.3, and everything about it is deliberate:

        * it is only ever reached by a CLICK -- no code path calls it on load, on
          selection, or after a config edit, because a number that changes 15x
          between two openings of the same project with no prompt is
          indistinguishable from data corruption;
        * it re-measures, it does not re-detect, so the curated event set survives;
        * it runs off the GUI thread, because 1641 events on the owner's 3 M-sample
          apo file cost ~3.3 s and a frozen window during a destructive-looking
          operation is how a user concludes the app has hung and force-quits it;
        * it is cancellable, through the ambient token the controller installs.

        Returns a status token for MainWindow's status line: ``'started'``,
        ``'no_detection'``, ``'no_signal'``, ``'no_config'`` or ``'busy'``.
        """
        if self._pending is not None:
            return self._refuse_baseline_job("busy")
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return self._refuse_baseline_job("no_detection")
        if res.module_id == CONSENSUS_MODULE_ID:
            return self._refuse_baseline_job("reconciled")
        signal, fs = self._active_signal(res.payload.provenance)
        if signal is None or fs is None:
            return self._refuse_baseline_job("no_signal")
        cfg = self._live_baseline_config()
        if cfg is None:
            return self._refuse_baseline_job("no_config")
        if not self._launch_remeasure(res, signal, float(fs), cfg, "remeasure"):
            # The run gate refused (a batch pass is in flight). Same blocker, same word:
            # "busy" already means "another analysis is still running".
            return self._refuse_baseline_job("busy")
        self._say_baseline(
            "started", f"Re-measuring {len(res.payload.df)} events against "
                       f"{cfg.describe()}…")
        return "started"

    def compare_baseline_estimators(self) -> str:
        """Run §4.2.7's comparison and show it. Installs nothing by itself.

        Same job, same code path, same numbers as the button it sits beside -- the
        dialog is handed the comparison that came out of the very rows a re-measure
        would install, so it cannot describe a computation the button would not do.
        """
        if self._pending is not None:
            return self._refuse_baseline_job("busy")
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return self._refuse_baseline_job("no_detection")
        if res.module_id == CONSENSUS_MODULE_ID:
            return self._refuse_baseline_job("reconciled")
        signal, fs = self._active_signal(res.payload.provenance)
        if signal is None or fs is None:
            return self._refuse_baseline_job("no_signal")
        cfg = self._live_baseline_config()
        if cfg is None:
            return self._refuse_baseline_job("no_config")
        if not self._launch_remeasure(res, signal, float(fs), cfg, "compare"):
            return self._refuse_baseline_job("busy")
        self._say_baseline(
            "started", "Comparing open-pore estimators on this recording…")
        return "started"

    #: The refusals both baseline entry points share, in the user's terms. One
    #: sentence per blocker, named here so the menu path and the banner path cannot
    #: describe the same refusal two different ways.
    _BASELINE_REFUSALS = {
        "busy": "Another analysis is still running — wait for it to finish, or "
                "cancel it.",
        "no_detection": "Run detection on a current channel first — there are no "
                        "events to measure.",
        "no_signal": "This recording has no sample rate, so an open-pore window in "
                     "seconds has no meaning here.",
        "no_config": "The open-pore baseline controls are not set — open the "
                     "Preprocess tab and choose (or probe for) a window.",
        "reconciled": _RECONCILED_BASELINE_REFUSAL,
    }

    def _say_baseline(self, token: str, message: str) -> None:
        """The ONE place `baselineJobStatus` is emitted. Every other spelling is a
        path that can forget to say it finished, and MainWindow leaves the last
        sentence in the status bar until something replaces it -- so a silent exit
        reads as a job still running. An AST guard in
        ``tests/gui/test_baseline_migration.py`` keeps this the only emitter."""
        self._baseline_said = True
        self.baselineJobStatus.emit(token, message)

    def _refuse_baseline_job(self, token: str) -> str:
        """Say why, then return the token. Never a silent no-op: a click that does
        nothing and explains nothing is indistinguishable from a hang."""
        self._say_baseline(token, self._BASELINE_REFUSALS.get(token, ""))
        return token

    def _launch_remeasure(self, res: AnalysisResult, signal: np.ndarray, fs: float,
                          config: Any, mode: str) -> bool:
        """Start the shared re-measure job off-thread; True when it actually started.
        ``mode`` decides what is done with the answer when it lands (install it, or only
        report it).

        False means the run gate refused (a batch pass is in flight). The caller must then
        NOT say "started" — this surface's contract is exactly one terminal message per
        started job, and a "Re-measuring 1641 events…" that no job will ever finish is the
        stuck-status-line defect that contract exists to prevent."""
        # FROZEN AT DISPATCH. `res.payload`'s frame is the one the results table, the
        # Event Inspector and Undo/Redo curate IN PLACE, and it stays live for the 3-4 s
        # this runs; handing it to the worker put a pandas frame under two threads.
        # After this line, everything the user does is unambiguously an edit made
        # DURING the job, which is what lets the install have one rule for all of them.
        events = frozen_for_dispatch(res.payload)
        # The detector's OWN polarity, not one measured here: a re-measure must
        # reproduce the frame the rows were detected in. `derive_polarity` reads +1
        # on the owner's raw drifting traces, and a one-sided mask pointed the wrong
        # way moves the baseline by up to 312 pA.
        pol = events.provenance.get("polarity") if events.provenance else None
        source = self._baseline.param_source()
        # The table stays LIVE for the seconds this runs, and the model curates the
        # shared frame in place. Snapshot what the curator had said when the job was
        # dispatched, so the install can carry over anything he says while it runs
        # instead of overwriting it with a three-second-old view of his intent.
        # ...and the params block the run is measuring with, captured at the same
        # instant as the config so the two cannot disagree. `BaselinePanel.params()`
        # is the ONE definition of which keys are parameters and which are derived
        # (the fingerprint is derived, and a derived value in a params dict is a
        # value that can contradict the parameters beside it).
        params_snapshot = dict(self._baseline.params())
        curation = curation_snapshot(events.df)

        def job():
            from ..analysis.remeasure import remeasure_events
            return remeasure_events(
                events, signal, fs, config=config,
                polarity=int(pol) if pol in (-1, 1) else None,
                param_source=source,
            )

        # Nothing about the in-flight job is recorded until the job exists. A refused
        # start that had already overwritten ``_remeasure_target`` / ``_remeasure_curation``
        # would leave a LATER answer to be installed against this one's snapshot.
        if not self._controller.run_callable(job, label=f"baseline_{mode}"):
            return False
        self._pending, self._pending_sel = mode, self._sel
        self._remeasure_target = (res.doc_id, res.channel, res.module_id)
        self._remeasure_curation = curation
        self._remeasure_params = params_snapshot
        return True

    def _on_remeasure_finished(self, result: Any, install: bool) -> None:
        """A finished re-measure lands here, and says so exactly once.

        The wrapper is the invariant, not politeness: every entry point emits
        ``baselineJobStatus('started', ...)`` and MainWindow puts that in the status
        bar, so a path that finishes without a terminal message leaves "Comparing
        open-pore estimators on this recording…" on screen for the rest of the
        session. Measured: the compare path emitted ONE signal ever when the dialog
        was dismissed rather than accepted, and four more exits (recording closed,
        detection replaced, target lost) said nothing at all. A guarantee that has to
        be repeated at six ``return`` statements is a guarantee one of them will
        forget, so it is written once here.
        """
        self._baseline_said = False
        try:
            self._remeasure_finished(result, install)
        finally:
            if not self._baseline_said:
                self._say_baseline(
                    "dropped",
                    "The re-measure finished, but the detection it was measuring is "
                    "no longer on screen. Nothing changed.")

    def _remeasure_finished(self, result: Any, install: bool) -> None:
        """The body. See `_on_remeasure_finished` for why it is wrapped.

        Three things can have changed under a job that takes seconds, and each is
        answered rather than assumed:

        * **The user curated.** The table stays live throughout, so the live rows are
          diffed against the snapshot taken at dispatch, per column, and anything he
          said in the meantime is carried onto the result by
          `remeasure.merge_late_curation` -- a retraction as faithfully as a
          rejection, while the row's freshly RE-DERIVED automatic tokens stay the new
          measurement's.
        * **The rows moved, or are not the same rows** -- an Undo, a Redo, an Event
          Inspector mask drag, a merge, a hand-added event, a fresh detection. Those
          are not what was measured, so the answer is dropped and SAID; installing
          over them would undo the edit under a status line denying it.
        * **The recording or the detection is gone.** The answer is dropped, the same
          rule the surrogate studies follow.

        The target is re-resolved from the STORE rather than from the result object
        the job captured, for exactly the first reason.
        """
        target, self._remeasure_target = self._remeasure_target, None
        before, self._remeasure_curation = self._remeasure_curation, None
        params_block, self._remeasure_params = self._remeasure_params, None
        if target is None:
            return
        doc_id, channel, module_id = target
        if not self._session.has_document(doc_id):
            return
        res = self._session.results.get(doc_id, channel, module_id)
        if res is None or not isinstance(res.payload, EventSet):
            return
        if not install:
            # Terminate the status line BEFORE the modal dialog: `exec()` blocks, so
            # a message emitted after it would leave "Comparing…" standing for as
            # long as the report is open, and would never arrive at all if the user
            # dismissed it (the measured defect -- one signal, ever).
            c = result.comparison
            self._say_baseline(
                "compared",
                f"Compared {c.n_events} events: median depth "
                f"{c.median_old_depth_pA:.4g} → {c.median_new_depth_pA:.4g} pA "
                f"({c.median_ratio:.3g}×). Nothing changed.")
            self._show_comparison(result.comparison)
            return

        df = result.events.df
        # UNCONDITIONAL. `merge_late_curation` is not only the curation carry -- it is
        # where the answer is checked against the rows the store holds NOW
        # (`remeasure.assert_answer_describes`), and that check is the only thing that
        # makes the sentence this method ends with true rather than asserted. Behind
        # `if before is not None` the whole precondition was skippable by whichever
        # path lost the snapshot; a null snapshot is the honest stand-in for "no
        # curation was made", and it leaves the geometry check running.
        if before is None:
            before = curation_snapshot(res.payload.df)
        try:
            df = merge_late_curation(df, res.payload.df, before)
        except ValueError as exc:
            # The rows MOVED, or are not the same rows (an Undo, a Redo, an Event
            # Inspector mask drag, a merge, a hand-added event, a fresh detection).
            # Those are not what was measured, so the answer is dropped and SAID --
            # installing over them would silently undo the edit the user just made,
            # under a status line claiming the geometry was unchanged.
            self._say_baseline("superseded", f"Re-measure discarded: {exc}")
            return

        # Through `reconcile_result`, so the result's warnings are REBUILT from the rows
        # and the provenance rather than replaced by the new baseline's list alone.
        # ``warnings=tuple(result.comparison.warnings)`` here deleted every advisory the
        # detector had authored -- filter-limited, boundary-clipped, saturated-rail,
        # notch-settling, smoothing, normalized-units, detrend, polarity -- and no
        # surface restored them; measured: four warnings before the click, one after,
        # with the three filter-limited rows still accepted and still exported (S7
        # MISSED-1). The new estimator's own advisories arrive through
        # ``provenance['baseline_warnings']``, which the engine now stamps.
        new_res = reconcile_result(
            res, EventSet(df, result.events.provenance),
            # The params a "Re-run" would reproduce this result with. Left alone, they
            # would go on naming the estimator that has just been replaced, so a
            # later calibration re-run would silently measure the depths back down
            # again -- and `params` and `provenance` would disagree about one result.
            params={**{k: v for k, v in (res.params or {}).items()
                       if not str(k).startswith("baseline_")},
                    **(params_block or {})},
            baseline_fingerprint=result.fingerprint,
        )
        # Through the store, so the SAME refresh path a merge and a curation edit use
        # repaints the table, the spans, the summary and the staleness banner.
        self._session.add_result(new_res)
        self._last_comparison = result.comparison
        c = result.comparison
        self._say_baseline(
            "remeasured",
            f"Re-measured {c.n_events} events: median depth "
            f"{c.median_old_depth_pA:.4g} → {c.median_new_depth_pA:.4g} pA "
            f"({c.median_ratio:.3g}×). Detection geometry and curation unchanged.")

    def _show_comparison(self, comparison: Any) -> None:
        from .panels import EstimatorCompareDialog

        self._last_comparison = comparison
        dlg = EstimatorCompareDialog(comparison, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            # "Re-measure now" from inside the report. Goes through the ordinary
            # entry point, so the gating, the target resolution and the off-thread
            # dispatch are the same ones the banner's button uses.
            self.remeasure_active_detection()

    @property
    def last_comparison(self) -> Any:
        """The most recent comparison this panel computed (tests and MainWindow's
        status line read it; nothing else depends on it)."""
        return self._last_comparison

    def select_event(self, event_id: int) -> None:
        """Public entry point (the Event Inspector) to select an event exactly as a table-row
        click would: highlight the matching row, which drives the trace zoom + marker via
        :meth:`_on_event_selected`. The panel stays the owner of the table + overlays."""
        self._table.select_event(int(event_id))

    def set_event_accepted(self, event_id: int, accepted: bool) -> None:
        """Public entry point (the Event Inspector) to curate one event THROUGH the table model,
        so the checkbox repaints and the summary + span overlays refresh via the model's
        ``acceptanceChanged``. The shared df is the single source of truth."""
        self._table.set_accepted_by_id(int(event_id), bool(accepted))

    def refresh_after_edit(self, event_id: int) -> None:
        """Public entry point (the Event Inspector) after it RESIZED event ``event_id`` on the
        shared df — new ``[start_idx, end_idx]`` and recomputed stats. Repaint that table row,
        redraw the span overlay (the span moves to the new bounds) and refresh the summary; if the
        edited event is the selected one, re-zoom + re-drop its marker at the new bounds. The panel
        stays the owner of the table + overlays — the inspector only edits the shared df + emits."""
        self._table.refresh_event(int(event_id))
        self._redraw_spans_from_df()
        self._refresh_summary_from_table()
        self._refresh_score()
        if self._selected_id is not None and int(event_id) == int(self._selected_id):
            self._on_event_selected(int(event_id))

    def resync_curation(self) -> None:
        """Public entry point (the Event Inspector) after a BATCH curation (Accept all / Reject all)
        rewrote the whole ``accepted`` column on the shared df. Reset the table so every checkbox
        repaints from the df, redraw the span overlay to the accepted set and refresh the summary —
        a model reset, not per-cell, because every row may have flipped at once."""
        res = self._stored_detection()
        if res is not None and isinstance(res.payload, EventSet):
            df = res.payload.df
            if CONSENSUS_LABEL_CONFLICTS_COL in df.columns and "flag" in df.columns:
                # This audit column is live curator state, not frozen creation metadata.
                # Applying/clearing a label can resolve a merged ``key=a|b`` conflict;
                # recompute before the label signal reaches the undo checkpoint/export gate.
                df[CONSENSUS_LABEL_CONFLICTS_COL] = pd.array(
                    ["|".join(sorted(label_conflicts(flag)))
                     for flag in df["flag"].to_numpy(dtype=object)],
                    dtype="string",
                )
        self._table.refresh_all()
        self._redraw_spans_from_df()
        self._refresh_summary_from_table()
        self._refresh_score()

    def _refresh_fit_text(self) -> None:
        """Show the fits THE SELECTED TARGET has stored — and nothing when it has none.

        WRITES ON EVERY EXIT, including the empty one. It used to write only when the
        target had a fit (``if parts:``, no else), and the only writer of the empty
        string was :meth:`_clear_results`, which a repoint reaches only when the new
        target has no DETECTION at all. So with two detected recordings, fitting A and
        then selecting B left A's dwell/depth summary verbatim in the Fits box beside B's
        own event table, title, Summary and baseline — measured: "n = 115 ... tau_1 =
        0.0161541 s" from the -1000 mV run under a title naming the -600 mV file whose
        Summary read 317 events. ``FitResult.summary()`` carries no doc_id, path or
        channel, so nothing on screen said whose numbers those were, and a round trip
        between the two recordings did not clear it.

        The empty box is the honest answer for a target with no fit: the Fit button
        beside it is live and says so. A fit that FAILED is a different sentence, written
        by :meth:`_on_fit_requested` from the exception it caught — that path computes
        text this one cannot re-derive from the store, which is why it stays its own
        writer and always runs after the ``add_result`` that lands here.
        """
        parts: list[str] = []
        if self._sel is not None:
            doc_id, channel = self._sel
            for mid in ("fit.dwell", "fit.depth"):
                r = self._session.results.get(doc_id, channel, mid)
                if r is not None and hasattr(r.payload, "summary"):
                    parts.append(r.payload.summary())
        self._stats.set_fit_text("\n\n".join(parts))

    # -- event navigation ------------------------------------------------------------------------
    def _on_event_selected(self, event_id: int) -> None:
        if self._sel is None:
            return
        doc_id, channel = self._sel
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return
        df = res.payload.df
        match = df.index[df["event_id"].to_numpy(dtype=np.int64) == int(event_id)]
        if len(match) == 0:
            return
        self._selected_id = int(event_id)   # remembered so a later mask edit can re-zoom + re-mark
        row = df.loc[match[0]]
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])

        # The axis the CURVE is drawn against (through the layer's alignment offset), not the
        # document's native one. Zooming to a native-axis x on an aligned layer frames a
        # stretch of trace `time_offset_s` away from the event the user clicked — the same
        # offset-0 mistake as the visible-range mapping above.
        xax = self._trace_view.x_axis_for(doc_id, channel)
        vals = xax.values
        if vals.size == 0:
            return
        i0 = int(np.clip(start_idx, 0, vals.size - 1))
        i1 = int(np.clip(end_idx, 0, vals.size - 1))
        x0, x1 = float(vals[i0]), float(vals[i1])
        # ONE padded-zoom implementation (TraceView.zoom_to_span), shared with the manual
        # measurement table's return-to-measurement, so the two framings never diverge.
        self._trace_view.zoom_to_span(
            x0, x1, min_pad=(vals[1] - vals[0]) * 8 if vals.size > 1 else 0.0)

        # A single MARKER overlay highlights the dived-into event.
        self._clear_marker()
        marker = OverlayHint(
            OverlayKind.MARKER,
            starts=np.array([start_idx], dtype=np.int64),
            stops=np.array([end_idx], dtype=np.int64),
            color=PALETTE["selected"],
        )
        self._marker_id = self._trace_view.add_overlay(doc_id, channel, marker)

        # Hand the selection to the measure tool so "Snap to selected event" can jump
        # the cursors here. The owner pushes it in; the measure panel never reaches back.
        self._measure.set_selected_event(doc_id, channel, start_idx, end_idx)

    # -- filter preview (D4) ---------------------------------------------------------------------
    def _on_preview_toggled(self, on: bool) -> None:
        if on:
            self._update_filter_preview()
        else:
            self._trace_view.clear_filter_preview()
            # Unticking withdraws the request; an error line kept standing would
            # describe a preview nobody is asking for (and would then survive into
            # the next tick, where it may no longer be true).
            self._detection.set_preview_error("")

    def _on_preview_state_changed(self, reason: str) -> None:
        """The plot's managed preview curve appeared, or could not be drawn.

        THE TICKED BOX AND THE PLOT MUST NOT DISAGREE IN SILENCE. Every other way a
        preview fails to appear already reaches the user through this one error line —
        an unknown sample rate, a cutoff above Nyquist, a detrend window that achieved
        nothing — and the two that did not were the ones the panel could not see: any
        `TraceView.rebuild` (View ▸ Stacked/Overlay, or showing/hiding ANY channel) tore
        the curve off the plot, and ticking the box for a channel that is not layered
        drew nothing. Both left "show processed trace on the plot" ticked over a plot
        with no orange curve and an empty error line — indistinguishable from a
        processed trace that happens to lie on the raw one, on the only surface that
        shows what the detector will actually run on.

        The view now re-applies the curve itself and reports what it could not do; this
        renders that report. Guarded on the checkbox because the line belongs to a
        request the user is making: with the box unticked there is no request, and
        `_on_preview_toggled` has already cleared the line.
        """
        if not self._detection.filter_preview.isChecked():
            return
        self._detection.set_preview_error(reason)

    def _on_params_changed(self, params: dict[str, Any]) -> None:
        # WHICH DETECTOR the combo names, before anything about the preview. The combo has
        # no signal of its own (the DetectionPanel deliberately wires nothing and emits
        # only ``paramsChanged``), so a change of detector arrives here indistinguishable
        # from a change of threshold -- and is told apart by the remembered value. See
        # :meth:`_show_detector_choice` for why a switch has to move the TABLE.
        chosen = str(params.get("detector") or _MODULE_ID)
        if chosen != self._combo_detector:
            self._combo_detector = chosen
            self._show_detector_choice(chosen)
        # THE MEASURE TOOL LEARNS ABOUT THE FORM ON THE SAME SIGNAL THE PREVIEW DOES.
        # Unconditionally, and BEFORE the preview's early return -- the Measure basis is
        # about what the numbers are computed from, not about what is drawn, so a chain
        # the user ticked with "show processed trace" OFF still has to reach it.
        self._push_measure_conditioning()
        self._mark_baseline_workspace_runtime_change(params)
        self._sync_baseline_workspace(params)
        # Recompute the preview if it is live and ANY stage is still on; clear when the
        # low-pass AND every pre-detection filter (notch / high-pass / detrend / savgol)
        # are off.
        if not self._detection.filter_preview.isChecked():
            return
        if params.get("filter_cutoff_hz") is None and not any_preprocess_enabled(params):
            self._trace_view.clear_filter_preview()
            self._detection.set_preview_error("")   # nothing left for it to be about
        else:
            self._update_filter_preview()

    def _push_measure_conditioning(self) -> None:
        """Tell the Measure tool which chain the Preprocess form is currently asking for.

        WHY THIS EXISTS. :class:`~nanoscope.gui.measure_panel.MeasurePanel` talks to
        Session and to its TraceView and to no sibling panel, so the form's parameters
        can only arrive from outside, through ``set_conditioning``. Until this call there
        was exactly one pusher — ``MainWindow._sync_processed_export``, which re-asks the
        form when the Analysis menu opens or the Measure action re-syncs — so a filter
        ticked on the Preprocess tab reached the Measure panel only if the user happened
        to open a menu first. Before that it fell back to
        ``MeasurePanel._stored_detection_chain``, i.e. a STORED detection, and with none
        on this channel the *Processed signal* item stayed greyed and a measurement on it
        was refused with "no filter stage is enabled on the Preprocess tab" — while the
        orange processed curve computed from that very form was on the plot. The
        measurement itself was never wrong; the panel was simply never told.

        SAME SIGNAL AS THE PREVIEW CURVE. ``paramsChanged`` is the one thing the panel
        gets whenever any knob on the Preprocess/Detection form moves, and it is already
        what re-draws the preview (:meth:`_update_filter_preview`); the basis now learns
        on the same edit, so the two surfaces that describe the conditioned signal can no
        longer disagree about whether it exists.

        THE MERGED FORM, not ``DetectionPanel.params()`` alone. ``set_conditioning``
        compares what it is handed against what it holds and throws its conditioned-trace
        memo away when the two differ; pushing a bare detection form here and the merged
        :meth:`detection_params` from ``MainWindow`` would make those two pushes differ on
        the ``baseline_*`` keys and re-condition the whole recording (16-85 ms at 749 k
        samples) on every menu open. One shape from both doors, so an unchanged form stays
        the no-op that ``set_conditioning``'s docstring promises its callers.

        Also called from :meth:`_on_selection`, because a form can arrive already
        configured with no edit at all: QSettings restores the Preprocess stages from the
        last session, and the first ``paramsChanged`` after that may never come.
        """
        self._measure.set_conditioning(self.detection_params())

    @staticmethod
    def _conditioned_from_form(
        raw: np.ndarray, fs: float, params: dict[str, Any]
    ) -> np.ndarray:
        """``raw`` as candidate localization will see it, from the LIVE form.

        ``lowpass(preprocess_for_detection(raw))`` -- the same two steps, in the same
        order, with the user's own low-pass DESIGN rather than the kernel defaults.

        This is the orange preview's candidate-localization path. The baseline probe
        uses :func:`~nanoscope.analysis.conditioning.characterization_trace` instead,
        so high-pass can help find boundaries without high-pass itself erasing the DC
        baseline used for classical event-table amplitudes. Every other enabled stage
        still applies; Normalize can still recenter that branch. The provenance-driven `_active_signal` is the sibling
        of this for rows that have ALREADY been measured; this one is for a live form
        that has not run yet.

        Raises whatever the kernels raise (a cutoff above Nyquist, a detrend window
        longer than the trace); the preview turns that into a visible message.

        DELEGATES to :func:`~nanoscope.analysis.conditioning.conditioned_trace`, which is
        THE definition. This body used to be a bit-identical copy of it, which is two
        writers for one array: the Measure tool and the processed-trace export read the
        pure module while the preview read this method. The method stays because it is
        the panel's own door (a GUI class may not be imported by ``analysis/``); what it
        no longer is, is a second answer.
        """
        return conditioned_trace(raw, fs, params)

    def _update_filter_preview(self) -> None:
        """Draw the candidate-localization trace: the full pre-detection chain
        (notch / high-pass / baseline-drift / savgol, whichever are on) THEN low-passed
        (if on). preprocess_for_detection performs the chain exactly as detection does;
        the low-pass is applied here to mirror what detect_events does internally, so the
        single managed orange preview is the true localization signal — even when only
        one pre-detection stage is enabled. With high-pass on, quantitative baseline and
        amplitude measurements use the companion chain with that one stage omitted."""
        if self._sel is None:
            return
        params = self._detection.params()
        cutoff = params.get("filter_cutoff_hz")
        preprocess_on = any_preprocess_enabled(params)
        doc_id, channel = self._sel
        ch = self._session.calibrated(doc_id, channel)
        fs = ch.sample_rate_hz
        if cutoff is None and not preprocess_on:
            # No stage on -> nothing to preview; any standing objection is about a
            # request that no longer exists.
            self._trace_view.clear_filter_preview()
            self._detection.set_preview_error("")
            return
        if fs is None:
            # Same first blocker as the detection gate (a legacy TDMS stores no
            # rate): every stage is designed in Hz / seconds, so the processed
            # trace cannot be computed. Refuse OUT LOUD through the same error line
            # as a misconfigured stage — a ticked box that draws nothing and says
            # nothing is exactly the vanished-curve failure this label exists to
            # eliminate.
            self._trace_view.clear_filter_preview()
            self._detection.set_preview_error(
                "sample rate unknown — the filter stages are designed in Hz and "
                "seconds, so the processed trace cannot be computed for this "
                "recording (detection is disabled for the same reason).")
            return
        try:
            with collect_baseline_warnings() as kernel_warns:
                processed = self._conditioned_from_form(
                    np.asarray(ch.data), float(fs), params)
        except Exception as exc:  # noqa: BLE001
            # A bad cutoff (>= fs/2) or a detrend window longer than the trace is a user
            # error, not a crash: clear the preview, but SAY which stage objected —
            # a silently vanishing curve gave the user nothing to act on.
            self._trace_view.clear_filter_preview()
            self._detection.set_preview_error(str(exc))
            return
        # A stage can also RETURN without raising while its kernel warns it achieved
        # nothing: a detrend window equal to the trace length yields ONE block ("the
        # baseline is effectively constant and cannot track drift",
        # ConstantBaselineWarning) and hands back the input untouched. One millisecond
        # more window raises the honest "longer than the recording" ValueError above;
        # this is the same misconfiguration one spinbox step to the other side of that
        # cliff, so it gets the same treatment — clear the curve, put the kernel's own
        # words on the error line (whose fixed prefix says "unavailable", which a
        # cleared curve keeps true). Softer at-risk advisories do NOT clear a preview
        # that is genuinely different from the raw trace.
        degenerate = [w for w in kernel_warns if isinstance(w, ConstantBaselineWarning)]
        if degenerate:
            self._trace_view.clear_filter_preview()
            self._detection.set_preview_error(" ".join(str(w) for w in degenerate))
            return
        self._detection.set_preview_error("")
        self._trace_view.set_filter_preview(doc_id, channel, np.asarray(processed))

    # -- fits (in-process; small event table) ----------------------------------------------------
    def _on_fit_requested(self, opts: dict[str, Any]) -> None:
        if self._sel is None:
            return
        doc_id, channel = self._sel
        res = self._stored_detection()
        if res is None or not isinstance(res.payload, EventSet):
            # Unreachable from the UI now that the Fit button is gated up front (see
            # StatsPanel.set_detection_present); kept as a guard for programmatic callers so a
            # missing detection can never reach the fitters.
            self._stats.set_fit_text("Run detection first — there are no events to fit.")
            return
        events = res.payload
        fp = self._session.calibration(doc_id, channel).fingerprint()
        # THIS target's recovery map, never "the last one anybody ran". eta(t) is a property
        # of a pore and its detector settings; dividing B's dwell distribution by A's
        # efficiency curve moves tau (3.79 -> 3.86 ms on the two measured here) and the fit
        # then PRINTS "measured eta(t) on [...]" as if it were B's.
        eff = self._recovery.get((doc_id, channel)) if opts.get("use_efficiency") else None

        # Filter-limited events are excluded by default (their depth is attenuated
        # and their dwell biased long); the toggle in the Fits box puts them back.
        use_fl = bool(opts.get("use_filter_limited", False))

        parts: list[str] = []
        try:
            dwell_res = fit_dwell(events, doc_id=doc_id, channel=channel,
                                  n_components=int(opts.get("n_components_dwell", 1)),
                                  efficiency=eff, use_filter_limited=use_fl,
                                  calibration_fingerprint=fp)
            self._session.add_result(dwell_res)
            parts.append(dwell_res.payload.summary())
        except Exception as exc:  # noqa: BLE001 — surface the failure in the readout
            parts.append(f"dwell fit failed: {exc}")
        try:
            depth_res = fit_depth(events, doc_id=doc_id, channel=channel,
                                  n_components=int(opts.get("n_components_depth", 1)),
                                  use_filter_limited=use_fl,
                                  calibration_fingerprint=fp)
            self._session.add_result(depth_res)
            parts.append(depth_res.payload.summary())
        except Exception as exc:  # noqa: BLE001
            parts.append(f"depth fit failed: {exc}")
        self._stats.set_fit_text("\n\n".join(parts))

    # -- staleness -------------------------------------------------------------------------------
    def _check_staleness(self, res: AnalysisResult) -> None:
        """Re-judge the shown result and announce it (``stalenessChanged``) -- every path,
        the badge-set ones and the clear one alike, because a surface that only hears
        "went stale" keeps its marks after the banner is gone."""
        self._judge_staleness(res)
        self.stalenessChanged.emit()

    def _judge_staleness(self, res: AnalysisResult) -> None:
        """Mark the shown result stale when either stamp no longer matches. Never fix it.

        The calibration stamp is checked first and, when it has moved, it is the one
        that gets the button: a gain change moved the thresholds the detector ran
        with, so the events themselves are questionable and only a re-detection
        answers it. A re-measure on top of a stale calibration would produce
        confidently-wrong depths against the right open-pore reference.

        The baseline stamp is the migration case. It is stale when it is ABSENT
        (this table predates the stamp -- and absence is exactly the signal: those
        are the rows the 10 ms block median measured) or when it differs from the
        estimator now configured in the Preprocess tab.
        """
        if self._sel is None:
            self._stale.setVisible(False)
            self._table.set_stale_columns(())
            return
        doc_id, channel = self._sel
        live = self._session.calibration(doc_id, channel).fingerprint()
        reconciled = res.module_id == CONSENSUS_MODULE_ID
        if tuple(res.calibration_fingerprint) != tuple(live):
            self._stale.show_message(
                "Calibration changed since this detection ran — the blockade depths and "
                "ECDs are computed against the old gain/offset and are now stale."
                + (" Refresh the source detector results, then reconcile again."
                   if reconciled else ""),
                kind=_STALE_CALIBRATION,
                action_label=("Refresh sources first" if reconciled else "Re-run"))
            self._sync_stale_action(res)
            # The amplitude columns are stale either way, and badging them is about
            # the NUMBERS, not about which stamp moved.
            #
            # `REMEASURED_COLUMNS`, not §4.2.5's six: ``baseline_pA`` moves too (0.36x
            # on the owner's apo file) and it is the column the other six are computed
            # FROM. Leaving it unbadged beside a badged ``depth_pA`` puts the wrong
            # open-pore reference on screen with no mark on it, and invites reading it
            # as the trustworthy number that explains the badge next to it.
            self._table.set_stale_columns(REMEASURED_COLUMNS)
            return
        source_reason = self._reconciliation_source_staleness(res)
        if source_reason is not None:
            self._stale.show_message(
                source_reason,
                kind=_STALE_RECONCILIATION_SOURCE,
                action_label="Review sources first",
            )
            self._sync_stale_action(res)
            # The preserved snapshot's frozen measurements have not changed. This is a
            # lineage warning, not a claim that its amplitude columns are numerically bad.
            self._table.set_stale_columns(())
            return
        reason = self._baseline_staleness(res)
        if reason is not None:
            self._stale.show_message(
                reason + (" Refresh the source detector results, then reconcile again."
                          if reconciled else ""),
                kind=_STALE_BASELINE,
                action_label=("Refresh sources first"
                              if reconciled else "Re-measure depths"))
            self._sync_stale_action(res)
            self._table.set_stale_columns(REMEASURED_COLUMNS)
            return
        self._stale.setVisible(False)
        self._table.set_stale_columns(())

    @staticmethod
    def _reconciliation_input_fingerprints(res: AnalysisResult) -> dict[str, str]:
        """Recorded source-module digests for a reconciled snapshot, if available."""
        if res.module_id != CONSENSUS_MODULE_ID or not isinstance(res.payload, EventSet):
            return {}
        provenance = res.payload.provenance or {}
        reconciliation = provenance.get("reconciliation")
        if not isinstance(reconciliation, Mapping):
            return {}
        raw = reconciliation.get("input_result_fingerprints")
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(module_id): str(digest)
            for module_id, digest in raw.items()
            if str(module_id) and digest is not None
        }

    def _reconciliation_source_staleness(self, res: AnalysisResult) -> str | None:
        """Explain whether a preserved Reconciled snapshot predates its source tables."""
        recorded = self._reconciliation_input_fingerprints(res)
        if not recorded:
            return None  # legacy snapshot: no source digest means no comparison to make
        missing: list[str] = []
        changed: list[str] = []
        for module_id, expected in sorted(recorded.items()):
            source = self._session.results.get(res.doc_id, res.channel, module_id)
            if source is None:
                missing.append(detector_name(module_id))
                continue
            try:
                actual = reconciliation_source_fingerprint(source)
            except (TypeError, ValueError):
                changed.append(detector_name(module_id))
                continue
            if actual != expected:
                changed.append(detector_name(module_id))
        if not missing and not changed:
            return None
        details = []
        if changed:
            details.append("changed: " + ", ".join(changed))
        if missing:
            details.append("missing: " + ", ".join(missing))
        return (
            "This Reconciled table is a preserved snapshot that predates changes to its "
            f"source results ({'; '.join(details)}). Review or restore those source "
            "tables, then reconcile again. This snapshot remains unchanged and available "
            "for review and export."
        )

    def _baseline_staleness(self, res: AnalysisResult) -> str | None:
        """Why this result's open-pore estimator is stale, or ``None`` when it is not.

        Spells the reason out rather than saying "stale", because the size of the
        change is the whole point: on the owner's -600 mV apo file the same events
        measured by the two estimators differ by more than an order of magnitude,
        and on a clean recording they differ by nothing. He cannot tell those apart
        from the stored row, which is why the message names the old estimator and
        points at the report that measures the difference on THIS recording.
        """
        if self._sel is None:
            return None
        resolved = self._live_baseline_for_run()
        if resolved is None:
            return None
        cfg, token, live, fs, n = resolved
        want = cfg.fingerprint(fs)
        have = res.baseline_fingerprint
        if have == want:
            return None
        prov = res.payload.provenance if isinstance(res.payload, EventSet) else None
        old = describe_stored_baseline(prov)
        if token == FALLBACK_CLAMPED:
            now = (f"{cfg.describe()} — the Preprocess tab's {live.describe()}, CLAMPED to "
                   f"this recording's {n / fs:g} s length, which is what a fresh run "
                   f"here would use")
        elif token is not None:
            now = (f"{cfg.describe()} — the Preprocess tab's {live.describe()} cannot run "
                   f"on this recording's {n:,} samples, so a fresh run here measures a "
                   f"global median")
        else:
            now = live.describe()
        if not have:
            return (
                f"These events were measured with {old} — this table predates the "
                "open-pore estimator now in use. Depth, frac_depth, ECD, mean/max "
                "deviation and SNR can change by more than 10x on recordings whose "
                "events are long compared with half that window. Nothing has been "
                "re-measured: use Analysis ▸ Compare baseline estimators… to see the "
                "size of the change on this recording first.")
        return (
            f"The open-pore estimator changed since this detection ran: these events "
            f"were measured with {old}, and {now} is configured now. "
            "Depth, frac_depth, ECD, mean/max deviation and SNR are stale.")

    def _live_baseline_for_run(self):
        """The estimator a fresh run WOULD use on the selected recording, or ``None``:
        ``(resolved config, clamp token or None, the form's config, fs, n_samples)``.

        The form's config is passed through `baseline.adapt_to_length` -- the SAME clamp
        `characterisation_baseline` applies before it fingerprints and stamps a run --
        so the staleness judge compares like with like. Comparing the FORM's fingerprint
        raised a permanent, false banner on every short cut: a 0.08 s segment of the
        owner's -600 mV file detected at the default 0.5 / 0.1 s is stamped with the
        clamped 80 / 16 ms fingerprint (sha256:31991df77507), the form's 500 / 100 ms
        never matches it, "The open-pore estimator changed since this detection ran"
        stood over a CLAMPED line saying the opposite, nine columns were badged on the
        table, the gallery and the inspector, and the offered Re-measure re-measured to
        the identical fingerprint and left the banner up for good (X4). Stale means
        "differs from what a fresh run would use on THIS recording", and a fresh run
        clamps.
        """
        live = self._live_baseline_config()
        if live is None:
            return None
        rate = self._active_rate_and_length()
        if rate is None:
            return None
        fs, n = rate
        cfg, token = adapt_to_length(live, n, fs)
        return cfg, token, live, fs, n

    def _live_baseline_config(self):
        """The estimator the Preprocess tab is currently set to, or ``None``.

        Read through the panel's own resolver so the comparison is against the
        config a fresh run WOULD use, not against a second reading of the form."""
        try:
            return self._baseline.config()
        except Exception:  # noqa: BLE001 — a half-typed form must not raise a banner
            return None

    def _active_rate_and_length(self) -> tuple[float, int] | None:
        """The selected channel's ``(sample rate, sample count)``, or ``None``. The
        fingerprint depends on both: ``decim`` is derived from ``base_s * fs``, and the
        windows themselves are clamped to the count (`adapt_to_length`), so a comparison
        made without them would call every result stale on a rate-less recording and
        every clamped one stale for ever. Read off the same calibrated channel a run
        measures on, so the count is the one the estimator will see."""
        if self._sel is None:
            return None
        doc_id, channel = self._sel
        if not self._session.has_document(doc_id):
            return None
        try:
            ch = self._session.calibrated(doc_id, channel)
            fs = ch.sample_rate_hz
            n = int(np.asarray(ch.data).size)
        except Exception:  # noqa: BLE001
            return None
        return (float(fs), n) if fs else None

    def _on_calibration_changed(self, doc_id: str, channel: int) -> None:
        if self._sel is None or self._sel[0] != doc_id:
            return
        # A "Treat this channel as" override is a calibration edit (it lives on the
        # ChannelCalibration), and it moves what this recording's selection RESOLVES to
        # and whether the target passes the gate (S7-CAL1). Follow the session: if the
        # override made another channel the recording's current channel, repoint there
        # exactly as MainWindow's own push would; if the target stands, re-gate it in
        # place so the Detect button and the Analysis menu (`can_run`) keep one answer.
        target = self._resolve_analysis_channel(doc_id, self._sel[1])
        if target != self._sel[1]:
            self.set_selection(doc_id, target)
            return
        if channel != self._sel[1]:
            return   # another channel of this recording; nothing here is about it
        self._regate(reassert=False)
        res = self._stored_detection()
        if res is not None:
            self._check_staleness(res)
            # THE CAPTION FOLLOWS A LABEL EDIT, NEVER A GAIN EDIT. `target_units` is a
            # label and does not touch the array ("Only gain/offset alter the ARRAY;
            # target_units is a label" — core/calibration.fingerprint), so declaring the
            # channel nA on an unchanged gain tells us what the numbers on screen have
            # been in all along, and the row labels should say so at once rather than
            # waiting for a re-detection while the PSD window already says "nA".
            #
            # When the GAIN moved, the depths above are the old gain's and the staleness
            # banner has just said so; re-captioning them in the new units would
            # re-describe measured numbers under a calibration they were never measured
            # under, which is the thing the stamp exists to prevent. So the caption
            # stands until a re-run re-measures them.
            live = self._session.calibration(doc_id, channel).fingerprint()
            if tuple(res.calibration_fingerprint) == tuple(live):
                self._stats.set_units(self._selection_units())
        # The live "processed trace" preview is computed from session.calibrated(),
        # which this edit just changed. Recompute it through the SAME path the tick
        # uses (_update_filter_preview reads the fresh calibrated data and redraws,
        # or clears with the error line naming the failing stage) — otherwise the
        # orange curve keeps OLD-calibration values on a pane whose raw curves and
        # axis label are already in the new one: two curves disagreeing by the whole
        # edit, and after a pA -> nA change y-autorange spans both, squashing the
        # real trace to a hairline. The measure panel re-measures on this same
        # signal for the same reason; the preview is the other surface derived from
        # the calibrated signal.
        if self._detection.filter_preview.isChecked():
            self._update_filter_preview()
        self._baseline_workspace.mark_preview_stale(
            "Calibration changed — refresh the baseline comparison before relying on it.")
        self._sync_baseline_workspace()

    def _on_baseline_params_changed(self, _params: dict[str, Any]) -> None:
        """The user edited (or applied a proposal to) the open-pore estimator.

        Re-judge the shown detection immediately, for the same reason the
        calibration edit does: a stamp that only gets compared when a result is
        first displayed leaves the user looking at depths measured by an estimator
        he has just replaced, with nothing on screen saying so.
        """
        res = self._stored_detection()
        if res is not None:
            self._check_staleness(res)
        self._remember_baseline_mark()
        self._baseline_workspace.mark_preview_stale(
            "Open-pore baseline settings changed — refresh the comparison before relying on it.")

    def _clear_baseline_notice(self) -> None:
        """Drop the baseline panel's REFUSED notice when the panel is repointed.

        Third readout of the same family, one panel along. ``BaselinePanel._post`` writes
        the notice — "REFUSED — 2 of the checks in Baseline health CONDEMN the windows
        proposed above (…)" — and NOTHING ever writes its absence: ``set_gate(ok=True)``
        hides the label without emptying it, and ``set_gate(ok=False)`` (the empty panel's
        own "Select a CURRENT channel…") touches neither its text nor its visibility. So a
        refusal survived every repoint that could not hide it: measured, a first Apply
        click refused on a gating recording left that banner VISIBLE, verbatim, after
        selecting a voltage-only recording where no baseline can be measured at all, and
        after closing every document, naming a proposal that no longer exists and checks
        that never condemned anything on screen.

        The notice is about ONE proposal on ONE recording, and a target change drops the
        proposal (`BaselinePanel.set_recording`) — so by the time the new target is stated
        there is nothing the sentence could be about. Called on the two paths that repoint
        this panel, immediately BEFORE the new gate is stated, so the new recording's own
        blocker (if it has one) is posted over silence rather than over the last one's.

        Like :meth:`_show_polarity_evidence`, this writes only the ABSENCE: ``_post``
        stays the single writer of the message itself, so the two files cannot drift over
        the wording. It would sit better in ``baseline_panel.py`` as part of
        ``set_recording``, beside the proposal it already drops; that file is not this
        change's to edit.
        """
        self._baseline.gate_label.setText("")
        self._baseline.gate_label.setVisible(False)

    def _remember_baseline_mark(self) -> None:
        """Follow the standing "ACCEPTED OVER … FAILED HEALTH CHECKS" mark for THIS target.

        ``paramsChanged`` is the one signal the panel gets whenever the mark can begin:
        `BaselinePanel` emits it from ``apply_proposal``, which is the only method that
        raises the mark. So this is the seam where an acceptance is observable, and it is
        the only writer of :attr:`_baseline_marks`.

        WHAT IS RECORDED IS A PAIR OF WINDOWS, NOT AN EVENT. The entry is keyed by the
        values in the form at the moment of the acceptance, so nothing here has to guess
        when the mark "ends": a hand edit, a Reset or a clean re-proposal simply move the
        form off those values and :meth:`_restore_baseline_mark` finds nothing to restore.
        That is the panel's own rule ("for as long as the form holds these numbers") read
        literally, and it is why an edit that is later DISCARDED — the round trip re-seeds
        the stored detection's params over it — correctly brings the mark back with them.

        A mark is remembered only when the offer behind it is still identifiable (the live
        proposal is the one this panel put on offer for this target), so what is replayed
        later is a real refusal the user really overrode, never a reconstruction.
        """
        sel = self._sel
        if sel is None or not self._baseline.accepted_condemned():
            return
        offer = self._baseline_offer
        if offer is None or offer[0] != sel or offer[1] is not self._baseline.proposal():
            # The mark stands but this panel cannot name the offer behind it (the replay
            # in `_restore_baseline_mark` is re-raising it, or a driver called
            # ``apply_proposal`` directly). Record nothing rather than a guess.
            return
        params = self._baseline.params()
        self._baseline_marks[
            (sel[0], sel[1], float(params["baseline_rough_s"]),
             float(params["baseline_base_s"]))] = (offer[1], offer[2])

    def _restore_baseline_mark(self) -> None:
        """Raise the mark again when the windows it was about come back. THE OTHER HALF.

        `BaselinePanel`'s own comment says the mark stands "for as long as the form holds
        these numbers, because the screen that has to say 'these windows were condemned'
        is the screen the user reads the resulting depth on". Two of its clears are
        written for the LEAVING direction — ``set_recording`` on a target change and
        ``set_params`` on a seed — and neither has a return direction, so the ordinary
        gesture of clicking to another recording and back put the condemned windows back
        in the form (out of the stored detection's own params) under the panel's most
        reassuring sentence, with ``accepted_condemned()`` empty and the amber styling
        gone. The health block below still listed the failing checks, so the two surfaces
        disagreed about one fact.

        THE MARK IS NOT RE-TYPED HERE, IT IS RE-EARNED. This does not compose the sentence
        or reach into the other panel's state: it puts the SAME proposal and the SAME
        proposal-scope health back on offer and confirms it with ``override=True`` — the
        call the user's second click made — so `BaselinePanel` stays the single writer of
        the mark, of its wording and of its styling, and the two files cannot drift. The
        offer is withdrawn immediately afterwards (``set_proposal(None)``), which also
        disarms the override, so no later condemned proposal can be accepted in one click.

        Called from :meth:`_show_result` because that is where the windows come back: the
        re-seed reaches the form only when there IS a stored detection to seed from, and
        the mark must not outlive the numbers. Guarded so it can only ever restore, never
        invent — the mark must be down, the windows must be exactly the remembered pair,
        and a live proposal on offer is never destroyed to make room for a replay.
        """
        sel = self._sel
        if sel is None:
            return
        if self._baseline.accepted_condemned():
            return          # still standing (same-target redraw): nothing to restore
        if self._baseline.proposal() is not None or self._baseline.is_busy():
            # A live offer is on screen, or a probe is in flight and about to put one
            # there. Neither is ours to replace — `set_proposal` would drop the offer and
            # take the spinner off a button that is still measuring.
            return
        params = self._baseline.params()
        # THE WHOLE QUESTION, asked in one lookup: were exactly THESE windows, on THIS
        # recording, accepted over a refusal? Anything else the form might be holding —
        # a hand edit, a Reset, a clean re-proposal — simply misses.
        key = (sel[0], sel[1], float(params["baseline_rough_s"]),
               float(params["baseline_base_s"]))
        entry = self._baseline_marks.get(key)
        if entry is None:
            return
        proposal, health = entry
        self._baseline.set_proposal(proposal)
        self._baseline.set_health(health, measured_for="proposal")
        self._baseline.apply_proposal(override=True)
        self._baseline.set_proposal(None)
        if not self._baseline.accepted_condemned():
            # The replay did not re-raise it (a proposal whose health no longer condemns
            # it cannot be "accepted over" anything). Forget it rather than leave a memory
            # that promises a mark the panel does not show.
            self._baseline_marks.pop(key, None)

    # -- teardown --------------------------------------------------------------------------------
    def _on_document_closed(self, doc_id: str) -> None:
        # Drop the closed recording's surrogate-study state. A file's doc_id is its path, so a
        # kept entry would re-attach to the SAME file on its next open — presenting a stale
        # eta(t) / FDR as a fresh measurement of a possibly re-calibrated recording.
        # ...and the standing baseline mark with them, for the same reason: it names the
        # windows of ONE load, and a kept entry would re-raise "ACCEPTED OVER 2 FAILED
        # HEALTH CHECKS" over a re-opened file whose user has been refused nothing.
        for store in (self._recovery, self._fdr, self._baseline_marks):
            for key in [k for k in store if k[0] == doc_id]:
                del store[key]
        # The OFFER too: a proposal is measured on one recording, and this one is gone.
        if self._baseline_offer is not None and self._baseline_offer[0][0] == doc_id:
            self._baseline_offer = None
        if self._sel is not None and self._sel[0] == doc_id:
            self._show_empty()
        else:
            self._sync_surrogate_readouts()

    def _clear_overlays(self) -> None:
        for oid in self._overlay_ids:
            self._trace_view.remove_overlay(oid)
        self._overlay_ids = []

    def _clear_marker(self) -> None:
        if self._marker_id is not None:
            self._trace_view.remove_overlay(self._marker_id)
            self._marker_id = None

    def _clear_results(self) -> None:
        """THE NEGATIVE TWIN OF :meth:`_show_result` — the other half of one repoint.

        Every readout on this panel describes ONE (doc_id, channel). Pointing the panel
        at a target that has nothing to show is not "leave the screen as it was": it is
        the moment the previous recording's answers silently become claims about this
        one. That is why :meth:`_sync_surrogate_readouts` exists for the FDR pair, why
        ``BaselinePanel.set_recording`` drops a proposal, and why the preview error line
        is emptied on a selection change. The rule belongs to the PAIR, not to either
        half of it:

            **whatever `_show_result` writes from a result, this writes from its
            absence.** A readout with only a positive writer states the last recording
            that had one, under this recording's title.

        Two were missing it, both found by the same sweep at two different boundaries,
        and both are now single-writer methods called from both halves:

        * the Fits box — :meth:`_refresh_fit_text` wrote only when the new target HAD a
          fit, so one recording's dwell/depth summary stood beside another's event table.
        * the Detection tab's measured-polarity line — see
          :meth:`_show_polarity_evidence`; pushed only when provenance carried a dict,
          never reset, so it survived onto a recording whose events go the opposite way
          and onto an empty window with no document open at all.

        The rest already held the rule and are listed so the next reader can check the
        pair at a glance: the event table (``set_events``), the stale-column badges, the
        clock the table's t-columns are labelled on, the Summary block (``clear_summary``
        also drops the warnings and the filter-limited count), the staleness banner, the
        detector score, and the three detection-present gates. Overlays and the marker
        belong to the previous target's PANE and are dropped by ``set_selection`` /
        ``_show_empty`` before this runs.

        THREE MORE LIVE ON THE BASELINE PANEL and are handled at the repoint itself
        rather than here, because they are not written from a RESULT: the open-pore
        windows and their source line, the standing ACCEPTED-OVER-FAILED-HEALTH-CHECKS
        mark (:meth:`_restore_baseline_mark` — the one readout that must come BACK, not
        merely be cleared, because it describes numbers the re-seed puts back), and the
        REFUSED notice (:meth:`_clear_baseline_notice`). All three used to survive a
        repoint, and all three survived onto an EMPTY panel — ``set_recording(None)``
        means "do not judge", so with the last document closed they still described it.
        """
        self._table.set_events(_empty_events())
        self._table.set_stale_columns(())
        # WHICH CLOCK the (now empty) table's t-columns are labelled on. `_show_result`
        # pushes this on every result for exactly this reason: the offset is a property
        # of the recording pointed at (10.25 / 61.30 / 0 s on the owner's three files),
        # so a header note left on the previous recording's t0 is a wrong label, and one
        # left standing over an empty table is the label the NEXT run's rows land under.
        self._table.set_clock_offset(
            self._trace_view.clock_offset_for(*self._sel)
            if self._sel is not None and self._session.has_document(self._sel[0])
            else None)
        self._stats.clear_summary()
        # ...and the row labels' UNITS with them. They belong to the channel, not to the
        # cleared numbers: left standing they would caption the NEXT run's depths — or
        # an empty panel — in the units of the recording the user just repointed away
        # from, which is the same wrong-label failure the clock offset above is cleared
        # for. With no selection this reads "" and the labels say "units unknown".
        self._stats.set_units(self._selection_units())
        self._refresh_fit_text()
        self._show_polarity_evidence(None)
        # The baseline health block, when what is on screen is the verdict on a
        # DETECTION: with no detection shown it describes a table that is not there.
        # A 'proposal'-scope block is left alone — it belongs to the proposal on offer
        # above it, which this method knows nothing about. (A target CHANGE also clears
        # it through `BaselinePanel.set_recording`; the twin must not depend on another
        # panel's housekeeping for the same-target paths that reach here.)
        if self._baseline.health_scope() == "detection":
            self._baseline.set_health(None, measured_for="detection")
        self._baseline.set_run_windows(None)
        self._stale.setVisible(False)
        self._score.clear()
        self._measure.set_detection_present(False)
        self._merge.set_detection_present(False)
        self._stats.set_detection_present(False)
        self._sync_result_selector()

    def _show_empty(self) -> None:
        self._clear_overlays()
        self._clear_marker()
        self._selected_id = None
        self._trace_view.clear_filter_preview()
        # The preview error line belongs to the recording it named; with none
        # selected there is nothing it could be about (same rule as set_selection).
        self._detection.set_preview_error("")
        # ...and so does the REQUEST. `set_selection` already drops the tick when the
        # target changes; this is the same gesture with nothing to move to (the last
        # recording was closed), and a box left ticked over a plot showing "Open a
        # recording" is a control claiming a state it no longer has.
        if self._detection.filter_preview.isChecked():
            self._detection.filter_preview.setChecked(False)
        self._measure.clear()
        self._sel = None
        self._sync_surrogate_readouts()   # no target -> no FDR, no efficiency
        self._title.setText("No channel selected")
        self._detection.set_runnable(False, "Select a CURRENT channel to run detection.")
        self._detection.set_recording(None)
        # Same message as the detector's, because it is the same first blocker: a
        # proposal and a health block both belong to a recording, and with none
        # selected there is nothing they could be about.
        #
        # `_NOTHING_SELECTED` rather than the ``None`` this used to pass: ``None`` means
        # "do not judge" to `BaselinePanel.set_recording`, so it dropped the proposal and
        # the health block but left the WINDOWS, the source line and the standing
        # ACCEPTED-OVER mark exactly as the closed recording had them — the panel went on
        # saying "These values: measured on this recording, and confirmed — ACCEPTED OVER
        # 2 FAILED HEALTH CHECKS … see Baseline health below" under the title "No channel
        # selected", pointing at a health block this method had just emptied. Naming the
        # empty selection as a target is the same repoint every other readout here gets.
        self._clear_baseline_notice()
        self._baseline.set_recording(None, _NOTHING_SELECTED)
        self._baseline.set_gate(False, "Select a CURRENT channel to measure its baseline.")
        self._sync_baseline_workspace()
        self._clear_results()


def _empty_events() -> EventSet:
    from ..analysis.events import empty_df
    return EventSet(empty_df(), {"analyzed_duration_s": 1.0})
