"""Pure, Qt-free derivation of the two ease-of-use affordances' state from a Session.

Two GUI features read everything they draw from here, so their logic is testable
without a QApplication and can never drift apart:

* the **calibration chip** (status bar) — is the ACTIVE recording's current-channel
  gain still at its default (uncalibrated, so every depth / ECD / conductance / noise
  magnitude is silently non-physical), user-set, or is there no current channel at all;
* the **workflow strip** (above the signal viewer) — a stepper over the fixed pipeline
  ``Open → Calibrate → Filter → Detect → Curate → Analyze`` telling the user where they
  are and what is unlocked next.

Design rules honoured here (see ``gui/main_window.py`` for the wiring contract):

* This module imports ONLY core/analysis value types — never a GUI panel, never Qt at
  runtime. ``Session`` is referenced for typing only (guarded by ``TYPE_CHECKING``), so
  importing this module does not pull in PySide6.
* "Calibrated" REUSES the calibration model's own user-set-vs-default notion — the exact
  test ``core.calibration.save_sidecar`` uses to decide whether a channel differs from
  ``CalibrationSet.default_for(info)``. The chip and the sidecar can therefore never
  disagree about whether a gain was set.
* Detection is read through the ``ResultStore`` (``session.results``), the same access the
  analysis panel uses — this never reimplements detection or touches the frozen 20-column
  EventSet schema.
* "Current channel" is ``Session.effective_kind`` — the user's Calibration-panel override
  when there is one, the header otherwise — resolved through
  ``Session.resolve_current_channel``, the SAME resolver the panes rank by and Batch detect
  targets. This module used to read ``ChannelInfo.kind`` (the header) directly, and the
  consequence was measured on a 2 s CSV of the owner's own -700 mV recording with neutral
  column headers: after "Treat this channel as: current" + pA + gain 2.5 was applied and
  Batch detect had stored 15 events on that channel, the chip read "No current channel"
  and the strip locked CALIBRATE / FILTER / DETECT with "No current channel on this
  recording" — a flatly false statement about a channel the app had just calibrated, drawn
  with a pA axis and detected on. The user reads that as "this recording cannot be
  analysed" and stops.
* A gain that RUINED the channel is not "calibrated". ``Session.calibration_issues`` is
  the authority on what a calibration did to the samples; a channel it reports non-finite
  is rendered DAMAGED (amber, the issue as the tooltip) and CALIBRATE is not done, so the
  chip can never certify as physical a channel the File info panel is reporting as ruined.
* When the user's own "Treat this channel as" override is WHY a recording has no current
  channel, the locked sentence names it -- and it is ONE sentence, built once
  (``_no_current_reason``), carried to the three locked stage reasons AND to
  ``CalibrationStatus.issue`` for the chip. It was measured reaching three of the four:
  the strip named IN15's override while the chip beside it said the recording had no
  live current channel; and on an undeclared recording it reached none, because the note
  only listed channels the FILE had declared current.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..analysis.events import EventSet
from ..core.calibration import CalibrationSet
from ..core.sources import DETECTION_MODULE_IDS
from ..core.types import ChannelKind

if TYPE_CHECKING:  # typing only — keeps this module free of a runtime Qt import
    from ..core.session import Session

__all__ = [
    "CalibrationState",
    "CalibrationStatus",
    "Stage",
    "StageStatus",
    "PipelineStatus",
    "PIPELINE_ORDER",
    "compute_status",
]

# ================================================================================================
# Calibration status — the chip's whole model
# ================================================================================================

class CalibrationState(Enum):
    """State of the ACTIVE recording's current-channel gain."""

    NO_CHANNEL = "no_channel"    # no live current channel to calibrate
    UNCALIBRATED = "uncalibrated"  # gain still at CalibrationSet.default_for() — magnitudes not physical
    CALIBRATED = "calibrated"    # user set a non-default gain — magnitudes are physical
    DAMAGED = "damaged"          # a gain is set but it made the channel non-finite — NOT physical
    # ^^ Measured on IN15 of the owner's 15 s -1000 mV ABF: gain 1e38 previewed finite
    # float64 numbers, Apply committed it, ``apply()`` (float32) turned every one of
    # 749,651 samples into -inf, and the chip read "✓ Gain set: ×1e+38 → pA" with the
    # tooltip "so reported magnitudes are physical" while the File info panel, one dock
    # over, said "every one of 749,651 samples is non-finite". The green tick is the one
    # claim a user relies on to stop checking; it may not be made over a ruined channel.


@dataclass(frozen=True)
class CalibrationStatus:
    """What the calibration chip renders.

    ``detail`` is a short "gain → units" summary populated for the CALIBRATED and DAMAGED
    cases; ``channel`` is the resolved current-channel index (``None`` when there is none);
    ``issue`` is the SPECIFIC sentence the chip must show in place of its generic tooltip,
    empty whenever the generic one is exact: ``Session.calibration_issues``' own sentence
    for DAMAGED, and for NO_CHANNEL the strip's own CALIBRATE reason when the user's
    "Treat this channel as" override is why there is no current channel. That second case
    was measured missing: after IN15 (header current) was re-declared a voltage, the strip's
    CALIBRATE tooltip read "(IN15 is declared current by the file but your Calibration-panel
    override treats it as voltage)" while the chip one glance away still said "This
    recording has no live current channel to calibrate." -- the reason was derived once and
    reached three of its four consumers. Carrying it here is what lets the chip and the
    strip read alike, the way the DAMAGED chip and the File info panel already do.
    """

    state: CalibrationState
    label: str
    detail: str = ""
    channel: int | None = None
    issue: str = ""

    @property
    def is_uncalibrated(self) -> bool:
        return self.state is CalibrationState.UNCALIBRATED

    @property
    def is_calibrated(self) -> bool:
        return self.state is CalibrationState.CALIBRATED

    @property
    def is_damaged(self) -> bool:
        return self.state is CalibrationState.DAMAGED


# ================================================================================================
# Pipeline stages — the strip's model
# ================================================================================================

class Stage(Enum):
    """The fixed analysis pipeline, in order. Values are stable string ids the strip emits
    on click and ``main_window`` maps to an existing reveal action."""

    OPEN = "open"
    CALIBRATE = "calibrate"
    FILTER = "filter"
    DETECT = "detect"
    CURATE = "curate"
    ANALYZE = "analyze"


PIPELINE_ORDER: tuple[Stage, ...] = (
    Stage.OPEN, Stage.CALIBRATE, Stage.FILTER, Stage.DETECT, Stage.CURATE, Stage.ANALYZE,
)

# The optional FILTER stage never blocks and is never the "current" step: current_stage
# walks Open → Calibrate → Detect → Curate → Analyze.
_CURRENT_WALK: tuple[Stage, ...] = (
    Stage.OPEN, Stage.CALIBRATE, Stage.DETECT, Stage.CURATE, Stage.ANALYZE,
)

_STAGE_LABELS: dict[Stage, str] = {
    Stage.OPEN: "Open",
    Stage.CALIBRATE: "Calibrate",
    Stage.FILTER: "Filter",
    Stage.DETECT: "Detect",
    Stage.CURATE: "Curate",
    Stage.ANALYZE: "Analyze",
}


@dataclass(frozen=True)
class StageStatus:
    """One stage of the strip.

    ``done``       — the step's goal is met.
    ``available``  — the step can be acted on now (its precondition is met).
    ``optional``   — informational; never blocks and is never "current" (FILTER only).
    ``current``    — the earliest available, not-done, non-optional step (the "you are here").
    ``reason``     — what the step does, or why it is locked; the tooltip and the status-bar
                     message shown when a locked stage is clicked.
    """

    stage: Stage
    label: str
    done: bool
    available: bool
    reason: str
    optional: bool = False
    current: bool = False


@dataclass(frozen=True)
class PipelineStatus:
    """Everything the strip and chip need for one (recording, channel) target."""

    calibration: CalibrationStatus
    stages: tuple[StageStatus, ...]
    current_stage: Stage | None

    def stage(self, stage: Stage) -> StageStatus:
        """The :class:`StageStatus` for ``stage`` (always present — the pipeline is fixed)."""
        for st in self.stages:
            if st.stage is stage:
                return st
        raise KeyError(stage)  # pragma: no cover - stages always covers PIPELINE_ORDER


# ================================================================================================
# Derivation
# ================================================================================================

def _resolve_current_channel(session: "Session", doc_id: str, channel: int | None) -> int | None:
    """Map the selection to the current channel the pipeline acts on — THROUGH THE SESSION'S
    resolver (``Session.resolve_current_channel``), which is the same policy the analysis
    panel uses and the same kind (``Session.effective_kind``) the panes rank by: keep the
    selection if it is already a live current channel, else the recording's first live
    current channel, else ``None``. Not re-implemented here on purpose: a second copy that
    read the header was how this module came to deny a channel the app had just calibrated."""
    return session.resolve_current_channel(doc_id, channel)


#: Header kinds under which a "Treat this channel as" override to a NON-current kind is why
#: the recording resolves to no current channel: the header said current and the override
#: took it away, or the header said nothing (``UNKNOWN`` / ``RAW_VOLTAGE``) and the override
#: chose something other than current for it. A re-declaration between two declared
#: non-current kinds (voltage <-> conductance) is not why there is no current channel and is
#: deliberately NOT named in that sentence.
_OVERRIDE_TAKES_THE_CURRENT_FROM: frozenset[ChannelKind] = frozenset(
    (ChannelKind.CURRENT, ChannelKind.UNKNOWN, ChannelKind.RAW_VOLTAGE))


def _overridden_away_note(session: "Session", doc_id: str) -> str:
    """Why a recording resolves to no current channel when the user's OWN "Treat this channel
    as" override is the reason -- the clause (no punctuation) the locked reasons and the
    chip's sentence are built from, or ``""`` when no override is involved.

    Named so the locked strip and chip do not read as "this recording cannot be analysed"
    when the real state is "you told the app this is a voltage". It covers BOTH shapes the
    override exists for: a channel the file declared current ("IN15 is declared current by
    the file but your Calibration-panel override treats it as voltage") and a channel the
    file left undeclared ("the file says Untitled is unknown and your Calibration-panel
    override treats it as voltage"). The wave-1 note listed only the first, and the
    consequence was measured on the undeclared 2 s CSV of the owner's -700 mV recording:
    corrected to current, detected on (15 events), then set back to voltage, the Detect gate
    read "this is a voltage channel (your Calibration-panel override; the file says
    unknown)" while the rail read "No current channel on this recording to detect on." --
    the user's own override, the only reason the channel the app had detected on a minute
    ago was gone, named nowhere on the two guided surfaces.

    "Override" is read as the session's effective kind differing from the header -- the same
    resolver every surface uses, and the same test the analysis panel's gate applies before
    it says "your Calibration-panel override"."""
    meta = session.document(doc_id).meta
    parts: list[str] = []
    for info in meta.channels:
        if info.is_dead or info.kind not in _OVERRIDE_TAKES_THE_CURRENT_FROM:
            continue
        treated = session.effective_kind(doc_id, info.index)
        if treated is info.kind or treated is ChannelKind.CURRENT:
            continue
        name = info.name or f"channel {info.index}"
        if info.kind is ChannelKind.CURRENT:
            parts.append(f"{name} is declared current by the file but your "
                         f"Calibration-panel override treats it as {treated.value}")
        else:
            parts.append(f"the file says {name} is {info.kind.value} and your "
                         f"Calibration-panel override treats it as {treated.value}")
    return ", ".join(parts)


def _no_current_reason(verb: str, note: str) -> str:
    """THE locked sentence -- "No current channel on this recording to <verb>." -- with the
    override clause in parentheses when there is one. One builder for the three locked
    stage reasons and the chip's sentence, so the four can never disagree about why."""
    return (f"No current channel on this recording to {verb} ({note})." if note
            else f"No current channel on this recording to {verb}.")


def _is_user_calibrated(session: "Session", doc_id: str, info) -> bool:
    """The authoritative "gain is user-set, not the default" test — byte-for-byte the notion
    ``core.calibration.save_sidecar`` uses to decide a channel differs from its default: the
    live calibration's (gain, offset) fingerprint OR its target units differ from
    ``CalibrationSet.default_for(info)``. For a current channel the default is identity, so
    this is true exactly when a real gain has been dialled in."""
    cal = session.calibration(doc_id, info.index)
    default = CalibrationSet.default_for(info)
    return not (cal.fingerprint() == default.fingerprint()
                and cal.target_units == default.target_units)


def _calibration_detail(session: "Session", doc_id: str, info) -> str:
    """A compact 'gain → units' summary for the CALIBRATED chip, e.g. ``×2500 → nS``."""
    cal = session.calibration(doc_id, info.index)
    units = cal.target_units or info.raw_units
    detail = f"×{cal.gain:g}"
    if cal.offset:
        detail += f" {cal.offset:+g}"
    if units:
        detail += f" → {units}"
    return detail


def _detection_eventset(session: "Session", doc_id: str, channel: int) -> EventSet | None:
    """The stored detection :class:`EventSet` for ``(doc_id, channel)``, or ``None``.

    Reuses the shared detector registry exactly as the analysis panel does (including the
    derived reconciled table), then any other result on this channel whose payload is an
    EventSet, so a future detector still registers as "detected". Fit results (whose payload
    is not an EventSet) are correctly ignored."""
    for mid in DETECTION_MODULE_IDS:
        res = session.results.get(doc_id, channel, mid)
        if res is not None and isinstance(res.payload, EventSet):
            return res.payload
    for res in session.results.for_doc(doc_id):
        if res.channel == channel and isinstance(res.payload, EventSet):
            return res.payload
    return None


def _accepted_count(events: EventSet) -> int:
    """Number of curated-in (accepted) events — the CURATE 'done' quantity."""
    return int(events.df["accepted"].sum())


def compute_status(session: "Session", doc_id: str | None, channel: int | None) -> PipelineStatus:
    """Derive the full :class:`PipelineStatus` for the ACTIVE target ``(doc_id, channel)``.

    Robust to every degenerate input: ``doc_id`` ``None`` or not open, a recording with no
    current channel, a dead current channel, a user-set gain that overflowed the channel
    (DAMAGED: CALIBRATE available and not done, its reason the session's own damage
    sentence, so the "you are here" lands on the remedy). OPEN is derived from the number of open
    recordings (not from the target), so the strip reports "a recording is open" even before a
    selection lands; every later stage is about the target recording's current channel.

    ``current_stage`` is the earliest AVAILABLE, not-done stage in
    ``Open → Calibrate → Detect → Curate → Analyze`` (the optional FILTER is skipped). It is
    OPEN on an empty session, advances as the user progresses, is ANALYZE once there is a
    curated detection (a terminal action, never marked done), and is ``None`` when nothing in
    the pipeline is actionable (e.g. a recording that has no current channel).
    """
    has_doc = doc_id is not None and session.has_document(doc_id)
    any_open = len(session.documents()) >= 1

    # -- resolve the current channel + calibration --------------------------------------------
    cur_ch: int | None = None
    cur_info = None
    no_current_note = ""
    if has_doc:
        meta = session.document(doc_id).meta
        cur_ch = _resolve_current_channel(session, doc_id, channel)
        if cur_ch is not None:
            cur_info = meta.channels[cur_ch]
        else:
            no_current_note = _overridden_away_note(session, doc_id)

    if cur_info is None:
        calibration = CalibrationStatus(
            state=CalibrationState.NO_CHANNEL,
            label="No current channel",
            channel=None,
            # The chip's sentence is the strip's CALIBRATE reason, verbatim, when the user's
            # own override is why; empty when the chip's generic tooltip is exact.
            issue=_no_current_reason("calibrate", no_current_note) if no_current_note else "",
        )
    elif _is_user_calibrated(session, doc_id, cur_info):
        # A user-set gain that RUINED the channel is reported as damage, never as
        # "calibrated": Session.calibration_issues is the one authority on what apply()
        # did to the samples (memoised per calibration; reads only the cached raw extremes
        # in the normal case), and the File info panel already prints its sentence. The
        # chip and the strip must agree with that panel, not contradict it in green.
        issues = session.calibration_issues(doc_id, cur_ch)
        if issues:
            calibration = CalibrationStatus(
                state=CalibrationState.DAMAGED,
                label="Calibration ruined the channel",
                detail=_calibration_detail(session, doc_id, cur_info),
                channel=cur_ch,
                issue=" ".join(issues),
            )
        else:
            calibration = CalibrationStatus(
                state=CalibrationState.CALIBRATED,
                label="Calibrated",
                detail=_calibration_detail(session, doc_id, cur_info),
                channel=cur_ch,
            )
    else:
        calibration = CalibrationStatus(
            state=CalibrationState.UNCALIBRATED,
            label="Uncalibrated",
            channel=cur_ch,
        )

    # -- detection + curation state -----------------------------------------------------------
    events = _detection_eventset(session, doc_id, cur_ch) if (has_doc and cur_ch is not None) else None
    has_detection = events is not None
    accepted = _accepted_count(events) if events is not None else 0

    has_current = cur_ch is not None

    # -- assemble the stages ------------------------------------------------------------------
    raw: dict[Stage, StageStatus] = {}

    raw[Stage.OPEN] = StageStatus(
        stage=Stage.OPEN,
        label=_STAGE_LABELS[Stage.OPEN],
        done=any_open,
        available=True,
        reason=("A recording is open." if any_open
                else "Open a recording (.abf, .h5, .tdms, .csv) to begin."),
    )

    if has_current:
        cal_done = calibration.state is CalibrationState.CALIBRATED
        if cal_done:
            cal_reason = "Current gain set — depth, ECD and conductance are physical."
        elif calibration.state is CalibrationState.DAMAGED:
            # Not done, still available: the strip's "you are here" lands on CALIBRATE
            # with the damage sentence, which is where the remedy is.
            cal_reason = (f"The current-channel gain ({calibration.detail}) ruined the "
                          f"channel, so nothing measured on it is physical: {calibration.issue}")
        else:
            cal_reason = ("Set the current-channel gain so blockage depth, ECD and conductance "
                          "are physical (they are non-physical at the default gain).")
        raw[Stage.CALIBRATE] = StageStatus(
            stage=Stage.CALIBRATE, label=_STAGE_LABELS[Stage.CALIBRATE],
            done=cal_done, available=True, reason=cal_reason,
        )
    else:
        reason = (_no_current_reason("calibrate", no_current_note)
                  if has_doc else "Open and select a recording first.")
        raw[Stage.CALIBRATE] = StageStatus(
            stage=Stage.CALIBRATE, label=_STAGE_LABELS[Stage.CALIBRATE],
            done=False, available=False, reason=reason)

    raw[Stage.FILTER] = StageStatus(
        stage=Stage.FILTER, label=_STAGE_LABELS[Stage.FILTER],
        done=False, available=has_current, optional=True,
        reason=("Optional: pre-filter the current signal (low-pass, notch, detrend, "
                "baseline) before detection." if has_current else
                _no_current_reason("filter", no_current_note)),
    )

    if has_current:
        raw[Stage.DETECT] = StageStatus(
            stage=Stage.DETECT, label=_STAGE_LABELS[Stage.DETECT],
            done=has_detection, available=True,
            reason=("Events detected." if has_detection else
                    "Detect translocation events on the current channel (Schmitt or ML)."),
        )
    else:
        reason = (_no_current_reason("detect on", no_current_note)
                  if has_doc else "Open and select a recording first.")
        raw[Stage.DETECT] = StageStatus(
            stage=Stage.DETECT, label=_STAGE_LABELS[Stage.DETECT],
            done=False, available=False, reason=reason)

    if has_detection:
        curate_done = accepted >= 1
        raw[Stage.CURATE] = StageStatus(
            stage=Stage.CURATE, label=_STAGE_LABELS[Stage.CURATE],
            done=curate_done, available=True,
            reason=(f"{accepted} accepted event(s)." if curate_done else
                    "Review detected events and accept/reject them in the Event inspector "
                    "(nothing is accepted yet)."),
        )
        raw[Stage.ANALYZE] = StageStatus(
            stage=Stage.ANALYZE, label=_STAGE_LABELS[Stage.ANALYZE],
            done=False, available=True,
            reason="Explore the events: plots, density / ensemble, I–V and exports.",
        )
    else:
        raw[Stage.CURATE] = StageStatus(
            stage=Stage.CURATE, label=_STAGE_LABELS[Stage.CURATE],
            done=False, available=False,
            reason="Detect events first, then curate them.")
        raw[Stage.ANALYZE] = StageStatus(
            stage=Stage.ANALYZE, label=_STAGE_LABELS[Stage.ANALYZE],
            done=False, available=False,
            reason="Detect events first to analyze them.")

    # -- pick the current stage ---------------------------------------------------------------
    current_stage: Stage | None = None
    for stage in _CURRENT_WALK:
        st = raw[stage]
        if st.available and not st.done:
            current_stage = stage
            break

    stages = tuple(
        raw[s] if raw[s].stage is not current_stage
        else StageStatus(stage=raw[s].stage, label=raw[s].label, done=raw[s].done,
                         available=raw[s].available, reason=raw[s].reason,
                         optional=raw[s].optional, current=True)
        for s in PIPELINE_ORDER
    )

    return PipelineStatus(calibration=calibration, stages=stages, current_stage=current_stage)
