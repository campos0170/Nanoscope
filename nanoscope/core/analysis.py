from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .document import CalibratedChannel
from .identity import SourceFingerprint
from .types import ChannelKind, RecordingMeta

# ================================================================================================
# THIS FILE IS THE ANSWER TO "point at exactly where v2 plugs in":
#   implement AnalysisModule, register it in ANALYSIS_REGISTRY, results land in Session.results.
# v1 implements NOTHING here. Do not add a detector, a filter, a fitter, or a classifier.
#
# SIGN CONVENTION (state it here once so v2-v5 never re-derive it; v1 writes NO code for it):
#   Stored data is SIGNED and is never rectified in place (Rule 5). [VERIFIED] the corpus contains
#   BOTH signs for the same physics: ABF/ferritin currents are ~-49 pA, while nisin_fibrin_trace.csv
#   is +0.295 (positive, and nA-scale). A detector that hard-codes "blockade = downward excursion"
#   is wrong on half the user's own files.
#   v2 MUST derive polarity from the sign of a robust baseline (median of a sample), work in
#   rectified space (signed * polarity => positive baseline, blockades downward), and treat
#   enhancing (upward) events as real physics rather than a sign bug. v1 renders signed.
# ================================================================================================

__all__ = [
    "AnalysisContext",
    "OverlayKind",
    "OverlayHint",
    "AnalysisResult",
    "AnalysisModule",
    "ResultStore",
    "ANALYSIS_REGISTRY",
    "register_module",
]


@runtime_checkable
class AnalysisContext(Protocol):
    """A module gets calibrated data through THIS and nothing else. It never imports Qt, never
    touches Session, never reaches into the cache, never draws. That is what makes v2 droppable
    rather than a rewrite — and it runs unchanged on the v1 worker."""

    @property
    def channel(self) -> CalibratedChannel: ...

    @property
    def meta(self) -> RecordingMeta: ...

    @property
    def params(self) -> Mapping[str, Any]: ...

    def report_progress(self, frac: float, msg: str = "") -> None: ...

    def cancelled(self) -> bool: ...


class OverlayKind(Enum):
    SPAN = "span"      # shaded [start, stop) regions — v2 events
    MARKER = "marker"
    STEP = "step"      # v4 fitted levels
    HLINE = "hline"    # baseline track


@dataclass(frozen=True)
class OverlayHint:
    """Modules DECLARE overlays as ARRAYS; they never construct graphics items.

    Two reasons this is declarative rather than a "return me some pg items" protocol:
      1. Returning pg.GraphicsObject would make analysis import pyqtgraph, coupling v2 to the GUI.
      2. It structurally prevents the 10k-LinearRegionItem trap. v2 will produce 1e3-1e5 events;
         one LinearRegionItem each is 1e5 full QGraphicsObjects with hover/drag machinery, which
         would destroy exactly the frame rate the no-pyramid decision exists to protect. Because
         the hint is arrays, the VIEW decides how to render and can paint N rects in a single
         paint() clipped to the visible x-range, with at most ONE real LinearRegionItem for the
         selected event.
    """

    kind: OverlayKind
    starts: np.ndarray                    # int64 SAMPLE INDICES (lossless; survive a rate fix)
    stops: np.ndarray | None = None
    values: np.ndarray | None = None
    label: str = ""
    color: str = "#FFB74D"
    name: str = ""
    # ^^ WHAT A RENDERER MAY PAINT, as distinct from ``label``, which is the overlay's
    # IDENTITY to whoever owns it. The distinction is not academic: the detection's span
    # overlay is found again -- by the panel that replaces it on every curation edit, and
    # by the project-IO tests that must tell it apart from the Measure panel's spans -- by
    # ``label == "events"``. Once the spans are cut into one hint per SUBPOPULATION they are
    # still, every one of them, the detection's spans, so they keep that label; what differs
    # between them is the human-readable name ("subpop 2") the span item writes over each
    # event once the view is zoomed in far enough for the text to be about one event. Reusing
    # ``label`` for that would have made "is this the detection's overlay?" depend on how many
    # classes a mixture happened to find.


@dataclass(frozen=True)
class AnalysisResult:
    module_id: str
    module_version: str
    doc_id: str
    channel: int
    params: Mapping[str, Any]
    calibration_fingerprint: tuple[float, float]
    # ^^ THE STALENESS STAMP. The user EXPLICITLY deferred the current-gain question, so the
    # calibration WILL be revised after events exist — that is the stated workflow, not an edge
    # case. Compare this to Session.calibration(doc_id, ch).fingerprint(); if they differ the
    # result's blockade depths/ECDs are silently wrong and the UI must mark it stale and offer a
    # re-run. v1 populates nothing, but the field must exist now: retrofitting it means auditing
    # every result path.
    payload: object = None                # v2: EventTable | v4: fits | v5: labels
    overlays: tuple[OverlayHint, ...] = ()
    scalars: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    baseline_fingerprint: str | None = None
    # ^^ THE OTHER STALENESS STAMP, and a sibling to calibration_fingerprint for exactly the
    # same reason. The open-pore estimator that produced this result's depths is a per-recording
    # choice the user confirms, so it WILL be revised — and when it is, depth_pA/frac_depth/
    # ecd_pA_s/mean_dev_pA/max_dev_pA/snr all move (measured on the owner's -600 mV apo file:
    # depth by 4.7x, frac_depth by 16x on the owner's apo file). Compare this to the live
    # baseline.BaselineConfig.fingerprint(fs); if they differ the result's depths were measured
    # against a different open-pore reference and the UI must mark it stale and offer a re-run.
    # NEVER silently re-measure: a number that changes 15x between two openings of the same
    # project is indistinguishable from data corruption.
    #
    # None means the result predates this stamp, or its detector was handed a baseline array by
    # its caller and so cannot name an estimator at all. Absence IS the signal — those are
    # precisely the rows measured by the block median that this change exists to replace.
    # The value is baseline.BaselineConfig.fingerprint(fs), the digest over
    # (method, rough_s, base_s, decim, mask_sigma, dilate_ms, window_s, despike_size).

    # Source evidence belongs to the result, including an explicitly unknown stamp.
    # Fresh computations obtain evidence from their immutable loaded document at save;
    # restored results retain recorded=True even when their historical stamp is None.
    source_fingerprint: SourceFingerprint | None = None
    source_fingerprint_recorded: bool = False


@runtime_checkable
class AnalysisModule(Protocol):
    id: str
    label: str
    version: str
    accepts_kinds: tuple[ChannelKind, ...]
    requires_time_base: bool
    # ^^ [VERIFIED] legacy TDMS ('example.tdms', group 'Current2.925730E-1', channels
    # 'Untitled'/'Untitled 1') has NO wf_increment: its rate is genuinely unknown. A detector sets
    # this True and the GUI DISABLES it with an honest reason rather than emitting fictional dwell
    # times — dwell time being the primary reported quantity of the entire field.

    def run(self, ctx: AnalysisContext) -> AnalysisResult: ...


class ResultStore:
    """Keyed by (doc_id, channel, module_id) and owned by SESSION, not by the document.
    THAT is what lets v3 population analysis iterate ACROSS documents — exactly the
    -1000/-700/-600/-500 voltage-dependence plot. Hang results off the document instead and v3
    has nowhere to stand."""

    def __init__(self) -> None:
        self._results: dict[tuple[str, int, str], AnalysisResult] = {}

    def put(self, res: AnalysisResult) -> None:
        self._results[(res.doc_id, res.channel, res.module_id)] = res

    def get(self, doc_id: str, channel: int, module_id: str) -> AnalysisResult | None:
        return self._results.get((doc_id, channel, module_id))

    def for_module(self, module_id: str) -> list[AnalysisResult]:   # <- v3 reads across docs
        return [r for k, r in self._results.items() if k[2] == module_id]

    def for_doc(self, doc_id: str) -> list[AnalysisResult]:
        return [r for k, r in self._results.items() if k[0] == doc_id]

    def all(self) -> list[AnalysisResult]:
        """Every stored result, across all docs/channels/modules. Used by project SAVE to
        gather the detection/ML EventSets to persist (it filters to EventSet payloads itself)."""
        return list(self._results.values())

    def drop(self, doc_id: str, channel: int, module_id: str) -> bool:
        """Forget ONE result. Returns whether there was anything to forget.

        This exists for results that can be RETIRED rather than replaced. A CUSUM level
        table belongs to the exact event table it was fitted on, so re-running detection
        under it does not supersede it -- it invalidates it, and there is no new value to
        put() in its place. Leaving the dead entry here was harmless only for as long as
        nothing read it: once project SAVE began persisting derived results, a retired
        level table was written into the file and came back attached to a detection it was
        never measured on, with level rows naming event_ids that now mean different events.
        """
        return self._results.pop((doc_id, int(channel), module_id), None) is not None

    def drop_doc(self, doc_id: str) -> None:
        for key in [k for k in self._results if k[0] == doc_id]:
            del self._results[key]


ANALYSIS_REGISTRY: dict[str, AnalysisModule] = {}   # v2 registers here. EMPTY in v1.


def register_module(m: AnalysisModule) -> AnalysisModule:
    """Decorator//registrar. Duplicate ids are a programming error, not a runtime condition."""
    if m.id in ANALYSIS_REGISTRY:
        raise ValueError(f"analysis module id {m.id!r} is already registered")
    ANALYSIS_REGISTRY[m.id] = m
    return m


# Rejected from the seam: `ParamSpec` + generic form builder (no producers and no forms in v1;
# purely additive when v2 exists), and `SignalStage` pipeline (a one-stage abstraction with no
# second implementation is speculative generality — the property that matters, *the viewer and a
# future detector consume the same data*, is guaranteed because Session.calibrated() is the
# single source).
