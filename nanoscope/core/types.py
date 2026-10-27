from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .acquisition import AcquisitionBandwidth

SOFT_SAMPLE_CEILING: int = 15_000_000   # BINDING. Above this: warn + offer a range. Never nag below.
CACHE_BUDGET_MB: int = 1024
LOAD_CHUNK_SAMPLES: int = 2_000_000     # readers load in chunks: progress granularity + cancel polling
DEAD_PROBE_SAMPLES: int = 20_000
DEAD_LSB_TOL: float = 8.0


class ChannelKind(Enum):
    CURRENT = "current"
    VOLTAGE = "voltage"          # a command/measured voltage in mV
    CONDUCTANCE = "conductance"
    RAW_VOLTAGE = "raw_voltage"  # volts straight off an ADC; meaning unknown until the user sets a gain
    UNKNOWN = "unknown"
    # NO TIME MEMBER. Time is the axis (RecordingMeta.sample_rate_hz + t0_s), never a channel.
    # HDF5 /time is read for its first 1000 samples to derive dt, then DROPPED. Never expose it.


class Provenance(Enum):
    MEASURED = "measured"    # read out of the signal itself
    FILE_META = "file_meta"  # declared in the file header
    FILENAME = "filename"    # parsed from the filename — UNRELIABLE. Render visibly differently.
    USER = "user"            # declared by the user in Inspector ▸ File Info; saved with the project
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChannelInfo:
    index: int                      # position in TraceDocument.channels; stable for the document's life
    name: str                       # "IN15", "current", "Dev2/ai0"
    raw_units: str                  # verbatim-from-disk, normalised for spelling only ("PA"->"pA",
                                    # "Volts"->"V"). "" means NOT DECLARED ON DISK. Never a guess.
    kind: ChannelKind
    is_dead: bool = False           # a HINT from detect_dead(), set at load, never by probe()
    dead_reason: str | None = None  # e.g. "no signal: max|x|=9.16e-04 <= 8x quantisation step 3.05e-04"
    acquisition_bandwidth: AcquisitionBandwidth = field(default_factory=AcquisitionBandwidth)
    # File declaration only. Session-owned user corrections never overwrite this header evidence.


@dataclass(frozen=True)
class SampleRange:
    """Half-open [start, stop). The unit of partial loading."""
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


#: The top-level HDF5 attribute key nanoscope/gui/study_event_bridge.py stamps onto a
#: prepared-review export. nanoscope/io/hdf5.py reads it to set
#: RecordingMeta.physical_reference_unavailable. Defined here (not in gui/) so io/ can
#: import it without an io -> gui dependency.
STUDY_BRIDGE_ATTR_KEY = "study_event_bridge"


@dataclass(frozen=True)
class RecordingMeta:
    path: Path
    format_name: str                        # "Axon ABF", "HDF5 flat", "TDMS (modern)", "TDMS (legacy)", "Text trace"
    n_samples: int                          # after any partial load; == len of every channel array
    sample_rate_hz: float | None            # None == GENUINELY UNKNOWN (legacy TDMS). Never fabricate 50 kHz.
    rate_provenance: Provenance
    channels: tuple[ChannelInfo, ...]
    t0_s: float = 0.0                       # experimental time of sample 0. READ FROM THE FILE, NEVER the filename.
    loaded_range: SampleRange | None = None  # None == whole file
    nominal_voltage_mV: float | None = None
    voltage_provenance: Provenance = Provenance.UNKNOWN
    notes: tuple[str, ...] = ()             # honest advisories shown in file_info_panel
    issues: tuple[str, ...] = ()            # non-fatal problems found while reading
    physical_reference_unavailable: bool = False
    # ^^ True for a document engineered with no real open-pore/physical reference (e.g. a
    # baseline-subtracted export from the manual-study bridge). ANY detector run against
    # such a document must refuse frac_depth, not just the bridge's own synthetic result
    # -- see core/acquisition.py's stamp_reference_availability, which every detector
    # calls, and characterize.py/events.py/reconcile.py's shared
    # fractional_blockage_reference_unavailable guard, which was already correct but had
    # only one producer before this field existed.
    load_uid: str = ""
    # ^^ THE PER-LOAD TOKEN. Identifies THIS load of this file, so that two loads of one
    # path are two different documents and a result computed on one can never be installed
    # onto the other (see core/identity.py, which owns the whole question). Stamped ONCE by
    # TraceDocument.__post_init__ — the single funnel every recording passes through — and
    # preserved by every ``dataclasses.replace`` after that, so a doc_id is stable for the
    # life of the load. "" means a meta that never became a document (a probe() result);
    # readers do not set it and must not.
    time_scale_s: float = 1.0
    # ^^ SECONDS PER UNIT OF THE FILE'S OWN TIME COLUMN, AS IT WAS READ. 1.0 for every format
    # that declares its rate (ABF, TDMS) and for a time column read as seconds; 1e-3 when a
    # CSV header named milliseconds (``time_ms``) and the reader scaled the column to seconds
    # on the way in. It is how ``sample_rate_hz`` and ``t0_s`` relate to the numbers in the
    # file, so a later declaration (core/sample_rate.py: "the column is in microseconds, not
    # milliseconds") can be resolved against what was read rather than typed over it.
    refused_rate_hz: float | None = None
    # ^^ THE RATE THE READER MEASURED AND REFUSED TO REPORT: a time column stepping by
    # 100 per sample reads as 0.01 Hz, which no recording is, so ``sample_rate_hz`` is
    # None (io.base.MIN_PLAUSIBLE_RATE_HZ) — but the interval was measured, and it is
    # exactly what a declaration of the column's unit re-reads ("those numbers are
    # microseconds": 0.01 Hz x 10^6). Kept here so that remedy exists without guessing.
    # None whenever the rate was not refused on that ground.

    def __post_init__(self) -> None:
        """Refuse a meta whose load window and sample count disagree.

        HERE, on the value object, and not in the four readers — for the same reason
        ``TraceDocument`` refuses an empty recording at its own door: a rule written in a
        reader has to be written four times and still misses the fifth. Three readers
        clamped their requested window against the file length and one assigned the request
        verbatim, so ``SampleRange(500, 8000)`` on a 2,000-row CSV produced
        ``n_samples=1500`` beside ``loaded_range=(500, 8000)``. Nothing downstream can tell
        those apart from a legitimate 7,500-sample window, and the wrong one is what the
        project manifest stored and replayed on the next restore.

        ``loaded_range`` is not "what was asked for". It is "which slice of the file this
        document holds", so its length IS ``n_samples``, by definition rather than by
        coincidence. :func:`nanoscope.io.contract.window_loaded` computes it from the number
        of samples actually read, which makes this check unreachable from the four shipped
        readers; it exists so a fifth cannot reintroduce the defect quietly.

        A ``ValueError`` rather than a :class:`~nanoscope.io.errors.ReaderError`, because a
        window that disagrees with its own contents is a bug in the producer, never a
        property of the user's file — every genuine file condition (a request past the end,
        a short read) is answered before this point, out loud, by the reader.
        """
        lr = self.loaded_range
        if lr is None:
            return
        if int(lr.start) < 0 or int(lr.stop) <= int(lr.start):
            raise ValueError(
                f"{self.path.name}: loaded_range {lr} is not a non-empty half-open window.")
        span = int(lr.stop) - int(lr.start)
        if span != int(self.n_samples):
            raise ValueError(
                f"{self.path.name}: loaded_range {lr} spans {span:,} samples but n_samples is "
                f"{int(self.n_samples):,}. A load window records what was READ, never what was "
                f"requested.")

    @property
    def is_partial(self) -> bool:
        return self.loaded_range is not None

    @property
    def duration_s(self) -> float | None:
        if self.sample_rate_hz is None:
            return None
        return self.n_samples / self.sample_rate_hz

    @property
    def stem_key(self) -> str:
        """Duplicate-representation detection: '..._-600 mV.abf' vs '..._-600 mV.csv'."""
        s = self.path.stem
        for suffix in (" Saved Data", "_Cut_1", "-Copy1", "_Cut_1-Copy1"):
            s = s.replace(suffix, "")
        return " ".join(s.split()).strip().lower()

    @property
    def voltage_group_mV(self) -> float | None:
        """Grouping key for file_dock. [VERIFIED] -699.89 -> -700.0 ; -999.9 -> -1000.0 ; -499.91 -> -500.0"""
        if self.nominal_voltage_mV is None:
            return None
        return round(self.nominal_voltage_mV / 10.0) * 10.0


# --- Display policy ---------------------------------------------------------------------------
# Keyed by EFFECTIVE UNITS (lowercased), NOT by ChannelKind.
# WHY: a CURRENT channel may be pA (ABF IN15) or an undeclared nA-scale trace (HDF5 /current).
# A kind-keyed 5.0 minimum span would flatten the HDF5 trace (whose whole signal spans 0.05).
# We impose a physical minimum span ONLY when we actually know the physical unit.
MIN_Y_SPAN_BY_UNITS: dict[str, float] = {
    "pa": 5.0,      # pA
    "na": 0.005,    # nA
    "mv": 50.0,     # mV  <- [VERIFIED] fixes the constant-voltage pane (see below)
    "v": 0.05,      # V
    "ns": 0.5,      # nS
}
# [VERIFIED] ABF IN14 is a constant command voltage whose ENTIRE excursion is 1.83 mV. Without a
# minimum span, y-autorange renders it as a full-height pane of pure ADC quantisation noise —
# the DEFAULT appearance of the voltage pane on every normal file. minYRange=50 fixes it.

# The trace view's overlay inks. DEFINED IN `core.layer` (as LAYER_INKS) and re-exported
# under the old name so the ~3 existing `from .types import LAYER_PALETTE` importers keep
# working. It moved because it is now one half of ONE colour identity — `core.layer` owns
# the registry that maps a recording to a slot, and a palette that lives away from the map
# that indexes it is how the two drift. The old Material swatches were NOT colour-vision
# safe (min dE2000 3.05 under tritanopia, 5.07 under deuteranopia); see LAYER_INKS for the
# measurements and for why a naive value-lift of Okabe-Ito measured worse still.
from .layer import LAYER_INKS as LAYER_PALETTE  # noqa: E402  (kept beside the display policy)


def normalise_units(raw: str) -> str:
    """Spelling normalisation ONLY. NEVER a scale inference. Unknown/blank stays ""."""
    table = {"pa": "pA", "picoamps": "pA", "na": "nA", "nanoamps": "nA",
             "mv": "mV", "millivolts": "mV", "v": "V", "volts": "V",
             "ns": "nS", "nanosiemens": "nS"}
    return table.get(raw.strip().lower(), raw.strip())


def kind_for_units(units: str) -> ChannelKind:
    return {"pA": ChannelKind.CURRENT, "nA": ChannelKind.CURRENT,
            "mV": ChannelKind.VOLTAGE, "V": ChannelKind.RAW_VOLTAGE,
            "nS": ChannelKind.CONDUCTANCE}.get(normalise_units(units), ChannelKind.UNKNOWN)


#: How many samples a non-finite scan looks at per pass. Bounds the transient bool mask
#: (1 byte per sample) at 4 MiB, whatever the recording length.
NONFINITE_SCAN_CHUNK: int = 1 << 22


def scan_nonfinite(x: np.ndarray) -> tuple[int, int]:
    """``(count, first_index)`` of the NaN/inf samples in ``x``; ``(0, -1)`` when clean.

    Chunked so the temporary mask is bounded regardless of recording length: a 50 M-sample
    channel costs ~4 MB of scratch instead of 50 MB.

    ``~isfinite`` + ``count_nonzero`` is what it looks like on purpose. Two cheaper-looking
    screens were measured and are both SLOWER on this data: a float64 ``sum()`` pre-check
    (0.75 ms vs 0.42 ms at 3 M samples; 12.45 vs 7.67 at 50 M — the float32→float64
    accumulation costs more than the vectorised mask it was meant to avoid) and reusing the
    mask via ``bad.sum()`` instead of ``count_nonzero`` (0.85 / 15.20 ms).
    """
    n = int(x.size)
    total = 0
    first = -1
    for start in range(0, n, NONFINITE_SCAN_CHUNK):
        block = np.asarray(x[start:start + NONFINITE_SCAN_CHUNK])
        bad = ~np.isfinite(block)
        k = int(np.count_nonzero(bad))
        if k:
            if first < 0:
                first = start + int(np.argmax(bad))
            total += k
    return total, first


def nonfinite_reason(count: int, total: int, first: int) -> str:
    """How badly a block of samples is non-finite — the ONE writer of that sentence.

    "Entirely non-finite" and "one sample in 15 M is non-finite" are different
    conditions with different causes and different remedies, and they must not read
    the same:

    * **Every** sample bad is not a recording with a defect, it is **no recording at
      all** — an infinite/zero gain, a unit conversion that overflowed float32, a
      channel read past the end of the file. Cropping cannot help; the acquisition or
      the calibration has to be fixed. This is the shape :func:`detect_dead` describes
      for a *flat* channel, one step further gone.
    * **A few** bad samples in a long trace is a localised **taint**: the recording is
      real and the rest of it is usable, so naming *where* is the actionable part —
      the user can crop past it or re-read the file.

    Every refusal in the app that counts non-finite samples formats its count through
    here (``nonfinite_issue`` at the load boundary, ``baseline.estimate_baseline`` and
    ``baseline.derive_polarity``, ``measure.measure_selection``), so the same condition
    is described the same way wherever the user meets it.

    Args:
        count: How many samples are non-finite.
        total: How many samples were looked at.
        first: Index of the first bad sample, in whatever coordinates the caller wants
            the user to act on (a sample number in the recording, usually), or < 0 to
            leave the location out.

    Returns:
        A sentence with no leading subject, e.g. "every one of 3,000 samples is
        non-finite (NaN/Inf) — ..."; the caller supplies what it is talking about.
    """
    count = int(count)
    total = int(total)
    where = f" (first at index {int(first):,})" if int(first) >= 0 else ""
    if total > 0 and count >= total:
        return (f"every one of {total:,} samples is non-finite (NaN/Inf) — a dead or "
                f"mis-calibrated channel (a zero/infinite gain, or a conversion that "
                f"overflowed float32), not a recording with a few bad samples.")
    verb = "is" if count == 1 else "are"
    return (f"{count:,} of {total:,} samples {verb} non-finite (NaN/Inf){where} — a "
            f"localised taint, not a dead channel.")


def nonfinite_issue(channel_index: int, name: str, count: int, first: int, n: int) -> str:
    """The meta.issues sentence for a channel carrying non-finite samples."""
    return (f"channel {channel_index} ({name or 'unnamed'}): "
            f"{nonfinite_reason(count, n, first)} Any statistic computed over them — RMS "
            f"noise, baseline, event depth — is undefined; a spectrum is poisoned in every "
            f"bin.")


def detect_dead(x: np.ndarray,
                n_probe: int = DEAD_PROBE_SAMPLES,
                lsb_tol: float = DEAD_LSB_TOL) -> tuple[bool, str | None]:
    """Return (is_dead, reason). A channel is dead if its entire excursion from zero lies within a
    few quantisation steps — i.e. it carries no signal, only ADC noise.

    *** READ THIS. THE BRIEF IS WRONG ON THIS POINT AND SO IS THE OBVIOUS IMPLEMENTATION. ***
    The brief states ABF IN13 is "ALL ZEROS". [VERIFIED] IT IS NOT. Over the whole -700 mV file,
    IN13 is +/-5 LSB of noise about a median of 0.0003 V:
        np.all(x == 0)   -> False
        np.ptp(x) == 0.0 -> False        (ptp is 0.00152588 V)
    BOTH obvious tests FAIL to flag it, leaving a full-height pane of quantisation noise on screen.

    [VERIFIED] whole-file measurements, ABF -700 mV, this rule:
        IN13  lsb=3.05e-04 V   max|x|=9.16e-04 V   max|x|/lsb =     3.0  -> DEAD
        IN14  lsb=0.305   mV   max|x|=6999.51  mV  max|x|/lsb = 22936.0  -> live (constant, meaningful)
        IN15  lsb=0.305   pA   max|x|=271.30   pA  max|x|/lsb =   889.0  -> live
    Separation is ~300x, so lsb_tol anywhere in 4..100 works. Note variance CANNOT separate IN13
    from IN14 (both span ~5-6 LSB); only distance-from-zero can.

    Fails safe: on genuinely continuous float data the estimated lsb is tiny, so max|x|/lsb is huge
    and the channel is never falsely condemned.
    """
    step = max(1, len(x) // n_probe)
    s = np.asarray(x[::step], dtype=np.float64)
    u = np.unique(s)
    if u.size == 1:
        if float(u[0]) == 0.0:
            return True, f"constant 0.0 (sampled {s.size})"
        return False, None
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return False, None
    lsb = float(d.min())
    mx = float(np.max(np.abs(s)))
    if mx <= lsb_tol * lsb:
        return True, (f"no signal: max|x|={mx:.3g} <= {lsb_tol:g}x quantisation step "
                      f"{lsb:.3g} (sampled {s.size})")
    return False, None
