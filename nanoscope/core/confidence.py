"""The ML detector's probability, KEPT with its result: :class:`ConfidenceTrack`.

`ml_detect.ml_detect_regions` has always returned the model's probability next to the
regions it decoded from it, and the adapter (`ml_modules.MLDetect.run`) threw it away the
moment the regions were out (``regions, _prob, ml_prov = ...``). So nothing could show WHY
the detector fired where it did, or how close it came where it did not: the one quantity
an ML table is made from was the one thing the app could not draw.

WHAT IS KEPT, AND AT WHAT RESOLUTION. The model's output contract
(`ml_detect._check_output_contract`) requires a PER-SAMPLE output, ``(None, W, 1)``; a model
whose output is one number per window is refused at load. Overlap-averaging those outputs
gives one probability per analysed sample, so the sample IS the native resolution, and that
is what is kept. Anything coarser -- a mean per hop, a max per window -- would be a
decimation of the detector's own output, and nothing in this app subsamples scientific
output without saying so. What is saved instead is the PRECISION: the probability is
quantised to ``uint8`` (``code = round(p * 255)``, a step of 1/255, at most 0.002 off), one
byte per analysed sample. That is a quarter of the float32 trace `ml_detect` returns and an
eighth of a float64 one -- 15 MB at the 15 M-sample soft ceiling, against the 60 MB the
channel's own float32 samples already cost. The ML tab sets the thresholds to two decimals,
so the rounding is a fifth of the smallest threshold step a user can type. The track is a
READOUT: the decode ran on the float32 probability before it was quantised, and nothing
re-decodes from these codes.

WHAT IT IS ON. ``start`` is the first analysed sample in DOCUMENT coordinates (the slice
offset of a sub-range run), so the track lands on the samples it was computed from whatever
the user zoomed or nudged afterwards: a view maps ``start + i`` through the same x axis the
trace curve is drawn on. ``covered`` is how many samples from ``start`` some model window
actually saw. The two notebook tail policies leave the last stride with no window at all,
and `predict_probability` hands those samples back as probability 0 -- "never asked" written
as "certainly not". Recording the boundary lets a view leave them empty instead.

LIFETIME. Immutable on purpose: the array is read-only, and ``copy``/``deepcopy`` return the
track itself. Every edit of an event table rebuilds its result with ``dataclasses.replace``
(`modules.reconcile_result`, `history._snapshot`), so the track rides through merges,
curation, Edit > Undo and Redo BY REFERENCE -- ten CURATION steps on one run hold one track,
not ten. A RE-RUN is different: it makes a new track, and the undo step that keeps the
previous run keeps that run's own track with it, so each re-run step holds one more track.
That is why `history.EventHistory.nbytes` counts each distinct track the stack holds against
the undo byte ceiling (`history.MAX_BYTES`), which then drops old re-runs like any large
table. Equality is identity (``eq=False``), which `history._values_equal` already reads as
"unchanged": a checkpoint never sees a phantom difference in it.

NOT SAVED IN PROJECTS. `core.project` stores an event table as a parquet frame plus JSON
provenance, and nothing else per result; a probability array would need a new member type and
a project-version bump (an older build would read the file and silently strip the member on
its next save -- the argument the v6 note in `project.py` makes). So the track lives in the
session that ran the detector, and a surface that wants it after a reopen must SAY it is not
there (:data:`NOT_KEPT_NOTE`) rather than draw nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConfidenceTrack", "QUANT_LEVELS", "QUANT_STEP", "NOT_KEPT_NOTE",
           "quantize_probability"]

#: The top code of the quantised probability: ``p = code / QUANT_LEVELS``.
QUANT_LEVELS: int = 255
#: One quantisation step in probability units (the rounding error is half of this).
QUANT_STEP: float = 1.0 / QUANT_LEVELS

#: What a surface says for an ML table that has no kept probability: a table restored from a
#: project, or one detected before the probability was kept. One sentence, written here once,
#: so the trace pane, the status bar and the guide cannot disagree about why it is missing.
NOT_KEPT_NOTE: str = ("ML probability is not saved in projects — "
                      "re-run ML detection to draw it")

# Quantising runs in blocks so the float temporaries stay a few MB on a 15 M-sample run
# instead of three more full-length float32 copies beside the one being converted.
_BLOCK = 1 << 20


def quantize_probability(prob: np.ndarray) -> tuple[np.ndarray, int]:
    """``prob`` as ``uint8`` codes, and how many values had to be clipped to get there.

    Non-finite values and values outside [0, 1] are COUNTED, then clipped (NaN to 0). A
    sigmoid output cannot produce them, but a model with a linear or ``clip_activation``
    head can, and a silent clip would make a broken head look like a confident one.
    """
    p = np.asarray(prob).ravel()
    codes = np.empty(p.size, dtype=np.uint8)
    bad = 0
    for lo in range(0, p.size, _BLOCK):
        block = np.asarray(p[lo:lo + _BLOCK], dtype=np.float32)
        finite = np.isfinite(block)
        bad += int(np.count_nonzero(~finite | (block < 0.0) | (block > 1.0)))
        block = np.where(finite, np.clip(block, 0.0, 1.0), 0.0)
        codes[lo:lo + block.size] = np.rint(block * QUANT_LEVELS).astype(np.uint8)
    return codes, bad


@dataclass(frozen=True, eq=False)
class ConfidenceTrack:
    """The ML model's per-sample probability over one analysed range, quantised to uint8.

    ``codes[i]`` is the probability of document sample ``start + i``, as
    ``codes[i] / QUANT_LEVELS``. Build one with :meth:`from_probability`.
    """

    codes: np.ndarray = field(repr=False)
    #: First analysed sample, in document coordinates (the run's ``range_start``).
    start: int
    #: Sample rate the model ran at, Hz.
    fs: float
    #: The model's window length ``W`` and the stride between windows (``W // 2``).
    window: int
    hop: int
    #: How many windows were predicted.
    n_windows: int
    #: Samples from ``start`` that at least one window covered. Past it the model never ran.
    covered: int
    #: The thresholds the decode applied (to the median-smoothed probability).
    enter: float
    exit: float
    #: The decode's median-smoothing kernel, in samples (1 = none).
    smoothing_kernel: int = 1
    #: Values that were non-finite or outside [0, 1] before quantising (see
    #: :func:`quantize_probability`).
    n_out_of_range: int = 0
    model_file: str = ""

    @classmethod
    def from_probability(cls, prob: np.ndarray, *, start: int, fs: float, window: int,
                         hop: int, n_windows: int, covered: int, enter: float, exit: float,
                         smoothing_kernel: int = 1, model_file: str = "") -> "ConfidenceTrack":
        codes, bad = quantize_probability(prob)
        codes.setflags(write=False)
        return cls(codes=codes, start=int(start), fs=float(fs), window=int(window),
                   hop=int(hop), n_windows=int(n_windows),
                   covered=int(min(max(int(covered), 0), codes.size)),
                   enter=float(enter), exit=float(exit),
                   smoothing_kernel=int(smoothing_kernel), n_out_of_range=int(bad),
                   model_file=str(model_file))

    # -- immutability: shared, never duplicated ------------------------------------------------
    def __copy__(self) -> "ConfidenceTrack":
        return self

    def __deepcopy__(self, memo) -> "ConfidenceTrack":
        return self

    # -- the time base --------------------------------------------------------------------------
    @property
    def n(self) -> int:
        """Analysed samples the track spans."""
        return int(self.codes.size)

    @property
    def stop(self) -> int:
        """One past the last analysed sample, in document coordinates."""
        return self.start + self.n

    @property
    def covered_stop(self) -> int:
        """One past the last sample a model window saw, in document coordinates."""
        return self.start + self.covered

    @property
    def uncovered_s(self) -> float:
        """Duration of the analysed tail no model window reached, in seconds."""
        return (self.n - self.covered) / self.fs if self.fs > 0 else 0.0

    @property
    def nbytes(self) -> int:
        return int(self.codes.nbytes)

    # -- reading it -----------------------------------------------------------------------------
    def probability(self, lo: int | None = None, hi: int | None = None) -> np.ndarray:
        """The probability of document samples ``[lo, hi)`` as float32 (a NEW array),
        clipped to the analysed range; the whole track when both are omitted."""
        a = self.start if lo is None else int(lo)
        b = self.stop if hi is None else int(hi)
        a, b = max(a, self.start), min(b, self.stop)
        if b <= a:
            return np.empty(0, dtype=np.float32)
        return self.codes[a - self.start:b - self.start].astype(np.float32) * np.float32(QUANT_STEP)

    def at(self, index: int) -> float | None:
        """The probability at document sample ``index``; None outside the covered samples."""
        i = int(index) - self.start
        if i < 0 or i >= self.covered:
            return None
        return float(self.codes[i]) * QUANT_STEP
