"""Ground truth and null controls: the machinery that makes the numbers defensible.

This module answers four questions a reviewer will ask, with measurements rather
than assertions:

1. `synth_trace`   -- what does the detector do on data whose answer we know?
2. `null_control`  -- how many "events" does the detector find in the user's own
                      noise, with the events removed but the noise intact?
3. `sweep_threshold` -- is there a detection regime at all, or is the event count
                      just a function of the threshold the user happened to pick?
4. `recovery_map`  -- what fraction of events of a given (dwell, depth) does the
                      detector actually recover? This is a *measured efficiency*,
                      meant to weight the likelihood in `stats.fit_dwell_exponential`.

Why AAFT surrogates and not a mirror control
--------------------------------------------
The tempting null control is to flip the trace and detect "events" in the
impossible direction, justified by "the pore cannot un-occlude". That control is
wrong. Enhancement events are real physics (`polarity == +1`): analytes that
carry counter-ions or increase local conductance genuinely raise the current.
Against such a recording the mirror control reports a ~100% false-positive rate
for events that are entirely real, which pushes the user to raise their threshold
until they have thresholded their own science away -- all while the app displays
a badge certifying rigor. A control that is confidently wrong is worse than no
control.

The Amplitude Adjusted Fourier Transform surrogate assumes nothing about
direction and nothing about Gaussianity. It preserves the measured power spectrum
(so 1/f noise, line pickup and the anti-alias filter roll-off all survive) and
preserves the amplitude distribution exactly (so heavy tails survive), while
randomising Fourier phases -- which destroys temporal localisation. A real
translocation is a localised, phase-coherent structure; it cannot survive phase
randomisation. Noise character can. Whatever the detector finds in a surrogate is,
by construction, a false positive.

Sign convention
---------------
`dev = polarity * (current - baseline)` is positive during an event, for
blockades and enhancement alike (see `nanoscope.analysis.events`). `synth_trace`
therefore sets `polarity = -sign(i0_pA)`: an occlusion reduces |I|, so with the
negative open-pore currents typical of negative applied voltage the trace deviates
*upward* and polarity is +1. Direction lives in the recording's polarity; depths
are always positive magnitudes.

Data model (PORT_PLAN §3.7): the SOURCE consumed an `io.Recording`. This port is
signal-first -- synthetic traces are invented, not read from a Session -- so a
tiny frozen `SynthRecording` shim carries `(data, sample_rate_hz, polarity)`.
Detection goes through the signal-first `analysis.detect.detect_events`, lazily
imported to avoid an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cancel import CancelFlag, cancel_scope, check_cancelled, resolve_cancel

__all__ = [
    "SynthRecording",
    "aaft_surrogate",
    "synth_trace",
    "null_control",
    "sweep_threshold",
    "recovery_map",
]


# ---------------------------------------------------------------------------
# minimal signal-first Recording shim (replaces io.Recording)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynthRecording:
    """A synthetic trace plus the little metadata the truth machinery needs.

    Signal-first stand-in for the SOURCE `io.Recording`: it carries the raw
    samples (`data`, float32), the sample rate, and the polarity that makes
    `dev = polarity * (current - baseline)` positive inside events. `current`
    and `fs` are read-only aliases so ported code and tests can use either name.
    """

    data: np.ndarray
    sample_rate_hz: float
    polarity: int
    name: str = "synthetic"
    units: str = "pA"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.data.size)

    @property
    def duration_s(self) -> float:
        return self.n / self.sample_rate_hz

    @property
    def current(self) -> np.ndarray:
        return self.data

    @property
    def fs(self) -> float:
        return self.sample_rate_hz

    def with_current(self, data, **m) -> "SynthRecording":
        return replace(
            self, data=np.asarray(data, np.float32), meta={**self.meta, **m}
        )


# ---------------------------------------------------------------------------
# lazy imports of sibling modules (avoid an import cycle; detect imports much)
# ---------------------------------------------------------------------------
def _detect() -> Callable[..., Any]:
    """A Recording-consuming detector built on the signal-first `detect_events`.

    Returns a callable `detect(rec, **kwargs) -> EventSet` so the ported truth
    machinery (and its tests) can keep the SOURCE's `detect(rec)` call shape. The
    recording's stored polarity is always passed explicitly; there is no stored
    polarity to cross-check in the signal-first model, so the SOURCE's
    `check_polarity` flag is a no-op and is stripped.
    """
    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.events import EventSet

    def detect(rec: SynthRecording, **kwargs: Any):
        kwargs.pop("check_polarity", None)
        df, prov = detect_events(
            rec.data,
            rec.sample_rate_hz,
            polarity=rec.polarity,
            recording=rec.name,
            **kwargs,
        )
        return EventSet(df, prov)

    return detect


def _baseline_of(
    rec: SynthRecording, config: Any = None
) -> tuple[np.ndarray, str]:
    """Full-length baseline for `rec`, falling back to a constant median.

    Returns (baseline, method_used) so callers can record which was used rather
    than silently reporting a number produced by an unstated method.

    This is the FOURTH call site of the app's characterisation baseline, and the
    one earlier accounts of the defect missed entirely -- it used to call
    ``estimate_baseline`` at its silent 10 ms default, so a truth study measured
    its recall against a baseline the ML characteriser (200 ms) would never have
    produced. It goes through the same door as the other three, with the same
    `BaselineConfig`, so a sweep validates the estimator the app actually ships.
    ``rec.polarity`` is passed explicitly -- a synthetic recording KNOWS which way
    its events run, and a derived polarity on a drifting trace reads the drift.
    """
    try:
        from nanoscope.analysis.baseline import characterisation_baseline
    except ImportError:
        return np.full(rec.n, float(np.median(rec.data)), dtype=np.float64), "median_constant"
    res = characterisation_baseline(
        np.asarray(rec.data), rec.sample_rate_hz,
        config=config, polarity=rec.polarity, health=False,
    )
    return np.asarray(res.baseline, dtype=np.float64), res.config.method


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ranges(starts: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat sample indices for many [start, start+length) ranges, plus event labels.

    Vectorised equivalent of `np.concatenate([np.arange(s, s+l) ...])`.
    """
    if starts.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    total = int(lengths.sum())
    label = np.repeat(np.arange(starts.size, dtype=np.int64), lengths)
    inner = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + inner, label


def _place_events(
    n_total: int, lengths: np.ndarray, gap: int, rng: np.random.Generator
) -> np.ndarray:
    """Random non-overlapping start indices for events of the given lengths.

    Guarantees at least `gap` samples between consecutive events and before the
    end of the trace. Raises if the events cannot fit -- a silently truncated
    event list would corrupt every efficiency computed from it.
    """
    n_ev = int(lengths.size)
    if n_ev == 0:
        return np.empty(0, dtype=np.int64)
    need = int(lengths.sum()) + gap * (n_ev + 1)
    free = n_total - need
    if free <= 0:
        raise ValueError(
            f"cannot place {n_ev} events (total {int(lengths.sum())} samples + gaps) "
            f"in {n_total} samples; shorten the events, reduce n_events, or lengthen the trace"
        )
    # Sorted uniform offsets partition the free space between the fixed-size blocks.
    slack = np.sort(rng.integers(0, free + 1, size=n_ev))
    before = np.cumsum(lengths) - lengths
    return (slack + before + gap * (1 + np.arange(n_ev))).astype(np.int64)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance 1/f (pink) noise via spectral shaping of white noise.

    Real nanopore noise is not white: low-frequency 1/f from surface charge
    fluctuations dominates below ~1 kHz. A white-noise-only simulator flatters
    every baseline estimator and every detector tested against it.
    """
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    scale = np.ones_like(f)
    scale[1:] = 1.0 / np.sqrt(f[1:])
    x = np.fft.irfft(spec * scale, n=n)
    sd = x.std()
    return x / sd if sd > 0 else x


# ---------------------------------------------------------------------------
# AAFT surrogate
# ---------------------------------------------------------------------------
def aaft_surrogate(
    x: np.ndarray, *, rng: Any = None, cancel: CancelFlag | None = None
) -> np.ndarray:
    """Amplitude Adjusted Fourier Transform surrogate of `x`.

    Preserves the amplitude distribution *exactly* (the output is a permutation of
    the input values) and the power spectrum closely, while randomising Fourier
    phases so that localised structures -- i.e. events -- cannot survive.

    Gotcha, and it is not a small one: the final rank-remap perturbs the spectrum, the
    more so the further `x` is from Gaussian. **On Gaussian noise that bias is
    immaterial; on real nanopore noise it is not.** Measured band by band against the
    trace each surrogate was built from -- the owner's own recordings, events already
    excised -- the surrogate's power is up to 2.2x the original's on the apo -600 mV
    file (kurtosis 9.8), 6.4x on ferritin_data2 and 14.2x on ferritin_data3, while the
    synthetic Gaussian fixture the spectrum test uses (kurtosis 3.000) matches to 4 %.
    So `test_aaft_preserves_spectrum` passes while the property is false for every real
    recording in the building.

    IAAFT iterates that bias away and does NOT rescue the null: driving the worst band
    from +0.78 to +0.009 in log made the apo file's surrogate detection count go UP,
    3867 -> 5763, against 56 on the real event-free trace. Phase randomisation of a
    real recording is simply not a null of it. This function is therefore left exactly
    as it is -- it is correct AAFT and it is the right tool on Gaussian noise -- and
    `null_control` checks whether the surrogate it built is a null of the recording
    before it reports a rate. See `_surrogate_is_faithful`.

    Args:
        x: 1-D signal (any units).
        rng: seed, Generator, or None.
        cancel: zero-arg cancel predicate; `None` uses the ambient token. Polled
            between the sorts and the transforms below -- on the owner's 2.99 M-sample
            recording this call alone is 1.68 s and it runs before the first
            detection of every surrogate study, so without these polls a Cancel
            pressed early would sit on it for the better part of two seconds. Each
            remaining step is a single numpy call and cannot be interrupted from
            Python, which puts the honest floor for cancel latency here at ~0.3 s.

    Returns:
        Surrogate array, same shape and dtype as `x`.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if x.size < 8:
        raise ValueError(f"x too short for a surrogate: {x.size} samples")
    gen = np.random.default_rng(rng)
    n = x.size

    # The steps below are exactly the original expressions, split at their natural
    # boundaries so a cancel can land between them. Nothing about the numerics moved.
    check_cancelled(cancel)
    order = np.argsort(x)
    check_cancelled(cancel)
    ranks = np.argsort(order)
    check_cancelled(cancel)
    gauss = np.sort(gen.standard_normal(n))[ranks]

    check_cancelled(cancel)
    spec = np.fft.rfft(gauss)
    phase = gen.uniform(0.0, 2.0 * np.pi, size=spec.size)
    phase[0] = 0.0  # DC must stay real
    if n % 2 == 0:
        phase[-1] = 0.0  # Nyquist bin of a real signal is real
    check_cancelled(cancel)
    y = np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=n)

    check_cancelled(cancel)
    y_order = np.argsort(y)
    check_cancelled(cancel)
    y_ranks = np.argsort(y_order)
    check_cancelled(cancel)
    return np.sort(x)[y_ranks]


# ---------------------------------------------------------------------------
# synthetic trace
# ---------------------------------------------------------------------------
def synth_trace(
    *,
    fs: float = 50_000.0,
    duration_s: float = 10.0,
    i0_pA: float = -1000.0,
    noise_pA: float = 15.0,
    n_events: int = 200,
    dwell_range_s: tuple[float, float] = (5e-5, 2e-3),
    depth_frac_range: tuple[float, float] = (0.05, 0.4),
    drift_pA: float = 0.0,
    seed: int = 0,
    pink_frac: float = 0.3,
    line_pA: float = 0.0,
    line_hz: float = 60.0,
    min_gap_s: float = 1e-3,
) -> tuple[SynthRecording, pd.DataFrame]:
    """Synthesise a nanopore trace with known events. The foundation of every test.

    The trace is `I0 + drift + line + noise`, with square events multiplying the
    *local* baseline by `(1 - frac_depth)` -- an occlusion reduces the pore's
    conductance, so the blockade is fractional in the local open-pore current, not
    a fixed number of pA.

    Noise is `noise_pA` RMS **in total**, mixed from a white and a 1/f component
    (`pink_frac` = pink share of the variance) and then rescaled so the realised
    RMS equals `noise_pA` exactly, which keeps tests deterministic.

    Polarity: `polarity = -sign(i0_pA)`, the sign that makes
    `polarity * (current - baseline)` positive during an event. With the default
    negative `i0_pA` an occlusion drives the trace toward zero, i.e. upward, so
    `rec.polarity == +1`. Depths in the ground truth are positive magnitudes.

    Args:
        fs: sampling rate, Hz.
        duration_s: trace length, s.
        i0_pA: open-pore current, pA (signed; must be non-zero).
        noise_pA: total noise RMS, pA.
        n_events: number of injected events.
        dwell_range_s: (lo, hi) dwell, sampled log-uniformly.
        depth_frac_range: (lo, hi) dI/I0, sampled uniformly.
        drift_pA: amplitude of a slow one-cycle sinusoidal baseline drift, pA.
        seed: RNG seed.
        pink_frac: fraction of noise *variance* that is 1/f (0..1).
        line_pA: amplitude of `line_hz` mains pickup, pA (0 = off).
        line_hz: mains frequency, Hz.
        min_gap_s: minimum event-free gap between events, s.

    Returns:
        (SynthRecording, ground_truth) where ground_truth has one row per event
        with columns t_start_s, dwell_s, depth_pA, frac_depth, start_idx, end_idx,
        baseline_pA. `dwell_s` is the *realised* (sample-quantised) dwell, which is
        the best any detector could recover.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if i0_pA == 0:
        raise ValueError("i0_pA must be non-zero: a pore with no open current has no blockades")
    if noise_pA < 0:
        raise ValueError(f"noise_pA must be >= 0, got {noise_pA}")
    if n_events < 0:
        raise ValueError(f"n_events must be >= 0, got {n_events}")
    if not 0.0 <= pink_frac <= 1.0:
        raise ValueError(f"pink_frac must be in [0, 1], got {pink_frac}")
    lo_d, hi_d = float(dwell_range_s[0]), float(dwell_range_s[1])
    lo_f, hi_f = float(depth_frac_range[0]), float(depth_frac_range[1])
    if not 0 < lo_d <= hi_d:
        raise ValueError(f"dwell_range_s must be 0 < lo <= hi, got {dwell_range_s}")
    if not 0 < lo_f <= hi_f < 1:
        raise ValueError(f"depth_frac_range must satisfy 0 < lo <= hi < 1, got {depth_frac_range}")
    if lo_d * fs < 1:
        raise ValueError(
            f"shortest dwell {lo_d * 1e6:.1f} us is under one sample at {fs / 1000:g} kHz"
        )

    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs

    baseline = np.full(n, float(i0_pA), dtype=np.float64)
    if drift_pA:
        baseline += drift_pA * np.sin(2.0 * np.pi * t / duration_s)
    if line_pA:
        baseline += line_pA * np.sin(2.0 * np.pi * line_hz * t + rng.uniform(0, 2 * np.pi))

    noise = np.zeros(n, dtype=np.float64)
    if noise_pA > 0:
        white = rng.standard_normal(n)
        noise = np.sqrt(1.0 - pink_frac) * white + np.sqrt(pink_frac) * _pink_noise(n, rng)
        noise *= noise_pA / noise.std()

    lengths = np.empty(0, dtype=np.int64)
    starts = np.empty(0, dtype=np.int64)
    fracs = np.empty(0, dtype=np.float64)
    if n_events > 0:
        dwells = np.exp(rng.uniform(np.log(lo_d), np.log(hi_d), size=n_events))
        lengths = np.maximum(1, np.round(dwells * fs).astype(np.int64))
        fracs = rng.uniform(lo_f, hi_f, size=n_events)
        starts = _place_events(n, lengths, max(1, int(round(min_gap_s * fs))), rng)
        order = np.argsort(starts)
        starts, lengths, fracs = starts[order], lengths[order], fracs[order]

    current = baseline + noise
    if n_events > 0:
        idx, label = _ranges(starts, lengths)
        current[idx] = baseline[idx] * (1.0 - fracs[label]) + noise[idx]
        mean_base = np.bincount(label, weights=baseline[idx], minlength=n_events) / lengths

    polarity = int(-np.sign(i0_pA))
    rec = SynthRecording(
        data=current.astype(np.float32),
        sample_rate_hz=float(fs),
        polarity=polarity,
        name="synthetic",
        units="pA",
        meta={
            "synthetic": True,
            "i0_pA": float(i0_pA),
            "noise_pA": float(noise_pA),
            "pink_frac": float(pink_frac),
            "drift_pA": float(drift_pA),
            "line_pA": float(line_pA),
            "n_events": int(n_events),
            "seed": int(seed),
        },
    )

    if n_events == 0:
        truth = pd.DataFrame(
            {
                c: pd.Series(dtype=d)
                for c, d in {
                    "t_start_s": "float64",
                    "dwell_s": "float64",
                    "depth_pA": "float64",
                    "frac_depth": "float64",
                    "start_idx": "int64",
                    "end_idx": "int64",
                    "baseline_pA": "float64",
                }.items()
            }
        )
        return rec, truth

    truth = pd.DataFrame(
        {
            "t_start_s": starts / fs,
            "dwell_s": lengths / fs,
            "depth_pA": np.abs(mean_base) * fracs,
            "frac_depth": fracs,
            "start_idx": starts,
            "end_idx": starts + lengths,
            "baseline_pA": mean_base,
        }
    )
    return rec, truth


# ---------------------------------------------------------------------------
# null control
# ---------------------------------------------------------------------------
def _event_free_trace(
    rec: SynthRecording, real, detect_kwargs: dict, rng: np.random.Generator
) -> np.ndarray:
    """`rec.data` with detected events excised, so the surrogate is pure noise.

    The whole premise of the AAFT null is that phase randomisation destroys events
    while preserving noise. But AAFT preserves the amplitude distribution *exactly*,
    and a trace that contains events contains their deep samples -- typically ~1% of
    all samples sit past the detection threshold. Surrogating the raw trace scatters
    those deep-amplitude samples across the record, where the detector re-fires on
    them: on a trace with plainly-detectable events the surrogate count runs into the
    thousands and `estimated_fdr` pins to 1.0 regardless of how clean the events are.
    That is the module's own "a control that is confidently wrong is worse than no
    control" failure. The null must therefore be built from the noise with the events
    *removed* -- excise each detected event (padded) and refill it with noise resampled
    from the event-free stretches, so the amplitude distribution fed to AAFT is the
    noise distribution, not noise+events.
    """
    x = rec.data.astype(np.float64).copy()
    df = real.df
    if not len(df):
        return x
    baseline, _ = _baseline_of(rec)
    pad = int(round(2e-3 * rec.sample_rate_hz))
    keep = np.ones(x.size, dtype=bool)
    s = np.maximum(0, df["start_idx"].to_numpy(dtype=np.int64) - pad)
    e = np.minimum(rec.n, df["end_idx"].to_numpy(dtype=np.int64) + pad)
    idx, _ = _ranges(s, e - s)
    keep[np.unique(idx)] = False
    quiet = (x - baseline)[keep]
    if quiet.size < 1000:
        raise ValueError(
            f"only {quiet.size} event-free samples remain after excising detections in "
            f"{rec.name!r}; too few to build a noise-only null. The trace is almost all "
            "event -- estimate the false-positive rate on a quieter recording."
        )
    bad = np.flatnonzero(~keep)
    x[bad] = baseline[bad] + _draw_noise(quiet, bad.size, rng)
    return x


def null_control(
    rec: SynthRecording,
    *,
    detect_kwargs: dict | None = None,
    n_surrogates: int = 5,
    rng: Any = None,
    cancel: CancelFlag | None = None,
) -> dict:
    """False-positive rate of the detector, measured on the user's own noise.

    Runs the *same* detector with the *same* settings on `n_surrogates` AAFT
    surrogates of `rec`, built from the recording's noise with its detected events
    removed. Surrogates keep this recording's spectrum and (noise) amplitude
    distribution but have no localised events, so every detection in one is a
    false positive.

    Why the events are removed first: AAFT preserves the amplitude distribution
    exactly, so surrogating the raw trace keeps the deep event samples and merely
    scatters them -- the detector then re-fires on them and `estimated_fdr` pins to
    ~1.0 for any trace with real events. See `_event_free_trace`.

    Args:
        rec: the real recording.
        detect_kwargs: passed verbatim to `detect` for both real and surrogate runs.
        n_surrogates: number of surrogates (>= 1).
        rng: seed, Generator, or None. **A CALLER THAT PRINTS `estimated_fdr` MUST PASS A
            SEED IT CAN NAME.** One Generator drives the noise refill and every surrogate,
            so `None` (OS entropy) makes the whole result unrepeatable — and the surrogates
            disagree with each other enough that this is not a rounding matter: on the
            owner's -700 mV apo recording, five unseeded runs against the same 1164
            real detections printed 17.0 / 24.0 / 20.3 / 13.8 / 19.0 %, with per-run counts
            spanning 42 to 643. Nor does a seed alone make the mean meaningful: report
            `per_surrogate` beside it. `gui.analysis_panel.AnalysisPanel._null_seed`
            derives the app's, and `gui.panels.StatsPanel.set_fdr` prints both.
        cancel: zero-arg cancel predicate; `None` uses the ambient token. See
            `sweep_threshold` for why every one of these takes one.

    THE CONTROL CHECKS ITS OWN PREMISE, and refuses rather than guessing. See
    `_surrogate_is_faithful`: the AAFT null is exact for Gaussian noise and is NOT a
    null of a real nanopore recording, so ``estimated_fdr`` comes back ``nan`` -- with
    the reason in ``notes`` -- whenever the surrogates find materially more than the
    event-free trace they were built from. Before that check this function returned
    ``1.00`` on all three of the owner's own recordings and the panel printed
    "estimated false-positive rate: 100.0%".

    Args:
        rec: the real recording.
        detect_kwargs: passed verbatim to `detect` for both real and surrogate runs.
        n_surrogates: number of surrogates (>= 1).
        rng: seed, Generator, or None.
        cancel: zero-arg cancel predicate; `None` uses the ambient token. See
            `sweep_threshold` for why every one of these takes one.

    Returns:
        dict with n_real, n_event_free, n_surrogate_mean, n_surrogate_std,
        estimated_fdr, per_surrogate, notes.

        * ``n_event_free`` -- detections on the recording's own noise with its events
          excised and refilled from its quiet stretches, NO surrogate involved. Those
          are false positives measured directly, on the real spectrum and the real
          baseline. It is a lower bound (the stretches that were not excised are, by
          construction, ones the detector already declined), and it is the honest
          number to look at when the surrogate is refused.
        * ``estimated_fdr`` -- ``nan`` when ``n_real == 0`` (undefined, not zero) and
          ``nan`` when the surrogate is not a faithful null of this recording.
        * ``notes`` -- tuple of sentences; empty when the control is clean.
    """
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates must be >= 1, got {n_surrogates}")
    with cancel_scope(resolve_cancel(cancel)):
        return _null_control(rec, detect_kwargs, n_surrogates, rng)


#: A faithful surrogate is the same noise as the trace it was built from, so the
#: detector must respond to it about the same way. These two numbers bound "about":
#: the tolerated surrogate count is ``FACTOR * n_event_free + ALLOWANCE``, the
#: allowance carrying the small-count case where Poisson scatter dominates a ratio.
#:
#: They are deliberately loose -- this is a check for a BROKEN premise, not a
#: significance test. Measured with them: the synthetic fixtures pass with 0 against
#: 0, and the three real recordings are refused with 3.6x to 23x of headroom
#: (apo 4040 against a tolerated 178; ferritin_data2 8225 against 649;
#: ferritin_data3 6059 against 1678).
_NULL_SURROGATE_FACTOR = 3.0
_NULL_SURROGATE_ALLOWANCE = 10.0


def _surrogate_is_faithful(n_event_free: int, n_surrogate_mean: float) -> bool:
    """Is the AAFT surrogate actually a null of THIS recording?

    THE PREMISE, AND WHY IT NEEDS CHECKING. The null assumes phase randomisation
    destroys events while leaving the noise alone, so that every detection in a
    surrogate is a false positive. `_event_free_trace` already removes the events, so
    the surrogate and the trace it was made from are supposed to be the same noise --
    and the same detector run over both must therefore find about the same number of
    things. Measured, ``k_in=5``, ``min_dwell_s=1e-4``, the app's own default
    estimator, three surrogates each:

    ======================  ============  ==============  ==========
    recording               event-free    AAFT surrogate  reported
    ======================  ============  ==============  ==========
    synthetic, Gaussian              0                 0  fdr 0.00
    apo -600 mV                     56              4040  fdr 1.00
    ferritin_data2                 213              8225  fdr 1.00
    ferritin_data3                 556              6059  fdr 1.00
    ======================  ============  ==============  ==========

    So on every real recording the owner has, the AAFT null was manufacturing 30x to
    70x more detections than the noise it was built from, and ``estimated_fdr``
    saturated at 1.00 -- a confidently wrong control, on a recording whose events are
    plainly real. The module's own docstring says a control that is confidently wrong
    is worse than no control; this is that failure, live.

    WHAT IS ACTUALLY WRONG, measured rather than assumed, and it is NOT the estimator
    change that raised the surrogate floor in `test_aaft_destroys_events`:

    * `aaft_surrogate`'s docstring calls its rank-remap's spectral bias "immaterial for
      false-positive counting". It is immaterial on Gaussian noise and severe on real
      nanopore noise (kurtosis 9.8 on the apo file's event-free trace against 3.000 for
      the synthetic one): band by band, the surrogate's power exceeds the original's by
      up to 2.2x on the apo file, 6.4x on data2 and 14.2x on data3, while the synthetic
      fixture matches to 4 %.
    * But fixing the spectrum does NOT fix the null. An IAAFT implementation that
      drives the apo file's worst band from +0.78 to +0.009 in log makes it WORSE:
      3867 -> 5763 detections, against 56 on the trace itself.
    * The bigger term is the OPEN-PORE DRIFT. Surrogating the residual and re-adding
      the estimated baseline takes the apo file from 4040 to 225; a synthetic control
      isolates it exactly -- band-limited Gaussian noise with clean events surrogates
      to 0 detections with no drift and to 817 with a 200 pA slow drift added. Phase
      randomisation is a null for a stationary spectrum plus Gaussian noise. A real
      recording is not that, and no amount of iterating makes it one.

    A surrogate that is a genuine null of a real nanopore recording is a research
    question, not a patch.

    So this function does not try to repair the surrogate. It establishes whether the
    surrogate IS one, from a comparison the control can afford (one extra detection
    run out of ``n_surrogates + 2``), and lets the caller refuse to publish a number
    that has no meaning.
    """
    tolerated = _NULL_SURROGATE_FACTOR * float(n_event_free) + _NULL_SURROGATE_ALLOWANCE
    return float(n_surrogate_mean) <= tolerated


def _null_control(rec, detect_kwargs, n_surrogates, rng) -> dict:
    detect = _detect()
    kw = dict(detect_kwargs or {})
    # A caller-supplied baseline was estimated on the real trace; it does not
    # describe a surrogate, whose samples are a permutation in time.
    kw.pop("baseline", None)
    gen = np.random.default_rng(rng)

    real = detect(rec, **dict(detect_kwargs or {}))
    n_real = len(real.df)
    # Surrogate the noise with the events removed, not the raw trace (see above).
    noise = _event_free_trace(rec, real, kw, gen)
    # A surrogate has no localised events, so its measured polarity is noise-driven
    # and may disagree with rec.polarity; we know it has no events, so the polarity
    # guard (meant for the user's real trace) must not fire here.
    kw["check_polarity"] = False

    # The control's own control: the same detector on the event-free trace itself,
    # before any surrogate exists. One extra detection buys both the direct
    # false-positive count and the evidence for whether the surrogate is a null at all.
    check_cancelled()
    n_event_free = len(detect(rec.with_current(noise, event_free=True), **kw).df)

    per = []
    for i in range(n_surrogates):
        check_cancelled()   # one surrogate = one full detection; stop between them
        per.append(len(detect(rec.with_current(aaft_surrogate(noise, rng=gen), surrogate=i), **kw).df))
    arr = np.asarray(per, dtype=float)
    mean = float(arr.mean())

    notes: list[str] = []
    faithful = _surrogate_is_faithful(n_event_free, mean)
    if not faithful:
        notes.append(
            f"the AAFT surrogates produced {mean:.0f} detections where this "
            f"recording's own event-free noise produces {n_event_free}, so the "
            "surrogates are not the same noise and a false-positive RATE cannot be "
            "read off them. Phase randomisation is a null only for a trace that IS a "
            "stationary spectrum plus Gaussian noise; a recording with slow open-pore "
            "drift or heavy tails is not, and the surrogate's power comes back "
            "inflated (measured on this machine's recordings: every band, by up to "
            "14x). Use n_event_free instead -- it is the false-positive count on this "
            "recording's own noise, measured with no surrogate at all, and it is a "
            "lower bound."
        )
    if n_real == 0:
        notes.append("no real detections, so a false-positive RATE is undefined "
                     "(not zero).")

    if not n_real or not faithful:
        fdr = float("nan")
    else:
        fdr = float(min(1.0, mean / n_real))
    return {
        "n_real": int(n_real),
        "n_event_free": int(n_event_free),
        "n_surrogate_mean": mean,
        "n_surrogate_std": float(arr.std(ddof=1)) if n_surrogates > 1 else float("nan"),
        "estimated_fdr": fdr,
        "per_surrogate": [int(v) for v in per],
        "notes": tuple(notes),
    }


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------
def sweep_threshold(
    rec: SynthRecording,
    *,
    k_values: np.ndarray | None = None,
    detect_kwargs: dict | None = None,
    plateau_tol: float = 0.15,
    rng: Any = None,
    cancel: CancelFlag | None = None,
) -> pd.DataFrame:
    """Event count vs detection threshold, with the plateau identified.

    This replaces a threshold slider. A slider makes the event count a matter of
    taste and hides that choice from the reader; a sweep makes it evidence. When a
    real detection regime exists there is a range of `k_in` over which the count
    barely moves -- events are far above threshold, noise far below. Outside the
    plateau the count is a function of the threshold, not of the sample.

    The plateau is the longest run of adjacent k where |d ln(n) / dk| <= plateau_tol.

    Args:
        rec: recording to sweep.
        k_values: thresholds in units of noise RMS. Default: 2.5 to 10.0 step 0.5.
        detect_kwargs: passed to `detect`; `k_in` is overridden by the sweep.
        plateau_tol: max |d ln n / dk| counted as flat, per unit k.
        rng: seed for the surrogate used for the `n_surrogate` column.
        cancel: zero-arg predicate answering "has the user asked to stop?".
            `None` uses the AMBIENT token (`analysis.cancel.cancel_scope`), which is
            what the GUI worker installs. THIS IS THE SLOWEST OPERATION IN THE APP:
            measured on the owner's smallest apo recording (2.99 M samples, -600 mV)
            it takes 18.9 s, because the default grid is 16 thresholds x 2 detections.
            Whatever token ends up in force is also installed for the nested
            `detect_events` calls, so a Cancel lands inside the current detection
            rather than at the next k. Raises `analysis.cancel.OperationCancelled`.

    Returns:
        DataFrame with k, n_events, median_dwell_s, median_frac_depth, n_surrogate,
        dlogn_dk, in_plateau. `df.attrs['k_recommended']` is the plateau centre, or
        nan if no plateau exists (which is itself the finding: there is no
        detection regime here).

        **`df.attrs['surrogate_is_a_null']` before you read `n_surrogate`.** On a real
        recording the AAFT surrogate manufactures detections rather than destroying
        them -- measured on the owner's -600 mV apo file, ~21 000 at every k against
        240-652 real ones, and flat in k. When the flag is False the accompanying
        `df.attrs['surrogate_note']` says so and `n_surrogate` means nothing; the
        plateau, which is read off `n_events` alone, is unaffected.
    """
    with cancel_scope(resolve_cancel(cancel)):
        return _sweep_threshold(rec, k_values, detect_kwargs, plateau_tol, rng)


def _sweep_threshold(rec, k_values, detect_kwargs, plateau_tol, rng) -> pd.DataFrame:
    detect = _detect()
    if k_values is None:
        k_values = np.arange(2.5, 10.01, 0.5)
    k_values = np.asarray(k_values, dtype=float)
    if k_values.ndim != 1 or k_values.size < 2:
        raise ValueError("k_values must be a 1-D array of at least 2 thresholds")
    if np.any(np.diff(k_values) <= 0):
        raise ValueError("k_values must be strictly increasing")
    kw = dict(detect_kwargs or {})
    kw.pop("k_in", None)

    surr_kw = {key: v for key, v in kw.items() if key != "baseline"}
    surr_kw["check_polarity"] = False  # a surrogate has no events; guard would misfire
    check_cancelled()   # the surrogate FFT below is one uninterruptible numpy call
    surr = rec.with_current(aaft_surrogate(rec.data, rng=np.random.default_rng(rng)), surrogate=0)

    rows = []
    for k in k_values:
        check_cancelled()
        es = detect(rec, k_in=float(k), **kw)
        d = es.df
        rows.append(
            {
                "k": float(k),
                "n_events": len(d),
                "median_dwell_s": float(d["dwell_s"].median()) if len(d) else float("nan"),
                "median_frac_depth": float(d["frac_depth"].median()) if len(d) else float("nan"),
                "n_surrogate": len(detect(surr, k_in=float(k), **surr_kw).df),
            }
        )
    df = pd.DataFrame(rows)

    n = df["n_events"].to_numpy(dtype=float)
    slope = np.full(len(df), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logn = np.where(n > 0, np.log(np.where(n > 0, n, 1.0)), np.nan)
    slope[:-1] = np.diff(logn) / np.diff(k_values)
    df["dlogn_dk"] = slope

    flat = np.isfinite(slope[:-1]) & (np.abs(slope[:-1]) <= plateau_tol)
    lo, hi = _longest_run(flat)
    in_plateau = np.zeros(len(df), dtype=bool)
    if lo is None:
        df["in_plateau"] = in_plateau
        df.attrs["k_recommended"] = float("nan")
    else:
        in_plateau[lo : hi + 2] = True  # a flat *pair* i covers k[i] and k[i+1]
        df["in_plateau"] = in_plateau
        df.attrs["k_recommended"] = float(0.5 * (k_values[lo] + k_values[hi + 1]))
    df.attrs["plateau_tol"] = float(plateau_tol)
    df.attrs["recording"] = rec.name

    # IS THE ``n_surrogate`` COLUMN A NULL OF THIS RECORDING? Same question
    # `null_control` asks, promoted here from a test assertion to a runtime one: this
    # module's own acceptance test for a working sweep is that the surrogate finds
    # almost nothing where the real trace plateaus. When instead it OUTNUMBERS the real
    # detections, phase randomisation has not destroyed events -- it has manufactured
    # them, and the column is an artefact of the surrogate rather than a property of
    # the detector. Measured on the owner's -600 mV apo recording at ``min_dwell_s
    # 1e-4``: 21256 / 21202 / 21111 / 20973 / 20932 surrogate detections at k = 4..8,
    # against 652 / 322 / 253 / 241 / 240 real ones -- 88x, and FLAT in k, which is the
    # giveaway that it is not responding to the threshold at all.
    #
    # Unlike `null_control` this cannot be repaired here: the sweep surrogates the RAW
    # trace, so it has no event-free trace to compare against and building one would
    # change what the column means. It is reported, and `null_control` is where the
    # question belongs.
    ns = df["n_surrogate"].to_numpy(dtype=float)
    ref = ns[in_plateau] if in_plateau.any() else ns
    real_ref = n[in_plateau] if in_plateau.any() else n
    is_null = bool(np.nanmedian(ref) <= np.nanmedian(real_ref))
    df.attrs["surrogate_is_a_null"] = is_null
    df.attrs["surrogate_note"] = "" if is_null else (
        f"the AAFT surrogate produced a median of {np.nanmedian(ref):.0f} detections "
        f"against {np.nanmedian(real_ref):.0f} real ones, so 'n_surrogate' is not a "
        "false-positive count for this recording -- phase randomisation is a null only "
        "for a stationary spectrum plus Gaussian noise, and a drifting pore is not "
        "that. Read the plateau from 'n_events'; take the false-positive question to "
        "null_control, which measures it on this recording's own event-free noise."
    )
    return df


def _longest_run(mask: np.ndarray) -> tuple[int | None, int | None]:
    """Inclusive [start, stop] of the longest True run in `mask`, or (None, None)."""
    best = (None, None)
    best_len = 0
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < mask.size and mask[j + 1]:
            j += 1
        if j - i + 1 > best_len:
            best_len, best = j - i + 1, (i, j)
        i = j + 1
    return best


# ---------------------------------------------------------------------------
# recovery map / detection efficiency
# ---------------------------------------------------------------------------
def _quiet_noise_pool(
    rec: SynthRecording, detect_kwargs: dict, pad_s: float
) -> tuple[np.ndarray, str, float]:
    """Baseline-subtracted noise from the event-free stretches of a real recording.

    Detected events (padded) are excised and the remaining segments concatenated
    *after* subtracting the local baseline, so the splices are steps of order the
    noise amplitude rather than of order I0. The pool is what the pore's noise
    actually looks like -- 1/f, line pickup, filter roll-off and all -- which is
    the whole point of injecting into it instead of into white noise.
    """
    detect = _detect()
    baseline, method = _baseline_of(rec)
    es = detect(rec, **detect_kwargs)
    keep = np.ones(rec.n, dtype=bool)
    if len(es.df):
        pad = int(round(pad_s * rec.sample_rate_hz))
        s = np.maximum(0, es.df["start_idx"].to_numpy(dtype=np.int64) - pad)
        e = np.minimum(rec.n, es.df["end_idx"].to_numpy(dtype=np.int64) + pad)
        idx, _ = _ranges(s, e - s)
        keep[idx] = False
    pool = rec.data.astype(np.float64)[keep] - baseline[keep]
    frac_quiet = float(keep.mean())
    if pool.size < 1000:
        raise ValueError(
            f"only {pool.size} event-free samples in {rec.name!r} ({frac_quiet:.1%} of the "
            "trace); cannot measure detection efficiency against this recording's noise"
        )
    return pool, method, frac_quiet


def _draw_noise(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` samples drawn as random contiguous chunks of `pool` (preserves 1/f locally)."""
    if pool.size >= n:
        i = int(rng.integers(0, pool.size - n + 1))
        return pool[i : i + n].copy()
    chunk = max(1, pool.size // 4)
    parts = []
    got = 0
    while got < n:
        take = min(chunk, n - got)
        i = int(rng.integers(0, pool.size - take + 1))
        parts.append(pool[i : i + take])
        got += take
    return np.concatenate(parts)


def _make_efficiency_fn(
    dwell_grid: np.ndarray, depth_grid: np.ndarray, eff: np.ndarray
) -> Callable[..., np.ndarray]:
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (np.log(dwell_grid), depth_grid),
        eff,
        method="linear",
        bounds_error=False,
        fill_value=None,  # extrapolate; clipped to [0, 1] below
    )

    def efficiency_fn(dwell_s, frac_depth=None) -> np.ndarray:
        """Interpolated detection efficiency in [0, 1].

        Args:
            dwell_s: scalar or array of dwell times, s.
            frac_depth: scalar or array of dI/I0. If None, the efficiency is
                averaged over the measured depth grid -- correct only if the true
                depth distribution matches that grid, so pass the observed depths
                when you have them.
        """
        d = np.atleast_1d(np.asarray(dwell_s, dtype=float))
        if frac_depth is None:
            pts = np.stack(
                [
                    np.repeat(np.log(d), depth_grid.size),
                    np.tile(depth_grid, d.size),
                ],
                axis=-1,
            )
            out = interp(pts).reshape(d.size, depth_grid.size).mean(axis=1)
        else:
            f = np.broadcast_to(np.atleast_1d(np.asarray(frac_depth, dtype=float)), d.shape)
            out = interp(np.stack([np.log(d), f], axis=-1))
        return np.clip(out, 0.0, 1.0)

    return efficiency_fn


def recovery_map(
    rec: SynthRecording,
    *,
    detect_kwargs: dict | None = None,
    dwell_grid: np.ndarray | None = None,
    depth_grid: np.ndarray | None = None,
    n_per_cell: int = 20,
    seed: int = 0,
    cancel: CancelFlag | None = None,
) -> pd.DataFrame:
    """Measured detection efficiency over a (dwell, depth) grid, using real noise.

    Synthetic events of known dwell and depth are injected into `rec`'s own
    event-free noise and the *same* detector is run over the result. The recovered
    fraction per cell is the detector's efficiency for that event class, measured
    rather than assumed.

    Why this must not be decoration: everyone computes this surface, plots a pretty
    contour of it, and then fits tau assuming efficiency == 1 above the filter
    cutoff -- which the same surface plainly shows is false, since efficiency rolls
    off smoothly over a decade of dwell. Missing short events biases tau long, and
    the bias is not a small correction. The output of this function is meant to be
    handed to `stats.fit_dwell_exponential(efficiency=...)` so the likelihood is
    weighted by what the detector can actually see.

    Args:
        rec: real recording, used for its noise and its open-pore current.
        detect_kwargs: passed to `detect`, both to find the quiet stretches and to
            score the injected events. Use the settings you will analyse with.
        dwell_grid: dwell times to test, s. Default 6 points, 2e-5..5e-3, log spaced.
        depth_grid: dI/I0 values to test. Default [0.02, 0.05, 0.1, 0.2, 0.4].
        n_per_cell: events injected per cell; efficiency has a binomial standard
            error of ~sqrt(e(1-e)/n_per_cell), so 20 buys about +/-0.11 at worst.
        seed: RNG seed.
        cancel: zero-arg cancel predicate; `None` uses the ambient token. The grid is
            `len(dwell_grid) * len(depth_grid)` full detections (30 by default), so
            this is the second-slowest operation in the app; the token is polled per
            cell and, being ambient, inside each detection too.

    Returns:
        DataFrame, one row per cell, with columns dwell_s, frac_depth, n_injected,
        n_recovered, efficiency, depth_bias (median recovered frac_depth / true),
        dwell_bias, and eff_se. Attributes for direct consumption by `stats`:
        `attrs['efficiency_fn']` -> callable(dwell_s, frac_depth=None) -> array in
        [0, 1]; plus `attrs['dwell_grid']`, `attrs['depth_grid']`,
        `attrs['efficiency_grid']` (shape (n_dwell, n_depth)).
    """
    with cancel_scope(resolve_cancel(cancel)):
        return _recovery_map(rec, detect_kwargs, dwell_grid, depth_grid, n_per_cell, seed)


def _recovery_map(rec, detect_kwargs, dwell_grid, depth_grid, n_per_cell, seed) -> pd.DataFrame:
    kw = dict(detect_kwargs or {})
    kw.pop("baseline", None)  # baselines belong to a trace, not to a synthetic splice
    # Shallow/short cells inject events near the detection floor, so the probe's
    # measured polarity is noise-driven; we set its polarity from i0, so silence
    # the guard for probe runs (it exists to catch a mislabelled *user* trace).
    probe_kw = {**kw, "check_polarity": False}
    if dwell_grid is None:
        dwell_grid = np.logspace(np.log10(2e-5), np.log10(5e-3), 6)
    if depth_grid is None:
        depth_grid = np.array([0.02, 0.05, 0.10, 0.20, 0.40])
    dwell_grid = np.asarray(dwell_grid, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float)
    if dwell_grid.size < 2 or depth_grid.size < 2:
        raise ValueError("dwell_grid and depth_grid need >= 2 points each to define a surface")
    if np.any(np.diff(dwell_grid) <= 0) or np.any(np.diff(depth_grid) <= 0):
        raise ValueError("dwell_grid and depth_grid must be strictly increasing")
    if dwell_grid[0] * rec.sample_rate_hz < 1:
        raise ValueError(
            f"shortest dwell {dwell_grid[0] * 1e6:.1f} us is under one sample at "
            f"{rec.sample_rate_hz / 1000:g} kHz"
        )
    if n_per_cell < 1:
        raise ValueError(f"n_per_cell must be >= 1, got {n_per_cell}")

    detect = _detect()
    rng = np.random.default_rng(seed)
    pool, base_method, frac_quiet = _quiet_noise_pool(rec, kw, pad_s=2e-3)
    i0 = float(np.median(rec.data)) if base_method == "median_constant" else _open_pore(rec)

    rows = []
    eff_grid = np.zeros((dwell_grid.size, depth_grid.size))
    for i, dwell in enumerate(dwell_grid):
        length = max(1, int(round(dwell * rec.sample_rate_hz)))
        gap = max(int(round(2e-3 * rec.sample_rate_hz)), 10 * length)
        n_samp = (n_per_cell + 1) * (length + gap) + gap
        for j, frac in enumerate(depth_grid):
            check_cancelled()   # one cell = one full detection over an injected trace
            lengths = np.full(n_per_cell, length, dtype=np.int64)
            starts = _place_events(n_samp, lengths, gap, rng)
            cur = i0 + _draw_noise(pool, n_samp, rng)
            idx, _ = _ranges(starts, lengths)
            cur[idx] -= i0 * frac  # occlusion: |I| falls by frac * |I0|

            probe = SynthRecording(
                data=cur.astype(np.float32),
                sample_rate_hz=rec.sample_rate_hz,
                polarity=int(-np.sign(i0)),
                name=f"{rec.name}:recovery[{dwell:.2e}s,{frac:.3f}]",
                units=rec.units,
                meta={"injected": True, "true_dwell_s": length / rec.sample_rate_hz, "true_frac_depth": frac},
            )
            found = detect(probe, **probe_kw).df
            n_rec, depth_bias, dwell_bias = _match(found, starts, starts + length, rec.sample_rate_hz, frac)

            eff = n_rec / n_per_cell
            eff_grid[i, j] = eff
            rows.append(
                {
                    "dwell_s": length / rec.sample_rate_hz,
                    "frac_depth": float(frac),
                    "n_injected": int(n_per_cell),
                    "n_recovered": int(n_rec),
                    "efficiency": float(eff),
                    "eff_se": float(np.sqrt(max(eff * (1 - eff), 0.0) / n_per_cell)),
                    "depth_bias": depth_bias,
                    "dwell_bias": dwell_bias,
                }
            )

    df = pd.DataFrame(rows)
    realised_dwell = np.array(
        [max(1, int(round(d * rec.sample_rate_hz))) / rec.sample_rate_hz for d in dwell_grid], dtype=float
    )
    df.attrs["dwell_grid"] = realised_dwell
    df.attrs["depth_grid"] = depth_grid
    df.attrs["efficiency_grid"] = eff_grid
    df.attrs["efficiency_fn"] = _make_efficiency_fn(realised_dwell, depth_grid, eff_grid)
    df.attrs["n_per_cell"] = int(n_per_cell)
    df.attrs["recording"] = rec.name
    df.attrs["frac_quiet"] = frac_quiet
    df.attrs["baseline_method"] = base_method
    df.attrs["detect_kwargs"] = dict(kw)
    return df


def _open_pore(rec: SynthRecording) -> float:
    try:
        from nanoscope.analysis.baseline import open_pore_current
    except ImportError:
        return float(np.median(rec.data))
    return float(open_pore_current(rec.data, rec.sample_rate_hz))


def _match(
    found: pd.DataFrame, starts: np.ndarray, ends: np.ndarray, fs: float, true_frac: float
) -> tuple[int, float, float]:
    """Count injected events overlapped by >= 1 detection; report depth/dwell bias.

    One detection may span at most one injected event (they are placed far apart),
    so overlap matching cannot double-count. Bias is median(measured)/true over the
    matched detections and is the empirical measure of filter attenuation.
    """
    if not len(found):
        return 0, float("nan"), float("nan")
    fs_start = found["start_idx"].to_numpy(dtype=np.int64)
    fs_end = found["end_idx"].to_numpy(dtype=np.int64)
    # overlap[i, j] : detection j overlaps injected event i
    overlap = (fs_start[None, :] < ends[:, None]) & (fs_end[None, :] > starts[:, None])
    hit = overlap.any(axis=1)
    n_rec = int(hit.sum())
    if n_rec == 0:
        return 0, float("nan"), float("nan")
    first = np.argmax(overlap[hit], axis=1)
    md = np.median(found["frac_depth"].to_numpy()[first])
    mt = np.median(found["dwell_s"].to_numpy()[first])
    true_dwell = float(np.median(ends - starts)) / fs
    return n_rec, float(md / true_frac), float(mt / true_dwell)
