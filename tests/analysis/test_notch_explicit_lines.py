"""The notch's EXPLICIT frequency list, and the topology question it raises.

The app parameterised its notch as a fundamental plus a COUNT of contiguous
harmonics, ``[freq_hz * h for h in 1..harmonics]``. The owner's CB7/adamantyl
recordings need ``{60, 120, 180, 300, 420, 540}`` Hz -- a set with GAPS, because his
own Welch census found peaks at those six and nothing at 240, 360 or 480 -- and the
contiguous form cannot express it: the nearest series (``harmonics=9``) notches three
clean bands for nothing. These tests pin the explicit list, and pin the two claims
made about it in :func:`~nanoscope.analysis.filters._notch_sos`'s docstring:

  * the app's ONE-STACKED-SOS topology and the notebooks' ONE-FILTFILT-PER-LINE
    topology are numerically the same filter (measured here, not assumed), and
  * the thing that really differs by tens of pA is the PAD, which is a separate axis
    and is where the app deliberately departs from the notebooks.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sp_signal

from nanoscope.analysis.filters import (
    notch,
    notch_frequencies,
    notch_settle_s,
    notch_unsettled_s,
    response,
    response_db,
)

FS = 50_000.0

#: The host-guest line set, from ``Project-2_Host-Guest`` Block 0. Six lines, three
#: gaps (240 / 360 / 480 are deliberately absent).
HG_LINES = [60.0, 120.0, 180.0, 300.0, 420.0, 540.0]

#: The Zn/Cu line set: nine lines, and it too is not contiguous (480 is missing).
ZN_LINES = [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0]


def _hum_trace(n: int, lines, amps, *, noise_pA: float = 5.0, seed: int = 0):
    """Mains hum on white noise, in pA. float64, as the kernels take it."""
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    x = noise_pA * rng.standard_normal(n)
    for f, a in zip(lines, amps):
        x += a * np.sin(2 * np.pi * f * t + 0.3 * f)
    return x


def _line_amplitude(x: np.ndarray, f_hz: float, *, trim_s: float = 2.0) -> float:
    """Amplitude of one sinusoid in a trace, by projection onto that line.

    ``trim_s`` seconds are dropped from each end before projecting. The notch is
    zero-phase and its zero sits exactly ON the line, so in the interior the residual
    is 0.0014 pA of a 30 pA line; what is left near the ends is the odd extension the
    pad is built from (0.72 pA over the whole 10 s trace), which is a statement about
    the pad and not about the filter's selectivity.
    """
    i = int(trim_s * FS)
    y = np.asarray(x, dtype=np.float64)[i: x.size - i]
    t = np.arange(i, x.size - i) / FS
    c = 2.0 * np.mean(y * np.cos(2 * np.pi * f_hz * t))
    s = 2.0 * np.mean(y * np.sin(2 * np.pi * f_hz * t))
    return float(np.hypot(c, s))


# ---------------------------------------------------------------------------
# the gap the old API could not express
# ---------------------------------------------------------------------------


def test_the_explicit_list_is_applied_exactly_as_given():
    assert notch_frequencies(FS, freqs=HG_LINES) == HG_LINES
    assert notch_frequencies(FS, freqs=ZN_LINES) == ZN_LINES


def test_the_contiguous_series_cannot_reach_540_without_notching_the_gaps():
    """The reason the parameter exists. To cover 540 Hz the harmonic form must run to
    ``harmonics=9``, which drags in 240, 360 and 480 -- three bands the owner's own
    Welch census found were NOT prominent."""
    series = notch_frequencies(FS, 60.0, harmonics=9)
    assert 540.0 in series
    assert {240.0, 360.0, 480.0} <= set(series)
    assert {240.0, 360.0, 480.0}.isdisjoint(notch_frequencies(FS, freqs=HG_LINES))


def test_the_gap_bands_keep_their_signal_and_the_named_lines_do_not():
    """Not a set-membership claim -- a measurement on the trace. A 30 pA line at 240 Hz
    is left alone by the explicit six and destroyed by the contiguous nine."""
    lines = [*HG_LINES, 240.0]
    x = _hum_trace(int(10 * FS), lines, [30.0] * len(lines))

    explicit = np.asarray(notch(x, FS, freqs=HG_LINES), dtype=np.float64)
    contiguous = np.asarray(notch(x, FS, 60.0, harmonics=9), dtype=np.float64)

    # Every named line is gone under both (measured: <= 0.002 pA of 30 pA).
    for f in HG_LINES:
        assert _line_amplitude(explicit, f) < 0.05
        assert _line_amplitude(contiguous, f) < 0.05
    # The gap band survives the explicit set and does not survive the other. 29.7 of
    # 30.0 pA, not 30.0: 240 Hz sits in the skirts of the 180 and 300 Hz notches, so
    # even the set that does not name it costs it ~1%. That is the price of the two
    # nearest lines, not of a line at 240.
    assert _line_amplitude(explicit, 240.0) > 29.0
    assert _line_amplitude(contiguous, 240.0) < 0.05


def test_the_response_plot_shows_the_same_comb_the_kernel_applies():
    """`response` takes the same list, so the picture in the panel is the filter."""
    f, mag = response("notch", FS, freqs=HG_LINES, n=4096)
    db = response_db(mag)
    for line in HG_LINES:
        i = int(np.argmin(np.abs(f - line)))
        assert db[i] < -20.0, f"{line} Hz should be notched, got {db[i]:.1f} dB"
    for gap in (240.0, 360.0, 480.0):
        i = int(np.argmin(np.abs(f - gap)))
        assert db[i] > -1.0, f"{gap} Hz should be untouched, got {db[i]:.1f} dB"


# ---------------------------------------------------------------------------
# the old API is untouched
# ---------------------------------------------------------------------------


def test_the_harmonic_series_api_is_bit_identical_to_spelling_it_out():
    """`freqs` is a new way to say the same thing, not a new filter. The contiguous
    series and the identical explicit list must produce the SAME BYTES, or every
    stored result measured through the old signature has quietly moved."""
    x = _hum_trace(int(4 * FS), [60.0, 120.0, 180.0], [30.0, 15.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old = notch(x, FS, 60.0, harmonics=3)
        new = notch(x, FS, freqs=[60.0, 120.0, 180.0])
    assert np.array_equal(old, new)


def test_freqs_overrides_the_fundamental_and_the_count_outright():
    """They are not merged. A stale ``freq_hz`` left in a params dict must not be able
    to add a line to an explicit list."""
    assert notch_frequencies(FS, 777.0, harmonics=9, freqs=HG_LINES) == HG_LINES
    x = _hum_trace(int(4 * FS), HG_LINES, [10.0] * 6)
    a = notch(x, FS, 777.0, harmonics=9, freqs=HG_LINES)
    b = notch(x, FS, 60.0, harmonics=3, freqs=HG_LINES)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# THE TOPOLOGY MEASUREMENT -- the claim in `_notch_sos`'s docstring
# ---------------------------------------------------------------------------


def _sequential(x: np.ndarray, lines, q: float, padlen: int | None) -> np.ndarray:
    """The notebooks' ``apply_notch_filters``: one filtfilt per line, ascending."""
    out = np.asarray(x, dtype=np.float64).copy()
    for f0 in lines:
        b, a = sp_signal.iirnotch(f0, q, fs=FS)
        out = (sp_signal.filtfilt(b, a, out) if padlen is None
               else sp_signal.filtfilt(b, a, out, padlen=padlen))
    return out


def test_stacking_the_biquads_is_the_same_filter_as_six_sequential_passes():
    """MEASURED, not assumed -- and the reason the sequential topology is NOT offered
    as an option. On a 10 s / 50 kHz trace carrying the owner's own six lines at
    3-30 pA over 5 pA noise, the app's single stacked ``sosfiltfilt`` and six
    sequential ``filtfilt`` passes GIVEN THE SAME PAD differ by 8.8e-5 pA peak
    (5.7e-6 pA RMS) -- about ten float32 quanta at the 100 pA scale, five orders of
    magnitude under the noise floor. A second code path that provably changes nothing
    is a liability, not a feature."""
    n = int(10 * FS)
    x = _hum_trace(n, HG_LINES, [30.0, 15.0, 10.0, 6.0, 4.0, 3.0])

    pad = min(int(np.ceil(notch_settle_s(FS, q=30.0, freqs=HG_LINES) * FS)), n - 1)
    stacked = np.asarray(notch(x, FS, freqs=HG_LINES), dtype=np.float64)
    seq = _sequential(x, HG_LINES, 30.0, pad)

    peak = float(np.abs(stacked - seq).max())
    # Bounded an order of magnitude above the 8.8e-5 pA measured here (and above the
    # 1.5e-3 pA worst case seen at 1 s / Q=10), so a real change in the float library
    # or in the cascade order is visible without the test being flaky about the last
    # bit. Still five orders under the 5 pA noise this trace carries.
    assert peak < 1e-2, f"topologies diverged by {peak:g} pA"


def test_what_actually_differs_by_tens_of_pA_is_the_pad_not_the_topology():
    """The same six sequential passes at scipy's DEFAULT filtfilt pad (9 samples for a
    biquad) depart from the settled result by ~20 pA at the head and ~40 pA at the
    tail -- the notch's own ring-down, six times over. That is why this module sizes
    its pad from `notch_settle_s` instead of reproducing the notebooks' padding, and
    it is the real content of the 'the cascade is not equivalent' folklore."""
    n = int(10 * FS)
    x = _hum_trace(n, HG_LINES, [30.0, 15.0, 10.0, 6.0, 4.0, 3.0])

    stacked = np.asarray(notch(x, FS, freqs=HG_LINES), dtype=np.float64)
    default_pad = _sequential(x, HG_LINES, 30.0, None)
    d = np.abs(stacked - default_pad)

    head = float(d[: int(0.1 * FS)].max())
    tail = float(d[-int(0.1 * FS):].max())
    interior = float(d[int(3 * FS): int(5 * FS)].max())
    assert head > 10.0, head
    assert tail > 10.0, tail
    # Deep in the interior the two agree to far under a float32 quantum at 100 pA --
    # the ring-down has decayed, and nothing but the pad ever separated them.
    assert interior < 1e-3, interior


# ---------------------------------------------------------------------------
# the pad, the settling report, and the Nyquist rule under an explicit list
# ---------------------------------------------------------------------------


def test_the_settling_time_is_read_off_the_lowest_line_of_the_explicit_set():
    """The pad must be sized from the filter that RAN. The slowest pole belongs to the
    lowest line, so a set starting at 60 Hz settles like the 60 Hz series does."""
    both = notch_settle_s(FS, q=30.0, freqs=HG_LINES)
    series = notch_settle_s(FS, 60.0, q=30.0, harmonics=9)
    assert both == pytest.approx(series, rel=1e-12)
    # Drop the 60 Hz line and the cascade forgets faster -- 120 Hz is twice as wide.
    faster = notch_settle_s(FS, q=30.0, freqs=HG_LINES[1:])
    assert faster < both / 1.5


def test_a_short_trace_reports_its_unsettled_head_for_an_explicit_set_too():
    need = notch_settle_s(FS, q=30.0, freqs=HG_LINES)
    n = int(0.4 * FS)
    left = notch_unsettled_s(FS, n, q=30.0, freqs=HG_LINES)
    assert left == pytest.approx(need - (n - 1) / FS, rel=1e-9)
    assert notch_unsettled_s(FS, int(30 * FS), q=30.0, freqs=HG_LINES) == 0.0


def test_lines_at_or_above_nyquist_are_dropped_with_a_warning_that_names_them():
    with pytest.warns(RuntimeWarning, match=r"Nyquist"):
        keep = notch(np.zeros(2048), 500.0, freqs=[60.0, 120.0, 300.0])
    assert keep.shape == (2048,)
    assert notch_frequencies(500.0, freqs=[60.0, 120.0, 300.0]) == [60.0, 120.0]


def test_a_set_entirely_above_nyquist_is_a_refusal_not_an_empty_filter():
    with pytest.raises(ValueError, match=r"no notch frequency below Nyquist"):
        notch_frequencies(500.0, freqs=[300.0, 600.0])
    with pytest.raises(ValueError, match=r"no notch frequency below Nyquist"):
        notch(np.zeros(2048), 500.0, freqs=[300.0, 600.0])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_a_duplicated_line_is_refused_rather_than_de_duplicated():
    """Two identical biquads are a DEEPER, longer-ringing notch than the one line the
    user believes they asked for. Silently dropping the second would equally hide a
    typo in a nine-line list."""
    with pytest.raises(ValueError, match=r"duplicate notch frequencies"):
        notch(np.zeros(4096), FS, freqs=[60.0, 120.0, 60.0])


def test_an_empty_list_is_refused_because_it_is_not_a_disabled_stage():
    with pytest.raises(ValueError, match=r"at least one frequency"):
        notch(np.zeros(4096), FS, freqs=[])


@pytest.mark.parametrize("bad", [[60.0, 0.0], [60.0, -120.0], [60.0, np.nan], [np.inf]])
def test_a_non_positive_or_non_finite_line_is_refused(bad):
    with pytest.raises(ValueError, match=r"positive and finite"):
        notch(np.zeros(4096), FS, freqs=bad)


# ---------------------------------------------------------------------------
# the owner's own data path, when this machine has it
# ---------------------------------------------------------------------------

#: The host-guest training run, whose written metadata carries
#: ``notch_hz: [60, 120, 180, 300, 420, 540]``.
HG_CSV = Path(
    "/Users/abrahamcampos/Desktop/Project-2_Host-Guest/CSVData/cb7ada_75mV.csv.gz")


@pytest.mark.skipif(not HG_CSV.exists(), reason="the owner's cb7ada_75mV CSV is not here")
def test_on_the_real_host_guest_recording_only_the_head_differs_from_the_notebook():
    """The topology/pad argument, settled on real data rather than a synthetic.

    Raw channel-0 current from the recording the six-line set was chosen for, filtered
    two ways: the app's stacked cascade with its settled pad, and the notebook's six
    sequential ``filtfilt`` passes at scipy's default pad. Measured over a 40 s slice:
    0.59 pA of difference in the first 0.1 s (0.26 of this recording's 2.26 pA noise
    sigma), 0.14 pA out to 1 s, and 1.9e-6 pA -- ONE float32 ULP -- through the whole
    interior. The slice's own tail carries the mirror-image transient (1.8 pA), which
    is why the comparison stops 3 s short of the end.

    So the two are the same filter on this data everywhere except two edge transients
    smaller than the noise, and the app's version is the one whose edges are sized
    from its own ring-down. There is nothing here to offer as an option.
    """
    import pandas as pd

    df = pd.read_csv(HG_CSV, usecols=["current_raw_pA"], nrows=2_000_000)
    raw = df["current_raw_pA"].to_numpy(np.float32)

    app = np.asarray(notch(raw, FS, freqs=HG_LINES), dtype=np.float64)
    book = _sequential(raw, HG_LINES, 30.0, None)
    d = np.abs(app - book)

    assert float(d[: int(0.1 * FS)].max()) < 1.0
    interior = d[int(3 * FS): raw.size - int(3 * FS)]
    assert float(interior.max()) < 1e-4
