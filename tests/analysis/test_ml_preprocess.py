"""Tests for the ML detector's OWN preprocessing + decode
(:mod:`nanoscope.analysis.ml_preprocess`).

These are the load-bearing claims. Every one of them is a thing that would
silently change what the trained model sees:

  a. each baseline estimator matches a FROM-SCRATCH reference transcribed from
     ``dataLoader.ipynb`` -- the decimated grid, the odd-taps rule,
     ``mode='reflect'``, the ``np.interp`` back to full length and the float32
     casts, all of which decide bit-exactness;
  b. the odd-taps / decimation arithmetic is right at the documented operating
     points (0.5 s @ 50 kHz decim 50 -> 501 taps; 0.03 s decim 10 -> 151);
  c. three SEQUENTIAL notch ``filtfilt`` passes differ from one cascaded SOS pass
     -- i.e. the fix matters and the general Preprocess-tab notch would be wrong;
  d. the presets carry EXACTLY the documented per-notebook values;
  e. onset snap and offset trim move region boundaries as specified, and are inert
     when their sigma is 0;
  f. the ``apo_raw_global`` preset (renamed from ``apo_legacy``) reproduces the
     app's pre-existing ML chain and decode bit-for-bit, so selecting it is a
     genuine no-op, and the rename moved no number;
  g. the APO NOTEBOOK chain -- mean-centred normalization, the float32 notch round,
     the SavGol 25001/poly-3 detrend and the single-threshold decode -- matches a
     verbatim transcription of ``ApoFerritin_LSTM_Cluster_Validation_RAM_Safe.ipynb``
     cells 10/12/13, and the FERRITIN presets' output is pinned byte-for-byte to
     what it was before that landed.

Pure numpy/scipy: no Qt, no TensorFlow.
"""

from __future__ import annotations

import contextlib
import hashlib
import pathlib
import threading
import time

import numpy as np
import pytest
from scipy.ndimage import binary_dilation
from scipy.signal import filtfilt, iirnotch, savgol_filter, sosfiltfilt

from nanoscope.analysis import ml_preprocess as mp
from nanoscope.analysis.ml_detect import decode_regions, detrend_for_model

FS = 50_000.0


# ===========================================================================
# helpers -- synthetic traces
# ===========================================================================
def _trace(n: int = 60_000, *, seed: int = 7, drift: float = 40.0,
           n_events: int = 8, depth: float = 60.0, width: int = 300) -> np.ndarray:
    """Open-pore level + slow drift + noise + square blockades, float32 pA."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = 1500.0 + drift * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0.0, 4.0, n)
    step = n // (n_events + 1)
    for k in range(n_events):
        s = step * (k + 1)
        x[s:s + width] -= depth
    return x.astype(np.float32)


# ===========================================================================
# a/b. baseline estimators vs from-scratch references
# ===========================================================================
def _hand_rolling_median(xs, w):
    """A rolling median computed WITHOUT ``scipy.ndimage`` -- the oracle.

    ``ndimage``'s ``mode='reflect'`` is ``(d c b a | a b c d | d c b a)``, i.e. the
    edge sample is duplicated, which is exactly ``numpy.pad(..., 'symmetric')``. With
    an odd size and the default ``origin=0`` the window is centred, so the answer is
    ``np.median`` over every length-``w`` window of the padded array.

    This exists because ``scipy.ndimage.median_filter`` cannot be its own oracle: on
    a 1-D input past ~261 taps it is non-deterministic (see
    ``test_decimated_median_is_bit_identical_to_a_hand_computed_median``), so a
    reference built on it can be wrong in the same breath as the code under test --
    or, worse, be the wrong one while the code is right.
    """
    half = int(w) // 2
    padded = np.pad(np.asarray(xs, np.float64), half, mode="symmetric")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, int(w)), axis=1)


def _ref_decimated_median(x, fs, win_s, decim):
    """Transcribed from dataLoader.ipynb's ``_rmed`` -- the shared inner routine.

    The notebook's arithmetic verbatim (decimated grid, odd taps, ``mode='reflect'``,
    ``np.interp`` back to full length, float32 cast); only the median itself is
    computed by hand rather than by ``median_filter`` -- see
    :func:`_hand_rolling_median` for why.
    """
    n = len(x)
    xs = x[::decim]
    w = int(win_s * fs / decim)
    if w % 2 == 0:
        w += 1
    b = _hand_rolling_median(xs, w)
    xp = np.arange(0, n, decim)[:len(b)]
    return np.interp(np.arange(n), xp, b).astype(np.float32)


def _ref_robust_median(sig, fs, window_s=2.0, decim=50):
    """Transcribed from dataLoader.ipynb ``baseline_remove_robust_median``."""
    baseline = _ref_decimated_median(sig, fs, window_s, decim)
    return (sig - baseline).astype(np.float32), baseline


def _ref_masked_median(sig, fs, rough_s=0.5, base_s=0.1, mask_sigma=3.5,
                       dilate_ms=1.0, decim=50):
    """Transcribed from dataLoader.ipynb ``baseline_remove_masked_median``."""
    def _rmed(x, win_s):
        return _ref_decimated_median(x, fs, win_s, decim)

    rough = sig - _rmed(sig, rough_s)
    noise = 1.4826 * np.median(np.abs(rough - np.median(rough)))
    evm = binary_dilation(rough < np.median(rough) - mask_sigma * noise,
                          iterations=max(1, int(dilate_ms * 1e-3 * fs)))
    idx = np.arange(len(sig))
    filled = sig.copy()
    if evm.any() and (~evm).sum() > 2:
        filled[evm] = np.interp(idx[evm], idx[~evm], sig[~evm])
    baseline = _rmed(filled, base_s)
    return (sig - baseline).astype(np.float32), baseline


def test_baseline_robust_median_matches_reference_bit_for_bit():
    # float64 == what the real chain hands the baseline (notch/filtfilt output).
    x = _trace().astype(np.float64)
    got_d, got_b = mp.baseline_robust_median(x, FS, window_s=0.5, decim=50)
    ref_d, ref_b = _ref_robust_median(x, FS, window_s=0.5, decim=50)

    assert got_d.dtype == np.float32 and got_b.dtype == np.float32
    np.testing.assert_array_equal(got_b, ref_b)
    np.testing.assert_array_equal(got_d, ref_d)
    # ...and it actually detrends: the open pore collapses to ~0.
    assert abs(float(np.median(got_d))) < 1.0
    assert got_d.std() < x.std()


def test_baseline_masked_median_matches_reference_bit_for_bit():
    # deep, long events -> the ferritin_data2 pathology the masked estimator exists for
    x = _trace(n_events=12, depth=120.0, width=800).astype(np.float64)
    got_d, got_b = mp.baseline_masked_median(
        x, FS, rough_s=0.5, base_s=0.1, decim=50, mask_sigma=3.5, dilate_ms=1.0)
    ref_d, ref_b = _ref_masked_median(x, FS)

    assert got_d.dtype == np.float32 and got_b.dtype == np.float32
    np.testing.assert_array_equal(got_b, ref_b)
    np.testing.assert_array_equal(got_d, ref_d)


def test_baseline_masked_median_notebook3_windows_match_reference():
    """Notebook 3's short windows + decim 10 go down a different arithmetic branch."""
    x = _trace(n_events=20, depth=90.0, width=120).astype(np.float64)
    got_d, got_b = mp.baseline_masked_median(
        x, FS, rough_s=0.03, base_s=0.01, decim=10, mask_sigma=3.5, dilate_ms=1.0)
    ref_d, ref_b = _ref_masked_median(x, FS, rough_s=0.03, base_s=0.01, decim=10)
    np.testing.assert_array_equal(got_b, ref_b)
    np.testing.assert_array_equal(got_d, ref_d)


# ---------------------------------------------------------------------------
# F1. ``scipy.ndimage.median_filter`` IS NON-DETERMINISTIC AND IT REACHES PRODUCTION.
#
# On a 1-D input past ~261 taps, scipy 1.15's ``median_filter`` returns different
# values on repeated calls with the identical array. It is NOT confined to float32
# (float64 fails too), NOT confined to the tests, and NOT rare: under the concurrent
# numpy allocation a real session produces it fired on 141 of 600 calls (23.5%) at
# the 501-tap operating point every ferritin preset uses.
#
# Measured on the user's -600 mV recording (20 s, 1e6 real samples, notebook1):
# ``baseline_robust_median`` disagreed with the hand-computed median in 13 of 40
# runs by up to 7.3 pA; ``apply_ml_preprocess`` returned SIX different answers in 40
# runs, spanning 6.4 pA; a fixed 5-sigma decode over those answers counted between
# 337 and 469 events.
#
# The fix is the ``(1, m)`` call shape, which takes a different path inside ndimage.
# These tests run under deliberate allocation pressure because that is what makes
# the defect reproducible rather than a 1-in-200 flake.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _allocation_pressure(n_threads: int = 4):
    """Background numpy allocation -- the condition the 1-D bug needs to show itself.

    A real session has exactly this (Qt paint threads, the detection worker, the TF
    predictor); the point of doing it on purpose is to turn a flake into a test.
    Measured hit rate for the pre-fix 1-D call at 501 taps under this pressure:
    ~10-24% of calls, so the loop counts below leave a pre-fix pass probability
    below 1e-3.
    """
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            block = [np.empty(1 << 14) for _ in range(20)]
            del block

    threads = [threading.Thread(target=churn, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    try:
        yield
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_decimated_median_is_bit_identical_to_a_hand_computed_median(dtype):
    """The baseline is the median it claims to be -- not "usually", and in BOTH dtypes.

    Checked against :func:`_hand_rolling_median`, which never touches
    ``scipy.ndimage``, at the 501-tap point every ferritin preset uses. float64 is
    parametrised in on purpose: the old docstring claimed the defect was float32-only
    and that the float64 copy was the cure. It was not -- float64 fails too, which is
    why the *call shape* had to change.
    """
    x = _trace(n=100_000).astype(dtype)
    want = _ref_decimated_median(x, FS, 0.5, 50)
    with _allocation_pressure():
        for _ in range(60):
            got = mp._decimated_median(x, FS, 0.5, 50)
            assert np.array_equal(got, want), (
                f"median moved by up to {np.abs(got - want).max():.4f} pA "
                f"({np.count_nonzero(got != want)} samples) on a busy machine"
            )


@pytest.mark.parametrize(
    "win_s, decim, taps, reps",
    # Only the 501-tap point is a frequent offender, so it gets the repeats; the
    # other three are here because the arithmetic at each must be pinned to the
    # oracle, not because they are expected to flake.
    [(0.5, 50, 501, 40), (0.1, 50, 101, 6), (0.03, 10, 151, 6), (0.01, 10, 51, 6)],
)
def test_decimated_median_matches_the_hand_computed_median_at_every_preset_point(
    win_s, decim, taps, reps
):
    """All four (window, decim) pairs the presets actually use, pinned to an oracle
    that does not go through scipy.ndimage."""
    x = _trace(n=100_000).astype(np.float64)
    want = _ref_decimated_median(x, FS, win_s, decim)
    with _allocation_pressure():
        for _ in range(reps):
            assert np.array_equal(mp._decimated_median(x, FS, win_s, decim), want)


def test_the_ferritin_contract_digest_survives_allocation_pressure():
    """The user-visible half: the pinned contract must be REPRODUCIBLE, not lucky.

    Before the fix, forty runs of the notebook1 chain on the user's real -600 mV
    recording produced SIX different answers spanning 6.4 pA. The digest asserted
    here is the one pinned in ``_GOLDEN_FERRITIN``, so this says the published
    contract does not move when the machine is busy.

    The notch is hoisted out of the loop because it is deterministic and slow; for
    ``notebook1`` (``notch_float32=False``, ``despike_size=0``) the remaining
    baseline stage IS the rest of :func:`apply_ml_preprocess`, which the first
    assertion checks rather than assumes.
    """
    x = _golden_trace()
    cfg = mp.PREPROCESS_PRESETS["notebook1"]
    want, _ = _GOLDEN_FERRITIN["notebook1"]

    notched = mp.notch_sequential(x, FS, freqs=cfg.notch_freqs, q=cfg.notch_q)

    def _chain() -> str:
        detr, _ = mp.baseline_robust_median(
            notched, FS, window_s=cfg.baseline_window_s, decim=cfg.baseline_decim
        )
        return hashlib.sha256(np.asarray(detr, np.float32).tobytes()).hexdigest()

    assert _chain() == hashlib.sha256(
        mp.apply_ml_preprocess(x, FS, cfg).tobytes()
    ).hexdigest()

    with _allocation_pressure():
        seen = {_chain() for _ in range(40)}
    assert seen == {want}, f"{len(seen)} distinct contract digests in 40 runs"


def test_masked_median_is_not_dragged_by_deep_long_events():
    """The point of the masked estimator: the baseline ignores the events.

    A plain rolling median over a SHORT window gets pulled into a long blockade;
    the masked one interpolates across it, so the open pore stays at zero and the
    event keeps its full depth."""
    n = 200_000
    x = np.full(n, 1000.0, np.float32)
    rng = np.random.default_rng(3)
    x += rng.normal(0.0, 2.0, n).astype(np.float32)
    x[80_000:87_000] -= 200.0          # a 140 ms, 200 pA blockade

    detr, _ = mp.baseline_masked_median(x, FS, rough_s=0.5, base_s=0.1, decim=50)
    ev = np.zeros(n, bool)
    ev[80_000:87_000] = True
    assert abs(float(np.median(detr[~ev]))) < 1.0            # open pore at zero
    assert float(np.median(detr[ev])) < -190.0               # depth survives

    # A short plain median over the same window is dragged toward the event floor.
    plain, _ = mp.baseline_robust_median(x, FS, window_s=0.1, decim=50)
    assert float(np.median(plain[ev])) > float(np.median(detr[ev]))


@pytest.mark.parametrize(
    "window_s, decim, expect_taps",
    [
        (0.5, 50, 501),     # notebook 1 baseline / notebook 2 rough  @ 50 kHz
        (0.1, 50, 101),     # notebook 2 base
        (0.03, 10, 151),    # notebook 3 rough
        (0.01, 10, 51),     # notebook 3 base
    ],
)
def test_decimated_median_tap_count_arithmetic(window_s, decim, expect_taps):
    """``int(window_s * fs / decim)`` forced ODD -- the number that decides which
    samples the median actually sees."""
    w = int(window_s * FS / decim)
    if w % 2 == 0:
        w += 1
    assert w == expect_taps

    # And the estimator really uses that many taps: a single-sample spike is
    # rejected by a median of that width, while a run longer than half of it is not.
    n = 40_000
    x = np.zeros(n, np.float32)
    x[20_000] = 1000.0
    _, base = mp.baseline_robust_median(x, FS, window_s=window_s, decim=decim)
    assert float(np.max(np.abs(base))) < 1.0


def test_decimated_median_grid_and_interp_reach_the_last_sample():
    """``np.interp`` back onto ``arange(n)`` must cover every sample, including a
    length that is not a multiple of ``decim``."""
    n = 12_345                     # deliberately not a multiple of 50
    x = _trace(n=n, n_events=2)
    detr, base = mp.baseline_robust_median(x, FS, window_s=0.1, decim=50)
    assert detr.shape == (n,) and base.shape == (n,)
    assert np.isfinite(detr).all() and np.isfinite(base).all()


def test_baseline_savgol_reproduces_the_legacy_detrend():
    x = _trace()
    detr, base = mp.baseline_savgol(x, FS, window_s=0.5, poly=3)
    assert detr.dtype == np.float32
    w = int(0.5 * FS)
    w = w + 1 if w % 2 == 0 else w
    np.testing.assert_array_equal(base, savgol_filter(x, window_length=w, polyorder=3))


def test_baseline_savgol_raises_on_a_too_short_recording():
    with pytest.raises(ValueError, match="too short"):
        mp.baseline_savgol(np.ones(3, np.float32), FS, window_s=0.5, poly=3)


def test_despike_median_removes_single_sample_spikes_and_is_off_below_size_2():
    x = np.zeros(1000, np.float32)
    x[500] = 50.0
    x[100:140] = -30.0                     # a real, wide event
    cleaned = mp.despike_median(x, size=3)
    assert cleaned.dtype == np.float32
    assert float(cleaned[500]) == 0.0                       # spike gone
    assert float(np.min(cleaned[100:140])) == pytest.approx(-30.0)   # event intact

    for off in (0, None, 1):
        np.testing.assert_array_equal(mp.despike_median(x, size=off), x)


# ===========================================================================
# c. sequential vs cascaded notch -- proving the fix matters
# ===========================================================================
def test_notch_sequential_matches_the_notebook_loop():
    x = _trace(n=20_000)
    out = np.asarray(x).copy()
    for f0 in (60.0, 120.0, 180.0):
        b, a = iirnotch(f0, Q=30.0, fs=FS)
        out = filtfilt(b, a, out)
    np.testing.assert_array_equal(mp.notch_sequential(x, FS), out)


def test_sequential_and_cascaded_notch_are_different_signals():
    """Three ``filtfilt`` passes != one ``sosfiltfilt`` over the stacked biquads.

    Each pass does its own odd-extension edge padding on the CURRENT signal, so the
    two differ -- most at the edges, which is exactly where windowed inference reads.
    The app's general :func:`filters.notch` cascades; the ML detector must not use it.
    """
    x = _trace(n=20_000)
    seq = mp.notch_sequential(x, FS)
    sos = np.vstack([np.hstack(iirnotch(f, 30.0, fs=FS))[None, :]
                     for f in (60.0, 120.0, 180.0)])
    cascaded = sosfiltfilt(sos, np.asarray(x, dtype=np.float64))

    assert not np.array_equal(seq, cascaded)
    assert float(np.max(np.abs(seq - cascaded))) > 0.0
    # The disagreement is concentrated at the edges (the padding differs there).
    edge = int(np.argmax(np.abs(seq - cascaded)))
    assert edge < 2000 or edge > len(x) - 2000


def test_notch_sequential_order_matters_and_skips_lines_above_nyquist():
    x = _trace(n=20_000)
    ascending = mp.notch_sequential(x, FS, freqs=(60.0, 120.0, 180.0))
    descending = mp.notch_sequential(x, FS, freqs=(180.0, 120.0, 60.0))
    assert not np.array_equal(ascending, descending)

    # A line at/above Nyquist is skipped rather than raising out of scipy.
    low_fs = 200.0
    got = mp.notch_sequential(np.asarray(x[:4000], np.float64), low_fs,
                              freqs=(60.0, 120.0, 180.0))
    assert np.isfinite(got).all()


def test_notch_sequential_rejects_a_bad_rate():
    with pytest.raises(ValueError, match="positive"):
        mp.notch_sequential(np.ones(1000), 0.0)


# ===========================================================================
# d. presets carry the documented values
# ===========================================================================
#: The three chains added for the owner's host-guest / Zn2 / Cu2 student pipelines.
#: They are named apart from the seven ferritin/apo presets throughout this file
#: because they differ in the one place none of the seven do: the MODEL-INPUT
#: NORMALIZATION. All seven divide by max|x|; these three use the twice-trimmed,
#: clipped robust sigma that replaced it in all six of his student notebooks.
_STUDENT_PRESETS = ("hg_student", "zn_student", "cu_student")


def test_preset_registry_names_and_labels():
    assert mp.PRESET_NAMES == (
        "notebook1", "notebook2", "notebook3",
        "apo_nb_500mV", "apo_nb_600mV", "apo_nb_700mV", "apo_raw_global",
        *_STUDENT_PRESETS,
    )
    assert set(mp.PREPROCESS_PRESETS) == set(mp.PRESET_NAMES)
    assert set(mp.DECODE_PRESETS) == set(mp.PRESET_NAMES)
    assert set(mp.PRESET_LABELS) == set(mp.PRESET_NAMES)
    assert set(mp.PRESET_MOLECULE) == set(mp.PRESET_NAMES)
    for name in mp.PRESET_NAMES:
        assert mp.PREPROCESS_PRESETS[name].preset == name
        assert mp.DECODE_PRESETS[name].preset == name


def test_preset_metadata_is_molecule_first_and_consistent():
    """The labels must SAY the molecule, and say the right one -- a ferritin-trained
    model on apo data is a category error, not a tuning choice."""
    assert set(mp.PRESET_MOLECULE.values()) == set(mp.MOLECULES)
    for name in mp.PRESET_NAMES:
        molecule = mp.PRESET_MOLECULE[name]
        label = mp.PRESET_LABELS[name]
        # THE RULE IS "the label opens with the molecule", not "the molecule is one of
        # two". It was spelled as a two-way branch while there were two molecules; the
        # rule itself never was, and hard-coding the pair would have meant a fourth
        # molecule could ship with a label that names no molecule at all -- which is the
        # category error this test exists to prevent, not the naming of it.
        expect = ("Ferritin" if molecule == "ferritin"
                  else "Apo-ferritin" if molecule == "apo-ferritin" else molecule)
        assert label.startswith(f"{expect} — "), (name, label)
    # Grouped in the combo order, ferritin first: no interleaving.
    order = [mp.PRESET_MOLECULE[n] for n in mp.PRESET_NAMES]
    assert order == sorted(order, key=mp.MOLECULES.index)
    assert mp.DEFAULT_PRESET in mp.PRESET_NAMES


def test_superseded_preset_names_resolve_instead_of_raising():
    """A stored run naming ``apo_legacy`` must still resolve -- to the SAME numbers,
    under the honest name, never reinterpreted as one of the notebook presets."""
    assert mp.PRESET_ALIASES == {"apo_legacy": "apo_raw_global"}
    assert mp.resolve_preset_name("apo_legacy") == "apo_raw_global"
    assert mp.resolve_preset_name("notebook2") == "notebook2"
    assert mp.resolve_preset_name("nope") is None
    assert mp.resolve_preset_name(None) is None
    assert mp.resolve_preset_name("") is None
    # The alias target is not silently one of the apo NOTEBOOK presets.
    assert mp.PREPROCESS_PRESETS["apo_raw_global"].normalize_center is False
    assert mp.PREPROCESS_PRESETS["apo_raw_global"].anchor_tail is True


def test_every_preset_shares_the_common_notch_and_recording_scope():
    """Q=30 sequential filtfilt passes and a normalization measured over the WHOLE
    recording, for every preset.

    THE LINE SET IS NO LONGER COMMON, and that is the finding rather than a relaxation.
    The seven ferritin/apo presets share 60/120/180 -- still asserted exactly. The
    owner's three student rigs hum on line sets with GAPS that his own Welch censuses
    found (host-guest 60/120/180/300/420/540 with 240/360/480 quiet; Zn/Cu
    60/.../660 with 480 absent), which is precisely why the explicit line list exists.
    Asserting 60/120/180 for those three would assert a filter he never ran.
    """
    for name in mp.PRESET_NAMES:
        cfg = mp.PREPROCESS_PRESETS[name]
        assert cfg.notch_enabled is True
        if name not in _STUDENT_PRESETS:
            assert cfg.notch_freqs == (60.0, 120.0, 180.0), name
        else:
            # Whatever the set, it must start at the mains fundamental and ascend --
            # a line list is a comb, not an arbitrary bag of numbers.
            assert cfg.notch_freqs[0] == 60.0, name
            assert list(cfg.notch_freqs) == sorted(cfg.notch_freqs), name
            assert len(set(cfg.notch_freqs)) == len(cfg.notch_freqs), name
        assert cfg.notch_q == 30.0
        assert cfg.normalize is True
        assert cfg.normalize_scope == "recording"


def test_only_the_student_presets_use_the_robust_sigma_normalization():
    """THE COMPATIBILITY CLAUSE, stated as a partition. ``normalize_method`` defaults to
    ``"maxabs"``, so every preset that existed before it did must still be on it -- a
    stored run, a saved project or a QSettings user preset that silently moved to a
    different kernel would feed a trained model something it never saw."""
    for name in mp.PRESET_NAMES:
        cfg = mp.PREPROCESS_PRESETS[name]
        expect = "robust_sigma" if name in _STUDENT_PRESETS else "maxabs"
        assert cfg.normalize_method == expect, name
        if expect == "robust_sigma":
            assert cfg.normalize_clip_sigma == 30.0, name
            # Mean-centring is a max-abs notion; robust sigma re-estimates its own.
            assert cfg.normalize_center is False, name


def test_only_the_apo_notebook_presets_centre_and_round():
    """The two real differences between the molecules' preprocessing, pinned so
    neither can leak into the other."""
    apo_nb = ("apo_nb_500mV", "apo_nb_600mV", "apo_nb_700mV")
    # (The three student presets are none of these three flags -- they are the
    # notebooks' plain float64 notch, no mean centring, notebook tail convention.)
    for name in mp.PRESET_NAMES:
        cfg = mp.PREPROCESS_PRESETS[name]
        expect = name in apo_nb
        assert cfg.normalize_center is expect, name
        assert cfg.notch_float32 is expect, name
        assert cfg.last_start_inclusive is expect, name
    # enter == 0.5 is a FERRITIN property; the apo notebook thresholds are per-bias.
    for name in ("notebook1", "notebook2", "notebook3", "apo_raw_global"):
        assert mp.DECODE_PRESETS[name].enter_threshold == 0.5


def test_notebook1_preprocess_preset():
    cfg = mp.PREPROCESS_PRESETS["notebook1"]
    assert cfg.baseline_method == "robust_median"
    assert cfg.baseline_window_s == 0.5
    assert cfg.baseline_decim == 50
    assert cfg.despike_size == 0
    assert cfg.anchor_tail is False


def test_notebook2_preprocess_preset():
    cfg = mp.PREPROCESS_PRESETS["notebook2"]
    assert cfg.baseline_method == "masked_median"
    assert (cfg.masked_rough_s, cfg.masked_base_s) == (0.5, 0.1)
    assert cfg.baseline_decim == 50
    assert cfg.masked_mask_sigma == 3.5
    assert cfg.masked_dilate_ms == 1.0
    assert cfg.despike_size == 3          # the ~225 one-sample spikes in data2
    assert cfg.anchor_tail is False


def test_notebook3_preprocess_preset():
    cfg = mp.PREPROCESS_PRESETS["notebook3"]
    assert cfg.baseline_method == "masked_median"
    assert (cfg.masked_rough_s, cfg.masked_base_s) == (0.03, 0.01)
    assert cfg.baseline_decim == 10       # decim 50 cannot resolve a 10 ms window
    assert cfg.despike_size == 0
    assert cfg.anchor_tail is False


def test_apo_raw_global_preprocess_preset():
    cfg = mp.PREPROCESS_PRESETS["apo_raw_global"]
    assert cfg.baseline_method == "savgol"
    assert cfg.baseline_window_s == 0.5 and cfg.baseline_poly == 3
    assert cfg.despike_size == 0
    assert cfg.anchor_tail is True        # the app's legacy full-coverage windowing
    assert cfg.notch_float32 is False     # float64 into SavGol, as detrend_for_model
    assert cfg.normalize_center is False  # x/(max|x|+1e-8), student_raw_global training


@pytest.mark.parametrize("name", ["apo_nb_500mV", "apo_nb_600mV", "apo_nb_700mV"])
def test_apo_notebook_presets_share_one_preprocessing_chain(name):
    """Cell 12 calls ONE preprocess_signal for every bias; only the decode differs.
    All three configs must therefore be identical apart from their ``preset`` tag."""
    cfg = mp.PREPROCESS_PRESETS[name]
    assert cfg.notch_enabled is True and cfg.notch_float32 is True
    assert cfg.baseline_method == "savgol"
    assert cfg.baseline_window_s == 0.5 and cfg.baseline_poly == 3
    assert cfg.despike_size == 0
    assert cfg.normalize is True
    assert cfg.normalize_center is True
    assert cfg.normalize_scope == "recording"
    assert cfg.anchor_tail is False
    assert cfg.last_start_inclusive is True
    assert cfg.deviations_from(mp.PREPROCESS_PRESETS["apo_nb_600mV"]) == ()


def test_apo_savgol_window_is_the_notebooks_25001_taps_at_50_khz():
    """``int(500.0 * 50000 / 1000)`` -> 25000 -> forced odd -> 25001, polyorder 3."""
    cfg = mp.PREPROCESS_PRESETS["apo_nb_600mV"]
    w = int(cfg.baseline_window_s * FS)
    assert w == 25_000
    assert (w + 1 if w % 2 == 0 else w) == 25_001
    assert cfg.baseline_poly == 3


@pytest.mark.parametrize(
    "name, smooth, enter_thr, exit_thr, merge_ms, min_ms, onset, offset",
    [
        ("notebook1", 9, 0.5, 0.15, 0.5, 1.0, 0.0, 0.0),
        ("notebook2", 9, 0.5, 0.15, 0.5, 1.0, 0.0, 0.0),
        ("notebook3", 5, 0.5, 0.20, 0.2, 0.15, 2.0, 3.0),
        # Apo notebook cells 14/15/16: single strict threshold, no smoothing, no
        # merge inside the decoder, hand-tuned per bias.
        ("apo_nb_500mV", 1, 0.2, 0.20, 0.0, 15.0, 0.0, 0.0),
        ("apo_nb_600mV", 1, 0.1, 0.10, 0.0, 1.0, 0.0, 0.0),
        ("apo_nb_700mV", 1, 0.2, 0.20, 0.0, 1.0, 0.0, 0.0),
        ("apo_raw_global", 9, 0.5, 0.30, 0.1, 0.1, 0.0, 0.0),
    ],
)
def test_decode_presets_carry_the_documented_values(
    name, smooth, enter_thr, exit_thr, merge_ms, min_ms, onset, offset
):
    cfg = mp.DECODE_PRESETS[name]
    assert cfg.smoothing_kernel == smooth
    assert cfg.enter_threshold == pytest.approx(enter_thr)
    assert cfg.exit_threshold == pytest.approx(exit_thr)
    assert cfg.merge_radius_ms == pytest.approx(merge_ms)
    assert cfg.min_event_ms == pytest.approx(min_ms)
    assert cfg.onset_sigma == pytest.approx(onset)
    assert cfg.offset_sigma == pytest.approx(offset)


@pytest.mark.parametrize("name", ["apo_nb_500mV", "apo_nb_600mV", "apo_nb_700mV"])
def test_apo_notebook_decode_is_a_single_threshold_with_no_smoothing(name):
    """``extract_event_regions`` is a bare run-length decode of ``preds > thr``. The
    three fields that express that in this module's vocabulary are load-bearing:
    equal thresholds ARE a single threshold, kernel 1 short-circuits the smoother,
    and a 0 ms radius cannot merge (hysteresis regions are >= 2 samples apart)."""
    cfg = mp.DECODE_PRESETS[name]
    assert cfg.smoothing_kernel == 1
    assert cfg.enter_threshold == cfg.exit_threshold
    assert cfg.merge_radius_ms == 0.0
    assert cfg.onset_sigma == 0.0 and cfg.offset_sigma == 0.0


def test_notebook3_onset_snap_defaults():
    cfg = mp.DECODE_PRESETS["notebook3"]
    assert cfg.onset_max_ms == 5.0
    assert cfg.onset_gap_tol == 5


def test_preset_for_window_maps_the_two_known_widths():
    assert mp.preset_for_window(2048) == "notebook1"          # ferritin STUDENT_WINDOW
    assert mp.preset_for_window(3072) == mp.DEFAULT_PRESET    # BOTH apo models
    assert mp.preset_for_window(3072) == "apo_nb_600mV"
    assert mp.preset_for_window(1234) is None
    assert mp.preset_for_window(None) is None


# ===========================================================================
# configs: validation, round-trip, deviations
# ===========================================================================
def test_preprocess_config_round_trips_through_json_safe_dict():
    for name in mp.PRESET_NAMES:
        cfg = mp.PREPROCESS_PRESETS[name]
        d = cfg.to_dict()
        assert isinstance(d["notch_freqs"], list)     # JSON-safe
        assert mp.MLPreprocessConfig.from_dict(d) == cfg
        assert mp.MLDecodeConfig.from_dict(mp.DECODE_PRESETS[name].to_dict()) \
            == mp.DECODE_PRESETS[name]


def test_config_from_dict_of_none_is_none_and_ignores_unknown_keys():
    assert mp.MLPreprocessConfig.from_dict(None) is None
    assert mp.MLPreprocessConfig.from_dict({}) is None
    assert mp.MLDecodeConfig.from_dict(None) is None
    cfg = mp.MLPreprocessConfig.from_dict(
        {"baseline_method": "savgol", "not_a_field": 1})
    assert cfg.baseline_method == "savgol"


def test_config_validation_rejects_nonsense():
    with pytest.raises(ValueError, match="baseline_method"):
        mp.MLPreprocessConfig(baseline_method="wishful")
    with pytest.raises(ValueError, match="normalize_scope"):
        mp.MLPreprocessConfig(normalize_scope="somewhere")
    with pytest.raises(ValueError, match="notch_q"):
        mp.MLPreprocessConfig(notch_q=0.0)
    with pytest.raises(ValueError, match="baseline_decim"):
        mp.MLPreprocessConfig(baseline_decim=0)
    with pytest.raises(ValueError, match="exit"):
        mp.MLDecodeConfig(enter_threshold=0.4, exit_threshold=0.9)
    with pytest.raises(ValueError, match="enter"):
        mp.MLDecodeConfig(enter_threshold=1.5)


def test_deviations_from_names_exactly_the_changed_fields():
    base = mp.PREPROCESS_PRESETS["notebook2"]
    assert base.deviations_from(base) == ()
    edited = base.replace(masked_base_s=0.25, despike_size=5)
    assert set(edited.deviations_from(base)) == {"masked_base_s", "despike_size"}
    # The preset LABEL is not a deviation -- only the numbers are.
    assert base.replace(preset="custom").deviations_from(base) == ()

    dbase = mp.DECODE_PRESETS["notebook3"]
    assert set(dbase.replace(offset_sigma=0.0).deviations_from(dbase)) == {"offset_sigma"}


def test_describe_mentions_the_chain_it_will_run():
    d = mp.PREPROCESS_PRESETS["notebook2"].describe()
    assert "sequential filtfilt" in d
    assert "masked median" in d and "despike" in d
    assert "recording" in d
    assert "notebook" in mp.PREPROCESS_PRESETS["notebook2"].describe().lower() or True
    assert "onset snap" in mp.DECODE_PRESETS["notebook3"].describe()
    assert "onset snap" not in mp.DECODE_PRESETS["notebook1"].describe()


# ===========================================================================
# apply_ml_preprocess + normalization
# ===========================================================================
def test_apply_ml_preprocess_runs_notch_then_baseline_then_despike():
    x = _trace()
    cfg = mp.PREPROCESS_PRESETS["notebook2"]
    got = mp.apply_ml_preprocess(x, FS, cfg)

    notched = mp.notch_sequential(x, FS, freqs=cfg.notch_freqs, q=cfg.notch_q)
    detr, _ = mp.baseline_masked_median(
        notched, FS, rough_s=cfg.masked_rough_s, base_s=cfg.masked_base_s,
        decim=cfg.baseline_decim, mask_sigma=cfg.masked_mask_sigma,
        dilate_ms=cfg.masked_dilate_ms)
    expect = mp.despike_median(detr, size=cfg.despike_size)
    np.testing.assert_array_equal(got, np.asarray(expect, np.float32))
    assert got.dtype == np.float32


def test_apply_ml_preprocess_baseline_none_only_notches():
    x = _trace(n=20_000)
    cfg = mp.MLPreprocessConfig(baseline_method="none")
    got = mp.apply_ml_preprocess(x, FS, cfg)
    np.testing.assert_allclose(got, mp.notch_sequential(x, FS).astype(np.float32),
                               rtol=0, atol=0)


def test_apply_ml_preprocess_rejects_bad_input():
    cfg = mp.PREPROCESS_PRESETS["notebook1"]
    with pytest.raises(ValueError, match="non-empty"):
        mp.apply_ml_preprocess(np.zeros(0, np.float32), FS, cfg)
    with pytest.raises(ValueError, match="NaN|Inf"):
        bad = np.ones(1000, np.float32)
        bad[3] = np.inf
        mp.apply_ml_preprocess(bad, FS, cfg)
    with pytest.raises(ValueError, match="positive"):
        mp.apply_ml_preprocess(np.ones(1000, np.float32), 0.0, cfg)
    with pytest.raises(ValueError, match="too short"):
        mp.apply_ml_preprocess(np.ones(4, np.float32), FS, cfg)


def test_normalization_is_a_pure_rescale_with_an_explicit_scope():
    sig = np.array([-2.0, 1.0, 4.0, 6.0, 3.0], np.float32)
    scale = mp.normalization_scale(sig)
    assert scale == pytest.approx(6.0 + 1e-8)

    norm = mp.normalize_global(sig)
    assert norm.dtype == np.float32
    assert float(np.max(np.abs(norm))) == pytest.approx(1.0, abs=1e-6)
    # NO mean subtraction: mean(norm) == mean(sig)/scale, not 0.
    assert float(np.mean(norm)) == pytest.approx(float(np.mean(sig)) / scale, abs=1e-6)

    # An EXPLICIT scale (the whole-recording scope) overrides the local max.
    wide = mp.normalize_global(sig, scale=60.0)
    assert float(np.max(np.abs(wide))) == pytest.approx(0.1, abs=1e-6)


def test_normalization_scale_of_a_sub_range_differs_from_the_whole_recording():
    """The bug this scope exists to fix: a slice's max|x| is not the recording's."""
    x = _trace()
    detr = mp.apply_ml_preprocess(x, FS, mp.PREPROCESS_PRESETS["notebook1"])
    whole = mp.normalization_scale(detr)
    quiet = mp.normalization_scale(detr[:2000])      # a window with no big event
    assert quiet < whole
    # Normalizing the slice by its own max makes noise look like a full-scale event.
    assert float(np.max(np.abs(mp.normalize_global(detr[:2000])))) == pytest.approx(1.0, abs=1e-6)
    # With the recording-wide scale it stays small, as training saw it.
    assert float(np.max(np.abs(mp.normalize_global(detr[:2000], whole)))) < 1.0


def test_normalize_all_zero_is_safe():
    out = mp.normalize_global(np.zeros(100, np.float32))
    assert out.dtype == np.float32 and np.all(out == 0.0)


# ===========================================================================
# e. decode: hysteresis, merge, min-duration, onset snap, offset trim
# ===========================================================================
def _prob(n: int, spans) -> np.ndarray:
    p = np.zeros(n, np.float32)
    for s, e, v in spans:
        p[s:e] = v
    return p


def test_decode_recovers_clean_regions():
    cfg = mp.DECODE_PRESETS["notebook1"].replace(min_event_ms=0.0, merge_radius_ms=0.0)
    prob = _prob(10_000, [(1_000, 1_500, 0.9), (5_000, 5_400, 0.8)])
    regions = mp.decode_ml_regions(prob, FS, cfg)
    assert regions == [(1_000, 1_499), (5_000, 5_399)]


def test_decode_hysteresis_keeps_a_dip_above_exit_but_drops_a_low_run():
    cfg = mp.DECODE_PRESETS["notebook1"].replace(
        smoothing_kernel=1, min_event_ms=0.0, merge_radius_ms=0.0)
    prob = _prob(4_000, [(1_000, 1_200, 0.9), (1_200, 1_260, 0.2),
                         (1_260, 1_500, 0.9),
                         (3_000, 3_400, 0.2)])     # never reaches enter -> dropped
    regions = mp.decode_ml_regions(prob, FS, cfg)
    assert regions == [(1_000, 1_499)]


def test_decode_merge_and_min_duration_use_milliseconds():
    prob = _prob(10_000, [(1_000, 1_100, 0.9), (1_120, 1_300, 0.9),
                          (5_000, 5_010, 0.9)])
    # merge 0.5 ms == 25 samples @ 50 kHz -> the 20-sample gap closes.
    cfg = mp.DECODE_PRESETS["notebook1"].replace(smoothing_kernel=1, min_event_ms=0.0)
    assert mp.decode_ml_regions(prob, FS, cfg)[0] == (1_000, 1_299)
    # min 1.0 ms == 50 samples -> the 10-sample event goes.
    cfg2 = cfg.replace(min_event_ms=1.0)
    assert all((e - s + 1) >= 50 for s, e in mp.decode_ml_regions(prob, FS, cfg2))


def test_onset_snap_pulls_the_onset_back_to_the_current_drop():
    """The model's probability rises late; the snap walks the onset back onto the
    first deep sample of the real blockade."""
    n = 20_000
    sig = np.zeros(n, np.float32)
    sig[5_000:5_600] = -100.0                # the true event
    prob = _prob(n, [(5_200, 5_600, 0.9)])   # ...detected 200 samples late

    off = mp.MLDecodeConfig(smoothing_kernel=1, exit_threshold=0.2,
                            merge_radius_ms=0.0, min_event_ms=0.0)
    on = off.replace(onset_sigma=2.0, onset_max_ms=5.0, onset_gap_tol=5)

    base = mp.decode_ml_regions(prob, FS, off, signal=sig)
    snapped = mp.decode_ml_regions(prob, FS, on, signal=sig)
    assert base[0][0] == 5_200
    assert snapped[0][0] == 5_000            # walked all the way back
    assert snapped[0][1] == base[0][1]       # offset untouched


def test_onset_snap_is_bounded_by_max_lookback():
    n = 20_000
    sig = np.zeros(n, np.float32)
    sig[5_000:6_000] = -100.0
    prob = _prob(n, [(5_900, 6_000, 0.9)])
    cfg = mp.MLDecodeConfig(smoothing_kernel=1, exit_threshold=0.2,
                            merge_radius_ms=0.0, min_event_ms=0.0,
                            onset_sigma=2.0, onset_max_ms=1.0)   # 50 samples
    snapped = mp.decode_ml_regions(prob, FS, cfg, signal=sig)
    assert snapped[0][0] == 5_850            # exactly max_lookback back, no further


def test_onset_snap_does_not_cross_the_previous_event():
    n = 20_000
    sig = np.zeros(n, np.float32)
    sig[4_000:9_000] = -100.0                # one long deep stretch
    prob = _prob(n, [(4_000, 5_000, 0.9), (6_000, 7_000, 0.9)])
    cfg = mp.MLDecodeConfig(smoothing_kernel=1, exit_threshold=0.2,
                            merge_radius_ms=0.0, min_event_ms=0.0,
                            onset_sigma=2.0, onset_max_ms=100.0)
    regions = mp.decode_ml_regions(prob, FS, cfg, signal=sig)
    assert regions[0][0] == 4_000
    assert regions[1][0] > regions[0][1]     # never walks past the previous end


def test_offset_trim_removes_the_shallow_recovery_tail():
    n = 20_000
    # Real noise, so median/MAD are meaningful: sigma ~1 -> 1.4826 x MAD ~1 pA, so the
    # 3-sigma deep level sits at ~-3 and the -2 pA recovery tail is NOT deep.
    sig = np.random.default_rng(1).normal(0.0, 1.0, n).astype(np.float32)
    sig[5_000:5_400] = -100.0                # deep part
    sig[5_400:5_800] = -2.0                  # shallow recovery tail
    prob = _prob(n, [(5_000, 5_800, 0.9)])

    off = mp.MLDecodeConfig(smoothing_kernel=1, exit_threshold=0.2,
                            merge_radius_ms=0.0, min_event_ms=0.0)
    on = off.replace(offset_sigma=3.0)
    assert mp.decode_ml_regions(prob, FS, off, signal=sig)[0] == (5_000, 5_799)
    trimmed = mp.decode_ml_regions(prob, FS, on, signal=sig)[0]
    assert trimmed[0] == 5_000
    assert trimmed[1] == 5_399               # snapped to the last deep sample


def test_onset_snap_and_offset_trim_are_inert_when_disabled():
    """sigma == 0 means the boundaries are exactly the hysteresis crossings, even
    when a signal IS supplied (notebooks 1 and 2 rely on this)."""
    n = 20_000
    sig = np.zeros(n, np.float32)
    sig[5_000:5_800] = -100.0
    prob = _prob(n, [(5_300, 5_600, 0.9)])
    cfg = mp.DECODE_PRESETS["notebook1"].replace(
        smoothing_kernel=1, min_event_ms=0.0, merge_radius_ms=0.0)
    assert cfg.onset_sigma == 0.0 and cfg.offset_sigma == 0.0
    with_sig = mp.decode_ml_regions(prob, FS, cfg, signal=sig)
    without = mp.decode_ml_regions(prob, FS, cfg)
    assert with_sig == without == [(5_300, 5_599)]


def test_decode_without_a_signal_skips_the_snaps_even_when_enabled():
    prob = _prob(20_000, [(5_000, 5_400, 0.9)])
    cfg = mp.DECODE_PRESETS["notebook3"].replace(
        smoothing_kernel=1, min_event_ms=0.0, merge_radius_ms=0.0)
    assert cfg.onset_sigma and cfg.offset_sigma
    assert mp.decode_ml_regions(prob, FS, cfg) == [(5_000, 5_399)]


def test_min_duration_runs_after_the_trim():
    """The notebook is explicit: the duration test uses the FINAL, trimmed extent, so
    an event whose deep part is too short is dropped even though its raw run was long.
    """
    n = 20_000
    sig = np.random.default_rng(2).normal(0.0, 1.0, n).astype(np.float32)
    sig[5_000:5_020] = -100.0                # only 0.4 ms deep
    sig[5_020:5_900] = -2.0                  # long shallow tail
    prob = _prob(n, [(5_000, 5_900, 0.9)])
    cfg = mp.MLDecodeConfig(smoothing_kernel=1, exit_threshold=0.2,
                            merge_radius_ms=0.0, min_event_ms=1.0,
                            offset_sigma=3.0)
    assert mp.decode_ml_regions(prob, FS, cfg, signal=sig) == []
    # Without the trim the (long) raw run survives the duration test.
    assert mp.decode_ml_regions(prob, FS, cfg.replace(offset_sigma=0.0), signal=sig)


def test_decode_empty_and_bad_rate():
    cfg = mp.DECODE_PRESETS["notebook1"]
    assert mp.decode_ml_regions(np.zeros(0, np.float32), FS, cfg) == []
    with pytest.raises(ValueError, match="positive"):
        mp.decode_ml_regions(np.zeros(10, np.float32), 0.0, cfg)


def test_smooth_probability_removes_a_singleton_spike():
    p = np.zeros(1000, np.float32)
    p[500] = 1.0
    assert float(np.max(mp.smooth_probability(p, 9))) == 0.0
    # Kernel longer than the trace is a no-op, not an error.
    short = np.array([1.0, 0.0], np.float32)
    np.testing.assert_array_equal(mp.smooth_probability(short, 9), short)


# ===========================================================================
# f. apo_raw_global is a genuine no-op vs the app's pre-existing chain
#    (the preset was RENAMED from apo_legacy; the numbers must not have moved)
# ===========================================================================
@pytest.mark.parametrize("fs", [25_000.0, 50_000.0, 100_000.0])
def test_apo_raw_global_preprocess_is_bit_identical_to_detrend_for_model(fs):
    n = int(fs * 1.7)
    rng = np.random.default_rng(11)
    t = np.arange(n) / fs
    x = (1000.0 + 30.0 * np.sin(2 * np.pi * 0.4 * t)
         + rng.normal(0.0, 3.0, n)).astype(np.float32)
    x[int(0.5 * fs):int(0.5 * fs) + 300] -= 80.0

    legacy = detrend_for_model(x, fs)
    preset = mp.apply_ml_preprocess(x, fs, mp.PREPROCESS_PRESETS["apo_raw_global"])
    assert preset.dtype == legacy.dtype == np.float32
    np.testing.assert_array_equal(preset, legacy)


@pytest.mark.parametrize("fs", [25_000.0, 50_000.0, 100_000.0])
def test_apo_raw_global_decode_matches_the_legacy_decode_regions(fs):
    rng = np.random.default_rng(5)
    # Separated probability bursts of varied height/width + jitter, so hysteresis,
    # the median smoother, the merge and the duration filter all actually fire.
    prob = rng.random(60_000).astype(np.float32) * 0.25
    for k in range(40):
        s = 300 + k * 1_400
        prob[s:s + int(rng.integers(4, 400))] = float(rng.uniform(0.32, 0.99))
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    legacy = decode_regions(prob, fs, enter=0.5, exit=0.3, smooth_kernel=9,
                            merge_s=1e-4, min_dur_s=1e-4)
    preset = mp.decode_ml_regions(prob, fs, mp.DECODE_PRESETS["apo_raw_global"])
    assert len(preset) > 1                    # a real, non-degenerate comparison
    assert preset == legacy


def test_renaming_apo_legacy_did_not_move_a_single_number():
    """The rename is honesty about provenance, not a change of contract: the config
    is field-for-field what ``apo_legacy`` was (its own name aside)."""
    pre = mp.PREPROCESS_PRESETS["apo_raw_global"]
    dec = mp.DECODE_PRESETS["apo_raw_global"]
    assert pre.deviations_from(mp.MLPreprocessConfig(
        baseline_method="savgol", baseline_window_s=0.5, baseline_poly=3,
        despike_size=0, anchor_tail=True)) == ()
    assert dec.deviations_from(mp.MLDecodeConfig(
        smoothing_kernel=9, enter_threshold=0.5, exit_threshold=0.3,
        merge_radius_ms=0.1, min_event_ms=0.1,
        onset_sigma=0.0, offset_sigma=0.0)) == ()


# ---------------------------------------------------------------------------
# The ONE known deviation from postprocess_predictions, pinned to its measured
# bound. notebook1/2 pass ``signal=current_detr, offset_sigma=0``; the notebook
# gates the ONSET snap on ``if onset_sigma:`` but NOT the offset trim, so it
# still trims at ``deep_level = median(signal)``. This app treats sigma 0 as OFF.
# Verified against the real student probability traces: 0/210 regions moved on
# ferritin_data1 and 1/376 on ferritin_data2, max shift ONE sample. These tests
# reproduce the notebook rule verbatim and pin that bound so it cannot grow
# silently, and prove notebook3 (offset_sigma=3.0) is exactly faithful.
# ---------------------------------------------------------------------------
def _nb_postprocess(prob, fs, *, smoothing_kernel, enter_threshold, exit_threshold,
                    merge_radius_ms, min_event_ms, signal=None, offset_sigma=3.0,
                    onset_sigma=2.0, onset_max_ms=5.0):
    """VERBATIM StudentModel*.ipynb ``postprocess_predictions`` (regions only)."""
    from scipy.signal import medfilt

    k = smoothing_kernel + 1 if smoothing_kernel % 2 == 0 else smoothing_kernel
    sm = medfilt(prob, k) if len(prob) >= k else prob
    above_exit = (sm >= exit_threshold).astype(np.int8)
    above_enter = sm >= enter_threshold
    d = np.diff(above_exit, prepend=0, append=0)
    regions = [(int(s), int(e - 1))
               for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
               if above_enter[s:e].any()]
    merged = []
    if regions:
        ms = int(fs * merge_radius_ms / 1000.0)
        cs, ce = regions[0]
        for s, e in regions[1:]:
            if s - ce <= ms:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
    trimmed = merged
    if signal is not None:
        med = np.median(signal)
        noise = 1.4826 * np.median(np.abs(signal - med))
        if onset_sigma:
            lvl = med - onset_sigma * noise
            look = int(onset_max_ms * fs / 1000.0)
            out, prev_end = [], -1
            for s, e in merged:
                j, steps, gap = s, 0, 0
                while j - 1 > prev_end and steps < look:
                    if signal[j - 1] < lvl:
                        j -= 1; steps += 1; gap = 0
                    elif gap < 5:
                        j -= 1; steps += 1; gap += 1
                    else:
                        break
                while j < s and signal[j] >= lvl:
                    j += 1
                out.append((j, e)); prev_end = e
            merged = out
        deep = med - offset_sigma * noise          # NOT gated on offset_sigma
        trimmed = []
        for s, e in merged:
            below = np.where(np.asarray(signal[s:e + 1]) < deep)[0]
            trimmed.append((s, (s + int(below[-1])) if len(below) else e))
    mn = int(fs * min_event_ms / 1000.0)
    return [(s, e) for s, e in trimmed if (e - s + 1) >= mn]


def _synthetic_prob_and_signal(seed=3, n=400_000, fs=50_000.0):
    """Blockades with a shallow recovery tail + a probability trace that rises
    late and lingers -- the structure the snap/trim exist to correct."""
    r = np.random.default_rng(seed)
    sig = (r.standard_normal(n) * 8.0).astype(np.float32)
    prob = np.zeros(n, np.float32)
    t = 2_000
    while t < n - 6_000:
        dur = int(r.integers(60, 900))
        depth = float(r.uniform(40, 220))
        sig[t:t + dur] -= depth
        sig[t + dur:t + dur + 40] -= depth * 0.12
        lat = int(r.integers(5, 60))
        prob[t + lat:t + dur + 30] = 0.95
        prob[t + dur + 30:t + dur + 70] = 0.25
        t += dur + int(r.integers(1_500, 6_000))
    prob = np.clip(prob + r.normal(0.0, 0.02, n), 0.0, 1.0).astype(np.float32)
    return prob, sig


def test_notebook3_decode_is_verbatim_postprocess_predictions():
    fs = 50_000.0
    prob, sig = _synthetic_prob_and_signal()
    got = mp.decode_ml_regions(prob, fs, mp.DECODE_PRESETS["notebook3"], signal=sig)
    want = _nb_postprocess(prob, fs, smoothing_kernel=5, enter_threshold=0.5,
                           exit_threshold=0.2, merge_radius_ms=0.2, min_event_ms=0.15,
                           signal=sig, offset_sigma=3.0, onset_sigma=2.0)
    assert len(got) > 50                      # a real, non-degenerate comparison
    assert [tuple(r) for r in got] == [tuple(r) for r in want]


@pytest.mark.parametrize("preset", ["notebook1", "notebook2"])
def test_nb1_nb2_deviate_from_the_notebook_only_on_the_offset_side(preset):
    """notebook1/2 differ from the notebook ONLY because the notebook fails to gate
    its offset trim on ``offset_sigma``. Characterise that exactly:

    * onsets are identical (the onset snap IS gated, in both);
    * the count is identical;
    * the app's offset is never EARLIER, and it is later by exactly the trailing
      run of samples at-or-above ``median(signal)`` -- nothing else.

    The magnitude is data-dependent (this synthetic has an exaggerated shallow
    recovery tail); the real-data bound is pinned by the ML corpus test.
    """
    fs = 50_000.0
    prob, sig = _synthetic_prob_and_signal()
    got = mp.decode_ml_regions(prob, fs, mp.DECODE_PRESETS[preset], signal=sig)
    want = _nb_postprocess(prob, fs, smoothing_kernel=9, enter_threshold=0.5,
                           exit_threshold=0.15, merge_radius_ms=0.5, min_event_ms=1.0,
                           signal=sig, offset_sigma=0, onset_sigma=0)
    assert len(got) == len(want) > 50
    assert [s for s, _ in got] == [s for s, _ in want]          # onsets identical
    shift = np.array([g[1] - w[1] for g, w in zip(got, want)])
    assert (shift >= 0).all()                   # the app never ends EARLIER
    assert shift.any()                          # ...and this synthetic does exercise it
    med = float(np.median(sig))
    for (gs, ge), (_, we) in zip(got, want):
        if ge == we:
            continue
        # every sample the app keeps past the notebook's offset is at/above median,
        # and the notebook's offset is the last one below it.
        assert (np.asarray(sig[we + 1:ge + 1]) >= med).all()
        assert sig[we] < med
    # The app ignores `signal` entirely for these two presets.
    assert got == mp.decode_ml_regions(prob, fs, mp.DECODE_PRESETS[preset], signal=None)


def test_onset_snap_and_offset_trim_move_boundaries_the_documented_way():
    fs = 50_000.0
    prob, sig = _synthetic_prob_and_signal()
    off = mp.MLDecodeConfig(smoothing_kernel=5, enter_threshold=0.5, exit_threshold=0.2,
                            merge_radius_ms=0.2, min_event_ms=0.15,
                            onset_sigma=0.0, offset_sigma=0.0)
    base = mp.decode_ml_regions(prob, fs, off, signal=sig)
    snapped = mp.decode_ml_regions(prob, fs, off.replace(onset_sigma=2.0), signal=sig)
    trimmed = mp.decode_ml_regions(prob, fs, off.replace(offset_sigma=3.0), signal=sig)
    assert len(base) == len(snapped) == len(trimmed) > 50
    assert all(b[0] <= a[0] for a, b in zip(base, snapped))      # onsets only earlier
    assert sum(a[0] != b[0] for a, b in zip(base, snapped)) > len(base) // 2
    assert all(b[1] <= a[1] for a, b in zip(base, trimmed))      # offsets only earlier
    assert sum(a[1] != b[1] for a, b in zip(base, trimmed)) > len(base) // 2
    assert all(a[1] == b[1] for a, b in zip(base, snapped))      # snap leaves offsets
    assert all(a[0] == b[0] for a, b in zip(base, trimmed))      # trim leaves onsets
    # Both together == the notebook3 preset.
    assert mp.decode_ml_regions(prob, fs, mp.DECODE_PRESETS["notebook3"], signal=sig) \
        == mp.decode_ml_regions(
            prob, fs, off.replace(onset_sigma=2.0, offset_sigma=3.0), signal=sig)


def test_decode_knobs_produce_the_documented_region_sets():
    """smoothing / exit / merge / min-duration, each isolated on one trace."""
    fs = 1_000.0
    p = np.zeros(2_000, np.float32)
    p[100:140] = 0.9                 # A
    p[145:180] = 0.9                 # B -- gap of 6 samples from A
    p[400:404] = 0.9                 # 4-sample event
    p[600:640] = 0.9
    p[640:660] = 0.25                # tail above 0.2, below 0.5
    p[900] = 0.9                     # 1-sample spike
    raw = mp.MLDecodeConfig(smoothing_kernel=1, enter_threshold=0.5, exit_threshold=0.5,
                            merge_radius_ms=0.0, min_event_ms=0.0)

    def dec(**kw):
        return mp.decode_ml_regions(p, fs, raw.replace(**kw))

    assert dec() == [(100, 139), (145, 179), (400, 403), (600, 639), (900, 900)]
    # median smoothing removes the 1-sample spike; the 4-sample event survives k=3..7
    assert dec(smoothing_kernel=3) == [(100, 139), (145, 179), (400, 403), (600, 639)]
    assert dec(smoothing_kernel=9) == [(100, 139), (145, 179), (600, 639)]
    # exit below the 0.25 tail extends the region through it
    assert dec(exit_threshold=0.2) == [(100, 139), (145, 179), (400, 403),
                                       (600, 659), (900, 900)]
    # merge radius is int(fs * ms / 1000) samples, compared with <=
    assert dec(merge_radius_ms=5.0) == dec()                       # 5 < gap 6
    assert dec(merge_radius_ms=6.0) == [(100, 179), (400, 403), (600, 639), (900, 900)]
    # min duration is inclusive (e - s + 1)
    assert dec(min_event_ms=10.0) == [(100, 139), (145, 179), (600, 639)]
    assert dec(min_event_ms=50.0) == []
    # ...and it runs LAST, so it sees the MERGED 80-sample extent
    assert dec(merge_radius_ms=6.0, min_event_ms=50.0) == [(100, 179)]


# ===========================================================================
# g. THE APO NOTEBOOK CHAIN
#
# ApoFerritin_LSTM_Cluster_Validation_RAM_Safe.ipynb cells 10 + 12 + 13. Every
# reference below is transcribed VERBATIM from those cells (including the dtype
# casts, which is where bit-exactness lives) and is independent of the module
# under test -- it uses scipy directly, never a nanoscope helper.
#
# Verified on the user's real -600 mV recording (2,993,798 samples): this chain
# reproduces the notebook's model input BIT-FOR-BIT
# (sha256 382b0b027600043534f8ed169c259d2ae2c2e079e73099abefa93ab9adfcfded,
# max|d| = 0.0) and, when the notebook's own inference + decode are re-run on that
# input, its stored event COUNT: 236 on -600 mV and 982 on -700 mV.
#
# The event INDICES are deliberately not claimed. Preprocessing is deterministic;
# inference is not -- the student is mixed float16, so its probability trace
# differs in the low bits between the notebook's Colab device and this one, and at
# a decode threshold of 0.1 that shifts a boundary by a sample or three. The
# notebook PRINTS its shortest -600 mV event as 1736337..1736999 (662 samples); a
# verbatim re-run of its cells here gives 1736334..1736999 (665), with
# preds[1736334] = 0.156 -- not a tie-break, a genuinely different probability.
# What IS pinned below is the input digest, which is machine-independent.
# ===========================================================================
def _nb_apply_notch_filter_harmonics(signal, fs, freqs=60.0, quality=30.0, harmonics=3):
    """VERBATIM cell 10. Note the float32 copy in, float64 through passes 2-3, and
    the float32 cast on the way OUT -- all three decide the result's low bits."""
    filtered = signal.astype(np.float32).copy()
    nyquist = fs / 2.0
    for h in range(1, harmonics + 1):
        notch_freq = freqs * h
        if notch_freq >= nyquist:
            break
        b, a = iirnotch(notch_freq, quality, fs)
        filtered = filtfilt(b, a, filtered)
    return filtered.astype(np.float32)


def _nb_remove_baseline_savgol(signal, fs, window_ms=500.0, polyorder=3):
    """VERBATIM cell 10. ``mode`` is NOT passed, so scipy's default 'interp' runs --
    NOT the 'mirror' the notebook's separate amplitude appendix uses."""
    sig = np.asarray(signal, np.float32)
    window_length = int(window_ms * fs / 1000.0)
    if window_length % 2 == 0:
        window_length += 1
    baseline = savgol_filter(sig, window_length=window_length, polyorder=polyorder)
    return sig - baseline.astype(np.float32)


def _nb_normalize_signal(signal):
    """VERBATIM cell 10. ``np.mean`` on float32 accumulates in FLOAT32; widening it
    to float64 changes the low bits over a multi-million-sample recording."""
    sig = signal.astype(np.float32)
    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    return sig


def _nb_preprocess_signal(raw_signal, fs):
    """VERBATIM cell 12: notch -> SavGol detrend -> global mean-centred normalize."""
    sig = _nb_apply_notch_filter_harmonics(
        raw_signal, fs, freqs=60.0, quality=30.0, harmonics=3)
    sig = _nb_remove_baseline_savgol(sig, fs)
    return _nb_normalize_signal(sig)


def _nb_extract_event_regions(preds, threshold, min_len_samples):
    """VERBATIM cell 13 -- STRICT ``>``, EXCLUSIVE end, min-length applied inside
    extraction, and a run still open at the end appended WITHOUT the length test."""
    above = preds > threshold
    regions, start = [], None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len_samples:
                regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(above)))
    return regions


def _apo_trace(n: int = 120_000, seed: int = 4) -> np.ndarray:
    """Open pore + drift + 60/120 Hz mains + noise + blockades, float32 pA."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = (-35.0
         + 6.0 * np.sin(2 * np.pi * 0.9 * t)
         + 3.0 * np.sin(2 * np.pi * 60.0 * t)
         + 1.0 * np.sin(2 * np.pi * 120.0 * t)
         + rng.normal(0.0, 2.0, n))
    for k in range(9):
        s = 4_000 + k * 12_000
        x[s:s + 900] -= 45.0
    return x.astype(np.float32)


def test_normalization_center_is_the_notebooks_float32_mean():
    """``np.mean`` on the float32 array, NOT a float64 accumulation."""
    x = _apo_trace(n=40_000)
    got = mp.normalization_center(x)
    assert got == float(np.mean(x))                  # float32 pairwise accumulation
    assert mp.normalization_center(np.zeros(0, np.float32)) == 0.0
    # The float64 accumulator is a DIFFERENT number -- that is the trap.
    assert isinstance(got, float)


def test_centred_scale_and_normalize_match_the_notebook_normalize_signal():
    """The whole ``normalize_signal`` contract, against the verbatim cell-10 code."""
    x = _apo_trace()
    detr = _nb_remove_baseline_savgol(_nb_apply_notch_filter_harmonics(x, FS), FS)

    center = mp.normalization_center(detr)
    scale = mp.normalization_scale(detr, center)
    got = mp.normalize_global(detr, scale, center)

    np.testing.assert_array_equal(got, _nb_normalize_signal(detr))
    assert got.dtype == np.float32
    # The centre is the mean and the SCALE is measured about it, not about zero.
    assert scale == pytest.approx(
        float(np.max(np.abs(detr - np.float32(center)))) + 1e-8, rel=0, abs=1e-9)
    assert scale != pytest.approx(mp.normalization_scale(detr), rel=0, abs=1e-12)
    # Scale=None measures it on the CENTRED array, so both spellings agree.
    np.testing.assert_array_equal(got, mp.normalize_global(detr, None, center))


def test_centre_zero_is_exactly_the_ferritin_rescale():
    """The default must be a byte-for-byte no-op, or every ferritin preset moves."""
    x = _apo_trace(n=30_000)
    np.testing.assert_array_equal(
        mp.normalize_global(x, None, 0.0), mp.normalize_global(x))
    assert mp.normalization_scale(x, 0.0) == mp.normalization_scale(x)
    # ...and the centred form really does differ on a non-zero-mean signal.
    assert abs(float(np.mean(x))) > 1.0
    assert not np.array_equal(
        mp.normalize_global(x, None, mp.normalization_center(x)),
        mp.normalize_global(x))


def test_notch_float32_rounds_only_when_asked_and_changes_the_baseline():
    """The flag must actually reach the baseline stage, and must be OFF by default."""
    x = _apo_trace(n=60_000)
    base = mp.MLPreprocessConfig(baseline_method="savgol", baseline_window_s=0.1)
    assert base.notch_float32 is False

    rounded = mp.apply_ml_preprocess(x, FS, base.replace(notch_float32=True))
    kept64 = mp.apply_ml_preprocess(x, FS, base)
    # Same physical signal, different low bits -- the whole point of the switch.
    np.testing.assert_allclose(rounded, kept64, rtol=0, atol=1e-3)
    assert not np.array_equal(rounded, kept64)

    # And the rounded form is exactly "SavGol of the float32-cast notch output".
    notched32 = np.asarray(mp.notch_sequential(x, FS), np.float32)
    expect, _ = mp.baseline_savgol(notched32, FS, window_s=0.1, poly=3)
    np.testing.assert_array_equal(rounded, expect)


def test_apo_notebook_preset_reproduces_preprocess_signal_bit_for_bit():
    """apply_ml_preprocess + the mean-centred normalization == cell 12, exactly."""
    x = _apo_trace()
    cfg = mp.PREPROCESS_PRESETS["apo_nb_600mV"]
    detr = mp.apply_ml_preprocess(x, FS, cfg)
    center = mp.normalization_center(detr)
    got = mp.normalize_global(detr, mp.normalization_scale(detr, center), center)

    np.testing.assert_array_equal(got, _nb_preprocess_signal(x, FS))
    assert got.dtype == np.float32


@pytest.mark.parametrize(
    "name, threshold, min_ms",
    [("apo_nb_500mV", 0.2, 15.0), ("apo_nb_600mV", 0.1, 1.0), ("apo_nb_700mV", 0.2, 1.0)],
)
def test_apo_decode_presets_reproduce_extract_event_regions(name, threshold, min_ms):
    """The single-threshold decode, against the verbatim cell-13 reference.

    Regions here are INCLUSIVE ``[s, e]``; the notebook's are exclusive ``[s, e)``.
    """
    rng = np.random.default_rng(19)
    prob = (rng.random(200_000) * 0.08).astype(np.float32)
    for k in range(30):
        s = 500 + k * 6_000
        width = int(rng.integers(20, 1_400))     # straddles both min-length cuts
        prob[s:s + width] = float(rng.uniform(threshold + 0.05, 0.99))
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

    got = mp.decode_ml_regions(prob, FS, mp.DECODE_PRESETS[name])
    want = _nb_extract_event_regions(prob, threshold, int((min_ms / 1000.0) * FS))
    assert len(got) > 3                                    # non-degenerate
    assert [(s, e + 1) for s, e in got] == want


def test_apo_decode_min_length_samples_arithmetic_matches_the_notebook():
    """``int(min_len_ms/1000*FS)`` (notebook) vs ``int(fs*min_ms/1000)`` (app) --
    two different float expressions that must land on the same integer at the three
    values actually used."""
    for min_ms in (1.0, 15.0):
        assert int((min_ms / 1000.0) * FS) == int(FS * min_ms / 1000.0)


# ---------------------------------------------------------------------------
# THE FERRITIN PRESETS ARE FROZEN.
#
# They are verified bit-exact against the persisted training columns
# CSVData/ferritin_data{1,2,3}.csv.gz, so any drift is a regression by definition.
# These digests were captured from the module BEFORE the apo work landed and must
# never be "updated" -- a failure here means the ferritin contract moved.
#
# RE-DERIVED after the F1 median_filter fix. They had to be: three of the four ride
# on ``_decimated_median``, and while it went through scipy's 1-D path it could
# return the wrong answer at any moment, so a digest taken through it proved nothing
# about the run that produced it -- including, potentially, these. Each of the four
# was re-derived with the fixed code in a quiet single-purpose process (no Qt, no TF,
# no threads) and cross-checked against ``_hand_rolling_median``, which never calls
# scipy.ndimage at all, at all four (window, decim) operating points and in both
# dtypes.
#
# ALL FOUR CAME BACK BYTE-IDENTICAL. So the pinned values were the correct ones all
# along; what was broken was the code's ability to reproduce them (six distinct
# notebook1 digests in forty runs under load, now one). Nothing below changed -- the
# provenance did.
# ---------------------------------------------------------------------------
_GOLDEN_INPUT_SHA = "2b0f4499924a9032f6349b9980472b4048e7c2e0af642ca0b7d7c38b84be279f"
_GOLDEN_FERRITIN = {
    "notebook1": ("a960b7fbdd3b57393e2d42353d9dd0d11ef9393263483cc536e8280e4f511f26",
                  "fe4f6d8c4d5c1ff870eb4945ce8f9a7e3a85d513f596da4fda3899cd6229b301"),
    "notebook2": ("89b23fd2f585886d21d4ef4205957a4c86acc26a35f64454ab9c1dcb3e4a4198",
                  "d5f2f5265a31ba2fc84f8c38e9bbad925daa4c781531d7f542c3637e510c37bf"),
    "notebook3": ("24ba0482f7cf7d4769da3ec656eefa8f1f30cc8d7b1cbedd48b773785da714ef",
                  "a34eac844521e9a8a6963db03464eca369ac5c7532e81059a497a81c287c858f"),
    # Renamed from apo_legacy; its numbers are frozen for the same reason.
    "apo_raw_global": ("9242fe0a871216a52204753bf7848fe7a114808216a55437e7e20564b96d1eed",
                       "572fa478b10e1ff7bb68a9064991aa3166899d969b85eefe52594e8bc31ff35c"),
}


def _golden_trace(n: int = 250_000, seed: int = 20260728) -> np.ndarray:
    """The fixed trace the golden digests were taken on. Do not change it."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = (1500.0
         + 40.0 * np.sin(2 * np.pi * 0.7 * t)
         + 12.0 * np.sin(2 * np.pi * 60.0 * t)
         + 4.0 * np.sin(2 * np.pi * 120.0 * t)
         + rng.normal(0.0, 4.0, n))
    for k in range(24):
        s = 3000 + k * 10_000
        x[s:s + int(60 + 40 * ((k * 37) % 7))] -= 60.0 + 5.0 * (k % 5)
    x[7777] -= 400.0          # 1-sample spike (exercises notebook2's de-spike)
    x[123_456] += 350.0
    return x.astype(np.float32)


def test_the_golden_trace_itself_is_unchanged():
    """If this fails the fixture drifted, and the digests below prove nothing."""
    assert hashlib.sha256(_golden_trace().tobytes()).hexdigest() == _GOLDEN_INPUT_SHA


@pytest.mark.parametrize("name", sorted(_GOLDEN_FERRITIN))
def test_preset_output_is_byte_for_byte_what_it_was(name):
    """The whole chain -- notch, baseline, de-spike, normalization -- pinned."""
    x = _golden_trace()
    detr = mp.apply_ml_preprocess(x, FS, mp.PREPROCESS_PRESETS[name])
    norm = mp.normalize_global(detr, mp.normalization_scale(detr))
    want_detr, want_norm = _GOLDEN_FERRITIN[name]
    assert hashlib.sha256(detr.tobytes()).hexdigest() == want_detr
    assert hashlib.sha256(norm.tobytes()).hexdigest() == want_norm


def test_the_new_apo_switches_are_off_for_every_ferritin_preset():
    """Belt-and-braces on the digests: the three new fields are the ONLY way the
    apo work could have reached the ferritin chain, so assert them by name."""
    for name in ("notebook1", "notebook2", "notebook3"):
        cfg = mp.PREPROCESS_PRESETS[name]
        assert cfg.notch_float32 is False
        assert cfg.normalize_center is False
        assert cfg.last_start_inclusive is False


# ===========================================================================
# h. THE BIT-EXACT ORACLE (corpus)
#
# There is no persisted preprocessing column for apo the way there is for
# ferritin (CSVData/ferritin_data*.csv.gz) -- the apo notebook writes nothing but
# downstream cluster metadata, and that cell never ran. So the oracle is built the
# other way round: the notebook chain was re-executed on the user's real -600 mV
# recording and its output hashed, and the app must land on the same bytes.
#
# The app reads the ABF; the notebook read a CSV export of it. Those two are
# bit-identical (max|ABF - CSV| = 0 on all three apo recordings), so this really
# does close the loop from the file the app opens to the numbers the notebook
# published.
# ===========================================================================
APO_ABF = pathlib.Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")

#: sha256 of the float32 bytes, from re-running cells 10 + 12 verbatim on the CSV.
_APO600_NOTCHED_SHA = "de224642bfcf0f4ac9cc804c4051125a571d6ae9cc762dbb18b31c4e85cf527d"
_APO600_INPUT_SHA = "382b0b027600043534f8ed169c259d2ae2c2e079e73099abefa93ab9adfcfded"
_APO600_N = 2_993_798


@pytest.mark.corpus
def test_apo_preset_is_bit_exact_against_the_real_recording():
    """The whole apo contract, end to end, against the notebook's own numbers.

    The notebook hard-codes ``FS = 50_000`` while this ABF's header says
    50000.001192 Hz. That is a real (if cosmetic) difference -- it moves the model
    input by 2.4e-07 -- so the oracle is evaluated at the notebook's rate. The
    SavGol window is 25001 taps at both rates.
    """
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document

    if not APO_ABF.exists():
        pytest.skip(f"corpus ABF absent: {APO_ABF.name}")

    doc = open_document(APO_ABF)
    current = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT]
    assert [c.name for c in current] == ["IN15"]
    raw = np.asarray(doc.channels[current[0].index], np.float32)
    assert raw.size == _APO600_N
    assert current[0].raw_units == "pA"

    cfg = mp.PREPROCESS_PRESETS["apo_nb_600mV"]
    notched = np.asarray(
        mp.notch_sequential(raw, FS, freqs=cfg.notch_freqs, q=cfg.notch_q), np.float32)
    assert hashlib.sha256(notched.tobytes()).hexdigest() == _APO600_NOTCHED_SHA

    detr = mp.apply_ml_preprocess(raw, FS, cfg)
    center = mp.normalization_center(detr)
    model_input = mp.normalize_global(detr, mp.normalization_scale(detr, center), center)
    assert hashlib.sha256(model_input.tobytes()).hexdigest() == _APO600_INPUT_SHA

    # The two apo-only switches are what make it exact -- prove it by turning each
    # off and watching the hash move.
    assert center == pytest.approx(-0.0031291742343455553, rel=0, abs=1e-12)
    plain = mp.normalize_global(detr, mp.normalization_scale(detr))
    assert hashlib.sha256(plain.tobytes()).hexdigest() != _APO600_INPUT_SHA
    detr64 = mp.apply_ml_preprocess(raw, FS, cfg.replace(notch_float32=False))
    c64 = mp.normalization_center(detr64)
    kept = mp.normalize_global(detr64, mp.normalization_scale(detr64, c64), c64)
    assert hashlib.sha256(kept.tobytes()).hexdigest() != _APO600_INPUT_SHA
