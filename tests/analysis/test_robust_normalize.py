"""`robust_normalize` -- the MODEL-INPUT rescale, pinned against the notebooks.

Six of the owner's notebooks (``Cu2Insulin_``/``Zn2Insulin_``/host-guest
``StudentModel`` and ``StudentInference``) define this function identically, and it
REPLACED ``x / max|x|`` for a stated, measured reason: that divisor is a single
extreme sample, it ranged 8.8-42.6 pA across his recordings, and ordinary noise
therefore reached the model at 0.049 in the training domain but 0.146 for the host
recording -- host noise arriving at the amplitude the model was taught to call an
event.

The load-bearing test here is :func:`test_bit_for_bit_against_the_notebook_function`,
which pins the port against a LITERAL TRANSCRIPTION of the notebook cell held below,
not against the port's own behaviour. Everything else states one claim from the
docstring in a form that can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.filters import (
    MODEL_INPUT_CLIP_SIGMA,
    MODEL_INPUT_MODE,
    NORMALIZE_MODES,
    NORMALIZE_MODES_ALL,
    normalize,
    robust_normalize,
)


# ---------------------------------------------------------------------------
# THE REFERENCE. Copied verbatim from Cu2Insulin_StudentInference.ipynb (identical
# in Cu2Insulin_StudentModel.ipynb and in the Zn / host-guest pairs). Do not tidy it,
# do not vectorise it, do not "fix" the float32: this is the thing being matched.
# ---------------------------------------------------------------------------
def _notebook_robust_normalize(signal, clip_sigma=30.0):
    x  = np.asarray(signal, np.float32)
    m0 = float(np.median(x)); s0 = 1.4826 * float(np.median(np.abs(x - m0)))
    k  = np.abs(x - m0) < 5 * max(s0, 1e-12)
    m1 = float(np.median(x[k])) if k.any() else m0
    s1 = 1.4826 * float(np.median(np.abs(x[k] - m1))) if k.any() else s0
    q  = np.abs(x - m1) < 3 * max(s1, 1e-12)
    med = float(np.median(x[q])) if q.sum() > 100 else m1
    mad = 1.4826 * float(np.median(np.abs(x[q] - med))) if q.sum() > 100 else s1
    if not np.isfinite(mad) or mad <= 0:
        mad = s0 if s0 > 0 else 1.0
    z = (x - med) / mad
    return (np.clip(z, -clip_sigma, clip_sigma) / clip_sigma).astype(np.float32)


FS = 50_000.0


def _recording(n=200_000, *, i0=-125.0, noise=2.2, depth=78.0, rate_hz=30.0,
               dwell_ms=0.24, seed=0, spike=None):
    """A Cu-shaped conductive recording in pA: negative open pore, deep rare events."""
    rng = np.random.default_rng(seed)
    x = i0 + rng.normal(0.0, noise, n)
    step = max(1, int(FS / rate_hz))
    width = max(1, int(dwell_ms * 1e-3 * FS))
    for i in range(step, n - width, step):
        x[i: i + width] -= depth
    if spike is not None:
        x[n // 3] = spike
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# the contract that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("clip_sigma", [30.0, 10.0, 5.0])
def test_bit_for_bit_against_the_notebook_function(seed, clip_sigma):
    """Not "close to". THE SAME BYTES. The model was trained on the notebook's output,
    so a port that agrees to 1e-7 is a port that feeds the model something it has
    never seen -- and the difference is invisible in every summary statistic."""
    x = _recording(seed=seed)
    want = _notebook_robust_normalize(x, clip_sigma=clip_sigma)
    got = robust_normalize(x, clip_sigma=clip_sigma)
    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_bit_for_bit_on_the_shapes_that_exercise_the_fallbacks():
    """The branches a smooth synthetic never reaches: a flat trace (mad == 0, the
    ``s0``/1.0 fallback), a trace under the 100-sample guard, a two-valued trace whose
    5-sigma trim keeps everything, and one dominated by a single extreme sample."""
    cases = {
        "flat": np.full(5_000, -125.0, np.float32),
        "under_the_guard": _recording(n=100, rate_hz=1e-9),
        "at_the_guard": _recording(n=101, rate_hz=1e-9),
        "two_valued": np.repeat([-125.0, -203.0], 2_500).astype(np.float32),
        "one_huge_sample": _recording(n=20_000, spike=-9_000.0),
        "single_sample": np.asarray([-125.0], np.float32),
    }
    for name, x in cases.items():
        assert np.array_equal(
            robust_normalize(x), _notebook_robust_normalize(x)
        ), f"diverged on {name}"


def test_the_hundred_sample_guard_is_a_strict_greater_than():
    """``q.sum() > 100``, not ``>= 100``. At exactly 100 survivors the stage keeps the
    5-sigma pass's estimates; at 101 it re-estimates. Off by one here and a short
    inference window divides by the MAD of a handful of samples."""
    rng = np.random.default_rng(11)
    for n, refit in ((100, False), (101, True)):
        x = rng.normal(-125.0, 2.2, n).astype(np.float32)
        m0 = float(np.median(x))
        s0 = 1.4826 * float(np.median(np.abs(x - m0)))
        k = np.abs(x - m0) < 5 * max(s0, 1e-12)
        m1 = float(np.median(x[k]))
        s1 = 1.4826 * float(np.median(np.abs(x[k] - m1)))
        q = np.abs(x - m1) < 3 * max(s1, 1e-12)
        assert q.sum() == n, "this fixture is only meaningful when every sample survives"
        expect_mad = (1.4826 * float(np.median(np.abs(x[q] - float(np.median(x[q])))))
                      if refit else s1)
        got = np.asarray(robust_normalize(x), dtype=np.float64)
        # Recover the divisor the kernel used from a sample far from the centre.
        i = int(np.argmax(np.abs(x - m1)))
        centre = float(np.median(x[q])) if refit else m1
        assert got[i] * MODEL_INPUT_CLIP_SIGMA == pytest.approx(
            (float(x[i]) - centre) / expect_mad, rel=1e-5)


def test_the_noise_floor_lands_at_one_over_the_clip_in_every_recording():
    """The whole point. Three recordings whose open pore, noise and extreme sample all
    differ arrive at the model with the SAME noise amplitude, 1/clip_sigma."""
    for i0, noise, spike in ((-125.0, 2.2, -900.0), (-316.0, 1.83, -70.0),
                             (-181.0, 2.45, -3_000.0)):
        x = _recording(i0=i0, noise=noise, spike=spike)
        y = np.asarray(robust_normalize(x), dtype=np.float64)
        # Robust sigma of the OPEN-PORE part, in output units.
        med = np.median(y)
        sigma = 1.4826 * np.median(np.abs(y - med))
        assert sigma == pytest.approx(1.0 / MODEL_INPUT_CLIP_SIGMA, rel=0.05)


def test_it_fixes_the_transfer_failure_that_max_abs_normalisation_caused():
    """The docstring's headline claim, as a measurement. Two recordings with the SAME
    noise but different extreme samples: under ``x / max|x|`` their noise reaches the
    model at amplitudes ~3x apart -- so one file's noise arrives where the other file
    taught the model to see an event. Under this kernel they agree to a few percent."""
    a = _recording(seed=5, spike=-260.0)     # modest extreme
    b = _recording(seed=5, spike=-900.0)     # one very deep sample, same noise

    def _sigma(y):
        y = np.asarray(y, dtype=np.float64)
        return 1.4826 * float(np.median(np.abs(y - np.median(y))))

    max_abs = [_sigma(np.asarray(v, np.float64) / np.abs(np.asarray(v, np.float64)).max())
               for v in (a, b)]
    robust = [_sigma(robust_normalize(v)) for v in (a, b)]
    assert max(max_abs) / min(max_abs) > 3.0
    assert max(robust) / min(robust) < 1.05


def test_clip_then_divide_bounds_the_output_and_keeps_deep_events_distinguishable():
    """Order matters: clip to +-clip_sigma FIRST, divide by clip_sigma SECOND. The
    output is therefore exactly [-1, +1] and a 78 pA event on 2.2 pA noise (~35 sigma)
    saturates at -1.0 rather than reaching -1.18."""
    x = _recording(depth=78.0, noise=2.2)
    y = np.asarray(robust_normalize(x), dtype=np.float64)
    assert y.min() >= -1.0 and y.max() <= 1.0
    assert y.min() == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# it is deliberately NOT a mode of `normalize`
# ---------------------------------------------------------------------------


def test_it_is_not_a_normalize_mode_and_normalize_refuses_its_name():
    """Two opposite degenerate-input contracts cannot share one ``mode=`` switch, so
    the name is not accepted there and the dispatch is on `MODEL_INPUT_MODE`."""
    assert MODEL_INPUT_MODE not in NORMALIZE_MODES
    assert NORMALIZE_MODES_ALL == (*NORMALIZE_MODES, MODEL_INPUT_MODE)
    with pytest.raises(ValueError, match=r"mode must be one of"):
        normalize(np.arange(100, dtype=np.float32), mode=MODEL_INPUT_MODE)


def test_a_flat_trace_falls_back_instead_of_raising_the_way_normalize_does():
    """`normalize` raises on a constant trace on purpose. This one cannot: it sits in
    front of a model on a windowed inference path, where one flat window would abort
    the whole run. It returns zeros, which is the honest answer."""
    flat = np.full(4_000, -125.0, np.float32)
    with pytest.raises(ValueError, match=r"no scale to divide by"):
        normalize(flat, mode="robust")
    y = robust_normalize(flat)
    assert np.array_equal(y, np.zeros(4_000, np.float32))


# ---------------------------------------------------------------------------
# the module's standing input rules
# ---------------------------------------------------------------------------


def test_output_contract_matches_every_other_kernel_in_the_module():
    x = _recording(n=10_000)
    y = robust_normalize(x)
    assert y.dtype == np.float32
    assert y.shape == x.shape
    assert y.flags["C_CONTIGUOUS"]


def test_an_empty_trace_and_a_non_finite_sample_are_refused():
    with pytest.raises(ValueError, match=r"empty trace"):
        robust_normalize(np.asarray([], np.float32))
    bad = _recording(n=1_000)
    bad[17] = np.nan
    with pytest.raises(ValueError, match=r"NaN/inf"):
        robust_normalize(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_a_degenerate_clip_is_refused(bad):
    with pytest.raises(ValueError, match=r"clip_sigma must be positive"):
        robust_normalize(_recording(n=1_000), clip_sigma=bad)


def test_the_default_clip_is_the_notebooks_thirty():
    assert MODEL_INPUT_CLIP_SIGMA == 30.0
    x = _recording(n=20_000)
    assert np.array_equal(robust_normalize(x), robust_normalize(x, clip_sigma=30.0))
