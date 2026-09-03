"""Tests for the runtime-loaded Keras event detector (``nanoscope.analysis.ml_detect``).

Strategy:
  * The PURE pieces -- detrend, global-raw normalize, decode (hysteresis / merge /
    min-duration), and even the windowed overlap-average of ``predict_probability``
    (driven by a FAKE model) -- are tested WITHOUT TensorFlow. These always run.
  * The TF-dependent pieces (loading a real ``.keras``, forcing CPU) are guarded on
    ``TENSORFLOW_AVAILABLE`` and on the model file existing.
  * One ``@pytest.mark.corpus`` test mirrors the smoke script: IF the real model and
    a real ABF are present, load the model and run ``ml_detect_regions`` on a small
    slice, asserting it returns events with probabilities spanning 0..1.

The load-bearing claims:
  a. detrend FLATTENS a smooth trend (a ramp collapses to ~0) and returns float32;
  b. normalize gives max|.| == 1 and does NOT shift the mean (pure rescale);
  c. decode recovers KNOWN regions with correct hysteresis, merge, and min-duration;
  d. predict_probability overlap-averages correctly and honours progress/cancel;
  e. TF import is lazy and forces CPU; missing TF surfaces as MLUnavailable, no crash;
  f. the real model runs end to end on real data (corpus).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis.ml_detect import (
    TENSORFLOW_AVAILABLE,
    KerasEventModel,
    MLCancelled,
    MLUnavailable,
    create_multiscale_features_A,
    decode_regions,
    detrend_for_model,
    load_keras_model,
    ml_detect_regions,
    normalize_global_raw,
    predict_probability,
)

FS = 50_000.0
MODEL_PATH = Path("/Users/abrahamcampos/Downloads/student_raw_global_raw_best.keras")
ABF_PATH = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
# The 12-feature / teacher (U-Net) model. Absent here -> the corpus_teacher tests
# skip, exactly like the raw MODEL_PATH corpus tests.
#: The 12-feature teacher, wherever it actually lives. A single hardcoded path here
#: pointed at ``~/teacher_trained.keras`` while the file sits on the Desktop, so both
#: corpus_teacher tests skipped silently — and went on skipping straight through the
#: change that made ``load_keras_model`` REFUSE teachers, which they would otherwise have
#: caught. A skipped test is not a passing test.
_TEACHER_CANDIDATES = (
    Path("/Users/abrahamcampos/teacher_trained.keras"),
    Path("/Users/abrahamcampos/Desktop/teacher_trained.keras"),
    Path("/Users/abrahamcampos/Desktop/HGteacher_cb7ada_75mV.keras"),
)
TEACHER_PATH = next((p for p in _TEACHER_CANDIDATES if p.exists()), _TEACHER_CANDIDATES[0])

_HAVE_MODEL = TENSORFLOW_AVAILABLE and MODEL_PATH.exists()
_HAVE_TEACHER = TENSORFLOW_AVAILABLE and TEACHER_PATH.exists()


# ===========================================================================
# a. detrend flattens a smooth trend
# ===========================================================================
def test_detrend_flattens_a_ramp():
    n = int(FS)  # 1 s @ 50 kHz -> longer than the 500 ms SavGol window
    t = np.arange(n) / FS
    ramp = (100.0 + 37.0 * t).astype(np.float32)  # slow linear trend, big offset

    detr = detrend_for_model(ramp, FS)

    assert detr.dtype == np.float32
    assert detr.shape == ramp.shape
    assert np.isfinite(detr).all()
    # A linear ramp is exactly representable by the poly-3 SavGol fit, so the
    # subtraction removes essentially all of it: the residual spread collapses by
    # orders of magnitude and sits at ~0.
    assert detr.std() < ramp.std() / 1_000.0
    assert float(np.mean(detr)) == pytest.approx(0.0, abs=1e-2)
    assert np.max(np.abs(detr)) < 0.5


def test_detrend_removes_slow_sine_but_keeps_a_square_event():
    n = int(2 * FS)
    t = np.arange(n) / FS
    drift = 200.0 + 80.0 * np.sin(2 * np.pi * 0.3 * t)  # slow open-pore wander
    x = drift.copy()
    s, e = 40_000, 40_400  # 8 ms square blockade of depth 60
    x[s:e] += 60.0
    detr = detrend_for_model(x.astype(np.float32), FS)

    ev = np.zeros(n, bool)
    ev[s:e] = True
    # Open pore flattened to ~0; the event still stands ~60 above the flattened floor.
    assert detr[~ev].std() < 8.0
    depth = float(np.median(detr[s:e]) - np.median(detr[~ev]))
    assert depth == pytest.approx(60.0, rel=0.1)


def test_detrend_rejects_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        detrend_for_model(np.ones(1000, np.float32), 0.0)
    with pytest.raises(ValueError, match="NaN|Inf"):
        bad = np.ones(1000, np.float32)
        bad[10] = np.nan
        detrend_for_model(bad, FS)
    with pytest.raises(ValueError, match="too short"):
        detrend_for_model(np.ones(3, np.float32), FS)


# ===========================================================================
# b. global-raw normalize: max|.| == 1, NO mean subtraction
# ===========================================================================
def test_normalize_global_raw_maxabs_one_and_no_mean_shift():
    sig = np.array([-2.0, 1.0, 4.0, 6.0, 3.0], dtype=np.float32)  # mean = 2.4, max|.| = 6
    norm = normalize_global_raw(sig)

    assert norm.dtype == np.float32
    # max absolute value is exactly 1 (divided by max-abs + tiny eps).
    assert np.max(np.abs(norm)) == pytest.approx(1.0, abs=1e-6)
    # It is a PURE rescale: no mean subtraction. mean(norm) == mean(sig)/scale != 0.
    scale = float(np.max(np.abs(sig)) + 1e-8)
    assert float(np.mean(norm)) == pytest.approx(float(np.mean(sig)) / scale, abs=1e-6)
    assert float(np.mean(norm)) > 0.3  # clearly not re-centred to ~0
    # Element-wise it is exactly signal/scale.
    np.testing.assert_allclose(norm * scale, sig, rtol=0, atol=1e-5)


def test_normalize_global_raw_all_zero_is_safe():
    norm = normalize_global_raw(np.zeros(100, np.float32))
    assert norm.dtype == np.float32
    assert np.all(norm == 0.0)  # 0 / (0 + 1e-8) == 0, no divide-by-zero


# ===========================================================================
# c. decode recovers known regions, with hysteresis / merge / min-duration
# ===========================================================================
def _prob_with_regions(n: int, spans: list[tuple[int, int, float]]) -> np.ndarray:
    p = np.zeros(n, np.float32)
    for s, e, v in spans:
        p[s:e] = v
    return p


def test_decode_recovers_two_clean_regions():
    n = 3_000
    prob = _prob_with_regions(n, [(200, 400, 0.95), (1500, 1800, 0.9)])
    regions = decode_regions(prob, FS, smooth_kernel=1, merge_s=0.0, min_dur_s=0.0)
    assert len(regions) == 2
    # Inclusive ends: [200, 399] and [1500, 1799].
    (s0, e0), (s1, e1) = regions
    assert (s0, e0) == (200, 399)
    assert (s1, e1) == (1500, 1799)


def test_decode_hysteresis_bridges_a_dip_above_exit():
    # One event that dips to 0.35 (>= exit 0.3, < enter 0.5) mid-way must stay a
    # SINGLE region; a separate bump that only reaches 0.45 (never >= enter) must
    # NOT be detected at all.
    n = 4_000
    prob = np.zeros(n, np.float32)
    prob[1000:1100] = 0.9
    prob[1100:1120] = 0.35  # dip, still above exit -> keeps the region alive
    prob[1120:1300] = 0.9
    prob[2500:2600] = 0.45  # above exit but never reaches enter -> rejected
    regions = decode_regions(prob, FS, smooth_kernel=1, merge_s=0.0, min_dur_s=0.0)
    assert len(regions) == 1
    (s, e), = regions
    assert s == 1000
    assert e == 1299  # inclusive end of the bridged event


def test_decode_merge_joins_close_regions():
    n = 3_000
    # Two regions separated by a 4-sample sub-exit gap.
    prob = _prob_with_regions(n, [(500, 700, 0.9), (704, 900, 0.9)])
    # merge_s=0 -> stay separate; the 4-sample gap is below exit so hysteresis splits them.
    sep = decode_regions(prob, FS, smooth_kernel=1, merge_s=0.0, min_dur_s=0.0)
    assert len(sep) == 2
    # merge_s = 1e-4 s == 5 samples at 50 kHz -> the 4-sample gap merges them into one.
    joined = decode_regions(prob, FS, smooth_kernel=1, merge_s=1e-4, min_dur_s=0.0)
    assert len(joined) == 1
    (s, e), = joined
    assert (s, e) == (500, 899)


def test_decode_min_duration_drops_short_events():
    n = 3_000
    # A long region (200 samples) and a short one (8 samples).
    prob = _prob_with_regions(n, [(300, 500, 0.9), (2000, 2008, 0.9)])
    # min_dur_s = 1e-3 s == 50 samples at 50 kHz -> the 8-sample event is dropped.
    regions = decode_regions(prob, FS, smooth_kernel=1, merge_s=0.0, min_dur_s=1e-3)
    assert len(regions) == 1
    (s, e), = regions
    assert (s, e) == (300, 499)


def test_decode_empty_when_all_below_enter():
    prob = np.full(2_000, 0.2, np.float32)  # never reaches enter
    assert decode_regions(prob, FS) == []
    assert decode_regions(np.zeros(0, np.float32), FS) == []


def test_decode_median_smoothing_removes_singleton_spike():
    n = 2_000
    prob = np.zeros(n, np.float32)
    prob[1000] = 1.0  # a 1-sample spike; median-smoothing (k=9) must erase it
    assert decode_regions(prob, FS, smooth_kernel=9) == []


def test_decode_rejects_bad_thresholds():
    prob = np.zeros(100, np.float32)
    with pytest.raises(ValueError, match="enter"):
        decode_regions(prob, FS, enter=0.0)
    with pytest.raises(ValueError, match="exit"):
        decode_regions(prob, FS, enter=0.5, exit=0.6)  # exit > enter


# ===========================================================================
# d. predict_probability overlap-average + progress + cancel, via a FAKE model
#    (no TensorFlow needed -- we only exercise the windowing/stitching math).
# ===========================================================================
class _FakeModel:
    """A stand-in for a Keras model. ``predict`` returns a constant probability for
    every sample of every window, shaped (B, W, 1) as the real model would."""

    def __init__(self, window: int, value: float = 0.7):
        self.window = window
        self.value = value
        self.calls = 0

    def predict(self, arr, verbose=0):
        self.calls += 1
        b = arr.shape[0]
        return np.full((b, self.window, 1), self.value, dtype=np.float16)


def _fake_wrapper(window: int, value: float = 0.7) -> KerasEventModel:
    fm = _FakeModel(window, value)
    return KerasEventModel(model=fm, window=window, path="<fake>", mtime=0.0)


def test_predict_probability_overlap_average_and_full_coverage():
    W = 64
    wrap = _fake_wrapper(W, value=0.7)
    # Length deliberately NOT a multiple of the step, to exercise tail anchoring.
    sig = np.zeros(1000, np.float32)
    prob = predict_probability(wrap, sig, batch=8)

    assert prob.dtype == np.float32
    assert prob.shape == sig.shape
    # Every sample is covered (tail anchored) and averages to the constant 0.7.
    assert np.all(counts_covered := (prob > 0))
    np.testing.assert_allclose(prob, 0.7, atol=1e-3)


def test_predict_probability_recovers_windowed_step_values():
    # A model that returns a per-window ramp lets us verify overlap-AVERAGING
    # (not summing) at the sample level.
    W = 100

    class _RampModel:
        window = W

        def predict(self, arr, verbose=0):
            b = arr.shape[0]
            ramp = np.linspace(0.0, 1.0, W, dtype=np.float16)
            return np.broadcast_to(ramp[None, :, None], (b, W, 1)).copy()

    wrap = KerasEventModel(model=_RampModel(), window=W, path="<ramp>", mtime=0.0)
    prob = predict_probability(wrap, np.zeros(400, np.float32), batch=16)
    assert prob.dtype == np.float32
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    # Overlap-average stays within the model's own output range everywhere.
    assert np.isfinite(prob).all()


def test_predict_probability_reports_progress():
    W = 64
    wrap = _fake_wrapper(W)
    seen: list[float] = []
    predict_probability(wrap, np.zeros(500, np.float32), batch=4,
                        progress=lambda frac, msg: seen.append(frac))
    assert seen, "progress was never called"
    assert seen[0] == pytest.approx(0.0)
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen == sorted(seen)  # monotonically non-decreasing


def test_predict_probability_cancel_aborts():
    W = 64
    wrap = _fake_wrapper(W)
    with pytest.raises(MLCancelled):
        predict_probability(wrap, np.zeros(5000, np.float32), batch=2,
                            cancel=lambda: True)


def test_predict_probability_rejects_empty_signal():
    with pytest.raises(ValueError, match="empty"):
        predict_probability(_fake_wrapper(64), np.zeros(0, np.float32))


# ---------------------------------------------------------------------------
# The THREE window-start policies. The two notebook families disagree about the
# loop bound, and they disagree exactly when (n - W) is a multiple of the stride:
#   ferritin  range(0, n - W,     step)   -- last_start_inclusive=False
#   apo       arange(0, n - W + 1, step)  -- last_start_inclusive=True
#   app       ferritin/apo + an anchored window at n - W  -- anchor_tail=True
# ---------------------------------------------------------------------------
def _covered(prob: np.ndarray) -> int:
    """Samples that got at least one window (uncovered ones stay exactly 0)."""
    return int(np.count_nonzero(prob))


@pytest.mark.parametrize("n", [1000, 1024, 1536, 2048])
def test_window_start_policies_differ_only_on_an_exact_stride(n):
    W, step = 512, 256
    wrap = _fake_wrapper(W, value=0.7)
    sig = np.zeros(n, np.float32)

    ferritin = predict_probability(wrap, sig, batch=8, anchor_tail=False)
    apo = predict_probability(wrap, sig, batch=8, anchor_tail=False,
                              last_start_inclusive=True)
    anchored = predict_probability(wrap, sig, batch=8, anchor_tail=True)

    # The anchored policy always covers everything; that is what it is for.
    assert _covered(anchored) == n

    exact = (n - W) % step == 0
    if exact:
        # Apo lands ON n - W and covers the recording; ferritin stops one window
        # early and leaves a whole stride at probability 0.
        assert _covered(apo) == n
        assert _covered(ferritin) == n - step
        assert not np.array_equal(apo, ferritin)
    else:
        np.testing.assert_array_equal(apo, ferritin)
        assert _covered(apo) < n            # both leave the same partial tail blind


def test_last_start_inclusive_is_ignored_when_the_tail_is_anchored():
    W = 512
    wrap = _fake_wrapper(W, value=0.7)
    sig = np.zeros(1000, np.float32)
    np.testing.assert_array_equal(
        predict_probability(wrap, sig, batch=8, anchor_tail=True),
        predict_probability(wrap, sig, batch=8, anchor_tail=True,
                            last_start_inclusive=True))


def test_a_recording_shorter_than_one_window_yields_one_padded_window():
    """The apo notebook's own ``max(1, n - W + 1)`` guard, and the ferritin
    ``or [0]`` fallback, must both still emit exactly one zero-padded window."""
    W = 512
    wrap = _fake_wrapper(W, value=0.7)
    short = np.zeros(300, np.float32)
    for kw in ({}, {"last_start_inclusive": True}):
        prob = predict_probability(wrap, short, batch=4, anchor_tail=False, **kw)
        assert prob.shape == short.shape
        np.testing.assert_allclose(prob, 0.7, atol=1e-3)


def test_the_preset_configs_select_their_notebooks_policy():
    """The wiring, not just the kwarg: each preset must reach predict_probability
    with the loop its notebook actually ran."""
    from nanoscope.analysis.ml_preprocess import PREPROCESS_PRESETS

    for name in ("notebook1", "notebook2", "notebook3"):
        cfg = PREPROCESS_PRESETS[name]
        assert (cfg.anchor_tail, cfg.last_start_inclusive) == (False, False)
    for name in ("apo_nb_500mV", "apo_nb_600mV", "apo_nb_700mV"):
        cfg = PREPROCESS_PRESETS[name]
        assert (cfg.anchor_tail, cfg.last_start_inclusive) == (False, True)
    cfg = PREPROCESS_PRESETS["apo_raw_global"]
    assert cfg.anchor_tail is True


def test_ml_detect_regions_threads_the_tail_policy_and_the_norm_centre():
    """End to end through ml_detect_regions with a fake model: the apo preset must
    mean-centre (and say so in provenance) while the ferritin preset must not."""
    from nanoscope.analysis import ml_preprocess as mp

    W = 3072
    n = W + 1536 * 4                      # an EXACT stride multiple, so the policies differ
    rng = np.random.default_rng(3)
    x = (-35.0 + rng.normal(0.0, 2.0, n)).astype(np.float32)
    x[20_000:20_600] -= 40.0
    wrap = _fake_wrapper(W, value=0.7)

    _r, _p, apo_prov = ml_detect_regions(
        x, FS, wrap,
        preprocess=mp.PREPROCESS_PRESETS["apo_nb_600mV"],
        decode=mp.DECODE_PRESETS["apo_nb_600mV"])
    assert apo_prov["last_start_inclusive"] is True
    assert apo_prov["anchor_tail"] is False
    assert apo_prov["norm_center"] != 0.0
    assert "arange(0, n-W+1, step)" in apo_prov["inference"]

    _r, _p, fer_prov = ml_detect_regions(
        x, FS, wrap,
        preprocess=mp.PREPROCESS_PRESETS["notebook1"],
        decode=mp.DECODE_PRESETS["notebook1"])
    assert fer_prov["last_start_inclusive"] is False
    assert fer_prov["norm_center"] == 0.0
    assert "uncovered (notebook)" in fer_prov["inference"]


def test_ml_detect_regions_end_to_end_with_fake_model():
    # Full detrend -> normalize -> predict -> decode wiring, with a fake model that
    # fires (prob ~1) so we get regions and a well-formed provenance dict.
    W = 128
    wrap = _fake_wrapper(W, value=0.99)
    sig = (100.0 + 5.0 * np.sin(np.linspace(0, 6, 4000))).astype(np.float32)
    ticks: list[float] = []
    regions, prob, prov = ml_detect_regions(
        sig, FS, wrap, enter=0.5, exit=0.3,
        progress=lambda f, m: ticks.append(f),
    )
    assert isinstance(regions, list) and len(regions) >= 1
    assert prob.dtype == np.float32 and prob.shape == sig.shape
    assert prov["detector"] == "ml_keras"
    assert prov["model_window"] == W
    assert prov["enter"] == 0.5 and prov["exit"] == 0.3
    assert prov["fs"] == FS
    assert prov["n_regions"] == len(regions)
    assert ticks and ticks[0] == pytest.approx(0.0) and ticks[-1] == pytest.approx(1.0)


# ===========================================================================
# g. FEATURE path (12-channel teacher contract) -- pure, NO TensorFlow.
#    create_multiscale_features_A + the channel dispatch in predict_probability /
#    ml_detect_regions are exercised with FAKE 12-channel models, so these run
#    everywhere (the real teacher is covered by the corpus tests below).
# ===========================================================================
def test_create_multiscale_features_A_shape_and_standardization():
    # A 2048-sample window with drift, oscillation, noise and a blockade-like dip.
    W = 2048
    rng = np.random.default_rng(1)
    t = np.linspace(0, 25, W)
    win = (30.0 + 4.0 * np.sin(t) + rng.normal(0, 1.0, W)).astype(np.float32)
    win[900:1050] -= 20.0

    F = create_multiscale_features_A(win, window_len=W, fs=FS)

    assert F.shape == (W, 12)          # exactly 12 features, one column each
    assert F.dtype == np.float32
    assert np.isfinite(F).all()        # no NaN/Inf leaks from any helper
    # Every feature column is self-standardized (_standardize): ~zero mean and
    # ~unit std. This is WHY the feature path must NOT apply global-raw scaling.
    means = F.mean(axis=0)
    stds = F.std(axis=0)
    assert np.max(np.abs(means)) < 1e-4
    assert np.all(stds > 0.5) and np.all(stds < 1.5)


def test_create_multiscale_features_A_requires_exact_window_length():
    with pytest.raises(ValueError, match="window_len"):
        create_multiscale_features_A(np.zeros(2000, np.float32), window_len=2048, fs=FS)


class _Fake12Model:
    """Stand-in teacher: records the batch shapes it is fed (to PROVE the FEATURE
    branch built ``(B, W, 12)`` inputs) and returns a constant ``(B, W, 1)``
    probability, shaped exactly as the real teacher's output."""

    def __init__(self, window: int, value: float = 0.99):
        self.window = window
        self.value = value
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, arr, verbose=0):
        self.shapes.append(tuple(arr.shape))
        assert arr.ndim == 3 and arr.shape[2] == 12, (
            f"feature branch must feed (B, W, 12), got {arr.shape}"
        )
        b, w, _ = arr.shape
        return np.full((b, w, 1), self.value, dtype=np.float16)


def _fake_wrapper_12(window: int, value: float = 0.99) -> tuple[KerasEventModel, _Fake12Model]:
    fm = _Fake12Model(window, value)
    wrap = KerasEventModel(
        model=fm, window=window, path="<fake12>", mtime=0.0,
        channels=12, input_shape=(None, window, 12), output_shape=(None, window, 1),
    )
    return wrap, fm


def test_predict_probability_feature_branch_builds_W12_windows():
    W = 256
    wrap, fm = _fake_wrapper_12(W, value=0.8)
    detr = detrend_for_model((50.0 + np.sin(np.linspace(0, 10, 1200))).astype(np.float32), FS)

    prob = predict_probability(wrap, detr, fs=FS, batch=3)

    assert prob.dtype == np.float32 and prob.shape == detr.shape
    np.testing.assert_allclose(prob, 0.8, atol=1e-2)
    assert fm.shapes, "model.predict was never called"
    # EVERY batch fed to the model is (B, W, 12) -- the feature branch, not (W, 1).
    assert all(s[2] == 12 for s in fm.shapes)
    assert all(s[1] == W for s in fm.shapes)


def test_predict_probability_feature_branch_requires_fs():
    W = 256
    wrap, _ = _fake_wrapper_12(W)
    with pytest.raises(ValueError, match="fs"):
        predict_probability(wrap, np.zeros(1000, np.float32), batch=4)  # no fs -> error


def test_channel_dispatch_selects_raw_vs_feature_path():
    """The SAME signal drives (B,W,1) on a channels==1 wrapper and (B,W,12) on a
    channels==12 wrapper, and only the RAW path global-raw normalizes."""
    W = 256
    sig = (40.0 + 3.0 * np.sin(np.linspace(0, 12, 1500))).astype(np.float32)

    class _RawRec:
        def __init__(self, window: int):
            self.window = window
            self.shapes: list[tuple[int, ...]] = []

        def predict(self, arr, verbose=0):
            self.shapes.append(tuple(arr.shape))
            return np.full((arr.shape[0], arr.shape[1], 1), 0.01, np.float16)

    raw_model = _RawRec(W)
    raw_wrap = KerasEventModel(
        model=raw_model, window=W, path="<raw>", mtime=0.0,
        channels=1, input_shape=(None, W, 1),
    )
    _, _, prov_raw = ml_detect_regions(sig, FS, raw_wrap)
    assert prov_raw["input_mode"] == "raw" and prov_raw["channels"] == 1
    assert raw_model.shapes and all(s[2] == 1 for s in raw_model.shapes)

    feat_wrap, feat_model = _fake_wrapper_12(W, value=0.01)
    _, _, prov_feat = ml_detect_regions(sig, FS, feat_wrap)
    assert prov_feat["input_mode"] == "features" and prov_feat["channels"] == 12
    assert feat_model.shapes and all(s[2] == 12 for s in feat_model.shapes)
    assert "NO global-raw" in prov_feat["preprocess"]


def test_resolve_channels_accepts_1_12_none_and_rejects_others():
    import nanoscope.analysis.ml_detect as m

    # None (unspecified last dim) and 1 both mean the RAW single channel; 12 is
    # the FEATURE contract. Anything else is rejected with a clear message.
    assert m._resolve_channels(None, (None, 2048, None)) == 1
    assert m._resolve_channels(1, (None, 2048, 1)) == 1
    assert m._resolve_channels(12, (None, 2048, 12)) == 12
    for bad in (2, 3, 7, 16):
        with pytest.raises(ValueError, match="input features"):
            m._resolve_channels(bad, (None, 2048, bad))


# ===========================================================================
# e. lazy TF + graceful degradation
# ===========================================================================
def test_ml_unavailable_is_raised_when_tf_import_fails(monkeypatch):
    """If TensorFlow cannot be imported, the loader surfaces MLUnavailable -- a
    clear, catchable error -- rather than a bare ImportError or a crash. Simulated
    by monkeypatching the import so this runs even where TF *is* installed."""
    import nanoscope.analysis.ml_detect as m

    monkeypatch.setattr(m, "_TF", None)  # force a fresh import attempt
    real_import = m.importlib.import_module

    def _boom(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("simulated: no tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(m.importlib, "import_module", _boom)
    with pytest.raises(MLUnavailable, match="TensorFlow"):
        m._import_tf()


def test_tensorflow_available_flag_is_bool():
    assert isinstance(TENSORFLOW_AVAILABLE, bool)


# ===========================================================================
# e/f. real TF: run in a SUBPROCESS so the parent pytest process never imports
#      tensorflow.
#
#      Why a subprocess: importing tensorflow(-metal) into the SAME process as
#      the pytest-qt gui suite aborts the harness with
#          "Fatal Python error: libshiboken: seterror_argument failed to call
#           update_mapping"   (raised inside pytestqt/qt_compat.py:exec)
#      -- a native TF <-> PySide6/shiboken state conflict *inside pytest-qt's
#      exception-capturing event loop*. A plain PySide6 app + TF coexist fine in
#      either import order, so the SHIPPED app is unaffected; only this test
#      harness is. Isolating every real-TF import in a child process keeps the
#      whole 610+ suite green while still exercising the REAL ml_detect code.
# ===========================================================================
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs inside a CHILD interpreter. Prints one "__RESULT__<json>" line on success.
_SUBPROCESS_PROGRAM = r"""
import os, sys, json
REPO_ROOT, MODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO_ROOT)
import numpy as np
from nanoscope.analysis import ml_detect

out = {}
if MODE == "cpu":
    tf = ml_detect._import_tf()                        # forces CPU
    out["n_gpus_visible"] = len(tf.config.get_visible_devices("GPU"))
    out["cached"] = ml_detect._import_tf() is tf       # module cached
elif MODE == "corpus":
    model_path, abf_path = sys.argv[3], sys.argv[4]
    wrap = ml_detect.load_keras_model(model_path)      # forces CPU + loads
    out["window"] = int(wrap.window)
    out["input_last_dim"] = wrap.input_shape[-1]
    out["cached_same"] = ml_detect.load_keras_model(model_path) is wrap
    from neo.rawio import AxonRawIO
    r = AxonRawIO(filename=abf_path); r.parse_header()
    total = r.get_signal_size(block_index=0, seg_index=0)
    n_slice = min(150000, total)                       # ~3 s @ 50 kHz, small = fast
    a = max(0, (total - n_slice) // 2)                 # middle: where events live
    raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=a, i_stop=a + n_slice)
    cur = r.rescale_signal_raw_to_float(raw, dtype="float32")[:, 2]  # ch 2 = current
    regions, prob, prov = ml_detect.ml_detect_regions(cur, 50000.0, wrap, enter=0.5, exit=0.3)
    out["n_slice"] = int(n_slice)
    out["n_regions"] = int(len(regions))
    out["prob_min"] = float(prob.min())
    out["prob_max"] = float(prob.max())
    out["prob_shape_ok"] = bool(prob.shape == cur.shape and prob.dtype == np.float32)
    out["regions_in_bounds"] = bool(all(0 <= s <= e < n_slice for s, e in regions))
    out["prov_window"] = int(prov["model_window"])
    out["prov_detector"] = prov["detector"]
elif MODE == "build_feature":
    W, save_path = int(sys.argv[3]), sys.argv[4]
    tf = ml_detect._import_tf()                        # forces CPU
    @tf.keras.utils.register_keras_serializable(package="Custom")
    def clip_activation(x):
        return tf.clip_by_value(x, 0.0, 1.0)
    inp = tf.keras.Input(shape=(W, 12))
    h = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(inp)
    o = tf.keras.layers.Conv1D(1, 1, padding="same", activation=clip_activation)(h)
    tf.keras.Model(inp, o).save(save_path)
    out["built"] = True
    out["exists"] = os.path.exists(save_path)
elif MODE == "run_feature":
    W, load_path = int(sys.argv[3]), sys.argv[4]
    # THE APP RULE: a 12-feature teacher is refused at the load door.
    try:
        ml_detect.load_keras_model(load_path)
        out["refused"] = False
    except ValueError as e:
        out["refused"] = True
        out["refuse_msg"] = str(e)[:200]
    tf = ml_detect._import_tf()
    # In this clean process clip_activation is NOT globally registered, so a plain
    # (no custom_objects) load must FAIL -- proving _CUSTOM_OBJECTS is what loads it.
    try:
        tf.keras.models.load_model(load_path, compile=False)
        out["needs_custom_objects"] = False
    except Exception:
        out["needs_custom_objects"] = True
    # BELOW the door the FEATURE machinery is RETAINED and must still work, so build the
    # wrapper directly and exercise it. Keeping this covered is the point of refusing at
    # the door rather than deleting the path: the rule is what changed, not the code.
    raw = tf.keras.models.load_model(load_path, compile=False,
                                     custom_objects=ml_detect._CUSTOM_OBJECTS)
    ishape = tuple(raw.input_shape)
    wrap = ml_detect.KerasEventModel(
        model=raw, window=int(ishape[1]), path=load_path, mtime=0.0,
        channels=int(ishape[-1]), input_shape=ishape,
        output_shape=tuple(raw.output_shape),
    )
    out["channels"] = int(wrap.channels)
    out["window"] = int(wrap.window)
    out["input_last_dim"] = wrap.input_shape[-1]
    rng = np.random.default_rng(0)
    sig = (100.0 + 5.0 * np.sin(np.linspace(0, 40, 6 * W)) + rng.normal(0, 1.0, 6 * W)).astype(np.float32)
    regions, prob, prov = ml_detect.ml_detect_regions(sig, 50000.0, wrap, enter=0.5, exit=0.3)
    out["prob_in_01"] = bool(float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0)
    out["prob_shape_ok"] = bool(prob.shape == sig.shape and prob.dtype == np.float32)
    out["input_mode"] = prov["input_mode"]
    out["prov_channels"] = int(prov["channels"])
elif MODE == "reject7":
    W, save_path = int(sys.argv[3]), sys.argv[4]
    tf = ml_detect._import_tf()
    inp = tf.keras.Input(shape=(W, 7))
    o = tf.keras.layers.Conv1D(1, 1, padding="same")(inp)
    tf.keras.Model(inp, o).save(save_path)
    try:
        ml_detect.load_keras_model(save_path)
        out["raised"] = False
    except ValueError as e:
        out["raised"] = True
        out["msg"] = str(e)[:160]
elif MODE == "corpus_teacher":
    model_path, abf_path = sys.argv[3], sys.argv[4]
    # THE APP RULE: the load door refuses a teacher (students only).
    try:
        ml_detect.load_keras_model(model_path)
        out["refused"] = False
    except ValueError:
        out["refused"] = True
    # The FEATURE machinery under that door is retained, so build the wrapper directly
    # to keep exercising it on a REAL teacher and a REAL recording.
    tf = ml_detect._import_tf()
    try:
        raw_model = tf.keras.models.load_model(model_path, compile=False,
                                               custom_objects=ml_detect._CUSTOM_OBJECTS)
    except Exception:
        # Same retry load_keras_model performs: a model saved by a NEWER Keras carries a
        # per-layer `quantization_config` this Keras cannot parse. Bypassing the door must
        # not also bypass that, or the test fails on the SAVE format rather than the rule.
        raw_model = ml_detect._load_quantization_stripped(
            tf, model_path, ml_detect._CUSTOM_OBJECTS)
    _ish = tuple(raw_model.input_shape)
    wrap = ml_detect.KerasEventModel(
        model=raw_model, window=int(_ish[1]), path=model_path, mtime=0.0,
        channels=int(_ish[-1]), input_shape=_ish, output_shape=tuple(raw_model.output_shape))
    out["window"] = int(wrap.window)
    out["channels"] = int(wrap.channels)
    out["input_last_dim"] = wrap.input_shape[-1]
    from neo.rawio import AxonRawIO
    r = AxonRawIO(filename=abf_path); r.parse_header()
    total = r.get_signal_size(block_index=0, seg_index=0)
    n_slice = min(150000, total)                       # ~3 s @ 50 kHz, small = fast
    a = max(0, (total - n_slice) // 2)                 # middle: where events live
    raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=a, i_stop=a + n_slice)
    cur = r.rescale_signal_raw_to_float(raw, dtype="float32")[:, 2]  # ch 2 = current
    regions, prob, prov = ml_detect.ml_detect_regions(cur, 50000.0, wrap, enter=0.5, exit=0.3)
    out["n_slice"] = int(n_slice)
    out["n_regions"] = int(len(regions))
    out["prob_min"] = float(prob.min())
    out["prob_max"] = float(prob.max())
    out["prob_shape_ok"] = bool(prob.shape == cur.shape and prob.dtype == np.float32)
    out["regions_in_bounds"] = bool(all(0 <= s <= e < n_slice for s, e in regions))
    out["prov_input_mode"] = prov["input_mode"]
    out["prov_channels"] = int(prov["channels"])
print("__RESULT__" + json.dumps(out))
"""


def _run_tf_subprocess(mode: str, *extra: str) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROGRAM, str(_REPO_ROOT), mode, *extra],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"TF subprocess (mode={mode}) failed rc={proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-3000:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise AssertionError(
        f"no __RESULT__ line from TF subprocess (mode={mode})\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-2000:]}"
    )


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="tensorflow not installed")
def test_import_tf_forces_cpu_and_is_cached():
    res = _run_tf_subprocess("cpu")
    # No GPU is visible to TF -> the mixed_float16 LSTM runs on CPU (no SIGABRT).
    assert res["n_gpus_visible"] == 0
    assert res["cached"] is True  # _import_tf() returns the cached module


@pytest.fixture(scope="module")
def corpus_ml_result() -> dict:
    """Run the whole real-model pipeline ONCE in a child process (TF import is
    expensive); the two corpus tests below assert on the parsed result."""
    if not _HAVE_MODEL:
        pytest.skip("TF and/or the real .keras model absent")
    if not ABF_PATH.exists():
        pytest.skip(f"corpus ABF absent: {ABF_PATH.name}")
    return _run_tf_subprocess("corpus", str(MODEL_PATH), str(ABF_PATH))


@pytest.mark.corpus
def test_load_keras_model_real_window_and_cache(corpus_ml_result):
    assert corpus_ml_result["window"] == 3072  # this model's input_shape[1]
    assert corpus_ml_result["input_last_dim"] in (None, 1)  # single raw channel
    # (path, mtime) cache -> the same file returns the same wrapper object.
    assert corpus_ml_result["cached_same"] is True


@pytest.mark.corpus
def test_ml_detect_regions_on_real_abf_slice(corpus_ml_result):
    r = corpus_ml_result
    assert r["prob_shape_ok"] is True
    # Probabilities span 0..1: open pore ~0, inside events ~1.
    assert r["prob_min"] < 0.1
    assert r["prob_max"] > 0.9
    # The middle of this apo-ferritin recording contains translocations.
    assert r["n_regions"] >= 1
    assert r["regions_in_bounds"] is True
    assert r["prov_window"] == 3072
    assert r["prov_detector"] == "ml_keras"


# ===========================================================================
# g/f. real TF for the FEATURE path -- also in a SUBPROCESS (same pytest-qt <-> TF
#      reason as above). A tiny REAL (W, 12) model whose output uses the custom
#      clip_activation is built+saved in one child, then loaded via
#      load_keras_model in a SEPARATE child so the custom_objects RETRY is genuinely
#      exercised (clean registry) before the FEATURE branch runs end to end.
# ===========================================================================
@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="tensorflow not installed")
def test_feature_model_loads_via_custom_objects_and_runs_feature_branch(tmp_path):
    W = 256
    model_path = tmp_path / "tiny_teacher_feat.keras"

    built = _run_tf_subprocess("build_feature", str(W), str(model_path))
    assert built["built"] is True and built["exists"] is True

    res = _run_tf_subprocess("run_feature", str(W), str(model_path))

    # THE APP RULE (students only): a 12-feature teacher is refused at the load door,
    # by INPUT WIDTH rather than by filename -- this model is called
    # "tiny_teacher_feat.keras" but the refusal must not depend on that.
    assert res["refused"] is True
    assert "TEACHER" in res["refuse_msg"]
    assert "STUDENT" in res["refuse_msg"]

    # ...and the FEATURE machinery UNDER that door is retained and still correct. It is
    # kept because refusing is a policy about what the app runs, not a claim the code is
    # wrong; deleting it would also delete the only coverage of the 12-channel dispatch.
    assert res["channels"] == 12
    assert res["window"] == W
    assert res["input_last_dim"] == 12
    # The clean process could NOT load it without custom objects -> clip_activation in
    # _CUSTOM_OBJECTS is what loads a teacher at all.
    assert res["needs_custom_objects"] is True
    # ml_detect_regions ran down the FEATURE branch and produced a valid trace.
    assert res["prob_in_01"] is True
    assert res["prob_shape_ok"] is True
    assert res["input_mode"] == "features"
    assert res["prov_channels"] == 12


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="tensorflow not installed")
def test_load_keras_model_rejects_seven_channel_model(tmp_path):
    W = 128
    model_path = tmp_path / "seven_ch.keras"
    res = _run_tf_subprocess("reject7", str(W), str(model_path))
    assert res["raised"] is True                 # a (W, 7) model is rejected...
    assert "input features" in res["msg"]        # ...with the clear channel message


@pytest.fixture(scope="module")
def corpus_teacher_result() -> dict:
    """Run the whole real-TEACHER (12-feature) pipeline ONCE in a child process.
    Skips unless the teacher .keras and a corpus ABF are both present (mirrors the
    raw ``corpus_ml_result`` fixture)."""
    if not _HAVE_TEACHER:
        pytest.skip("TF and/or the real teacher .keras model absent")
    if not ABF_PATH.exists():
        pytest.skip(f"corpus ABF absent: {ABF_PATH.name}")
    return _run_tf_subprocess("corpus_teacher", str(TEACHER_PATH), str(ABF_PATH))


@pytest.mark.corpus
def test_teacher_model_loads_as_12_channel(corpus_teacher_result):
    # Students only: the real teacher is refused at the load door...
    assert corpus_teacher_result["refused"] is True
    # ...and is still a well-formed 12-feature model underneath.
    assert corpus_teacher_result["channels"] == 12
    assert corpus_teacher_result["input_last_dim"] == 12
    assert corpus_teacher_result["window"] == 2048  # create_multiscale_features_A default


@pytest.mark.corpus
def test_ml_detect_regions_feature_path_on_real_abf_slice(corpus_teacher_result):
    r = corpus_teacher_result
    assert r["prob_shape_ok"] is True
    # Probabilities span 0..1: open pore ~0, inside events ~1.
    assert r["prob_min"] < 0.1
    assert r["prob_max"] > 0.9
    assert r["n_regions"] >= 1
    assert r["regions_in_bounds"] is True
    assert r["prov_input_mode"] == "features"
    assert r["prov_channels"] == 12


def test_strip_config_key_removes_quantization_config_recursively():
    """A newer-Keras `quantization_config` is stripped everywhere in a nested config
    so an older Keras can still load the model (weights untouched)."""
    from nanoscope.analysis.ml_detect import _strip_config_key
    cfg = {
        "class_name": "Functional",
        "quantization_config": {"mode": "int8"},
        "config": {"layers": [
            {"class_name": "Dense", "quantization_config": {"mode": "int8"}, "name": "d1"},
            {"class_name": "Dense", "name": "d2"},
        ]},
    }
    _strip_config_key(cfg, "quantization_config")
    import json
    assert "quantization_config" not in json.dumps(cfg)
    # non-target keys survive
    assert cfg["class_name"] == "Functional"
    assert [l["name"] for l in cfg["config"]["layers"]] == ["d1", "d2"]


def test_teacher_models_are_refused_by_input_width_not_by_name():
    """THE app rule: students only. A teacher is the distillation source that labels
    training data, not a detector to deploy.

    Asserted on ``_reject_teacher`` directly so it needs no TensorFlow: the rule is a
    statement about the input contract, and it must hold for a file called anything at
    all. Across the owner's corpus name and width agree perfectly (every ``*student*``
    is 1-channel, all seven ``*teacher*`` files are 12) -- but a name is a label someone
    can mistype and an input shape is what the architecture IS.
    """
    from nanoscope.analysis.ml_detect import TEACHER_CHANNELS, _reject_teacher

    assert TEACHER_CHANNELS == 12
    # A student passes, whatever it is called.
    _reject_teacher(1, (None, 2048, 1), "/tmp/teacher_named_but_single_channel.keras")
    # A teacher is refused, whatever it is called.
    with pytest.raises(ValueError) as exc:
        _reject_teacher(12, (None, 2048, 12), "/tmp/innocuous_name.keras")
    msg = str(exc.value)
    assert "TEACHER" in msg and "STUDENT" in msg
    assert "innocuous_name.keras" in msg          # names the file it refused
    assert "(None, 2048, 1)" in msg               # ...and what to load instead


def test_peek_model_channels_reads_the_feature_width_without_tensorflow(tmp_path):
    """The GUI-thread gate must not import TF — that is the whole reason it exists.

    ``browse_model`` runs on the main thread; importing TensorFlow there would freeze
    the UI for seconds on every model pick. Both peeks are pure zipfile+json.
    """
    import json
    import subprocess as sp
    import sys as _sys
    import zipfile

    # A SYNTHETIC .keras. The peek is pure zipfile+json, so it needs no real model and
    # no TensorFlow -- and this test must never skip. Gated on a hardcoded
    # ``~/teacher_trained.keras`` it silently skipped on the very machine whose teachers
    # live on the Desktop, so the "must not import TF" assertion never ran at all.
    fake = tmp_path / "synthetic_teacher.keras"
    with zipfile.ZipFile(fake, "w") as z:
        z.writestr("config.json", json.dumps(
            {"config": {"layers": [{"class_name": "InputLayer",
                                    "config": {"batch_shape": [None, 256, 12]}}]}}))

    src = (
        "import sys\n"
        "from nanoscope.analysis.ml_detect import is_teacher_model, peek_model_channels\n"
        f"p = {str(fake)!r}\n"
        "print(peek_model_channels(p), is_teacher_model(p), 'tensorflow' in sys.modules)\n"
    )
    out = sp.run([_sys.executable, "-c", src], capture_output=True, text=True,
                 cwd=str(Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr[-2000:]
    channels, teacher, tf_loaded = out.stdout.split()
    assert channels == "12"
    assert teacher == "True"
    assert tf_loaded == "False", "the pick-time gate must not import TensorFlow"


def test_an_unreadable_model_is_not_called_a_teacher(tmp_path):
    """The cheap gate errs toward ALLOWING: load_keras_model is the authority.

    A config it cannot parse must fall through rather than block a legitimate student
    the peek simply failed to read.
    """
    from nanoscope.analysis.ml_detect import is_teacher_model, peek_model_channels

    missing = tmp_path / "nope.keras"
    assert peek_model_channels(missing) is None
    assert is_teacher_model(missing) is False
    junk = tmp_path / "junk.keras"
    junk.write_bytes(b"not a zip")
    assert is_teacher_model(junk) is False
