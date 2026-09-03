"""Round 5: the second detector on the Detection tab, and the model-input
normalization the owner's three student pipelines all share.

Two things are pinned here that no other file can pin:

  * THE TWO DETECTORS COEXIST ON ONE FORM. ``params()`` carries both threshold sets so
    switching does not discard the other's settings, and neither set ever reaches the
    other detector's run. The leak is prevented structurally, in `_detect_kwargs`, not
    by hoping every producer of a params dict remembers -- so it is tested at the seam
    the module actually uses.
  * THE THREE ML PRESETS ARE THE OWNER'S OWN CHAINS, and their normalization is a
    DIFFERENT KERNEL from the one every earlier preset uses. The default has to stay
    max-abs or every stored preset, saved run and QSettings user preset silently
    changes what it feeds a trained model.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect_cusum import (
    DEFAULT_CUSUM_H,
    DEFAULT_CUSUM_K,
    DEFAULT_CUSUM_MERGE_GAP_S,
    DEFAULT_CUSUM_MIN_DWELL_S,
)
from nanoscope.analysis.ml_preprocess import (
    DEFAULT_NORMALIZE_CLIP_SIGMA,
    NORMALIZE_METHODS,
    PREPROCESS_PRESETS,
    PRESET_LABELS,
    PRESET_NAMES,
    MLPreprocessConfig,
    model_input_normalization,
)
from nanoscope.analysis.modules import _detect_kwargs
from nanoscope.gui.panels import DetectionPanel


# ---------------------------------------------------------------------------
# the detector selector
# ---------------------------------------------------------------------------
def test_the_tab_offers_both_detectors_by_module_id(qtbot):
    """The combo carries MODULE IDS, not nicknames, so the panel and
    ``ANALYSIS_REGISTRY`` cannot disagree about what "cusum" names."""
    from nanoscope.analysis.modules import register_all
    from nanoscope.core.analysis import ANALYSIS_REGISTRY

    register_all()
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    ids = [mid for _label, mid in panel._DETECTORS]
    assert ids == ["detect.schmitt", "detect.cusum"]
    for mid in ids:
        assert mid in ANALYSIS_REGISTRY
    assert panel.detector_choice() == "detect.schmitt"   # unchanged default


def test_the_selection_round_trips_and_swaps_the_threshold_page(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.detector_stack.currentIndex() == 0

    panel.set_detector("detect.cusum")
    assert panel.detector_choice() == "detect.cusum"
    assert panel.detector_stack.currentIndex() == 1
    assert panel.params()["detector"] == "detect.cusum"

    panel.set_params({"detector": "detect.schmitt"})
    assert panel.detector_choice() == "detect.schmitt"
    assert panel.detector_stack.currentIndex() == 0

    panel.set_detector("detect.nonexistent")            # ignored, never a crash
    assert panel.detector_choice() == "detect.schmitt"


def test_the_thresholds_are_labelled_dimensionless_not_picoamps(qtbot):
    """The owner asked for this explicitly: k and h are multiples of the MAD noise
    sigma -- the same sigma the table's snr column is divided by -- and a bare "28" in
    a nanopore app reads as picoamps."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_detector("detect.cusum")
    assert panel.cusum_k.suffix().strip() == "σ"
    assert "σ" in panel.cusum_h.suffix()
    assert "NOISE SIGMAS" in panel.cusum_k.toolTip()
    assert "DIMENSIONLESS" in panel.cusum_h.toolTip()
    assert "DIMENSIONLESS" in panel.detector_note.text()


def test_switching_to_cusum_seeds_its_own_merge_and_dwell_defaults(qtbot):
    """They differ by 25-50x. Inheriting the Schmitt numbers would run the owner's
    algorithm at settings his calibration never saw, and silently."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.params()["min_dwell_s"] == 2e-5
    assert panel.params()["merge_gap_s"] is None        # 0 ms == off

    panel.set_detector("detect.cusum")
    p = panel.params()
    assert p["min_dwell_s"] == pytest.approx(DEFAULT_CUSUM_MIN_DWELL_S)
    assert p["merge_gap_s"] == pytest.approx(DEFAULT_CUSUM_MERGE_GAP_S)
    assert p["cusum_k"] == pytest.approx(DEFAULT_CUSUM_K)
    assert p["cusum_h"] == pytest.approx(DEFAULT_CUSUM_H)


def test_a_value_the_user_chose_is_never_re_seeded(qtbot):
    """The re-seed compares against the OTHER detector's default and replaces only when
    the widget still IS that default. A number the user typed is theirs."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.min_dwell_s.setValue(3.5e-4)
    panel.merge_gap_ms.setValue(7.0)

    panel.set_detector("detect.cusum")
    assert panel.params()["min_dwell_s"] == pytest.approx(3.5e-4)
    assert panel.params()["merge_gap_s"] == pytest.approx(7.0e-3)


def test_switching_detectors_does_not_discard_the_others_thresholds(qtbot):
    """Which is the whole reason both sets ride in one params dict."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.k_in.setValue(7.5)
    panel.cusum_h.setValue(12.0)
    panel.set_detector("detect.cusum")
    panel.set_detector("detect.schmitt")
    assert panel.params()["k_in"] == pytest.approx(7.5)
    assert panel.params()["cusum_h"] == pytest.approx(12.0)


def test_the_selected_detector_receives_only_its_own_thresholds(qtbot):
    """Tested at the seam the module actually uses, not by inspecting the dict: this is
    the call whose kwargs reach `detect_events` / `detect_events_cusum`."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    for mid, mine, theirs in (
        ("detect.schmitt", {"k_in", "k_out"}, {"cusum_k", "cusum_h"}),
        ("detect.cusum", {"cusum_k", "cusum_h"}, {"k_in", "k_out"}),
    ):
        panel.set_detector(mid)
        kwargs = set(_detect_kwargs(panel.params(), mid))
        assert mine <= kwargs
        assert not theirs & kwargs
        assert "detector" not in kwargs


def test_a_cusum_run_end_to_end_from_the_panels_own_params(qtbot):
    """The strongest form of the contract: hand the real detector exactly what the form
    emits, through the real strip, and get the 20-column table back."""
    from nanoscope.analysis.detect_cusum import detect_events_cusum
    from nanoscope.analysis.events import SCHEMA

    fs = 50_000.0
    rng = np.random.default_rng(0)
    x = (-1000.0 + 8.0 * rng.standard_normal(120_000)).astype(np.float32)
    for s in range(10_000, 120_000, 20_000):
        x[s:s + 200] -= 120.0          # conductive: the SIGNED current goes further down

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_detector("detect.cusum")
    kwargs = _detect_kwargs(panel.params(), "detect.cusum")
    kwargs.pop("filter_cutoff_hz", None)   # off-sentinel is fine, but keep it explicit
    df, prov = detect_events_cusum(x, fs, **kwargs)

    assert list(df.columns)[:len(SCHEMA)] == list(SCHEMA)
    assert prov["detector"] == "cusum_sequential"
    assert len(df) >= 5
    assert (df["depth_pA"] > 0).all()
    # The audit identity this detector must satisfy (no ``n_rejected_below_enter``).
    assert (prov["n_cusum_firings"] - prov["n_merged"] - prov["n_rejected_short"]
            - prov["n_rejected_excluded"]) == prov["n_events"]


# ---------------------------------------------------------------------------
# robust_normalize as the ML detector's model input
# ---------------------------------------------------------------------------
def test_the_default_normalization_is_still_max_abs_bit_for_bit():
    """THE COMPATIBILITY CLAUSE. Every preset, stored run and QSettings-saved user
    preset written before this field existed must reproduce exactly -- `from_dict`
    drops unknown keys and fills missing ones from the dataclass defaults, so an old
    dict has to land on max-abs."""
    from nanoscope.analysis.ml_preprocess import normalize_global

    assert MLPreprocessConfig().normalize_method == "maxabs"
    old_dict = {"preset": "notebook1", "baseline_method": "robust_median",
                "normalize": True, "normalize_center": False}
    cfg = MLPreprocessConfig.from_dict(old_dict)
    assert cfg.normalize_method == "maxabs"
    assert cfg.normalize_clip_sigma == DEFAULT_NORMALIZE_CLIP_SIGMA

    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 5.0, 4_096).astype(np.float32)
    np.testing.assert_array_equal(
        model_input_normalization(x, cfg), normalize_global(x, None, 0.0))


@pytest.mark.parametrize(
    "name", [n for n in PRESET_NAMES if not n.endswith("_student")])
def test_no_pre_existing_preset_moved_off_max_abs(name):
    assert PREPROCESS_PRESETS[name].normalize_method == "maxabs"


def test_a_stored_user_preset_from_before_the_field_still_loads():
    """The dataclass round-trips through QSettings as JSON. An old entry has neither
    key; a new one has both. Both must load, and the old one must be unchanged."""
    old = {"preset": "mine", "baseline_method": "savgol", "baseline_window_s": 0.5,
           "notch_freqs": [60.0, 120.0, 180.0]}
    cfg = MLPreprocessConfig.from_dict(old)
    assert cfg.normalize_method == "maxabs"
    # ...and a full round trip is stable.
    assert MLPreprocessConfig.from_dict(cfg.to_dict()) == cfg


def test_robust_sigma_is_the_kernel_the_notebooks_defined():
    from nanoscope.analysis.filters import robust_normalize

    cfg = MLPreprocessConfig(normalize_method="robust_sigma",
                             normalize_clip_sigma=30.0)
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 5.0, 8_192).astype(np.float32)
    x[100:150] -= 900.0                       # far past 30 sigma: it must SATURATE
    got = model_input_normalization(x, cfg)
    np.testing.assert_array_equal(got, robust_normalize(x, clip_sigma=30.0))
    assert float(got.min()) == pytest.approx(-1.0)


def test_robust_sigma_refuses_a_precomputed_scale_rather_than_ignoring_it():
    """It is not affine, so there is no (center, scale) pair that reproduces it and no
    way to "apply the recording's statistics to a slice". A caller that needs recording
    scope must normalize the whole span and slice afterwards -- which is what
    `MLDetect.run` does. Silently ignoring the pair is how a sub-range run would end up
    normalized to itself without anyone noticing."""
    cfg = MLPreprocessConfig(normalize_method="robust_sigma")
    x = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    with pytest.raises(ValueError, match="estimates its own"):
        model_input_normalization(x, cfg, scale=2.0)


def test_the_method_vocabulary_is_validated():
    with pytest.raises(ValueError, match="normalize_method"):
        MLPreprocessConfig(normalize_method="whitening")
    with pytest.raises(ValueError, match="normalize_clip_sigma"):
        MLPreprocessConfig(normalize_method="robust_sigma", normalize_clip_sigma=0.0)
    assert NORMALIZE_METHODS == ("maxabs", "robust_sigma")


# ---------------------------------------------------------------------------
# the three shipped ML presets
# ---------------------------------------------------------------------------
_ML_RECIPES = {
    "hg_student": dict(
        notch_freqs=(60.0, 120.0, 180.0, 300.0, 420.0, 540.0),
        baseline_method="masked_median", masked_rough_s=0.5, masked_base_s=0.02,
        baseline_decim=10, masked_mask_sigma=3.5, masked_dilate_ms=0.2),
    "zn_student": dict(
        notch_freqs=(60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0),
        baseline_method="masked_median", masked_rough_s=0.5, masked_base_s=0.2,
        baseline_decim=10, masked_mask_sigma=3.0, masked_dilate_ms=2.0),
    "cu_student": dict(
        notch_freqs=(60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0),
        # SAVITZKY-GOLAY, NOT masked median -- the notebook's own cell-0 markdown is
        # stale prose; Block 0's BASELINE_METHOD dict and all four written
        # cu_*_metadata.json say "sg" with masked_median_params null.
        baseline_method="savgol", baseline_window_s=0.2, baseline_poly=3,
        baseline_decim=50),
}


@pytest.mark.parametrize("name", list(_ML_RECIPES))
def test_each_shipped_ml_preset_is_the_owners_chain_exactly(name):
    cfg = PREPROCESS_PRESETS[name]
    assert cfg.preset == name
    assert cfg.notch_enabled and cfg.notch_q == 30.0
    assert cfg.normalize is True
    assert cfg.normalize_method == "robust_sigma"
    assert cfg.normalize_clip_sigma == 30.0
    assert cfg.normalize_scope == "recording"
    for key, want in _ML_RECIPES[name].items():
        assert getattr(cfg, key) == want, f"{name}: {key}"


@pytest.mark.parametrize("name", list(_ML_RECIPES))
def test_the_label_no_longer_warns_that_the_decode_is_uncalibrated(name):
    """THE INVERSE OF WHAT THIS ASSERTED, and deliberately so.

    When only the PREPROCESSING had been recovered, these three labels ended
    "— DECODE UNCALIBRATED" because a preset that invented a decode threshold and
    shipped it under the owner's project name would have been worse than one that
    said so. The decode HAS now been recovered -- it is ``DETECTOR_CONTRACT`` in
    cell 1 of each StudentInference notebook, pinned value-by-value in
    ``tests/analysis/test_ml_decode_amplitude_gates.py`` -- so the caveat is false,
    and a warning left standing after its cause is fixed only teaches the user to
    ignore the next one. The label now states the decode instead of apologising for
    its absence.
    """
    label = PRESET_LABELS[name]
    assert "UNCALIBRATED" not in label.upper()
    assert "decode" in label


@pytest.mark.parametrize("name", list(_ML_RECIPES))
def test_the_shipped_ml_presets_round_trip_through_the_panel(qtbot, name):
    """A preset the form cannot express is a preset that silently deviates the moment
    the user opens the tab."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.apply_ml_preset(name)
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS[name]
    assert panel.ml_params()["ml_preset_customized"] is False


def test_the_panel_gates_the_two_knobs_that_belong_to_one_kernel(qtbot):
    """``clip (σ)`` is read only by robust σ; mean-centring only by max-abs (robust σ
    estimates its own trimmed centre). A live switch that does nothing is the defect
    this round is removing, not adding."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.apply_ml_preset("cu_student")
    assert panel.ml_norm_clip.isEnabled()
    assert not panel.ml_norm_center.isEnabled()
    panel.apply_ml_preset("apo_nb_600mV")
    assert not panel.ml_norm_clip.isEnabled()
    assert panel.ml_norm_center.isEnabled()


# ---------------------------------------------------------------------------
# ...and that a REAL MLDetect run actually uses it
# ---------------------------------------------------------------------------
class _RecordingModel:
    """A stand-in Keras model that records every window it is shown and fires on the
    deep normalised samples, so the test can assert what reached the network."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.seen: list[np.ndarray] = []

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)          # (B, W, 1)
        self.seen.append(x.copy())
        return (np.abs(x) > 0.5).astype(np.float16)


def test_a_real_ml_run_feeds_the_model_the_robust_sigma_input(monkeypatch,
                                                              context_factory):
    """W3's real claim. Everything else here checks the CONFIG; this checks that a run
    changes what the network is shown.

    `ml_detect.ml_detect_regions` normalizes with `normalize_global`, one affine map,
    and `robust_normalize` is not affine (it clips) -- so `MLDetect.run` normalizes the
    whole span itself and hands the core a finished model input with ``normalize=False``.
    If that wiring were missing the model would silently receive the max-abs input,
    which is the exact defect the owner's notebooks replaced.
    """
    from nanoscope.analysis import ml_detect, ml_modules
    from nanoscope.analysis.filters import robust_normalize
    from nanoscope.analysis.ml_preprocess import apply_ml_preprocess

    fs = 50_000.0
    window = 1024
    rng = np.random.default_rng(7)
    n = 40_000
    x = (-1000.0 + 6.0 * rng.standard_normal(n)).astype(np.float32)
    for s in range(4_000, n, 8_000):
        x[s:s + 300] -= 260.0

    model = _RecordingModel(window)
    wrapper = ml_detect.KerasEventModel(
        model=model, window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _p: wrapper)

    cfg = PREPROCESS_PRESETS["cu_student"]
    ctx = context_factory(
        x, fs=fs,
        params={"model_path": "<fake>", "ml_preprocess": cfg.to_dict(),
                "enter": 0.5, "exit": 0.5},
        units="pA", path="cu_2h.abf")
    res = ml_modules.MLDetect().run(ctx)

    # The whole-span model input the run SHOULD have produced.
    want = robust_normalize(apply_ml_preprocess(x, fs, cfg), clip_sigma=30.0)
    assert model.seen, "the model was never called"
    first = model.seen[0][0, :, 0]
    np.testing.assert_allclose(first, want[:window], rtol=0, atol=1e-6)
    # ...and it is bounded, which the max-abs input is not by construction.
    assert float(np.abs(want).max()) <= 1.0

    prov = res.payload.provenance
    assert prov["ml_normalize_method"] == "robust_sigma"
    assert prov["ml_normalize_clip_sigma"] == 30.0
    # The core was handed normalize=False and honestly said it normalized nothing;
    # provenance must carry the TRUTH, or a stored result is not re-runnable.
    assert prov["normalized"] is True
    assert prov["norm_scale"] is None          # there is no single scale; it clips
    assert prov["ml_preprocess"]["normalize"] is True
    assert prov["ml_preprocess"]["normalize_method"] == "robust_sigma"
    assert prov["ml_preprocess_deviations"] == []   # it IS the preset, undoctored


def test_a_max_abs_ml_run_is_untouched_by_the_new_branch(monkeypatch,
                                                         context_factory):
    """The other half: the pre-existing path must not have moved. Same trace, the
    default preset, and the model sees exactly `normalize_global`'s output."""
    from nanoscope.analysis import ml_detect, ml_modules
    from nanoscope.analysis.ml_preprocess import (
        apply_ml_preprocess, normalization_center, normalization_scale,
        normalize_global)

    fs = 50_000.0
    window = 1024
    rng = np.random.default_rng(8)
    x = (-1000.0 + 6.0 * rng.standard_normal(30_000)).astype(np.float32)
    x[5_000:5_300] -= 260.0

    model = _RecordingModel(window)
    wrapper = ml_detect.KerasEventModel(
        model=model, window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _p: wrapper)

    cfg = PREPROCESS_PRESETS["apo_nb_600mV"]
    ctx = context_factory(
        x, fs=fs,
        params={"model_path": "<fake>", "ml_preprocess": cfg.to_dict()},
        units="pA", path="apo.abf")
    res = ml_modules.MLDetect().run(ctx)

    detr = apply_ml_preprocess(x, fs, cfg)
    centre = normalization_center(detr) if cfg.normalize_center else 0.0
    want = normalize_global(detr, normalization_scale(detr, centre), centre)
    np.testing.assert_allclose(model.seen[0][0, :, 0], want[:window],
                               rtol=0, atol=1e-6)
    prov = res.payload.provenance
    assert "ml_normalize_method" not in prov     # the branch never fired
    assert prov["norm_scale"] is not None


def test_the_surrogate_studies_get_a_schmitt_only_dict(qtbot):
    """REGRESSION. Sweep / null control / recovery map forward what they are handed to
    `detect_events` after stripping only the PREPROCESSING keys, so ``detector`` and
    ``cusum_*`` would arrive as unexpected keyword arguments and kill the run. They are
    Schmitt-only studies (they sweep ``k_in``), so they get a narrowed dict -- and the
    narrowing is taken from `DETECTOR_PRIVATE_KEYS`, not re-spelled here or there.
    """
    import inspect

    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.modules import DETREND_PARAM_KEYS, FILTER_PARAM_KEYS

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_detector("detect.cusum")           # even with CUSUM selected

    sp = panel.surrogate_params()
    assert "detector" not in sp
    assert not {"cusum_k", "cusum_h"} & set(sp)
    assert {"k_in", "k_out"} <= set(sp)

    # ...and what the receiver builds from it is a legal detect_events call. This is the
    # exact filter `AnalysisPanel._detect_kwargs_for_truth` applies.
    drop = {"polarity", "k_in", *DETREND_PARAM_KEYS, *FILTER_PARAM_KEYS}
    kw = {k: v for k, v in sp.items() if k not in drop and v is not None}
    assert set(kw) <= set(inspect.signature(detect_events).parameters), sorted(kw)


def test_the_detect_button_still_carries_the_detector_choice(qtbot):
    """The narrowing is for the surrogate studies ONLY -- the Detect button must still
    say which detector to run, or the owning panel cannot pick a module id."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_detector("detect.cusum")
    got: list[dict] = []
    panel.detectRequested.connect(got.append)
    panel.detect_button.click()
    assert got[0]["detector"] == "detect.cusum"

    sweeps: list[dict] = []
    panel.sweepRequested.connect(sweeps.append)
    panel.sweep_button.click()
    assert "detector" not in sweeps[0]
