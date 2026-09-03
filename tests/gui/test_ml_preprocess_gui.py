"""The ML tab's own preprocessing/decode controls (offscreen, no TensorFlow).

The ML detector runs the pipeline the model was TRAINED with, not the general
Preprocess tab chain. These tests cover the UI that makes that choosable and the
plumbing that carries it into the module:

  a. the preset combo populates every parameter widget, and switching preset
     REPOPULATES them;
  b. edited values reach ``ml_params()`` and are flagged as deviations from the
     preset (consistency with training is a correctness property, so it has to be
     visible);
  c. the config round-trips panel -> params -> ``MLPreprocessConfig`` /
     ``MLDecodeConfig`` -> the module, and lands in the result provenance;
  d. the default preset is the apo notebook's own reference bias, and the
     superseded ``apo_legacy`` key still resolves to the SAME numbers under its
     honest name (``apo_raw_global``);
  e. the model's input width defaults the preset (2048 -> a ferritin notebook,
     3072 -> an apo preset) WITHOUT overriding an explicit user choice, and without
     importing TensorFlow -- and the UI says that a width only NARROWS the choice;
  f. the normalization scope is explicit, defaults to the whole recording, and the
     whole-recording scale is what a sub-range run actually uses (the zoom bug).

Everything runs with a FAKE Keras model injected into ``ml_detect``, so no
TensorFlow is imported.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import zipfile

import numpy as np
import pytest

from nanoscope.analysis import ml_detect
from nanoscope.analysis.ml_preprocess import (
    DECODE_PRESETS,
    DEFAULT_PRESET,
    PREPROCESS_PRESETS,
    PRESET_LABELS,
    PRESET_MOLECULE,
    PRESET_NAMES,
    MLDecodeConfig,
    MLPreprocessConfig,
    apply_ml_preprocess,
    normalization_center,
    normalization_scale,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel

FS = 100_000.0
FAKE_W = 3072
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders (mirroring tests/gui/test_ml_detection.py so the two stay comparable)
# ================================================================================================
def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, *, n: int = 100_000, name: str = "pore_mlpre.dat") -> TraceDocument:
    arr = _event_trace(n)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


class _FakeEventModel:
    """Fires on deep, normalised samples; records what it was fed."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.batches: list[np.ndarray] = []

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)
        self.batches.append(x.copy())
        return (np.abs(x) > 0.5).astype(np.float16)


def _fake_wrapper(window: int = FAKE_W) -> ml_detect.KerasEventModel:
    return ml_detect.KerasEventModel(
        model=_FakeEventModel(window), window=window, path="<fake>", mtime=0.0,
        input_shape=(None, window, 1), output_shape=(None, window, 1))


def _use_fake_model(monkeypatch, wrapper=None) -> ml_detect.KerasEventModel:
    wrapper = wrapper or _fake_wrapper()
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)
    return wrapper


def _run_ml(window: MainWindow, qtbot, *, range_source: str = "whole") -> None:
    panel = window._analysis_panel
    panel._detection.ml_range.setCurrentIndex(0 if range_source == "visible" else 1)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._run_ml_detection()
    qtbot.wait(20)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _run_ml_range(window: MainWindow, qtbot, det, i0: int, i1: int) -> None:
    """Launch an ML run over an explicit sample range, carrying the panel's configs."""
    params = {
        "model_path": "/fake/model.keras",
        "enter": float(det.ml_enter.value()),
        "exit": float(det.ml_exit.value()),
        "range_source": "visible",
        "range_start": int(i0),
        "range_stop": int(i1),
        **{k: v for k, v in det.ml_params().items() if k.startswith("ml_")},
    }
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel._launch_ml(params)
    qtbot.wait(20)


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    p = DetectionPanel()
    qtbot.addWidget(p)
    return p


# ================================================================================================
# a. the preset combo populates and REPOPULATES every widget
# ================================================================================================
def test_preset_combo_lists_every_pipeline_in_order(panel):
    """Every preset is offered, in registry order. Separators carry no data, so they
    are skipped -- but they must EXIST, one per molecule boundary (see below)."""
    data = [panel.ml_preset.itemData(i) for i in range(panel.ml_preset.count())]
    assert [d for d in data if d is not None] == list(PRESET_NAMES)


def test_the_combo_separates_the_molecules(panel):
    """The list must not read as a flat menu of settings: a ferritin-trained model on
    apo data is a category error, so the molecules are visually split AND every label
    names its molecule first.

    Generalised from "the two molecules" when the owner's CB7 host-guest, Zn2-insulin and
    Cu2-insulin recipes shipped: the claim was never about the NUMBER of molecules, it was
    that a run of presets is never mixed and every boundary is drawn. Asserting a literal
    separator count made a fourth molecule fail a test about the second."""
    n = panel.ml_preset.count()
    data = [panel.ml_preset.itemData(i) for i in range(n)]
    seps = [i for i, d in enumerate(data) if d is None]

    # Split into runs at the separators; every run is one molecule, and no molecule is
    # split across two runs (which would let a mixed pair sit side by side undivided).
    runs, cur = [], []
    for d in data:
        if d is None:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(d)
    if cur:
        runs.append(cur)

    molecules = [{PRESET_MOLECULE[d] for d in run} for run in runs]
    assert all(len(m) == 1 for m in molecules), molecules      # no run mixes molecules
    flat = [next(iter(m)) for m in molecules]
    assert len(flat) == len(set(flat)), flat                   # no molecule split in two
    assert len(seps) == len(runs) - 1                          # exactly one per boundary
    assert {"ferritin", "apo-ferritin"} <= set(flat)
    for i, d in enumerate(data):
        if d is not None:
            assert panel.ml_preset.itemText(i) == PRESET_LABELS[d]


def test_the_panel_states_the_selected_presets_molecule(panel):
    panel.apply_ml_preset("notebook2")
    assert "FERRITIN" in panel.ml_preset_molecule.text()
    assert "APO" not in panel.ml_preset_molecule.text()
    panel.apply_ml_preset("apo_nb_700mV")
    assert "APO-FERRITIN" in panel.ml_preset_molecule.text()


def test_default_preset_is_the_apo_notebook_reference_bias(panel):
    """An untouched panel runs the pipeline that was actually executed and validated
    -- the apo notebook at its own reference bias (-600 mV)."""
    assert DEFAULT_PRESET == "apo_nb_600mV"
    assert panel.ml_preset_name() == DEFAULT_PRESET
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS[DEFAULT_PRESET]
    assert panel.ml_decode_config() == DECODE_PRESETS[DEFAULT_PRESET]
    # A single threshold, no probability smoothing, no merge inside the decoder.
    assert panel.ml_smoothing.value() == 1
    assert panel.ml_enter.value() == pytest.approx(0.1)
    assert panel.ml_exit.value() == pytest.approx(0.1)
    assert panel.ml_merge_ms.value() == pytest.approx(0.0)
    assert panel.ml_min_event_ms.value() == pytest.approx(1.0)
    assert "matches" in panel.ml_preset_status.text()


def test_apo_raw_global_still_reproduces_todays_parameters(panel):
    """The superseded preset is still selectable and still numerically what it was --
    it IS the training contract of a model the user holds."""
    panel.apply_ml_preset("apo_raw_global")
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS["apo_raw_global"]
    assert panel.ml_decode_config() == DECODE_PRESETS["apo_raw_global"]
    assert panel.ml_enter.value() == pytest.approx(ml_detect.DEFAULT_ENTER)
    assert panel.ml_exit.value() == pytest.approx(ml_detect.DEFAULT_EXIT)
    assert panel.ml_smoothing.value() == ml_detect.DEFAULT_SMOOTH_KERNEL
    assert panel.ml_merge_ms.value() / 1000.0 == pytest.approx(ml_detect.DEFAULT_MERGE_S)
    assert panel.ml_min_event_ms.value() / 1000.0 == pytest.approx(ml_detect.DEFAULT_MIN_DUR_S)
    assert "matches" in panel.ml_preset_status.text()


def test_the_old_apo_legacy_key_still_selects_the_same_numbers(panel):
    """A restored run naming the superseded key must not raise, and must NOT be
    reinterpreted as one of the apo notebook presets."""
    panel.apply_ml_preset("apo_legacy")
    assert panel.ml_preset_name() == "apo_raw_global"
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS["apo_raw_global"]
    assert panel.ml_decode_config() == DECODE_PRESETS["apo_raw_global"]
    assert panel.ml_norm_center.isChecked() is False
    assert panel.ml_anchor_tail.isChecked() is True
    with pytest.raises(ValueError, match="unknown ML preset"):
        panel.apply_ml_preset("no_such_preset")


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_selecting_a_preset_populates_every_widget_from_it(panel, name):
    panel.apply_ml_preset(name)
    assert panel.ml_preset_name() == name
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS[name]
    assert panel.ml_decode_config() == DECODE_PRESETS[name]
    assert "matches" in panel.ml_preset_status.text()


def test_switching_preset_repopulates_widgets_and_clears_deviations(panel):
    panel.apply_ml_preset("notebook2")
    assert panel.ml_baseline_method.currentData() == "masked_median"
    assert panel.ml_masked_rough_s.value() == pytest.approx(0.5)
    assert panel.ml_masked_base_s.value() == pytest.approx(0.1)
    assert panel.ml_baseline_decim.value() == 50
    assert panel.ml_despike.value() == 3

    # Deviate...
    panel.ml_masked_base_s.setValue(0.42)
    assert "CUSTOM" in panel.ml_preset_status.text()
    assert "masked_base_s" in panel.ml_preset_status.text()

    # ...then switch preset: every widget is overwritten and the flag clears.
    panel.apply_ml_preset("notebook3")
    assert panel.ml_masked_rough_s.value() == pytest.approx(0.03)
    assert panel.ml_masked_base_s.value() == pytest.approx(0.01)
    assert panel.ml_baseline_decim.value() == 10
    assert panel.ml_despike.value() == 0
    assert panel.ml_smoothing.value() == 5
    assert panel.ml_onset_sigma.value() == pytest.approx(2.0)
    assert panel.ml_offset_sigma.value() == pytest.approx(3.0)
    assert "matches" in panel.ml_preset_status.text()
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS["notebook3"]


def test_baseline_rows_follow_the_selected_estimator(panel):
    """Only the parameters the chosen estimator actually reads are shown."""
    form = panel._ml_preprocess_form

    def _visible(w):
        idx, _role = form.getWidgetPosition(w)
        return form.isRowVisible(idx)

    panel.apply_ml_preset("notebook1")            # robust median
    assert _visible(panel.ml_baseline_window_s) and _visible(panel.ml_baseline_decim)
    assert not _visible(panel.ml_masked_rough_s)
    assert not _visible(panel.ml_baseline_poly)

    panel.apply_ml_preset("notebook2")            # masked median
    assert _visible(panel.ml_masked_rough_s) and _visible(panel.ml_masked_base_s)
    assert _visible(panel.ml_mask_sigma) and _visible(panel.ml_dilate_ms)
    assert not _visible(panel.ml_baseline_poly)

    panel.apply_ml_preset("apo_legacy")           # savgol
    assert _visible(panel.ml_baseline_window_s) and _visible(panel.ml_baseline_poly)
    assert not _visible(panel.ml_masked_rough_s)


def test_group_states_that_the_preprocess_tab_is_not_used(panel):
    text = panel.ml_pipeline_note.text()
    assert "not the Preprocess tab" in text
    assert panel.ml_preprocess_box.title() == "Model preprocessing"
    assert panel.ml_decode_box.title() == "Model decode"


def test_ml_widgets_live_on_the_ml_page(panel):
    """The new controls belong to the ML tab, not to Preprocess/Detection."""
    for attr in ("ml_preset", "ml_preset_molecule", "ml_notch_enabled",
                 "ml_notch_freqs", "ml_notch_q", "ml_notch_float32",
                 "ml_baseline_method", "ml_baseline_window_s", "ml_baseline_decim",
                 "ml_masked_rough_s", "ml_masked_base_s", "ml_mask_sigma",
                 "ml_dilate_ms", "ml_despike", "ml_norm_enabled", "ml_norm_center",
                 "ml_norm_scope", "ml_anchor_tail", "ml_last_start_inclusive",
                 "ml_smoothing", "ml_enter", "ml_exit",
                 "ml_merge_ms", "ml_min_event_ms", "ml_onset_sigma",
                 "ml_onset_max_ms", "ml_onset_gap_tol", "ml_offset_sigma"):
        w = getattr(panel, attr)
        seen = set()
        while w is not None and w not in (panel._ml_page, panel._preprocess_page,
                                          panel._detection_page):
            if id(w) in seen:
                break
            seen.add(id(w))
            w = w.parentWidget()
        assert w is panel._ml_page, f"{attr} is not on the ML page"


# ================================================================================================
# b. edited values reach ml_params(), flagged as deviations
# ================================================================================================
def test_edited_values_reach_ml_params(panel):
    panel.apply_ml_preset("notebook1")
    panel.ml_baseline_window_s.setValue(0.75)
    panel.ml_baseline_decim.setValue(25)
    panel.ml_despike.setValue(5)
    panel.ml_notch_q.setValue(45.0)
    panel.ml_notch_freqs.setText("50, 100")
    panel.ml_smoothing.setValue(15)
    panel.ml_merge_ms.setValue(1.25)
    panel.ml_min_event_ms.setValue(2.5)
    panel.ml_onset_sigma.setValue(1.5)
    panel.ml_offset_sigma.setValue(2.5)
    panel.ml_norm_scope.setCurrentIndex(panel.ml_norm_scope.findData("range"))
    panel.ml_anchor_tail.setChecked(True)

    p = panel.ml_params()
    pre = p["ml_preprocess"]
    dec = p["ml_decode"]
    assert pre["baseline_window_s"] == pytest.approx(0.75)
    assert pre["baseline_decim"] == 25
    assert pre["despike_size"] == 5
    assert pre["notch_q"] == pytest.approx(45.0)
    assert pre["notch_freqs"] == [50.0, 100.0]
    assert pre["normalize_scope"] == "range"
    assert pre["anchor_tail"] is True
    assert dec["smoothing_kernel"] == 15
    assert dec["merge_radius_ms"] == pytest.approx(1.25)
    assert dec["min_event_ms"] == pytest.approx(2.5)
    assert dec["onset_sigma"] == pytest.approx(1.5)
    assert dec["offset_sigma"] == pytest.approx(2.5)
    # The preset NAME is kept, with the deviations named -- not flattened to "custom".
    assert p["ml_preset"] == "notebook1"
    assert p["ml_preset_customized"] is True
    assert "baseline_window_s" in p["ml_preset_deviations"]
    assert "smoothing_kernel" in p["ml_preset_deviations"]


def test_the_apo_only_knobs_round_trip_and_flag_as_deviations(panel):
    """mean-centring, the float32 notch round and the start bound are user-editable
    like every other parameter, and turning one on inside a FERRITIN preset must be
    reported as a deviation -- that combination is exactly the mistake to catch."""
    panel.apply_ml_preset("notebook1")
    assert panel.ml_norm_center.isChecked() is False
    assert panel.ml_notch_float32.isChecked() is False
    assert panel.ml_last_start_inclusive.isChecked() is False

    panel.ml_norm_center.setChecked(True)
    panel.ml_notch_float32.setChecked(True)
    panel.ml_last_start_inclusive.setChecked(True)

    pre = panel.ml_params()["ml_preprocess"]
    assert pre["normalize_center"] is True
    assert pre["notch_float32"] is True
    assert pre["last_start_inclusive"] is True
    assert MLPreprocessConfig.from_dict(pre) == panel.ml_preprocess_config()

    status = panel.ml_preset_status.text()
    assert "CUSTOM" in status and "3 change(s)" in status
    devs = panel.ml_params()["ml_preset_deviations"]
    assert set(devs) == {"notch_float32", "normalize_center", "last_start_inclusive"}


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_round_trips_through_ml_params(panel, name):
    """panel -> params dict -> config must be the preset itself, for all of them."""
    panel.apply_ml_preset(name)
    p = panel.ml_params()
    assert p["ml_preset"] == name
    assert p["ml_preset_customized"] is False
    assert p["ml_preset_deviations"] == []
    assert MLPreprocessConfig.from_dict(p["ml_preprocess"]) == PREPROCESS_PRESETS[name]
    assert MLDecodeConfig.from_dict(p["ml_decode"]) == DECODE_PRESETS[name]


def test_unedited_preset_reports_no_deviation(panel):
    panel.apply_ml_preset("notebook2")
    p = panel.ml_params()
    assert p["ml_preset_customized"] is False
    assert p["ml_preset_deviations"] == []


def test_notch_line_box_parses_and_falls_back_on_a_typo(panel):
    panel.ml_notch_freqs.setText("60;120 / 180")
    assert panel.ml_preprocess_config().notch_freqs == (60.0, 120.0, 180.0)
    panel.ml_notch_freqs.setText("fifty, -3, 0")
    # Unparseable / non-positive entries are dropped; an empty result falls back to
    # the trained lines rather than silently running un-notched.
    assert panel.ml_preprocess_config().notch_freqs == (60.0, 120.0, 180.0)
    panel.ml_notch_freqs.setText("50")
    assert panel.ml_preprocess_config().notch_freqs == (50.0,)


def test_ml_params_are_json_serialisable(panel):
    """The configs ride in AnalysisResult.params, so they must survive a JSON trip."""
    panel.apply_ml_preset("notebook3")
    p = panel.ml_params()
    p["model_path"] = "/x/m.keras"
    restored = json.loads(json.dumps(p))
    assert MLPreprocessConfig.from_dict(restored["ml_preprocess"]) \
        == PREPROCESS_PRESETS["notebook3"]
    assert MLDecodeConfig.from_dict(restored["ml_decode"]) == DECODE_PRESETS["notebook3"]


# ================================================================================================
# e. model input width defaults the preset (no TensorFlow involved)
# ================================================================================================
def _fake_keras_file(tmp_path, window: int, name: str = "m.keras"):
    """A minimal .keras zip carrying only the input shape peek_model_window reads."""
    path = tmp_path / name
    cfg = {"class_name": "Functional", "config": {"layers": [
        {"class_name": "InputLayer", "config": {"batch_shape": [None, window, 1]}}]}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps(cfg))
        z.writestr("metadata.json", "{}")
    return path


def test_peek_model_window_reads_the_width_without_tensorflow(tmp_path):
    assert ml_detect.peek_model_window(_fake_keras_file(tmp_path, 2048)) == 2048
    assert ml_detect.peek_model_window(
        _fake_keras_file(tmp_path, 3072, "b.keras")) == 3072
    # Anything unreadable is "unknown", never an exception.
    assert ml_detect.peek_model_window(tmp_path / "nope.keras") is None
    plain = tmp_path / "plain.keras"
    plain.write_bytes(b"not a zip")
    assert ml_detect.peek_model_window(plain) is None


def test_loading_a_2048_model_defaults_to_a_ferritin_notebook_preset(panel, tmp_path):
    assert panel.ml_preset_name() == DEFAULT_PRESET
    panel.set_model_path(str(_fake_keras_file(tmp_path, 2048)))
    assert panel.model_input_width() == 2048
    assert panel.ml_preset_name() == "notebook1"
    assert panel.ml_preprocess_config() == PREPROCESS_PRESETS["notebook1"]


def test_loading_a_3072_model_defaults_to_the_apo_notebook_preset(panel, tmp_path):
    panel.apply_ml_preset("notebook2")          # programmatic, not a user choice
    panel.set_model_path(str(_fake_keras_file(tmp_path, 3072)))
    assert panel.ml_preset_name() == DEFAULT_PRESET
    assert PRESET_MOLECULE[panel.ml_preset_name()] == "apo-ferritin"


def test_the_width_note_says_a_width_cannot_identify_a_pipeline(panel, tmp_path):
    """Both apo models are 3072 and an apo student also exists at 2048, so the guess
    must be surfaced as "narrowed", never as "matched"."""
    panel.set_model_path(str(_fake_keras_file(tmp_path, 3072)))
    note = panel.ml_pipeline_note.text()
    assert "3072" in note and "BOTH apo pipelines" in note
    assert "raw-global" in note

    panel.set_model_path(str(_fake_keras_file(tmp_path, 2048, "w2048.keras")))
    note = panel.ml_pipeline_note.text()
    assert "2048" in note and "all three ferritin presets" in note
    assert "does not even settle the molecule" in note


def test_an_explicit_user_choice_is_never_overridden(panel, tmp_path):
    panel.apply_ml_preset("notebook3", user_choice=True)
    panel.set_model_path(str(_fake_keras_file(tmp_path, 3072)))
    assert panel.model_input_width() == 3072
    assert panel.ml_preset_name() == "notebook3"      # untouched


def test_an_unknown_width_changes_nothing(panel, tmp_path):
    panel.apply_ml_preset("notebook2")
    panel.set_model_path(str(_fake_keras_file(tmp_path, 999)))
    assert panel.ml_preset_name() == "notebook2"
    panel.set_model_path("/not/a/real/model.keras")
    assert panel.ml_preset_name() == "notebook2"


# ================================================================================================
# c. the config round-trips into the module and lands in provenance
# ================================================================================================
def test_config_reaches_the_module_and_is_stamped_into_provenance(
    window, qtbot, tmp_path, monkeypatch
):
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("notebook2", user_choice=True)
    det.ml_masked_base_s.setValue(0.2)          # one deliberate deviation

    _run_ml(window, qtbot, range_source="whole")

    res = window._session.results.get(doc_id, channel, "detect.keras")
    assert res is not None
    # The params the module was given carry the whole contract...
    assert res.params["ml_preset"] == "notebook2"
    assert MLPreprocessConfig.from_dict(res.params["ml_preprocess"]).masked_base_s \
        == pytest.approx(0.2)
    # ...and the provenance stamps every effective parameter, so the run is auditable.
    prov = res.payload.provenance
    assert prov["ml_preset"] == "notebook2"
    assert prov["ml_preprocess"]["baseline_method"] == "masked_median"
    assert prov["ml_preprocess"]["masked_base_s"] == pytest.approx(0.2)
    assert prov["ml_decode"]["smoothing_kernel"] == 9
    assert prov["ml_decode_preset"] == "notebook2"
    assert "masked median" in prov["ml_preprocess_desc"]
    assert prov["ml_preset_customized"] is True
    assert "masked_base_s" in prov["ml_preprocess_deviations"]
    assert prov["norm_scope"] == "recording"
    assert prov["anchor_tail"] is False        # notebook windowing, tail uncovered
    assert np.isfinite(prov["norm_scale"]) and prov["norm_scale"] > 0


def test_apo_raw_global_run_reproduces_todays_provenance_parameters(
    window, qtbot, tmp_path, monkeypatch
):
    """The superseded preset is a genuine no-op: same decode numbers, same windowing.
    Selected by its OLD key, to prove a restored run lands on the same contract."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("apo_legacy", user_choice=True)
    _run_ml(window, qtbot, range_source="whole")

    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["ml_preset"] == "apo_raw_global"
    assert prov["enter"] == pytest.approx(ml_detect.DEFAULT_ENTER)
    assert prov["exit"] == pytest.approx(ml_detect.DEFAULT_EXIT)
    assert prov["smooth_kernel"] == ml_detect.DEFAULT_SMOOTH_KERNEL
    assert prov["merge_s"] == pytest.approx(ml_detect.DEFAULT_MERGE_S)
    assert prov["min_dur_s"] == pytest.approx(ml_detect.DEFAULT_MIN_DUR_S)
    assert prov["anchor_tail"] is True
    assert prov["norm_center"] == 0.0            # NOT the apo notebook's mean-centring
    assert prov["ml_preset_customized"] is False
    assert "SavGol" in prov["ml_preprocess_desc"]


def test_apo_notebook_run_stamps_the_centre_and_the_tail_policy(
    window, qtbot, tmp_path, monkeypatch
):
    """The two apo-only choices must be visible in provenance, and the centre must be
    the whole recording's mean -- the notebook computes nothing per-chunk."""
    _use_fake_model(monkeypatch)
    doc = _synth_doc(tmp_path)
    doc_id, channel = _install(window, doc)
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("apo_nb_600mV", user_choice=True)
    _run_ml(window, qtbot, range_source="whole")

    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["ml_preset"] == "apo_nb_600mV"
    assert prov["ml_decode_preset"] == "apo_nb_600mV"
    assert prov["anchor_tail"] is False
    assert prov["last_start_inclusive"] is True
    assert prov["ml_preprocess"]["notch_float32"] is True
    assert prov["ml_preprocess"]["normalize_center"] is True
    assert "(x-mean)/(max|x-mean|+1e-8)" in prov["ml_preprocess_desc"]
    assert "arange(0, n-W+1, step)" in prov["inference"]

    cfg = MLPreprocessConfig.from_dict(prov["ml_preprocess"])
    full = apply_ml_preprocess(np.asarray(doc.channels[0], np.float32), FS, cfg)
    assert prov["norm_center"] == pytest.approx(normalization_center(full), rel=1e-9)
    assert prov["norm_center"] != 0.0
    assert prov["norm_scale"] == pytest.approx(
        normalization_scale(full, prov["norm_center"]), rel=1e-9)


def test_rerun_preserves_the_pipeline_config(window, qtbot, tmp_path, monkeypatch):
    """Re-running after a calibration change must not silently revert to the default
    pipeline -- the stored params carry the configs."""
    from nanoscope.core.calibration import ChannelCalibration

    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("notebook1", user_choice=True)
    _run_ml(window, qtbot, range_source="whole")

    window._session.set_calibration(
        doc_id, channel,
        ChannelCalibration(gain=2.0, offset=0.0, target_units="pA",
                           provenance=Provenance.UNKNOWN))
    qtbot.wait(10)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel._on_rerun()
    qtbot.wait(20)

    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["ml_preset"] == "notebook1"
    assert prov["ml_preprocess"]["baseline_method"] == "robust_median"


def test_ml_run_does_not_use_the_general_preprocess_chain(
    window, qtbot, tmp_path, monkeypatch
):
    """Turning every Preprocess-tab stage on must not change what the model is fed."""
    import nanoscope.analysis.modules as modules_mod
    import nanoscope.gui.analysis_panel as panel_mod

    wrapper = _use_fake_model(monkeypatch)
    calls: list[int] = []
    real = modules_mod.preprocess_for_detection

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    # Patch BOTH the definition site and the already-bound name in the GUI module,
    # so a call through either import path is caught.
    monkeypatch.setattr(modules_mod, "preprocess_for_detection", _spy)
    monkeypatch.setattr(panel_mod, "preprocess_for_detection", _spy)

    doc_id, channel = _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.set_notch_enabled(True)
    det.set_hp_enabled(True)
    det.set_savgol_enabled(True)
    det.detrend_enabled.setChecked(True)

    _run_ml(window, qtbot, range_source="whole")
    assert not calls, "the ML detector must not run preprocess_for_detection"
    assert wrapper.model.batches, "the fake model was never called"


# ================================================================================================
# f. the normalization-scope fix
# ================================================================================================
def test_normalization_scope_defaults_to_the_whole_recording(panel):
    assert panel.ml_norm_scope.currentData() == "recording"
    for name in PRESET_NAMES:
        panel.apply_ml_preset(name)
        assert panel.ml_preprocess_config().normalize_scope == "recording"


def test_sub_range_run_normalizes_by_the_whole_recording_scale(
    window, qtbot, tmp_path, monkeypatch
):
    """The zoom bug: max|x| used to come from the analysed slice.

    With scope='recording' a sub-range run divides by the WHOLE channel's max|x| --
    the scale training used -- so the model's input no longer depends on the zoom.
    """
    wrapper = _use_fake_model(monkeypatch)
    doc = _synth_doc(tmp_path)
    doc_id, channel = _install(window, doc)
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")

    # A quiet sub-range: no event inside it, so its own max|x| is much smaller than
    # the recording's.
    i0, i1 = 20_000, 28_000
    _run_ml_range(window, qtbot, det, i0, i1)

    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["norm_scope"] == "recording"
    assert prov["ml_preprocess_scope"] == "recording"

    cfg = MLPreprocessConfig.from_dict(prov["ml_preprocess"])
    full = apply_ml_preprocess(np.asarray(doc.channels[0], np.float32), FS, cfg)
    # The centre is a whole-recording statistic too, and it is taken FIRST -- the
    # scale is max|x - mean| about it, not about zero.
    center = normalization_center(full) if cfg.normalize_center else 0.0
    whole_scale = normalization_scale(full, center)
    slice_scale = normalization_scale(full[i0:i1], center)
    assert slice_scale < whole_scale                  # the two really differ
    assert prov["norm_center"] == pytest.approx(center, rel=1e-9, abs=1e-12)
    assert prov["norm_scale"] == pytest.approx(whole_scale, rel=1e-6)

    # And the model was actually fed windows scaled that way: the quiet range never
    # reaches full scale, which it would if it had been normalised by its own max.
    fed = np.concatenate([b.ravel() for b in wrapper.model.batches])
    assert float(np.max(np.abs(fed))) < 0.95


def test_range_scope_reproduces_the_old_slice_local_normalization(
    window, qtbot, tmp_path, monkeypatch
):
    wrapper = _use_fake_model(monkeypatch)
    doc = _synth_doc(tmp_path)
    doc_id, channel = _install(window, doc)
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.ml_norm_scope.setCurrentIndex(det.ml_norm_scope.findData("range"))

    i0, i1 = 20_000, 28_000
    _run_ml_range(window, qtbot, det, i0, i1)

    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["norm_scope"] == "range"
    assert prov["ml_preprocess_scope"] == "range"
    # Slice-local scaling drives the quiet window all the way to full scale.
    fed = np.concatenate([b.ravel() for b in wrapper.model.batches])
    assert float(np.max(np.abs(fed))) == pytest.approx(1.0, abs=1e-3)


def test_whole_recording_run_reports_recording_scope(window, qtbot, tmp_path, monkeypatch):
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    window._analysis_panel._detection.set_model_path("/fake/model.keras")
    _run_ml(window, qtbot, range_source="whole")
    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["norm_scope"] == "recording"
    # The slice IS the recording, so the baseline saw the whole file too.
    assert prov["ml_preprocess_scope"] == "recording"


def test_a_sub_range_run_is_fed_the_whole_recordings_detrend_bit_for_bit(
    window, qtbot, tmp_path, monkeypatch
):
    """The scope fix is not "a similar scale", it is the SAME BYTES.

    ``norm_scale``/``norm_center`` being approximately right is not enough: the
    SavGol envelope and the masked-median MAD are span-dependent too, so the slice
    the model sees must be cut out of the whole-recording detrend rather than
    re-derived from the slice. Approximate assertions cannot tell those two apart --
    they agree to several digits in the interior and diverge only near the cut, which
    is exactly where windowed inference reads. So compare bytes.

    Verified against the real -600 mV apo recording, where a range-scoped centre
    would have been -0.0223 instead of the recording's -0.00313 and the scale 248.8
    instead of 260.5 -- a 4.5 % error in the model's input amplitude.
    """
    wrapper = _use_fake_model(monkeypatch)
    doc = _synth_doc(tmp_path)
    doc_id, channel = _install(window, doc)
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("apo_nb_600mV", user_choice=True)

    i0, i1 = 20_000, 60_000
    _run_ml_range(window, qtbot, det, i0, i1)
    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance

    cfg = PREPROCESS_PRESETS["apo_nb_600mV"]
    full = apply_ml_preprocess(doc.channels[channel], FS, cfg)
    center = normalization_center(full)
    scale = normalization_scale(full, center)
    # Bit-for-bit, not approx: these are the exact float32/float64 values the
    # notebook's own whole-recording pass produces.
    assert prov["norm_center"] == center
    assert prov["norm_scale"] == scale
    assert prov["norm_scope"] == "recording"

    # And the model really was fed (full[i0:i1] - center) / scale, byte for byte.
    # ``batches`` holds OVERLAPPING windows (stride W//2), not the signal laid end to
    # end, so compare window k against its own slice rather than concatenating.
    want = ((full[i0:i1] - np.float32(center)).astype(np.float32) / scale).astype(np.float32)
    fed = np.concatenate([b.reshape(-1, FAKE_W) for b in wrapper.model.batches])
    step = FAKE_W // 2
    assert len(fed) >= 3
    for k in range(3):
        got = np.ascontiguousarray(fed[k], dtype=np.float32)
        exp = np.ascontiguousarray(want[k * step: k * step + FAKE_W])
        assert np.array_equal(got.view(np.uint8), exp.view(np.uint8)), f"window {k}"


def test_the_notebook_tail_convention_blinds_the_end_of_an_analysed_sub_range(
    window, qtbot, tmp_path, monkeypatch
):
    """The apo/ferritin loops leave the SLICE's tail uncovered, not just the file's.

    ``n`` in ``arange(0, n - W + 1, step)`` is the length of the analysed range, so
    with ``anchor_tail=False`` the last ``(n - W) % step`` samples of every sub-range
    sit at probability 0 -- up to ``step - 1`` = 1535 samples, 30.7 ms at 50 kHz. An
    event straddling the end of a "Visible view" range therefore does not come back
    truncated, it does not come back at all.

    That is the NOTEBOOK'S convention faithfully reproduced, not a scope bug (the
    test above pins the scope), and it is why this is asserted rather than fixed --
    ``anchor_tail`` is the knob that changes it, and the notebook presets set it
    False on purpose. Measured on the real -600 mV recording: the event at
    1736334..1736999 is found by a whole-file run and absent from a run over
    [1600000, 1736600). Reproduced here on the synthetic doc.
    """
    _use_fake_model(monkeypatch)
    doc = _synth_doc(tmp_path)
    doc_id, channel = _install(window, doc)
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("apo_nb_600mV", user_choice=True)
    det.ml_min_event_ms.setValue(0.5)

    straddled = _EVENTS[2]                       # (50_000, 50_400)
    i0 = 10_000
    W, step = FAKE_W, FAKE_W // 2
    # Cut mid-event, and land the slice length so the event sits in the blind tail.
    i1 = straddled[0] + 200
    n = i1 - i0
    blind_from = i0 + ((n - W) // step) * step + W
    assert blind_from <= straddled[0] < i1, "fixture must put the event in the tail"

    _run_ml_range(window, qtbot, det, i0, i1)
    res = window._session.results.get(doc_id, channel, "detect.keras")
    starts = res.payload.df["start_idx"].to_numpy()
    assert not ((starts >= blind_from) & (starts < i1)).any(), (
        f"samples [{blind_from}, {i1}) are the uncovered tail; nothing can be "
        "detected there under the notebook window loop")
    assert res.payload.provenance["anchor_tail"] is False
    assert res.payload.provenance["last_start_inclusive"] is True
    assert "arange(0, n-W+1, step)" in res.payload.provenance["inference"]

    # NON-VACUITY: anchoring the tail is the documented way to cover it, and doing so
    # DOES surface the straddled event. Without this half, the assertion above would
    # pass just as happily if there were nothing there to find.
    det.ml_anchor_tail.setChecked(True)
    _run_ml_range(window, qtbot, det, i0, i1)
    res2 = window._session.results.get(doc_id, channel, "detect.keras")
    assert res2.payload.provenance["anchor_tail"] is True
    starts2 = res2.payload.df["start_idx"].to_numpy()
    assert ((starts2 >= blind_from) & (starts2 < i1)).any(), (
        "the anchored run must find the event the notebook loop is blind to, "
        "otherwise the absence above proves nothing")


# ================================================================================================
# g. A decode preset carried in params must NOT have its thresholds silently
#    replaced by the legacy apo defaults when the caller sent no top-level
#    enter/exit. The GUI always sends them, but a stored result re-run from its own
#    params -- or any programmatic caller -- does not, and defaulting them to
#    0.5/0.3 overwrote every notebook preset's exit (0.15 / 0.15 / 0.20).
# ================================================================================================
@pytest.mark.parametrize(
    "preset,exit_thr", [("notebook1", 0.15), ("notebook2", 0.15), ("notebook3", 0.20)]
)
def test_decode_preset_thresholds_survive_absent_top_level_enter_exit(preset, exit_thr):
    from nanoscope.analysis.ml_modules import _resolve_configs
    from nanoscope.analysis.ml_preprocess import DECODE_PRESETS, PREPROCESS_PRESETS

    params = {
        "ml_preprocess": PREPROCESS_PRESETS[preset].to_dict(),
        "ml_decode": DECODE_PRESETS[preset].to_dict(),
    }
    # exactly how MLDetect.run resolves them when the keys are missing
    enter = float(params.get("enter", ml_detect.DEFAULT_ENTER))
    exit_default = float(params.get("exit", ml_detect.DEFAULT_EXIT))
    assert exit_default == 0.3                      # the legacy fallback is in play

    _pre, dec = _resolve_configs(params, enter, exit_default)
    assert dec.exit_threshold == pytest.approx(exit_thr)
    assert dec.enter_threshold == pytest.approx(DECODE_PRESETS[preset].enter_threshold)
    assert dec.deviations_from(DECODE_PRESETS[preset]) == ()


def test_explicit_top_level_enter_exit_still_win_over_the_preset():
    """The panel's headline spins remain authoritative when they ARE sent."""
    from nanoscope.analysis.ml_modules import _resolve_configs
    from nanoscope.analysis.ml_preprocess import DECODE_PRESETS

    params = {"ml_decode": DECODE_PRESETS["notebook1"].to_dict(), "enter": 0.8, "exit": 0.4}
    _pre, dec = _resolve_configs(params, 0.8, 0.4)
    assert (dec.enter_threshold, dec.exit_threshold) == (0.8, 0.4)
    assert set(dec.deviations_from(DECODE_PRESETS["notebook1"])) == {
        "enter_threshold", "exit_threshold"}


def test_provenance_reports_the_decode_thresholds_that_actually_ran(
    window, qtbot, tmp_path, monkeypatch
):
    """MLDetect._provenance writes enter/exit LAST, over the DSP core's. It must
    write the resolved decode config's values, not the params fallbacks."""
    _use_fake_model(monkeypatch)
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.set_model_path("/fake/model.keras")
    det.apply_ml_preset("notebook1", user_choice=True)

    _run_ml(window, qtbot, range_source="whole")
    prov = window._session.results.get(doc_id, channel, "detect.keras").payload.provenance
    assert prov["exit"] == pytest.approx(0.15)      # notebook1's exit, not 0.3
    assert prov["enter"] == pytest.approx(0.5)
    assert prov["ml_decode"]["exit_threshold"] == pytest.approx(0.15)

# ================================================================================================
# owner note #7: every model loaded this session is switchable from the combo
# ================================================================================================

def test_loaded_models_accumulate_in_the_combo_and_switch_back(qapp, qtbot, tmp_path):
    from nanoscope.gui.panels import DetectionPanel
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    combo = panel.ml_model_combo
    assert not combo.isEnabled(), "empty state: nothing loaded yet"

    a = str(_fake_keras_file(tmp_path, 2048, "student_a.keras"))
    b = str(_fake_keras_file(tmp_path, 3072, "student_b.keras"))
    panel.set_model_path(a)
    panel.set_model_path(b)
    assert combo.isEnabled()
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "student_a.keras", "student_b.keras"]
    assert panel.model_path() == b
    assert combo.currentIndex() == 1

    # Switching back through the combo IS a model switch — path, label, input width.
    combo.setCurrentIndex(0)
    assert panel.model_path() == a
    assert panel.ml_model_label.text() == "student_a.keras"

    # Re-loading a model already listed selects it instead of duplicating it.
    panel.set_model_path(b)
    assert combo.count() == 2 and combo.currentIndex() == 1


# ===========================================================================
# g. the three amplitude gates on the decode form
# ===========================================================================
def test_the_decode_form_can_express_the_three_amplitude_gates(panel):
    """A preset the form cannot express is a preset that silently deviates the moment
    the user opens the tab — and ``cu_student`` is the first preset that ships any of
    these gates on, so the widgets have to exist before its contract can land."""
    panel.apply_ml_preset("cu_student")
    assert panel.ml_min_depth_sigma.value() == pytest.approx(4.0)
    assert panel.ml_min_depth_pa.value() == pytest.approx(10.0)
    assert panel.ml_min_fracblock.value() == pytest.approx(0.05)
    assert panel.ml_decode_config() == DECODE_PRESETS["cu_student"]
    assert panel.ml_params()["ml_preset_customized"] is False


def test_switching_from_cu_to_a_gateless_preset_clears_the_gates(panel):
    """The gates are per-preset, so a stale 10 pA floor left over from Cu would
    silently re-gate the next recording. Switching must repopulate, not merge."""
    panel.apply_ml_preset("cu_student")
    panel.apply_ml_preset("apo_nb_600mV")
    assert panel.ml_min_depth_sigma.value() == 0.0
    assert panel.ml_min_depth_pa.value() == 0.0
    assert panel.ml_min_fracblock.value() == 0.0
    assert panel.ml_decode_config() == DECODE_PRESETS["apo_nb_600mV"]


def test_an_edited_gate_reaches_ml_params_and_reads_as_a_deviation(panel):
    """Changing a gate changes which events exist, so it is a correctness risk and
    has to be visible as a deviation from the preset — same as every other field."""
    panel.apply_ml_preset("zn_student")
    panel.ml_min_depth_pa.setValue(25.0)
    p = panel.ml_params()
    assert p["ml_decode"]["min_depth_pa"] == pytest.approx(25.0)
    assert p["ml_preset_customized"] is True
    assert "min_depth_pa" in p["ml_preset_deviations"]


def test_the_gates_default_to_off_on_an_untouched_panel(panel):
    """The default preset ships them off, and an untouched form must reproduce the
    decode this app has always run."""
    assert panel.ml_min_depth_sigma.value() == 0.0
    assert panel.ml_min_depth_pa.value() == 0.0
    assert panel.ml_min_fracblock.value() == 0.0
    assert panel.ml_min_depth_sigma.specialValueText() == "off"


def test_the_robust_sigma_path_gates_on_PICOAMPS_not_on_the_model_input(
    window, qtbot, monkeypatch, tmp_path
):
    """THE unit trap in this whole feature, and the reason ``gate_signal`` exists.

    ``cu_student`` normalises with ``robust_sigma``, which is not affine — it
    re-estimates a trimmed median/MAD and then CLIPS — so the adapter cannot hand the
    DSP core a (centre, scale) pair and instead hands it a FINISHED model input.
    That trace is unitless and saturated. Its ``min_depth_pa = 10.0`` and
    ``min_fracblock = 0.05`` are picoamps and picoamps-over-picoamps; read against
    the model input they would keep or drop events essentially at random.

    So the pre-normalization detrended pA trace has to reach the gate separately.
    This asserts it does, by scale: the real trace has ~30 pA events on ~1.5 pA
    noise, while a robust-sigma model input is bounded by its clip.
    """
    seen: dict[str, np.ndarray] = {}
    real = ml_detect.decode_ml_regions

    def _spy(prob, fs, cfg, *, signal=None, gate_signal=None, raw=None):
        seen["signal"] = np.asarray(signal)
        seen["gate"] = np.asarray(gate_signal)
        seen["raw"] = np.asarray(raw)
        return real(prob, fs, cfg, signal=signal, gate_signal=gate_signal, raw=raw)

    monkeypatch.setattr(ml_detect, "decode_ml_regions", _spy)
    _use_fake_model(monkeypatch)
    _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    # Model FIRST, preset second: loading a model defaults the preset from its input
    # width, so applying the preset first would have it silently replaced here.
    det.set_model_path(str(_fake_keras_file(tmp_path, FAKE_W, "cu.keras")))
    det.apply_ml_preset("cu_student")
    _run_ml(window, qtbot)

    # The snap trace is the model input: unitless, and `robust_normalize` bounds it
    # to [-1, +1] by dividing through by the clip.
    assert np.max(np.abs(seen["signal"])) <= 1.0 + 1e-6
    # The gate trace is picoamps: ~30 pA events standing off a ~0 pA baseline.
    gate = seen["gate"]
    assert gate.shape == seen["signal"].shape
    assert np.min(gate) < -20.0, "the gate must see real picoamp depths"
    assert abs(float(np.median(gate))) < 2.0, "...detrended, so baseline ~0 pA"
    # The raw trace still carries the open-pore level the local I0 IS — detrending
    # is exactly what removed it from the gate trace.
    assert abs(float(np.median(seen["raw"]))) > 50.0


def test_a_gateless_preset_needs_no_separate_gate_trace(window, qtbot, monkeypatch, tmp_path):
    """For every preset that ships the gates off, nothing extra is materialised: the
    adapter passes ``gate_signal=None`` and the core falls back to ``detr``."""
    seen: dict[str, object] = {}
    real = ml_detect.ml_detect_regions

    def _spy(*a, **kw):
        seen["gate_signal"] = kw.get("gate_signal")
        return real(*a, **kw)

    monkeypatch.setattr(ml_detect, "ml_detect_regions", _spy)
    _use_fake_model(monkeypatch)
    _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.set_model_path(str(_fake_keras_file(tmp_path, FAKE_W, "apo.keras")))
    det.apply_ml_preset("apo_nb_600mV")
    _run_ml(window, qtbot)
    assert seen["gate_signal"] is None
