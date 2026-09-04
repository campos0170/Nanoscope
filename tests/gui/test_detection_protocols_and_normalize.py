"""DetectionPanel: the Normalize stage, user-definable menus, and the model readout.

Four owner-driven changes are pinned here (pytest-qt, offscreen):

* ITEM 1A -- a NORMALIZE stage in the protocol-able preprocessing chain. It is the one
  stage that changes the UNIT of the trace the detector measures on, so as much of the
  test weight is on the disclosure as on the parameters.
* ITEM 1B -- the ML "Model preprocessing" menu is user-definable: a saved preset is a
  first-class preset (recall, deviation indicator, `ml_params()`), not a second-class
  copy of one. Three call sites indexed ``PREPROCESS_PRESETS`` by name and raised
  ``KeyError`` on anything the user saved; one of them is on the Run path.
* ITEM 1C -- Save-as no longer overwrites a protocol silently, and Rename exists (which
  MOVES the stored chain, unlike save-then-delete, which would capture whatever is on
  screen now).
* ITEM 6 -- ``set_model_path(None)`` leaves the loaded-models combo and the model label
  agreeing that nothing is loaded.
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
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QInputDialog, QMessageBox

from nanoscope.analysis import filters
from nanoscope.analysis.ml_preprocess import (
    DECODE_PRESETS,
    DEFAULT_PRESET,
    PREPROCESS_PRESETS,
    PRESET_NAMES,
)
from nanoscope.gui.panels import DetectionPanel

# The Normalize page is appended to the stack, so its index is one past Savitzky-Golay.
NORMALIZE_INDEX = 7


@pytest.fixture(autouse=True)
def _clean_stores():
    """Both user stores start and end empty.

    The suite redirects QSettings to a temp .ini (see tests/conftest.py), but the store
    is process-wide, so a protocol left behind by one test is visible to the next one's
    combo. Clearing both keys keeps each test's menu contents its own."""
    def _clear():
        s = QSettings()
        s.remove(DetectionPanel._PROTO_SETTINGS_KEY)
        s.remove(DetectionPanel._ML_PRESET_SETTINGS_KEY)
        s.sync()
    _clear()
    yield
    _clear()


def _answer(monkeypatch, button):
    """Make the next QMessageBox.question return ``button`` without a modal."""
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: button))


def _typed(monkeypatch, *names):
    """Feed QInputDialog.getText the given answers in order (all accepted)."""
    queue = list(names)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (queue.pop(0), True)))


def _fake_keras_file(tmp_path, window: int, name: str = "m.keras"):
    """A minimal .keras zip carrying only the input shape peek_model_window reads."""
    path = tmp_path / name
    cfg = {"class_name": "Functional", "config": {"layers": [
        {"class_name": "InputLayer", "config": {"batch_shape": [None, window, 1]}}]}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps(cfg))
        z.writestr("metadata.json", "{}")
    return path


# ================================================================================================
# ITEM 1A -- the Normalize stage
# ================================================================================================

def test_normalize_page_is_in_the_stack_with_its_widgets(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.filter_selector.itemText(NORMALIZE_INDEX) == "Normalize"
    panel.filter_selector.setCurrentIndex(NORMALIZE_INDEX)
    assert panel.filter_stack.currentIndex() == NORMALIZE_INDEX
    assert panel.normalize_enabled.isVisibleTo(panel)
    assert panel.normalize_mode.isVisibleTo(panel)
    # A widget belonging to a different page is not.
    assert not panel.savgol_window_s.isVisibleTo(panel)


def test_mode_combo_offers_exactly_the_kernel_modes(qtbot):
    """The menu is built from filters.NORMALIZE_MODES_ALL, so it cannot offer a mode
    the stage cannot dispatch to (or hide one it can).

    IT WIDENED FROM ``NORMALIZE_MODES`` TO ``NORMALIZE_MODES_ALL`` when the model-input
    normalization landed, and the distinction is load-bearing rather than cosmetic:
    ``normalize`` still REFUSES ``'model_input'``, because that kernel is a trimmed
    three-pass re-estimation followed by a CLIP -- not affine, bounded, and with the
    opposite degenerate-input contract. `modules._apply_normalize` is the one place the
    two are told apart, so the combo must offer the union while each mode is still
    checked against the kernel that really takes it.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    data = [panel.normalize_mode.itemData(i)
            for i in range(panel.normalize_mode.count())]
    assert tuple(data) == filters.NORMALIZE_MODES_ALL
    for mode in filters.NORMALIZE_MODES_ALL:
        panel.set_normalize_mode(mode)
        assert panel.normalize_mode_choice() == mode
        # ...and the choice is a string the kernel that owns it really accepts.
        x = np.linspace(0.0, 1.0, 256)
        if mode == filters.MODEL_INPUT_MODE:
            filters.robust_normalize(x)
            with pytest.raises(ValueError):
                filters.normalize(x, mode=mode)
        else:
            filters.normalize(x, mode=mode)


def test_params_carry_the_stage_and_default_off(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    p = panel.params()
    assert p["normalize_enabled"] is False
    assert p["normalize_mode"] == "zscore"
    panel.set_normalize_enabled(True)
    panel.set_normalize_mode("minmax")
    p = panel.params()
    assert p["normalize_enabled"] is True and p["normalize_mode"] == "minmax"


def test_set_params_round_trips_the_stage(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_params({"normalize_enabled": True, "normalize_mode": "robust"})
    assert panel.normalize_enabled.isChecked()
    assert panel.normalize_mode_choice() == "robust"
    # An unknown mode is IGNORED, never silently swapped: a different divisor is a
    # different measurement.
    panel.set_params({"normalize_mode": "whitening"})
    assert panel.normalize_mode_choice() == "robust"


def test_unknown_mode_string_leaves_the_choice_alone(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_normalize_mode("nonsense")
    assert panel.normalize_mode_choice() == "zscore"


def test_protocol_keys_are_all_real_params_keys(qtbot):
    """THE INVARIANT the Normalize keys had to be added to _PROTOCOL_KEYS without
    breaking: a protocol may only capture keys params() actually produces, or recalling
    one would restore a key set_params() never reads."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert set(DetectionPanel._PROTOCOL_KEYS) - set(panel.params()) == set()
    assert {"normalize_enabled", "normalize_mode"} <= set(DetectionPanel._PROTOCOL_KEYS)


def test_protocol_captures_and_recalls_the_normalize_stage(qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_normalize_enabled(True)
    panel.set_normalize_mode("robust")
    _typed(monkeypatch, "ml-feed")
    panel._save_protocol()
    assert panel.protocol_combo.currentText() == "ml-feed"

    # Hand-edit it away, then recall: the stage comes back with its mode.
    panel.set_normalize_enabled(False)
    panel.set_normalize_mode("zscore")
    panel.protocol_combo.setCurrentText("ml-feed")
    assert panel.normalize_enabled.isChecked()
    assert panel.normalize_mode_choice() == "robust"


def test_preview_toggle_treats_normalize_as_a_stage(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.filter_preview.isEnabled()
    panel.set_normalize_enabled(True)
    assert panel.filter_preview.isEnabled()
    panel.set_normalize_enabled(False)
    assert not panel.filter_preview.isEnabled()


def test_active_summary_lists_normalize(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_normalize_enabled(True)
    assert "normalize zscore" in panel.active_filters_label.text()
    assert panel.filter_selector.itemText(NORMALIZE_INDEX) == "● Normalize"


def test_unit_banner_appears_from_any_page_and_names_the_mode(qtbot):
    """The disclosure names affected current-derived fields without claiming that
    indices, time, curation and other non-amplitude exports have changed units."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.normalize_units_label.isVisibleTo(panel)

    panel.set_normalize_enabled(True)
    panel.filter_selector.setCurrentIndex(0)          # look at a DIFFERENT stage
    assert panel.normalize_units_label.isVisibleTo(panel)
    text = panel.normalize_units_label.text()
    assert "zscore" in text and "pA" in text
    assert "DIMENSIONLESS" in text
    assert "dI/I0" in text
    assert "every exported column" not in text.lower()

    panel.set_normalize_mode("minmax")
    assert "minmax" in panel.normalize_units_label.text()
    assert "span" in panel.normalize_units_label.text()

    panel.set_normalize_enabled(False)
    assert not panel.normalize_units_label.isVisibleTo(panel)


def test_response_note_explains_there_is_no_transfer_function(qtbot):
    """The clipped model-input option makes a fixed affine |H(f)| actively false."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_recording(50_000.0)
    panel.filter_selector.setCurrentIndex(NORMALIZE_INDEX)
    note = panel.response_note.text()
    assert "Normalize" in note and "UNIT" in note
    assert "no fixed |H(f)|" in note
    assert "single affine" not in note
    assert panel.response_curve.getData()[0] is None or \
        len(panel.response_curve.getData()[0]) == 0


# ================================================================================================
# ITEM 1C -- Save-as no longer overwrites in silence, and Rename moves the stored chain
# ================================================================================================

def test_save_as_onto_an_existing_name_asks_before_overwriting(qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_enabled(True)
    panel.notch_harmonics.setValue(5)
    _typed(monkeypatch, "mains", "mains")
    panel._save_protocol()
    assert DetectionPanel._load_protocols()["mains"]["notch_harmonics"] == 5

    # Second save under the SAME name, answered No -> the stored chain is untouched.
    panel.notch_harmonics.setValue(2)
    _answer(monkeypatch, QMessageBox.No)
    panel._save_protocol()
    assert DetectionPanel._load_protocols()["mains"]["notch_harmonics"] == 5


def test_save_as_overwrites_when_confirmed(qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_enabled(True)
    panel.notch_harmonics.setValue(5)
    _typed(monkeypatch, "mains", "mains")
    panel._save_protocol()
    panel.notch_harmonics.setValue(2)
    _answer(monkeypatch, QMessageBox.Yes)
    panel._save_protocol()
    assert DetectionPanel._load_protocols()["mains"]["notch_harmonics"] == 2
    assert panel.protocol_combo.currentText() == "mains"


def test_rename_moves_the_saved_chain_not_the_one_on_screen(qtbot, monkeypatch):
    """The whole point of Rename over save-as-and-delete: after a hand edit, save-as
    would store the EDITED chain under the new name."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_enabled(True)
    panel.notch_harmonics.setValue(5)
    _typed(monkeypatch, "typoo")
    panel._save_protocol()

    # Edit the form (the combo drops to manual), then re-select and rename.
    panel.notch_harmonics.setValue(2)
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL
    panel.protocol_combo.setCurrentText("typoo")
    _typed(monkeypatch, "eu-mains")
    panel._rename_protocol()

    stored = DetectionPanel._load_protocols()
    assert "typoo" not in stored
    assert stored["eu-mains"]["notch_harmonics"] == 5      # the SAVED value, not 2
    assert panel.protocol_combo.currentText() == "eu-mains"


def test_rename_onto_an_occupied_name_asks_first(qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_enabled(True)
    panel.notch_harmonics.setValue(5)
    _typed(monkeypatch, "a")
    panel._save_protocol()
    panel.notch_harmonics.setValue(3)
    _typed(monkeypatch, "b")
    panel._save_protocol()

    panel.protocol_combo.setCurrentText("a")
    _typed(monkeypatch, "b")
    _answer(monkeypatch, QMessageBox.No)
    panel._rename_protocol()
    stored = DetectionPanel._load_protocols()
    assert set(stored) == {"a", "b"}
    assert stored["b"]["notch_harmonics"] == 3            # b survived intact


def test_rename_and_delete_are_enabled_together_and_only_on_a_real_protocol(
        qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.protocol_rename_btn.isEnabled()
    assert not panel.protocol_delete_btn.isEnabled()
    _typed(monkeypatch, "chain")
    panel._save_protocol()
    assert panel.protocol_rename_btn.isEnabled()
    assert panel.protocol_delete_btn.isEnabled()
    # A hand edit takes the name off the chain -> both go dead again.
    panel.notch_harmonics.setValue(4)
    assert not panel.protocol_rename_btn.isEnabled()
    assert not panel.protocol_delete_btn.isEnabled()


# ================================================================================================
# ITEM 1B -- user-definable model-preprocessing presets
# ================================================================================================

def test_save_recall_and_persist_a_user_ml_preset(qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.ml_despike.setValue(9)
    panel.ml_smoothing.setValue(11)
    _typed(monkeypatch, "my-student")
    panel._save_ml_preset_as()

    assert panel.ml_preset_name() == "my-student"
    assert "my-student" in DetectionPanel._load_ml_user_presets()
    # The stored config claims its OWN name, not the shipped one it was derived from.
    stored = DetectionPanel._load_ml_user_presets()["my-student"]
    assert stored["preprocess"]["preset"] == "my-student"
    assert stored["decode"]["preset"] == "my-student"

    # Move away and back: the saved numbers come back through the same writer a
    # shipped preset uses.
    panel.apply_ml_preset(DEFAULT_PRESET)
    assert panel.ml_despike.value() != 9
    panel.apply_ml_preset("my-student")
    assert panel.ml_despike.value() == 9
    assert panel.ml_smoothing.value() == 11

    # A second panel sees it (QSettings, per user, like the preprocessing protocols).
    panel2 = DetectionPanel()
    qtbot.addWidget(panel2)
    assert panel2.ml_preset.findData("my-student") >= 0
    panel2.apply_ml_preset("my-student")
    assert panel2.ml_despike.value() == 9


def test_a_user_preset_persists_the_three_amplitude_gates(qtbot, monkeypatch):
    """The gates go through QSettings on the same ``to_dict``/``from_dict`` path as
    everything else, so a silently dropped one would come back as 0.0 = OFF — a
    preset that looks recalled and quietly detects a different population. Checked at
    the STORED DICT, not just at the widget, because that is where a field goes
    missing."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.ml_min_depth_sigma.setValue(4.5)
    panel.ml_min_depth_pa.setValue(12.0)
    panel.ml_min_fracblock.setValue(0.08)
    _typed(monkeypatch, "gated-student")
    panel._save_ml_preset_as()

    stored = DetectionPanel._load_ml_user_presets()["gated-student"]["decode"]
    assert stored["min_depth_sigma"] == pytest.approx(4.5)
    assert stored["min_depth_pa"] == pytest.approx(12.0)
    assert stored["min_fracblock"] == pytest.approx(0.08)

    # Away and back through the same writer a shipped preset uses.
    panel.apply_ml_preset(DEFAULT_PRESET)
    assert panel.ml_min_depth_sigma.value() == 0.0
    panel.apply_ml_preset("gated-student")
    assert panel.ml_min_depth_sigma.value() == pytest.approx(4.5)
    assert panel.ml_min_depth_pa.value() == pytest.approx(12.0)
    assert panel.ml_min_fracblock.value() == pytest.approx(0.08)
    assert panel.ml_params()["ml_preset_customized"] is False


def test_user_preset_does_not_key_error_the_status_or_ml_params(qtbot, monkeypatch):
    """_refresh_ml_preset_status and ml_params() both indexed PREPROCESS_PRESETS by
    name. The second is on the Run path, so this was a crash, not a cosmetic bug."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.ml_despike.setValue(7)
    _typed(monkeypatch, "mine")
    panel._save_ml_preset_as()

    assert "mine" in panel.ml_preset_status.text()
    assert "matches" in panel.ml_preset_status.text()
    p = panel.ml_params()                    # must not raise
    assert p["ml_preset"] == "mine"
    assert p["ml_preset_customized"] is False
    assert p["ml_preset_deviations"] == []

    # Deviating from a USER preset is reported the same way as from a shipped one.
    panel.ml_despike.setValue(8)
    assert "CUSTOM" in panel.ml_preset_status.text()
    assert panel.ml_params()["ml_preset_customized"] is True


def test_user_preset_says_its_molecule_is_not_recorded(qtbot, monkeypatch):
    """The shipped presets name a molecule because we measured them. A user's own
    pipeline has no such fact behind it, and printing "UNKNOWN" as though it were one
    would be worse than saying whose answer it is."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    _typed(monkeypatch, "mine")
    panel._save_ml_preset_as()
    assert "not recorded" in panel.ml_preset_molecule.text()


def test_loading_a_model_while_a_user_preset_is_in_force_does_not_raise(
        qtbot, monkeypatch, tmp_path):
    """set_model_input_width's mismatch branch indexed PRESET_MOLECULE/PRESET_LABELS by
    the in-force name. A user preset has no recorded molecule, so there is no evidence
    of disagreement to report — and certainly no KeyError to raise."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    _typed(monkeypatch, "mine")
    panel._save_ml_preset_as()
    panel.set_model_path(str(_fake_keras_file(tmp_path, 3072)))   # must not raise
    assert panel.ml_preset_name() == "mine"                       # not overruled


def test_shipped_preset_names_are_refused(qtbot, monkeypatch):
    """Shadowing a shipped key would make a stored run replay under different numbers
    than it ran with."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    _typed(monkeypatch, PRESET_NAMES[0])
    panel._save_ml_preset_as()
    assert PRESET_NAMES[0] not in DetectionPanel._load_ml_user_presets()
    # ...and the shipped preset itself is untouched.
    assert PREPROCESS_PRESETS[PRESET_NAMES[0]] is not None


def test_saving_onto_an_existing_user_preset_asks_first(qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.ml_despike.setValue(5)
    _typed(monkeypatch, "mine", "mine")
    panel._save_ml_preset_as()
    panel.ml_despike.setValue(6)
    _answer(monkeypatch, QMessageBox.No)
    panel._save_ml_preset_as()
    stored = DetectionPanel._load_ml_user_presets()["mine"]
    assert stored["preprocess"]["despike_size"] == 5


def test_delete_only_offered_for_user_presets_and_falls_back_to_the_default(
        qtbot, monkeypatch):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.ml_preset_delete_btn.isEnabled()   # a shipped preset is selected
    _typed(monkeypatch, "mine")
    panel._save_ml_preset_as()
    assert panel.ml_preset_delete_btn.isEnabled()

    panel._delete_ml_preset()
    assert DetectionPanel._load_ml_user_presets() == {}
    assert panel.ml_preset.findData("mine") < 0
    assert panel.ml_preset_name() == DEFAULT_PRESET     # never a dangling selection
    assert not panel.ml_preset_delete_btn.isEnabled()


def test_shipped_presets_still_load_exactly_as_before(qtbot, monkeypatch):
    """The user store must not disturb the shipped block of the combo."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    _typed(monkeypatch, "mine")
    panel._save_ml_preset_as()
    for name in PRESET_NAMES:
        panel.apply_ml_preset(name)
        assert panel.ml_preprocess_config() == PREPROCESS_PRESETS[name]
        assert panel.ml_decode_config() == DECODE_PRESETS[name]


def test_a_corrupt_user_preset_store_is_ignored_not_fatal(qtbot):
    QSettings().setValue(DetectionPanel._ML_PRESET_SETTINGS_KEY, "{not json")
    panel = DetectionPanel()                 # constructing must not raise
    qtbot.addWidget(panel)
    assert DetectionPanel._load_ml_user_presets() == {}
    assert panel.ml_preset_name() == DEFAULT_PRESET


def test_set_ml_params_names_a_user_preset_it_can_recognise(qtbot, monkeypatch):
    """A stored run made under a user preset resolves through neither PRESET_NAMES nor
    the aliases; the user store is consulted so the combo can still say which preset the
    run belonged to."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.ml_despike.setValue(4)
    _typed(monkeypatch, "mine")
    panel._save_ml_preset_as()
    stored_run = panel.ml_params()

    panel2 = DetectionPanel()
    qtbot.addWidget(panel2)
    panel2.set_ml_params(stored_run)
    assert panel2.ml_preset_name() == "mine"
    assert panel2.ml_despike.value() == 4


# ================================================================================================
# ITEM 6 -- the model label and the model combo agree about what is loaded
# ================================================================================================

def test_clearing_the_model_puts_the_combo_back_on_none_loaded(qtbot, tmp_path):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    a = str(_fake_keras_file(tmp_path, 2048, "student_a.keras"))
    panel.set_model_path(a)
    assert panel.ml_model_label.text() == "student_a.keras"

    panel.set_model_path(None)
    combo = panel.ml_model_combo
    assert panel.ml_model_label.text() == "none"
    # The two readouts agree: the combo shows the sentinel, not the last model.
    assert combo.currentIndex() == 0
    assert combo.itemData(0) is None
    assert combo.itemText(0) == DetectionPanel._ML_NO_MODEL_ITEM
    assert panel.model_path() is None
    # The session's history is kept BELOW the sentinel, so it can be picked back up.
    assert combo.findData(a) > 0
    combo.setCurrentIndex(combo.findData(a))
    assert panel.model_path() == a
    # ...and selecting it drops the sentinel again.
    assert combo.itemData(0) is not None
    assert panel.ml_model_label.text() == "student_a.keras"


def test_selecting_the_sentinel_is_inert(qtbot, tmp_path):
    """It carries no path, so picking it must not be treated as a model switch."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    a = str(_fake_keras_file(tmp_path, 2048, "student_a.keras"))
    panel.set_model_path(a)
    panel.set_model_path(None)
    panel.ml_model_combo.setCurrentIndex(0)
    assert panel.model_path() is None


# ===========================================================================
# ROUND 5 -- the silent-swallow regression, the second notch spelling, and the
# shipped protocols
# ===========================================================================
def test_every_kernel_detrend_method_is_reachable_from_the_combo(qtbot):
    """The combo is BUILT from ``baseline.METHODS``, so it cannot drift from the set
    the kernel implements. It drifted once: three of the seven were unreachable."""
    from nanoscope.analysis import baseline as baseline_mod

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    offered = [m for _label, m in panel._DETREND_METHODS]
    # Everything the kernel accepts except the one deliberate omission.
    assert set(offered) == set(baseline_mod.METHODS) - {"constant"}
    assert {"masked_median", "robust_median", "savgol"} <= set(offered)
    for method in offered:
        assert panel.set_detrend_method(method) is None
        assert panel.detrend_method_choice() == method
        assert panel.params()["detrend_method"] == method


def test_an_unknown_detrend_method_is_REFUSED_not_swallowed(qtbot):
    """THE REGRESSION. ``set_detrend_method`` used to loop the combo and ``return`` on
    no match, leaving a DIFFERENT estimator selected: a protocol saying
    ``masked_median`` left ``detrend_enabled`` True and ``detrend_method``
    ``block_median`` -- the wrong estimator, no warning, no visible difference, and
    provenance that read correct.

    Ignoring is right for the normalize MODE (it keeps the same estimator) and wrong
    here (it runs a different one). The refusal is returned AND raised as a banner,
    because the user who triggers it is recalling a protocol, not reading this page.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_detrend_method("percentile")

    complaint = panel.set_detrend_method("not_an_estimator")
    assert complaint                                   # it SAYS so
    assert "percentile" in complaint                   # ...names what is really running
    assert panel.detrend_method_choice() == "percentile"   # ...and did not swap it
    assert panel.detrend_refusal_label.text() == complaint

    # ...and set_params, the path a protocol and a project restore both take, surfaces
    # it rather than absorbing it.
    panel.set_params({"detrend_enabled": True, "detrend_method": "not_an_estimator"})
    assert panel.detrend_refusal_label.text()
    assert panel.detrend_method_choice() == "percentile"

    # A real method clears the banner again.
    assert panel.set_detrend_method("masked_median") is None
    assert not panel.detrend_refusal_label.text()


def test_a_method_the_kernel_knows_but_the_panel_omits_says_which_it_is(qtbot):
    """'constant' is a real estimator that is deliberately not offered. The complaint
    distinguishes that from a name nothing in the app knows -- the user can act on the
    difference."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    known = panel.set_detrend_method("constant")
    unknown = panel.set_detrend_method("wavelet_magic")
    assert "does not offer it" in known
    assert "not a baseline estimator this app knows" in unknown


def test_the_window_spin_is_disabled_for_the_method_it_does_nothing_for(qtbot):
    """PINNED BY MEASUREMENT upstream: two masked_median runs at window_s 0.05 and 5.0
    are byte-identical. A knob the user turns while the numbers do not move, with
    nothing saying why, is worse than no knob."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    for method in ("block_median", "percentile", "asls", "robust_median", "savgol"):
        panel.set_detrend_method(method)
        assert panel.detrend_window_s.isEnabled(), method
    panel.set_detrend_method("masked_median")
    assert not panel.detrend_window_s.isEnabled()
    assert "never reads this window" in panel.detrend_window_note.text()


def test_the_new_shape_knobs_round_trip_through_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    stored = {
        "detrend_enabled": True, "detrend_method": "masked_median",
        "detrend_rough_s": 0.5, "detrend_base_s": 0.02, "detrend_decim": 10,
        "detrend_mask_sigma": 3.5, "detrend_dilate_ms": 0.2,
        "detrend_sg_poly": 4, "detrend_sg_decim": 25,
    }
    panel.set_params(stored)
    got = panel.params()
    for k, v in stored.items():
        assert got[k] == v, k


# -- the explicit notch line list -------------------------------------------
def test_the_notch_line_list_round_trips_and_says_which_spelling_is_live(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.params()["notch_freqs"] is None       # series spelling by default
    assert panel.notch_freq_hz.isEnabled()

    lines = [60.0, 120.0, 180.0, 300.0, 420.0, 540.0]
    panel.set_params({"notch_enabled": True, "notch_freqs": lines})
    assert panel.params()["notch_freqs"] == lines
    assert panel.notch_use_lines.isChecked()
    # The two spellings are EXCLUSIVE in the kernel, so the form makes it visible.
    assert not panel.notch_freq_hz.isEnabled()
    assert not panel.notch_harmonics.isEnabled()
    assert "NOT read" in panel.notch_lines_note.text()

    panel.set_params({"notch_freqs": None})
    assert panel.params()["notch_freqs"] is None
    assert panel.notch_freq_hz.isEnabled()


def test_an_empty_line_list_falls_back_visibly_never_silently(qtbot):
    """The kernel REFUSES ``[]`` by design ("must name at least one frequency"), so the
    panel must never emit it -- a run that dies at the door is not an improvement. It
    answers None and says so in orange instead."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.notch_use_lines.setChecked(True)
    panel.notch_freqs.setText("   ,  ; ")
    panel._refresh_notch_mode()
    assert panel.params()["notch_freqs"] is None
    assert "EMPTY" in panel.notch_lines_note.text()


def test_the_line_list_is_captured_by_a_protocol(qtbot):
    """A protocol that stored six lines but recalled (60 Hz, 3 harmonics) would run a
    DIFFERENT filter under the same name."""
    assert "notch_freqs" in DetectionPanel._PROTOCOL_KEYS
    assert "normalize_clip_sigma" in DetectionPanel._PROTOCOL_KEYS
    assert {"detrend_rough_s", "detrend_base_s", "detrend_mask_sigma",
            "detrend_dilate_ms", "detrend_decim", "detrend_sg_poly",
            "detrend_sg_decim"} <= set(DetectionPanel._PROTOCOL_KEYS)


# -- the three shipped protocols --------------------------------------------
_RECIPES = {
    "Host-guest (CB7 + adamantyl)": {
        "notch_freqs": [60.0, 120.0, 180.0, 300.0, 420.0, 540.0],
        "notch_q": 30.0,
        "detrend_method": "masked_median", "detrend_rough_s": 0.5,
        "detrend_base_s": 0.02, "detrend_decim": 10, "detrend_mask_sigma": 3.5,
        "detrend_dilate_ms": 0.2,
        "normalize_mode": "model_input", "normalize_clip_sigma": 30.0,
    },
    "Zn2-insulin (zn_30h)": {
        "notch_freqs": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0],
        "notch_q": 30.0,
        "detrend_method": "masked_median", "detrend_rough_s": 0.5,
        "detrend_base_s": 0.2, "detrend_decim": 10, "detrend_mask_sigma": 3.0,
        "detrend_dilate_ms": 2.0,
        "normalize_mode": "model_input", "normalize_clip_sigma": 30.0,
    },
    "Cu2-insulin (SavGol baseline)": {
        "notch_freqs": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0],
        "notch_q": 30.0,
        # SAVITZKY-GOLAY, NOT masked median. The Cu notebook's own cell-0 markdown says
        # "event-masked robust median" and is WRONG; Block 0's BASELINE_METHOD dict and
        # all four written cu_*_metadata.json say "sg" with masked_median_params null.
        "detrend_method": "savgol", "detrend_window_s": 0.2,
        "detrend_sg_poly": 3, "detrend_sg_decim": 50,
        "normalize_mode": "model_input", "normalize_clip_sigma": 30.0,
    },
}


@pytest.mark.parametrize("name", list(_RECIPES))
def test_each_shipped_protocol_is_the_owners_recipe_exactly(qtbot, name):
    """Selecting the protocol puts the owner's own numbers on the form -- every one of
    them, not an approximation. These were recovered from his notebooks and
    cross-checked against the ``*_metadata.json`` his code writes."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.protocol_combo.setCurrentText(name)
    got = panel.params()
    assert got["notch_enabled"] is True
    assert got["detrend_enabled"] is True
    assert got["normalize_enabled"] is True
    for key, want in _RECIPES[name].items():
        assert got[key] == want, f"{name}: {key}"


def test_a_shipped_protocol_cannot_be_shadowed_renamed_or_deleted(qtbot, monkeypatch):
    """The value of the three built-ins is that the name means ONE verified chain."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.protocol_combo.setCurrentText("Cu2-insulin (SavGol baseline)")
    assert not panel.protocol_rename_btn.isEnabled()
    assert not panel.protocol_delete_btn.isEnabled()
    # Save-as onto a shipped name is refused (the modal is stubbed out).
    said: list[str] = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: said.append(a[2] if len(a) > 2 else ""))
    assert panel._protocol_name_ok("Cu2-insulin (SavGol baseline)",
                                   title="Save protocol") is False
    assert said and "built-in" in said[0]
    assert panel._protocol_name_ok("my own chain", title="Save protocol") is True


def test_a_shipped_protocol_survives_a_round_trip_through_set_params(qtbot):
    """The recall path is `set_params` -- the same round trip a project restore takes --
    so a protocol that cannot survive it is a protocol that reproduces nothing."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.protocol_combo.setCurrentText("Host-guest (CB7 + adamantyl)")
    first = panel.params()

    other = DetectionPanel()
    qtbot.addWidget(other)
    other.set_params({k: first[k] for k in DetectionPanel._PROTOCOL_KEYS})
    second = other.params()
    for key in DetectionPanel._PROTOCOL_KEYS:
        assert second[key] == first[key], key


# -- the model-input normalization ------------------------------------------
def test_the_units_banner_names_the_clip_and_the_saturation(qtbot):
    """This mode's sentence differs from the other three, and the extra clause is the
    one that can silently corrupt an analysis: the transform is BOUNDED, so two events
    of different real depth come out identical above the clip."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_normalize_mode(filters.MODEL_INPUT_MODE)
    panel.set_normalize_enabled(True)
    text = panel.normalize_units_label.text()
    assert panel.normalize_units_label.isVisibleTo(panel) or text
    assert "30" in text and "CLIPPED" in text and "BOUNDED" in text
    assert "boundaries and counts" in text and "dI/I0" in text

    panel.normalize_clip_sigma.setValue(10.0)
    assert "10" in panel.normalize_units_label.text()


def test_the_clip_spin_is_live_only_for_the_mode_that_reads_it(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_normalize_enabled(True)
    panel.set_normalize_mode("zscore")
    assert not panel.normalize_clip_sigma.isEnabled()
    panel.set_normalize_mode(filters.MODEL_INPUT_MODE)
    assert panel.normalize_clip_sigma.isEnabled()
