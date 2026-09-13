"""Opt-in, recording-specific reference controls never analyse or silently apply."""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.baseline import BASELINE_PARAM_KEYS
from nanoscope.gui.baseline_panel import BaselinePanel


@pytest.fixture
def panel(qapp, qtbot):
    widget = BaselinePanel()
    qtbot.addWidget(widget)
    widget.set_recording(1000.0, ("recording-a", 0), duration_s=10.0)
    widget.set_gate(True)
    return widget


def choose(panel, mode):
    editor = panel.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData(mode))
    return editor


def known(value=-120, notes=""):
    recipe = {"version": 1, "mode": "constant", "current_pA": value,
              "intervals_s": [], "interpolation": "constant"}
    if notes:
        recipe["notes"] = notes
    return {"baseline_open_pore_reference": recipe}


def test_automatic_params_keep_exact_legacy_shape(panel):
    assert panel.reference_mode() == "automatic"
    assert set(panel.params()) == set(BASELINE_PARAM_KEYS) | {"baseline_param_source"}
    assert panel.reference_error() == ""
    assert not panel.automatic_controls.isHidden()


@pytest.mark.parametrize("text,expected", [("-120", -120), ("−120.25", -120.25),
                                          ("2.4e3", 2400)])
def test_known_current_preserves_sign_and_units(panel, text, expected):
    editor = choose(panel, "constant")
    editor.current.setText(text)
    assert panel.params()["baseline_open_pore_reference"]["current_pA"] == expected
    assert panel.reference_error() == ""
    assert "pA" in editor.current.toolTip()
    assert panel.automatic_controls.isHidden()


@pytest.mark.parametrize("text", ["", "nan", "inf", "-inf", "oops"])
def test_invalid_known_input_is_visible_not_automatic_fallback(panel, text):
    editor = choose(panel, "constant")
    editor.current.setText(text)
    assert panel.reference_mode() == "constant"
    assert panel.reference_error()
    assert not editor.validation_label.isHidden()
    assert panel.params()["baseline_open_pore_reference"]["current_pA"] is None


@pytest.mark.parametrize("text", ["0", "-0", "1e-300"])
def test_zero_known_current_is_refused_not_silently_accepted(panel, text):
    """beta-sweep B06 (2026-09-12): a 0 pA (or anything that underflows to 0 once the
    app's float32 pipeline holds it, e.g. 1e-300) open-pore current used to be accepted
    with no error, then silently produced frac_depth=NaN on every event and a NaN
    Summary median with no explanation. Unlike the unparseable-text case above, the text
    DOES parse -- ``params()`` still reflects the typed value, exactly as it does for any
    other physically-questionable-but-parseable field (e.g. a reversed interval) -- and
    ``reference_error()`` is the separate authority every caller (detection, preview,
    probe) already gates on before touching that value."""
    editor = choose(panel, "constant")
    editor.current.setText(text)
    assert panel.reference_mode() == "constant"
    assert "cannot be zero" in panel.reference_error()
    assert not editor.validation_label.isHidden()
    assert np.float32(panel.params()["baseline_open_pore_reference"]["current_pA"]) == 0.0


def test_selection_request_is_intent_only(panel, qtbot):
    editor = choose(panel, "intervals")
    before = panel.params()
    with qtbot.assertNotEmitted(panel.probeRequested), qtbot.assertNotEmitted(panel.paramsChanged):
        with qtbot.waitSignal(panel.addReferenceSelectionRequested):
            editor.selection_button.click()
    assert panel.params() == before


def test_add_selection_is_sorted_explicit_and_emits_once(panel, qtbot):
    emissions = []
    panel.paramsChanged.connect(emissions.append)
    with qtbot.assertNotEmitted(panel.probeRequested):
        assert panel.add_reference_interval(5.2, 5.5)
        assert panel.add_reference_interval(0.1, 0.5)
    assert len(emissions) == 2
    assert panel.params()["baseline_open_pore_reference"]["intervals_s"] == [[0.1, 0.5], [5.2, 5.5]]
    assert panel.reference_error() == ""


def test_linear_requires_two_intervals(panel):
    panel.add_reference_interval(0, 0.5)
    editor = panel.reference_editor
    editor.method.setCurrentIndex(editor.method.findData("linear"))
    assert "two" in panel.reference_error()
    panel.add_reference_interval(9.5, 10)
    assert panel.reference_error() == ""
    assert panel.params()["baseline_open_pore_reference"]["interpolation"] == "linear"
    assert "endpoint" in editor.method.toolTip()


def test_intervals_need_five_samples_and_recording_bounds(panel):
    panel.add_reference_interval(0, 0.004)
    assert "five" in panel.reference_error()
    editor = panel.reference_editor
    editor.table.item(0, 1).setText("0.005")
    assert panel.reference_error() == ""
    editor.table.item(0, 1).setText("11")
    assert "beyond" in panel.reference_error()


def test_overlap_selection_is_refused_without_replacing_recipe(panel, qtbot):
    panel.add_reference_interval(1, 2)
    before = panel.params()
    with qtbot.assertNotEmitted(panel.paramsChanged):
        assert not panel.add_reference_interval(1.5, 2.5)
    assert panel.params() == before
    assert "overlap" in panel.reference_editor.validation_label.text()


def test_manual_overlap_and_out_of_order_are_invalid_drafts(panel):
    panel.add_reference_interval(1, 2)
    panel.add_reference_interval(3, 4)
    panel.reference_editor.table.item(1, 0).setText("0.5")
    assert "time order" in panel.reference_error()


def test_remove_selected_does_not_touch_other_intervals(panel):
    panel.add_reference_interval(1, 2)
    panel.add_reference_interval(3, 4)
    panel.reference_editor.table.selectRow(0)
    panel.reference_editor.remove_button.click()
    assert panel.params()["baseline_open_pore_reference"]["intervals_s"] == [[3, 4]]


def test_manual_mode_cannot_apply_or_request_estimator_proposals(panel, qtbot):
    editor = choose(panel, "constant")
    editor.current.setText("-120")
    before = panel.params()
    with qtbot.assertNotEmitted(panel.probeRequested), qtbot.assertNotEmitted(panel.paramsChanged):
        panel.probe_button.click()
        panel.apply_button.click()
        assert panel.apply_proposal(override=True) is False
    assert panel.params() == before
    assert "user-provided reference" in panel.gate_label.text()


def test_draft_edits_never_emit_a_measurement_request(panel, qtbot):
    with qtbot.assertNotEmitted(panel.probeRequested):
        editor = choose(panel, "constant")
        editor.current.setText("-120")
        editor.notes.setText("Measured at start; notebook 8")
    assert panel.params()["baseline_open_pore_reference"]["notes"] == "Measured at start; notebook 8"
    assert "not independently verified" in editor.notice.text()


def test_preview_emits_only_deliberate_intent_for_valid_reference(panel, qtbot):
    editor = choose(panel, "constant")
    with qtbot.assertNotEmitted(panel.referencePreviewRequested):
        editor.preview_button.click()
    assert "finite" in panel.gate_label.text()
    editor.current.setText("-120")
    before = panel.params()
    with qtbot.assertNotEmitted(panel.probeRequested), qtbot.assertNotEmitted(panel.paramsChanged):
        with qtbot.waitSignal(panel.referencePreviewRequested):
            editor.preview_button.click()
    assert panel.params() == before


def test_preview_gate_explains_first_real_blocker(panel, qtbot):
    editor = choose(panel, "constant")
    panel.set_gate(False, "Select a calibrated current channel.")
    with qtbot.assertNotEmitted(panel.referencePreviewRequested):
        editor.preview_button.click()
    assert "calibrated current channel" in panel.gate_label.text()
    choose(panel, "automatic")
    panel.set_gate(True)
    with qtbot.assertNotEmitted(panel.referencePreviewRequested):
        editor.preview_button.click()


def test_same_recording_retains_user_draft_and_ignores_stored_seed(panel):
    editor = choose(panel, "constant")
    editor.current.setText("-120")
    panel.set_recording(1000, ("recording-a", 0), duration_s=10)
    panel.set_params(known(-200))
    assert panel.params()["baseline_open_pore_reference"]["current_pA"] == -120


def test_different_recording_resets_reference_without_signal(panel, qtbot):
    editor = choose(panel, "constant")
    editor.current.setText("-120")
    editor.notes.setText("Only recording A")
    with qtbot.assertNotEmitted(panel.paramsChanged):
        panel.set_recording(2000, ("recording-b", 0), duration_s=5)
    assert panel.reference_mode() == "automatic"
    assert "baseline_open_pore_reference" not in panel.params()
    assert editor.current.text() == editor.notes.text() == ""
    assert editor.table.rowCount() == 0


def test_stored_recipe_round_trips_without_signal(panel, qtbot):
    with qtbot.assertNotEmitted(panel.paramsChanged):
        panel.set_params(known(-108.123456789, "Lab notebook <8>"))
    assert panel.params()["baseline_open_pore_reference"] == known(-108.123456789, "Lab notebook <8>")["baseline_open_pore_reference"]
    panel.set_recording(1000, ("recording-b", 0), duration_s=10)
    recipe = {"version": 1, "mode": "intervals", "current_pA": None,
              "intervals_s": [[0, 1], [9, 10]], "interpolation": "linear"}
    panel.set_params({"baseline_open_pore_reference": recipe})
    assert panel.params()["baseline_open_pore_reference"] == recipe


def test_switching_back_to_automatic_preserves_old_estimator_values(panel):
    panel.rough_s.setValue(2)
    before = panel.params()
    editor = choose(panel, "constant")
    editor.current.setText("-120")
    choose(panel, "automatic")
    assert panel.params() == before
    assert not panel.automatic_controls.isHidden()


def test_user_reference_does_not_inherit_dormant_batch_source(panel):
    from nanoscope.analysis.baseline import baseline_config_from_params

    panel.set_params({"baseline_param_source": "batch", "baseline_rough_s": 2})
    editor = choose(panel, "constant")
    editor.current.setText("-120")
    assert panel.params()["baseline_param_source"] == "user"
    assert panel.param_source() == "user"
    cfg, source = baseline_config_from_params(panel.params())
    assert cfg.open_pore_reference is not None
    assert source == "user"
    choose(panel, "automatic")
    assert panel.params()["baseline_param_source"] == "batch"
    assert panel.rough_s.value() == 2


def test_bad_stored_recipe_is_explicitly_flagged(panel):
    panel.set_params({"baseline_open_pore_reference": {"version": 2, "mode": "constant", "current_pA": -120}})
    assert "unsupported" in panel.reference_error()
    assert panel.reference_mode() == "constant"
    assert panel.params()["baseline_open_pore_reference"]["version"] == 2


@pytest.mark.parametrize("recipe", ["invalid", {"mode": "future"},
                                   {"mode": "intervals", "intervals_s": 3},
                                   {"mode": "constant", "current_pA": -120, "intervals_s": [[0, 1]]}])
def test_malformed_recipe_never_silently_changes_scientific_inputs(panel, recipe):
    panel.set_params({"baseline_open_pore_reference": recipe})
    assert panel.reference_error()
    assert panel.params()["baseline_open_pore_reference"] == recipe


def test_all_new_actions_have_tooltips_and_fit_narrow_inspector(panel):
    editor = panel.reference_editor
    for widget in (editor.mode, editor.current, editor.table, editor.method, editor.notes,
                   editor.selection_button, editor.add_button, editor.remove_button, editor.preview_button):
        assert widget.toolTip()
    choose(panel, "intervals")
    panel.resize(380, panel.heightForWidth(380))
    panel.show()
    assert editor.minimumSizeHint().width() <= 380
    QtWidgets.QApplication.processEvents()
    assert editor.table.geometry().bottom() < editor.selection_button.geometry().top()
