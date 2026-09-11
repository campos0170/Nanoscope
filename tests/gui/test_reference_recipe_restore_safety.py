"""A restored active-reference assertion must remain invalid, visible and non-mutating."""
from dataclasses import replace

import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.baseline import BaselineConfig, DEFAULT_BASELINE, baseline_config_from_params
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.scoring import ensure_added_column
from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.analysis_panel import schmitt_params_from_result
from .test_baseline_migration import _signal, _window


KEY = "baseline_open_pore_reference"
FLAGS = ["baseline_reference_applied", "baseline_estimator_bypassed"]
ERROR = "baseline_reference_restore_error"
RECIPE = {"version": 1, "mode": "constant", "current_pA": -100.,
          "intervals_s": [], "interpolation": "constant", "notes": "notebook 4"}
REFERENCE = BaselineConfig(open_pore_reference=RECIPE)


def invalid_params(flag, null_recipe):
    params = dict(DEFAULT_BASELINE.to_provenance(), **{flag: True})
    if null_recipe:
        params[KEY] = None
    return params


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("null_recipe", [False, True])
def test_editor_retains_missing_active_recipe_and_can_recover_deliberately(qapp, qtbot, flag, null_recipe):
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    panel.set_recording(1000., ("recording-a", 0), duration_s=10.)
    panel.set_gate(True)
    with qtbot.assertNotEmitted(panel.paramsChanged):
        panel.set_params(invalid_params(flag, null_recipe))
    assert "missing its recipe" in panel.reference_error()
    assert panel.params()[flag] is True
    assert panel.params()[KEY] is None
    assert not panel.reference_editor.validation_label.isHidden()
    with pytest.raises(ValueError, match="lack.*recipe"):
        baseline_config_from_params(panel.params())
    # Deliberately choose a mode, then Automatic. A restore or passive refresh
    # must not make that choice for the user.
    editor = panel.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.mode.setCurrentIndex(editor.mode.findData("automatic"))
    assert not panel.reference_error()
    assert KEY not in panel.params()
    assert flag not in panel.params()
    assert baseline_config_from_params(panel.params())[0].open_pore_reference is None


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("null_recipe", [False, True])
@pytest.mark.parametrize("params_kind", ["empty", "partial", "active"])
def test_saved_missing_recipe_keeps_table_but_blocks_detection_on_selection(
        qapp, qtbot, tmp_path, flag, null_recipe, params_kind):
    win, doc_id = _window(qtbot, tmp_path, _signal(n=20_000), config=DEFAULT_BASELINE, select=False)
    result = win._session.results.get(doc_id, 0, "detect.schmitt")
    # MainWindow's established first-display path annotates detector rows with
    # added=False. Seed that non-scientific compatibility column before taking
    # an exact snapshot; this test isolates reference handling, not that loader.
    ensure_added_column(result.payload.df)
    before = result.payload.df.copy(deep=True)
    prov = dict(result.payload.provenance, **{flag: True})
    prov.pop(KEY, None)
    if null_recipe:
        prov[KEY] = None
    params = {} if params_kind == "empty" else dict(result.params)
    if params_kind == "active":
        params[flag] = True
        if null_recipe:
            params[KEY] = None
    restored = replace(result, params=params, payload=EventSet(before.copy(deep=True), prov))
    win._session.add_result(restored)
    # This is the same order as loading results before selecting their recording.
    win._on_selection(doc_id, 0)
    QtWidgets.QApplication.processEvents()
    panel = win._analysis_panel
    assert "recipe" in panel._baseline.reference_error()
    assert panel._baseline.params()[flag] is True
    assert panel._baseline.params()[KEY] is None
    assert not panel.can_run()
    assert not panel.can_run_ml()
    assert panel._open_pore_reference_problem()
    assert panel.current_detection() is restored
    pd.testing.assert_frame_equal(panel.current_detection().payload.df, before, check_exact=True)
    # Repeated form seeding and passive navigation cannot clear the error.
    panel._seed_form_from(restored)
    win._on_selection(doc_id, 0)
    assert not panel.can_run()
    assert panel._baseline.reference_error()
    pd.testing.assert_frame_equal(panel.current_detection().payload.df, before, check_exact=True)


@pytest.mark.parametrize("params_kind", ["empty", "partial", "same_recipe", "same_config"])
def test_valid_stored_reference_is_recovered_from_provenance_when_params_are_partial(
        qapp, qtbot, tmp_path, params_kind):
    win, doc_id = _window(qtbot, tmp_path, _signal(n=20_000), config=REFERENCE, select=False)
    result = win._session.results.get(doc_id, 0, "detect.schmitt")
    ensure_added_column(result.payload.df)
    params = {} if params_kind == "empty" else dict(result.params)
    if params_kind == "same_recipe":
        params[KEY] = RECIPE
    if params_kind == "same_config":
        params["baseline_config"] = REFERENCE
    restored = replace(result, params=params)
    before = result.payload.df.copy(deep=True)
    win._session.add_result(restored)
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    assert not panel._baseline.reference_error()
    assert baseline_config_from_params(panel._baseline.params())[0].open_pore_reference == REFERENCE.open_pore_reference
    assert panel.can_run()
    assert panel.current_detection() is restored
    pd.testing.assert_frame_equal(restored.payload.df, before, check_exact=True)


@pytest.mark.parametrize("explicit", [
    {KEY: RECIPE | {"current_pA": -110.}},
    {KEY: None}, {KEY: {}}, {"baseline_config": DEFAULT_BASELINE},
    {"baseline_config": BaselineConfig(open_pore_reference=RECIPE | {"current_pA": -110.})},
])
def test_conflicting_saved_recipes_are_kept_as_invalid_draft_and_never_choose_a_winner(
        qapp, qtbot, tmp_path, explicit):
    win, doc_id = _window(qtbot, tmp_path, _signal(n=20_000), config=REFERENCE, select=False)
    result = win._session.results.get(doc_id, 0, "detect.schmitt")
    ensure_added_column(result.payload.df)
    params = dict(result.params, **explicit)
    restored = replace(result, params=params)
    before = result.payload.df.copy(deep=True)
    win._session.add_result(restored)
    seed = schmitt_params_from_result(restored)
    assert seed[ERROR]
    for key, value in explicit.items():
        assert seed[key] == value
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    assert panel._baseline.reference_error()
    assert panel._baseline.params()[ERROR] == seed[ERROR]
    assert not panel.can_run()
    assert panel._open_pore_reference_problem(seed) == seed[ERROR]
    with pytest.raises(ValueError):
        baseline_config_from_params(panel._form_params())
    pd.testing.assert_frame_equal(restored.payload.df, before, check_exact=True)
    assert restored.params == params
    # An explicit user decision is the only operation that clears this draft.
    editor = panel._baseline.reference_editor
    editor.mode.setCurrentIndex(editor.mode.findData("constant"))
    editor.current.setText("-105")
    assert not panel._baseline.reference_error()
    assert ERROR not in panel._form_params()
    assert panel.can_run()
    pd.testing.assert_frame_equal(restored.payload.df, before, check_exact=True)
