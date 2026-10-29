"""The Measure panel's side of the trace mechanics: ``addRequested``, the target caption beside the
Add button, ``set_active(..., seed=False)``, and the readers that must treat "Measure on, no mask"
as "nothing to measure".

The Add button no longer decides where the row goes. It announces ``addRequested`` and its owner
(the event add door) answers; ``add_current`` stays a plain method for the legacy measurement list
and for tests.
"""
from __future__ import annotations

from PySide6 import QtWidgets

from .test_trace_rig import (  # noqa: F401  (fixtures are used by name)
    Recorder, free_mouse, rig, zoom_middle)


def _mask(rig, lo_idx: int = 40_000, hi_idx: int = 41_000) -> None:
    rig.panel.set_active(True, seed=False)
    xs = rig.xs()
    rig.view.set_measure_region(float(xs[lo_idx]), float(xs[hi_idx]))
    QtWidgets.QApplication.processEvents()


# ================================================================================================
# the Add button asks, the owner answers
# ================================================================================================
def test_the_add_button_emits_add_requested_and_adds_nothing_itself(rig):
    zoom_middle(rig, 0.5)
    _mask(rig)
    asked = Recorder(rig.panel.addRequested)
    changed = Recorder(rig.panel.measurementsChanged)
    assert rig.panel._add_button.isEnabled()

    rig.panel._add_button.click()

    assert asked.calls == [()]
    assert len(rig.panel.measurements_df) == 0 and len(changed) == 0


def test_add_current_is_still_a_plain_working_method(rig):
    zoom_middle(rig, 0.5)
    _mask(rig)
    asked = Recorder(rig.panel.addRequested)
    assert rig.panel.add_current() is True
    assert len(rig.panel.measurements_df) == 1 and len(asked) == 0


def test_the_add_button_is_off_while_there_is_no_mask(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    assert rig.panel._add_button.isEnabled() is False
    _mask(rig)
    assert rig.panel._add_button.isEnabled() is True
    rig.view.discard_measure_region()
    assert rig.panel._add_button.isEnabled() is False


# ================================================================================================
# "→ Manual": the caption beside the button
# ================================================================================================
def test_the_target_text_shows_beside_the_button_and_empty_hides_it(rig):
    panel = rig.panel
    assert panel.add_target_text() == "" and panel._add_target_label.isHidden()
    panel.set_add_target_text("→ Manual")
    assert panel.add_target_text() == "→ Manual"
    assert not panel._add_target_label.isHidden()
    assert panel._add_target_label.toolTip() == "→ Manual"
    panel.set_add_target_text("→ Schmitt")
    assert panel.add_target_text() == "→ Schmitt"
    panel.set_add_target_text("")
    assert panel._add_target_label.isHidden() and panel.add_target_text() == ""
    panel.set_add_target_text(None)                                  # tolerated like empty
    assert panel._add_target_label.isHidden()


def test_the_target_text_costs_the_dock_no_width(rig):
    """The Add/Remove row once floored the whole Inspector dock; a caption must not do it again."""
    panel = rig.panel
    panel.show()
    QtWidgets.QApplication.processEvents()
    before = panel.minimumSizeHint().width()
    panel.set_add_target_text("→ Schmitt · a very long table name indeed")
    QtWidgets.QApplication.processEvents()
    assert panel.minimumSizeHint().width() == before
    panel.set_add_target_text("")
    assert panel.minimumSizeHint().width() == before
    panel.hide()


def test_the_add_button_sits_with_the_cursor_tools_beside_clear_mask(rig):
    panel = rig.panel
    panel.show()
    panel.set_add_target_text("→ Manual")
    QtWidgets.QApplication.processEvents()
    assert panel._add_button.geometry().top() <= panel._clear_mask_btn.geometry().top() + 2
    assert panel._add_button.geometry().left() < panel._clear_mask_btn.geometry().left()
    assert panel._add_button.width() > 40
    assert panel._add_button.text() == "Add event"
    panel.hide()


# ================================================================================================
# set_active(seed=)
# ================================================================================================
def test_set_active_seed_false_arms_the_tool_with_no_mask_and_no_recompute(rig, monkeypatch):
    from nanoscope.gui import measure_panel as module

    calls = []
    real = module.measure_selection
    monkeypatch.setattr(module, "measure_selection", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    assert calls == [] and rig.view.measure_empty() is True
    assert rig.panel.is_active() is True and rig.panel._clear_mask_btn.isEnabled()


def test_set_active_seed_is_ignored_when_the_toggle_does_not_flip(rig):
    zoom_middle(rig, 0.5)
    _mask(rig)
    rig.panel.set_active(True, seed=False)                          # already on: nothing happens
    assert rig.view.measure_bounds() == (40_000, 41_000)


def test_the_seed_flag_does_not_leak_into_the_next_toggle(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    rig.panel.set_active(False)
    rig.panel._toggle.setChecked(True)                              # a user click: seeds, as ever
    assert rig.view.measure_empty() is False and rig.view.measure_bounds() is not None


def test_clear_mask_still_turns_the_tool_off(rig):
    """The button keeps its meaning; discarding with Measure left on is the view's own call."""
    zoom_middle(rig, 0.5)
    _mask(rig)
    assert rig.panel.clear_mask() is True
    assert rig.panel.is_active() is False and rig.view.measure_empty() is False
    assert rig.view.measure_target() is None


# ================================================================================================
# the readers of the selection treat "no mask" as "no selection"
# ================================================================================================
def test_the_export_of_the_processed_trace_is_the_whole_recording_when_the_mask_is_empty(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    bounds = rig.view.measure_bounds()
    assert bounds is None                                           # which export reads as "whole recording"


def test_a_stale_selection_cannot_be_updated_in_place(rig):
    zoom_middle(rig, 0.5)
    _mask(rig)
    assert rig.panel.add_current() is True
    rig.panel._select_source_row(0)
    assert rig.panel._selected_update_candidate() is not None
    rig.view.discard_measure_region()
    assert rig.panel._selected_update_candidate() is None
    assert rig.panel._update_button.isEnabled() is False
    assert rig.panel.update_selected() is False
