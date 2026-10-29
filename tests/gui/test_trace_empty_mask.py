"""Measure ON with NO mask: the state after an add, the X key, and ``set_active(True, seed=False)``.

Until now there was no such state: clearing the mask turned Measure off, and turning it on
seeded a mask over the middle fifth of the view. A missing stored range already means "seed a
mask", so the empty state is an explicit flag (``TraceView.measure_empty()``). A probe of the
naive reading ("no range") found four failures, and each has a test here: the first rebuild
brought a seeded mask back, ``set_measure_region`` did nothing so Snap to view / Snap to selected
event / Go to measurement / a double-click placed nothing, the Navigator band stayed drawn, and
Ctrl+M saved the event just added a second time from the stale readout.
"""
from __future__ import annotations

import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets

from nanoscope.core.layer import ViewMode
from nanoscope.core.sample_rate import SampleRateDeclaration

from .test_trace_rig import (  # noqa: F401  (fixtures are used by name)
    CURRENT, FS, Recorder, double_click, free_mouse, point_at, rig, zoom_middle)


def _empty(rig) -> None:
    rig.panel.set_active(True, seed=False)
    QtWidgets.QApplication.processEvents()


def _regions(rig):
    return [it for pane in rig.view.panes() for it in pane.plot.items
            if isinstance(it, pg.LinearRegionItem)]


def _mask(rig, lo_idx: int, hi_idx: int) -> None:
    xs = rig.xs()
    rig.view.set_measure_region(float(xs[lo_idx]), float(xs[hi_idx]))
    QtWidgets.QApplication.processEvents()


def _band_visible(rig) -> bool:
    return bool(rig.navigator._measure_band.isVisible())


# ================================================================================================
# entering and leaving the state
# ================================================================================================
def test_arming_without_a_seed_is_measure_on_with_no_mask_and_emits_nothing_but_the_activation(rig):
    zoom_middle(rig, 0.5)
    changed, finished = Recorder(rig.view.sigMeasureChanged), Recorder(rig.view.sigMeasureFinished)
    active = Recorder(rig.view.sigMeasureActiveChanged)

    rig.panel.set_active(True, seed=False)

    assert rig.view.measure_empty() is True
    assert rig.view.measure_bounds() is None and rig.view.measure_x_range() is None
    assert rig.view.measure_target() == (rig.doc_id, CURRENT)
    assert rig.panel.is_active() is True
    assert _regions(rig) == []
    assert active.calls == [(True,)]
    assert len(changed) == len(finished) == 0
    assert rig.panel.current_measurement() is None
    assert _band_visible(rig) is False


def test_the_default_toggle_still_seeds_a_mask_over_the_middle_fifth(rig):
    lo, hi = zoom_middle(rig, 0.5)
    changed, finished = Recorder(rig.view.sigMeasureChanged), Recorder(rig.view.sigMeasureFinished)
    rig.panel.set_active(True)
    assert rig.view.measure_empty() is False and len(_regions(rig)) == 1
    b0, b1 = rig.view.measure_bounds()
    assert (b1 - b0) / FS == pytest.approx(0.2 * (hi - lo), abs=0.01)
    assert len(changed) >= 1 and len(finished) >= 1


def test_discarding_the_mask_keeps_measure_on_and_tells_the_listeners(rig):
    zoom_middle(rig, 0.5)
    rig.panel.set_active(True)
    assert rig.panel.current_measurement() is not None and _band_visible(rig)
    active = Recorder(rig.view.sigMeasureActiveChanged)
    cleared = Recorder(rig.view.sigMeasureCleared)

    assert rig.view.discard_measure_region() is True

    assert cleared.calls == [()] and active.calls == []          # NOT deactivated
    assert rig.view.measure_empty() is True
    assert rig.view.measure_bounds() is None and rig.view.measure_x_range() is None
    assert _regions(rig) == []
    assert rig.panel.is_active() is True
    assert rig.panel.current_measurement() is None               # the readout and its stats are gone
    assert rig.panel.readout_value("n_samples") == "—"
    assert _band_visible(rig) is False                           # the Navigator band cleared
    assert rig.panel._add_button.isEnabled() is False


def test_discarding_with_nothing_to_discard_is_a_quiet_false(rig):
    cleared = Recorder(rig.view.sigMeasureCleared)
    assert rig.view.discard_measure_region() is False            # Measure off
    _empty(rig)
    assert rig.view.discard_measure_region() is False            # on, already empty
    assert len(cleared) == 0


def test_a_mask_can_be_placed_on_an_empty_target_at_exactly_the_range_given(rig):
    _empty(rig)
    changed, finished = Recorder(rig.view.sigMeasureChanged), Recorder(rig.view.sigMeasureFinished)
    xs = rig.xs()

    rig.view.set_measure_region(float(xs[40_000]), float(xs[45_000]))

    assert rig.view.measure_empty() is False
    assert rig.view.measure_bounds() == (40_000, 45_000)
    # One announcement, of THIS range: no seeded mask was announced first.
    assert changed.calls == [(40_000, 45_000)] and finished.calls == [(40_000, 45_000)]
    assert len(_regions(rig)) == 1


def test_set_measure_region_with_measure_off_still_places_nothing(rig):
    xs = rig.xs()
    rig.view.set_measure_region(float(xs[40_000]), float(xs[45_000]))
    assert rig.view.measure_bounds() is None and _regions(rig) == []


def test_arming_without_a_seed_on_a_standing_mask_leaves_it_and_on_another_target_removes_it(rig):
    _empty(rig)
    _mask(rig, 40_000, 45_000)
    rig.view.set_measure_active(rig.doc_id, CURRENT, True, seed=False)
    assert rig.view.measure_bounds() == (40_000, 45_000)         # same target: untouched
    cleared = Recorder(rig.view.sigMeasureCleared)
    rig.view.set_measure_active(rig.doc_id, 0, True, seed=False)  # re-targeted to the other pane
    assert rig.view.measure_empty() is True and rig.view.measure_target() == (rig.doc_id, 0)
    assert cleared.calls == [()] and _regions(rig) == []


# ================================================================================================
# the empty state survives what rebuilds the view
# ================================================================================================
def test_a_rebuild_does_not_bring_a_seeded_mask_back(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    changed, finished = Recorder(rig.view.sigMeasureChanged), Recorder(rig.view.sigMeasureFinished)
    active = Recorder(rig.view.sigMeasureActiveChanged)

    rig.view.rebuild()
    QtWidgets.QApplication.processEvents()

    assert rig.view.measure_empty() is True and rig.view.measure_bounds() is None
    assert _regions(rig) == [] and rig.panel.is_active() is True
    assert len(changed) == len(finished) == 0 and active.calls == []


def test_the_empty_state_survives_a_view_mode_change(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    for mode in (ViewMode.STACKED, ViewMode.OVERLAY):
        rig.session.set_view_mode(mode)
        QtWidgets.QApplication.processEvents()
        assert rig.view.measure_empty() is True, mode
        assert _regions(rig) == [] and rig.view.measure_bounds() is None
        assert rig.panel.is_active() is True


def test_the_empty_state_survives_a_rate_declaration(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    rig.session.set_sample_rate_declaration(
        rig.doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=FS * 2))
    QtWidgets.QApplication.processEvents()
    assert rig.view.measure_empty() is True and _regions(rig) == []
    assert rig.view.measure_bounds() is None and rig.panel.is_active() is True


def test_a_standing_mask_still_survives_a_rate_declaration_by_sample_index(rig):
    """The thing the empty state must not have broken: the carry-by-index of a real mask."""
    zoom_middle(rig, 0.5)
    _empty(rig)
    _mask(rig, 40_000, 45_000)
    rig.session.set_sample_rate_declaration(
        rig.doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=FS * 2))
    QtWidgets.QApplication.processEvents()
    assert rig.view.measure_bounds() == (40_000, 45_000)


def test_hiding_the_target_pane_ends_measure_even_when_the_mask_is_empty(rig):
    _empty(rig)
    active = Recorder(rig.view.sigMeasureActiveChanged)
    rig.session.set_layer_visible(f"{rig.doc_id}#{CURRENT}", False)
    QtWidgets.QApplication.processEvents()
    assert active.calls and all(call == (False,) for call in active.calls)   # reattach + the panel's own
    assert rig.view.measure_empty() is False and rig.view.measure_target() is None
    assert rig.panel.is_active() is False


def test_activation_is_reported_from_the_pane_not_from_a_mask(rig):
    """The old expression was ``on and region is not None``: true only with a mask."""
    active = Recorder(rig.view.sigMeasureActiveChanged)
    rig.view.set_measure_active(rig.doc_id, CURRENT, True, seed=False)
    assert active.calls == [(True,)]
    rig.session.set_layer_visible(f"{rig.doc_id}#{CURRENT}", False)
    rig.view.set_measure_active(rig.doc_id, CURRENT, True, seed=False)   # its pane is not drawn
    assert active.calls[-1] == (False,)


# ================================================================================================
# everything that must place a mask still does, from the empty state
# ================================================================================================
def test_snap_to_view_places_a_mask_from_the_empty_state(rig):
    lo, hi = zoom_middle(rig, 0.5)
    _empty(rig)
    rig.panel._snap_view_btn.click()
    QtWidgets.QApplication.processEvents()
    b = rig.view.measure_bounds()
    assert b is not None and rig.view.measure_empty() is False
    assert (b[1] - b[0]) / FS == pytest.approx(0.2 * (hi - lo), abs=0.01)
    assert rig.panel.current_measurement() is not None


def test_snap_to_selected_event_places_a_mask_from_the_empty_state(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    rig.panel.set_selected_event(rig.doc_id, CURRENT, 48_000, 49_500)
    rig.panel._snap_event_btn.click()
    assert rig.view.measure_bounds() == (48_000, 49_500)


def test_a_plain_double_click_places_the_mask_from_the_empty_state(rig):
    c = sum(zoom_middle(rig, 0.5)) / 2
    _empty(rig)
    double_click(rig.view, point_at(rig.view, rig.target_vb, c))
    assert rig.view.measure_bounds() is not None and rig.view.measure_empty() is False


def test_go_to_measurement_places_a_mask_from_the_empty_state(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    _mask(rig, 40_000, 41_000)
    assert rig.panel.add_current() is True                        # one saved row
    rig.view.discard_measure_region()
    assert rig.view.measure_empty() is True
    rig.view.zoom_x_to(0.0, 10.0)

    assert rig.panel.navigate_to_row(0) is True

    assert rig.view.measure_bounds() == (40_000, 41_000)
    lo, hi = rig.x_range()
    assert hi - lo < 2.0                                          # and it zoomed to the span


# ================================================================================================
# saving: no mask, no row -- and a lagging readout is never what is saved
# ================================================================================================
def test_add_after_a_discard_saves_nothing_a_second_time(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    _mask(rig, 40_000, 41_000)
    assert rig.panel.add_current() is True
    assert len(rig.panel.measurements_df) == 1
    rig.view.discard_measure_region()                              # what an add now does

    assert rig.panel.add_current() is False                       # Ctrl+M
    assert len(rig.panel.measurements_df) == 1


def test_add_reads_the_mask_afresh_even_when_the_readout_still_describes_a_discarded_one(rig):
    """The belt to the signal's braces: no ``sigMeasureCleared`` reaches the panel here."""
    zoom_middle(rig, 0.5)
    _empty(rig)
    _mask(rig, 40_000, 41_000)
    assert rig.panel.current_measurement() is not None
    rig.view.sigMeasureCleared.disconnect(rig.panel._on_trace_measure_cleared)
    rig.view.discard_measure_region()
    assert rig.panel.current_measurement() is not None            # stale, as the old code left it
    assert rig.panel.add_current() is False
    assert len(rig.panel.measurements_df) == 0
    assert rig.panel.current_measurement() is None                 # and it dropped the stale readout


def test_add_saves_the_mask_as_it_is_now_when_the_readout_lags_the_debounce(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    _mask(rig, 40_000, 41_000)
    xs = rig.xs()
    # A drag frame: only the debounced signal fired, so the readout still describes 40k-41k.
    rig.view.preview_measure_region(float(xs[60_000]), float(xs[62_000]))
    assert rig.panel.current_measurement()["start_idx"] == 40_000

    assert rig.panel.add_current() is True

    row = rig.panel.measurements_df.iloc[0]
    assert (int(row["start_idx"]), int(row["end_idx"])) == (60_000, 62_000)
    assert rig.panel.current_measurement()["start_idx"] == 60_000


def test_an_empty_mask_blocks_the_other_readers_of_the_selection(rig):
    _empty(rig)
    assert rig.panel._selected_update_candidate() is None
    assert rig.view.translate_measure_region(0.5) is False
    assert rig.panel.exact_measurement() is None
    assert rig.panel.can_measure() is True


# ================================================================================================
# moving along the trace with no mask carries nothing
# ================================================================================================
def test_next_and_previous_window_carry_no_mask_when_there_is_none(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    before = rig.x_range()
    assert rig.bar.step_window(+1) is True
    after = rig.x_range()
    assert after[0] > before[0]
    assert rig.view.measure_empty() is True and _regions(rig) == []
    assert rig.bar.step_window(-1) is True
    assert rig.x_range()[0] == pytest.approx(before[0], abs=1e-6)
    assert rig.view.measure_empty() is True


def test_next_window_still_carries_a_standing_mask(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    _mask(rig, 49_000, 50_000)
    before = rig.view.measure_bounds()
    assert rig.bar.next_view() is True
    after = rig.view.measure_bounds()
    assert after[0] > before[0] and (after[1] - after[0]) == pytest.approx(before[1] - before[0], abs=2)


# ================================================================================================
# the Navigator band follows the mask through every state
# ================================================================================================
def test_the_navigator_band_appears_with_a_mask_and_goes_with_it(rig):
    zoom_middle(rig, 0.5)
    _empty(rig)
    assert _band_visible(rig) is False
    _mask(rig, 40_000, 45_000)
    assert _band_visible(rig) is True
    rig.view.discard_measure_region()
    assert _band_visible(rig) is False
    _mask(rig, 41_000, 46_000)
    assert _band_visible(rig) is True
    rig.panel.set_active(False)
    assert _band_visible(rig) is False
