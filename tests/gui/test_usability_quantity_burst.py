"""A draw-time quantity witness is metadata, not a new author edit or a burst boundary."""
from dataclasses import replace

import nanoscope  # noqa: F401
import pandas as pd

from nanoscope.core.figure_quantity_basis import read_quantity_bases
from nanoscope.core.quantity_display import AxisDisplayBasis, quantity_for
from nanoscope.gui.figure_text import AxisPin
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from .test_p0_reset_on_quantity_change import vdep, _pick, _section


def _assert_current(page, key):
    widget = page._appearance_widget
    for state in (widget.state(), page._appearance, widget._history.base):
        assert read_quantity_bases(state.quantity_display)['left'].quantity.key == key
    assert page.plot_item().getAxis('left')._nanoscope_quantity_basis.quantity.key == key
    assert read_quantity_bases(page.appearance_state()['quantity_display'])['left'].quantity.key == key


def test_metric_change_keeps_pending_size_run_and_current_witness_through_undo(vdep):
    page, _ = vdep
    widget = page._appearance_widget
    widget._burst.setInterval(60000)
    start = widget.state().title_size
    depth = widget._history.depth()
    widget.title_size.setValue(15)
    timer_id = widget._burst.timerId()
    _section(page, 'Analysis')
    _pick(page._metric_combo, text='Median depth (pA)')
    assert widget._burst.isActive() and widget._burst.timerId() == timer_id
    assert widget._history.depth() == depth
    _assert_current(page, 'median_depth')
    before = page._plot_data().copy(deep=True)
    widget.title_size.setValue(16)
    assert widget.undo()
    assert widget.state().title_size == start
    assert widget._history.depth() == (depth[0], 1)
    _assert_current(page, 'median_depth')
    assert widget.redo()
    assert widget.state().title_size == 16
    _assert_current(page, 'median_depth')
    capsule = widget.capture_history_capsule()
    widget.restore_history_capsule(capsule)
    _assert_current(page, 'median_depth')
    pd.testing.assert_frame_equal(page._plot_data(), before)


def test_witness_adoption_preserves_pending_values_pins_and_history(qtbot):
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    old = {'left': AxisDisplayBasis(quantity_for('dwell_s'))}
    new = {'left': AxisDisplayBasis(quantity_for('depth_pA'))}
    widget.migrate_quantity_bases(old)
    widget.apply_appearance(replace(widget.state(), title_size=12,
        axes=(('left', AxisPin(1, 5, tick_major=2)),)))
    widget._burst.setInterval(60000)
    widget.title_size.setValue(15)
    timer_id = widget._burst.timerId()
    prior, base = widget.state(), widget._history.base
    past, future = tuple(widget._history._past), tuple(widget._history._future)
    changed = []; restored = []
    widget.changed.connect(changed.append)
    widget.restored.connect(lambda: restored.append(True))
    current = widget.adopt_quantity_bases(new)
    assert widget._burst.isActive() and widget._burst.timerId() == timer_id
    assert replace(current, quantity_display=prior.quantity_display) == prior
    assert replace(widget._history.base, quantity_display=base.quantity_display) == base
    assert tuple(widget._history._past) == past and tuple(widget._history._future) == future
    assert current.axis_pin('left') == AxisPin(1, 5, tick_major=2)
    assert not changed and not restored
    assert current.title_size == 15 and widget._history.base.title_size == 12
    assert read_quantity_bases(current.quantity_display)['left'].quantity.key == 'depth_pA'


def test_older_quantity_history_replays_style_with_current_metadata_and_pins(vdep):
    page, _ = vdep
    widget = page._appearance_widget
    pin = AxisPin(1., 3., tick_major=1.)
    widget.apply_appearance(replace(widget.state(), title_size=11,
        axes=(('left', pin),)))
    widget.apply_appearance(replace(widget.state(), title_size=13))
    old_key = read_quantity_bases(widget._history._past[-1].quantity_display)['left'].quantity.key
    assert old_key != 'median_depth'
    _section(page, 'Analysis')
    _pick(page._metric_combo, text='Median depth (pA)')
    before = page._plot_data().copy(deep=True)
    # The metric change clears its Y pin as one legitimate reset step.
    assert widget.state().axis_pin('left') == AxisPin()
    for replay, expected, expected_pin in (
            (widget.undo, 13, pin), (widget.undo, 11, pin),
            (widget.redo, 13, pin), (widget.redo, 13, AxisPin())):
        assert replay()
        assert widget.state().title_size == expected
        assert widget.state().axis_pin('left') == expected_pin
        _assert_current(page, 'median_depth')
        pd.testing.assert_frame_equal(page._plot_data(), before)
    capsule = widget.capture_history_capsule()
    widget.restore_history_capsule(capsule)
    _assert_current(page, 'median_depth')
