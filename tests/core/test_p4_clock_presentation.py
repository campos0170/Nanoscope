import pytest

from nanoscope.core import clock


def test_clock_display_conversion_is_explicit_and_does_not_change_default_or_clock_arithmetic():
    offset = 61.3034
    assert clock.format_t0(offset) == '61.3034'
    assert clock.format_t0(offset, display_unit='ms') == '61303.4'
    assert clock.to_instrument_s(offset, .002) == pytest.approx(61.3054)
    assert clock.both_clocks(offset, recording_s=.002, display_unit='ms', decimals=1) == '2.0 ms in  ·  61305.4 ms instrument'
    assert '61303.4 ms' in clock.axis_label('s', clock.INSTRUMENT, offset, display_unit='ms')
    assert 'milliseconds' in clock.field_note(clock.RECORDING, offset, display_unit='ms')
    assert clock.axis_label('sample', clock.RECORDING, None, display_unit='ms') == 'sample'


def test_segment_origin_and_layer_nudge_are_stated_once_in_ms():
    offset = clock.ClockOffset(.025, rebased_from_s=10.)
    label = clock.axis_label('s', clock.INSTRUMENT, offset, display_unit='ms')
    assert '10000.0 ms' in label and '25.0 ms' in label and 'rebased' in label
    assert clock.both_clocks(offset, recording_s=.002, display_unit='ms', decimals=1) == '2.0 ms in  ·  10027.0 ms instrument'
