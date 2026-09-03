"""``nanoscope.core.clock`` — the app's ONE answer to "which clock is this second on?".

These are GUARDS ON NEW CODE, not reproductions of the defect: before High-6 this
module did not exist, so nothing here can fail in a pre-fix tree. The tests that
reproduce the defect are the SURFACE tests in ``tests/gui/test_clock_invariant.py`` —
the axis label, the crosshair, the event-table header and the Measure→PSD hand-off —
and those fail before and pass after.

The defect this module exists to close, measured on the owner's apoferritin
recordings: the trace x-axis is drawn on the instrument clock (``arange(n)/fs +
meta.t0_s``, with ``t0_s`` = 10.2526 / 61.3034 / 0 s), while the event table and every
analysis window in seconds are 0-based on the recording — and all of them were
labelled ``s``. Reading a quiet window off the trace and typing it into the PSD panel
returned total RMS 22.66 pA / α 1.822 instead of 3.99 pA / α 0.985.
"""

from __future__ import annotations

import pytest

from nanoscope.core import clock

#: The −600 mV apoferritin recording's own t₀, used verbatim so the numbers here are
#: the numbers in the report.
T0_APO = 61.3034
FS = 50_000.0


def test_the_two_clocks_convert_both_ways_through_one_offset():
    off = clock.clock_offset_s(_Meta(t0_s=T0_APO, sample_rate_hz=FS))
    assert off == pytest.approx(T0_APO)
    assert clock.to_recording_s(off, T0_APO + 1.5) == pytest.approx(1.5)
    assert clock.to_instrument_s(off, 1.5) == pytest.approx(T0_APO + 1.5)


def test_a_layer_alignment_nudge_is_part_of_the_offset():
    """The trace is drawn at ``t0_s + time_offset_s``, so THAT is what separates the
    axis from the event table — not ``t0_s`` alone. An aligned layer whose nudge was
    dropped would be labelled with an offset it is not drawn at."""
    off = clock.clock_offset_s(_Meta(t0_s=T0_APO, sample_rate_hz=FS), 0.25)
    assert off == pytest.approx(T0_APO + 0.25)


def test_a_recording_with_no_sample_rate_has_no_clock_at_all():
    """``Session.x_axis`` refuses to fabricate seconds there, and so does this: None
    is "there is no clock", not "the offset is zero"."""
    assert clock.clock_offset_s(_Meta(t0_s=0.0, sample_rate_hz=None)) is None
    assert clock.to_recording_s(None, 5.0) is None
    assert clock.axis_label("sample", clock.INSTRUMENT, None) == "sample"


def test_t0_is_printed_to_the_resolution_the_recordings_differ_at():
    """``%g`` would render 61.3034 as 61.3 and invite a number 3.4 ms wrong."""
    assert clock.format_t0(T0_APO) == "61.3034"
    assert clock.format_t0(10.2526) == "10.2526"


@pytest.mark.parametrize("which", [clock.INSTRUMENT, clock.RECORDING])
def test_every_label_names_its_clock_and_its_conversion(which):
    label = clock.axis_label("s", which, T0_APO)
    assert clock.CLOCK_SHORT[which] in label
    assert "61.3034" in label
    note = clock.field_note(which, T0_APO)
    assert "61.3034" in note
    assert ("subtract" in note) if which is clock.INSTRUMENT else ("add" in note)


def test_a_t0_of_zero_still_names_its_clock_and_says_the_two_agree():
    """THE −700 mV FILE. Its t₀ is 0, so a label that only appeared when the clocks
    differed would leave a user believing the app has one clock — until the day it
    does not."""
    label = clock.axis_label("s", clock.INSTRUMENT, 0.0)
    assert clock.CLOCK_SHORT[clock.INSTRUMENT] in label
    assert "both clocks read alike" in label


def test_both_clocks_prints_one_instant_twice_only_when_that_is_informative():
    assert clock.both_clocks(T0_APO, recording_s=10.0) == (
        "10.0000 s in  ·  71.3034 s instrument")
    assert clock.both_clocks(0.0, recording_s=10.0) == "10.0000 s"


class _Meta:
    def __init__(self, *, t0_s, sample_rate_hz):
        self.t0_s = t0_s
        self.sample_rate_hz = sample_rate_hz
