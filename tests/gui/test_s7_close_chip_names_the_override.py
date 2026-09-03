"""S7-D3 / S7-X8 — the calibration chip says WHY there is no current channel.

``workflow_status`` derives one sentence for "this recording resolves to no current channel
because YOUR 'Treat this channel as' override took it away", and carries it on
``CalibrationStatus.issue``. Three of its four consumers rendered it — the CALIBRATE, FILTER
and DETECT reasons on the workflow strip — while ``CalibrationChip.set_status`` was gated on
``state is DAMAGED`` and fell back to its generic tooltip, so the strip read

    "No current channel on this recording to calibrate (IN15 is declared current by the file
     but your Calibration-panel override treats it as voltage)."

while the chip one glance away read

    "This recording has no live current channel to calibrate."

flatly denying a channel the user had just told the app about, on the widget whose whole job
is saying whether the reported magnitudes can be trusted. The producer never needed changing;
the consumer's guard did.

The DAMAGED case keeps its own remedy ("lower the gain"), which would be nonsense advice for
an override, so the suffix is chosen per state — that split is what
``test_the_override_tooltip_does_not_tell_the_user_to_lower_a_gain`` pins.
"""

from __future__ import annotations

import pytest

from nanoscope.gui.status_chip import (
    AMBER,
    NEUTRAL,
    CalibrationChip,
)
from nanoscope.gui.workflow_status import CalibrationState, CalibrationStatus

OVERRIDE_REASON = (
    "No current channel on this recording to calibrate (IN15 is declared current by the "
    "file but your Calibration-panel override treats it as voltage)."
)
DAMAGE_REASON = (
    "channel 2 (IN15) AFTER calibration (gain 1e+38, offset 0): every one of 749,651 "
    "samples is non-finite (NaN/Inf) ... Lower the gain, or work in the raw units."
)


@pytest.fixture()
def chip(qapp):
    widget = CalibrationChip()
    yield widget
    widget.deleteLater()


def test_a_no_channel_status_carrying_a_reason_shows_that_reason(chip):
    chip.set_status(CalibrationStatus(state=CalibrationState.NO_CHANNEL,
                                      label="No current channel",
                                      issue=OVERRIDE_REASON))
    assert OVERRIDE_REASON in chip.toolTip()
    assert "your Calibration-panel override" in chip.toolTip()


def test_the_chip_and_the_strip_give_one_reason_not_two(chip):
    """The generic sentence must not survive beside the specific one — that pairing is the
    contradiction the defect was: two surfaces, two answers, one of them false."""
    chip.set_status(CalibrationStatus(state=CalibrationState.NO_CHANNEL,
                                      label="No current channel",
                                      issue=OVERRIDE_REASON))
    assert "has no live current channel to calibrate" not in chip.toolTip()


def test_the_override_tooltip_does_not_tell_the_user_to_lower_a_gain(chip):
    chip.set_status(CalibrationStatus(state=CalibrationState.NO_CHANNEL,
                                      label="No current channel",
                                      issue=OVERRIDE_REASON))
    assert "lower the gain" not in chip.toolTip().lower()
    assert chip.toolTip().endswith("Click to open the Calibration panel.")


def test_a_no_channel_status_with_no_reason_keeps_the_generic_tooltip(chip):
    """The recording genuinely has no current channel and no override is involved: the
    generic sentence is exact, and inventing a cause would be worse than saying nothing."""
    chip.set_status(CalibrationStatus(state=CalibrationState.NO_CHANNEL,
                                      label="No current channel"))
    assert chip.toolTip() == "This recording has no live current channel to calibrate."


def test_the_chip_still_reads_no_current_channel_and_stays_neutral(chip):
    """The visible chip text and colour are unchanged — the reason belongs in the tooltip,
    not in a status-bar chip that would grow under the pointer."""
    chip.set_status(CalibrationStatus(state=CalibrationState.NO_CHANNEL,
                                      label="No current channel",
                                      issue=OVERRIDE_REASON))
    assert chip.text() == "No current channel"
    assert chip.accent() == NEUTRAL


def test_the_damaged_channel_keeps_its_own_sentence_and_its_own_remedy(chip):
    """The case the guard was written for still behaves exactly as it did."""
    chip.set_status(CalibrationStatus(state=CalibrationState.DAMAGED,
                                      label="Gain set",
                                      detail="×1e+38 → pA",
                                      issue=DAMAGE_REASON))
    assert DAMAGE_REASON in chip.toolTip()
    assert chip.toolTip().endswith("Click to open the Calibration panel and lower the gain.")
    assert chip.accent() == AMBER
