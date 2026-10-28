"""S8-DATA1 -- ONE rule for "is this detection table on its recording's live clock?".

MEASURED DEFECT (sweep 8). After Inspector > File Info > Sample rate / time unit... the app
marks every detection made at the old rate stale ("every start time, dwell, rise/fall time
and ECD in this table was computed at the old rate") and says stored detections are never
silently re-scaled. Three in-place editing doors never asked: a hand-added event, an
Event-inspector edge drag and Merge wrote ``start_idx / fs_live`` and ``(end - start) / fs_live``
into the table the amber line had just called entirely old-clock -- one table, two clocks.
The fourth door that re-measures DID ask, but with its own private copy of the rule
(``remeasure._require_same_rate``), and the amber line had a third copy
(``AnalysisPanel._sample_rate_staleness``). Three copies of one tolerance is how a refusal ends
up disagreeing with the banner: a door that refuses where the banner offers no Re-run strands
the user, a door that edits where the banner is up writes the second clock.

THE FIX (the rule). ``analysis.events.stale_rate_hz`` is the one rule (the stamped rate, when it
is not the live one), ``stale_rate_reason`` the one sentence built on it (the amber line's own).
The banner, the three editing doors, the re-measure front gate and the kernel's
``_require_same_rate`` all ask there.

These tests pin the rule itself, headless: what counts as the same rate, what counts as no
stamp (nothing to judge by, so nothing is refused), what the sentence names, and that the
kernel's guard and the banner's question give the same answer for every pair of rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import (
    RATE_STAMP_RTOL,
    EventSet,
    empty_df,
    stale_rate_hz,
    stale_rate_reason,
)
from nanoscope.analysis.remeasure import _require_same_rate, remeasure_events

OLD = 50_000.0
NEW = 40_000.0


def test_a_table_on_the_live_rate_is_not_stale():
    assert stale_rate_hz({"fs": OLD}, OLD) is None
    assert stale_rate_reason({"fs": OLD}, OLD) is None


def test_rounding_noise_in_the_rate_is_the_same_clock_a_real_gap_is_not():
    """The tolerance is the kernel's own (1e-9, relative): a rate re-read off the same header
    or column differs in the last digits at most; a declaration differs by far more."""
    assert stale_rate_hz({"fs": OLD}, OLD * (1 + 0.5 * RATE_STAMP_RTOL)) is None
    assert stale_rate_hz({"fs": OLD}, OLD * (1 + 10 * RATE_STAMP_RTOL)) == OLD
    assert stale_rate_hz({"fs": 50_000.001192}, 50_000.001192) is None
    assert stale_rate_hz({"fs": OLD}, NEW) == OLD


def test_the_sentence_is_the_amber_lines_and_names_both_rates_and_the_way_forward():
    said = stale_rate_reason({"fs": OLD}, NEW)
    assert said is not None
    assert "The sample rate changed since this detection ran (50,000 Hz → 40,000 Hz)" in said
    assert ("every start time, dwell, rise/fall time and ECD in this table was computed at "
            "the old rate") in said
    assert said.endswith("Re-run detection under the declared rate.")


def test_a_recording_that_has_lost_its_rate_is_stale_for_a_stamped_table():
    said = stale_rate_reason({"fs": OLD}, None)
    assert said is not None and "(50,000 Hz → unknown)" in said
    assert stale_rate_hz({"fs": OLD}, None) == OLD


@pytest.mark.parametrize("provenance", [
    None, {}, {"fs": None}, {"fs": "fast"}, {"fs": 0}, {"fs": -50_000.0},
    {"fs": float("nan")}, {"fs": float("inf")}, "not a mapping", 50_000.0,
])
def test_a_table_with_no_usable_stamp_is_never_judged(provenance):
    """An older table, a payload that is not an EventSet, a stamp that is not a finite
    positive rate: there is nothing to compare with, so nothing is said and nothing is
    refused -- the reading the amber line has always taken."""
    assert stale_rate_hz(provenance, NEW) is None
    assert stale_rate_reason(provenance, NEW) is None


def test_a_real_eventset_provenance_is_read_through_the_same_door():
    es = EventSet(empty_df(), {"fs": OLD, "polarity": -1})
    assert stale_rate_hz(es.provenance, OLD) is None
    assert stale_rate_hz(es.provenance, NEW) == OLD


@pytest.mark.parametrize("stamped, live", [
    (OLD, OLD), (OLD, NEW), (OLD, OLD * (1 + 1e-12)), (OLD, OLD * (1 + 1e-6)),
    (49_999.999998908606, 50_000.0), (50_000.001192, 50_000.0), (100.0, 100_000.0),
])
def test_the_kernels_guard_and_the_banners_question_never_disagree(stamped, live):
    """``remeasure._require_same_rate`` used to carry its own copy of the tolerance. The door
    that refuses and the line that offers Re-run must answer alike for every pair."""
    prov = {"fs": stamped}
    refuses = False
    try:
        _require_same_rate(prov, live)
    except ValueError:
        refuses = True
    assert refuses == (stale_rate_reason(prov, live) is not None)
    assert refuses == (stale_rate_hz(prov, live) is not None)


def test_remeasure_events_still_refuses_a_table_detected_at_another_rate():
    """The backstop under the GUI's front gate is unchanged: scripts that hand the kernel
    the wrong rate still get the sentence addressed to them, and no row moves."""
    rng = np.random.default_rng(1)
    x = (100.0 + rng.normal(0.0, 1.0, 20_000)).astype(np.float32)
    df = empty_df()
    es = EventSet(df, {"fs": OLD, "polarity": -1})
    with pytest.raises(ValueError, match="detected at 50000.0 Hz but this re-measure was handed"):
        remeasure_events(es, x, NEW)
    assert isinstance(es.df, pd.DataFrame) and len(es.df) == 0
