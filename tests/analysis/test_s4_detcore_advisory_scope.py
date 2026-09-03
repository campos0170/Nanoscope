"""VS-04 -- the open-pore advisory may only claim what it actually measured.

THE DEFECT. `level_occupancy_warning` ended with

    "Every depth in the table is the same {D:.1f} pA either way; what changes is dI/I0,
     which reads {here} against this baseline and {there} against the other one."

``D`` is ``abs(alt_offset_pA)``: the SEPARATION between the two levels the baseline probe
found. It is not a depth from the event table and the table's depth column is not
constant. Measured against the app's own exported event CSVs:

    -700 mV   claims 91.9 pA   depth_pA 13.43-167.41 (12x), median 96.13, 20.1% within 5%
    -500 mV   claims 55.8 pA   depth_pA 10.99-126.04,       median 56.34, 13.0% within 5%
    -600 mV   claims 86.1 pA   depth_pA 12.66-162.66,       median 87.59, 32.2% within 5%

and the dI/I0 the same sentence quoted was off the table's own median frac_depth by
5.80% / 5.77% / 1.50%. The sentence is reprinted verbatim for every pooled recording in
the voltage-dependence DETECTION CAVEATS block, directly above a fitted "Median depth
(pA)" trend -- the moment the reader is deciding how far to trust that law.

WHAT IS PINNED HERE. The advisory names its own basis: the number is the separation, it
says so, it says explicitly that it is NOT a table depth, it gives the dI/I0 reading of
that separation against each candidate open pore, and it does not go on to predict the
table -- under the other reading the baseline every row is measured from moves too, so
both the numerator and the denominator change and this block cannot say by how much.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.detect import detect_events

FS = 100_000.0

GONE = "Every depth in the table is the same"


def test_the_sentence_no_longer_calls_the_separation_a_depth():
    """The -700 mV numbers, straight into the writer: 91.9 pA of separation against a
    -48.5 pA tracked level."""
    txt = B.level_occupancy_warning(
        0.90, alt_occupancy=0.08, alt_offset_pA=-91.9, tracked_pA=-48.5)
    assert GONE not in txt, txt
    assert "91.9 pA SEPARATION between the two levels" in txt
    assert "is NOT a depth from the event table" in txt
    assert "measured per event and varies from event to event" in txt
    # Both readings of dI/I0 for THAT separation are still there -- that half was right.
    here, there = 91.9 / 48.5, 91.9 / 140.4
    assert f"{here:.3f} against this baseline" in txt, txt
    assert f"{there:.3f} against the other one" in txt, txt
    # ... and it refuses to predict the table it cannot predict.
    assert "row by row, is not something this block can tell you" in txt
    # The actionable advice the check has always carried is untouched.
    assert "COMPARE THE BASELINE with the open-pore current you know" in txt


def test_the_older_form_without_the_two_currents_is_unchanged():
    """A health block restored from a project written before ``alt_offset_pA`` existed
    reaches the second branch, which never made the claim and must not grow one."""
    txt = B.level_occupancy_warning(0.90, alt_occupancy=0.08)
    assert GONE not in txt
    assert "SEPARATION" not in txt
    assert "brief returns to the open pore" in txt


def _two_level_trace(n: int = 600_000, seed: int = 4) -> np.ndarray:
    """A pore with a genuine SECOND LEVEL (40 long states at exactly -120 pA, 4.6% of
    the recording -- what `alternative_level` finds and `ALT_LEVEL_OCCUPANCY_MIN`
    triggers on) and, separately, events of MANY different depths. That is the shape the
    old sentence described with a single number."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 1000.0)
    for i in range(30):
        s = 2_000 + i * (n - 4_000) // 30
        x[s:s + 900] -= 120.0
    for i, d in enumerate(np.linspace(30.0, 220.0, 90)):
        s = 800 + i * (n - 4_000) // 90
        x[s:s + 300] -= float(d)
    return x + rng.normal(0.0, 5.0, n)


def test_on_a_table_whose_depths_span_7x_the_advisory_scopes_its_claim():
    """The end-to-end form of the same thing, through the detector: the depth column
    spans a factor of several, and the advisory that rides on the result no longer
    asserts one depth for all of it."""
    x = _two_level_trace()
    df, prov = detect_events(x, FS, min_dwell_s=1e-4)
    depths = np.asarray(df["depth_pA"], dtype=float)
    assert len(df) > 20 and depths.max() / depths.min() > 3.0, (len(df), depths.min(),
                                                                depths.max())
    warns = [str(w) for w in prov["baseline_warnings"]]
    advisory = [w for w in warns if "NOT CERTIFIED, only chosen" in w]
    assert advisory, ("the fixture must reach the open-pore advisory or this test "
                      "proves nothing", warns)
    txt = advisory[0]
    assert GONE not in txt, txt
    assert "SEPARATION between the two levels" in txt, txt
    quoted = float(txt.split(" pA SEPARATION")[0].split("The ")[-1])
    health = prov["baseline_health"]
    assert quoted == pytest.approx(abs(float(health["alt_level_offset_pA"])), abs=0.05), (
        "the number must be the separation the probe measured, not anything else")
    # The point of the whole defect: most of the table is NOT that number.
    within5 = float(np.mean(np.abs(depths - quoted) < 0.05 * quoted))
    assert within5 < 0.5, (
        "the fixture is supposed to have a spread-out depth column; if most rows sit "
        "on the quoted number this test proves nothing", within5)
    assert "is NOT a depth from the event table" in txt
