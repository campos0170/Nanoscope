"""DET3-01 — the Detection tab's polarity line must be ONE measurement.

THE DEFECT
==========
CF2-4 gave :func:`nanoscope.analysis.detect.detect_polarity` a second statistic (the
coherent SAMPLE mass) that can overrule or corroborate the run vote. It replaced
``evidence['confidence']`` with the sample-mass confidence and left ``n_down``/``n_up``
as the run counts. ``DetectionPanel.set_polarity_evidence`` renders all of them in one
sentence::

    f"{physical} ({direction}): {n_down} down / {n_up} up (confidence {conf:.2f}, {verdict})"

so the line the manual tells the user to read before trusting the events came out as
"blockade (down): 5 down / 10 up (confidence 0.81, measured)" — the counts say UP won
10:5, the direction says down, and 0.81 is not |5-10|/15 = 0.33 nor anything else a
reader can compute from the line. On the owner's own -600 mV recording at a 20 ms
savgol window the same line read "enhancement (down): 247 down / 402 up (confidence
0.73)". The mirror case was the corroborated one: "25 down / 22 up (confidence 0.06)"
printed for a direction the app itself rates at 0.88 by the second statistic and
deliberately does NOT warn about, so the only reading available was "worthless".

THE RULE THIS PINS, and it is a rule about the SEAM, not about one widget: the counts,
the ratio and the confidence in the evidence dict are one vote, always — whichever
statistic decided is the statistic reported, and ``reported_vote`` names it. The
readout is then checkable BY ARITHMETIC from the line itself, which is what these
tests do: they parse the rendered sentence and recompute the confidence from the two
numbers printed in it. That predicate holds however the line is worded, so it survives
a rewording of the widget and fails again if either half is ever taken from a different
computation.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -----------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events, detect_polarity
from nanoscope.analysis.filters import savgol
from nanoscope.gui.panels import DetectionPanel

FS = 100_000.0

#: "blockade (down): 1734 down / 177 up (confidence 0.81, measured)"
_LINE = re.compile(
    r"^(?P<physical>\w+) \((?P<direction>\w+)\): (?P<down>\d+) down / (?P<up>\d+) up "
    r"\(confidence (?P<conf>[\d.]+), (?P<verdict>\w+)\)$"
)


def _read(panel: DetectionPanel, evidence: dict) -> dict:
    """Render the evidence through the real widget and parse the sentence back."""
    panel.set_polarity_evidence(evidence)
    text = panel.polarity_readout.text()
    m = _LINE.match(text)
    assert m is not None, f"the readout no longer parses as one vote: {text!r}"
    return {"text": text, "down": int(m["down"]), "up": int(m["up"]),
            "conf": float(m["conf"]), "direction": m["direction"],
            "physical": m["physical"], "verdict": m["verdict"]}


def _equal_blockades() -> np.ndarray:
    """5 identical 2.5-4 ms blockades — the CF2-4 fixture, whose savgol edge lobes
    outvote the events 10:5 in the run count."""
    n = 100_000
    rng = np.random.default_rng(0)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
                 (70_000, 70_200), (85_000, 85_350)]:
        x[s:e] -= 30.0
    return x


def _mixed_blockades() -> np.ndarray:
    """5 deep 5 ms events among 20 shallow 0.4 ms ones: the run vote is a coin flip
    (25 vs 22) while the sample mass agrees with it decisively."""
    n = 300_000
    rng = np.random.default_rng(0)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    pos = np.linspace(5000, n - 5000, 25).astype(int)
    rng.shuffle(pos)
    for i, s in enumerate(pos):
        x[s:s + (500 if i < 5 else 40)] -= (40.0 if i < 5 else 12.0)
    return x


@pytest.fixture
def panel(qapp, qtbot):
    p = DetectionPanel()
    qtbot.addWidget(p)
    return p


# ===========================================================================
# 1. THE INVARIANT — the printed confidence belongs to the printed counts
# ===========================================================================
def test_the_overruled_run_vote_readout_is_checkable_from_its_own_line(panel):
    """FACE A. The sample mass overrules the run vote; the sentence must still be one
    arithmetic. Recomputed from the two numbers ON SCREEN, not from the dict."""
    x = savgol(_equal_blockades(), FS, window_s=0.001, polyorder=3)
    _, prov = detect_events(x, FS)
    ev = prov["polarity_evidence"]
    assert ev["decided_by"] == "coherent_samples", "fixture no longer arbitrates"

    shown = _read(panel, ev)
    recomputed = abs(shown["down"] - shown["up"]) / (shown["down"] + shown["up"])
    assert shown["conf"] == pytest.approx(recomputed, abs=0.005), (
        f"the confidence printed in {shown['text']!r} is not derivable from the two "
        "counts printed beside it"
    )
    # ...and the direction agrees with those counts, which is the half a reader
    # notices first: 'down' beside 5 down / 10 up was the original complaint.
    assert shown["direction"] == ("down" if shown["down"] > shown["up"] else "up")
    assert ev["reported_vote"] == "coherent_samples"


def test_the_corroborated_run_vote_reports_the_statistic_that_settled_it(panel):
    """FACE B. Two statistics agreeing is not a weak measurement, and the line must not
    read like one: the app suppresses the low-confidence warning here, so a printed
    0.06 was the only thing the user had and it said the opposite."""
    x = savgol(_mixed_blockades(), FS, window_s=0.001, polyorder=3)
    _, prov = detect_events(x, FS)
    ev = prov["polarity_evidence"]
    assert ev.get("corroborated_by_samples") is True, "fixture no longer corroborates"
    assert prov["polarity_warning"] is None, "fixture no longer suppresses the warning"

    shown = _read(panel, ev)
    recomputed = abs(shown["down"] - shown["up"]) / (shown["down"] + shown["up"])
    assert shown["conf"] == pytest.approx(recomputed, abs=0.005), shown["text"]
    assert shown["conf"] > 0.5, (
        f"a measurement the app rates at {ev['sample_confidence']:.2f} and declines to "
        f"warn about is still being printed as a coin flip: {shown['text']!r}"
    )
    assert shown["direction"] == "down"


@pytest.mark.parametrize("window_s", [0.001, 0.005, 0.020])
def test_every_rendered_line_is_one_vote_across_the_filter_sweep(panel, window_s):
    """THE PROPERTY, not the two reproductions. Sweeping the savgol window walks the
    run vote from decisive (249:2) through arbitrated to corroborated; at no point may
    the sentence stop being self-consistent."""
    x = savgol(_equal_blockades(), FS, window_s=window_s, polyorder=3)
    pol, ev = detect_polarity(x, float(np.median(x)))
    shown = _read(panel, ev)
    total = shown["down"] + shown["up"]
    assert shown["conf"] == pytest.approx(abs(shown["down"] - shown["up"]) / total,
                                          abs=0.005), shown["text"]
    assert shown["direction"] == ("down" if pol == -1 else "up")


# ===========================================================================
# 2. NOTHING IS REPLACED SILENTLY — the overruled vote stays on the record
# ===========================================================================
def test_the_run_counts_survive_and_the_warning_quotes_them_as_runs():
    """The fix moves what the READOUT shows; it must not lose what was overruled. The
    arbitration warning is the one surface that talks about the run vote as runs, and
    it must not start printing sample counts under the words "coherent runs"."""
    x = savgol(_equal_blockades(), FS, window_s=0.001, polyorder=3)
    _, prov = detect_events(x, FS)
    ev = prov["polarity_evidence"]

    assert (ev["n_down_runs"], ev["n_up_runs"]) == (5, 10), ev
    assert ev["run_confidence"] == pytest.approx(5 / 15, abs=1e-9)
    w = prov["polarity_warning"]
    assert "5 down / 10 up coherent runs" in w, w
    assert f"{ev['n_down_samples']} down vs {ev['n_up_samples']} up samples" in w, w


def test_a_decisive_run_vote_still_reports_the_runs_and_stamps_nothing():
    """The CONTROL. The byte-identical guarantee for every decisive measurement is
    what keeps this change out of the ordinary path: no arbitration keys at all, and
    the counts are the run counts they always were."""
    pol, ev = detect_polarity(_equal_blockades(), 100.0)
    assert pol == -1 and ev["confidence"] == 1.0
    assert (ev["n_down"], ev["n_up"]) == (5, 0)
    for key in ("decided_by", "run_confidence", "sample_confidence",
                "corroborated_by_samples", "reported_vote", "n_down_runs",
                "n_up_runs"):
        assert key not in ev, f"a decisive run vote stamped {key}"


def test_the_ratio_is_the_ratio_of_the_counts_that_are_reported():
    """One arithmetic all the way down the dict: a ratio taken from the run counts
    under a sample-mass confidence is the same defect one field further along."""
    x = savgol(_equal_blockades(), FS, window_s=0.001, polyorder=3)
    _, ev = detect_polarity(x, float(np.median(x)))
    assert ev["reported_vote"] == "coherent_samples"
    assert ev["ratio"] == pytest.approx(ev["n_down"] / ev["n_up"])
    assert ev["n_down"] == ev["n_down_samples"] and ev["n_up"] == ev["n_up_samples"]
