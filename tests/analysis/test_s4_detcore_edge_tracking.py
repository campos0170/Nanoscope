"""B4DA-02 -- a baseline that never TRACKED the trace at its own edge may not read green.

THE DEFECT. On 4 s of nothing but noise and slow open-pore wander (100 kHz, 5 pA noise,
``1000 - 100t + 250 sin(pi t)``), `characterisation_baseline` at the shipped default
0.5 / 0.1 s was **+69.5 pA wrong over the first 10 ms** -- 13.9x the noise floor --
against 1.23 pA mid-trace, and every one of the nine health checks read ``ok`` with zero
warnings. `detect_events` then shipped **18 events, 17 of them accepted**, all inside the
first 83 ms, on a trace that contains none.

The mechanism is the boundary and nothing else: pass 1 filters with ``mode='reflect'``
(`ml_preprocess._decimated_median`), so at an end its median is biased by the trace's own
SLOPE; when that bias reaches ``mask_sigma`` x the rough sigma the one-sided
`_rough_pass` masks the whole head as one state, pass 2 bridges it, and the baseline is
pinned at a level the trace only reaches later. The same samples with 2 s of the same
wander in front of them are 0.67 pA out, with no events there at all.

WHAT IS PINNED HERE. The ``edge_tracking`` row (`EDGE_OFFSET_SIGMA`, `EDGE_DRIFT_RATIO`):
it fires on that fixture, it is SILENT on a genuine state sitting at the boundary and on
a drifting trace whose edges the estimator does follow, it is present even when it cannot
be measured, and -- the invariant that matters most -- it stays green on the owner's own
recordings at the windows they are analysed with. A check that reddens ordinary good
recordings is worse than the defect it is for.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pathlib

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.detect import detect_events

FS = 100_000.0
SIGMA = 5.0
N = int(FS * 4.0)


def _wander(n: int = N, t0: float = 0.0, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """The dossier's fixture: open-pore wander, ZERO events. Returns (trace, truth)."""
    t = np.arange(n) / FS + t0
    truth = 1000.0 - 100.0 * t + 250.0 * np.sin(2 * np.pi * 0.5 * t)
    rng = np.random.default_rng(seed)
    return truth + rng.normal(0.0, SIGMA, n), truth


def _events_flat(n: int = N, seed: int = 11) -> np.ndarray:
    """A flat 1 nA trace with 60 short blockades -- the healthy control."""
    rng = np.random.default_rng(seed)
    x = 1000.0 + rng.normal(0.0, SIGMA, n)
    for i in range(60):
        s = 2000 + i * 6500
        x[s:s + 200] -= 100.0
    return x


def test_the_untracked_head_is_reported_instead_of_reading_green():
    """The fixture the defect was measured on: the estimate is 69 pA out across the
    head, and the block must now say so rather than certifying it."""
    x, truth = _wander()
    res = B.characterisation_baseline(x, FS, polarity=-1)
    err = np.asarray(res.baseline, float) - truth

    # The error the dossier measured is still there -- this test is about the REPORT,
    # not about a silently changed estimate.
    assert abs(err[:1000].mean()) > 10.0 * SIGMA, err[:1000].mean()
    assert abs(err[50_000:350_000]).mean() < 2.0, "mid-trace was never the problem"

    row = res.health.check("edge_tracking")
    assert row is not None, [c.name for c in res.health.checks]
    assert row.level == "warn", row.message
    assert res.health.ok is False
    assert "never TRACKED" in row.message
    assert "REFLECTION" in row.message
    # It must name WHICH end, and it must not tell the reader to trust the depths there.
    assert "first 0.1 s" in row.message
    assert "DO NOT read a depth" in row.message

    # And it reaches the surfaces that print advisories: the result's warnings tuple
    # (what `AnalysisResult.warnings` and the Stats panel show) and the detector's
    # provenance (what `modules._build_warnings` reads).
    assert any("never TRACKED" in w for w in res.warnings), res.warnings
    _df, prov = detect_events(x, FS, min_dwell_s=1e-4)
    assert any("never TRACKED" in str(w) for w in prov["baseline_warnings"]), (
        "the detector that shipped the 17 accepted head rows still says nothing")


def test_the_number_on_the_row_is_the_offset_it_names():
    """A count counts what it says: ``value`` is the signed pA offset the message
    quotes, measured against the interior -- not a sigma, not a ratio."""
    x, _ = _wander()
    res = B.characterisation_baseline(x, FS, polarity=-1)
    row = res.health.check("edge_tracking")
    assert row.value is not None
    assert f"{row.value:+.2f} pA" in row.message
    b = np.asarray(res.baseline, float)
    w = int(round(B.DEFAULT_BASELINE.tracking_window_s * FS))
    resid = b - x
    expect = float(np.median(resid[:w])) - float(np.median(resid[w:-w]))
    assert row.value == pytest.approx(expect, abs=1e-6)


def test_a_real_state_at_the_boundary_is_not_called_an_estimator_failure():
    """The other half of the criterion. A recording that OPENS inside a blockade is
    the ordinary case (the owner's -600 mV file does exactly that), the bridge over it
    is correct, and reddening it would be the false alarm this check must not raise."""
    x = _events_flat()
    x[0:9000] -= 100.0            # a real 90 ms state at t = 0
    res = B.characterisation_baseline(x, FS, polarity=-1)
    row = res.health.check("edge_tracking")
    assert row.level == "ok", row.message
    assert "STEP the drift cannot explain" in row.message
    assert "Read as signal" in row.message
    assert res.health.warnings == (), res.health.warnings


def test_a_drifting_trace_whose_edges_are_followed_stays_green():
    """Drift as such is not the trigger -- unfollowed drift AT THE EDGE is. This trace
    wanders 250 pA with the same amplitude as the fixture above but is stationary at
    both ends, so pass 1's reflected window is unbiased and the estimate tracks it."""
    t = np.arange(N) / FS
    rng = np.random.default_rng(7)
    x = 1000.0 + 250.0 * np.cos(2 * np.pi * 0.5 * t) + rng.normal(0.0, SIGMA, N)
    res = B.characterisation_baseline(x, FS, polarity=-1)
    row = res.health.check("edge_tracking")
    assert row.level == "ok", row.message
    assert "tracks the trace at both ends" in row.message


def test_the_healthy_control_stays_green():
    x = _events_flat()
    res = B.characterisation_baseline(x, FS, polarity=-1)
    assert res.health.check("edge_tracking").level == "ok"
    assert res.health.warnings == (), res.health.warnings


def test_the_row_is_present_even_when_it_cannot_be_measured():
    """A check missing from the block is not a check that passed -- the rule
    `base_vs_state` and `baseline_intrusion` already follow. A 20 ms segment has no
    interior to compare its ends against, and the row says that in as many words.
    (0.2 s here: two 0.1 s tracking windows, so the head, the tail and an interior
    cannot all exist. A 20 ms segment would be CLAMPED to a shorter window and the row
    would then be measurable, which is the honest outcome for it.)"""
    rng = np.random.default_rng(5)
    x = 1000.0 + rng.normal(0.0, SIGMA, 20_000)      # 0.2 s: two tracking windows
    res = B.characterisation_baseline(x, FS)
    row = res.health.check("edge_tracking")
    assert row is not None, [c.name for c in res.health.checks]
    assert row.level == "ok" and row.value is None
    assert "COULD NOT BE MEASURED" in row.message
    # ... and it must not add a second sentence to a short recording's advisory block,
    # which already carries the length one.
    assert not any("COULD NOT BE MEASURED" in w for w in res.warnings)


def test_the_boundary_artefact_follows_the_boundary_and_the_row_follows_it_too():
    """The verifier's control, as an assertion: the SAME samples embedded 2 s into a
    longer trace are no longer at an edge, and the row that condemns them at the head
    reads them as fine in the middle."""
    x_short, truth_short = _wander()
    short = B.characterisation_baseline(x_short, FS, polarity=-1)
    x_long, truth_long = _wander(int(FS * 6.0), t0=-2.0)
    long_ = B.characterisation_baseline(x_long, FS, polarity=-1)
    head_err = float(np.mean(np.asarray(short.baseline, float)[:1000] - truth_short[:1000]))
    same_err = float(np.mean(np.asarray(long_.baseline, float)[200_000:201_000]
                             - truth_long[200_000:201_000]))
    assert abs(head_err) > 10.0 * SIGMA and abs(same_err) < 2.0 * SIGMA, (head_err, same_err)
    assert short.health.check("edge_tracking").level == "warn"


# ---------------------------------------------------------------------------
# THE INVARIANT THAT MATTERS MOST: the owner's own recordings must not redden.
# ---------------------------------------------------------------------------
CORPUS = pathlib.Path("/Users/abrahamcampos")
PFX = "500 pM Apo ferritin_1 x PBS_22 nm_"
#: (file, the windows that recording is actually analysed at, expected |offset| bound)
REAL = [
    (PFX + "0006_Process_-600 mV.abf", (2.0, 0.5)),      # the probe's proposal
    (PFX + "0006_Process_-600 mV.abf", None),            # and at the app default
    (PFX + "0005_Process_-700 mV.abf", (1.0, 0.3)),
    (PFX + "0005_Process_-700 mV.abf", None),
    (PFX + "0008_process data_-500 mV.abf", None),
    (PFX + "0004_process _ -1000 mV_ 385-400 s time trace.abf", None),
]


@pytest.mark.corpus
@pytest.mark.parametrize("fname,windows", REAL)
def test_no_real_recording_gains_a_warning(fname, windows):
    """Measured before the row existed, every one of these read zero warnings. The
    -600 mV file at the DEFAULT window is the interesting one: it genuinely opens
    ~23 pA below its open pore for 70 ms (5.4 sigma of offset), and the drift
    comparison holds it green by 8x because that is a STATE, not an untracked edge."""
    from nanoscope.io.base import open_document

    p = CORPUS / fname
    if not p.exists():
        pytest.skip(f"corpus file absent: {fname}")
    doc = open_document(str(p))
    x = np.asarray(doc.raw(2), dtype=np.float64)
    fs = float(doc.meta.sample_rate_hz)
    cfg = (None if windows is None else
           B.BaselineConfig(method="masked_median", rough_s=windows[0], base_s=windows[1]))
    res = B.characterisation_baseline(x, fs, config=cfg)
    row = res.health.check("edge_tracking")
    assert row is not None and row.level == "ok", row.message if row else "row missing"
    assert not any("never TRACKED" in w for w in res.warnings), res.warnings
