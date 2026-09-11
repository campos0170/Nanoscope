"""GA5-01 -- DetectSchmitt reports real intermediate progress, not just 0.0 and 1.0.

THE DEFECT. On a whole-channel detect of the owner's 606 MB ATP recording (the one
file the 15 M-sample ceiling machinery exists for) `DetectSchmitt.run` took ~30 s and
reported progress only at 0.0 ("detecting events") and 1.0 -- so the status-bar
QProgressBar, seeded determinate at 0 by ``_on_analysis_started``, sat at 0% for the
entire run. A user watching zero fill for half a minute cannot distinguish the run
from a stall. [MEASURED before the fix: run 29.7 s / 30.2 s, fractions {0.0, 1.0},
bar values {0, 100}.]

THE FIX, at the seam this module owns. ~86% of the run at that scale is
`characterisation_baseline` [measured: door 20.5 s of run 23.9 s on a 50 M-sample
trace], and `detect_events`'s signature is not ours to grow -- so the baseline module
carries an AMBIENT progress channel (`baseline_progress_scope`, the same thread-local
pattern the cancel token uses), its kernels report beside the cancel polls they
already make, and `DetectSchmitt.run` installs the scope around `detect_events`,
mapping the door's 0..1 onto [0.02, 0.88] of the job.

WHAT IS PINNED HERE. Through a real module run: intermediate fractions strictly
between 0 and 1 exist, the whole sequence is monotone non-decreasing, endpoints are
still 0.0 and 1.0, and every intermediate message names the stage it is in using the
module's own status-bar labels -- "detecting candidates (<baseline stage>)" for
candidate localisation and, on the high-pass split (3ba94ab, 2026-09-04),
"measuring candidates (<baseline stage>)" for the no-high-pass measurement pass plus
that path's two bare stage markers. Plus the channel's own contract: no scope
installed -> reports are no-ops; out-of-order reports are dropped, never delivered
backwards.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from conftest import make_context
from nanoscope.analysis import baseline as B
from nanoscope.analysis.modules import DetectSchmitt

FS = 250_000.0

# THE STAGE LABELS THE MODULE EMITS, read from `DetectSchmitt.run` and
# `_run_highpass_dual_path` (nanoscope/analysis/modules.py), not guessed. Commit
# 3ba94ab (2026-09-04) split the run into a candidate-localisation stage and a
# no-high-pass measurement stage and relabelled the user-facing status-bar text to
# match: the historical one-array path (high-pass off) now says
# "detecting candidates (<baseline stage>)" where it used to say
# "detecting events (<baseline stage>)". Both stages wrap the baseline door's own
# stage text in parentheses -- "measuring polarity (i/n)", "estimating baseline
# (tracking median)", "judging the baseline (pass-1 mask)", "baseline health (...)".
LOCALISE_PREFIX = "detecting candidates ("
MEASURE_PREFIX = "measuring candidates ("
# The high-pass split also emits two bare stage markers of its own between the
# parenthesised ladders: one opening every measurement pass, one when the
# characterisation pass finds high-pass ringing reversed an Auto polarity vote and
# the single localisation retry is about to run.
HIGHPASS_MARKERS = (
    "measuring candidates without high-pass",
    "high-pass ringing reversed Auto polarity; repeating localization",
)


def _names_a_stage(m: str, prefix: str) -> bool:
    """`prefix` + a NON-EMPTY stage label + the closing paren, nothing less."""
    return m.startswith(prefix) and m.endswith(")") and len(m) > len(prefix) + 1


def _trace(n: int = 3_000_000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = rng.normal(0.0, 2.0, n).astype(np.float32) + 100.0
    for s in range(100_000, n - 1_000, 500_000):
        sig[s:s + 300] -= 40.0
    return sig


def test_detect_emits_monotone_intermediate_fractions():
    ctx = make_context(_trace(), fs=FS, params={"min_dwell_s": 5e-4})
    seen: list[tuple[float, str]] = []
    ctx.progress_cb = lambda f, m="": seen.append((float(f), m))

    res = DetectSchmitt().run(ctx)

    assert len(res.payload.df) > 0
    fracs = [f for f, _ in seen]
    # The endpoints the old contract had are still there...
    assert fracs[0] == 0.0 and fracs[-1] == 1.0
    # ...and between them the run now proves it is alive: several real fractions,
    # delivered in order. Stage-based, so the count does not depend on file size.
    mid = [f for f in fracs if 0.0 < f < 1.0]
    assert len(mid) >= 4, fracs
    assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
    # Every intermediate message still says which job stage this is (the bar renders
    # the message as its format string) and names the baseline stage inside it. With
    # high-pass off this is the historical one-array path, so the ONLY label that can
    # appear here is the localisation one -- pinned exactly, not as "any string".
    # [2026-09-11: was `startswith("detecting events (")`; 3ba94ab (2026-09-04)
    # relabelled the candidate stage, see LOCALISE_PREFIX above.]
    for f, m in seen:
        if 0.0 < f < 1.0:
            assert _names_a_stage(m, LOCALISE_PREFIX), m


def test_highpass_split_labels_localisation_and_measurement_stages():
    """3ba94ab (2026-09-04) split the high-pass run into a localisation pass and a
    no-high-pass measurement pass and gave each its own status-bar label. Pin the
    whole ladder: same endpoints and monotone carrier as the one-array path, every
    intermediate message is one of the module's own stage labels, BOTH parenthesised
    stages appear, and the measurement pass is announced by its bare marker
    immediately before its first parenthesised stage. A polarity-mismatch retry
    (which this synthetic trace happens to trigger) may interleave a second
    localisation ladder; the test tolerates it but does not require it."""
    ctx = make_context(
        _trace(1_000_000), fs=FS,
        params={"min_dwell_s": 5e-4, "hp_enabled": True},
    )
    seen: list[tuple[float, str]] = []
    ctx.progress_cb = lambda f, m="": seen.append((float(f), m))

    res = DetectSchmitt().run(ctx)

    assert len(res.payload.df) > 0
    fracs = [f for f, _ in seen]
    assert fracs[0] == 0.0 and fracs[-1] == 1.0
    assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
    mid = [(f, m) for f, m in seen if 0.0 < f < 1.0]
    assert len(mid) >= 4, fracs
    for _, m in mid:
        assert (
            _names_a_stage(m, LOCALISE_PREFIX)
            or _names_a_stage(m, MEASURE_PREFIX)
            or m in HIGHPASS_MARKERS
        ), m
    msgs = [m for _, m in mid]
    assert msgs[0].startswith(LOCALISE_PREFIX), msgs[0]
    measure_idx = [i for i, m in enumerate(msgs) if m.startswith(MEASURE_PREFIX)]
    assert measure_idx, msgs
    assert msgs[measure_idx[0] - 1] == HIGHPASS_MARKERS[0], msgs[: measure_idx[0] + 1]


def test_progress_reports_are_noops_without_a_scope():
    """The `check_cancelled` discipline: a non-worker caller (row rebuild, probe,
    test) pays nothing and no callback is ever invoked."""
    # Nothing to observe directly -- the assertion is that this does not raise and
    # that installing NO scope leaves module state untouched for a later scope.
    B._report_baseline_progress(0.5, "orphan report")
    got: list[float] = []
    with B.baseline_progress_scope(lambda f, m: got.append(f)):
        B._report_baseline_progress(0.25, "a")
    assert got == [0.25]


def test_the_carrier_is_monotone_and_clamped():
    got: list[tuple[float, str]] = []
    with B.baseline_progress_scope(lambda f, m: got.append((f, m))):
        B._report_baseline_progress(0.5, "half")
        B._report_baseline_progress(0.25, "backwards -- must be dropped")
        B._report_baseline_progress(2.0, "clamped to 1")
        B._report_baseline_progress(0.9, "after the top -- dropped")
    assert [f for f, _ in got] == [0.5, 1.0]
    # Nested spans rescale into the parent's window and stay monotone.
    got.clear()
    with B.baseline_progress_scope(lambda f, m: got.append((f, m))):
        with B._progress_span(0.0, 0.5):
            B._report_baseline_progress(1.0, "end of first half")
        with B._progress_span(0.5, 1.0):
            B._report_baseline_progress(0.0, "start of second half -- not backwards")
            B._report_baseline_progress(1.0, "end")
    assert [f for f, _ in got] == [0.5, 1.0]
