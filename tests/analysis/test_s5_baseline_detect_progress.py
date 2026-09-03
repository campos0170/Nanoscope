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
still 0.0 and 1.0, and every message still names the job. Plus the channel's own
contract: no scope installed -> reports are no-ops; out-of-order reports are dropped,
never delivered backwards.
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
    # Every intermediate message still says which job this is (the bar renders the
    # message as its format string) and names the stage inside it.
    for f, m in seen:
        if 0.0 < f < 1.0:
            assert m.startswith("detecting events ("), m


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
