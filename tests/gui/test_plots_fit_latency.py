"""Curating with a fit on screen must stay interactive (offscreen, pytest-qt).

THE DEFECT THIS PINS, AND WHY IT CAME BACK TWICE.

The Plots window prints two advisory notes beside a fit: a cross-family BIC table naming
a better model, and a "k components fit markedly better" hint. Both refit the data. Both
were, at different times, on the redraw path -- and after C-1 wired the standalone windows
to every signal that changes what they draw, the redraw path IS the curation click path.

    round 1  unconditional scans, bootstrap CIs included   3.40 s per accept/reject
    round 2  point fits only, scans gated on "the fit was  0.4-1.0 s per accept/reject
             REJECTED" -- costed as an exceptional path
    round 3  deferred off the redraw path entirely         measured 44 ms

Round 2 is the interesting failure. Its gate is sound reasoning about the wrong
population: the docstring rests on "a model that passes its KS test costs nothing here",
and on the owner's three apo-ferritin recordings **no model passes** -- the exponential is
rejected on all of them -- so the exceptional path was the only path. Caching could not
save it either, because the cache key is the accepted-event sample's bytes, which change
on every single accept/reject by construction.

So the invariant is not "make the scans cheaper" or "gate them better". It is:

    **Drawing a fit and explaining it are two operations with two latency budgets.
    Nothing advisory may sit between a curation click and the repaint.**

These tests assert the budget, not the implementation, so a future fourth attempt at
making the scans fast enough to inline still fails here if it is ever wrong again.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

#: The budget. A curator works through hundreds of events in a row; anything above this
#: is felt as lag. Measured on the owner's `_0006` recording through a real MainWindow:
#: 305 ms per click with the scans inline, 44 ms deferred, 36 ms with no fit at all.
MAX_CLICK_MS = 150.0


def _rejecting_dwells(n: int = 300, *, seed: int = 3) -> pd.DataFrame:
    """Dwell times an exponential CANNOT describe — a narrow mode plus a short-dwell spike.

    This is the shape of the owner's real data (a ~16-28 ms mode with a hard ceiling and a
    sub-millisecond tail), and it matters that the fixture has it: the round-2 gate only
    opens when the goodness-of-fit test REJECTS, so a well-specified fixture would leave
    the expensive path untested and the test would pass against the defect.
    """
    rng = np.random.default_rng(seed)
    spike = np.abs(rng.normal(4.0e-4, 8e-5, n // 10))
    mode = np.abs(rng.normal(1.9e-2, 2.0e-3, n - n // 10))
    dwell = np.r_[spike, mode]
    return pd.DataFrame({
        "dwell_s": dwell,
        "amplitude": np.abs(rng.normal(90.0, 12.0, n)),
        "frac_depth": np.abs(rng.normal(1.9, 0.3, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _fit_combo(win: PlotsWindow):
    from PySide6 import QtWidgets
    for child in win.findChildren(QtWidgets.QComboBox):
        texts = [child.itemText(i).lower() for i in range(child.count())]
        if any("exponential" in t for t in texts) and any(t == "none" for t in texts):
            return child
    raise AssertionError("no fit combo found")


def _select(combo, needle: str) -> None:
    for i in range(combo.count()):
        if needle.lower() in combo.itemText(i).lower():
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"no fit option matching {needle!r}")


def _curate_ms(win: PlotsWindow, df: pd.DataFrame, qtbot, n: int = 5, *, start: int = 0) -> float:
    """Median wall time of one accept/reject round trip, the way curation drives it.

    ``start`` offsets which rows are toggled, and it is load-bearing: the note cache is
    keyed on the accepted-event sample's bytes, so a second pass that re-toggles the SAME
    rows to the SAME values leaves the sample identical and every key HITS. A first draft
    of :func:`test_a_fit_costs_little_more_than_no_fit` did exactly that and passed
    against the defect — it was timing cache hits, not curation. Each pass must move rows
    the previous pass did not.
    """
    times = []
    for i in range(n):
        row = df.index[start + i]
        df.loc[row, "accepted"] = not bool(df.loc[row, "accepted"])
        t0 = time.perf_counter()
        win.set_data({"Detected events": df})     # what a curation signal triggers
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _settle(win: PlotsWindow, qtbot, *, timeout_s: float = 30.0) -> None:
    """Pump the loop until every deferred note has landed, or fail loudly.

    This replaces a fixed ``qtbot.wait(_NOTE_SETTLE_MS * 3)``. The scans run on a
    worker whose wall time is the machine's business, and under CPU load they overran
    that fixed wait -- the test then read the pane mid-flight and reported "the note
    never arrived" about a note that was legitimately still on its way. Polling keeps
    the assertion meaningful: a note that truly never lands still fails here, at the
    deadline, naming what was left pending instead of blaming the clock.
    """
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        qtbot.wait(10)
        if (not win._pending_notes and not win._note_timer.isActive()
                and not win._inflight_notes):
            qtbot.wait(10)
            return
    raise AssertionError(
        f"the deferred notes never settled: "
        f"pending={sorted(k[0] for k in win._pending_notes)} "
        f"inflight={sorted(k[0] for k in win._inflight_notes)}")


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1100, 800)
    return w


# ================================================================================================
# the budget
# ================================================================================================

def test_curating_with_a_rejected_fit_on_screen_stays_interactive(win, qtbot):
    """THE REGRESSION. A fit the data rejects is the expensive case, and the common one."""
    df = _rejecting_dwells()
    win.set_data({"Detected events": df})
    combo = _fit_combo(win)
    _select(combo, "exponential")
    qtbot.wait(50)

    median = _curate_ms(win, df, qtbot)
    assert median < MAX_CLICK_MS, (
        f"a curation click with a fit on screen took {median:.0f} ms "
        f"(budget {MAX_CLICK_MS:.0f} ms) — an advisory note is back on the redraw path")


def test_a_fit_costs_little_more_than_no_fit(win, qtbot):
    """The comparison that localises it: the penalty is HAVING a fit, not drawing one."""
    df = _rejecting_dwells()
    win.set_data({"Detected events": df})
    combo = _fit_combo(win)

    _select(combo, "none")
    qtbot.wait(50)
    bare = _curate_ms(win, df, qtbot, start=0)

    _select(combo, "exponential")
    qtbot.wait(50)
    fitted = _curate_ms(win, df, qtbot, start=50)   # rows the bare pass never touched

    assert fitted - bare < MAX_CLICK_MS, (
        f"having a fit on screen added {fitted - bare:.0f} ms per click "
        f"({bare:.0f} -> {fitted:.0f} ms)")


def test_the_expensive_notes_are_deferred_not_deleted(win, qtbot):
    """The other half: deferral must not become silence.

    A fix that made curation fast by dropping the advice would pass the budget tests and
    lose the thing the advice exists for — telling the user the drawn curve is wrong.
    """
    df = _rejecting_dwells()
    win.set_data({"Detected events": df})
    _select(_fit_combo(win), "exponential")
    qtbot.wait(50)
    win.set_data({"Detected events": df})

    during = win.details_text()
    assert "still computing" in during.lower(), (
        "the placeholder must say the comparison is pending — an empty line reads as "
        "'no better model exists', the opposite of the truth here")

    _settle(win, qtbot)
    settled = win.details_text()
    assert "still computing" not in settled.lower(), "the note never arrived after settling"
    assert "BIC" in settled, "the settled note should carry the cross-family comparison"


def test_a_burst_of_clicks_computes_nothing(win, qtbot):
    """Each click re-arms the timer, so a burst pays for no scan at all."""
    df = _rejecting_dwells()
    win.set_data({"Detected events": df})
    _select(_fit_combo(win), "exponential")
    qtbot.wait(50)

    calls = {"n": 0}
    real = PlotsWindow._settle_notes

    def counting(self):
        calls["n"] += 1
        return real(self)

    PlotsWindow._settle_notes = counting
    try:
        for i in range(8):
            df.loc[df.index[i], "accepted"] = bool(i % 2)
            win.set_data({"Detected events": df})
            qtbot.wait(20)          # faster than the settle interval
        assert calls["n"] == 0, "the scans ran mid-burst"
        qtbot.wait(int(PlotsWindow._NOTE_SETTLE_MS * 3))
        assert calls["n"] >= 1, "the scans never ran after the burst ended"
    finally:
        PlotsWindow._settle_notes = real


def test_a_failing_note_does_not_re_arm_the_timer_forever(win, qtbot):
    """A scan that raises must cache its failure, or the panel never settles."""
    df = _rejecting_dwells()
    win.set_data({"Detected events": df})
    _select(_fit_combo(win), "exponential")
    qtbot.wait(50)

    boom = {"n": 0}

    def raiser():
        boom["n"] += 1
        raise RuntimeError("no")

    key = ("rivals", "synthetic", 0.0, None, 1, "histogram")
    win._deferred_note(key, raiser)
    _settle(win, qtbot)
    assert boom["n"] == 1
    ready, _ = win._deferred_note(key, raiser)
    assert ready, "the failure was not remembered; the timer would re-arm forever"
    assert boom["n"] == 1
