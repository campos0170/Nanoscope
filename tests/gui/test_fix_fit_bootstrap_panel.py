"""H3 + M9 -- the Plots window computes NO reported number on the GUI thread.

THE TWO DEFECTS, MEASURED THROUGH THE REAL GUI ON THE OWNER'S OWN RECORDING.

Real MainWindow, real -700 mV apo-ferritin ABF, real probe/detect/merge (n = 984), the
Plots window opened from the real ``Analysis > Plots...`` action, and twelve REAL
checkbox clicks on the event table with the event loop drained after each:

    H3, the click            before -> after        M9, the GUI thread after ONE
    (median of 12)                                  click, then sitting still
    ------------------------------------------      ------------------------------
    none                     44 ->  56 ms           35 ->  34 ms   (control)
    exponential              99 ->  73 ms          383 ->  38 ms
    gamma (peaked)          604 ->  54 ms          707 ->  35 ms
    inverse-gaussian       1372 ->  71 ms         1283 ->  36 ms
    exponential + gamma     258 -> 252 ms         1007 -> 214 ms
    gaussian-mixture         91 ->  67 ms          458 ->  33 ms

(M9's column is the worst gap between ticks of a 10 ms probe timer -- i.e. how long a
paint-time callback was kept waiting -- and ``_settle_notes`` itself went from
368-1266 ms to 0.1 ms in every family.)

The invariant behind both, which these tests assert rather than the implementation:

    **No reported number may be computed synchronously on an interaction path, and no
    timer may reintroduce it.** The point fit is cheap and stays; the interval moves to a
    worker; and the drawn curve is honest about which state it is in.

The honesty clause is not decoration. ``gamma`` and ``inverse-gaussian`` returned
``ci = {}`` until report finding #10, and it read as "this family reports no
uncertainty". A deferral that merely blanked the interval would re-open exactly that.

Reverted by: restoring ``nanoscope/gui/plots_panel.py`` and
``nanoscope/analysis/stats.py`` to their pre-fix state.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading   # noqa: E402
import time        # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import pytest         # noqa: E402
from PySide6 import QtCore, QtWidgets   # noqa: E402

from nanoscope.analysis import stats            # noqa: E402
from nanoscope.analysis.cancel import is_cancelled   # noqa: E402
from nanoscope.core.session import Session      # noqa: E402
from nanoscope.gui import worker_thread         # noqa: E402
from nanoscope.gui.plots_panel import PlotsWindow   # noqa: E402

APO = Path("/Users/abrahamcampos/Desktop/nanoscope-reports/apo-run-2026-08-22")
APO_PROV = {"min_dwell_s": 20e-6, "fs": 50000.0}
APO_TMIN = 20e-6

#: The curation budget this window is held to. A curator works through hundreds of
#: events in a row; anything above this is felt as lag.
MAX_CLICK_MS = 150.0

_GAMMA = "gamma (peaked)"
_INVGAUSS = "inverse-gaussian (first passage)"


def _apo(run: str = "apo_-700mV") -> pd.DataFrame:
    path = APO / run / "events_after_merge.csv"
    if not path.exists():
        pytest.skip(f"apo-ferritin merged event set not available: {path}")
    return pd.read_csv(path)


def _drain(win: PlotsWindow, timeout_s: float = 60.0) -> None:
    """Run the event loop until this window has nothing outstanding.

    ``_pending_notes`` means "the current draw wants this and does not have it yet",
    whether it is waiting for the settle timer or already on the worker -- so this is a
    real quiescence test and not a sleep. A test that returns with work still in flight
    leaves a thread running into the next one.
    """
    app = QtWidgets.QApplication.instance()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if (not win._pending_notes and not win._note_timer.isActive()
                and not win._inflight_notes):
            app.processEvents()
            return
        time.sleep(0.002)
    raise AssertionError(
        f"the panel never settled: pending={sorted(k[0] for k in win._pending_notes)} "
        f"inflight={sorted(k[0] for k in win._inflight_notes)}")


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1100, 800)
    yield w
    w._abort_note_job()


@pytest.fixture
def apo700(win):
    win.set_data({"apo_-700mV": _apo()}, {"apo_-700mV": dict(APO_PROV)})
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    return win


def _curate_ms(win: PlotsWindow, df: pd.DataFrame, n: int = 5, *, start: int = 0) -> float:
    """Median wall time of one accept/reject round trip, the way curation drives it.

    ``start`` is load-bearing: every deferred note is keyed on the accepted-event
    sample's BYTES, so a second pass that re-toggles the same rows to the same values
    leaves the sample identical and every key HITS. Each pass must move rows the previous
    pass did not, or the timing is of cache hits and not of curation.
    """
    app = QtWidgets.QApplication.instance()
    times = []
    for i in range(n):
        row = df.index[start + i]
        df.loc[row, "accepted"] = not bool(df.loc[row, "accepted"])
        t0 = time.perf_counter()
        win.set_data({"apo_-700mV": df}, {"apo_-700mV": dict(APO_PROV)})
        app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents, 0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ================================================================================================
# WHICH TREE IS THIS?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert stub with the wrong signature once manufactured four fake failures here
    and was believed. Resolve the package, say where it came from, and let the runner
    declare which root it MEANT to test."""
    import inspect
    from nanoscope.gui import plots_panel
    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(plots_panel)).resolve().is_relative_to(root)
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)


# ================================================================================================
# H3 -- the interval is not on the click, and the curve says so
# ================================================================================================

@pytest.mark.parametrize("family,model", [(_GAMMA, "gamma"), (_INVGAUSS, "inverse-gaussian")])
def test_the_drawn_peaked_fit_arrives_without_its_interval_and_admits_it(apo700, family, model):
    """THE REGRESSION. 200 Nelder-Mead replicates used to run inside the redraw.

    The estimate is complete the moment the curve is drawn -- it comes out of the
    maximisation, before any resampling -- and the interval is explicitly pending, in
    ``meta`` AND in the sentence the user reads.
    """
    apo700.set_fit(family)
    fit = apo700.fit_result()
    assert fit is not None and np.isfinite(fit.bic)
    assert fit.meta.get("ci_pending") is True, (
        f"{family}: the bootstrap ran on the draw -- that is the 507/1101 ms click")
    assert fit.ci == {}
    text = apo700.details_text()
    assert "CONFIDENCE INTERVALS" in text and "still computing" in text, (
        "a blank interval that says nothing reads as 'this family reports no "
        "uncertainty' -- which is the defect finding #10 fixed, standing right here:\n"
        + text)


@pytest.mark.parametrize("family,model", [(_GAMMA, "gamma"), (_INVGAUSS, "inverse-gaussian")])
def test_the_interval_lands_and_is_the_one_the_synchronous_fit_would_have_given(
        apo700, family, model):
    """Deferral must not become deletion, and it must not become a DIFFERENT number."""
    apo700.set_fit(family)
    _drain(apo700)
    fit = apo700.fit_result()
    assert "ci_pending" not in fit.meta, "the interval never arrived"
    assert fit.meta["n_boot_ok"] == 200
    assert "still computing" not in apo700.details_text().lower()

    df = _apo()
    sample = df.loc[df["accepted"].astype(bool), "dwell_s"].to_numpy(dtype=np.float64)
    sample = sample[np.isfinite(sample) & (sample > 0)]
    ref = stats.fit_dwell_peaked(sample, model=model, t_min=APO_TMIN)
    assert {k: (repr(a), repr(b)) for k, (a, b) in fit.ci.items()} == \
           {k: (repr(a), repr(b)) for k, (a, b) in ref.ci.items()}, (fit.ci, ref.ci)
    assert {k: repr(v) for k, v in fit.params.items()} == \
           {k: repr(v) for k, v in ref.params.items()}
    assert repr(fit.bic) == repr(ref.bic)


@pytest.mark.parametrize("family", [_GAMMA, _INVGAUSS])
def test_curating_with_a_peaked_fit_on_screen_costs_what_no_fit_costs(apo700, family):
    """The comparison that localises it: the penalty is HAVING the fit, not drawing it.

    Compared against the fit-free click in the SAME process rather than against an
    absolute wall time, because these tests share a machine with five other suites and an
    absolute budget would be measuring the load.
    """
    df = _apo()
    apo700.set_fit("none")
    _drain(apo700)
    bare = _curate_ms(apo700, df, start=0)

    apo700.set_fit(family)
    _drain(apo700)
    fitted = _curate_ms(apo700, df, start=60)   # rows the bare pass never touched

    assert fitted - bare < MAX_CLICK_MS, (
        f"{family} added {fitted - bare:.0f} ms per curation click "
        f"({bare:.0f} -> {fitted:.0f} ms); the bootstrap is back on the redraw path")


# ================================================================================================
# M9 -- and the settle timer must not be where it went instead
# ================================================================================================

def test_the_settle_computes_nothing_on_the_gui_thread(apo700):
    """The 400 ms timer moved the freeze; it did not remove it. Measured on the owner's
    -700 mV run, ONE click then sitting still blocked the GUI thread for 368 ms
    (exponential), 694 ms (gamma), 1266 ms (inverse-gaussian), 991 ms (exp+gamma).

    ``_settle_notes`` must now be a dispatch and nothing else."""
    apo700.set_fit(_INVGAUSS)
    assert apo700._pending_notes, "nothing was deferred; this test proves nothing"
    t0 = time.perf_counter()
    apo700._settle_notes()
    dt = (time.perf_counter() - t0) * 1000.0
    assert dt < 50.0, (
        f"_settle_notes blocked the GUI thread for {dt:.0f} ms -- it is still computing "
        f"there, which is M9")
    assert apo700._inflight_notes, "nothing was dispatched"
    assert apo700._note_worker is not None and apo700._note_thread is not None
    _drain(apo700)


def test_the_deferred_work_actually_runs_off_the_gui_thread(apo700):
    """Not 'later on this thread'. Another thread."""
    seen: dict = {}

    def probe():
        seen["thread"] = threading.get_ident()
        seen["cancellable"] = not is_cancelled()   # an ambient token is installed
        return "done"

    key = ("betterk", "synthetic-offthread", 0.0, None, 1, "x", "y")
    ready, _ = apo700._deferred_note(key, probe)
    assert ready is False
    _drain(apo700)
    assert seen.get("thread") is not None, "the deferred callable never ran"
    assert seen["thread"] != threading.get_ident(), (
        "the deferred work ran on the GUI thread -- a timer is not a worker")
    assert seen["cancellable"] is True
    assert apo700._note_cache[key] == "done"


def test_a_note_that_raises_is_remembered_and_never_re_armed(apo700):
    """An advisory must never break the plot, and its failure must be cached or every
    redraw re-arms the timer for work that cannot succeed."""
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("no")

    key = ("rivals", "synthetic-boom", 0.0, None, 1, "histogram")
    apo700._deferred_note(key, boom)
    _drain(apo700)
    assert calls["n"] == 1
    ready, value = apo700._deferred_note(key, boom)
    assert ready is True and value == []
    assert calls["n"] == 1


def test_a_burst_of_clicks_starts_no_batch_at_all(apo700):
    """Each redraw re-arms the settle timer, so a burst pays for nothing -- and the
    worker is never even spun, which is what keeps it from competing with the next click
    for the interpreter."""
    df = _apo()
    apo700.set_fit(_GAMMA)
    _drain(apo700)
    app = QtWidgets.QApplication.instance()
    started = []
    real = PlotsWindow._settle_notes
    PlotsWindow._settle_notes = lambda self: (started.append(1), real(self))[1]
    try:
        for i in range(8):
            df.loc[df.index[i], "accepted"] = bool(i % 2)
            apo700.set_data({"apo_-700mV": df}, {"apo_-700mV": dict(APO_PROV)})
            end = time.time() + 0.02       # faster than the settle interval
            while time.time() < end:
                app.processEvents()
        assert started == [], "a batch ran mid-burst"
    finally:
        PlotsWindow._settle_notes = real
    _drain(apo700)


# ================================================================================================
# Supersession, cancellation, and the thread that must not be left behind
# ================================================================================================

def test_new_rows_supersede_an_in_flight_batch_at_once(apo700):
    """A batch that was running when the table changed is, by construction, about events
    the user is no longer looking at.

    It must be stopped THEN, not 400 ms later when the settle timer next looks: the jobs
    are ``scipy.optimize.minimize(Nelder-Mead)``, which is Python, so a stale batch does
    not merely waste a core -- it holds the GIL against the next click. Measured with
    ``exponential + gamma`` on screen, a batch left to run alongside the burst took the
    median click from 258 ms to 1014 ms.
    """
    saw = {"cancelled": False}
    running = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow():
        running.set()
        release.wait(30.0)
        saw["runs"] = saw.get("runs", 0) + 1
        saw["cancelled"] = is_cancelled()
        finished.set()
        return "stale"

    key = ("rivals", "synthetic-stale", 0.0, None, 1, "histogram")
    apo700._deferred_note(key, slow)
    apo700._settle_notes()
    assert running.wait(30.0), "the batch never started"
    assert apo700._inflight_notes == {key}

    apo700.set_data({"apo_-700mV": _apo()}, {"apo_-700mV": dict(APO_PROV)})
    assert apo700._inflight_notes == set(), "the stale batch was left running"
    assert apo700._note_worker is None
    release.set()
    assert finished.wait(30.0), "the superseded job never came back"
    _drain(apo700)
    assert saw["cancelled"] is True, "the superseded job was never told to stop"
    assert saw["runs"] == 1, (
        "the superseded job was dispatched AGAIN by a later settle: its key stayed in\n"
        "_pending_notes after the abort, so a fresh worker recomputed an answer nobody\n"
        "wants and cached it into a 16-slot cache")
    assert key not in apo700._note_cache, (
        "a superseded answer was cached; the next redraw would paint an interval that is "
        "not about the events on screen")


def test_the_panel_never_terminates_a_worker_and_leaves_no_thread(qapp, qtbot):
    """No ``terminate()``, no unbounded ``wait()``, and nothing parked afterwards."""
    worker_thread.drain_detached()
    before = worker_thread.detached_count()
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"apo_-700mV": _apo()}, {"apo_-700mV": dict(APO_PROV)})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit(_INVGAUSS)
    w._settle_notes()
    assert w._note_thread is not None
    w.close()                       # mid-bootstrap
    assert w._note_worker is None and w._note_thread is None
    assert w._inflight_notes == set() and w._pending_notes == {}
    deadline = time.time() + 60.0
    while time.time() < deadline and worker_thread.detached_count() > before:
        qapp.processEvents()
        worker_thread.drain_detached()
        time.sleep(0.01)
    assert worker_thread.detached_count() == before, (
        "a detached advisory thread never wound down -- the cancel token is not reaching "
        "the bootstrap loop")
    # ...and the job never held the analysis domain, so it can refuse nothing.
    assert all(j.label != "plots-fit-notes" for j in worker_thread.live_jobs())


def test_the_component_ladder_takes_its_range_instead_of_reading_the_spin():
    """``_compute_better_k`` runs on the worker now, and reading a QWidget from a worker
    thread is undefined behaviour that happens to work until it does not."""
    import ast
    import inspect
    import textwrap
    sig = inspect.signature(PlotsWindow._compute_better_k)
    assert "ks" in sig.parameters, (
        "the BIC ladder still resolves its own component range, which means it reads the "
        "fit-k spin box off a worker thread")
    # The CODE, not the prose: the docstring names `_component_range` on purpose, to say
    # where the range comes from now.
    fn = ast.parse(textwrap.dedent(inspect.getsource(PlotsWindow._compute_better_k))).body[0]
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    reads = [n.attr for stmt in body for n in ast.walk(stmt)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
             and n.value.id == "self"]
    assert "_component_range" not in reads, (
        f"the ladder still reads the spin box off the worker thread: self.{reads}")
    assert not any(a.startswith("_fit_") for a in reads), (
        f"the ladder touches a widget from the worker thread: self.{reads}")
