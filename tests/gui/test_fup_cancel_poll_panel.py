"""H3b — the Plots window, through a real MainWindow: what it PROMISES and what it SAYS.

Four of the six re-measured defects in the round before this one were in *what the app
tells you*, not what it computes, and every one of them was invisible to a test that
constructs the widget in isolation. So every string these tests assert is read out of
``details_text()`` on a ``PlotsWindow`` that a real :class:`MainWindow` opened through
its real ``Analysis ▸ Plots…`` action, over events a real detection produced, with the
event loop drained.

THE THREE DEFECTS PINNED HERE

1. **A placeholder that resolved to nothing.** ``_CI_PENDING_TEXT`` told the user
   *"95% CONFIDENCE INTERVALS: still computing — they appear when you stop clicking."*
   The deferred interval was then applied to the ``FitResult`` and **never printed in
   the Details pane at all**, in either state — nor was the exponential's, nor the depth
   mixture's, both of which have bootstrapped one on every fit for far longer. Measured
   on the owner's -700 mV run: the settled pane read ``k = 25.45   (n = 984)`` with no
   interval anywhere, while ``fit.ci['k']`` held ``(16.786, 39.976)``.

2. **A promise with no way to be broken.** A bootstrap that RAISED left
   ``meta['ci_pending'] = True`` on the fit for ever, because only ``apply_ci`` ever
   cleared it and the failure path did not call it.

3. **A join that always burned its budget.** Every curation click aborts the in-flight
   advisory batch with a 100 ms bounded join on the GUI thread. Nothing the batch runs
   polled the cancel token, so the join always timed out and detached a still-running,
   GIL-holding worker into the next click.

...and one guard whose condition was not the one its docstring documented: the
``_abort_note_job()`` branch inside ``_settle_notes`` could not execute, because a
dispatched key stayed in ``_pending_notes`` and made the in-flight set a subset of
"still wanted" by construction.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.analysis import stats
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope, is_cancelled
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import worker_thread
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plots_panel import PlotsWindow

FS = 100_000.0
_GAMMA = "gamma (peaked)"
_EXP = "exponential"
_EXP_GAMMA = "exponential + gamma (bimodal)"

#: An abort must be a return, not a wait. The panel joins its worker for
#: ``worker_thread.DEFAULT_JOIN_MS`` (100 ms) and that join is on the GUI thread, so a
#: batch that cannot stop turns every curation click into a fixed 100 ms tax.
MAX_ABORT_MS = 30.0


# ================================================================================================
# a recording whose dwell times an exponential cannot describe
# ================================================================================================

def _plan(n_events: int, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A trace with ``n_events`` blockades: a narrow ~2 ms mode plus a short spike.

    The SHAPE is load-bearing twice over. The cross-family scan is gated on the drawn
    fit being REJECTED by its KS test, so a well-specified fixture would leave the
    expensive path — the one this file is about — untested. And ``exponential + gamma``
    is only expensive on data with two lobes; on a clean exponential sample its
    twelve starts all converge at once.
    """
    rng = np.random.default_rng(seed)
    spike = np.maximum(rng.normal(2.0e-4, 4e-5, n_events // 8), 8e-5)
    mode = np.maximum(rng.normal(2.0e-3, 2.5e-4, n_events - n_events // 8), 5e-4)
    dwells = np.r_[spike, mode]
    rng.shuffle(dwells)
    gap = 3_500
    n = gap * (n_events + 2)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.2, n)
    spans = []
    for i, d in enumerate(dwells):
        s = gap * (i + 1)
        e = s + max(int(round(d * FS)), 4)
        x[s:e] -= 30.0
        spans.append((s, e))
    return np.ascontiguousarray(x, dtype=np.float32), spans


def _doc(tmp_path, n_events: int, seed: int, name: str) -> TraceDocument:
    arr, _ = _plan(n_events, seed)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _wired_plots(window: MainWindow, qtbot, tmp_path, *, n_events: int, seed: int = 5):
    """MainWindow → open a recording → real detection → real ``Analysis ▸ Plots…``."""
    session = window._session
    doc = _doc(tmp_path, n_events, seed, f"apo_synth_{n_events}.dat")
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-5)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        panel.trigger_detection()
    qtbot.wait(30)

    window._sync_plots_action()
    assert window._act_plots.isEnabled(), "the real Plots action never lit up"
    window._act_plots.trigger()
    qtbot.wait(60)
    pp = window._plots_window
    assert isinstance(pp, PlotsWindow)
    pp.set_plot_type("Histogram")
    pp.set_histogram_column("dwell_s")
    qtbot.wait(30)
    return pp


def _drain(pp: PlotsWindow, qtbot, *, timeout_s: float = 60.0) -> None:
    """Settle every deferred note. A silent timeout here is indistinguishable from a
    product regression, so it is never silent."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qtbot.wait(10)
        if (not pp._pending_notes and not pp._note_timer.isActive()
                and not pp._inflight_notes):
            qtbot.wait(10)
            return
    raise AssertionError(
        f"the panel never settled: pending={sorted(k[0] for k in pp._pending_notes)} "
        f"inflight={sorted(k[0] for k in pp._inflight_notes)}")


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    if w._plots_window is not None:
        w._plots_window.close()
    w._analysis_controller.abort()


@pytest.fixture
def plots(window, qtbot, tmp_path):
    return _wired_plots(window, qtbot, tmp_path, n_events=90)


# ================================================================================================
# where is this tree?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert stub with the wrong signature once manufactured four fake failures on
    this project and was believed."""
    import inspect

    from nanoscope.gui import plots_panel

    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(plots_panel)).resolve().is_relative_to(root)
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)


# ================================================================================================
# 1 — the placeholder resolves where it was made
# ================================================================================================

def test_the_pending_interval_promise_resolves_into_numbers_the_user_can_read(plots, qtbot):
    """THE REGRESSION. ``_CI_PENDING_TEXT`` says the intervals "appear when you stop
    clicking". They never appeared: nothing in this window has ever printed a ``ci``.

    Read out of the real Details pane, in both states, with the loop drained between."""
    pp = plots
    pp.set_fit(_GAMMA)
    qtbot.wait(5)
    pending = pp.details_text()
    assert PlotsWindow._CI_PENDING_TEXT in pending, (
        "the fit went up without saying its interval was still coming; this test cannot "
        "prove the promise is kept if the promise was never made")

    _drain(pp, qtbot)
    settled = pp.details_text()
    fit = pp.fit_result()
    assert fit is not None and fit.ci, "the deferred bootstrap never landed on the fit"

    assert PlotsWindow._CI_PENDING_TEXT not in settled, "still promising after it landed"
    assert "95% CI" in settled, (
        "the Details pane resolved 'still computing' into NOTHING AT ALL. The interval "
        "is on the FitResult and is printed nowhere:\n" + settled)
    # ...and it is THIS fit's interval, not a plausible-looking line.
    lo, hi = fit.ci["k"]
    assert f"k [{lo:.4g}, {hi:.4g}]" in settled, (
        f"the printed interval is not the fit's own k = [{lo!r}, {hi!r}]:\n{settled}")


def test_the_interval_is_printed_in_the_unit_its_own_parameter_is_printed_in(plots, qtbot):
    """A fit is not all one unit: ``gamma`` returns a dimensionless shape ``k`` beside a
    ``theta`` in seconds. The estimate line has obeyed that since report Low-28/High-4;
    the interval line must obey it too, or the two disagree on the same page."""
    pp = plots
    pp.set_fit(_GAMMA)
    _drain(pp, qtbot)
    text = pp.details_text()
    fit = pp.fit_result()

    line = next(ln for ln in text.splitlines() if ln.startswith("95% CI"))
    t_lo, t_hi = fit.ci["theta"]
    assert f"theta [{t_lo * 1e3:.4g}, {t_hi * 1e3:.4g}] ms" in line, (
        f"theta's interval is not in the ms the axis and the estimate use: {line}")
    k_lo, k_hi = fit.ci["k"]
    assert f"k [{k_lo:.4g}, {k_hi:.4g}]" in line and f"k [{k_lo:.4g}, {k_hi:.4g}] " not in (
        line + " ").replace(f"k [{k_lo:.4g}, {k_hi:.4g}]   ", "  "), (
        f"the dimensionless shape k was given a unit: {line}")


def test_the_exponentials_own_interval_was_never_printed_either(plots, qtbot):
    """Not a deferral defect at all — ``fit_exponential`` has bootstrapped a CI on every
    fit since long before H3, and this pane showed the estimate alone. An estimate
    printed with no uncertainty beside it reads as a measurement."""
    pp = plots
    pp.set_fit(_EXP)
    _drain(pp, qtbot)
    fit, text = pp.fit_result(), pp.details_text()
    assert fit.ci, "the exponential fit has no interval; this test proves nothing"
    lo, hi = fit.ci["tau_1"]
    assert f"tau_1 [{lo * 1e3:.4g}, {hi * 1e3:.4g}] ms" in text, text


def test_a_family_that_offers_no_interval_stays_silent(plots, qtbot):
    """The third state. ``exponential + gamma`` reports no CI by design (every replicate
    is a 12-start optimisation) and SAYS so in its own warnings, which Details already
    prints. A second, emptier voice on the same subject is not honesty."""
    pp = plots
    pp.set_fit(_EXP_GAMMA)
    _drain(pp, qtbot)
    text = pp.details_text()
    assert pp.fit_result().ci == {}
    assert "95% CI" not in text
    assert PlotsWindow._CI_PENDING_TEXT not in text
    assert PlotsWindow._CI_FAILED_TEXT not in text
    assert "no bootstrap CIs for this family" in text, (
        "the family's own explanation is missing, so 'no CI line' now reads as an "
        "omission rather than a decision:\n" + text)


# ================================================================================================
# 2 — a promise that can be broken
# ================================================================================================

def test_a_bootstrap_that_raises_stops_promising_an_interval(plots, qtbot, monkeypatch):
    """``meta['ci_pending'] = True`` means "an interval is coming". Only ``apply_ci``
    ever cleared it, so a bootstrap that raised left that standing for the life of the
    FitResult — into ``fit_to_json``, into a saved project, into any surface that reads
    the flag rather than this window's text."""
    real = stats.deferred_ci

    def _boom():
        raise RuntimeError("planted bootstrap failure")

    @contextlib.contextmanager
    def exploding():
        with real() as jobs:
            yield jobs
        if jobs:                      # the panel reads jobs[0] AFTER the with-block
            jobs[:] = [_boom]

    monkeypatch.setattr(stats, "deferred_ci", exploding)

    pp = plots
    pp.set_fit(_GAMMA)
    _drain(pp, qtbot)

    text = pp.details_text()
    assert PlotsWindow._CI_FAILED_TEXT in text, text
    fit = pp.fit_result()
    assert fit.ci == {}
    assert "ci_pending" not in fit.meta, (
        "the fit still claims an interval is on its way, for ever: the bootstrap that "
        "was going to produce it raised and nothing discharged the promise")
    assert fit.meta.get("ci_failed"), "and it does not record that one was attempted"
    assert "ci_pending" not in (stats.fit_to_json(fit).get("meta") or {})


# ================================================================================================
# 3 — the abort is a return, not a wait
# ================================================================================================

def test_aborting_an_in_flight_batch_does_not_burn_the_join(window, qtbot, tmp_path):
    """THE LATENCY REGRESSION, at its mechanism. ``compare_dwell_models`` and
    ``_compute_better_k`` polled nothing, so ``retire_worker_thread``'s 100 ms bounded
    join — on the GUI thread, on every curation click — always timed out and DETACHED a
    still-running worker into the next click.

    Two assertions, and the second is the one that cannot be argued with: the join must
    return early, and nothing may be left parked.

    THE PRECONDITIONS ARE THE TEST. A first draft asserted the abort budget straight
    after ``_settle_notes()`` and passed against the defect, because ``QThread.quit()``
    is honoured before ``exec()`` is even reached -- it was timing a thread that had not
    started working yet. So this one measures how long the batch really takes, refuses
    to run on a fixture too small to matter, and does not abort until a real job has
    told it that it is inside the product's own arithmetic.
    """
    pp = _wired_plots(window, qtbot, tmp_path, n_events=220, seed=9)
    pp.set_fit(_EXP_GAMMA)          # the most expensive ladder the worker runs
    _drain(pp, qtbot)

    def _curate(row: int) -> None:
        """What a curation click reaches: new rows, so every note key is new."""
        df = window._analysis_panel.current_events_df().copy()
        df.loc[df.index[row], "accepted"] = not bool(df.loc[df.index[row], "accepted"])
        pp.set_data({"apo_synth": df})

    # 1. how long IS a batch here? Without this the budget below proves nothing. (The
    # floor on this number moved: it used to be asserted right here, against a flat
    # 4 * DEFAULT_JOIN_MS = 400 ms -- a proxy for the fact that actually matters, how
    # much work REMAINS at the moment of the abort. On the owner's machine the batch
    # measures 390-460 ms, so the proxy failed 2-in-3 on wall-clock luck with the poll
    # working perfectly. The direct statement below asks only for what the test needs
    # and has ~2x headroom. The measurement stays here because it must time an
    # undisturbed batch, not the one being aborted.)
    _curate(0)
    t0 = time.perf_counter()
    pp._settle_notes()
    assert pp._inflight_notes, "no batch was dispatched"
    _drain(pp, qtbot)
    batch_ms = (time.perf_counter() - t0) * 1000.0

    # 2. dispatch another, and wait until the product's own job is actually running.
    _curate(1)
    started = threading.Event()

    def _watch(fn):
        def go():
            started.set()
            return fn()
        return go

    assert pp._pending_notes, "the curation click deferred nothing"
    pp._pending_notes = {k: _watch(v) for k, v in pp._pending_notes.items()}
    pp._settle_notes()
    assert started.wait(30.0), "no job ever began"
    # From here the batch's own arithmetic is running. Progress is clocked from THIS
    # instant, not from the dispatch: thread start-up latency (measured up to ~150 ms)
    # happens before ``started`` fires and no work happens during it, so counting it
    # as progress would spend the margin below on nothing.
    t_work0 = time.perf_counter()
    time.sleep(0.03)                       # ...and is now well inside it

    worker_thread.drain_detached()
    parked_before = worker_thread.detached_count()
    t0 = time.perf_counter()
    pp._abort_note_job()
    dt = (time.perf_counter() - t0) * 1000.0
    worked_ms = (t0 - t_work0) * 1000.0

    # The fixture-size check, stated as the guarantee it exists to give: when the abort
    # landed, at least TWO full joins of batch work still lay ahead. ``batch_ms`` from
    # step 1 stands in for this batch's length -- same rows +-1, same scans (it also
    # carries a few tens of ms of dispatch and drain overhead, small beside the band).
    # One join of remaining work is the minimum for the test to mean anything: a worker
    # that never polls could otherwise just finish inside the join and pass both asserts
    # below by luck. The second join is the margin, so a faster machine or scipy shrinks
    # slack (~350 ms of remaining work measured here) before it ever fakes a failure.
    assert batch_ms - worked_ms > 2 * worker_thread.DEFAULT_JOIN_MS, (
        f"only {batch_ms - worked_ms:.0f} ms of the {batch_ms:.0f} ms batch remained "
        f"when the abort landed ({worked_ms:.0f} ms of work in); the "
        f"{worker_thread.DEFAULT_JOIN_MS} ms join could have outlasted that on its own, "
        f"so this fixture cannot tell a polling batch from a lucky one -- grow it")

    assert dt < MAX_ABORT_MS, (
        f"the abort blocked the GUI thread for {dt:.0f} ms of a {batch_ms:.0f} ms batch "
        f"-- the work never observed the cancel token, so the "
        f"{worker_thread.DEFAULT_JOIN_MS} ms join ran out")
    assert worker_thread.detached_count() == parked_before, (
        "the join timed out and a still-running, GIL-holding worker was detached into "
        "the next curation click -- which is the whole mechanism")
    _drain(pp, qtbot)


def test_a_cancelled_component_ladder_is_never_cached_as_no_better_answer(plots, qtbot):
    """The other half of the poll. ``_compute_better_k`` ends in
    ``except Exception: return ""`` and ``OperationCancelled`` IS an ``Exception`` —
    so a stopped ladder would come back as a real answer, be cached, and print
    "no better component count" for a sample it never finished scanning."""
    pp = plots
    pp.set_fit(_EXP)
    _drain(pp, qtbot)
    fit = pp.fit_result()

    from nanoscope.analysis import populations
    sample = np.asarray(
        pp._current_frame()["dwell_s"].to_numpy(dtype=float))
    sample = sample[sample > 0]

    def refit(kk):
        return populations.fit_exponential(sample, t_min=float(sample.min()) * 0.5,
                                           n_components=int(kk))

    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            out = pp._compute_better_k(refit, 1, fit, (1, 2, 3, 4))
            raise AssertionError(
                f"a cancelled ladder returned {out!r} as if it were an answer")


# ================================================================================================
# 4 — the guard whose condition was not the one its docstring documented
# ================================================================================================

def test_the_settle_cancels_a_batch_the_current_draw_no_longer_wants(plots, qtbot):
    """``_settle_notes`` documents: "if anything in flight is still wanted, let the
    batch finish; if none of it is, cancel it". The second branch could not execute.

    ``_pending_notes`` is documented to mean "the current draw wants this and does not
    have it yet", but a dispatched key stayed in it for ever, so ``_inflight_notes`` was
    a SUBSET of "still wanted" by construction and the intersection was never empty.
    The path that reaches the branch is real and ordinary: a batch is running and the
    user changes the bins, the component count or the fit range — no ``set_data``, so
    nothing else aborts anything.
    """
    pp = plots
    pp.set_fit(_GAMMA)
    _drain(pp, qtbot)

    running, release, finished = threading.Event(), threading.Event(), threading.Event()
    saw: dict = {}

    def slow():
        running.set()
        release.wait(30.0)
        saw["cancelled"] = is_cancelled()
        finished.set()
        return "stale"

    key = ("rivals", "a-sample-nobody-is-looking-at", 0.0, None, 1, "histogram")
    pp._deferred_note(key, slow)
    pp._settle_notes()
    assert running.wait(30.0), "the batch never started"
    assert pp._inflight_notes == {key}

    # A redraw that is NOT set_data: the user nudges the bin count.
    pp.set_bins(41)
    qtbot.wait(5)
    assert key not in pp._pending_notes, (
        "the new draw still lists a key it never asked for -- _pending_notes is not "
        "'what the current draw wants', so the guard below cannot be reached")

    pp._settle_notes()
    assert pp._inflight_notes == set(), (
        "the settle left a batch running that nothing on screen wants: its abort branch "
        "is unreachable")
    assert pp._note_worker is None

    release.set()
    assert finished.wait(30.0)
    assert saw["cancelled"] is True, "the superseded batch was never told to stop"
    assert key not in pp._note_cache, "a superseded answer was cached"
    _drain(pp, qtbot)


def test_a_key_the_draw_still_wants_is_re_registered_by_that_same_draw(plots, qtbot):
    """The other side of emptying ``_pending_notes`` on every redraw: a key that IS
    still wanted must come straight back, inside the same call, or the drain loops that
    wait on it (and the user watching the placeholder) would see the panel go quiet
    while a batch is still running.

    NOT ONE TURN OF THE EVENT LOOP between the dispatch and the check. ``set_bins``
    reaches ``_redraw`` synchronously through ``_changed``, and that is deliberate
    here: a ``qtbot.wait`` would let the batch's own ``done`` be delivered, cache the
    answers, and leave nothing pending for reasons that have nothing to do with this
    invariant. (It did, in the first draft, but only when this file ran whole.)
    """
    pp = plots
    pp.set_fit(_GAMMA)
    qtbot.wait(5)
    assert pp._pending_notes, "nothing was deferred; this test proves nothing"
    before = set(pp._pending_notes)
    pp._settle_notes()
    assert pp._inflight_notes == before

    pp.set_bins(37)                      # a synchronous redraw wanting the SAME keys
    assert set(pp._pending_notes) >= before, (
        "an in-flight key the current draw still wants was dropped from _pending_notes")
    assert pp._inflight_notes == before, "the still-wanted batch was aborted"
    _drain(pp, qtbot)


# ================================================================================================
# nothing else moved
# ================================================================================================

def test_the_rest_of_the_details_pane_is_untouched(plots, qtbot):
    """The CI line is an ADDITION. Everything the pane said before must still be there,
    and the estimate it now stands under must be unchanged."""
    pp = plots
    pp.set_fit(_GAMMA)
    _drain(pp, qtbot)
    text = pp.details_text()
    fit = pp.fit_result()
    assert f"BIC = {fit.bic:.6g}   AIC = {fit.aic:.6g}   components = 1" in text
    assert "Histogram of dwell_s" in text
    assert "MODEL COMPARISON" in text, "the cross-family table stopped being printed"
    assert "KS D = " in text
    # the interval sits with the fit's other numbers, immediately under BIC/AIC
    lines = text.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("BIC = "))
    assert lines[i + 1].startswith("95% CI"), lines[i:i + 3]


def test_the_window_still_shuts_down_clean_with_a_batch_running(plots, qtbot):
    worker_thread.drain_detached()
    before = worker_thread.detached_count()
    pp = plots
    pp.set_fit(_EXP_GAMMA)
    pp._settle_notes()
    pp.close()
    assert pp._note_worker is None and pp._note_thread is None
    assert pp._inflight_notes == set() and pp._pending_notes == {}
    deadline = time.time() + 60.0
    while time.time() < deadline and worker_thread.detached_count() > before:
        QtCore.QCoreApplication.processEvents()
        worker_thread.drain_detached()
        time.sleep(0.01)
    assert worker_thread.detached_count() == before
