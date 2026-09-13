"""B22: opening the I-V window on a large, never-before-medianed channel must not
block the GUI thread.

``MainWindow._open_iv -> IVWindow.__init__ -> refresh -> iv.iv_points`` ran
``np.median`` over the FULL current channel (and, when present, the FULL voltage
channel) SYNCHRONOUSLY on the GUI thread, with no busy cursor or status text, the
first time a channel's END-003 memo entry was cold. On the owner's 12.5 M-sample
apo-ferritin ABF this held the GUI thread for ~300 ms on the window's very first
open; a second open/refresh of the SAME channel was already cheap (the memo).

The fix: :func:`nanoscope.analysis.iv.pending_cold_channel_samples` answers,
metadata-only, whether the next gather would have to run a FRESH median over
enough samples to matter (:data:`nanoscope.analysis.iv.IV_ASYNC_SAMPLE_THRESHOLD`).
Below it, :class:`~nanoscope.gui.iv_panel.IVWindow`'s constructor still calls
:meth:`refresh` synchronously, exactly as before -- every OTHER IVWindow test in
this codebase (a few thousand samples) takes that branch, unchanged. Above it, the
constructor shows a busy message immediately and gathers on a background thread
instead, then re-runs the ordinary synchronous :meth:`refresh` once the memo is
warm -- which is what makes the numbers it eventually shows bit-identical to a
synchronous gather's, just not paid for on the GUI thread's first paint.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import iv
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.main_window import MainWindow

WHERE = "(B22: IVWindow's first gather on a large channel must not block the GUI thread)"

#: Large enough that the two recordings below (one channel each) sum past a
#: generous "this would be worth deferring" line on any machine, WITHOUT itself
#: referencing ``iv.IV_ASYNC_SAMPLE_THRESHOLD`` -- the point of this test is the
#: constructor's OBSERVABLE behaviour (does ``points()`` come back already
#: populated?), not a number this file would otherwise be pinning circularly.
_N = 700_000


def _big_current_doc(tmp_path, *, i_open_pA: float, nominal_mV: float, name: str) -> TraceDocument:
    """A large, noise-free current-only recording: ``median`` is exact and cheap to
    check, but the ARRAY is big enough that a synchronous gather is the thing under
    test, not an approximation of one."""
    arr = np.ascontiguousarray(np.full(_N, i_open_pA, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=_N,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _ohmic_session(tmp_path) -> Session:
    """2 large current-only docs on I[pA] = 100 nS * V[mV], at -200 / -1000 mV."""
    session = Session()
    for mV in (-200.0, -1000.0):
        session.add_document(_big_current_doc(
            tmp_path, i_open_pA=100.0 * mV, nominal_mV=mV, name=f"big_{int(-mV)}.dat"))
    return session


@pytest.fixture(autouse=True)
def _fresh_memo():
    """Every test starts from an empty memo -- the cold-miss condition B22 is about."""
    iv.forget_channel_medians()
    yield
    iv.forget_channel_medians()


# ================================================================================================
# the new gate itself
# ================================================================================================

def test_pending_cold_channel_samples_is_metadata_only(tmp_path):
    """Asking the pending cost must never itself pay for a median (or it would be
    exactly the GUI-thread stall it exists to let a caller avoid)."""
    session = _ohmic_session(tmp_path)
    before = iv.channel_median_stats()["misses"]
    pending = iv.pending_cold_channel_samples(session)
    assert iv.channel_median_stats()["misses"] == before, (
        f"pending_cold_channel_samples computed a median itself {WHERE}")
    assert pending == 2 * _N, WHERE                     # both channels are cold
    assert pending > iv.IV_ASYNC_SAMPLE_THRESHOLD, (
        f"this test's own fixture must exceed the threshold it means to exercise {WHERE}")

    # And once a gather has actually happened, the memo is warm and nothing is left
    # pending -- the same content key iv_points/iv_exclusions themselves read.
    iv.iv_points(session)
    assert iv.pending_cold_channel_samples(session) == 0, WHERE


# ================================================================================================
# the B22 root: IVWindow's constructor
# ================================================================================================

def test_iv_window_construction_does_not_gather_synchronously_on_a_cold_large_channel(
        qapp, qtbot, tmp_path):
    """THE B22 ROOT. Before the fix, ``IVWindow.__init__`` always called ``refresh()``
    inline, so ``points()`` was already populated the instant the constructor
    returned -- for a channel this large, that is the synchronous ``np.median`` this
    bug is about. After the fix, the constructor must return with the gather still
    outstanding (a busy message, not a plot), and the SAME numbers a synchronous
    gather would have produced must show up once it settles."""
    session = _ohmic_session(tmp_path)

    # An oracle computed independently, straight from the (still cold) session --
    # not read off `w` -- so a shortcut that skipped or mis-memoised a median could
    # not pass by accident.
    oracle_points = {p.doc_id: p.i_open for p in iv.iv_points(session)}
    iv.forget_channel_medians()   # ...and back to cold, for the window under test.
    assert len(oracle_points) == 2, f"fixture sanity: expected 2 usable points {WHERE}"

    w = IVWindow(session)
    qtbot.addWidget(w)

    # The constructor must NOT have run the gather inline: on a 700,000-sample
    # channel that is a real, measurable np.median, and this assertion is exactly
    # what distinguishes "deferred" from "already done" regardless of how fast any
    # one machine's numpy happens to be.
    assert w.points() == [], (
        f"IVWindow() already gathered synchronously on a {_N}-sample channel {WHERE}")
    assert "Gathering" in w.details_text() or w.details_text() == "", (
        f"no busy state shown while the background gather is outstanding {WHERE}")

    qtbot.waitUntil(lambda: len(w.points()) == 2, timeout=5_000)

    got_points = {p.doc_id: p.i_open for p in w.points()}
    assert got_points == oracle_points, (
        f"the async gather produced different numbers than a synchronous one {WHERE}")
    fit = w.fit()
    assert fit is not None, WHERE
    assert fit.conductance_nS == pytest.approx(100.0, rel=1e-6), WHERE


def test_deferred_initial_refresh_still_gathers_nothing_up_front(qapp, qtbot, tmp_path):
    """``defer_initial_refresh=True`` (project restore) must still gather nothing on
    construction -- this path was never the one B22 is about, and must stay that
    way regardless of channel size."""
    session = _ohmic_session(tmp_path)
    w = IVWindow(session, defer_initial_refresh=True)
    qtbot.addWidget(w)
    assert w.points() == [], WHERE
    assert iv.channel_median_stats()["misses"] == 0, (
        f"defer_initial_refresh=True gathered anyway {WHERE}")


def test_reset_analytical_workspace_retires_an_in_flight_gather_before_deleting_the_window(
        qtbot, tmp_path):
    """A project reset/switch (MainWindow._reset_analytical_workspace) deletes the IV
    window directly, bypassing the teardown sweep's _stop_holders discovery. Without
    calling IVWindow.abort() first, a gather still in flight is orphaned: its QThread
    keeps running with nothing left to deliver its `done` signal to once the window's
    C++ object is gone -- the exact class of hazard app.py's own docstring documents at
    length for a worker outliving what it reports back to."""
    import shiboken6

    window = MainWindow()
    qtbot.addWidget(window)
    n = 2_000_000
    data = np.random.default_rng(11).normal(-500, 1, n).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "big.dat", format_name="synthetic", n_samples=n,
                         sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
                         channels=(info,), nominal_voltage_mV=-1000.0,
                         voltage_provenance=Provenance.FILENAME)
    doc = TraceDocument(meta=meta, channels=(data,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    iv.forget_channel_medians()
    window._open_iv()
    iv_window = window._iv_window
    assert iv_window._gather_worker is not None, (
        f"fixture sanity: {n} samples must trigger the async gather {WHERE}")
    thread_ref = iv_window._gather_thread

    window._reset_analytical_workspace()

    assert window._iv_window is None
    assert not thread_ref.isRunning(), (
        "the gather's QThread was still running after the window that owned it was "
        f"deleted -- reset_analytical_workspace must abort() before deleteLater() {WHERE}")
    qtbot.waitUntil(lambda: not shiboken6.isValid(iv_window), timeout=2_000)
