"""S4 endurance END-003 — curation must not re-median every open recording.

An I-V point is ``median(calibrated current)`` against the applied voltage. Both are
properties of the SIGNAL, so accepting or rejecting an event cannot move either. The
I-V window nevertheless re-derives its contents from the session on every curation
signal (``MainWindow._refresh_surface_windows``), and ``iv_exclusions`` re-asked the
voltage question ``iv_points`` had just answered — so one checkbox click cost
~230-250 ms of ``np.median`` over the owner's three open recordings (9,074,042
samples in one of them) to arrive at bit-identical numbers.

The fix is a memo in :mod:`nanoscope.analysis.iv` keyed on the CONTENT key the rest
of the app uses — ``(document, channel, gain, offset, target_units, n_samples)`` —
so it is a speed change only. These tests pin BOTH halves:

* the work is not repeated (``iv.channel_median_stats()``), through a real
  :class:`MainWindow` and the production ``set_event_accepted`` gesture; and
* every number and every refusal is unchanged, including after the two edits that
  MUST invalidate it: a new gain, and declaring a channel's units (which a bare
  ``ChannelCalibration.fingerprint()`` does not cover, because it is (gain, offset)
  only, "target_units is a label").
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import iv
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

WHERE = "(S4 endurance END-003: curation cannot move an open-pore median)"


@pytest.fixture(autouse=True)
def _fresh_memo():
    """Every test starts from an empty memo and leaves one behind."""
    iv.forget_channel_medians()
    yield
    iv.forget_channel_medians()


def _doc(tmp_path, *, name: str, mV: float, i_pA: float, n: int = 20_000,
         v_units: str = "mV", with_events: bool = False) -> TraceDocument:
    if with_events:
        # A REAL trace with real events, so the Schmitt detector fills a real table
        # and the curation gestures below act on real rows.
        from nanoscope.analysis.truth import synth_trace
        rec, _ = synth_trace(fs=50_000.0, duration_s=n / 50_000.0, i0_pA=i_pA,
                             noise_pA=2.0, n_events=12, seed=3)
        cur = np.ascontiguousarray(np.asarray(rec.current, dtype=np.float32))
        n = int(cur.size)
    else:
        rng = np.random.default_rng(0)
        cur = np.ascontiguousarray((i_pA + rng.normal(0.0, 1.0, n)).astype(np.float32))
    volt = np.ascontiguousarray(np.full(n, mV, dtype=np.float32))
    infos = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="command", raw_units=v_units, kind=ChannelKind.VOLTAGE),
    )
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=infos)
    return TraceDocument(meta=meta, channels=(cur, volt))


def _wired_window(qtbot, tmp_path) -> MainWindow:
    """A real MainWindow: three recordings open, one detected, and the I-V window up.

    The I-V window reads the SIGNAL, not the event table (`grep -n accepted
    iv_panel.py` finds nothing), so one detection is all a curation gesture needs --
    and three open recordings is the shape of the owner's own I-V workflow.
    """
    win = MainWindow()
    qtbot.addWidget(win)
    first = None
    for mV, i_pA in ((-500.0, -280.0), (-600.0, -350.0), (-700.0, -490.0)):
        doc = _doc(tmp_path, name=f"pore_{int(abs(mV))}.dat", mV=mV, i_pA=i_pA,
                   with_events=first is None)
        doc_id = win._session.add_document(doc)
        win._create_default_layers(doc)
        win._session.set_calibration(doc_id, 1, ChannelCalibration(target_units="mV"),
                                     write_sidecar=False)
        first = first or doc_id
    win._on_selection(first, 0)
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-5)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)
    win._act_iv.trigger()
    assert win._iv_window is not None, WHERE
    return win


# ================================================================================================
# 1. the cost invariant
# ================================================================================================

def test_repeated_iv_refreshes_do_not_re_median_anything(qapp, qtbot, tmp_path):
    """THE END-003 ROOT: two consecutive refreshes returned bit-identical numbers and
    paid for both."""
    win = _wired_window(qtbot, tmp_path)
    w = win._iv_window
    iv.forget_channel_medians()
    w.refresh()
    first = iv.channel_median_stats()["misses"]
    assert first > 0, "nothing was computed at all -- the fixture is wrong"
    before = [(p.doc_id, p.v_volts, p.i_open) for p in w.points()]

    for _ in range(5):
        w.refresh()
    after = iv.channel_median_stats()
    assert after["misses"] == first, (
        f"{after['misses'] - first} channel(s) were re-medianed by a plain refresh {WHERE}")
    assert [(p.doc_id, p.v_volts, p.i_open) for p in w.points()] == before, WHERE


def test_a_curation_gesture_re_medians_nothing(qapp, qtbot, tmp_path):
    """The shipped gesture: the Inspector/Gallery checkbox path, through MainWindow."""
    win = _wired_window(qtbot, tmp_path)
    w = win._iv_window
    ap = win._analysis_panel
    w.refresh()
    baseline = iv.channel_median_stats()["misses"]
    before = [(p.doc_id, p.v_volts, p.i_open) for p in w.points()]
    excl_before = [(e.doc_id, e.reason) for e in w.exclusions()]

    # Count the refreshes, so a test that passes because NOTHING happened fails
    # instead: the point is that the refresh runs and costs nothing, not that the
    # wiring quietly stopped reaching this window.
    refreshes = {"n": 0}
    real_refresh = w.refresh

    def counted() -> None:
        refreshes["n"] += 1
        real_refresh()

    w.refresh = counted                     # type: ignore[method-assign]
    try:
        ids = [int(i) for i in ap.current_events().df["event_id"].tolist()[:3]]
        assert ids, "no events to curate -- the fixture is wrong"
        for eid in ids:
            ap.set_event_accepted(eid, False)
        for eid in ids:
            ap.set_event_accepted(eid, True)
    finally:
        del w.refresh
    assert refreshes["n"] >= len(ids), (
        f"only {refreshes['n']} I-V refresh(es) for {2 * len(ids)} curation clicks -- "
        f"the gesture no longer reaches the window this test is about {WHERE}")

    after = iv.channel_median_stats()
    assert after["misses"] == baseline, (
        f"{after['misses'] - baseline} whole-channel median(s) were recomputed by "
        f"{2 * len(ids)} accept/reject clicks, for numbers curation cannot move {WHERE}")
    # ...and the numbers are the same numbers, which is the whole reason it is safe.
    assert [(p.doc_id, p.v_volts, p.i_open) for p in w.points()] == before, WHERE
    assert [(e.doc_id, e.reason) for e in w.exclusions()] == excl_before, WHERE


def test_iv_points_and_iv_exclusions_share_the_voltage_they_both_read(qapp, qtbot, tmp_path):
    """iv_exclusions used to re-decide, from scratch, what iv_points had just decided."""
    session = Session()
    for mV, i_pA in ((-500.0, -28.0), (-700.0, -49.0)):
        doc = _doc(tmp_path, name=f"p{int(abs(mV))}.dat", mV=mV, i_pA=i_pA)
        session.add_document(doc)
    iv.forget_channel_medians()
    iv.iv_points(session)
    after_points = iv.channel_median_stats()["misses"]
    iv.iv_exclusions(session)
    assert iv.channel_median_stats()["misses"] == after_points, (
        f"iv_exclusions recomputed a median iv_points had already taken {WHERE}")


# ================================================================================================
# 2. the correctness invariant -- what MUST still invalidate it
# ================================================================================================

def test_a_new_gain_moves_the_open_pore_current(qapp, qtbot, tmp_path):
    """The Calibration panel dials the gain live; a memo that survived it would lie."""
    session = Session()
    doc = _doc(tmp_path, name="p500.dat", mV=-500.0, i_pA=-28.0)
    doc_id = session.add_document(doc)
    first = iv.iv_point(session, doc_id)
    session.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0, target_units="pA"),
                            write_sidecar=False)
    second = iv.iv_point(session, doc_id)
    assert second.i_open == pytest.approx(first.i_open * 2.0, rel=1e-5), (
        f"{first.i_open} -> {second.i_open} {WHERE}")


def test_declaring_the_voltage_units_is_seen_immediately(qapp, qtbot, tmp_path):
    """THE KEY THAT A BARE FINGERPRINT WOULD MISS. ChannelCalibration.fingerprint() is
    (gain, offset) only -- target_units is "a label" -- and declaring the units is the
    very fix this module tells the user to make."""
    session = Session()
    doc = _doc(tmp_path, name="mystery.h5", mV=-600.0, i_pA=-35.0, v_units="")
    doc_id = session.add_document(doc)
    assert iv.iv_point(session, doc_id) is None, WHERE
    assert "voltage units" in iv.iv_point_refusal(session, doc_id), WHERE
    assert [e.label for e in iv.iv_exclusions(session)] == ["mystery.h5"], WHERE

    session.set_calibration(doc_id, 1, ChannelCalibration(target_units="mV"),
                            write_sidecar=False)
    point = iv.iv_point(session, doc_id)
    assert point is not None, f"the memo outlived the user's own remedy {WHERE}"
    assert point.v_volts == pytest.approx(-0.6, abs=1e-3), WHERE
    assert iv.iv_exclusions(session) == [], WHERE


def test_a_refusal_is_never_memoised_as_a_value(qapp, qtbot, tmp_path):
    """open_pore_current refuses an all-non-finite channel; that has to keep refusing."""
    n = 512
    cur = np.ascontiguousarray(np.full(n, np.nan, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(n, -500.0, dtype=np.float32))
    infos = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="command", raw_units="mV", kind=ChannelKind.VOLTAGE),
    )
    meta = RecordingMeta(
        path=tmp_path / "nan.dat", format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=infos)
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(cur, volt)))
    for _ in range(2):
        with pytest.raises(ValueError, match="no finite samples"):
            iv.iv_point(session, doc_id)


def test_the_memo_is_bounded(qapp, qtbot, tmp_path):
    """A cache on a doc_id that is unique per load must not grow without a bound."""
    session = Session()
    for i in range(iv._MEDIAN_MEMO_MAX + 40):
        doc = _doc(tmp_path, name=f"d{i}.dat", mV=-500.0, i_pA=-28.0, n=64)
        iv.iv_point(session, session.add_document(doc))
    assert iv.channel_median_stats()["entries"] <= iv._MEDIAN_MEMO_MAX, WHERE
