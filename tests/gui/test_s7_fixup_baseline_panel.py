"""Sweep-7 fix-batch repairs to the open-pore baseline panel (E1, E2 / X-SURF-4, E6).

The wave-1 fixes for S7-DET1 and S7-DET3 added a ``'batch'`` source and an "actually
ran at" line to :class:`~nanoscope.gui.baseline_panel.BaselinePanel`. The skeptics
then measured three holes in them, all on the owner's own recordings:

* **E2 / X-SURF-4** -- the line "ran at rough 80 ms / base 16 ms, not the 0.5 s /
  0.1 s above" was baked once at push time; turn the Rough window to 1.0 s and the
  spin read 1.000 s while the sentence two rows under it went on saying 0.5 s. It is
  now re-derived from the live form on every edit, seed, confirmation and reset, and
  both halves carry one unit.
* **E6** -- a ``'batch'``-seeded form followed the user onto a recording nobody had
  batch-run or measured (the -700 mV file), and Detect there measured 1,172 events
  at the copied 2.0 / 0.5 s and stamped them ``'batch'``. Every source is now
  per-recording: a target change puts the form at the app default.
* **E1 (the panel's half)** -- a ``'batch'``-seeded form named its own target as the
  recording its windows were confirmed FOR, so a second Batch detect launched from
  the never-measured -1000 mV file stamped the measured -600 mV file ``'batch'``. It
  now claims none. The other half -- a re-batch keeping a stored confirmed stamp at
  identical windows -- is the batch runner's, and is pinned here as an xfail.
* **E4 (the panel's half)** -- the ``'batch'`` badge said "proposed or confirmed for
  a DIFFERENT recording" over windows the user had TYPED into the other recording's
  form. It is now built from `baseline.BATCH_PROVENANCE_CLAUSE`, the one clause the
  result advisory is built from, so the two surfaces cannot drift apart again.

Every GUI claim is made on a real widget: the bare panel's spins and label, or the
real :class:`MainWindow` selection and detection paths.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.main_window import MainWindow

# The runner half's own module owns the one working way to answer the batch re-run
# prompt in this suite; see `answer_rerun_all` there for why a timer cannot.
from .test_s7_close_e1_rebatch_keeps_confirmed_stamp import answer_rerun_all

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]
_CLAMPED = {"baseline_rough_s": 0.08, "baseline_base_s": 0.016,
            "baseline_fallback": B.FALLBACK_CLAMPED}


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _proposal(rough=2.0, base=0.5, source="proposed"):
    probe = P.BaselineProbe(
        fs=FS, n=3_000_000, polarity=-1, sigma_pA=4.60, noise_pA=4.44,
        longest_state_s=0.496, n_states=489, mask_fraction=0.0931, gating=False,
        gating_score=0.51, level_dwell_s=None, probe_rough_s=5.0, usable=True,
        elapsed_s=0.23)
    cfg = B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)
    return P.BaselineProposal(
        config=cfg, param_source=source, probe=probe,
        reason="Longest state 496 ms (489 states, 9.3% of the trace) -> rough 2 s.")


@pytest.fixture
def panel(qapp, qtbot):
    p = BaselinePanel()
    qtbot.addWidget(p)
    p.set_recording(FS, ("doc-A", 0))
    return p


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._batch_controller.abort()


def _add(win, tmp_path, signal, name):
    doc_id = win._session.add_document(_doc(tmp_path, signal, name=name))
    win._create_default_layers(win._session.document(doc_id))
    return doc_id


def _install_stamped(win, doc_id, signal, *, rough, base, source) -> EventSet:
    """A stored Schmitt result whose params carry the baseline block a real batch run
    stores (``rb.params['baseline_param_source'] == 'batch'`` in the wave-1 test)."""
    cfg = B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base)
    df, prov = detect_events(signal, FS, baseline_config=cfg, polarity=-1,
                             min_dwell_s=1e-4, baseline_param_source=source)
    es = EventSet(df, prov)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4, "baseline_rough_s": rough, "baseline_base_s": base,
                "baseline_param_source": source},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=es, scalars={"n_events": float(len(df))},
        baseline_fingerprint=prov.get("baseline_fingerprint")))
    return es


# ===========================================================================
# E2 / X-SURF-4 -- the "actually ran at ... not the X above" line follows the form
# ===========================================================================
def test_the_run_windows_line_quotes_the_form_as_it_is_now(panel):
    lab = panel.run_windows_label
    panel.set_run_windows(_CLAMPED)
    assert lab.isVisibleTo(panel)
    assert "rough 0.08 s / base 0.016 s" in lab.text()
    assert "not the 0.5 s / 0.1 s above" in lab.text()
    assert "ms" not in lab.text(), "one unit for both halves of one line"

    # The spins themselves -- the gesture the skeptic measured.
    panel.rough_s.setValue(1.0)
    panel.base_s.setValue(0.25)
    assert "not the 1 s / 0.25 s above" in lab.text(), lab.text()
    assert "0.5 s / 0.1 s" not in lab.text()
    assert "CLAMPED" in lab.text()

    # Back to the asked-for windows: the original contrast returns.
    panel.rough_s.setValue(0.5)
    panel.base_s.setValue(0.1)
    assert "not the 0.5 s / 0.1 s above" in lab.text()

    # Typed to the clamped widths: still on screen (the depths are still not
    # comparable), but "ran at 0.08 s, not the 0.08 s above" is not a sentence.
    panel.rough_s.setValue(0.08)
    panel.base_s.setValue(0.016)
    assert lab.isVisibleTo(panel)
    assert "not the" not in lab.text() and "CLAMPED" in lab.text()

    # Back to default: the same writer runs, the contrast is back.
    panel.reset_button.click()
    assert "not the 0.5 s / 0.1 s above" in lab.text()


def test_the_run_windows_line_follows_a_seed_a_confirmation_and_a_plain_edit(panel):
    lab = panel.run_windows_label
    panel.set_run_windows(_CLAMPED)
    # A seed moves the form with `_on_edited` muted; the line must still follow.
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "restored"})
    assert "not the 2 s / 0.5 s above" in lab.text(), lab.text()
    # ...and so does the confirmation click.
    panel.set_proposal(_proposal(rough=1.0, base=0.2))
    assert panel.apply_proposal()
    assert "not the 1 s / 0.2 s above" in lab.text(), lab.text()

    # A run at the form's own windows, not clamped: nothing to say...
    panel.set_run_windows({"baseline_rough_s": 1.0, "baseline_base_s": 0.2})
    assert not lab.isVisibleTo(panel)
    # ...until the form leaves them, and then the line names what the table was
    # measured at -- the documented rule, "shown only when it disagrees with the
    # form", read in both directions.
    panel.rough_s.setValue(1.5)
    assert lab.isVisibleTo(panel)
    assert "ran at rough 1 s / base 0.2 s, not the 1.5 s / 0.2 s above" in lab.text()
    assert "CLAMPED" not in lab.text()
    panel.rough_s.setValue(1.0)
    assert not lab.isVisibleTo(panel)

    panel.set_run_windows(None)
    panel.rough_s.setValue(3.0)
    assert not lab.isVisibleTo(panel) and lab.text() == ""


def test_the_wired_line_follows_the_knob_above_it_on_a_clamped_detection(
        qapp, win, qtbot, tmp_path):
    """Through the real window: a recording shorter than the tracking window, the
    real Detect, then the real spin."""
    signal = _signal([(1_000, 1_200)], n=4_000)          # 0.08 s at 50 kHz
    doc_id = _add(win, tmp_path, signal, "short.dat")
    win._on_selection(doc_id, 0)
    qtbot.wait(20)
    ap = win._analysis_panel
    bl = ap.baseline_panel
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        ap.trigger_detection()
    qtbot.wait(50)
    res = ap.current_detection()
    assert res is not None
    assert res.payload.provenance.get("baseline_fallback") == B.FALLBACK_CLAMPED
    lab = bl.run_windows_label
    assert lab.isVisibleTo(bl)
    assert "not the 0.5 s / 0.1 s above" in lab.text(), lab.text()

    bl.rough_s.setValue(1.0)
    qapp.processEvents()
    assert float(bl.rough_s.value()) == 1.0
    assert "not the 1 s / 0.1 s above" in lab.text(), lab.text()
    assert "0.5 s / 0.1 s" not in lab.text()


def test_the_run_windows_line_sits_under_the_button_row_so_a_click_there_never_moves_the_button(
        qapp, qtbot):
    """A line that follows the form appears and vanishes on this row's own clicks.
    Measured in the real window at the dock's 460 px before this placement: "Use
    these values" over a detection run at the defaults moved the pressed button
    163 px with the line above the row against 142 px without it -- the 21 px is the
    line, and the 142 is the staleness banner above the panel, which is not this
    widget's. Hosted here the way the page hosts it (a column ending in a stretch)."""
    host = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(host)
    panel = BaselinePanel()
    column.addWidget(panel)
    column.addStretch(1)
    qtbot.addWidget(host)
    host.resize(460, 900)
    host.show()
    qtbot.wait(50)
    panel.set_recording(FS, ("doc-A", 0))
    panel.set_gate(True)
    y = lambda w: w.mapTo(host, QtCore.QPoint(0, 0)).y()  # noqa: E731

    # Below the row, whatever it says.
    panel.set_run_windows(_CLAMPED)
    qtbot.wait(50)
    assert panel.run_windows_label.isVisibleTo(panel)
    assert y(panel.run_windows_label) > y(panel.apply_button)

    # Apply over a detection run at the defaults: the line APPEARS; the row stays.
    panel.set_run_windows({"baseline_rough_s": 0.5, "baseline_base_s": 0.1})
    panel.set_proposal(_proposal(rough=2.0, base=0.5))
    panel.set_health(None, measured_for="proposal")
    qtbot.wait(50)
    assert not panel.run_windows_label.isVisibleTo(panel)
    before = y(panel.apply_button)
    qtbot.mouseClick(panel.apply_button, QtCore.Qt.LeftButton)
    qtbot.wait(50)
    assert panel.run_windows_label.isVisibleTo(panel)
    assert "not the 2 s / 0.5 s above" in panel.run_windows_label.text()
    assert y(panel.apply_button) == before

    # Back to default: the line VANISHES; the row stays.
    before = y(panel.reset_button)
    qtbot.mouseClick(panel.reset_button, QtCore.Qt.LeftButton)
    qtbot.wait(50)
    assert not panel.run_windows_label.isVisibleTo(panel)
    assert y(panel.reset_button) == before


# ===========================================================================
# E6 -- every source is per-recording
# ===========================================================================
@pytest.mark.parametrize("source", ["batch", "restored", "fallback"])
def test_a_seeded_form_does_not_follow_the_user_to_the_next_recording(panel, source):
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": source})
    assert panel.param_source() == source
    assert (panel.rough_s.value(), panel.base_s.value()) == pytest.approx((2.0, 0.5))

    # Re-selecting the same recording is navigation and moves nothing.
    panel.set_recording(FS, ("doc-A", 0))
    assert panel.param_source() == source
    assert (panel.rough_s.value(), panel.base_s.value()) == pytest.approx((2.0, 0.5))

    # Another recording: the form goes to the app default, and says so.
    panel.set_recording(FS, ("doc-B", 0))
    assert panel.param_source() == "default"
    assert panel.rough_s.value() == pytest.approx(B.DEFAULT_BASELINE.rough_s)
    assert panel.base_s.value() == pytest.approx(B.DEFAULT_BASELINE.base_s)
    assert "app default" in panel.source_label.text()
    assert "COPIED" not in panel.source_label.text()


def test_a_batch_stamp_on_one_recording_does_not_reach_the_next_ones_detect(
        win, qtbot, tmp_path):
    """The measured flow, on synthetic recordings: a recording seeded ``'batch'`` from
    its stored result, then a fresh recording with no result; Detect there runs the
    default and stamps it -- never the copied windows, never ``'batch'``."""
    sig_a = _signal(_FRAGMENTS)
    sig_b = _signal(_FRAGMENTS, seed=1)
    a = _add(win, tmp_path, sig_a, "a.dat")
    _install_stamped(win, a, sig_a, rough=0.2, base=0.05, source="batch")
    ap = win._analysis_panel
    bl = ap.baseline_panel

    win._on_selection(a, 0)
    qtbot.wait(20)
    assert bl.param_source() == "batch"
    assert (bl.rough_s.value(), bl.base_s.value()) == pytest.approx((0.2, 0.05))
    assert "COPIED from the batch form" in bl.source_label.text()

    b = _add(win, tmp_path, sig_b, "b.dat")
    win._on_selection(b, 0)
    qtbot.wait(20)
    assert bl.param_source() == "default", "the copied windows followed the user"
    assert (bl.rough_s.value(), bl.base_s.value()) == pytest.approx(
        (B.DEFAULT_BASELINE.rough_s, B.DEFAULT_BASELINE.base_s))

    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        ap.trigger_detection()
    qtbot.wait(50)
    rb = win._session.results.get(b, 0, "detect.schmitt")
    assert rb is not None
    prov = rb.payload.provenance
    assert prov["baseline_param_source"] == "default"
    assert prov["baseline_rough_s"] == pytest.approx(B.DEFAULT_BASELINE.rough_s)
    assert not any("COPIED FROM THE BATCH FORM" in w for w in rb.warnings)
    assert any("DEFAULT" in w for w in rb.warnings), "the never-measured advisory"
    assert bl.param_source() == "default"


# ===========================================================================
# E1 -- a 'batch'-seeded form claims no recording as the home of its windows
# ===========================================================================
def test_a_batch_seeded_form_claims_no_recording_as_the_home_of_its_windows(panel):
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "batch"})
    bp = panel.batch_params()
    assert bp["baseline_param_source"] == "batch"
    assert B.BATCH_SCOPE_KEY not in bp, "the panel's own target is NOT whose these are"
    # ...so the resolver stamps the panel's own target 'batch' as well, never
    # anything a human is credited with.
    assert B.resolve_param_source(bp, ("doc-A", 0), "batch") == "batch"
    assert B.resolve_param_source(bp, ("doc-Z", 0), "batch") == "batch"

    # Control: a source set FOR the target still names it.
    for i, src in enumerate(("proposed", "user", "restored")):
        target = (f"doc-{i}", 0)
        panel.set_recording(FS, target)
        panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                          "baseline_param_source": src})
        assert panel.batch_params()[B.BATCH_SCOPE_KEY] == target


def test_the_batch_badge_is_built_from_the_one_clause_the_advisory_shares(panel):
    """E4, the panel's half. A hand-typed form is stamped ``'batch'`` on the other
    recordings exactly like a confirmed proposal, and the stamp keeps neither -- so
    the badge may not name one, and it must read the same clause the Stats advisory
    reads, from the same constant."""
    panel.set_params({"baseline_rough_s": 1.5, "baseline_base_s": 0.5,
                      "baseline_param_source": "batch"})
    label = panel.source_label.text()
    assert label.startswith("These values: COPIED from the batch form: "), label
    assert B.BATCH_PROVENANCE_CLAUSE in label
    low = label.lower()
    assert not any(w in low for w in ("proposed", "confirmed", "by hand", "restored")), label
    cfg = B.DEFAULT_BASELINE.replace(rough_s=1.5, base_s=0.5)
    assert B.BATCH_PROVENANCE_CLAUSE in B.batch_copied_warning(cfg)


def test_a_second_batch_from_the_never_measured_recording_keeps_the_measured_ones_stamp(
        win, qtbot, tmp_path, monkeypatch):
    """The runner half, LANDED (fix-batch close, E1): the xfail this test carried is
    gone because ``BatchDetectController.start`` now does per target what the panel
    cannot -- when the stored ``detect.schmitt`` result carries ``'proposed'`` /
    ``'user'`` at the batch form's own windows, ``merged[BATCH_SCOPE_KEY] = (doc_id,
    channel)`` and ``merged['baseline_param_source'] = <that stamp>``. Measured on the
    owner's -600 / -1000 mV files: re-batch from the -1000 form left -600
    ``'proposed'`` (panel "measured on this recording, and confirmed", no COPIED
    advisory in Stats, exported .meta.json ``'proposed'``), -1000 stayed ``'batch'``,
    row counts 276 / 98 unchanged; a re-batch at OTHER windows typed on -600 stamped
    ``'user'`` at home and ``'batch'`` elsewhere. The gate and its edges are pinned in
    ``tests/gui/test_s7_close_e1_rebatch_keeps_confirmed_stamp.py``, whose
    ``answer_rerun_all`` this test now borrows: the local timer-on-``activeModalWidget``
    helper it used to carry could never fire (the suite's modal guard has already
    answered the box with its cancelling default), so the second batch never started
    and the 60 s ``waitSignal`` timeout was being reported as the expected xfail."""
    a = _add(win, tmp_path, _signal(_FRAGMENTS), "a.dat")
    b = _add(win, tmp_path, _signal(_FRAGMENTS, seed=1), "b.dat")
    ap = win._analysis_panel
    bl = ap.baseline_panel
    win._on_selection(a, 0)
    qtbot.wait(20)
    bl.set_params({"baseline_rough_s": 0.2, "baseline_base_s": 0.05,
                   "baseline_param_source": "proposed"})
    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)
    get = lambda d: win._session.results.get(d, 0, "detect.schmitt").payload.provenance  # noqa: E731
    assert get(a)["baseline_param_source"] == "proposed"
    assert get(b)["baseline_param_source"] == "batch"

    # The never-measured recording is selected; its form reads 'batch'.
    win._on_selection(b, 0)
    qtbot.wait(20)
    assert bl.param_source() == "batch"
    assert B.BATCH_SCOPE_KEY not in ap.batch_detection_params()

    asked = answer_rerun_all(monkeypatch)
    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)
    assert asked, "the second batch did not go through the re-run prompt at all"
    assert get(b)["baseline_param_source"] == "batch"
    assert get(a)["baseline_param_source"] == "proposed", (
        "the re-batch downgraded the recording the user measured and confirmed")
