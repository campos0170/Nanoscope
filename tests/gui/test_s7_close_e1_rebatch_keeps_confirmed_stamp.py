"""E1, the runner half: a re-batch must not downgrade a stamp a human chose.

The panel half landed in wave 1 of the sweep-7 fix batch -- a ``'batch'``-seeded
:class:`~nanoscope.gui.baseline_panel.BaselinePanel` no longer names its own target
as the recording its windows were confirmed FOR. That alone left E1 open, because
with NO scope in the params `baseline.resolve_param_source` stamps ``'batch'`` on
every target, the measured one included: on the owner's -600 mV / -1000 mV files,
measure + confirm on -600 (2.0 s / 0.5 s, ``'proposed'``), Batch detect, select the
never-measured -1000, Batch detect again -> the -600 result's provenance, this
panel's badge, the Stats advisory and the exported ``.meta.json`` all said "COPIED
FROM THE BATCH FORM ... this recording was never measured" about the one recording
the user HAD measured, with no undo (fix-batch E1).

The keep lives in :meth:`BatchDetectController.start` because that is the only place
that holds the Session AND every target, and it is gated on THE WINDOWS, not on which
recording the batch was launched from: a target whose STORED result carries a
confirmed stamp (``'proposed'`` / ``'user'``) at the batch form's own windows keeps
that stamp; change either window and the keep is gone, because the form no longer
holds what that recording was measured at.

Every claim here is made through the real controller or the real
:class:`MainWindow`; nothing reads a ``connect()`` line.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import baseline as B
from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_worker import BatchDetectController
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]

#: The windows the -600 mV probe proposes on the owner's file, scaled to a 60k
#: synthetic: what matters is that they are NOT `B.DEFAULT_BASELINE`'s, so a keep
#: cannot be confused with the app default passing through.
ROUGH, BASE = 0.2, 0.05


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans=_FRAGMENTS, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
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


def answer_rerun_all(monkeypatch) -> list[str]:
    """Answer ``MainWindow._batch_detect``'s "Re-run all / Only missing" box as a user
    clicking *Re-run all*. Returns the texts of the boxes it answered.

    NOT a timer watching ``activeModalWidget``: that spelling cannot work in this suite
    and fails silently. ``tests/conftest.py`` installs a session-wide modal guard that
    has already replaced ``exec`` with a cancelling stub -- Rejected, ``clickedButton()
    is None`` -- so ``_batch_detect`` falls to its ``else: return`` branch before any
    timer sees a box, no batch starts, and a ``qtbot.waitSignal`` on ``finished`` sits
    there for its full timeout. Measured: the first version of this helper timed out at
    60 s on every MainWindow test here, and on the wave-1 xfail that carried the same
    helper (where ``strict=False`` reported the timeout as an expected failure and hid
    it). Patching ``exec``/``clickedButton`` is what the rest of the suite does
    (``test_analysis_launcher.py``). Any OTHER box keeps the guard's cancelling answer,
    so this helper can only ever say yes to the prompt it is named for."""
    seen: list[str] = []

    def _exec(self):
        seen.append(self.text())
        self._answer = next(
            (b for b in self.buttons() if b.text().replace("&", "") == "Re-run all"), None)
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton",
                        lambda self: getattr(self, "_answer", None))
    return seen


def _batch(controller, qtbot, targets, params):
    with qtbot.waitSignal(controller.finished, timeout=60_000) as blocker:
        assert controller.start(targets, params)
    return blocker.args[0]


def _stamp(outcome, doc_id):
    for res in outcome.results:
        if res.doc_id == doc_id:
            return res.payload.provenance.get("baseline_param_source")
    raise AssertionError(f"{doc_id} is not in the outcome")


# ===========================================================================
# The door E1 was measured through: two batches, the second from the other file
# ===========================================================================
def test_a_second_batch_from_the_never_measured_recording_keeps_the_measured_ones_stamp(
        win, qtbot, tmp_path, monkeypatch):
    """The whole E1 flow through the real window, the shape it was measured in on the
    owner's -600 / -1000 mV recordings: confirm on A, Batch detect, select the
    never-measured B, Batch detect again. A must still say a human chose its windows,
    on its provenance, in the panel badge and in the Stats advisory."""
    a = _add(win, tmp_path, _signal(), "a.dat")
    b = _add(win, tmp_path, _signal(seed=1), "b.dat")
    ap = win._analysis_panel
    bl = ap.baseline_panel
    win._on_selection(a, 0)
    qtbot.wait(20)
    bl.set_params({"baseline_rough_s": ROUGH, "baseline_base_s": BASE,
                   "baseline_param_source": "proposed"})

    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)
    get = lambda d: win._session.results.get(d, 0, "detect.schmitt")  # noqa: E731
    assert get(a).payload.provenance["baseline_param_source"] == "proposed"
    assert get(b).payload.provenance["baseline_param_source"] == "batch"
    rows_a, rows_b = len(get(a).payload), len(get(b).payload)

    # The never-measured recording is selected; its form reads 'batch' and, since the
    # wave-1 panel fix, claims no recording as the home of those windows.
    win._on_selection(b, 0)
    qtbot.wait(20)
    assert bl.param_source() == "batch"
    assert B.BATCH_SCOPE_KEY not in ap.batch_detection_params()

    asked = answer_rerun_all(monkeypatch)
    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)
    assert asked and "already have a Schmitt detection" in asked[0], (
        "the second batch did not go through the re-run prompt at all")

    assert get(b).payload.provenance["baseline_param_source"] == "batch"
    assert get(a).payload.provenance["baseline_param_source"] == "proposed", (
        "the re-batch downgraded the recording the user measured and confirmed")
    # ...and the three surfaces that read the stamp agree with it.
    assert not any("COPIED FROM THE BATCH FORM" in str(x) for x in get(a).warnings)
    assert any("COPIED FROM THE BATCH FORM" in str(x) for x in get(b).warnings)
    win._on_selection(a, 0)
    qtbot.wait(20)
    assert "measured on this recording, and confirmed" in bl.source_label.text()
    assert "COPIED FROM THE BATCH FORM" not in ap._stats.warnings_label.text()
    # Numbers did not move: the keep changes a stamp, never a row.
    assert (len(get(a).payload), len(get(b).payload)) == (rows_a, rows_b)


# ===========================================================================
# The gate is the WINDOWS, not the recording the batch was launched from
# ===========================================================================
def test_a_re_batch_at_other_windows_stamps_batch_away_and_the_typed_source_at_home(
        win, qtbot, tmp_path, monkeypatch):
    """Type different windows into A's form and re-batch from A: the stored
    ``'proposed'`` on A was confirmed at the OLD windows, so nothing is kept from it
    -- A is stamped from the form it is the home of (``'user'``) and B stays
    ``'batch'``. A keep gated on "which recording launched the batch" would have
    handed B's stamp back to A here."""
    a = _add(win, tmp_path, _signal(), "a.dat")
    b = _add(win, tmp_path, _signal(seed=1), "b.dat")
    ap = win._analysis_panel
    bl = ap.baseline_panel
    win._on_selection(a, 0)
    qtbot.wait(20)
    bl.set_params({"baseline_rough_s": ROUGH, "baseline_base_s": BASE,
                   "baseline_param_source": "proposed"})
    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)
    get = lambda d: win._session.results.get(d, 0, "detect.schmitt")  # noqa: E731
    assert get(a).payload.provenance["baseline_param_source"] == "proposed"

    # A hand edit on A's own form -> 'user', at windows nothing was confirmed at.
    bl.rough_s.setValue(0.3)
    bl.base_s.setValue(0.06)
    qtbot.wait(20)
    assert bl.param_source() == "user"
    asked = answer_rerun_all(monkeypatch)
    with qtbot.waitSignal(win._batch_controller.finished, timeout=60_000):
        win._batch_detect()
    qtbot.wait(50)
    assert asked, "the second batch did not go through the re-run prompt at all"

    assert get(a).payload.provenance["baseline_param_source"] == "user"
    assert get(b).payload.provenance["baseline_param_source"] == "batch"
    assert get(a).payload.provenance["baseline_rough_s"] == pytest.approx(0.3)


def test_the_keep_needs_both_windows_to_match_the_form_exactly(
        qapp, qtbot, tmp_path):
    """Controller-level, one knob at a time: same windows -> kept; rough moved ->
    dropped; base moved -> dropped. The stored stamp is only true OF the windows it
    was confirmed at, so a partial match is no match."""
    session = Session()
    a = session.add_document(_doc(tmp_path, _signal(), name="a.dat"))
    b = session.add_document(_doc(tmp_path, _signal(seed=1), name="b.dat"))
    controller = BatchDetectController(session)
    targets = [(a, 0), (b, 0)]
    confirmed = {"min_dwell_s": 1e-4, "baseline_rough_s": ROUGH,
                 "baseline_base_s": BASE, "baseline_param_source": "proposed",
                 B.BATCH_SCOPE_KEY: (a, 0)}

    first = _batch(controller, qtbot, targets, confirmed)
    for res in first.results:
        session.add_result(res)
    assert (_stamp(first, a), _stamp(first, b)) == ("proposed", "batch")

    same = {"min_dwell_s": 1e-4, "baseline_rough_s": ROUGH,
            "baseline_base_s": BASE, "baseline_param_source": "batch"}
    kept = _batch(controller, qtbot, targets, same)
    assert (_stamp(kept, a), _stamp(kept, b)) == ("proposed", "batch")

    for moved in ({**same, "baseline_rough_s": ROUGH * 1.5},
                  {**same, "baseline_base_s": BASE * 1.5}):
        out = _batch(controller, qtbot, targets, moved)
        assert (_stamp(out, a), _stamp(out, b)) == ("batch", "batch"), moved


@pytest.mark.parametrize("stored", ["batch", "restored", "fallback", "default"])
def test_only_a_stamp_a_human_chose_is_kept(qapp, qtbot, tmp_path, stored):
    """``'proposed'`` and ``'user'`` are the two stamps that mean A HUMAN CHOSE THESE
    FOR THIS RECORDING. A stored ``'batch'`` says the opposite -- these windows are
    about some other file -- and keeping it would let one batch's copy immunise a
    recording against every later batch; ``'restored'`` / ``'fallback'`` / the app
    default were never confirmed against this recording at all."""
    session = Session()
    a = session.add_document(_doc(tmp_path, _signal(), name="a.dat"))
    b = session.add_document(_doc(tmp_path, _signal(seed=1), name="b.dat"))
    controller = BatchDetectController(session)
    targets = [(a, 0), (b, 0)]

    seed = {"min_dwell_s": 1e-4, "baseline_rough_s": ROUGH,
            "baseline_base_s": BASE, "baseline_param_source": stored,
            B.BATCH_SCOPE_KEY: (a, 0)}
    first = _batch(controller, qtbot, targets, seed)
    for res in first.results:
        session.add_result(res)
    assert _stamp(first, a) == stored

    again = {"min_dwell_s": 1e-4, "baseline_rough_s": ROUGH,
             "baseline_base_s": BASE, "baseline_param_source": "batch"}
    out = _batch(controller, qtbot, targets, again)
    assert _stamp(out, a) == "batch"
    assert _stamp(out, b) == "batch"


def test_the_keep_is_per_target_and_never_reaches_the_callers_params(
        qapp, qtbot, tmp_path):
    """The scope and the kept stamp are written on ONE target's job. The caller's own
    dict must come back untouched (the panel hands the same dict to the next batch),
    and the kept stamp must not be recorded as a param a "Re-run" would replay onto
    another recording -- `modules._resolve_run_baseline` strips `BATCH_SCOPE_KEY` and
    records the RESOLVED source, so the stored params say ``'proposed'`` on A and
    ``'batch'`` on B."""
    session = Session()
    a = session.add_document(_doc(tmp_path, _signal(), name="a.dat"))
    b = session.add_document(_doc(tmp_path, _signal(seed=1), name="b.dat"))
    controller = BatchDetectController(session)
    targets = [(a, 0), (b, 0)]

    first = _batch(controller, qtbot, targets, {
        "min_dwell_s": 1e-4, "baseline_rough_s": ROUGH, "baseline_base_s": BASE,
        "baseline_param_source": "user", B.BATCH_SCOPE_KEY: (a, 0)})
    for res in first.results:
        session.add_result(res)

    caller = {"min_dwell_s": 1e-4, "baseline_rough_s": ROUGH,
              "baseline_base_s": BASE, "baseline_param_source": "batch"}
    before = dict(caller)
    out = _batch(controller, qtbot, targets, caller)

    assert caller == before, "start() mutated the params dict its caller still holds"
    assert (_stamp(out, a), _stamp(out, b)) == ("user", "batch")
    stored = {res.doc_id: (res.params or {}) for res in out.results}
    assert stored[a]["baseline_param_source"] == "user"
    assert stored[b]["baseline_param_source"] == "batch"
    assert B.BATCH_SCOPE_KEY not in stored[a] and B.BATCH_SCOPE_KEY not in stored[b]


def test_the_keep_never_speaks_for_the_recording_the_form_is_about(
        qapp, qtbot, tmp_path):
    """Confirm on A, batch, then TYPE the same two numbers into A's own form and
    re-batch: A must come back ``'user'``, the gesture actually made.

    The keep reads what a recording was stamped LAST TIME; the form on screen is what
    is happening NOW, and when the form is A's own (the params carry
    `B.BATCH_SCOPE_KEY` naming A) it is the authority. The first cut of this fix had
    no such guard and handed ``'proposed'`` back over a hand-typed ``'user'`` -- the
    panel then read "measured on this recording, and confirmed" and the exported
    ``.meta.json`` said ``'proposed'`` about a confirmation that never happened, which
    is E4 all over again."""
    session = Session()
    a = session.add_document(_doc(tmp_path, _signal(), name="a.dat"))
    b = session.add_document(_doc(tmp_path, _signal(seed=1), name="b.dat"))
    controller = BatchDetectController(session)
    targets = [(a, 0), (b, 0)]
    windows = {"min_dwell_s": 1e-4, "baseline_rough_s": ROUGH, "baseline_base_s": BASE}

    first = _batch(controller, qtbot, targets,
                   {**windows, "baseline_param_source": "proposed",
                    B.BATCH_SCOPE_KEY: (a, 0)})
    for res in first.results:
        session.add_result(res)
    assert _stamp(first, a) == "proposed"

    typed = _batch(controller, qtbot, targets,
                   {**windows, "baseline_param_source": "user",
                    B.BATCH_SCOPE_KEY: (a, 0)})
    assert _stamp(typed, a) == "user", (
        "the keep reported the last confirmation instead of the gesture just made")
    assert _stamp(typed, b) == "batch"


def test_a_target_with_no_stored_result_is_stamped_from_the_form(
        qapp, qtbot, tmp_path):
    """The keep reads a stored result; with none there is nothing to keep and the
    ordinary S7-DET1 rule stands, so a first batch is exactly what it was."""
    session = Session()
    a = session.add_document(_doc(tmp_path, _signal(), name="a.dat"))
    b = session.add_document(_doc(tmp_path, _signal(seed=1), name="b.dat"))
    controller = BatchDetectController(session)

    out = _batch(controller, qtbot, [(a, 0), (b, 0)], {
        "min_dwell_s": 1e-4, "baseline_rough_s": ROUGH, "baseline_base_s": BASE,
        "baseline_param_source": "proposed", B.BATCH_SCOPE_KEY: (a, 0)})
    assert (_stamp(out, a), _stamp(out, b)) == ("proposed", "batch")
