"""H4b — the three subpopulation messages, read off a WIRED MainWindow.

Every defect this file covers is in what the app TELLS you, not in what it
computes, and every one of them is invisible to a test that builds
``SubpopReportWindow`` directly and asserts an internal call. So each test here
opens a real :class:`~nanoscope.gui.main_window.MainWindow` on a real recording,
runs the real baseline probe, the real detector and the real
``Analysis > Event subpopulations (GMM)…`` action off the real controller thread,
drains the event loop, and asserts the string in the caveat strip the reader is
looking at.

THE TWO RECORDINGS ARE THE DEFECTS' OWN SHAPES, not a convenient fixture:

* ``back`` — 90 % of the events land in the SECOND half, and their depths are
  drawn from one continuous distribution rather than from two families. The
  first half of that is defect 1 (``early_frac`` is a FIRST-half fraction, and
  the caveat printed it in front of the word "second"); the second half is
  defect 3 (one cloud a mixture subdivides, where the old rival comparison
  fired and so did it on the positive control).
* ``front`` — 90 % land in the FIRST half, in two depth families drawn with the
  SAME time distribution, so every class is shaped exactly like the recording.
  That is defect 2: the caveat told the reader to compare each class against
  0.90 and then condemned all of them for not being near 0.50.

WHAT IS DELIBERATELY NOT ASSERTED HERE: the status bar's own sentence. It is
built in ``gui/main_window.py``, which this pass does not own. It is READ below
and printed, as evidence, but its wording is another agent's to fix.

The mixture search grid is shrunk through the app's own ``SubpopConfig`` so two
wired runs fit in a test suite; nothing about the messages under test depends on
it, and every string asserted below comes out of the real report object the real
window was handed.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -----------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import functools
import pathlib
import re

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N_SAMPLES = 1_100_000        # 22 s
N_EVENTS = 700


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees, so a failure below cannot be an import that didn't happen."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    from nanoscope.gui import subpop_window as SW
    for mod in (SG, SR, SW):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[H4b] nanoscope resolved from {root}")


# ---------------------------------------------------------------------------
# the two recordings
# ---------------------------------------------------------------------------


def _trace(case: str, seed: int = 3):
    """A synthetic current trace with a deliberately lopsided event supply.

    ``early_share`` of the events land in the FIRST half of the recording. In
    ``front`` the depths come from two well-separated families, both drawn with
    that same time distribution, so each class is shaped like the recording. In
    ``back`` they come from ONE broad distribution over a noisy baseline — the
    apo recordings' shape, a cloud with no separated partition at any K, which
    the mixture nevertheless subdivides.
    """
    rng = np.random.default_rng(seed)
    noise = 7.0 if case == "back" else 1.5
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, noise, N_SAMPLES)
    early_share = 0.10 if case == "back" else 0.90
    n_early = int(round(N_EVENTS * early_share))
    half = N_SAMPLES // 2
    starts = np.concatenate([
        np.linspace(4_000, half - 6_000, n_early),
        np.linspace(half + 4_000, N_SAMPLES - 8_000, N_EVENTS - n_early),
    ]).astype(int)
    starts = starts + rng.integers(-300, 300, starts.size)
    for i, s in enumerate(starts):
        if case == "back":
            depth = float(rng.normal(45.0, 14.0))
            dwell = int(np.exp(rng.normal(4.4, 0.35)))
        else:
            deep = bool(i % 2)
            depth = (58.0 if deep else 24.0) + rng.normal(0.0, 2.0)
            dwell = int(rng.integers(95, 145) if deep else rng.integers(25, 55))
        s = int(max(500, s))
        x[s:s + dwell] -= depth
    return np.ascontiguousarray(x, dtype=np.float32)


def _document(tmp_dir: pathlib.Path, case: str) -> TraceDocument:
    arr = _trace(case)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_dir / f"synthetic_{case}.csv", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


# ---------------------------------------------------------------------------
# one wired run per recording, shared by the tests that read it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """A PRIVATE QSettings namespace — the suite must never touch the real store."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("H4bSubpopMessagesPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


class _Run:
    """Everything a reader can see after one wired subpopulation run."""

    def __init__(self, win, report):
        self.win = win
        self.report = report
        sw = win._subpop_window
        self.header = sw.header()
        self.caveats = sw.caveats()
        self.text = sw.text()
        self.status = win.statusBar().currentMessage()

    @property
    def q(self):
        return self.report.quantification

    def caveat(self, prefix: str) -> str:
        for w in self.report.warnings:
            if w.startswith(prefix):
                assert w in self.caveats, "a caveat the report raised is not on screen"
                return w
        return ""


def _pump(qapp, ms: int = 60) -> None:
    """Drain the real event loop for ``ms`` — queued cross-thread signals included."""
    import time
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)


def _until(qapp, pred, timeout: float, label: str) -> None:
    import time
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if pred():
            return
        time.sleep(0.002)
    raise AssertionError(f"timed out after {timeout}s waiting for {label}")


def _wired_run(qapp, tmp_path_factory, monkeypatch, case: str) -> _Run:
    tmp = tmp_path_factory.mktemp(f"h4b_{case}")
    # The app's own config object, with the SEARCH shrunk so two wired runs fit in
    # a suite. Nothing under test reads any of these fields.
    monkeypatch.setattr(
        SR, "SubpopConfig",
        functools.partial(SR.SubpopConfig, k_grid=(1, 2, 3, 4), n_splits=2,
                          n_boot=0, with_tsne=False),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    # A modal the harness does not answer blocks the event loop forever, and an
    # error box swallowed in silence looks exactly like a slow run. Record them.
    boxed: list[str] = []

    def _dialog_exec(self, *a, **k):
        """MainWindow reports a failed analysis with `QMessageBox(self).exec()`.

        That is an INSTANCE modal, which the static stubs below cannot see, and in
        an offscreen harness it blocks the event loop until the wait times out —
        a crash in the job looks exactly like a slow run. Recorded and refused.
        """
        try:
            boxed.append(f"{self.windowTitle()!r}: {self.text()!r}")
        except Exception:                      # noqa: BLE001 — not a QMessageBox
            boxed.append(repr(self))
        return QtWidgets.QDialog.Rejected

    for _cls in (QtWidgets.QDialog, QtWidgets.QMessageBox):
        # BOTH: `QMessageBox` carries its own `exec`, so patching the base class
        # alone leaves the failure box modal and the wait looks like a slow run.
        monkeypatch.setattr(_cls, "exec", _dialog_exec, raising=False)
        monkeypatch.setattr(_cls, "exec_", _dialog_exec, raising=False)
    for _nm in ("information", "warning", "critical", "about"):
        monkeypatch.setattr(
            QtWidgets.QMessageBox, _nm,
            staticmethod(lambda *a, _n=_nm, **k: (
                boxed.append(f"{_n}: {a[2] if len(a) > 2 else a}"),
                QtWidgets.QMessageBox.StandardButton.Cancel)[1]))

    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    _pump(qapp, 200)
    doc = _document(tmp, case)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    _pump(qapp, 100)

    ap = win._analysis_panel
    dp, bp = ap.detection_panel, ap.baseline_panel
    dp.show_preprocess_tab()
    _pump(qapp, 60)
    bp.probe_button.click()
    _until(qapp, lambda: (not bp.is_busy()) and bp.proposal() is not None, 300.0, "probe")
    _pump(qapp, 60)
    bp.apply_button.click()
    _pump(qapp, 100)

    dp.show_detection_tab()
    _pump(qapp, 60)
    dp.detect_button.click()
    _until(qapp, lambda: ap.current_detection() is not None, 300.0, "detection")
    _pump(qapp, 200)
    n_acc = int(ap.current_events_df()["accepted"].sum())
    assert n_acc > 400, f"fixture guard: only {n_acc} accepted events"

    assert win._act_subpop.isEnabled()
    win._act_subpop.trigger()
    _pump(qapp, 200)
    assert win._subpop_pending, "the wired action did not start a run"
    _until(qapp,
           lambda: win._subpop_window is not None
           and win._subpop_window.report() is not None,
           600.0, f"the subpopulation report window (modals: {boxed})")
    _pump(qapp, 200)

    assert not boxed, f"the wired run raised a modal: {boxed}"
    run = _Run(win, win._subpop_window.report())
    print(f"\n[H4b/{case}] accepted={n_acc}  header:\n{run.header}")
    print(f"[H4b/{case}] status: {run.status}")
    print(f"[H4b/{case}] caveats:\n{run.caveats}")
    win._analysis_controller.abort()
    win._controller.abort()
    _pump(qapp, 100)
    return run


@pytest.fixture(scope="module")
def back(qapp, tmp_path_factory, monkeypatch_module):
    run = _wired_run(qapp, tmp_path_factory, monkeypatch_module, "back")
    yield run
    run.win.close()
    _pump(qapp, 50)


@pytest.fixture(scope="module")
def front(qapp, tmp_path_factory, monkeypatch_module):
    run = _wired_run(qapp, tmp_path_factory, monkeypatch_module, "front")
    yield run
    run.win.close()
    _pump(qapp, 50)


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ===========================================================================
# DEFECT 1 — the half named in the window is the half the number belongs to
# ===========================================================================


def _pooled(run) -> float:
    """The pooled first-half fraction, from the app's OWN population table."""
    ef = run.q.table["early_frac"].to_numpy(dtype=float)
    n = run.q.table["n"].to_numpy(dtype=float)
    return float((ef * n).sum() / n.sum())


@pytest.mark.parametrize("which", ["back", "front"])
def test_the_stationarity_sentence_the_reader_sees_matches_the_apps_own_table(
        which, request):
    """Defect 1's invariant, checked against the report the window is showing.

    Either the caveat is on screen and the percentage in it belongs to the half
    it names — computed from the app's own ``early_frac`` column, not from the
    fixture — or the caveat is absent and the app's pooled fraction really is
    inside the band, i.e. it was right to stay quiet. There is no third outcome,
    and the shipped code produced one: the first-half number under the word
    "second".

    ON REACHABILITY, measured here and reported upward. ``early_frac`` is built
    from ``time_half``, which splits at each scope's MEDIAN ONSET
    (``analysis/event_features.time_half``), so the pooled fraction over a
    single-recording report is 0.5 by construction however lopsided the
    recording is — the ``back`` recording below delivers nine events in ten into
    its second half and still reports ~0.50. The sentence is therefore
    unreachable from this action today. That is a defect in the M8 caveat, in a
    file outside this pass's fence; the string is fixed where it is written and
    the invariant is pinned here so it stays fixed if the column is ever made to
    mean what the sentence says.
    """
    run = request.getfixturevalue(which)
    pooled = _pooled(run)
    note = run.caveat("THE RECORDING IS NOT STATIONARY")
    lo, hi = SG.STABILITY_BAND
    print(f"\n[H4b/{which}] pooled early_frac = {pooled:.4f}  caveat={'yes' if note else 'no'}")
    if not note:
        assert lo <= pooled <= hi, (
            f"pooled {pooled:.3f} is outside {lo}-{hi} and the reader was told nothing")
        return
    m = re.search(r"(\d+) % of all gated events fall in the (first|second) half", note)
    assert m, note
    pct, half = int(m.group(1)), m.group(2)
    share = pooled if half == "first" else 1.0 - pooled
    assert pct == round(100 * share), (pct, half, pooled)
    assert f"read against this {pooled:.2f}" in note


@pytest.mark.parametrize("which", ["back", "front"])
def test_no_surface_shows_the_first_half_number_under_the_second_half_name(
        which, request):
    """The exact wrong sentence, built from the app's own numbers, on any surface."""
    run = request.getfixturevalue(which)
    pooled = _pooled(run)
    wrong = f"{100 * pooled:.0f} % of all gated events fall in the second half"
    if abs(pooled - 0.5) < 0.005:
        pytest.skip("pooled is 0.50: both halves round to the same number here")
    for surface in (run.caveats, run.text, run.header):
        assert wrong not in surface


# ===========================================================================
# DEFECT 2 — the caveat and the gate one line under it agree
# ===========================================================================


@pytest.mark.parametrize("which", ["back", "front"])
def test_no_class_is_condemned_for_being_shaped_like_its_own_recording(which, request):
    """Defect 2, as an invariant over the app's own report.

    WHAT THE READER SAW on a 90 %-front-loaded recording: "every early_frac must
    be read against this 0.90, not against 0.50" — and, one line under it, all
    four classes condemned for sitting outside 0.25-0.75.

    Checked here against the app's own ``early_frac`` column and the app's own
    printed caveats: no class within the 3:1 odds band of the recording's pooled
    fraction may be named TIME-UNSTABLE. On a stationary report that band IS
    0.25-0.75, so the assertion has teeth on every run rather than only on the
    lopsided one.
    """
    run = request.getfixturevalue(which)
    table = run.q.table
    pooled = _pooled(run)
    lo, hi = SG.stability_band_for(pooled)
    unstable = run.caveat("TIME-UNSTABLE CLASS(ES)")
    named = {int(m) for m in re.findall(r"subpop (\d+) \(n=", unstable)}
    proportional = [int(s) for s, v in zip(table["subpop"], table["early_frac"])
                    if lo <= float(v) <= hi]
    assert proportional, "fixture guard: at least one class must be proportional"
    print(f"\n[H4b/{which}] pooled={pooled:.3f} band={lo:.2f}-{hi:.2f} "
          f"early_frac={list(table['early_frac'])} condemned={sorted(named)}")
    for sp in proportional:
        assert sp not in named, (
            f"subpop {sp} sits inside {lo:.2f}-{hi:.2f} around the pooled "
            f"{pooled:.2f} and is still condemned:\n{unstable}")


def test_the_two_caveats_the_reader_sees_cannot_contradict_each_other(front):
    """The same invariant, taken from the printed sentences and nothing else.

    If the window tells the reader to compare each class against X, then no class
    inside the 3:1 band around X may be named in the class-level caveat. Parsed
    out of the strings on screen, so a future rewording is checked, not trusted.
    """
    note = front.caveat("THE RECORDING IS NOT STATIONARY")
    unstable = front.caveat("TIME-UNSTABLE CLASS(ES)")
    if not note:
        pytest.skip("no non-stationarity claim on screen to contradict")
    printed = float(note.split("read against this ")[1].split(",")[0])
    lo, hi = SG.stability_band_for(printed)
    named = {int(m) for m in re.findall(r"subpop (\d+) \(n=", unstable)}
    for sp, v in zip(front.q.table["subpop"], front.q.table["early_frac"]):
        if lo <= float(v) <= hi:
            assert int(sp) not in named, (printed, sp, v, unstable)


# ===========================================================================
# DEFECT 3 — the rival comparison is gone from every surface
# ===========================================================================


_OLD_CAUSAL = "signature of a density being subdivided rather than of classes being found"


@pytest.mark.parametrize("which", ["back", "front"])
def test_the_old_cross_k_verdict_is_on_no_surface_of_the_app(which, request):
    """It compared silhouettes at two different K and read the difference
    causally, and it fired on all three apo recordings AND on the Zn2+ positive
    control. Gone from the strip, the document and the header."""
    run = request.getfixturevalue(which)
    for surface in (run.caveats, run.text, run.header):
        assert _OLD_CAUSAL not in surface


def test_the_ceiling_reaches_the_reader_when_the_cloud_has_no_separated_partition(back):
    """The replacement, on the recording whose events come from ONE distribution.

    If it fires, it must say the thing a ceiling supports — that no searched K is
    separated — and carry the report's own ceiling numbers rather than a second
    author's.
    """
    sep = back.q.separation
    assert sep is not None and sep.ceiling_silhouette is not None, (
        "fixture guard: this recording must reach K >= 2 for a silhouette to exist")
    print(f"\n[H4b/back] K={back.q.fit.k} silhouette={sep.silhouette:.3f} "
          f"ceiling={sep.ceiling_silhouette:.3f} at K={sep.ceiling_k} "
          f"profile={ {k: round(v, 3) for k, v in sep.reference_profile.items()} }")
    note = back.caveat("NOT THE BEST-SEPARATED PARTITION ON OFFER")
    assert sep.ceiling_substantial is False, (
        f"fixture guard: this recording is meant to have NO separated partition at "
        f"any K, but its ceiling is {sep.ceiling_silhouette:.3f} > "
        f"{SG.SEPARATION_FLOOR}")
    assert note, "ceiling under the floor and the reader is told nothing"
    assert "AND NEITHER IS ANY OTHER" in note
    assert "at any K" in note
    assert f"{sep.ceiling_silhouette:.3f}" in note
    assert f"K={sep.ceiling_k}" in note


def test_the_report_document_prints_the_ceiling_as_a_bound_not_as_a_rival(back):
    sep = back.q.separation
    if sep is None or not sep.reference_profile:
        pytest.skip("K = 1 on this run")
    assert "ceiling: the best plain k-means at ANY searched K scores" in back.text
    assert "k-means silhouette by K:" in back.text
    assert "BOUND on separability" in back.text
    for k, v in sep.reference_profile.items():
        assert f"K={k}:{v:.3f}" in back.text


# ===========================================================================
# no regression — what H4 put on these surfaces is still on them
# ===========================================================================


@pytest.mark.parametrize("which", ["back", "front"])
def test_the_header_still_carries_the_count_with_its_separation(which, request):
    run = request.getfixturevalue(which)
    q = run.q
    assert run.report.headline() in run.header
    assert f"{q.fit.k} subpopulation(s)" in run.header
    if q.separation is not None:
        assert f"{q.separation.silhouette:.3f}" in run.header
        assert q.separation.verdict in run.header
    assert f"{len(run.report.warnings)} caveat" in run.header
    assert "beside the figures and at the bottom of the text" in run.header


@pytest.mark.parametrize("which", ["back", "front"])
def test_every_caveat_the_run_raised_is_in_the_strip_verbatim(which, request):
    run = request.getfixturevalue(which)
    for w in run.report.warnings:
        assert w in run.caveats


@pytest.mark.parametrize("which", ["back", "front"])
def test_the_status_bar_still_reports_the_run(which, request):
    """Read, not asserted on wording: main_window.py is another agent's file."""
    run = request.getfixturevalue(which)
    assert "Subpopulations:" in run.status
    assert f"{len(run.report.warnings)} caveat" in run.status
