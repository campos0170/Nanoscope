"""DRIFT — the stability column and the sentences built on it, off a WIRED window.

Every defect here is in what the app TELLS you. A test that builds
``SubpopReportWindow`` in isolation and asserts an internal call proves nothing
about it, so each test below opens a real
:class:`~nanoscope.gui.main_window.MainWindow` on a real recording, runs the real
baseline probe, the real detector and the real ``Analysis > Event
subpopulations (GMM)…`` action off the real controller thread, drains the event
loop, and asserts the string in the caveat strip / header / status bar the reader
is looking at.

THE TWO RECORDINGS ARE THE DEFECT'S OWN SHAPE:

* ``burst`` — 22 seconds long, and **every event fires in the first eight**. The
  recording is as non-stationary as a recording can be, and until this pass the
  app said nothing at all about it: ``event_features.time_half`` split at the
  scope's own MEDIAN ONSET, a median halves its own sample by definition, and the
  pooled first-half fraction was therefore 0.50 — on this recording, on the
  owner's, on any recording. ``subpop_gmm._stability_warnings`` gates its
  recording-level caveat on that pooled value leaving 0.25-0.75, so the caveat
  was arithmetically unreachable. Three tests from the previous batch skipped
  themselves saying exactly that. This is the recording that makes it fire.
* ``lateshort`` — stationary overall (about half its events in each half of the
  clock), with one family of SHORT SHALLOW events confined to its last quarter,
  which the mixture does find. That is the class-level check, and it is where the
  band printed beside a class and the gate that named it have to be the same
  numbers.

WHY IT MATTERS ON THE OWNER'S FILES: the three apo recordings were acquired
−700 → −600 → −500 over nine minutes, so voltage and elapsed time are perfectly
confounded, and drift within a recording is the confound most likely to be
reported as a subpopulation. An app that looks like it checks for drift and does
not is worse than one that never claimed to.
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

from nanoscope.analysis import event_features as EF
from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N_SAMPLES = 1_100_000        # 22 s
N_EVENTS = 600
DURATION_S = N_SAMPLES / FS


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees, so a failure below cannot be an import that didn't happen."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    from nanoscope.gui import subpop_window as SW
    for mod in (EF, SG, SR, SW):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[DRIFT] nanoscope resolved from {root}")


# ---------------------------------------------------------------------------
# the two recordings
# ---------------------------------------------------------------------------


def _trace(case: str, seed: int = 3):
    """A synthetic current trace whose EVENT SUPPLY is the thing under test.

    ``burst`` fires all 600 events inside the first 7.8 s of a 22 s recording, so
    the recording is as non-stationary as a recording can be while every class in
    it is perfectly "stable" relative to the supply it came from — the
    recording-level check and nothing else.

    ``lateshort`` spreads the same 600 events over the whole recording and makes
    the last quarter's events SHORT and SHALLOW where the rest are long and deep,
    a family the mixture reliably separates. Its event supply is even across the
    clock, so the recording-level check is silent and the class-level one is not
    — which is the pair of outcomes that has to be told apart.
    """
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 7.0, N_SAMPLES)
    if case == "burst":
        starts = np.linspace(4_000, 390_000, N_EVENTS)
    else:
        starts = np.linspace(4_000, N_SAMPLES - 8_000, N_EVENTS)
    starts = (starts + rng.integers(-300, 300, starts.size)).astype(int)
    for s in starts:
        s = int(max(500, s))
        if case == "burst":
            depth = float(rng.normal(45.0, 14.0))
            dwell = int(np.exp(rng.normal(4.4, 0.35)))
        elif s > 0.75 * N_SAMPLES:
            depth, dwell = float(rng.normal(32.0, 3.0)), int(rng.integers(18, 28))
        else:
            depth, dwell = float(rng.normal(55.0, 8.0)), int(np.exp(rng.normal(4.4, 0.25)))
        x[s:s + dwell] -= depth
    return np.ascontiguousarray(x, dtype=np.float32)


def _document(tmp_dir: pathlib.Path, case: str) -> TraceDocument:
    arr = _trace(case)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_dir / f"drift_{case}.csv", format_name="synthetic",
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
    qapp.setApplicationName("RkDriftTimeHalfPrivate")
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

    def onsets(self) -> np.ndarray:
        """The app's own per-event onsets, from the report it is showing."""
        return self.report.features["onset_s"].to_numpy(dtype=float)


def _pump(qapp, ms: int = 60) -> None:
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
    tmp = tmp_path_factory.mktemp(f"rkdrift_{case}")
    monkeypatch.setattr(
        SR, "SubpopConfig",
        functools.partial(SR.SubpopConfig, k_grid=(1, 2, 3, 4), n_splits=2,
                          n_boot=0, with_tsne=False),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    boxed: list[str] = []

    def _dialog_exec(self, *a, **k):
        try:
            boxed.append(f"{self.windowTitle()!r}: {self.text()!r}")
        except Exception:                      # noqa: BLE001 — not a QMessageBox
            boxed.append(repr(self))
        return QtWidgets.QDialog.Rejected

    for _cls in (QtWidgets.QDialog, QtWidgets.QMessageBox):
        monkeypatch.setattr(_cls, "exec", _dialog_exec, raising=False)
        monkeypatch.setattr(_cls, "exec_", _dialog_exec, raising=False)
    for _nm in ("information", "warning", "critical", "about"):
        monkeypatch.setattr(
            QtWidgets.QMessageBox, _nm,
            staticmethod(lambda *a, _n=_nm, **k: (
                boxed.append(f"{_n}: {a[2] if len(a) > 2 else a}"),
                QtWidgets.QMessageBox.StandardButton.Cancel)[1]))

    win = MainWindow()
    win.resize(1900, 900)
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
    assert n_acc > 300, f"fixture guard: only {n_acc} accepted events"

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
    print(f"\n[DRIFT/{case}] accepted={n_acc}  header:\n{run.header}")
    print(f"[DRIFT/{case}] status ({len(run.status)} chars): {run.status}")
    print(f"[DRIFT/{case}] caveats:\n{run.caveats}")
    win._analysis_controller.abort()
    win._controller.abort()
    _pump(qapp, 100)
    return run


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def burst(qapp, tmp_path_factory, monkeypatch_module):
    run = _wired_run(qapp, tmp_path_factory, monkeypatch_module, "burst")
    yield run
    run.win.close()
    _pump(qapp, 50)


@pytest.fixture(scope="module")
def lateshort(qapp, tmp_path_factory, monkeypatch_module):
    run = _wired_run(qapp, tmp_path_factory, monkeypatch_module, "lateshort")
    yield run
    run.win.close()
    _pump(qapp, 50)


# ===========================================================================
# 1. THE DRIFT CAVEAT IS REACHABLE FROM THE APPLICATION
# ===========================================================================


def test_the_reader_is_told_the_recording_fired_everything_into_one_half(burst):
    """THE DEFECT, as the reader experiences it.

    Every event of this recording lands in its first eight seconds of twenty-two.
    Before this pass the app printed no non-stationarity caveat at all — not
    because it decided the recording was fine, but because the number it tests
    was 0.50 by construction and could not leave the band.

    The percentage and the half named are checked against the APP'S OWN onsets
    and the APP'S OWN duration, not against the fixture's constants.
    """
    note = burst.caveat("THE RECORDING IS NOT STATIONARY")
    assert note, (
        "the recording fires everything into its first eight seconds and the "
        f"reader was told nothing:\n{burst.caveats}")
    t = burst.onsets()
    duration = float(burst.report.meta["duration_s"])
    true_early = float((t <= duration / 2.0).mean())
    print(f"\n[DRIFT/burst] duration={duration:.3f}s  onsets {t.min():.3f}..{t.max():.3f}"
          f"  clock early fraction={true_early:.4f}\n{note}")
    m = re.search(r"(\d+) % of all gated events fall in the (first|second) half", note)
    assert m, note
    pct, half = int(m.group(1)), m.group(2)
    share = true_early if half == "first" else 1.0 - true_early
    assert pct == round(100 * share), (pct, half, true_early)
    assert half == "first" and pct == 100
    assert f"read against this {true_early:.2f}" in note


def test_the_column_the_caveat_reads_is_the_clock_and_not_the_median(burst):
    """The invariant under the sentence, checked on the app's own column.

    ``time_half`` as the app produced it must equal the clock split of the app's
    own onsets over the app's own duration — and must NOT equal the median
    split, which for this recording says a comfortable half-and-half.
    """
    feats = burst.report.features
    t = burst.onsets()
    duration = float(burst.report.meta["duration_s"])
    shown = feats["time_half"].to_numpy()
    clock = EF.time_half(t, span=(0.0, duration))
    median = EF.time_half(t)
    assert np.array_equal(shown, clock)
    assert (shown == "early").all(), "every event is in the first half of the clock"
    frac_median = float((median == "early").mean())
    print(f"\n[DRIFT/burst] clock says {float((clock == 'early').mean()):.3f} early; "
          f"the median split said {frac_median:.3f}")
    assert abs(frac_median - 0.5) <= 1.0 / len(t)
    assert not np.array_equal(shown, median)


def test_the_pooled_number_on_screen_is_a_measurement_and_not_a_constant(lateshort):
    """On the OTHER recording the same column reports a different number.

    A statistic that reads 0.50 whatever the recording did is not measuring the
    recording. These two wired runs are the same code path on two different event
    supplies; the pooled value has to tell them apart.
    """
    t = lateshort.onsets()
    duration = float(lateshort.report.meta["duration_s"])
    pooled = float((lateshort.report.features["time_half"] == "early").mean())
    assert pooled == pytest.approx(float((t <= duration / 2.0).mean()))
    print(f"\n[DRIFT/lateshort] pooled early fraction = {pooled:.4f}")
    assert 0.25 < pooled < 0.75, "this recording IS roughly stationary overall"
    assert not lateshort.caveat("THE RECORDING IS NOT STATIONARY")


# ===========================================================================
# 2. THE BAND ON SCREEN IS THE BAND THE GATE USED
# ===========================================================================


def test_every_class_named_on_screen_is_outside_the_band_printed_beside_it(lateshort):
    """THE DEFECT: at pooled 0.505 the band prints ``0.25-0.75`` and a class
    printed at ``0.25`` was condemned for being outside it, because the gate ran
    on a full-precision band recomputed from an already-rounded column.

    Read here off the printed strings and the printed table, both from the window
    the reader is looking at: named IF AND ONLY IF outside, no exceptions.
    """
    unstable = lateshort.caveat("TIME-UNSTABLE CLASS(ES)")
    assert unstable, (
        "fixture guard: this recording's late class must be named\n"
        + lateshort.caveats)
    print(f"\n[DRIFT/lateshort] {unstable}")
    named = {int(m) for m in re.findall(r"subpop (\d+) \(n=", unstable)}
    lo, hi = (float(x) for x in
              re.search(r"outside (\d\.\d\d)-(\d\.\d\d)", unstable).groups())
    table = lateshort.q.table
    seen = 0
    for sp, v in zip(table["subpop"], table["early_frac"]):
        if not np.isfinite(v):
            continue
        seen += 1
        outside = not (lo <= float(v) <= hi)
        assert (int(sp) in named) == outside, (
            f"subpop {sp} prints {float(v):.2f}, the band prints {lo:.2f}-{hi:.2f}, "
            f"named={int(sp) in named}\n{unstable}")
    assert seen >= 2 and named, (seen, named)


def test_the_band_on_screen_recomputes_from_the_pooled_value_on_screen(lateshort):
    """A reader with the sentence and a calculator must land on the same band."""
    unstable = lateshort.caveat("TIME-UNSTABLE CLASS(ES)")
    lo, hi = (float(x) for x in
              re.search(r"outside (\d\.\d\d)-(\d\.\d\d)", unstable).groups())
    pooled = float(
        re.search(r"pooled early fraction of (\d\.\d\d)", unstable).group(1))
    rlo, rhi = SG.stability_band_for(pooled)
    assert (round(rlo, 2), round(rhi, 2)) == (lo, hi), (pooled, lo, hi, rlo, rhi)
    # ...and that pooled is the app's own column, not a third number
    shown = float((lateshort.report.features["time_half"] == "early").mean())
    assert abs(pooled - shown) <= 0.015, (pooled, shown)


def test_the_class_named_on_screen_carries_the_ratio_the_gate_measured(lateshort):
    """The gate is an ODDS RATIO against the recording's own early odds, and the
    sentence used to give the reader only the two fractions and a claim about
    them ("3:1 into one half") that is a different statement. The measured ratio
    is now beside the class, on screen, and recomputes from the two printed
    numbers either side of it."""
    unstable = lateshort.caveat("TIME-UNSTABLE CLASS(ES)")
    pooled = float(
        re.search(r"pooled early fraction of (\d\.\d\d)", unstable).group(1))
    m = re.search(
        r"subpop \d+ \(n=\d+, early_frac (\d\.\d\d), early odds (1/)?([\d.]+)",
        unstable)
    assert m, unstable
    v, inverted, ratio = float(m.group(1)), bool(m.group(2)), float(m.group(3))
    measured = (v / (1 - v)) / (pooled / (1 - pooled))
    expect = 1.0 / measured if inverted else measured
    print(f"\n[DRIFT/lateshort] early_frac {v} vs pooled {pooled} -> printed "
          f"{'1/' if inverted else ''}{ratio}, recomputed {expect:.3f}")
    assert (measured < 1.0) == inverted, (measured, inverted, unstable)
    assert ratio == pytest.approx(expect, abs=0.06), (ratio, expect)


def test_the_sentence_only_makes_the_absolute_claim_where_it_is_true(lateshort):
    """"at least 3:1 concentrated into one half" is a claim about the CLASS."""
    unstable = lateshort.caveat("TIME-UNSTABLE CLASS(ES)")
    named = {int(m) for m in re.findall(r"subpop (\d+) \(n=", unstable)}
    table = lateshort.q.table
    band_lo, band_hi = SG.STABILITY_BAND
    all_absolute = all(
        not (band_lo <= float(v) <= band_hi)
        for sp, v in zip(table["subpop"], table["early_frac"]) if int(sp) in named)
    claim = "at least 3:1 concentrated into one half of the recording" in unstable
    assert claim == all_absolute, (claim, all_absolute, unstable)


# ===========================================================================
# 3. THE ONE SENTENCE THAT TRAVELS
# ===========================================================================


def test_no_travelling_surface_carries_the_number_the_report_disowns(lateshort):
    """THE DEFECT. ``one_line()`` ended "; k-means K=3 on the same coordinates
    scores 0.197" — on the window header and, since the status bar was routed
    through ``headline()``, on the status bar too — while the report's own
    ceiling caveat calls a number of that kind "a CEILING ... not a rival result"
    and quotes a different one."""
    assert lateshort.q is not None and lateshort.q.separation is not None
    for name, surface in (("headline", lateshort.report.headline()),
                          ("window header", lateshort.header),
                          ("status bar", lateshort.status)):
        assert "k-means" not in surface, f"{name}: {surface}"
        assert "same coordinates" not in surface, f"{name}: {surface}"


def test_the_reference_is_still_in_the_document_where_it_is_framed(lateshort):
    """Deleted from the sentence that travels, NOT from the report."""
    text = lateshort.text
    sep = lateshort.q.separation
    assert f"k-means K={sep.reference_k} on the same coordinates" in text
    assert "BOUND on separability" in text
    assert text.index("BOUND on separability") > text.index("k-means K=")


def test_the_verdict_survives_the_status_bar_being_elided(burst):
    """WORD ORDER AS A CORRECTNESS PROPERTY, measured on the real widget.

    The bar is one elided line. On the owner's −700 mV recording only about 72 %
    of the old 425-character message fitted at 1900 px; what fell off the end was
    "reads as ONE CONTINUUM · N caveats" and what survived included the bare
    k-means rival. The truncation kept the misleading half.

    ``burst``'s message is the same shape — long enough to be elided even on a
    1900 px bar. Swept from a bar barely wide enough to name the count down to
    the full width: whenever the reader can see the class count at all, the
    verdict that qualifies it is on screen too, and the disowned number never is.
    """
    bar = burst.win.statusBar()
    fm = bar.fontMetrics()
    msg = burst.status
    verdict = burst.q.separation.verdict
    full = fm.horizontalAdvance(msg)
    real = max(bar.width() - 20, 1)
    print(f"\n[DRIFT/burst] bar {bar.width()} px, message {len(msg)} chars, "
          f"advance {full} px, fits {100 * min(1.0, real / full):.1f} %")
    print(f"  at the real width: {fm.elidedText(msg, QtCore.Qt.ElideRight, real)}")
    assert full > real, (
        "fixture guard: this message must actually be elided at the real width")
    # 1. at the width the reader actually has, the verdict is on the bar
    assert verdict in fm.elidedText(msg, QtCore.Qt.ElideRight, real)
    # 2. the verdict is the FIRST qualifier after the count, so no truncation can
    #    show a qualifier without it — swept over every width, on the real widget
    sil = f"{burst.q.separation.silhouette:.3f}"
    later = [sil, "nearer another class's centre", "confidently assigned"]
    i_v, i_c = msg.index(verdict), msg.index("subpopulation(s)")
    assert 0 < i_v - i_c < 40, msg
    for bit in later:
        assert msg.index(bit) > i_v, (bit, msg)
    checked = 0
    for avail in range(200, full + 200, 50):
        shown = fm.elidedText(msg, QtCore.Qt.ElideRight, avail)
        assert "k-means" not in shown, (avail, shown)
        if any(b in shown for b in later):
            checked += 1
            assert verdict in shown, (avail, shown)
    assert checked >= 5, checked


def test_the_status_bar_is_still_the_reports_own_sentence(lateshort):
    """No regression on the single-source rule this pass inherited."""
    assert lateshort.status == f"Subpopulations: {lateshort.report.headline()}"
    assert lateshort.report.headline() in lateshort.header
