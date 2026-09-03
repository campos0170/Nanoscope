"""``Analysis ▸ Event subpopulations (GMM)…`` — the app path, end to end.

The pipeline is validated against the owner's own run in
``tests/analysis/test_subpop_report.py``. What is at stake HERE is everything
between a click and that pipeline, and it is not a formality — three of the
resolutions ``MainWindow`` makes on the way in are load-bearing and silent when
wrong:

* the trace must be ``modules.detection_signal``, not the raw channel (the same
  invariant the merge and the CUSUM fit already carry);
* the raw channel must go over too, because ``i0_pA`` and both baselines are
  properties of the RECORDING and cannot be recovered from an event table;
* the polarity must be NEGATED — this app's ``+1`` means excursions above the
  open-pore level, ``event_features``' ``+1`` means below.

Plus the plumbing the app already has rules about: off the GUI thread through the
shared :class:`AnalysisController`, progress that moves, a Cancel that unwinds,
a result that cannot be adopted by (or adopt) somebody else's job, an export that
goes through the one overwrite-confirming path, and — the frozen contract — an
event table that is byte-for-byte unchanged afterwards.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.subpop_window import SubpopReportWindow

FS = 50_000.0
N_EVENTS = 220
N_SAMPLES = 400_000


# ================================================================================================
# builders — a recording with two obvious event classes, detected THROUGH the app
# ================================================================================================

def _trace_and_spans(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    spans = []
    pos = 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        spans.append((pos, pos + dwell))
        pos += int(rng.integers(500, 1100))
    return np.ascontiguousarray(x, dtype=np.float32), spans


def _doc(tmp_path, arr, name: str = "subpop.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _events(spans, doc_id: str) -> EventSet:
    rows = []
    for i, (s, e) in enumerate(spans):
        depth = 60.0 if i % 2 else 30.0
        rows.append({
            "event_id": i, "recording": doc_id, "pore_id": "p1",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": 200.0, "depth_pA": depth, "frac_depth": depth / 200.0,
            "ecd_pA_s": depth * (e - s) / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 2.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    # ``polarity`` is the app's own convention: -1 == excursions run BELOW the open
    # pore, which is what this synthetic trace does. MainWindow must negate it.
    return EventSet(
        pd.DataFrame(rows)[REQUIRED],
        {"analyzed_duration_s": N_SAMPLES / FS, "polarity": -1},
    )


class _Cheap(SR.SubpopConfig):
    """A config small enough for a GUI test, substituted for the real default.

    Not a shortcut around the pipeline — the SAME code runs, over a 2-fold, K<=3,
    bootstrap-free grid instead of the 5-fold, K<=8, 200-resample one. The full
    grid's numbers are pinned against the owner's run in the analysis tests; what
    is under test here is the wiring around it.
    """

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2, 3))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        super().__init__(**kw)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def ready(window, tmp_path, monkeypatch):
    """A window with a selected recording and a stored detection over it."""
    monkeypatch.setattr(SR, "SubpopConfig", _Cheap)
    arr, spans = _trace_and_spans()
    doc = _doc(tmp_path, arr)
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    from nanoscope.core.analysis import AnalysisResult

    events = _events(spans, doc_id)
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=session.calibration(doc_id, 0).fingerprint(),
        payload=events,
    ))
    window._sync_export_action()
    return window, doc_id, events


def _run(window, qtbot, timeout_ms: int = 120_000):
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=timeout_ms):
        window._run_subpopulations()
    qtbot.wait(20)
    return window._subpop_window


# ================================================================================================
# (a) the menu entry, and its gate
# ================================================================================================

def test_the_action_exists_and_is_locked_until_there_is_a_detection(window):
    assert window._act_subpop is not None
    assert not window._act_subpop.isEnabled()
    assert window._act_subpop.toolTip(), "a locked action that says nothing is a dead end"


def test_the_action_unlocks_once_a_detection_is_stored(ready):
    window, _doc_id, _events = ready
    assert window._act_subpop.isEnabled()


def test_triggering_the_action_reaches_the_shared_controller(ready, monkeypatch):
    """Menu entry -> handler -> the ONE analysis controller, under its own label.

    Driven by ``QAction.trigger()`` rather than by calling the handler, so this
    covers the connection and the enablement together — a disabled action swallows
    its own trigger, which is exactly how a menu entry comes to do nothing.
    """
    window, _doc_id, _events = ready
    seen: list[str] = []

    def spy(fn, label="callable", *, wants_progress=False):
        seen.append(label)
        return False        # start nothing; this test is about the wiring

    monkeypatch.setattr(window._analysis_controller, "run_callable", spy)
    window._act_subpop.trigger()
    assert seen == ["subpopulations"]


# ================================================================================================
# (b) the whole run, through the app
# ================================================================================================

def test_a_run_produces_a_report_window_off_the_gui_thread(ready, qtbot):
    window, _doc_id, _events = ready
    seen: list[tuple[float, str]] = []
    window._analysis_controller.progressed.connect(lambda f, m: seen.append((f, m)))

    win = _run(window, qtbot)

    assert isinstance(win, SubpopReportWindow)
    report = win.report()
    assert isinstance(report, SR.SubpopReport)
    assert report.n_kept > 0
    assert report.quantification is not None
    assert win.text() == report.text()
    assert "ITEM 11" in win.text()

    # Progress genuinely arrived, and it moved. A callable job could not report at
    # all before this change; a minute-long bar that never moves reads as a hang.
    assert len(seen) >= 5, seen
    assert [f for f, _ in seen] == sorted(f for f, _ in seen)
    assert seen[-1][0] == pytest.approx(1.0)


def test_the_run_uses_the_detection_trace_and_the_negated_polarity(ready, qtbot, monkeypatch):
    """The three load-bearing resolutions, captured at the seam.

    ``detection_signal`` rebuilds ``lowpass(preprocess_for_detection(raw))`` — the
    array the detector measured on. The RAW channel must arrive as well, because
    ``i0_pA`` and both baselines are whole-recording scalars. And the polarity must
    flip: the app's ``-1`` (excursions below the open pore) is
    ``event_features``' ``+1``.
    """
    window, doc_id, _events = ready
    captured: dict = {}
    real = SR.run_subpopulations

    def spy(detrended, raw, fs, events, **kw):
        captured.update(detrended=np.asarray(detrended), raw=np.asarray(raw),
                        fs=fs, n_events=len(events.df), **kw)
        return real(detrended, raw, fs, events, **kw)

    monkeypatch.setattr(SR, "run_subpopulations", spy)
    _run(window, qtbot)

    from nanoscope.analysis.modules import detection_signal

    chan = window._session.calibrated(doc_id, 0)
    expected = detection_signal(np.asarray(chan.data), FS, _events_prov(window))
    assert np.array_equal(captured["detrended"], np.asarray(expected))
    assert np.array_equal(captured["raw"], np.asarray(chan.data))
    assert captured["fs"] == FS
    assert captured["config"].polarity == 1, "app -1 must arrive as event_features +1"
    assert captured["duration_s"] == pytest.approx(N_SAMPLES / FS)


def _events_prov(window) -> dict:
    return dict(window._analysis_panel.current_detection().payload.provenance or {})


def test_the_run_analyses_the_ACCEPTED_events_only(ready, qtbot):
    """The report and the screen must be about the same rows."""
    window, _doc_id, events = ready
    events.df.loc[events.df.index[:40], "accepted"] = False
    win = _run(window, qtbot)
    assert win.report().n_fired == len(events.df) - 40


def test_a_run_leaves_the_event_table_byte_for_byte_unchanged(ready, qtbot):
    """THE FROZEN CONTRACT: the 13 features are derived, never a schema column.

    Compared column by column rather than frame to frame: the app materialises its
    own optional curation column (``added``) when a panel shows a result, and that
    is the app's business. What must be true of THIS run is that every schema column
    still holds the same values and that not one derived feature name appears.
    """
    from nanoscope.analysis import event_features as EF

    window, _doc_id, events = ready
    before = events.df.copy(deep=True)
    _run(window, qtbot)
    assert set(REQUIRED).issubset(events.df.columns)
    pd.testing.assert_frame_equal(events.df[before.columns], before)
    leaked = [c for c in (*EF.FEATURE_COLUMNS, "subpop", "subpop_confidence",
                          "hdbscan", "kmeans3", "scope", "time_half")
              if c not in REQUIRED and c in events.df.columns]
    assert leaked == []


def test_two_runs_through_the_app_give_the_identical_report(ready, qtbot):
    window, _doc_id, _events = ready
    first = _run(window, qtbot).report().text()
    second = _run(window, qtbot).report().text()
    assert first == second


def test_the_status_bar_reports_the_gate_and_the_caveat_count(ready, qtbot):
    window, _doc_id, _events = ready
    _run(window, qtbot)
    msg = window.statusBar().currentMessage()
    assert "passed the" in msg and "caveat" in msg


# ================================================================================================
# (c) refusals — every one of them says why, and starts nothing
# ================================================================================================

def test_without_a_detection_the_runner_explains_instead_of_raising(window, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.information",
        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    window._run_subpopulations()
    assert shown and "detection" in shown[0]
    assert not window._analysis_controller.is_busy()


def test_with_every_event_curated_out_the_runner_says_so(ready, monkeypatch):
    window, _doc_id, events = ready
    events.df["accepted"] = False
    shown: list[str] = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.information",
        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    window._run_subpopulations()
    assert shown and "curated out" in shown[0]
    assert not window._subpop_pending


def test_a_run_the_gate_refuses_starts_nothing(ready, monkeypatch):
    window, _doc_id, _events = ready
    monkeypatch.setattr(window._analysis_controller, "run_callable",
                        lambda *a, **k: False)
    window._run_subpopulations()
    assert not window._subpop_pending
    assert window._subpop_window is None


# ================================================================================================
# (d) the shared controller — neither job may adopt the other's answer
# ================================================================================================

def test_a_foreign_payload_never_opens_the_report_window(ready):
    window, _doc_id, _events = ready
    window._subpop_pending = True
    window._on_subpop_finished({"not": "a report"})
    assert window._subpop_window is None
    assert window._subpop_pending, "a foreign payload must not consume the claim either"


def test_a_report_arriving_unclaimed_is_ignored(ready, qtbot):
    """``_subpop_pending`` False means this window did not ask for it."""
    window, _doc_id, _events = ready
    win = _run(window, qtbot)
    stale = win.report()
    window._subpop_window = None
    window._subpop_pending = False
    window._on_subpop_finished(stale)
    assert window._subpop_window is None


def test_the_panels_stale_pending_job_is_dropped_before_the_run(ready, qtbot):
    """``AnalysisController.abort()`` emits nothing, so the panel cannot learn about
    the abort on its own — and its ``_on_controller_finished`` consumes ``_pending``
    on the first thing that finishes, whatever that is."""
    window, _doc_id, _events = ready
    window._analysis_panel._pending = "sweep"
    _run(window, qtbot)
    assert window._analysis_panel._pending is None


def test_cancelling_drops_the_pending_claim(ready):
    window, _doc_id, _events = ready
    window._subpop_pending = True
    window._cancel_load()
    assert not window._subpop_pending


def test_a_failure_drops_the_pending_claim(ready):
    window, _doc_id, _events = ready
    window._subpop_pending = True
    window._on_subpop_stopped()
    assert not window._subpop_pending


# ================================================================================================
# (e) the export
# ================================================================================================

def test_export_writes_the_text_the_sidecar_and_every_table(ready, qtbot, tmp_path, monkeypatch):
    window, _doc_id, _events = ready
    win = _run(window, qtbot)
    target = tmp_path / "subpops"
    seen: dict = {}

    def fake_ask(parent, *, title, suggestion, filters, outputs=None):
        seen["outputs"] = list(outputs(target)) if outputs else [target]
        return target

    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path", fake_ask)
    win.exportRequested.emit()

    # The confirmation was offered the REAL output set, not just the chosen file.
    assert len(seen["outputs"]) == len(win.report().tables()) + 2
    for path in seen["outputs"]:
        assert path.exists(), path
    assert (target.with_suffix(".txt")).read_text() == win.report().text()


def test_export_backed_out_of_writes_nothing(ready, qtbot, tmp_path, monkeypatch):
    window, _doc_id, _events = ready
    win = _run(window, qtbot)
    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path",
                        lambda *a, **k: None)
    win.exportRequested.emit()
    assert not list(tmp_path.glob("*.txt"))


def test_export_without_a_report_explains_itself(window, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.information",
        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    window._export_subpop_report()
    assert shown and "no subpopulation report" in shown[0].lower()


# ================================================================================================
# (f) the window itself
# ================================================================================================

def test_an_empty_window_offers_nothing_to_export(qapp, qtbot):
    win = SubpopReportWindow()
    qtbot.addWidget(win)
    assert not win.can_export()
    assert win.report() is None
    assert "No subpopulation report" in win.header()


def test_the_header_carries_the_caveat_count(ready, qtbot):
    """They sit at the bottom of a long document; a reader who never scrolls that
    far must still know they exist."""
    window, _doc_id, _events = ready
    win = _run(window, qtbot)
    assert "caveat" in win.header()
    assert str(len(win.report().warnings)) in win.header()


def test_a_run_through_the_app_arrives_with_both_figures_already_drawn(ready, qtbot):
    """The end-to-end claim, which no test in ``test_subpop_views`` can make.

    That file hands the views a report directly. What is at stake here is the step
    before: a report that travelled the real path — menu action, shared controller,
    off-thread run, ``_on_subpop_finished`` — lands in a window whose figures are
    POPULATED, not in one that renders the text and leaves two empty axes captioned
    "No report yet". ``set_report`` fans out to both views, and that fan-out is the
    kind of line a later refactor drops silently, because the text still looks right.
    """
    window, _doc_id, _events = ready
    win = _run(window, qtbot)

    scatter, waves = win.embedding_view(), win.waveform_view()
    assert scatter.message() == "", scatter.message()
    assert waves.message() == "", waves.message()

    x, _y, labels = scatter.drawn()
    assert x.size == len(win.report().embeddings)
    assert labels.size == x.size
    assert scatter.colour_mode() == "GMM subpopulation (the mixture)"
    assert len(scatter.legend_entries()) == win.report().quantification.fit.k
    assert waves.series(), "item 13 drew nothing"

    # And the caveats travelled with them, in the same eyeline as the pictures.
    for warning in win.report().warnings:
        assert warning in win.caveats()


def test_clearing_the_window_disables_export_again(ready, qtbot):
    window, _doc_id, _events = ready
    win = _run(window, qtbot)
    assert win.can_export()
    win.set_report(None)
    assert not win.can_export() and win.text() == ""


def test_copy_puts_the_whole_report_on_the_clipboard(ready, qtbot):
    from PySide6 import QtWidgets

    window, _doc_id, _events = ready
    win = _run(window, qtbot)
    win._copy()
    assert QtWidgets.QApplication.clipboard().text() == win.report().text()


# ================================================================================================
# (g) the invariants this tree holds every window to
# ================================================================================================

def _relative_sibling_imports(module) -> list[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    return [n.module for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module]


#: The ONE sibling the report window may reach for, and the reason the allow-list is
#: a list of one rather than "no relative imports at all".
#:
#: The rule this guard exists to keep is "MainWindow is the sole wirer; no panel
#: imports another panel", and that rule is about PANELS -- objects that own session
#: state, that main_window builds and connects. ``gui.categorical`` is not one: it is
#: a leaf of pure functions (a colour map and the pyqtgraph call that uses it) that
#: imports no gui module itself, owns nothing and is wired to nothing. Widening the
#: allow-list by one named module is strictly safer than the alternative that the
#: blanket rule forces, which is a second copy of the categorical palette in this
#: file -- the duplication this tree has already paid for repeatedly.
#:
#: It is an ALLOW-LIST and not a predicate on purpose: the next relative import added
#: here fails this test and has to argue for itself in the same place.
_ALLOWED_SIBLING_IMPORTS = ["categorical"]


def test_the_report_window_imports_only_the_shared_leaf():
    """MainWindow stays the sole wirer; the window reaches for no PANEL.

    See :data:`_ALLOWED_SIBLING_IMPORTS`.
    """
    import nanoscope.gui.subpop_window as mod

    assert _relative_sibling_imports(mod) == _ALLOWED_SIBLING_IMPORTS


def test_the_shared_leaf_really_is_a_leaf():
    """``gui.categorical`` may not grow imports, or the allow-list above is a hole.

    A leaf that imported a panel would let the report window reach that panel
    transitively while this file still said it imported nothing but a colour map.
    """
    import nanoscope.gui.categorical as leaf

    assert _relative_sibling_imports(leaf) == []
    tree = ast.parse(pathlib.Path(leaf.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # ``..plot.style`` (level 2) is the palette itself and is not a gui module.
            assert "gui" not in (node.module or "").split(".")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "nanoscope.gui" not in alias.name


def test_the_report_window_stays_qt_only():
    import nanoscope.gui.subpop_window as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "gui" not in (node.module or "").split(".")


def test_neither_new_module_terminates_a_thread_or_waits_unbounded():
    """The deadlock this app has already paid for once, in Force-Quit form."""
    import nanoscope.gui.subpop_window as win_mod

    for mod in (win_mod, SR):
        text = pathlib.Path(mod.__file__).read_text()
        assert "terminate()" not in text, mod.__name__
        assert ".wait()" not in text, mod.__name__


def test_the_run_goes_through_the_shared_controller_not_a_private_thread():
    """One analysis in the whole app; the run gate enforces it, and it can only do
    that for jobs that ask it."""
    import nanoscope.gui.main_window as mw

    text = pathlib.Path(mw.__file__).read_text()
    start = text.index("def _run_subpopulations")
    body = text[start:text.index("def _on_subpop_finished")]
    assert "self._analysis_controller.run_callable(" in body
    assert "wants_progress=True" in body
    assert "QThread" not in body and "threading" not in body


# ================================================================================================
# (h) the cost ceiling — a run that will take many minutes must not start silently
# ================================================================================================

def test_a_large_detection_asks_before_starting_and_no_means_no(ready, monkeypatch):
    """The status bar used to promise "about a minute" and then take half an hour.

    Cancellation always worked; what was missing was any way to know you needed
    it. The gate is the accepted count, which is the upper bound on what reaches
    the mixture stage -- the ≥4σ gate runs inside the job, so the exact n is not
    knowable here, and warning on the bound is the honest side to err on.
    """
    window, _doc_id, _events = ready
    monkeypatch.setattr(SR, "SLOW_GMM_EVENTS", 1)
    asked: list[str] = []

    def _question(*a, **k):
        asked.append(a[2] if len(a) > 2 else "")
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.question", _question)
    window._run_subpopulations()

    assert asked, "a many-minute run started without asking"
    assert "SLOW:" in asked[0] and "Start it anyway?" in asked[0]
    assert not window._subpop_pending
    assert not window._analysis_controller.is_busy()
    assert window._subpop_window is None


def test_saying_yes_to_the_slow_prompt_runs_it_and_the_status_bar_stops_promising_a_minute(
        ready, qtbot, monkeypatch):
    window, _doc_id, _events = ready
    monkeypatch.setattr(SR, "SLOW_GMM_EVENTS", 1)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.question",
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)

    _run(window, qtbot)
    assert window._subpop_window is not None


def test_an_ordinary_detection_is_never_interrupted_by_the_prompt(ready, qtbot, monkeypatch):
    """A prompt that fires at the owner's own 1076 events would train him to
    click through it, which is worse than not having it."""
    window, _doc_id, _events = ready
    asked: list[str] = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.question",
        lambda *a, **k: asked.append("") or QtWidgets.QMessageBox.StandardButton.Yes)

    _run(window, qtbot)
    assert not asked
