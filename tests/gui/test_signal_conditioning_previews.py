"""Signal Conditioning previews (D-beta), driven through a real MainWindow.

One long-lived worker thread per window, a 250 ms debounce coalescing to the latest
request, cancellation, previews of the VISIBLE window only, the honest "window estimate"
versus "exact" line, the on-demand exact pass, Follow main trace through
``sigRangeChangedManually``, the Time / Spectrum / Response views and Send to notch.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import threading
import time
from pathlib import Path

import nanoscope  # noqa: F401  -> Qt plugin bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.analysis import conditioning_preview as cp
from nanoscope.analysis.cancel import OperationCancelled, check_cancelled
from nanoscope.analysis.conditioning_recipes import complete_recipe
from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import signal_conditioning as sc
from nanoscope.gui import worker_thread as wt
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.signal_conditioning import SignalConditioningWindow

FS = 50_000.0


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)


def _signal(n=300_000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = np.full(n, -1000.0) + rng.normal(0.0, 8.0, n) + 5.0 * np.sin(2 * np.pi * 4570.8 * t)
    x += 30.0 * np.sin(2 * np.pi * 0.5 * t)
    for s in range(10_000, n, 30_000):
        x[s:s + 300] += 300.0
    return x.astype(np.float32)


def _doc(tmp_path, name="a.dat", n=300_000) -> TraceDocument:
    sig = _signal(n)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _new_window(tmp: Path | None = None) -> tuple[MainWindow, str | None]:
    import nanoscope.gui.main_window as mw

    w = MainWindow()
    # Teardown runs outside the suite's function-scoped close-guard patch.
    w._confirm_close_choice = lambda: mw._CLOSE_DISCARD
    w.resize(1512, 875)
    w.show()
    did = None
    if tmp is not None:
        did = w._session.add_document(_doc(tmp))
        w._create_default_layers(w._session.document(did))
        w._on_selection(did, 0)
        _pump(20)
    return w, did


def _dispose(w: MainWindow) -> None:
    if w._signal_conditioning is not None:
        w._signal_conditioning.abort()
    w._analysis_controller.abort()
    w.close()
    w.deleteLater()
    _pump(10)


@pytest.fixture(scope="module")
def shared(qapp, tmp_path_factory):
    """ONE MainWindow, and so one preview thread, for the whole module: each MainWindow
    teardown leaves ~1,400 dead Qt wrappers behind in PySide6 6.11, and a second window's
    QThread construction late in the file tripped its SignalManager crash (measured 1 run
    in 6). The teardown test is therefore the LAST to use it."""
    register_all()
    tmp = tmp_path_factory.mktemp("sc_previews")
    w, did = _new_window(tmp)
    state = {"window": w, "did": did, "tmp": tmp}
    yield state
    _dispose(w)


@pytest.fixture
def loaded(shared):
    """The shared window, put back to a fresh form, the Time tab and the whole trace."""
    window, did = shared["window"], shared["did"]
    det = window._analysis_panel.detection_panel
    det.set_params(complete_recipe({}))
    det.set_notch_lines(None)
    det.response_chain_check.setChecked(False)
    det.select_stage("lowpass")
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    ws.resize(ws.DEFAULT_WIDTH, 820)            # the default width; a test may resize it
    ws._memo.clear()
    ws.set_follow_main(True)
    ws.set_time_mode("overlay")
    ws.set_response_view("magnitude")
    ws.set_preview_tab("time")
    ws.lines_note.setText("")
    window._trace_view.zoom_x_to(0.0, 300_000 / FS)
    _pump(30)
    window._push_conditioning_source()
    _idle(None, ws)
    yield window, did
    # Closing cancels the work and keeps the window's one thread; abort() is teardown's.
    if window._signal_conditioning is not None:
        window._signal_conditioning.close()


def _pump(ms: int) -> None:
    """Process events for ``ms`` without building a QTimer/QEventLoop per poll.

    qtbot.wait/waitUntil construct both on every call; thousands of those per file
    trip PySide6 6.11's SignalManager crash (see the 2026-09-13 root-cause note)."""
    deadline = time.monotonic() + ms / 1000.0
    while True:
        QtWidgets.QApplication.processEvents()
        if time.monotonic() >= deadline:
            return
        time.sleep(0.003)


def _until(predicate, timeout=20_000) -> None:
    deadline = time.monotonic() + timeout / 1000.0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout} ms")
        QtWidgets.QApplication.processEvents()
        time.sleep(0.003)


def _idle(qtbot, ws, timeout=20_000):
    _until(lambda: ws.preview_state() != "pending" and not ws._debounce.isActive(), timeout)
    _pump(5)


def _open(window, qtbot, tab="time"):
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    ws.set_preview_tab(tab)
    _idle(qtbot, ws)
    return ws


def _zoom_main(window, qtbot, x0, x1):
    window._trace_view.zoom_x_to(x0, x1)
    _pump(30)


# -- one worker, debounced, cancellable, off the GUI thread ----------------------------------
def test_one_long_lived_worker_runs_debounced_previews_off_the_gui_thread(
        loaded, qtbot, monkeypatch):
    window, _ = loaded
    seen = []
    real = cp.conditioned_trace

    def recording(raw, fs, params):
        seen.append((threading.current_thread(), raw.size))
        return real(raw, fs, params)

    monkeypatch.setattr(cp, "conditioned_trace", recording)
    ws = _open(window, qtbot)
    thread = ws.worker_thread()
    assert thread is not None and thread.isRunning()
    det = window._analysis_panel.detection_panel
    seen.clear()
    for k in range(10):                         # ten rapid spin steps
        det.set_params({"filter_cutoff_hz": 9_000.0 - 100.0 * k})
    _idle(qtbot, ws)
    assert len(seen) == 1, "debounced and coalesced to the latest request"
    assert all(t is not threading.main_thread() for t, _n in seen)
    assert ws.worker_thread() is thread, "the same thread serves every preview"
    assert ws.last_result().processed_window is not None
    det.set_params({"filter_cutoff_hz": 4_000.0})
    _idle(qtbot, ws)
    assert ws.worker_thread() is thread
    built = []
    real_start = sc.start_worker_thread
    monkeypatch.setattr(sc, "start_worker_thread",
                        lambda *a, **k: built.append(1) or real_start(*a, **k))
    for _ in range(3):                          # close and reopen: no thread is rebuilt
        ws.close()
        _pump(5)
        assert ws.worker_thread() is thread and thread.isRunning()
        window._open_signal_conditioning()
        _idle(qtbot, ws)
    assert ws.worker_thread() is thread and built == []
    assert ws.last_result() is not None and ws.last_result().processed_window is not None


def test_a_new_edit_cancels_the_running_preview(loaded, qtbot, monkeypatch):
    window, _ = loaded
    ws = _open(window, qtbot)
    running = threading.Event()
    outcomes = []
    real = cp.conditioned_trace

    def slow(raw, fs, params):
        running.set()
        try:
            for _ in range(400):
                check_cancelled()
                time.sleep(0.005)
        except OperationCancelled:
            outcomes.append(("cancelled", params.get("filter_cutoff_hz")))
            raise
        outcomes.append(("done", params.get("filter_cutoff_hz")))
        return real(raw, fs, params)

    monkeypatch.setattr(cp, "conditioned_trace", slow)
    det = window._analysis_panel.detection_panel
    det.set_params({"filter_cutoff_hz": 7_000.0})
    _until(running.is_set, 5_000)
    det.set_params({"filter_cutoff_hz": 3_000.0})
    _idle(qtbot, ws, timeout=30_000)
    assert outcomes[0] == ("cancelled", 7_000.0)
    assert outcomes[-1] == ("done", 3_000.0)
    assert ws.last_result().token == ws._window_token


def test_a_hidden_window_computes_nothing_until_it_is_shown(loaded, qtbot, monkeypatch):
    window, _ = loaded
    ws = _open(window, qtbot)
    thread = ws.worker_thread()
    ws.close()
    _pump(10)
    assert ws.worker_thread() is thread and thread.isRunning(), "kept, idle, for the next open"
    assert ws.last_result() is None, "a closed window frees its preview arrays"
    calls = []
    real = cp.conditioned_trace
    monkeypatch.setattr(cp, "conditioned_trace",
                        lambda raw, fs, p: calls.append(1) or real(raw, fs, p))
    window._analysis_panel.detection_panel.set_params({"filter_cutoff_hz": 6_000.0})
    _pump(400)
    assert calls == []
    window._open_signal_conditioning()
    _idle(qtbot, ws)
    assert calls and ws.preview_state() == "exact"


def test_answers_reach_the_gui_thread_and_a_superseded_one_is_dropped(
        loaded, qtbot, monkeypatch):
    """A wrapped handler (a profiler, a debugging shim) wired straight to a worker signal
    ran ON THE WORKER THREAD and drew widgets there; answers are collected on the GUI
    thread from the mailbox, so even a patched handler runs there. An answer to a request
    that was superseded is dropped by its token."""
    window, _ = loaded
    ws = _open(window, qtbot)
    seen = []
    real = SignalConditioningWindow._on_outcome

    def wrapped(self, outcome):
        seen.append(threading.current_thread())
        return real(self, outcome)

    monkeypatch.setattr(SignalConditioningWindow, "_on_outcome", wrapped)
    window._analysis_panel.detection_panel.set_params({"filter_cutoff_hz": 5_500.0})
    _idle(qtbot, ws, timeout=10_000)
    assert seen and all(t is threading.main_thread() for t in seen)
    assert ws.preview_state() == "exact"
    res = ws.last_result()
    stale = dataclasses.replace(res)                     # would be drawn if accepted
    ws._on_outcome(sc._PreviewOutcome(ws._window_token - 1, "window", "ok", stale))
    assert ws.last_result() is res, "a superseded answer is dropped"


# -- the visible window, follow main ----------------------------------------------------------
def test_the_preview_is_the_main_traces_visible_window(loaded, qtbot):
    window, _ = loaded
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 1.0, 1.5)
    _idle(qtbot, ws)
    rng = window._analysis_panel.visible_sample_range()
    res = ws.last_result()
    assert (res.start, res.stop) == rng == ws.current_window()
    assert rng[1] - rng[0] < 300_000 / 4


def test_follow_survives_show_and_resize_and_a_manual_pan_turns_it_off(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    det.set_hp_enabled(True)                # forces Split: two x-linked plots
    ws = _open(window, qtbot)
    assert ws.time_mode() == "split"
    ws.resize(900, 700)
    ws.hide()
    ws.show()
    _idle(qtbot, ws)
    _zoom_main(window, qtbot, 2.0, 2.4)
    _idle(qtbot, ws)
    assert ws.follows_main(), "a resize, a show or a programmatic range never turns it off"
    followed = []
    ws.followMainChanged.connect(followed.append)
    vb = ws.split_processed_plot.getViewBox()
    vb.setXRange(3.0, 3.2, padding=0)
    vb.sigRangeChangedManually.emit([True, False])        # what a drag or wheel emits
    assert not ws.follows_main() and followed == [False]
    a, b = ws.current_window()
    assert abs(a - 3.0 * FS) <= 1 and abs(b - 3.2 * FS) <= 1
    _idle(qtbot, ws)
    assert (ws.last_result().start, ws.last_result().stop) == (a, b)
    _zoom_main(window, qtbot, 0.5, 0.9)
    assert ws.current_window() == (a, b), "no longer following"
    vb.sigRangeChangedManually.emit([False, True])        # a y-only change
    assert ws.current_window() == (a, b)


def test_a_forced_split_keeps_the_range_the_user_panned_to(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 1.0, 3.0)
    _idle(qtbot, ws)
    vb = ws.overlay_plot.getPlotItem().getViewBox()
    vb.setXRange(2.0, 2.1, padding=0)
    vb.sigRangeChangedManually.emit([True, False])
    _idle(qtbot, ws)
    det.set_hp_enabled(True)                 # Split is forced: another page, another view box
    _idle(qtbot, ws)
    assert ws.time_mode() == "split" and not ws.follows_main()
    x0, x1 = ws.split_raw_plot.getViewBox().viewRange()[0]
    assert (x0, x1) == pytest.approx((2.0, 2.1), abs=1e-6)
    assert ws.split_processed_plot.getViewBox().viewRange()[0] == pytest.approx([2.0, 2.1],
                                                                                abs=1e-6)


# -- Time tab ---------------------------------------------------------------------------------
def _de2000():
    path = Path(__file__).resolve().parents[1] / "core" / "test_colour_identity.py"
    spec = importlib.util.spec_from_file_location("_colour_identity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda a, b: mod._de2000(mod._lab(mod._rgb(a)), mod._lab(mod._rgb(b)))


def test_the_time_inks_are_distinct_and_the_legend_names_both(loaded, qtbot):
    window, _ = loaded
    window._analysis_panel.detection_panel.set_params({"filter_cutoff_hz": 8_000.0})
    ws = _open(window, qtbot)
    raw = ws.raw_curve.opts["pen"].color().name()
    proc = ws.processed_curve.opts["pen"].color().name()
    diff = ws.difference_curve.opts["pen"].color().name()
    assert (raw, proc, diff) == ("#8b9199", "#56b4e9", "#cc79a7")
    de = _de2000()
    assert de(raw, proc) >= 15 and de(proc, diff) >= 15 and de(raw, diff) >= 15
    legend = ws.legend_texts()
    assert "raw (calibrated)" in legend and "processed — exact" in legend
    assert ws.raw_curve.yData is not None and ws.processed_curve.yData is not None
    ws.set_time_mode("difference")
    _idle(qtbot, ws)
    y = ws.difference_curve.yData             # the data, not the peak-downsampled view
    res = ws.last_result()
    np.testing.assert_allclose(y, res.raw_window.astype(np.float64) - res.processed_window,
                               atol=1e-3)


def test_normalize_forces_split_with_its_own_unit(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    ws.set_time_mode("difference")
    det.set_normalize_enabled(True)
    _idle(qtbot, ws)
    assert ws.time_mode() == "split" and ws.time_mode_choice() == "difference"
    assert not ws.time_mode_buttons["overlay"].isEnabled()
    assert not ws.time_mode_buttons["difference"].isEnabled()
    axis = ws.split_processed_plot.getAxis("left")
    full = "processed (σ, whole-trace z-score)"
    assert ws.axis_label_text(ws.split_processed_plot, "left") == full == ws.processed_axis_label()
    assert axis.toolTip() == full
    # The full unit when the axis is tall enough, else a shorter one — never clipped.
    assert axis.labelText in (full, "processed (σ)", "σ")
    assert "raw (pA)" in ws.split_raw_plot.getAxis("left").labelText
    det.set_normalize_enabled(False)
    _idle(qtbot, ws)
    assert ws.time_mode() == "difference", "the user's choice comes back"


def test_highpass_forces_split_but_keeps_the_difference_view(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    det.set_hp_enabled(True)
    _idle(qtbot, ws)
    assert ws.time_mode() == "split"
    assert not ws.time_mode_buttons["overlay"].isEnabled()
    assert ws.time_mode_buttons["difference"].isEnabled()
    assert ws.processed_axis_label() == "processed (pA, DC removed by 100 Hz high-pass)"


def test_the_time_axis_labels_fit_their_axes_and_keep_the_full_text(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    det.set_hp_enabled(True)
    full_unit = "processed (pA, DC removed by 100 Hz high-pass)"
    for size in ((ws.MIN_WIDTH, ws.MIN_HEIGHT), (1400, 1300)):
        ws.resize(*size)
        _idle(qtbot, ws)
        _pump(30)
        plot = ws.split_processed_plot
        vb = plot.getViewBox().geometry()
        left, bottom = plot.getAxis("left"), plot.getAxis("bottom")
        assert left.label.boundingRect().width() <= vb.height(), size
        assert bottom.label.boundingRect().width() <= vb.width(), size
        assert left.toolTip() == full_unit
        # The shortest tier drops the word the legend already says, never the "DC".
        assert left.labelText in (full_unit, "processed (pA, no DC)", "pA, no DC"), size
        assert bottom.toolTip().startswith("s (instrument clock")
    assert left.labelText == full_unit, "a tall enough axis shows the whole unit"


def test_the_time_controls_never_reflow(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    widgets = [*ws.time_mode_buttons.values(), ws.exact_button, ws.time_status]

    def geometry():
        return [w.geometry() for w in widgets]

    before = geometry()
    for edit in (lambda: det.set_hp_enabled(True), lambda: det.set_normalize_enabled(True),
                 lambda: det.set_stage_enabled("detrend", True), lambda: ws.compute_exact(),
                 lambda: det.set_params({"filter_cutoff_hz": 40_000.0})):
        edit()
        _idle(qtbot, ws, timeout=30_000)
        assert geometry() == before


# -- window estimate versus exact -----------------------------------------------------------
def test_estimators_are_a_window_estimate_until_compute_exact(loaded, qtbot, monkeypatch):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 1.0, 2.0)
    full_calls = []
    real = cp.conditioned_trace
    monkeypatch.setattr(cp, "conditioned_trace", lambda raw, fs, p: (
        full_calls.append(raw.size) if raw.size == 300_000 else None) or real(raw, fs, p))
    det.set_stage_enabled("detrend", True)
    _idle(qtbot, ws)
    _pump(800)
    assert full_calls == [], "the whole-recording pass is never automatic"
    assert ws.preview_state() == "window"
    assert ws.time_status.text().startswith("Window estimate — Baseline drift uses "
                                            "whole-recording statistics")
    assert "processed — window estimate" in ws.legend_texts()
    assert ws.exact_button.isEnabled()
    width = ws.exact_button.width()
    ws.exact_button.click()
    assert ws.exact_pending() and not ws.exact_button.isEnabled()
    # What is happening now leads the status (and is what is painted, not only the tip),
    # and the button says it too, at the same width.
    assert ws.time_status.text().startswith("Computing exact (whole recording)… · Window "
                                            "estimate — Baseline drift")
    assert QtWidgets.QLabel.text(ws.time_status).startswith("Computing exact")
    assert ws.exact_button.text() == "Computing…" and ws.exact_button.width() == width
    _until(lambda: not ws.exact_pending(), 30_000)
    _idle(qtbot, ws)
    assert full_calls == [300_000]
    assert ws.preview_state() == "exact" and ws.last_result().from_exact_pass
    assert ws.time_status.text().startswith("Whole recording (exact)")
    assert "processed — exact" in ws.legend_texts()
    assert not ws.exact_button.isEnabled()
    res = ws.last_result()
    exact = real(window._session.calibrated_array(*window._analysis_panel._sel), FS, ws._chain)
    np.testing.assert_array_equal(res.processed_window, exact[res.start:res.stop])
    det.set_stage_enabled("detrend", False)
    _idle(qtbot, ws)
    det.set_stage_enabled("detrend", True)
    _idle(qtbot, ws)
    assert ws.last_result().from_exact_pass, "the memo serves the same chain again"
    assert full_calls == [300_000]


def test_a_stage_settling_past_the_pad_is_an_estimate_named_on_two_fixed_lines(
        loaded, qtbot, monkeypatch):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 2.0, 2.4)
    _idle(qtbot, ws)
    assert ws.exact_button.toolTip().startswith("Not needed"), "nothing on: exact already"
    height = ws.time_status.height()
    monkeypatch.setattr(cp, "MAX_PREVIEW_PAD_SAMPLES", 10_000)        # a 0.2 s pad limit
    det.set_params({"hp_enabled": True, "hp_cutoff_hz": 20.0})         # settles over 2 s
    _idle(qtbot, ws)
    assert ws.preview_state() == "window"
    text = ws.time_status.text()
    assert text.startswith("Window estimate — High-pass settles over 2 s, longer than the "
                           "±0.2 s padding"), text
    assert "whole-recording statistics" not in text
    assert ws.exact_button.isEnabled() and ws.exact_button.toolTip().startswith("Condition")
    shown = QtWidgets.QLabel.text(ws.time_status)        # what is painted, after eliding
    assert shown.startswith("Window estimate — High-pass") and shown.count("\n") <= 2
    assert ws.time_status.height() == height, "a longer status never moves the plot"
    # A stage whose settling prints like the pad (a 1 Hz low-pass: 2,000,001 samples
    # against 2,000,000) says "more than", never "over 40 s, longer than the ±40 s".
    assert sc._settles_text("Low-pass", 40.00002, 39.99999) == (
        "Low-pass settles over more than the ±40 s padding")
    assert sc._settles_text("High-pass", 400.0, 40.0) == (
        "High-pass settles over 400 s, longer than the ±40 s padding")


def test_every_terminal_path_clears_a_pending_exact_pass(loaded, qtbot, monkeypatch):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 1.0, 2.0)
    det.set_normalize_enabled(True)
    _idle(qtbot, ws)
    real = cp.conditioned_trace
    gate = threading.Event()

    def blocking(raw, fs, params):
        if raw.size == 300_000:
            while not gate.is_set():
                check_cancelled()
                time.sleep(0.005)
        return real(raw, fs, params)

    monkeypatch.setattr(cp, "conditioned_trace", blocking)
    # 1. cancelled by a chain change
    assert ws.compute_exact() and ws.exact_pending()
    det.set_params({"filter_cutoff_hz": 6_000.0})
    assert not ws.exact_pending()
    _idle(qtbot, ws)
    assert ws.preview_state() == "window" and ws.exact_button.isEnabled()
    assert "Computing exact" not in ws.time_status.text()
    assert ws.exact_button.text() == "Compute exact"
    # 2. cancelled by closing the window, whose thread is kept
    thread = ws.worker_thread()
    assert ws.compute_exact()
    ws.close()
    assert not ws.exact_pending() and ws.worker_thread() is thread
    gate.set()
    window._open_signal_conditioning()
    _idle(qtbot, ws)
    assert ws.exact_button.isEnabled()
    # 3. failed: said first, and the button offers the retry
    gate.clear()

    def failing(raw, fs, params):
        if raw.size == 300_000:
            raise MemoryError("not enough memory for the whole recording")
        return real(raw, fs, params)

    monkeypatch.setattr(cp, "conditioned_trace", failing)
    assert ws.compute_exact()
    _until(lambda: not ws.exact_pending(), 10_000)
    _idle(qtbot, ws)
    assert ws.time_status.text().startswith(
        "Exact pass failed: not enough memory for the whole recording · Window estimate")
    assert QtWidgets.QLabel.text(ws.time_status).startswith("Exact pass failed")
    assert ws.exact_button.text() == "Retry exact" and ws.exact_button.isEnabled()
    assert ws.preview_state() == "window"
    # 4. the failure belongs to that chain: another chain is not reported as failed
    det.set_params({"filter_cutoff_hz": 5_000.0})
    _idle(qtbot, ws)
    assert "failed" not in ws.time_status.text()
    assert ws.exact_button.text() == "Compute exact"


def _blocking_kernel(monkeypatch, gate, *, only_size=None):
    real = cp.conditioned_trace

    def blocking(raw, fs, params):
        if only_size is None or raw.size == only_size:
            while not gate.is_set():
                check_cancelled()
                time.sleep(0.005)
        return real(raw, fs, params)

    monkeypatch.setattr(cp, "conditioned_trace", blocking)


def test_a_pending_preview_never_shows_the_previous_chains_verdict(loaded, qtbot, monkeypatch):
    """From an exact chain to an estimator chain behind a slow kernel: while the new
    preview is pending the status leads with "Updating…" and gives the NEW chain's verdict,
    and the legend calls the curve still shown the previous chain's."""
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    det.set_params({"filter_cutoff_hz": 10_000.0})
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 1.0, 2.0)
    _idle(qtbot, ws)
    assert ws.time_status.text().startswith("Exact for these stages · window 1.00 s")
    assert "processed — exact" in ws.legend_texts()
    gate = threading.Event()
    _blocking_kernel(monkeypatch, gate)
    det.set_params({"detrend_enabled": True, "normalize_enabled": True})
    _pump(5)
    seen = 0
    deadline = time.monotonic() + 0.8
    while time.monotonic() < deadline:
        assert ws.preview_state() == "pending"
        text = ws.time_status.text()
        assert text.startswith("Updating… · Window estimate — Baseline drift/Normalize use "
                               "whole-recording statistics"), text
        assert "Exact for these stages" not in text
        assert "processed — previous chain" in ws.legend_texts()
        assert "processed — exact" not in ws.legend_texts()
        seen += 1
        _pump(50)
    assert seen >= 5
    gate.set()
    _idle(qtbot, ws)
    assert ws.time_status.text().startswith("Window estimate — Baseline drift/Normalize")
    assert "processed — window estimate" in ws.legend_texts()
    assert "processed — previous chain" not in ws.legend_texts()


def test_the_response_views_never_wait_behind_the_exact_pass(loaded, qtbot, monkeypatch):
    """Phase / group delay / a short step are computed on the GUI thread (a few ms), so a
    seconds-long exact pass on the worker does not hold them; only a slow stage's step is
    queued, and then it says what it waits for."""
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    det.set_params({"normalize_enabled": True})
    det.filter_cutoff_hz.setValue(10_000.0)              # the low-pass stays OFF: a design
    ws = _open(window, qtbot)
    gate = threading.Event()
    _blocking_kernel(monkeypatch, gate, only_size=300_000)
    assert ws.compute_exact()
    _pump(60)
    ws.set_preview_tab("response")
    ws.set_response_view("step")
    _pump(ws.DEBOUNCE_MS + 100)
    widgets = ws.response_view_widgets("step")
    assert widgets["applied"].getData()[0] is not None
    assert widgets["note"].text().startswith("Applied: 10–90 % rise")
    ws.set_response_view("group_delay")
    _pump(ws.DEBOUNCE_MS + 100)
    assert ws.response_view_widgets("group_delay")["applied"].getData()[0] is not None
    assert ws.exact_pending(), "the exact pass is still running"
    ws.set_response_view("step")
    det.filter_cutoff_hz.setValue(1.0)                   # a 1 Hz design: ~800k samples
    _pump(ws.DEBOUNCE_MS + 100)
    assert ws.exact_pending()
    assert widgets["note"].text().startswith("Waiting for the exact pass to finish")
    gate.set()
    _until(lambda: not ws.exact_pending() and ws.preview_state() != "pending", 30_000)
    _pump(20)
    assert widgets["note"].text().startswith("Applied: 10–90 % rise")
    t, _y = widgets["applied"].getData()
    assert t[-1] - t[0] > 1.0, "the slow design's step, drawn once the worker is free"


# -- Spectrum ---------------------------------------------------------------------------------
def test_spectrum_lines_go_to_the_notch_through_the_panel(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot, tab="spectrum")
    texts = ws.spectral_line_texts()
    assert any(t.startswith("4570.8 Hz") for t in texts), texts
    assert "Δf" in ws.spectrum_note.text()
    row = next(i for i, t in enumerate(texts) if t.startswith("4570.8 Hz"))
    ws.lines_list.item(row).setCheckState(QtCore.Qt.Checked)
    ws.send_to_notch_button.click()
    params = det.params()
    assert params["notch_enabled"] and params["notch_freqs"] == [4570.8]
    assert window.statusBar().currentMessage().startswith("Notch: added 4570.8 Hz")
    assert ws.lines_note.text().startswith("Notch: added 4570.8 Hz")
    _idle(qtbot, ws)
    assert any(t.startswith("4570.8 Hz") and "notched" in t for t in ws.spectral_line_texts())
    res = ws.last_result()
    band = (res.spectrum.f_hz > 4560) & (res.spectrum.f_hz < 4580)
    assert res.spectrum.processed_psd[band].max() < 0.1 * res.spectrum.raw_psd[band].max()
    # Over the cap: refused, the box untouched.
    det.set_notch_lines([100.0 * (k + 1) for k in range(32)])
    before = det.params()["notch_freqs"]
    window._add_notch_lines_from_spectrum([4570.8 + 300.0])
    assert det.params()["notch_freqs"] == before
    assert "at most 32" in window.statusBar().currentMessage()
    assert QtWidgets.QLabel.text(ws.lines_note).startswith("Refused: the notch takes at most 32")


def test_return_sends_the_current_line_and_an_off_notch_keeps_its_list(loaded, qtbot):
    """Return sends the current row (macOS item views give Return to editing, never to
    activated), and an off notch's own list is kept, the notch left off — said in a
    two-line note of its own, consequence first."""
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    kept = [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0]
    det.set_notch_lines(kept)
    det.set_notch_enabled(False)
    ws = _open(window, qtbot, tab="spectrum")
    row = next(i for i, t in enumerate(ws.spectral_line_texts()) if t.startswith("4570.8 Hz"))
    ws.lines_list.setCurrentRow(row)
    QtTest.QTest.keyClick(ws.lines_list, QtCore.Qt.Key_Return)
    params = det.params()
    assert params["notch_freqs"] == kept + [4570.8] and not params["notch_enabled"]
    note = ws.lines_note
    assert note.text().startswith("Notch still OFF: added 4570.8 Hz to the list it keeps")
    painted = QtWidgets.QLabel.text(note)
    assert painted.startswith("Notch still OFF: added 4570.8 Hz") and painted.count("\n") == 1
    button = ws.send_to_notch_button
    assert note.mapTo(ws, QtCore.QPoint(0, 0)).y() >= button.mapTo(
        ws, QtCore.QPoint(0, button.height())).y(), "below the button, not beside it"
    assert note.width() >= ws.lines_list.width() - 2
    QtTest.QTest.keyClick(ws.lines_list, QtCore.Qt.Key_Enter, QtCore.Qt.KeypadModifier)
    assert det.params()["notch_freqs"] == kept + [4570.8], "already in the kept list"
    assert note.text().startswith("Notch still OFF: 4570.8 Hz is already in the list")


# -- Response ---------------------------------------------------------------------------------
def test_the_response_views_draw_phase_group_delay_and_step(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    det.set_params({"filter_cutoff_hz": 10_000.0, "median_enabled": True})
    ws = _open(window, qtbot, tab="response")
    det.response_chain_check.setChecked(True)
    for view in ("phase", "group_delay", "step"):
        ws.set_response_view(view)
        _idle(qtbot, ws)
        widgets = ws.response_view_widgets(view)
        assert ws.response_stack.currentIndex() == ("magnitude", "phase", "group_delay",
                                                    "step").index(view)
        x, y = widgets["applied"].getData()
        assert x is not None and len(x) > 10, view
        assert widgets["reference"].getData()[0] is not None
        note = widgets["note"].text()
        assert "Not representable as a linear response: Median" in note, note
        assert widgets["title"].text() == "Enabled linear chain — Low-pass"
    assert "rise" in ws.response_view_widgets("step")["note"].text()
    det.response_chain_check.setChecked(False)
    det.select_stage("highpass")
    _idle(qtbot, ws)
    assert ws.response_view_widgets("step")["title"].text() == "High-pass — OFF · design preview"
    ws.set_response_view("magnitude")
    assert det.response_plot.isVisibleTo(ws)


# -- settings, lifetime ---------------------------------------------------------------------
def test_the_view_choices_persist_and_bad_values_fall_back(loaded, qtbot):
    window, _ = loaded
    ws = _open(window, qtbot)
    ws.set_preview_tab("spectrum")
    ws.set_time_mode("split")
    ws.set_response_view("step")
    window._save_settings()
    s = QtCore.QSettings()
    assert (s.value("signalConditioning/tab"), s.value("signalConditioning/timeMode"),
            s.value("signalConditioning/responseView")) == ("spectrum", "split", "step")
    ws.set_preview_tab("time")
    ws.set_time_mode("overlay")
    ws.set_response_view("magnitude")
    window._place_signal_conditioning(ws)          # what a first open reads
    assert (ws.preview_tab(), ws.time_mode_choice(), ws.response_view()) == (
        "spectrum", "split", "step")
    s.setValue("signalConditioning/tab", "spectrogram")
    s.setValue("signalConditioning/timeMode", 3)
    ws.set_preview_tab("time")
    ws.set_time_mode("overlay")
    window._place_signal_conditioning(ws)
    assert ws.preview_tab() == "time" and ws.time_mode_choice() == "overlay"
    message = window.statusBar().currentMessage()
    assert "signalConditioning/tab" in message and "signalConditioning/timeMode" in message


def test_a_closed_document_leaves_the_memo_and_the_window(loaded, shared, qtbot):
    window, did = loaded
    det = window._analysis_panel.detection_panel
    ws = _open(window, qtbot)
    _zoom_main(window, qtbot, 1.0, 2.0)
    det.set_normalize_enabled(True)
    _idle(qtbot, ws)
    ws.compute_exact()
    _until(lambda: not ws.exact_pending(), 30_000)
    assert len(ws._memo) == 1
    try:
        window._session.close_document(did)
        _pump(20)
        assert len(ws._memo) == 0 and ws._source is None
        assert ws.time_status.text().startswith("No recording selected")
        assert ws.spectrum_note.text() == "No recording selected."
        assert ws.lines_note.text() == ""
    finally:                                        # the shared window gets a recording back
        new = window._session.add_document(_doc(shared["tmp"], name="b.dat"))
        window._create_default_layers(window._session.document(new))
        window._on_selection(new, 0)
        shared["did"] = new
        _pump(20)


def test_main_window_teardown_leaves_no_preview_thread(loaded, qtbot):
    """LAST to use the shared window: its teardown is final."""
    window, _ = loaded
    ws = _open(window, qtbot)
    thread = ws.worker_thread()
    assert thread in wt.unstopped_worker_threads()
    ws.close()
    assert ws.worker_thread() is thread, "a closed window keeps its one thread"
    assert window._teardown() == 0
    assert ws.worker_thread() is None
    assert thread not in wt.unstopped_worker_threads()
    window._open_signal_conditioning()
    _pump(ws.DEBOUNCE_MS + 50)
    assert ws.worker_thread() is None, "nothing restarts it after the teardown"


def test_the_window_class_reaches_only_the_allowed_gui_seams():
    import ast
    import inspect

    import nanoscope.gui.signal_conditioning as mod

    tree = ast.parse(inspect.getsource(mod))
    gui = {n.module for n in ast.walk(tree)
           if isinstance(n, ast.ImportFrom) and n.level == 1}
    assert gui == {"axes", "trace_view", "worker_thread"}
    assert SignalConditioningWindow.DEBOUNCE_MS == 250
