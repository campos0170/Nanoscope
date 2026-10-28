"""S8-NUM1 — Signal Conditioning's cache of "Compute exact" passes served the pass computed at the
OLD sample rate as the exact trace of the NEW one.

MEASURED DEFECT (sweep 8, the -600 mV apo-ferritin ABF). Signal Conditioning open on the 30–40 s
view, chain = baseline drift (block median, 0.5 s, blockade polarity) + 5 kHz Bessel low-pass,
Compute exact (the pass is cached); Inspector ▸ File Info ▸ Sample rate / time unit… ▸ 25,000 Hz
(the file said 50,000); Signal Conditioning hidden and opened again. The window now held the
25 kHz source and chain, found the pass computed at 50 kHz under the same cache key and showed it:
the status line read "Whole recording (exact)", the Compute exact button was disabled with "Done:
the preview shows this chain's whole-recording (exact) samples", and the trace drawn was 26 pA from
what the chain makes of the same samples at the declared rate (noise sigma 4.5 pA) — to the last
digit the 50 kHz pass.

THE FIX. The cache key (``conditioning_preview.memo_key``) named no rate: ``fs`` went into the
chain's signature, which uses it only to prune a notch's lines at Nyquist, so a baseline drift or a
low-pass is "the same chain" at 50 kHz and at 25 kHz. S8-SC3 (already in the base) gave
``ConditioningSource.key`` the rate, which closed this route through the window; the key of the
memo still depended on its caller doing so. It now carries the rate itself, so a pass belongs to the
rate it was computed at whoever builds the source key. The Qt-free half is
tests/analysis/test_s8_num1_exact_pass_memo_key_names_the_rate.py.

These tests drive the real MainWindow, the real SampleRateDialog (the one File Info opens), the
real window (hide it, Analysis ▸ Signal conditioning… again) and the Session's calibration, on a
synthetic recording, and read what the user reads: the status line, the Compute exact button and
its tip, the legend, and the samples drawn — compared with what ``conditioned_trace`` makes of the
whole calibrated channel at the rate in force. One test per route to the finding, one for taking
the declaration away again (each rate keeps its own pass, and the first is found again), one for the
calibration edit that already dropped the pass (kept), and one for a pass still running when the
window is closed. The tests of the routes S8-SC3 closed pass on the base by design; the key-level
half of this fix is what failed before it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import replace

import nanoscope  # noqa: F401  -> Qt plugin bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.conditioning import conditioned_trace, conditioning_params
from nanoscope.analysis.conditioning_recipes import complete_recipe
from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import signal_conditioning as sc
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.sample_rate_dialog import SampleRateDialog

FS = 50_000.0                   # what the file's header says (and what it was acquired at)
DECLARED = 25_000.0             # what the user declares in the finding
N = 200_000                     # 4 s at 50 kHz, 8 s once declared at 25 kHz
WINDOW_ESTIMATE = "Window estimate — Baseline drift uses whole-recording"
WHOLE_RECORDING = "Whole recording (exact)"


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)


# ================================================================================================
# the recording and the bench (ONE MainWindow for the module, as the neighbouring files do)
# ================================================================================================
def _recording(tmp, name: str = "header_rate.dat") -> TraceDocument:
    """A recording whose HEADER declares 50 kHz (an ABF's shape: only a rate can be declared on
    it). A noisy -1000 pA baseline, a slow 0.5 Hz drift (so a baseline-drift pass depends on the
    clock), a 4570.8 Hz tone and a blockade every 30,000 samples."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / FS
    x = np.full(N, -1000.0) + rng.normal(0.0, 8.0, N) + 5.0 * np.sin(2 * np.pi * 4570.8 * t)
    x += 30.0 * np.sin(2 * np.pi * 0.5 * t)
    for s in range(10_000, N, 30_000):
        x[s:s + 300] += 300.0
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(x.astype(np.float32),))


def _chain() -> dict:
    """The finding's chain."""
    return {"filter_cutoff_hz": 5_000.0, "detrend_enabled": True,
            "detrend_method": "block_median", "detrend_window_s": 0.5, "polarity": -1}


def _pump(ms: int) -> None:
    deadline = time.monotonic() + ms / 1000.0
    while True:
        QtWidgets.QApplication.processEvents()
        if time.monotonic() >= deadline:
            return
        time.sleep(0.003)


def _until(predicate, timeout: int = 20_000) -> None:
    deadline = time.monotonic() + timeout / 1000.0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout} ms")
        QtWidgets.QApplication.processEvents()
        time.sleep(0.003)


def _settle(ws, timeout: int = 30_000) -> None:
    """The window has nothing scheduled, posted, running or coming back."""
    _until(lambda: ws.preview_state() != "pending" and not ws._debounce.isActive()
           and not ws.exact_pending() and ws.worker_idle(), timeout)
    _pump(5)


@pytest.fixture(scope="module")
def rig(qapp, tmp_path_factory):
    import nanoscope.gui.main_window as mw

    register_all()
    window = MainWindow()
    window._confirm_close_choice = lambda: mw._CLOSE_DISCARD
    window.resize(1512, 875)
    window.show()
    _pump(20)
    yield {"window": window, "tmp": tmp_path_factory.mktemp("s8_num1")}
    if window._signal_conditioning is not None:
        window._signal_conditioning.abort()
    window._analysis_controller.abort()
    window.close()
    window.deleteLater()
    _pump(10)


class Bench:
    """The shared window with a fresh form, and a few verbs for what the user does."""

    def __init__(self, window: MainWindow, tmp) -> None:
        self.window, self.tmp = window, tmp
        self.session = window._session
        self.det = window._analysis_panel.detection_panel

    def open_recording(self):
        """Open the recording, select its channel, give the form the finding's chain, look at
        part of it (1–2 s) and open Signal Conditioning on it."""
        did = self.session.add_document(_recording(self.tmp))
        self.window._create_default_layers(self.session.document(did))
        self.window._on_selection(did, 0)
        _pump(20)
        self.det.set_params(_chain())
        self.window._trace_view.zoom_x_to(1.0, 2.0)
        _pump(30)
        return did, self.open_window()

    def open_window(self):
        """Analysis ▸ Signal conditioning… (⇧⌘F), however many times."""
        self.window._open_signal_conditioning()
        ws = self.window._signal_conditioning
        ws.set_preview_tab("time")
        _settle(ws)
        return ws

    def look_at(self, start_s: float, stop_s: float) -> None:
        """Pan the main trace (the window follows it): a part of the recording, not all of it."""
        self.window._trace_view.zoom_x_to(start_s, stop_s)
        _pump(30)

    def declare(self, did: str, rate: float | None = None) -> str:
        """Inspector ▸ File Info ▸ Sample rate / time unit… ▸ ``rate`` Hz (or, with no rate, "Use
        the file's own time base"); Apply. The real dialog. Returns what it says."""
        dialog = SampleRateDialog(self.session, did, self.window)
        try:
            if rate is not None:
                dialog.mode.setCurrentIndex(dialog.mode.findData("sample_rate"))
                dialog.rate.setText(f"{rate:g}")
            else:
                dialog.mode.setCurrentIndex(dialog.mode.findData("file"))
            dialog.apply()
            return dialog.error.text()
        finally:
            dialog.close()
            dialog.deleteLater()

    def compute_exact(self, ws) -> None:
        assert ws.compute_exact(), "an estimator stage on part of the recording offers the pass"
        _until(lambda: not ws.exact_pending(), 60_000)
        _settle(ws)

    def truth(self, did: str):
        """What the chain really makes of the whole calibrated channel at the rate in force."""
        raw = np.asarray(self.session.calibrated(did, 0).data)
        rate = float(self.session.document(did).meta.sample_rate_hz)
        return np.asarray(conditioned_trace(raw, rate, conditioning_params(self.det.params())),
                          dtype=np.float64)


@pytest.fixture
def bench(rig, monkeypatch):
    window = rig["window"]
    bench = Bench(window, rig["tmp"])
    # PySide6 reports an exception raised inside a slot through sys.excepthook and carries
    # on, so a slot that fails would not fail a test by itself.
    raised: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: raised.append(exc))
    bench.det.set_params(complete_recipe({}))
    bench.det.set_notch_lines(None)
    bench.det.response_chain_check.setChecked(False)
    bench.det.select_stage("lowpass")
    ws = window._signal_conditioning
    if ws is not None:
        ws.close()
        ws._memo.clear()
        ws.set_follow_main(True)
        ws.set_preview_tab("time")
    yield bench
    if window._signal_conditioning is not None:
        window._signal_conditioning.close()
    for doc in list(bench.session.documents()):
        bench.session.close_document(doc.doc_id)
    _pump(20)
    assert raised == [], f"a slot raised: {raised[:1]}"


def _shown(ws) -> np.ndarray:
    return np.asarray(ws.last_result().processed_window, dtype=np.float64)


def _says_press_compute_exact(ws) -> None:
    """Everything the user reads says this chain has no exact pass at this rate yet."""
    text = ws.time_status.text()
    assert text.startswith(WINDOW_ESTIMATE), text
    assert "press Compute exact for the true trace" in text and "(exact)" not in text
    assert ws.exact_button.isEnabled() and ws.exact_button.text() == "Compute exact"
    assert ws.exact_button.toolTip().startswith("Condition the WHOLE recording"), (
        ws.exact_button.toolTip())
    assert "processed — window estimate" in ws.legend_texts()
    assert not ws.last_result().exact and not ws.last_result().from_exact_pass


def _says_exact_pass(ws) -> None:
    text = ws.time_status.text()
    assert text.startswith(WHOLE_RECORDING), text
    assert not ws.exact_button.isEnabled()
    assert ws.exact_button.toolTip() == (
        "Done: the preview shows this chain's whole-recording (exact) samples.")
    assert "processed — exact" in ws.legend_texts()
    assert ws.last_result().from_exact_pass and ws.last_result().exact


# ================================================================================================
# 1. the finding: Compute exact, declare, hide the window, open it again
# ================================================================================================
def test_a_reopened_window_does_not_show_the_old_rates_pass_as_exact(bench):
    did, ws = bench.open_recording()
    assert ws.time_status.text().startswith(WINDOW_ESTIMATE)
    bench.compute_exact(ws)
    _says_exact_pass(ws)
    old_key = ws._current_memo_key()
    old_pass = np.array(ws._memo.get(old_key)[0], dtype=np.float64)

    ws.close()                                           # hide it: the pass stays in the cache
    said = bench.declare(did, DECLARED)
    assert said.startswith("Declared: 25,000 Hz."), said
    bench.look_at(2.0, 4.0)                              # part of the 8 s recording
    ws = bench.open_window()                             # Analysis ▸ Signal conditioning… again

    truth = bench.truth(did)
    assert ws._source.fs == pytest.approx(DECLARED) and ws._fs == pytest.approx(DECLARED)
    assert np.abs(old_pass - truth).max() > 1.0, "the two clocks really condition differently"
    _says_press_compute_exact(ws)
    res = ws.last_result()
    assert np.abs(_shown(ws) - old_pass[res.start:res.stop]).max() > 1e-3, (
        "the 50 kHz pass is not what is drawn")
    assert ws._current_memo_key() != old_key and ws._memo.get(ws._current_memo_key()) is None

    bench.compute_exact(ws)                              # the button the status asked for
    _says_exact_pass(ws)
    res = ws.last_result()
    np.testing.assert_allclose(_shown(ws), truth[res.start:res.stop], rtol=0, atol=1e-3)
    assert len(ws._memo) == 2, "the new pass sits beside the old one, not over it"


# ================================================================================================
# 2. the same declaration with the window open
# ================================================================================================
def test_an_open_window_does_not_show_the_old_rates_pass_as_exact_either(bench):
    did, ws = bench.open_recording()
    bench.compute_exact(ws)
    _says_exact_pass(ws)
    old_pass = np.array(ws._memo.get(ws._current_memo_key())[0], dtype=np.float64)

    bench.declare(did, DECLARED)
    bench.look_at(2.0, 4.0)
    _settle(ws)

    truth = bench.truth(did)
    assert np.abs(old_pass - truth).max() > 1.0
    _says_press_compute_exact(ws)
    bench.compute_exact(ws)
    _says_exact_pass(ws)
    res = ws.last_result()
    np.testing.assert_allclose(_shown(ws), truth[res.start:res.stop], rtol=0, atol=1e-3)


# ================================================================================================
# 3. taking the declaration away again: each rate keeps its own pass
# ================================================================================================
def test_each_rate_keeps_its_own_pass_and_the_first_is_found_again(bench):
    did, ws = bench.open_recording()
    bench.compute_exact(ws)                              # the pass at the file's 50 kHz
    truth_50 = bench.truth(did)

    bench.declare(did, DECLARED)
    bench.look_at(2.0, 4.0)
    _settle(ws)
    _says_press_compute_exact(ws)
    bench.compute_exact(ws)                              # the pass at the declared 25 kHz
    truth_25 = bench.truth(did)
    assert len(ws._memo) == 2
    assert np.abs(truth_50 - truth_25).max() > 1.0

    assert bench.declare(did).startswith("The file's own time base is in use again.")
    bench.look_at(1.0, 2.0)
    _settle(ws)
    assert ws._source.fs == pytest.approx(FS)
    _says_exact_pass(ws)                                 # served, not recomputed: it IS the 50 kHz pass
    res = ws.last_result()
    np.testing.assert_allclose(_shown(ws), truth_50[res.start:res.stop], rtol=0, atol=1e-3)

    bench.declare(did, DECLARED)
    bench.look_at(2.0, 4.0)
    _settle(ws)
    _says_exact_pass(ws)                                 # and the 25 kHz one is still its own
    res = ws.last_result()
    np.testing.assert_allclose(_shown(ws), truth_25[res.start:res.stop], rtol=0, atol=1e-3)
    assert len(ws._memo) == 2, "two slots, as before"


# ================================================================================================
# 4. what already invalidated the pass still does
# ================================================================================================
def test_a_calibration_edit_still_drops_the_pass_and_putting_it_back_finds_it(bench):
    did, ws = bench.open_recording()
    bench.compute_exact(ws)
    _says_exact_pass(ws)
    key = ws._current_memo_key()
    original = bench.session.calibration(did, 0)

    bench.session.set_calibration(did, 0, replace(original, gain=2.0), write_sidecar=False)
    _settle(ws)
    assert ws._current_memo_key() != key
    _says_press_compute_exact(ws)

    bench.session.set_calibration(did, 0, original, write_sidecar=False)
    _settle(ws)
    assert ws._current_memo_key() == key
    _says_exact_pass(ws)


# ================================================================================================
# 5. a pass still running when the window is closed
# ================================================================================================
def test_a_pass_still_running_at_the_old_rate_when_the_window_is_closed_is_never_filed(
        bench, monkeypatch):
    did, ws = bench.open_recording()
    real = sc.compute_exact
    entered, gate = threading.Event(), threading.Event()

    def slow(req):
        entered.set()
        assert gate.wait(30), "the test never released the worker"
        return real(req)

    monkeypatch.setattr(sc, "compute_exact", slow)
    assert ws.compute_exact()
    _until(entered.is_set, 10_000)

    ws.close()                                           # the X button, mid-pass
    bench.declare(did, DECLARED)
    gate.set()                                           # the 50 kHz pass finishes now
    _until(ws.worker_idle, 60_000)
    _pump(30)
    assert len(ws._memo) == 0, "a pass at 50 kHz was not filed for anything"

    bench.look_at(2.0, 4.0)
    ws = bench.open_window()
    _says_press_compute_exact(ws)
    assert not ws.exact_pending() and "Computing exact" not in ws.time_status.text()
