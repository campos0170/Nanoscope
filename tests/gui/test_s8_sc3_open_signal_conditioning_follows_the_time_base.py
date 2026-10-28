"""S8-SC3 — A time base declared in File Info never reaches an open Signal Conditioning window.

MEASURED DEFECT (sweep 8; the finder and two skeptics, on a CSV whose bare ``time`` column is in
milliseconds, on the owner's -600 mV ABF re-declared at 25 kHz and on a neutral-header CSV
declared at 100 kHz). With Signal Conditioning (⇧⌘F) open, Inspector ▸ File Info ▸ Sample rate /
time unit… moved the status bar, File Info, the Detection form and the main trace to the declared
clock — and not the window. It went on previewing at the OLD rate: the Time tab spanned 0–2000 s
against the trace's 0–2 s under "Exact for these stages · window 2000.00 s", the Spectrum tab
stopped at the old Nyquist (25 Hz), and when the user then turned on the 10 kHz low-pass the form
now accepted, the preview refused it as "at or above Nyquist (25 Hz) for fs=50 Hz" — the very
units error the user had just corrected. Nothing on the window said it was stale. Handing the
time base back to the file left it the other way round (window at 50 kHz, recording at 50 Hz). A
detection run after a declaration pushed the chain at the NEW rate into a window whose source was
still at the old one, so it held two rates at once under one "Exact" verdict. Closing and
reopening the window healed it; a chain edit did not. Reopening it also served the whole-
recording ("Compute exact") pass computed at the old rate as the exact trace of the new one
(S8-NUM1: 26 pA off a 4.5 pA noise sigma).

THE FIX is three small things. ``MainWindow._wire_signal_conditioning`` now listens to
``Session.sampleRateChanged`` (``_follow_conditioning_time_base``): when the window is on screen
and the declared recording is the one it shows, it re-pushes the source AND the chain together
through ``_push_conditioning_source`` — the door a recording switch and a calibration edit
already use — so the two rates cannot disagree; a hidden window is brought up to date by its
opener, as before. ``ConditioningSource.key`` (``analysis/conditioning_preview.py``) ends with the
rate, so ``SignalConditioningWindow.set_source`` no longer mistakes a re-read recording for the
same samples: it cancels the exact pass, clears the old answers and reschedules; an answer still
coming back from the worker at the old rate is discarded by the same key; and the exact-pass
memo, keyed by it, cannot answer for another rate. And a Time plot the user had panned (Follow
off) is re-framed on the new clock instead of keeping a range in the old clock's seconds.

These tests drive the real MainWindow, the real SampleRateDialog (the one File Info opens) and
the Session call it makes, and read what the user reads: the status line, the stage rows, the
curves' x data, the Spectrum note and its frequency axis, the legends and the Compute exact
button. Each facet of the finding has its own test, and one test holds the whole window to the
standard the finding itself offered as a remedy: after a declaration, and after taking it away,
the window that stayed open shows exactly what a window opened afterwards shows. The Qt-free
half (the key and the memo) is
tests/analysis/test_s8_sc3_conditioning_source_key_carries_the_rate.py.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

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

FS = 50_000.0                   # what both recordings really were acquired at
N_CSV = 100_000                 # 2 s at 50 kHz — the CSV the finding's user opened
N_HEADER = 200_000              # 4 s at 50 kHz — a recording whose header declares its rate
T0_AS_READ = 10.0               # the CSV's first time stamp, in the (wrong) unit it was read in
TIMING = re.compile(r" · \d+ ms$")      # the pass time the verdict ends with


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)


# ================================================================================================
# the recordings
# ================================================================================================
def _signal(n: int, seed: int = 0) -> np.ndarray:
    """A noisy -1000 pA baseline with a 4570.8 Hz tone (a line the Spectrum tab can find), a
    slow 0.5 Hz drift (so a baseline-drift pass depends on the clock) and a blockade every
    30,000 samples. Samples only: what clock they sit on is the recording's business."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = np.full(n, -1000.0) + rng.normal(0.0, 8.0, n) + 5.0 * np.sin(2 * np.pi * 4570.8 * t)
    x += 30.0 * np.sin(2 * np.pi * 0.5 * t)
    for s in range(10_000, n, 30_000):
        x[s:s + 300] += 300.0
    return x.astype(np.float32)


def _csv_in_milliseconds(tmp, name: str = "ms_column.csv") -> TraceDocument:
    """A 'time,current' export whose bare time column is in MILLISECONDS: 50 kHz data whose
    0.02-per-row steps read as seconds, i.e. 50 Hz, ``Provenance.MEASURED`` — the recording
    the new Issue talks about and the one a time-unit declaration exists for."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp / name, format_name="Text trace", n_samples=N_CSV,
                         sample_rate_hz=50.0, rate_provenance=Provenance.MEASURED,
                         channels=(info,), t0_s=T0_AS_READ)
    return TraceDocument(meta=meta, channels=(_signal(N_CSV),))


def _recording_with_a_header_rate(tmp, name: str = "header_rate.dat") -> TraceDocument:
    """A recording whose HEADER declares 50 kHz (an ABF's shape): only a sample rate can be
    declared on it, never a time unit."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp / name, format_name="synthetic", n_samples=N_HEADER,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(N_HEADER),))


def _recording_without_a_rate(tmp, name: str = "legacy.tdms") -> TraceDocument:
    """A legacy TDMS: samples and no time base at all — only a declared rate can give it one."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp / name, format_name="TDMS (legacy)", n_samples=N_HEADER,
                         sample_rate_hz=None, rate_provenance=Provenance.UNKNOWN,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(N_HEADER),))


# ================================================================================================
# the bench: ONE MainWindow for the module (each teardown leaves ~1,400 dead Qt wrappers behind
# in PySide6 6.11; the neighbouring previews tests share one window for the same reason)
# ================================================================================================
def _pump(ms: int) -> None:
    """Process events for ``ms`` without building a QTimer/QEventLoop per poll (thousands of
    those per file trip PySide6 6.11's SignalManager crash; see the 2026-09-13 note)."""
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
    yield {"window": window, "tmp": tmp_path_factory.mktemp("s8_sc3")}
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
        self.panel = window._analysis_panel
        self.det = window._analysis_panel.detection_panel

    def open(self, doc: TraceDocument, tab: str = "time"):
        """Open ``doc``, select its channel, open Signal Conditioning on it and let it settle."""
        did = self.session.add_document(doc)
        self.window._create_default_layers(self.session.document(did))
        self.window._on_selection(did, 0)
        _pump(20)
        return did, self.open_window(tab)

    def open_window(self, tab: str = "time"):
        """Analysis ▸ Signal conditioning… (⇧⌘F), however many times."""
        self.window._open_signal_conditioning()
        ws = self.window._signal_conditioning
        ws.set_preview_tab(tab)
        _settle(ws)
        return ws

    def declare(self, did: str, *, unit: str | None = None, rate: float | None = None) -> str:
        """Inspector ▸ File Info ▸ Sample rate / time unit… with ``unit`` (the time column is
        in…), with ``rate`` (Hz), or — neither — "Use the file's own time base"; Apply. The real
        dialog, so the call is the one File Info makes. Returns what the dialog says."""
        dialog = SampleRateDialog(self.session, did, self.window)
        try:
            if unit is not None:
                dialog.mode.setCurrentIndex(dialog.mode.findData("time_unit"))
                dialog.unit.setCurrentIndex(dialog.unit.findData(unit))
            elif rate is not None:
                dialog.mode.setCurrentIndex(dialog.mode.findData("sample_rate"))
                dialog.rate.setText(f"{rate:g}")
            else:
                dialog.mode.setCurrentIndex(dialog.mode.findData("file"))
            dialog.apply()
            return dialog.error.text()
        finally:
            dialog.close()
            dialog.deleteLater()

    def status(self) -> str:
        return TIMING.sub("", self.window._signal_conditioning.time_status.text())


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
        ws.set_time_mode("overlay")
        ws.set_response_view("magnitude")
        ws.set_preview_tab("time")
        ws.lines_note.setText("")
    yield bench
    if window._signal_conditioning is not None:
        window._signal_conditioning.close()
    for doc in list(bench.session.documents()):
        bench.session.close_document(doc.doc_id)
    _pump(20)
    assert raised == [], f"a slot raised: {raised[:1]}"


def _x_span(curve) -> tuple[float, float]:
    x = curve.xData
    assert x is not None and len(x), "the curve has no data"
    return float(x[0]), float(x[-1])


# ================================================================================================
# 1. the finding: the Time tab, the status and the refusal follow a declared time unit
# ================================================================================================
def test_declaring_the_time_unit_moves_the_open_window_to_the_declared_clock(bench):
    window = bench.window
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    assert ws._source.fs == pytest.approx(50.0)
    assert bench.status() == "No stage is on, so processed = raw · window 2000.00 s"
    assert _x_span(ws.raw_curve) == pytest.approx((T0_AS_READ, T0_AS_READ + (N_CSV - 1) / 50.0))

    said = bench.declare(did, unit="ms")
    assert said.startswith("Declared: 50,000 Hz."), said
    _settle(ws)

    doc = window._session.document(did)
    assert doc.meta.sample_rate_hz == pytest.approx(50_000.0)
    assert bench.det.sample_rate() == pytest.approx(50_000.0), "the form moved (it always did)"
    assert ws._source.fs == pytest.approx(50_000.0), "...and now so does the window's source"
    assert ws._fs == pytest.approx(50_000.0), "...and its chain"
    t0 = doc.meta.t0_s
    assert t0 == pytest.approx(T0_AS_READ / 1000.0)
    assert _x_span(ws.raw_curve) == pytest.approx((t0, t0 + (N_CSV - 1) / FS)), (
        "the Time tab spans the recording's 2 s, not 2000 s")
    trace_x = window._trace_view.visible_x_range_for(did, 0)
    assert _x_span(ws.raw_curve) == pytest.approx(trace_x, abs=0.05), "the main trace's clock"
    assert bench.status() == "No stage is on, so processed = raw · window 2.00 s"
    assert ws.preview_state() == "exact"
    assert "processed — exact" in ws.legend_texts()


def test_the_cutoff_the_form_accepts_after_a_declaration_is_not_refused_by_the_old_nyquist(bench):
    """The finding's own sentence: the form takes 10 kHz at the declared 50 kHz while the
    preview, still at 50 Hz, printed 'at or above Nyquist (25 Hz) for fs=50 Hz'."""
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    bench.declare(did, unit="ms")
    bench.det.set_params({"filter_cutoff_hz": 10_000.0})
    _settle(ws)
    assert bench.det.params()["filter_cutoff_hz"] == 10_000.0, "the form accepted it"
    assert "Nyquist" not in ws.time_status.text() and "Preview unavailable" not in (
        ws.time_status.text())
    assert ws.preview_state() == "exact" and ws.last_result().error == ""
    assert bench.status().startswith("Exact for these stages · window 2.00 s, padded ±")
    low_pass = [r.text for r in ws.stage_rows() if "Low-pass" in r.text][0]
    assert low_pass.endswith("Bessel 8 · 10 kHz (applied inside detection)")
    assert ws._source.fs == ws._fs == pytest.approx(50_000.0)


def test_the_spectrum_tab_follows_a_declaration_frequency_axis_note_and_lines(bench):
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp), tab="spectrum")
    sp = ws.last_result().spectrum
    assert sp.f_hz[-1] == pytest.approx(25.0, rel=0.05), "as read: Nyquist is 25 Hz"
    assert "on a 2000.00 s window" in ws.spectrum_note.text()
    assert "in 20 – 25 Hz." in ws.lines_note.text()

    bench.declare(did, unit="ms")
    _settle(ws)
    sp = ws.last_result().spectrum
    assert sp.f_hz[-1] == pytest.approx(25_000.0, rel=0.05), "the declared clock's Nyquist"
    assert sp.window_s == pytest.approx(N_CSV / FS)
    assert "on a 2.00 s window" in ws.spectrum_note.text() and "2000" not in (
        ws.spectrum_note.text())
    x0, x1 = ws.spectrum_plot.getPlotItem().getViewBox().viewRange()[0]
    assert 10 ** x1 == pytest.approx(sp.f_hz[-1], rel=0.01), "the plotted axis ends at it"
    tone = [float(line.split(" Hz")[0]) for line in ws.spectral_line_texts()]
    assert any(abs(f - 4570.8) < 10.0 for f in tone), (
        f"the 4570.8 Hz tone is found on the declared clock: {ws.spectral_line_texts()}")
    assert "in 20 Hz – 25 kHz." in ws.lines_note.text(), "the band is searched to the new Nyquist"


# ================================================================================================
# 2. either direction: handing the time base back to the file
# ================================================================================================
def test_giving_the_time_base_back_to_the_file_moves_the_open_window_back(bench):
    """'Removing the declaration again while the window is open leaves it the other way
    round (window at 50 kHz, document at 50 Hz).'"""
    window = bench.window
    csv = _csv_in_milliseconds(bench.tmp)
    did = bench.session.add_document(csv)
    window._create_default_layers(bench.session.document(did))
    window._on_selection(did, 0)
    bench.declare(did, unit="ms")                      # declared BEFORE the window opens
    ws = bench.open_window()
    assert ws._source.fs == ws._fs == pytest.approx(50_000.0)
    assert bench.status() == "No stage is on, so processed = raw · window 2.00 s"

    said = bench.declare(did)                          # Use the file's own time base
    assert said.startswith("The file's own time base is in use again."), said
    _settle(ws)
    assert window._session.document(did).meta.sample_rate_hz == pytest.approx(50.0)
    assert bench.det.sample_rate() == pytest.approx(50.0)
    assert ws._source.fs == pytest.approx(50.0) and ws._fs == pytest.approx(50.0)
    assert bench.status() == "No stage is on, so processed = raw · window 2000.00 s"
    assert _x_span(ws.raw_curve) == pytest.approx((T0_AS_READ, T0_AS_READ + (N_CSV - 1) / 50.0))
    assert ws.preview_state() == "exact"
    ws.set_preview_tab("spectrum")
    _settle(ws)
    assert ws.last_result().spectrum.f_hz[-1] == pytest.approx(25.0, rel=0.05)
    assert "on a 2000.00 s window" in ws.spectrum_note.text()


def test_a_rate_declared_on_a_recording_that_had_none_reaches_the_open_window(bench):
    """'Preview unavailable: sample rate unknown' is the window's reading of a recording with no
    time base; declaring one is the same event as declaring another, and so is undoing it."""
    did, ws = bench.open(_recording_without_a_rate(bench.tmp))
    assert ws._source.fs is None and ws._fs is None
    assert bench.status().startswith("Preview unavailable: sample rate unknown")

    bench.declare(did, rate=50_000)
    _settle(ws)
    assert ws._source.fs == pytest.approx(50_000.0) and ws._fs == pytest.approx(50_000.0)
    assert bench.status() == "No stage is on, so processed = raw · window 4.00 s"
    assert ws.preview_state() == "exact"

    bench.declare(did)
    _settle(ws)
    assert ws._source.fs is None and ws._fs is None
    assert bench.status().startswith("Preview unavailable: sample rate unknown")


# ================================================================================================
# 3. the same root, found by a skeptic: one rate in the window, whatever runs next
# ================================================================================================
def test_a_declared_rate_leaves_the_source_and_the_chain_on_one_rate_through_a_detection(bench):
    """Declare 25,000 Hz on a recording whose header says 50,000, then run a detection (which
    pushes the chain at the form's rate): the window held source 50,000 and chain 25,000."""
    panel = bench.panel
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    assert ws._source.fs == ws._fs == FS
    assert bench.status() == "No stage is on, so processed = raw · window 4.00 s"

    bench.declare(did, rate=25_000)
    _settle(ws)
    assert ws._source.fs == ws._fs == pytest.approx(25_000.0), "one rate, at once"
    assert bench.status() == "No stage is on, so processed = raw · window 8.00 s"

    bench.det.set_params({"filter_cutoff_hz": 5_000.0, "min_dwell_s": 5e-4})
    panel.trigger_detection()
    _until(lambda: panel._pending is None, 60_000)
    _pump(150)
    _settle(ws)
    assert panel.current_detection() is not None, "the detection ran"
    assert ws._source.fs == ws._fs == pytest.approx(25_000.0), "and did not split the rates"
    assert bench.status().startswith("Exact for these stages · window 8.00 s, padded ±")
    assert ws.preview_state() == "exact"


# ================================================================================================
# 4. stages that stop fitting at the new rate
# ================================================================================================
@pytest.mark.filterwarnings("ignore:cutoff_hz=.*is above fs/2.5:RuntimeWarning")
def test_a_cutoff_the_new_nyquist_clamps_is_previewed_clamped_not_refused(bench):
    """A 10 kHz low-pass fits 50 kHz; declare 4,000 Hz and the form pulls it under the new
    Nyquist (set_recording → the spin's range, as for a recording switch). The window shows
    that chain at the new rate: no stale Nyquist, no stale 'exact' at 50 kHz."""
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    bench.det.set_params({"filter_cutoff_hz": 10_000.0})
    _settle(ws)
    assert bench.status().startswith("Exact for these stages · window 4.00 s")

    bench.declare(did, rate=4_000)
    _settle(ws)
    clamped = bench.det.params()["filter_cutoff_hz"]
    assert clamped < 2_000.0, f"the form clamps under the new Nyquist (2 kHz): {clamped}"
    assert ws._source.fs == ws._fs == pytest.approx(4_000.0)
    low_pass = [r.text for r in ws.stage_rows() if "Low-pass" in r.text][0]
    assert "10 kHz" not in low_pass and "1.9999 kHz" in low_pass, low_pass
    assert bench.status().startswith("Exact for these stages · window 50.00 s, padded ±")
    assert "Preview unavailable" not in ws.time_status.text()
    assert ws.preview_state() == "exact" and ws.last_result().error == ""


def test_a_stage_the_form_cannot_clamp_is_refused_with_the_new_rate_not_the_old(bench):
    """An explicit notch list [60, 120] Hz is legal at 50 kHz and has no line below the
    Nyquist of a declared 100 Hz. The form does not rewrite a list; the window must refuse it
    in the new rate's words instead of going on saying 'Exact for these stages' at 50 kHz."""
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    bench.det.set_params({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    _settle(ws)
    assert bench.status().startswith("Exact for these stages")

    bench.declare(did, rate=100)
    _settle(ws)
    assert ws.preview_state() == "error"
    assert bench.status() == ("Preview unavailable: no notch frequency below Nyquist (50 Hz): "
                              "freqs=[60.0, 120.0] is unusable at fs=100 Hz")
    assert ws._source.fs == ws._fs == pytest.approx(100.0)
    assert "Exact for these stages" not in ws.time_status.text()


def test_a_baseline_window_longer_than_the_declared_recording_is_refused_in_its_new_length(bench):
    """A 60 s baseline-drift window fits the 2000 s the file seemed to hold; declared in
    milliseconds the recording is 2 s long, and the kernel says so — in the new length, the way
    the window words any stage that does not fit its recording after a source change."""
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    bench.det.set_params({"detrend_enabled": True, "detrend_method": "block_median",
                          "detrend_window_s": 60.0})
    _settle(ws)
    assert ws.preview_state() == "exact"
    assert bench.status() == "Exact for these stages · window 2000.00 s, padded ±120.00 s"

    bench.declare(did, unit="ms")
    _settle(ws)
    assert ws.preview_state() == "error"
    assert bench.status() == ("Preview unavailable: window_s=60s is longer than the recording "
                              "(2s); no block can be formed. Use method='constant' if that is "
                              "what you meant.")
    assert ws._source.fs == ws._fs == pytest.approx(50_000.0)


# ================================================================================================
# 5. the Response tab and a panned Time tab
# ================================================================================================
def test_the_response_tab_is_drawn_at_the_declared_rate(bench):
    """Phase, group delay and Step are designed at the chain's rate; the window used to
    hold the chain at the old rate until the next chain edit."""
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp), tab="response")
    bench.det.set_params({"filter_cutoff_hz": 20.0})                    # legal at 50 Hz
    ws.set_response_view("phase")
    _settle(ws)
    applied = ws.response_view_widgets("phase")["applied"]
    assert applied.xData[-1] == pytest.approx(25.0, rel=0.05), "as read: out to 25 Hz"

    bench.declare(did, unit="ms")
    _settle(ws)
    applied = ws.response_view_widgets("phase")["applied"]
    assert applied.xData[-1] == pytest.approx(25_000.0, rel=0.05), "now out to 25 kHz"
    assert ws._source.fs == ws._fs == pytest.approx(50_000.0)


def test_a_time_plot_the_user_panned_is_reframed_on_the_new_clock(bench):
    """Follow is off after a pan, so the plot keeps the range the user chose — in OLD seconds.
    The previewed samples are the same ones, now on a clock twice as slow: the plot must show
    them, not the first 3 s of 6."""
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    vb = ws.overlay_plot.getPlotItem().getViewBox()
    vb.setXRange(0.0, 3.0, padding=0)
    vb.sigRangeChangedManually.emit([True, False])         # what a drag or wheel emits
    _settle(ws)
    assert not ws.follows_main() and ws.current_window() == (0, int(3.0 * FS))

    bench.declare(did, rate=25_000)
    _settle(ws)
    assert ws.current_window() == (0, int(3.0 * FS)), "the same samples are still previewed"
    assert _x_span(ws.raw_curve) == pytest.approx((0.0, (3.0 * FS - 1) / 25_000.0))
    x0, x1 = vb.viewRange()[0]
    assert (x0, x1) == pytest.approx((0.0, (3.0 * FS - 1) / 25_000.0), abs=1e-6)


# ================================================================================================
# 6. a window that is not on screen, and one that is mid-computation
# ================================================================================================
def test_a_closed_window_computes_nothing_and_reopens_on_the_declared_rate(bench, monkeypatch):
    window = bench.window
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    ws.close()
    _pump(10)
    calls = []
    real = sc.compute_preview
    monkeypatch.setattr(sc, "compute_preview", lambda req: calls.append(req.fs) or real(req))
    bench.declare(did, unit="ms")
    _pump(ws.DEBOUNCE_MS + 150)
    assert calls == [] and not ws.isVisible(), "a hidden window computes nothing"
    assert ws.last_result() is None

    ws = bench.open_window()                                # Analysis ▸ Signal conditioning…
    assert calls and all(fs == pytest.approx(50_000.0) for fs in calls), (
        "and it comes up on the declared rate")
    assert ws._source.fs == ws._fs == pytest.approx(50_000.0)
    assert bench.status() == "No stage is on, so processed = raw · window 2.00 s"
    t0 = window._session.document(did).meta.t0_s
    assert _x_span(ws.raw_curve) == pytest.approx((t0, t0 + (N_CSV - 1) / FS))


def test_a_preview_scheduled_at_the_old_rate_is_posted_at_the_new_one(bench, monkeypatch):
    """The user edits the chain and, inside the 250 ms debounce, declares the rate: nothing
    is ever computed at 50 kHz after that edit."""
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    calls = []
    real = sc.compute_preview
    monkeypatch.setattr(sc, "compute_preview", lambda req: calls.append(req.fs) or real(req))
    bench.det.set_params({"filter_cutoff_hz": 5_000.0})
    assert ws._debounce.isActive(), "a preview at 50 kHz is scheduled"
    bench.declare(did, rate=25_000)
    _settle(ws)
    assert calls and all(fs == pytest.approx(25_000.0) for fs in calls), calls
    assert bench.status().startswith("Exact for these stages · window 8.00 s")


def test_an_answer_computed_at_the_old_rate_never_lands_after_the_declaration(bench, monkeypatch):
    """The worker is inside a preview at 50 kHz when the user declares 25 kHz. When it
    finishes, its answer is for a clock the window has left: it may not be drawn, and nothing
    computed at the old rate is on screen when the new one arrives."""
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    real = sc.compute_preview
    entered, gate = threading.Event(), threading.Event()
    armed = {"on": False}

    def spy(req):
        if armed["on"] and float(req.fs) == FS:
            armed["on"] = False
            entered.set()
            assert gate.wait(30), "the test never released the worker"
        res = real(req)
        res.computed_at_hz = float(req.fs)          # which clock THIS answer belongs to
        return res

    monkeypatch.setattr(sc, "compute_preview", spy)
    armed["on"] = True
    bench.det.set_params({"median_enabled": True, "median_window_s": 1e-3})   # a new request
    _until(entered.is_set, 10_000)                  # ...which the worker is now inside, at 50 kHz

    arrived, drawn = [], []
    real_outcome, real_render = ws._on_outcome, ws._render_time
    monkeypatch.setattr(ws, "_on_outcome", lambda o: arrived.append(
        (o.status, getattr(o.result, "computed_at_hz", None))) or real_outcome(o))
    monkeypatch.setattr(ws, "_render_time", lambda: ws.last_result() is not None and drawn.append(
        ws.last_result().computed_at_hz) or real_render())
    monkeypatch.setattr(ws, "DEBOUNCE_MS", 5_000)   # keep the new rate's own request waiting

    bench.declare(did, rate=25_000)
    gate.set()                                      # the old-rate answer is let go
    _until(lambda: ws.worker_idle() and ("ok", FS) in arrived, 10_000)
    assert ws.last_result() is None and drawn == [], "the 50 kHz answer was not drawn"
    assert ws.preview_state() == "pending"

    ws._schedule(immediate=True)                    # the new rate's request, now
    _settle(ws)
    assert set(drawn) == {25_000.0} and ws.last_result().computed_at_hz == 25_000.0
    assert ws._source.fs == ws._fs == pytest.approx(25_000.0)
    assert bench.status().startswith("Exact for these stages · window 8.00 s, padded ±")


def _observe(bench, ws) -> dict:
    """Everything the window shows that depends on a rate, on its Time and Spectrum tabs."""
    seen = {}
    for tab in ("time", "spectrum"):
        ws.set_preview_tab(tab)
        _settle(ws)
        res = ws.last_result()
        seen[tab] = {
            "title": ws.windowTitle(), "source_fs": ws._source.fs, "chain_fs": ws._fs,
            "status": bench.status(), "legends": ws.legend_texts(),
            "rows": [r.text for r in ws.stage_rows()],
            "x": _x_span(ws.raw_curve),
            "exact_button": (ws.exact_button.isEnabled(), ws.exact_button.text(),
                             ws.exact_button.toolTip()),
        }
        if tab == "spectrum":           # the notes are only rewritten when this tab is drawn
            seen[tab].update({
                "spectrum": (float(res.spectrum.f_hz[-1]), res.spectrum.nperseg,
                             res.spectrum.window_s),
                "spectrum_note": ws.spectrum_note.text(), "lines_note": ws.lines_note.text(),
                "lines": ws.spectral_line_texts()})
    ws.set_preview_tab("time")
    return seen


def test_an_open_window_shows_what_a_reopened_one_shows_in_both_directions(bench):
    """The finding's own remedy was 'close it and open it again'. This pins the standard: after
    a declaration, and after taking it away, the window that stayed open shows exactly what a
    window opened afterwards shows — every rate-dependent word and number of two tabs."""
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    bench.det.set_params({"filter_cutoff_hz": 10.0})        # a stage with a rate in its words
    for step, declare in (("declared in ms", dict(unit="ms")), ("handed back", {})):
        bench.declare(did, **declare)
        _settle(ws)
        kept_open = _observe(bench, ws)
        ws.close()
        ws = bench.open_window()
        reopened = _observe(bench, ws)
        for tab, shown in kept_open.items():
            for what, value in shown.items():
                assert value == reopened[tab][what], (step, tab, what)
    assert reopened["time"]["source_fs"] == pytest.approx(50.0), "and it ended where it began"


# ================================================================================================
# 7. the exact pass and its cache (S8-NUM1 shares the key)
# ================================================================================================
def _estimator_chain() -> dict:
    return {"filter_cutoff_hz": 5_000.0, "detrend_enabled": True,
            "detrend_method": "block_median", "detrend_window_s": 0.5, "polarity": -1}


def _compute_exact_and_wait(ws) -> None:
    assert ws.compute_exact(), "an estimator stage on part of the recording offers the pass"
    _until(lambda: not ws.exact_pending(), 60_000)
    _settle(ws)


def _truth_at(bench, did: str, rate_hz: float) -> np.ndarray:
    """What the chain really makes of the whole calibrated channel at ``rate_hz``."""
    raw = np.asarray(bench.session.calibrated(did, 0).data)
    return np.asarray(conditioned_trace(raw, float(rate_hz), conditioning_params(
        bench.det.params())), dtype=np.float64)


def test_a_pass_computed_at_the_old_rate_is_not_the_exact_trace_at_the_new_one(bench):
    window = bench.window
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    bench.det.set_params(_estimator_chain())
    window._trace_view.zoom_x_to(1.0, 2.0)
    _pump(30)
    _settle(ws)
    assert bench.status().startswith("Window estimate — Baseline drift uses whole-recording")
    _compute_exact_and_wait(ws)
    assert bench.status().startswith("Whole recording (exact) · showing 1.00 s")
    old_pass = np.array(ws._memo.get(ws._current_memo_key())[0], dtype=np.float64)

    bench.declare(did, rate=25_000)
    _settle(ws)
    window._trace_view.zoom_x_to(2.0, 4.0)                  # part of the recording again
    _pump(30)
    _settle(ws)
    truth = _truth_at(bench, did, 25_000.0)
    assert np.abs(old_pass - truth).max() > 1.0, "the two clocks really condition differently"
    assert "Whole recording (exact)" not in ws.time_status.text()
    assert bench.status().startswith("Window estimate — Baseline drift uses whole-recording")
    assert "processed — window estimate" in ws.legend_texts()
    assert ws.exact_button.isEnabled() and ws.exact_button.text() == "Compute exact"

    _compute_exact_and_wait(ws)
    res = ws.last_result()
    assert res.from_exact_pass and bench.status().startswith("Whole recording (exact)")
    shown = np.asarray(res.processed_window, dtype=np.float64)
    np.testing.assert_allclose(shown, truth[res.start:res.stop], rtol=0, atol=1e-3)


def test_a_reopened_window_does_not_serve_the_old_rates_exact_pass(bench):
    """S8-NUM1's own route: Compute exact, declare, hide the window and open it again. The
    reopened window found the pass computed at the old rate under an unchanged key."""
    window = bench.window
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    bench.det.set_params(_estimator_chain())
    window._trace_view.zoom_x_to(1.0, 2.0)
    _pump(30)
    _settle(ws)
    _compute_exact_and_wait(ws)
    assert bench.status().startswith("Whole recording (exact)")

    ws.close()
    bench.declare(did, rate=25_000)
    window._trace_view.zoom_x_to(2.0, 4.0)
    _pump(30)
    ws = bench.open_window()
    truth = _truth_at(bench, did, 25_000.0)
    assert ws._source.fs == pytest.approx(25_000.0)
    assert "Whole recording (exact)" not in ws.time_status.text()
    assert bench.status().startswith("Window estimate — Baseline drift uses whole-recording")
    assert ws.exact_button.isEnabled()
    assert "press Compute exact" in ws.time_status.text()

    _compute_exact_and_wait(ws)
    res = ws.last_result()
    np.testing.assert_allclose(np.asarray(res.processed_window, dtype=np.float64),
                               truth[res.start:res.stop], rtol=0, atol=1e-3)


def test_an_exact_pass_still_running_at_the_old_rate_is_cancelled_by_the_declaration(
        bench, monkeypatch):
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp))
    bench.det.set_params(_estimator_chain())
    bench.window._trace_view.zoom_x_to(1.0, 2.0)
    _pump(30)
    _settle(ws)
    real = sc.compute_exact
    entered, gate = threading.Event(), threading.Event()

    def slow(req):
        entered.set()
        assert gate.wait(30), "the test never released the worker"
        res = real(req)
        res.computed_at_hz = float(req.fs)
        return res

    monkeypatch.setattr(sc, "compute_exact", slow)
    assert ws.compute_exact()
    _until(entered.is_set, 10_000)
    assert ws.exact_pending()

    bench.declare(did, rate=25_000)
    assert not ws.exact_pending(), "an edit of the clock cancels it, as an edit of the chain does"
    assert ws.exact_button.text() == "Compute exact"
    gate.set()
    _settle(ws)
    assert len(ws._memo) == 0, "the late pass at 50 kHz was not stored under the new rate"
    assert "Computing exact" not in ws.time_status.text()


# ================================================================================================
# 8. which declarations concern the window, and how often it hears them
# ================================================================================================
def test_a_declaration_on_another_recording_does_not_disturb_the_open_window(bench, monkeypatch):
    window = bench.window
    did, ws = bench.open(_recording_with_a_header_rate(bench.tmp, "shown.dat"))
    other = bench.session.add_document(_csv_in_milliseconds(bench.tmp, "elsewhere.csv"))
    window._create_default_layers(bench.session.document(other))
    _pump(20)
    shown = ws.last_result()
    pushed = []
    real = ws.set_source
    monkeypatch.setattr(ws, "set_source", lambda source: pushed.append(source) or real(source))

    bench.declare(other, unit="ms")
    _pump(ws.DEBOUNCE_MS + 100)
    assert pushed == [], "the window shows another recording: nothing to re-read"
    assert ws.last_result() is shown and ws.preview_state() == "exact"
    assert ws._source.doc_id == did and ws._source.fs == pytest.approx(FS)


def test_a_segment_in_the_window_follows_the_declaration_made_on_its_parent(bench):
    """A segment is declared through its parent; it is re-made on the parent's clock and says
    so with its own sampleRateChanged, AFTER the parent's."""
    window = bench.window
    parent = bench.session.add_document(_csv_in_milliseconds(bench.tmp, "parent.csv"))
    window._create_default_layers(bench.session.document(parent))
    seg = bench.session.create_segment(parent, 10_000, 60_000, name="cut")
    window._create_default_layers(bench.session.document(seg))
    window._on_selection(seg, 0)
    _pump(20)
    ws = bench.open_window()
    assert ws._source.doc_id == seg and ws._source.fs == pytest.approx(50.0)
    assert bench.status() == "No stage is on, so processed = raw · window 1000.00 s"

    bench.declare(parent, unit="ms")
    _settle(ws)
    assert bench.session.document(seg).meta.sample_rate_hz == pytest.approx(50_000.0)
    assert ws._source.fs == ws._fs == pytest.approx(50_000.0)
    assert bench.status() == "No stage is on, so processed = raw · window 1.00 s"


def test_the_window_hears_a_declaration_once_however_often_it_was_opened(bench, monkeypatch):
    """The connection is made once, with the window's neighbours, not once per open: three
    open/close cycles later one declaration is one re-read."""
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    for _ in range(3):
        ws.close()
        _pump(5)
        ws = bench.open_window()
    pushed = []
    real = ws.set_source
    monkeypatch.setattr(ws, "set_source", lambda source: pushed.append(
        None if source is None else source.fs) or real(source))
    bench.declare(did, unit="ms")
    _settle(ws)
    assert pushed == [pytest.approx(50_000.0)]


def test_a_declaration_after_the_window_was_stopped_neither_raises_nor_restarts_it(bench):
    """LAST to use the shared window: ``abort()`` is MainWindow's teardown and is final. The
    slot that follows the declaration reaches a window that is still on screen but whose
    thread is retired for good; it must hand it the new rate and start nothing."""
    did, ws = bench.open(_csv_in_milliseconds(bench.tmp))
    assert ws.worker_thread() is not None
    ws.abort()
    assert ws.worker_thread() is None
    bench.declare(did, unit="ms")
    _pump(ws.DEBOUNCE_MS + 100)
    assert ws.worker_thread() is None, "nothing restarts the preview thread after the teardown"
    assert ws._source.fs == pytest.approx(50_000.0), "...but the window was still told"
