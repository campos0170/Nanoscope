"""S8-TU1 — The Navigator strip stays on the OLD clock after a sample-rate / time-unit declaration.

MEASURED DEFECT (sweep 8, area TU; the finder, two skeptics and the coordinator all re-ran it).
A bare ``time,current`` CSV whose time column is in milliseconds reads as 50 Hz, t = 10,000 s.
Inspector ▸ File Info ▸ Sample rate / time unit… ▸ "The time column is in… milliseconds" ▸ Apply
re-frames the trace to 10–12 s and every other surface moves with it — the status bar, File
Info, the Scan bar. The strip under the trace did not: ``Navigator.bind`` listened to
``layersChanged`` / ``viewModeChanged`` / ``documentClosed`` / ``calibrationChanged`` and never to
``Session.sampleRateChanged``, and ``_refresh`` is the only place the strip re-reads the axis.
So the curve, the extent, the region's bounds and the lens stayed on the old clock while the view
they mirror was on the new one. Measured on the 2 s slice of the owner's −600 mV recording the
sweep used (here: the same shape on a 0.4 s synthetic one): the label flipped to "Local context ·
9× view" over a 2 s recording, 0 of its 100,000 samples were inside the x-range the strip was
showing (the curve sat at 10,000–12,000 under a strip showing 10–28), and dragging its shaded
window to 13–15 s moved the main trace to 13–15 s, past the last sample at 11.99998 s, so the pane
went blank. Zoom-to-full did not repair it; only selecting another recording and coming back did.
The same strip is wrong the other way round on "Use the file's own time base", on a
sample-rate-only declaration (an ABF declared at another rate: the curve fills half the strip, or
the region's bound stops half way along the recording), and on a refused recording whose time axis
is in SAMPLES declared in microseconds (strip extent 0–99,999 samples under a 10–12 s trace).

THE FIX is in ``Navigator`` (nanoscope/gui/navigator.py): ``bind`` now also listens to
``sampleRateChanged`` and re-feeds the strip — curve, extent, region bounds, lens and label,
Measure band — from the Session, for a recording that is drawn. It does not rely on the trace
view having re-framed first: the extent and the curve come from the Session, and when the view
re-frames afterwards its own ``sigVisibleRangeChanged`` carries the range to the already-fresh
strip, so either order of the two slots ends in the same place.

These tests build the recordings in memory (what the CSV reader reports for the two time columns
— 50 Hz *measured* from t = 10,000 and a refused 0.1 Hz from t = 0.05 — and an ABF's header rate),
drive the real MainWindow, press the real File Info button (the modal loop the offscreen suite
suppresses is replaced by the user's choice and Apply), click the real Scan-bar buttons, and
drag the shaded window with real mouse events. They assert what the strip shows: its label, the
x-range it displays, the x-range of the samples it plots and how many of them fall inside, the
shaded window and where the Measure band sits — on the clock the trace is drawn on.

MEASURED BEFORE (this tree without the fix, run from the same checkout): all 12 fail, the
strip reading e.g. ``label='Local context · 9× view', seen=(10.0, 13.6), curve=(10000, 10399.98),
drawn=0/20000`` after the millisecond declaration, and the trace ending at 10.95 s after a real
drag over a recording that ends at 10.39998 s.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.core.document import TraceDocument
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.sample_rate_dialog import SampleRateDialog

N = 20_000
#: A time column written in milliseconds (20 µs steps, 0.02 per row) starting at 10,000 ms. Read
#: as seconds it is 50 Hz and t0 = 10,000 s; declared in milliseconds it is 50 kHz from t0 = 10 s.
MS_AS_READ_HZ = 50.0
MS_AS_READ_T0 = 10_000.0
MS_DECLARED_END = 10.0 + (N - 1) / 50_000.0                  # 10.39998 s
READ_END = MS_AS_READ_T0 + (N - 1) / MS_AS_READ_HZ           # 10,399.98 "s"
ABF_HZ = 20_000.0
MS = SampleRateDeclaration("time_unit", time_unit="ms")


# ================================================================================================
# fixtures / builders
# ================================================================================================

@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace, so a saved ``windowState`` cannot resize the strip."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("S8TU1NavigatorClock")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def window(isolated_settings, qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()                       # the strip is dragged with real mouse events: it needs a size
    QTest.qWait(20)
    yield w
    w._analysis_controller.abort()
    w._dirty = False               # a declaration is project state; leave without the Save prompt


def _trace(n: int, seed: int = 0) -> np.ndarray:
    """A quiet baseline with a handful of blockades, so the overview has something to draw."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -50.0) + rng.normal(0.0, 0.5, n)
    for start in range(n // 10, n, n // 5):
        x[start:start + n // 100] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _document(path, *, rate, provenance, t0=0.0, refused=None, fmt="Text trace") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name=fmt, n_samples=N, sample_rate_hz=rate,
                         rate_provenance=provenance, channels=(info,), t0_s=t0,
                         refused_rate_hz=refused)
    return TraceDocument(meta=meta, channels=(_trace(N),))


def _csv_in_ms(tmp_path) -> TraceDocument:
    """What the CSV reader reports for a bare ``time,current`` file whose column is in ms."""
    return _document(tmp_path / "bare_ms.csv", rate=MS_AS_READ_HZ, provenance=Provenance.MEASURED,
                     t0=MS_AS_READ_T0)


def _csv_in_us(tmp_path) -> TraceDocument:
    """...and for one in microseconds: 10 per row is 0.1 Hz, refused as a clock, so the axis is
    in SAMPLES until the unit is declared (the interval it measured is kept)."""
    return _document(tmp_path / "bare_us.csv", rate=None, provenance=Provenance.UNKNOWN, t0=0.05,
                     refused=0.1)


def _abf_at_20k(tmp_path) -> TraceDocument:
    """A header-declared rate: no column to re-read, so only a rate can be declared."""
    return _document(tmp_path / "pore.abf", rate=ABF_HZ, provenance=Provenance.FILE_META,
                     fmt="Axon ABF")


def _show(window: MainWindow, doc: TraceDocument, *, declared=None) -> str:
    """What a finished open does: register the recording, layer it, select it. ``declared`` is
    a declaration already in force when it first reaches the screen, as a reopened project has."""
    doc_id = window._session.add_document(doc)
    if declared is not None:
        window._session.set_sample_rate_declaration(doc_id, declared)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    QTest.qWait(10)
    return doc_id


def _declare(window: MainWindow, monkeypatch, *, unit=None, rate=None) -> str:
    """Inspector ▸ File Info ▸ Sample rate / time unit… — the real button opens the real dialog,
    and the test plays the user in place of the modal loop the offscreen suite suppresses:
    choose, then press Apply. Neither ``unit`` nor ``rate`` is "Use the file's own time base"."""
    said: list[str] = []

    def play_the_user(dialog) -> int:
        if unit is not None:
            dialog.mode.setCurrentIndex(dialog.mode.findData("time_unit"))
            dialog.unit.setCurrentIndex(dialog.unit.findData(unit))
        elif rate is not None:
            dialog.mode.setCurrentIndex(dialog.mode.findData("sample_rate"))
            dialog.rate.setText(repr(float(rate)))
        else:
            dialog.mode.setCurrentIndex(dialog.mode.findData("file"))
        dialog.apply()
        said.append(dialog.error.text())
        return int(QtWidgets.QDialog.Accepted)

    monkeypatch.setattr(SampleRateDialog, "exec", play_the_user)
    button = window._file_info_panel._edit_rate
    assert button.isEnabled()
    QTest.mouseClick(button, QtCore.Qt.LeftButton)
    QTest.qWait(10)
    assert said and not said[0].startswith("Not applied"), said
    return said[0]


@dataclass(frozen=True)
class Strip:
    """Everything the navigator strip shows, read off the widgets the user is looking at."""
    label: str                          # "Full recording" / "Local context · 9× view"
    local: bool
    seen: tuple[float, float]           # the x-range the strip displays
    curve: tuple[float, float]          # the x-range of the samples it plots
    drawn: int                          # how many of them fall inside what it displays
    total: int
    region: tuple[float, float]         # the shaded window


def _strip(window: MainWindow) -> Strip:
    nav = window._navigator
    xs = np.asarray(nav._curve.xData)
    lo, hi = nav.getPlotItem().vb.viewRange()[0]
    slack = 1e-9 * max(abs(hi - lo), 1e-12)
    return Strip(
        label=nav._mode_label.text(), local=nav.is_local_mode(), seen=(float(lo), float(hi)),
        curve=(float(xs[0]), float(xs[-1])),
        drawn=int(np.count_nonzero((xs >= lo - slack) & (xs <= hi + slack))), total=int(xs.size),
        region=tuple(float(v) for v in nav._region.getRegion()))


def _trace_range(window: MainWindow) -> tuple[float, float]:
    lo, hi = window._trace_view.anchor_plot().vb.viewRange()[0]
    return float(lo), float(hi)


def _extent(window: MainWindow, doc_id: str) -> tuple[float, float]:
    """The recording's first and last x on the clock it is drawn on right now."""
    values = window._session.x_axis(doc_id).values
    return float(values[0]), float(values[-1])


def _assert_the_strip_shows_the_whole_recording(window: MainWindow, doc_id: str) -> None:
    """The trace is framed on the whole recording; the strip must be showing that same recording
    on that same clock — all of it, with every sample inside, and the shaded window equal to the
    trace's own."""
    extent, trace, strip = _extent(window, doc_id), _trace_range(window), _strip(window)
    assert trace == pytest.approx(extent, abs=1e-6), "premise: the trace re-framed on the new clock"
    assert strip.label == "Full recording" and not strip.local, strip
    assert strip.seen == pytest.approx(extent, abs=1e-6), strip
    assert strip.curve == pytest.approx(extent, abs=1e-6), strip
    assert strip.drawn == strip.total, strip
    assert strip.region == pytest.approx(trace, abs=1e-6), strip


def _zoom_in(window: MainWindow, presses: int) -> None:
    """The Scan bar's ``+``, one octave per press, around the centre of the view."""
    button = window._pan_scrollbar._zoom_in_button
    for _ in range(presses):
        assert button.isEnabled()
        QTest.mouseClick(button, QtCore.Qt.LeftButton)
        QTest.qWait(5)


def _drag_the_shaded_window_to(window: MainWindow, x: float) -> None:
    """Press on the body of the shaded window with the real mouse and drag until the pointer is
    over data-x ``x`` — which the caller chooses beyond the end of the recording.

    pyqtgraph drops mouse moves that come closer together than 1 / ``mouseRateLimit`` seconds
    and leaves a drag where the last move that survived put it (the release only ends it). A
    hand never outruns that; a test stepping every few milliseconds does, and would stop
    wherever the clock happened to let a move through. The limit is lifted for the gesture so
    every step lands, and put back."""
    nav = window._navigator
    viewport, vb = nav.viewport(), nav.getPlotItem().vb

    def pixel(at: float) -> QtCore.QPoint:
        point = nav.mapFromScene(vb.mapViewToScene(QtCore.QPointF(at, 0.0)))
        return QtCore.QPoint(point.x(), viewport.height() // 2)

    lo, hi = nav._region.getRegion()
    start, end = pixel(0.5 * (lo + hi)), pixel(x)
    rate_limit = pg.getConfigOption("mouseRateLimit")
    pg.setConfigOption("mouseRateLimit", 0)
    try:
        QTest.mousePress(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
        steps = 12
        for i in range(1, steps + 1):
            QTest.mouseMove(viewport, QtCore.QPoint(start.x() + (end.x() - start.x()) * i // steps,
                                                    start.y()))
            QtWidgets.QApplication.processEvents()
        QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, end)
    finally:
        pg.setConfigOption("mouseRateLimit", rate_limit)
    QTest.qWait(10)


def _record_the_order(window: MainWindow) -> list[str]:
    """Which of the two slots that re-frame / re-feed on a declaration runs first."""
    order: list[str] = []
    view, nav = window._trace_view, window._navigator
    rebuild, refresh = view.rebuild, nav._refresh
    view.rebuild = lambda: (order.append("trace"), rebuild())[1]
    nav._refresh = lambda: (order.append("navigator"), refresh())[1]
    return order


def _navigator_first(window: MainWindow) -> None:
    """Reconnect the trace view's rate-change slot AFTER the navigator's, so the signal reaches
    the strip while the trace is still on the old clock."""
    session, view = window._session, window._trace_view
    session.sampleRateChanged.disconnect(view._on_sample_rate_changed)
    session.sampleRateChanged.connect(view._on_sample_rate_changed)


# ================================================================================================
# 1. the time-unit declaration — the report
# ================================================================================================

def test_a_millisecond_declaration_puts_the_whole_strip_on_the_new_clock(
        window, monkeypatch, tmp_path):
    """THE REPORT: open the millisecond CSV (read as 50 Hz from t = 10,000 s), declare the column
    to be in milliseconds. Measured before the fix: label 'Local context · 9× view', strip
    showing 10–13.6 s with its curve at 10,000–10,400, 0 of 20,000 samples inside."""
    doc_id = _show(window, _csv_in_ms(tmp_path))
    assert window._rate_label.text() == "50 Hz · measured"
    before = _strip(window)
    assert before.label == "Full recording"
    assert before.seen == pytest.approx((MS_AS_READ_T0, READ_END), rel=1e-9)

    _declare(window, monkeypatch, unit="ms")

    assert window._rate_label.text() == "50,000 Hz · user"
    assert _extent(window, doc_id) == pytest.approx((10.0, MS_DECLARED_END), abs=1e-9)
    _assert_the_strip_shows_the_whole_recording(window, doc_id)


def test_the_shaded_window_cannot_be_dragged_past_the_end_of_the_data(
        window, monkeypatch, tmp_path):
    """Measured before the fix: the region's bounds were min(old extent, view) .. max(old extent,
    view), here 10 .. 10,399.98, so a real drag of the shaded window took the main trace to
    10.85–10.95 s over a recording that ends at 10.39998 s, and the pane went blank."""
    doc_id = _show(window, _csv_in_ms(tmp_path))
    _declare(window, monkeypatch, unit="ms")
    end = _extent(window, doc_id)[1]
    _zoom_in(window, 2)
    lo_before, hi_before = _trace_range(window)

    _drag_the_shaded_window_to(window, end + 5 * (hi_before - lo_before))

    lo, hi = _trace_range(window)
    assert lo > lo_before, "the press landed on the shaded window and moved the trace"
    assert hi == pytest.approx(end, abs=1e-6), (lo, hi, end)
    assert _strip(window).region[1] <= end + 1e-6


def test_zoom_to_full_after_a_declaration_shows_the_whole_recording(
        window, monkeypatch, tmp_path):
    """Measured before the fix: the Scan bar's Full left 'Local context · 9× view', strip 10–13.6,
    0 samples drawn — only re-selecting the recording repaired it."""
    doc_id = _show(window, _csv_in_ms(tmp_path))
    _declare(window, monkeypatch, unit="ms")
    _zoom_in(window, 4)

    QTest.mouseClick(window._pan_scrollbar._full_trace_button, QtCore.Qt.LeftButton)
    QTest.qWait(10)

    _assert_the_strip_shows_the_whole_recording(window, doc_id)


def test_the_lens_of_a_zoomed_view_is_cut_from_the_new_clock(window, monkeypatch, tmp_path):
    """At event scale the strip is a lens of nine view-widths. After the declaration it has to be
    cut from the recording as it now is: nine widths of the new view, on the data. Measured
    before the fix, at a 25 ms view of the 0.4 s recording: a lens clamped to the OLD extent, so
    it ran 10.19–10.41 s over a recording that ends at 10.39998 s."""
    doc_id = _show(window, _csv_in_ms(tmp_path))
    _declare(window, monkeypatch, unit="ms")

    _zoom_in(window, 4)

    nav = window._navigator
    lens, view = nav.context_range(), _trace_range(window)
    first, last = _extent(window, doc_id)
    assert nav.is_local_mode() and nav._mode_label.text() == "Local context · 9× view"
    assert lens[1] - lens[0] == pytest.approx(9.0 * (view[1] - view[0]), rel=1e-6)
    assert first - 1e-9 <= lens[0] and lens[1] <= last + 1e-9, "the lens stays on the data"
    assert lens[0] <= view[0] and view[1] <= lens[1]


# ================================================================================================
# 2. the other ways the clock moves
# ================================================================================================

@pytest.mark.parametrize("hz", [40_000.0, 10_000.0], ids=["recording_gets_shorter",
                                                          "recording_gets_longer"])
def test_a_sample_rate_declaration_re_measures_the_strip(window, monkeypatch, tmp_path, hz):
    """An ABF declared at another rate (the header's rate cannot be re-read from a column, so a
    rate is the only declaration). Measured before the fix, shorter: the curve and the extent
    stayed 0–1 s under a 0–0.5 s trace, so the shaded window was half the strip at full view and
    could be dragged to 1 s; longer: the curve filled the left half of the strip and the region's
    bound stopped at the old 1 s, half way along the 2 s recording."""
    doc_id = _show(window, _abf_at_20k(tmp_path))
    assert _extent(window, doc_id)[1] == pytest.approx((N - 1) / ABF_HZ)

    _declare(window, monkeypatch, rate=hz)

    end = (N - 1) / hz
    assert _extent(window, doc_id) == pytest.approx((0.0, end), abs=1e-9)
    _assert_the_strip_shows_the_whole_recording(window, doc_id)
    _zoom_in(window, 2)
    lo_before, hi_before = _trace_range(window)
    _drag_the_shaded_window_to(window, end + 5 * (hi_before - lo_before))
    lo, hi = _trace_range(window)
    assert lo > lo_before
    assert hi == pytest.approx(end, abs=1e-6), (
        "the shaded window runs to the last sample of the recording as it now is, and no further")


def test_removing_a_declaration_puts_the_strip_back_on_the_files_clock(
        window, monkeypatch, tmp_path):
    """'Use the file's own time base', from a recording that reached the screen already
    declared (as a reopened project has it): the trace goes back to 10,000–10,400 and the strip
    must follow. Measured before the fix: 'Local context · 9× view', strip 6,800–10,400, its
    curve still at 10–10.4, 0 samples drawn, and a region whose lower bound was 10, not the
    file's first sample at 10,000."""
    doc_id = _show(window, _csv_in_ms(tmp_path), declared=MS)
    assert _extent(window, doc_id) == pytest.approx((10.0, MS_DECLARED_END), abs=1e-9)
    _assert_the_strip_shows_the_whole_recording(window, doc_id)

    _declare(window, monkeypatch)                     # "Use the file's own time base"

    assert window._rate_label.text() == "50 Hz · measured"
    start, last = _extent(window, doc_id)
    assert (start, last) == pytest.approx((MS_AS_READ_T0, READ_END), rel=1e-9)
    _assert_the_strip_shows_the_whole_recording(window, doc_id)
    _zoom_in(window, 2)
    lo_before, hi_before = _trace_range(window)
    _drag_the_shaded_window_to(window, start - 5 * (hi_before - lo_before))
    lo, hi = _trace_range(window)
    assert hi < hi_before, "the press landed on the shaded window and moved the trace"
    assert lo == pytest.approx(start, abs=1e-6), "...as far as the first sample, and no further"

    _declare(window, monkeypatch, unit="ms")          # ...and the round trip is not a one-off
    _assert_the_strip_shows_the_whole_recording(window, doc_id)


def test_a_refused_sample_axis_recording_declared_in_microseconds_leaves_samples_behind(
        window, monkeypatch, tmp_path):
    """The recording whose 10-per-row column the 1 Hz floor refused: its axis is SAMPLES, 0 to
    19,999. Declared in microseconds it is 100 kHz from t0 = 5e-8 s. Measured before the fix:
    'Local context · 9× view', a strip still showing 0–1.8 with the curve at 0–19,999 samples
    (2 of 20,000 inside), and a region whose bound was 19,999."""
    doc_id = _show(window, _csv_in_us(tmp_path))
    assert window._session.x_axis(doc_id).unit == "sample"
    assert _strip(window).seen == pytest.approx((0.0, N - 1.0))

    _declare(window, monkeypatch, unit="us")

    assert window._session.x_axis(doc_id).unit == "s"
    end = _extent(window, doc_id)[1]
    assert end == pytest.approx(0.19999, abs=1e-6)
    _assert_the_strip_shows_the_whole_recording(window, doc_id)
    _zoom_in(window, 2)
    lo_before, hi_before = _trace_range(window)
    _drag_the_shaded_window_to(window, end + 5 * (hi_before - lo_before))
    assert _trace_range(window)[1] == pytest.approx(end, abs=1e-6)


# ================================================================================================
# 3. what rides on the strip, and the order of the two slots
# ================================================================================================

@pytest.mark.parametrize("navigator_first", [False, True], ids=["trace_first", "navigator_first"])
def test_the_strip_and_its_measure_band_follow_in_either_connection_order(
        window, monkeypatch, tmp_path, navigator_first):
    """The trace view's own slot and the strip's react to the same signal and nothing may depend
    on which runs first. With Measure on a blockade (samples 5,000–6,000, placed in the clock
    the file was read on), the blue band must end on the same samples on the new clock, inside
    the strip, and the strip must be whole — whichever slot ran first."""
    doc_id = _show(window, _csv_in_ms(tmp_path))
    window._act_measure.trigger()                                  # Analysis ▸ Measure event
    window._trace_view.set_measure_region(10_100.0, 10_120.0)
    QTest.qWait(10)
    nav = window._navigator
    assert tuple(nav._measure_band.getRegion()) == pytest.approx((10_100.0, 10_120.0))
    if navigator_first:
        _navigator_first(window)
    order = _record_the_order(window)

    _declare(window, monkeypatch, unit="ms")

    _assert_the_strip_shows_the_whole_recording(window, doc_id)
    assert window._trace_view.measure_bounds() == (5_000, 6_000), "carried by sample index"
    band = tuple(float(v) for v in nav._measure_band.getRegion())
    assert band == pytest.approx((10.10, 10.12), abs=1e-9)
    assert band == pytest.approx(window._trace_view.measure_x_range(), abs=1e-9)
    seen = _strip(window).seen
    assert seen[0] <= band[0] and band[1] <= seen[1]
    assert order == (["navigator", "trace"] if navigator_first else ["trace", "navigator"]), (
        "the test ran the two slots in the order it claims")


def test_a_segment_on_screen_beside_its_parent_follows_with_it(window, monkeypatch, tmp_path):
    """Segments follow their parent's declaration, each announced by its own signal after the
    parent's — the first arrives while the segment is still on the old clock (200 s long, from
    its own t = 0). Over an overlay of both the trace frames the union, the strip is fed from
    the parent (the selected recording), and both must end on the new clock."""
    parent_id = _show(window, _csv_in_ms(tmp_path))
    segment_id = window._session.create_segment(parent_id, 0, 10_000, name="cut")
    window._on_selection(parent_id, 0)                 # the parent's row: File Info declares THIS
    window._add_document_to_view(segment_id)           # ...then tick the segment's channel
    QTest.qWait(10)
    assert {lay.doc_id for lay in window._session.layers() if lay.visible} == {parent_id,
                                                                               segment_id}

    _declare(window, monkeypatch, unit="ms")

    assert _extent(window, segment_id) == pytest.approx((0.0, 9_999 / 50_000.0), abs=1e-9)
    union = window._trace_view.full_x_extent()
    assert union == pytest.approx((0.0, MS_DECLARED_END), abs=1e-9)
    assert _trace_range(window) == pytest.approx(union, abs=1e-6)
    strip = _strip(window)
    assert strip.label == "Full recording" and not strip.local, strip
    assert strip.seen == pytest.approx(union, abs=1e-6), strip
    assert strip.curve == pytest.approx(_extent(window, parent_id), abs=1e-6), strip
    assert strip.drawn == strip.total, strip
    assert strip.region == pytest.approx(union, abs=1e-6), strip


# ================================================================================================
# 4. the signal when there is nothing for it to do
# ================================================================================================

def test_a_declaration_on_a_recording_that_is_not_drawn_leaves_the_strip_alone(
        window, qtbot, tmp_path):
    """Open in the File Manager but not on the plot: nothing the strip shows moved, so it is not
    re-fed (the trace view makes the same call — its own test asserts it rebuilds nothing). The
    signal for a recording nobody opened, or one just closed, raises nothing either."""
    drawn_id = _show(window, _abf_at_20k(tmp_path))
    hidden_id = window._session.add_document(_csv_in_ms(tmp_path))
    nav = window._navigator
    strip_before = _strip(window)
    calls: list[int] = []
    refresh = nav._refresh
    nav._refresh = lambda: (calls.append(1), refresh())[1]

    window._session.set_sample_rate_declaration(hidden_id, MS)
    assert calls == []
    assert _strip(window) == strip_before

    window._session.set_sample_rate_declaration(
        drawn_id, SampleRateDeclaration("sample_rate", sample_rate_hz=10_000.0))
    assert len(calls) == 1
    _assert_the_strip_shows_the_whole_recording(window, drawn_id)

    with qtbot.capture_exceptions() as caught:
        window._session.sampleRateChanged.emit("a recording nobody opened")
        window._session.close_document(drawn_id)
        window._session.sampleRateChanged.emit(drawn_id)
        QTest.qWait(10)
    assert caught == []
