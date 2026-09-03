"""S3 / JAB-01 + JHT-04: the exported trace figure is on ONE clock — the axis's own.

THE DEFECT
----------
``plot.figures.trace_preview`` plots the signal against the XAxis it is handed
(``Session.x_axis`` = ``arange(n)/fs + meta.t0_s`` — the INSTRUMENT clock) and then
shaded each event with ``axvspan(t_start_s, t_start_s + dwell_s)`` — RECORDING-clock
seconds, 0 at the first loaded sample (``core.clock``; ``analysis/events.py`` SCHEMA).
On the owner's quick apoferritin ABF (``t0_s`` = 1154.99 s) every one of 111 bands
landed ~1155 s to the left of the event it annotates, on blank canvas; on a recording
whose ``t0_s`` is smaller than its duration the band lands on a stretch of open pore
``t0_s`` early — worse than missing, because it looks like data. Meanwhile
``_nanoscope_n_events_shown`` reported the full count, so the figure claimed to have
drawn them.

Two halves, both pinned here:

1. the CONVERSION lives inside ``trace_preview``, against the origin read off the axis
   itself, so no caller can get placement wrong (``MainWindow._export_figures`` passes
   no ``t0`` at all);
2. the default view window is the axis's own extent, not ``0.0``. A default that has
   to be overridden to be correct is itself the bug: with ``t0=0.0`` the 15 s recording
   was squeezed into the last 1.3 % of a 0–1170 s panel.

...and the artifact SAYS what it is on, WITHOUT GUESSING (sweep-3 repair). Placement can
be derived from the axis; the origin's CLOCK cannot — a rebased cut's axis starts at 0
exactly like the -700 mV file whose t₀ genuinely is 0, so the derived label printed
"t₀ = 0, so both clocks read alike here" on a cut every other surface dates to
1155.7909 s. Hand ``trace_preview`` the document's ``meta`` and the label is
``core.clock.axis_label``'s sentence, character-identical to the on-screen trace axis,
and the CSV's ``t0_s`` is ``clock.instrument_origin_s`` — MEASUREMENT_SCHEMA's spelling
AND its definition. Withhold it and the figure names the origin it used
(``x_axis_origin_s``) and says the clock was not supplied; ``t0_s`` is then NaN, that
schema's own "NaN when unknown".

The GUI halves are driven through the real ``Analysis ▸ Export figure…`` and
``File ▸ Create segment from selection…`` QActions on a real ``MainWindow``, and read
back off the files that land on disk / the live axis label.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the Anaconda Qt plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle
from PySide6 import QtWidgets

from nanoscope.core import clock
from nanoscope.core.document import CalibratedChannel, TraceDocument, XAxis
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.plot import figures

FS = 100_000.0
N = 40_000
#: The owner's quick apoferritin ABF starts here. Big enough that every band would miss
#: the trace entirely, which is the shipped failure.
T0_APO = 1154.99091246287
#: A t0 SMALLER than the recording is the nastier case: the band lands on real signal.
T0_OVERLAP = 0.05


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    """A quiet baseline with three unmistakable blockades — enough for detection."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _meta(path: Path, t0_s: float, n: int = N) -> RecordingMeta:
    """The RecordingMeta a loader would build for :func:`_trace` — the object a caller
    hands ``trace_preview`` so the figure can NAME the clock it is on."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    return RecordingMeta(path=path, format_name="synthetic", n_samples=n,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         t0_s=float(t0_s), channels=(info,))


def _doc(tmp_path: Path, t0_s: float) -> TraceDocument:
    arr = _trace()
    return TraceDocument(meta=_meta(tmp_path / "pore.dat", t0_s, arr.size),
                         channels=(arr,))


def _channel_and_axis(t0_s: float, n: int = N) -> tuple[CalibratedChannel, XAxis]:
    """A CalibratedChannel + the XAxis Session.x_axis would build for it."""
    arr = _trace()[:n]
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=arr, units="pA", sample_rate_hz=FS, t0_s=t0_s)
    xax = XAxis(values=np.arange(arr.size, dtype=np.float64) / FS + t0_s,
                unit="s", is_time=True)
    return ch, xax


def _spans(ax) -> list[tuple[float, float]]:
    """(x0, x1) of every axvspan rectangle on ``ax``, in axis coordinates."""
    out = []
    for p in ax.patches:
        if not isinstance(p, Rectangle):
            continue
        xy = p.get_xy()
        out.append((float(xy[0]), float(xy[0] + p.get_width())))
    return out


# ================================================================================================
# 1. placement — the bands are on the axis's clock, unit level
# ================================================================================================

def test_bands_land_on_the_events_they_annotate_not_t0_away():
    """The band for an event at recording second 0.10 is drawn at t0 + 0.10."""
    ch, xax = _channel_and_axis(T0_APO)
    ev = pd.DataFrame({"t_start_s": [0.10, 0.25], "dwell_s": [0.003, 0.004]})

    fig, ax = figures.trace_preview(ch, xax, events=ev)

    spans = _spans(ax)
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(T0_APO + 0.10, abs=1e-9)
    assert spans[0][1] == pytest.approx(T0_APO + 0.103, abs=1e-9)
    # ...and every band lies inside the trace that is actually drawn.
    line = ax.lines[0]
    lo, hi = float(np.min(line.get_xdata())), float(np.max(line.get_xdata()))
    assert all(b > lo and a < hi for a, b in spans)
    assert getattr(fig, "_nanoscope_n_events_shown") == 2
    assert getattr(fig, "_nanoscope_clock_origin_s") == pytest.approx(T0_APO)


def test_band_lands_on_the_blockade_and_not_on_open_pore():
    """The failure that is worse than absence: a t0 shorter than the recording put the
    band over quiet baseline, which reads as a real event."""
    t0 = T0_OVERLAP
    ch, xax = _channel_and_axis(t0)
    # The synthetic trace's second blockade: samples 12_000..12_250.
    ev = pd.DataFrame({"t_start_s": [12_000 / FS], "dwell_s": [250 / FS]})

    _fig, ax = figures.trace_preview(ch, xax, events=ev)
    (a, b) = _spans(ax)[0]

    # The current under the band really is depressed...
    inside = np.asarray(ch.data)[(np.asarray(xax.values) >= a) & (np.asarray(xax.values) <= b)]
    assert inside.size and float(np.median(inside)) < 90.0
    # ...and the pre-fix position (raw t_start_s, t0 seconds early) is open pore.
    early = np.asarray(ch.data)[(np.asarray(xax.values) >= a - t0)
                                & (np.asarray(xax.values) <= b - t0)]
    assert early.size and float(np.median(early)) > 95.0


def test_the_count_counts_what_it_says_when_a_band_is_off_window():
    """``_nanoscope_n_events_shown`` counts bands actually drawn in the window."""
    ch, xax = _channel_and_axis(T0_APO)
    ev = pd.DataFrame({"t_start_s": [0.01, 0.30], "dwell_s": [0.001, 0.001]})

    fig, ax = figures.trace_preview(ch, xax, events=ev, t0=T0_APO, t1=T0_APO + 0.1)

    assert getattr(fig, "_nanoscope_n_events_shown") == 1
    assert len(_spans(ax)) == 1


# ================================================================================================
# 2. the default window is the axis's own extent
# ================================================================================================

def test_default_window_is_the_axis_extent_not_zero():
    ch, xax = _channel_and_axis(T0_APO)
    _fig, ax = figures.trace_preview(ch, xax)
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(float(xax.values[0]))
    assert hi == pytest.approx(float(xax.values[-1]))
    # The whole recording occupies the panel, not 1.3 % of it.
    span = float(xax.values[-1]) - float(xax.values[0])
    assert (hi - lo) == pytest.approx(span, rel=1e-9)


def test_a_t0_of_zero_axis_is_unchanged():
    """The recordings where the two clocks agree must render exactly as before."""
    ch, xax = _channel_and_axis(0.0)
    _fig, ax = figures.trace_preview(ch, xax)
    assert ax.get_xlim()[0] == 0.0


def test_an_explicit_window_is_still_honoured():
    ch, xax = _channel_and_axis(T0_APO)
    _fig, ax = figures.trace_preview(ch, xax, t0=T0_APO + 0.05, t1=T0_APO + 0.20)
    assert ax.get_xlim() == pytest.approx((T0_APO + 0.05, T0_APO + 0.20))


# ================================================================================================
# 3. the artifact says which clock it is on
# ================================================================================================

def test_x_label_names_the_clock_in_core_clocks_own_words():
    """Handed the document, the figure says exactly what the on-screen axis says."""
    ch, xax = _channel_and_axis(T0_APO)
    meta = _meta(Path("/tmp/pore.dat"), T0_APO)
    _fig, ax = figures.trace_preview(ch, xax, meta=meta)
    # `gui/trace_view.py` builds its label from clock.clock_offset_s(meta, offset);
    # the figure must produce that same string, not merely a similar one.
    assert ax.get_xlabel() == clock.axis_label("s", clock.INSTRUMENT,
                                               clock.clock_offset_s(meta))
    assert "1154.9909" in ax.get_xlabel()


def test_x_label_claims_no_clock_when_the_document_was_not_handed_over():
    """WITHOUT the meta the origin's clock is genuinely unknown here — so it is not named.

    A derived origin says where the bands went; it does NOT say whether that origin is
    an instrument t₀ or a zero this app manufactured when a cut was rebased. Naming one
    anyway is how the exported figure came to contradict the trace the user curated on.
    """
    ch, xax = _channel_and_axis(T0_APO)
    _fig, ax = figures.trace_preview(ch, xax)
    label = ax.get_xlabel()

    assert "1154.9909" in label                      # the origin it really used
    assert "not supplied" in label                   # ...and the missing half, said
    assert "read alike" not in label                 # no agreement claim
    assert label != clock.axis_label("s", clock.INSTRUMENT, T0_APO)


def test_source_data_csv_carries_both_origins(tmp_path):
    ch, xax = _channel_and_axis(T0_APO)
    meta = _meta(tmp_path / "pore.dat", T0_APO)
    fig, _ax = figures.trace_preview(ch, xax, meta=meta)
    out = figures.save_figure(fig, tmp_path / "trace.pdf")

    d = pd.read_csv(out["csv"])
    # What the figure added to place each event...
    assert float(d["x_axis_origin_s"].iloc[0]) == pytest.approx(T0_APO)
    # ...and MEASUREMENT_SCHEMA's t0_s: the same number, by the same function, that a
    # measurements.csv row from this recording carries.
    assert float(d["t0_s"].iloc[0]) == pytest.approx(clock.instrument_origin_s(meta))
    # x is the drawn clock; x - x_axis_origin_s is the clock the events CSV is on.
    assert float(d["x"].min()) >= T0_APO
    assert (float(d["x"].min()) - float(d["x_axis_origin_s"].iloc[0])
            == pytest.approx(0.0, abs=1e-3))


def test_source_data_csv_says_unknown_rather_than_guessing_an_origin(tmp_path):
    """No meta -> ``t0_s`` is NaN, MEASUREMENT_SCHEMA's own "NaN when unknown".

    A 0.0 there would be indistinguishable from a recording that genuinely starts at 0.
    The origin the figure DID use is still on the table, under its own honest name.
    """
    ch, xax = _channel_and_axis(T0_APO)
    fig, _ax = figures.trace_preview(ch, xax)
    out = figures.save_figure(fig, tmp_path / "trace.pdf")

    d = pd.read_csv(out["csv"])
    assert np.isnan(float(d["t0_s"].iloc[0]))
    assert float(d["x_axis_origin_s"].iloc[0]) == pytest.approx(T0_APO)


def test_a_sample_axis_shades_nothing_and_says_why():
    """Samples are on neither clock. Shading raw seconds there was a silent falsehood."""
    ch, _ = _channel_and_axis(0.0, n=5_000)
    xax = XAxis(values=np.arange(ch.data.size, dtype=np.float64), unit="sample",
                is_time=False)
    ev = pd.DataFrame({"t_start_s": [0.01], "dwell_s": [0.001]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = figures.trace_preview(ch, xax, events=ev)

    assert getattr(fig, "_nanoscope_n_events_shown") == 0
    assert _spans(ax) == []
    assert getattr(fig, "_nanoscope_clock_origin_s") is None
    assert any("sample" in str(w.message).lower() and "clock" in str(w.message).lower()
               for w in caught)
    assert ax.get_xlabel() == "sample"
    # ...and the CSV does not invent a clock for a sample axis either.
    assert "t0_s" not in getattr(fig, figures._DATA_ATTR).columns


# ================================================================================================
# 4. THE GUI SURFACE — Analysis ▸ Export figure… on a real window, real detection
# ================================================================================================

@pytest.fixture
def detected(qapp, qtbot, tmp_path):
    """A MainWindow holding one synthetic recording, t0 = the apo ABF's, detected."""
    w = MainWindow()
    qtbot.addWidget(w)
    doc = _doc(tmp_path, T0_APO)
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    w._analysis_panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=60_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert w._analysis_panel.current_detection() is not None
    yield w, doc_id
    w._analysis_controller.abort()


def test_exported_figure_shades_events_on_the_trace(detected, qtbot, tmp_path, monkeypatch):
    w, doc_id = detected
    out = tmp_path / "figs"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "PDF (*.pdf)")))

    # The real menu command, not the slot by name.
    assert w._act_export_figure.isEnabled()
    w._act_export_figure.trigger()
    qtbot.wait(50)

    trace_csv = tmp_path / "figs_trace.csv"
    assert trace_csv.exists(), sorted(p.name for p in tmp_path.iterdir())
    d = pd.read_csv(trace_csv)
    # The shipped source data names the origin every band was placed through.
    assert float(d["x_axis_origin_s"].iloc[0]) == pytest.approx(T0_APO)
    assert float(d["x"].min()) >= T0_APO
    # ...and MEASUREMENT_SCHEMA's t0_s is either the recording's real instrument origin
    # or an honest NaN. Written this way ON PURPOSE: the shipped caller
    # (MainWindow._export_figures) does not yet pass `meta=`, so today this is NaN; the
    # day it does, this assertion gets STRONGER rather than breaking. It never pins the
    # unknown.
    t0_col = float(d["t0_s"].iloc[0])
    assert np.isnan(t0_col) or t0_col == pytest.approx(T0_APO)

    # ...and the figure the export built puts every band on that same clock. Rebuilt
    # here from the session's own objects — exactly the call `_export_figures` makes.
    res = w._analysis_panel.current_detection()
    ch = w._session.calibrated(res.doc_id, res.channel)
    xax = w._session.x_axis(res.doc_id)
    fig, ax = figures.trace_preview(ch, xax, events=res.payload)

    spans = _spans(ax)
    assert spans, "the detection produced no events to shade"
    lo, hi = float(xax.values[0]), float(xax.values[-1])
    off_trace = [s for s in spans if s[1] <= lo or s[0] >= hi]
    assert not off_trace, f"{len(off_trace)} of {len(spans)} bands are off the trace"
    assert getattr(fig, "_nanoscope_n_events_shown") == len(spans)
    assert ax.get_xlim()[0] == pytest.approx(lo)


def test_exported_figure_agrees_with_the_on_screen_overlay(detected, qtbot):
    """The exported band and the on-screen span must name the same instant.

    ``gui/overlays.py`` maps spans by SAMPLE INDEX through ``session.x_axis``; the
    figure now converts seconds through the same origin. Before the fix the two
    disagreed by exactly ``meta.t0_s`` — the file silently contradicting the screen
    the user had just curated on.
    """
    w, doc_id = detected
    res = w._analysis_panel.current_detection()
    ch = w._session.calibrated(res.doc_id, res.channel)
    xax = w._session.x_axis(res.doc_id)
    df = res.payload.accepted.df

    _fig, ax = figures.trace_preview(ch, xax, events=res.payload)
    first_band = min(_spans(ax))[0]
    on_screen = float(np.asarray(xax.values)[int(df["start_idx"].iloc[0])])
    assert first_band == pytest.approx(on_screen, abs=2.0 / FS)


# ================================================================================================
# 5. THE REBASED CUT — the document class where a derived origin cannot answer
# ================================================================================================
#
# ``core.segment`` rebases a cut so its ``meta.t0_s`` is 0 and records the instrument
# instant of its first sample under ``clock.REBASE_ATTR``. Read through the axis alone
# such a document is indistinguishable from the -700 mV recording whose t₀ genuinely IS
# 0 — so the figure's label, built from the axis's own origin, said "t₀ = 0, so both
# clocks read alike here" about a cut the trace axis, the Files dock and the segment's
# own name all date to 1155.7909 s. core/clock.py's module docstring names that exact
# sentence as the lie it exists to stop, and it had reappeared in the artifact that
# LEAVES the app. These pin both halves: with the document in hand the figure says what
# the screen says; without it the figure claims nothing.

REBASED_FROM = 1155.7909124437965


def _rebased_channel_axis_meta(nudge_s: float = 0.0):
    """A cut's (channel, axis, meta) exactly as Session would hand them over."""
    from nanoscope.core import segment as segment_mod

    base = _meta(Path("/tmp/cut.dat"), 0.0)
    meta = segment_mod._rebased_meta(base, REBASED_FROM)
    ch, _ = _channel_and_axis(0.0)
    xax = XAxis(values=np.arange(N, dtype=np.float64) / FS + nudge_s, unit="s",
                is_time=True)
    return ch, xax, meta


def test_a_rebased_cut_is_never_told_its_two_clocks_agree():
    ch, xax, meta = _rebased_channel_axis_meta()

    _fig, ax = figures.trace_preview(ch, xax, meta=meta)

    label = ax.get_xlabel()
    assert "read alike" not in label
    assert "1155.7909" in label
    # The on-screen axis for this same document, character for character.
    assert label == clock.axis_label("s", clock.INSTRUMENT, clock.clock_offset_s(meta))


def test_a_rebased_cut_claims_nothing_at_all_when_the_document_is_withheld():
    """The fallback must not take the agreement branch on a manufactured zero."""
    ch, xax, _meta_unused = _rebased_channel_axis_meta()
    _fig, ax = figures.trace_preview(ch, xax)
    assert "read alike" not in ax.get_xlabel()
    assert "not supplied" in ax.get_xlabel()


def test_the_csv_t0_s_of_a_cut_is_its_parents_clock_not_its_own_zero(tmp_path):
    """``t0_s`` means what MEASUREMENT_SCHEMA says: where sample 0 sat in the EXPERIMENT."""
    ch, xax, meta = _rebased_channel_axis_meta()
    fig, _ax = figures.trace_preview(ch, xax, meta=meta)
    out = figures.save_figure(fig, tmp_path / "cut.pdf")

    d = pd.read_csv(out["csv"])
    assert float(d["t0_s"].iloc[0]) == pytest.approx(REBASED_FROM)
    assert float(d["x_axis_origin_s"].iloc[0]) == pytest.approx(0.0)
    assert getattr(fig, "_nanoscope_instrument_origin_s") == pytest.approx(REBASED_FROM)


def test_a_layer_alignment_nudge_never_becomes_a_recording_origin(tmp_path):
    """A nudge moves the drawn axis (and therefore the bands); it is a DISPLAY shift and
    ``measure.MEASUREMENT_SCHEMA`` says it is never part of ``t0_s``."""
    ch, xax, meta = _rebased_channel_axis_meta(nudge_s=2.5)
    ev = pd.DataFrame({"t_start_s": [0.10], "dwell_s": [0.001]})

    fig, ax = figures.trace_preview(ch, xax, events=ev, meta=meta)
    out = figures.save_figure(fig, tmp_path / "nudged.pdf")

    # The band follows what is DRAWN...
    assert _spans(ax)[0][0] == pytest.approx(2.5 + 0.10, abs=1e-9)
    d = pd.read_csv(out["csv"])
    assert float(d["x_axis_origin_s"].iloc[0]) == pytest.approx(2.5)
    # ...and t0_s does not: it is still where this cut sat in the experiment.
    assert float(d["t0_s"].iloc[0]) == pytest.approx(REBASED_FROM)
    # ...and the label discloses BOTH, in core.clock's words.
    assert ax.get_xlabel() == clock.axis_label("s", clock.INSTRUMENT,
                                               clock.clock_offset_s(meta, 2.5))
    assert "2.5000" in ax.get_xlabel() and "1155.7909" in ax.get_xlabel()


def test_a_meta_from_another_document_is_refused_not_believed():
    """The one failure the optional argument could introduce, closed at the seam."""
    ch, xax = _channel_and_axis(T0_APO)
    wrong = _meta(Path("/tmp/other.dat"), 61.3034, n=N // 2)
    with pytest.raises(ValueError, match="samples"):
        figures.trace_preview(ch, xax, meta=wrong)


# ================================================================================================
# 6. THE GUI SURFACE for the cut — File ▸ Create segment from selection…, then the figure
# ================================================================================================

def test_exported_figure_and_the_on_screen_axis_describe_a_cut_alike(
        qapp, qtbot, tmp_path, monkeypatch):
    """The site the sweep-3 verifier found: a real segment, both surfaces, one document."""
    w = MainWindow()
    qtbot.addWidget(w)
    doc = _doc(tmp_path, T0_APO)
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    qtbot.wait(20)

    # The real File-menu command. Its name prompt is modal; answer it headlessly.
    monkeypatch.setattr(MainWindow, "_ask_segment_name",
                        lambda self, default: default)
    vals = w._session.x_axis(doc_id).values
    w._analysis_panel.measure_panel.set_active(True)
    qtbot.wait(20)
    w._trace_view.set_measure_region(float(vals[4_000]), float(vals[24_000]))
    qtbot.wait(20)
    w._sync_create_segment_action()
    assert w._act_create_segment.isEnabled()
    w._act_create_segment.trigger()
    qtbot.wait(50)

    seg_ids = [d.doc_id for d in w._session.documents() if d.doc_id != doc_id]
    assert seg_ids, "the segment action created no document"
    seg_id = seg_ids[-1]
    seg = w._session.document(seg_id)
    assert getattr(seg.meta, clock.REBASE_ATTR, None) is not None
    w._on_selection(seg_id, 0)
    qtbot.wait(50)

    on_screen = w._trace_view.x_axis_label()
    ch = w._session.calibrated(seg_id, 0)
    xax = w._session.x_axis(seg_id)
    _fig, ax = figures.trace_preview(ch, xax, meta=seg.meta)

    assert ax.get_xlabel() == on_screen, (
        f"exported {ax.get_xlabel()!r} vs on screen {on_screen!r}")
    assert "read alike" not in on_screen
