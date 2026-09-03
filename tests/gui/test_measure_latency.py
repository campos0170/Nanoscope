"""Latency-shaped tests for the Measure panel: coalescing, bounded previews, exact commits.

The bug these pin: the live readout used to recompute SYNCHRONOUSLY ON THE GUI THREAD, once
per ``sigRegionChanged`` (which fires per drag FRAME), over the whole selection with no size
cap — ~1.1 s per frame on a 15 M-sample recording, and "Snap to view" on a zoomed-out trace
walks straight into it. The fix is a debounce plus a bounded preview, and it is only
acceptable because a SAVED row is still measured over every sample. So:

  - a simulated drag produces ONE recompute, not one per frame, and a superseded frame is
    dropped rather than queued;
  - releasing the mouse (and any programmatic placement) settles at once, without waiting
    out the debounce;
  - the live path NEVER calls ``measure_selection`` unbounded — including on the eager
    settle path, which is where "Snap to view" lands;
  - under the bound nothing strides and nothing is flagged;
  - over it the readout is a flagged preview whose blanked statistics read as "—";
  - ``add_current`` stores values equal to a full-precision ``measure_selection``, even when
    the readout it was clicked on was approximate;
  - the flag appears and clears.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import measure as measure_mod
from nanoscope.analysis.conditioning import SIGNAL_BASIS_COLUMN
from nanoscope.analysis.measure import (
    MEASUREMENT_DIAGNOSTIC_KEYS,
    MEASUREMENT_META_COLUMNS,
    MEASUREMENT_SCHEMA,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import measure_panel as mp_mod
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
# Comfortably over _LIVE_MAX_SAMPLES (200 k) so a whole-trace selection really strides, but
# small enough that the EXACT commit these tests force is still fast.
N = 600_000
_BIG = (0, N)                       # the whole trace: what "Snap to view" selects zoomed out
_SMALL = (100_000, 130_000)         # 30 k samples: under the bound, must be untouched

# Every schema column that comes from the SIGNAL rather than from the caller's stamp. A saved
# row must match a full-precision measure_selection on all of them.
_STAT_COLUMNS = tuple(c for c in MEASUREMENT_SCHEMA if c not in MEASUREMENT_META_COLUMNS)


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 4) -> np.ndarray:
    """100 pA open pore + 1.5 pA noise with one long 30 pA blockade and a short deep spike."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    x[150_000:450_000] -= 30.0
    x[300_000:300_050] -= 40.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, arr: np.ndarray, *, name: str = "big.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


class _Spy:
    """Counts ``measure_selection`` calls and records the ``max_samples`` each was given."""

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []
        real = mp_mod.measure_selection

        def wrapper(current, fs, start, stop, **kwargs):
            self.calls.append({"start": int(start), "stop": int(stop),
                               "max_samples": kwargs.get("max_samples", None),
                               "amplitude_method": kwargs.get("amplitude_method")})
            return real(current, fs, start, stop, **kwargs)

        monkeypatch.setattr(mp_mod, "measure_selection", wrapper)

    def __len__(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()

    @property
    def bounds(self) -> list[tuple[int, int]]:
        return [(c["start"], c["stop"]) for c in self.calls]


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def panel(window, tmp_path):
    """A Measure panel armed on a 600 k-sample recording, with the seed measurement done."""
    doc_id, _ch = _install(window, _doc(tmp_path, _trace()))
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    return mp, doc_id


def _place(window, doc_id: str, i0: int, i1: int) -> None:
    """Put the cursors on sample span [i0, i1) — the real programmatic placement path."""
    vals = window._session.x_axis(doc_id).values
    x1 = float(vals[i1]) if i1 < vals.size else float(vals[-1]) + (
        float(vals[-1]) - float(vals[-2]))
    window._trace_view.set_measure_region(float(vals[i0]), x1)


def _drag(window, doc_id: str, frames: list[tuple[int, int]]) -> None:
    """Move the region's right edge once per frame, exactly as a mouse drag does.

    Deliberately NOT ``set_measure_region``: pyqtgraph's ``setRegion`` also calls
    ``lineMoveFinished``, i.e. it SETTLES, and coalescing is entirely about the frames
    before the settle. Driving ``lines[1].setValue`` reproduces the real mid-drag path —
    ``InfiniteLine.setPos`` emits only ``sigPositionChanged``, which the region turns into
    ``sigRegionChanged`` — so the panel sees the same stream a real drag produces, and reads
    the same authoritative bounds back off the TraceView rather than trusting the payload."""
    region = window._trace_view._measure_region
    assert region is not None, "the Measure tool must be armed to simulate a drag"
    vals = window._session.x_axis(doc_id).values
    for i0, i1 in frames:
        region.lines[0].setValue(float(vals[i0]))
        region.lines[1].setValue(float(vals[min(i1, vals.size - 1)]))


# ================================================================================================
# (a) coalescing: a drag is not one recompute per frame
# ================================================================================================

def test_a_drag_does_not_recompute_per_frame(panel, window, monkeypatch, qtbot):
    mp, doc_id = panel
    spy = _Spy(monkeypatch)

    _drag(window, doc_id, [(0, 100_000 + 500 * i) for i in range(30)])
    assert len(spy) == 0, "a drag frame must schedule a recompute, never perform one"
    assert mp._recompute_timer.isActive()

    qtbot.wait(mp_mod._RECOMPUTE_DEBOUNCE_MS * 4)
    assert len(spy) == 1, "30 frames must collapse to ONE recompute once the edges settle"


def test_a_superseded_recompute_is_dropped_not_queued(panel, window, monkeypatch, qtbot):
    """The single-shot timer restarts on every frame, so only the LAST selection is measured."""
    mp, doc_id = panel
    _place(window, doc_id, *_SMALL)
    spy = _Spy(monkeypatch)

    _drag(window, doc_id, [(10_000, 20_000), (10_000, 30_000), (10_000, 40_000)])
    qtbot.wait(mp_mod._RECOMPUTE_DEBOUNCE_MS * 4)

    assert len(spy) == 1
    assert mp.current_measurement()["n_samples"] == 30_000   # the last frame, not the first
    assert spy.bounds == [(10_000, 40_000)]


def test_pausing_mid_drag_still_reads_live(panel, window, monkeypatch, qtbot):
    """Coalescing must not mean "no readout until you let go": each pause produces one."""
    mp, doc_id = panel
    spy = _Spy(monkeypatch)
    for stop in (200_000, 300_000, 400_000):
        _drag(window, doc_id, [(0, stop - 1_000), (0, stop)])
        qtbot.wait(mp_mod._RECOMPUTE_DEBOUNCE_MS * 4)
    assert len(spy) == 3
    assert mp.current_measurement()["n_samples"] == 400_000


def test_releasing_the_mouse_settles_immediately(panel, window, monkeypatch, qtbot):
    """sigMeasureFinished must not wait out the debounce, and must cancel the pending frame."""
    mp, doc_id = panel
    spy = _Spy(monkeypatch)

    _drag(window, doc_id, [(50_000, 60_000)])
    assert len(spy) == 0
    window._trace_view.sigMeasureFinished.emit(50_000, 60_000)

    assert len(spy) == 1, "the settle must compute at once"
    assert not mp._recompute_timer.isActive(), "the pending frame must be cancelled"
    qtbot.wait(mp_mod._RECOMPUTE_DEBOUNCE_MS * 4)
    assert len(spy) == 1, "...and must not then fire a second time"


def test_a_pending_recompute_is_dropped_when_the_tool_is_switched_off(panel, window,
                                                                     monkeypatch, qtbot):
    mp, doc_id = panel
    spy = _Spy(monkeypatch)
    _drag(window, doc_id, [(0, 200_000)])
    mp.set_active(False)
    qtbot.wait(mp_mod._RECOMPUTE_DEBOUNCE_MS * 4)
    assert len(spy) == 0
    assert mp.current_measurement() is None


def test_a_programmatic_placement_measures_once(panel, window, monkeypatch):
    """Snap / go-to place the region with setRegion, which emits changed AND finished. That
    must still be ONE measurement, not one per signal."""
    mp, doc_id = panel
    spy = _Spy(monkeypatch)
    _place(window, doc_id, *_SMALL)
    assert len(spy) == 1
    assert not mp._recompute_timer.isActive()


# ================================================================================================
# (b) the live path is bounded — always, including the eager settle
# ================================================================================================

def test_every_live_measurement_passes_the_bound(panel, window, monkeypatch, qtbot):
    mp, doc_id = panel
    spy = _Spy(monkeypatch)

    _place(window, doc_id, *_SMALL)                       # settle path
    _drag(window, doc_id, [(0, N)])                       # debounce path
    qtbot.wait(mp_mod._RECOMPUTE_DEBOUNCE_MS * 4)
    mp.set_method("mode")                                 # combo path

    assert len(spy) >= 3
    assert all(c["max_samples"] == mp_mod._LIVE_MAX_SAMPLES for c in spy.calls), \
        "a live recompute must never run unbounded"


def test_snap_to_view_on_a_fully_zoomed_out_trace_is_bounded(panel, window, monkeypatch):
    """THE trap in the original bug: snap sets the region to the whole visible window, which
    zoomed out is the whole recording, and the settle path then measures it eagerly."""
    mp, doc_id = panel
    vals = window._session.x_axis(doc_id).values
    window._trace_view.sigVisibleRangeChanged.emit(float(vals[0]), float(vals[-1]))
    spy = _Spy(monkeypatch)

    mp._snap_to_view()

    assert len(spy) >= 1
    assert all(c["max_samples"] == mp_mod._LIVE_MAX_SAMPLES for c in spy.calls)
    stats = mp.current_measurement()
    assert stats is not None and stats["approximate"] is True
    assert stats["n_measured"] <= mp_mod._LIVE_MAX_SAMPLES
    assert stats["n_samples"] > mp_mod._LIVE_MAX_SAMPLES     # ...over the whole span


def test_no_live_measurement_ever_reads_more_than_the_bound(panel, window, qtbot):
    """The deterministic stand-in for "the preview is fast": whatever the selection, the
    live path reads at most _LIVE_MAX_SAMPLES samples."""
    mp, doc_id = panel
    for span in (_SMALL, (0, 250_000), (0, N)):
        _place(window, doc_id, *span)
        assert mp.current_measurement()["n_measured"] <= mp_mod._LIVE_MAX_SAMPLES


def test_below_the_bound_the_live_readout_is_exact_and_unflagged(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_SMALL)

    stats = mp.current_measurement()
    assert stats["n_samples"] == 30_000
    assert stats["approximate"] is False and stats["stride"] == 1
    assert not mp.is_preview()
    assert mp.approx_note() == ""
    for key in ("min", "max", "ptp"):
        assert np.isfinite(stats[key])
        assert mp.readout_value(key) not in ("", "—")


def test_above_the_bound_the_live_readout_strides(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)

    stats = mp.current_measurement()
    assert stats["approximate"] is True
    assert stats["stride"] == 3 and stats["n_measured"] <= mp_mod._LIVE_MAX_SAMPLES
    assert stats["n_samples"] == N                     # geometry EXACT
    assert stats["dwell_s"] == pytest.approx(N / FS, rel=1e-12)


# ================================================================================================
# (d) the state is visible
# ================================================================================================

def test_the_preview_indicator_appears_and_says_what_it_withheld(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)

    note = mp.approx_note()
    assert note, "an approximate readout must announce itself"
    assert "1:3" in note                               # the sampling ratio
    assert f"{N:,}" in note                            # ...of how many samples
    assert "every sample" in note                      # ...and the promise for the saved row
    for key in ("min", "max", "ptp"):
        assert key in note
        assert mp.readout_value(key) == "—", f"{key} must read as withheld, not as a number"


def test_the_preview_indicator_clears_when_the_selection_shrinks(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    assert mp.is_preview() and mp.approx_note()

    _place(window, doc_id, *_SMALL)
    assert not mp.is_preview()
    assert mp.approx_note() == ""
    assert mp._approx_label.isVisible() is False


def test_an_extreme_amplitude_method_blanks_the_amplitude_in_a_preview(panel, window):
    """'min' takes the deepest sample; a subsample cannot find it, so the readout says so
    rather than showing a systematically shallow blockade."""
    mp, doc_id = panel
    mp.set_method("min")
    _place(window, doc_id, *_BIG)

    assert mp.is_preview()
    for key in ("level", "amplitude", "frac_amplitude", "snr"):
        assert mp.readout_value(key) == "—", key
    assert "amplitude" in mp.approx_note()
    # the statistics that DO converge are still shown
    for key in ("mean", "median", "std", "baseline"):
        assert mp.readout_value(key) not in ("", "—"), key


def test_the_indicator_clears_when_the_tool_is_switched_off(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    assert mp.is_preview()
    mp.set_active(False)
    assert mp.approx_note() == ""
    assert not mp.is_preview()


# ================================================================================================
# (c) INVARIANT: a saved measurement is always exact
# ================================================================================================

def test_add_current_stores_exact_values_after_an_approximate_preview(panel, window):
    """The invariant, end to end: the readout was a 1:3 preview; the stored row must equal a
    full-precision measure_selection over EVERY sample of the same span."""
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    assert mp.current_measurement()["approximate"] is True

    assert mp.add_current() is True
    row = mp.measurements_df.iloc[-1]

    chan = window._session.calibrated(doc_id, 0)
    truth = measure_mod.measure_selection(
        np.asarray(chan.data), chan.sample_rate_hz, int(row["start_idx"]),
        int(row["end_idx"]), amplitude_method=str(row["amplitude_method"]))

    for col in _STAT_COLUMNS:
        got, want = row[col], truth[col]
        if isinstance(want, float) and np.isnan(want):
            assert np.isnan(float(got)), col
        elif isinstance(want, str):
            assert str(got) == want, col
        else:
            assert float(got) == float(want), f"{col}: stored {got!r} != exact {want!r}"


@pytest.mark.parametrize("method", ["median", "mean", "min", "max", "mode"])
def test_a_saved_row_is_never_blank_whatever_the_preview_withheld(panel, window, method):
    mp, doc_id = panel
    mp.set_method(method)
    _place(window, doc_id, *_BIG)
    assert mp.current_measurement()["suppressed"], "this preview did withhold something"

    assert mp.add_current() is True
    row = mp.measurements_df.iloc[-1]
    for col in ("min", "max", "ptp", "level", "amplitude", "frac_amplitude", "snr"):
        assert np.isfinite(float(row[col])), f"{col} was saved blank"


def test_add_current_leaves_the_readout_showing_what_it_stored(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    assert mp.is_preview()

    mp.add_current()
    assert not mp.is_preview(), "after saving, the readout must be the exact values"
    assert mp.approx_note() == ""
    row = mp.measurements_df.iloc[-1]
    assert mp.current_measurement()["max"] == pytest.approx(float(row["max"]))
    assert mp.readout_value("max") not in ("", "—")


def test_a_saved_row_carries_no_preview_diagnostics(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    mp.add_current()
    # The 32 schema columns plus the additive `signal_basis` stamp the table has carried
    # since 2026-09-02 — and, the point of this test, none of the preview diagnostics.
    assert list(mp.measurements_df.columns) == [*MEASUREMENT_SCHEMA, SIGNAL_BASIS_COLUMN]
    assert not set(MEASUREMENT_DIAGNOSTIC_KEYS) & set(mp.measurements_df.columns)


def test_an_exact_readout_is_not_recomputed_on_commit(panel, window, monkeypatch):
    """Below the bound the readout is already exact, so saving must add no work at all —
    the pre-change behaviour, preserved."""
    mp, doc_id = panel
    _place(window, doc_id, *_SMALL)
    spy = _Spy(monkeypatch)
    assert mp.add_current() is True
    assert len(spy) == 0


def test_commit_recomputes_exactly_once_when_the_readout_was_a_preview(panel, window,
                                                                      monkeypatch):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    spy = _Spy(monkeypatch)
    assert mp.add_current() is True
    assert len(spy) == 1
    assert spy.calls[0]["max_samples"] is None, "the commit measurement must be unbounded"
    assert spy.bounds == [_BIG]


def test_the_busy_indicator_is_raised_and_always_dropped_again(panel, window, monkeypatch):
    """A slow exact commit must show a wait cursor rather than silently freeze — and must
    never leave one behind, which would wedge the cursor for the rest of the session."""
    from PySide6 import QtWidgets

    mp, doc_id = panel
    monkeypatch.setattr(mp_mod, "_BUSY_CURSOR_SAMPLES", 1_000)   # make _BIG count as slow
    seen: list[bool] = []
    real = mp_mod.measure_selection

    def wrapper(*a, **kw):
        seen.append(QtWidgets.QApplication.overrideCursor() is not None)
        return real(*a, **kw)

    _place(window, doc_id, *_BIG)
    monkeypatch.setattr(mp_mod, "measure_selection", wrapper)

    assert QtWidgets.QApplication.overrideCursor() is None
    assert mp.add_current() is True
    assert seen == [True], "the exact commit must run under a wait cursor"
    assert QtWidgets.QApplication.overrideCursor() is None, "the cursor leaked"
    assert "Added measurement" in mp._status.text()


def test_exact_measurement_is_public_and_unbounded(panel, window, monkeypatch):
    mp, doc_id = panel
    _place(window, doc_id, *_BIG)
    spy = _Spy(monkeypatch)
    out = mp.exact_measurement()
    assert out is not None and out["approximate"] is False and out["stride"] == 1
    assert spy.calls[0]["max_samples"] is None


def test_mark_missed_characterizes_raw_samples_not_the_preview(panel, window, qtbot):
    """The other persisting path. Its owner reads the cursor bounds off the TraceView and
    characterizes the raw span itself, so the preview never reaches the event set — the
    appended event's geometry must match the cursors exactly and its amplitude must be the
    real 30 pA, not something a 1-in-3 subsample suggested."""
    from nanoscope.analysis.events import EventSet, empty_df
    from nanoscope.core.analysis import AnalysisResult

    mp, doc_id = panel
    ap = window._analysis_panel
    # A "missed" event needs a detection to be added TO; an empty one is the cleanest.
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0),
        payload=EventSet(empty_df(), {"polarity": -1})))
    ap.set_selection(doc_id, 0)
    qtbot.wait(10)

    _place(window, doc_id, 150_000, 450_000)
    assert mp.current_measurement()["approximate"] is True    # the readout IS a preview

    new_id = ap.add_current_selection_as_event()
    assert new_id is not None
    added = ap.current_events_df().iloc[-1]
    assert int(added["start_idx"]) == 150_000
    assert int(added["end_idx"]) == 450_000
    assert float(added["dwell_s"]) == pytest.approx(300_000 / FS, rel=1e-9)
    assert float(added["depth_pA"]) == pytest.approx(30.0, abs=2.0)
