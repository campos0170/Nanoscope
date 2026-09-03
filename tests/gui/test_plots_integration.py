"""End-to-end wiring of the standalone plots window into the app (offscreen).

The :class:`~nanoscope.gui.plots_panel.PlotsWindow` is opened the way a click on
``Analysis ▸ Plots…`` would open it — through the real :class:`MainWindow`, its
:class:`AnalysisPanel` and the shared :class:`TraceView`. These tests assert the
Phase-3 wiring, not the window internals (covered in test_plots_panel.py):

* the Plots action is dead until there is SOME plottable data, and lights up once
  a detection has run;
* triggering it opens the single PlotsWindow, snapshotted with one source per open
  recording that has a detection — named by the RECORDING, with the app's active
  selection pre-ticked — whose ``dwell_s`` column is plottable;
* re-opening focuses the same window instance (never a second one);
* adding a manual measurement makes the "Manual measurements" source appear after
  the window's Refresh re-pulls the snapshot;
* two open recordings (and a segment named identically to its parent) are offered
  as separate sources and can be overlaid — the comparison the old two-frame
  snapshot made structurally impossible.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.core.document import TraceDocument
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plots_panel import PlotsWindow

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders / drivers
# ================================================================================================

def _event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, *, n: int = 100_000, name: str = "pore_p1.dat") -> TraceDocument:
    arr = _event_trace(n)
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


def _run_detection(window: MainWindow, qtbot, *, min_dwell_s: float = 5e-4) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# tests
# ================================================================================================

def test_plots_action_gated_until_there_is_data(window, qtbot, tmp_path):
    # No document at all: nothing to plot.
    window._sync_plots_action()
    assert not window._act_plots.isEnabled()
    assert window._plots_window is None

    # A selected doc with no detection and no measurements is still nothing to plot.
    _install(window, _synth_doc(tmp_path))
    window._sync_plots_action()
    assert not window._act_plots.isEnabled()


def test_plots_action_opens_populated_window_with_detected_events(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    # The finished detection (via resultsChanged) lit up the Plots action.
    assert window._act_plots.isEnabled()

    # Trigger the action exactly as a menu click would.
    window._act_plots.trigger()

    w = window._plots_window
    assert w is not None and isinstance(w, PlotsWindow)
    assert w.isVisible()

    # It was snapshotted with the RECORDING's detection as the source — sources are
    # named by the document now (a two-pore overlay needs to tell them apart), and
    # the app's active selection is the one pre-ticked.
    assert w.current_source_name() == "pore_p1.dat"
    assert w.source_labels() == ["pore_p1.dat"]
    # ...whose dwell_s column is plottable as a histogram.
    hist_cols = [w._hist_col.itemText(i) for i in range(w._hist_col.count())]
    assert "dwell_s" in hist_cols
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._hist_item is not None
    assert f"n = {len(_EVENTS)}" in w.details_text()


def test_reopening_focuses_the_same_window_instance(window, qtbot, tmp_path):
    _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_plots.trigger()
    first = window._plots_window
    assert first is not None

    window._act_plots.trigger()
    assert window._plots_window is first, "a second PlotsWindow was created"


def test_manual_measurement_source_appears_on_refresh(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path))
    _run_detection(window, qtbot)

    window._act_plots.trigger()
    w = window._plots_window
    # Before any measurement, only the recording's detection is offered.
    assert w.source_labels() == ["pore_p1.dat"]

    # Hand-measure one event and save it to the manual list.
    mp = window._analysis_panel.measure_panel
    assert mp.can_measure()
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    s, e = _EVENTS[0]
    window._trace_view.set_measure_region(float(vals[s]), float(vals[e]))
    assert mp.add_current() is True
    assert len(mp.measurements_df) == 1

    # The window mirrors the app only on an explicit Refresh (snapshot semantics):
    # emitting refreshRequested drives MainWindow to re-pull both sources.
    w.refreshRequested.emit()

    source_names = w.source_labels()
    assert "pore_p1.dat" in source_names
    assert "Manual measurements" in source_names

    # The manual source is plottable on its own numeric columns (e.g. dwell_s).
    w.set_source("Manual measurements")
    assert w.current_source_name() == "Manual measurements"
    meas_cols = [w._hist_col.itemText(i) for i in range(w._hist_col.count())]
    assert "dwell_s" in meas_cols


# ================================================================================================
# multi-source discovery through the real app — the two-pore / two-segment comparison
# ================================================================================================

def test_two_open_recordings_are_both_offered_and_can_be_overlaid(window, qtbot, tmp_path):
    """The capability the window did not have: MainWindow now snapshots EVERY open
    recording that has a detection, so two pores can be laid over one another."""
    a_id, _ = _install(window, _synth_doc(tmp_path, name="pore_p1.dat"))
    _run_detection(window, qtbot)
    b_id, _ = _install(window, _synth_doc(tmp_path, name="pore_p2.dat"))
    _run_detection(window, qtbot)

    window._act_plots.trigger()
    w = window._plots_window

    assert w.source_labels() == ["pore_p1.dat", "pore_p2.dat"]
    # The app's active selection (the second pore) is the one pre-ticked.
    assert w.current_source_name() == "pore_p2.dat"

    w.set_sources(["pore_p1.dat", "pore_p2.dat"])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert len(w._hist_items) == 2
    assert len({i.opts["pen"].color().name() for i in w._hist_items}) == 2
    assert len(w.legend_labels()) == 2

    # Each source carries its own detector provenance, not a shared one.
    keys = w.selected_source_keys()
    assert len(keys) == 2
    assert all(w.sources()[i].provenance.get("min_dwell_s") for i in range(2))


def test_a_segment_of_the_current_recording_becomes_its_own_source(window, qtbot, tmp_path):
    """A segment is an ordinary document, so "this segment vs that segment" is just a
    two-source overlay — and a segment named after its parent must not collide with it."""
    doc_id, channel = _install(window, _synth_doc(tmp_path, name="baseline.dat"))
    _run_detection(window, qtbot)

    seg_id = window._session.create_segment(doc_id, 0, 60_000, name="baseline.dat")
    window._create_default_layers(window._session.document(seg_id))
    window._on_selection(seg_id, 0)
    _run_detection(window, qtbot)

    window._act_plots.trigger()
    w = window._plots_window

    assert len(w.sources()) == 2
    # Identity is the persistent key, so the identically-named segment is its own source...
    assert len({s.key for s in w.sources()}) == 2
    # ...and the legend can still tell them apart.
    assert len(set(w.source_labels())) == 2
    assert any(s.is_segment for s in w.sources())

    w.set_sources([s.key for s in w.sources()])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert len(w._hist_items) == 2
    assert len(w.legend_labels()) == 2


def test_closing_a_recording_drops_its_source_on_refresh(window, qtbot, tmp_path):
    a_id, _ = _install(window, _synth_doc(tmp_path, name="pore_p1.dat"))
    _run_detection(window, qtbot)
    b_id, _ = _install(window, _synth_doc(tmp_path, name="pore_p2.dat"))
    _run_detection(window, qtbot)

    window._act_plots.trigger()
    w = window._plots_window
    w.set_sources(["pore_p1.dat", "pore_p2.dat"])
    assert len(w.selected_source_keys()) == 2

    window._session.close_document(b_id)
    w.refreshRequested.emit()

    assert w.source_labels() == ["pore_p1.dat"]
    assert w.selected_source_keys() == [w.sources()[0].key]
