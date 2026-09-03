"""GUI tests for the event-plots window (pytest-qt, offscreen).

The standalone :class:`~nanoscope.gui.event_plot_panel.EventPlotWindow`, exercised
the way the user would after a detection:

* fed a Session with one synthetic recording whose CURRENT trace really carries the
  events its stored ``detect.schmitt`` EventSet points at, the DENSITY view bins an
  X vs Y metric into a non-empty image with populated top/right marginals, the
  metric combos (including the on-demand ECD) re-render, and the log toggles apply;
* the ENSEMBLE view overlays the accepted events and draws a mean + median trace of
  the requested length, honouring the align / points / pad controls;
* *Save image* writes a PNG for either view, and an empty source shows a message;
* through the real :class:`MainWindow` the ``Analysis > Event plots...`` action gates
  on the active detection having >= 1 accepted event and opens a single, lazily-built
  window.

The signal and EventSet are built together so the plotted numbers trace back to a
known waveform; nothing here touches the frozen 20-column EventSet schema.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_plot_panel import DENSITY, ENSEMBLE, EventPlotWindow
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N_SAMPLES = 40_000
BASE = 1000.0
_SPACING = 1_200
_START0 = 500


# ================================================================================================
# builders -- a signal that really carries the events its EventSet points at
# ================================================================================================

def _doc_and_events(tmp_path, *, name: str = "rec.dat", n_events: int = 24,
                    accepted: list[bool] | None = None) -> tuple[TraceDocument, EventSet]:
    """A CURRENT recording whose trace has ``n_events`` ramp blockades + the matching EventSet."""
    rng = np.random.default_rng(0)
    sig = np.full(N_SAMPLES, BASE, dtype=np.float32)
    rows = []
    for i in range(n_events):
        start = _START0 + i * _SPACING
        length = int(rng.integers(90, 380))
        depth = float(rng.uniform(40.0, 160.0))
        sig[start:start + length] = BASE - np.linspace(0.0, depth, length)
        rows.append((start, start + length, depth, length))

    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N_SAMPLES,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(sig,))

    acc = accepted if accepted is not None else [True] * n_events
    df = empty_df()
    for i, (start, end, depth, length) in enumerate(rows):
        df.loc[i] = {
            "event_id": i, "recording": name, "pore_id": "P",
            "start_idx": start, "end_idx": end,
            "t_start_s": start / FS, "dwell_s": length / FS,
            "baseline_pA": BASE, "depth_pA": depth, "frac_depth": depth / BASE,
            "ecd_pA_s": 0.0, "max_dev_pA": depth, "mean_dev_pA": depth / 2.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": bool(acc[i]), "flag": "",
        }
    df = df.astype({c: d for c, (d, _, _) in SCHEMA.items()})
    return doc, EventSet(df, {"analyzed_duration_s": 1.0, "recording": name})


def _install_detection(session: Session, doc_id: str, ev: EventSet, channel: int = 0) -> None:
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=ev))


def _fed_window(qtbot, tmp_path, **kw) -> EventPlotWindow:
    session = Session()
    doc, ev = _doc_and_events(tmp_path, **kw)
    doc_id = session.add_document(doc)
    _install_detection(session, doc_id, ev)
    w = EventPlotWindow(session)
    qtbot.addWidget(w)
    w.set_source(doc_id, 0, ev)
    return w


@pytest.fixture
def window(qapp, qtbot, tmp_path):
    return _fed_window(qtbot, tmp_path)


# ================================================================================================
# density: image + marginals
# ================================================================================================

def test_density_renders_image_and_marginals(window):
    w = window
    w.set_view(DENSITY)
    counts = w.density_counts()
    assert counts is not None
    assert counts.shape == (40, 40)                       # default bins
    assert counts.sum() == 24                             # every accepted event is placed
    assert isinstance(w._density_image, pg.ImageItem)
    assert w._density_image.image is not None and np.any(w._density_image.image > 0)

    # the marginals are the X and Y projections of the 2-D counts
    assert isinstance(w._top_marginal_item, pg.BarGraphItem)
    assert isinstance(w._right_marginal_item, pg.BarGraphItem)
    np.testing.assert_array_equal(
        np.asarray(w._top_marginal_item.opts["height"]), counts.sum(axis=1))
    np.testing.assert_array_equal(
        np.asarray(w._right_marginal_item.opts["width"]), counts.sum(axis=0))


def test_bins_spinner_reshapes_the_grid(window):
    window.set_view(DENSITY)
    window.set_bins(15)
    counts = window.density_counts()
    assert counts.shape == (15, 15)
    assert counts.sum() == 24


# ================================================================================================
# density: metric combos incl ECD, and log toggles
# ================================================================================================

def test_metric_combos_including_ecd_rerender(window):
    w = window
    w.set_view(DENSITY)
    # default X=dwell_s, Y=depth_pA
    base = w.density_counts().copy()

    w.set_x_metric("ecd")                                 # the on-demand charge deficit
    ecd_counts = w.density_counts()
    assert ecd_counts is not None and ecd_counts.sum() == 24
    assert not np.array_equal(base, ecd_counts)           # a different metric re-binned

    w.set_y_metric("frac_depth")
    assert w.density_counts().sum() == 24                 # still renders every event
    w.set_x_metric("dwell_s")
    assert w.density_counts().sum() == 24


def test_ecd_metric_axis_label_mentions_ecd(window):
    w = window
    w.set_view(DENSITY)
    w.set_density_log(False, False)                       # ECD is signed; keep a linear axis
    w.set_x_metric("ecd")
    label = w._density_main.getAxis("bottom").labelText
    assert "ECD" in label


def test_log_toggles_apply(window):
    w = window
    w.set_view(DENSITY)
    w.set_density_log(True, False)                        # dwell on a log axis
    assert w.density_counts().sum() == 24                 # all dwell values positive -> kept
    assert "log10" in w._density_main.getAxis("bottom").labelText
    # log10 of a sub-millisecond dwell is negative, so the binned x-range is < 0
    xr = w._density_main.viewRange()[0]
    assert xr[1] < 0

    w.set_density_log(False, False)
    assert "log10" not in w._density_main.getAxis("bottom").labelText


def test_colormap_choice_changes_the_lut(window):
    w = window
    w.set_view(DENSITY)
    w.set_colormap("magma")
    assert w._density_image is not None                   # re-rendered without error
    w.set_colormap("viridis")
    assert w._density_image is not None


# ================================================================================================
# ensemble: mean + median of the right length, and the controls
# ================================================================================================

def test_ensemble_draws_mean_and_median_of_right_length(window):
    w = window
    w.set_view(ENSEMBLE)
    w.set_normalize_time(True)
    w.set_n_points(120)
    T, matrix, mean, median = w.ensemble_traces()
    assert T.shape == (120,)
    assert matrix.shape == (24, 120)                      # one row per accepted event
    assert mean.shape == (120,) and median.shape == (120,)
    assert isinstance(w._mean_item, pg.PlotDataItem)
    assert isinstance(w._median_item, pg.PlotDataItem)
    assert w._mean_item.yData.size == 120
    assert w._median_item.yData.size == 120
    assert w._events_item is not None                     # the faint overlay is drawn


def test_ensemble_controls_align_points_pad(window):
    w = window
    w.set_view(ENSEMBLE)
    w.set_normalize_time(False)                            # real-time mode: alignment bites
    w.set_align("center")
    w.set_n_points(64)
    w.set_pad_percent(50)
    T, matrix, mean, median = w.ensemble_traces()
    assert matrix.shape[1] == 64
    assert T.shape == (64,) and mean.shape == (64,)


def test_ensemble_uses_accepted_events_only(qtbot, tmp_path):
    acc = [True] * 20
    acc[3] = acc[7] = acc[11] = False                     # reject three
    w = _fed_window(qtbot, tmp_path, n_events=20, accepted=acc)
    w.set_view(ENSEMBLE)
    _, matrix, _, _ = w.ensemble_traces()
    assert matrix.shape[0] == 17                          # only the accepted events overlaid
    # ...and the density view agrees
    w.set_view(DENSITY)
    assert w.density_counts().sum() == 17


# ================================================================================================
# save image + empty state
# ================================================================================================

def test_save_image_writes_png_for_both_views(window, tmp_path):
    window.set_view(DENSITY)
    d = window.save_image(str(tmp_path / "density.png"))
    assert d is not None and d.exists() and d.stat().st_size > 0

    window.set_view(ENSEMBLE)
    e = window.save_image(str(tmp_path / "ensemble.png"))
    assert e is not None and e.exists() and e.stat().st_size > 0


def test_empty_source_shows_message_state(qapp, qtbot):
    w = EventPlotWindow(Session())
    qtbot.addWidget(w)
    w.set_source(None, None, None)
    assert not w.has_data()
    assert w.density_counts() is None
    assert w.ensemble_traces() is None


def test_all_rejected_is_empty_state(qtbot, tmp_path):
    w = _fed_window(qtbot, tmp_path, n_events=6, accepted=[False] * 6)
    assert not w.has_data()
    assert w.density_counts() is None
    assert w.ensemble_traces() is None


# ================================================================================================
# wiring through MainWindow
# ================================================================================================

def _install(window: MainWindow, doc: TraceDocument) -> str:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    return doc_id


def test_event_plots_action_gated_and_opens_single_window(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)

    win._sync_event_plots_action()
    assert not win._act_event_plots.isEnabled()           # no detection -> dead

    doc, ev = _doc_and_events(tmp_path, name="a.dat")
    doc_id = _install(win, doc)
    _install_detection(win._session, doc_id, ev)
    win._on_selection(doc_id, 0)                          # point the panel at the detection
    win._sync_event_plots_action()
    assert win._act_event_plots.isEnabled()               # >=1 accepted event -> live

    win._open_event_plots()
    first = win._event_plot_window
    assert first is not None
    assert first.has_data()
    assert first.density_counts().sum() == len(ev.df)
    # re-opening focuses the SAME instance (never a second window)
    win._open_event_plots()
    assert win._event_plot_window is first


def test_event_plots_action_dead_without_accepted_events(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc, ev = _doc_and_events(tmp_path, name="b.dat", n_events=5, accepted=[False] * 5)
    doc_id = _install(win, doc)
    _install_detection(win._session, doc_id, ev)
    win._on_selection(doc_id, 0)
    win._sync_event_plots_action()
    assert not win._act_event_plots.isEnabled()           # a detection, but nothing accepted


def test_event_plots_action_dead_without_a_detection(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc, _ = _doc_and_events(tmp_path, name="c.dat")
    doc_id = _install(win, doc)
    win._on_selection(doc_id, 0)                          # opened, but never detected
    win._sync_event_plots_action()
    assert not win._act_event_plots.isEnabled()
