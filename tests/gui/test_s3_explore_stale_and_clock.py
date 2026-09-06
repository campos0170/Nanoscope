"""S3 JAB-05 / JHT-05 — the Plots window says what the table it plots already says.

Two facts that stopped at the event table and never reached the window where those
numbers stop being a table and become the number that gets reported.

**JAB-05 — a superseded calibration.** Apply a units-changing calibration and the event
table immediately badges ``baseline_pA``, ``depth_pA``, ``frac_depth``, ``ecd_pA_s``,
``mean_dev_pA``, ``max_dev_pA``, ``snr``, ``rise_s``, ``fall_s`` with a ⚠ header and
amber cells (``AnalysisPanel._check_staleness``). Two panels away, Plots drew the same
``depth_pA``, printed "μ = 76.14 pA   (n = 111)" in the title, the legend and Details --
and a scan for "stale", "superseded", "no longer", "re-measure", "⚠" across the whole
window returned nothing. The app's doctrine for a superseded measurement is "never
silently re-measure": the numbers stay and every surface that shows them says so
(USER_GUIDE §9a -- such a project stays "marked, badged, and otherwise fully readable,
plottable and exportable"). *Plottable* is where the mark stopped.

**JHT-05 — which clock ``t_start_s`` is on.** ``t_start_s`` is seconds since THIS
recording's first loaded sample. The event table header reads ``t_start_s (t₀=0)``, the
measurement table carries a per-row ``t0_s``, the PSD segment spin box names the clock in
its tooltip -- and the Plots window, the only surface that POOLS the column across
recordings, labelled the axis ``t_start (s)`` and said nothing. A windowed recording
(t₀ = 160 s) overlaid with a segment of itself (rebased to 0) draws ONE physical event at
two different x positions. ``core/clock.py``: "One clock, or every control that shows or
takes a time says which clock it is in."

Both are wired through ONE seam (``PlotsWindow._column_notes`` / ``_stale_headline``)
that the histogram, the scatter and the PCA all call, so a caveat cannot be true in one
view of the same numbers and absent from another. These tests exercise all three.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource, discover_plot_sources
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.plots_panel import PlotsWindow

FS = 50_000.0
STAMPED_FP = (1.0, 0.0)


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, name: str, *, t0_s: float = 0.0, n: int = 4096) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -200.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,), t0_s=float(t0_s))
    return TraceDocument(meta=meta, channels=(arr,))


def _events(n: int = 60, *, seed: int = 5, t_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 10 * i, "end_idx": 10 * i + 5,
            "t_start_s": t_shift + 1e-3 * i, "dwell_s": float(rng.exponential(2e-3) + 2e-5),
            "baseline_pA": -1000.0, "depth_pA": float(abs(rng.normal(80.0, 12.0))),
            "frac_depth": 0.08, "ecd_pA_s": 0.05, "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 6e-5, "fall_s": 7e-5, "snr": 20.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _session_with_detection(tmp_path, *, name="pore.dat", t0_s=0.0, t_shift=0.0, seed=5):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, name, t0_s=t0_s))
    ev = EventSet(_events(seed=seed, t_shift=t_shift), {"min_dwell_s": 2e-5, "fs": FS})
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=STAMPED_FP, payload=ev))
    return session, doc_id


def _supersede(session: Session, doc_id: str) -> None:
    """Install a calibration the stored detection was NOT measured under."""
    session.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0, offset=0.0,
                                                          target_units="pA"),
                            write_sidecar=False)
    assert tuple(session.calibration(doc_id, 0).fingerprint()) != STAMPED_FP


def _plots(qtbot, session: Session) -> PlotsWindow:
    w = PlotsWindow(session)
    qtbot.addWidget(w)
    w.set_data(discover_plot_sources(
        session.documents(), session.results.get,
        calibration_fingerprint=lambda d, c: session.calibration(d, c).fingerprint()))
    return w


def _title(w: PlotsWindow) -> str:
    return str(w._plot.titleLabel.text or "")


# ================================================================================================
# JAB-05 -- a superseded calibration reaches the plot
# ================================================================================================

def test_a_superseded_calibration_is_named_in_the_histogram_details(qapp, qtbot, tmp_path):
    session, doc_id = _session_with_detection(tmp_path)
    _supersede(session, doc_id)
    w = _plots(qtbot, session)
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    text = w.details_text()
    assert "SUPERSEDED CALIBRATION" in text, text
    assert "depth_pA" in text
    assert "Re-run detection" in text


def test_the_mark_reaches_the_title_which_is_what_an_exported_image_keeps(qapp, qtbot,
                                                                         tmp_path):
    session, doc_id = _session_with_detection(tmp_path)
    _supersede(session, doc_id)
    w = _plots(qtbot, session)
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    assert "SUPERSEDED CALIBRATION" not in _title(w)
    assert "SUPERSEDED CALIBRATION" in w.details_text()


def test_the_same_seam_marks_the_scatter_and_the_pca(qapp, qtbot, tmp_path):
    session, doc_id = _session_with_detection(tmp_path)
    _supersede(session, doc_id)
    w = _plots(qtbot, session)

    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    assert "SUPERSEDED CALIBRATION" in w.details_text()
    assert "SUPERSEDED CALIBRATION" not in _title(w)
    assert "SUPERSEDED CALIBRATION" in w.details_text()

    w.set_feature_selection(["depth_pA", "dwell_s", "snr"])
    w.run_pca()
    w.set_plot_type("PCA")
    assert "SUPERSEDED CALIBRATION" in w.details_text(), w.details_text()


def test_a_column_the_calibration_does_not_touch_is_not_marked(qapp, qtbot, tmp_path):
    """``dwell_s`` is not in REMEASURED_COLUMNS: a gain change does not move it."""
    session, doc_id = _session_with_detection(tmp_path)
    _supersede(session, doc_id)
    w = _plots(qtbot, session)
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert "SUPERSEDED CALIBRATION" not in w.details_text()
    assert "SUPERSEDED" not in _title(w)


def test_a_current_calibration_produces_no_mark_at_all(qapp, qtbot, tmp_path):
    """The silence must still mean "nothing to report" when nothing is wrong."""
    session, _doc_id = _session_with_detection(tmp_path)
    w = _plots(qtbot, session)
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    assert "SUPERSEDED" not in w.details_text()
    assert "SUPERSEDED" not in _title(w)


# ================================================================================================
# JHT-05 -- which clock t_start_s is on
# ================================================================================================

def _two_source_window(qtbot) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    a = PlotSource(key="a", label="parent", df=_events(seed=1, t_shift=5.0))
    b = PlotSource(key="b", label="segment", df=_events(seed=2, t_shift=1.0))
    w.set_data([a, b])
    w.set_sources(["parent", "segment"])
    return w


def test_pooling_t_start_across_recordings_names_the_clock(qapp, qtbot):
    w = _two_source_window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column("t_start_s")
    text = w.details_text()
    assert "CLOCK:" in text, text
    assert "t₀" in text, text
    assert "own first loaded sample" in text, text
    assert "not simultaneous" in text, text


def test_the_axis_label_names_the_clock_too(qapp, qtbot):
    """The axis label is what an exported image keeps; Details is not exported."""
    w = _two_source_window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column("t_start_s")
    label = w._plot.getAxis("bottom").labelString()
    assert "origins differ" in label, label

    w.set_plot_type("Scatter")
    w.set_scatter_columns("t_start_s", "depth_pA")
    assert "origins differ" in w._plot.getAxis("bottom").labelString()


def test_a_single_recording_states_its_own_origin(qapp, qtbot, tmp_path):
    session, _doc_id = _session_with_detection(tmp_path, t0_s=61.3034)
    w = _plots(qtbot, session)
    w.set_plot_type("Histogram")
    w.set_histogram_column("t_start_s")
    label = w._plot.getAxis("bottom").labelString()
    assert "t₀ = 61.3034 s" in label, label
    text = w.details_text()
    assert "CLOCK:" in text and "61.3034" in text, text


@pytest.mark.parametrize("col", ["dwell_s", "rise_s", "fall_s"])
def test_a_duration_column_gets_no_clock_clause(qapp, qtbot, col):
    """A duration has no origin; annotating it would be a second falsehood."""
    w = _two_source_window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    label = w._plot.getAxis("bottom").labelString()
    assert "origins" not in label and "t₀" not in label, label
    assert "CLOCK:" not in w.details_text()


def test_a_pushed_frame_says_why_its_origin_is_unknown(qapp, qtbot):
    """No open recording behind the rows is a different silence from "no sample rate",
    and ``clock.field_note`` answers None with the latter -- so this window says the
    true one rather than borrowing a sentence about a fact it has not established."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"pushed": _events()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("t_start_s")
    text = w.details_text()
    assert "CLOCK:" in text, text
    assert "No open recording stands behind these rows" in text, text
    assert "declares no sample rate" not in text, text
