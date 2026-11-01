"""Event waveforms ▸ Ensemble: extremum alignment, the variability band, the subset filter.

Synthetic recordings only (the ramp blockades of ``test_event_plot``: each event's deepest
sample is its LAST one). Checks that:

* alignment on the extremum puts every event's deepest sample at t = 0, with the polarity
  named in Details and in the CSV;
* the band is a reduction of the traces already drawn -- choosing it never runs the
  ensemble kernel, never reads the trace and keeps the zoom -- and it is exported, saved
  in the workspace snapshot and described (with its n per point) in Details;
* the Events subset narrows the ensemble to one of the gallery's curation subsets, always
  within the accepted events.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from unittest.mock import Mock

import nanoscope  # noqa: F401  -> Qt bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtTest

from nanoscope.analysis import event_shape
from nanoscope.core.analysis_workspace import capture_event_plots, restore_event_plots
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.gui.event_plot_panel import ENSEMBLE, EventPlotWindow

_helpers = runpy.run_path(str(Path(__file__).with_name("test_event_plot.py")))
_fed_window = _helpers["_fed_window"]
_doc_and_events = _helpers["_doc_and_events"]
_install_detection = _helpers["_install_detection"]
BASE = _helpers["BASE"]


def _forbid_analysis(monkeypatch, window):
    """Fail the test if the ensemble kernel runs or the trace is read again."""
    monkeypatch.setattr(event_shape, "event_ensemble",
                        Mock(side_effect=AssertionError("ensemble recomputed")))
    monkeypatch.setattr(window._session, "calibrated",
                        Mock(side_effect=AssertionError("trace read again")))


def _real_time_extremum(w, *, pad=0, polarity=-1):
    # The detector's polarity, as a detection records it. (This noiseless synthetic trace
    # has no noise scale for derive_polarity to measure against; the kernel tests cover
    # the measured fallback on a noisy one.)
    w._eventset.provenance["polarity"] = polarity
    w.set_view(ENSEMBLE)
    w.set_normalize_time(False)
    w.set_pad_percent(pad)
    w.set_align("extremum")
    acc = w._eventset.accepted.df
    width = int((acc["end_idx"] - acc["start_idx"]).max())
    w.set_n_points(2 * width)          # one point per sample of the common window
    return acc, width


# ================================================================================================
# E1 extremum alignment
# ================================================================================================

def test_extremum_alignment_puts_each_deepest_sample_at_t0(qtbot, tmp_path):
    w = _fed_window(qtbot, tmp_path)
    acc, width = _real_time_extremum(w)
    T, matrix, mean, _median = w.ensemble_traces()
    assert matrix.shape == (len(acc), 2 * width)
    zero = int(np.flatnonzero(T == 0.0)[0])
    # Each ramp's deepest sample is its last: BASE - depth.
    np.testing.assert_allclose(matrix[:, zero], BASE - acc["depth_pA"].to_numpy(), rtol=0,
                               atol=1e-3)
    assert T[0] == pytest.approx(-width / 1e5)
    assert w._ensemble_alignment == (-1, "the polarity the event table was detected with")
    details = w.details_text()
    assert "Aligned at each event's extremum" in details
    assert "polarity -1" in details and "current decrease" in details
    data = w._plot_data()
    assert set(data["align"]) == {"extremum"}
    assert set(data["align_polarity"]) == {-1}
    assert data["identity_available"].all()
    # The replayed identities are the kept events, in order.
    ids = data.drop_duplicates("trace_index")["event_id"].tolist()
    assert ids == acc["event_id"].tolist()


def test_extremum_reads_a_current_increase_with_polarity_plus_one(qtbot, tmp_path):
    session = Session()
    doc, ev = _doc_and_events(tmp_path)
    sig = doc.channels[0]
    flipped = (2 * BASE - sig).astype(np.float32)        # every ramp now RISES above I0
    doc = type(doc)(meta=doc.meta, channels=(flipped,))
    ev.provenance["polarity"] = 1
    doc_id = session.add_document(doc)
    _install_detection(session, doc_id, ev)
    w = EventPlotWindow(session)
    qtbot.addWidget(w)
    w.set_source(doc_id, 0, ev)
    acc, _width = _real_time_extremum(w, polarity=1)
    T, matrix, _mean, _median = w.ensemble_traces()
    zero = int(np.flatnonzero(T == 0.0)[0])
    np.testing.assert_allclose(matrix[:, zero], BASE + acc["depth_pA"].to_numpy(), rtol=0,
                               atol=1e-3)
    assert w._ensemble_alignment == (1, "the polarity the event table was detected with")
    assert "current increase" in w.details_text()


# ================================================================================================
# E2 variability band
# ================================================================================================

def test_band_is_drawn_from_the_traces_on_screen_and_keeps_the_zoom(qtbot, tmp_path, monkeypatch):
    w = _fed_window(qtbot, tmp_path)
    w.set_view(ENSEMBLE)
    T, matrix, mean, median = w.ensemble_traces()
    view = w._ensemble_plot.getViewBox()
    view.setRange(xRange=(0.2, 0.6), yRange=(900.0, 1010.0), padding=0)
    ranges = np.asarray(view.viewRange()).copy()
    _forbid_analysis(monkeypatch, w)
    changed = QtTest.QSignalSpy(w.workspaceChanged)

    w.set_band("sd")
    lower, upper, n = w.ensemble_band()
    np.testing.assert_allclose(lower, mean - matrix.std(axis=0, ddof=1))
    np.testing.assert_allclose(upper, mean + matrix.std(axis=0, ddof=1))
    np.testing.assert_array_equal(n, np.full(T.size, matrix.shape[0]))
    fill = w._band_items[0]
    assert isinstance(fill, pg.FillBetweenItem) and fill in w._ensemble_plot.items
    assert fill.brush().color().name() == pg.mkPen(w._mean_item.opts["pen"]).color().name()
    np.testing.assert_allclose(view.viewRange(), ranges, rtol=0, atol=1e-12)
    assert changed.count() == 1
    assert w.ensemble_traces()[1] is matrix               # the ensemble itself untouched
    assert "shaded mean ± 1 SD" in w._ensemble_plot.titleLabel.text
    details = w.details_text()
    assert "mean ± 1 sample standard deviation (ddof = 1)" in details
    assert (f"Every point of the band uses n = {matrix.shape[0]} events, the same events "
            "as its curve.") in details
    # The same readout is on screen beside the figure, collapsed to "Full details"; with
    # nothing to qualify there is no caution summary line.
    notes = w._ensemble_notes
    assert f"n = {matrix.shape[0]} events" in notes.full_text()
    assert notes.summary.text() == "" and not notes.isHidden()

    w.set_band("iqr")
    lower, upper, _n = w.ensemble_band()
    np.testing.assert_allclose(lower, np.percentile(matrix, 25, axis=0))
    np.testing.assert_allclose(upper, np.percentile(matrix, 75, axis=0))
    fill = w._band_items[0]
    assert fill.brush().color().name() == pg.mkPen(w._median_item.opts["pen"]).color().name()
    assert "25th to 75th percentile" in w.details_text()

    w.set_band("")
    assert w.ensemble_band() is None and w._band_items == ()
    assert fill not in w._ensemble_plot.items
    np.testing.assert_allclose(view.viewRange(), ranges, rtol=0, atol=1e-12)


def test_band_and_its_n_per_point_are_in_the_csv(qtbot, tmp_path):
    w = _fed_window(qtbot, tmp_path)
    w.set_view(ENSEMBLE)
    data = w._plot_data()
    assert set(data["band_kind"]) == {"none"} and data["band_lower"].isna().all()
    w.set_band("sd")
    data = w._plot_data()
    lower, upper, n = w.ensemble_band()
    first = data[data["trace_index"] == 0].sort_values("sample_index")
    np.testing.assert_allclose(first["band_lower"], lower)
    np.testing.assert_allclose(first["band_upper"], upper)
    np.testing.assert_array_equal(first["n_events_at_point"], n)
    assert set(data["band_kind"]) == {"mean_pm_1sd"}


def _window_with_signal(qtbot, tmp_path, edit):
    """The synthetic recording with ``edit(signal, events)`` applied to its trace."""
    session = Session()
    doc, ev = _doc_and_events(tmp_path)
    sig = doc.channels[0].copy()
    edit(sig, ev)
    doc = type(doc)(meta=doc.meta, channels=(sig,))
    doc_id = session.add_document(doc)
    _install_detection(session, doc_id, ev)
    w = EventPlotWindow(session)
    qtbot.addWidget(w)
    w.set_source(doc_id, 0, ev)
    w.set_view(ENSEMBLE)
    return w, ev


@pytest.mark.parametrize("kind", ["sd", "iqr"])
def test_band_and_its_n_agree_with_the_curves_where_an_event_is_not_finite(
        qtbot, tmp_path, kind):
    def dropout(sig, ev):
        s0 = int(ev.df["start_idx"].iloc[3])
        sig[s0 + 20:s0 + 50] = np.nan                 # a 30-sample dropout inside one event

    w, _ev = _window_with_signal(qtbot, tmp_path, dropout)
    w.set_band(kind)
    T, matrix, mean, median = w.ensemble_traces()
    lower, upper, n = w.ensemble_band()
    gap = ~np.isfinite(matrix).all(axis=0)
    assert gap.any() and not gap.all()
    # The mean and median take every event at every point, so they are NaN in the gap;
    # the band is left out with them, and the n behind them there is 0.
    assert np.isnan(mean[gap]).all() and np.isnan(median[gap]).all()
    assert np.isnan(lower[gap]).all() and np.isnan(upper[gap]).all()
    np.testing.assert_array_equal(n[gap], 0)
    # Everywhere else every drawn event is behind both, and the band surrounds its curve.
    np.testing.assert_array_equal(n[~gap], matrix.shape[0])
    centre = mean if kind == "sd" else median
    assert np.isfinite(lower[~gap]).all() and np.isfinite(upper[~gap]).all()
    assert (lower[~gap] <= centre[~gap]).all() and (centre[~gap] <= upper[~gap]).all()
    if kind == "sd":
        np.testing.assert_allclose((lower + upper)[~gap] / 2.0, mean[~gap], rtol=1e-12)
    # The CSV never pairs an n with a summary value it does not lie behind.
    first = w._plot_data().query("trace_index == 0").sort_values("sample_index")
    undefined = first["mean_current"].isna().to_numpy()
    np.testing.assert_array_equal(undefined, gap)
    np.testing.assert_array_equal(first["n_events_at_point"].to_numpy(), n)
    assert first["band_lower"].isna().to_numpy()[undefined].all()
    details = w.details_text()
    assert (f"At {int(gap.sum())} of {T.size} points at least one event's window holds a "
            "non-finite sample") in details
    assert "Ensemble notes (1)" in w._ensemble_notes.summary.text()
    # With no band chosen the column is the same count behind the mean and median.
    w.set_band("")
    none = w._plot_data().query("trace_index == 0").sort_values("sample_index")
    np.testing.assert_array_equal(none["n_events_at_point"].to_numpy(), n)


def test_an_event_with_no_finite_sample_is_not_counted_as_edge_skipped(qtbot, tmp_path):
    def blank(sig, ev):
        s, e = (int(ev.df[c].iloc[3]) for c in ("start_idx", "end_idx"))
        sig[s:e] = np.nan                             # nothing to find a deepest sample in

    w, ev = _window_with_signal(qtbot, tmp_path, blank)
    _real_time_extremum(w)
    _T, matrix, _mean, _median = w.ensemble_traces()
    assert matrix.shape[0] == len(ev.df) - 1
    title = w._ensemble_plot.titleLabel.text
    assert "1 with no finite sample" in title and "edge-skipped" not in title
    data = w._plot_data()
    assert set(data["edge_skipped_count"]) == {0}
    assert set(data["no_finite_sample_count"]) == {1}
    assert "no finite sample inside their own span" in w.details_text()
    assert data.drop_duplicates("trace_index")["event_id"].tolist() == \
        [i for i in ev.df["event_id"].tolist() if i != ev.df["event_id"].iloc[3]]
    restored = EventPlotWindow(Session())
    qtbot.addWidget(restored)
    restore_event_plots(restored, capture_event_plots(w))
    restored.set_view(ENSEMBLE)
    assert restored._ensemble_plot.titleLabel.text == title
    # Start alignment keeps the all-NaN row (it has a start), so nothing is skipped there.
    w.set_align("start")
    assert w.ensemble_traces()[1].shape[0] == len(ev.df)
    assert "with no finite sample" not in w._ensemble_plot.titleLabel.text


def test_band_follows_the_summary_colour_after_an_appearance_edit(qtbot, tmp_path, monkeypatch):
    w = _fed_window(qtbot, tmp_path)
    w.set_view(ENSEMBLE)
    w.set_band("sd")
    _forbid_analysis(monkeypatch, w)
    w.set_appearance_state({**w.appearance_state(), "fit_color": "#00ff00"})
    fill, lo_item, hi_item = w._band_items
    assert pg.mkPen(w._mean_item.opts["pen"]).color().name() == "#00ff00"
    assert fill.brush().color().name() == "#00ff00"
    assert lo_item.opts["pen"].color().name() == hi_item.opts["pen"].color().name() == "#00ff00"
    # The band stays under every other item, the faint per-event lines included.
    assert fill.zValue() < w._events_item.zValue() and fill.zValue() < w._mean_item.zValue()


def test_band_subset_and_alignment_roundtrip_through_a_project(qtbot, tmp_path, monkeypatch):
    w = _fed_window(qtbot, tmp_path)
    _real_time_extremum(w, pad=10)
    w.set_band("iqr")
    expected_band = tuple(np.asarray(v).copy() for v in w.ensemble_band())
    expected_title = w._ensemble_plot.titleLabel.text
    artifact = capture_event_plots(w)
    assert artifact.payload["config"]["ensemble_band"] == "iqr"
    assert artifact.payload["config"]["ensemble_subset"] == "Accepted only"
    path = tmp_path / "ensemble.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=[artifact])))

    restored = EventPlotWindow(Session())
    qtbot.addWidget(restored)
    monkeypatch.setattr(event_shape, "event_ensemble",
                        Mock(side_effect=AssertionError("ensemble recomputed on restore")))
    monkeypatch.setattr(event_shape, "ensemble_band",
                        Mock(side_effect=AssertionError("band recomputed on restore")))
    report = restore_event_plots(restored, load_project(path).workspace.artifacts[0])
    assert report.skipped == ()
    restored.set_view(ENSEMBLE)
    for got, want in zip(restored.ensemble_band(), expected_band):
        np.testing.assert_array_equal(got, want)
    assert isinstance(restored._band_items[0], pg.FillBetweenItem)
    assert restored._band_combo.currentData() == "iqr"
    assert restored._align_combo.currentText() == "extremum"
    assert restored._ensemble_alignment == w._ensemble_alignment
    assert restored._ensemble_plot.titleLabel.text == expected_title
    assert "polarity -1" in restored.details_text()
    assert capture_event_plots(restored).payload["config"]["ensemble_band"] == "iqr"


def test_a_snapshot_from_before_the_band_reopens_without_one(qtbot, tmp_path):
    w = _fed_window(qtbot, tmp_path)
    w.set_view(ENSEMBLE)
    artifact = capture_event_plots(w)
    for key in ("ensemble_band", "ensemble_subset"):
        artifact.payload["config"].pop(key)
    for key in ("ensemble_band", "ensemble_alignment", "ensemble_input_count"):
        artifact.payload.pop(key)
    restored = EventPlotWindow(Session())
    qtbot.addWidget(restored)
    report = restore_event_plots(restored, artifact)
    assert report.skipped == ()
    assert restored.ensemble_band() is None and restored._band_items == ()
    assert restored._subset_combo.currentData() == "Accepted only"
    assert restored._mean_item is not None


# ================================================================================================
# E3 subset filter
# ================================================================================================

def _labelled_window(qtbot, tmp_path):
    accepted = [True] * 24
    accepted[2] = accepted[5] = False
    session = Session()
    doc, ev = _doc_and_events(tmp_path, accepted=accepted)
    flags = ["type=collision" if i % 3 == 2 else "" for i in range(24)]
    ev.df["flag"] = flags
    doc_id = session.add_document(doc)
    _install_detection(session, doc_id, ev)
    w = EventPlotWindow(session)
    qtbot.addWidget(w)
    w.set_source(doc_id, 0, ev)
    w.set_view(ENSEMBLE)
    return w, ev


def test_subset_offers_the_gallery_filters_within_accepted_only(qtbot, tmp_path):
    w, ev = _labelled_window(qtbot, tmp_path)
    offered = [w._subset_combo.itemData(i) for i in range(w._subset_combo.count())]
    assert offered[0] == "Accepted only"
    assert "Labelled: type=collision" in offered and "Unlabelled" in offered
    assert "All" not in offered and "Rejected only" not in offered
    assert w._ensemble_plot.titleLabel.text.startswith("Ensemble of 22 event(s)")
    with pytest.raises(ValueError):
        w.set_ensemble_subset("Rejected only")

    w.set_ensemble_subset("Labelled: type=collision")
    _T, matrix, _mean, _median = w.ensemble_traces()
    wanted = ev.df[ev.df["accepted"] & (ev.df["flag"] == "type=collision")]
    assert matrix.shape[0] == len(wanted) == 6           # 8 labelled, 2 of them rejected
    data = w._plot_data()
    assert data.drop_duplicates("trace_index")["event_id"].tolist() == \
        wanted["event_id"].tolist()
    assert set(data["ensemble_subset"]) == {"Labelled: type=collision"}
    assert "Labelled: type=collision" in w._ensemble_plot.titleLabel.text
    assert "Events: Accepted · Labelled: type=collision (6 drawn)." in w.details_text()
    # The heatmap is not narrowed by the ensemble's subset.
    assert w.density_counts().sum() == 22


def test_subset_is_saved_and_an_empty_one_says_so(qtbot, tmp_path):
    w, _ev = _labelled_window(qtbot, tmp_path)
    w.set_ensemble_subset("Unlabelled")
    assert w.ensemble_traces()[1].shape[0] == 16
    artifact = capture_event_plots(w)
    assert artifact.payload["config"]["ensemble_subset"] == "Unlabelled"
    restored = EventPlotWindow(Session())
    qtbot.addWidget(restored)
    restore_event_plots(restored, artifact)
    assert restored._subset_combo.currentData() == "Unlabelled"
    assert "Unlabelled" in restored._ensemble_plot.titleLabel.text

    w.set_ensemble_subset("Labelled: type=absent")
    assert w.ensemble_traces() is None
    texts = [item.textItem.toPlainText() for item in w._ensemble_plot.items
             if isinstance(item, pg.TextItem)]
    assert any("No accepted events in the subset" in t for t in texts)
