"""Trace overviews must retain short pulses and mark the assigned event itself."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nanoscope.plot import incubation_galleries as galleries


def _recording():
    signal = np.zeros(600_011)
    # Neither event minimum survives the former every-third-sample overview.
    signal[103] = -90.
    signal[300_007] = -60.  # Minimum at the inclusive end of the second event.
    signal[-1] = -120.  # Tail and last recording sample must also survive.
    catalog = pd.DataFrame({
        "start_idx": [100, 300_000, 201], "end_idx": [110, 300_007, 205],
        "dwell_ms": [.22, .16, .1], "depth_peak_pA": [90., 60., 1.],
        "depth_sust_pA": [40., 30., .5], "passes_gate": [True, True, False],
    })
    assignments = pd.DataFrame({
        "tag": ["zn_test", "zn_test"], "event_id": ["event_a", "event_b"],
        "start_idx": [100, 300_000], "end_idx": [110, 300_007],
        "onset_s": [100 / 50_000., 300_000 / 50_000.],
        "subpop": [0, 1], "confidence": [.81, .72],
        # Gallery titles must use accepted-catalog amplitudes, not this appendix
        # feature's different baseline convention.
        "depth_peak_pA": [91., 61.],
    })
    return SimpleNamespace(tag="zn_test", detrended=signal, raw=signal + 100.,
                           fs_hz=50_000., duration_s=len(signal) / 50_000.,
                           catalog=catalog), assignments


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []

    def capture(fig, folder, name, *, cancel=None):
        figures.append(fig)
        return Path(folder) / f"{name}.png"

    # Inspect plotting artists and real CSV/JSON exports without an image encoder
    # or unrelated example-gallery selection obscuring the alignment regression.
    monkeypatch.setattr(galleries, "_save", capture)
    monkeypatch.setattr(galleries, "_stats", lambda *args: SimpleNamespace(baseline_pA=0., noise_pA=1.))
    monkeypatch.setattr(galleries, "_gallery", lambda *args, **kwargs: Path(args[3]) / f"{args[4]}.png")
    yield figures
    for figure in figures:
        figure.clear()


def test_overview_preserves_extrema_in_every_block_and_partial_tail():
    signal = np.array([0, -5, 2, 7, 3, 1, 0, 1,
                       1, 2, 3, -8, 0, 9, 2, 3,
                       2, -13, 4, 15, 1, 0, 2.])
    indices = galleries._overview_indices(signal, max_bins=3)
    assert np.all(np.diff(indices) > 0)
    assert indices[0] == 0 and indices[-1] == len(signal) - 1
    for start, end in ((0, 8), (8, 16), (16, 23)):
        shown = signal[indices[(indices >= start) & (indices < end)]]
        assert shown.min() == signal[start:end].min()
        assert shown.max() == signal[start:end].max()


def test_population_markers_use_event_minima_and_export_exact_coordinates(tmp_path, captured_figures):
    analysis, assignments = _recording()
    before = assignments.copy(deep=True)
    galleries.population_galleries(assignments, analysis, tmp_path)
    pd.testing.assert_frame_equal(assignments, before)
    top, raster = captured_figures[0].axes
    exported = pd.read_csv(tmp_path / "population_onsets_zn_test_events.csv")
    assert exported.event_id.tolist() == ["event_a", "event_b"]
    np.testing.assert_array_equal(exported.subpop, before.subpop)
    np.testing.assert_array_equal(exported.start_idx, before.start_idx)
    np.testing.assert_array_equal(exported.end_idx, before.end_idx)
    np.testing.assert_array_equal(exported.depth_peak_pA, [90., 60.])
    np.testing.assert_array_equal(exported.marker_sample_idx, [103, 300_007])
    np.testing.assert_array_equal(exported.marker_current_pA, [-90., -60.])
    np.testing.assert_array_equal(exported.onset_current_pA, [0., 0.])
    np.testing.assert_allclose(exported.marker_time_s, exported.marker_sample_idx / analysis.fs_hz,
                               rtol=0, atol=1e-15)
    for index, (marker, onset_marker) in enumerate(zip(top.collections, raster.collections)):
        row = exported.iloc[index]
        np.testing.assert_allclose(marker.get_offsets(), [[row.marker_time_s, row.marker_current_pA]])
        np.testing.assert_allclose(onset_marker.get_offsets(), [[row.start_idx / analysis.fs_hz, row.subpop + 1]])
        assert marker.get_zorder() > top.lines[0].get_zorder()
    assert top.get_shared_x_axes().joined(top, raster)
    assert top.get_xlim() == raster.get_xlim() == (0., analysis.duration_s)
    assert "event minima" in top.get_title()
    assert "Event onsets" in raster.get_title()
    assert top.get_legend_handles_labels()[1] == ["P1 (n=1)", "P2 (n=1)"]
    overview = pd.read_csv(tmp_path / "population_onsets_zn_test_trace_points.csv")
    assert {103, 300_007, len(analysis.detrended) - 1} <= set(overview.sample_idx)
    np.testing.assert_array_equal(overview.prepared_detrended_pA, analysis.detrended[overview.sample_idx])
    np.testing.assert_allclose(top.lines[0].get_ydata(), overview.prepared_detrended_pA)
    methods = json.loads((tmp_path / "population_onsets_zn_test_methods.json").read_text())
    assert methods["overview_method"] == "minmax_envelope"
    assert methods["population_labels"] == {"P1": 0, "P2": 1}
    assert methods["assigned_events"] == 2
    assert methods["shared_time_limits_s"] == [0., analysis.duration_s]


def test_population_rejects_assignments_to_rejected_catalog_rows(tmp_path, captured_figures):
    analysis, assignments = _recording()
    assignments.loc[0, ["start_idx", "end_idx"]] = [201, 205]
    with pytest.raises(ValueError, match="accepted catalog"):
        galleries.population_galleries(assignments, analysis, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("start,end", [(-1, 10), (10, 9), (10, 600_011), (10.5, 12), (10, np.nan)])
def test_population_minima_refuse_invalid_source_intervals(start, end):
    analysis, assignments = _recording()
    assignments["start_idx"] = assignments.start_idx.astype(float)
    assignments["end_idx"] = assignments.end_idx.astype(float)
    assignments.loc[0, ["start_idx", "end_idx"]] = [start, end]
    with pytest.raises(ValueError, match="bounds"):
        galleries._event_minima(analysis, assignments)


def test_recording_context_uses_the_same_extrema_preserving_overview(tmp_path, captured_figures):
    analysis, _ = _recording()
    galleries._trace_context(analysis, SimpleNamespace(baseline_pA=0., noise_pA=1.), tmp_path)
    points = pd.read_csv(tmp_path / "trace_context_zn_test_points.csv")
    for panel, source in ((1, analysis.raw), (2, analysis.detrended)):
        shown = points.loc[points.panel == panel]
        assert {103, 300_007, len(source) - 1} <= set(shown.sample_idx)
        assert shown.current_pA.min() == source.min()
        assert shown.current_pA.max() == source.max()
        np.testing.assert_array_equal(shown.current_pA, source[shown.sample_idx])
    methods = json.loads((tmp_path / "trace_context_zn_test_methods.json").read_text())
    assert methods["overview_method"] == "minmax_envelope"
    assert "overview_stride" not in methods
    panels = pd.read_csv(tmp_path / "trace_context_zn_test_panels.csv")
    assert panels.display_method.tolist() == ["minmax_envelope", "minmax_envelope",
                                            "every_source_sample", "every_source_sample"]


def test_population_gallery_honors_manual_acceptance_over_native_gate(tmp_path, captured_figures):
    analysis, assignments = _recording()
    analysis.catalog['accepted'] = True
    assignments.loc[0, ['start_idx', 'end_idx', 'onset_s']] = [201, 205, 201 / analysis.fs_hz]
    galleries.population_galleries(assignments, analysis, tmp_path)
    exported = pd.read_csv(tmp_path / 'population_onsets_zn_test_events.csv')
    assert exported.iloc[0].start_idx == 201
    assert exported.iloc[0].depth_peak_pA == 1.
    analysis.catalog.loc[2, 'accepted'] = False
    with pytest.raises(ValueError, match='accepted catalog'):
        galleries.population_galleries(assignments, analysis, tmp_path)


def _event_style_gallery(tmp_path, *, events=None, signal=None):
    from nanoscope.plot.study_figure_style import capture_figure
    if signal is None:
        signal = np.zeros(4000)
        signal[100:121] = -5.
        signal[2000:2021] = -50.
    if events is None:
        events = pd.DataFrame(dict(event_id=['weak_A', 'deep_B'],
            start_idx=[100, 2000], end_idx=[120, 2020], dwell_ms=[.42, .42],
            depth_peak_pA=[5., 50.], depth_sust_pA=[5., 50.]))
    analysis = SimpleNamespace(tag='test', detrended=signal, fs_hz=50000.)
    stats = SimpleNamespace(baseline_pA=0., noise_pA=1., quiet=np.ones(len(signal), bool))
    with capture_figure('gallery') as captured:
        galleries._gallery(analysis, events, stats, tmp_path, 'gallery', 'Accepted events')
    return captured.figure, events


def test_event_annotations_follow_identity_after_gallery_curation(tmp_path):
    from nanoscope.plot.study_figure_style import StudyFigureStyle
    original, events = _event_style_gallery(tmp_path)
    first = StudyFigureStyle(original)
    identities = [p['event_identity'] for p in first.panels()]
    saved = dict(event_titles={identities[0]: 'Weak event A', identities[1]: 'Deep event B'},
                 appearance={'title_size': 18, 'line_width': 2},
                 panels={'0': {'x_label': 'Latency (ms)'}}, colors={'#333333': '#ff0000'})
    # B moves from panel 2 to panel 1 after A is rejected. Its own annotation
    # follows; A's annotation must never label B's waveform.
    changed, _ = _event_style_gallery(tmp_path, events=events.iloc[[1]])
    try:
        styler = StudyFigureStyle(changed)
        before = changed.axes[0].lines[0].get_xydata().copy()
        assert styler.panels()[0]['event_identity'] == identities[1]
        styler.apply(**saved)
        ax = changed.axes[0]
        assert ax.get_title() == 'Deep event B'
        assert ax.get_xlabel() == 'Latency (ms)'
        assert ax.title.get_fontsize() == 18
        assert ax.lines[0].get_linewidth() == 2
        np.testing.assert_array_equal(ax.lines[0].get_xydata(), before)
        # Legacy ordinal titles fail closed without dropping generic styling.
        styler.apply(panels={'0': {'title': 'Weak event A', 'x_label': 'Latency (ms)'}})
        assert ax.get_title().startswith('deep_B\n')
        assert ax.get_xlabel() == 'Latency (ms)'
    finally:
        original.clear(); changed.clear()


@pytest.mark.parametrize('change', ['identity', 'bounds', 'measurement', 'trace'])
def test_event_title_is_not_reused_after_scientific_context_changes(tmp_path, change):
    from nanoscope.plot.study_figure_style import StudyFigureStyle
    original, events = _event_style_gallery(tmp_path)
    identity = StudyFigureStyle(original).panels()[0]['event_identity']
    edited = events.iloc[[0]].copy()
    signal = None
    if change == 'identity': edited.loc[:, 'event_id'] = 'replacement'
    elif change == 'bounds': edited.loc[:, 'end_idx'] = 121
    elif change == 'measurement': edited.loc[:, 'depth_peak_pA'] = 9.
    else:
        signal = np.zeros(4000); signal[100:121] = -9.
    changed, _ = _event_style_gallery(tmp_path, events=edited, signal=signal)
    try:
        styling = StudyFigureStyle(changed)
        automatic = changed.axes[0].get_title()
        assert styling.panels()[0]['event_identity'] != identity
        styling.apply(event_titles={identity: 'Old event annotation'})
        assert changed.axes[0].get_title() == automatic
    finally:
        original.clear(); changed.clear()


def test_empty_event_gallery_cannot_inherit_legacy_title(tmp_path):
    from nanoscope.plot.study_figure_style import StudyFigureStyle
    original, events = _event_style_gallery(tmp_path)
    empty, _ = _event_style_gallery(tmp_path, events=events.iloc[:0])
    try:
        StudyFigureStyle(empty).apply(panels={'0': {'title': 'Previously accepted event'}})
        assert empty.axes[0].get_title() == ''
        assert any(t.get_text() == 'No accepted events available' for t in empty.axes[0].texts)
    finally:
        original.clear(); empty.clear()
