from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nanoscope.plot.incubation_figures import _save
from nanoscope.plot.incubation_galleries import _save as _gallery_save
from nanoscope.plot.study_figure_style import (
    StudyFigureStyle, capture_figure, styled_figures,
)


def canvas():
    figure = Figure(figsize=(5, 3))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.plot([1, 2, 3], [.5, 1.5, 1.], 'o-', color='#147d92', alpha=.4, label='2 hours')
    ax.scatter([1, 2, 3], [.8, 1.8, 1.3], color='#ca7627', label='30 hours')
    ax.set(title='Measured events', xlabel='Dwell (ms)', ylabel='Current (pA)', xscale='log')
    ax.legend()
    return figure, ax


def test_styles_preserve_data_scales_and_restore_originals():
    figure, ax = canvas()
    styling = StudyFigureStyle(figure)
    line = ax.lines[0]
    scatter = ax.collections[0]
    before = dict(x=line.get_xdata().copy(), y=line.get_ydata().copy(),
                  offsets=scatter.get_offsets().copy(), limits=(ax.get_xlim(), ax.get_ylim()),
                  widths=line.get_linewidth(), sizes=scatter.get_sizes().copy(),
                  background=figure.get_facecolor(), title_size=ax.title.get_fontsize())
    styling.apply(appearance={'title': 'Publication', 'background': 'dark', 'title_size': 24,
                              'tick_size': 14, 'marker_size': 9, 'line_width': 3, 'grid': True,
                              'legend_visible': False},
                  panels={'0': {'title': 'Edited label', 'x_label': 'Duration (ms)'}},
                  colors={'#147d92': '#7f00ff'})
    assert to_rgba(line.get_color()) == to_rgba('#7f00ff')
    assert line.get_alpha() == .4
    assert to_rgba(ax.get_legend().legend_handles[0].get_color()) == to_rgba('#7f00ff')
    assert ax.get_title() == 'Edited label'
    assert not ax.get_legend().get_visible()
    assert ax.get_xscale() == 'log'
    assert (ax.get_xlim(), ax.get_ylim()) == before['limits']
    np.testing.assert_array_equal(line.get_xdata(), before['x'])
    np.testing.assert_array_equal(line.get_ydata(), before['y'])
    np.testing.assert_array_equal(scatter.get_offsets(), before['offsets'])
    styling.apply()
    figure.canvas.draw()
    assert figure._suptitle is None
    assert ax.get_title() == 'Measured events'
    assert ax.get_xlabel() == 'Dwell (ms)'
    assert ax.get_legend().get_visible()
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert line.get_color() == '#147d92'
    assert line.get_linewidth() == before['widths']
    assert ax.title.get_fontsize() == before['title_size']
    assert figure.get_facecolor() == before['background']
    np.testing.assert_array_equal(scatter.get_sizes(), before['sizes'])


def test_continuous_colors_and_invalid_edits_are_preserved():
    figure, ax = canvas()
    continuous = ax.scatter([1, 2], [.4, .8], c=[0., 1.], cmap='viridis')
    style = StudyFigureStyle(figure)
    before = continuous.get_facecolors().copy()
    style.apply(colors={'#147d92': '#ff0000'})
    np.testing.assert_array_equal(continuous.get_facecolors(), before)
    with pytest.raises(ValueError, match='panel'):
        style.apply(panels={'bad': {'title': 'bad'}})
    assert to_rgba(ax.lines[0].get_color()) == to_rgba('#ff0000')


def test_removed_panels_and_categories_keep_saved_style_usable():
    figure, ax = canvas()
    style = StudyFigureStyle(figure)
    saved = {'appearance': {'title_size': 20},
             'panels': {'0': {'title': 'Current panel'}, '4': {'title': 'Absent after curation'}},
             'colors': {'#147d92': '#ff0000', '#123456': '#00ff00'}}
    from copy import deepcopy
    before = deepcopy(saved)
    style.apply(**saved)
    assert ax.get_title() == 'Current panel'
    assert to_rgba(ax.lines[0].get_color()) == to_rgba('#ff0000')
    assert saved == before


@pytest.mark.parametrize('legend_scope', ['axes', 'figure'])
def test_errorbar_legend_markers_caps_and_intervals_follow_color_and_reset(legend_scope):
    figure = Figure(figsize=(5, 3))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    bars = ax.errorbar([2., 10., 30.], [1., 2., 4.], xerr=.3, yerr=[.1, .2, .4],
                      fmt='o-', capsize=4, color='#147d92', alpha=.6, label='Cu event rate')
    ax.plot([2., 10., 30.], [2., 1., 3.], color='#ca7627', label='Other series')
    legend = ax.legend() if legend_scope == 'axes' else figure.legend()
    styler = StudyFigureStyle(figure)
    artists = legend.findobj(lambda artist: isinstance(artist, (Line2D, Collection)))
    original = [(artist, [to_rgba(artist.get_color())] if isinstance(artist, Line2D)
                 else list(map(tuple, artist.get_edgecolors()))) for artist in artists]
    # Horizontal and vertical intervals, four caps, and line/marker proxies.
    assert sum(any(np.allclose(c[:3], to_rgba('#147d92')[:3]) for c in colors)
               for _, colors in original) >= 7
    data_line, caps, intervals = bars.lines
    original_points = data_line.get_xydata().copy()
    original_intervals = [np.asarray(interval.get_segments()).copy() for interval in intervals]
    frame_face, frame_edge = legend.get_frame().get_facecolor(), legend.get_frame().get_edgecolor()
    styler.apply(colors={'#147d92': '#7f00ff'})
    for artist, colors in original:
        actual = ([to_rgba(artist.get_color())] if isinstance(artist, Line2D)
                  else list(map(tuple, artist.get_edgecolors())))
        for before, after in zip(colors, actual):
            expected = (*to_rgba('#7f00ff')[:3], before[3]) if np.allclose(before[:3], to_rgba('#147d92')[:3]) else before
            np.testing.assert_allclose(after, expected)
    assert legend.get_frame().get_facecolor() == frame_face
    assert legend.get_frame().get_edgecolor() == frame_edge
    np.testing.assert_array_equal(data_line.get_xydata(), original_points)
    for interval, saved in zip(intervals, original_intervals):
        np.testing.assert_array_equal(interval.get_segments(), saved)
    styler.apply()
    for artist, colors in original:
        actual = ([to_rgba(artist.get_color())] if isinstance(artist, Line2D)
                  else list(map(tuple, artist.get_edgecolors())))
        np.testing.assert_array_equal(actual, colors)


def test_capture_keeps_one_original_and_detaches_data(tmp_path):
    source = pd.DataFrame({'event_id': ['a'], 'current': [1.23456789]})
    with styled_figures({'target': {'appearance': {'title': 'Would change'}}}):
        with capture_figure('target.png') as selected:
            discarded, _ = canvas()
            _save(discarded, tmp_path, 'other', source)
            chosen, _ = canvas()
            _save(chosen, tmp_path, 'target', source)
    assert selected.figure is chosen
    assert not discarded.axes
    assert chosen._suptitle is None
    assert not list(tmp_path.iterdir())
    source.loc[0, 'current'] = 0
    assert selected.data.current.iloc[0] == 1.23456789
    chosen.clear()


def test_capture_is_thread_local_and_normal_regeneration_styled(tmp_path):
    with capture_figure('target') as selected:
        def worker():
            figure, _ = canvas()
            return _gallery_save(figure, tmp_path, 'worker')
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(worker).result().is_file()
        figure, _ = canvas()
        _gallery_save(figure, tmp_path, 'target')
    assert selected.figure is figure
    selected.figure.clear()
    # The normal save hook styles only its matching name and keeps data exact.
    with styled_figures({'styled': {'appearance': {'background': 'dark'}}}):
        figure, _ = canvas()
        path = _gallery_save(figure, tmp_path, 'styled')
    from PIL import Image
    assert Image.open(path).convert('RGB').getpixel((0, 0)) == (23, 25, 29)


def test_capture_missing_name_reports_problem(tmp_path):
    with pytest.raises(ValueError, match='did not produce'):
        with capture_figure('missing'):
            figure, _ = canvas()
            _save(figure, tmp_path, 'other')


def test_failed_capture_context_does_not_leak_canvas(tmp_path):
    with pytest.raises(RuntimeError):
        with capture_figure('target') as result:
            figure, _ = canvas()
            _save(figure, tmp_path, 'target')
            raise RuntimeError('cancelled by caller')
    assert not result.figure.axes
    figure, _ = canvas()
    assert _save(figure, tmp_path, 'normal').is_file()


@pytest.mark.parametrize('titles', [{'not-an-identity': 'title'}, {'a'*64: 42}])
def test_invalid_event_title_identity_cannot_partially_apply_style(titles):
    figure, ax = canvas()
    styling = StudyFigureStyle(figure)
    before = ax.get_title(), figure.get_facecolor()
    try:
        with pytest.raises(ValueError, match='Event titles'):
            styling.apply(event_titles=titles, appearance={'background': 'dark'})
        assert (ax.get_title(), figure.get_facecolor()) == before
    finally:
        figure.clear()
