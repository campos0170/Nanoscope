"""The shared Series provider keeps figure-owned edits in the existing Undo owner."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.study_figure_window import StudyFigureWindow
from .test_comparison_figures_panel import source, forbid_prepare, assert_tables_equal


def test_comparison_provider_edit_undo_and_frozen_restore(qtbot, monkeypatch):
    page = ComparisonFiguresWindow(); qtbot.addWidget(page)
    page.set_sources([source('a'), source('b', 'Recording B')], selected_keys=['a', 'b'])
    assert page.analyze()
    provider = page.inspector_adapter().format.series_provider
    offered = provider()
    assert [entry.identity_key for entry in offered] == ['a', 'b']
    before = deepcopy(page._tables)
    snapshot = deepcopy(page._snapshot)
    forbid_prepare(monkeypatch)
    widget = page._appearance_widget
    mark = widget.history_mark()
    entry = offered[0]
    widget.apply_appearance(replace(widget.state(), series_styles={
        entry.series_id: {'color': '#8833aa', 'marker_size_pt': 7.25},
        offered[1].series_id: {'visible': False}}))
    assert widget.history_mark() == mark + 1
    saved = page.appearance_state()
    assert saved['series_styles'][entry.series_id]['marker_size_pt'] == 7.25
    assert len(provider()) == 2  # hidden series remains reachable for Show
    assert page._snapshot == snapshot
    assert_tables_equal(page._tables, before)
    payload, tables = page.snapshot_state()
    restored = ComparisonFiguresWindow(); qtbot.addWidget(restored)
    restored.restore_state(payload, tables)
    assert restored.appearance_state()['series_styles'] == saved['series_styles']
    restored_descriptors = restored.series_descriptors()
    assert [(d.series_id, d.identity_key, d.capabilities) for d in restored_descriptors] == [
        (d.series_id, d.identity_key, d.capabilities) for d in offered]
    assert restored_descriptors[0].extra['effective_style']['marker_size_pt'] == 7.25
    assert restored_descriptors[0].extra['effective_style']['color'] == '#8833aa'
    assert 'visible' not in restored_descriptors[1].extra['effective_style']
    assert_tables_equal(restored._tables, before)
    assert widget.undo()
    assert not widget.state().series_styles
    assert widget.redo() and page.appearance_state()['series_styles'] == saved['series_styles']


@pytest.mark.parametrize('kind', ['study', 'report'])
def test_retained_owner_series_is_one_undo_and_roundtrips_whole_envelope(qtbot, kind):
    figure = Figure(figsize=(5, 3)); FigureCanvasAgg(figure)
    axis = figure.subplots()
    line, = axis.plot([1., 2., 3.], [3., 1., 4.], 'o-', color='#147d92', label='Trace')
    axis.legend()
    editor = StudyFigureWindow(figure, pd.DataFrame({'raw': [.001, .002]}), {},
                               name='Retained', provenance={}, owner_kind=kind)
    qtbot.addWidget(editor)
    offered = editor._styler.series_descriptors()
    assert offered and offered[0].identity_kind == 'artist'
    provider = editor._format_pane._spec.series_provider
    assert provider() == offered
    target = offered[0]
    data = line.get_xydata().copy()
    before = editor.style()
    widget = editor._appearance_widget
    mark = editor.history_mark()
    widget.apply_appearance(replace(widget.state(), series_styles={target.series_id: {
        'color': '#8833aa', 'line_width_pt': 2.5, 'marker_size_pt': 6.25, 'visible': False}}))
    editor._commit_style_run()
    assert editor.history_mark() == mark + 1
    assert not line.get_visible() and line.get_markersize() == 6.25
    assert to_rgba(line.get_color()) == to_rgba('#8833aa')
    saved = editor.style()
    assert editor.undo_style() and editor.style() == before and line.get_visible()
    assert editor.redo_style() and editor.style() == saved and not line.get_visible()
    np.testing.assert_array_equal(line.get_xydata(), data)
    assert editor._install_style(deepcopy(saved))
    assert widget.state().to_dict()['series_styles'] == saved['appearance']['series_styles']
