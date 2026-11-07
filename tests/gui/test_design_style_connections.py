"""Cross-route editing retains one appearance owner and never reruns science."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_hex
from PySide6 import QtWidgets

from nanoscope.core.figure_style import builtin_styles
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.fit_label import FitLabelState
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.figure_text import TextStyle


@pytest.mark.parametrize('keep', [True, False])
def test_style_keep_hand_covers_fit_label_ink_but_always_keeps_its_content(keep):
    fit = FitLabelState(size_pt=30, color='#ff0000', template='Literal {equation}',
                        fields=('r2',), placement='custom', offset=(15., 22.))
    before = PlotAppearance(fit_label=fit).with_text_style('title', TextStyle(size_pt=30))
    named = builtin_styles()[0]
    look = PlotAppearance.from_mapping(bind_native_style({}, named))
    after = before.restyled(look, keep_hand=keep)
    assert after.fit_label.template == fit.template
    assert after.fit_label.fields == fit.fields
    assert after.fit_label.offset == fit.offset and after.fit_label.placement == 'custom'
    assert after.fit_label.size_pt == (30 if keep else None)
    assert after.fit_label.color == ('#ff0000' if keep else None)
    assert after.effective_fit_label_state().size_pt == (30 if keep else 7)
    assert after.text_style('title').size_pt == (30 if keep else None)


@pytest.mark.parametrize('kind', ['distribution_summaries', 'comparison_figures'])
def test_source_editor_undo_redo_and_dormant_history_share_the_same_value(qtbot, monkeypatch, kind):
    from .test_comparison_figures_panel import source
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
    from nanoscope.gui.figure_adapter_pages import NativeFigureAdapter, move_history
    from nanoscope.core.figure_codecs import decode_legacy, encode_legacy, merge_native_capture
    cls = ComparisonFiguresWindow if kind == 'comparison_figures' else DistributionSummaryWindow
    page = cls(); qtbot.addWidget(page)
    offered = source(); page.set_sources([offered]); page.analyze()
    numbers = {key: value.copy(deep=True) for key, value in page._tables.items()}
    widget = page._appearance_widget; widget.clear_history()
    before = page.source_appearance_state(); mark = widget.history_mark()
    monkeypatch.setattr(page, 'analyze', lambda: pytest.fail('Appearance ran analysis'))
    editor = page._source_appearance_editor
    editor.label.setText('Local alias'); editor._color_override = '#e69f00'; editor.apply.click()
    changed = {offered.key: {'label': 'Local alias', 'color': '#e69f00'}}
    assert page.source_appearance_state() == changed and widget.history_mark() == mark + 1
    assert widget.undo() and page.source_appearance_state() == before
    assert editor.state() == before
    assert widget.redo() and page.source_appearance_state() == changed
    native = NativeFigureAdapter(page, kind)
    record = decode_legacy(native.capture_native(), figure_id=str(uuid4()), name='Audit').record
    undone, history = move_history(record, native.capture_transient(), 'undo')
    projected = encode_legacy(undone.spec, undone.resources)
    assert projected.payload['source_appearance'] == before
    assert projected.payload['appearance']['source_appearance'] == before
    assert history.future
    # An older caller emits only the top-level map. Its explicit clear must not
    # revive a previously saved nested map as an unknown forward field.
    captured = native.capture_native()
    legacy = deepcopy(captured.payload)
    legacy['appearance'].pop('source_appearance')
    legacy['source_appearance'] = {}
    cleared = merge_native_capture(replace(captured, payload=legacy), record)
    legacy_projection = encode_legacy(cleared.spec, cleared.resources).payload
    assert legacy_projection['source_appearance'] == {}
    assert not legacy_projection['appearance'].get('source_appearance')
    for key, frame in numbers.items():
        pd.testing.assert_frame_equal(page._tables[key], frame, check_exact=True)
    payload, tables = page.snapshot_state()
    payload['appearance'].pop('source_appearance')  # old top-level-only project
    reopened = cls(); qtbot.addWidget(reopened); reopened.restore_state(payload, tables)
    assert reopened.source_appearance_state() == changed
    assert reopened._appearance_widget.state().source_appearance == changed
    assert reopened._appearance_widget.state().style_only().source_appearance is None
    assert reopened._appearance_widget.state().template_form().source_appearance is None


def _report(qtbot, style=None):
    from nanoscope.gui.report_figure_window import ReportFigureWindow
    figure = Figure(figsize=(4, 3)); FigureCanvasAgg(figure)
    axis = figure.subplots(); axis.plot([1., 2.], [2., 4.], color='#147d92', label='Source')
    axis.set_title('Automatic title')
    editor = ReportFigureWindow(figure, pd.DataFrame({'x': [1., 2.]}), name='audit',
                                provenance={}, style=style)
    qtbot.addWidget(editor); editor.show()
    return editor, axis


@pytest.mark.parametrize('kind, view', [('distribution_summaries', 'summary'), ('comparison_figures', 'comparison')])
def test_legacy_source_color_replaces_only_that_sources_shared_ink_in_one_undo(qtbot, kind, view):
    from .test_comparison_figures_panel import source
    from nanoscope.core.figure_series import stable_series_id
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
    from nanoscope.gui.figure_series import pg_series_descriptors
    cls = ComparisonFiguresWindow if kind == 'comparison_figures' else DistributionSummaryWindow
    page = cls(); qtbot.addWidget(page)
    offered = source(); page.set_sources([offered]); page.analyze(); page.show()
    widget = page._appearance_widget
    identity = stable_series_id(view, 'data', offered.key)
    other = stable_series_id(view, 'data', 'another-source')
    widget.apply_appearance(replace(widget.state(), series_styles={
        identity: {'color': '#cc00cc', 'line_width_pt': 2.5, 'marker_size_pt': 8},
        other: {'color': '#123456', 'visible': False}}))
    before, mark = widget.state(), widget.history_mark()
    tables = {key: value.copy(deep=True) for key, value in page._tables.items()}
    legacy = page._source_appearance_editor
    assert not legacy.isVisibleTo(page) and not page.inspector_adapter().format.extras
    legacy.label.setText('Source alias'); legacy._color_override = '#e69f00'; legacy.apply.click()
    assert widget.history_mark() == mark + 1
    assert widget.state().series_styles[identity] == {'line_width_pt': 2.5, 'marker_size_pt': 8}
    assert widget.state().series_styles[other] == before.series_styles[other]
    descriptors = (page.series_descriptors() if kind == 'comparison_figures'
                   else pg_series_descriptors([page._plot]))
    target = next(entry for entry in descriptors if entry.series_id == identity)
    assert target.label == 'Source alias'
    assert target.extra['effective_style']['color'] == '#e69f00'
    assert widget.undo() and widget.state() == before
    for key, frame in tables.items():
        pd.testing.assert_frame_equal(page._tables[key], frame, check_exact=True)


@pytest.mark.parametrize('colors', [{'#147d92': '#ff0000'}, {'#147D92': 'red'}])
def test_report_legacy_words_and_colors_migrate_to_one_visible_format_owner(qtbot, colors):
    from nanoscope.core.figure_series import stable_series_id
    old = {'panels': {'0': {'title': 'Old saved title'}}, 'colors': colors}
    unchanged = deepcopy(old)
    editor, axis = _report(qtbot, old)
    controller = next(c for c in editor._text_controllers if c.panel != 'figure' and c.target('title'))
    assert controller.target('title').style.text == 'Old saved title'
    assert axis.get_title() == 'Old saved title'
    assert not editor._panel_edits['title'].isVisibleTo(editor)
    assert not editor._color.isVisibleTo(editor)
    assert not editor.style()['panels'] and not editor.style()['colors']
    assert to_hex(axis.lines[0].get_color()) == '#ff0000'
    data = axis.lines[0].get_xydata().copy()
    before = editor.style()
    controller.set_style_property('title', 'text', 'Format title')
    assert axis.get_title() == 'Format title'
    assert editor.undo_style() and editor.style() == before
    assert axis.get_title() == 'Old saved title'
    controller.set_style_property('title', 'text', None)
    assert axis.get_title() == 'Automatic title'  # no hidden legacy fallback
    key = stable_series_id('study', 'data', '#147d92')
    state = editor._appearance_widget.state()
    editor._appearance_widget.apply_appearance(replace(state, series_styles={key: {'color': '#009e73'}}))
    assert to_hex(axis.lines[0].get_color()) == '#009e73'
    editor._appearance_widget.apply_appearance(replace(editor._appearance_widget.state(), series_styles=None))
    assert to_hex(axis.lines[0].get_color()) == '#147d92'
    np.testing.assert_array_equal(axis.lines[0].get_xydata(), data)
    assert old == unchanged


def test_retained_editor_batch_hook_honors_keep_hand_and_undo(qtbot):
    editor, axis = _report(qtbot)
    controller = next(c for c in editor._text_controllers if c.panel != 'figure' and c.target('title'))
    controller.set_style_property('title', 'size_pt', 30)
    before = editor.style(); history = editor.history()
    look = PlotAppearance.from_mapping(bind_native_style({}, builtin_styles()[0], 'matplotlib_appearance'))
    assert editor.apply_look(look, keep_hand=False)
    assert axis.title.get_fontsize() == 7
    assert editor.history().mark == history.mark + 1
    assert editor.undo_style() and editor.style() == before
    assert axis.title.get_fontsize() == 30


def test_report_style_menu_offers_pinned_computer_styles(qtbot, monkeypatch):
    from nanoscope.gui import computer_styles
    editor, _axis = _report(qtbot)
    style = replace(builtin_styles()[1], style_id='computer.audit', name='Computer only',
                    provenance={'computer_style': True})
    monkeypatch.setattr(computer_styles, 'styles', lambda: (style,))
    menu = QtWidgets.QMenu(editor); editor._fill_style_menu(menu)
    actions = menu.actions(); names = [a.text() for a in actions]
    assert names.index('Computer only') > names.index('On this computer')
    next(a for a in actions if a.text() == 'Computer only').trigger()
    assert editor.style()['style_binding']['ref'] == style.ref.to_dict()


def test_all_live_styles_include_retained_editors_and_batch_undo_tracks_their_owner(qtbot):
    from nanoscope.gui.plot_styles import StyleService
    first, first_axis = _report(qtbot)
    second, _second_axis = _report(qtbot)
    targets = [('one', 'Report one', first), ('two', 'Study two', second)]
    explorer = SimpleNamespace(figure_workspace=lambda: None, current_page_id=lambda: None,
                               page_ids=lambda: (), adapter=lambda _key: None,
                               statusBar=lambda: SimpleNamespace(showMessage=lambda *_args: None))
    service = StyleService(explorer)
    service.set_external_provider(lambda: targets)
    look = PlotAppearance.from_mapping(bind_native_style({}, builtin_styles()[0]))
    before = [editor.style() for editor in (first, second)]
    assert not service.apply('Audit', look, scope='selected').applied
    result = service.apply('Audit', look, scope='all', keep_hand=False)
    assert set(result.applied) == {'retained:one', 'retained:two'} and not result.failed
    assert first_axis.title.get_fontsize() == 7
    assert service.batch_intact() and service.undo_last_batch() == 2
    assert [editor.style() for editor in (first, second)] == before

    service.apply('Audit', look, scope='all', keep_hand=False)
    controller = next(c for c in first._text_controllers if c.panel != 'figure' and c.target('title'))
    controller.set_style_property('title', 'text', 'Later edit')
    assert not service.batch_intact() and service.undo_last_batch() == 0
    assert first_axis.get_title() == 'Later edit'

    # A new editor with the same public ID cannot inherit the old batch's Undo.
    service.apply('Other', PlotAppearance(title_size=13), scope='all', keep_hand=False)
    replacement, _axis = _report(qtbot)
    targets[0] = ('one', 'Report one', replacement)
    assert not service.batch_intact() and service.undo_last_batch() == 0
