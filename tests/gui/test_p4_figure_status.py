"""Visible figure state and pinned style identity are metadata reads, not analysis."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from nanoscope.core.figure_library import figure_status
from nanoscope.core.figure_spec import FigurePresentation
from nanoscope.core.figure_style import FigureStyle
from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.figure_list import FigureList
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.plot_appearance import PlotAppearance

from . import p1_hosting as H
from .p2_shell_rig import Controller, record


def test_visible_state_keeps_names_selection_and_loaded_state_distinct(qtbot):
    listing = FigureList(); qtbot.addWidget(listing)
    current, stale, empty = (record(name) for name in ('Current name', 'Stale name', 'Empty name'))
    stale.spec.result.status = 'stale'; empty.spec.result.status = 'empty'
    records = (current, stale, empty)
    key = current.spec.figure_id
    listing.set_records(records, (), selected=(key,), current=key, loaded=(key,))
    assert listing._items[key].text(0) == 'Current name'
    assert 'Current name' in listing._items[key].toolTip(0)
    assert listing._items[key].text(1) == 'Current'
    assert listing._items[stale.spec.figure_id].text(1) == 'Stale · Dormant'
    assert listing._items[empty.spec.figure_id].text(1) == 'No data · Dormant'
    selected = listing.selected_ids(); activated = []
    listing.liveActivated.connect(activated.append)
    listing.set_records(records, (), selected=selected, current=key)
    assert listing.selected_ids() == selected and not activated
    assert listing._items[key].text(1) == 'Current · Dormant'


def test_saved_lineage_tracks_sources_without_following_live_figure_freshness(qtbot):
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    live = record('Current live figure'); owner = Controller(live)
    explorer.bind_figure_workspace(owner)
    saved = NS(artifact_id='figure.saved', payload={'name': 'Saved', 'page_id': 'iv',
        'capture_status': 'current', 'source_state': {'revision': 1}})
    before = deepcopy(saved.payload)
    source = {'revision': 1}
    explorer.set_saved_figure_provider(lambda: (saved,), status_provider=lambda entry: figure_status(entry, source))
    explorer.refresh_figures()
    assert explorer.figures_list()._items[saved.artifact_id].text(1) == 'Current'
    source['revision'] = 2
    explorer.schedule_figures_refresh()
    qtbot.waitUntil(lambda: explorer.figures_list()._items[saved.artifact_id].text(1) == 'Outdated')
    assert explorer.figures_list()._items[live.spec.figure_id].text(1) == 'Current · Dormant'
    assert saved.payload == before and explorer.page_count() == 0 and not owner.calls
    explorer.set_saved_figure_provider(lambda: (saved,))
    explorer.refresh_figures()
    assert explorer.figures_list()._items[saved.artifact_id].text(1) == 'Saved snapshot'


def test_dormant_style_uses_pinned_snapshot_not_latest_registry_or_equal_pixels(qtbot):
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    first = FigureStyle('project.apo', 1, 'Apo look', {'font_family': 'Arial'})
    value = record('Apo figure')
    value.spec.presentation = FigurePresentation(style_ref=first.ref.to_dict(), style_snapshot=first.to_dict())
    owner = Controller(value)
    owner.available_styles = lambda: (replace(first, revision=2, name='Renamed look'),)
    explorer.bind_figure_workspace(owner)
    explorer.refresh_figures()
    button = explorer.toolbar().style_button
    assert 'Apo look · r1' in button.text() and 'Renamed' not in button.text()
    assert 'inherited' in button.toolTip() and explorer.page_count() == 0 and not owner.calls
    value.spec.presentation = replace(value.spec.presentation, overrides={'title_size_pt': 9.})
    owner.recordChanged.emit(value.spec.figure_id)
    qtbot.waitUntil(lambda: '●' in button.text())
    value.spec.presentation = FigurePresentation(overrides={'font_family': 'Arial'})
    explorer.refresh_figures()
    assert button.text() == 'Style: custom ▾'  # same family is not proof of a named reference


def test_live_style_name_and_override_follow_native_undo_without_checkpoint(qtbot):
    page = H.SyntheticPlotPage('iv')
    explorer = H.hosted(qtbot, page, 'iv')
    widget = page._appearance_widget
    style = FigureStyle('project.apo', 3, 'Apo look', {'font_family': 'Arial'})
    widget.apply_appearance(PlotAppearance.from_mapping(bind_native_style({}, style)))
    button = explorer.toolbar().style_button
    assert 'Apo look · r3' in button.text()
    widget.edit_text_style('title', TextStyle(bold=True))
    assert '●' in button.text()
    widget.undo()
    assert 'Apo look · r3' in button.text() and '●' not in button.text()
    assert page.updates == 0


def test_new_figure_menu_exposes_each_tools_inputs_without_disabling_empty_drafts(qtbot):
    from nanoscope.gui.analysis_catalog import ENTRY_BY_ROUTE, NEW_FIGURES
    listing = FigureList(); qtbot.addWidget(listing)
    assert listing.new_menu.toolTipsVisible()
    actions = listing.new_menu.actions()
    requested = []
    listing.newRequested.connect(lambda *args: requested.append(args))
    for action, (_label, kind, _recipe) in zip(actions, NEW_FIGURES):
        assert ENTRY_BY_ROUTE[kind].inputs in action.toolTip()
        assert action.isEnabled() and 'Update plot' in action.toolTip()
    actions[0].trigger()
    assert requested == [('distributions', {'plot_type': 'Histogram'})]


@pytest.mark.parametrize('held_by', ['record', 'legacy_frozen', 'legacy_pending'])
def test_saved_live_header_is_a_result_while_snapshot_library_keeps_its_name(qtbot, held_by):
    from nanoscope.gui.main_window import MainWindow
    from nanoscope.gui.analysis_catalog import STATUS_HELP
    value = record('Editable saved result')
    value.spec.kind = 'iv'
    value.spec.result.status = 'frozen'
    artifact_id = MainWindow._analysis_page_artifact('iv')
    owner = NS(_figure_workspace=NS(get=lambda _key: value) if held_by == 'record' else None,
        _workspace_frozen={artifact_id} if held_by == 'legacy_frozen' else set(),
        _workspace_pending={artifact_id} if held_by == 'legacy_pending' else set(),
        _analysis_page_artifact=MainWindow._analysis_page_artifact)
    page = H.SyntheticPlotPage('iv'); qtbot.addWidget(page)
    shown, state = MainWindow._analysis_explorer_status(owner, 'iv', page)
    assert (shown, state) == ('SAVED RESULT', 'snapshot')  # internal state code stays compatible
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    explorer._set_status(shown, state)
    assert explorer._status_label.text() == 'SAVED RESULT'
    assert 'editable live figure' in explorer._status_label.toolTip()
    assert 'Frozen Snapshots are separate' in STATUS_HELP
    assert value.spec.result.status == 'frozen' and page.updates == 0
    saved = NS(payload={'page_id': 'iv'})
    assert figure_status(saved, None)[0] == 'Saved snapshot'


@pytest.mark.parametrize('kind', ['distribution_summaries', 'comparison_figures'])
def test_summary_and_comparison_caption_names_generated_result_without_mutating_it(qtbot, kind):
    page = H.page_builders()[kind](); qtbot.addWidget(page)
    page._snapshot = ({'series': [{'label': 'Apo', 'n': 3}], 'primitives': []}
                      if kind == 'distribution_summaries' else
                      {'sources': [{'label': 'Apo', 'n': 3}]})
    before = deepcopy(page._snapshot)
    page._stale = False
    page._update_status()
    assert page._status.text().startswith('Generated result · Apo:')
    assert page._status.accessibleDescription() == page._status.toolTip()
    page._stale = True
    page._update_status()
    assert page._status.text().startswith('Unapplied data/analysis changes')
    assert 'snapshot' not in page._status.toolTip().lower()
    assert page._snapshot == before and page._stale
