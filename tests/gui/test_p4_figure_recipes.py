"""Explicit recipes carry their look, optionally words, and never claim computed results."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6 import QtCore

from nanoscope.core.figure_codecs import encode_legacy
from nanoscope.gui.figure_recipes import RecipeEntry, initial_recipe, save_page, rename, delete, entries, SUPPORTED
from nanoscope.gui.figure_adapter_pages import _native_appearance
from nanoscope.gui.figure_adapter_plots import _native_config
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plot_templates import encode_window_template, load_window_templates, store_window_templates
from nanoscope.gui.plots_panel import PlotConfig, plot_config_to_json, FrozenMap, PlotsWindow
from .test_p2_workspace_controller import rig  # noqa: F401


def distribution_recipe():
    config = PlotConfig()
    config = replace(config, style=replace(config.style, title='Apo dwell', font_size_pt=17.,
        text_styles=FrozenMap({'title': {'bold': True}, 'annotation.provenance': {'text': 'Typed caption'}}),
        fit_label=FrozenMap({'template': 'My fit: {equation}', 'color': '#123456'})))
    return RecipeEntry('distributions', 'Dwell setup', plot_config_to_json(config))


@pytest.mark.parametrize('include_words', (False, True))
def test_distribution_recipe_words_are_explicit_but_look_always_travels(rig, monkeypatch, include_words):
    entry = distribution_recipe(); original = deepcopy(entry.payload)
    recipe = initial_recipe(entry, include_words=include_words)
    owner = rig.owner
    monkeypatch.setattr(owner, 'activate', lambda _key: False)
    key = owner.create(entry.kind, initial_recipe=recipe)
    record = owner.get(key)
    config = _native_config(encode_legacy(record.spec, record.resources))
    assert config.style.font_size_pt == 17.
    assert config.style.title == ('Apo dwell' if include_words else '')
    assert config.style.text_styles['title']['bold'] is True
    if include_words:
        assert config.style.text_styles['annotation.provenance']['text'] == 'Typed caption'
    else:
        assert 'annotation.provenance' not in config.style.text_styles  # empty styles are pruned
    assert config.style.fit_label['template'] == ('My fit: {equation}' if include_words else '')
    assert record.spec.result.status == 'empty' and owner.history(key).mark == 0 and not rig.built
    assert entry.payload == original
    pending = owner._metadata(owner._entries[key])['initial_recipe']
    assert 'recipe_style' not in pending and 'style' not in pending['plot_template']
    assert owner.apply_look_to_figure(key, PlotAppearance(title_size=23), keep_hand=False)
    assert owner.undo(key)
    restored = _native_config(encode_legacy(owner.get(key).spec, owner.get(key).resources))
    assert restored.style == config.style


@pytest.mark.parametrize('include_words', (False, True))
def test_native_recipe_preserves_fractional_look_and_optional_words_without_sources(rig, monkeypatch, include_words):
    from nanoscope.gui.figure_text import TextStyle
    appearance = PlotAppearance(title='My ECDF', x_label='Dwell caption', title_size=8.125,
        line_width=.625, text=(('title', TextStyle(bold=True)),
                              ('legend.entry.old-source', TextStyle(text='Old pore'))))
    payload = encode_window_template('distribution-summaries', {'kind': 'ecdf', 'metric': 'dwell_s'}, appearance.to_dict())
    entry = RecipeEntry('distribution_summaries', 'ECDF recipe', payload)
    recipe = initial_recipe(entry, include_words=include_words)
    owner = rig.owner; monkeypatch.setattr(owner, 'activate', lambda _key: False)
    key = owner.create(entry.kind, initial_recipe=recipe)
    record = owner.get(key)
    restored = _native_appearance(encode_legacy(record.spec, record.resources), entry.kind)
    assert restored.title_size == 8.125 and restored.line_width == .625
    assert restored.title == ('My ECDF' if include_words else None)
    assert restored.text_style('title').bold is True
    assert not restored.text_style('legend.entry.old-source').text
    assert not record.spec.sources and record.spec.result.status == 'empty' and not rig.built


def test_shared_recipe_store_renames_exact_payload_and_preserves_unrelated_future_entries():
    kind = 'distribution_summaries'; store = 'distribution-summaries'
    before = load_window_templates(store)
    try:
        future = {'format': 'future', 'version': 99, 'keep': [None, {'raw': 7}]}
        store_window_templates(store, {'Future': future})
        page = SimpleNamespace(template_settings=lambda: ({'kind': 'box'}, {'title': 'Words', 'line_width': .625}))
        entry = save_page(page, kind, 'My recipe')
        rename(entry, 'Renamed')
        current = next(item for item in entries() if item.kind == kind and item.name == 'Renamed')
        assert current.payload == entry.payload
        assert load_window_templates(store)['Future'] == future
        delete(current)
        assert load_window_templates(store) == {'Future': future}
    finally:
        store_window_templates(store, before)


@pytest.mark.parametrize('kind', ('distributions', 'distribution_summaries', 'comparison_figures',
                                  'event_waveforms', 'noise', 'iv', 'voltage_series', 'subpopulations'))
def test_common_export_recipe_routes_match_real_capabilities(qtbot, kind):
    from nanoscope.gui.analysis_explorer import AnalysisExplorer
    from .p2_shell_rig import Controller, record
    from . import p1_hosting as H
    value = record(kind=kind); owner = Controller(value)
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer); explorer.bind_figure_workspace(owner)
    page = H.SyntheticPlotPage(kind)
    explorer.add_page(value.spec.figure_id, page, kind=kind)
    actions = explorer.adapter(value.spec.figure_id).exports()
    labels = [entry.text for entry in actions]
    assert ('Save as recipe…' in labels) == (kind in SUPPORTED)
    assert not any(group.key == 'recipes' for group in explorer.adapter(value.spec.figure_id).groups)
    requests = []
    explorer.manageRecipesRequested.connect(requests.append)
    next(entry for entry in actions if 'recipes…' in entry.text).run()
    assert requests == [kind] and page.updates == 0 and not owner.calls


@pytest.mark.parametrize('include_words', (False, True))
def test_actual_new_recipe_hydration_uses_seeded_look_without_compute(qtbot, monkeypatch, include_words):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow(recovery_enabled=False); qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    factory = owner._factory; calls = []
    def guarded(kind, key):
        page = factory(kind, key)
        monkeypatch.setattr(page, '_run_plot', lambda *args: calls.append('plot'))
        return page
    owner._factory = guarded
    key = owner.create('distributions', initial_recipe=initial_recipe(distribution_recipe(), include_words=include_words))
    page = owner.page(key)
    assert page is not None and not calls
    assert page.config().style.font_size_pt == 17.
    assert page.config().style.title == ('Apo dwell' if include_words else '')
    assert owner.history(key).mark == 0 and owner.get(key).spec.result.status == 'empty'


def test_managed_distributions_import_button_is_reachable_and_keeps_native_undo(qtbot, monkeypatch, tmp_path):
    import pandas as pd
    from PySide6 import QtWidgets
    from nanoscope.gui.analysis_explorer import AnalysisExplorer
    from nanoscope.gui.plots_panel import write_plot_config
    from .p2_shell_rig import Controller, record
    from . import p1_hosting as H

    value = record(kind='distributions')
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(Controller(value))
    page = H.page_builders()['distributions']()
    explorer.show_page(value.spec.figure_id, page, kind='distributions')
    explorer.show(); qtbot.waitExposed(explorer)
    assert explorer.reveal('configuration_import')
    QtWidgets.QApplication.processEvents()
    frame = explorer.inspector().pane(value.spec.figure_id).data_pane.group('configuration_import')
    assert frame.body.isAncestorOf(page._import_cfg_btn) and page._import_cfg_btn.isVisible()
    assert not any(group.key == 'recipes' for group in explorer.adapter(value.spec.figure_id).groups)
    page._history_commit()
    original = page.config()
    sources = {key: source.df.copy(deep=True) for key, source in page._sources.items()}
    path = tmp_path / 'imported-settings.json'
    write_plot_config(path, replace(original, style=replace(original.style, title='Imported caption')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *_a, **_k: (str(path), ''))
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: warnings.append(args))
    H.click(page._import_cfg_btn)
    assert not warnings and page.config().style.title == 'Imported caption'
    H.press_undo(explorer)
    assert page.config() == original
    for key, source in page._sources.items():
        pd.testing.assert_frame_equal(source.df, sources[key])


@pytest.mark.parametrize('payload', ('not valid JSON', '{"format":"nanoscope-plot-config","version":999}'))
def test_managed_distributions_import_refusals_preserve_the_figure(qtbot, monkeypatch, tmp_path, payload):
    from PySide6 import QtWidgets
    from nanoscope.gui.analysis_explorer import AnalysisExplorer
    from .p2_shell_rig import Controller, record
    from . import p1_hosting as H

    value = record(kind='distributions')
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(Controller(value))
    page = H.page_builders()['distributions']()
    explorer.show_page(value.spec.figure_id, page, kind='distributions')
    explorer.show(); qtbot.waitExposed(explorer)
    assert explorer.reveal('configuration_import')
    original = page.config()
    path = tmp_path / 'unreadable-settings.json'; path.write_text(payload)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *_a, **_k: (str(path), ''))
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: warnings.append(args))
    monkeypatch.setattr(page, 'apply_config', lambda *_a, **_k: pytest.fail('Invalid JSON reached configuration install'))
    H.click(page._import_cfg_btn)
    assert len(warnings) == 1 and path.name in warnings[0][-1]
    assert page.config() == original
