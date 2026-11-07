"""The local shared-style catalog is separate, immutable and portable in figure bindings."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from PySide6 import QtCore

from nanoscope.core.figure_style import FigureStyle
from nanoscope.core.figure_spec import thaw_json
from nanoscope.gui import computer_styles
from nanoscope.gui.plot_appearance import PlotAppearanceWidget
from nanoscope.gui.plot_styles import StyleService
from .test_p2_workspace_controller import rig  # noqa: F401


@pytest.fixture(autouse=True)
def isolated_catalog(qapp):
    settings = QtCore.QSettings()
    before = {key: settings.value(key) for key in (computer_styles.KEY, PlotAppearanceWidget._PRESETS_KEY)}
    for key in before:
        settings.remove(key)
    yield
    for key, value in before.items():
        settings.remove(key) if value is None else settings.setValue(key, value)
    settings.sync()


def test_full_shared_look_and_columns_survive_local_revisions_and_deletion():
    source = FigureStyle('project.apo', 2, 'Source',
        {'line_width_pt': .625, 'title_size_pt': 8.125, 'future': {'retain': [None, 3]}},
        columns={'single_mm': 85.125, 'extra': {'why': 'measured'}},
        provenance={'inherit_native_defaults': True, 'project_style': True})
    saved = computer_styles.save(source, 'Lab style')
    assert saved.revision == 1 and saved.style_id.startswith('computer.')
    assert saved.look == source.look and saved.columns == source.columns
    assert saved.provenance['inherit_native_defaults'] and 'project_style' not in saved.provenance
    assert computer_styles.styles() == (saved,)
    revised = computer_styles.save(replace(source, look={'title_size_pt': 10.5}), 'Lab style', overwrite=True)
    assert revised.style_id == saved.style_id and revised.revision == 2
    assert saved.look['title_size_pt'] == 8.125
    assert computer_styles.delete('Lab style') and computer_styles.styles() == ()


def test_unrecognized_catalog_is_never_overwritten_and_unknown_entries_roundtrip():
    settings = QtCore.QSettings()
    future = json.dumps({'version': 99, 'future': [1, None]})
    settings.setValue(computer_styles.KEY, future)
    style = FigureStyle('project.simple', 1, 'Simple', {'title_size_pt': 8})
    with pytest.raises(ValueError):
        computer_styles.save(style, 'New')
    assert settings.value(computer_styles.KEY) == future
    settings.setValue(computer_styles.KEY, json.dumps({'version': 1, 'styles': {
        'future': {'version': 88, 'unknown': True}}, 'unknown_header': [4, None]}))
    computer_styles.save(style, 'New')
    value = json.loads(settings.value(computer_styles.KEY))
    assert value['styles']['future'] == {'version': 88, 'unknown': True}
    assert value['unknown_header'] == [4, None]
    # A legacy same-name fallback cannot authorize replacing opaque shared data.
    settings.setValue(PlotAppearanceWidget._PRESETS_KEY, json.dumps({'future': {'title_size': 9}}))
    with pytest.raises(ValueError):
        computer_styles.save(style, 'future', overwrite=True)
    with pytest.raises(ValueError):
        computer_styles.delete('future')
    assert json.loads(settings.value(computer_styles.KEY))['styles']['future'] == {'version': 88, 'unknown': True}


def test_save_on_computer_does_not_edit_project_registry_or_hydrate(rig, monkeypatch):
    owner = rig.owner
    monkeypatch.setattr(owner, 'activate', lambda _key: False)
    key = owner.create('iv')
    service = StyleService(SimpleNamespace(figure_workspace=lambda: owner))
    service._controller()  # perform the existing one-time legacy migration before the assertion
    before = owner.style_registry()
    record = owner.get(key)
    # Dormant create has no selected figure until activation; supply the current key.
    owner._current = key
    assert service.save_computer_look('Portable look')
    assert owner.style_registry() == before and owner.get(key) == record and not rig.built
    local = dict(service.computer_looks())['Portable look']
    assert local.title is None and local.x_label is None
    assert owner.apply_look_to_figure(key, local, keep_hand=False)
    pinned = owner.get(key).spec.presentation
    assert pinned.style_ref['style_id'].startswith('computer.')
    computer_styles.delete('Portable look')
    owner.install_workspace(owner.capture_workspace())
    assert owner.get(key).spec.presentation == pinned
    assert not rig.built


def test_project_style_menu_keeps_computer_section_separate(qtbot, rig, monkeypatch):
    from nanoscope.gui.analysis_explorer import AnalysisExplorer
    owner = rig.owner
    monkeypatch.setattr(owner, 'activate', lambda _key: False)
    key = owner.create('iv'); owner._current = key
    computer_styles.save(FigureStyle('project.example', 1, 'External', {'title_size_pt': 10}), 'Local only')
    explorer = AnalysisExplorer(); qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    menu = explorer._style_menu
    menu._sync()
    labels = [action.text() for action in menu.scope_menu('page').actions()]
    assert 'Built in' in labels and 'Project styles' in labels and 'On this computer' in labels
    assert labels.index('Local only') > labels.index('On this computer')
    assert not any(style.name == 'Local only' for style in owner.available_styles())
    assert menu._save_computer.isVisible() and menu._manage_computer.isVisible()
