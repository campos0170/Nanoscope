"""Composition lettering consumes shared styles without revisiting source artwork."""
from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest
from PySide6 import QtWidgets

from nanoscope.core.figure_style import FigureStyle, style_binding
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_composition import compose_figures
from nanoscope.gui.figure_composition_revision import extract_composition
from .test_figure_composer import entries  # noqa: F401


def _style(size=9.125, name='Paper'):
    return FigureStyle('paper.panel', 2, name, {'panel_size_pt': size,
        'font_family': 'Times New Roman', 'text_roles': {'panel': {'bold': False,
        'color': '#123456', 'italic': True}}})


def _snapshots(entries):
    return [FigureSnapshot.from_storage(entry.payload['snapshot'], entry.blobs) for entry in entries]


def test_project_default_letters_use_fractional_points_and_preserve_frozen_sources(entries, qtbot):
    originals = deepcopy(entries)
    dialog = FigureComposerDialog(entries, default_style=_style())
    qtbot.addWidget(dialog)
    assert dialog._label_size.value() == 9.125
    assert dialog._label_size.suffix() == ' pt'
    dialog._table.item(0, 1).setText('Apo α')
    dialog._spacing.setValue(37)
    dialog.update_preview()
    assert dialog.snapshot is not None, dialog._message.text()
    saved = dialog.snapshot.config()
    assert saved['version'] == 3
    assert saved['style_binding'] == style_binding(_style())
    assert saved['label_overrides'] == {}
    assert saved['panel_labels'] == ['Apo α', 'B'] and saved['spacing'] == 37
    letters = [node for node in ET.fromstring(dialog.snapshot.svg_bytes).iter()
               if node.get('id', '').endswith('_label')]
    assert len(letters) == 2 and letters[0].text == 'Apo α'
    assert float(letters[0].get('font-size')) == pytest.approx(9.125 * 96 / 72)
    assert letters[0].get('font-style') == 'italic'
    panels, _layout = extract_composition(dialog.snapshot)
    for panel, source in zip(panels, _snapshots(entries)):
        assert panel.snapshot.to_storage() == source.to_storage()
    assert [(x.payload, x.blobs) for x in entries] == [(x.payload, x.blobs) for x in originals]


def test_apply_keeps_manual_letters_and_layout_then_explicitly_resets_only_typography(entries, qtbot):
    dialog = FigureComposerDialog(entries, default_style=_style())
    qtbot.addWidget(dialog)
    dialog._label_size.setValue(8.625)
    dialog._label_bold.setChecked(True)
    dialog._table.item(0, 1).setText('Custom letter')
    dialog._table.selectRow(0); dialog._move(1)
    dialog._spacing.setValue(31)
    assert dialog.apply_style(_style(11.125, 'Other'))
    assert dialog._label_size.value() == 8.625 and dialog._label_bold.isChecked()
    dialog.update_preview()
    before = dialog.snapshot.config()
    assert before['label_overrides'] == {'size_pt': 8.625, 'bold': True}
    assert before['panel_labels'] == ['B', 'Custom letter']
    dialog._keep_label_edits.setChecked(False)
    assert dialog.apply_style(_style(11.125, 'Other'))
    assert dialog._label_size.value() == 11.125 and not dialog._label_bold.isChecked()
    dialog.update_preview()
    after = dialog.snapshot.config()
    assert after['label_overrides'] == {}
    for key in ('panel_labels', 'panel_spans', 'spacing', 'columns'):
        assert after[key] == before[key]


def test_shared_composition_revision_restores_pinned_snapshot_and_manual_overrides(entries, qtbot):
    source = compose_figures(_snapshots(entries), ['X', 'Y', 'Z'], 2,
        style_binding=style_binding(_style()), label_overrides={'size_pt': 8.625},
        panel_spans=[[1, 1]] * 3, spacing=28)
    untouched = source.to_storage()
    _panels, layout = extract_composition(source)
    dialog = FigureComposerDialog(entries, initial_layout=layout, default_style=_style(20, 'Today'))
    qtbot.addWidget(dialog)
    assert dialog._label_size.value() == 8.625
    assert dialog._label_binding['snapshot']['name'] == 'Paper'
    dialog.update_preview()
    assert dialog.snapshot is not None, dialog._message.text()
    saved = dialog.snapshot.config()
    for key in ('style_binding', 'label_overrides', 'label_style', 'panel_labels', 'spacing', 'panel_spans'):
        assert saved[key] == source.config()[key]
    assert source.to_storage() == untouched


def test_legacy_revision_keeps_pixels_until_an_explicit_style_action(entries, qtbot):
    native = {'family': 'Helvetica', 'size_px': 30, 'bold': False, 'color': '#111111'}
    source = compose_figures(_snapshots(entries), ['A', 'B', 'C'], 2, label_style=native)
    _panels, layout = extract_composition(source)
    dialog = FigureComposerDialog(entries, initial_layout=layout, default_style=_style())
    qtbot.addWidget(dialog)
    assert dialog._label_size.value() == 22.5
    dialog.update_preview()
    assert dialog.snapshot.config()['version'] == 2
    assert dialog.snapshot.config()['label_style'] == native
    assert dialog.apply_style(_style())
    dialog.update_preview()
    assert dialog.snapshot.config()['version'] == 3
    assert dialog.snapshot.config()['label_style']['size_px'] == 30
    assert dialog.snapshot.config()['label_overrides']['size_pt'] == 22.5


def test_project_style_actions_refuse_a_changed_project_epoch(entries, qtbot):
    class Workspace:
        _epoch = 4
        def style_registry(self): return object()
        def default_style(self): return _style()
        def available_styles(self): return (_style(),)
    owner = QtWidgets.QWidget(); owner._figure_workspace = Workspace(); qtbot.addWidget(owner)
    dialog = FigureComposerDialog(entries, owner); qtbot.addWidget(dialog)
    dialog._fill_style_menu()
    action = next(action for action in dialog._style_menu.actions() if action.text() == 'Paper')
    before = deepcopy(dialog._label_binding)
    owner._figure_workspace._epoch += 1
    action.trigger()
    assert dialog._label_binding == before
    assert 'project changed' in dialog._message.text().lower()


def test_computer_style_is_available_for_composition_letters(entries, qtbot, monkeypatch):
    from nanoscope.gui import computer_styles
    local = _style(name='Computer lettering')
    monkeypatch.setattr(computer_styles, 'styles', lambda: (local,))
    dialog = FigureComposerDialog(entries); qtbot.addWidget(dialog)
    dialog._fill_style_menu()
    names = [action.text() for action in dialog._style_menu.actions()]
    assert names.index('Computer lettering') > names.index('On this computer')
    next(action for action in dialog._style_menu.actions() if action.text() == 'Computer lettering').trigger()
    assert dialog._label_binding == style_binding(local)
