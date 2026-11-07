"""Shared project source edits preserve exact IDs and never alter frozen figures."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.project_sources import ProjectSourcesDialog
from nanoscope.gui.source_appearance import SourceAppearanceEditor

from . import p1_hosting as H


def sources():
    frame = pd.DataFrame({'dwell_s': [.001, .002], 'depth_pA': [4., 5.]})
    base = PlotSource('recording#0', 'Same acquisition', frame, colour='#123456',
                      channel=0, module_id='detect.schmitt')
    return (base, replace(base, key='recording#0@detect.cusum', module_id='detect.cusum'),
            replace(base, key='segment:abc#0', is_segment=True),
            replace(base, key='recording#1', channel=1))


def test_project_table_edits_one_exact_identity_and_retains_data_and_figure_overrides(qtbot):
    session = Session(); values = sources()
    saved = {'source_appearance': {values[1].key: {'label': 'Frozen name', 'color': '#abcdef'}}}
    snapshot = deepcopy(saved)
    local = SourceAppearanceEditor(session=session); qtbot.addWidget(local)
    local.set_sources(values); local.set_state(saved['source_appearance'])
    local_changes = []; local.changed.connect(local_changes.append)
    calls = []
    dialog = ProjectSourcesDialog(session, lambda: calls.append('catalog') or values)
    qtbot.addWidget(dialog)
    assert len(calls) == 1 and dialog.table.topLevelItemCount() == 4
    assert dialog.editor.apply.isHidden()
    before_frames = [value.df.copy(deep=True) for value in values]
    exact = values[1].key
    dialog.table.setCurrentItem(dialog.table.topLevelItem(1))
    assert dialog.editor.source.currentData() == exact
    dialog.editor.label.setText('CUSUM display alias')
    dialog.editor._color_override = '#654321'
    dialog.editor.project.click()
    assert session.source_labels.labels == {exact: 'CUSUM display alias'}
    assert session.colours.inks[exact] == '#654321'
    assert dialog.table.currentItem().text(0) == 'CUSUM display alias'
    assert dialog.table.currentItem().data(0, QtCore.Qt.ItemDataRole.UserRole) == exact
    assert not local_changes and local.state() == saved['source_appearance']
    assert saved == snapshot
    for value, before in zip(values, before_frames):
        pd.testing.assert_frame_equal(value.df, before, check_exact=True)
        assert value.label == 'Same acquisition' and value.colour == '#123456'


def test_project_defaults_clear_independently_and_unavailable_keys_remain_exact(qtbot):
    session = Session(); values = sources()
    session.set_source_appearance(values[0].key, label='Override', color='#654321')
    session.set_source_appearance('unavailable#7@detect.cusum', label='Old source', color='#aabbcc')
    dialog = ProjectSourcesDialog(session, lambda: values); qtbot.addWidget(dialog)
    assert dialog.table.topLevelItemCount() == 5
    assert dialog._rows['unavailable#7@detect.cusum']['available'] is False
    dialog.editor.source.setCurrentIndex(dialog.editor.source.findData(values[0].key))
    dialog.editor.label.clear()
    dialog.editor._reset_color()
    dialog.editor.project.click()
    assert values[0].key not in session.source_labels.labels
    assert values[0].key not in session.colours.inks
    assert session.source_labels.label('unavailable#7@detect.cusum') == 'Old source'
    assert session.colours.inks['unavailable#7@detect.cusum'] == '#aabbcc'
    assert dialog.table.currentItem().text(0) == 'Same acquisition'
    # Only scalar metadata is retained in the dialog, not its frames or plot sources.
    assert all('df' not in row for row in dialog._rows.values())


def test_no_sources_is_read_only_and_refresh_is_explicit(qtbot):
    session = Session(); calls = []
    dialog = ProjectSourcesDialog(session, lambda: calls.append(1) or ())
    qtbot.addWidget(dialog)
    assert calls == [1] and dialog.table.topLevelItemCount() == 0
    assert not dialog.editor.project.isEnabled()
    dialog.refresh()
    assert calls == [1, 1] and not session.source_labels.labels and not session.colours.inks


@pytest.mark.parametrize('kind', ('distributions', 'distribution_summaries', 'comparison_figures',
                                  'event_waveforms', 'noise', 'iv', 'voltage_series', 'subpopulations'))
def test_each_native_format_series_opens_the_shared_project_table_without_update(qtbot, monkeypatch, kind):
    session = Session(); calls = []
    explorer = AnalysisExplorer(source_session=session, source_provider=lambda: calls.append('catalog') or sources())
    qtbot.addWidget(explorer)
    page = H.page_builders()[kind]()
    explorer.add_page(kind, page)
    explorer.show_page(kind, page)
    adapter = explorer.adapter(kind)
    monkeypatch.setattr(adapter, 'update', lambda: pytest.fail('Source appearance ran an analysis'))
    shown = []
    monkeypatch.setattr(ProjectSourcesDialog, 'exec', lambda self: shown.append(self.table.topLevelItemCount()))
    explorer.inspector().set_tab(1)
    from nanoscope.gui.format_inspector import ElementRef
    pane = explorer.inspector().pane(kind).format_pane
    pane.show_element(ElementRef('', 'series.data'))
    button = pane.findChild(QtWidgets.QPushButton, 'formatProjectSources')
    assert button is not None and button.isVisible()
    button.click()
    assert calls == ['catalog'] and shown == [4]
    assert not session.source_labels.labels and not session.colours.inks
