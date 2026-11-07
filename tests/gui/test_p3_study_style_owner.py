"""Retained Study history includes ref/snapshot, panel words and local colour state."""
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nanoscope.core.figure_style import FigureStyle
from nanoscope.gui.study_figure_window import StudyFigureWindow
from nanoscope.plot.study_figure_style import bind_event_panel


def figure(events=(1, 2)):
    value = Figure(figsize=(6, 3), dpi=100)
    FigureCanvasAgg(value)
    for axis, event in zip(value.subplots(1, len(events), squeeze=False).flat, events):
        axis.plot([0., 1.], [0., -float(event)], color='#147d92', label='Original')
        axis.set_title('Event ' + str(event))
        bind_event_panel(axis, tag='study', event_id=event, start_idx=event * 10, end_idx=event * 10 + 1)
    return value


def test_shared_style_and_per_event_words_are_one_owner_with_undo_and_reorder(qtbot):
    editor = StudyFigureWindow(figure(), pd.DataFrame({'x': [1.]}), {}, name='gallery',
                               provenance={}, style={'future_extension': {'keep': [1, 2]}})
    qtbot.addWidget(editor)
    editor.show()
    before = [axis.lines[0].get_xydata().copy() for axis in editor._figure.axes]
    panel = next(c for c in editor._text_controllers if c.panel.startswith('event:'))
    panel.set_style_property('title', 'text', 'Verified β $event$')
    assert editor.history_mark() == 1
    saved = editor.style()
    editor._view.zoom(1.25)
    view_before = (editor._view.transform(), editor._view.horizontalScrollBar().value(),
                   editor._view.verticalScrollBar().value())
    paper = FigureStyle(style_id='project.paper', revision=2, name='Paper',
                        look={'title_size_pt': 7.25, 'line_width_pt': .75})
    assert editor.apply_style(paper)
    assert editor.history_mark() == 2
    assert (editor._view.transform(), editor._view.horizontalScrollBar().value(),
            editor._view.verticalScrollBar().value()) == view_before
    assert editor.style()['style_binding']['ref'] == paper.ref.to_dict()
    assert panel.target('title').style.text == 'Verified β $event$'
    assert editor.undo_style() and editor.style() == saved
    assert editor.redo_style()
    assert editor.style()['future_extension'] == {'keep': [1, 2]}
    for axis, values in zip(editor._figure.axes, before):
        np.testing.assert_array_equal(axis.lines[0].get_xydata(), values)
    # Controller keys follow the actual event, not its column ordinal.
    reopened = StudyFigureWindow(figure((2, 1)), pd.DataFrame(), {}, name='reordered',
                                  provenance={}, style=editor.style())
    qtbot.addWidget(reopened)
    original = next(c for c in reopened._text_controllers if c.panel == panel.panel)
    assert original.target('title').style.text == 'Verified β $event$'
    assert reopened._figure.axes[1].get_title() == 'Verified β $event$'


def test_save_stays_explicit_and_roundtrips_whole_owner_envelope(qtbot):
    writes = []
    def save(style, snapshot, editor):
        writes.append((deepcopy(style), snapshot.config(), editor))
    editor = StudyFigureWindow(figure((1,)), pd.DataFrame({'exact': [.123456789]}), {},
                               name='one', provenance={'workspace': '/study/A', 'revision': 4},
                               save_style=save)
    qtbot.addWidget(editor)
    panel = next(c for c in editor._text_controllers if c.panel.startswith('event:'))
    panel.set_style_property('title', 'italic', True)
    assert writes == []
    editor.save_to_study()
    assert len(writes) == 1
    assert writes[0][0] == writes[0][1]['style'] == editor.style()
    assert writes[0][2] is editor
    editor.reset_style()
    assert editor.undo_style()
    assert editor.style() == writes[0][0]


def test_acid_save_refuses_different_folder_even_if_revision_matches(qtbot, tmp_path, monkeypatch):
    from nanoscope.gui.acid_study_window import AcidStudyWindow
    from .test_study_figure_window import _acid_study_with_a_figure
    from .test_acid_study_window import complete
    from nanoscope.gui.worker_thread import drain_detached
    window = AcidStudyWindow()
    qtbot.addWidget(window)
    try:
        _acid_study_with_a_figure(window, tmp_path, qtbot)
        complete(qtbot, window, window.style_selected_figure)
        editor = window._style_windows[-1]
        revision = window._revision
        window.folder = tmp_path / 'different-study'
        assert window._revision == revision
        monkeypatch.setattr(window, '_stage', lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('A stale editor reached the save stage')))
        editor.save_to_study()
        assert 'study changed' in editor._status.text().lower()
        assert not window.folder.exists()
    finally:
        window.abort(); drain_detached(5000); window.close()


def test_rejected_history_restore_preserves_both_visible_style_and_history(qtbot, monkeypatch):
    import pytest
    from nanoscope.core.figure_history_state import HistoryCapsule
    editor = StudyFigureWindow(figure((1,)), pd.DataFrame(), {}, name='retained', provenance={})
    qtbot.addWidget(editor)
    panel = next(c for c in editor._text_controllers if c.panel.startswith('event:'))
    panel.set_style_property('title', 'text', 'Retained words')
    previous = editor.style()
    history = editor._appearance_widget.capture_history_capsule()
    native = editor._appearance_widget.state().to_dict()
    target = deepcopy(previous)
    target['future_extension'] = {'reject': True}
    incoming = HistoryCapsule('study-figure-style-v1', target)
    apply = editor.apply_appearance
    rejected = []
    def reject_target():
        if editor._style.get('future_extension', {}).get('reject'):
            rejected.append(True)
            return False
        return apply()
    monkeypatch.setattr(editor, 'apply_appearance', reject_target)
    with pytest.raises(ValueError, match='Could not restore'):
        editor._appearance_widget.restore_history_capsule(incoming)
    assert rejected == [True]
    assert editor.style() == previous
    assert editor._style_history == history
    assert editor._appearance_widget.state().to_dict() == native
    assert editor._figure.axes[0].get_title() == 'Retained words'
