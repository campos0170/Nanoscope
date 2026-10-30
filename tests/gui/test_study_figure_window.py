"""Native study styling and publication tools preserve committed science."""
from copy import deepcopy
import json
from types import SimpleNamespace
from xml.etree import ElementTree
import zipfile

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.plot_appearance import PlotAppearanceWidget
from nanoscope.gui.plot_export_options import resolve_export_options
from nanoscope.gui.study_figure_window import StudyFigureWindow
from .test_acid_study_window import complete, seeded


def test_close_before_initial_fit_does_not_call_a_deleted_canvas(qtbot):
    figure = Figure(figsize=(3, 2))
    FigureCanvasAgg(figure)
    figure.subplots().plot([0, 1], [2, 3])
    page = StudyFigureWindow(figure, pd.DataFrame(), {}, name='closing', provenance={})
    page.show()
    # Closing before the event loop delivers the initial fit used to leave a
    # queued Python callback targeting an already-deleted comparison canvas.
    page.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qtbot.wait(10)


@pytest.fixture(autouse=True)
def no_study_engines(monkeypatch):
    from nanoscope.analysis import manual_detection, manual_comparisons, study_signal
    def forbidden(*args, **kwargs):
        pytest.fail('Appearance/export must not invoke a study analysis engine.')
    for module, names in (
            (manual_detection, ('run_inference', 'find_candidates', 'measure_candidates', 'measure_imported_events')),
            (manual_comparisons, ('compare_recordings', 'extract_features', 'cluster_features', 'fit_mixture')),
            (study_signal, ('analyze', 'detect_bounds', 'null_controls'))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def page(qtbot):
    figure = Figure(figsize=(6, 3), dpi=100, layout='constrained')
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    left.plot([1., 2., 3.], [.12345678901234567, .5, .2], 'o-', color='#147d92', label='2 hours')
    left.set(xlabel='Dwell (ms)', ylabel='Current (pA)', title='Accepted events', xscale='log')
    left.legend()
    right.scatter([1., 2., 3.], [2., 1., 3.], color='#ca7627', label='30 hours')
    right.set(title='Other recording', xlabel='Time (s)', ylabel='Current (pA)')
    right.legend()
    frame = pd.DataFrame({'event_id': ['first', 'second', 'third'],
                          'dwell_ms': [1., 2., 3.],
                          'current_pA': [.12345678901234567, .5, .2],
                          'accepted': [True, False, True]}, index=[7, 7, 9])
    page = StudyFigureWindow(figure, frame, {'recording_methods.json': b'{"end":"inclusive"}'},
                             name='study_test', provenance={'study': 'Small Cu fixture', 'review_count': 2})
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(1)
    yield page
    # pytest-qt closes/deletes its registered widgets before fixture finalizers.
    figure.clear()


def test_shared_preset_panel_palette_and_reset_preserve_science(page, monkeypatch):
    assert isinstance(page._appearance_widget, PlotAppearanceWidget)
    data = page._data.copy(deep=True)
    ax = page._figure.axes[0]
    line = ax.lines[0]
    x, y = line.get_xdata().copy(), line.get_ydata().copy()
    limits = ax.get_xlim(), ax.get_ylim()
    original_style = page.style()
    page._appearance_widget._apply_preset(2)
    page._panel.setCurrentIndex(0)
    page._panel_edits['title'].textEdited.emit('Curated recording')
    page._panel_edits['x_label'].textEdited.emit('Duration (ms)')
    page._color.setCurrentIndex(page._color.findData('#147d92'))
    monkeypatch.setattr(QtWidgets.QColorDialog, 'getColor', lambda *a, **kw: QtGui.QColor('#7f00ff'))
    page._choose_color()
    assert page._style != page._applied
    assert ax.get_title() == 'Accepted events'  # explicit Apply boundary
    assert page.apply_appearance()
    assert ax.get_title() == 'Curated recording'
    assert ax.get_xlabel() == 'Duration (ms)'
    assert to_rgba(line.get_color()) == to_rgba('#7f00ff')
    assert to_rgba(ax.get_legend().legend_handles[0].get_color()) == to_rgba('#7f00ff')
    assert ax.get_xscale() == 'log'
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    pd.testing.assert_frame_equal(page._data, data, check_exact=True)
    page.reset_style()
    assert page.style() == original_style
    assert ax.get_title() == 'Accepted events'
    assert ax.get_xlabel() == 'Dwell (ms)'
    assert line.get_color() == '#147d92'
    assert page._item.native_figure_svg().startswith(b'<?xml')


@pytest.mark.parametrize('extension', ['.png', '.svg', '.pdf'])
def test_export_native_image_formats(page, tmp_path, extension):
    before = page._data.copy(deep=True)
    path = tmp_path / ('study' + extension)
    assert page.export_image(path=path) == path, page._status.text()
    assert path.stat().st_size > 100
    if extension == '.png':
        image = QtGui.QImage(str(path))
        assert not image.isNull() and image.width() == 600 and image.height() == 300
    elif extension == '.svg':
        document = ElementTree.fromstring(path.read_bytes())
        assert document.tag.endswith('svg')
        assert list(document.iter('{http://www.w3.org/2000/svg}path'))
    else:
        assert path.read_bytes().startswith(b'%PDF-')
    pd.testing.assert_frame_equal(page._data, before, check_exact=True)


def test_interactive_export_honors_publication_size(page, tmp_path, monkeypatch):
    import nanoscope.gui.study_figure_window as module
    path = tmp_path / 'physical.png'
    options = resolve_export_options((600, 300), extension='.png', native=False,
                                     width=50.8, height=25.4, dpi=300)
    monkeypatch.setattr(module, 'choose_live_plot_export', lambda *a, **kw: (path, options))
    assert page.export_image() == path, page._status.text()
    image = QtGui.QImage(str(path))
    assert (image.width(), image.height()) == (600, 300)
    assert abs(image.dotsPerMeterX() - 300 / .0254) < 2


def test_bundle_dialog_asks_through_the_export_seam(page, tmp_path, monkeypatch):
    """Save figure bundle… asks ``export_paths.ask_export_path`` for its path.

    It called ``QFileDialog.getSaveFileName`` itself, which no gui module may do outside
    the seam (``test_export_overwrite.py``). The bundle writes exactly the chosen file, so
    the seam's default output set is declared, and None from the seam is a cancel.
    """
    import nanoscope.gui.study_figure_window as module
    asked = []
    chosen = tmp_path / 'chosen.zip'
    monkeypatch.setattr(module, 'ask_export_path', lambda parent, **kw: asked.append((parent, kw)) or chosen)
    assert page.export_bundle() == chosen, page._status.text()
    assert asked == [(page, dict(title='Save figure bundle', suggestion='study_test.zip',
                                 filters='Figure bundle (*.zip)'))]
    assert zipfile.is_zipfile(chosen)
    monkeypatch.setattr(module, 'ask_export_path', lambda parent, **kw: None)
    status = page._status.text()
    assert page.export_bundle() is None
    assert page._status.text() == status


def test_snapshot_and_bundle_roundtrip_detach_numbers_and_resources(page, tmp_path):
    original = page._data.copy(deep=True)
    snapshot = page.snapshot()
    metadata, blobs = snapshot.to_storage()
    restored = FigureSnapshot.from_storage(metadata, blobs)
    pd.testing.assert_frame_equal(restored.dataframe(), original, check_exact=True)
    assert restored.provenance()['analysis_recomputed'] is False
    assert restored.provenance()['appearance_only'] is True
    assert dict(restored.additional_resources)['study_data/recording_methods.json'] == b'{"end":"inclusive"}'
    assert dict(restored.additional_resources)['rendering/figure.qpic']
    bundle = tmp_path / 'study.zip'
    assert page.export_bundle(path=bundle) == bundle, page._status.text()
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['format'] == 'nanoscope-figure-snapshot'
        assert any(name.endswith('.svg') for name in archive.namelist())
        assert archive.read('study_data/recording_methods.json') == b'{"end":"inclusive"}'
    page._data.loc[:, 'current_pA'] = 0
    pd.testing.assert_frame_equal(restored.dataframe(), original, check_exact=True)


def test_library_uses_real_frozen_artifact_and_study_save_callback(page):
    calls = []
    owner = SimpleNamespace(
        _append_figure=lambda entry: calls.append(('append', entry)),
        _mark_dirty=lambda: calls.append(('dirty', None)),
        _open_figure_library=lambda: calls.append(('open', None)),
        _figure_library=SimpleNamespace(refresh=lambda identity: calls.append(('refresh', identity))))
    page._library_owner = owner
    page._appearance_widget._apply_preset(1)
    entry = page.save_to_library(name='Publication Cu events')
    assert entry is not None, page._status.text()
    assert [name for name, _ in calls] == ['append', 'dirty', 'open', 'refresh']
    assert calls[-1][1] == entry.artifact_id
    assert entry.payload['name'] == 'Publication Cu events'
    assert entry.payload['source_state'] is None
    restored = FigureSnapshot.from_storage(entry.payload['snapshot'], entry.blobs)
    pd.testing.assert_frame_equal(restored.dataframe(), page._data, check_exact=True)
    assert not any(line.get_visible() for ax in page._figure.axes for line in ax.get_xgridlines())
    saved = []
    page._save_style = lambda style, snap, window: saved.append((deepcopy(style), snap, window))
    page._panel_edits['title'].textEdited.emit('Save this draft')
    page.save_to_study()
    assert len(saved) == 1
    assert saved[0][0]['panels']['0']['title'] == 'Save this draft'
    assert saved[0][2] is page
    assert saved[0][1].config()['style'] == saved[0][0]
    page.saved()
    assert 'styling saved' in page._status.text()


def test_frozen_canvas_survives_hidden_window_and_cleared_matplotlib(page, tmp_path):
    before = page.snapshot().dataframe()
    page.hide()
    page._figure.clear()
    path = tmp_path / 'closed_canvas.svg'
    assert page.export_image(path=path) == path, page._status.text()
    pd.testing.assert_frame_equal(page.snapshot().dataframe(), before, check_exact=True)
    assert path.stat().st_size > 1000


def test_invalid_style_does_not_replace_frozen_preview(page):
    original = page._item
    page._style['panels']['bad'] = {'title': 'Invalid'}
    assert not page.apply_appearance()
    assert page._item is original
    assert 'could not be applied' in page._status.text()


def test_close_hides_and_frees_editor_but_snapshot_still_exports(qtbot, tmp_path):
    """Close hides the editor and frees what it held; it never deletes the window.

    The editor set ``WA_DeleteOnClose``, the one attribute nanoscope/gui forbids
    (``test_fup_wiring_and_suite_green.py::test_no_gui_widget_turns_close_into_a_delete``):
    MainWindow's quit stops windows by closing them, and a save still running when the
    user closed the editor called ``saved()`` on a deleted window. Now the window outlives
    its close, empty: no figure, no rendering in its scene, no data. It says so through
    ``closed``, and a second close (the quit's) finds nothing left to free.
    """
    import shiboken6
    from nanoscope.gui.figure_bundle import write_frozen_figure_image
    figure = Figure(figsize=(3, 2), dpi=100)
    FigureCanvasAgg(figure)
    figure.subplots().plot([0., 1.], [1., 2.])
    data = pd.DataFrame({'x': [0., 1.], 'y': [1., 2.]})
    editor = StudyFigureWindow(figure, data, {'methods.json': b'{}'}, name='disposable',
                               provenance={'study': 'Cu'}, save_style=lambda *args: None)
    qtbot.addWidget(editor)
    closed = []
    editor.closed.connect(lambda: closed.append('closed'))
    editor.show()
    snapshot = editor.snapshot()
    editor.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert shiboken6.isValid(editor) and editor.isHidden()
    assert closed == ['closed']
    assert not editor._view.scene().items(), 'the closed editor still holds its rendering'
    assert editor._figure is None and not figure.axes
    assert editor._styler is None and editor._data is None
    assert not editor._resources and editor._save_style is None
    # A second close must find nothing left to free (MainWindow's quit closes every
    # window it owns). Qt 6 sends no event for close() on a closed window, so send one.
    QtWidgets.QApplication.sendEvent(editor, QtGui.QCloseEvent())
    assert shiboken6.isValid(editor) and closed == ['closed', 'closed']
    pd.testing.assert_frame_equal(snapshot.dataframe(), data, check_exact=True)
    path = tmp_path / 'after-close.svg'
    options = resolve_export_options((snapshot.width, snapshot.height), extension='.svg')
    write_frozen_figure_image(path, snapshot, options)
    assert path.stat().st_size > 1000


@pytest.fixture
def acid_window(qtbot):
    from nanoscope.gui.acid_study_window import AcidStudyWindow
    from nanoscope.gui.worker_thread import drain_detached
    window = AcidStudyWindow(); qtbot.addWidget(window)
    yield window
    window.abort(); drain_detached(5000); window.close()


def _acid_study_with_a_figure(window, tmp_path, qtbot):
    from nanoscope.analysis import manual_workspace as checkpoint
    from nanoscope.plot.acid_figure_recipe import capture_figure
    seeded(window, tmp_path)
    complete(qtbot, window, window.confirm_selection)
    folder = window.folder / 'runs' / 'initial'; folder.mkdir(parents=True)
    path = folder / 'test_acid_plot.png'
    figure = Figure(figsize=(5, 3)); FigureCanvasAgg(figure)
    figure.subplots().plot([1, 2, 3], [4, 8, 6], color='#147d92', label='Accepted')
    figure.savefig(path); capture_figure(figure, path)
    window.state['records'][0]['prepared'] = {'figures': [path]}
    window.state['figures'] = []
    window._revision = checkpoint.save_workspace(window.state, window.folder, expected_revision=window._revision,
                                                 owned_outputs=checkpoint.stage_output_files(folder))
    window.refresh()
    return path


def test_acid_editor_closed_mid_save_reports_the_save_and_is_freed_by_the_next_open(acid_window, tmp_path, qtbot):
    """Closing the Acid figure editor while its save runs no longer fails the save.

    Measured before the fix: close the editor after Save styling to study and before the
    save lands, and the study committed the styling, then called ``saved()`` on the
    deleted editor, showed "Completed result could not be displayed: libshiboken: Internal
    C++ object (PySide6.QtWidgets.QLabel) already deleted." and emitted
    ``operationFailed``. Close now only hides the editor (its ``closed`` signal drops it
    from ``_style_windows``), the save reports success, and the study window disposes of
    the closed editor when the next one opens, when no save can still call back into it.
    """
    import shiboken6
    window = acid_window
    path = _acid_study_with_a_figure(window, tmp_path, qtbot)
    complete(qtbot, window, window.style_selected_figure)
    editor = window._style_windows[-1]
    editor._panel_edits['title'].textEdited.emit('Closed mid-save')
    failed, finished = [], []
    window.operationFailed.connect(failed.append); window.operationFinished.connect(finished.append)
    editor.save_to_study()
    assert window.is_busy(), window._status.text()
    editor.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert shiboken6.isValid(editor) and editor.isHidden()
    assert editor not in window._style_windows
    qtbot.waitUntil(lambda: not window.is_busy(), timeout=20000)
    assert not failed, failed
    assert finished == ['Save acid figure styling']
    assert window._status.text() == 'Save acid figure styling complete.'
    assert window.state['figures'][0] != path
    complete(qtbot, window, window.style_selected_figure)
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert not shiboken6.isValid(editor), 'the closed editor was kept after the next one opened'
    assert len(window._style_windows) == 1 and window._style_windows[0].isVisible()


def test_manual_study_disposes_a_closed_editor_when_the_next_one_opens(qtbot, tmp_path, monkeypatch):
    """Close only hides an editor, so the Manual study window frees the closed ones itself.

    Otherwise each Style selected figure… would leave one hidden editor, with its window
    backing store, parented to the study until quit. An editor still open is never touched.
    """
    import shiboken6
    from nanoscope.gui import study_figure_recipe
    from nanoscope.gui.manual_study_window import ManualStudyWindow
    from nanoscope.gui.worker_thread import drain_detached
    window = ManualStudyWindow(); qtbot.addWidget(window)
    failures = []; window.operationFailed.connect(failures.append)
    try:
        window.new_workspace(tmp_path / 'workspace')
        image = QtGui.QImage(40, 30, QtGui.QImage.Format_RGB32); image.fill(0xffffff)
        path = window.folder / 'saved-figure.png'; assert image.save(str(path))
        window.state['figures'] = [path]; window.refresh()
        def reconstruct(state, figure_path, cancel=None):
            figure = Figure(figsize=(3, 2)); FigureCanvasAgg(figure)
            figure.subplots().plot([0., 1.], [1., 2.])
            return figure, pd.DataFrame({'x': [0., 1.]}), {}, {'kind': 'fixture'}
        monkeypatch.setattr(study_figure_recipe, 'reconstruct', reconstruct)
        def open_editor():
            assert window.style_selected_figure() is not False
            qtbot.waitUntil(lambda: not window.is_busy(), timeout=20000)
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            return window._figure_editor
        first = open_editor(); first.close()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        assert shiboken6.isValid(first) and first.isHidden()
        second = open_editor()
        assert not shiboken6.isValid(first), 'the closed editor was kept after the next one opened'
        third = open_editor()
        assert shiboken6.isValid(second) and shiboken6.isValid(third), 'an open editor was disposed'
        assert not failures, failures
    finally:
        window.abort(); drain_detached(3000)


def test_legacy_gallery_title_requires_explicit_edit_and_survives_event_reorder(qtbot):
    from nanoscope.plot.study_figure_style import bind_event_panel
    def event_figure(events):
        figure = Figure(figsize=(6, 3), dpi=100)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, len(events), squeeze=False).flat
        for ax, event in zip(axes, events):
            ax.plot([0., 1.], [0., -float(event)])
            ax.set_title('Event ' + str(event))
            bind_event_panel(ax, tag='study', event_id=event, start_idx=event*10, end_idx=event*10+1)
        return figure
    first = StudyFigureWindow(event_figure([1, 2]), pd.DataFrame(), {}, name='gallery', provenance={},
                              style={'panels': {'0': {'title': 'Unverified old event label'}}})
    qtbot.addWidget(first)
    assert first._figure.axes[0].get_title() == 'Event 1'
    assert 'without event identity' in first._title_notice.text()
    assert first._panel_edits['title'].text() == ''
    first._panel_edits['title'].textEdited.emit('Verified event one')
    first._panel.setCurrentIndex(1)
    first._panel_edits['title'].textEdited.emit('Verified event two')
    assert first.apply_appearance()
    saved = first.snapshot().config()['style']  # serialization is part of binding
    second = StudyFigureWindow(event_figure([2]), pd.DataFrame(), {}, name='gallery', provenance={}, style=saved)
    qtbot.addWidget(second)
    assert second._figure.axes[0].get_title() == 'Verified event two'
    assert second._panel_edits['title'].text() == 'Verified event two'
    assert 'changed or are absent' in second._title_notice.text()
    second.reset_style()
    assert second._figure.axes[0].get_title() == 'Event 2'
    assert not second._title_notice.text()


def test_bundle_excludes_obsolete_style_resource_and_keeps_current_style(page, tmp_path):
    # Recreate the editor with resources from a previous saved edit. Current
    # presentation belongs to config.json, never an old study_data sidecar.
    figure = Figure(figsize=(3, 2)); FigureCanvasAgg(figure)
    figure.subplots().plot([0., 1.], [1., 2.])
    editor = StudyFigureWindow(figure, pd.DataFrame({'value': [1.23456789]}),
        {'figure.style.json': b'{"panels":{"0":{"title":"old"}}}', 'methods.json': b'{"science":"unchanged"}'},
        name='figure', provenance={})
    try:
        editor._panel_edits['title'].textEdited.emit('current')
        path = tmp_path / 'current.zip'
        assert editor.export_bundle(path=path) == path, editor._status.text()
        with zipfile.ZipFile(path) as archive:
            assert not any(name.endswith('.style.json') for name in archive.namelist())
            assert json.loads(archive.read('config.json'))['style']['panels']['0']['title'] == 'current'
            assert archive.read('study_data/methods.json') == b'{"science":"unchanged"}'
    finally:
        editor.close()
