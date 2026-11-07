"""Report editing owns an existing Figure and exports it without a generator callback."""
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from nanoscope.gui.report_figure_window import open_report_figure


def test_report_retains_generated_artists_and_clears_once_after_explicit_export(qtbot, monkeypatch, tmp_path):
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    line, = axis.plot([1., 2.], [.123456789, .987654321])
    axis.set_title('Generated report')
    provenance = {'source': {'recording': 'A'}, 'fit': {'stored': True}}
    data = pd.DataFrame({'x': [1., 2.], 'y': [.123456789, .987654321]})
    expected = data.copy(deep=True)
    editor = open_report_figure(owner, figure, data, name='amplitude', provenance=provenance)
    assert editor._figure is figure and editor in owner._report_figure_editors
    assert not any(action.text() == 'Save styling to study' for toolbar in editor.findChildren(QtWidgets.QToolBar) for action in toolbar.actions())
    data.iloc[0, 0] = 999
    provenance['source']['recording'] = 'wrong owner'
    controller = next(c for c in editor._text_controllers if c.panel != 'figure')
    controller.set_style_property('title', 'text', 'Report Ω $literal$')
    assert axis.get_title() == 'Report Ω $literal$'
    assert figure.axes[0].lines[0] is line
    np.testing.assert_array_equal(line.get_ydata(), expected.y.to_numpy())
    pd.testing.assert_frame_equal(editor._data, expected, check_exact=True)
    assert editor._provenance['source']['recording'] == 'A'
    output = tmp_path / 'report.svg'
    assert editor.export_image(path=output) == output and output.is_file()
    calls = []
    clear = figure.clear
    def clear_once(*args, **kwargs):
        calls.append(True)
        return clear(*args, **kwargs)
    monkeypatch.setattr(figure, 'clear', clear_once)
    editor.close(); editor.close()
    assert calls == [True]
    assert editor not in owner._report_figure_editors
    assert editor._figure is None


def test_opening_another_report_does_not_retire_a_merely_hidden_editor(qtbot):
    import shiboken6
    from PySide6 import QtCore
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    def fresh(name):
        figure = Figure(figsize=(3, 2)); FigureCanvasAgg(figure)
        figure.subplots().plot([1, 2], [3, 4])
        return open_report_figure(owner, figure, pd.DataFrame(), name=name, provenance={})
    first = fresh('first')
    controller = next(c for c in first._text_controllers if c.panel != 'figure')
    controller.set_style_property('title', 'text', 'Unsaved words')
    first.hide()
    second = fresh('second')
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert shiboken6.isValid(first) and first._figure is not None
    assert controller.target('title').style.text == 'Unsaved words'
    assert first in owner._report_figure_editors
    first.close()
    third = fresh('third')
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert not shiboken6.isValid(first)
    second.close(); third.close()


def test_report_handoff_releases_pyplot_manager_but_preserves_retained_artists(qtbot, monkeypatch):
    from matplotlib import pyplot as plt
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    existing = set(plt.get_fignums())
    figure, axis = plt.subplots(figsize=(3, 2))
    number = figure.number
    line, = axis.plot([1., 2.], [3., 4.])
    assert number in plt.get_fignums()
    calls = []
    clear = figure.clear
    def counted(*args, **kwargs):
        calls.append(True)
        return clear(*args, **kwargs)
    monkeypatch.setattr(figure, 'clear', counted)
    editor = open_report_figure(owner, figure, pd.DataFrame({'x': [1., 2.]}),
                                name='managed', provenance={})
    assert set(plt.get_fignums()) == existing
    assert not calls and editor._figure is figure and axis.lines[0] is line
    np.testing.assert_array_equal(line.get_ydata(), [3., 4.])
    editor.close(); editor.close()
    assert calls == [True] and set(plt.get_fignums()) == existing
