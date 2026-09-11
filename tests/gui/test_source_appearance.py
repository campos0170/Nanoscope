import numpy as np
import pandas as pd
import pytest
from PySide6 import QtGui, QtWidgets

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.source_appearance import SourceAppearanceEditor


def source():
    return PlotSource("file#0", "A long acquisition file", pd.DataFrame({
        "event_id": np.arange(20), "dwell_s": np.linspace(.001, .02, 20),
        "depth_pA": np.linspace(10., 25., 20), "accepted": True}), colour="#123456")


def test_editor_changes_are_explicit_and_scoped(qtbot):
    session = Session()
    session.colours.set_ink("file#0", "#123456")
    editor = SourceAppearanceEditor(session=session)
    qtbot.addWidget(editor)
    editor.set_sources([source()])
    changes = []
    editor.changed.connect(changes.append)
    editor.label.setText("Apo -700 mV")
    assert not changes and not session.source_labels.labels
    editor.apply.click()
    assert changes == [{"file#0": {"label": "Apo -700 mV"}}]
    assert not session.source_labels.labels
    editor.project.click()
    assert session.source_labels.label("file#0") == "Apo -700 mV"
    assert session.colours.inks["file#0"] == "#123456"
    editor.label.clear()
    editor.project.click()
    assert not session.source_labels.labels


def test_editor_set_sources_and_state_never_emit(qtbot):
    editor = SourceAppearanceEditor()
    qtbot.addWidget(editor)
    changes = []
    editor.changed.connect(changes.append)
    editor.set_sources([source()])
    editor.set_state({"file#0": {"label": "Apo", "color": "#f12345"}})
    editor.set_sources([{"key": "file#0", "label": "Pinned original", "color": "#123456"}])
    assert not changes and not editor.project.isEnabled()
    assert editor.label.text() == "Apo"


def test_population_project_defaults_do_not_redraw_or_change_numerical_state(qtbot, monkeypatch):
    from nanoscope.gui.plots_panel import PlotsWindow
    session = Session()
    page = PlotsWindow(session)
    qtbot.addWidget(page)
    page.set_data([source()])
    page._source_list.setCurrentRow(0)
    before = page.figure_applied_config()
    original = page._sources["file#0"].df.copy(deep=True)
    items = list(page._plot.items)
    monkeypatch.setattr(page, "_redraw", lambda *args, **kwargs: pytest.fail("Appearance ran analysis"))
    monkeypatch.setattr(page, "_run_plot", lambda *args, **kwargs: pytest.fail("Appearance ran Plot"))
    page._source_list.currentItem().setText("Apo -700 mV")
    page._save_project_source_label()
    assert session.source_labels.label("file#0") == "Apo -700 mV"
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", lambda *args, **kwargs: QtGui.QColor("#654321"))
    page._pick_project_source_color()
    assert session.colours.inks["file#0"] == "#654321"
    assert page._sources["file#0"].colour == "#654321"
    assert list(page._plot.items) == items
    assert not page._stale
    pd.testing.assert_frame_equal(page._sources["file#0"].df, original, check_exact=True)
    after = page.figure_applied_config()
    before.pop("style", None)
    after.pop("style", None)
    assert before == after
    monkeypatch.undo()  # close/reset deliberately redraws its empty prompt
