"""Regression coverage for deliberate manual-measurement replacement and mask clearing."""
from __future__ import annotations

import os

import nanoscope  # noqa: F401 -- initialise Qt's packaged plugin path before PySide imports

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow


FS = 100_000.0
N = 10_000
_FIRST = (2_000, 3_000)
_SECOND = (4_000, 4_800)


def _document(tmp_path) -> TraceDocument:
    data = np.full(N, 100.0, dtype=np.float32)
    data[_FIRST[0]:_FIRST[1]] = 72.0
    data[_SECOND[0]:_SECOND[1]] = 48.0
    meta = RecordingMeta(
        path=tmp_path / "manual-edit.dat",
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(ChannelInfo(index=0, name="current", raw_units="pA",
                              kind=ChannelKind.CURRENT),),
    )
    return TraceDocument(meta=meta, channels=(data,))


@pytest.fixture
def window(qapp, qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win._analysis_controller.abort()


def _install(window: MainWindow, tmp_path) -> tuple[str, int]:
    doc = _document(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


def _place(window: MainWindow, doc_id: str, start: int, stop: int) -> None:
    x0 = window._trace_view.x_for_index(doc_id, 0, start)
    x1 = window._trace_view.x_for_index(doc_id, 0, stop)
    assert x0 is not None and x1 is not None
    window._trace_view.set_measure_region(x0, x1)


def test_update_selected_overwrites_one_saved_measurement_in_place(window, qtbot, tmp_path):
    """The revised span replaces the selected record, preserving its ID and annotation."""
    doc_id, _channel = _install(window, tmp_path)
    panel = window._analysis_panel.measure_panel
    panel.set_active(True)
    _place(window, doc_id, *_FIRST)
    assert panel.add_current() is True

    label_col = panel._model.column_index("label")
    assert panel._model.setData(panel._model.index(0, label_col), "training exemplar")
    panel._table_view.selectRow(0)
    original = panel.measurements_df.copy(deep=True)
    measurement_id = int(original.loc[0, "measurement_id"])

    _place(window, doc_id, *_SECOND)
    assert panel._update_button.isEnabled()
    with qtbot.waitSignal(panel.measurementsChanged, timeout=1_000):
        assert panel.update_selected() is True

    revised = panel.measurements_df
    assert len(revised) == 1
    assert int(revised.loc[0, "measurement_id"]) == measurement_id
    assert str(revised.loc[0, "label"]) == "training exemplar"
    assert (int(revised.loc[0, "start_idx"]), int(revised.loc[0, "end_idx"])) == _SECOND
    assert int(revised.loc[0, "n_samples"]) == _SECOND[1] - _SECOND[0]

    # The overwrite uses the normal history signal rather than a one-off writer.
    assert window._measure_history.undo() == "Edit measurement"
    restored = panel.measurements_df
    assert (int(restored.loc[0, "start_idx"]), int(restored.loc[0, "end_idx"])) == _FIRST
    assert str(restored.loc[0, "label"]) == "training exemplar"
    assert int(restored.loc[0, "measurement_id"]) == measurement_id
    assert not original.equals(revised)


def test_clear_mask_after_double_click_keeps_saved_measurements_unchanged(window, tmp_path):
    """A mistaken trace double-click clears only transient cursors, never a saved row."""
    doc_id, _channel = _install(window, tmp_path)
    panel = window._analysis_panel.measure_panel
    panel.set_active(True)
    _place(window, doc_id, *_FIRST)
    assert panel.add_current() is True
    saved = panel.measurements_df.copy(deep=True)

    x = window._trace_view.x_for_index(doc_id, 0, 4_400)
    assert x is not None
    panel._on_trace_measure_placement_requested(x)  # the same owner route as a trace double-click
    assert panel.is_active()
    assert window._trace_view.measure_bounds() is not None
    assert panel._clear_mask_btn.isEnabled()

    changed: list[bool] = []
    panel.measurementsChanged.connect(lambda: changed.append(True))
    assert panel.clear_mask() is True

    assert not panel.is_active()
    assert window._trace_view.measure_bounds() is None
    assert not panel._clear_mask_btn.isEnabled()
    assert changed == []
    assert_frame_equal(panel.measurements_df, saved)
    assert "Saved measurements and events were not changed" in panel._status.text()
