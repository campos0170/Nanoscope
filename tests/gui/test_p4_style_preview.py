"""A bounded synthetic preview uses the real style resolver without touching figures."""
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.core.figure_style import FigureStyle, resolve_bound_text, resolve_series_color
from nanoscope.gui.project_style_editor import ProjectStyleEditor


def test_preview_uses_same_palette_point_font_and_frame_projection(qtbot):
    style = FigureStyle('project.preview', 3, 'Preview', look={'palette': ['#112233', '#8899aa'],
        'line_width_pt': .75, 'marker_size_pt': 3., 'panel_size_pt': 8.5},
        columns={'single_mm': 89., 'double_mm': 183.})
    original = deepcopy(style.to_dict())
    editor = ProjectStyleEditor(style); qtbot.addWidget(editor)
    before = [(item.xData.copy(), item.yData.copy()) for item in editor._preview_curves]
    editor.palette.setText('#aa2255, #2277aa')
    editor.sizes['line_width_pt'].setValue(2.25)
    editor.sizes['marker_size_pt'].setValue(6.75)
    editor.sizes['panel_size_pt'].setValue(9.25)
    editor.frame.setCurrentIndex(editor.frame.findData(True))
    editor.canvas_sizes['canvas_width_mm'].setValue(100.)
    editor.canvas_sizes['canvas_height_mm'].setValue(65.)
    editor._update_preview()
    state = editor._preview_state
    for index, item in enumerate(editor._preview_curves):
        expected = resolve_series_color(state.style_binding, index=index)
        assert pg.mkPen(item.opts['pen']).color().name() == expected
        assert pg.mkBrush(item.opts['symbolBrush']).color().name() == expected
        assert pg.mkPen(item.opts['pen']).widthF() == pytest.approx(3.)
        assert item.opts['symbolSize'] == pytest.approx(9.)
        np.testing.assert_array_equal(item.xData, before[index][0])
        np.testing.assert_array_equal(item.yData, before[index][1])
    values, _ = resolve_bound_text(state.style_binding, 'annotation.panel.preview')
    panel_font = editor._preview_labels['annotation.panel.preview'].textItem.font()
    assert panel_font.pointSizeF() == values['size_pt'] == 9.25
    assert panel_font.bold() is True
    assert editor.preview.getPlotItem().getViewBox().border is not None
    assert '100 × 65 mm' in editor.preview_note.text()
    assert '89 mm' in editor.preview_note.text() and '183 mm' in editor.preview_note.text()
    assert editor.preview.maximumHeight() == editor.preview.minimumHeight() == 210
    assert style.to_dict() == original


def test_invalid_draft_keeps_last_valid_preview_and_cancel_preserves_style(qtbot):
    style = FigureStyle('project.preview', 1, 'Preview', look={'palette': ['#123456']})
    editor = ProjectStyleEditor(style); qtbot.addWidget(editor)
    before = editor._preview_state
    pens = [pg.mkPen(item.opts['pen']) for item in editor._preview_curves]
    editor.palette.setText('this is not a colour')
    editor._update_preview()
    assert editor._preview_state == before
    assert 'last valid style' in editor.error.text()
    assert [pg.mkPen(item.opts['pen']) for item in editor._preview_curves] == pens
    editor.reject()
    assert style.look['palette'] == ('#123456',)
