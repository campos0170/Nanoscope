"""Heatmap count scale and both marginals fit a narrow native Explorer canvas."""
import numpy as np
import pytest
from PySide6 import QtCore
from .test_event_plot import _fed_window


@pytest.mark.parametrize('width', [297, 640])
def test_heatmap_scale_ticks_and_marginals_fit_without_changing_data(qtbot, tmp_path, monkeypatch, width):
    page = _fed_window(qtbot, tmp_path, n_events=24)
    page._on_adopt()
    page.set_appearance_state({'background': 'light', 'title_size': 7, 'axis_size': 7, 'tick_size': 7})
    counts = page.density_counts().copy()
    x_edges, y_edges = page._density_x_edges.copy(), page._density_y_edges.copy()
    image = page._density_image.image
    levels = page._colorbar.levels()
    from nanoscope.analysis import event_shape
    monkeypatch.setattr(event_shape, 'density_2d', lambda *args, **kw: pytest.fail('resize recomputed data'))
    canvas = page._density_glw
    previous_parent = canvas.parentWidget()
    canvas.setParent(None)
    try:
        canvas.resize(width, 652); canvas.show(); qtbot.wait(80)
        canvas.grab()  # Force actual text/tick layout, not just widget size hints.
        viewport = canvas.viewport().rect()
        for item in (page._colorbar, page._colorbar.axis, page._density_right,
                     page._density_main, page._density_top):
            drawn = canvas.mapFromScene(item.sceneBoundingRect()).boundingRect()
            assert drawn.left() >= viewport.left() - 1
            assert drawn.right() <= viewport.right() + 1, (type(item).__name__, drawn, viewport)
        assert page._colorbar.axis.style['showValues']
        assert page.label_text('colorbar.label') == 'Event count'
        assert page._colorbar.horizontal
        assert page._colorbar.axis.label.isVisible()
        assert page._density_main.getAxis('left').label.isVisible()
        assert page._density_right.getAxis('bottom').label.isVisible()
        np.testing.assert_array_equal(page.density_counts(), counts)
        np.testing.assert_array_equal(page._density_x_edges, x_edges)
        np.testing.assert_array_equal(page._density_y_edges, y_edges)
        assert page._density_image.image is image
        assert page._colorbar.levels() == levels
    finally:
        canvas.hide()
        canvas.setParent(previous_parent)
