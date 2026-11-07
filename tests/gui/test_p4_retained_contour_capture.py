"""Real density outlines survive retained preview, recolour and immutable reopen."""
from dataclasses import replace
from pathlib import Path
import runpy

import nanoscope  # noqa: F401: Qt bootstrap
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtGui

from nanoscope.gui import figure_bundle as B
from nanoscope.gui.plot_export_options import resolve_export_options
from nanoscope.gui.study_figure_window import StudyFigureWindow

_contour_helpers = runpy.run_path(str(Path(__file__).parents[1] / 'plot' / 'test_p4_retained_contours.py'))
density_figure = _contour_helpers['density_figure']
assert_unfilled_svg = _contour_helpers['assert_unfilled_svg']


def _rgba(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()


def test_report_contours_preview_style_snapshot_and_reopen_never_fill_or_reestimate(qtbot, monkeypatch, tmp_path):
    import scipy.stats
    figure, axis = density_figure()
    source = figure._nanoscope_source_data.copy(deep=True)
    monkeypatch.setattr(scipy.stats, 'gaussian_kde', lambda *a, **k: pytest.fail('Retained path reran KDE'))
    editor = StudyFigureWindow(figure, source, {}, name='Apo density', provenance={}, owner_kind='report')
    qtbot.addWidget(editor)
    target = editor._styler.series_descriptors()[0]
    widget = editor._appearance_widget
    widget.apply_appearance(replace(widget.state(), series_styles={
        target.series_id: {'color': '#aa2277', 'line_width_pt': 1.5}}))
    editor._commit_style_run()
    snapshot = editor.snapshot()
    assert_unfilled_svg(snapshot.svg_bytes)
    figure.canvas.draw()
    reference = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    actual = _rgba(QtGui.QImage.fromData(snapshot.png_bytes))[:, :, :3]
    assert actual.shape == reference.shape
    # Line contours occupy a thin ink mask. A filled contour has a vastly larger
    # coloured area even if its boundary/axes are in the correct location.
    reference_ink = np.ptp(reference.astype(int), axis=2) > 30
    actual_ink = np.ptp(actual.astype(int), axis=2) > 30
    assert reference_ink.any()
    assert .7 < actual_ink.sum() / reference_ink.sum() < 1.3
    pd.testing.assert_frame_equal(snapshot.dataframe(), source, check_exact=True)
    metadata, blobs = snapshot.to_storage()
    restored = B.FigureSnapshot.from_storage(metadata, blobs)
    assert restored.config()['style']['appearance']['series_styles'][target.series_id]['color'] == '#aa2277'
    editor.close()  # Reopening/export has only retained bytes, never the source Figure.
    assert editor._figure is None
    for extension in ('png', 'svg'):
        options = resolve_export_options((restored.width, restored.height), extension=extension, native=True)
        destination = tmp_path / ('reopened-contours.' + extension)
        B.write_frozen_figure_image(destination, restored, options)
        if extension == 'svg':
            assert_unfilled_svg(destination.read_bytes())
        else:
            np.testing.assert_array_equal(_rgba(QtGui.QImage(str(destination)))[:, :, :3], actual)
