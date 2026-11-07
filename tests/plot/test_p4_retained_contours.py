"""Retained styling must never turn scientific line contours into filled regions."""
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, to_hex, to_rgba
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

from nanoscope.plot.figures import joint_density
from nanoscope.plot.study_figure_style import StudyFigureStyle


def density_figure():
    # The actual report builder estimates two small, asymmetric source clouds once.
    rng = np.random.default_rng(412)
    groups = {}
    for name, center, depth in [('Apo 500 mV', .002, 20.), ('Apo 600 mV', .006, 55.)]:
        groups[name] = pd.DataFrame({'dwell_s': rng.lognormal(np.log(center), .25, 40),
            'depth_pA': rng.normal(depth, 3., 40), 'accepted': np.ones(40, dtype=bool)})
    figure, axis = joint_density(groups, gridsize=28, levels=4, legend_loc='none')
    FigureCanvasAgg(figure).draw()
    return figure, axis


def assert_unfilled_svg(payload, *, groups=2):
    root = ET.fromstring(payload)
    contours = [node for node in root.iter() if node.attrib.get('id', '').startswith('QuadContourSet_')]
    assert len(contours) == groups
    for group in contours:
        paths = [node for node in group.iter() if node.tag.endswith('path')]
        assert paths
        for path in paths:
            style = dict(part.strip().split(':', 1) for part in path.attrib.get('style', '').split(';')
                         if ':' in part)
            assert style.get('fill', '').strip() == 'none'
            assert style.get('stroke', '').strip() not in ('', 'none')


def _svg(figure):
    stream = io.BytesIO(); figure.savefig(stream, format='svg')
    return stream.getvalue()


def _pixels(figure):
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba()).copy()


def test_real_joint_density_noop_recolour_reset_and_saved_style_keep_unfilled_geometry(monkeypatch):
    import scipy.stats
    figure, axis = density_figure()
    contours = [item for item in axis.collections if isinstance(item, ContourSet)]
    assert len(contours) == 2 and all(not item.filled for item in contours)
    geometry = [[path.vertices.copy() for path in item.get_paths()] for item in contours]
    levels = [item.levels.copy() for item in contours]
    sources = figure._nanoscope_source_data.copy(deep=True)
    mappings = [(item.norm, item.cmap, item.get_array().copy()) for item in contours]
    baseline = _pixels(figure)
    assert_unfilled_svg(_svg(figure))
    monkeypatch.setattr(scipy.stats, 'gaussian_kde', lambda *a, **k: pytest.fail('Styling reran KDE'))
    styler = StudyFigureStyle(figure)
    try:
        styler.apply()
        # Same renderer and unchanged style: require identical pixels, not merely
        # a valid file or a small error that could overlook the filled interiors.
        np.testing.assert_array_equal(_pixels(figure), baseline)
        assert all(not item.get_fill() and len(item.get_facecolors()) == 0 for item in contours)
        assert_unfilled_svg(_svg(figure))
        original = to_hex(contours[0].get_edgecolors()[0])
        target = next(entry for entry in styler.series_descriptors() if entry.identity_key == original)
        assert {'color', 'line_width_pt'} <= set(target.capabilities)
        saved = {'appearance': {'series_styles': {
            target.series_id: {'color': '#aa2277', 'line_width_pt': 1.5}}}}
        styler.apply(**saved)
        recoloured = _pixels(figure)
        np.testing.assert_allclose(contours[0].get_edgecolors(),
            np.tile(to_rgba('#aa2277'), (len(contours[0].levels), 1)))
        assert_unfilled_svg(_svg(figure))
        assert all(not item.get_fill() and len(item.get_facecolors()) == 0 for item in contours)
        styler.apply()
        np.testing.assert_array_equal(_pixels(figure), baseline)
        styler.apply(**json.loads(json.dumps(saved)))
        np.testing.assert_array_equal(_pixels(figure), recoloured)
        for item, paths, expected_levels, (norm, cmap, array) in zip(contours, geometry, levels, mappings):
            assert item.norm is norm and item.cmap is cmap
            np.testing.assert_array_equal(item.levels, expected_levels)
            np.testing.assert_array_equal(item.get_array(), array)
            for path, expected in zip(item.get_paths(), paths):
                np.testing.assert_array_equal(path.vertices, expected)
        pd.testing.assert_frame_equal(figure._nanoscope_source_data, sources, check_exact=True)
    finally:
        from matplotlib import pyplot as plt
        plt.close(figure)


@pytest.mark.parametrize('filled', [False, True])
def test_continuous_contour_mapping_keeps_native_fill_and_norm_semantics(filled):
    figure = Figure(figsize=(3, 2)); FigureCanvasAgg(figure)
    axis = figure.subplots()
    x, y = np.meshgrid(np.linspace(-2., 2., 17), np.linspace(-1., 1., 15))
    z = x * x + y * y
    draw = axis.contourf if filled else axis.contour
    contour = draw(x, y, z, levels=[.2, .6, 1.2, 2.], cmap='viridis', norm=Normalize(.2, 2.))
    figure.canvas.draw()
    native = (contour.get_facecolors().copy(), contour.get_edgecolors().copy())
    norm, cmap = contour.norm, contour.cmap
    array, levels = contour.get_array().copy(), contour.levels.copy()
    styler = StudyFigureStyle(figure)
    styler.apply(appearance={'line_width': .75})
    figure.canvas.draw()
    assert contour.get_fill() == filled
    for current, original in zip((contour.get_facecolors(), contour.get_edgecolors()), native):
        np.testing.assert_array_equal(current, original)
    assert contour.norm is norm and contour.cmap is cmap
    contour.set_clim(.1, 3.)
    figure.canvas.draw()
    mapped = contour.get_facecolors() if filled else contour.get_edgecolors()
    np.testing.assert_allclose(mapped, contour.to_rgba(contour.cvalues, contour.get_alpha()))
    np.testing.assert_array_equal(contour.get_array(), array)
    np.testing.assert_array_equal(contour.levels, levels)
    figure.clear()
