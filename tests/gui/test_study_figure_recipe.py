"""Reconstruction retains scientific sidecars, not prior presentation edits."""
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nanoscope.gui import study_figure_recipe as recipes
from nanoscope.plot.incubation_figures import _save


def test_reconstruction_excludes_obsolete_style_sidecars(tmp_path, monkeypatch):
    image = tmp_path / 'test.png'
    image.write_bytes(b'Original image is not read by reconstruction')
    (tmp_path / 'test.style.json').write_text('{"old":"styling"}')
    (tmp_path / 'methods.json').write_text('{"event_end":"inclusive"}')
    (tmp_path / 'original.csv').write_text('value\n1.25\n')
    frame = pd.DataFrame({'event_id': ['a'], 'value': [1.23456789]})
    def render(state, recipe, folder, **kwargs):
        figure = Figure(figsize=(3, 2)); FigureCanvasAgg(figure)
        figure.subplots().plot([0., 1.], [1., 2.])
        _save(figure, folder, 'test', frame)
        (folder / 'test.style.json').write_text('{"also":"not science"}')
        (folder / 'display.csv').write_text('x,y\n0,1\n1,2\n')
    monkeypatch.setattr(recipes, 'render_group', render)
    state = {'figure_recipes': {'test': {'group': 0, 'tag': 'test'}}}
    figure, data, resources, recipe = recipes.reconstruct(state, image)
    try:
        assert set(resources) == {'methods.json', 'original.csv', 'display.csv'}
        assert resources['methods.json'] == b'{"event_end":"inclusive"}'
        pd.testing.assert_frame_equal(data, frame, check_exact=True)
    finally:
        figure.clear()
