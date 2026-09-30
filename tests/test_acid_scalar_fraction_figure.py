"""Pandas plots must remain in the acid worker's owned non-Qt figure."""
import numpy as np
import pandas as pd
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends.backend_agg import FigureCanvasAgg

from nanoscope.analysis import acid_incubation as acid


def test_scalar_fractions_have_bars_in_private_agg_figure(tmp_path):
    managers = tuple(Gcf.get_all_fig_managers())
    with acid._math_session(7):
        runtime = acid.AcidRuntime(acid.default_settings(), tmp_path,
            {'tag': 'test', 'source_path': str(tmp_path / 'unused.abf'), 'incubation_h': 1})
        runtime.namespace['primary_residualized'] = pd.DataFrame({
            'time_block': [0, 0, 0, 1, 1, 1],
            'scalar_component': [0, 0, 1, 0, 1, 1]})
        captured = []
        runtime.namespace['finish_figure'] = lambda name: captured.append((name, runtime.plot.gcf()))
        runtime.run(50)
        assert len(captured) == 1
        name, figure = captured[0]
        assert name == '05_scalar_components_by_time'
        assert isinstance(figure.canvas, FigureCanvasAgg)
        assert figure.canvas.manager is None
        assert len(figure.axes) == 1
        heights = [patch.get_height() for patch in figure.axes[0].patches]
        np.testing.assert_allclose(heights, [2/3, 1/3, 1/3, 2/3])
        assert tuple(Gcf.get_all_fig_managers()) == managers
        runtime.plot.close(figure)
