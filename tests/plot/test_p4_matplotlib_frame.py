"""Matplotlib frame choices never expose hidden marginal axes or change data."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from nanoscope.core.figure_style import FigureStyle, style_binding
from nanoscope.plot.study_figure_style import StudyFigureStyle


def test_retained_frame_is_reversible_and_leaves_hidden_marginal_axes_off():
    figure = Figure(figsize=(5, 3)); FigureCanvasAgg(figure)
    main, marginal = figure.subplots(1, 2)
    line, = main.plot([1., 2.], [3., 4.], color='#334455')
    main.set_xlim(1., 2.); main.set_ylim(2., 5.)
    main.spines['top'].set_visible(False)
    marginal.set_axis_off()
    original = {side: spine.get_visible() for side, spine in main.spines.items()}
    limits = main.get_xlim(), main.get_ylim()
    styler = StudyFigureStyle(figure)
    bound = style_binding(FigureStyle('box', 1, 'Box', look={'frame': True,
                                                            'palette': ['#ff0000', '#00ff00']}))
    styler.apply(style_binding=bound)
    assert main.spines['top'].get_visible() and main.spines['right'].get_visible()
    assert not marginal.axison
    assert to_rgba(line.get_color()) == to_rgba('#334455')  # retained RGB group is an identity, not a palette slot
    assert (main.get_xlim(), main.get_ylim()) == limits
    np.testing.assert_array_equal(line.get_xydata(), [[1., 3.], [2., 4.]])
    styler.apply(appearance={'frame': False})
    assert not main.spines['top'].get_visible() and not main.spines['right'].get_visible()
    assert main.spines['bottom'].get_visible() and main.spines['left'].get_visible()
    styler.apply()
    assert {side: spine.get_visible() for side, spine in main.spines.items()} == original
    assert not marginal.axison
