"""Shared panel letters retain physical metrics and explicit manual ownership."""
from copy import deepcopy

import pytest

from nanoscope.core.figure_style import FigureStyle, resolve_bound_text, style_binding, builtin_styles
from nanoscope.core.figure_composition_style import (DEFAULT_LABEL_STYLE, panel_style,
                                                      panel_overrides_from_legacy)


def test_panel_role_and_exact_target_override_have_distinct_origins():
    style = FigureStyle('panel', 2, 'Panel', {'panel_size_pt': 9.125,
        'text_roles': {'panel': {'color': '#123456', 'bold': False},
                       'annotation.panel.first': {'italic': True}}})
    bound = style_binding(style)
    values, origins = resolve_bound_text(bound, 'annotation.panel.first', {'bold': True})
    assert values['size_pt'] == 9.125 and origins['size_pt'] == 'style'
    assert values['color'] == '#123456' and values['italic'] is True
    assert values['bold'] is True and origins['bold'] == 'element'
    assert resolve_bound_text(bound, 'annotation.panel')[0]['size_pt'] == 9.125
    assert all(style.look['panel_size_pt'] == style.look['body_size_pt'] + 1 for style in builtin_styles())
    default_panel = resolve_bound_text(style_binding(builtin_styles()[0]), 'annotation.panel')[0]
    assert default_panel['size_pt'] == 8 and default_panel['bold'] is True


def test_composition_binding_is_only_projection_authority_and_keeps_manual_edits():
    binding = style_binding(FigureStyle('panel', 1, 'Paper', {'panel_size_pt': 9.125,
        'font_family': 'Times New Roman', 'text_roles': {'panel': {'bold': False}}}))
    before = deepcopy(binding)
    native = panel_style(binding, {'color': '#abcdef', 'italic': True})
    assert native['size_px'] == pytest.approx(9.125 * 96 / 72)
    assert native['family'] == 'Times New Roman' and native['bold'] is False
    assert native['color'] == '#abcdef' and native['italic'] is True
    assert binding == before
    # A former stored projection does not override the current bound role.
    native['size_px'] = 69
    assert panel_style(binding, legacy=native)['size_px'] == pytest.approx(9.125 * 96 / 72)


def test_old_composition_typography_round_trips_without_adopting_new_defaults():
    legacy = dict(DEFAULT_LABEL_STYLE, size_px=30, family='Helvetica', bold=False)
    assert panel_style(None, legacy=legacy) == legacy
    overrides = panel_overrides_from_legacy(legacy)
    assert overrides == {'size_pt': 22.5, 'family': 'Helvetica', 'bold': False}
    assert panel_overrides_from_legacy(DEFAULT_LABEL_STYLE) == {}
    assert panel_style(style_binding(builtin_styles()[0]), overrides)['size_px'] == 30


@pytest.mark.parametrize('bad', [True, 0, -2, float('nan')])
def test_panel_size_uses_physical_metric_validation(bad):
    with pytest.raises(ValueError):
        FigureStyle('panel', 1, 'Bad', {'panel_size_pt': bad})
