"""Print estimates are conservative, read-only and explicit about their scope."""
import math

import pytest

from nanoscope.gui.plot_print_preflight import inspect_print_metrics, readability_enlargement


def svg(contents):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">'
            + contents + '</svg>').encode()


def test_explicit_fonts_and_lines_use_physical_size_not_raster_dpi():
    payload = svg('<g font-size="12"><text x="5" y="20">Label</text>'
                  '<path d="M0 0L100 10" stroke="black" stroke-width="2"/></g>')
    before = bytes(payload)
    metrics = inspect_print_metrics(payload)
    assert metrics.text_size == 12
    assert metrics.stroke_size == 2
    assert metrics.at_size(50.8, 25.4) == pytest.approx((2.16, 0.36))
    assert payload == before
    factor = readability_enlargement(metrics, 50.8, 25.4)
    assert factor == pytest.approx(8 / 2.16)
    assert metrics.at_size(50.8 * factor, 25.4 * factor)[0] == pytest.approx(8)


def test_inherited_style_matrix_rotation_and_hidden_defs():
    payload = svg('<defs><text font-size="0.1">Unused</text></defs>'
                  '<g font-size="100" transform="translate(20 10) scale(2)">'
                  '<text style="font-size:9pt" transform="rotate(90)">Rotated</text>'
                  '<text font-size="0.1" display="none">Hidden</text></g>')
    metrics = inspect_print_metrics(payload)
    assert metrics.text_size == pytest.approx(24)
    assert metrics.text_count == 1


def test_nonuniform_transform_uses_conservative_smallest_scale():
    metrics = inspect_print_metrics(svg('<text font-size="10" transform="matrix(2,0,0,3,0,0)">T</text>'))
    assert metrics.text_size == pytest.approx(20)


def test_cosmetic_strokes_and_outlined_glyphs_do_not_get_invented_sizes():
    metrics = inspect_print_metrics(svg('<path stroke="black" stroke-width="1" '
                                       'vector-effect="non-scaling-stroke" d="M0 0L10 10"/>'
                                       '<use href="#font-glyph"/>'))
    assert metrics.text_size is metrics.stroke_size is None
    assert any("Cosmetic" in note for note in metrics.notes)
    assert any("glyphs" in note for note in metrics.notes)
    assert readability_enlargement(metrics, 85, 42.5) == 1


@pytest.mark.parametrize("content", [
    '<text font-size="1em">Relative</text>',
    '<text font-size="12" transform="skewX(15)">Transform</text>',
    '<svg viewBox="0 0 20 20"><text font-size="12">Nested</text></svg>',
    '<style>text {font-size:12px}</style><text>CSS</text>',
    '<g font-size="20"><text style="font: 1px Arial">Shorthand</text></g>',
])
def test_unmeasurable_cases_are_disclosed(content):
    metrics = inspect_print_metrics(svg(content))
    assert metrics.text_size is None
    assert metrics.notes


@pytest.mark.parametrize("payload", [None, b"", b"<broken", b'<svg/>',
                                     b'<!DOCTYPE svg><svg viewBox="0 0 800 400"/>'])
def test_bad_or_unavailable_svg_is_nonfatal(payload):
    metrics = inspect_print_metrics(payload)
    assert metrics.text_size is None
    assert metrics.notes


def test_enlargement_never_shrinks_or_accepts_invalid_targets():
    metrics = inspect_print_metrics(svg('<text font-size="100">Large</text>'))
    assert readability_enlargement(metrics, 180, 90) == 1
    for value in (0, math.nan, math.inf, -1):
        with pytest.raises(ValueError, match="targets"):
            readability_enlargement(metrics, 180, 90, text_target_pt=value)
