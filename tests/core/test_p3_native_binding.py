"""The native look is a projection; its shared style and overrides own the values."""
from copy import deepcopy

from nanoscope.core.figure_style import FigureStyle, builtin_styles
from nanoscope.gui.figure_style_binding import (
    bind_native_style, project_native, reconcile_native_edit, resolve_bound_text)


def test_same_physical_style_projects_to_pg_pixels_and_matplotlib_points():
    style = builtin_styles()[0]
    original = {"title": "Typed words", "axes": {"bottom": {"range": [1, 4]}},
                "fit_label": {"template": "my tau", "fields": ["n"], "offset": [4, 8]}}
    pg = bind_native_style(original, style)
    mpl = bind_native_style(original, style, "matplotlib_appearance")
    assert pg["line_width"] == 1. and mpl["line_width"] == .75
    assert pg["marker_size"] == 4. and mpl["marker_size"] == 3.
    for key, value in original.items():
        assert pg[key] == mpl[key] == value
    damaged = {**pg, "line_width": 999}
    assert project_native(damaged)["line_width"] == 1.


def test_deliberate_native_edit_updates_override_without_detaching_style():
    before = bind_native_style({"title": "kept"}, builtin_styles()[0])
    changed = reconcile_native_edit(before, {**before, "line_width": 2.})
    assert changed["style_binding"]["ref"] == before["style_binding"]["ref"]
    assert changed["style_binding"]["overrides"]["line_width_pt"] == 1.5
    cleared = reconcile_native_edit(changed, {**changed, "line_width": None})
    assert cleared["line_width"] == 1.
    assert "line_width_pt" not in cleared["style_binding"]["overrides"]
    assert cleared["title"] == "kept"


def test_explicit_value_equal_to_style_stays_explicit_during_another_edit():
    before = bind_native_style({}, builtin_styles()[0])
    before["style_binding"]["overrides"]["line_width_pt"] = .75
    after = reconcile_native_edit(before, {**before, "title_size": 9.})
    assert after["style_binding"]["overrides"] == {"line_width_pt": .75, "title_size_pt": 9.}


def test_role_resolution_does_not_flatten_inheritance_into_element_storage():
    style = FigureStyle(style_id="project.test", revision=1, name="Test",
        look={"title_size_pt": 9., "text_roles": {"title": {"bold": True, "color": "#ff0000"}}})
    native = bind_native_style({"text": {"title": {"italic": True}}}, style)
    before = deepcopy(native)
    values, origins = resolve_bound_text(native["style_binding"], "title", native["text"]["title"])
    assert values["size_pt"] == 9. and values["bold"] is True and values["italic"] is True
    assert origins["size_pt"] == origins["bold"] == "style"
    assert origins["italic"] == "element"
    assert native == before
