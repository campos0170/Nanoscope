from dataclasses import replace
import pytest

from nanoscope.core.figure_spec import FigureValidationError
from nanoscope.core.figure_style import FigureStyle, builtin_styles
from nanoscope.core.figure_style_registry import (WORKSPACE_KEY, StyleRegistry,
    registry_from_workspace, registry_to_workspace, resolve_ref)


def test_project_default_import_keeps_custom_physical_look_and_is_idempotent():
    extra = {'future': {'untouched': [1]}, 'figure_workspace_v1': {'default_style':
        {'font_family': 'Custom Font', 'title_size': 19, 'line_width': 2.5, 'grid': True}}}
    registry = registry_from_workspace(extra).registry
    default = resolve_ref(registry, registry.default_ref)
    assert default.look['font_family'] == 'Custom Font'
    assert default.look['line_width_pt'] == 1.875
    saved = registry_to_workspace(extra, registry)
    assert registry_from_workspace(saved).registry == registry
    assert saved['future'] == extra['future'] and WORKSPACE_KEY not in extra
    assert [s.name for s in registry.available()][:2] == [s.name for s in builtin_styles()][:2]


def test_pinned_revision_is_immutable_and_delete_does_not_break_snapshot():
    registry = registry_from_workspace({}).registry
    first = FigureStyle('custom', 1, 'Custom', {'body_size_pt': 8})
    registry = registry.add_revision(first)
    second = replace(first, revision=2, look={'body_size_pt': 9})
    registry = registry.add_revision(second)
    assert resolve_ref(registry, first.ref).look['body_size_pt'] == 8
    assert [s.revision for s in registry.available() if s.style_id == 'custom'] == [2]
    hidden = registry.hide('custom')
    assert all(s.style_id != 'custom' for s in hidden.available())
    assert resolve_ref(hidden, first.ref) == first
    assert resolve_ref(None, first.ref, first.to_dict()) == first
    with pytest.raises(FigureValidationError): registry.add_revision(second)
    with pytest.raises(FigureValidationError): registry.hide(registry.default_ref.style_id)


def test_import_content_identity_detaches_and_preserves_local_name_collisions():
    registry = registry_from_workspace({}).registry
    native = {'line_width': 1, 'title': 'not style', 'future': {'a': 1}}
    one, ref = registry.import_legacy_style('7 pt Arial · Nature, Elsevier', native)
    two, again = one.import_legacy_style('Different local name', native)
    assert two == one and again == ref
    style = resolve_ref(one, ref)
    assert style.name.endswith('(imported 2)')
    assert style.extra['legacy_remainder']['title'] == 'not style'
    native['future']['a'] = 9
    assert style.extra['legacy_remainder']['future']['a'] == 1


@pytest.mark.parametrize('raw', [{'version': 99, 'future': ['keep']}, ['opaque'],
                               {'version': 1, 'styles': [{'broken': True}]}])
def test_unreadable_registry_is_carried_without_default_substitution(raw):
    extra = {WORKSPACE_KEY: raw, 'untouched': 8}
    read = registry_from_workspace(extra)
    assert read.registry is None and read.opaque == raw and read.warnings
    with pytest.raises(FigureValidationError, match='Cannot overwrite'):
        registry_to_workspace(extra, StyleRegistry())
    assert extra[WORKSPACE_KEY] == raw


def test_unreadable_phase2_custom_default_is_not_replaced():
    extra = {'figure_workspace_v1': {'default_style': {'title_size': 'broken'}}}
    read = registry_from_workspace(extra)
    assert read.registry is None and read.warnings


def test_unknown_registry_fields_round_trip_and_default_cannot_disappear():
    raw = registry_from_workspace({}).registry.to_dict()
    raw['future'] = {'one': 1}
    registry = StyleRegistry.from_dict(raw)
    assert registry.to_dict() == raw
    raw['default_ref']['revision'] = 99
    with pytest.raises(FigureValidationError): StyleRegistry.from_dict(raw)


def test_editing_default_advances_new_default_without_mutating_pinned_revisions():
    registry = registry_from_workspace({}).registry
    first = FigureStyle('custom', 1, 'Custom', {'body_size_pt': 9})
    registry = registry.add_revision(first).with_default(first.ref)
    second = replace(first, revision=2, look={'body_size_pt': 11})
    changed = registry.add_revision(second)
    assert changed.default_ref == second.ref
    assert resolve_ref(changed, first.ref) == first
    assert registry.default_ref == first.ref


def test_legacy_import_identity_ignores_writer_version_but_retains_original_payload():
    registry = registry_from_workspace({}).registry
    one, ref = registry.import_legacy_style('Local', {'title_size': 8, 'app_version': 'old'})
    two, again = one.import_legacy_style('Local', {'title_size': 8, 'app_version': 'new'})
    assert two == one and ref == again
    assert resolve_ref(two, ref).extra['legacy_remainder']['app_version'] == 'old'
