"""Real v12 artifact boundaries plus canonical style/history ownership."""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import ast

import pytest

from nanoscope.core.project import load_project
from nanoscope.core.figure_spec import FigureRecord, FrozenMap
from nanoscope.core.figure_style import builtin_styles, style_binding
from nanoscope.core.figure_style_mapping import project_legacy_style, STYLE_SETTINGS_OWNERS
from nanoscope.core.figure_codecs import (NATIVE_KINDS, decode_legacy, encode_legacy,
    merge_native_capture, record_to_artifact, record_from_artifact, _native_style_path)

ID = '64d9be44-d980-43ab-8da5-bcc95a827e45'


def native(identity):
    fixture = Path(__file__).parent / 'fixtures' / 'phase2' / 'all_pages_v12.nanoscope'
    return next(a for a in load_project(fixture).workspace.artifacts if a.artifact_id == identity)


def style_at(artifact):
    kind = NATIVE_KINDS[artifact.artifact_id][1]
    path, vocabulary = _native_style_path(kind)
    value = artifact.payload
    for part in path: value = value[part]
    return value, vocabulary


def bound_native(identity, style=None):
    artifact = native(identity)
    appearance, vocabulary = style_at(artifact)
    projected = project_legacy_style(style_binding(style or builtin_styles()[0]), vocabulary, appearance)
    appearance.clear(); appearance.update(projected)
    return artifact


@pytest.mark.parametrize('identity', tuple(NATIVE_KINDS))
def test_real_eight_kinds_own_one_binding_and_inverse_projection(identity):
    artifact = bound_native(identity)
    original = deepcopy(artifact.payload)
    decoded = decode_legacy(artifact, figure_id=ID, name='Own')
    presentation = decoded.spec.presentation
    assert presentation.style_ref['style_id'] == builtin_styles()[0].style_id
    assert not presentation.overrides  # inherited look is not flattened into hand values
    native_remainder = decoded.spec.result.payload
    for part in _native_style_path(decoded.spec.kind)[0]: native_remainder = native_remainder[part]
    assert 'style_binding' not in native_remainder and 'line_width' not in native_remainder
    assert encode_legacy(decoded.spec, decoded.resources).payload == original
    reopened = record_from_artifact(record_to_artifact(decoded.record))
    assert reopened.spec == decoded.spec
    changed = replace(decoded.spec, presentation=replace(presentation, overrides={'line_width_pt': 1.5}))
    edited = encode_legacy(changed, decoded.resources)
    appearance, vocabulary = style_at(edited)
    assert appearance['line_width'] == (1.5 if vocabulary == 'matplotlib_appearance' else 2.)
    # Outside the native appearance subtree, every committed numeric result is unchanged.
    a, b = deepcopy(original), deepcopy(edited.payload)
    path, _ = _native_style_path(changed.kind)
    for part in path[:-1]: a, b = a[part], b[part]
    a.pop(path[-1]); b.pop(path[-1]); assert a == b


def test_stale_projection_cannot_override_binding_and_is_reported():
    artifact = bound_native('builtin.iv')
    appearance, _ = style_at(artifact)
    appearance['line_width'] = 99
    decoded = decode_legacy(artifact, figure_id=ID, name='Own')
    assert decoded.report.warnings
    assert style_at(encode_legacy(decoded.spec, decoded.resources))[0]['line_width'] == 1.
    assert appearance['line_width'] == 99  # ingress is detached


def test_native_undo_restores_binding_revision_in_same_capture_and_reset_drops_it():
    first = builtin_styles()[0]
    before = decode_legacy(bound_native('builtin.iv', first), figure_id=ID, name='Own').record
    after_style = replace(first, style_id='project.style', revision=1, name='Custom')
    changed = merge_native_capture(bound_native('builtin.iv', after_style), before)
    assert changed.spec.presentation.style_ref['style_id'] == 'project.style'
    undone = merge_native_capture(bound_native('builtin.iv', first), changed)
    assert undone.spec.presentation.style_ref['style_id'] == first.style_id
    reset = merge_native_capture(native('builtin.iv'), undone)
    assert reset.spec.presentation.style_ref is None and not reset.spec.presentation.style_snapshot
    assert 'style_binding_metadata' not in reset.spec.presentation.extra


def test_native_workspace_encoder_thaws_binding_arrays_without_tuple_tags():
    from nanoscope.core.analysis_workspace import _encode, _Resources
    @dataclass(frozen=True)
    class NativeStyle:
        style_binding: object
    binding = style_binding(builtin_styles()[0])
    binding['snapshot']['extra']['future_array'] = [1, {'nested': [2]}]
    packed = _encode(NativeStyle(FrozenMap(binding)), _Resources(), 'style')
    assert packed['fields']['style_binding']['snapshot']['extra']['future_array'] == [1, {'nested': [2]}]


def test_every_native_style_settings_field_has_an_explicit_owner():
    source = Path(__file__).parents[2] / 'nanoscope' / 'gui' / 'plots_panel.py'
    module = ast.parse(source.read_text())
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'StyleSettings')
    fields = {n.target.id for n in node.body if isinstance(n, ast.AnnAssign)}
    assert not fields - set(STYLE_SETTINGS_OWNERS)
