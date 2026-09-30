"""Study profiles and lightweight routing, independent of numerical engines.

Profiles supply named defaults for NEW workspaces/recordings. Opening a saved
workspace never reapplies those defaults, guesses chemistry from file names, or
changes its scientific method. The normal engine loader still validates all
checkpoint files after the launcher has selected the appropriate workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


MANUAL_WORKSPACE_KIND = 'manual_incubation_v1'
ACID_WORKSPACE_KIND = 'acid_incubation_v1'
ACID_SCHEMA = 'nanoscope-acid-incubation-v1'


@dataclass(frozen=True)
class IncubationProfile:
    id: str
    label: str
    engine: str
    description: str
    default_name: str
    default_condition_label: str
    median_engine: str | None


PROFILES = (
    IncubationProfile('general', 'General incubation', 'manual',
        'For any sample or condition. Prepare recordings, detect without a model, '
        'curate with the main event tools, and compare reviewed events. A trained model is optional.',
        'Incubation study', '', 'reflect'),
    IncubationProfile('zn', 'Zn incubation', 'manual',
        'The general staged workflow with reflected median-window edges. '
        'Detection can use the signal, the main event tools, or an optional trained model.',
        'Zn incubation study', 'Zn', 'reflect'),
    IncubationProfile('cu', 'Cu incubation', 'manual',
        'The general staged workflow with shrinking median-window edges. '
        'Channel combinations, retained intervals, polarity and detection remain editable.',
        'Cu incubation study', 'Cu', 'shrinking'),
    IncubationProfile('acid', 'Acid incubation', 'acid',
        'The calibrated Acid CUSUM method with its saved preparation, measurement '
        'and family-analysis definitions. No trained model is required.',
        'Acid incubation study', 'Acid', None),
)
_BY_ID = {profile.id: profile for profile in PROFILES}


def get_profile(profile_id='general'):
    """Resolve an explicit profile ID; recording names never participate."""
    if not isinstance(profile_id, str):
        raise ValueError('Choose a supported incubation profile.')
    profile = _BY_ID.get(profile_id.strip().casefold())
    if profile is None:
        raise ValueError(f'Unsupported incubation profile: {profile_id!r}.')
    return profile


def recording_defaults(profile='general'):
    """Return independent, scientifically established new-recording defaults."""
    selected = get_profile(profile)
    return {'median_engine': selected.median_engine} if selected.engine == 'manual' else {}


def _text(value, label, *, fallback=''):
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise ValueError(f'{label} must be text.')
    return value.strip()


def initialize_study_metadata(state, profile='general', *, name='', sample='', condition_label=None):
    """Return a new empty state with explicit profile/condition metadata.

    This is a creation helper, never a migration. Numeric preparation,
    detection and comparison settings remain exactly as supplied by the engine.
    """
    selected = get_profile(profile)
    if not isinstance(state, dict):
        raise ValueError('New study state must be a mapping.')
    if ('study_profile' in state or state.get('records') or state.get('history')
            or state.get('figures') or any(state.get(key) is not None
                for key in ('analysis', 'comparison', 'features', 'clustering', 'mixture'))):
        raise ValueError('Choose a profile when creating a new empty study; saved scientific settings are preserved.')
    schema, kind = state.get('schema'), state.get('workspace_kind')
    if selected.engine == 'acid':
        if schema != ACID_SCHEMA or kind not in (None, ACID_WORKSPACE_KIND):
            raise ValueError('The Acid profile requires a new calibrated Acid workspace.')
    elif schema is not None or kind not in (None, MANUAL_WORKSPACE_KIND):
        raise ValueError('General, Zn and Cu profiles require a new general workspace.')
    if _engine(state) != selected.engine:
        raise ValueError('The new incubation profile does not match its scientific workspace.')
    return dict(state,
        workspace_kind=ACID_WORKSPACE_KIND if selected.engine == 'acid' else MANUAL_WORKSPACE_KIND,
        study_profile=selected.id,
        name=_text(name, 'Study name') or selected.default_name,
        sample=_text(sample, 'Sample'),
        condition_label=_text(condition_label, 'Condition label', fallback=selected.default_condition_label))


_MANUAL_KEYS = frozenset(('name', 'model_path', 'records', 'comparison', 'features',
    'clustering', 'mixture', 'figures', 'history', 'comparison_settings',
    'cluster_settings', 'mixture_settings'))
_ACID_KEYS = frozenset(('schema', 'name', 'records', 'settings', 'figures', 'history', 'analysis'))


def _engine(values):
    """Recognize explicit scientific schemas before considering legacy shape."""
    schema, kind = values.get('schema'), values.get('workspace_kind')
    if schema is not None and schema != ACID_SCHEMA:
        raise ValueError(f'Unsupported incubation workspace schema: {schema!r}.')
    if kind is not None and kind not in (MANUAL_WORKSPACE_KIND, ACID_WORKSPACE_KIND):
        raise ValueError(f'Unsupported incubation workspace kind: {kind!r}.')
    if schema == ACID_SCHEMA or kind == ACID_WORKSPACE_KIND:
        if kind == MANUAL_WORKSPACE_KIND or not _ACID_KEYS <= values.keys():
            raise ValueError('Conflicting or incomplete calibrated Acid workspace metadata.')
        return 'acid'
    if not _MANUAL_KEYS <= values.keys():
        raise ValueError('This checkpoint is not a recognized incubation study workspace.')
    return 'manual'


def saved_profile(state):
    """Read a saved profile; old markerless manual studies stay general."""
    if not isinstance(state, dict):
        raise ValueError('Invalid incubation workspace metadata.')
    engine = _engine(state)
    selected = get_profile(state.get('study_profile', 'acid' if engine == 'acid' else 'general'))
    if selected.engine != engine:
        raise ValueError('The saved incubation profile conflicts with its scientific workspace.')
    return selected.id


@dataclass(frozen=True)
class WorkspaceRoute:
    engine: str
    profile_id: str
    name: str
    path: Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate fields in the workspace manifest.')
        result[key] = value
    return result


def _manifest_fields(encoded):
    """Peek ONLY the root mapping; do not unpack arrays, paths or result objects."""
    if not isinstance(encoded, dict) or encoded.get('type') != 'mapping':
        raise ValueError('Invalid incubation workspace state envelope.')
    items = encoded.get('items')
    if not isinstance(items, list):
        raise ValueError('Invalid incubation workspace state fields.')
    fields = {}
    for pair in items:
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
            raise ValueError('Incubation workspace fields must have text names.')
        key, value = pair
        if key in fields:
            raise ValueError('Duplicate fields in the incubation workspace state.')
        fields[key] = value
    for key in ('name', 'schema', 'workspace_kind', 'study_profile', 'model_path'):
        if key in fields and not isinstance(fields[key], str):
            raise ValueError(f'Invalid saved incubation {key.replace("_", " ")}.')
    for key in ('records', 'history', 'figures'):
        value = fields.get(key)
        if not isinstance(value, dict) or value.get('type') != 'list' or not isinstance(value.get('values'), list):
            raise ValueError(f'Invalid saved incubation {key}.')
    return fields


def peek_workspace(path):
    """Select a workspace engine by its manifest, with no scientific data loads.

    This is routing, not integrity verification: the selected engine subsequently
    opens the checkpoint through its normal hash-validating loader. Reference
    recipe/result bundles are opened by the separate reference-reproduction UI.
    """
    path = Path(path).expanduser().resolve()
    folder = path.parent if path.name == 'workspace.json' and path.is_file() else path
    try:
        manifest = json.loads((folder / 'workspace.json').read_text(encoding='utf-8'),
            object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Choose a saved incubation workspace containing workspace.json.') from exc
    if (not isinstance(manifest, dict) or manifest.get('format') != 'nanoscope-manual-workspace'
            or type(manifest.get('version')) is not int or manifest['version'] != 1):
        raise ValueError('This is not a supported incubation workspace checkpoint.')
    fields = _manifest_fields(manifest.get('state'))
    profile_id = saved_profile(fields)
    return WorkspaceRoute(get_profile(profile_id).engine, profile_id, fields['name'], folder)
