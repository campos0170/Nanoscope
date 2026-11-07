"""Project-owned style revisions. Reading never changes the computer's presets."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib

from .figure_spec import _Node, FrozenMap, FigureValidationError, thaw_json, canonical_json
from .figure_style import FigureStyle, StyleRef, builtin_styles
from .figure_style_mapping import import_legacy_look

WORKSPACE_KEY = 'figure_styles_v1'


@dataclass(frozen=True)
class StyleRegistry(_Node):
    styles: tuple[FigureStyle, ...] = ()
    default_ref: StyleRef | None = None
    imported_legacy_ids: tuple[str, ...] = ()
    hidden_ids: tuple[str, ...] = ()
    version: int = 1
    _sequences = {'styles': FigureStyle}

    def __post_init__(self):
        super().__post_init__()
        if type(self.version) is not int or self.version != 1:
            raise FigureValidationError('Unsupported project style registry version')
        if self.default_ref is not None and not isinstance(self.default_ref, StyleRef):
            raise FigureValidationError('Invalid default style reference')
        for name in ('imported_legacy_ids', 'hidden_ids'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(type(v) is not str for v in values):
                raise FigureValidationError(f'{name} requires text identities')
            object.__setattr__(self, name, tuple(values))
        refs = [(v.style_id, v.revision) for v in self.styles]
        if len(refs) != len(set(refs)):
            raise FigureValidationError('Duplicate project style revision')
        if self.default_ref is not None and (self.default_ref.style_id, self.default_ref.revision) not in refs:
            raise FigureValidationError('Project default refers to a missing style revision')
        if self.default_ref is not None and self.default_ref.style_id in self.hidden_ids:
            raise FigureValidationError('Project default cannot be hidden')

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, Mapping):
            raise FigureValidationError('StyleRegistry requires an object')
        raw = dict(raw)
        if raw.get('default_ref') is not None:
            raw['default_ref'] = StyleRef.from_dict(raw['default_ref'])
        return super().from_dict(raw)

    def available(self):
        latest = {}
        for style in self.styles:
            if style.style_id not in self.hidden_ids and (
                style.style_id not in latest or latest[style.style_id].revision < style.revision):
                latest[style.style_id] = style
        return tuple(latest.values())

    def add_revision(self, style):
        if not isinstance(style, FigureStyle):
            raise FigureValidationError('Expected FigureStyle')
        existing = [s for s in self.styles if s.style_id == style.style_id]
        if any(s.provenance.get('builtin') for s in existing):
            raise FigureValidationError('Built-in styles cannot be overwritten; save a project style')
        if existing and style.revision != max(s.revision for s in existing) + 1:
            raise FigureValidationError('A style edit must append the next immutable revision')
        if not existing and style.revision != 1:
            raise FigureValidationError('A new style starts at revision 1')
        if any(s.name == style.name and s.style_id != style.style_id for s in self.available()):
            raise FigureValidationError('A different project style already uses this name')
        default = (style.ref if self.default_ref is not None and
                   self.default_ref.style_id == style.style_id else self.default_ref)
        return replace(self, styles=(*self.styles, style), default_ref=default,
                       hidden_ids=tuple(x for x in self.hidden_ids if x != style.style_id))

    def with_default(self, ref):
        if isinstance(ref, Mapping): ref = StyleRef.from_dict(ref)
        resolve_ref(self, ref)
        return replace(self, default_ref=ref)

    def hide(self, style_id):
        if self.default_ref is not None and self.default_ref.style_id == style_id:
            raise FigureValidationError('Choose another project default before deleting this style')
        if any(s.style_id == style_id and s.provenance.get('builtin') for s in self.styles):
            raise FigureValidationError('Built-in styles cannot be deleted')
        return replace(self, hidden_ids=tuple(dict.fromkeys((*self.hidden_ids, style_id))))

    def import_legacy_style(self, name, mapping, vocabulary='plot_appearance'):
        identity_payload = ({key: value for key, value in mapping.items() if key != 'app_version'}
                            if isinstance(mapping, Mapping) else mapping)
        identity = 'legacy.' + hashlib.sha256(canonical_json(
            {'vocabulary': vocabulary, 'payload': identity_payload}).encode()).hexdigest()
        existing = next((s for s in self.styles if s.style_id == identity), None)
        if existing is not None:
            return self, existing.ref
        conversion = import_legacy_look(mapping, vocabulary)
        if conversion.report['unreadable']:
            raise FigureValidationError('Legacy look contains unreadable properties; original retained')
        used = {s.name for s in self.available()}
        candidate, number = name, 2
        while candidate in used:
            candidate = f'{name} (imported {number})'; number += 1
        style = FigureStyle(identity, 1, candidate, conversion.look,
            provenance={'legacy_vocabulary': vocabulary, 'legacy_identity': identity},
            extra={'legacy_remainder': thaw_json(conversion.remainder),
                   'conversion_report': thaw_json(conversion.report)})
        registry = self.add_revision(style)
        return replace(registry, imported_legacy_ids=(*registry.imported_legacy_ids, identity)), style.ref


@dataclass(frozen=True)
class RegistryRead:
    registry: StyleRegistry | None
    warnings: tuple[str, ...] = ()
    opaque: object = None


def resolve_ref(registry, ref, snapshot=None):
    if isinstance(ref, Mapping): ref = StyleRef.from_dict(ref)
    if not isinstance(ref, StyleRef):
        raise FigureValidationError('A pinned style reference is required')
    found = next((s for s in registry.styles if s.ref.style_id == ref.style_id and
                  s.ref.revision == ref.revision), None) if registry is not None else None
    if found is not None:
        return found
    if snapshot is not None:
        style = snapshot if isinstance(snapshot, FigureStyle) else FigureStyle.from_dict(snapshot)
        if (style.style_id, style.revision) == (ref.style_id, ref.revision):
            return style
    raise FigureValidationError('Missing style revision and matching snapshot')


def registry_from_workspace(workspace_or_extra):
    extra = workspace_or_extra.extra if hasattr(workspace_or_extra, 'extra') else workspace_or_extra
    if not isinstance(extra, Mapping):
        raise FigureValidationError('Workspace extras must be an object')
    if WORKSPACE_KEY in extra:
        raw = extra[WORKSPACE_KEY]
        try:
            return RegistryRead(StyleRegistry.from_dict(raw))
        except (ValueError, TypeError) as exc:
            return RegistryRead(None, (f'Project styles could not be read: {exc}',),
                                thaw_json(FrozenMap({'raw': raw}))['raw'])
    styles = builtin_styles()
    registry = StyleRegistry(styles, styles[0].ref)
    legacy = extra.get('figure_workspace_v1')
    if isinstance(legacy, Mapping) and 'default_style' in legacy:
        try:
            registry, ref = registry.import_legacy_style('Imported project default',
                legacy['default_style'], 'plot_appearance')
            registry = registry.with_default(ref)
        except (ValueError, TypeError) as exc:
            return RegistryRead(None, (f'Project default could not be imported: {exc}',),
                                thaw_json(FrozenMap({'raw': legacy}))['raw'])
    return RegistryRead(registry)


def registry_to_workspace(workspace_or_extra, registry):
    """Return a detached replacement. An unreadable section cannot be overwritten."""
    extra = workspace_or_extra.extra if hasattr(workspace_or_extra, 'extra') else workspace_or_extra
    current = registry_from_workspace(extra)
    if current.registry is None:
        raise FigureValidationError('Cannot overwrite unreadable project styles')
    if not isinstance(registry, StyleRegistry):
        raise FigureValidationError('Expected StyleRegistry')
    cooked = thaw_json(FrozenMap(extra))
    cooked[WORKSPACE_KEY] = registry.to_dict()
    return replace(workspace_or_extra, extra=cooked) if hasattr(workspace_or_extra, 'extra') else cooked
