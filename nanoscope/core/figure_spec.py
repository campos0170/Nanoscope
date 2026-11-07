"""Immutable, Qt-free figure values. Native resources never enter the JSON model.

Unset, null, and omitted legacy fields are distinct. Codecs own their interpretation;
this module validates the wire types and identities without importing a renderer.
"""
from __future__ import annotations

from collections.abc import Mapping, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
import json
import math
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import UUID

SPEC_VERSION = 1
FIGURE_ARTIFACT_KIND = "nanoscope.figure"
FIGURE_ARTIFACT_SCHEMA = 1
FIGURE_KINDS = frozenset({"distributions", "distribution_summaries", "comparison_figures",
                          "event_waveforms", "noise", "iv", "voltage_series", "subpopulations"})


class FigureValidationError(ValueError):
    """A value cannot be represented by this figure schema without changing meaning."""


class FrozenMap(Mapping):
    """An immutable JSON object, recursively copied on ingress."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping | None = None):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise FigureValidationError("A JSON object requires a mapping")
        cooked = freeze_json(values)
        object.__setattr__(self, "_items", cooked._items)
        object.__setattr__(self, "_lookup", cooked._lookup)

    def __setattr__(self, name, value):
        raise AttributeError("FrozenMap is immutable")

    @classmethod
    def _make(cls, items):
        obj = object.__new__(cls)
        object.__setattr__(obj, "_items", tuple(items))
        object.__setattr__(obj, "_lookup", MappingProxyType(dict(obj._items)))
        return obj

    def __getitem__(self, key):
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lookup)

    def __len__(self):
        return len(self._lookup)

    def __hash__(self):
        return hash(self._items)

    def __repr__(self):
        return f"FrozenMap({dict(self._items)!r})"


def freeze_json(value: Any):
    """Validate strict finite JSON and copy it into immutable mappings/tuples."""
    active: set[int] = set()

    def visit(item, depth=0):
        if depth > 100:
            raise FigureValidationError("Figure JSON nesting exceeds 100 levels")
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise FigureValidationError("Figure JSON numbers must be finite")
            return item
        if isinstance(item, FrozenMap):
            return item
        if isinstance(item, Mapping) or type(item) in (list, tuple):
            identity = id(item)
            if identity in active:
                raise FigureValidationError("Figure JSON cannot contain a cycle")
            active.add(identity)
            try:
                if isinstance(item, Mapping):
                    if any(type(key) is not str for key in item):
                        raise FigureValidationError("Figure JSON object keys must be strings")
                    return FrozenMap._make((key, visit(item[key], depth + 1)) for key in sorted(item))
                return tuple(visit(child, depth + 1) for child in item)
            finally:
                active.remove(identity)
        raise FigureValidationError(f"Not a JSON value: {type(item).__name__}")

    return visit(value)


def thaw_json(value):
    """Return an independent JSON-native copy of a frozen value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(thaw_json(freeze_json(value)), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def parse_json(raw: str):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise FigureValidationError(f"Duplicate JSON key: {key!r}")
            out[key] = value
        return out

    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               FigureValidationError(f"Nonfinite JSON number: {value}")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise FigureValidationError(f"Invalid figure JSON: {exc}") from exc
    return thaw_json(freeze_json(value))


@dataclass(frozen=True)
class _Node:
    extra: FrozenMap = field(default_factory=FrozenMap, kw_only=True)
    _unknown_fields: FrozenMap = field(default_factory=FrozenMap, kw_only=True, repr=False)
    _strings: ClassVar[tuple[str, ...]] = ()
    _maps: ClassVar[tuple[str, ...]] = ()
    _nodes: ClassVar[dict[str, type]] = {}
    _sequences: ClassVar[dict[str, type]] = {}

    def __post_init__(self):
        object.__setattr__(self, "extra", FrozenMap(self.extra))
        object.__setattr__(self, "_unknown_fields", FrozenMap(self._unknown_fields))
        for name in self._strings:
            if type(getattr(self, name)) is not str:
                raise FigureValidationError(f"{type(self).__name__}.{name} must be text")
        for name in self._maps:
            object.__setattr__(self, name, FrozenMap(getattr(self, name)))
        for name, cls in self._nodes.items():
            if not isinstance(getattr(self, name), cls):
                raise FigureValidationError(f"{type(self).__name__}.{name} must be {cls.__name__}")
        for name, cls in self._sequences.items():
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or not all(isinstance(v, cls) for v in value):
                raise FigureValidationError(f"{type(self).__name__}.{name} has invalid members")
            object.__setattr__(self, name, tuple(value))

    def to_dict(self):
        out = {}
        for f in fields(self):
            if f.name == "_unknown_fields":
                continue
            value = getattr(self, f.name)
            if isinstance(value, _Node):
                out[f.name] = value.to_dict()
            elif f.name in self._sequences:
                out[f.name] = [v.to_dict() if isinstance(v, _Node) else v for v in value]
            else:
                out[f.name] = thaw_json(value)
        # Unknown future fields stay at their original node level. Explicit extra
        # remains its own container; neither may shadow fields this schema owns.
        unknown = thaw_json(self._unknown_fields)
        if any(key in out for key in unknown):
            raise FigureValidationError("Unknown figure fields shadow canonical fields")
        out.update(unknown)
        return out

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, Mapping):
            raise FigureValidationError(f"{cls.__name__} requires an object")
        names = {f.name for f in fields(cls)} - {"_unknown_fields"}
        extra = thaw_json(FrozenMap(raw.get("extra", {})))
        unknown = {key: value for key, value in raw.items() if key not in names}
        args = {key: value for key, value in raw.items() if key in names and key != "extra"}
        args["extra"] = extra
        args["_unknown_fields"] = unknown
        for key, node in cls._nodes.items():
            if key in args:
                args[key] = node.from_dict(args[key])
        for key, node in cls._sequences.items():
            if key in args and issubclass(node, _Node):
                if not isinstance(args[key], (tuple, list)):
                    raise FigureValidationError(f"{key} requires an array")
                args[key] = tuple(node.from_dict(v) for v in args[key])
        try:
            return cls(**args)
        except TypeError as exc:
            raise FigureValidationError(f"Invalid {cls.__name__}: {exc}") from exc


@dataclass(frozen=True)
class FigureRecipe(_Node):
    codec: str = ""
    codec_version: int = 1
    draft: FrozenMap = field(default_factory=FrozenMap)
    applied: FrozenMap = field(default_factory=FrozenMap)
    quantity_identity: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("codec",)
    _maps = ("draft", "applied", "quantity_identity")

    def __post_init__(self):
        super().__post_init__()
        if type(self.codec_version) is not int or self.codec_version != 1:
            raise FigureValidationError("Unsupported figure recipe codec version")


@dataclass(frozen=True)
class SourceRef(_Node):
    source_id: str = ""
    recipe: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("source_id",)
    _maps = ("recipe",)


@dataclass(frozen=True)
class AxisPinSpec(_Node):
    # Native values are preserved in values, including omitted-vs-null and typed
    # range strings. A codec declares units rather than guessing from the label.
    values: FrozenMap = field(default_factory=FrozenMap)
    unit_basis: str = "display"
    units: str = ""
    _strings = ("unit_basis", "units")
    _maps = ("values",)

    def __post_init__(self):
        super().__post_init__()
        if self.unit_basis not in ("data", "display", "mixed"):
            raise FigureValidationError("Axis pin unit_basis must be data, display or mixed")
        if self.unit_basis == "mixed":
            basis = self.extra.get("value_unit_basis")
            if (not isinstance(basis, Mapping) or set(basis) != set(self.values)
                    or any(v not in ("data", "display") for v in basis.values())):
                raise FigureValidationError("Mixed axis pins require each value's unit basis")


@dataclass(frozen=True)
class AxisSpec(_Node):
    axis_id: str = ""
    view_id: str = ""
    side: str = ""
    label_text_id: str = ""
    scale: str = "linear"
    pin: AxisPinSpec = field(default_factory=AxisPinSpec)
    automatic: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("axis_id", "view_id", "side", "label_text_id", "scale")
    _maps = ("automatic",)
    _nodes = {"pin": AxisPinSpec}


@dataclass(frozen=True)
class SeriesSpec(_Node):
    series_id: str = ""
    axes_ids: tuple[str, ...] = ()
    role: str = "data"
    primitive: str = ""
    source_ids: tuple[str, ...] = ()
    data: FrozenMap = field(default_factory=FrozenMap)
    label_text_id: str = ""
    visible: bool = True
    overrides: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("series_id", "role", "primitive", "label_text_id")
    _maps = ("data", "overrides")
    _sequences = {"axes_ids": str, "source_ids": str}

    def __post_init__(self):
        super().__post_init__()
        if type(self.visible) is not bool:
            raise FigureValidationError("Series visibility must be boolean")


@dataclass(frozen=True)
class TextWords(_Node):
    mode: str = "automatic"
    value: Any = None
    _strings = ("mode",)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "value", freeze_json(self.value))
        if self.mode not in ("automatic", "hidden", "literal", "legacy"):
            raise FigureValidationError("Invalid text-word mode")
        if self.mode == "literal" and type(self.value) is not str:
            raise FigureValidationError("Literal words must be text")


@dataclass(frozen=True)
class TextSpec(_Node):
    element_id: str = ""
    words: TextWords = field(default_factory=TextWords)
    overrides: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("element_id",)
    _maps = ("overrides",)
    _nodes = {"words": TextWords}


@dataclass(frozen=True)
class AnnotationSpec(_Node):
    element_id: str = ""
    role: str = ""
    text_id: str = ""
    visible: bool = True
    position: FrozenMap = field(default_factory=FrozenMap)
    options: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("element_id", "role", "text_id")
    _maps = ("position", "options")

    def __post_init__(self):
        super().__post_init__()
        if type(self.visible) is not bool:
            raise FigureValidationError("Annotation visibility must be boolean")
        frame = self.position.get("coordinate_frame")
        if frame is not None and frame not in ("viewport_pixels", "plot_fraction", "axes_fraction"):
            raise FigureValidationError("Unknown annotation coordinate frame")


@dataclass(frozen=True)
class FigurePresentation(_Node):
    style_ref: Any = None
    style_snapshot: FrozenMap = field(default_factory=FrozenMap)
    overrides: FrozenMap = field(default_factory=FrozenMap)
    page_extras: FrozenMap = field(default_factory=FrozenMap)
    _maps = ("style_snapshot", "overrides", "page_extras")

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "style_ref", freeze_json(self.style_ref))


@dataclass(frozen=True)
class FigureView(_Node):
    active_view: str = ""
    ranges: FrozenMap = field(default_factory=FrozenMap)
    canvas_view: FrozenMap = field(default_factory=FrozenMap)
    page_layout: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("active_view",)
    _maps = ("ranges", "canvas_view", "page_layout")


@dataclass(frozen=True)
class ResultState(_Node):
    native_kind: str = ""
    native_schema: int = 1
    payload: FrozenMap = field(default_factory=FrozenMap)
    table_refs: tuple[str, ...] = ()
    blob_refs: tuple[str, ...] = ()
    status: str = "frozen"
    _strings = ("native_kind", "status")
    _maps = ("payload",)
    _sequences = {"table_refs": str, "blob_refs": str}

    def __post_init__(self):
        super().__post_init__()
        if type(self.native_schema) is not int or self.native_schema < 1:
            raise FigureValidationError("Native schema must be a positive integer")


@dataclass(frozen=True)
class FigureSpec(_Node):
    figure_id: str = ""
    kind: str = ""
    name: str = ""
    created_at: str = ""
    spec_version: int = SPEC_VERSION
    recipe: FigureRecipe = field(default_factory=FigureRecipe)
    sources: tuple[SourceRef, ...] = ()
    axes: tuple[AxisSpec, ...] = ()
    series: tuple[SeriesSpec, ...] = ()
    texts: tuple[TextSpec, ...] = ()
    annotations: tuple[AnnotationSpec, ...] = ()
    presentation: FigurePresentation = field(default_factory=FigurePresentation)
    view: FigureView = field(default_factory=FigureView)
    result: ResultState = field(default_factory=ResultState)
    lineage: FrozenMap = field(default_factory=FrozenMap)
    _strings = ("figure_id", "kind", "name", "created_at")
    _maps = ("lineage",)
    _nodes = {"recipe": FigureRecipe, "presentation": FigurePresentation,
              "view": FigureView, "result": ResultState}
    _sequences = {"sources": SourceRef, "axes": AxisSpec, "series": SeriesSpec,
                  "texts": TextSpec, "annotations": AnnotationSpec}

    def __post_init__(self):
        super().__post_init__()
        if type(self.spec_version) is not int or self.spec_version != SPEC_VERSION:
            raise FigureValidationError("Unsupported FigureSpec version")
        try:
            parsed = UUID(self.figure_id.removeprefix("live."))
        except ValueError as exc:
            raise FigureValidationError("Figure identity requires a UUID") from exc
        if self.figure_id not in (str(parsed), "live." + str(parsed)):
            raise FigureValidationError("Figure identity must use a canonical UUID")
        if self.kind not in FIGURE_KINDS or not self.name.strip():
            raise FigureValidationError("Figure kind and nonempty name are required")
        for sequence, identity in ((self.sources, "source_id"), (self.axes, "axis_id"),
                                   (self.series, "series_id"), (self.texts, "element_id"),
                                   (self.annotations, "element_id")):
            ids = [getattr(node, identity) for node in sequence]
            if any(not key for key in ids) or len(ids) != len(set(ids)):
                raise FigureValidationError(f"Duplicate or empty {identity}")
        source_ids, axis_ids = {s.source_id for s in self.sources}, {a.axis_id for a in self.axes}
        for series in self.series:
            if not set(series.source_ids) <= source_ids or not set(series.axes_ids) <= axis_ids:
                raise FigureValidationError("Series refers to an unknown source or axis")

    @property
    def artifact_id(self):
        return self.figure_id if self.figure_id.startswith("live.") else "live." + self.figure_id

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, raw: str):
        return cls.from_dict(parse_json(raw))

    def renamed(self, name: str):
        return replace(self, name=name)


class ResourceSet:
    """Detached resources, copied on input and output; JSON holds logical names only."""

    __slots__ = ("_tables", "_blobs")

    def __init__(self, tables: Mapping | None = None, blobs: Mapping | None = None):
        from pandas import DataFrame

        tables, blobs = tables or {}, blobs or {}
        if not isinstance(tables, Mapping) or not isinstance(blobs, Mapping):
            raise FigureValidationError("Figure resources require mappings")
        if any(type(k) is not str or not k for k in (*tables, *blobs)):
            raise FigureValidationError("Resource names must be nonempty strings")
        if any(not isinstance(value, DataFrame) for value in tables.values()):
            raise FigureValidationError("Figure tables must be DataFrames")
        if any(type(value) is not bytes for value in blobs.values()):
            raise FigureValidationError("Figure blobs must be bytes")
        self._tables = MappingProxyType({key: self._copy_table(value) for key, value in tables.items()})
        self._blobs = MappingProxyType(dict(blobs))

    @staticmethod
    def _copy_table(value):
        # pandas deep=True leaves mutable Python objects inside object columns
        # shared. A figure resource must also detach those cells and metadata.
        copied = value.copy(deep=True)
        for column, dtype in enumerate(value.dtypes):
            if dtype == object:
                for row in range(len(value)):
                    copied.iat[row, column] = deepcopy(value.iat[row, column])
        copied.attrs = deepcopy(value.attrs)
        return copied

    @property
    def tables(self):
        return {key: self._copy_table(value) for key, value in self._tables.items()}

    @property
    def blobs(self):
        return dict(self._blobs)

    @property
    def table_names(self):
        return tuple(self._tables)

    @property
    def blob_names(self):
        return tuple(self._blobs)

    def validate(self, spec: FigureSpec):
        if (len(set(spec.result.table_refs)) != len(spec.result.table_refs)
                or len(set(spec.result.blob_refs)) != len(spec.result.blob_refs)):
            raise FigureValidationError("Duplicate figure resource references")
        if set(spec.result.table_refs) != set(self._tables) or set(spec.result.blob_refs) != set(self._blobs):
            raise FigureValidationError("Figure resource references do not match its resources")


@dataclass(frozen=True)
class FigureRecord:
    spec: FigureSpec
    resources: ResourceSet = field(default_factory=ResourceSet)
    extra: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        object.__setattr__(self, "extra", FrozenMap(self.extra))
        if not isinstance(self.spec, FigureSpec) or not isinstance(self.resources, ResourceSet):
            raise FigureValidationError("FigureRecord requires a spec and resource set")
        self.resources.validate(self.spec)
