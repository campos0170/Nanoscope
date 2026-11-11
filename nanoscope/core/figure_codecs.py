"""Lossless native-page codecs for the eight v12 analytical figure kinds.

Every claimed native path has one canonical owner and an inverse binding. Unknown
leaves remain in ResultState.payload. The bindings contain paths, never stale values;
editing a canonical node is therefore the only way to edit its persisted native value.
No renderer, widget, analysis kernel or dynamic type import is used here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
from typing import Any

from .figure_spec import (FigureSpec, FigureRecipe, FigurePresentation, FigureView,
                          SourceRef, AxisSpec, AxisPinSpec, SeriesSpec, TextSpec, TextWords,
                          AnnotationSpec, ResultState, ResourceSet, FigureRecord, FrozenMap,
                          FigureValidationError, canonical_json, freeze_json, thaw_json)

TAG = "__nanoscope_workspace_value__"
CODEC_VERSION = 1
# Explicit registry: an unrelated artifact that happens to contain appearance is opaque.
NATIVE_KINDS = {
    "builtin.plots": ("nanoscope.plots", "distributions", "Distributions"),
    "builtin.distribution-summaries": ("nanoscope.distribution-summaries", "distribution_summaries", "Distribution summaries"),
    "builtin.comparison-figures": ("nanoscope.comparison-figures", "comparison_figures", "Comparison figures"),
    "builtin.event-plots": ("nanoscope.event-plots", "event_waveforms", "Event waveforms"),
    "builtin.psd": ("nanoscope.psd", "noise", "Noise"),
    "builtin.iv": ("nanoscope.iv", "iv", "I–V"),
    "builtin.voltage-dependence": ("nanoscope.voltage-dependence", "voltage_series", "Voltage series"),
    "builtin.subpopulation": ("nanoscope.subpopulation-report", "subpopulations", "Subpopulations"),
}
APPEARANCE_FIELDS = frozenset({"legend_visible", "legend_size", "font_family", "title_size",
    "axis_size", "tick_size", "grid", "background", "data_color", "fit_color", "line_width", "marker_size", "frame"})
STYLE_FIELDS = frozenset({"show_legend", "font_family", "font_size_pt", "data_color", "fit_color",
    "point_size", "line_width", "background", "export_w", "export_h", "legend_font_size_pt",
    "legend_counts", "legend_include", "show_grid", "title_font_size_pt", "axis_label_font_size_pt",
    "tick_font_size_pt", "show_ci_band", "show_provenance", "show_residuals", "heatmap_colormap", "frame"})
WORD_FIELDS = {"title": "title", "x_label": "axis.bottom.label", "y_label": "axis.left.label"}
_MISSING = object()
_NO_DEFAULT = object()


def _copy(value):
    return thaw_json(freeze_json(value))


def _get(root, path, default=_NO_DEFAULT):
    try:
        for key in path:
            root = root[key]
        return root
    except (KeyError, IndexError, TypeError):
        if default is not _NO_DEFAULT:
            return default
        raise FigureValidationError(f"Missing codec path: {path!r}") from None


def _put(root, path, value):
    parent = root
    for key, following in zip(path[:-1], path[1:]):
        if isinstance(parent, dict) and key not in parent:
            parent[key] = [] if type(following) is int else {}
        parent = parent[key]
    if isinstance(parent, list):
        if type(path[-1]) is not int or not 0 <= path[-1] < len(parent):
            raise FigureValidationError("Invalid native list binding")
    parent[path[-1]] = value


def _remove(root, path):
    parent = _get(root, path[:-1])
    if isinstance(parent, list):
        parent[path[-1]] = None  # A structural hole, named explicitly by the binding.
    else:
        del parent[path[-1]]


def leaf_paths(value, path=()):
    """All original scalar or empty-container paths, for exhaustive migration accounting."""
    if isinstance(value, Mapping) and value:
        return tuple(p for key, child in value.items() for p in leaf_paths(child, path + (key,)))
    if isinstance(value, (list, tuple)) and value:
        return tuple(p for index, child in enumerate(value) for p in leaf_paths(child, path + (index,)))
    return (path,)


def _words(value):
    if value is None or value == "":
        return TextWords("automatic", value)
    if type(value) is str:
        return TextWords("hidden" if not value.strip() else "literal", value)
    return TextWords("legacy", value)


def _pointer(path):
    return "/" + "/".join(str(k).replace("~", "~0").replace("/", "~1") for k in path)


@dataclass(frozen=True)
class MigrationReport:
    artifact_id: str
    claimed_paths: tuple[tuple, ...] = ()
    carried_paths: tuple[tuple, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecodedFigure:
    spec: FigureSpec
    resources: ResourceSet
    report: MigrationReport

    @property
    def record(self):
        return FigureRecord(self.spec, self.resources)


class _Builder:
    def __init__(self, payload):
        self.original = _copy(payload)
        self.remainder = _copy(payload)
        self.bindings = []
        self.texts, self.axes, self.annotations, self.sources, self.series = [], [], [], [], []
        self.text_indices = {}
        self.overrides, self.page_extras = {}, {}
        self.style_binding = None
        self.style_vocabulary = "plot_appearance"
        self.draft, self.applied, self.quantity = {}, {}, {}
        self.view = {"ranges": {}, "canvas_view": {}, "page_layout": {}}

    def has(self, path):
        return _get(self.remainder, path, _MISSING) is not _MISSING

    def take(self, path, target, *, targets=None, transform="direct"):
        # A claimed parent and child cannot coexist: that would create two owners.
        path = tuple(path)
        for binding in self.bindings:
            old = tuple(binding["native"])
            if path[:len(old)] == old or old[:len(path)] == path:
                raise FigureValidationError("Overlapping native claims")
        value = _get(self.remainder, path)
        self.bindings.append({"native": list(path), "target": list(target), "transform": transform,
                              **({"targets": targets} if targets is not None else {})})
        _remove(self.remainder, path)
        return value

    def map_field(self, path, section, key):
        if not self.has(path):
            return
        owner = {"draft": self.draft, "applied": self.applied, "quantity_identity": self.quantity,
                 "overrides": self.overrides, "page_extras": self.page_extras}[section]
        family = "recipe" if section in ("draft", "applied", "quantity_identity") else "presentation"
        owner[key] = self.take(path, (family, section, key))

    def text(self, path, element_id, *, style=False):
        if not self.has(path):
            return
        if element_id not in self.text_indices:
            self.text_indices[element_id] = len(self.texts)
            self.texts.append({"element_id": element_id, "words": TextWords().to_dict(), "overrides": {}})
        index = self.text_indices[element_id]
        if style:
            value = _get(self.remainder, path)
            for key in list(value):
                self.texts[index]["overrides"][key] = self.take(path + (key,), ("texts", index, "overrides", key))
        else:
            value = self.take(path, ("texts", index, "words", "value"))
            self.texts[index]["words"] = _words(value).to_dict()

    def quantity_display(self, path, *, namespace=""):
        witness = _get(self.remainder, path, None)
        if not isinstance(witness, dict) or type(witness.get("version")) is not int or witness["version"] != 1:
            return  # Unknown future witnesses stay opaque and lossless in the remainder.
        from .quantity_display import AxisDisplayBasis
        declared = witness.get("axes", {})
        if not isinstance(declared, dict):
            raise FigureValidationError("Display quantity axes must be an object")
        for key, raw in list(declared.items()):
            basis = AxisDisplayBasis.from_dict(raw)
            axis_id = namespace + key
            index = next((i for i, axis in enumerate(self.axes) if axis.axis_id == axis_id), len(self.axes))
            if index == len(self.axes):
                view, _, side = key.rpartition(".")
                self.axes.append(AxisSpec(axis_id, view_id=view, side=side or key))
            axis = self.axes[index]
            claimed = self.take(path + ("axes", key), ("axes", index, "extra", "display_basis"))
            self.axes[index] = replace(axis, extra={**thaw_json(axis.extra), "display_basis": claimed},
                                      pin=replace(axis.pin, units=basis.quantity.display_unit))
        metadata = self.view.setdefault("extra", {}).setdefault("quantity_display_metadata", {})
        for key in list(witness):
            if key != "axes":
                metadata[key] = self.take(path + (key,), ("view", "extra", "quantity_display_metadata", key))

    def appearance(self, path, *, legacy_style=False, namespace=""):
        appearance = _get(self.remainder, path, None)
        if not isinstance(appearance, dict):
            return
        bound = appearance.get("style_binding")
        if bound:
            from .figure_style import normalize_style_binding
            from .figure_style_mapping import import_legacy_look
            if self.style_binding is not None:
                raise FigureValidationError("A figure cannot have two independent style bindings")
            self.style_binding = normalize_style_binding(bound)
            self.overrides.update(self.style_binding["overrides"])
            self.take(path + ("style_binding",), ("presentation",), transform="style_binding")
            vocabulary = "style_settings" if legacy_style else self.style_vocabulary
            projected = import_legacy_look(appearance, vocabulary).report["claimed"]
            for field in projected:
                self.take(path + (field,), ("presentation",), transform="style_field")
                self.bindings[-1].update(vocabulary=vocabulary, field=field)
        series_styles = _get(self.remainder, path + ("series_styles",), None)
        if isinstance(series_styles, dict):
            from .figure_series import validate_series_styles, stable_series_id
            validate_series_styles(series_styles)
            for key, style in list(series_styles.items()):
                index = len(self.series)
                overrides = self.take(path + ("series_styles", key), ("series", index, "overrides"))
                self.series.append(SeriesSpec(stable_series_id(namespace, "presentation", key),
                    role="presentation", primitive="style-target",
                    data={"native_series_id": key}, overrides=overrides))
        for key, element in WORD_FIELDS.items():
            self.text(path + (key,), namespace + element)
        style_key = "text_styles" if legacy_style else "text"
        styles = _get(self.remainder, path + (style_key,), None)
        if isinstance(styles, dict):
            for key in list(styles):
                value = styles[key]
                if isinstance(value, dict):
                    # Text in a per-element style is itself words, except a legacy
                    # duplicate title field: retain that duplicate as opaque data.
                    element = namespace + key
                    if "text" in value and element not in self.text_indices:
                        self.text(path + (style_key, key, "text"), element)
                    self.text(path + (style_key, key), element, style=True)
        for key in sorted(STYLE_FIELDS if legacy_style else APPEARANCE_FIELDS):
            self.map_field(path + (key,), "page_extras" if self.style_binding is not None else "overrides", namespace + key)
        pins = _get(self.remainder, path + ("axes",), None)
        if isinstance(pins, dict):
            for key, pin in list(pins.items()):
                if not isinstance(pin, dict):
                    continue
                index = len(self.axes)
                value = self.take(path + ("axes", key), ("axes", index, "pin", "values"))
                view, _, side = key.rpartition(".")
                self.axes.append(AxisSpec(namespace + key, view_id=view, side=side or key,
                                          pin=AxisPinSpec(value, "display")))
        self.quantity_display(path + ("quantity_display",), namespace=namespace)
        annotations = _get(self.remainder, path + ("annotations",), None)
        if isinstance(annotations, dict):
            for key, value in list(annotations.items()):
                if not isinstance(value, dict):
                    continue
                index = len(self.annotations)
                position = {"coordinate_frame": "viewport_pixels"}
                if "offset" in value:
                    position["offset"] = self.take(path + ("annotations", key, "offset"),
                                                   ("annotations", index, "position", "offset"))
                options = {}
                for option in list(value):
                    options[option] = self.take(path + ("annotations", key, option), ("annotations", index, "options", option))
                self.annotations.append(AnnotationSpec(namespace + key, role="annotation", position=position, options=options))
        fit = _get(self.remainder, path + ("fit_label",), None)
        if isinstance(fit, dict):
            index = len(self.annotations)
            # One options map owns the complete native fit-label vocabulary. Its
            # template remains here, not duplicated in TextSpec.
            value = self.take(path + ("fit_label",), ("annotations", index, "options"))
            self.annotations.append(AnnotationSpec(namespace + "fit.label", role="fit-label",
                position={"coordinate_frame": "viewport_pixels"}, options=value))
        if legacy_style:
            for key in ("fit_values_on_plot", "show_r2", "show_gof", "title_bold", "title_italic"):
                self.map_field(path + (key,), "page_extras", namespace + "legacy_style." + key)
            labels = _get(self.remainder, path + ("label_overrides",), None)
            if isinstance(labels, dict):
                for key in list(labels):
                    self.text(path + ("label_overrides", key), "legend.entry." + key)
            if self.has(path + ("legend_pos",)):
                index = len(self.annotations)
                offset = self.take(path + ("legend_pos",), ("annotations", index, "position", "offset"))
                self.annotations.append(AnnotationSpec("legend", role="legend",
                    position={"coordinate_frame": "plot_fraction", "offset": offset}))

    def source(self, path):
        value = _get(self.remainder, path, None)
        if not isinstance(value, (dict, list)):
            return
        recipes = value if isinstance(value, list) else [value]
        if any(not isinstance(v, dict) for v in recipes):
            return
        targets = []
        for recipe in recipes:
            identity = {k: recipe[k] for k in ("kind", "source_key", "channel", "channel_name", "channel_kind", "module_id") if k in recipe}
            # The persistent recipe, never a window/session doc_id or display name.
            digest = hashlib.sha256(canonical_json(identity or recipe).encode()).hexdigest()[:20]
            source_id = "source." + digest
            existing = {s.source_id for s in self.sources}
            suffix = 1
            while source_id in existing:
                suffix += 1
                source_id = "source." + digest + "." + str(suffix)
            targets.append(["sources", len(self.sources), "recipe"])
            self.sources.append(SourceRef(source_id, recipe))
        self.take(path, (), targets=targets, transform="list" if isinstance(value, list) else "single")

    def data(self, path, *, primitive="native", role="data", source_ids=()):
        if not self.has(path):
            return
        index = len(self.series)
        value = self.take(path, ("series", index, "data", "native"))
        self.series.append(SeriesSpec("series." + _pointer(path), role=role,
            primitive=primitive, source_ids=source_ids, data={"native": value}))

    def iv_points(self, path, *, role):
        points = _get(self.remainder, path, None)
        if not isinstance(points, list) or any(not isinstance(v, dict) for v in points):
            return
        identity_keys = ("source_key", "channel", "channel_name", "channel_kind")
        series_index = len(self.series)
        data, sources = [], []
        for row, point in enumerate(points):
            recipe = {key: point[key] for key in identity_keys if key in point}
            source_id = "source." + hashlib.sha256(canonical_json(recipe).encode()).hexdigest()[:20]
            source_id += "." + role + "." + str(row)
            source_index = len(self.sources)
            for key in recipe:
                self.take(path + (row, key), ("sources", source_index, "recipe", key))
            self.sources.append(SourceRef(source_id, recipe))
            sources.append(source_id)
            coordinates = {}
            for key in list(point):
                coordinates[key] = self.take(path + (row, key),
                    ("series", series_index, "data", "points", row, key))
            data.append(coordinates)
        self.series.append(SeriesSpec("iv." + role, role=role, primitive="points",
                                     source_ids=tuple(sources), data={"points": data}))

    def view_field(self, path, section, key):
        if self.has(path):
            self.view[section][key] = self.take(path, ("view", section, key))

    def resources(self, table_names, blob_names):
        def visit(value, path):
            if isinstance(value, dict):
                tag = value.get(TAG)
                ref = value.get("table") if tag in ("dataframe", "series", "ndarray", "index", "multi-index") else value.get("blob") if tag == "bytes" else None
                if ref is not None:
                    expected = blob_names if tag == "bytes" else table_names
                    if type(ref) is not str or ref not in expected:
                        raise FigureValidationError(f"Missing native resource at {_pointer(path)}")
                    index = len(self.series)
                    native = self.take(path, ("series", index, "data", "native"))
                    self.series.append(SeriesSpec("resource." + _pointer(path), role="frozen", primitive=tag, data={"native": native}))
                    return
                for key, child in list(value.items()):
                    visit(child, path + (key,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, path + (index,))
        visit(self.remainder, ())
        resources = _get(self.remainder, ("table_resources",), None)
        if isinstance(resources, dict):
            for key, ref in list(resources.items()):
                if type(ref) is not str or ref not in blob_names:
                    raise FigureValidationError("Missing figure table blob")
                index = len(self.series)
                native = self.take(("table_resources", key), ("series", index, "data", "blob"))
                self.series.append(SeriesSpec("table." + key, role="frozen", primitive="figure-table", data={"blob": native}))
        for section in ("cluster", "pca"):
            group = _get(self.remainder, (section,), None)
            if isinstance(group, dict):
                for key in ("labels_table", "means_table", "scores_table", "loadings_table"):
                    if key not in group:
                        continue
                    ref = group[key]
                    if ref not in table_names:
                        raise FigureValidationError("Missing Distributions result table")
                    index = len(self.series)
                    native = self.take((section, key), ("series", index, "data", "table"))
                    self.series.append(SeriesSpec(section + "." + key, role="frozen", primitive="table", data={"table": native}))


def _claim_kind(b, kind):
    if kind == "distributions":
        config = _get(b.remainder, ("config",), None)
        if not isinstance(config, dict) or config.get(TAG) != "dataclass" or not isinstance(config.get("fields"), dict):
            raise FigureValidationError("Distributions requires the tagged v12 PlotConfig envelope")
        base = ("config", "fields")
        style = base + ("style", "fields")
        b.appearance(style, legacy_style=True)
        axes = _get(b.remainder, base + ("axes", "fields"), None)
        if isinstance(axes, dict):
            for side, prefix in (("bottom", "x"), ("left", "y")):
                index = len(b.axes)
                values = {}
                for key in (prefix + "_min", prefix + "_max", "manual_" + prefix):
                    if b.has(base + ("axes", "fields", key)):
                        values[key] = b.take(base + ("axes", "fields", key), ("axes", index, "pin", "values", key))
                if values:
                    b.axes.append(AxisSpec(side, side=side, pin=AxisPinSpec(values, "mixed", extra={
                        "value_unit_basis": {key: "data" if key.startswith("manual_") else "display" for key in values},
                        "display_time_unit": str(axes.get("time_unit", ""))})))
            b.quantity_display(base + ("axes", "fields", "quantity_display"))
            b.map_field(base + ("axes", "fields", "time_unit"), "quantity_identity", "time_unit")
        # Keep each tagged section envelope in the native remainder. Claim its
        # individual fields so unknown future fields remain at their original path.
        sections = ("selection", "histogram", "scatter", "kde", "heatmap", "cluster", "fit")
        for section in sections:
            group = _get(b.remainder, base + (section, "fields"), None)
            if isinstance(group, dict):
                for key in list(group):
                    target = "applied" if section == "fit" and key.startswith("applied") else "draft"
                    b.map_field(base + (section, "fields", key), target, section + "." + key)
        for key in ("plot_type", "norm_user_set"):
            b.map_field(base + (key,), "draft", key)
        b.source(("selected_sources",))
        b.map_field(("phase2_render", "applied_config"), "applied", "plot_config")
        b.source(("phase2_render", "drawn_sources"))
        nodes = _get(b.remainder, ("phase2_render", "nodes"), None)
        if isinstance(nodes, list):
            for index in range(len(nodes)):
                b.data(("phase2_render", "nodes", index), primitive="sampled-curve")
        # v12 did not freeze ordinary histogram/scatter coordinates. These series
        # describe their durable source + recipe reference without inventing data.
        for source in b.sources:
            b.series.append(SeriesSpec("source-series." + source.source_id, source_ids=(source.source_id,),
                primitive="source-recipe", data={"recipe_ref": ["recipe", "draft"]}))
    elif kind in ("distribution_summaries", "comparison_figures"):
        b.appearance(("appearance",))
        for section in ("draft", "applied"):
            value = _get(b.remainder, (section,), None)
            if isinstance(value, dict):
                for key in list(value):
                    b.map_field((section, key), section, key)
        b.source(("source_recipes",))
        source_styles = _get(b.remainder, ("source_appearance",), None)
        if isinstance(source_styles, dict):
            for source, values in list(source_styles.items()):
                if isinstance(values, dict):
                    b.text(("source_appearance", source, "label"), "legend.entry." + source)
                    b.map_field(("source_appearance", source, "color"), "page_extras", "source_color." + source)
        b.view_field(("view_range",), "ranges", "primary")
        b.view_field(("canvas_view",), "canvas_view", "native")
        group = "primitives" if kind == "distribution_summaries" else "panels"
        primitives = _get(b.remainder, ("snapshot", group), None)
        if isinstance(primitives, list):
            for index in range(len(primitives)):
                b.data(("snapshot", group, index), primitive="summary-primitive" if group == "primitives" else "comparison-panel")
    elif kind == "iv":
        b.appearance(("appearance",))
        b.iv_points(("points",), role="data")
        b.iv_points(("exclusions",), role="excluded")
    elif kind == "subpopulations":
        b.appearance(("appearance",))
        b.source(("source",))
        b.view_field(("view",), "page_layout", "native")
        for key in ("accepted_event_ids",):
            b.map_field((key,), "applied", key)
    else:
        b.appearance(("config", "appearance"))
        if kind == "noise":
            b.appearance(("config", "style"), legacy_style=True, namespace="psd.")
            b.source(("config", "target_source"))
            b.source(("config", "compare_sources"))
            curves = _get(b.remainder, ("curves",), None)
            if isinstance(curves, list):
                for index in range(len(curves)):
                    b.data(("curves", index), primitive="psd-curve")
        elif kind == "event_waveforms":
            b.source(("source",))
            for key, element in (("density_title", "density.title"), ("density_x_label", "density.axis.bottom.label"),
                                 ("density_y_label", "density.axis.left.label"), ("ensemble_title", "ensemble.title")):
                b.text(("config", key), element)
        elif kind == "voltage_series":
            b.source(("point_sources",))
            b.source(("series_excluded",))
            b.data(("series", "points"), primitive="voltage-points")
        config = _get(b.remainder, ("config",), None)
        if isinstance(config, dict):
            # Appearance/style have already been split. Unknown keys in those maps
            # stay opaque instead of being copied into a second appearance owner.
            for key in list(config):
                if key in ("appearance", "style"):
                    continue
                if key == "view":
                    b.view_field(("config", key), "page_layout", "active_view")
                else:
                    b.map_field(("config", key), "applied" if key == "applied_fit" else "draft", key)


def _validate_native_resources(payload, tables, blobs):
    def visit(value):
        if isinstance(value, Mapping):
            tag = value.get(TAG)
            field = "table" if tag in ("dataframe", "series", "ndarray", "index", "multi-index") else "blob" if tag == "bytes" else None
            if field is not None:
                ref = value.get(field)
                if type(ref) is not str or ref not in (tables if field == "table" else blobs):
                    raise FigureValidationError("Missing tagged native resource")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(payload)
    for curve in payload.get("curves", ()):
        if isinstance(curve, Mapping) and (type(curve.get("table")) is not str or curve["table"] not in tables):
            raise FigureValidationError("Missing PSD curve table")


def _native_style_path(kind):
    if kind == "distributions":
        return ("config", "fields", "style", "fields"), "style_settings"
    if kind in ("event_waveforms", "noise", "voltage_series"):
        return ("config", "appearance"), "plot_appearance"
    return ("appearance",), "matplotlib_appearance" if kind == "comparison_figures" else "plot_appearance"


def _binding_from_presentation(presentation):
    from .figure_style import normalize_style_binding
    metadata = thaw_json(presentation.extra.get("style_binding_metadata", {"version": 1}))
    return normalize_style_binding({**metadata, "ref": thaw_json(presentation.style_ref),
        "snapshot": thaw_json(presentation.style_snapshot), "overrides": thaw_json(presentation.overrides)})


def decode_legacy(artifact, *, figure_id: str, name: str, created_at: str = "") -> DecodedFigure:
    entry = NATIVE_KINDS.get(artifact.artifact_id)
    if entry is None or entry[0] != artifact.kind or type(artifact.schema_version) is not int or artifact.schema_version != 1:
        raise FigureValidationError("Unsupported native figure identity or schema")
    if not isinstance(artifact.payload, Mapping):
        raise FigureValidationError("Native figure payload requires an object")
    resources = ResourceSet(artifact.tables, artifact.blobs)
    _validate_native_resources(artifact.payload, set(resources.table_names), set(resources.blob_names))
    native = _copy(artifact.payload)
    style_path, vocabulary = _native_style_path(entry[1])
    style = _get(native, style_path, None)
    warnings = ()
    if isinstance(style, Mapping) and style.get("style_binding"):
        from .figure_style_mapping import project_legacy_style
        projected = project_legacy_style(style["style_binding"], vocabulary, style)
        if projected != style:
            warnings = ("Compatibility look was regenerated from its authoritative style binding.",)
        _put(native, style_path, projected)
    b = _Builder(native)
    b.style_vocabulary = vocabulary
    _claim_kind(b, entry[1])
    b.resources(set(resources.table_names), set(resources.blob_names))
    claims = tuple(tuple(v["native"]) for v in b.bindings)
    leaves = leaf_paths(artifact.payload)
    claimed = tuple(p for p in leaves if any(p[:len(c)] == c for c in claims))
    carried = tuple(p for p in leaves if p not in set(claimed))
    spec = FigureSpec(figure_id=figure_id, kind=entry[1], name=name, created_at=created_at,
        recipe=FigureRecipe(entry[1], draft=b.draft, applied=b.applied, quantity_identity=b.quantity),
        sources=tuple(b.sources), axes=tuple(b.axes), series=tuple(b.series),
        texts=tuple(TextSpec.from_dict(t) for t in b.texts), annotations=tuple(b.annotations),
        presentation=FigurePresentation(overrides=b.overrides, page_extras=b.page_extras,
            style_ref=b.style_binding["ref"] if b.style_binding else None,
            style_snapshot=b.style_binding["snapshot"] if b.style_binding else {},
            extra={"style_binding_metadata": {k: v for k, v in b.style_binding.items()
                if k not in ("ref", "snapshot", "overrides")}} if b.style_binding else {}),
        view=FigureView(**b.view), result=ResultState(artifact.kind, artifact.schema_version, b.remainder,
            resources.table_names, resources.blob_names),
        extra={"native_codec": {"version": CODEC_VERSION, "artifact_id": artifact.artifact_id,
                                "artifact_extra": _copy(artifact.extra), "bindings": b.bindings}})
    # Validate the inverse before the caller is allowed to replace any artifact.
    restored = encode_legacy(spec, resources)
    if restored.payload != b.original or restored.extra != artifact.extra:
        raise FigureValidationError("Native figure codec did not preserve the complete payload")
    return DecodedFigure(spec, resources, MigrationReport(artifact.artifact_id, claimed, carried, warnings))


def encode_legacy(spec: FigureSpec, resources: ResourceSet, *, materialize_defaults=False):
    """Reconstruct exactly the old native shape, using current canonical values."""
    from .project import WorkspaceArtifact

    resources.validate(spec)
    codec = spec.extra.get("native_codec")
    if not isinstance(codec, Mapping) or codec.get("version") != CODEC_VERSION:
        raise FigureValidationError("Missing or unsupported native codec bindings")
    entry = NATIVE_KINDS.get(codec.get("artifact_id"))
    if entry is None or entry[1] != spec.kind or entry[0] != spec.result.native_kind:
        raise FigureValidationError("Native codec kind disagrees with the figure")
    payload = thaw_json(spec.result.payload)
    value = spec.to_dict()
    claims = []
    for binding in codec.get("bindings", ()):
        path = tuple(binding["native"])
        if not path or any(path[:len(old)] == old or old[:len(path)] == path for old in claims):
            raise FigureValidationError("Invalid or overlapping native codec bindings")
        claims.append(path)
        transform = binding.get("transform", "direct")
        if transform in ("single", "list"):
            result = [_get(value, p) for p in binding["targets"]]
            if transform == "single":
                if len(result) != 1:
                    raise FigureValidationError("Invalid native source binding")
                result = result[0]
        elif transform in ("style_binding", "style_field"):
            result = _binding_from_presentation(spec.presentation)
            if transform == "style_field":
                from .figure_style_mapping import project_legacy_style
                result = project_legacy_style(result, binding["vocabulary"], {})[binding["field"]]
        elif transform == "direct":
            result = _get(value, binding["target"])
        else:
            raise FigureValidationError("Unknown native binding transform")
        parent = _get(payload, path[:-1])
        if isinstance(parent, dict) and path[-1] in parent:
            raise FigureValidationError("Claimed native value also exists in the remainder")
        if isinstance(parent, list) and parent[path[-1]] is not None:
            raise FigureValidationError("Claimed native list value also exists in the remainder")
        _put(payload, path, _copy(result))
    artifact = WorkspaceArtifact(codec["artifact_id"], spec.result.native_kind, spec.result.native_schema,
                                 payload, resources.tables, resources.blobs, thaw_json(codec.get("artifact_extra", {})))
    return materialize_v12_defaults(artifact) if materialize_defaults else artifact


# Named public functions in analysis_workspace; consumers may import that core
# module when actually capturing/restoring. This registry imports no GUI classes.
PAGE_CODECS = FrozenMap({entry[1]: {"artifact_id": identity, "native_kind": entry[0],
    "capture": "capture_" + method, "restore": "restore_" + method}
    for identity, entry in NATIVE_KINDS.items()
    for method in [{"distributions": "plots", "distribution_summaries": "distribution_summaries",
                    "comparison_figures": "comparison_figures", "event_waveforms": "event_plots",
                    "noise": "psd", "iv": "iv", "voltage_series": "voltage_dependence",
                    "subpopulations": "subpopulation"}[entry[1]]]})


def record_to_artifact(record: FigureRecord):
    from .project import WorkspaceArtifact
    from .figure_spec import FIGURE_ARTIFACT_KIND, FIGURE_ARTIFACT_SCHEMA

    record.resources.validate(record.spec)
    return WorkspaceArtifact(record.spec.artifact_id, FIGURE_ARTIFACT_KIND, FIGURE_ARTIFACT_SCHEMA,
                             record.spec.to_dict(), record.resources.tables, record.resources.blobs, thaw_json(record.extra))


def record_from_artifact(artifact) -> FigureRecord:
    from .figure_spec import FIGURE_ARTIFACT_KIND, FIGURE_ARTIFACT_SCHEMA

    if (artifact.kind != FIGURE_ARTIFACT_KIND or type(artifact.schema_version) is not int
            or artifact.schema_version != FIGURE_ARTIFACT_SCHEMA):
        raise FigureValidationError("Unsupported figure artifact kind or schema")
    spec = FigureSpec.from_dict(artifact.payload)
    if artifact.artifact_id != spec.artifact_id:
        raise FigureValidationError("Figure artifact identity disagrees with its payload")
    return FigureRecord(spec, ResourceSet(artifact.tables, artifact.blobs), artifact.extra)


# Frozen from the v12 dataclass declarations at 65afa5c by AST inspection,
# never read from a live widget or regenerated at import time.
V12_PLOT_DEFAULTS = FrozenMap({'__nanoscope_workspace_value__': 'dataclass',
 'type': 'nanoscope.gui.plots_panel:PlotConfig',
 'fields': {'plot_type': 'Histogram',
            'selection': {'__nanoscope_workspace_value__': 'dataclass',
                          'type': 'nanoscope.gui.plots_panel:SelectionConfig',
                          'fields': {'source_keys': {'__nanoscope_workspace_value__': 'tuple',
                                                     'items': []},
                                     'accepted_only': True}},
            'histogram': {'__nanoscope_workspace_value__': 'dataclass',
                          'type': 'nanoscope.gui.plots_panel:HistogramConfig',
                          'fields': {'column': '',
                                     'bins': 30,
                                     'normalization': 'count',
                                     'colour_by': 'None',
                                     'log_x': False,
                                     'log_y': False,
                                     'include_filter_limited': False,
                                     'kde_overlay': False}},
            'scatter': {'__nanoscope_workspace_value__': 'dataclass',
                        'type': 'nanoscope.gui.plots_panel:ScatterConfig',
                        'fields': {'x': '',
                                   'y': '',
                                   'log_x': False,
                                   'log_y': False,
                                   'colour_by': 'None'}},
            'kde': {'__nanoscope_workspace_value__': 'dataclass',
                    'type': 'nanoscope.gui.plots_panel:KdeConfig',
                    'fields': {'x': '', 'y': '', 'log_x': True, 'log_y': False, 'levels': 8}},
            'heatmap': {'__nanoscope_workspace_value__': 'dataclass',
                        'type': 'nanoscope.gui.plots_panel:HeatmapConfig',
                        'fields': {'x': '', 'y': '', 'log_x': True, 'log_y': False, 'bins': 40}},
            'cluster': {'__nanoscope_workspace_value__': 'dataclass',
                        'type': 'nanoscope.gui.plots_panel:ClusterConfig',
                        'fields': {'features': {'__nanoscope_workspace_value__': 'tuple',
                                                'items': []},
                                   'method': 'GMM',
                                   'k_mode': 'Auto (BIC)',
                                   'k': 2}},
            'fit': {'__nanoscope_workspace_value__': 'dataclass',
                    'type': 'nanoscope.gui.plots_panel:FitConfig',
                    'fields': {'kind': 'none',
                               'k': 1,
                               'k_auto': False,
                               'scatter_kind': 'none',
                               'applied': 'none',
                               'applied_k': 1,
                               'applied_k_auto': False,
                               'applied_scatter': 'none',
                               'range_lo': '',
                               'range_hi': '',
                               'range_unit': 'ms'}},
            'axes': {'__nanoscope_workspace_value__': 'dataclass',
                     'type': 'nanoscope.gui.plots_panel:AxesConfig',
                     'fields': {'time_unit': 'ms',
                                'x_min': '',
                                'x_max': '',
                                'y_min': '',
                                'y_max': '',
                                'manual_x': None,
                                'manual_y': None}},
            'style': {'__nanoscope_workspace_value__': 'dataclass',
                      'type': 'nanoscope.gui.plots_panel:StyleSettings',
                      'fields': {'title': '',
                                 'x_label': '',
                                 'y_label': '',
                                 'show_legend': True,
                                 'font_family': '',
                                 'font_size_pt': 10,
                                 'data_color': '#0072B2',
                                 'fit_color': '#D55E00',
                                 'point_size': 7,
                                 'line_width': 1,
                                 'background': 'dark',
                                 'export_w': 0,
                                 'export_h': 0,
                                 'label_overrides': {},
                                 'legend_pos': None,
                                 'legend_font_size_pt': 0,
                                 'legend_counts': True,
                                 'legend_include': {},
                                 'show_grid': True,
                                 'title_font_size_pt': 0,
                                 'axis_label_font_size_pt': 0,
                                 'tick_font_size_pt': 0,
                                 'show_ci_band': False,
                                 'show_provenance': False,
                                 'show_residuals': False,
                                 'heatmap_colormap': 'viridis',
                                 'text_styles': {},
                                 'fit_label': {}}},
            'norm_user_set': False}})


def materialize_v12_defaults(artifact):
    """Supply historical missing PlotConfig fields only for the renderer boundary.

    The canonical document and encode_legacy's default inverse preserve omission.
    A reused widget cannot donate its own current values to an older saved config.
    """
    from dataclasses import replace

    if artifact.artifact_id != "builtin.plots":
        return artifact
    payload = _copy(artifact.payload)
    config = payload.get("config")
    if not isinstance(config, dict) or config.get(TAG) != "dataclass":
        raise FigureValidationError("Invalid tagged v12 PlotConfig")

    def fill(saved, defaults):
        if not isinstance(saved, dict) or saved.get(TAG) != defaults.get(TAG):
            raise FigureValidationError("Malformed tagged v12 configuration section")
        fields = saved.get("fields")
        if not isinstance(fields, dict):
            raise FigureValidationError("Tagged configuration fields require an object")
        for key, default in defaults["fields"].items():
            if key not in fields:
                fields[key] = _copy(default)
            elif isinstance(default, Mapping) and default.get(TAG) == "dataclass":
                fill(fields[key], default)
    fill(config, V12_PLOT_DEFAULTS)
    _fold_v12_legacy_style(config["fields"]["style"]["fields"])
    return replace(artifact, payload=payload)


def _fold_v12_legacy_style(style):
    """Frozen D1/D2 duplicate-switch semantics at the restore boundary only."""
    legacy = ("fit_values_on_plot", "show_r2", "show_gof", "title_bold", "title_italic")
    if not any(key in style for key in legacy):
        return
    numbers = bool(style.get("fit_values_on_plot", False))
    no_r2, no_ks = not bool(style.get("show_r2", True)), not bool(style.get("show_gof", True))
    if numbers or no_r2 or no_ks:
        saved = style.get("fit_label")
        label = dict(saved) if isinstance(saved, dict) and saved else {
            "visible": True, "fields": {TAG: "tuple", "items": ["equation", "r2", "n"]},
            "template": "", "placement": "top-left", "in_legend": False}
        encoded_fields = label.get("fields", ["equation", "r2", "n"])
        if isinstance(encoded_fields, dict) and encoded_fields.get(TAG) == "tuple":
            encoded_fields = encoded_fields.get("items", [])
        fields = set(encoded_fields) if isinstance(encoded_fields, (list, tuple)) else {"equation", "r2", "n"}
        template = label.get("template", "")
        if numbers and not saved:
            fields.update(("bic", "ks"))
        if no_r2:
            fields.discard("r2")
            if isinstance(template, str):
                template = template.replace("{r2}", "—")
        if no_ks:
            fields.discard("ks")
            if isinstance(template, str):
                template = template.replace("{ks}", "—")
        label["fields"] = {TAG: "tuple", "items": [key for key in ("equation", "r2", "n", "ci", "bic", "ks") if key in fields]}
        label["template"] = template
        style["fit_label"] = label
    for old, new in (("title_bold", "bold"), ("title_italic", "italic")):
        if bool(style.get(old, False)):
            styles = style.setdefault("text_styles", {})
            title = styles.setdefault("title", {})
            if title.get(new) is None:
                title[new] = True
    # Do not ask a future GUI's legacy folder to interpret these a second time.
    for key in legacy:
        style.pop(key, None)


_ROOT_FIELDS = {
    "distributions": {"config", "selected_sources", "source_frame_fingerprint", "cluster", "pca", "fits"},
    "distribution_summaries": {"schema_version", "draft", "applied", "appearance", "source_appearance", "source_recipes", "snapshot", "stale", "view_range", "table_resources"},
    "comparison_figures": {"schema_version", "draft", "applied", "appearance", "source_appearance", "source_recipes", "snapshot", "stale", "canvas_view", "table_resources"},
    "iv": {"points", "exclusions", "appearance", "fit", "details_text", "headline"},
    "event_waveforms": {"source", "units", "accepted_event_count", "config", "density_counts", "amplitude_warnings", "amplitude_snapshot_units", "density_edges", "ensemble", "ensemble_band", "ensemble_alignment", "ensemble_input_count", "ensemble_unalignable_count", "pending_analysis"},
    "noise": {"config", "curves", "fft", "spectrogram", "all_points", "histogram_levels", "compute_notes", "pending_analysis", "signal"},
    "voltage_series": {"config", "series", "series_excluded", "point_sources", "fit", "not_in_fit", "details_text", "warning_text", "headline"},
    "subpopulations": {"source", "accepted_event_ids", "report", "report_features_fingerprint", "subtitle", "view", "assignment_state", "assignment_reason", "curation_note", "appearance"},
}
_TEXT_FIELDS = {"text", "family", "size_pt", "bold", "italic", "color", "align"}
_FIT_LABEL_FIELDS = {"visible", "fields", "template", "placement", "offset", "size_pt", "color", "in_legend"}
_LEGACY_STYLE_FIELDS = {"fit_values_on_plot", "show_r2", "show_gof", "title_bold", "title_italic"}
_CONFIG_FIELDS = {
    "event_waveforms": {"view", "x_metric", "y_metric", "log_x", "log_y", "bins", "colormap", "density_title", "density_x_label", "density_y_label", "normalize_time", "align", "n_points", "pad_percent", "ensemble_subset", "ensemble_band", "ensemble_title", "appearance"},
    "noise": {"target_key", "compare_keys", "target_source", "compare_sources", "signal_basis", "start_s", "end_s", "nperseg", "window", "overlap_percent", "log_log", "asd", "rms_lo_hz", "rms_hi_hz", "view", "spectrogram_nperseg", "spectrogram_colormap", "spectrogram_levels_mode", "spectrogram_level_low", "spectrogram_level_high", "spectrogram_log_frequency", "histogram_bins", "histogram_clip", "histogram_levels", "appearance", "style"},
    "voltage_series": {"metric", "selected_fit", "applied_fit", "absolute_voltage", "log_y", "accepted_only", "appearance"},
}


def _v12_keys(kind, path, value):
    if not path:
        return _ROOT_FIELDS[kind]
    if path == ("config",) and kind != "distributions":
        return _CONFIG_FIELDS.get(kind, set(value))
    if value.get(TAG) == "dataclass":
        return {TAG, "type", "fields"}
    if path[-1] == "fields" and path[0] == "config" and kind == "distributions":
        default = _get(V12_PLOT_DEFAULTS, path[1:], None)
        return set(default or {}) | ((_LEGACY_STYLE_FIELDS | {"style_binding", "series_styles", "frame"}) if "style" in path else set())
    if path[-1] == "appearance":
        return set(WORD_FIELDS) | APPEARANCE_FIELDS | {"text", "fit_label", "axes", "annotations", "style_binding", "quantity_display", "series_styles", "source_appearance"}
    if path[-1] == "style_binding":
        return {"version", "ref", "snapshot", "overrides"}
    if "style_binding" in path:
        return set(value)
    if path[-1] == "style" and kind == "noise":
        return {"title", "data_color", "fit_color", "background", "export_w", "export_h"}
    if len(path) >= 2 and path[-2] in ("text", "text_styles"):
        return _TEXT_FIELDS
    if path[-1] == "fit_label":
        return _FIT_LABEL_FIELDS
    if len(path) >= 2 and path[-2] == "axes":
        return {"min", "max", "log", "tick_major"}
    if len(path) >= 2 and path[-2] == "annotations":
        return {"offset"}
    if len(path) == 2 and path[0] == "source_appearance":
        return {"label", "color"}
    # Existing dynamic containers (element ids, source ids) own their child names.
    # Unknown fields inside those children are still carried by the cases above.
    return set(value)


def merge_native_capture(captured, previous: FigureRecord) -> FigureRecord:
    """Checkpoint native state while retaining forward fields a page cannot emit.

    Known reset fields stay reset. Missing known result branches stay missing. The
    old binding ledger is never replayed over a newly captured appearance.
    """
    from dataclasses import replace

    if isinstance(captured, DecodedFigure):
        artifact = encode_legacy(captured.spec, captured.resources)
    elif isinstance(captured, FigureRecord):
        artifact = encode_legacy(captured.spec, captured.resources)
    else:
        artifact = captured
    old = encode_legacy(previous.spec, previous.resources)
    if (artifact.artifact_id, artifact.kind) != (old.artifact_id, old.kind):
        raise FigureValidationError("A figure capture cannot change its native kind")
    kind = previous.spec.kind
    current = _copy(artifact.payload)

    def merge(before, after, path=()):
        if not isinstance(before, dict) or not isinstance(after, dict):
            return
        known = _v12_keys(kind, path, before)
        for key, value in before.items():
            child = path + (key,)
            if key == "style_binding" and not after.get(key):
                # An explicit native reset/old-state Undo owns removal. Forward
                # metadata is carried separately below, never revived as a binding.
                continue
            if key in after:
                merge(value, after[key], child)
            elif key not in known:
                after[key] = _copy(value)
            elif isinstance(value, dict) and ("appearance" in child or "source_appearance" in child
                    or (kind == "distributions" and child[0] == "config")):
                unknown = {}
                merge(value, unknown, child)
                if unknown:
                    after[key] = unknown
    merge(old.payload, current)

    def strings(value):
        if isinstance(value, dict):
            return {s for child in value.values() for s in strings(child)}
        if isinstance(value, list):
            return {s for child in value for s in strings(child)}
        return {value} if type(value) is str else set()
    old_refs, refs = strings(old.payload), strings(current)
    tables, blobs = dict(artifact.tables), dict(artifact.blobs)
    for names, source in ((tables, old.tables), (blobs, old.blobs)):
        for key, value in source.items():
            if key not in names and (key in refs or key not in old_refs):
                names[key] = value
    artifact = replace(artifact, payload=current, tables=tables, blobs=blobs,
                       extra={**old.extra, **artifact.extra})
    decoded = decode_legacy(artifact, figure_id=previous.spec.figure_id,
                            name=previous.spec.name, created_at=previous.spec.created_at)
    presentation = decoded.spec.presentation
    # Native Undo owns the binding. Carry only the old Phase2 out-of-band snapshot;
    # restoring it over a newly captured binding would undo a native Undo itself.
    if ("style_binding_metadata" not in presentation.extra and
            "style_binding_metadata" not in previous.spec.presentation.extra):
        presentation = replace(presentation, style_ref=previous.spec.presentation.style_ref,
                               style_snapshot=previous.spec.presentation.style_snapshot)
    spec = replace(decoded.spec, lineage=previous.spec.lineage,
                   presentation=presentation,
                   extra={**thaw_json(previous.spec.extra), **thaw_json(decoded.spec.extra)})
    spec = _carry_forward_schema(spec, previous.spec)
    if "style_binding_metadata" not in decoded.spec.presentation.extra:
        extra = thaw_json(spec.presentation.extra)
        metadata = extra.pop("style_binding_metadata", None)
        if metadata:
            opaque = {key: value for key, value in metadata.items() if key != "version"}
            if opaque:
                extra["retired_style_binding_extensions"] = opaque
        spec = replace(spec, presentation=replace(spec.presentation, extra=extra))
    return FigureRecord(spec, decoded.resources, previous.extra)


def _carry_forward_schema(current, previous):
    """Carry opaque schema extensions across a native page's partial vocabulary.

    Scientific values come entirely from the capture. An absent presentation element
    may retain an extension-only shell, so resetting known text or pins does not erase
    a future reader's fields. Removed data/source identities are not resurrected.
    """
    from dataclasses import fields, replace
    from .figure_spec import _Node

    if not isinstance(current, _Node) or type(current) is not type(previous):
        return current
    changes = {"extra": {**thaw_json(previous.extra), **thaw_json(current.extra)},
               "_unknown_fields": {**thaw_json(previous._unknown_fields), **thaw_json(current._unknown_fields)}}
    identities = {"sources": "source_id", "axes": "axis_id", "series": "series_id",
                  "texts": "element_id", "annotations": "element_id"}
    for descriptor in fields(current):
        key = descriptor.name
        if key in changes:
            continue
        now, old = getattr(current, key), getattr(previous, key)
        if isinstance(now, _Node):
            changes[key] = _carry_forward_schema(now, old)
        elif key in identities and isinstance(now, tuple) and isinstance(old, tuple):
            identity = identities[key]
            before = {getattr(node, identity): node for node in old}
            after = {getattr(node, identity) for node in now}
            nodes = [_carry_forward_schema(node, before[getattr(node, identity)])
                     if getattr(node, identity) in before else node for node in now]
            if key in ("texts", "axes", "annotations"):
                for node in old:
                    if getattr(node, identity) in after:
                        continue
                    # Comparing to a default shell finds actual retained extensions
                    # without mistaking an old word, numeric pin or position for one.
                    shell = type(node)(**{identity: getattr(node, identity)})
                    carried = _carry_forward_schema(shell, node)
                    if carried != shell:
                        nodes.append(carried)
            changes[key] = tuple(nodes)
    return replace(current, **changes)
