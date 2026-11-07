"""Distributions' full-configuration implementation of the figure surface seam."""
from __future__ import annotations

from dataclasses import asdict

from ..core.figure_spec import FigureValidationError
from .figure_adapter_pages import FigureViewAdapter, _plain


class PlotsFigureAdapter(FigureViewAdapter):
    def __init__(self, page):
        super().__init__(page, "distributions")
        page._phase2_figure_managed = True

    def _commit_edits(self):
        super()._commit_edits()
        self.page._history_commit()

    def _capture_history(self):
        return self.page.capture_history_capsule()

    def _install_history(self, capsule):
        self.page.restore_history_capsule(capsule)

    def _capture_memos(self):
        memos = super()._capture_memos()
        memos["_identity"] = _plain(asdict(self.page._identity))
        return memos

    def _install_memos(self, raw):
        from .plots_panel import AnalysisIdentity
        raw = dict(raw)
        identity = dict(raw.pop("_identity"))
        identity["features"] = tuple(identity["features"])
        self.page._identity = self.page.config().identity  # installed config owns migrated units/density factors
        super()._install_memos(raw)
        self.page._sync_run_state()

    def suspend(self):
        super().suspend()
        page = self.page
        page._note_timer.stop()
        page._history_timer.stop()
        page._stale_redraw_timer.stop()
        page._abort_note_job()
        page._pending_notes.clear()
        # No closeEvent, reset_state or _persist_results: the record owns these results.

    def capture_native(self):
        from dataclasses import replace
        from ..core.analysis_workspace import capture_plots
        self._commit_edits()
        page = self.page
        applied = getattr(page, "_figure_applied_config", None)
        if applied is None:
            artifact = capture_plots(page)
        else:
            # PCA/clustering belong to the drawn primary, even while the source draft
            # names another input. Capture its fingerprint/preprocessing without moving UI.
            source = next((page._sources[key] for key in applied.selection.source_keys
                           if key in page._sources), None)
            frame = None if source is None else source.df
            if frame is not None and applied.selection.accepted_only and "accepted" in frame:
                frame = frame[frame["accepted"].astype(bool)]

            class CaptureView:
                def __getattr__(self, name):
                    return getattr(page, name)

                def _current_frame(self):
                    return frame

            artifact = capture_plots(CaptureView())
        render, tables = _capture_render(self.page, artifact)
        snapshots, source_tables, source_blobs = _capture_sources(self.page, artifact)
        render["source_snapshots"] = snapshots
        return replace(artifact, payload={**artifact.payload, "phase2_render": render},
                       tables={**tables, **source_tables}, blobs={**artifact.blobs, **source_blobs})

    def restore_native(self, artifact):
        from dataclasses import replace
        from ..core.analysis_workspace import restore_plots, _restore_fit_display, _restore_nonlinear_fit
        from .plots_panel import plot_config_to_json, plot_config_from_json, migrate_plot_quantity
        render = artifact.payload.get("phase2_render")
        if render is None:
            return restore_plots(self.page, artifact)
        _validate_render(render, artifact.tables)
        page = self.page
        _install_sources(page, render, artifact)
        raw_draft = _native_config(artifact, migrate=False)
        draft = migrate_plot_quantity(raw_draft)
        applied_json = render.get("applied_config")
        if not applied_json:
            # A never-plotted draft remains a never-plotted draft; native restoration
            # would draw it, which is an analysis action the user has not requested.
            page._set_config(draft, draw=False, own=True)
            page._plot_armed = False
            page._stale = True
            page._stale_reasons = ["not plotted yet"]
            page._sync_run_state()
            page.adopt_restored_state()
            page._resolve_frozen_plot_source = None
            from ..core.analysis_workspace import RestoreReport
            return RestoreReport(restored=(artifact.artifact_id,))
        applied = _config_for_applied(raw_draft, applied_json)
        shown = _with_native_config(artifact, applied)
        shown = replace(shown, payload={**shown.payload,
                                       "selected_sources": render["drawn_sources"]})
        page._phase2_rendering = (render, artifact.tables)
        try:
            report = restore_plots(page, shown)
            if report.skipped:
                return report
            render = _render_text_in_display_units(page, render, artifact.tables, applied)
            _install_render_items(page, render, artifact.tables)
            _restore_fit_display(page, artifact)
            _restore_nonlinear_fit(page, artifact)
            page._set_config(draft, draw=False, own=True)
            page._figure_applied_config = applied
            page._applied_kde_overlay = bool(applied.histogram.kde_overlay)
            page._kde_pending_after_snapshot = False
            page._scatter_fit_pending_after_snapshot = False
            gate = render["gate"]
            page._plot_armed = gate["armed"]
            page._stale, page._stale_reason = gate["stale"], gate["reason"]
            page._stale_reasons = list(gate["reasons"])
            page._figure_sources = dict(zip(render["drawn_source_keys"], render["drawn_sources"]))
            page._presentation_sources = tuple(tuple(row) for row in render["presentation_sources"])
            page._presentation_provenance = list(render["provenance"])
            page._presentation_provenance_by_source = list(render["provenance_by_source"])
            page._restyle_committed_decorations()
            page._apply_series_styles()
            page._set_details(render["details"])
            page._sync_fit_apply_buttons()
            page._sync_run_state()
            page.adopt_restored_state()
            return report
        finally:
            page._phase2_rendering = None
            page._resolve_frozen_plot_source = None


_RENDER_VERSION = 1
_RENDER_BYTES = 64 * 1024 * 1024
_RENDER_ITEMS = 4096


def plotted_staleness_columns(config):
    """Storage columns on which the retained picture depends (no calculation)."""
    from .plots_panel import HISTOGRAM, SCATTER, KDE, HEATMAP, PCA, _COLOUR_CLUSTER
    kind=config.plot_type
    if kind == HISTOGRAM:
        columns=[config.histogram.column]
        clustered=config.histogram.colour_by == _COLOUR_CLUSTER
    elif kind in (SCATTER,KDE,HEATMAP):
        section={SCATTER:config.scatter,KDE:config.kde,HEATMAP:config.heatmap}[kind]
        columns=[section.x,section.y]
        clustered=kind == SCATTER and config.scatter.colour_by == _COLOUR_CLUSTER
    else:
        columns=[];clustered=kind == PCA
    if clustered:columns.extend(config.cluster.features)
    return tuple(dict.fromkeys(columns))


def record_table_is_stale(record, owner, provider):
    """Judge a dormant figure from its retained recipe, without making a Qt page."""
    from ..core.analysis_workspace import _resolve_source
    from ..core.figure_codecs import encode_legacy
    native=encode_legacy(record.spec,record.resources,materialize_defaults=True)
    render=native.payload.get('phase2_render',{})
    draft=_native_config(native,migrate=False)
    applied=render.get('applied_config')
    config=_config_for_applied(draft,applied) if applied else draft
    columns=set(plotted_staleness_columns(config))
    recipes=render.get('drawn_sources',native.payload.get('selected_sources',()))
    for recipe in recipes:
        doc_id,channel,_note=_resolve_source(owner,recipe)
        module=recipe.get('module_id')
        if doc_id is None or channel is None or not module:continue
        stale,_sentence=provider(doc_id,channel,module)
        stale=set(stale)
        if 'depth_pA' in stale:stale.add('amplitude_pA')
        if columns.intersection(stale):return True
    return False


def _capture_render(page, artifact):
    """Detach sampled estimates only; source/result tables remain in the native capture."""
    import numpy as np
    import pandas as pd
    import pyqtgraph as pg
    from dataclasses import asdict

    tables, nodes, legends = dict(artifact.tables), [], []
    total_bytes = 0
    fit_items = list(page._scatter_fit_items)
    readouts = list(page._presentation_readouts)
    item_indices = {}
    for item in page._plot.items:
        is_fit = item in fit_items
        is_kde = getattr(item, "_plot_data_part", None) == "kde"
        is_contour = isinstance(item, pg.IsocurveItem)
        if not (is_fit or is_kde or is_contour):
            continue
        if len(nodes) >= _RENDER_ITEMS:
            raise FigureValidationError("Figure sampled output exceeds the checkpoint item bound")
        name = f"phase2.render.{len(nodes)}"
        if name in tables:
            raise FigureValidationError("Figure sampled output resource name collision")
        pen = item.opts["pen"] if not is_contour else item.pen
        node = {"table": name, "primitive": "contour" if is_contour else "curve",
                "role": "scatter_fit" if is_fit else "kde" if is_kde else "contour",
                "pen": {"color": pen.color().name(), "width": float(pen.widthF()),
                        "style": int(pen.style().value)},
                "attributes": {}}
        for key in ("_presentation_role", "_presentation_width_extra", "_plot_data_part",
                    "_kde_series", "_presentation_fit_family", "_figure_series_meta"):
            if hasattr(item, key):
                node["attributes"][key] = getattr(item, key)
        if is_contour:
            array = np.asarray(item.data).copy()
            if array.ndim != 2:
                raise FigureValidationError("Invalid sampled contour grid")
            transform = item.transform()
            node["level"] = float(item.level)
            node["axis_order"] = item.axisOrder
            node["transform"] = [float(getattr(transform, method)()) for method in
                                 ("m11", "m12", "m13", "m21", "m22", "m23", "m31", "m32", "m33")]
            frame = pd.DataFrame(array, columns=[str(i) for i in range(array.shape[1])])
        else:
            x, y = np.asarray(item.xData), np.asarray(item.yData)
            if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
                raise FigureValidationError("Invalid sampled curve coordinates")
            frame = pd.DataFrame({"x": x.copy(), "y": y.copy()})
            if is_fit:
                index = fit_items.index(item)
                node["source"] = readouts[index][0] if index < len(readouts) else ""
        total_bytes += int(frame.memory_usage(index=True, deep=True).sum())
        if total_bytes > _RENDER_BYTES:
            raise FigureValidationError("Figure sampled output exceeds the 64 MiB checkpoint bound")
        tables[name] = frame
        item_indices[id(item)] = len(nodes)
        nodes.append(node)
    for _entry, (item, label, kind, key) in page._presentation_legend.items():
        index = item_indices.get(id(item))
        if index is not None or (kind == "kde" and any(n["role"] == "kde" for n in nodes)):
            legends.append({"node": index, "label": str(label), "kind": kind, "key": key})
        elif nodes and all(node["role"] == "contour" for node in nodes):
            pen = item.opts["pen"]
            legends.append({"node": None, "label": str(label), "kind": kind, "key": key,
                            "pen": {"color": pen.color().name(), "width": float(pen.widthF()),
                                    "style": int(pen.style().value)}})
    applied_config = page.figure_applied_config()
    if applied_config:
        applied_config.pop("style", None)
        applied_config["axes"] = {key: applied_config["axes"][key] for key in ("time_unit", "quantity_display")}
    render = {"version": _RENDER_VERSION, "applied_config": applied_config,
              "gate": {"armed": bool(page._plot_armed), "stale": bool(page._stale),
                       "reason": page._stale_reason, "reasons": list(page._stale_reasons)},
              "drawn_sources": page.figure_source_recipes(),
              "drawn_source_keys": list(page._figure_sources), "nodes": nodes, "legends": legends,
              "readouts": [[key, asdict(value)] for key, value in readouts],
              "details": page.details_text(), "title": page._title_presentation,
              "axis_labels": page._axis_auto_labels, "kde_frame": page._kde_view_frame,
              "presentation_sources": page._presentation_sources,
              "provenance": page._presentation_provenance,
              "provenance_by_source": page._presentation_provenance_by_source}
    return _plain(render), tables


def _validate_render(render, tables):
    import numpy as np
    from collections.abc import Mapping

    if not isinstance(render, Mapping) or render.get("version") != _RENDER_VERSION:
        raise FigureValidationError("Unsupported figure sampled-output version")
    nodes = render.get("nodes")
    if not isinstance(nodes, (list, tuple)) or len(nodes) > _RENDER_ITEMS:
        raise FigureValidationError("Invalid sampled-output nodes")
    total = 0
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("table") not in tables:
            raise FigureValidationError("Missing sampled-output table")
        frame = tables[node["table"]]
        total += int(frame.memory_usage(index=True, deep=True).sum())
        if total > _RENDER_BYTES:
            raise FigureValidationError("Sampled output exceeds its checkpoint bound")
        values = frame.to_numpy()
        if values.dtype.kind not in "fiu":
            raise FigureValidationError("Sampled output must contain numerical values")
        # NaN curve breaks are intentional; infinities are never drawable.
        if np.isinf(values).any():
            raise FigureValidationError("Sampled output contains infinite coordinates")
        if node.get("primitive") == "curve":
            if list(frame.columns) != ["x", "y"]:
                raise FigureValidationError("Invalid sampled curve columns")
        elif node.get("primitive") == "contour":
            if not np.isfinite(values).all() or min(values.shape, default=0) < 2:
                raise FigureValidationError("Invalid sampled contour grid")
            if len(node.get("transform", ())) != 9:
                raise FigureValidationError("Invalid sampled contour transform")
        else:
            raise FigureValidationError("Unknown sampled-output primitive")


def _scatter_readouts_in_display_units(render, tables, applied):
    """Re-express saved analytic lines from their retained geometry, never refit data.

    A Phase 2 envelope has sampled linear/power-law curves plus generated readout text.
    Its two endpoints determine the same equation algebraically; no source rows, loss
    function, estimator, significance statistic, or user wording is involved here.
    """
    from copy import deepcopy
    import math
    import numpy as np
    from .plots_panel import _plot_quantity_bases
    result = deepcopy(render)
    bases = _plot_quantity_bases(applied)
    xq, yq = bases["bottom"].quantity, bases["left"].quantity
    units = ", ".join(value for value in (
        f"x in {xq.display_unit}" if xq.display_unit else "",
        f"y in {yq.display_unit}" if yq.display_unit else "") if value)
    suffix = f"   ({units})" if units else ""
    by_source = {key: value for key, value in result["readouts"]}
    for node in result["nodes"]:
        if node["role"] != "scatter_fit":
            continue
        value = by_source.get(node.get("source", ""))
        if value is None or "equation" not in value["parts"]:
            continue
        frame = tables[node["table"]]
        x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        family = node["attributes"].get("_presentation_fit_family")
        if len(x) < 2 or x[0] == x[-1] or family not in ("linear", "power law"):
            raise FigureValidationError("Saved regression geometry cannot supply its display equation")
        if family == "power law":
            if min(x[0], x[-1], y[0], y[-1]) <= 0:
                raise FigureValidationError("Saved power-law geometry has no positive endpoint equation")
            exponent = math.log(float(y[-1] / y[0])) / math.log(float(x[-1] / x[0]))
            coefficient = float(y[0]) / float(x[0]) ** exponent
            equation = f"y = {coefficient * yq.factor / xq.factor ** exponent:.4g}·x^{exponent:.4g}{suffix}"
        else:
            slope = float(y[-1] - y[0]) / float(x[-1] - x[0])
            intercept = float(y[0]) - slope * float(x[0])
            equation = f"y = {slope * yq.factor / xq.factor:.4g}·x + {intercept * yq.factor:.4g}{suffix}"
        previous = value["parts"]["equation"]
        value["parts"]["equation"] = equation
        result["details"] = result["details"].replace(previous, equation)
    return result


def _render_text_in_display_units(page, render, tables, applied):
    from dataclasses import asdict
    from .plots_panel import HISTOGRAM, SCATTER
    if applied.plot_type == HISTOGRAM:
        # Native histogram restore already formats the saved fit parameters in ms.
        # Keep those readouts instead of replacing them with a pre-migration cache.
        details = page.details_text()
        if any(node["role"] == "kde" for node in render["nodes"]):
            details = details.replace(
                "KDE overlay: restored as a setting and NOT recomputed on project "
                "open (estimating it is an analysis run). Press Plot to draw it "
                "from the attached source data.",
                "KDE overlay restored from its saved sampled coordinates; no estimator was run.")
        return {**render, "details": details,
                "readouts": [[key, asdict(value)] for key, value in page._presentation_readouts]}
    previous_axes = (render.get("applied_config") or {}).get("axes", {})
    if applied.plot_type == SCATTER and previous_axes.get("time_unit", "ms") != "ms":
        return _scatter_readouts_in_display_units(render, tables, applied)
    return render


def _legacy_series_of_render(render, tables):
    """Recover only identities guaranteed by the version-1 writer's draw ordering.

    That writer stores the ordered source/cluster legend keys and emits a fixed-size
    contour block per successful source. Validate every repeated grid/transform before
    using that contract; otherwise leave honest retained-artist targets available.
    """
    import numpy as np
    nodes, found = render["nodes"], {}
    sources = list(render.get("drawn_source_keys", ()))
    if nodes and all(node["primitive"] == "contour" for node in nodes):
        legends = [entry for entry in render["legends"] if entry["kind"] == "series" and entry.get("key")]
        groups = ([(entry["key"], entry["label"]) for entry in legends]
                  if legends else [(sources[0], sources[0])] if len(sources) == 1 else [])
        levels = (render.get("applied_config") or {}).get("kde", {}).get("levels")
        if type(levels) is int and levels > 0 and groups and len(nodes) == len(groups) * levels:
            for group, (key, label) in enumerate(groups):
                indices = range(group * levels, (group + 1) * levels)
                first = nodes[group * levels]
                valid = all(nodes[i]["transform"] == first["transform"] and
                    np.array_equal(tables[nodes[i]["table"]].to_numpy(),
                                   tables[first["table"]].to_numpy(), equal_nan=True) for i in indices)
                if not valid:
                    return {}
                found.update({i: (key, label, "source") for i in indices})
    else:
        clusters = [entry for entry in render["legends"] if entry["kind"] == "cluster"]
        for index, node in enumerate(nodes):
            if node["role"] != "kde":
                continue
            group = node["attributes"].get("_kde_series")
            if type(group) is not int or group < 0:
                continue
            if clusters and group < len(clusters):
                found[index] = (clusters[group]["key"], clusters[group]["label"], "category")
            elif group < len(sources):
                found[index] = (sources[group], sources[group], "source")
    return found


def _install_render_items(page, render, tables, *, contours_only=False):
    """Draw stored samples and readouts; no estimator or source-data read is invoked."""
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui
    from .fit_label import FitReadout

    _validate_render(render, tables)
    created = {}
    legacy_series = _legacy_series_of_render(render, tables)
    for index, node in enumerate(render["nodes"]):
        contour = node["primitive"] == "contour"
        if contour != contours_only:
            continue
        frame = tables[node["table"]]
        pen = node["pen"]
        pen = pg.mkPen(pen["color"], width=pen["width"],
                       style=QtCore.Qt.PenStyle(pen["style"]))
        if contour:
            item = pg.IsocurveItem(data=frame.to_numpy(copy=True), level=node["level"], pen=pen,
                                    axisOrder=node["axis_order"])
            item.setTransform(QtGui.QTransform(*node["transform"]))
        else:
            item = pg.PlotDataItem(x=frame["x"].to_numpy(copy=True),
                                   y=frame["y"].to_numpy(copy=True), pen=pen, connect="finite")
        for key, value in node["attributes"].items():
            if key not in {"_presentation_role", "_presentation_width_extra", "_plot_data_part",
                           "_kde_series", "_presentation_fit_family", "_figure_series_meta"}:
                raise FigureValidationError("Unknown sampled curve attribute")
            setattr(item, key, value)
        if not hasattr(item, "_figure_series_meta"):
            # Old sampled envelopes lacked source tags. A regression's source key is
            # explicit; an otherwise unidentified retained contour remains an artist
            # target under its stable resource key, never guessed from color or label.
            source = node.get("source")
            role = "fit" if node["role"] == "scatter_fit" else "kde" if node["role"] == "kde" else "data"
            key, label, identity_kind = legacy_series.get(index, (
                source or node["table"], source or f"Saved {node['role']} {index + 1}",
                "source" if source else "artist"))
            page._tag_series(item, role, key, label, identity_kind=identity_kind,
                             colour=node["pen"]["color"] if node["attributes"].get("_presentation_role") == "category" else None)
        page._plot.addItem(item)
        created[index] = item
        if node["role"] == "scatter_fit":
            page._scatter_fit_items.append(item)
            if page._scatter_fit_item is None:
                page._scatter_fit_item = item
    for legend in render["legends"]:
        if legend["node"] is not None:
            item = created.get(legend["node"])
            if item is None:
                continue
        else:
            if contours_only != (legend["kind"] != "kde"):
                continue
            pen = legend.get("pen", {"color": "#9aa7bd", "width": page._style.line_width + 1,
                                     "style": int(QtCore.Qt.PenStyle.DotLine.value)})
            item = pg.PlotDataItem(pen=pg.mkPen(pen["color"], width=pen["width"],
                                              style=QtCore.Qt.PenStyle(pen["style"])))
        page._legend_add(item, legend["label"], kind=legend["kind"], key=legend["key"])
    if not contours_only:
        page._presentation_readouts = [(key, FitReadout(value["parts"], tuple(value["refusals"]),
                                                      value["source"], value["colour"]))
                                       for key, value in render["readouts"]]
        page._draw_fit_label()
    page._set_details(render["details"])


def restore_sampled_contours(page):
    """Frozen-render hook used only by a new-record workspace restore."""
    frozen = getattr(page, "_phase2_rendering", None)
    if frozen is None:
        return False
    render, tables = frozen
    _validate_render(render, tables)
    if not any(node["primitive"] == "contour" for node in render["nodes"]):
        return False
    page._clear_plot()
    _install_render_items(page, render, tables, contours_only=True)
    cfg = page.config().kde
    page._plot.setLogMode(x=cfg.log_x, y=cfg.log_y)
    for side, column, log in (("bottom", cfg.x, cfg.log_x), ("left", cfg.y, cfg.log_y)):
        page._plot.setLabel(side, page._axis_label_for(side, column, log), **page._label_style())
    page._apply_title(render["title"][0], render["title"][2])
    page._apply_background()
    page._apply_fonts()
    frame = render["kde_frame"]
    if frame is not None:
        page._kde_view_frame = tuple(frame)
        page._plot.getViewBox().setRange(xRange=frame[:2], yRange=frame[2:], padding=0)
    return True


def _config_for_applied(raw_draft, applied_json):
    """Reconstruct a saved picture in its own units before either draft migrates.

    Presentation fields still come from the figure's current appearance, but an old
    applied recipe must never inherit a witness naming a different pending metric.
    """
    from .plots_panel import plot_config_to_json, plot_config_from_json, migrate_plot_quantity
    draft_json = plot_config_to_json(raw_draft)
    merged = {**draft_json, **dict(applied_json), "style": draft_json["style"]}
    applied_axes = dict(applied_json.get("axes", {}))
    axes = {**draft_json["axes"], **applied_axes}
    if "quantity_display" not in applied_axes:
        axes.pop("quantity_display", None)
    merged["axes"] = axes
    return migrate_plot_quantity(plot_config_from_json(merged))


def _native_config(artifact, *, migrate=True):
    from ..core.analysis_workspace import _decode_like
    from .plots_panel import PlotConfig, migrate_plot_quantity
    config = _decode_like(PlotConfig(), artifact.payload["config"], artifact)
    return migrate_plot_quantity(config) if migrate else config


def _with_native_config(artifact, config):
    from dataclasses import replace
    from ..core.analysis_workspace import _encode, _Resources
    payload = dict(artifact.payload)
    resources = _Resources()
    payload["config"] = _encode(config, resources, "config")
    if resources.tables or resources.blobs:
        raise FigureValidationError("A plot configuration unexpectedly contains resources")
    return replace(artifact, payload=payload)


def _restyled_config(config, look, keep_hand):
    """The same value transform as PlotsWindow.set_appearance_state, with no widgets."""
    from dataclasses import replace
    from .plot_appearance import PlotAppearance
    from .plots_panel import PlotsWindow, StyleSettings, FrozenMap, _NO_ENTRIES, _text_kept_in_styles

    before = PlotAppearance.from_mapping(PlotsWindow._appearance_of(config.style))
    state = before.restyled(look, keep_hand=keep_hand)
    defaults = StyleSettings()
    values = {}
    for key, field in PlotsWindow._APPEARANCE_FIELDS.items():
        value = getattr(state, key)
        value = getattr(defaults, field) if value is None else value
        if isinstance(getattr(defaults, field), int) and not isinstance(value, bool):
            value = int(round(value))
            if field in ("line_width", "point_size"):
                value = max(1, value)
        current = getattr(config.style, field)
        if (field in ("data_color", "fit_color") and isinstance(value, str)
                and isinstance(current, str) and value.lower() == current.lower()):
            value = current
        values[field] = value
    values["text_styles"] = FrozenMap({key: (style if _text_kept_in_styles(key)
        else replace(style, text=None)).to_dict() for key, style in (state.text or ())})
    values["fit_label"] = FrozenMap(state.fit_label.to_dict()) if state.fit_label is not None else _NO_ENTRIES
    from .plots_panel import _deep_frozen
    values["style_binding"] = _deep_frozen(state.to_dict().get("style_binding", {}))
    values["series_styles"] = _deep_frozen(state.to_dict().get("series_styles", {}))
    from .plots_panel import _project_style_settings, _encode_dc, _decode_dc
    from .figure_style_binding import preserve_legacy_baseline
    native = preserve_legacy_baseline(_encode_dc(replace(config.style, **values)),
                                      _encode_dc(config.style), "style_settings")
    return replace(config, style=_project_style_settings(_decode_dc(StyleSettings, native)))


def _check_history(capsule, config):
    from ..core.figure_history_state import HistoryCapsule
    from ..core.figure_spec import thaw_json
    from .plots_panel import plot_config_to_json, plot_config_from_json, migrate_plot_quantity
    from dataclasses import replace
    if not isinstance(capsule, HistoryCapsule):
        capsule = HistoryCapsule.from_dict(capsule)
    if capsule.codec == "nanoscope.plot-config-history/1":
        migrate = lambda value: plot_config_to_json(migrate_plot_quantity(plot_config_from_json(thaw_json(value))))
        capsule = replace(capsule, base=migrate(capsule.base),
            past=tuple(migrate(value) for value in capsule.past), future=tuple(migrate(value) for value in capsule.future))
    if capsule.codec != "nanoscope.plot-config-history/1" or thaw_json(capsule.base) != plot_config_to_json(config):
        raise FigureValidationError("Dormant configuration history disagrees with the figure")
    extra = thaw_json(capsule.extra)
    if (len(extra.get("past_origins", ())) != len(capsule.past)
            or len(extra.get("future_origins", ())) != len(capsule.future)):
        raise FigureValidationError("Dormant configuration history has invalid origin arrays")
    return capsule


def apply_look_to_record(record, look, transient, *, keep_hand=True):
    from dataclasses import replace
    from ..core.figure_spec import thaw_json
    from ..core.figure_codecs import encode_legacy, merge_native_capture
    from .figure_adapter_pages import _preserve_view
    from .plots_panel import plot_config_to_json

    artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    before = _native_config(artifact)
    transient = _check_history(transient, before)
    after = _restyled_config(before, look, keep_hand)
    if after == before:
        return record, transient
    extra = thaw_json(transient.extra)
    moved = transient.note(plot_config_to_json(after), limit=extra["depth"])
    origins = extra["past_origins"] + [extra["origin"]]
    extra["past_origins"] = origins[-len(moved.past):]
    extra["future_origins"] = []
    result = merge_native_capture(_with_native_config(artifact, after), record)
    return _preserve_view(result, record), replace(moved, extra=extra)


def move_history(record, capsule, direction="undo"):
    from dataclasses import replace
    from ..core.figure_spec import thaw_json
    from ..core.figure_codecs import encode_legacy, merge_native_capture
    from .figure_adapter_pages import _preserve_view, NeedsResidentHistory
    from .plots_panel import plot_config_from_json

    if direction not in ("undo", "redo"):
        raise FigureValidationError("History direction must be undo or redo")
    artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    before = _native_config(artifact)
    capsule = _check_history(capsule, before)
    moved = capsule.undo() if direction == "undo" else capsule.redo()
    if moved is capsule:
        return record, capsule
    after = plot_config_from_json(thaw_json(moved.base))
    if replace(after, style=before.style) != before:
        raise NeedsResidentHistory("This configuration step must run through the live plot's Undo")
    extra = thaw_json(moved.extra)
    if direction == "undo":
        extra["future_origins"].append(extra["origin"])
        extra["origin"] = extra["past_origins"].pop()
    else:
        extra["past_origins"].append(extra["origin"])
        extra["origin"] = extra["future_origins"].pop()
    surface = extra.get("surface")
    if surface is not None:
        from .quantity_reset import stands, typed_words
        took = surface["memos"].get("_reset_took", ())
        if took and not stands(took, typed_words(after.style) + after.axes.ranges()):
            if surface["status"] == surface["memos"].get("_reset_said"):
                surface["status"] = ""
            surface["memos"].update(_reset_said="", _reset_took=[])
    result = merge_native_capture(_with_native_config(artifact, after), record)
    return _preserve_view(result, record), replace(moved, extra=extra)


_SOURCE_FIELDS = ("key", "label", "doc_key", "provenance", "colour", "doc_id", "channel",
                  "module_id", "kind", "is_segment", "display_name", "calibration_fingerprint",
                  "current", "warnings")
_SOURCE_BYTES = 256 * 1024 * 1024


def _capture_sources(page, artifact):
    """Own exact inputs used by the drawn result and its pending draft, with provenance."""
    from ..core.analysis_workspace import _plot_source_recipe, _encode, _Resources
    resources = _Resources()
    wanted = set(page.selected_source_keys()) | set(page._figure_sources) | set(page._fits)
    snapshots, total = [], 0
    for key, source in page._sources.items():
        if key not in wanted:
            continue
        frame = source.df
        total += int(frame.memory_usage(index=True, deep=True).sum())
        if total > _SOURCE_BYTES:
            raise FigureValidationError("Figure frozen inputs exceed the 256 MiB checkpoint bound")
        table = resources.table(frame, f"phase2.sources.{len(snapshots)}")
        metadata = {field: getattr(source, field) for field in _SOURCE_FIELDS}
        snapshots.append({"key": key, "table": table,
                          "metadata": _encode(metadata, resources, f"phase2.source_meta.{len(snapshots)}"),
                          "recipe": _plot_source_recipe(page, source)})
    if set(resources.tables) & set(artifact.tables) or set(resources.blobs) & set(artifact.blobs):
        raise FigureValidationError("Frozen input resource names collide")
    return snapshots, resources.tables, resources.blobs


def _frozen_source_recipe(page, source):
    from ..core.analysis_workspace import fingerprint_frame
    raw = getattr(page, "_phase2_source_recipes", {}).get(source.key)
    if raw is not None and fingerprint_frame(source.df) == raw.get("event_frame_fingerprint"):
        return dict(raw)
    return None


def _install_sources(page, render, artifact):
    import weakref
    from ..core.analysis_workspace import _decode, _resolve_source, fingerprint_frame
    from ..core.sources import PlotSource
    from ..core.figure_spec import canonical_json

    snapshots = render.get("source_snapshots", ())
    if not isinstance(snapshots, (list, tuple)):
        raise FigureValidationError("Invalid frozen input catalog")
    sources, recipes, identities, total = dict(page._sources), {}, {}, 0
    for row in snapshots:
        table = artifact.tables.get(row.get("table"))
        if table is None:
            raise FigureValidationError("A frozen figure input table is missing")
        total += int(table.memory_usage(index=True, deep=True).sum())
        if total > _SOURCE_BYTES:
            raise FigureValidationError("Frozen figure inputs exceed the checkpoint bound")
        recipe = row["recipe"]
        if fingerprint_frame(table) != recipe.get("event_frame_fingerprint"):
            raise FigureValidationError("A frozen figure input fingerprint changed")
        metadata = _decode(row["metadata"], artifact.tables, artifact.blobs)
        if set(metadata) != set(_SOURCE_FIELDS) or metadata["key"] != row["key"]:
            raise FigureValidationError("Invalid frozen source metadata")
        # Frozen inputs remain usable without a live recording. Publication needs a real
        # current binding, so never preserve an old launch's doc_id when it cannot resolve.
        if metadata["kind"] != "manual":
            doc_id, channel, _warning = _resolve_source(page, recipe)
            metadata["doc_id"], metadata["channel"] = doc_id, channel
        source = PlotSource(df=table.copy(deep=True), **metadata)
        sources[source.key] = source
        recipes[source.key] = dict(recipe)
        signature = canonical_json(recipe)
        if signature in identities:
            raise FigureValidationError("Ambiguous frozen figure source identity")
        identities[signature] = source.key
    was_auto, was_armed = page.auto_plot, page._plot_armed
    page.auto_plot, page._plot_armed = False, False
    try:
        page.set_data(list(sources.values()))
    finally:
        page.auto_plot, page._plot_armed = was_auto, was_armed
    page._phase2_source_recipes = recipes
    ref = weakref.ref(page)
    page._frozen_plot_source_recipe = lambda source: _frozen_source_recipe(ref(), source) if ref() else None

    def resolve(recipe, *, require_same_rows=False):
        key = identities.get(canonical_json(recipe))
        if key is None:
            return None, "The figure's frozen source recipe is unavailable"
        return key, None

    page._resolve_frozen_plot_source = resolve
