"""Opt-in comparison figures with frozen, auditable drawing coordinates.

Only ``prepare_comparison`` computes explicit count histograms, the existing
NanoScope box summary, or strip/swarm placement. Rendering uses frozen tables;
it never detects, fits, estimates KDE/CI, samples, or accesses source recordings.
"""
from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
import json
import math
import warnings

import numpy as np
import pandas as pd

from nanoscope.analysis.distribution_summary import summarize_distribution
from nanoscope.core.source_style import compact_source_labels, validate_source_appearance
from nanoscope.plot.style import OKABE_ITO, nanoscope_style

SCHEMA_VERSION = 1
MAX_SOURCES = 8
MAX_OBSERVATIONS = 50_000
MAX_PAIR_OBSERVATIONS = 10_000
MAX_PLOTTED_POINTS = 250_000
MAX_SWARM_PER_SOURCE = 500
MAX_SWARM_OBSERVATIONS = 2_000
MAX_FEATURES = 5
POINT_COLUMNS = ["panel_id", "source_key", "source_row", "observation_index", "kind", "x", "y"]
HIST_COLUMNS = ["panel_id", "source_key", "bin_index", "left", "right", "count"]
BOX_COLUMNS = ["panel_id", "source_key", "center", "q1", "median", "q3", "lower_whisker", "upper_whisker"]
IDENTITY_COLUMNS = ["source_key", "source_row", "event_id", "observation_index"]


def _int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low:,} to {high:,}.")
    return value


def _text(value, name, maximum=1_024, *, empty=False):
    if not isinstance(value, str) or (not value and not empty) or len(value) > maximum:
        raise ValueError(f"{name} must be a {'nonempty ' if not empty else ''}string of at most {maximum} characters.")
    return value


def _source_color(value, index):
    from matplotlib.colors import is_color_like, to_hex
    if value is None:
        value = OKABE_ITO[index]
    if not isinstance(value, str) or len(value) > 100 or not is_color_like(value):
        raise ValueError("Recording colors must be valid color names or hex colors.")
    return to_hex(value)


def _config(config):
    if not isinstance(config, Mapping):
        raise ValueError("Comparison settings must be a mapping.")
    kind = config.get("kind", "facets")
    if kind not in ("facets", "pairs", "events"):
        raise ValueError("Choose faceted comparisons, a pair matrix, or individual events.")
    features = config.get("features", ["dwell_s", "depth_pA", "ecd_pA_s", "snr"])
    if not isinstance(features, (list, tuple)) or len(features) > 32:
        raise ValueError("Feature settings must be a list of at most 32 names.")
    for feature in features:
        _text(feature, "Feature", 200)
    if kind == "pairs" and (not 2 <= len(features) <= MAX_FEATURES or len(set(features)) != len(features)):
        raise ValueError("Select two to five different features for the pair matrix.")
    # The pair-matrix draft is retained while another kind is shown. An
    # incomplete hidden draft must not disable valid facets or event overlays.
    if kind != "pairs":
        features = list(dict.fromkeys(features))
    result = dict(kind=kind, x=config.get("x", "dwell_s"), y=config.get("y", "depth_pA"),
                  metric=config.get("metric", "depth_pA"), features=list(features),
                  bins=_int(config.get("bins", 40), "Bin count", 2, 200),
                  columns=_int(config.get("columns", 2), "Facet columns", 1, 4),
                  points=config.get("points", "strip"), show_box=config.get("show_box", False),
                  log_x=config.get("log_x", False), log_y=config.get("log_y", False))
    for name in ("x", "y", "metric"):
        _text(result[name], name, 200)
    if result["points"] not in ("strip", "swarm"):
        raise ValueError("Individual-event layout must be strip or swarm.")
    for name in ("show_box", "log_x", "log_y"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be true or false.")
    if kind == "events" and result["log_x"]:
        raise ValueError("The individual-event horizontal axis is categorical; use log Y for the measured feature.")
    labels = config.get("labels", {})
    if not isinstance(labels, Mapping) or len(labels) > 100:
        raise ValueError("Feature labels must be a bounded mapping.")
    result["labels"] = {_text(key, "Label key", 200): _text(value, "Label", 400, empty=True)
                        for key, value in labels.items()}
    return result


def _features(config):
    if config["kind"] == "pairs":
        return list(config["features"])
    if config["kind"] == "events":
        return [config["metric"]]
    return list(dict.fromkeys([config["x"], config["y"]]))


def _numbers(series, name):
    if (not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)):
        raise ValueError(f"{name} must contain real numeric observations; text values are not coerced.")
    if pd.api.types.is_integer_dtype(series.dtype) and (
            (series.dropna() > 2**53).any() or (series.dropna() < -(2**53)).any()):
        raise ValueError(f"{name} contains integers beyond exact floating-axis precision (2**53).")
    return series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)


def _limits(values, log=False, *, count=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if log:
        values = values[values > 0]
    if not len(values):
        return [1.0, 10.0] if log else [0.0, 1.0]
    low, high = float(values.min()), float(values.max())
    if count and high == 0:
        return [0.0, 1.0]
    if log:
        loglow, loghigh = math.log10(low), math.log10(high)
        gap = max((loghigh - loglow) * .04, .025)
        result = [10.0 ** max(-300., loglow - gap), 10.0 ** min(300., loghigh + gap)]
    else:
        gap = max((high - low) * .05, abs(low) * .025 if high == low else 0., 1e-12)
        result = [0.0 if count else low - gap, high + gap]
    if not all(math.isfinite(v) for v in result) or not result[0] < result[1]:
        raise ValueError("The selected values exceed the supported finite plot range; choose another feature.")
    if low < result[0] or high > result[1]:
        raise ValueError("The selected values exceed the supported plot range; no values were clipped.")
    return result


def _included(part, x, y, log_x, log_y):
    xv = part[x].to_numpy(dtype=float)
    yv = part[y].to_numpy(dtype=float) if y else np.zeros(len(part))
    finite_x, finite_y = np.isfinite(xv), np.isfinite(yv)
    finite = finite_x & finite_y
    log_bad = finite & (((xv <= 0) if log_x else False) | ((yv <= 0) if log_y else False))
    mask = finite & ~log_bad
    counts = dict(total=len(part), included=int(mask.sum()), excluded_nonfinite=int((~finite).sum()),
                  excluded_log=int(log_bad.sum()), nonfinite_x=int((~finite_x).sum()), nonfinite_y=int((~finite_y).sum()))
    return mask, xv, yv, counts


def _swarm_positions(groups, ylim, log_y, width):
    """Use Seaborn once; copy final offsets, never retain its draw callback."""
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, 5.2), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale("log")
    frame = pd.DataFrame({"group": np.concatenate([np.full(len(v), i) for i, v in enumerate(groups)]),
                          "value": np.concatenate(groups)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sns.swarmplot(data=frame, x="group", y="value", order=list(range(len(groups))), color=OKABE_ITO[0],
                      size=3.5, linewidth=0, warn_thresh=0, ax=ax)
        canvas.draw()
    offsets = [np.asarray(collection.get_offsets(), dtype=float).copy() for collection in ax.collections]
    fig.clear()
    if any("cannot be placed" in str(item.message) for item in caught):
        raise ValueError("This swarm is too crowded to place every marker. Choose Strip, fewer sources, "
                         "or a smaller explicit event selection. No events were sampled or dropped.")
    if len(offsets) != len(groups):
        raise ValueError("Seaborn returned an unexpected swarm layout. Choose Strip instead.")
    for i, (xy, values) in enumerate(zip(offsets, groups)):
        if xy.shape != (len(values), 2) or not np.allclose(xy[:, 1], values, rtol=1e-12, atol=0):
            raise ValueError("Swarm layout did not preserve observation identity; choose Strip instead.")
        # Only categorical X offsets are retained: log-axis round trips must
        # never round the original measured Y values in the saved tables.
        if len(xy) and (not np.isfinite(xy).all() or np.max(np.abs(xy[:, 0] - i)) > .5):
            raise ValueError("Seaborn returned out-of-bounds swarm coordinates.")
    return [xy[:, 0] for xy in offsets]


def _append_points(rows, subset, panel_id, source_key, xs, ys, kind):
    for source_row, observation_index, x, y in zip(subset["source_row"], subset["observation_index"], xs, ys):
        rows.append(dict(panel_id=panel_id, source_key=source_key, source_row=int(source_row),
                         observation_index=int(observation_index), kind=kind, x=float(x), y=float(y)))


def prepare_comparison(sources, config):
    """Copy selected observations; freeze explicit counts/summaries/placement.

    Frames must have unique original positional ``_nanoscope_row`` identities.
    Missing features are disclosed per panel; pairs never use a global
    complete-case drop. Sources and their input frames remain unchanged.
    """
    cfg = _config(config)
    features = _features(cfg)
    if any(name in IDENTITY_COLUMNS or name == "_nanoscope_row" for name in features):
        raise ValueError("Source identity columns cannot be plotted as measured features.")
    if not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError(f"Select one to {MAX_SOURCES} recordings; no sources are sampled automatically.")
    budget = MAX_PAIR_OBSERVATIONS if cfg["kind"] == "pairs" else MAX_OBSERVATIONS
    if not all(isinstance(source, Mapping) and isinstance(source.get("frame"), pd.DataFrame) for source in sources):
        raise ValueError("Each source must contain a key, label, and event DataFrame.")
    total = sum(len(source["frame"]) for source in sources)
    if not 1 <= total <= budget:
        raise ValueError(f"This comparison supports 1–{budget:,} selected events. Select fewer events "
                         "or recordings; no automatic sampling is performed.")
    source_labels = config.get("source_labels", {})
    if not isinstance(source_labels, Mapping):
        raise ValueError("Source labels must be a mapping.")
    metadata, frames, keys = [], [], set()
    offset = 0
    for i, source in enumerate(sources):
        key = _text(source.get("key"), "Source key")
        label = _text(source_labels.get(key, source.get("label", key)), "Source label", 4_096)
        if key in keys:
            raise ValueError("Each selected recording must have a unique source key.")
        keys.add(key)
        frame = source["frame"]
        if not frame.columns.is_unique:
            raise ValueError(f"{label}: an event DataFrame with unique columns is required.")
        if "_nanoscope_row" not in frame:
            raise ValueError(f"{label}: original event-row identities are required.")
        rows = frame["_nanoscope_row"]
        if (not pd.api.types.is_integer_dtype(rows.dtype) or rows.isna().any() or
                (rows < 0).any() or not rows.is_unique or (rows > 2**53).any()):
            raise ValueError(f"{label}: original event-row identities must be unique nonnegative integers.")
        missing = [name for name in features if name not in frame]
        data = {"source_key": [key] * len(frame), "source_row": rows.to_numpy(copy=True),
                "event_id": pd.Series(frame["event_id"].to_numpy(copy=True), dtype=object) if "event_id" in frame else pd.Series([None] * len(frame), dtype=object),
                "observation_index": np.arange(offset, offset + len(frame), dtype=np.int64)}
        for name in features:
            data[name] = _numbers(frame[name], f"{label}: {name}") if name in frame else np.full(len(frame), np.nan)
        frames.append(pd.DataFrame(data))
        metadata.append(dict(key=key, label=label, n=len(frame), missing_columns=missing,
                             color=_source_color(source.get("color"), i)))
        offset += len(frame)
    display_labels = compact_source_labels(metadata)
    for source in metadata:
        source["plot_label"] = display_labels[source["key"]]
    observations = pd.concat(frames, ignore_index=True)
    panels, point_rows, hist_rows, box_rows, notes = [], [], [], [], []
    if cfg["kind"] == "facets":
        columns = min(cfg["columns"], len(sources))
        rows = math.ceil(len(sources) / columns)
        layout = dict(rows=rows, columns=columns, width=max(6.8, 3.2 * columns), height=max(3.5, 2.8 * rows))
        definitions = [(f"facet:{i}", i // columns, i % columns, cfg["x"], cfg["y"], [source])
                       for i, source in enumerate(metadata)]
    elif cfg["kind"] == "pairs":
        n = len(features)
        layout = dict(rows=n, columns=n, width=max(7.2, 2.25 * n), height=max(7.2, 2.1 * n))
        definitions = [(f"pair:{r}:{c}", r, c, x, None if r == c else y, metadata)
                       for r, y in enumerate(features) for c, x in enumerate(features)]
        if len(observations) * n * n > MAX_PLOTTED_POINTS:
            raise ValueError(f"The matrix would exceed {MAX_PLOTTED_POINTS:,} panel observations. "
                             "Select fewer features or events; no automatic sampling is performed.")
        notes.append("Pairwise finite observations are used separately in every panel; no global complete-case filter is applied.")
        notes.append("Diagonal panels show event counts in explicit, common bins per feature, not probability density or KDE.")
    else:
        layout = dict(rows=1, columns=1, width=max(6.8, .95 * len(sources)), height=5.2)
        definitions = [("events", 0, 0, cfg["metric"], None, metadata)]
    for panel_id, row, col, x, y, panel_sources in definitions:
        is_events = cfg["kind"] == "events"
        diag = cfg["kind"] == "pairs" and y is None
        xlog, ylog = (cfg["log_y"], False) if is_events else (cfg["log_x"], cfg["log_y"] if y else False)
        panel = dict(id=panel_id, row=row, col=col, kind="events" if is_events else "hist" if diag else "scatter",
                     x=x, y=y, title=panel_sources[0]["plot_label"] if cfg["kind"] == "facets" else "",
                     x_label="Recording" if is_events else cfg["labels"].get(x, x),
                     y_label=cfg["labels"].get(x if is_events else y, x if is_events else y) if not diag else "Event count",
                     xscale="log" if (cfg["log_x"] and not is_events) else "linear",
                     yscale="log" if cfg["log_y"] and (is_events or y) else "linear", counts=[])
        selected, groups = [], []
        for source in panel_sources:
            part = observations.loc[observations["source_key"] == source["key"]]
            mask, xv, yv, counts = _included(part, x, y, xlog, ylog)
            counts["source_key"] = source["key"]
            panel["counts"].append(counts)
            subset = part.loc[mask]
            xx, yy = xv[mask], yv[mask]
            groups.append(xx)
            selected.append((source, subset, xx, yy))
        all_x = np.concatenate(groups)
        if is_events:
            panel["xlim"], panel["ylim"] = [-.6, len(metadata) - .4], _limits(all_x, cfg["log_y"])
            if cfg["points"] == "swarm":
                if len(all_x) > MAX_SWARM_OBSERVATIONS or any(len(group) > MAX_SWARM_PER_SOURCE for group in groups):
                    raise ValueError(f"Swarm layout supports at most {MAX_SWARM_PER_SOURCE:,} events per recording "
                                     f"and {MAX_SWARM_OBSERVATIONS:,} total. Choose Strip or select fewer events; "
                                     "no automatic sampling is performed.")
                positions = _swarm_positions(groups, panel["ylim"], cfg["log_y"], layout["width"]) if len(all_x) else [np.array([])] * len(groups)
                notes.append("Swarm offsets were frozen once at 3.5 pt markers, 100 DPI and the saved canvas size. "
                             "Larger markers or a smaller export can overlap; appearance never re-swarms the observations.")
            else:
                rng = np.random.default_rng(0)
                positions = [i + rng.uniform(-.2, .2, len(group)) for i, group in enumerate(groups)]
                notes.append("Strip offsets are deterministic uniform jitter in ±0.2 category units (NumPy generator seed 0), frozen on Generate.")
            for i, (source, subset, xx, yy) in enumerate(selected):
                if cfg["show_box"] and len(xx):
                    summary = summarize_distribution(xx, kind="box")
                    box_rows.append(dict(panel_id=panel_id, source_key=source["key"], center=float(i),
                                         **{key: float(summary.stats[key]) for key in BOX_COLUMNS[3:]}))
                    notes.extend(f"{source['label']}: {warning}" for warning in summary.warnings)
                _append_points(point_rows, subset, panel_id, source["key"], positions[i], xx, "point")
            if cfg["show_box"]:
                notes.append("Box overlay: existing NanoScope linear-interpolation quartiles and actual-sample 1.5-IQR whiskers; "
                             "all retained events, including outliers, remain visible. No uncertainty interval is estimated.")
        elif diag:
            limits = _limits(all_x, cfg["log_x"])
            edges = np.geomspace(*limits, cfg["bins"] + 1) if cfg["log_x"] else np.linspace(*limits, cfg["bins"] + 1)
            if not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
                raise ValueError("Distinct finite common histogram edges cannot be represented for this feature.")
            for source, subset, xx, yy in selected:
                counts, _ = np.histogram(xx, bins=edges)
                for index, count in enumerate(counts):
                    hist_rows.append(dict(panel_id=panel_id, source_key=source["key"], bin_index=index,
                                          left=float(edges[index]), right=float(edges[index + 1]), count=int(count)))
                _append_points(point_rows, subset, panel_id, source["key"], xx, np.zeros(len(xx)), "sample")
            panel["xlim"] = limits
            panel["ylim"] = _limits([item["count"] for item in hist_rows if item["panel_id"] == panel_id], count=True)
        else:
            for source, subset, xx, yy in selected:
                _append_points(point_rows, subset, panel_id, source["key"], xx, yy, "point")
            panel["xlim"] = _limits(all_x, cfg["log_x"])
            panel["ylim"] = _limits(np.concatenate([values[3] for values in selected]), cfg["log_y"])
        panels.append(panel)
    if not point_rows:
        raise ValueError("No observations can be shown with these features and axis scales. Choose other features or linear axes.")
    if cfg["kind"] == "facets":
        xlim = _limits([point["x"] for point in point_rows], cfg["log_x"])
        ylim = _limits([point["y"] for point in point_rows], cfg["log_y"])
        for panel in panels:
            panel["xlim"], panel["ylim"] = list(xlim), list(ylim)
    elif cfg["kind"] == "pairs":
        for feature in features:
            same_x = [p for p in panels if p["x"] == feature]
            # An empty off-diagonal panel's fallback limits must not widen
            # every panel for a feature (e.g. 0.001–0.002 seconds to 0–1).
            limits = next(p["xlim"] for p in same_x if p["kind"] == "hist")
            for panel in same_x:
                panel["xlim"] = limits.copy()
            same_y = [p for p in panels if p["y"] == feature]
            if same_y:
                limits = _limits(observations[feature].to_numpy(), cfg["log_y"])
                for panel in same_y:
                    panel["ylim"] = limits.copy()
    exclusions = []
    for panel in panels:
        for counts in panel["counts"]:
            if counts["excluded_nonfinite"] or counts["excluded_log"]:
                exclusions.append(f"{panel['id']} / {counts['source_key']}: {counts['included']}/{counts['total']} shown; "
                                  f"{counts['excluded_nonfinite']} non-finite, {counts['excluded_log']} non-positive on log axes excluded.")
    notes.insert(0, "These are unweighted, descriptive views of caller-selected event observations. No censoring, "
                 "missed-event, calibration, detection, fitting, or biological-replicate correction is inferred.")
    notes.extend(exclusions)
    details = ["Comparison figure", *[f"{s['plot_label']} = {s['label']} [source key: {s['key']}; {s['n']} selected events]" for s in metadata],
               "", *notes, "", "Per-panel counts (exclusions are not imputed):"]
    for panel in panels:
        for counts in panel["counts"]:
            details.append(f"{panel['id']} / {counts['source_key']}: total={counts['total']}, shown={counts['included']}, "
                           f"nonfinite={counts['excluded_nonfinite']}, log_nonpositive={counts['excluded_log']}")
    versions = {}
    for package in ("seaborn", "matplotlib", "numpy", "pandas", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    snapshot = dict(schema_version=SCHEMA_VERSION, config=cfg, sources=metadata, features=features,
                    panels=panels, layout=layout, warnings=list(dict.fromkeys(notes)), details="\n".join(details),
                    versions=versions, no_automatic_sampling=True, frozen_geometry=True)
    tables = dict(observations=observations, plotted=pd.DataFrame(point_rows, columns=POINT_COLUMNS),
                  histograms=pd.DataFrame(hist_rows, columns=HIST_COLUMNS), boxes=pd.DataFrame(box_rows, columns=BOX_COLUMNS))
    validate_comparison(snapshot, tables)
    return snapshot, tables


def validate_comparison(snapshot, tables):
    """Read-only shape, bound and identity checks, without recalculating plots."""
    try:
        _validate(snapshot, tables)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError(f"Invalid saved comparison: {exc}") from exc
    return True


def _validate(snapshot, tables):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported saved comparison schema.")
    if len(json.dumps(snapshot, allow_nan=False)) > 1_000_000:
        raise ValueError("Saved comparison metadata exceeds its size limit.")
    cfg = _config(snapshot["config"])
    if cfg != snapshot["config"] or snapshot["features"] != _features(cfg):
        raise ValueError("Saved comparison settings are inconsistent.")
    if snapshot.get("no_automatic_sampling") is not True or snapshot.get("frozen_geometry") is not True:
        raise ValueError("Saved comparison must contain frozen, unsampled observations.")
    sources = snapshot["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError("Invalid saved comparison source count.")
    keys = [_text(source["key"], "Source key") for source in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("Saved source keys are not unique.")
    plot_labels = [_text(source["plot_label"], "Plot label", 40) for source in sources]
    if len(set(plot_labels)) != len(plot_labels):
        raise ValueError("Saved plot labels must be unique.")
    for index, source in enumerate(sources):
        _text(source["label"], "Source label", 4_096)
        _int(source["n"], "Source observation count", 0, MAX_OBSERVATIONS)
        if _source_color(source["color"], index) != source["color"]:
            raise ValueError("Saved recording colors must be normalized hex colors.")
        missing = source["missing_columns"]
        if (not isinstance(missing, list) or any(not isinstance(v, str) for v in missing)
                or len(set(missing)) != len(missing) or not set(missing).issubset(snapshot["features"])):
            raise ValueError("Invalid saved missing-feature metadata.")
    if not isinstance(tables, Mapping) or set(tables) != {"observations", "plotted", "histograms", "boxes"}:
        raise ValueError("Saved comparison requires observations, plotted, histograms and boxes tables.")
    for name, columns in (("observations", IDENTITY_COLUMNS + snapshot["features"]),
                          ("plotted", POINT_COLUMNS), ("histograms", HIST_COLUMNS), ("boxes", BOX_COLUMNS)):
        frame = tables[name]
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns or not frame.columns.is_unique:
            raise ValueError(f"Invalid saved {name} table schema.")
        if len(frame) > (MAX_PLOTTED_POINTS if name == "plotted" else MAX_OBSERVATIONS):
            raise ValueError(f"Saved {name} table exceeds the supported row count.")
    obs, points, hist, boxes = [tables[name] for name in ("observations", "plotted", "histograms", "boxes")]
    if not len(obs) or not len(points) or len(obs) != sum(source["n"] for source in sources):
        raise ValueError("Saved comparison observation counts are inconsistent or empty.")
    if cfg["kind"] == "pairs" and (len(obs) > MAX_PAIR_OBSERVATIONS or len(obs) * len(snapshot["features"])**2 > MAX_PLOTTED_POINTS):
        raise ValueError("Saved pair matrix exceeds its explicit event budget.")
    for col in ("source_row", "observation_index"):
        values = _numbers(obs[col], col)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Invalid saved observation identity.")
    if not np.array_equal(obs["observation_index"].to_numpy(), np.arange(len(obs))):
        raise ValueError("Saved observation indices must be contiguous and ordered.")
    if obs.duplicated(["source_key", "source_row"]).any() or not obs["source_key"].isin(keys).all():
        raise ValueError("Saved observation source identities are invalid.")
    for feature in snapshot["features"]:
        _numbers(obs[feature], feature)
    for source in sources:
        if int((obs["source_key"] == source["key"]).sum()) != source["n"]:
            raise ValueError("Saved source counts do not match observations.")
    layout = snapshot["layout"]
    rows, cols = _int(layout["rows"], "Figure rows", 1, 8), _int(layout["columns"], "Figure columns", 1, 5)
    expected_cols = min(cfg["columns"], len(sources)) if cfg["kind"] == "facets" else len(snapshot["features"]) if cfg["kind"] == "pairs" else 1
    expected_rows = math.ceil(len(sources) / expected_cols) if cfg["kind"] == "facets" else expected_cols
    if (rows, cols) != (expected_rows, expected_cols):
        raise ValueError("Saved layout does not match the selected comparison kind.")
    for dim in ("width", "height"):
        if type(layout[dim]) not in (float, int) or not 1 <= layout[dim] <= 30:
            raise ValueError("Invalid saved figure dimensions.")
    panels = snapshot["panels"]
    expected_panels = len(sources) if cfg["kind"] == "facets" else len(snapshot["features"])**2 if cfg["kind"] == "pairs" else 1
    if not isinstance(panels, list) or len(panels) != expected_panels:
        raise ValueError("Invalid saved panel count.")
    ids, positions = set(), set()
    for panel in panels:
        panel_id = _text(panel["id"], "Panel id", 100)
        position = (_int(panel["row"], "Panel row", 0, rows - 1), _int(panel["col"], "Panel column", 0, cols - 1))
        if panel_id in ids or position in positions:
            raise ValueError("Duplicate saved comparison panel.")
        ids.add(panel_id)
        positions.add(position)
        if panel["kind"] not in ("events", "hist", "scatter") or panel["x"] not in snapshot["features"]:
            raise ValueError("Invalid saved panel feature or kind.")
        if panel["y"] is not None and panel["y"] not in snapshot["features"]:
            raise ValueError("Invalid saved panel Y feature.")
        if cfg["kind"] == "facets":
            expected = (f"facet:{panel['row'] * cols + panel['col']}", "scatter", cfg["x"], cfg["y"])
        elif cfg["kind"] == "pairs":
            diagonal = panel["row"] == panel["col"]
            expected = (f"pair:{panel['row']}:{panel['col']}", "hist" if diagonal else "scatter",
                        snapshot["features"][panel["col"]], None if diagonal else snapshot["features"][panel["row"]])
        else:
            expected = ("events", "events", cfg["metric"], None)
        if (panel_id, panel["kind"], panel["x"], panel["y"]) != expected:
            raise ValueError("Saved panel geometry is inconsistent with its comparison settings.")
        expected_xscale = "log" if cfg["log_x"] and cfg["kind"] != "events" else "linear"
        expected_yscale = "log" if cfg["log_y"] and panel["kind"] != "hist" else "linear"
        if (panel["xscale"], panel["yscale"]) != (expected_xscale, expected_yscale):
            raise ValueError("Saved axis scales do not match the selected settings.")
        for axis in ("x", "y"):
            if panel[axis + "scale"] not in ("linear", "log"):
                raise ValueError("Invalid saved axis scale.")
            bounds = panel[axis + "lim"]
            if (not isinstance(bounds, list) or len(bounds) != 2 or
                    not all(type(v) in (int, float) and math.isfinite(v) for v in bounds) or
                    not bounds[0] < bounds[1] or (panel[axis + "scale"] == "log" and bounds[0] <= 0)):
                raise ValueError("Invalid saved finite axis bounds.")
            _text(panel[axis + "_label"], "Axis label", 400, empty=True)
        _text(panel["title"], "Panel title", 100, empty=True)
    for name, frame, numeric in (("plotted", points, ["source_row", "observation_index", "x", "y"]),
                                 ("histograms", hist, ["bin_index", "left", "right", "count"]),
                                 ("boxes", boxes, BOX_COLUMNS[2:])):
        if len(frame) and (not frame["panel_id"].isin(ids).all() or not frame["source_key"].isin(keys).all()):
            raise ValueError(f"Unknown source or panel in saved {name} geometry.")
        for col in numeric:
            if len(frame) and not np.isfinite(_numbers(frame[col], col)).all():
                raise ValueError(f"Non-finite saved {name} geometry.")
    indices = points["observation_index"].to_numpy(dtype=float)
    if (indices != np.floor(indices)).any() or (indices < 0).any() or (indices >= len(obs)).any():
        raise ValueError("Saved plotted observation indices are out of range.")
    referenced = obs.iloc[indices.astype(int)].reset_index(drop=True)
    if (not np.array_equal(referenced["source_key"], points["source_key"]) or
            not np.array_equal(referenced["source_row"], points["source_row"]) or
            points.duplicated(["panel_id", "observation_index"]).any()):
        raise ValueError("Saved plotted source identities do not match observations.")
    for panel in panels:
        subset = points[points["panel_id"] == panel["id"]]
        refs = obs.iloc[subset["observation_index"].to_numpy(dtype=int)]
        expected_kind = "sample" if panel["kind"] == "hist" else "point"
        if not (subset["kind"] == expected_kind).all():
            raise ValueError("Saved point geometry has an inconsistent primitive kind.")
        if panel["kind"] == "events":
            expected_y = refs[panel["x"]].to_numpy(dtype=float)
            centers = subset["source_key"].map({key: i for i, key in enumerate(keys)}).to_numpy(dtype=float)
            if (np.abs(subset["x"].to_numpy() - centers) > .5).any():
                raise ValueError("Saved categorical offsets are out of bounds.")
        else:
            if not np.array_equal(subset["x"].to_numpy(), refs[panel["x"]].to_numpy(dtype=float)):
                raise ValueError("Saved X coordinates differ from the original observations.")
            expected_y = np.zeros(len(subset)) if panel["kind"] == "hist" else refs[panel["y"]].to_numpy(dtype=float)
        if not np.array_equal(subset["y"].to_numpy(), expected_y):
            raise ValueError("Saved Y coordinates differ from the original observations.")
        for axis in ("x", "y"):
            if panel["kind"] == "hist" and axis == "y":
                continue
            values = subset[axis].to_numpy()
            if (values < panel[axis + "lim"][0]).any() or (values > panel[axis + "lim"][1]).any():
                raise ValueError("Saved observation coordinates lie outside the saved axes.")
        counts = panel["counts"]
        wanted = [sources[panel["row"] * cols + panel["col"]]["key"]] if cfg["kind"] == "facets" else keys
        if not isinstance(counts, list) or [item["source_key"] for item in counts] != wanted:
            raise ValueError("Saved per-panel source count records are inconsistent.")
        for item in counts:
            for field in ("total", "included", "excluded_nonfinite", "excluded_log", "nonfinite_x", "nonfinite_y"):
                _int(item[field], field, 0, len(obs))
            n = int((subset["source_key"] == item["source_key"]).sum())
            total = sources[keys.index(item["source_key"])]["n"]
            if (item["included"] != n or item["total"] != total or
                    item["included"] + item["excluded_nonfinite"] + item["excluded_log"] != total):
                raise ValueError("Saved per-panel exclusions do not match the plotted observations.")
            part = obs[obs["source_key"] == item["source_key"]]
            is_events = panel["kind"] == "events"
            mask, _, _, actual = _included(part, panel["x"], panel["y"],
                                          cfg["log_y"] if is_events else cfg["log_x"],
                                          cfg["log_y"] if panel["y"] else False)
            if any(item[field] != actual[field] for field in actual):
                raise ValueError("Saved non-finite/log exclusion counts disagree with the observations.")
            expected_ids = part.loc[mask, "observation_index"].to_numpy()
            actual_ids = subset.loc[subset["source_key"] == item["source_key"], "observation_index"].to_numpy()
            if not np.array_equal(expected_ids, actual_ids):
                raise ValueError("Saved panel does not preserve all eligible observations in original order.")
            if is_events and cfg["points"] == "swarm" and n > MAX_SWARM_PER_SOURCE:
                raise ValueError("Saved swarm exceeds its per-recording event budget.")
        if panel["kind"] == "events" and cfg["points"] == "swarm" and len(subset) > MAX_SWARM_OBSERVATIONS:
            raise ValueError("Saved swarm exceeds its total event budget.")
    if cfg["kind"] != "pairs" and len(hist):
        raise ValueError("Unexpected histogram geometry.")
    for panel in [p for p in panels if p["kind"] == "hist"]:
        reference_edges = None
        for key in keys:
            bars = hist[(hist["panel_id"] == panel["id"]) & (hist["source_key"] == key)]
            if len(bars) != cfg["bins"] or not np.array_equal(bars["bin_index"], np.arange(cfg["bins"])):
                raise ValueError("Saved histogram bin indices are inconsistent.")
            left, right, count = [bars[name].to_numpy() for name in ("left", "right", "count")]
            if ((right <= left).any() or not np.array_equal(right[:-1], left[1:]) or
                    (count < 0).any() or (count != np.floor(count)).any()):
                raise ValueError("Invalid saved histogram edges or counts.")
            edges = np.r_[left, right[-1]]
            if reference_edges is not None and not np.array_equal(reference_edges, edges):
                raise ValueError("Histogram bins must be common across recordings.")
            reference_edges = edges
            if edges[0] < panel["xlim"][0] or edges[-1] > panel["xlim"][1] or count.max() > panel["ylim"][1]:
                raise ValueError("Saved histogram geometry lies outside the saved axes.")
            expected = next(item["included"] for item in panel["counts"] if item["source_key"] == key)
            if count.sum() != expected:
                raise ValueError("Saved histogram counts do not include every retained observation.")
    if len(hist) != sum(p["kind"] == "hist" for p in panels) * len(keys) * cfg["bins"]:
        raise ValueError("Unexpected saved histogram rows.")
    if len(boxes):
        if cfg["kind"] != "events" or not cfg["show_box"] or boxes.duplicated(["panel_id", "source_key"]).any():
            raise ValueError("Unexpected saved box summaries.")
        for box in boxes.to_dict("records"):
            if (box["center"] != keys.index(box["source_key"]) or
                    not box["q1"] <= box["median"] <= box["q3"] or
                    not box["lower_whisker"] <= box["median"] <= box["upper_whisker"]):
                raise ValueError("Invalid saved box summary geometry.")
            if any(not panels[0]["ylim"][0] <= box[field] <= panels[0]["ylim"][1] for field in BOX_COLUMNS[3:]):
                raise ValueError("Saved box summary lies outside the saved axes.")
    expected_boxes = sum(item["included"] > 0 for item in panels[0]["counts"]) if cfg["kind"] == "events" and cfg["show_box"] else 0
    if len(boxes) != expected_boxes:
        raise ValueError("Saved comparison is missing an expected box summary.")
    if cfg["kind"] == "events" and cfg["show_box"]:
        expected_keys = [item["source_key"] for item in panels[0]["counts"] if item["included"] > 0]
        if boxes["source_key"].tolist() != expected_keys:
            raise ValueError("Saved box summaries do not match their source order.")
    _text(snapshot["details"], "Details", 200_000)
    if not isinstance(snapshot["warnings"], list) or len(snapshot["warnings"]) > 500:
        raise ValueError("Invalid saved comparison warnings.")
    for note in snapshot["warnings"]:
        _text(note, "Warning", 10_000)


def render_comparison(snapshot, tables, appearance=None, *, source_appearance=None):
    """Draw frozen geometry only; warnings stay in Details/provenance."""
    validate_comparison(snapshot, tables)
    source_appearance = validate_source_appearance(source_appearance or {})
    import seaborn as sns
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    state = appearance.to_dict() if hasattr(appearance, "to_dict") else dict(appearance or {})
    dark = state.get("background") == "dark"
    foreground, background = ("#eeeeee", "#17191d") if dark else ("#202226", "#ffffff")
    title_size = state.get("title_size", 11.)
    axis_size, tick_size = state.get("axis_size", 9.), state.get("tick_size", 8.)
    marker_size, line_width = state.get("marker_size", 3.5), state.get("line_width", 1.0)
    layout, cfg = snapshot["layout"], snapshot["config"]
    colors = {source["key"]: source_appearance.get(source["key"], {}).get("color", source["color"])
              for source in snapshot["sources"]}
    # With no explicit label edit, old snapshots keep their exact saved labels.
    # A figure-local edit changes drawing only, never the committed geometry.
    display_labels = ({source["key"]: source["plot_label"] for source in snapshot["sources"]}
                      if not any("label" in entry for entry in source_appearance.values())
                      else compact_source_labels(snapshot["sources"], source_appearance))
    # Category colors preserve recording identities across all panels. Shared
    # single-color overrides are intentionally ignored for comparison figures.
    overrides = {"figure.facecolor": background, "axes.facecolor": background,
                 "axes.edgecolor": foreground, "axes.labelcolor": foreground, "text.color": foreground,
                 "xtick.color": foreground, "ytick.color": foreground, "axes.titlesize": title_size,
                 "axes.labelsize": axis_size, "xtick.labelsize": tick_size, "ytick.labelsize": tick_size,
                 # Keep short dwell-time ticks legible in narrow comparison
                 # panels while displaying the scale explicitly in axis units.
                 "axes.formatter.limits": (-3, 4)}
    if state.get("font_family"):
        overrides["font.family"] = state["font_family"]
    with nanoscope_style(**overrides):
        figure = Figure(figsize=(layout["width"], layout["height"]), dpi=100, layout="constrained")
        axes = figure.subplots(layout["rows"], layout["columns"], squeeze=False)
        active = set()
        for panel in snapshot["panels"]:
            ax = axes[panel["row"], panel["col"]]
            active.add((panel["row"], panel["col"]))
            ax.set_xscale(panel["xscale"])
            ax.set_yscale(panel["yscale"])
            ax.set_xlim(panel["xlim"])
            ax.set_ylim(panel["ylim"])
            if panel["kind"] == "hist":
                for source in snapshot["sources"]:
                    bars = tables["histograms"]
                    bars = bars[(bars["panel_id"] == panel["id"]) & (bars["source_key"] == source["key"])]
                    edges = np.r_[bars["left"].to_numpy(), bars["right"].iloc[-1]]
                    ax.stairs(bars["count"].to_numpy(), edges, color=colors[source["key"]], linewidth=line_width)
            else:
                for source in snapshot["sources"]:
                    points = tables["plotted"]
                    points = points[(points["panel_id"] == panel["id"]) & (points["source_key"] == source["key"])]
                    if len(points):
                        sns.scatterplot(x=points["x"].to_numpy(), y=points["y"].to_numpy(), ax=ax,
                                        color=colors[source["key"]], s=float(marker_size)**2,
                                        alpha=.75, linewidth=0, legend=False)
                for box in tables["boxes"][tables["boxes"]["panel_id"] == panel["id"]].to_dict("records"):
                    color, x, halfwidth = colors[box["source_key"]], box["center"], .28
                    ax.add_patch(Rectangle((x - halfwidth, box["q1"]), 2 * halfwidth, box["q3"] - box["q1"],
                                           fill=False, edgecolor=color, linewidth=line_width, zorder=3))
                    for low, high in ((box["lower_whisker"], box["q1"]), (box["q3"], box["upper_whisker"])):
                        ax.plot([x, x], [low, high], color=color, linewidth=line_width, zorder=3)
                    for level, width in ((box["median"], halfwidth), (box["lower_whisker"], halfwidth / 2),
                                         (box["upper_whisker"], halfwidth / 2)):
                        ax.plot([x - width, x + width], [level, level], color=color, linewidth=line_width, zorder=3)
            if panel["kind"] == "events":
                ax.set_xticks(range(len(snapshot["sources"])),
                              [display_labels[s["key"]] for s in snapshot["sources"]], rotation=25, ha="right")
            panel_title = (display_labels[panel["counts"][0]["source_key"]]
                           if cfg["kind"] == "facets" else panel["title"])
            ax.set_title(panel_title, fontsize=title_size)
            # Pair matrices always keep each feature's own unit-bearing label.
            ax.set_xlabel(panel["x_label"] if cfg["kind"] == "pairs" else state.get("x_label", panel["x_label"]), fontsize=axis_size)
            ax.set_ylabel(panel["y_label"] if cfg["kind"] == "pairs" else state.get("y_label", panel["y_label"]), fontsize=axis_size)
            if state.get("grid", False):
                ax.grid(True, alpha=.2)
            else:
                ax.grid(False)
            ax.tick_params(labelsize=tick_size)
        for row in range(layout["rows"]):
            for col in range(layout["columns"]):
                if (row, col) not in active:
                    axes[row, col].set_visible(False)
        title = state.get("title", {"facets": "Recording comparisons", "pairs": "Event-feature pair matrix",
                                    "events": "Individual-event comparison"}[cfg["kind"]])
        if title:
            figure.suptitle(title, fontsize=title_size)
        if state.get("legend_visible", True) and cfg["kind"] != "facets":
            handles = [Line2D([], [], linestyle="none", marker="o", markersize=marker_size,
                              color=colors[source["key"]], label=display_labels[source["key"]]) for source in snapshot["sources"]]
            figure.legend(handles=handles, loc="outside lower center", ncols=min(4, len(handles)),
                          fontsize=state.get("legend_size", 8), labelcolor=foreground)
    return figure
