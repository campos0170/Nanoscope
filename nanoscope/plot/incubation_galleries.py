"""Saved trace/galleries for incubation studies; source indices accompany every panel."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..analysis import event_features as EF
from ..analysis.cancel import check_cancelled


def _figure(rows=1, cols=1, *, width=12, height=4):
    fig = Figure(figsize=(width, height), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(rows, cols, squeeze=False)).ravel()
    return fig, axes


def _save(fig, folder, name, *, cancel=None):
    check_cancelled(cancel)
    path = Path(folder) / f"{name}.png"
    from .study_figure_style import capture_saved_figure
    if capture_saved_figure(fig, name):
        check_cancelled(cancel)
        return path
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    fig.clear()
    check_cancelled(cancel)
    return path


def _json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _ordered_picks(frame, order, count, separation_s):
    chosen, used = [], []
    for index in order:
        index = int(index)
        if index in chosen:
            continue
        onset = float(frame.iloc[index].onset_s)
        if all(abs(onset - other) > separation_s for other in used):
            chosen.append(index)
            used.append(onset)
            if len(chosen) == count:
                break
    return chosen


def _quantile_picks(events, key, count=50):
    frame = events.sort_values(key, kind="stable").reset_index(drop=True)
    wanted = min(count, len(frame))
    if not wanted:
        return frame
    targets = np.unique(np.linspace(0, len(frame) - 1, wanted).round().astype(int))
    picks = _ordered_picks(frame, [*targets, *range(len(frame))], wanted, .02)
    return frame.iloc[sorted(picks)].copy()


def _typical_picks(events, count=4):
    if not len(events):
        return events.copy()
    frame = events.reset_index(drop=True)
    log_dwell = np.log10(frame.dwell_ms.to_numpy(float))
    peak = frame.depth_peak_pA.to_numpy(float)
    distance = ((log_dwell - np.median(log_dwell)) / (np.std(log_dwell) + 1e-9)) ** 2
    distance += ((peak - np.median(peak)) / (np.std(peak) + 1e-9)) ** 2
    return frame.iloc[_ordered_picks(frame, np.argsort(distance), min(count, len(frame)), .05)].copy()


def _stats(analysis, cancel=None):
    check_cancelled(cancel)
    stats = EF.gate_scalars(analysis.detrended)
    check_cancelled(cancel)
    return stats


def _overview_indices(signal, max_bins=100_000):
    """Keep both extrema of each source block, in order, including the tail.

    Uniform sample strides can erase an entire short event. This display-only
    envelope shares the same source-index convention across both trace figures.
    Include recording endpoints so the visible trace spans the complete clock.
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n <= 2 * max_bins:
        return np.arange(n)
    block = max(1, math.ceil(n / max_bins))
    n_full = n // block
    head = signal[:n_full * block].reshape(n_full, block)
    starts = np.arange(n_full) * block
    indices = [starts + head.argmin(axis=1), starts + head.argmax(axis=1),
               np.array([0, n - 1])]
    tail_start = n_full * block
    if tail_start < n:
        tail = signal[tail_start:]
        indices.append(tail_start + np.array([tail.argmin(), tail.argmax()]))
    return np.unique(np.concatenate(indices))


def _event_minima(analysis, events, *, cancel=None):
    """Exact downward extrema of the unchanged inclusive assigned intervals."""
    signal, fs = analysis.detrended, float(analysis.fs_hz)
    bounds = events[["start_idx", "end_idx"]].to_numpy(float)
    if (not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds))
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] >= len(signal))
            or np.any(bounds[:, 1] < bounds[:, 0])):
        raise ValueError("Population event bounds do not address the prepared trace")
    starts, ends = bounds.astype(np.int64).T
    minima = []
    for start, end in zip(starts, ends):
        check_cancelled(cancel)
        minima.append(start + int(np.argmin(signal[start:end + 1])))
    minima = np.asarray(minima, dtype=np.int64)
    result = events.copy()
    result["onset_time_s"] = starts / fs
    result["onset_current_pA"] = signal[starts]
    result["marker_sample_idx"] = minima
    result["marker_time_s"] = minima / fs
    result["marker_current_pA"] = signal[minima]
    return result


def _panel_data(analysis, row, stats, *, local_baseline=True):
    signal = analysis.detrended
    fs = float(analysis.fs_hz)
    start, end = int(row.start_idx), int(row.end_idx)
    if not 0 <= start <= end < len(signal):
        raise ValueError(f"{analysis.tag}: event bounds do not address this prepared trace")
    end_exclusive = end + 1
    dwell = (end_exclusive - start) / fs * 1000
    padding = int((np.clip(1.5 * dwell, 1., 5.) if local_baseline else 2.) * .001 * fs)
    lo, hi = max(0, start - padding), min(len(signal), end_exclusive + padding)
    baseline = 0.
    baseline_rule = "prepared_detrended_without_recentering"
    if local_baseline:
        window = int(.020 * fs)
        left, right = max(0, start - window), min(len(signal), end_exclusive + window)
        quiet = stats.quiet[left:right].copy()
        quiet[start - left:end_exclusive - left] = False
        baseline = float(np.median(signal[left:right][quiet])) if quiet.sum() > 20 else stats.baseline_pA
        baseline_rule = "quiet_20ms_median" if quiet.sum() > 20 else "global_quiet_fallback"
    samples = np.arange(lo, hi)
    identity = str(row.get("event_id", f"{analysis.tag}:{start}:{end}"))
    info = dict(tag=analysis.tag, event_id=identity, start_idx=start, end_idx=end,
                end_convention="inclusive", display_start_idx=lo, display_end_exclusive=hi,
                dwell_ms=dwell, baseline_pA=baseline, baseline_rule=baseline_rule,
                noise_mad_pA=stats.noise_pA, onset_s=start / fs)
    points = pd.DataFrame({"sample_idx": samples, "time_from_onset_ms": (samples - start) / fs * 1000,
                           "prepared_detrended_pA": signal[lo:hi], "display_current_pA": signal[lo:hi] - baseline})
    return info, points


def _gallery(analysis, picks, stats, folder, name, title, *, cols=5, local_baseline=True, cancel=None):
    from .study_figure_style import bind_event_panel
    columns = min(cols, max(1, len(picks)))
    rows = max(1, math.ceil(len(picks) / columns))
    fig, axes = _figure(rows, columns, width=3.1 * columns, height=max(2.6, 2.15 * rows))
    for axis in axes:
        # Empty cells also reject legacy event titles; they have no event to bind.
        axis._nanoscope_event_panel_identity = ''
    maps, point_tables = [], []
    for panel, (_, row) in enumerate(picks.iterrows()):
        check_cancelled(cancel)
        info, points = _panel_data(analysis, row, stats, local_baseline=local_baseline)
        info.update(panel=panel + 1, depth_peak_pA=float(row.depth_peak_pA),
                    depth_sust_pA=float(row.depth_sust_pA))
        if "subpop" in row and pd.notna(row["subpop"]):
            info["subpop"] = int(row["subpop"])
        maps.append(info)
        points.insert(0, "panel", panel + 1)
        points.insert(1, "event_id", info["event_id"])
        point_tables.append(points)
        axis = axes[panel]
        axis.plot(points.time_from_onset_ms, points.display_current_pA, color="0.2", lw=.8)
        axis.axvspan(0, info["dwell_ms"], color="#009E73", alpha=.16)
        if local_baseline:
            axis.axhspan(-3 * stats.noise_pA, 3 * stats.noise_pA, color="0.5", alpha=.13, lw=0)
        axis.axhline(0, color="0.6", ls="--", lw=.6)
        population = f"P{int(row.subpop) + 1} · " if "subpop" in row and pd.notna(row["subpop"]) else ""
        axis.set_title(f"{population}{info['event_id']}\n{info['dwell_ms']:.2f} ms · peak {row.depth_peak_pA:.1f} pA", fontsize=8)
        bind_event_panel(axis, tag=analysis.tag, event_id=row.get('event_id', info['event_id']),
                         start_idx=info['start_idx'], end_idx=info['end_idx'])
        axis.set_xlabel("ms from onset", fontsize=8)
        axis.set_ylabel("pA relative to local baseline" if local_baseline else "Prepared detrended (pA)", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes[len(picks):]:
        axis.axis("off")
    if not len(picks):
        axes[0].text(.5, .5, "No accepted events available", ha="center", va="center", transform=axes[0].transAxes)
    fig.suptitle(title, fontsize=13)
    map_path = Path(folder) / f"{name}_panels.csv"
    pd.DataFrame(maps, columns=None if maps else ["panel", "event_id", "start_idx", "end_idx"]).to_csv(map_path, index=False)
    (pd.concat(point_tables, ignore_index=True) if point_tables else pd.DataFrame(
        columns=["panel", "event_id", "sample_idx", "time_from_onset_ms", "prepared_detrended_pA", "display_current_pA"])
     ).to_csv(Path(folder) / f"{name}_points.csv", index=False)
    check_cancelled(cancel)
    return _save(fig, folder, name, cancel=cancel)


def _trace_context(analysis, stats, folder, *, cancel=None):
    n, fs = len(analysis.detrended), float(analysis.fs_hz)
    peak = int(np.argmin(analysis.detrended))
    fig, axes = _figure(4, 1, width=13, height=11)
    frames, maps = [], []
    for panel, (label, array) in enumerate((("Raw current (original CSV sign)", analysis.raw),
                                          ("Prepared detrended (events down)", analysis.detrended))):
        check_cancelled(cancel)
        indices = _overview_indices(array)
        axes[panel].plot(indices / fs, array[indices], color="0.25", lw=.4)
        axes[panel].set(title=f"{analysis.tag} — {label}", xlabel="Recording time (s)", ylabel="Current (pA)")
        frames.append(pd.DataFrame({"panel": panel + 1, "sample_idx": indices, "time_s": indices / fs,
                                    "display_time": indices / fs, "current_pA": array[indices]}))
        maps.append(dict(panel=panel + 1, source="raw" if panel == 0 else "oriented_detrended",
                         start_idx=0, end_exclusive=n, display_method="minmax_envelope",
                         display_time_units="s"))
    for panel, duration in ((2, 2.), (3, .1)):
        half = max(1, int(duration * fs / 2))
        lo, hi = max(0, peak - half), min(n, peak + half)
        indices = np.arange(lo, hi)
        relative = (indices - peak) / fs * 1000
        axes[panel].plot(relative, analysis.detrended[indices], color="0.25", lw=.7)
        axes[panel].axhline(stats.baseline_pA, color="0.6", lw=.7)
        for sigma, color in ((3, "#E69F00"), (5, "#D55E00")):
            axes[panel].axhline(stats.baseline_pA - sigma * stats.noise_pA, color=color, ls="--", lw=.8,
                              label=f"{sigma} × quiet MAD")
        axes[panel].set(title=f"{duration:g} s context at deepest excursion ({peak / fs:.5g} s)",
                        xlabel="Time from deepest excursion (ms)", ylabel="Prepared detrended (pA)")
        axes[panel].legend(fontsize=8)
        frames.append(pd.DataFrame({"panel": panel + 1, "sample_idx": indices, "time_s": indices / fs,
                                    "display_time": relative, "current_pA": analysis.detrended[indices]}))
        maps.append(dict(panel=panel + 1, source="oriented_detrended", start_idx=lo, end_exclusive=hi,
                         display_method="every_source_sample", display_time_units="ms from deepest excursion"))
    name = f"trace_context_{analysis.tag}"
    pd.concat(frames, ignore_index=True).to_csv(Path(folder) / f"{name}_points.csv", index=False)
    pd.DataFrame(maps).to_csv(Path(folder) / f"{name}_panels.csv", index=False)
    _json(Path(folder) / f"{name}_methods.json", dict(
        deepest_sample_idx=peak, overview_method="minmax_envelope", overview_max_bins=100_000,
        overview_rule="Both extrema of each contiguous source block, in temporal order; include tail and recording endpoints. Traces with <=200000 samples use every sample.",
        quiet_baseline_pA=stats.baseline_pA,
        quiet_noise_mad_pA=stats.noise_pA, raw_sign="unchanged CSV", detrended_sign="oriented events down",
        display_only="Overview preserves block extrema; no analytical table is downsampled.",
        difference_from_notebook_cell31="Displays raw and prepared detrended separately; does not reconstruct the unavailable prepared baseline curve. Noise lines use the detector's two-pass quiet MAD."))
    return _save(fig, folder, name, cancel=cancel)


def recording_figures(analysis, accepted: pd.DataFrame, folder: Path, *, cancel=None) -> list[Path]:
    """Notebook cell29 typical events, cell30/cell49 ladders and trace context."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stats = _stats(analysis, cancel)
    paths = []
    for name, key, detail in (("gallery50", "depth_sigma", "peak depth / quiet MAD; notebook cell30"),
                              ("gallery_sustained", "sust_sigma", "sustained depth / quiet MAD; notebook cell49")):
        picks = _quantile_picks(accepted, key) if len(accepted) else accepted.copy()
        paths.append(_gallery(analysis, picks, stats, folder, f"{name}_{analysis.tag}",
                    f"{analysis.tag}: {len(picks)} / {len(accepted)} accepted events · {detail}\nWeakest → strongest; onsets separated by >20 ms", cancel=cancel))
    typical = _typical_picks(accepted)
    paths.append(_gallery(analysis, typical, stats, folder, f"typical_events_{analysis.tag}",
                 f"{analysis.tag}: typical events · nearest median in (log dwell, peak depth) / standard deviation\nOnsets separated by >50 ms; prepared detrended current", cols=4,
                 local_baseline=False, cancel=cancel))
    paths.append(_trace_context(analysis, stats, folder, cancel=cancel))
    _json(folder / f"gallery_methods_{analysis.tag}.json", dict(
        event_end="inclusive input, end+1 exclusive for displayed shading and sample slicing",
        local_baseline="median of globally quiet samples within ±20ms excluding event; >20 quiet samples else global quiet baseline",
        panel_selection="50 evenly spaced rank quantiles; >20ms onset separation; backfill weakest first",
        typical_selection="up to4 nearest median in standardized log10(dwell_ms), depth_peak_pA; >50ms onset separation",
        fewer_than_requested="Report actual count when time separation or accepted count prevents filling the grid",
        source_trace=analysis.provenance.get("trace", {}).get("path", "")))
    return paths


def population_galleries(assignments, analysis, folder: Path, *, cancel=None) -> list[Path]:
    """Descriptive GMM examples plus every assigned onset on the recording clock."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if "subpop" not in assignments:
        return []
    sub = assignments.loc[(assignments.tag == analysis.tag) & assignments.subpop.notna()].copy()
    if not len(sub):
        return []
    catalog = analysis.catalog
    acceptance = "accepted" if "accepted" in catalog else "passes_gate"
    if acceptance in catalog:
        catalog = catalog.loc[catalog[acceptance].eq(True)]
    # Appendix feature amplitudes use a different baseline convention. Titles and
    # panel maps here must quote the original accepted catalog measurements.
    fields = ["dwell_ms", "depth_peak_pA", "depth_sust_pA"]
    sub = sub.drop(columns=fields, errors="ignore").merge(
        catalog[["start_idx", "end_idx", *fields]], on=["start_idx", "end_idx"],
        validate="one_to_one", how="left")
    if sub[fields].isna().any().any():
        raise ValueError("Population events do not resolve to the recording's accepted catalog")
    sub = _event_minima(analysis, sub, cancel=cancel)
    stats = _stats(analysis, cancel)
    samples = []
    for population, rows in sub.groupby("subpop", sort=True):
        check_cancelled(cancel)
        samples.append(_typical_picks(rows))
    chosen = pd.concat(samples, ignore_index=True)
    name = f"population_examples_{analysis.tag}"
    paths = [_gallery(analysis, chosen, stats, folder, name,
              f"{analysis.tag}: GMM component examples (up to4 per component)\nDescriptive components; physical species are not established", cols=4, cancel=cancel)]
    fig, axes = _figure(2, 1, width=13, height=7)
    axes[1].sharex(axes[0])
    n, fs = len(analysis.detrended), float(analysis.fs_hz)
    indices = _overview_indices(analysis.detrended)
    axes[0].plot(indices / fs, analysis.detrended[indices], color="0.65", lw=.4, zorder=1)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    for population, rows in sub.groupby("subpop", sort=True):
        check_cancelled(cancel)
        color = colors[int(population) % len(colors)]
        axes[0].scatter(rows.marker_time_s, rows.marker_current_pA, s=10, color=color,
                        label=f"P{int(population)+1} (n={len(rows)})", zorder=3)
        axes[1].scatter(rows.onset_time_s, np.full(len(rows), int(population)+1), s=6, color=color)
    axes[0].set(ylabel="Prepared detrended (pA)",
                title=f"{analysis.tag}: assigned event minima across the full recording")
    axes[0].legend(fontsize=8, ncols=min(6, sub.subpop.nunique()))
    axes[1].set(xlabel="Recording time (s)", ylabel="GMM component", xlim=(0, analysis.duration_s),
                title="Event onsets (same assignments; one marker per event)")
    axes[1].set_yticks(sorted(sub.subpop.astype(int).unique()+1))
    name = f"population_onsets_{analysis.tag}"
    sub.to_csv(folder / f"{name}_events.csv", index=False)
    pd.DataFrame({"sample_idx": indices, "time_s": indices / fs, "prepared_detrended_pA": analysis.detrended[indices]}).to_csv(
        folder / f"{name}_trace_points.csv", index=False)
    _json(folder / f"{name}_methods.json", dict(
        event_end="inclusive; source slice start_idx:end_idx+1",
        top_markers="First minimum of prepared detrended current inside each assigned event; actual sample time and current, without baseline recentering.",
        lower_markers="Event onset start_idx/fs_hz; identical assignments to top panel.",
        marker_columns=["marker_sample_idx", "marker_time_s", "marker_current_pA"],
        onset_columns=["onset_time_s", "onset_current_pA"],
        overview_method="minmax_envelope", overview_max_bins=100_000,
        overview_rule="Both extrema of each contiguous source block, in temporal order; include tail and recording endpoints. Traces with <=200000 samples use every sample.",
        fs_hz=fs, samples=n, shared_time_limits_s=[0., float(analysis.duration_s)],
        population_labels={f"P{int(k)+1}": int(k) for k in sorted(sub.subpop.unique())},
        assigned_events=len(sub),
        display_only="Event boundaries, accepted membership, measurements and population assignments are unchanged."))
    paths.append(_save(fig, folder, name, cancel=cancel))
    return paths
