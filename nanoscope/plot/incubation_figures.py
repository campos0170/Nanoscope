"""Study figures rendered with explicit Agg canvases, without pyplot or Qt.

Repeated notebook cosmetic variants are consolidated into panels. Each image
ships its source observations (and KDE coordinates where applicable). Undefined
KDEs for tiny/constant samples are explicitly rendered as observations.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from ..analysis.cancel import check_cancelled

def _pore_caveat(summary):
    """Describe the identities actually included in this figure's subset."""
    if len(summary) <= 1:
        return "Single recording; descriptive summary"
    if "pore_id" not in summary or summary.pore_id.isna().any() or summary.pore_id.astype(str).str.strip().eq("").any():
        identity = "Physical pore identities incomplete"
    else:
        count = summary.pore_id.astype(str).nunique()
        if count == 1:
            identity = "Same physical pore across recordings"
        elif count == len(summary):
            identity = "Different physical pores"
        else:
            identity = f"{count} physical pores across {len(summary)} recordings; shared and distinct pore IDs"
    return f"{identity}; descriptive comparison\nDetector transfer may vary by recording"


def _selection_caption(events, summary):
    policies = summary.attrs.get("selection_columns", events.attrs.get("selection_columns", {}))
    sources = set()
    for tag in summary.tag:
        source = policies.get(tag)
        if source is None:
            rows = events[events.tag == tag]
            # Prior checkpoints predate explicit per-recording selection attrs.
            # A populated accepted column is still an unambiguous review state;
            # mixed concatenations leave NaN for sources using only the gate.
            source = ("accepted" if "accepted" in rows and
                      (not len(rows) or rows.accepted.notna().all()) else
                      "passes_gate" if "passes_gate" in rows else "unknown")
        sources.add(source)
    if sources == {"accepted"}:
        return "reviewed acceptance", "All accepted events (review state)"
    if sources == {"passes_gate"}:
        return "native gate", "All native-gate events"
    if sources <= {"accepted", "passes_gate"} and sources:
        return "reviewed acceptance or native gate by recording", "All selected events (review or gate)"
    return "recorded event selection", "All selected events"


def _comparison_caption(events, summary):
    selection, _ = _selection_caption(events, summary)
    return f"{_pore_caveat(summary)}\nSelection: {selection}"


def _figure(*, width=13, height=8):
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _save(fig, folder, name, source=None, *, cancel=None):
    check_cancelled(cancel)
    path = Path(folder) / f"{name}.png"
    for ax in fig.axes:
        if ax.get_title():
            ax.set_title("\n".join(textwrap.fill(line, 43) for line in ax.get_title().splitlines()), fontsize=10)
    from .study_figure_style import capture_saved_figure
    if capture_saved_figure(fig, name, source):
        check_cancelled(cancel)
        return path
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if source is not None:
        source.to_csv(path.with_suffix(".data.csv"), index=False)
    fig.clear()
    check_cancelled(cancel)
    return path


def _finite(values, positive=False):
    array = np.asarray(values, float)
    return array[np.isfinite(array) & ((array > 0) if positive else True)]


def _density(ax, values, label, color, *, log=False):
    v = _finite(values, positive=log)
    v = np.log10(v) if log else v
    if len(v) < 25 or np.ptp(v) <= np.finfo(float).eps:
        if len(v):
            ax.plot(v, np.zeros(len(v)), "|", color=color, label=f"{label}: n={len(v)}; KDE unavailable")
        return []
    grid = np.linspace(v.min() - .3*v.std(), v.max() + .3*v.std(), 400)
    y = gaussian_kde(v)(grid)
    ax.plot(grid, y, color=color, label=f"{label}: n={len(v)}")
    ax.axvline(float(np.mean(v)), color=color, linestyle=":", alpha=.65)
    return [dict(condition=label, x=float(x), density=float(d)) for x, d in zip(grid, y)]


def _distribution_set(events, folder, name, amplitude, amplitude_label, *, cancel=None, caption=""):
    fig = _figure(height=10)
    axes = fig.subplots(2, 3)
    curves = []
    groups = list(events.groupby("condition", sort=False))
    for i, (label, group) in enumerate(groups):
        check_cancelled(cancel)
        color = f"C{i % 10}"
        x = np.log10(group.dwell_ms.to_numpy(float))
        y = group[amplitude].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        axes[0,0].scatter(x, y, s=8, alpha=.3, color=color, label=f"{label} (n={len(group)})")
        if len(x) >= 25 and np.linalg.matrix_rank(np.cov(x, y)) == 2:
            # Contours of the same 2D kernel as the notebook's joint view.
            gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 60), np.linspace(y.min(), y.max(), 60))
            zz = gaussian_kde(np.vstack([x,y]))(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
            axes[0,0].contour(gx, gy, zz, levels=6, colors=[color], linewidths=.7, alpha=.7)
        for ax, values, metric, log in ((axes[0,1], group[amplitude], amplitude, False), (axes[0,2], group.dwell_ms, "dwell_ms", True)):
            curves += [dict(metric=metric, **row) for row in _density(ax, values, label, color, log=log)]
        if len(y):
            axes[1,0].boxplot([y], positions=[i], widths=.5, patch_artist=True,
                              boxprops=dict(facecolor=color, alpha=.4), showfliers=True)
        if len(x):
            axes[1,1].boxplot([x], positions=[i], widths=.5, patch_artist=True,
                              boxprops=dict(facecolor=color, alpha=.4), showfliers=True)
        for values, marker, metric in ((y, "o", amplitude), (x, "s", "log10 dwell_ms")):
            if len(values):
                q1, med, q3 = np.percentile(values, [25,50,75])
                axes[1,2].errorbar(i + (0.12 if marker == "s" else -.12), med,
                    yerr=[[med-q1],[q3-med]], fmt=marker, capsize=3, color=color,
                    label=f"{label}: {metric}")
    axes[0,0].set(xlabel="log10 dwell (ms)", ylabel=amplitude_label, title="Joint distributions and observations")
    axes[0,1].set(xlabel=amplitude_label, ylabel="Density", title="Amplitude distributions; dotted = mean")
    axes[0,2].set(xlabel="log10 dwell (ms)", ylabel="Density per log10(ms)", title="Dwell distributions; dotted = mean(log dwell)")
    for ax, label in ((axes[1,0], amplitude_label), (axes[1,1], "log10 dwell (ms)")):
        ax.set_xticks(range(len(groups)), [label for label,_ in groups], rotation=25, ha="right", fontsize=7)
        ax.set(ylabel=label, title="Median / IQR; whiskers 1.5 IQR")
    # Separate coordinates instead of overlaying incomparable amplitude/dwell units.
    axes[1,2].clear()
    for i, (label, group) in enumerate(groups):
        vals = _finite(group[amplitude])
        if len(vals) >= 2 and np.ptp(vals) > 0:
            result = axes[1,2].violinplot(vals, positions=[i], showmedians=True)
            for body in result["bodies"]:
                body.set_facecolor(f"C{i%10}")
        elif len(vals):
            axes[1,2].plot([i]*len(vals), vals, "_", color=f"C{i%10}")
    axes[1,2].set_xticks(range(len(groups)), [label for label,_ in groups], rotation=25, ha="right", fontsize=7)
    axes[1,2].set(ylabel=amplitude_label, title="Amplitude violins; line = median")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    if groups:
        axes[0,0].legend(fontsize=7)
    fig.suptitle(f"{name.replace('_', ' ')}\n{caption}", fontsize=12)
    path = _save(fig, folder, name, events, cancel=cancel)
    pd.DataFrame(curves, columns=["metric", "condition", "x", "density"]).to_csv(path.with_suffix(".kde.csv"), index=False)
    return path


def comparison_figures(events, summary, tables, folder, *, cancel=None, settings=None):
    paths = []
    caption = _comparison_caption(events, summary)
    selection, all_events_title = _selection_caption(events, summary)
    definitions = (("incubation_sustained_local", "fracblock", "Sustained blockade / local I0"),
                   ("aggregation_peak_global", "aggregation_fracblock_global", "Peak blockade / global I0 (cell 28 code)"),
                   ("kde_sustained_global", "f_sust_global", "Sustained blockade / global I0 (cell 43 fallback)"))
    for name, col, label in definitions:
        paths.append(_distribution_set(events, folder, name, col, label, cancel=cancel, caption=caption))
    # Every size-matched subgroup is explicit about recording identity.
    for diameter, rows in summary.groupby("pore_nm", sort=True):
        if len(rows) > 1:
            paths.append(_distribution_set(events[events.tag.isin(rows.tag)], folder,
                        f"size_matched_{diameter:g}nm_sustained_local", "fracblock",
                        "Sustained blockade / local I0", cancel=cancel,
                        caption=_comparison_caption(events[events.tag.isin(rows.tag)], rows)))
    fig = _figure(height=5)
    axes = fig.subplots(1, 3)
    for i, row in enumerate(summary.itertuples()):
        check_cancelled(cancel)
        color = f"C{i%10}"
        axes[0].errorbar(row.incubation_h, row.rate_per_s,
              yerr=[[max(0,row.rate_per_s-row.lo)], [max(0,row.hi-row.rate_per_s)]], fmt="o", color=color, capsize=4,
              label=f"{row.tag}: {row.n_events} events")
        sub = events[events.tag == row.tag]
        for ax, col in ((axes[1], "fracblock"), (axes[2], "dwell_ms")):
            v = _finite(sub[col])
            if len(v):
                q1, median, q3 = np.percentile(v, [25,50,75])
                ax.errorbar(row.incubation_h, median, yerr=[[median-q1],[q3-median]], fmt="o", color=color, capsize=4)
    blocks = settings.n_blocks if settings is not None else 20
    draws = settings.n_boot if settings is not None else 4000
    axes[0].set(ylabel="Accepted event rate (/s)", title=f"95% time-block bootstrap ({blocks} blocks; {draws} draws)")
    axes[1].set(ylabel="Sustained / local I0", title="Median ± IQR")
    axes[2].set(ylabel="Dwell (ms)", title="Median ± IQR")
    if len(events):
        axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Incubation (h)"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(caption, fontsize=11)
    paths.append(_save(fig, folder, "incubation_rates_and_medians", summary, cancel=cancel))
    fig = _figure(height=5)
    axes = fig.subplots(1, 3)
    sweep = tables["gate_transfer_sweep"]
    for ax, (definition, group) in zip(axes, sweep.groupby("definition", sort=False)):
        for i, (tag, rows) in enumerate(group.groupby("tag", sort=False)):
            ax.plot(rows.floor * 100, rows.rate, "o-", ms=3, color=f"C{i%10}", label=tag)
            gate = float(summary.loc[summary.tag == tag, "gate_frac_of_I0"].iloc[0])
            ax.axvline(gate * 100, color=f"C{i%10}", alpha=.35, ls=":")
        ax.axvline(float(summary.gate_frac_of_I0.max())*100, color="black", ls="--", label="global native gate")
        ax.set(xlabel="Common fractional floor (%)", ylabel="Rate above floor (/s)", title=definition.replace("_", " "))
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(f"Gate-transfer sensitivity; {selection}\n{_pore_caveat(summary)}", fontsize=11)
    paths.append(_save(fig, folder, "gate_transfer_sweep", sweep, cancel=cancel))
    # Common-floor distributions separate all observations from the jointly observable region.
    fig = _figure(height=5)
    axes = fig.subplots(1, 3)
    floor = float(summary.gate_frac_of_I0.max())
    curves = []
    for i, (label, grp) in enumerate(events.groupby("condition", sort=False)):
        for ax, column, filtered, log in ((axes[0], "dwell_ms", False, True),
                                          (axes[1], "f_sust_global", False, False),
                                          (axes[2], "f_sust_global", True, False)):
            sub = grp[grp.f_sust_global >= floor] if filtered else grp
            curves.extend(dict(variable=column, above_common_floor=filtered, **r)
                          for r in _density(ax, sub[column], label, f"C{i%10}", log=log))
    axes[0].set(xlabel="log10 dwell (ms)", ylabel="Density", title="Dwell")
    axes[1].set(xlabel="Sustained / global I0", ylabel="Density", title=all_events_title)
    axes[2].set(xlabel="Sustained / global I0", ylabel="Density", title=f"Common floor ≥ {floor:.3g}")
    for ax in axes:
        ax.grid(alpha=.2)
        if len(events):
            ax.legend(fontsize=6)
    gallery_method = ("Kernel-density gallery: selected events and common native-gate floor" if settings is not None else
                      "Kernel-density gallery: source cell43 fresh-detector fallback")
    fig.suptitle(f"{gallery_method}\n{caption}", fontsize=11)
    path = _save(fig, folder, "common_floor_kde", events, cancel=cancel)
    pd.DataFrame(curves).to_csv(path.with_suffix(".kde.csv"), index=False)
    paths.append(path)
    return paths


def clustering_figures(table, folder, prefix="original12", *, cancel=None, settings=None):
    kmeans_column = f"kmeans{settings.kmeans_k}" if settings is not None else "kmeans3"
    if not len(table) or kmeans_column not in table:
        return []
    paths = []
    for name, x, y in (("pca", "PC1", "PC2"), ("tsne", "tsne_x", "tsne_y")):
        if x not in table or y not in table:
            continue
        fig = _figure(height=4)
        axes = fig.subplots(1, 4)
        for ax, column in zip(axes, (kmeans_column, "hdbscan", "incubation_h", "log_dwell")):
            check_cancelled(cancel)
            if column == "log_dwell":
                points = ax.scatter(table[x], table[y], c=table[column], s=8, alpha=.6)
                fig.colorbar(points, ax=ax)
            else:
                for i, (value, rows) in enumerate(table.groupby(column, sort=True)):
                    ax.scatter(rows[x], rows[y], s=8, alpha=.5, color="0.65" if value == -1 else f"C{i%10}", label=f"{value} (n={len(rows)})")
                ax.legend(fontsize=6)
            ax.set(title=column, xlabel=x, ylabel=y)
        title = ("Original notebook 12-feature" if settings is None else
                 f"Selected {len(settings.feature_columns)}-feature")
        sources = table.drop_duplicates("tag") if "tag" in table else pd.DataFrame(index=[0, 1])
        fig.suptitle(f"{title} pooled {name.upper()} — descriptive clusters\n{_pore_caveat(sources)}", fontsize=11)
        paths.append(_save(fig, folder, f"{prefix}_{name}", table, cancel=cancel))
    return paths


def recording_figures(analysis, accepted, folder, *, cancel=None):
    from .incubation_galleries import recording_figures as render
    paths = render(analysis, accepted, folder, cancel=cancel)
    # Single-recording notebook cells8-13 use ALL student candidates and peak depth.
    candidates = analysis.catalog.copy()
    baseline = float(np.median(analysis.detrended))
    candidates["single_peak_global_baseline_pA"] = [
        round(baseline - float(analysis.detrended[int(row.start_idx):int(row.end_idx)+1].min()), 1)
        for row in candidates.itertuples()]
    candidates["single_dwell_rounded_ms"] = [round(float(value), 3) for value in candidates.dwell_ms]
    fig = _figure(height=4)
    axes = fig.subplots(1, 3)
    for ax, values, label in ((axes[0], candidates.single_peak_global_baseline_pA, "Peak / full-trace baseline (pA), rounded0.1pA"),
                               (axes[1], np.log10(candidates.single_dwell_rounded_ms.to_numpy(float)), "log10 dwell (ms), rounded0.001ms")):
        ax.hist(_finite(values), bins=40, color="C0", alpha=.7)
        ax.set(xlabel=label, ylabel="Count")
    axes[2].scatter(candidates.single_dwell_rounded_ms, candidates.single_peak_global_baseline_pA, s=8, alpha=.4)
    axes[2].set(xlabel="Dwell (ms)", ylabel="Peak depth (pA)", title="Ungated candidates, full-trace baseline")
    fig.suptitle(f"{analysis.tag}: single-recording summary ({len(candidates)} candidates; gate not applied)")
    paths.append(_save(fig, folder, f"single_recording_candidates_{analysis.tag}", candidates, cancel=cancel))
    return paths


def _plot_population_fractions(axis, quant):
    """Show the actual weight interval even when it excludes the fitted point."""
    components = np.arange(quant.fit.k)
    weights = np.asarray(quant.fit.weights, float)
    assigned = np.array([quant.fit.sizes.get(int(k), 0) / quant.n_events for k in components])
    lo, hi = np.asarray(quant.ci.lo, float), np.asarray(quant.ci.hi, float)
    weight_x = components - .08
    valid = np.isfinite(lo) & np.isfinite(hi)
    # Separate interval segments retain their endpoints without clipping a
    # negative error length when the point estimate is outside the interval.
    axis.vlines(weight_x[valid], lo[valid], hi[valid], color="C0", label="Mixture weight bootstrap 95% interval")
    axis.hlines(lo[valid], weight_x[valid]-.035, weight_x[valid]+.035, color="C0")
    axis.hlines(hi[valid], weight_x[valid]-.035, weight_x[valid]+.035, color="C0")
    axis.plot(weight_x, weights, "o", color="C0", label="Fitted mixture weight")
    axis.plot(components+.08, assigned, "s", color="C1", fillstyle="none", label="Assigned event fraction")
    axis.set(xticks=components, xlabel="Subpopulation", ylabel="Fraction",
             title="Mixture weights and assigned events")
    axis.legend(fontsize=7)


def appendix_figures(assignments, quant, analysis, appendix_folder, folder, *, cancel=None):
    if quant is None:
        return []
    paths = []
    fig = _figure(height=5)
    axes = fig.subplots(1, 3)
    image = axes[0].imshow(quant.selection.grid, origin="lower", aspect="auto",
                         extent=[min(quant.selection.k_grid)-.5,max(quant.selection.k_grid)+.5,
                                 min(quant.selection.p_grid)-.5,max(quant.selection.p_grid)+.5])
    if (np.any(np.diff(quant.selection.k_grid) != 1)
            or np.any(np.diff(quant.selection.p_grid) != 1)):
        # A sparse or reordered user grid has categorical cells, not evenly spaced
        # numerical coordinates between its smallest and largest search values.
        image.set_extent([-.5, len(quant.selection.k_grid)-.5,
                          -.5, len(quant.selection.p_grid)-.5])
        axes[0].set_xticks(range(len(quant.selection.k_grid)), quant.selection.k_grid)
        axes[0].set_yticks(range(len(quant.selection.p_grid)), quant.selection.p_grid)
    fig.colorbar(image, ax=axes[0], label="Held-out log likelihood")
    axes[0].set(xlabel="Mixture K", ylabel="PCA dimensions", title="Joint model selection")
    _plot_population_fractions(axes[1], quant)
    waves = pd.read_csv(Path(appendix_folder) / "population_median_waveforms.csv")
    for column in waves:
        if column != "T":
            axes[2].plot(waves["T"], waves[column], label=column)
    axes[2].set(xlabel="Normalized event time", ylabel="Raw current (pA)", title="Median waveforms")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"Separate NanoScope {len(quant.features)}-feature quantitative appendix\n{quant.claim()}", fontsize=10)
    population_table = pd.read_csv(Path(appendix_folder) / "population_quantification.csv", float_precision="round_trip")
    paths.append(_save(fig, folder, "quantitative_appendix_selection_fractions_waveforms", population_table, cancel=cancel))
    from .incubation_galleries import population_galleries
    paths.extend(population_galleries(assignments, analysis, folder, cancel=cancel))
    return paths
