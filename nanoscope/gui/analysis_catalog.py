"""Canonical analysis navigation metadata. No widgets, recordings or analysis imports."""
from dataclasses import dataclass

PAGE_HOME = "analysis_home"
PAGE_IDS = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
            "noise", "iv", "voltage_series", "subpopulations")
ANALYSIS_LABELS = {
    "distributions": "Distributions",
    "distribution_summaries": "Distribution summaries",
    "comparison_figures": "Comparison figures",
    "event_waveforms": "Event waveforms",
    "noise": "Noise spectrum",
    "iv": "I-V / conductance",
    "voltage_series": "Voltage dependence",
    "subpopulations": "Subpopulations",
    "report_figures": "Report figures",
    "event_gallery": "Event gallery",
    "figure_library": "Figure Library",
    "guided_first_analysis": "Guided first analysis",
    PAGE_HOME: "Analysis home",
}

STATUS_HELP = (
    "Current: the displayed result matches its source and settings. This does not certify calibration, "
    "event review or scientific validity.\n\n"
    "Stale: source data or analytical settings changed after this result was generated. Inspect the "
    "difference before refreshing; the visible numbers have not silently changed.\n\n"
    "Saved snapshot: a frozen project result is shown without recomputation. It may differ from today's "
    "live source and remains available for inspection and export.\n\n"
    "No data: this view has no result yet. Read its input requirements before running an analysis."
)


@dataclass(frozen=True)
class AnalysisEntry:
    route: str
    group: str
    description: str
    inputs: str

    @property
    def label(self):
        return ANALYSIS_LABELS[self.route]


ANALYSIS_ENTRIES = (
    AnalysisEntry("guided_first_analysis", "Start here",
        "Follow a recording through calibration, baseline preparation, detection, event review and a first figure.",
        "Start with a recording, or open the guide before loading one. Each scientific action remains explicit."),
    AnalysisEntry("distributions", "Event analysis",
        "Explore event histograms and fitted overlays, scatter plots, PCA and clustering.",
        "Detected events or saved manual measurements. Existing saved project results can also be reopened."),
    AnalysisEntry("distribution_summaries", "Event analysis",
        "Compare ECDF, empirical survival, box and violin summaries of existing event measurements.",
        "Choose event sources and a metric, then generate the summary. Empirical survival is uncensored, not Kaplan–Meier."),
    AnalysisEntry("comparison_figures", "Event analysis",
        "Build faceted comparisons, pair-plot matrices and individual-event strip or swarm overlays.",
        "Existing event measurements, source conditions and selected metrics. Opening the controls does not generate a figure."),
    AnalysisEntry("event_waveforms", "Event analysis",
        "Inspect event-count heatmaps and aligned waveform ensembles with mean and median traces.",
        "A detection with at least one accepted event, or a saved result. Event counts are not KDE probability density."),
    AnalysisEntry("subpopulations", "Event analysis",
        "Inspect or explicitly generate an exploratory report about groups within the accepted events.",
        "Accepted events and their recording for a new report, or a saved report. Statistical groups are not identified molecular species."),
    AnalysisEntry("noise", "Recording and pore quality",
        "Examine the current-noise power spectrum and band-integrated RMS noise.",
        "A calibrated current channel with a known sample rate, or a saved spectrum. No event detection is required."),
    AnalysisEntry("iv", "Recording and pore quality",
        "Compare current against applied voltage and inspect conductance fits.",
        "Current and known applied voltage, or a saved result. Review gain, offset and the open-pore reference before interpreting conductance."),
    AnalysisEntry("voltage_series", "Recording and pore quality",
        "Compare event rate, dwell and depth across applied voltages.",
        "At least two detected recordings with known applied voltage, or a saved result. Counts alone are not rates."),
    AnalysisEntry("event_gallery", "Review and figures",
        "Review events as waveform tiles, sort them, and select groups for curation.",
        "A selected detection result. This edits event review decisions, unlike the Figure Library."),
    AnalysisEntry("report_figures", "Review and figures",
        "Generate the standard report figure set from a detection result, including source-data sidecars.",
        "A selected detection result. This generates standard figures; it does not export the styled Explorer canvas."),
    AnalysisEntry("figure_library", "Review and figures",
        "Inspect named frozen figures, compose publication panels, and export pictures or reproducible bundles.",
        "Open the library at any time. Save an Explorer figure first to populate it. Original plotted data and provenance stay with each saved figure."),
)
ENTRY_BY_ROUTE = {entry.route: entry for entry in ANALYSIS_ENTRIES}
