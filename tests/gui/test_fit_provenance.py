"""The two ways the app used to show a user a tau it did not compute (offscreen).

**B1 — two different tau for the same events.** The Plots window was pushed bare
DataFrames with no provenance, so its dwell fit fell back to
``min(observed dwell)`` as the detection limit while the Fits panel used the
detector's real ``min_dwell_s``. Same events, same button, two taus — biased
apart by roughly ``t_min`` (~25% at a 50 us floor against a 200 us tau) and
invisible on the plot, because the histogram is missing exactly the events the
fit is missing. These tests assert the two agree to floating point, that the
window SAYS which ``t_min`` it used, and that it shades the region below it the
way ``plot.figures`` does.

**B2 — the exported figure discarded the fit.** ``Export figure…`` called
``figures.dwell_histogram(events, fit=True)``, and ``fit=True`` refits from
scratch at ``n_components=1``. A user who set "dwell components = 2", read
tau1/tau2 in the panel and exported got a figure annotated with a different,
one-component tau. These tests run the real fit through the real panel and assert
the exported figure's own source-data CSV carries THAT fit's curve.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import fit_dwell
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plots_panel import PlotsWindow

FS = 250_000.0
MIN_DWELL_S = 5e-5


# ================================================================================================
# builders
# ================================================================================================

def _planted_trace(seconds: float = 12.0, *, seed: int = 0, tau: float = 3e-4,
                   cutoff_limited: bool = False) -> np.ndarray:
    """A current trace with exponentially-distributed planted blockades."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    x = np.full(n, 1000.0) + rng.normal(0.0, 5.0, n)
    pos = 2000
    while True:
        d = max(6 if cutoff_limited else 20, int(rng.exponential(tau) * FS))
        if pos + d + 3000 >= n:
            break
        x[pos:pos + d] -= abs(rng.normal(60.0, 12.0))
        pos += d + int(rng.integers(2000, 5000))
    return np.ascontiguousarray(x, dtype=np.float32)


def _detection(**kwargs) -> EventSet:
    from nanoscope.analysis.detect import detect_events

    df, prov = detect_events(_planted_trace(**kwargs.pop("trace_kwargs", {})), FS,
                             min_dwell_s=MIN_DWELL_S, k_in=5, k_out=2, **kwargs)
    return EventSet(df, prov)


def _synth_doc(tmp_path, name: str = "pore_p1.dat", **trace_kwargs) -> TraceDocument:
    arr = _planted_trace(**trace_kwargs)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _install_and_detect(window, qtbot, tmp_path, *, cutoff_hz: float | None = None,
                        **trace_kwargs) -> tuple[str, int]:
    doc = _synth_doc(tmp_path, **trace_kwargs)
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    panel = window._analysis_panel.detection_panel
    panel.min_dwell_s.setValue(MIN_DWELL_S)
    panel.filter_enabled.setChecked(cutoff_hz is not None)
    if cutoff_hz is not None:
        panel.filter_cutoff_hz.setValue(float(cutoff_hz))
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(20)
    return doc_id, channel


# ================================================================================================
# B1 — the Plots window and the Fits panel must agree
# ================================================================================================

def _plots_window(qtbot, events: EventSet, *, with_provenance: bool = True) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"Detected events": events.df},
               {"Detected events": dict(events.provenance)} if with_provenance else None)
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_log_x(True)
    return w


def test_plots_and_fits_agree_on_tau_to_floating_point(qapp, qtbot):
    """THE regression: one set of events must not produce two different taus."""
    events = _detection()
    panel_fit = fit_dwell(events, doc_id="d", channel=0, n_boot=0).payload

    w = _plots_window(qtbot, events)
    w.set_fit("exponential")
    plots_fit = w.fit_result()

    assert plots_fit is not None
    assert plots_fit.params["tau_1"] == panel_fit.params["tau_1"]
    assert plots_fit.n == panel_fit.n
    assert plots_fit.meta["t_min"] == panel_fit.meta["t_min"]


def test_plots_uses_the_detectors_min_dwell_not_the_shortest_observed_event(qapp, qtbot):
    events = _detection()
    observed_min = float(events.df[events.df["accepted"]]["dwell_s"].min())
    assert observed_min > MIN_DWELL_S, "fixture must make the two floors differ"

    w = _plots_window(qtbot, events)
    w.set_fit("exponential")

    fit = w.fit_result()
    assert fit.meta["t_min"] == MIN_DWELL_S
    assert fit.meta["t_min"] != observed_min
    # ...and the number is VISIBLE, not just correct.
    assert f"detection limit t_min = {MIN_DWELL_S:g} s (from min_dwell_s)" in w.details_text()


def test_plots_says_so_when_it_has_to_guess_the_detection_limit(qapp, qtbot):
    """Without provenance the fallback is the old behaviour — but it is now LOUD.

    figures.dwell_histogram raises a RuntimeWarning in the same situation; the
    interactive window prints the same sentence where the number is shown."""
    events = _detection()
    observed_min = float(events.df[events.df["accepted"]]["dwell_s"].min())
    w = _plots_window(qtbot, events, with_provenance=False)
    w.set_fit("exponential")

    details = w.details_text()
    assert f"detection limit t_min = {observed_min:g} s (from observed_min)" in details
    assert "provenance has no 'min_dwell_s'" in details
    assert "pass an EventSet from detect() to get it right" in details

    fit = w.fit_result()
    assert fit.meta["t_min"] == pytest.approx(observed_min)
    assert fit.meta["t_min_source"] == "observed_min"


def test_a_non_dwell_column_does_not_borrow_the_dwell_detection_floor(qapp, qtbot):
    """An exponential on rise_s is not a dwell fit; min_dwell_s does not describe it."""
    events = _detection()
    w = _plots_window(qtbot, events)
    w.set_histogram_column("ecd_pA_s")
    w.set_fit("exponential")

    assert "no detection limit is known" in w.details_text()
    assert "biases tau SHORT" in w.details_text()
    assert w.fit_result().meta["t_min"] != MIN_DWELL_S


def test_the_guessed_floor_note_names_the_right_DIRECTION_of_the_bias(qapp, qtbot):
    """``min(observed)`` can only ever be >= the detector's real gate, so substituting
    it over-subtracts in ``tau_hat = mean(t) - t_min`` and biases tau SHORT.

    The note used to say LONG. That is the direction of the *other* mistake --
    ignoring truncation altogether (``scipy.stats.expon.fit``), which gives
    ``tau + t_min``. Telling a user the wrong sign is worse than telling them
    nothing, so this pins the sign against the arithmetic."""
    from nanoscope.analysis.stats import fit_dwell_exponential

    events = _detection()
    dwell = events.df[events.df["accepted"]]["dwell_s"].to_numpy(dtype=float)
    real = fit_dwell_exponential(dwell, t_min=MIN_DWELL_S, n_boot=0).params["tau_1"]
    guessed = fit_dwell_exponential(
        dwell, t_min=float(dwell.min()), n_boot=0).params["tau_1"]
    assert float(dwell.min()) >= MIN_DWELL_S     # the detector cannot report below its gate
    assert guessed <= real                        # ...so the guess biases SHORT, never long

    w = _plots_window(qtbot, events)
    w.set_histogram_column("ecd_pA_s")
    w.set_fit("exponential")
    text = w.details_text()
    assert "biases" in text and "SHORT" in text
    assert "LONG" not in text


def test_plots_shades_the_region_below_the_detection_limit(qapp, qtbot):
    """figures.py shades it; the interactive view must not be the one that hides it."""
    events = _detection()
    w = _plots_window(qtbot, events)

    assert w._limit_region is not None, "no below-detection-limit shading"
    assert w._limit_line is not None
    lo, hi = w._limit_region.getRegion()
    # Log x: the shading lives in log10 coordinates because setLogMode does not
    # transform items added straight to the scene.
    assert hi == pytest.approx(np.log10(MIN_DWELL_S))
    assert lo < hi

    w.set_log_x(False)
    lo, hi = w._limit_region.getRegion()
    assert hi == pytest.approx(MIN_DWELL_S)


def test_shading_is_absent_for_a_column_the_detector_floor_does_not_describe(qapp, qtbot):
    events = _detection()
    w = _plots_window(qtbot, events)
    w.set_histogram_column("frac_depth")
    assert w._limit_region is None
    assert "detection limit t_min" not in w.details_text()


def test_main_window_pushes_detection_provenance_into_the_plots_window(window, qtbot, tmp_path):
    _install_and_detect(window, qtbot, tmp_path)
    window._act_plots.trigger()
    w = window._plots_window

    # The provenance now rides ON the source object, one per open recording, so the
    # per-source fit below resolves t_min from THIS detector's own min_dwell_s.
    src = w.primary_source()
    assert src is not None
    assert src.provenance.get("min_dwell_s") == pytest.approx(MIN_DWELL_S)
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    assert w.fit_result().meta["t_min_source"] == "min_dwell_s"


# ================================================================================================
# B4c in the Plots UI — the same default, the same toggle
# ================================================================================================

def test_plots_window_excludes_filter_limited_by_default_and_the_toggle_restores(qapp, qtbot):
    from nanoscope.analysis.detect import detect_events

    df, prov = detect_events(_planted_trace(tau=2e-4, cutoff_limited=True), FS,
                             min_dwell_s=2e-5, filter_cutoff_hz=10_000.0, k_in=5, k_out=2)
    events = EventSet(df, prov)
    n_limited = int(df["filter_limited"].sum())
    assert n_limited > 5, "fixture must produce filter-limited events"

    w = _plots_window(qtbot, events)
    w.set_fit("exponential")

    excluded = w.fit_result()
    assert excluded.n == len(df) - n_limited
    assert f"{n_limited} filter-limited event(s) excluded" in w.details_text()

    w._use_filter_limited.setChecked(True)
    kept = w.fit_result()
    assert kept.n == len(df)
    assert kept.params["tau_1"] != excluded.params["tau_1"]


def test_a_single_nan_row_does_not_kill_the_fit(qapp, qtbot):
    """stats refuses a column with a NaN in it; the window drops it out loud."""
    events = _detection()
    events.df.loc[events.df.index[3], "frac_depth"] = np.nan

    w = _plots_window(qtbot, events)
    w.set_histogram_column("frac_depth")
    w.set_fit("gaussian-mixture")

    assert w.fit_result() is not None
    assert "1 non-finite value(s) excluded from the fit." in w.details_text()
    assert "Fit unavailable" not in w.details_text()


def test_fits_panel_toggle_reaches_the_fit_and_the_count_is_shown(window, qtbot, tmp_path):
    """B4c end to end: the Fits box drives the exclusion, and says how many."""
    doc_id, channel = _install_and_detect(
        window, qtbot, tmp_path, cutoff_hz=10_000.0, tau=2e-4, cutoff_limited=True)
    panel = window._analysis_panel
    stats = panel._stats
    df = panel.current_events().df
    n_limited = int((df["accepted"].to_numpy(dtype=bool)
                     & df["filter_limited"].to_numpy(dtype=bool)).sum())
    assert n_limited > 0, "fixture must produce filter-limited events"

    # The count is visible BEFORE the fit runs...
    assert str(n_limited) in stats.filter_limited_label.text()
    assert "excluded from the fits" in stats.filter_limited_label.text()

    assert stats.use_filter_limited.isChecked() is False, "exclusion is the default"
    stats.fit_button.click()
    excluded = window._session.results.get(doc_id, channel, "fit.dwell").payload
    assert excluded.meta["use_filter_limited"] is False
    assert excluded.meta["n_excluded_filter_limited"] == n_limited

    stats.use_filter_limited.setChecked(True)
    assert "INCLUDED in the fits" in stats.filter_limited_label.text()
    stats.fit_button.click()
    kept = window._session.results.get(doc_id, channel, "fit.dwell").payload
    assert kept.meta["use_filter_limited"] is True
    assert kept.meta["n_excluded_filter_limited"] == 0
    assert kept.n == excluded.n + n_limited


# ================================================================================================
# B2 — the exported figure carries the fit the panel showed
# ================================================================================================

def _export_to(window, tmp_path, monkeypatch) -> Path:
    stem = tmp_path / "figs"
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(stem) + ".pdf", "PDF (*.pdf)")),
    )
    window._export_figures()
    return stem


def test_exported_dwell_figure_uses_the_two_component_fit_the_panel_displayed(
        window, qtbot, tmp_path, monkeypatch):
    from nanoscope.plot import figures

    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)

    # The user asks for two dwell components and reads tau1/tau2 in the panel.
    stats = window._analysis_panel._stats
    stats.n_components_dwell.setValue(2)
    stats.fit_button.click()
    stored = window._session.results.get(doc_id, channel, "fit.dwell")
    assert stored is not None and stored.payload.meta["n_components"] == 2
    panel_taus = [v for k, v in stored.payload.params.items() if k.startswith("tau")]
    assert len(panel_taus) == 2
    assert "tau_2" in window._analysis_panel._stats.results.toPlainText()

    stem = _export_to(window, tmp_path, monkeypatch)
    exported = pd.read_csv(f"{stem}_dwell.csv")

    events = window._analysis_panel.current_events()
    same, _ = figures.dwell_histogram(events, fit=stored.payload)
    refit, _ = figures.dwell_histogram(events, fit=True)   # what the bug produced
    import matplotlib.pyplot as plt

    same_df = getattr(same, "_nanoscope_source_data")
    refit_df = getattr(refit, "_nanoscope_source_data")
    plt.close(same)
    plt.close(refit)

    np.testing.assert_allclose(exported["fit_expected_count"].to_numpy(),
                               same_df["fit_expected_count"].to_numpy(), rtol=1e-12)
    # ...and the one-component refit really is a different curve, so this test bites.
    assert not np.allclose(refit_df["fit_expected_count"].to_numpy(),
                           same_df["fit_expected_count"].to_numpy(), rtol=1e-3)


def test_exported_depth_figure_uses_the_stored_fit_too(window, qtbot, tmp_path, monkeypatch):
    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    stats = window._analysis_panel._stats
    stats.n_components_depth.setValue(2)
    stats.fit_button.click()
    stored = window._session.results.get(doc_id, channel, "fit.depth")
    assert stored is not None and stored.payload.meta["n_components"] == 2

    stem = _export_to(window, tmp_path, monkeypatch)
    exported = pd.read_csv(f"{stem}_depth.csv")
    centers = exported["bin_center"].to_numpy()

    np.testing.assert_allclose(exported["fit_density"].to_numpy(),
                               np.asarray(stored.payload.predict(centers), dtype=float),
                               rtol=1e-12)


def test_export_without_a_saved_fit_keeps_todays_behaviour_and_says_so(
        window, qtbot, tmp_path, monkeypatch):
    _install_and_detect(window, qtbot, tmp_path)   # no Fit button pressed

    stem = _export_to(window, tmp_path, monkeypatch)

    assert Path(f"{stem}_dwell.pdf").exists()
    msg = window.statusBar().currentMessage()
    assert "no saved fit" in msg and "refit fresh at k=1" in msg


def test_a_fit_from_before_a_curation_edit_is_not_stamped_on_the_figure(
        window, qtbot, tmp_path, monkeypatch):
    """A stale fit on a publication figure is worse than an honest refit."""
    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    stats = window._analysis_panel._stats
    stats.fit_button.click()
    assert window._session.results.get(doc_id, channel, "fit.dwell") is not None

    # The user then rejects some events; the stored fit now describes a different sample.
    df = window._analysis_panel.current_events().df
    df.loc[df.index[:20], "accepted"] = False

    _export_to(window, tmp_path, monkeypatch)

    msg = window.statusBar().currentMessage()
    assert "stale (the accepted events changed since it ran)" in msg


def test_export_status_names_the_fit_it_used(window, qtbot, tmp_path, monkeypatch):
    _install_and_detect(window, qtbot, tmp_path)
    stats = window._analysis_panel._stats
    stats.n_components_dwell.setValue(2)
    stats.fit_button.click()

    _export_to(window, tmp_path, monkeypatch)

    msg = window.statusBar().currentMessage()
    assert "dwell: used your saved fit (k=2)" in msg


# ================================================================================================
# B1's THIRD leg — the exported figure's own annotation, not just its source-data CSV
# ================================================================================================

def _annotation(ax) -> str:
    """The fit annotation block ``figures._annotate`` writes (the '$\\tau$ = ...' text)."""
    return next(t.get_text() for t in ax.texts if "tau" in t.get_text())


def test_all_three_readouts_annotate_the_same_tau(window, qtbot, tmp_path, monkeypatch):
    """Fits panel, Plots window and the exported figure's ANNOTATION must agree.

    The CSV-curve assertions above pin the plotted curve; this pins the NUMBER a
    reader takes off the page, which is the one that gets published."""
    import matplotlib.pyplot as plt

    from nanoscope.plot import figures

    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    events = window._analysis_panel.current_events()
    observed_min = float(events.df[events.df["accepted"]]["dwell_s"].min())
    assert observed_min > MIN_DWELL_S, "the two candidate floors must differ, or this is vacuous"

    stats = window._analysis_panel._stats
    stats.fit_button.click()
    panel = window._session.results.get(doc_id, channel, "fit.dwell").payload

    window._act_plots.trigger()
    plots = window._plots_window
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    plotted = plots.fit_result()

    stored, reason = window._stored_fit(
        window._analysis_panel.current_detection(), "fit.dwell")
    assert stored is not None, reason
    fig, ax = figures.dwell_histogram(events, fit=stored)
    try:
        text = _annotation(ax)
    finally:
        plt.close(fig)

    assert plotted.params["tau_1"] == panel.params["tau_1"]          # (a) == (b), exactly
    assert stored.params["tau_1"] == panel.params["tau_1"]           # (c) is that same object
    assert rf"$\tau$ = {panel.params['tau_1'] * 1e6:.1f} " in text   # ...and it is what is drawn
    assert rf"$t_{{min}}$ = {MIN_DWELL_S * 1e6:.1f} $\mu$s" in text
    assert panel.meta["t_min"] == MIN_DWELL_S == plotted.meta["t_min"]


def test_exported_two_component_figure_annotates_BOTH_taus(
        window, qtbot, tmp_path, monkeypatch):
    """B2 read off the page: a k=2 fit must print two taus, not one refit tau."""
    import matplotlib.pyplot as plt

    from nanoscope.plot import figures

    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    stats = window._analysis_panel._stats
    stats.n_components_dwell.setValue(2)
    stats.fit_button.click()
    stored = window._session.results.get(doc_id, channel, "fit.dwell").payload
    assert stored.meta["n_components"] == 2

    events = window._analysis_panel.current_events()
    fig, ax = figures.dwell_histogram(events, fit=stored)
    refit, rax = figures.dwell_histogram(events, fit=True)
    try:
        text, refit_text = _annotation(ax), _annotation(rax)
    finally:
        plt.close(fig)
        plt.close(refit)

    assert text.count(r"$\tau$ = ") == 2
    assert refit_text.count(r"$\tau$ = ") == 1     # the bug's output, for contrast
    for _, val, _ in sorted(
            ((k, v, None) for k, v in stored.params.items() if k.startswith("tau"))):
        assert rf"$\tau$ = {val * 1e6:.1f} " in text


def test_figure_shades_at_the_floor_the_fit_actually_used(qapp, tmp_path):
    """The dashed 'below detection limit' line must sit at the floor the FIT used.

    Was an xfail: dwell_histogram shaded at resolve_t_min(provenance) while the
    supplied fit had raised its own floor to 2*filter_rise_s by excluding
    filter-limited events -- so the annotation named one number and the line was
    drawn at another, leaving the excluded bars unmarked. figures.py now adopts
    the supplied fit's meta['t_min'] for the shading, the curve start and the
    n_fit fallback alike.
    """
    import matplotlib.pyplot as plt

    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.modules import fit_dwell
    from nanoscope.plot import figures

    df, prov = detect_events(
        _planted_trace(seconds=25.0, seed=17, tau=1.2e-4, cutoff_limited=True), FS,
        min_dwell_s=1e-5, k_in=5, k_out=2,
        filter_cutoff_hz=20_000.0, filter_kind="bessel", filter_order=8)
    assert prov["n_filter_limited"] > 0
    events = EventSet(df, prov)
    fit = fit_dwell(events, doc_id="d", channel=0, n_boot=0).payload
    assert fit.meta["t_min_source"] == "2*filter_rise_s"

    fig, ax = figures.dwell_histogram(events, fit=fit)
    try:
        dashed = [line.get_xdata()[0] for line in ax.lines
                  if line.get_linestyle() not in ("-", "solid")]
    finally:
        plt.close(fig)

    assert dashed, "no detection-limit line drawn"
    assert dashed[0] / 1e6 == pytest.approx(fit.meta["t_min"], rel=1e-9)
