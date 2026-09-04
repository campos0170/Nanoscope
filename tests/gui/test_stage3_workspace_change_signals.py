"""Stage 3 analytical-window dirty notifications.

The windows are refreshed by MainWindow as session data changes.  Those
maintenance refreshes must stay silent; only durable user gestures should mark a
project dirty.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from PySide6 import QtGui

from nanoscope.analysis import iv, populations, voltage_dep
from nanoscope.analysis.stats import FitResult
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui import plots_panel
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow


def _emissions(window) -> list[str]:
    seen: list[str] = []
    window.workspaceChanged.connect(lambda: seen.append("changed"))
    return seen


def _plot_source() -> PlotSource:
    frame = pd.DataFrame({
        "dwell_s": np.linspace(2e-5, 8e-4, 80),
        "depth_pA": np.r_[np.linspace(20.0, 30.0, 40),
                           np.linspace(60.0, 70.0, 40)],
        "accepted": np.ones(80, dtype=bool),
    })
    return PlotSource(
        key="saved-source", doc_key="/data/saved.abf", label="saved source",
        df=frame, provenance={}, colour="#0072B2", doc_id="launch-doc-id",
        channel=0, module_id="detect.threshold",
        calibration_fingerprint=(1.0, 0.0))


def _forbid(*_args, **_kwargs):
    raise AssertionError("workspace snapshot rendering must not run an analysis kernel")


def test_plots_workspace_snapshot_draws_saved_histogram_fit_without_refitting(
        qtbot, monkeypatch):
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    source = _plot_source()
    window.set_data([source])
    window.set_histogram_column("dwell_s")
    window._applied_fit = plots_panel._FIT_EXP
    window._applied_fit_k = 1
    tau = 2.5e-4
    fit = FitResult(
        params={"tau_1": tau, "w_1": 1.0}, ci={}, loglik=12.0,
        aic=-20.0, bic=-17.0, n=len(source.df), model="saved exponential",
        meta={"param_units": {"tau_1": "s", "w_1": ""}, "warnings": []},
        _pdf=lambda x: np.exp(-np.asarray(x, dtype=float) / tau) / tau)
    window._fits = {source.key: fit}
    window._fit_columns = {source.key: "dwell_s"}
    window._last_fit = fit

    for name in ("fit_exponential", "fit_gaussian_mixture", "linear_fit",
                 "pca", "cluster"):
        monkeypatch.setattr(populations, name, _forbid)
    monkeypatch.setattr(window, "_goodness_note", _forbid)
    monkeypatch.setattr(window, "_rival_family_note", _forbid)
    seen = _emissions(window)

    window._render_workspace_snapshot()

    assert window._hist_item is not None
    assert window._fit_item is not None
    assert len(window.plot_item().listDataItems()) >= 2
    assert window.fit_results()[source.key] is fit
    assert "restored snapshot; not recomputed" in window.details_text()
    assert seen == []


def test_plots_workspace_snapshot_draws_saved_pca_clusters_without_running_solvers(
        qtbot, monkeypatch):
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    source = _plot_source()
    window.set_data([source])
    cfg = replace(
        window.config(), plot_type=plots_panel.PCA,
        cluster=replace(window.config().cluster,
                        features=("dwell_s", "depth_pA")))
    window._set_config(cfg, draw=False, own=True)
    kept = source.df.index.copy()
    window._pca_result = populations.PCAResult(
        scores=np.column_stack((np.linspace(-2.0, 2.0, len(kept)),
                                np.linspace(1.0, -1.0, len(kept)))),
        kept_index=kept, explained_variance_ratio=np.array([0.75, 0.20]),
        components=np.array([[0.8, 0.6], [-0.6, 0.8]]),
        columns=["dwell_s", "depth_pA"], n_kept=len(kept),
        meta={"n_dropped": 0, "max_rank": 2})
    labels = pd.Series(np.repeat([0, 1], len(kept) // 2), index=kept)
    window._cluster_result = populations.ClusterResult(
        labels=labels, k=2, method="kmeans",
        columns=["dwell_s", "depth_pA"], sizes={0: 40, 1: 40},
        means=pd.DataFrame({"dwell_s": [2e-4, 6e-4],
                            "depth_pA": [25.0, 65.0]}),
        bic=None, chosen_k=2, n_kept=len(kept),
        warnings=(), meta={"n_dropped": 0, "random_state": 17})

    for name in ("fit_exponential", "fit_gaussian_mixture", "linear_fit",
                 "pca", "cluster"):
        monkeypatch.setattr(populations, name, _forbid)
    seen = _emissions(window)

    window._render_workspace_snapshot()

    assert window._scatter_item is not None
    assert len(window._scatter_item.data) == len(source.df)
    assert sum("cluster" in label for label in window.legend_labels()) == 2
    assert seen == []


def test_plots_close_requests_clean_checkpoint_before_reset(qtbot):
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    window.set_data([_plot_source()])
    order: list[tuple[str, bool]] = []
    changed: list[bool] = []
    window.workspaceCheckpointRequested.connect(
        lambda: order.append(("checkpoint", bool(window._sources))))
    window.workspaceChanged.connect(lambda: changed.append(True))

    window.closeEvent(QtGui.QCloseEvent())

    assert order == [("checkpoint", True)]
    assert changed == []
    assert window._sources == {}


def test_iv_programmatic_refresh_is_silent_but_button_refresh_emits_afterward(qtbot):
    window = IVWindow(Session())
    qtbot.addWidget(window)
    seen = _emissions(window)

    window.refresh()
    assert seen == []

    order: list[str] = []
    original_refresh = window.refresh

    def observed_refresh() -> None:
        order.append("refresh")
        original_refresh()

    window.refresh = observed_refresh  # type: ignore[method-assign]
    window.workspaceChanged.connect(lambda: order.append("changed"))
    window._refresh_btn.click()

    assert seen == ["changed"]
    assert order == ["refresh", "changed"]


def test_vdep_programmatic_restore_is_silent_and_user_controls_emit_once(qtbot):
    window = VDepWindow(Session())
    qtbot.addWidget(window)
    seen = _emissions(window)

    # These are the same public calls project restoration uses.
    window.set_metric("median_dwell_s")
    window.set_fit("linear")
    window.set_abs_voltage(True)
    window.set_log_y(True)
    window.set_accepted_only(False)
    window.refresh()
    assert seen == []

    # QComboBox.activated and QAbstractButton.clicked are user-only in Qt.  Emit
    # the former directly so the test does not depend on a platform popup.
    window._metric_combo.activated.emit(window._metric_combo.currentIndex())
    assert len(seen) == 1
    window._fit_combo.activated.emit(window._fit_combo.currentIndex())
    assert len(seen) == 2

    window._abs_v.click()
    assert len(seen) == 3
    window._log_y.click()
    assert len(seen) == 4
    window._accepted_only.click()
    assert len(seen) == 5

    # Make a pending proposal and exercise the explicit Apply gate.
    window._applied_fit = "none"
    window._sync_fit_apply_btn()
    assert window._fit_apply_btn.isEnabled()
    window._fit_apply_btn.click()
    assert len(seen) == 6

    window._refresh_btn.click()
    assert len(seen) == 7


def test_iv_workspace_snapshot_renders_without_gathering_or_refitting(qtbot, monkeypatch):
    window = IVWindow(Session())
    qtbot.addWidget(window)
    window._points = [
        iv.IVPoint(-0.5, -50.0, "pA", "nominal", "a", "a.dat"),
        iv.IVPoint(-0.7, -70.0, "pA", "nominal", "b", "b.dat"),
    ]
    window._fit = iv.IVFit(
        slope=100.0, intercept=0.0, r2=1.0, n=2,
        current_units="pA", conductance_nS=0.1)
    window._details.setPlainText("saved I-V details")
    window._headline.setText("saved I-V headline")

    monkeypatch.setattr(
        iv, "iv_fit", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot restore must not refit")))
    window._render_workspace_snapshot()

    assert window._scatter_item is not None
    assert window._fit_item is not None
    assert window.details_text() == "saved I-V details"
    assert window._headline.text() == "saved I-V headline"


def test_vdep_workspace_snapshot_renders_without_gathering_or_refitting(qtbot, monkeypatch):
    window = VDepWindow(Session())
    qtbot.addWidget(window)
    points = (
        voltage_dep.VDepPoint(-0.5, 10.0, "a.dat"),
        voltage_dep.VDepPoint(-0.7, 20.0, "b.dat"),
    )
    window._series = voltage_dep.vdep_comparability(
        points, metric="capture_rate_hz")
    window._points = list(points)
    window._fit = {
        "model": "linear",
        "params": {"slope": -100.0, "intercept": -40.0},
        "r2": 1.0,
        "n": 2,
    }
    window._not_in_fit = {"b.dat": "saved exclusion reason"}
    window._details.setPlainText("saved voltage-dependence details")
    window._headline.setText("saved voltage-dependence headline")
    window._set_warnings(("saved voltage-dependence warning",))

    monkeypatch.setattr(
        voltage_dep, "vdep_fit", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot restore must not refit")))
    window._render_workspace_snapshot()

    assert window._scatter_item is not None
    assert window._fit_item is not None
    assert window.details_text() == "saved voltage-dependence details"
    assert window._headline.text() == "saved voltage-dependence headline"
    assert window.warning_text() == "saved voltage-dependence warning"
    assert window._not_in_fit == {"b.dat": "saved exclusion reason"}
