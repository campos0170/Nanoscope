"""Focused contracts for the Stage-3 analytical-workspace adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nanoscope.analysis.populations import ClusterResult, PCAResult
from nanoscope.analysis.stats import FitResult
from nanoscope.core import analysis_workspace as workspace
from nanoscope.core.project import AnalyticalWorkspace, WorkspaceArtifact
from nanoscope.core.types import ChannelInfo, ChannelKind


@dataclass(frozen=True)
class _Selection:
    source_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Config:
    selection: _Selection = field(default_factory=_Selection)
    title: str = ""


class _ResultStore:
    def __init__(self, results=()):
        self._results = {
            (result.doc_id, result.channel, result.module_id): result for result in results
        }

    def get(self, doc_id, channel, module_id):
        return self._results.get((doc_id, channel, module_id))


class _Session:
    def __init__(self, documents=(), results=()):
        self._documents = tuple(documents)
        self.results = _ResultStore(results)

    def documents(self):
        return self._documents

    def document(self, doc_id):
        return next(doc for doc in self._documents if doc.doc_id == doc_id)


def _document(source_key: str, doc_id: str, channels: tuple[ChannelInfo, ...]):
    return SimpleNamespace(
        source_key=source_key,
        doc_id=doc_id,
        display_name=Path(source_key).name,
        meta=SimpleNamespace(channels=channels),
    )


def _source(key: str, frame: pd.DataFrame, *, doc=None, channel=None,
            module_id=None, kind="detection"):
    return SimpleNamespace(
        key=key,
        doc_key=("manual-measurements" if kind == "manual" else doc.source_key),
        label=key,
        df=frame,
        doc_id=None if doc is None else doc.doc_id,
        channel=channel,
        module_id=module_id,
        kind=kind,
    )


class _PlotWindow:
    def __init__(self, sources, selected, *, session=None, cluster=None, pca=None, fits=None):
        self._sources = {source.key: source for source in sources}
        self._session = session or _Session()
        self._config = _Config(_Selection(tuple(selected)), "saved title")
        self._cluster_result = cluster
        self._pca_result = pca
        self._fits = dict(fits or {})
        self._fit_columns = {key: "a" for key in self._fits}
        self._last_fit = None
        self.redraws = 0

    def config(self):
        return self._config

    def apply_config(self, config):
        self._config = config

    def selected_source_keys(self):
        return [key for key in self._config.selection.source_keys if key in self._sources]

    def _current_frame(self):
        selected = self.selected_source_keys()
        return self._sources[selected[0]].df if selected else None

    def cluster_result(self):
        return self._cluster_result

    def pca_result(self):
        return self._pca_result

    def _armed_redraw(self):
        self.redraws += 1


def _derived(frame: pd.DataFrame):
    cluster = ClusterResult(
        labels=pd.Series([0, 1], index=frame.index),
        k=2,
        method="kmeans",
        columns=["a", "b"],
        sizes={0: 1, 1: 1},
        means=pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 5.0]}),
        bic=None,
        chosen_k=2,
        n_kept=2,
        meta={"standardize": True, "random_state": 37, "n_init": 20},
    )
    pca = PCAResult(
        scores=np.array([[-1.0], [1.0]]),
        kept_index=frame.index.copy(),
        explained_variance_ratio=np.array([0.875]),
        components=np.array([[0.6, 0.8]]),
        columns=["a", "b"],
        n_kept=2,
        meta={"standardize": True, "random_state": 19},
    )
    fit = FitResult(
        params={"mu": 1.5}, ci={"mu": (1.0, 2.0)}, loglik=-4.0,
        aic=12.0, bic=13.0, n=2, model="test fit",
        meta={"family": "unknown-test-family", "seed": 5},
        _pdf=lambda x: np.ones_like(np.asarray(x, dtype=float)),
    )
    return cluster, pca, fit


def test_plots_capture_keeps_bulk_outputs_in_tables_and_restores_exact_values():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 5.0]}, index=[10, 20])
    source = _source("manual-measurements", frame, kind="manual")
    cluster, pca, fit = _derived(frame)
    before = _PlotWindow([source], [source.key], cluster=cluster, pca=pca,
                         fits={source.key: fit})

    artifact = workspace.capture_plots(before)

    assert artifact.payload["pca"]["scores_table"] in artifact.tables
    assert artifact.payload["pca"]["loadings_table"] in artifact.tables
    assert artifact.payload["cluster"]["labels_table"] in artifact.tables
    assert "scores" not in artifact.payload["pca"]
    assert artifact.payload["pca"]["random_state"] == 19
    assert artifact.payload["cluster"]["random_state"] == 37

    after = _PlotWindow([source], [source.key])
    report = workspace.restore_plots(after, artifact)

    assert report.skipped == (), report.warnings
    assert np.array_equal(after._pca_result.scores, pca.scores)
    assert np.array_equal(after._pca_result.components, pca.components)
    assert after._cluster_result.labels.equals(cluster.labels)
    assert after._cluster_result.meta == cluster.meta
    assert after._fits[source.key].params == fit.params
    assert after._fit_columns == {source.key: "a"}


def test_plot_source_recipe_relocates_a_reordered_channel(tmp_path):
    path = str(tmp_path / "recording.abf")
    old_channels = (
        ChannelInfo(0, "current", "pA", ChannelKind.CURRENT),
        ChannelInfo(1, "command", "mV", ChannelKind.VOLTAGE),
    )
    new_channels = (
        ChannelInfo(0, "command", "mV", ChannelKind.VOLTAGE),
        ChannelInfo(1, "current", "pA", ChannelKind.CURRENT),
    )
    old_doc = _document(path, "old-load", old_channels)
    new_doc = _document(path, "new-load", new_channels)
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    old_source = _source(f"{path}#0", frame, doc=old_doc, channel=0,
                         module_id="detect.schmitt")
    new_source = _source(f"{path}#1", frame, doc=new_doc, channel=1,
                         module_id="detect.schmitt")
    artifact = workspace.capture_plots(
        _PlotWindow([old_source], [old_source.key], session=_Session([old_doc])))

    target = _PlotWindow([new_source], [], session=_Session([new_doc]))
    report = workspace.restore_plots(target, artifact)

    assert report.skipped == ()
    assert target.config().selection.source_keys == (new_source.key,)
    assert any("now channel 1" in warning for warning in report.warnings)


def test_malformed_numerical_resource_cannot_abort_plot_configuration_restore():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 5.0]})
    source = _source("manual-measurements", frame, kind="manual")
    cluster, pca, _fit = _derived(frame)
    artifact = workspace.capture_plots(
        _PlotWindow([source], [source.key], cluster=cluster, pca=pca))
    broken = copy.deepcopy(artifact)
    del broken.tables[broken.payload["pca"]["scores_table"]]
    target = _PlotWindow([source], [source.key])

    report = workspace.restore_plots(target, broken)

    assert report.restored == (workspace.PLOTS_ARTIFACT_ID,)
    assert target.config().title == "saved title"
    assert target._cluster_result is None and target._pca_result is None
    assert any("invalid" in warning for warning in report.warnings)


def test_plot_restore_uses_draw_free_config_install_and_never_treats_open_as_run():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 5.0]})
    source = _source("manual-measurements", frame, kind="manual")
    cluster, pca, _fit = _derived(frame)
    artifact = workspace.capture_plots(
        _PlotWindow([source], [source.key], cluster=cluster, pca=pca))

    class NoRunWindow(_PlotWindow):
        def _set_config(self, config, *, draw=True, own=False):
            assert draw is False and own is True
            self._config = config

        def apply_config(self, _config):
            raise AssertionError("project open used the run-gesture API")

        def _armed_redraw(self):
            raise AssertionError("project open recomputed a plot")

    target = NoRunWindow([source], [source.key])

    assert workspace.restore_plots(target, artifact).skipped == ()
    assert target._pca_result is not None and target._cluster_result is not None


def test_unavailable_plot_outputs_and_sources_survive_an_open_resave_cycle():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 5.0]})
    source = _source("manual-measurements", frame, kind="manual")
    cluster, pca, fit = _derived(frame)
    original = workspace.capture_plots(
        _PlotWindow([source], [source.key], cluster=cluster, pca=pca,
                    fits={source.key: fit}))
    unavailable = _PlotWindow([], [])

    report = workspace.restore_plots(unavailable, original)
    recaptured = workspace.capture_plots(unavailable)

    assert report.restored == (workspace.PLOTS_ARTIFACT_ID,)
    assert recaptured.payload["cluster"] is not None
    assert recaptured.payload["pca"] is not None
    assert len(recaptured.payload["fits"]) == 1
    assert recaptured.payload["selected_sources"] == original.payload["selected_sources"]
    for name in (recaptured.payload["cluster"]["labels_table"],
                 recaptured.payload["pca"]["scores_table"]):
        assert name in recaptured.tables


def test_capture_workspace_preserves_plugins_and_future_owned_schema():
    plugin = WorkspaceArtifact(
        "org.example.embedding", "org.example.future", schema_version=7,
        payload={"future": [1, 2]}, tables={"scores": pd.DataFrame({"x": [1.0]})},
        blobs={"opaque": b"plugin-state"}, extra={"vendor": {"revision": 9}})
    future_builtin = WorkspaceArtifact(
        workspace.PLOTS_ARTIFACT_ID, "nanoscope.plots-v2", schema_version=2,
        payload={"do_not_downgrade": True})
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    manual = _source("manual-measurements", frame, kind="manual")

    captured = workspace.capture_analysis_workspace(
        AnalyticalWorkspace(version=3, artifacts=[plugin, future_builtin],
                            extra={"future_layout": {"columns": 4}}),
        plots=_PlotWindow([manual], [manual.key]))

    assert [item.artifact_id for item in captured.artifacts] == [
        plugin.artifact_id, workspace.PLOTS_ARTIFACT_ID]
    assert captured.artifacts[1].kind == "nanoscope.plots-v2"
    assert captured.artifacts[1].payload == {"do_not_downgrade": True}
    assert captured.artifacts[0].tables["scores"].equals(plugin.tables["scores"])
    assert captured.version == 3
    assert captured.extra == {"future_layout": {"columns": 4}}
    assert captured.artifacts[0].blobs == {"opaque": b"plugin-state"}
    assert captured.artifacts[0].extra == {"vendor": {"revision": 9}}
    assert captured.artifacts[0].tables["scores"] is not plugin.tables["scores"]


def test_subpopulation_artifact_exposes_durable_result_lineage(tmp_path):
    from nanoscope.core.analysis import AnalysisResult

    path = str(tmp_path / "source.abf")
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    document = _document(path, "launch-local-id", (info,))
    frame = pd.DataFrame({"event_id": [0, 1], "accepted": [True, True]})
    result = AnalysisResult(
        module_id="detect.schmitt", module_version="3.0", doc_id=document.doc_id,
        channel=0, params={"threshold": 5.0}, calibration_fingerprint=(1.0, 0.0),
        payload=SimpleNamespace(df=frame, provenance={"recording": "source.abf"}),
    )

    class View:
        def embedding(self):
            return "pca"

        def colour_mode(self):
            return "cluster"

    class Window:
        _nanoscope_workspace_source_result = result
        _nanoscope_workspace_accepted_event_ids = (9, 2, 5, 2)

        def __init__(self):
            self._owner = SimpleNamespace(_session=_Session([document], [result]))

        def parent(self):
            return self._owner

        def report(self):
            return None

        def embedding_view(self):
            return View()

        def assignment_state(self):
            return "attached", ""

        def header(self):
            return "Subpopulations"

        def curation_note(self):
            return ""

    launch_recipe = workspace.result_recipe(Window(), result)
    artifact = workspace.capture_subpopulation(Window())
    recipe = workspace.artifact_source_recipe(artifact)

    assert recipe["source_key"] == path
    assert recipe["channel"] == 0
    assert recipe["channel_name"] == "current"
    assert recipe["module_id"] == "detect.schmitt"
    assert recipe["module_version"] == "3.0"
    assert recipe["event_frame_fingerprint"] == workspace.fingerprint_frame(frame)
    assert recipe["result_fingerprint"]
    assert "doc_id" not in recipe
    assert artifact.payload["accepted_event_ids"] == [2, 5, 9]
    resolved, warning = workspace.resolve_result_recipe(Window(), recipe)
    assert resolved is result
    assert warning is None

    # An asynchronous report may land after the recording/result disappears.  The
    # durable recipe captured at launch wins over the now-unresolvable live result.
    late_window = Window()
    late_window._nanoscope_workspace_source = launch_recipe
    late_window._owner = SimpleNamespace(_session=_Session([], []))
    late_artifact = workspace.capture_subpopulation(late_window)
    assert workspace.artifact_source_recipe(late_artifact) == launch_recipe

    class RestoreView:
        def set_embedding(self, value):
            self.embedding = value

        def set_colour_mode(self, value):
            self.colour = value

    class RestoreWindow:
        def __init__(self):
            self.view = RestoreView()

        def set_report(self, report, subtitle=""):
            self.report = report
            self.subtitle = subtitle

        def embedding_view(self):
            return self.view

        def set_assignment_state(self, state, reason):
            self.assignment = state, reason

        def set_curation_note(self, value):
            self.note = value

    restored_window = RestoreWindow()
    assert workspace.restore_subpopulation(restored_window, artifact).skipped == ()
    assert restored_window._nanoscope_workspace_accepted_event_ids == (2, 5, 9)


def test_result_recipe_rejects_every_stale_lineage_dimension(tmp_path):
    from nanoscope.core.analysis import AnalysisResult

    path = str(tmp_path / "source.abf")
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    document = _document(path, "live-id", (info,))
    frame = pd.DataFrame({"event_id": [0], "accepted": [True]})
    result = AnalysisResult(
        module_id="detect.schmitt", module_version="3.0", doc_id=document.doc_id,
        channel=0, params={"threshold": 5.0}, calibration_fingerprint=(1.0, 0.0),
        payload=SimpleNamespace(df=frame, provenance={"recording": "source.abf"}),
    )
    owner = SimpleNamespace(_session=_Session([document], [result]))
    recipe = workspace.result_recipe(owner, result)

    mutations = (
        ("module_id", "detect.other", "not available"),
        ("module_version", "99", "module version"),
        ("event_frame_fingerprint", "0" * 64, "event-table content"),
        ("result_fingerprint", "f" * 64, "different parameters"),
    )
    for field, value, expected_warning in mutations:
        stale = copy.deepcopy(recipe)
        stale[field] = value
        resolved, warning = workspace.resolve_result_recipe(owner, stale)
        assert resolved is None
        assert expected_warning in warning


def test_model_fingerprint_cache_hits_once_and_detects_same_size_rewrite_with_restored_mtime(
        tmp_path, monkeypatch):
    model = tmp_path / "detector.keras"
    model.write_bytes(b"AAAA")
    first_stat = model.stat()
    workspace._MODEL_FINGERPRINT_CACHE.clear()
    real_fingerprint = workspace.file_fingerprint
    calls = []

    def counted(path, **kwargs):
        calls.append(Path(path))
        return real_fingerprint(path, **kwargs)

    monkeypatch.setattr(workspace, "file_fingerprint", counted)
    first = workspace.capture_model_reference(model)
    assert workspace.capture_model_reference(model) == first
    assert len(calls) == 1
    time.sleep(0.01)
    model.write_bytes(b"BBBB")
    os.utime(model, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))

    second = workspace.capture_model_reference(model)

    assert len(calls) == 2
    assert first["fingerprint"]["size"] == second["fingerprint"]["size"]
    assert first["fingerprint"]["sha256"] != second["fingerprint"]["sha256"]


def test_explicit_model_clear_retires_carried_unavailable_reference(qapp, qtbot):
    from nanoscope.gui.panels import DetectionPanel

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel._nanoscope_unresolved_model_reference = {
        "path": "/missing/student.keras", "fingerprint": None, "status": "missing",
    }

    panel.set_model_path(None)

    assert panel.model_path() is None
    assert not hasattr(panel, "_nanoscope_unresolved_model_reference")


def test_missing_model_reference_survives_restore_signal_and_resave_until_explicit_clear(
        qapp, qtbot, tmp_path):
    from PySide6 import QtWidgets
    from nanoscope.gui.panels import DetectionPanel

    detection = DetectionPanel()
    qtbot.addWidget(detection)

    class Baseline:
        def __init__(self):
            self.show_track = QtWidgets.QCheckBox()
            self.subtract_view = QtWidgets.QCheckBox()
            self._params = {}

        def params(self):
            return dict(self._params)

        def set_params(self, value):
            self._params = dict(value)

    class Measure:
        def __init__(self):
            self._method_combo = QtWidgets.QComboBox()
            self._method_combo.addItem("median")
            self._scope_check = QtWidgets.QCheckBox()
            self._markers_check = QtWidgets.QCheckBox()
            self._markers_check.setChecked(True)
            self._basis = "raw"
            self._active = False

        def measurement_basis(self):
            return self._basis

        def is_active(self):
            return self._active

        def markers_visible(self):
            return self._markers_check.isChecked()

        def set_method(self, value):
            self._method_combo.setCurrentText(str(value))

        def set_measurement_basis(self, value):
            self._basis = str(value)

        def set_scoped(self, value):
            self._scope_check.setChecked(bool(value))

        def set_active(self, value):
            self._active = bool(value)

    class Merge:
        def __init__(self):
            self.value = 0.01

        def window_s(self):
            return self.value

        def set_window_s(self, value):
            self.value = float(value)

    class Panel:
        def __init__(self):
            self.detection_panel = detection
            self.baseline_panel = Baseline()
            self.measure_panel = Measure()
            self.merge_controls = Merge()
            self._active_detection = "detect.schmitt"
            self._stats = SimpleNamespace(
                n_components_dwell=QtWidgets.QSpinBox(),
                n_components_depth=QtWidgets.QSpinBox(),
                use_efficiency=QtWidgets.QCheckBox(),
                use_filter_limited=QtWidgets.QCheckBox(),
            )

        def current_target(self):
            return None

        def current_detection(self):
            return None

    panel = Panel()
    artifact = workspace.capture_analysis_panel(panel)
    missing_reference = {
        "path": str(tmp_path / "missing.keras"),
        "fingerprint": {"size": 42, "mtime_ns": 7, "sha256": "a" * 64},
        "status": "ok",
    }
    artifact.payload["ml_model"] = copy.deepcopy(missing_reference)
    seen_during_signal = []
    detection.mlModelChanged.connect(
        lambda _path: seen_during_signal.append(
            workspace.capture_analysis_panel(panel).payload["ml_model"]))

    report = workspace.restore_analysis_panel(panel, artifact)

    assert report.skipped == (), report.warnings
    assert detection.model_path() is None
    assert seen_during_signal[-1] == missing_reference
    assert workspace.capture_analysis_panel(panel).payload["ml_model"] == missing_reference

    detection.set_model_path(None)

    assert workspace.capture_analysis_panel(panel).payload["ml_model"] is None


def test_deferred_analysis_window_constructors_run_no_gather_or_numeric_kernel(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import iv as iv_module
    from nanoscope.analysis import psd as psd_module
    from nanoscope.analysis import voltage_dep as vdep_module
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.session import Session
    from nanoscope.core.types import Provenance, RecordingMeta
    from nanoscope.gui.iv_panel import IVWindow
    from nanoscope.gui.psd_panel import PSDWindow
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    samples = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "deferred.abf", format_name="synthetic", n_samples=len(samples),
        sample_rate_hz=10_000.0, rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    session.add_document(TraceDocument(meta, (samples,)))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("deferred constructor invoked an analysis gather/kernel")

    monkeypatch.setattr(psd_module, "welch_psd", forbidden)
    monkeypatch.setattr(iv_module, "iv_points", forbidden)
    monkeypatch.setattr(iv_module, "iv_exclusions", forbidden)
    monkeypatch.setattr(vdep_module, "vdep_series", forbidden)

    psd_window = PSDWindow(session, defer_initial_refresh=True)
    iv_window = IVWindow(session, defer_initial_refresh=True)
    vdep_window = VDepWindow(session, defer_initial_refresh=True)
    for window in (psd_window, iv_window, vdep_window):
        qtbot.addWidget(window)

    assert len(psd_window._recordings) == 1
    assert psd_window.curves() == []
    assert iv_window.points() == []
    assert vdep_window.points() == []


def test_psd_restore_paints_saved_auxiliary_view_without_invoking_analysis_kernels(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import allpoints as allpoints_module
    from nanoscope.analysis import psd as psd_module
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.session import Session
    from nanoscope.core.types import Provenance, RecordingMeta
    from nanoscope.gui.psd_panel import (
        PSDWindow, VIEW_ALLPOINTS, VIEW_FFT, VIEW_SPECTROGRAM,
    )

    samples = (np.random.default_rng(4).normal(size=8192) + 20.0).astype(np.float32)
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "psd.abf", format_name="synthetic", n_samples=len(samples),
        sample_rate_hz=20_000.0, rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta, (samples,)))
    window = PSDWindow(session)
    qtbot.addWidget(window)
    window.set_target(doc_id, 0)
    window.set_view(VIEW_FFT)
    window.set_view(VIEW_SPECTROGRAM)
    window.set_view(VIEW_ALLPOINTS)
    artifact = workspace.capture_psd(window)
    saved_levels = copy.deepcopy(window.hist_levels())

    # Restore into a project where the source is unavailable.  The saved result is still
    # self-contained and must remain viewable/exportable without touching a trace.
    restored = PSDWindow(Session())
    qtbot.addWidget(restored)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("restore invoked an analysis kernel")

    monkeypatch.setattr(psd_module, "welch_psd", forbidden)
    monkeypatch.setattr(psd_module, "fft_magnitude", forbidden)
    monkeypatch.setattr(psd_module, "spectrogram_psd", forbidden)
    monkeypatch.setattr(allpoints_module, "all_points_histogram", forbidden)
    monkeypatch.setattr(allpoints_module, "level_peaks", forbidden)

    report = workspace.restore_psd(restored, artifact)

    assert report.skipped == ()
    assert restored.current_view() == VIEW_ALLPOINTS
    assert restored.fft_spectrum() is not None
    assert restored.spectrogram() is not None
    assert restored.all_points() is not None
    assert restored.curve_item() is not None
    assert restored._hist_item is not None
    assert restored.caption_text(VIEW_ALLPOINTS)
    assert restored.hist_levels() == saved_levels
    assert [float(mark.value()) for mark in restored._hist_marks] == [
        float(level["center"]) for level in saved_levels]


def test_event_plot_restore_paints_saved_grids_without_invoking_analysis_kernels(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import event_shape
    from nanoscope.analysis.events import EventSet, SCHEMA, empty_df
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.session import Session
    from nanoscope.core.types import Provenance, RecordingMeta
    from nanoscope.gui.event_plot_panel import EventPlotWindow

    fs = 10_000.0
    samples = np.full(8_000, 1000.0, dtype=np.float32)
    rows = []
    for event_id, (start, length, depth) in enumerate(
            ((300, 40, 80.0), (1200, 65, 120.0), (3000, 90, 55.0))):
        samples[start:start + length] -= depth
        rows.append({
            "event_id": event_id, "recording": "events.abf", "pore_id": "P",
            "start_idx": start, "end_idx": start + length,
            "t_start_s": start / fs, "dwell_s": length / fs,
            "baseline_pA": 1000.0, "depth_pA": depth,
            "frac_depth": depth / 1000.0, "ecd_pA_s": depth * length / fs,
            "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1.0 / fs, "fall_s": 1.0 / fs, "snr": 20.0,
            "n_levels": 1, "filter_limited": False, "accepted": True, "flag": "",
        })
    frame = pd.concat([empty_df(), pd.DataFrame(rows)], ignore_index=True)
    frame = frame.astype({name: spec[0] for name, spec in SCHEMA.items()})
    events = EventSet(frame, {"analyzed_duration_s": len(samples) / fs})
    info = ChannelInfo(0, "current", "pA", ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "events.abf", format_name="synthetic", n_samples=len(samples),
        sample_rate_hz=fs, rate_provenance=Provenance.FILE_META, channels=(info,))
    source_session = Session()
    doc_id = source_session.add_document(TraceDocument(meta, (samples,)))
    source = EventPlotWindow(source_session)
    qtbot.addWidget(source)
    source.set_source(doc_id, 0, events)
    source.set_normalize_time(False)
    artifact = workspace.capture_event_plots(source)
    saved_counts = source.density_counts()
    saved_edges = source.density_edges()
    saved_ensemble = tuple(value.copy() for value in source.ensemble_traces())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("restore invoked an event-plot analysis kernel")

    monkeypatch.setattr(event_shape, "density_2d", forbidden)
    monkeypatch.setattr(event_shape, "event_ensemble", forbidden)
    monkeypatch.setattr(event_shape, "charge_deficit", forbidden)

    # MainWindow uses this bind-only path immediately before adapter restore.  It
    # must attach the exact live source without calculating a replacement first.
    bound = EventPlotWindow(source_session)
    qtbot.addWidget(bound)
    bound.set_source(doc_id, 0, events, defer_analysis=True)
    assert bound._doc_id == doc_id
    assert bound._eventset is events
    assert bound._signal is not None
    assert bound.density_counts() is None
    assert bound.ensemble_traces() is None

    restored = EventPlotWindow(Session())
    qtbot.addWidget(restored)

    report = workspace.restore_event_plots(restored, artifact)

    assert report.skipped == ()
    np.testing.assert_array_equal(restored.density_counts(), saved_counts)
    for actual, expected in zip(restored.density_edges(), saved_edges):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(restored.ensemble_traces(), saved_ensemble):
        np.testing.assert_array_equal(actual, expected)
    assert restored._density_image is not None
    assert restored._top_marginal_item is not None
    assert restored._right_marginal_item is not None
    assert restored._events_item is not None
    assert restored._mean_item is not None
    assert restored._median_item is not None
    assert restored._align_combo.isEnabled()
    rect = restored._density_image.mapRectToParent(restored._density_image.boundingRect())
    np.testing.assert_allclose(rect.left(), np.log10(saved_edges[0][0]))
    np.testing.assert_allclose(rect.right(), np.log10(saved_edges[0][-1]))


def test_iv_and_voltage_dependence_restore_do_not_run_fit_solvers(qapp, qtbot, monkeypatch):
    from nanoscope.analysis import iv as iv_module
    from nanoscope.analysis import voltage_dep as vdep_module
    from nanoscope.core.session import Session
    from nanoscope.gui.iv_panel import IVWindow
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    session = Session()
    iv_window = IVWindow(session)
    vdep_window = VDepWindow(session)
    qtbot.addWidget(iv_window)
    qtbot.addWidget(vdep_window)
    iv_window._points = [
        iv_module.IVPoint(-0.1, -10.0, "pA", "nominal", "missing-a", "A"),
        iv_module.IVPoint(-0.2, -21.0, "pA", "nominal", "missing-b", "B"),
    ]
    iv_window._fit = iv_module.iv_fit(
        np.array([-0.1, -0.2]), np.array([-10.0, -21.0]), current_units="pA")
    points = (
        vdep_module.VDepPoint(-0.1, 10.0, "A"),
        vdep_module.VDepPoint(-0.2, 25.0, "B"),
    )
    vdep_window._series = vdep_module.VDepSeries(
        metric="capture_rate_hz", points=points, depends_on_baseline=False,
        depends_on_gain=False, baseline_groups=(), merge_groups=(), gain_groups=(),
        warnings=())
    vdep_window._points = list(points)
    vdep_window._fit = vdep_module.vdep_fit(
        np.array([-0.1, -0.2]), np.array([10.0, 25.0]), model="linear")
    vdep_window._applied_fit = "linear"
    vdep_window._fit_combo.setCurrentText("linear")
    iv_artifact = workspace.capture_iv(iv_window)
    vdep_artifact = workspace.capture_voltage_dependence(vdep_window)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("restore invoked a fit solver")

    monkeypatch.setattr(iv_module, "iv_fit", forbidden)
    monkeypatch.setattr(vdep_module, "vdep_fit", forbidden)

    iv_window._points = []
    iv_window._fit = None
    vdep_window._points = []
    vdep_window._fit = None
    assert workspace.restore_iv(iv_window, iv_artifact).skipped == ()
    assert workspace.restore_voltage_dependence(vdep_window, vdep_artifact).skipped == ()
    assert len(iv_window.points()) == 2 and iv_window.fit() is not None
    assert len(vdep_window.points()) == 2 and vdep_window.fit() is not None
    assert iv_window._scatter_item is not None and iv_window._fit_item is not None
    assert vdep_window._scatter_item is not None and vdep_window._fit_item is not None
