"""The report editor launches from the current source and never silently fits it."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from nanoscope.analysis.events import EventSet, SCHEMA
from nanoscope.core import analysis_workspace
from nanoscope.core.figure_style import builtin_styles
from nanoscope.gui import report_figure_window
from nanoscope.gui.main_window import MainWindow, SURFACE_FIGURES
from nanoscope.plot import figures


@pytest.fixture
def launch(monkeypatch):
    frame = pd.DataFrame({key: pd.Series([], dtype=spec[0]) for key, spec in SCHEMA.items()})
    result = SimpleNamespace(payload=EventSet(frame), doc_id="recording", channel=0)
    source = {"source_key": "durable-source", "channel": 0, "result_fingerprint": "A"}
    style = builtin_styles()[0]
    workspace = SimpleNamespace(style_registry=lambda: object(), default_style=lambda: style)
    messages, opened, built = [], [], []
    owner = SimpleNamespace(
        _analysis_panel=SimpleNamespace(current_detection=lambda: result),
        _session=SimpleNamespace(has_document=lambda _: True,
            calibrated=lambda *_: None, x_axis=lambda _: SimpleNamespace(is_time=False)),
        _figure_project_epoch=3, _refuse_stale_export=lambda _: False,
        _ensure_figure_workspace=lambda: workspace,
        _stored_fit=lambda *_: (None, "not computed"),
        statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None))
    monkeypatch.setattr(analysis_workspace, "result_recipe", lambda *_: deepcopy(source))
    for name in ("warning", "information"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            lambda _owner, title, message: messages.append(message))
    def builder(events, **kwargs):
        figure = Figure()
        figure.subplots().plot([1, 2], [3, 4])
        figure._nanoscope_source_data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        built.append((figure, kwargs))
        return figure, figure.axes[0]
    monkeypatch.setattr(figures, "dwell_histogram", builder)
    monkeypatch.setattr(figures, "depth_histogram", builder)
    monkeypatch.setattr(figures, "scatter_dwell_depth", builder)
    def open_editor(owner, figure, data, **kwargs):
        opened.append((figure, data.copy(deep=True), deepcopy(kwargs)))
        return "editor"
    monkeypatch.setattr(report_figure_window, "open_report_figure", open_editor)
    return SimpleNamespace(owner=owner, result=result, source=source, style=style,
                           messages=messages, opened=opened, built=built, builder=builder)


@pytest.mark.parametrize("kind", ["dwell", "depth"])
def test_missing_fit_is_false_and_default_look_is_pinned(launch, kind):
    assert MainWindow._edit_report_figure(launch.owner, kind=kind) == "editor"
    assert len(launch.built) == len(launch.opened) == 1
    figure, kwargs = launch.built[0]
    assert kwargs == {"fit": False}
    assert launch.opened[0][0] is figure and figure.axes
    pd.testing.assert_frame_equal(launch.opened[0][1], figure._nanoscope_source_data)
    details = launch.opened[0][2]
    assert details["provenance"]["source"] == launch.source
    assert details["style"]["style_binding"]["ref"] == launch.style.ref.to_dict()
    assert not launch.messages


def test_existing_fit_is_passed_exactly_once(launch):
    stored = object()
    launch.owner._stored_fit = lambda *_: (stored, "computed")
    MainWindow._edit_report_figure(launch.owner, kind="dwell")
    assert len(launch.built) == 1
    assert launch.built[0][1]["fit"] is stored


def test_source_change_inside_chooser_prevents_generation(launch, monkeypatch):
    def choose(*args):
        launch.source["result_fingerprint"] = "B"
        return "Dwell histogram", True
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", choose)
    assert MainWindow._edit_report_figure(launch.owner) is None
    assert not launch.built and not launch.opened
    assert "source changed" in launch.messages[0]


def test_trace_uses_calibration_after_the_chooser(launch, monkeypatch):
    calibration = [object()]
    original = calibration[0]
    axis = SimpleNamespace(is_time=True)
    launch.owner._session.calibrated = lambda *_: calibration[0]
    launch.owner._session.x_axis = lambda *_: axis
    def choose(*args):
        calibration[0] = object()
        return "Trace preview", True
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", choose)
    observed = []
    def trace(channel, time_axis, *, events):
        observed.append((channel, time_axis, events))
        return launch.builder(events)
    monkeypatch.setattr(figures, "trace_preview", trace)
    assert MainWindow._edit_report_figure(launch.owner) == "editor"
    assert observed == [(calibration[0], axis, launch.result.payload)]
    assert observed[0][0] is not original


def test_project_change_during_builder_clears_untransferred_figure(launch, monkeypatch):
    def changing_builder(events, **kwargs):
        built = launch.builder(events, **kwargs)
        launch.owner._figure_project_epoch += 1
        return built
    monkeypatch.setattr(figures, "dwell_histogram", changing_builder)
    assert MainWindow._edit_report_figure(launch.owner, kind="dwell") is None
    assert not launch.opened
    assert launch.built[0][0].axes == []
    assert "source changed" in launch.messages[0]


def test_stale_result_is_refused_before_building(launch):
    launch.owner._refuse_stale_export = lambda _: True
    assert MainWindow._edit_report_figure(launch.owner, kind="scatter") is None
    assert not launch.built and not launch.opened


def test_unreadable_registry_does_not_install_a_replacement_default(launch):
    launch.owner._ensure_figure_workspace = lambda: SimpleNamespace(style_registry=lambda: None)
    assert MainWindow._edit_report_figure(launch.owner, kind="scatter") == "editor"
    assert launch.opened[0][2]["style"] is None


def test_editor_and_export_share_one_gate_evaluation():
    evaluated, enabled, published = [], [], []
    def gate():
        evaluated.append(True)
        return False, "No detection"
    owner = SimpleNamespace(
        _surface_gates=lambda: {SURFACE_FIGURES: gate},
        _surface_actions=lambda: {SURFACE_FIGURES: SimpleNamespace(setEnabled=lambda x: enabled.append(("export", x)))},
        _act_edit_report_figure=SimpleNamespace(setEnabled=lambda x: enabled.append(("editor", x))),
        _analysis_launcher=SimpleNamespace(set_gate=lambda *args: published.append(args)))
    MainWindow._apply_surface_gate(owner, SURFACE_FIGURES)
    assert evaluated == [True]
    assert enabled == [("export", False), ("editor", False)]
    assert published == [(SURFACE_FIGURES, False, "No detection")]
