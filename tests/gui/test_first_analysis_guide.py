"""Onboarding must navigate without silently changing scientific state."""
from contextlib import ExitStack

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_catalog import ANALYSIS_LABELS, PAGE_HOME, PAGE_IDS
from nanoscope.gui.first_analysis_guide import (
    FirstAnalysisGuide, STEPS, practice_path, practice_trace, write_practice_recording,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.workflow_status import Stage


@pytest.fixture
def window(qtbot):
    widget = MainWindow(recovery_enabled=False)
    qtbot.addWidget(widget)
    yield widget
    widget._analysis_controller.abort()
    widget._teardown()


def add_recording(window, tmp_path, name="recording.dat"):
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
        n_samples=2000, sample_rate_hz=10000., rate_provenance=Provenance.FILE_META,
        channels=(info,))
    doc = TraceDocument(meta=meta, channels=(np.ones(2000, dtype=np.float32),))
    doc_id = window._session.add_document(doc)
    window._current_selection = (doc_id, 0)
    window._refresh_workflow_status()
    return doc


def test_guide_navigation_checkmarks_and_context_are_only_learning_notes(qtbot):
    context = ["one", "Recording one"]
    guide = FirstAnalysisGuide(context_provider=lambda: tuple(context))
    qtbot.addWidget(guide)
    with qtbot.assertNotEmitted(guide.actionRequested):
        guide.show()
        guide.check.setChecked(True)
        guide.next.click()
        assert guide.steps.currentRow() == 1
        guide.previous.click()
        assert guide.check.isChecked()
        context[:] = ["two", "Recording two"]
        guide.refresh_context()
    assert not any(guide._checks)
    assert guide.context.text() == "Recording two"
    assert guide.steps.count() == len(STEPS) == 7


def test_guide_rechecks_gate_at_activation(qtbot):
    allowed = [True]
    guide = FirstAnalysisGuide(availability_provider=lambda _: (allowed[0], "Needs a recording"))
    qtbot.addWidget(guide)
    calls = []
    guide.actionRequested.connect(calls.append)
    allowed[0] = False
    guide._buttons[0].click()
    assert calls == []
    allowed[0] = True
    guide._buttons[0].click()
    assert calls == ["open_recording"]


def test_home_and_guide_never_construct_analytical_pages_or_change_state(window, qtbot):
    panel = window._analysis_panel.detection_panel
    before = (window._analysis_panel.detection_params(), window._dirty,
              list(window._undo_journal), list(window._redo_journal))
    with ExitStack() as stack:
        for signal in (window._session.resultsChanged, panel.paramsChanged,
                       panel.detectRequested, panel.mlDetectRequested):
            stack.enter_context(qtbot.assertNotEmitted(signal))
        window._on_stage_activated(Stage.ANALYZE.value)
        explorer = window._analysis_explorer
        assert explorer.current_page_id() == PAGE_HOME
        assert explorer.page_count() == 0
        for route in PAGE_IDS:
            explorer.home().set_current_route(route)
        explorer.refresh_current()
        window._open_first_analysis_guide()
        guide = window._first_analysis_guide
        for row in range(len(STEPS)):
            guide.steps.setCurrentRow(row)
            guide.check.setChecked(True)
        window._sync_analysis_surfaces()
    assert explorer.page_count() == 0
    assert all(getattr(window, attr) is None for attr in (
        "_plots_window", "_distribution_summary_window", "_comparison_figures_window",
        "_psd_window", "_iv_window", "_vdep_window", "_event_plot_window", "_subpop_window"))
    assert (window._analysis_panel.detection_params(), window._dirty,
            list(window._undo_journal), list(window._redo_journal)) == before


def test_guide_control_routes_preserve_science(window, tmp_path, qtbot):
    add_recording(window, tmp_path)
    panel = window._analysis_panel.detection_panel
    before = window._analysis_panel.detection_params()
    journals = (list(window._undo_journal), list(window._redo_journal))
    with qtbot.assertNotEmitted(window._session.resultsChanged):
        with qtbot.assertNotEmitted(panel.paramsChanged):
            for route in ("essential", "baseline", "advanced", "detection", "calibration"):
                window._first_analysis_action(route)
    assert panel.control_level() == "advanced"
    assert window._analysis_panel.detection_params() == before
    assert (list(window._undo_journal), list(window._redo_journal)) == journals
    # Existing tab navigation is persisted as view metadata and can mark a
    # project unsaved. That is not an analysis-parameter or result mutation.
    assert window._dirty


def test_open_subpopulations_does_not_run_mixture(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_run_subpopulations", lambda: calls.append("run"))
    window._launch_analysis_route("subpopulations")
    assert window._analysis_explorer.current_page_id() == "subpopulations"
    assert window._subpop_window.report() is None
    assert calls == []
    window._analysis_explorer.refresh_current()
    assert calls == ["run"]


def test_canonical_menu_destinations_and_live_gate(window, monkeypatch):
    for route, attr in (("distributions", "_act_plots"), ("noise", "_act_psd"),
                        ("event_waveforms", "_act_event_plots"), ("iv", "_act_iv")):
        assert getattr(window, attr).text() == ANALYSIS_LABELS[route] + "…"
    calls = []
    monkeypatch.setattr(window, "_open_psd", lambda: calls.append("noise"))
    window._launch_analysis_route("noise")
    assert calls == []
    monkeypatch.setattr(window, "_gate_psd", lambda: (True, ""))
    window._launch_analysis_route("noise")
    assert calls == ["noise"]


def test_practice_trace_reproducible_and_csv_loads_as_current(tmp_path):
    from nanoscope.io.text import TextTraceReader
    time, current = practice_trace()
    time2, current2 = practice_trace()
    assert len(time) == 50000 and time[-1] == pytest.approx(4.9999)
    np.testing.assert_array_equal(time, time2)
    np.testing.assert_array_equal(current, current2)
    baseline = 100 + 4 * np.sin(2 * np.pi * time / 8)
    generated_noise = np.random.default_rng(20260911).normal(0, 1.2, len(time))
    deviation = current - baseline - generated_noise
    assert np.count_nonzero(np.abs(deviation + 25) < 1e-10) == 1600
    target = write_practice_recording(tmp_path / "example")
    assert target.name == "SYNTHETIC-example.csv"
    assert practice_path(target) == target
    assert practice_path(tmp_path / "non-synthetic-data.csv").name == "SYNTHETIC-non-synthetic-data.csv"
    np.testing.assert_allclose(np.loadtxt(target, delimiter=",", skiprows=1),
                               np.column_stack((time, current)), rtol=0, atol=0)
    doc = TextTraceReader(target).load()
    assert doc.meta.sample_rate_hz == pytest.approx(10000.)
    assert doc.meta.n_samples == 50000
    assert doc.meta.channels[0].kind == ChannelKind.CURRENT
    assert doc.meta.channels[0].raw_units == ""


def test_practice_creation_requires_consent_and_uses_normal_loader(window, tmp_path, monkeypatch):
    from nanoscope.gui import main_window
    calls = []
    monkeypatch.setattr(window, "open_paths", lambda paths: calls.append(paths))
    monkeypatch.setattr(main_window, "ask_export_path", lambda *args, **kw: tmp_path / "practice.csv")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **kw: QtWidgets.QMessageBox.Cancel)
    before = window._analysis_panel.detection_params()
    window._create_practice_recording()
    assert not calls and list(tmp_path.iterdir()) == []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **kw: QtWidgets.QMessageBox.Ok)
    window._create_practice_recording()
    assert calls == [[tmp_path / "SYNTHETIC-practice.csv"]]
    assert window._analysis_panel.detection_params() == before


def test_practice_dialog_confirms_actual_normalized_output(window, tmp_path, monkeypatch):
    from nanoscope.gui import export_paths
    actual = tmp_path / "SYNTHETIC-example.csv"
    actual.write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **kw: QtWidgets.QMessageBox.Ok)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(tmp_path / "example"), ""))
    seen = []
    monkeypatch.setattr(export_paths, "confirm_overwrite", lambda parent, title, paths: seen.extend(paths) or False)
    window._create_practice_recording()
    assert seen == [actual]
    assert actual.read_text(encoding="utf-8") == "do not overwrite"


def test_save_project_uses_existing_requester_workflow(window, monkeypatch):
    calls = []
    window._open_first_analysis_guide()
    monkeypatch.setattr(window, "_save_project_from", lambda requester, **kw: calls.append((requester, kw)))
    window._first_analysis_action("save_project")
    assert calls == [(window._first_analysis_guide, {"save_as": False})]


def test_practice_refuses_replacing_open_recording(window, tmp_path, monkeypatch):
    from nanoscope.gui import main_window
    target = tmp_path / "SYNTHETIC-practice.csv"
    add_recording(window, tmp_path, target.name)
    target.write_text("existing recording", encoding="utf-8")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **kw: QtWidgets.QMessageBox.Ok)
    monkeypatch.setattr(main_window, "ask_export_path", lambda *a, **kw: target)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: warnings.append(a[1]))
    window._create_practice_recording()
    assert warnings == ["Recording is already open"]
    assert target.read_text(encoding="utf-8") == "existing recording"
