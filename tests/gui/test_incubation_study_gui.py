"""Study setup, saved review and owned-worker boundaries without model inference."""
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pandas as pd
import pytest

from nanoscope.analysis import incubation_study as study
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.gui.incubation_window import IncubationStudyWindow
from nanoscope.gui.worker_thread import drain_detached


def example_spec(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text("time_s,current_detr_pA,current_raw_pA\n0,0,-100\n")
    metadata = tmp_path / "trace_metadata.json"
    metadata.write_text('{"fs_hz":50000}')
    model = tmp_path / "student.keras"
    model.write_bytes(b"test model is never inferred")
    recording = study.StudyRecording(tag="sample_30h", trace_path=trace,
                                    metadata_path=metadata, incubation_h=30,
                                    pore_id="pore-A", pore_diameter_nm=20,
                                    concentration_pm=100, voltage_mv=-200)
    return study.StudySpec(name="Synthetic study", recordings=(recording,), model_path=model,
                           mixture_tag="sample_30h", recipe_version="zn_student_incubation_v1", seed=0)


def result(tmp_path):
    return SimpleNamespace(output_dir=tmp_path, summary=pd.DataFrame({"tag": ["sample_30h"], "count": [2]}),
                           events=pd.DataFrame({"tag": ["sample_30h"] * 2, "start_idx": [10, 20]}),
                           metadata={"study_name": "Synthetic study", "source": "trace.csv"}, figure_paths=[])


@pytest.fixture
def window(qtbot):
    w = IncubationStudyWindow()
    qtbot.addWidget(w)
    yield w
    w.abort()
    drain_detached(2000)


def configure(window, tmp_path):
    spec = example_spec(tmp_path)
    window.set_spec(spec)
    window._output_edit.setText(str(tmp_path / "new-result"))
    return spec


@pytest.mark.parametrize("column,value,fragment", [
    (1, "nan", "incubation"), (1, "-1", "incubation"),
    (2, "", "pore ID"), (3, "0", "pore diameter"),
    (4, "inf", "concentration"), (5, "0", "voltage"),
    (4, "1_20", "concentration"), (4, "١٢٠", "concentration"),
])
def test_invalid_metadata_never_starts_a_job(window, tmp_path, monkeypatch, column, value, fragment):
    configure(window, tmp_path)
    window._recordings.item(0, column).setText(value)
    monkeypatch.setattr(study, "run_study", lambda *a, **kw: pytest.fail("Invalid input started analysis"))
    assert not window.start_study()
    assert not window.is_busy()
    assert fragment.lower() in window._status.text().lower()


def test_duplicate_tags_are_refused_and_saved_output_is_preserved(window, tmp_path):
    spec = configure(window, tmp_path)
    window._set_recordings([spec.recordings[0], spec.recordings[0]])
    assert not window.start_study()
    assert "unique" in window._status.text()
    window.set_spec(spec)
    output = tmp_path / "new-result"
    output.mkdir()
    evidence = output / "report.md"
    evidence.write_text("existing study")
    assert not window.start_study()
    assert "preserve" in window._status.text()
    assert evidence.read_text() == "existing study"


def test_recipe_roundtrip_captures_edits_with_resolved_sources(window, tmp_path):
    configure(window, tmp_path)
    window._name_edit.setText("Edited study")
    window._recordings.item(0, 1).setText("18")
    recipe = tmp_path / "recipe.json"
    window.save_recipe(recipe)
    loaded = study.load_spec(recipe)
    assert loaded.name == "Edited study"
    assert loaded.recordings[0].incubation_h == 18
    assert Path(loaded.recordings[0].trace_path).is_absolute()
    window.load_recipe(recipe)
    assert window.capture_spec().mixture_tag == "sample_30h"


def test_saved_result_load_displays_saved_values_without_inference(window, tmp_path, monkeypatch):
    saved = result(tmp_path)
    saved.events = pd.DataFrame({"source_tag": ["sample_30h"] * 50000, "start_idx": range(50000)})
    (tmp_path / "report.md").write_text("# Saved study\nReviewed source: trace.csv\n")
    monkeypatch.setattr(study, "load_study_result", lambda path: saved)
    monkeypatch.setattr(study, "run_study", lambda *a, **kw: pytest.fail("Opening a saved bundle reran inference"))
    assert window.load_result(tmp_path) is saved
    assert window._summary_model.rowCount() == 1
    assert window._events_model.rowCount() == 50000
    assert "trace.csv" in window._report.toPlainText()
    assert window._report.document().baseUrl().toLocalFile().rstrip("/") == str(tmp_path)
    assert str(tmp_path) in window._identity.text()
    assert "no analysis was rerun" in window._status.text()
    assert window._events_model.data(window._events_model.index(49999, 1)) == "49999"


def test_running_spec_is_captured_and_cancel_waits_for_acknowledgement(window, tmp_path, monkeypatch, qtbot):
    configure(window, tmp_path)
    entered = threading.Event()
    acknowledge = threading.Event()
    observed = []

    def run(spec, *, output_dir, progress, cancel):
        observed.append(spec)
        progress(.25, "Reading synthetic recording")
        entered.set()
        while not acknowledge.wait(.005):
            pass
        assert cancel()
        raise OperationCancelled("requested")

    monkeypatch.setattr(study, "run_study", run)
    try:
        assert window.start_study()
        qtbot.waitUntil(entered.is_set, timeout=2000)
        window._name_edit.setText("Later widget edit")
        assert observed[0].name == "Synthetic study"
        assert not window._setup.isEnabled()
        window.cancel()
        assert window.is_busy()
        assert "Cancellation requested" in window._status.text()
        assert not window._run_btn.isEnabled()
        with qtbot.waitSignal(window.studyCancelled, timeout=3000):
            acknowledge.set()
        qtbot.waitUntil(lambda: not window.is_busy(), timeout=2000)
        assert window._result is None
        assert "cancelled" in window._status.text()
    finally:
        acknowledge.set()


def test_close_parks_unfinished_worker_and_discards_its_late_result(window, tmp_path, monkeypatch, qtbot):
    configure(window, tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def run(spec, *, output_dir, progress, cancel):
        entered.set()
        release.wait(5)
        return result(tmp_path)

    monkeypatch.setattr(study, "run_study", run)
    thread = None
    try:
        assert window.start_study()
        qtbot.waitUntil(entered.is_set, timeout=2000)
        thread = window._thread
        started = time.monotonic()
        window.close()
        assert time.monotonic() - started < .7
        assert thread.isRunning()
        assert window.is_busy()
        assert window._worker is None
        assert not window.start_study()
        release.set()
        qtbot.waitUntil(lambda: not window.is_busy(), timeout=3000)
        assert window._result is None
        assert not window._thread
    finally:
        release.set()
        if thread is not None:
            thread.quit()
            assert thread.wait(3000)
        drain_detached(2000)


def test_abort_stops_the_lazily_opened_manual_workspaces_worker(window, tmp_path, qtbot):
    window._open_manual_workflow()
    workspace = window._manual_workspace
    qtbot.addWidget(workspace)
    workspace.new_workspace(tmp_path / "manual-workspace")
    entered = threading.Event()

    def compute(progress, cancel):
        entered.set()
        while not cancel():
            time.sleep(.005)
        raise OperationCancelled("stopped by owner abort")

    assert workspace._stage("Slow operation", compute, lambda state, result: None)
    qtbot.waitUntil(entered.is_set, timeout=2000)
    thread = workspace._thread
    window.abort()
    assert not thread.isRunning(), (
        "IncubationStudyWindow.abort() must also abort() the manual workspace it "
        "lazily opened, or that workspace's worker thread outlives the window")
    drain_detached(2000)


def test_worker_failure_is_visible_and_allows_correction(window, tmp_path, monkeypatch, qtbot):
    configure(window, tmp_path)

    def fail(*args, **kwargs):
        raise ValueError("Missing current_raw_pA in trace.csv")

    monkeypatch.setattr(study, "run_study", fail)
    with qtbot.waitSignal(window.studyFailed, timeout=3000):
        assert window.start_study()
    qtbot.waitUntil(lambda: not window.is_busy(), timeout=2000)
    assert "Missing current_raw_pA" in window._status.text()
    assert window._setup.isEnabled()
    assert window._result is None


def test_success_installs_the_completed_bundle_and_releases_worker(window, tmp_path, monkeypatch, qtbot):
    configure(window, tmp_path)
    saved = result(tmp_path)
    monkeypatch.setattr(study, "run_study", lambda *a, **kw: saved)
    with qtbot.waitSignal(window.studyFinished, timeout=3000):
        assert window.start_study()
    qtbot.waitUntil(lambda: window._run_btn.isEnabled(), timeout=2000)
    assert window._worker is None
    assert window._result is saved
    assert window._progress.value() == 100
    assert window._tabs.currentIndex() == 1


def test_menu_opens_one_independent_study_and_preserves_session(qtbot):
    from nanoscope.gui.main_window import MainWindow
    main = MainWindow()
    qtbot.addWidget(main)
    before = main._session
    assert main._act_incubation_study.isEnabled()
    assert main._act_incubation_study in main._analysis_menu.actions()
    main._act_incubation_study.trigger()
    first = main._incubation_study_window
    from nanoscope.gui.incubation_launcher import IncubationStudyLauncher
    assert isinstance(first, IncubationStudyLauncher)
    main._act_incubation_study.trigger()
    assert main._incubation_study_window is first
    assert main._session is before
    assert not main._session.documents()
    assert first in main._owned_background_holders()
    assert "close" in main._stop_protocols(first)
    first.close()
