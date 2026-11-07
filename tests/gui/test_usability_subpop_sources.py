"""Report updates retain their source while main-window selection changes."""
from __future__ import annotations

from nanoscope.core.analysis_workspace import result_recipe
from nanoscope.gui.analysis_catalog import PAGE_IDS
from .test_s7_fixup_subpop_map_recovery_and_disclosure import (
    two, recordings, _registered, _report_for, SCHMITT,
)


def _stored_report(window, doc_id):
    result, report, _ = _report_for(window, doc_id)
    owner = window._ensure_figure_workspace()
    key = owner.create('subpopulations')
    window._install_subpop_figure_report(
        key, report, source_result=result, source_recipe=result_recipe(window, result),
        accepted_event_ids=tuple(result.payload.accepted.df.event_id))
    return owner, key, owner.page(key)


def _capture_launch(window, monkeypatch):
    jobs = []
    def run(job, **kwargs):
        jobs.append(job)
        return True
    monkeypatch.setattr(window._analysis_controller, 'run_callable', run)
    return jobs


def test_update_a_report_while_b_selected_uses_a(two, qtbot, monkeypatch):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    a_result = window._session.results.get(a, 0, SCHMITT)
    window._on_selection(b, 0)
    owner.activate(key)
    assert window._analysis_panel.current_detection().doc_id == b
    jobs = _capture_launch(window, monkeypatch)
    window._analysis_explorer.update_current()
    assert jobs
    assert window._subpop_target is a_result, (
        'Update plot replaced report A using the currently selected recording B')
    assert window._analysis_panel.current_detection().doc_id == b
    window._on_subpop_stopped()


import copy
from dataclasses import replace
import pytest
from nanoscope.core.analysis_workspace import capture_subpopulation, restore_subpopulation
from nanoscope.gui.main_window import MainWindow


@pytest.mark.parametrize('route', ['update', 'refresh'])
def test_update_sources_survive_a_b_a_navigation(two, qtbot, monkeypatch, route):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    window._on_selection(b, 0)
    window._on_selection(a, 0)
    window._on_selection(b, 0)
    owner.activate(key)
    jobs = _capture_launch(window, monkeypatch)
    if route == 'update':
        window._analysis_explorer.update_current()
    else:
        window._analysis_explorer.refresh_current()
    assert len(jobs) == 1
    assert window._subpop_target.doc_id == a
    assert window._analysis_panel.current_detection().doc_id == b
    assert 'a.csv' in page._source_label.text()
    assert 'b.csv' not in page._source_label.text()
    window._on_subpop_stopped()


def test_new_empty_figure_uses_current_detection(two, monkeypatch):
    window, a, b = two
    _stored_report(window, a)
    window._on_selection(b, 0)
    owner = window._figure_workspace
    key = owner.create('subpopulations')
    assert owner.page(key).report() is None
    jobs = _capture_launch(window, monkeypatch)
    window._analysis_explorer.update_current()
    assert jobs and window._subpop_target.doc_id == b
    window._on_subpop_stopped()


@pytest.mark.parametrize('change', ['missing', 'replacement', 'boundaries', 'depth', 'params', 'version', 'provenance', 'baseline', 'calibration', 'clock'])
def test_unavailable_changed_or_stale_source_refuses_without_using_b(two, monkeypatch, change):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    report = page.report()
    source = window._session.results.get(a, 0, SCHMITT)
    window._on_selection(b, 0)
    if change == 'missing':
        window._session.close_document(a)
    elif change in ('replacement', 'boundaries', 'depth', 'params', 'version', 'provenance', 'baseline'):
        frame = source.payload.df.copy()
        if change == 'boundaries':
            frame.loc[frame.index[0], 'start_idx'] += 1
        if change == 'depth':
            frame.loc[frame.index[0], 'depth_pA'] += 1
        replacement = replace(source, payload=replace(source.payload, df=frame))
        if change == 'params':
            replacement = replace(replacement, params={**source.params, 'k_in': 7.})
        elif change == 'version':
            replacement = replace(replacement, module_version='different')
        elif change == 'provenance':
            replacement = replace(replacement, payload=replace(replacement.payload, provenance={**source.payload.provenance, 'polarity': 1}))
        elif change == 'baseline':
            replacement = replace(replacement, baseline_fingerprint='different')
        window._session.add_result(replacement)
    elif change == 'calibration':
        calibration = window._session.calibration(a, 0)
        window._session.set_calibration(a, 0, replace(calibration, gain=2.0))
    else:
        from nanoscope.core.sample_rate import SampleRateDeclaration
        window._session.set_sample_rate_declaration(a, SampleRateDeclaration(
            mode='sample_rate', sample_rate_hz=50_000.))
    jobs = _capture_launch(window, monkeypatch)
    assert window._run_subpop_for_figure(key) is False
    assert not jobs and page.report() is report
    assert not owner._entries[key].busy
    assert window._analysis_panel.current_detection().doc_id == b


def test_curation_updates_original_source_and_uses_current_accepted_set(two, qtbot, monkeypatch):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    source = window._session.results.get(a, 0, SCHMITT)
    event_id = int(source.payload.df.iloc[0].event_id)
    window._analysis_panel.set_event_accepted(event_id, False)
    window._on_selection(b, 0)
    jobs = _capture_launch(window, monkeypatch)
    window._run_subpop_for_figure(key)
    assert jobs and window._subpop_target.doc_id == a, window._subpop_report_source(page)[1]
    assert event_id not in window._subpop_accepted_ids_pending
    assert len(window._subpop_accepted_ids_pending) == len(source.payload.df) - 1
    window._on_subpop_stopped()


def test_reopened_project_resolves_original_source_then_allows_curation(two, qtbot, tmp_path, monkeypatch):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    project = tmp_path / 'bound-report.nanoscope'
    assert window._write_project(str(project))
    reopened = MainWindow(); qtbot.addWidget(reopened)
    with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
        reopened._open_project(str(project))
    docs = {doc.meta.path.name: doc.doc_id for doc in reopened._session.documents()}
    a2, b2 = docs['a.csv'], docs['b.csv']
    reopened._on_selection(a2, 0)
    res = reopened._session.results.get(a2, 0, SCHMITT)
    event_id = int(res.payload.df.iloc[0].event_id)
    reopened._analysis_panel.set_event_accepted(event_id, False)
    reopened._on_selection(b2, 0)
    owner2 = reopened._ensure_figure_workspace()
    key2 = owner2.most_recent('subpopulations')
    assert owner2.activate(key2)
    page2 = owner2.page(key2)
    assert page2._nanoscope_workspace_source['subpopulation_detection_fingerprint']
    jobs = _capture_launch(reopened, monkeypatch)
    reopened._analysis_explorer.update_current()
    assert jobs and reopened._subpop_target.doc_id == a2, reopened._subpop_report_source(page2)[1]
    assert event_id not in reopened._subpop_accepted_ids_pending
    reopened._on_subpop_stopped()
    reopened._controller.abort(); reopened._analysis_controller.abort()


@pytest.mark.parametrize('terminal', ['cancel', 'delete', 'replacement'])
def test_late_result_after_cancel_delete_or_report_replacement_is_ignored(two, monkeypatch, terminal):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    original = page.report()
    _capture_launch(window, monkeypatch)
    window._run_subpop_for_figure(key)
    token = window._subpop_figure_request
    payload = copy.deepcopy(original)
    payload._nanoscope_figure_request = token
    if terminal == 'cancel':
        window._cancel_load()
        assert not owner._entries[key].busy
    elif terminal == 'delete':
        owner.delete(key)
    else:
        replacement = copy.deepcopy(original)
        page.set_report(replacement)
    window._on_subpop_finished(payload)
    assert not window._subpop_pending and window._subpop_figure_request is None
    if terminal == 'replacement':
        assert page.report() is replacement
    elif terminal == 'cancel':
        assert page.report() is original


def test_cancelled_report_cannot_consume_another_report_launch(two, monkeypatch):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    _capture_launch(window, monkeypatch)
    window._run_subpop_for_figure(key)
    payload = copy.deepcopy(page.report())
    payload._nanoscope_figure_request = window._subpop_figure_request
    window._cancel_load()
    window._on_selection(b, 0)
    other = owner.create('subpopulations')
    window._run_subpop_for_figure(other)
    new_token = window._subpop_figure_request
    window._on_subpop_finished(payload)
    assert window._subpop_figure_request == new_token
    assert window._subpop_target.doc_id == b
    assert owner.page(other).report() is None
    window._on_subpop_stopped()


@pytest.mark.parametrize('redo', [False, True])
def test_curation_undo_redo_remains_the_same_detection(two, qtbot, monkeypatch, redo):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    event_id = int(window._analysis_panel.current_detection().payload.df.iloc[0].event_id)
    window._analysis_panel.set_event_accepted(event_id, False)
    history_key = (a, 0, SCHMITT)
    assert window._history.undo(history_key)
    if redo:
        assert window._history.redo(history_key)
    window._on_selection(b, 0)
    jobs = _capture_launch(window, monkeypatch)
    window._run_subpop_for_figure(key)
    assert jobs, window._subpop_report_source(page)[1]
    assert window._subpop_target.doc_id == a
    assert (event_id in window._subpop_accepted_ids_pending) is (not redo)
    window._on_subpop_stopped()


@pytest.mark.parametrize('redo', [False, True])
def test_identical_replacement_after_curation_replay_is_not_the_replay(two, monkeypatch, redo):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    report = page.report()
    event_id = int(window._analysis_panel.current_detection().payload.df.iloc[0].event_id)
    window._analysis_panel.set_event_accepted(event_id, False)
    history_key = (a, 0, SCHMITT)
    assert window._history.undo(history_key)
    if redo:
        assert window._history.redo(history_key)
    replay = window._session.results.get(a, 0, SCHMITT)
    replacement = replace(replay, payload=replace(replay.payload, df=replay.payload.df.copy()))
    window._session.add_result(replacement)
    # Equal data pushes no history step, so an older Curate label remains present.
    assert any('Curate' in label for label in (
        window._history.undo_label(history_key), window._history.redo_label(history_key)))
    window._on_selection(b, 0)
    jobs = _capture_launch(window, monkeypatch)
    assert window._run_subpop_for_figure(key) is False
    assert not jobs and page.report() is report
    assert 'replaced' in window._subpop_report_source(page)[1]
    assert window._analysis_panel.current_detection().doc_id == b


def test_generate_menu_uses_current_detection_explicitly(two, monkeypatch):
    window, a, b = two
    _stored_report(window, a)
    window._on_selection(b, 0)
    jobs = _capture_launch(window, monkeypatch)
    window._act_subpop.trigger()
    assert jobs and window._subpop_target.doc_id == b
    window._on_subpop_stopped()


def test_reopened_project_refuses_same_path_with_changed_raw_contents(two, qtbot, tmp_path, monkeypatch):
    window, a, b = two
    owner, key, page = _stored_report(window, a)
    project = tmp_path / 'changed-raw-report.nanoscope'
    assert window._write_project(str(project))
    path = window._session.document(a).meta.path
    # Same header, sample count and clock; a replaced current trace must still be refused.
    import pandas as pd
    frame = pd.read_csv(path)
    frame.iloc[:1000, 1] += 50
    frame.to_csv(path, index=False)
    reopened = MainWindow(); qtbot.addWidget(reopened)
    with qtbot.waitSignal(reopened.projectRestored, timeout=30_000):
        reopened._open_project(str(project))
    owner2 = reopened._ensure_figure_workspace()
    key2 = owner2.most_recent('subpopulations'); owner2.activate(key2)
    page2 = owner2.page(key2); original = page2.report()
    jobs = _capture_launch(reopened, monkeypatch)
    assert reopened._run_subpop_for_figure(key2) is False
    assert not jobs and page2.report() is original
    assert 'changed' in reopened._subpop_report_source(page2)[1]
    reopened._controller.abort(); reopened._analysis_controller.abort()
