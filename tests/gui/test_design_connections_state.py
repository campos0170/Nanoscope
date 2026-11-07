"""Cross-route state contracts discovered by the full-design audit."""
from dataclasses import replace
from types import SimpleNamespace

from nanoscope.gui.figure_list import FigureList
from nanoscope.gui.plot_workflow import PendingState
from .test_p2_workspace_controller import rig
from .p2_shell_rig import record


def test_pending_state_matches_stored_record_and_undo_preserves_source_staleness(rig):
    owner = rig.owner
    key = owner.create('comparison_figures')
    page = owner.page(key)
    count = [0]
    page.inspector_adapter = lambda: SimpleNamespace(pending=lambda: PendingState(count[0]))
    owner.mark_current(key)
    count[0] = 1
    page.workspaceChanged.emit()
    assert owner.get(key).spec.result.status == 'stale'
    owner.checkpoint(key)
    count[0] = 0
    page.workspaceChanged.emit()
    assert owner.get(key).spec.result.status == 'current'
    count[0] = 1
    page.workspaceChanged.emit()
    owner.sources_changed()
    count[0] = 0
    page.workspaceChanged.emit()
    assert owner.get(key).spec.result.status == 'stale'


def test_empty_workspace_roundtrip_stays_empty_and_export_rejects_it(rig, tmp_path):
    from nanoscope.gui.figure_batch_export import build_export_plan
    owner = rig.owner
    key = owner.create('iv')
    owner.sources_changed()
    assert owner.get(key).spec.result.status == 'empty'
    owner.install_workspace(owner.capture_workspace())
    assert owner.get(key).spec.result.status == 'empty'
    plan = build_export_plan(owner, (key,), tmp_path, formats=('svg',))
    assert 'no committed result' in plan.targets[0].unavailable


def test_pending_reload_retains_stale_and_saved_base(rig):
    owner = rig.owner
    key = owner.create('comparison_figures')
    page = owner.page(key)
    page.inspector_adapter = lambda: SimpleNamespace(pending=lambda: PendingState(1))
    owner.mark_current(key)
    page.workspaceChanged.emit()
    owner.install_workspace(owner.capture_workspace())
    assert owner.get(key).spec.result.status == 'stale'
    assert owner.get(key).spec.extra['figure_workspace']['pending_result_base'] == 'frozen'


def test_saved_actions_use_selected_id_and_new_activates_new_live_row(qtbot):
    listing = FigureList()
    qtbot.addWidget(listing)
    first, second = record('A'), record('B')
    saved = SimpleNamespace(artifact_id='figure.saved',payload={'name':'Saved A','page_id':'iv'})
    listing.set_records((first,), (saved,), current=first.spec.figure_id)
    listing.tree.setCurrentItem(listing._items[saved.artifact_id])
    assert not listing.snapshot_button.isEnabled()
    calls = []
    listing.duplicateRequested.connect(calls.append)
    listing.duplicate_current()
    assert calls == [saved.artifact_id]
    assert listing.rename_button.isEnabled() and listing.delete_button.isEnabled()
    listing.set_records((first,second),(saved,),selected=(second.spec.figure_id,),current=second.spec.figure_id)
    assert listing.current_id() == second.spec.figure_id
    assert not listing.selected_ids('saved')
    assert listing.snapshot_button.isEnabled() and listing.rename_button.isEnabled()


def test_noise_source_choice_survives_update_and_header_ignores_global_selection(qtbot, tmp_path):
    from nanoscope.gui.main_window import MainWindow
    from .test_psd import _current_doc
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    a = window._session.add_document(_current_doc(tmp_path,name='A.dat',n=8000,sigma=1))
    b = window._session.add_document(_current_doc(tmp_path,name='B.dat',n=8000,sigma=4))
    window._current_selection = (a,0)
    owner = window._ensure_figure_workspace()
    key = owner.create('noise')
    page = owner.page(key)
    window._refresh_figure(key)
    assert page.curves()[0].doc_id == a
    before = page.psd_result()[1]
    page._rec_combo.setCurrentIndex(next(i for i in range(page._rec_combo.count())
                                         if page._rec_combo.itemData(i)==(b,0)))
    assert owner.binding(key)['source']['source_key'].endswith('B.dat')
    assert page.curves()[0].doc_id == a  # pending controls preserve the visible result
    assert page.inspector_adapter().pending().count > 0
    page.inspector_adapter().update()
    assert page._current_target() == (b,0) and page.curves()[0].doc_id == b
    assert not page.inspector_adapter().pending().count
    assert not (page.psd_result()[1] == before).all()
    window._current_selection = (a,0)
    assert 'B.dat' in window._analysis_explorer_context()
    assert 'A.dat' not in window._analysis_explorer_context()
    assert owner.get(key).spec.result.status == 'current'


def test_new_distributions_header_does_not_claim_uncomputed_result(qtbot):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner=window._ensure_figure_workspace()
    key=owner.create('distributions')
    assert window._analysis_explorer_status(key,owner.page(key)) == ('NO DATA','empty')


def test_following_event_header_uses_displayed_source_not_original_pin():
    from nanoscope.gui.main_window import MainWindow
    page=SimpleNamespace(_doc_id='B',_channel=0)
    owner=SimpleNamespace(current_id=lambda:'event',get=lambda key:SimpleNamespace(spec=SimpleNamespace(kind='event_waveforms')),
        page=lambda key:page,binding=lambda key:{'follow':True,'source':{'source_key':'A'}})
    session=SimpleNamespace(documents=lambda:(),has_document=lambda key:key=='B',
                            document=lambda key:SimpleNamespace(display_name='Recording B'))
    window=SimpleNamespace(_figure_workspace=owner,_session=session)
    assert MainWindow._analysis_explorer_context(window)=='Recording B · channel 0'


def test_reopened_event_header_uses_applied_source_recipe(monkeypatch):
    from nanoscope.gui.main_window import MainWindow
    from nanoscope.core import analysis_workspace
    monkeypatch.setattr(analysis_workspace, '_resolve_source', lambda win, source: (source['source_key'],0,''))
    page=SimpleNamespace(_doc_id=None,_nanoscope_workspace_event_source={'source_key':'A'})
    owner=SimpleNamespace(current_id=lambda:'event',get=lambda key:SimpleNamespace(spec=SimpleNamespace(kind='event_waveforms')),
        page=lambda key:page,binding=lambda key:{'follow':True,'source':{'source_key':'B'}})
    session=SimpleNamespace(documents=lambda:(),has_document=lambda key:False,
                            document=lambda key:SimpleNamespace(display_name='Recording '+key))
    window=SimpleNamespace(_figure_workspace=owner,_session=session)
    assert MainWindow._analysis_explorer_context(window)=='Recording A · channel 0'
