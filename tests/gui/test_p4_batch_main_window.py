"""Real selected exports preserve dormant figures, frozen snapshots, and navigation."""
import hashlib

import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.gui.analysis_catalog import PAGE_HOME
from .test_comparison_figures_integration import window  # noqa: F401
from .test_p2_workspace_controller import rig  # noqa: F401


def test_export_readiness_rejects_busy_and_preserves_dormant_state(rig):
    owner = rig.owner
    key = owner.create("iv")
    assert owner.export_ready(key)[0]
    owner.page(key).ready = False
    assert not owner.export_ready(key)[0]
    owner.page(key).ready = True
    token = owner.request_token(key)
    assert not owner.export_ready(key)[0]
    owner.finish_request(token)
    assert owner.suspend(key)
    assert owner.export_ready(key) == (True, "")
    assert owner.page(key) is None
    owner.clear_current()
    assert owner.current_id() is None and owner.get(key) is not None
    assert not owner.export_ready("missing")[0]


def populated(window):
    owner = window._ensure_figure_workspace()
    key = owner.create("comparison_figures")
    page = owner.page(key)
    page.set_config({"kind": "facets"})
    window._refresh_figure(key)
    assert page.has_data() and owner.get(key).spec.result.status != "empty"
    owner.rename(key, "Apo comparison")
    duplicate = owner.duplicate(key)
    assert duplicate and owner.suspend(duplicate)
    snapshot = window._save_analysis_figure(key, name="Apo frozen snapshot")
    assert snapshot is not None
    window._figure_library.hide()
    return owner, key, duplicate, snapshot


def test_mixed_selection_exports_exact_frozen_data_and_restores_home(window, monkeypatch, tmp_path):
    from nanoscope.plot import comparison_figures
    owner, key, dormant, snapshot = populated(window)
    page = owner.page(key)
    before = {name: table.copy(deep=True) for name, table in page._tables.items()}
    def forbid(*args, **kwargs):
        pytest.fail("Batch export recalculated a comparison")
    monkeypatch.setattr(comparison_figures, "prepare_comparison", forbid)
    explorer = window._analysis_explorer
    explorer.show_home(); explorer.refresh_figures()
    listing = explorer.figures_list()
    listing.restore_selection((key, dormant, snapshot.artifact_id), current_id=snapshot.artifact_id)
    previous_current = owner.current_id()
    results = window._export_selected_figures(directory=tmp_path, formats=("svg", "png"),
        sizing="single", selected_ids=(key, dormant), saved_ids=(snapshot.artifact_id,), interactive=False)
    assert len(results) == 6 and all(r.status == "written" for r in results), results
    assert all(r.path.is_file() for r in results)
    assert owner.page(dormant) is None and owner.current_id() == previous_current
    assert explorer.current_page_id() in (PAGE_HOME, None)
    assert set(listing.selected_ids()) == {key, dormant}
    assert listing.selected_ids("saved") == (snapshot.artifact_id,)
    assert listing.current_id("snapshot") == snapshot.artifact_id
    for name, table in before.items():
        pd.testing.assert_frame_equal(page._tables[name], table, check_exact=True)
    old_hashes = {r.path: hashlib.sha256(r.path.read_bytes()).hexdigest() for r in results}
    second = window._export_selected_figures(directory=tmp_path, formats=("svg", "png"),
        sizing="single", selected_ids=(key, dormant), saved_ids=(snapshot.artifact_id,), interactive=False)
    assert len(second) == 6 and all(r.status == "failed" for r in second)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in old_hashes.items())


def test_changed_project_during_options_writes_nothing(window, monkeypatch, tmp_path):
    from nanoscope.gui import figure_batch_export_dialog
    owner = window._ensure_figure_workspace()
    key = owner.create("comparison_figures")
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    def choose(*args):
        window._figure_project_epoch += 1
        return tmp_path, ("svg",), "native", 300
    monkeypatch.setattr(figure_batch_export_dialog, "choose_batch_export_options", choose)
    assert window._export_selected_figures(selected_ids=(key,), saved_ids=()) == ()
    assert warnings and "changed" in warnings[0]
    assert list(tmp_path.iterdir()) == []


def test_deleted_snapshot_during_options_is_refused(window, monkeypatch, tmp_path):
    from nanoscope.gui import figure_batch_export_dialog
    owner, key, dormant, snapshot = populated(window)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    def choose(*args):
        window._analytical_workspace.artifacts.remove(snapshot)
        return tmp_path, ("svg",), "native", 300
    monkeypatch.setattr(figure_batch_export_dialog, "choose_batch_export_options", choose)
    assert window._export_selected_figures(selected_ids=(), saved_ids=(snapshot.artifact_id,)) == ()
    assert warnings and "changed" in warnings[0]
    assert list(tmp_path.iterdir()) == []


def test_export_restores_a_dormant_current_identity_without_hydrating_it(window, tmp_path):
    owner, key, dormant, snapshot = populated(window)
    assert owner.suspend(key)
    assert owner.restore_navigation(key)
    window._analysis_explorer.show_home()
    made = []
    factory = owner._factory
    def counted(kind, identity):
        made.append(identity)
        return factory(kind, identity)
    owner._factory = counted
    results = window._export_selected_figures(directory=tmp_path, formats=('svg',),
        selected_ids=(dormant,), saved_ids=(), interactive=False)
    assert len(results) == 1 and results[0].status == 'written'
    assert made == [dormant]
    assert owner.current_id() == key
    assert owner.page(key) is None and owner.page(dormant) is None
