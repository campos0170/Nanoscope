"""Bounded residency, busy ownership and late-result guards."""
from .test_p2_workspace_controller import rig


def test_residency_is_bounded_and_switching_restores_without_analysis(rig):
    owner = rig.owner
    ids = [owner.create("iv") for _ in range(7)]
    assert len(owner.records()) == 7 and len(owner.resident_ids()) == 2
    assert owner.current_id() == ids[-1]
    before = owner.revision_token(ids[0])
    assert owner.activate(ids[0])
    assert owner.page(ids[0]).restores == ["resume"]
    assert owner.revision_token(ids[0]) == before
    assert len(owner.resident_ids()) == 2


def test_busy_cache_retains_new_logical_record_and_old_foreground(rig):
    owner = rig.owner
    first, second = owner.create("iv"), owner.create("iv")
    owner.page(first).ready = owner.page(second).ready = False
    third = owner.create("iv")
    assert owner.get(third) is not None and owner.page(third) is None
    assert owner.current_id() == second and len(rig.built) == 2
    owner.page(first).ready = True
    assert owner.activate(third) and len(owner.resident_ids()) == 2


def test_request_token_names_figure_and_epoch_and_busy_figure_cannot_be_deleted(rig):
    owner = rig.owner
    key = owner.create("iv")
    token = owner.request_token(key)
    owner.create("iv")
    assert owner.accepts_result(token) and not owner.delete(key)
    assert owner.finish_request(token)
    assert not owner.accepts_result(token)
    assert owner.delete(key)
    owner.restore_deleted(key)
    next_token = owner.request_token(key)
    owner.clear()
    assert not owner.accepts_result(token) and not owner.accepts_result(next_token)
