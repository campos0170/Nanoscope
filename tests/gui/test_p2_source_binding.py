"""Selection changes affect only explicit followers and never hydrate hidden figures."""
from dataclasses import replace

from .test_p2_workspace_controller import rig


def test_source_selection_stages_only_opted_in_figures_without_hydrating_dormant_ones(rig):
    owner = rig.owner
    first = owner.create("event_waveforms")
    second = owner.create("noise")
    owner.sources_changed(selection_only=True)
    assert not rig.bindings
    owner.set_follow_selection(second, True)
    owner.sources_changed(selection_only=True)
    assert rig.bindings == [(second, owner.binding(second), True)] * 2
    owner.activate(first)
    rig.bindings.clear()
    owner.sources_changed(selection_only=True)
    assert rig.bindings == [(second, owner.binding(second), True)]
    assert len(rig.built) == 2
    assert owner.suspend(second)
    rig.bindings.clear()
    owner.sources_changed(selection_only=True)
    assert not rig.bindings and len(rig.built) == 2


def test_project_loaded_followers_remain_frozen_until_explicit_update(rig):
    owner = rig.owner
    key = owner.create("noise")
    owner.set_follow_selection(key, True)
    owner.mark_current(key)  # This fixture represents a committed saved result.
    saved = owner.capture_workspace()
    owner.install_workspace(saved)
    assert not owner.resident_ids()
    owner.activate(key)
    rig.bindings.clear()
    owner.sources_changed(selection_only=True)
    # The callback stages the source in managed Noise/Event pages; it no longer
    # runs kernels. The real-page regression verifies the arrays stay frozen.
    assert rig.bindings == [(key, owner.binding(key), True)]
    assert owner.get(key).spec.result.status == "frozen"
