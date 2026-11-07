"""Serial batch exports freeze identities, preserve old files and release dormant pages."""
from types import SimpleNamespace

import pandas as pd
import pyqtgraph as pg
import pytest

from nanoscope.gui.figure_bundle import capture_figure_snapshot
from nanoscope.gui import figure_batch_export as B


class Owner:
    def __init__(self):
        self.current, self.selected = 'a', ('a', 'c')
        self.pages = {'a': object(), 'b': object()}
        self.generation = {'a': 0, 'b': 0, 'c': 0}
        self.busy, self.captures, self.released = set(), [], []
        self.records = {key: SimpleNamespace(spec=SimpleNamespace(name='Same / name',
            result=SimpleNamespace(status='frozen'),
            presentation=SimpleNamespace(style_snapshot={}))) for key in self.generation}
    def get(self, key): return self.records.get(key)
    def revision_token(self, key): return (1, key, self.generation[key]) if key in self.records else None
    def checkpoint(self, key): return True
    def current_id(self): return self.current
    def selected_ids(self): return self.selected
    def set_selected(self, ids): self.selected = tuple(ids)
    def page(self, key): return self.pages.get(key)
    def export_ready(self, key): return (key not in self.busy, 'Worker running' if key in self.busy else '')
    def activate(self, key):
        self.current = key; self.pages.setdefault(key, object()); self.selected = (key,)
        return True
    def suspend(self, key):
        self.pages.pop(key, None); self.released.append(key)
        return True


@pytest.fixture
def snapshot(qtbot):
    plot = pg.PlotWidget(); qtbot.addWidget(plot)
    plot.plot([1., 2.], [3., 4.])
    return capture_figure_snapshot(plot.getPlotItem(), data=pd.DataFrame({'raw_s': [.001, .002]}))


def test_two_resident_one_dormant_and_saved_snapshot_export_exact_bytes_serially(snapshot, tmp_path):
    owner = Owner()
    plan = B.build_export_plan(owner, ('a', 'b', 'c'), tmp_path, formats=('svg',),
                               snapshots={'saved': ('Same / name', snapshot)})
    assert len({path.name.casefold() for path in plan.paths}) == 4
    def capture(key):
        assert owner.page(key) is not None
        owner.captures.append(key)
        return snapshot
    results = B.run_export_plan(plan, owner=owner, capture=capture, is_current=lambda: True)
    assert [result.status for result in results] == ['written'] * 4
    assert owner.captures == ['a', 'b', 'c'] and owner.released == ['c']
    assert set(owner.pages) == {'a', 'b'} and owner.current == 'a' and owner.selected == ('a', 'c')
    assert all(result.path.read_bytes() == snapshot.svg_bytes for result in results)


def test_cancel_is_latched_between_figures_and_preserves_completed_results(snapshot, tmp_path):
    owner = Owner(); checks = iter((False, True, False))
    plan = B.build_export_plan(owner, ('a', 'b', 'c'), tmp_path, formats=('svg',))
    results = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot,
                               is_current=lambda: True, cancelled=lambda: next(checks))
    assert [result.status for result in results] == ['written', 'cancelled', 'cancelled']
    assert results[0].path.is_file() and not any(result.path.exists() for result in results[1:])


def test_busy_or_changed_figure_never_materializes_and_other_files_continue(snapshot, tmp_path):
    owner = Owner(); owner.busy.add('a')
    plan = B.build_export_plan(owner, ('a', 'b', 'c'), tmp_path, formats=('svg',))
    owner.generation['b'] += 1
    captured = []
    results = B.run_export_plan(plan, owner=owner,
        capture=lambda key: (captured.append(key), snapshot)[1], is_current=lambda: True)
    assert [r.status for r in results] == ['failed', 'failed', 'written']
    assert 'Worker running' in results[0].error and 'changed after' in results[1].error
    assert captured == ['c'] and owner.released == ['c']


def test_writer_failure_and_destination_change_keep_prior_files(snapshot, tmp_path, monkeypatch):
    owner = Owner()
    initial = B.build_export_plan(owner, ('a',), tmp_path, formats=('svg',))
    path = initial.paths[0]; path.write_bytes(b'previous')
    plan = B.build_export_plan(owner, ('a',), tmp_path, formats=('svg',))
    def broken(stage, *_args):
        stage.write_bytes(b'partial')
        raise OSError('injected storage failure')
    monkeypatch.setattr(B, 'write_frozen_figure_image', broken)
    result, = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot,
                               is_current=lambda: True, overwrite=True)
    assert result.status == 'failed' and path.read_bytes() == b'previous'
    path.write_bytes(b'new external file')
    result, = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot,
                               is_current=lambda: True, overwrite=True)
    assert result.status == 'failed' and 'destination changed' in result.error.lower()
    assert path.read_bytes() == b'new external file'


def test_project_switch_during_write_abandons_stage(snapshot, tmp_path, monkeypatch):
    owner = Owner(); valid = [True]
    plan = B.build_export_plan(owner, ('a',), tmp_path, formats=('svg',))
    write = B.write_frozen_figure_image
    def switched(*args):
        value = write(*args); valid[0] = False
        return value
    monkeypatch.setattr(B, 'write_frozen_figure_image', switched)
    result, = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot,
                               is_current=lambda: valid[0])
    assert result.status == 'failed' and not result.path.exists()
    assert 'project changed' in result.error.lower()


def test_missing_style_width_is_explicit_and_snapshot_bytes_remain_immutable(snapshot, tmp_path):
    owner = Owner()
    plan = B.build_export_plan(owner, (), tmp_path, sizing='single', formats=('svg',),
                               snapshots={'saved': ('Captured', snapshot)})
    before = snapshot.to_storage()
    result, = B.run_export_plan(plan, owner=owner,
        capture=lambda key: pytest.fail('Saved snapshot was materialized'), is_current=lambda: True)
    assert result.status == 'failed' and 'no single-column width' in result.error
    assert snapshot.to_storage() == before and not result.path.exists()


def test_deleted_snapshot_refuses_remaining_formats_without_reloading_its_artwork(snapshot, tmp_path):
    owner = Owner(); present = [True]
    plan = B.build_export_plan(owner, (), tmp_path, formats=('svg', 'pdf'),
                               snapshots={'saved': ('Frozen', snapshot)})
    def progress(result, done, total):
        if result.status == 'written': present[0] = False
    results = B.run_export_plan(plan, owner=owner,
        capture=lambda key: pytest.fail('Saved artwork was reloaded'), is_current=lambda: True,
        saved_is_current=lambda key: present[0], progress=progress)
    assert [result.status for result in results] == ['written', 'failed']
    assert results[0].path.read_bytes() == snapshot.svg_bytes and not results[1].path.exists()
    assert 'saved figure was deleted' in results[1].error


def test_progress_failure_does_not_relabel_an_already_written_file(snapshot, tmp_path):
    owner = Owner()
    plan = B.build_export_plan(owner, ('a',), tmp_path, formats=('svg',))
    def broken(*args): raise RuntimeError('status view closed')
    result, = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot,
                               is_current=lambda: True, progress=broken)
    assert result.status == 'written' and result.path.read_bytes() == snapshot.svg_bytes
    assert 'Progress could not be displayed' in result.error


def test_vector_marker_budget_refuses_before_capture_and_can_be_explicitly_reviewed(qtbot, snapshot, tmp_path):
    plot = pg.PlotWidget(); qtbot.addWidget(plot)
    curve = plot.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], symbol='o')
    owner = Owner()
    owner.pages['a'] = SimpleNamespace(figure_export_target=lambda: plot.getPlotItem())
    plan = B.build_export_plan(owner, ('a',), tmp_path, formats=('svg',))
    result, = B.run_export_plan(plan, owner=owner,
        capture=lambda key: pytest.fail('Oversized vector capture started'), is_current=lambda: True,
        vector_point_limit=5)
    assert result.status == 'failed' and '5 individual vector markers' in result.error
    assert 'individual Export' in result.error and not result.path.exists()
    result, = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot,
                               is_current=lambda: True, vector_point_limit=6)
    assert result.status == 'written' and len(curve.xData) == 5


def test_batch_style_canvas_uses_each_pinned_size_and_leaves_undefined_figures_native(snapshot, tmp_path, monkeypatch):
    from nanoscope.core.figure_style import FigureStyle
    owner = Owner()
    style = FigureStyle('project.canvas', 1, 'Canvas', {'canvas_width_mm': 85., 'canvas_height_mm': 60.})
    from nanoscope.core.figure_spec import FigureRecord, FigureSpec, FigurePresentation, ResultState
    owner.records['a'] = FigureRecord(FigureSpec(
        figure_id='00000000-0000-0000-0000-000000000001', kind='iv', name='Canvas',
        result=ResultState(status='frozen'), presentation=FigurePresentation(
            style_snapshot=style.to_dict(), style_ref=style.ref.to_dict(),
            overrides={'canvas_width_mm': 91.})))
    plan = B.build_export_plan(owner, ('a', 'b'), tmp_path, formats=('svg',), sizing='style')
    assert not plan.targets[0].unavailable
    assert (plan.targets[0].width_mm, plan.targets[0].height_mm) == (91., 60.)
    assert (plan.targets[1].width_mm, plan.targets[1].height_mm) == (None, None)
    writes = []
    def write(path, captured, options):
        writes.append(options)
        path.write_bytes(b'captured-placeholder')
    monkeypatch.setattr(B, 'write_frozen_figure_image', write)
    results = B.run_export_plan(plan, owner=owner, capture=lambda key: snapshot, is_current=lambda: True)
    assert all(result.status == 'written' for result in results)
    assert (writes[0].physical_width_mm, writes[0].physical_height_mm) == (91., 60.)
    assert writes[1].native
    assert owner.records['a'].spec.presentation.style_snapshot == style.to_dict()


def test_snapshot_spec_pinned_canvas_and_columns_survive_storage(snapshot, tmp_path):
    import json
    from dataclasses import replace
    from nanoscope.core.figure_style import FigureStyle
    from nanoscope.core.figure_spec import FigureSpec, FigurePresentation, ResultState
    from nanoscope.gui.figure_bundle import FigureSnapshot
    style = FigureStyle('project.snapshot-canvas', 2, 'Snapshot canvas',
        {'canvas_width_mm': 85., 'canvas_height_mm': 60.}, columns={'single_mm': 89.})
    spec = FigureSpec(figure_id='00000000-0000-0000-0000-000000000002', kind='iv',
        name='Snapshot', result=ResultState(status='frozen'),
        presentation=FigurePresentation(style_ref=style.ref.to_dict(),
            style_snapshot=style.to_dict(), overrides={'canvas_width_mm': 91.}))
    held = replace(snapshot, additional_resources=(*snapshot.additional_resources,
        ('figure-spec.json', json.dumps(spec.to_dict()).encode())))
    metadata, resources = held.to_storage()
    restored = FigureSnapshot.from_storage(metadata, resources)
    before = restored.to_storage()
    for sizing, dimensions in (('style', (91., 60.)), ('single', (89., None))):
        plan = B.build_export_plan(Owner(), (), tmp_path, sizing=sizing, formats=('svg',),
            snapshots={'snapshot': ('Snapshot', restored)})
        target, = plan.targets
        assert not target.unavailable
        assert (target.width_mm, target.height_mm) == dimensions
    assert restored.to_storage() == before


def test_composition_pinned_style_reaches_individual_and_batch_export(snapshot, tmp_path, qtbot):
    from xml.etree import ElementTree as ET
    from nanoscope.core.figure_style import FigureStyle, style_binding
    from nanoscope.gui.figure_bundle import FigureSnapshot
    from nanoscope.gui.figure_composition import compose_figures
    from nanoscope.gui.figure_export_style import snapshot_export_style
    from nanoscope.gui.plot_export_options import ExportOptionsDialog

    style = FigureStyle('project.composition-size', 2, 'Composition columns',
        {'canvas_width_mm': 92.5, 'canvas_height_mm': 65.},
        columns={'single_mm': 92.5, 'double_mm': 190.})
    original_panel = snapshot.to_storage()
    composition = compose_figures([snapshot, snapshot], ['A', 'B'], 2,
                                 style_binding=style_binding(style))
    restored = FigureSnapshot.from_storage(*composition.to_storage())
    before = restored.to_storage()
    captured_style = snapshot_export_style(restored)
    assert captured_style == style
    dialog = ExportOptionsDialog(native_size=(restored.width, restored.height),
                                 figure_style=captured_style)
    qtbot.addWidget(dialog)
    options = dialog.options()
    assert (options.physical_width_mm, options.physical_height_mm) == (92.5, 65.)
    assert dialog.preset_combo.findData(92.5) >= 0
    assert dialog.preset_combo.findData(190.) >= 0
    for sizing, dimensions in (('style', (92.5, 65.)), ('single', (92.5, None)),
                               ('double', (190., None))):
        plan = B.build_export_plan(Owner(), (), tmp_path, sizing=sizing, formats=('svg',),
            snapshots={'composition': (sizing, restored)})
        target, = plan.targets
        assert not target.unavailable
        assert (target.width_mm, target.height_mm) == dimensions
        result, = B.run_export_plan(plan, owner=Owner(),
            capture=lambda key: pytest.fail('A saved composition cannot recapture a live figure'),
            is_current=lambda: True)
        assert result.status == 'written'
        assert ET.fromstring(result.path.read_bytes()).get('width') == f'{dimensions[0]:g}mm'
    assert restored.to_storage() == before
    assert snapshot.to_storage() == original_panel
