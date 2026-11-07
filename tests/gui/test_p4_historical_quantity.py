"""Authentic legacy files migrate at use time without changing frozen measurements."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json

import nanoscope  # noqa: F401 bootstrap before native value classes
import pandas as pd
import pytest

from nanoscope.core.project import load_project
from nanoscope.core.figure_codecs import decode_legacy, encode_legacy
from nanoscope.core.figure_migration_v12 import migrated_figure_id
from nanoscope.core.figure_spec import FigureRecord, thaw_json
from nanoscope.gui.figure_adapter_pages import initial_history, apply_look_to_record, move_history
from nanoscope.gui.figure_adapter_plots import _native_config
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plots_panel import migrate_plot_quantity, plot_config_from_json, plot_config_to_json

FIXTURES = Path(__file__).parents[1] / 'core' / 'fixtures' / 'phase2'


def _record(native):
    decoded = decode_legacy(native, figure_id=migrated_figure_id(native.artifact_id), name='Historical figure')
    return FigureRecord(decoded.spec, decoded.resources)


def _same_resources(left, right):
    assert left.blobs == right.blobs
    assert set(left.tables) == set(right.tables)
    for key in left.tables:
        pd.testing.assert_frame_equal(left.tables[key], right.tables[key])


@pytest.mark.parametrize('dialect', ['d1', 'd2', 'v11'])
@pytest.mark.parametrize('variant', ['defaults', 'legacy-look'])
def test_authentic_legacy_plots_migrate_once_and_dormant_style_undo_preserves_raw_data(dialect, variant, monkeypatch):
    folder = FIXTURES / 'legacy'
    filename = f'{dialect}-{variant}.nanoscope'
    provenance = json.loads((folder / f'{dialect}-provenance.json').read_text())
    assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == provenance['files'][filename]['sha256']
    native = load_project(folder / filename).workspace.artifacts[0]
    payload = deepcopy(native.payload)
    record = _record(native)
    materialized = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    raw = _native_config(materialized, migrate=False)
    current = _native_config(materialized)
    assert current.axes.time_unit == 'ms'
    assert current.axes.quantity_display['version'] == 1
    assert current.axes.manual_x == raw.axes.manual_x
    assert current.axes.manual_y == raw.axes.manual_y
    assert (current.style.title, current.style.x_label, current.style.y_label) == (
        raw.style.title, raw.style.x_label, raw.style.y_label)
    assert migrate_plot_quantity(current) == current
    from nanoscope.gui.plots_panel import PlotsWindow
    monkeypatch.setattr(PlotsWindow, '__init__', lambda *_a, **_k: pytest.fail('hydrated a dormant figure'))
    history = initial_history(record)
    styled, after = apply_look_to_record(record, PlotAppearance(title_size=21.125), history)
    assert len(after.past) == 1
    undone, history_after_undo = move_history(styled, after)
    restored = encode_legacy(undone.spec, undone.resources, materialize_defaults=True)
    assert _native_config(restored) == current
    assert history_after_undo.base == history.base
    _same_resources(materialized, restored)
    assert native.payload == payload
    assert encode_legacy(record.spec, record.resources).payload == payload


@pytest.mark.parametrize('index', range(8))
def test_actual_all_pages_v12_dormant_style_undo_preserves_units_words_and_resources(index):
    path = FIXTURES / 'all_pages_v12.nanoscope'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads((FIXTURES / 'PROVENANCE.json').read_text())['sha256']
    native = load_project(path).workspace.artifacts[index]
    original = deepcopy(native.payload)
    record = _record(native)
    history = initial_history(record)
    styled, after = apply_look_to_record(record, PlotAppearance(title_size=21.125), history)
    undone, final = move_history(styled, after)
    assert final.base == history.base
    restored = encode_legacy(undone.spec, undone.resources)
    _same_resources(native, restored)
    assert native.payload == original
    assert encode_legacy(record.spec, record.resources).payload == original
    # Words and all explicit pins are included in this exact history value comparison.
    assert initial_history(undone).base == history.base
    state = thaw_json(history.base)
    if record.spec.kind == 'distributions':
        assert state['axes']['time_unit'] == 'ms'
        assert state['axes']['quantity_display']['version'] == 1
    elif record.spec.kind in ('distribution_summaries', 'event_waveforms', 'noise', 'voltage_series'):
        assert state['quantity_display']['version'] == 1


def test_legacy_plot_history_migrates_every_state_and_preserves_raw_pins_and_origins():
    from nanoscope.core.figure_history_state import HistoryCapsule
    from nanoscope.gui.figure_adapter_plots import _check_history
    from nanoscope.gui.plots_panel import PlotConfig, AxesConfig, HistogramConfig
    from nanoscope.gui.plots_panel import _HISTORY_DEPTH
    original = PlotConfig(histogram=HistogramConfig(column='dwell_s', normalization='density'),
        axes=AxesConfig(time_unit='s', x_min='.002', x_max='.008', y_min='0', y_max='200',
                        manual_x=(.002, .008), manual_y=(0., 200.)))
    past = replace(original, axes=replace(original.axes, x_min='.001', manual_x=(.001, .008)))
    future = replace(original, axes=replace(original.axes, x_min='.003', manual_x=(.003, .008)))
    capsule = HistoryCapsule('nanoscope.plot-config-history/1', plot_config_to_json(original),
        past=(plot_config_to_json(past),), future=(plot_config_to_json(future),), dropped=3,
        extra={'depth': _HISTORY_DEPTH, 'origin': 4, 'past_origins': [2], 'future_origins': [7],
               'opened': 8, 'norm_refusal_records': [], 'config_loss_records': []})
    current = migrate_plot_quantity(original)
    migrated = _check_history(capsule, current)
    assert migrated.extra == capsule.extra and migrated.dropped == 3
    for old, value in zip((original, past, future), (migrated.base, migrated.past[0], migrated.future[0])):
        new = plot_config_from_json(thaw_json(value))
        assert new.axes.manual_x == old.axes.manual_x and new.axes.manual_y == old.axes.manual_y
        assert float(new.axes.x_min) == float(old.axes.x_min) * 1000.
        assert float(new.axes.y_max) == .2
    assert _check_history(migrated, current) == migrated


@pytest.mark.parametrize('index,kind', [(1, 'distribution_summaries'), (2, 'comparison_figures')])
def test_dormant_column_only_style_revision_hydrates_and_undoes_as_one_step(qtbot, monkeypatch, index, kind):
    from nanoscope.core.figure_style import FigureStyle
    from nanoscope.core.figure_style_mapping import import_legacy_look
    from nanoscope.gui.figure_style_binding import bind_native_style
    from nanoscope.gui.figure_adapter_pages import _native_appearance, adapter_for
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
    native = load_project(FIXTURES / 'all_pages_v12.nanoscope').workspace.artifacts[index]
    record = _record(native)
    vocabulary = 'matplotlib_appearance' if kind == 'comparison_figures' else 'plot_appearance'
    appearance = _native_appearance(native, kind)
    style = FigureStyle('historical-column-style', 1, 'Apo paper',
        look=import_legacy_look(appearance.to_dict(), vocabulary).look,
        columns={'single_mm': 89., 'double_mm': 183.}, provenance={'inherit_native_defaults': True})
    first_look = PlotAppearance.from_mapping(bind_native_style({}, style, vocabulary))
    first, _ = apply_look_to_record(record, first_look, initial_history(record))
    revised = replace(style, revision=2, columns={'single_mm': 90., 'double_mm': 180.})
    second_look = PlotAppearance.from_mapping(bind_native_style({}, revised, vocabulary))
    second, history = apply_look_to_record(first, second_look, initial_history(first))
    assert len(history.past) == 1
    before, after = (PlotAppearance.from_mapping(thaw_json(state)) for state in (history.past[0], history.base))
    assert replace(before, style_binding=after.style_binding) == after
    cls = ComparisonFiguresWindow if kind == 'comparison_figures' else DistributionSummaryWindow
    monkeypatch.setattr(cls, 'analyze', lambda *_a, **_k: pytest.fail('analysis during hydrate'))
    page = cls(); qtbot.addWidget(page)
    adapter = adapter_for(page, kind)
    report = adapter.restore_record(second, mode='resume')
    assert not report.skipped, report
    adapter.install_transient(history)
    assert page._appearance_widget.state().style_binding['ref']['revision'] == 2
    assert page._appearance_widget.undo()
    assert page._appearance_widget.state().style_binding['ref']['revision'] == 1
    assert page._appearance_widget.redo()
    assert page._appearance_widget.state().style_binding['ref']['revision'] == 2
    from nanoscope.core.figure_tables import decode_figure_table
    for name, blob in native.payload['table_resources'].items():
        pd.testing.assert_frame_equal(page._tables[name], decode_figure_table(native.blobs[blob]))
