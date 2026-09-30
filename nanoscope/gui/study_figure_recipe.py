"""Rebuild study canvases from committed results for the native figure tools."""
from dataclasses import replace
from pathlib import Path
import tempfile

import pandas as pd

from ..analysis import manual_comparisons as compare
from ..analysis.incubation_study import StudyRecording
from ..plot import incubation_figures, signal_study_figures


def condition(record):
    return StudyRecording(record['tag'], record['source_path'], record['source_path'],
                          record['incubation_h'], record['pore_id'], record['pore_diameter_nm'],
                          record['concentration_pm'], record['voltage_mv'])


def infer_recipe(state, path):
    name=Path(path).stem
    saved=state.get('figure_recipes',{}).get(name)
    if saved is not None: return dict(saved)
    if name.startswith(('quantitative_appendix_', 'population_examples_', 'population_onsets_')):
        return dict(group=3, tag=state.get('mixture_tag'))
    if name.startswith('pooled_'): return dict(group=2)
    for record in sorted(state['records'],key=lambda r:len(r['tag']),reverse=True):
        tag=record['tag']
        if name.startswith('signal_'+tag+'_') or name in {
                prefix+tag for prefix in ('single_recording_candidates_', 'gallery50_',
                'gallery_sustained_', 'trace_context_', 'typical_events_')}:
            return dict(group=0,tag=tag)
    if name in {'aggregation_peak_global','common_floor_kde','gate_transfer_sweep',
                'incubation_rates_and_medians','incubation_sustained_local',
                'kde_sustained_global'} or name.startswith('signal_comparison_') or (
                name.startswith('size_matched_') and name.endswith('nm_sustained_local')):
        return dict(group=1)
    raise ValueError('This saved image has no editable study recipe. Generate its figure group first.')


def render_group(state, recipe, folder, *, cancel=None):
    """Presentation only: consume saved detection, comparison and population results.

    Plot-side densities/envelopes are reconstructed by the original figure builders;
    detectors, feature extraction, clustering and mixture fitting are never invoked.
    """
    group=recipe['group']
    if group==0:
        record=next(r for r in state['records'] if r['tag']==recipe['tag'])
        all_events,accepted,_=compare.reviewed_catalog(record['analysis'],condition(record))
        rendered=replace(record['analysis'],catalog=all_events)
        if 'schmitt_settings' in rendered.provenance:
            return signal_study_figures.recording_figures(rendered,record.get('signal_nulls'),folder,cancel=cancel)
        return incubation_figures.recording_figures(rendered,accepted,folder,cancel=cancel)
    if group==1:
        result=state['comparison']
        if result is None: raise ValueError('Compute recording comparisons first.')
        if result.metadata.get('signal_detector'):
            return signal_study_figures.comparison_figures(result,state['records'],folder,cancel=cancel)
        return incubation_figures.comparison_figures(result.events,result.summary,result.tables,folder,
                    cancel=cancel,settings=compare.ComparisonSettings(**state['comparison_settings']))
    if group==2:
        result=state['clustering']
        if result is None: raise ValueError('Run pooled exploratory clustering first.')
        return incubation_figures.clustering_figures(result[0],folder,prefix='pooled',cancel=cancel,
                    settings=compare.ClusterSettings(**state['cluster_settings']))
    if group==3:
        result=state['mixture']
        if result is None or recipe['tag']!=state.get('mixture_tag'):
            raise ValueError('The saved mixture does not match this figure.')
        record=next(r for r in state['records'] if r['tag']==recipe['tag'])
        all_events,_,_=compare.reviewed_catalog(record['analysis'],condition(record))
        return incubation_figures.appendix_figures(result[0],result[1],replace(record['analysis'],catalog=all_events),
                    state['mixture_folder'],folder,cancel=cancel)
    raise ValueError('Unknown study figure group.')


def reconstruct(state, path, *, cancel=None):
    from ..plot.study_figure_style import capture_figure
    path=Path(path); recipe=infer_recipe(state,path)
    def scientific_sidecar(path):
        return (path.is_file() and path.suffix in ('.csv', '.json')
                and not path.name.endswith('.style.json'))
    with tempfile.TemporaryDirectory(prefix='nanoscope-study-figure-') as tmp:
        with capture_figure(path.stem) as captured:
            render_group(state,recipe,Path(tmp),cancel=cancel)
        if captured.figure is None: raise ValueError('The saved figure recipe did not produce the requested canvas.')
        resources={p.name:p.read_bytes() for p in Path(tmp).iterdir() if scientific_sidecar(p)}
        # Keep the numerical sidecars that accompanied the committed figure as well.
        for p in path.parent.iterdir():
            if scientific_sidecar(p):
                resources.setdefault(p.name,p.read_bytes())
        data=captured.data
        if data is None:
            candidates=[Path(tmp)/(path.stem+suffix) for suffix in ('_points.csv','_events.csv','_panels.csv')]
            candidate=next((p for p in candidates if p.is_file()),None)
            data=pd.read_csv(candidate,float_precision='round_trip') if candidate else pd.DataFrame()
        return captured.figure,data,resources,recipe
