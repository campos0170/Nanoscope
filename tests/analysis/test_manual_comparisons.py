"""Independent stage composition, review selection and parameter propagation."""
import json
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from nanoscope.analysis import incubation_study as preset, manual_comparisons as manual
from nanoscope.analysis.incubation_detection import RecordingAnalysis


def sample(tag='source', n_events=60, seed=4):
    rng=np.random.default_rng(seed); fs=10000.; n=60000
    signal=rng.normal(0,.8,n).astype(np.float32)
    raw=-200.0000031+np.linspace(-2,2,n)+signal.astype(float)
    rows=[]
    for i in range(n_events):
        start=300+i*500; width=4+i%19; end=start+width-1; depth=8+i%7
        signal[start:end+1]-=depth; raw[start:end+1]-=depth
        sustained=-float(np.median(signal[start:end+1])); peak=-float(signal[start:end+1].min())
        rows.append(dict(tag=tag,start_idx=start,end_idx=end,onset_s=start/fs,end_s=(end+1)/fs,
            dwell_ms=width/fs*1000,depth_peak_pA=peak,depth_sust_pA=sustained,
            fracblock=sustained/(150+i),passes_gate=i!=0))
    a=RecordingAnalysis(tag,raw,signal,fs,n/fs,pd.DataFrame(rows),dict(event_polarity=1,
        baseline_pA=0.,noise_mad_pA=.8,metadata=dict(qc_mad_noise_pA=.9),inference=dict(uncovered_tail_samples=32)))
    r=preset.StudyRecording(tag,'unused.csv','unused.json',2.,'pore_'+tag,18.)
    return a,r


@pytest.mark.parametrize('kind,change',[
    (manual.ComparisonSettings,dict(n_blocks=1)),(manual.ComparisonSettings,dict(seed=True)),
    (manual.ComparisonSettings,dict(floor_step=0)),(manual.ComparisonSettings,dict(floor_stop=float('nan'))),
    (manual.ClusterSettings,dict(kmeans_k=0)),(manual.ClusterSettings,dict(pca_variance=1.01)),
    (manual.ClusterSettings,dict(feature_columns=('not_measured',))),
    (manual.MixtureSettings,dict(cv_folds=1)),(manual.MixtureSettings,dict(p_grid=(14,))),
    (manual.MixtureSettings,dict(k_grid=(1,1))),
])
def test_settings_reject_inert_or_invalid_choices(kind,change):
    with pytest.raises(ValueError): kind(**change)


def test_compare_defaults_match_reference_formulas_and_preserve_inputs(tmp_path):
    a,r=sample(); b,s=sample('second',seed=12)
    originals=[x.catalog.copy(deep=True) for x in (a,b)]
    got=manual.compare_recordings([a,b],[r,s],folder=tmp_path)
    for name,frame in preset._comparison_tables(got.events,got.summary).items():
        pd.testing.assert_frame_equal(got.tables[name],frame)
    for row,x in zip(got.summary.itertuples(),(a,b)):
        rate=preset.block_bootstrap_rate(x.catalog.loc[x.catalog.passes_gate,'onset_s'],x.duration_s)
        assert row.rate_per_s==rate['rate'] and row.lo==rate['lo'] and row.hi==rate['hi']
    for x,old in zip((a,b),originals): pd.testing.assert_frame_equal(x.catalog,old)
    assert json.loads((tmp_path/'report.json').read_text())['settings']['n_boot']==4000


def test_manual_acceptance_replaces_gate_and_is_shared_with_features(tmp_path):
    a,r=sample(n_events=8); a.catalog['accepted']=False; a.catalog.loc[[0,3],'accepted']=True
    a.catalog['event_id']=['kept-id-'+str(i) for i in range(8)]
    result=manual.compare_recordings([a],[r],folder=tmp_path)
    features=manual.extract_features([a],[r])
    assert result.summary.iloc[0].n_events==2
    assert list(result.events.event_id)==list(features.event_id)==['kept-id-0','kept-id-3']
    assert result.metadata['selection_columns']==features.attrs['methods']['selection_columns']=={'source':'accepted'}
    assert not result.events.iloc[0].passes_gate


def test_parameter_changes_recompute_rates_and_sweep(tmp_path):
    a,r=sample()
    settings=manual.ComparisonSettings(n_blocks=4,n_boot=21,seed=19,floor_start=.03,floor_stop=.09,
        floor_step=.03,sensitivity_blocks=(3,7))
    got=manual.compare_recordings([a],[r],settings,folder=tmp_path)
    rate=preset.block_bootstrap_rate(got.events.onset_s,a.duration_s,4,21,19)
    assert got.summary.iloc[0].lo==rate['lo'] and got.summary.iloc[0].hi==rate['hi']
    assert got.tables['gate_transfer_sweep'].floor.unique().tolist()==[.03,.06,.09]
    assert got.tables['rate_block_sensitivity'].n_blocks.tolist()==[3,7]
    assert set(got.tables['rate_block_sensitivity'].n_boot)=={21}


def test_corrupt_clock_and_ambiguous_acceptance_are_refused(tmp_path):
    a,r=sample(n_events=8); a.catalog.loc[2,'onset_s']+=.1
    with pytest.raises(ValueError,match='clock'): manual.extract_features([a],[r])
    a,r=sample(n_events=8); a.catalog['accepted']='False'
    with pytest.raises(ValueError,match='boolean'): manual.compare_recordings([a],[r],folder=tmp_path)


def test_feature_stage_matches_reference_without_downstream_analysis(monkeypatch):
    a,r=sample(); _,accepted,_=manual.reviewed_catalog(a,r)
    expected=preset._recording_features(a,r,accepted).reset_index(drop=True)
    def forbidden(*args,**kwargs): raise AssertionError('Feature stage invoked downstream analysis')
    monkeypatch.setattr(manual.SC,'cluster_scopes',forbidden)
    monkeypatch.setattr(manual.SG,'quantify_subpopulations',forbidden)
    pd.testing.assert_frame_equal(manual.extract_features([a],[r]),expected)


def test_mirrored_upward_features_match_downward_without_changing_notebook_recipe(tmp_path):
    down, condition = sample(n_events=8)
    up = deepcopy(down); up.provenance['event_polarity'] = -1; up.raw = -down.raw
    down_features = manual.extract_features([down], [condition])
    up_features = manual.extract_features([up], [condition])
    pd.testing.assert_frame_equal(up_features, down_features, check_exact=True)
    assert (up_features[['depth_sust_pA', 'depth_max_pA', 'snr', 'area_pA_ms']] > 0).all().all()
    assert up_features.attrs['methods']['feature_orientation_version'] == 2
    assert up_features.attrs['methods']['source_event_polarities'] == {'source': -1}
    _, accepted, _ = manual.reviewed_catalog(up, condition)
    legacy = preset._recording_features(up, condition, accepted)
    assert (legacy.depth_sust_pA < 0).all()  # frozen original-sign notebook compatibility
    labels, metadata = manual.cluster_features(up_features, folder=tmp_path)
    assert metadata['feature_methods'] == up_features.attrs['methods']
    assert metadata['amplitude'] == up_features.attrs['methods']['amplitude']
    assert labels.attrs['methods']['feature_orientation_version'] == 2


def test_legacy_upward_feature_cache_requires_reextraction_only_for_affected_tags():
    down, condition = sample(n_events=8)
    old = manual.extract_features([down], [condition]); old.attrs.clear()
    assert manual.feature_orientation_issue(old, [down]) is None
    up = deepcopy(down); up.provenance['event_polarity'] = -1
    assert 'Extract waveform features again' in manual.feature_orientation_issue(old, [up])
    fresh = manual.extract_features([up], [condition])
    assert manual.feature_orientation_issue(fresh, [up]) is None
    up.tag = 'not_in_these_features'
    assert manual.feature_orientation_issue(old, [up]) is None


def test_clustering_defaults_match_reference_and_knobs_work(tmp_path):
    a,r=sample(); features=manual.extract_features([a],[r]); before=features.copy(deep=True)
    original=tmp_path/'reference'; original.mkdir()
    expected,info=preset._original_clustering(features,original)
    got,meta=manual.cluster_features(features,folder=tmp_path/'manual')
    pd.testing.assert_frame_equal(got,expected)
    assert meta['n_pcs']==info['n_pcs']
    changed,meta2=manual.cluster_features(features,manual.ClusterSettings(kmeans_k=2,max_pcs=2,
        pca_variance=.80,with_tsne=False),folder=tmp_path/'changed')
    assert 'kmeans2' in changed and 'kmeans3' not in changed and 'tsne_x' not in changed
    assert set(changed.kmeans2)=={0,1} and meta2['n_pcs']<=2
    pd.testing.assert_frame_equal(features,before)


def test_mixture_uses_selected_source_and_editable_parameters(tmp_path,monkeypatch):
    a,r=sample(n_events=75); b,s=sample('second',n_events=75,seed=10)
    features=manual.extract_features([a,b],[r,s]); before=features.copy(deep=True)
    original=manual.SG.quantify_subpopulations; calls=[]
    def recorded(frame,**kwargs):
        calls.append((set(frame.tag),dict(kwargs))); return original(frame,**kwargs)
    monkeypatch.setattr(manual.SG,'quantify_subpopulations',recorded)
    settings=manual.MixtureSettings(seed=7,n_boot=3,cv_folds=2,bootstrap_seed=9,p_grid=(2,),k_grid=(1,2))
    assigned,quant,info=manual.fit_mixture(features,b,settings,folder=tmp_path)
    assert calls[0][0]=={'second'}
    kw=calls[0][1]
    assert (kw['n_boot'],kw['n_splits'],kw['random_state'],kw['bootstrap_seed'])==(3,2,7,9)
    assert kw['p_grid']==(2,) and kw['k_grid']==(1,2)
    assert assigned.loc[assigned.tag=='source','subpop'].isna().all()
    assert assigned.loc[assigned.tag=='second','subpop'].notna().all()
    assert quant.n_events==74 and quant.ci.n_boot==3
    assert info['feature_methods'] == features.attrs['methods']
    table=pd.read_csv(tmp_path/'population_quantification.csv')
    assert 'assigned_event_fraction' in table and 'mixture_weight' in table
    assert np.isclose(table.assigned_event_fraction.sum(),1.) and info['mixture_tag']=='second'
    pd.testing.assert_frame_equal(features,before)


def test_figure_labels_follow_committed_comparison_and_cluster_settings(tmp_path,monkeypatch):
    from nanoscope.plot import incubation_figures as figures
    a,r=sample(); captures={}
    def capture(fig,folder,name,source=None,**kwargs):
        captures[name]=fig
        return tmp_path/f'{name}.png'
    monkeypatch.setattr(figures,'_save',capture)
    settings=manual.ComparisonSettings(n_blocks=4,n_boot=21,seed=8)
    comparison=manual.compare_recordings([a],[r],settings,folder=tmp_path/'comparison')
    figures.comparison_figures(comparison.events,comparison.summary,comparison.tables,tmp_path,settings=settings)
    assert '4 blocks; 21 draws' in captures['incubation_rates_and_medians'].axes[0].get_title()
    features=manual.extract_features([a],[r])
    cs=manual.ClusterSettings(kmeans_k=2,feature_columns=manual.NOTEBOOK_FEATURES[:5],with_tsne=False)
    labels,_=manual.cluster_features(features,cs,folder=tmp_path/'clusters')
    figures.clustering_figures(labels,tmp_path,settings=cs)
    assert captures['original12_pca'].axes[0].get_title()=='kmeans2'
    assert 'Selected 5-feature' in captures['original12_pca']._suptitle.get_text()
    assert 'Single recording' in captures['original12_pca']._suptitle.get_text()
    assert 'pore confound remains' not in captures['original12_pca']._suptitle.get_text()
    for figure in captures.values(): figure.clear()


@pytest.mark.parametrize('mixed', [False, True])
def test_comparison_captions_use_real_pores_and_review_selection(tmp_path, monkeypatch, mixed):
    from nanoscope.plot import incubation_figures as figures
    analyses, records = [], []
    for i in range(3 if mixed else 2):
        analysis, record = sample(tag=f'recording_{i}', n_events=8, seed=4+i)
        if i < 2:
            analysis.catalog['accepted'] = True  # includes a native-gate failure
        records.append(replace(record, pore_id='shared' if i < 2 else 'other',
            pore_diameter_nm=18. if i < 2 else 20.))
        analyses.append(analysis)
    settings = manual.ComparisonSettings(n_boot=5)
    comparison = manual.compare_recordings(analyses, records, settings, folder=tmp_path/'tables')
    captures = {}
    def capture(fig, folder, name, source=None, **kwargs):
        captures[name] = (fig._suptitle.get_text(), [ax.get_title() for ax in fig.axes])
        fig.clear()
        return tmp_path/f'{name}.png'
    monkeypatch.setattr(figures, '_save', capture)
    figures.comparison_figures(comparison.events, comparison.summary, comparison.tables,
                               tmp_path, settings=settings)
    expected_pores = 'shared and distinct pore IDs' if mixed else 'Same physical pore'
    for name, (title, _) in captures.items():
        assert ('Same physical pore' if name.startswith('size_matched_18') else expected_pores) in title
        assert 'reviewed acceptance' in title
        assert 'retained native-gate events' not in title
        assert 'detector transfer differs' not in title
    title, panels = captures['common_floor_kde']
    assert 'fresh-detector fallback' not in title
    assert ('All selected events (review or gate)' if mixed else
            'All accepted events (review state)') in panels
    if mixed:
        assert 'reviewed acceptance or native gate by recording' in captures['gate_transfer_sweep'][0]
        assert 'or native gate' not in captures['size_matched_18nm_sustained_local'][0]
    # The plotted inputs remain the reviewed selection, including gate failures.
    assert not comparison.events.iloc[0].passes_gate
    assert len(comparison.events) == (23 if mixed else 16)


def test_legacy_comparison_captions_infer_gate_and_handle_missing_pore_identity(tmp_path, monkeypatch):
    from nanoscope.plot import incubation_figures as figures
    a, r = sample(n_events=8); b, s = sample('second', n_events=8)
    comparison = manual.compare_recordings([a,b], [r,s],
        manual.ComparisonSettings(n_boot=5), folder=tmp_path)
    # Older serialized comparisons have neither policy attrs nor accepted flags.
    comparison.events.attrs.clear(); comparison.summary.attrs.clear()
    caption = figures._comparison_caption(comparison.events, comparison.summary)
    assert 'Different physical pores' in caption and 'Selection: native gate' in caption
    assert figures._selection_caption(comparison.events, comparison.summary)[1] == 'All native-gate events'
    comparison.summary.loc[0, 'pore_id'] = None
    assert 'Physical pore identities incomplete' in figures._comparison_caption(comparison.events, comparison.summary)


def test_common_floor_uses_user_measurement_gate_including_disabled_gate(tmp_path):
    a,r=sample()
    a.provenance['contract']={'min_depth_sigma':4.}
    a.provenance['measurement_settings']={'min_depth_sigma':6.}
    result=manual.compare_recordings([a],[r],folder=tmp_path/'six')
    row=result.summary.iloc[0]
    assert row.gate_frac_of_I0==6*.9/row.I0_pA
    assert row.measured_gate_frac_global==6*.8/row.I0_pA
    a.provenance['measurement_settings']['min_depth_sigma']=0.
    zero=manual.compare_recordings([a],[r],folder=tmp_path/'zero')
    assert zero.summary.iloc[0].gate_frac_of_I0==0.
