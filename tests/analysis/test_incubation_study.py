"""Scientific-definition and publication tests for the headless study runner.

No TensorFlow/model download or external recording. Detection is replaced by a
known catalog; kernels, bootstrap, tables, transforms and Agg exports stay real.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nanoscope.analysis import incubation_study as study
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.incubation_detection import RecordingAnalysis


def _analysis(tag, n_events=8, *, seed=4, polarity=1):
    rng = np.random.default_rng(seed)
    fs, n = 10000.0, 50000
    signal = rng.normal(0, .8, n).astype(np.float32)
    raw = -200.0000031 + np.linspace(-2, 2, n) + signal.astype(float)
    rows = []
    for i in range(n_events):
        start, width = 300 + i * 500, 4 + i % 19
        end = start + width - 1
        depth = 8.0 + i % 7
        signal[start:end+1] -= depth
        raw[start:end+1] -= depth
        sustained, peak = -float(np.median(signal[start:end+1])), -float(signal[start:end+1].min())
        local = 150.0 + i
        rows.append(dict(tag=tag, start_idx=start, end_idx=end, onset_s=start/fs, end_s=(end+1)/fs,
            dwell_ms=width/fs*1000, duration_ms=width/fs*1000,
            depth_peak_pA=peak, depth_pA=peak, depth_sust_pA=sustained,
            peak_sigma=peak/.8, depth_sigma=peak/.8, sust_sigma=sustained/.8, depth_sust_sigma=sustained/.8,
            I0_local_pA=local, I0_global_pA=200., f_peak=peak/local, f_sust=sustained/local,
            fracblock=sustained/local, passes_gate=i != 0))
    cols = ['tag','start_idx','end_idx','onset_s','end_s','dwell_ms','duration_ms','depth_peak_pA','depth_pA','depth_sust_pA','peak_sigma','depth_sigma','sust_sigma','depth_sust_sigma','I0_local_pA','I0_global_pA','f_peak','f_sust','fracblock','passes_gate']
    catalog = pd.DataFrame(rows, columns=cols)
    if not len(catalog):
        catalog = catalog.astype({c: float for c in cols if c not in ('tag','passes_gate')})
        catalog['passes_gate'] = catalog.passes_gate.astype(bool)
    provenance = dict(event_polarity=polarity, baseline_pA=0., noise_mad_pA=.8,
                      metadata=dict(tag=tag, fs_hz=fs, duration_s=n/fs, qc_mad_noise_pA=.9),
                      inference=dict(uncovered_tail_samples=32), warnings=['Synthetic test input'])
    return RecordingAnalysis(tag, raw, signal, fs, n/fs, catalog, provenance)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    model = tmp_path / 'student.keras'; model.write_text('synthetic; inference patched')
    records, analyses = [], {}
    for hours, diameter in ((2,16),(10,18),(18,18),(30,20)):
        tag = f'zn_{hours}h'
        trace = tmp_path / f'{tag}.csv'; trace.write_text('synthetic trace')
        meta = tmp_path / f'{tag}_metadata.json'
        meta.write_text(json.dumps(dict(tag=tag, fs_hz=10000, duration_s=5., qc_mad_noise_pA=.9)))
        records.append(study.StudyRecording(tag, trace, meta, hours, f'pore_{tag}', diameter))
        analyses[tag] = _analysis(tag, seed=hours)
    def detect(trace, metadata, model, *, progress=None, cancel=None):
        if progress:
            progress(.3, 'Synthetic inference')
            progress(1, 'Synthetic detection complete')
        result = analyses[Path(trace).stem]
        result.provenance['model'] = dict(path=str(model), sha256=study._sha256(Path(model)))
        return result
    monkeypatch.setattr(study, 'analyze_recording', detect)
    return study.StudySpec('Synthetic incubation', records, model), analyses


def test_manifest_resolves_relative_paths_and_freezes_recordings(configured, tmp_path, monkeypatch):
    spec, _ = configured
    mutable = list(spec.recordings)
    snapshot = replace(spec, recordings=mutable)
    mutable.clear()
    assert len(snapshot.recordings) == 4
    path = study.save_spec(snapshot, tmp_path/'recipe'/'study.json')
    contents = json.loads(path.read_text())
    assert contents['recordings'][0]['trace_path'].startswith('../')
    monkeypatch.chdir(tmp_path.parent)
    loaded = study.load_spec(path)
    assert loaded.recordings == snapshot.recordings
    assert loaded.model_path == Path(spec.model_path)
    assert loaded.seed == 0


def test_saving_over_a_read_only_recipe_is_refused_and_changes_nothing(configured, tmp_path):
    """A chmod-400 recipe was measured being replaced without a word. save_spec staged
    with its own mkstemp + os.replace, and os.replace needs write permission on the
    folder, not on the file. It now saves through core.atomic_write's staged_path,
    which refuses by the recipe's own name and leaves the folder as it was."""
    spec, _ = configured
    recipe = tmp_path/'recipe.json'
    recipe.write_text('{"kept": true}\n')
    os.chmod(recipe, 0o400)
    listing = sorted(tmp_path.iterdir())
    try:
        # The door's own sentence, not just the name: an in-place write of a read-only
        # file raises the OS's PermissionError naming recipe.json too.
        with pytest.raises(PermissionError, match=r'recipe\.json is marked read-only'):
            study.save_spec(spec, recipe)
        assert recipe.read_text() == '{"kept": true}\n'
        assert sorted(tmp_path.iterdir()) == listing
    finally:
        os.chmod(recipe, 0o600)


def test_saving_a_recipe_keeps_the_files_own_permissions(configured, tmp_path):
    """mkstemp makes its file 0600 and os.replace installed that file, so a 0644 recipe
    was measured coming back 0600 after a save. staged_path gives the replacement the
    recipe's mode before it lands. 0640 is neither mkstemp's 0600 nor what the usual 022
    umask gives a new file, so a save that installs a new file keeps it only by copying the
    mode across."""
    spec, _ = configured
    recipe = tmp_path/'recipe.json'
    recipe.write_text('{}\n')
    os.chmod(recipe, 0o640)
    study.save_spec(spec, recipe)
    assert recipe.stat().st_mode & 0o777 == 0o640
    assert study.load_spec(recipe).recordings == spec.recordings


def _xattr_names(path):
    # Read by the OS's own tool, not through the ctypes binding the save itself uses.
    if sys.platform == 'darwin':
        return sorted(subprocess.run(['xattr', str(path)], capture_output=True, text=True).stdout.split())
    return sorted(os.listxattr(path))


@pytest.mark.skipif(sys.platform not in ('darwin', 'linux'), reason='extended attributes are a POSIX feature')
def test_saving_a_recipe_keeps_its_extended_attributes(configured, tmp_path):
    """os.replace installs a new file, so the hand-rolled save was measured dropping an
    extended attribute (a Finder tag, a user note) from the recipe on every save.
    staged_path copies them onto the replacement before it lands."""
    spec, _ = configured
    recipe = tmp_path/'recipe.json'
    recipe.write_text('{}\n')
    try:
        if sys.platform == 'darwin':
            subprocess.run(['xattr', '-w', 'user.note', 'lab recipe', str(recipe)], check=True)
        else:
            os.setxattr(recipe, 'user.note', b'lab recipe')
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('this filesystem does not support extended attributes')
    before = _xattr_names(recipe)
    assert 'user.note' in before
    study.save_spec(spec, recipe)
    assert _xattr_names(recipe) == before
    assert study.load_spec(recipe).recordings == spec.recordings


@pytest.mark.parametrize('change,message', [({'seed':68}, 'fixes seed 0'), ({'recipe_version':'other'},'Unsupported')])
def test_reference_recipe_refuses_inert_knobs(configured, change, message):
    spec, _ = configured
    with pytest.raises(ValueError, match=message):
        study._validate_spec(replace(spec, **change))


def test_recording_identity_cannot_be_duplicated(configured):
    spec, _ = configured
    twin = replace(spec.recordings[0], tag='different_label')
    with pytest.raises(ValueError, match='same trace_path'):
        study._validate_spec(replace(spec, recordings=(*spec.recordings,twin)))


def test_bootstrap_matches_literal_notebook_and_respects_time_blocks():
    onsets = np.r_[np.linspace(.01,.05,40), .51, .52, .95]
    got = study.block_bootstrap_rate(onsets, 1.)
    rng = np.random.default_rng(0)
    counts = np.histogram(onsets, bins=np.linspace(0,1,21))[0].astype(float)
    draws = counts[rng.integers(0,20,(4000,20))].sum(1)
    assert got['rate'] == 43
    assert got['lo'] == np.percentile(draws,2.5)
    assert got['hi'] == np.percentile(draws,97.5)
    assert got['fano'] == counts.var(ddof=1)/counts.mean()
    assert got['hi']-got['lo'] > 2*1.96*np.sqrt(len(onsets))
    assert study.block_bootstrap_rate([],1)['hi'] == 0
    with pytest.raises(ValueError, match='clock'):
        study.block_bootstrap_rate([1.0],1.)


def test_features_match_literal_cell51_and_use_exclusive_window_adapter():
    analysis = _analysis('zn_30h', polarity=-1)
    rec = study.StudyRecording('zn_30h',Path('trace'),Path('metadata'),30,'pore',20)
    accepted = analysis.catalog[analysis.catalog.passes_gate].copy()
    for key, val in (('event_id','test'),('incubation_h',30),('pore_nm',20),('pore_id','pore')):
        accepted[key] = val
    measured = study._recording_features(analysis, rec, accepted)
    sig = -analysis.detrended  # actual source cell51 re-reads CSV, un-oriented
    med = float(np.median(sig)); mad = 1.4826*float(np.median(np.abs(sig-med)))
    i0 = abs(float(np.median(analysis.raw)))
    for row_id, row in accepted.iterrows():
        d = med-sig[int(row.start_idx):int(row.end_idx)+1].astype(float)
        n = len(d)
        expected = dict(dwell_ms=n/analysis.fs_hz*1000, log_dwell=np.log10(max(n/analysis.fs_hz*1000,1e-3)),
            depth_max_pA=d.max(),depth_sust_pA=np.median(d),mean_pA=d.mean(),std_pA=d.std(),min_dev_pA=d.min(),
            skewness=stats.skew(d) if n>=5 else 0.,kurtosis=stats.kurtosis(d) if n>=5 else 0.,
            area_pA_ms=d.sum()/analysis.fs_hz*1000,t_peak_frac=np.argmax(d)/max(n-1,1),
            f_sust=np.median(d)/i0,snr=np.median(d)/mad)
        for column, value in expected.items():
            assert measured.loc[row_id,column] == value
    assert len(study.NOTEBOOK_FEATURES) == 12
    assert 'entropy' not in study.NOTEBOOK_FEATURES
    assert (measured.time_half == 'early').all()  # all onsets before actual midpoint
    assert set(measured.time_half_legacy_onset_median) == {'early','late'}


def test_complete_bundle_contains_distinct_definitions_and_reopens(configured, tmp_path):
    spec, analyses = configured
    progress = []
    result = study.run_study(spec, output_dir=tmp_path/'result', progress=lambda f,m: progress.append(f))
    assert len(result.events) == 28
    assert len(result.summary) == 4
    assert np.all(np.diff(progress) >= 0)
    assert progress[-1] == 1
    assert result.metadata['original_clustering']['feature_columns'] == list(study.NOTEBOOK_FEATURES)
    assert result.metadata['original_clustering']['status'] == 'computed'
    assert result.metadata['quantitative_appendix']['status'] == 'mixture_insufficient_events'
    original = pd.read_csv(result.output_dir/'original_clustering'/'features_assignments.csv')
    assert {'hdbscan','kmeans3','PC1'}.issubset(original)
    assert 'entropy' not in original
    for tag, analysis in analyses.items():
        candidates = pd.read_csv(result.output_dir/'catalogs'/f'events_catalog_{tag}.csv')
        assert len(candidates) == 8
        assert candidates.passes_gate.sum() == 7
        assert result.metadata['input_provenance'][tag]['inference']['uncovered_tail_samples'] == 32
        rows = result.events[result.events.tag == tag]
        np.testing.assert_array_equal(rows.fracblock, analysis.catalog.loc[analysis.catalog.passes_gate,'fracblock'])
        i0 = abs(float(np.median(np.asarray(analysis.raw,np.float32))))
        np.testing.assert_array_equal(rows.aggregation_fracblock_global, rows.depth_peak_pA/i0)
        assert not np.allclose(rows.fracblock, rows.aggregation_fracblock_global)
    assert len(result.figure_paths) >= 15
    assert all(path.read_bytes().startswith(b'\x89PNG') for path in result.figure_paths)
    assert result.report_path.is_file()
    assert 'source_module_sha256' in result.metadata
    assert result.metadata['model_identity']['sha256'] == study._sha256(Path(spec.model_path))
    assert all(prov['model']['sha256'] == result.metadata['model_identity']['sha256']
               for prov in result.metadata['input_provenance'].values())
    reopened = study.load_study_result(result.output_dir)
    pd.testing.assert_frame_equal(reopened.events,result.events,check_dtype=False)
    pd.testing.assert_frame_equal(reopened.summary,result.summary,check_dtype=False)
    assert not list(tmp_path.glob('.result.stage-*'))
    assert not list(tmp_path.glob('*.study.lock'))
    (result.output_dir/'summary.csv').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        study.load_study_result(result.output_dir)


def test_aggregation_sweep_preserves_nearest_grid_and_accepted_population():
    events = pd.DataFrame(dict(tag=['a','a','b'],onset_s=[.1,.2,.1],dwell_ms=[1.,2.,3.],
        fracblock=[.08,.02,.03],aggregation_fracblock_global=[.034,.027,.02],f_sust_global=[.03,.025,.01]))
    summary = pd.DataFrame([dict(tag='a',condition='A',duration_s=1.,n_events=2,gate_frac_of_I0=.036,pore_nm=18,pore_id='p1'),
                            dict(tag='b',condition='B',duration_s=1.,n_events=1,gate_frac_of_I0=.025,pore_nm=18,pore_id='p2')])
    tables=study._comparison_tables(events,summary)
    pairs=tables['gate_transfer_pairs'].set_index('definition')
    assert pairs.loc['notebook_cell28_peak_global','native_common_floor'] == .036
    assert pairs.loc['notebook_cell28_peak_global','evaluated_floor'] == .04
    assert pairs.loc['notebook_cell28_peak_global','n_a'] == 0
    assert pairs.loc['sustained_local','n_a'] == 1
    assert pairs.size_matched.all() and not pairs.same_pore_id.any()
    assert len(tables['gate_transfer_sweep']) == 2*3*19
    assert set(tables['rate_block_sensitivity'].n_blocks) == {10,20,40}


@pytest.mark.parametrize('failure', ['cancel','plot_error','destination_race'])
def test_no_partial_result_on_interruption(configured, tmp_path, monkeypatch, failure):
    spec, _ = configured
    from nanoscope.plot import incubation_figures
    target=tmp_path/'result'; sentinel=target/'user_file'
    def interrupted(*args,**kwargs):
        if failure=='cancel':
            raise OperationCancelled('stopped')
        if failure=='plot_error':
            raise RuntimeError('export failed')
        target.mkdir();sentinel.write_text('do not overwrite')
        return []
    monkeypatch.setattr(incubation_figures,'recording_figures',interrupted)
    if failure=='destination_race':
        # Race once, subsequent recording plots are harmless. Rest stays real.
        calls=[]
        def race_once(*args,**kwargs):
            if not calls:
                interrupted();calls.append(True)
            return []
        monkeypatch.setattr(incubation_figures,'recording_figures',race_once)
        error=FileExistsError
    else:
        error=OperationCancelled if failure=='cancel' else RuntimeError
    with pytest.raises(error):
        study.run_study(spec,output_dir=target)
    if failure=='destination_race':
        assert sentinel.read_text()=='do not overwrite'
        assert list(target.iterdir())==[sentinel]
    else:
        assert not target.exists()
    assert not list(tmp_path.glob('.result.stage-*'))
    assert not (tmp_path/'.result.study.lock').exists()


def test_preexisting_outputs_and_precancel_do_not_infer(configured,tmp_path,monkeypatch):
    spec,_=configured
    def forbidden(*a,**k):
        pytest.fail('No inference should run')
    monkeypatch.setattr(study,'analyze_recording',forbidden)
    target=tmp_path/'results';target.mkdir();(target/'keep').write_text('data')
    with pytest.raises(FileExistsError):
        study.run_study(spec,output_dir=target)
    with pytest.raises(OperationCancelled):
        study.run_study(spec,output_dir=tmp_path/'cancelled',cancel=lambda:True)
    assert (target/'keep').read_text()=='data'


def test_model_replaced_between_recordings_aborts_entire_study(configured,tmp_path,monkeypatch):
    spec,_=configured
    from nanoscope.plot import incubation_figures
    analyzed_tags=[]
    detector=study.analyze_recording
    def record_inference(*args,**kwargs):
        result=detector(*args,**kwargs)
        analyzed_tags.append(result.tag)
        return result
    monkeypatch.setattr(study,'analyze_recording',record_inference)
    replacement=tmp_path/'replacement.keras'
    replacement.write_bytes(b'a different model selected between recordings')
    def replace_after_first_recording(*args,**kwargs):
        # This hook runs after the first recording's checked catalog is staged,
        # before the second detector starts. The next provenance honestly names
        # the new content, just as real analyze_recording would.
        replacement.replace(spec.model_path)
        return []
    monkeypatch.setattr(incubation_figures,'recording_figures',replace_after_first_recording)
    target=tmp_path/'mixed-model-study'
    with pytest.raises(ValueError,match='same frozen model'):
        study.run_study(spec,output_dir=target)
    assert analyzed_tags==['zn_2h','zn_10h']
    assert not target.exists()
    assert not list(tmp_path.glob('.mixed-model-study.stage-*'))
    assert not (tmp_path/'.mixed-model-study.study.lock').exists()
    assert Path(spec.model_path).read_bytes()==b'a different model selected between recordings'


def test_tiny_actual_gmm_appendix_keeps_labels_aligned(configured,tmp_path,monkeypatch):
    spec,_=configured
    analysis=_analysis('zn_30h',n_events=65,seed=123)
    accepted=analysis.catalog[analysis.catalog.passes_gate].copy()
    for key,val in (('event_id','test'),('incubation_h',30),('pore_nm',20),('pore_id','pore')):
        accepted[key]=val
    features=study._recording_features(analysis,spec.recordings[-1],accepted).reset_index(drop=True)
    real_quantify=study.SG.quantify_subpopulations
    def bounded(*args,**kwargs):
        # Keep real selection/fit/bootstrap/waveform integration; only bound the grid.
        return real_quantify(*args,**kwargs,p_grid=(2,),k_grid=(1,2),n_splits=2,n_boot=3,
                             cv_n_init=1,fit_n_init=1,bic_n_init=1,boot_n_init=1)
    monkeypatch.setattr(study.SG,'quantify_subpopulations',bounded)
    folder=tmp_path/'appendix';folder.mkdir()
    table,quant,info=study._appendix(features,analysis,spec,folder)
    assert quant is not None and quant.n_events==64
    np.testing.assert_array_equal(table.subpop,quant.fit.labels)
    np.testing.assert_allclose(table[[c for c in table if c.startswith('posterior_')]].sum(axis=1),1)
    assert (folder/'population_quantification.csv').is_file()
    population=pd.read_csv(folder/'population_quantification.csv',float_precision='round_trip')
    np.testing.assert_array_equal(population.assigned_event_fraction,population.n/quant.n_events)
    np.testing.assert_array_equal(population.mixture_weight,quant.fit.weights)
    np.testing.assert_array_equal(population.mixture_weight_lo95,quant.ci.lo)
    np.testing.assert_array_equal(population.mixture_weight_hi95,quant.ci.hi)
    assert not {'fraction','frac_lo95','frac_hi95'}.intersection(population.columns)
    assert 'mixture_weight_interval' in info['fraction_definitions']
    assert info['bootstrap']['diagnostic_estimand']=='fitted mixture weights'
    assert info['bootstrap']['warnings']==list(quant.ci.warnings)
    assert pd.read_csv(folder/'population_median_waveforms.csv').shape[0]==200
    assert info['time_half'].startswith('recording_clock_midpoint')
    from nanoscope.plot.incubation_figures import appendix_figures
    figures=tmp_path/'figures';figures.mkdir()
    paths=appendix_figures(table,quant,analysis,folder,figures)
    assert len(paths)==3
    assert all(p.read_bytes().startswith(b'\x89PNG') for p in paths)
    figure_data=pd.read_csv(figures/'quantitative_appendix_selection_fractions_waveforms.data.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(figure_data,population)



def test_zero_accepted_events_is_a_complete_explicit_result(configured,tmp_path,monkeypatch):
    spec,analyses=configured
    for analysis in analyses.values():
        analysis.catalog.loc[:,"passes_gate"]=False
    from nanoscope.plot import incubation_figures
    # Gallery empty rendering is covered by its own helper check; this verifies
    # the complete empty scientific/table path without repeating large trace PNGs.
    monkeypatch.setattr(incubation_figures,"recording_figures",lambda *a,**k:[])
    result=study.run_study(spec,output_dir=tmp_path/"empty")
    assert result.events.empty
    assert (result.summary.n_events==0).all()
    assert (result.summary.rate_per_s==0).all()
    assert result.metadata["original_clustering"]["status"]=="insufficient_events"
    assert result.metadata["quantitative_appendix"]["status"]=="insufficient_events"
    assert study.load_study_result(result.output_dir).events.empty


def test_frozen_archive_uses_embedded_build_without_reading_virtual_sources(tmp_path,monkeypatch):
    import sys
    from nanoscope import build_info
    monkeypatch.setattr(sys,"frozen",True,raising=False)
    monkeypatch.setattr(build_info,"build_info",lambda:dict(schema_version=1,build_id="test-build",
                            source_sha256="12345",git_commit="abc",environment={"packages":{"numpy":"test"}}))
    identity=study._source_identity({"analysis/incubation_study.py":tmp_path/"virtual-PYZ"/"incubation_study.py"})
    assert identity["source_module_hash_status"]=="unavailable_in_frozen_archive"
    assert identity["source_module_sha256"]=={}
    assert identity["build_record"]["source_sha256"]=="12345"
    assert identity["unavailable_module_sources"]==["analysis/incubation_study.py"]
