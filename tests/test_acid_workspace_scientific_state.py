"""Independent tests of scientific identity across portable GUI stages.

Numerical kernels are replaced with tiny deterministic stand-ins only where
necessary to isolate workspace orchestration, never to certify numerical parity.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from nanoscope.analysis import acid_incubation as acid


def catalog(offset=0):
    return pd.DataFrame(dict(event_id=[0,1,2,3],start_idx=np.arange(4)*100+offset,
        end_idx=np.arange(4)*100+offset+9,accepted=[True]*4))


def state_one(tmp_path):
    state=acid.new_state(records=[dict(tag='acid_30h_whole',source_path=tmp_path/'raw.abf',
        source_info={'sha256':'rawsha'},incubation_h=30)])
    row=state['records'][0]
    row['prepared']={'parquet_path':tmp_path/'prepared.parquet','metadata':{'fs_hz':50000,'n_samples':1000},
        'settings_sha256':acid._digest({k:state['settings'][k] for k in acid.PREPARE_KEYS})}
    row['catalog']=catalog()
    row['detection']={'context_path':tmp_path/'context','primary_set':'cusum_matched','settings_sha256':'settings'}
    return state


def test_restore_retains_actual_calibration_fallback(tmp_path,monkeypatch):
    state=state_one(tmp_path);row=state['records'][0]
    ns=acid.default_settings()
    def load(rt,prepared):
        rt.namespace.update(signal=np.zeros(1000),raw=np.zeros(1000),fs=50000,n_samples=1000)
    monkeypatch.setattr(acid,'_load_prepared',load)
    monkeypatch.setattr(acid.checkpoint,'load_workspace',lambda p:dict(primary_polarity=1,analysis_sets={'cusum_matched':catalog()},detector_catalogs={'cusum':catalog()},cusum_selected={'pulse':(8,10),'level':(5,70)}))
    def definitions():
        ns.update(robust_quiet_stats=lambda x:(0,1,np.ones(len(x),bool)),
            regions_to_mask=lambda n,regions:np.zeros(n,bool))
    rt=SimpleNamespace(namespace=ns,cancel=None,definitions=definitions)
    acid._restore_context(rt,row)
    assert ns['PRIMARY_SET_NAME']=='cusum_matched', 'Restored analysis relabels fallback-selected events as unconfirmed CUSUM'
    assert ns['CUSUM_SCALES']['pulse']['block_ms']==2.
    assert ns['CUSUM_SCALES']['level']['block_ms']==50.


def test_alternative_primary_preserves_detector_source_identity(tmp_path,monkeypatch):
    state=state_one(tmp_path);state['settings']['PRIMARY_SET_NAME']='threshold'
    threshold=catalog(offset=30);cusum=catalog(offset=10)
    class Runtime:
        def __init__(self,settings,folder,record,allrecords,**kwargs):
            self.figures=[]
            self.namespace=dict(cusum_catalog=cusum,threshold_catalog=threshold,
                primary_catalog=threshold,PRIMARY_SET_NAME='threshold',primary_polarity=1,
                analysis_sets={'threshold':threshold},detector_catalogs={'threshold':threshold,'cusum':cusum},n_samples=1000,OUT_DIR=Path(folder),cusum_selected={},
                cusum_calibration_ok=True,cusum_false_alarm_rate=0.,cusum_false_alarm_rate_gated=0.)
        def run(self,index):pass
    monkeypatch.setattr(acid,'AcidRuntime',Runtime)
    monkeypatch.setattr(acid,'_load_prepared',lambda *a:None)
    monkeypatch.setattr(acid,'_save_context',lambda ns,p:p)
    result=acid.detect_record(state,'acid_30h_whole',tmp_path)
    pd.testing.assert_series_equal(result['records'][0]['catalog']['start_idx'],threshold['start_idx'])


@pytest.mark.parametrize("review_first",[False,True])
def test_family_curation_rerun_keeps_frozen_reference_model(tmp_path,monkeypatch,review_first):
    state=acid.confirm_selection(state_one(tmp_path),'acid_30h_whole')
    state['settings']['FAMILY_REFIT_REFERENCE']=True
    class Runtime:
        def __init__(self,settings,folder,record,allrecords,**kwargs):
            self.figures=[];self.record=record;self.folder=Path(folder);self.refit=settings.get('FAMILY_REFIT_REFERENCE',False)
            self.namespace=dict(OUT_DIR=Path(folder)/(record['tag']+'_subpopulation'),
                FAMILY_MODEL_PATH=Path(folder)/'event_family_model.json',PRIMARY_SET_NAME='cusum_matched',
                primary_polarity=1,duration_s=20.,FIGURE_GROUPS={'cohort':[]})
        def run(self,index):
            if index!=32:return
            ns=self.namespace;ns['OUT_DIR'].mkdir(parents=True,exist_ok=True)
            path=ns['FAMILY_MODEL_PATH']
            # This is the exact load-existing-else-fit policy of stage_032.
            if not path.exists() or self.refit:
                path.write_text(json.dumps({'reference_ids':ns['primary_catalog'].event_id.tolist()}))
            fam=ns['primary_catalog'].copy();fam['family']=fam.event_id%2
            fam['family_posterior']=1.;fam['in_burst']=False
            ns['_fam']=fam;ns['event_families']={}
    def restore(rt,row):rt.namespace['primary_catalog']=row['catalog'].loc[row['catalog'].accepted].copy()
    monkeypatch.setattr(acid,'AcidRuntime',Runtime)
    monkeypatch.setattr(acid,'_restore_context',restore)
    monkeypatch.setattr(acid,'CORE_ANALYSIS_STAGES',(32,))
    monkeypatch.setattr(acid,'SUBSTRUCTURE_STAGES',())
    first=acid.build_families(state,tmp_path) if review_first else acid.analyze_study(state,tmp_path)
    if review_first:assert first['analysis'] is None
    original=Path(first['family_model_path']).read_bytes()
    curated=acid.set_family_acceptance(first,'acid_30h_whole',[0],False)
    assert first['records'][0]['catalog'].accepted.all(), 'Curation must not mutate previous revision'
    assert curated['analysis'] is None, 'Curation must retire prior analysis'
    curated=acid.confirm_selection(curated,'acid_30h_whole')
    second=acid.analyze_study(curated,tmp_path)
    assert second['records'][0]['analysis']['summary']['n_accepted']==2
    assert Path(second['analysis']['family_model_path']).read_bytes()==original, 'Family rejection must not redefine the family model or its IDs'
    assert set(second['records'][0]['families'].event_id)=={0,1,2,3}, 'Rejected family identities must remain available for reacceptance'
    reaccepted=acid.set_family_acceptance(second,'acid_30h_whole',[0],True)
    assert reaccepted['records'][0]['catalog'].accepted.all()


def test_math_session_restores_random_and_plot_state_on_failure():
    import matplotlib
    before=np.random.get_state()
    size=matplotlib.rcParams['font.size']
    try:
        with acid._math_session(91):
            np.random.random(100)
            matplotlib.rcParams['font.size']=91.
            raise RuntimeError('simulated cancelled stage')
    except RuntimeError:
        pass
    after=np.random.get_state()
    assert before[0]==after[0]
    np.testing.assert_array_equal(before[1],after[1])
    assert before[2:]==after[2:]
    assert matplotlib.rcParams['font.size']==size


def test_restored_context_keeps_derived_amplitude_contract(tmp_path):
    needed={'AMP_BASE':'peak','AMP_PA_COL':'peak_depth_pA','AMP_SIGMA_COL':'peak_depth_sigma',
            'AMP_FRAC_COL':'peak_fracchange','PRIMARY_SET_NAME':'cusum_matched','SUSTAINED_GATE_PA':4.2}
    location=acid._save_context(needed,tmp_path/'context')
    actual=acid.checkpoint.load_workspace(location)
    assert {k:actual.get(k) for k in needed}==needed, 'Baseline sensitivity must use the same derived amplitude columns after review/reopen'


@pytest.mark.parametrize('orientation',[1,-1])
def test_prepared_orientation_is_applied_once_to_current_and_signal(tmp_path,monkeypatch,orientation):
    detr=np.array([1.,2.,-3.],np.float32);raw=np.array([-100.,-101.,-96.],np.float32)
    meta={'fs_hz':50000,'event_polarity':orientation,'raw_mean_pA':-99.}
    ns={'definitions_007':lambda:None,'load_stage0':lambda *a:(detr.copy(),raw.copy(),meta)}
    rt=SimpleNamespace(namespace=ns,record={'tag':'test','source_path':tmp_path/'raw.abf','source_info':{'sha256':'raw'}},cancel=None)
    monkeypatch.setattr(acid,'_file_hash',lambda *a:'prepared')
    acid._load_prepared(rt,{'parquet_path':tmp_path/'signal.parquet','sha256':'prepared'})
    np.testing.assert_array_equal(ns['signal'],orientation*detr)
    np.testing.assert_array_equal(ns['raw'],orientation*raw)
    assert ns['polarity']==orientation and ns['metadata_i0']==99.


def test_detection_scale_metadata_survives_callable_calibration_helpers(tmp_path):
    scales={'pulse':{'block_ms':2.,'pass1_block_ms':50.,'cell_mask':lambda frame:frame},
            'level':{'block_ms':50.,'pass1_block_ms':1000.,'cell_mask':lambda frame:frame}}
    location=acid._save_context({'CUSUM_SCALES':scales},tmp_path/'context')
    restored=acid.checkpoint.load_workspace(location)
    assert {k:restored['CUSUM_SCALES'][k]['block_ms'] for k in scales}=={'pulse':2.,'level':50.}
    assert all('cell_mask' not in restored['CUSUM_SCALES'][k] for k in scales)
    assert callable(scales['pulse']['cell_mask']), 'Saving must not mutate the live calibration contract'


def test_curated_primary_is_separate_from_noise_gate_diagnostics(tmp_path,monkeypatch):
    state=state_one(tmp_path);row=state['records'][0]
    row['catalog']['passes_gate']=[True,True,False,True]
    row['catalog']['accepted']=[True,False,True,False]  # includes explicit acceptance below automatic gate
    ns=acid.default_settings()
    def load(rt,prepared):rt.namespace.update(signal=np.zeros(1000),raw=np.zeros(1000),fs=50000,n_samples=1000)
    monkeypatch.setattr(acid,'_load_prepared',load)
    context=dict(primary_polarity=1,analysis_sets={'cusum_matched':catalog()},detector_catalogs={'cusum':catalog()},cusum_selected={'pulse':(8,10),'level':(5,70)})
    monkeypatch.setattr(acid.checkpoint,'load_workspace',lambda p:context)
    def definitions():ns.update(robust_quiet_stats=lambda x:(0,1,np.ones(len(x),bool)),regions_to_mask=lambda n,regions:np.zeros(n,bool))
    acid._restore_context(SimpleNamespace(namespace=ns,cancel=None,definitions=definitions),row)
    assert set(ns['primary_catalog'].event_id)=={0,2}
    assert set(ns['analysis_sets']['cusum_matched'].event_id)=={0,2}
    assert set(ns['cusum_catalog'].loc[ns['cusum_catalog'].accepted,'event_id'])=={0,1,3}
    assert set(ns['cusum_catalog'].loc[ns['cusum_catalog'].selected,'event_id'])=={0,2}
    assert set(row['catalog'].loc[row['catalog'].accepted,'event_id'])=={0,2}, 'Restore must not replace user selection with the automatic gate'
