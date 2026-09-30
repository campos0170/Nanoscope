"""Inspectable, staged acid-insulin incubation workflow with portable checkpoints.

All numerical methods are static NanoScope Python functions. A private namespace
per AcidRuntime preserves the notebook's function contracts without shared
scientific globals, notebook execution, or writes to original data directories.
Public stages return a new state only after success; the GUI publishes it with
the normal atomic workspace codec. Event and family curation retire dependent
results. No default decision is represented as human review.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import threading
import types
import uuid
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .cancel import check_cancelled
from . import manual_workspace as checkpoint
from .manual_preparation import inspect_raw
from .acid_incubation_config import DEFAULTS

SCHEMA = 'nanoscope-acid-incubation-v1'
_MATH_LOCK = threading.RLock()  # matplotlib's stateful canvas is not thread-safe.
PREPARE_KEYS = frozenset(('ABF_CHANNEL','LINE_NOISE_METHOD','MAINS_NOMINAL_HZ','MAINS_MIN_PROMINENCE',
    'NOTCH_FREQS','NOTCH_BW_HZ','LINE_FIT_WINDOW_S','LINE_FIT_MASK_SIGMA','BASELINE_WINDOW_S',
    'BASELINE_DECIM','STORE_RAW_AS_CODES','MATCH_TAU_MS','MATCH_MIN_ROLLOFF_HZ','BESSEL_ORDER','FAMILY_REFERENCE_TAG'))
DETECTION_STAGES = (10,11,12,13,14,16,17,18,20,22,24,25,26)
CORE_ANALYSIS_STAGES = (28,30,32,34)
SUBSTRUCTURE_STAGES = (40,41,42,44,45,47,48,49,50,51,53,54,56,57,58,59,60,62,63,65,66,67,69,71,73)


@contextmanager
def _math_session(seed):
    import matplotlib
    with _MATH_LOCK,matplotlib.rc_context():
        previous=np.random.get_state()
        np.random.seed(seed)
        try:yield
        finally:np.random.set_state(previous)


class _AcidPlot:
    """Minimal private Agg pyplot surface; it never creates a Qt manager."""
    def __init__(self,on_saved=None):self.current=None;self.owned=[];self.on_saved=on_saved
    def figure(self,*args,**kwargs):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        kwargs.pop('num',None)
        self.current=Figure(**kwargs);FigureCanvasAgg(self.current);self.owned.append(self.current)
        if self.on_saved is not None:
            fig=self.current;original=fig.savefig
            def savefig(path,*a,**kw):
                result=original(path,*a,**kw)
                if isinstance(path,(str,Path)) and str(path).endswith('.png'):self.on_saved(fig,Path(path))
                return result
            fig.savefig=savefig
        return self.current
    def subplots(self,nrows=1,ncols=1,**kwargs):
        subplot={k:kwargs.pop(k) for k in ('sharex','sharey','squeeze','subplot_kw','gridspec_kw') if k in kwargs}
        fig=self.figure(**kwargs);return fig,fig.subplots(nrows,ncols,**subplot)
    def gcf(self):return self.current if self.current is not None else self.figure()
    def gca(self):return self.gcf().axes[-1] if self.gcf().axes else self.gcf().add_subplot()
    def close(self,figure=None):
        figure=figure or self.current
        if figure in self.owned:self.owned.remove(figure)
        if self.current is figure:self.current=self.owned[-1] if self.owned else None
        if figure is not None:figure.clear()
    def show(self,*a,**k):self.close()
    def colorbar(self,*a,**k):return self.gcf().colorbar(*a,**k)
    def joint_grid(self,height=7,ratio=4):
        fig=self.figure(figsize=(height,height));g=fig.add_gridspec(ratio+1,ratio+1,hspace=.05,wspace=.05)
        joint=fig.add_subplot(g[1:,:-1]);mx=fig.add_subplot(g[0,:-1],sharex=joint);my=fig.add_subplot(g[1:,-1],sharey=joint)
        mx.tick_params(labelbottom=False);my.tick_params(labelleft=False)
        return types.SimpleNamespace(fig=fig,figure=fig,ax_joint=joint,ax_marg_x=mx,ax_marg_y=my)
    def __getattr__(self,name):
        import matplotlib
        if name=='rcParams':return matplotlib.rcParams
        if name=='Line2D':
            from matplotlib.lines import Line2D;return Line2D
        if name=='Rectangle':
            from matplotlib.patches import Rectangle;return Rectangle
        names={'xlabel':'set_xlabel','ylabel':'set_ylabel','title':'set_title'}
        return getattr(self.gca(),names.get(name,name))


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, pd.DataFrame): return value.to_dict(orient='records')
    if isinstance(value, pd.Series): return value.tolist()
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=_plain,allow_nan=True).encode()).hexdigest()


def _file_hash(path,cancel=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            check_cancelled(cancel); h.update(block)
    return h.hexdigest()


def default_settings():
    return deepcopy(DEFAULTS)


def validate_settings(settings):
    if not isinstance(settings,dict) or set(settings)-set(DEFAULTS):
        raise ValueError('Unknown acid incubation settings')
    result=default_settings();result.update(deepcopy(settings))
    if result['RUN_STUDENT'] or result['RUN_LATENT_CORROBORATION']:
        raise ValueError('The acid calibration workflow currently supports the notebook model-free route; optional student/latent stages are not enabled.')
    enums={'LINE_NOISE_METHOD':('sinusoid','notch','none'),
        'GATE_STATISTIC':('peak','sustained'),'SUSTAINED_GATE_MODE':('timescale','absolute','sigma'),
        'AMPLITUDE_STATISTIC':('peak','peak_corrected','sustained','q80','half_peak'),
        'BASELINE_MODE':('global','local','block'),'GATE_BASELINE_MODE':('global','local','block'),
        'PRIMARY_SET_NAME':('cusum','cusum_matched','threshold','cusum_resistive','cusum_resistive_matched')}
    for key,choices in enums.items():
        if result[key] not in choices: raise ValueError(f'{key}: choose one of {choices}')
    # PRIMARY_DETECTOR is never GUI-exposed and never enum-validated (unlike
    # PRIMARY_SET_NAME above) -- it exists only to index detector_catalogs in the math
    # kernel, so it must always name the SAME underlying detector as PRIMARY_SET_NAME,
    # never an independently-configured value that can silently drift from it (e.g.
    # PRIMARY_SET_NAME='cusum_resistive' with a stale PRIMARY_DETECTOR='cusum' would
    # gate-report and drift-summarize the wrong, opposite-polarity catalog).
    result['PRIMARY_DETECTOR']=result['PRIMARY_SET_NAME'].removesuffix('_matched')
    for key,default in DEFAULTS.items():
        value=result[key]
        if key=='MATCH_TAU_MS' and value is None:continue
        if isinstance(default,bool) and not isinstance(value,bool):raise ValueError(f'{key} must be boolean')
        if isinstance(default,(int,float)) and not isinstance(default,bool):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError(f'{key} must be finite')
            if isinstance(default,int) and (int(value)!=value):raise ValueError(f'{key} must be an integer')
            if isinstance(default,int):result[key]=int(value)
            if default>0 and value<=0:raise ValueError(f'{key} must be positive')
    if result['SEED']<0 or result['ABF_CHANNEL']<0:raise ValueError('Seed and channel cannot be negative')
    if result['MATCH_TAU_MS'] is not None and (not isinstance(result['MATCH_TAU_MS'],(int,float)) or result['MATCH_TAU_MS']<=0):raise ValueError('MATCH_TAU_MS must be positive or null')
    for key in ('CUSUM_PULSE_DELTA_GRID','CUSUM_PULSE_H_GRID','CUSUM_LEVEL_DELTA_GRID','CUSUM_LEVEL_H_GRID',
                'CUSUM_INJECT_DEPTHS_PA','CUSUM_INJECT_DWELLS_MS','NOTCH_FREQS'):
        values=result[key]
        if not isinstance(values,(tuple,list)) or not values or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or v<=0 for v in values):
            raise ValueError(f'{key} needs finite positive values')
    for key in ('CUSUM_PULSE_OVERRIDE','CUSUM_LEVEL_OVERRIDE'):
        value=result[key]
        if value is not None and (not isinstance(value,(tuple,list)) or len(value)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in value)):
            raise ValueError(f'{key} must be null or two finite positive numbers')
    for key in ('FAMILY_MIN_WEIGHT','FAMILY_FLAT_FRACTION','FAMILY_DEEP_FRAC','SUSTAINED_GATE_NOISE_QUANTILE',
                'GMM_MIN_WEIGHT','GMM_MIN_BLOCK_PRESENCE','MIN_POSTERIOR_FOR_LABEL','STUDENT_ENTER_THRESHOLD','STUDENT_EXIT_THRESHOLD'):
        if not 0<result[key]<=1:raise ValueError(f'{key} must be in (0,1]')
    return result


def inspect_source(path):
    path=Path(path).expanduser().resolve()
    info=inspect_raw(path)
    info['sha256']=_file_hash(path)
    return info


def new_state(name='Acid insulin incubation',records=(),settings=None):
    cfg=validate_settings(settings or {})
    rows=[];seen=set()
    for i,record in enumerate(records):
        if isinstance(record,(str,Path)):record={'source_path':record}
        row=dict(record);path=Path(row.get('source_path',row.get('path',''))).expanduser().resolve()
        tag=str(row.get('tag',f'acid_recording_{i+1}'))
        if not re.fullmatch('[A-Za-z0-9_-]+',tag) or tag in seen:raise ValueError('Recording tags must be unique letters, numbers, underscore or hyphen')
        seen.add(tag)
        info=row.get('source_info') or inspect_source(path)
        hours=row.get('incubation_h')
        if hours is None:
            match=re.search(r'(\d+)h',tag);hours=float(match.group(1)) if match else 0.
        if not isinstance(hours,(int,float)) or not np.isfinite(hours) or hours<0:raise ValueError('Incubation hours must be finite and nonnegative')
        row.update(source_path=path,source_info=info,tag=tag,incubation_h=float(hours),
            recording_id=row.get('recording_id',uuid.uuid4().hex),catalog=None,families=None,
            pore_id=row.get('pore_id',tag),voltage_mv=row.get('voltage_mv',-300.),concentration_pm=row.get('concentration_pm',100.))
        rows.append(row)
    if len({str(r['source_path']) for r in rows})!=len(rows):raise ValueError('Do not add the same acquisition twice')
    return dict(schema=SCHEMA,name=str(name),records=rows,settings=cfg,figures=[],figure_styles={},
        analysis=None,history=[],created_at=_stamp(),review_policy='Automatic detector gate; no human review asserted')


def _copy(state):
    if state.get('schema')!=SCHEMA:raise ValueError('Not an acid incubation workspace')
    result=dict(state);result['records']=[dict(r) for r in state['records']]
    result['history']=list(state.get('history',[]));result['figures']=list(state.get('figures',[]))
    result['figure_styles']=deepcopy(state.get('figure_styles',{}))
    return result


def _record(state,tag):
    rows=[r for r in state['records'] if r['tag']==tag]
    if len(rows)!=1:raise ValueError('Select an existing recording tag')
    return rows[0]


def _history(state,action,**details):
    state['history'].append(dict(at=_stamp(),action=action,**details))


def _retire(state,stage='analysis'):
    state['analysis']=None;state['figures']=[]
    for row in state['records']:
        row.pop('analysis',None)
        if stage in ('prepare','preparation','sources'):
            row.pop('prepared',None)
        if stage in ('prepare','preparation','sources','detect','detection'):
            row['catalog']=None;row['families']=None;row.pop('detection',None);row.pop('review_receipt',None)


def _invalidate_families(state):
    for key in ('family_model_path','family_model_sha256','family_basis'):state.pop(key,None)
    for record in state['records']:
        record['families']=None
        record.pop('family_summary',None);record.pop('family_figures',None)


def update_settings(state,settings,stage='analysis'):
    updated=_copy(state);new=validate_settings(settings)
    changed={k for k in new if new[k]!=state['settings'].get(k)}
    updated['settings']=new
    if changed:
        analysis_prefixes=('FAMILY_','BURST_','GMM_','HDBSCAN_','MORPH_','EXT_','OVERLAY_')
        analysis_keys={'N_TIME_BLOCKS','N_BLOCK_BOOTSTRAPS','MIN_ANALYSIS_EVENTS','MIN_POSTERIOR_FOR_LABEL',
            'PHYSICAL_FEATURES','RUN_MORPHOLOGY','RUN_LATENT_CORROBORATION','FIG_DPI','FIG_MIN_N','AMP_VAR'}
        effective=('prepare' if changed & PREPARE_KEYS else 'analysis' if all(k in analysis_keys or k.startswith(analysis_prefixes) for k in changed) else 'detect')
        _retire(updated,effective)
        _invalidate_families(updated)
        _history(updated,'Change acid settings',keys=sorted(changed),invalidated=effective)
    return updated


def _selection_hash(row):
    frame=row.get('catalog')
    if not isinstance(frame,pd.DataFrame):raise ValueError('Detect events first')
    return _digest(dict(tag=row['tag'],source=row['source_info'].get('sha256'),
        detection=row.get('detection',{}).get('settings_sha256'),
        rows=frame[['event_id','start_idx','end_idx','accepted']].to_dict('records')))


def confirm_selection(state,tag,method='user confirmed'):
    updated=_copy(state);row=_record(updated,tag)
    row['review_receipt']=dict(fingerprint=_selection_hash(row),method=method,at=_stamp(),
        human_review=(method=='user confirmed'),all_events_personally_inspected=False)
    _history(updated,'Confirm selected events',tag=tag,method=method)
    return updated


def selection_is_current(row):
    try:return row.get('review_receipt',{}).get('fingerprint')==_selection_hash(row)
    except (ValueError,KeyError,TypeError):return False


def set_event_acceptance(state,tag,event_ids,accepted):
    if not isinstance(accepted,(bool,np.bool_)):raise ValueError('Acceptance must be boolean')
    updated=_copy(state);row=_record(updated,tag);frame=row.get('catalog')
    if not isinstance(frame,pd.DataFrame):raise ValueError('Detect events first')
    ids=set(event_ids)
    if not ids or not ids<=set(frame.event_id):raise ValueError('Choose existing event IDs')
    frame=frame.copy(deep=True);selection=frame.event_id.isin(ids)
    frame.loc[selection,'accepted']=bool(accepted);frame.loc[selection,'event_review']='reviewed'
    row['catalog']=frame;row.pop('review_receipt',None)
    _retire(updated,'analysis')
    _history(updated,'Review individual events',tag=tag,event_ids=sorted(ids),accepted=bool(accepted))
    return updated


def set_family_acceptance(state,tag,family_ids,accepted):
    row=_record(state,tag);families=row.get('families')
    if not isinstance(families,pd.DataFrame):raise ValueError('Run the family analysis before reviewing families')
    ids=set(int(v) for v in family_ids)
    if not ids or not ids<=set(families.family):raise ValueError('Select existing family IDs')
    events=families.loc[families.family.isin(ids),'event_id'].tolist()
    updated=set_event_acceptance(state,tag,events,accepted)
    _history(updated,'Review complete families',tag=tag,families=sorted(ids),accepted=bool(accepted))
    return updated


def _stage_dir(folder,name):
    root=Path(folder).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<600*1024**2:raise ValueError('Less than 600 MiB free; free disk space before this study stage')
    path=root/'runs'/(name+'-'+uuid.uuid4().hex[:10]);path.mkdir(parents=True)
    return path


class AcidRuntime:
    """One numerical session with static function code and private stage data."""
    def __init__(self,settings,folder,record,allrecords=(),*,progress=None,cancel=None):
        import os,tempfile
        os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'nanoscope-numba-cache'))
        from . import acid_incubation_math as math_source
        cfg=validate_settings(settings)
        self.record=record;self.folder=Path(folder);self.progress=progress;self.cancel=cancel;self.figures=[]
        self.folder.mkdir(parents=True,exist_ok=True)
        self.namespace={**math_source.__dict__,**deepcopy(cfg)}
        ns=self.namespace
        self.plot=_AcidPlot(self._figure_saved);ns['plt']=self.plot;ns['_acid_joint_grid']=self.plot.joint_grid
        for name,value in list(ns.items()):
            if isinstance(value,types.FunctionType) and value.__module__==math_source.__name__:
                rebound=types.FunctionType(value.__code__,ns,name,value.__defaults__,value.__closure__)
                rebound.__kwdefaults__=value.__kwdefaults__;rebound.__annotations__=value.__annotations__
                ns[name]=rebound
        rows=list(allrecords) or [record]
        arms={r['tag']:(f"{r.get('incubation_h',0):g} h" + (f" ({r['nominal_pore_nm']:g} nm nominal)" if r.get('nominal_pore_nm') is not None and not r.get('pore_uncertain') else '')) for r in rows}
        series=[[arms[r['tag']],[r['tag']]] for r in sorted(rows,key=lambda x:x.get('incubation_h',0))]
        voltages=sorted(set(float(r.get('voltage_mv',-300)) for r in rows))
        voltage_text=f'{voltages[0]:g} mV' if len(voltages)==1 else 'mixed voltages; descriptive comparison'
        ns.update(_acid_cancel=cancel,TAG=record['tag'],TAG_ARMS=arms,TAGS_TO_RUN=[r['tag'] for r in rows],
            ABF_FILES={record['tag']:str(Path(record['source_path']).resolve())},ABF_DIR=Path('/'),
            FILE_CHANNELS={record['tag']:int(record.get('channel',cfg['ABF_CHANNEL']))},
            BASE_DIR=self.folder,DATA_DIR=self.folder/'prepared',OUT_DIR=self.folder/(record['tag']+'_subpopulation'),
            FAMILY_MODEL_PATH=self.folder/'event_family_model.json',STUDENT_MODEL_PATH=self.folder/'disabled-student.keras',
            PREPROCESS='force',REUSE_PROBABILITIES=False,SAVE_FIGURES=True,SERIES_RERUN_INLINE=False,NOTEBOOK_PATH=None,
            FIGURE_GROUPS={'Acid incubation series at '+voltage_text:series},_acid_figure_callback=self._finish_figure,
            _acid_hours={r['tag']:float(r.get('incubation_h',0)) for r in rows},
            finish_figure=self._finish_figure)
        same=[r for r in rows if r.get('nominal_pore_nm')==18 and not r.get('pore_uncertain')]
        if len(same)>1:ns['FIGURE_GROUPS']['Nominal 18 nm pores']=[[arms[r['tag']],[r['tag']]] for r in same]
        ns['CACHE_DIR']=ns['OUT_DIR']/'cache';ns['FIG_DIR']=ns['OUT_DIR']/'figures'
        for key in ('DATA_DIR','OUT_DIR','CACHE_DIR','FIG_DIR'):ns[key].mkdir(parents=True,exist_ok=True)
        ns['CSV_PATH']=ns['DATA_DIR']/(record['tag']+'.csv.gz');ns['META_PATH']=ns['DATA_DIR']/(record['tag']+'_metadata.json')
        ns['plt'].rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':17,'axes.titleweight':'bold',
            'axes.labelsize':13,'axes.labelweight':'bold','figure.titlesize':18,'figure.titleweight':'bold'})

    def _finish_figure(self,name):
        check_cancelled(self.cancel)
        fig=self.plot.gcf();fig.tight_layout()
        path=self.namespace['FIG_DIR']/(str(name)+'.png')
        fig.savefig(path,dpi=int(self.namespace['FIG_DPI']),bbox_inches='tight')
        self.plot.close(fig);check_cancelled(self.cancel)

    def _figure_saved(self,fig,path):
        from ..plot.acid_figure_recipe import capture_figure
        capture_figure(fig,path)
        if path not in self.figures:self.figures.append(path)

    def run(self,index):
        check_cancelled(self.cancel)
        if self.progress:self.progress(0.,f"{self.record['tag']}: acid stage {index}")
        self.namespace[f'stage_{index:03d}']()
        self.namespace['finish_figure']=self._finish_figure
        check_cancelled(self.cancel)

    def definitions(self):
        for name in sorted(self.namespace):
            if name.startswith('definitions_'):self.namespace[name]()
        self.namespace['finish_figure']=self._finish_figure


def _load_prepared(runtime,prepared):
    ns=runtime.namespace
    ns['definitions_007']()
    if prepared.get('sha256')!=_file_hash(prepared['parquet_path'],runtime.cancel):raise ValueError('Prepared acid data changed; prepare again')
    detr,raw,meta=ns['load_stage0'](runtime.record['tag'],Path(prepared['parquet_path']).parent)
    orientation=int(meta.get('event_polarity',1))
    if orientation not in (-1,1):raise ValueError('Prepared event polarity must be -1 or +1')
    if orientation==-1:detr=-detr;raw=-raw
    ns.update(signal=detr,raw=raw,meta=meta,fs=float(meta['fs_hz']),n_samples=len(detr),duration_s=len(detr)/float(meta['fs_hz']),
        polarity=int(meta.get('event_polarity',1)),metadata_i0=abs(float(meta.get('raw_mean_pA',np.nan))),raw_source=meta.get('raw_storage','current_raw_pA'),
        student_probability=np.zeros(len(detr),np.float32),
        INPUT_FINGERPRINTS={'abf':{'path':str(runtime.record['source_path']),'sha256':runtime.record['source_info'].get('sha256')},
            'data':{'path':str(prepared['parquet_path']),'sha256':prepared.get('sha256')},'model':None})


def prepare_record(state,tag,folder,progress=None,cancel=None):
    updated=_copy(state);row=_record(updated,tag);path=_stage_dir(folder,'prepare-'+tag)
    try:
        channel=row.get('channel',state['settings']['ABF_CHANNEL'])
        channels=row['source_info']['channels']
        if isinstance(channel,bool) or not isinstance(channel,int) or not 0<=channel<len(channels):raise ValueError('Select an existing current channel')
        if channels[channel]['units']!='pA':raise ValueError('The selected acid channel must declare current in pA')
        before=_file_hash(row['source_path'],cancel)
        if row['source_info'].get('sha256') not in (None,before):raise ValueError('Raw acid source changed since it was added; add the correct source again')
        with _math_session(state['settings']['SEED']):
            rt=AcidRuntime(state['settings'],path,row,state['records'],progress=progress,cancel=cancel)
            rt.run(7);rt.run(8)
        after=_file_hash(row['source_path'],cancel)
        if after!=before:raise ValueError('Raw acid source changed during preparation')
        data=rt.namespace['DATA_DIR'];pq=data/(tag+'.parquet');meta=rt.namespace['meta']
        row['prepared']=dict(parquet_path=pq,metadata_path=data/(tag+'_metadata.json'),
            baseline_path=data/(tag+'_baseline.parquet'),metadata=meta,sha256=_file_hash(pq,cancel),
            settings_sha256=_digest({k:state['settings'][k] for k in PREPARE_KEYS}),
            source_sha256=after,figures=rt.figures)
        row['catalog']=None;row['families']=None;row.pop('detection',None);row.pop('review_receipt',None)
        _retire(updated,'analysis');_invalidate_families(updated);_history(updated,'Prepare raw acid acquisition',tag=tag,source_sha256=row['prepared']['source_sha256'])
        return updated
    except BaseException:
        shutil.rmtree(path,ignore_errors=True);raise


def _portable(value,depth=0):
    if depth>15:raise TypeError('nested runtime temporary')
    if value is None or isinstance(value,(bool,int,float,str,Path,np.generic,pd.DataFrame,pd.Series)):return value
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise TypeError('object array')
        return value
    if isinstance(value,(list,tuple)):
        return type(value)(_portable(v,depth+1) for v in value)
    if isinstance(value,dict):return {k:_portable(v,depth+1) for k,v in value.items()}
    raise TypeError(type(value).__name__)


_CONTEXT_EXCLUDE={'signal','raw','analysis_signal','student_probability','global_quiet_mask','union_event_mask',
    'event_mask_by_polarity','_busy','_quiet_concat','_surrogate_segments','_injected_segments','_calibration_signals',
    'X','x','y','t','fig','ax','axes','im','_t','_ts','_rng_test','_t2','_table_t','_regions_t',
    '_seg_clean','_seg_injected','_seg_surrogate','_seg_inject_mask','_quiet_block_sigmas','injected_all','pooled','seg',
    '_res_threshold_mask','_cond_mask','_thr_mask','_cusum_mask','cusum_grid_raw','_runs'}


def _save_context(ns,path):
    context={}
    for key,value in ns.items():
        if key.startswith('__') or key in _CONTEXT_EXCLUDE or key.startswith(('stage_','definitions_')):continue
        if (key.isupper() and key not in {'PRIMARY_SET_NAME','AMP_BASE','AMP_PA_COL','AMP_COL','AMP_SIGMA_COL','AMP_FRAC_COL','AMP_LABEL','SUSTAINED_GATE_PA','CUSUM_DELTA','CUSUM_H','CUSUM_PULSE_DELTA','CUSUM_PULSE_H','CUSUM_SCALES'}) or key.endswith('_DIR') or key.endswith('_PATH'):continue
        try:
            if key=='CUSUM_SCALES':
                value={name:{k:v for k,v in config.items() if not callable(v)} for name,config in value.items()}
            converted=_portable(value)
            if isinstance(converted,np.ndarray) and converted.nbytes>32*1024**2:continue
            context[key]=converted
        except (TypeError,ValueError):pass
    checkpoint.save_workspace(context,path,expected_revision=None)
    return path


def _restore_context(rt,row):
    ns=rt.namespace
    _load_prepared(rt,row['prepared'])
    context=checkpoint.load_workspace(row['detection']['context_path'])
    ns.update(context)
    # Paths/settings always belong to this new transaction, never the saved run.
    ns['TAG']=row['tag'];ns['_acid_cancel']=rt.cancel
    ns['PRIMARY_SET_NAME']=row['detection']['primary_set']
    rt.definitions()
    if 'CUSUM_SCALES' not in ns:
        # Calibration's cell-mask lambdas are transient; a reopened analysis
        # only needs these immutable scale settings for report provenance.
        ns['CUSUM_SCALES']={name:{'block_ms':ns[block_key],'pass1_block_ms':ns['CUSUM_PASS1_BLOCK_MS']}
            for name,block_key in (('pulse','CUSUM_PULSE_BLOCK_MS'),('level','CUSUM_BLOCK_MS'))
            if name in ns['cusum_selected']}
    ns['analysis_signal']=ns['signal'] if ns.get('primary_polarity',1)==1 else -ns['signal']
    ns['global_quiet_mask']=ns['robust_quiet_stats'](ns['signal'])[2]
    ns['union_event_mask']=ns['regions_to_mask'](ns['n_samples'],sum((ns.get(k,[]) for k in ('student_candidate_regions','threshold_candidate_regions','cusum_candidate_regions','threshold_candidate_regions_res','cusum_candidate_regions_res')),[]))
    ns['event_mask_by_polarity']={1:ns['regions_to_mask'](ns['n_samples'],sum((ns.get(k,[]) for k in ('student_candidate_regions','threshold_candidate_regions','cusum_candidate_regions')),[])),
        -1:ns['regions_to_mask'](ns['n_samples'],ns.get('threshold_candidate_regions_res',[])+ns.get('cusum_candidate_regions_res',[]))}
    ns['primary_catalog']=row['catalog'].loc[row['catalog'].accepted].copy().reset_index(drop=True)
    detector=row['detection'].get('catalog_detector','cusum')
    diagnostic_catalog=row['catalog'].copy(deep=True)
    diagnostic_catalog['selected']=diagnostic_catalog.accepted.astype(bool)
    if 'passes_gate' in diagnostic_catalog:
        diagnostic_catalog['accepted']=diagnostic_catalog.passes_gate.astype(bool)
    ns['detector_catalogs'][detector]=diagnostic_catalog
    ns[detector+'_catalog']=diagnostic_catalog.copy(deep=True)
    ns['analysis_sets'][ns['PRIMARY_SET_NAME']]=ns['primary_catalog'].copy(deep=True)


def detect_record(state,tag,folder,progress=None,cancel=None):
    updated=_copy(state);row=_record(updated,tag)
    if not row.get('prepared'):raise ValueError('Prepare this recording first')
    expected=_digest({k:state['settings'][k] for k in PREPARE_KEYS})
    if row['prepared']['settings_sha256']!=expected:raise ValueError('Preparation settings changed; prepare this recording again')
    path=_stage_dir(folder,'detect-'+tag)
    try:
        with _math_session(state['settings']['SEED']):
            rt=AcidRuntime(state['settings'],path,row,state['records'],progress=progress,cancel=cancel)
            _load_prepared(rt,row['prepared'])
            for i,index in enumerate(DETECTION_STAGES):
                if progress:progress(i/len(DETECTION_STAGES),f'{tag}: detection / calibration stage {index}')
                rt.run(index)
            ns=rt.namespace
            detector=ns['PRIMARY_SET_NAME'].removesuffix('_matched')
            if detector not in ns['detector_catalogs']:raise ValueError('Chosen primary detector did not produce a candidate catalog')
            catalog=ns['detector_catalogs'][detector].copy(deep=True)
            catalog['passes_gate']=catalog.accepted.astype(bool)
            chosen=set(ns['primary_catalog'].event_id)
            catalog['automatic_accepted']=catalog.event_id.isin(chosen)
            catalog['accepted']=catalog.automatic_accepted
            catalog['event_review']='unreviewed'
            if catalog.event_id.duplicated().any():raise ValueError('Duplicate acid event identity')
            if len(catalog) and not ((catalog.start_idx>=0)&(catalog.end_idx>=catalog.start_idx)&(catalog.end_idx<ns['n_samples'])).all():raise ValueError('Invalid acid event bounds')
            row['catalog']=catalog;row['families']=None;row.pop('review_receipt',None)
            context_path=_save_context(ns,path/'context')
            row['detection']=dict(context_path=context_path,output_folder=ns['OUT_DIR'],figures=rt.figures,
                settings_sha256=_digest(state['settings']),primary_set=ns['PRIMARY_SET_NAME'],
                catalog_detector=detector,
                candidates=len(catalog),accepted=int(catalog.accepted.sum()),
                selected_parameters=deepcopy(ns['cusum_selected']),calibration_ok=bool(ns['cusum_calibration_ok']),
                false_alarms_per_s=float(ns['cusum_false_alarm_rate']),false_alarms_gated_per_s=float(ns['cusum_false_alarm_rate_gated']),
                acceptance_method='Notebook calibrated peak gate; no human decisions',human_review=False)
        _retire(updated,'analysis');_invalidate_families(updated);_history(updated,'Detect and characterize acid events',tag=tag,candidates=len(catalog),accepted=int(catalog.accepted.sum()))
        return updated
    except BaseException:
        shutil.rmtree(path,ignore_errors=True);raise


def build_families(state,folder,progress=None,cancel=None):
    """Freeze reference families for review before full downstream analysis.

    Acceptance remains unchanged. Existing family labels and the fitted model
    survive rejection/reacceptance so review never silently redefines a family.
    """
    updated=_copy(state);rows=[r for r in updated['records'] if not r.get('comparison_only')];cfg=state['settings']
    if not rows:raise ValueError('Add primary recordings first')
    reference=cfg['FAMILY_REFERENCE_TAG']
    if reference not in {r['tag'] for r in rows}:raise ValueError('Include the selected reference acquisition')
    for row in rows:
        if not row.get('detection') or not selection_is_current(row):raise ValueError(f"Detect and confirm selected events for {row['tag']} first")
        if not bool(row['catalog'].accepted.any()):raise ValueError(f"{row['tag']} has no selected events for family assignment")
    path=_stage_dir(folder,'acid-families');previous=state.get('family_model_path')
    try:
        with _math_session(cfg['SEED']):
            if previous and Path(previous).is_file():
                if state.get('family_model_sha256')!=_file_hash(previous,cancel):raise ValueError('Saved family model changed')
                shutil.copyfile(previous,path/Path(previous).name)
            for i,row in enumerate(sorted(rows,key=lambda r:(r['tag']!=reference,-r.get('incubation_h',0)))):
                if progress:progress(i/len(rows),f"Build review families: {row['tag']}")
                effective_cfg={**cfg,'FAMILY_REFIT_REFERENCE':False} if previous else cfg
                rt=AcidRuntime(effective_cfg,path,row,rows,progress=progress,cancel=cancel)
                _restore_context(rt,row);rt.run(26);rt.run(32)
                if row['tag']==reference:model_path=rt.namespace['FAMILY_MODEL_PATH']
                current=rt.namespace['_fam'][['event_id','family','family_posterior','in_burst']].copy(deep=True)
                old=row.get('families')
                if previous and isinstance(old,pd.DataFrame):current=pd.concat([old.loc[~old.event_id.isin(current.event_id)],current],ignore_index=True)
                row['families']=current
                row['family_summary']=deepcopy(rt.namespace['event_families'])
                row['family_figures']=list(rt.figures)
                del rt
        updated['family_model_path']=model_path
        updated['family_model_sha256']=_file_hash(updated['family_model_path'],cancel)
        updated['family_basis']={r['tag']:_selection_hash(r) for r in rows}
        _history(updated,'Build frozen families for review',reference_tag=reference,model_sha256=updated['family_model_sha256'])
        return updated
    except BaseException:
        shutil.rmtree(path,ignore_errors=True);raise


def analyze_study(state,folder,progress=None,cancel=None):
    updated=_copy(state);rows=[r for r in updated['records'] if not r.get('comparison_only')];cfg=state['settings']
    if not rows:raise ValueError('Add recordings first')
    for row in rows:
        if not row.get('detection'):raise ValueError(f"Detect {row['tag']} first")
        if not selection_is_current(row):raise ValueError(f"Confirm the selected events for {row['tag']} first")
        if not bool(row['catalog'].accepted.any()):raise ValueError(f"{row['tag']} has no selected events; accept events or exclude the recording from the primary series")
    reference=cfg['FAMILY_REFERENCE_TAG']
    if reference not in {r['tag'] for r in rows}:raise ValueError('The selected reference recording must be included')
    order=sorted(rows,key=lambda r:(r['tag']!=reference,-r.get('incubation_h',0)))
    path=_stage_dir(folder,'acid-analysis');all_figures=[];summaries=[]
    try:
        with _math_session(cfg['SEED']):
            previous_model=state.get('family_model_path')
            if previous_model and Path(previous_model).is_file():
                if state.get('family_model_sha256')!=_file_hash(previous_model,cancel):raise ValueError('Saved reference-family model changed')
                shutil.copyfile(previous_model,path/Path(previous_model).name)
            for pos,row in enumerate(order):
                effective_cfg={**cfg,'FAMILY_REFIT_REFERENCE':False} if previous_model else cfg
                rt=AcidRuntime(effective_cfg,path,row,rows,progress=progress,cancel=cancel)
                _restore_context(rt,row)
                # Cross-recording loaders read this transaction's selected catalog.
                # Publish the curated selection before family/comparison stages.
                rt.run(26)
                for index in CORE_ANALYSIS_STAGES:
                    if progress:progress(pos/len(order),f"{row['tag']}: family / feature stage {index}")
                    rt.run(index)
                ns=rt.namespace
                if row['tag']==reference:model_path=ns['FAMILY_MODEL_PATH']
                current_members=ns['_fam'][['event_id','family','family_posterior','in_burst']].copy(deep=True)
                old_members=row.get('families')
                if previous_model and isinstance(old_members,pd.DataFrame):
                    current_members=pd.concat([old_members.loc[~old_members.event_id.isin(current_members.event_id)],current_members],ignore_index=True)
                row['families']=current_members
                thin=len(ns['primary_catalog'])<cfg['MIN_ANALYSIS_EVENTS']
                if not thin:
                    for index in SUBSTRUCTURE_STAGES:rt.run(index)
                else:
                    ns['screening_outcome']=f"Substructure not run: {len(ns['primary_catalog'])} selected events, below {cfg['MIN_ANALYSIS_EVENTS']} minimum."
                # Each record has its own family distributions. Cross-record
                # figure groups are rendered once, after every selected record
                # has published its curated event/family tables.
                figure_groups=ns['FIGURE_GROUPS'];ns['FIGURE_GROUPS']={}
                try:rt.run(36)
                finally:ns['FIGURE_GROUPS']=figure_groups
                output=ns['OUT_DIR'];output.mkdir(exist_ok=True)
                row['catalog'].to_csv(output/(row['tag']+'_all_candidates.csv'),index=False)
                row['families'].to_csv(output/(row['tag']+'_family_memberships.csv'),index=False)
                summary=dict(tag=row['tag'],incubation_h=row['incubation_h'],n_candidates=len(row['catalog']),
                    n_accepted=int(row['catalog'].accepted.sum()),duration_s=ns['duration_s'],rate_per_s=len(ns['primary_catalog'])/ns['duration_s'],
                    primary_set=ns['PRIMARY_SET_NAME'],families=ns['event_families'],
                    screening_outcome=ns.get('screening_outcome'),extended_outcome=ns.get('extended_outcome'),
                    calibration=row['detection'],thin_substructure=thin,
                    pore_note=row.get('pore_note','Separate physical pore; descriptive comparison'))
                (output/'nanoscope_summary.json').write_text(json.dumps(summary,indent=2,default=_plain))
                row['analysis']=dict(folder=output,summary=summary,selection_sha256=_selection_hash(row))
                summaries.append(summary);all_figures.extend(rt.figures)
                # Save the lightweight namespace context needed for final cohort plots.
                if row['tag']==order[-1]['tag']:final=rt
                else:
                    del rt,ns
                    import gc;gc.collect()
            for index in (36,38,76):final.run(index)
            all_figures.extend(p for p in final.figures if p not in all_figures)
        for row in rows:
            all_figures.extend(row.get('prepared',{}).get('figures',[]))
            all_figures.extend(row.get('detection',{}).get('figures',[]))
        updated['figures']=list(dict.fromkeys(all_figures))
        updated['analysis']=dict(folder=path,summaries=summaries,reference_tag=reference,
            family_model_path=model_path,settings_sha256=_digest(cfg),
            selected_event_fingerprints={r['tag']:_selection_hash(r) for r in rows},
            cohort=[r['tag'] for r in rows],fresh_execution=True)
        updated['family_model_path']=model_path
        updated['family_model_sha256']=_file_hash(updated['family_model_path'],cancel)
        _history(updated,'Analyze selected acid events',reference_tag=reference,figure_count=len(updated['figures']))
        return updated
    except BaseException:
        shutil.rmtree(path,ignore_errors=True);raise


def figure_style_key(state,path):
    """Stable per-record identity survives copied output folders and reopen."""
    path=Path(path)
    explicit=state.get('figure_keys',{}).get(path)
    if explicit is None: explicit=state.get('figure_keys',{}).get(str(path))
    if explicit: return str(explicit)
    for row in state.get('records',[]):
        tag=row['tag']
        if any(parent.name==tag+'_subpopulation' for parent in path.parents):
            return tag+'::'+path.stem
    return 'study::'+path.stem


def figure_source_path(state,path):
    """Copied presentation artifacts refer back to their immutable recipe.

    A newly analyzed figure is its own source even when an older run had the
    same semantic key; curation must never reuse the old scientific artists.
    """
    path=Path(path);mapped=state.get('figure_keys',{}).get(path)
    if mapped is None:mapped=state.get('figure_keys',{}).get(str(path))
    return Path(state.get('figure_sources',{}).get(mapped,path)) if mapped else path


def render_figures(state,folder,progress=None,cancel=None):
    if not state.get('analysis'):raise ValueError('Run the selected-event analysis first')
    cohort=set(state['analysis'].get('cohort',state['analysis']['selected_event_fingerprints']))
    for row in state['records']:
        if row['tag'] in cohort and state['analysis']['selected_event_fingerprints'].get(row['tag'])!=_selection_hash(row):
            raise ValueError('Event selection changed; rerun analysis')
    updated=_copy(state);path=_stage_dir(folder,'acid-figures');made=[]
    updated['figure_keys']=dict(state.get('figure_keys',{}))
    updated['figure_sources']=dict(state.get('figure_sources',{}))
    try:
        from ..plot.acid_figure_recipe import capture_figure
        for i,original in enumerate(state['figures']):
            check_cancelled(cancel)
            if progress:progress(i/max(1,len(state['figures'])),f'Render {Path(original).stem}')
            fig=None
            try:
                fig,data,_=reconstruct_figure(state,original,cancel=cancel)
                target=path/str(i+1).zfill(3)/Path(original).name
                target.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(target,dpi=int(state['settings']['FIG_DPI']),bbox_inches='tight')
                capture_figure(fig,target,data);made.append(target)
                key=figure_style_key(state,original)
                updated['figure_keys'][target]=key
                updated['figure_sources'][key]=figure_source_path(state,original)
            finally:
                if fig is not None:fig.clear()
        updated['figures']=made;_history(updated,'Render acid figures',count=len(made))
        return updated
    except BaseException:
        shutil.rmtree(path,ignore_errors=True);raise


def read_trace(state,tag,*,raw=False,start_idx=0,stop_idx=None,max_points=10000):
    row=_record(state,tag)
    if not row.get('prepared'):raise ValueError('Prepare this recording first')
    meta=row['prepared']['metadata'];fs=float(meta['fs_hz']);n=int(meta['n_samples'])
    start=max(0,int(start_idx));stop=n if stop_idx is None else min(n,int(stop_idx))
    if stop<start:raise ValueError('Invalid trace range')
    stride=max(1,int(math.ceil((stop-start)/max_points))) if max_points else 1
    indices=np.arange(start,stop,stride,dtype=np.int64);values=np.empty(len(indices),np.float32)
    import pyarrow.parquet as pq
    reader=pq.ParquetFile(row['prepared']['parquet_path'])
    column=('current_raw_code' if 'current_raw_code' in reader.schema.names else 'current_raw_pA') if raw else 'current_detr_pA'
    offset=0
    for group in range(reader.num_row_groups):
        end=offset+reader.metadata.row_group(group).num_rows
        lo,hi=np.searchsorted(indices,[offset,end])
        if hi>lo:
            data=reader.read_row_group(group,columns=[column]).column(0).to_numpy()
            values[lo:hi]=data[indices[lo:hi]-offset]
        offset=end
        if offset>=stop:break
    if column=='current_raw_code':values*=np.float32(meta['adc_step_pA'])
    return indices/fs,values


def reconstruct_figure(state,path,progress=None,cancel=None,*,apply_style=True):
    check_cancelled(cancel)
    from ..plot.acid_figure_recipe import reconstruct
    key=figure_style_key(state,path)
    original=figure_source_path(state,path)
    fig,data,resources=reconstruct(Path(original))
    style=state.get('figure_styles',{}).get(key)
    if apply_style and style:
        from ..plot.study_figure_style import StudyFigureStyle
        StudyFigureStyle(fig).apply(**style)
    return fig,data,resources
