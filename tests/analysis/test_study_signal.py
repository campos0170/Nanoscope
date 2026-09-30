from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from nanoscope.analysis.study_signal import (SourceOptions, SchmittSettings, NullSettings,
    retain_samples, source_indices, detect_bounds, thresholds, analyze, null_controls)
from nanoscope.analysis.incubation_detection import PreparedRecording
from nanoscope.analysis.cancel import OperationCancelled


def recording():
    rng=np.random.default_rng(113); x=rng.normal(0,1,10000).astype('f4')
    x[500:510]-=15; x[2000:2060]-=12; x[4000:4100]-=2
    return PreparedRecording('sample',np.full(len(x),-100.),x,10000.,1.,
        {'event_polarity':1,'qc_mad_noise_pA':1}, {'event_polarity':1,'metadata':{'event_polarity':1,'qc_mad_noise_pA':1}})


def test_selection_preserves_both_boundary_conventions():
    raw,ranges=retain_samples(np.arange(20),10.,SourceOptions(keep_start_s=.2,keep_stop_s=1.8,exclude_s='.4:.6;1:1.2'))
    assert raw.tolist()==[2,3,7,8,9,13,14,15,16,17]
    r=recording(); r.raw=raw; r.metadata={'retained_ranges':ranges}
    assert np.array_equal(source_indices(r,np.arange(len(raw))),raw)


def test_an_event_cannot_bridge_an_excluded_source_interval():
    x=np.zeros(100,dtype='f4');x[40:60]=-20
    cfg=SchmittSettings(smooth_samples=1,min_ms=.1,merge_ms=100)
    events=detect_bounds(x,1000.,cfg,(0.,1.,4.,1.),ranges=[[0,50,0,50],[50,100,100,150]])
    assert events[['start_idx','end_idx']].values.tolist()==[[40,49],[50,59]]


def test_gate_review_and_nulls_do_not_overwrite_each_other():
    r=recording(); cfg=SchmittSettings(min_ms=.2,plateau_ms=2)
    a=analyze(r,cfg); before=a.catalog.copy(deep=True)
    assert {'spike','plateau'} <= set(a.catalog.kind)
    assert np.array_equal(a.catalog.passes_gate,a.catalog.kind!='shallow_long')
    a.catalog.loc[:,'accepted']=False
    first=null_controls(a,NullSettings(global_seeds=2,block_seeds=2,block_s=.1))
    second=null_controls(a,NullSettings(global_seeds=2,block_seeds=2,block_s=.1))
    pd.testing.assert_frame_equal(first,second)
    assert not a.catalog.accepted.any()
    assert np.array_equal(a.catalog.passes_gate,before.passes_gate)


def test_cancel_before_fft_or_detection():
    r=recording()
    with pytest.raises(OperationCancelled): analyze(r,SchmittSettings(),cancel=lambda:True)
    a=analyze(r,SchmittSettings())
    with pytest.raises(OperationCancelled): null_controls(a,NullSettings(),cancel=lambda:True)


@pytest.mark.parametrize('kwargs',[{'enter_sigma':1,'exit_sigma':2},{'smooth_samples':2},
    {'min_ms':0},{'merge_ms':float('nan')},{'i0_window_s':0},{'enter_min_pa':-1}])
def test_invalid_detection_settings(kwargs):
    with pytest.raises(ValueError): SchmittSettings(**kwargs)


@pytest.mark.parametrize('kwargs',[{'add_channel':-1},{'keep_start_s':10,'keep_stop_s':2},
    {'exclude_s':'4:2'},{'exclude_s':'nan:10'},{'median_engine':'unknown'},{'keep_start_s':True}])
def test_invalid_source_settings(kwargs):
    with pytest.raises(ValueError): SourceOptions(**kwargs)


def test_zero_noise_is_explicitly_rejected():
    r=recording(); r.detrended[:]=0
    with pytest.raises(ValueError,match='MAD noise'): analyze(r,SchmittSettings())
