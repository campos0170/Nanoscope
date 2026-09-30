"""Independent physical/coordinate invariants for the static acid port.

These tests use small synthetic arrays, no study data and no notebook execution.
They deliberately test externally meaningful properties rather than counts from
one saved notebook execution.
"""
import copy
import types
import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from nanoscope.analysis import acid_incubation_math as kernel
from nanoscope.analysis.acid_incubation_config import DEFAULTS


def scope():
    ns = vars(kernel).copy()
    ns.update(copy.deepcopy(DEFAULTS))
    ns['_acid_cancel'] = None
    ns['finish_figure'] = lambda name: None
    return ns


def run(ns, cell):
    f = getattr(kernel, f'stage_{cell:03d}')
    return types.FunctionType(f.__code__, ns)()


def helper(ns, cell, name):
    """Load one already-compiled static helper without running its heavy stage."""
    stage = getattr(kernel, f'stage_{cell:03d}')
    code = next(c for c in stage.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == name)
    assert not code.co_freevars, (name, code.co_freevars)
    fn = types.FunctionType(code, ns)
    ns[name] = fn
    return fn


@pytest.fixture
def prepared():
    ns = scope(); run(ns, 7); run(ns, 10)
    return ns


def test_inclusive_bounds_and_last_sample(prepared):
    r=[(2,4),(9,9)]
    mask=prepared['regions_to_mask'](10,r)
    np.testing.assert_array_equal(np.flatnonzero(mask),[2,3,4,9])
    assert prepared['dwell_ms']((2,4),50_000)==pytest.approx(.06)
    assert prepared['dwell_ms']((9,9),50_000)==pytest.approx(.02)


def test_threshold_physical_sign_symmetry(prepared):
    x=np.zeros(200,dtype=np.float32);x[70:85]=-12
    kw=dict(fs=50_000,baseline=0,noise=1,enter_sigma=4,exit_sigma=1.5,merge_gap_ms=.05,min_event_ms=.05)
    down=prepared['threshold_regions'](x,**kw)
    up_when_oriented=prepared['threshold_regions'](-(-x),**kw)
    assert down==up_when_oriented==[(70,84)]
    assert prepared['threshold_regions'](-x,**kw)==[]


def test_median_baseline_rejects_short_excursion_preserves_depth(prepared):
    x=np.full(1001,100,dtype=np.float32);x[499:502]=88
    detr,b=prepared['baseline_rolling_median'](x,1000,.2,1)
    assert detr.dtype==np.float32 and b.dtype==np.float32
    np.testing.assert_array_equal(b,np.full(1001,100,dtype=np.float32))
    np.testing.assert_array_equal(detr[499:502],[-12,-12,-12])
    assert np.count_nonzero(detr)==3


def test_gate_returns_four_sigma_for_one_independent_sample():
    ns=scope();f=helper(ns,17,'gaussian_max_multiplier')
    assert float(f(1,norm.sf(4)))==pytest.approx(4,abs=1e-10)
    values=f(np.array([1,2,10,100,1000]),norm.sf(4))
    assert np.isfinite(values).all() and np.all(np.diff(values)>0)


def test_peak_floor_extrapolates_long_dwell_instead_of_clamping():
    ns=scope();ns.update(_pk_log_ms=np.log([.02,50]),_pk_log_sigma=np.log([1,2]),_pk_log_neff=np.log([1,25]),_gate_tail_p=norm.sf(4))
    for name in ['gaussian_max_multiplier','n_eff_at_timescale','peak_gate_multiplier','peak_floor_at_timescale']:helper(ns,17,name)
    assert ns['n_eff_at_timescale'](100)==pytest.approx(50)
    assert ns['peak_floor_at_timescale'](100)>ns['peak_floor_at_timescale'](50)
    assert ns['peak_floor_at_timescale'](.02)==pytest.approx(4)


def test_context_excludes_all_event_samples_and_expands_at_edge():
    ns=scope();f=helper(ns,20,'quiet_context_indices')
    mask=np.zeros(100,bool);mask[:5]=True;mask[7:9]=True
    idx,used,valid=f(0,4,mask,fs=1000,context_ms=2,max_context_ms=20,min_samples=10)
    assert valid and used>2 and len(idx)>=10
    assert np.all(idx>=5) and not mask[idx].any()


def test_global_measurement_does_not_substitute_for_local_gate(prepared):
    ns=prepared
    ns.update(fs=1000,n_samples=1000,duration_s=1,global_baseline=20,global_noise=1,global_i0=100,
              cusum_baseline_blocks=np.full(10,20.),cusum_info={'block_len':100})
    ns['peak_gate_multiplier']=lambda d:4.
    for name in ['quiet_context_indices','local_rate_hz','build_event_catalog']:helper(ns,20,name)
    signal=np.full(1000,10,dtype=np.float32);signal[500:505]=7
    mask=np.zeros(1000,bool);mask[500:505]=True
    cat=ns['build_event_catalog']([(500,504)],detector='cusum',signal=signal,raw=np.full(1000,-100,dtype=np.float32),
       probability=np.zeros(1000),event_mask=mask,fs=1000,global_base=20,block_baselines=np.full(10,20.))
    r=cat.iloc[0]
    assert r.n_samples==5 and r.dwell_ms==5 and r.offset_s==pytest.approx(.505)
    assert r.peak_depth_pA==13 and r.peak_depth_local_pA==3
    assert r.peak_fracchange==pytest.approx(.13)
    ns['acceptance_floor_pA']=lambda d,noise=None:np.full(np.asarray(d).shape,5.)
    gate=helper(ns,17,'passes_acceptance_gate')
    assert not bool(gate(r.sustained_depth_local_pA, r.half_peak_mean_depth_local_pA,r.half_peak_ms,r.dwell_ms,None,
                        peak_depth_pA=r.peak_depth_local_pA,local_noise_pA=r.local_noise_mad_pA))
    assert bool(gate(r.sustained_depth_pA,r.half_peak_mean_depth_pA,r.half_peak_ms,r.dwell_ms,None,
                    peak_depth_pA=r.peak_depth_pA,local_noise_pA=r.local_noise_mad_pA))


def test_resistive_event_survives_when_no_accepted_conductive_event(prepared):
    ns=prepared
    columns=['event_id','start_idx','end_idx','accepted','polarity']
    empty=pd.DataFrame(columns=columns).astype({'start_idx':'int64','end_idx':'int64','accepted':'bool','polarity':'int64'})
    res=pd.DataFrame([['R000000',100,109,True,-1]],columns=columns)
    rt=pd.DataFrame([['U000000',100,109,True,-1]],columns=columns)
    ns.update(cusum_catalog=empty.copy(),threshold_catalog=empty.copy(),student_catalog=empty.copy(),
              detector_catalogs={'cusum_resistive':res,'threshold_resistive':rt},fs=50_000,duration_s=.1,
              cusum_calibration_ok=True,signal=np.zeros(5000,dtype=np.float32),cusum_info={},cusum_info_res={})
    run(ns,25)
    assert len(ns['analysis_sets']['cusum_resistive'])==1
    assert not bool(ns['detector_catalogs']['cusum_resistive']['precursor_of_conductive'].iloc[0])
    assert np.isinf(ns['detector_catalogs']['cusum_resistive']['gap_to_next_conductive_ms'].iloc[0])


def test_record_namespaces_do_not_share_mutable_settings():
    a=scope();b=scope()
    a['CUSUM_PULSE_DELTA_GRID'].append(999)
    a['PRIMARY_SET_NAME']='cusum_matched'
    assert 999 not in b['CUSUM_PULSE_DELTA_GRID'] and b['PRIMARY_SET_NAME']=='cusum'


def test_zero_events_has_schema_for_downstream_detection_merge(prepared):
    ns=prepared
    ns.update(fs=1000,n_samples=100,duration_s=.1,global_baseline=0,global_noise=1,global_i0=100,
              cusum_baseline_blocks=np.zeros(1),cusum_info={'block_len':100})
    ns['peak_gate_multiplier']=lambda d:4.
    for name in ['quiet_context_indices','local_rate_hz','build_event_catalog']:helper(ns,20,name)
    cat=ns['build_event_catalog']([],detector='cusum',signal=np.zeros(100),raw=np.full(100,-100),
       probability=np.zeros(100),event_mask=np.zeros(100,bool),fs=1000,global_base=0,block_baselines=np.zeros(1))
    assert len(cat)==0
    required={'event_id','start_idx','end_idx','accepted','time_block','onset_s','offset_s','dwell_ms','peak_depth_pA','local_I0_pA'}
    assert required<=set(cat.columns), 'Zero detected events is a valid scientific result, not an unmergeable table'


def test_quantized_rolling_baseline_matches_explicit_reflected_window_medians(prepared):
    # Scientific definition, independent of SciPy's rank-filter implementation.
    # SciPy 1.15.1 has a confirmed 1D median regression (upstream #22333).
    x=(np.round(np.random.default_rng(681).normal(-290,4,5001)/.30517578125)*.30517578125).astype(np.float32)
    w=1001;r=w//2
    expected=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(x,(r,r),mode='symmetric'),w),axis=1).astype(np.float32)
    detr,baseline=prepared['baseline_rolling_median'](x,1000,1,1)
    np.testing.assert_array_equal(baseline,expected)
    np.testing.assert_array_equal(detr,x-expected)
