import numpy as np
import pandas as pd
import pytest
from nanoscope.analysis.modules import resolve_t_min, dwell_fit_inputs
from nanoscope.analysis.stats import fit_dwell_exponential


def test_cusum_floor_uses_detector_sample_gate_and_schmitt_keeps_duration_gate():
    fs = 50000.001192092925
    for detector, samples in [('cusum_sequential',50),('detect.cusum',None)]:
        prov={'detector':detector,'fs':fs,'min_dwell_s':.001}
        if samples:prov['min_dwell_samples']=samples
        floor,source,note=resolve_t_min(prov)
        assert floor == 50/fs and source=='min_dwell_samples/fs' and note
    assert resolve_t_min({'detector':'schmitt_hysteresis','fs':fs,'min_dwell_s':.001})[0]==.001


def test_csv_quantization_at_cusum_floor_does_not_require_editing_events():
    fs=50000.001192092925
    dwell=np.r_[float('0.0009999999761581'), .001+np.random.default_rng(4).exponential(.01,99)]
    frame=pd.DataFrame({'dwell_s':dwell,'accepted':True,'filter_limited':False})
    before=frame.copy(deep=True)
    inp=dwell_fit_inputs((frame,{'detector':'cusum_sequential','fs':fs,'min_dwell_s':.001,'min_dwell_samples':50}))
    result=fit_dwell_exponential(inp.dwell,t_min=inp.t_min,n_boot=0)
    assert result.n==100
    pd.testing.assert_frame_equal(before,frame)
    # A deliberately stricter analyst floor remains an explicit analytical choice.
    with pytest.raises(ValueError):fit_dwell_exponential(dwell,t_min=.001,n_boot=0)
