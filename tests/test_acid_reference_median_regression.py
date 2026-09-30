"""Independent oracle for the actual quantized reference trigger and boundaries."""
from pathlib import Path
import numpy as np
import pytest
from nanoscope.analysis.acid_incubation_numerics import median_filter_1d

FIXTURE=Path(__file__).parent/'fixtures/reference_raw_prefix_decim10.npy'


def direct(values,w,mode):
    r=w//2;x=np.pad(values,(r,r),mode='symmetric' if mode=='reflect' else 'edge')
    return np.array([np.median(x[i:i+w]) for i in range(len(values))],dtype=values.dtype)


@pytest.mark.parametrize('values',[np.array([7.],np.float32),np.array([1.,9.],np.float32),
    np.array([9.,2.,7.,4.,3.],np.float32),np.arange(13,dtype=np.float64)])
@pytest.mark.parametrize('mode',['reflect','nearest'])
@pytest.mark.parametrize('window',[1,3,9,31])
def test_exact_padding_small_and_larger_than_signal(values,mode,window):
    np.testing.assert_array_equal(median_filter_1d(values,window,mode),direct(values,window,mode))


def test_reference_prefix_actual_raw_windows_and_length_independence():
    x=np.load(FIXTURE,allow_pickle=False);w=10001;r=w//2
    y=median_filter_1d(x,w,'reflect')
    padded=np.pad(x,(r,r),mode='symmetric')
    selected=np.unique(np.r_[0,1,4999,np.arange(5000,25001,500),len(x)-2,len(x)-1])
    expected=np.array([np.median(padded[i:i+w]) for i in selected],dtype=x.dtype)
    np.testing.assert_array_equal(y[selected],expected)
    assert y[5000]==np.float32(-291.748046875)
    shorter=median_filter_1d(x[:25001],w,'reflect')
    np.testing.assert_array_equal(shorter[:20001],y[:20001])
    assert y.dtype==x.dtype and len(y)==len(x)
