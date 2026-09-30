"""Independent numerical reference and lifecycle checks for raw preparation."""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from scipy.signal import filtfilt, iirnotch
from scipy.ndimage import median_filter, binary_dilation
from nanoscope.analysis import manual_preparation as prep
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.incubation_detection import load_prepared_recording
from nanoscope.analysis.ml_preprocess import MLPreprocessConfig

@pytest.fixture
def source(tmp_path, monkeypatch):
    path=tmp_path/'test.abf';path.write_bytes(b'ABF test source')
    counts=np.random.default_rng(19).integers(-710,-680,(9000,2),dtype=np.int16)
    counts[1400:1450,0]-=160;counts[5400:5550,0]-=200
    gain=np.array([0.30517576675492886,0.03051757880712104]);offsets=np.array([0.,0.])
    class Native:
        header={'signal_channels':np.rec.fromarrays([gain,offsets],names=['gain','offset'])}
        def segment_count(self,block):return 1
        def get_analogsignal_chunk(self,*,i_start,i_stop,**kwargs):return counts[i_start:i_stop]
        def rescale_signal_raw_to_float(self,raw,dtype):
            out=raw.astype(dtype);out*=gain;out+=offsets;return out
    class Reader:
        def probe(self):
            return SimpleNamespace(meta=SimpleNamespace(sample_rate_hz=5000.,n_samples=len(counts),
                t0_s=3.5,format_name='Axon ABF',notes=(),issues=(),channels=[
                SimpleNamespace(index=0,name='IN11',raw_units='pA'),
                SimpleNamespace(index=1,name='IN14',raw_units='mV')]))
        def _open_reader(self):return Native()
    monkeypatch.setattr(prep,'_reader',lambda p:Reader())
    return path,counts,gain

def config(**changes):
    return MLPreprocessConfig(notch_freqs=(60.,120.),notch_q=30.,notch_enabled=True,
        notch_float32=False,baseline_method='masked_median',baseline_decim=5,
        masked_rough_s=.05,masked_base_s=.02,masked_mask_sigma=3.,masked_dilate_ms=.2,
        despike_size=0,normalize=True,normalize_method='robust_sigma').replace(**changes)

def notebook_reference(raw,fs,cfg):
    filtered=raw.copy()
    for f in cfg.notch_freqs:
        b,a=iirnotch(f,Q=cfg.notch_q,fs=fs);filtered=filtfilt(b,a,filtered)
    def baseline(x,seconds):
        xs=x[::cfg.baseline_decim];w=int(seconds*fs/cfg.baseline_decim)
        if w%2==0:w+=1
        b=median_filter(xs,size=w,mode='reflect')
        return np.interp(np.arange(len(x)),np.arange(0,len(x),cfg.baseline_decim),b).astype(np.float32)
    rough=filtered-baseline(filtered,cfg.masked_rough_s)
    sigma=1.4826*np.median(abs(rough-np.median(rough)))
    mask=binary_dilation(rough-np.median(rough)<-cfg.masked_mask_sigma*sigma,
        iterations=max(1,int(cfg.masked_dilate_ms*1e-3*fs)))
    filled=filtered.copy();idx=np.arange(len(raw))
    if mask.any() and (~mask).sum()>2:filled[mask]=np.interp(idx[mask],idx[~mask],filtered[~mask])
    base=baseline(filled,cfg.masked_base_s)
    return filtered,base,(filtered-base).astype(np.float32)

def test_manual_preparation_matches_independent_notebook(source):
    path,counts,gain=source;cfg=config();record=prep.prepare_raw(path,0,'example',cfg)
    raw=counts[:,0].astype(np.float32)*np.float32(gain[0])
    filtered,base,detr=notebook_reference(raw,5000,cfg)
    np.testing.assert_array_equal(record.raw.astype(np.float32),raw)
    np.testing.assert_array_equal(record.filtered,filtered)
    np.testing.assert_array_equal(record.baseline,base)
    np.testing.assert_array_equal(record.detrended,detr)
    assert abs(record.detrended.min())>1
    assert record.provenance['normalization_applied'] is False
    assert record.provenance['raw_file']['sha256']
    assert record.metadata['instrument_origin_s']==3.5

def test_reader_precision_is_explicit(source):
    path,_,_=source
    a=prep.prepare_raw(path,0,'legacy',config(),reader_scaling='notebook_float32')
    b=prep.prepare_raw(path,0,'standard',config(),reader_scaling='standard')
    assert np.any(a.raw!=b.raw) and np.max(abs(a.raw-b.raw))<.0001
    assert np.all(a.raw<0) and np.all(b.raw<0)
    assert b.provenance['reader_scaling']=='standard'

def test_upward_export_reopens_without_second_inversion(source,tmp_path):
    path,_,_=source;record=prep.prepare_raw(path,0,'upward',config(),event_polarity=-1)
    trace,meta=prep.export_prepared(record,tmp_path/'export');reopened=load_prepared_recording(trace,meta)
    np.testing.assert_array_equal(reopened.raw,record.raw)
    np.testing.assert_array_equal(reopened.detrended,record.detrended)
    native=pd.read_csv(trace)
    np.testing.assert_array_equal(native.current_detr_pA.to_numpy(np.float32),-record.detrended)
    np.testing.assert_array_equal(native.baseline_pA.to_numpy(np.float32),record.baseline)
    assert json.loads(meta.read_text())['preparation_provenance']['raw_file']['sha256']

def test_truncated_source_export_reopens(source,tmp_path):
    # Regression: export_prepared() writes 'time_s' as the ORIGINAL, pre-truncation
    # source time (via source_indices), which is only zero-origin when nothing was
    # cut. load_prepared_recording() must validate the actual zero-origin, contiguous
    # clock ('retained_time_s'), or any recording prepared with keep_start_s/
    # keep_stop_s/exclude_s could never be reopened.
    from nanoscope.analysis.study_signal import SourceOptions
    path,_,_=source
    record=prep.prepare_raw(path,0,'truncated',config(),source_options=SourceOptions(keep_start_s=1.0))
    assert record.metadata['retained_ranges']!=[[0,9000,0,9000]]
    trace,meta=prep.export_prepared(record,tmp_path/'export-truncated')
    reopened=load_prepared_recording(trace,meta)
    np.testing.assert_array_equal(reopened.raw,record.raw)
    np.testing.assert_array_equal(reopened.detrended,record.detrended)

def test_invalid_input_is_refused(source):
    path,_,_=source
    for channel in (1,-1,2,True):
        with pytest.raises(ValueError):prep.prepare_raw(path,channel,'x',config())
    with pytest.raises(ValueError,match='Nyquist'):prep.prepare_raw(path,0,'x',config(notch_freqs=(3000.,)))
    for tag in ('../escape','x/y','','.hidden'):
        with pytest.raises(ValueError):prep.prepare_raw(path,0,tag,config())
    with pytest.raises(ValueError):prep.prepare_raw(path,0,'x',config(),event_polarity=0)

def test_changed_source_is_refused(source):
    path,_,_=source
    def change(fraction,message):
        if fraction>=.8:path.write_bytes(b'changed source during preparation')
    with pytest.raises(ValueError,match='changed during preparation'):
        prep.prepare_raw(path,0,'x',config(),progress=change)

def test_cancel_does_not_publish_partial_files(source,tmp_path):
    path,_,_=source
    with pytest.raises(OperationCancelled):prep.prepare_raw(path,0,'x',config(),cancel=lambda:True)
    record=prep.prepare_raw(path,0,'x',config());state={'stop':False}
    with pytest.raises(OperationCancelled):
        prep.export_prepared(record,tmp_path/'cancelled',cancel=lambda:state['stop'],
                             progress=lambda f,m:state.update(stop=True))
    assert list((tmp_path/'cancelled').iterdir())==[]

def test_export_never_overwrites(source,tmp_path):
    path,_,_=source;record=prep.prepare_raw(path,0,'x',config())
    out=tmp_path/'export';out.mkdir();meta=out/'x_metadata.json';meta.write_text('do not replace')
    with pytest.raises(FileExistsError):prep.export_prepared(record,out)
    assert meta.read_text()=='do not replace' and not (out/'x.csv.gz').exists()
    meta.unlink();trace,meta=prep.export_prepared(record,out);before=(trace.read_bytes(),meta.read_bytes())
    with pytest.raises(FileExistsError):prep.export_prepared(record,out)
    assert before==(trace.read_bytes(),meta.read_bytes())
