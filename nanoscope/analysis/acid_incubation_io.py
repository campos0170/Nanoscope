"""Validated readers for the acid notebook's standard and corrected ABF1 sources."""
from pathlib import Path
import struct
import numpy as np
from .cancel import check_cancelled, OperationCancelled


def read_abf_all_channels(path,*,cancel=None):
    path=Path(path)
    check_cancelled(cancel)
    with path.open('rb') as f:header=f.read(6144)
    if header[:4]==b'ABF2':
        from .manual_preparation import inspect_raw
        info=inspect_raw(path,cancel=cancel)
        if not info['channels'] or info['channels'][0]['units']!='pA':raise ValueError('Acid current channel must declare pA units')
        try:
            import pyabf
            abf=pyabf.ABF(str(path));data=np.asarray(abf.data,dtype=np.float32).T;fs=int(round(abf.dataRate))
        except OperationCancelled:raise
        except Exception as exc:raise ValueError('Standard ABF2 reader failed; no raw fallback is permitted') from exc
        if data.shape!=(info['n_samples'],len(info['channels'])) or not np.isclose(fs,info['fs_hz']) or not np.isfinite(data).all():
            raise ValueError('ABF2 reader returned inconsistent or non-finite samples')
        fields={name:getattr(abf,name,None) for name in ('abfVersionString','dataRate','sweepCount','channelCount')}
        check_cancelled(cancel)
        return data,fs,f"pyabf/standard  units={list(getattr(abf,'adcUnits',[]))}",fields
    if len(header)<1024 or header[:4]!=b'ABF ':raise ValueError('Acid preparation requires a validated ABF1 recording')
    fmt=struct.unpack_from('<h',header,100)[0]
    channels=struct.unpack_from('<h',header,120)[0]
    interval=struct.unpack_from('<f',header,122)[0]
    total=struct.unpack_from('<i',header,10)[0]
    ignored=struct.unpack_from('<h',header,14)[0]
    block=struct.unpack_from('<i',header,40)[0]
    if fmt not in (0,1) or not 1<=channels<=16 or not np.isfinite(interval) or interval<=0:
        raise ValueError('Invalid ABF1 format, channel count or sample interval')
    if total<=0 or total%channels or ignored<0 or block<1:raise ValueError('Invalid ABF1 sample geometry')
    offset=block*512+ignored
    itemsize=2 if fmt==0 else 4
    if offset<512 or path.stat().st_size<offset+total*itemsize:raise ValueError('Truncated ABF1 sample payload')
    fs_exact=1e6/interval/channels
    if not np.isfinite(fs_exact) or fs_exact<=0:raise ValueError('Invalid ABF1 sample rate')
    declared_units=[header[602+i*8:610+i*8].decode('ascii','replace').strip('\x00 ') for i in range(channels)]
    if not declared_units or declared_units[0]!='pA':raise ValueError('Acid current channel must declare pA units')
    result_header={}
    try:
        import pyabf
        abf=pyabf.ABF(str(path))
        data=np.asarray(abf.data,dtype=np.float32).T
        fs=int(round(abf.dataRate))
        if data.shape!=(total//channels,channels) or not np.isclose(fs,fs_exact,rtol=1e-5):raise ValueError('ABF reader disagrees with sample geometry')
        description=f"pyabf/standard  units={list(getattr(abf,'adcUnits',declared_units))}"
        for name in ('abfVersionString','dataRate','sweepCount','channelCount'):result_header[name]=getattr(abf,name,None)
        h=getattr(abf,'_headerV1',None)
    except OperationCancelled:
        raise
    except Exception as exc:
        if fmt!=1:raise ValueError('Standard int16 ABF reader failed; unsafe float32 fallback refused') from exc
        check_cancelled(cancel)
        with path.open('rb') as f:
            f.seek(offset);data=np.frombuffer(f.read(total*4),dtype='<f4').reshape(total//channels,channels).copy()
        fs=int(1e6/interval/channels)
        description=f'float32-manual  {channels}ch  (pyabf failed: {type(exc).__name__})'
        from pyabf.abf import HeaderV1
        with path.open('rb') as f:h=HeaderV1(f)
    if not np.isfinite(data).all():raise ValueError('ABF current contains NaN/Inf')
    if h is not None:
        for name in ('nTelegraphEnable','fTelegraphFilter','fADCRange','lADCResolution','fSignalGain','fInstrumentScaleFactor'):
            value=getattr(h,name,None)
            if value is not None:result_header[name]=value if np.isscalar(value) else list(np.asarray(value).ravel()[:4].tolist())
    check_cancelled(cancel)
    return data,fs,description,result_header
