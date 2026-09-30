"""Adversarial ABF fallback inputs using portable synthetic header bytes.

The original study files remain read-only. Every test owns its tiny ABF fixture.
"""
import struct
from pathlib import Path
import numpy as np
import pytest
import pyabf
from nanoscope.analysis.acid_incubation_io import read_abf_all_channels


@pytest.fixture
def loader(monkeypatch):
    def standard_failure(*a,**kw):raise ValueError('Simulated pyABF standard reader failure')
    monkeypatch.setattr(pyabf,'ABF',standard_failure)
    return read_abf_all_channels


def fixture_abf(tmp_path,*,fmt=1,channels=2,interval=10.,truncate=False,nan=False,unit=b'pA'):
    header=bytearray(6144)
    header[:4]=b'ABF '
    struct.pack_into('<f',header,4,1.83)
    struct.pack_into('<i',header,40,12)
    struct.pack_into('<i',header,10,40)
    struct.pack_into('<h',header,100,fmt)
    struct.pack_into('<h',header,120,channels)
    struct.pack_into('<f',header,122,interval)
    header[602:610]=unit.ljust(8,b' ')
    values=np.tile(np.array([-100,-300],dtype='<f4'),20)
    if nan:values[4]=np.nan
    data=values.tobytes()
    if truncate:data=data[:-4]
    p=tmp_path/'fixture.abf';p.write_bytes(header+data)
    return p


def test_known_float32_abf1_fallback_retains_current_units_and_clock(loader,tmp_path):
    data,fs,fmt,header=loader(fixture_abf(tmp_path))
    assert data.shape==(20,2) and fs==50_000
    np.testing.assert_array_equal(data[:,0],np.full(20,-100,dtype=np.float32))

@pytest.mark.parametrize('kwargs',[{'fmt':0},{'channels':0},{'interval':0.},{'truncate':True},{'nan':True}])
def test_invalid_fallback_is_typed_value_error(loader,tmp_path,kwargs):
    with pytest.raises(ValueError):loader(fixture_abf(tmp_path,**kwargs))


def test_cancellation_is_never_converted_to_a_fallback_read(tmp_path,monkeypatch):
    from nanoscope.analysis.cancel import OperationCancelled
    from nanoscope.analysis.acid_incubation_io import read_abf_all_channels
    def cancelled(*a,**kw):raise OperationCancelled('cancelled')
    monkeypatch.setattr(pyabf,'ABF',cancelled)
    with pytest.raises(OperationCancelled):read_abf_all_channels(fixture_abf(tmp_path))


def test_wrong_current_unit_rejected(loader,tmp_path):
    with pytest.raises(ValueError):loader(fixture_abf(tmp_path,unit=b'mV'))


def test_abf2_standard_reader_is_allowed_but_shape_checked(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from nanoscope.analysis import manual_preparation
    from nanoscope.analysis.acid_incubation_io import read_abf_all_channels
    path=tmp_path/'standard.abf';path.write_bytes(b'ABF2'+b'\0'*6140)
    monkeypatch.setattr(manual_preparation,'inspect_raw',lambda *a,**k:dict(channels=[{'units':'pA'},{'units':'mV'}],n_samples=20,fs_hz=50000))
    data=np.tile(np.array([[-100],[-300]],np.float32),(1,20))
    monkeypatch.setattr(pyabf,'ABF',lambda *a,**k:SimpleNamespace(data=data,dataRate=50000,adcUnits=['pA','mV']))
    got,fs,fmt,meta=read_abf_all_channels(path)
    assert got.shape==(20,2) and fs==50000
    monkeypatch.setattr(pyabf,'ABF',lambda *a,**k:SimpleNamespace(data=data[:,:19],dataRate=50000,adcUnits=['pA','mV']))
    with pytest.raises(ValueError):read_abf_all_channels(path)


@pytest.mark.parametrize('channel',[1,2,-1,True])
def test_public_preparation_refuses_voltage_or_invalid_selected_channel(tmp_path,channel):
    from nanoscope.analysis import acid_incubation as acid
    state=acid.new_state(records=[dict(tag='acid_30h_whole',source_path=tmp_path/'fixture.abf',channel=channel,
        source_info={'sha256':'unused','channels':[{'units':'pA'},{'units':'mV'}]})])
    with pytest.raises(ValueError,match='channel'):
        acid.prepare_record(state,'acid_30h_whole',tmp_path/'outputs')
