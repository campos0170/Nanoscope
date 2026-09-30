"""Signal figure captions follow explicit study metadata and saved observations."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nanoscope.plot import signal_study_figures as figures


def _comparison():
    summary=pd.DataFrame(dict(tag=['first','second'],incubation_h=[2.,10.],
        pore_id=['same-pore','same-pore'],rate_per_s=[1.,2.],duty=[.1,.2],
        dwell_med=[2.,3.],depth_sust_med=[8.,9.],f_sust_med=[.08,.09],G_nS=[1.,1.]))
    events=pd.DataFrame(dict(tag=['first','second'],dwell_ms=[2.,3.],
        depth_sust_pA=[8.,9.],f_sust=[.08,.09]))
    pairs=pd.DataFrame(dict(tag_a=['first'],tag_b=['second'],nominal_nm=[10.],
        rate_ratio_b_over_a=[2.],conductance_ratio_b_over_a=[1.],duty_ratio_b_over_a=[2.]))
    types=pd.DataFrame([dict(tag=tag,kind=kind,n_accepted=1,rate_per_s=1.)
        for tag in summary.tag for kind in ('spike','plateau')])
    return SimpleNamespace(summary=summary,events=events,
        tables={'nominal_pairs':pairs,'signal_types':types})


@pytest.mark.parametrize('metadata,expected',[
    (None,'Incubation study'),
    ({'name':'Zinc kinetics','sample':'Protein batch 2','condition_label':'Zn'},
     'Zinc kinetics — Sample: Protein batch 2; Condition: Zn'),
    ({'name':'Salt comparison','sample':'Peptide','condition_label':'Mg'},
     'Salt comparison — Sample: Peptide; Condition: Mg'),
])
def test_comparison_titles_match_study_and_pore_identity(tmp_path,monkeypatch,metadata,expected):
    result=_comparison(); before=deepcopy(result); captured={}
    def save(fig,folder,name,source,**kwargs):
        captured[name]=(fig._suptitle.get_text(),source.copy(deep=True))
        fig.clear()
        return folder/(name+'.png')
    monkeypatch.setattr(figures,'_save',save)
    paths=figures.comparison_figures(result,[],tmp_path,study_metadata=metadata)
    assert len(paths)==5
    assert all(title.startswith(expected+'\n') for title,_ in captured.values())
    summary=captured['signal_comparison_summary'][0]
    assert 'Same physical pore across recordings' in summary
    assert 'different physical pores' not in summary.lower()
    assert 'Cu incubation' not in summary
    pd.testing.assert_frame_equal(result.summary,before.summary)
    pd.testing.assert_frame_equal(result.events,before.events)
    pd.testing.assert_frame_equal(captured['signal_comparison_summary'][1],before.summary)
    pd.testing.assert_frame_equal(captured['signal_comparison_distributions'][1],before.events)


def test_recording_titles_carry_metadata_without_altering_catalog(tmp_path,monkeypatch):
    from nanoscope.analysis.incubation_detection import PreparedRecording
    from nanoscope.analysis.study_signal import SchmittSettings,analyze
    rng=np.random.default_rng(113)
    signal=rng.normal(0,1,10000).astype('f4'); signal[500:510]-=15; signal[2000:2060]-=12
    prepared=PreparedRecording('trace',np.full(len(signal),-100.),signal,10000.,1.,
        {'event_polarity':1,'qc_mad_noise_pA':1},
        {'event_polarity':1,'metadata':{'event_polarity':1,'qc_mad_noise_pA':1}})
    analysis=analyze(prepared,SchmittSettings(min_ms=.2,plateau_ms=2))
    before=analysis.catalog.copy(deep=True); titles=[]
    def save(fig,folder,name,source,**kwargs):
        titles.append(fig._suptitle.get_text()); fig.clear()
        return folder/(name+'.png')
    monkeypatch.setattr(figures,'_save',save)
    paths=figures.recording_figures(analysis,None,tmp_path,
        study_metadata={'name':'New incubation','sample':'Aβ','condition_label':'General'})
    assert len(paths)==5
    assert all(title.startswith('New incubation — Sample: Aβ; Condition: General\ntrace:') for title in titles)
    pd.testing.assert_frame_equal(analysis.catalog,before)
