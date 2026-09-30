"""Descriptive comparisons for reviewed signal events; no implied depth floor."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .cancel import check_cancelled
from .manual_comparisons import ComparisonResult, reviewed_catalog, _inputs
from .incubation_study import block_bootstrap_rate


def compare_recordings(analyses, recordings, settings, *, folder, progress=None, cancel=None):
    frames=[]; summaries=[]; drift=[]; types=[]
    for a,r in _inputs(analyses,recordings):
        check_cancelled(cancel)
        if 'schmitt_settings' not in a.provenance:
            raise ValueError('Signal comparisons require signal detections for every primary recording')
        cat,ev,i0=reviewed_catalog(a,r); frames.append(ev)
        bb=block_bootstrap_rate(ev.onset_s,a.duration_s,settings.n_blocks,settings.n_boot,settings.seed)
        p=a.provenance
        summary=dict(tag=r.tag,incubation_h=r.incubation_h,pore_nm=r.pore_diameter_nm,
            pore_id=r.pore_id,condition=f'{r.incubation_h:g} h',I0_pA=i0,
            G_nS=i0/abs(r.voltage_mv) if r.voltage_mv else np.nan,
            duration_s=a.duration_s,n_all=len(cat),n_events=len(ev),
            n_automatic=int(cat.passes_gate.sum()),rate_per_s=len(ev)/a.duration_s,
            lo=bb['lo'],hi=bb['hi'],duty=ev.dwell_ms.sum()/1000/a.duration_s,
            dwell_med=ev.dwell_ms.median(),dwell_p90=ev.dwell_ms.quantile(.9),
            depth_sust_med=ev.depth_sust_pA.median(),f_sust_med=ev.f_sust.median(),
            mad=p['noise_mad_pA'],enter_pA=p['enter_pA'])
        summaries.append(summary)
        for kind in ('spike','plateau','shallow_long'):
            selected=ev[ev.kind==kind]
            types.append(dict(tag=r.tag,kind=kind,n_candidates=int((cat.kind==kind).sum()),
                n_accepted=len(selected),rate_per_s=len(selected)/a.duration_s,
                dwell_med=selected.dwell_ms.median(),depth_sust_med=selected.depth_sust_pA.median()))
        for variable in ('dwell_ms','depth_sust_pA','f_sust'):
            ok=len(ev)>=8 and ev[variable].nunique()>1
            rho=float(spearmanr(ev.original_onset_s,ev[variable]).statistic) if ok else np.nan
            drift.append(dict(tag=r.tag,variable=variable,rho=rho,n=len(ev),status='descriptive' if ok else 'insufficient variation or fewer than 8 events'))
    summary=pd.DataFrame(summaries).sort_values('incubation_h'); events=pd.concat(frames,ignore_index=True)
    pairs=[]
    for _,grp in summary.groupby('pore_nm'):
        from itertools import combinations
        for a,b in combinations(grp.itertuples(),2):
            pairs.append(dict(tag_a=a.tag,tag_b=b.tag,nominal_nm=a.pore_nm,
                same_physical_pore=a.pore_id==b.pore_id,conductance_ratio_b_over_a=b.G_nS/a.G_nS,
                rate_ratio_b_over_a=b.rate_per_s/a.rate_per_s if a.rate_per_s else np.nan,
                duty_ratio_b_over_a=b.duty/a.duty if a.duty else np.nan))
    monotonic=[]
    for col in ('rate_per_s','duty','dwell_med','depth_sust_med','f_sust_med','G_nS'):
        values=summary[col].to_numpy(float); ok=len(values)>2 and np.isfinite(values).all() and np.ptp(values)>0
        monotonic.append(dict(variable=col,n_recordings=len(values),rho=float(spearmanr(summary.incubation_h,values).statistic) if ok else np.nan,
            strictly_monotone=bool(ok and (np.all(np.diff(values)>0) or np.all(np.diff(values)<0)))))
    tables=dict(signal_types=pd.DataFrame(types),nominal_pairs=pd.DataFrame(pairs),drift=pd.DataFrame(drift),monotonicity=pd.DataFrame(monotonic))
    info=dict(signal_detector=True,detector='plateau-aware Schmitt',selection='accepted, separately from automatic class',
        settings=settings.__dict__,clock='rates and equal-duration block bootstrap use retained acquisition time; drift uses original source timestamps',
        amplitude='unsmoothed whole-trace median deviation; fractional change uses 5-second local raw median',
        caveat='one recording per condition; nominal pore labels do not establish conductance matching; descriptive event-level statistics',
        floor='No single sustained-depth floor describes the spike/plateau selection; single-depth-floor gate-transfer calculations are not applicable')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    events.to_csv(folder/'events.csv',index=False); summary.to_csv(folder/'summary.csv',index=False)
    for name,table in tables.items(): table.to_csv(folder/(name+'.csv'),index=False)
    (folder/'report.json').write_text(json.dumps(info,indent=2))
    if progress: progress(1.,'Signal comparisons complete')
    return ComparisonResult(events,summary,tables,info)
