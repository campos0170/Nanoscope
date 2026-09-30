"""Signal-study plots from committed detections and review, with numerical sidecars."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .incubation_figures import _figure, _save, _pore_caveat
from ..analysis.study_signal import source_indices
from ..analysis.cancel import check_cancelled

COLORS={'spike':'#147D92','plateau':'#CA7627','shallow_long':'#9B9B9B'}


def _study_heading(metadata=None):
    metadata=metadata or {}
    name=str(metadata.get('name') or '').strip() or 'Incubation study'
    details=[]
    for key,label in (('sample','Sample'),('condition_label','Condition')):
        value=str(metadata.get(key) or '').strip()
        if value: details.append(f'{label}: {value}')
    return name+(' — '+'; '.join(details) if details else '')


def _label_study(fig,metadata):
    title=fig._suptitle.get_text() if fig._suptitle is not None else ''
    fig.suptitle(_study_heading(metadata)+('\n'+title if title else ''))


def _source_time(a,idx):
    # RecordingAnalysis carries source selection in provenance, not metadata.
    class View: pass
    view=View(); view.raw=a.raw; view.metadata=a.provenance['metadata']
    return source_indices(view,idx)/a.fs_hz


def _envelope(a,values,budget=10000):
    width=max(1,int(np.ceil(len(values)/budget))); n=len(values)//width
    cut=np.asarray(values[:n*width]).reshape(n,width)
    lo=np.argmin(cut,axis=1)+np.arange(n)*width; hi=np.argmax(cut,axis=1)+np.arange(n)*width
    indices=np.sort(np.r_[lo,hi,len(values)-1]); return _source_time(a,indices),np.asarray(values)[indices]


def recording_figures(a,nulls,folder,*,cancel=None,study_metadata=None):
    from .study_figure_style import bind_event_panel
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True); paths=[]
    cat=a.catalog; ev=cat[cat.accepted]; tag=a.tag
    def save(fig,name,data):
        check_cancelled(cancel); _label_study(fig,study_metadata)
        paths.append(_save(fig,folder,f'signal_{tag}_{name}',data,cancel=cancel))
    fig=_figure(height=7); axes=fig.subplots(2,1)
    for ax,values,label in zip(axes,(a.raw,a.detrended),('Raw current (pA)','Prepared current (pA)')):
        x,y=_envelope(a,values); ax.plot(x,y,lw=.45,color='#354C59')
        ax.set(xlabel='Original source time (s)',ylabel=label)
    for kind,color in COLORS.items():
        selected=ev[ev.kind==kind]; axes[1].scatter(selected.original_onset_s,
            [a.detrended[int(s):int(e)+1].min() for s,e in zip(selected.start_idx,selected.end_idx)],s=9,color=color,label=f'{kind}: {len(selected)} accepted')
    axes[1].legend(fontsize=8); fig.suptitle(f'{tag}: raw and prepared trace; {len(ev)} accepted / {len(cat)} candidates')
    trace=pd.DataFrame({'original_time_s':x,'prepared_envelope_pA':y})
    save(fig,'trace',trace)
    fig=_figure(height=5); axes=fig.subplots(1,3)
    for kind,color in COLORS.items():
        selected=ev[ev.kind==kind]
        if not len(selected): continue
        axes[0].scatter(selected.dwell_ms,selected.depth_sust_pA,s=12,alpha=.65,color=color,label=kind)
        axes[1].hist(np.log10(selected.dwell_ms),bins=20,alpha=.55,color=color)
        axes[2].hist(selected.f_sust*100,bins=20,alpha=.55,color=color)
    axes[0].set(xscale='log',xlabel='Dwell (ms)',ylabel='Sustained current change (pA)'); axes[0].legend(fontsize=8)
    axes[1].set(xlabel='log10 dwell (ms)',ylabel='Accepted events'); axes[2].set(xlabel='Local fractional current change (%)',ylabel='Accepted events')
    fig.suptitle(f'{tag}: accepted event measurements (unsmoothed trace)'); save(fig,'distributions',ev)
    fig=_figure(height=9); axes=fig.subplots(3,3); examples=[]
    for row,kind in enumerate(COLORS):
        selected=cat[cat.kind==kind].sort_values('depth_sust_pA')
        for column,q in enumerate((.1,.5,.9)):
            ax=axes[row,column]
            ax._nanoscope_event_panel_identity = ''
            if selected.empty: ax.text(.5,.5,f'No {kind} events',ha='center',transform=ax.transAxes); continue
            event=selected.iloc[min(len(selected)-1,int(q*(len(selected)-1)))]; s,e=int(event.start_idx),int(event.end_idx)
            pad=max(100,e-s+1); lo=max(0,s-pad); hi=min(len(a.raw),e+1+pad)
            x=(np.arange(lo,hi)-s)/a.fs_hz*1000
            ax.plot(x,a.detrended[lo:hi],color='#354C59',lw=.7); ax.axvspan(0,event.dwell_ms,color=COLORS[kind],alpha=.2)
            ax.set(title=f'{kind} / depth quantile {q:g}\nsource {event.original_onset_s:.5f} s; '+('accepted' if event.accepted else 'excluded'),xlabel='Time relative to onset (ms)',ylabel='Current (pA)')
            bind_event_panel(ax, tag=tag, event_id=event.get('event_id', f'{tag}:{s}:{e}'),
                             start_idx=s, end_idx=e)
            examples.append(event.to_dict())
    fig.suptitle(f'{tag}: deterministic candidate examples; shaded inclusive event bounds'); save(fig,'examples',pd.DataFrame(examples))
    fig=_figure(height=5); axes=fig.subplots(1,3)
    source=[]
    for ax,kind in zip(axes,COLORS):
        actual=int((cat.kind==kind).sum()); ax.scatter([0],[actual],s=45,color=COLORS[kind],label='real candidates')
        source.append(dict(family='real',seed=0,kind=kind,n=actual))
        for x,family in enumerate(('global_phase','block_phase','sign_flip'),1):
            subset=nulls[nulls.family==family] if nulls is not None else pd.DataFrame()
            if not subset.empty:
                ax.scatter(np.full(len(subset),x),subset[kind],s=25,alpha=.6,color='#555555')
                source.extend(dict(family=family,seed=int(r.seed),kind=kind,n=int(getattr(r,kind))) for r in subset.itertuples())
        ax.set_xticks(range(4),['Real','Global phase','Block phase','Sign flip'],rotation=25)
        ax.set(ylabel='Candidate count',title=kind)
    fig.suptitle(f'{tag}: thresholds pinned to real trace; null counts are diagnostic, not measured FDR')
    if nulls is None: fig.text(.5,.01,'Null controls have not been run',ha='center')
    save(fig,'null_controls',pd.DataFrame(source))
    fig=_figure(height=7); axes=fig.subplots(2,2)
    for kind,color in COLORS.items():
        selected=ev[ev.kind==kind]
        axes[0,0].scatter(selected.original_onset_s,selected.depth_sust_pA,s=8,color=color,label=kind)
        axes[0,1].scatter(selected.original_onset_s,selected.dwell_ms,s=8,color=color)
    axes[0,0].set(ylabel='Sustained change (pA)'); axes[0,0].legend(fontsize=8)
    axes[0,1].set(ylabel='Dwell (ms)',yscale='log')
    axes[1,0].scatter(ev.original_onset_s,ev.I0_local_pA,s=8,color='#147D92'); axes[1,0].set(ylabel='Local |I0| (pA)')
    counts,edges=np.histogram(ev.onset_s,bins=np.linspace(0,a.duration_s,21)); axes[1,1].stairs(counts/np.diff(edges),edges,color='#147D92')
    axes[1,1].set(xlabel='Retained acquisition time (s)',ylabel='Accepted events / s')
    for ax in (axes[0,0],axes[0,1],axes[1,0]): ax.set_xlabel('Original source time (s)')
    fig.suptitle(f'{tag}: within-recording changes; exposure {a.duration_s:.5f} s'); save(fig,'time_dependence',ev)
    (folder/f'signal_{tag}_methods.json').write_text(json.dumps(a.provenance,indent=2,default=str))
    return paths


def comparison_figures(result,records,folder,*,cancel=None,study_metadata=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True); paths=[]
    s=result.summary; ev=result.events; x=np.arange(len(s)); labels=[f'{v:g} h' for v in s.incubation_h]
    def save(fig,name,data):
        _label_study(fig,study_metadata)
        paths.append(_save(fig,folder,'signal_comparison_'+name,data,cancel=cancel))
    fig=_figure(height=7); axes=fig.subplots(2,3)
    for ax,col,label in zip(axes.flat,('rate_per_s','duty','dwell_med','depth_sust_med','f_sust_med','G_nS'),('Accepted events / s','Occupied fraction','Median dwell (ms)','Median sustained change (pA)','Median local fractional change','Open-pore conductance (nS)')):
        ax.plot(x,s[col],'o-',color='#147D92'); ax.set_xticks(x,labels); ax.set_ylabel(label)
    fig.suptitle('Accepted-event summaries\n'+_pore_caveat(s))
    save(fig,'summary',s)
    pairs=result.tables['nominal_pairs']; fig=_figure(height=5); ax=fig.subplots()
    px=np.arange(len(pairs)); width=.25
    for offset,col,label,color in zip((-width,0,width),('rate_ratio_b_over_a','conductance_ratio_b_over_a','duty_ratio_b_over_a'),('Event-rate ratio','Conductance ratio','Duty ratio'),('#147D92','#CA7627','#646B85')):
        ax.bar(px+offset,pairs[col],width,label=label,color=color)
    ax.axhline(1,color='#888888',ls='--'); ax.set_xticks(px,[f'{r.tag_a} → {r.tag_b}\n{r.nominal_nm:g} nm labels' for r in pairs.itertuples()]); ax.legend()
    ax.set_ylabel('Ratio: later / earlier'); fig.suptitle('Nominal pore labels do not establish conductance matching'); save(fig,'nominal_pairs',pairs)
    fig=_figure(height=5); axes=fig.subplots(1,3)
    for ax,col,label in zip(axes,('dwell_ms','depth_sust_pA','f_sust'),('Dwell (ms)','Sustained current change (pA)','Local fractional current change')):
        for i,row in enumerate(s.itertuples()):
            vals=np.sort(ev.loc[ev.tag==row.tag,col]); ax.step(vals,np.arange(1,len(vals)+1)/max(1,len(vals)),where='post',label=f'{row.incubation_h:g} h; n={len(vals)}',color=f'C{i}')
        ax.set(xlabel=label,ylabel='Cumulative accepted-event fraction')
        if col=='dwell_ms': ax.set_xscale('log')
    axes[0].legend(fontsize=8); fig.suptitle('Empirical distributions; events are not independent biological replicates'); save(fig,'distributions',ev)
    fig=_figure(height=6); axes=fig.subplots(1,2)
    for i,kind in enumerate(('spike','plateau')):
        table=result.tables['signal_types']; table=table[table.kind==kind].set_index('tag').reindex(s.tag)
        axes[0].bar(x+i*.35,table.n_accepted,.35,label=kind,color=COLORS[kind]); axes[1].plot(x,table.rate_per_s,'o-',label=kind,color=COLORS[kind])
    for ax in axes: ax.set_xticks(x+.175 if ax is axes[0] else x,labels); ax.legend()
    axes[0].set_ylabel('Accepted count'); axes[1].set_ylabel('Accepted events / s')
    fig.suptitle('Spike and plateau contributions; shallow-long crossings retained for review'); save(fig,'event_types',result.tables['signal_types'])
    supplementary=[r for r in records if r.get('comparison_only') and 'analysis' in r]
    fig=_figure(height=5); axes=fig.subplots(1,3); rows=[]
    for r in records:
        if not supplementary or not any(r['incubation_h']==e['incubation_h'] for e in supplementary): continue
        a=r['analysis']; accepted=a.catalog[a.catalog.accepted]; i0=abs(float(np.median(np.asarray(a.raw,np.float32))))
        rows.append(dict(tag=r['tag'],supplementary=r.get('comparison_only',False),I0_pA=i0,
            rate_per_s=len(accepted)/a.duration_s,dwell_med=accepted.dwell_ms.median(),duty=accepted.dwell_ms.sum()/1000/a.duration_s))
    table=pd.DataFrame(rows)
    for ax,col,label in zip(axes,('I0_pA','rate_per_s','dwell_med'),('|I0| (pA)','Accepted events / s','Median dwell (ms)')):
        if len(table): ax.bar(np.arange(len(table)),table[col],color=['#147D92' if not v else '#CA7627' for v in table.supplementary]); ax.set_xticks(np.arange(len(table)),table.tag,rotation=20)
        else: ax.text(.5,.5,'No supplementary recording',ha='center',transform=ax.transAxes)
        ax.set_ylabel(label)
    fig.suptitle('Same nominal incubation, different recording state; supplementary excerpts are not additional timepoints')
    save(fig,'excerpt_check',table)
    return paths
