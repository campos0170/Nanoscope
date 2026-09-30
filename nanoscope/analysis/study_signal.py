"""Explicit source selection and plateau-aware signal detection for manual studies.

Event indices and onset_s use retained acquisition time. original_onset_s maps
back to the source ABF, including removed intervals. No event crosses a cut.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .cancel import check_cancelled
from .manual_detection import _number, _integer, recording_identity, _verify_sources
from .incubation_detection import RecordingAnalysis


@dataclass(frozen=True)
class SourceOptions:
    add_channel: int = 0  # UI convention: 0 disables; otherwise one-based
    add_channel_as_pa: bool = False  # explicit unit interpretation for mislabeled process exports
    keep_start_s: float = 0.
    keep_stop_s: float = 0.  # 0 means source end
    exclude_s: str = ''  # start:stop; start:stop, inclusive endpoints
    median_engine: str = 'reflect'

    def __post_init__(self):
        if not isinstance(self.add_channel_as_pa,bool): raise ValueError('Channel unit override must be boolean')
        object.__setattr__(self, 'add_channel', _integer(self.add_channel, 'additional channel', minimum=0))
        for name in ('keep_start_s', 'keep_stop_s'):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.keep_stop_s and self.keep_stop_s <= self.keep_start_s:
            raise ValueError('Keep stop must be later than keep start, or zero for source end')
        if self.median_engine not in ('reflect', 'shrinking'):
            raise ValueError('Median edges must be reflect or shrinking')
        self.exclusions()

    def exclusions(self):
        if not isinstance(self.exclude_s,str): raise ValueError('Excluded intervals must be text in seconds')
        pairs = []
        for item in self.exclude_s.split(';'):
            if not item.strip(): continue
            try: a, b = map(float, item.split(':'))
            except ValueError as exc: raise ValueError('Excluded intervals use start:stop; start:stop in seconds') from exc
            if not np.isfinite([a,b]).all() or not 0 <= a < b:
                raise ValueError('Excluded intervals must be finite, nonnegative and ordered')
            pairs.append((a,b))
        return pairs


def retain_samples(raw, fs, options):
    # Time comparisons reproduce the notebook's half-open keep and closed cuts.
    t = np.arange(len(raw), dtype=np.float64) / fs
    if options.keep_start_s >= len(raw)/fs:
        raise ValueError('Keep start is outside the recording')
    keep = t >= options.keep_start_s
    if options.keep_stop_s: keep &= t < options.keep_stop_s
    for a,b in options.exclusions(): keep &= ~((t >= a) & (t <= b))
    edges = np.diff(np.r_[False, keep, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts): raise ValueError('The selected intervals contain no samples')
    ranges = []; at = 0
    for a,b in zip(starts,ends):
        ranges.append([at, at+int(b-a), int(a), int(b)])
        at += int(b-a)
    return raw[keep], ranges


def source_indices(recording, indices):
    indices = np.asarray(indices)
    ranges = recording.metadata.get('retained_ranges', [[0,len(recording.raw),0,len(recording.raw)]])
    starts = np.array([r[0] for r in ranges])
    offsets = np.array([r[2]-r[0] for r in ranges])
    return indices + offsets[np.searchsorted(starts, indices, side='right')-1]


def shrinking_baseline(sig, fs, config, *, cancel=None):
    def median(x, seconds):
        check_cancelled(cancel)
        decim = config.baseline_decim; xs = x[::decim]
        width = max(1, int(seconds*fs/decim)); width += (width % 2 == 0)
        med = pd.Series(xs).rolling(width, center=True, min_periods=1).median().to_numpy()
        return np.interp(np.arange(len(x)),np.arange(0,len(x),decim),med).astype(np.float32)
    if config.baseline_method == 'robust_median':
        base = median(sig, config.baseline_window_s)
    elif config.baseline_method == 'masked_median':
        rough = sig - median(sig, config.masked_rough_s)
        center = np.median(rough); noise = 1.4826*np.median(np.abs(rough-center))
        mask = binary_dilation(rough-center < -config.masked_mask_sigma*noise,
                              iterations=max(1,int(config.masked_dilate_ms*1e-3*fs)))
        filled = sig.copy(); idx = np.arange(len(sig))
        if mask.any() and (~mask).sum() > 2:
            filled[mask] = np.interp(idx[mask],idx[~mask],sig[~mask])
        base = median(filled,config.masked_base_s)
    else: raise ValueError('Shrinking median edges require a median baseline method')
    return (sig-base).astype(np.float32),base


@dataclass(frozen=True)
class SchmittSettings:
    smooth_samples: int = 5
    enter_sigma: float = 4.
    enter_min_pa: float = 8.
    exit_sigma: float = 1.
    merge_ms: float = 5.
    min_ms: float = .2
    plateau_ms: float = 1.
    plateau_sigma: float = 3.
    i0_window_s: float = 5.
    i0_decim: int = 50

    def __post_init__(self):
        for name in ('smooth_samples','i0_decim'):
            object.__setattr__(self,name,_integer(getattr(self,name),name))
        if self.smooth_samples % 2 != 1: raise ValueError('Smoothing needs an odd sample count')
        for name in ('enter_sigma','enter_min_pa','exit_sigma','merge_ms','min_ms','plateau_ms','plateau_sigma','i0_window_s'):
            object.__setattr__(self,name,_number(getattr(self,name),name))
        if self.enter_sigma <= self.exit_sigma or self.min_ms <= 0 or self.i0_window_s <= 0:
            raise ValueError('Enter sigma must exceed exit sigma; duration and I0 window must be positive')


def thresholds(sig, settings):
    med=np.median(sig); mad=1.4826*np.median(np.abs(sig-med))
    if not np.isfinite(mad) or mad <= 0: raise ValueError('Schmitt detection needs nonzero finite MAD noise')
    return med,mad,max(settings.enter_sigma*mad,settings.enter_min_pa),settings.exit_sigma*mad


def detect_bounds(sig, fs, settings, fixed, *, sign=-1, ranges=None, cancel=None):
    med,mad,enter,exit_ = fixed; output=[]
    ranges = ranges or [[0,len(sig),0,len(sig)]]
    for lo,hi,*_ in ranges:
        check_cancelled(cancel)
        x=sign*(sig[lo:hi]-med); n=settings.smooth_samples
        if len(x)<n: continue
        c=np.cumsum(np.r_[0,x],dtype=np.float64); avg=(c[n:]-c[:-n])/n
        xs=np.r_[np.full(n//2,avg[0]),avg,np.full(len(x)-len(avg)-n//2,avg[-1])].astype(np.float32)
        above=xs>enter
        if not above.any(): continue
        ext=xs>exit_
        starts=np.flatnonzero(np.diff(above.astype(np.int8))==1)+1
        if above[0]: starts=np.r_[0,starts]
        edges=np.diff(np.r_[False,ext,False].astype(np.int8))
        es=np.flatnonzero(edges==1); ee=np.flatnonzero(edges==-1)-1
        li=np.searchsorted(es,starts,side='right')-1
        regs=sorted({(int(es[i]),int(ee[i])) for i in li if 0<=i<len(ee)})
        merged=[]; gap=int(settings.merge_ms*1e-3*fs)
        for s,e in regs:
            if merged and s-merged[-1][1]<=gap: merged[-1]=(merged[-1][0],max(e,merged[-1][1]))
            else: merged.append((s,e))
        for s,e in merged:
            check_cancelled(cancel)
            if e-s+1 < int(settings.min_ms*1e-3*fs): continue
            seg=x[s:e+1]; dwell=(e-s+1)/fs*1000.; sustained=float(np.median(seg))
            kind='spike' if dwell<settings.plateau_ms else ('plateau' if sustained>=settings.plateau_sigma*mad else 'shallow_long')
            output.append(dict(start_idx=s+lo,end_idx=e+lo,dwell_ms=dwell,
                depth_peak_pA=float(seg.max()),depth_sust_pA=sustained,kind=kind))
    return pd.DataFrame(output,columns=['start_idx','end_idx','dwell_ms','depth_peak_pA','depth_sust_pA','kind'])


def analyze(recording, settings, *, progress=None, cancel=None, imported=None):
    if not isinstance(settings,SchmittSettings): raise TypeError('settings must be SchmittSettings')
    identity=recording_identity(recording,cancel=cancel); _verify_sources(recording.provenance,cancel=cancel)
    sig=np.asarray(recording.detrended,np.float32); fs=recording.fs_hz
    fixed=thresholds(sig,settings); med,mad,enter,exit_=fixed
    if progress: progress(.15,'Finding smoothed hysteresis events')
    if imported is None:
        cat=detect_bounds(sig,fs,settings,fixed,ranges=recording.metadata.get('retained_ranges'),cancel=cancel)
    else:
        # The native import validator supplies a detached, measured snapshot.
        if imported.tag != recording.tag or imported.provenance.get('recording_identity') != identity:
            raise ValueError('Imported events do not belong to this prepared recording')
        cat=imported.catalog.copy(deep=True)
        seams=[r[0] for r in recording.metadata.get('retained_ranges',[])[1:]]
        for seam in seams:
            if ((cat.start_idx < seam) & (cat.end_idx >= seam)).any():
                raise ValueError('An imported event crosses an excluded source interval')
        depths=[]
        for row in cat.itertuples():
            check_cancelled(cancel)
            body=-(sig[int(row.start_idx):int(row.end_idx)+1]-med)
            depths.append((float(body.max()),float(np.median(body))))
        cat['depth_peak_pA']=[v[0] for v in depths]; cat['depth_sust_pA']=[v[1] for v in depths]
        cat['kind']=np.where(cat.dwell_ms<settings.plateau_ms,'spike',np.where(cat.depth_sust_pA>=settings.plateau_sigma*mad,'plateau','shallow_long'))
    if progress: progress(.5,'Measuring unsmoothed depth and local raw current')
    raw=np.asarray(recording.raw,np.float32); decim=settings.i0_decim
    b=pd.Series(raw[::decim]).rolling(max(1,int(settings.i0_window_s*fs/decim)),center=True,min_periods=1).median().to_numpy()
    # Match the notebook's interpolate -> float32 -> abs convention at event onsets.
    at=cat.start_idx.to_numpy(np.int64)
    i0=np.abs(np.interp(at,np.arange(0,len(raw),decim),b).astype(np.float32))
    global_i0=abs(float(np.median(raw)))
    if global_i0<=0 or np.any(i0<=0): raise ValueError('Raw open-pore reference must be nonzero')
    cat['tag']=recording.tag
    if imported is None: cat['event_id']=[f'{recording.tag}:schmitt:{i}' for i in range(len(cat))]
    for name in ('start_idx','end_idx'): cat[name]=cat[name].astype(np.int64)
    cat['onset_s']=cat.start_idx/fs; cat['end_s']=(cat.end_idx+1)/fs
    cat['original_onset_s']=source_indices(recording,at)/fs
    cat['original_end_s']=(source_indices(recording,cat.end_idx.to_numpy(np.int64))+1)/fs
    cat['duration_ms']=cat.dwell_ms; cat['I0_local_pA']=i0; cat['I0_global_pA']=global_i0
    cat['f_sust']=cat.depth_sust_pA/i0; cat['f_peak']=cat.depth_peak_pA/i0
    cat['fracblock']=cat.f_sust; cat['sust_sigma']=cat.depth_sust_pA/mad
    cat['peak_sigma']=cat.depth_peak_pA/mad; cat['depth_pA']=cat.depth_peak_pA
    cat['depth_sigma']=cat.peak_sigma; cat['depth_sust_sigma']=cat.sust_sigma
    cat['passes_gate']=cat.kind.ne('shallow_long') & ((cat.end_idx-cat.start_idx+1)>=int(settings.min_ms*1e-3*fs))
    if imported is None: cat['accepted']=cat.passes_gate
    provenance=deepcopy(recording.provenance)
    provenance.update(stage='signal_detection',recipe_id='schmitt_plateau_v4',recording_identity=identity,
        schmitt_settings=asdict(settings),baseline_pA=float(med),noise_mad_pA=float(mad),enter_pA=float(enter),exit_pA=float(exit_),
        gate_statistic='spike (< plateau_ms), or sustained depth >= plateau_sigma * whole-trace MAD',
        acceptance_source='automatic Schmitt class; shallow_long retained for review',
        index_convention='inclusive sample bounds on retained trace; onset_s is retained acquisition time; original_onset_s maps to source ABF',
        cut_policy='baseline on retained concatenated samples; detection and merging cannot cross a removed interval',
        i0_detail='absolute centered rolling raw median, decimated grid, shrinking edges; evaluated at onset',
        candidate_count=len(cat),accepted_count=int(cat.accepted.sum()))
    if imported is not None:
        provenance['imported_events']=deepcopy(imported.provenance['imported_events'])
        provenance['acceptance_source']='explicit imported workspace acceptance; Schmitt class is advisory'
    _verify_sources(recording.provenance,cancel=cancel)
    if progress: progress(1.,f'{len(cat):,} candidates; {cat.accepted.sum():,} accepted')
    return RecordingAnalysis(recording.tag,recording.raw,sig,fs,recording.duration_s,cat,provenance)


@dataclass(frozen=True)
class NullSettings:
    global_seeds: int = 5
    block_seeds: int = 3
    block_s: float = 10.

    def __post_init__(self):
        for name in ('global_seeds','block_seeds'):
            object.__setattr__(self,name,_integer(getattr(self,name),name,minimum=0))
            if getattr(self,name)>100: raise ValueError('At most 100 surrogate seeds per family')
        object.__setattr__(self,'block_s',_number(self.block_s,'block seconds',positive=True))


def null_controls(analysis, settings, *, progress=None, cancel=None):
    sig=np.asarray(analysis.detrended,np.float32); p=analysis.provenance
    detector=SchmittSettings(**p['schmitt_settings']); fs=analysis.fs_hz
    fixed=thresholds(sig,detector); med=fixed[0]; rows=[]
    def surrogate(x,seed):
        check_cancelled(cancel); rng=np.random.default_rng(seed); n=len(x); f=np.fft.rfft(x-med)
        ph=np.exp(1j*rng.uniform(0,2*np.pi,len(f))); ph[0]=1
        if n%2==0: ph[-1]=1
        return np.fft.irfft(np.abs(f)*ph,n=n).astype(np.float32)+med
    count=settings.global_seeds+settings.block_seeds+1
    for family, seeds in [('global_phase',settings.global_seeds),('block_phase',settings.block_seeds),('sign_flip',1)]:
        for seed in range(seeds):
            check_cancelled(cancel)
            if family=='global_phase': null=surrogate(sig,seed)
            elif family=='block_phase':
                null=np.empty_like(sig); width=max(1,int(settings.block_s*fs))
                for k,lo in enumerate(range(0,len(sig),width)):
                    null[lo:lo+width]=surrogate(sig[lo:lo+width],seed*100003+k)
            else: null=sig
            events=detect_bounds(null,fs,detector,fixed,sign=1 if family=='sign_flip' else -1,
                ranges=p['metadata'].get('retained_ranges'),cancel=cancel)
            counts=events.kind.value_counts()
            rows.append(dict(tag=analysis.tag,family=family,seed=seed,
                **{kind:int(counts.get(kind,0)) for kind in ('spike','plateau','shallow_long')}))
            if progress: progress(len(rows)/count,f'{analysis.tag}: {family}, seed {seed}')
            del null
    result=pd.DataFrame(rows); result.attrs['settings']=asdict(settings)
    result.attrs['method']='Thresholds pinned to real trace; null counts are diagnostics, not a measured false-discovery rate'
    return result
