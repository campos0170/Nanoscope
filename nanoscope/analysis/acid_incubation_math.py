"""Acid incubation numerical stages, statically ported from the owner's notebook.

Source: Acid30h_Preprocessing_and_Subpopulation_Workflow.ipynb
The original numerical order, formulas and stage boundaries are retained.
Stage globals belong to an isolated AcidRuntime namespace, not this module's
shared namespace. They are not user Python code. Runtime never reads/executes a
notebook. IO paths are assigned by the public engine within its workspace.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, math, os, warnings
import numpy as np
import pandas as pd
import matplotlib
from .acid_incubation_numerics import median_filter_1d as median_filter
from scipy.stats import chi2_contingency, norm, spearmanr, theilslopes
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler
import hdbscan
class _LazyUMAP:
    def __getattr__(self,name):
        import umap
        return getattr(umap,name)
umap = _LazyUMAP()
from diptest import diptest
from .cancel import check_cancelled
HAVE_DIPTEST = True
HAVE_TF = False
tf = None
def display(value):
    return None
def tqdm(values, **kwargs):
    return values
NOTEBOOK_SHA256 = "a74152cce46f052db7710018255bf196e0f0aef8746fc3be3ce5a78a65ef0c97"


# Original notebook cell 7
def stage_007():
    global _bessel, _maximum_filter1d, _median_filter, _sanitise_header, _sosfilt, _tail_tau_ms, baseline_rolling_median, bessel_cutoff_for_tau, estimate_adc_step, iirnotch, line_noise_notch, line_noise_sinusoid, load_stage0, match_bandwidth, measure_mains_frequency, preprocess_abf, psd_summary, quiet_psd, read_abf_all_channels, sosfiltfilt, tf2sos, welch
    check_cancelled(_acid_cancel)
    from scipy.signal import iirnotch, sosfiltfilt, tf2sos, welch, bessel as _bessel, sosfilt as _sosfilt
    from .acid_incubation_numerics import median_filter_1d as _median_filter
    from scipy.ndimage import maximum_filter1d as _maximum_filter1d

    def read_abf_all_channels(path: Path) -> tuple[np.ndarray, int, str, dict]:
        from .acid_incubation_io import read_abf_all_channels as validated_read
        return validated_read(path, cancel=_acid_cancel)

    def _sanitise_header(header: dict) -> dict:
        """Keep only finite, plausible header values (unset ABF1 fields read as garbage)."""
        clean = {}
        for k, v in header.items():
            check_cancelled(_acid_cancel)
            vals = v if isinstance(v, list) else [v]
            ok = []
            for x in vals:
                check_cancelled(_acid_cancel)
                try:
                    xf = float(x)
                except (TypeError, ValueError):
                    ok.append(x)
                    continue
                if np.isfinite(xf) and abs(xf) < 10000000.0:
                    ok.append(x)
            if ok:
                clean[k] = ok if isinstance(v, list) else ok[0]
        enable = clean.get('nTelegraphEnable')
        enable = enable if isinstance(enable, list) else [enable] if enable is not None else []
        if enable and (not all((int(e) in (0, 1) for e in enable if isinstance(e, (int, float))))):
            clean.pop('nTelegraphEnable', None)
            clean.pop('fTelegraphFilter', None)
            clean['telegraph_note'] = 'telegraph fields unset in this ABF1 header'
        elif enable and (not any((int(e) == 1 for e in enable))):
            clean.pop('fTelegraphFilter', None)
            clean['telegraph_note'] = 'telegraph not enabled; the acquisition filter setting is not in the header'
        return clean

    def estimate_adc_step(x: np.ndarray, n_check: int=2000000) -> tuple[float, bool]:
        """Smallest non-zero spacing between distinct sample values; True if the whole trace sits on
    that grid (so it can be stored losslessly as int16 codes)."""
        sub = np.asarray(x[:n_check], dtype=np.float64)
        vals = np.unique(sub)
        if len(vals) < 3:
            return (np.nan, False)
        diffs = np.diff(vals)
        step = float(np.min(diffs[diffs > 1e-09]))
        codes = np.asarray(x, dtype=np.float64) / step
        on_grid = bool(np.max(np.abs(codes - np.round(codes))) < 0.001 and np.max(np.abs(codes)) < 32767)
        return (step, on_grid)

    def line_noise_notch(sig: np.ndarray, fs: float, freqs, bw_hz: float) -> np.ndarray:
        sos = np.vstack([tf2sos(*iirnotch(f0, Q=f0 / bw_hz, fs=fs)) for f0 in freqs])
        return sosfiltfilt(sos, sig.astype(np.float64)).astype(np.float32)

    def measure_mains_frequency(sig: np.ndarray, fs: float, nominal: float=60.0, span_hz: float=0.5, min_prominence: float=5.0) -> tuple[float, float]:
        """(frequency, prominence) of the mains line: the spectral peak within +/- span_hz of the nominal
    frequency, measured against the median spectrum in the surrounding +/- 5 Hz. A prominence below
    min_prominence (or a peak sitting on the search-band edge) means there is no detectable mains line
    and the caller skips the line-noise step rather than fitting noise."""
        x = np.asarray(sig[:min(len(sig), int(fs * 300))], dtype=np.float64)
        x = x - x.mean()
        n = len(x)
        spec = np.abs(np.fft.rfft(x * np.hanning(n)))
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        band = (f >= nominal - span_hz) & (f <= nominal + span_hz)
        if not band.any():
            return (float(nominal), 0.0)
        k = int(np.argmax(spec[band]))
        f_peak = float(f[band][k])
        surround = (f >= nominal - 5.0) & (f <= nominal + 5.0) & ~band
        prominence = float(spec[band][k] / max(np.median(spec[surround]), 1e-30)) if surround.any() else 0.0
        on_edge = k == 0 or k == band.sum() - 1
        if on_edge:
            prominence = 0.0
        return (f_peak, prominence)

    def line_noise_sinusoid(sig: np.ndarray, fs: float, harmonics, window_s: float, mask_sigma: float, f_mains: float) -> tuple[np.ndarray, np.ndarray]:
        """Fit a_k cos + b_k sin at each harmonic of the measured mains frequency in consecutive windows,
    on samples that are not inside an excursion, interpolate the coefficients between window centres,
    synthesise the line noise and subtract it. Returns (cleaned, synthesised_line_noise)."""
        x = np.asarray(sig, dtype=np.float64)
        n = len(x)
        rough = x - _median_filter(x[::10], size=int(0.5 * fs / 10) | 1, mode='reflect').repeat(10)[:n]
        noise = 1.4826 * np.median(np.abs(rough - np.median(rough)))
        mask = _maximum_filter1d((np.abs(rough - np.median(rough)) > mask_sigma * noise).astype(np.uint8), size=2 * max(1, int(0.0005 * fs)) + 1).astype(bool)
        win = int(window_s * fs)
        n_win = int(math.ceil(n / win))
        centres = np.empty(n_win)
        coefs = np.empty((n_win, 2 * len(harmonics)))
        omega = 2 * np.pi * np.asarray([f_mains * k for k in harmonics]) / fs
        for w in range(n_win):
            check_cancelled(_acid_cancel)
            lo, hi = (w * win, min(n, (w + 1) * win))
            idx = np.arange(lo, hi)
            keep = ~mask[lo:hi]
            if keep.sum() < 100:
                keep = np.ones(hi - lo, dtype=bool)
            t = idx[keep].astype(np.float64)
            X = np.empty((len(t), 2 * len(harmonics)))
            for j, om in enumerate(omega):
                check_cancelled(_acid_cancel)
                X[:, 2 * j] = np.cos(om * t)
                X[:, 2 * j + 1] = np.sin(om * t)
            y = x[lo:hi][keep]
            coef, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
            coefs[w] = coef
            centres[w] = 0.5 * (lo + hi)
        t_all = np.arange(n, dtype=np.float64)
        line = np.zeros(n)
        for j, om in enumerate(omega):
            check_cancelled(_acid_cancel)
            a = np.interp(t_all, centres, coefs[:, 2 * j])
            b = np.interp(t_all, centres, coefs[:, 2 * j + 1])
            line += a * np.cos(om * t_all) + b * np.sin(om * t_all)
        return ((x - line).astype(np.float32), line.astype(np.float32))

    def _tail_tau_ms(y: np.ndarray, fs: float) -> float:
        """Exponential time constant of a pulse's recovery, fitted the same way add-on E fits real events."""
        p = int(np.argmax(y))
        tail = y[p:]
        below = np.flatnonzero(tail <= 0.2 * y[p])
        n_fit = int(below[0]) + int(0.0005 * fs) if len(below) else len(tail)
        t = np.arange(n_fit) / fs * 1000.0
        yy = tail[:n_fit]
        best = None
        for tau in np.geomspace(0.02, 3.0, 60):
            check_cancelled(_acid_cancel)
            e = np.exp(-t / tau)
            X = np.column_stack((e, np.ones_like(e)))
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            r = float(np.sum((yy - X @ coef) ** 2))
            if best is None or r < best[0]:
                best = (r, tau)
        return best[1]

    def bessel_cutoff_for_tau(target_tau_ms: float, fs: float, order: int=4) -> float:
        """Cut-off of a causal Bessel low-pass whose impulse response recovers with the target constant
    (the way the reference recording's acquisition filter does). Bisection on a synthetic impulse."""
        x = np.zeros(int(0.4 * fs))
        x[100] = 1.0
        lo, hi = (200.0, min(10000.0, 0.45 * fs))
        for _ in range(40):
            check_cancelled(_acid_cancel)
            mid = math.sqrt(lo * hi)
            sos = _bessel(order, mid, btype='low', fs=fs, output='sos', norm='mag')
            if _tail_tau_ms(_sosfilt(sos, x), fs) > target_tau_ms:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def match_bandwidth(sig: np.ndarray, fs: float, cutoff_hz: float, order: int=4) -> np.ndarray:
        """Causal Bessel low-pass (like the analog acquisition filter): no pre-ringing, the same shape
    for every event; the small group delay is common to all events and irrelevant downstream."""
        sos = _bessel(order, cutoff_hz, btype='low', fs=fs, output='sos', norm='mag')
        return _sosfilt(sos, np.asarray(sig, dtype=np.float64)).astype(np.float32)

    def baseline_rolling_median(sig: np.ndarray, fs: float, window_s: float, decim: int) -> tuple[np.ndarray, np.ndarray]:
        """2.0 s rolling median on a decimated copy, interpolated back (identical to the validated recipe).
    Returns (detrended float32, decimated baseline float32)."""
        n = len(sig)
        xs = np.asarray(sig[::decim], dtype=np.float32)
        w = int(window_s * fs / decim)
        if w % 2 == 0:
            w += 1
        b_dec = _median_filter(xs, size=w, mode='reflect')
        xp = np.arange(0, n, decim)[:len(b_dec)]
        baseline = np.interp(np.arange(n), xp, b_dec).astype(np.float32)
        return ((np.asarray(sig, dtype=np.float32) - baseline).astype(np.float32), b_dec.astype(np.float32))

    def quiet_psd(x: np.ndarray, fs: float, mask_sigma: float=4.0, max_seconds: float=120.0) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD over the longest event-free stretches of a detrended trace."""
        y = np.asarray(x[:int(max_seconds * fs)], dtype=np.float64)
        noise = 1.4826 * np.median(np.abs(y - np.median(y)))
        busy = _maximum_filter1d((np.abs(y - np.median(y)) > mask_sigma * noise).astype(np.uint8), size=2 * max(1, int(0.003 * fs)) + 1).astype(bool)
        quiet = y[~busy]
        nper = int(min(len(quiet), 2 ** 17))
        f, p = welch(quiet, fs=fs, nperseg=nper, noverlap=nper // 2, detrend='constant')
        return (f, p)

    def psd_summary(f: np.ndarray, p: np.ndarray, freqs, fs: float) -> dict:
        out = {}
        df = f[1] - f[0]
        total = float(np.trapezoid(p, f)) if hasattr(np, 'trapezoid') else float(np.trapz(p, f))
        line = 0.0
        for f0 in freqs:
            check_cancelled(_acid_cancel)
            band = (f >= f0 - 3 * df) & (f <= f0 + 3 * df)
            if band.any():
                floor = np.median(p[(f >= f0 - 20) & (f <= f0 + 20)])
                line += float(np.sum(np.clip(p[band] - floor, 0, None)) * df)
        out['line_power_fraction'] = line / max(total, 1e-30)
        lo, hi = ((f >= 10) & (f <= 1000), None)
        if lo.sum() > 10:
            slope = np.polyfit(np.log10(f[lo]), np.log10(p[lo]), 1)[0]
            out['one_over_f_exponent_10Hz_1kHz'] = float(-slope)
        ref = np.median(p[(f >= 300) & (f <= 600)]) if ((f >= 300) & (f <= 600)).any() else np.nan
        if np.isfinite(ref):
            above = (f > 600) & (p < 0.1 * ref)
            out['rolloff_10x_below_300_600Hz_hz'] = float(f[above][0]) if above.any() else np.nan
        # Was 'else np.nan' -- unlike 'total' above, this had NO numpy<2.0 fallback, so
        # it silently went NaN on any numpy without np.trapezoid (this deployment's
        # numpy 1.26.4 among them) instead of falling back to np.trapz like its sibling.
        _band_p, _band_f = (p[(f >= 20) & (f <= 5000)], f[(f >= 20) & (f <= 5000)])
        out['noise_rms_pA_20Hz_5kHz'] = float(np.sqrt(np.trapezoid(_band_p, _band_f))) if hasattr(np, 'trapezoid') else float(np.sqrt(np.trapz(_band_p, _band_f)))
        return out

    def preprocess_abf(abf_path: Path, tag: str, out_dir: Path, *, channel: int) -> dict:
        """Run stage 0 for one file and write <tag>.parquet, <tag>_baseline.parquet, <tag>_metadata.json."""
        print(f'[stage 0] {tag}: reading {abf_path.name}')
        data, fs_abf, fmt, header = read_abf_all_channels(abf_path)
        header = _sanitise_header(header)
        n_ch = data.shape[1]
        ch = channel if 0 <= channel < n_ch else n_ch - 1
        current = data[:, ch].astype(np.float32)
        del data
        n = len(current)
        print(f'[stage 0] reader: {fmt}; channel {ch} of {n_ch}; {n:,} samples @ {fs_abf:,} Hz ({n / fs_abf:.1f} s); raw mean {current.mean():+.2f} pA')
        step, on_grid = estimate_adc_step(current)
        print(f'[stage 0] ADC step {step:.6f} pA; on a single grid: {on_grid}')
        if LINE_NOISE_METHOD == 'notch':
            cleaned = line_noise_notch(current, fs_abf, NOTCH_FREQS, NOTCH_BW_HZ)
            f_mains = np.nan
            line_note = f'constant-bandwidth notches, {NOTCH_BW_HZ:g} Hz, one SOS cascade, zero-phase'
        elif LINE_NOISE_METHOD == 'sinusoid':
            f_mains, prominence = measure_mains_frequency(current, fs_abf, nominal=MAINS_NOMINAL_HZ)
            if prominence >= MAINS_MIN_PROMINENCE:
                harmonics = [int(round(f0 / MAINS_NOMINAL_HZ)) for f0 in NOTCH_FREQS]
                cleaned, _ = line_noise_sinusoid(current, fs_abf, harmonics, LINE_FIT_WINDOW_S, LINE_FIT_MASK_SIGMA, f_mains)
                line_note = f'windowed sinusoid fit at {f_mains:.4f} Hz (peak prominence {prominence:.0f}x) x {harmonics}, {LINE_FIT_WINDOW_S:g} s windows, event-masked'
            else:
                cleaned = current.copy()
                f_mains = np.nan
                line_note = f'none: no mains line detected near {MAINS_NOMINAL_HZ:g} Hz (peak prominence {prominence:.1f}x < {MAINS_MIN_PROMINENCE:g}x)'
        elif LINE_NOISE_METHOD == 'none':
            cleaned = current.copy()
            f_mains = np.nan
            line_note = 'none'
        else:
            raise ValueError("LINE_NOISE_METHOD must be 'notch', 'sinusoid' or 'none'")
        print(f'[stage 0] line noise: {line_note}')
        raw_detr, _ = baseline_rolling_median(current, fs_abf, BASELINE_WINDOW_S, BASELINE_DECIM)
        f_before, p_before = quiet_psd(raw_detr, fs_abf)
        qc_before = psd_summary(f_before, p_before, NOTCH_FREQS, fs_abf)
        rolloff = qc_before.get('rolloff_10x_below_300_600Hz_hz', np.nan)
        matched_hz = None
        if MATCH_TAU_MS is not None and tag != FAMILY_REFERENCE_TAG and np.isfinite(rolloff) and (rolloff > MATCH_MIN_ROLLOFF_HZ):
            matched_hz = bessel_cutoff_for_tau(MATCH_TAU_MS, fs_abf, BESSEL_ORDER)
            cleaned = match_bandwidth(cleaned, fs_abf, matched_hz, BESSEL_ORDER)
            print(f'[stage 0] bandwidth: roll-off {rolloff:.0f} Hz > {MATCH_MIN_ROLLOFF_HZ:g} Hz -> causal {BESSEL_ORDER}-pole Bessel at {matched_hz:.0f} Hz (recovery constant {MATCH_TAU_MS:g} ms, matching {FAMILY_REFERENCE_TAG})')
        else:
            print(f'[stage 0] bandwidth: roll-off {rolloff:.0f} Hz; no matching filter applied' + (' (reference tag)' if tag == FAMILY_REFERENCE_TAG else ''))
        detr, base_dec = baseline_rolling_median(cleaned, fs_abf, BASELINE_WINDOW_S, BASELINE_DECIM)
        print(f'[stage 0] baseline: rolling median {BASELINE_WINDOW_S:g} s, decim {BASELINE_DECIM}; detrended mean {detr.mean():+.4f} pA')
        f_after, p_after = quiet_psd(detr, fs_abf)
        qc_after = psd_summary(f_after, p_after, NOTCH_FREQS, fs_abf)
        x = detr.astype(np.float64)
        mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
        print(f"[stage 0] QC: MAD noise {mad:.2f} pA; line power fraction {qc_before['line_power_fraction']:.3f} -> {qc_after['line_power_fraction']:.3f}; 1/f exponent {qc_after.get('one_over_f_exponent_10Hz_1kHz', float('nan')):.2f}; roll-off (10x below the 300-600 Hz level) at {qc_after.get('rolloff_10x_below_300_600Hz_hz', float('nan')):.0f} Hz")
        fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
        t_plot = np.arange(0, n, max(1, n // 200000)) / fs_abf
        axes[0].plot(t_plot, current[::max(1, n // 200000)], lw=0.4, color='gray', alpha=0.7, label='raw')
        axes[0].plot(np.arange(len(base_dec)) * BASELINE_DECIM / fs_abf, base_dec, lw=1.0, color='tomato', label='baseline')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('pA')
        axes[0].set_title(f'{tag}: raw and baseline')
        axes[0].legend(fontsize=8)
        axes[1].plot(t_plot, detr[::max(1, n // 200000)], lw=0.4, color='steelblue')
        axes[1].axhline(0, color='k', lw=0.6, ls='--')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('pA')
        axes[1].set_title('Detrended')
        axes[2].loglog(f_before[1:], p_before[1:], lw=0.8, alpha=0.7, label='raw (minus baseline)')
        axes[2].loglog(f_after[1:], p_after[1:], lw=0.8, alpha=0.9, label='after line-noise removal' + (f' + Bessel {matched_hz:.0f} Hz' if matched_hz else ''))
        for f0 in NOTCH_FREQS:
            check_cancelled(_acid_cancel)
            axes[2].axvline(f0, color='k', lw=0.3, alpha=0.3)
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('PSD (pA²/Hz)')
        axes[2].set_title('Quiet-stretch PSD')
        axes[2].legend(fontsize=8)
        finish_figure(f'S0_{tag}_preprocessing')
        out_dir.mkdir(parents=True, exist_ok=True)
        pq_path = out_dir / f'{tag}.parquet'
        base_path = out_dir / f'{tag}_baseline.parquet'
        meta_path = out_dir / f'{tag}_metadata.json'
        if STORE_RAW_AS_CODES and on_grid:
            frame = pd.DataFrame({'current_raw_code': np.round(current / step).astype(np.int16), 'current_detr_pA': detr})
            raw_storage = 'int16_codes'
        else:
            frame = pd.DataFrame({'current_raw_pA': current, 'current_detr_pA': detr})
            raw_storage = 'float32'
        frame.to_parquet(pq_path, index=False, compression='zstd')
        pd.DataFrame({'baseline_pA': base_dec}).to_parquet(base_path, index=False, compression='zstd')
        meta = {'tag': tag, 'filename': abf_path.name, 'fs_hz': int(fs_abf), 'n_samples': int(n), 'duration_s': float(n / fs_abf), 'match_settings': [MATCH_TAU_MS, MATCH_MIN_ROLLOFF_HZ, BESSEL_ORDER, FAMILY_REFERENCE_TAG], 'abf_channel': int(ch), 'n_channels': int(n_ch), 'reader': fmt, 'abf_header': header, 'event_polarity': 1, 'raw_mean_pA': float(current.mean()), 'raw_std_pA': float(current.std()), 'detr_mean_pA': float(detr.mean()), 'detr_std_pA': float(detr.std()), 'adc_step_pA': step, 'raw_storage': raw_storage, 'line_noise_method': LINE_NOISE_METHOD, 'line_noise_note': line_note, 'notch_hz': list(NOTCH_FREQS), 'notch_bw_hz': NOTCH_BW_HZ if LINE_NOISE_METHOD == 'notch' else None, 'mains_frequency_hz': None if not np.isfinite(f_mains) else f_mains, 'mains_min_prominence': MAINS_MIN_PROMINENCE if LINE_NOISE_METHOD == 'sinusoid' else None, 'baseline_method': 'median', 'baseline_window_s': BASELINE_WINDOW_S, 'baseline_decim': BASELINE_DECIM, 'match_tau_ms': MATCH_TAU_MS if matched_hz is not None else None, 'matched_bandwidth_hz': matched_hz, 'raw_rolloff_hz': None if not np.isfinite(rolloff) else float(rolloff), 'qc_mad_noise_pA': mad, 'qc_psd_before': qc_before, 'qc_psd_after': qc_after, 'parquet_file': pq_path.name, 'baseline_file': base_path.name}
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f'[stage 0] wrote {pq_path.name} ({pq_path.stat().st_size / 1000000.0:.0f} MB), {base_path.name}, {meta_path.name}')
        return meta

    def load_stage0(tag: str, out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        """(detrended float32, raw float32, metadata) from the parquet cache, or from a legacy CSV."""
        pq_path, meta_path = (out_dir / f'{tag}.parquet', out_dir / f'{tag}_metadata.json')
        if pq_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            frame = pd.read_parquet(pq_path)
            detr = frame['current_detr_pA'].to_numpy(np.float32)
            if 'current_raw_code' in frame:
                raw_pA = frame['current_raw_code'].to_numpy(np.int16).astype(np.float32) * np.float32(meta['adc_step_pA'])
            else:
                raw_pA = frame['current_raw_pA'].to_numpy(np.float32)
            return (detr, raw_pA, meta)
        csv_path = out_dir / f'{tag}.csv.gz'
        if csv_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            frame = pd.read_csv(csv_path, compression='gzip', usecols=['current_detr_pA', 'current_raw_pA'])
            return (frame['current_detr_pA'].to_numpy(np.float32), frame['current_raw_pA'].to_numpy(np.float32), meta)
        raise FileNotFoundError(f'No stage-0 output for {tag} in {out_dir}')


def definitions_007():
    global _bessel, _maximum_filter1d, _median_filter, _sanitise_header, _sosfilt, _tail_tau_ms, baseline_rolling_median, bessel_cutoff_for_tau, estimate_adc_step, iirnotch, line_noise_notch, line_noise_sinusoid, load_stage0, match_bandwidth, measure_mains_frequency, preprocess_abf, psd_summary, quiet_psd, read_abf_all_channels, sosfiltfilt, tf2sos, welch
    from scipy.signal import iirnotch, sosfiltfilt, tf2sos, welch, bessel as _bessel, sosfilt as _sosfilt
    from .acid_incubation_numerics import median_filter_1d as _median_filter
    from scipy.ndimage import maximum_filter1d as _maximum_filter1d

    def read_abf_all_channels(path: Path) -> tuple[np.ndarray, int, str, dict]:
        from .acid_incubation_io import read_abf_all_channels as validated_read
        return validated_read(path, cancel=_acid_cancel)

    def _sanitise_header(header: dict) -> dict:
        """Keep only finite, plausible header values (unset ABF1 fields read as garbage)."""
        clean = {}
        for k, v in header.items():
            check_cancelled(_acid_cancel)
            vals = v if isinstance(v, list) else [v]
            ok = []
            for x in vals:
                check_cancelled(_acid_cancel)
                try:
                    xf = float(x)
                except (TypeError, ValueError):
                    ok.append(x)
                    continue
                if np.isfinite(xf) and abs(xf) < 10000000.0:
                    ok.append(x)
            if ok:
                clean[k] = ok if isinstance(v, list) else ok[0]
        enable = clean.get('nTelegraphEnable')
        enable = enable if isinstance(enable, list) else [enable] if enable is not None else []
        if enable and (not all((int(e) in (0, 1) for e in enable if isinstance(e, (int, float))))):
            clean.pop('nTelegraphEnable', None)
            clean.pop('fTelegraphFilter', None)
            clean['telegraph_note'] = 'telegraph fields unset in this ABF1 header'
        elif enable and (not any((int(e) == 1 for e in enable))):
            clean.pop('fTelegraphFilter', None)
            clean['telegraph_note'] = 'telegraph not enabled; the acquisition filter setting is not in the header'
        return clean

    def estimate_adc_step(x: np.ndarray, n_check: int=2000000) -> tuple[float, bool]:
        """Smallest non-zero spacing between distinct sample values; True if the whole trace sits on
    that grid (so it can be stored losslessly as int16 codes)."""
        sub = np.asarray(x[:n_check], dtype=np.float64)
        vals = np.unique(sub)
        if len(vals) < 3:
            return (np.nan, False)
        diffs = np.diff(vals)
        step = float(np.min(diffs[diffs > 1e-09]))
        codes = np.asarray(x, dtype=np.float64) / step
        on_grid = bool(np.max(np.abs(codes - np.round(codes))) < 0.001 and np.max(np.abs(codes)) < 32767)
        return (step, on_grid)

    def line_noise_notch(sig: np.ndarray, fs: float, freqs, bw_hz: float) -> np.ndarray:
        sos = np.vstack([tf2sos(*iirnotch(f0, Q=f0 / bw_hz, fs=fs)) for f0 in freqs])
        return sosfiltfilt(sos, sig.astype(np.float64)).astype(np.float32)

    def measure_mains_frequency(sig: np.ndarray, fs: float, nominal: float=60.0, span_hz: float=0.5, min_prominence: float=5.0) -> tuple[float, float]:
        """(frequency, prominence) of the mains line: the spectral peak within +/- span_hz of the nominal
    frequency, measured against the median spectrum in the surrounding +/- 5 Hz. A prominence below
    min_prominence (or a peak sitting on the search-band edge) means there is no detectable mains line
    and the caller skips the line-noise step rather than fitting noise."""
        x = np.asarray(sig[:min(len(sig), int(fs * 300))], dtype=np.float64)
        x = x - x.mean()
        n = len(x)
        spec = np.abs(np.fft.rfft(x * np.hanning(n)))
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        band = (f >= nominal - span_hz) & (f <= nominal + span_hz)
        if not band.any():
            return (float(nominal), 0.0)
        k = int(np.argmax(spec[band]))
        f_peak = float(f[band][k])
        surround = (f >= nominal - 5.0) & (f <= nominal + 5.0) & ~band
        prominence = float(spec[band][k] / max(np.median(spec[surround]), 1e-30)) if surround.any() else 0.0
        on_edge = k == 0 or k == band.sum() - 1
        if on_edge:
            prominence = 0.0
        return (f_peak, prominence)

    def line_noise_sinusoid(sig: np.ndarray, fs: float, harmonics, window_s: float, mask_sigma: float, f_mains: float) -> tuple[np.ndarray, np.ndarray]:
        """Fit a_k cos + b_k sin at each harmonic of the measured mains frequency in consecutive windows,
    on samples that are not inside an excursion, interpolate the coefficients between window centres,
    synthesise the line noise and subtract it. Returns (cleaned, synthesised_line_noise)."""
        x = np.asarray(sig, dtype=np.float64)
        n = len(x)
        rough = x - _median_filter(x[::10], size=int(0.5 * fs / 10) | 1, mode='reflect').repeat(10)[:n]
        noise = 1.4826 * np.median(np.abs(rough - np.median(rough)))
        mask = _maximum_filter1d((np.abs(rough - np.median(rough)) > mask_sigma * noise).astype(np.uint8), size=2 * max(1, int(0.0005 * fs)) + 1).astype(bool)
        win = int(window_s * fs)
        n_win = int(math.ceil(n / win))
        centres = np.empty(n_win)
        coefs = np.empty((n_win, 2 * len(harmonics)))
        omega = 2 * np.pi * np.asarray([f_mains * k for k in harmonics]) / fs
        for w in range(n_win):
            check_cancelled(_acid_cancel)
            lo, hi = (w * win, min(n, (w + 1) * win))
            idx = np.arange(lo, hi)
            keep = ~mask[lo:hi]
            if keep.sum() < 100:
                keep = np.ones(hi - lo, dtype=bool)
            t = idx[keep].astype(np.float64)
            X = np.empty((len(t), 2 * len(harmonics)))
            for j, om in enumerate(omega):
                check_cancelled(_acid_cancel)
                X[:, 2 * j] = np.cos(om * t)
                X[:, 2 * j + 1] = np.sin(om * t)
            y = x[lo:hi][keep]
            coef, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
            coefs[w] = coef
            centres[w] = 0.5 * (lo + hi)
        t_all = np.arange(n, dtype=np.float64)
        line = np.zeros(n)
        for j, om in enumerate(omega):
            check_cancelled(_acid_cancel)
            a = np.interp(t_all, centres, coefs[:, 2 * j])
            b = np.interp(t_all, centres, coefs[:, 2 * j + 1])
            line += a * np.cos(om * t_all) + b * np.sin(om * t_all)
        return ((x - line).astype(np.float32), line.astype(np.float32))

    def _tail_tau_ms(y: np.ndarray, fs: float) -> float:
        """Exponential time constant of a pulse's recovery, fitted the same way add-on E fits real events."""
        p = int(np.argmax(y))
        tail = y[p:]
        below = np.flatnonzero(tail <= 0.2 * y[p])
        n_fit = int(below[0]) + int(0.0005 * fs) if len(below) else len(tail)
        t = np.arange(n_fit) / fs * 1000.0
        yy = tail[:n_fit]
        best = None
        for tau in np.geomspace(0.02, 3.0, 60):
            check_cancelled(_acid_cancel)
            e = np.exp(-t / tau)
            X = np.column_stack((e, np.ones_like(e)))
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            r = float(np.sum((yy - X @ coef) ** 2))
            if best is None or r < best[0]:
                best = (r, tau)
        return best[1]

    def bessel_cutoff_for_tau(target_tau_ms: float, fs: float, order: int=4) -> float:
        """Cut-off of a causal Bessel low-pass whose impulse response recovers with the target constant
    (the way the reference recording's acquisition filter does). Bisection on a synthetic impulse."""
        x = np.zeros(int(0.4 * fs))
        x[100] = 1.0
        lo, hi = (200.0, min(10000.0, 0.45 * fs))
        for _ in range(40):
            check_cancelled(_acid_cancel)
            mid = math.sqrt(lo * hi)
            sos = _bessel(order, mid, btype='low', fs=fs, output='sos', norm='mag')
            if _tail_tau_ms(_sosfilt(sos, x), fs) > target_tau_ms:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def match_bandwidth(sig: np.ndarray, fs: float, cutoff_hz: float, order: int=4) -> np.ndarray:
        """Causal Bessel low-pass (like the analog acquisition filter): no pre-ringing, the same shape
    for every event; the small group delay is common to all events and irrelevant downstream."""
        sos = _bessel(order, cutoff_hz, btype='low', fs=fs, output='sos', norm='mag')
        return _sosfilt(sos, np.asarray(sig, dtype=np.float64)).astype(np.float32)

    def baseline_rolling_median(sig: np.ndarray, fs: float, window_s: float, decim: int) -> tuple[np.ndarray, np.ndarray]:
        """2.0 s rolling median on a decimated copy, interpolated back (identical to the validated recipe).
    Returns (detrended float32, decimated baseline float32)."""
        n = len(sig)
        xs = np.asarray(sig[::decim], dtype=np.float32)
        w = int(window_s * fs / decim)
        if w % 2 == 0:
            w += 1
        b_dec = _median_filter(xs, size=w, mode='reflect')
        xp = np.arange(0, n, decim)[:len(b_dec)]
        baseline = np.interp(np.arange(n), xp, b_dec).astype(np.float32)
        return ((np.asarray(sig, dtype=np.float32) - baseline).astype(np.float32), b_dec.astype(np.float32))

    def quiet_psd(x: np.ndarray, fs: float, mask_sigma: float=4.0, max_seconds: float=120.0) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD over the longest event-free stretches of a detrended trace."""
        y = np.asarray(x[:int(max_seconds * fs)], dtype=np.float64)
        noise = 1.4826 * np.median(np.abs(y - np.median(y)))
        busy = _maximum_filter1d((np.abs(y - np.median(y)) > mask_sigma * noise).astype(np.uint8), size=2 * max(1, int(0.003 * fs)) + 1).astype(bool)
        quiet = y[~busy]
        nper = int(min(len(quiet), 2 ** 17))
        f, p = welch(quiet, fs=fs, nperseg=nper, noverlap=nper // 2, detrend='constant')
        return (f, p)

    def psd_summary(f: np.ndarray, p: np.ndarray, freqs, fs: float) -> dict:
        out = {}
        df = f[1] - f[0]
        total = float(np.trapezoid(p, f)) if hasattr(np, 'trapezoid') else float(np.trapz(p, f))
        line = 0.0
        for f0 in freqs:
            check_cancelled(_acid_cancel)
            band = (f >= f0 - 3 * df) & (f <= f0 + 3 * df)
            if band.any():
                floor = np.median(p[(f >= f0 - 20) & (f <= f0 + 20)])
                line += float(np.sum(np.clip(p[band] - floor, 0, None)) * df)
        out['line_power_fraction'] = line / max(total, 1e-30)
        lo, hi = ((f >= 10) & (f <= 1000), None)
        if lo.sum() > 10:
            slope = np.polyfit(np.log10(f[lo]), np.log10(p[lo]), 1)[0]
            out['one_over_f_exponent_10Hz_1kHz'] = float(-slope)
        ref = np.median(p[(f >= 300) & (f <= 600)]) if ((f >= 300) & (f <= 600)).any() else np.nan
        if np.isfinite(ref):
            above = (f > 600) & (p < 0.1 * ref)
            out['rolloff_10x_below_300_600Hz_hz'] = float(f[above][0]) if above.any() else np.nan
        # Was 'else np.nan' -- unlike 'total' above, this had NO numpy<2.0 fallback, so
        # it silently went NaN on any numpy without np.trapezoid (this deployment's
        # numpy 1.26.4 among them) instead of falling back to np.trapz like its sibling.
        _band_p, _band_f = (p[(f >= 20) & (f <= 5000)], f[(f >= 20) & (f <= 5000)])
        out['noise_rms_pA_20Hz_5kHz'] = float(np.sqrt(np.trapezoid(_band_p, _band_f))) if hasattr(np, 'trapezoid') else float(np.sqrt(np.trapz(_band_p, _band_f)))
        return out

    def preprocess_abf(abf_path: Path, tag: str, out_dir: Path, *, channel: int) -> dict:
        """Run stage 0 for one file and write <tag>.parquet, <tag>_baseline.parquet, <tag>_metadata.json."""
        print(f'[stage 0] {tag}: reading {abf_path.name}')
        data, fs_abf, fmt, header = read_abf_all_channels(abf_path)
        header = _sanitise_header(header)
        n_ch = data.shape[1]
        ch = channel if 0 <= channel < n_ch else n_ch - 1
        current = data[:, ch].astype(np.float32)
        del data
        n = len(current)
        print(f'[stage 0] reader: {fmt}; channel {ch} of {n_ch}; {n:,} samples @ {fs_abf:,} Hz ({n / fs_abf:.1f} s); raw mean {current.mean():+.2f} pA')
        step, on_grid = estimate_adc_step(current)
        print(f'[stage 0] ADC step {step:.6f} pA; on a single grid: {on_grid}')
        if LINE_NOISE_METHOD == 'notch':
            cleaned = line_noise_notch(current, fs_abf, NOTCH_FREQS, NOTCH_BW_HZ)
            f_mains = np.nan
            line_note = f'constant-bandwidth notches, {NOTCH_BW_HZ:g} Hz, one SOS cascade, zero-phase'
        elif LINE_NOISE_METHOD == 'sinusoid':
            f_mains, prominence = measure_mains_frequency(current, fs_abf, nominal=MAINS_NOMINAL_HZ)
            if prominence >= MAINS_MIN_PROMINENCE:
                harmonics = [int(round(f0 / MAINS_NOMINAL_HZ)) for f0 in NOTCH_FREQS]
                cleaned, _ = line_noise_sinusoid(current, fs_abf, harmonics, LINE_FIT_WINDOW_S, LINE_FIT_MASK_SIGMA, f_mains)
                line_note = f'windowed sinusoid fit at {f_mains:.4f} Hz (peak prominence {prominence:.0f}x) x {harmonics}, {LINE_FIT_WINDOW_S:g} s windows, event-masked'
            else:
                cleaned = current.copy()
                f_mains = np.nan
                line_note = f'none: no mains line detected near {MAINS_NOMINAL_HZ:g} Hz (peak prominence {prominence:.1f}x < {MAINS_MIN_PROMINENCE:g}x)'
        elif LINE_NOISE_METHOD == 'none':
            cleaned = current.copy()
            f_mains = np.nan
            line_note = 'none'
        else:
            raise ValueError("LINE_NOISE_METHOD must be 'notch', 'sinusoid' or 'none'")
        print(f'[stage 0] line noise: {line_note}')
        raw_detr, _ = baseline_rolling_median(current, fs_abf, BASELINE_WINDOW_S, BASELINE_DECIM)
        f_before, p_before = quiet_psd(raw_detr, fs_abf)
        qc_before = psd_summary(f_before, p_before, NOTCH_FREQS, fs_abf)
        rolloff = qc_before.get('rolloff_10x_below_300_600Hz_hz', np.nan)
        matched_hz = None
        if MATCH_TAU_MS is not None and tag != FAMILY_REFERENCE_TAG and np.isfinite(rolloff) and (rolloff > MATCH_MIN_ROLLOFF_HZ):
            matched_hz = bessel_cutoff_for_tau(MATCH_TAU_MS, fs_abf, BESSEL_ORDER)
            cleaned = match_bandwidth(cleaned, fs_abf, matched_hz, BESSEL_ORDER)
            print(f'[stage 0] bandwidth: roll-off {rolloff:.0f} Hz > {MATCH_MIN_ROLLOFF_HZ:g} Hz -> causal {BESSEL_ORDER}-pole Bessel at {matched_hz:.0f} Hz (recovery constant {MATCH_TAU_MS:g} ms, matching {FAMILY_REFERENCE_TAG})')
        else:
            print(f'[stage 0] bandwidth: roll-off {rolloff:.0f} Hz; no matching filter applied' + (' (reference tag)' if tag == FAMILY_REFERENCE_TAG else ''))
        detr, base_dec = baseline_rolling_median(cleaned, fs_abf, BASELINE_WINDOW_S, BASELINE_DECIM)
        print(f'[stage 0] baseline: rolling median {BASELINE_WINDOW_S:g} s, decim {BASELINE_DECIM}; detrended mean {detr.mean():+.4f} pA')
        f_after, p_after = quiet_psd(detr, fs_abf)
        qc_after = psd_summary(f_after, p_after, NOTCH_FREQS, fs_abf)
        x = detr.astype(np.float64)
        mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
        print(f"[stage 0] QC: MAD noise {mad:.2f} pA; line power fraction {qc_before['line_power_fraction']:.3f} -> {qc_after['line_power_fraction']:.3f}; 1/f exponent {qc_after.get('one_over_f_exponent_10Hz_1kHz', float('nan')):.2f}; roll-off (10x below the 300-600 Hz level) at {qc_after.get('rolloff_10x_below_300_600Hz_hz', float('nan')):.0f} Hz")
        fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
        t_plot = np.arange(0, n, max(1, n // 200000)) / fs_abf
        axes[0].plot(t_plot, current[::max(1, n // 200000)], lw=0.4, color='gray', alpha=0.7, label='raw')
        axes[0].plot(np.arange(len(base_dec)) * BASELINE_DECIM / fs_abf, base_dec, lw=1.0, color='tomato', label='baseline')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('pA')
        axes[0].set_title(f'{tag}: raw and baseline')
        axes[0].legend(fontsize=8)
        axes[1].plot(t_plot, detr[::max(1, n // 200000)], lw=0.4, color='steelblue')
        axes[1].axhline(0, color='k', lw=0.6, ls='--')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('pA')
        axes[1].set_title('Detrended')
        axes[2].loglog(f_before[1:], p_before[1:], lw=0.8, alpha=0.7, label='raw (minus baseline)')
        axes[2].loglog(f_after[1:], p_after[1:], lw=0.8, alpha=0.9, label='after line-noise removal' + (f' + Bessel {matched_hz:.0f} Hz' if matched_hz else ''))
        for f0 in NOTCH_FREQS:
            check_cancelled(_acid_cancel)
            axes[2].axvline(f0, color='k', lw=0.3, alpha=0.3)
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('PSD (pA²/Hz)')
        axes[2].set_title('Quiet-stretch PSD')
        axes[2].legend(fontsize=8)
        finish_figure(f'S0_{tag}_preprocessing')
        out_dir.mkdir(parents=True, exist_ok=True)
        pq_path = out_dir / f'{tag}.parquet'
        base_path = out_dir / f'{tag}_baseline.parquet'
        meta_path = out_dir / f'{tag}_metadata.json'
        if STORE_RAW_AS_CODES and on_grid:
            frame = pd.DataFrame({'current_raw_code': np.round(current / step).astype(np.int16), 'current_detr_pA': detr})
            raw_storage = 'int16_codes'
        else:
            frame = pd.DataFrame({'current_raw_pA': current, 'current_detr_pA': detr})
            raw_storage = 'float32'
        frame.to_parquet(pq_path, index=False, compression='zstd')
        pd.DataFrame({'baseline_pA': base_dec}).to_parquet(base_path, index=False, compression='zstd')
        meta = {'tag': tag, 'filename': abf_path.name, 'fs_hz': int(fs_abf), 'n_samples': int(n), 'duration_s': float(n / fs_abf), 'match_settings': [MATCH_TAU_MS, MATCH_MIN_ROLLOFF_HZ, BESSEL_ORDER, FAMILY_REFERENCE_TAG], 'abf_channel': int(ch), 'n_channels': int(n_ch), 'reader': fmt, 'abf_header': header, 'event_polarity': 1, 'raw_mean_pA': float(current.mean()), 'raw_std_pA': float(current.std()), 'detr_mean_pA': float(detr.mean()), 'detr_std_pA': float(detr.std()), 'adc_step_pA': step, 'raw_storage': raw_storage, 'line_noise_method': LINE_NOISE_METHOD, 'line_noise_note': line_note, 'notch_hz': list(NOTCH_FREQS), 'notch_bw_hz': NOTCH_BW_HZ if LINE_NOISE_METHOD == 'notch' else None, 'mains_frequency_hz': None if not np.isfinite(f_mains) else f_mains, 'mains_min_prominence': MAINS_MIN_PROMINENCE if LINE_NOISE_METHOD == 'sinusoid' else None, 'baseline_method': 'median', 'baseline_window_s': BASELINE_WINDOW_S, 'baseline_decim': BASELINE_DECIM, 'match_tau_ms': MATCH_TAU_MS if matched_hz is not None else None, 'matched_bandwidth_hz': matched_hz, 'raw_rolloff_hz': None if not np.isfinite(rolloff) else float(rolloff), 'qc_mad_noise_pA': mad, 'qc_psd_before': qc_before, 'qc_psd_after': qc_after, 'parquet_file': pq_path.name, 'baseline_file': base_path.name}
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f'[stage 0] wrote {pq_path.name} ({pq_path.stat().st_size / 1000000.0:.0f} MB), {base_path.name}, {meta_path.name}')
        return meta

    def load_stage0(tag: str, out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        """(detrended float32, raw float32, metadata) from the parquet cache, or from a legacy CSV."""
        pq_path, meta_path = (out_dir / f'{tag}.parquet', out_dir / f'{tag}_metadata.json')
        if pq_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            frame = pd.read_parquet(pq_path)
            detr = frame['current_detr_pA'].to_numpy(np.float32)
            if 'current_raw_code' in frame:
                raw_pA = frame['current_raw_code'].to_numpy(np.int16).astype(np.float32) * np.float32(meta['adc_step_pA'])
            else:
                raw_pA = frame['current_raw_pA'].to_numpy(np.float32)
            return (detr, raw_pA, meta)
        csv_path = out_dir / f'{tag}.csv.gz'
        if csv_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            frame = pd.read_csv(csv_path, compression='gzip', usecols=['current_detr_pA', 'current_raw_pA'])
            return (frame['current_detr_pA'].to_numpy(np.float32), frame['current_raw_pA'].to_numpy(np.float32), meta)
        raise FileNotFoundError(f'No stage-0 output for {tag} in {out_dir}')


# Original notebook cell 8
def stage_008():
    global ABF_PATH, CSV_PATH, INPUT_FINGERPRINTS, META_PATH, PARQUET_PATH, _abf_available, _cache_ok, _data_file, _detr, _hdr, _old, _qc, _raw_pA, _ro, _run_stage0, _stage0_settings, duration_s, fs, meta, metadata_i0, n_samples, partial_fingerprint, polarity, raw, raw_source, signal
    check_cancelled(_acid_cancel)

    def partial_fingerprint(path: Path, chunk_bytes: int=1048576) -> dict:
        h = hashlib.sha256()
        size = path.stat().st_size
        with path.open('rb') as handle:
            h.update(handle.read(chunk_bytes))
            if size > chunk_bytes:
                handle.seek(max(0, size - chunk_bytes))
                h.update(handle.read(chunk_bytes))
        h.update(str(size).encode())
        return {'path': str(path), 'size_bytes': size, 'partial_sha256': h.hexdigest()}
    _stage0_settings = {'line_noise_method': LINE_NOISE_METHOD, 'notch_hz': list(NOTCH_FREQS), 'notch_bw_hz': NOTCH_BW_HZ if LINE_NOISE_METHOD == 'notch' else None, 'baseline_window_s': BASELINE_WINDOW_S, 'baseline_decim': BASELINE_DECIM, 'mains_min_prominence': MAINS_MIN_PROMINENCE if LINE_NOISE_METHOD == 'sinusoid' else None, 'match_settings': [MATCH_TAU_MS, MATCH_MIN_ROLLOFF_HZ, BESSEL_ORDER, FAMILY_REFERENCE_TAG]}
    ABF_PATH = ABF_DIR / ABF_FILES[TAG] if TAG in ABF_FILES else None
    PARQUET_PATH = DATA_DIR / f'{TAG}.parquet'
    META_PATH = DATA_DIR / f'{TAG}_metadata.json'
    CSV_PATH = DATA_DIR / f'{TAG}.csv.gz'
    _cache_ok = False
    _abf_available = ABF_PATH is not None and ABF_PATH.is_file()
    if META_PATH.is_file() and PARQUET_PATH.is_file():
        _old = json.loads(META_PATH.read_text())
        _cache_ok = all((_old.get(k) == v for k, v in _stage0_settings.items()))
        if not _cache_ok:
            print('Stage-0 cache exists but was made with different settings:', {k: (_old.get(k), v) for k, v in _stage0_settings.items() if _old.get(k) != v})
    elif META_PATH.is_file() and CSV_PATH.is_file():
        _cache_ok = not _abf_available
        print('Legacy CSV cache found' + (': ABF present, so stage 0 will re-preprocess it' if _abf_available else ' and no ABF: using it as is (note: made with the old notch cascade)'))
    _run_stage0 = PREPROCESS == 'force' or (PREPROCESS == 'auto' and (not _cache_ok))
    if _run_stage0:
        if not _abf_available:
            if _cache_ok:
                print(f'ABF not found ({ABF_PATH}); using the existing cache.')
                _run_stage0 = False
            else:
                raise FileNotFoundError(f'Stage 0 needs the ABF ({ABF_PATH}) or a cache in {DATA_DIR}')
    if _run_stage0:
        meta = preprocess_abf(ABF_PATH, TAG, DATA_DIR, channel=FILE_CHANNELS.get(TAG, ABF_CHANNEL))
    elif PREPROCESS == 'skip' and (not _cache_ok):
        raise FileNotFoundError(f"PREPROCESS='skip' but no stage-0 output for {TAG} in {DATA_DIR}")
    _detr, _raw_pA, meta = load_stage0(TAG, DATA_DIR)
    fs = float(meta['fs_hz'])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'Invalid sampling rate: {fs}')
    polarity = int(meta.get('event_polarity', 1) or 1)
    if polarity not in (-1, 1):
        raise ValueError(f'event_polarity must be -1 or +1, got {polarity}')
    signal = (polarity * _detr).astype(np.float32, copy=False)
    raw = (polarity * _raw_pA).astype(np.float32, copy=False)
    raw_source = meta.get('raw_storage', 'current_raw_pA')
    del _detr, _raw_pA
    n_samples = len(signal)
    duration_s = n_samples / fs
    try:
        metadata_i0 = abs(float(meta.get('raw_mean_pA', np.nan)))
    except (TypeError, ValueError):
        metadata_i0 = np.nan
    _data_file = PARQUET_PATH if PARQUET_PATH.is_file() else CSV_PATH
    INPUT_FINGERPRINTS = {'data': partial_fingerprint(_data_file), 'metadata': partial_fingerprint(META_PATH), 'abf': partial_fingerprint(ABF_PATH) if ABF_PATH is not None and ABF_PATH.is_file() else None, 'model': partial_fingerprint(STUDENT_MODEL_PATH) if RUN_STUDENT else None}
    if RUN_STUDENT and (not HAVE_TF):
        raise RuntimeError('RUN_STUDENT = True requires TensorFlow')
    if RUN_STUDENT and (not STUDENT_MODEL_PATH.is_file()):
        raise FileNotFoundError(f'Missing frozen student model: {STUDENT_MODEL_PATH}')
    print(f'Loaded {TAG}: {n_samples:,} samples @ {fs:,.1f} Hz ({duration_s:.2f} s) from {_data_file.name}')
    print('Event polarity orientation:', polarity, '| raw storage:', raw_source, '| ADC step:', meta.get('adc_step_pA', 'n/a'))
    print('Line-noise removal:', meta.get('line_noise_note', meta.get('notch_hz', 'n/a')))
    _ro = meta.get('raw_rolloff_hz')
    print('Bandwidth:', (f'raw roll-off {_ro:.0f} Hz; ' if isinstance(_ro, (int, float)) else 'raw roll-off n/a; ') + (f"matched with Bessel {meta['matched_bandwidth_hz']:.0f} Hz to tau {meta['match_tau_ms']} ms" if meta.get('matched_bandwidth_hz') else 'no matching filter'))
    print('Metadata |I0| fallback:', metadata_i0 if np.isfinite(metadata_i0) else 'unavailable')
    _hdr = meta.get('abf_header', {}) or {}
    if _hdr:
        print('ABF header:', {k: _hdr[k] for k in ('nTelegraphEnable', 'fTelegraphFilter', 'fADCRange', 'lADCResolution') if k in _hdr})
    _qc = meta.get('qc_psd_after') or {}
    if _qc:
        print('Quiet-PSD QC:', {k: round(v, 3) if isinstance(v, float) else v for k, v in _qc.items()})


def definitions_008():
    global partial_fingerprint

    def partial_fingerprint(path: Path, chunk_bytes: int=1048576) -> dict:
        h = hashlib.sha256()
        size = path.stat().st_size
        with path.open('rb') as handle:
            h.update(handle.read(chunk_bytes))
            if size > chunk_bytes:
                handle.seek(max(0, size - chunk_bytes))
                h.update(handle.read(chunk_bytes))
        h.update(str(size).encode())
        return {'path': str(path), 'size_bytes': size, 'partial_sha256': h.hexdigest()}


# Original notebook cell 10
def stage_010():
    global dwell_ms, extract_hysteresis_regions, filter_min_duration, inference_starts, mad_std, merge_regions, probability_regions, regions_to_mask, robust_normalize, robust_quiet_stats, student_predict, threshold_regions
    check_cancelled(_acid_cancel)

    def mad_std(x: np.ndarray, center: float | None=None) -> float:
        x = np.asarray(x, dtype=float)
        if center is None:
            center = float(np.median(x))
        value = 1.4826 * float(np.median(np.abs(x - center)))
        return value if np.isfinite(value) and value > 0 else 1.0

    def robust_quiet_stats(x: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Iterative event-excluded baseline and noise estimate."""
        x = np.asarray(x, np.float32)
        m0 = float(np.median(x))
        s0 = mad_std(x, m0)
        keep0 = np.abs(x - m0) < 5.0 * s0
        m1 = float(np.median(x[keep0])) if keep0.any() else m0
        s1 = mad_std(x[keep0], m1) if keep0.any() else s0
        quiet = np.abs(x - m1) < 3.0 * s1
        if quiet.sum() < 100:
            quiet = keep0
        baseline = float(np.median(x[quiet])) if quiet.any() else m1
        noise = mad_std(x[quiet], baseline) if quiet.any() else s1
        return (baseline, noise, quiet)

    def robust_normalize(x: np.ndarray, clip_sigma: float=30.0) -> np.ndarray:
        """Match the student's training/inference normalization."""
        baseline, noise, _ = robust_quiet_stats(x)
        z = (np.asarray(x, np.float32) - baseline) / max(noise, 1e-12)
        return (np.clip(z, -clip_sigma, clip_sigma) / clip_sigma).astype(np.float32)

    def inference_starts(n: int, window: int, step: int) -> list[int]:
        """Cover the complete recording, including the final non-aligned window."""
        if n < window:
            raise ValueError(f'signal length {n} is shorter than model window {window}')
        starts = list(range(0, n - window + 1, step))
        final_start = n - window
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def student_predict(model: 'tf.keras.Model', x: np.ndarray, *, step: int | None=None, batch_size: int=64) -> np.ndarray:
        window = int(model.input_shape[1])
        step = int(step or window // 2)
        if step <= 0 or step > window:
            raise ValueError('step must be in [1, window]')
        x_norm = robust_normalize(x)
        starts = inference_starts(len(x_norm), window, step)
        pred_sum = np.zeros(len(x_norm), np.float32)
        coverage = np.zeros(len(x_norm), np.float32)
        batch = []
        batch_starts = []

        def flush() -> None:
            if not batch:
                return
            xb = np.stack(batch).astype(np.float32, copy=False)
            yb = np.asarray(model.predict(xb, verbose=0))[..., 0]
            for start, y in zip(batch_starts, yb):
                check_cancelled(_acid_cancel)
                pred_sum[start:start + window] += y
                coverage[start:start + window] += 1.0
            batch.clear()
            batch_starts.clear()
        for start in tqdm(starts, desc='Student inference'):
            check_cancelled(_acid_cancel)
            batch.append(x_norm[start:start + window, None])
            batch_starts.append(start)
            if len(batch) >= batch_size:
                flush()
        flush()
        if np.any(coverage == 0):
            raise RuntimeError('Inference left uncovered samples; check inference_starts().')
        return pred_sum / coverage

    def extract_hysteresis_regions(values: np.ndarray, enter: float, exit_: float) -> list[tuple[int, int]]:
        values = np.asarray(values)
        above_exit = values >= exit_
        diff = np.diff(above_exit.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)
        regions = []
        for start, stop in zip(starts, stops):
            check_cancelled(_acid_cancel)
            if np.any(values[start:stop] >= enter):
                regions.append((int(start), int(stop - 1)))
        return regions

    def merge_regions(regions: list[tuple[int, int]], fs: float, gap_ms: float) -> list[tuple[int, int]]:
        if not regions:
            return []
        max_gap = int(round(gap_ms * fs / 1000.0))
        regions = sorted(regions)
        merged = []
        cur_s, cur_e = regions[0]
        for start, end in regions[1:]:
            check_cancelled(_acid_cancel)
            gap_samples = start - cur_e - 1
            if gap_samples <= max_gap:
                cur_e = max(cur_e, end)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = (start, end)
        merged.append((cur_s, cur_e))
        return merged

    def filter_min_duration(regions: list[tuple[int, int]], fs: float, min_ms: float) -> list[tuple[int, int]]:
        min_samples = max(1, int(math.ceil(min_ms * fs / 1000.0)))
        return [(s, e) for s, e in regions if e - s + 1 >= min_samples]

    def probability_regions(probability: np.ndarray, *, fs: float, enter: float, exit_: float, smoothing_kernel: int, merge_gap_ms: float, min_event_ms: float) -> list[tuple[int, int]]:
        kernel = max(1, int(smoothing_kernel))
        if kernel % 2 == 0:
            kernel += 1
        smooth = median_filter(np.asarray(probability, np.float32), size=kernel, mode='nearest')
        regions = extract_hysteresis_regions(smooth, enter=enter, exit_=exit_)
        regions = merge_regions(regions, fs=fs, gap_ms=merge_gap_ms)
        return filter_min_duration(regions, fs=fs, min_ms=min_event_ms)

    def threshold_regions(x: np.ndarray, *, fs: float, baseline: float, noise: float, enter_sigma: float, exit_sigma: float, merge_gap_ms: float, min_event_ms: float) -> list[tuple[int, int]]:
        """Model-free downward-event hysteresis on the oriented detrended trace."""
        enter_level = baseline - enter_sigma * noise
        exit_level = baseline - exit_sigma * noise
        below_exit = np.asarray(x) <= exit_level
        diff = np.diff(below_exit.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)
        regions = [(int(s), int(t - 1)) for s, t in zip(starts, stops) if np.any(x[s:t] <= enter_level)]
        regions = merge_regions(regions, fs=fs, gap_ms=merge_gap_ms)
        return filter_min_duration(regions, fs=fs, min_ms=min_event_ms)

    def regions_to_mask(length: int, regions: list[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(length, dtype=bool)
        for start, end in regions:
            check_cancelled(_acid_cancel)
            mask[start:end + 1] = True
        return mask

    def dwell_ms(region: tuple[int, int], fs: float) -> float:
        start, end = region
        return (end - start + 1) / fs * 1000.0


def definitions_010():
    global dwell_ms, extract_hysteresis_regions, filter_min_duration, inference_starts, mad_std, merge_regions, probability_regions, regions_to_mask, robust_normalize, robust_quiet_stats, student_predict, threshold_regions

    def mad_std(x: np.ndarray, center: float | None=None) -> float:
        x = np.asarray(x, dtype=float)
        if center is None:
            center = float(np.median(x))
        value = 1.4826 * float(np.median(np.abs(x - center)))
        return value if np.isfinite(value) and value > 0 else 1.0

    def robust_quiet_stats(x: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Iterative event-excluded baseline and noise estimate."""
        x = np.asarray(x, np.float32)
        m0 = float(np.median(x))
        s0 = mad_std(x, m0)
        keep0 = np.abs(x - m0) < 5.0 * s0
        m1 = float(np.median(x[keep0])) if keep0.any() else m0
        s1 = mad_std(x[keep0], m1) if keep0.any() else s0
        quiet = np.abs(x - m1) < 3.0 * s1
        if quiet.sum() < 100:
            quiet = keep0
        baseline = float(np.median(x[quiet])) if quiet.any() else m1
        noise = mad_std(x[quiet], baseline) if quiet.any() else s1
        return (baseline, noise, quiet)

    def robust_normalize(x: np.ndarray, clip_sigma: float=30.0) -> np.ndarray:
        """Match the student's training/inference normalization."""
        baseline, noise, _ = robust_quiet_stats(x)
        z = (np.asarray(x, np.float32) - baseline) / max(noise, 1e-12)
        return (np.clip(z, -clip_sigma, clip_sigma) / clip_sigma).astype(np.float32)

    def inference_starts(n: int, window: int, step: int) -> list[int]:
        """Cover the complete recording, including the final non-aligned window."""
        if n < window:
            raise ValueError(f'signal length {n} is shorter than model window {window}')
        starts = list(range(0, n - window + 1, step))
        final_start = n - window
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def student_predict(model: 'tf.keras.Model', x: np.ndarray, *, step: int | None=None, batch_size: int=64) -> np.ndarray:
        window = int(model.input_shape[1])
        step = int(step or window // 2)
        if step <= 0 or step > window:
            raise ValueError('step must be in [1, window]')
        x_norm = robust_normalize(x)
        starts = inference_starts(len(x_norm), window, step)
        pred_sum = np.zeros(len(x_norm), np.float32)
        coverage = np.zeros(len(x_norm), np.float32)
        batch = []
        batch_starts = []

        def flush() -> None:
            if not batch:
                return
            xb = np.stack(batch).astype(np.float32, copy=False)
            yb = np.asarray(model.predict(xb, verbose=0))[..., 0]
            for start, y in zip(batch_starts, yb):
                check_cancelled(_acid_cancel)
                pred_sum[start:start + window] += y
                coverage[start:start + window] += 1.0
            batch.clear()
            batch_starts.clear()
        for start in tqdm(starts, desc='Student inference'):
            check_cancelled(_acid_cancel)
            batch.append(x_norm[start:start + window, None])
            batch_starts.append(start)
            if len(batch) >= batch_size:
                flush()
        flush()
        if np.any(coverage == 0):
            raise RuntimeError('Inference left uncovered samples; check inference_starts().')
        return pred_sum / coverage

    def extract_hysteresis_regions(values: np.ndarray, enter: float, exit_: float) -> list[tuple[int, int]]:
        values = np.asarray(values)
        above_exit = values >= exit_
        diff = np.diff(above_exit.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)
        regions = []
        for start, stop in zip(starts, stops):
            check_cancelled(_acid_cancel)
            if np.any(values[start:stop] >= enter):
                regions.append((int(start), int(stop - 1)))
        return regions

    def merge_regions(regions: list[tuple[int, int]], fs: float, gap_ms: float) -> list[tuple[int, int]]:
        if not regions:
            return []
        max_gap = int(round(gap_ms * fs / 1000.0))
        regions = sorted(regions)
        merged = []
        cur_s, cur_e = regions[0]
        for start, end in regions[1:]:
            check_cancelled(_acid_cancel)
            gap_samples = start - cur_e - 1
            if gap_samples <= max_gap:
                cur_e = max(cur_e, end)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = (start, end)
        merged.append((cur_s, cur_e))
        return merged

    def filter_min_duration(regions: list[tuple[int, int]], fs: float, min_ms: float) -> list[tuple[int, int]]:
        min_samples = max(1, int(math.ceil(min_ms * fs / 1000.0)))
        return [(s, e) for s, e in regions if e - s + 1 >= min_samples]

    def probability_regions(probability: np.ndarray, *, fs: float, enter: float, exit_: float, smoothing_kernel: int, merge_gap_ms: float, min_event_ms: float) -> list[tuple[int, int]]:
        kernel = max(1, int(smoothing_kernel))
        if kernel % 2 == 0:
            kernel += 1
        smooth = median_filter(np.asarray(probability, np.float32), size=kernel, mode='nearest')
        regions = extract_hysteresis_regions(smooth, enter=enter, exit_=exit_)
        regions = merge_regions(regions, fs=fs, gap_ms=merge_gap_ms)
        return filter_min_duration(regions, fs=fs, min_ms=min_event_ms)

    def threshold_regions(x: np.ndarray, *, fs: float, baseline: float, noise: float, enter_sigma: float, exit_sigma: float, merge_gap_ms: float, min_event_ms: float) -> list[tuple[int, int]]:
        """Model-free downward-event hysteresis on the oriented detrended trace."""
        enter_level = baseline - enter_sigma * noise
        exit_level = baseline - exit_sigma * noise
        below_exit = np.asarray(x) <= exit_level
        diff = np.diff(below_exit.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)
        regions = [(int(s), int(t - 1)) for s, t in zip(starts, stops) if np.any(x[s:t] <= enter_level)]
        regions = merge_regions(regions, fs=fs, gap_ms=merge_gap_ms)
        return filter_min_duration(regions, fs=fs, min_ms=min_event_ms)

    def regions_to_mask(length: int, regions: list[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(length, dtype=bool)
        for start, end in regions:
            check_cancelled(_acid_cancel)
            mask[start:end + 1] = True
        return mask

    def dwell_ms(region: tuple[int, int], fs: float) -> float:
        start, end = region
        return (end - start + 1) / fs * 1000.0


# Original notebook cell 11
def stage_011():
    global _test_prob, _test_starts
    check_cancelled(_acid_cancel)
    assert abs(dwell_ms((10, 12), 50000) - 0.06) < 1e-12
    _test_starts = inference_starts(100, window=32, step=16)
    assert _test_starts[0] == 0 and _test_starts[-1] == 68
    _test_prob = np.zeros(40, dtype=float)
    _test_prob[10:15] = 0.8
    _test_prob[15:18] = 0.2
    assert extract_hysteresis_regions(_test_prob, enter=0.5, exit_=0.15) == [(10, 17)]
    print('✅ Detection helper checks passed')


# Original notebook cell 12
def stage_012():
    global STUDENT_WINDOW, cache_valid, candidate, model_student, old_signature, prob_meta_path, prob_path, prob_signature, step, student_probability
    check_cancelled(_acid_cancel)
    if RUN_STUDENT:
        tf.keras.mixed_precision.set_global_policy('float32')
        model_student = tf.keras.models.load_model(STUDENT_MODEL_PATH, compile=False)
        STUDENT_WINDOW = int(model_student.input_shape[1])
        if int(model_student.input_shape[-1]) != 1:
            raise ValueError(f'Expected one input channel, got {model_student.input_shape}')
        step = max(1, int(round(STUDENT_WINDOW * INFERENCE_STEP_FRACTION)))
        prob_path = CACHE_DIR / f'{TAG}_student_probability.npy'
        prob_meta_path = CACHE_DIR / f'{TAG}_student_probability_meta.json'
        prob_signature = {'tag': TAG, 'n_samples': n_samples, 'fs_hz': fs, 'model_input_shape': list(model_student.input_shape), 'step': step, 'normalization': 'robust_quiet_MAD_clip30', 'data_fingerprint': INPUT_FINGERPRINTS['data'], 'model_fingerprint': INPUT_FINGERPRINTS['model']}
        cache_valid = False
        if REUSE_PROBABILITIES and prob_path.is_file() and prob_meta_path.is_file():
            try:
                old_signature = json.loads(prob_meta_path.read_text())
                candidate = np.load(prob_path, mmap_mode='r')
                cache_valid = old_signature == prob_signature and candidate.shape == (n_samples,)
            except Exception:
                cache_valid = False
        if cache_valid:
            student_probability = np.load(prob_path, mmap_mode='r')
            print('✅ Loaded cached student probabilities:', prob_path)
        else:
            student_probability = student_predict(model_student, signal, step=step, batch_size=INFERENCE_BATCH_SIZE).astype(np.float32)
            np.save(prob_path, student_probability)
            prob_meta_path.write_text(json.dumps(prob_signature, indent=2))
            print('✅ Saved student probabilities:', prob_path)
        print('Student model:', model_student.name)
        print('Student window:', STUDENT_WINDOW, f'({1000 * STUDENT_WINDOW / fs:.2f} ms)')
        print('Probability range:', float(np.min(student_probability)), float(np.max(student_probability)))
    else:
        model_student = None
        STUDENT_WINDOW = None
        student_probability = np.zeros(n_samples, dtype=np.float32)
        print('Frozen student skipped (RUN_STUDENT = False); student probability set to 0 everywhere.')


# Original notebook cell 13
def stage_013():
    global student_candidate_regions
    check_cancelled(_acid_cancel)
    if RUN_STUDENT:
        student_candidate_regions = probability_regions(student_probability, fs=fs, enter=STUDENT_ENTER_THRESHOLD, exit_=STUDENT_EXIT_THRESHOLD, smoothing_kernel=SMOOTHING_KERNEL, merge_gap_ms=MERGE_GAP_MS, min_event_ms=MIN_EVENT_MS)
    else:
        student_candidate_regions = []
    print(f'Student candidate regions: {len(student_candidate_regions):,}')


# Original notebook cell 14
def stage_014():
    global global_baseline, global_i0, global_noise, global_quiet_mask, threshold_candidate_regions, threshold_candidate_regions_res
    check_cancelled(_acid_cancel)
    global_baseline, global_noise, global_quiet_mask = robust_quiet_stats(signal)
    threshold_candidate_regions = threshold_regions(signal, fs=fs, baseline=global_baseline, noise=global_noise, enter_sigma=THRESHOLD_ENTER_SIGMA, exit_sigma=THRESHOLD_EXIT_SIGMA, merge_gap_ms=MERGE_GAP_MS, min_event_ms=MIN_EVENT_MS)
    if DETECT_RESISTIVE:
        threshold_candidate_regions_res = threshold_regions(-signal, fs=fs, baseline=-global_baseline, noise=global_noise, enter_sigma=THRESHOLD_ENTER_SIGMA, exit_sigma=THRESHOLD_EXIT_SIGMA, merge_gap_ms=MERGE_GAP_MS, min_event_ms=MIN_EVENT_MS)
    else:
        threshold_candidate_regions_res = []
    if raw is not None:
        global_i0 = abs(float(np.median(raw[global_quiet_mask])))
    elif np.isfinite(metadata_i0) and metadata_i0 > 0:
        global_i0 = metadata_i0
    else:
        global_i0 = np.nan
    print(f'Global event-excluded baseline: {global_baseline:+.3f} pA')
    print(f'Global event-excluded MAD noise: {global_noise:.3f} pA')
    print(f'Global |I0|: {global_i0:.3f} pA' if np.isfinite(global_i0) else 'Global |I0| unavailable')
    print(f'Model-free candidate regions: {len(threshold_candidate_regions):,} conductive, {len(threshold_candidate_regions_res):,} resistive')


# Original notebook cell 16
def stage_016():
    global HAVE_NUMBA, _, _cusum_core, _njit, _r2, _regions_s, _regions_t, _rng_test, _ru, _s, _t, _t2, _table_t, _ts, _tu, block_statistics, cusum_detect, merge_region_sets
    check_cancelled(_acid_cancel)
    try:
        from numba import njit as _njit
        HAVE_NUMBA = True
    except Exception:
        HAVE_NUMBA = False

        def _njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f
        print('numba unavailable: the CUSUM loop will run in pure Python (minutes instead of seconds).')

    @_njit(cache=False)
    def _cusum_core(x, mu_b, sig_b, block_len, delta, h, end_tol, level_est_samples, max_event_samples, max_events, max_levels, pre_base_samples):
        n = x.shape[0]
        nb = mu_b.shape[0]
        ev_start = np.empty(max_events, np.int64)
        ev_end = np.empty(max_events, np.int64)
        ev_lev0 = np.empty(max_events, np.int64)
        ev_nlev = np.empty(max_events, np.int64)
        ev_trunc = np.zeros(max_events, np.int8)
        lv_start = np.empty(max_levels, np.int64)
        lv_end = np.empty(max_levels, np.int64)
        lv_mean = np.empty(max_levels, np.float64)
        n_ev = 0
        n_lv = 0
        overflow = 0
        state = 0
        g_dn = 0.0
        g_up = 0.0
        zero_dn = -1
        zero_up = -1
        lev_start = 0
        lev_sum = 0.0
        lev_cnt = 0
        lev_mean = 0.0
        lev_frozen = False
        ev_onset = 0
        ev_first_lev = 0
        pre_base = 0.0
        win_sum = 0.0
        half = 0.5 * block_len
        i = 0
        while i < n:
            p = (i - half) / block_len
            b0 = int(np.floor(p))
            if b0 < 0:
                mu = mu_b[0]
                sg = sig_b[0]
            elif b0 >= nb - 1:
                mu = mu_b[nb - 1]
                sg = sig_b[nb - 1]
            else:
                f = p - b0
                mu = (1.0 - f) * mu_b[b0] + f * mu_b[b0 + 1]
                sg = (1.0 - f) * sig_b[b0] + f * sig_b[b0 + 1]
            xi = x[i]
            if state == 0:
                g_dn += (mu - xi) / sg - 0.5 * delta
                if g_dn <= 0.0:
                    g_dn = 0.0
                    zero_dn = i
                if g_dn > h:
                    if n_ev >= max_events or n_lv >= max_levels:
                        overflow = 1
                        break
                    onset = zero_dn + 1
                    if onset > i:
                        onset = i
                    ev_onset = onset
                    ev_first_lev = n_lv
                    pre_lo = onset - pre_base_samples
                    if pre_lo < 0:
                        pre_lo = 0
                    if onset - pre_lo >= 5:
                        pre_base = np.median(x[pre_lo:onset])
                    else:
                        pre_base = mu
                    lev_start = onset
                    lev_sum = 0.0
                    lev_cnt = 0
                    for j in range(onset, i + 1):
                        lev_sum += x[j]
                        lev_cnt += 1
                    lev_mean = lev_sum / lev_cnt
                    lev_frozen = lev_cnt >= level_est_samples
                    win_sum = 0.0
                    for j in range(max(onset, i - level_est_samples + 1), i + 1):
                        win_sum += x[j]
                    g_dn = 0.0
                    g_up = 0.0
                    zero_dn = i
                    zero_up = i
                    state = 1
            else:
                n_in = i - ev_onset + 1
                win_sum += xi
                if n_in > level_est_samples:
                    win_sum -= x[i - level_est_samples]
                if n_in >= level_est_samples:
                    m_w = win_sum / level_est_samples
                    if (mu - m_w) / sg < end_tol or (pre_base - m_w) / sg < end_tol:
                        if (mu - m_w) / sg < end_tol:
                            ref = mu
                        else:
                            ref = pre_base
                        c = i - level_est_samples + 1
                        for j in range(i, i - level_est_samples, -1):
                            if (ref - x[j]) / sg > 2.0 * end_tol:
                                c = j + 1
                                break
                        if c <= ev_onset:
                            n_lv = ev_first_lev
                            state = 0
                            g_dn = 0.0
                            g_up = 0.0
                            zero_dn = i
                            zero_up = i
                            i += 1
                            continue
                        if c <= lev_start:
                            c = lev_start + 1
                        sm = 0.0
                        for j in range(lev_start, c):
                            sm += x[j]
                        if n_lv >= max_levels or n_ev >= max_events:
                            overflow = 1
                            break
                        lv_start[n_lv] = lev_start
                        lv_end[n_lv] = c - 1
                        lv_mean[n_lv] = sm / (c - lev_start)
                        n_lv += 1
                        ev_start[n_ev] = ev_onset
                        ev_end[n_ev] = c - 1
                        ev_lev0[n_ev] = ev_first_lev
                        ev_nlev[n_ev] = n_lv - ev_first_lev
                        n_ev += 1
                        state = 0
                        g_dn = 0.0
                        g_up = 0.0
                        zero_dn = c - 1
                        zero_up = c - 1
                        i = c
                        continue
                if not lev_frozen:
                    lev_sum += xi
                    lev_cnt += 1
                    lev_mean = lev_sum / lev_cnt
                    if lev_cnt >= level_est_samples:
                        lev_frozen = True
                g_dn += (lev_mean - xi) / sg - 0.5 * delta
                if g_dn <= 0.0:
                    g_dn = 0.0
                    zero_dn = i
                g_up += (xi - lev_mean) / sg - 0.5 * delta
                if g_up <= 0.0:
                    g_up = 0.0
                    zero_up = i
                alarm = g_dn > h or g_up > h
                runaway = i - ev_onset + 1 >= max_event_samples
                if alarm or runaway:
                    if runaway:
                        c = i + 1
                    else:
                        if g_dn > h:
                            c = zero_dn + 1
                        else:
                            c = zero_up + 1
                        if c <= lev_start:
                            c = lev_start + 1
                        if c > i:
                            c = i
                    s = 0.0
                    for j in range(lev_start, c):
                        s += x[j]
                    if n_lv >= max_levels:
                        overflow = 1
                        break
                    lv_start[n_lv] = lev_start
                    lv_end[n_lv] = c - 1
                    lv_mean[n_lv] = s / (c - lev_start)
                    n_lv += 1
                    k_end = c + level_est_samples
                    if k_end > n:
                        k_end = n
                    m = 0.0
                    cnt = 0
                    for j in range(c, k_end):
                        m += x[j]
                        cnt += 1
                    if cnt > 0:
                        m = m / cnt
                    is_baseline = runaway or cnt == 0 or (mu - m) / sg < end_tol or ((pre_base - m) / sg < end_tol)
                    if is_baseline:
                        ev_start[n_ev] = ev_onset
                        ev_end[n_ev] = c - 1
                        ev_lev0[n_ev] = ev_first_lev
                        ev_nlev[n_ev] = n_lv - ev_first_lev
                        if runaway:
                            ev_trunc[n_ev] = 1
                        n_ev += 1
                        state = 0
                        g_dn = 0.0
                        g_up = 0.0
                        zero_dn = c - 1
                        zero_up = c - 1
                        i = c
                        continue
                    lev_start = c
                    lev_sum = 0.0
                    lev_cnt = 0
                    for j in range(c, i + 1):
                        lev_sum += x[j]
                        lev_cnt += 1
                    lev_mean = lev_sum / lev_cnt
                    lev_frozen = lev_cnt >= level_est_samples
                    g_dn = 0.0
                    g_up = 0.0
                    zero_dn = i
                    zero_up = i
            i += 1
        if state == 1 and overflow == 0:
            s = 0.0
            for j in range(lev_start, n):
                s += x[j]
            if n_lv < max_levels and n_ev < max_events:
                lv_start[n_lv] = lev_start
                lv_end[n_lv] = n - 1
                lv_mean[n_lv] = s / (n - lev_start)
                n_lv += 1
                ev_start[n_ev] = ev_onset
                ev_end[n_ev] = n - 1
                ev_lev0[n_ev] = ev_first_lev
                ev_nlev[n_ev] = n_lv - ev_first_lev
                ev_trunc[n_ev] = 1
                n_ev += 1
            else:
                overflow = 1
        return (ev_start[:n_ev], ev_end[:n_ev], ev_lev0[:n_ev], ev_nlev[:n_ev], ev_trunc[:n_ev], lv_start[:n_lv], lv_end[:n_lv], lv_mean[:n_lv], overflow)

    def block_statistics(x: np.ndarray, block_len: int, exclude_mask: np.ndarray | None=None, min_valid_fraction: float=0.2, sigma_floor: float | None=None) -> tuple[np.ndarray, np.ndarray]:
        """Per-block median and MAD noise, linearly interpolated across blocks with too few
    valid samples. Returns float64 arrays of length ceil(n / block_len)."""
        n = len(x)
        n_blocks = int(math.ceil(n / block_len))
        padded = np.full(n_blocks * block_len, np.nan, dtype=np.float32)
        padded[:n] = x
        if exclude_mask is not None:
            padded[:n][exclude_mask] = np.nan
        blocks = padded.reshape(n_blocks, block_len)
        valid_fraction = np.mean(np.isfinite(blocks), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if exclude_mask is None and n_blocks > 1:
                med = np.empty(n_blocks)
                mad = np.empty(n_blocks)
                med[:-1] = np.median(blocks[:-1], axis=1)
                mad[:-1] = 1.4826 * np.median(np.abs(blocks[:-1] - med[:-1, None]), axis=1)
                med[-1] = np.nanmedian(blocks[-1])
                mad[-1] = 1.4826 * np.nanmedian(np.abs(blocks[-1] - med[-1]))
            else:
                med = np.nanmedian(blocks, axis=1)
                mad = 1.4826 * np.nanmedian(np.abs(blocks - med[:, None]), axis=1)
        bad = (valid_fraction < min_valid_fraction) | ~np.isfinite(med) | ~np.isfinite(mad) | (mad <= 0)
        med = pd.Series(np.where(bad, np.nan, med)).interpolate(limit_direction='both').to_numpy(float)
        mad = pd.Series(np.where(bad, np.nan, mad)).interpolate(limit_direction='both').to_numpy(float)
        if not np.all(np.isfinite(med)):
            med = np.nan_to_num(med, nan=float(np.nanmedian(x)))
        if not np.all(np.isfinite(mad)):
            mad = np.nan_to_num(mad, nan=mad_std(x))
        if sigma_floor is None:
            sigma_floor = 0.25 * float(np.median(mad))
        mad = np.maximum(mad, max(sigma_floor, 1e-06))
        return (med, mad)

    def cusum_detect(x: np.ndarray, *, fs: float, delta: float, h: float, end_tol_sigma: float=1.5, block_ms: float=50.0, level_est_ms: float=0.4, max_event_s: float=2.0, min_event_ms: float=0.05, passes: int=2, mask_margin_ms: float=2.0, exclude_mask: np.ndarray | None=None, pass1_block_ms: float | None=None, pre_baseline_ms: float=1.0) -> tuple[list[tuple[int, int]], pd.DataFrame, dict]:
        """Two-pass CUSUM event detection. Returns (regions, event_table, info).

    event_table rows are 1:1 with regions (sorted) and carry the level segmentation:
    n_levels, deepest level depth (pA, below the local block baseline), the minimum
    separation between adjacent levels (sigma units), and a truncation flag.
    """
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = len(x)
        block_len = max(50, int(round(block_ms * fs / 1000.0)))
        level_est = max(3, int(round(level_est_ms * fs / 1000.0)))
        max_event = max(level_est + 1, int(round(max_event_s * fs)))
        min_event = max(1, int(math.ceil(min_event_ms * fs / 1000.0)))
        max_events = max(1000, n // 20)
        max_levels = 4 * max_events
        pre_base_samples = max(5, int(round(pre_baseline_ms * fs / 1000.0)))
        margin = int(round(mask_margin_ms * fs / 1000.0))
        mask = exclude_mask
        info = {'passes': passes, 'block_len': block_len}
        n_pass = max(1, passes)
        for p in range(n_pass):
            check_cancelled(_acid_cancel)
            blk = block_len
            if p < n_pass - 1 and pass1_block_ms is not None:
                blk = max(block_len, int(round(pass1_block_ms * fs / 1000.0)))
            mu_b, sig_b = block_statistics(x, blk, exclude_mask=mask)
            out = _cusum_core(x, mu_b, sig_b, blk, float(delta), float(h), float(end_tol_sigma), level_est, max_event, max_events, max_levels, pre_base_samples)
            ev_start, ev_end, ev_lev0, ev_nlev, ev_trunc, lv_start, lv_end, lv_mean, overflow = out
            if overflow:
                raise RuntimeError('CUSUM capacity exceeded: delta/h are far too permissive for this trace.')
            if p < passes - 1:
                new_mask = np.zeros(n, dtype=bool)
                for s, e in zip(ev_start, ev_end):
                    check_cancelled(_acid_cancel)
                    new_mask[max(0, s - margin):min(n, e + 1 + margin)] = True
                mask = new_mask if exclude_mask is None else new_mask | exclude_mask
                info[f'pass{p + 1}_events'] = int(len(ev_start))
        info['mu_blocks'] = mu_b
        info['sigma_blocks'] = sig_b
        rows = []
        regions = []
        for k in range(len(ev_start)):
            check_cancelled(_acid_cancel)
            s, e = (int(ev_start[k]), int(ev_end[k]))
            if e - s + 1 < min_event:
                continue
            l0, nl = (int(ev_lev0[k]), int(ev_nlev[k]))
            b = min(len(mu_b) - 1, max(0, int((s - 0.5 * block_len) // block_len)))
            base, sig = (float(mu_b[b]), float(sig_b[b]))
            means = lv_mean[l0:l0 + nl]
            lengths = lv_end[l0:l0 + nl] - lv_start[l0:l0 + nl] + 1
            depths = base - means
            body_depth = base - x[s:e + 1].astype(float)
            _pk = float(np.max(body_depth))
            _half = body_depth >= 0.5 * _pk if _pk > 0 else np.zeros(len(body_depth), dtype=bool)
            sep = float(np.min(np.abs(np.diff(means))) / sig) if nl >= 2 else np.nan
            regions.append((s, e))
            rows.append({'start_idx': s, 'end_idx': e, 'cusum_n_levels': nl, 'cusum_sustained_depth_pA': float(np.median(body_depth)), 'cusum_q80_depth_pA': float(np.quantile(body_depth, 0.8)), 'cusum_peak_depth_pA': _pk, 'cusum_half_peak_mean_pA': float(np.mean(body_depth[_half])) if _half.any() else np.nan, 'cusum_half_peak_ms': float(_half.sum()) / fs * 1000.0, 'cusum_block_noise_pA': sig, 'cusum_block_baseline_pA': base, 'cusum_deepest_level_pA': float(np.max(depths)) if nl else np.nan, 'cusum_shallowest_level_pA': float(np.min(depths)) if nl else np.nan, 'cusum_min_level_sep_sigma': sep, 'cusum_longest_level_fraction': float(np.max(lengths) / (e - s + 1)) if nl else np.nan, 'cusum_truncated': bool(ev_trunc[k])})
        table = pd.DataFrame(rows, columns=['start_idx', 'end_idx', 'cusum_n_levels', 'cusum_sustained_depth_pA', 'cusum_q80_depth_pA', 'cusum_peak_depth_pA', 'cusum_half_peak_mean_pA', 'cusum_half_peak_ms', 'cusum_block_noise_pA', 'cusum_block_baseline_pA', 'cusum_deepest_level_pA', 'cusum_shallowest_level_pA', 'cusum_min_level_sep_sigma', 'cusum_longest_level_fraction', 'cusum_truncated'])
        info['levels'] = (lv_start, lv_end, lv_mean)
        return (regions, table, info)

    def merge_region_sets(x: np.ndarray, sets: list[tuple[list[tuple[int, int]], pd.DataFrame, str]], fs_merge: float=50000.0) -> tuple[list[tuple[int, int]], pd.DataFrame]:
        """Union of several detectors' regions. Overlapping/adjacent regions become one region; each
    merged region takes its level attributes from the source region with the largest overlap and
    has its depth statistics recomputed over the merged span with that source's block baseline.
    A cusum_scale column records which scale(s) contributed."""
        entries = []
        for regions, table, name in sets:
            check_cancelled(_acid_cancel)
            for (s, e), (_, row) in zip(regions, table.iterrows()):
                check_cancelled(_acid_cancel)
                entries.append((int(s), int(e), name, row))
        entries.sort(key=lambda t: (t[0], t[1]))
        merged_regions, merged_rows = ([], [])
        cur = None
        for s, e, name, row in entries:
            check_cancelled(_acid_cancel)
            if cur is None or s > cur[1] + 1:
                if cur is not None:
                    merged_regions.append((cur[0], cur[1]))
                    merged_rows.append(cur[2])
                cur = [s, e, [(s, e, name, row)]]
            else:
                cur[1] = max(cur[1], e)
                cur[2].append((s, e, name, row))
        if cur is not None:
            merged_regions.append((cur[0], cur[1]))
            merged_rows.append(cur[2])
        rows = []
        for (s, e), members in zip(merged_regions, merged_rows):
            check_cancelled(_acid_cancel)
            best = max(members, key=lambda m: min(m[1], e) - max(m[0], s) + 1)
            row = best[3].copy()
            base = float(row['cusum_block_baseline_pA'])
            body_depth = base - x[s:e + 1].astype(float)
            row['start_idx'], row['end_idx'] = (s, e)
            row['cusum_sustained_depth_pA'] = float(np.median(body_depth))
            row['cusum_q80_depth_pA'] = float(np.quantile(body_depth, 0.8))
            pk = float(np.max(body_depth))
            half = body_depth >= 0.5 * pk if pk > 0 else np.zeros(len(body_depth), dtype=bool)
            row['cusum_peak_depth_pA'] = pk
            row['cusum_half_peak_mean_pA'] = float(np.mean(body_depth[half])) if half.any() else np.nan
            row['cusum_half_peak_ms'] = float(half.sum()) / fs_merge * 1000.0
            row['cusum_scale'] = '+'.join(sorted({m[2] for m in members}))
            rows.append(row)
        table = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=['start_idx', 'end_idx', 'cusum_n_levels', 'cusum_sustained_depth_pA', 'cusum_q80_depth_pA', 'cusum_peak_depth_pA', 'cusum_half_peak_mean_pA', 'cusum_half_peak_ms', 'cusum_block_noise_pA', 'cusum_block_baseline_pA', 'cusum_deepest_level_pA', 'cusum_shallowest_level_pA', 'cusum_min_level_sep_sigma', 'cusum_longest_level_fraction', 'cusum_truncated', 'cusum_scale'])
        return (merged_regions, table)
    _rng_test = np.random.default_rng(0)
    _t = _rng_test.normal(0.0, 1.0, 200000).astype(np.float32)
    _t[50000:50100] -= 8.0
    _t[120000:120400] -= 5.0
    _t[120200:120400] -= 6.0
    _t[150000:151900] -= 6.0
    _regions_t, _table_t, _ = cusum_detect(_t, fs=50000.0, delta=2.0, h=8.0, passes=2, pass1_block_ms=1000.0)
    assert len(_regions_t) == 3, _regions_t
    assert abs(_regions_t[2][0] - 150000) <= 3 and abs(_regions_t[2][1] - 151899) <= 3, _regions_t
    assert abs(_regions_t[0][0] - 50000) <= 2 and abs(_regions_t[0][1] - 50099) <= 2, _regions_t
    assert abs(_regions_t[1][0] - 120000) <= 2 and abs(_regions_t[1][1] - 120399) <= 2, _regions_t
    assert int(_table_t.loc[1, 'cusum_n_levels']) == 2 and int(_table_t.loc[0, 'cusum_n_levels']) == 1, _table_t
    assert abs(_table_t.loc[1, 'cusum_deepest_level_pA'] - 11.0) < 0.5, _table_t
    _r2, _t2, _ = cusum_detect(_t, fs=50000.0, delta=4.0, h=8.0, passes=2, block_ms=2.0)
    _ru, _tu = merge_region_sets(_t, [(_regions_t, _table_t, 'level'), (_r2, _t2, 'pulse')])
    assert len(_ru) >= 3 and set(_tu['cusum_scale']) <= {'level', 'pulse', 'level+pulse'}, _tu
    _ts = np.random.default_rng(2).normal(0.0, 1.0, 300000).astype(np.float32)
    for _s in (100000, 200000):
        check_cancelled(_acid_cancel)
        _ts[_s:_s + 250] -= 4.0
        _ts[_s + 250:_s + 650] -= 4.0 * np.exp(-(np.arange(400) + 1.0) / 50.0)
    _regions_s, _, _ = cusum_detect(_ts, fs=50000.0, delta=4.0, h=100.0, passes=2, pass1_block_ms=1000.0, mask_margin_ms=3.0)
    assert len(_regions_s) == 2 and all((abs(s0 - s) <= 5 and abs(e - (s + 249)) <= 150 for (s0, e), s in zip(_regions_s, (100000, 200000)))), _regions_s
    print('✅ CUSUM core checks passed', '(numba)' if HAVE_NUMBA else '(pure Python)')


def definitions_016():
    global HAVE_NUMBA, _cusum_core, _njit, block_statistics, cusum_detect, merge_region_sets
    try:
        from numba import njit as _njit
        HAVE_NUMBA = True
    except Exception:
        HAVE_NUMBA = False

        def _njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f
        print('numba unavailable: the CUSUM loop will run in pure Python (minutes instead of seconds).')

    @_njit(cache=False)
    def _cusum_core(x, mu_b, sig_b, block_len, delta, h, end_tol, level_est_samples, max_event_samples, max_events, max_levels, pre_base_samples):
        n = x.shape[0]
        nb = mu_b.shape[0]
        ev_start = np.empty(max_events, np.int64)
        ev_end = np.empty(max_events, np.int64)
        ev_lev0 = np.empty(max_events, np.int64)
        ev_nlev = np.empty(max_events, np.int64)
        ev_trunc = np.zeros(max_events, np.int8)
        lv_start = np.empty(max_levels, np.int64)
        lv_end = np.empty(max_levels, np.int64)
        lv_mean = np.empty(max_levels, np.float64)
        n_ev = 0
        n_lv = 0
        overflow = 0
        state = 0
        g_dn = 0.0
        g_up = 0.0
        zero_dn = -1
        zero_up = -1
        lev_start = 0
        lev_sum = 0.0
        lev_cnt = 0
        lev_mean = 0.0
        lev_frozen = False
        ev_onset = 0
        ev_first_lev = 0
        pre_base = 0.0
        win_sum = 0.0
        half = 0.5 * block_len
        i = 0
        while i < n:
            p = (i - half) / block_len
            b0 = int(np.floor(p))
            if b0 < 0:
                mu = mu_b[0]
                sg = sig_b[0]
            elif b0 >= nb - 1:
                mu = mu_b[nb - 1]
                sg = sig_b[nb - 1]
            else:
                f = p - b0
                mu = (1.0 - f) * mu_b[b0] + f * mu_b[b0 + 1]
                sg = (1.0 - f) * sig_b[b0] + f * sig_b[b0 + 1]
            xi = x[i]
            if state == 0:
                g_dn += (mu - xi) / sg - 0.5 * delta
                if g_dn <= 0.0:
                    g_dn = 0.0
                    zero_dn = i
                if g_dn > h:
                    if n_ev >= max_events or n_lv >= max_levels:
                        overflow = 1
                        break
                    onset = zero_dn + 1
                    if onset > i:
                        onset = i
                    ev_onset = onset
                    ev_first_lev = n_lv
                    pre_lo = onset - pre_base_samples
                    if pre_lo < 0:
                        pre_lo = 0
                    if onset - pre_lo >= 5:
                        pre_base = np.median(x[pre_lo:onset])
                    else:
                        pre_base = mu
                    lev_start = onset
                    lev_sum = 0.0
                    lev_cnt = 0
                    for j in range(onset, i + 1):
                        lev_sum += x[j]
                        lev_cnt += 1
                    lev_mean = lev_sum / lev_cnt
                    lev_frozen = lev_cnt >= level_est_samples
                    win_sum = 0.0
                    for j in range(max(onset, i - level_est_samples + 1), i + 1):
                        win_sum += x[j]
                    g_dn = 0.0
                    g_up = 0.0
                    zero_dn = i
                    zero_up = i
                    state = 1
            else:
                n_in = i - ev_onset + 1
                win_sum += xi
                if n_in > level_est_samples:
                    win_sum -= x[i - level_est_samples]
                if n_in >= level_est_samples:
                    m_w = win_sum / level_est_samples
                    if (mu - m_w) / sg < end_tol or (pre_base - m_w) / sg < end_tol:
                        if (mu - m_w) / sg < end_tol:
                            ref = mu
                        else:
                            ref = pre_base
                        c = i - level_est_samples + 1
                        for j in range(i, i - level_est_samples, -1):
                            if (ref - x[j]) / sg > 2.0 * end_tol:
                                c = j + 1
                                break
                        if c <= ev_onset:
                            n_lv = ev_first_lev
                            state = 0
                            g_dn = 0.0
                            g_up = 0.0
                            zero_dn = i
                            zero_up = i
                            i += 1
                            continue
                        if c <= lev_start:
                            c = lev_start + 1
                        sm = 0.0
                        for j in range(lev_start, c):
                            sm += x[j]
                        if n_lv >= max_levels or n_ev >= max_events:
                            overflow = 1
                            break
                        lv_start[n_lv] = lev_start
                        lv_end[n_lv] = c - 1
                        lv_mean[n_lv] = sm / (c - lev_start)
                        n_lv += 1
                        ev_start[n_ev] = ev_onset
                        ev_end[n_ev] = c - 1
                        ev_lev0[n_ev] = ev_first_lev
                        ev_nlev[n_ev] = n_lv - ev_first_lev
                        n_ev += 1
                        state = 0
                        g_dn = 0.0
                        g_up = 0.0
                        zero_dn = c - 1
                        zero_up = c - 1
                        i = c
                        continue
                if not lev_frozen:
                    lev_sum += xi
                    lev_cnt += 1
                    lev_mean = lev_sum / lev_cnt
                    if lev_cnt >= level_est_samples:
                        lev_frozen = True
                g_dn += (lev_mean - xi) / sg - 0.5 * delta
                if g_dn <= 0.0:
                    g_dn = 0.0
                    zero_dn = i
                g_up += (xi - lev_mean) / sg - 0.5 * delta
                if g_up <= 0.0:
                    g_up = 0.0
                    zero_up = i
                alarm = g_dn > h or g_up > h
                runaway = i - ev_onset + 1 >= max_event_samples
                if alarm or runaway:
                    if runaway:
                        c = i + 1
                    else:
                        if g_dn > h:
                            c = zero_dn + 1
                        else:
                            c = zero_up + 1
                        if c <= lev_start:
                            c = lev_start + 1
                        if c > i:
                            c = i
                    s = 0.0
                    for j in range(lev_start, c):
                        s += x[j]
                    if n_lv >= max_levels:
                        overflow = 1
                        break
                    lv_start[n_lv] = lev_start
                    lv_end[n_lv] = c - 1
                    lv_mean[n_lv] = s / (c - lev_start)
                    n_lv += 1
                    k_end = c + level_est_samples
                    if k_end > n:
                        k_end = n
                    m = 0.0
                    cnt = 0
                    for j in range(c, k_end):
                        m += x[j]
                        cnt += 1
                    if cnt > 0:
                        m = m / cnt
                    is_baseline = runaway or cnt == 0 or (mu - m) / sg < end_tol or ((pre_base - m) / sg < end_tol)
                    if is_baseline:
                        ev_start[n_ev] = ev_onset
                        ev_end[n_ev] = c - 1
                        ev_lev0[n_ev] = ev_first_lev
                        ev_nlev[n_ev] = n_lv - ev_first_lev
                        if runaway:
                            ev_trunc[n_ev] = 1
                        n_ev += 1
                        state = 0
                        g_dn = 0.0
                        g_up = 0.0
                        zero_dn = c - 1
                        zero_up = c - 1
                        i = c
                        continue
                    lev_start = c
                    lev_sum = 0.0
                    lev_cnt = 0
                    for j in range(c, i + 1):
                        lev_sum += x[j]
                        lev_cnt += 1
                    lev_mean = lev_sum / lev_cnt
                    lev_frozen = lev_cnt >= level_est_samples
                    g_dn = 0.0
                    g_up = 0.0
                    zero_dn = i
                    zero_up = i
            i += 1
        if state == 1 and overflow == 0:
            s = 0.0
            for j in range(lev_start, n):
                s += x[j]
            if n_lv < max_levels and n_ev < max_events:
                lv_start[n_lv] = lev_start
                lv_end[n_lv] = n - 1
                lv_mean[n_lv] = s / (n - lev_start)
                n_lv += 1
                ev_start[n_ev] = ev_onset
                ev_end[n_ev] = n - 1
                ev_lev0[n_ev] = ev_first_lev
                ev_nlev[n_ev] = n_lv - ev_first_lev
                ev_trunc[n_ev] = 1
                n_ev += 1
            else:
                overflow = 1
        return (ev_start[:n_ev], ev_end[:n_ev], ev_lev0[:n_ev], ev_nlev[:n_ev], ev_trunc[:n_ev], lv_start[:n_lv], lv_end[:n_lv], lv_mean[:n_lv], overflow)

    def block_statistics(x: np.ndarray, block_len: int, exclude_mask: np.ndarray | None=None, min_valid_fraction: float=0.2, sigma_floor: float | None=None) -> tuple[np.ndarray, np.ndarray]:
        """Per-block median and MAD noise, linearly interpolated across blocks with too few
    valid samples. Returns float64 arrays of length ceil(n / block_len)."""
        n = len(x)
        n_blocks = int(math.ceil(n / block_len))
        padded = np.full(n_blocks * block_len, np.nan, dtype=np.float32)
        padded[:n] = x
        if exclude_mask is not None:
            padded[:n][exclude_mask] = np.nan
        blocks = padded.reshape(n_blocks, block_len)
        valid_fraction = np.mean(np.isfinite(blocks), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if exclude_mask is None and n_blocks > 1:
                med = np.empty(n_blocks)
                mad = np.empty(n_blocks)
                med[:-1] = np.median(blocks[:-1], axis=1)
                mad[:-1] = 1.4826 * np.median(np.abs(blocks[:-1] - med[:-1, None]), axis=1)
                med[-1] = np.nanmedian(blocks[-1])
                mad[-1] = 1.4826 * np.nanmedian(np.abs(blocks[-1] - med[-1]))
            else:
                med = np.nanmedian(blocks, axis=1)
                mad = 1.4826 * np.nanmedian(np.abs(blocks - med[:, None]), axis=1)
        bad = (valid_fraction < min_valid_fraction) | ~np.isfinite(med) | ~np.isfinite(mad) | (mad <= 0)
        med = pd.Series(np.where(bad, np.nan, med)).interpolate(limit_direction='both').to_numpy(float)
        mad = pd.Series(np.where(bad, np.nan, mad)).interpolate(limit_direction='both').to_numpy(float)
        if not np.all(np.isfinite(med)):
            med = np.nan_to_num(med, nan=float(np.nanmedian(x)))
        if not np.all(np.isfinite(mad)):
            mad = np.nan_to_num(mad, nan=mad_std(x))
        if sigma_floor is None:
            sigma_floor = 0.25 * float(np.median(mad))
        mad = np.maximum(mad, max(sigma_floor, 1e-06))
        return (med, mad)

    def cusum_detect(x: np.ndarray, *, fs: float, delta: float, h: float, end_tol_sigma: float=1.5, block_ms: float=50.0, level_est_ms: float=0.4, max_event_s: float=2.0, min_event_ms: float=0.05, passes: int=2, mask_margin_ms: float=2.0, exclude_mask: np.ndarray | None=None, pass1_block_ms: float | None=None, pre_baseline_ms: float=1.0) -> tuple[list[tuple[int, int]], pd.DataFrame, dict]:
        """Two-pass CUSUM event detection. Returns (regions, event_table, info).

    event_table rows are 1:1 with regions (sorted) and carry the level segmentation:
    n_levels, deepest level depth (pA, below the local block baseline), the minimum
    separation between adjacent levels (sigma units), and a truncation flag.
    """
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = len(x)
        block_len = max(50, int(round(block_ms * fs / 1000.0)))
        level_est = max(3, int(round(level_est_ms * fs / 1000.0)))
        max_event = max(level_est + 1, int(round(max_event_s * fs)))
        min_event = max(1, int(math.ceil(min_event_ms * fs / 1000.0)))
        max_events = max(1000, n // 20)
        max_levels = 4 * max_events
        pre_base_samples = max(5, int(round(pre_baseline_ms * fs / 1000.0)))
        margin = int(round(mask_margin_ms * fs / 1000.0))
        mask = exclude_mask
        info = {'passes': passes, 'block_len': block_len}
        n_pass = max(1, passes)
        for p in range(n_pass):
            check_cancelled(_acid_cancel)
            blk = block_len
            if p < n_pass - 1 and pass1_block_ms is not None:
                blk = max(block_len, int(round(pass1_block_ms * fs / 1000.0)))
            mu_b, sig_b = block_statistics(x, blk, exclude_mask=mask)
            out = _cusum_core(x, mu_b, sig_b, blk, float(delta), float(h), float(end_tol_sigma), level_est, max_event, max_events, max_levels, pre_base_samples)
            ev_start, ev_end, ev_lev0, ev_nlev, ev_trunc, lv_start, lv_end, lv_mean, overflow = out
            if overflow:
                raise RuntimeError('CUSUM capacity exceeded: delta/h are far too permissive for this trace.')
            if p < passes - 1:
                new_mask = np.zeros(n, dtype=bool)
                for s, e in zip(ev_start, ev_end):
                    check_cancelled(_acid_cancel)
                    new_mask[max(0, s - margin):min(n, e + 1 + margin)] = True
                mask = new_mask if exclude_mask is None else new_mask | exclude_mask
                info[f'pass{p + 1}_events'] = int(len(ev_start))
        info['mu_blocks'] = mu_b
        info['sigma_blocks'] = sig_b
        rows = []
        regions = []
        for k in range(len(ev_start)):
            check_cancelled(_acid_cancel)
            s, e = (int(ev_start[k]), int(ev_end[k]))
            if e - s + 1 < min_event:
                continue
            l0, nl = (int(ev_lev0[k]), int(ev_nlev[k]))
            b = min(len(mu_b) - 1, max(0, int((s - 0.5 * block_len) // block_len)))
            base, sig = (float(mu_b[b]), float(sig_b[b]))
            means = lv_mean[l0:l0 + nl]
            lengths = lv_end[l0:l0 + nl] - lv_start[l0:l0 + nl] + 1
            depths = base - means
            body_depth = base - x[s:e + 1].astype(float)
            _pk = float(np.max(body_depth))
            _half = body_depth >= 0.5 * _pk if _pk > 0 else np.zeros(len(body_depth), dtype=bool)
            sep = float(np.min(np.abs(np.diff(means))) / sig) if nl >= 2 else np.nan
            regions.append((s, e))
            rows.append({'start_idx': s, 'end_idx': e, 'cusum_n_levels': nl, 'cusum_sustained_depth_pA': float(np.median(body_depth)), 'cusum_q80_depth_pA': float(np.quantile(body_depth, 0.8)), 'cusum_peak_depth_pA': _pk, 'cusum_half_peak_mean_pA': float(np.mean(body_depth[_half])) if _half.any() else np.nan, 'cusum_half_peak_ms': float(_half.sum()) / fs * 1000.0, 'cusum_block_noise_pA': sig, 'cusum_block_baseline_pA': base, 'cusum_deepest_level_pA': float(np.max(depths)) if nl else np.nan, 'cusum_shallowest_level_pA': float(np.min(depths)) if nl else np.nan, 'cusum_min_level_sep_sigma': sep, 'cusum_longest_level_fraction': float(np.max(lengths) / (e - s + 1)) if nl else np.nan, 'cusum_truncated': bool(ev_trunc[k])})
        table = pd.DataFrame(rows, columns=['start_idx', 'end_idx', 'cusum_n_levels', 'cusum_sustained_depth_pA', 'cusum_q80_depth_pA', 'cusum_peak_depth_pA', 'cusum_half_peak_mean_pA', 'cusum_half_peak_ms', 'cusum_block_noise_pA', 'cusum_block_baseline_pA', 'cusum_deepest_level_pA', 'cusum_shallowest_level_pA', 'cusum_min_level_sep_sigma', 'cusum_longest_level_fraction', 'cusum_truncated'])
        info['levels'] = (lv_start, lv_end, lv_mean)
        return (regions, table, info)

    def merge_region_sets(x: np.ndarray, sets: list[tuple[list[tuple[int, int]], pd.DataFrame, str]], fs_merge: float=50000.0) -> tuple[list[tuple[int, int]], pd.DataFrame]:
        """Union of several detectors' regions. Overlapping/adjacent regions become one region; each
    merged region takes its level attributes from the source region with the largest overlap and
    has its depth statistics recomputed over the merged span with that source's block baseline.
    A cusum_scale column records which scale(s) contributed."""
        entries = []
        for regions, table, name in sets:
            check_cancelled(_acid_cancel)
            for (s, e), (_, row) in zip(regions, table.iterrows()):
                check_cancelled(_acid_cancel)
                entries.append((int(s), int(e), name, row))
        entries.sort(key=lambda t: (t[0], t[1]))
        merged_regions, merged_rows = ([], [])
        cur = None
        for s, e, name, row in entries:
            check_cancelled(_acid_cancel)
            if cur is None or s > cur[1] + 1:
                if cur is not None:
                    merged_regions.append((cur[0], cur[1]))
                    merged_rows.append(cur[2])
                cur = [s, e, [(s, e, name, row)]]
            else:
                cur[1] = max(cur[1], e)
                cur[2].append((s, e, name, row))
        if cur is not None:
            merged_regions.append((cur[0], cur[1]))
            merged_rows.append(cur[2])
        rows = []
        for (s, e), members in zip(merged_regions, merged_rows):
            check_cancelled(_acid_cancel)
            best = max(members, key=lambda m: min(m[1], e) - max(m[0], s) + 1)
            row = best[3].copy()
            base = float(row['cusum_block_baseline_pA'])
            body_depth = base - x[s:e + 1].astype(float)
            row['start_idx'], row['end_idx'] = (s, e)
            row['cusum_sustained_depth_pA'] = float(np.median(body_depth))
            row['cusum_q80_depth_pA'] = float(np.quantile(body_depth, 0.8))
            pk = float(np.max(body_depth))
            half = body_depth >= 0.5 * pk if pk > 0 else np.zeros(len(body_depth), dtype=bool)
            row['cusum_peak_depth_pA'] = pk
            row['cusum_half_peak_mean_pA'] = float(np.mean(body_depth[half])) if half.any() else np.nan
            row['cusum_half_peak_ms'] = float(half.sum()) / fs_merge * 1000.0
            row['cusum_scale'] = '+'.join(sorted({m[2] for m in members}))
            rows.append(row)
        table = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=['start_idx', 'end_idx', 'cusum_n_levels', 'cusum_sustained_depth_pA', 'cusum_q80_depth_pA', 'cusum_peak_depth_pA', 'cusum_half_peak_mean_pA', 'cusum_half_peak_ms', 'cusum_block_noise_pA', 'cusum_block_baseline_pA', 'cusum_deepest_level_pA', 'cusum_shallowest_level_pA', 'cusum_min_level_sep_sigma', 'cusum_longest_level_fraction', 'cusum_truncated', 'cusum_scale'])
        return (merged_regions, table)


# Original notebook cell 17
def stage_017():
    global CUSUM_DELTA, CUSUM_H, CUSUM_PULSE_DELTA, CUSUM_PULSE_H, CUSUM_SCALES, _, _bias_pivot, _bmax, _busy, _cal_margin, _cells, _ctx, _d, _delta, _dilate, _gate_tail_p, _h, _k, _m, _m_grid, _minimize, _mu_q, _n_eff, _n_union_fa, _n_union_fa_gated, _pk_log_ms, _pk_log_neff, _pk_log_sigma, _plan_rng, _quiet_block_sigma, _quiet_block_sigmas, _quiet_concat, _recall_pivot, _recall_table, _run_segments, _runs, _s_eff, _score_grid_point, _seg_clean, _seg_inject_mask, _seg_injected, _seg_surrogate, _seg_truth, _segments, _sg_q, _share, _spacing, _stride, _surrogate_rng, _tau, _tol, _total, _ts_log_ms, _ts_log_sigma, _union_regions_injected, _wanted, _would_pass_gate, acceptance_floor_pA, adm, ax, axes, best, blocks, calibration_seconds, calibration_segments, cfg, cusum_calibration, cusum_calibration_ok, cusum_false_alarm_rate, cusum_false_alarm_rate_gated, cusum_grid_cells, cusum_grid_raw, cusum_grid_rows, cusum_recall_map, cusum_scale_ok, cusum_selected, cusum_selection_note, cusum_selection_notes, d, depth, dwell, dwell_samples, e, end, fig, find_quiet_segments, fit_gaussian_max, flank, g, gaussian_max_multiplier, grid, h, im, im2, injected_all, injection_plan, injection_pulse, k, means, n_eff_at_timescale, near, noise_timescale_rows, noise_timescales, out, passes_acceptance_gate, peak_floor_at_timescale, peak_gate_multiplier, peak_null_rows, peak_null_timescales, per_cell, phase_randomised_surrogate, pooled, pos, pulse, pulse_len, rm, s, scale, seg, seg_len, sel, sets_m, si, sigma_at_timescale, summary, tm
    check_cancelled(_acid_cancel)

    def find_quiet_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        quiet = ~mask
        diff = np.diff(quiet.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)
        return [(int(s), int(e - 1)) for s, e in zip(starts, stops) if e - s >= min_len]

    def injection_pulse(depth_pA: float, dwell_samples: int, tau_samples: float) -> np.ndarray:
        tail = int(math.ceil(8 * tau_samples))
        j = np.arange(dwell_samples + tail, dtype=float)
        rise = 1.0 - np.exp(-(j[:dwell_samples] + 1.0) / tau_samples)
        top = 1.0 - math.exp(-dwell_samples / tau_samples)
        fall = top * np.exp(-(j[dwell_samples:] - dwell_samples + 1.0) / tau_samples)
        return depth_pA * np.concatenate((rise, fall)).astype(np.float32)

    def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
        if margin <= 0:
            return mask
        from scipy.ndimage import maximum_filter1d
        return maximum_filter1d(mask.astype(np.uint8), size=2 * margin + 1).astype(bool)
    _cal_margin = int(round(CUSUM_CAL_MARGIN_MS * fs / 1000.0))
    _busy = _dilate(regions_to_mask(n_samples, threshold_candidate_regions + threshold_candidate_regions_res), _cal_margin)
    _segments = find_quiet_segments(_busy, int(round(CUSUM_CAL_MIN_SEGMENT_MS * fs / 1000.0)))
    _segments.sort(key=lambda se: se[1] - se[0], reverse=True)
    calibration_segments = []
    _total = 0
    for s, e in _segments:
        check_cancelled(_acid_cancel)
        if _total >= CUSUM_CAL_TOTAL_S * fs:
            break
        calibration_segments.append((s, e))
        _total += e - s + 1
    calibration_seconds = _total / fs
    if calibration_seconds <= 0:
        raise RuntimeError('No threshold-quiet segment of at least CUSUM_CAL_MIN_SEGMENT_MS found; lower it or the margin.')
    if 3.0 / calibration_seconds > CUSUM_FA_BUDGET_PER_S / 2:
        warnings.warn(f'{calibration_seconds:.0f} s of quiet recording can only certify a false-alarm rate down to ~{3.0 / calibration_seconds:.3f}/s (95% bound for zero observed alarms); admissibility uses the point estimate.')
    print(f'{len(calibration_segments)} threshold-quiet segments totalling {calibration_seconds:.1f} s selected (of {len(_segments)} segments >= {CUSUM_CAL_MIN_SEGMENT_MS:.0f} ms; {100 * (~_busy).mean():.1f}% of the recording is quiet).')
    _quiet_concat = [signal[s:e + 1].astype(np.float64) for s, e in calibration_segments]
    _m_grid = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]
    noise_timescale_rows = []
    for _m in _m_grid:
        check_cancelled(_acid_cancel)
        pooled = []
        for seg in _quiet_concat:
            check_cancelled(_acid_cancel)
            k = len(seg) // _m
            if k >= 4:
                means = seg[:k * _m].reshape(k, _m).mean(axis=1)
                pooled.append(means - np.median(means))
        if pooled:
            pooled = np.concatenate(pooled)
            noise_timescale_rows.append({'m_samples': _m, 'averaging_ms': _m / fs * 1000.0, 'sigma_m_pA': 1.4826 * float(np.median(np.abs(pooled))), 'n_means': int(len(pooled))})
    noise_timescales = pd.DataFrame(noise_timescale_rows)
    noise_timescales['white_reference_pA'] = noise_timescales['sigma_m_pA'].iloc[0] / np.sqrt(noise_timescales['m_samples'])
    noise_timescales['excess_over_white'] = noise_timescales['sigma_m_pA'] / noise_timescales['white_reference_pA']
    noise_timescales.to_csv(OUT_DIR / f'{TAG}_noise_timescales.csv', index=False)
    print('Quiet-trace noise versus averaging time (excess_over_white = 1 for white noise):')
    display(noise_timescales.round(3))
    plt.figure(figsize=(7, 4.5))
    plt.loglog(noise_timescales['averaging_ms'], noise_timescales['sigma_m_pA'], marker='o', label='measured sigma_m')
    plt.loglog(noise_timescales['averaging_ms'], noise_timescales['white_reference_pA'], linestyle='--', label='white noise sigma_1/sqrt(m)')
    plt.xlabel('Averaging time (ms)')
    plt.ylabel('MAD of block means (pA)')
    plt.title('Noise versus timescale on quiet segments')
    plt.legend(fontsize=9)
    finish_figure('00b_noise_timescales')
    _ts_log_ms = np.log(noise_timescales['averaging_ms'].to_numpy(float))
    _ts_log_sigma = np.log(noise_timescales['sigma_m_pA'].to_numpy(float))

    def sigma_at_timescale(dwell_ms) -> np.ndarray:
        """Quiet-trace noise (pA) at the averaging time closest to an event duration (log-log interpolation)."""
        x = np.log(np.clip(np.asarray(dwell_ms, float), np.exp(_ts_log_ms[0]), np.exp(_ts_log_ms[-1])))
        return np.exp(np.interp(x, _ts_log_ms, _ts_log_sigma))
    from scipy.optimize import minimize as _minimize
    _ctx = max(1, int(round(LOCAL_CONTEXT_MS * fs / 1000.0)))
    _gate_tail_p = float(norm.sf(SUSTAINED_GATE_SIGMA))

    def fit_gaussian_max(x, max_points: int=20000, seed: int=0) -> tuple[float, float]:
        """Maximum-likelihood (sigma_eff, n_eff >= 1) of the max-of-n-Gaussians model for pooled noise-only peaks."""
        x = np.asarray(x, float)
        if len(x) > max_points:
            x = np.random.default_rng(seed).choice(x, max_points, replace=False)
        s0 = max(1.4826 * float(np.median(np.abs(x - np.median(x)))), 1e-06)

        def nll(p):
            s, n = (np.exp(p[0]), 1.0 + np.exp(p[1]))
            z = x / s
            return -np.sum(np.log(n) - np.log(s) + norm.logpdf(z) + (n - 1.0) * norm.logcdf(z))
        best = None
        for n0 in (1.05, 5.0, 30.0):
            check_cancelled(_acid_cancel)
            r = _minimize(nll, x0=[np.log(s0), np.log(n0 - 1.0)], method='Nelder-Mead', options={'xatol': 0.0001, 'fatol': 0.001, 'maxiter': 2000})
            if best is None or r.fun < best.fun:
                best = r
        return (float(np.exp(best.x[0])), float(1.0 + np.exp(best.x[1])))

    def gaussian_max_multiplier(n_eff, tail_p: float) -> np.ndarray:
        """Multiple of the noise width exceeded with probability tail_p by the maximum of n_eff Gaussian draws."""
        log_F = np.log1p(-tail_p) / np.maximum(np.asarray(n_eff, float), 1.0)
        return norm.isf(-np.expm1(log_F))
    peak_null_rows = []
    for _m in _m_grid:
        check_cancelled(_acid_cancel)
        _stride = _m + 2 * _ctx
        pooled = []
        for seg in _quiet_concat:
            check_cancelled(_acid_cancel)
            k = len(seg) // _stride
            if k >= 4:
                blocks = seg[:k * _stride].reshape(k, _stride)
                flank = np.concatenate((blocks[:, :_ctx], blocks[:, _ctx + _m:]), axis=1)
                pooled.append(np.median(flank, axis=1) - blocks[:, _ctx:_ctx + _m].min(axis=1))
        if pooled and sum((len(p) for p in pooled)) >= 50:
            pooled = np.concatenate(pooled)
            _s_eff, _n_eff = fit_gaussian_max(pooled)
            peak_null_rows.append({'m_samples': _m, 'duration_ms': _m / fs * 1000.0, 'n_windows': int(len(pooled)), 'peak_null_median_pA': float(np.median(pooled)), 'sigma_eff_pA': _s_eff, 'n_eff': _n_eff, 'floor_pA': float(_s_eff * gaussian_max_multiplier(_n_eff, _gate_tail_p)), 'model_q999_pA': float(_s_eff * gaussian_max_multiplier(_n_eff, 0.001)), 'empirical_q999_pA': float(np.quantile(pooled, 0.999))})
    peak_null_timescales = pd.DataFrame(peak_null_rows)
    if len(peak_null_timescales) < 2:
        raise RuntimeError('Too little quiet trace to measure the noise-only peak depth versus duration.')
    peak_null_timescales.to_csv(OUT_DIR / f'{TAG}_peak_null_timescales.csv', index=False)
    print(f'Noise-only peak depth versus window duration on the quiet trace (max of n_eff Gaussian draws, fitted; floor = quantile at tail probability {_gate_tail_p:.1e} = Phi(-{SUSTAINED_GATE_SIGMA:g}); model vs empirical 99.9% quantile checks the fit):')
    display(peak_null_timescales.round(3))
    plt.figure(figsize=(7, 4.5))
    plt.loglog(peak_null_timescales['duration_ms'], peak_null_timescales['peak_null_median_pA'].clip(lower=0.001), marker='o', label='noise-only peak depth, median')
    plt.loglog(peak_null_timescales['duration_ms'], peak_null_timescales['floor_pA'], marker='s', label=f'acceptance floor (Phi(-{SUSTAINED_GATE_SIGMA:g}) quantile)')
    plt.loglog(noise_timescales['averaging_ms'], SUSTAINED_GATE_SIGMA * noise_timescales['sigma_m_pA'], linestyle='--', label=f'{SUSTAINED_GATE_SIGMA:g} x sigma_m (previous median-depth floor)')
    plt.xlabel('Window / event duration (ms)')
    plt.ylabel('Depth (pA)')
    plt.title('Minimum-value floor versus duration on quiet segments')
    plt.legend(fontsize=9)
    finish_figure('00c_peak_null_timescales')
    _pk_log_ms = np.log(peak_null_timescales['duration_ms'].to_numpy(float))
    _pk_log_sigma = np.log(peak_null_timescales['sigma_eff_pA'].to_numpy(float))
    _pk_log_neff = np.log(peak_null_timescales['n_eff'].to_numpy(float))

    def n_eff_at_timescale(dwell_ms) -> np.ndarray:
        """Independent noise samples inside a window of the given duration: log-log interpolation of the fit
    inside the measured range; beyond the longest measured window n_eff grows in proportion to duration
    (stationary noise), so very long events are not judged with the 50 ms multiplier."""
        d = np.asarray(dwell_ms, float)
        x = np.log(np.clip(d, np.exp(_pk_log_ms[0]), np.exp(_pk_log_ms[-1])))
        n = np.exp(np.interp(x, _pk_log_ms, _pk_log_neff))
        return n * np.maximum(d / np.exp(_pk_log_ms[-1]), 1.0)

    def peak_gate_multiplier(dwell_ms) -> np.ndarray:
        """Noise multiple the peak must clear at a duration: the k-sigma rule corrected for n_eff samples."""
        return gaussian_max_multiplier(n_eff_at_timescale(dwell_ms), _gate_tail_p)

    def peak_floor_at_timescale(dwell_ms) -> np.ndarray:
        """Floor (pA) for the peak depth at an event duration, from the quiet-trace sigma_eff and n_eff."""
        x = np.log(np.clip(np.asarray(dwell_ms, float), np.exp(_pk_log_ms[0]), np.exp(_pk_log_ms[-1])))
        return np.exp(np.interp(x, _pk_log_ms, _pk_log_sigma)) * peak_gate_multiplier(dwell_ms)
    _quiet_block_sigmas = []
    for seg in _quiet_concat:
        check_cancelled(_acid_cancel)
        _mu_q, _sg_q = block_statistics(seg.astype(np.float32), max(50, int(round(CUSUM_BLOCK_MS * fs / 1000.0))))
        _quiet_block_sigmas.append(_sg_q)
    _quiet_block_sigma = float(np.median(np.concatenate(_quiet_block_sigmas)))

    def acceptance_floor_pA(dwell_ms, block_noise_pA=None) -> np.ndarray:
        """Floor (pA) the gate statistic must clear at an event duration.
    GATE_STATISTIC = "peak": noise width x the max-of-n_eff multiplier at that duration, the width being
    the quiet-trace sigma_eff (timescale mode), the noise passed in (sigma mode) or the recording-wide
    quiet-block noise (absolute mode).
    GATE_STATISTIC = "sustained": SUSTAINED_GATE_SIGMA x the noise at the event's averaging time (timescale)
    or x the flat noise (other modes)."""
        dwell_ms = np.asarray(dwell_ms, float)
        if GATE_STATISTIC == 'peak':
            if SUSTAINED_GATE_MODE == 'timescale':
                return peak_floor_at_timescale(dwell_ms)
            if SUSTAINED_GATE_MODE == 'sigma' and block_noise_pA is not None:
                return np.asarray(block_noise_pA, float) * peak_gate_multiplier(dwell_ms)
            return _quiet_block_sigma * peak_gate_multiplier(dwell_ms)
        if SUSTAINED_GATE_MODE == 'timescale':
            return SUSTAINED_GATE_SIGMA * sigma_at_timescale(dwell_ms)
        if SUSTAINED_GATE_MODE == 'sigma' and block_noise_pA is not None:
            return SUSTAINED_GATE_SIGMA * np.asarray(block_noise_pA, float)
        return np.full(dwell_ms.shape, SUSTAINED_GATE_SIGMA * _quiet_block_sigma)

    def passes_acceptance_gate(sustained_depth_pA, half_peak_mean_pA, half_peak_ms, dwell_ms, block_noise_pA=None, *, peak_depth_pA=None, local_noise_pA=None) -> np.ndarray:
        """GATE_STATISTIC = "peak": the minimum value inside the event (peak depth) must clear the floor of
    acceptance_floor_pA at its duration (in sigma mode the per-event local noise sets the width).
    GATE_STATISTIC = "sustained": under SUSTAINED_GATE_MODE = "timescale" only, the previous timescale
    rule applies - a level shift is credible if EITHER its median depth clears the floor at the full
    duration (plateau path) OR the mean depth of its samples above half the peak clears the floor at
    the duration of that core (pulse path). Under "sigma"/"absolute" there is no pulse path: only the
    median (plateau) depth is compared to the flat noise-relative floor."""
        dwell = np.asarray(dwell_ms, float)
        if GATE_STATISTIC == 'peak':
            if peak_depth_pA is None:
                raise ValueError("passes_acceptance_gate needs peak_depth_pA when GATE_STATISTIC = 'peak'")
            peak = np.asarray(peak_depth_pA, float)
            noise = local_noise_pA if local_noise_pA is not None else block_noise_pA
            return np.isfinite(peak) & (peak >= acceptance_floor_pA(dwell, noise))
        sustained = np.asarray(sustained_depth_pA, float)
        core_mean = np.asarray(half_peak_mean_pA, float)
        core_ms = np.asarray(half_peak_ms, float)
        if SUSTAINED_GATE_MODE == 'timescale':
            plateau_ok = sustained >= acceptance_floor_pA(dwell)
            pulse_ok = np.isfinite(core_mean) & (core_mean >= acceptance_floor_pA(np.maximum(core_ms, 3.0 / fs * 1000.0)))
            return plateau_ok | pulse_ok
        return sustained >= acceptance_floor_pA(dwell, block_noise_pA)
    print(f'Acceptance-floor preview ({SUSTAINED_GATE_MODE} mode, {GATE_STATISTIC} statistic): ' + ', '.join((f'{d} ms -> {float(acceptance_floor_pA(d)):.1f} pA' for d in (0.1, 0.2, 0.5, 1.0, 5.0, 20.0))))
    _tau = CUSUM_INJECT_EDGE_TAU_MS * fs / 1000.0
    _spacing = int(round(CUSUM_CAL_SPACING_MS * fs / 1000.0))
    _cells = [(d, w) for d in CUSUM_INJECT_DEPTHS_PA for w in CUSUM_INJECT_DWELLS_MS]
    _plan_rng = np.random.default_rng(SEED)
    _wanted = list(_cells) * CUSUM_INJECT_PER_CELL
    _plan_rng.shuffle(_wanted)
    injection_plan = []
    _k = 0
    for si, (s, e) in enumerate(calibration_segments):
        check_cancelled(_acid_cancel)
        pos = int(round(0.4 * _spacing))
        seg_len = e - s + 1
        while _k < len(_wanted):
            check_cancelled(_acid_cancel)
            depth, dwell = _wanted[_k]
            dwell_samples = max(1, int(round(dwell * fs / 1000.0)))
            pulse_len = dwell_samples + int(math.ceil(8 * _tau))
            if pos + pulse_len + int(0.4 * _spacing) > seg_len:
                break
            injection_plan.append((si, pos, float(depth), float(dwell), dwell_samples))
            _k += 1
            pos += max(_spacing, pulse_len + int(0.4 * _spacing))
        if _k >= len(_wanted):
            break
    if _k < len(_wanted):
        warnings.warn(f'Only {_k} of {len(_wanted)} planned injections fit into the quiet segments.')
    print(f'Injections placed: {len(injection_plan):,} over {len(_cells)} (depth, dwell) cells.')
    _seg_clean = [signal[s:e + 1].astype(np.float32) for s, e in calibration_segments]

    def phase_randomised_surrogate(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(seg, dtype=np.float64)
        mean = x.mean()
        spec = np.fft.rfft(x - mean)
        phases = rng.uniform(0.0, 2.0 * np.pi, len(spec))
        phases[0] = 0.0
        if len(x) % 2 == 0:
            phases[-1] = 0.0
        surrogate = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=len(x)) + mean
        return surrogate.astype(np.float32)
    _surrogate_rng = np.random.default_rng(SEED + 7)
    _seg_surrogate = [phase_randomised_surrogate(seg, _surrogate_rng) for seg in _seg_clean]
    _seg_injected = [seg.copy() for seg in _seg_clean]
    _seg_inject_mask = [np.zeros(len(seg), dtype=bool) for seg in _seg_clean]
    _seg_truth = [[] for _ in _seg_clean]
    for si, pos, depth, dwell, dwell_samples in injection_plan:
        check_cancelled(_acid_cancel)
        pulse = injection_pulse(depth, dwell_samples, _tau)
        end = min(len(_seg_injected[si]), pos + len(pulse))
        _seg_injected[si][pos:end] -= pulse[:end - pos]
        _seg_inject_mask[si][pos:end] = True
        _seg_truth[si].append((pos, pos + dwell_samples - 1, depth, dwell))
    _tol = int(round(MATCH_TOLERANCE_MS * fs / 1000.0)) + int(math.ceil(_tau))
    CUSUM_SCALES = {'pulse': {'block_ms': CUSUM_PULSE_BLOCK_MS, 'pass1_block_ms': CUSUM_PASS1_BLOCK_MS, 'delta_grid': CUSUM_PULSE_DELTA_GRID, 'h_grid': CUSUM_PULSE_H_GRID, 'cell_mask': lambda cells: cells['dwell_ms'] <= CUSUM_PULSE_MAX_DWELL_MS, 'override': CUSUM_PULSE_OVERRIDE}, 'level': {'block_ms': CUSUM_BLOCK_MS, 'pass1_block_ms': CUSUM_PASS1_BLOCK_MS, 'delta_grid': CUSUM_LEVEL_DELTA_GRID, 'h_grid': CUSUM_LEVEL_H_GRID, 'cell_mask': lambda cells: cells['dwell_ms'] >= CUSUM_LEVEL_MIN_DWELL_MS, 'override': CUSUM_LEVEL_OVERRIDE}}
    if not CUSUM_DUAL_SCALE:
        CUSUM_SCALES = {'level': dict(CUSUM_SCALES['level'], cell_mask=lambda cells: cells['dwell_ms'] >= 0.0)}
    _share = 1.0 / len(CUSUM_SCALES)

    def _run_segments(segments, delta, h, block_ms, pass1_block_ms, masks=None):
        regions_out, tables_out = ([], [])
        for k, seg in enumerate(segments):
            check_cancelled(_acid_cancel)
            regions, table, _ = cusum_detect(seg, fs=fs, delta=delta, h=h, end_tol_sigma=CUSUM_END_TOL_SIGMA, block_ms=block_ms, level_est_ms=CUSUM_LEVEL_EST_MS, max_event_s=CUSUM_MAX_EVENT_S, min_event_ms=MIN_EVENT_MS, passes=2, pass1_block_ms=pass1_block_ms, mask_margin_ms=CUSUM_CAL_MARGIN_MS, pre_baseline_ms=CUSUM_PRE_BASELINE_MS, exclude_mask=None if masks is None else masks[k])
            regions_out.append(regions)
            tables_out.append(table)
        return (regions_out, tables_out)

    def _would_pass_gate(table: pd.DataFrame) -> int:
        if not len(table):
            return 0
        dwell = (table['end_idx'] - table['start_idx'] + 1).to_numpy(float) / fs * 1000.0
        ok = passes_acceptance_gate(table['cusum_sustained_depth_pA'], table['cusum_half_peak_mean_pA'], table['cusum_half_peak_ms'], dwell, table['cusum_block_noise_pA'].to_numpy(float), peak_depth_pA=table['cusum_peak_depth_pA'], local_noise_pA=table['cusum_block_noise_pA'].to_numpy(float))
        return int(np.sum(ok))

    def _recall_table(injected_regions) -> tuple[pd.DataFrame, int]:
        cell_rows = []
        fa_injected = 0
        for si, regions in enumerate(injected_regions):
            check_cancelled(_acid_cancel)
            truth = _seg_truth[si]
            used = np.zeros(len(regions), dtype=bool)
            r_start = np.array([r[0] for r in regions], dtype=int)
            r_end = np.array([r[1] for r in regions], dtype=int)
            for ts, te, depth, dwell in truth:
                check_cancelled(_acid_cancel)
                hit = np.flatnonzero((r_end >= ts - _tol) & (r_start <= te + _tol))
                if len(hit):
                    lengths_ms = (r_end[hit] - r_start[hit] + 1) / fs * 1000.0
                    hit = hit[lengths_ms <= 10.0 * dwell + 0.5]
                if len(hit):
                    used[hit] = True
                    detected_dwell = float(np.sum(r_end[hit] - r_start[hit] + 1)) / fs * 1000.0
                    cell_rows.append({'depth_pA': depth, 'dwell_ms': dwell, 'recovered': 1, 'dwell_bias_ms': detected_dwell - dwell})
                else:
                    cell_rows.append({'depth_pA': depth, 'dwell_ms': dwell, 'recovered': 0, 'dwell_bias_ms': np.nan})
            fa_injected += int(np.sum(~used))
        cells = pd.DataFrame(cell_rows)
        per_cell = cells.groupby(['depth_pA', 'dwell_ms']).agg(recall=('recovered', 'mean'), n=('recovered', 'size'), dwell_bias_ms=('dwell_bias_ms', 'median')).reset_index()
        return (per_cell, fa_injected)

    def _score_grid_point(scale: str, delta: float, h: float) -> tuple[dict, pd.DataFrame, dict]:
        cfg = CUSUM_SCALES[scale]
        surrogate, surrogate_tables = _run_segments(_seg_surrogate, delta, h, cfg['block_ms'], cfg['pass1_block_ms'])
        n_fa = sum((len(r) for r in surrogate))
        n_fa_gated = sum((_would_pass_gate(t) for t in surrogate_tables))
        injected, _ = _run_segments(_seg_injected, delta, h, cfg['block_ms'], cfg['pass1_block_ms'], masks=_seg_inject_mask)
        per_cell, fa_injected = _recall_table(injected)
        target = per_cell.loc[(per_cell['depth_pA'] >= CUSUM_TARGET_MIN_DEPTH_PA) & (per_cell['dwell_ms'] >= CUSUM_TARGET_MIN_DWELL_MS)]
        scale_cells = target.loc[cfg['cell_mask'](target)]
        summary = {'scale': scale, 'delta': delta, 'h': h, 'false_alarms_per_s_clean': n_fa / calibration_seconds, 'false_alarms_per_s_gated': n_fa_gated / calibration_seconds, 'false_alarms_per_s_gated_upper95': (n_fa_gated + 3.0 if n_fa_gated == 0 else n_fa_gated + 1.96 * math.sqrt(n_fa_gated) + 1.0) / calibration_seconds, 'false_alarms_per_s_injected': fa_injected / calibration_seconds, 'recall_scale_cells': float(scale_cells['recall'].mean()) if len(scale_cells) else np.nan, 'recall_target': float(target['recall'].mean()) if len(target) else np.nan, 'recall_all': float(per_cell['recall'].mean())}
        runs = {'surrogate': surrogate, 'surrogate_tables': surrogate_tables, 'injected': injected}
        return (summary, per_cell, runs)
    cusum_grid_rows, cusum_grid_cells, cusum_grid_raw = ([], {}, {})
    for scale, cfg in CUSUM_SCALES.items():
        check_cancelled(_acid_cancel)
        for _delta in tqdm(cfg['delta_grid'], desc=f'CUSUM calibration ({scale} scale)'):
            check_cancelled(_acid_cancel)
            for _h in cfg['h_grid']:
                check_cancelled(_acid_cancel)
                summary, per_cell, _runs = _score_grid_point(scale, float(_delta), float(_h))
                cusum_grid_rows.append(summary)
                cusum_grid_cells[scale, float(_delta), float(_h)] = per_cell
                cusum_grid_raw[scale, float(_delta), float(_h)] = _runs
    cusum_calibration = pd.DataFrame(cusum_grid_rows)
    cusum_calibration['admissible'] = (cusum_calibration['false_alarms_per_s_gated'] <= CUSUM_FA_BUDGET_PER_S * _share) & (cusum_calibration['false_alarms_per_s_clean'] <= CUSUM_RAW_FA_MAX_PER_S * _share)
    cusum_calibration.to_csv(OUT_DIR / f'{TAG}_cusum_calibration_grid.csv', index=False)
    cusum_selected = {}
    cusum_selection_notes = {}
    cusum_scale_ok = {}
    for scale, cfg in CUSUM_SCALES.items():
        check_cancelled(_acid_cancel)
        grid = cusum_calibration.loc[cusum_calibration['scale'] == scale]
        if cfg['override'] is not None:
            d, h = (float(cfg['override'][0]), float(cfg['override'][1]))
            if (scale, d, h) not in cusum_grid_cells:
                summary, per_cell, _runs = _score_grid_point(scale, d, h)
                summary['admissible'] = bool(summary['false_alarms_per_s_gated'] <= CUSUM_FA_BUDGET_PER_S * _share and summary['false_alarms_per_s_clean'] <= CUSUM_RAW_FA_MAX_PER_S * _share)
                cusum_calibration = pd.concat([cusum_calibration, pd.DataFrame([summary])], ignore_index=True)
                cusum_grid_cells[scale, d, h] = per_cell
                cusum_grid_raw[scale, d, h] = _runs
                grid = cusum_calibration.loc[cusum_calibration['scale'] == scale]
            cusum_selected[scale] = (d, h)
            cusum_selection_notes[scale] = 'manual override'
            cusum_scale_ok[scale] = bool(grid.loc[(grid['delta'] == d) & (grid['h'] == h), 'admissible'].iloc[0])
            continue
        adm = grid.loc[grid['admissible']]
        if len(adm):
            near = adm.loc[adm['recall_scale_cells'] >= adm['recall_scale_cells'].max() - CUSUM_RECALL_TOLERANCE]
            best = near.sort_values(['false_alarms_per_s_clean', 'recall_scale_cells'], ascending=[True, False]).iloc[0]
            cusum_selection_notes[scale] = f'recall within {CUSUM_RECALL_TOLERANCE:.0%} of the best admissible point, fewest raw alarms (gate-surviving false alarms <= {CUSUM_FA_BUDGET_PER_S * _share:.3f}/s, raw <= {CUSUM_RAW_FA_MAX_PER_S * _share:.2f}/s)'
            cusum_scale_ok[scale] = True
        else:
            best = grid.sort_values(['false_alarms_per_s_gated', 'false_alarms_per_s_clean']).iloc[0]
            cusum_selection_notes[scale] = 'NO grid point met the false-alarm budget; lowest-false-alarm point used'
            cusum_scale_ok[scale] = False
            warnings.warn(f'[{scale} scale] ' + cusum_selection_notes[scale])
        cusum_selected[scale] = (float(best['delta']), float(best['h']))
    cusum_calibration_ok = all(cusum_scale_ok.values())
    _union_regions_injected = []
    _n_union_fa, _n_union_fa_gated = (0, 0)
    for si in range(len(_seg_clean)):
        check_cancelled(_acid_cancel)
        sets_m, injected_all = ([], [])
        for scale, (d, h) in cusum_selected.items():
            check_cancelled(_acid_cancel)
            _runs = cusum_grid_raw[scale, d, h]
            sets_m.append((_runs['surrogate'][si], _runs['surrogate_tables'][si], scale))
            injected_all.extend(_runs['injected'][si])
        rm, tm = merge_region_sets(_seg_surrogate[si], sets_m, fs_merge=fs)
        _n_union_fa += len(rm)
        _n_union_fa_gated += _would_pass_gate(tm)
        injected_all.sort()
        out = []
        for s, e in injected_all:
            check_cancelled(_acid_cancel)
            if out and s <= out[-1][1] + 1:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        _union_regions_injected.append(out)
    cusum_false_alarm_rate = _n_union_fa / calibration_seconds
    cusum_false_alarm_rate_gated = _n_union_fa_gated / calibration_seconds
    cusum_recall_map, _ = _recall_table(_union_regions_injected)
    cusum_recall_map.to_csv(OUT_DIR / f'{TAG}_cusum_recall_map.csv', index=False)
    CUSUM_DELTA, CUSUM_H = cusum_selected['level']
    CUSUM_PULSE_DELTA, CUSUM_PULSE_H = cusum_selected.get('pulse', (np.nan, np.nan))
    cusum_selection_note = '; '.join((f'{k}: delta={v[0]}, h={v[1]} ({cusum_selection_notes[k]})' for k, v in cusum_selected.items()))
    print('Calibration grid. false_alarms_per_s_clean = alarms on phase-randomised SURROGATES of the quiet trace (noise only); false_alarms_per_s_gated = those that would also pass the acceptance gate; recall_scale_cells = mean recall over the cells that scale is responsible for; the budget and raw cap are split equally between scales.')
    display(cusum_calibration.round(4))
    for scale, (d, h) in cusum_selected.items():
        check_cancelled(_acid_cancel)
        print(f'Selected {scale} scale: delta = {d}, h = {h} -> {cusum_selection_notes[scale]}')
    print(f'Union of the selected scales on the surrogate noise: {cusum_false_alarm_rate:.4f} raw alarms/s (~{cusum_false_alarm_rate * duration_s:.0f} over the recording), of which {cusum_false_alarm_rate_gated:.4f}/s (~{cusum_false_alarm_rate_gated * duration_s:.0f}) would pass the acceptance gate.')
    if not cusum_calibration_ok:
        print('!! At least one scale did not meet its false-alarm budget. Unless CUSUM_FALLBACK_TO_MATCHED is False, the primary set will be switched to cusum_matched (CUSUM events confirmed by the threshold detector).')
    _recall_pivot = cusum_recall_map.pivot(index='depth_pA', columns='dwell_ms', values='recall')
    _bias_pivot = cusum_recall_map.pivot(index='depth_pA', columns='dwell_ms', values='dwell_bias_ms')
    print('Recall of the union detector by injected depth (rows, pA) and dwell (columns, ms):')
    display(_recall_pivot.round(2))
    print('Median dwell bias of recovered injections (detected - true, ms):')
    display(_bias_pivot.round(3))
    fig, axes = plt.subplots(1, 4, figsize=(23, 5))
    im = axes[0].imshow(_recall_pivot.to_numpy(), vmin=0, vmax=1, cmap='viridis', aspect='auto', origin='lower')
    axes[0].set_xticks(range(len(_recall_pivot.columns)))
    axes[0].set_xticklabels(_recall_pivot.columns)
    axes[0].set_yticks(range(len(_recall_pivot.index)))
    axes[0].set_yticklabels(_recall_pivot.index)
    axes[0].set_xlabel('Injected dwell (ms)')
    axes[0].set_ylabel('Injected depth (pA)')
    axes[0].set_title('Recall of the union detector')
    plt.colorbar(im, ax=axes[0])
    for ax, scale in zip(axes[1:3], list(CUSUM_SCALES)):
        check_cancelled(_acid_cancel)
        grid = cusum_calibration.loc[cusum_calibration['scale'] == scale]
        for _d in sorted(grid['delta'].unique()):
            check_cancelled(_acid_cancel)
            g = grid.loc[grid['delta'] == _d].sort_values('h')
            ax.plot(g['false_alarms_per_s_gated'], g['recall_scale_cells'], marker='o', label=f'delta={_d}')
        d, h = cusum_selected[scale]
        sel = grid.loc[(grid['delta'] == d) & (grid['h'] == h)]
        ax.scatter(sel['false_alarms_per_s_gated'], sel['recall_scale_cells'], s=140, facecolors='none', edgecolors='k', linewidths=2, zorder=5, label='selected')
        ax.axvline(CUSUM_FA_BUDGET_PER_S * _share, linestyle=':', linewidth=1)
        ax.set_xscale('symlog', linthresh=0.001)
        ax.set_xlabel('Gate-surviving false alarms per second')
        ax.set_ylabel(f'Mean recall, {scale}-scale cells')
        ax.set_title(f"{scale} scale (block {CUSUM_SCALES[scale]['block_ms']:g} ms)")
        ax.legend(fontsize=7)
    if len(CUSUM_SCALES) == 1:
        axes[2].axis('off')
    _bmax = np.nanmax(np.abs(_bias_pivot.to_numpy())) if np.isfinite(_bias_pivot.to_numpy()).any() else 1.0
    im2 = axes[3].imshow(_bias_pivot.to_numpy(), cmap='coolwarm', aspect='auto', origin='lower', vmin=-_bmax, vmax=_bmax)
    axes[3].set_xticks(range(len(_bias_pivot.columns)))
    axes[3].set_xticklabels(_bias_pivot.columns)
    axes[3].set_yticks(range(len(_bias_pivot.index)))
    axes[3].set_yticklabels(_bias_pivot.index)
    axes[3].set_xlabel('Injected dwell (ms)')
    axes[3].set_ylabel('Injected depth (pA)')
    axes[3].set_title('Dwell bias (detected - true, ms)')
    plt.colorbar(im2, ax=axes[3])
    finish_figure('00_cusum_calibration')


def definitions_017():
    global _dilate, _minimize, _recall_table, _run_segments, _score_grid_point, _would_pass_gate, acceptance_floor_pA, find_quiet_segments, fit_gaussian_max, gaussian_max_multiplier, injection_pulse, n_eff_at_timescale, passes_acceptance_gate, peak_floor_at_timescale, peak_gate_multiplier, phase_randomised_surrogate, sigma_at_timescale

    def find_quiet_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        quiet = ~mask
        diff = np.diff(quiet.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)
        return [(int(s), int(e - 1)) for s, e in zip(starts, stops) if e - s >= min_len]

    def injection_pulse(depth_pA: float, dwell_samples: int, tau_samples: float) -> np.ndarray:
        tail = int(math.ceil(8 * tau_samples))
        j = np.arange(dwell_samples + tail, dtype=float)
        rise = 1.0 - np.exp(-(j[:dwell_samples] + 1.0) / tau_samples)
        top = 1.0 - math.exp(-dwell_samples / tau_samples)
        fall = top * np.exp(-(j[dwell_samples:] - dwell_samples + 1.0) / tau_samples)
        return depth_pA * np.concatenate((rise, fall)).astype(np.float32)

    def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
        if margin <= 0:
            return mask
        from scipy.ndimage import maximum_filter1d
        return maximum_filter1d(mask.astype(np.uint8), size=2 * margin + 1).astype(bool)

    def sigma_at_timescale(dwell_ms) -> np.ndarray:
        """Quiet-trace noise (pA) at the averaging time closest to an event duration (log-log interpolation)."""
        x = np.log(np.clip(np.asarray(dwell_ms, float), np.exp(_ts_log_ms[0]), np.exp(_ts_log_ms[-1])))
        return np.exp(np.interp(x, _ts_log_ms, _ts_log_sigma))
    from scipy.optimize import minimize as _minimize

    def fit_gaussian_max(x, max_points: int=20000, seed: int=0) -> tuple[float, float]:
        """Maximum-likelihood (sigma_eff, n_eff >= 1) of the max-of-n-Gaussians model for pooled noise-only peaks."""
        x = np.asarray(x, float)
        if len(x) > max_points:
            x = np.random.default_rng(seed).choice(x, max_points, replace=False)
        s0 = max(1.4826 * float(np.median(np.abs(x - np.median(x)))), 1e-06)

        def nll(p):
            s, n = (np.exp(p[0]), 1.0 + np.exp(p[1]))
            z = x / s
            return -np.sum(np.log(n) - np.log(s) + norm.logpdf(z) + (n - 1.0) * norm.logcdf(z))
        best = None
        for n0 in (1.05, 5.0, 30.0):
            check_cancelled(_acid_cancel)
            r = _minimize(nll, x0=[np.log(s0), np.log(n0 - 1.0)], method='Nelder-Mead', options={'xatol': 0.0001, 'fatol': 0.001, 'maxiter': 2000})
            if best is None or r.fun < best.fun:
                best = r
        return (float(np.exp(best.x[0])), float(1.0 + np.exp(best.x[1])))

    def gaussian_max_multiplier(n_eff, tail_p: float) -> np.ndarray:
        """Multiple of the noise width exceeded with probability tail_p by the maximum of n_eff Gaussian draws."""
        log_F = np.log1p(-tail_p) / np.maximum(np.asarray(n_eff, float), 1.0)
        return norm.isf(-np.expm1(log_F))

    def n_eff_at_timescale(dwell_ms) -> np.ndarray:
        """Independent noise samples inside a window of the given duration: log-log interpolation of the fit
    inside the measured range; beyond the longest measured window n_eff grows in proportion to duration
    (stationary noise), so very long events are not judged with the 50 ms multiplier."""
        d = np.asarray(dwell_ms, float)
        x = np.log(np.clip(d, np.exp(_pk_log_ms[0]), np.exp(_pk_log_ms[-1])))
        n = np.exp(np.interp(x, _pk_log_ms, _pk_log_neff))
        return n * np.maximum(d / np.exp(_pk_log_ms[-1]), 1.0)

    def peak_gate_multiplier(dwell_ms) -> np.ndarray:
        """Noise multiple the peak must clear at a duration: the k-sigma rule corrected for n_eff samples."""
        return gaussian_max_multiplier(n_eff_at_timescale(dwell_ms), _gate_tail_p)

    def peak_floor_at_timescale(dwell_ms) -> np.ndarray:
        """Floor (pA) for the peak depth at an event duration, from the quiet-trace sigma_eff and n_eff."""
        x = np.log(np.clip(np.asarray(dwell_ms, float), np.exp(_pk_log_ms[0]), np.exp(_pk_log_ms[-1])))
        return np.exp(np.interp(x, _pk_log_ms, _pk_log_sigma)) * peak_gate_multiplier(dwell_ms)

    def acceptance_floor_pA(dwell_ms, block_noise_pA=None) -> np.ndarray:
        """Floor (pA) the gate statistic must clear at an event duration.
    GATE_STATISTIC = "peak": noise width x the max-of-n_eff multiplier at that duration, the width being
    the quiet-trace sigma_eff (timescale mode), the noise passed in (sigma mode) or the recording-wide
    quiet-block noise (absolute mode).
    GATE_STATISTIC = "sustained": SUSTAINED_GATE_SIGMA x the noise at the event's averaging time (timescale)
    or x the flat noise (other modes)."""
        dwell_ms = np.asarray(dwell_ms, float)
        if GATE_STATISTIC == 'peak':
            if SUSTAINED_GATE_MODE == 'timescale':
                return peak_floor_at_timescale(dwell_ms)
            if SUSTAINED_GATE_MODE == 'sigma' and block_noise_pA is not None:
                return np.asarray(block_noise_pA, float) * peak_gate_multiplier(dwell_ms)
            return _quiet_block_sigma * peak_gate_multiplier(dwell_ms)
        if SUSTAINED_GATE_MODE == 'timescale':
            return SUSTAINED_GATE_SIGMA * sigma_at_timescale(dwell_ms)
        if SUSTAINED_GATE_MODE == 'sigma' and block_noise_pA is not None:
            return SUSTAINED_GATE_SIGMA * np.asarray(block_noise_pA, float)
        return np.full(dwell_ms.shape, SUSTAINED_GATE_SIGMA * _quiet_block_sigma)

    def passes_acceptance_gate(sustained_depth_pA, half_peak_mean_pA, half_peak_ms, dwell_ms, block_noise_pA=None, *, peak_depth_pA=None, local_noise_pA=None) -> np.ndarray:
        """GATE_STATISTIC = "peak": the minimum value inside the event (peak depth) must clear the floor of
    acceptance_floor_pA at its duration (in sigma mode the per-event local noise sets the width).
    GATE_STATISTIC = "sustained": under SUSTAINED_GATE_MODE = "timescale" only, the previous timescale
    rule applies - a level shift is credible if EITHER its median depth clears the floor at the full
    duration (plateau path) OR the mean depth of its samples above half the peak clears the floor at
    the duration of that core (pulse path). Under "sigma"/"absolute" there is no pulse path: only the
    median (plateau) depth is compared to the flat noise-relative floor."""
        dwell = np.asarray(dwell_ms, float)
        if GATE_STATISTIC == 'peak':
            if peak_depth_pA is None:
                raise ValueError("passes_acceptance_gate needs peak_depth_pA when GATE_STATISTIC = 'peak'")
            peak = np.asarray(peak_depth_pA, float)
            noise = local_noise_pA if local_noise_pA is not None else block_noise_pA
            return np.isfinite(peak) & (peak >= acceptance_floor_pA(dwell, noise))
        sustained = np.asarray(sustained_depth_pA, float)
        core_mean = np.asarray(half_peak_mean_pA, float)
        core_ms = np.asarray(half_peak_ms, float)
        if SUSTAINED_GATE_MODE == 'timescale':
            plateau_ok = sustained >= acceptance_floor_pA(dwell)
            pulse_ok = np.isfinite(core_mean) & (core_mean >= acceptance_floor_pA(np.maximum(core_ms, 3.0 / fs * 1000.0)))
            return plateau_ok | pulse_ok
        return sustained >= acceptance_floor_pA(dwell, block_noise_pA)

    def phase_randomised_surrogate(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(seg, dtype=np.float64)
        mean = x.mean()
        spec = np.fft.rfft(x - mean)
        phases = rng.uniform(0.0, 2.0 * np.pi, len(spec))
        phases[0] = 0.0
        if len(x) % 2 == 0:
            phases[-1] = 0.0
        surrogate = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=len(x)) + mean
        return surrogate.astype(np.float32)

    def _run_segments(segments, delta, h, block_ms, pass1_block_ms, masks=None):
        regions_out, tables_out = ([], [])
        for k, seg in enumerate(segments):
            check_cancelled(_acid_cancel)
            regions, table, _ = cusum_detect(seg, fs=fs, delta=delta, h=h, end_tol_sigma=CUSUM_END_TOL_SIGMA, block_ms=block_ms, level_est_ms=CUSUM_LEVEL_EST_MS, max_event_s=CUSUM_MAX_EVENT_S, min_event_ms=MIN_EVENT_MS, passes=2, pass1_block_ms=pass1_block_ms, mask_margin_ms=CUSUM_CAL_MARGIN_MS, pre_baseline_ms=CUSUM_PRE_BASELINE_MS, exclude_mask=None if masks is None else masks[k])
            regions_out.append(regions)
            tables_out.append(table)
        return (regions_out, tables_out)

    def _would_pass_gate(table: pd.DataFrame) -> int:
        if not len(table):
            return 0
        dwell = (table['end_idx'] - table['start_idx'] + 1).to_numpy(float) / fs * 1000.0
        ok = passes_acceptance_gate(table['cusum_sustained_depth_pA'], table['cusum_half_peak_mean_pA'], table['cusum_half_peak_ms'], dwell, table['cusum_block_noise_pA'].to_numpy(float), peak_depth_pA=table['cusum_peak_depth_pA'], local_noise_pA=table['cusum_block_noise_pA'].to_numpy(float))
        return int(np.sum(ok))

    def _recall_table(injected_regions) -> tuple[pd.DataFrame, int]:
        cell_rows = []
        fa_injected = 0
        for si, regions in enumerate(injected_regions):
            check_cancelled(_acid_cancel)
            truth = _seg_truth[si]
            used = np.zeros(len(regions), dtype=bool)
            r_start = np.array([r[0] for r in regions], dtype=int)
            r_end = np.array([r[1] for r in regions], dtype=int)
            for ts, te, depth, dwell in truth:
                check_cancelled(_acid_cancel)
                hit = np.flatnonzero((r_end >= ts - _tol) & (r_start <= te + _tol))
                if len(hit):
                    lengths_ms = (r_end[hit] - r_start[hit] + 1) / fs * 1000.0
                    hit = hit[lengths_ms <= 10.0 * dwell + 0.5]
                if len(hit):
                    used[hit] = True
                    detected_dwell = float(np.sum(r_end[hit] - r_start[hit] + 1)) / fs * 1000.0
                    cell_rows.append({'depth_pA': depth, 'dwell_ms': dwell, 'recovered': 1, 'dwell_bias_ms': detected_dwell - dwell})
                else:
                    cell_rows.append({'depth_pA': depth, 'dwell_ms': dwell, 'recovered': 0, 'dwell_bias_ms': np.nan})
            fa_injected += int(np.sum(~used))
        cells = pd.DataFrame(cell_rows)
        per_cell = cells.groupby(['depth_pA', 'dwell_ms']).agg(recall=('recovered', 'mean'), n=('recovered', 'size'), dwell_bias_ms=('dwell_bias_ms', 'median')).reset_index()
        return (per_cell, fa_injected)

    def _score_grid_point(scale: str, delta: float, h: float) -> tuple[dict, pd.DataFrame, dict]:
        cfg = CUSUM_SCALES[scale]
        surrogate, surrogate_tables = _run_segments(_seg_surrogate, delta, h, cfg['block_ms'], cfg['pass1_block_ms'])
        n_fa = sum((len(r) for r in surrogate))
        n_fa_gated = sum((_would_pass_gate(t) for t in surrogate_tables))
        injected, _ = _run_segments(_seg_injected, delta, h, cfg['block_ms'], cfg['pass1_block_ms'], masks=_seg_inject_mask)
        per_cell, fa_injected = _recall_table(injected)
        target = per_cell.loc[(per_cell['depth_pA'] >= CUSUM_TARGET_MIN_DEPTH_PA) & (per_cell['dwell_ms'] >= CUSUM_TARGET_MIN_DWELL_MS)]
        scale_cells = target.loc[cfg['cell_mask'](target)]
        summary = {'scale': scale, 'delta': delta, 'h': h, 'false_alarms_per_s_clean': n_fa / calibration_seconds, 'false_alarms_per_s_gated': n_fa_gated / calibration_seconds, 'false_alarms_per_s_gated_upper95': (n_fa_gated + 3.0 if n_fa_gated == 0 else n_fa_gated + 1.96 * math.sqrt(n_fa_gated) + 1.0) / calibration_seconds, 'false_alarms_per_s_injected': fa_injected / calibration_seconds, 'recall_scale_cells': float(scale_cells['recall'].mean()) if len(scale_cells) else np.nan, 'recall_target': float(target['recall'].mean()) if len(target) else np.nan, 'recall_all': float(per_cell['recall'].mean())}
        runs = {'surrogate': surrogate, 'surrogate_tables': surrogate_tables, 'injected': injected}
        return (summary, per_cell, runs)


# Original notebook cell 18
def stage_018():
    global _cond_mask, _cusum_mask, _cusum_overlaps_threshold, _dilated_mask, _dw_res, _dwells, _res_threshold_mask, _thr_mask, _threshold_overlaps_cusum, cusum_baseline_blocks, cusum_candidate_regions, cusum_candidate_regions_res, cusum_info, cusum_info_res, cusum_level_table, cusum_level_table_res, cusum_noise_blocks, cusum_scale_results, cusum_scale_results_res, run_cusum_scales
    check_cancelled(_acid_cancel)

    def _dilated_mask(regions, margin_ms: float) -> np.ndarray:
        mask = regions_to_mask(n_samples, regions)
        margin = int(round(margin_ms * fs / 1000.0))
        if margin > 0 and mask.any():
            from scipy.ndimage import maximum_filter1d
            mask = maximum_filter1d(mask.astype(np.uint8), size=2 * margin + 1).astype(bool)
        return mask

    def run_cusum_scales(trace: np.ndarray, label: str, exclude_mask: np.ndarray | None=None) -> tuple[dict, list, pd.DataFrame]:
        """Events of the OTHER polarity are passed as exclude_mask so that they cannot inflate the
    short-block noise of this polarity's pass-2 statistics (a 100 pA conductive spike inside a
    2 ms block would otherwise blind the resistive pulse scale around it)."""
        results = {}
        for _scale, (_d, _h) in cusum_selected.items():
            check_cancelled(_acid_cancel)
            _cfg = CUSUM_SCALES[_scale]
            _regions, _table, _info = cusum_detect(trace, fs=fs, delta=_d, h=_h, end_tol_sigma=CUSUM_END_TOL_SIGMA, block_ms=_cfg['block_ms'], level_est_ms=CUSUM_LEVEL_EST_MS, max_event_s=CUSUM_MAX_EVENT_S, min_event_ms=MIN_EVENT_MS, passes=2, pass1_block_ms=_cfg['pass1_block_ms'], mask_margin_ms=CUSUM_CAL_MARGIN_MS, pre_baseline_ms=CUSUM_PRE_BASELINE_MS, exclude_mask=exclude_mask)
            results[_scale] = {'regions': _regions, 'table': _table, 'info': _info, 'delta': _d, 'h': _h}
            print(f"[{label}] {_scale} scale (delta={_d}, h={_h}, block {_cfg['block_ms']:g} ms): {len(_regions):,} regions (pass 1: {_info.get('pass1_events', 'n/a')}); block noise {_info['sigma_blocks'].min():.2f}-{_info['sigma_blocks'].max():.2f} pA")
        regions, table = merge_region_sets(trace, [(r['regions'], r['table'], name) for name, r in results.items()], fs_merge=fs)
        return (results, regions, table)
    _res_threshold_mask = _dilated_mask(threshold_candidate_regions_res, CUSUM_CAL_MARGIN_MS) if DETECT_RESISTIVE else None
    cusum_scale_results, cusum_candidate_regions, cusum_level_table = run_cusum_scales(signal, 'conductive', exclude_mask=_res_threshold_mask)
    cusum_info = cusum_scale_results['level']['info']
    cusum_baseline_blocks = cusum_info['mu_blocks']
    cusum_noise_blocks = cusum_info['sigma_blocks']
    if DETECT_RESISTIVE:
        _cond_mask = _dilated_mask(cusum_candidate_regions + threshold_candidate_regions, CUSUM_CAL_MARGIN_MS)
        cusum_scale_results_res, cusum_candidate_regions_res, cusum_level_table_res = run_cusum_scales(-signal, 'resistive', exclude_mask=_cond_mask)
        cusum_info_res = cusum_scale_results_res['level']['info']
    else:
        cusum_scale_results_res, cusum_candidate_regions_res, cusum_level_table_res, cusum_info_res = ({}, [], cusum_level_table.iloc[0:0].copy(), None)
    _thr_mask = regions_to_mask(n_samples, threshold_candidate_regions)
    _cusum_overlaps_threshold = np.array([bool(_thr_mask[s:e + 1].any()) for s, e in cusum_candidate_regions])
    _cusum_mask = regions_to_mask(n_samples, cusum_candidate_regions)
    _threshold_overlaps_cusum = np.array([bool(_cusum_mask[s:e + 1].any()) for s, e in threshold_candidate_regions])
    # Empty like cusum_candidate_regions_res below (a NaN placeholder, not an empty
    # array) -- np.quantile/np.median raise IndexError on a truly empty array, and a
    # zero-candidate conductive pass is a real outcome, not a bug, on a quiet file.
    _dwells = np.array([dwell_ms(r, fs) for r in cusum_candidate_regions]) if cusum_candidate_regions else np.array([np.nan])
    print(f'Conductive union: {len(cusum_candidate_regions):,} CUSUM candidate regions ({len(cusum_candidate_regions) / duration_s:.3f}/s; expected raw false alarms ~ {cusum_false_alarm_rate * duration_s:.0f}, gate-surviving ~ {cusum_false_alarm_rate_gated * duration_s:.0f})')
    print('Regions by contributing scale:', cusum_level_table['cusum_scale'].value_counts().to_dict())
    print(f"Multi-level CUSUM regions (>= 2 levels): {int((cusum_level_table['cusum_n_levels'] >= 2).sum()):,}; truncated at CUSUM_MAX_EVENT_S: {int(cusum_level_table['cusum_truncated'].sum())}")
    print(f'CUSUM dwell quantiles (ms): 10% {np.nanquantile(_dwells, 0.1):.3f}, 50% {np.nanmedian(_dwells):.3f}, 90% {np.nanquantile(_dwells, 0.9):.3f}')
    print(f'CUSUM regions overlapping a threshold region: {_cusum_overlaps_threshold.mean():.3f}; threshold regions overlapping a CUSUM region: {_threshold_overlaps_cusum.mean():.3f}')
    if DETECT_RESISTIVE:
        _dw_res = np.array([dwell_ms(r, fs) for r in cusum_candidate_regions_res]) if cusum_candidate_regions_res else np.array([np.nan])
        print(f'Resistive union: {len(cusum_candidate_regions_res):,} candidate regions ({len(cusum_candidate_regions_res) / duration_s:.3f}/s); dwell median {np.nanmedian(_dw_res):.3f} ms')


def definitions_018():
    global _dilated_mask, run_cusum_scales

    def _dilated_mask(regions, margin_ms: float) -> np.ndarray:
        mask = regions_to_mask(n_samples, regions)
        margin = int(round(margin_ms * fs / 1000.0))
        if margin > 0 and mask.any():
            from scipy.ndimage import maximum_filter1d
            mask = maximum_filter1d(mask.astype(np.uint8), size=2 * margin + 1).astype(bool)
        return mask

    def run_cusum_scales(trace: np.ndarray, label: str, exclude_mask: np.ndarray | None=None) -> tuple[dict, list, pd.DataFrame]:
        """Events of the OTHER polarity are passed as exclude_mask so that they cannot inflate the
    short-block noise of this polarity's pass-2 statistics (a 100 pA conductive spike inside a
    2 ms block would otherwise blind the resistive pulse scale around it)."""
        results = {}
        for _scale, (_d, _h) in cusum_selected.items():
            check_cancelled(_acid_cancel)
            _cfg = CUSUM_SCALES[_scale]
            _regions, _table, _info = cusum_detect(trace, fs=fs, delta=_d, h=_h, end_tol_sigma=CUSUM_END_TOL_SIGMA, block_ms=_cfg['block_ms'], level_est_ms=CUSUM_LEVEL_EST_MS, max_event_s=CUSUM_MAX_EVENT_S, min_event_ms=MIN_EVENT_MS, passes=2, pass1_block_ms=_cfg['pass1_block_ms'], mask_margin_ms=CUSUM_CAL_MARGIN_MS, pre_baseline_ms=CUSUM_PRE_BASELINE_MS, exclude_mask=exclude_mask)
            results[_scale] = {'regions': _regions, 'table': _table, 'info': _info, 'delta': _d, 'h': _h}
            print(f"[{label}] {_scale} scale (delta={_d}, h={_h}, block {_cfg['block_ms']:g} ms): {len(_regions):,} regions (pass 1: {_info.get('pass1_events', 'n/a')}); block noise {_info['sigma_blocks'].min():.2f}-{_info['sigma_blocks'].max():.2f} pA")
        regions, table = merge_region_sets(trace, [(r['regions'], r['table'], name) for name, r in results.items()], fs_merge=fs)
        return (results, regions, table)


# Original notebook cell 20
def stage_020():
    global AMPLITUDE_COLUMNS, AMP_BASE, AMP_FRAC_COL, AMP_PA_COL, AMP_SIGMA_COL, SUSTAINED_GATE_PA, _cat, _floors, _gate_desc, _gb, _min_samples_gate, _name, _noise_pool, build_event_catalog, catalog_polarity, cusum_catalog, cusum_resistive_catalog, detector_catalogs, event_mask_by_polarity, local_rate_hz, quiet_context_indices, student_catalog, threshold_catalog, threshold_resistive_catalog, union_event_mask
    check_cancelled(_acid_cancel)
    union_event_mask = regions_to_mask(n_samples, student_candidate_regions + threshold_candidate_regions + cusum_candidate_regions + threshold_candidate_regions_res + cusum_candidate_regions_res)

    def quiet_context_indices(start: int, end: int, event_mask: np.ndarray, *, fs: float, context_ms: float, max_context_ms: float, min_samples: int) -> tuple[np.ndarray, float, bool]:
        """Expand symmetric pre/post context until enough non-event samples exist."""
        context = max(1, int(round(context_ms * fs / 1000.0)))
        max_context = max(context, int(round(max_context_ms * fs / 1000.0)))
        while True:
            check_cancelled(_acid_cancel)
            left = np.arange(max(0, start - context), start, dtype=np.int64)
            right = np.arange(end + 1, min(len(event_mask), end + 1 + context), dtype=np.int64)
            idx = np.concatenate((left, right))
            if idx.size:
                idx = idx[~event_mask[idx]]
            if idx.size >= min_samples or context >= max_context:
                return (idx, context / fs * 1000.0, idx.size >= min_samples)
            context = min(max_context, context * 2)

    def local_rate_hz(onsets: np.ndarray, center: int, fs: float, window_ms: float) -> float:
        radius = max(1, int(round(window_ms * fs / 1000.0)))
        lo = np.searchsorted(onsets, center - radius, side='left')
        hi = np.searchsorted(onsets, center + radius, side='right')
        duration = 2.0 * radius / fs
        return max(0, hi - lo - 1) / max(duration, 1e-12)

    def build_event_catalog(regions: list[tuple[int, int]], *, detector: str, signal: np.ndarray, raw: np.ndarray | None, probability: np.ndarray, event_mask: np.ndarray, fs: float, global_base: float | None=None, block_baselines: np.ndarray | None=None) -> pd.DataFrame:
        global_base = float(global_baseline) if global_base is None else float(global_base)
        block_baselines = cusum_baseline_blocks if block_baselines is None else block_baselines
        id_prefix = {'cusum': 'C', 'threshold': 'T', 'student': 'S', 'cusum_resistive': 'R', 'threshold_resistive': 'U'}.get(detector, detector[0].upper())
        regions = sorted(regions)
        onsets = np.asarray([s for s, _ in regions], dtype=np.int64)
        _block_centers = (np.arange(len(block_baselines)) + 0.5) * cusum_info['block_len']
        min_samples = max(1, int(math.ceil(MIN_EVENT_MS * fs / 1000.0)))
        rows = []
        for i, (start, end) in enumerate(tqdm(regions, desc=f'Features: {detector}')):
            check_cancelled(_acid_cancel)
            quiet_idx, context_used_ms, context_valid = quiet_context_indices(start, end, event_mask, fs=fs, context_ms=LOCAL_CONTEXT_MS, max_context_ms=MAX_LOCAL_CONTEXT_MS, min_samples=MIN_QUIET_SAMPLES)
            if quiet_idx.size:
                local_base = float(np.median(signal[quiet_idx]))
                local_noise = mad_std(signal[quiet_idx], local_base)
            else:
                local_base = global_base
                local_noise = global_noise
            if raw is not None and quiet_idx.size:
                local_i0 = abs(float(np.median(raw[quiet_idx])))
                i0_source = 'local_raw'
            elif np.isfinite(global_i0) and global_i0 > 0:
                local_i0 = float(global_i0)
                i0_source = 'global_fallback'
            else:
                local_i0 = np.nan
                i0_source = 'unavailable'
            body = signal[start:end + 1].astype(float)
            block_base = float(np.interp(0.5 * (start + end), _block_centers, block_baselines))
            _bases = {'local': local_base, 'global': global_base, 'block': block_base}
            if BASELINE_MODE not in _bases:
                raise ValueError("BASELINE_MODE must be 'local', 'global' or 'block'")
            amp_base = _bases[BASELINE_MODE]
            depth_trace = amp_base - body
            _per_mode = {}
            for _m, _b in _bases.items():
                check_cancelled(_acid_cancel)
                _dm = _b - body
                _pk = float(np.max(_dm))
                _hm = _dm >= 0.5 * _pk if _pk > 0 else np.zeros(len(_dm), dtype=bool)
                _per_mode[_m] = (_pk, float(np.median(_dm)), float(np.mean(_dm[_hm])) if _hm.any() else np.nan)
            positive_depth = np.clip(depth_trace, 0.0, None)
            n_event_samples = end - start + 1
            dwell = n_event_samples / fs * 1000.0
            peak_depth = float(np.max(depth_trace))
            sustained_depth = float(np.median(depth_trace))
            q80_depth = float(np.quantile(depth_trace, 0.8))
            area = float(np.sum(positive_depth) / fs * 1000.0)
            peak_sigma = peak_depth / max(local_noise, 1e-12)
            sustained_sigma = sustained_depth / max(local_noise, 1e-12)
            q80_sigma = q80_depth / max(local_noise, 1e-12)
            if np.isfinite(local_i0) and local_i0 > 0:
                peak_frac = peak_depth / local_i0
                sustained_frac = sustained_depth / local_i0
                q80_frac = q80_depth / local_i0
            else:
                peak_frac = sustained_frac = q80_frac = np.nan
            peak_to_sustained = peak_depth / max(sustained_depth, 1e-09)
            _half_mask = depth_trace >= 0.5 * peak_depth if peak_depth > 0 else np.zeros(len(depth_trace), dtype=bool)
            half_peak_mean = float(np.mean(depth_trace[_half_mask])) if _half_mask.any() else np.nan
            half_peak_ms = float(_half_mask.sum()) / fs * 1000.0
            _noise_max = float(norm.ppf((n_event_samples - 0.375) / (n_event_samples + 0.25))) if n_event_samples >= 2 else 0.0
            peak_corrected_depth = peak_depth - local_noise * _noise_max
            peak_corrected_sigma = peak_corrected_depth / max(local_noise, 1e-12)
            half_peak_sigma = half_peak_mean / max(local_noise, 1e-12) if np.isfinite(half_peak_mean) else np.nan
            if np.isfinite(local_i0) and local_i0 > 0:
                peak_corrected_frac = peak_corrected_depth / local_i0
                half_peak_frac = half_peak_mean / local_i0 if np.isfinite(half_peak_mean) else np.nan
            else:
                peak_corrected_frac = half_peak_frac = np.nan
            shape_factor = area / max(sustained_depth * dwell, 1e-09)
            roughness_sigma = float(np.median(np.abs(np.diff(body)))) / max(local_noise, 1e-12) if len(body) >= 2 else np.nan
            previous_gap = (start - regions[i - 1][1] - 1) / fs * 1000.0 if i > 0 else np.nan
            next_gap = (regions[i + 1][0] - end - 1) / fs * 1000.0 if i + 1 < len(regions) else np.nan
            if GATE_STATISTIC == 'peak':
                accepted_base = bool(context_valid and n_event_samples >= min_samples and (peak_depth > 0))
                accepted_sigma = bool(accepted_base and peak_depth >= local_noise * float(peak_gate_multiplier(dwell)))
            else:
                accepted_base = bool(context_valid and n_event_samples >= min_samples and (sustained_depth > 0))
                accepted_sigma = bool(accepted_base and np.isfinite(sustained_sigma) and (sustained_sigma >= SUSTAINED_GATE_SIGMA))
            accepted = accepted_sigma
            rows.append({**{f'{k}_{m}_pA': v for m, (pk, med, hm) in _per_mode.items() for k, v in (('peak_depth', pk), ('sustained_depth', med), ('half_peak_mean_depth', hm), ('peak_corrected_depth', pk - local_noise * _noise_max))}, 'global_baseline_pA': global_base, 'block_baseline_pA': block_base, 'amplitude_baseline_pA': amp_base, 'amplitude_baseline_mode': BASELINE_MODE, 'event_id': f'{id_prefix}{i:06d}', 'detector': detector, 'start_idx': int(start), 'end_idx': int(end), 'n_samples': int(n_event_samples), 'onset_s': start / fs, 'offset_s': (end + 1) / fs, 'dwell_ms': dwell, 'log_dwell': math.log10(max(dwell, 1e-09)), 'local_baseline_detr_pA': local_base, 'local_noise_mad_pA': local_noise, 'local_I0_pA': local_i0, 'I0_source': i0_source, 'peak_depth_pA': peak_depth, 'sustained_depth_pA': sustained_depth, 'q80_depth_pA': q80_depth, 'peak_depth_sigma': peak_sigma, 'sustained_depth_sigma': sustained_sigma, 'q80_depth_sigma': q80_sigma, 'peak_fracchange': peak_frac, 'sustained_fracchange': sustained_frac, 'q80_fracchange': q80_frac, 'area_pA_ms': area, 'log_area': math.log10(max(area, 1e-12)), 'peak_to_sustained': peak_to_sustained, 'half_peak_mean_depth_pA': half_peak_mean, 'half_peak_mean_depth_sigma': half_peak_sigma, 'half_peak_mean_fracchange': half_peak_frac, 'half_peak_ms': half_peak_ms, 'peak_corrected_depth_pA': peak_corrected_depth, 'peak_corrected_depth_sigma': peak_corrected_sigma, 'peak_corrected_fracchange': peak_corrected_frac, 'log_peak_to_sustained': math.log10(max(peak_to_sustained, 1e-09)), 'shape_factor': shape_factor, 'roughness_sigma': roughness_sigma, 'previous_gap_ms': previous_gap, 'next_gap_ms': next_gap, 'local_rate_hz': local_rate_hz(onsets, start, fs, LOCAL_RATE_WINDOW_MS), 'log_local_rate_hz': math.log10(1.0 + local_rate_hz(onsets, start, fs, LOCAL_RATE_WINDOW_MS)), 'student_probability_mean': float(np.mean(probability[start:end + 1])), 'student_probability_max': float(np.max(probability[start:end + 1])), 'quiet_samples': int(quiet_idx.size), 'context_used_ms': context_used_ms, 'context_valid': bool(context_valid), 'edge_event': bool(start == 0 or end == len(signal) - 1), 'accepted': accepted, 'accepted_sigma': accepted_sigma, 'accepted_base': accepted_base})
        catalog = pd.DataFrame(rows) if rows else pd.DataFrame({k: pd.Series(dtype=bool if k in ('accepted', 'accepted_sigma', 'accepted_base', 'context_valid', 'edge_event') else 'int64' if k in ('start_idx', 'end_idx', 'n_samples', 'time_block') else object if k in ('event_id', 'detector', 'I0_source', 'amplitude_baseline_mode') else float) for k in ['global_baseline_pA', 'block_baseline_pA', 'amplitude_baseline_pA', 'amplitude_baseline_mode', 'event_id', 'detector', 'start_idx', 'end_idx', 'n_samples', 'onset_s', 'offset_s', 'dwell_ms', 'log_dwell', 'local_baseline_detr_pA', 'local_noise_mad_pA', 'local_I0_pA', 'I0_source', 'peak_depth_pA', 'sustained_depth_pA', 'q80_depth_pA', 'peak_depth_sigma', 'sustained_depth_sigma', 'q80_depth_sigma', 'peak_fracchange', 'sustained_fracchange', 'q80_fracchange', 'area_pA_ms', 'log_area', 'peak_to_sustained', 'half_peak_mean_depth_pA', 'half_peak_mean_depth_sigma', 'half_peak_mean_fracchange', 'half_peak_ms', 'peak_corrected_depth_pA', 'peak_corrected_depth_sigma', 'peak_corrected_fracchange', 'log_peak_to_sustained', 'shape_factor', 'roughness_sigma', 'previous_gap_ms', 'next_gap_ms', 'local_rate_hz', 'log_local_rate_hz', 'student_probability_mean', 'student_probability_max', 'quiet_samples', 'context_used_ms', 'context_valid', 'edge_event', 'accepted', 'accepted_sigma', 'accepted_base', 'peak_depth_local_pA', 'sustained_depth_local_pA', 'half_peak_mean_depth_local_pA', 'peak_corrected_depth_local_pA', 'peak_depth_global_pA', 'sustained_depth_global_pA', 'half_peak_mean_depth_global_pA', 'peak_corrected_depth_global_pA', 'peak_depth_block_pA', 'sustained_depth_block_pA', 'half_peak_mean_depth_block_pA', 'peak_corrected_depth_block_pA', 'time_block', 'onset_fraction']})
        if len(catalog):
            block = np.floor(catalog['onset_s'] / duration_s * N_TIME_BLOCKS).astype(int)
            catalog['time_block'] = np.clip(block, 0, N_TIME_BLOCKS - 1)
            catalog['onset_fraction'] = catalog['onset_s'] / duration_s
        return catalog
    cusum_catalog = build_event_catalog(cusum_candidate_regions, detector='cusum', signal=signal, raw=raw, probability=student_probability, event_mask=union_event_mask, fs=fs)
    cusum_catalog = cusum_catalog.merge(cusum_level_table, on=['start_idx', 'end_idx'], how='left')
    threshold_catalog = build_event_catalog(threshold_candidate_regions, detector='threshold', signal=signal, raw=raw, probability=student_probability, event_mask=union_event_mask, fs=fs)
    if RUN_STUDENT and student_candidate_regions:
        student_catalog = build_event_catalog(student_candidate_regions, detector='student', signal=signal, raw=raw, probability=student_probability, event_mask=union_event_mask, fs=fs)
    else:
        student_catalog = threshold_catalog.iloc[0:0].copy()
    detector_catalogs = {'cusum': cusum_catalog, 'threshold': threshold_catalog, 'student': student_catalog}
    catalog_polarity = {'cusum': 1, 'threshold': 1, 'student': 1}
    if DETECT_RESISTIVE and cusum_candidate_regions_res:
        cusum_resistive_catalog = build_event_catalog(cusum_candidate_regions_res, detector='cusum_resistive', signal=-signal, raw=raw, probability=student_probability, event_mask=union_event_mask, fs=fs, global_base=-global_baseline, block_baselines=cusum_info_res['mu_blocks']).merge(cusum_level_table_res, on=['start_idx', 'end_idx'], how='left')
        threshold_resistive_catalog = build_event_catalog(threshold_candidate_regions_res, detector='threshold_resistive', signal=-signal, raw=raw, probability=student_probability, event_mask=union_event_mask, fs=fs, global_base=-global_baseline, block_baselines=cusum_info_res['mu_blocks'])
        detector_catalogs['cusum_resistive'] = cusum_resistive_catalog
        detector_catalogs['threshold_resistive'] = threshold_resistive_catalog
        catalog_polarity.update({'cusum_resistive': -1, 'threshold_resistive': -1})
    for _name, _cat in detector_catalogs.items():
        check_cancelled(_acid_cancel)
        if len(_cat):
            _cat['polarity'] = catalog_polarity[_name]
    event_mask_by_polarity = {1: regions_to_mask(n_samples, student_candidate_regions + threshold_candidate_regions + cusum_candidate_regions), -1: regions_to_mask(n_samples, threshold_candidate_regions_res + cusum_candidate_regions_res)}
    _min_samples_gate = max(1, int(math.ceil(MIN_EVENT_MS * fs / 1000.0)))
    _gb = GATE_BASELINE_MODE if GATE_BASELINE_MODE in ('local', 'global', 'block') else BASELINE_MODE
    _gate_desc = 'peak (minimum value, n_eff-corrected)' if GATE_STATISTIC == 'peak' else 'sustained'
    if SUSTAINED_GATE_MODE == 'timescale':
        SUSTAINED_GATE_PA = np.nan
        for _cat in detector_catalogs.values():
            check_cancelled(_acid_cancel)
            if len(_cat):
                _cat['acceptance_floor_pA'] = acceptance_floor_pA(_cat['dwell_ms'].to_numpy(float))
                _cat['accepted'] = _cat['accepted_base'] & passes_acceptance_gate(_cat[f'sustained_depth_{_gb}_pA'], _cat[f'half_peak_mean_depth_{_gb}_pA'], _cat['half_peak_ms'], _cat['dwell_ms'], peak_depth_pA=_cat[f'peak_depth_{_gb}_pA'], local_noise_pA=_cat['local_noise_mad_pA'])
        _floors = ', '.join((f'{d} ms -> {float(acceptance_floor_pA(d)):.1f} pA' for d in (0.1, 0.2, 0.5, 1.0, 5.0, 20.0)))
        if GATE_STATISTIC == 'peak':
            print(f'Acceptance gate (depths measured against the {GATE_BASELINE_MODE} baseline): the minimum value inside the event (peak depth) must clear the noise-only peak depth at the event duration (max of n_eff Gaussian draws fitted on the quiet trace, tail probability Phi(-{SUSTAINED_GATE_SIGMA:g})): ' + _floors)
        else:
            print(f'Acceptance gate (depths measured against the {GATE_BASELINE_MODE} baseline): duration-dependent floor, median depth >= {SUSTAINED_GATE_SIGMA:.1f} x quiet-trace noise at the event duration (plateau path) OR mean depth of the samples above half peak >= the same floor at the duration of that core (pulse path): ' + _floors)
    elif SUSTAINED_GATE_MODE == 'absolute':
        # Empty like the sigma/timescale branches -- pd.concat([]) raises ValueError
        # when every detector catalog is empty (a real, non-buggy outcome on a quiet
        # or heavily-excluded file), and must not crash the whole study.
        _noise_parts = [c['local_noise_mad_pA'] for c in detector_catalogs.values() if len(c)]
        _noise_pool = pd.concat(_noise_parts) if _noise_parts else pd.Series([], dtype=float)
        SUSTAINED_GATE_PA = float(SUSTAINED_GATE_SIGMA * np.quantile(_noise_pool, SUSTAINED_GATE_NOISE_QUANTILE)) if len(_noise_pool) else np.nan
        for _cat in detector_catalogs.values():
            check_cancelled(_acid_cancel)
            if len(_cat):
                if GATE_STATISTIC == 'peak':
                    _cat['acceptance_floor_pA'] = SUSTAINED_GATE_PA / SUSTAINED_GATE_SIGMA * peak_gate_multiplier(_cat['dwell_ms'].to_numpy(float))
                    _cat['accepted'] = _cat['accepted_base'] & (_cat[f'peak_depth_{_gb}_pA'] >= _cat['acceptance_floor_pA'])
                else:
                    # Was 'sustained_depth_pA' (always the BASELINE_MODE/reporting-baseline
                    # column) -- GATE_BASELINE_MODE must gate the SAME baseline the peak
                    # branch above already reads via the {_gb}-suffixed column.
                    _cat['accepted'] = _cat['accepted_base'] & (_cat[f'sustained_depth_{_gb}_pA'] >= SUSTAINED_GATE_PA)
        print(f'Acceptance gate (depths measured against the {GATE_BASELINE_MODE} baseline): absolute {_gate_desc} depth >= {SUSTAINED_GATE_PA:.2f} pA ({SUSTAINED_GATE_SIGMA:.1f} x the {SUSTAINED_GATE_NOISE_QUANTILE:.0%} quantile of local noise)')
    elif SUSTAINED_GATE_MODE == 'sigma':
        SUSTAINED_GATE_PA = np.nan
        for _cat in detector_catalogs.values():
            check_cancelled(_acid_cancel)
            if len(_cat):
                # Was 'accepted_sigma', a column build_event_catalog derives from
                # BASELINE_MODE alone -- it can never see GATE_BASELINE_MODE, so a
                # local/global split (the shipped default) silently gated on the wrong
                # baseline. Recomputed here from the {_gb}-suffixed columns through the
                # SAME passes_acceptance_gate the timescale branch above already uses.
                _noise = _cat['local_noise_mad_pA']
                _cat['accepted'] = _cat['accepted_base'] & passes_acceptance_gate(_cat[f'sustained_depth_{_gb}_pA'], _cat[f'half_peak_mean_depth_{_gb}_pA'], _cat['half_peak_ms'], _cat['dwell_ms'], _noise, peak_depth_pA=_cat[f'peak_depth_{_gb}_pA'], local_noise_pA=_noise)
        print(f'Acceptance gate (depths measured against the {GATE_BASELINE_MODE} baseline): per-event {_gate_desc} depth >= {SUSTAINED_GATE_SIGMA:.1f} x local noise')
    else:
        raise ValueError("SUSTAINED_GATE_MODE must be 'timescale', 'absolute' or 'sigma'")
    AMPLITUDE_COLUMNS = {'peak': 'peak', 'peak_corrected': 'peak_corrected', 'sustained': 'sustained', 'q80': 'q80', 'half_peak': 'half_peak_mean'}
    if AMPLITUDE_STATISTIC not in AMPLITUDE_COLUMNS:
        raise ValueError(f'AMPLITUDE_STATISTIC must be one of {list(AMPLITUDE_COLUMNS)}')
    AMP_BASE = AMPLITUDE_COLUMNS[AMPLITUDE_STATISTIC]
    AMP_PA_COL, AMP_SIGMA_COL, AMP_FRAC_COL = (f'{AMP_BASE}_depth_pA', f'{AMP_BASE}_depth_sigma', f'{AMP_BASE}_fracchange')
    print(f'Amplitude statistic: {AMPLITUDE_STATISTIC} -> {AMP_PA_COL} / {AMP_SIGMA_COL} / {AMP_FRAC_COL}')
    for _name, _cat in detector_catalogs.items():
        check_cancelled(_acid_cancel)
        if len(_cat):
            print(f"{_name.capitalize()} accepted: {int(_cat['accepted'].sum())} / {len(_cat)}")
        else:
            print(f'{_name.capitalize()}: not run')


def definitions_020():
    global build_event_catalog, local_rate_hz, quiet_context_indices

    def quiet_context_indices(start: int, end: int, event_mask: np.ndarray, *, fs: float, context_ms: float, max_context_ms: float, min_samples: int) -> tuple[np.ndarray, float, bool]:
        """Expand symmetric pre/post context until enough non-event samples exist."""
        context = max(1, int(round(context_ms * fs / 1000.0)))
        max_context = max(context, int(round(max_context_ms * fs / 1000.0)))
        while True:
            check_cancelled(_acid_cancel)
            left = np.arange(max(0, start - context), start, dtype=np.int64)
            right = np.arange(end + 1, min(len(event_mask), end + 1 + context), dtype=np.int64)
            idx = np.concatenate((left, right))
            if idx.size:
                idx = idx[~event_mask[idx]]
            if idx.size >= min_samples or context >= max_context:
                return (idx, context / fs * 1000.0, idx.size >= min_samples)
            context = min(max_context, context * 2)

    def local_rate_hz(onsets: np.ndarray, center: int, fs: float, window_ms: float) -> float:
        radius = max(1, int(round(window_ms * fs / 1000.0)))
        lo = np.searchsorted(onsets, center - radius, side='left')
        hi = np.searchsorted(onsets, center + radius, side='right')
        duration = 2.0 * radius / fs
        return max(0, hi - lo - 1) / max(duration, 1e-12)

    def build_event_catalog(regions: list[tuple[int, int]], *, detector: str, signal: np.ndarray, raw: np.ndarray | None, probability: np.ndarray, event_mask: np.ndarray, fs: float, global_base: float | None=None, block_baselines: np.ndarray | None=None) -> pd.DataFrame:
        global_base = float(global_baseline) if global_base is None else float(global_base)
        block_baselines = cusum_baseline_blocks if block_baselines is None else block_baselines
        id_prefix = {'cusum': 'C', 'threshold': 'T', 'student': 'S', 'cusum_resistive': 'R', 'threshold_resistive': 'U'}.get(detector, detector[0].upper())
        regions = sorted(regions)
        onsets = np.asarray([s for s, _ in regions], dtype=np.int64)
        _block_centers = (np.arange(len(block_baselines)) + 0.5) * cusum_info['block_len']
        min_samples = max(1, int(math.ceil(MIN_EVENT_MS * fs / 1000.0)))
        rows = []
        for i, (start, end) in enumerate(tqdm(regions, desc=f'Features: {detector}')):
            check_cancelled(_acid_cancel)
            quiet_idx, context_used_ms, context_valid = quiet_context_indices(start, end, event_mask, fs=fs, context_ms=LOCAL_CONTEXT_MS, max_context_ms=MAX_LOCAL_CONTEXT_MS, min_samples=MIN_QUIET_SAMPLES)
            if quiet_idx.size:
                local_base = float(np.median(signal[quiet_idx]))
                local_noise = mad_std(signal[quiet_idx], local_base)
            else:
                local_base = global_base
                local_noise = global_noise
            if raw is not None and quiet_idx.size:
                local_i0 = abs(float(np.median(raw[quiet_idx])))
                i0_source = 'local_raw'
            elif np.isfinite(global_i0) and global_i0 > 0:
                local_i0 = float(global_i0)
                i0_source = 'global_fallback'
            else:
                local_i0 = np.nan
                i0_source = 'unavailable'
            body = signal[start:end + 1].astype(float)
            block_base = float(np.interp(0.5 * (start + end), _block_centers, block_baselines))
            _bases = {'local': local_base, 'global': global_base, 'block': block_base}
            if BASELINE_MODE not in _bases:
                raise ValueError("BASELINE_MODE must be 'local', 'global' or 'block'")
            amp_base = _bases[BASELINE_MODE]
            depth_trace = amp_base - body
            _per_mode = {}
            for _m, _b in _bases.items():
                check_cancelled(_acid_cancel)
                _dm = _b - body
                _pk = float(np.max(_dm))
                _hm = _dm >= 0.5 * _pk if _pk > 0 else np.zeros(len(_dm), dtype=bool)
                _per_mode[_m] = (_pk, float(np.median(_dm)), float(np.mean(_dm[_hm])) if _hm.any() else np.nan)
            positive_depth = np.clip(depth_trace, 0.0, None)
            n_event_samples = end - start + 1
            dwell = n_event_samples / fs * 1000.0
            peak_depth = float(np.max(depth_trace))
            sustained_depth = float(np.median(depth_trace))
            q80_depth = float(np.quantile(depth_trace, 0.8))
            area = float(np.sum(positive_depth) / fs * 1000.0)
            peak_sigma = peak_depth / max(local_noise, 1e-12)
            sustained_sigma = sustained_depth / max(local_noise, 1e-12)
            q80_sigma = q80_depth / max(local_noise, 1e-12)
            if np.isfinite(local_i0) and local_i0 > 0:
                peak_frac = peak_depth / local_i0
                sustained_frac = sustained_depth / local_i0
                q80_frac = q80_depth / local_i0
            else:
                peak_frac = sustained_frac = q80_frac = np.nan
            peak_to_sustained = peak_depth / max(sustained_depth, 1e-09)
            _half_mask = depth_trace >= 0.5 * peak_depth if peak_depth > 0 else np.zeros(len(depth_trace), dtype=bool)
            half_peak_mean = float(np.mean(depth_trace[_half_mask])) if _half_mask.any() else np.nan
            half_peak_ms = float(_half_mask.sum()) / fs * 1000.0
            _noise_max = float(norm.ppf((n_event_samples - 0.375) / (n_event_samples + 0.25))) if n_event_samples >= 2 else 0.0
            peak_corrected_depth = peak_depth - local_noise * _noise_max
            peak_corrected_sigma = peak_corrected_depth / max(local_noise, 1e-12)
            half_peak_sigma = half_peak_mean / max(local_noise, 1e-12) if np.isfinite(half_peak_mean) else np.nan
            if np.isfinite(local_i0) and local_i0 > 0:
                peak_corrected_frac = peak_corrected_depth / local_i0
                half_peak_frac = half_peak_mean / local_i0 if np.isfinite(half_peak_mean) else np.nan
            else:
                peak_corrected_frac = half_peak_frac = np.nan
            shape_factor = area / max(sustained_depth * dwell, 1e-09)
            roughness_sigma = float(np.median(np.abs(np.diff(body)))) / max(local_noise, 1e-12) if len(body) >= 2 else np.nan
            previous_gap = (start - regions[i - 1][1] - 1) / fs * 1000.0 if i > 0 else np.nan
            next_gap = (regions[i + 1][0] - end - 1) / fs * 1000.0 if i + 1 < len(regions) else np.nan
            if GATE_STATISTIC == 'peak':
                accepted_base = bool(context_valid and n_event_samples >= min_samples and (peak_depth > 0))
                accepted_sigma = bool(accepted_base and peak_depth >= local_noise * float(peak_gate_multiplier(dwell)))
            else:
                accepted_base = bool(context_valid and n_event_samples >= min_samples and (sustained_depth > 0))
                accepted_sigma = bool(accepted_base and np.isfinite(sustained_sigma) and (sustained_sigma >= SUSTAINED_GATE_SIGMA))
            accepted = accepted_sigma
            rows.append({**{f'{k}_{m}_pA': v for m, (pk, med, hm) in _per_mode.items() for k, v in (('peak_depth', pk), ('sustained_depth', med), ('half_peak_mean_depth', hm), ('peak_corrected_depth', pk - local_noise * _noise_max))}, 'global_baseline_pA': global_base, 'block_baseline_pA': block_base, 'amplitude_baseline_pA': amp_base, 'amplitude_baseline_mode': BASELINE_MODE, 'event_id': f'{id_prefix}{i:06d}', 'detector': detector, 'start_idx': int(start), 'end_idx': int(end), 'n_samples': int(n_event_samples), 'onset_s': start / fs, 'offset_s': (end + 1) / fs, 'dwell_ms': dwell, 'log_dwell': math.log10(max(dwell, 1e-09)), 'local_baseline_detr_pA': local_base, 'local_noise_mad_pA': local_noise, 'local_I0_pA': local_i0, 'I0_source': i0_source, 'peak_depth_pA': peak_depth, 'sustained_depth_pA': sustained_depth, 'q80_depth_pA': q80_depth, 'peak_depth_sigma': peak_sigma, 'sustained_depth_sigma': sustained_sigma, 'q80_depth_sigma': q80_sigma, 'peak_fracchange': peak_frac, 'sustained_fracchange': sustained_frac, 'q80_fracchange': q80_frac, 'area_pA_ms': area, 'log_area': math.log10(max(area, 1e-12)), 'peak_to_sustained': peak_to_sustained, 'half_peak_mean_depth_pA': half_peak_mean, 'half_peak_mean_depth_sigma': half_peak_sigma, 'half_peak_mean_fracchange': half_peak_frac, 'half_peak_ms': half_peak_ms, 'peak_corrected_depth_pA': peak_corrected_depth, 'peak_corrected_depth_sigma': peak_corrected_sigma, 'peak_corrected_fracchange': peak_corrected_frac, 'log_peak_to_sustained': math.log10(max(peak_to_sustained, 1e-09)), 'shape_factor': shape_factor, 'roughness_sigma': roughness_sigma, 'previous_gap_ms': previous_gap, 'next_gap_ms': next_gap, 'local_rate_hz': local_rate_hz(onsets, start, fs, LOCAL_RATE_WINDOW_MS), 'log_local_rate_hz': math.log10(1.0 + local_rate_hz(onsets, start, fs, LOCAL_RATE_WINDOW_MS)), 'student_probability_mean': float(np.mean(probability[start:end + 1])), 'student_probability_max': float(np.max(probability[start:end + 1])), 'quiet_samples': int(quiet_idx.size), 'context_used_ms': context_used_ms, 'context_valid': bool(context_valid), 'edge_event': bool(start == 0 or end == len(signal) - 1), 'accepted': accepted, 'accepted_sigma': accepted_sigma, 'accepted_base': accepted_base})
        catalog = pd.DataFrame(rows) if rows else pd.DataFrame({k: pd.Series(dtype=bool if k in ('accepted', 'accepted_sigma', 'accepted_base', 'context_valid', 'edge_event') else 'int64' if k in ('start_idx', 'end_idx', 'n_samples', 'time_block') else object if k in ('event_id', 'detector', 'I0_source', 'amplitude_baseline_mode') else float) for k in ['global_baseline_pA', 'block_baseline_pA', 'amplitude_baseline_pA', 'amplitude_baseline_mode', 'event_id', 'detector', 'start_idx', 'end_idx', 'n_samples', 'onset_s', 'offset_s', 'dwell_ms', 'log_dwell', 'local_baseline_detr_pA', 'local_noise_mad_pA', 'local_I0_pA', 'I0_source', 'peak_depth_pA', 'sustained_depth_pA', 'q80_depth_pA', 'peak_depth_sigma', 'sustained_depth_sigma', 'q80_depth_sigma', 'peak_fracchange', 'sustained_fracchange', 'q80_fracchange', 'area_pA_ms', 'log_area', 'peak_to_sustained', 'half_peak_mean_depth_pA', 'half_peak_mean_depth_sigma', 'half_peak_mean_fracchange', 'half_peak_ms', 'peak_corrected_depth_pA', 'peak_corrected_depth_sigma', 'peak_corrected_fracchange', 'log_peak_to_sustained', 'shape_factor', 'roughness_sigma', 'previous_gap_ms', 'next_gap_ms', 'local_rate_hz', 'log_local_rate_hz', 'student_probability_mean', 'student_probability_max', 'quiet_samples', 'context_used_ms', 'context_valid', 'edge_event', 'accepted', 'accepted_sigma', 'accepted_base', 'peak_depth_local_pA', 'sustained_depth_local_pA', 'half_peak_mean_depth_local_pA', 'peak_corrected_depth_local_pA', 'peak_depth_global_pA', 'sustained_depth_global_pA', 'half_peak_mean_depth_global_pA', 'peak_corrected_depth_global_pA', 'peak_depth_block_pA', 'sustained_depth_block_pA', 'half_peak_mean_depth_block_pA', 'peak_corrected_depth_block_pA', 'time_block', 'onset_fraction']})
        if len(catalog):
            block = np.floor(catalog['onset_s'] / duration_s * N_TIME_BLOCKS).astype(int)
            catalog['time_block'] = np.clip(block, 0, N_TIME_BLOCKS - 1)
            catalog['onset_fraction'] = catalog['onset_s'] / duration_s
        return catalog


# Original notebook cell 22
def stage_022():
    global EXTENDED_FEATURE_COLUMNS, _gate_report, _kurtosis, _name, _primary_cat, _skew, _trace, add_extended_features, cusum_catalog, extended_event_features, hjorth, level_entropy, permutation_entropy, spectral_features, student_catalog, teager_kaiser, threshold_catalog
    check_cancelled(_acid_cancel)
    from scipy.stats import skew as _skew, kurtosis as _kurtosis

    def teager_kaiser(x: np.ndarray) -> np.ndarray:
        """Discrete Teager-Kaiser energy operator: psi[n] = x[n]^2 - x[n-1] x[n+1].

    Returns an array aligned with x; the two boundary samples are NaN.
    """
        x = np.asarray(x, float)
        psi = np.full(len(x), np.nan)
        if len(x) >= 3:
            psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
        return psi

    def permutation_entropy(x: np.ndarray, order: int=3, delay: int=1) -> float:
        """Bandt-Pompe permutation entropy, normalized to [0, 1] by log(order!)."""
        x = np.asarray(x, float)
        n = len(x) - (order - 1) * delay
        if n < 2 * math.factorial(order):
            return np.nan
        patterns = np.stack([x[i * delay:i * delay + n] for i in range(order)], axis=1)
        ranks = np.argsort(patterns, axis=1)
        codes = (ranks * order ** np.arange(order)).sum(axis=1)
        _, counts = np.unique(codes, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum() / math.log(math.factorial(order)))

    def spectral_features(z: np.ndarray, fs: float) -> tuple[float, float]:
        """Normalized spectral entropy (0-1) and spectral centroid (Hz) of the event body.

    DC is removed and a Hann window applied; the zero-frequency bin is excluded.
    """
        z = np.asarray(z, float)
        n = len(z)
        if n < 8:
            return (np.nan, np.nan)
        zc = (z - z.mean()) * np.hanning(n)
        power = np.abs(np.fft.rfft(zc)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power, freqs = (power[1:], freqs[1:])
        total = power.sum()
        if not np.isfinite(total) or total <= 0:
            return (np.nan, np.nan)
        p = power / total
        p_nz = p[p > 0]
        entropy = float(-(p_nz * np.log(p_nz)).sum() / math.log(len(p))) if len(p) > 1 else np.nan
        centroid = float((freqs * p).sum())
        return (entropy, centroid)

    def level_entropy(d: np.ndarray, n_bins: int=8) -> float:
        """Shannon entropy (0-1) of the within-event depth histogram after normalizing the
    event by its own robust depth. Fixed bins on a fixed normalized range so events of
    different amplitude are comparable. High values = the current spends comparable
    time at several levels (multi-level candidate)."""
        d = np.asarray(d, float)
        positive = d[d > 0]
        if len(positive) < 4:
            return np.nan
        scale = float(np.quantile(positive, 0.9))
        if not np.isfinite(scale) or scale <= 0:
            return np.nan
        u = np.clip(d / scale, -0.5, 1.5)
        counts, _ = np.histogram(u, bins=n_bins, range=(-0.5, 1.5))
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum() / math.log(n_bins))

    def hjorth(z: np.ndarray) -> tuple[float, float]:
        """Hjorth mobility and complexity (scale-free shape descriptors)."""
        z = np.asarray(z, float)
        if len(z) < 4:
            return (np.nan, np.nan)
        v0 = float(np.var(z))
        d1 = np.diff(z)
        v1 = float(np.var(d1))
        d2 = np.diff(d1)
        v2 = float(np.var(d2))
        if v0 <= 0 or v1 <= 0:
            return (np.nan, np.nan)
        mobility = math.sqrt(v1 / v0)
        complexity = math.sqrt(v2 / v1) / mobility if mobility > 0 else np.nan
        return (mobility, complexity)

    def extended_event_features(start: int, end: int, *, signal: np.ndarray, local_base: float, local_noise: float, local_i0: float, event_mask: np.ndarray, fs: float) -> dict:
        sigma = max(float(local_noise), 1e-12)
        n_total = len(signal)
        body = signal[start:end + 1].astype(float)
        d = local_base - body
        z = d / sigma
        n = len(d)
        out = {}
        out['power_sigma2'] = float(np.mean(z ** 2))
        out['log_power_sigma2'] = math.log10(max(out['power_sigma2'], 1e-12))
        rms = float(np.sqrt(np.mean(d ** 2)))
        if np.isfinite(local_i0) and local_i0 > 0:
            out['rms_fracchange'] = rms / local_i0
            out['log_rms_fracchange'] = math.log10(max(rms / local_i0, 1e-09))
        else:
            out['rms_fracchange'] = np.nan
            out['log_rms_fracchange'] = np.nan
        lo = max(0, start - 1)
        hi = min(n_total, end + 2)
        z_ext = (local_base - signal[lo:hi].astype(float)) / sigma
        psi_ext = teager_kaiser(z_ext)
        psi = psi_ext[start - lo:start - lo + n]
        psi_valid = psi[np.isfinite(psi)]
        if len(psi_valid):
            out['tkeo_mean_sigma2'] = float(np.mean(psi_valid))
            out['tkeo_max_sigma2'] = float(np.max(psi_valid))
            out['log_tkeo_mean_sigma2'] = math.log10(max(out['tkeo_mean_sigma2'], 1e-06))
        else:
            out['tkeo_mean_sigma2'] = out['tkeo_max_sigma2'] = out['log_tkeo_mean_sigma2'] = np.nan
        if n >= EXT_MOMENT_MIN_SAMPLES and len(psi_valid) >= 6:
            k = max(1, int(math.ceil(0.1 * n)))
            edge = np.concatenate((psi[:k], psi[-k:]))
            interior = psi[k:-k]
            edge, interior = (edge[np.isfinite(edge)], interior[np.isfinite(interior)])
            if len(edge) and len(interior):
                out['tkeo_edge_log_ratio'] = math.log10(max(float(np.mean(edge)), 0.0) + 1.0) - math.log10(max(float(np.mean(interior)), 0.0) + 1.0)
            else:
                out['tkeo_edge_log_ratio'] = np.nan
        else:
            out['tkeo_edge_log_ratio'] = np.nan
        out['max_slope_sigma'] = float(np.max(np.abs(np.diff(z_ext)))) if len(z_ext) >= 2 else np.nan
        out['log_max_slope_sigma'] = math.log10(max(out['max_slope_sigma'], 0.001)) if np.isfinite(out['max_slope_sigma']) else np.nan
        peak_idx = int(np.argmax(d))
        peak = float(d[peak_idx])
        out['peak_position_fraction'] = peak_idx / (n - 1) if n > 1 else 0.5
        out['fraction_above_half_peak'] = float(np.mean(d >= 0.5 * peak)) if peak > 0 else np.nan
        after = d[peak_idx + 1:]
        if peak > 0 and len(after):
            below = np.flatnonzero(after < peak / math.e)
            out['post_peak_efold_fraction'] = (below[0] + 1) / (n - peak_idx) if len(below) else 1.0
        else:
            out['post_peak_efold_fraction'] = np.nan
        ctx = max(1, int(round(EXT_CONTEXT_MS * fs / 1000.0)))
        pre_idx = np.arange(max(0, start - ctx), start)
        post_idx = np.arange(end + 1, min(n_total, end + 1 + ctx))
        pre_idx = pre_idx[~event_mask[pre_idx]] if len(pre_idx) else pre_idx
        post_idx = post_idx[~event_mask[post_idx]] if len(post_idx) else post_idx
        out['pre_enhancement_sigma'] = float(max(0.0, np.max(signal[pre_idx] - local_base)) / sigma) if len(pre_idx) else np.nan
        out['post_enhancement_sigma'] = float(max(0.0, np.max(signal[post_idx] - local_base)) / sigma) if len(post_idx) else np.nan
        if n >= EXT_MOMENT_MIN_SAMPLES and np.unique(d).size >= 4:
            out['depth_skew'] = float(_skew(d, bias=False))
            out['depth_kurtosis'] = float(_kurtosis(d, fisher=True, bias=False))
            mean_d = float(np.mean(d))
            out['depth_cv'] = float(min(np.std(d, ddof=1) / max(mean_d, 1e-09), 10.0)) if mean_d > 0 else np.nan
            centered = d - np.median(d)
            signs = np.sign(centered[centered != 0])
            out['zero_crossing_rate'] = float(np.mean(signs[1:] != signs[:-1])) if len(signs) >= 2 else np.nan
            out['hjorth_mobility'], out['hjorth_complexity'] = hjorth(z)
        else:
            for key in ('depth_skew', 'depth_kurtosis', 'depth_cv', 'zero_crossing_rate', 'hjorth_mobility', 'hjorth_complexity'):
                check_cancelled(_acid_cancel)
                out[key] = np.nan
        if n >= EXT_SPECTRAL_MIN_SAMPLES:
            out['spectral_entropy'], out['spectral_centroid_hz'] = spectral_features(z, fs)
            out['permutation_entropy'] = permutation_entropy(d, order=3, delay=1)
            out['level_entropy'] = level_entropy(d)
        else:
            out['spectral_entropy'] = out['spectral_centroid_hz'] = np.nan
            out['permutation_entropy'] = out['level_entropy'] = np.nan
        return out
    EXTENDED_FEATURE_COLUMNS = ['power_sigma2', 'log_power_sigma2', 'rms_fracchange', 'log_rms_fracchange', 'tkeo_mean_sigma2', 'tkeo_max_sigma2', 'log_tkeo_mean_sigma2', 'tkeo_edge_log_ratio', 'max_slope_sigma', 'log_max_slope_sigma', 'peak_position_fraction', 'fraction_above_half_peak', 'post_peak_efold_fraction', 'pre_enhancement_sigma', 'post_enhancement_sigma', 'depth_skew', 'depth_kurtosis', 'depth_cv', 'zero_crossing_rate', 'hjorth_mobility', 'hjorth_complexity', 'spectral_entropy', 'spectral_centroid_hz', 'permutation_entropy', 'level_entropy']

    def add_extended_features(catalog: pd.DataFrame, *, label: str, trace: np.ndarray, event_mask: np.ndarray) -> pd.DataFrame:
        rows = []
        for row in tqdm(catalog.itertuples(index=False), total=len(catalog), desc=f'Extended features: {label}'):
            check_cancelled(_acid_cancel)
            rows.append(extended_event_features(int(row.start_idx), int(row.end_idx), signal=trace, local_base=float(row.amplitude_baseline_pA), local_noise=float(row.local_noise_mad_pA), local_i0=float(row.local_I0_pA) if pd.notna(row.local_I0_pA) else np.nan, event_mask=event_mask, fs=fs))
        extra = pd.DataFrame(rows, index=catalog.index)[EXTENDED_FEATURE_COLUMNS]
        catalog = catalog.drop(columns=[c for c in EXTENDED_FEATURE_COLUMNS if c in catalog], errors='ignore')
        return pd.concat([catalog, extra], axis=1)
    for _name in list(detector_catalogs):
        check_cancelled(_acid_cancel)
        if len(detector_catalogs[_name]):
            _trace = signal if catalog_polarity.get(_name, 1) == 1 else -signal
            detector_catalogs[_name] = add_extended_features(detector_catalogs[_name], label=_name, trace=_trace, event_mask=event_mask_by_polarity[catalog_polarity.get(_name, 1)])
    cusum_catalog, threshold_catalog, student_catalog = (detector_catalogs['cusum'], detector_catalogs['threshold'], detector_catalogs['student'])
    _primary_cat = detector_catalogs[PRIMARY_DETECTOR]
    _gate_report = pd.DataFrame({'gate': ['none (n >= 3)', f'moment (n >= {EXT_MOMENT_MIN_SAMPLES})', f'spectral (n >= {EXT_SPECTRAL_MIN_SAMPLES})'], f'{PRIMARY_DETECTOR}_accepted_events': [int(_primary_cat['accepted'].sum()), int((_primary_cat['accepted'] & (_primary_cat['n_samples'] >= EXT_MOMENT_MIN_SAMPLES)).sum()), int((_primary_cat['accepted'] & (_primary_cat['n_samples'] >= EXT_SPECTRAL_MIN_SAMPLES)).sum())]})
    print(f'Added {len(EXTENDED_FEATURE_COLUMNS)} extended features to every detector catalog.')
    display(_gate_report)


def definitions_022():
    global _kurtosis, _skew, add_extended_features, extended_event_features, hjorth, level_entropy, permutation_entropy, spectral_features, teager_kaiser
    from scipy.stats import skew as _skew, kurtosis as _kurtosis

    def teager_kaiser(x: np.ndarray) -> np.ndarray:
        """Discrete Teager-Kaiser energy operator: psi[n] = x[n]^2 - x[n-1] x[n+1].

    Returns an array aligned with x; the two boundary samples are NaN.
    """
        x = np.asarray(x, float)
        psi = np.full(len(x), np.nan)
        if len(x) >= 3:
            psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
        return psi

    def permutation_entropy(x: np.ndarray, order: int=3, delay: int=1) -> float:
        """Bandt-Pompe permutation entropy, normalized to [0, 1] by log(order!)."""
        x = np.asarray(x, float)
        n = len(x) - (order - 1) * delay
        if n < 2 * math.factorial(order):
            return np.nan
        patterns = np.stack([x[i * delay:i * delay + n] for i in range(order)], axis=1)
        ranks = np.argsort(patterns, axis=1)
        codes = (ranks * order ** np.arange(order)).sum(axis=1)
        _, counts = np.unique(codes, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum() / math.log(math.factorial(order)))

    def spectral_features(z: np.ndarray, fs: float) -> tuple[float, float]:
        """Normalized spectral entropy (0-1) and spectral centroid (Hz) of the event body.

    DC is removed and a Hann window applied; the zero-frequency bin is excluded.
    """
        z = np.asarray(z, float)
        n = len(z)
        if n < 8:
            return (np.nan, np.nan)
        zc = (z - z.mean()) * np.hanning(n)
        power = np.abs(np.fft.rfft(zc)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power, freqs = (power[1:], freqs[1:])
        total = power.sum()
        if not np.isfinite(total) or total <= 0:
            return (np.nan, np.nan)
        p = power / total
        p_nz = p[p > 0]
        entropy = float(-(p_nz * np.log(p_nz)).sum() / math.log(len(p))) if len(p) > 1 else np.nan
        centroid = float((freqs * p).sum())
        return (entropy, centroid)

    def level_entropy(d: np.ndarray, n_bins: int=8) -> float:
        """Shannon entropy (0-1) of the within-event depth histogram after normalizing the
    event by its own robust depth. Fixed bins on a fixed normalized range so events of
    different amplitude are comparable. High values = the current spends comparable
    time at several levels (multi-level candidate)."""
        d = np.asarray(d, float)
        positive = d[d > 0]
        if len(positive) < 4:
            return np.nan
        scale = float(np.quantile(positive, 0.9))
        if not np.isfinite(scale) or scale <= 0:
            return np.nan
        u = np.clip(d / scale, -0.5, 1.5)
        counts, _ = np.histogram(u, bins=n_bins, range=(-0.5, 1.5))
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum() / math.log(n_bins))

    def hjorth(z: np.ndarray) -> tuple[float, float]:
        """Hjorth mobility and complexity (scale-free shape descriptors)."""
        z = np.asarray(z, float)
        if len(z) < 4:
            return (np.nan, np.nan)
        v0 = float(np.var(z))
        d1 = np.diff(z)
        v1 = float(np.var(d1))
        d2 = np.diff(d1)
        v2 = float(np.var(d2))
        if v0 <= 0 or v1 <= 0:
            return (np.nan, np.nan)
        mobility = math.sqrt(v1 / v0)
        complexity = math.sqrt(v2 / v1) / mobility if mobility > 0 else np.nan
        return (mobility, complexity)

    def extended_event_features(start: int, end: int, *, signal: np.ndarray, local_base: float, local_noise: float, local_i0: float, event_mask: np.ndarray, fs: float) -> dict:
        sigma = max(float(local_noise), 1e-12)
        n_total = len(signal)
        body = signal[start:end + 1].astype(float)
        d = local_base - body
        z = d / sigma
        n = len(d)
        out = {}
        out['power_sigma2'] = float(np.mean(z ** 2))
        out['log_power_sigma2'] = math.log10(max(out['power_sigma2'], 1e-12))
        rms = float(np.sqrt(np.mean(d ** 2)))
        if np.isfinite(local_i0) and local_i0 > 0:
            out['rms_fracchange'] = rms / local_i0
            out['log_rms_fracchange'] = math.log10(max(rms / local_i0, 1e-09))
        else:
            out['rms_fracchange'] = np.nan
            out['log_rms_fracchange'] = np.nan
        lo = max(0, start - 1)
        hi = min(n_total, end + 2)
        z_ext = (local_base - signal[lo:hi].astype(float)) / sigma
        psi_ext = teager_kaiser(z_ext)
        psi = psi_ext[start - lo:start - lo + n]
        psi_valid = psi[np.isfinite(psi)]
        if len(psi_valid):
            out['tkeo_mean_sigma2'] = float(np.mean(psi_valid))
            out['tkeo_max_sigma2'] = float(np.max(psi_valid))
            out['log_tkeo_mean_sigma2'] = math.log10(max(out['tkeo_mean_sigma2'], 1e-06))
        else:
            out['tkeo_mean_sigma2'] = out['tkeo_max_sigma2'] = out['log_tkeo_mean_sigma2'] = np.nan
        if n >= EXT_MOMENT_MIN_SAMPLES and len(psi_valid) >= 6:
            k = max(1, int(math.ceil(0.1 * n)))
            edge = np.concatenate((psi[:k], psi[-k:]))
            interior = psi[k:-k]
            edge, interior = (edge[np.isfinite(edge)], interior[np.isfinite(interior)])
            if len(edge) and len(interior):
                out['tkeo_edge_log_ratio'] = math.log10(max(float(np.mean(edge)), 0.0) + 1.0) - math.log10(max(float(np.mean(interior)), 0.0) + 1.0)
            else:
                out['tkeo_edge_log_ratio'] = np.nan
        else:
            out['tkeo_edge_log_ratio'] = np.nan
        out['max_slope_sigma'] = float(np.max(np.abs(np.diff(z_ext)))) if len(z_ext) >= 2 else np.nan
        out['log_max_slope_sigma'] = math.log10(max(out['max_slope_sigma'], 0.001)) if np.isfinite(out['max_slope_sigma']) else np.nan
        peak_idx = int(np.argmax(d))
        peak = float(d[peak_idx])
        out['peak_position_fraction'] = peak_idx / (n - 1) if n > 1 else 0.5
        out['fraction_above_half_peak'] = float(np.mean(d >= 0.5 * peak)) if peak > 0 else np.nan
        after = d[peak_idx + 1:]
        if peak > 0 and len(after):
            below = np.flatnonzero(after < peak / math.e)
            out['post_peak_efold_fraction'] = (below[0] + 1) / (n - peak_idx) if len(below) else 1.0
        else:
            out['post_peak_efold_fraction'] = np.nan
        ctx = max(1, int(round(EXT_CONTEXT_MS * fs / 1000.0)))
        pre_idx = np.arange(max(0, start - ctx), start)
        post_idx = np.arange(end + 1, min(n_total, end + 1 + ctx))
        pre_idx = pre_idx[~event_mask[pre_idx]] if len(pre_idx) else pre_idx
        post_idx = post_idx[~event_mask[post_idx]] if len(post_idx) else post_idx
        out['pre_enhancement_sigma'] = float(max(0.0, np.max(signal[pre_idx] - local_base)) / sigma) if len(pre_idx) else np.nan
        out['post_enhancement_sigma'] = float(max(0.0, np.max(signal[post_idx] - local_base)) / sigma) if len(post_idx) else np.nan
        if n >= EXT_MOMENT_MIN_SAMPLES and np.unique(d).size >= 4:
            out['depth_skew'] = float(_skew(d, bias=False))
            out['depth_kurtosis'] = float(_kurtosis(d, fisher=True, bias=False))
            mean_d = float(np.mean(d))
            out['depth_cv'] = float(min(np.std(d, ddof=1) / max(mean_d, 1e-09), 10.0)) if mean_d > 0 else np.nan
            centered = d - np.median(d)
            signs = np.sign(centered[centered != 0])
            out['zero_crossing_rate'] = float(np.mean(signs[1:] != signs[:-1])) if len(signs) >= 2 else np.nan
            out['hjorth_mobility'], out['hjorth_complexity'] = hjorth(z)
        else:
            for key in ('depth_skew', 'depth_kurtosis', 'depth_cv', 'zero_crossing_rate', 'hjorth_mobility', 'hjorth_complexity'):
                check_cancelled(_acid_cancel)
                out[key] = np.nan
        if n >= EXT_SPECTRAL_MIN_SAMPLES:
            out['spectral_entropy'], out['spectral_centroid_hz'] = spectral_features(z, fs)
            out['permutation_entropy'] = permutation_entropy(d, order=3, delay=1)
            out['level_entropy'] = level_entropy(d)
        else:
            out['spectral_entropy'] = out['spectral_centroid_hz'] = np.nan
            out['permutation_entropy'] = out['level_entropy'] = np.nan
        return out

    def add_extended_features(catalog: pd.DataFrame, *, label: str, trace: np.ndarray, event_mask: np.ndarray) -> pd.DataFrame:
        rows = []
        for row in tqdm(catalog.itertuples(index=False), total=len(catalog), desc=f'Extended features: {label}'):
            check_cancelled(_acid_cancel)
            rows.append(extended_event_features(int(row.start_idx), int(row.end_idx), signal=trace, local_base=float(row.amplitude_baseline_pA), local_noise=float(row.local_noise_mad_pA), local_i0=float(row.local_I0_pA) if pd.notna(row.local_I0_pA) else np.nan, event_mask=event_mask, fs=fs))
        extra = pd.DataFrame(rows, index=catalog.index)[EXTENDED_FEATURE_COLUMNS]
        catalog = catalog.drop(columns=[c for c in EXTENDED_FEATURE_COLUMNS if c in catalog], errors='ignore')
        return pd.concat([catalog, extra], axis=1)


# Original notebook cell 24
def stage_024():
    global ENTRY_SHOULDER_MIN_MS, EXIT_SPIKE_MIN_RATIO, SPIKE_CONTEXT_COLUMNS, TAIL_WINDOW_MS, _TAU_GRID_MS, _name, _pc, _pc_acc, _trace, add_spike_context_features, cusum_catalog, spike_context_features, student_catalog, threshold_catalog
    check_cancelled(_acid_cancel)
    ENTRY_SHOULDER_MIN_MS = 0.1
    EXIT_SPIKE_MIN_RATIO = 3.0
    TAIL_WINDOW_MS = 0.5
    _TAU_GRID_MS = np.geomspace(0.04, 3.0, 28)

    def spike_context_features(start: int, end: int, *, local_base: float, local_noise: float, peak_depth: float, pre_enh_sigma: float, post_enh_sigma: float, trace: np.ndarray) -> dict:
        sigma = max(float(local_noise), 1e-12)
        d = local_base - trace[start:end + 1].astype(float)
        n = len(d)
        p = int(np.argmax(d))
        peak = float(d[p]) if n else np.nan
        out = {}
        if not (np.isfinite(peak) and peak > 0):
            return {k: np.nan for k in SPIKE_CONTEXT_COLUMNS}
        tail_w = max(3, int(round(TAIL_WINDOW_MS * fs / 1000.0)))
        after = d[p + 1:]
        below_e = np.flatnonzero(after <= peak / math.e)
        out['recovery_efold_ms'] = (below_e[0] + 1) / fs * 1000.0 if len(below_e) else np.nan
        below_20 = np.flatnonzero(after <= 0.2 * peak)
        if len(below_20):
            f20 = p + 1 + int(below_20[0])
            out['fast_recovery_inside_region'] = True
        else:
            f20 = max(p + 1, n - tail_w)
            out['fast_recovery_inside_region'] = False
        fit_idx = np.arange(start + p + 1, min(len(trace), start + f20 + tail_w))
        if len(fit_idx):
            keep = ~union_event_mask[fit_idx] | (fit_idx <= end)
            fit_idx = fit_idx[keep]
        win_idx = fit_idx[fit_idx >= start + f20]
        out['tail_level_raw_pA'] = float(np.median(local_base - trace[win_idx].astype(float))) if len(win_idx) >= 3 else np.nan
        if out['fast_recovery_inside_region'] and len(fit_idx) >= 6:
            t_ms = (fit_idx - (start + p)) / fs * 1000.0
            y = local_base - trace[fit_idx].astype(float)
            best = None
            b_lo, b_hi = (-0.2 * peak, peak)
            for tau_ms in _TAU_GRID_MS:
                check_cancelled(_acid_cancel)
                e = np.exp(-t_ms / tau_ms)
                X = np.column_stack((e, np.ones_like(e)))
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                a_fit, b_fit = (float(coef[0]), float(coef[1]))
                if not b_lo <= b_fit <= b_hi:
                    b_fit = min(max(b_fit, b_lo), b_hi)
                    a_fit = float(np.dot(e, y - b_fit) / max(np.dot(e, e), 1e-12))
                resid = float(np.sum((y - (a_fit * e + b_fit)) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, tau_ms, a_fit, b_fit)
            _, tau_ms, amp_fit, shoulder = best
            out['decay_tau_ms'] = tau_ms
            out['tail_level_pA'] = shoulder
            out['tail_to_peak_ratio'] = shoulder / peak
            out['tail_level_sigma'] = shoulder / sigma
        else:
            out['decay_tau_ms'] = np.nan
            out['tail_level_pA'] = out['tail_level_raw_pA']
            out['tail_to_peak_ratio'] = out['tail_level_raw_pA'] / peak if np.isfinite(out['tail_level_raw_pA']) else np.nan
            out['tail_level_sigma'] = out['tail_level_raw_pA'] / sigma if np.isfinite(out['tail_level_raw_pA']) else np.nan
        out['tail_ms'] = (n - f20) / fs * 1000.0
        at_half = np.flatnonzero(d[:p + 1] >= 0.5 * peak)
        r50 = int(at_half[0]) if len(at_half) else p
        out['entry_ms'] = r50 / fs * 1000.0
        if r50 >= max(2, int(round(ENTRY_SHOULDER_MIN_MS * fs / 1000.0))):
            level = float(np.median(d[:r50]))
            out['entry_shoulder_pA'] = level
            out['entry_shoulder_ratio'] = max(level, 0.0) / peak
        else:
            out['entry_shoulder_pA'] = 0.0
            out['entry_shoulder_ratio'] = 0.0
        if n >= 10:
            k60 = max(2, int(0.6 * n))
            k70 = max(k60, int(0.7 * n))
            level = float(np.median(d[:k60]))
            term_max = float(np.max(d[k70:]))
            out['pre_exit_level_pA'] = level
            out['pre_exit_level_ms'] = k70 / fs * 1000.0
            out['terminal_spike_pA'] = term_max
            out['terminal_spike_ratio'] = term_max / max(level, 0.25 * sigma)
            out['exit_spike'] = bool(level >= 2.0 * sigma and out['terminal_spike_ratio'] >= EXIT_SPIKE_MIN_RATIO)
        else:
            out['pre_exit_level_pA'] = out['pre_exit_level_ms'] = out['terminal_spike_pA'] = out['terminal_spike_ratio'] = np.nan
            out['exit_spike'] = False
        out['pre_bump_pA'] = float(pre_enh_sigma) * sigma if np.isfinite(pre_enh_sigma) else np.nan
        out['post_overshoot_pA'] = float(post_enh_sigma) * sigma if np.isfinite(post_enh_sigma) else np.nan
        out['pre_bump_to_peak_ratio'] = out['pre_bump_pA'] / peak if np.isfinite(out['pre_bump_pA']) else np.nan
        out['post_overshoot_to_peak_ratio'] = out['post_overshoot_pA'] / peak if np.isfinite(out['post_overshoot_pA']) else np.nan
        return out
    SPIKE_CONTEXT_COLUMNS = ['recovery_efold_ms', 'decay_tau_ms', 'fast_recovery_inside_region', 'tail_level_raw_pA', 'tail_level_pA', 'tail_to_peak_ratio', 'tail_level_sigma', 'tail_ms', 'entry_ms', 'entry_shoulder_pA', 'entry_shoulder_ratio', 'pre_exit_level_pA', 'pre_exit_level_ms', 'terminal_spike_pA', 'terminal_spike_ratio', 'exit_spike', 'pre_bump_pA', 'post_overshoot_pA', 'pre_bump_to_peak_ratio', 'post_overshoot_to_peak_ratio']

    def add_spike_context_features(catalog: pd.DataFrame, *, label: str, trace: np.ndarray) -> pd.DataFrame:
        rows = []
        for row in tqdm(catalog.itertuples(index=False), total=len(catalog), desc=f'Spike context: {label}'):
            check_cancelled(_acid_cancel)
            rows.append(spike_context_features(int(row.start_idx), int(row.end_idx), local_base=float(row.amplitude_baseline_pA), local_noise=float(row.local_noise_mad_pA), peak_depth=float(row.peak_depth_pA), pre_enh_sigma=float(row.pre_enhancement_sigma), post_enh_sigma=float(row.post_enhancement_sigma), trace=trace))
        extra = pd.DataFrame(rows, index=catalog.index)[SPIKE_CONTEXT_COLUMNS]
        catalog = catalog.drop(columns=[c for c in SPIKE_CONTEXT_COLUMNS if c in catalog], errors='ignore')
        return pd.concat([catalog, extra], axis=1)
    for _name in list(detector_catalogs):
        check_cancelled(_acid_cancel)
        if len(detector_catalogs[_name]):
            _trace = signal if catalog_polarity.get(_name, 1) == 1 else -signal
            detector_catalogs[_name] = add_spike_context_features(detector_catalogs[_name], label=_name, trace=_trace)
    cusum_catalog, threshold_catalog, student_catalog = (detector_catalogs['cusum'], detector_catalogs['threshold'], detector_catalogs['student'])
    _pc = detector_catalogs[PRIMARY_DETECTOR]
    _pc_acc = _pc.loc[_pc['accepted']]
    print(f"Added {len(SPIKE_CONTEXT_COLUMNS)} spike-context features. Among accepted {PRIMARY_DETECTOR} events: fast recovery inside the region {100 * _pc_acc['fast_recovery_inside_region'].mean():.0f}%, exit spikes {int(_pc_acc['exit_spike'].sum())}, entry shoulders (ratio > 0.2) {int((_pc_acc['entry_shoulder_ratio'] > 0.2).sum())}, median tail/peak {_pc_acc['tail_to_peak_ratio'].median():.3f}, median pre-bump/peak {_pc_acc['pre_bump_to_peak_ratio'].median():.3f}")


def definitions_024():
    global add_spike_context_features, spike_context_features

    def spike_context_features(start: int, end: int, *, local_base: float, local_noise: float, peak_depth: float, pre_enh_sigma: float, post_enh_sigma: float, trace: np.ndarray) -> dict:
        sigma = max(float(local_noise), 1e-12)
        d = local_base - trace[start:end + 1].astype(float)
        n = len(d)
        p = int(np.argmax(d))
        peak = float(d[p]) if n else np.nan
        out = {}
        if not (np.isfinite(peak) and peak > 0):
            return {k: np.nan for k in SPIKE_CONTEXT_COLUMNS}
        tail_w = max(3, int(round(TAIL_WINDOW_MS * fs / 1000.0)))
        after = d[p + 1:]
        below_e = np.flatnonzero(after <= peak / math.e)
        out['recovery_efold_ms'] = (below_e[0] + 1) / fs * 1000.0 if len(below_e) else np.nan
        below_20 = np.flatnonzero(after <= 0.2 * peak)
        if len(below_20):
            f20 = p + 1 + int(below_20[0])
            out['fast_recovery_inside_region'] = True
        else:
            f20 = max(p + 1, n - tail_w)
            out['fast_recovery_inside_region'] = False
        fit_idx = np.arange(start + p + 1, min(len(trace), start + f20 + tail_w))
        if len(fit_idx):
            keep = ~union_event_mask[fit_idx] | (fit_idx <= end)
            fit_idx = fit_idx[keep]
        win_idx = fit_idx[fit_idx >= start + f20]
        out['tail_level_raw_pA'] = float(np.median(local_base - trace[win_idx].astype(float))) if len(win_idx) >= 3 else np.nan
        if out['fast_recovery_inside_region'] and len(fit_idx) >= 6:
            t_ms = (fit_idx - (start + p)) / fs * 1000.0
            y = local_base - trace[fit_idx].astype(float)
            best = None
            b_lo, b_hi = (-0.2 * peak, peak)
            for tau_ms in _TAU_GRID_MS:
                check_cancelled(_acid_cancel)
                e = np.exp(-t_ms / tau_ms)
                X = np.column_stack((e, np.ones_like(e)))
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                a_fit, b_fit = (float(coef[0]), float(coef[1]))
                if not b_lo <= b_fit <= b_hi:
                    b_fit = min(max(b_fit, b_lo), b_hi)
                    a_fit = float(np.dot(e, y - b_fit) / max(np.dot(e, e), 1e-12))
                resid = float(np.sum((y - (a_fit * e + b_fit)) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, tau_ms, a_fit, b_fit)
            _, tau_ms, amp_fit, shoulder = best
            out['decay_tau_ms'] = tau_ms
            out['tail_level_pA'] = shoulder
            out['tail_to_peak_ratio'] = shoulder / peak
            out['tail_level_sigma'] = shoulder / sigma
        else:
            out['decay_tau_ms'] = np.nan
            out['tail_level_pA'] = out['tail_level_raw_pA']
            out['tail_to_peak_ratio'] = out['tail_level_raw_pA'] / peak if np.isfinite(out['tail_level_raw_pA']) else np.nan
            out['tail_level_sigma'] = out['tail_level_raw_pA'] / sigma if np.isfinite(out['tail_level_raw_pA']) else np.nan
        out['tail_ms'] = (n - f20) / fs * 1000.0
        at_half = np.flatnonzero(d[:p + 1] >= 0.5 * peak)
        r50 = int(at_half[0]) if len(at_half) else p
        out['entry_ms'] = r50 / fs * 1000.0
        if r50 >= max(2, int(round(ENTRY_SHOULDER_MIN_MS * fs / 1000.0))):
            level = float(np.median(d[:r50]))
            out['entry_shoulder_pA'] = level
            out['entry_shoulder_ratio'] = max(level, 0.0) / peak
        else:
            out['entry_shoulder_pA'] = 0.0
            out['entry_shoulder_ratio'] = 0.0
        if n >= 10:
            k60 = max(2, int(0.6 * n))
            k70 = max(k60, int(0.7 * n))
            level = float(np.median(d[:k60]))
            term_max = float(np.max(d[k70:]))
            out['pre_exit_level_pA'] = level
            out['pre_exit_level_ms'] = k70 / fs * 1000.0
            out['terminal_spike_pA'] = term_max
            out['terminal_spike_ratio'] = term_max / max(level, 0.25 * sigma)
            out['exit_spike'] = bool(level >= 2.0 * sigma and out['terminal_spike_ratio'] >= EXIT_SPIKE_MIN_RATIO)
        else:
            out['pre_exit_level_pA'] = out['pre_exit_level_ms'] = out['terminal_spike_pA'] = out['terminal_spike_ratio'] = np.nan
            out['exit_spike'] = False
        out['pre_bump_pA'] = float(pre_enh_sigma) * sigma if np.isfinite(pre_enh_sigma) else np.nan
        out['post_overshoot_pA'] = float(post_enh_sigma) * sigma if np.isfinite(post_enh_sigma) else np.nan
        out['pre_bump_to_peak_ratio'] = out['pre_bump_pA'] / peak if np.isfinite(out['pre_bump_pA']) else np.nan
        out['post_overshoot_to_peak_ratio'] = out['post_overshoot_pA'] / peak if np.isfinite(out['post_overshoot_pA']) else np.nan
        return out

    def add_spike_context_features(catalog: pd.DataFrame, *, label: str, trace: np.ndarray) -> pd.DataFrame:
        rows = []
        for row in tqdm(catalog.itertuples(index=False), total=len(catalog), desc=f'Spike context: {label}'):
            check_cancelled(_acid_cancel)
            rows.append(spike_context_features(int(row.start_idx), int(row.end_idx), local_base=float(row.amplitude_baseline_pA), local_noise=float(row.local_noise_mad_pA), peak_depth=float(row.peak_depth_pA), pre_enh_sigma=float(row.pre_enhancement_sigma), post_enh_sigma=float(row.post_enhancement_sigma), trace=trace))
        extra = pd.DataFrame(rows, index=catalog.index)[SPIKE_CONTEXT_COLUMNS]
        catalog = catalog.drop(columns=[c for c in SPIKE_CONTEXT_COLUMNS if c in catalog], errors='ignore')
        return pd.concat([catalog, extra], axis=1)


# Original notebook cell 25
def stage_025():
    global PRIMARY_SET_NAME, _cond_onsets, _gap, _k, _nxt, _res, analysis_sets, analysis_signal, attach_matches, match_catalogs_one_to_one, match_table, primary_catalog, primary_cusum_info, primary_polarity, resistive_match_table, student_match_table, summary
    check_cancelled(_acid_cancel)

    def match_catalogs_one_to_one(left: pd.DataFrame, right: pd.DataFrame, *, fs: float, tolerance_ms: float) -> pd.DataFrame:
        """Greedy one-to-one matching ranked by overlap, IoU, then center distance."""
        left = left.sort_values('start_idx').reset_index(drop=True)
        right = right.sort_values('start_idx').reset_index(drop=True)
        tolerance = int(round(tolerance_ms * fs / 1000.0))
        r_start = right['start_idx'].to_numpy(int)
        r_end = right['end_idx'].to_numpy(int)
        candidates = []
        j0 = 0
        for li, row in left.iterrows():
            check_cancelled(_acid_cancel)
            ls, le = (int(row.start_idx), int(row.end_idx))
            while j0 < len(right) and r_end[j0] < ls - tolerance:
                check_cancelled(_acid_cancel)
                j0 += 1
            j = j0
            while j < len(right) and r_start[j] <= le + tolerance:
                check_cancelled(_acid_cancel)
                rs, re = (int(r_start[j]), int(r_end[j]))
                overlap = max(0, min(le, re) - max(ls, rs) + 1)
                union = le - ls + 1 + (re - rs + 1) - overlap
                iou = overlap / union if union > 0 else 0.0
                center_delta = abs((ls + le) / 2.0 - (rs + re) / 2.0)
                if overlap > 0 or center_delta <= tolerance:
                    candidates.append((overlap > 0, iou, overlap, -center_delta, li, j))
                j += 1
        candidates.sort(reverse=True)
        used_left = set()
        used_right = set()
        rows = []
        for has_overlap, iou, overlap, neg_center_delta, li, ri in candidates:
            check_cancelled(_acid_cancel)
            if li in used_left or ri in used_right:
                continue
            used_left.add(li)
            used_right.add(ri)
            rows.append({'student_event_id': left.loc[li, 'event_id'], 'threshold_event_id': right.loc[ri, 'event_id'], 'overlap_samples': int(overlap), 'iou': float(iou), 'center_delta_ms': float(-neg_center_delta / fs * 1000.0)})
        return pd.DataFrame(rows)

    def attach_matches(left: pd.DataFrame, right: pd.DataFrame, *, left_name: str, right_name: str) -> pd.DataFrame:
        """Match accepted events of two catalogs one-to-one and annotate both in place."""
        left_ok = left.loc[left['accepted']].copy()
        right_ok = right.loc[right['accepted']].copy()
        table = match_catalogs_one_to_one(left_ok, right_ok, fs=fs, tolerance_ms=MATCH_TOLERANCE_MS)
        table = table.rename(columns={'student_event_id': f'{left_name}_event_id', 'threshold_event_id': f'{right_name}_event_id'})
        left['detector_match'] = False
        left[f'{right_name}_event_id'] = pd.NA
        left['match_iou'] = np.nan
        left['match_center_delta_ms'] = np.nan
        right[f'{left_name}_match'] = False
        right[f'{left_name}_event_id'] = pd.NA
        if len(table):
            lm = table.set_index(f'{left_name}_event_id')
            rm = table.set_index(f'{right_name}_event_id')
            m = left['event_id'].isin(lm.index)
            left.loc[m, 'detector_match'] = True
            left.loc[m, f'{right_name}_event_id'] = left.loc[m, 'event_id'].map(lm[f'{right_name}_event_id'])
            left.loc[m, 'match_iou'] = left.loc[m, 'event_id'].map(lm['iou'])
            left.loc[m, 'match_center_delta_ms'] = left.loc[m, 'event_id'].map(lm['center_delta_ms'])
            m = right['event_id'].isin(rm.index)
            right.loc[m, f'{left_name}_match'] = True
            right.loc[m, f'{left_name}_event_id'] = right.loc[m, 'event_id'].map(rm[f'{left_name}_event_id'])
            assert table[f'{left_name}_event_id'].is_unique and table[f'{right_name}_event_id'].is_unique
        return table
    match_table = attach_matches(cusum_catalog, threshold_catalog, left_name='cusum', right_name='threshold')
    threshold_catalog['detector_match'] = threshold_catalog['cusum_match']
    analysis_sets = {'cusum': cusum_catalog.loc[cusum_catalog['accepted']].copy(), 'cusum_matched': cusum_catalog.loc[cusum_catalog['accepted'] & cusum_catalog['detector_match']].copy(), 'threshold': threshold_catalog.loc[threshold_catalog['accepted']].copy()}
    resistive_match_table = pd.DataFrame()
    if DETECT_RESISTIVE and 'cusum_resistive' in detector_catalogs and len(detector_catalogs['cusum_resistive']):
        resistive_match_table = attach_matches(detector_catalogs['cusum_resistive'], detector_catalogs['threshold_resistive'], left_name='cusum_resistive', right_name='threshold_resistive')
        analysis_sets['cusum_resistive'] = detector_catalogs['cusum_resistive'].loc[detector_catalogs['cusum_resistive']['accepted']].copy()
        analysis_sets['cusum_resistive_matched'] = detector_catalogs['cusum_resistive'].loc[detector_catalogs['cusum_resistive']['accepted'] & detector_catalogs['cusum_resistive']['detector_match']].copy()
        _cond_onsets = np.sort(cusum_catalog.loc[cusum_catalog['accepted'], 'start_idx'].to_numpy())
        _res = detector_catalogs['cusum_resistive']
        _nxt = np.searchsorted(_cond_onsets, _res['end_idx'].to_numpy())
        _gap = np.where(_nxt < len(_cond_onsets), (_cond_onsets[np.minimum(_nxt, len(_cond_onsets) - 1)] - _res['end_idx'].to_numpy()) / fs * 1000.0, np.inf) if len(_cond_onsets) else np.full(len(_res), np.inf)
        _res['gap_to_next_conductive_ms'] = _gap
        _res['precursor_of_conductive'] = _gap <= PRECURSOR_MAX_GAP_MS
        for _k in ('cusum_resistive', 'cusum_resistive_matched'):
            check_cancelled(_acid_cancel)
            analysis_sets[_k] = analysis_sets[_k].loc[~_res.loc[analysis_sets[_k].index, 'precursor_of_conductive']].copy()
        print(f"Resistive precursors (end within {PRECURSOR_MAX_GAP_MS} ms of a conductive onset): {int((_res['accepted'] & _res['precursor_of_conductive']).sum())} of {int(_res['accepted'].sum())} accepted resistive events; excluded from the resistive analysis sets")
    student_match_table = pd.DataFrame()
    if RUN_STUDENT and len(student_catalog):
        student_match_table = attach_matches(student_catalog, threshold_catalog, left_name='student', right_name='threshold')
        analysis_sets['student'] = student_catalog.loc[student_catalog['accepted']].copy()
        analysis_sets['matched'] = student_catalog.loc[student_catalog['accepted'] & student_catalog['detector_match']].copy()
    if PRIMARY_SET_NAME == 'cusum' and CUSUM_FALLBACK_TO_MATCHED and (not cusum_calibration_ok):
        print("!! CUSUM calibration did not meet the false-alarm budget: switching PRIMARY_SET_NAME from 'cusum' to 'cusum_matched' so the primary set only contains CUSUM events confirmed by the independent threshold detector. Set CUSUM_FALLBACK_TO_MATCHED = False to override.")
        PRIMARY_SET_NAME = 'cusum_matched'
    if PRIMARY_SET_NAME not in analysis_sets:
        raise ValueError(f'PRIMARY_SET_NAME must be one of {list(analysis_sets)}')
    primary_catalog = analysis_sets[PRIMARY_SET_NAME].copy().reset_index(drop=True)
    primary_polarity = int(primary_catalog['polarity'].iloc[0]) if 'polarity' in primary_catalog and len(primary_catalog) else 1
    analysis_signal = signal if primary_polarity == 1 else (-signal).astype(np.float32)
    primary_cusum_info = cusum_info if primary_polarity == 1 else cusum_info_res
    if len(primary_catalog) < MIN_ANALYSIS_EVENTS:
        print(f"!! Primary set '{PRIMARY_SET_NAME}' has only {len(primary_catalog)} events (< MIN_ANALYSIS_EVENTS = {MIN_ANALYSIS_EVENTS}): the incubation deliverables will be produced; the substructure sections will stop at their gate.")
    summary = pd.DataFrame([{'set': name, 'n_events': len(frame), 'rate_per_s': len(frame) / duration_s} for name, frame in analysis_sets.items()])
    print(f'One-to-one CUSUM/threshold matches: {len(match_table):,} conductive' + (f', {len(resistive_match_table):,} resistive' if DETECT_RESISTIVE else ''))
    print(f"Primary polarity: {('conductive (downward)' if primary_polarity == 1 else 'resistive (upward)')}")
    if RUN_STUDENT:
        print(f'One-to-one student/threshold matches: {len(student_match_table):,}')
    print(f'Primary set: {PRIMARY_SET_NAME} ({len(primary_catalog):,} events)')
    display(summary.round(3))


def definitions_025():
    global attach_matches, match_catalogs_one_to_one

    def match_catalogs_one_to_one(left: pd.DataFrame, right: pd.DataFrame, *, fs: float, tolerance_ms: float) -> pd.DataFrame:
        """Greedy one-to-one matching ranked by overlap, IoU, then center distance."""
        left = left.sort_values('start_idx').reset_index(drop=True)
        right = right.sort_values('start_idx').reset_index(drop=True)
        tolerance = int(round(tolerance_ms * fs / 1000.0))
        r_start = right['start_idx'].to_numpy(int)
        r_end = right['end_idx'].to_numpy(int)
        candidates = []
        j0 = 0
        for li, row in left.iterrows():
            check_cancelled(_acid_cancel)
            ls, le = (int(row.start_idx), int(row.end_idx))
            while j0 < len(right) and r_end[j0] < ls - tolerance:
                check_cancelled(_acid_cancel)
                j0 += 1
            j = j0
            while j < len(right) and r_start[j] <= le + tolerance:
                check_cancelled(_acid_cancel)
                rs, re = (int(r_start[j]), int(r_end[j]))
                overlap = max(0, min(le, re) - max(ls, rs) + 1)
                union = le - ls + 1 + (re - rs + 1) - overlap
                iou = overlap / union if union > 0 else 0.0
                center_delta = abs((ls + le) / 2.0 - (rs + re) / 2.0)
                if overlap > 0 or center_delta <= tolerance:
                    candidates.append((overlap > 0, iou, overlap, -center_delta, li, j))
                j += 1
        candidates.sort(reverse=True)
        used_left = set()
        used_right = set()
        rows = []
        for has_overlap, iou, overlap, neg_center_delta, li, ri in candidates:
            check_cancelled(_acid_cancel)
            if li in used_left or ri in used_right:
                continue
            used_left.add(li)
            used_right.add(ri)
            rows.append({'student_event_id': left.loc[li, 'event_id'], 'threshold_event_id': right.loc[ri, 'event_id'], 'overlap_samples': int(overlap), 'iou': float(iou), 'center_delta_ms': float(-neg_center_delta / fs * 1000.0)})
        return pd.DataFrame(rows)

    def attach_matches(left: pd.DataFrame, right: pd.DataFrame, *, left_name: str, right_name: str) -> pd.DataFrame:
        """Match accepted events of two catalogs one-to-one and annotate both in place."""
        left_ok = left.loc[left['accepted']].copy()
        right_ok = right.loc[right['accepted']].copy()
        table = match_catalogs_one_to_one(left_ok, right_ok, fs=fs, tolerance_ms=MATCH_TOLERANCE_MS)
        table = table.rename(columns={'student_event_id': f'{left_name}_event_id', 'threshold_event_id': f'{right_name}_event_id'})
        left['detector_match'] = False
        left[f'{right_name}_event_id'] = pd.NA
        left['match_iou'] = np.nan
        left['match_center_delta_ms'] = np.nan
        right[f'{left_name}_match'] = False
        right[f'{left_name}_event_id'] = pd.NA
        if len(table):
            lm = table.set_index(f'{left_name}_event_id')
            rm = table.set_index(f'{right_name}_event_id')
            m = left['event_id'].isin(lm.index)
            left.loc[m, 'detector_match'] = True
            left.loc[m, f'{right_name}_event_id'] = left.loc[m, 'event_id'].map(lm[f'{right_name}_event_id'])
            left.loc[m, 'match_iou'] = left.loc[m, 'event_id'].map(lm['iou'])
            left.loc[m, 'match_center_delta_ms'] = left.loc[m, 'event_id'].map(lm['center_delta_ms'])
            m = right['event_id'].isin(rm.index)
            right.loc[m, f'{left_name}_match'] = True
            right.loc[m, f'{left_name}_event_id'] = right.loc[m, 'event_id'].map(rm[f'{left_name}_event_id'])
            assert table[f'{left_name}_event_id'].is_unique and table[f'{right_name}_event_id'].is_unique
        return table


# Original notebook cell 26
def stage_026():
    check_cancelled(_acid_cancel)
    cusum_catalog.to_parquet(OUT_DIR / f'{TAG}_cusum_catalog.parquet', index=False)
    threshold_catalog.to_parquet(OUT_DIR / f'{TAG}_threshold_catalog.parquet', index=False)
    match_table.to_parquet(OUT_DIR / f'{TAG}_cusum_threshold_matches.parquet', index=False)
    if DETECT_RESISTIVE and 'cusum_resistive' in detector_catalogs:
        detector_catalogs['cusum_resistive'].to_parquet(OUT_DIR / f'{TAG}_cusum_resistive_catalog.parquet', index=False)
        detector_catalogs['threshold_resistive'].to_parquet(OUT_DIR / f'{TAG}_threshold_resistive_catalog.parquet', index=False)
        resistive_match_table.to_parquet(OUT_DIR / f'{TAG}_cusum_resistive_threshold_matches.parquet', index=False)
    if RUN_STUDENT and len(student_catalog):
        student_catalog.to_parquet(OUT_DIR / f'{TAG}_student_catalog.parquet', index=False)
        student_match_table.to_parquet(OUT_DIR / f'{TAG}_student_threshold_matches.parquet', index=False)
    primary_catalog.to_parquet(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_catalog_preclustering.parquet', index=False)
    print('✅ Saved immutable pre-clustering catalogs')


# Original notebook cell 28
def stage_028():
    global BASE_SCALAR_FEATURES, FEATURE_POOL, _RAW_DUPLICATES, _corr, _disc_mask, _qc, coupling_rows, coverage, dip, feature, feature_coupling, finite, flagged_features, g, k, multimodality_scan, n, n_unique, name, nuisance, nuisance_cols, p, pair, redundant_pairs, row, scan_rows, values, x, x_all, x_disc, x_val
    check_cancelled(_acid_cancel)
    BASE_SCALAR_FEATURES = ['log_dwell', 'peak_fracchange', 'peak_corrected_fracchange', 'tail_to_peak_ratio', 'recovery_efold_ms', 'decay_tau_ms', 'entry_shoulder_ratio', 'terminal_spike_ratio', 'pre_bump_to_peak_ratio', 'post_overshoot_to_peak_ratio', 'peak_depth_sigma', 'log_area', 'log_peak_to_sustained', 'shape_factor', 'roughness_sigma', 'student_probability_mean', 'log_local_rate_hz']
    _RAW_DUPLICATES = {'power_sigma2', 'rms_fracchange', 'tkeo_mean_sigma2', 'max_slope_sigma'}
    FEATURE_POOL = [c for c in BASE_SCALAR_FEATURES + EXTENDED_FEATURE_COLUMNS if c in primary_catalog and c not in _RAW_DUPLICATES]
    _qc = primary_catalog.copy()
    _qc['n_bin'] = pd.cut(_qc['n_samples'], bins=[0, EXT_MOMENT_MIN_SAMPLES - 1, EXT_SPECTRAL_MIN_SAMPLES - 1, np.inf], labels=[f'<{EXT_MOMENT_MIN_SAMPLES}', f'{EXT_MOMENT_MIN_SAMPLES}-{EXT_SPECTRAL_MIN_SAMPLES - 1}', f'>={EXT_SPECTRAL_MIN_SAMPLES}'])
    coverage = _qc.groupby('n_bin', observed=True)[EXTENDED_FEATURE_COLUMNS].apply(lambda g: g.notna().mean()).T
    coverage['overall'] = _qc[EXTENDED_FEATURE_COLUMNS].notna().mean()
    print(f"Fraction of primary events with a finite value, by event-length bin (events per bin: {_qc['n_bin'].value_counts(sort=False).to_dict()})")
    display(coverage.round(3))
    nuisance_cols = ['log_dwell', 'local_noise_mad_pA', AMP_FRAC_COL, 'onset_fraction']
    coupling_rows = []
    for feature in FEATURE_POOL:
        check_cancelled(_acid_cancel)
        row = {'feature': feature, 'n_valid': int(_qc[feature].notna().sum())}
        for nuisance in nuisance_cols:
            check_cancelled(_acid_cancel)
            if feature == nuisance:
                row[f'rho_{nuisance}'] = np.nan
                continue
            pair = _qc[[feature, nuisance]].replace([np.inf, -np.inf], np.nan).dropna()
            row[f'rho_{nuisance}'] = spearmanr(pair[feature], pair[nuisance])[0] if len(pair) >= 30 else np.nan
        coupling_rows.append(row)
    feature_coupling = pd.DataFrame(coupling_rows)
    feature_coupling.to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_feature_nuisance_coupling.csv', index=False)
    print('Spearman correlation of each feature with the nuisance axes (|rho| > 0.7 with log_dwell or local noise = treat as a proxy, not a new axis):')
    display(feature_coupling.round(3))
    _corr = _qc[FEATURE_POOL].replace([np.inf, -np.inf], np.nan).corr(method='spearman', min_periods=30)
    redundant_pairs = [{'feature_a': a, 'feature_b': b, 'spearman_rho': _corr.loc[a, b]} for i, a in enumerate(FEATURE_POOL) for b in FEATURE_POOL[i + 1:] if np.isfinite(_corr.loc[a, b]) and abs(_corr.loc[a, b]) >= 0.85]
    redundant_pairs = pd.DataFrame(redundant_pairs).sort_values('spearman_rho', key=np.abs, ascending=False) if redundant_pairs else pd.DataFrame(columns=['feature_a', 'feature_b', 'spearman_rho'])
    print(f'Feature pairs with |Spearman| >= 0.85 ({len(redundant_pairs)}): keep one per pair in any clustering space.')
    display(redundant_pairs.round(3))
    scan_rows = []
    _disc_mask = (_qc['time_block'] % 2 == 0).to_numpy()
    for feature in FEATURE_POOL:
        check_cancelled(_acid_cancel)
        values = _qc[feature].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        finite = np.isfinite(values)
        x_disc = values[finite & _disc_mask]
        x_val = values[finite & ~_disc_mask]
        n_unique = int(np.unique(values[finite]).size)
        row = {'feature': feature, 'n_discovery': len(x_disc), 'n_validation': len(x_val), 'n_unique': n_unique}
        for name, x in (('discovery', x_disc), ('validation', x_val)):
            check_cancelled(_acid_cancel)
            if HAVE_DIPTEST and len(x) >= 30:
                dip, p = diptest(np.sort(x))
                row[f'dip_{name}'] = float(dip)
                row[f'dip_p_{name}'] = float(p)
            else:
                row[f'dip_{name}'] = np.nan
                row[f'dip_p_{name}'] = np.nan
        x_all = values[finite]
        if len(x_all) >= 30:
            n = len(x_all)
            g = float(_skew(x_all, bias=False))
            k = float(_kurtosis(x_all, fisher=True, bias=False))
            row['bimodality_coefficient'] = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        else:
            row['bimodality_coefficient'] = np.nan
        row['flag'] = bool(np.isfinite(row.get('dip_p_discovery', np.nan)) and np.isfinite(row.get('dip_p_validation', np.nan)) and (row['dip_p_discovery'] < 0.05) and (row['dip_p_validation'] < 0.05) and (n_unique >= 30))
        scan_rows.append(row)
    multimodality_scan = pd.DataFrame(scan_rows).sort_values(['flag', 'dip_p_validation'], ascending=[False, True])
    multimodality_scan.to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_univariate_multimodality_scan.csv', index=False)
    print('Per-feature dip test on discovery (even) and validation (odd) blocks. flag = p < 0.05 in both halves and >= 30 unique values. BC > 0.555 is the usual bimodality-coefficient threshold (skew-sensitive; use as a hint only).')
    display(multimodality_scan.round(4))
    flagged_features = multimodality_scan.loc[multimodality_scan['flag'], 'feature'].tolist()
    print('Features multimodal in both halves:', flagged_features if flagged_features else 'none')


# Original notebook cell 30
def stage_030():
    global SHOULDER_MIN_RATIO, _, _dip_both_halves, _f, alone, ax, axes, binomtest, bins, ex, exit_spikes, fig, frac_tail_significant, hi, k, ks_2samp, ks_p, ks_stat, lim, lo, mannwhitneyu, mw_p, n_pre_larger, p_disc, p_val, pad, paired, picks, pl, pl_pA, plateau_like, pre_ratio_by_tertile, ratio, ratio_by_tertile, ratio_mad, rho_pre_peak, rho_tail_peak, rho_tau_peak, row, s, sign_p, sp, sp_pA, spike_alone, spike_context_tests, spike_shaped, spike_shoulder, t, tau, tau_by_tertile, tertiles, y
    check_cancelled(_acid_cancel)
    from scipy.stats import ks_2samp, mannwhitneyu, binomtest
    _f = primary_catalog.copy()
    _f['half'] = np.where(_f['time_block'] % 2 == 0, 'discovery', 'validation')
    spike_shaped = _f['fast_recovery_inside_region'].fillna(False).astype(bool) & (_f['dwell_ms'] <= 5.0)
    SHOULDER_MIN_RATIO = 0.05
    spike_alone = spike_shaped & (_f['tail_to_peak_ratio'] < SHOULDER_MIN_RATIO) & (_f['entry_shoulder_ratio'] < 0.2) & (_f['dwell_ms'] <= 1.5)
    spike_shoulder = spike_shaped & (_f['tail_to_peak_ratio'] >= SHOULDER_MIN_RATIO)
    plateau_like = ~_f['fast_recovery_inside_region'].fillna(False).astype(bool) & (_f['dwell_ms'] >= 1.0)
    exit_spikes = _f['exit_spike'].fillna(False).astype(bool)
    print(f'Primary events {len(_f)}: spike-shaped {int(spike_shaped.sum())} (spike-alone {int(spike_alone.sum())}, with shoulder {int(spike_shoulder.sum())}), plateau-like {int(plateau_like.sum())}, exit spikes {int(exit_spikes.sum())}')

    def _dip_both_halves(values: pd.Series, mask) -> tuple[float, float]:
        out = []
        for half in ('discovery', 'validation'):
            check_cancelled(_acid_cancel)
            x = values.loc[mask & (_f['half'] == half)].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            out.append(float(diptest(np.sort(x))[1]) if HAVE_DIPTEST and len(x) >= 30 else np.nan)
        return (out[0], out[1])
    spike_context_tests = {}
    s = _f.loc[spike_shaped]
    ratio = s['tail_to_peak_ratio']
    p_disc, p_val = _dip_both_halves(_f['tail_to_peak_ratio'], spike_shaped)
    rho_tail_peak = spearmanr(s['tail_level_pA'], s['peak_depth_pA'])[0] if len(s) >= 30 else np.nan
    tertiles = pd.qcut(s['peak_depth_pA'], 3, labels=['small', 'mid', 'large']) if len(s) >= 30 else None
    ratio_by_tertile = ratio.groupby(tertiles, observed=True).median() if tertiles is not None else pd.Series(dtype=float)
    ratio_mad = 1.4826 * float((ratio - ratio.median()).abs().median()) if len(ratio) else np.nan
    frac_tail_significant = float((s['tail_level_sigma'] >= 2.0).mean()) if len(s) else np.nan
    tau = s['decay_tau_ms'].dropna()
    tau_by_tertile = tau.groupby(tertiles.loc[tau.index], observed=True).median() if tertiles is not None and len(tau) >= 30 else pd.Series(dtype=float)
    rho_tau_peak = spearmanr(s.loc[tau.index, 'peak_depth_pA'], tau)[0] if len(tau) >= 30 else np.nan
    spike_context_tests['shoulder'] = {'n_spike_shaped': int(len(s)), 'tail_to_peak_median': float(ratio.median()) if len(s) else np.nan, 'tail_to_peak_robust_sd': ratio_mad, 'dip_p_discovery': p_disc, 'dip_p_validation': p_val, 'spearman_tail_vs_peak': float(rho_tail_peak) if np.isfinite(rho_tail_peak) else np.nan, 'ratio_median_by_peak_tertile': {str(k): float(v) for k, v in ratio_by_tertile.items()}, 'fraction_tail_above_2sigma': frac_tail_significant, 'fraction_with_shoulder_ratio_ge_0.05': float((ratio >= SHOULDER_MIN_RATIO).mean()) if len(s) else np.nan, 'decay_tau_median_ms': float(tau.median()) if len(tau) else np.nan, 'decay_tau_robust_sd_ms': 1.4826 * float((tau - tau.median()).abs().median()) if len(tau) else np.nan, 'decay_tau_by_peak_tertile_ms': {str(k): float(v) for k, v in tau_by_tertile.items()}, 'spearman_tau_vs_peak': float(rho_tau_peak) if np.isfinite(rho_tau_peak) else np.nan}
    print('\nTEST 1 - shoulder after the fast recovery (spike-shaped events)')
    print(f'  tail/peak median {ratio.median():.3f}, robust SD {ratio_mad:.3f}; {100 * frac_tail_significant:.0f}% of tails are >= 2 sigma')
    print(f'  dip test on tail/peak: p = {p_disc:.4f} (discovery), {p_val:.4f} (validation)  -> bimodal in both halves: {bool(p_disc < 0.05 and p_val < 0.05)}')
    print(f'  Spearman(tail level, peak) = {rho_tail_peak:+.3f}; median tail/peak by peak tertile: ' + ', '.join((f'{k} {v:.3f}' for k, v in ratio_by_tertile.items())))
    print(f'  decay time constant: median {tau.median():.3f} ms, robust SD {1.4826 * float((tau - tau.median()).abs().median()):.3f} ms; by peak tertile ' + ', '.join((f'{k} {v:.3f}' for k, v in tau_by_tertile.items())) + f'; Spearman vs peak {rho_tau_peak:+.3f}')
    print('  Reading: a fixed slow response component gives a unimodal ratio with a small robust SD, constant across tertiles, and a high tail-vs-peak correlation; a physical shoulder gives a bimodal ratio and/or tails that do not scale with the peak.')
    ex = _f.loc[exit_spikes]
    alone = _f.loc[spike_alone]
    if len(ex) >= 5 and len(alone) >= 20:
        ks_stat, ks_p = ks_2samp(ex['terminal_spike_pA'].dropna(), alone['peak_depth_pA'].dropna())
    else:
        ks_stat, ks_p = (np.nan, np.nan)
    spike_context_tests['exit_spike'] = {'n_exit_spikes': int(len(ex)), 'n_spike_alone': int(len(alone)), 'exit_spike_amplitude_median_pA': float(ex['terminal_spike_pA'].median()) if len(ex) else np.nan, 'spike_alone_amplitude_median_pA': float(alone['peak_depth_pA'].median()) if len(alone) else np.nan, 'ks_statistic': float(ks_stat) if np.isfinite(ks_stat) else np.nan, 'ks_p': float(ks_p) if np.isfinite(ks_p) else np.nan, 'pre_exit_level_median_pA': float(ex['pre_exit_level_pA'].median()) if len(ex) else np.nan, 'pre_exit_level_median_ms': float(ex['pre_exit_level_ms'].median()) if len(ex) else np.nan, 'exit_spikes_per_block': ex['time_block'].value_counts().sort_index().to_dict()}
    print('\nTEST 2 - shallow level terminated by a deep spike')
    if len(ex):
        print(f"  {len(ex)} exit-spike events (blocks: {spike_context_tests['exit_spike']['exit_spikes_per_block']}); preceding level median {ex['pre_exit_level_pA'].median():.1f} pA for {ex['pre_exit_level_ms'].median():.2f} ms; terminal spike median {ex['terminal_spike_pA'].median():.1f} pA vs stand-alone spike median {alone['peak_depth_pA'].median():.1f} pA (n = {len(alone)}); KS p = {ks_p:.3f}")
        print('  Reading: KS p well above 0.05 means the exit spikes are drawn from the same amplitude distribution as stand-alone spikes.')
    else:
        print('  no exit-spike events in the primary set')
    sp = _f.loc[spike_shaped, 'pre_bump_to_peak_ratio'].dropna()
    pl = _f.loc[plateau_like, 'pre_bump_to_peak_ratio'].dropna()
    sp_pA = _f.loc[spike_shaped, 'pre_bump_pA'].dropna()
    pl_pA = _f.loc[plateau_like, 'pre_bump_pA'].dropna()
    mw_p = mannwhitneyu(sp_pA, pl_pA, alternative='two-sided')[1] if len(sp_pA) >= 10 and len(pl_pA) >= 10 else np.nan
    paired = _f.loc[spike_shaped, ['pre_bump_pA', 'post_overshoot_pA']].dropna()
    n_pre_larger = int((paired['pre_bump_pA'] > paired['post_overshoot_pA']).sum())
    sign_p = binomtest(n_pre_larger, len(paired), 0.5).pvalue if len(paired) >= 10 else np.nan
    rho_pre_peak = spearmanr(_f.loc[spike_shaped, 'pre_bump_pA'], _f.loc[spike_shaped, 'peak_depth_pA'], nan_policy='omit')[0] if spike_shaped.sum() >= 30 else np.nan
    pre_ratio_by_tertile = _f.loc[spike_shaped, 'pre_bump_to_peak_ratio'].groupby(tertiles, observed=True).median() if tertiles is not None else pd.Series(dtype=float)
    spike_context_tests['pre_bump'] = {'spike_family_pre_bump_median_pA': float(sp_pA.median()) if len(sp_pA) else np.nan, 'plateau_family_pre_bump_median_pA': float(pl_pA.median()) if len(pl_pA) else np.nan, 'mannwhitney_p_spike_vs_plateau': float(mw_p) if np.isfinite(mw_p) else np.nan, 'spike_family_pre_bump_to_peak_median': float(sp.median()) if len(sp) else np.nan, 'pre_bump_to_peak_by_peak_tertile': {str(k): float(v) for k, v in pre_ratio_by_tertile.items()}, 'spearman_pre_bump_vs_peak': float(rho_pre_peak) if np.isfinite(rho_pre_peak) else np.nan, 'post_overshoot_median_pA': float(paired['post_overshoot_pA'].median()) if len(paired) else np.nan, 'fraction_pre_larger_than_post': n_pre_larger / len(paired) if len(paired) else np.nan, 'sign_test_p': float(sign_p) if np.isfinite(sign_p) else np.nan}
    print('\nTEST 3 - pre-bump')
    print(f'  spike family: pre-bump median {sp_pA.median():.1f} pA ({sp.median():.3f} of the peak; by peak tertile ' + ', '.join((f'{k} {v:.3f}' for k, v in pre_ratio_by_tertile.items())) + f'; Spearman vs peak {rho_pre_peak:+.3f})')
    print(f"  plateau family: pre-bump median {(pl_pA.median() if len(pl_pA) else float('nan')):.1f} pA; Mann-Whitney p = {mw_p:.2e}")
    print(f"  pre-bump vs post-event overshoot (spike family, n = {len(paired)}): pre larger in {100 * spike_context_tests['pre_bump']['fraction_pre_larger_than_post']:.0f}% (post median {paired['post_overshoot_pA'].median():.1f} pA); sign test p = {sign_p:.2e}")
    print('  Reading: a zero-phase filter gives pre and post excursions of similar size in every class and a fixed pre/peak ratio; a class-dependent, asymmetric pre-bump is a precursor in the current itself.')
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    ax.hist(ratio.clip(-0.2, 1.2), bins=48, histtype='stepfilled', alpha=0.6)
    ax.set_xlabel('tail / peak (spike-shaped events)')
    ax.set_ylabel('Events')
    ax.set_title('Shoulder after the fast recovery')
    ax = axes[0, 1]
    ax.scatter(s['peak_depth_pA'], s['tail_level_pA'], s=9, alpha=0.5)
    ax.set_xlabel('Peak depth (pA)')
    ax.set_ylabel('Tail level (pA)')
    ax.set_title(f'Tail vs peak (Spearman {rho_tail_peak:+.2f})')
    ax = axes[0, 2]
    if len(ex):
        bins = np.geomspace(max(1.0, min(alone['peak_depth_pA'].min(), ex['terminal_spike_pA'].min())), max(alone['peak_depth_pA'].max(), ex['terminal_spike_pA'].max()) * 1.05, 30)
        ax.hist(alone['peak_depth_pA'], bins=bins, histtype='step', density=True, linewidth=1.6, label=f'stand-alone spikes (n={len(alone)})')
        ax.hist(ex['terminal_spike_pA'], bins=bins, histtype='step', density=True, linewidth=1.6, label=f'exit spikes (n={len(ex)})')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
    ax.set_xlabel('Spike amplitude (pA)')
    ax.set_ylabel('Density')
    ax.set_title(f'Exit spikes vs stand-alone spikes (KS p = {ks_p:.3f})' if np.isfinite(ks_p) else 'Exit spikes vs stand-alone spikes')
    ax = axes[1, 0]
    if len(sp) and len(pl):
        bins = np.linspace(0, max(float(sp.quantile(0.99)), float(pl.quantile(0.99)), 0.05) * 1.05, 40)
        ax.hist(sp, bins=bins, histtype='step', density=True, linewidth=1.6, label=f'spike family (n={len(sp)})')
        ax.hist(pl, bins=bins, histtype='step', density=True, linewidth=1.6, label=f'plateau family (n={len(pl)})')
        ax.legend(fontsize=8)
    ax.set_xlabel('pre-bump / peak')
    ax.set_ylabel('Density')
    ax.set_title('Pre-bump by event family')
    ax = axes[1, 1]
    if len(paired):
        lim = max(float(paired.max().max()), 1.0)
        ax.scatter(paired['post_overshoot_pA'], paired['pre_bump_pA'], s=9, alpha=0.5)
        ax.plot([0, lim], [0, lim], linestyle=':', color='k', linewidth=1)
    ax.set_xlabel('Post-event overshoot (pA)')
    ax.set_ylabel('Pre-bump (pA)')
    ax.set_title('Pre-bump vs post-event overshoot')
    ax = axes[1, 2]
    if len(ex):
        picks = ex.sort_values('time_block').groupby('time_block', observed=True).head(1).head(6)
        for k, (_, row) in enumerate(picks.iterrows()):
            check_cancelled(_acid_cancel)
            pad = int(round(0.3 * (row.end_idx - row.start_idx + 1)))
            lo, hi = (max(0, int(row.start_idx) - pad), min(n_samples, int(row.end_idx) + 1 + pad))
            t = (np.arange(lo, hi) - int(row.start_idx)) / fs * 1000.0
            y = analysis_signal[lo:hi] - row.amplitude_baseline_pA
            ax.plot(t / max(t.max(), 1e-09), y - 1.4 * k * max(1.0, row.peak_depth_pA), linewidth=0.7)
        ax.set_xlabel('Normalized time (onset = 0)')
        ax.set_ylabel('Depth (pA), offset per event')
        ax.set_title('Exit-spike examples (one per block)')
    else:
        ax.axis('off')
    finish_figure('12_spike_context_tests')
    (OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_spike_context_tests.json').write_text(json.dumps(json_ready(spike_context_tests) if 'json_ready' in globals() else spike_context_tests, indent=2, default=str))


def definitions_030():
    global _dip_both_halves, binomtest, ks_2samp, mannwhitneyu
    from scipy.stats import ks_2samp, mannwhitneyu, binomtest

    def _dip_both_halves(values: pd.Series, mask) -> tuple[float, float]:
        out = []
        for half in ('discovery', 'validation'):
            check_cancelled(_acid_cancel)
            x = values.loc[mask & (_f['half'] == half)].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            out.append(float(diptest(np.sort(x))[1]) if HAVE_DIPTEST and len(x) >= 30 else np.nan)
        return (out[0], out[1])


# Original notebook cell 32
def stage_032():
    global FAMILY_FEATURES, FAMILY_MODEL_PATH, K, X_fam, _FAM_TAG, _apply_model, _close, _fam, _fit_reference_model, _fit_rows, _gap_ms, _labelled, _ok, _rejected, _run_id, _run_size, _saved, _saved_ok, _tau_design, _tau_instr, _tau_measured, amp_class, ax, axes, bins, bottom, cramers_v, dwell_amplitude_crosstab, dwell_class, event_families, family_by_block, family_feature_frame, family_model, family_model_fingerprint, family_names, family_table, family_time_v, fig, g, hard, k, marker_fractions, multivariate_normal, n_bursts, plot_event_gallery, post, stratified_sample_rows, tags
    check_cancelled(_acid_cancel)
    from scipy.stats import multivariate_normal
    if 'cramers_v' not in globals():

        def cramers_v(table: pd.DataFrame) -> float:
            if table.shape[0] < 2 or table.shape[1] < 2:
                return np.nan
            chi2 = chi2_contingency(table, correction=False)[0]
            n = table.to_numpy().sum()
            phi2 = chi2 / max(n, 1)
            r, k = table.shape
            phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
            rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
            kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
            denom = min(kcorr - 1, rcorr - 1)
            return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan
    if 'stratified_sample_rows' not in globals():

        def stratified_sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
            rng = np.random.default_rng(seed)
            selected = []
            blocks = list(sorted(frame['time_block'].unique()))
            while len(selected) < min(n, len(frame)):
                check_cancelled(_acid_cancel)
                progressed = False
                for block in blocks:
                    check_cancelled(_acid_cancel)
                    candidates = frame.loc[(frame['time_block'] == block) & ~frame.index.isin(selected)]
                    if len(candidates):
                        selected.append(int(rng.choice(candidates.index.to_numpy())))
                        progressed = True
                        if len(selected) >= min(n, len(frame)):
                            break
                if not progressed:
                    break
            return frame.loc[selected]
    if 'plot_event_gallery' not in globals():

        def plot_event_gallery(frame: pd.DataFrame, label_col: str, prefix: str, n_per_group: int=12, pad_ms: float=1.5) -> None:
            groups = sorted((g for g in frame[label_col].dropna().unique() if g != -1))
            pad_samples = int(round(pad_ms * fs / 1000.0))
            for group in groups:
                check_cancelled(_acid_cancel)
                g = frame.loc[frame[label_col] == group]
                selected = stratified_sample_rows(g, n_per_group, SEED + int(group) + 100)
                cols = 4
                rows = int(math.ceil(len(selected) / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
                for ax in axes.ravel():
                    check_cancelled(_acid_cancel)
                    ax.axis('off')
                for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
                    check_cancelled(_acid_cancel)
                    ax.axis('on')
                    start, end = (int(row.start_idx), int(row.end_idx))
                    lo, hi = (max(0, start - pad_samples), min(n_samples, end + 1 + pad_samples))
                    t = (np.arange(lo, hi) - start) / fs * 1000.0
                    ax.plot(t, analysis_signal[lo:hi], linewidth=0.7)
                    ax.axhline(row.amplitude_baseline_pA, linestyle=':', linewidth=0.8)
                    ax.axvspan(0, (end - start + 1) / fs * 1000.0, alpha=0.16)
                    ax.set_title(f'{row.event_id} | block {int(row.time_block)}\n{row.dwell_ms:.3f} ms, {getattr(row, AMP_SIGMA_COL):.1f}σ {AMPLITUDE_STATISTIC}', fontsize=8)
                    ax.set_xlabel('ms from onset', fontsize=8)
                    ax.set_ylabel('pA', fontsize=8)
                fig.suptitle(f'{prefix} {group}: random events spanning time blocks')
                finish_figure(f"07_{prefix.lower().replace(' ', '_')}_{group}")
    FAMILY_FEATURES = ['log_dwell', 'log_peak_frac', 'log_area_frac', 'precursor_sigma', 'flat_fraction', 'log_max_slope']

    def family_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        out['log_dwell'] = np.log10(frame['dwell_ms'].clip(lower=0.001))
        out['log_peak_frac'] = np.log10(frame['peak_fracchange'].clip(lower=0.0001))
        out['log_area_frac'] = np.log10((frame['area_pA_ms'] / frame['local_I0_pA']).clip(lower=1e-05))
        out['precursor_sigma'] = frame['pre_enhancement_sigma'].fillna(0.0).clip(0, 12)
        out['flat_fraction'] = frame['cusum_longest_level_fraction'].fillna(1.0).clip(0, 1)
        out['log_max_slope'] = np.log10(frame['max_slope_sigma'].fillna(1.0).clip(lower=0.3))
        return out
    _fam = primary_catalog.copy().reset_index(drop=True)
    _tau_measured = float(_fam.loc[_fam['fast_recovery_inside_region'].fillna(False).astype(bool), 'decay_tau_ms'].median())
    _tau_design = (meta.get('match_tau_ms') if isinstance(meta, dict) else None) or (MATCH_TAU_MS if TAG == FAMILY_REFERENCE_TAG else None)
    if not _tau_design and (not np.isfinite(_tau_measured)):
        # A recording with zero fast-recovery events makes _tau_measured NaN; without a
        # configured match_tau_ms/reference tag to fall back on, _tau_instr used to
        # become NaN too, and 'dwell_ms > 3.0*NaN' is False for every row -- 'resolved'
        # and 'deep_long' went silently all-False instead of surfacing that the
        # instrument decay constant is simply unknown here.
        raise ValueError(f"{TAG}: no instrument decay constant available -- no fast-recovery events to measure one from, and no match_tau_ms configured or reference tag set. Set MATCH_TAU_MS or designate a FAMILY_REFERENCE_TAG.")
    _tau_instr = float(_tau_design) if _tau_design else _tau_measured
    if _tau_design and np.isfinite(_tau_measured) and (abs(np.log(max(_tau_measured, 1e-09) / _tau_design)) > np.log(1.25)):
        print(f"Note: fast-recovering events decay with a median constant of {_tau_measured:.3f} ms, {_tau_measured / _tau_design:.2f}x the instrument's {_tau_design:.3f} ms; the resolved threshold uses the instrument value.")
    _fam['resolved'] = _fam['dwell_ms'] > 3.0 * _tau_instr
    _fam['precursor'] = _fam['pre_enhancement_sigma'].fillna(0.0) >= FAMILY_PRECURSOR_SIGMA
    _fam['flat'] = _fam['cusum_longest_level_fraction'].fillna(1.0) >= FAMILY_FLAT_FRACTION
    _fam['sharp_onset'] = _fam['max_slope_sigma'].fillna(0.0) >= FAMILY_SHARP_SLOPE_SIGMA
    _fam['deep_long'] = (_fam['peak_fracchange'] >= FAMILY_DEEP_FRAC) & _fam['resolved']
    _fam = _fam.sort_values('start_idx').reset_index(drop=True)
    _gap_ms = np.diff(_fam['start_idx'].to_numpy()) / fs * 1000.0
    _close = np.concatenate(([False], _gap_ms <= BURST_MAX_GAP_MS))
    _run_id = np.cumsum(~_close)
    _run_size = pd.Series(_run_id).map(pd.Series(_run_id).value_counts()).to_numpy()
    _fam['burst_id'] = np.where(_run_size >= BURST_MIN_EVENTS, _run_id, -1)
    _fam['in_burst'] = _fam['burst_id'] >= 0
    n_bursts = int(_fam.loc[_fam['in_burst'], 'burst_id'].nunique())
    print(f"Bursts (>= {BURST_MIN_EVENTS} events with onsets <= {BURST_MAX_GAP_MS:g} ms apart): {n_bursts}, containing {int(_fam['in_burst'].sum())} events" + (f"; first at {_fam.loc[_fam['in_burst'], 'onset_s'].min():.1f} s" if n_bursts else ''))
    X_fam = family_feature_frame(_fam)
    _ok = X_fam.replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
    _fit_rows = _ok & (~_fam['in_burst'].to_numpy() if FAMILY_EXCLUDE_BURSTS else True)

    def _fit_reference_model(X: pd.DataFrame, blocks: np.ndarray) -> dict:
        disc = blocks % 2 == 0
        scaler = RobustScaler().fit(X.loc[disc])
        Z = np.clip(scaler.transform(X), -6, 6)
        rows, models = ([], {})
        for k in range(1, 7):
            check_cancelled(_acid_cancel)
            m = GaussianMixture(k, n_init=10, reg_covar=0.0001, random_state=SEED).fit(Z[disc])
            models[k] = m
            rows.append({'k': k, 'bic': m.bic(Z[disc]), 'heldout': m.score(Z[~disc]), 'min_weight': float(m.weights_.min())})
        scores = pd.DataFrame(rows)
        adm = scores.loc[scores['min_weight'] >= FAMILY_MIN_WEIGHT]
        k_best = int(adm.loc[adm['heldout'].idxmax(), 'k']) if len(adm) else 1
        m = models[k_best]
        order = np.argsort(m.means_[:, 0])
        return {'reference_tag': TAG, 'features': FAMILY_FEATURES, 'k': k_best, 'scaler_center': scaler.center_.tolist(), 'scaler_scale': scaler.scale_.tolist(), 'weights': m.weights_[order].tolist(), 'means': m.means_[order].tolist(), 'covariances': m.covariances_[order].tolist(), 'model_selection': scores.round(4).to_dict(orient='records')}

    def _apply_model(model: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = np.clip((X[model['features']].to_numpy(float) - np.asarray(model['scaler_center'])) / np.asarray(model['scaler_scale']), -6, 6)
        logp = np.column_stack([np.log(w) + multivariate_normal(mean=np.asarray(mu), cov=np.asarray(cov), allow_singular=True).logpdf(Z) for w, mu, cov in zip(model['weights'], model['means'], model['covariances'])])
        logp -= logp.max(axis=1, keepdims=True)
        post = np.exp(logp)
        post /= post.sum(axis=1, keepdims=True)
        return (post, post.argmax(axis=1))
    FAMILY_MODEL_PATH = FAMILY_MODEL_PATH if primary_polarity == 1 else FAMILY_MODEL_PATH.with_name('event_family_model_resistive.json')
    _saved = json.loads(FAMILY_MODEL_PATH.read_text()) if FAMILY_MODEL_PATH.is_file() else None
    _saved_ok = _saved is not None and list(_saved.get('features', [])) == FAMILY_FEATURES
    if _saved is not None and (not _saved_ok):
        print(f"Saved family model uses features {_saved.get('features')} != {FAMILY_FEATURES}: it will be refitted on this tag.")
    if _saved_ok and TAG != FAMILY_REFERENCE_TAG:
        family_model = _saved
        print(f"Family model: loaded from {FAMILY_MODEL_PATH.name} (reference tag {family_model['reference_tag']}, k = {family_model['k']})")
    elif _saved_ok and TAG == FAMILY_REFERENCE_TAG and (not FAMILY_REFIT_REFERENCE):
        family_model = _saved
        print(f"Family model: reusing the saved reference model (k = {family_model['k']}); set FAMILY_REFIT_REFERENCE = True to refit")
    else:
        family_model = _fit_reference_model(X_fam.loc[_fit_rows], _fam.loc[_fit_rows, 'time_block'].to_numpy())
        FAMILY_MODEL_PATH.write_text(json.dumps(family_model, indent=2))
        print(f"Family model: fitted on this tag as the reference (k = {family_model['k']}) and saved to {FAMILY_MODEL_PATH}")
        display(pd.DataFrame(family_model['model_selection']))
    post, hard = _apply_model(family_model, X_fam.loc[_ok])
    _fam['family'] = -1
    _fam.loc[_ok, 'family'] = hard
    _fam['family_posterior'] = np.nan
    _fam.loc[_ok, 'family_posterior'] = post.max(axis=1)
    K = family_model['k']
    family_names = {}
    for k in range(K):
        check_cancelled(_acid_cancel)
        g = _fam.loc[_fam['family'] == k]
        if not len(g):
            family_names[k] = f'F{k}: (empty in this tag)'
            continue
        tags = []
        tags.append('resolved' if g['resolved'].mean() >= 0.5 else 'unresolved')
        tags.append('precursor' if g['precursor'].mean() >= 0.5 else 'no precursor')
        tags.append('flat' if g['flat'].mean() >= 0.5 else 'spike')
        if g['flat'].mean() >= 0.5:
            tags.append('sharp edges' if g['sharp_onset'].mean() >= 0.5 else 'slow edges')
        family_names[k] = f"F{k}: {g['dwell_ms'].median():.2f} ms, {g['peak_fracchange'].median():.3f} I0, " + ', '.join(tags)
    family_table = pd.DataFrame([{'family': k, 'name': family_names[k], 'n': int((_fam['family'] == k).sum()), 'share': float((_fam['family'] == k).mean()), 'soft_share': float(post[:, k].mean()), 'rate_per_s': float((_fam['family'] == k).sum() / duration_s), 'dwell_median_ms': g['dwell_ms'].median(), 'dwell_iqr_ms': float(g['dwell_ms'].quantile(0.75) - g['dwell_ms'].quantile(0.25)), 'peak_frac_median': g['peak_fracchange'].median(), 'area_frac_median': (g['area_pA_ms'] / g['local_I0_pA']).median(), 'precursor_fraction': g['precursor'].mean(), 'flat_fraction': g['flat'].mean(), 'resolved_fraction': g['resolved'].mean(), 'sharp_onset_fraction': g['sharp_onset'].mean(), 'deep_long_fraction': g['deep_long'].mean(), 'in_burst_fraction': g['in_burst'].mean(), 'decay_tau_median_ms': g['decay_tau_ms'].median(), 'blocks_present': int(g['time_block'].nunique())} for k, g in ((k, _fam.loc[_fam['family'] == k]) for k in range(K))])
    _labelled = _fam.loc[_fam['family'] >= 0]
    family_by_block = pd.crosstab(_labelled['time_block'], _labelled['family'], normalize='index').reindex(columns=range(K), fill_value=0.0)
    family_time_v = cramers_v(pd.crosstab(_labelled['time_block'], _labelled['family'])) if K > 1 else np.nan
    # Was hard-coded to cusum_catalog, which only ever holds polarity==1 (conductive)
    # rows -- so 'cusum_catalog[...polarity == primary_polarity]' silently matched
    # nothing whenever the resistive detector was primary (primary_polarity == -1).
    # detector_catalogs[PRIMARY_DETECTOR] is the primary detector's OWN raw catalog
    # regardless of polarity, so the filter below is now a true (defensive) invariant
    # check rather than the thing silently emptying the result.
    _primary_raw = detector_catalogs[PRIMARY_DETECTOR]
    _rejected = _primary_raw.loc[_primary_raw['accepted_base'] & ~_primary_raw['accepted'] & (_primary_raw['polarity'] == primary_polarity)] if 'accepted_base' in _primary_raw else _primary_raw.iloc[0:0]
    marker_fractions = {'n_events': int(len(_fam)), 'instrument_tau_ms': _tau_instr, 'instrument_tau_measured_ms': _tau_measured, 'resolved_fraction': float(_fam['resolved'].mean()), 'precursor_fraction': float(_fam['precursor'].mean()), 'flat_fraction': float(_fam['flat'].mean()), 'precursor_and_unresolved_fraction': float((_fam['precursor'] & ~_fam['resolved']).mean()), 'flat_no_precursor_fraction': float((_fam['flat'] & ~_fam['precursor']).mean()), 'sharp_onset_fraction': float(_fam['sharp_onset'].mean()), 'flat_no_precursor_sharp_fraction': float((_fam['flat'] & ~_fam['precursor'] & _fam['sharp_onset']).mean()), 'flat_no_precursor_slow_fraction': float((_fam['flat'] & ~_fam['precursor'] & ~_fam['sharp_onset']).mean()), 'n_bursts': n_bursts, 'events_in_bursts': int(_fam['in_burst'].sum()), 'deep_long_fraction': float(_fam['deep_long'].mean()), 'deep_long_count': int(_fam['deep_long'].sum()), 'gate_rejected_candidates': int(len(_rejected)), 'gate_rejected_short_shallow': int(((_rejected['dwell_ms'] < 1.0) & (_rejected['peak_depth_pA'] < 12)).sum()) if len(_rejected) else 0, 'dwell_quantiles_ms': {str(q): float(_fam['dwell_ms'].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, 'peak_frac_quantiles': {str(q): float(_fam['peak_fracchange'].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, 'precursor_to_peak_median': float(_fam.loc[_fam['precursor'], 'pre_bump_to_peak_ratio'].median()) if _fam['precursor'].any() else np.nan}
    dwell_class = pd.cut(_fam['dwell_ms'], [0, 0.5, 2.0, 10.0, np.inf], labels=['<0.5 ms', '0.5-2 ms', '2-10 ms', '>10 ms'])
    amp_class = pd.cut(_fam['peak_fracchange'], [0, 0.05, 0.1, 0.2, np.inf], labels=['<5%', '5-10%', '10-20%', '>20%'])
    dwell_amplitude_crosstab = pd.crosstab(dwell_class, amp_class)
    print(f"Instrument recovery constant {_tau_instr:.3f} ms ({('bandwidth-match value' if _tau_design else 'measured from the events')}; measured {_tau_measured:.3f} ms) -> events shorter than {3 * _tau_instr:.2f} ms are unresolved (their peak measures charge, not depth)")
    print(f"Markers: resolved {100 * marker_fractions['resolved_fraction']:.0f}%, precursor {100 * marker_fractions['precursor_fraction']:.0f}%, flat {100 * marker_fractions['flat_fraction']:.0f}%, sharp onset {100 * marker_fractions['sharp_onset_fraction']:.0f}%; flat-without-precursor {100 * marker_fractions['flat_no_precursor_fraction']:.0f}% (sharp edges {100 * marker_fractions['flat_no_precursor_sharp_fraction']:.0f}%, slow edges {100 * marker_fractions['flat_no_precursor_slow_fraction']:.0f}%); deep-long (>= {FAMILY_DEEP_FRAC:.0%} I0, resolved) {marker_fractions['deep_long_count']}; gate-rejected candidates {marker_fractions['gate_rejected_candidates']} (short & shallow: {marker_fractions['gate_rejected_short_shallow']})")
    print(f"Families (reference model from {family_model['reference_tag']}, k = {K}); family x time-block Cramér's V = {family_time_v:.3f}")
    display(family_table.round(4))
    print('Family fractions by time block:')
    display(family_by_block.round(3))
    print('Dwell class x amplitude class (peak as fraction of I0):')
    display(dwell_amplitude_crosstab)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    if len(_rejected):
        ax.scatter(np.log10(_rejected['dwell_ms'].clip(lower=0.001)), np.log10(_rejected['peak_fracchange'].clip(lower=0.0001)), s=6, color='lightgray', alpha=0.6, label=f'gate-rejected candidates (n={len(_rejected)})')
    for k in range(K):
        check_cancelled(_acid_cancel)
        g = _fam.loc[_fam['family'] == k]
        ax.scatter(X_fam.loc[g.index, 'log_dwell'], X_fam.loc[g.index, 'log_peak_frac'], s=10, alpha=0.7, label=family_names[k][:48])
    ax.axvline(np.log10(3 * _tau_instr), linestyle=':', color='k', linewidth=1)
    ax.text(np.log10(3 * _tau_instr), ax.get_ylim()[1], ' unresolved | resolved', va='top', fontsize=8)
    ax.set_xlabel('log10 dwell (ms)')
    ax.set_ylabel('log10 peak / I0')
    ax.set_title('Dwell-amplitude map')
    ax.legend(fontsize=7, loc='lower right')
    ax = axes[0, 1]
    for k in range(K):
        check_cancelled(_acid_cancel)
        g = _fam.loc[_fam['family'] == k]
        ax.scatter(X_fam.loc[g.index, 'log_dwell'], X_fam.loc[g.index, 'log_area_frac'], s=10, alpha=0.7)
    ax.set_xlabel('log10 dwell (ms)')
    ax.set_ylabel('log10 charge deficit / I0 (pA·ms / pA)')
    ax.set_title('Dwell-charge map')
    ax = axes[1, 0]
    bottom = np.zeros(len(family_by_block))
    for k in range(K):
        check_cancelled(_acid_cancel)
        ax.bar(family_by_block.index, family_by_block[k], bottom=bottom, label=f'F{k}')
        bottom += family_by_block[k].to_numpy()
    ax.set_xlabel('Time block')
    ax.set_ylabel('Fraction')
    ax.set_title(f"Family fractions by block (Cramér's V = {family_time_v:.2f})")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    bins = np.linspace(X_fam['log_dwell'].min(), X_fam['log_dwell'].max(), 40)
    for k in range(K):
        check_cancelled(_acid_cancel)
        g = _fam.loc[_fam['family'] == k]
        ax.hist(X_fam.loc[g.index, 'log_dwell'], bins=bins, histtype='step', linewidth=1.5, label=f'F{k} (n={len(g)})')
    ax.set_xlabel('log10 dwell (ms)')
    ax.set_ylabel('Events')
    ax.set_title('Dwell by family')
    ax.legend(fontsize=8)
    finish_figure('14_event_families')
    plot_event_gallery(_fam.rename(columns={'family': 'event_family'}), 'event_family', 'Event family')
    family_model_fingerprint = hashlib.sha256(json.dumps({k: family_model[k] for k in ('features', 'k', 'weights', 'means', 'covariances', 'scaler_center', 'scaler_scale')}, sort_keys=True).encode()).hexdigest()[:12]
    event_families = {'tag': TAG, 'duration_s': duration_s, 'primary_set': PRIMARY_SET_NAME, 'primary_polarity': primary_polarity, 'family_model_fingerprint': family_model_fingerprint, 'instrument_tau_ms': _tau_instr, 'instrument_tau_measured_ms': _tau_measured, 'amplitude_statistic': AMPLITUDE_STATISTIC, 'gate_statistic': GATE_STATISTIC, 'matched_bandwidth_hz': meta.get('matched_bandwidth_hz'), 'raw_rolloff_hz': meta.get('raw_rolloff_hz'), 'n_events': int(len(_fam)), 'rate_per_s': len(_fam) / duration_s, 'global_i0_pA': float(global_i0) if np.isfinite(global_i0) else None, 'global_noise_pA': float(global_noise), 'reference_tag': family_model['reference_tag'], 'k': K, 'family_names': family_names, 'family_table': family_table.to_dict(orient='records'), 'family_by_block': family_by_block.round(4).to_dict(orient='index'), 'family_time_cramers_v': family_time_v, 'markers': marker_fractions, 'dwell_amplitude_crosstab': {str(r): {str(c): int(v) for c, v in row.items()} for r, row in dwell_amplitude_crosstab.iterrows()}, 'resistive_independent_events': int(len(analysis_sets['cusum_resistive'])) if 'cusum_resistive' in analysis_sets else None, 'detector': {'cusum_selected': {k: list(v) for k, v in cusum_selected.items()}, 'recall_map': cusum_recall_map.to_dict(orient='records'), 'gate_statistic': GATE_STATISTIC, 'acceptance_floor_pA': {str(d): float(acceptance_floor_pA(d)) for d in (0.1, 0.2, 0.5, 1.0, 5.0, 20.0)}}}
    (OUT_DIR / f'{TAG}_event_families.json').write_text(json.dumps(json_ready(event_families) if 'json_ready' in globals() else event_families, indent=2, default=str))
    _fam[['event_id', 'family', 'family_posterior', 'resolved', 'precursor', 'flat', 'sharp_onset', 'deep_long', 'in_burst', 'burst_id']].to_parquet(OUT_DIR / f'{TAG}_event_families.parquet', index=False)
    _FAM_TAG = TAG
    print('Wrote', OUT_DIR / f'{TAG}_event_families.json')


def definitions_032():
    global _apply_model, _fit_reference_model, cramers_v, family_feature_frame, multivariate_normal, plot_event_gallery, stratified_sample_rows
    from scipy.stats import multivariate_normal

    def cramers_v(table: pd.DataFrame) -> float:
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan
        chi2 = chi2_contingency(table, correction=False)[0]
        n = table.to_numpy().sum()
        phi2 = chi2 / max(n, 1)
        r, k = table.shape
        phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
        rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
        kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
        denom = min(kcorr - 1, rcorr - 1)
        return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan

    def stratified_sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        selected = []
        blocks = list(sorted(frame['time_block'].unique()))
        while len(selected) < min(n, len(frame)):
            check_cancelled(_acid_cancel)
            progressed = False
            for block in blocks:
                check_cancelled(_acid_cancel)
                candidates = frame.loc[(frame['time_block'] == block) & ~frame.index.isin(selected)]
                if len(candidates):
                    selected.append(int(rng.choice(candidates.index.to_numpy())))
                    progressed = True
                    if len(selected) >= min(n, len(frame)):
                        break
            if not progressed:
                break
        return frame.loc[selected]

    def plot_event_gallery(frame: pd.DataFrame, label_col: str, prefix: str, n_per_group: int=12, pad_ms: float=1.5) -> None:
        groups = sorted((g for g in frame[label_col].dropna().unique() if g != -1))
        pad_samples = int(round(pad_ms * fs / 1000.0))
        for group in groups:
            check_cancelled(_acid_cancel)
            g = frame.loc[frame[label_col] == group]
            selected = stratified_sample_rows(g, n_per_group, SEED + int(group) + 100)
            cols = 4
            rows = int(math.ceil(len(selected) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
            for ax in axes.ravel():
                check_cancelled(_acid_cancel)
                ax.axis('off')
            for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
                check_cancelled(_acid_cancel)
                ax.axis('on')
                start, end = (int(row.start_idx), int(row.end_idx))
                lo, hi = (max(0, start - pad_samples), min(n_samples, end + 1 + pad_samples))
                t = (np.arange(lo, hi) - start) / fs * 1000.0
                ax.plot(t, analysis_signal[lo:hi], linewidth=0.7)
                ax.axhline(row.amplitude_baseline_pA, linestyle=':', linewidth=0.8)
                ax.axvspan(0, (end - start + 1) / fs * 1000.0, alpha=0.16)
                ax.set_title(f'{row.event_id} | block {int(row.time_block)}\n{row.dwell_ms:.3f} ms, {getattr(row, AMP_SIGMA_COL):.1f}σ {AMPLITUDE_STATISTIC}', fontsize=8)
                ax.set_xlabel('ms from onset', fontsize=8)
                ax.set_ylabel('pA', fontsize=8)
            fig.suptitle(f'{prefix} {group}: random events spanning time blocks')
            finish_figure(f"07_{prefix.lower().replace(' ', '_')}_{group}")

    def family_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        out['log_dwell'] = np.log10(frame['dwell_ms'].clip(lower=0.001))
        out['log_peak_frac'] = np.log10(frame['peak_fracchange'].clip(lower=0.0001))
        out['log_area_frac'] = np.log10((frame['area_pA_ms'] / frame['local_I0_pA']).clip(lower=1e-05))
        out['precursor_sigma'] = frame['pre_enhancement_sigma'].fillna(0.0).clip(0, 12)
        out['flat_fraction'] = frame['cusum_longest_level_fraction'].fillna(1.0).clip(0, 1)
        out['log_max_slope'] = np.log10(frame['max_slope_sigma'].fillna(1.0).clip(lower=0.3))
        return out

    def _fit_reference_model(X: pd.DataFrame, blocks: np.ndarray) -> dict:
        disc = blocks % 2 == 0
        scaler = RobustScaler().fit(X.loc[disc])
        Z = np.clip(scaler.transform(X), -6, 6)
        rows, models = ([], {})
        for k in range(1, 7):
            check_cancelled(_acid_cancel)
            m = GaussianMixture(k, n_init=10, reg_covar=0.0001, random_state=SEED).fit(Z[disc])
            models[k] = m
            rows.append({'k': k, 'bic': m.bic(Z[disc]), 'heldout': m.score(Z[~disc]), 'min_weight': float(m.weights_.min())})
        scores = pd.DataFrame(rows)
        adm = scores.loc[scores['min_weight'] >= FAMILY_MIN_WEIGHT]
        k_best = int(adm.loc[adm['heldout'].idxmax(), 'k']) if len(adm) else 1
        m = models[k_best]
        order = np.argsort(m.means_[:, 0])
        return {'reference_tag': TAG, 'features': FAMILY_FEATURES, 'k': k_best, 'scaler_center': scaler.center_.tolist(), 'scaler_scale': scaler.scale_.tolist(), 'weights': m.weights_[order].tolist(), 'means': m.means_[order].tolist(), 'covariances': m.covariances_[order].tolist(), 'model_selection': scores.round(4).to_dict(orient='records')}

    def _apply_model(model: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = np.clip((X[model['features']].to_numpy(float) - np.asarray(model['scaler_center'])) / np.asarray(model['scaler_scale']), -6, 6)
        logp = np.column_stack([np.log(w) + multivariate_normal(mean=np.asarray(mu), cov=np.asarray(cov), allow_singular=True).logpdf(Z) for w, mu, cov in zip(model['weights'], model['means'], model['covariances'])])
        logp -= logp.max(axis=1, keepdims=True)
        post = np.exp(logp)
        post /= post.sum(axis=1, keepdims=True)
        return (post, post.argmax(axis=1))


# Original notebook cell 34
def stage_034():
    global OVERLAY_ENVELOPE_MS, OVERLAY_MIN_SPAN_FRACTION, OVERLAY_N_ZOOM, OVERLAY_ZOOM_SPAN_S, _, _blocks, _bottom, _centers, _counts, _edges, _env, _events_in, _fam_colors, _n_env, _other_label, _rejected_primary, _res_set, _rng_overlay, _t_env, ax, axes, b_id, bg, c, ev, fig, g, h, j, k, plot_trace_window, title
    check_cancelled(_acid_cancel)
    OVERLAY_ZOOM_SPAN_S = 1.0
    OVERLAY_N_ZOOM = 6
    OVERLAY_ENVELOPE_MS = 20.0
    OVERLAY_MIN_SPAN_FRACTION = 0.004
    _fam_colors = {k: f'C{k}' for k in range(K)}
    if primary_polarity == 1:
        _res_set = analysis_sets.get('cusum_resistive', cusum_catalog.iloc[0:0])
        _other_label = 'resistive (independent)'
    else:
        _res_set = analysis_sets.get('cusum', cusum_catalog.iloc[0:0])
        _other_label = 'conductive'
    # Same fix as _rejected above: cusum_catalog only ever holds polarity==1 rows, so
    # this silently returned nothing whenever the resistive detector was primary.
    _primary_raw_overlay = detector_catalogs[PRIMARY_DETECTOR]
    _rejected_primary = _primary_raw_overlay.loc[_primary_raw_overlay['accepted_base'] & ~_primary_raw_overlay['accepted'] & (_primary_raw_overlay['polarity'] == primary_polarity)] if 'accepted_base' in _primary_raw_overlay else _primary_raw_overlay.iloc[0:0]

    def _events_in(frame: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
        return frame.loc[(frame['end_idx'] >= lo) & (frame['start_idx'] < hi)]

    def plot_trace_window(center_s: float, span_s: float=OVERLAY_ZOOM_SPAN_S, ax=None, title: str | None=None, legend: bool=True):
        """Full-resolution trace around center_s with family-coloured event regions."""
        lo = max(0, int(round((center_s - span_s / 2) * fs)))
        hi = min(n_samples, int(round((center_s + span_s / 2) * fs)))
        if ax is None:
            _, ax = plt.subplots(figsize=(16, 4))
        t = np.arange(lo, hi) / fs
        y = analysis_signal[lo:hi]
        ax.plot(t, y, linewidth=0.5, color='0.25')
        y_lo, y_hi = (float(np.min(y)), float(np.max(y)))
        pad = 0.08 * (y_hi - y_lo + 1e-09)
        ax.set_ylim(y_lo - 2.5 * pad, y_hi + pad)
        shown = set()
        min_w = OVERLAY_MIN_SPAN_FRACTION * span_s
        for _, ev in _events_in(_fam, lo, hi).iterrows():
            check_cancelled(_acid_cancel)
            k = int(ev['family'])
            label = family_names.get(k, f'F{k}')[:52] if k not in shown else None
            shown.add(k)
            s0, s1 = (ev['start_idx'] / fs, (ev['end_idx'] + 1) / fs)
            if s1 - s0 < min_w:
                mid = 0.5 * (s0 + s1)
                s0, s1 = (mid - min_w / 2, mid + min_w / 2)
            ax.axvspan(s0, s1, color=_fam_colors.get(k, '0.5'), alpha=0.28, lw=0, label=label)
            seg = analysis_signal[int(ev['start_idx']):int(ev['end_idx']) + 1]
            p = int(np.argmin(seg))
            ax.plot([(ev['start_idx'] + p) / fs], [seg[p]], marker='o', ms=4, color=_fam_colors.get(k, '0.5'), mec='k', mew=0.4, zorder=4)
        if len(_res_set):
            first = True
            for _, ev in _events_in(_res_set, lo, hi).iterrows():
                check_cancelled(_acid_cancel)
                ax.axvspan(ev['start_idx'] / fs, (ev['end_idx'] + 1) / fs, facecolor='none', edgecolor='k', hatch='//', lw=0.8, label=_other_label if first else None)
                first = False
        for b_id, bg in _events_in(_fam.loc[_fam['in_burst']], lo, hi).groupby('burst_id'):
            check_cancelled(_acid_cancel)
            ax.axvspan(bg['start_idx'].min() / fs, (bg['end_idx'].max() + 1) / fs, facecolor='gold', alpha=0.15, lw=0, label='burst (fragmented excursion)' if 'burst' not in shown else None)
            shown.add('burst')
        if len(_rejected_primary):
            rej = _events_in(_rejected_primary, lo, hi)
            if len(rej):
                ax.vlines(rej['start_idx'] / fs, y_lo - 2.2 * pad, y_lo - 1.2 * pad, color='0.6', lw=1.0, label='gate-rejected candidate')
        ax.set_xlim(lo / fs, hi / fs)
        ax.set_xlabel('Recording time (s)')
        ax.set_ylabel('Detrended current (pA)')
        ax.set_title(title or f'{center_s:.2f} s ± {span_s / 2:g} s')
        if legend:
            ax.legend(fontsize=7, loc='lower right', ncol=2)
        return ax
    _env = max(1, int(round(OVERLAY_ENVELOPE_MS * fs / 1000.0)))
    _n_env = n_samples // _env
    _blocks = analysis_signal[:_n_env * _env].reshape(_n_env, _env)
    _t_env = (np.arange(_n_env) + 0.5) * _env / fs
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax = axes[0]
    ax.fill_between(_t_env, _blocks.min(axis=1), _blocks.max(axis=1), color='0.75', lw=0, label=f'trace envelope ({OVERLAY_ENVELOPE_MS:g} ms min/max)')
    for k in range(K):
        check_cancelled(_acid_cancel)
        g = _fam.loc[_fam['family'] == k]
        ax.scatter(g['onset_s'], -g['peak_depth_pA'], s=9, color=_fam_colors[k], alpha=0.8, label=family_names[k][:52], zorder=3)
    if len(_res_set):
        ax.scatter(_res_set['onset_s'], _res_set['peak_depth_pA'], s=14, facecolors='none', edgecolors='k', lw=0.7, label=_other_label, zorder=3)
    for b_id, bg in _fam.loc[_fam['in_burst']].groupby('burst_id'):
        check_cancelled(_acid_cancel)
        ax.axvspan(bg['onset_s'].min() - 0.5, bg['onset_s'].max() + 0.5, facecolor='gold', alpha=0.35, lw=0, zorder=1)
    ax.set_ylabel('Detrended current (pA)')
    ax.set_title(f'{TAG}: whole recording with primary events at their onset and peak (colour = family)')
    ax.legend(fontsize=7, loc='lower left', ncol=3)
    ax = axes[1]
    _edges = np.linspace(0, duration_s, 61)
    _bottom = np.zeros(len(_edges) - 1)
    for k in range(K):
        check_cancelled(_acid_cancel)
        h, _ = np.histogram(_fam.loc[_fam['family'] == k, 'onset_s'], bins=_edges)
        ax.bar(0.5 * (_edges[1:] + _edges[:-1]), h / np.diff(_edges), width=np.diff(_edges), bottom=_bottom, color=_fam_colors[k], lw=0)
        _bottom += h / np.diff(_edges)
    ax.set_ylabel('Events / s')
    ax.set_xlabel('Recording time (s)')
    ax.set_title('Family rates in 20 s bins')
    finish_figure('16_trace_overview')
    _rng_overlay = np.random.default_rng(SEED + 11)
    _centers = []
    for k in range(K):
        check_cancelled(_acid_cancel)
        g = _fam.loc[_fam['family'] == k]
        if len(g):
            ev = g.iloc[int(_rng_overlay.integers(len(g)))]
            _centers.append((float(ev['onset_s']), f"F{k} example {ev['event_id']} — {family_names[k].split(': ', 1)[-1]}"))
    if len(_centers) < OVERLAY_N_ZOOM:
        _counts, _ = np.histogram(_fam['onset_s'], bins=np.arange(0, duration_s + OVERLAY_ZOOM_SPAN_S, OVERLAY_ZOOM_SPAN_S))
        for j in np.argsort(_counts)[::-1]:
            check_cancelled(_acid_cancel)
            c = (j + 0.5) * OVERLAY_ZOOM_SPAN_S
            if all((abs(c - c0) > OVERLAY_ZOOM_SPAN_S for c0, _ in _centers)):
                _centers.append((float(c), f'busy stretch ({_counts[j]} events in {OVERLAY_ZOOM_SPAN_S:g} s)'))
            if len(_centers) >= OVERLAY_N_ZOOM:
                break
    _centers = _centers[:OVERLAY_N_ZOOM]
    fig, axes = plt.subplots(len(_centers), 1, figsize=(18, 3.6 * len(_centers)))
    for ax, (c, title) in zip(np.atleast_1d(axes), _centers):
        check_cancelled(_acid_cancel)
        plot_trace_window(c, OVERLAY_ZOOM_SPAN_S, ax=ax, title=title, legend=ax is np.atleast_1d(axes)[0])
    finish_figure('17_trace_zoom')
    print(f'Overlay: {len(_fam)} primary events in {K} families, {len(_res_set)} {_other_label} events, {len(_rejected_primary)} gate-rejected candidates. Call plot_trace_window(center_s, span_s) for any other window.')


def definitions_034():
    global _events_in, plot_trace_window

    def _events_in(frame: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
        return frame.loc[(frame['end_idx'] >= lo) & (frame['start_idx'] < hi)]

    def plot_trace_window(center_s: float, span_s: float=OVERLAY_ZOOM_SPAN_S, ax=None, title: str | None=None, legend: bool=True):
        """Full-resolution trace around center_s with family-coloured event regions."""
        lo = max(0, int(round((center_s - span_s / 2) * fs)))
        hi = min(n_samples, int(round((center_s + span_s / 2) * fs)))
        if ax is None:
            _, ax = plt.subplots(figsize=(16, 4))
        t = np.arange(lo, hi) / fs
        y = analysis_signal[lo:hi]
        ax.plot(t, y, linewidth=0.5, color='0.25')
        y_lo, y_hi = (float(np.min(y)), float(np.max(y)))
        pad = 0.08 * (y_hi - y_lo + 1e-09)
        ax.set_ylim(y_lo - 2.5 * pad, y_hi + pad)
        shown = set()
        min_w = OVERLAY_MIN_SPAN_FRACTION * span_s
        for _, ev in _events_in(_fam, lo, hi).iterrows():
            check_cancelled(_acid_cancel)
            k = int(ev['family'])
            label = family_names.get(k, f'F{k}')[:52] if k not in shown else None
            shown.add(k)
            s0, s1 = (ev['start_idx'] / fs, (ev['end_idx'] + 1) / fs)
            if s1 - s0 < min_w:
                mid = 0.5 * (s0 + s1)
                s0, s1 = (mid - min_w / 2, mid + min_w / 2)
            ax.axvspan(s0, s1, color=_fam_colors.get(k, '0.5'), alpha=0.28, lw=0, label=label)
            seg = analysis_signal[int(ev['start_idx']):int(ev['end_idx']) + 1]
            p = int(np.argmin(seg))
            ax.plot([(ev['start_idx'] + p) / fs], [seg[p]], marker='o', ms=4, color=_fam_colors.get(k, '0.5'), mec='k', mew=0.4, zorder=4)
        if len(_res_set):
            first = True
            for _, ev in _events_in(_res_set, lo, hi).iterrows():
                check_cancelled(_acid_cancel)
                ax.axvspan(ev['start_idx'] / fs, (ev['end_idx'] + 1) / fs, facecolor='none', edgecolor='k', hatch='//', lw=0.8, label=_other_label if first else None)
                first = False
        for b_id, bg in _events_in(_fam.loc[_fam['in_burst']], lo, hi).groupby('burst_id'):
            check_cancelled(_acid_cancel)
            ax.axvspan(bg['start_idx'].min() / fs, (bg['end_idx'].max() + 1) / fs, facecolor='gold', alpha=0.15, lw=0, label='burst (fragmented excursion)' if 'burst' not in shown else None)
            shown.add('burst')
        if len(_rejected_primary):
            rej = _events_in(_rejected_primary, lo, hi)
            if len(rej):
                ax.vlines(rej['start_idx'] / fs, y_lo - 2.2 * pad, y_lo - 1.2 * pad, color='0.6', lw=1.0, label='gate-rejected candidate')
        ax.set_xlim(lo / fs, hi / fs)
        ax.set_xlabel('Recording time (s)')
        ax.set_ylabel('Detrended current (pA)')
        ax.set_title(title or f'{center_s:.2f} s ± {span_s / 2:g} s')
        if legend:
            ax.legend(fontsize=7, loc='lower right', ncol=2)
        return ax


# Original notebook cell 36
def stage_036():
    global AMP_LABEL, D, D_fam, EVENTS_ALL, _arms, _available, _cliffs_delta, _e, _fam_arms, _figure_style, _kde_mode, _ks, _mma_box, _mwu, _name, _tag, figure_set, figure_set_results, gaussian_kde, lab, load_tag_events, parts, pfx, sns, sub, tags
    check_cancelled(_acid_cancel)
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    from scipy.stats import gaussian_kde, mannwhitneyu as _mwu, ks_2samp as _ks
    AMP_LABEL = {'peak_fracchange': 'peak $\\Delta I/I_0$', 'sustained_fracchange': 'sustained $\\Delta I/I_0$ (median)', 'half_peak_mean_fracchange': 'half-peak core $\\Delta I/I_0$', 'peak_depth_pA': 'peak $\\Delta I$ (pA)', 'sustained_depth_pA': 'sustained $\\Delta I$ (pA)', 'area_frac': 'charge deficit / $I_0$ (pA·ms/pA)'}.get(AMP_VAR, AMP_VAR)

    def _figure_style():
        plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 12, 'legend.fontsize': 9})

    def load_tag_events(tag: str) -> pd.DataFrame:
        """Primary events of a tag with family labels: from this kernel when the family cell has run for this
    very tag, otherwise from the tag's output folder, using the primary set recorded in its family summary
    (the pre-clustering catalog), so every tag enters with its own primary definition."""
        if tag == TAG and globals().get('_FAM_TAG') == tag:
            frame = _fam.copy()
        else:
            folder = BASE_DIR / f'{tag}_subpopulation'
            fam_json = folder / f'{tag}_event_families.json'
            fam_path = folder / f'{tag}_event_families.parquet'
            primary_set = None
            if fam_json.is_file():
                try:
                    primary_set = json.loads(fam_json.read_text()).get('primary_set')
                except Exception as exc:
                    print(f'[{tag}] could not read {fam_json.name}: {exc}')
            cat_path = folder / f'{tag}_{primary_set}_catalog_preclustering.parquet' if primary_set else None
            if cat_path is not None and cat_path.is_file() and fam_path.is_file():
                frame = pd.read_parquet(cat_path).merge(pd.read_parquet(fam_path), on='event_id', how='inner')
            else:
                cat_path = folder / f'{tag}_cusum_catalog.parquet'
                if not cat_path.is_file():
                    return pd.DataFrame()
                frame = pd.read_parquet(cat_path)
                frame = frame.loc[frame['accepted'] & (frame.get('polarity', 1) == 1)].copy()
                if fam_path.is_file():
                    frame = frame.merge(pd.read_parquet(fam_path), on='event_id', how='inner')
                else:
                    frame['family'] = -1
                    frame['in_burst'] = False
        frame['tag'] = tag
        if 'area_frac' not in frame:
            frame['area_frac'] = frame['area_pA_ms'] / frame['local_I0_pA']
        if FAMILY_EXCLUDE_BURSTS and 'in_burst' in frame:
            frame = frame.loc[~frame['in_burst'].fillna(False).astype(bool)]
        return frame
    _available = {}
    for _tag in TAG_ARMS:
        check_cancelled(_acid_cancel)
        _e = load_tag_events(_tag)
        if len(_e):
            _available[_tag] = _e
    EVENTS_ALL = pd.concat(_available.values(), ignore_index=True) if _available else pd.DataFrame()
    print('Tags with events available for the figure set:', {t: len(e) for t, e in _available.items()})

    def _mma_box(ax, vals, y, colour, h=0.34):
        """Box + whiskers with Tukey near (1.5 IQR) and far (3 IQR) fences marked."""
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        near_lo, near_hi = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        far_lo, far_hi = (q1 - 3 * iqr, q3 + 3 * iqr)
        inl = v[(v >= near_lo) & (v <= near_hi)]
        ax.add_patch(plt.Rectangle((q1, y - h / 2), q3 - q1, h, facecolor=colour, alpha=0.55, edgecolor='0.3'))
        ax.plot([q2, q2], [y - h / 2, y + h / 2], color='w', lw=2)
        if len(inl):
            ax.plot([inl.min(), inl.max()], [y, y], color='0.3', lw=1, zorder=0)
        near = v[(v > near_hi) & (v <= far_hi) | (v < near_lo) & (v >= far_lo)]
        far = v[(v > far_hi) | (v < far_lo)]
        ax.plot(near, [y] * len(near), 'o', ms=3.2, color='k', mfc='k')
        ax.plot(far, [y] * len(far), 'o', ms=4.0, color='k', mfc='none')
        for x_ in (near_hi, far_hi):
            check_cancelled(_acid_cancel)
            ax.axvline(x_, color='0.55', ls='--', lw=0.9)
        return dict(n=len(v), n_near=len(near), n_far=len(far), near_hi=near_hi, far_hi=far_hi)

    def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
        a, b = (np.asarray(a, float), np.asarray(b, float))
        if not len(a) or not len(b):
            return np.nan
        gt = sum((np.sum(x > b) for x in a))
        lt = sum((np.sum(x < b) for x in a))
        return (gt - lt) / (len(a) * len(b))

    def _kde_mode(v: np.ndarray) -> float:
        if len(v) < FIG_MIN_N:
            return np.nan
        k = gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), 400)
        return float(xs[np.argmax(k(xs))])

    def figure_set(name: str, arms: list, save_prefix: str, frame: pd.DataFrame, arm_col: str='arm') -> dict:
        D = frame
        if not len(D):
            print(f'[skip] {name}: no events')
            return {}
        order = [lab for lab, _ in arms if lab in set(D[arm_col])]
        pal = dict(zip(order, [f'C{i}' for i in range(len(order))]))
        _figure_style()
        if sns is not None:
            g = _acid_joint_grid(height=7, ratio=4)
            for lab in order:
                check_cancelled(_acid_cancel)
                s = D[D[arm_col] == lab]
                x, y = (np.log10(s['dwell_ms'].clip(lower=0.001)), s[AMP_VAR])
                if len(s) >= FIG_MIN_N:
                    sns.kdeplot(x=x, y=y, ax=g.ax_joint, levels=8, color=pal[lab], linewidths=1.2)
                    sns.kdeplot(x=x, ax=g.ax_marg_x, color=pal[lab], fill=True, alpha=0.15)
                    sns.kdeplot(y=y, ax=g.ax_marg_y, color=pal[lab], fill=True, alpha=0.15)
                else:
                    g.ax_joint.scatter(x, y, s=14, color=pal[lab])
            g.ax_joint.set_xlabel('log$_{10}$ dwell (ms)')
            g.ax_joint.set_ylabel(AMP_LABEL)
            g.ax_joint.legend(handles=[plt.Line2D([], [], color=pal[l], lw=2, label=f'{l} (n={int((D[arm_col] == l).sum())})') for l in order], fontsize=9)
            g.figure.suptitle(f'{name} — joint kernel density', y=1.02, fontsize=15, fontweight='bold')
            if SAVE_FIGURES:
                g.figure.savefig(FIG_DIR / f'{save_prefix}_joint.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.show()
        else:
            print('seaborn not available: joint KDE skipped')
        for var, lab_, logx, fn in ((AMP_VAR, AMP_LABEL, False, 'ampkde'), ('dwell_ms', 'dwell (ms)', True, 'dwellkde')):
            check_cancelled(_acid_cancel)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            handles = []
            for lab in order:
                check_cancelled(_acid_cancel)
                v = D.loc[D[arm_col] == lab, var].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0 if logx else True)]
                if len(v) < FIG_MIN_N:
                    continue
                d = np.log10(v) if logx else v
                k = gaussian_kde(d)
                xs = np.linspace(d.min() - 0.3 * d.std(), d.max() + 0.3 * d.std(), 500)
                ax.plot(10 ** xs if logx else xs, k(xs), color=pal[lab], lw=2.2, label=f'{lab} (n={len(v)})')
                m = float(np.mean(v))
                ax.axvline(m, color=pal[lab], ls=':', lw=2.4)
                handles.append(plt.Line2D([], [], color=pal[lab], ls=':', lw=2.4, label=f'{lab}: {m:.3g}'))
            if logx:
                ax.set_xscale('log')
            ax.set_xlabel(lab_)
            ax.set_ylabel('kernel density' + (' (per log$_{10}$)' if logx else ''))
            ax.set_title(f"{('Dwell time' if logx else 'Amplitude')} — {name}")
            l1 = ax.legend(fontsize=9, loc='upper right')
            ax.add_artist(l1)
            if handles:
                ax.legend(handles=handles, title='mean (dotted)', fontsize=8, loc='center right')
            ax.grid(alpha=0.3)
            finish_figure(f'{save_prefix}_{fn}')
        fences = {}
        for var, lab_, logx, fn in ((AMP_VAR, AMP_LABEL, False, 'ampbox'), ('dwell_ms', 'dwell (ms)', True, 'dwellbox')):
            check_cancelled(_acid_cancel)
            fig, ax = plt.subplots(figsize=(11, 1.0 + 0.85 * len(order)))
            for i, lab in enumerate(order):
                check_cancelled(_acid_cancel)
                v = D.loc[D[arm_col] == lab, var].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0)]
                r = _mma_box(ax, np.log10(v) if logx else v, i, pal[lab])
                fences[var, lab] = r
                if r:
                    ax.text(1.005, i, f"n={r['n']}, near {r['n_near']}, far {r['n_far']}", transform=ax.get_yaxis_transform(), va='center', fontsize=8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=10, fontweight='bold')
            if logx:
                tk = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
                ax.set_xticks(np.log10(tk))
                ax.set_xticklabels(tk)
            ax.set_xlabel(lab_)
            ax.set_title(f"{('Dwell time' if logx else 'Amplitude')} — {name}")
            ax.margins(x=0.06)
            finish_figure(f'{save_prefix}_{fn}')
        rows = []
        for lab in order:
            check_cancelled(_acid_cancel)
            s = D[D[arm_col] == lab]
            a = s[AMP_VAR].to_numpy(float)
            a = a[np.isfinite(a)]
            d = np.log10(s['dwell_ms'].clip(lower=0.001).to_numpy(float))
            rows.append({'arm': lab, 'n': len(s), 'amp_mean': a.mean() if len(a) else np.nan, 'amp_median': np.median(a) if len(a) else np.nan, 'amp_iqr': np.percentile(a, 75) - np.percentile(a, 25) if len(a) else np.nan, 'amp_kde_mode': _kde_mode(a), 'dwell_median_ms': float(10 ** np.median(d)), 'dwell_iqr_log10': float(np.percentile(d, 75) - np.percentile(d, 25)), 'dwell_kde_mode_ms': float(10 ** _kde_mode(d)) if len(d) >= FIG_MIN_N else np.nan, 'amp_far_outliers': (fences.get((AMP_VAR, lab)) or {}).get('n_far', np.nan), 'dwell_far_outliers': (fences.get(('dwell_ms', lab)) or {}).get('n_far', np.nan)})
        summary = pd.DataFrame(rows)
        tests = []
        for i in range(len(order)):
            check_cancelled(_acid_cancel)
            for j in range(i + 1, len(order)):
                check_cancelled(_acid_cancel)
                a1, a2 = (D.loc[D[arm_col] == order[i], AMP_VAR].dropna().to_numpy(float), D.loc[D[arm_col] == order[j], AMP_VAR].dropna().to_numpy(float))
                d1, d2 = (np.log10(D.loc[D[arm_col] == order[i], 'dwell_ms'].clip(lower=0.001)), np.log10(D.loc[D[arm_col] == order[j], 'dwell_ms'].clip(lower=0.001)))
                if len(a1) < 5 or len(a2) < 5:
                    continue
                tests.append({'arm_a': order[i], 'arm_b': order[j], 'amp_mannwhitney_p': _mwu(a1, a2, alternative='two-sided').pvalue, 'amp_ks': _ks(a1, a2).statistic, 'amp_cliffs_delta': _cliffs_delta(a1, a2), 'dwell_mannwhitney_p': _mwu(d1, d2, alternative='two-sided').pvalue, 'dwell_ks': _ks(d1, d2).statistic, 'dwell_cliffs_delta': _cliffs_delta(d1, d2)})
        tests = pd.DataFrame(tests)
        print(f'{name}: arms {order}')
        display(summary.round(4))
        if len(tests):
            print("Pairwise tests (Cliff's delta: |0.15| small, |0.33| medium, |0.47| large):")
            display(tests.round(4))
        summary.to_csv(OUT_DIR / f'{save_prefix}_summary.csv', index=False)
        tests.to_csv(OUT_DIR / f'{save_prefix}_pairwise.csv', index=False)
        return {'summary': summary.to_dict(orient='records'), 'pairwise': tests.to_dict(orient='records')}
    figure_set_results = {}
    if len(_available) >= 2:
        for _name, _arms in FIGURE_GROUPS.items():
            check_cancelled(_acid_cancel)
            parts = []
            for lab, tags in _arms:
                check_cancelled(_acid_cancel)
                sub = EVENTS_ALL[EVENTS_ALL['tag'].isin(tags)].copy()
                if len(sub):
                    sub['arm'] = lab
                    parts.append(sub)
            D = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            pfx = 'figset_' + ''.join((ch if ch.isalnum() else '_' for ch in _name))[:34]
            figure_set_results[_name] = figure_set(_name, _arms, pfx, D)
    else:
        print(f'Only {len(_available)} tag available: drawing the figure set with the families of {TAG} as arms.')
    _fam_arms = [(family_names[k][:28], [k]) for k in range(K) if (_fam['family'] == k).sum() >= 5]
    D_fam = _fam.copy()
    D_fam['area_frac'] = D_fam['area_pA_ms'] / D_fam['local_I0_pA']
    if FAMILY_EXCLUDE_BURSTS:
        D_fam = D_fam.loc[~D_fam['in_burst']]
    D_fam['arm'] = D_fam['family'].map({k: family_names[k][:28] for k in range(K)})
    figure_set_results[f'{TAG} families'] = figure_set(f'{TAG} by event family', _fam_arms, f'figset_{TAG}_families', D_fam)
    (OUT_DIR / f'{TAG}_figure_set_stats.json').write_text(json.dumps(figure_set_results, indent=2, default=str))
    print(f'Amplitude variable = {AMP_VAR}; set AMP_VAR in the configuration cell for the pA version.')


def definitions_036():
    global _cliffs_delta, _figure_style, _kde_mode, _ks, _mma_box, _mwu, figure_set, gaussian_kde, load_tag_events, sns
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    from scipy.stats import gaussian_kde, mannwhitneyu as _mwu, ks_2samp as _ks

    def _figure_style():
        plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 12, 'legend.fontsize': 9})

    def load_tag_events(tag: str) -> pd.DataFrame:
        """Primary events of a tag with family labels: from this kernel when the family cell has run for this
    very tag, otherwise from the tag's output folder, using the primary set recorded in its family summary
    (the pre-clustering catalog), so every tag enters with its own primary definition."""
        if tag == TAG and globals().get('_FAM_TAG') == tag:
            frame = _fam.copy()
        else:
            folder = BASE_DIR / f'{tag}_subpopulation'
            fam_json = folder / f'{tag}_event_families.json'
            fam_path = folder / f'{tag}_event_families.parquet'
            primary_set = None
            if fam_json.is_file():
                try:
                    primary_set = json.loads(fam_json.read_text()).get('primary_set')
                except Exception as exc:
                    print(f'[{tag}] could not read {fam_json.name}: {exc}')
            cat_path = folder / f'{tag}_{primary_set}_catalog_preclustering.parquet' if primary_set else None
            if cat_path is not None and cat_path.is_file() and fam_path.is_file():
                frame = pd.read_parquet(cat_path).merge(pd.read_parquet(fam_path), on='event_id', how='inner')
            else:
                cat_path = folder / f'{tag}_cusum_catalog.parquet'
                if not cat_path.is_file():
                    return pd.DataFrame()
                frame = pd.read_parquet(cat_path)
                frame = frame.loc[frame['accepted'] & (frame.get('polarity', 1) == 1)].copy()
                if fam_path.is_file():
                    frame = frame.merge(pd.read_parquet(fam_path), on='event_id', how='inner')
                else:
                    frame['family'] = -1
                    frame['in_burst'] = False
        frame['tag'] = tag
        if 'area_frac' not in frame:
            frame['area_frac'] = frame['area_pA_ms'] / frame['local_I0_pA']
        if FAMILY_EXCLUDE_BURSTS and 'in_burst' in frame:
            frame = frame.loc[~frame['in_burst'].fillna(False).astype(bool)]
        return frame

    def _mma_box(ax, vals, y, colour, h=0.34):
        """Box + whiskers with Tukey near (1.5 IQR) and far (3 IQR) fences marked."""
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        near_lo, near_hi = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        far_lo, far_hi = (q1 - 3 * iqr, q3 + 3 * iqr)
        inl = v[(v >= near_lo) & (v <= near_hi)]
        ax.add_patch(plt.Rectangle((q1, y - h / 2), q3 - q1, h, facecolor=colour, alpha=0.55, edgecolor='0.3'))
        ax.plot([q2, q2], [y - h / 2, y + h / 2], color='w', lw=2)
        if len(inl):
            ax.plot([inl.min(), inl.max()], [y, y], color='0.3', lw=1, zorder=0)
        near = v[(v > near_hi) & (v <= far_hi) | (v < near_lo) & (v >= far_lo)]
        far = v[(v > far_hi) | (v < far_lo)]
        ax.plot(near, [y] * len(near), 'o', ms=3.2, color='k', mfc='k')
        ax.plot(far, [y] * len(far), 'o', ms=4.0, color='k', mfc='none')
        for x_ in (near_hi, far_hi):
            check_cancelled(_acid_cancel)
            ax.axvline(x_, color='0.55', ls='--', lw=0.9)
        return dict(n=len(v), n_near=len(near), n_far=len(far), near_hi=near_hi, far_hi=far_hi)

    def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
        a, b = (np.asarray(a, float), np.asarray(b, float))
        if not len(a) or not len(b):
            return np.nan
        gt = sum((np.sum(x > b) for x in a))
        lt = sum((np.sum(x < b) for x in a))
        return (gt - lt) / (len(a) * len(b))

    def _kde_mode(v: np.ndarray) -> float:
        if len(v) < FIG_MIN_N:
            return np.nan
        k = gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), 400)
        return float(xs[np.argmax(k(xs))])

    def figure_set(name: str, arms: list, save_prefix: str, frame: pd.DataFrame, arm_col: str='arm') -> dict:
        D = frame
        if not len(D):
            print(f'[skip] {name}: no events')
            return {}
        order = [lab for lab, _ in arms if lab in set(D[arm_col])]
        pal = dict(zip(order, [f'C{i}' for i in range(len(order))]))
        _figure_style()
        if sns is not None:
            g = _acid_joint_grid(height=7, ratio=4)
            for lab in order:
                check_cancelled(_acid_cancel)
                s = D[D[arm_col] == lab]
                x, y = (np.log10(s['dwell_ms'].clip(lower=0.001)), s[AMP_VAR])
                if len(s) >= FIG_MIN_N:
                    sns.kdeplot(x=x, y=y, ax=g.ax_joint, levels=8, color=pal[lab], linewidths=1.2)
                    sns.kdeplot(x=x, ax=g.ax_marg_x, color=pal[lab], fill=True, alpha=0.15)
                    sns.kdeplot(y=y, ax=g.ax_marg_y, color=pal[lab], fill=True, alpha=0.15)
                else:
                    g.ax_joint.scatter(x, y, s=14, color=pal[lab])
            g.ax_joint.set_xlabel('log$_{10}$ dwell (ms)')
            g.ax_joint.set_ylabel(AMP_LABEL)
            g.ax_joint.legend(handles=[plt.Line2D([], [], color=pal[l], lw=2, label=f'{l} (n={int((D[arm_col] == l).sum())})') for l in order], fontsize=9)
            g.figure.suptitle(f'{name} — joint kernel density', y=1.02, fontsize=15, fontweight='bold')
            if SAVE_FIGURES:
                g.figure.savefig(FIG_DIR / f'{save_prefix}_joint.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.show()
        else:
            print('seaborn not available: joint KDE skipped')
        for var, lab_, logx, fn in ((AMP_VAR, AMP_LABEL, False, 'ampkde'), ('dwell_ms', 'dwell (ms)', True, 'dwellkde')):
            check_cancelled(_acid_cancel)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            handles = []
            for lab in order:
                check_cancelled(_acid_cancel)
                v = D.loc[D[arm_col] == lab, var].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0 if logx else True)]
                if len(v) < FIG_MIN_N:
                    continue
                d = np.log10(v) if logx else v
                k = gaussian_kde(d)
                xs = np.linspace(d.min() - 0.3 * d.std(), d.max() + 0.3 * d.std(), 500)
                ax.plot(10 ** xs if logx else xs, k(xs), color=pal[lab], lw=2.2, label=f'{lab} (n={len(v)})')
                m = float(np.mean(v))
                ax.axvline(m, color=pal[lab], ls=':', lw=2.4)
                handles.append(plt.Line2D([], [], color=pal[lab], ls=':', lw=2.4, label=f'{lab}: {m:.3g}'))
            if logx:
                ax.set_xscale('log')
            ax.set_xlabel(lab_)
            ax.set_ylabel('kernel density' + (' (per log$_{10}$)' if logx else ''))
            ax.set_title(f"{('Dwell time' if logx else 'Amplitude')} — {name}")
            l1 = ax.legend(fontsize=9, loc='upper right')
            ax.add_artist(l1)
            if handles:
                ax.legend(handles=handles, title='mean (dotted)', fontsize=8, loc='center right')
            ax.grid(alpha=0.3)
            finish_figure(f'{save_prefix}_{fn}')
        fences = {}
        for var, lab_, logx, fn in ((AMP_VAR, AMP_LABEL, False, 'ampbox'), ('dwell_ms', 'dwell (ms)', True, 'dwellbox')):
            check_cancelled(_acid_cancel)
            fig, ax = plt.subplots(figsize=(11, 1.0 + 0.85 * len(order)))
            for i, lab in enumerate(order):
                check_cancelled(_acid_cancel)
                v = D.loc[D[arm_col] == lab, var].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0)]
                r = _mma_box(ax, np.log10(v) if logx else v, i, pal[lab])
                fences[var, lab] = r
                if r:
                    ax.text(1.005, i, f"n={r['n']}, near {r['n_near']}, far {r['n_far']}", transform=ax.get_yaxis_transform(), va='center', fontsize=8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=10, fontweight='bold')
            if logx:
                tk = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
                ax.set_xticks(np.log10(tk))
                ax.set_xticklabels(tk)
            ax.set_xlabel(lab_)
            ax.set_title(f"{('Dwell time' if logx else 'Amplitude')} — {name}")
            ax.margins(x=0.06)
            finish_figure(f'{save_prefix}_{fn}')
        rows = []
        for lab in order:
            check_cancelled(_acid_cancel)
            s = D[D[arm_col] == lab]
            a = s[AMP_VAR].to_numpy(float)
            a = a[np.isfinite(a)]
            d = np.log10(s['dwell_ms'].clip(lower=0.001).to_numpy(float))
            rows.append({'arm': lab, 'n': len(s), 'amp_mean': a.mean() if len(a) else np.nan, 'amp_median': np.median(a) if len(a) else np.nan, 'amp_iqr': np.percentile(a, 75) - np.percentile(a, 25) if len(a) else np.nan, 'amp_kde_mode': _kde_mode(a), 'dwell_median_ms': float(10 ** np.median(d)), 'dwell_iqr_log10': float(np.percentile(d, 75) - np.percentile(d, 25)), 'dwell_kde_mode_ms': float(10 ** _kde_mode(d)) if len(d) >= FIG_MIN_N else np.nan, 'amp_far_outliers': (fences.get((AMP_VAR, lab)) or {}).get('n_far', np.nan), 'dwell_far_outliers': (fences.get(('dwell_ms', lab)) or {}).get('n_far', np.nan)})
        summary = pd.DataFrame(rows)
        tests = []
        for i in range(len(order)):
            check_cancelled(_acid_cancel)
            for j in range(i + 1, len(order)):
                check_cancelled(_acid_cancel)
                a1, a2 = (D.loc[D[arm_col] == order[i], AMP_VAR].dropna().to_numpy(float), D.loc[D[arm_col] == order[j], AMP_VAR].dropna().to_numpy(float))
                d1, d2 = (np.log10(D.loc[D[arm_col] == order[i], 'dwell_ms'].clip(lower=0.001)), np.log10(D.loc[D[arm_col] == order[j], 'dwell_ms'].clip(lower=0.001)))
                if len(a1) < 5 or len(a2) < 5:
                    continue
                tests.append({'arm_a': order[i], 'arm_b': order[j], 'amp_mannwhitney_p': _mwu(a1, a2, alternative='two-sided').pvalue, 'amp_ks': _ks(a1, a2).statistic, 'amp_cliffs_delta': _cliffs_delta(a1, a2), 'dwell_mannwhitney_p': _mwu(d1, d2, alternative='two-sided').pvalue, 'dwell_ks': _ks(d1, d2).statistic, 'dwell_cliffs_delta': _cliffs_delta(d1, d2)})
        tests = pd.DataFrame(tests)
        print(f'{name}: arms {order}')
        display(summary.round(4))
        if len(tests):
            print("Pairwise tests (Cliff's delta: |0.15| small, |0.33| medium, |0.47| large):")
            display(tests.round(4))
        summary.to_csv(OUT_DIR / f'{save_prefix}_summary.csv', index=False)
        tests.to_csv(OUT_DIR / f'{save_prefix}_pairwise.csv', index=False)
        return {'summary': summary.to_dict(orient='records'), 'pairwise': tests.to_dict(orient='records')}


# Original notebook cell 38
def stage_038():
    global _family_files, _hours_from_tag, _p, _re, _ref_summary, _summaries, _taus, ax2, ax3, axes, c, exc, f, fig, incubation_table, k, lab, name, row, rows, s, share_cols
    check_cancelled(_acid_cancel)
    import re as _re
    _family_files = sorted(BASE_DIR.glob('*_subpopulation/*_event_families.json'))
    _summaries = []
    for _p in _family_files:
        check_cancelled(_acid_cancel)
        try:
            _summaries.append(json.loads(_p.read_text()))
        except Exception as exc:
            print(f'could not read {_p.name}: {exc}')

    def _hours_from_tag(tag: str) -> float:
        return float(_acid_hours[tag])
    if len(_summaries) < 2:
        print(f'Cross-tag comparison needs at least two tags; found {len(_summaries)} family summary file(s) under {BASE_DIR}. Run the notebook with TAG set to each incubation time (the reference model is reused automatically).')
    else:
        rows = []
        for s in _summaries:
            check_cancelled(_acid_cancel)
            row = {'tag': s['tag'], 'hours': _hours_from_tag(s['tag']), 'n_events': s['n_events'], 'rate_per_s': s['rate_per_s'], 'I0_pA': s.get('global_i0_pA'), 'noise_pA': s.get('global_noise_pA'), 'resolved': s['markers']['resolved_fraction'], 'precursor': s['markers']['precursor_fraction'], 'flat_no_precursor': s['markers']['flat_no_precursor_fraction'], 'flat_no_precursor_sharp': s['markers'].get('flat_no_precursor_sharp_fraction'), 'deep_long': s['markers'].get('deep_long_fraction'), 'dwell_median_ms': s['markers']['dwell_quantiles_ms']['0.5'], 'peak_frac_median': s['markers']['peak_frac_quantiles']['0.5'], 'resistive_independent': s.get('resistive_independent_events'), 'floor_0.5ms_pA': s['detector']['acceptance_floor_pA']['0.5'], 'floor_5ms_pA': s['detector']['acceptance_floor_pA']['5.0'], 'reference_tag': s['reference_tag'], 'model_fingerprint': s.get('family_model_fingerprint', 'n/a'), 'instrument_tau_ms': s.get('instrument_tau_ms'), 'tau_measured_ms': s.get('instrument_tau_measured_ms'), 'matched_bandwidth_hz': s.get('matched_bandwidth_hz'), 'raw_rolloff_hz': s.get('raw_rolloff_hz')}
            for f in s['family_table']:
                check_cancelled(_acid_cancel)
                row[f"F{f['family']}_share"] = f['share']
                row[f"F{f['family']}_soft"] = f['soft_share']
            rows.append(row)
        incubation_table = pd.DataFrame(rows).sort_values('hours').reset_index(drop=True)
        incubation_table.to_csv(OUT_DIR / 'incubation_comparison.csv', index=False)
        if incubation_table['model_fingerprint'].nunique() > 1:
            print(f"!! tags were assigned with DIFFERENT versions of the family model (fingerprints {incubation_table['model_fingerprint'].unique().tolist()}); rerun the older tags so that all share one model before comparing family shares")
        _taus = incubation_table['instrument_tau_ms'].dropna()
        if len(_taus) >= 2 and _taus.max() / max(_taus.min(), 1e-09) > 1.25:
            print(f"!! recordings differ in effective bandwidth (instrument recovery constants {dict(zip(incubation_table['tag'], incubation_table['instrument_tau_ms'].round(3)))}); dwell, peak and precursor visibility are not comparable across them until stage 0 matches the bandwidth (MATCH_TAU_MS)")
        _ref_summary = ([s for s in _summaries if s.get('tag') == FAMILY_REFERENCE_TAG] or [s for s in _summaries if s.get('tag') == s.get('reference_tag')] or _summaries)[0]
        print(f"Family names (from the reference tag {_ref_summary.get('tag')}):")
        for k, name in _ref_summary['family_names'].items():
            check_cancelled(_acid_cancel)
            print(f'  {name}')
        display(incubation_table.round(4))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        share_cols = sorted((c for c in incubation_table if c.endswith('_soft')))
        for c in share_cols:
            check_cancelled(_acid_cancel)
            axes[0].plot(incubation_table['hours'], incubation_table[c], marker='o', label=c.replace('_soft', ''))
        axes[0].set_xlabel('Incubation time (h)')
        axes[0].set_ylabel('Family share (soft)')
        axes[0].set_title('Family composition vs incubation')
        axes[0].legend(fontsize=8)
        for c, lab in (('precursor', 'precursor present'), ('flat_no_precursor_sharp', 'flat, no precursor, sharp edges'), ('deep_long', 'deep-long (>= 30% I0)'), ('resolved', 'resolved (dwell > 3 tau)')):
            check_cancelled(_acid_cancel)
            axes[1].plot(incubation_table['hours'], incubation_table[c], marker='o', label=lab)
        axes[1].set_xlabel('Incubation time (h)')
        axes[1].set_ylabel('Fraction of events')
        axes[1].set_title('Physical markers vs incubation')
        axes[1].legend(fontsize=8)
        ax2 = axes[2]
        ax2.plot(incubation_table['hours'], incubation_table['rate_per_s'], marker='o', color='tab:blue', label='event rate (/s)')
        ax2.set_xlabel('Incubation time (h)')
        ax2.set_ylabel('Event rate (/s)', color='tab:blue')
        ax3 = ax2.twinx()
        ax3.plot(incubation_table['hours'], incubation_table['peak_frac_median'], marker='s', color='tab:orange', label='median peak / I0')
        ax3.set_ylabel('Median peak / I0', color='tab:orange')
        ax2.set_title('Rate and amplitude vs incubation')
        finish_figure('15_incubation_comparison')
        print('Read each change against the floors: acceptance floor at 0.5 ms per tag =', dict(zip(incubation_table['tag'], incubation_table['floor_0.5ms_pA'].round(1))), '; I0 per tag =', dict(zip(incubation_table['tag'], incubation_table['I0_pA'].round(0) if incubation_table['I0_pA'].notna().all() else incubation_table['I0_pA'])))


def definitions_038():
    global _hours_from_tag, _re
    import re as _re

    def _hours_from_tag(tag: str) -> float:
        return float(_acid_hours[tag])


# Original notebook cell 40
def stage_040():
    check_cancelled(_acid_cancel)
    if len(primary_catalog) < MIN_ANALYSIS_EVENTS:
        raise RuntimeError(f"Primary set '{PRIMARY_SET_NAME}' has {len(primary_catalog)} events; the substructure sections need >= {MIN_ANALYSIS_EVENTS}. The family summary, overlay, figure set and incubation comparison for {TAG} are already written to {OUT_DIR}.")
    print(f'{len(primary_catalog)} primary events: substructure sections enabled.')


# Original notebook cell 41
def stage_041():
    global _, _blk, _bx, _lv_end, _lv_mean, _lv_start, _sel, ax, ax2, center, center_row, e, fig, hi, lo, m, pad, row, s, t
    check_cancelled(_acid_cancel)
    center_row = primary_catalog.iloc[len(primary_catalog) // 2]
    center = int(center_row.start_idx)
    pad = int(round(0.02 * fs))
    lo, hi = (max(0, center - pad), min(n_samples, center + pad))
    t = (np.arange(lo, hi) - center) / fs * 1000.0
    _lv_start, _lv_end, _lv_mean = primary_cusum_info['levels']
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, analysis_signal[lo:hi], linewidth=0.7, color='tab:blue', label='detrended current (primary polarity down)')
    _blk = primary_cusum_info['block_len']
    _bx = (np.arange(len(primary_cusum_info['mu_blocks'])) + 0.5) * _blk
    ax.plot(t, np.interp(np.arange(lo, hi), _bx, primary_cusum_info['mu_blocks']), linestyle='--', linewidth=0.9, color='gray', label='block baseline')
    for _, row in cusum_catalog.loc[cusum_catalog['accepted']].iterrows():
        check_cancelled(_acid_cancel)
        if row.end_idx < lo or row.start_idx >= hi:
            continue
        ax.axvspan((row.start_idx - center) / fs * 1000.0, (row.end_idx + 1 - center) / fs * 1000.0, alpha=0.16, color='tab:orange')
    _sel = (_lv_end >= lo) & (_lv_start < hi)
    for s, e, m in zip(_lv_start[_sel], _lv_end[_sel], _lv_mean[_sel]):
        check_cancelled(_acid_cancel)
        ax.hlines(m, (s - center) / fs * 1000.0, (e + 1 - center) / fs * 1000.0, colors='tab:red', linewidth=2.0)
    for _, row in threshold_catalog.loc[threshold_catalog['accepted']].iterrows():
        check_cancelled(_acid_cancel)
        if row.end_idx < lo or row.start_idx >= hi:
            continue
        ax.axvline((row.start_idx - center) / fs * 1000.0, linestyle=':', linewidth=0.8, color='k')
    ax.set_xlabel('Time from representative event onset (ms)')
    ax.set_ylabel('Oriented detrended current (pA)')
    ax.set_title('CUSUM regions and levels (' + ', '.join((f'{k}: delta={v[0]:g}, h={v[1]:g}' for k, v in cusum_selected.items())) + ') with threshold onsets')
    ax.legend(fontsize=8, loc='lower left')
    if RUN_STUDENT:
        ax2 = ax.twinx()
        ax2.plot(t, np.asarray(student_probability[lo:hi]), linewidth=0.9, alpha=0.65, color='tab:green', label='student probability')
        ax2.set_ylabel('Student probability')
        ax2.set_ylim(-0.02, 1.02)
    finish_figure('01_detector_overlay')


# Original notebook cell 42
def stage_042():
    global _prim_acc, amp_col, ax, axes, fig, group, label, matched
    check_cancelled(_acid_cancel)
    _prim_acc = detector_catalogs[PRIMARY_DETECTOR].loc[detector_catalogs[PRIMARY_DETECTOR]['accepted']].copy()
    amp_col = AMP_FRAC_COL if _prim_acc[AMP_FRAC_COL].notna().mean() > 0.8 else AMP_SIGMA_COL
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for matched, label in [(True, 'matched to threshold'), (False, f'{PRIMARY_DETECTOR}-only')]:
        check_cancelled(_acid_cancel)
        group = _prim_acc.loc[_prim_acc['detector_match'] == matched]
        if len(group):
            axes[0].hist(group['dwell_ms'], bins=np.geomspace(_prim_acc['dwell_ms'].min(), _prim_acc['dwell_ms'].max(), 50), histtype='step', density=True, linewidth=1.6, label=f'{label} (n={len(group)})')
            axes[1].hist(group[amp_col].dropna(), bins=50, histtype='step', density=True, linewidth=1.6, label=f'{label} (n={len(group)})')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Dwell time (ms)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Detector agreement versus dwell')
    axes[1].set_xlabel(amp_col.replace('_', ' '))
    axes[1].set_ylabel('Density')
    axes[1].set_title('Detector agreement versus amplitude')
    for ax in axes:
        check_cancelled(_acid_cancel)
        ax.legend(fontsize=8)
    finish_figure('02_matched_vs_unmatched')


# Original notebook cell 44
def stage_044():
    global drift_metric, drift_table, summarize_time_blocks, time_summary
    check_cancelled(_acid_cancel)

    def summarize_time_blocks(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        block_duration = duration_s / N_TIME_BLOCKS
        for block in range(N_TIME_BLOCKS):
            check_cancelled(_acid_cancel)
            g = frame.loc[frame['time_block'] == block]
            rows.append({'time_block': block, 'mid_time_s': (block + 0.5) * block_duration, 'n_events': len(g), 'rate_per_s': len(g) / block_duration, 'dwell_median_ms': g['dwell_ms'].median(), 'amplitude_sigma_median': g[AMP_SIGMA_COL].median(), 'amplitude_fracchange_median': g[AMP_FRAC_COL].median(), 'local_I0_median_pA': g['local_I0_pA'].median(), 'local_noise_median_pA': g['local_noise_mad_pA'].median()})
        return pd.DataFrame(rows)

    def drift_metric(frame: pd.DataFrame, column: str) -> dict:
        valid = frame[['time_block', 'onset_s', column]].dropna()
        block_medians = valid.groupby('time_block', as_index=False).agg(onset_s=('onset_s', 'median'), value=(column, 'median')).dropna()
        if len(block_medians) < 3 or block_medians['value'].nunique() < 2:
            return {'column': column, 'n_events': len(valid), 'n_blocks': len(block_medians), 'total_change': np.nan, 'change_over_iqr': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan}
        slope = theilslopes(block_medians['value'], block_medians['onset_s'])[0]
        total_change = float(slope * duration_s)
        iqr = float(valid[column].quantile(0.75) - valid[column].quantile(0.25))
        rho, p = spearmanr(block_medians['onset_s'], block_medians['value'])
        return {'column': column, 'n_events': len(valid), 'n_blocks': len(block_medians), 'total_change': total_change, 'change_over_iqr': total_change / max(abs(iqr), 1e-12), 'spearman_rho': float(rho), 'spearman_p': float(p)}
    time_summary = summarize_time_blocks(primary_catalog)
    drift_table = pd.DataFrame([drift_metric(primary_catalog, col) for col in [AMP_FRAC_COL, AMP_SIGMA_COL, 'dwell_ms', 'local_I0_pA', 'local_noise_mad_pA', 'local_rate_hz']])
    time_summary.to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_time_blocks.csv', index=False)
    drift_table.to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_drift_metrics.csv', index=False)
    display(time_summary.round(4))
    display(drift_table.round(4))


def definitions_044():
    global drift_metric, summarize_time_blocks

    def summarize_time_blocks(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        block_duration = duration_s / N_TIME_BLOCKS
        for block in range(N_TIME_BLOCKS):
            check_cancelled(_acid_cancel)
            g = frame.loc[frame['time_block'] == block]
            rows.append({'time_block': block, 'mid_time_s': (block + 0.5) * block_duration, 'n_events': len(g), 'rate_per_s': len(g) / block_duration, 'dwell_median_ms': g['dwell_ms'].median(), 'amplitude_sigma_median': g[AMP_SIGMA_COL].median(), 'amplitude_fracchange_median': g[AMP_FRAC_COL].median(), 'local_I0_median_pA': g['local_I0_pA'].median(), 'local_noise_median_pA': g['local_noise_mad_pA'].median()})
        return pd.DataFrame(rows)

    def drift_metric(frame: pd.DataFrame, column: str) -> dict:
        valid = frame[['time_block', 'onset_s', column]].dropna()
        block_medians = valid.groupby('time_block', as_index=False).agg(onset_s=('onset_s', 'median'), value=(column, 'median')).dropna()
        if len(block_medians) < 3 or block_medians['value'].nunique() < 2:
            return {'column': column, 'n_events': len(valid), 'n_blocks': len(block_medians), 'total_change': np.nan, 'change_over_iqr': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan}
        slope = theilslopes(block_medians['value'], block_medians['onset_s'])[0]
        total_change = float(slope * duration_s)
        iqr = float(valid[column].quantile(0.75) - valid[column].quantile(0.25))
        rho, p = spearmanr(block_medians['onset_s'], block_medians['value'])
        return {'column': column, 'n_events': len(valid), 'n_blocks': len(block_medians), 'total_change': total_change, 'change_over_iqr': total_change / max(abs(iqr), 1e-12), 'spearman_rho': float(rho), 'spearman_p': float(p)}


# Original notebook cell 45
def stage_045():
    global ax, axes, column, fig, plots, ylabel
    check_cancelled(_acid_cancel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    plots = [('rate_per_s', 'Event rate (s⁻¹)'), ('local_I0_median_pA', 'Median local |I0| (pA)'), ('local_noise_median_pA', 'Median local MAD noise (pA)'), ('amplitude_fracchange_median', f'Median {AMPLITUDE_STATISTIC} |ΔI|/|I0|')]
    for ax, (column, ylabel) in zip(axes, plots):
        check_cancelled(_acid_cancel)
        ax.plot(time_summary['mid_time_s'], time_summary[column], marker='o')
        ax.set_xlabel('Recording time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' by contiguous block')
    finish_figure('03_time_block_stability')


# Original notebook cell 47
def stage_047():
    global AMPLITUDE_TARGET, choose_amplitude_target, discovery_validation_residualize, make_nuisance_model, nuisance_covariates, primary_residualized
    check_cancelled(_acid_cancel)

    def choose_amplitude_target(frame: pd.DataFrame) -> str:
        """The configured amplitude statistic as a fraction of I0, or in noise units when I0 is unavailable."""
        frac_valid = frame[AMP_FRAC_COL].notna().mean()
        return AMP_FRAC_COL if frac_valid >= 0.8 else AMP_SIGMA_COL

    def make_nuisance_model() -> Pipeline:
        return Pipeline([('impute', SimpleImputer(strategy='median')), ('spline', SplineTransformer(n_knots=5, degree=2, include_bias=False)), ('scale', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-4, 4, 17)))])

    def discovery_validation_residualize(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
        """Strictly preserve alternating validation blocks during nuisance adjustment.

    Discovery events receive leave-one-discovery-block-out predictions. Validation
    events are predicted only by a final model fit on every discovery block.
    """
        work = frame.copy().reset_index(drop=True)
        work['split'] = np.where(work['time_block'].to_numpy() % 2 == 0, 'discovery', 'validation')
        discovery = work['split'].to_numpy() == 'discovery'
        validation = ~discovery
        candidate_covariates = ['log_dwell', 'onset_fraction', 'local_noise_mad_pA', 'log_local_rate_hz']
        if work.loc[discovery, 'local_I0_pA'].notna().mean() >= 0.8:
            candidate_covariates.append('local_I0_pA')
        covariates = [c for c in candidate_covariates if work.loc[discovery, c].notna().sum() >= 20 and work.loc[discovery, c].nunique(dropna=True) > 1]
        if not covariates:
            raise RuntimeError('No usable nuisance covariates remain after discovery-only checks')
        y = work[target].to_numpy(float)
        pred = np.full(len(work), np.nan, dtype=float)
        valid_y = np.isfinite(y)
        discovery_valid = discovery & valid_y
        validation_valid = validation & valid_y
        discovery_blocks = sorted(work.loc[discovery_valid, 'time_block'].unique())
        for block in discovery_blocks:
            check_cancelled(_acid_cancel)
            test = discovery_valid & (work['time_block'].to_numpy() == block)
            train = discovery_valid & ~test
            if train.sum() < 50 or test.sum() == 0:
                continue
            model = make_nuisance_model()
            model.fit(work.loc[train, covariates], y[train])
            pred[test] = model.predict(work.loc[test, covariates])
        final_discovery_model = make_nuisance_model()
        final_discovery_model.fit(work.loc[discovery_valid, covariates], y[discovery_valid])
        pred[validation_valid] = final_discovery_model.predict(work.loc[validation_valid, covariates])
        # Was: pred[missing_discovery] = final_discovery_model.predict(...). That model
        # was fit on ALL discovery_valid rows -- INCLUDING these, whose own block had
        # too few other discovery rows (train.sum() < 50 above) for a genuine held-out
        # fit. Predicting them with a model trained on themselves is in-sample, not
        # held-out, and biases their residual toward zero. Leave them NaN instead --
        # nanmedian/nanstd below already tolerate it, and cluster_feature_space's own
        # notna() completeness filter excludes them downstream, same as any other
        # non-finite feature.
        missing_discovery = discovery_valid & ~np.isfinite(pred)
        if missing_discovery.any():
            print(f'{target}: {int(missing_discovery.sum())} discovery events left unresidualized (their block had too few other discovery events for a held-out fit); excluded from residual-based screens.')
        residual = y - pred
        center = float(np.nanmedian(residual[discovery_valid]))
        scale = 1.4826 * float(np.nanmedian(np.abs(residual[discovery_valid] - center)))
        scale = scale if np.isfinite(scale) and scale > 0 else float(np.nanstd(residual[discovery_valid]) or 1.0)
        work['amplitude_target'] = target
        work['amplitude_predicted'] = pred
        work['amplitude_residual'] = residual
        work['amplitude_resid_z'] = (residual - center) / scale
        return (work, covariates)
    AMPLITUDE_TARGET = choose_amplitude_target(primary_catalog)
    primary_residualized, nuisance_covariates = discovery_validation_residualize(primary_catalog, AMPLITUDE_TARGET)
    print('Amplitude target:', AMPLITUDE_TARGET)
    print('Nuisance covariates:', nuisance_covariates)


def definitions_047():
    global choose_amplitude_target, discovery_validation_residualize, make_nuisance_model

    def choose_amplitude_target(frame: pd.DataFrame) -> str:
        """The configured amplitude statistic as a fraction of I0, or in noise units when I0 is unavailable."""
        frac_valid = frame[AMP_FRAC_COL].notna().mean()
        return AMP_FRAC_COL if frac_valid >= 0.8 else AMP_SIGMA_COL

    def make_nuisance_model() -> Pipeline:
        return Pipeline([('impute', SimpleImputer(strategy='median')), ('spline', SplineTransformer(n_knots=5, degree=2, include_bias=False)), ('scale', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-4, 4, 17)))])

    def discovery_validation_residualize(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
        """Strictly preserve alternating validation blocks during nuisance adjustment.

    Discovery events receive leave-one-discovery-block-out predictions. Validation
    events are predicted only by a final model fit on every discovery block.
    """
        work = frame.copy().reset_index(drop=True)
        work['split'] = np.where(work['time_block'].to_numpy() % 2 == 0, 'discovery', 'validation')
        discovery = work['split'].to_numpy() == 'discovery'
        validation = ~discovery
        candidate_covariates = ['log_dwell', 'onset_fraction', 'local_noise_mad_pA', 'log_local_rate_hz']
        if work.loc[discovery, 'local_I0_pA'].notna().mean() >= 0.8:
            candidate_covariates.append('local_I0_pA')
        covariates = [c for c in candidate_covariates if work.loc[discovery, c].notna().sum() >= 20 and work.loc[discovery, c].nunique(dropna=True) > 1]
        if not covariates:
            raise RuntimeError('No usable nuisance covariates remain after discovery-only checks')
        y = work[target].to_numpy(float)
        pred = np.full(len(work), np.nan, dtype=float)
        valid_y = np.isfinite(y)
        discovery_valid = discovery & valid_y
        validation_valid = validation & valid_y
        discovery_blocks = sorted(work.loc[discovery_valid, 'time_block'].unique())
        for block in discovery_blocks:
            check_cancelled(_acid_cancel)
            test = discovery_valid & (work['time_block'].to_numpy() == block)
            train = discovery_valid & ~test
            if train.sum() < 50 or test.sum() == 0:
                continue
            model = make_nuisance_model()
            model.fit(work.loc[train, covariates], y[train])
            pred[test] = model.predict(work.loc[test, covariates])
        final_discovery_model = make_nuisance_model()
        final_discovery_model.fit(work.loc[discovery_valid, covariates], y[discovery_valid])
        pred[validation_valid] = final_discovery_model.predict(work.loc[validation_valid, covariates])
        # Was: pred[missing_discovery] = final_discovery_model.predict(...). That model
        # was fit on ALL discovery_valid rows -- INCLUDING these, whose own block had
        # too few other discovery rows (train.sum() < 50 above) for a genuine held-out
        # fit. Predicting them with a model trained on themselves is in-sample, not
        # held-out, and biases their residual toward zero. Leave them NaN instead --
        # nanmedian/nanstd below already tolerate it, and cluster_feature_space's own
        # notna() completeness filter excludes them downstream, same as any other
        # non-finite feature.
        missing_discovery = discovery_valid & ~np.isfinite(pred)
        if missing_discovery.any():
            print(f'{target}: {int(missing_discovery.sum())} discovery events left unresidualized (their block had too few other discovery events for a held-out fit); excluded from residual-based screens.')
        residual = y - pred
        center = float(np.nanmedian(residual[discovery_valid]))
        scale = 1.4826 * float(np.nanmedian(np.abs(residual[discovery_valid] - center)))
        scale = scale if np.isfinite(scale) and scale > 0 else float(np.nanstd(residual[discovery_valid]) or 1.0)
        work['amplitude_target'] = target
        work['amplitude_predicted'] = pred
        work['amplitude_residual'] = residual
        work['amplitude_resid_z'] = (residual - center) / scale
        return (work, covariates)


# Original notebook cell 48
def stage_048():
    global adjacent_separation, component_block_presence, ordered_component_labels, ordered_gmm_parameters, primary_residualized, primary_scalar, scalar_mixture_screen
    check_cancelled(_acid_cancel)

    def ordered_gmm_parameters(model: GaussianMixture) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = model.means_.ravel()
        variances = model.covariances_.reshape(model.n_components, -1)[:, 0]
        stds = np.sqrt(np.maximum(variances, 1e-12))
        weights = model.weights_.ravel()
        order = np.argsort(means)
        return (means[order], stds[order], weights[order], order)

    def ordered_component_labels(model: GaussianMixture, values: np.ndarray, min_posterior: float | None=None) -> tuple[np.ndarray, np.ndarray]:
        probabilities = model.predict_proba(values.reshape(-1, 1))
        _, _, _, order = ordered_gmm_parameters(model)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        raw = np.argmax(probabilities, axis=1)
        labels = rank[raw]
        strength = np.max(probabilities, axis=1)
        if min_posterior is not None:
            labels = labels.astype(int)
            labels[strength < min_posterior] = -1
        return (labels, strength)

    def adjacent_separation(means: np.ndarray, stds: np.ndarray) -> float:
        if len(means) < 2:
            return np.nan
        values = []
        for i in range(len(means) - 1):
            check_cancelled(_acid_cancel)
            pooled = math.sqrt((stds[i] ** 2 + stds[i + 1] ** 2) / 2.0)
            values.append(abs(means[i + 1] - means[i]) / max(pooled, 1e-12))
        return float(min(values))

    def component_block_presence(frame: pd.DataFrame, labels: np.ndarray, n_components: int) -> float:
        presences = []
        blocks = sorted(frame['time_block'].unique())
        for component in range(n_components):
            check_cancelled(_acid_cancel)
            present = 0
            for block in blocks:
                check_cancelled(_acid_cancel)
                in_block = frame['time_block'].to_numpy() == block
                # Was 'max(5, ...)' with no cap -- a block with fewer than 5 events
                # TOTAL could never reach a floor of 5 for ANY component, so it could
                # never register as present for anything, mechanically dragging down
                # min(presences) (the reported statistic) regardless of how cleanly
                # that block's few events actually sort into one component.
                required = min(int(in_block.sum()), max(5, int(math.ceil(0.01 * in_block.sum()))))
                present += int(np.sum(in_block & (labels == component)) >= required)
            presences.append(present / max(len(blocks), 1))
        return float(min(presences)) if presences else np.nan

    def scalar_mixture_screen(frame: pd.DataFrame, set_name: str) -> dict:
        target = choose_amplitude_target(frame)
        work, covariates = discovery_validation_residualize(frame, target)
        valid = work['amplitude_residual'].notna().to_numpy()
        discovery = valid & (work['split'].to_numpy() == 'discovery')
        validation = valid & (work['split'].to_numpy() == 'validation')
        if discovery.sum() < 30 or validation.sum() < 15:
            raise RuntimeError(f'Insufficient discovery/validation events for {set_name}')
        x_disc = work.loc[discovery, 'amplitude_residual'].to_numpy(float).reshape(-1, 1)
        x_val = work.loc[validation, 'amplitude_residual'].to_numpy(float).reshape(-1, 1)
        models = {}
        rows = []
        for k in GMM_COMPONENTS:
            check_cancelled(_acid_cancel)
            model = GaussianMixture(n_components=k, covariance_type='full', n_init=10, reg_covar=1e-07, random_state=SEED).fit(x_disc)
            models[k] = model
            rows.append({'k': k, 'bic_discovery': model.bic(x_disc), 'heldout_loglik_per_event': model.score(x_val)})
        scores = pd.DataFrame(rows)
        best_bic_k = int(scores.loc[scores['bic_discovery'].idxmin(), 'k'])
        best_val_k = int(scores.loc[scores['heldout_loglik_per_event'].idxmax(), 'k'])
        selected = models[best_bic_k]
        all_x = work.loc[valid, 'amplitude_residual'].to_numpy(float)
        labels_valid, strength_valid = ordered_component_labels(selected, all_x, min_posterior=MIN_POSTERIOR_FOR_LABEL)
        labels_hard, _ = ordered_component_labels(selected, all_x, min_posterior=None)
        labels_all = np.full(len(work), -1, dtype=int)
        strength_all = np.full(len(work), np.nan, dtype=float)
        hard_all = np.full(len(work), -1, dtype=int)
        labels_all[valid] = labels_valid
        strength_all[valid] = strength_valid
        hard_all[valid] = labels_hard
        work['scalar_component'] = labels_all
        work['scalar_component_strength'] = strength_all
        means, stds, weights, _ = ordered_gmm_parameters(selected)
        bic1 = float(scores.loc[scores['k'] == 1, 'bic_discovery'].iloc[0])
        bicbest = float(scores.loc[scores['k'] == best_bic_k, 'bic_discovery'].iloc[0])
        ll1 = float(scores.loc[scores['k'] == 1, 'heldout_loglik_per_event'].iloc[0])
        llbest = float(scores.loc[scores['k'] == best_bic_k, 'heldout_loglik_per_event'].iloc[0])
        delta_bic = bic1 - bicbest
        heldout_gain = llbest - ll1
        separation = adjacent_separation(means, stds)
        block_presence = component_block_presence(work.loc[valid].reset_index(drop=True), labels_hard, best_bic_k)
        dip, dip_p = (np.nan, np.nan)
        if HAVE_DIPTEST:
            dip, dip_p = diptest(np.sort(x_val.ravel()))
        candidate = bool(best_bic_k >= 2 and best_bic_k == best_val_k and (delta_bic >= GMM_MIN_DELTA_BIC) and (heldout_gain > GMM_MIN_HELDOUT_LL_GAIN) and (float(np.min(weights)) >= GMM_MIN_WEIGHT) and (separation >= GMM_MIN_SEPARATION) and (block_presence >= GMM_MIN_BLOCK_PRESENCE) and (not HAVE_DIPTEST or dip_p < 0.05))
        return {'set_name': set_name, 'work': work, 'target': target, 'covariates': covariates, 'scores': scores, 'models': models, 'selected_model': selected, 'best_bic_k': best_bic_k, 'best_val_k': best_val_k, 'delta_bic_vs_1': float(delta_bic), 'heldout_ll_gain_vs_1': float(heldout_gain), 'means': means, 'stds': stds, 'weights': weights, 'min_separation': float(separation), 'min_block_presence': float(block_presence), 'dip': float(dip), 'dip_p': float(dip_p), 'candidate': candidate}
    primary_scalar = scalar_mixture_screen(primary_catalog, PRIMARY_SET_NAME)
    primary_residualized = primary_scalar['work']
    print('Selected scalar GMM k:', primary_scalar['best_bic_k'])
    print('Held-out best k:', primary_scalar['best_val_k'])
    print('ΔBIC versus k=1:', primary_scalar['delta_bic_vs_1'])
    print('Held-out log-likelihood gain/event versus k=1:', primary_scalar['heldout_ll_gain_vs_1'])
    print('Ordered means:', np.round(primary_scalar['means'], 6))
    print('Ordered weights:', np.round(primary_scalar['weights'], 4))
    print('Minimum adjacent separation:', round(primary_scalar['min_separation'], 3))
    print('Minimum block presence:', round(primary_scalar['min_block_presence'], 3))
    print('Validation dip-test p:', primary_scalar['dip_p'] if HAVE_DIPTEST else 'not run')
    print('SCALAR SUBPOPULATION CANDIDATE:', primary_scalar['candidate'])
    display(primary_scalar['scores'].round(5))


def definitions_048():
    global adjacent_separation, component_block_presence, ordered_component_labels, ordered_gmm_parameters, scalar_mixture_screen

    def ordered_gmm_parameters(model: GaussianMixture) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = model.means_.ravel()
        variances = model.covariances_.reshape(model.n_components, -1)[:, 0]
        stds = np.sqrt(np.maximum(variances, 1e-12))
        weights = model.weights_.ravel()
        order = np.argsort(means)
        return (means[order], stds[order], weights[order], order)

    def ordered_component_labels(model: GaussianMixture, values: np.ndarray, min_posterior: float | None=None) -> tuple[np.ndarray, np.ndarray]:
        probabilities = model.predict_proba(values.reshape(-1, 1))
        _, _, _, order = ordered_gmm_parameters(model)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        raw = np.argmax(probabilities, axis=1)
        labels = rank[raw]
        strength = np.max(probabilities, axis=1)
        if min_posterior is not None:
            labels = labels.astype(int)
            labels[strength < min_posterior] = -1
        return (labels, strength)

    def adjacent_separation(means: np.ndarray, stds: np.ndarray) -> float:
        if len(means) < 2:
            return np.nan
        values = []
        for i in range(len(means) - 1):
            check_cancelled(_acid_cancel)
            pooled = math.sqrt((stds[i] ** 2 + stds[i + 1] ** 2) / 2.0)
            values.append(abs(means[i + 1] - means[i]) / max(pooled, 1e-12))
        return float(min(values))

    def component_block_presence(frame: pd.DataFrame, labels: np.ndarray, n_components: int) -> float:
        presences = []
        blocks = sorted(frame['time_block'].unique())
        for component in range(n_components):
            check_cancelled(_acid_cancel)
            present = 0
            for block in blocks:
                check_cancelled(_acid_cancel)
                in_block = frame['time_block'].to_numpy() == block
                # Was 'max(5, ...)' with no cap -- a block with fewer than 5 events
                # TOTAL could never reach a floor of 5 for ANY component, so it could
                # never register as present for anything, mechanically dragging down
                # min(presences) (the reported statistic) regardless of how cleanly
                # that block's few events actually sort into one component.
                required = min(int(in_block.sum()), max(5, int(math.ceil(0.01 * in_block.sum()))))
                present += int(np.sum(in_block & (labels == component)) >= required)
            presences.append(present / max(len(blocks), 1))
        return float(min(presences)) if presences else np.nan

    def scalar_mixture_screen(frame: pd.DataFrame, set_name: str) -> dict:
        target = choose_amplitude_target(frame)
        work, covariates = discovery_validation_residualize(frame, target)
        valid = work['amplitude_residual'].notna().to_numpy()
        discovery = valid & (work['split'].to_numpy() == 'discovery')
        validation = valid & (work['split'].to_numpy() == 'validation')
        if discovery.sum() < 30 or validation.sum() < 15:
            raise RuntimeError(f'Insufficient discovery/validation events for {set_name}')
        x_disc = work.loc[discovery, 'amplitude_residual'].to_numpy(float).reshape(-1, 1)
        x_val = work.loc[validation, 'amplitude_residual'].to_numpy(float).reshape(-1, 1)
        models = {}
        rows = []
        for k in GMM_COMPONENTS:
            check_cancelled(_acid_cancel)
            model = GaussianMixture(n_components=k, covariance_type='full', n_init=10, reg_covar=1e-07, random_state=SEED).fit(x_disc)
            models[k] = model
            rows.append({'k': k, 'bic_discovery': model.bic(x_disc), 'heldout_loglik_per_event': model.score(x_val)})
        scores = pd.DataFrame(rows)
        best_bic_k = int(scores.loc[scores['bic_discovery'].idxmin(), 'k'])
        best_val_k = int(scores.loc[scores['heldout_loglik_per_event'].idxmax(), 'k'])
        selected = models[best_bic_k]
        all_x = work.loc[valid, 'amplitude_residual'].to_numpy(float)
        labels_valid, strength_valid = ordered_component_labels(selected, all_x, min_posterior=MIN_POSTERIOR_FOR_LABEL)
        labels_hard, _ = ordered_component_labels(selected, all_x, min_posterior=None)
        labels_all = np.full(len(work), -1, dtype=int)
        strength_all = np.full(len(work), np.nan, dtype=float)
        hard_all = np.full(len(work), -1, dtype=int)
        labels_all[valid] = labels_valid
        strength_all[valid] = strength_valid
        hard_all[valid] = labels_hard
        work['scalar_component'] = labels_all
        work['scalar_component_strength'] = strength_all
        means, stds, weights, _ = ordered_gmm_parameters(selected)
        bic1 = float(scores.loc[scores['k'] == 1, 'bic_discovery'].iloc[0])
        bicbest = float(scores.loc[scores['k'] == best_bic_k, 'bic_discovery'].iloc[0])
        ll1 = float(scores.loc[scores['k'] == 1, 'heldout_loglik_per_event'].iloc[0])
        llbest = float(scores.loc[scores['k'] == best_bic_k, 'heldout_loglik_per_event'].iloc[0])
        delta_bic = bic1 - bicbest
        heldout_gain = llbest - ll1
        separation = adjacent_separation(means, stds)
        block_presence = component_block_presence(work.loc[valid].reset_index(drop=True), labels_hard, best_bic_k)
        dip, dip_p = (np.nan, np.nan)
        if HAVE_DIPTEST:
            dip, dip_p = diptest(np.sort(x_val.ravel()))
        candidate = bool(best_bic_k >= 2 and best_bic_k == best_val_k and (delta_bic >= GMM_MIN_DELTA_BIC) and (heldout_gain > GMM_MIN_HELDOUT_LL_GAIN) and (float(np.min(weights)) >= GMM_MIN_WEIGHT) and (separation >= GMM_MIN_SEPARATION) and (block_presence >= GMM_MIN_BLOCK_PRESENCE) and (not HAVE_DIPTEST or dip_p < 0.05))
        return {'set_name': set_name, 'work': work, 'target': target, 'covariates': covariates, 'scores': scores, 'models': models, 'selected_model': selected, 'best_bic_k': best_bic_k, 'best_val_k': best_val_k, 'delta_bic_vs_1': float(delta_bic), 'heldout_ll_gain_vs_1': float(heldout_gain), 'means': means, 'stds': stds, 'weights': weights, 'min_separation': float(separation), 'min_block_presence': float(block_presence), 'dip': float(dip), 'dip_p': float(dip_p), 'candidate': candidate}


# Original notebook cell 49
def stage_049():
    global _, _grid, _means, _model, _stds, _valid, _weights, _x, axes, fig, i, mean, std, weight
    check_cancelled(_acid_cancel)
    _valid = primary_residualized['amplitude_residual'].notna()
    _x = primary_residualized.loc[_valid, 'amplitude_residual'].to_numpy(float)
    _model = primary_scalar['selected_model']
    _means, _stds, _weights, _ = ordered_gmm_parameters(_model)
    _grid = np.linspace(np.percentile(_x, 0.2), np.percentile(_x, 99.8), 500)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].scatter(primary_residualized['amplitude_predicted'], primary_residualized[primary_scalar['target']], s=6, alpha=0.25)
    axes[0].set_xlabel('Blocked-CV nuisance prediction')
    axes[0].set_ylabel(primary_scalar['target'].replace('_', ' '))
    axes[0].set_title('Nuisance model')
    axes[1].scatter(primary_residualized['onset_s'], primary_residualized['amplitude_residual'], s=6, alpha=0.25)
    axes[1].axhline(0, linestyle=':', linewidth=1)
    axes[1].set_xlabel('Recording time (s)')
    axes[1].set_ylabel('Amplitude residual')
    axes[1].set_title('Residual versus time')
    axes[2].hist(_x, bins=70, density=True, alpha=0.35, label='events')
    for i, (mean, std, weight) in enumerate(zip(_means, _stds, _weights)):
        check_cancelled(_acid_cancel)
        axes[2].plot(_grid, weight * norm.pdf(_grid, mean, std), linewidth=2, label=f'component {i}: w={weight:.2f}')
    axes[2].set_xlabel('Dwell/context-adjusted amplitude residual')
    axes[2].set_ylabel('Density')
    axes[2].set_title(f"Scalar mixture: k={primary_scalar['best_bic_k']}")
    axes[2].legend(fontsize=8)
    finish_figure('04_scalar_residual_mixture')


# Original notebook cell 50
def stage_050():
    global _component_fractions, ax, fig
    check_cancelled(_acid_cancel)
    _component_fractions = primary_residualized.groupby(['time_block', 'scalar_component']).size().unstack(fill_value=0).div(primary_residualized.groupby('time_block').size(), axis=0)
    display(_component_fractions.round(3))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _component_fractions.plot(kind='bar', stacked=True, ax=ax)
    plt.xlabel('Contiguous time block')
    plt.ylabel('Fraction of events')
    plt.title('Scalar-component fractions across recording time')
    plt.legend(title='component', fontsize=8)
    finish_figure('05_scalar_components_by_time')


# Original notebook cell 51
def stage_051():
    global available_candidates, detector_consistent_scalar, exc, frame, name, result, scalar_comparison, scalar_comparison_rows, scalar_screens
    check_cancelled(_acid_cancel)
    scalar_screens = {PRIMARY_SET_NAME: primary_scalar}
    for name, frame in analysis_sets.items():
        check_cancelled(_acid_cancel)
        if name in scalar_screens:
            continue
        try:
            scalar_screens[name] = scalar_mixture_screen(frame, name)
        except Exception as exc:
            print(f'Scalar screen skipped for {name}: {exc}')
    scalar_comparison_rows = []
    for name, result in scalar_screens.items():
        check_cancelled(_acid_cancel)
        scalar_comparison_rows.append({'set': name, 'n_events': len(result['work']), 'target': result['target'], 'best_bic_k': result['best_bic_k'], 'best_heldout_k': result['best_val_k'], 'delta_bic_vs_1': result['delta_bic_vs_1'], 'heldout_ll_gain_vs_1': result['heldout_ll_gain_vs_1'], 'minimum_weight': float(np.min(result['weights'])), 'minimum_separation': result['min_separation'], 'minimum_block_presence': result['min_block_presence'], 'dip_p': result['dip_p'], 'candidate': result['candidate']})
    scalar_comparison = pd.DataFrame(scalar_comparison_rows)
    scalar_comparison.to_csv(OUT_DIR / f'{TAG}_scalar_detector_sensitivity.csv', index=False)
    display(scalar_comparison.round(5))
    available_candidates = scalar_comparison['candidate'].astype(bool)
    detector_consistent_scalar = bool(len(available_candidates) >= 2 and available_candidates.all())
    print('Candidate survives every available detector definition:', detector_consistent_scalar)


# Original notebook cell 53
def stage_053():
    global BASELINE_MODES, _acc, _alt_cols, _blk, _cmp, _i, _mode, _off, _ov, _stat, _verdicts, ax, axes, baseline_amplitude_table, baseline_screen_table, baseline_screens, c, d, exc, fig, frame_alt, hi, lim, lo, m, pa, pad, q, resid, row, rows, t
    check_cancelled(_acid_cancel)
    BASELINE_MODES = ['local', 'global', 'block']
    _cmp = primary_catalog.copy()
    _stat = AMP_BASE
    _alt_cols = {}
    if not all((f'{_stat}_depth_{_mode}_pA' in _cmp for _mode in BASELINE_MODES)):
        print(f"No per-baseline columns for '{_stat}'; the comparison uses the peak instead.")
        _stat = 'peak'
    for _mode in BASELINE_MODES:
        check_cancelled(_acid_cancel)
        _alt_cols[_mode] = f'{_stat}_depth_{_mode}_pA'
    _off = pd.DataFrame({'local - global (pA)': _cmp['local_baseline_detr_pA'] - _cmp['global_baseline_pA'], 'block - global (pA)': _cmp['block_baseline_pA'] - _cmp['global_baseline_pA'], 'local - block (pA)': _cmp['local_baseline_detr_pA'] - _cmp['block_baseline_pA']})
    print('Baseline offsets across primary events (median [IQR]):')
    for c in _off:
        check_cancelled(_acid_cancel)
        q = _off[c].quantile([0.25, 0.5, 0.75])
        print(f'  {c}: {q[0.5]:+.2f} [{q[0.25]:+.2f}, {q[0.75]:+.2f}]')
    rows = []
    for _mode in BASELINE_MODES:
        check_cancelled(_acid_cancel)
        d = _cmp[_alt_cols[_mode]] - _cmp[_alt_cols['local']]
        rows.append({'baseline': _mode, f'{_stat} median (pA)': float(_cmp[_alt_cols[_mode]].median()), 'diff vs local median (pA)': float(d.median()), 'diff vs local IQR (pA)': float(d.quantile(0.75) - d.quantile(0.25)), 'spearman diff vs log_dwell': float(spearmanr(d, _cmp['log_dwell'], nan_policy='omit')[0]) if _mode != 'local' else np.nan, 'spearman diff vs onset_fraction': float(spearmanr(d, _cmp['onset_fraction'], nan_policy='omit')[0]) if _mode != 'local' else np.nan, 'spearman amplitude vs local': float(spearmanr(_cmp[_alt_cols[_mode]], _cmp[_alt_cols['local']], nan_policy='omit')[0])})
    baseline_amplitude_table = pd.DataFrame(rows)
    display(baseline_amplitude_table.round(3))
    _acc = {}
    for _mode in BASELINE_MODES:
        check_cancelled(_acid_cancel)
        _acc[_mode] = _cmp['accepted_base'] & passes_acceptance_gate(_cmp[f'sustained_depth_{_mode}_pA'], _cmp[f'half_peak_mean_depth_{_mode}_pA'], _cmp['half_peak_ms'], _cmp['dwell_ms'], peak_depth_pA=_cmp[f'peak_depth_{_mode}_pA'], local_noise_pA=_cmp['local_noise_mad_pA'])
    _ov = pd.DataFrame({a: {b: int((_acc[a] & _acc[b]).sum()) for b in BASELINE_MODES} for a in BASELINE_MODES})
    print('Events of the primary set that would pass the acceptance gate under each baseline (diagonal) and under both (off-diagonal):')
    display(_ov)
    baseline_screens = {}
    for _mode in BASELINE_MODES:
        check_cancelled(_acid_cancel)
        frame_alt = _cmp.copy()
        pa = _alt_cols[_mode]
        frame_alt[AMP_PA_COL] = frame_alt[pa]
        frame_alt[AMP_SIGMA_COL] = frame_alt[pa] / frame_alt['local_noise_mad_pA'].clip(lower=1e-12)
        frame_alt[AMP_FRAC_COL] = frame_alt[pa] / frame_alt['local_I0_pA']
        try:
            baseline_screens[_mode] = scalar_mixture_screen(frame_alt, f'{PRIMARY_SET_NAME}:{_mode}')
        except Exception as exc:
            print(f'Scalar screen skipped for baseline {_mode}: {exc}')
    baseline_screen_table = pd.DataFrame([{'baseline': m, 'target': r['target'], 'best_bic_k': r['best_bic_k'], 'best_heldout_k': r['best_val_k'], 'delta_bic_vs_1': r['delta_bic_vs_1'], 'heldout_ll_gain_vs_1': r['heldout_ll_gain_vs_1'], 'min_weight': float(np.min(r['weights'])), 'min_separation': r['min_separation'], 'min_block_presence': r['min_block_presence'], 'dip_p': r['dip_p'], 'candidate': r['candidate']} for m, r in baseline_screens.items()])
    baseline_screen_table.to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_baseline_sensitivity.csv', index=False)
    print(f'Scalar screen on {AMP_FRAC_COL} under each baseline (canonical = {BASELINE_MODE}):')
    display(baseline_screen_table.round(4))
    _verdicts = {m: (r['best_bic_k'], r['candidate']) for m, r in baseline_screens.items()}
    print('Verdict consistent across baselines:', len({v for v in _verdicts.values()}) == 1, _verdicts)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    lim = float(np.nanmax(_cmp[[_alt_cols['local'], _alt_cols['global']]].to_numpy())) * 1.05
    ax.scatter(_cmp[_alt_cols['local']], _cmp[_alt_cols['global']], s=8, alpha=0.5)
    ax.plot([0, lim], [0, lim], linestyle=':', color='k', linewidth=1)
    ax.set_xlabel(f'{_stat} depth, local baseline (pA)')
    ax.set_ylabel(f'{_stat} depth, global baseline (pA)')
    ax.set_title('Global vs local baseline')
    ax = axes[0, 1]
    ax.scatter(_cmp['dwell_ms'], _cmp[_alt_cols['global']] - _cmp[_alt_cols['local']], s=8, alpha=0.5, label='global - local')
    ax.scatter(_cmp['dwell_ms'], _cmp[_alt_cols['block']] - _cmp[_alt_cols['local']], s=8, alpha=0.5, label='block - local')
    ax.axhline(0, linestyle=':', color='k', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Dwell (ms)')
    ax.set_ylabel('Amplitude difference (pA)')
    ax.set_title('Difference versus dwell')
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    for c in _off:
        check_cancelled(_acid_cancel)
        ax.hist(_off[c].clip(-15, 15), bins=60, histtype='step', linewidth=1.5, label=c)
    ax.set_xlabel('Baseline offset (pA)')
    ax.set_ylabel('Events')
    ax.set_title('What the local context absorbs')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    _blk = _cmp.groupby('time_block')[[_alt_cols[m] for m in BASELINE_MODES]].median()
    for m in BASELINE_MODES:
        check_cancelled(_acid_cancel)
        ax.plot(_blk.index, _blk[_alt_cols[m]], marker='o', label=m)
    ax.set_xlabel('Time block')
    ax.set_ylabel(f'{_stat} depth, median over events (pA)')
    ax.set_title('Per-block median by baseline')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for m in BASELINE_MODES:
        check_cancelled(_acid_cancel)
        if m in baseline_screens:
            resid = baseline_screens[m]['work']['amplitude_residual'].dropna()
            ax.hist(resid.clip(resid.quantile(0.005), resid.quantile(0.995)), bins=60, histtype='step', linewidth=1.5, density=True, label=f"{m} (k={baseline_screens[m]['best_bic_k']})")
    ax.set_xlabel('Dwell/context-adjusted amplitude residual')
    ax.set_ylabel('Density')
    ax.set_title('Scalar residual by baseline')
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    _i = int((_cmp[_alt_cols['global']] - _cmp[_alt_cols['local']]).abs().idxmax())
    row = _cmp.loc[_i]
    pad = int(round(3.0 * fs / 1000.0))
    lo, hi = (max(0, int(row.start_idx) - pad), min(n_samples, int(row.end_idx) + 1 + pad))
    t = (np.arange(lo, hi) - int(row.start_idx)) / fs * 1000.0
    ax.plot(t, analysis_signal[lo:hi], linewidth=0.7)
    ax.axhline(row.local_baseline_detr_pA, linestyle=':', color='tab:blue', label='local')
    ax.axhline(row.global_baseline_pA, linestyle='--', color='tab:orange', label='global')
    ax.axhline(row.block_baseline_pA, linestyle='-.', color='tab:green', label='block')
    ax.axvspan(0, (row.end_idx - row.start_idx + 1) / fs * 1000.0, alpha=0.15)
    ax.set_xlabel('ms from onset')
    ax.set_ylabel('pA')
    ax.set_title(f'{row.event_id}: largest global-local disagreement')
    ax.legend(fontsize=8)
    finish_figure('13_baseline_comparison')


# Original notebook cell 54
def stage_054():
    global block_bootstrap_gmm, bootstrap_parameters, bootstrap_summary, early, early_fit, early_late, fit_ordered_gmm, late, late_fit, midpoint
    check_cancelled(_acid_cancel)

    def block_bootstrap_gmm(frame: pd.DataFrame, k: int, n_boot: int, seed: int=SEED) -> pd.DataFrame:
        if k < 2:
            return pd.DataFrame()
        rng = np.random.default_rng(seed)
        blocks = np.array(sorted(frame['time_block'].unique()))
        output = []
        for boot in tqdm(range(n_boot), desc='Block bootstrap GMM'):
            check_cancelled(_acid_cancel)
            sampled = rng.choice(blocks, size=len(blocks), replace=True)
            values = np.concatenate([frame.loc[frame['time_block'] == block, 'amplitude_residual'].dropna().to_numpy(float) for block in sampled])
            if len(values) < 50:
                continue
            try:
                model = GaussianMixture(k, n_init=5, reg_covar=1e-07, random_state=seed + boot).fit(values[:, None])
                means, stds, weights, _ = ordered_gmm_parameters(model)
                for component in range(k):
                    check_cancelled(_acid_cancel)
                    output.append({'bootstrap': boot, 'component': component, 'mean': means[component], 'std': stds[component], 'weight': weights[component]})
            except Exception:
                continue
        return pd.DataFrame(output)
    bootstrap_parameters = block_bootstrap_gmm(primary_residualized, primary_scalar['best_bic_k'], N_BLOCK_BOOTSTRAPS)
    if len(bootstrap_parameters):
        bootstrap_summary = bootstrap_parameters.groupby('component')[['mean', 'std', 'weight']].quantile([0.025, 0.5, 0.975]).rename_axis(index=['component', 'quantile']).reset_index()
        bootstrap_parameters.to_parquet(OUT_DIR / f'{TAG}_scalar_block_bootstrap.parquet', index=False)
        display(bootstrap_summary.round(5))
    else:
        bootstrap_summary = pd.DataFrame()
        print('Block bootstrap not applicable because the selected scalar model has one component.')

    def fit_ordered_gmm(values: np.ndarray, k: int, seed: int) -> dict:
        model = GaussianMixture(k, n_init=10, reg_covar=1e-07, random_state=seed).fit(values.reshape(-1, 1))
        means, stds, weights, _ = ordered_gmm_parameters(model)
        return {'means': means, 'stds': stds, 'weights': weights}
    if primary_scalar['best_bic_k'] >= 2:
        midpoint = duration_s / 2
        early = primary_residualized.loc[primary_residualized['onset_s'] < midpoint, 'amplitude_residual'].dropna().to_numpy(float)
        late = primary_residualized.loc[primary_residualized['onset_s'] >= midpoint, 'amplitude_residual'].dropna().to_numpy(float)
        early_fit = fit_ordered_gmm(early, primary_scalar['best_bic_k'], SEED)
        late_fit = fit_ordered_gmm(late, primary_scalar['best_bic_k'], SEED + 1)
        early_late = pd.DataFrame({'component': np.arange(primary_scalar['best_bic_k']), 'early_mean': early_fit['means'], 'late_mean': late_fit['means'], 'early_weight': early_fit['weights'], 'late_weight': late_fit['weights']})
        display(early_late.round(5))
    else:
        early_late = pd.DataFrame()


def definitions_054():
    global block_bootstrap_gmm, fit_ordered_gmm

    def block_bootstrap_gmm(frame: pd.DataFrame, k: int, n_boot: int, seed: int=SEED) -> pd.DataFrame:
        if k < 2:
            return pd.DataFrame()
        rng = np.random.default_rng(seed)
        blocks = np.array(sorted(frame['time_block'].unique()))
        output = []
        for boot in tqdm(range(n_boot), desc='Block bootstrap GMM'):
            check_cancelled(_acid_cancel)
            sampled = rng.choice(blocks, size=len(blocks), replace=True)
            values = np.concatenate([frame.loc[frame['time_block'] == block, 'amplitude_residual'].dropna().to_numpy(float) for block in sampled])
            if len(values) < 50:
                continue
            try:
                model = GaussianMixture(k, n_init=5, reg_covar=1e-07, random_state=seed + boot).fit(values[:, None])
                means, stds, weights, _ = ordered_gmm_parameters(model)
                for component in range(k):
                    check_cancelled(_acid_cancel)
                    output.append({'bootstrap': boot, 'component': component, 'mean': means[component], 'std': stds[component], 'weight': weights[component]})
            except Exception:
                continue
        return pd.DataFrame(output)

    def fit_ordered_gmm(values: np.ndarray, k: int, seed: int) -> dict:
        model = GaussianMixture(k, n_init=10, reg_covar=1e-07, random_state=seed).fit(values.reshape(-1, 1))
        means, stds, weights, _ = ordered_gmm_parameters(model)
        return {'means': means, 'stds': stds, 'weights': weights}


# Original notebook cell 56
def stage_056():
    global PCA_PHYS, X_all_scaled, X_disc_imp, X_disc_scaled, X_phys, X_phys_disc, X_phys_val, cumulative, discovery_mask, imputer_phys, missing_features, n_pcs, pca_full, physical_valid, physical_work, scaler_phys, validation_mask
    check_cancelled(_acid_cancel)
    physical_work = primary_residualized.copy().reset_index(drop=True)
    missing_features = [c for c in PHYSICAL_FEATURES if c not in physical_work]
    if missing_features:
        raise KeyError(f'Missing physical clustering features: {missing_features}')
    physical_valid = physical_work[PHYSICAL_FEATURES].replace([np.inf, -np.inf], np.nan).notna().sum(axis=1) >= max(2, len(PHYSICAL_FEATURES) - 1)
    physical_work = physical_work.loc[physical_valid].copy().reset_index(drop=True)
    discovery_mask = physical_work['time_block'].to_numpy() % 2 == 0
    validation_mask = ~discovery_mask
    if discovery_mask.sum() < 30 or validation_mask.sum() < 15:
        raise RuntimeError('Insufficient discovery/validation events for physical clustering')
    imputer_phys = SimpleImputer(strategy='median')
    scaler_phys = RobustScaler()
    X_disc_imp = imputer_phys.fit_transform(physical_work.loc[discovery_mask, PHYSICAL_FEATURES])
    X_disc_scaled = np.clip(scaler_phys.fit_transform(X_disc_imp), -6.0, 6.0)
    X_all_scaled = np.clip(scaler_phys.transform(imputer_phys.transform(physical_work[PHYSICAL_FEATURES])), -6.0, 6.0)
    pca_full = PCA(n_components=min(len(PHYSICAL_FEATURES), discovery_mask.sum() - 1), random_state=SEED)
    pca_full.fit(X_disc_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_pcs = max(2, int(np.searchsorted(cumulative, 0.95) + 1))
    n_pcs = min(n_pcs, pca_full.n_components_)
    PCA_PHYS = PCA(n_components=n_pcs, random_state=SEED).fit(X_disc_scaled)
    X_phys = PCA_PHYS.transform(X_all_scaled)
    X_phys_disc = X_phys[discovery_mask]
    X_phys_val = X_phys[validation_mask]
    print('Physical features:', PHYSICAL_FEATURES)
    print('PCA components retained:', n_pcs)
    print('Explained variance retained:', PCA_PHYS.explained_variance_ratio_.sum())


# Original notebook cell 57
def stage_057():
    global _grid_seen, _hdbscan_params, _n_phys_disc, all_clusters_recur, cluster_id, cluster_ids, frac, grid_rows, hdbscan_grid, hdbscan_metrics, hdbscan_model, labels, labels_disc, labels_val, mcs, mcs_default, min_samples, model, ms_default, n_clusters, n_physical_clusters, noise_fraction, physical_labels, physical_noise_fraction, physical_silhouette, physical_strength, reference_labels, relative_validity, required_validation, silhouette, stability_median_ari, strength_disc, strength_val, validation_assigned_fraction, validation_presence
    check_cancelled(_acid_cancel)

    def hdbscan_metrics(labels: np.ndarray, X: np.ndarray) -> tuple[int, float, float]:
        clusters = sorted(set(labels) - {-1})
        noise_fraction = float(np.mean(labels == -1))
        silhouette = np.nan
        nonnoise = labels != -1
        if len(clusters) >= 2 and nonnoise.sum() > len(clusters):
            silhouette = float(silhouette_score(X[nonnoise], labels[nonnoise]))
        return (len(clusters), noise_fraction, silhouette)
    _n_phys_disc = len(X_phys_disc)

    def _hdbscan_params(mcs: int, min_samples: int) -> tuple[int, int]:
        return (int(min(mcs, max(2, _n_phys_disc - 1))), int(min(min_samples, max(1, _n_phys_disc - 1))))
    mcs_default, ms_default = _hdbscan_params(max(HDBSCAN_MIN_CLUSTER_FLOOR, int(round(HDBSCAN_DEFAULT_MIN_CLUSTER_FRACTION * len(X_phys_disc)))), HDBSCAN_DEFAULT_MIN_SAMPLES)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=mcs_default, min_samples=ms_default, prediction_data=True, gen_min_span_tree=True).fit(X_phys_disc)
    labels_disc = hdbscan_model.labels_.astype(int)
    strength_disc = hdbscan_model.probabilities_.astype(float)
    if len(set(labels_disc) - {-1}) > 0:
        labels_val, strength_val = hdbscan.approximate_predict(hdbscan_model, X_phys_val)
        labels_val = labels_val.astype(int)
        strength_val = strength_val.astype(float)
    else:
        labels_val = np.full(len(X_phys_val), -1, dtype=int)
        strength_val = np.zeros(len(X_phys_val), dtype=float)
        print('HDBSCAN found no discovery cluster; every validation event is left unassigned.')
    physical_labels = np.full(len(physical_work), -1, dtype=int)
    physical_strength = np.full(len(physical_work), np.nan, dtype=float)
    physical_labels[discovery_mask] = labels_disc
    physical_labels[validation_mask] = labels_val
    physical_strength[discovery_mask] = strength_disc
    physical_strength[validation_mask] = strength_val
    physical_work['physical_cluster'] = physical_labels
    physical_work['physical_cluster_strength'] = physical_strength
    physical_work['split'] = np.where(discovery_mask, 'discovery', 'validation')
    reference_labels = labels_disc.copy()
    grid_rows = []
    _grid_seen = set()
    for frac in HDBSCAN_MIN_CLUSTER_FRACTIONS:
        check_cancelled(_acid_cancel)
        for min_samples in HDBSCAN_MIN_SAMPLES_GRID:
            check_cancelled(_acid_cancel)
            mcs, min_samples = _hdbscan_params(max(HDBSCAN_MIN_CLUSTER_FLOOR, int(round(frac * len(X_phys_disc)))), min_samples)
            if (mcs, min_samples) in _grid_seen:
                continue
            _grid_seen.add((mcs, min_samples))
            model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, prediction_data=True, gen_min_span_tree=True).fit(X_phys_disc)
            labels = model.labels_.astype(int)
            n_clusters, noise_fraction, silhouette = hdbscan_metrics(labels, X_phys_disc)
            try:
                relative_validity = float(model.relative_validity_)
            except Exception:
                relative_validity = np.nan
            grid_rows.append({'min_cluster_fraction': frac, 'min_cluster_size': mcs, 'min_samples': min_samples, 'n_clusters': n_clusters, 'noise_fraction': noise_fraction, 'silhouette_nonnoise': silhouette, 'relative_validity': relative_validity, 'ARI_vs_default': adjusted_rand_score(reference_labels, labels)})
    hdbscan_grid = pd.DataFrame(grid_rows)
    hdbscan_grid.to_csv(OUT_DIR / f'{TAG}_physical_hdbscan_grid.csv', index=False)
    display(hdbscan_grid.round(4))
    n_physical_clusters, physical_noise_fraction, physical_silhouette = hdbscan_metrics(physical_labels, X_phys)
    stability_median_ari = float(hdbscan_grid['ARI_vs_default'].median())
    validation_assigned_fraction = float(np.mean(labels_val != -1))
    cluster_ids = sorted(set(labels_disc) - {-1})
    validation_presence = {}
    for cluster_id in cluster_ids:
        check_cancelled(_acid_cancel)
        validation_presence[cluster_id] = int(np.sum(labels_val == cluster_id))
    required_validation = max(10, int(math.ceil(0.01 * len(labels_val))))
    all_clusters_recur = bool(cluster_ids and all((v >= required_validation for v in validation_presence.values())))
    print('Default HDBSCAN clusters:', n_physical_clusters)
    print('Noise/unassigned fraction:', round(physical_noise_fraction, 3))
    print('Validation assigned fraction:', round(validation_assigned_fraction, 3))
    print('Validation counts by discovery cluster:', validation_presence)
    print('Median ARI across parameter grid:', round(stability_median_ari, 3))


def definitions_057():
    global _hdbscan_params, hdbscan_metrics

    def hdbscan_metrics(labels: np.ndarray, X: np.ndarray) -> tuple[int, float, float]:
        clusters = sorted(set(labels) - {-1})
        noise_fraction = float(np.mean(labels == -1))
        silhouette = np.nan
        nonnoise = labels != -1
        if len(clusters) >= 2 and nonnoise.sum() > len(clusters):
            silhouette = float(silhouette_score(X[nonnoise], labels[nonnoise]))
        return (len(clusters), noise_fraction, silhouette)

    def _hdbscan_params(mcs: int, min_samples: int) -> tuple[int, int]:
        return (int(min(mcs, max(2, _n_phys_disc - 1))), int(min(min_samples, max(1, _n_phys_disc - 1))))


# Original notebook cell 58
def stage_058():
    global _, cluster_id, cluster_rows, cramers_v, g, nonnoise, physical_cluster_candidate, physical_cluster_summary, physical_time_cramers_v, physical_time_p, time_table
    check_cancelled(_acid_cancel)

    def cramers_v(table: pd.DataFrame) -> float:
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan
        chi2 = chi2_contingency(table, correction=False)[0]
        n = table.to_numpy().sum()
        phi2 = chi2 / max(n, 1)
        r, k = table.shape
        phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
        rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
        kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
        denom = min(kcorr - 1, rcorr - 1)
        return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan
    cluster_rows = []
    for cluster_id in sorted(set(physical_work['physical_cluster']) - {-1}):
        check_cancelled(_acid_cancel)
        g = physical_work.loc[physical_work['physical_cluster'] == cluster_id]
        cluster_rows.append({'cluster': int(cluster_id), 'n_total': len(g), 'n_discovery': int(np.sum(g['split'] == 'discovery')), 'n_validation': int(np.sum(g['split'] == 'validation')), 'blocks_present': int(g['time_block'].nunique()), 'dwell_median_ms': g['dwell_ms'].median(), 'amplitude_resid_median': g['amplitude_residual'].median(), 'amplitude_fracchange_median': g[AMP_FRAC_COL].median(), 'peak_to_sustained_median': g['peak_to_sustained'].median(), 'shape_factor_median': g['shape_factor'].median()})
    physical_cluster_summary = pd.DataFrame(cluster_rows)
    physical_cluster_summary.to_csv(OUT_DIR / f'{TAG}_physical_cluster_summary.csv', index=False)
    display(physical_cluster_summary.round(4))
    nonnoise = physical_work['physical_cluster'] != -1
    time_table = pd.crosstab(physical_work.loc[nonnoise, 'physical_cluster'], physical_work.loc[nonnoise, 'time_block'])
    physical_time_cramers_v = cramers_v(time_table)
    if time_table.shape[0] >= 2 and time_table.shape[1] >= 2:
        _, physical_time_p, _, _ = chi2_contingency(time_table)
    else:
        physical_time_p = np.nan
    print("Physical cluster × time-block Cramér's V:", physical_time_cramers_v)
    print('Physical cluster × time-block chi-square p:', physical_time_p)
    physical_cluster_candidate = bool(n_physical_clusters >= 2 and all_clusters_recur and (validation_assigned_fraction >= 0.5) and (stability_median_ari >= 0.5) and (not np.isfinite(physical_time_cramers_v) or physical_time_cramers_v < 0.3))
    print('STABLE PHYSICAL-SPACE CLUSTER CANDIDATE:', physical_cluster_candidate)


def definitions_058():
    global cramers_v

    def cramers_v(table: pd.DataFrame) -> float:
        if table.shape[0] < 2 or table.shape[1] < 2:
            return np.nan
        chi2 = chi2_contingency(table, correction=False)[0]
        n = table.to_numpy().sum()
        phi2 = chi2 / max(n, 1)
        r, k = table.shape
        phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
        rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
        kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
        denom = min(kcorr - 1, rcorr - 1)
        return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan


# Original notebook cell 59
def stage_059():
    global axes, embedding, fig, reducer, scatter
    check_cancelled(_acid_cancel)
    reducer = umap.UMAP(n_neighbors=min(30, max(2, len(X_phys) - 1)), min_dist=0.1, random_state=SEED)
    embedding = reducer.fit_transform(X_phys)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=physical_labels, s=8, alpha=0.55, cmap='tab10')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('Physical/PCA labels displayed in UMAP')
    plt.colorbar(scatter, ax=axes[0], label='physical cluster (-1 = noise)')
    scatter = axes[1].scatter(physical_work['dwell_ms'], physical_work['amplitude_residual'], c=physical_labels, s=8, alpha=0.55, cmap='tab10')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Dwell time (ms)')
    axes[1].set_ylabel('Adjusted amplitude residual')
    axes[1].set_title('Physical clusters in direct observable space')
    plt.colorbar(scatter, ax=axes[1], label='physical cluster')
    finish_figure('06_physical_clusters')


# Original notebook cell 60
def stage_060():
    global _common, k, model, phys_gmm_best_bic, phys_gmm_best_val, phys_gmm_hdbscan_ari, phys_gmm_model, phys_gmm_models, phys_gmm_rows, phys_gmm_scores
    check_cancelled(_acid_cancel)
    phys_gmm_rows = []
    phys_gmm_models = {}
    for k in (1, 2, 3, 4):
        check_cancelled(_acid_cancel)
        model = GaussianMixture(k, n_init=10, reg_covar=1e-06, random_state=SEED).fit(X_phys_disc)
        phys_gmm_models[k] = model
        phys_gmm_rows.append({'k': k, 'bic_discovery': model.bic(X_phys_disc), 'heldout_loglik_per_event': model.score(X_phys_val)})
    phys_gmm_scores = pd.DataFrame(phys_gmm_rows)
    phys_gmm_best_bic = int(phys_gmm_scores.loc[phys_gmm_scores['bic_discovery'].idxmin(), 'k'])
    phys_gmm_best_val = int(phys_gmm_scores.loc[phys_gmm_scores['heldout_loglik_per_event'].idxmax(), 'k'])
    # Was phys_gmm_best_bic -- the labels actually assigned to events (and the
    # GMM/HDBSCAN ARI computed from them just below) came from the BIC-selected model,
    # not the held-out-validated one the prints and report already call "selected"
    # (gmm_best_heldout_k). The near-identical extended-tier cluster_feature_space
    # uses its held-out/admissible k for exactly this; this simpler physical-tier copy
    # had drifted from that.
    phys_gmm_model = phys_gmm_models[phys_gmm_best_val]
    physical_work['physical_gmm_component'] = phys_gmm_model.predict(X_phys)
    _common = physical_work['physical_cluster'] != -1
    phys_gmm_hdbscan_ari = adjusted_rand_score(physical_work.loc[_common, 'physical_cluster'], physical_work.loc[_common, 'physical_gmm_component']) if _common.sum() >= 20 and n_physical_clusters >= 2 else np.nan
    display(phys_gmm_scores.round(5))
    print('Physical GMM best BIC k:', phys_gmm_best_bic)
    print('Physical GMM best held-out k:', phys_gmm_best_val)
    print('GMM/HDBSCAN ARI on non-noise events:', phys_gmm_hdbscan_ari)


# Original notebook cell 62
def stage_062():
    global CHARACTERIZATION_COLUMNS, REDUNDANCY_RHO, SCALED_CLIP, _resolved, _tier, characterize_groups, cluster_feature_space, drop_redundant_features, extended_results, label_cols, name, result, tier_a_features, tier_b_features
    check_cancelled(_acid_cancel)
    REDUNDANCY_RHO = 0.85
    SCALED_CLIP = 6.0
    CHARACTERIZATION_COLUMNS = ['dwell_ms', 'peak_fracchange', 'peak_corrected_fracchange', 'rms_fracchange', 'peak_to_sustained', 'depth_kurtosis', 'tkeo_mean_sigma2', 'spectral_entropy', 'pre_enhancement_sigma', 'post_peak_efold_fraction', 'tail_to_peak_ratio', 'decay_tau_ms', 'entry_shoulder_ratio', 'terminal_spike_ratio', 'pre_bump_to_peak_ratio', 'cusum_n_levels', 'cusum_min_level_sep_sigma', 'cusum_longest_level_fraction']

    def drop_redundant_features(frame: pd.DataFrame, features: list[str], rho: float) -> tuple[list[str], list[tuple[str, str, float]]]:
        corr = frame[features].corr(method='spearman', min_periods=30)
        kept = []
        dropped = []
        for feature in features:
            check_cancelled(_acid_cancel)
            partner = next((k for k in kept if np.isfinite(corr.loc[feature, k]) and abs(corr.loc[feature, k]) >= rho), None)
            if partner is None:
                kept.append(feature)
            else:
                dropped.append((feature, partner, float(corr.loc[feature, partner])))
        return (kept, dropped)

    def characterize_groups(frame: pd.DataFrame, labels: np.ndarray, split: np.ndarray) -> pd.DataFrame:
        columns = [c for c in CHARACTERIZATION_COLUMNS if c in frame]
        rows = []
        for group in sorted(set(labels) - {-1}):
            check_cancelled(_acid_cancel)
            m = labels == group
            g = frame.loc[m]
            row = {'group': int(group), 'n_total': int(m.sum()), 'n_discovery': int((m & (split == 'discovery')).sum()), 'n_validation': int((m & (split == 'validation')).sum()), 'blocks_present': int(g['time_block'].nunique())}
            row.update({f'{c}_median': g[c].median() for c in columns})
            rows.append(row)
        return pd.DataFrame(rows)

    def cluster_feature_space(frame: pd.DataFrame, features: list[str], *, label: str, min_cluster_fractions=HDBSCAN_MIN_CLUSTER_FRACTIONS, min_samples_grid=EXT_MIN_SAMPLES_GRID, min_cluster_floor=HDBSCAN_MIN_CLUSTER_FLOOR, pca_variance: float=0.95, gmm_ks=(1, 2, 3, 4), seed: int=SEED) -> dict:
        features = list(dict.fromkeys(features))
        work = frame.copy().reset_index(drop=True)
        X_raw = work[features].replace([np.inf, -np.inf], np.nan)
        complete = X_raw.notna().all(axis=1).to_numpy()
        n_dropped = int((~complete).sum())
        work = work.loc[complete].reset_index(drop=True)
        X_raw = X_raw.loc[complete].reset_index(drop=True)
        discovery = work['time_block'].to_numpy() % 2 == 0
        validation = ~discovery
        if discovery.sum() < 30 or validation.sum() < 15:
            print(f'[{label}] skipped: {int(discovery.sum())} discovery / {int(validation.sum())} validation events with every feature finite ({n_dropped} dropped for a non-finite feature); a tier needs >= 30 / 15.')
            return None
        features, dropped_features = drop_redundant_features(X_raw.loc[discovery], features, REDUNDANCY_RHO)
        X_raw = X_raw[features]
        scaler = RobustScaler().fit(X_raw.loc[discovery])
        X_scaled = np.clip(scaler.transform(X_raw), -SCALED_CLIP, SCALED_CLIP)
        pca_probe = PCA(n_components=min(len(features), int(discovery.sum()) - 1), random_state=seed).fit(X_scaled[discovery])
        if pca_probe.n_components_ < 2:
            # np.clip(a, a_min, a_max) with a_min > a_max returns a_max, not the floor --
            # so clipping to (2, pca_probe.n_components_) below would silently return 1,
            # not 2, whenever only 1 feature survives drop_redundant_features. The 2D
            # scatter/plot further down unconditionally indexes a second PCA column, so
            # this tier cannot proceed with fewer than 2 components; skip it like the
            # other insufficient-data cases above.
            print(f'[{label}] skipped: only {pca_probe.n_components_} PCA component available after redundancy pruning ({len(features)} feature(s)); need >= 2.')
            return None
        cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
        n_pcs = int(np.clip(np.searchsorted(cumulative, pca_variance) + 1, 2, pca_probe.n_components_))
        pca = PCA(n_components=n_pcs, random_state=seed).fit(X_scaled[discovery])
        X = pca.transform(X_scaled)
        X_disc, X_val = (X[discovery], X[validation])
        loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f'PC{i + 1}' for i in range(n_pcs)])
        split = np.where(discovery, 'discovery', 'validation')
        grid_rows, fits = ([], {})
        for frac in min_cluster_fractions:
            check_cancelled(_acid_cancel)
            mcs = max(min_cluster_floor, int(round(frac * len(X_disc))))
            for min_samples in min_samples_grid:
                check_cancelled(_acid_cancel)
                mcs_fit, ms_fit = (int(min(mcs, max(2, len(X_disc) - 1))), int(min(int(min_samples), max(1, len(X_disc) - 1))))
                key = (mcs_fit, ms_fit)
                if key in fits:
                    continue
                model = hdbscan.HDBSCAN(min_cluster_size=mcs_fit, min_samples=ms_fit, prediction_data=True, gen_min_span_tree=True).fit(X_disc)
                labels = model.labels_.astype(int)
                n_clusters, noise_fraction, silhouette = hdbscan_metrics(labels, X_disc)
                try:
                    relative_validity = float(model.relative_validity_)
                except Exception:
                    relative_validity = np.nan
                fits[key] = (model, labels)
                grid_rows.append({'min_cluster_size': mcs_fit, 'min_samples': ms_fit, 'n_clusters': n_clusters, 'noise_fraction': noise_fraction, 'silhouette_nonnoise': silhouette, 'relative_validity': relative_validity})
        grid = pd.DataFrame(grid_rows)
        nondegenerate = grid.loc[grid['n_clusters'] >= 2]
        n_degenerate = int(len(grid) - len(nondegenerate))
        hdb_labels_all = np.full(len(work), -1, dtype=int)
        hdb_strength_all = np.full(len(work), np.nan)
        reference = None
        median_ari = np.nan
        validation_presence = {}
        all_recur = False
        assigned_fraction = 0.0
        hdb_time_v = np.nan
        if len(nondegenerate):
            score = nondegenerate['relative_validity'].fillna(nondegenerate['silhouette_nonnoise'])
            best = nondegenerate.loc[score.idxmax()]
            reference = (int(best['min_cluster_size']), int(best['min_samples']))
            ref_model, ref_labels = fits[reference]
            aris = []
            for _, row in nondegenerate.iterrows():
                check_cancelled(_acid_cancel)
                key = (int(row['min_cluster_size']), int(row['min_samples']))
                aris.append(adjusted_rand_score(ref_labels, fits[key][1]))
            grid['ARI_vs_reference'] = [adjusted_rand_score(ref_labels, fits[int(r['min_cluster_size']), int(r['min_samples'])][1]) if r['n_clusters'] >= 2 else np.nan for _, r in grid.iterrows()]
            median_ari = float(np.median(aris))
            labels_val, strength_val = hdbscan.approximate_predict(ref_model, X_val)
            hdb_labels_all[discovery] = ref_labels
            hdb_labels_all[validation] = labels_val.astype(int)
            hdb_strength_all[discovery] = ref_model.probabilities_
            hdb_strength_all[validation] = strength_val
            cluster_ids = sorted(set(ref_labels) - {-1})
            required_validation = max(10, int(math.ceil(0.01 * len(X_val))))
            validation_presence = {int(c): int(np.sum(labels_val == c)) for c in cluster_ids}
            all_recur = bool(cluster_ids and all((v >= required_validation for v in validation_presence.values())))
            assigned_fraction = float(np.mean(labels_val != -1))
            nonnoise = hdb_labels_all != -1
            if nonnoise.sum():
                hdb_time_v = cramers_v(pd.crosstab(hdb_labels_all[nonnoise], work.loc[nonnoise, 'time_block']))
        else:
            grid['ARI_vs_reference'] = np.nan
        hdb_candidate = bool(reference is not None and len(nondegenerate) >= 3 and all_recur and (assigned_fraction >= 0.5) and (median_ari >= 0.5) and (not np.isfinite(hdb_time_v) or hdb_time_v < 0.3))
        hdb_summary = characterize_groups(work, hdb_labels_all, split) if reference is not None else pd.DataFrame()
        gmm_rows, gmm_models = ([], {})
        for k in gmm_ks:
            check_cancelled(_acid_cancel)
            model = GaussianMixture(k, n_init=10, reg_covar=1e-06, random_state=seed).fit(X_disc)
            gmm_models[k] = model
            gmm_rows.append({'k': k, 'bic_discovery': model.bic(X_disc), 'heldout_loglik_per_event': model.score(X_val)})
        gmm_scores = pd.DataFrame(gmm_rows)
        gmm_scores['min_weight'] = [float(np.min(gmm_models[k].weights_)) for k in gmm_scores['k']]
        gmm_scores['admissible'] = gmm_scores['min_weight'] >= GMM_MIN_WEIGHT
        gmm_best_bic = int(gmm_scores.loc[gmm_scores['bic_discovery'].idxmin(), 'k'])
        gmm_best_val_any = int(gmm_scores.loc[gmm_scores['heldout_loglik_per_event'].idxmax(), 'k'])
        admissible = gmm_scores.loc[gmm_scores['admissible']]
        gmm_best_val = int(admissible.loc[admissible['heldout_loglik_per_event'].idxmax(), 'k']) if len(admissible) else gmm_best_val_any
        gmm_model = gmm_models[gmm_best_val]
        gmm_labels = gmm_model.predict(X).astype(int)
        gmm_post = gmm_model.predict_proba(X).max(axis=1)
        gmm_summary = characterize_groups(work, gmm_labels, split)
        gmm_summary.insert(1, 'weight', [float(np.mean(gmm_labels == g)) for g in gmm_summary['group']])
        gmm_time_v = cramers_v(pd.crosstab(gmm_labels, work['time_block'])) if gmm_best_val >= 2 else np.nan
        ll1 = float(gmm_scores.loc[gmm_scores['k'] == 1, 'heldout_loglik_per_event'].iloc[0])
        gmm_heldout_gain = float(gmm_scores.loc[gmm_scores['k'] == gmm_best_val, 'heldout_loglik_per_event'].iloc[0] - ll1)
        nonnoise = hdb_labels_all != -1
        gmm_hdb_ari = float(adjusted_rand_score(hdb_labels_all[nonnoise], gmm_labels[nonnoise])) if reference is not None and nonnoise.sum() >= 20 else np.nan
        work[f'{label}_hdbscan_cluster'] = hdb_labels_all
        work[f'{label}_hdbscan_strength'] = hdb_strength_all
        work[f'{label}_gmm_component'] = gmm_labels
        work[f'{label}_gmm_posterior'] = gmm_post
        work[f'{label}_split'] = split
        print('=' * 88)
        print(f'[{label}] {len(work)} events with every feature finite ({n_dropped} dropped); {int(discovery.sum())} discovery / {int(validation.sum())} validation')
        print(f'[{label}] features used ({len(features)}): {features}')
        if dropped_features:
            print(f'[{label}] dropped as redundant: ' + ', '.join((f'{a} (~{b}, rho={r:+.2f})' for a, b, r in dropped_features)))
        print(f'[{label}] PCA components retained: {n_pcs} (explained {pca.explained_variance_ratio_.sum():.3f})')
        display(loadings.round(3))
        print(f'[{label}] HDBSCAN grid: {len(nondegenerate)} non-degenerate / {n_degenerate} degenerate (0-1 clusters)')
        display(grid.round(4))
        if reference is not None:
            print(f'[{label}] reference solution min_cluster_size={reference[0]}, min_samples={reference[1]}; median ARI across non-degenerate solutions = {median_ari:.3f}')
            print(f"[{label}] validation assigned fraction {assigned_fraction:.3f}; validation counts by cluster {validation_presence}; time-block Cramér's V = {hdb_time_v:.3f}" if np.isfinite(hdb_time_v) else f'[{label}] validation assigned fraction {assigned_fraction:.3f}; validation counts by cluster {validation_presence}')
            display(hdb_summary.round(4))
        print(f'[{label}] HDBSCAN STABLE CLUSTER CANDIDATE: {hdb_candidate}')
        display(gmm_scores.round(4))
        print(f"[{label}] GMM best BIC k = {gmm_best_bic}; best held-out k = {gmm_best_val_any} unconstrained, {gmm_best_val} among admissible (min weight >= {GMM_MIN_WEIGHT}) -> characterized below (held-out gain vs k=1: {gmm_heldout_gain:+.3f} nats/event); time-block Cramér's V = {gmm_time_v:.3f}" + (f'; GMM/HDBSCAN ARI = {gmm_hdb_ari:.3f}' if np.isfinite(gmm_hdb_ari) else ''))
        display(gmm_summary.round(4))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, labels, title in [(axes[0], hdb_labels_all, f'{label}: HDBSCAN reference labels (-1 = noise)'), (axes[1], gmm_labels, f'{label}: GMM k={gmm_best_val} (held-out best)')]:
            check_cancelled(_acid_cancel)
            sc = ax.scatter(X[:, 0], X[:, 1], c=labels, s=9, alpha=0.6, cmap='tab10')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(title)
            plt.colorbar(sc, ax=ax)
        finish_figure(f'11_extended_{label}')
        return {'label': label, 'work': work, 'features': features, 'dropped_features': dropped_features, 'n_pcs': n_pcs, 'loadings': loadings, 'grid': grid, 'reference': reference, 'median_ari_nondegenerate': median_ari, 'n_degenerate': n_degenerate, 'validation_presence': validation_presence, 'assigned_fraction': assigned_fraction, 'hdbscan_time_cramers_v': hdb_time_v, 'hdbscan_candidate': hdb_candidate, 'hdbscan_summary': hdb_summary, 'gmm_scores': gmm_scores, 'gmm_best_bic_k': gmm_best_bic, 'gmm_best_heldout_k': gmm_best_val, 'gmm_best_heldout_k_unconstrained': gmm_best_val_any, 'gmm_heldout_gain_vs_1': gmm_heldout_gain, 'gmm_time_cramers_v': gmm_time_v, 'gmm_hdbscan_ari': gmm_hdb_ari, 'gmm_summary': gmm_summary}
    tier_a_features = list(dict.fromkeys(PHYSICAL_FEATURES + EXT_TIER_A_FEATURES))
    tier_b_features = list(dict.fromkeys(tier_a_features + EXT_TIER_B_FEATURES))
    _resolved = primary_residualized.loc[primary_residualized['n_samples'] >= EXT_SPECTRAL_MIN_SAMPLES]
    extended_results = {}
    _tier = cluster_feature_space(primary_residualized, tier_a_features, label='tierA_all_events')
    if _tier is not None:
        extended_results['tierA_all_events'] = _tier
    if len(_resolved) >= MORPH_MIN_EVENTS:
        _tier = cluster_feature_space(_resolved, tier_b_features, label='tierB_resolved_events')
        if _tier is not None:
            extended_results['tierB_resolved_events'] = _tier
    else:
        print(f'Tier B skipped: only {len(_resolved)} events reach EXT_SPECTRAL_MIN_SAMPLES (need {MORPH_MIN_EVENTS}).')
    for name, result in extended_results.items():
        check_cancelled(_acid_cancel)
        label_cols = [c for c in result['work'].columns if c.startswith(f'{name}_')]
        try:
            result['work'][['event_id'] + label_cols].to_parquet(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_{name}_labels.parquet', index=False)
        except ImportError:
            result['work'][['event_id'] + label_cols].to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_{name}_labels.csv', index=False)
        result['grid'].to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_{name}_hdbscan_grid.csv', index=False)
        result['gmm_summary'].to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_{name}_gmm_components.csv', index=False)
    print('Saved extended-tier labels, grids, and component tables to', OUT_DIR)


def definitions_062():
    global characterize_groups, cluster_feature_space, drop_redundant_features

    def drop_redundant_features(frame: pd.DataFrame, features: list[str], rho: float) -> tuple[list[str], list[tuple[str, str, float]]]:
        corr = frame[features].corr(method='spearman', min_periods=30)
        kept = []
        dropped = []
        for feature in features:
            check_cancelled(_acid_cancel)
            partner = next((k for k in kept if np.isfinite(corr.loc[feature, k]) and abs(corr.loc[feature, k]) >= rho), None)
            if partner is None:
                kept.append(feature)
            else:
                dropped.append((feature, partner, float(corr.loc[feature, partner])))
        return (kept, dropped)

    def characterize_groups(frame: pd.DataFrame, labels: np.ndarray, split: np.ndarray) -> pd.DataFrame:
        columns = [c for c in CHARACTERIZATION_COLUMNS if c in frame]
        rows = []
        for group in sorted(set(labels) - {-1}):
            check_cancelled(_acid_cancel)
            m = labels == group
            g = frame.loc[m]
            row = {'group': int(group), 'n_total': int(m.sum()), 'n_discovery': int((m & (split == 'discovery')).sum()), 'n_validation': int((m & (split == 'validation')).sum()), 'blocks_present': int(g['time_block'].nunique())}
            row.update({f'{c}_median': g[c].median() for c in columns})
            rows.append(row)
        return pd.DataFrame(rows)

    def cluster_feature_space(frame: pd.DataFrame, features: list[str], *, label: str, min_cluster_fractions=HDBSCAN_MIN_CLUSTER_FRACTIONS, min_samples_grid=EXT_MIN_SAMPLES_GRID, min_cluster_floor=HDBSCAN_MIN_CLUSTER_FLOOR, pca_variance: float=0.95, gmm_ks=(1, 2, 3, 4), seed: int=SEED) -> dict:
        features = list(dict.fromkeys(features))
        work = frame.copy().reset_index(drop=True)
        X_raw = work[features].replace([np.inf, -np.inf], np.nan)
        complete = X_raw.notna().all(axis=1).to_numpy()
        n_dropped = int((~complete).sum())
        work = work.loc[complete].reset_index(drop=True)
        X_raw = X_raw.loc[complete].reset_index(drop=True)
        discovery = work['time_block'].to_numpy() % 2 == 0
        validation = ~discovery
        if discovery.sum() < 30 or validation.sum() < 15:
            print(f'[{label}] skipped: {int(discovery.sum())} discovery / {int(validation.sum())} validation events with every feature finite ({n_dropped} dropped for a non-finite feature); a tier needs >= 30 / 15.')
            return None
        features, dropped_features = drop_redundant_features(X_raw.loc[discovery], features, REDUNDANCY_RHO)
        X_raw = X_raw[features]
        scaler = RobustScaler().fit(X_raw.loc[discovery])
        X_scaled = np.clip(scaler.transform(X_raw), -SCALED_CLIP, SCALED_CLIP)
        pca_probe = PCA(n_components=min(len(features), int(discovery.sum()) - 1), random_state=seed).fit(X_scaled[discovery])
        if pca_probe.n_components_ < 2:
            # np.clip(a, a_min, a_max) with a_min > a_max returns a_max, not the floor --
            # so clipping to (2, pca_probe.n_components_) below would silently return 1,
            # not 2, whenever only 1 feature survives drop_redundant_features. The 2D
            # scatter/plot further down unconditionally indexes a second PCA column, so
            # this tier cannot proceed with fewer than 2 components; skip it like the
            # other insufficient-data cases above.
            print(f'[{label}] skipped: only {pca_probe.n_components_} PCA component available after redundancy pruning ({len(features)} feature(s)); need >= 2.')
            return None
        cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
        n_pcs = int(np.clip(np.searchsorted(cumulative, pca_variance) + 1, 2, pca_probe.n_components_))
        pca = PCA(n_components=n_pcs, random_state=seed).fit(X_scaled[discovery])
        X = pca.transform(X_scaled)
        X_disc, X_val = (X[discovery], X[validation])
        loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f'PC{i + 1}' for i in range(n_pcs)])
        split = np.where(discovery, 'discovery', 'validation')
        grid_rows, fits = ([], {})
        for frac in min_cluster_fractions:
            check_cancelled(_acid_cancel)
            mcs = max(min_cluster_floor, int(round(frac * len(X_disc))))
            for min_samples in min_samples_grid:
                check_cancelled(_acid_cancel)
                mcs_fit, ms_fit = (int(min(mcs, max(2, len(X_disc) - 1))), int(min(int(min_samples), max(1, len(X_disc) - 1))))
                key = (mcs_fit, ms_fit)
                if key in fits:
                    continue
                model = hdbscan.HDBSCAN(min_cluster_size=mcs_fit, min_samples=ms_fit, prediction_data=True, gen_min_span_tree=True).fit(X_disc)
                labels = model.labels_.astype(int)
                n_clusters, noise_fraction, silhouette = hdbscan_metrics(labels, X_disc)
                try:
                    relative_validity = float(model.relative_validity_)
                except Exception:
                    relative_validity = np.nan
                fits[key] = (model, labels)
                grid_rows.append({'min_cluster_size': mcs_fit, 'min_samples': ms_fit, 'n_clusters': n_clusters, 'noise_fraction': noise_fraction, 'silhouette_nonnoise': silhouette, 'relative_validity': relative_validity})
        grid = pd.DataFrame(grid_rows)
        nondegenerate = grid.loc[grid['n_clusters'] >= 2]
        n_degenerate = int(len(grid) - len(nondegenerate))
        hdb_labels_all = np.full(len(work), -1, dtype=int)
        hdb_strength_all = np.full(len(work), np.nan)
        reference = None
        median_ari = np.nan
        validation_presence = {}
        all_recur = False
        assigned_fraction = 0.0
        hdb_time_v = np.nan
        if len(nondegenerate):
            score = nondegenerate['relative_validity'].fillna(nondegenerate['silhouette_nonnoise'])
            best = nondegenerate.loc[score.idxmax()]
            reference = (int(best['min_cluster_size']), int(best['min_samples']))
            ref_model, ref_labels = fits[reference]
            aris = []
            for _, row in nondegenerate.iterrows():
                check_cancelled(_acid_cancel)
                key = (int(row['min_cluster_size']), int(row['min_samples']))
                aris.append(adjusted_rand_score(ref_labels, fits[key][1]))
            grid['ARI_vs_reference'] = [adjusted_rand_score(ref_labels, fits[int(r['min_cluster_size']), int(r['min_samples'])][1]) if r['n_clusters'] >= 2 else np.nan for _, r in grid.iterrows()]
            median_ari = float(np.median(aris))
            labels_val, strength_val = hdbscan.approximate_predict(ref_model, X_val)
            hdb_labels_all[discovery] = ref_labels
            hdb_labels_all[validation] = labels_val.astype(int)
            hdb_strength_all[discovery] = ref_model.probabilities_
            hdb_strength_all[validation] = strength_val
            cluster_ids = sorted(set(ref_labels) - {-1})
            required_validation = max(10, int(math.ceil(0.01 * len(X_val))))
            validation_presence = {int(c): int(np.sum(labels_val == c)) for c in cluster_ids}
            all_recur = bool(cluster_ids and all((v >= required_validation for v in validation_presence.values())))
            assigned_fraction = float(np.mean(labels_val != -1))
            nonnoise = hdb_labels_all != -1
            if nonnoise.sum():
                hdb_time_v = cramers_v(pd.crosstab(hdb_labels_all[nonnoise], work.loc[nonnoise, 'time_block']))
        else:
            grid['ARI_vs_reference'] = np.nan
        hdb_candidate = bool(reference is not None and len(nondegenerate) >= 3 and all_recur and (assigned_fraction >= 0.5) and (median_ari >= 0.5) and (not np.isfinite(hdb_time_v) or hdb_time_v < 0.3))
        hdb_summary = characterize_groups(work, hdb_labels_all, split) if reference is not None else pd.DataFrame()
        gmm_rows, gmm_models = ([], {})
        for k in gmm_ks:
            check_cancelled(_acid_cancel)
            model = GaussianMixture(k, n_init=10, reg_covar=1e-06, random_state=seed).fit(X_disc)
            gmm_models[k] = model
            gmm_rows.append({'k': k, 'bic_discovery': model.bic(X_disc), 'heldout_loglik_per_event': model.score(X_val)})
        gmm_scores = pd.DataFrame(gmm_rows)
        gmm_scores['min_weight'] = [float(np.min(gmm_models[k].weights_)) for k in gmm_scores['k']]
        gmm_scores['admissible'] = gmm_scores['min_weight'] >= GMM_MIN_WEIGHT
        gmm_best_bic = int(gmm_scores.loc[gmm_scores['bic_discovery'].idxmin(), 'k'])
        gmm_best_val_any = int(gmm_scores.loc[gmm_scores['heldout_loglik_per_event'].idxmax(), 'k'])
        admissible = gmm_scores.loc[gmm_scores['admissible']]
        gmm_best_val = int(admissible.loc[admissible['heldout_loglik_per_event'].idxmax(), 'k']) if len(admissible) else gmm_best_val_any
        gmm_model = gmm_models[gmm_best_val]
        gmm_labels = gmm_model.predict(X).astype(int)
        gmm_post = gmm_model.predict_proba(X).max(axis=1)
        gmm_summary = characterize_groups(work, gmm_labels, split)
        gmm_summary.insert(1, 'weight', [float(np.mean(gmm_labels == g)) for g in gmm_summary['group']])
        gmm_time_v = cramers_v(pd.crosstab(gmm_labels, work['time_block'])) if gmm_best_val >= 2 else np.nan
        ll1 = float(gmm_scores.loc[gmm_scores['k'] == 1, 'heldout_loglik_per_event'].iloc[0])
        gmm_heldout_gain = float(gmm_scores.loc[gmm_scores['k'] == gmm_best_val, 'heldout_loglik_per_event'].iloc[0] - ll1)
        nonnoise = hdb_labels_all != -1
        gmm_hdb_ari = float(adjusted_rand_score(hdb_labels_all[nonnoise], gmm_labels[nonnoise])) if reference is not None and nonnoise.sum() >= 20 else np.nan
        work[f'{label}_hdbscan_cluster'] = hdb_labels_all
        work[f'{label}_hdbscan_strength'] = hdb_strength_all
        work[f'{label}_gmm_component'] = gmm_labels
        work[f'{label}_gmm_posterior'] = gmm_post
        work[f'{label}_split'] = split
        print('=' * 88)
        print(f'[{label}] {len(work)} events with every feature finite ({n_dropped} dropped); {int(discovery.sum())} discovery / {int(validation.sum())} validation')
        print(f'[{label}] features used ({len(features)}): {features}')
        if dropped_features:
            print(f'[{label}] dropped as redundant: ' + ', '.join((f'{a} (~{b}, rho={r:+.2f})' for a, b, r in dropped_features)))
        print(f'[{label}] PCA components retained: {n_pcs} (explained {pca.explained_variance_ratio_.sum():.3f})')
        display(loadings.round(3))
        print(f'[{label}] HDBSCAN grid: {len(nondegenerate)} non-degenerate / {n_degenerate} degenerate (0-1 clusters)')
        display(grid.round(4))
        if reference is not None:
            print(f'[{label}] reference solution min_cluster_size={reference[0]}, min_samples={reference[1]}; median ARI across non-degenerate solutions = {median_ari:.3f}')
            print(f"[{label}] validation assigned fraction {assigned_fraction:.3f}; validation counts by cluster {validation_presence}; time-block Cramér's V = {hdb_time_v:.3f}" if np.isfinite(hdb_time_v) else f'[{label}] validation assigned fraction {assigned_fraction:.3f}; validation counts by cluster {validation_presence}')
            display(hdb_summary.round(4))
        print(f'[{label}] HDBSCAN STABLE CLUSTER CANDIDATE: {hdb_candidate}')
        display(gmm_scores.round(4))
        print(f"[{label}] GMM best BIC k = {gmm_best_bic}; best held-out k = {gmm_best_val_any} unconstrained, {gmm_best_val} among admissible (min weight >= {GMM_MIN_WEIGHT}) -> characterized below (held-out gain vs k=1: {gmm_heldout_gain:+.3f} nats/event); time-block Cramér's V = {gmm_time_v:.3f}" + (f'; GMM/HDBSCAN ARI = {gmm_hdb_ari:.3f}' if np.isfinite(gmm_hdb_ari) else ''))
        display(gmm_summary.round(4))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, labels, title in [(axes[0], hdb_labels_all, f'{label}: HDBSCAN reference labels (-1 = noise)'), (axes[1], gmm_labels, f'{label}: GMM k={gmm_best_val} (held-out best)')]:
            check_cancelled(_acid_cancel)
            sc = ax.scatter(X[:, 0], X[:, 1], c=labels, s=9, alpha=0.6, cmap='tab10')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(title)
            plt.colorbar(sc, ax=ax)
        finish_figure(f'11_extended_{label}')
        return {'label': label, 'work': work, 'features': features, 'dropped_features': dropped_features, 'n_pcs': n_pcs, 'loadings': loadings, 'grid': grid, 'reference': reference, 'median_ari_nondegenerate': median_ari, 'n_degenerate': n_degenerate, 'validation_presence': validation_presence, 'assigned_fraction': assigned_fraction, 'hdbscan_time_cramers_v': hdb_time_v, 'hdbscan_candidate': hdb_candidate, 'hdbscan_summary': hdb_summary, 'gmm_scores': gmm_scores, 'gmm_best_bic_k': gmm_best_bic, 'gmm_best_heldout_k': gmm_best_val, 'gmm_best_heldout_k_unconstrained': gmm_best_val_any, 'gmm_heldout_gain_vs_1': gmm_heldout_gain, 'gmm_time_cramers_v': gmm_time_v, 'gmm_hdbscan_ari': gmm_hdb_ari, 'gmm_summary': gmm_summary}


# Original notebook cell 63
def stage_063():
    global plot_event_gallery, stratified_sample_rows
    check_cancelled(_acid_cancel)

    def stratified_sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        selected = []
        blocks = list(sorted(frame['time_block'].unique()))
        while len(selected) < min(n, len(frame)):
            check_cancelled(_acid_cancel)
            progressed = False
            for block in blocks:
                check_cancelled(_acid_cancel)
                candidates = frame.loc[(frame['time_block'] == block) & ~frame.index.isin(selected)]
                if len(candidates):
                    selected.append(int(rng.choice(candidates.index.to_numpy())))
                    progressed = True
                    if len(selected) >= min(n, len(frame)):
                        break
            if not progressed:
                break
        return frame.loc[selected]

    def plot_event_gallery(frame: pd.DataFrame, label_col: str, prefix: str, n_per_group: int=12, pad_ms: float=1.5) -> None:
        groups = sorted((g for g in frame[label_col].dropna().unique() if g != -1))
        pad_samples = int(round(pad_ms * fs / 1000.0))
        for group in groups:
            check_cancelled(_acid_cancel)
            g = frame.loc[frame[label_col] == group]
            selected = stratified_sample_rows(g, n_per_group, SEED + int(group) + 100)
            cols = 4
            rows = int(math.ceil(len(selected) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
            for ax in axes.ravel():
                check_cancelled(_acid_cancel)
                ax.axis('off')
            for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
                check_cancelled(_acid_cancel)
                ax.axis('on')
                start, end = (int(row.start_idx), int(row.end_idx))
                lo, hi = (max(0, start - pad_samples), min(n_samples, end + 1 + pad_samples))
                t = (np.arange(lo, hi) - start) / fs * 1000.0
                ax.plot(t, analysis_signal[lo:hi], linewidth=0.7)
                ax.axhline(row.amplitude_baseline_pA, linestyle=':', linewidth=0.8)
                ax.axvspan(0, (end - start + 1) / fs * 1000.0, alpha=0.16)
                ax.set_title(f'{row.event_id} | block {int(row.time_block)}\n{row.dwell_ms:.3f} ms, {getattr(row, AMP_SIGMA_COL):.1f}σ {AMPLITUDE_STATISTIC}', fontsize=8)
                ax.set_xlabel('ms from onset', fontsize=8)
                ax.set_ylabel('pA', fontsize=8)
            fig.suptitle(f'{prefix} {group}: random events spanning time blocks')
            finish_figure(f"07_{prefix.lower().replace(' ', '_')}_{group}")
    plot_event_gallery(primary_residualized, 'scalar_component', 'Scalar component')
    if n_physical_clusters >= 2:
        plot_event_gallery(physical_work, 'physical_cluster', 'Physical cluster')


def definitions_063():
    global plot_event_gallery, stratified_sample_rows

    def stratified_sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        selected = []
        blocks = list(sorted(frame['time_block'].unique()))
        while len(selected) < min(n, len(frame)):
            check_cancelled(_acid_cancel)
            progressed = False
            for block in blocks:
                check_cancelled(_acid_cancel)
                candidates = frame.loc[(frame['time_block'] == block) & ~frame.index.isin(selected)]
                if len(candidates):
                    selected.append(int(rng.choice(candidates.index.to_numpy())))
                    progressed = True
                    if len(selected) >= min(n, len(frame)):
                        break
            if not progressed:
                break
        return frame.loc[selected]

    def plot_event_gallery(frame: pd.DataFrame, label_col: str, prefix: str, n_per_group: int=12, pad_ms: float=1.5) -> None:
        groups = sorted((g for g in frame[label_col].dropna().unique() if g != -1))
        pad_samples = int(round(pad_ms * fs / 1000.0))
        for group in groups:
            check_cancelled(_acid_cancel)
            g = frame.loc[frame[label_col] == group]
            selected = stratified_sample_rows(g, n_per_group, SEED + int(group) + 100)
            cols = 4
            rows = int(math.ceil(len(selected) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
            for ax in axes.ravel():
                check_cancelled(_acid_cancel)
                ax.axis('off')
            for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
                check_cancelled(_acid_cancel)
                ax.axis('on')
                start, end = (int(row.start_idx), int(row.end_idx))
                lo, hi = (max(0, start - pad_samples), min(n_samples, end + 1 + pad_samples))
                t = (np.arange(lo, hi) - start) / fs * 1000.0
                ax.plot(t, analysis_signal[lo:hi], linewidth=0.7)
                ax.axhline(row.amplitude_baseline_pA, linestyle=':', linewidth=0.8)
                ax.axvspan(0, (end - start + 1) / fs * 1000.0, alpha=0.16)
                ax.set_title(f'{row.event_id} | block {int(row.time_block)}\n{row.dwell_ms:.3f} ms, {getattr(row, AMP_SIGMA_COL):.1f}σ {AMPLITUDE_STATISTIC}', fontsize=8)
                ax.set_xlabel('ms from onset', fontsize=8)
                ax.set_ylabel('pA', fontsize=8)
            fig.suptitle(f'{prefix} {group}: random events spanning time blocks')
            finish_figure(f"07_{prefix.lower().replace(' ', '_')}_{group}")


# Original notebook cell 65
def stage_065():
    global extract_morphology, first_crossing_down, first_crossing_up, morphology, morphology_waveforms, two_level_candidate
    check_cancelled(_acid_cancel)

    def first_crossing_up(x: np.ndarray, level: float) -> float:
        idx = np.flatnonzero(x >= level)
        return float(idx[0]) if len(idx) else np.nan

    def first_crossing_down(x: np.ndarray, level: float) -> float:
        idx = np.flatnonzero(x <= level)
        return float(idx[0]) if len(idx) else np.nan

    def two_level_candidate(body_normalized: np.ndarray) -> tuple[bool, float, float, float]:
        x = np.asarray(body_normalized, float)
        if len(x) < 25 or np.unique(x).size < 5:
            return (False, np.nan, np.nan, np.nan)
        x2 = x.reshape(-1, 1)
        g1 = GaussianMixture(1, n_init=5, reg_covar=1e-05, random_state=SEED).fit(x2)
        g2 = GaussianMixture(2, n_init=10, reg_covar=1e-05, random_state=SEED).fit(x2)
        means, stds, weights, _ = ordered_gmm_parameters(g2)
        delta_bic = float(g1.bic(x2) - g2.bic(x2))
        separation = adjacent_separation(means, stds)
        min_weight = float(np.min(weights))
        candidate = bool(delta_bic > 10 and separation > 1.5 and (min_weight > 0.15))
        return (candidate, delta_bic, separation, min_weight)

    def extract_morphology(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        min_samples = int(math.ceil(MORPH_MIN_DWELL_MS * fs / 1000.0))
        eligible = frame.loc[frame['n_samples'] >= min_samples].copy()
        waveforms = []
        rows = []
        for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc='Morphology'):
            check_cancelled(_acid_cancel)
            start, end = (int(row.start_idx), int(row.end_idx))
            depth = row.amplitude_baseline_pA - analysis_signal[start:end + 1].astype(float)
            positive = depth[depth > 0]
            if len(positive) < 4:
                continue
            scale = float(np.quantile(positive, 0.9))
            if not np.isfinite(scale) or scale <= 0:
                continue
            body = np.clip(depth / scale, -0.25, 2.0)
            waveform = np.interp(np.linspace(0, len(body) - 1, MORPH_POINTS), np.arange(len(body)), body)
            peak = int(np.argmax(waveform))
            pre = waveform[:peak + 1]
            post = waveform[peak:]
            r10 = first_crossing_up(pre, 0.1)
            r90 = first_crossing_up(pre, 0.9)
            f90 = first_crossing_down(post, 0.9)
            f10 = first_crossing_down(post, 0.1)
            rise_fraction = (r90 - r10) / max(MORPH_POINTS - 1, 1) if np.isfinite(r10 + r90) and r90 >= r10 else np.nan
            fall_fraction = (f10 - f90) / max(MORPH_POINTS - 1, 1) if np.isfinite(f90 + f10) and f10 >= f90 else np.nan
            asymmetry = (fall_fraction - rise_fraction) / (fall_fraction + rise_fraction) if np.isfinite(rise_fraction) and np.isfinite(fall_fraction) and (rise_fraction + fall_fraction > 0) else np.nan
            weights = np.clip(waveform, 0, None)
            center_of_mass = float(np.sum(np.arange(MORPH_POINTS) * weights) / max(np.sum(weights), 1e-12))
            com_asymmetry = (center_of_mass - (MORPH_POINTS - 1) / 2) / max((MORPH_POINTS - 1) / 2, 1)
            plateau_fraction = float(np.mean(waveform >= 0.8))
            core = waveform[waveform >= 0.5]
            intra_variability = float(np.std(core)) if len(core) >= 4 else np.nan
            multilevel, delta_bic, level_sep, level_min_weight = two_level_candidate(body)
            waveforms.append(waveform.astype(np.float32))
            rows.append({'event_id': row.event_id, 'time_block': int(row.time_block), 'dwell_ms': row.dwell_ms, 'rise_fraction': rise_fraction, 'fall_fraction': fall_fraction, 'edge_asymmetry': asymmetry, 'center_of_mass_asymmetry': com_asymmetry, 'plateau_fraction': plateau_fraction, 'intra_variability': intra_variability, 'two_level_candidate': multilevel, 'two_level_delta_bic': delta_bic, 'two_level_separation': level_sep, 'two_level_min_weight': level_min_weight})
        return (pd.DataFrame(rows), np.asarray(waveforms, np.float32))
    if RUN_MORPHOLOGY:
        morphology, morphology_waveforms = extract_morphology(primary_residualized)
        print(f'Morphology-eligible usable events: {len(morphology):,} / {len(primary_residualized):,}')
    else:
        morphology = pd.DataFrame()
        morphology_waveforms = np.empty((0, MORPH_POINTS), np.float32)
        print('Morphology disabled in configuration.')


def definitions_065():
    global extract_morphology, first_crossing_down, first_crossing_up, two_level_candidate

    def first_crossing_up(x: np.ndarray, level: float) -> float:
        idx = np.flatnonzero(x >= level)
        return float(idx[0]) if len(idx) else np.nan

    def first_crossing_down(x: np.ndarray, level: float) -> float:
        idx = np.flatnonzero(x <= level)
        return float(idx[0]) if len(idx) else np.nan

    def two_level_candidate(body_normalized: np.ndarray) -> tuple[bool, float, float, float]:
        x = np.asarray(body_normalized, float)
        if len(x) < 25 or np.unique(x).size < 5:
            return (False, np.nan, np.nan, np.nan)
        x2 = x.reshape(-1, 1)
        g1 = GaussianMixture(1, n_init=5, reg_covar=1e-05, random_state=SEED).fit(x2)
        g2 = GaussianMixture(2, n_init=10, reg_covar=1e-05, random_state=SEED).fit(x2)
        means, stds, weights, _ = ordered_gmm_parameters(g2)
        delta_bic = float(g1.bic(x2) - g2.bic(x2))
        separation = adjacent_separation(means, stds)
        min_weight = float(np.min(weights))
        candidate = bool(delta_bic > 10 and separation > 1.5 and (min_weight > 0.15))
        return (candidate, delta_bic, separation, min_weight)

    def extract_morphology(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        min_samples = int(math.ceil(MORPH_MIN_DWELL_MS * fs / 1000.0))
        eligible = frame.loc[frame['n_samples'] >= min_samples].copy()
        waveforms = []
        rows = []
        for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc='Morphology'):
            check_cancelled(_acid_cancel)
            start, end = (int(row.start_idx), int(row.end_idx))
            depth = row.amplitude_baseline_pA - analysis_signal[start:end + 1].astype(float)
            positive = depth[depth > 0]
            if len(positive) < 4:
                continue
            scale = float(np.quantile(positive, 0.9))
            if not np.isfinite(scale) or scale <= 0:
                continue
            body = np.clip(depth / scale, -0.25, 2.0)
            waveform = np.interp(np.linspace(0, len(body) - 1, MORPH_POINTS), np.arange(len(body)), body)
            peak = int(np.argmax(waveform))
            pre = waveform[:peak + 1]
            post = waveform[peak:]
            r10 = first_crossing_up(pre, 0.1)
            r90 = first_crossing_up(pre, 0.9)
            f90 = first_crossing_down(post, 0.9)
            f10 = first_crossing_down(post, 0.1)
            rise_fraction = (r90 - r10) / max(MORPH_POINTS - 1, 1) if np.isfinite(r10 + r90) and r90 >= r10 else np.nan
            fall_fraction = (f10 - f90) / max(MORPH_POINTS - 1, 1) if np.isfinite(f90 + f10) and f10 >= f90 else np.nan
            asymmetry = (fall_fraction - rise_fraction) / (fall_fraction + rise_fraction) if np.isfinite(rise_fraction) and np.isfinite(fall_fraction) and (rise_fraction + fall_fraction > 0) else np.nan
            weights = np.clip(waveform, 0, None)
            center_of_mass = float(np.sum(np.arange(MORPH_POINTS) * weights) / max(np.sum(weights), 1e-12))
            com_asymmetry = (center_of_mass - (MORPH_POINTS - 1) / 2) / max((MORPH_POINTS - 1) / 2, 1)
            plateau_fraction = float(np.mean(waveform >= 0.8))
            core = waveform[waveform >= 0.5]
            intra_variability = float(np.std(core)) if len(core) >= 4 else np.nan
            multilevel, delta_bic, level_sep, level_min_weight = two_level_candidate(body)
            waveforms.append(waveform.astype(np.float32))
            rows.append({'event_id': row.event_id, 'time_block': int(row.time_block), 'dwell_ms': row.dwell_ms, 'rise_fraction': rise_fraction, 'fall_fraction': fall_fraction, 'edge_asymmetry': asymmetry, 'center_of_mass_asymmetry': com_asymmetry, 'plateau_fraction': plateau_fraction, 'intra_variability': intra_variability, 'two_level_candidate': multilevel, 'two_level_delta_bic': delta_bic, 'two_level_separation': level_sep, 'two_level_min_weight': level_min_weight})
        return (pd.DataFrame(rows), np.asarray(waveforms, np.float32))


# Original notebook cell 66
def stage_066():
    global PCA_MORPH, W_centered, Z_morph, _physical_map, cluster, common, g, mlabels, mlabels_disc, mlabels_val, morph_assigned, morph_center, morph_clusters, morph_cum, morph_disc, morph_mcs, morph_model, morph_physical_ari, morph_recur, morph_required_val, morph_time_cramers_v, morph_val, morphology_cluster_candidate, morphology_cluster_summary, mstrength, mstrength_disc, mstrength_val, mt, n_morph_pcs, pca_morph_full, rows
    check_cancelled(_acid_cancel)
    morphology_cluster_candidate = False
    morphology_cluster_summary = pd.DataFrame()
    morph_time_cramers_v = np.nan
    morph_physical_ari = np.nan
    if len(morphology) >= MORPH_MIN_EVENTS:
        morph_disc = morphology['time_block'].to_numpy() % 2 == 0
        morph_val = ~morph_disc
        if morph_disc.sum() >= 50 and morph_val.sum() >= 20:
            morph_center = StandardScaler(with_std=False).fit(morphology_waveforms[morph_disc])
            W_centered = morph_center.transform(morphology_waveforms)
            pca_morph_full = PCA(n_components=min(10, MORPH_POINTS, morph_disc.sum() - 1), random_state=SEED).fit(W_centered[morph_disc])
            morph_cum = np.cumsum(pca_morph_full.explained_variance_ratio_)
            n_morph_pcs = max(2, int(np.searchsorted(morph_cum, 0.9) + 1))
            n_morph_pcs = min(n_morph_pcs, pca_morph_full.n_components_)
            PCA_MORPH = PCA(n_components=n_morph_pcs, random_state=SEED).fit(W_centered[morph_disc])
            Z_morph = PCA_MORPH.transform(W_centered)
            morph_mcs = max(20, int(round(0.03 * morph_disc.sum())))
            morph_model = hdbscan.HDBSCAN(min_cluster_size=morph_mcs, min_samples=max(10, morph_mcs // 4), prediction_data=True).fit(Z_morph[morph_disc])
            mlabels_disc = morph_model.labels_.astype(int)
            mstrength_disc = morph_model.probabilities_.astype(float)
            if len(set(mlabels_disc) - {-1}) > 0:
                mlabels_val, mstrength_val = hdbscan.approximate_predict(morph_model, Z_morph[morph_val])
                mlabels_val = mlabels_val.astype(int)
                mstrength_val = mstrength_val.astype(float)
            else:
                mlabels_val = np.full(int(morph_val.sum()), -1, dtype=int)
                mstrength_val = np.zeros(int(morph_val.sum()), dtype=float)
                print('Morphology HDBSCAN found no discovery cluster; validation events remain unassigned.')
            mlabels = np.full(len(morphology), -1, dtype=int)
            mstrength = np.full(len(morphology), np.nan, dtype=float)
            mlabels[morph_disc] = mlabels_disc
            mlabels[morph_val] = mlabels_val
            mstrength[morph_disc] = mstrength_disc
            mstrength[morph_val] = mstrength_val
            morphology['morphology_cluster'] = mlabels
            morphology['morphology_cluster_strength'] = mstrength
            morphology['split'] = np.where(morph_disc, 'discovery', 'validation')
            morph_clusters = sorted(set(mlabels_disc) - {-1})
            morph_required_val = max(5, int(math.ceil(0.01 * morph_val.sum())))
            morph_recur = bool(morph_clusters and all((np.sum(mlabels_val == c) >= morph_required_val for c in morph_clusters)))
            morph_assigned = float(np.mean(mlabels_val != -1))
            rows = []
            for cluster in sorted(set(mlabels) - {-1}):
                check_cancelled(_acid_cancel)
                g = morphology.loc[morphology['morphology_cluster'] == cluster]
                rows.append({'cluster': int(cluster), 'n_total': len(g), 'n_discovery': int(np.sum(g['split'] == 'discovery')), 'n_validation': int(np.sum(g['split'] == 'validation')), 'blocks_present': int(g['time_block'].nunique()), 'dwell_median_ms': g['dwell_ms'].median(), 'edge_asymmetry_median': g['edge_asymmetry'].median(), 'plateau_fraction_median': g['plateau_fraction'].median(), 'two_level_fraction': g['two_level_candidate'].mean()})
            morphology_cluster_summary = pd.DataFrame(rows)
            mt = pd.crosstab(morphology.loc[morphology['morphology_cluster'] != -1, 'morphology_cluster'], morphology.loc[morphology['morphology_cluster'] != -1, 'time_block'])
            morph_time_cramers_v = cramers_v(mt)
            morphology_cluster_candidate = bool(len(morph_clusters) >= 2 and morph_recur and (morph_assigned >= 0.5) and (not np.isfinite(morph_time_cramers_v) or morph_time_cramers_v < 0.3))
            _physical_map = physical_work.set_index('event_id')['physical_cluster']
            morphology['physical_cluster'] = morphology['event_id'].map(_physical_map)
            common = (morphology['morphology_cluster'] != -1) & (morphology['physical_cluster'].fillna(-1) != -1)
            if common.sum() >= 20 and morphology.loc[common, 'morphology_cluster'].nunique() >= 2:
                morph_physical_ari = adjusted_rand_score(morphology.loc[common, 'morphology_cluster'], morphology.loc[common, 'physical_cluster'])
            print('Morphology PCs retained:', n_morph_pcs)
            print('Morphology discovery clusters:', len(morph_clusters))
            print('Morphology validation assigned fraction:', morph_assigned)
            print("Morphology cluster × time Cramér's V:", morph_time_cramers_v)
            print('Morphology/physical ARI:', morph_physical_ari)
            print('STABLE LONG-EVENT MORPHOLOGY CANDIDATE:', morphology_cluster_candidate)
            display(morphology_cluster_summary.round(4))
        else:
            print('Long events exist, but one discovery/validation side is too small for morphology clustering.')
    else:
        print(f'Need at least {MORPH_MIN_EVENTS} usable long events for morphology clustering.')


# Original notebook cell 67
def stage_067():
    global axes, cluster, fig, mask, mean, morphology_gallery, scatter, sem, x_norm
    check_cancelled(_acid_cancel)
    if len(morphology) >= MORPH_MIN_EVENTS and 'morphology_cluster' in morphology:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(np.cumsum(PCA_MORPH.explained_variance_ratio_), marker='o')
        axes[0].axhline(0.9, linestyle=':')
        axes[0].set_xlabel('Number of morphology PCs')
        axes[0].set_ylabel('Cumulative explained variance')
        axes[0].set_title('Long-event waveform PCA')
        scatter = axes[1].scatter(Z_morph[:, 0], Z_morph[:, 1], c=morphology['morphology_cluster'], s=12, alpha=0.6, cmap='tab10')
        axes[1].set_xlabel('Morphology PC1')
        axes[1].set_ylabel('Morphology PC2')
        axes[1].set_title('Held-out-validated morphology labels')
        plt.colorbar(scatter, ax=axes[1], label='morphology cluster')
        finish_figure('08_morphology_pca')
        x_norm = np.linspace(0, 1, MORPH_POINTS)
        plt.figure(figsize=(9, 5))
        for cluster in sorted(set(morphology['morphology_cluster']) - {-1}):
            check_cancelled(_acid_cancel)
            mask = morphology['morphology_cluster'].to_numpy() == cluster
            mean = morphology_waveforms[mask].mean(axis=0)
            sem = morphology_waveforms[mask].std(axis=0) / math.sqrt(max(mask.sum(), 1))
            plt.plot(x_norm, mean, linewidth=2, label=f'cluster {cluster} (n={mask.sum()})')
            plt.fill_between(x_norm, mean - sem, mean + sem, alpha=0.15)
        plt.xlabel('Normalized event time')
        plt.ylabel('Locally normalized depth')
        plt.title('Mean long-event waveform by morphology cluster')
        plt.legend(fontsize=8)
        finish_figure('09_morphology_mean_waveforms')
        morphology_gallery = morphology.merge(primary_residualized[['event_id', 'start_idx', 'end_idx', 'local_baseline_detr_pA', 'amplitude_baseline_pA', AMP_SIGMA_COL]].loc[:, lambda d: ~d.columns.duplicated()], on='event_id', how='left')
        plot_event_gallery(morphology_gallery, 'morphology_cluster', 'Morphology cluster')


# Original notebook cell 69
def stage_069():
    global Z_latent, Zs, _, ax, axes, batch_meta, batch_windows, bottleneck, center, ds_start, ds_stop, end, exc, extractor, fig, flush_latent_batch, half, label_col, latent_embedding, latent_event_ids, latent_events, latent_length, latent_pca, latent_summary, latent_vectors, local_start, local_stop, pool, row, sc, sig_norm, start, title, window_end, window_start
    check_cancelled(_acid_cancel)
    latent_summary = {}
    if RUN_LATENT_CORROBORATION and RUN_STUDENT:
        try:
            bottleneck = model_student.get_layer('morphology_bottleneck')
        except ValueError as exc:
            raise RuntimeError('Student model has no layer named morphology_bottleneck') from exc
        extractor = tf.keras.Model(model_student.input, bottleneck.output)
        latent_length = int(extractor.output_shape[1])
        pool = STUDENT_WINDOW // latent_length
        half = STUDENT_WINDOW // 2
        sig_norm = robust_normalize(signal)
        latent_vectors = []
        latent_event_ids = []
        batch_windows = []
        batch_meta = []

        def flush_latent_batch() -> None:
            if not batch_windows:
                return
            states = extractor.predict(np.stack(batch_windows), verbose=0)
            for state, (event_id, ds_start, ds_stop) in zip(states, batch_meta):
                check_cancelled(_acid_cancel)
                h = state[ds_start:ds_stop]
                if len(h) == 0:
                    continue
                if len(h) == 1:
                    anchored = np.repeat(h, LATENT_ANCHORS, axis=0)
                else:
                    anchors = np.linspace(0, len(h) - 1, LATENT_ANCHORS)
                    anchored = np.stack([np.interp(anchors, np.arange(len(h)), h[:, unit]) for unit in range(h.shape[1])], axis=1)
                latent_vectors.append(anchored.ravel().astype(np.float32))
                latent_event_ids.append(event_id)
            batch_windows.clear()
            batch_meta.clear()
        for _, row in tqdm(primary_residualized.iterrows(), total=len(primary_residualized), desc='Latent windows'):
            check_cancelled(_acid_cancel)
            start, end = (int(row.start_idx), int(row.end_idx))
            center = (start + end) // 2
            window_start = center - half
            window_end = window_start + STUDENT_WINDOW
            if window_start < 0 or window_end > n_samples:
                continue
            local_start = start - window_start
            local_stop = end - window_start + 1
            if local_stop - local_start >= 0.95 * STUDENT_WINDOW:
                continue
            ds_start = max(0, local_start // pool)
            ds_stop = min(latent_length, max(ds_start + 1, int(math.ceil(local_stop / pool))))
            batch_windows.append(sig_norm[window_start:window_end, None])
            batch_meta.append((row.event_id, ds_start, ds_stop))
            if len(batch_windows) >= INFERENCE_BATCH_SIZE:
                flush_latent_batch()
        flush_latent_batch()
        Z_latent = np.asarray(latent_vectors, np.float32)
        latent_events = pd.DataFrame({'event_id': latent_event_ids})
        latent_events = latent_events.merge(physical_work[['event_id', 'physical_cluster', 'physical_gmm_component']], on='event_id', how='left').merge(primary_residualized[['event_id', 'scalar_component']], on='event_id', how='left')
        Zs = RobustScaler().fit_transform(Z_latent)
        latent_pca = PCA(n_components=min(20, Zs.shape[1], len(Zs) - 1), random_state=SEED).fit_transform(Zs)
        latent_embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(latent_pca)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, label_col, title in [(axes[0], 'physical_cluster', 'Latent space colored by physical cluster'), (axes[1], 'scalar_component', 'Latent space colored by scalar component')]:
            check_cancelled(_acid_cancel)
            sc = ax.scatter(latent_embedding[:, 0], latent_embedding[:, 1], c=latent_events[label_col].fillna(-1), s=8, alpha=0.55, cmap='tab10')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(title)
            plt.colorbar(sc, ax=ax, label=label_col)
        finish_figure('10_latent_corroboration')
        np.savez_compressed(OUT_DIR / f'{TAG}_duration_normalized_latents.npz', event_id=np.asarray(latent_event_ids), latents=Z_latent)
        latent_summary = {'n_events': len(Z_latent), 'latent_dimensions': Z_latent.shape[1]}
        print('Latent corroboration complete:', latent_summary)
    else:
        print('Latent corroboration skipped (requires RUN_LATENT_CORROBORATION and RUN_STUDENT).')


def definitions_069():
    global flush_latent_batch

    def flush_latent_batch() -> None:
        if not batch_windows:
            return
        states = extractor.predict(np.stack(batch_windows), verbose=0)
        for state, (event_id, ds_start, ds_stop) in zip(states, batch_meta):
            check_cancelled(_acid_cancel)
            h = state[ds_start:ds_stop]
            if len(h) == 0:
                continue
            if len(h) == 1:
                anchored = np.repeat(h, LATENT_ANCHORS, axis=0)
            else:
                anchors = np.linspace(0, len(h) - 1, LATENT_ANCHORS)
                anchored = np.stack([np.interp(anchors, np.arange(len(h)), h[:, unit]) for unit in range(h.shape[1])], axis=1)
            latent_vectors.append(anchored.ravel().astype(np.float32))
            latent_event_ids.append(event_id)
        batch_windows.clear()
        batch_meta.clear()


# Original notebook cell 71
def stage_071():
    global config_export, final_catalog, json_ready, morph_export_cols, physical_label_columns, report, report_path, scalar_candidate, screening_outcome
    check_cancelled(_acid_cancel)
    final_catalog = primary_residualized.copy()
    physical_label_columns = ['event_id', 'physical_cluster', 'physical_cluster_strength', 'physical_gmm_component', 'split']
    final_catalog = final_catalog.merge(physical_work[physical_label_columns], on='event_id', how='left', suffixes=('', '_physical'))
    if len(morphology):
        morph_export_cols = [c for c in ['event_id', 'rise_fraction', 'fall_fraction', 'edge_asymmetry', 'center_of_mass_asymmetry', 'plateau_fraction', 'intra_variability', 'two_level_candidate', 'two_level_delta_bic', 'two_level_separation', 'morphology_cluster', 'morphology_cluster_strength'] if c in morphology]
        final_catalog = final_catalog.merge(morphology[morph_export_cols], on='event_id', how='left')
        morphology.to_parquet(OUT_DIR / f'{TAG}_long_event_morphology.parquet', index=False)
        np.savez_compressed(OUT_DIR / f'{TAG}_long_event_waveforms.npz', event_id=morphology['event_id'].to_numpy(str), waveforms=morphology_waveforms)
        if len(morphology_cluster_summary):
            morphology_cluster_summary.to_csv(OUT_DIR / f'{TAG}_morphology_cluster_summary.csv', index=False)
    final_catalog.to_parquet(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_final_catalog.parquet', index=False)
    primary_scalar['scores'].to_csv(OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_scalar_gmm_scores.csv', index=False)
    phys_gmm_scores.to_csv(OUT_DIR / f'{TAG}_physical_gmm_scores.csv', index=False)

    def json_ready(value):
        if isinstance(value, dict):
            return {str(k): json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and (not np.isfinite(value)):
            return None
        return value
    scalar_candidate = bool(primary_scalar['candidate'])
    if not scalar_candidate and (not physical_cluster_candidate) and (not morphology_cluster_candidate):
        screening_outcome = 'No robust discrete subpopulation detected; a continuum/tail description is currently favored.'
    elif morphology_cluster_candidate and (not scalar_candidate) and (not physical_cluster_candidate):
        screening_outcome = 'Stable morphology candidates occur only within the long-event subset.'
    elif scalar_candidate and detector_consistent_scalar:
        screening_outcome = 'A detector-consistent scalar subpopulation candidate passed the screening criteria.'
    else:
        screening_outcome = 'At least one candidate structure was found, but evidence is method- or detector-dependent.'
    report = {'tag': TAG, 'primary_set': PRIMARY_SET_NAME, 'input_fingerprints': INPUT_FINGERPRINTS, 'recording': {'n_samples': n_samples, 'fs_hz': fs, 'duration_s': duration_s, 'event_polarity': polarity, 'raw_source': raw_source}, 'detection': {'primary_detector': PRIMARY_DETECTOR, 'cusum': {'dual_scale': CUSUM_DUAL_SCALE, 'scales': {name: {'delta_sigma': d, 'h_sigma': h, 'block_ms': CUSUM_SCALES[name]['block_ms'], 'selection': cusum_selection_notes[name], 'budget_met': cusum_scale_ok[name], 'n_regions': len(cusum_scale_results[name]['regions'])} for name, (d, h) in cusum_selected.items()}, 'regions_by_scale': cusum_level_table['cusum_scale'].value_counts().to_dict(), 'delta_sigma': CUSUM_DELTA, 'h_sigma': CUSUM_H, 'selection': cusum_selection_note, 'false_alarms_per_s_quiet': cusum_false_alarm_rate, 'false_alarms_per_s_gate_surviving': cusum_false_alarm_rate_gated, 'expected_false_alarms_full_recording': cusum_false_alarm_rate * duration_s, 'expected_gate_surviving_false_alarms_full_recording': cusum_false_alarm_rate_gated * duration_s, 'budget_met': cusum_calibration_ok, 'noise_timescales': noise_timescales.to_dict(orient='records'), 'calibration_quiet_seconds': calibration_seconds, 'n_injections': len(injection_plan), 'recall_map': cusum_recall_map.to_dict(orient='records'), 'candidates': len(cusum_candidate_regions), 'accepted': int(cusum_catalog['accepted'].sum()), 'multi_level_candidates': int((cusum_level_table['cusum_n_levels'] >= 2).sum())}, 'acceptance_gate': {'statistic': GATE_STATISTIC, 'mode': SUSTAINED_GATE_MODE, 'sigma': SUSTAINED_GATE_SIGMA, 'absolute_pA': SUSTAINED_GATE_PA, 'floor_pA_by_dwell_ms': {str(d): float(acceptance_floor_pA(d)) for d in (0.1, 0.2, 0.5, 1.0, 5.0, 20.0, 100.0)}}, 'threshold_candidates': len(threshold_candidate_regions), 'threshold_accepted': int(threshold_catalog['accepted'].sum()), 'cusum_threshold_matches': len(match_table), 'resistive_detected': DETECT_RESISTIVE, 'resistive_cusum_candidates': len(cusum_candidate_regions_res), 'resistive_cusum_accepted': int(detector_catalogs['cusum_resistive']['accepted'].sum()) if 'cusum_resistive' in detector_catalogs else 0, 'resistive_threshold_candidates': len(threshold_candidate_regions_res), 'resistive_cusum_threshold_matches': len(resistive_match_table), 'resistive_precursors_of_conductive': int((detector_catalogs['cusum_resistive']['accepted'] & detector_catalogs['cusum_resistive']['precursor_of_conductive']).sum()) if 'cusum_resistive' in detector_catalogs else 0, 'primary_polarity': primary_polarity, 'stage0': {k: meta.get(k) for k in ('line_noise_method', 'line_noise_note', 'mains_frequency_hz', 'baseline_window_s', 'baseline_decim', 'adc_step_pA', 'raw_storage', 'qc_mad_noise_pA', 'qc_psd_before', 'qc_psd_after', 'abf_header')}, 'student_run': RUN_STUDENT, 'student_candidates': len(student_candidate_regions), 'student_accepted': int(student_catalog['accepted'].sum()) if len(student_catalog) else 0, 'student_threshold_matches': len(student_match_table), 'primary_events': len(primary_catalog)}, 'scalar_screen': {'target': primary_scalar['target'], 'best_bic_k': primary_scalar['best_bic_k'], 'best_heldout_k': primary_scalar['best_val_k'], 'delta_bic_vs_1': primary_scalar['delta_bic_vs_1'], 'heldout_ll_gain_vs_1': primary_scalar['heldout_ll_gain_vs_1'], 'means': primary_scalar['means'], 'stds': primary_scalar['stds'], 'weights': primary_scalar['weights'], 'min_separation': primary_scalar['min_separation'], 'min_block_presence': primary_scalar['min_block_presence'], 'dip_p': primary_scalar['dip_p'], 'candidate': scalar_candidate, 'detector_consistent': detector_consistent_scalar}, 'physical_clustering': {'n_clusters': n_physical_clusters, 'noise_fraction': physical_noise_fraction, 'validation_assigned_fraction': validation_assigned_fraction, 'parameter_grid_median_ari': stability_median_ari, 'all_clusters_recur_in_validation': all_clusters_recur, 'time_cramers_v': physical_time_cramers_v, 'candidate': physical_cluster_candidate, 'gmm_best_bic_k': phys_gmm_best_bic, 'gmm_best_heldout_k': phys_gmm_best_val, 'gmm_hdbscan_ari': phys_gmm_hdbscan_ari}, 'morphology': {'enabled': RUN_MORPHOLOGY, 'usable_long_events': len(morphology), 'time_cramers_v': morph_time_cramers_v, 'physical_ari': morph_physical_ari, 'candidate': morphology_cluster_candidate}, 'latent_corroboration': latent_summary, 'screening_outcome': screening_outcome, 'caution': 'Any positive result requires confirmation in an independent 30 h recording before molecular interpretation.'}
    report_path = OUT_DIR / f'{TAG}_subpopulation_report.json'
    report_path.write_text(json.dumps(json_ready(report), indent=2))
    config_export = {name: value for name, value in globals().copy().items() if name in {'TAG', 'STUDENT_ENTER_THRESHOLD', 'STUDENT_EXIT_THRESHOLD', 'SMOOTHING_KERNEL', 'MERGE_GAP_MS', 'MIN_EVENT_MS', 'SUSTAINED_GATE_SIGMA', 'THRESHOLD_ENTER_SIGMA', 'THRESHOLD_EXIT_SIGMA', 'LOCAL_CONTEXT_MS', 'MAX_LOCAL_CONTEXT_MS', 'MATCH_TOLERANCE_MS', 'PRIMARY_SET_NAME', 'N_TIME_BLOCKS', 'PHYSICAL_FEATURES', 'MORPH_MIN_DWELL_MS', 'MORPH_POINTS', 'RUN_STUDENT', 'PRIMARY_DETECTOR', 'AMPLITUDE_STATISTIC', 'BASELINE_MODE', 'SUSTAINED_GATE_MODE', 'SUSTAINED_GATE_NOISE_QUANTILE', 'CUSUM_DUAL_SCALE', 'CUSUM_DELTA', 'CUSUM_H', 'CUSUM_PULSE_DELTA', 'CUSUM_PULSE_H', 'CUSUM_BLOCK_MS', 'CUSUM_PULSE_BLOCK_MS', 'CUSUM_PASS1_BLOCK_MS', 'CUSUM_END_TOL_SIGMA', 'CUSUM_PRE_BASELINE_MS', 'CUSUM_LEVEL_EST_MS', 'CUSUM_MAX_EVENT_S', 'CUSUM_FA_BUDGET_PER_S', 'CUSUM_RAW_FA_MAX_PER_S', 'CUSUM_RECALL_TOLERANCE', 'CUSUM_FALLBACK_TO_MATCHED', 'CUSUM_TARGET_MIN_DEPTH_PA', 'CUSUM_TARGET_MIN_DWELL_MS', 'CUSUM_INJECT_DEPTHS_PA', 'CUSUM_INJECT_DWELLS_MS', 'CUSUM_INJECT_EDGE_TAU_MS', 'PREPROCESS', 'LINE_NOISE_METHOD', 'NOTCH_FREQS', 'NOTCH_BW_HZ', 'LINE_FIT_WINDOW_S', 'LINE_FIT_MASK_SIGMA', 'BASELINE_WINDOW_S', 'BASELINE_DECIM', 'STORE_RAW_AS_CODES', 'DETECT_RESISTIVE'}}
    (OUT_DIR / f'{TAG}_analysis_config.json').write_text(json.dumps(json_ready(config_export), indent=2))
    print('=' * 88)
    print('SCREENING OUTCOME')
    print(screening_outcome)
    print('Independent-recording confirmation is still required before molecular interpretation.')
    print('=' * 88)
    print('Final catalog:', OUT_DIR / f'{TAG}_{PRIMARY_SET_NAME}_final_catalog.parquet')
    print('Report:', report_path)
    print('Figures:', FIG_DIR)


def definitions_071():
    global json_ready

    def json_ready(value):
        if isinstance(value, dict):
            return {str(k): json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and (not np.isfinite(value)):
            return None
        return value


# Original notebook cell 73
def stage_073():
    global _any_extended, _report, _row, _tiers, extended_outcome, name, result
    check_cancelled(_acid_cancel)
    _report = json.loads(report_path.read_text())
    _tiers = {}
    for name, result in extended_results.items():
        check_cancelled(_acid_cancel)
        _tiers[name] = {'n_events': int(len(result['work'])), 'features_used': result['features'], 'features_dropped_as_redundant': [a for a, _, _ in result['dropped_features']], 'pca_components': int(result['n_pcs']), 'hdbscan': {'reference_min_cluster_size': None if result['reference'] is None else result['reference'][0], 'reference_min_samples': None if result['reference'] is None else result['reference'][1], 'n_clusters': int(len(result['validation_presence'])), 'n_nondegenerate_grid_points': int(len(result['grid']) - result['n_degenerate']), 'n_degenerate_grid_points': int(result['n_degenerate']), 'median_ari_nondegenerate': result['median_ari_nondegenerate'], 'validation_assigned_fraction': result['assigned_fraction'], 'validation_counts_by_cluster': result['validation_presence'], 'time_cramers_v': result['hdbscan_time_cramers_v'], 'candidate': result['hdbscan_candidate']}, 'gmm': {'best_bic_k': result['gmm_best_bic_k'], 'best_heldout_k_admissible': result['gmm_best_heldout_k'], 'best_heldout_k_unconstrained': result['gmm_best_heldout_k_unconstrained'], 'heldout_gain_vs_1': result['gmm_heldout_gain_vs_1'], 'time_cramers_v': result['gmm_time_cramers_v'], 'gmm_hdbscan_ari': result['gmm_hdbscan_ari'], 'components': result['gmm_summary'].to_dict(orient='records')}}
    _any_extended = any((t['hdbscan']['candidate'] for t in _tiers.values()))
    if _any_extended:
        extended_outcome = 'At least one extended-feature tier yields a held-out-stable cluster candidate; compare its component table with the scalar and morphology sections before interpreting.'
    elif flagged_features:
        extended_outcome = f'No stable multivariate candidate in the extended tiers, but these features are multimodal in both halves and deserve a scalar screen: {flagged_features}.'
    else:
        extended_outcome = 'Extended feature tiers agree with the base analysis: no reproducible discrete structure.'
    _report['event_families'] = event_families
    _report['spike_context_tests'] = spike_context_tests
    _report['amplitude_baseline_comparison'] = {'canonical': BASELINE_MODE, 'amplitudes': baseline_amplitude_table.to_dict(orient='records'), 'scalar_screen': baseline_screen_table.to_dict(orient='records'), 'gate_overlap': _ov.to_dict()}
    _report['extended_feature_tiers'] = {'gates': {'moment_min_samples': EXT_MOMENT_MIN_SAMPLES, 'spectral_min_samples': EXT_SPECTRAL_MIN_SAMPLES, 'context_ms': EXT_CONTEXT_MS}, 'univariate_multimodal_in_both_halves': flagged_features, 'tiers': _tiers, 'extended_outcome': extended_outcome}
    report_path.write_text(json.dumps(json_ready(_report), indent=2))
    print('=' * 88)
    print('INCUBATION READING (add-on H): reference-model families in this tag')
    for _row in event_families['family_table']:
        check_cancelled(_acid_cancel)
        print(f"  {_row['name']}: {100 * _row['share']:.0f}% ({_row['n']} events, {_row['rate_per_s']:.3f}/s, in {_row['blocks_present']} blocks)")
    print(f"  markers: precursor {100 * event_families['markers']['precursor_fraction']:.0f}%, flat-without-precursor {100 * event_families['markers']['flat_no_precursor_fraction']:.0f}%, resolved {100 * event_families['markers']['resolved_fraction']:.0f}%; family x time Cramér's V = {event_families['family_time_cramers_v']:.2f}")
    print('=' * 88)
    print('EXTENDED-FEATURE OUTCOME')
    print(extended_outcome)
    print('=' * 88)
    print('Report updated:', report_path)


# Cross-recording dashboard (no dynamic series driver)
def stage_076():
    global _, _hours, _pal, _tags, _txt, ax, axes, bottom, c, d, fig, it, j, k, lab, logx, row, s, share_cols, t, v, var, x, xs, y
    if 'EVENTS_ALL' in globals() and len(EVENTS_ALL) and (EVENTS_ALL['tag'].nunique() >= 2):
        _tags = [t for t in TAG_ARMS if t in set(EVENTS_ALL['tag'])]
        _pal = {t: f'C{i}' for i, t in enumerate(_tags)}
        _hours = {t: float(_acid_hours[t]) for t in _tags}
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        ax = axes[0, 0]
        for t in _tags:
            s = EVENTS_ALL[EVENTS_ALL['tag'] == t]
            x, y = (np.log10(s['dwell_ms'].clip(lower=0.001)), s[AMP_VAR])
            if sns is not None and len(s) >= FIG_MIN_N:
                sns.kdeplot(x=x, y=y, ax=ax, levels=6, color=_pal[t], linewidths=1.2)
            else:
                ax.scatter(x, y, s=10, color=_pal[t])
        ax.set_xlabel('log$_{10}$ dwell (ms)')
        ax.set_ylabel(AMP_LABEL)
        ax.set_title('Dwell vs amplitude: joint KDE per recording')
        ax.legend(handles=[plt.Line2D([], [], color=_pal[t], lw=2, label=f"{TAG_ARMS.get(t, t)} (n={int((EVENTS_ALL['tag'] == t).sum())})") for t in _tags], fontsize=8)
        for ax, var, lab, logx in ((axes[0, 1], AMP_VAR, AMP_LABEL, False), (axes[0, 2], 'dwell_ms', 'dwell (ms)', True)):
            for t in _tags:
                v = EVENTS_ALL.loc[EVENTS_ALL['tag'] == t, var].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0 if logx else True)]
                if len(v) < FIG_MIN_N:
                    continue
                d = np.log10(v) if logx else v
                k = gaussian_kde(d)
                xs = np.linspace(d.min() - 0.3 * d.std(), d.max() + 0.3 * d.std(), 400)
                ax.plot(10 ** xs if logx else xs, k(xs), color=_pal[t], lw=2, label=f'{TAG_ARMS.get(t, t)}: median {np.median(v):.3g}')
                ax.axvline(np.median(v), color=_pal[t], ls=':', lw=1.5)
            if logx:
                ax.set_xscale('log')
            ax.set_xlabel(lab)
            ax.set_ylabel('kernel density')
            ax.legend(fontsize=8)
        axes[0, 1].set_title('Amplitude distribution, all recordings')
        axes[0, 2].set_title('Dwell distribution, all recordings')
        if 'incubation_table' in globals() and len(incubation_table) >= 2:
            it = incubation_table.sort_values('hours')
            ax = axes[1, 0]
            share_cols = sorted((c for c in it if c.endswith('_soft')))
            bottom = np.zeros(len(it))
            for c in share_cols:
                ax.bar(it['tag'], it[c], bottom=bottom, label=c.replace('_soft', ''))
                bottom += it[c].to_numpy()
            ax.set_ylabel('Family share (soft)')
            ax.set_title('Family composition by recording')
            ax.legend(fontsize=8)
            ax = axes[1, 1]
            for c, lab in (('precursor', 'precursor present'), ('flat_no_precursor_sharp', 'flat, no precursor, sharp edges'), ('deep_long', 'deep-long (>= 30% I0)'), ('resolved', 'resolved')):
                if c in it:
                    ax.plot(it['hours'], it[c], marker='o', label=lab)
            ax.set_xlabel('Incubation time (h)')
            ax.set_ylabel('Fraction of events')
            ax.set_title('Physical markers vs incubation')
            ax.legend(fontsize=8)
            ax = axes[1, 2]
            ax.bar(it['tag'], it['rate_per_s'], color='tab:blue', alpha=0.7)
            ax.set_ylabel('Event rate (/s)', color='tab:blue')
            ax.set_title('Rate, pore and floor by recording')
            for j, (_, row) in enumerate(it.iterrows()):
                _txt = f"I0 {row['I0_pA']:.0f} pA\nfloor {row['floor_0.5ms_pA']:.1f} pA" + (f"\ntau {row['instrument_tau_ms']:.2f} ms" if np.isfinite(row.get('instrument_tau_ms', np.nan)) else '')
                ax.text(j, 0.5 * row['rate_per_s'], _txt, ha='center', va='center', fontsize=8, color='w', fontweight='bold')
        else:
            for ax in axes[1]:
                ax.axis('off')
        finish_figure('18_series_dashboard')
    else:
        print('Dashboard needs at least two recordings with family summaries; run the series (TAGS_TO_RUN) first.')
