"""The Signal Conditioning previews: a padded window of the recording, the honest
"window estimate" label, the exact pass, the line finder and the notch merge."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis import conditioning_preview as cp
from nanoscope.analysis import filter_views
from nanoscope.analysis.baseline import ConstantBaselineWarning
from nanoscope.analysis.conditioning import conditioned_trace
from nanoscope.analysis.conditioning_recipes import BUILTIN_RECIPES, complete_recipe

FS = 50_000.0


@pytest.fixture(scope="module")
def trace() -> np.ndarray:
    rng = np.random.default_rng(1)
    n = 600_000
    t = np.arange(n) / FS
    x = (-1000 + rng.normal(0, 8, n) + 30 * np.sin(2 * np.pi * 60 * t)
         + 10 * np.sin(2 * np.pi * 180 * t) + 40 * np.sin(2 * np.pi * 0.3 * t))
    for s in range(5000, n, 37000):
        x[s:s + 300] += 200
    return x.astype(np.float32)


def _req(params, raw, start, stop, **kw) -> cp.PreviewRequest:
    return cp.PreviewRequest(1, "window", complete_recipe(params), FS, raw=raw, start=start,
                             stop=stop, **kw)


LOCAL_CHAINS = {
    "low-pass": {"filter_cutoff_hz": 10_000.0},
    "high-pass 100 Hz": {"hp_enabled": True},
    "notch 60 Hz Q30": {"notch_enabled": True},
    "median": {"median_enabled": True},
    "savgol": {"savgol_enabled": True},
    "running mean": {"rmean_enabled": True},
    "all local stages": {"filter_cutoff_hz": 8_000.0, "hp_enabled": True,
                         "notch_enabled": True, "median_enabled": True,
                         "savgol_enabled": True, "rmean_enabled": True},
}


@pytest.mark.parametrize("name", list(LOCAL_CHAINS))
def test_a_padded_window_is_exact_for_local_stages(trace, name):
    params = complete_recipe(LOCAL_CHAINS[name])
    assert cp.window_is_exact(params, FS)
    full = conditioned_trace(trace, FS, params)
    sigma = float(np.std(trace))
    for a, b in ((250_000, 300_000), (400_000, 410_000), (100_000, 200_000)):
        res = cp.compute_preview(_req(params, trace, a, b))
        assert res.exact and not res.error and not res.estimate_stages
        worst = np.max(np.abs(res.processed_window.astype(np.float64) - full[a:b])) / sigma
        assert worst < 1e-4, (name, a, b, worst)
        np.testing.assert_array_equal(res.raw_window, trace[a:b])


def test_estimator_stages_are_flagged_as_a_window_estimate(trace):
    for params in ({"detrend_enabled": True}, {"normalize_enabled": True},
                   *BUILTIN_RECIPES.values()):
        p = complete_recipe(params)
        assert not cp.window_is_exact(p, FS)
        assert cp.estimate_stage_titles(p)
    p = complete_recipe({"normalize_enabled": True})
    res = cp.compute_preview(_req(p, trace, 100_000, 150_000))
    assert not res.exact and res.estimate_stages == ("Normalize",)
    full = conditioned_trace(trace, FS, p)
    assert np.max(np.abs(res.processed_window - full[100_000:150_000])) > 1e-3, \
        "a z-score of a window is not the z-score of the recording"


def test_a_window_that_reaches_both_ends_is_exact_even_with_estimators(trace):
    p = complete_recipe({"normalize_enabled": True, "detrend_enabled": True})
    res = cp.compute_preview(_req(p, trace, 0, trace.size))
    assert res.exact
    np.testing.assert_array_equal(res.processed_window, conditioned_trace(trace, FS, p))


def test_the_pad_is_the_longest_stage_and_is_capped():
    p = complete_recipe({"notch_enabled": True, "filter_cutoff_hz": 10_000.0,
                         "median_enabled": True})
    pads = cp.stage_pad_samples(p, FS)
    assert cp.preview_pad_samples(p, FS) == max(pads.values()) == pads["notch"]
    slow = complete_recipe({"hp_enabled": True, "hp_cutoff_hz": 0.1})
    assert cp.stage_pad_samples(slow, FS)["highpass"] > cp.MAX_PREVIEW_PAD_SAMPLES
    assert cp.preview_pad_samples(slow, FS) == cp.MAX_PREVIEW_PAD_SAMPLES
    assert not cp.window_is_exact(slow, FS)
    assert cp.preview_pad_samples(complete_recipe({}), FS) == 0


def test_a_stage_that_settles_past_the_pad_limit_is_named():
    """A capped pad makes a LOCAL stage an estimate too, and the status must say which
    one (not "uses whole-recording statistics", which is the estimators' reason)."""
    slow = complete_recipe({"hp_enabled": True, "hp_cutoff_hz": 0.1, "notch_enabled": True})
    limited = cp.pad_limited_stages(slow, FS)
    assert [title for title, _s in limited] == ["High-pass"]
    assert limited[0][1] == pytest.approx(400.0)          # 40·fs/fc samples, in seconds
    assert not cp.estimate_stage_titles(slow)
    assert cp.pad_limited_stages(complete_recipe({"hp_enabled": True}), FS) == ()
    detrend = complete_recipe({"detrend_enabled": True, "detrend_window_s": 100.0})
    assert cp.pad_limited_stages(detrend, FS) == (), "the estimator's own reason covers it"


def test_the_detrend_pad_never_leaves_the_window_too_short(trace):
    """A visible window much shorter than the detrend window still gets enough recording
    either side for the estimator to run (an estimate, never a refusal)."""
    p = complete_recipe({"detrend_enabled": True, "detrend_window_s": 0.5})
    for a in (0, 300_000, trace.size - 2_500):
        res = cp.compute_preview(_req(p, trace, a, a + 2_500))
        assert not res.error and not res.degenerate and not res.window_too_short, a
        assert res.processed_window.size == 2_500


def test_the_preview_builds_only_through_conditioned_trace(trace, monkeypatch):
    calls = []
    real = cp.conditioned_trace

    def counting(raw, fs, params):
        calls.append(raw.size)
        return real(raw, fs, params)

    monkeypatch.setattr(cp, "conditioned_trace", counting)
    p = complete_recipe({"filter_cutoff_hz": 10_000.0, "notch_enabled": True})
    cp.compute_preview(_req(p, trace, 200_000, 220_000, want_spectrum=True))
    assert len(calls) == 1 and calls[0] < trace.size
    for module in (cp, filter_views):
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                  for n in ast.walk(ast.parse(inspect.getsource(module)))
                  if isinstance(n, ast.Call)}
        assert not {"preprocess_for_detection", "lowpass"} & called, module.__name__


def test_kernel_errors_are_reported_not_raised(trace):
    p = complete_recipe({"filter_cutoff_hz": 40_000.0})
    res = cp.compute_preview(_req(p, trace, 0, 10_000))
    assert "Nyquist" in res.error and res.processed_window is None
    assert res.raw_window is not None


def test_a_degenerate_exact_pass_stays_degenerate_through_the_memo():
    rng = np.random.default_rng(0)
    x = (rng.normal(0, 5, 50_000) - 900).astype(np.float32)
    p = complete_recipe({"detrend_enabled": True, "detrend_window_s": 50_000 / FS})
    exact = cp.compute_exact(cp.PreviewRequest(7, "exact", p, FS, raw=x, stop=x.size))
    assert exact.degenerate and exact.full is not None
    assert any(isinstance(w, ConstantBaselineWarning) for w in exact.full[1])
    memo = cp.ConditionedTraceMemo()
    memo.put(("d", 0), *exact.full)
    res = cp.compute_preview(_req(p, x, 1_000, 2_000, memo=memo.get(("d", 0))))
    assert res.from_exact_pass and res.degenerate == exact.degenerate
    assert res.processed_window is None, "the main trace's rule: a no-op stage draws nothing"


def test_compute_exact_conditions_the_whole_recording(trace):
    p = complete_recipe({"normalize_enabled": True})
    res = cp.compute_exact(cp.PreviewRequest(3, "exact", p, FS, raw=trace, stop=trace.size))
    np.testing.assert_array_equal(res.full[0], conditioned_trace(trace, FS, p))
    assert res.exact and not res.error


def test_the_spectrum_pair_shares_one_segment_and_omits_normalize(trace):
    p = complete_recipe({"normalize_enabled": True, "filter_cutoff_hz": 5_000.0})
    res = cp.compute_preview(_req(p, trace, 100_000, 200_000, want_spectrum=True))
    sp = res.spectrum
    assert sp.normalize_omitted and sp.processed_psd is not None
    assert sp.nperseg == 8192 and sp.df_hz == pytest.approx(FS / 8192)
    assert sp.f_hz[0] > 0 and sp.raw_psd.shape == sp.processed_psd.shape
    hi = sp.f_hz > 10_000
    assert np.median(sp.processed_psd[hi]) < 1e-3 * np.median(sp.raw_psd[hi]), \
        "the after curve is in the raw unit (normalize left out), low-passed"
    assert {round(line.freq_hz) for line in res.lines} >= {60, 180}


def _lines_signal(seconds: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = rng.normal(0, 10, n)
    for f, a in ((60, 3), (120, 2), (180, 2), (300, 1.5), (4570.8, 1.5)):
        x += a * np.sin(2 * np.pi * f * t)
    return x


def test_the_line_finder_finds_the_injected_lines():
    sp = cp.spectrum_pair(_lines_signal(40.0), None, FS)
    lines = cp.find_spectral_lines(sp.f_hz, sp.raw_psd)
    assert [round(line.freq_hz, 1) for line in lines] == [60.0, 120.0, 180.0, 300.0, 4570.8]
    assert [(line.mains_hz, line.harmonic) for line in lines] == [
        (60.0, 1), (60.0, 2), (60.0, 3), (60.0, 5), (None, 0)]
    assert all(line.prominence_db >= 10 for line in lines)
    assert lines[0].text == f"60.0 Hz · +{lines[0].prominence_db:.1f} dB · mains 60×1"


def test_the_line_finder_is_deterministic_and_quiet_on_noise():
    sp = cp.spectrum_pair(_lines_signal(2.0), None, FS)
    runs = [cp.find_spectral_lines(sp.f_hz, sp.raw_psd) for _ in range(30)]
    assert all(r == runs[0] for r in runs) and len(runs[0]) == 5
    for seed in range(4):
        noise = np.random.default_rng(seed).normal(0, 10, int(20 * FS))
        sp = cp.spectrum_pair(noise, None, FS)
        assert cp.find_spectral_lines(sp.f_hz, sp.raw_psd) == ()


def test_a_line_is_placeable_only_when_its_frequency_is_known_to_the_notch_width():
    snapped = cp.SpectralLine(60.0, 20.0, 60.0, 1, measured_hz=60.02, snapped=True)
    near_mains = cp.SpectralLine(60.2, 20.0, 60.0, 1, measured_hz=60.2)
    narrow_low = cp.SpectralLine(37.0, 20.0)
    high = cp.SpectralLine(4570.8, 20.0)
    assert cp.line_is_placeable(snapped, 0.76, 100.0)        # on the harmonic itself
    assert not cp.line_is_placeable(near_mains, 6.1, 30.0), "a mains tag is no exemption"
    assert not cp.line_is_placeable(narrow_low, 6.1, 30.0)     # f/Q = 1.2 Hz < Δf
    assert cp.line_is_placeable(narrow_low, 0.76, 30.0)
    assert cp.line_is_placeable(high, 6.1, 30.0)


def _notched_amplitude(x, f_hz, notch_hz, q):
    import warnings

    p = complete_recipe({"notch_enabled": True, "notch_freqs": [notch_hz], "notch_q": q})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.asarray(conditioned_trace(x, FS, p), dtype=np.float64)
    t = np.arange(y.size) / FS
    mid = slice(y.size // 5, 4 * y.size // 5)
    return 2 * abs(np.dot(y[mid], np.exp(-2j * np.pi * f_hz * t[mid]))) / (t[mid].size)


@pytest.mark.parametrize("f_true", [245.0, 1045.0])
def test_an_off_harmonic_line_is_listed_where_it_was_measured_at_a_coarse_resolution(f_true):
    """At Δf 6.1 Hz a 245 Hz line lies within one bin of 240 Hz (60×4): snapping it there
    put the notch 5 Hz off the line, which left 61 % of it (Q 30) yet read "notched"."""
    rng = np.random.default_rng(0)
    n = 100_000
    t = np.arange(n) / FS
    x = (rng.normal(0, 8, n) + 6 * np.sin(2 * np.pi * f_true * t)).astype(np.float32)
    sp = cp.spectrum_pair(x, None, FS)
    assert sp.df_hz == pytest.approx(6.1035, abs=1e-3)
    line = min(cp.find_spectral_lines(sp.f_hz, sp.raw_psd),
               key=lambda found: abs(found.peak_hz - f_true))
    assert line.mains_hz is None and not line.snapped and "mains" not in line.text
    assert abs(line.freq_hz - f_true) <= 0.2
    for q in (30.0, 100.0):
        placeable = cp.line_is_placeable(line, sp.df_hz, q)
        assert placeable == (sp.df_hz <= f_true / q)
        if placeable:
            assert _notched_amplitude(x, f_true, line.freq_hz, q) < 0.6, "≥ 20 dB off"


def test_a_mains_line_is_snapped_only_when_the_resolution_resolves_it():
    rng = np.random.default_rng(2)
    for seconds, snapped in ((2.0, False), (20.0, True)):
        n = int(seconds * FS)
        t = np.arange(n) / FS
        x = rng.normal(0, 8, n) + 6 * np.sin(2 * np.pi * 180.0 * t)
        sp = cp.spectrum_pair(x, None, FS)
        line = min(cp.find_spectral_lines(sp.f_hz, sp.raw_psd),
                   key=lambda found: abs(found.peak_hz - 180.0))
        assert (line.mains_hz, line.harmonic, line.snapped) == (60.0, 3, snapped)
        assert abs(line.peak_hz - 180.0) < 0.2
        if snapped:
            assert line.freq_hz == 180.0 and "· mains 60×3" in line.text
        else:
            assert line.freq_hz == round(line.peak_hz, 1) and "near mains 60×3" in line.text


def test_notch_covers_means_at_least_20_db_off_the_line():
    assert cp.notch_covers(240.0, 240.0, 30.0)
    assert cp.notch_covers(241.3, 240.0, 30.0)         # half-bandwidth 4 Hz / 3
    assert not cp.notch_covers(245.0, 240.0, 30.0)
    assert not cp.notch_covers(241.0, 240.0, 100.0)
    rng = np.random.default_rng(4)
    t = np.arange(200_000) / FS
    x = rng.normal(0, 1, t.size) + 6 * np.sin(2 * np.pi * 241.3 * t)
    assert _notched_amplitude(x, 241.3, 240.0, 30.0) < 0.6


def test_merge_notch_lines_dedups_appends_and_refuses_over_the_cap():
    m = cp.merge_notch_lines([60.0, 120.0], [120.2, 4570.83, 4570.8], q=30.0, max_lines=32)
    assert m.lines == [60.0, 120.0, 4570.8] and m.added == (4570.8,)
    assert m.skipped == (120.2, 4570.8)
    many = [100.0 * (k + 1) for k in range(31)]
    refused = cp.merge_notch_lines(many, [5000.0, 7000.0], q=30.0, max_lines=32)
    assert refused.lines is None and "at most 32" in refused.refusal


def test_units_after_chain_labels():
    assert cp.units_after_chain({}, "pA").label == "processed (pA)"
    hp = cp.units_after_chain({"hp_enabled": True, "hp_cutoff_hz": 100.0}, "pA")
    assert hp.removes_dc and not hp.changes_units
    assert hp.label == "processed (pA, DC removed by 100 Hz high-pass)"
    assert hp.short_label == "processed (pA, no DC)"
    z = cp.units_after_chain({"normalize_enabled": True, "normalize_mode": "zscore"}, "pA")
    assert z.changes_units and z.label == "processed (σ, whole-trace z-score)"
    assert z.short_label == "processed (σ)" and z.unit_label == "σ"
    assert hp.unit_label == "pA, no DC", "the shortest tier keeps what the stage did to DC"
    m = cp.units_after_chain({"normalize_enabled": True, "normalize_mode": "model_input",
                              "normalize_clip_sigma": 30.0}, "pA")
    assert "clipped at ±30σ" in m.label


def test_clamp_window_centres_and_caps():
    assert cp.clamp_window(-5, 50, 40) == (0, 40, False)
    a, b, clamped = cp.clamp_window(0, 12_500_000, 12_500_000)
    assert clamped and b - a == cp.MAX_PREVIEW_WINDOW_SAMPLES
    assert a + (b - a) // 2 == 6_250_000
    a, b, _ = cp.clamp_window(12_000_000, 12_500_000, 12_500_000, max_samples=100)
    assert (a, b) == (12_249_950, 12_250_050)


def test_the_memo_is_lru_and_drops_documents():
    memo = cp.ConditionedTraceMemo(capacity=2)
    a, b, c = np.zeros(3), np.ones(3), np.full(3, 2.0)
    memo.put(("doc1", 0, "s"), a)
    memo.put(("doc2", 0, "s"), b)
    assert memo.get(("doc1", 0, "s"))[0] is a      # touch doc1: doc2 is now the oldest
    memo.put(("doc3", 0, "s"), c, ("w",))
    assert ("doc2", 0, "s") not in memo and len(memo) == 2
    assert memo.get(("doc3", 0, "s"))[1] == ("w",)
    memo.drop_doc("doc1")
    assert memo.keys() == [("doc3", 0, "s")]
    memo.clear()
    assert len(memo) == 0


def test_the_memo_key_ignores_settings_of_a_disabled_stage():
    src = ("doc", 0, 1.0, 0.0, 100)
    a = cp.memo_key(src, complete_recipe({"hp_cutoff_hz": 50.0}), FS)
    b = cp.memo_key(src, complete_recipe({"hp_cutoff_hz": 200.0}), FS)
    c = cp.memo_key(src, complete_recipe({"hp_enabled": True}), FS)
    assert a == b != c


def test_the_resolution_helper_matches_welchs_segment_rule():
    need = cp.window_for_resolution_s(FS, 1.0)
    n = int(np.ceil(need * FS))
    sp = cp.spectrum_pair(np.random.default_rng(0).normal(size=n), None, FS)
    assert sp.df_hz <= 1.0
    sp = cp.spectrum_pair(np.random.default_rng(0).normal(size=n - 8), None, FS)
    assert sp.df_hz > 1.0
    assert cp.window_for_resolution_s(250_000.0, 1.0) is None


def test_the_two_new_modules_import_no_qt_and_no_gui():
    root = Path(cp.__file__).parent
    for name in ("conditioning_preview.py", "filter_views.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add("." * node.level + (node.module or ""))
        bad = {m for m in mods if m.split(".")[0] in ("PySide6", "pyqtgraph", "shiboken6")
               or "gui" in m.split(".")}
        assert bad == set(), (name, bad)


def test_only_a_short_input_complaint_is_called_window_too_short(trace, monkeypatch):
    p = complete_recipe({"detrend_enabled": True, "filter_cutoff_hz": 40_000.0})
    res = cp.compute_preview(_req(p, trace, 100_000, 110_000))
    assert "Nyquist" in res.error and not res.window_too_short, "a chain error is the chain's"

    def short(raw, fs, params):
        raise ValueError("window_s=9s is longer than the recording (0.4s); shorten it")

    monkeypatch.setattr(cp, "conditioned_trace", short)
    p = complete_recipe({"detrend_enabled": True})
    assert cp.compute_preview(_req(p, trace, 100_000, 110_000)).window_too_short
    whole = cp.compute_preview(_req(p, trace, 0, trace.size))
    assert whole.error and not whole.window_too_short, "the whole recording is no window"
