"""The event-masked two-pass open-pore baseline: kernel reuse, config, health, polarity.

WHAT THIS FILE IS DEFENDING
---------------------------
`nanoscope.analysis.baseline.estimate_baseline` used to track I0(t) with a block
median over a fixed window, and its three call sites used three different windows,
so one event reported a different ``depth_pA`` depending on which detector found
it. On the owner's own apo recording the 10 ms block median stamps **80.6 % of
events with a baseline more than 10 pA from the true open pore** and understates
``depth_pA`` by **4.7x** (measured; the spec's 14.7x does not reproduce). The fix is to characterise with the estimator he already
uses in his own analysis (``dataLoader.ipynb::baseline_remove_masked_median``),
whose implementation ALREADY EXISTS in this repo as
`nanoscope.analysis.ml_preprocess.baseline_masked_median` and is bit-exact against
his persisted training corpus.

So the load-bearing claims here are, in order of how much damage their failure does:

  a. the kernel is IMPORTED, not copied -- exactly one `_decimated_median` in the
     tree, `ml_preprocess.py` byte-for-byte unchanged, and no 1-D
     ``median_filter`` call anywhere in ``nanoscope/analysis`` (the 1-D form is
     non-deterministic in this environment and has reached production once already);
  b. the wired-up path reproduces ``CSVData/ferritin_data{1,2,3}.csv.gz`` at
     ``max|diff| = 0`` and the four stored QC scalars exactly;
  c. polarity is implemented by MIRRORING, so at ``polarity=-1`` the wrapper is a
     bit-exact no-op and (b) holds by construction rather than by assertion;
  d. the four legacy estimators did not move -- pinned by digests that were proved
     byte-identical to the pre-batch PyInstaller build (``dist/NanoScope/NanoScope``,
     Jul 28 21:13), the only witness to the pre-change behaviour in a tree with no
     git history;
  e. every silent-garbage path the reference algorithm leaves open is now either
     refused or reported: too short, non-finite, mask found nothing, mask swallowed
     everything, rough pass measuring drift instead of events;
  f. the estimator is deterministic under allocation churn and polls the cancel token.

Pure numpy/scipy; the corpus-marked tests skip cleanly when the research data is
absent.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import pathlib
import threading
import warnings

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import ml_preprocess as mp
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

FS = 50_000.0
ANALYSIS_DIR = pathlib.Path(B.__file__).parent
CORPUS = pathlib.Path("/Users/abrahamcampos/Desktop/Research-Data/FerritinModeler/CSVData")
APO_ABF = pathlib.Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")

#: md5 of ``ml_preprocess.py`` as shipped. It is a FROZEN CONTRACT: the kernel is
#: imported from it, never edited, and its own golden digests must never be
#: "updated". A failure here means someone reached into the kernel.
#:
#: Pinned to the 2026-09-04 revision (commit 3ba94ab), which added input validation
#: and finite checks ONLY: finite/positive checks on the notch ``q``/``freqs`` and on
#: the ``MLPreprocessConfig`` fields, exact-int checks on its integer fields, and a
#: ``require_finite`` after the notch, baseline and despike stages. No arithmetic
#: changed (an AST census of the two revisions differs in exactly
#: ``notch_sequential``, ``MLPreprocessConfig.__post_init__`` and
#: ``apply_ml_preprocess``). Equivalence against the prior revision (8032ec6) was
#: verified bit-exact on 2026-09-11 with an OLD-vs-CURRENT harness (both modules
#: loaded side by side; ``np.array_equal`` with dtype and shape): 2,591 comparisons,
#: 0 differences, covering ``notch_sequential`` (12 freq/q sets incl. lines at and
#: above Nyquist, float32 and float64 input), ``_decimated_median``,
#: ``baseline_robust_median``, ``baseline_masked_median``, ``baseline_savgol``,
#: ``despike_median``, ``apply_ml_preprocess`` for all 10 presets + the
#: ``apo_legacy`` alias + 13 hand-built configs over all four baseline methods,
#: ``normalize_global``, ``model_input_normalization``, ``decode_ml_regions`` and
#: ``gate_regions_by_amplitude``, on a 1e6-sample event train and a 1e6-sample
#: drifting sinusoid (float32) at 50 kHz and 25 kHz (Savitzky-Golay configs at
#: 25 kHz on 250k-sample slices) plus 100k/20k-sample traces in both dtypes; the
#: corpus-marked tests here and in ``test_ml_preprocess.py``
#: (ferritin_data1/2/3 and the apo -600 mV ABF, 13 tests) passed on the real data.
#: The contract remains: no numerical edit without re-verification.
_ML_PREPROCESS_MD5 = "48e364c4264dcef2c25a2fadad9bc4a1"


# ===========================================================================
# helpers -- synthetic traces
# ===========================================================================
def _flat(n=1_000_000, *, seed=3, i0=-100.0, noise=5.0):
    rng = np.random.default_rng(seed)
    return (i0 + rng.normal(0, noise, n)).astype(np.float32)


def _events(n=1_000_000, *, seed=3, i0=-100.0, noise=5.0, depth=80.0,
            dwell=1000, period=10_000, sign=-1.0):
    rng = np.random.default_rng(seed)
    x = i0 + rng.normal(0, noise, n)
    for s in range(0, n - dwell, period):
        x[s:s + dwell] += sign * depth
    return x.astype(np.float32)


def _drifting(n=1_000_000, *, seed=3, amp=300.0, f=5.0, i0=-100.0, noise=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = i0 + amp * np.sin(2 * np.pi * f * t) + rng.normal(0, noise, n)
    for s in range(0, n - 1000, 10_000):
        x[s:s + 1000] -= 80.0
    return x.astype(np.float32)


# ===========================================================================
# a. THE KERNEL IS IMPORTED, NOT COPIED
# ===========================================================================
def test_ml_preprocess_is_byte_for_byte_unchanged():
    """The kernel's own file is frozen. Importing from it changes nothing in it."""
    digest = hashlib.md5((ANALYSIS_DIR / "ml_preprocess.py").read_bytes()).hexdigest()
    assert digest == _ML_PREPROCESS_MD5, (
        "ml_preprocess.py has been modified. It is bit-exact against the owner's "
        "persisted training corpus and digest-pinned; the baseline work IMPORTS it."
    )


def test_exactly_one_decimated_median_in_the_tree():
    """R1, the highest-cost failure: two copies of a 20-line function with five
    silent bit-exactness traps (the odd-window bump, ndimage's 'reflect' semantics,
    where the float32 cast lands, the (1,m) call shape, ``int()`` truncation) WILL
    drift, and the drift surfaces as a scientific disagreement between the ML path
    and the Schmitt path on the same recording."""
    defs = []
    for path in sorted(pathlib.Path(B.__file__).parents[1].rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_decimated_median", "baseline_masked_median", "baseline_robust_median",
                "despike_median",
            ):
                defs.append((path.name, node.name))
    assert sorted(defs) == sorted([
        ("ml_preprocess.py", "_decimated_median"),
        ("ml_preprocess.py", "baseline_masked_median"),
        ("ml_preprocess.py", "baseline_robust_median"),
        ("ml_preprocess.py", "despike_median"),
    ]), f"the kernel has been re-defined outside ml_preprocess.py: {defs}"


def test_baseline_reexports_the_kernel_objects_themselves():
    """Not a copy with the same name -- the same function object."""
    assert B.decimated_median is mp._decimated_median
    assert B.despike_median is mp.despike_median
    assert B._masked_median_kernel is mp.baseline_masked_median
    assert B._robust_median_kernel is mp.baseline_robust_median


def test_no_one_dimensional_median_filter_call_in_analysis():
    """P10 -- the only durable defence against the determinism bug reappearing.

    ``scipy.ndimage.median_filter`` 1.15 is non-deterministic on a **1-D** input
    once the footprint passes ~261 taps: this repo measured six different answers in
    forty runs on a real recording, spanning 6.4 pA, with a fixed 5-sigma decode
    counting anywhere from 337 to 469 events. The ``(1, m)`` form takes a different
    path inside ndimage and is stable (0 wrong in 600 calls). It is invisible in
    review, so it is asserted structurally.
    """
    offenders = []
    checked = 0
    for path in sorted(ANALYSIS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        # Resolve the binding: ``nanoscope.analysis.filters.median_filter`` is a
        # DIFFERENT function (it wraps ``scipy.signal.medfilt``), and flagging its
        # call sites would make this guard cry wolf until someone deleted it.
        ndimage_names = {"scipy.ndimage.median_filter"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scipy.ndimage":
                for a in node.names:
                    if a.name == "median_filter":
                        ndimage_names.add(a.asname or a.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "scipy":
                for a in node.names:
                    if a.name == "ndimage":
                        ndimage_names.add(f"{a.asname or a.name}.median_filter")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                name = f"{fn.value.id}.{fn.attr}"
            elif isinstance(fn, ast.Name):
                name = fn.id
            else:
                continue
            if name not in ndimage_names:
                continue
            checked += 1
            arg = node.args[0]
            two_d = (
                isinstance(arg, ast.Subscript)
                and isinstance(arg.slice, ast.Tuple)
                and any(isinstance(e, ast.Constant) and e.value is None
                        for e in arg.slice.elts)
            )
            if not two_d:
                offenders.append(f"{path.name}:{node.lineno}")
    assert checked >= 2, (
        "the guard resolved no scipy.ndimage.median_filter call sites at all -- it has "
        "stopped guarding anything"
    )
    assert not offenders, (
        "scipy.ndimage.median_filter called on a 1-D array at "
        f"{offenders}; use median_filter(x[None, :], size=(1, w), mode='reflect')[0]"
    )


# ===========================================================================
# b. BaselineConfig -- the single owner of the default
# ===========================================================================
def test_default_is_the_notebooks_own_fallback():
    d = B.DEFAULT_BASELINE
    assert (d.method, d.rough_s, d.base_s, d.decim, d.mask_sigma, d.dilate_ms,
            d.despike_size) == ("masked_median", 0.5, 0.1, None, 3.5, 1.0, 0)


def test_estimate_baseline_default_method_is_unchanged():
    """`estimate_baseline`'s own default stays 'block_median' at 10 ms: every stored
    result and every existing test was measured with it. The NEW default lives on
    `BaselineConfig`, which is what the app characterises with."""
    import inspect

    sig = inspect.signature(B.estimate_baseline)
    assert sig.parameters["method"].default == "block_median"
    # window_s is now a None sentinel meaning "this method's own default"...
    assert sig.parameters["window_s"].default is None
    # ...and for the block methods that default is still exactly 10 ms.
    x = _flat(60_000)
    a = B.estimate_baseline(x, FS, polarity=-1)
    b = B.estimate_baseline(x, FS, polarity=-1, window_s=0.01)
    assert a.tobytes() == b.tobytes()


def test_robust_median_window_default_is_not_ten_milliseconds():
    """A bare ``method='robust_median'`` must not silently run an 11-tap median.

    0.01 s at decim 50 and 50 kHz is ``int(0.01*50000/50)`` = 10 -> 11 taps, which
    tracks the events instead of rejecting them. The None sentinel gives it
    `DEFAULT_BASELINE.window_s` (0.5 s) instead.
    """
    x = _events(200_000)
    bare = B.estimate_baseline(x, FS, method="robust_median", polarity=-1, decim=50)
    half = B.estimate_baseline(x, FS, method="robust_median", polarity=-1,
                               window_s=0.5, decim=50)
    tenms = B.estimate_baseline(x, FS, method="robust_median", polarity=-1,
                                window_s=0.01, decim=50)
    assert bare.tobytes() == half.tobytes()
    assert bare.tobytes() != tenms.tobytes()


@pytest.mark.parametrize("kw, needle", [
    (dict(rough_s=0.0), "rough_s must be a positive"),
    (dict(base_s=-1.0), "base_s must be a positive"),
    (dict(rough_s=0.1, base_s=0.5), "must not exceed rough_s"),
    (dict(decim=0), "decim must be >= 1"),
    (dict(mask_sigma=0.0), "mask_sigma must be a positive"),
    (dict(mask_sigma=12.0), "beyond any real event threshold"),
    (dict(dilate_ms=-1.0), "dilate_ms must be >= 0"),
    (dict(despike_size=1), "despike_size must be 0"),
    (dict(despike_size=11), "despike_size must be 0"),
    (dict(window_s=0.0), "window_s must be a positive"),
    (dict(method="nope"), "unknown method"),
])
def test_config_refuses_each_invalid_parameter_with_its_own_message(kw, needle):
    """E10. The reference validates none of these and returns silent garbage for
    every one; a bad parameter must name itself."""
    with pytest.raises(ValueError, match=needle):
        B.BaselineConfig(**kw)


def test_the_same_validation_fires_through_the_bare_keyword_path():
    """The invariant, not the call site: a caller must not be able to reach the
    kernel with a parameter set that only `BaselineConfig` polices."""
    x = _flat(200_000)
    with pytest.raises(ValueError, match="must not exceed rough_s"):
        B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                            rough_s=0.1, base_s=0.5)
    with pytest.raises(ValueError, match="mask_sigma must be a positive"):
        B.estimate_baseline(x, FS, method="masked_median", polarity=-1, mask_sigma=-1.0)


def test_masked_methods_refuse_the_legacy_two_pass_flag():
    x = _flat(200_000)
    with pytest.raises(ValueError, match="does not take two_pass"):
        B.estimate_baseline(x, FS, method="masked_median", polarity=-1, two_pass=True)


def test_decim_is_derived_from_the_tracking_window_and_clamped():
    """2.4: cost goes as ``n * win_s * fs / decim**2``, so decim is the throttle and
    is derived from a tap target rather than exposed as a free knob."""
    assert B.derive_decim(0.1, FS) == int(0.1 * FS / B.TAPS_TARGET)  # 66
    assert B.derive_decim(0.01, FS) == int(0.01 * FS / B.TAPS_TARGET)  # 6
    # never below 1, never so high that fewer than MIN_TAPS taps remain
    assert B.derive_decim(1e-4, FS) == 1
    assert int(1e-4 * FS / B.derive_decim(1e-4, FS)) >= B.MIN_TAPS
    with pytest.raises(ValueError, match="at least 3 taps"):
        B.derive_decim(2e-5, FS)


def test_a_pinned_decim_always_wins_over_the_derivation():
    """What keeps the corpus parameter sets (50, 50, 10) verbatim."""
    cfg = B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50)
    assert cfg.resolved_decim(FS) == 50 != B.derive_decim(0.1, FS)
    assert cfg.resolve(FS).decim == 50


def test_a_pinned_decim_that_starves_the_filter_is_refused():
    """Not a warning: at n <= 50 with decim 50 the reference's 'baseline' is
    literally one sample broadcast over the whole recording."""
    cfg = B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=5000)
    with pytest.raises(ValueError, match="leaves only 1 tap"):
        cfg.resolved_decim(FS)


def test_fingerprint_moves_with_every_parameter_and_is_stable():
    base = B.DEFAULT_BASELINE
    seen = {base.fingerprint(FS)}
    for kw in (dict(method="robust_median"), dict(rough_s=2.0), dict(base_s=0.05),
               dict(decim=50), dict(mask_sigma=4.0), dict(dilate_ms=0.2),
               dict(window_s=0.3), dict(despike_size=3)):
        seen.add(base.replace(**kw).fingerprint(FS))
    assert len(seen) == 9, "two different estimators share a fingerprint"
    assert base.fingerprint(FS) == base.fingerprint(FS)
    assert base.fingerprint(FS).startswith("sha256:")
    # a derived decim is resolved first, so two rates do not share a stamp while
    # running different strides
    assert base.fingerprint(50_000.0) != base.fingerprint(250_000.0)


def test_provenance_round_trips_every_baseline_key():
    """I4. The param name and the provenance name are the same string, so
    ``preprocess_params_from_provenance`` needs no translation-table entry."""
    cfg = B.BaselineConfig(method="masked_median", rough_s=2.0, base_s=0.5, decim=64,
                           mask_sigma=4.0, dilate_ms=0.5, window_s=0.25, despike_size=3)
    prov = cfg.to_provenance(FS)
    assert set(prov) == {
        "baseline_method", "baseline_rough_s", "baseline_base_s", "baseline_decim",
        "baseline_mask_sigma", "baseline_dilate_ms", "baseline_window_s",
        "baseline_despike_size", "baseline_fingerprint",
    }
    assert json.loads(json.dumps(prov)) == prov  # JSON-serialisable, flat
    assert B.BaselineConfig.from_provenance(prov) == cfg


def test_provenance_without_a_method_reconstructs_the_pre_change_estimator():
    """A row with no ``baseline_method`` is a PRE-CHANGE row; describing it as
    anything other than the 10 ms block median that measured it would be a lie, and
    the staleness comparison depends on the two fingerprints differing."""
    old = B.BaselineConfig.from_provenance({"pore_id": "x"})
    assert (old.method, old.window_s) == ("block_median", 0.01)
    assert old.fingerprint(FS) != B.DEFAULT_BASELINE.fingerprint(FS)
    assert B.BaselineConfig.from_provenance(None) == B.DEFAULT_BASELINE


def test_result_provenance_carries_the_measured_keys_too():
    x = _events(200_000)
    r = B.estimate_baseline_result(x, FS, polarity=-1)
    prov = r.provenance(param_source="proposed")
    assert prov["baseline_param_source"] == "proposed"
    assert prov["baseline_fingerprint"] == r.fingerprint
    assert prov["baseline_median_pA"] == pytest.approx(r.median_pA)
    assert prov["baseline_mask_fraction"] == r.health.mask_fraction
    assert prov["baseline_health"]["mask_fraction_ok"] is True
    assert json.loads(json.dumps(prov)) == prov


# ===========================================================================
# c. POLARITY BY MIRRORING
# ===========================================================================
def test_at_polarity_minus_one_the_wrapper_is_a_literal_no_op():
    """P7. The corpus contract holds BY CONSTRUCTION: at pol == -1 there is no
    arithmetic in the wrapper at all, so the bytes that come back are the kernel's."""
    x = _events(300_000)
    for cfg in (B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50),
                B.BaselineConfig(method="robust_median", window_s=0.5, decim=50)):
        kernel = (mp.baseline_masked_median if cfg.method == "masked_median"
                  else mp.baseline_robust_median)
        _, want = kernel(x, FS, **cfg.kernel_kwargs(FS))
        got = B.estimate_baseline_result(x, FS, config=cfg, polarity=-1, health=False)
        assert np.asarray(got.baseline, np.float32).tobytes() == np.asarray(want).tobytes()


def test_polarity_plus_one_is_the_exact_mirror():
    """E6. Enhancement events are real physics below ~400 mM salt, and the reference
    mask is one-sided DOWNWARD with no polarity parameter. Mirroring is exact: every
    step (odd-window order statistics, medians, ``np.interp``, the float32 cast) is
    sign-symmetric in IEEE754."""
    down = _events(300_000, sign=-1.0)
    up = -down
    cfg = B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50)
    a = B.estimate_baseline_result(down, FS, config=cfg, polarity=-1, health=False)
    b = B.estimate_baseline_result(up, FS, config=cfg, polarity=+1, health=False)
    assert (np.asarray(-np.asarray(b.baseline), np.float32).tobytes()
            == np.asarray(a.baseline, np.float32).tobytes())


def test_robust_median_is_exactly_polarity_invariant():
    """It has no mask, and a rolling median of a negated signal is the negation of
    the rolling median EXACTLY -- so the mirror is a no-op both ways. Measured on
    ferritin_data1 as well: max|base(-1) - base(+1)| = 0.0000 pA."""
    x = _events(300_000)
    cfg = B.BaselineConfig(method="robust_median", window_s=0.5, decim=50)
    a = B.estimate_baseline_result(x, FS, config=cfg, polarity=-1, health=False)
    b = B.estimate_baseline_result(x, FS, config=cfg, polarity=+1, health=False)
    assert np.asarray(a.baseline).tobytes() == np.asarray(b.baseline).tobytes()


def test_polarity_changes_the_masked_baseline_where_it_matters():
    """The mirror is not cosmetic: pointing the one-sided mask the wrong way masks
    the noise tail instead of the events. (Measured on the corpus: up to 312 pA of
    baseline difference on ferritin_data2, 282 pA on data3.)"""
    x = _events(300_000, depth=200.0)
    kw = dict(method="masked_median", rough_s=0.5, base_s=0.1, decim=50)
    a = B.estimate_baseline(x, FS, polarity=-1, **kw)
    b = B.estimate_baseline(x, FS, polarity=+1, **kw)
    assert np.abs(a - b).max() > 10.0


# ===========================================================================
# d. LENGTH, FINITENESS AND THE OTHER SILENT-GARBAGE PATHS
# ===========================================================================
def test_a_recording_shorter_than_the_tracking_window_is_refused():
    """E1. The reference raises nothing and warns nothing: at n <= 50 with decim 50
    its 'baseline' is one sample broadcast; at n = 11 the mask comes back 1.0000 and
    the interpolation is skipped; at n = 51/101 the constant is 3.7 pA off."""
    short = _flat(4_000)  # 80 ms, against a 100 ms base_s
    with pytest.raises(ValueError, match="tracking window"):
        B.estimate_baseline_result(short, FS)
    with pytest.raises(ValueError, match="tracking window"):
        B.estimate_baseline(short, FS, method="masked_median", polarity=-1)
    # ...and the reference really does return silent garbage, with no error, at every
    # length this refusal protects against. At n <= 50 with decim 50 there is exactly
    # ONE decimated sample, so the "baseline" is that sample broadcast over the whole
    # recording -- and nothing raises, warns or reports it.
    for n in (2, 3, 5, 11, 50):
        _, ref = mp.baseline_masked_median(_flat(n), FS, rough_s=0.5, base_s=0.1, decim=50)
        assert np.ptp(np.asarray(ref)) == 0.0, "expected one sample broadcast"
    for n in (51, 101, 4_000):
        mp.baseline_masked_median(_flat(n), FS, rough_s=0.5, base_s=0.1, decim=50)


def test_a_recording_shorter_than_the_rough_window_only_warns():
    """E2. Arbitrary, but not inoperative: mask fractions of 1.0000 / 0.4520 /
    0.0902 for rough windows of 0.001 / 0.01 / 0.1 s on the same trace, and no
    error at any length."""
    x = _events(15_000, period=3_000, dwell=300)  # 0.3 s, against a 0.5 s rough_s
    r = B.estimate_baseline_result(x, FS, config=B.BaselineConfig(base_s=0.1, decim=50))
    check = r.health.check("length_rough")
    assert check is not None and check.level == "warn"
    assert "SHORTER than rough_s" in check.message


def test_non_finite_input_is_refused_before_anything_else():
    """E3. Refused, and the refusal names the first bad sample."""
    x = _flat(200_000)
    x[123_456] = np.nan
    with pytest.raises(ValueError, match=r"first at index 123,456"):
        B.estimate_baseline_result(x, FS)
    x[123_456] = np.inf
    with pytest.raises(ValueError):
        B.estimate_baseline_result(x, FS)
    with pytest.raises(ValueError, match="empty recording"):
        B.require_finite(np.empty(0, np.float32))


def test_a_nan_reaching_the_kernel_silently_disables_the_whole_event_mask():
    """E3, the reason the finiteness check is a REFUSAL and sits before the notch.

    This is documented behaviour of the reference, not a supported path: one NaN
    makes ``noise`` NaN, so ``rough < median(rough) - 3.5*noise`` is all-False and
    the estimator degrades to a plain short-window median with no error and no
    warning.
    """
    x = _flat(200_000)
    x[100_000] = np.nan
    rough = x - mp._decimated_median(x, FS, 0.5, 50)
    noise = 1.4826 * np.median(np.abs(rough - np.median(rough)))
    assert not np.isfinite(noise)
    assert not (rough < np.median(rough) - 3.5 * noise).any(), "the mask is disabled"


def test_an_infinity_does_not_corrupt_the_kernels_mask():
    """E4. +-inf sorts correctly inside a median, so the reference tolerates it
    (measured mask fraction 0.0234 against 0.0235 clean). The app still refuses it --
    a baseline fitted through an infinity is not a number anyone should act on --
    but the distinction is worth pinning, because it is why NaN and inf get
    different explanations."""
    x = _flat(200_000)
    x[100_000] = np.inf
    _, base = mp.baseline_masked_median(x, FS, rough_s=0.5, base_s=0.1, decim=50)
    assert np.isfinite(np.asarray(base)).all()


def test_the_clamped_constant_tail_when_n_is_not_divisible_by_decim():
    """E9. ``np.interp`` clamps beyond the last knot, so the final up-to-(decim-1)
    samples carry a CONSTANT baseline. It is not always zero-length -- do not rely
    on that."""
    n = 500_003
    x = _flat(n)
    b = B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                            rough_s=0.5, base_s=0.1, decim=50)
    last_knot = 50 * ((n - 1) // 50)
    assert n - 1 - last_knot == 2
    assert np.ptp(b[last_knot:]) == 0.0


def test_a_zero_sigma_trace_masks_nothing_and_does_not_raise():
    """E8, the ``if evm.any() and (~evm).sum() > 2`` guard's other branch. It is NOT
    a duty-cycle guard -- it fires only when two or fewer samples remain unmasked."""
    x = np.full(60_000, -100.0, np.float32)
    _, mask, sigma, _ = B._rough_pass(x, FS, config=B.DEFAULT_BASELINE)
    assert sigma == 0.0 and not mask.any()
    b = B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                            rough_s=0.5, base_s=0.1, decim=50)
    assert np.isfinite(b).all() and np.ptp(b) == 0.0


# ===========================================================================
# e. HEALTH -- what the reference cannot tell you about its own answer
# ===========================================================================
def test_the_event_free_mask_fraction_is_the_dilated_false_positive_rate():
    """S5, and the whole reason the floor is 2.5 % rather than 0.

    The +-1 ms dilation multiplies the 3.5-sigma false-positive rate by 2k+1 = 101,
    so an event-free trace masks ``Phi(-3.5) * 101`` = 2.35 % BY CONSTRUCTION. A mask
    fraction near that number means "no events found", not "events found" -- and the
    floor check must say so.

    It must NOT say only that, which is what it used to. The identical check pair
    (``mask_fraction_floor`` + ``overshoot``, same levels) fires on the >50%-duty
    INVERSION, where the mask has landed on the open pore and the baseline sits 195 pA
    away at the blockade level. Same reading, opposite meaning, and nothing in the
    health block separates them -- so the message names both and says to look at the
    trace, rather than asserting the benign one. Both halves are asserted here and in
    ``test_the_inversion_and_the_empty_trace_are_not_told_apart_by_the_checks``.
    """
    x = _flat(1_000_000)
    r = B.estimate_baseline_result(x, FS, polarity=-1)
    h = r.health
    predicted = 0.5 * math.erfc(3.5 / math.sqrt(2.0)) * 101
    assert h.mask_fraction_event_free == pytest.approx(predicted, rel=1e-12)
    assert h.mask_fraction == pytest.approx(predicted, abs=0.005)
    assert abs(r.median_pA + 100.0) < 0.5, "event-free bias must be under 0.5 pA"
    floor = h.check("mask_fraction_floor")
    assert floor is not None and floor.level == "warn"
    assert "there really are no events" in floor.message
    assert "INVERTED" in floor.message


def test_the_inversion_and_the_empty_trace_are_not_told_apart_by_the_checks():
    """The floor message's second reading, measured.

    An event-free trace and a 60%-duty inversion produce the SAME two warnings at the
    SAME levels: one has a baseline correct to 0.02 pA, the other one 195 pA wrong
    with its "events" being the open pore. A message that asserted the first was
    telling a user with the second that his events were missing."""
    empty = _flat(1_000_000)
    r_empty = B.estimate_baseline_result(empty, FS, polarity=-1)
    busy = _events(1_000_000, depth=200.0, dwell=6_000, period=10_000)   # 60% duty
    r_busy = B.estimate_baseline_result(busy, FS, polarity=-1)

    fired = lambda h: sorted(c.name for c in h.checks if c.level in ("warn", "refuse"))
    assert fired(r_empty.health) == fired(r_busy.health) == [
        "mask_fraction_floor", "overshoot"]
    assert abs(r_empty.median_pA + 100.0) < 0.5
    assert abs(r_busy.median_pA + 100.0) > 100.0, (
        f"the inversion did not invert ({r_busy.median_pA:.1f} pA); rewrite this test "
        "against a duty that does, but do not delete the point"
    )


def test_a_healthy_trace_passes_the_mask_fraction_band():
    x = _events(1_000_000, depth=80.0, dwell=1000, period=10_000)
    h = B.estimate_baseline_result(x, FS, polarity=-1).health
    assert B.MASK_FRACTION_FLOOR < h.mask_fraction < B.MASK_FRACTION_CEILING
    assert h.check("mask_fraction").level == "ok"
    assert h.check("rough_sigma_ratio").level == "ok"


def test_ninety_five_percent_blockade_duty_is_caught():
    """E5. Occupancy past 50 % is a hard INVERSION, not a degradation: the rough
    median sits on the blocked level, the true open pore looks like an upward
    excursion, and the one-sided mask never fires. The reference returns a -299.6 pA
    baseline for a -100 pA open pore with no warning and no exception."""
    x = _events(1_000_000, depth=200.0, dwell=950, period=1_000)
    r = B.estimate_baseline_result(x, FS, polarity=-1)
    assert r.median_pA < -250.0, "expected the inverted baseline this test is about"
    bad = {c.name for c in r.health.checks if not c.ok}
    assert "mask_fraction_floor" in bad and "overshoot" in bad
    assert r.health.overshoot > 100.0
    assert not r.health.ok


def test_drift_faster_than_the_rough_window_is_caught():
    """E7. A 300 pA sinusoid at a 0.2 s period takes the rough-pass sigma from ~9 to
    ~262 pA, puts the 3.5-sigma threshold out of reach, and collapses the mask to
    EXACTLY zero while the median baseline error hits tens of pA."""
    x = _drifting(1_000_000)
    h = B.estimate_baseline_result(x, FS, polarity=-1).health
    assert h.mask_fraction == 0.0
    assert h.rough_sigma_ratio > B.ROUGH_SIGMA_RATIO_MAX
    bad = {c.name for c in h.checks if not c.ok}
    assert {"mask_fraction_floor", "rough_sigma_ratio"} <= bad
    assert "measuring DRIFT" in h.check("rough_sigma_ratio").message


def test_health_is_structured_data_not_log_lines():
    """The GUI has to render this, badge the affected columns and put it in a
    provenance sidecar -- none of which is possible with a warning stream."""
    x = _events(300_000)
    h = B.estimate_baseline_result(x, FS, polarity=-1).health
    assert isinstance(h, B.BaselineHealth)
    assert all(isinstance(c, B.BaselineCheck) for c in h.checks)
    assert {c.name for c in h.checks} >= {"length_tracking", "length_rough",
                                          "rough_sigma_ratio", "overshoot"}
    assert all(c.level in ("ok", "warn", "refuse") for c in h.checks)
    d = h.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["ok"] is h.ok and d["warnings"] == list(h.warnings)


def test_the_non_masked_methods_still_get_a_health_block():
    """One struct for all five methods, or the GUI needs two renderers. The mask
    diagnostics are None where there is no pass 1."""
    x = _events(300_000)
    h = B.estimate_baseline_result(
        x, FS, config=B.BaselineConfig(method="block_median", window_s=0.01),
        polarity=-1).health
    assert h.mask_fraction is None and h.rough_sigma_ratio is None
    assert np.isfinite(h.noise_pA) and np.isfinite(h.overshoot)
    assert h.check("length_rough") is None


def test_the_diagnostic_rough_pass_reconstructs_the_kernel_exactly():
    """The one thing that makes recomputing pass 1 for diagnostics safe.

    `_rough_pass` re-derives the kernel's mask from the same imported
    `decimated_median`. Feeding that mask through the kernel's own fill + pass 2 must
    give back `baseline_masked_median`'s answer BYTE FOR BYTE -- which is only true
    if the mask is the kernel's mask. If the kernel's masking ever changes, this
    fails instead of the diagnostics quietly describing a different algorithm.
    """
    x = _events(300_000, depth=120.0)
    cfg = B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50)
    _, mask, _, _ = B._rough_pass(x, FS, config=cfg)
    idx = np.arange(x.size)
    filled = x.copy()
    if mask.any() and (~mask).sum() > 2:
        filled[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    rebuilt = B.decimated_median(filled, FS, cfg.base_s, 50)
    _, want = mp.baseline_masked_median(x, FS, **cfg.kernel_kwargs(FS))
    assert np.asarray(rebuilt).tobytes() == np.asarray(want).tobytes()


def test_the_qc_triple_is_a_report_and_never_an_objective():
    """R5, honestly stated. ``level`` / ``drift`` / ``noise`` are MONOTONICALLY
    improved by shrinking the window, so an optimiser over them selects the shortest
    possible window -- exactly the failure being fixed.

    Note what this test does NOT claim: the section-2.7 health checks do **not**
    reject the degenerate set. Measured on the corpus, ``rough_s=0.01 /
    base_s=0.002 / decim=1`` keeps the mask fraction in band (data2 0.0762, data3
    0.1200) and the sigma ratio near 1 (1.20, 1.45), so every check passes. Only
    ``overshoot`` degrades (0.093 -> 0.178 on data2, 0.099 -> 0.139 on data3), and it
    stays under the 0.3 flag. **The defence against auto-tuning is architectural --
    propose and confirm, never auto-apply -- not a health check.**
    """
    x = _events(1_000_000, depth=80.0, dwell=1000, period=10_000)
    shipped = B.estimate_baseline_result(
        x, FS, config=B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50),
        polarity=-1).health
    degen = B.estimate_baseline_result(
        x, FS, config=B.BaselineConfig(rough_s=0.01, base_s=0.002, decim=1),
        polarity=-1).health
    # the three seductive metrics all "improve"...
    assert abs(degen.open_pore_level_pA) <= abs(shipped.open_pore_level_pA)
    assert degen.open_pore_drift_pA <= shipped.open_pore_drift_pA
    assert degen.noise_pA <= shipped.noise_pA
    # ...while the one that resists gets worse.
    assert degen.overshoot > shipped.overshoot


# ===========================================================================
# f. DETERMINISM AND CANCELLATION
# ===========================================================================
def test_twenty_runs_under_allocation_churn_are_byte_identical():
    """P9. The 1-D ``median_filter`` non-determinism has reached production in this
    project once already: six different answers in forty runs, spanning 6.4 pA, with
    a fixed 5-sigma decode counting 337-469 events."""
    x = _events(400_000, depth=120.0)
    rng = np.random.default_rng(0)
    seen = set()
    for i in range(20):
        junk = [rng.standard_normal(int(rng.integers(10_000, 200_000))) for _ in range(4)]
        b = B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                                rough_s=0.5, base_s=0.1, decim=50)
        seen.add(hashlib.sha256(np.asarray(b, np.float64).tobytes()).hexdigest())
        del junk
        if i % 4 == 0:
            gc.collect()
    assert len(seen) == 1, f"{len(seen)} distinct baselines in 20 runs"


def test_the_estimator_polls_the_ambient_cancel_token():
    """Cancellation is an ambient ContextVar token, not a keyword threaded by hand,
    so a new kernel becomes cancellable by polling -- no call site changes."""
    x = _events(400_000)
    for kw in (dict(method="masked_median", rough_s=0.5, base_s=0.1, decim=50),
               dict(method="robust_median", window_s=0.5, decim=50),
               dict(method="block_median", window_s=0.01),
               dict(method="block_median", window_s=0.01, two_pass=True)):
        with cancel_scope(lambda: True):
            with pytest.raises(OperationCancelled):
                B.estimate_baseline(x, FS, polarity=-1, **kw)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            B.estimate_baseline_result(x, FS, polarity=-1)


def test_cancellation_is_scoped_to_the_thread_that_installed_it():
    x = _events(200_000)
    out: list = []

    def worker():
        try:
            out.append(B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                                           rough_s=0.5, base_s=0.1, decim=50).size)
        except OperationCancelled:  # pragma: no cover - the failure this guards
            out.append("cancelled")

    with cancel_scope(lambda: True):
        t = threading.Thread(target=worker)
        t.start()
        t.join(30)
    assert out == [x.size]


# ===========================================================================
# g. THE FOUR LEGACY ESTIMATORS DID NOT MOVE
#
# These digests were captured from the current code AFTER proving it byte-identical
# to the pre-batch PyInstaller build (`dist/NanoScope/NanoScope`, Jul 28 21:13,
# which embeds the pre-change `nanoscope.analysis.baseline` as marshalled bytecode
# and demonstrably predates this work -- its METHODS tuple has four entries and its
# ``window_s`` default is 0.01). 48 method x polarity x two_pass combinations, four
# explicit windows, `remove_baseline_wander`, `estimated_wander`, `derive_polarity`,
# `open_pore_current` and `baseline_intrusion` all came back byte-identical.
#
# NEVER "update" these. A failure means an additive change stopped being additive.
# ===========================================================================
_GOLDEN_TRACE_SHA = "9c15f47cf34587f895b60601a7bdf388f64e054df02ddb7847f812212d4b7a91"
_GOLDEN_LEGACY = {
    ("block_median", False): "34b1d9442c593d089d85ab9489fe3928e0bc956f1febfe38052a71114f097d73",
    ("block_median", True): "528f39fe7419c76723782111163aea34e6e375f84058191c7ca6f1a4217d4305",
    ("percentile", False): "75244331d8cc507434ed646b383946518b1b0b216759234ce98cb9d0e0988bcc",
    ("percentile", True): "4dd7c0552b9607e8d07eec70524d945f2b4c750ddafd195dd244887bee95da28",
    ("asls", False): "760b4aa1318bc4fe13a0f787ab93716ccb0178c08b1823e035bf05a4423b2c9c",
    ("asls", True): "4ddebbee98730dc0baef2c83e5f46dd9b77091d218381ebf79b58dfa83562299",
    ("constant", False): "77b84b7adff13250a126c17f7f3ccfa297f4456355515ed2bc0208bf23f83a39",
    ("constant", True): "77b84b7adff13250a126c17f7f3ccfa297f4456355515ed2bc0208bf23f83a39",
}


def _legacy_trace(n=200_000, seed=20260801):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = -1000.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 15.0, n)
    for k in range(40):
        s = 1500 + k * 5000
        x[s:s + 600] += 300.0
    return x.astype(np.float32)


def test_the_legacy_fixture_itself_is_unchanged():
    """If this fails the fixture drifted and the digests below prove nothing."""
    assert hashlib.sha256(_legacy_trace().tobytes()).hexdigest() == _GOLDEN_TRACE_SHA


@pytest.mark.parametrize("key", sorted(_GOLDEN_LEGACY))
def test_legacy_estimator_output_is_byte_for_byte_what_it_was(key):
    method, two_pass = key
    x = _legacy_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", B.BaselineWarning)
        b = B.estimate_baseline(x, FS, method=method, two_pass=two_pass, polarity=+1)
    assert hashlib.sha256(b.tobytes()).hexdigest() == _GOLDEN_LEGACY[key]


def test_methods_grew_additively_and_kept_its_order():
    """UPDATED when ``'savgol'`` was added (the Cu2-insulin decimated Savitzky-Golay
    baseline: SG_WINDOW_S 0.2 / SG_DECIM 50 / SG_POLY 3, the method all four written
    ``cu_*_metadata.json`` name). The guarantee this test exists for is ADDITIVE-ONLY
    WITH A STABLE ORDER -- the frozen four first, the two imported-kernel methods
    next, anything new on the END where no positional index can move -- not a fixed
    LENGTH. The two assertions that carry that guarantee are unchanged; the third is
    re-spelled to name the new tail rather than to forbid one."""
    assert B.METHODS[:4] == ("block_median", "percentile", "asls", "constant")
    assert B.MASKED_METHODS == ("masked_median", "robust_median")
    assert B.METHODS == (*B.METHODS[:4], *B.MASKED_METHODS, "savgol")


# ===========================================================================
# h. THE ARITHMETIC THAT DECIDES BIT-EXACTNESS
# ===========================================================================
@pytest.mark.parametrize("win_s, decim, taps", [
    (0.5, 50, 501),    # ferritin_data1 single pass / data2 rough
    (0.1, 50, 101),    # data2 base
    (0.03, 10, 151),   # data3 rough
    (0.01, 10, 51),    # data3 base
])
def test_tap_counts_at_the_five_operating_points(win_s, decim, taps):
    """P5. ``w = int(win_s * fs / decim)`` -- truncation toward zero -- then forced
    ODD by adding one. Windows in SECONDS, on the DECIMATED grid; not tap counts and
    not block sizes."""
    w = int(win_s * FS / decim)
    assert (w + 1 if w % 2 == 0 else w) == taps


def test_dilation_grows_the_mask_by_exactly_k_samples_and_does_not_wrap():
    """P6. scipy's default 1-D structure is [1,1,1] and ``border_value`` defaults to
    0, so ``iterations=k`` grows the mask by exactly k samples each side and the
    mask does not bleed in from the array ends. ``iterations=50`` at 50 kHz is the
    +-1 ms of the default."""
    from scipy.ndimage import binary_dilation

    m = np.zeros(100, bool)
    m[10] = True
    assert np.flatnonzero(binary_dilation(m, iterations=3)).tolist() == list(range(7, 14))
    m = np.zeros(100, bool)
    m[0] = True
    assert np.flatnonzero(binary_dilation(m, iterations=3)).tolist() == [0, 1, 2, 3]
    m = np.zeros(1000, bool)
    m[500] = True
    assert int(binary_dilation(m, iterations=50).sum()) == 101
    assert max(1, int(1.0 * 1e-3 * FS)) == 50


def test_a_subrange_baseline_is_not_a_slice_of_the_full_one():
    """R10, asserted as EXPECTED behaviour so nobody "fixes" it.

    The mask threshold comes from a GLOBAL MAD over the whole trace, so estimating a
    baseline on a user's zoom rectangle genuinely differs from the same window cut
    out of a full-trace estimate (measured penalty on real data: 16.95 pA, more than
    1 sigma). The notebook truncates the RAW signal before filtering for exactly this
    reason, and this app measures over the WHOLE channel.
    """
    x = _events(600_000, depth=150.0)
    m = 200_000
    full = B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                               rough_s=0.5, base_s=0.1, decim=50)
    part = B.estimate_baseline(x[:m], FS, method="masked_median", polarity=-1,
                               rough_s=0.5, base_s=0.1, decim=50)
    assert part.tobytes() != full[:m].tobytes()


# ===========================================================================
# i. THE CORPUS PINS -- never re-baseline these
# ===========================================================================
_CORPUS_CASES = {
    "ferritin_data1": B.BaselineConfig(method="robust_median", window_s=0.5, decim=50),
    "ferritin_data2": B.BaselineConfig(method="masked_median", rough_s=0.5, base_s=0.1,
                                       decim=50, despike_size=3),
    "ferritin_data3": B.BaselineConfig(method="masked_median", rough_s=0.03, base_s=0.01,
                                       decim=10),
}


@pytest.mark.corpus
@pytest.mark.parametrize("tag", sorted(_CORPUS_CASES))
def test_the_wired_up_path_reproduces_the_training_corpus_bit_exactly(tag):
    """P1-P4. ``max|diff| = 0`` -- byte-identical float32 -- for ``baseline_pA`` and
    ``current_detr_pA``, and the four stored QC scalars to every bit.

    This is the contract. If it moves, the estimator moved, and every model trained
    on those columns is describing a different signal chain.
    """
    import pandas as pd

    csv = CORPUS / f"{tag}.csv.gz"
    if not csv.exists():
        pytest.skip(f"corpus absent: {csv}")
    cfg = _CORPUS_CASES[tag]
    df = pd.read_csv(csv, usecols=["current_raw_pA", "baseline_pA", "current_detr_pA"])
    raw = np.asarray(df["current_raw_pA"], np.float32)
    want_base = np.asarray(df["baseline_pA"], np.float32)
    want_detr = np.asarray(df["current_detr_pA"], np.float32)
    del df

    filt = mp.notch_sequential(raw, FS)
    # polarity is passed EXPLICITLY: derive_polarity reads +1 on the raw, drifting
    # data2/data3 traces (it reads -1 on the detrended ones), and the corpus was
    # produced by the downward-masking reference. See the seam note in the module.
    res = B.estimate_baseline_result(filt, FS, config=cfg, polarity=-1, health=True)

    got_base = np.asarray(res.baseline, np.float32)
    got_detr = (filt - res.baseline).astype(np.float32)
    if cfg.despike_size:
        got_detr = mp.despike_median(got_detr, size=cfg.despike_size)
    assert got_base.tobytes() == want_base.tobytes()
    assert got_detr.tobytes() == want_detr.tobytes()

    meta = json.loads((CORPUS / f"{tag}_metadata.json").read_text())
    h = res.health
    assert h.noise_pA == meta["qc_mad_noise_pA"]
    assert h.open_pore_level_pA == meta["qc_open_pore_level_pA"]
    assert h.open_pore_drift_pA == meta["qc_open_pore_drift_pA"]
    assert h.overshoot == meta["qc_overshoot"]

    # P7: the polarity wrapper contributed nothing.
    kernel = (mp.baseline_masked_median if cfg.method == "masked_median"
              else mp.baseline_robust_median)
    _, kbase = kernel(filt, FS, **cfg.kernel_kwargs(FS))
    assert got_base.tobytes() == np.asarray(kbase).tobytes()


@pytest.mark.corpus
def test_the_shipped_corpus_mask_fractions():
    """Re-measured for the build spec and pinned here: these two numbers are what
    the mask-fraction band was calibrated against."""
    import pandas as pd

    for tag, want in (("ferritin_data2", 0.0560), ("ferritin_data3", 0.1239)):
        csv = CORPUS / f"{tag}.csv.gz"
        if not csv.exists():
            pytest.skip(f"corpus absent: {csv}")
        raw = np.asarray(pd.read_csv(csv, usecols=["current_raw_pA"])["current_raw_pA"],
                         np.float32)
        filt = mp.notch_sequential(raw, FS)
        h = B.estimate_baseline_result(filt, FS, config=_CORPUS_CASES[tag],
                                       polarity=-1).health
        assert h.mask_fraction == pytest.approx(want, abs=5e-5)
        assert B.MASK_FRACTION_FLOOR < h.mask_fraction < B.MASK_FRACTION_CEILING
        assert h.ok, f"{tag} at its own shipped parameters must be healthy: {h.warnings}"


@pytest.mark.corpus
def test_derive_polarity_reads_plus_one_on_the_raw_corpus_traces():
    """THE SEAM HAZARD, pinned so it cannot be discovered again by accident.

    ``derive_polarity`` measures from a GLOBAL-MEDIAN centre. On a drifting or
    gating raw trace the excursions it counts are the drift, not the events, so it
    returns +1 for ferritin_data2 and data3 -- and it returns -1 on the same
    recordings once they are detrended. Pointing the one-sided mask the wrong way
    moves the baseline by up to 312 pA (data2) / 282 pA (data3).

    `derive_polarity`'s behaviour must NOT change (it is frozen). What must happen
    is that the seam feeds `estimate_baseline_result` a polarity it trusts -- the
    detector's own ``polarity_source`` provenance, or one measured on a detrended
    trace -- rather than ``polarity=None`` on a raw one.
    """
    import pandas as pd

    for tag in ("ferritin_data2", "ferritin_data3"):
        csv = CORPUS / f"{tag}.csv.gz"
        if not csv.exists():
            pytest.skip(f"corpus absent: {csv}")
        df = pd.read_csv(csv, usecols=["current_raw_pA", "current_detr_pA"])
        raw = np.asarray(df["current_raw_pA"], np.float32)
        detr = np.asarray(df["current_detr_pA"], np.float32)
        del df
        assert B.derive_polarity(raw) == 1
        assert B.derive_polarity(detr) == -1


@pytest.mark.corpus
@pytest.mark.parametrize("tag", ["ferritin_data1", "ferritin_data2", "ferritin_data3"])
def test_the_door_reads_the_corpus_polarity_from_the_data_at_every_configuration(tag):
    """The seam's whole job, on the recordings the contract is written against.

    ``-1`` is the direction the reference produced these three columns with (see
    ``test_the_wired_up_path_reproduces_the_training_corpus_bit_exactly``, which passes
    it explicitly), and it must fall out of the DATA -- at every configuration the app
    can be in, because the mask direction is a property of the recording and not of
    anybody's window.

    Before the scale ladder this was a single detrend at the configured ``rough_s``,
    with a stride derived from ``base_s``: on ``ferritin_data3`` at a 0.5 s window,
    stride 264 answers -1 and stride 333 answers +1, and at the baseline probe's own
    5 s frame nothing crosses 5 sigma at all so `derive_polarity` returns its
    documented flat-trace ``+1`` -- a default reported as a measurement. Pointing the
    one-sided mask backwards moves the baseline by 312 pA on data2 and 282 on data3.
    """
    import pandas as pd

    csv = CORPUS / f"{tag}.csv.gz"
    if not csv.exists():
        pytest.skip(f"corpus absent: {csv}")
    raw = np.asarray(pd.read_csv(csv, usecols=["current_raw_pA"])["current_raw_pA"],
                     np.float32)
    filt = mp.notch_sequential(raw, FS)
    del raw

    configs = (
        B.DEFAULT_BASELINE,
        _CORPUS_CASES["ferritin_data1"],
        _CORPUS_CASES["ferritin_data2"],
        _CORPUS_CASES["ferritin_data3"],
        B.BaselineConfig(rough_s=2.0, base_s=0.5),          # the apo file's own
        B.BaselineConfig(rough_s=5.0, base_s=1.25),         # the probe's own frame
        B.BaselineConfig(method="block_median", window_s=0.01),
    )
    # A FRESH ARRAY PER CONFIGURATION. `_PolarityMemo` is keyed on the trace's object
    # identity, so re-asking about the same object returns the first answer and this
    # assertion would hold no matter what the ladder did. A copy carries the same
    # samples and misses the memo, so every configuration is genuinely re-measured --
    # which is what makes this test discriminate. (It fails on ferritin_data3 against
    # the pre-batch single-window derivation, memo or no memo.)
    assert {B.measured_polarity(filt.copy(), FS, config=c) for c in configs} == {-1}
    assert B.measured_polarity(filt.copy(), FS) == -1


@pytest.mark.corpus
def test_masked_median_beats_the_block_median_on_the_real_apo_recording():
    """S1-S3, and THE regression test for the defect this whole change exists to fix.

    On ``..._0006_Process_-600 mV.abf`` the 10 ms block median stamps most events
    with a baseline that is inside the event. Three independent estimates of the true
    open pore on that file -- the modal current, the global median and a GLOBALLY
    computed 10 ms block median -- all land near -35 pA; the per-event stamped
    baseline lands near -100.
    """
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document

    if not APO_ABF.exists():
        pytest.skip(f"corpus ABF absent: {APO_ABF.name}")
    doc = open_document(APO_ABF)
    ch = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT][0]
    x = np.asarray(doc.channels[ch.index], np.float32)
    assert x.size == 2_993_798

    masked = B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                                 rough_s=0.5, base_s=0.1, decim=50)
    tuned = B.estimate_baseline(x, FS, method="masked_median", polarity=-1,
                                rough_s=2.0, base_s=0.5, decim=50)
    block = B.estimate_baseline(x, FS, polarity=-1, window_s=0.01)

    # S1: the masked baseline agrees with the mode of the current to within 2 pA.
    # Measured: modal -34.426, global median -34.790, masked median -34.491.
    hist, edges = np.histogram(x, bins=2000)
    modal = float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
    assert abs(float(np.median(masked)) - modal) < 2.0
    assert abs(float(np.median(x)) - modal) < 2.0, "the two independent estimates agree"

    # events, from the estimator's own mask -- no detector needed
    _, mask, _, _ = B._rough_pass(x, FS, config=B.BaselineConfig(rough_s=0.5, base_s=0.1,
                                                                decim=50))
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= int(2e-3 * FS)
    starts, ends = starts[keep], ends[keep]
    assert starts.size > 100, "expected a few hundred events on this recording"

    stamp_m = np.array([np.median(masked[s:e]) for s, e in zip(starts, ends)])
    stamp_t = np.array([np.median(tuned[s:e]) for s, e in zip(starts, ends)])
    stamp_b = np.array([np.median(block[s:e]) for s, e in zip(starts, ends)])
    level = np.array([np.median(x[s:e]) for s, e in zip(starts, ends)])

    # S2: fraction of events stamped more than 10 pA from the true open pore.
    # Measured over these 354 regions: block median 0.822, masked at the DEFAULT
    # parameters 0.124, masked at this recording's measured optimum 0.0000.
    off_b = float(np.mean(np.abs(stamp_b - modal) > 10.0))
    off_m = float(np.mean(np.abs(stamp_m - modal) > 10.0))
    off_t = float(np.mean(np.abs(stamp_t - modal) > 10.0))
    assert off_b > 0.50
    assert off_m < 0.20 and off_m < off_b / 5.0

    # R4/2.5: and the residual 12 % is exactly why parameters are PER-RECORDING.
    # This file has a 521 ms partial-clog state at t = 37.13 s; a 500 ms rough window
    # sitting inside a 521 ms state *is* that state, so the rough residual there is
    # ~0 and nothing gets masked (coverage 15.1 % at rough_s 0.5, 99.96 % at 2.0).
    # One global default cannot exist: this file wants 2.0 s, ferritin_data3 wants
    # 0.03 s -- a factor of 67.
    assert off_t == 0.0, "the file's measured optimum must clear the last 12 %"

    # S3: median depth under masked_median exceeds the block median's by > 5x.
    # Measured 34.8x here (2.35 pA -> 81.76 pA): the wrong baseline sits INSIDE the
    # event, so the residual it leaves is nearly zero and the depth ratio is far
    # larger than the ~3x ratio of the two baselines.
    depth_m = float(np.median(np.abs(level - stamp_m)))
    depth_b = float(np.median(np.abs(level - stamp_b)))
    assert depth_m / max(depth_b, 1e-9) > 5.0


@pytest.mark.corpus
def test_the_default_is_worse_than_the_estimator_it_replaced_on_the_gating_pore():
    """The one recording where this change makes things WORSE unless the user probes,
    measured, and pinned so nobody has to rediscover it.

    Against ``ferritin_data3``'s own persisted ``baseline_pA`` column -- the one the
    owner's notebook produced for that file -- median |error|:

        block median, 10 ms (what this change replaces)    3.71 pA
        masked median at DEFAULT_BASELINE, 0.5 / 0.1 s    12.36 pA
        masked median at his own 0.03 / 0.01 s, decim 10    2.55 pA

    and at the correct polarity the default's health block reads ``ok`` on all five
    checks with an EMPTY ``warnings``. So the propose-and-confirm surface is not a
    convenience on this file: without it the depths are 3.3x further from the truth
    than before this build, and no check says so.

    What stops that being silent is `UNPROBED_DEFAULT_WARNING`, asserted here on the
    trace that motivates it rather than on a synthetic.
    """
    import pandas as pd

    csv = CORPUS / "ferritin_data3.csv.gz"
    if not csv.exists():
        pytest.skip(f"corpus absent: {csv}")
    df = pd.read_csv(csv, usecols=["current_raw_pA", "baseline_pA"])
    raw = np.asarray(df["current_raw_pA"], np.float64)
    truth = np.asarray(df["baseline_pA"], np.float64)
    del df

    def med_err(cfg, param_source="user"):
        r = B.characterisation_baseline(raw, FS, config=cfg, polarity=-1,
                                        param_source=param_source, health=True)
        return float(np.median(np.abs(r.baseline - truth))), r

    old, _ = med_err(B.BaselineConfig(method="block_median", window_s=0.01))
    default, res = med_err(B.DEFAULT_BASELINE, param_source="default")
    his, _ = med_err(_CORPUS_CASES["ferritin_data3"])

    assert old == pytest.approx(3.71, abs=0.05)
    assert default == pytest.approx(12.36, abs=0.05)
    assert his == pytest.approx(2.55, abs=0.05)
    assert default > 3.0 * old, (
        f"the default is no longer worse than the block median here ({default:.2f} "
        f"vs {old:.2f}); if that is a real improvement, rewrite this test, but do "
        "not delete the measurement"
    )

    # ...and the health block is no help at all, which is the point.
    assert res.health is not None and res.health.ok
    assert not res.health.warnings

    # The only thing that speaks is the advisory, and it is on the result.
    assert B.UNPROBED_DEFAULT_WARNING in res.warnings
    assert "Measure this recording" in B.UNPROBED_DEFAULT_WARNING

    # A user who has confirmed a window does not get told to go and confirm one.
    confirmed = B.characterisation_baseline(raw, FS, config=_CORPUS_CASES["ferritin_data3"],
                                            polarity=-1, param_source="proposed", health=True)
    assert B.UNPROBED_DEFAULT_WARNING not in confirmed.warnings


# ===========================================================================
# g. A SEGMENT SHORT ENOUGH TO NEED CLAMPING IS STILL MEASURED, AND SOUNDLY
# ===========================================================================
def _apo_events(x):
    """Event regions on the apo file, from the estimator's own mask -- no detector."""
    _, mask, _, _ = B._rough_pass(
        x, FS, config=B.BaselineConfig(rough_s=0.5, base_s=0.1, decim=50))
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= int(2e-3 * FS)
    return starts[keep], ends[keep]


@pytest.mark.corpus
def test_a_clamped_segment_reproduces_the_whole_recording_depth():
    """THE reason the length fallback changed, on the owner's own recording.

    A hand-cut segment is a legal document (``MIN_SEGMENT_SAMPLES`` is 2) and the
    ML model's own window is 3 072 samples, so 3 072 samples is a length this app has
    to answer for. Cut 3 072 samples around each of ~27 real events and ask each
    candidate estimator for that event's depth, against the depth the WHOLE-RECORDING
    masked median gives the same event:

        clamped masked median   0.958 of it   (measured, median over the events)
        block median at 10 ms   0.011 of it   -- a 90x understatement

    The block median is what used to stand in here, and it is the very failure the
    two-pass estimator was brought in to remove: its baseline sits inside the event,
    so the residual it leaves is nearly nothing.
    """
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document

    if not APO_ABF.exists():
        pytest.skip(f"corpus ABF absent: {APO_ABF.name}")
    doc = open_document(APO_ABF)
    ch = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT][0]
    x = np.asarray(doc.channels[ch.index], np.float32)

    whole = B.estimate_baseline_result(x, FS, config=B.DEFAULT_BASELINE,
                                       polarity=-1, health=False).baseline
    starts, ends = _apo_events(x)

    L = 3072                                   # the model's own window, 61.4 ms
    clamped_cfg, token = B.adapt_to_length(B.DEFAULT_BASELINE, L, FS)
    assert token == B.FALLBACK_CLAMPED and clamped_cfg.method == "masked_median"
    block_cfg = B.BaselineConfig(method="block_median", window_s=B._LEGACY_WINDOW_S)

    ratio_clamped, ratio_block = [], []
    for s0, e0 in zip(starts, ends):
        if (e0 - s0) >= L:
            continue
        a = max(0, min(x.size - L, (s0 + e0) // 2 - L // 2))
        if not (a <= s0 and e0 <= a + L):
            continue
        level = float(np.median(x[s0:e0]))
        want = abs(level - float(np.median(whole[s0:e0])))
        if want < 20.0:                        # only unambiguous, deep events
            continue
        seg = x[a:a + L]
        for cfg, out in ((clamped_cfg, ratio_clamped), (block_cfg, ratio_block)):
            b = B.estimate_baseline_result(seg, FS, config=cfg, polarity=-1,
                                           health=False).baseline
            out.append(abs(level - float(np.median(b[s0 - a:e0 - a]))) / want)

    assert len(ratio_clamped) > 15, "expected a couple of dozen usable events"
    med_clamped = float(np.median(ratio_clamped))
    med_block = float(np.median(ratio_block))
    assert med_clamped == pytest.approx(0.958, abs=0.05)
    assert med_block < 0.10, (
        f"the block median reports {med_block:.3f} of the whole-recording depth; if "
        "that has changed, re-measure before believing it"
    )
    assert med_clamped > 20.0 * med_block


@pytest.mark.corpus
def test_the_clamp_is_wrong_less_often_and_silent_far_less_often_than_what_it_replaced():
    """Clamping is not sound at EVERY length, and the claim that carries this change
    is not "it never fails" -- it is "it fails less, and says so far more often" than
    the block median that used to stand in here.

    At 1 536 samples (30.7 ms) the apo file's ~20 ms events fill two thirds of the
    clamped rough window, so the rough median IS the event level, nothing is masked,
    and the depth collapses exactly as the block median's does. Measured over all 236
    usable events -- segments whose depth came out below half the whole-recording
    answer, and how many of THOSE reported ``health.ok``:

        segment length     1 536                2 048          3 072       5 000
        clamped masked     222 wrong, 13 silent  50 / 0        0 / 0       1 / 0
        block median       236 wrong, 207 silent 236 / 235     236 / 19    237 / 16

    (The obvious extra guard -- ``rough_s`` against the mask's own longest run -- was
    measured and REJECTED: when the rough window is too short nothing is masked, so
    ``mask_run_p99_s`` collapses toward zero and the test fires on the sound segments
    while missing 22 of 26 collapsed ones. That is why that metric carries no
    threshold. Nor does the residual 13 have a signature to key on: their mask
    fraction and rough-sigma ratio sit inside the healthy band while the SOUND
    segments at that length sit above `MASK_FRACTION_CEILING`.)
    """
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document

    if not APO_ABF.exists():
        pytest.skip(f"corpus ABF absent: {APO_ABF.name}")
    doc = open_document(APO_ABF)
    ch = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT][0]
    x = np.asarray(doc.channels[ch.index], np.float32)

    whole = B.estimate_baseline_result(x, FS, config=B.DEFAULT_BASELINE,
                                       polarity=-1, health=False).baseline
    starts, ends = _apo_events(x)
    block_cfg = B.BaselineConfig(method="block_median", window_s=B._LEGACY_WINDOW_S)

    def tally(L, cfg):
        wrong = silent = total = 0
        for s0, e0 in zip(starts, ends):
            if (e0 - s0) >= L:
                continue
            a = max(0, min(x.size - L, (s0 + e0) // 2 - L // 2))
            if not (a <= s0 and e0 <= a + L):
                continue
            level = float(np.median(x[s0:e0]))
            want = abs(level - float(np.median(whole[s0:e0])))
            if want < 20.0:
                continue
            total += 1
            r = B.estimate_baseline_result(x[a:a + L], FS, config=cfg, polarity=-1,
                                           health=True)
            got = abs(level - float(np.median(r.baseline[s0 - a:e0 - a])))
            if got / want < 0.5:
                wrong += 1
                silent += bool(r.health.ok)
        return wrong, silent, total

    for L, max_wrong, max_silent in ((1536, 230, 30), (2048, 60, 0), (3072, 0, 0)):
        cfg, _ = B.adapt_to_length(B.DEFAULT_BASELINE, L, FS)
        w, s, total = tally(L, cfg)
        bw, bs, btotal = tally(L, block_cfg)
        assert total > 200 and btotal == total, "fixture changed"
        assert w <= max_wrong and s <= max_silent, (
            f"L={L}: clamped is wrong on {w}/{total} and silent on {s} of those"
        )
        assert bw == total, (
            f"L={L}: the block median used to be wrong on every one of {total} "
            f"segments; it is now wrong on {bw}. Re-measure before believing it."
        )
        assert s < bs or bs == 0, (
            f"L={L}: clamped silent {s}, block median silent {bs} -- the whole point "
            "of the change is that it is silent less often"
        )
