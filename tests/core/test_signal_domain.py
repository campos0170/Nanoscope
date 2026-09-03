"""The SIGNAL-DOMAIN half of the pooling rule: is the number on this axis pA at all?

REGRESSION FOR THE SILENT NORMALIZED OVERLAY.

``Preprocess ▸ Normalize`` is stage 7 of ``preprocess_for_detection``, it runs BEFORE
either detector, and its own docstring states the consequence: *"NORMALIZE IS LAST,
AND IT TAKES THE PICOAMPS WITH IT … depth_pA, the baseline, the thresholds and every
exported column are still spelled 'pA' while carrying sigmas or span-fractions. The
detector cannot detect that, so the disclosure is the panel's job."*

The panel did its job — ``DetectionPanel._refresh_normalize_units_note`` keeps a
banner up for as long as the stage is enabled. But a banner does not TRAVEL. Detect
with Normalize on, detect a second recording without it, open ``Analysis ▸ Plots``,
tick both, plot ``depth_pA``: two curves on one axis, one in picoamps and one in
sigmas, and not one word anywhere. ``normalize_enabled`` was in provenance the whole
time (``stamp_preprocess_provenance``); nothing read it. ``_build_warnings`` had no
clause for it, and ``pool_comparability`` judged three properties none of which move
when the stage is toggled.

The SECOND way the same axis lies, closed by the same half: ``target_units``.
``ChannelCalibration.fingerprint()`` is ``(gain, offset)`` only, deliberately
("target_units is a label"), so a recording calibrated to nA and one calibrated to pA
carry the SAME fingerprint, agree on every half of the rule, and overlay a
thousand-fold apart.

What these pin:

* the domain is READ from provenance that already recorded it, by one reader;
* a mixed pool WARNS, naming both domains and the mode (and, for ``model_input``, the
  clip — the remedy sentence has to name a number);
* a uniform pool stays quiet, and the geometry columns stay quiet always;
* the normalize remedy does NOT offer ``frac_depth``. dI/I0 survives a change of gain
  or of unit (both pure scales) and does NOT survive Normalize, which is affine: it
  subtracts a location too, so the denominator |baseline_pA| lands at ~0. MEASURED —
  see ``test_frac_depth_is_destroyed_by_normalize_not_merely_rescaled``.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.filters import MODEL_INPUT_CLIP_SIGMA, MODEL_INPUT_MODE
from nanoscope.core.sources import (
    ABSOLUTE_CURRENT_COLUMNS,
    DEFAULT_SIGNAL_DOMAIN,
    GAIN_UNSTATED,
    NORMALIZED_DOMAIN_PREFIX,
    PooledItem,
    column_depends_on_gain,
    column_depends_on_signal_domain,
    is_normalized_domain,
    pool_comparability,
    signal_domain_of,
)
from nanoscope.core.sources import (
    _BASELINE_FREE_COLUMNS,
    _MODEL_INPUT_MODE,
    _PLOTS_ALTERNATIVES,
    _SIGNAL_DOMAIN_FREE_COLUMNS,
)

UNCAL = (1.0, 0.0)
NORM_ROBUST = {"normalize_enabled": True, "normalize_mode": "robust"}
NORM_MODEL = {"normalize_enabled": True, "normalize_mode": MODEL_INPUT_MODE,
              "normalize_clip_sigma": 30.0}


def _pool(*domains, quantity="depth_pA", **kw):
    """A pool of items that differ in NOTHING but their signal domain."""
    kw.setdefault("depends_on_baseline", False)
    kw.setdefault("depends_on_gain", False)
    kw.setdefault("alternatives", _PLOTS_ALTERNATIVES)
    return pool_comparability(
        [PooledItem(label=chr(ord("a") + i), signal_domain=d)
         for i, d in enumerate(domains)],
        quantity=quantity, **kw)


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------
def test_an_ordinary_detection_is_in_picoamps():
    assert signal_domain_of({}) == DEFAULT_SIGNAL_DOMAIN
    assert signal_domain_of(None) == DEFAULT_SIGNAL_DOMAIN
    assert signal_domain_of({"units": "pA"}) == "pA"
    # A falsy/blank units string is the app's "unknown", and the app spells unknown
    # units "pA" everywhere else (``ch.units or "pA"``). A second, distinct unknown
    # domain here would fire the warning on the manual-measurement table.
    assert signal_domain_of({"units": ""}) == DEFAULT_SIGNAL_DOMAIN
    assert signal_domain_of({"units": "  pA  "}) == "pA"


def test_normalize_off_is_not_a_normalized_domain_whatever_else_is_stamped():
    """The flag is what decides, not the presence of the mode key. A params dict that
    carries ``normalize_mode`` with the stage DISABLED describes a stage that did not
    run, and calling that trace dimensionless would block a perfectly poolable pA
    overlay."""
    prov = {"normalize_enabled": False, "normalize_mode": "robust",
            "normalize_clip_sigma": 30.0, "units": "pA"}
    assert signal_domain_of(prov) == "pA"
    assert not is_normalized_domain(signal_domain_of(prov))


@pytest.mark.parametrize("mode", ["zscore", "robust", "minmax"])
def test_each_affine_mode_is_its_own_domain(mode):
    d = signal_domain_of({"normalize_enabled": True, "normalize_mode": mode})
    assert is_normalized_domain(d)
    assert mode in d
    # The modes are NOT interchangeable: 'robust' divides by the baseline noise sigma
    # and 'minmax' by the trace's full span, which the one largest spike in the
    # recording sets. Pooling a run of each is as wrong as pooling either with pA, so
    # the labels must differ.
    others = {signal_domain_of({"normalize_enabled": True, "normalize_mode": m})
              for m in ("zscore", "robust", "minmax") if m != mode}
    assert d not in others


def test_the_model_input_domain_names_its_clip():
    """The clip is in the LABEL because the remedy sentence has to name a number: the
    transform is BOUNDED, so two runs clipped at different sigmas disagree about where
    depth stops being monotone in the real picoamps."""
    a = signal_domain_of(NORM_MODEL)
    b = signal_domain_of(dict(NORM_MODEL, normalize_clip_sigma=12.0))
    assert "30" in a and "12" in b
    assert a != b
    assert "CLIPPED" in a


def test_a_model_input_domain_with_no_clip_stamped_does_not_invent_one():
    """Only ``model_input`` stamps the clip, so a hand-built or pre-key provenance can
    reach here without one. Say the mode and the fact of the bound; a sigma this rule
    guessed would read in the warning as a number somebody measured."""
    d = signal_domain_of({"normalize_enabled": True, "normalize_mode": MODEL_INPUT_MODE})
    assert is_normalized_domain(d) and "CLIPPED" in d
    assert str(MODEL_INPUT_CLIP_SIGMA) not in d and "30" not in d


def test_the_model_input_mode_string_is_pinned_to_the_filters_constant():
    """``core`` must not import ``analysis`` (``analysis.voltage_dep`` imports
    ``core.sources``), so the one string crosses as a literal. This is the pin that
    stops the two drifting: rename the mode in ``filters`` and this fails rather than
    ``signal_domain_of`` quietly falling through to the affine branch and printing a
    clip-free label for a clipped trace."""
    assert _MODEL_INPUT_MODE == MODEL_INPUT_MODE


# ---------------------------------------------------------------------------
# the per-column half
# ---------------------------------------------------------------------------
def test_the_domain_free_columns_are_exactly_the_geometry_ones():
    """Pinned to ``_BASELINE_FREE_COLUMNS``: a rescale of the trace moves no sample
    index and no dwell, so the two lists agree today. Stated as its own name because
    they answer different questions, and pinned so a column added to one is not
    silently assumed into the other."""
    assert _SIGNAL_DOMAIN_FREE_COLUMNS == _BASELINE_FREE_COLUMNS
    for col in _SIGNAL_DOMAIN_FREE_COLUMNS:
        assert not column_depends_on_signal_domain(col), col


def test_the_domain_half_is_WIDER_than_the_gain_half_and_frac_depth_is_why():
    """The whole reason this is not just ``column_depends_on_gain``. A change of GAIN
    is a pure scale, which divides out of a ratio, so ``frac_depth`` survives it.
    Normalize is AFFINE — it subtracts a location too — so the dI/I0 DENOMINATOR goes
    to ~0 and the ratio does not survive."""
    assert not column_depends_on_gain("frac_depth")
    assert column_depends_on_signal_domain("frac_depth")
    for col in ABSOLUTE_CURRENT_COLUMNS:
        assert column_depends_on_signal_domain(col), col


def test_an_unknown_column_fails_closed():
    assert column_depends_on_signal_domain("some_future_amplitude_pA")


# ---------------------------------------------------------------------------
# the pooling half
# ---------------------------------------------------------------------------
def test_a_normalized_run_pooled_with_an_ordinary_one_is_no_longer_silent():
    v = _pool("pA", signal_domain_of(NORM_ROBUST))
    assert not v.comparable
    assert len(v.domain_warnings) == 1
    w = v.domain_warnings[0]
    assert w.startswith("NOT COMPARABLE")
    assert "depth_pA" in w
    # Both domains named, with the labels to blame — a warning that does not say
    # WHICH source is the normalized one cannot be acted on.
    assert "pA -- a" in w and "robust" in w and "-- b" in w


def test_a_uniformly_normalized_pool_says_nothing_here():
    """Groups DISAGREEING is the test. Two runs normalized identically are on one
    scale; that their columns are still mislabelled "pA" is a per-RUN disclosure, and
    ``modules._build_warnings`` carries it onto every one of them."""
    d = signal_domain_of(NORM_ROBUST)
    assert _pool(d, d).comparable


def test_two_normalize_modes_are_two_domains():
    v = _pool(signal_domain_of(NORM_ROBUST),
              signal_domain_of({"normalize_enabled": True, "normalize_mode": "minmax"}))
    assert not v.comparable and len(v.domain_warnings) == 1


def test_two_clips_of_the_model_input_mode_are_two_domains():
    v = _pool(signal_domain_of(NORM_MODEL),
              signal_domain_of(dict(NORM_MODEL, normalize_clip_sigma=12.0)))
    assert not v.comparable
    assert "30" in v.domain_warnings[0] and "12" in v.domain_warnings[0]


def test_a_pool_of_one_warns_about_nothing():
    assert _pool(signal_domain_of(NORM_ROBUST)).comparable


def test_a_geometry_column_stays_quiet_across_the_same_two_domains():
    """A guard that fires on everything trains the user to ignore it. Normalize
    rescales amplitudes; it moves no sample index and no dwell."""
    v = _pool("pA", signal_domain_of(NORM_ROBUST),
              quantity="dwell_s", depends_on_signal_domain=False)
    assert v.comparable


def test_the_normalize_remedy_refuses_to_offer_frac_depth():
    """The exact defect the gain guard had before ``GAIN_ALTERNATIVES`` existed:
    advice that walks the user off one cliff onto another. dI/I0 is destroyed by
    Normalize (see the measurement below), so the sentence names it only to rule it
    out, and offers the caller's baseline-free list instead."""
    w = _pool("pA", signal_domain_of(NORM_ROBUST)).domain_warnings[0]
    assert "frac_depth is NOT a way out" in w
    assert _PLOTS_ALTERNATIVES in w


def test_a_units_mismatch_is_the_same_half_with_a_different_remedy():
    """C3-3. ``target_units`` is deliberately outside ``ChannelCalibration.fingerprint()``
    ("target_units is a label"), so a pA recording and an nA one agree on the GAIN
    half and used to pool in silence. Same class of bug, same rule — but pA-to-nA is a
    pure scale, so here ``frac_depth`` genuinely IS the way out and the sentence says
    so."""
    v = _pool("pA", "nA")
    assert not v.comparable
    w = v.domain_warnings[0]
    assert "pA -- a" in w and "nA -- b" in w
    assert "Calibration" in w
    assert "frac_depth is NOT a way out" not in w
    assert "plot frac_depth or dwell_s" in w


def test_the_domain_half_is_gated_off_when_the_caller_says_the_column_is_immune():
    v = _pool("pA", "nA", depends_on_signal_domain=False)
    assert v.comparable and v.domain_warnings == ()


def test_the_groups_are_reported_even_when_the_half_is_gated_off():
    """``domain_groups`` is a FACT about the pool, like ``baseline_groups``; only the
    WARNING is conditional. A surface that wants to show a per-source column of
    domains must not have to re-derive them."""
    v = _pool("pA", "nA", depends_on_signal_domain=False)
    assert v.domain_groups == (("pA", ("a",)), ("nA", ("b",)))


def test_a_caller_that_never_heard_of_the_field_still_gets_a_verdict():
    """``PooledItem.signal_domain`` defaults to pA on purpose: a surface not yet
    widened says "picoamps", which is what every caller meant before the field
    existed, rather than a fourth unknown that would warn about nothing (or, worse,
    about everything)."""
    v = pool_comparability([PooledItem("a"), PooledItem("b")],
                           quantity="depth_pA", depends_on_baseline=False,
                           depends_on_gain=False)
    assert v.comparable and v.domain_groups == ((DEFAULT_SIGNAL_DOMAIN, ("a", "b")),)


def test_the_domain_warning_lands_before_the_gain_one():
    """Most fundamental first. The baseline half decides what the number is measured
    AGAINST, the domain half what UNIT it is in, the gain half only WHICH pA — a
    question that is moot once the domain answer is "sigmas". A reader who fixes the
    gain first has fixed nothing."""
    v = pool_comparability(
        [PooledItem("a", calibration_fingerprint=UNCAL),
         PooledItem("b", calibration_fingerprint=UNCAL,
                    signal_domain=signal_domain_of(NORM_ROBUST))],
        quantity="depth_pA", depends_on_baseline=False, depends_on_gain=True,
        alternatives=_PLOTS_ALTERNATIVES)
    assert len(v.warnings) == 2
    assert v.warnings[0] is v.domain_warnings[0]
    assert v.warnings[1] is v.gain_warnings[0]


def test_the_manual_table_is_evidence_against_a_normalized_neighbour():
    """Unlike the GAIN half, there is no "unstated" exemption here. A hand-measured
    depth really is in picoamps — that is a claim, not an abstention — so it belongs
    in the pA group and must not let a normalized overlay through."""
    v = pool_comparability(
        [PooledItem("Manual measurements", calibration_fingerprint=GAIN_UNSTATED),
         PooledItem("b", calibration_fingerprint=UNCAL,
                    signal_domain=signal_domain_of(NORM_ROBUST))],
        quantity="depth_pA", depends_on_baseline=False, depends_on_gain=True,
        alternatives=_PLOTS_ALTERNATIVES)
    assert v.domain_warnings and "Manual measurements" in v.domain_warnings[0]


# ---------------------------------------------------------------------------
# the measurement the advice rests on
# ---------------------------------------------------------------------------
def test_frac_depth_is_destroyed_by_normalize_not_merely_rescaled():
    """THE NUMBER BEHIND THE REMEDY SENTENCE, re-measured on every run.

    The reason ``GAIN_ALTERNATIVES`` cannot be reused for this half. A synthetic
    100 pA blockade in a 1000 pA pore, 10 pA noise, 200 events, pushed through the
    REAL ``preprocess_for_detection`` + ``detect_events`` — the same code the data
    goes through — with each normalize mode in turn:

        raw          baseline_pA  999.97   depth_pA 100.1   frac_depth  0.10012
        zscore                       0.069            7.10              87.3   (872x)
        robust                       0.009            9.90             105.0  (1048x)
        minmax                       0.222            0.50               2.27  (22.6x)
        model_input                 -0.000            0.34             104.5  (1044x)

    ``snr`` came through all four at 9.8 — depth and noise_rms scale together and
    neither carries the offset. ``frac_depth`` does not, because Normalize is AFFINE
    and the location it subtracts is the open-pore level, i.e. the DENOMINATOR.
    """
    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.modules import preprocess_for_detection

    fs = 100_000.0
    n = int(fs * 20)
    rng = np.random.default_rng(7)
    sig = np.full(n, 1000.0) + rng.normal(0, 10.0, n)
    for i in range(200):
        s = 2000 + i * (n - 4000) // 200
        sig[s:s + 100] -= 100.0
    raw = sig.astype(np.float32)

    df_raw, _ = detect_events(raw, fs, units="pA", recording="r")
    frac_raw = float(np.median(df_raw["frac_depth"]))
    snr_raw = float(np.median(df_raw["snr"]))
    assert frac_raw == pytest.approx(0.100, abs=0.005)

    for mode in ("zscore", "robust", "minmax", MODEL_INPUT_MODE):
        params = {"normalize_enabled": True, "normalize_mode": mode,
                  "normalize_clip_sigma": 30.0}
        df, _ = detect_events(preprocess_for_detection(raw, fs, params), fs,
                              units="pA", recording="r")
        assert len(df) == len(df_raw), mode
        # The ratio the gain half calls safe is off by more than an order of
        # magnitude in every mode -- 22x in the mildest of them.
        assert abs(float(np.median(df["frac_depth"])) / frac_raw) > 20, mode
        # ...while snr, which divides by a quantity that scales the SAME way and
        # carries no offset, is untouched. (It is still on the closed side of
        # `_SIGNAL_DOMAIN_FREE_COLUMNS`: `model_input` CLIPS, and a clipped depth is
        # not monotone in the real one, so this list must not promise it there.)
        assert float(np.median(df["snr"])) == pytest.approx(snr_raw, rel=0.02), mode


def test_the_prefix_is_what_makes_a_domain_normalized():
    assert is_normalized_domain(f"{NORMALIZED_DOMAIN_PREFIX}: anything")
    assert not is_normalized_domain("pA")
    assert not is_normalized_domain("nA")
