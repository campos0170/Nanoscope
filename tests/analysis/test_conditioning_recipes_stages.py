"""Per-stage Revert and Reset, and the thresholds the chain judge weighs (editing
foundation: design 1.2 defect 7, section 6, 7.2; decision 8). Pure: no Qt in this file.

Revert used to put back all 37 chain keys and fill any the run never recorded from the
defaults, so the user's own numbers on stages nothing had changed were lost to it; and a
k_in edited after a run left the table on screen looking current.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import conditioning_recipes as cr

FS = 50_000.0

#: Both detectors' thresholds, as ``DetectionPanel.params`` always carries them.
_THRESHOLDS = {"k_in": 5.0, "k_out": 2.0, "cusum_k": 0.75, "cusum_h": 28.0,
               "min_dwell_s": 2e-5, "max_dwell_s": None, "merge_gap_s": None,
               "polarity": None}

#: Every stage running, so each key of each stage is live.
_ALL_ON = {"filter_cutoff_hz": 10_000.0, "notch_enabled": True, "median_enabled": True,
           "hp_enabled": True, "detrend_enabled": True, "rmean_enabled": True,
           "savgol_enabled": True, "normalize_enabled": True,
           "normalize_mode": "model_input"}


def _form(**over):
    return dict(cr.CHAIN_DEFAULTS, **over)


def _run(**over):
    """A run as a live detection records it: the whole form, chain and thresholds."""
    return {**cr.CHAIN_DEFAULTS, **_THRESHOLDS, **over}


def _nudged(key, value):
    """Another valid setting for ``key``."""
    if isinstance(value, bool):
        return not value
    if key == "notch_freqs":
        return [50.0, 150.0] if value is None else None
    if isinstance(value, str):
        return {"filter_kind": "butter", "hp_kind": "bessel", "detrend_method": "percentile",
                "normalize_mode": "zscore"}[key]
    return value + 1 if isinstance(value, int) else value * 1.5


# -- the stages' keys ---------------------------------------------------------------------
def test_stage_param_keys_hold_every_chain_key_exactly_once():
    assert tuple(cr.STAGE_PARAM_KEYS) == cr.STAGE_KEYS
    owned = [k for keys in cr.STAGE_PARAM_KEYS.values() for k in keys]
    assert sorted(owned) == sorted(cr.CHAIN_KEYS) and len(set(owned)) == len(owned) == 37
    for stage, keys in cr.STAGE_PARAM_KEYS.items():
        assert (cr.STAGE_ENABLE_KEY[stage] or "filter_cutoff_hz") in keys, stage
        assert list(keys) == [k for k in cr.CHAIN_KEYS if k in keys], "CHAIN_KEYS order"
    assert cr.STAGE_PARAM_KEYS["lowpass"] == ("filter_cutoff_hz", "filter_order", "filter_kind")
    assert cr.STAGE_PARAM_KEYS["highpass"] == ("hp_enabled", "hp_cutoff_hz", "hp_order",
                                               "hp_kind")


@pytest.mark.parametrize("stage", cr.STAGE_KEYS)
def test_a_change_to_a_stages_key_is_judged_on_that_stage_and_its_revert_clears_it(stage):
    """The split is the judge's own: whatever the judge names on a stage, that stage's
    Revert puts back, and it never names a key on another stage."""
    run = _form(**_ALL_ON)
    switch = cr.STAGE_ENABLE_KEY[stage] or "filter_cutoff_hz"
    assert cr.stages_that_differ(dict(run, **{switch: _nudged(switch, run[switch])}),
                                 run) == {stage}
    for key in cr.STAGE_PARAM_KEYS[stage]:
        form = dict(run, **{key: _nudged(key, run[key])})
        assert cr.stages_that_differ(form, run) <= {stage}, key
        back = cr.revert_params(form, run, stages={stage})
        assert set(back) == set(cr.STAGE_PARAM_KEYS[stage]), key
        assert cr.stages_that_differ(dict(form, **back), run) == frozenset(), key


# -- Revert -------------------------------------------------------------------------------
def test_reverting_one_stage_puts_back_that_stage_and_leaves_the_others():
    run = _run(filter_cutoff_hz=10_000.0, notch_enabled=True, median_enabled=True)
    form = dict(run, filter_cutoff_hz=5_000.0, notch_q=20.0, median_window_s=3e-4)
    assert cr.stages_that_differ(form, run) == {"notch", "median", "lowpass"}
    back = cr.revert_params(form, run, stages={"notch"})
    assert back == {k: run[k] for k in cr.STAGE_PARAM_KEYS["notch"]}
    after = dict(form, **back)
    assert cr.stages_that_differ(after, run) == {"median", "lowpass"}
    assert (after["median_window_s"], after["filter_cutoff_hz"]) == (3e-4, 5_000.0)
    # A stage's Revert is that stage's: a changed threshold is not part of it.
    edited = dict(form, k_in=6.0)
    assert "k_in" not in cr.revert_params(edited, run, stages={"notch"},
                                          module_id="detect.schmitt")


def test_revert_never_fills_a_key_the_run_did_not_record():
    """Defect 7. A legacy provenance records the values of the stages that ran and only
    the off flag of the others; Revert filled every other key from the defaults, so the
    user's numbers on stages it had no reason to touch were lost."""
    run = {"filter_cutoff_hz": 10_000.0, "filter_order": 8, "filter_kind": "bessel",
           "notch_enabled": True, "notch_freqs": [60.0, 120.0], "notch_q": 30.0,
           "median_enabled": False, "polarity": None}
    form = _form(filter_cutoff_hz=10_000.0, notch_enabled=True, notch_freqs=[60.0, 120.0],
                 notch_q=20.0, notch_freq_hz=50.0, notch_harmonics=5, median_enabled=True,
                 median_window_s=3e-4, hp_cutoff_hz=250.0, polarity=None)
    status = cr.judge_chain(form, run, module_id="detect.schmitt", detector_label="Schmitt")
    assert status.kind == "differs"
    assert set(status.revert_params) <= set(run), "only keys the run recorded come back"
    assert status.revert_params == {"notch_enabled": True, "notch_freqs": [60.0, 120.0],
                                    "notch_q": 30.0, "median_enabled": False}
    after = dict(form, **status.revert_params)
    assert cr.judge_chain(after, run, module_id="detect.schmitt").kind == "match"
    assert (after["median_window_s"], after["notch_freq_hz"], after["notch_harmonics"],
            after["hp_cutoff_hz"]) == (3e-4, 50.0, 5, 250.0)


def test_a_stage_the_run_never_recorded_reverts_to_off_and_keeps_its_settings():
    """A build older than a stage recorded nothing for it, so it did not run it. The judge
    reads that stage off; Revert must agree, or the notice names a difference no Revert
    clears. Only the on/off comes back: the stage's settings stay the user's."""
    legacy = {k: v for k, v in _run(filter_cutoff_hz=10_000.0).items()
              if not k.startswith("savgol_")}
    form = dict(legacy, filter_cutoff_hz=5_000.0, savgol_enabled=True, savgol_window_s=2e-3,
                savgol_polyorder=4)
    status = cr.judge_chain(form, legacy, module_id="detect.schmitt", detector_label="Schmitt")
    assert [d.text for d in status.diffs] == ["Savitzky-Golay off → on (2 ms · order 4)",
                                              "low-pass 10000 → 5000 Hz"]
    assert status.revert_params == {"savgol_enabled": False, "filter_cutoff_hz": 10_000.0,
                                    "filter_order": 8, "filter_kind": "bessel"}
    after = dict(form, **status.revert_params)
    assert cr.judge_chain(after, legacy, module_id="detect.schmitt").kind == "match"
    assert (after["savgol_window_s"], after["savgol_polyorder"]) == (2e-3, 4)
    # The low-pass's on/off is its cutoff: a run that recorded none ran without it.
    no_lowpass = {k: v for k, v in legacy.items() if not k.startswith("filter_")}
    form = dict(legacy, filter_order=4)
    assert cr.revert_params(form, no_lowpass, stages="lowpass") == {"filter_cutoff_hz": None}


def test_revert_is_withheld_where_the_run_did_not_record_a_setting_that_differs():
    """A build older than a knob recorded nothing for it, and the judge reads it at its
    default. Revert will not invent that value (defect 7), so it is not offered: offered,
    it reported the form reverted while the notice still named the difference."""
    run = _run(filter_cutoff_hz=10_000.0, detrend_enabled=True, detrend_method="robust_median")
    del run["detrend_decim"]
    form = dict(run, detrend_decim=10)
    status = cr.judge_chain(form, run, module_id="detect.schmitt", detector_label="Schmitt")
    assert [d.text for d in status.diffs] == ["baseline drift stride auto → 10"]
    assert status.kind == "differs" and status.revert_params is None
    assert status.sentence == (
        "⚠ The form differs from the chain the Schmitt detection on screen ran with: "
        "baseline drift stride auto → 10. Run detection to apply it — Revert cannot restore "
        "a setting the run did not record.")
    assert cr.revert_params(form, run, stages="detrend") is None
    # All or nothing: the notice's Revert does not half-revert a low-pass edit beside it,
    # while the low-pass stage's own Revert still puts that stage back.
    both = dict(form, filter_cutoff_hz=5_000.0)
    assert cr.judge_chain(both, run, module_id="detect.schmitt").revert_params is None
    back = cr.revert_params(both, run, stages="lowpass")
    assert back == cr.revert_params(both, run, stages={"lowpass"})
    assert cr.stages_that_differ(dict(both, **back), run) == {"detrend"}
    k_in = cr.judge_chain(dict(form, k_in=6.0), run, module_id="detect.schmitt",
                          detector_label="Schmitt")
    assert k_in.revert_params is None
    assert k_in.sentence.endswith("Run detection to apply them — Revert cannot restore a "
                                  "setting the run did not record.")


def test_a_recorded_none_comes_back_only_where_it_is_a_setting():
    # An off low-pass: its None cutoff IS the setting. The stamp's None order and kind
    # mean "not applicable" (the form's int() would raise on them), so they stay out.
    run = {"filter_cutoff_hz": None, "filter_order": None, "filter_kind": None}
    form = _form(filter_cutoff_hz=8_000.0, filter_order=4, filter_kind="butter")
    assert cr.revert_params(form, run) == {"filter_cutoff_hz": None}
    # A provenance stamps the auto detrend stride as the kernel's None: it is the form's 0.
    run = {"detrend_enabled": True, "detrend_method": "robust_median",
           "detrend_window_s": 0.5, "detrend_decim": None}
    form = _form(detrend_enabled=True, detrend_method="robust_median", detrend_window_s=0.5,
                 detrend_decim=10)
    assert [d.text for d in cr.chain_differences(form, run)] == [
        "baseline drift stride auto → 10"]
    back = cr.revert_params(form, run)
    assert back["detrend_decim"] == 0 and cr.chains_equal(dict(form, **back), run)


def test_reverting_the_drift_stage_brings_back_the_polarity_it_ran_with():
    """The drift stage reads the polarity. A Revert that turns the stage back on without
    the run's polarity leaves a polarity difference for a second Revert."""
    run = _run(detrend_enabled=True, polarity=-1)
    form = dict(run, detrend_enabled=False, polarity=1)
    assert [d.text for d in cr.chain_differences(form, run)] == ["baseline drift on → off"]
    back = cr.revert_params(form, run)
    assert back["detrend_enabled"] is True and back["polarity"] == -1
    assert cr.chains_equal(dict(form, **back), run)
    # A drift stage that did not run read no polarity, so none comes back with it.
    ran_off = dict(run, detrend_enabled=False)
    back = cr.revert_params(dict(ran_off, detrend_enabled=True, polarity=1), ran_off)
    assert back["detrend_enabled"] is False and "polarity" not in back


def test_the_judges_revert_holds_only_what_differs():
    run = _run(filter_cutoff_hz=10_000.0, notch_enabled=True, detrend_enabled=True,
               polarity=-1)
    form = dict(run, notch_q=20.0, hp_cutoff_hz=250.0, k_out=1.5)
    status = cr.judge_chain(form, run, module_id="detect.schmitt")
    assert status.revert_params == {
        **{k: run[k] for k in cr.STAGE_PARAM_KEYS["notch"]}, "k_out": 2.0}
    after = dict(form, **status.revert_params)
    assert cr.judge_chain(after, run, module_id="detect.schmitt").kind == "match"
    assert after["hp_cutoff_hz"] == 250.0, "an off stage's setting is the user's to keep"
    # A CUSUM table does not weigh k_out, so its Revert leaves it alone.
    assert "k_out" not in cr.judge_chain(form, run, module_id="detect.cusum").revert_params
    assert cr.revert_params(run, run, module_id="detect.schmitt") == {}


# -- Reset --------------------------------------------------------------------------------
@pytest.mark.parametrize("stage", cr.STAGE_KEYS)
def test_reset_restores_the_stage_defaults_and_leaves_it_on_or_off(stage):
    switch = cr.STAGE_ENABLE_KEY[stage] or "filter_cutoff_hz"
    reset = cr.reset_stage_params(stage)
    assert set(reset) == set(cr.STAGE_PARAM_KEYS[stage]) - {switch}
    assert reset == {k: cr.CHAIN_DEFAULTS[k] for k in reset}
    run = _form(**_ALL_ON)
    edited = dict(run, **{k: _nudged(k, run[k]) for k in reset})
    after = cr.effective_chain(dict(edited, **reset))
    if stage == "lowpass":
        assert reset == {"filter_order": 8, "filter_kind": "bessel"}
        assert after["lowpass"] == {"cutoff_hz": 10_000.0, "kind": "bessel", "order": 8}
    else:
        fresh = cr.effective_chain(dict(cr.CHAIN_DEFAULTS, **{switch: True}))
        assert after[stage] == fresh[stage], "a running stage keeps running, at defaults"
    off = dict(edited, **{switch: cr.CHAIN_DEFAULTS[switch]}, **reset)
    assert stage not in cr.effective_chain(off), "an off stage stays off"


# -- thresholds ---------------------------------------------------------------------------
def test_thresholds_are_judged_for_schmitt_and_cusum_in_the_chains_words():
    run = _run(filter_cutoff_hz=10_000.0)
    form = dict(run, k_in=6.0, cusum_h=30.0, min_dwell_s=5e-5, max_dwell_s=2.0,
                merge_gap_s=1e-3)
    schmitt = cr.threshold_differences(form, run, "detect.schmitt")
    assert [d.text for d in schmitt] == ["k_in 5 → 6", "min dwell 20 µs → 50 µs",
                                         "max dwell off → 2 s", "merge gap off → 1 ms"]
    assert [(d.stage, d.key) for d in schmitt] == [
        ("thresholds", "k_in"), ("thresholds", "min_dwell_s"), ("thresholds", "max_dwell_s"),
        ("thresholds", "merge_gap_s")]
    cusum = cr.threshold_differences(form, run, "detect.cusum")
    assert [d.text for d in cusum] == ["CUSUM h 28 → 30", "min dwell 20 µs → 50 µs",
                                       "max dwell off → 2 s", "merge gap off → 1 ms"]
    for other in ("detect.keras", "detect.reconciled", None):
        assert cr.threshold_differences(form, run, other) == ()
    status = cr.judge_chain(form, run, module_id="detect.schmitt", detector_label="Schmitt")
    assert status.kind == "differs" and status.diffs == schmitt     # the chain matches
    assert status.revert_params == {"k_in": 5.0, "min_dwell_s": 2e-5, "max_dwell_s": None,
                                    "merge_gap_s": None}
    after = dict(form, **status.revert_params)
    assert cr.judge_chain(after, run, module_id="detect.schmitt").kind == "match"


def test_a_threshold_the_run_did_not_record_is_not_judged():
    chain = _form(filter_cutoff_hz=10_000.0)
    form = {**chain, **_THRESHOLDS, "k_in": 9.0, "merge_gap_s": 1e-3}
    assert cr.threshold_differences(form, chain, "detect.schmitt") == ()
    assert cr.judge_chain(form, chain, module_id="detect.schmitt").kind == "match"
    # Only what it recorded is weighed; a None k_in is no setting, a None merge gap is off.
    partial = dict(chain, k_in=None, merge_gap_s=None)
    assert [d.key for d in cr.threshold_differences(form, partial, "detect.schmitt")] == [
        "merge_gap_s"]


def test_zero_and_none_are_both_off_and_floats_agree_to_one_part_in_a_billion():
    run = _run(max_dwell_s=0.0, merge_gap_s=0.0)
    assert cr.threshold_differences(_run(k_in=5.0 * (1 + 1e-12)), run, "detect.schmitt") == ()
    assert cr.threshold_differences(_run(k_in=5.0 * (1 + 1e-6)), run, "detect.schmitt")


@pytest.mark.parametrize("module_id", ["detect.schmitt", "detect.cusum"])
def test_a_detection_run_judges_as_matching_its_own_thresholds(context_factory, module_id):
    """Read back from a real run, by either route, the thresholds equal the form that ran
    them: otherwise every fresh table would open on a difference no Revert clears."""
    from nanoscope.analysis.modules import (
        DetectCusum,
        DetectSchmitt,
        detection_params_from_provenance,
    )

    rng = np.random.default_rng(5)
    n = 100_000
    raw = (-1000.0 + rng.normal(0, 8.0, n)).astype(np.float32)
    for s in range(10_000, n, 20_000):
        raw[s:s + 300] += 250.0
    form = {**_form(filter_cutoff_hz=10_000.0, detrend_enabled=True, detrend_window_s=0.2),
            **_THRESHOLDS, "max_dwell_s": 0.5, "merge_gap_s": 1e-3, "detector": module_id}
    module = DetectSchmitt() if module_id == "detect.schmitt" else DetectCusum()
    res = module.run(context_factory(raw, fs=FS, params=dict(form)))
    runs = (res.params, detection_params_from_provenance(res.payload.provenance))
    for run in runs:
        assert cr.judge_chain(form, run, module_id=module_id, fs=FS).kind == "match"
    for run in runs:
        status = cr.judge_chain(dict(form, min_dwell_s=5e-5), run, module_id=module_id, fs=FS)
        assert [d.text for d in status.diffs] == ["min dwell 20 µs → 50 µs"]
        assert status.revert_params == {"min_dwell_s": 2e-5}


# -- the wording --------------------------------------------------------------------------
def test_a_chain_difference_keeps_its_words_and_a_threshold_says_settings_changed():
    run = _run(filter_cutoff_hz=10_000.0)
    chain = cr.judge_chain(dict(run, filter_cutoff_hz=5_000.0), run,
                           module_id="detect.schmitt", detector_label="Schmitt")
    assert not chain.thresholds_differ
    assert chain.headline == "⚠ Differs from the Schmitt run: low-pass 10000 → 5000 Hz"
    assert chain.sentence == ("⚠ The form differs from the chain the Schmitt detection on "
                              "screen ran with: low-pass 10000 → 5000 Hz. Revert, or run "
                              "detection to apply it.")
    k_in = cr.judge_chain(dict(run, k_in=6.0), run, module_id="detect.schmitt",
                          detector_label="Schmitt")
    assert k_in.thresholds_differ
    assert k_in.headline == "⚠ Settings changed: k_in 5 → 6"
    assert k_in.sentence == ("⚠ Settings changed since the Schmitt detection on screen ran: "
                             "k_in 5 → 6. Revert, or run detection to apply them.")
    # More chain differences than the line names: the thresholds still lead, because no
    # stage list shows them.
    both = cr.judge_chain(dict(run, filter_cutoff_hz=5_000.0, notch_enabled=True,
                               median_enabled=True, savgol_enabled=True, k_in=6.0), run,
                          module_id="detect.schmitt", detector_label="Schmitt")
    assert [d.stage for d in both.diffs] == ["notch", "median", "savgol", "lowpass",
                                             "thresholds"]
    assert both.headline.startswith("⚠ Settings changed: k_in 5 → 6; notch off → on")
    assert both.headline.endswith("(+2 more)")
    assert both.sentence.startswith("⚠ Settings changed since the Schmitt detection on "
                                    "screen ran: k_in 5 → 6; ")
