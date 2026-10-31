"""``analysis.lineage``: one record of what a table was made with, one judge of each input
(docs/EDITING_FOUNDATION_DESIGN.md 7.1).

THE GAP. A stored table records five inputs: the sample rate (``provenance['fs']``), the
calibration and open-pore estimator stamps, and the chain and thresholds it ran with. Each surface
asked about them on its own. The amber line names only the FIRST stamp that moved, so a table stale
on two counts said one. The chain notice compares the chain alone, and nothing compared the
thresholds: a k_in edited after a run left the table on screen looking current (design 1.4).

THE MODULE. ``made_with(result)`` reads those inputs off a result without adding a stamp: params
first, the provenance stamp as the fallback, as the form's own reader does.
``judge_inputs(made, now)`` gives each input its own verdict — same, changed or unknown — in a fixed
order, with its own action (re-run, re-measure or revert) and a sentence. The rules are the ones
already in force: the rate is ``events.stale_rate_hz``'s, and the chain and the thresholds are the
chain notice's (``chain_differences``, ``threshold_differences``: per detector, None as off),
worded by ``describe_differences``.

Pure: these tests build results by hand and never open a window.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import numpy as np

from nanoscope.analysis import lineage
from nanoscope.analysis.conditioning_recipes import (
    CHAIN_DEFAULTS,
    THRESHOLD_KEYS_BY_MODULE,
    chain_differences,
    describe_differences,
    threshold_differences,
)
from nanoscope.analysis.events import EventSet, empty_df, stale_rate_hz
from nanoscope.analysis.lineage import (
    INPUT_NAMES,
    InputVerdict,
    LiveInputs,
    MadeWith,
    judge_inputs,
    made_with,
)
from nanoscope.analysis.modules import fit_dwell
from nanoscope.core.analysis import AnalysisResult

FS = 50_000.0
BASELINE = "sha256:31991df77507"
OTHER_BASELINE = "sha256:0d9a4e1c7b22"


# ================================================================================================
# builders
# ================================================================================================

def _form(**over: Any) -> dict[str, Any]:
    """The Detection form's ``params()`` as a run hands it to its module: the whole chain, the
    polarity and BOTH detectors' thresholds (the tab carries both sets)."""
    params: dict[str, Any] = dict(CHAIN_DEFAULTS)
    params.update(filter_cutoff_hz=10_000.0, polarity=-1, k_in=5.0, k_out=3.0, cusum_k=0.5,
                  cusum_h=8.0, min_dwell_s=2e-5, max_dwell_s=None, merge_gap_s=None,
                  detector="schmitt")
    params.update(over)
    return params


def _table(module_id: str = "detect.schmitt", *, params: dict[str, Any] | None = None,
           provenance: dict[str, Any] | None = None,
           calibration: tuple[float, float] = (1.0, 0.0),
           baseline: str | None = BASELINE, df=None) -> AnalysisResult:
    """A stored detection that ran with ``params`` (the form by default)."""
    return AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id="doc", channel=0,
        params=_form() if params is None else dict(params),
        calibration_fingerprint=calibration,
        payload=EventSet(empty_df() if df is None else df,
                         {"fs": FS} if provenance is None else dict(provenance)),
        baseline_fingerprint=baseline)


def _live(**over: Any) -> LiveInputs:
    """Everything as the table above ran with it, unless ``over`` says otherwise."""
    fields: dict[str, Any] = dict(rate_hz=FS, calibration=(1.0, 0.0),
                                  baseline_fingerprint=BASELINE, chain=_form(),
                                  thresholds=_form())
    fields.update(over)
    return LiveInputs(**fields)


def _judge(table: AnalysisResult, live: LiveInputs) -> dict[str, InputVerdict]:
    return {v.name: v for v in judge_inputs(made_with(table), live)}


# ================================================================================================
# the list
# ================================================================================================

def test_every_input_is_judged_once_in_the_amber_lines_order():
    verdicts = judge_inputs(made_with(_table()), _live())

    assert INPUT_NAMES == ("rate", "calibration", "baseline", "chain", "thresholds")
    assert tuple(v.name for v in verdicts) == INPUT_NAMES


def test_a_table_judged_against_what_it_ran_with_is_the_same_on_every_input():
    verdicts = judge_inputs(made_with(_table()), _live())

    assert [v.state for v in verdicts] == ["same"] * 5
    assert [v.action for v in verdicts] == [""] * 5


def test_a_table_stale_on_two_counts_says_both():
    """The amber line names the first stamp that moved; the list names every one."""
    verdicts = _judge(_table(), _live(rate_hz=25_000.0, calibration=(2.0, 0.0)))

    assert verdicts["rate"].state == verdicts["calibration"].state == "changed"
    assert verdicts["baseline"].state == "same"


# ================================================================================================
# each input: same, changed, unknown
# ================================================================================================

def test_the_sample_rate_follows_the_amber_lines_rule_and_says_both_clocks():
    changed = _judge(_table(), _live(rate_hz=25_000.0))["rate"]
    assert changed.state == "changed" and changed.action == "re-run"
    assert (changed.was, changed.now) == ("50 kHz", "25 kHz")
    assert changed.sentence == ("Sample rate: 50 kHz when detected, 25 kHz now. "
                                "Re-run to measure on the new clock.")

    # ONE rule: whatever the live rate, "changed" is exactly what `stale_rate_hz` calls stale --
    # its tolerance, and a recording that has lost its rate.
    for live in (FS, FS * (1 + 1e-12), FS * (1 + 1e-6), 25_000.0, None):
        verdict = _judge(_table(), _live(rate_hz=live))["rate"]
        assert (verdict.state == "changed") == (stale_rate_hz({"fs": FS}, live) is not None), live

    # A recording that has lost its rate has no new clock to measure on: the way forward is the
    # amber line's, a re-run under the rate the user declares.
    lost = _judge(_table(), _live(rate_hz=None))["rate"]
    assert (lost.was, lost.now, lost.action) == ("50 kHz", "unknown", "re-run")
    assert lost.sentence == ("Sample rate: 50 kHz when detected; the recording's rate is not "
                             "known now. Declare it, then re-run.")

    unstamped = _judge(_table(provenance={}), _live())["rate"]
    assert unstamped.state == "unknown" and unstamped.action == ""
    assert unstamped.was == "not recorded"


def test_the_calibration_is_its_stamp_compared_exactly():
    assert _judge(_table(), _live())["calibration"].state == "same"

    changed = _judge(_table(), _live(calibration=(2.0, 0.0)))["calibration"]
    assert changed.state == "changed" and changed.action == "re-run"
    assert (changed.was, changed.now) == ("gain 1, offset 0", "gain 2, offset 0")
    assert changed.sentence == ("Calibration: gain 1, offset 0 when detected; gain 2, offset 0 "
                                "now. Re-run to measure under the new calibration.")

    # A change below six digits is still a change, and still printed as two numbers.
    tiny = _judge(_table(), _live(calibration=(1.0 + 1e-9, 0.0)))["calibration"]
    assert tiny.state == "changed" and tiny.was != tiny.now

    unknown = judge_inputs(MadeWith("detect.schmitt", rate_hz=FS), _live())[1]
    assert (unknown.name, unknown.state, unknown.action) == ("calibration", "unknown", "")


def test_the_open_pore_estimator_is_re_measured_not_re_run():
    assert _judge(_table(), _live())["baseline"].state == "same"

    changed = _judge(_table(), _live(baseline_fingerprint=OTHER_BASELINE))["baseline"]
    assert changed.state == "changed" and changed.action == "re-measure"
    assert (changed.was, changed.now) == (BASELINE, OTHER_BASELINE)

    # The amber line's migration case: no stamp is a table that predates it. Unknown -- but the
    # repair the line offers for it is the same one.
    legacy = _judge(_table(baseline=None), _live(baseline_fingerprint=OTHER_BASELINE))["baseline"]
    assert legacy.state == "unknown" and legacy.action == "re-measure"
    assert "predates" in legacy.sentence

    # When the estimator a fresh run would use here is not known, nothing is said or offered.
    blind = _judge(_table(), _live(baseline_fingerprint=None))["baseline"]
    assert blind.state == "unknown" and blind.action == ""


def test_the_chain_is_judged_and_worded_by_the_chain_judge():
    run, form = _form(), _form(filter_cutoff_hz=5_000.0)
    chain = _judge(_table(params=run), _live(chain=form))["chain"]

    assert chain.state == "changed" and chain.action == "revert"
    assert chain.sentence == ("Signal chain: low-pass 10000 → 5000 Hz. "
                              "Revert, or re-run to apply it.")
    assert describe_differences(chain_differences(form, run, fs=FS)) in chain.sentence
    assert (chain.was, chain.now) == ("Low-pass Bessel 8 · 10 kHz", "Low-pass Bessel 8 · 5 kHz")

    # A stage turned on joins the same sentence, in the chain judge's own words.
    more = _judge(_table(params=run),
                  _live(chain=_form(filter_cutoff_hz=5_000.0, notch_enabled=True)))["chain"]
    assert "notch off → on" in more.sentence and "low-pass 10000 → 5000 Hz" in more.sentence

    # A threshold is not a chain setting.
    assert _judge(_table(params=run), _live(chain=_form(k_in=6.0)))["chain"].state == "same"


def test_the_detrend_polarity_is_part_of_the_chain_as_the_chain_judge_has_it():
    """``polarity`` is not a chain key, but the detrend stage reads it and the chain judge
    compares it there; leaving it out of the record would invent or hide a difference."""
    run = _form(detrend_enabled=True, polarity=-1)
    flipped = _judge(_table(params=run), _live(chain=_form(detrend_enabled=True, polarity=1)))
    assert flipped["chain"].state == "changed" and "polarity" in flipped["chain"].sentence

    # Without a detrend the polarity conditions nothing, so it is not a chain difference.
    plain = _judge(_table(params=_form(polarity=-1)), _live(chain=_form(polarity=1)))
    assert plain["chain"].state == "same"


def test_each_detector_is_judged_on_its_own_thresholds():
    schmitt, cusum = made_with(_table("detect.schmitt")), made_with(_table("detect.cusum"))
    assert tuple(schmitt.thresholds) == THRESHOLD_KEYS_BY_MODULE["detect.schmitt"]
    assert tuple(cusum.thresholds) == THRESHOLD_KEYS_BY_MODULE["detect.cusum"]

    k_in = _form(k_in=6.0)
    moved = judge_inputs(schmitt, _live(thresholds=k_in))[4]
    assert (moved.name, moved.state, moved.action) == ("thresholds", "changed", "revert")
    assert moved.sentence == "Thresholds: k_in 5 → 6. Revert, or re-run to apply them."
    assert (moved.was, moved.now) == ("k_in 5", "k_in 6")
    # The chain notice's own judge and words, so the notice and this list cannot disagree.
    assert describe_differences(
        threshold_differences(k_in, schmitt.thresholds, "detect.schmitt")) in moved.sentence
    assert judge_inputs(cusum, _live(thresholds=k_in))[4].state == "same"

    cusum_h = _form(cusum_h=9.0)
    assert judge_inputs(schmitt, _live(thresholds=cusum_h))[4].state == "same"
    moved = judge_inputs(cusum, _live(thresholds=cusum_h))[4]
    assert moved.state == "changed" and "CUSUM h 8 → 9" in moved.sentence


def test_a_threshold_that_is_off_is_off_however_it_was_written():
    """The form reports an unticked max dwell or merge gap as None, the detector stamps what it
    was handed, and the spin box writes None back as 0: three spellings of one setting."""
    stamped_zero = _table(params=_form(max_dwell_s=0.0, merge_gap_s=0.0))
    assert _judge(stamped_zero, _live())["thresholds"].state == "same"

    turned_on = _judge(_table(), _live(thresholds=_form(max_dwell_s=0.01, merge_gap_s=0.002)))
    verdict = turned_on["thresholds"]
    assert verdict.state == "changed"
    assert "max dwell off → 10 ms" in verdict.sentence
    assert "merge gap off → 2 ms" in verdict.sentence
    assert (verdict.was, verdict.now) == ("max dwell off; merge gap off",
                                          "max dwell 10 ms; merge gap 2 ms")

    # The chain judge's own tolerance: a value read back from a project is not a change.
    assert _judge(_table(), _live(thresholds=_form(k_in=5.0 * (1 + 1e-12))))[
        "thresholds"].state == "same"


# ================================================================================================
# what a result recorded
# ================================================================================================

def test_a_legacy_table_with_no_stamps_is_unknown_on_everything_it_did_not_record():
    legacy = _table(params={}, provenance={}, baseline=None)
    made = made_with(legacy)
    assert (made.rate_hz, made.baseline_fingerprint, made.chain, made.thresholds) == (
        None, None, None, None)

    verdicts = _judge(legacy, _live())
    for name in ("rate", "baseline", "chain", "thresholds"):
        assert verdicts[name].state == "unknown", verdicts[name]
    assert "not recorded" in verdicts["chain"].sentence
    assert verdicts["chain"].action == verdicts["thresholds"].action == ""
    # Every result carries a calibration stamp, so that one is still judged.
    assert verdicts["calibration"].state == "same"

    nothing = judge_inputs(MadeWith("detect.schmitt"), _live())
    assert [v.state for v in nothing] == ["unknown"] * 5


def test_params_come_first_and_the_provenance_stamp_is_the_fallback():
    """As the form's own reader (``schmitt_params_from_result``): a v1/v2 project restores with
    ``params={}``, and the detector's stamp in the provenance is then the truth about the run."""
    stamp = {"fs": FS, "filter_cutoff_hz": 10_000.0, "filter_kind": "bessel", "filter_order": 8,
             "k_in": 5.0, "k_out": 3.0, "min_dwell_s": 2e-5, "max_dwell_s": None,
             "merge_gap_s": None, "polarity": -1, "polarity_requested": None}
    restored = made_with(_table(params={}, provenance=stamp))
    assert restored.chain["filter_cutoff_hz"] == 10_000.0
    assert restored.chain["polarity"] is None          # what the user asked for, not the resolved
    assert restored.thresholds["k_in"] == 5.0

    live_run = made_with(_table(params=_form(k_in=7.0), provenance=stamp))
    assert live_run.thresholds["k_in"] == 7.0          # the dict the module was handed wins

    # A CUSUM stamp keeps its own thresholds.
    cusum = made_with(_table("detect.cusum", params={},
                             provenance={**stamp, "cusum_k": 0.5, "cusum_h": 8.0}))
    assert (cusum.thresholds["cusum_k"], cusum.thresholds["cusum_h"]) == (0.5, 8.0)


def test_a_table_the_chain_did_not_make_is_not_compared_with_the_form():
    """The ML detector runs its own model chain; the form's chain and thresholds did not make
    its table, so nothing is compared or offered -- and the sentence says why."""
    verdicts = _judge(_table("detect.keras", params=_form(k_in=9.0)), _live())
    for name in ("chain", "thresholds"):
        assert verdicts[name].state == "unknown" and verdicts[name].action == ""
        assert "not compared" in verdicts[name].sentence
        assert "ML detector" in verdicts[name].sentence
    assert verdicts["rate"].state == "same"

    # A reconciled table was made from Schmitt/CUSUM tables: its sources are what to compare.
    reconciled = _judge(_table("detect.reconciled"), _live())
    for name in ("chain", "thresholds"):
        assert reconciled[name].state == "unknown" and reconciled[name].action == ""
        assert "compare the form with its sources" in reconciled[name].sentence


def test_a_fit_states_the_rate_and_estimator_of_the_table_it_was_made_on():
    rng = np.random.default_rng(0)
    n = 40
    df = empty_df().reindex(range(n)).reset_index(drop=True)
    df["event_id"] = np.arange(n, dtype=np.int64)
    df["dwell_s"] = 3e-5 + rng.exponential(4e-4, n)
    df["frac_depth"] = np.abs(rng.normal(0.3, 0.03, n))
    df["filter_limited"] = False
    df["accepted"] = True
    df["flag"] = ""
    table = _table(df=df, provenance={"fs": FS, "min_dwell_s": 3e-5})
    fit = fit_dwell(table, doc_id="doc", channel=0, n_boot=20,
                    calibration_fingerprint=table.calibration_fingerprint,
                    detection_module_id=table.module_id)

    made = made_with(fit)
    assert made.module_id == "fit.dwell"
    assert (made.rate_hz, made.calibration, made.baseline_fingerprint) == (
        FS, (1.0, 0.0), BASELINE)
    assert made.chain is None and made.thresholds is None

    verdicts = {v.name: v for v in judge_inputs(made, _live(rate_hz=25_000.0))}
    assert verdicts["rate"].state == "changed"
    assert verdicts["baseline"].state == "same"
    # Its chain and thresholds are its table's, judged on that table.
    for name in ("chain", "thresholds"):
        assert verdicts[name].state == "unknown"
        assert "judge the table it was made on" in verdicts[name].sentence


# ================================================================================================
# purity
# ================================================================================================

def test_the_module_is_qt_free():
    """A value-object layer: importable and testable with no window in sight."""
    tree = ast.parse(pathlib.Path(lineage.__file__).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    assert not any("Qt" in m or "PySide" in m or "pyqtgraph" in m for m in modules), modules
    assert not any("gui" in m.split(".") for m in modules), modules
