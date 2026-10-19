"""conditioning_recipes: complete protocols, the effective chain, the judge, the protocol
file and Copy as Python. Pure: no Qt anywhere in this file."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis import baseline, filters, modules
from nanoscope.analysis import conditioning_recipes as cr
from nanoscope.analysis.conditioning import (
    CONDITIONING_PARAM_KEYS,
    _STAGE_SPELLING,
    chain_description,
    conditioned_trace,
)

FS = 50_000.0

#: The shipped protocols' owner numbers (the same table test_detection_protocols_and_
#: normalize.py pins on the form).
_OWNER = {
    "Host-guest (CB7 + adamantyl)": {
        "notch_freqs": [60.0, 120.0, 180.0, 300.0, 420.0, 540.0], "notch_q": 30.0,
        "detrend_method": "masked_median", "detrend_rough_s": 0.5, "detrend_base_s": 0.02,
        "detrend_decim": 10, "detrend_mask_sigma": 3.5, "detrend_dilate_ms": 0.2,
        "normalize_mode": "model_input", "normalize_clip_sigma": 30.0,
    },
    "Zn2-insulin (zn_30h)": {
        "notch_freqs": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0],
        "detrend_method": "masked_median", "detrend_base_s": 0.2, "detrend_mask_sigma": 3.0,
        "detrend_dilate_ms": 2.0, "normalize_mode": "model_input",
    },
    "Cu2-insulin (SavGol baseline)": {
        "notch_freqs": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0],
        "detrend_method": "savgol", "detrend_window_s": 0.2, "detrend_sg_poly": 3,
        "detrend_sg_decim": 50, "normalize_mode": "model_input",
    },
}


def _form(**over):
    return dict(cr.CHAIN_DEFAULTS, **over)


# -- keys and defaults ------------------------------------------------------------------
def test_chain_keys_are_the_conditioning_keys_without_polarity():
    assert len(cr.CHAIN_KEYS) == 37
    assert len(set(cr.CHAIN_KEYS)) == 37
    assert set(cr.CHAIN_KEYS) == set(CONDITIONING_PARAM_KEYS) - {"polarity"}
    assert tuple(cr.CHAIN_DEFAULTS) == cr.CHAIN_KEYS


def test_chain_defaults_are_the_kernel_defaults():
    d = cr.CHAIN_DEFAULTS
    assert d["notch_freq_hz"] == modules.DEFAULT_NOTCH_FREQ_HZ
    assert d["notch_harmonics"] == modules.DEFAULT_NOTCH_HARMONICS
    assert d["notch_q"] == modules.DEFAULT_NOTCH_Q
    assert d["hp_cutoff_hz"] == modules.DEFAULT_HP_CUTOFF_HZ
    assert (d["hp_order"], d["hp_kind"]) == (modules.DEFAULT_HP_ORDER, modules.DEFAULT_HP_KIND)
    assert d["median_window_s"] == modules.DEFAULT_MEDIAN_WINDOW_S
    assert d["rmean_window_s"] == modules.DEFAULT_RMEAN_WINDOW_S
    assert d["savgol_window_s"] == modules.DEFAULT_SAVGOL_WINDOW_S
    assert d["savgol_polyorder"] == modules.DEFAULT_SAVGOL_POLYORDER
    assert d["detrend_window_s"] == baseline.DEFAULT_DETREND_WINDOW_S
    assert d["normalize_clip_sigma"] == filters.MODEL_INPUT_CLIP_SIGMA
    assert (d["filter_order"], d["filter_kind"], d["filter_cutoff_hz"]) == (8, "bessel", None)
    for method, spec in modules.DETREND_METHOD_PARAMS.items():
        for key, _kernel, default in spec:
            assert d[key] == default, (method, key)
    assert not any(d[k] for k in cr.CHAIN_KEYS if k.endswith("_enabled"))


def test_every_builtin_recipe_is_complete_and_keeps_the_owners_numbers():
    assert set(cr.BUILTIN_RECIPES) == set(_OWNER)
    for name, recipe in cr.BUILTIN_RECIPES.items():
        assert tuple(recipe) == cr.CHAIN_KEYS, name
        for key, want in _OWNER[name].items():
            assert recipe[key] == want, (name, key)
        # SP-04: every stage the protocol does not name is OFF, the low-pass included.
        assert recipe["filter_cutoff_hz"] is None
        for flag in ("hp_enabled", "median_enabled", "rmean_enabled", "savgol_enabled"):
            assert recipe[flag] is False, (name, flag)
        assert recipe["notch_enabled"] and recipe["detrend_enabled"] and recipe["normalize_enabled"]


def test_complete_recipe_keeps_given_values_and_fills_the_rest():
    got = cr.complete_recipe({"notch_enabled": True, "notch_freqs": (50, 150),
                              "k_in": 5.0, "polarity": -1, "detrend_decim": None})
    assert tuple(got) == cr.CHAIN_KEYS
    assert got["notch_enabled"] is True and got["notch_freqs"] == [50.0, 150.0]
    assert got["detrend_decim"] == 0                         # None = not recorded
    assert "k_in" not in got and "polarity" not in got
    assert got["filter_cutoff_hz"] is None
    assert cr.complete_recipe({"filter_cutoff_hz": None})["filter_cutoff_hz"] is None


# -- the effective chain ----------------------------------------------------------------
def test_effective_chain_ignores_disabled_stage_parameters():
    a = _form(hp_cutoff_hz=5.0, median_window_s=0.01, notch_q=3.0)
    assert cr.effective_chain(a) == {}
    assert cr.chains_equal(a, cr.CHAIN_DEFAULTS)
    assert not cr.recipes_equal(a, cr.CHAIN_DEFAULTS)


def test_effective_chain_resolves_the_notch_spelling():
    series = _form(notch_enabled=True, notch_freq_hz=60.0, notch_harmonics=3)
    explicit = _form(notch_enabled=True, notch_freqs=[180.0, 60.0, 120.0])
    assert cr.chains_equal(series, explicit)
    assert cr.chains_equal(series, explicit, fs=FS)
    # Lines at or above Nyquist are pruned once the rate is known (as the kernel does).
    many = _form(notch_enabled=True, notch_freq_hz=3000.0, notch_harmonics=10)
    assert cr.effective_chain(many, 10_000.0)["notch"]["lines"] == [3000.0]
    assert len(cr.effective_chain(many)["notch"]["lines"]) == 10


def test_effective_chain_ignores_window_for_masked_median():
    a = _form(detrend_enabled=True, detrend_method="masked_median", detrend_window_s=0.05)
    b = dict(a, detrend_window_s=5.0)
    assert cr.chains_equal(a, b)
    c = dict(a, detrend_method="block_median")
    d = dict(c, detrend_window_s=5.0)
    assert not cr.chains_equal(c, d)
    # Only the ACTIVE method's knobs count.
    assert cr.chains_equal(c, dict(c, detrend_asls_lam=1e7, detrend_rough_s=9.0))


def test_effective_chain_treats_list_and_tuple_notch_lines_equal():
    a = _form(notch_enabled=True, notch_freqs=[60.0, 120.0])
    b = _form(notch_enabled=True, notch_freqs=(60, 120))
    assert cr.chain_signature(a) == cr.chain_signature(b)


def test_polarity_counts_only_with_detrend():
    off = _form()
    assert cr.chains_equal(dict(off, polarity=-1), dict(off, polarity=1))
    on = _form(detrend_enabled=True)
    assert not cr.chains_equal(dict(on, polarity=-1), dict(on, polarity=1))


def test_the_gaussian_lowpass_order_is_ignored():
    a = _form(filter_cutoff_hz=5000.0, filter_kind="gaussian", filter_order=2)
    assert cr.chains_equal(a, dict(a, filter_order=8))
    b = dict(a, filter_kind="bessel", filter_order=8)
    assert not cr.chains_equal(b, dict(b, filter_order=2))


# -- differences, in words ----------------------------------------------------------------
def test_chain_differences_names_a_lowpass_change():
    run = _form(filter_cutoff_hz=10_000.0)
    form = _form(filter_cutoff_hz=5_000.0)
    diffs = cr.chain_differences(form, run)
    assert [d.text for d in diffs] == ["low-pass 10000 → 5000 Hz"]
    on_off = cr.chain_differences(_form(median_enabled=True), _form())
    assert [d.text for d in on_off] == ["median off → on (0.1 ms)"]
    assert [d.text for d in cr.chain_differences(_form(), _form(hp_enabled=True))] == [
        "high-pass on → off"]
    method = cr.chain_differences(
        _form(detrend_enabled=True, detrend_method="masked_median"),
        _form(detrend_enabled=True, detrend_method="block_median"))
    assert [d.text for d in method] == ["baseline drift block median → event-masked median"]
    q = cr.chain_differences(_form(notch_enabled=True, notch_q=20.0), _form(notch_enabled=True))
    assert [d.text for d in q] == ["notch Q 30 → 20"]


def test_differences_follow_the_stage_order_and_truncate():
    run = _form(filter_cutoff_hz=10_000.0, notch_enabled=True)
    form = _form(filter_cutoff_hz=5_000.0, notch_enabled=True, notch_q=20.0,
                 median_enabled=True, savgol_enabled=True)
    diffs = cr.chain_differences(form, run)
    assert [d.stage for d in diffs] == ["notch", "median", "savgol", "lowpass"]
    text = cr.describe_differences(diffs)
    assert text.endswith("(+1 more)") and text.count(";") == 2


# -- the judge ----------------------------------------------------------------------------
def test_judge_chain_kinds():
    run = _form(filter_cutoff_hz=10_000.0, detrend_enabled=True, polarity=-1)
    assert cr.judge_chain(run, run, module_id=None).kind == "none"
    match = cr.judge_chain(run, run, module_id="detect.schmitt", detector_label="Schmitt")
    assert match.kind == "match" and "Schmitt" in match.sentence
    form = dict(run, filter_cutoff_hz=5_000.0, polarity=1)
    differs = cr.judge_chain(form, run, module_id="detect.cusum", detector_label="CUSUM")
    assert differs.kind == "differs"
    assert differs.sentence.startswith("⚠ The form differs from the chain the CUSUM detection")
    assert "low-pass 10000 → 5000 Hz" in differs.sentence
    assert differs.revert_params is not None
    assert {k: differs.revert_params[k] for k in cr.CHAIN_KEYS} == cr.complete_recipe(run)
    assert differs.revert_params["polarity"] == -1           # named by the diff
    no_pol = cr.judge_chain(dict(run, filter_cutoff_hz=5_000.0), run, module_id="detect.schmitt")
    assert "polarity" not in no_pol.revert_params
    assert cr.judge_chain(run, {}, module_id="detect.schmitt").kind == "unknown"
    ml = cr.judge_chain(run, None, module_id="detect.keras")
    assert ml.kind == "not_classical" and "ML detector" in ml.sentence
    assert cr.judge_chain(run, None, module_id="detect.reconciled").kind == "not_classical"
    assert cr.judge_chain(run, None, module_id="detect.study").kind == "not_classical"


def test_the_headline_puts_the_differences_first():
    """Review DA-3: one elided status line must show WHAT differs, not a fixed preamble."""
    run = _form(filter_cutoff_hz=10_000.0)
    form = dict(run, filter_cutoff_hz=5_000.0, median_enabled=True)
    differs = cr.judge_chain(form, run, module_id="detect.schmitt", detector_label="Schmitt")
    assert differs.headline == ("⚠ Differs from the Schmitt run: "
                                + cr.describe_differences(differs.diffs))
    assert differs.headline.index("median off → on") < 45
    for same in (cr.judge_chain(run, run, module_id="detect.schmitt", detector_label="Schmitt"),
                 cr.judge_chain(run, None, module_id="detect.keras"), cr.ChainStatus.none()):
        assert same.headline == same.sentence


def test_a_detection_run_judges_as_matching_its_own_form(context_factory):
    from nanoscope.analysis.modules import DetectSchmitt, detection_params_from_provenance

    rng = np.random.default_rng(3)
    n = 200_000
    t = np.arange(n) / FS
    raw = (-1000.0 + rng.normal(0, 8.0, n) + 15.0 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)
    for s in range(20_000, n, 40_000):
        raw[s:s + 300] += 250.0
    form = {"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 2e-5, "polarity": None,
            **_form(filter_cutoff_hz=10_000.0, notch_enabled=True, detrend_enabled=True,
                    detrend_window_s=0.2)}
    res = DetectSchmitt().run(context_factory(raw, fs=FS, params=dict(form)))
    prov = res.payload.provenance
    assert cr.judge_chain(form, detection_params_from_provenance(prov),
                          module_id="detect.schmitt", fs=FS).kind == "match"
    assert cr.judge_chain(form, res.params, module_id="detect.schmitt", fs=FS).kind == "match"


# -- the stage list -----------------------------------------------------------------------
def _flags_in_order(fn) -> list[str]:
    tree = ast.parse(inspect.getsource(fn))
    flags = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).endswith("_enabled")):
            flags.append((node.lineno, node.args[0].value))
    return [f for _, f in sorted(flags)]


def test_stage_rows_follow_the_kernel_order():
    rows = cr.stage_rows(cr.CHAIN_DEFAULTS)
    assert tuple(r.key for r in rows) == cr.STAGE_KEYS
    assert [r.position for r in rows] == list(range(1, 9))
    kernel = []
    for flag in (_flags_in_order(modules.preprocess_for_detection)
                 + _flags_in_order(modules._apply_smoothers)
                 + _flags_in_order(modules._apply_normalize)):
        if flag not in kernel:
            kernel.append(flag)
    assert kernel == [cr.STAGE_ENABLE_KEY[k] for k in cr.STAGE_KEYS[:-1]]
    assert [flag for flag, _ in _STAGE_SPELLING] == kernel
    assert cr.STAGE_KEYS[-1] == "lowpass"


def test_stage_row_summaries_and_roles():
    zn = {r.key: r for r in cr.stage_rows(cr.BUILTIN_RECIPES["Zn2-insulin (zn_30h)"])}
    assert zn["notch"].summary == "9 lines 60…660 Hz · Q 30"
    assert zn["detrend"].summary == "event-masked median · rough 0.5 s / base 0.2 s"
    assert zn["normalize"].summary == "model input · clip 30σ"
    assert zn["detrend"].role == "estimator" and zn["normalize"].role == "changes units"
    assert zn["lowpass"].role == "applied inside detection" and not zn["lowpass"].enabled
    assert zn["notch"].enabled and not zn["median"].enabled
    rows = {r.key: r for r in cr.stage_rows(
        _form(filter_cutoff_hz=10_000.0, hp_enabled=True), FS)}
    assert rows["highpass"].summary == "100 Hz · Butterworth 4"
    assert rows["highpass"].role == "localization only"
    # The true corner is in the row's tooltip (StageRow.detail). UPDATED (review DA-5): the
    # setting now precedes the role note, so a right-elided row loses the note (which the
    # tooltip repeats), never the cutoff.
    assert rows["lowpass"].summary == "Bessel 8 · 10 kHz"
    assert rows["lowpass"].detail == "−3 dB 7.6 kHz"
    assert rows["lowpass"].text == "8  Low-pass — Bessel 8 · 10 kHz (applied inside detection)"
    assert rows["highpass"].text == "3  High-pass — 100 Hz · Butterworth 4 (localization only)"
    assert cr.stage_rows(_form())[0].text == "1  Notch — 60/120/180 Hz · Q 30", "no role, no ()"
    assert rows["median"].summary == "0.1 ms" and not rows["median"].linear
    assert cr.stage_rows(_form(), lowpass_design_hz=5000.0)[-1].summary == "Bessel 8 · 5 kHz"
    assert cr.stage_rows(_form())[-1].summary == "off"


# -- validation and the protocol file -----------------------------------------------------
def test_validate_recipe_refuses_bad_values_and_unknown_keys():
    ok, why = cr.validate_recipe({"notch_enabled": True, "notch_freqs": [50, 100]})
    assert why == () and ok["notch_freqs"] == [50.0, 100.0] and tuple(ok) == cr.CHAIN_KEYS
    for bad, word in (
            ({"k_in": 5.0}, "k_in"),
            ({"notch_freqs": [float(i + 1) for i in range(33)]}, "33 lines"),
            ({"filter_kind": "elliptic"}, "filter_kind"),
            ({"detrend_method": "constant"}, "detrend_method"),
            ({"detrend_window_s": -1.0}, "detrend_window_s"),
            ({"notch_enabled": "yes"}, "notch_enabled"),
            ({"filter_order": 11}, "filter_order"),
            ({"savgol_window_s": float("nan")}, "savgol_window_s")):
        recipe, why = cr.validate_recipe(bad)
        assert recipe is None, bad
        assert any(word in w for w in why), (bad, why)
    # 0 is legal where the form allows it (auto stride, no dilation).
    ok, why = cr.validate_recipe({"detrend_decim": 0, "detrend_dilate_ms": 0.0})
    assert why == ()


@pytest.mark.parametrize("key, value", [
    ("hp_cutoff_hz", 0.05), ("notch_freq_hz", 0.5), ("filter_cutoff_hz", 0.5),
    ("filter_cutoff_hz", 6e6), ("hp_cutoff_hz", 6e6), ("notch_freq_hz", 2e5)])
def test_validate_recipe_refuses_frequencies_outside_the_boxes_static_ranges(key, value):
    """Review D2: below a box's minimum the form would CLAMP the value on recall, which the
    import promised never to do."""
    recipe, why = cr.validate_recipe({key: value})
    assert recipe is None and any(key in w for w in why), why
    ok, why = cr.validate_recipe({"hp_cutoff_hz": 0.1, "notch_freq_hz": 1.0,
                                  "filter_cutoff_hz": 1.0})
    assert why == ()


def test_lenient_validation_checks_types_and_choices_but_not_ranges():
    recipe, why = cr.validate_recipe({"hp_cutoff_hz": 0.05, "detrend_window_s": 120.0},
                                     ranges=False)
    assert why == () and recipe["hp_cutoff_hz"] == 0.05
    for bad in ({"notch_freqs": 5}, {"filter_kind": "elliptic"}, {"hp_order": 2.5},
                {"notch_enabled": "yes"}):
        assert cr.validate_recipe(bad, ranges=False)[0] is None, bad


def test_recipe_key_differences_names_each_changed_key():
    a = cr.complete_recipe({"hp_cutoff_hz": 0.05, "filter_cutoff_hz": 40_000.0})
    b = dict(a, hp_cutoff_hz=0.1, filter_cutoff_hz=24_999.9)
    assert cr.recipe_key_differences(a, b) == [
        "filter_cutoff_hz 40000 → 24999.9", "hp_cutoff_hz 0.05 → 0.1"]
    assert cr.recipe_key_differences(a, a) == []


def test_protocol_file_round_trip_is_staged_and_versioned(tmp_path, monkeypatch):
    from nanoscope.core import atomic_write

    staged: list[Path] = []
    real = atomic_write.staged_path

    def spy(target, **kw):
        staged.append(Path(target))
        return real(target, **kw)

    monkeypatch.setattr(cr, "staged_path", spy)
    lib = {"mine": cr.complete_recipe({"median_enabled": True}),
           **{k: v for k, v in cr.BUILTIN_RECIPES.items() if k.startswith("Cu2")}}
    out = cr.write_protocols_file(tmp_path / "p.json", lib, app_version="9.9")
    assert staged == [tmp_path / "p.json"] and out.exists()
    doc = json.loads(out.read_text())
    assert doc["format"] == cr.PROTOCOL_FILE_FORMAT and doc["version"] == 1
    assert doc["app_version"] == "9.9" and "exported" in doc
    back, refused = cr.read_protocols_file(out)
    assert refused == () and back == lib


def test_a_foreign_json_is_refused_by_name(tmp_path):
    p = tmp_path / "other.json"
    p.write_text(json.dumps({"format": "something-else", "version": 1, "protocols": {}}))
    with pytest.raises(ValueError, match="not a NanoScope conditioning-protocol file"):
        cr.read_protocols_file(p)
    p.write_text(json.dumps({"format": cr.PROTOCOL_FILE_FORMAT, "version": 2, "protocols": {}}))
    with pytest.raises(ValueError, match="version 2"):
        cr.read_protocols_file(p)
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not a NanoScope"):
        cr.read_protocols_file(p)
    p.write_text(json.dumps({"format": cr.PROTOCOL_FILE_FORMAT, "version": 1, "protocols": {
        "good": {"median_enabled": True}, "bad": {"k_in": 3.0}}}))
    got, refused = cr.read_protocols_file(p)
    assert list(got) == ["good"] and len(refused) == 1 and "k_in" in refused[0]


def test_import_never_overwrites():
    mine = cr.complete_recipe({"median_enabled": True})
    other = cr.complete_recipe({"savgol_enabled": True})
    existing = {"A": mine}
    reserved = {**cr.BUILTIN_RECIPES, cr.MANUAL_PROTOCOL: None}
    merged, report = cr.merge_imported_protocols(
        existing, {"A": other, "B": mine, "Zn2-insulin (zn_30h)": other}, reserved=reserved)
    assert merged["A"] == mine and merged["A (imported)"] == other
    assert merged["B"] == mine
    assert merged["Zn2-insulin (zn_30h) (imported)"] == other
    assert "Zn2-insulin (zn_30h)" not in merged
    assert report.added == ["B"] and len(report.renamed) == 2
    again, report2 = cr.merge_imported_protocols(merged, {"A": mine, "A (imported)": other},
                                                 reserved=reserved)
    assert again == merged and report2.identical == ["A", "A (imported)"]
    assert report2.sentence() == "Imported 0 protocols (2 already present)."
    assert report.sentence() == "Imported 3 protocols (2 renamed)."
    _m, r3 = cr.merge_imported_protocols(merged, {"A": other}, reserved=reserved)
    assert r3.renamed == [("A", "A (imported 2)")]


# -- Copy as Python -----------------------------------------------------------------------
_RANDOM_CHAINS = [
    _form(filter_cutoff_hz=8000.0),
    _form(notch_enabled=True, notch_freqs=[60.0, 180.0], filter_cutoff_hz=5000.0,
          filter_kind="butter", filter_order=4),
    _form(median_enabled=True, savgol_enabled=True),
    _form(hp_enabled=True, hp_cutoff_hz=50.0, filter_cutoff_hz=9000.0),
    _form(detrend_enabled=True, detrend_window_s=0.05, rmean_enabled=True),
    _form(notch_enabled=True, detrend_enabled=True, detrend_method="masked_median",
          detrend_rough_s=0.05, detrend_base_s=0.01, normalize_enabled=True,
          normalize_mode="model_input"),
]


class _FakeDoc:
    def __init__(self, raw, fs):
        self._raw = raw
        self.meta = type("M", (), {"sample_rate_hz": fs})()

    def raw(self, ch):
        assert ch == 1
        return self._raw


def _exec_script(script: str, monkeypatch, doc) -> dict:
    import nanoscope.io.base as io_base

    monkeypatch.setattr(io_base, "open_document", lambda path, rng=None: doc)
    ns: dict = {}
    exec(compile(script, "<copied>", "exec"), ns)
    return ns


@pytest.mark.parametrize("chain", _RANDOM_CHAINS, ids=[chain_description(c)[:40] for c in _RANDOM_CHAINS])
def test_python_script_rebuilds_the_same_bytes(chain, monkeypatch):
    rng = np.random.default_rng(7)
    raw = (100.0 + rng.normal(0, 5.0, 60_000)).astype(np.float32)
    raw[30_000:30_400] -= 40.0
    gain, offset = 2.5, 1.0
    script = cr.python_script(chain, path="/data/x.abf", channel_index=1, channel_name="IN 1",
                              units="pA", fs=FS, cal_gain=gain, cal_offset=offset,
                              window=(1000, 9000), polarity=-1, app_version="test")
    ast.parse(script)
    assert chain_description(chain) in script
    ns = _exec_script(script, monkeypatch, _FakeDoc(raw, FS))
    calibrated = raw * np.float32(gain) + np.float32(offset)
    want = conditioned_trace(calibrated, FS, dict(chain, polarity=-1))
    assert hashlib.sha256(np.asarray(ns["processed"]).tobytes()).hexdigest() == \
        hashlib.sha256(np.asarray(want).tobytes()).hexdigest()
    assert np.array_equal(ns["processed_view"], want[1000:9000])
    if chain["hp_enabled"]:
        assert "measurement" in ns


def test_python_script_reads_a_real_file_through_its_reader(tmp_path):
    from nanoscope.io.base import open_document

    n = 20_000
    rng = np.random.default_rng(1)
    values = 50.0 + rng.normal(0, 2.0, n)
    path = tmp_path / "trace.csv"
    rows = ["time,current"] + [f"{i / FS!r},{v!r}" for i, v in enumerate(values)]
    path.write_text("\n".join(rows) + "\n")
    doc = open_document(path)
    chain = _form(filter_cutoff_hz=4000.0, median_enabled=True)
    script = cr.python_script(chain, path=str(path), channel_index=0,
                              fs=doc.meta.sample_rate_hz, cal_gain=1.0, cal_offset=0.0)
    ns: dict = {}
    exec(compile(script, "<copied>", "exec"), ns)
    want = conditioned_trace(doc.raw(0), doc.meta.sample_rate_hz, chain)
    assert np.array_equal(ns["processed"], want)


def test_python_script_without_a_path_says_so_and_raises():
    script = cr.python_script(_form(), path=None, channel_index=0, fs=FS)
    assert "PATH = None  # set this to the recording" in script
    with pytest.raises(SystemExit, match="Set PATH"):
        exec(compile(script, "<copied>", "exec"), {})


def test_python_script_cuts_a_segment_before_conditioning(monkeypatch):
    rng = np.random.default_rng(2)
    raw = (10.0 + rng.normal(0, 1.0, 40_000)).astype(np.float32)
    chain = _form(detrend_enabled=True, detrend_window_s=0.02)
    script = cr.python_script(chain, path="/p.abf", channel_index=1, fs=FS,
                              segment=(5_000, 25_000))
    ns = _exec_script(script, monkeypatch, _FakeDoc(raw, FS))
    want = conditioned_trace(np.ascontiguousarray(raw[5_000:25_000]), FS, chain)
    assert np.array_equal(ns["processed"], want)


def test_the_recipes_module_imports_no_qt_and_no_gui():
    src = Path(cr.__file__).read_text()
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    assert not any("PySide" in n or "pyqtgraph" in n or "gui" in n for n in names), names
