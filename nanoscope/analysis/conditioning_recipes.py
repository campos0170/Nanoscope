"""Signal-chain recipes: complete protocols, the effective chain and the chain judge.

Pure and Qt-free. The Signal Conditioning window, the Preprocess card, the protocol
library and the event-table "chain changed" chip all answer the same questions from
here, so they cannot disagree:

* :data:`CHAIN_KEYS` / :data:`CHAIN_DEFAULTS` / :func:`complete_recipe` -- a protocol is
  a COMPLETE chain. Recalling a partial dict left unrelated stages on under the
  protocol's name (SP-04).
* :func:`effective_chain` / :func:`chain_differences` / :func:`judge_chain` -- does the
  form's chain differ from the chain the detection on screen ran with, and how.
* :func:`stage_rows` -- the eight stages in EXECUTION order, as the window lists them.
* :func:`validate_recipe` / :func:`write_protocols_file` / :func:`read_protocols_file` /
  :func:`merge_imported_protocols` -- the JSON protocol exchange.
* :func:`python_script` -- "Copy as Python".
"""

from __future__ import annotations

import json
import math
import pprint
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..core.atomic_write import staged_path
from . import baseline as baseline_mod
from . import filters
from .conditioning import CONDITIONING_PARAM_KEYS, chain_description
from .modules import (
    DETREND_PARAM_KEYS,
    FILTER_PARAM_KEYS,
    detrend_method_kwargs,
    notch_freqs_param,
)

__all__ = [
    "BUILTIN_RECIPES",
    "CHAIN_DEFAULTS",
    "CHAIN_KEYS",
    "DETREND_METHOD_LABELS",
    "MANUAL_PROTOCOL",
    "PROTOCOL_FILE_FORMAT",
    "PROTOCOL_FILE_VERSION",
    "SCOPE_SENTENCE",
    "STAGE_ENABLE_KEY",
    "STAGE_KEYS",
    "STAGE_TITLES",
    "ChainDiff",
    "ChainStatus",
    "MergeReport",
    "StageRow",
    "chain_differences",
    "chain_signature",
    "chains_equal",
    "complete_recipe",
    "describe_differences",
    "effective_chain",
    "judge_chain",
    "merge_imported_protocols",
    "python_script",
    "read_protocols_file",
    "recipe_key_differences",
    "recipes_equal",
    "stage_rows",
    "validate_recipe",
    "write_protocols_file",
]

#: Every key a conditioning protocol captures, in the order the protocol store has
#: always written them. The polarity is deliberately absent: it is a detector choice
#: that the detrend stage also reads, and a protocol must not rewrite it.
CHAIN_KEYS: tuple[str, ...] = (
    "filter_cutoff_hz", "filter_order", "filter_kind",
    "detrend_enabled", "detrend_method", "detrend_window_s", "detrend_percentile",
    "detrend_asls_lam", "detrend_asls_p", "detrend_asls_niter", "detrend_asls_decim",
    "detrend_rough_s", "detrend_base_s", "detrend_mask_sigma", "detrend_dilate_ms",
    "detrend_decim", "detrend_sg_poly", "detrend_sg_decim",
    "notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q", "notch_freqs",
    "median_enabled", "median_window_s",
    "hp_enabled", "hp_cutoff_hz", "hp_order", "hp_kind",
    "rmean_enabled", "rmean_window_s",
    "savgol_enabled", "savgol_window_s", "savgol_polyorder",
    "normalize_enabled", "normalize_mode", "normalize_clip_sigma",
)
assert set(CHAIN_KEYS) == set(CONDITIONING_PARAM_KEYS) - {"polarity"}

#: A fresh Preprocess form, key for key (measured on a fresh DetectionPanel). Every
#: stage is off and the low-pass is off.
CHAIN_DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "filter_cutoff_hz": None, "filter_order": 8, "filter_kind": "bessel",
    "detrend_enabled": False, "detrend_method": "block_median",
    "detrend_window_s": baseline_mod.DEFAULT_DETREND_WINDOW_S,
    "detrend_percentile": 80.0,
    "detrend_asls_lam": 6e4, "detrend_asls_p": 1e-3, "detrend_asls_niter": 10,
    "detrend_asls_decim": 0.1,
    "detrend_rough_s": 0.5, "detrend_base_s": 0.1, "detrend_mask_sigma": 3.5,
    "detrend_dilate_ms": 1.0, "detrend_decim": 0,
    "detrend_sg_poly": baseline_mod.SAVGOL_POLY, "detrend_sg_decim": baseline_mod.SAVGOL_DECIM,
    "notch_enabled": False, "notch_freq_hz": 60.0, "notch_harmonics": 3, "notch_q": 30.0,
    "notch_freqs": None,
    "median_enabled": False, "median_window_s": 1e-4,
    "hp_enabled": False, "hp_cutoff_hz": 100.0, "hp_order": 4, "hp_kind": "butter",
    "rmean_enabled": False, "rmean_window_s": 5e-4,
    "savgol_enabled": False, "savgol_window_s": 1e-3, "savgol_polyorder": 3,
    "normalize_enabled": False, "normalize_mode": "zscore",
    "normalize_clip_sigma": float(filters.MODEL_INPUT_CLIP_SIGMA),
})

#: Keys for which ``None`` is a value rather than "absent".
_NONE_IS_A_VALUE = frozenset({"filter_cutoff_hz", "notch_freqs"})

#: Baseline-drift estimator labels, keyed by the method string
#: :func:`nanoscope.analysis.baseline.remove_baseline_wander` accepts.
#:
#: THE SAVITZKY-GOLAY LABEL SAYS "(subtract)" FOR A REASON. There are two
#: Savitzky-Golay stages in the Preprocess tab and they are OPPOSITE operations
#: sharing a name: the smoother on the Savitzky-Golay page RETURNS the polynomial fit
#: (it keeps the smooth trace), while this estimator SUBTRACTS it (it keeps the
#: residual). Labelling both "Savitzky-Golay" is how a user ends up smoothing when
#: they meant to detrend.
DETREND_METHOD_LABELS: Mapping[str, str] = MappingProxyType({
    "block_median": "block median",
    "percentile": "percentile",
    "asls": "ASLS",
    "masked_median": "event-masked median",
    "robust_median": "robust median",
    "savgol": "Savitzky-Golay baseline (subtract)",
})

#: Methods whose ``window`` knob the kernel never reads (see
#: ``DetectionPanel._DETREND_WINDOW_IGNORED``).
_DETREND_WINDOW_IGNORED = frozenset({"masked_median"})

#: The stages in KERNEL order (``modules.preprocess_for_detection``), with the
#: detector-owned low-pass last.
STAGE_KEYS: tuple[str, ...] = (
    "notch", "median", "highpass", "detrend", "rmean", "savgol", "normalize", "lowpass")
STAGE_TITLES: Mapping[str, str] = MappingProxyType({
    "notch": "Notch", "median": "Median", "highpass": "High-pass",
    "detrend": "Baseline drift", "rmean": "Running mean", "savgol": "Savitzky-Golay",
    "normalize": "Normalize", "lowpass": "Low-pass",
})
#: Each stage's enable key; the low-pass is on when ``filter_cutoff_hz`` is not None.
STAGE_ENABLE_KEY: Mapping[str, str | None] = MappingProxyType({
    "notch": "notch_enabled", "median": "median_enabled", "highpass": "hp_enabled",
    "detrend": "detrend_enabled", "rmean": "rmean_enabled", "savgol": "savgol_enabled",
    "normalize": "normalize_enabled", "lowpass": None,
})
_STAGE_PHRASE = {
    "notch": "notch", "median": "median", "highpass": "high-pass",
    "detrend": "baseline drift", "rmean": "running mean", "savgol": "Savitzky-Golay",
    "normalize": "normalize", "lowpass": "low-pass",
}
_STAGE_ROLE = {
    "highpass": "localization only", "detrend": "estimator",
    "normalize": "changes units", "lowpass": "applied inside detection",
}
_LINEAR_STAGES = frozenset({"notch", "highpass", "rmean", "savgol", "lowpass"})

#: Where this chain applies, said on the card, in the window and on the stage list.
SCOPE_SENTENCE = (
    "Used by Schmitt/CUSUM detection, Measure ‘Processed signal’ and Export processed "
    "trace — not by the ML detector (ML tab ▸ Model preprocessing).")

#: The protocol combo's no-protocol entry.
MANUAL_PROTOCOL = "— manual —"

_KIND_LABELS = {"bessel": "Bessel", "butter": "Butterworth", "gaussian": "Gaussian"}
_NORMALIZE_LABELS = {
    "zscore": "z-score", "robust": "robust z", "minmax": "min–max",
    filters.MODEL_INPUT_MODE: "model input",
}


# ---------------------------------------------------------------------------
# the three shipped protocols
# ---------------------------------------------------------------------------
#: SHIPPED preprocessing protocols, listed above the user's own and not editable.
#:
#: These are the owner's own three pipelines, recovered from his notebooks and
#: cross-checked against the ``*_metadata.json`` his code writes beside every
#: processed recording. They exist as DETECTION protocols (this combo) and, under
#: matching names, as ML "Model preprocessing" presets
#: (``ml_preprocess.PREPROCESS_PRESETS``: ``hg_student`` / ``zn_student`` /
#: ``cu_student``) -- the two mechanisms drive two different chains, and each recipe
#: needs both halves because the same conditioning feeds both detectors.
#:
#: A protocol captures PREPROCESSING ONLY (`CHAIN_KEYS`), so none of these
#: touches a threshold, a polarity or a dwell limit. What they cannot express is
#: named in each entry's own comment; nothing is silently approximated.
_BUILTIN_PARTIALS: dict[str, dict[str, Any]] = {
    # HOST-GUEST (CB7 + N-adamantyl), +75 mV, RESISTIVE. Six notch lines with three
    # gaps -- 240, 360 and 480 Hz were quiet in the Welch census, so the series
    # spelling cannot express this set without notching three clean bands.
    "Host-guest (CB7 + adamantyl)": {
        "notch_enabled": True,
        "notch_freqs": [60.0, 120.0, 180.0, 300.0, 420.0, 540.0],
        "notch_freq_hz": 60.0, "notch_harmonics": 3, "notch_q": 30.0,
        "detrend_enabled": True, "detrend_method": "masked_median",
        "detrend_rough_s": 0.5, "detrend_base_s": 0.02, "detrend_decim": 10,
        "detrend_mask_sigma": 3.5, "detrend_dilate_ms": 0.2,
        "normalize_enabled": True, "normalize_mode": "model_input",
        "normalize_clip_sigma": 30.0,
    },
    # Zn2-INSULIN (zn_30h), -200 mV, CONDUCTIVE. Nine lines, 480 Hz absent.
    # A Savitzky-Golay baseline was TESTED AND REJECTED on this data (the events are
    # shallow and abundant, so a polynomial envelope dives into them); the
    # event-masked median is what survived.
    "Zn2-insulin (zn_30h)": {
        "notch_enabled": True,
        "notch_freqs": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0,
                        660.0],
        "notch_freq_hz": 60.0, "notch_harmonics": 3, "notch_q": 30.0,
        "detrend_enabled": True, "detrend_method": "masked_median",
        "detrend_rough_s": 0.5, "detrend_base_s": 0.2, "detrend_decim": 10,
        "detrend_mask_sigma": 3.0, "detrend_dilate_ms": 2.0,
        "normalize_enabled": True, "normalize_mode": "model_input",
        "normalize_clip_sigma": 30.0,
    },
    # Cu2-INSULIN, -200 mV, CONDUCTIVE. SAVITZKY-GOLAY, NOT masked median: the
    # notebook's own cell-0 markdown still says "event-masked robust median" and is
    # WRONG -- Block 0's BASELINE_METHOD dict says "sg" for all four cu_* recordings
    # and all four written cu_*_metadata.json agree, with masked_median_params null.
    # VERIFIED: window 0.2 s / poly 3 / decim 50 reproduces the baseline_pA column
    # the owner wrote for all four recordings bit-identically (max |diff| exactly
    # 0.0 pA over 13.9-20.2 M samples each), once the nine-line notch has run.
    "Cu2-insulin (SavGol baseline)": {
        "notch_enabled": True,
        "notch_freqs": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0,
                        660.0],
        "notch_freq_hz": 60.0, "notch_harmonics": 3, "notch_q": 30.0,
        "detrend_enabled": True, "detrend_method": "savgol",
        "detrend_window_s": 0.2, "detrend_sg_poly": 3, "detrend_sg_decim": 50,
        "normalize_enabled": True, "normalize_mode": "model_input",
        "normalize_clip_sigma": 30.0,
    },
}


def complete_recipe(partial: Mapping[str, Any] | None) -> dict[str, Any]:
    """``partial``'s chain keys over :data:`CHAIN_DEFAULTS`: every stage, in CHAIN_KEYS order.

    Keys outside the chain are dropped. ``None`` is kept only where it is a value
    (``filter_cutoff_hz`` = low-pass off, ``notch_freqs`` = series spelling); elsewhere
    it means "not recorded" (a legacy provenance) and the default is used.
    """
    partial = partial or {}
    out = dict(CHAIN_DEFAULTS)
    for key in CHAIN_KEYS:
        if key not in partial:
            continue
        value = partial[key]
        if value is None and key not in _NONE_IS_A_VALUE:
            continue
        out[key] = value
    if out["notch_freqs"] is not None:
        out["notch_freqs"] = [float(f) for f in out["notch_freqs"]]
    return {key: out[key] for key in CHAIN_KEYS}


BUILTIN_RECIPES: Mapping[str, dict[str, Any]] = MappingProxyType(
    {name: complete_recipe(p) for name, p in _BUILTIN_PARTIALS.items()})


# ---------------------------------------------------------------------------
# the effective chain
# ---------------------------------------------------------------------------
def _notch_lines(p: Mapping[str, Any], fs: float | None) -> list[float]:
    explicit = notch_freqs_param(p)
    freq = float(p["notch_freq_hz"])
    harmonics = int(p["notch_harmonics"])
    if fs:
        try:
            lines = filters.notch_frequencies(float(fs), freq, harmonics=harmonics,
                                              freqs=explicit)
            return sorted(float(x) for x in lines)
        except (TypeError, ValueError):
            pass
    if explicit is not None:
        return sorted(float(x) for x in explicit)
    return [freq * k for k in range(1, max(harmonics, 0) + 1)]


def effective_chain(params: Mapping[str, Any] | None, fs: float | None = None) -> dict[str, Any]:
    """The values that actually change the array, per ENABLED stage, in kernel order.

    Disabled stages contribute nothing; the notch spelling is resolved to its line set
    (Nyquist-pruned when ``fs`` is known); only the active detrend method's knobs count
    and ``window_s`` not at all for the event-masked median; the polarity counts only
    inside an enabled detrend; the order of an order-less low-pass kind is dropped.
    """
    params = params or {}
    p = complete_recipe(params)
    chain: dict[str, Any] = {}
    if p["notch_enabled"]:
        chain["notch"] = {"lines": _notch_lines(p, fs), "q": float(p["notch_q"])}
    if p["median_enabled"]:
        chain["median"] = {"window_s": float(p["median_window_s"])}
    if p["hp_enabled"]:
        chain["highpass"] = {"cutoff_hz": float(p["hp_cutoff_hz"]),
                             "order": int(p["hp_order"]), "kind": str(p["hp_kind"])}
    if p["detrend_enabled"]:
        method = str(p["detrend_method"])
        stage: dict[str, Any] = {"method": method}
        if method not in _DETREND_WINDOW_IGNORED:
            stage["window_s"] = float(p["detrend_window_s"])
        stage.update(detrend_method_kwargs(p))
        polarity = params.get("polarity")
        stage["polarity"] = None if polarity is None else int(polarity)
        chain["detrend"] = stage
    if p["rmean_enabled"]:
        chain["rmean"] = {"window_s": float(p["rmean_window_s"])}
    if p["savgol_enabled"]:
        chain["savgol"] = {"window_s": float(p["savgol_window_s"]),
                           "polyorder": int(p["savgol_polyorder"])}
    if p["normalize_enabled"]:
        mode = str(p["normalize_mode"])
        stage = {"mode": mode}
        if mode == filters.MODEL_INPUT_MODE:
            stage["clip_sigma"] = float(p["normalize_clip_sigma"])
        chain["normalize"] = stage
    if p["filter_cutoff_hz"] is not None:
        kind = str(p["filter_kind"])
        stage = {"cutoff_hz": float(p["filter_cutoff_hz"]), "kind": kind}
        if kind not in filters.KINDS_WITHOUT_ORDER:
            stage["order"] = int(p["filter_order"])
        chain["lowpass"] = stage
    return chain


def chain_signature(params: Mapping[str, Any] | None, fs: float | None = None) -> str:
    """A stable string for :func:`effective_chain` (dict key / cache key)."""
    return json.dumps(effective_chain(params, fs), sort_keys=True)


def _same(a: Any, b: Any, rel: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-12)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rel) for k in a)
    return a == b


def chains_equal(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, *,
                 fs: float | None = None, rel: float = 1e-9) -> bool:
    """Whether two params dicts condition the signal identically (floats to ``rel``)."""
    return _same(effective_chain(a, fs), effective_chain(b, fs), rel)


def recipes_equal(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, *,
                  rel: float = 1e-9) -> bool:
    """Whether two params dicts are the same COMPLETE recipe, key for key (floats to
    ``rel``) — disabled stages' settings included, unlike :func:`chains_equal`."""
    return _same(complete_recipe(a), complete_recipe(b), rel)


# ---------------------------------------------------------------------------
# differences, in words
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainDiff:
    """One difference, worded run → form (``'low-pass 10000 → 5000 Hz'``)."""

    stage: str
    text: str
    key: str = ""


def _g(v: Any) -> str:
    return f"{float(v):g}"


def _ms(v: Any) -> str:
    return f"{float(v) * 1e3:g}"


def _lines_text(lines: Sequence[float]) -> str:
    if len(lines) > 6:
        return f"{len(lines)} lines {_g(lines[0])}…{_g(lines[-1])} Hz"
    return "/".join(_g(x) for x in lines) + " Hz" if lines else "no lines"


def _polarity_text(v: Any) -> str:
    return {None: "auto", -1: "blockade", 1: "enhancement"}.get(v, str(v))


_DETREND_KNOB_TEXT: dict[str, tuple[str, Any]] = {
    "window_s": ("window", lambda v: f"{_g(v)} s"),
    "percentile": ("percentile", _g),
    "asls_lam": ("λ", _g),
    "asls_p": ("asymmetry p", _g),
    "asls_niter": ("iterations", _g),
    "asls_decim": ("ASLS decimation", _g),
    "rough_s": ("rough", lambda v: f"{_g(v)} s"),
    "base_s": ("base", lambda v: f"{_g(v)} s"),
    "mask_sigma": ("mask", lambda v: f"{_g(v)}σ"),
    "dilate_ms": ("dilate", lambda v: f"{_g(v)} ms"),
    "decim": ("stride", lambda v: "auto" if v is None else _g(v)),
    "sg_poly": ("SG order", _g),
    "sg_decim": ("SG decimation", _g),
}


def _stage_summary(key: str, st: Mapping[str, Any]) -> str:
    """The short setting text shown on a stage row / in an 'off → on (…)' diff."""
    if key == "notch":
        lines = st["lines"]
        return f"{_lines_text(lines)} · Q {_g(st['q'])}"
    if key in ("median", "rmean"):
        return f"{_ms(st['window_s'])} ms"
    if key == "highpass":
        return f"{_g(st['cutoff_hz'])} Hz · {_KIND_LABELS.get(st['kind'], st['kind'])} {st['order']}"
    if key == "detrend":
        label = DETREND_METHOD_LABELS.get(st["method"], st["method"]).replace(" (subtract)", "")
        if st["method"] == "masked_median":
            return f"{label} · rough {_g(st['rough_s'])} s / base {_g(st['base_s'])} s"
        return f"{label} · {_g(st.get('window_s', 0.0))} s"
    if key == "savgol":
        return f"{_ms(st['window_s'])} ms · order {st['polyorder']}"
    if key == "normalize":
        label = _NORMALIZE_LABELS.get(st["mode"], st["mode"])
        if "clip_sigma" in st:
            return f"{label} · clip {_g(st['clip_sigma'])}σ"
        return label
    if key == "lowpass":
        kind = _KIND_LABELS.get(st["kind"], st["kind"])
        cutoff = float(st["cutoff_hz"])
        freq = f"{cutoff / 1e3:g} kHz" if cutoff >= 1e3 else f"{cutoff:g} Hz"
        return f"{kind} {st['order']} · {freq}" if "order" in st else f"{kind} · {freq}"
    return ""


def _field_diffs(key: str, run: Mapping[str, Any], form: Mapping[str, Any],
                 rel: float) -> list[ChainDiff]:
    name = _STAGE_PHRASE[key]
    out: list[ChainDiff] = []

    def add(text: str, field_key: str = "") -> None:
        out.append(ChainDiff(key, text, field_key))

    if key == "lowpass":
        if not _same(run["cutoff_hz"], form["cutoff_hz"], rel):
            add(f"{name} {_g(run['cutoff_hz'])} → {_g(form['cutoff_hz'])} Hz", "cutoff_hz")
        if run["kind"] != form["kind"]:
            add(f"{name} {_KIND_LABELS.get(run['kind'], run['kind'])} → "
                f"{_KIND_LABELS.get(form['kind'], form['kind'])}", "kind")
        if "order" in run and "order" in form and run["order"] != form["order"]:
            add(f"{name} order {run['order']} → {form['order']}", "order")
    elif key == "highpass":
        if not _same(run["cutoff_hz"], form["cutoff_hz"], rel):
            add(f"{name} {_g(run['cutoff_hz'])} → {_g(form['cutoff_hz'])} Hz", "cutoff_hz")
        if run["kind"] != form["kind"]:
            add(f"{name} {_KIND_LABELS.get(run['kind'], run['kind'])} → "
                f"{_KIND_LABELS.get(form['kind'], form['kind'])}", "kind")
        if run["order"] != form["order"]:
            add(f"{name} order {run['order']} → {form['order']}", "order")
    elif key == "notch":
        if not _same(run["lines"], form["lines"], rel):
            add(f"{name} lines {_lines_text(run['lines'])} → {_lines_text(form['lines'])}",
                "lines")
        if not _same(run["q"], form["q"], rel):
            add(f"{name} Q {_g(run['q'])} → {_g(form['q'])}", "q")
    elif key in ("median", "rmean"):
        if not _same(run["window_s"], form["window_s"], rel):
            add(f"{name} {_ms(run['window_s'])} → {_ms(form['window_s'])} ms", "window_s")
    elif key == "savgol":
        if not _same(run["window_s"], form["window_s"], rel):
            add(f"{name} {_ms(run['window_s'])} → {_ms(form['window_s'])} ms", "window_s")
        if run["polyorder"] != form["polyorder"]:
            add(f"{name} order {run['polyorder']} → {form['polyorder']}", "polyorder")
    elif key == "normalize":
        if run["mode"] != form["mode"]:
            add(f"{name} {_NORMALIZE_LABELS.get(run['mode'], run['mode'])} → "
                f"{_NORMALIZE_LABELS.get(form['mode'], form['mode'])}", "mode")
        elif not _same(run.get("clip_sigma"), form.get("clip_sigma"), rel):
            add(f"{name} clip {_g(run['clip_sigma'])} → {_g(form['clip_sigma'])}σ",
                "clip_sigma")
    elif key == "detrend":
        if run["method"] != form["method"]:
            add(f"{name} {DETREND_METHOD_LABELS.get(run['method'], run['method'])} → "
                f"{DETREND_METHOD_LABELS.get(form['method'], form['method'])}", "method")
        else:
            for knob, (label, fmt) in _DETREND_KNOB_TEXT.items():
                if knob in run and not _same(run.get(knob), form.get(knob), rel):
                    add(f"{name} {label} {fmt(run[knob])} → {fmt(form.get(knob))}", knob)
        if run.get("polarity") != form.get("polarity"):
            add(f"{name} polarity {_polarity_text(run.get('polarity'))} → "
                f"{_polarity_text(form.get('polarity'))}", "polarity")
    return out


def chain_differences(form: Mapping[str, Any] | None, run: Mapping[str, Any] | None, *,
                      fs: float | None = None, rel: float = 1e-9) -> tuple[ChainDiff, ...]:
    """Every way ``form`` differs from ``run``, in :data:`STAGE_KEYS` order."""
    a, b = effective_chain(run, fs), effective_chain(form, fs)
    diffs: list[ChainDiff] = []
    for key in STAGE_KEYS:
        name = _STAGE_PHRASE[key]
        if key in a and key not in b:
            diffs.append(ChainDiff(key, f"{name} on → off", "enabled"))
        elif key in b and key not in a:
            diffs.append(ChainDiff(key, f"{name} off → on ({_stage_summary(key, b[key])})",
                                   "enabled"))
        elif key in a:
            diffs.extend(_field_diffs(key, a[key], b[key], rel))
    return tuple(diffs)


def describe_differences(diffs: Sequence[ChainDiff], limit: int = 3) -> str:
    """``'low-pass 10000 → 5000 Hz; notch Q 30 → 20 (+1 more)'``."""
    shown = "; ".join(d.text for d in diffs[:limit])
    extra = len(diffs) - limit
    return f"{shown} (+{extra} more)" if extra > 0 else shown


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------
_CLASSICAL = frozenset({"detect.schmitt", "detect.cusum"})


@dataclass(frozen=True)
class ChainStatus:
    """How the form's chain relates to the chain of the detection on screen.

    ``kind``: ``none`` (no detection shown) | ``match`` | ``differs`` | ``unknown``
    (the run's chain was not recorded or could not be read) | ``not_classical`` (the
    table did not come from this chain at all).
    """

    kind: str
    sentence: str = ""
    diffs: tuple[ChainDiff, ...] = ()
    revert_params: dict[str, Any] | None = None
    detector_label: str = ""

    @property
    def headline(self) -> str:
        """The verdict for one elided line, differences first (the full :attr:`sentence`
        is for its tooltip)."""
        if self.kind == "differs" and self.diffs:
            label = self.detector_label or "detection"
            return f"⚠ Differs from the {label} run: {describe_differences(self.diffs)}"
        return self.sentence

    @classmethod
    def none(cls) -> "ChainStatus":
        return cls("none")

    @classmethod
    def unknown(cls, sentence: str = "") -> "ChainStatus":
        return cls("unknown", sentence or (
            "The chain the detection on screen ran with could not be read, so the form "
            "is not compared with it."))


def judge_chain(form_params: Mapping[str, Any] | None,
                run_params: Mapping[str, Any] | None, *,
                module_id: str | None, detector_label: str = "",
                fs: float | None = None) -> ChainStatus:
    """Compare the form's chain with the chain a stored detection ran with."""
    if not module_id:
        return ChainStatus.none()
    label = detector_label or str(module_id)
    if module_id == "detect.keras":
        return ChainStatus("not_classical", (
            "The event table on screen came from the ML detector, which runs its own "
            "model chain (ML tab) — this chain did not produce it."), detector_label=label)
    if module_id == "detect.reconciled":
        return ChainStatus("not_classical", (
            "The event table on screen is a reconciled table; compare the chain with its "
            "source detections."), detector_label=label)
    if module_id not in _CLASSICAL:
        return ChainStatus("not_classical", (
            f"The event table on screen ({label}) was not produced by the Schmitt/CUSUM "
            f"chain, so this chain is not compared with it."), detector_label=label)
    if not run_params:
        return ChainStatus("unknown", (
            f"The chain the {label} detection on screen ran with is not recorded, so the "
            f"form cannot be compared with it."), detector_label=label)
    diffs = chain_differences(form_params, run_params, fs=fs)
    if not diffs:
        return ChainStatus("match", (
            f"The form's chain is the one the {label} detection on screen ran with."),
            detector_label=label)
    revert = complete_recipe(run_params)
    if any(d.key == "polarity" for d in diffs):
        revert["polarity"] = run_params.get("polarity")
    sentence = (f"⚠ The form differs from the chain the {label} detection on screen ran "
                f"with: {describe_differences(diffs)}. Revert, or run detection to apply it.")
    return ChainStatus("differs", sentence, diffs, revert, label)


# ---------------------------------------------------------------------------
# the stage list
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageRow:
    """One row of the execution-order stage list."""

    key: str
    position: int
    title: str
    enabled: bool
    summary: str
    role: str
    linear: bool
    #: Extra detail for the tooltip (the low-pass's true −3 dB corner), or "".
    detail: str = ""

    @property
    def text(self) -> str:
        """The row label. The setting comes before the role note, so a right-elided row
        loses the note (also in the tooltip), never the cutoff or window."""
        role = f" ({self.role})" if self.role else ""
        return f"{self.position}  {self.title} — {self.summary}{role}"


def _lowpass_corner(st: Mapping[str, Any], fs: float) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, mag = filters.response("lowpass", float(fs), cutoff_hz=float(st["cutoff_hz"]),
                                      order=int(st.get("order", 8)), kind=str(st["kind"]))
        corner = filters.response_crossing_hz(f, mag)
    except Exception:  # noqa: BLE001 - a label, never a crash
        return ""
    if not corner:
        return ""
    return f"−3 dB {corner / 1e3:.1f} kHz" if corner >= 1e3 else f"−3 dB {corner:.0f} Hz"


def stage_rows(params: Mapping[str, Any] | None, fs: float | None = None, *,
               lowpass_design_hz: float | None = None) -> tuple[StageRow, ...]:
    """The eight stages in execution order, each with its CURRENT setting.

    A disabled stage still shows the setting it would run with; an off low-pass shows
    ``lowpass_design_hz`` (the cutoff box, which ``params()`` reports as None) if given.
    """
    p = complete_recipe(params)
    shown = dict(p, notch_enabled=True, median_enabled=True, hp_enabled=True,
                 detrend_enabled=True, rmean_enabled=True, savgol_enabled=True,
                 normalize_enabled=True)
    if shown["filter_cutoff_hz"] is None:
        shown["filter_cutoff_hz"] = float(lowpass_design_hz or 10_000.0)
    if params and "polarity" in params:
        shown["polarity"] = params["polarity"]
    full = effective_chain(shown, fs)
    live = effective_chain(params, fs)
    rows = []
    for i, key in enumerate(STAGE_KEYS, start=1):
        summary = _stage_summary(key, full[key])
        detail = ""
        if key == "lowpass" and p["filter_cutoff_hz"] is None and not lowpass_design_hz:
            summary = "off"
        elif key == "lowpass" and fs:
            detail = _lowpass_corner(full[key], fs)
        rows.append(StageRow(key, i, STAGE_TITLES[key], key in live, summary,
                             _STAGE_ROLE.get(key, ""), key in _LINEAR_STAGES, detail))
    return tuple(rows)


def recipe_key_differences(expected: Mapping[str, Any] | None,
                           got: Mapping[str, Any] | None, *, rel: float = 1e-9) -> list[str]:
    """``'hp_cutoff_hz 0.05 → 0.1'`` for every chain key whose value differs."""
    def show(v: Any) -> str:
        return f"{v:g}" if isinstance(v, float) else repr(v)

    a, b = complete_recipe(expected), complete_recipe(got)
    return [f"{k} {show(a[k])} → {show(b[k])}" for k in CHAIN_KEYS
            if not _same(a[k], b[k], rel)]


# ---------------------------------------------------------------------------
# validation and the protocol file
# ---------------------------------------------------------------------------
PROTOCOL_FILE_FORMAT = "nanoscope-conditioning-protocols"
PROTOCOL_FILE_VERSION = 1

_ALLOWED_KEYS = frozenset({*FILTER_PARAM_KEYS, *DETREND_PARAM_KEYS,
                           "filter_cutoff_hz", "filter_order", "filter_kind"})
_BOOL_KEYS = frozenset(k for k in CHAIN_KEYS if k.endswith("_enabled"))
#: Integer knobs and their widget ranges (a value outside is refused, never clamped).
_INT_RANGES: dict[str, tuple[int, int]] = {
    "filter_order": (1, 10), "hp_order": (1, 10), "notch_harmonics": (1, 20),
    "savgol_polyorder": (2, 6), "detrend_asls_niter": (1, 100),
    "detrend_decim": (0, 100_000), "detrend_sg_poly": (1, 5), "detrend_sg_decim": (1, 10_000),
}
#: Float knobs and their widget ranges. The three frequencies carry the boxes' STATIC
#: ranges (``DetectionPanel._STATIC_*_RANGE``); their recording-dependent Nyquist cap
#: cannot be checked here, so a recall that the form clamps is caught at recall time.
_FLOAT_RANGES: dict[str, tuple[float, float | None]] = {
    "filter_cutoff_hz": (1.0, 5_000_000.0), "hp_cutoff_hz": (0.1, 5_000_000.0),
    "notch_freq_hz": (1.0, 100_000.0),
    "notch_q": (1.0, 500.0),
    "median_window_s": (1e-5, 1.0), "rmean_window_s": (1e-5, 1.0),
    "savgol_window_s": (1e-4, 1.0), "normalize_clip_sigma": (0.1, 1000.0),
    "detrend_window_s": (1e-3, 60.0), "detrend_percentile": (0.1, 99.9),
    "detrend_asls_lam": (1.0, 1e9), "detrend_asls_p": (1e-5, 0.5),
    "detrend_asls_decim": (1e-3, 1.0), "detrend_rough_s": (1e-3, 60.0),
    "detrend_base_s": (1e-4, 60.0), "detrend_mask_sigma": (0.5, 20.0),
    "detrend_dilate_ms": (0.0, 1000.0),
}
_STR_CHOICES: dict[str, tuple[str, ...]] = {
    "filter_kind": tuple(filters.KINDS),
    "hp_kind": tuple(filters.HP_KINDS),
    "normalize_mode": tuple(filters.NORMALIZE_MODES_ALL),
    "detrend_method": tuple(m for m in baseline_mod.METHODS if m != "constant"),
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def validate_recipe(raw: Any, *, max_notch_lines: int = 32, ranges: bool = True
                    ) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """``(complete recipe, ())`` for a valid protocol, ``(None, reasons)`` otherwise.

    Every key must be a signal-chain key (FILTER_PARAM_KEYS / DETREND_PARAM_KEYS and
    the low-pass triple); an unknown key is refused BY NAME, never dropped. Values are
    checked against the kernels' choices and the form's ranges, so nothing an import
    applies is silently clamped. ``ranges=False`` checks types and choices only (the
    user's own stored protocols: a recall the form clamps is reported at recall time).
    """
    if not isinstance(raw, Mapping):
        return None, ("not a key/value mapping",)
    problems: list[str] = []
    unknown = sorted(str(k) for k in raw if k not in _ALLOWED_KEYS)
    if unknown:
        problems.append(
            "unknown key(s) " + ", ".join(unknown) + " — a conditioning protocol may only "
            "name signal-chain keys (FILTER_PARAM_KEYS / DETREND_PARAM_KEYS and the "
            "low-pass filter_cutoff_hz / filter_order / filter_kind)")
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _ALLOWED_KEYS:
            continue
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                problems.append(f"{key} must be true or false, got {value!r}")
                continue
            clean[key] = value
        elif key in _INT_RANGES:
            num = _number(value)
            lo, hi = _INT_RANGES[key]
            if num is None or num != int(num) or (ranges and not lo <= num <= hi):
                problems.append(f"{key} must be an integer in [{lo}, {hi}], got {value!r}")
                continue
            clean[key] = int(num)
        elif key in _FLOAT_RANGES:
            if key == "filter_cutoff_hz" and value is None:
                clean[key] = None
                continue
            num = _number(value)
            lo, hi = _FLOAT_RANGES[key]
            bad = num is None or (ranges and (num < lo or (hi is not None and num > hi)))
            if bad:
                rng = f"> {lo:g}" if hi is None else f"in [{lo:g}, {hi:g}]"
                problems.append(f"{key} must be a finite number {rng}, got {value!r}")
                continue
            clean[key] = num
        elif key in _STR_CHOICES:
            if value not in _STR_CHOICES[key]:
                problems.append(f"{key} must be one of {', '.join(_STR_CHOICES[key])}, "
                                f"got {value!r}")
                continue
            clean[key] = str(value)
        elif key == "notch_freqs":
            if value is None:
                clean[key] = None
                continue
            if not isinstance(value, (list, tuple)) or not value:
                problems.append("notch_freqs must be null or a non-empty list of frequencies")
                continue
            nums = [_number(v) for v in value]
            if any(n is None or n <= 0 for n in nums):
                problems.append("notch_freqs must hold only positive, finite frequencies")
                continue
            if len(nums) > max_notch_lines:
                problems.append(f"notch_freqs names {len(nums)} lines; at most "
                                f"{max_notch_lines} are allowed")
                continue
            if len(set(nums)) != len(nums):
                problems.append("notch_freqs repeats a line")
                continue
            clean[key] = [float(n) for n in nums]
    if problems:
        return None, tuple(problems)
    return complete_recipe(clean), ()


def write_protocols_file(path: str | Path, protocols: Mapping[str, Mapping[str, Any]], *,
                         app_version: str = "") -> Path:
    """Write ``protocols`` (name -> recipe) as a versioned JSON exchange file."""
    doc = {
        "format": PROTOCOL_FILE_FORMAT,
        "version": PROTOCOL_FILE_VERSION,
        "app_version": str(app_version),
        "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocols": {str(name): complete_recipe(recipe)
                      for name, recipe in protocols.items()},
    }
    target = Path(path)
    with staged_path(target) as staged:
        staged.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return target


def read_protocols_file(path: str | Path, *, max_notch_lines: int = 32
                        ) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """``({name: recipe}, refusals)`` from a protocol exchange file.

    Raises ``ValueError`` when the file is not one (wrong format, a version newer than
    this build reads, or not JSON). A protocol that fails :func:`validate_recipe` is
    left out and named in the refusals.
    """
    name = Path(path).name
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not a NanoScope conditioning-protocol file "
                         f"(it could not be read as JSON: {exc}).") from exc
    if not isinstance(doc, Mapping) or doc.get("format") != PROTOCOL_FILE_FORMAT:
        raise ValueError(f"{name} is not a NanoScope conditioning-protocol file "
                         f"(its \"format\" is not \"{PROTOCOL_FILE_FORMAT}\").")
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{name}: the protocol-file version {version!r} is not valid.")
    if version > PROTOCOL_FILE_VERSION:
        raise ValueError(f"{name} was written in protocol-file version {version}; this "
                         f"build reads version {PROTOCOL_FILE_VERSION}. Update NanoScope "
                         f"to import it.")
    raw_protocols = doc.get("protocols")
    if not isinstance(raw_protocols, Mapping):
        raise ValueError(f"{name} holds no \"protocols\" table.")
    out: dict[str, dict[str, Any]] = {}
    refused: list[str] = []
    for pname, raw in raw_protocols.items():
        pname = str(pname).strip()
        if not pname or pname == MANUAL_PROTOCOL:
            refused.append(f"a protocol named {pname!r} (reserved or empty name)")
            continue
        recipe, why = validate_recipe(raw, max_notch_lines=max_notch_lines)
        if recipe is None:
            refused.append(f"“{pname}”: " + "; ".join(why))
            continue
        out[pname] = recipe
    return out, tuple(refused)


@dataclass
class MergeReport:
    """What an import did: names added, renamed (``old -> new``), skipped, refused."""

    added: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    identical: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    def sentence(self) -> str:
        n = len(self.added) + len(self.renamed)
        noun = "protocol" if n == 1 else "protocols"
        parts = []
        if self.renamed:
            parts.append(f"{len(self.renamed)} renamed")
        if self.identical:
            parts.append(f"{len(self.identical)} already present")
        text = f"Imported {n} {noun}" + (f" ({', '.join(parts)})" if parts else "") + "."
        if self.refused:
            text += (f" Refused {len(self.refused)}: " + " | ".join(self.refused))
        return text


def merge_imported_protocols(existing: Mapping[str, Mapping[str, Any]],
                             imported: Mapping[str, Mapping[str, Any]], *,
                             reserved: Mapping[str, Mapping[str, Any] | None] | Sequence[str] = (),
                             refused: Sequence[str] = ()) -> tuple[dict[str, dict[str, Any]], MergeReport]:
    """``existing`` plus ``imported``, NEVER overwriting.

    A name clash (with a stored protocol or a reserved/shipped name) imports as
    ``'<name> (imported)'`` (then ``(imported 2)``…); a recipe identical to the one
    already under that name is skipped.
    """
    merged = {k: complete_recipe(v) for k, v in existing.items()}
    reserved_map: dict[str, Mapping[str, Any] | None] = (
        dict(reserved) if isinstance(reserved, Mapping) else {r: None for r in reserved})
    report = MergeReport(refused=list(refused))
    for name, recipe in imported.items():
        recipe = complete_recipe(recipe)
        taken = name in merged or name in reserved_map
        if taken:
            current = merged.get(name, reserved_map.get(name))
            if current is not None and complete_recipe(current) == recipe:
                report.identical.append(name)
                continue
            n = 1
            new = f"{name} (imported)"
            while new in merged or new in reserved_map:
                n += 1
                new = f"{name} (imported {n})"
            merged[new] = recipe
            report.renamed.append((name, new))
        else:
            merged[name] = recipe
            report.added.append(name)
    return merged, report


# ---------------------------------------------------------------------------
# Copy as Python
# ---------------------------------------------------------------------------
def python_script(params: Mapping[str, Any], *, path: str | None, channel_index: int,
                  channel_name: str = "", units: str = "", fs: float | None = None,
                  cal_gain: float = 1.0, cal_offset: float = 0.0,
                  loaded_range: tuple[int, int] | None = None,
                  segment: tuple[int, int] | None = None, t0_s: float = 0.0,
                  window: tuple[int, int] | None = None, app_version: str = "",
                  polarity: int | None = None, path_note: str = "") -> str:
    """A standalone script that rebuilds the processed trace NanoScope shows.

    It reads the recording with the app's own reader (``open_document``), applies the
    channel calibration exactly as ``ChannelCalibration.apply`` does, conditions the
    WHOLE channel with :func:`~nanoscope.analysis.conditioning.conditioned_trace` and only
    then slices the window. ``segment`` is a derived segment's ``[i0, i1)`` in its parent
    (a segment is conditioned on its own samples, so it is cut BEFORE conditioning).
    """
    chain = complete_recipe(params)
    if polarity is not None and chain.get("detrend_enabled"):
        chain["polarity"] = int(polarity)
    body = pprint.pformat(chain, sort_dicts=False, width=88)
    body = "\n".join(("PARAMS = " if i == 0 else "         ") + line
                     for i, line in enumerate(body.splitlines()))
    description = chain_description(chain) or "no stage enabled (the raw calibrated signal)"
    hp = bool(chain.get("hp_enabled"))
    lines = [
        '"""Rebuild the processed trace NanoScope shows in Signal Conditioning.',
        "",
        f"Generated by NanoScope {app_version or '(unknown version)'} on "
        f"{datetime.now(timezone.utc).date().isoformat()}.",
        f"Chain: {description}",
        '"""',
        "import numpy as np",
        "",
        "from nanoscope.analysis.conditioning import characterization_trace, conditioned_trace",
        "from nanoscope.core.types import SampleRange",
        "from nanoscope.io.base import open_document",
        "",
    ]
    if path is None:
        lines.append("PATH = None  # set this to the recording" + (f" ({path_note})" if path_note else ""))
    else:
        lines.append(f"PATH = {str(path)!r}")
    lr = "None" if loaded_range is None else f"SampleRange({int(loaded_range[0])}, {int(loaded_range[1])})"
    lines += [
        f"LOADED_RANGE = {lr}  # the part of the file NanoScope loaded (None = whole file)",
        f"CHANNEL = {int(channel_index)}  # {channel_name or 'channel'} ({units or 'units unknown'})",
        f"GAIN = {float(cal_gain)!r}  # Calibration panel: calibrated = raw * gain + offset",
        f"OFFSET = {float(cal_offset)!r}",
        f"FS = {None if fs is None else float(fs)!r}  # Hz",
        f"T0_S = {float(t0_s)!r}  # instrument-clock time of the first loaded sample",
    ]
    if segment is not None:
        lines.append(f"SEGMENT = ({int(segment[0])}, {int(segment[1])})  "
                     f"# this derived segment's samples in its parent")
    lines.append(body)
    lines += [
        "",
        "if PATH is None:",
        "    raise SystemExit('Set PATH to the recording this chain was built on.')",
        "doc = open_document(PATH, LOADED_RANGE)",
        "raw = doc.raw(CHANNEL)",
    ]
    if segment is not None:
        lines.append("raw = np.ascontiguousarray(raw[SEGMENT[0]:SEGMENT[1]])")
    lines += [
        "if (GAIN, OFFSET) != (1.0, 0.0):",
        "    raw = raw * np.float32(GAIN) + np.float32(OFFSET)",
        "fs = doc.meta.sample_rate_hz",
        "if FS is not None and fs != FS:",
        "    raise SystemExit(f'expected {FS} Hz, the file says {fs} Hz')",
        "",
        "# Condition the WHOLE channel, then slice: stages such as the baseline estimator",
        "# and normalize use whole-recording statistics, so a slice conditioned on its own",
        "# is a different array.",
        "processed = conditioned_trace(raw, fs, PARAMS)",
    ]
    if hp:
        lines += [
            "# High-pass is on: events are LOCALIZED on `processed`, and their amplitudes are",
            "# measured on this companion trace with only the high-pass omitted.",
            "measurement = characterization_trace(raw, fs, PARAMS)",
        ]
    start, stop = (0, None) if window is None else (int(window[0]), int(window[1]))
    lines += [
        "",
        "t = T0_S + np.arange(raw.size) / fs  # seconds, instrument clock (the trace's x axis)",
        f"START, STOP = {start}, {stop if stop is not None else 'raw.size'}  "
        f"# the Signal Conditioning window's view, in samples",
        "t_view, processed_view = t[START:STOP], processed[START:STOP]",
    ]
    return "\n".join(lines) + "\n"
