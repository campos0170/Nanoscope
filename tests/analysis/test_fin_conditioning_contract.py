"""FIN-H5 / FIN-NR-3: the conditioner's own two claims about itself.

Two residues from the 2026-09-02 wiring round, both about ``analysis/conditioning.py``
describing an arrangement other than the one it is in:

  - H5, the FOSSIL. The module docstring said ``AnalysisPanel._conditioned_from_form``
    was "STILL a second copy" of :func:`conditioned_trace` and named the delegation as
    queued work. The delegation landed the same day — the panel's whole body is now
    ``return conditioned_trace(raw, fs, params)`` — so the paragraph was describing last
    month's codebase to the next reader of the definition. A prose test is worth having
    for exactly this class of sentence: it is the one part of a module nothing else
    checks, and it is what a maintainer reads BEFORE deciding where to put a new stage.

  - NR-3, the WIDE DICT. The Measure panel's new fallback chain is a stored detection's
    ``AnalysisResult.params`` — the whole form, thresholds and dwell bounds included —
    handed to the conditioner to pick from. The arithmetic survives it (the chain reads
    only conditioning keys), but the sidecar did not: ``params`` is stamped as the
    machine-readable recipe for rebuilding the exported samples, and it was recording
    ``k_in``/``min_dwell_s``/``baseline_*`` as if they had filtered the trace. The module
    now declares :data:`CONDITIONING_PARAM_KEYS`, narrows with
    :func:`conditioning_params` at the stamping door, and the census below holds that set
    against the kernels' own source so narrowing can never drop a key the chain reads.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

import numpy as np
import pytest

from nanoscope.analysis import conditioning, modules
from nanoscope.analysis.conditioning import (
    CONDITIONING_PARAM_KEYS,
    chain_description,
    conditioned_trace,
    conditioning_params,
    processed_trace_provenance,
    write_processed_trace,
)

FS = 50_000.0
N = 60_000

#: A live Detection form, near enough: the chain the owner runs, plus everything else the
#: form carries. The second half is what must never reach the recipe — and, if a stale key
#: ever could steer a filter, what would steer it.
WIDE_FORM: dict = {
    # --- the chain ---------------------------------------------------------------
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0, "notch_harmonics": 3,
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.05,
    "savgol_enabled": True, "savgol_window_s": 0.001, "savgol_polyorder": 3,
    "filter_cutoff_hz": 5000.0, "filter_order": 8, "filter_kind": "bessel",
    "polarity": -1,
    # --- everything else on the form -------------------------------------------
    "k_in": 5.0, "k_out": 3.0, "cusum_k": 0.5, "cusum_h": 28.0,
    "min_dwell_s": 5e-5, "max_dwell_s": 0.01, "merge_gap_s": 1e-4,
    "detector": "detect.schmitt", "pore_id": "22 nm", "recording": "apo-1000mV",
    "units": "pA",
    "baseline_rough_s": 0.5, "baseline_base_s": 0.1, "baseline_param_source": "form",
    "_calibration_fingerprint": (1.0, 0.0),
}

#: The keys above that are NOT the chain. Spelled out rather than derived, so this test
#: fails if the module ever widens its set to swallow one of them.
NON_CHAIN_KEYS = frozenset({
    "k_in", "k_out", "cusum_k", "cusum_h", "min_dwell_s", "max_dwell_s", "merge_gap_s",
    "detector", "pore_id", "recording", "units",
    "baseline_rough_s", "baseline_base_s", "baseline_param_source",
    "_calibration_fingerprint",
})


def _trace(seed: int = 11) -> np.ndarray:
    """60 k float32 at 50 kHz: open pore, mains hum, slow drift, one blockade."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / FS
    x = (100.0
         + 1.5 * rng.normal(size=N)
         + 2.0 * np.sin(2 * np.pi * 60.0 * t)
         + 3.0 * np.sin(2 * np.pi * 0.4 * t))
    x[25_000:27_000] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ================================================================================================
# H5 — the module says what it IS, not what it was queued to become
# ================================================================================================
#: The exact claims the docstring made before the delegation landed. Substrings, because
#: what is being caught is the SENTENCE, and a fossil that survives a reword is still a
#: fossil pointing the next reader at a second copy that no longer exists.
_FOSSIL_CLAIMS = (
    "STILL a second copy",
    "the delegation is queued",
    "queued one-line change",
    "Three functions in the",          # ...package apply the two-step chain
    "and is the duplicate",
)


@pytest.mark.parametrize("claim", _FOSSIL_CLAIMS)
def test_the_module_docstring_no_longer_describes_the_panel_as_a_second_copy(claim):
    """The paragraph the wiring round flagged, held to the tree it is actually in.

    Each of these was true on 2026-09-01 and is false now: the panel's body is one call,
    the census is two builders and not three, and nothing about the delegation is queued.
    """
    doc = (conditioning.__doc__ or "") + (conditioned_trace.__doc__ or "")
    assert claim not in doc, (
        f"conditioning.py still tells the reader {claim!r}. The delegation landed on "
        "2026-09-02: AnalysisPanel._conditioned_from_form is one line now. A module "
        "docstring that describes the previous arrangement sends the next author to "
        "keep a copy in sync that does not exist.")


def test_the_module_docstring_describes_the_delegation_that_actually_landed():
    """Not merely the absence of the old claim — the presence of the current one.

    Deleting the paragraph would pass the test above and leave the definition unexplained;
    what the reader needs is the reason the panel's method still exists (it is the GUI's
    own door: ``analysis/`` may not be imported from a panel) beside the fact that it no
    longer builds anything.
    """
    doc = conditioning.__doc__ or ""
    assert "return conditioned_trace(raw, fs, params)" in doc, \
        "the docstring should quote the panel's whole body, which is one call"
    assert "TWO functions in the package apply" in doc, \
        "the census is two builders now — the live form and a detection's provenance"
    # The panel's own delegation note and the single-writer test both key off this: the
    # body was COPIED and later delegated, never moved.
    assert "verbatim move" not in doc


# ================================================================================================
# NR-3 — the chain's key set, and the narrowing that uses it
# ================================================================================================
def _keys_read_from_params() -> set[str]:
    """Every params key the conditioning chain reads, scraped from the kernels' source.

    The census is source-level and deliberately dumb: ``params.get("x")`` /
    ``params["x"]`` inside the functions :func:`conditioned_trace` actually reaches. A
    narrowing function is only safe if it is complete, and "complete" has to be measured
    against the code that does the reading rather than against a second hand-written list.
    """
    fns = (modules.preprocess_for_detection, modules._apply_normalize,
           modules._apply_smoothers, modules.notch_freqs_param,
           conditioning.conditioned_trace)
    found: set[str] = set()
    for fn in fns:
        src = ast.unparse(ast.parse(_dedent_source(fn)))
        found |= set(re.findall(r'params\.get\(\s*[\'"]([A-Za-z_][A-Za-z_0-9]*)[\'"]', src))
        found |= set(re.findall(r'params\[\s*[\'"]([A-Za-z_][A-Za-z_0-9]*)[\'"]\s*\]', src))
    # `detrend_method_kwargs` reads its keys through a variable, from this table.
    for spec in modules.DETREND_METHOD_PARAMS.values():
        found |= {param_key for param_key, _kernel, _default in spec}
    return found


def _dedent_source(fn) -> str:
    import inspect
    import textwrap
    return textwrap.dedent(inspect.getsource(fn))


def test_the_key_set_covers_every_key_the_chain_actually_reads():
    """Narrowing may not lose a stage. This is the guard that says it cannot.

    If ``preprocess_for_detection`` grows a knob and nobody adds it to
    ``FILTER_PARAM_KEYS``/``DETREND_PARAM_KEYS``, ``conditioning_params`` would silently
    strip it and the export sidecar would stop being a recipe — the exact defect the
    narrowing exists to prevent, one stage later.
    """
    missing = _keys_read_from_params() - set(CONDITIONING_PARAM_KEYS)
    assert not missing, (
        f"the chain reads {sorted(missing)}, which conditioning_params would strip. "
        "Add them to the kernels' own DETREND_PARAM_KEYS / FILTER_PARAM_KEYS "
        "(CONDITIONING_PARAM_KEYS is assembled from those), never to a second list.")
    # And the set is not a synonym for "everything": the detector's own knobs are out.
    assert not (NON_CHAIN_KEYS & set(CONDITIONING_PARAM_KEYS))


def test_narrowing_keeps_the_chain_and_drops_the_detection_form():
    """``conditioning_params`` is the door: chain in, form out."""
    narrowed = conditioning_params(WIDE_FORM)
    assert set(narrowed) & NON_CHAIN_KEYS == set(), \
        f"detection settings survived the narrowing: {sorted(set(narrowed) & NON_CHAIN_KEYS)}"
    for key in ("notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q",
                "detrend_enabled", "detrend_method", "detrend_window_s",
                "savgol_enabled", "savgol_window_s", "savgol_polyorder",
                "filter_cutoff_hz", "filter_order", "filter_kind", "polarity"):
        assert narrowed[key] == WIDE_FORM[key], f"the chain lost {key}"
    assert conditioning_params(None) == {} and conditioning_params({}) == {}
    # A door, not a mutator: the caller's dict is untouched and a new one comes back.
    narrowed["notch_enabled"] = False
    assert WIDE_FORM["notch_enabled"] is True


def test_the_narrowed_chain_is_the_same_array_to_the_byte():
    """The whole justification for narrowing the RECORD is that it does not move the ARRAY.

    Both directions matter: identical bytes means the sidecar stays a recipe, and it also
    demonstrates the claim the docstring makes — that the detection keys were inert on
    this path all along, so nothing that a user measured has changed.
    """
    raw = _trace()
    wide = conditioned_trace(raw, FS, WIDE_FORM)
    narrow = conditioned_trace(raw, FS, conditioning_params(WIDE_FORM))
    assert _digest(wide) == _digest(narrow)
    assert wide.dtype == narrow.dtype == np.float32
    assert chain_description(WIDE_FORM) == chain_description(conditioning_params(WIDE_FORM))


def test_the_export_sidecar_records_the_chain_and_not_the_detection_form(tmp_path):
    """The recipe stamp, which is where the wide dict was actually doing harm.

    ``params`` in the sidecar is documented as the dict a reader feeds back to
    ``conditioned_trace`` to rebuild these samples. Listing the Schmitt thresholds beside
    the filter stages makes it a description of a detection RUN instead, and a reader has
    no way to tell which half of it conditioned the array.
    """
    raw = _trace()
    data = conditioned_trace(raw, FS, WIDE_FORM)
    out = write_processed_trace(tmp_path / "trace.csv", data, FS, units="pA",
                                params=WIDE_FORM, cal_gain=1.0, cal_offset=0.0)
    meta = json.loads(out["meta"].read_text(encoding="utf-8"))

    leaked = NON_CHAIN_KEYS & set(meta["params"])
    assert not leaked, (
        f"the sidecar's recipe carries {sorted(leaked)} — detection settings that never "
        "touched these samples. It is stamped as the dict to feed back to "
        "conditioned_trace, so it must be the chain and nothing else.")
    assert meta["params"]["filter_cutoff_hz"] == 5000.0
    assert meta["chain"] == chain_description(WIDE_FORM)
    # Still a recipe: the narrowed dict rebuilds the exported samples bit for bit.
    assert _digest(conditioned_trace(raw, meta["sample_rate_hz"], meta["params"])) == \
        _digest(data)


def test_the_provenance_block_narrows_even_without_a_file():
    """The same rule one layer down, so a caller that only wants the block gets it too."""
    prov = processed_trace_provenance(
        params=WIDE_FORM, fs=FS, n_samples=10, start_idx=0, units="pA")
    assert not (NON_CHAIN_KEYS & set(prov["params"]))
    assert prov["params"]["notch_freq_hz"] == 60.0
    # `polarity` is a chain input (the detrend stage reads it) even though the detector
    # reads it too — dropping it would change the array it claims to rebuild.
    assert prov["params"]["polarity"] == -1
