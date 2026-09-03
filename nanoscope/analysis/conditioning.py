"""The CONDITIONED trace: the array the user is looking at, as a pure function.

Every surface in this app that shows or measures "the processed signal" has to mean the
same array, sample for sample. Before this module existed exactly one place could build
it — ``AnalysisPanel._conditioned_from_form``, a private staticmethod on a GUI panel —
and its own docstring says why it exists: *"ONE function, because two callers need the
identical array"*. That worked while the two callers were both inside that panel (the
live preview and the baseline probe). It stops working the moment a THIRD surface needs
it, because a panel in this codebase may not reach for a sibling panel (there are AST
guard tests that enforce it) and ``analysis/`` may not import ``gui/`` at all. The
measured consequence, from the 2026-09-02 audit: with the orange processed curve on
screen, the Manual measurement tool was still measuring ``session.calibrated(...).data``
— the RAW channel — and there was no export of the processed trace anywhere (zero
``to_csv``/``savetxt``/``np.save`` calls in the whole ``gui`` package). A user could
filter, detrend and normalize, watch it happen, and then neither measure on it nor get
the numbers out.

So the array moves here, to a pure module with no Qt in it:

* :func:`conditioned_trace` is THE DEFINITION of the conditioned trace — the function
  ``analysis/``, the Measure tool and the export all call, and since 2026-09-02 the only
  BODY of it in the codebase. ``AnalysisPanel._conditioned_from_form`` used to be a
  second copy of the same two calls; its whole body is now
  ``return conditioned_trace(raw, fs, params)``. The method itself stays, because it is
  the panel's own door — a GUI class may not be imported by ``analysis/``, and the two
  in-panel callers (the live preview and the baseline probe) ask for the array by that
  name — but it is no longer a second ANSWER. The delegation was landed on evidence
  rather than by inspection: while both bodies still existed they hashed identically
  (``sha256`` over the arrays' bytes) on the owner's own -1000 mV apo recording
  (749,651 samples, ch2) under five chains — no stage, 5 kHz Bessel low-pass, notch
  60 Hz + harmonics, detrend + low-pass, and the full notch+detrend+low-pass chain
  (``tests/analysis/test_pc_conditioning.py::test_conditioned_trace_is_bit_identical_to_the_panels_own_chain``)
  — and over a randomized 24-chain matrix in
  ``tests/analysis/test_pc2_conditioning_single_writer.py``, which sampled the stages the
  five hand-written chains never touched. Those two files now pin the ARRANGEMENT rather
  than the copy: an AST census that fails if a third body of the chain appears anywhere
  in the package, and a source check that fails if the panel's method grows a body of its
  own again.

  ONE BUILDER PER INPUT is the rule, not one builder full stop.
  TWO functions in the package apply ``lowpass(preprocess_for_detection(...))`` and both
  should: this one, from the LIVE form's params (what is on screen), and
  :func:`~nanoscope.analysis.modules.detection_signal`, from a finished detection's
  recorded PROVENANCE (what the reported ``depth_pA`` were read off) — different inputs,
  different tolerance for a legacy field, genuinely two functions. The census pins that
  pair exactly, so a THIRD builder — the seat ``_conditioned_from_form`` used to occupy —
  is a failure whoever writes it.

  WHAT COUNTS AS ``params`` is deliberately narrow, and :data:`CONDITIONING_PARAM_KEYS`
  says which keys those are: callers arrive holding wider dicts (the whole Detection
  form, or a stored run's ``AnalysisResult.params`` recovered as the Measure panel's
  fallback chain), and :func:`conditioning_params` is the door that narrows one to the
  chain. See its docstring for why the wide dict is inert in the arithmetic and not in
  the record.
* :func:`chain_description` is the one sentence that names the chain, so the panel that
  measures on it, the row that records it and the export sidecar cannot describe the
  same array three different ways.
* :func:`write_processed_trace` is the export: samples, time base, units, and the
  provenance needed to rebuild the array from the recording on disk — which is the chain
  AND the calibration it ran on, because the chain runs on the CALIBRATED channel and a
  gain of 10 makes a different array out of the same file and the same params.

WHAT "PROCESSED" MEANS HERE — and what it does NOT mean
-------------------------------------------------------
``lowpass(preprocess_for_detection(raw, fs, params), fs, cutoff)``: the pre-detection
stages (notch / median / high-pass / baseline-drift / running-mean / SavGol / normalize,
each gated by its own ``*_enabled`` flag) and THEN the low-pass, which
:func:`~nanoscope.analysis.modules.preprocess_for_detection` deliberately leaves out
because ``detect_events`` owns it. That is what the preview curve draws, so it is what
"measure what is on screen" has to mean.

It is NOT the open-pore baseline estimator. Two different things on the Preprocess tab
wear the word "baseline" — the characterisation baseline every ``depth_pA`` is measured
against, and the ``Baseline drift`` detrend stage — and only the second one is in this
chain. See the audit's note 4; nothing here changes which is which.

THE BASIS STAMP (why a measured row carries :data:`SIGNAL_BASIS_COLUMN`)
------------------------------------------------------------------------
A measurement whose basis is ambiguous is worse than one that is merely raw: 30 pA
measured on the raw channel and 30 pA measured after a 5 kHz Bessel + normalize are
different numbers about different arrays, and pooled in one table with no column to tell
them apart they are permanently indistinguishable. So the basis is a STAMP on the row,
in the same spirit as the ``units``/``cal_gain``/``cal_offset``/``t0_s`` stamps
:mod:`nanoscope.analysis.measure` already carries.

It is declared HERE, outside :data:`~nanoscope.analysis.measure.MEASUREMENT_SCHEMA`, the
way ``scoring.ADDED_COLUMN`` is declared outside the frozen event ``SCHEMA``: an old
project that has never heard of it still loads (:func:`ensure_signal_basis_column` fills
``''`` = "not recorded", which reads as unknown and never as "raw"), and no existing
export changes shape. ``''`` is the honest default for every row saved before the column
existed — those rows WERE raw, but they were saved by a build in which no other basis
was reachable, and inventing a stamp nobody wrote is how a record stops being a record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.atomic_write import staged_path
from . import filters
from .modules import (
    DETREND_PARAM_KEYS,
    FILTER_PARAM_KEYS,
    any_preprocess_enabled,
    notch_freqs_param,
    preprocess_for_detection,
)

__all__ = [
    "BASIS_PROCESSED",
    "BASIS_RAW",
    "BASIS_UNKNOWN",
    "CONDITIONING_PARAM_KEYS",
    "SIGNAL_BASIS_COLUMN",
    "SIGNAL_BASIS_TIP",
    "basis_label",
    "chain_description",
    "conditioned_trace",
    "conditioning_enabled",
    "conditioning_params",
    "ensure_signal_basis_column",
    "processed_trace_export_paths",
    "processed_trace_provenance",
    "write_processed_trace",
]


# ---------------------------------------------------------------------------
# the array
# ---------------------------------------------------------------------------
def conditioned_trace(
    raw: np.ndarray, fs: float, params: Mapping[str, Any]
) -> np.ndarray:
    """``raw`` as the detector — and the preview curve — see it, from ``params``.

    ``lowpass(preprocess_for_detection(raw, fs, params))``: the same two steps, in the
    same order, with the form's own low-pass DESIGN rather than the kernel defaults.
    This is the definition, and since 2026-09-02 the only body of it:
    ``AnalysisPanel._conditioned_from_form`` — the preview's and the baseline probe's own
    door into this function — delegates here in one line. It was a bit-identical copy
    before that, held to this function by hash tests on the owner's real recording,
    because a preview the user is looking at and a measurement taken "on what is on
    screen" that differ by one filter design are two traces nobody can tell apart by eye.

    ``params`` MAY BE WIDER THAN THE CHAIN and the extra keys are inert: a caller often
    holds the whole Detection form, or a stored run's ``AnalysisResult.params``, and
    every key read on this path is a conditioning key (:data:`CONDITIONING_PARAM_KEYS`),
    so no threshold or dwell bound can reach a filter. What a wide dict changes is what
    gets WRITTEN DOWN, which is why :func:`processed_trace_provenance` stamps
    :func:`conditioning_params` of it rather than the dict it was handed.

    ``fs`` is required and must be a real rate: every stage is designed in Hz/seconds.
    (``preprocess_for_detection`` tolerates ``fs=None`` by applying only the normalize
    stage — the callers here refuse instead, out loud, exactly as the preview does.)

    Raises whatever the kernels raise — a cutoff above Nyquist, a detrend window longer
    than the trace. Every caller turns that into a visible message rather than a silently
    unconditioned array.
    """
    processed = preprocess_for_detection(raw, fs, params)
    cutoff = params.get("filter_cutoff_hz")
    if cutoff is None:
        return np.asarray(processed)
    return np.asarray(filters.lowpass(
        processed, fs, float(cutoff),
        order=int(params.get("filter_order", 8)),
        kind=str(params.get("filter_kind", "bessel")),
    ))


def conditioning_enabled(params: Mapping[str, Any] | None) -> bool:
    """True when :func:`conditioned_trace` would return something other than ``raw``.

    The low-pass OR any pre-detection stage — the same two-part test the preview uses to
    decide whether there is anything to draw (``AnalysisPanel._update_filter_preview``:
    ``params.get("filter_cutoff_hz") is None and not any_preprocess_enabled(params)`` is
    its "nothing on" case). One writer for the question "is there a processed trace at
    all?", so the Measure tool cannot offer a basis the plot has nothing to show for.
    """
    if not params:
        return False
    return (params.get("filter_cutoff_hz") is not None
            or any_preprocess_enabled(params))


#: EVERY ``params`` key the chain can read, and no other.
#:
#: Assembled from the tuples the READING code publishes — ``DETREND_PARAM_KEYS`` and
#: ``FILTER_PARAM_KEYS`` are :mod:`~nanoscope.analysis.modules`' own declaration of what
#: :func:`~nanoscope.analysis.modules.preprocess_for_detection` consumes — rather than
#: re-spelled here, because a second hand-written list of forty key names is a list that
#: goes stale the first time a stage grows a knob. The two additions are the keys that
#: are read on THIS side of the boundary: the low-pass triple, which
#: :func:`conditioned_trace` applies itself (``preprocess_for_detection`` deliberately
#: leaves it to ``detect_events``), and ``polarity``, which the detrend stage reads and
#: which is therefore a chain input even though it is also a detector parameter.
CONDITIONING_PARAM_KEYS: frozenset[str] = frozenset({
    *DETREND_PARAM_KEYS,
    *FILTER_PARAM_KEYS,
    "filter_cutoff_hz", "filter_order", "filter_kind",
    "polarity",
})


def conditioning_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """``params`` narrowed to the chain: :data:`CONDITIONING_PARAM_KEYS` and nothing else.

    THE DOOR FOR A CALLER HOLDING A WIDER DICT. Two of them exist and both are real: the
    Detection form's ``params()`` carries both detectors' thresholds, the dwell bounds and
    the whole ``baseline_`` namespace beside the filter stages, and the Measure panel's
    fallback chain is a stored ``AnalysisResult.params`` — the same form as it stood when
    that run was launched. Handing either straight to :func:`conditioned_trace` works, and
    that is the trap: it works because of a coincidence rather than a contract.

    The arithmetic is safe. Every key the chain reads is in this set by construction (a
    census test pins it against the kernels' own source), so a ``k_in`` or a ``min_dwell_s``
    is never looked at and cannot move a filter — today. It stays true only for as long as
    no detector key is ever spelled like a conditioning key, which is nobody's invariant:
    ``polarity`` is already read by both sides, and it is only by luck that the two mean
    the same thing.

    The RECORD is not safe, and that is the part this fixes. ``params`` is stamped into the
    export sidecar as the machine-readable recipe — the dict a reader feeds back to
    :func:`conditioned_trace` to rebuild the samples — and a recipe that also lists the
    Schmitt thresholds the trace was never filtered by describes a run, not an array. So
    :func:`processed_trace_provenance` narrows before it stamps, and a caller that wants
    the same discipline at its own door (the fallback chain, a saved protocol) calls this.

    Returns a new plain ``dict``; ``None`` and ``{}`` both give ``{}``. Order is the
    caller's, not this set's, because the value is a params dict and not a sentence.
    """
    if not params:
        return {}
    return {k: v for k, v in params.items() if k in CONDITIONING_PARAM_KEYS}


#: ``params`` flag -> how that stage is spelled in :func:`chain_description`, in the
#: order :func:`~nanoscope.analysis.modules.preprocess_for_detection` applies them. The
#: order is the chain's, not alphabetical, because the description is meant to be read
#: as the recipe: "notch 60 Hz → detrend 0.05 s → low-pass" is reproducible and
#: "detrend, low-pass, notch" is a different trace.
_STAGE_SPELLING: tuple[tuple[str, str], ...] = (
    ("notch_enabled", "notch"),
    ("median_enabled", "median"),
    ("hp_enabled", "high-pass"),
    ("detrend_enabled", "detrend"),
    ("rmean_enabled", "running-mean"),
    ("savgol_enabled", "Savitzky-Golay"),
    ("normalize_enabled", "normalize"),
)


def _stage_detail(flag: str, params: Mapping[str, Any]) -> str:
    """The one number that identifies a stage's setting, for the chain sentence."""
    if flag == "notch_enabled":
        # THE SAME reader the kernel and the provenance stamp use
        # (`modules.notch_freqs_param`), never `params["notch_freqs"]` directly: the two
        # spellings — an explicit line list, or a fundamental + harmonics series — are
        # mutually exclusive, and a sentence that merged them would describe a filter
        # nobody ran.
        freqs = notch_freqs_param(params)
        if freqs:
            return " " + "/".join(f"{float(f):g}" for f in freqs) + " Hz"
        harmonics = int(params.get("notch_harmonics", 1))
        base = f" {float(params.get('notch_freq_hz', 60.0)):g} Hz"
        return base + (f" ×{harmonics}" if harmonics > 1 else "")
    if flag == "median_enabled":
        return f" {float(params.get('median_window_s', 0.0)) * 1e3:g} ms"
    if flag == "hp_enabled":
        return f" {float(params.get('hp_cutoff_hz', 0.0)):g} Hz"
    if flag == "detrend_enabled":
        return (f" {str(params.get('detrend_method', 'block_median'))}"
                f" {float(params.get('detrend_window_s', 0.0)):g} s")
    if flag == "rmean_enabled":
        return f" {float(params.get('rmean_window_s', 0.0)) * 1e3:g} ms"
    if flag == "savgol_enabled":
        return f" {float(params.get('savgol_window_s', 0.0)) * 1e3:g} ms"
    if flag == "normalize_enabled":
        return f" {str(params.get('normalize_mode', 'zscore'))}"
    return ""


def chain_description(params: Mapping[str, Any] | None) -> str:
    """The processing chain as one line a human can check against the form.

    ``"notch 60 Hz → detrend block_median 0.05 s → low-pass bessel 5000 Hz (order 8)"``,
    or ``""`` when nothing is enabled. Written once, here, and read by all three surfaces
    that have to agree about it: the Measure panel's basis note (what the numbers on
    screen were measured on), the saved row's status line, and the export sidecar's
    human-readable ``chain`` field beside the machine-readable ``params``. The exact
    parameters still ride the sidecar as a dict — this sentence is for the reader, the
    dict is for the rerun.
    """
    if not params:
        return ""
    parts = [f"{name}{_stage_detail(flag, params)}"
             for flag, name in _STAGE_SPELLING if params.get(flag)]
    cutoff = params.get("filter_cutoff_hz")
    if cutoff is not None:
        parts.append(f"low-pass {str(params.get('filter_kind', 'bessel'))} "
                     f"{float(cutoff):g} Hz (order {int(params.get('filter_order', 8))})")
    return " → ".join(parts)


# ---------------------------------------------------------------------------
# the basis stamp
# ---------------------------------------------------------------------------
#: The measurement-table column that records WHICH array a row was measured on.
#: Additive: declared outside ``measure.MEASUREMENT_SCHEMA`` (see the module docstring).
SIGNAL_BASIS_COLUMN = "signal_basis"

#: The row was measured on the calibrated channel exactly as loaded.
BASIS_RAW = "raw"
#: The row was measured on :func:`conditioned_trace` of that channel.
BASIS_PROCESSED = "processed"
#: Not recorded — a row saved before this column existed. NEVER read as "raw":
#: "nobody wrote this down" and "this was the raw channel" are different facts.
BASIS_UNKNOWN = ""

#: The column's header tooltip. Lives here, with the column, so the table and the
#: sidecar cannot define the same word two ways.
SIGNAL_BASIS_TIP = (
    "Which signal this row was measured on: 'raw' = the calibrated channel as loaded, "
    "'processed' = the conditioned trace (the filter chain on the Preprocess tab — the "
    "same array the on-trace preview curve draws). Blank = not recorded (a row saved "
    "before this column existed); it is NOT a claim that the row was raw."
)


def basis_label(basis: str) -> str:
    """A basis value as a short phrase for a status line ('the raw channel' / …)."""
    if basis == BASIS_PROCESSED:
        return "the processed trace"
    if basis == BASIS_RAW:
        return "the raw channel"
    return "an unrecorded signal"


def ensure_signal_basis_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a default-``''`` :data:`SIGNAL_BASIS_COLUMN` to ``df`` IN PLACE if missing.

    Returns the same frame for convenience. The exact shape of
    ``scoring.ensure_added_column`` and for the same reason: an older project (or an
    older CSV) simply has no such column, and filling it with the dtype's documented
    default is what makes the schema genuinely additive rather than a migration. Existing
    values are never touched — idempotent, so it is safe to call on every append and on
    every restore.
    """
    if SIGNAL_BASIS_COLUMN not in df.columns:
        df[SIGNAL_BASIS_COLUMN] = pd.Series(
            [BASIS_UNKNOWN] * len(df), index=df.index, dtype="string")
    return df


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------
def processed_trace_export_paths(path: str | Path) -> tuple[Path, ...]:
    """Every file :func:`write_processed_trace` writes for ``path``, in write order.

    Two, not one: the CSV and its ``<stem>.meta.json`` provenance sidecar — the same
    contract, and the same reason, as ``scoring.events_export_paths``. A caller that has
    to ask "may I overwrite?" must ask about both files, not just the one the user named.
    """
    p = Path(path)
    data = p.with_suffix(".csv") if p.suffix == "" else p
    return (data, data.with_suffix(".meta.json"))


def _json_safe(obj: Any) -> Any:
    """numpy scalars/arrays -> plain Python, so a params dict survives ``json.dump``."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _stamp(value: float | None) -> float | None:
    """A float stamp, or ``None`` for "not recorded" — and NaN/inf count as not recorded.

    One rule for ``t0_s`` and the calibration pair, because a sidecar with ``NaN`` in it
    is not JSON any more (``json.dump`` writes a bare ``NaN`` literal that strict readers
    reject) and a stamp nobody can read is the same fact as a stamp nobody wrote.
    """
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def processed_trace_provenance(
    *,
    params: Mapping[str, Any],
    fs: float,
    n_samples: int,
    start_idx: int,
    units: str,
    recording: str = "",
    doc_key: str = "",
    channel: str = "",
    channel_index: int | None = None,
    t0_s: float | None = None,
    span: str = "",
    cal_gain: float | None = None,
    cal_offset: float | None = None,
) -> dict[str, Any]:
    """The sidecar block for an exported processed trace: everything but the samples.

    THE REPRODUCTION RECIPE, and what has to be in the file for it to run. The chain does
    not run on the recording as stored — it runs on the CALIBRATED channel — so rebuilding
    an exported trace from ``doc_key`` takes two facts, not one::

        raw   = stored_channel * float32(cal_gain) + float32(cal_offset)
        again = conditioned_trace(raw, sample_rate_hz, params)[start_idx:end_idx]

    (the chain on the WHOLE channel, sliced after — every stage is a filter with memory,
    so conditioning the cut instead gives different numbers at its edges). Both facts are
    here: ``cal_gain``/``cal_offset``, the same stamp
    :data:`~nanoscope.analysis.measure.MEASUREMENT_SCHEMA` puts on a measured row and the
    same pair as ``ChannelCalibration.fingerprint`` — the only two numbers that alter the
    array — and the params, machine-readable, to feed back to :func:`conditioned_trace`.
    Those params are :func:`conditioning_params` of what the caller passed, not the dict
    itself: callers hold wider dicts (a whole Detection form, or the stored
    ``AnalysisResult.params`` the Measure panel falls back to), the chain reads none of
    the extra keys, and a recipe that lists the Schmitt thresholds this trace was never
    filtered by is describing a detection run rather than these samples. Narrowing changes
    no array — the round-trip above still rebuilds the export byte for byte — it changes
    what the file CLAIMS. Without the calibration the sidecar was measurably not a
    recipe: the same file and the same params under gain 1 and gain 10 produced two
    different arrays and two provenance blocks identical in every field but the file name,
    and a reader's only recourse — assume gain 1 — missed by 923 pA on the 2026-09-02
    audit's probe.

    ``None`` for either is NOT gain 1: it means the caller did not record the calibration,
    it is written as JSON ``null``, and it reads exactly like :data:`BASIS_UNKNOWN` —
    "nobody wrote this down". Such a sidecar still documents the chain and still names the
    slice; it is the one thing it cannot do that this docstring no longer claims, which is
    rebuild the samples from the file on disk.

    The rest is what the numbers mean rather than how to remake them: the human sentence
    for the chain, which slice of which recording/channel it is, the units, and the
    instrument-clock origin so ``time_s`` can be put back on the experiment's clock.
    ``normalize`` is in that params dict and it is the reason ``units`` may read
    ``"pA (normalized — dimensionless)"``: after that stage the numbers are no longer
    picoamps and the export must not pretend otherwise.
    """
    chain = conditioning_params(params)
    return {
        "export": "processed_trace",
        "app": "nanoscope",
        "recording": recording,
        "doc_key": doc_key,
        "channel": channel,
        "channel_index": (None if channel_index is None else int(channel_index)),
        "span": span,
        "start_idx": int(start_idx),
        "end_idx": int(start_idx) + int(n_samples),
        "n_samples": int(n_samples),
        "sample_rate_hz": float(fs),
        "units": str(units),
        "cal_gain": _stamp(cal_gain),
        "cal_offset": _stamp(cal_offset),
        "t0_s": _stamp(t0_s),
        "signal_basis": BASIS_PROCESSED,
        "chain": chain_description(chain),
        "params": _json_safe(chain),
    }


#: ``%.9g`` on a float32: 9 significant digits is more than the ~7 a float32 carries, so
#: the CSV round-trips every sample exactly while staying half the width of ``repr``.
_SAMPLE_FMT = "%.9g"


def write_processed_trace(
    path: str | Path,
    samples: np.ndarray,
    fs: float,
    *,
    units: str = "",
    start_idx: int = 0,
    t0_s: float | None = None,
    params: Mapping[str, Any] | None = None,
    recording: str = "",
    doc_key: str = "",
    channel: str = "",
    channel_index: int | None = None,
    span: str = "",
    cal_gain: float | None = None,
    cal_offset: float | None = None,
) -> dict[str, Path]:
    """Write the conditioned samples as CSV + a ``.meta.json`` provenance sidecar.

    The CSV is two columns and a header naming the units:
    ``time_s,current_pA`` (or ``current`` when the channel declared no units), one row per
    sample, ``time_s`` counted from the RECORDING's first sample —
    ``(start_idx + i) / fs`` — so a slice exported out of the middle of a recording still
    says where it came from. ``t0_s`` rides the sidecar rather than the column so the
    numbers in the file are the app's own axis and the experiment's clock is one addition
    away, exactly as ``measure.MEASUREMENT_SCHEMA`` treats ``t_start_s``/``t0_s``.

    ``cal_gain``/``cal_offset`` are the calibration ``samples`` were conditioned UNDER —
    pass ``ChannelCalibration.fingerprint()`` from the same channel the array came from.
    They are what makes the sidecar a recipe instead of a description, because the chain
    ran on the calibrated channel and not on the file; leave them out and they are written
    as ``null``, which says "not recorded" and does not say gain 1.

    Written through ``core.atomic_write.staged_path`` like every other export over user
    data: a truncated half-file with a plausible header is worse than no file, because it
    opens. The sidecar is written second and to the same rule, so a reader that sees the
    CSV can rely on the JSON beside it being complete.

    Returns ``{"data": <csv>, "meta": <json>}``.
    """
    data_path, meta_path = processed_trace_export_paths(path)
    arr = np.asarray(samples, dtype=np.float64).ravel()
    n = int(arr.size)
    t = (np.arange(n, dtype=np.float64) + float(start_idx)) / float(fs)
    header = "time_s," + ("current_" + units.replace(" ", "_") if units else "current")
    with staged_path(data_path) as staged:
        np.savetxt(staged, np.column_stack([t, arr]), delimiter=",",
                   header=header, comments="", fmt=_SAMPLE_FMT)
    prov = processed_trace_provenance(
        params=params or {}, fs=fs, n_samples=n, start_idx=start_idx, units=units,
        recording=recording, doc_key=doc_key, channel=channel,
        channel_index=channel_index, t0_s=t0_s, span=span,
        cal_gain=cal_gain, cal_offset=cal_offset)
    prov["data_file"] = data_path.name
    with staged_path(meta_path) as staged:
        # ``staged`` IS a Path (staged_path yields one), so it is written directly rather
        # than through ``Path(staged)``. That is not tidying: tests/core/test_persist_
        # invariant.py::test_no_module_in_the_app_writes_a_path_it_did_not_stage is a
        # SYNTACTIC guard — it accepts a write whose receiver is a name bound by the
        # ``with``, and ``Path(staged)`` is a Call, so the wrapper made a correctly staged
        # write read as an unstaged one and failed the build. Keep the receiver a bare name.
        staged.write_text(json.dumps(prov, indent=2, sort_keys=False) + "\n",
                          encoding="utf-8")
    return {"data": data_path, "meta": meta_path}
