"""What a result was made with, and one judge of whether each of those inputs still holds.

Pure and Qt-free. A stored event table already records the inputs it was made from: the sample
rate in its provenance (``fs``), the calibration and open-pore estimator stamps on the result, and
the chain and thresholds in the params the detector was handed. Each surface asked about them on
its own. The amber line judges rate, calibration and estimator in an order of precedence and names
only the first that moved, so a table stale on two counts said one; the chain notice compares the
form with the run and nothing else (``docs/EDITING_FOUNDATION_DESIGN.md`` 1.4).

* :func:`made_with` reads those inputs off a result into a :class:`MadeWith`. It adds no stamp: an
  input a result did not record (a legacy table) is ``None``.
* :func:`judge_inputs` gives EACH input its own :class:`InputVerdict` -- same, changed (was and
  now) or unknown -- in the fixed order of :data:`INPUT_NAMES`, with the action that repairs it
  and a sentence that says so.

The rules are the ones already in force, reused rather than restated: the rate is
:func:`~nanoscope.analysis.events.stale_rate_hz`'s, the chain and the thresholds are the chain
notice's (:func:`~nanoscope.analysis.conditioning_recipes.chain_differences`,
:func:`~nanoscope.analysis.conditioning_recipes.threshold_differences` and
:func:`~nanoscope.analysis.conditioning_recipes.polarity_differences`), and the calibration and
the estimator are their stamps compared as the amber line compares them
(``AnalysisPanel._calibration_moved`` and ``_baseline_staleness``). The amber line keeps its
headline and its precedence; this is the list under it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .conditioning import conditioning_params
from .conditioning_recipes import (
    CHAIN_KEYS,
    THRESHOLD_KEYS_BY_MODULE,
    ChainDiff,
    chain_differences,
    describe_differences,
    polarity_differences,
    revert_withheld,
    stage_rows,
    summary_hidden_differences,
    threshold_differences,
)
from .events import EventSet, _usable_rate, stale_rate_hz
from .modules import FIT_BASELINE_KEY, FIT_RATE_KEY, preprocess_params_from_provenance

__all__ = [
    "INPUT_NAMES",
    "InputVerdict",
    "LiveInputs",
    "MadeWith",
    "judge_inputs",
    "made_with",
]

#: The inputs :func:`judge_inputs` judges, in the order it lists them: the amber line's order of
#: precedence (rate, then calibration, then the estimator), then the form's two.
INPUT_NAMES: tuple[str, ...] = ("rate", "calibration", "baseline", "chain", "thresholds")


@dataclass(frozen=True)
class MadeWith:
    """The inputs a result recorded it was made with; ``None`` wherever it recorded nothing.

    ``rate_hz`` is the sample rate the table was detected at (``provenance['fs']``),
    ``calibration`` the ``(gain, offset)`` it was measured under, and ``baseline_fingerprint``
    the stamp of the open-pore estimator that measured its depths. ``chain`` holds the
    conditioning keys of the params the detector ran with
    (:func:`~nanoscope.analysis.conditioning.conditioning_params`: the chain keys, and the
    polarity the detrend stage and the detector read, judged with the thresholds when the
    drift stage does not name it) and ``thresholds`` the keys of
    :data:`~nanoscope.analysis.conditioning_recipes.THRESHOLD_KEYS_BY_MODULE` it recorded. Both
    are ``None`` for a result the Schmitt/CUSUM chain did not make (the ML detector runs its own
    model chain).
    """

    module_id: str
    rate_hz: float | None = None
    calibration: tuple[float, float] | None = None
    baseline_fingerprint: str | None = None
    chain: Mapping[str, Any] | None = None
    thresholds: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LiveInputs:
    """The same inputs as they stand now. The caller fills it.

    ``rate_hz`` is the recording's live rate and ``calibration`` the channel's live
    ``Session.calibration(...).fingerprint()``. ``baseline_fingerprint`` is the fingerprint of the
    estimator a fresh run would use on THIS recording, clamp included, as
    ``AnalysisPanel._live_baseline_for_run`` resolves it: the form's own fingerprint would leave a
    short recording stale for good. ``chain`` and ``thresholds`` are the Detection form's
    ``params()``, one dict for both. ``None`` is "not known now". A rate not known now reads as
    changed, as the amber line reads a recording that has lost its rate; any other input reads as
    unknown.
    """

    rate_hz: float | None = None
    calibration: tuple[float, float] | None = None
    baseline_fingerprint: str | None = None
    chain: Mapping[str, Any] | None = None
    thresholds: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class InputVerdict:
    """One input's verdict.

    ``name`` is one of :data:`INPUT_NAMES` and ``state`` is ``"same"``, ``"changed"`` or
    ``"unknown"``. ``was`` and ``now`` show the input as recorded and as it stands: for the chain
    and the thresholds, only the stages or thresholds that differ, as each side sets them
    (``"k_in 5"`` and ``"k_in 6"``). ``action`` is what repairs it: ``"re-run"`` (rate,
    calibration), ``"re-measure"`` (the estimator), ``"revert"`` (the chain and the thresholds,
    which a re-run would settle as well; ``"re-run"`` when Revert cannot put them back), or
    ``""``. ``sentence`` says all of it in plain English.
    """

    name: str
    state: str
    was: str
    now: str
    action: str
    sentence: str


# ---------------------------------------------------------------------------
# what a result was made with
# ---------------------------------------------------------------------------
def made_with(result: Any) -> MadeWith:
    """The inputs ``result`` recorded, read from what it already stores.

    An event table holds its rate in its provenance and its calibration and estimator stamps on
    the result. A Schmitt or CUSUM table's chain and thresholds come from the params it ran with
    (:func:`_run_params`). A ``fit.dwell`` / ``fit.depth`` holds its TABLE's rate and estimator
    stamps in the fit's meta, where ``modules.fit_dwell`` / ``fit_depth`` record them, and no
    chain.
    """
    module_id = str(getattr(result, "module_id", "") or "")
    payload = getattr(result, "payload", None)
    if isinstance(payload, EventSet):
        rate = (payload.provenance or {}).get("fs")
        baseline = getattr(result, "baseline_fingerprint", None)
    else:
        meta = getattr(payload, "meta", None)
        meta = meta if isinstance(meta, Mapping) else {}
        rate = meta.get(FIT_RATE_KEY)
        baseline = getattr(result, "baseline_fingerprint", None) or meta.get(FIT_BASELINE_KEY)
    chain = thresholds = None
    keys = THRESHOLD_KEYS_BY_MODULE.get(module_id)
    if keys is not None:
        run = _run_params(result)
        if any(key in run for key in CHAIN_KEYS):
            chain = conditioning_params(run)
        thresholds = {key: run[key] for key in keys if key in run} or None
    return MadeWith(
        module_id=module_id,
        rate_hz=_usable_rate(rate),
        calibration=_gain_offset(getattr(result, "calibration_fingerprint", None)),
        baseline_fingerprint=str(baseline) if baseline else None,
        chain=chain,
        thresholds=thresholds,
    )


def _run_params(result: Any) -> dict[str, Any]:
    """The settings a stored detection ran under, as far as it recorded them: its ``params`` (the
    dict the module was handed), else the detector's own stamp in its provenance.

    The two sources, in the same order, of ``gui.analysis_panel.schmitt_params_from_result``,
    which re-seeds the form and supplies Revert, so the judge and the Revert it offers read one
    record. Its open-pore reference repair is left out: that concerns the estimator, which is
    judged by its fingerprint. The provenance is read through
    :func:`~nanoscope.analysis.modules.preprocess_params_from_provenance`, which turns the
    detrend knobs and the requested polarity back into form keys and leaves the CUSUM
    thresholds in. The panel's fallback reaches the same stamp through its own projection onto
    the form's keys (``gui.analysis_panel._params_from_stamp``) and puts the CUSUM thresholds
    back, so a legacy table (``params`` empty, provenance only) is judged on its k and h by the
    chain notice and by the Details list alike.
    """
    params = dict(getattr(result, "params", None) or {})
    # The estimator's keys alone name no run (a v1/v2 table re-measured before the stamp was
    # kept): read the stamp under them, as the panel does.
    if any(not str(k).startswith("baseline_") for k in params):
        return params
    payload = getattr(result, "payload", None)
    if isinstance(payload, EventSet) and payload.provenance:
        return {**preprocess_params_from_provenance(payload.provenance), **params}
    return params


def _gain_offset(value: Any) -> tuple[float, float] | None:
    """``value`` as a ``(gain, offset)`` pair, or None when it is not one."""
    try:
        gain, offset = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return gain, offset


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------
def judge_inputs(made: MadeWith, now: LiveInputs, *,
                 fs: float | None = None) -> tuple[InputVerdict, ...]:
    """Each input of ``made`` against ``now``: one verdict per input, in :data:`INPUT_NAMES` order.

    ``fs`` is the rate the chain's notch lines are resolved at (``chain_differences``); left out,
    it is the live rate. An input ``made`` did not record is "unknown". So is the estimator when
    the table has no stamp, but there the action is "re-measure": the amber line reads a missing
    stamp as a table that predates it, measured by the estimator the stamp replaced, and offers
    exactly that repair.
    """
    fs = now.rate_hz if fs is None else fs
    return (
        _judge_rate(made, now),
        _judge_calibration(made, now),
        _judge_baseline(made, now),
        _judge_chain(made, now, fs),
        _judge_thresholds(made, now, fs),
    )


def _hz(rate: float | None, digits: int = 6) -> str:
    if rate is None:
        return "unknown"
    return f"{rate / 1e3:.{digits}g} kHz" if rate >= 1e3 else f"{rate:.{digits}g} Hz"


def _gain_offset_text(cal: tuple[float, float] | None, digits: int = 6) -> str:
    if cal is None:
        return "unknown"
    return f"gain {cal[0]:.{digits}g}, offset {cal[1]:.{digits}g}"


def _apart(fmt: Callable[..., str], was: Any, now: Any) -> tuple[str, str]:
    """``fmt`` of a value that changed, at full precision when six digits would print the change
    as two equal numbers ("50 kHz when detected, 50 kHz now")."""
    a, b = fmt(was), fmt(now)
    if a == b:
        a, b = fmt(was, 17), fmt(now, 17)
    return a, b


def _judge_rate(made: MadeWith, now: LiveInputs) -> InputVerdict:
    stamped, live = _usable_rate(made.rate_hz), _usable_rate(now.rate_hz)
    if stamped is None:
        return InputVerdict("rate", "unknown", "not recorded", _hz(live), "",
                            "Sample rate: not recorded with this table, so it is not compared.")
    if stale_rate_hz({"fs": stamped}, live) is None:
        text = _hz(stamped)
        return InputVerdict("rate", "same", text, text, "",
                            f"Sample rate: {text}, the rate this table was detected at.")
    if live is None:
        # There is no new clock to measure on: the amber line's own way forward for this case
        # (``events.stale_rate_reason``) is a re-run under the rate the user declares.
        was = _hz(stamped)
        return InputVerdict("rate", "changed", was, "unknown", "re-run",
                            f"Sample rate: {was} when detected; the recording's rate is not "
                            "known now. Declare it, then re-run.")
    was, is_ = _apart(_hz, stamped, live)
    return InputVerdict("rate", "changed", was, is_, "re-run",
                        f"Sample rate: {was} when detected, {is_} now. "
                        "Re-run to measure on the new clock.")


def _judge_calibration(made: MadeWith, now: LiveInputs) -> InputVerdict:
    stamped, live = made.calibration, _gain_offset(now.calibration)
    if stamped is None:
        return InputVerdict("calibration", "unknown", "not recorded", _gain_offset_text(live), "",
                            "Calibration: not recorded with this table, so it is not compared.")
    if live is None:
        was = _gain_offset_text(stamped)
        return InputVerdict("calibration", "unknown", was, "unknown", "",
                            f"Calibration: {was} when detected; the channel's calibration now is "
                            "not known, so it is not compared.")
    if tuple(stamped) == live:
        text = _gain_offset_text(live)
        return InputVerdict("calibration", "same", text, text, "",
                            f"Calibration: {text}, as when detected.")
    was, is_ = _apart(_gain_offset_text, stamped, live)
    return InputVerdict("calibration", "changed", was, is_, "re-run",
                        f"Calibration: {was} when detected; {is_} now. "
                        "Re-run to measure under the new calibration.")


def _judge_baseline(made: MadeWith, now: LiveInputs) -> InputVerdict:
    have, want = made.baseline_fingerprint, now.baseline_fingerprint
    if not want:
        return InputVerdict("baseline", "unknown", have or "not recorded", "unknown", "",
                            "Open-pore estimator: the one a fresh run would use here is not "
                            "known, so it is not compared.")
    if not have:
        return InputVerdict("baseline", "unknown", "not recorded", want, "re-measure",
                            "Open-pore estimator: not recorded — this table predates the "
                            "estimator stamp. Re-measure to measure its depths against the one "
                            "now configured.")
    if have == want:
        return InputVerdict("baseline", "same", have, want, "",
                            "Open-pore estimator: the one now configured, as when these depths "
                            "were measured.")
    return InputVerdict("baseline", "changed", have, want, "re-measure",
                        "Open-pore estimator: changed since these depths were measured. "
                        "Re-measure to measure them against the one now configured.")


def _not_recorded(name: str, label: str, module_id: str) -> InputVerdict:
    """The verdict on a form input ``made`` did not record: unknown, and why, told by the kind
    of result as the chain notice tells it (``conditioning_recipes.judge_chain``)."""
    if module_id in THRESHOLD_KEYS_BY_MODULE:
        why = "not recorded with this table, so the form is not compared with it."
    elif module_id == "detect.keras":
        why = "not compared — the ML detector made this table with its own model and settings."
    elif module_id == "detect.reconciled":
        why = "not compared — this is a reconciled table; compare the form with its sources."
    elif module_id.startswith("fit."):
        why = "not compared — a fit has none of its own; judge the table it was made on."
    else:
        why = "not compared — the Schmitt/CUSUM chain did not make this result."
    return InputVerdict(name, "unknown", "not recorded", "", "", f"{label}: {why}")


def _stage_settings(params: Mapping[str, Any], stages: Sequence[str], fs: float | None) -> str:
    """``stages`` as ``params`` sets them: ``"Low-pass Bessel 8 · 10 kHz; Notch off"``."""
    rows = {row.key: row for row in stage_rows(params, fs)}
    return "; ".join(f"{rows[key].title} {rows[key].summary if rows[key].enabled else 'off'}"
                     for key in stages)


def _chain_sides(run: Mapping[str, Any], form: Mapping[str, Any], diffs: Sequence[ChainDiff],
                 fs: float | None) -> tuple[str, str]:
    """The chain row's was and now: each stage that differs as each side sets it, plus, in the
    chain judge's words, what its summary does not show (the drift stage's polarity, mask,
    dilate). A stage whose two summaries read the same is told by those words alone, each side
    taken as the thresholds row takes it, so the row never reads "X → X" (a polarity edit with
    the drift stage on read "Baseline drift block median · 0.5 s" on both sides)."""
    hidden = set(summary_hidden_differences(form, run, fs=fs))
    back = {(d.stage, d.key): d.text for d in chain_differences(run, form, fs=fs)}

    def form_side(d: ChainDiff) -> str:
        text = back.get((d.stage, d.key))
        return text.partition(" → ")[0] if text else d.text.partition(" → ")[2]

    was: list[str] = []
    now: list[str] = []
    for stage in dict.fromkeys(d.stage for d in diffs):
        a, b = _stage_settings(run, (stage,), fs), _stage_settings(form, (stage,), fs)
        told = [d for d in diffs if d.stage == stage and (a == b or d in hidden)]
        if a != b:
            was.append(a)
            now.append(b)
        was.extend(d.text.partition(" → ")[0] for d in told)
        now.extend(form_side(d) for d in told)
    return "; ".join(was), "; ".join(now)


def _judge_chain(made: MadeWith, now: LiveInputs, fs: float | None) -> InputVerdict:
    if made.chain is None:
        return _not_recorded("chain", "Signal chain", made.module_id)
    if now.chain is None:
        return InputVerdict("chain", "unknown", "", "", "",
                            "Signal chain: the form's is not known, so it is not compared.")
    try:
        diffs = chain_differences(now.chain, made.chain, fs=fs)
        was, is_ = _chain_sides(made.chain, now.chain, diffs, fs) if diffs else ("", "")
    except (TypeError, ValueError, KeyError):
        # A recorded value the chain cannot read (a malformed legacy stamp): the judge says so
        # rather than raise, as the chain notice does (``ChainStatus.unknown``).
        return InputVerdict("chain", "unknown", "not readable", "", "",
                            "Signal chain: the chain this table ran with could not be read, so "
                            "the form is not compared with it.")
    if not diffs:
        return InputVerdict("chain", "same", "", "", "",
                            "Signal chain: the form's is the one this table ran with.")
    withheld = _revert_withheld(made, now, fs)
    if withheld:
        return InputVerdict("chain", "changed", was, is_, "re-run",
                            f"Signal chain: {describe_differences(diffs)}. Re-run to apply it "
                            f"— Revert cannot restore {withheld}.")
    return InputVerdict("chain", "changed", was, is_, "revert",
                        f"Signal chain: {describe_differences(diffs)}. "
                        "Revert, or re-run to apply it.")


def _revert_withheld(made: MadeWith, now: LiveInputs, fs: float | None) -> str:
    """Why the form's Revert cannot put the form back, or ``""``: the chain notice's own rule
    (``revert_withheld``) on the same run, so this list never offers a Revert that every Revert
    control withholds (S10-STAGE2). One Revert puts back the chain and the thresholds together,
    so it is the verdict of both rows."""
    if made.chain is None or now.chain is None:
        return ""
    run = {**made.chain, **(made.thresholds or {})}
    form = {**now.chain, **(now.thresholds or {})}
    try:
        return revert_withheld(form, run, fs=fs)
    except (TypeError, ValueError, KeyError):
        return ""   # an unreadable stamp: the chain row already says so


def _judge_thresholds(made: MadeWith, now: LiveInputs, fs: float | None = None) -> InputVerdict:
    if made.thresholds is None:
        return _not_recorded("thresholds", "Thresholds", made.module_id)
    form = now.thresholds or {}
    if not any(key in form for key in made.thresholds):
        return InputVerdict("thresholds", "unknown", "", "", "",
                            "Thresholds: the form's are not known, so they are not compared.")
    # The chain notice's own judge and words ("k_in 5 → 6", "merge gap off → 1 ms"): one rule
    # for whether a threshold moved, so the notice and this list cannot disagree. The polarity
    # the drift stage does not name is the notice's "Settings changed" too, so it is listed
    # here; it is recorded with the chain (``MadeWith.chain``).
    diffs = (threshold_differences(form, made.thresholds, made.module_id)
             + polarity_differences(now.chain, made.chain))
    if not diffs:
        return InputVerdict("thresholds", "same", "", "", "",
                            "Thresholds: the form's are the ones this table ran with.")
    # Each side in those same words: a difference reads run → form, so the run's settings are
    # what precedes its arrow, and the form's are the same differences taken form → run.
    back = (threshold_differences(made.thresholds, form, made.module_id)
            + polarity_differences(made.chain, now.chain))
    was, is_ = ("; ".join(d.text.partition(" → ")[0] for d in side) for side in (diffs, back))
    withheld = _revert_withheld(made, now, fs)
    if withheld:
        return InputVerdict("thresholds", "changed", was, is_, "re-run",
                            f"Thresholds: {describe_differences(diffs)}. Re-run to apply them "
                            f"— Revert cannot restore {withheld}.")
    return InputVerdict("thresholds", "changed", was, is_, "revert",
                        f"Thresholds: {describe_differences(diffs)}. "
                        "Revert, or re-run to apply them.")
