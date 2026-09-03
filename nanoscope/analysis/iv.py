"""I-V / conductance analysis: make the absolute current physical.

For a nanopore the open-pore conductance is simply Ohm's law on the open channel,

    G = I_open / V_applied,

with ``I_open`` the current with no molecule in the pore (the open-pore level)
and ``V_applied`` the command voltage. It is the single number that says whether
the recording's absolute current scale is believable at all: a 22 nm pore in 1x
PBS sits near ~100 nS, so a channel whose gain is off by 100x betrays itself as a
~1 nS (or ~10 uS) "pore" long before any event analysis. That is the whole point
of the deferred current-gain calibration -- the user dials the CURRENT-channel
gain until ``G`` lands in a physical range, and this module is what turns a gain
into a conductance they can judge.

TWO THINGS LIVE HERE, both pure (numpy + :mod:`.populations`, no Qt):

* the per-recording read-out -- :func:`open_pore_current`, :func:`conductance`
  and the unit-aware :func:`conductance_in_nS` -- which the Calibration panel
  recomputes live as the gain is edited; and
* the I-V characterisation -- :func:`iv_point` gathers one ``(V, I_open)`` per
  open recording and :func:`iv_fit` fits a straight line whose **slope is the
  pore conductance** and whose intercept is the leak current.

UNITS ARE THE ONLY SUBTLETY. Conductance in nanosiemens is ``I[A]/V[V] * 1e9``;
expressed in the units the app actually carries that is a clean scale factor:

    pA / mV  = nS   (x1)        nA / mV = uS   (x1000 -> nS)
    pA / V   = pS   (x1e-3 -> nS)   nA / V  = nS   (x1)

:func:`to_nS` is the single place that scale lives; every other function routes
through it, so there is exactly one table of unit conversions to get right. All
currents keep their signed value (at -1000 mV the open pore is negative and so is
V, so ``G`` comes out positive as it must).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np

from ..core.document import TraceDocument
from ..core.types import ChannelKind, normalise_units
from .populations import linear_fit

if TYPE_CHECKING:  # avoid importing Qt (Session pulls in PySide6); duck-typed at runtime.
    from ..core.session import Session

__all__ = [
    "open_pore_current",
    "conductance",
    "conductance_in_nS",
    "conductance_note",
    "voltage_units_clause",
    "voltage_units_note",
    "to_nS",
    "current_scale",
    "AppliedVoltage",
    "applied_voltage_readout",
    "applied_voltage_mV",
    "IVPoint",
    "IVFit",
    "IVExclusion",
    "iv_point",
    "iv_point_refusal",
    "iv_points",
    "iv_exclusions",
    "iv_fit",
    "channel_median_stats",
    "forget_channel_medians",
]

# Current unit -> amps, voltage unit -> volts. The ONLY conversion tables in the
# module; `to_nS` and `current_scale` are the only consumers, so a mistake here is
# caught in one place -- and a surface that needs to convert a current MUST route
# through `current_scale` rather than keep a second table of its own. The I-V window
# kept one ({"pA","nA","uA"}, no "A") and therefore called a unit "unknown" that this
# module converts correctly, in the same session, for the same channel (JHT-03).
_I_TO_A: dict[str, float] = {"pA": 1e-12, "nA": 1e-9, "uA": 1e-6, "A": 1.0}
_V_TO_V: dict[str, float] = {"mV": 1e-3, "V": 1.0}


# ---------------------------------------------------------------------------
# open-pore current
# ---------------------------------------------------------------------------


def open_pore_current(current: np.ndarray, *, method: str = "median") -> float:
    """Robust single-number open-pore current ``I_open`` of a current channel.

    The open pore is the level the trace sits at when nothing is translocating;
    events are brief excursions off it. The median is the right estimator because
    it ignores those excursions (up to ~50% occupancy) without needing to detect
    them first -- no baseline, no sample rate, no threshold. Signed, in whatever
    units the array carries (negative at negative bias); take ``abs`` yourself if
    you want a magnitude.

    Args:
        current: The current trace (1-D array, any units). ``float32`` in is fine.
        method: ``"median"`` (the default and only robust estimator) or ``"mean"``
            (offered for completeness; NOT robust to events -- use it only on a
            trace you know is event-free).

    Returns:
        The open-pore current as a Python float.

    Raises:
        ValueError: on an empty trace, a trace with no finite samples, or an
            unknown ``method`` -- a silently wrong open-pore level would poison
            every conductance downstream.
    """
    x = np.asarray(current, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot take the open-pore current of an empty trace")
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("current has no finite samples; cannot estimate an open-pore level")
    if method == "median":
        return float(np.median(finite))
    if method == "mean":
        return float(np.mean(finite))
    raise ValueError(f"unknown method {method!r}; expected 'median' or 'mean'")


# ---------------------------------------------------------------------------
# whole-channel medians, computed ONCE per (document, channel, calibration)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS (END-003). Every number this module reads off a trace is a median
# of a WHOLE channel: the open-pore current (the CURRENT channel) and the applied
# voltage read-out (the VOLTAGE channel). Both are properties of the SIGNAL, so
# nothing a user does to the event table can move them -- accepting or rejecting an
# event, an Accept all / Reject all, a hand measurement, a label, an undo. The I-V
# window nevertheless re-derives its contents from the session on every one of those
# gestures (MainWindow._refresh_surface_windows), and iv_exclusions re-asks the
# voltage question iv_points had just answered, so one checkbox click was costing
# ~230-250 ms of np.median over the owner's three open recordings (9,074,042 samples
# in one of them alone) to arrive at bit-identical numbers.
#
# The key is the CONTENT key the rest of the app uses -- (document, channel, gain,
# offset) -- plus two things a bare gain/offset fingerprint does NOT cover:
#
#   * ``target_units``. ChannelCalibration.fingerprint() is (gain, offset) only,
#     deliberately ("target_units is a label"), but the units are half of what the
#     callers here return, and declaring the units of an undeclared channel is
#     precisely the fix this module tells the user to make. Keyed on the fingerprint
#     alone, a channel the user had just declared as volts would keep reading back
#     "units unknown" from the memo -- a cache that outlives the user's own remedy.
#   * ``n_samples``, so a document that somehow presents a different length under the
#     same identity can never read a stale number.
#
# A document's raw samples are immutable for as long as it is open and a re-opened
# file gets a NEW doc_id (Document.doc_id is unique per load), so no other
# invalidation is needed; the LRU bound keeps closed documents from accumulating.
_MEDIAN_MEMO: "OrderedDict[tuple, Any]" = OrderedDict()
_MEDIAN_MEMO_MAX = 128
_MEDIAN_HITS = 0
_MEDIAN_MISSES = 0


def _channel_key(kind: str, session: "Session", doc_id: str, ch: int) -> tuple | None:
    """The content key for one calibrated channel, or ``None`` when it cannot be formed.

    ``None`` means "do not memoise" -- the caller then computes as it always did.
    Fails OPEN, never wrong: a key that cannot be built costs time, not correctness.
    """
    try:
        cal = session.calibration(doc_id, int(ch))
        n = int(session.document(doc_id).meta.n_samples)
        return (kind, doc_id, int(ch), float(cal.gain), float(cal.offset),
                str(cal.target_units or ""), n)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _memoised_channel(kind: str, session: "Session", doc_id: str, ch: int,
                      compute: "Callable[[], Any]") -> Any:
    """``compute()``, run at most once per ``(kind, document, channel, calibration)``.

    An exception from ``compute`` is propagated and NOT memoised -- a refusal
    (``open_pore_current`` on an all-non-finite channel) has to be re-raised for the
    caller that asks next, and caching it would freeze a transient into a verdict.
    """
    global _MEDIAN_HITS, _MEDIAN_MISSES
    key = _channel_key(kind, session, doc_id, ch)
    if key is None:
        return compute()
    if key in _MEDIAN_MEMO:
        _MEDIAN_MEMO.move_to_end(key)
        _MEDIAN_HITS += 1
        return _MEDIAN_MEMO[key]
    value = compute()
    _MEDIAN_MISSES += 1
    _MEDIAN_MEMO[key] = value
    while len(_MEDIAN_MEMO) > _MEDIAN_MEMO_MAX:
        _MEDIAN_MEMO.popitem(last=False)     # least-recently-used
    return value


def channel_median_stats() -> dict[str, int]:
    """How the whole-channel median memo is doing: ``{hits, misses, entries}``.

    Instrumentation, not decoration: the memo exists to keep a curation click off a
    ~9 M-sample np.median, and a COST invariant that cannot be asserted is a cost
    invariant that regresses. tests/gui/test_s4_vs_iv_median_cache.py reads it.
    """
    return {"hits": _MEDIAN_HITS, "misses": _MEDIAN_MISSES,
            "entries": len(_MEDIAN_MEMO)}


def forget_channel_medians() -> None:
    """Drop every memoised channel median (and reset the counters). For tests."""
    global _MEDIAN_HITS, _MEDIAN_MISSES
    _MEDIAN_MEMO.clear()
    _MEDIAN_HITS = _MEDIAN_MISSES = 0


def _open_pore_of(session: "Session", doc_id: str, ch: int) -> tuple[float, str]:
    """``(open-pore current, units)`` of a document's calibrated CURRENT channel."""
    def compute() -> tuple[float, str]:
        cur = session.calibrated(doc_id, ch)
        return (open_pore_current(cur.data), cur.units)
    return _memoised_channel("i_open", session, doc_id, ch, compute)


def _voltage_median_of(session: "Session", doc_id: str, ch: int) -> tuple[float, str]:
    """``(median, units)`` of a document's calibrated VOLTAGE channel.

    Deliberately NOT :func:`open_pore_current`: that estimator refuses a channel with
    no finite samples, and this read-out has always answered ``nan`` for one. The
    memo is a speed change only; it must not change a single number or a single
    refusal.
    """
    def compute() -> tuple[float, str]:
        v = session.calibrated(doc_id, ch)
        return (float(np.median(np.asarray(v.data, dtype=np.float64))), v.units)
    return _memoised_channel("v_med", session, doc_id, ch, compute)


# ---------------------------------------------------------------------------
# conductance (unit-agnostic core + unit-aware helpers)
# ---------------------------------------------------------------------------


def conductance(i_open: float, v_applied: float) -> float | None:
    """Open-pore conductance ``G = I_open / V`` in whatever units the two share.

    Unit-agnostic on purpose: pass ``(pA, mV)`` and you get nS; pass ``(nA, V)``
    and you also get nS; pass ``(A, V)`` and you get siemens. The one thing it
    refuses is a division that has no physical meaning -- ``V == 0`` (or a
    non-finite ``V``) returns ``None`` rather than an infinity that would render
    as a nonsense "pore".

    Args:
        i_open: Open-pore current (signed).
        v_applied: Applied voltage (signed), in units consistent with ``i_open``.

    Returns:
        ``I_open / V`` as a float, or ``None`` when ``V`` is zero / non-finite.
    """
    if not np.isfinite(v_applied) or v_applied == 0.0:
        return None
    return float(i_open) / float(v_applied)


def to_nS(ratio: float, current_units: str, voltage_units: str) -> float | None:
    """Convert a conductance expressed as ``current_units / voltage_units`` to nS.

    ``ratio`` is a bare number ``I / V`` whose physical meaning depends on the two
    units it was formed from. In siemens that conductance is
    ``ratio * amps_per(current_units) / volts_per(voltage_units)``; nanosiemens is
    ``* 1e9``. So ``pA/mV`` scales by 1, ``nA/V`` by 1, ``pA/V`` by 1e-3 and
    ``nA/mV`` by 1e3 -- see the module docstring.

    Returns:
        The conductance in nanosiemens, or ``None`` when either unit is not a
        recognised current/voltage unit (nS is then undefined and must not be
        faked).
    """
    ci = _I_TO_A.get(normalise_units(current_units))
    cv = _V_TO_V.get(normalise_units(voltage_units))
    if ci is None or cv is None or not np.isfinite(ratio):
        return None
    return float(ratio) * ci / cv * 1e9


def current_scale(from_units: str, to_units: str) -> float | None:
    """Multiplier that converts a current from ``from_units`` into ``to_units``.

    The second (and last) consumer of :data:`_I_TO_A`, for the surfaces that have to
    put several recordings' currents on ONE axis: ``current_scale("pA", "nA")`` is
    ``1e-3``. It exists so no caller has to write its own table -- the I-V window
    did, left ``"A"`` out of it, and so dropped a recording as "current units “A”
    unknown" while :func:`to_nS` was converting that very declaration to nS on the
    Calibration panel. One table, one verdict, everywhere.

    Args:
        from_units: The units the value is in (``"pA"``/``"nA"``/``"uA"``/``"A"``).
        to_units: The units wanted, same vocabulary.

    Returns:
        The factor, or ``None`` when EITHER unit is not a recognised current unit
        (the conversion is then undefined and must not be guessed -- ``""``, the
        undeclared state of every HDF5/CSV channel, lands here).
    """
    a_from = _I_TO_A.get(normalise_units(from_units))
    a_to = _I_TO_A.get(normalise_units(to_units))
    if a_from is None or a_to is None:
        return None
    return a_from / a_to


def conductance_in_nS(i_open: float, current_units: str, voltage_mV: float) -> float | None:
    """Open-pore conductance in nanosiemens from ``I_open`` and the voltage in mV.

    The convenience the GUI wants: hand it the open-pore current in its own units
    and the applied voltage **in mV** (the app's native voltage unit), get back
    nS. ``None`` when the voltage is zero/non-finite (no conductance) or the
    current units are unrecognised (nS undefined).

    Args:
        i_open: Open-pore current (signed) in ``current_units``.
        current_units: Units of ``i_open`` (``"pA"``, ``"nA"``, ...).
        voltage_mV: Applied voltage in millivolts (signed).

    Returns:
        Conductance in nS, or ``None``.
    """
    ratio = conductance(i_open, voltage_mV)
    if ratio is None:
        return None
    return to_nS(ratio, current_units, "mV")


def _unit_clause(kind: str, units: str) -> str:
    """How an unusable unit is named in every refusal this module writes.

    The value's own spelling when it has one, else the Calibration panel's
    ``(unknown)`` -- one spelling, so a user matching a refusal against the panel's
    units row is matching the same word.
    """
    return f"{kind} units {units or '(unknown)'}"


_NOT_IN_nS = "-> conductance not in nS"


def conductance_note(current_units: str, voltage_units: str = "mV") -> str:
    """A one-line human note on the unit scaling, for a panel/read-out caption.

    Returns text like ``"pA / mV -> nS"`` when both units are known, so the UI can
    say WHY a nS value is or isn't shown instead of a bare dash.

    NAMES EVERY UNIT THAT IS BLOCKING, not just the first. A nS conductance needs
    BOTH units, and a freshly opened HDF5/CSV declares neither (io/hdf5.py and
    io/text.py leave ``raw_units=""`` on every channel; docs/USER_GUIDE.md line 88
    says so and tells the user to set them in Calibration). Stopping at the current
    units sent that user to fill in one box, after which the note flipped to the
    other one -- a refusal has to list the whole bill.
    """
    cu = normalise_units(current_units)
    vu = normalise_units(voltage_units)
    missing = []
    if cu not in _I_TO_A:
        missing.append(_unit_clause("current", current_units))
    if vu not in _V_TO_V:
        missing.append(_unit_clause("voltage", voltage_units))
    if missing:
        return f"{' and '.join(missing)} {_NOT_IN_nS}"
    return f"{cu} / {vu} -> nS"


def voltage_units_clause(voltage_units: str) -> str:
    """NAMES the blocking voltage-units state, with no consequence attached, or ``""``.

    The half of :func:`voltage_units_note` that every surface shares. The note below
    spells out the I-V window's consequence ("conductance not in nS"); the
    voltage-dependence series has a different one (a point that cannot be placed on a
    volts axis, no conductance involved). Both must call the STATE by the same name --
    the same "(unknown)" the Calibration panel prints -- so a user matching one
    refusal against the other is matching the same word, which is why the spelling
    lives here once instead of being re-worded per surface.

    Returns:
        ``"voltage units (unknown)"`` (or the value's own spelling), or ``""`` when
        the units ARE usable and there is nothing to refuse.
    """
    if normalise_units(voltage_units) in _V_TO_V:
        return ""
    return _unit_clause("voltage", voltage_units)


def voltage_units_note(voltage_units: str) -> str:
    """The refusal wording for a surface whose gate is the VOLTAGE units ALONE.

    :func:`conductance_note` answers "why is there no nS number", which both units
    can block. A different question -- "why is this recording not on the I-V axes"
    (:func:`iv_exclusions`, whose gate is :func:`_applied_voltage_mV`) -- has only
    one possible answer, and it must not borrow a reason that may name the other
    channel: the I-V window used to refuse a freshly opened HDF5 with "current
    units (unknown)", and a user who declared the current units as instructed was
    still excluded (JHT-02). The current units never exclude anything; a point with
    undeclared current units plots fine and merely has no nS conductance.

    The wording is deliberately identical to :func:`conductance_note`'s voltage
    clause, so this refusal and the Calibration panel's note remain one voice.

    Returns:
        The reason, or ``""`` when ``voltage_units`` IS usable (nothing to refuse).
    """
    clause = voltage_units_clause(voltage_units)
    return f"{clause} {_NOT_IN_nS}" if clause else ""


# ---------------------------------------------------------------------------
# per-recording I-V point
# ---------------------------------------------------------------------------


class IVPoint(NamedTuple):
    """One ``(voltage, open-pore current)`` sample for the I-V characterisation.

    Unpacks as ``v_volts, i_open, current_units, ...`` for the terse call sites
    while still carrying the provenance the I-V window's table wants.

    Attributes:
        v_volts: Applied voltage in **volts** (signed).
        i_open: Open-pore current in ``current_units`` (signed).
        current_units: Units of ``i_open`` (``"pA"``/``"nA"``/``""`` if unknown).
        voltage_source: ``"measured"`` when the voltage came from a VOLTAGE
            channel's median, ``"nominal"`` when it came from
            ``nominal_voltage_mV`` (filename/header) -- less trustworthy.
        doc_id: The source document id.
        label: The document's display name, for the window's per-point table.
    """

    v_volts: float
    i_open: float
    current_units: str
    voltage_source: str
    doc_id: str
    label: str


def _current_channel(session: "Session", doc_id: str) -> int | None:
    """The document's live CURRENT channel index, or ``None`` -- BY EFFECTIVE KIND.

    ``Session.resolve_current_channel(doc_id, None)``: the recording's first channel
    whose effective kind (the user's Calibration-panel override when there is one,
    else the header) is CURRENT and that is not flagged dead. This used to read
    ``meta.channels[i].kind`` -- the header alone -- and was the last resolver still
    on it after sweep 7 moved the chip, the workflow rail, the Detect gate, the batch
    picker, the Calibration panel and the PSD window to the session's answer.

    Measured (sweep-7 X2, a 2 s CSV of the owner's -700 mV recording with neutral
    column headers; ch0 corrected to current / pA / x2.5 and ch1 to voltage / mV
    through the real panel's Apply; Detect stored 15 events on ch0): Analysis > I-V /
    Conductance... was ENABLED -- ``MainWindow._has_iv_data`` asks the effective kind
    -- while ``iv_point`` was ``None`` with the refusal "no live current channel",
    ``applied_voltage_readout`` fell back to the FILENAME nominal (700 mV) instead of
    the corrected ch1, and the window it opened read "No usable recordings." The
    voltage-dependence series, the I-V window and its exclusion list all resolve
    here, so all three denied the recording together. Resolving through the session
    is what makes the reverse override (a header-current channel re-declared as
    voltage) empty this window for the same reason it greys the menu item out.
    """
    return session.resolve_current_channel(doc_id, None)


def _voltage_channel(session: "Session", doc_id: str) -> object | None:
    """The document's first channel whose EFFECTIVE kind is VOLTAGE (its
    ``ChannelInfo``), or ``None``.

    The same walk the Calibration panel's ``_resolve_partners`` makes for its
    Command-voltage box, so the voltage this module divides by is the one that box
    shows -- a channel corrected to voltage / mV in the panel is the command channel
    here too, and one re-declared away from voltage stops being it. Dead-ness is the
    caller's business, unchanged: a dead first voltage channel still falls back to
    the nominal exactly as it always did, so no number moves for a header-declared
    recording.
    """
    meta = session.document(doc_id).meta
    return next((c for c in meta.channels
                 if session.effective_kind(doc_id, c.index) is ChannelKind.VOLTAGE), None)


class AppliedVoltage(NamedTuple):
    """The applied-voltage read-out in the channel's OWN calibrated units.

    This is the same number the Calibration panel's conductance box shows
    (``_readout_applied_voltage``): the measured command-channel median with
    whatever units the calibration actually carries, or the filename/header
    nominal. Units ``""`` == undeclared -- the loaders (io/hdf5.py, CSV) leave
    them blank on purpose and docs/USER_GUIDE.md says the user sets them in
    Calibration -- so a consumer must not label the value mV or divide it into
    a nS conductance (the CF-5 contract).

    Attributes:
        value: The voltage read-out (signed), in ``units``.
        units: The channel's calibrated units (``"mV"``/``"V"``/``""`` undeclared).
        source: ``"measured"`` (VOLTAGE-channel median) or ``"nominal"``
            (filename/header ``nominal_voltage_mV``, mV by that field's contract).
    """

    value: float
    units: str
    source: str


def applied_voltage_readout(session: "Session", doc_id: str) -> AppliedVoltage | None:
    """The applied voltage as recorded -- value, units, source -- or ``None``.

    Prefers the MEASURED command voltage -- the median of a VOLTAGE channel, taken
    on the calibrated data so the ABF 10x gain is already applied -- and falls
    back to the filename/header ``nominal_voltage_mV`` only when the recording
    carries no voltage channel (HDF5/CSV current-only files). No unit is ever
    inferred: the value comes back in the channel's own calibrated units, blank
    when the file never declared any.

    "A VOLTAGE channel" is by EFFECTIVE kind (:func:`_voltage_channel`), not the
    header: on the undeclared CSV of sweep-7 X2, with ch1 corrected to voltage / mV
    in the Calibration panel, this read-out answered the filename nominal
    ``(700.0, "mV", "nominal")`` while the panel's own Command-voltage box showed
    the corrected channel's median -- the same document, two applied voltages.
    """
    doc = session.document(doc_id)
    meta = doc.meta
    volt_info = _voltage_channel(session, doc_id)
    if volt_info is not None and not volt_info.is_dead:
        # Memoised per (document, channel, calibration) -- curation cannot move a
        # command-voltage median, and this read-out is on the curation-refresh path
        # TWICE per I-V refresh (iv_points and iv_exclusions both ask). See END-003.
        med, units = _voltage_median_of(session, doc_id, volt_info.index)
        return AppliedVoltage(value=med, units=units, source="measured")
    if meta.nominal_voltage_mV is not None:
        return AppliedVoltage(value=float(meta.nominal_voltage_mV), units="mV",
                              source="nominal")
    return None


def _volts_per(voltage_units: str) -> float | None:
    """Volts per one unit of ``voltage_units``, or ``None`` when it is not a voltage unit.

    The ONE reading of :data:`_V_TO_V` for the "can this be called a voltage at all"
    question, shared by :func:`_applied_voltage_mV` and :func:`_iv_gate` so the
    resolver and the refusal cannot answer it differently for one channel.
    """
    return _V_TO_V.get(normalise_units(voltage_units))


def _applied_voltage_mV(session: "Session", doc_id: str) -> tuple[float, str] | None:
    """Applied voltage in mV plus its source, or ``None`` when unknown.

    A thin unit conversion over :func:`applied_voltage_readout`: declared mV
    passes through, declared V is scaled by 1000 (the ABF path), and a voltage
    whose units are UNDECLARED (``""``, or any spelling ``_V_TO_V`` does not
    recognise) is refused with ``None`` -- the number cannot be asserted to be
    millivolts, and the Calibration panel already refuses exactly that claim for
    the same document (CF-5: "units unknown", conductance "--"). Fabricating mV
    here would put the two surfaces in contradiction; use
    :func:`applied_voltage_readout` when you want the value with its honest units.
    """
    got = applied_voltage_readout(session, doc_id)
    if got is None:
        return None
    volts_per = _volts_per(got.units)
    if volts_per is None:
        return None
    return got.value * volts_per * 1000.0, got.source


#: Public name for the resolver above. It is the ONE place that answers "what bias was this
#: recording taken at", and callers outside this module need it: the density figure used to
#: carry a private half-copy that looked only at a VOLTAGE channel, so every current-only CSV
#: (which is all of the owner's apo-ferritin exports) fell through to a filename stub and the
#: legend read "6_Process_-6" instead of "-600 mV". It answers ONLY in mV: for a
#: measured voltage with undeclared units it returns ``None`` (never a value it
#: would have to mislabel), so callers fall back honestly -- the density legend to
#: its filename stub, the I-V window to :func:`iv_exclusions` below.
applied_voltage_mV = _applied_voltage_mV


class IVExclusion(NamedTuple):
    """A recording refused from the I-V fit because its voltage units are undeclared.

    The I-V axes are physical (volts, and a slope in nS), so a measured voltage
    whose units the file never declared cannot be placed on them -- but the
    refusal must not be silent: the window lists these recordings with the value
    in its honest form and the same reason string the Calibration panel shows
    (CF-5), so the two surfaces speak with one voice.

    Attributes:
        doc_id: The refused document.
        label: Its display name, for the window's list.
        v_value: The measured command-channel median, in ``v_units``.
        v_units: The voltage channel's calibrated units -- ``""`` == undeclared.
        i_open: The open-pore current (median of the calibrated CURRENT channel).
        current_units: Units of ``i_open``.
        reason: Why THIS recording is refused -- :func:`voltage_units_note`, e.g.
            ``"voltage units (unknown) -> conductance not in nS"``. It names the
            voltage and only the voltage, because the voltage units are the whole
            gate: the reason a refusal carries must be the thing the user has to
            change to clear it. (The current units may be undeclared too; that costs
            the nS conductance, not the point, and :func:`conductance_note` on the
            Calibration panel says so there.)
    """

    doc_id: str
    label: str
    v_value: float
    v_units: str
    i_open: float
    current_units: str
    reason: str


def iv_exclusions(session: "Session") -> list[IVExclusion]:
    """Recordings that WOULD contribute an I-V point but have unusable voltage units.

    The complement of :func:`iv_points` for exactly one cause: the document has a
    live current channel and a measured applied voltage, but that voltage's units
    are undeclared/unrecognised, so no honest mV (and hence no point in volts) can
    be formed. Other reasons for skipping a document -- no current channel, no
    voltage at all -- stay silent as before; this cause must not, because the
    recording LOOKS complete and the fix (declare the units in Calibration) is
    the user's to make.
    """
    out: list[IVExclusion] = []
    for doc in session.documents():
        if not isinstance(doc, TraceDocument):
            continue
        cur_ch = _current_channel(session, doc.doc_id)
        if cur_ch is None:
            continue
        got = applied_voltage_readout(session, doc.doc_id)
        if got is None or normalise_units(got.units) in _V_TO_V:
            continue   # no voltage at all, or a usable one -- not this refusal
        i_open, cur_units = _open_pore_of(session, doc.doc_id, cur_ch)
        out.append(IVExclusion(
            doc_id=doc.doc_id,
            label=doc.display_name,
            v_value=got.value,
            v_units=got.units,
            i_open=i_open,
            current_units=cur_units,
            # The gate two lines up is `got.units` alone, so the reason names the
            # voltage alone -- never the current units, which do not exclude anything.
            reason=voltage_units_note(got.units),
        ))
    return out


#: The causes an ``(V, I₀)`` point cannot be formed, in the order the gate asks them.
#: Public because a pooled surface has to NAME the recording it lost and say why:
#: :func:`iv_exclusions` is scoped to the one cause the I-V window lists (undeclared
#: voltage units, the cause with a fix the user can act on), while the
#: voltage-dependence series names all of them -- nothing may leave a pooled fit
#: unnamed. Both read them from :func:`iv_point_refusal`, so the two windows cannot
#: come to different conclusions about the same recording.
IV_NOT_A_TRACE = "not a trace recording — it carries no signal to take an open-pore level from"
IV_NO_CURRENT_CHANNEL = "no live current channel"
IV_NO_VOLTAGE = ("no applied voltage — the recording has no voltage channel, and no "
                 "nominal voltage in its filename or header")


def _iv_gate(session: "Session", doc_id: str) -> tuple[IVPoint | None, str]:
    """``(point, "")`` when a document contributes one, else ``(None, reason)``.

    ONE code path for the point and for the reason there is no point, so a surface
    that lists refusals can never disagree with the surface that plots the points --
    the two are the same walk down the same three gates.
    """
    doc = session.document(doc_id)
    if not isinstance(doc, TraceDocument):
        return None, IV_NOT_A_TRACE
    cur_ch = _current_channel(session, doc_id)
    if cur_ch is None:
        return None, IV_NO_CURRENT_CHANNEL
    got = applied_voltage_readout(session, doc_id)
    if got is None:
        return None, IV_NO_VOLTAGE
    volts_per = _volts_per(got.units)
    if volts_per is None:
        # The one refusal with a remedy the user owns; name the state the way every
        # other surface names it (voltage_units_clause) and say where the fix lives.
        return None, (f"{voltage_units_clause(got.units)} — declare them in "
                      f"Calibration so this recording can go on a volts axis")
    v_mV = got.value * volts_per * 1000.0
    i_open, cur_units = _open_pore_of(session, doc_id, cur_ch)
    return IVPoint(
        v_volts=v_mV / 1000.0,
        i_open=i_open,
        current_units=cur_units,
        voltage_source=got.source,
        doc_id=doc_id,
        label=doc.display_name,
    ), ""


def iv_point(session: "Session", doc_id: str) -> IVPoint | None:
    """One ``IVPoint`` for a document, or ``None`` when it cannot contribute.

    Builds the pair the I-V fit needs: the applied voltage (measured VOLTAGE
    channel median, else nominal) converted to **volts**, and the open-pore
    current as the median of the **calibrated** CURRENT channel (so the current
    gain the user is dialing is reflected). Returns ``None`` -- never a fabricated
    point -- when the document is not a trace, has no live current channel, or has
    no known voltage. :func:`iv_point_refusal` is the twin that says WHICH.

    Which channel is "current" and which is "voltage" is the SESSION's answer
    (``Session.resolve_current_channel`` / ``Session.effective_kind``): the user's
    Calibration-panel override when there is one, else the header -- the same
    resolver the menu item that opens the I-V window is gated by.

    Args:
        session: The owning :class:`~nanoscope.core.session.Session` (duck-typed;
            not imported, to keep this module Qt-free; it must answer
            ``document``, ``calibrated``, ``effective_kind`` and
            ``resolve_current_channel``).
        doc_id: The document to sample.

    Returns:
        An :class:`IVPoint`, or ``None``.
    """
    return _iv_gate(session, doc_id)[0]


def iv_point_refusal(session: "Session", doc_id: str) -> str:
    """Why :func:`iv_point` returns ``None`` here — ``""`` when it returns a point.

    The reason in one sentence, already phrased for a reader: one of
    :data:`IV_NOT_A_TRACE`, :data:`IV_NO_CURRENT_CHANNEL`, :data:`IV_NO_VOLTAGE`, or
    the undeclared-voltage-units refusal with the fix attached. It exists so a POOLED
    surface can name what it lost: ``voltage_dep.vdep_series`` used to consume this
    module's ``None`` with a bare ``continue``, so a recording the user had opened,
    calibrated and detected on vanished from the voltage series while the I-V window,
    reading the identical refusal from the identical resolver, listed it with its
    value and its remedy.
    """
    return _iv_gate(session, doc_id)[1]


def iv_points(session: "Session") -> list[IVPoint]:
    """Every usable :class:`IVPoint` across the session's open documents.

    Skips documents that cannot contribute a point (no current channel, no known
    voltage); the order follows :meth:`Session.documents`.
    """
    out: list[IVPoint] = []
    for doc in session.documents():
        pt = iv_point(session, doc.doc_id)
        if pt is not None:
            out.append(pt)
    return out


# ---------------------------------------------------------------------------
# I-V fit -- slope IS the pore conductance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IVFit:
    """A straight-line fit of open-pore current against applied voltage.

    The line is ``I = slope * V + intercept`` with ``V`` in volts and ``I`` in
    ``current_units``. Physically:

    * ``slope`` is the **pore conductance** in ``current_units / V`` (its value in
      nanosiemens is :attr:`conductance_nS`);
    * ``intercept`` is the **leak current** at ``V = 0`` in ``current_units`` -- a
      non-zero value is an offset/asymmetry in the pore or amplifier.

    Attributes:
        slope: Conductance in ``current_units`` per volt.
        intercept: Leak current at 0 V, in ``current_units``.
        r2: Coefficient of determination of the linear fit.
        n: Number of ``(V, I)`` points fit.
        current_units: Units of the current axis (``"pA"``/``"nA"``/``""``).
        conductance_nS: :attr:`slope` expressed in nanosiemens, or ``None`` when
            ``current_units`` is unrecognised (nS undefined).
    """

    slope: float
    intercept: float
    r2: float
    n: int
    current_units: str = ""
    conductance_nS: float | None = None

    def predict(self, v_volts: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted current over voltages (in volts), in current units."""
        v = np.asarray(v_volts, dtype=np.float64)
        return self.slope * v + self.intercept

    def __repr__(self) -> str:
        g = "?" if self.conductance_nS is None else f"{self.conductance_nS:.4g} nS"
        return (f"<IVFit G={g} slope={self.slope:.4g} {self.current_units}/V "
                f"leak={self.intercept:.4g} {self.current_units} r2={self.r2:.3f} n={self.n}>")


def iv_fit(voltages_V, currents, *, current_units: str = "") -> IVFit:
    """Fit ``I`` vs ``V`` and report the pore conductance (the slope).

    A thin, physics-labelled wrapper over
    :func:`nanoscope.analysis.populations.linear_fit`: the least-squares line
    through ``(V, I)`` is exactly the I-V characterisation, its slope is the
    conductance and its intercept the leak. ``voltages_V`` must be in **volts** so
    the slope comes out in ``current_units / V`` and converts to nS with the same
    :func:`to_nS` scale everything else uses.

    Args:
        voltages_V: Applied voltages in volts (array-like).
        currents: Open-pore currents in ``current_units`` (array-like, same
            length as ``voltages_V``).
        current_units: Units of ``currents`` (``"pA"``/``"nA"``/...); when known,
            the slope is also reported in nanosiemens.

    Returns:
        An :class:`IVFit`.

    Raises:
        ValueError: propagated from ``linear_fit`` -- fewer than 2 points, or a
            degenerate voltage axis (every recording at the same voltage, so the
            slope is undefined). A single-point I-V is not a fit; the caller (the
            window) catches this and asks for another voltage.
    """
    lf = linear_fit(voltages_V, currents, xlabel="V (V)", ylabel=f"I ({current_units})")
    g_nS = to_nS(lf.slope, current_units, "V")
    return IVFit(
        slope=lf.slope,
        intercept=lf.intercept,
        r2=lf.r2,
        n=lf.n,
        current_units=current_units,
        conductance_nS=g_nS,
    )
