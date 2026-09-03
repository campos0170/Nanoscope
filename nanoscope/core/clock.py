"""WHICH CLOCK a number in seconds is on. The app's one answer, in one place.

THE DEFECT THIS EXISTS TO CLOSE
===============================
NanoScope carries **two** time origins and used to label both of them ``s``:

* the **recording clock** -- 0 at the first sample that was LOADED. Every
  ``t_start_s`` / ``t_end_s`` in the event table is on it, every window a panel takes
  in seconds (the PSD segment, a merge region expressed as ``i/fs``) is on it, and
  ``analysis`` never sees any other.
* the **instrument clock** -- ``RecordingMeta.t0_s`` at that same first sample, i.e.
  the experimental time the acquisition software wrote into the file. The trace
  x-axis is drawn on it (``Session.x_axis`` = ``arange(n)/fs + meta.t0_s``), because
  ``io.contract`` is deliberate that ``t0`` is read from the file and never
  fabricated.

On the owner's own apoferritin recordings ``t0_s`` is 10.2526 s, 61.3034 s and 0 s.
Reading a quiet stretch off the trace at -500 mV (an instrument-clock number) and
typing it into a panel that wanted a recording-clock number analysed a *different*
stretch and returned **total RMS 22.66 pA / alpha 1.822** where the right window gives
**3.99 pA / alpha 0.985** -- a 5.7x error in a headline noise number, with nothing on
screen to say anything had gone wrong. The -700 mV file has ``t0_s = 0``, so the two
clocks agree there and the mistake is invisible on two recordings out of three.

THE RULE
========
**One clock, or every control that shows or takes a time says which clock it is in.**

The app cannot have one clock: dropping ``t0_s`` would throw away the experimental
time the file carries (``io/contract.py`` documents why that is a bug, not a
simplification), and rebasing the event table onto the instrument clock would put an
offset into 20 frozen schema columns and into every stored detection ever written.
So it is the second half, and this module is what makes it cheap enough that no
surface can plead ignorance:

* :func:`clock_offset_s` -- the ONE conversion constant, read from the document.
* :func:`axis_label` / :func:`field_note` -- the label text a widget must show.
* :func:`both_clocks` -- a readout that states an instant on both clocks at once, so
  the number a user copies is next to the number the other surface wants.

THE THIRD CASE THE FIRST VERSION MISSED: A **REBASED** DOCUMENT
================================================================
``t0_s`` is read from the file and never fabricated -- for a document that came from a
file. ``core.segment`` makes documents that did not: a segment cut from an open
recording is a DERIVED recording whose own axis is rebased so its first sample is at
``t0_s = 0`` (``tests/core/test_segment.py`` pins that, and every panel that runs on a
segment depends on it). Its ``0`` is therefore not a file's declared zero; it is a
zero this app made, and the instrument time of that same sample -- 1158.9909 s on the
owner's quick apoferritin cut -- is still known and still printed by the Files dock,
the segment's name and its File Info notes.

Read through ``meta.t0_s`` alone, such a document is indistinguishable from the
-700 mV file whose t₀ genuinely IS 0, so :func:`clocks_agree` said "these clocks
agree", :func:`axis_label` said "t₀ = 0, so both clocks read alike here" and
:func:`field_note` said "This recording's t₀ is 0" -- on a window that names the very
same data ``[1159-1165 s]``. Four surfaces, one number, and the three built to prevent
a clock error were the wrong ones.

Both halves of that stay true and neither is enough on its own:

* the two clocks this module distinguishes (the drawn axis and the event table) DO
  read alike on a rebased document -- both start at 0 -- so :func:`clocks_agree` is
  right to keep saying so, and the badge/branching every caller does off it is right;
* and NEITHER of them is the instrument clock, which is where the parent recording,
  the segment's name, its notes and the Files dock all still are.

So the offset a caller gets back carries the missing number with it. :func:`clock_offset_s`
returns a :class:`ClockOffset` -- a ``float`` that IS the offset, so every existing
arithmetic caller is untouched -- with :attr:`ClockOffset.rebased_from_s` set to the
instrument-clock instant of the document's sample 0 when the document was rebased.
:func:`axis_label`, :func:`field_note` and :func:`both_clocks` then have a third branch
and cannot omit it, because the number reaches them through the ONE seam every surface
already goes through. A producer of rebased documents opts in by putting that instant on
its meta under :data:`REBASE_ATTR` (``core.segment`` does; this module imports nothing to
read it).

AND THE CORROLLARY, which is what actually stops the 5.7x error rather than merely
annotating it: **a time that crosses a surface boundary travels as SAMPLE INDICES,
never as seconds.** ``TraceView.measure_bounds()`` already returns ``(start_idx,
stop_idx)``; :data:`TRANSFER_RULE` is the sentence, and ``MainWindow`` is where a
selection made on the trace is handed to a panel with the conversion done in code
instead of in the user's head.

This module holds no state, imports no sibling gui module, and defines no widget.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RECORDING",
    "INSTRUMENT",
    "CLOCK_NAMES",
    "CLOCK_SHORT",
    "TRANSFER_RULE",
    "REBASE_ATTR",
    "ClockOffset",
    "rebased_origin_from_meta",
    "clock_offset_s",
    "clocks_agree",
    "instrument_origin_s",
    "rebased_origin_s",
    "to_recording_s",
    "to_instrument_s",
    "format_t0",
    "axis_label",
    "field_note",
    "both_clocks",
]

#: 0 at the first LOADED sample. The event table, every analysis window in seconds.
RECORDING = "recording"

#: ``meta.t0_s`` at the first loaded sample. The trace x-axis, and only the trace.
INSTRUMENT = "instrument"

CLOCK_NAMES = {
    RECORDING: "seconds from the start of this recording",
    INSTRUMENT: "seconds on the instrument clock stored in the file",
}

#: The short form that fits in an axis label or a column header.
CLOCK_SHORT = {
    RECORDING: "from recording start",
    INSTRUMENT: "instrument clock",
}

TRANSFER_RULE = (
    "A time handed from one surface to another travels as sample indices, never as "
    "seconds: the two surfaces may be on different clocks and an index is on neither."
)

#: Two clocks are "the same clock" below this. Not an epsilon on a measurement -- it
#: is the difference between a file that declares t0 = 0 and one that does not, and
#: float64 seconds carry ~1e-10 s of resolution at the 1e5 s an instrument clock can
#: reach.
_AGREE_TOL_S = 1e-9

#: The attribute name a REBASED document's ``meta`` carries: the instrument-clock
#: instant of ITS OWN sample 0, i.e. the origin the rebase to ``t0_s = 0`` threw away.
#: A plain file meta does not have it and never should — reading it back off a file
#: recording would be exactly the fabrication ``io.contract`` forbids. This module names
#: the attribute and imports nothing to find it; ``core.segment`` is the one producer.
REBASE_ATTR = "rebased_from_t0_s"


class ClockOffset(float):
    """The clock offset, PLUS the origin a rebase removed. Numerically just the offset.

    Subclassing ``float`` rather than returning a pair is what makes the third case
    (see the module docstring) reach every surface without a single caller changing:
    ``TraceView.clock_offset_for`` still hands MainWindow a number it subtracts from an
    on-trace x, ``clocks_agree`` still takes ``abs()`` of it, an f-string still formats
    it. Only this module's own label writers and clock-name conversions look at
    :attr:`rebased_from_s` — which is the point: they are the surfaces that were lying,
    and they now cannot be handed the offset without also being handed the origin.

    :attr:`rebased_from_s` is ``None`` for every document whose ``t0_s`` came from a
    file, and the instrument-clock instant of sample 0 for a document this app rebased.

    EQUALITY IS BY CLOCK STATEMENT, NOT BY NUMBER, and that is load-bearing rather than
    tidy: two panels guard their repaint with ``if offset_s == self._clock_offset_s:
    return``. Selecting a segment of a t₀ = 1158.99 s recording after a genuine t₀ = 0
    file gives 0.0 both times, so a plain-float equality would skip the repaint and
    leave the header saying "this recording's t₀ is 0" about the wrong recording —
    stale state, which is the defect this whole module exists to close, one level up.
    """

    __slots__ = ("rebased_from_s",)

    def __new__(cls, value: float, rebased_from_s: float | None = None) -> "ClockOffset":
        self = super().__new__(cls, value)
        # object.__setattr__ is not needed (float is not frozen, only immutable in value);
        # a plain assignment onto the __slots__ entry is what stores the provenance.
        self.rebased_from_s = (None if rebased_from_s is None else float(rebased_from_s))
        return self

    def __eq__(self, other: object) -> bool:
        same_number = float.__eq__(self, other)
        if same_number is NotImplemented:
            return NotImplemented          # let pytest.approx & friends answer
        if not same_number:
            return False
        return rebased_origin_s(other) == self.rebased_from_s

    def __ne__(self, other: object) -> bool:
        # Spelled out because ``float`` already DEFINES ``__ne__``, so the usual "Python
        # derives != from ==" does not apply here: without this, ``a != b`` would go back
        # to comparing the bare numbers and the repaint guards would answer differently
        # depending on which operator the call site happened to use.
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    # Defining __eq__ would otherwise make this unhashable. Two offsets that differ only
    # in their rebase origin then share a hash, which is allowed (and never happens in a
    # dict key in this app; the offsets live in plain attributes).
    __hash__ = float.__hash__

    def __repr__(self) -> str:            # a debugger/log line should show the provenance
        if self.rebased_from_s is None:
            return f"ClockOffset({float(self)!r})"
        return f"ClockOffset({float(self)!r}, rebased_from_s={self.rebased_from_s!r})"


def rebased_origin_s(offset_s: Any) -> float | None:
    """The instrument-clock origin carried on ``offset_s``, or None.

    Safe on a plain ``float``, on ``None`` and on anything else a caller happens to
    hold, so a surface can ask "was this document rebased?" without first having to
    know which producer made it.
    """
    origin = getattr(offset_s, "rebased_from_s", None)
    return None if origin is None else float(origin)


def instrument_origin_s(meta: Any) -> float | None:
    """The experimental time of this document's sample 0 **on the instrument clock**.

    THE ONE ANSWER to "where did this recording's first sample sit in the experiment",
    and the value a provenance stamp must carry so that rows pooled from several
    recordings can be read back onto one clock. For a file recording that is
    ``meta.t0_s``. For a REBASED document (:data:`REBASE_ATTR`) ``meta.t0_s`` is the
    zero this app made, and the answer is the origin the rebase recorded — 1158.9909 s
    for a segment cut 4 s into a recording that started at 1154.9909 s.

    None on a recording with no sample rate, for the same reason as
    :func:`clock_offset_s`: no time base, no instant, and inventing one is the bug.

    NOT the same question as :func:`clock_offset_s`, which answers "how far apart are
    the two clocks ON SCREEN" and therefore includes the layer's display nudge and is 0
    on a rebased document. A caller that wants to say where a number sits in the
    EXPERIMENT wants this one.
    """
    if getattr(meta, "sample_rate_hz", None) is None:
        return None
    rebased = getattr(meta, REBASE_ATTR, None)
    if rebased is not None:
        try:
            return float(rebased)
        except (TypeError, ValueError):
            return None
    t0 = getattr(meta, "t0_s", None)
    if t0 is None:
        return None
    try:
        return float(t0)
    except (TypeError, ValueError):
        return None


def clock_offset_s(meta: Any, time_offset_s: float = 0.0) -> ClockOffset | None:
    """``instrument_seconds - recording_seconds`` for this document, or None.

    None means **there is no clock here at all**: a recording with no known sample
    rate is drawn against SAMPLES (``Session.x_axis`` says so and refuses to
    fabricate seconds), so neither clock exists and a caller must not invent one.

    ``time_offset_s`` is the layer's user alignment nudge, which ``Session.x_axis``
    adds on top of ``meta.t0_s``: the trace is drawn at ``t0_s + time_offset_s``, so
    that -- not ``t0_s`` alone -- is what separates what is on screen from what the
    event table holds.

    The return value is a :class:`ClockOffset` — a float, so every caller's arithmetic
    is unchanged — carrying the rebase origin when this document was rebased. That is
    the one seam every clock surface in the app already reads, which is why the missing
    number is put back HERE rather than in each of them.
    """
    if getattr(meta, "sample_rate_hz", None) is None:
        return None
    t0 = getattr(meta, "t0_s", None)
    if t0 is None:
        return None
    try:
        value = float(t0) + float(time_offset_s)
    except (TypeError, ValueError):
        return None
    return ClockOffset(value, rebased_origin_from_meta(meta))


def rebased_origin_from_meta(meta: Any) -> float | None:
    """The rebase origin declared on ``meta``, or None when it was not rebased.

    Split out from :func:`instrument_origin_s` because the two answer different
    questions: this one is "was a zero manufactured here, and from what", which is what
    a LABEL must disclose; the other is "what instant is sample 0", which every
    document has.
    """
    rebased = getattr(meta, REBASE_ATTR, None)
    if rebased is None:
        return None
    try:
        return float(rebased)
    except (TypeError, ValueError):
        return None


def clocks_agree(offset_s: float | None) -> bool:
    """True when the two clocks read the same number, so no distinction is needed.

    A file with ``t0_s = 0`` (the -700 mV apo recording is one) makes every
    clock confusion invisible. That is exactly why this returns a bool the callers
    branch on rather than something they can forget: the label still names its clock
    on such a file, it just also says the two agree.

    A REBASED document is NOT such a file and answers False even though its own two
    numbers coincide. Read the clause that matters: *so no distinction is needed*. On a
    segment cut 4 s into a 1154.9909 s recording the trace axis and the event table both
    say 2.0000 s and the rest of the app — the segment's own name ``[1159-1165 s]``, its
    File Info notes, the Files dock, the parent recording, every other recording of the
    experiment — says 1160.9909 s. A distinction is exactly what is needed, so every
    caller that branches on this (the axis label, the two table headers, the crosshair
    read-out, the PSD span read-out) must take its "say which clock this is" path. The
    numeric offset stays 0; what changes is that nobody is told there is nothing to say.
    """
    if offset_s is None:
        return False
    if not _is_zero(rebased_origin_s(offset_s)):
        return False
    return abs(float(offset_s)) < _AGREE_TOL_S


def _is_zero(value: float | None) -> bool:
    """None or indistinguishable from 0 at :data:`_AGREE_TOL_S`. ``None`` counts as zero
    because "this document was never rebased" and "it was rebased off 0" are the same
    thing to every reader of a label."""
    return value is None or abs(float(value)) < _AGREE_TOL_S


def _total_offset(offset_s: float | None) -> float | None:
    """``instrument_seconds - recording_seconds`` INCLUDING a rebase.

    :func:`clock_offset_s` answers "how far apart are the two clocks on screen", which
    is 0 on a rebased document because the rebase moved BOTH of them. The distance to
    the instrument clock is that plus the origin the rebase removed, and that is what a
    conversion between the two clock NAMES has to use — otherwise a caller converts
    2.0000 s and gets 2.0000 s back labelled "instrument".
    """
    if offset_s is None:
        return None
    origin = rebased_origin_s(offset_s)
    return float(offset_s) + (0.0 if origin is None else origin)


def to_recording_s(offset_s: float | None, instrument_s: float) -> float | None:
    """An instrument-clock instant (an x read off the trace) in recording seconds."""
    total = _total_offset(offset_s)
    if total is None:
        return None
    return float(instrument_s) - total


def to_instrument_s(offset_s: float | None, recording_s: float) -> float | None:
    """A recording-clock instant (a ``t_start_s``) in instrument seconds."""
    total = _total_offset(offset_s)
    if total is None:
        return None
    return float(recording_s) + total


def format_t0(offset_s: float | None) -> str:
    """``t0`` as the user will compare it against a spin box: fixed, four decimals.

    Four, because that is the resolution the recordings actually differ at
    (10.2526 / 61.3034 / 0.0000 s) and because a ``%g`` here would render 61.3034 as
    ``61.3`` and invite the user to type a number that is 3.4 ms wrong.
    """
    if offset_s is None:
        return "—"
    return f"{float(offset_s):.4f}"


def axis_label(unit: str, clock: str, offset_s: float | None) -> str:
    """The bottom-axis / spin-box-suffix text for a time in ``unit`` on ``clock``.

    A non-time axis (``"sample"``) is returned untouched: samples are on neither
    clock, and annotating them would be a second falsehood rather than a fix.

    A REBASED document gets its own branch, ahead of the others: its two on-screen
    numbers coincide, but the reassuring "t₀ = 0, so both clocks read alike here" is
    false about the DATA — the cut's first sample was recorded at ``rebased_from_s`` on
    the instrument clock, which is the number the same window puts in the segment's name
    and its Notes.
    """
    if unit != "s":
        return unit
    if offset_s is None:
        return unit
    short = CLOCK_SHORT.get(clock, clock)
    rebased = rebased_origin_s(offset_s)
    if not _is_zero(rebased):
        text = (f"s (this cut's own clock, rebased to t₀ = 0 — its first sample was "
                f"recorded at {format_t0(rebased)} s on the instrument clock")
        if not _is_zero(offset_s):
            # A rebased document can ALSO carry a layer alignment nudge; then the axis is
            # not even at the cut's own zero, and saying only one of the two is a
            # half-truth of exactly the kind this module exists to stop.
            text += (f", and this layer is drawn shifted by {format_t0(offset_s)} s "
                     f"for alignment")
        return text + ")"
    if clocks_agree(offset_s):
        return f"s ({short}; t₀ = 0, so both clocks read alike here)"
    if clock == INSTRUMENT:
        return (f"s ({short} — this recording starts at "
                f"t₀ = {format_t0(offset_s)} s)")
    return (f"s ({short} — add t₀ = {format_t0(offset_s)} s for the "
            f"instrument clock)")


def field_note(clock: str, offset_s: float | None) -> str:
    """One sentence for a tooltip / header, naming the clock and the conversion.

    Longer than :func:`axis_label` because a tooltip has room, and because this is
    where the *other* clock's arithmetic is spelled out: a reader who has the wrong
    number in front of them needs to be told how to get the right one, not merely
    that the one they have is on a different clock.
    """
    if offset_s is None:
        return ("This recording declares no sample rate, so its x-axis is in samples "
                "and no time in seconds is defined for it.")
    rebased = rebased_origin_s(offset_s)
    if not _is_zero(rebased):
        # The rebased branch, ahead of the two it would otherwise fall into. Two facts,
        # in this order: which conversion the reader does NOT have to do here, and the
        # one they do — because neither of this document's own numbers is the instrument
        # clock the rest of the app puts this data on.
        same = ("the trace x-axis and the event table read the same number here"
                if _is_zero(offset_s) else
                f"the trace x-axis is additionally drawn shifted by "
                f"{format_t0(offset_s)} s to align this layer with another")
        note = (f"Time {CLOCK_NAMES.get(clock, clock)}. This recording is a CUT of "
                f"another one and its clock was rebased to 0, so {same} — but neither "
                f"is the instrument clock: the cut's first sample was recorded at "
                f"t₀ = {format_t0(rebased)} s, so add {format_t0(rebased)} s to a "
                f"number here to place it in the experiment (that is the number the "
                f"segment's name, its File Info notes and the Files dock all show).")
        return note
    if clocks_agree(offset_s):
        return (f"Time {CLOCK_NAMES.get(clock, clock)}. This recording's "
                f"t₀ is 0, so the trace x-axis and the event table read the same "
                f"number — they do NOT on a recording whose t₀ is not 0.")
    if clock == INSTRUMENT:
        return (f"Time {CLOCK_NAMES[INSTRUMENT]}: the first sample of this recording "
                f"is at t₀ = {format_t0(offset_s)} s. The event table and every "
                f"analysis window in seconds are on the OTHER clock — subtract "
                f"{format_t0(offset_s)} s from a number read here before typing it "
                f"into one of those.")
    return (f"Time {CLOCK_NAMES[RECORDING]}: 0 is the first sample. The trace x-axis "
            f"is on the OTHER clock — add t₀ = {format_t0(offset_s)} s to a "
            f"number here to find it on the trace.")


def both_clocks(offset_s: float | None, *, recording_s: float | None = None,
                instrument_s: float | None = None, decimals: int = 4) -> str:
    """One instant stated on BOTH clocks, e.g. ``10.2526 s in · 20.5052 s instrument``.

    Give exactly one of ``recording_s`` / ``instrument_s``; the other is derived.
    When the two clocks agree (or there is no clock) this collapses to the single
    number rather than printing it twice.

    On a REBASED document the two on-screen numbers coincide, but the instrument clock
    is elsewhere and :func:`clocks_agree` is False there, so this does NOT collapse:
    the second figure is the instant IN THE EXPERIMENT — the number the Files dock, the
    segment's name and its notes are on. That comes out of :func:`to_instrument_s`,
    which carries the rebase, so this function needs no branch of its own.
    """
    if recording_s is None and instrument_s is None:
        raise ValueError("both_clocks needs one of recording_s / instrument_s")
    if recording_s is not None and instrument_s is not None:
        raise ValueError("both_clocks takes one instant, not two")
    if offset_s is None:
        val = recording_s if recording_s is not None else instrument_s
        return f"{float(val):.{decimals}f}"
    if recording_s is None:
        recording_s = to_recording_s(offset_s, float(instrument_s))  # type: ignore[arg-type]
    else:
        instrument_s = to_instrument_s(offset_s, float(recording_s))
    if clocks_agree(offset_s):
        return f"{float(recording_s):.{decimals}f} s"
    return (f"{float(recording_s):.{decimals}f} s in  ·  "
            f"{float(instrument_s):.{decimals}f} s instrument")
