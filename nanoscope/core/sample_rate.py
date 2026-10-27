"""A user's declaration of a recording's time base: its time column's unit, or its rate.

A sample rate this app MEASURES from a file's time column is only as good as that column's
unit, and a CSV (or an HDF5 ``/time``) declares none unless its header names one; a legacy
TDMS declares no rate at all. Each leaves the user holding a fact the file does not — *the
column is in milliseconds*, or *this was acquired at 10 kHz* — and until this existed the only
remedy was to re-export the recording. [MEASURED] a 'time,current' export at 100 kHz whose
column was in ms loaded as 100 Hz with an empty Issues box; nothing anywhere could correct it.

This object is that fact, stated once, in the user's own terms, and RESOLVED against the time
base as the reader read it (:class:`TimeBaseAsRead`) — never typed over the reader's numbers,
so "use the file's own time base" is always one step away. Two modes, because the two facts
are different things:

* ``time_unit`` — the file's time column is in this unit. The rate AND the start time move
  together (``rate · scale_read / scale``, ``t0 · scale / scale_read``), so the recording's
  clock stays self-consistent. It re-reads a rate that was MEASURED from a column — or one
  the reader measured and refused as implausible (``refused_rate_hz``), which is exactly the
  microsecond column this mode exists for — and is refused for a rate the file declared in
  a header (there is no column to re-read) or a file with no time base at all.
* ``sample_rate`` — the recording was acquired at this many hertz. The start time is left as
  read, except for a column the reader refused as a clock: a t₀ read in an unknown unit is
  not "as read", so it becomes 0. The only option for a file that carries no time column.

Kept by :class:`~nanoscope.core.session.Session` (never on the frozen document — the document
is REBUILT so every consumer sees the declared rate), saved with the project by
:mod:`nanoscope.core.sample_rate_state`, and never written beside a raw recording. It is
rendered as :attr:`~nanoscope.core.types.Provenance.USER` wherever a rate's provenance is
shown, with the file's own reading kept on screen beside it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, Mapping, NamedTuple

from .types import Provenance

__all__ = ["MODES", "TIME_UNITS", "SampleRateDeclaration", "TimeBaseAsRead"]

#: unit key -> (name for a sentence, seconds per unit). The keys are what the project stores.
TIME_UNITS: dict[str, tuple[str, float]] = {
    "s": ("seconds", 1.0),
    "ms": ("milliseconds", 1e-3),
    "us": ("microseconds", 1e-6),
    "ns": ("nanoseconds", 1e-9),
}
MODES: tuple[str, ...] = ("time_unit", "sample_rate")


class TimeBaseAsRead(NamedTuple):
    """What the reader (or, for a segment, the cut) said about a document's clock, kept by the
    Session from the moment the document is added so a declaration can always be undone."""
    sample_rate_hz: float | None
    rate_provenance: Provenance
    t0_s: float
    time_scale_s: float
    refused_rate_hz: float | None = None

    @property
    def can_declare_unit(self) -> bool:
        """A unit re-reads a rate MEASURED from a column (or refused from one); a rate a
        header declared (ABF, TDMS) has no column behind it, and a file with no time base
        has nothing to re-read."""
        return self.refused_rate_hz is not None or (
            self.sample_rate_hz is not None and self.rate_provenance is Provenance.MEASURED)


@dataclass(frozen=True)
class SampleRateDeclaration:
    mode: str
    time_unit: str | None = None       # mode "time_unit": a TIME_UNITS key
    sample_rate_hz: float | None = None  # mode "sample_rate": hertz, finite and positive
    note: str = ""                       # where the user knows this from
    source: str = "user"                 # the only value accepted; mirrors AcquisitionBandwidth

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown declaration mode {self.mode!r}; one of {', '.join(MODES)}")
        if self.source != "user":
            raise ValueError("a sample-rate declaration is user-provided metadata")
        if not isinstance(self.note, str):
            raise ValueError("the declaration note must be text")
        object.__setattr__(self, "note", self.note.strip())
        if self.mode == "time_unit":
            if self.time_unit not in TIME_UNITS:
                raise ValueError(
                    f"unknown time unit {self.time_unit!r}; one of {', '.join(TIME_UNITS)}")
            if self.sample_rate_hz is not None:
                raise ValueError("a time-unit declaration carries no rate of its own")
            return
        hz = self.sample_rate_hz
        if (isinstance(hz, bool) or not isinstance(hz, Real) or not math.isfinite(hz)
                or hz <= 0):
            raise ValueError("a declared sample rate must be a finite positive number in Hz")
        object.__setattr__(self, "sample_rate_hz", float(hz))
        if self.time_unit is not None:
            raise ValueError("a sample-rate declaration names no time unit")

    # -- resolution against what was read ------------------------------------------------------
    def resolve(self, read: TimeBaseAsRead) -> tuple[float, float, float]:
        """``(sample_rate_hz, t0_s, time_scale_s)`` this declaration makes of ``read``.

        Raises ``ValueError`` for a time-unit declaration on a document with no measured
        rate: there is no column to re-read in another unit, and a rate would be invented.
        """
        if self.mode == "time_unit":
            if not read.can_declare_unit:
                if read.sample_rate_hz is not None:
                    raise ValueError(
                        "this recording's rate is declared in its header, not measured from a "
                        "time column, so there is no column to read in another unit; declare "
                        "its sample rate instead")
                raise ValueError(
                    "this recording has no usable time column to read in another unit; "
                    "declare its sample rate instead")
            measured = (read.sample_rate_hz if read.sample_rate_hz is not None
                        else read.refused_rate_hz)
            scale = TIME_UNITS[self.time_unit][1]
            factor = float(read.time_scale_s) / scale
            return (float(measured) * factor, float(read.t0_s) / factor, scale)
        # A t0 read off a column the reader refused as a clock is in an unknown unit; a
        # declared rate cannot place it, so the recording starts at 0 rather than at
        # 2,500,000 "seconds" — what the HDF5 reader already does at the same refusal.
        t0 = 0.0 if (read.sample_rate_hz is None and read.refused_rate_hz is not None) \
            else float(read.t0_s)
        return (float(self.sample_rate_hz), t0, float(read.time_scale_s))

    def describe(self) -> str:
        if self.mode == "time_unit":
            return f"time column declared in {TIME_UNITS[self.time_unit][0]}"
        return f"sample rate declared as {self.sample_rate_hz:,.6g} Hz"

    # -- JSON (strict both ways, like AcquisitionBandwidth) ------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "time_unit": self.time_unit,
            "sample_rate_hz": self.sample_rate_hz,
            "note": self.note,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SampleRateDeclaration":
        if not isinstance(payload, Mapping):
            raise ValueError("a sample-rate declaration must be a mapping")
        known = {f.name for f in fields(cls)}
        extras = set(payload) - known
        if extras:
            raise ValueError(f"unrecognized sample-rate declaration fields: {sorted(extras, key=str)}")
        return cls(**{k: payload[k] for k in known if k in payload})
