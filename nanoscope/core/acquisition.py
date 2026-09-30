"""Acquisition bandwidth is provenance, not another software filter.

This object cannot process samples and deliberately supplies no cutoff-to-rise-time
conversion: a corner frequency alone does not identify the transfer function. A
header-reported value is a declaration, never a measurement or calibration truth.
Session overrides should keep this original object on ChannelInfo, not replace it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Mapping


@dataclass(frozen=True)
class AcquisitionBandwidth:
    """One declared hardware low-pass stage (or explicit lack of knowledge).

    ``disabled`` means the *reported stage* is bypassed; it does not certify that
    every component in the acquisition chain is unfiltered. ``header_fields``
    retains header values verbatim as diagnostic strings, including invalid and
    sentinel values, without putting NaN/Infinity into project JSON.
    """

    cutoff_hz: float | None = None
    status: str = "unknown"
    source: str = "unknown"
    filter_type: str = ""
    filter_order: int | None = None
    note: str = ""
    header_fields: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("unknown", "reported", "disabled"):
            raise ValueError("acquisition bandwidth status must be unknown, reported, or disabled")
        if self.source not in ("unknown", "file_header", "user"):
            raise ValueError("acquisition bandwidth source must be unknown, file_header, or user")
        cutoff = self.cutoff_hz
        if cutoff is not None:
            if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
                raise ValueError("acquisition cutoff must be a finite positive number in Hz")
            cutoff = float(cutoff)
            if not math.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("acquisition cutoff must be a finite positive number in Hz")
            object.__setattr__(self, "cutoff_hz", cutoff)
        if (self.status == "reported") != (cutoff is not None):
            raise ValueError("only reported acquisition bandwidth has a cutoff in Hz")
        if self.status != "unknown" and self.source == "unknown":
            raise ValueError("reported or disabled acquisition bandwidth needs a source")
        for name in ("filter_type", "note"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"acquisition bandwidth {name} must be text")
        order = self.filter_order
        if order is not None:
            if (isinstance(order, bool) or not isinstance(order, Integral)
                    or not 1 <= order <= 2147483647):
                raise ValueError("acquisition filter order must be a positive 32-bit integer or unknown")
            if self.status != "reported" or not self.filter_type.strip():
                raise ValueError("acquisition filter order needs a reported cutoff and filter type")
            object.__setattr__(self, "filter_order", int(order))
        if not isinstance(self.header_fields, (tuple, list)):
            raise ValueError("acquisition header fields must be name/value pairs")
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for pair in self.header_fields:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(item, str) for item in pair)):
                raise ValueError("acquisition header fields must be text name/value pairs")
            name, value = pair
            if not name or name in seen:
                raise ValueError("acquisition header field names must be nonempty and unique")
            seen.add(name)
            normalized.append((name, value))
        object.__setattr__(self, "header_fields", tuple(normalized))

    def to_dict(self) -> dict:
        """Strict-JSON-compatible project/export payload; preserve all provenance."""
        return {"cutoff_hz": self.cutoff_hz, "status": self.status, "source": self.source,
                "filter_type": self.filter_type, "filter_order": self.filter_order,
                "note": self.note, "header_fields": [list(pair) for pair in self.header_fields]}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AcquisitionBandwidth":
        """Refuse malformed overrides instead of silently substituting unknown."""
        if not isinstance(payload, Mapping):
            raise ValueError("acquisition bandwidth must be an object")
        extras = set(payload) - {field.name for field in fields(cls)}
        if extras:
            raise ValueError(f"unrecognized acquisition bandwidth fields: {sorted(extras, key=str)}")
        return cls(**dict(payload))

    def fingerprint(self) -> tuple:
        """Provenance identity only; it must never enter signal-filter cache keys."""
        return (self.cutoff_hz, self.status, self.source, self.filter_type,
                self.filter_order, self.note, self.header_fields)


def describe_bandwidth(value: AcquisitionBandwidth) -> str:
    if value.status == "reported":
        response = value.filter_type or "type unknown"
        order = f", order {value.filter_order}" if value.filter_order is not None else ", order unknown"
        return f"{value.cutoff_hz:g} Hz ({response}{order}; {value.source})"
    if value.status == "disabled":
        return f"Reported stage bypassed ({value.source}); other acquisition stages may still filter"
    return f"Unknown ({value.source})"


def stamp_acquisition_bandwidth(provenance: dict, ctx) -> None:
    """Freeze both the original header and effective user declaration at run time."""
    effective = getattr(ctx.channel.info, "acquisition_bandwidth", AcquisitionBandwidth())
    header = next((getattr(c, "acquisition_bandwidth", AcquisitionBandwidth())
                   for c in ctx.meta.channels if c.index == ctx.channel.info.index),
                  AcquisitionBandwidth())
    provenance["acquisition_bandwidth"] = {
        "version": 1, "effective": effective.to_dict(), "file_header": header.to_dict(),
        "resolution_assessment": "not_assessed",
        "filter_limited_scope": "existing software-filter assessment only",
    }


def stamp_reference_availability(provenance: dict, ctx) -> None:
    """Propagate 'this document has no physical open-pore reference' onto every
    detector's own provenance, not just the bridge's synthetic result -- see
    RecordingMeta.physical_reference_unavailable."""
    if getattr(ctx.meta, "physical_reference_unavailable", False):
        provenance["fractional_blockage_reference_unavailable"] = True


def acquisition_snapshot_messages(provenance: Mapping) -> tuple[str, ...]:
    """Stored acquisition facts for Details, never rendered into plot artwork."""
    raw = provenance.get("acquisition_bandwidth")
    if raw is None:
        return ("Acquisition bandwidth was not recorded for this result. Hardware-limited "
                "temporal resolution has not been assessed.",)
    try:
        if not isinstance(raw, Mapping) or raw.get("version") != 1:
            raise ValueError("unsupported snapshot")
        value = AcquisitionBandwidth.from_dict(raw["effective"])
    except (KeyError, TypeError, ValueError):
        return ("Stored acquisition bandwidth is unreadable; hardware-limited temporal "
                "resolution has not been assessed.",)
    text = "Acquisition low-pass at detection: " + describe_bandwidth(value) + "."
    if value.note:
        text += " " + value.note
    return (text + " Metadata only: combined acquisition response and hardware-limited "
            "event resolution are not assessed. The existing filter_limited flag covers "
            "software conditioning, not the full acquisition chain.",)
