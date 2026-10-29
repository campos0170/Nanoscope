"""Turn the old manual-measurement list into Manual events (docs/MANUAL_EVENT_TABLE_DESIGN.md, 10).

The measurement list was a session-wide table of hand-selected spans that was not an event table:
its rows never reached the gallery, the inspector, Reconcile or an event export. A hand-made event
is an event row now, so an old row can become one. This module is the PLAN: for each row of the
list, whether it can be placed on the recording and channel on screen, and if not, why. The person
ticks the placeable rows they want (a baseline or noise check is not an event, and the event model
has no neutral state, so nothing is converted unasked); the panel then measures each as an event
under the Manual table's contract.

Placement is by exact identity, never by guess: the row's ``doc_key`` must be the open recording's
persistent key and its ``channel_index`` and channel NAME must be the selected channel's. The
navigation fallbacks the measurement panel uses to find a row's recording are not used here: a row
placed on the wrong trace would measure whatever happens to live at those sample numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
import pandas as pd

from ..core.identity import same_source

__all__ = ["ConversionRow", "TargetChannel", "plan_conversion"]


@dataclass(frozen=True)
class TargetChannel:
    """The channel a conversion would place rows on: what the Analysis panel has selected."""

    doc_key: str            # the recording's persistent key (source path, or segment uid)
    channel_index: int
    channel_name: str
    n_samples: int
    fs: float | None
    units: str
    cal_gain: float
    cal_offset: float


@dataclass(frozen=True)
class ConversionRow:
    """One measurement row, and whether and how it can become an event."""

    measurement_id: int
    recording: str
    channel: str
    start: int
    end: int
    label: str
    #: Can this row be placed on the target channel and converted now?
    placeable: bool
    #: Why not, as a sentence naming the way forward; ``""`` when placeable.
    reason: str
    #: The row's event numbers will differ from its stored measurement numbers: it was measured
    #: under another calibration or units, or on the processed trace.
    numbers_change: bool = False
    #: The row is already one of the events some Manual table holds (``legacy_measurement_ids``).
    already_converted: bool = False


def _text(value: Any) -> str:
    return "" if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)) \
        else str(value)


def _differs(row_value: Any, live: float) -> bool:
    try:
        v = float(row_value)
    except (TypeError, ValueError):
        return False                     # unknown: not a claim that it differs
    return bool(np.isfinite(v) and not np.isclose(v, float(live), rtol=1e-9, atol=0.0))


def plan_conversion(
    measurements: pd.DataFrame | None,
    target: TargetChannel | None,
    *,
    converted_ids: Collection[int] = (),
    open_doc_keys: Collection[str] = (),
) -> list[ConversionRow]:
    """One :class:`ConversionRow` per measurement row, in the list's own order.

    ``converted_ids`` are the measurement ids some live Manual table already holds as events
    (``legacy_measurement_ids``): such a row is listed as converted and not offered again.
    ``open_doc_keys`` are the persistent keys of every open recording, so a row on another open
    recording can be told to "select that recording" while one on a recording that is not open
    says so. ``target`` is None when no channel is selected, and then nothing is placeable.
    """
    if measurements is None or not len(measurements):
        return []
    done = {int(i) for i in converted_ids}
    opened = [str(k) for k in open_doc_keys]
    rows: list[ConversionRow] = []
    for rec in measurements.to_dict("records"):
        mid = int(rec.get("measurement_id", -1))
        start, end = int(rec.get("start_idx", -1)), int(rec.get("end_idx", -1))
        doc_key = _text(rec.get("doc_key")).strip()
        ch_index = rec.get("channel_index", -1)
        try:
            ch_index = int(ch_index)
        except (TypeError, ValueError):
            ch_index = -1
        reason = ""
        converted = mid in done
        if converted:
            reason = "already converted: a Manual table holds it as an event"
        elif target is None:
            reason = "select a recording and channel first"
        elif not doc_key:
            reason = "this row did not record which recording it was taken on"
        elif ch_index < 0:
            reason = "this row did not record which channel it was taken on"
        elif not same_source(doc_key, target.doc_key):
            reason = ("select that recording to convert it" if any(same_source(doc_key, k)
                                                                     for k in opened)
                      else "that recording is not open")
        elif ch_index != target.channel_index or (
                _text(rec.get("channel")) and _text(rec.get("channel")) != target.channel_name):
            reason = "it was taken on another channel: select that channel to convert it"
        elif target.fs is None:
            reason = "the recording has no sample rate; declare one in File Info"
        elif end <= start or start < 0 or end > target.n_samples:
            reason = "its span lies outside the loaded data"
        changes = False
        if not reason:
            basis = _text(rec.get("signal_basis")).strip().lower()
            changes = (_differs(rec.get("cal_gain"), target.cal_gain)
                       or _differs(rec.get("cal_offset"), target.cal_offset)
                       or (bool(_text(rec.get("units"))) and bool(target.units)
                           and _text(rec.get("units")) != target.units)
                       or basis == "processed")
        rows.append(ConversionRow(
            measurement_id=mid, recording=_text(rec.get("recording")),
            channel=_text(rec.get("channel")), start=start, end=end,
            label=_text(rec.get("label")), placeable=not reason, reason=reason,
            numbers_change=changes, already_converted=converted))
    return rows
