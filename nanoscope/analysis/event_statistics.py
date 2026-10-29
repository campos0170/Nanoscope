"""Statistics of event rows, in the measurement table's own columns.

The manual measurement list used to be the only place a hand-selected span got its full
statistics (mean, median, std, variance, min, max, ptp, rms, skewness, kurtosis and the area).
Hand-made events are event rows now, in event tables, so those statistics are computed FROM the
rows: for any table, automatic or manual, on the basis the table was measured on.

The frame returned has the 32 `measure.MEASUREMENT_SCHEMA` columns in their frozen order, then
`signal_basis` (the 33rd, see `conditioning.SIGNAL_BASIS_COLUMN`), then `table` and `event_id` so
rows pooled from several tables can be told apart. Everything about HOW each row was measured
(its baseline, its noise, its polarity) is read from the table, never re-estimated from a local
flank, so the statistics are the same numbers the event table shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .conditioning import (
    BASIS_RAW,
    SIGNAL_BASIS_COLUMN,
    chain_description,
    conditioning_enabled,
)
from .measure import (
    MEASUREMENT_SCHEMA,
    UNKNOWN_CHANNEL_INDEX,
    measure_selection,
)

__all__ = ["StatisticsSkipped", "event_statistics_frame", "signal_basis_text"]

#: The two columns after `signal_basis`: which table the row came from and its id there.
EXTRA_COLUMNS = ("table", "event_id")


@dataclass(frozen=True)
class StatisticsSkipped:
    """Rows that could not be measured, with the reason, so an export never drops one silently."""

    event_id: int
    reason: str


def signal_basis_text(provenance: Mapping[str, Any] | None) -> str:
    """What the rows of a table were measured on, as one phrase for the ``signal_basis`` column.

    ``"raw"`` for a table measured on the calibrated channel as loaded, else ``"processed: "``
    and the chain (``chain_description``: ``"notch 60 Hz → low-pass bessel 5000 Hz (order 8)"``).
    """
    prov = dict(provenance or {})
    if prov.get("signal_basis") == BASIS_RAW:
        return BASIS_RAW
    chain = chain_description(prov) if (conditioning_enabled(prov)
                                        or prov.get("filter_cutoff_hz") is not None) else ""
    return f"processed: {chain}" if chain else BASIS_RAW


def event_statistics_frame(
    events: pd.DataFrame,
    signal: np.ndarray,
    fs: float,
    *,
    provenance: Mapping[str, Any],
    table_name: str,
    units: str,
    recording: str,
    doc_key: str,
    channel_name: str,
    channel_index: int,
    cal_gain: float = float("nan"),
    cal_offset: float = float("nan"),
    t0_s: float = float("nan"),
    amplitude_method: str = "median",
    only_event_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, list[StatisticsSkipped]]:
    """The full statistics of every row of ``events``, measured on ``signal``.

    ``signal`` is the array the table's rows were measured on (``detection_signal`` for a
    detector table, the raw channel for a Manual one). Each row keeps the baseline the TABLE
    gave it (``baseline_pA``), the table's noise (``noise_rms_pA``) and polarity, so the
    amplitude, fractional amplitude and SNR here are the event table's numbers, not a second
    set read off a local flank.
    """
    basis = signal_basis_text(provenance)
    # Two conventions meet here. An event table's polarity makes dev = polarity * (x - baseline)
    # positive INSIDE an event (a blockade on a positive pore is -1); the measurement table's
    # makes amplitude = polarity * (baseline - level) positive (the same blockade is +1).
    # They are opposite signs for the same physical event.
    polarity = provenance.get("polarity")
    polarity = -int(polarity) if polarity in (-1, 1) else None
    noise = provenance.get("noise_rms_pA")
    noise = float(noise) if noise is not None and np.isfinite(float(noise)) and float(noise) > 0 \
        else None
    rows: list[dict[str, Any]] = []
    skipped: list[StatisticsSkipped] = []
    note_col = events["note"] if "note" in events.columns else None
    for i in range(len(events)):
        event_id = int(events["event_id"].iloc[i])
        if only_event_ids is not None and event_id not in only_event_ids:
            continue
        start, end = int(events["start_idx"].iloc[i]), int(events["end_idx"].iloc[i])
        baseline = float(events["baseline_pA"].iloc[i])
        try:
            stats = measure_selection(
                signal, fs, start, end, amplitude_method=amplitude_method, polarity=polarity,
                baseline=baseline if np.isfinite(baseline) else None, noise_rms=noise,
                units=units)
        except ValueError as exc:
            skipped.append(StatisticsSkipped(event_id, str(exc)))
            continue
        row = {c: stats[c] for c in MEASUREMENT_SCHEMA if c in stats}
        note = None if note_col is None else note_col.iloc[i]
        row.update({
            "measurement_id": event_id,
            "recording": recording,
            "doc_key": doc_key,
            "channel": channel_name,
            "channel_index": int(channel_index) if channel_index is not None
            else UNKNOWN_CHANNEL_INDEX,
            "label": "" if note is None or pd.isna(note) else str(note),
            "units": units,
            "cal_gain": float(cal_gain),
            "cal_offset": float(cal_offset),
            "t0_s": float(t0_s),
            SIGNAL_BASIS_COLUMN: basis,
            "table": table_name,
            "event_id": event_id,
        })
        rows.append(row)
    columns = [*MEASUREMENT_SCHEMA, SIGNAL_BASIS_COLUMN, *EXTRA_COLUMNS]
    frame = pd.DataFrame(rows, columns=columns)
    if len(frame):
        frame = frame.astype({c: d for c, (d, _u, _desc) in MEASUREMENT_SCHEMA.items()
                              if c in frame.columns})
    return frame, skipped
