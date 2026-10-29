"""What the Measure tab is told about the span under the cursors.

A plain value, in its own module because the two panels that exchange it (the Analysis panel
answers, the Measure panel shows) must not import each other.
"""
from __future__ import annotations

from typing import Any, NamedTuple

__all__ = ["HandRowPreview"]


class HandRowPreview(NamedTuple):
    """The table the next hand-made event goes to, the signal that table measures on, and (when
    it can be added) the row it would store.

    ``reason`` is set when there is NO basis to measure under (no rate, a non-current channel, a
    stale or missing table); the readout then falls back to its own local measurement.
    ``refusal`` is set when the basis is fine but THIS span would not be added (it overlaps an
    event, or lies outside what the detector searched); the readout still describes the span."""

    table: str = ""
    module_id: str = ""
    basis_kind: str = "raw"
    basis_text: str = ""
    reason: str = ""
    refusal: str = ""
    signal: Any = None
    fs: float = 0.0
    #: The TABLE's polarity convention (``dev = polarity * (x - baseline)`` is positive in an
    #: event). ``measure_selection`` uses the opposite sign; the Measure panel negates it.
    polarity: int = 0
    noise_rms: float = float("nan")
    #: Mean of the table's baseline track over the span: what the stored row's ``baseline_pA``
    #: would be, so the unstored statistics answer against the same reference.
    baseline: float = float("nan")
    row: Any = None
    hand_counts: str = ""
