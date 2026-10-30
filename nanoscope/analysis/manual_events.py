"""Hand-made events and the Manual table: the pure half (no Qt, no session).

A manual event is an ordinary event row that a person drew. It carries the frozen 20
`events.SCHEMA` columns plus the pair `scoring.ADDED_COLUMN` (``True``) and
`events.SOURCE_COL` (``"manual"``), and it lives in an event table: either the table the
user is curating (Schmitt, CUSUM, Keras, a study table) or the channel's own **Manual
table**, module id :data:`~nanoscope.analysis.events.MANUAL_MODULE_ID`, which exists for
manual work with no detector behind it.

This module owns what is independent of the GUI:

* the additive extra columns a hand-made row may carry (:data:`MANUAL_EXTRAS`);
* the **measurement contract** of a Manual table (:class:`ManualContract`), the provenance
  that every row of the table is measured under, built once per recording and channel;
* measuring one hand-made row (:func:`measure_hand_row`) through the ONE row-rebuild path,
  :func:`~nanoscope.analysis.characterize.recharacterize_span`, and inserting it in time
  order without disturbing the frame's dtypes (:func:`insert_hand_row`);
* the whole-table repair (:func:`rebuild_manual_events`) and the small readers
  (:func:`hand_made_mask`).

Design: ``docs/MANUAL_EVENT_TABLE_DESIGN.md``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .baseline import (
    BaselineConfig,
    characterisation_baseline,
    measured_polarity,
    repair_nonfinite,
)
from .characterize import recharacterize_span
from .edge_fit import REASON_NOT_REQUESTED
from .events import (
    MANUAL_MODULE_ID,
    ORIGINAL_BOUNDS_SCHEMA,
    REVIEW_COL,
    REVIEW_DONE,
    SCHEMA,
    SOURCE_COL,
    SOURCE_MANUAL,
    SOURCE_STATUS_SCHEMA,
    EventSet,
    empty_df,
)
from .filters import noise_rms
from .scoring import ADDED_COLUMN, ensure_added_column, hand_made_mask

__all__ = [
    "BOUNDS_SOURCE_COL",
    "DRAWN_END_COL",
    "DRAWN_START_COL",
    "EdgeRecord",
    "FIT_JSON_COL",
    "ID_JOINER",
    "LEGACY_IDS_COL",
    "MANUAL_EXTRAS",
    "MANUAL_MODULE_VERSION",
    "ManualContract",
    "NOTE_COL",
    "NOTE_JOINER",
    "ORIGIN_COL",
    "CARRY_MIN_OVERLAP",
    "CarryReport",
    "CopyReport",
    "copy_hand_rows",
    "build_contract",
    "carry_hand_rows",
    "contract_key",
    "delete_hand_rows",
    "empty_manual_events",
    "ensure_manual_extras",
    "hand_made_mask",
    "insert_hand_row",
    "join_edge_fits",
    "join_ids",
    "join_notes",
    "measure_hand_row",
    "merged_edge_cells",
    "measurement_noise_rms",
    "nullable_extras",
    "rebuild_manual_events",
]

#: Bump when the contract's keys or the row recipe change in a way a stored table must notice.
MANUAL_MODULE_VERSION = "1"

# ---------------------------------------------------------------------------
# The additive extras. None of them is in `events.SCHEMA` (frozen at 20) and none of them may
# be: `characterize` ends with ``df[list(SCHEMA)]`` and `EventSet` refuses a missing SCHEMA
# column, so an extra is what lets an old project load and a rebuild drop nothing the caller
# re-applies. Every one is NULLABLE, because a table mixes rows that have them and rows that
# do not (a detected row has no ``note``).
# ---------------------------------------------------------------------------
#: Free text the user can edit. A converted measurement's ``label`` lands here.
NOTE_COL = "note"
#: What joins the notes of several rows that become ONE event (:func:`join_notes`), in time
#: order: a re-run's carry-over and a merge.
NOTE_JOINER = "; "
#: Where a copied or carried row came from, for example ``"detect.manual:41"``; a merged row lists
#: every fragment's, ``"detect.manual:41,detect.manual:42"``.
ORIGIN_COL = "origin"
#: The ids of the legacy measurement rows this event was converted from, ``"12"`` or ``"12,15"``.
LEGACY_IDS_COL = "legacy_measurement_ids"
#: What separates the ids of an ``origin`` or ``legacy_measurement_ids`` cell that lists several
#: (:func:`join_ids`), the form `AnalysisPanel.converted_measurement_ids` reads back.
ID_JOINER = ","
#: How each edge was made: ``"start=fit:step;end=placed"``. Values: drawn, placed, fit:step,
#: fit:schmitt, fit:cusum, edited, stale.
BOUNDS_SOURCE_COL = "bounds_source"
#: The bounds before any fit, so a fit can be redone under another rule.
DRAWN_START_COL = "drawn_start_idx"
DRAWN_END_COL = "drawn_end_idx"
#: Method, version and every gate the edge fit used, as JSON.
FIT_JSON_COL = "fit_json"

#: Column -> pandas dtype. The same shape as `events.ORIGINAL_BOUNDS_SCHEMA`'s first field.
MANUAL_EXTRAS: dict[str, str] = {
    NOTE_COL: "string",
    ORIGIN_COL: "string",
    LEGACY_IDS_COL: "string",
    BOUNDS_SOURCE_COL: "string",
    DRAWN_START_COL: "Int64",
    DRAWN_END_COL: "Int64",
    FIT_JSON_COL: "string",
}

#: Columns whose dtype is declared somewhere in the app, so a grown frame casts them to THAT
#: dtype and never to a guess. Anything else outside `SCHEMA` is an unknown extra (a native
#: study table's ``study_*`` columns) and gets a nullable dtype.
_DECLARED_EXTRAS: dict[str, str] = {
    **{c: d for c, (d, _u, _desc) in ORIGINAL_BOUNDS_SCHEMA.items()},
    **{c: d for c, (d, _u, _desc) in SOURCE_STATUS_SCHEMA.items()},
    **MANUAL_EXTRAS,
    ADDED_COLUMN: "bool",
}


@dataclass(frozen=True)
class EdgeRecord:
    """What the add door records about how a row's bounds were made (cursor design 5.5).

    ``bounds_source`` is the per-edge string; ``drawn_start`` / ``drawn_end`` the bounds
    before any fit; ``fit_json`` the fit's own record. All optional: Ctrl+M on a plain
    bracket records nothing but ``drawn``.
    """

    bounds_source: str | None = None
    drawn_start: int | None = None
    drawn_end: int | None = None
    fit_json: str | None = None


@dataclass(frozen=True)
class ManualContract:
    """The measurement basis of one channel's Manual table, fixed at creation.

    ``provenance`` is the dict the Manual `EventSet` carries, and therefore the dict every
    `recharacterize_span` call reads: the signal basis is the raw calibrated channel (no
    conditioning keys, so `modules.detection_signal` returns it untouched), the polarity is
    measured over the whole channel, the baseline block is the confirmed Detection-form
    `BaselineConfig`, and the noise is ``noise_rms(trace - characterisation_baseline)`` over
    the whole recording, the number every detector's own rows divide by.

    Deliberately absent: the analysed span, ``analyzed_duration_s``, ``min_dwell_s``,
    ``max_dwell_s`` and every threshold. Nobody knows how much was searched by hand, so the
    table has no event rate.
    """

    key: tuple[Any, ...]
    provenance: Mapping[str, Any]
    calibration_fingerprint: tuple[float, float]
    baseline_fingerprint: str | None
    polarity: int
    noise_rms_pA: float
    open_pore_tau: float | None

    def events_provenance(self) -> dict[str, Any]:
        """A fresh, mutable copy of the provenance for a new `EventSet`."""
        return dict(self.provenance)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Extras and dtypes
# ---------------------------------------------------------------------------
def ensure_manual_extras(df: pd.DataFrame, *, only: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Add any missing manual extra to ``df`` IN PLACE with its nullable dtype and all-NA
    cells. The sibling of `scoring.ensure_added_column`; idempotent."""
    for col, dtype in MANUAL_EXTRAS.items():
        if only is not None and col not in only:
            continue
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), index=df.index, dtype=dtype)
    return df


def _nullable(series: pd.Series) -> pd.Series:
    """``series`` in a dtype that can hold "not recorded" per row, or unchanged when it holds
    something that is not a plain scalar column."""
    dt = series.dtype
    if pd.api.types.is_bool_dtype(dt):
        return series.astype("boolean")
    if pd.api.types.is_integer_dtype(dt):
        return series.astype("Int64")
    if pd.api.types.is_float_dtype(dt) or isinstance(dt, pd.api.extensions.ExtensionDtype):
        return series
    if pd.api.types.is_object_dtype(dt):
        values = series.dropna()
        if all(isinstance(v, str) for v in values):
            return series.astype("string")
    return series


def nullable_extras(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``df`` whose UNKNOWN extras (not `SCHEMA`, not a declared extra) have nullable
    dtypes.

    A native study table carries int64 and bool ``study_*`` columns. A hand-made row has no
    value for them, and concatenating one with a cast back to int64 raises, while a cast back
    to bool turns NaN into True. Casting them to ``Int64`` / ``boolean`` first makes the
    missing cell representable. Declared extras keep their declared dtypes
    (`_DECLARED_EXTRAS`), and the 20 `SCHEMA` columns are never touched here: the caller
    restores them to their own dtypes.
    """
    out = df.copy()
    for col in out.columns:
        if col in SCHEMA or col in _DECLARED_EXTRAS:
            continue
        out[col] = _nullable(out[col])
    return out


def _restore_dtypes(grown: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """``grown`` with `SCHEMA` columns back at the template's dtypes, declared extras at their
    declared dtypes, and unknown extras nullable."""
    out = grown
    for col in out.columns:
        try:
            if col in SCHEMA and col in template.columns:
                out[col] = out[col].astype(template[col].dtype)
            elif col in _DECLARED_EXTRAS:
                out[col] = out[col].astype(_DECLARED_EXTRAS[col])
            elif col in SCHEMA:
                out[col] = out[col].astype(SCHEMA[col][0])
        except (TypeError, ValueError):
            # A column that will not take its declared dtype keeps what concat gave it: a
            # visible odd cell is better than refusing the whole add.
            continue
    return out


def empty_manual_events(contract: ManualContract) -> EventSet:
    """An empty Manual table over ``contract``: the 20 `SCHEMA` columns, ``added`` and the
    source/review pair, the original-bounds pair and every manual extra, all at their dtypes."""
    df = empty_df()
    ensure_added_column(df)
    df[SOURCE_COL] = pd.array([], dtype="string")
    df[REVIEW_COL] = pd.array([], dtype="string")
    for col, (dtype, _unit, _desc) in ORIGINAL_BOUNDS_SCHEMA.items():
        df[col] = pd.array([], dtype=dtype)
    ensure_manual_extras(df)
    return EventSet(df, contract.events_provenance())


# ---------------------------------------------------------------------------
# The measurement contract
# ---------------------------------------------------------------------------
def measurement_noise_rms(signal: np.ndarray, baseline: np.ndarray) -> float:
    """``noise_rms(signal - baseline)`` over the finite samples: THE SNR denominator a
    detector stamps as ``provenance['noise_rms_pA']`` and every re-measured row divides by.

    A non-finite sample is left out rather than refused, as the baseline door does with
    ``on_nonfinite="interpolate"`` for a whole-recording estimate: one bad sample must not
    stop a table existing.

    Raises:
        ValueError: when no noise is measurable (a flat or empty trace), with a sentence a
            person can act on.
    """
    resid = np.asarray(signal, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    resid = resid[np.isfinite(resid)]
    if resid.size < 16:
        raise ValueError(
            "The recording is too short or too empty to measure its noise, so a Manual table "
            "cannot be measured on it.")
    noise = float(noise_rms(resid))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError(
            "The recording shows no measurable noise (the trace is flat), so a Manual table "
            "cannot be measured on it.")
    return noise


def contract_key(doc_id: str, channel: int, n: int, fs: float,
                 calibration_fingerprint: tuple[float, float],
                 baseline_fingerprint: str | None) -> tuple[Any, ...]:
    """The cache key a pending contract is stored under (design 5.4): recording, channel,
    length, rate, calibration and baseline estimator."""
    return (str(doc_id), int(channel), int(n), float(fs),
            tuple(float(v) for v in calibration_fingerprint), baseline_fingerprint)


def build_contract(
    signal: np.ndarray,
    fs: float,
    units: str | None,
    *,
    config: BaselineConfig | None,
    param_source: str = "default",
    recording: str = "",
    calibration_fingerprint: tuple[float, float] = (1.0, 0.0),
    doc_id: str = "",
    channel: int = 0,
    tau: float | None = None,
    detected_at: str | None = None,
) -> ManualContract:
    """Measure a recording once and return the basis every row of its Manual table uses.

    ``signal`` is the array every later `recharacterize_span` call will be handed, and it must
    be that SAME OBJECT: the whole-recording baseline is memoised on the identity of the array
    (`baseline._BaselineCache`), so a contract built on one copy and rows measured on another
    would each pay the full estimate. The panel keeps one conditioned array per key.

    This runs the whole-recording baseline (about 0.3 s per 3 M samples) and the noise over it,
    so it belongs on a worker thread, never on a key press.
    """
    sig = np.asarray(signal)
    fs = float(fs)
    cfg = config
    # The polarity vote refuses a non-finite trace; one bad sample must not stop a table from
    # existing, so the VOTE (not the rows) reads a repaired copy. Rows are measured on `sig`
    # itself, where `recharacterize_span` poisons the few rows that actually read the sample.
    pol = int(measured_polarity(repair_nonfinite(sig)[0], fs, config=cfg))
    result = characterisation_baseline(
        sig, fs, config=cfg, polarity=pol, param_source=param_source, health=False,
        on_nonfinite="interpolate")
    noise = measurement_noise_rms(sig, result.baseline)
    if tau is None:
        # The open-pore autocorrelation time the cursor design's edge fit needs, measured in
        # the same pass over the same residual (pseudo-events stand in for a table with no
        # rows). A degenerate trace costs the fit its tau, never the table its existence.
        try:
            from .edge_fit import estimate_open_pore_tau

            tau = estimate_open_pore_tau(
                np.asarray(sig, dtype=np.float64) - np.asarray(result.baseline, dtype=np.float64),
                None, noise_rms=noise, fs=fs)
        except Exception:  # noqa: BLE001
            tau = None
    prov: dict[str, Any] = {
        "detector": "manual",
        "module": MANUAL_MODULE_ID,
        "recording": recording or None,
        "fs": fs,
        "units": units or "pA",
        "polarity": pol,
        "polarity_source": "measured_polarity",
        "noise_rms_pA": noise,
        "signal_basis": "raw",
        "detected_at": detected_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_events": 0,
        "n_filter_limited": 0,
        "n_clipped_boundary": 0,
    }
    prov.update(result.provenance(param_source=param_source))
    if tau is not None and np.isfinite(float(tau)) and float(tau) > 0:
        prov["open_pore_tau"] = float(tau)
    base_fp = prov.get("baseline_fingerprint")
    return ManualContract(
        key=contract_key(doc_id, channel, sig.size, fs, calibration_fingerprint, base_fp),
        provenance=prov,
        calibration_fingerprint=tuple(float(v) for v in calibration_fingerprint),
        baseline_fingerprint=base_fp,
        polarity=pol,
        noise_rms_pA=noise,
        open_pore_tau=prov.get("open_pore_tau"),
    )


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------
def _scalar(value: Any) -> Any:
    return pd.NA if value is None else value


def join_notes(*notes: Any) -> Any:
    """The notes given, in that order, joined by :data:`NOTE_JOINER`; a blank one or a repeat
    adds nothing, and ``pd.NA`` when none is left. A note is a person's own words, so two rows
    that become one event keep both.

    Hand it EVERY note at once: a note already joined is one piece of text here, so joining
    onto it again would not see that a later note repeats one of its parts."""
    out: list[str] = []
    seen: set[str] = set()
    for note in notes:
        if note is None or pd.isna(note):
            continue
        text = str(note)
        if not text.strip() or text.strip() in seen:
            continue
        seen.add(text.strip())
        out.append(text)
    return NOTE_JOINER.join(out) if out else pd.NA


def join_ids(*cells: Any) -> Any:
    """Every id in the ``origin`` or ``legacy_measurement_ids`` cells given, in that order, joined
    by :data:`ID_JOINER`; a blank or a repeat adds nothing, and ``pd.NA`` when none is left.

    Unlike a note, each cell is split first: an id is a token, not a person's words, and a row
    merged before already lists several, so ``"12,15"`` and ``"15"`` make ``"12,15"``."""
    out: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        for part in _id_parts(cell):
            if part not in seen:
                seen.add(part)
                out.append(part)
    return ID_JOINER.join(out) if out else pd.NA


def _id_parts(cell: Any) -> list[str]:
    """The ids one ``origin`` or ``legacy_measurement_ids`` cell lists."""
    if cell is None or pd.isna(cell):
        return []
    return [part.strip() for part in str(cell).split(ID_JOINER) if part.strip()]


def measure_hand_row(
    signal: np.ndarray,
    fs: float,
    start: int,
    end: int,
    *,
    provenance: Mapping[str, Any],
    units: str | None,
    recording: str,
    pore_id: str,
    event_id: int,
    note: str | None = None,
    origin: str | None = None,
    legacy_ids: str | None = None,
    prior: Any = None,
    edge: EdgeRecord | None = None,
) -> pd.DataFrame:
    """One hand-made row, measured under ``provenance``.

    The row carries every key a hand-made row can have (the 20 `SCHEMA` columns, ``added``,
    ``source``, ``review``, the original-bounds pair and the manual extras); `insert_hand_row`
    aligns it with the table it joins.

    Delegates the measurement to `recharacterize_span`, the ONE row-rebuild path (the gap
    merge, the inspector's mask drag and every rebuild use it), so a hand-made row is
    measured on the same basis as its neighbours. ``prior`` is ``None`` for a new row: it
    replaces nothing, so nothing is inherited and the automatic rules judge the new span.

    The row arrives as ``accepted=True`` unless an automatic rule rejects it (a span clipped
    at the recording's edge, say), is stamped ``added=True``, ``source="manual"`` and
    ``review="reviewed"`` (drawing it is a ruling), and has no detector bounds, so Revert
    stays off.

    Raises:
        ValueError: when the span cannot be measured (empty after clipping).
    """
    out = recharacterize_span(
        signal, float(fs), int(start), int(end),
        provenance=provenance, prior=prior,
        recording=str(recording), pore_id=str(pore_id), units=units or "pA")
    if len(out) == 0:
        raise ValueError("The span could not be measured.")
    row = out.iloc[0].to_dict()
    row["event_id"] = int(event_id)
    row[ADDED_COLUMN] = True
    row[SOURCE_COL] = SOURCE_MANUAL
    row[REVIEW_COL] = REVIEW_DONE
    for col in ORIGINAL_BOUNDS_SCHEMA:
        row[col] = pd.NA
    row[NOTE_COL] = _scalar(note)
    row[ORIGIN_COL] = _scalar(origin)
    row[LEGACY_IDS_COL] = _scalar(legacy_ids)
    rec = edge or EdgeRecord()
    row[BOUNDS_SOURCE_COL] = _scalar(rec.bounds_source)
    row[DRAWN_START_COL] = _scalar(None if rec.drawn_start is None else int(rec.drawn_start))
    row[DRAWN_END_COL] = _scalar(None if rec.drawn_end is None else int(rec.drawn_end))
    row[FIT_JSON_COL] = _scalar(rec.fit_json)
    return pd.DataFrame([row])


def insert_hand_row(df: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """``df`` with ``row`` inserted in time order and every dtype kept.

    Returns a NEW frame (the caller installs it under the same key). `SCHEMA` columns go back
    to the table's own dtypes, declared extras to their declared dtypes, and unknown extras
    become nullable first, which is what lets a native study table (int64 ``study_*``
    columns) take a row with no value for them.
    """
    base = nullable_extras(df)
    # A table from before these columns existed (or a detector table that never took a hand
    # row) gets only the columns this row needs: the pair and the original bounds always, an
    # extra only when the row says something in it. A Schmitt export therefore grows
    # `bounds_source` when a hand row lands in it, and `note` only when someone wrote one.
    for col in row.columns:
        if col in base.columns:
            continue
        cell = row[col].iloc[0]
        required = col in (ADDED_COLUMN, SOURCE_COL, REVIEW_COL, *ORIGINAL_BOUNDS_SCHEMA)
        if not (required or not pd.isna(cell)):
            continue
        if col == ADDED_COLUMN:
            base[col] = np.zeros(len(base), dtype=bool)
        else:
            base[col] = pd.Series([pd.NA] * len(base), index=base.index,
                                  dtype=_DECLARED_EXTRAS.get(col, "object"))
    new = row.reindex(columns=base.columns)
    for col in base.columns:
        if col in SCHEMA or col in _DECLARED_EXTRAS:
            continue
        try:
            new[col] = new[col].astype(base[col].dtype)
        except (TypeError, ValueError):
            pass
    with warnings.catch_warnings():
        # Concatenating an all-NA extra with a populated one is exactly the case: the declared
        # dtypes are restored below, so the future dtype rule does not matter here.
        warnings.simplefilter("ignore", FutureWarning)
        grown = pd.concat([base, new], ignore_index=True)
    grown = grown.sort_values("start_idx", kind="stable", ignore_index=True)
    return _restore_dtypes(grown, df)


def delete_hand_rows(df: pd.DataFrame, event_ids: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """``(df without the hand-made rows named in event_ids, the ids actually removed)``.

    Only hand-made rows can be deleted: a detected row is the detector's record, and rejecting
    it is the way to take it out of the count.
    """
    if not len(df):
        return df, []
    wanted = set(int(i) for i in event_ids)
    ids = df["event_id"].to_numpy(dtype=np.int64)
    drop = hand_made_mask(df) & np.fromiter((int(i) in wanted for i in ids), dtype=bool, count=len(ids))
    removed = [int(i) for i in ids[drop]]
    if not drop.any():
        return df, []
    return df.loc[~drop].reset_index(drop=True), removed


# ---------------------------------------------------------------------------
# Whole-table repair
# ---------------------------------------------------------------------------
_MEASURED = (
    "t_start_s", "dwell_s", "baseline_pA", "depth_pA", "frac_depth", "ecd_pA_s", "max_dev_pA",
    "mean_dev_pA", "rise_s", "fall_s", "snr", "n_levels", "filter_limited", "accepted", "flag",
)


def rebuild_manual_events(
    events: EventSet,
    signal: np.ndarray,
    contract: ManualContract,
    *,
    units: str | None,
) -> EventSet:
    """Every row re-measured from its stored indices under ``contract``.

    The repair for a Manual table whose rate, calibration or baseline moved: indices and
    ``event_id`` are held, times and amplitudes move to the live basis. Each row is rebuilt
    with ``prior=row``, so a hand rejection and the curator's own flag tokens survive while
    automatic tokens are re-derived. The extras are untouched.
    """
    df = events.df.copy()
    fs = float(contract.provenance["fs"])
    prov = contract.events_provenance()
    for i in range(len(df)):
        row = df.iloc[i]
        out = recharacterize_span(
            signal, fs, int(row["start_idx"]), int(row["end_idx"]),
            provenance=prov, prior=row,
            recording=str(row["recording"]), pore_id=str(row["pore_id"]),
            units=units or "pA")
        if len(out) == 0:
            continue
        new = out.iloc[0]
        for col in _MEASURED:
            if col in df.columns:
                df.iat[i, df.columns.get_loc(col)] = new[col]
    # Keep what the table already said about itself (when it was created), and replace the
    # measurement basis wholesale.
    prov["detected_at"] = events.provenance.get("detected_at", prov.get("detected_at"))
    prov["rebuilt_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prov["n_events"] = int(len(df))
    return EventSet(_restore_dtypes(df, events.df), prov)


def fit_record_json(payload: Mapping[str, Any]) -> str:
    """Compact, key-sorted JSON for the ``fit_json`` column."""
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=float)


# ---------------------------------------------------------------------------
# The edge record of a row made of several
# ---------------------------------------------------------------------------
_SIDES = ("start", "end")


def _edge_sources(cell: Any) -> dict[str, str]:
    """``{"start": ..., "end": ...}`` from a ``bounds_source`` cell; a side it does not name is
    absent."""
    if cell is None or pd.isna(cell):
        return {}
    out: dict[str, str] = {}
    for part in str(cell).split(";"):
        name, sep, value = part.partition("=")
        if sep and name.strip() in _SIDES and value.strip():
            out[name.strip()] = value.strip()
    return out


def _index_cell(value: Any) -> Any:
    return pd.NA if value is None or pd.isna(value) else int(value)


def join_edge_fits(own: Mapping[str, str], bounds: tuple[int, int], *,
                   shared: str | None = None) -> str | None:
    """One ``fit_json`` from the record of the fit that set each edge (``own``: edge name -> that
    record, start first), in the shape the trace keys store for edges set by two fits (cursor
    design 5.5): each edge's entry, window (``W``), rise and placed position from its own record,
    an edge no record set marked "not requested" at its bound, and the shared parameters from
    ``shared`` when it is one of the records (the trace keys pass the latest fit's), else from the
    first. One record that set both edges comes back as it is; ``None`` when there is none."""
    if not own:
        return None
    if len(own) == 2 and own["start"] == own["end"]:
        return own["start"]
    fits = {name: json.loads(text) for name, text in own.items()}
    out = json.loads(shared if shared in own.values() else next(iter(own.values())))
    for key in ("W", "rise"):
        part = {name: fit[key][name] for name, fit in fits.items() if name in (fit.get(key) or {})}
        if part or key in out:
            out[key] = part
    out["edges"] = {
        name: fits[name]["edges"][name] if name in fits else {
            "pos": int(pos), "fitted": False, "z": 0.0, "step": 0.0,
            "reason": REASON_NOT_REQUESTED}
        for name, pos in zip(_SIDES, bounds)}
    out["placed"] = [fits[name]["placed"][i] if name in fits else int(bounds[i])
                     for i, name in enumerate(_SIDES)]
    out["edges_requested"] = list(fits)
    return json.dumps(out, separators=(",", ":"), sort_keys=True, allow_nan=False)


def merged_edge_cells(first: Mapping[str, Any], last: Mapping[str, Any],
                      bounds: tuple[int, int]) -> dict[str, Any]:
    """The edge record of one row made of several (a merge), as its four cells: the start side as
    ``first`` (the fragment whose start is the row's start) recorded it and the end side as
    ``last`` (the one whose end is the row's end) recorded it, cursor design 5.5.
    ``bounds_source`` joins the two sides' parts, ``drawn_start_idx`` and ``drawn_end_idx`` come
    from their own fragments, and ``fit_json`` holds each side's own fit (:func:`join_edge_fits`),
    or is ``<NA>`` when neither side is ``fit:step``, as the add door stores it.

    A side whose fragment recorded nothing (a detected row) records nothing here either: its part
    of ``bounds_source`` is left out, its drawn bound is ``<NA>``, and its ``fit_json`` entry is
    the "not requested" one an edge no fit set gets. The vocabulary has no word for a detector's
    own edge (``fit:schmitt`` and ``fit:cusum`` name the table rule run on a hand-made row), and
    the row's ``source`` already says ``mixed``. With nothing on either side every cell is
    ``<NA>``, as on a detected row."""
    s_src = _edge_sources(first.get(BOUNDS_SOURCE_COL)).get("start")
    e_src = _edge_sources(last.get(BOUNDS_SOURCE_COL)).get("end")
    sources = ";".join(f"{name}={src}" for name, src in zip(_SIDES, (s_src, e_src)) if src)
    own = {name: str(row.get(FIT_JSON_COL))
           for name, row, src in zip(_SIDES, (first, last), (s_src, e_src))
           if src not in (None, "drawn") and not pd.isna(row.get(FIT_JSON_COL, pd.NA))}
    fit = join_edge_fits(own, bounds) if "fit:step" in (s_src, e_src) else None
    return {BOUNDS_SOURCE_COL: _scalar(sources or None),
            DRAWN_START_COL: _index_cell(first.get(DRAWN_START_COL)),
            DRAWN_END_COL: _index_cell(last.get(DRAWN_END_COL)),
            FIT_JSON_COL: _scalar(fit)}


# ---------------------------------------------------------------------------
# A re-run keeps what people drew
# ---------------------------------------------------------------------------
#: A hand row and a detected row are "the same event" when at least half of the SHORTER span is
#: shared: reconcile's overlap coefficient (`ReconcileConfig.min_overlap_fraction`), not IoU.
CARRY_MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class CarryReport:
    """What a re-run did with the hand-made rows of the table it replaced.

    Every tuple is made of the ids a person can find: the OLD table's ``event_id`` first.
    """

    #: ``(old_id, new_id)``: kept as hand-made rows, re-measured under the new run.
    kept: tuple[tuple[int, int], ...] = ()
    #: ``(old_id, detected_id)``: the new run found the same event; the detected row stays and
    #: the hand row's labels, note and ruling moved onto it.
    moved: tuple[tuple[int, int], ...] = ()
    #: ``(old_id, removed_detected_id)``: the detector's own row for it was rejected by an
    #: automatic rule, which does not outvote a ruling; the hand row stays and that row goes.
    replaced_rejected: tuple[tuple[int, int], ...] = ()
    #: ``(old_id, detected_id)``: overlaps a detected row by less than half the shorter span.
    #: Both stay, for the person to resolve.
    unresolved: tuple[tuple[int, int], ...] = ()
    #: ``(old_id, reason)``: left out, because the new run never searched there.
    left_out: tuple[tuple[int, str], ...] = ()

    @property
    def n_hand_rows(self) -> int:
        """How many hand-made rows the replaced table held."""
        return len(self.kept) + len(self.moved) + len(self.left_out)

    @property
    def carried_ids(self) -> tuple[int, ...]:
        """New-table ids of the rows this carry-over added (not the detector's)."""
        return tuple(new for _old, new in self.kept)

    @property
    def dropped_ids(self) -> tuple[int, ...]:
        """New-table ids of detector rows the carry-over removed."""
        return tuple(det for _old, det in self.replaced_rejected)

    def sentence(self, table_name: str, n_events: int) -> str:
        """The one status line: what was kept and what was lost."""
        if not (self.kept or self.moved or self.left_out):
            return f"Re-ran {table_name}: {n_events:,} events."
        parts = [f"Re-ran {table_name}: {n_events:,} events."]
        if self.kept or self.moved:
            n = len(self.kept) + len(self.moved)
            text = f"Kept {n} hand-made event{'s' if n != 1 else ''}"
            if self.moved:
                ids = ", ".join(str(d) for _o, d in self.moved[:6])
                more = "…" if len(self.moved) > 6 else ""
                text += (f"; {len(self.moved)} {'is' if len(self.moved) == 1 else 'are'} now "
                         f"detected ({ids}{more}), their labels moved onto "
                         f"{'it' if len(self.moved) == 1 else 'them'}")
            parts.append(text + ".")
        if self.unresolved:
            n = len(self.unresolved)
            parts.append(f"{n} overlap{'s' if n == 1 else ''} a detected event partly: both "
                         "stay, for you to resolve.")
        if self.left_out:
            n = len(self.left_out)
            parts.append(f"{n} hand-made event{'s were' if n != 1 else ' was'} left out because "
                         "this run did not search there.")
        return " ".join(parts)


def _overlap_coefficients(starts: np.ndarray, ends: np.ndarray, s: int, e: int) -> np.ndarray:
    """Overlap coefficient of ``[s, e)`` against each ``[starts, ends)``: shared samples divided
    by the shorter span's length (0 where they do not meet)."""
    shared = np.minimum(ends, e) - np.maximum(starts, s)
    shared = np.clip(shared, 0, None).astype(float)
    shorter = np.minimum(ends - starts, e - s).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(shorter > 0, shared / shorter, 0.0)
    return out


def carry_hand_rows(
    new_events: EventSet,
    old_df: pd.DataFrame,
    signal: np.ndarray,
    *,
    fs: float,
    units: str | None,
    old_module_id: str,
) -> tuple[EventSet, CarryReport]:
    """``new_events`` (a fresh detection) with the hand-made rows of ``old_df`` carried into it.

    Called once, at the single place a detection replaces a table, so a re-run no longer throws
    away what a person drew. Each hand row is re-measured under the NEW run's provenance with
    ``prior=row`` (a hand rejection and the curator's own flag tokens, labels included, survive;
    automatic tokens are re-derived), keeps its ``note`` and bounds record, and gets ``origin``
    ``"<old module>:<old event_id>"`` and a new ``event_id``.

    * A hand row the new run ALSO detected (overlap coefficient >= :data:`CARRY_MIN_OVERLAP`
      with a viable detected row): the detected row stays, and the hand row's non-derived flag
      tokens (labels), note, review and any hand rejection move onto it. When several hand rows
      match one detected row, their notes are joined in time order, each once (:func:`join_notes`).
    * ...with a detected row an AUTOMATIC rule rejected: the hand row stays and that row goes,
      since an automatic rejection does not outvote a ruling.
    * Overlapping by less than half: both stay, listed.
    * Outside the span the new run analysed, or inside a region it excluded: left out, listed.
    """
    from .characterize import recharacterize_span
    from .detect import _exclusion_samples
    from .events import (
        DERIVED_FLAGS, combine_flags, flag_tokens, is_automatic_rejection, mark_reviewed,
    )

    hand = hand_made_mask(old_df)
    if not hand.any():
        return new_events, CarryReport()
    prov = new_events.provenance
    new_df = new_events.df.copy()
    n = int(np.asarray(signal).size)
    a0, a1 = analysed_span_for(prov, n)
    try:
        excluded = _exclusion_samples(prov.get("exclude_regions") or None, n, float(fs))
    except (TypeError, ValueError):
        excluded = []

    old_rows = old_df.loc[hand].sort_values("start_idx", kind="stable")
    next_id = (int(new_df["event_id"].max()) + 1) if len(new_df) else 0
    kept: list[tuple[int, int]] = []
    moved: list[tuple[int, int]] = []
    replaced: list[tuple[int, int]] = []
    unresolved: list[tuple[int, int]] = []
    left_out: list[tuple[int, str]] = []
    drop_ids: set[int] = set()
    add_rows: list[pd.DataFrame] = []
    # Every note moved onto a detected row so far, by the row's position, in time order: the
    # row's note is joined from this list (`join_notes`) each time another hand row lands on it.
    landed_notes: dict[int, list[Any]] = {}

    for _i, row in old_rows.iterrows():
        old_id = int(row["event_id"])
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if s < a0 or e > a1:
            left_out.append((old_id, "outside the span this run searched"))
            continue
        if any(s < x1 and e > x0 for x0, x1 in excluded):
            left_out.append((old_id, "inside a region this run excluded"))
            continue

        det_ids = new_df["event_id"].to_numpy(dtype=np.int64)
        live = ~np.isin(det_ids, list(drop_ids)) if drop_ids else np.ones(len(new_df), dtype=bool)
        cov = _overlap_coefficients(
            new_df["start_idx"].to_numpy(dtype=np.int64),
            new_df["end_idx"].to_numpy(dtype=np.int64), s, e)
        cov = np.where(live, cov, 0.0)
        best = int(np.argmax(cov)) if len(cov) else -1
        if best >= 0 and cov[best] >= CARRY_MIN_OVERLAP:
            det_flag = new_df["flag"].iloc[best]
            det_accepted = bool(new_df["accepted"].iloc[best])
            auto_rejected = (not det_accepted) and is_automatic_rejection(det_flag)
            det_id = int(det_ids[best])
            if not auto_rejected:
                # The detected row stays; what a person said about the event moves onto it.
                moved_tokens = [t for t in flag_tokens(row["flag"]) if t not in DERIVED_FLAGS]
                new_df.iat[best, new_df.columns.get_loc("flag")] = combine_flags(
                    det_flag, *moved_tokens)
                if (not bool(row["accepted"])) and not is_automatic_rejection(row["flag"]):
                    new_df.iat[best, new_df.columns.get_loc("accepted")] = False
                note = row.get(NOTE_COL, pd.NA)
                if not pd.isna(note):
                    if NOTE_COL not in new_df.columns:
                        new_df[NOTE_COL] = pd.Series([pd.NA] * len(new_df), index=new_df.index,
                                                     dtype="string")
                    # Joined with what earlier hand rows moved there: two halves drawn by hand
                    # of an event the run finds whole both land on this row, and the first
                    # note used to be overwritten by the second (S9-06). Joined from the whole
                    # list each time, so a third row repeating the first's note adds nothing.
                    at = new_df.columns.get_loc(NOTE_COL)
                    landed = landed_notes.setdefault(best, [new_df.iat[best, at]])
                    landed.append(note)
                    new_df.iat[best, at] = join_notes(*landed)
                mark_reviewed(new_df, rows=np.array([best], dtype=np.int64))
                moved.append((old_id, det_id))
                continue
            drop_ids.add(det_id)
            replaced.append((old_id, det_id))
        else:
            hits = np.flatnonzero(cov > 0)
            if hits.size:
                for h in hits[:1]:
                    unresolved.append((old_id, int(det_ids[int(h)])))
        # carried as a hand-made row, re-measured under the new run
        measured = recharacterize_span(
            signal, float(fs), s, e, provenance=prov, prior=row,
            recording=str(row["recording"]), pore_id=str(row["pore_id"]), units=units or "pA")
        if len(measured) == 0:
            left_out.append((old_id, "could not be re-measured"))
            continue
        fresh = measured.iloc[0].to_dict()
        fresh["event_id"] = next_id
        fresh[ADDED_COLUMN] = True
        fresh[SOURCE_COL] = SOURCE_MANUAL
        fresh[REVIEW_COL] = REVIEW_DONE
        for col in ORIGINAL_BOUNDS_SCHEMA:
            fresh[col] = pd.NA
        for col in (NOTE_COL, LEGACY_IDS_COL, BOUNDS_SOURCE_COL, DRAWN_START_COL, DRAWN_END_COL,
                    FIT_JSON_COL):
            fresh[col] = row.get(col, pd.NA)
        fresh[ORIGIN_COL] = f"{old_module_id}:{old_id}"
        add_rows.append(pd.DataFrame([fresh]))
        kept.append((old_id, next_id))
        next_id += 1

    if drop_ids:
        new_df = new_df.loc[~new_df["event_id"].isin(list(drop_ids))].reset_index(drop=True)
    for frame in add_rows:
        new_df = insert_hand_row(new_df, frame)

    report = CarryReport(tuple(kept), tuple(moved), tuple(replaced), tuple(unresolved),
                         tuple(left_out))
    stamped = dict(prov)
    stamped["carried_hand_rows"] = {
        "carried_ids": list(report.carried_ids),
        "dropped_ids": list(report.dropped_ids),
        "moved": [list(p) for p in report.moved],
        "unresolved": [list(p) for p in report.unresolved],
        "left_out": [[i, r] for i, r in report.left_out],
        "from": str(old_module_id),
    }
    return EventSet(new_df, stamped), report


def analysed_span_for(provenance: Mapping[str, Any], n: int) -> tuple[int, int]:
    """`events.analysed_span` under a name that reads at the call site."""
    from .events import analysed_span
    return analysed_span(provenance, n)


# ---------------------------------------------------------------------------
# Copy hand-made events from one table to another
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CopyReport:
    """What :func:`copy_hand_rows` did, by old event id, so nothing is skipped silently."""

    #: ``(source_id, new_id)``: copied, re-measured under the target's provenance.
    copied: tuple[tuple[int, int], ...] = ()
    #: source ids whose ``origin`` is already in the target: copying twice copies nothing twice.
    already_copied: tuple[int, ...] = ()
    #: ``(source_id, target_event_id, rejected)``: overlaps an event the target already holds.
    #: ``rejected`` is True when that event is a rejected candidate, whose mask has to be dragged
    #: off the span first.
    overlapping: tuple[tuple[int, int, bool], ...] = ()
    #: ``(source_id, reason)``: outside what the target searched, or could not be measured.
    left_out: tuple[tuple[int, str], ...] = ()
    #: Hand-made rows of the source that were rejected and so not offered (``only_accepted``).
    n_rejected_skipped: int = 0

    def sentence(self, source_name: str, target_name: str) -> str:
        n = len(self.copied)
        parts = [f"Copied {n} hand-made event{'s' if n != 1 else ''} from {source_name} to "
                 f"{target_name}."]
        if self.already_copied:
            m = len(self.already_copied)
            parts.append(f"{m} {'was' if m == 1 else 'were'} already there (copied before).")
        if self.overlapping:
            m = len(self.overlapping)
            ids = ", ".join(str(t) for _s, t, _r in self.overlapping[:5])
            parts.append(f"{m} overlap{'s' if m == 1 else ''} events {target_name} already "
                         f"holds ({ids}{'…' if m > 5 else ''}) and {'was' if m == 1 else 'were'} "
                         "not copied.")
        if self.left_out:
            m = len(self.left_out)
            parts.append(f"{m} left out: {self.left_out[0][1]}"
                         + ("." if m == 1 else " (and others)."))
        if self.n_rejected_skipped:
            parts.append(f"{self.n_rejected_skipped} rejected hand-made "
                         f"event{'s' if self.n_rejected_skipped != 1 else ''} not offered.")
        parts.append("The copies are independent: editing one does not change the other.")
        return " ".join(parts)


def copy_hand_rows(
    source_df: pd.DataFrame,
    target: EventSet,
    signal: np.ndarray,
    *,
    fs: float,
    units: str | None,
    source_module_id: str,
    only_accepted: bool = True,
) -> tuple[EventSet, CopyReport]:
    """``target`` with the hand-made rows of ``source_df`` copied into it.

    Each copy is re-measured under the TARGET's provenance (``signal`` is the array the target
    was measured on), so a copy is comparable with the target's own rows; it carries the
    source row's note and bounds record, and ``origin`` ``"<source module>:<source event_id>"``
    -- the key that makes the operation idempotent. A row is NOT copied when: its origin is
    already in the target; it is outside the span the target searched or inside a region it
    excluded; it overlaps an event the target holds (a rejected candidate included: drag that
    candidate's mask off the span first); it could not be measured. Rejected hand-made rows are
    not offered unless ``only_accepted`` is False. The source is never touched, and the copies
    are independent of it from here on.
    """
    from .detect import _exclusion_samples
    from .events import analysed_span

    hand = hand_made_mask(source_df)
    prov = target.provenance
    df = target.df
    n = int(np.asarray(signal).size)
    a0, a1 = analysed_span(prov, n)
    try:
        excluded = _exclusion_samples(prov.get("exclude_regions") or None, n, float(fs))
    except (TypeError, ValueError):
        excluded = []
    # Every id a cell lists: a merge of two copies keeps both origins (`join_ids`), and each of
    # them is a row this target already holds.
    have_origin = ({o for cell in df[ORIGIN_COL] for o in _id_parts(cell)}
                   if ORIGIN_COL in df.columns else set())
    rows = source_df.loc[hand].sort_values("start_idx", kind="stable")
    copied: list[tuple[int, int]] = []
    already: list[int] = []
    overlapping: list[tuple[int, int, bool]] = []
    left_out: list[tuple[int, str]] = []
    skipped_rejected = 0
    next_id = (int(df["event_id"].max()) + 1) if len(df) else 0
    recording = str(df["recording"].iloc[0]) if len(df) else None
    pore_id = str(df["pore_id"].iloc[0]) if len(df) else None
    out = df
    for _i, row in rows.iterrows():
        old_id = int(row["event_id"])
        if only_accepted and not bool(row["accepted"]):
            skipped_rejected += 1
            continue
        origin = f"{source_module_id}:{old_id}"
        if origin in have_origin:
            already.append(old_id)
            continue
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if s < a0 or e > a1:
            left_out.append((old_id, "outside the span the target searched"))
            continue
        if any(s < x1 and e > x0 for x0, x1 in excluded):
            left_out.append((old_id, "inside a region the target excluded"))
            continue
        hits = np.flatnonzero(
            (out["start_idx"].to_numpy() < e) & (out["end_idx"].to_numpy() > s)) if len(out) else []
        if len(hits):
            hit = out.iloc[int(hits[0])]
            overlapping.append((old_id, int(hit["event_id"]), not bool(hit["accepted"])))
            continue
        try:
            fresh = measure_hand_row(
                signal, fs, s, e, provenance=prov, units=units,
                recording=recording if recording is not None else str(row["recording"]),
                pore_id=pore_id if pore_id is not None else str(row["pore_id"]),
                event_id=next_id,
                note=None if pd.isna(row.get(NOTE_COL, pd.NA)) else str(row[NOTE_COL]),
                origin=origin,
                edge=EdgeRecord(
                    bounds_source=None if pd.isna(row.get(BOUNDS_SOURCE_COL, pd.NA))
                    else str(row[BOUNDS_SOURCE_COL]),
                    drawn_start=None if pd.isna(row.get(DRAWN_START_COL, pd.NA))
                    else int(row[DRAWN_START_COL]),
                    drawn_end=None if pd.isna(row.get(DRAWN_END_COL, pd.NA))
                    else int(row[DRAWN_END_COL]),
                    fit_json=None if pd.isna(row.get(FIT_JSON_COL, pd.NA))
                    else str(row[FIT_JSON_COL])))
        except (ValueError, IndexError) as exc:
            left_out.append((old_id, f"could not be measured ({exc})"))
            continue
        # The ruling travels with the event: a hand rejection (not offered by default) stays one.
        if not only_accepted and not bool(row["accepted"]):
            fresh.loc[:, "accepted"] = False
            fresh.loc[:, "flag"] = row["flag"]
        out = insert_hand_row(out, fresh)
        have_origin.add(origin)
        copied.append((old_id, next_id))
        next_id += 1
    report = CopyReport(tuple(copied), tuple(already), tuple(overlapping), tuple(left_out),
                        skipped_rejected)
    return EventSet(out, dict(prov)), report
