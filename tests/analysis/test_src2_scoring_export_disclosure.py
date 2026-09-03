"""EXPORT-1: the export must DISCLOSE what it withheld, and the training table must not
grow a leaky column.

Two halves, both in ``analysis/scoring.py``:

* **The events export.** ``Analysis > Export events`` writes ``events.accepted.df``. The
  sidecar already said HOW MANY rows did not come; it did not say WHAT they were, and "3
  rows were withheld" cannot tell a curator throwing away three artifacts from the
  detector clipping three at the recording boundary -- opposite facts about the dataset.
  Hand :func:`~nanoscope.analysis.scoring.write_events_export` the source table and the
  ``curation`` block breaks the withheld rows down the same two ways it already breaks
  down the ones that came. The three ways of knowing the total are now tried in order
  instead of one-shot, so a caller that guesses badly no longer DELETES the disclosure
  the provenance fallback would have given.

* **The training export.** ``source``/``review`` are deliberately NOT columns of
  ``TRAINING_COLUMNS`` -- a ``source`` column is a leaky feature that teaches a model to
  reproduce the detector that found the row, which is the one thing training on the
  CURATED set exists to avoid, and ``added``/``accepted`` already carry the two labels.
  The counts go in the sidecar instead: the additive door, counts not columns.

AND NO MEASURED NUMBER MOVED. Every test that adds a sidecar key re-hashes the CSV body
beside it, because this is description, not measurement.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import events as ev
from nanoscope.analysis import scoring

FS = 50_000.0


# ================================================================================================
# builders
# ================================================================================================

def _detection(n: int = 16) -> tuple[pd.DataFrame, dict]:
    """A stamped 20-column detection with THREE kinds of not-accepted row.

    Two the DETECTOR de-accepted (``clipped_by_boundary`` -- `events.is_automatic_rejection`
    is the shared definition) and one a PERSON rejected (review recorded, no automatic
    reason). That difference is the whole of what the withheld breakdown exists to say.
    """
    rng = np.random.default_rng(7)
    starts = np.arange(n, dtype=np.int64) * 5000 + 1000
    df = pd.DataFrame({c: np.zeros(n, dtype=float) for c in ev.SCHEMA})
    df["start_idx"] = starts
    df["end_idx"] = starts + 300
    df["t_start_s"] = starts / FS
    df["t_end_s"] = (starts + 300) / FS
    df["dwell_s"] = 300.0 / FS
    df["depth_pA"] = rng.normal(50.0, 3.0, n)
    df["baseline_pA"] = -34.0
    df["accepted"] = True
    df["flag"] = pd.array([pd.NA] * n, dtype="string")
    df["event_id"] = np.arange(n, dtype=np.int64)
    df = ev.stamp_source_status(df, "detect.schmitt")

    df.loc[df.index[0], "accepted"] = False
    df.loc[df.index[0], "flag"] = "clipped_by_boundary"
    df.loc[df.index[1], "accepted"] = False
    df.loc[df.index[1], "flag"] = "clipped_by_boundary"
    df.loc[df.index[2], "accepted"] = False
    df.loc[df.index[2], ev.REVIEW_COL] = ev.REVIEW_DONE          # a person ruled: rejected

    prov = {"fs": FS, "n_events": n, "polarity": -1, "units": "pA"}
    return df, prov


def _accepted(df: pd.DataFrame) -> pd.DataFrame:
    """What ``main_window._export_events`` hands the writer: ``events.accepted.df``."""
    return df[df["accepted"].to_numpy(dtype=bool)]


def _block(out: dict) -> dict:
    return json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]


def _body_hash(path) -> str:
    """SHA-256 of the CSV bytes -- the table itself, independent of any sidecar."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen20_hash(path) -> str:
    """SHA-256 of the 20 frozen columns as written, re-read from the file."""
    body = pd.read_csv(path)
    cols = [c for c in ev.SCHEMA if c in body.columns]
    assert cols == list(ev.SCHEMA)
    return hashlib.sha256(
        (",".join(cols) + "\n" + body[cols].to_csv(index=False, float_format="%.17g")
         ).encode()).hexdigest()


# ================================================================================================
# (a) the events export -- what was withheld, not only how much
# ================================================================================================

def test_the_sidecar_names_what_the_withheld_rows_were_not_only_how_many(tmp_path):
    """THE DEFECT. ``n_withheld: 3`` over a table where one row is a curator's rejection
    and two are the detector's own auto-rejections describes neither. A reader outside
    this app cannot tell "the user threw away three artifacts" from "three events ran off
    the end of the recording", and those license opposite conclusions about the pore."""
    df, prov = _detection()

    out = scoring.write_events_export(_accepted(df), tmp_path / "ev.csv",
                                      provenance=prov, source_df=df)
    block = _block(out)

    assert block["n_withheld"] == 3
    assert block["withheld_by_status"] == {ev.STATUS_AUTO_REJECTED: 2,
                                           ev.STATUS_REJECTED: 1}
    assert block["withheld_by_source"] == {"detect.schmitt": 3}


def test_the_source_table_outranks_a_count_because_a_table_cannot_be_stale(tmp_path):
    """The three ways of knowing the total are ordered. The frame itself wins: it is not
    a claim ABOUT the table, it is the table."""
    df, prov = _detection()
    prov["n_events"] = 99                                   # a wrong claim, deliberately

    out = scoring.write_events_export(_accepted(df), tmp_path / "ev.csv",
                                      provenance=prov, n_total=99, source_df=df)
    block = _block(out)

    assert block["counted_from"] == "source_table"
    assert block["n_in_table"] == len(df) and block["n_withheld"] == 3


def test_a_bad_caller_total_falls_back_instead_of_deleting_the_disclosure(tmp_path):
    """A total SMALLER than the rows in hand is refused -- a negative "withheld" is the
    fabricated number this key exists to stop. But refusing it must not end the search:
    the old rule stopped at the first candidate, so one wrong keyword from the GUI wiring
    would have silently replaced a correct disclosure with "this file does not know"."""
    df, prov = _detection()
    accepted = _accepted(df)

    bad = scoring.write_events_export(accepted, tmp_path / "bad.csv",
                                      provenance=prov, n_total=3)
    silent = scoring.write_events_export(accepted, tmp_path / "silent.csv",
                                         provenance=prov)

    assert _block(bad)["n_withheld"] == 3
    assert _block(bad)["counted_from"] == "provenance.n_events"
    assert _block(bad) == _block(silent)                    # the bad hint changed nothing


def test_nothing_at_all_still_reports_unknown_and_never_zero(tmp_path):
    """With no source table, no caller total and a stale provenance, the block says it
    does not know. ``0`` would assert that nothing was withheld."""
    df, prov = _detection()
    prov["n_events"] = 2

    out = scoring.write_events_export(df, tmp_path / "ev.csv", provenance=prov)
    block = _block(out)

    assert block["n_in_table"] is None and block["n_withheld"] is None
    assert block["counted_from"] is None
    assert block["withheld_by_status"] is None and block["withheld_by_source"] is None


def test_a_source_table_that_is_not_this_table_describes_nothing(tmp_path):
    """The breakdown is matched on the frame's own identity, never on position. A caller
    that reset the index would otherwise make "rows 0-12 of 0-15" name THE WRONG THREE
    ROWS, and a confidently wrong breakdown is worse here than none: this block is the
    only thing on disk a reader can check the CSV against."""
    df, _ = _detection()
    accepted = _accepted(df).copy()
    accepted["event_id"] = np.arange(len(accepted), dtype=np.int64)   # renumbered
    accepted["start_idx"] = np.arange(len(accepted), dtype=np.int64)

    out = scoring.write_events_export(accepted, tmp_path / "ev.csv", source_df=df)
    block = _block(out)

    assert block["withheld_by_status"] is None
    assert block["withheld_by_source"] is None
    assert block["n_in_table"] == len(df)          # the COUNT is still safe to state
    assert block["n_withheld"] == 3


def test_a_full_export_says_nothing_was_withheld_rather_than_claiming_it_was_curated(
        tmp_path):
    """The include-rejected export is a GUI option away (needs_wiring). When it lands the
    writer must already describe it correctly: no withholding, and a note that does not
    tell the reader the file "holds the accepted rows only"."""
    df, prov = _detection()

    out = scoring.write_events_export(df, tmp_path / "ev.csv", provenance=prov,
                                      source_df=df)
    block = _block(out)

    assert block["n_withheld"] == 0
    assert block["withheld_by_status"] == {} and block["withheld_by_source"] == {}
    assert "Nothing was withheld" in block["note"]
    assert "holds the accepted rows only" not in block["note"]


def test_an_export_of_nothing_names_every_row_as_withheld(tmp_path):
    """Everything rejected is a path the user reaches on purpose."""
    df, prov = _detection()

    out = scoring.write_events_export(df.iloc[:0], tmp_path / "ev.csv",
                                      provenance=prov, source_df=df)
    block = _block(out)

    assert block["n_withheld"] == len(df)
    assert sum(block["withheld_by_source"].values()) == len(df)


def test_the_new_keyword_moves_no_number_in_the_csv_beside_it(tmp_path):
    """THE INVARIANT, at this seam. ``source_df`` writes into the sidecar and nowhere
    else: the CSV bytes are identical with and without it."""
    df, prov = _detection()
    accepted = _accepted(df)

    plain = scoring.write_events_export(accepted, tmp_path / "a.csv", provenance=prov)
    described = scoring.write_events_export(accepted, tmp_path / "b.csv",
                                            provenance=prov, source_df=df)

    assert _body_hash(plain["data"]) == _body_hash(described["data"])
    assert _frozen20_hash(plain["data"]) == _frozen20_hash(described["data"])


def test_the_curation_block_still_describes_the_rows_that_did_come(tmp_path):
    """The half that already worked keeps working -- the new keys sit beside it."""
    df, prov = _detection()

    out = scoring.write_events_export(_accepted(df), tmp_path / "ev.csv",
                                      provenance=prov, source_df=df)
    block = _block(out)

    assert block["columns"] == [ev.SOURCE_COL, ev.REVIEW_COL]
    assert block["by_source"] == {"detect.schmitt": 13}
    assert block["by_kind"] == {ev.KIND_CONVENTIONAL: 13}
    assert sum(block["by_status"].values()) == 13


# ================================================================================================
# (b) the training export -- the decision, and the door that was taken instead
# ================================================================================================

def test_the_training_table_does_not_carry_source_because_it_would_be_a_leaky_feature():
    """THE DECISION, pinned. A ``source`` column tells a model which detector found the
    row; a model trained on the CURATED set with that column learns to reproduce the
    detector instead of the signal, which is the one thing this loop exists to escape --
    the owner's note says final analyses use all accepted events "regardless of how they
    were detected". ``added`` already carries the manual/automatic split the labels need,
    and ``review`` is a fact about the process whose PRODUCT is ``accepted``."""
    df, _ = _detection()

    frame = scoring.build_training_frame(df)

    assert list(frame.columns) == scoring.TRAINING_COLUMNS
    assert ev.SOURCE_COL not in frame.columns
    assert ev.REVIEW_COL not in frame.columns


def test_the_training_sidecar_carries_the_counts_the_columns_do_not(tmp_path):
    """THE DEFECT'S OTHER HALF. The owner feeds this file to his models, and it said
    ``selection: accepted`` with no way to know which detector found the rows or how many
    did not come. Counts, not columns: the additive door."""
    df, prov = _detection()

    out = scoring.write_training_export(df, tmp_path / "train.csv", provenance=prov,
                                        source_file="r.abf", detector="detect.schmitt")
    meta = json.loads(out["meta"].read_text())
    block = meta[scoring.CURATION_BLOCK_KEY]

    assert block["by_source"] == {"detect.schmitt": 13}
    assert block["by_kind"] == {ev.KIND_CONVENTIONAL: 13}
    assert block["n_in_table"] == len(df) and block["n_withheld"] == 3
    assert block["counted_from"] == "source_table"
    assert block["withheld_by_status"] == {ev.STATUS_AUTO_REJECTED: 2,
                                           ev.STATUS_REJECTED: 1}


def test_the_training_sidecar_counts_agree_with_the_table_beside_it(tmp_path):
    """Two writers for one fact drift. The block and the written table come from ONE
    definition of which rows go (``_training_row_mask``), so they cannot."""
    df, prov = _detection()

    out = scoring.write_training_export(df, tmp_path / "train.csv", provenance=prov)
    meta = json.loads(out["meta"].read_text())
    written = pd.read_csv(out["data"])
    block = meta[scoring.CURATION_BLOCK_KEY]

    assert len(written) == meta["n_rows"] == 13
    assert sum(block["by_source"].values()) == len(written)
    assert sum(block["by_status"].values()) == len(written)
    assert block["n_withheld"] == len(df) - len(written)


def test_an_include_all_training_export_is_described_as_the_full_table(tmp_path):
    """``include_all=True`` withholds nothing, and the note must not tell the reader the
    file holds the accepted rows only."""
    df, prov = _detection()

    out = scoring.write_training_export(df, tmp_path / "train.csv", provenance=prov,
                                        include_all=True)
    meta = json.loads(out["meta"].read_text())
    block = meta[scoring.CURATION_BLOCK_KEY]

    assert meta["selection"] == "all" and meta["n_rows"] == len(df)
    assert block["n_withheld"] == 0
    assert "Nothing was withheld" in block["note"]
    assert sum(block["by_status"].values()) == len(df)


def test_the_training_csv_body_is_byte_identical_to_one_written_without_the_block(
        tmp_path):
    """THE INVARIANT for the file the models actually read. The block is a sidecar key;
    the table is untouched, column set and bytes alike."""
    df, prov = _detection()

    out = scoring.write_training_export(df, tmp_path / "train.csv", provenance=prov)
    reference = scoring.build_training_frame(df)
    reference_csv = tmp_path / "reference.csv"
    reference.to_csv(reference_csv, index=False)

    assert list(pd.read_csv(out["data"]).columns) == scoring.TRAINING_COLUMNS
    assert _body_hash(out["data"]) == _body_hash(reference_csv)


def test_a_training_export_of_a_legacy_frame_says_the_columns_are_absent(tmp_path):
    """A project written before the pair existed must read as "not recorded", never as
    "one detector, all reviewed"."""
    df, prov = _detection()
    legacy = df.drop(columns=[ev.SOURCE_COL, ev.REVIEW_COL])

    out = scoring.write_training_export(legacy, tmp_path / "train.csv", provenance=prov)
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert block["columns"] == []
    assert block["by_source"] is None and block["by_kind"] is None
    assert block["withheld_by_source"] is None
    assert block["n_withheld"] == 3                     # the count survives regardless


def test_one_writer_describes_both_exports_the_same_way(tmp_path):
    """``curation_stamp`` is THE writer of this block. The events sidecar and the
    training sidecar for the same curation must not come to disagree."""
    df, prov = _detection()

    events_out = scoring.write_events_export(_accepted(df), tmp_path / "ev.csv",
                                             provenance=prov, source_df=df)
    train_out = scoring.write_training_export(df, tmp_path / "train.csv", provenance=prov)

    assert _block(events_out) == json.loads(
        train_out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]


@pytest.mark.parametrize("include_all", [False, True])
def test_the_row_mask_is_the_selection_the_frame_builder_makes(include_all):
    """The shared definition, asserted against the builder it was factored out of."""
    df, _ = _detection()

    mask = scoring._training_row_mask(df, include_all=include_all)
    frame = scoring.build_training_frame(df, include_all=include_all)

    assert int(mask.sum()) == len(frame)
    np.testing.assert_array_equal(
        frame["start_idx"].to_numpy(), df.loc[mask, "start_idx"].to_numpy())
