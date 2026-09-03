"""``source``/``review`` through the two durable doors: the project file and the export.

The pair is declared OUTSIDE ``events.SCHEMA`` on purpose (see the block at
``analysis/events.py``), which buys backward compatibility -- an old project has neither
column and still loads -- at the price that every path has to carry them DELIBERATELY.
Two of those paths are here:

* ``core/project.py`` writes each result's frame WHOLE to ``results/<i>.parquet``. That
  already carried the pair; what this file pins is that it keeps doing so, that the
  dtype survives, that ``pd.NA`` survives as ``pd.NA``, and that a project written before
  the columns existed still opens and reads "unknown" rather than raising or guessing.
* ``analysis/scoring.write_events_export`` writes the CSV a co-author actually receives.
  The columns ride in the table; the ``curation`` block in the ``.meta.json`` sidecar is
  what tells that reader the CSV holds the ACCEPTED rows only, and how many did not come.

AND THAT NO NUMBER MOVED. Every test that adds a column also re-hashes the 20 frozen
columns, because this change is description, not measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.analysis import events as ev
from nanoscope.analysis import scoring
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.core.document import CalibratedChannel
from nanoscope.core.project import (
    ProjectState,
    ResultRec,
    load_project,
    save_project,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]

#: The pre-change tree, kept beside the repo for exactly this: a project file written by
#: a build that had never heard of these columns. Tests that need it SKIP when it is not
#: there rather than failing -- it is evidence, not a dependency of the code.
_SNAPSHOT = Path("/Users/abrahamcampos/Desktop/nanoscope-reports/pre-layout-2026-09-02")


# ================================================================================================
# builders -- the real Schmitt detector, so the frame is the genuine 20-column article
# ================================================================================================

def _trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _detection() -> tuple[pd.DataFrame, dict]:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=arr, units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(
        path=Path("pore_a.csv"), format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    res = DetectSchmitt().run(
        SimpleAnalysisContext(channel=ch, meta=meta, params={"min_dwell_s": 5e-4}))
    return res.payload.df.copy(), dict(res.payload.provenance)


def _legacy(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` as a build that had never heard of these columns would have produced it.

    The detectors STAMP the pair now (``modules.DetectSchmitt`` ends with
    ``stamp_source_status(stamp_original_bounds(df), self.id)``), so "a table without
    them" has to be made on purpose -- which is exactly what a project saved before that
    wiring holds.
    """
    return df.drop(columns=[c for c in (ev.SOURCE_COL, ev.REVIEW_COL) if c in df.columns])


def _frozen20_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the 20 frozen columns, at full float precision.

    The whole invariant in one line: this string may not move, whatever description is
    bolted on beside it.
    """
    return hashlib.sha256(
        df[list(ev.SCHEMA)].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def _state(df: pd.DataFrame, prov: dict) -> ProjectState:
    return ProjectState(results=[ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df, prov)])


def _run_in_snapshot(script: str) -> None:
    """Execute ``script`` against the PRE-CHANGE tree, from a neutral cwd."""
    env = dict(os.environ, PYTHONPATH=str(_SNAPSHOT), QT_QPA_PLATFORM="offscreen",
               MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                          cwd="/", env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"pre-change tree failed:\n{proc.stdout}\n{proc.stderr}"


# ================================================================================================
# the project file
# ================================================================================================

def test_a_saved_project_gives_back_source_and_review_with_their_values_and_dtype(tmp_path):
    df, prov = _detection()
    assert ev.has_source_status(df), "the detector stamps the pair; the project must keep it"
    df.loc[df.index[0], ev.SOURCE_COL] = ev.SOURCE_MANUAL
    df.loc[df.index[0], ev.REVIEW_COL] = ev.REVIEW_DONE
    before = _frozen20_hash(df)

    back = load_project(save_project(tmp_path / "p.nanoscope", _state(df, prov))).results[0].events

    assert ev.has_source_status(back)
    assert list(back[ev.SOURCE_COL]) == list(df[ev.SOURCE_COL])
    assert list(back[ev.REVIEW_COL]) == list(df[ev.REVIEW_COL])
    assert back[ev.SOURCE_COL].dtype == "string" and back[ev.REVIEW_COL].dtype == "string"
    assert _frozen20_hash(back) == before


def test_a_review_nobody_recorded_comes_back_as_pd_na_and_not_as_a_string(tmp_path):
    """``pd.NA`` is the third state, and a round-trip that turned it into ``"<NA>"`` or
    ``None`` would make an unrecorded row indistinguishable from a recorded one."""
    df, prov = _detection()
    df.loc[df.index[1], ev.REVIEW_COL] = pd.NA

    back = load_project(save_project(tmp_path / "p.nanoscope", _state(df, prov))).results[0].events

    assert back[ev.REVIEW_COL].isna().tolist() == df[ev.REVIEW_COL].isna().tolist()
    assert back[ev.REVIEW_COL].iloc[1] is pd.NA
    assert ev.review_of(back.iloc[1]) is None
    assert list(ev.event_status(back))[1] == ev.STATUS_UNKNOWN


def test_a_project_saved_without_the_columns_still_loads_and_every_row_reads_unknown(tmp_path):
    """The whole point of declaring the pair outside ``SCHEMA``: the owner does not have
    to migrate a saved project. A frame with neither column must load, must build an
    ``EventSet`` (whose ``__post_init__`` raises only on a missing REQUIRED column), and
    must read UNKNOWN -- never ``unreviewed``, which would claim the rows are fresh
    candidates nobody has looked at."""
    df, prov = _detection()
    df = _legacy(df)
    assert not ev.has_source_status(df)

    back = load_project(save_project(tmp_path / "p.nanoscope", _state(df, prov))).results[0].events

    assert ev.SOURCE_COL not in back.columns and ev.REVIEW_COL not in back.columns
    assert ev.EventSet(back, prov).df is not None
    assert set(ev.event_status(back)) <= {ev.STATUS_UNKNOWN, ev.STATUS_AUTO_REJECTED}
    assert ev.source_of(back.iloc[0]) is None and ev.is_reviewed(back.iloc[0]) is None


def test_a_review_column_stored_as_a_nullable_boolean_comes_back_settled_to_string(
        tmp_path):
    """A project file is written by whichever build the user had, so it is the door a
    ``boolean``-dtyped ``review`` arrives through -- and ``boolean`` survives parquet.
    Load settles it, and that is the ONLY place it gets settled: nothing downstream
    repairs the dtype, so without this it reloads, re-saves and survives a re-measure
    install still ``boolean``.

    This test used to be justified by a crash in ``merge._merged_row``'s bool carry. That
    justification is retired -- the same change routes ``source``/``review`` to
    ``_combined_provenance_cell`` before that branch -- and the re-measured WHY, with the
    merge that now succeeds, is pinned in ``test_src2_restore_event_extras_why.py``."""
    df, prov = _detection()
    df[ev.SOURCE_COL] = pd.array([ev.SOURCE_MANUAL] * len(df), dtype="string")
    df[ev.REVIEW_COL] = pd.array([True] + [pd.NA] * (len(df) - 1), dtype="boolean")

    back = load_project(save_project(tmp_path / "p.nanoscope", _state(df, prov))).results[0].events

    assert back[ev.REVIEW_COL].dtype == "string"
    assert not pd.api.types.is_bool_dtype(back[ev.REVIEW_COL])
    # The exact expression `merge._merged_row` runs on a non-schema column.
    assert bool(back[ev.REVIEW_COL].to_numpy(dtype=object)[0]) is True
    # The value is KEPT, not reinterpreted: "True" is not one of the two tokens, so the
    # reader declines to call it evidence that a person ruled.
    assert ev.review_of(back.iloc[0]) is None


def test_load_neither_invents_the_columns_nor_reorders_the_frozen_twenty(tmp_path):
    """``_restore_event_extras`` settles a dtype and does nothing else."""
    df, prov = _detection()
    before_cols, before_hash = list(df.columns), _frozen20_hash(df)

    back = load_project(save_project(tmp_path / "p.nanoscope", _state(df, prov))).results[0].events

    assert list(back.columns) == before_cols
    assert _frozen20_hash(back) == before_hash


@pytest.mark.skipif(not _SNAPSHOT.is_dir(), reason="pre-change tree snapshot is not present")
def test_a_project_written_by_the_pre_change_build_opens_in_this_one(tmp_path):
    """The compatibility promise, tested against the build that made the promise
    necessary rather than against a frame this build pretended was old."""
    old = tmp_path / "old.nanoscope"
    _run_in_snapshot(f"""
        import numpy as np, pandas as pd
        from pathlib import Path
        from nanoscope.analysis.context import SimpleAnalysisContext
        from nanoscope.analysis.modules import DetectSchmitt
        from nanoscope.core.document import CalibratedChannel
        from nanoscope.core.project import ProjectState, ResultRec, save_project
        from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
        rng = np.random.default_rng(0)
        x = np.full(100000, 100.0) + rng.normal(0.0, 1.5, 100000)
        for s, e in {_EVENTS!r}:
            x[s:e] -= 30.0
        arr = np.ascontiguousarray(x, dtype=np.float32)
        info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
        ch = CalibratedChannel(info=info, data=arr, units="pA", sample_rate_hz={FS!r}, t0_s=0.0)
        meta = RecordingMeta(path=Path("pore_a.csv"), format_name="synthetic",
                             n_samples=arr.size, sample_rate_hz={FS!r},
                             rate_provenance=Provenance.FILE_META, channels=(info,))
        res = DetectSchmitt().run(SimpleAnalysisContext(channel=ch, meta=meta,
                                                        params={{"min_dwell_s": 5e-4}}))
        df = res.payload.df.copy()
        assert "source" not in df.columns and "review" not in df.columns
        save_project({str(old)!r}, ProjectState(
            results=[ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df,
                               dict(res.payload.provenance))]))
    """)

    back = load_project(old).results[0].events

    assert not ev.has_source_status(back)
    assert len(back) == len(_EVENTS)
    assert set(ev.event_status(back)) <= {ev.STATUS_UNKNOWN, ev.STATUS_AUTO_REJECTED}
    # And the pair can be materialised onto it afterwards, still reading unknown until
    # somebody says otherwise.
    ev.ensure_source_status(back)
    assert back[ev.SOURCE_COL].isna().all() and back[ev.REVIEW_COL].isna().all()


@pytest.mark.skipif(not _SNAPSHOT.is_dir(), reason="pre-change tree snapshot is not present")
def test_the_pre_change_build_re_saves_a_new_project_without_stripping_the_columns(tmp_path):
    """The description columns themselves remain backward-compatible inside parquet.

    v10 now correctly prevents a v9 build from opening a project whose new result stamps
    it would strip. To isolate this older column invariant, make a v9-compatible copy by
    removing only those v10 manifest keys; the old build must still read and rewrite the
    whole parquet member unchanged.
    """
    df, prov = _detection()
    df = ev.stamp_source_status(_legacy(df), "detect.keras")
    current = save_project(tmp_path / "current.nanoscope", _state(df, prov))
    new = tmp_path / "new-v9-compatible.nanoscope"
    with zipfile.ZipFile(current) as src, zipfile.ZipFile(new, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["version"] = 9
                for result in manifest["results"]:
                    result.pop("module_version", None)
                    result.pop("baseline_fingerprint", None)
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)
    again = tmp_path / "again.nanoscope"

    _run_in_snapshot(f"""
        from nanoscope.core.project import load_project, save_project
        save_project({str(again)!r}, load_project({str(new)!r}))
    """)

    back = load_project(again).results[0].events
    assert list(back[ev.SOURCE_COL]) == ["detect.keras"] * len(df)
    assert list(back[ev.REVIEW_COL]) == [ev.REVIEW_NONE] * len(df)
    assert _frozen20_hash(back) == _frozen20_hash(df)


# ================================================================================================
# the events export -- CSV body + sidecar
# ================================================================================================

def test_the_exported_csv_carries_the_two_columns_and_still_hashes_the_same_frozen_twenty(
        tmp_path):
    df, prov = _detection()
    plain = scoring.write_events_export(_legacy(df), tmp_path / "plain.csv", provenance=prov)
    plain_body = pd.read_csv(plain["data"])

    stamped = ev.stamp_source_status(_legacy(df), "detect.schmitt")
    out = scoring.write_events_export(stamped, tmp_path / "ev.csv", provenance=prov)
    body = pd.read_csv(out["data"])

    assert list(body[ev.SOURCE_COL]) == ["detect.schmitt"] * len(df)
    assert list(body[ev.REVIEW_COL]) == [ev.REVIEW_NONE] * len(df)
    # The two new columns are APPENDED; the frozen 20 are untouched, in order, and to
    # the last digit -- hashed off the two CSV BODIES, which is the artefact the
    # invariant is about. (Not off the in-memory frame: ``to_csv`` writes pandas' own
    # shortest round-trip repr, so re-reading moves the last ULP of ``baseline_pA``
    # whether or not anything was added. Both files pay that identically.)
    assert list(body.columns)[:len(ev.SCHEMA)] == list(ev.SCHEMA)
    assert _frozen20_hash(body) == _frozen20_hash(plain_body)
    # And the numbers in the file are the numbers in the frame. Only the numeric columns:
    # a CSV has no dtypes, so ``flag``'s empty strings read back as NaN whatever is or is
    # not appended, and asserting on that would be a test of pandas, not of this change.
    numeric = [c for c in ev.SCHEMA if df[c].dtype.kind in "fiub"]
    assert_frame_equal(body[numeric], df[numeric].reset_index(drop=True), check_dtype=False)


def test_the_sidecar_says_how_many_rows_were_withheld_from_the_csv(tmp_path):
    """``Analysis > Export events`` writes ``events.accepted.df``. Before this, the
    sidecar said ``n_rows: 4`` over a detection of 5 and nothing on disk named the row
    that did not come."""
    df, prov = _detection()
    df.loc[df.index[1], "accepted"] = False
    accepted = df[df["accepted"]]

    out = scoring.write_events_export(accepted, tmp_path / "ev.csv", provenance=prov,
                                      n_total=len(df))
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert block["n_in_table"] == len(df)
    assert block["n_withheld"] == 1
    assert block["counted_from"] == "caller"
    assert "accepted=False" in block["note"]


def test_the_sidecar_falls_back_to_the_detection_provenance_when_the_caller_is_silent(
        tmp_path):
    """A caller that has not been wired yet still discloses the drop, because the
    detection's own ``n_events`` is in the provenance the export already carries."""
    df, prov = _detection()
    accepted = df.iloc[:-2]

    out = scoring.write_events_export(accepted, tmp_path / "ev.csv", provenance=prov)
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert prov["n_events"] == len(df)
    assert block["n_withheld"] == 2
    assert block["counted_from"] == "provenance.n_events"


def test_a_stale_provenance_smaller_than_the_table_reports_unknown_not_a_negative_count(
        tmp_path):
    """A negative "withheld" would be a fabricated number in the one key that exists to
    stop a fabricated number -- the defect ``events.reconcile_row_counts`` documents."""
    df, prov = _detection()
    prov["n_events"] = 2

    out = scoring.write_events_export(df, tmp_path / "ev.csv", provenance=prov)
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert block["n_withheld"] is None and block["n_in_table"] is None
    assert block["counted_from"] is None


def test_the_sidecar_breaks_the_written_rows_down_by_source_kind_and_status(tmp_path):
    df, prov = _detection()
    df = ev.stamp_source_status(_legacy(df), "detect.keras")
    df.loc[df.index[0], ev.SOURCE_COL] = ev.SOURCE_MANUAL
    df.loc[df.index[0], ev.REVIEW_COL] = ev.REVIEW_DONE

    out = scoring.write_events_export(df, tmp_path / "ev.csv", provenance=prov)
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert block["columns"] == [ev.SOURCE_COL, ev.REVIEW_COL]
    assert block["by_source"] == {"detect.keras": len(df) - 1, ev.SOURCE_MANUAL: 1}
    assert block["by_kind"] == {ev.KIND_ML: len(df) - 1, ev.KIND_MANUAL: 1}
    assert block["by_status"] == {ev.STATUS_UNREVIEWED: len(df) - 1, ev.STATUS_ACCEPTED: 1}


def test_a_legacy_export_says_the_columns_are_absent_rather_than_letting_silence_speak(
        tmp_path):
    """No ``by_source`` at all, and ``columns: []`` saying why. An empty breakdown would
    read as "no rows came from anywhere"; a missing key with nothing beside it would read
    as "one detector, all reviewed"."""
    df, prov = _detection()

    out = scoring.write_events_export(_legacy(df), tmp_path / "ev.csv", provenance=prov)
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert block["columns"] == []
    assert block["by_source"] is None and block["by_kind"] is None
    assert block["by_status"] == {ev.STATUS_UNKNOWN: len(df)}


def test_an_empty_export_claims_nothing_about_rows_it_does_not_have(tmp_path):
    """Everything rejected is a path the user reaches on purpose. No rows, no claim."""
    df, prov = _detection()

    out = scoring.write_events_export(df.iloc[:0], tmp_path / "ev.csv", provenance=prov)
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]

    assert block["by_source"] is None and block["by_status"] is None
    assert block["n_withheld"] == len(df)


def test_the_export_still_writes_both_files_or_neither(tmp_path):
    """The curation block is built BEFORE the staged set opens, so a frame it cannot
    describe fails with neither file on disk rather than leaving a CSV beside a missing
    sidecar. A bare two-column frame (``test_persist_invariant`` writes exactly this one)
    is the case: no ``accepted``, so there is no status to derive, and it must still
    write both files."""
    out = scoring.write_events_export(pd.DataFrame({"start_idx": [1, 2]}), tmp_path / "e.csv")
    assert out["data"].exists() and out["meta"].exists()
    block = json.loads(out["meta"].read_text())[scoring.CURATION_BLOCK_KEY]
    assert block["by_status"] is None            # no `accepted` column: nothing to derive
