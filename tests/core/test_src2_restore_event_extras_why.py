"""What ``project._restore_event_extras`` actually protects against IN THIS TREE.

The guard's stated WHY was a measured crash: a ``boolean``-dtyped ``review`` survives
parquet, ``merge._merged_row``'s generic carry does ``series.to_numpy(dtype=bool).any()``
on it, and that raises on ``pd.NA`` -- so reopen an old project, press "Merge all", and
the failure comes out of a Qt slot. That was true of the tree BEFORE this feature. It is
false of this one, because the same change routes ``source``/``review`` to
``merge._combined_provenance_cell`` on the line above that branch.

This file re-measures the question and pins the answer, so neither half can rot:

* the merge the old WHY cited SUCCEEDS here (and the merged row's ``review`` comes from
  ``combine_reviews``, which is the routing, observed rather than asserted about);
* the docstring may not go on citing a failure this tree cannot produce;
* what the guard DOES buy is measured too -- without it the wrong dtype is PERMANENT
  (it reloads ``boolean``, re-saves ``boolean``, survives a re-measure install as
  ``boolean``, and turns ``object`` the moment it meets a stamped frame);
* and it moves no value and no number, which is why it is a normaliser and not a repair.

The guard is neutralised with ``monkeypatch`` -- never by editing the source -- so every
"without it" claim is measured against the shipping file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import events as ev
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.analysis.remeasure import merge_late_curation, remeasure_events
from nanoscope.core import project as proj
from nanoscope.core.document import CalibratedChannel
from nanoscope.core.project import ProjectState, ResultRec, load_project, save_project
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


def _trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture(scope="module")
def detection() -> tuple[np.ndarray, pd.DataFrame, dict]:
    """A real Schmitt detection, so the frame is the genuine 20-column article."""
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=arr, units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(
        path=Path("pore_a.csv"), format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    res = DetectSchmitt().run(
        SimpleAnalysisContext(channel=ch, meta=meta, params={"min_dwell_s": 5e-4}))
    return arr, res.payload.df.copy(), dict(res.payload.provenance)


def _legacy_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` as a build that typed ``review`` the obvious (and wrong) way wrote it."""
    out = df.copy()
    out[ev.SOURCE_COL] = pd.array([ev.SOURCE_MANUAL] * len(out), dtype="string")
    out[ev.REVIEW_COL] = pd.array([True] + [pd.NA] * (len(out) - 1), dtype="boolean")
    return out


def _save(path: Path, df: pd.DataFrame, prov: dict) -> Path:
    return save_project(path, ProjectState(
        results=[ResultRec("/data/pore_a.csv", 0, "detect.schmitt", df, prov)]))


def _load_unguarded(path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Open ``path`` with ``_restore_event_extras`` neutralised -- the source is NOT
    edited, so what this measures is the shipping file minus exactly one function."""
    monkeypatch.setattr(proj, "_restore_event_extras", lambda df: df)
    return load_project(path).results[0].events


def _frozen20_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        df[list(ev.SCHEMA)].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


# ================================================================================================
# the WHY that was measured on the PRE-change tree
# ================================================================================================

def test_the_merge_the_old_why_cited_completes_on_a_boolean_review(
        tmp_path, monkeypatch, detection):
    """The crash the docstring used to cite is unreachable in this tree.

    ``merge._merged_row`` reaches ``is_bool_dtype`` only for a column it has not already
    routed, and this same feature routes ``source``/``review`` to
    ``_combined_provenance_cell`` first. Measured with the guard OFF, i.e. on the exact
    frame the old WHY said would fail: the merge succeeds.
    """
    arr, df, prov = detection
    p = _save(tmp_path / "p.nanoscope", _legacy_boolean(df), prov)
    raw = _load_unguarded(p, monkeypatch)
    assert pd.api.types.is_bool_dtype(raw[ev.REVIEW_COL]), "parquet kept the wrong dtype"

    b = pd.concat([raw, raw.assign(start_idx=raw["start_idx"] + 320,
                                   end_idx=raw["end_idx"] + 320,
                                   event_id=raw["event_id"] + 1000)], ignore_index=True)
    b = b.sort_values("start_idx").reset_index(drop=True)

    merged, n_merged, *_ = merge_by_gap(ev.EventSet(b, dict(prov)), arr, FS, max_gap_s=0.02)

    assert int(n_merged) > 0, "the probe has to actually merge something to prove anything"
    # The routing, observed: a merged row's review is `combine_reviews`' answer (nobody
    # has reviewed the whole span, so UNKNOWN) -- never the boolean OR of the fragments.
    assert all(ev.review_of(r) is None for _, r in merged.df.iterrows())


def test_the_docstring_does_not_cite_a_failure_this_tree_cannot_produce(detection):
    """A WHY is evidence, and evidence that has stopped being true is worse than none.

    The old text taught the next reader that removing the guard reopens a ValueError out
    of a Qt slot. It does not -- ``merge`` routes around that branch now. So the
    docstring may keep the history, but only WITH the routing that retired it; and it
    must say what the guard buys TODAY, which is that the dtype the module declares is
    the one the live frame has.
    """
    doc = proj._restore_event_extras.__doc__ or ""

    if "bool'-dtype NumPy array with missing values" in doc or "is_bool_dtype" in doc:
        assert "_combined_provenance_cell" in doc, (
            "the docstring cites merge's bool carry as the reason for this guard but "
            "does not record that merge routes source/review away from that branch "
            "before it -- the cited failure cannot happen in this tree")
    # And it states a live consequence, not only a retired one.
    assert any(w in doc for w in ("re-measure", "re-saved", "permanent", "sticky")), (
        "the docstring must say what the guard protects against in THIS tree")


# ================================================================================================
# the WHY that is measurable TODAY
# ================================================================================================

def test_without_the_guard_the_wrong_dtype_is_permanent(tmp_path, monkeypatch, detection):
    """Nothing downstream repairs it, so load is the only place it can be settled.

    Measured, guard off: the column reloads ``boolean``, is written back to the project
    file as ``boolean``, comes out of a re-measure install still ``boolean``, and goes
    ``object`` -- neither declared dtype -- as soon as it is concatenated with a frame a
    detector stamped. Guard on, all four are ``string``.
    """
    arr, df, prov = detection
    p = _save(tmp_path / "p.nanoscope", _legacy_boolean(df), prov)

    guarded = load_project(p).results[0].events
    assert guarded[ev.REVIEW_COL].dtype == "string"

    raw = _load_unguarded(p, monkeypatch)
    assert raw[ev.REVIEW_COL].dtype == "boolean"

    # (1) it goes straight back into the file
    again = _load_unguarded(_save(tmp_path / "q.nanoscope", raw.copy(), prov), monkeypatch)
    assert again[ev.REVIEW_COL].dtype == "boolean"

    # (2) a re-measure installs it unchanged
    rm = remeasure_events(ev.EventSet(raw.copy(), dict(prov)), arr, FS)
    live = raw.copy()
    live.loc[live.index[2], "flag"] = "noise"          # one click during the job
    installed = merge_late_curation(rm.events.df, live, raw)
    assert installed[ev.REVIEW_COL].dtype == "boolean"

    # (3) and it poisons the dtype of anything it is concatenated with
    stamped = ev.ensure_source_status(df.copy(), source="detect.schmitt")
    assert stamped[ev.REVIEW_COL].dtype == "string"
    mixed = pd.concat([stamped, raw], ignore_index=True)
    assert mixed[ev.REVIEW_COL].dtype == "object"
    assert pd.concat([stamped, guarded], ignore_index=True)[ev.REVIEW_COL].dtype == "string"


def test_the_guard_repairs_no_value_and_moves_no_number(tmp_path, monkeypatch, detection):
    """It is a NORMALISER, not a repair -- which is why it cannot move a measurement.

    Every reader goes through ``astype('string')`` of its own, so the answers are
    identical with the guard on and off: ``True`` is not one of the two tokens either
    way, and the row reads unknown rather than being promoted to "a person ruled".
    """
    _, df, prov = detection
    p = _save(tmp_path / "p.nanoscope", _legacy_boolean(df), prov)
    guarded = load_project(p).results[0].events
    raw = _load_unguarded(p, monkeypatch)

    assert list(ev.event_status(guarded)) == list(ev.event_status(raw))
    assert [ev.review_of(r) for _, r in guarded.iterrows()] == \
           [ev.review_of(r) for _, r in raw.iterrows()]
    assert ev.review_of(guarded.iloc[0]) is None, "'True' is not evidence a person ruled"
    assert list(guarded[ev.REVIEW_COL].astype("string")) == \
           list(raw[ev.REVIEW_COL].astype("string"))
    assert _frozen20_hash(guarded) == _frozen20_hash(raw) == _frozen20_hash(df)
