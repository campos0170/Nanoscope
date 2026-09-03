"""S7 fix-up E3 / X7 -- the subpopulation report's summary line and its CURATION caveat
use ONE word for the count they share.

After the S7-DET4 fix ``provenance['n_events']`` follows the table (re-read on every
reshape), so after Detect (317) -> Merge all at 10 ms (247) the CURATION caveat honestly
reads "1 of the 247 region(s) left in the table after a merge (the last merge folded 317
rows into 247)". The summary line two rows above it -- the line that points AT that
caveat -- still hard-coded its noun and printed
``curated out (not below) : 1  (of 247 detected region(s); see the CURATION caveat)``.
Measured on the owner's -600 mV file through the real Merge all button and the real
Export: 317 detected, 247 merged; after one hand-added event
``(of 248 detected region(s); ...)`` beside a caveat saying "merged and edited since
detection". One file, two words for the same number, and the summary block is the part a
reader copies.

The fix is at the invariant: :func:`rows_in_hand` decides ONCE what ``n_detected`` counts
(:func:`_table_shape`) and stamps the word as ``meta["table_noun"]``; the caveat and the
summary line both phrase from that decision, so they cannot disagree. The caveat's own
text is unchanged, and a table nothing has reshaped is still called "detected".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet

FS = 50_000.0
N_ROWS = 160
N_BEFORE_MERGE = N_ROWS + 70          # the detector's table, before the merge


def _synth(seed: int = 5, n: int = 260_000):
    """A trace with two obvious event classes and the table that describes them."""
    rng = np.random.default_rng(seed)
    baseline, sigma = 200.0, 2.0
    sig = (baseline + rng.normal(0.0, sigma, n)).astype(np.float32)
    rows, pos, i = [], 500, 0
    while pos + 400 < n and i < N_ROWS:
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        depth = 60.0 if deep else 30.0
        sig[pos:pos + dwell] -= depth
        rows.append({
            "event_id": i, "recording": "synth", "pore_id": "p1",
            "start_idx": pos, "end_idx": pos + dwell, "t_start_s": pos / FS,
            "dwell_s": dwell / FS, "baseline_pA": baseline, "depth_pA": depth,
            "frac_depth": depth / baseline, "ecd_pA_s": depth * dwell / FS,
            "max_dev_pA": depth, "mean_dev_pA": depth, "rise_s": 1 / FS,
            "fall_s": 1 / FS, "snr": depth / sigma, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
        pos += int(rng.integers(500, 1100))
        i += 1
    return sig, pd.DataFrame(rows)[REQUIRED]


CHEAP = SR.SubpopConfig(k_grid=(1, 2), n_splits=2, n_boot=0, with_tsne=False,
                        with_umap=False)


def _curated_rows(provenance: dict) -> tuple[np.ndarray, EventSet]:
    """The rows the app hands in: ``events.accepted`` with exactly ONE row curated out."""
    sig, df = _synth()
    df.loc[df.index[0], "accepted"] = False
    return sig, EventSet(df, provenance).accepted


def _run(sig, events) -> SR.SubpopReport:
    return SR.run_subpopulations(sig, sig.astype(np.float64), FS, events,
                                 recording="synth", duration_s=len(sig) / FS,
                                 config=CHEAP)


def _summary_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("curated out (not below)")]
    assert len(lines) == 1, lines
    return lines[0]


def _curation_caveat(text: str) -> str:
    lines = [ln for ln in text.splitlines() if "CURATION:" in ln]
    assert len(lines) == 1, lines
    return lines[0]


# ================================================================================================
# rows_in_hand -- the one resolver stamps the word the count is printed with
# ================================================================================================

def test_rows_in_hand_names_the_merged_table_as_the_denominator():
    """317 -> 247 merge, one row curated out: the denominator is the MERGED table and
    the word carried in meta says so -- it is not the detector's count."""
    _sig, es = _curated_rows({"n_events": N_ROWS,
                              "merged": {"n_before": N_BEFORE_MERGE, "n_after": N_ROWS}})
    meta, note = SR.rows_in_hand(es, N_ROWS - 1)
    assert meta["n_detected"] == N_ROWS
    assert meta["table_noun"] == SR.TABLE_NOUNS["merged"]
    assert meta["table_noun"] == "region(s) in the merged table"
    assert f"1 of the {N_ROWS} region(s) left in the table after a merge" in note, (
        "the caveat's wave-1 wording must not move")


def test_rows_in_hand_names_a_table_edited_since_the_last_merge():
    """A merge is on record but the table has changed shape since (a hand-added event):
    the count is neither the detector's nor the merge's, and the word says so."""
    _sig, es = _curated_rows({"n_events": N_ROWS,
                              "merged": {"n_before": N_BEFORE_MERGE, "n_after": N_ROWS - 1}})
    meta, note = SR.rows_in_hand(es, N_ROWS - 1)
    assert meta["table_noun"] == SR.TABLE_NOUNS["edited"]
    assert "merged and edited since detection" in note


def test_rows_in_hand_keeps_the_detectors_noun_when_nothing_reshaped_the_table():
    """No report that was already accurate changes: with no merge on record the count
    is the detector's and is still called that, in the caveat and in meta alike."""
    _sig, es = _curated_rows({"n_events": N_ROWS})
    meta, note = SR.rows_in_hand(es, N_ROWS - 1)
    assert meta["table_noun"] == SR.TABLE_NOUNS["detected"] == "detected region(s)"
    assert f"1 of the detector's {N_ROWS} region(s)" in note


def test_a_refused_stale_stamp_still_never_calls_a_merged_table_the_detection():
    """When the count is refused (smaller than the rows in hand) nothing prints it, but
    the word carried beside it must not claim "detected" for a table a merge reshaped."""
    _sig, es = _curated_rows({"n_events": 3,
                              "merged": {"n_before": N_BEFORE_MERGE, "n_after": 3}})
    meta, _note = SR.rows_in_hand(es, N_ROWS - 1)
    assert meta["n_detected"] is None
    assert meta["table_noun"] != SR.TABLE_NOUNS["detected"]


# ================================================================================================
# the written report -- both lines, one word
# ================================================================================================

@pytest.fixture(scope="module")
def merged_report() -> SR.SubpopReport:
    sig, es = _curated_rows({"n_events": N_ROWS,
                             "merged": {"n_before": N_BEFORE_MERGE, "n_after": N_ROWS}})
    return _run(sig, es)


@pytest.fixture(scope="module")
def edited_report() -> SR.SubpopReport:
    sig, es = _curated_rows({"n_events": N_ROWS,
                             "merged": {"n_before": N_BEFORE_MERGE, "n_after": N_ROWS - 1}})
    return _run(sig, es)


def test_the_summary_line_uses_the_caveats_word_after_a_merge(merged_report):
    text = merged_report.text()
    line = _summary_line(text)
    assert line == (
        f"curated out (not below) : 1  (of {N_ROWS} region(s) in the merged table; "
        f"see the CURATION caveat)"
    ), line
    assert "left in the table after a merge" in _curation_caveat(text)
    assert "detected region(s)" not in text, (
        "the detector produced N_BEFORE_MERGE rows; nothing in this file may call the "
        "merged table's size 'detected'")
    assert merged_report.table_noun == SR.TABLE_NOUNS["merged"]


def test_the_summary_line_uses_the_caveats_word_after_a_merge_and_an_edit(edited_report):
    text = edited_report.text()
    line = _summary_line(text)
    assert line == (
        f"curated out (not below) : 1  (of {N_ROWS} region(s) in the table, merged and "
        f"edited since detection; see the CURATION caveat)"
    ), line
    assert "merged and edited since detection" in _curation_caveat(text)
    assert "detected region(s)" not in text


def test_the_summary_line_and_the_caveat_quote_the_same_count_with_one_word(merged_report):
    """The invariant itself: whatever number the summary line prints as the denominator,
    the caveat it points at describes with the same words -- neither says "detected"
    for a merged table, and both carry the same count."""
    text = merged_report.text()
    m = re.search(r"\(of (\d+) (.+?); see the CURATION caveat\)", _summary_line(text))
    assert m, _summary_line(text)
    n, noun = int(m.group(1)), m.group(2)
    assert n == N_ROWS
    assert noun == merged_report.table_noun
    assert f"of the {n} region(s) left in the table after a merge" in _curation_caveat(text)


def test_an_unmerged_curated_report_is_worded_exactly_as_before():
    """Byte-identity guard for the case the wave-1 tests pin: no merge on record, the
    summary line and caveat keep their pre-existing wording."""
    sig, es = _curated_rows({"n_events": N_ROWS})
    rep = _run(sig, es)
    text = rep.text()
    assert _summary_line(text) == (
        f"curated out (not below) : 1  (of {N_ROWS} detected region(s); see the "
        f"CURATION caveat)"
    )
    assert f"1 of the detector's {N_ROWS} region(s)" in _curation_caveat(text)
