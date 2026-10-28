"""S8-RD1 (analysis layer) — a stored fit names the table it was made on, and ONE function
judges whether it belongs to the table now on screen.

MEASURED DEFECT (sweep 8). The result store holds one ``fit.dwell`` and one ``fit.depth`` per
(recording, channel), keyed with no detector lineage, and the only guard on reading one back —
``MainWindow._stored_fit`` for Export figure: calibration stamp, then accepted count — lived in the
window. The Results dock's Fits tab applied neither, so it printed Schmitt's n = 316 fit under
CUSUM's table, under a merged table (246 accepted) and under a re-run (44 events), while the export
refused the same object as "saved fit is stale".

THE FIX (this half). ``fit_dwell`` / ``fit_depth`` take the ``module_id`` of the table they were
handed and record it in ``FitResult.meta['detection_module_id']`` — an ADDED key, beside
``n_input_events``, so a saved project keeps reading in older builds. ``judge_stored_fit`` is the
one place that says whether a stored fit belongs to a table: the export's own two guards
(``stored_fit_problem``, whose words are unchanged and still what Export figure prints) plus the
lineage. A fit that names no table — saved by an earlier build, or filed by the Plots window — is
``FIT_UNRECORDED``: kept, never silently assumed to match.

These tests call the functions with synthetic tables; the GUI half (what the Fits tab prints, and
the project round trip through the real window) is in
``tests/gui/test_s8_rd1_fits_tab_follows_the_shown_table.py``.
"""

from __future__ import annotations

import numpy as np

from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.analysis.modules import (
    FIT_CALIBRATION_MOVED,
    FIT_CURRENT,
    FIT_DETECTION_KEY,
    FIT_EVENTS_CHANGED,
    FIT_OTHER_TABLE,
    FIT_UNRECORDED,
    fit_depth,
    fit_dwell,
    judge_stored_fit,
    stored_fit_problem,
)
from nanoscope.analysis.stats import fit_from_json, fit_to_json
from nanoscope.core.analysis import AnalysisResult

FS = 50_000.0
T_MIN = 3e-5


def _table(module_id: str = "detect.schmitt", *, accepted: int = 40, rejected: int = 0,
           fingerprint: tuple[float, float] = (1.0, 0.0)) -> AnalysisResult:
    """A stored detection: ``accepted`` curated-in rows and ``rejected`` curated-out ones."""
    rng = np.random.default_rng(0)
    n = accepted + rejected
    df = empty_df()
    df["dwell_s"] = T_MIN + rng.exponential(4e-4, n)
    df["frac_depth"] = np.abs(rng.normal(0.3, 0.03, n))
    df["accepted"] = np.r_[np.ones(accepted, dtype=bool), np.zeros(rejected, dtype=bool)]
    es = EventSet(df, {"min_dwell_s": T_MIN, "fs": FS, "analyzed_duration_s": 1.0})
    return AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id="doc", channel=0, params={},
        calibration_fingerprint=fingerprint, payload=es)


def _dwell_fit(table: AnalysisResult, *, names_table: bool = True) -> AnalysisResult:
    """The dwell fit of ``table`` as the Fit button makes it (``names_table=False``: as an
    earlier build, or the Plots window, would have filed it)."""
    return fit_dwell(
        table, doc_id="doc", channel=0, n_boot=20,
        calibration_fingerprint=table.calibration_fingerprint,
        detection_module_id=table.module_id if names_table else None)


# ------------------------------------------------------------------------------------------------
# the stamp
# ------------------------------------------------------------------------------------------------

def test_both_fits_record_the_table_they_were_handed_beside_the_count_they_already_carried():
    table = _table("detect.cusum", accepted=40)
    dwell = fit_dwell(table, doc_id="doc", channel=0, n_boot=20,
                      detection_module_id=table.module_id)
    depth = fit_depth(table, doc_id="doc", channel=0, n_boot=20,
                      detection_module_id=table.module_id)

    for fit in (dwell, depth):
        assert fit.payload.meta[FIT_DETECTION_KEY] == "detect.cusum"
        assert fit.payload.meta["n_input_events"] == 40      # the key the export reads
    assert FIT_DETECTION_KEY == "detection_module_id"        # the spelling is the file format


def test_a_fit_made_without_naming_a_table_records_none():
    """Not an empty string, not a guess: absence is what an older build wrote, and the verdict
    below reads it as 'unrecorded'."""
    table = _table()
    fit = fit_dwell(table, doc_id="doc", channel=0, n_boot=20)
    assert FIT_DETECTION_KEY not in fit.payload.meta


def test_the_stamp_survives_the_project_json_and_an_old_blob_reads_as_unrecorded():
    table = _table("detect.reconciled")
    stored = _dwell_fit(table)
    blob = fit_to_json(stored.payload)
    assert blob["meta"][FIT_DETECTION_KEY] == "detect.reconciled"
    assert blob["meta"]["n_input_events"] == 40

    restored = AnalysisResult(
        module_id="fit.dwell", module_version="", doc_id="doc", channel=0, params={},
        calibration_fingerprint=table.calibration_fingerprint,
        payload=fit_from_json(blob))
    assert judge_stored_fit(restored, table).reason == FIT_CURRENT

    # The same blob as an earlier build wrote it: every old key, no lineage.
    old = {**blob, "meta": {k: v for k, v in blob["meta"].items() if k != FIT_DETECTION_KEY}}
    legacy = AnalysisResult(
        module_id="fit.dwell", module_version="", doc_id="doc", channel=0, params={},
        calibration_fingerprint=table.calibration_fingerprint,
        payload=fit_from_json(old))
    assert judge_stored_fit(legacy, table).reason == FIT_UNRECORDED


# ------------------------------------------------------------------------------------------------
# the verdict
# ------------------------------------------------------------------------------------------------

def test_a_fit_belongs_to_the_table_it_was_made_on_while_nothing_has_changed():
    table = _table("detect.schmitt", accepted=40, rejected=3)
    verdict = judge_stored_fit(_dwell_fit(table), table)

    assert verdict.reason == FIT_CURRENT and verdict.current
    assert verdict.made_on == "detect.schmitt"
    assert (verdict.n_fit, verdict.n_now) == (40, 40)        # rejected rows are not counted


def test_a_fit_made_on_another_table_does_not_belong_whatever_the_counts():
    cusum = _table("detect.cusum", accepted=40)
    schmitt = _table("detect.schmitt", accepted=40)          # the SAME count: only lineage can tell
    verdict = judge_stored_fit(_dwell_fit(cusum), schmitt)

    assert verdict.reason == FIT_OTHER_TABLE and not verdict.current
    assert verdict.made_on == "detect.cusum"

    # ...and it is the first thing said, ahead of a count that differs as well.
    other = _table("detect.schmitt", accepted=50)
    assert judge_stored_fit(_dwell_fit(cusum), other).reason == FIT_OTHER_TABLE


def test_the_accepted_count_moving_marks_the_fit_with_both_numbers():
    fitted_on = _table("detect.schmitt", accepted=40)
    fit = _dwell_fit(fitted_on)
    curated = _table("detect.schmitt", accepted=39, rejected=1)
    verdict = judge_stored_fit(fit, curated)

    assert verdict.reason == FIT_EVENTS_CHANGED
    assert (verdict.n_fit, verdict.n_now) == (40, 39)
    assert verdict.made_on == "detect.schmitt"


def test_a_calibration_change_is_judged_before_the_count():
    fitted_on = _table("detect.schmitt", accepted=40, fingerprint=(1.0, 0.0))
    fit = _dwell_fit(fitted_on)
    regained = _table("detect.schmitt", accepted=39, rejected=1, fingerprint=(2.0, 0.0))

    assert judge_stored_fit(fit, regained).reason == FIT_CALIBRATION_MOVED


def test_a_fit_that_names_no_table_is_never_assumed_to_match():
    """The conservative rule for what an older build (or the Plots window) filed: if the
    calibration or the count already disagree, say THAT (it is the more specific truth); when
    everything else agrees, the one thing left unknown — which table — is still not a yes."""
    table = _table("detect.schmitt", accepted=40)
    unnamed = _dwell_fit(table, names_table=False)

    verdict = judge_stored_fit(unnamed, table)
    assert verdict.reason == FIT_UNRECORDED and not verdict.current
    assert verdict.made_on == ""

    assert judge_stored_fit(
        unnamed, _table("detect.schmitt", accepted=39, rejected=1)).reason == FIT_EVENTS_CHANGED
    assert judge_stored_fit(
        unnamed, _table("detect.schmitt", accepted=40, fingerprint=(2.0, 0.0))
    ).reason == FIT_CALIBRATION_MOVED


# ------------------------------------------------------------------------------------------------
# the export's rule is the same function, in the same words
# ------------------------------------------------------------------------------------------------

def test_the_export_keeps_its_two_guards_and_its_words():
    """``stored_fit_problem`` IS what ``MainWindow._stored_fit`` says. Its sentences are the
    figure export's status-line wording (pinned by tests/gui/test_fit_provenance.py) and are
    not touched by the lineage: a fit with no record of its table still serves the export
    exactly as before."""
    table = _table("detect.schmitt", accepted=40)
    fit = _dwell_fit(table)

    assert stored_fit_problem(fit, table) == ""
    assert stored_fit_problem(
        fit, _table("detect.schmitt", accepted=39, rejected=1)
    ) == "saved fit is stale (the accepted events changed since it ran)"
    assert stored_fit_problem(
        fit, _table("detect.schmitt", accepted=40, fingerprint=(2.0, 0.0))
    ) == "saved fit predates a calibration change"
    # Calibration still comes first, as it always did.
    assert stored_fit_problem(
        fit, _table("detect.schmitt", accepted=39, rejected=1, fingerprint=(2.0, 0.0))
    ) == "saved fit predates a calibration change"
    # No record of a table is not a problem for the export.
    assert stored_fit_problem(_dwell_fit(table, names_table=False), table) == ""


def test_a_fit_that_carries_no_count_cannot_disagree_on_it():
    """The export's rule, preserved verbatim: an absent ``n_input_events`` is not a mismatch."""
    table = _table("detect.schmitt", accepted=40)
    fit = _dwell_fit(table)
    fit.payload.meta.pop("n_input_events")

    assert stored_fit_problem(fit, table) == ""
    assert judge_stored_fit(fit, table).reason == FIT_CURRENT
