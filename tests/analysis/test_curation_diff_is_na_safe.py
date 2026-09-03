"""THE LATE-CURATION DIFF MUST ANSWER FOR EVERY VALUE ``flag`` CAN HOLD.

``flag`` is declared ``"string"`` in :data:`nanoscope.analysis.events.SCHEMA`. The missing
value of a pandas ``string`` column is ``pd.NA``, and ``pd.NA.__bool__`` **raises**. The
install seam introduced with C-4/M-1 decides "did the curator change this row" with a bare
``x != y`` inside a list comprehension, so one missing flag anywhere in the table did not
give a wrong answer — it raised ``TypeError: boolean value of NA is ambiguous`` out of a
Qt slot. Nothing was installed, and because the raise skipped the status update the user
was then told "the detection it was measuring is no longer on screen. Nothing changed",
which is false: the detection was on screen and the measurement was simply thrown away.
The install seam catches ``ValueError``, not ``TypeError``.

REACHABLE, and measured on three paths:
  * ``pd.concat([events.empty_df(), partial_row])`` — the app's own frame builder —
    leaves ``flag`` as ``<NA>``;
  * ``pd.read_csv(..., dtype={"flag": "string"})`` of an exported events CSV with an
    empty flag cell gives ``<NA>``;
  * a ``pd.NA`` flag survives a project save + reopen with its dtype intact.

Every OTHER reader of the column already tolerates it — ``flag_tokens(pd.NA) -> ()``,
``is_automatic_rejection(pd.NA) -> False``, ``inherited_curation([pd.NA], [False]) ->
('', False)``, ``_hand_curation`` counts it — which is exactly why the fix belongs in the
comparison rather than in a caller being asked to clean the frame first.

The pre-fix rule did not raise on this input (it went through ``str(...)``), so the raise
is new to that batch.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA, empty_df, flag_tokens, inherited_curation
from nanoscope.analysis.remeasure import CURATION_COLUMNS, merge_late_curation


def _frame(flags) -> pd.DataFrame:
    rows = [{"event_id": i, "start_idx": 1000 * (i + 1), "end_idx": 1000 * (i + 1) + 200,
             "accepted": True, "flag": f} for i, f in enumerate(flags)]
    df = pd.concat([empty_df(), pd.DataFrame(rows)], ignore_index=True)
    for col, spec in SCHEMA.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(spec[0])
            except (TypeError, ValueError):
                pass
    return df


# ================================================================================================
# the value is reachable
# ================================================================================================

def test_the_declared_dtype_of_flag_admits_a_missing_value():
    assert SCHEMA["flag"][0] == "string"
    df = _frame(["", pd.NA, "manual"])
    assert str(df["flag"].dtype) == "string"
    assert df["flag"].isna().iloc[1]


def test_a_partial_row_concatenated_onto_the_empty_frame_produces_it():
    """The app's own builder shape, not a contrivance."""
    df = pd.concat(
        [empty_df(), pd.DataFrame([{"event_id": 0, "start_idx": 10, "end_idx": 20,
                                    "accepted": True}])],
        ignore_index=True)
    assert df["flag"].isna().iloc[0]


def test_an_exported_csv_with_an_empty_flag_cell_reads_back_as_missing():
    csv = "event_id,start_idx,end_idx,accepted,flag\n0,10,20,True,\n1,30,40,True,manual\n"
    back = pd.read_csv(io.StringIO(csv), dtype={"flag": "string"})
    assert back["flag"].isna().iloc[0]


# ================================================================================================
# THE REPRO
# ================================================================================================

def test_a_missing_flag_anywhere_does_not_blow_up_the_install():
    """Before the fix this raised ``TypeError: boolean value of NA is ambiguous`` from the
    diff loop, and the whole re-measure was discarded with a false explanation.

    Row 1's flag is missing; row 0 is the one the curator touched.
    """
    before = _frame(["", pd.NA, ""])
    out = before.copy()
    live = before.copy()
    live.loc[0, "accepted"] = False

    merged = merge_late_curation(out, live, before)
    assert list(merged["accepted"]) == [False, True, True]


def test_the_missing_flag_belongs_to_a_row_nobody_curated_and_stays_missing():
    """A row the curator never touched must not be rewritten — a diff that normalises
    ``<NA>`` to ``''`` on an untouched row is editing the table behind him."""
    before = _frame(["", pd.NA, ""])
    out, live = before.copy(), before.copy()
    live.loc[0, "accepted"] = False
    merged = merge_late_curation(out, live, before)
    assert pd.isna(merged["flag"].iloc[1])


def test_a_missing_flag_on_the_curated_row_itself_is_handled(recwarn):
    """The harder half: the row that CHANGED is the one carrying the missing flag."""
    before = _frame(["", pd.NA, ""])
    out, live = before.copy(), before.copy()
    live.loc[1, "accepted"] = False
    merged = merge_late_curation(out, live, before)
    assert list(merged["accepted"]) == [True, False, True]


def test_the_numpy_spelling_of_missing_is_handled_too():
    """``np.nan`` reaches the same column through an object-dtype frame."""
    before = _frame(["", np.nan, ""])
    out, live = before.copy(), before.copy()
    live.loc[2, "flag"] = "manual"
    merged = merge_late_curation(out, live, before)
    assert "manual" in flag_tokens(merged["flag"].iloc[2])


def test_an_untouched_table_with_a_missing_flag_still_installs_unchanged():
    """The identity fast path — `out` itself — must survive the value too."""
    before = _frame(["", pd.NA, "manual"])
    out = before.copy()
    assert merge_late_curation(out, before.copy(), before) is out


# ================================================================================================
# the predicate, stated
# ================================================================================================

@pytest.mark.parametrize("a,b,changed", [
    (pd.NA, pd.NA, False),      # the same absence is not a curation
    (np.nan, pd.NA, False),     # ...in either spelling
    (pd.NA, "", True),          # somebody wrote a flag
    ("", pd.NA, True),          # ...or cleared one
    ("manual", "manual", False),
    ("manual", "noise", True),
    (True, True, False),
    (True, False, True),
])
def test_the_curation_predicate_answers_for_every_value_the_dtype_admits(a, b, changed):
    from nanoscope.analysis.remeasure import _curated_differently
    assert _curated_differently(a, b) is changed


def test_every_curation_column_goes_through_the_one_predicate():
    """``accepted`` and ``flag`` are compared by the same rule; a per-column special case
    is how the next dtype gets forgotten."""
    assert CURATION_COLUMNS == ("accepted", "flag")


# ================================================================================================
# controls — the other readers were already right, and must stay right
# ================================================================================================

def test_the_other_readers_of_flag_still_tolerate_a_missing_value():
    """PASSES BOTH — a control, not a repro. It is the reason the fix belongs in the
    comparison: everything else in the package already answers for ``pd.NA``."""
    from nanoscope.analysis.remeasure import is_automatic_rejection

    assert flag_tokens(pd.NA) == ()
    assert is_automatic_rejection(pd.NA) is False
    assert inherited_curation([pd.NA], [False]) == ("", False)
