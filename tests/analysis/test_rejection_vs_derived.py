"""One flag set was answering two different questions, and got one of them wrong.

``DERIVED_FLAGS`` answers *"will a rebuild re-derive this token rather than inherit
it?"*. Three readers were using it to answer a second, different question: *"does
this token explain why the APP rejected the row?"*. The two agree for
``dwell>max_dwell_s``, ``clipped_by_boundary`` and ``non_finite``, and disagree for
``zero_baseline`` -- which ``characterize`` re-derives (so it belongs in
``DERIVED_FLAGS``) but has never rejected anything
(``accepted = ~(too_long | clipped | non_finite)``: it is a caveat about
``frac_depth``).

FAILING BEFORE -- a hand rejection that happened to carry ``zero_baseline`` was
un-rejected by every rebuild::

    inherited_curation('zero_baseline', False) -> ('', True)     the rejection, gone
    inherited_curation('',              False) -> ('', False)    control, survives

FAILING BEFORE -- and the count that exists to ANNOUNCE that loss did not see it::

    _hand_curation(flag='zero_baseline') -> n_hand_rejected = 0
    _hand_curation(flag='')              -> n_hand_rejected = 1

``n_hand_rejected`` is the number the "this will destroy N hand rejections" warning
and its undo step are computed from, so the loss was silent at exactly the level that
exists to make it loud.

FAILING BEFORE, in the other direction -- the same line over-counted. Written as
``bool(tokens) and not bool(tokens - DERIVED_FLAGS)``, a row rejected
``clipped_by_boundary;molecule=apo`` read as a hand rejection while
``inherited_curation`` (which excludes labels from the reasons) re-derived it. The two
readers of one rule disagreed in BOTH directions, which is what a second reader of a
rule always eventually does.

REACH. ``|baseline_pA| == 0.0`` exactly is not reachable on the owner's data -- 0
exact zeros in the current channel of the -600 mV apo file, 0 in its estimated
baseline, and no shipped control produces a zero baseline. The label case is reachable
today. Both are fixed by the same split, because the defect is the conflation and not
either symptom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanoscope.analysis.events import (
    DERIVED_FLAGS,
    FLAG_CLIPPED,
    FLAG_MAX_DWELL,
    FLAG_NON_FINITE,
    FLAG_ZERO_BASELINE,
    EventSet,
    inherited_curation,
)
from nanoscope.core.history import _hand_curation

FS = 50_000.0


def _row(flag: str, accepted: bool) -> pd.DataFrame:
    return pd.DataFrame({"flag": [flag], "accepted": [accepted]})


# ===========================================================================
# 1. THE REBUILD RULE
# ===========================================================================
def test_a_hand_rejection_carrying_zero_baseline_survives_a_rebuild():
    """FAILING BEFORE. ``zero_baseline`` explained nothing about the rejection, so the
    rejection was a person's and a rebuild has no standing to reverse it."""
    assert inherited_curation(FLAG_ZERO_BASELINE, False) == ("", False)


def test_the_control_and_the_three_real_reasons_are_unchanged():
    """PASSES BOTH. The half that already worked, pinned so the split cannot be paid
    for out of it: a bare rejection is a person's, and the three tokens
    ``characterize`` actually rejects on are re-judged."""
    assert inherited_curation("", False) == ("", False)
    for token in (FLAG_MAX_DWELL, FLAG_CLIPPED, FLAG_NON_FINITE):
        assert inherited_curation(token, False) == ("", True), token


def test_a_caveat_beside_a_real_reason_does_not_make_the_rejection_a_persons():
    """PASSES BOTH BEFORE AND AFTER -- the ceiling, and the failure mode the naive fix
    walks into. Excluding ``zero_baseline`` from the
    REASONS is right; treating its presence as evidence of a person would leave a row
    flagged ``zero_baseline;clipped_by_boundary`` rejected for ever with its reason
    dropped on the next rebuild -- "rejected with NO REASON", invisible to every
    filter and count, which is the failure this vocabulary was built to end."""
    flag = f"{FLAG_ZERO_BASELINE};{FLAG_CLIPPED}"
    assert inherited_curation(flag, False) == ("", True)


def test_a_label_is_still_not_a_reason():
    """PASSES BOTH. ``molecule=apo`` says what an event is, not why it was rejected."""
    assert inherited_curation(f"{FLAG_CLIPPED};molecule=apo", False) == (
        "molecule=apo", True)


def test_a_curator_token_still_makes_the_rejection_a_persons():
    """PASSES BOTH, and it is the invariant the split must not spend: a reason
    outside the vocabulary is a person's word for why, and it carries."""
    assert inherited_curation("noise", False) == ("noise", False)


def test_zero_baseline_is_still_RE_DERIVED_rather_than_inherited():
    """PASSES BOTH, and it is the half of ``DERIVED_FLAGS`` that is right: the token
    is a statement about THIS measurement, so a rebuild must re-derive it and an
    ACCEPTED row carrying it must not drag it along."""
    assert FLAG_ZERO_BASELINE in DERIVED_FLAGS
    assert inherited_curation(FLAG_ZERO_BASELINE, True) == ("", True)


def test_a_reason_that_cannot_be_re_derived_is_still_inherited_whole():
    """PASSES BOTH. ``max_dwell_s`` unknown means the limit cannot be re-applied, so
    ``recharacterize_span`` narrows the derived set and the rejection has to travel
    with its reason. The split must be read through the caller's set, not a module
    constant, or this case silently loses its rejection."""
    narrowed = DERIVED_FLAGS - {FLAG_MAX_DWELL}
    assert inherited_curation(FLAG_MAX_DWELL, False, derived=narrowed) == (
        FLAG_MAX_DWELL, False)


# ===========================================================================
# 2. THE COUNT THE WARNING IS COMPUTED FROM
# ===========================================================================
def test_the_hand_rejection_count_sees_a_zero_baseline_rejection():
    """FAILING BEFORE. ``n_hand_rejected`` feeds the "this will destroy N hand
    rejections" warning and the undo step beside it."""
    assert _hand_curation(_row(FLAG_ZERO_BASELINE, False))["n_hand_rejected"] == 1
    assert _hand_curation(_row("", False))["n_hand_rejected"] == 1


def test_the_hand_rejection_count_does_not_count_a_labelled_automatic_one():
    """FAILING BEFORE, the over-count. A label beside a derived reason read as a
    person's rejection while the rebuild re-derived it -- so the warning quoted a
    number for curation that costs nothing to lose, which is the fastest way to teach
    someone to click past it."""
    assert _hand_curation(
        _row(f"{FLAG_CLIPPED};molecule=apo", False))["n_hand_rejected"] == 0


def test_the_count_and_the_rebuild_rule_agree_on_every_combination():
    """The invariant, not the two symptoms: ONE question, so exactly one answer. A
    row is counted as a hand rejection iff ``inherited_curation`` would carry its
    rejection through a rebuild."""
    tokens = ["", FLAG_ZERO_BASELINE, FLAG_CLIPPED, FLAG_MAX_DWELL, FLAG_NON_FINITE,
              "noise", "molecule=apo", f"{FLAG_ZERO_BASELINE};{FLAG_CLIPPED}",
              f"{FLAG_CLIPPED};molecule=apo", f"{FLAG_ZERO_BASELINE};noise",
              f"{FLAG_NON_FINITE};{FLAG_ZERO_BASELINE};molecule=apo"]
    for flag in tokens:
        counted = _hand_curation(_row(flag, False))["n_hand_rejected"] == 1
        _, kept = inherited_curation(flag, False)
        carried = not kept
        assert counted is carried, (flag, counted, carried)


def test_an_accepted_row_is_never_a_hand_rejection():
    """PASSES BOTH -- the trivial half, pinned because the shared predicate is now
    asked about rows that are accepted too."""
    for flag in ("", FLAG_ZERO_BASELINE, "noise", FLAG_CLIPPED):
        assert _hand_curation(_row(flag, True))["n_hand_rejected"] == 0


def test_the_note_count_is_a_different_question_and_keeps_its_answer():
    """PASSES BOTH. "Did a person write anything here" is not "does this explain the
    rejection": a label IS a note. Pinned so the split does not quietly take a third
    question with it."""
    assert _hand_curation(_row("molecule=apo", True))["n_notes"] == 1
    assert _hand_curation(_row(FLAG_ZERO_BASELINE, True))["n_notes"] == 0
    assert _hand_curation(_row("noise", True))["n_notes"] == 1


# ===========================================================================
# 3. THE TWO SETS, AND THE PREMISE THEY REST ON
# ===========================================================================
def test_the_rejecting_set_is_exactly_what_characterize_rejects_on():
    """THE PREMISE. ``characterize`` computes
    ``accepted = ~(too_long | clipped | non_finite)`` -- three rules, three tokens,
    and ``zero_baseline`` is not among them. If that line ever gains or loses a rule,
    this test fails and the set has to be revised with it rather than drifting."""
    import inspect

    from nanoscope.analysis import characterize as C
    from nanoscope.analysis.events import AUTOMATIC_REJECTION_FLAGS

    assert AUTOMATIC_REJECTION_FLAGS == {FLAG_MAX_DWELL, FLAG_CLIPPED, FLAG_NON_FINITE}
    assert AUTOMATIC_REJECTION_FLAGS < DERIVED_FLAGS
    assert DERIVED_FLAGS - AUTOMATIC_REJECTION_FLAGS == {FLAG_ZERO_BASELINE}
    src = inspect.getsource(C.characterize)
    assert "accepted = ~(too_long | clipped | non_finite)" in src


def test_the_predicate_is_the_only_definition_of_automatic():
    """The lesson this defect is an instance of: a rule with two readers is a rule
    that will be written two ways. Both readers must reach
    ``is_automatic_rejection``."""
    import inspect

    from nanoscope.analysis import events as E
    from nanoscope.core import history as H

    assert "is_automatic_rejection" in inspect.getsource(E.inherited_curation)
    assert "is_automatic_rejection" in inspect.getsource(H._hand_curation)


# ===========================================================================
# 4. END TO END -- the rebuild engines the rule reaches
# ===========================================================================
def _two_events(i0: float) -> tuple[np.ndarray, EventSet]:
    from nanoscope.analysis.detect import detect_events

    rng = np.random.default_rng(11)
    n = 40_000
    x = np.full(n, i0) + rng.normal(0.0, 8.0, n)
    x[10_000:10_200] -= 300.0
    x[10_220:10_500] -= 300.0
    df, prov = detect_events(x.astype(np.float64), FS, polarity=-1, min_dwell_s=1e-4,
                             baseline=np.full(n, i0), recording="r", pore_id="p")
    return x, EventSet(df, prov)


def test_a_merge_keeps_a_hand_rejection_on_a_zero_baseline_row():
    """FAILING BEFORE, through the engine the user actually reaches. Two events, a
    20-sample gap, both hand-rejected; ``merge_by_gap`` collapses them and the
    merged row must still be rejected. The control run differs only in the open-pore
    level, so the flag token is provably the cause."""
    from nanoscope.analysis.merge import merge_by_gap

    for i0, token in ((0.0, FLAG_ZERO_BASELINE), (-1000.0, "")):
        x, ev = _two_events(i0)
        assert len(ev.df) == 2, i0
        assert set(ev.df["flag"]) == {token}, (i0, set(ev.df["flag"]))
        df = ev.df.copy()
        df["accepted"] = False                       # the curator, on both rows
        merged, _n = merge_by_gap(EventSet(df, dict(ev.provenance)), max_gap_s=1e-3,
                                  signal=x, fs=FS)
        frame = merged.df
        assert len(frame) == 1, (i0, len(frame))
        assert not bool(frame.loc[0, "accepted"]), (
            f"the hand rejection was lost on the i0={i0} run")
