"""C1 / C21 / C10 -- the flag/accept invariant, pinned at the invariant.

These three "separate" defects are one defect: ``flag`` was a SLOT that every writer
overwrote and ``accepted`` was recombined by whichever caller happened to rebuild the
row, so a guarantee applied at one call site never reached the others.

  * **C1** ``merge_by_gap`` rebuilt each row through ``characterize`` (flag always '')
    and OR-ed ``accepted``: a rejected event came back accepted with its reason erased.
    Measured on the owner's -600 mV recording under a declared ``max_dwell_s`` of 2 ms,
    a 5 ms gap merge produced **240 accepted rows over the limit, the worst 24.88 ms
    (12.4x)** -- flowing into the table, both histograms, the censored dwell fit (whose
    ``t_max`` IS ``max_dwell_s``) and every export.
  * **C21** an Event Inspector mask resize rewrote ``flag`` to '' while keeping
    ``accepted=False``: a rejected row with NO REASON, invisible to the
    ``clipped_by_boundary`` filter and to the count meant to find exactly those rows.
  * **C10** the boundary-clip rule lived in ``detect_events``, so ``MLDetect`` -- which
    calls ``characterize`` directly -- admitted events cut off by the ZOOM RECTANGLE as
    fully-measured, accepted rows.

The fix is one guarantee in two places every path must pass through:
``events.combine_flags`` / ``events.inherited_curation`` (the vocabulary) and
``characterize`` (the automatic rules). Nothing here reaches into a caller.

The GUI half of the same invariant is in ``tests/gui/test_flag_invariant_gui.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.characterize import characterize, recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    DERIVED_FLAGS,
    FLAG_CLIPPED,
    FLAG_MAX_DWELL,
    FLAG_NON_FINITE,
    FLAG_ZERO_BASELINE,
    EventSet,
    combine_flags,
    flag_tokens,
    has_flag,
    inherited_curation,
)
from nanoscope.analysis.merge import merge_by_gap

FS = 50_000.0
I0 = -1000.0
DEPTH = 300.0
NOISE = 8.0


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _trace(spans, *, n=50_000, depth=DEPTH, polarity=-1, i0=I0, noise=NOISE, seed=0):
    rng = np.random.default_rng(seed)
    x = np.full(n, i0, dtype=np.float64) + rng.normal(0.0, noise, n)
    for s, e in spans:
        x[s:e] += polarity * depth
    return x.astype(np.float32)


def _detect(spans, **kw):
    signal = _trace(spans, n=kw.pop("n", 50_000), seed=kw.pop("seed", 0))
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4, **kw)
    return EventSet(df, prov), signal


#: Three fragments of one shattered translocation, 20 samples apart. The full span is
#: 1000 samples = 20 ms, so a 2 ms max_dwell_s limit is exceeded by the MERGE even
#: though no single fragment (200-480 samples = 4-9.6 ms)... also exceeds it. Use a
#: tighter set below when a single fragment must be under the limit.
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000)]


# ===========================================================================
# the vocabulary
# ===========================================================================
def test_a_flag_is_a_set_not_a_slot():
    assert flag_tokens("") == ()
    assert flag_tokens(None) == ()
    assert flag_tokens(pd.NA) == ()
    assert flag_tokens(float("nan")) == ()
    assert flag_tokens("a") == ("a",)
    assert flag_tokens("a;b;a") == ("a", "b")
    assert flag_tokens(" a ; b ") == ("a", "b")

    assert combine_flags("a", "b") == "a;b"
    assert combine_flags("a;b", "b;c") == "a;b;c"
    assert combine_flags("", "a") == "a" == combine_flags("a", "")
    assert combine_flags() == ""


def test_has_flag_finds_a_token_that_equality_would_miss():
    """The reason ``has_flag`` exists: once a row can carry two reasons, every
    ``df['flag'] == token`` in the codebase becomes a silent miss."""
    union = combine_flags(FLAG_ZERO_BASELINE, FLAG_CLIPPED)
    assert union != FLAG_CLIPPED               # equality would say "not clipped"
    assert has_flag(union, FLAG_CLIPPED) is True
    assert has_flag(union, FLAG_ZERO_BASELINE) is True
    assert has_flag(union, FLAG_MAX_DWELL) is False

    col = pd.Series(["", FLAG_CLIPPED, union], dtype="string")
    assert list(has_flag(col, FLAG_CLIPPED)) == [False, True, True]
    assert list((col == FLAG_CLIPPED).to_numpy()) == [False, True, False]   # the miss


def test_inherited_curation_keeps_a_hand_reject_and_drops_an_automatic_one():
    # a bare reject (no reason) is a person's decision -> survives
    assert inherited_curation("", False) == ("", False)
    # a reject fully explained by a re-derivable rule -> re-judged, not inherited
    assert inherited_curation(FLAG_MAX_DWELL, False) == ("", True)
    assert inherited_curation(FLAG_CLIPPED, False) == ("", True)
    # a curator's own token survives, and so does the rejection it explains
    assert inherited_curation("looks_like_a_bubble", False) == ("looks_like_a_bubble", False)
    # mixed: the curator's token is not a derived one, so the reject is not automatic
    got = inherited_curation(combine_flags(FLAG_CLIPPED, "bubble"), False)
    assert got == ("bubble", False)
    # accepted rows contribute nothing to reject
    assert inherited_curation([FLAG_MAX_DWELL, ""], [False, True]) == ("", True)
    # ...and ONE hand-rejected row in a run rejects the run (AND, never OR)
    assert inherited_curation(["", ""], [False, True]) == ("", False)


def test_a_rule_that_cannot_be_re_derived_is_inherited_instead():
    """``max_dwell_s`` is only re-derivable when the limit is known. Narrow the derived
    set and the token has to survive, or the reason for a rejection is destroyed."""
    narrowed = DERIVED_FLAGS - {FLAG_MAX_DWELL}
    assert inherited_curation(FLAG_MAX_DWELL, False, derived=narrowed) == (
        FLAG_MAX_DWELL, False
    )


# ===========================================================================
# characterize owns the rules
# ===========================================================================
def _flat(n=2000):
    rng = np.random.default_rng(0)
    return np.full(n, 100.0) + rng.normal(0.0, 1.0, n)


def test_characterize_flags_a_clipped_event_without_any_caller_asking():
    cur = _flat()
    cur[:300] -= 50.0                       # an event running off the START of the array
    base = np.full(cur.size, 100.0)
    df = characterize(cur, base, np.array([0]), np.array([300]), FS, -1, noise_rms=1.0)
    assert has_flag(df["flag"].iloc[0], FLAG_CLIPPED)
    assert bool(df["accepted"].iloc[0]) is False


def test_clip_bounds_lets_a_padded_window_tell_a_window_edge_from_a_recording_edge():
    """Without the translation every rebuilt row would look clipped at both ends,
    because a padded window ALWAYS has its event touching neither... and the window
    itself always ends at the window edge."""
    cur = _flat()
    cur[100:400] -= 50.0
    base = np.full(cur.size, 100.0)
    # the window was cut out at absolute lo=5_000 of a 100_000-sample recording
    df = characterize(cur, base, np.array([100]), np.array([400]), FS, -1,
                      noise_rms=1.0, clip_bounds=(0 - 5_000, 100_000 - 5_000))
    assert str(df["flag"].iloc[0]) == ""
    assert bool(df["accepted"].iloc[0]) is True


def test_characterize_composes_two_reasons_instead_of_overwriting_one():
    cur = _flat()
    cur[:300] -= 50.0
    base = np.zeros(cur.size)                # |baseline| == 0 -> zero_baseline
    df = characterize(cur, base, np.array([0]), np.array([300]), FS, -1, noise_rms=1.0)
    tokens = flag_tokens(df["flag"].iloc[0])
    assert FLAG_ZERO_BASELINE in tokens and FLAG_CLIPPED in tokens


def test_inherit_flag_and_accepted_are_unioned_and_anded():
    cur = _flat()
    cur[100:400] -= 50.0
    base = np.full(cur.size, 100.0)
    df = characterize(cur, base, np.array([100]), np.array([400]), FS, -1, noise_rms=1.0,
                      clip_bounds=(-5_000, 95_000),
                      inherit_flag="bubble", inherit_accepted=False)
    assert flag_tokens(df["flag"].iloc[0]) == ("bubble",)
    assert bool(df["accepted"].iloc[0]) is False


def test_max_dwell_is_characterizes_rule_now_and_a_clipped_row_still_wins():
    """Pins the precedence the detector documented: a clipped event's dwell is not a
    measurement, so ``dwell>max_dwell_s`` would be a FALSE explanation of the row."""
    cur = _flat()
    cur[:300] -= 50.0
    base = np.full(cur.size, 100.0)
    df = characterize(cur, base, np.array([0]), np.array([300]), FS, -1,
                      noise_rms=1.0, max_dwell_s=1e-4)
    assert flag_tokens(df["flag"].iloc[0]) == (FLAG_CLIPPED,)


# ===========================================================================
# C1 -- the merge
# ===========================================================================
def test_c1_a_merge_cannot_un_reject_an_over_length_event_and_erase_the_reason():
    """THE headline. Three fragments, one already flagged ``dwell>max_dwell_s``, merged
    under a declared 2 ms limit. Before the fix the merged row came back ACCEPTED with
    ``flag=''`` at 20 ms -- 10x the limit -- and fed the censored dwell fit whose
    ``t_max`` is that very limit."""
    es, signal = _detect(_FRAGMENTS, max_dwell_s=2e-3)
    assert has_flag(es.df["flag"], FLAG_MAX_DWELL).any(), "fixture must flag a fragment"

    merged, n_merged = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert n_merged == 2 and len(merged) == 1
    row = merged.df.iloc[0]
    assert float(row["dwell_s"]) > 2e-3                    # it really is over the limit
    assert has_flag(row["flag"], FLAG_MAX_DWELL), row["flag"]
    assert bool(row["accepted"]) is False

    # ...and the accepted subset -- what every fit and export reads -- is empty.
    assert len(merged.accepted.df) == 0


def test_c1_the_merged_row_is_re_judged_not_merely_inherited():
    """The rule is applied to the MERGED span on its own merits: a run whose fragments
    are each under the limit but whose union is over it is flagged, even though no
    source row carried the token."""
    es, signal = _detect(_FRAGMENTS, max_dwell_s=1.5e-2)
    assert not has_flag(es.df["flag"], FLAG_MAX_DWELL).any(), "no fragment is over 15 ms"
    assert bool(es.df["accepted"].all())

    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    row = merged.df.iloc[0]
    assert float(row["dwell_s"]) > 1.5e-2
    assert has_flag(row["flag"], FLAG_MAX_DWELL)
    assert bool(row["accepted"]) is False


def test_c1_a_merged_clipped_event_keeps_saying_it_is_clipped():
    spans = [(0, 200), (220, 500)]      # the first fragment runs off the start of the span
    es, signal = _detect(spans)
    assert has_flag(es.df["flag"], FLAG_CLIPPED).any()

    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    row = merged.df.iloc[0]
    assert int(row["start_idx"]) == 0
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False


def test_c1_provenance_cannot_claim_a_clipped_event_no_row_admits_to():
    """The tell the hunt found: ``n_clipped_boundary=1`` while zero rows carried the
    flag. The count is read back off the rows now, on both sides of a merge."""
    spans = [(0, 200), (220, 500), (30_000, 30_400)]
    es, signal = _detect(spans)
    n = int(es.provenance["n_clipped_boundary"])
    assert n == int(has_flag(es.df["flag"], FLAG_CLIPPED).sum()) == 1

    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert int(merged.provenance["n_clipped_boundary"]) == int(
        has_flag(merged.df["flag"], FLAG_CLIPPED).sum()
    )


def test_an_ordinary_merge_is_untouched_by_all_of_this():
    """The control: nothing flagged in, nothing flagged out, still accepted."""
    es, signal = _detect(_FRAGMENTS)
    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert str(merged.df["flag"].iloc[0]) == ""
    assert bool(merged.df["accepted"].iloc[0]) is True


# ===========================================================================
# C21 -- a rebuilt row, in the pure layer the Inspector calls
# ===========================================================================
def test_c21_a_resized_row_never_ends_up_rejected_with_no_reason():
    """The C21 shape exactly: the prior row is clipped + rejected, and the resize moves
    it OFF the boundary. Before the fix ``flag`` was blanked and ``accepted=False`` was
    preserved -- rejected, with nothing to say why, invisible to the very filter that
    is supposed to find it."""
    signal = _trace([(0, 400)], n=50_000)
    prior = pd.Series({"flag": FLAG_CLIPPED, "accepted": False})

    # still touching the recording edge -> still clipped, still rejected, still SAYS so
    row = recharacterize_span(signal, FS, 0, 400, prior=prior, polarity=-1)
    assert has_flag(row["flag"].iloc[0], FLAG_CLIPPED)
    assert bool(row["accepted"].iloc[0]) is False

    # dragged off the edge -> the automatic reason is gone, so the rejection goes too
    row = recharacterize_span(signal, FS, 10, 400, prior=prior, polarity=-1)
    assert str(row["flag"].iloc[0]) == ""
    assert bool(row["accepted"].iloc[0]) is True


def test_c21_a_hand_reject_survives_a_resize():
    signal = _trace([(1_000, 1_400)], n=50_000)
    prior = pd.Series({"flag": "", "accepted": False})       # a bare reject == by hand
    row = recharacterize_span(signal, FS, 1_000, 1_300, prior=prior, polarity=-1)
    assert bool(row["accepted"].iloc[0]) is False


def test_c21_a_rebuilt_row_off_the_boundary_is_not_flagged_by_its_own_padding():
    signal = _trace([(1_000, 1_400)], n=50_000)
    row = recharacterize_span(signal, FS, 1_000, 1_400, prior=None, polarity=-1)
    assert str(row["flag"].iloc[0]) == ""
    assert bool(row["accepted"].iloc[0]) is True
    assert int(row["start_idx"].iloc[0]) == 1_000       # absolute, not window-relative
    assert int(row["end_idx"].iloc[0]) == 1_400


def test_recharacterize_span_re_derives_max_dwell_from_provenance():
    signal = _trace([(1_000, 1_400)], n=50_000)
    prov = {"max_dwell_s": 1e-3, "polarity": -1}
    row = recharacterize_span(signal, FS, 1_000, 1_400, prior=None, provenance=prov)
    assert float(row["dwell_s"].iloc[0]) > 1e-3
    assert has_flag(row["flag"].iloc[0], FLAG_MAX_DWELL)
    assert bool(row["accepted"].iloc[0]) is False


def test_recharacterize_span_inherits_max_dwell_when_the_limit_is_unknown():
    """No ``max_dwell_s`` in provenance means the rule cannot be re-derived, so the
    token must be carried rather than silently dropped."""
    signal = _trace([(1_000, 1_400)], n=50_000)
    prior = pd.Series({"flag": FLAG_MAX_DWELL, "accepted": False})
    row = recharacterize_span(signal, FS, 1_000, 1_400, prior=prior, polarity=-1)
    assert has_flag(row["flag"].iloc[0], FLAG_MAX_DWELL)
    assert bool(row["accepted"].iloc[0]) is False


# ===========================================================================
# the structural guard: no second copy of the rule
# ===========================================================================
_SRC = Path(__file__).resolve().parents[2] / "nanoscope"


def _called_name(node: ast.Call) -> str | None:
    """The function name a call site spells, bare OR as a module attribute.

    ``ast.Name`` alone was a hole in both call-shape guards below: an out-of-tree
    repo copy with ``from ..analysis import characterize as _char_mod`` and
    ``_char_mod.characterize(...)`` / ``_char_mod.recharacterize_span(sig, fs, a, b,
    prior=None)`` added to ``gui/event_inspector.py`` -- a fourth private rebuild
    path and a rebuild with no provenance, i.e. the exact defect these guards
    exist for -- passed 52/52. The bare-name spellings of the same two violations
    fail. A guard on a call SHAPE has to see every spelling of the call.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


#: The one place each rule may be written. Everything else must route through
#: ``characterize`` / ``recharacterize_span``.
_RULE_OWNER = {
    "nanoscope/analysis/characterize.py",
    "nanoscope/analysis/events.py",
}


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


@pytest.mark.parametrize(
    "token", [FLAG_CLIPPED, FLAG_MAX_DWELL, FLAG_ZERO_BASELINE, FLAG_NON_FINITE]
)
def test_no_module_spells_a_flag_token_out_by_hand(token):
    """The defect was three call sites each applying a rule its own way. A module that
    writes the LITERAL is a module deciding for itself what the rule means; import the
    constant and let ``characterize`` apply it.

    Docstrings and comments are exempt (they are not ``ast.Constant`` unless they are
    the module/def docstring, which this filters), so the prose above can keep naming
    the tokens."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC.parent))
        if rel in _RULE_OWNER:
            continue
        tree = ast.parse(path.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and token in node.value and node.value not in docstrings):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"{token!r} is written by hand in {offenders}; import it from analysis.events "
        "and let characterize apply the rule"
    )


def test_every_row_rebuild_goes_through_the_one_function():
    """``characterize`` may be called directly only by a DETECTOR (which owns a whole
    analysed span). Anything rebuilding a row inside a padded window must call
    ``recharacterize_span``, or it is a fourth private copy waiting to drift."""
    allowed = {
        "nanoscope/analysis/characterize.py",   # defines it
        "nanoscope/analysis/detect.py",         # the Schmitt detector
        "nanoscope/analysis/detect_cusum.py",   # the CUSUM detector
        "nanoscope/analysis/ml_modules.py",     # the ML detector
    }
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC.parent))
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == "characterize":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"characterize() is called directly from {offenders}; a row rebuilt outside a "
        "detector must go through recharacterize_span so it cannot forget flag/accepted"
    )


def test_c21_holds_when_the_row_also_carries_a_user_label():
    """C21, re-checked with labelling in the picture — the path an adversarial review found.

    A curator boxes a run of tiles in the Event gallery (default filter "All", so
    auto-rejected tiles are in the sweep), presses a digit, then tidies that event's mask in
    the Event inspector. Before the fix the resize returned ``accepted=False`` with the
    ``clipped_by_boundary`` reason correctly re-derived away — a real, fully-measured event
    silently dropped from every export, both histograms and the censored dwell fit, with
    nothing on the row saying why. Exactly the shape C21 exists to ban.

    The cause: ``inherited_curation`` read ANY non-derived token as a curator's reason, and a
    label is a non-derived token. A label says what an event IS, never why it was rejected.
    """
    import numpy as np
    import pandas as pd

    from nanoscope.analysis.characterize import recharacterize_span
    from nanoscope.analysis.events import FLAG_CLIPPED
    from nanoscope.analysis.labels import parse_labels, set_label

    rng = np.random.default_rng(0)
    fs = 50_000.0
    sig = np.full(2000, 100.0) + rng.normal(0, 1.0, 2000)
    sig[0:400] -= 30.0                       # an event straddling the analysed-span edge

    control = recharacterize_span(
        sig, fs, 10, 400,
        prior=pd.Series({"flag": FLAG_CLIPPED, "accepted": False}), polarity=-1)
    labelled = recharacterize_span(
        sig, fs, 10, 400,
        prior=pd.Series({"flag": set_label(FLAG_CLIPPED, "molecule", "apo-ferritin"),
                         "accepted": False}), polarity=-1)

    def _one(row, col):
        """recharacterize_span hands back a one-row frame; take the scalar."""
        v = row[col]
        return v.iloc[0] if hasattr(v, "iloc") else v

    assert bool(_one(control, "accepted")) is True, "the unlabelled control no longer re-judges"
    assert bool(_one(labelled, "accepted")) is True, "a label froze an automatic rejection"
    assert parse_labels(str(_one(labelled, "flag"))) == {"molecule": "apo-ferritin"}
