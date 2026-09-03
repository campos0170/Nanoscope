"""PERS-004 — the consequence sentence needs no fingerprint, so it may not be gated
behind one.

A project restore says two different things about the recording it reattached to:

  * "this is not the file the project was saved from" — a claim about the FILE, which is
    only ever made from evidence (a saved :class:`SourceFingerprint`), and whose silence
    means "no evidence", never "verified identical"; and
  * "9 of your 11 restored events lie past the end of this recording" — a comparison
    between the project's own saved event indices and the sample count of the file now in
    hand, which needs no stamp at all.

The second used to be composed INSIDE the first, so a project carrying no fingerprint —
every project written before manifest version 4 (``project.MIN_READ_VERSION`` is frozen
at 1 so that "every project the owner already has still opens"), and any project whose
source could not be read at save time — reattached its whole saved event table onto a
re-exported, shorter recording in total silence. Measured on one project saved two ways:
the stamped copy named the 9 events in one sentence; the unstamped copy restored all 11,
9 of them entirely past the end of a 20,000-sample file, with ``meta.issues == []``, no
banner, no dialog and a status bar reading only "Project restored."

:func:`~nanoscope.core.identity.describe_restored_source` is the seam both surfaces pass
through. These tests pin it, and pin that the stamped sentence did not move a character.
"""

from __future__ import annotations

from nanoscope.core.identity import (
    SourceFingerprint,
    describe_restored_source,
    describe_source_change,
)

NAME = "pore.csv"
ADVICE = ("Re-detect on it, or point the project at the original recording — the events "
          "on screen were measured on different data.")


def _saved(n_samples=120_000, size=3_145_403, digest="sha256:aaaa", mtime_ns=1_000):
    return SourceFingerprint(n_samples=n_samples, size=size, digest=digest,
                             mtime_ns=mtime_ns)


def _now(size=524_129, digest="sha256:bbbb", mtime_ns=2_000):
    return SourceFingerprint(n_samples=0, size=size, digest=digest, mtime_ns=mtime_ns)


# ================================================================================================
# 1. the case that was silent
# ================================================================================================

def test_a_project_with_no_fingerprint_still_reports_the_events_that_do_not_fit():
    said = describe_restored_source(
        NAME, None, None, loaded_n_samples=20_000,
        restored_events=11, events_past_end=9)
    assert said
    assert "9 of 11 restored event(s) lie past the end of this recording" in said
    assert "20,000 samples" in said
    assert "carries no fingerprint" in said       # names WHY the file itself is unnamed
    assert ADVICE in said
    # It must NOT make the claim it has no evidence for.
    assert "is not the file this project was saved from" not in said


def test_the_unknown_fingerprint_is_still_never_a_mismatch():
    """The documented contract a fix must not trample: silence about the FILE stays."""
    assert describe_source_change(None, _now(), loaded_n_samples=999) == ""
    # and with no harm to report, the whole sentence is still silence
    assert describe_restored_source(NAME, None, _now(), loaded_n_samples=999,
                                    restored_events=11, events_past_end=0) == ""


def test_a_stamped_project_says_exactly_what_it_said_before():
    """Byte-for-byte the sentence the shipped build already produced: a rewording would
    make two builds disagree about one file for no reason."""
    said = describe_restored_source(
        NAME, _saved(), _now(mtime_ns=0), loaded_n_samples=20_000,
        restored_events=11, events_past_end=9)
    assert said == (
        "“pore.csv” is not the file this project was saved from: it now holds 20,000 "
        "samples where the project recorded 120,000 and it is 524,129 bytes where the "
        "project recorded 3,145,403. Its 11 restored event(s) were reattached anyway, "
        "and 9 of them now lie past the end of this recording. " + ADVICE)


def test_a_stamped_project_whose_events_all_fit_says_only_that_the_file_changed():
    said = describe_restored_source(
        NAME, _saved(), _now(mtime_ns=0), loaded_n_samples=20_000,
        restored_events=11, events_past_end=0)
    assert "were reattached anyway." in said
    assert "past the end" not in said


# ================================================================================================
# 2. the two silences stay silent, and the third story stays distinct
# ================================================================================================

def test_nothing_is_said_when_nothing_is_wrong():
    fp = _saved(n_samples=20_000, size=524_129, digest="sha256:same", mtime_ns=7)
    same = SourceFingerprint(n_samples=0, size=524_129, digest="sha256:same", mtime_ns=7)
    assert describe_restored_source(NAME, fp, same, loaded_n_samples=20_000,
                                    restored_events=11, events_past_end=0) == ""
    assert describe_restored_source(NAME, None, None, loaded_n_samples=20_000,
                                    restored_events=0, events_past_end=0) == ""


def test_a_matching_file_with_events_past_its_end_blames_the_project_not_the_file():
    """The stamp says the recording did not move, so the sentence must not imply it did."""
    fp = _saved(n_samples=20_000, size=524_129, digest="sha256:same", mtime_ns=7)
    same = SourceFingerprint(n_samples=0, size=524_129, digest="sha256:same", mtime_ns=7)
    said = describe_restored_source(NAME, fp, same, loaded_n_samples=20_000,
                                    restored_events=11, events_past_end=9)
    assert "still matches the fingerprint" in said
    assert "the project's own event table" in said
    assert "carries no fingerprint" not in said
    assert "is not the file this project was saved from" not in said


def test_the_document_is_named_in_every_sentence_it_writes():
    for saved in (None, _saved()):
        said = describe_restored_source(NAME, saved, _now(), loaded_n_samples=20_000,
                                        restored_events=11, events_past_end=9)
        assert said.startswith("“pore.csv”")


def test_a_missing_sample_count_costs_the_parenthetical_and_nothing_else():
    said = describe_restored_source(NAME, None, None, loaded_n_samples=None,
                                    restored_events=11, events_past_end=9)
    assert "9 of 11 restored event(s) lie past the end of this recording." in said
    assert "samples)" not in said
