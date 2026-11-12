"""The first line of an advisory, for the places that cannot hold all of it.

An advisory (``AnalysisResult.warnings``) is written to be complete: the longest run to
several hundred words, with the reason, the measurement behind it and what to do about
it. Printed whole under a table or in a tooltip, a handful of them push the numbers they
qualify off the screen. This module cuts the READING copy down to a headline and nothing
else: the stored text, the provenance, the exports and the saved project all keep the
advisory byte for byte, and the full text is one click away on the Summary tab.

It is plain Python on purpose: no Qt, nothing from ``gui``. Two panels use it (the
Summary tab's advisory block and the toolbar link's tooltip), and keeping it free of
widgets is what lets both import it without reaching each other.
"""

from __future__ import annotations

import re

__all__ = ["advisory_headline"]

#: Appended to a headline that was cut mid-sentence. A sentence that ends inside the limit
#: needs none: it ends on its own full stop.
_ELLIPSIS = "…"

#: The end of a sentence: ``.``, ``!`` or ``?`` (and any bracket or quotation mark that
#: closes over it) with more text after a space. The stop of "i.e.", "e.g.", "vs." and
#: "cf." belongs to the abbreviation ("-- i.e. the boundary is off"), not to a sentence.
_SENTENCE_END = re.compile(
    r"(?<!i\.e)(?<!e\.g)(?<!\bvs)(?<!\bcf)[.!?][)\]”\"']*(?=\s+\S)")

#: A line break ends a headline too: what follows it is another line, not more of this one.
_LINE_BREAK = re.compile(r"\s*\n")

#: The dash the advisories use for an aside: "... FORM (0.5 s) -- the one selected when the
#: form was filled in -- and this recording ...". The two dashes open and close it.
_DASH = " -- "

#: Where a sentence too long for the limit is cut, best first. The last one that fits is
#: taken, so the headline keeps as much of the sentence as it can.
_CLAUSE_BREAKS = (_DASH, ": ", ", ")

#: Opening and closing marks that must be matched in what is kept: a headline that stops
#: inside "(0.5 s / 0.1 s" or inside a quotation reads as broken.
_PAIRS = (("(", ")"), ("[", "]"), ("“", "”"))

#: A comma that leaves this many words or fewer after the last colon or dash is not worth
#: cutting at: what it keeps is the first item of a list, not a clause.
_FRAGMENT_WORDS = 3

#: Words a phrase cannot end on. A headline cut at a space never ends on one of them
#: ("... are omitted from the"); it steps back to the word before.
_DANGLING = frozenset((
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "at", "in", "on", "by", "for",
    "with", "from", "than", "that", "which", "its", "this", "as", "be", "it"))


def _leaves_open(text: str) -> bool:
    """Whether ``text`` opens a bracket, a curly quotation or a `` -- `` aside it does not
    close (an odd number of dashes), or stops on a dash: a headline never ends on one."""
    return (any(text.count(opening) > text.count(closing) for opening, closing in _PAIRS)
            or text.count(_DASH) % 2 == 1 or text.endswith(_DASH.rstrip()))


def _between_two_numbers(kept: str, rest: str) -> bool:
    """Whether a comma between ``kept`` and ``rest`` sits inside a list of numbers
    ("... of 1.0000, 0.4520 and 0.0902"): ``kept`` ends in one and ``rest`` starts with
    another. Cut there, the first number reads as the only one."""
    return bool(re.match(r"[-+]?\d", rest)) and bool(re.search(r"\d\S*$", kept))


def _dangles(kept: str) -> bool:
    """Whether ``kept`` ends on a word a phrase cannot end on (see ``_DANGLING``)."""
    return kept.split()[-1].lower().strip("(") in _DANGLING


def advisory_headline(text: str, limit: int = 230) -> str:
    """The first sentence of ``text``, in at most ``limit`` characters.

    The headline is always a prefix of ``text.strip()``, with ``…`` added when it was cut
    short: the words are the advisory's own, in its own order, none added and none
    reworded.

    A sentence that ends inside the limit is returned as it ends, full stop and all.
    A longer one is cut where it pauses: at the last `` -- ``, ``: `` or ``, `` that fits,
    never earlier than a third of the limit, and never

    * inside an open bracket or quotation, or between the two dashes of an aside,
    * between two numbers of a list ("... of 1.0, 0.45 and 0.09"),
    * at a comma that keeps only a few words (``_FRAGMENT_WORDS``) after the last colon or
      dash: it cuts at that colon or dash instead.

    With no break to use it is cut at the last space, stepping back over a word a phrase
    cannot end on ("the", "of", "from" ...) and out of anything still open; with no space
    at the limit. ``…`` follows the cut. So ``headline == text.strip()`` means nothing
    was left out, and anything else means the rest of the advisory is hidden: that is how
    a caller decides whether to offer the full text.
    """
    text = str(text).strip()
    ends = [len(text)]
    sentence = _SENTENCE_END.search(text)
    if sentence:
        ends.append(sentence.end())
    line = _LINE_BREAK.search(text)
    if line:
        ends.append(line.start())
    first = text[:min(ends)]
    if len(first) <= limit:
        return first

    # Every cut leaves one character of the limit for the ellipsis.
    room = max(limit - len(_ELLIPSIS), 0)
    breaks = sorted(
        {(found.start(), sep) for sep in _CLAUSE_BREAKS
         for found in re.finditer(re.escape(sep), first) if found.start() <= room},
        reverse=True)
    for cut, sep in breaks:
        if (cut < limit // 3 or _leaves_open(first[:cut])
                or (sep == ", " and _between_two_numbers(first[:cut], first[cut + len(sep):]))):
            continue
        if sep == ", ":
            # The nearest colon or dash before this comma. A comma only a word or two
            # past it ends on the first item of a list ("...: masked median"): take the
            # colon or dash, which is where the sentence paused.
            for earlier, earlier_sep in breaks:
                if earlier < cut and earlier_sep != ", ":
                    words = first[earlier + len(earlier_sep):cut].split()
                    if (len(words) <= _FRAGMENT_WORDS and earlier >= limit // 3
                            and not _leaves_open(first[:earlier])):
                        cut = earlier
                    break
        return first[:cut].rstrip() + _ELLIPSIS

    # No clause break worth taking: the last word that fits.
    cut = first.rfind(" ", 0, room + 1)
    while cut > 0 and (_leaves_open(first[:cut]) or _dangles(first[:cut])):
        cut = first.rfind(" ", 0, cut)
    if cut <= 0:
        cut = room
    return first[:cut].rstrip() + _ELLIPSIS
