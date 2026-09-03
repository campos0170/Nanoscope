"""What every reader must answer the same way.

Four readers decode four formats, but they are asked the same four questions, and every
defect in this module's history came from four different answers to one of them:

1. **How many samples are really here?**  A truncated TDMS reported ``n_samples`` = whatever
   nptdms managed to recover, with ``issues=()``; a truncated ABF was correctly refused.
   That number is the denominator of every event rate, capture rate and duration in the app,
   so a short read is not a cosmetic problem — it silently re-attributes the whole result to
   a recording the user does not have. :func:`short_read_issue` is the one sentence for it.

2. **What is the sample rate, and how sure are we?**  Both the text and the HDF5 reader
   measure ``dt`` from the first ~1000 rows and then describe the whole file with it. Measured
   on a CSV whose interval changes at row 1500: 0.0600 s reported for a 0.3300 s recording —
   5.5x wrong, ``issues=()``. The rate itself is not the fix (changing how it is derived would
   move every dwell time on every well-formed file); *saying that the rest of the file
   disagrees* is. :func:`rate_disagrees_issue` is the one sentence for it.

3. **Which channels exist, and why was one rejected?**  The HDF5 reader silently dropped a
   ``/current`` stored as int32 raw ADC counts, or shaped ``(1, N)``: the channel was simply
   absent from the Files list with nothing anywhere saying why. A refusal the user cannot see
   is indistinguishable from a file that never had the channel. :func:`rejected_channel_issue`
   is the one sentence for it.

4. **What encoding is this text in?**  The probe opened with ``errors="replace"`` and
   ``pd.read_csv`` did not, so a Latin-1 CSV off Windows instrument software probed clean with
   a channel called ``curr?nt`` and then failed at load. :func:`fallback_encoding_note`.

Plus the one structural rule the four of them each re-implemented, three of them differently:

5. **The load window has TWO halves, and both are answered here.**  ``meta.loaded_range`` is
   derived from what was READ, never from what was ASKED: the text reader assigned the request
   verbatim, so ``(500, 8000)`` on a 2000-row CSV produced ``n_samples=1500`` beside
   ``loaded_range=(500, 8000)`` — a value that then went into the project manifest and was
   replayed on restore. :func:`window_loaded` is the only writer of that field, and
   :meth:`nanoscope.core.types.RecordingMeta.__post_init__` refuses a meta whose window and
   sample count disagree, so a fifth reader cannot get it wrong quietly.

   ``meta.t0_s`` is the other half, and it was the half nobody unified: it is the experimental
   time of the FIRST LOADED SAMPLE, because that is what the app builds its x-axis from
   (``core.session.x_axis``: ``arange(n)/fs + meta.t0_s``) and what a segment rebases against
   (``core.segment``: ``t0_abs_s = parent.meta.t0_s + a/fs``). The HDF5 reader read it at the
   window's own index and the other three left it at the file's origin, so the same window of
   the same experiment carried two different clocks depending on the format. Measured on the
   owner's ABF: a window starting 20 s in reported ``t0_s`` = 61.303 s, the file's own start.
   :func:`window_t0` is the one answer.

Nothing here decodes anything. It is deliberately the boring half of ``io``: the sentences,
the clamp, and the one mechanism (:func:`decoder_messages`) that stops a decoding library's
complaint from reaching Python logging and nowhere else.
"""
from __future__ import annotations

import logging
import math
import threading
from contextlib import contextmanager
from typing import Iterator

from ..core.types import SampleRange
from .errors import UnsupportedFormatError

__all__ = [
    "ReaderReport",
    "clamp_request",
    "window_loaded",
    "window_t0",
    "short_read_issue",
    "short_window_issue",
    "rate_disagrees_issue",
    "rate_interval_changes_issue",
    "rejected_channel_issue",
    "partial_timebase_issue",
    "oversized_timebase_issue",
    "fallback_encoding_note",
    "lossy_encoding_issue",
    "decoder_messages",
    "decoder_message_issue",
]


# =================================================================================================
# The accumulator
# =================================================================================================

class ReaderReport:
    """What a reader has to tell the user, gathered in one place and in order.

    ``issues`` are non-fatal PROBLEMS (the recording is not what it looks like); ``notes`` are
    honest ADVISORIES (something true the user needs to know to interpret it). Both are rendered
    verbatim by the Files dock, which is the whole point: a problem that reaches only Python
    logging has not been reported to anybody.

    Deduplicating, because ``probe()`` and ``load()`` share one scan and a rebuilt
    ``RecordingMeta`` must not accumulate the same sentence twice (the same idempotence
    ``TraceDocument._report_nonfinite`` needs, for the same reason).
    """

    __slots__ = ("_issues", "_notes")

    def __init__(self, *, issues: tuple[str, ...] = (), notes: tuple[str, ...] = ()) -> None:
        self._issues: list[str] = []
        self._notes: list[str] = []
        for text in issues:
            self.issue(text)
        for text in notes:
            self.note(text)

    def issue(self, text: str) -> None:
        if text and text not in self._issues:
            self._issues.append(text)

    def note(self, text: str) -> None:
        if text and text not in self._notes:
            self._notes.append(text)

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(self._issues)

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(self._notes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ReaderReport(issues={self._issues!r}, notes={self._notes!r})"


# =================================================================================================
# 5. The load window
# =================================================================================================

def clamp_request(requested: SampleRange, full_n: int, *, name: str,
                  report: "ReaderReport | None" = None) -> tuple[int, int]:
    """The ONE clamp of a requested window against a known file length.

    Returns ``(start, stop)`` with ``0 <= start < stop <= full_n``. Raises
    :class:`~nanoscope.io.errors.UnsupportedFormatError` when the request selects nothing at
    all, which is the one case a reader cannot honour and must refuse out loud.

    Pass ``report`` and the clamp also SAYS SO, via :func:`short_window_issue`. That argument
    exists because reporting it was left to the readers and three of the four forgot: ABF,
    HDF5 and TDMS all clamped correctly and then said nothing, so a project reopened against a
    re-exported shorter recording came back quietly holding a different span than the one its
    stored results were measured on. The clamp is the only place that can see the discrepancy,
    so the sentence belongs here rather than at four call sites — a fifth reader gets it by
    calling the function it already has to call.

    Readers that only learn their length by reading (the text reader: a CSV's row count is an
    estimate until the last row is parsed) do not call this — they clamp ``start`` themselves
    and let :func:`window_loaded` state the truth afterwards, reporting the same sentence
    directly.
    """
    start = max(0, min(int(requested.start), int(full_n)))
    stop = max(start, min(int(requested.stop), int(full_n)))
    if stop <= start:
        raise UnsupportedFormatError(
            f"{name}: the requested sample range [{requested.start:,}, {requested.stop:,}) "
            f"selects nothing in a {int(full_n):,}-sample recording.")
    asked = int(requested.stop) - int(requested.start)
    if report is not None and stop - start < asked:
        report.issue(short_window_issue(
            name, start=start, asked=asked, got=stop - start))
    return start, stop


def window_loaded(requested: SampleRange | None, start: int, n_read: int) -> SampleRange | None:
    """The ONLY writer of ``RecordingMeta.loaded_range``.

    ``None`` for a whole-file load. Otherwise the span the document ACTUALLY holds:
    ``start`` plus the number of samples that were read — never the number that was asked for.

    This is the whole fix for the class of defect where a reader recorded its request: the
    value cannot disagree with ``n_samples`` because it is computed from it, and
    ``RecordingMeta`` refuses any meta where the two disagree anyway.
    """
    if requested is None:
        return None
    return SampleRange(int(start), int(start) + int(n_read))


def window_t0(file_t0_s: float, start: int, sample_rate_hz: float | None) -> float:
    """The ONE answer to "what experimental time does sample 0 of this document carry".

    ``meta.t0_s`` is not the file's origin — it is the origin OF WHAT WAS LOADED. Everything
    downstream reads it that way: :meth:`nanoscope.core.session.Session.x_axis` builds
    ``arange(n)/fs + meta.t0_s``, the inverse in the Event Inspector is
    ``(x - meta.t0_s) * fs``, and :func:`nanoscope.core.segment.make_segment` rebases a segment
    with ``parent.meta.t0_s + a/fs``. A reader that leaves the file's own ``t0`` on a windowed
    load therefore labels every cursor readout, every hand measurement and every segment's
    absolute time with a clock that is ``start/fs`` seconds fast.

    Measured before this existed: the owner's 606 MB ATP recording is the one file over
    ``SOFT_SAMPLE_CEILING``, so a window IS its normal load path; a window at sample 30,000,000
    reported the file's ``t0`` and put every experimental time 600 s early. HDF5 alone was
    right, because it reads the value out of ``/time`` at the window's own index.

    ``start`` = 0 (or an unknown/unusable rate) returns the file's ``t0`` unchanged, so a
    whole-file load is bit-identical to what it always was.
    """
    start_i = int(start)
    t0 = float(file_t0_s)
    if start_i <= 0:
        return t0
    if sample_rate_hz is None:
        return t0
    fs = float(sample_rate_hz)
    if not math.isfinite(fs) or fs <= 0.0:
        return t0
    return t0 + start_i / fs


# =================================================================================================
# 1. How many samples are really here
# =================================================================================================

def short_read_issue(name: str, *, recovered: int, detail: str) -> str:
    """The one sentence for "this file is truncated and you got part of a recording".

    Says what is at stake rather than what happened: the sample count is the denominator of
    every rate in the app, so a short read silently re-attributes the whole analysis to a
    recording that does not exist.
    """
    return (f"{name} is truncated — {detail}. Only the {recovered:,} samples that survived "
            f"were read, so every duration, event rate and capture rate computed here belongs "
            f"to those {recovered:,} samples, not to the recording this acquisition started. "
            f"Do not pool them with results from complete files.")


def short_window_issue(name: str, *, start: int, asked: int, got: int) -> str:
    """The one sentence for "the window asked for more of this file than there is".

    Deliberately NOT :func:`short_read_issue`: the file is not damaged, the REQUEST outran it.
    That happens on a project restore whose recording has been re-exported shorter since the
    project was saved — the document comes back correctly clamped (that is what
    :func:`window_loaded` guarantees) and used to say nothing at all about the samples the
    stored analysis was measured on and this one is not.
    """
    return (f"{name}: this load asked for {asked:,} samples from sample {start:,} and the file "
            f"ends after {start + got:,}, so it holds {got:,}. Nothing is wrong with the file — "
            f"but if this window came from a saved project, the recording is not the length it "
            f"was when that project was saved, and results measured on the two are not the same "
            f"recording's.")


# =================================================================================================
# 2. What is the sample rate, and how sure are we
# =================================================================================================

def rate_disagrees_issue(*, hz: float, measured_over: int, source: str,
                         implied_s: float, actual_s: float) -> str:
    """The one sentence for "the rate was measured from a window and the rest disagrees".

    The rate is deliberately NOT silently corrected. It is measured the same way it always
    was — changing that would move every dwell time on every well-formed file — and this
    sentence states the size of the error instead, in the unit the user cares about (the
    recording's duration) and as the factor every derived number is wrong by.
    """
    factor = (actual_s / implied_s) if implied_s else float("inf")
    return (f"the sample rate {hz:,.6g} Hz was measured from the first {measured_over:,} "
            f"{source}, and the rest of the file disagrees: the time base spans "
            f"{actual_s:,.6g} s where that rate implies {implied_s:,.6g} s ({factor:,.4g}x). "
            f"The interval is not constant through this file, so every dwell time, event rate "
            f"and duration derived from that rate is wrong by about that factor.")


def rate_interval_changes_issue(*, hz: float, measured_over: int, source: str,
                                dt_head: float, dt_tail: float) -> str:
    """The one sentence for "the interval near the end of the file is not the one at the start".

    The estimate-free half of question 2, and the only half a cheap probe can answer: comparing
    two directly measured intervals needs no sample count, so it is true before anything has
    been loaded. The exact size of the error comes later, from
    :func:`rate_disagrees_issue`, once the real sample count exists.
    """
    factor = (dt_tail / dt_head) if dt_head else float("inf")
    return (f"the sample rate {hz:,.6g} Hz was measured from the first {measured_over:,} "
            f"{source}, and the interval near the END of this file is different: "
            f"{dt_tail:,.6g} s against {dt_head:,.6g} s ({factor:,.4g}x). The interval is not "
            f"constant through this file, so every dwell time, event rate and duration derived "
            f"from that one rate is wrong for part of it.")


def partial_timebase_issue(*, n_time: int, n_signal: int) -> str:
    """The one sentence for "the time base does not cover the signal"."""
    return (f"/time holds {n_time:,} samples for {n_signal:,} signal samples, so the time base "
            f"covers only part of this recording; the sample rate above is measured from the "
            f"part that exists and the rest is assumed to continue at the same rate.")


def oversized_timebase_issue(*, n_time: int, n_signal: int, dt: float) -> str:
    """The one sentence for "the time base is LONGER than the signal".

    The mirror of :func:`partial_timebase_issue`, and it was silent while its mirror spoke.
    It matters more, not less: a ``/time`` that runs past the signal means either the signal
    datasets stopped early (an interrupted write, so this is a truncated recording) or the time
    base does not belong to these samples at all. Either way the duration the app reports is
    ``n_signal`` long and the file's own clock says otherwise.
    """
    return (f"/time holds {n_time:,} samples but the signal channels hold {n_signal:,}, so this "
            f"file's own time base runs {(n_time - n_signal) * float(dt):,.6g} s past the last "
            f"sample there is. Either the signal datasets were cut short while being written or "
            f"this time base is not this recording's; the {n_signal:,} samples that exist are "
            f"what every duration and rate here is computed from.")


# =================================================================================================
# 3. Which channels exist, and why one was rejected
# =================================================================================================

def rejected_channel_issue(dataset: str, why: str) -> str:
    """The one sentence for "this file has a channel we recognised and could not use".

    A channel that is simply missing from the list, with nothing said, is indistinguishable
    from a file that never had one — which is how an int32 ``/current`` (raw ADC counts) came
    to be dropped from a real recording in silence.
    """
    return (f"{dataset} is present in this file but is not loaded as a channel: {why}. "
            f"That is why it does not appear in the channel list.")


# =================================================================================================
# 4. What encoding is this text in
# =================================================================================================

def fallback_encoding_note(encoding: str, *, why: str) -> str:
    """The one sentence for "this file is not UTF-8 and here is what we read it as"."""
    return (f"This file is not valid UTF-8 ({why}), so it was read as {encoding}. Column names "
            f"with accented or symbol characters may not be exactly as the instrument wrote "
            f"them; the numbers are unaffected.")


def lossy_encoding_issue(name: str, encoding: str, *, why: str) -> str:
    """The one sentence for "the declared encoding broke part-way through the file".

    An ``issue`` rather than a note, unlike :func:`fallback_encoding_note`: the head of the file
    decoded cleanly and something further in did not, which means the file is not consistently
    in one encoding and no reader can be sure which characters it got right.
    """
    return (f"{name} could not be decoded as {encoding} all the way through ({why}), so it was "
            f"re-read as Latin-1. Every byte is accounted for, but a file that changes encoding "
            f"part-way is not one this reader can vouch for: check any column name that looks "
            f"wrong, and re-export as UTF-8 if it is.")


# =================================================================================================
# The mechanism: a decoding library's complaint must not stop at Python logging
# =================================================================================================

class _ThreadScopedCapture(logging.Handler):
    """Collect records emitted on THIS thread only.

    Thread-scoped because a reader runs on the load worker while the rest of the app keeps
    logging on the main thread, and an issue attributed to the user's file must actually have
    come from reading the user's file.
    """

    def __init__(self, thread_id: int) -> None:
        super().__init__(level=logging.WARNING)
        self._thread_id = thread_id
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        if record.thread != self._thread_id:
            return
        try:
            text = record.getMessage()
        except Exception:  # pragma: no cover - a broken %-format in someone else's log call
            text = str(record.msg)
        if text and text not in self.messages:
            self.messages.append(text)


@contextmanager
def decoder_messages(*logger_names: str) -> Iterator[list[str]]:
    """Capture what a decoding library complains about, so the user can be told.

    nptdms recovers what it can from a truncated file and says so — to the ``nptdms`` logger,
    which in this app goes to stderr and nowhere the user will ever look. Measured on a file
    cut to 30 %: 20,000 samples became 5,983, ``issues=()``, ``notes=()``, and two WARNING
    lines on a terminal nobody has open.

    Yields a list that fills as the block runs; the caller decides which messages are worth a
    ``meta.issues`` entry (usually via :func:`decoder_message_issue`).

    Attaches a handler rather than replacing configuration, so the library's own logging keeps
    working exactly as before and nothing downstream of it changes. A logger's level is touched
    only when its EFFECTIVE level would have swallowed the warning outright, and only ever
    downward — no record that used to reach another handler stops reaching it — and it is put
    back the way it was found, ``NOTSET`` included.
    """
    handler = _ThreadScopedCapture(threading.get_ident())
    attached: list[logging.Logger] = []
    lowered: list[tuple[logging.Logger, int]] = []
    try:
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            if logger.getEffectiveLevel() > logging.WARNING:
                lowered.append((logger, logger.level))
                logger.setLevel(logging.WARNING)
            logger.addHandler(handler)
            attached.append(logger)
        yield handler.messages
    finally:
        for logger in attached:
            logger.removeHandler(handler)
        for logger, previous in lowered:
            logger.setLevel(previous)


def decoder_message_issue(library: str, message: str) -> str:
    """The one wrapper for a decoding library's own words.

    Quoted rather than paraphrased, and attributed, because the library is describing something
    about the file that this app did not model: paraphrasing it would either lose the detail or
    invent one.
    """
    return f"the {library} decoder reported a problem reading this file: “{message}”"
