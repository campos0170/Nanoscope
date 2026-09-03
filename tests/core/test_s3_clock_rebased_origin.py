"""Regression tests for JAB-04 / MEAS3-02, at the seam: a REBASED document still has a
parent clock, and ``core.clock`` is the one place that knows it.

THE RULE THESE PIN (one clock story per document, told the same way by every surface
that tells one). ``core.segment`` rebases a cut's ``meta.t0_s`` to 0 — deliberately,
permanently, and pinned by ``tests/core/test_segment.py``. Read through ``meta.t0_s``
alone, a segment of the owner's quick apoferritin ABF (t₀ = 1154.9909 s, cut 4 s in) is
then indistinguishable from the −700 mV file whose t₀ genuinely IS 0, so every clock
label in the app said "t₀ = 0, so both clocks read alike here" about data the SAME
window names ``[1159–1165 s]``.

The fix is not per-surface text. ``build_segment_document`` stamps the origin it
rebased away onto the meta (:class:`~nanoscope.core.segment.RebasedMeta`), and
:func:`~nanoscope.core.clock.clock_offset_s` — the single seam the trace axis label, the
event-table header, the PSD span read-out, the crosshair and the manual-measurement
table all read their clock story through — hands it on as
:attr:`~nanoscope.core.clock.ClockOffset.rebased_from_s`. So the label writers cannot
omit it, and a surface added tomorrow inherits the fix.

Pure core: no Qt, no Session. The GUI half is tests/gui/test_s3_segment_clock_surfaces.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core import clock
from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import (
    RebasedMeta,
    build_segment_document,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 50_000.0
N = 400_000
#: The owner's quick −1000 mV ABF t₀, verbatim, so the numbers here are the numbers in
#: the report: a cut at sample 200,000 of it holds data recorded at 1158.9909 s.
T0_QUICK = 1154.9909123675027
CUT = (200_000, 300_000)
CUT_ORIGIN = T0_QUICK + CUT[0] / FS


def _parent(*, t0_s: float = T0_QUICK, fs: float | None = FS) -> TraceDocument:
    x = np.linspace(0.0, 1.0, N, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=__import__("pathlib").Path("/tmp/apo_quick.abf"),
                         format_name="synthetic", n_samples=N, sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META, t0_s=t0_s,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(x,))


def _cut(parent: TraceDocument, i0: int = CUT[0], i1: int = CUT[1], *, uid="seg-a"):
    return build_segment_document(parent, i0, i1, name="clean stretch", segment_uid=uid)


# ================================================================================================
# the witness: a rebased meta carries the origin the rebase removed
# ================================================================================================

def test_the_rebase_is_witnessed_on_the_meta_not_only_on_the_origin_stamp():
    """``t0_s`` still rebases to 0 — that is the contract every panel depends on — and
    the instant it rebased AWAY FROM travels on the meta, where ``core.clock`` is
    handed it. On ``SegmentOrigin`` alone it was invisible to every clock surface,
    because they are all given ``document.meta``, never the document."""
    seg = _cut(_parent())

    assert seg.meta.t0_s == 0.0                       # unchanged contract
    assert isinstance(seg.meta, RebasedMeta)
    assert seg.meta.rebased_from_t0_s == pytest.approx(CUT_ORIGIN)
    assert seg.origin.t0_abs_s == pytest.approx(CUT_ORIGIN)
    # ...and the attribute is the one core.clock looks for, by name.
    assert getattr(seg.meta, clock.REBASE_ATTR) == pytest.approx(CUT_ORIGIN)


def test_the_witness_survives_a_dataclasses_replace():
    """``TraceDocument.__post_init__`` stamps ``load_uid`` with ``replace``, and
    ``Session.add_document`` folds sidecar issues in the same way. A witness that a
    routine ``replace`` dropped would be a witness that vanished on the way to the
    Session — i.e. exactly nowhere useful."""
    from dataclasses import replace

    seg = _cut(_parent())
    again = replace(seg.meta, notes=("edited",))
    assert isinstance(again, RebasedMeta)
    assert again.rebased_from_t0_s == pytest.approx(CUT_ORIGIN)


def test_a_plain_file_recording_carries_no_witness_at_all():
    """The witness means "this app manufactured this zero". A file's declared t₀ = 0 is
    not manufactured, and reading a rebase onto it would be the fabrication
    ``io.contract`` forbids."""
    meta = _parent(t0_s=0.0).meta
    assert clock.rebased_origin_from_meta(meta) is None
    assert clock.rebased_origin_s(clock.clock_offset_s(meta)) is None
    assert clock.clocks_agree(clock.clock_offset_s(meta)) is True


# ================================================================================================
# the seam: one offset, carrying both numbers
# ================================================================================================

def test_the_offset_is_numerically_the_offset_and_carries_the_origin():
    """The value MUST stay 0.0: ``MainWindow`` subtracts it from an x read off the
    trace to get recording seconds, and a segment's axis really does start at 0. The
    provenance rides along instead of replacing it."""
    off = clock.clock_offset_s(_cut(_parent()).meta)

    assert float(off) == 0.0
    assert off + 2.0 == pytest.approx(2.0)            # arithmetic callers untouched
    assert clock.rebased_origin_s(off) == pytest.approx(CUT_ORIGIN)


def test_instrument_origin_is_the_experiment_time_of_sample_zero():
    """The other question, and the one a provenance STAMP wants: not "how far apart are
    the two clocks on screen" (0 here) but "where was this recorded"."""
    parent = _parent()
    seg = _cut(parent)

    assert clock.instrument_origin_s(parent.meta) == pytest.approx(T0_QUICK)
    assert clock.instrument_origin_s(seg.meta) == pytest.approx(CUT_ORIGIN)
    # No time base -> no instant, and none is invented.
    assert clock.instrument_origin_s(_parent(fs=None).meta) is None


def test_two_offsets_that_are_the_same_number_are_not_the_same_clock_statement():
    """Both panels guard their repaint with ``if offset == self._clock_offset_s: return``.
    A genuine t₀ = 0 file and a segment of a t₀ = 1154.99 s recording both give 0.0, so
    plain-float equality left the header saying "this recording's t₀ is 0" about the
    wrong recording — stale state wearing an honest label."""
    zero_file = clock.clock_offset_s(_parent(t0_s=0.0).meta)
    segment = clock.clock_offset_s(_cut(_parent()).meta)

    assert float(zero_file) == float(segment) == 0.0
    assert zero_file != segment
    assert segment != 0.0
    assert zero_file == 0.0                            # a real zero still compares equal
    # ...and it is still a float everywhere else.
    assert zero_file == pytest.approx(0.0)
    assert segment == pytest.approx(0.0)


# ================================================================================================
# what the surfaces are told
# ================================================================================================

def test_a_rebased_document_is_not_told_that_there_is_nothing_to_convert():
    """``clocks_agree`` reads "no distinction is needed". On a segment the two on-screen
    numbers coincide and a distinction IS needed: everything else in the app — the
    segment's own name, its notes, the Files dock, the parent recording — is 1158.99 s
    away. False here is what routes every caller onto its "say which clock" path."""
    seg_off = clock.clock_offset_s(_cut(_parent()).meta)
    assert clock.clocks_agree(seg_off) is False

    # A cut of a t₀ = 0 recording taken from its very first sample HAS no distinction to
    # draw, and must not be given a spurious one.
    flush = clock.clock_offset_s(_cut(_parent(t0_s=0.0), 0, 100_000, uid="seg-z").meta)
    assert clock.clocks_agree(flush) is True


def test_the_axis_label_names_the_instant_the_cut_was_recorded_at():
    label = clock.axis_label("s", clock.INSTRUMENT,
                             clock.clock_offset_s(_cut(_parent()).meta))

    assert "1158.9909" in label
    assert "both clocks read alike here" not in label
    assert "s (" in label and label.startswith("s ")


def test_the_field_note_gives_the_arithmetic_back_to_the_instrument_clock():
    note = clock.field_note(clock.RECORDING,
                            clock.clock_offset_s(_cut(_parent()).meta))

    assert "1158.9909" in note
    assert "This recording's t₀ is 0" not in note
    assert "add" in note.lower()


def test_a_rebased_layer_that_is_also_nudged_says_both_things():
    """A segment can carry a Files-dock alignment nudge too. Naming one of the two
    shifts and silently keeping the other is the half-truth this module exists to stop."""
    off = clock.clock_offset_s(_cut(_parent()).meta, 5.0)

    label = clock.axis_label("s", clock.INSTRUMENT, off)
    assert "1158.9909" in label and "5.0000" in label
    note = clock.field_note(clock.RECORDING, off)
    assert "1158.9909" in note and "5.0000" in note


def test_a_conversion_between_the_two_clock_names_carries_the_rebase():
    """``to_instrument_s`` answers "where is this on the instrument clock". Returning
    the same number back for a segment — which is what adding an offset of 0 does — is
    a conversion that converts nothing while claiming to."""
    off = clock.clock_offset_s(_cut(_parent()).meta)

    assert clock.to_instrument_s(off, 2.0) == pytest.approx(CUT_ORIGIN + 2.0)
    assert clock.to_recording_s(off, CUT_ORIGIN + 2.0) == pytest.approx(2.0)
    # and the both-clocks read-out therefore states two different numbers, not one twice
    both = clock.both_clocks(off, recording_s=2.0)
    assert "2.0000" in both and "1160.9909" in both


# ================================================================================================
# the same rule one generation deeper
# ================================================================================================

def test_a_cut_of_a_cut_keeps_the_grandparent_s_origin():
    """The instance the sweep would have found next. ``t0_abs_s`` was computed from
    ``parent.meta.t0_s``, which for a segment parent is the manufactured 0 — so a
    nested cut of the 1154.99 s recording came back claiming its data was recorded at
    2 s, on every surface at once (name, notes, Files dock, axis label, row stamp)."""
    inner = _cut(_parent())                             # origin 1158.9909
    nested = build_segment_document(inner, 50_000, 90_000,
                                    name="deeper", segment_uid="seg-b")
    expected = CUT_ORIGIN + 50_000 / FS

    assert nested.origin.t0_abs_s == pytest.approx(expected)
    assert nested.meta.rebased_from_t0_s == pytest.approx(expected)
    assert clock.instrument_origin_s(nested.meta) == pytest.approx(expected)
    assert f"{expected:.6g}" in nested.meta.notes[0]
    assert "1159.9909" in clock.axis_label(
        "s", clock.INSTRUMENT, clock.clock_offset_s(nested.meta))


def test_a_cut_of_a_recording_with_no_sample_rate_still_has_no_clock():
    """No time base, no instant — and the rebase witness must not conjure one."""
    seg = _cut(_parent(fs=None), 0, 1000, uid="seg-nofs")
    assert clock.clock_offset_s(seg.meta) is None
    assert clock.instrument_origin_s(seg.meta) is None
    assert clock.axis_label("sample", clock.INSTRUMENT, None) == "sample"
