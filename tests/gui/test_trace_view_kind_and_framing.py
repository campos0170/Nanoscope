"""Two things the trace view got wrong about the picture it draws, both reported by the owner.

**R0 — "why does the app place the pA on the top panel for some traces, and the voltage on the
bottom yet reverses in others."** The ordering rule was never the problem: ``_KIND_RANK`` is
deterministic and both ``plan_panes`` branches go through ``_current_first``. What was wrong is
that the KIND it sorts on and the UNITS it labels with came from different places —
``_effective_units`` honoured ``ChannelCalibration.target_units`` and ``_kind`` read only the
file header — so a channel the user had corrected to 'pA' in the Calibration panel got the pA
axis label and kept the UNKNOWN rank, i.e. stayed BELOW the voltage pane. Two recordings of one
experiment then opened with their panes in opposite orders depending on which of them declared
its channels. The same split un-groups the owner's nisin TDMS, whose two channels are both
``(RAW_VOLTAGE, 'V')`` and therefore shared one pane and one y-axis in OVERLAY.

**"The mask on the gallery does not match the mask on the main panel."** The gallery tile has
always framed an event at ``_PAD_FRACTION = 0.5`` (the event is half the tile); ``zoom_to_span``
defaulted to ``pad_frac=1.5`` (the event is a QUARTER of the pane), and both call sites pass
only ``min_pad``, so the 1.5 is what shipped. One event, two sizes, on two surfaces that are
meant to show the same thing.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.calibration import (
    CalibrationSet,
    ChannelCalibration,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.trace_view import TraceView, plan_panes

FS = 50_000.0
N = 40_000


def _doc(tmp_path: Path, channels: tuple[ChannelInfo, ...], name: str = "rec.dat") -> TraceDocument:
    rng = np.random.default_rng(0)
    data = tuple(rng.standard_normal(N).astype(np.float32) for _ in channels)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=channels,
    )
    return TraceDocument(meta=meta, channels=data)


def _session(doc: TraceDocument, *, mode: ViewMode = ViewMode.STACKED) -> Session:
    session = Session()
    session.add_document(doc)
    for info in doc.meta.channels:
        session.add_layer(doc.doc_id, info.index)
    session.set_view_mode(mode)
    return session


#: An undeclared signal channel above a declared command voltage — the owner's text/TDMS shape.
_UNDECLARED_AND_VOLTAGE = (
    ChannelInfo(index=0, name="signal", raw_units="", kind=ChannelKind.UNKNOWN),
    ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
)
#: Both channels raw Volts off an ADC — the owner's nisin TDMS.
_TWO_RAW_VOLTS = (
    ChannelInfo(index=0, name="ai0", raw_units="V", kind=ChannelKind.RAW_VOLTAGE),
    ChannelInfo(index=1, name="ai1", raw_units="V", kind=ChannelKind.RAW_VOLTAGE),
)


# ================================================================================================
# R0 — the kind the panes are planned from is the kind the user can correct.
# ================================================================================================

def test_without_an_override_the_header_kind_still_rules(qapp, tmp_path):
    """The unchanged case, pinned first: a recording that declares its channels honestly plans
    exactly the panes it always did, and nothing about this reads the calibration."""
    session = _session(_doc(tmp_path, _UNDECLARED_AND_VOLTAGE))
    assert [s.kind for s in plan_panes(session)] == [ChannelKind.VOLTAGE, ChannelKind.UNKNOWN]


def test_a_corrected_channel_takes_the_top_pane(qapp, tmp_path):
    """*** THE OWNER'S OPENING QUESTION. *** Correcting an undeclared channel to pA relabelled
    the axis and left the pane ranked UNKNOWN, i.e. BELOW the command voltage — so the same
    experiment came up current-on-top or current-on-bottom depending only on whether the file
    happened to declare its channels. The units and the kind now come from one place."""
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    session = _session(doc)
    session.set_calibration(doc.doc_id, 0,
                            ChannelCalibration(gain=25.0, target_units="pA",
                                               kind=ChannelKind.CURRENT))
    specs = plan_panes(session)
    assert [s.kind for s in specs] == [ChannelKind.CURRENT, ChannelKind.VOLTAGE]
    assert specs[0].units == "pA", "the axis label and the rank agree, from one answer"


def test_the_override_unmerges_two_channels_of_the_same_kind_and_units(qapp, tmp_path):
    """The other half of the same defect, and the reason ONE change fixes both: ``_kind`` feeds
    the OVERLAY grouping key as well as the rank. On the owner's nisin TDMS both channels are
    (RAW_VOLTAGE, 'V'), so they grouped into a single pane — a -49 pA current and its command
    voltage sharing one y-axis, with nothing in the app able to separate them."""
    doc = _doc(tmp_path, _TWO_RAW_VOLTS)
    session = _session(doc, mode=ViewMode.OVERLAY)
    before = plan_panes(session)
    assert len(before) == 1 and len(before[0].layer_ids) == 2, "premise: one shared pane"

    session.set_calibration(doc.doc_id, 0,
                            ChannelCalibration(gain=1000.0, target_units="pA",
                                               kind=ChannelKind.CURRENT))
    after = plan_panes(session)
    assert [s.kind for s in after] == [ChannelKind.CURRENT, ChannelKind.RAW_VOLTAGE]
    assert all(len(s.layer_ids) == 1 for s in after)


def test_committing_a_kind_change_re_plans_the_live_panes(qapp, tmp_path):
    """A calibration commit reaches the view as ``refresh_channel``, which deliberately does NOT
    rebuild — rebuilding throws away the pane the user is inside. But re-setting curve data
    cannot say "this layer belongs in another pane now", so a commit that changes the PLAN is
    the one that has to. Without this the override was correct in ``plan_panes`` and invisible
    on screen until some unrelated layer toggle happened to rebuild."""
    doc = _doc(tmp_path, _TWO_RAW_VOLTS)
    session = _session(doc, mode=ViewMode.OVERLAY)
    view = TraceView(session)
    view.resize(1000, 600)
    assert len(view.panes()) == 1

    session.set_calibration(doc.doc_id, 0,
                            ChannelCalibration(target_units="pA", kind=ChannelKind.CURRENT))
    assert [p.spec.kind for p in view.panes()] == [ChannelKind.CURRENT, ChannelKind.RAW_VOLTAGE]


def test_a_plain_gain_edit_still_does_not_rebuild(qapp, tmp_path):
    """...and the rule is asked as "did the PLAN move", not "did the user touch the kind box",
    so the ordinary gain edit keeps the no-rebuild guarantee the docstring promises. Pinned by
    identity of the pane objects: a rebuild replaces them."""
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    session = _session(doc)
    view = TraceView(session)
    view.resize(1000, 600)
    panes_before = list(view.panes())

    session.set_calibration(doc.doc_id, 1, ChannelCalibration(gain=0.1, target_units="mV"))
    assert [p for p in view.panes()] == panes_before, "a gain edit must not rebuild the panes"


def test_a_units_change_alone_still_does_not_rebuild(qapp, tmp_path):
    """A pA -> nA edit changes the PLAN (units are part of a PaneSpec) but moves no layer,
    and it must stay on the in-place path: ``refresh_units`` relabels the axis and
    ``remap_y_window`` carries a box-zoomed y-window through the unit change, which a rebuild
    would simply throw away. Phrasing the rebuild rule as "the plan differs" broke exactly
    that — three tests in ``test_s2_viewer_units_zoom_window`` — so the rule is the LAYOUT."""
    doc = _doc(tmp_path, (ChannelInfo(index=0, name="IN15", raw_units="pA",
                                      kind=ChannelKind.CURRENT),))
    session = _session(doc)
    view = TraceView(session)
    view.resize(1000, 600)
    pane = view.panes()[0]

    session.set_calibration(doc.doc_id, 0,
                            ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)
    assert view.panes()[0] is pane, "the pane object itself must survive a units edit"
    assert pane.spec.units == "nA", "...and relabel itself in place"


def test_a_units_change_that_REGROUPS_an_overlay_pane_still_does_not_rebuild(qapp, tmp_path):
    """The single-document test above cannot catch this one, and the "the rule is the LAYOUT"
    phrasing passed it while failing here: in OVERLAY the units are part of the GROUPING key,
    so a pA -> nA edit on one of two grouped recordings genuinely moves the layout — one pane
    becomes two. Rebuilding on that is the very thing the rule exists to prevent, and it also
    silently undid `refresh_units`' mixed-units fallback (the regrouped pane held one curve
    again, so it went on confidently claiming 'pA').

    The rebuild test is the KIND, so this stays on the in-place path: one pane, two curves,
    two units, and an axis that claims neither.
    """
    infos = (ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),)
    doc_a = _doc(tmp_path / "a", infos)
    doc_b = _doc(tmp_path / "b", infos)
    session = _session(doc_a)
    session.add_document(doc_b)
    session.add_layer(doc_b.doc_id, 0)
    session.set_view_mode(ViewMode.OVERLAY)
    view = TraceView(session)
    view.resize(1200, 600)

    pane = view.pane_for_layer(f"{doc_a.doc_id}#0")
    assert pane is view.pane_for_layer(f"{doc_b.doc_id}#0")
    assert pane.plot.getAxis("left").labelText == "pA"

    session.set_calibration(doc_b.doc_id, 0,
                            ChannelCalibration(gain=0.001, target_units="nA"),
                            write_sidecar=False)

    assert view.pane_for_layer(f"{doc_a.doc_id}#0") is pane, "a units edit must not rebuild"
    assert pane.spec.units == "", "a pane holding two units vouches for neither"
    assert pane.plot.getAxis("left").labelText == "current"


def test_a_kind_override_DOES_rebuild_because_nothing_else_can_move_a_layer(qapp, tmp_path):
    """The other side of the same rule. Re-setting curve data cannot express "this layer
    belongs in a different pane now", so the one edit that must rebuild is the override that
    changes what `_kind` answers — here an undeclared channel corrected to a current, which
    both re-ranks its pane and, in OVERLAY, regroups it."""
    doc = _doc(tmp_path, (ChannelInfo(index=0, name="IN13", raw_units="",
                                      kind=ChannelKind.UNKNOWN),))
    session = _session(doc)
    view = TraceView(session)
    view.resize(1000, 600)
    pane = view.panes()[0]
    assert pane.spec.kind is ChannelKind.UNKNOWN

    session.set_calibration(doc.doc_id, 0,
                            ChannelCalibration(target_units="pA", kind=ChannelKind.CURRENT),
                            write_sidecar=False)

    assert view.panes()[0] is not pane, "the kind override is the one edit that rebuilds"
    assert view.panes()[0].spec.kind is ChannelKind.CURRENT


# ================================================================================================
# R0 — the override survives the trip to disk, and old sidecars keep their meaning.
# ================================================================================================

def test_a_kind_only_override_is_worth_a_sidecar(qapp, tmp_path):
    """A user who only says "this channel is a current" — no gain, no units — has still made a
    change. If "differs from default" ignored the kind, that commit wrote nothing and the panes
    came back in the old order on the next open, so the correction had to be re-made on every
    open of the same file."""
    doc = _doc(tmp_path, _TWO_RAW_VOLTS)
    calib = CalibrationSet()
    calib.set("d", 0, ChannelCalibration(kind=ChannelKind.CURRENT))
    assert save_sidecar(doc.meta.path, "d", calib, doc.meta.channels) == "written"

    entry = json.loads(sidecar_path(doc.meta.path).read_text())["channels"]["0"]
    assert entry["treat_as_kind"] == "current"
    # NOT under "kind": that key is the header's own declaration, recorded so the entry can be
    # re-attached after a re-export. Overloading it would send resolve_channel hunting for a
    # channel that never existed.
    assert entry["kind"] == ChannelKind.RAW_VOLTAGE.value


def test_the_override_round_trips_through_the_sidecar(qapp, tmp_path):
    doc = _doc(tmp_path, _TWO_RAW_VOLTS)
    out = CalibrationSet()
    out.set("d", 0, ChannelCalibration(gain=1000.0, target_units="pA",
                                       kind=ChannelKind.CURRENT, note="headstage 1 nA/V"))
    save_sidecar(doc.meta.path, "d", out, doc.meta.channels)

    back = CalibrationSet()
    assert load_sidecar(doc.meta.path, "d2", back, doc.meta.channels) == []
    cal = back.get("d2", 0)
    assert cal.kind is ChannelKind.CURRENT
    assert (cal.gain, cal.target_units, cal.note) == (1000.0, "pA", "headstage 1 nA/V")


def test_a_sidecar_written_before_the_field_existed_reads_as_no_override(qapp, tmp_path):
    """BACKWARD COMPATIBILITY, and the owner's own `.npcal.json` files are in this shape.
    An entry with no ``treat_as_kind`` is not an entry that overrides to nothing — it is an
    entry that says nothing about the kind, and the header must still rule."""
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    sidecar_path(doc.meta.path).write_text(json.dumps({
        "version": 1,
        "channels": {"0": {"gain": 2.0, "offset": 0.0, "target_units": "pA", "note": "old"}},
    }))
    calib = CalibrationSet()
    assert load_sidecar(doc.meta.path, "d", calib, doc.meta.channels) == []
    assert calib.get("d", 0).kind is None
    assert calib.get("d", 0).gain == 2.0


def test_an_empty_kind_is_not_an_override(qapp):
    """`""` and None both mean "the header stands". A build that writes the key unconditionally
    (or a user who empties it by hand) must not thereby create an override to nothing."""
    assert ChannelCalibration(kind="").kind is None
    assert ChannelCalibration(kind=None).kind is None
    assert ChannelCalibration(kind="current").kind is ChannelKind.CURRENT


def test_a_hand_edited_kind_that_names_nothing_is_refused_out_loud(qapp, tmp_path):
    """`.npcal.json` sits next to the user's data and is hand-editable. A kind that names no
    ChannelKind is refused by the value object, which the sidecar reader already reports as a
    per-channel issue rather than raising — a malformed sidecar must never block a load."""
    with pytest.raises(ValueError, match="calibration kind must be one of"):
        ChannelCalibration(kind="Current")          # capitalised: not a ChannelKind value

    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    sidecar_path(doc.meta.path).write_text(json.dumps({
        "version": 2,
        "channels": {"0": {"gain": 1.0, "offset": 0.0, "target_units": "pA", "note": "",
                           "name": "signal", "kind": "unknown", "treat_as_kind": "amps"}},
    }))
    calib = CalibrationSet()
    issues = load_sidecar(doc.meta.path, "d", calib, doc.meta.channels)
    assert issues and "malformed" in issues[0]
    assert calib.entries == {}, "a refused entry installs nothing at all"


def test_the_kind_is_not_part_of_the_cache_fingerprint(qapp):
    """The fingerprint keys the calibrated-array cache and stamps every result's staleness. The
    kind reaches no sample — it decides which pane the channel is drawn in — so including it
    would invalidate every stored result over a pane ordering."""
    plain = ChannelCalibration(gain=2.0, offset=1.0)
    declared = ChannelCalibration(gain=2.0, offset=1.0, kind=ChannelKind.CURRENT)
    assert plain.fingerprint() == declared.fingerprint()
    # ...but it IS part of "has this channel been touched", which is what decides whether a
    # sidecar is written at all.
    assert plain.is_identity is False and ChannelCalibration().is_identity is True
    assert ChannelCalibration(kind=ChannelKind.CURRENT).is_identity is False


# ================================================================================================
# R0 — the Calibration panel is where the answer is given.
# ================================================================================================

def test_the_panel_offers_and_commits_the_override(qapp, tmp_path):
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    session = _session(doc)
    panel = CalibrationPanel(session)
    panel.set_selection(doc.doc_id, 0)

    assert panel._treat_as.itemData(0) is None, "the default answer is 'as the file declares'"
    offered = [panel._treat_as.itemData(i) for i in range(1, panel._treat_as.count())]
    assert offered == [ChannelKind.CURRENT, ChannelKind.VOLTAGE, ChannelKind.CONDUCTANCE]

    panel._treat_as.setCurrentIndex(panel._treat_as.findData(ChannelKind.CURRENT))
    panel._units.setText("pA")
    panel._apply()

    assert session.calibration(doc.doc_id, 0).kind is ChannelKind.CURRENT
    assert [s.kind for s in plan_panes(session)] == [ChannelKind.CURRENT, ChannelKind.VOLTAGE]


def test_the_panel_shows_the_override_the_channel_already_has(qapp, tmp_path):
    """Selecting a channel must READ BACK what is stored on it — a form that shows 'as the file
    declares' over a channel that is overridden invites the user to Apply the default away."""
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    session = _session(doc)
    session.set_calibration(doc.doc_id, 0, ChannelCalibration(kind=ChannelKind.CONDUCTANCE))
    panel = CalibrationPanel(session)
    panel.set_selection(doc.doc_id, 0)
    assert panel._treat_as.currentData() is ChannelKind.CONDUCTANCE


def test_reset_to_default_clears_the_override(qapp, tmp_path):
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    session = _session(doc)
    session.set_calibration(doc.doc_id, 0,
                            ChannelCalibration(target_units="pA", kind=ChannelKind.CURRENT))
    panel = CalibrationPanel(session)
    panel.set_selection(doc.doc_id, 0)
    # Everything is back at its default afterwards, so the commit path wants to DELETE the
    # sidecar and asks first; answer yes without a live modal.
    panel._confirm_sidecar_removal = lambda *_a, **_k: True

    panel._reset()
    assert session.calibration(doc.doc_id, 0).kind is None
    assert panel._treat_as.currentData() is None
    assert [s.kind for s in plan_panes(session)] == [ChannelKind.VOLTAGE, ChannelKind.UNKNOWN]


# ================================================================================================
# The dive-in framing matches the gallery tile's.
# ================================================================================================

def test_the_dive_in_frames_the_event_as_half_the_window(qapp, tmp_path):
    """*** ONE EVENT, ONE SIZE, ON BOTH SURFACES. *** ``pad_frac`` defaulted to 1.5 — a
    4x-span window, the event a QUARTER of the pane — while the gallery tile has always used
    0.5, the event HALF the tile. Both zoom_to_span call sites pass only ``min_pad``, so the
    default is what the user actually got, and clicking a gallery event redrew it at a quarter
    of the size it had just been inspected at."""
    from nanoscope.gui.event_gallery import _PAD_FRACTION

    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    session = _session(doc)
    view = TraceView(session)
    view.resize(1000, 600)

    assert TraceView.ZOOM_PAD_FRACTION == _PAD_FRACTION, (
        "the trace's dive-in pad and the gallery tile's pad are one number")

    x0, x1 = 0.20, 0.24                      # a 40 ms "event" inside a 0.8 s recording
    view.zoom_to_span(x0, x1)
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    assert (x1 - x0) / (hi - lo) == pytest.approx(0.5, rel=1e-6)
    assert 0.5 * (lo + hi) == pytest.approx(0.5 * (x0 + x1)), "and centred on the event"


def test_an_explicit_pad_fraction_still_wins(qapp, tmp_path):
    """The default moved; the parameter did not go away. Zero is a number, not a missing
    argument — a caller asking for edge-to-edge must get edge-to-edge."""
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    view = TraceView(_session(doc))
    view.resize(1000, 600)
    view.zoom_to_span(0.20, 0.24, pad_frac=0.0)
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    assert (lo, hi) == pytest.approx((0.20, 0.24))


def test_the_minimum_pad_still_floors_a_one_sample_span(qapp, tmp_path):
    """The floor is what keeps a degenerate span from zooming to nothing, and it is unaffected
    by the smaller proportional pad — both call sites rely on it."""
    doc = _doc(tmp_path, _UNDECLARED_AND_VOLTAGE)
    view = TraceView(_session(doc))
    view.resize(1000, 600)
    view.zoom_to_span(0.30, 0.30, min_pad=0.01)
    lo, hi = view.anchor_plot().vb.viewRange()[0]
    assert (lo, hi) == pytest.approx((0.29, 0.31))

# ================================================================================================
# the pyqtgraph context menu is OFF on the trace panes — Log X would break the level
# overlay's pixel floor (linear-units geometry under a log transform → sub-pixel again)
# ================================================================================================

def test_trace_panes_have_no_pyqtgraph_context_menu(qapp, tmp_path):
    """The default menu's Transforms ▸ Log X flips the pane into log mode, where
    StepOverlayItem's ~3-px floor (computed in linear data units) collapses to
    sub-pixel — the exact invisibility the floor exists to prevent — and Log Y on a
    negative-current trace blanks the curve outright. The navigator and the subpop
    embedding already run menu-less; the trace panes must too."""
    doc = _doc(tmp_path, _TWO_RAW_VOLTS)
    session = _session(doc, mode=ViewMode.STACKED)
    view = TraceView(session)
    view.resize(1000, 600)
    assert view.panes(), "fixture regression: no panes to check"
    for pane in view.panes():
        assert not pane.plot.vb.menuEnabled(), (
            "a trace pane still offers the pyqtgraph context menu")
