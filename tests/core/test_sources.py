"""The multi-source model — every per-event table an analysis window may overlay.

Before :mod:`nanoscope.core.sources` the Plots window was handed exactly TWO
frames: the CURRENT detection and the manual-measurement table. Both comparisons
a nanopore paper is built out of — *this pore vs that pore* and *this segment vs
that segment* — were therefore structurally impossible, not merely inconvenient.

These tests pin the discovery rule (which is PURE, so it needs no window):

  * every open document with a stored detection is offered, segments included;
  * a document with no detection, or one whose detection is empty, is not;
  * identity is :func:`~nanoscope.core.segment.persistent_key`, so a segment named
    IDENTICALLY to its parent gets its own key instead of clobbering it — the
    collision that would silently turn a two-source overlay into one curve;
  * colliding display names are still made distinguishable for the legend;
  * each source carries ITS OWN provenance (a per-source fit must resolve its own
    ``t_min``) and its own calibration fingerprint;
  * :func:`~nanoscope.core.sources.gain_comparability` tells the truth about when
    absolute pA may be compared across recordings.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (Session is a QObject). ---------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd

from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import build_segment_document, derive_segment_uid, persistent_key
from nanoscope.core.sources import (
    MANUAL_SOURCE_KEY,
    SOURCE_COLOURS,
    PlotSource,
    discover_plot_sources,
    gain_comparability,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
N = 20_000
MODULE = "detect.schmitt"


# ================================================================================================
# builders
# ================================================================================================

def _events_df(n: int = 12, *, dwell: float = 3e-4) -> pd.DataFrame:
    """A full 20-column EventSet frame (the schema is frozen; EventSet enforces it)."""
    from nanoscope.analysis.events import SCHEMA

    rng = np.random.default_rng(0)
    cols = {}
    for name, (dtype, _unit, _doc) in SCHEMA.items():
        if dtype == "int64":
            cols[name] = np.arange(n, dtype=np.int64)
        elif dtype == "bool":
            cols[name] = np.zeros(n, dtype=bool)
        elif dtype == "string":
            cols[name] = pd.Series([""] * n, dtype="string")
        else:
            cols[name] = np.zeros(n, dtype=np.float64)
    cols["dwell_s"] = np.abs(rng.normal(dwell, dwell / 5, n))
    cols["frac_depth"] = np.abs(rng.normal(0.25, 0.02, n))
    cols["depth_pA"] = np.abs(rng.normal(30.0, 3.0, n))
    cols["baseline_pA"] = np.full(n, 120.0)
    cols["accepted"] = np.ones(n, dtype=bool)
    return pd.DataFrame(cols)


def _doc(tmp_path: Path, name: str, *, channels: int = 1) -> TraceDocument:
    infos = tuple(
        ChannelInfo(index=i, name=f"current {i}" if i else "current",
                    raw_units="pA", kind=ChannelKind.CURRENT)
        for i in range(channels)
    )
    arrs = tuple(np.zeros(N, dtype=np.float32) for _ in range(channels))
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=infos,
    )
    return TraceDocument(meta=meta, channels=arrs)


def _result(doc, channel: int, df: pd.DataFrame, prov: dict,
            warnings: tuple[str, ...] = ()) -> AnalysisResult:
    return AnalysisResult(
        module_id=MODULE, module_version="1", doc_id=doc.doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=EventSet(df, prov),
        warnings=warnings,
    )


class _Results:
    """A stand-in for ``Session.results.get`` — the only session surface discovery uses."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, int, str], AnalysisResult] = {}

    def add(self, res: AnalysisResult) -> None:
        self._by_key[(res.doc_id, res.channel, res.module_id)] = res

    def get(self, doc_id: str, channel: int, module_id: str):
        return self._by_key.get((doc_id, channel, module_id))


# ================================================================================================
# discovery across documents
# ================================================================================================

def test_every_open_document_with_a_detection_is_offered(tmp_path):
    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(10), {"min_dwell_s": 5e-5}))
    res.add(_result(b, 0, _events_df(20), {"min_dwell_s": 1e-4}))

    sources = discover_plot_sources([a, b], res.get)

    assert [s.label for s in sources] == ["pore_a.dat", "pore_b.dat"]
    assert [s.n_rows for s in sources] == [10, 20]
    # Distinct Okabe-Ito inks, assigned in discovery order.
    assert [s.colour for s in sources] == [SOURCE_COLOURS[0], SOURCE_COLOURS[1]]
    assert len({s.colour for s in sources}) == 2


def test_a_document_without_a_detection_is_not_offered(tmp_path):
    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {}))

    assert [s.label for s in discover_plot_sources([a, b], res.get)] == ["pore_a.dat"]


def test_an_empty_detection_is_not_offered(tmp_path):
    """A detection that found nothing is a dead tab, not a source."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(0), {"min_dwell_s": 5e-5}))

    assert discover_plot_sources([a], res.get) == []


def test_manual_measurements_are_appended_last_and_only_when_non_empty(tmp_path):
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {}))
    meas = pd.DataFrame({"dwell_s": [1e-4, 2e-4]})

    with_meas = discover_plot_sources([a], res.get, manual_df=meas)
    assert [s.label for s in with_meas] == ["pore_a.dat", "Manual measurements"]
    assert with_meas[-1].key == MANUAL_SOURCE_KEY
    assert with_meas[-1].kind == "manual"
    assert with_meas[-1].provenance == {}       # hand measurements have no detector

    assert len(discover_plot_sources([a], res.get, manual_df=meas.iloc[:0])) == 1
    assert len(discover_plot_sources([a], res.get, manual_df=None)) == 1


# ================================================================================================
# segments: the collision the persistent key exists to prevent
# ================================================================================================

def _segment_of(parent: TraceDocument, i0: int, i1: int, name: str):
    uid = derive_segment_uid(persistent_key(parent), i0, i1)
    return build_segment_document(parent, i0, i1, name=name, segment_uid=uid)


def test_segments_are_offered_like_any_other_document(tmp_path):
    parent = _doc(tmp_path, "pore_a.dat")
    seg = _segment_of(parent, 0, 5_000, "first half")
    res = _Results()
    res.add(_result(parent, 0, _events_df(30), {"min_dwell_s": 5e-5}))
    res.add(_result(seg, 0, _events_df(8), {"min_dwell_s": 5e-5}))

    sources = discover_plot_sources([parent, seg], res.get)
    assert [s.label for s in sources] == ["pore_a.dat", "first half"]
    assert [s.is_segment for s in sources] == [False, True]


def test_a_segment_named_exactly_like_its_parent_does_not_collide(tmp_path):
    """THE collision. Nothing stops a user naming a crop after the recording it came
    from, and the display name is what a legend shows — so if identity were the name,
    two sources would share one key and the overlay would silently draw one curve."""
    parent = _doc(tmp_path, "baseline.dat")
    seg = _segment_of(parent, 100, 9_000, parent.display_name)   # identical display name
    assert seg.display_name == parent.display_name

    res = _Results()
    res.add(_result(parent, 0, _events_df(30), {"min_dwell_s": 5e-5}))
    res.add(_result(seg, 0, _events_df(7), {"min_dwell_s": 2e-5}))

    sources = discover_plot_sources([parent, seg], res.get)

    assert len(sources) == 2
    # Keys are derived from the PERSISTENT key, so they differ...
    assert sources[0].key != sources[1].key
    assert sources[0].doc_key == persistent_key(parent)
    assert sources[1].doc_key == persistent_key(seg)
    assert len({s.key for s in sources}) == 2
    # ...and the labels are still made distinguishable, because a legend reading
    # "baseline.dat / baseline.dat" names nothing.
    assert len({s.label for s in sources}) == 2
    assert sources[1].display_name == "baseline.dat"     # the raw name is preserved
    # Each keeps its own rows and its own provenance.
    assert sources[0].n_rows == 30 and sources[1].n_rows == 7
    assert sources[0].provenance["min_dwell_s"] == 5e-5
    assert sources[1].provenance["min_dwell_s"] == 2e-5


def test_two_files_with_the_same_basename_also_get_distinct_keys(tmp_path):
    """Same trap, different cause: two pores' files in different folders."""
    d1, d2 = tmp_path / "run1", tmp_path / "run2"
    d1.mkdir()
    d2.mkdir()
    a, b = _doc(d1, "pore.dat"), _doc(d2, "pore.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(5), {}))
    res.add(_result(b, 0, _events_df(6), {}))

    sources = discover_plot_sources([a, b], res.get)
    assert len({s.key for s in sources}) == 2
    assert len({s.label for s in sources}) == 2


def test_two_channels_of_one_document_are_separate_sources_labelled_by_channel(tmp_path):
    a = _doc(tmp_path, "pore_a.dat", channels=2)
    res = _Results()
    res.add(_result(a, 0, _events_df(4), {}))
    res.add(_result(a, 1, _events_df(9), {}))

    sources = discover_plot_sources([a], res.get)
    assert len(sources) == 2
    assert len({s.key for s in sources}) == 2
    assert all("current" in s.label for s in sources)   # disambiguated by channel name
    assert [s.channel for s in sources] == [0, 1]


# ================================================================================================
# provenance + the current selection
# ================================================================================================

def test_each_source_carries_its_own_provenance(tmp_path):
    """The regression guard's foundation: a per-source fit resolves t_min from the
    source's own detector, so the frames and provenances must not be separable."""
    a, b = _doc(tmp_path, "fast.dat"), _doc(tmp_path, "slow.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {"min_dwell_s": 2e-5, "fs": FS}))
    res.add(_result(b, 0, _events_df(), {"min_dwell_s": 4e-4, "fs": FS}))

    sa, sb = discover_plot_sources([a, b], res.get)
    assert sa.provenance["min_dwell_s"] == 2e-5
    assert sb.provenance["min_dwell_s"] == 4e-4
    # .events() is the (frame, provenance) pair analysis.modules takes.
    frame, prov = sb.events()
    assert prov["min_dwell_s"] == 4e-4
    assert frame is sb.df


def test_the_active_selection_is_flagged_current(tmp_path):
    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {}))
    res.add(_result(b, 0, _events_df(), {}))

    sources = discover_plot_sources([a, b], res.get, current=(b.doc_id, 0))
    assert [s.current for s in sources] == [False, True]


def test_a_channel_that_has_run_two_detectors_is_offered_twice(tmp_path):
    """REPLACES ``test_the_preferred_detector_wins_when_a_channel_has_two``, which
    pinned the first-match rule this file's discovery no longer uses.

    That rule was the bug: ``ResultStore`` is keyed (doc_id, channel, module_id) and
    holds a channel's detections side by side, so keeping only the best-first hit made
    every other one unreachable from every window built on this list. Measured on the
    owner's -1000 mV recording on 2026-09-02 -- Schmitt 110 events and CUSUM 250 both
    stored, the Analysis panel showing CUSUM, and the Plots window offering ONE source,
    drawing Schmitt's 110 rows under the bare filename. ``module_ids`` is now the
    ORDER the sources are offered in, not a preference among them.
    """
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    schmitt = _result(a, 0, _events_df(11), {"min_dwell_s": 5e-5})
    ml = AnalysisResult(module_id="detect.keras", module_version="1", doc_id=a.doc_id,
                        channel=0, params={}, calibration_fingerprint=(1.0, 0.0),
                        payload=EventSet(_events_df(3), {}))
    res.add(schmitt)
    res.add(ml)

    sources = discover_plot_sources([a], res.get)
    assert [s.module_id for s in sources] == ["detect.schmitt", "detect.keras"]
    assert [s.n_rows for s in sources] == [11, 3]
    # Two sources of ONE recording, so the legend has to name the detector.
    assert [s.label for s in sources] == ["Schmitt · pore_a.dat", "Keras · pore_a.dat"]

    # ...and the caller may ask for the other order.
    sources = discover_plot_sources([a], res.get,
                                    module_ids=("detect.keras", "detect.schmitt"))
    assert sources[0].module_id == "detect.keras"


# ================================================================================================
# gain comparability — absolute pA is not comparable across uncalibrated recordings
# ================================================================================================

def _src(key: str, fp) -> PlotSource:
    return PlotSource(key=key, label=key, df=_events_df(3), calibration_fingerprint=fp)


def test_one_source_is_always_self_comparable():
    ok, why = gain_comparability([_src("a", (1.0, 0.0))])
    assert ok and why == ""


def test_identical_identity_gain_is_NOT_comparable():
    """Two recordings that share the identity calibration share nothing: neither has
    been calibrated, so each pA number is its own amplifier's raw scale."""
    ok, why = gain_comparability([_src("a", (1.0, 0.0)), _src("b", (1.0, 0.0))])
    assert not ok
    assert "UNCALIBRATED" in why
    assert "frac_depth" in why and "dwell_s" in why   # says what IS comparable


def test_different_gains_are_not_comparable():
    ok, why = gain_comparability([_src("a", (1.0, 0.0)), _src("b", (2.5, 0.0))])
    assert not ok
    assert "do NOT share a current calibration" in why


def test_a_shared_real_calibration_is_comparable():
    ok, why = gain_comparability([_src("a", (2.5, -3.0)), _src("b", (2.5, -3.0))])
    assert ok and why == ""


def test_the_manual_table_does_not_trigger_the_gain_warning():
    """Only detections carry a recording's gain; the manual table is not a second pore."""
    a = _src("a", (1.0, 0.0))
    manual = PlotSource(key="m", label="m", df=_events_df(2), kind="manual")
    ok, _why = gain_comparability([a, manual])
    assert ok


def test_calibration_fingerprint_is_read_per_source(tmp_path):
    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {}))
    res.add(_result(b, 0, _events_df(), {}))
    fps = {a.doc_id: (1.0, 0.0), b.doc_id: (3.0, 1.0)}

    sources = discover_plot_sources([a, b], res.get,
                                    calibration_fingerprint=lambda d, _c: fps[d])
    assert [s.calibration_fingerprint for s in sources] == [(1.0, 0.0), (3.0, 1.0)]
    ok, why = gain_comparability(sources)
    assert not ok and "do NOT share a current calibration" in why


def test_a_raising_calibration_lookup_does_not_break_discovery(tmp_path):
    """A missing calibration entry must cost the window a warning, not the source."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {}))

    def boom(_d, _c):
        raise KeyError("no such calibration")

    sources = discover_plot_sources([a], res.get, calibration_fingerprint=boom)
    assert len(sources) == 1 and sources[0].calibration_fingerprint is None


# ================================================================================================
# purity
# ================================================================================================

def test_the_module_is_qt_free():
    """The discovery rule is a value-object layer: importable, and testable, with no
    window in sight. (It is imported by MainWindow, never the other way round.)"""
    import ast
    import pathlib

    import nanoscope.core.sources as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert not any("Qt" in n or "PySide" in n or "pyqtgraph" in n for n in names), names
    assert not any(n.startswith("nanoscope.gui") for n in names), names


# ================================================================================================
# the detection's own caveats travel with the source
# ================================================================================================

def test_a_detections_warnings_travel_onto_its_source(tmp_path):
    """REGRESSION. ``AnalysisResult.warnings`` is what the Summary panel renders — the
    "measured at the unprobed default baseline" class of caveat. A plotting surface
    that cannot see it shows the same number with fewer caveats than the table it came
    from, so the source carries the tuple WHOLE and unfiltered."""
    doc = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    caveat = "Detected at the UNPROBED default baseline window; run Preprocess ▸ probe."
    res.add(_result(doc, 0, _events_df(), {}, warnings=(caveat, "second caveat")))

    src = discover_plot_sources([doc], res.get)[0]
    assert src.warnings == (caveat, "second caveat")


def test_a_detection_without_warnings_carries_an_empty_tuple(tmp_path):
    doc = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(doc, 0, _events_df(), {}))
    assert discover_plot_sources([doc], res.get)[0].warnings == ()


def test_the_manual_table_carries_no_detection_warnings(tmp_path):
    doc = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(doc, 0, _events_df(), {}, warnings=("a caveat",)))
    sources = discover_plot_sources([doc], res.get, manual_df=_events_df(3))
    assert sources[-1].kind == "manual" and sources[-1].warnings == ()


# ================================================================================================
# D1 — colour is keyed by IDENTITY, not by position
# ================================================================================================

def test_the_source_palette_has_not_drifted_from_okabe_ito():
    """``SOURCE_COLOURS`` is a hand-written COPY of ``plot.style.OKABE_ITO``.

    The copy is deliberate — ``core`` must not import ``plot.style``, which pulls in
    matplotlib — but it is the one palette copy in the tree that no guard covered.
    ``tests/gui/test_subpop_views.py::test_neither_caller_hard_codes_a_categorical_palette``
    reads the AST of ``gui.subpop_window`` and ``gui.plots_panel`` only; this tuple
    sits an import level BELOW the palette it copies and has never been in its scope,
    so the two could have drifted silently and given the Plots panel a different blue
    from every matplotlib figure exported beside it.

    Pinned BY VALUE rather than by extending the AST guard: the AST guard would forbid
    the copy, and the copy is the point. What must never happen is disagreement.
    """
    from nanoscope.plot.style import OKABE_ITO

    assert SOURCE_COLOURS == OKABE_ITO, (
        "core.sources.SOURCE_COLOURS has drifted from plot.style.OKABE_ITO; they are the "
        "same palette and one of them was edited alone")


def test_a_source_keeps_its_colour_when_a_neighbour_is_closed(tmp_path):
    """THE DEFECT D1 NAMES. Positional inks repaint every source when the set changes.

    Open three recordings, close the FIRST, and under the old rule the remaining two
    each shifted one slot down the palette — so "the Zn run is the green one" became
    false without the user touching anything. Keyed by identity, they do not move.
    """
    from nanoscope.core.layer import ColourRegistry

    a, b, c = (_doc(tmp_path, f"pore_{n}.dat") for n in "abc")
    res = _Results()
    for d in (a, b, c):
        res.add(_result(d, 0, _events_df(), {"min_dwell_s": 5e-5}))

    reg = ColourRegistry()
    before = {s.key: s.colour for s in discover_plot_sources([a, b, c], res.get, colours=reg)}
    after = {s.key: s.colour for s in discover_plot_sources([b, c], res.get, colours=reg)}

    assert len(set(before.values())) == 3
    assert all(after[k] == before[k] for k in after), (
        f"closing pore_a repainted the survivors: {before} -> {after}")

    # ...and the positional path is what it always was, which is why it is the fallback.
    moved = {s.key: s.colour for s in discover_plot_sources([b, c], res.get)}
    assert moved != after


def test_the_manual_table_keeps_its_colour_when_a_recording_is_opened(tmp_path):
    """The manual table's positional index was ``len(sources)``, so every recording
    opened moved the hand-measured points to a new ink. It has an identity now."""
    from nanoscope.core.layer import ColourRegistry

    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    for d in (a, b):
        res.add(_result(d, 0, _events_df(), {}))
    meas = pd.DataFrame({"dwell_s": [1e-4, 2e-4]})

    reg = ColourRegistry()
    one = discover_plot_sources([a], res.get, manual_df=meas, colours=reg)[-1]
    two = discover_plot_sources([a, b], res.get, manual_df=meas, colours=reg)[-1]
    assert one.key == two.key == MANUAL_SOURCE_KEY
    assert one.colour == two.colour


def test_a_hand_picked_colour_wins_over_the_palette(tmp_path):
    """"Same colour for Zn data in the inspector and the population plots" is exactly
    this: the user's ink is returned whatever palette the asking surface uses."""
    from nanoscope.core.layer import ColourRegistry, colour_key

    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, _events_df(), {}))

    reg = ColourRegistry()
    key = colour_key(persistent_key(a), 0)
    reg.set_ink(key, "#FF00FF")

    src = discover_plot_sources([a], res.get, colours=reg)[0]
    assert src.key == key, "PlotSource.key must BE the registry key, or no surface can ask"
    assert src.colour == "#FF00FF"


def test_session_layer_and_plot_source_share_one_user_ink(tmp_path):
    """The integration seam: Session writes the persistent registry and discovery reads it.

    Default trace/plot inks deliberately differ because their backgrounds differ; a USER
    ink must be byte-for-byte identical on both surfaces and after either refreshes.
    """
    from nanoscope.core.layer import colour_key
    from nanoscope.core.session import Session

    doc = _doc(tmp_path, "pore_a.dat")
    session = Session()
    session.add_document(doc)
    layer = session.add_layer(doc.doc_id, 0)
    session.add_result(_result(doc, 0, _events_df(), {}))

    key = colour_key(persistent_key(doc), 0)
    assert session.colours.known(key)
    session.set_layer_color(layer.layer_id, "#FF00FF")

    source = discover_plot_sources(
        session.documents(), session.results.get, colours=session.colours)[0]
    assert source.key == key
    assert layer.color == source.colour == "#FF00FF"


def test_discovery_without_a_registry_still_colours_every_source(tmp_path):
    """The pure callers have no project to key against and must not be refused a colour."""
    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    for d in (a, b):
        res.add(_result(d, 0, _events_df(), {}))
    assert [s.colour for s in discover_plot_sources([a, b], res.get)] == [
        SOURCE_COLOURS[0], SOURCE_COLOURS[1]]
