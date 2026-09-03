"""One plot source PER DETECTOR — the fix for "the Plots window drew a table the
Analysis panel was not showing".

``ResultStore`` has always been keyed ``(doc_id, channel, module_id)`` and genuinely
holds a channel's detections side by side. ``discover_plot_sources`` did not: it read
:data:`~nanoscope.core.sources.DETECTION_MODULE_IDS` BEST-FIRST and kept the first hit
per ``(doc, channel)``, so every other stored detection was unreachable from every
window built on that list.

MEASURED on the owner's -1000 mV recording (749k samples) on 2026-09-02, driven
through the real MainWindow at 1512x900: Schmitt (110 events) and CUSUM (250) both in
the store, the Analysis panel showing CUSUM's 250 rows, and the Plots window offering
ONE source -- Schmitt's -- ticked, drawn, captioned "(n = 110)", flagged
``current=True``, and labelled with the bare filename. A figure exported from that
window is of a table the user was not looking at, and nothing on it says so.
``detect.cusum`` was not even in the tuple, so a CUSUM-only recording produced ZERO
plot sources and a greyed "Run detection first".

These pin the four properties the fix rests on:

  * every stored detection on a channel is offered, in ``module_ids`` order;
  * the identity key names the DETECTOR, while the default detector keeps the bare
    channel key that every stored colour and label override was written against;
  * two sources of one recording are named apart by their detector, and a recording
    with one detection is still named by itself;
  * the ACTIVE detection -- the ``(doc, channel, module_id)`` triple, not the
    ``(doc, channel)`` pair -- is the one flagged ``current``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (Session is a QObject). ---------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd

from nanoscope.analysis.events import SCHEMA, EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ColourRegistry, colour_key
from nanoscope.core.segment import build_segment_document, derive_segment_uid, persistent_key
from nanoscope.core.sources import (
    MANUAL_SOURCE_KEY,
    MANUAL_SOURCE_LABEL,
    detector_name,
    discover_plot_sources,
    source_key,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
N = 20_000


# ================================================================================================
# builders
# ================================================================================================

def _events_df(n: int = 12) -> pd.DataFrame:
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
    cols["dwell_s"] = np.full(n, 3e-4)
    cols["depth_pA"] = np.full(n, 30.0)
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


def _result(doc, channel: int, module_id: str, n: int) -> AnalysisResult:
    return AnalysisResult(
        module_id=module_id, module_version="1", doc_id=doc.doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0),
        payload=EventSet(_events_df(n), {"min_dwell_s": 5e-5}),
    )


class _Results:
    """A stand-in for ``Session.results.get`` — keyed (doc_id, channel, module_id), as
    the real store has always been."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, int, str], AnalysisResult] = {}

    def add(self, res: AnalysisResult) -> None:
        self._by_key[(res.doc_id, res.channel, res.module_id)] = res

    def get(self, doc_id: str, channel: int, module_id: str):
        return self._by_key.get((doc_id, channel, module_id))


# ================================================================================================
# every stored detection is offered
# ================================================================================================

def test_every_detector_that_has_run_on_a_channel_becomes_its_own_source(tmp_path):
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.cusum", 250))
    res.add(_result(a, 0, "detect.keras", 80))

    sources = discover_plot_sources([a], res.get)

    assert [s.module_id for s in sources] == [
        "detect.schmitt", "detect.cusum", "detect.keras"], (
        "one source per stored detection, in module_ids order")
    assert [s.n_rows for s in sources] == [110, 250, 80]


def test_a_cusum_only_recording_is_offered_at_all(tmp_path):
    """The measured symptom of ``detect.cusum`` missing from the tuple: 250 events on
    screen, zero plot sources, and Analysis > Plots greyed with "Run detection first"."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.cusum", 250))

    sources = discover_plot_sources([a], res.get)

    assert [s.module_id for s in sources] == ["detect.cusum"]
    assert sources[0].n_rows == 250


def test_a_detector_that_found_nothing_is_not_offered_as_an_empty_source(tmp_path):
    """The empty-frame rule is per DETECTOR now, not per channel: an empty CUSUM run
    must not take the channel's place, and must not add a curve with no points."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.cusum", 0))

    sources = discover_plot_sources([a], res.get)

    assert [s.module_id for s in sources] == ["detect.schmitt"]


# ================================================================================================
# the key discipline
# ================================================================================================

def test_the_default_detector_keeps_the_bare_channel_key_and_the_others_name_themselves(tmp_path):
    """A key is what a colour, a legend rename and a saved template are stored against,
    so the Schmitt source -- the one every such key already refers to -- must not move,
    and the second detector must not collide with it."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.cusum", 250))

    schmitt, cusum = discover_plot_sources([a], res.get)
    bare = colour_key(persistent_key(a), 0)

    assert schmitt.key == bare, "the default detector keeps the pre-existing key"
    assert cusum.key == f"{bare}@detect.cusum"
    assert cusum.key != schmitt.key, "two detections of one channel are two identities"
    assert source_key(persistent_key(a), 0, "detect.cusum") == cusum.key


def test_a_detectors_key_does_not_move_when_a_sibling_detection_appears(tmp_path):
    """The anchor is the DEFAULT detector, never "whichever was found first", so
    running a second detector does not repaint or rename the first."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.keras", 80))
    alone = discover_plot_sources([a], res.get)[0].key

    res.add(_result(a, 0, "detect.schmitt", 110))
    joined = {s.module_id: s.key for s in discover_plot_sources([a], res.get)}

    assert joined["detect.keras"] == alone
    assert joined["detect.schmitt"] == colour_key(persistent_key(a), 0)


def test_two_detectors_on_one_channel_draw_in_two_colours(tmp_path):
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.cusum", 250))
    reg = ColourRegistry()

    inks = [s.colour for s in discover_plot_sources([a], res.get, colours=reg)]

    assert inks[0] != inks[1], "two curves of one recording cannot be one ink"
    # ...and stable: asking again gives the same two.
    assert [s.colour for s in discover_plot_sources([a], res.get, colours=reg)] == inks


# ================================================================================================
# the labels
# ================================================================================================

def test_two_detectors_on_one_recording_are_named_apart_by_their_detector(tmp_path):
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.keras", 80))

    labels = [s.label for s in discover_plot_sources([a], res.get)]

    assert labels == ["Schmitt · pore_a.dat", "Keras · pore_a.dat"]
    assert detector_name("detect.cusum") == "CUSUM"


def test_a_recording_with_one_detection_is_still_named_by_itself(tmp_path):
    """The detector is named where it TELLS TWO SOURCES APART and nowhere else; a
    legend that says "Schmitt ·" in front of every entry has told the reader nothing."""
    a, b = _doc(tmp_path, "pore_a.dat"), _doc(tmp_path, "pore_b.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(b, 0, "detect.cusum", 250))

    assert [s.label for s in discover_plot_sources([a, b], res.get)] == [
        "pore_a.dat", "pore_b.dat"]


def test_two_detectors_on_two_channels_are_told_apart_by_both(tmp_path):
    """The detector pass runs BEFORE the channel pass and composes with it."""
    a = _doc(tmp_path, "pore_a.dat", channels=2)
    res = _Results()
    for ch in (0, 1):
        res.add(_result(a, ch, "detect.schmitt", 110))
        res.add(_result(a, ch, "detect.cusum", 250))

    labels = [s.label for s in discover_plot_sources([a], res.get)]

    assert len(set(labels)) == 4, f"four sources, four names: {labels}"
    assert all("Schmitt · " in l or "CUSUM · " in l for l in labels)
    assert any("current 1" in l for l in labels)


# ================================================================================================
# which one is "what the user is looking at"
# ================================================================================================

def test_the_active_detection_is_the_one_flagged_current(tmp_path):
    """A ``(doc, channel)`` pair no longer identifies a table, so the caller names the
    module too -- otherwise the window pre-selects a detector the panel is not showing,
    which is the whole defect."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.cusum", 250))

    sources = discover_plot_sources([a], res.get,
                                    current=(a.doc_id, 0, "detect.cusum"))

    assert [s.current for s in sources] == [False, True]
    assert next(s for s in sources if s.current).n_rows == 250


def test_a_caller_that_names_no_detector_flags_exactly_one_source(tmp_path):
    """The two-element form is a GUESS (it names the channel's first detection), but it
    must never flag two tables as "the one on screen" -- a window reading the flag would
    then pick by dict order."""
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))
    res.add(_result(a, 0, "detect.cusum", 250))

    sources = discover_plot_sources([a], res.get, current=(a.doc_id, 0))

    assert [s.current for s in sources] == [True, False]


# ================================================================================================
# the things per-detector discovery must NOT have broken
# ================================================================================================

def test_the_manual_measurement_table_is_still_a_source_and_names_no_detector(tmp_path):
    a = _doc(tmp_path, "pore_a.dat")
    res = _Results()
    res.add(_result(a, 0, "detect.schmitt", 110))

    sources = discover_plot_sources([a], res.get, manual_df=_events_df(4))

    manual = sources[-1]
    assert manual.key == MANUAL_SOURCE_KEY
    assert manual.label == MANUAL_SOURCE_LABEL
    assert manual.kind == "manual"
    assert manual.module_id is None and manual.detector_label == ""


def test_a_segment_named_after_its_parent_still_gets_its_own_key(tmp_path):
    parent = _doc(tmp_path, "baseline.dat")
    uid = derive_segment_uid(persistent_key(parent), 100, 9_000)
    seg = build_segment_document(parent, 100, 9_000, name=parent.display_name,
                                 segment_uid=uid)
    res = _Results()
    res.add(_result(parent, 0, "detect.schmitt", 110))
    res.add(_result(seg, 0, "detect.schmitt", 40))

    a, b = discover_plot_sources([parent, seg], res.get)

    assert a.key != b.key, "the pre-existing collision guard still holds"
    assert b.key == source_key(uid, 0, "detect.schmitt")
    assert a.label != b.label, "a segment named after its parent is still told apart"
    # ...and per-detector keys stack ON TOP of that, they do not replace it.
    res.add(_result(seg, 0, "detect.cusum", 12))
    keys = {s.key for s in discover_plot_sources([parent, seg], res.get)}
    assert len(keys) == 3
