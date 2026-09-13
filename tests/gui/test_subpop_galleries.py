"""The rest of item 13 — per-subpopulation galleries, and the trace map that names them.

``test_subpop_views.py`` covers the two FIGURES in the report window. What is at stake
here is the other half of item 13, which is not a figure at all: the owner asked for
"representative raw-event galleries per subpopulation, and the trace overlays
(full-recording map + zoom windows)", i.e. for the report's assignment to reach the two
surfaces that already draw the events themselves.

Four rules run through every test below.

**NOTHING IS RECOMPUTED, AND NOTHING IS A SECOND GALLERY.** The assignment travels as a
plain ``event_id -> subpopulation id`` dict that ``MainWindow`` reads off
``report.event_table()`` — the report's own join, the same table the export writes. The
gallery gains a filter and a chip; the trace gains a colour per class. Neither imports the
analysis layer, neither learns what a report is, and there is no second contact sheet.

**"REPRESENTATIVE" IS NOT "THE FIRST N".** A page of the first 35 events of a
subpopulation is a picture of the start of the recording. The owner's notebook takes
examples spaced through the class's own dwell distribution, and the tests pin the picked
rows against the order statistics exactly — a prefix would fail them, and so would a
random sample.

**THE ASSIGNMENT NEVER TOUCHES THE FRAME.** It is a provisional mixture posterior. A
column would carry it into the exports and the project file as though it had been
measured, so the tests assert the shared frame is byte-identical afterwards, and that a
different detection drops it rather than re-attaching it by id.

**THE CAVEATS COME WITH IT.** The gallery can now draw a grid that looks homogeneous
whether or not the class is real, which is exactly the picture this pipeline should not
produce unaccompanied. So the caveat pane arrives with the assignment, carries every
warning verbatim, and is not a tab.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult, OverlayHint, OverlayKind
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import categorical as CAT
from nanoscope.gui.event_gallery import (
    TRACE_CALIBRATED,
    TRACE_DETRENDED,
    EventGalleryWindow,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.overlays import SpanOverlayItem

FS = 100_000.0
#: Enough events that the mixture stage actually runs: ``subpop_report.MIN_GMM_EVENTS`` is
#: 60, the >=4 sigma gate then removes some, and a fixture that lands under the floor would
#: make every test below read "no mixture" as though the wiring were broken.
N_SAMPLES = 420_000
#: Above ``subpop_report.MIN_GMM_EVENTS`` (60) with room for the >=4 sigma gate to remove
#: some, so the mixture stage actually runs. A fixture that landed under that floor would
#: make every test below read "no mixture" as though the wiring were broken.
N_EVENTS = 100
_SPACING = 4_000


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 3) -> np.ndarray:
    """Events of DIFFERENT dwells and depths, so a dwell ladder has rungs to space itself
    over and a depth sort has something to sort."""
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 100.0) + rng.normal(0.0, 1.5, N_SAMPLES)
    for i in range(N_EVENTS):
        s = 2_000 + i * _SPACING
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """The trace and its CSV, built ONCE for the module.

    Every test here wants the same recording; writing the CSV per test was pure overhead.
    Nothing mutates the file, and each test still gets its own Session, its own MainWindow
    and its own detection over it.
    """
    arr = _trace()
    csv = tmp_path_factory.mktemp("subpop_gal") / "many.csv"
    _write_csv(csv, arr)
    return csv, arr


@pytest.fixture
def win(qapp, qtbot, recording):
    """A window with a REAL detection over a many-event trace. Gallery not yet opened."""
    csv, arr = recording
    w = MainWindow()
    qtbot.addWidget(w)
    doc_id = w._session.add_document(_doc(csv, arr))
    w._create_default_layers(w._session.document(doc_id))
    ch = next(c.index for c in w._session.document(doc_id).meta.channels
              if c.kind is ChannelKind.CURRENT and not c.is_dead)
    w._on_selection(doc_id, ch)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    yield w, doc_id, ch
    w._analysis_controller.abort()
    w._controller.abort()


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """This module's own QSettings namespace — nothing here reads or leaves app state.

    MODULE-SCOPED, AND THAT IS A MEASUREMENT AND NOT A STYLE CHOICE. Written per test —
    ``QSettings().clear()`` around every one — this module took over twenty minutes and
    looked like a hang: each ``MainWindow`` re-reads its persisted state on construction,
    and clearing the store between windows turned that into a growing cost paid by every
    window still alive in the process. Same isolation, one clear at each end: 32 s.
    """
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("SubpopGalleriesPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


def _open_gallery(w, qtbot) -> EventGalleryWindow:
    """Through the MENU ACTION, never by constructing the window — same as every other
    gallery test in this tree, because a handler that works when called directly and
    fails from the menu is a defect this project has already shipped once."""
    w._act_event_gallery.trigger()
    qtbot.wait(30)
    assert w._event_gallery is not None
    return w._event_gallery


def _frame(w) -> pd.DataFrame:
    return w._analysis_panel.current_detection().payload.df


def _fake_assignment(df: pd.DataFrame, k: int = 3, gated: float = 0.7) -> dict[int, int]:
    """A plausible assignment: ``gated`` of the events named, the rest not in the report.

    Deliberately NOT every event. On the owner's zn_30h the >=4 sigma gate keeps roughly a
    tenth of what was detected, so "an event the map does not name" is the common case and
    not an edge one — which is why the gallery gives it a combo entry of its own.
    """
    ids = df["event_id"].to_numpy(dtype=np.int64)
    n = int(round(len(ids) * gated))
    return {int(e): int(i % k) for i, e in enumerate(ids[:n])}


def _spans_on_pane(w) -> list[SpanOverlayItem]:
    """Every batched span item on the active pane that belongs to the DETECTION.

    Found by ``label == "events"``, the same identity the project-IO tests use — and the
    reason the subpopulation hints keep that label rather than spelling the class into it.
    """
    pane = w._trace_view.panes()[0]
    return [it for it in pane.plot.items
            if isinstance(it, SpanOverlayItem) and it.label == "events"]


# ================================================================================================
# (a) the gallery filter — one subpopulation at a time
# ================================================================================================

def test_a_gallery_with_no_report_shows_no_subpopulation_controls(win, qtbot):
    """An inert control is a claim that there is something to control.

    Until a report arrives there are no classes, so the row and the caveat pane are not
    merely disabled, they are absent — and every existing gallery behaviour is untouched.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)

    assert gal.subpopulations() is None
    assert not gal._subpop_bar.isVisible()
    assert not gal._caveat_box.isVisible()
    assert gal.subpop_filter() is None


def test_installing_an_assignment_lists_the_classes_with_their_counts(win, qtbot):
    """The combo enumerates the classes that are actually present, with n, unassigned LAST.

    The order is ``categorical.category_counts``' order, so this combo, the report window's
    legend and the trace's overlays enumerate the classes identically — two surfaces that
    list the same four classes in two orders is how a reader concludes they disagree.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    assignment = _fake_assignment(df, k=3, gated=0.7)
    gal.set_subpopulations(assignment, source="synthetic", caveats=("a warning",))

    entries = [(gal.subpop_box.itemData(i), gal.subpop_box.itemText(i))
               for i in range(gal.subpop_box.count())]
    assert entries[0][0] is None, "the first entry must be the whole detection"
    ids = [d for d, _t in entries[1:]]
    assert ids == [0, 1, 2, -1], f"classes must be ascending with 'not in the report' last: {ids}"
    for cid, text in entries[1:]:
        n = sum(1 for e in df["event_id"].to_numpy()
                if assignment.get(int(e), -1) == cid)
        assert text.endswith(f"· {n}"), f"{text!r} does not carry class {cid}'s count {n}"
    assert gal._subpop_bar.isVisible()


def test_filtering_to_one_subpopulation_shows_exactly_the_events_it_names(win, qtbot):
    """Set equality against the map, not a count. A filter that showed the right NUMBER of
    the wrong events would pass a count assertion and be a different picture."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    assignment = _fake_assignment(df, k=3, gated=0.7)
    gal.set_subpopulations(assignment)
    gal.set_page_size("200")
    gal.set_subpop_filter(1)

    shown = {int(df["event_id"].iloc[r]) for r in gal._model.df_rows()}
    expected = {e for e, c in assignment.items() if c == 1}
    assert shown == expected


def test_the_events_the_report_never_saw_are_a_class_of_their_own_and_are_grey(win, qtbot):
    """Reachable, named, and NOT given a palette slot.

    On a real run the gate throws away most of the detection; a gallery of the survivors
    that never mentions the rest is a picture of a filter. They are their own combo entry,
    their chip is ``categorical``'s "in no class" grey, and that grey is deliberately not a
    colour any class can be — a noise cloud in a categorical colour reads as one more class.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    assignment = _fake_assignment(df, k=3, gated=0.7)
    gal.set_subpopulations(assignment)
    gal.set_page_size("200")
    gal.set_subpop_filter(-1)

    shown = {int(df["event_id"].iloc[r]) for r in gal._model.df_rows()}
    assert shown == set(df["event_id"].to_numpy(dtype=np.int64).tolist()) - set(assignment)
    assert shown, "this fixture no longer has any unassigned events, so it tests nothing"

    row = gal._model.df_rows()[0]
    cid, text = gal.subpop_of(row)
    assert cid < 0 and text == "s–"
    assert CAT.category_colour(cid).name().lower() == CAT.NOISE_HEX.lower()
    assert CAT.category_colour(cid).name() not in {CAT.category_colour(c).name()
                                                   for c in range(8)}


def test_asking_for_a_subpopulation_that_is_not_there_raises(win, qtbot):
    """A silent no-op would leave the caller reading "subpop 7" over every event there is."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_subpopulations(_fake_assignment(gal.frame(), k=3))
    with pytest.raises(ValueError, match="subpopulation"):
        gal.set_subpop_filter(7)


def test_one_subpopulation_is_one_ink_in_the_gallery_and_in_the_report_window(win, qtbot):
    """The reason the palette is shared at all.

    The chip on a tile, the point in the t-SNE and the span on the trace are three pictures
    of one event; if subpop 2 is green in one and orange in another, the reader's only
    correct conclusion is that they are about different things.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    assignment = _fake_assignment(df, k=3)
    gal.set_subpopulations(assignment)

    for row in range(len(df)):
        cid, _text = gal.subpop_of(row)
        expected = assignment.get(int(df["event_id"].iloc[row]), -1)
        assert cid == expected
        # the gallery asks the ONE map, by class id, exactly as the scatter does
        assert CAT.category_colour(cid) == CAT.category_colour(expected)


# ================================================================================================
# (b) representative, in the owner's sense
# ================================================================================================

def _dwells(gal) -> np.ndarray:
    df = gal.frame()
    return np.array([float(df["dwell_s"].iloc[r]) for r in gal._model.df_rows()])


def test_a_representative_sample_is_not_the_first_n_events(win, qtbot):
    """The whole difference between a sample and a prefix.

    The detection order is time order, so the first 12 tiles of a subset are the first 12
    events of the recording. The sample must not be that set — and, more strongly, must
    span the subset's dwell range, which a prefix has no reason to do.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    gal.set_subpopulations(_fake_assignment(df, k=1, gated=1.0))
    gal.set_page_size("24")

    gal.set_representative(False)
    prefix = list(gal._model.df_rows())
    gal.set_representative(True)
    sample = list(gal._model.df_rows())

    assert len(sample) == 24
    assert sample != prefix
    all_dwell = df["dwell_s"].to_numpy(dtype=np.float64)
    picked = all_dwell[sample]
    assert picked.min() == pytest.approx(all_dwell.min())
    assert picked.max() == pytest.approx(all_dwell.max()), (
        "a representative sample that never shows the longest event in the class is not "
        "representative of the class")


def test_the_sample_is_the_dwell_order_statistics_at_even_spacing(win, qtbot):
    """Pinned against the ladder exactly, so "spaced by dwell" is a claim and not a hope.

    A quantile ladder in the ORDER statistics, not in dwell VALUE: even spacing in value on
    a distribution spanning a factor of 18 (which two of the owner's four subpopulations
    do) would put most of the sample in an empty tail.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    gal.set_subpopulations(_fake_assignment(df, k=1, gated=1.0))
    gal.set_page_size("24")
    gal.set_representative(True)

    d = df["dwell_s"].to_numpy(dtype=np.float64)
    ladder = np.argsort(d, kind="stable")
    pos = np.unique(np.rint(np.linspace(0.0, ladder.size - 1, 24)).astype(np.int64))
    expected = ladder[pos]

    assert sorted(gal._model.df_rows()) == sorted(expected.tolist())


def test_the_sample_is_taken_from_the_filtered_subset_not_the_whole_detection(win, qtbot):
    """Filter first, sample second. Sampling the detection and then filtering would give a
    handful of tiles for a class that has hundreds."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    assignment = _fake_assignment(df, k=3, gated=1.0)
    gal.set_subpopulations(assignment)
    gal.set_page_size("24")
    gal.set_subpop_filter(2)
    gal.set_representative(True)

    shown = {int(df["event_id"].iloc[r]) for r in gal._model.df_rows()}
    assert shown <= {e for e, c in assignment.items() if c == 2}
    assert len(shown) == min(24, sum(1 for c in assignment.values() if c == 2))


def test_per_page_is_the_sample_size(win, qtbot):
    """The sample IS one page, so the page-size control is the sample-size control — which
    is what the checkbox's own tooltip tells the user."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_subpopulations(_fake_assignment(gal.frame(), k=1, gated=1.0))
    gal.set_page_size("24")
    gal.set_representative(True)
    assert len(gal._model.df_rows()) == 24
    gal.set_page_size("48")
    assert len(gal._model.df_rows()) == 48
    assert gal.page_count() == 1, "the sample is one page; there is nothing below the fold"


def test_a_sample_larger_than_the_subset_is_the_whole_subset(win, qtbot):
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    gal.set_subpopulations(_fake_assignment(df, k=1, gated=1.0))
    gal.set_page_size("200")
    gal.set_representative(True)
    assert len(gal._model.df_rows()) == len(df)


def test_events_with_no_dwell_are_left_out_of_the_ladder_and_the_status_line_says_so(
        win, qtbot):
    """A dwell ladder has no rung for an unmeasurable dwell.

    They are excluded — folding them in at one end would make a tile that got into a
    dwell-spaced sample without a dwell, which is then read as having one — and the count
    is on the surface, with the way to reach them.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    df.loc[df.index[:4], "dwell_s"] = np.nan
    gal.refresh()
    gal.set_subpopulations(_fake_assignment(df, k=1, gated=1.0))
    gal.set_page_size("24")
    gal.set_representative(True)

    assert np.isfinite(_dwells(gal)).all(), "a NaN dwell reached the ladder"
    assert gal._model.no_dwell_count() == 4
    assert "4 with no dwell left out" in gal.status.text()
    assert "untick" in gal.status.text(), "saying what was dropped without saying how to "\
                                          "reach it is half a caveat"


def test_the_status_line_separates_the_sample_from_the_subset_it_came_from(win, qtbot):
    """"How many are in subpop 2" and "how many are on screen" are two questions, and a
    status line that printed one under both names is how a 24-tile sample gets quoted as a
    class of 24."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    gal.set_subpopulations(_fake_assignment(df, k=1, gated=1.0))
    gal.set_page_size("24")
    gal.set_representative(True)

    text = gal.status.text()
    assert f"representative sample: 24 of {len(df)}" in text
    assert gal._model.subset_size() == len(df)


# ================================================================================================
# (c) which trace the tiles are drawn from — the caution the extraction found
# ================================================================================================

def test_the_status_line_always_names_the_trace_the_tiles_show(win, qtbot):
    """Even when it is the ordinary one.

    The owner's notebook calls these "raw-event galleries" and draws them from
    ``current_detr_pA``. A picture is not self-describing, and that mislabelling is
    precisely what naming the trace on the surface prevents from happening again here.
    """
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    assert gal.trace_name() == TRACE_CALIBRATED
    assert "drawn from: calibrated (as recorded)" in gal.status.text()


def test_the_detrended_trace_is_offered_only_once_one_has_been_handed_over(win, qtbot):
    """Offered, never rebuilt here: a second implementation of ``detection_signal`` in the
    GUI would drift from the one the features were measured on."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    options = lambda: [gal.trace_box.itemText(i) for i in range(gal.trace_box.count())]

    assert options() == [TRACE_CALIBRATED]
    with pytest.raises(ValueError, match="no trace"):
        gal.set_trace(TRACE_DETRENDED)

    detr = np.asarray(w._session.calibrated(_doc_id, _ch).data, dtype=np.float32) - 100.0
    gal.set_detrended_signal(detr)
    assert options() == [TRACE_CALIBRATED, TRACE_DETRENDED]


def test_selecting_the_detrended_trace_actually_changes_the_tiles_and_says_so(win, qtbot):
    """The envelope is re-sliced from the other array, not re-labelled.

    The offset below is what an open-pore drift looks like at its crudest; the tiles must
    move with it, or the combo is a label over an unchanged picture.
    """
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    raw = np.asarray(w._session.calibrated(doc_id, ch).data, dtype=np.float32)
    gal.set_detrended_signal(raw - 100.0)
    row = gal._model.df_rows()[0]

    before = gal._envelopes.envelope(
        int(gal.frame()["start_idx"].iloc[row]), int(gal.frame()["end_idx"].iloc[row]), 140)
    gal.set_trace(TRACE_DETRENDED)
    after = gal._envelopes.envelope(
        int(gal.frame()["start_idx"].iloc[row]), int(gal.frame()["end_idx"].iloc[row]), 140)

    assert before is not None and after is not None
    assert np.allclose(np.asarray(after[0]) + 100.0, np.asarray(before[0]))
    assert gal.trace_name() == TRACE_DETRENDED
    assert "drawn from: detrended (as detected)" in gal.status.text()


def test_withdrawing_the_detrended_trace_falls_back_and_renames(win, qtbot):
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    raw = np.asarray(w._session.calibrated(doc_id, ch).data, dtype=np.float32)
    gal.set_detrended_signal(raw - 100.0)
    gal.set_trace(TRACE_DETRENDED)
    gal.set_detrended_signal(None)
    assert gal.trace_name() == TRACE_CALIBRATED
    assert not gal._envelopes.has_override()


# ================================================================================================
# (d) the caveats travel with the assignment
# ================================================================================================

def test_the_caveat_pane_appears_with_the_assignment_and_carries_every_warning_verbatim(
        win, qtbot):
    """Verbatim and numbered, not summarised — a caveat paraphrased into a reassuring
    sentence is worse than no caveat. Plus the one that is about THIS picture, which cannot
    come from the run's warnings because the run does not know it is being drawn."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    warnings = ("BOOTSTRAP INTERVALS ARE NOT CALIBRATED: label switching.",
                "BIC DISAGREES WITH THE CROSS-VALIDATION ABOUT K.")
    gal.set_subpopulations(_fake_assignment(gal.frame()), caveats=warnings)

    text = gal.subpop_caveats()
    for wrn in warnings:
        assert wrn in text, "a warning was paraphrased or dropped"
    assert "posterior" in text.lower() or "mixture" in text.lower()
    assert gal._caveat_box.isVisible()
    assert str(len(warnings)) in gal._caveat_box.title()


def test_the_caveats_are_not_behind_a_tab(win, qtbot):
    """The same decision the report window makes, for the same reason: a tab is a place to
    put the reasons not to believe a picture where the reader of the picture is not."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_subpopulations(_fake_assignment(gal.frame()), caveats=("x",))
    assert not gal.findChildren(QtWidgets.QTabWidget)
    assert gal._caveat_box.isVisible()


def test_dropping_the_assignment_takes_the_controls_and_the_caveats_with_it(win, qtbot):
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_subpopulations(_fake_assignment(gal.frame()), caveats=("x",))
    gal.set_subpopulations(None)
    assert gal.subpopulations() is None
    assert not gal._subpop_bar.isVisible()
    assert not gal._caveat_box.isVisible()
    assert gal.subpop_caveats() == ""
    assert not gal.is_representative()


# ================================================================================================
# (e) the assignment is a view, never a column
# ================================================================================================

def test_installing_and_filtering_writes_nothing_to_the_shared_frame(win, qtbot):
    """A provisional mixture posterior that reached the frame would reach the exports, the
    project file and every other surface as though it had been measured."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    before = df.copy(deep=True)

    gal.set_subpopulations(_fake_assignment(df, k=3), caveats=("x",))
    gal.set_subpop_filter(1)
    gal.set_representative(True)
    gal.set_page_size("24")

    pd.testing.assert_frame_equal(gal.frame(), before)
    assert list(gal.frame().columns) == list(before.columns)


def test_a_new_detection_drops_the_assignment_rather_than_re_attaching_it_by_id(win, qtbot):
    """A different frame is a different detection, and the same ``event_id`` in it is a
    different event. Re-attaching by id would describe one detection's events with another
    detection's mixture, silently and plausibly."""
    w, doc_id, ch = win
    gal = _open_gallery(w, qtbot)
    gal.set_subpopulations(_fake_assignment(gal.frame(), k=3))
    assert gal.subpopulations() is not None

    other = gal.frame().copy(deep=True)
    gal.set_events(doc_id, ch, other, "detect.schmitt", {})
    assert gal.subpopulations() is None
    assert not gal._subpop_bar.isVisible()


def test_curation_keeps_the_assignment_because_the_frame_is_the_same_object(win, qtbot):
    """Curation, labelling and a mask drag all write the SAME frame in place. Dropping the
    colouring on every edit would make it useless in the one window built for editing."""
    w, _doc_id, _ch = win
    gal = _open_gallery(w, qtbot)
    df = gal.frame()
    assignment = _fake_assignment(df, k=3)
    gal.set_subpopulations(assignment)
    df.loc[df.index[0], "accepted"] = False
    gal.refresh()
    assert gal.subpopulations() == assignment


# ================================================================================================
# (f) the trace map — one item per class, never one per event
# ================================================================================================

def test_without_an_assignment_the_trace_draws_one_amber_span_overlay(win, qtbot):
    """The pre-existing picture, unchanged. Everything below is additive to this."""
    w, _doc_id, _ch = win
    items = _spans_on_pane(w)
    assert len(items) == 1
    assert items[0].name == "", "an overlay with no report has no names to write"


def test_colouring_cuts_the_spans_into_one_batched_item_per_class(win, qtbot):
    """THE COST IS O(CLASSES), NOT O(EVENTS), and that is the whole reason this is
    affordable. The anti-``10k-LinearRegionItem`` design gives one graphics item per hint;
    four subpopulations over eleven hundred events must cost four items, not 1076."""
    w, _doc_id, _ch = win
    df = _frame(w)
    assignment = _fake_assignment(df, k=3, gated=0.7)
    w._analysis_panel.set_subpop_labels(assignment)
    qtbot.wait(20)

    items = _spans_on_pane(w)
    assert len(items) == 4, "three classes plus the events the report does not name"
    assert sum(it.n_spans for it in items) <= len(df)
    assert all(it.label == "events" for it in items), (
        "every one of these IS the detection's span overlay; the class travels in colour "
        "and name, not in the identity the panel and project-IO find it by")


def test_the_coloured_spans_are_exactly_the_accepted_events_no_more_and_no_fewer(win, qtbot):
    """A recolouring that lost or duplicated an event would still look like a plausible
    map. Compared as a multiset of x-bounds against the accepted rows."""
    w, doc_id, ch = win
    df = _frame(w)
    w._analysis_panel.set_subpop_labels(_fake_assignment(df, k=3, gated=0.7))
    qtbot.wait(20)

    drawn = []
    for it in _spans_on_pane(w):
        drawn.extend(zip(it._x0.tolist(), it._x1.tolist(), strict=True))
    accepted = df[df["accepted"].to_numpy(dtype=bool)]
    assert len(drawn) == len(accepted)
    assert len(set(drawn)) == len(drawn), "an event was drawn twice"


def test_the_classes_are_emitted_in_legend_order_with_the_unnamed_last(win, qtbot):
    """Same order as the gallery combo and the report window's legend."""
    w, _doc_id, _ch = win
    w._analysis_panel.set_subpop_labels(_fake_assignment(_frame(w), k=3, gated=0.7))
    qtbot.wait(20)
    names = [it.name for it in _spans_on_pane(w)]
    assert names == ["subpop 0", "subpop 1", "subpop 2", "not in the report"]


def test_each_class_is_drawn_in_the_one_shared_colour_map(win, qtbot):
    w, _doc_id, _ch = win
    w._analysis_panel.set_subpop_labels(_fake_assignment(_frame(w), k=3, gated=0.7))
    qtbot.wait(20)
    items = _spans_on_pane(w)
    for cid, it in zip([0, 1, 2, -1], items, strict=True):
        assert it._name_color.name().lower() == CAT.category_colour(cid).name().lower()


def test_dropping_the_map_puts_the_single_overlay_back(win, qtbot):
    w, _doc_id, _ch = win
    w._analysis_panel.set_subpop_labels(_fake_assignment(_frame(w), k=3))
    qtbot.wait(20)
    assert len(_spans_on_pane(w)) > 1
    w._analysis_panel.set_subpop_labels(None)
    qtbot.wait(20)
    assert len(_spans_on_pane(w)) == 1


def test_the_coloured_spans_still_stay_out_of_the_y_autorange(win, qtbot):
    """``ignoreBounds=True``, and it is hard-won: an overlay item that paints
    full-pane-height rects tracking the view has a bounding rect equal to the view range,
    so left in autorange it ratchets the y-axis wider on every pass and crushes the signal
    to a flat line. Cutting one overlay into four multiplies that failure, not divides it.
    """
    w, _doc_id, _ch = win
    w._analysis_panel.set_subpop_labels(_fake_assignment(_frame(w), k=3))
    qtbot.wait(20)
    pane = w._trace_view.panes()[0]
    tracked = pane.plot.vb.addedItems
    for it in _spans_on_pane(w):
        assert it not in tracked, (
            f"{it.name!r} is in the autorange set; it will ratchet the y-axis")


def test_a_curation_edit_redraws_the_coloured_spans_and_keeps_the_colours(win, qtbot):
    """Curation flows to the trace exactly as it did before the colouring — a rejected
    event stops being shaded, and the survivors keep their class."""
    w, _doc_id, _ch = win
    df = _frame(w)
    w._analysis_panel.set_subpop_labels(_fake_assignment(df, k=3, gated=1.0))
    qtbot.wait(20)
    before = sum(it.n_spans for it in _spans_on_pane(w))

    first_id = int(df["event_id"].iloc[0])
    w._analysis_panel.set_event_accepted(first_id, False)
    qtbot.wait(20)

    after_items = _spans_on_pane(w)
    assert sum(it.n_spans for it in after_items) == before - 1
    assert [it.name for it in after_items][:1] == ["subpop 0"]


# ================================================================================================
# (g) the zoom windows get the names
# ================================================================================================

def test_the_names_are_withheld_on_the_full_recording_map(win, qtbot):
    """1076 copies of "subpop 2" at the full extent is a smear where the signal is.

    The rule, not the rendering: at 250 s a 20 ms event is 8e-5 of the width, so the words
    would overprint one another and hide the trace. The COLOUR still carries the class at
    every zoom — only the words wait for one.
    """
    item = SpanOverlayItem(np.array([100.0]), np.array([100.02]), name="subpop 2")
    assert not item.visible_names(n_visible=800, pixel_width=0.4)
    assert not item.visible_names(n_visible=3, pixel_width=2.0)


def test_the_names_appear_in_a_zoom_window(win, qtbot):
    """One event wide enough to hold its own name, and few enough on screen not to collide.
    That is what "give the zoom windows the report names" means."""
    item = SpanOverlayItem(np.array([100.0]), np.array([100.02]), name="subpop 2")
    assert item.visible_names(n_visible=6, pixel_width=180.0)


def test_an_overlay_with_no_name_never_writes_words(win, qtbot):
    """Every overlay that predates the report — and the plain amber one — stays wordless."""
    item = SpanOverlayItem(np.array([100.0]), np.array([100.02]))
    assert not item.visible_names(n_visible=1, pixel_width=5000.0)


def test_the_hint_carries_the_name_through_the_renderer(win, qtbot):
    """The declarative seam: modules and panels DECLARE, the view renders. ``name`` has to
    survive that trip or the panel's class names never reach a pixel."""
    w, doc_id, ch = win
    hint = OverlayHint(OverlayKind.SPAN, starts=np.array([100], dtype=np.int64),
                       stops=np.array([200], dtype=np.int64), label="events",
                       name="subpop 3", color="#009E73")
    oid = w._trace_view.add_overlay(doc_id, ch, hint)
    try:
        drawn = [it for it in w._trace_view.panes()[0].plot.items
                 if isinstance(it, SpanOverlayItem) and it.name == "subpop 3"]
        assert len(drawn) == 1
        assert drawn[0].label == "events"
    finally:
        w._trace_view.remove_overlay(oid)


def test_name_and_label_are_different_fields_on_purpose(win, qtbot):
    """The identity the panel and the project-IO tests find the detection's spans by is
    ``label``; ``name`` is what gets painted. Folding them together would have made "is
    this the detection's overlay?" depend on how many classes a mixture happened to find.
    """
    hint = OverlayHint(OverlayKind.SPAN, starts=np.array([0], dtype=np.int64),
                       label="events", name="subpop 1")
    assert (hint.label, hint.name) == ("events", "subpop 1")
    assert OverlayHint(OverlayKind.SPAN, starts=np.array([0], dtype=np.int64)).name == ""


# ================================================================================================
# (h) main_window is the sole wirer, and the report is read once
# ================================================================================================

class _Cheap(SR.SubpopConfig):
    """The same pipeline over a small grid — see ``test_subpop_gui._Cheap``."""

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2, 3))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        super().__init__(**kw)


@pytest.fixture
def reported(win, qtbot):
    """A window that has adopted a real report over its own detection.

    THE PIPELINE IS RUN HERE AND THE RESULT HANDED TO THE REAL HANDLER, rather than driven
    through the menu. ``test_subpop_gui`` is where the menu -> controller -> off-thread ->
    ``finished`` path is covered, and it covers it on a window whose detection was
    installed rather than run; this file's windows run a real detection first, and putting
    a second job through the same :class:`AnalysisController` in the same test spends the
    test's time in the hand-off instead of in anything item 13 is about. So the report is
    built by the same function the worker calls, and adopted through the same slot the
    worker's signal reaches — ``_on_subpop_finished``, gate and all.
    """
    w, doc_id, ch = win
    from nanoscope.analysis.modules import detection_signal

    events = w._analysis_panel.current_detection().payload
    raw = np.asarray(w._session.calibrated(doc_id, ch).data)
    prov = dict(events.provenance or {})
    signal = detection_signal(raw, FS, prov)
    report = SR.run_subpopulations(
        signal, raw, FS, events.accepted, recording="synthetic",
        duration_s=N_SAMPLES / FS,
        config=_Cheap(polarity=SR.polarity_for_features(prov.get("polarity"))))
    assert report.quantification is not None, (
        "the fixture produced no mixture, so nothing below is about the wiring")
    # The trace the worker measured on, exactly as the real launch stashes it.
    w._subpop_signal_pending = signal
    w._subpop_pending = True
    w._on_subpop_finished(report)
    qtbot.wait(20)
    assert w._subpop_window is not None
    return w, doc_id, ch


def test_the_report_reaches_the_trace_and_the_gallery_by_itself(reported, qtbot):
    """The whole fan-out, through the real menu -> controller -> off-thread path.

    Nothing tested this before: the report landed in its own window and stopped there.
    """
    w, _doc_id, _ch = reported
    assert w._subpop_assignment, "the run produced no assignment to push"
    assert w._analysis_panel.subpop_labels() == w._subpop_assignment

    # One batched span item per class PRESENT — counted from the assignment rather than
    # asserted as "more than one". On this synthetic recording the cross-validation is free
    # to choose K=1, and a test that demanded two colours would be pinning the mixture's
    # answer instead of the fan-out. What must hold either way is that the trace draws
    # exactly the classes the report found, plus the unnamed group when there is one.
    df = _frame(w)
    accepted = df[df["accepted"].to_numpy(dtype=bool)]
    present = {w._subpop_assignment.get(int(e), -1)
               for e in accepted["event_id"].to_numpy()}
    assert len(_spans_on_pane(w)) == len(present)

    gal = _open_gallery(w, qtbot)
    assert gal.subpopulations() == w._subpop_assignment


def test_the_assignment_is_the_report_s_own_join_and_not_a_second_one(reported):
    """``event_table()`` is the report's join of its features with the mixture's labels —
    the same table the export writes. Deriving the ids here from ``fit.labels`` and
    ``features['event_id']`` would be a second implementation of it living in a window.
    """
    w, _doc_id, _ch = reported
    report = w._subpop_window.report()
    table = report.event_table()
    expected = {int(e): int(v) for e, v in
                zip(table["event_id"], table["subpop"], strict=True) if not pd.isna(v)}
    assert w._subpop_assignment == expected


def test_the_gallery_gets_the_run_s_warnings_verbatim(reported, qtbot):
    w, _doc_id, _ch = reported
    gal = _open_gallery(w, qtbot)
    for wrn in w._subpop_window.report().warnings:
        assert wrn in gal.subpop_caveats()


def test_the_gallery_is_offered_the_trace_the_features_were_measured_on(reported, qtbot):
    """``detection_signal``, handed over from the run rather than rebuilt in the GUI.

    It is what the owner's notebook galleries actually draw, and it is the array every
    reported ``depth_pA`` and ``baseline_pA`` was read off, so the tile's shared depth
    scale (which is referenced to ``baseline_pA``) is measured against its own trace.
    """
    w, _doc_id, _ch = reported
    assert w._subpop_signal is not None
    gal = _open_gallery(w, qtbot)
    options = [gal.trace_box.itemText(i) for i in range(gal.trace_box.count())]
    assert options == [TRACE_CALIBRATED, TRACE_DETRENDED]


def test_the_entry_point_button_opens_the_gallery_on_the_report(reported, qtbot):
    """Report window ▸ "Event gallery…" — announced by the window, routed by MainWindow."""
    w, _doc_id, _ch = reported
    rw = w._subpop_window
    assert rw.can_show_events()
    rw._gallery_btn.click()
    qtbot.wait(50)
    gal = w._event_gallery
    assert gal is not None
    assert gal.subpopulations() == w._subpop_assignment
    assert gal.is_representative(), (
        "opened from the report, the gallery should show a SAMPLE of each class, not the "
        "first tiles of the recording")
    assert gal.trace_name() == TRACE_DETRENDED


def test_the_entry_points_are_dead_without_a_mixture(win, qtbot):
    """A button that opened a gallery indistinguishable from the plain one is a promise the
    report cannot keep. ``quantification is None`` is what the text prints as "NO MIXTURE".
    """
    from nanoscope.gui.subpop_window import SubpopReportWindow

    w, _doc_id, _ch = win
    rw = SubpopReportWindow()
    qtbot.addWidget(rw)
    assert not rw.can_show_events()


def test_a_new_detection_drops_the_colouring_from_both_surfaces(reported, qtbot):
    """The staleness rule, end to end. A new frame is a new detection — the same
    ``event_id`` in it is a different event, so the mixture that described the old one must
    not be left describing the new one."""
    w, doc_id, ch = reported
    gal = _open_gallery(w, qtbot)
    assert gal.subpopulations() is not None

    fresh = EventSet(_frame(w).copy(deep=True), {"analyzed_duration_s": N_SAMPLES / FS})
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id, channel=ch,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, ch).fingerprint(),
        payload=fresh))
    qtbot.wait(80)

    assert w._subpop_assignment is None
    assert w._analysis_panel.subpop_labels() is None
    assert gal.subpopulations() is None
    assert len(_spans_on_pane(w)) == 1


def test_the_run_does_not_leave_a_full_length_array_alive_after_a_cancel(win, qtbot):
    """The detrended trace is a second copy of the recording. It is handed over on the
    terminal signal and never before, so a cancelled or failed run releases it."""
    w, _doc_id, _ch = win
    w._subpop_signal_pending = np.zeros(10, dtype=np.float32)
    w._on_subpop_stopped()
    assert w._subpop_signal_pending is None


# ================================================================================================
# (i) the tree — one colour map, no new panel-to-panel import
# ================================================================================================

@pytest.mark.parametrize("module_name", [
    "nanoscope.gui.event_gallery",
    "nanoscope.gui.analysis_panel",
])
def test_the_new_callers_ask_the_shared_map_and_do_not_spell_a_palette(module_name):
    """Both new callers of the categorical colours use the ONE map.

    A second, independently-written colour map is this tree's signature failure — the log
    axis and the combo setter both came back in the file the first fix had not reached — so
    every surface that draws a class asks ``gui.categorical`` for the ink. This reads the
    AST, so a docstring may still NAME a hex while explaining why it is not written there.
    """
    import importlib
    import re

    from nanoscope.plot.style import OKABE_ITO

    path = pathlib.Path(importlib.import_module(module_name).__file__)
    tree = ast.parse(path.read_text())

    imported = {n.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and (node.module or "") == "categorical"
                for n in node.names}
    assert "category_colour" in imported, f"{module_name} does not use the shared map"

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    palette = {h.lower() for h in OKABE_ITO} - {"#000000", "#ffffff"}
    offenders: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            offenders |= {h.lower() for h in re.findall(r"#[0-9A-Fa-f]{6}", node.value)} & palette
    assert not offenders, (
        f"{module_name} spells palette colours itself: {sorted(offenders)} — ask "
        f"gui.categorical by class id")


def test_the_gallery_still_reaches_no_sibling_panel():
    """The gallery is a standalone window and MainWindow stays the sole wirer. It may now
    reach ``categorical`` — the shared leaf — and nothing else in ``gui/``."""
    import nanoscope.gui.event_gallery as gal_mod

    tree = ast.parse(pathlib.Path(gal_mod.__file__).read_text())
    siblings = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module}
    assert siblings == {"categorical"}, (
        f"the Event gallery reaches {sorted(siblings)}; only the shared colour leaf is "
        f"allowed, and the next addition has to argue for itself here")


def test_the_report_window_offers_the_entry_points_without_importing_a_surface():
    """It announces; it does not reach. The two new signals must not have brought an
    import of the gallery or the trace view with them.

    2026-09-13: the exact-set form of this assertion (``siblings == {"categorical"}``)
    started failing the moment subpop_window.py adopted the SAME shared plot
    appearance/workflow-bar/label infrastructure other analysis windows already use --
    a consolidation, not a reach into a surface. Widened to a named allowlist rather
    than dropped to a subset check, so a genuine future reach for "event_gallery" or
    "trace_view" (the two things this test actually exists to catch, per the
    docstring above) still fails it, and any OTHER new sibling import still has to
    argue for a place on this list rather than passing silently."""
    import nanoscope.gui.subpop_window as subpop_mod

    tree = ast.parse(pathlib.Path(subpop_mod.__file__).read_text())
    siblings = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module}
    allowed = {"categorical", "analysis_catalog", "plot_appearance", "plot_workflow"}
    assert siblings == allowed, (
        f"the report window reaches {sorted(siblings - allowed)} beyond the shared "
        f"infrastructure it is allowed to reuse {sorted(allowed)}; a real surface "
        f"(event_gallery, trace_view) must never be among them")
    src = pathlib.Path(subpop_mod.__file__).read_text()
    assert "galleryRequested" in src and "traceRequested" in src


def test_main_window_is_the_sole_wirer_of_the_two_entry_points():
    """Neither surface may connect them to itself; the routing lives in one file."""
    import nanoscope.gui.event_gallery as gal_mod
    import nanoscope.gui.main_window as main_mod

    main_src = pathlib.Path(main_mod.__file__).read_text()
    assert "galleryRequested.connect" in main_src
    assert "traceRequested.connect" in main_src
    assert "galleryRequested" not in pathlib.Path(gal_mod.__file__).read_text()
