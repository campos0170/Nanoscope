"""THE SHOWN-RECORDING RULE — opening a recording shows THAT recording.

The complaint, verbatim: *"if I open a dataset on the app and then wish to change it for
another, the app places both of them on the detector window. This does not make sense."*

He was right, and the behaviour was deliberate: ``_create_default_layers`` layered every
channel of the first document and, for every later one, every channel "whose kind is already
layered", explicitly "so the 2nd/3rd/4th voltage run overlays comparably instead of piling up
new panes". That is the right default for ONE workflow (compare the -1000/-700/-600/-500 mV
runs) and the wrong default for the ordinary one (open a file, look at it) — and it made the
ordinary one impossible to express at all.

What these tests pin:

* opening a second recording DISPLAYS ONLY IT, while the first stays open, listed, calibrated,
  detected and analysable;
* clicking the first one in the File Manager brings it back — *with its colour and its time
  alignment*, because "shown" is layer VISIBILITY and never layer membership;
* the overlay is still there and is now OPT-IN: ticking a second recording's channel draws it
  alongside, and Overlay/Stacked still govern how several layers render;
* a click on a check indicator does not double as "switch to that recording" (Qt routes it to
  the item delegate and leaves the current item alone — the property the opt-in rests on);
* a project RESTORE is exempt: a deliberately saved multi-recording overlay comes back as one;
* the multi-document analyses (batch detect, I-V, voltage dependence) still see every OPEN
  recording, not merely the shown one;
* creating a segment shows the segment;
* closing the shown recording falls back to a survivor instead of a blank viewer.

Nothing here touches the frozen 20-column EventSet schema.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from conftest import live_doc_id
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.file_dock import _COL_NAME, _ROLE
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.trace_view import plan_panes

FS = 50_000.0
N = 20_000


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, *, name="rec.dat", with_voltage=False, n=N, nominal_mV=None) -> TraceDocument:
    chans = [ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
    arrays = [np.ascontiguousarray(np.full(n, -50.0, dtype=np.float32))]
    if with_voltage:
        chans.append(ChannelInfo(index=1, name="voltage", raw_units="mV",
                                 kind=ChannelKind.VOLTAGE))
        arrays.append(np.ascontiguousarray(np.full(n, -500.0, dtype=np.float32)))
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=tuple(chans),
        nominal_voltage_mV=nominal_mV)
    return TraceDocument(meta=meta, channels=tuple(arrays))


def _open(win: MainWindow, doc: TraceDocument) -> str:
    """Land ``doc`` through the loader's own slot — the real open path, minus the thread.

    ``_on_loaded`` is where a finished load arrives, so this exercises the actual policy
    (default layers + show + focus) rather than a test-only shortcut."""
    win._on_loaded(doc)
    return doc.doc_id


def _shown(win: MainWindow) -> set[str]:
    """The doc_ids actually DRAWN, read off the pane plan the trace view is built from —
    not off the layer list, so this cannot pass by agreeing with the bug."""
    out: set[str] = set()
    for spec in plan_panes(win._session):
        for layer_id in spec.layer_ids:
            out.add(win._session.layer(layer_id).doc_id)
    return out


def _listed(win: MainWindow) -> set[str]:
    """The doc_ids the File Manager tree currently lists as document rows."""
    out: set[str] = set()
    it = QtWidgets.QTreeWidgetItemIterator(win._file_dock._tree)
    while it.value():
        payload = it.value().data(_COL_NAME, _ROLE)
        if payload and payload[0] == "doc":
            out.add(str(payload[1]))
        it += 1
    return out


def _channel_item(win: MainWindow, doc_id: str, ch: int):
    it = QtWidgets.QTreeWidgetItemIterator(win._file_dock._tree)
    while it.value():
        payload = it.value().data(_COL_NAME, _ROLE)
        if payload and tuple(payload) == ("channel", doc_id, ch):
            return it.value()
        it += 1
    return None


def _is_ticked(win: MainWindow, doc_id: str, ch: int) -> bool:
    item = _channel_item(win, doc_id, ch)
    assert item is not None, f"no channel row for {doc_id}#{ch}"
    return item.checkState(_COL_NAME) == QtCore.Qt.Checked


@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace, so ``MainWindow._restore_settings`` cannot restore the
    REAL user's saved ``windowState`` over the layout these tests measure — a developer who
    happens to have collapsed the File Manager in the actual app would otherwise hand the
    geometry tests here a zero-sized dock."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("ShownRecording")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def win(isolated_settings, qapp, qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


# ================================================================================================
# 1. opening a recording shows THAT recording
# ================================================================================================

def test_opening_a_second_recording_shows_only_it(win, tmp_path):
    """THE COMPLAINT. Two opens used to put two recordings on the trace."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    assert _shown(win) == {a}

    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b}, "opening a second recording still piles it onto the first"


def test_the_first_recording_stays_open_and_listed(win, tmp_path):
    """"No longer drawn" must never quietly mean "closed" — nothing may be thrown away."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))

    assert win._session.has_document(a)
    assert _listed(win) == {a, b}
    assert {d.doc_id for d in win._session.documents()} == {a, b}
    # ...and its layer is still THERE, merely invisible: nothing was destroyed.
    assert win._session.has_layer(a, 0)
    assert win._session.layer(f"{a}#0").visible is False


def test_a_third_open_shows_only_the_third(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    c = _open(win, _doc(tmp_path, name="c.dat"))
    assert _shown(win) == {c}
    assert {d.doc_id for d in win._session.documents()} == {a, b, c}


def test_a_newly_opened_recording_gets_all_its_channels(win, tmp_path):
    """The old rule layered only channels "whose kind is already layered" for the 2nd document.
    Open a 3-channel ABF after a current-only CSV and its voltage channel was silently missing —
    harmless while everything piled up, wrong now that the new file is shown on its own."""
    _open(win, _doc(tmp_path, name="current_only.dat"))
    b = _open(win, _doc(tmp_path, name="with_voltage.dat", with_voltage=True))
    assert win._session.has_layer(b, 0)
    assert win._session.has_layer(b, 1), "the second recording's voltage channel was dropped"
    assert _shown(win) == {b}


def test_opening_several_files_at_once_still_shows_one(win, tmp_path):
    """A DECISION, not an oversight. Multi-selecting four voltage runs in the Open dialog is
    tempting to read as "compare these", but guessing at intent is what produced the complaint
    in the first place: the rule is the same however many files an open names. All four stay
    open, the last is shown, and one tick (or right-click ▸ Add to view) builds the comparison."""
    ids = [_open(win, _doc(tmp_path, name=f"run{i}.dat")) for i in range(4)]
    assert _shown(win) == {ids[-1]}
    assert {d.doc_id for d in win._session.documents()} == set(ids)


def test_reopening_an_already_open_file_shows_it_again(win, tmp_path):
    """File ▸ Open (or a Recent-files entry) on something already open must not be the one open
    that leaves you looking at something else."""
    a_doc = _doc(tmp_path, name="a.dat")
    a = _open(win, a_doc)
    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b}

    win.open_paths([str(a_doc.meta.path)])       # already open: no reload, but still an open
    assert _shown(win) == {a}
    assert win._session.has_document(b)


def test_opening_the_second_recording_says_the_first_is_still_open(win, tmp_path):
    """Discoverability at the exact moment the old behaviour used to do the comparing for you."""
    _open(win, _doc(tmp_path, name="a.dat"))
    _open(win, _doc(tmp_path, name="b.dat"))
    msg = win.statusBar().currentMessage()
    assert "b.dat" in msg
    assert "stay open" in msg
    assert "tick" in msg.lower()


def test_the_first_open_announces_nothing(win, tmp_path):
    """There is nothing to explain when there is nothing else open."""
    _open(win, _doc(tmp_path, name="a.dat"))
    assert "stay open" not in win.statusBar().currentMessage()


# ================================================================================================
# 2. picking a recording in the File Manager switches to it
# ================================================================================================

def test_clicking_the_first_recording_brings_it_back(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b}

    assert win._file_dock._focus(("doc", a)) is True     # a real click on the document row
    assert _shown(win) == {a}
    assert win._session.has_document(b)                  # ...and B is still open


def test_clicking_a_channel_row_switches_too(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat", with_voltage=True))
    _open(win, _doc(tmp_path, name="b.dat"))
    assert win._file_dock.focus_channel(a, 1) is True
    assert _shown(win) == {a}
    assert win._current_selection == (a, 1)              # selectionChanged semantics intact


def test_switching_back_preserves_colour_and_alignment(win, tmp_path):
    """WHY the mechanism is visibility and not membership. Removing A's layer on a switch and
    re-adding it on the way back would hand it the NEXT palette colour and reset the time
    offset the user dialled in to align the two runs — data loss for a mere view change."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    sess = win._session
    sess.set_layer_color(f"{a}#0", "#E57373")
    sess.set_layer_time_offset(f"{a}#0", 0.25)

    _open(win, _doc(tmp_path, name="b.dat"))
    win._file_dock._focus(("doc", a))

    layer = sess.layer(f"{a}#0")
    assert layer.color == "#E57373"
    assert layer.time_offset_s == pytest.approx(0.25)
    assert layer.visible is True


def test_the_checkbox_tracks_what_is_drawn(win, tmp_path):
    """A tick means "you can see it" — never "there is a hidden layer for it"."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    assert _is_ticked(win, a, 0)

    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert not _is_ticked(win, a, 0), "A reads as drawn while it is not on the trace"
    assert _is_ticked(win, b, 0)

    win._file_dock._focus(("doc", a))
    assert _is_ticked(win, a, 0)
    assert not _is_ticked(win, b, 0)


def test_switching_reframes_instead_of_inheriting_the_other_recordings_zoom(win, tmp_path):
    """A switch is not a channel toggle: the incoming recording must be framed on its own
    extent, not left in the outgoing one's window (which, on a shorter recording, is off the
    end of it)."""
    long_doc = _doc(tmp_path, name="long.dat", n=N)
    short_doc = _doc(tmp_path, name="short.dat", n=N // 4)
    a = _open(win, long_doc)
    b = _open(win, short_doc)

    lo, hi = win._trace_view.anchor_plot().vb.viewRange()[0]
    assert hi - lo == pytest.approx((N // 4 - 1) / FS, rel=0.05)

    win._file_dock._focus(("doc", a))
    lo, hi = win._trace_view.anchor_plot().vb.viewRange()[0]
    assert hi - lo == pytest.approx((N - 1) / FS, rel=0.05)
    assert b in {d.doc_id for d in win._session.documents()}


# ================================================================================================
# 3. the overlay survives, as an explicit opt-in
# ================================================================================================

def test_ticking_a_second_recordings_channel_overlays_it(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b}

    _channel_item(win, a, 0).setCheckState(_COL_NAME, QtCore.Qt.Checked)
    assert _shown(win) == {a, b}, "ticking a second recording no longer overlays it"


def test_a_click_on_the_check_indicator_does_not_switch_recordings(win, tmp_path, qtbot):
    """The property the whole opt-in rests on, asserted through a REAL mouse click.

    Selecting a recording now switches to it, and a checkbox lives on a selectable row — if a
    click on the indicator also moved the current item, ticking A's channel to compare it with
    B would hide B on the way in and the overlay could never be built by the one gesture the
    UI offers for it. Qt sends a press inside the check rect to the item delegate and returns
    before touching the selection; this pins that, in this tree, with these flags."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    win.resize(1600, 900)
    win.show()
    QtWidgets.QApplication.processEvents()
    win.resizeDocks([win._file_dock], [420], QtCore.Qt.Horizontal)
    QtWidgets.QApplication.processEvents()

    tree = win._file_dock._tree
    item = _channel_item(win, a, 0)
    tree.scrollToItem(item)
    QtWidgets.QApplication.processEvents()
    idx = tree.indexFromItem(item, _COL_NAME)
    opt = QtWidgets.QStyleOptionViewItem()
    opt.rect = tree.visualRect(idx)
    opt.features |= QtWidgets.QStyleOptionViewItem.HasCheckIndicator
    ind = tree.style().subElementRect(
        QtWidgets.QStyle.SE_ItemViewItemCheckIndicator, opt, tree)
    assert ind.isValid() and tree.viewport().rect().contains(ind.center())

    QTest.mouseClick(tree.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, ind.center())
    QtWidgets.QApplication.processEvents()

    assert _is_ticked(win, a, 0)
    assert _shown(win) == {a, b}, "clicking the tick box switched instead of overlaying"
    assert win._current_selection[0] == b, "the tick box moved the selection"


def test_overlay_and_stacked_still_govern_a_two_recording_view(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    _channel_item(win, a, 0).setCheckState(_COL_NAME, QtCore.Qt.Checked)
    assert _shown(win) == {a, b}

    win._file_dock._btn_overlay.click()                 # comparable channels share one pane
    assert win._session.view_mode is ViewMode.OVERLAY
    assert len(plan_panes(win._session)) == 1
    assert len(win._trace_view._panes) == 1

    win._file_dock._btn_stacked.click()                 # one pane each
    assert win._session.view_mode is ViewMode.STACKED
    assert len(plan_panes(win._session)) == 2
    assert len(win._trace_view._panes) == 2


def test_unticking_takes_the_channel_off_and_keeps_it_off(win, tmp_path):
    """Unticking is a decision about the CHANNEL, not about which recording is on screen: it
    must not be resurrected by switching away and back.

    C15: it used to keep that promise by DELETING the layer, which took the channel's
    colour and its Δt alignment with it. The decision is recorded on the layer now
    (``Layer.excluded``), so the channel stays off the trace without anything being
    destroyed — see ``test_unticking_a_channel_keeps_its_colour_and_alignment``."""
    a = _open(win, _doc(tmp_path, name="a.dat", with_voltage=True))
    _channel_item(win, a, 1).setCheckState(_COL_NAME, QtCore.Qt.Unchecked)
    assert win._session.layer(f"{a}#1").visible is False
    assert win._session.layer(f"{a}#1").excluded is True
    assert 1 not in {lay.channel for lay in win._session.layers()
                     if lay.doc_id == a and lay.visible}

    b = _open(win, _doc(tmp_path, name="b.dat"))
    win._file_dock._focus(("doc", a))
    assert _shown(win) == {a}
    assert win._session.layer(f"{a}#0").visible is True
    assert win._session.layer(f"{a}#1").visible is False, \
        "the unticked channel came back on a switch"
    assert win._session.has_document(b)


def test_unticking_a_channel_keeps_its_colour_and_alignment(win, tmp_path):
    """C15. Two behaviours were behind one checkbox: ``remove_layer`` (destructive) here,
    ``show_only`` (a hide) everywhere else. Measured before the fix: a channel with
    ``time_offset_s=1.5`` and a hand-picked colour came back at 0.0 s in the next palette
    colour after one untick/re-tick — and ``time_offset_s`` is persisted and changes what
    the ML detector measures."""
    a = _open(win, _doc(tmp_path, name="a.dat", with_voltage=True))
    lid = f"{a}#0"
    win._session.set_layer_time_offset(lid, 1.5)
    win._session.set_layer_color(lid, "#FF00FF")

    item = _channel_item(win, a, 0)
    item.setCheckState(_COL_NAME, QtCore.Qt.Unchecked)
    assert win._session.layer(lid) is not None, "unticking must not destroy the layer"
    assert win._session.layer(lid).visible is False

    item.setCheckState(_COL_NAME, QtCore.Qt.Checked)
    lay = win._session.layer(lid)
    assert lay.time_offset_s == 1.5
    assert lay.color == "#FF00FF"
    assert lay.excluded is False


def test_an_excluded_channel_survives_a_project_round_trip(win, tmp_path):
    """The untick is persisted state like the colour beside it, so reopening the project
    shows the trace the user left, not one channel more."""
    a = _open(win, _doc(tmp_path, name="a.dat", with_voltage=True))
    _channel_item(win, a, 1).setCheckState(_COL_NAME, QtCore.Qt.Unchecked)

    from nanoscope.core.project import load_project
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj)) is True
    state = load_project(str(proj))
    rec = {(lay.doc_id, lay.channel): lay for lay in state.layers}
    excluded = [k for k, lay in rec.items() if lay.excluded]
    assert [ch for _d, ch in excluded] == [1]


def test_a_project_written_before_excluded_existed_still_loads(tmp_path):
    """MIN_READ_VERSION stays 1: the key is additive and its absence means False, which is
    exactly what it meant (an untick used to delete the layer, so a layer that IS in an
    older project was never an excluded one)."""
    import json
    import zipfile

    from nanoscope.core.project import FORMAT, load_project
    manifest = {
        # v10 is the last format before the mandatory v11 analytical-workspace
        # section.  Calling a manifest "old" while stamping today's VERSION makes it
        # a malformed current file, not a backward-compatibility fixture.
        "format": FORMAT, "version": 10, "app_version": "0.1.0",
        "documents": [], "calibrations": [],
        "layers": [{"doc_id": "/x/a.dat", "channel": 0, "color": "#4FC3F7",
                    "visible": True, "time_offset_s": 0.0}],
        "view_mode": "overlay", "results": [], "measurements": None,
    }
    path = tmp_path / "old.nanoscope"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    state = load_project(str(path))
    assert state.layers[0].excluded is False


def test_add_to_view_draws_a_recording_alongside(win, tmp_path):
    """The right-click affordance — the same opt-in, spelled out in words."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    win._file_dock.addToViewRequested.emit(a)
    assert _shown(win) == {a, b}
    assert "Added" in win.statusBar().currentMessage()


def test_show_only_this_recording_switches(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    win._file_dock.addToViewRequested.emit(a)
    assert _shown(win) == {a, b}
    win._file_dock.showOnlyRequested.emit(b)
    assert _shown(win) == {b}


def test_show_only_retargets_the_analysis_surfaces_at_what_it_shows(win, tmp_path):
    """A WRONG-TARGET regression, not a cosmetic one.

    The context menu is raised on whichever row was right-clicked, which need not be the
    current one. *Show only this recording* used to be wired straight to ``_show_document``,
    so it changed what was DRAWN without moving the selection: the trace showed A while
    Detect, Measure, Calibrate, the Inspector and the workflow strip all still pointed at B —
    which was now invisible. Pressing Detect then ran on a recording that was not on screen
    and reported its event count as if it were the one being looked at.
    """
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b} and win._current_selection[0] == b

    win._file_dock.showOnlyRequested.emit(a)     # right-click on A while B is selected

    assert _shown(win) == {a}
    assert win._current_selection[0] == a, "analysis left pointing at the hidden recording"
    assert win._analysis_panel.measure_panel._target[0] == a
    assert win._file_dock._tree.currentItem().data(_COL_NAME, _ROLE)[1] == a


def test_show_only_on_the_already_current_row_still_shows_it(win, tmp_path):
    """The other half of the same wiring. Routing through the dock's selection is not enough
    on its own: when the requested row is ALREADY current Qt emits no selectionChanged, so
    the show has to happen explicitly or the menu item would silently do nothing."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    win._file_dock.addToViewRequested.emit(a)
    assert _shown(win) == {a, b}
    assert win._current_selection[0] == b        # the tick/add did not move the selection

    win._file_dock.showOnlyRequested.emit(b)     # B's row is already the current one
    assert _shown(win) == {b}


def test_the_compare_affordance_is_discoverable(win, tmp_path):
    """If the user has to guess that a tick box is how you compare two runs, it is not an
    affordance. It is said on the row, on the recording, and under the tree."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    dock = win._file_dock

    tip = _channel_item(win, a, 0).toolTip(_COL_NAME)
    assert "Tick to draw this channel" in tip
    assert "compare" in tip

    doc_item = next(i for i in (dock._tree.topLevelItem(0).child(j)
                                for j in range(dock._tree.topLevelItem(0).childCount()))
                    if tuple(i.data(_COL_NAME, _ROLE)) == ("doc", a))
    assert "the others stay open" in doc_item.toolTip(_COL_NAME)
    assert "Add to view" in doc_item.toolTip(_COL_NAME)

    assert dock._hint.isVisibleTo(dock)
    assert "compare" in dock._hint.toolTip()


def test_the_hint_costs_one_line_and_no_dock_width(win):
    """The hint elides; it must not wrap a squeezed dock into three lines of chrome, nor floor
    the dock's width at its own sentence — the two height-and-width-for-nothing mistakes the
    toolbar row above it had to be rescued from."""
    win.resize(1600, 900)
    win.show()
    QtWidgets.QApplication.processEvents()
    dock = win._file_dock

    win.resizeDocks([dock], [420], QtCore.Qt.Horizontal)
    QtWidgets.QApplication.processEvents()
    roomy_hint = dock._hint.height()
    assert dock._hint.text()                                   # something is actually shown

    win.resizeDocks([dock], [120], QtCore.Qt.Horizontal)
    QtWidgets.QApplication.processEvents()
    assert dock.width() == pytest.approx(120, abs=12)          # the dock still squeezes…
    assert dock._hint.height() == roomy_hint                   # …at the SAME one line
    assert len(dock._hint.text()) < len(dock._hint._full)      # elided, not wrapped
    # (what the squeeze costs the tree is the toolbar's scrollbar and nothing else — pinned by
    #  test_window_layout.test_the_file_manager_toolbar_scrolls_instead_of_flooring_the_dock.)


# ================================================================================================
# 4. project restore is EXEMPT — a saved overlay comes back as an overlay
# ================================================================================================

def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _csv_doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(np.ascontiguousarray(arr, dtype=np.float32),))


def test_a_saved_two_recording_overlay_restores_as_an_overlay(qapp, qtbot, tmp_path):
    """THE most likely way to break this change. A restore reconstructs a layer set the user
    deliberately built; the new "showing one recording" rule must not touch it — not on the
    opens it drives, and not on the selection it makes at the end."""
    csv_a, csv_b = tmp_path / "pore_a.csv", tmp_path / "pore_b.csv"
    arr = np.full(4096, -50.0, dtype=np.float32)
    _write_csv(csv_a, arr)
    _write_csv(csv_b, arr)

    src = MainWindow()
    qtbot.addWidget(src)
    a = _open(src, _csv_doc(csv_a, arr))
    b = _open(src, _csv_doc(csv_b, arr))
    src._file_dock.addToViewRequested.emit(a)            # deliberately build the comparison
    assert _shown(src) == {a, b}
    src._session.set_layer_color(f"{a}#0", "#E57373")
    src._session.set_layer_time_offset(f"{a}#0", 0.125)
    src._session.set_view_mode(ViewMode.STACKED)
    proj = tmp_path / "overlay.nanoscope"
    src._write_project(str(proj))
    src._controller.abort()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)

    # Re-resolved: reopening a file is a NEW LOAD with a new doc_id (core/identity.py); the
    # identity that crosses the round-trip is the source path, which is what the project
    # stores. The VIEW is what this test is about, and it is unchanged.
    a2, b2 = live_doc_id(w2._session, a), live_doc_id(w2._session, b)
    assert a2 is not None and b2 is not None and {a2, b2} != {a, b}
    assert _shown(w2) == {a2, b2}, "the restore collapsed a saved overlay to one recording"
    assert w2._session.layer(f"{a2}#0").color == "#E57373"
    assert w2._session.layer(f"{a2}#0").time_offset_s == pytest.approx(0.125)
    assert w2._session.view_mode is ViewMode.STACKED
    assert w2._restoring is False
    w2._controller.abort()


def test_a_saved_hidden_layer_stays_hidden_through_a_restore(qapp, qtbot, tmp_path):
    """The other direction: a layer saved invisible must come back invisible, and the restore's
    own final selection must not turn it on."""
    csv_a, csv_b = tmp_path / "h_a.csv", tmp_path / "h_b.csv"
    arr = np.full(4096, -50.0, dtype=np.float32)
    _write_csv(csv_a, arr)
    _write_csv(csv_b, arr)

    src = MainWindow()
    qtbot.addWidget(src)
    a = _open(src, _csv_doc(csv_a, arr))
    b = _open(src, _csv_doc(csv_b, arr))     # B shown, A open-but-hidden
    assert _shown(src) == {b}
    proj = tmp_path / "hidden.nanoscope"
    src._write_project(str(proj))
    src._controller.abort()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)

    a2, b2 = live_doc_id(w2._session, a), live_doc_id(w2._session, b)
    assert a2 is not None and b2 is not None
    assert w2._session.has_layer(a2, 0)
    assert w2._session.layer(f"{a2}#0").visible is False
    # ...and the restore's own final selection did not turn it on. _select_after_restore
    # prefers a target that is ALREADY DRAWN precisely so that "show what you analyse" never
    # costs the user the view they saved.
    assert _shown(w2) == {b2}
    assert w2._current_selection[0] == b2
    w2._controller.abort()


# ================================================================================================
# 5. segments
# ================================================================================================

def _select_region(win: MainWindow, doc_id: str, i0: int, i1: int) -> None:
    win._analysis_panel.measure_panel.set_active(True)
    vals = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(vals[i0]), float(vals[i1]))
    QtWidgets.QApplication.processEvents()


def test_creating_a_segment_shows_the_segment(win, tmp_path):
    parent = _open(win, _doc(tmp_path, name="parent.dat"))
    _select_region(win, parent, 2_000, 6_000)
    win._ask_segment_name = lambda default: "cut"
    win._create_segment_from_selection()

    seg = win._session.segment_documents()[0]
    assert _shown(win) == {seg.doc_id}, "the segment was cut but not shown"
    assert win._current_selection[0] == seg.doc_id
    assert win._session.has_document(parent)          # the parent is still open


def test_switching_back_to_the_parent_shows_the_parent(win, tmp_path):
    parent = _open(win, _doc(tmp_path, name="parent.dat"))
    _select_region(win, parent, 2_000, 6_000)
    win._ask_segment_name = lambda default: "cut"
    win._create_segment_from_selection()
    seg_id = win._session.segment_documents()[0].doc_id

    win._file_dock._focus(("doc", parent))
    assert _shown(win) == {parent}
    assert win._session.has_document(seg_id)


# ================================================================================================
# 6. multi-document analyses see every OPEN recording, shown or not
# ================================================================================================

def _install_detection(win: MainWindow, doc_id: str) -> None:
    """A minimal stored detection, so voltage dependence has something per recording.
    Built on the FROZEN 20-column schema via ``empty_df`` — never a hand-shaped frame."""
    from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
    from nanoscope.core.analysis import AnalysisResult

    df = empty_df()
    for i in range(4):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": 1e-3,
            "baseline_pA": -1000.0, "depth_pA": 20.0, "frac_depth": 0.02,
            "ecd_pA_s": 0.05, "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        }
    df = df.astype({c: d for c, (d, _, _) in SCHEMA.items()})
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0),
        payload=EventSet(df, {"analyzed_duration_s": 1.0, "recording": "rec"})))


def test_batch_detect_targets_every_open_recording(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    c = _open(win, _doc(tmp_path, name="c.dat"))
    assert _shown(win) == {c}
    assert {doc_id for (doc_id, _ch, _name) in win._detectable_targets()} == {a, b, c}


def test_iv_sees_every_open_recording(win, qtbot, tmp_path):
    ids = [_open(win, _doc(tmp_path, name=f"v{i}.dat", with_voltage=True, nominal_mV=mV))
           for i, mV in enumerate((-200.0, -500.0, -1000.0))]
    assert _shown(win) == {ids[-1]}
    win._open_iv()
    qtbot.addWidget(win._iv_window)
    assert len(win._iv_window.points()) == len(ids)


def test_the_noise_spectrum_window_sees_every_open_recording(win, qtbot, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b}
    win._open_psd()
    qtbot.addWidget(win._psd_window)
    assert {r["doc_id"] for r in win._psd_window._gather_recordings()} == {a, b}


def test_voltage_dependence_sees_every_open_recording(win, qtbot, tmp_path):
    from nanoscope.analysis.voltage_dep import vdep_points

    ids = [_open(win, _doc(tmp_path, name=f"w{i}.dat", with_voltage=True, nominal_mV=mV))
           for i, mV in enumerate((-200.0, -500.0, -1000.0))]
    for doc_id in ids:
        _install_detection(win, doc_id)
    assert _shown(win) == {ids[-1]}
    names = {p.recording for p in vdep_points(win._session, metric="median_depth")}
    # display_name, not Path(doc_id).name — a doc_id is an opaque per-load id, not a path.
    assert names == {win._session.document(d).display_name for d in ids}


def test_a_detection_survives_being_switched_away_from(win, tmp_path):
    """"Not drawn" must not cost the user any WORK. The results store is keyed by document, so
    a detection on the recording you stepped away from is still there when you step back."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    _install_detection(win, a)
    assert win._session.results.get(a, 0, "detect.schmitt") is not None

    _open(win, _doc(tmp_path, name="b.dat"))
    assert win._session.results.get(a, 0, "detect.schmitt") is not None

    win._file_dock._focus(("doc", a))
    assert win._analysis_panel.current_detection() is not None
    assert win._act_export_events.isEnabled()


def test_manual_measurement_markers_follow_the_shown_recording(win, tmp_path):
    """The saved-measurement markers are drawn onto panes, and a switch destroys and rebuilds
    every pane. They must come and go with their recording, without complaint."""
    a = _open(win, _doc(tmp_path, name="a.dat"))
    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    vals = win._session.x_axis(a).values
    win._trace_view.set_measure_region(float(vals[2_000]), float(vals[6_000]))
    QtWidgets.QApplication.processEvents()
    assert mp.add_current() is True
    assert len(mp.measurements_df) == 1

    b = _open(win, _doc(tmp_path, name="b.dat"))     # rebuilds every pane
    assert _shown(win) == {b}
    assert len(mp.measurements_df) == 1              # the measurement itself is untouched

    win._file_dock._focus(("doc", a))
    assert _shown(win) == {a}
    assert len(mp.measurements_df) == 1


# ================================================================================================
# 7. closing never leaves the viewer blank while recordings are listed
# ================================================================================================

def test_closing_the_shown_recording_falls_back_to_a_survivor(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    assert _shown(win) == {b}

    win._close_document(b)
    assert _shown(win) == {a}, "closing the shown recording left the viewer blank"
    assert _listed(win) == {a}


def test_closing_a_hidden_recording_does_not_disturb_the_view(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    b = _open(win, _doc(tmp_path, name="b.dat"))
    win._close_document(a)
    assert _shown(win) == {b}


def test_closing_the_last_recording_shows_nothing(win, tmp_path):
    a = _open(win, _doc(tmp_path, name="a.dat"))
    win._close_document(a)
    assert _shown(win) == set()
    assert win._trace_view._panes == []

# ================================================================================================
# owner note #1: pane order is normalised — current on top, no matter the file's channel order
# ================================================================================================

def test_plan_panes_puts_current_above_voltage_regardless_of_file_order():
    """One recording put current above voltage and another the reverse (the files disagree
    about channel order), so the eye had to re-find the signal on every open. plan_panes
    now normalises: CURRENT first, then conductance, voltage, raw voltage, unknown —
    stable within each kind, so a deliberately arranged overlay keeps its own ordering."""
    from nanoscope.gui.trace_view import PaneSpec, _current_first
    from nanoscope.core.types import ChannelKind

    voltage_first = [
        PaneSpec(kind=ChannelKind.VOLTAGE, units="mV", layer_ids=("v",)),
        PaneSpec(kind=ChannelKind.CURRENT, units="pA", layer_ids=("i1",)),
        PaneSpec(kind=ChannelKind.CURRENT, units="pA", layer_ids=("i2",)),
        PaneSpec(kind=ChannelKind.UNKNOWN, units="", layer_ids=("x",)),
    ]
    ordered = _current_first(voltage_first)
    assert [s.kind for s in ordered] == [
        ChannelKind.CURRENT, ChannelKind.CURRENT, ChannelKind.VOLTAGE, ChannelKind.UNKNOWN]
    # ...and STABLY: i1 stays above i2.
    assert [s.layer_ids for s in ordered[:2]] == [("i1",), ("i2",)]
