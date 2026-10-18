"""Change B — curation throughput in the Event inspector.

Advance after decision (keyboard only, off by default), the Unreviewed / Unlabelled /
Labelled filters, ⌘U to the first unreviewed event, the progress line, the Label row and
⌘Z — each pinned on a standalone window over a real detection.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import FLAG_CLIPPED, REVIEW_COL, EventSet, reviewed_mask
from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import event_inspector as insp_mod
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.event_inspector import EventInspectorWindow

FS = 100_000.0
N_EVENTS = 12
_SETTINGS_KEY = "curation/advance_after_decision"


@pytest.fixture(autouse=True)
def _advance_setting_is_clean():
    """The session conftest clears QSettings once; a leaked True would make other tests'
    A/R move on. Removed before and after every test here."""
    QtCore.QSettings().remove(_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove(_SETTINGS_KEY)


def _trace(n_events: int = N_EVENTS) -> np.ndarray:
    n = 20_000 * (n_events + 1)
    rng = np.random.default_rng(0)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for k in range(n_events):
        s = 10_000 + 20_000 * k
        x[s:s + 300] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _session(tmp_path, n_events: int = N_EVENTS):
    arr = _trace(n_events)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "b.dat", format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    df, prov = detect_events(arr, FS, min_dwell_s=5e-4)
    assert len(df) == n_events
    return session, doc_id, EventSet(df, prov)


def _vocab(*tokens: str) -> LabelVocabulary:
    v = LabelVocabulary()
    for t in tokens:
        k, _, val = t.partition("=")
        v.add(k, val)
    return v


@pytest.fixture
def insp(qapp, qtbot, tmp_path):
    session, doc_id, events = _session(tmp_path)
    w = EventInspectorWindow(session)
    qtbot.addWidget(w)
    w.set_events(doc_id, 0, events.df)
    w._doc_for_test = doc_id
    return w


def _key(w, key, text="", mods=Qt.NoModifier, *, repeat=False) -> bool:
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods, text, repeat)
    return w._handle_own_key(ev)


def _ids(w) -> list[int]:
    return [int(i) for i in w._df["event_id"]]


# ---------------------------------------------------------------------------------------------
# Advance after decision
# ---------------------------------------------------------------------------------------------
def test_advance_is_off_by_default_and_keys_do_not_navigate(insp):
    assert insp.advance_after_decision() is False
    assert not insp._advance_check.isChecked()
    assert insp._advance_check.objectName() == "advanceAfterDecision"
    assert insp._advance_check.focusPolicy() == Qt.NoFocus
    ids = _ids(insp)
    insp.show_event(ids[2])
    assert _key(insp, Qt.Key_R, "r") is True
    assert insp.current_event_id() == ids[2]
    assert not bool(insp._df["accepted"].iloc[2])


def test_with_advance_on_r_writes_then_moves_to_the_next_event_of_the_filter(insp):
    insp.set_advance_after_decision(True)
    ids = _ids(insp)
    insp.show_event(ids[2])
    seen: list[int] = []
    insp.eventSelected.connect(seen.append)
    assert _key(insp, Qt.Key_R, "r")
    assert seen == [ids[3]] and insp.current_event_id() == ids[3]
    assert not bool(insp._df["accepted"].iloc[2])
    assert bool(reviewed_mask(insp._df)[2])
    # Space and A are the same door.
    assert insp.decide_toggle() is True and insp.current_event_id() == ids[4]
    assert not bool(insp._df["accepted"].iloc[3])


def test_under_unreviewed_a_decision_moves_to_the_next_unreviewed_and_skips_reviewed_ones(insp):
    ids = _ids(insp)
    from nanoscope.analysis.events import mark_reviewed
    mark_reviewed(insp._df, rows=np.array([2, 3]))
    insp.set_filter("Unreviewed")
    insp.show_event(ids[1])
    n_before = insp.event_count()
    insp.set_advance_after_decision(True)
    assert insp.decide_accept(True) is True
    assert insp.current_event_id() == ids[4]
    assert insp.event_count() == n_before - 1


def test_advance_at_the_end_stays_and_the_progress_line_says_why(insp):
    ids = _ids(insp)
    insp.set_advance_after_decision(True)
    insp.show_event(ids[-1])
    assert insp.decide_accept(False) is False
    assert insp.current_event_id() == ids[-1]
    assert insp.progress_text().startswith("End of All")
    assert "last event" in insp._progress.toolTip()
    # Under Unreviewed with earlier members left, the note counts them and names ⌘U.
    insp.set_filter("Unreviewed")
    insp.show_event(ids[-2])
    assert insp.decide_accept(False) is False      # ids[-1] was reviewed above
    text = insp.progress_text()
    assert text.startswith(f"End of Unreviewed · {N_EVENTS - 2} earlier"), text
    assert insp_mod._native_key("Ctrl+U") in text
    # The next navigation retires the note, and so does another surface's write.
    insp.prev_event()
    assert not insp.progress_text().startswith("End of")
    insp.show_event(ids[-1])
    insp.decide_accept(False)
    assert insp.progress_text().startswith("End of")
    insp.reflect_acceptance(ids[0], True)
    assert insp.progress_text().startswith("Reviewed")


def test_a_refused_accept_does_not_advance(insp):
    ids = _ids(insp)
    df = insp._df
    df.iat[5, df.columns.get_loc("flag")] = FLAG_CLIPPED
    df.iat[5, df.columns.get_loc("accepted")] = False
    insp.set_advance_after_decision(True)
    insp.show_event(ids[5])
    assert insp.decide_accept(True) is False
    assert insp.current_event_id() == ids[5]
    assert "kept rejected" in insp.refusal_note()


def test_a_label_key_advances_and_writes_one_label_and_one_review(insp):
    ids = _ids(insp)
    insp.set_vocabulary(_vocab("type=collision", "type=junk"))
    insp.set_advance_after_decision(True)
    insp.show_event(ids[0])
    emitted: list[int] = []
    insp.labelsChanged.connect(lambda: emitted.append(1))
    assert _key(insp, Qt.Key_2, "2")
    assert emitted == [1]
    assert insp._df["flag"].iloc[0] == "type=junk"
    assert insp.current_event_id() == ids[1]
    # An unarmed digit writes nothing and stays.
    assert insp.decide_label("7") is False and insp.current_event_id() == ids[1]


def test_buttons_checkbox_and_label_row_never_advance_even_when_on(insp):
    ids = _ids(insp)
    insp.set_vocabulary(_vocab("type=collision"))
    insp.set_advance_after_decision(True)
    insp.show_event(ids[3])
    insp._reject_btn.click()
    insp._accept_btn.click()
    insp._accept_check.setChecked(False)
    insp._label_apply_btn.click()
    insp._label_clear_btn.click()
    assert insp.current_event_id() == ids[3]
    assert not bool(insp._df["accepted"].iloc[3])


def test_an_auto_repeated_decision_key_is_consumed_and_ignored(insp):
    ids = _ids(insp)
    insp.set_advance_after_decision(True)
    insp.show_event(ids[1])
    assert _key(insp, Qt.Key_R, "r", repeat=True) is True
    assert insp.current_event_id() == ids[1]
    assert bool(insp._df["accepted"].iloc[1])
    assert REVIEW_COL not in insp._df.columns or not reviewed_mask(insp._df).any()
    # Navigation keeps repeating.
    assert _key(insp, Qt.Key_Right, repeat=True) and insp.current_event_id() == ids[2]


def test_the_advance_choice_is_shared_by_both_windows_and_remembered(qapp, qtbot, insp):
    insp.set_advance_after_decision(True)
    assert QtCore.QSettings().value(_SETTINGS_KEY) in (True, "true")
    other = EventInspectorWindow(insp._session)
    qtbot.addWidget(other)
    assert other.advance_after_decision() is True and other._advance_check.isChecked()
    gal = EventGalleryWindow(insp._session)
    qtbot.addWidget(gal)
    assert gal.advance_after_decision() is True
    gal.set_advance_after_decision(False)
    # The inspector re-reads the one key before acting on it.
    assert insp.advance_after_decision() is False and not insp._advance_check.isChecked()


# ---------------------------------------------------------------------------------------------
# Filters and First unreviewed
# ---------------------------------------------------------------------------------------------
def test_unreviewed_and_label_filters_subset_the_browsed_events(insp):
    insp.set_filter("Unreviewed")
    assert insp.current_filter() == "Unreviewed" and insp.event_count() == N_EVENTS
    assert "Unlabelled" not in insp.filter_options()
    insp.set_vocabulary(_vocab("type=collision", "type=rejected-noise"))
    assert insp.filter_options()[-3:] == (
        "Unlabelled", "Labelled: type=collision", "Labelled: type=rejected-noise")
    combo = [insp._filter_combo.itemText(i) for i in range(insp._filter_combo.count())]
    assert "Labelled: type=collision" in combo
    insp.set_filter("label:type=collision")
    assert insp.current_filter() == "Labelled: type=collision"
    assert insp._filter_combo.currentText() == "Labelled: type=collision"
    insp.set_filter("Labelled: type=rejected-noise")
    assert insp.current_filter() == "Labelled: type=rejected-noise"   # not "Rejected only"
    # A label filter nobody offers is still selectable and stays offered while browsed.
    insp.set_filter("Labelled: molecule=apo")
    assert insp._filter_combo.currentText() == "Labelled: molecule=apo"
    insp.set_filter("unlabelled")
    assert insp.current_filter() == "Unlabelled" and insp.event_count() == N_EVENTS


def test_the_filter_combo_keeps_its_width_when_label_filters_arrive(insp):
    w0 = insp._filter_combo.sizeHint().width()
    insp.set_vocabulary(_vocab("molecule=apo-ferritin-with-a-very-long-name", "type=x"))
    assert insp._filter_combo.sizeHint().width() == w0
    assert insp._filter_combo.view().minimumWidth() > w0


def test_first_unreviewed_jumps_within_the_filter_and_reports_when_none(insp):
    ids = _ids(insp)
    from nanoscope.analysis.events import mark_reviewed
    mark_reviewed(insp._df, rows=np.array([0, 1, 2]))
    insp.show_event(ids[6])
    assert _key(insp, Qt.Key_U, "", Qt.ControlModifier)
    assert insp.current_event_id() == ids[3]
    # Accepted only, with only rejected events left unreviewed: say how many remain.
    df = insp._df
    mark_reviewed(df, rows=np.arange(3, N_EVENTS - 2))
    df.iloc[-2:, df.columns.get_loc("accepted")] = False
    insp.set_filter("Accepted only")
    assert insp.first_unreviewed() is None
    assert insp.progress_text().startswith("None unreviewed here · 2 elsewhere")
    assert "Filter ▸ Unreviewed" in insp._progress.toolTip()
    mark_reviewed(df)
    insp.set_filter("All")
    assert insp.first_unreviewed() is None
    assert insp.progress_text().startswith("Every event reviewed")


# ---------------------------------------------------------------------------------------------
# The progress line and the Label row
# ---------------------------------------------------------------------------------------------
def _geometry(w, *, with_readout: bool = True) -> dict:
    """What the user's pointer could be over. The readout box (and the plot above it) are
    left out for label writes only: a labels line in the readout is the readout's own
    pre-existing growth, not something the progress line or the Label row causes."""
    out = {"min_width": w.minimumSizeHint().width()}
    names = ["_next_btn", "_reject_btn", "_advance_check", "_label_combo",
             "_label_apply_btn", "_first_unreviewed_btn", "_progress", "_curation_status"]
    if with_readout:
        names.append("_plot_widget")
        out["box"] = w._readout.parentWidget().geometry().getRect()
        out["min"] = w.minimumSizeHint().toTuple()
    for name in names:
        widget = getattr(w, name)
        out[name] = (widget.mapTo(w, QtCore.QPoint(0, 0)).toTuple(), widget.size().toTuple())
    return out


def test_the_progress_line_follows_every_write_and_moves_nothing(qtbot, insp):
    insp.set_vocabulary(_vocab("type=collision", "type=junk"))
    insp.resize(1200, 860)
    insp.show()
    qtbot.wait(50)
    ids = _ids(insp)
    insp.show_event(ids[1])
    qtbot.wait(20)
    before = _geometry(insp)
    before_rows = _geometry(insp, with_readout=False)
    texts = [insp.progress_text()]
    assert texts[0] == "Reviewed 0 / 12 · accepted 12 · labels: none yet"
    for step, labels in ((lambda: insp.decide_accept(False), False),
                         (lambda: insp.decide_label("1"), True),
                         (lambda: insp.clear_label_current("type"), True),
                         (insp.accept_all, False),
                         (lambda: insp.set_vocabulary(
                             _vocab("type=collision", "type=junk", "type=z")), False)):
        step()
        qtbot.wait(20)
        texts.append(insp.progress_text())
        if labels:
            assert _geometry(insp, with_readout=False) == before_rows
        else:
            assert _geometry(insp) == before
    assert texts[1:] == [
        "Reviewed 1 / 12 · accepted 11 · labels: none yet",
        "Reviewed 1 / 12 · accepted 11 · labels: type=collision 1",
        "Reviewed 1 / 12 · accepted 11 · labels: none yet",
        "Reviewed 12 / 12 · accepted 12 · labels: none yet",
        "Reviewed 12 / 12 · accepted 12 · labels: none yet"], texts
    insp.decide_label("3")                       # the palette grew: its third key is armed
    assert insp.progress_text().endswith("labels: type=z 1")
    assert insp._progress.objectName() == "curationProgress"
    assert insp._curation_status.objectName() == "curationStatus"
    assert "whole detection" in insp._progress.toolTip()


def test_the_label_row_applies_and_clears_through_the_shared_writers(monkeypatch, insp):
    calls: list[str] = []
    real_label, real_unlabel = insp_mod.label_rows, insp_mod.unlabel_rows
    monkeypatch.setattr(insp_mod, "label_rows",
                        lambda *a, **k: calls.append("label") or real_label(*a, **k))
    monkeypatch.setattr(insp_mod, "unlabel_rows",
                        lambda *a, **k: calls.append("unlabel") or real_unlabel(*a, **k))
    insp.set_vocabulary(_vocab("type=collision", "molecule=apo"))
    ids = _ids(insp)
    insp.show_event(ids[4])
    emitted: list[int] = []
    insp.labelsChanged.connect(lambda: emitted.append(1))
    assert insp._label_combo.isEnabled() and insp._label_combo.count() == 2
    insp._label_combo.setCurrentIndex(1)
    insp._label_apply_btn.click()
    assert insp._df["flag"].iloc[4] == "molecule=apo"
    assert "labels:          molecule=apo" in insp._readout.text()
    insp._label_combo.setCurrentIndex(0)
    insp._label_apply_btn.click()
    assert insp._df["flag"].iloc[4] == "molecule=apo;type=collision"
    insp._label_combo.setCurrentIndex(1)
    insp._label_clear_btn.click()
    assert insp._df["flag"].iloc[4] == "type=collision"
    assert calls == ["label", "label", "unlabel"] and emitted == [1, 1, 1]
    assert bool(reviewed_mask(insp._df)[4])
    assert insp.current_event_id() == ids[4]


def test_the_label_row_is_disabled_without_a_palette_or_an_event(qapp, qtbot, tmp_path):
    session, doc_id, events = _session(tmp_path)
    w = EventInspectorWindow(session)
    qtbot.addWidget(w)
    assert not w._label_apply_btn.isEnabled()
    assert w._label_row.isHidden()                     # no event: hidden, like Accept/Reject
    w.set_events(doc_id, 0, events.df)
    assert not w._label_row.isHidden()
    assert not w._label_apply_btn.isEnabled()          # no palette yet
    w.set_vocabulary(_vocab("type=a"))
    assert w._label_apply_btn.isEnabled() and w._label_clear_btn.isEnabled()


def test_the_label_row_and_progress_do_not_grow_the_window(qapp, qtbot, tmp_path):
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        session, doc_id, events = _session(tmp_path, 5)
        w = EventInspectorWindow(session)
        qtbot.addWidget(w)
        w.set_events(doc_id, 0, events.df)
        base = w.minimumSizeHint()
        for n in (9, 20):
            w.set_vocabulary(_vocab(*[f"molecule=value-{i}-with-a-sixty-character-label-"
                                      f"value-xxxxxxxxxxxx" for i in range(n)]))
            assert w.minimumSizeHint() == base
        assert base.width() <= 1100
        # The Label row lives in the Event box, not in a root row of its own: the waveform
        # keeps its 300 px floor (a root row once took 38 px of it).
        assert w._plot_widget.minimumHeight() == insp_mod._PLOT_MIN_H == 300
        # Updated (review B-1): the row is the readout's own bottom-right corner, so no
        # readout line can set its x or the window's width.
        assert w._label_row.parentWidget() is w._readout
        assert w._curation_status.parentWidget() is w.statusBar()
    finally:
        qapp.setFont(old)


def test_labelling_at_the_minimum_width_neither_widens_the_window_nor_moves_apply(
        qapp, qtbot, tmp_path):
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        session, doc_id, events = _session(tmp_path, 5)
        w = EventInspectorWindow(session)
        qtbot.addWidget(w)
        w.set_events(doc_id, 0, events.df)
        w.set_vocabulary(_vocab("molecule=apo-ferritin", "molecule=holo-ferritin",
                                "type=collision"))
        w.resize(w.minimumSizeHint().width(), 900)
        w.show()
        qtbot.wait(50)
        min_w = w.minimumSizeHint().width()
        ids = _ids(w)
        w.show_event(ids[1])
        qtbot.wait(20)
        spots = [b.mapTo(w, QtCore.QPoint(0, 0)).toTuple()
                 for b in (w._label_apply_btn, w._label_clear_btn)]
        w._label_combo.setCurrentIndex(1)
        w._label_apply_btn.click()
        w._label_combo.setCurrentIndex(2)
        w._label_apply_btn.click()
        qtbot.wait(30)
        assert "molecule=holo-ferritin, type=collision" in w._readout.text()
        assert w.minimumSizeHint().width() == min_w
        assert [b.mapTo(w, QtCore.QPoint(0, 0)).toTuple()
                for b in (w._label_apply_btn, w._label_clear_btn)] == spots
    finally:
        qapp.setFont(old)


def test_a_label_filter_and_a_label_entry_keep_their_value_on_a_narrow_face(qtbot, insp):
    insp.show()
    qtbot.wait(30)
    insp.set_vocabulary(_vocab("molecule=apo-ferritin-from-horse-spleen",
                               "molecule=holo-ferritin"))
    insp.set_filter("Labelled: molecule=apo-ferritin-from-horse-spleen")
    assert insp._filter_combo.currentText() == "Labelled: molecule=apo-ferritin-from-horse-spleen"
    # Updated (review B-CLIP-2): a value too long for the face is cut in the middle, so the
    # face fits and both ends of the value stay; it never starts with the prefix.
    face = insp._filter_combo.shown_text()
    assert face.startswith("apo-") and face.endswith("spleen") and "…" in face, face
    assert _face_fits(insp._filter_combo)
    insp.set_filter("Unreviewed")
    assert insp._filter_combo.shown_text() == "Unreviewed"
    # The hotkey survives in every form, the middle-cut last one included.
    insp._label_combo.setCurrentIndex(1)
    face = insp._label_combo.shown_text()
    assert face.endswith("[2]") and "holo-ferritin" in face, face
    insp._label_combo.setCurrentIndex(0)
    face = insp._label_combo.shown_text()
    assert face.endswith("[1]") and "…" in face and _face_fits(insp._label_combo), face


def _face_fits(combo) -> bool:
    opt = QtWidgets.QStyleOptionComboBox()
    combo.initStyleOption(opt)
    field = combo.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt,
                                         QtWidgets.QStyle.SubControl.SC_ComboBoxEditField, combo)
    return combo.fontMetrics().horizontalAdvance(combo.shown_text()) <= field.width()


def _real_font_window(qapp, qtbot, tmp_path, n_events: int = 5):
    from nanoscope.gui.app import _apply_theme
    _apply_theme(qapp)
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    session, doc_id, events = _session(tmp_path, n_events)
    w = EventInspectorWindow(session)
    qtbot.addWidget(w)
    w.set_events(doc_id, 0, events.df)
    return w


def _spots(w) -> list:
    return [(b.mapTo(w, QtCore.QPoint(0, 0)).toTuple(), b.size().toTuple())
            for b in (w._label_combo, w._label_apply_btn, w._label_clear_btn, w._merge_next_btn)]


def test_long_labels_and_a_long_label_filter_never_widen_the_window_or_move_the_label_row(
        qapp, qtbot, tmp_path):
    """Review B-1 / B-GEO-1: the readout's longest line used to set the window's minimum
    width, so a three-label line or "filtered out of Labelled: …" grew the window and slid
    Apply / Clear under the pointer that had just clicked them."""
    old = qapp.font()
    try:
        w = _real_font_window(qapp, qtbot, tmp_path)
        w.set_vocabulary(_vocab("molecule=apo-ferritin-from-horse-spleen",
                                "molecule=holo-ferritin", "type=translocation",
                                "quality=needs-second-look"))
        w.resize(w.minimumSizeHint().width(), 900)
        w.show()
        qtbot.wait(50)
        ids = _ids(w)
        w.show_event(ids[1])
        qtbot.wait(20)
        size, min_w, spots = w.size(), w.minimumSizeHint().width(), _spots(w)
        for ch in "234":
            w.decide_label(ch)
        df = w._df
        df.iat[1, df.columns.get_loc("flag")] = (
            "molecule=apo-ferritin-from-horse-spleen|holo-ferritin;type=translocation;"
            "quality=needs-second-look;note=double-peak-shoulder")
        w._refresh_readout(1)
        qtbot.wait(30)
        line = next(l for l in w._readout.text().splitlines() if l.startswith("labels:"))
        assert len(line) >= 100
        assert (w.size(), w.minimumSizeHint().width(), _spots(w)) == (size, min_w, spots)
        shown = [l for l in QtWidgets.QLabel.text(w._readout).splitlines()
                 if l.startswith("labels:")][0]
        assert shown.endswith("…") and line in w._readout.toolTip()
        # A long label filter on an event that leaves it: the header names the filter.
        w.set_filter("Labelled: molecule=holo-ferritin")
        w.show_event(ids[1])
        w._label_combo.setCurrentIndex(1)
        w._label_clear_btn.click()                  # it leaves the filter, stays on screen
        qtbot.wait(30)
        assert "filtered out of Labelled: molecule=holo-ferritin" in w._readout.text()
        assert (w.size(), w.minimumSizeHint().width(), _spots(w)) == (size, min_w, spots)
        w.show_event(ids[2])                       # a plain event: nothing to give back
        qtbot.wait(20)
        assert (w.size(), _spots(w)) == (size, spots)
    finally:
        qapp.setFont(old)


def test_the_first_label_at_the_default_height_moves_nothing(qapp, qtbot, tmp_path):
    """The labels line comes and goes with a label write; its height is held, so at the
    window's minimum height (where it opens) Apply does not drop under the pointer."""
    old = qapp.font()
    try:
        w = _real_font_window(qapp, qtbot, tmp_path)
        w.set_vocabulary(_vocab("type=collision", "type=junk"))
        w.resize(w.minimumSizeHint())
        w.show()
        qtbot.wait(50)
        size, spots = w.size(), _spots(w)
        lines = w._readout.text().count("\n")
        w._label_apply_btn.click()
        qtbot.wait(30)
        assert w._readout.text().count("\n") == lines + 1       # the labels line arrived
        assert (w.size(), _spots(w)) == (size, spots)
        w._label_clear_btn.click()
        qtbot.wait(30)
        assert (w.size(), _spots(w)) == (size, spots)
    finally:
        qapp.setFont(old)


def test_a_cut_readout_line_keeps_its_stale_mark(qapp):
    fm = QtGui.QFontMetrics(qapp.font())
    cut = insp_mod._ReadoutLabel._elide(fm, "dwell:  " + "9" * 120 + insp_mod._STALE_MARK, 200)
    assert cut.endswith("…" + insp_mod._STALE_MARK) and fm.horizontalAdvance(cut) <= 200
    assert insp_mod._ReadoutLabel._elide(fm, "short", 200) == "short"


def test_the_counter_is_sized_for_its_widest_count(qapp, qtbot, tmp_path):
    """Review B-2: digits are proportional, so "of 10" is wider than "of 11"; a width sized
    for the frame's own count clipped the last digit of a smaller one."""
    old = qapp.font()
    try:
        w = _real_font_window(qapp, qtbot, tmp_path, 11)
        w.show()
        qtbot.wait(20)
        lab, fm = w._count_label, w._count_label.fontMetrics()
        assert all(lab.width() >= fm.horizontalAdvance(f"of {k}") for k in range(1, 12))
        w.set_accepted_current(False)
        w.set_filter("Accepted only")
        w.next_event()
        assert lab.text() == "of 10" and lab.width() >= fm.horizontalAdvance("of 10")
    finally:
        qapp.setFont(old)


def test_the_counter_widths_never_change_while_the_subset_shrinks(qtbot, tmp_path, qapp):
    session, doc_id, events = _session(tmp_path, 12)
    w = EventInspectorWindow(session)
    qtbot.addWidget(w)
    w.set_events(doc_id, 0, events.df)
    w.resize(1200, 860)
    w.show()
    qtbot.wait(30)
    w.set_filter("Unreviewed")
    w.set_advance_after_decision(True)
    xs = set()
    for _ in range(11):                                 # 12 -> 1 unreviewed: 2 digits -> 1
        xs.add((w._next_btn.x(), w._reject_btn.x()))
        w.decide_accept(True)
        qtbot.wait(5)
    xs.add((w._next_btn.x(), w._reject_btn.x()))
    assert len(xs) == 1, xs


# ---------------------------------------------------------------------------------------------
# Keep place across undo / redo, and ⌘Z
# ---------------------------------------------------------------------------------------------
def test_an_undo_refeed_of_the_same_events_keeps_filter_and_event(insp):
    ids = _ids(insp)
    insp.set_filter("Unreviewed")
    insp.show_event(ids[7])
    doc = insp._doc_for_test
    insp.set_events(doc, 0, insp._df.copy(deep=True), keep_place=True)
    assert (insp.current_filter(), insp.current_event_id()) == ("Unreviewed", ids[7])
    # Without the hint (a new detection), today's reset.
    insp.set_events(doc, 0, insp._df.copy(deep=True))
    assert (insp.current_filter(), insp.current_event_id()) == ("All", ids[0])


def test_keep_place_refuses_a_frame_whose_ids_were_renumbered(insp):
    ids = _ids(insp)
    insp.show_event(ids[5])
    merged = insp._df.drop(index=insp._df.index[4]).reset_index(drop=True)
    merged["event_id"] = np.arange(len(merged))
    insp.set_events(insp._doc_for_test, 0, merged, keep_place=True)
    assert insp.current_event_id() == 0 and insp.current_filter() == "All"


def test_ctrl_z_in_the_inspector_asks_for_undo_and_returns_to_the_decided_event(insp):
    ids = _ids(insp)
    doc = insp._doc_for_test
    snapshot = insp._df.copy(deep=True)

    def fake_undo():                                    # what MainWindow's undo re-feed does
        insp.set_events(doc, 0, snapshot.copy(deep=True), keep_place=True)

    asked: list[str] = []
    insp.undoRequested.connect(lambda: asked.append("undo"))
    insp.undoRequested.connect(fake_undo)
    insp.redoRequested.connect(lambda: asked.append("redo"))
    insp.set_advance_after_decision(True)
    insp.show_event(ids[5])
    insp.decide_accept(False)
    assert insp.current_event_id() == ids[6]
    assert _key(insp, Qt.Key_Z, "", Qt.ControlModifier)
    assert asked == ["undo"]
    assert insp.current_event_id() == ids[5] and bool(insp._df["accepted"].iloc[5])
    assert _key(insp, Qt.Key_Z, "", Qt.ControlModifier | Qt.ShiftModifier)
    assert asked == ["undo", "redo"]
    # A second ⌘Z with nothing advanced is a plain request: no view move.
    insp.show_event(ids[8])
    assert _key(insp, Qt.Key_Z, "", Qt.ControlModifier)
    assert insp.current_event_id() == ids[8]


def test_ctrl_z_after_a_no_op_decision_never_jumps_to_an_event_the_undo_left_alone(insp):
    """R on an event already rejected writes nothing, so the last checkpoint is another
    event's. ⌘Z reverts THAT one; the window must not jump back to the no-op event, where
    the undo did nothing, and hide what it did revert."""
    ids = _ids(insp)
    doc = insp._doc_for_test
    insp.set_advance_after_decision(True)
    insp.show_event(ids[5])
    insp.decide_accept(False)                            # a real write on 5
    insp.show_event(ids[8])
    before_8 = insp._df.copy(deep=True)
    insp.decide_accept(False)                            # a real write on 8: the last step
    insp.show_event(ids[5])
    assert insp.decide_accept(False) is True             # no-op on 5, still advances
    assert insp.current_event_id() == ids[6]
    insp.undoRequested.connect(
        lambda: insp.set_events(doc, 0, before_8.copy(deep=True), keep_place=True))
    assert _key(insp, Qt.Key_Z, "", Qt.ControlModifier)
    assert bool(insp._df["accepted"].iloc[8]) and not bool(insp._df["accepted"].iloc[5])
    assert insp.current_event_id() == ids[6]


def test_ctrl_z_that_installs_no_new_frame_moves_nothing(insp):
    """A measurements undo (or nothing to undo) leaves the frame object in place."""
    ids = _ids(insp)
    insp.set_advance_after_decision(True)
    insp.show_event(ids[2])
    insp.decide_accept(False)
    assert insp.current_event_id() == ids[3]
    assert _key(insp, Qt.Key_Z, "", Qt.ControlModifier)
    assert insp.current_event_id() == ids[3]


def test_every_new_key_is_a_widget_with_children_shortcut_in_the_registry(insp):
    by_key = {sc.key().toString(): sc for sc in insp._shortcuts}
    for seq in ("Ctrl+U", "Ctrl+Z", "Ctrl+Shift+Z"):
        assert seq in by_key, sorted(by_key)
        assert by_key[seq].context() == Qt.WidgetWithChildrenShortcut
    for seq in ("A", "R", "Space", "1", "9", "Ctrl+U"):
        assert by_key[seq].autoRepeat() is False, seq
    for seq in ("Left", "Right"):
        assert by_key[seq].autoRepeat() is True, seq
    # No new bare letter was reserved: J, K and U stay free for label hotkeys.
    assert not {"J", "K", "U"} & set(by_key)
