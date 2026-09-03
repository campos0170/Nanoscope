"""WA-WIRE — the four seams that only ``nanoscope/gui/main_window.py`` can close.

Four fixes landed in four other files this wave, and each of them ended at a public seam
whose other half is a connection in the sole wirer. A seam that is never connected is worse
than no seam: the source says the feature exists, the screen says it does not, and this
project has lost three rounds to exactly that. So each wire below is asserted through the
GESTURE, never through a ``connect`` line.

Measured before the wiring, on a real ``MainWindow`` at 1512x900 driven with QTest against
the owner's -1000 mV apo-ferritin recording (110 events, scratchpad/wave-a/wire/before.log):

* Analysis ▸ Run detection over 10 hand rejections: ``confirm_replacing_curation called:
  False``, a run started, the curation was gone. The Detect BUTTON asked; the menu item —
  the same gesture, one pixel of chrome away — did not.
* ``willReplaceEvents receivers = 0``. Commit at 110 accepted, reject 20 rows in place
  without announcing, re-run: Edit ▸ Undo read "Undo Re-run detection", and pressing it
  gave back 110 accepted. The 20 rejections were gone and the menu had claimed to be
  walking the run back.
* ``undoRequested receivers = 0``. Ctrl+A in the gallery took 35 tiles, R took accepted
  110 → 75, this window's Edit menu read "Undo Curate 35 events" ENABLED — and Ctrl+Z on
  the gallery's view, Ctrl+Z on the gallery window, and a click on the gallery's own Undo
  button all left it at 75. The gallery has no menu bar; the shortcut belongs to the window
  behind it.
* Schmitt (110) and CUSUM (250) stored on one channel with the panel showing CUSUM: the
  Plots source flagged ``current`` was Schmitt's, and ``MainWindow.active_detection_key``
  did not exist.

The same script after the wiring: all four True (scratchpad/wave-a/wire/after.log).

These tests fail against the pre-fix file, which is loaded at runtime by the out-of-tree
plugin ``scratchpad/wave-a/wire/prefix/wa_wire_prefix.py`` — no source is edited back.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, stamp_original_bounds
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400), (50_000, 50_700)]
SCHMITT = "detect.schmitt"
CUSUM = "detect.cusum"


# ================================================================================================
# builders — synthetic, so the whole file is seconds rather than minutes
# ================================================================================================

def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


def _install(window, tmp_path):
    sig = _signal()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa_wire.dat", format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(sig,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, sig


def _detect_into(window, doc_id, signal, module_id=SCHMITT, min_dwell_s=1e-4):
    """Store a detection the way ``analysis/modules.py`` does — original bounds stamped."""
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=min_dwell_s)
    events = EventSet(stamp_original_bounds(df), prov)
    window._session.add_result(AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id=doc_id, channel=0,
        params={"detector": module_id},
        calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        payload=events))
    return events


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    w.resize(1512, 900)          # the owner's display; a control off-viewport is not wired
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, tmp_path):
    """One recording, one stored Schmitt detection, the panel showing it."""
    doc_id, sig = _install(window, tmp_path)
    _detect_into(window, doc_id, sig)
    panel = window._analysis_panel
    assert panel.current_detection() is not None
    return window, doc_id, sig, panel


def _accepted(panel) -> int:
    df = panel.current_detection().payload.df
    return int(df["accepted"].to_numpy(dtype=bool).sum())


# ================================================================================================
# W3 — Analysis ▸ Run detection is a person's click, so it asks what the button asks
# ================================================================================================

def test_the_run_detection_menu_item_asks_before_it_replaces_hand_curation(
        detected, qtbot, blocked_modals):
    """The menu item and the Detect button are one gesture wearing two hats.

    ``blocked_modals`` answers as a user who cancelled, so the assertion is both halves of
    the promise at once: the question was asked, naming what is at stake, and the table
    that was at stake is still there.
    """
    window, _doc_id, _sig, panel = detected
    df = panel.current_detection().payload.df
    for row in (1, 3):
        panel._table.set_accepted_by_id(int(df["event_id"].iloc[row]), False)
    before_ids = list(panel.current_detection().payload.df["event_id"])
    assert _accepted(panel) == len(df) - 2

    assert window._act_run_detection.isEnabled()
    window._act_run_detection.trigger()
    qtbot.wait(50)

    assert len(blocked_modals) == 1, (
        f"Analysis ▸ Run detection replaced a curated table without asking: {blocked_modals}")
    assert "2 events you rejected by hand" in blocked_modals[0].text, blocked_modals[0].text
    after = panel.current_detection().payload.df
    assert _accepted(panel) == len(df) - 2, "the run went ahead over a dialog nobody answered"
    assert list(after["event_id"]) == before_ids


def test_the_menu_item_does_not_ask_when_there_is_no_hand_curation_to_lose(
        detected, qtbot, blocked_modals):
    """A gate that stops an ordinary re-run is a gate the user learns to click through.

    Nothing has been curated here, so the same action must run straight away — the prompt
    is about hand work at stake, not about detection.
    """
    window, _doc_id, _sig, panel = detected
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._act_run_detection.trigger()
    qtbot.wait(50)
    assert blocked_modals == [], f"an uncurated re-run asked a question: {blocked_modals}"


def test_the_programmatic_run_still_goes_ahead_unprompted(detected, qtbot, blocked_modals):
    """THE NO-REGRESSION GUARD, and it passes on both sides of the change.

    ``trigger_detection`` is the door the stale banner's Re-run, the scripts and the suite
    use, and a modal nobody clicked for is not a gate — it is a hang. Wiring the menu to
    the prompting door must not have moved this one.
    """
    window, _doc_id, _sig, panel = detected
    df = panel.current_detection().payload.df
    panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(50)
    assert blocked_modals == [], f"the programmatic door raised a modal: {blocked_modals}"


# ================================================================================================
# W1 — a run commits the table it is about to overwrite, announced or not
# ================================================================================================

def test_a_run_commits_an_unannounced_curation_edit_before_it_replaces_the_table(
        detected, qtbot):
    """The reason the seam fires BEFORE the run rather than after it.

    ``checkpoint`` pushes the state it last COMMITTED, so an in-place write that announced
    nothing — the whole class ``checkpoint_all`` exists to catch — is jumped past by the
    checkpoint the arriving result triggers, and Ctrl+Z then restores the table as it stood
    BEFORE that write while the menu says it is undoing the run. Measured on the real
    recording: 20 unannounced rejections, re-run, Undo → 110 accepted, the 20 gone.
    """
    window, _doc_id, _sig, panel = detected
    window._checkpoint_everything()
    df = panel.current_detection().payload.df
    col = df.columns.get_loc("accepted")
    for row in (0, 1, 2):                     # THE UNANNOUNCED DOOR: write, tell nobody
        df.iat[row, col] = False
    hand = _accepted(panel)
    assert hand == len(df) - 3

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(100)
    assert _accepted(panel) == len(panel.current_detection().payload.df), (
        "the re-run did not actually replace the table, so this proves nothing")

    assert window._act_undo.isEnabled()
    window._act_undo.trigger()
    qtbot.wait(50)
    assert _accepted(panel) == hand, (
        f"Edit ▸ Undo said {window._act_undo.text()!r} but gave back "
        f"{_accepted(panel)} accepted, not the {hand} that were on the table when the "
        f"run started")


# ================================================================================================
# W2 — ⌘Z in the Event Gallery reaches the one undo stack
# ================================================================================================

@pytest.fixture
def gallery(detected, qtbot):
    window, doc_id, sig, panel = detected
    window.show()
    window._open_event_gallery()
    gal = window._event_gallery
    gal.resize(1200, 860)
    qtbot.addWidget(gal)
    qtbot.wait(50)
    return window, panel, gal


def test_ctrl_z_inside_the_event_gallery_undoes_the_batch_it_was_pressed_over(
        gallery, qtbot):
    """The gesture, in the window the mistake was made in.

    The gallery is a top-level window with no menu bar, so Edit ▸ Undo's WindowShortcut
    belongs to the window BEHIND it and the key arrived nowhere. Driven here as a key on
    the tile view, which is where a curator's hands are.
    """
    window, panel, gal = gallery
    before = _accepted(panel)
    QTest.keyClick(gal.view, Qt.Key_A, Qt.ControlModifier)
    qtbot.wait(30)
    n_sel = len(gal.selected_event_ids())
    assert n_sel > 1, "nothing was selected, so the R below curates nothing"
    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(50)
    curated = _accepted(panel)
    assert curated < before, "R over the selection wrote nothing"
    assert window._act_undo.isEnabled(), window._act_undo.text()

    QTest.keyClick(gal.view, Qt.Key_Z, Qt.ControlModifier)
    qtbot.wait(50)
    assert _accepted(panel) == before, (
        f"⌘Z in the gallery left {_accepted(panel)} accepted; the main window's Edit menu "
        f"says {window._act_undo.text()!r}, so the step exists and the key did not reach it")


def test_the_gallerys_own_undo_button_presses_the_same_step(gallery, qtbot):
    """The button is the visible half of the same seam, for the hand that is on the mouse."""
    window, panel, gal = gallery
    before = _accepted(panel)
    gal.select_all_shown()
    gal.set_accepted_selected(False)
    qtbot.wait(50)
    assert _accepted(panel) < before
    assert gal.undo_btn.isEnabled()
    QTest.mouseClick(gal.undo_btn, Qt.LeftButton)
    qtbot.wait(50)
    assert _accepted(panel) == before, "the gallery's Undo button is not connected to anything"


def test_the_gallerys_undo_button_says_what_the_edit_menu_says(gallery, qtbot):
    """One stack, so one label: the button's tooltip carries the ACTION's own text.

    The return leg hangs off ``QAction.changed`` rather than a call beside each of the ten
    places that re-sync the menu, so it cannot fall behind. Without it the window is left
    with its own lower bound — enabled, captioned a bare "Undo" — while the menu names the
    step.
    """
    window, panel, gal = gallery
    gal.select_all_shown()
    gal.set_accepted_selected(False)
    qtbot.wait(50)
    label = window._act_undo.text()
    assert label.startswith("Undo ") and label != "Undo", (
        f"the menu itself does not name the step ({label!r}); nothing to mirror")
    assert label in gal.undo_btn.toolTip(), (
        f"button tooltip {gal.undo_btn.toolTip()!r} does not carry the menu's {label!r}")


def test_a_gallery_opened_over_an_existing_step_already_knows_about_it(detected, qtbot):
    """The seed call. ``changed`` has not fired yet for a window built after the fact.

    A gallery opened onto a stack that already has steps must not have to wait for the next
    curation before its Undo can say what it would walk back.
    """
    window, _doc_id, _sig, panel = detected
    df = panel.current_detection().payload.df
    panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    qtbot.wait(30)
    assert window._act_undo.isEnabled()
    label = window._act_undo.text()

    window.show()
    window._open_event_gallery()
    gal = window._event_gallery
    qtbot.addWidget(gal)
    qtbot.wait(30)
    assert label in gal.undo_btn.toolTip(), (
        f"a gallery opened over an existing step says {gal.undo_btn.toolTip()!r}, "
        f"while the menu says {label!r}")


def test_nothing_to_undo_reaches_the_gallery_as_a_disabled_button(window, tmp_path, qtbot):
    """The other direction, which is the half that keeps the button honest.

    A control that claims a step that does not exist is the same lie as a shortcut that
    does nothing, so the disabled state has to travel too.
    """
    doc_id, sig = _install(window, tmp_path)
    _detect_into(window, doc_id, sig)
    window.show()
    window._open_event_gallery()
    gal = window._event_gallery
    qtbot.addWidget(gal)
    qtbot.wait(30)
    assert not window._act_undo.isEnabled(), window._act_undo.text()
    assert not gal.undo_btn.isEnabled(), (
        "the gallery offers an Undo the application does not have")


# ================================================================================================
# W4 — "the table on screen" has ONE answer in this window
# ================================================================================================

@pytest.fixture
def two_detectors(detected, qtbot):
    """Schmitt and CUSUM stored on ONE channel, the panel showing the CUSUM table.

    ``ResultStore`` has always been keyed (doc_id, channel, module_id); since discovery
    began offering one source per detector, ``(doc_id, channel)`` no longer names a table.
    """
    window, doc_id, sig, panel = detected
    _detect_into(window, doc_id, sig, module_id=CUSUM, min_dwell_s=2e-4)
    qtbot.wait(30)
    res = panel.current_detection()
    assert res is not None and res.module_id == CUSUM, (
        f"the panel is showing {res and res.module_id!r}, so there is nothing to disagree with")
    return window, doc_id, panel


def test_the_plots_source_flagged_current_is_the_detection_the_panel_is_showing(
        two_detectors):
    """The defect this closes drew one detector's rows and captioned them as the other's.

    Discovery is told the (doc, channel, module_id) TRIPLE. Handed the two-element
    selection it has to guess, and it guesses the channel's first detection — Schmitt,
    while the panel is on CUSUM.
    """
    window, _doc_id, panel = two_detectors
    sources = window._plots_sources()
    by_module = {s.module_id: s for s in sources if getattr(s, "module_id", None)}
    assert {SCHMITT, CUSUM} <= set(by_module), sorted(by_module)

    flagged = [s for s in sources if s.current]
    assert len(flagged) == 1, [(s.key, s.current) for s in sources]
    assert flagged[0].module_id == CUSUM, (
        f"Plots flags {flagged[0].module_id!r} as the current source while the panel is "
        f"showing {panel.current_detection().module_id!r}")


def test_the_plots_window_is_told_the_active_detection_when_it_is_given_the_sources(
        two_detectors, qtbot):
    """The push, and the ORDER of it: the window decides which row to tick from this."""
    window, doc_id, panel = two_detectors
    window._open_plots()
    qtbot.addWidget(window._plots_window)
    qtbot.wait(30)
    assert window._plots_window._active_detection == (doc_id, 0, CUSUM), (
        f"the Plots window was handed {window._plots_window._active_detection!r}")


def test_undo_and_the_plots_window_ask_one_resolver_which_table_is_on_screen(two_detectors):
    """A second spelling of "the table on screen" is how the two came apart in the first
    place, so ``_undo_key`` is now ``active_detection_key`` under the name undo reads it by.
    """
    window, doc_id, _panel = two_detectors
    assert window.active_detection_key() == (doc_id, 0, CUSUM)
    assert window._undo_key() == window.active_detection_key()


def test_with_no_detection_at_all_the_sources_still_come_back(window, tmp_path):
    """The fallback: a window with no detection at all must still resolve.

    ``active_detection_key()`` is None before anything is detected, which is why
    ``_plots_sources`` passes ``key or self._current_selection`` — a bare None means
    "flag nothing", and discovery would then have no idea what the user is looking at the
    moment the first run lands. Against the pre-fix file this fails on the missing
    resolver rather than on a wrong flag, and that is what it is: the seam did not exist.
    """
    doc_id, _sig = _install(window, tmp_path)
    assert window.active_detection_key() is None
    assert window._plots_sources() == [] or all(
        not s.current for s in window._plots_sources())
