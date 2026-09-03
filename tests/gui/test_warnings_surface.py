"""FIX 1 — THE UNPROBED-DEFAULT ADVISORY REACHED NO WIDGET.

``nanoscope.analysis.baseline`` builds ``UNPROBED_DEFAULT_WARNING`` (``:706``), appends it
to ``BaselineResult.warnings`` whenever a run used the app's untouched default window
(``:1922``), ``modules._build_warnings`` carries it onto ``AnalysisResult.warnings``, and
``main_window`` persists and restores it at four call sites. Then it stopped.

    $ grep -rl UNPROBED_DEFAULT_WARNING nanoscope/gui/
    $                       # nothing

A sweep of every text-bearing widget in a live ``MainWindow`` found zero occurrences of
"DEFAULT open-pore", "never measured" or "before trusting a depth". What that silence is
worth, measured on the owner's own ``…_0006_Process_-600 mV.abf`` (ch 2, 50 kHz):

    =================================  ========  =================  ==========
    open-pore window                   n events  median depth (pA)  median dI/I0
    =================================  ========  =================  ==========
    app DEFAULT, 0.5 / 0.1 s (unprobed)     317              85.68       2.423
    measured for this recording             276              87.59       2.523
    =================================  ========  =================  ==========

The Summary showed **85.68 pA** with no caveat, no row tooltip and nothing anywhere else,
and the owner had been told the app warns here.

THE FIX IS AT THE INVARIANT, NOT AT THE STRING. ``AnalysisResult.warnings`` is a general
channel — an undetermined polarity, a filter-limited fraction, a boundary-clipped row, a
baseline that fell back, a health check that fired, all travel on it — with four writers
and no reader. ``StatsPanel.set_warnings`` renders the tuple WHOLE, directly under the
Summary table, and ``AnalysisPanel._show_result`` pushes it on every result shown. So the
tests below are in two layers:

* the GUARD (``test_any_warning_on_a_result_reaches_a_widget_a_user_reads``) — a result
  carrying an *arbitrary* sentinel must produce that sentinel in a visible, text-bearing
  widget. It knows nothing about baselines, and it fails the day someone adds a warning
  the panel drops on the floor;
* the ORIGINAL FAILURE — the real advisory, produced by a real unprobed detection,
  readable on screen; and the same thing end to end through a real ``MainWindow`` on the
  real −600 mV recording, where probing → confirming → re-detecting makes it go away.

Dismissible by FIXING, never by clicking: there is no "don't show again" anywhere in the
block, and ``test_the_block_offers_no_way_to_silence_it`` pins that.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.baseline import DEFAULT_BASELINE, UNPROBED_DEFAULT_WARNING
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 400_000                      # 8 s @ 50 kHz — long enough for the 0.5 s rough pass
I0 = -600.0
DEPTH = 180.0

REAL_ABF = Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_needs_abf = pytest.mark.skipif(
    not REAL_ABF.exists(), reason="owner's -600 mV recording not present")

#: Phrases from the real advisory that a user would actually recognise on screen. Asserted
#: individually rather than as one blob so a reflow of the sentence does not silently turn
#: this test green-by-accident on a panel that renders half of it.
_ADVISORY_PHRASES = ("DEFAULT open-pore", "never measured", "before trusting a depth")


# ================================================================================================
# THE SWEEP — what a user can actually read, with no knowledge of which widget holds it
# ================================================================================================
def _visible_text(root: QtWidgets.QWidget) -> str:
    """Every string a user could read off ``root``, as one blob.

    Deliberately structure-blind: it walks the real widget tree and takes whatever each
    widget exposes as text, so it keeps working if the advisory is later moved to another
    widget or another panel. ``isVisibleTo(root)`` rather than ``isVisible()`` — an
    offscreen test never realises the window, and ``isVisible()`` is False for every child
    of an unshown top level, which would make this sweep vacuously empty and every
    assertion below meaningless in BOTH directions.
    """
    out: list[str] = []
    for w in root.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(root):
            continue
        for attr in ("text", "toPlainText", "toolTip"):
            fn = getattr(w, attr, None)
            if callable(fn):
                try:
                    val = fn()
                except (TypeError, RuntimeError):
                    continue
                if isinstance(val, str) and val:
                    out.append(val)
        if isinstance(w, QtWidgets.QTableWidget):
            for r in range(w.rowCount()):
                for c in range(w.columnCount()):
                    item = w.item(r, c)
                    if item is not None:
                        out.append(item.text())
    return "\n".join(out)


# ================================================================================================
# builders
# ================================================================================================
def _signal(seed: int = 0) -> np.ndarray:
    """A −600 mV-ish open pore with 40 clean translocations."""
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 6.0, N)
    for i in range(40):
        s = 5_000 + i * 9_500
        x[s:s + 400] += DEPTH          # resistive at negative I0 -> toward zero
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str = "unprobed.dat") -> TraceDocument:
    sig = _signal()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(sig.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture(autouse=True)
def _private_settings(qapp):
    """A namespace of this module's own, restored afterwards.

    ``MainWindow`` writes ``geometry``/``windowState`` on close, so a test that shows a
    real window can otherwise leave a layout behind for the next one to restore — and,
    worse, restore the developer's own. "First launch" here means first launch.
    """
    app = QtWidgets.QApplication.instance()
    org, name = app.organizationName(), app.applicationName()
    app.setOrganizationName("NanoScopeTest")
    app.setApplicationName("WarningsSurface")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        app.setOrganizationName(org)
        app.setApplicationName(name)


@pytest.fixture
def panel(qapp, qtbot, tmp_path):
    """A real ``AnalysisPanel`` on a real session, with a document selected."""
    session = Session()
    tv = TraceView(session)
    qtbot.addWidget(tv)
    controller = AnalysisController(session)
    p = AnalysisPanel(session, controller, tv)
    qtbot.addWidget(p)
    doc_id = session.add_document(_doc(tmp_path))
    session.add_layer(doc_id, 0)
    p.set_selection(doc_id, 0)
    yield p, session, doc_id, controller
    controller.abort()


def _detect(p: AnalysisPanel, controller, qtbot) -> AnalysisResult:
    """Run the panel's own detection and return the installed result."""
    with qtbot.waitSignal(controller.finished, timeout=120_000):
        p.trigger_detection()
    qtbot.wait(30)
    res = p._stored_detection()
    assert res is not None, "the fixture must produce a detection to caveat"
    return res


def _install(session, res: AnalysisResult) -> None:
    """Install a result through the STORE, so the panel repaints by its real path."""
    session.add_result(res)


# ================================================================================================
# THE GUARD — the invariant, not the string
# ================================================================================================
def test_any_warning_on_a_result_reaches_a_widget_a_user_reads(panel, qtbot):
    """A result carrying an arbitrary warning must render it. The point of the fix.

    The sentinel is nonsense on purpose: the panel is not allowed to know which warnings
    it is willing to show. Before the fix this failed for EVERY string, which is exactly
    why the real advisory could be built, appended, carried, saved, restored — and read
    by nobody.
    """
    p, session, doc_id, controller = panel
    res = _detect(p, controller, qtbot)
    sentinel = "ZZQ-SENTINEL-4417: this number is not to be trusted"

    from dataclasses import replace
    _install(session, replace(res, warnings=(sentinel,)))
    qtbot.wait(20)

    assert sentinel in _visible_text(p), (
        "AnalysisResult.warnings is a general channel with four writers; a panel that "
        "renders none of it makes every future warning invisible by construction"
    )


def test_every_warning_is_rendered_not_only_the_first(panel, qtbot):
    """The tuple is a LIST of caveats, and a table can be wrong in several ways at once —
    an unprobed window AND a filter-limited fraction AND a clipped row all travel here."""
    p, session, doc_id, controller = panel
    res = _detect(p, controller, qtbot)
    warns = ("ZZQ-ONE first caveat", "ZZQ-TWO second caveat", "ZZQ-THREE third caveat")

    from dataclasses import replace
    _install(session, replace(res, warnings=warns))
    qtbot.wait(20)

    text = _visible_text(p)
    for w in warns:
        assert w in text, f"{w!r} was dropped; the panel renders the tuple, not its head"


# ================================================================================================
# THE ORIGINAL FAILURE — the real advisory, from a real unprobed detection
# ================================================================================================
def test_an_unprobed_detection_says_so_beside_the_depth_it_qualifies(panel, qtbot):
    """The reported defect, reproduced. No hand-built warnings tuple anywhere: the panel
    runs its own detection at its own untouched defaults, exactly as a user who never
    opened the Preprocess tab would, and the advisory the estimator produces must be
    readable on screen."""
    p, session, doc_id, controller = panel

    # The precondition the advisory is ABOUT: nobody has measured this recording.
    assert p._baseline.param_source() == "default", (
        "this test is only meaningful on an UNTOUCHED default window")
    assert p._baseline.config().rough_s == DEFAULT_BASELINE.rough_s

    res = _detect(p, controller, qtbot)
    assert UNPROBED_DEFAULT_WARNING in res.warnings, (
        "upstream must still put the advisory on the result — if this fails the defect "
        "moved, and the GUI half of this file is testing the wrong thing")

    text = _visible_text(p)
    for phrase in _ADVISORY_PHRASES:
        assert phrase in text, (
            f"{phrase!r} is nowhere a user can read it. On the owner's -600 mV file the "
            f"un-caveated Summary reads 85.68 pA where the measured window gives 87.59."
        )


def test_the_advisory_is_beside_the_summary_not_buried_in_a_tooltip(panel, qtbot):
    """WHERE it appears is half the fix. A caveat reachable only by hovering an unmarked
    row is the state this project already ruled against, so it must be in the Summary
    panel's own visible text — not merely somewhere in the widget tree."""
    p, session, doc_id, controller = panel
    _detect(p, controller, qtbot)

    shown = p._stats.warnings_text()
    assert shown, "the Summary's advisory block is empty on an unprobed detection"
    for phrase in _ADVISORY_PHRASES:
        assert phrase in shown
    assert "⚠" in shown, (
        "a glyph as well as a colour — a screenshot of an amber label read by someone "
        "else is a screenshot of a plain sentence")


# ================================================================================================
# DISMISSIBLE BY FIXING, NEVER BY CLICKING
# ================================================================================================
def test_the_block_offers_no_way_to_silence_it(panel, qtbot):
    """No checkbox, no 'don't show again', no close button anywhere in the advisory.

    A dismiss control would let a user silence the caveat while the wrong depth stayed on
    screen — which is precisely the state the caveat exists to prevent.
    """
    p, session, doc_id, controller = panel
    _detect(p, controller, qtbot)
    assert p._stats.warnings_text()

    stats = p._stats
    forbidden = ("dismiss", "don't show", "do not show", "hide this", "ignore", "×")
    for w in stats.findChildren(QtWidgets.QAbstractButton):
        label = (w.text() or "").strip().lower()
        assert not any(f in label for f in forbidden), (
            f"the advisory block grew a dismiss control ({label!r})")
    # And the label itself is not a clickable link that could hide it.
    assert not stats.warnings_label.openExternalLinks()


def test_a_result_without_warnings_clears_the_block(panel, qtbot):
    """The condition going away is what takes the sentence away. A cleared block must not
    be a stale one either: ``warnings_text`` reads "" only because the widget is hidden."""
    p, session, doc_id, controller = panel
    res = _detect(p, controller, qtbot)
    assert p._stats.warnings_text()

    from dataclasses import replace
    _install(session, replace(res, warnings=()))
    qtbot.wait(20)

    assert p._stats.warnings_text() == ""
    assert not p._stats.warnings_label.isVisibleTo(p._stats)
    text = _visible_text(p)
    for phrase in _ADVISORY_PHRASES:
        assert phrase not in text


def test_a_curation_edit_leaves_the_caveats_exactly_where_they_are(panel, qtbot):
    """Rejecting a row re-derives the Summary counts through ``_refresh_summary_from_table``,
    which has no result in hand. It must not blank the advisory: which open-pore window the
    depths were measured under is not something curation changes."""
    p, session, doc_id, controller = panel
    _detect(p, controller, qtbot)
    before = p._stats.warnings_text()
    assert before

    p._refresh_summary_from_table()
    qtbot.wait(10)
    assert p._stats.warnings_text() == before


def test_clearing_the_panel_clears_the_caveats(panel, qtbot):
    """A caveat about a table that is no longer on screen is a statement about nothing."""
    p, session, doc_id, controller = panel
    _detect(p, controller, qtbot)
    assert p._stats.warnings_text()

    p._stats.clear_summary()
    assert p._stats.warnings_text() == ""


# ================================================================================================
# END TO END — a real MainWindow, the real −600 mV recording, probe → confirm → re-detect
# ================================================================================================
@_needs_abf
def test_probing_and_redetecting_the_real_recording_removes_the_advisory(qapp, qtbot):
    """THE WHOLE ARC, on the file the numbers in this module's docstring were measured on.

    1. open the recording, detect WITHOUT touching the Preprocess tab -> the advisory is
       on screen, beside the Summary's median depth;
    2. press "Measure this recording", confirm the proposal it comes back with;
    3. detect again -> the advisory is gone, because the condition is gone.

    Driven through the panel's real handlers (``_on_probe_requested`` via the button's own
    ``probeRequested``, then ``apply_proposal`` — the confirmation click), not by writing
    the parameters in by hand: what is being tested is that the app's own repair path
    clears its own advisory.
    """
    from nanoscope.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    # Genuinely shown, with the Analysis tab genuinely raised — the state a user is in
    # when they read a depth. On an unshown window every child is invisible and the
    # sweep below would be vacuous in both directions.
    w.resize(1600, 900)
    w.show()
    w._inspector.show()
    w._inspector.raise_()
    w._inspector_tabs.setCurrentWidget(w._analysis_panel)
    QtWidgets.QApplication.processEvents()
    assert w._analysis_panel.isVisible(), "the Analysis tab was never actually raised"

    with qtbot.waitSignal(w._session.documentOpened, timeout=120_000):
        w.open_paths([str(REAL_ABF)])
    qtbot.wait(50)
    doc_id = w._session.doc_id_for_source(str(REAL_ABF.resolve()))
    assert doc_id is not None
    channel = next(c.index for c in w._session.document(doc_id).meta.channels
                   if c.kind is ChannelKind.CURRENT)
    w._on_selection(doc_id, channel)
    qtbot.wait(30)

    p = w._analysis_panel
    ctrl = w._analysis_controller
    try:
        # -- 1. unprobed detection -------------------------------------------------
        assert p._baseline.param_source() == "default"
        with qtbot.waitSignal(ctrl.finished, timeout=600_000):
            p.trigger_detection()
        qtbot.wait(50)
        res = p._stored_detection()
        assert res is not None and len(res.payload.df), "no detection to caveat"
        assert UNPROBED_DEFAULT_WARNING in res.warnings

        text = _visible_text(w)
        for phrase in _ADVISORY_PHRASES:
            assert phrase in text, (
                f"{phrase!r} is not on screen after an unprobed detection of the "
                f"owner's own -600 mV recording — the reported defect, verbatim")

        # -- 2. measure this recording, then confirm -------------------------------
        with qtbot.waitSignal(ctrl.finished, timeout=600_000):
            p._baseline.probeRequested.emit()
        qtbot.wait(50)
        proposal = p._baseline.proposal()
        assert proposal is not None, "the probe returned no proposal to confirm"
        assert p._baseline.apply_proposal() is True
        assert p._baseline.param_source() != "default", (
            "the confirmation click must move the parameters off the default")

        # -- 3. re-detect: the condition is gone, so the sentence is gone ----------
        with qtbot.waitSignal(ctrl.finished, timeout=600_000):
            p.trigger_detection()
        qtbot.wait(50)
        res2 = p._stored_detection()
        assert res2 is not None
        assert UNPROBED_DEFAULT_WARNING not in res2.warnings
        # The repair is not cosmetic: measured through this very path, the probe offers
        # 2.0 / 0.5 s and the table goes 317 -> 276 events, 85.68 -> 87.59 pA median
        # depth, dI/I0 2.423 -> 2.523. Asserted as "it moved" rather than as those four
        # literals, because pinning detection numbers is another brief's contract and a
        # failure here should mean the ADVISORY broke, not that a detector improved.
        assert len(res2.payload.df) != len(res.payload.df), (
            "the confirmed window produced an identical table, so this run cannot show "
            "that the advisory was worth anything")

        text2 = _visible_text(w)
        for phrase in _ADVISORY_PHRASES:
            assert phrase not in text2, (
                f"{phrase!r} survived a measured, confirmed, re-detected run — the "
                f"advisory must be dismissible by FIXING the condition")
    finally:
        ctrl.abort()
        w._controller.abort()
