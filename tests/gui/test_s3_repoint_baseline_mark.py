"""Sweep-3 regression DET3-02 — the baseline panel's readouts across a REPOINT.

THE RULE (the same one JM-01 and JM-04 broke, on the panel next door)
====================================================================
No readout may show one recording's answer while the panel is pointed at another, and
no control may claim a state it no longer has. ``BaselinePanel`` states its own half of
it in the code that raises the standing mark:

    THE STANDING MARK. The refusal above is a moment; this is for as long as the form
    holds these numbers, because the screen that has to say "these windows were
    condemned" is the screen the user reads the resulting depth on, not the one they
    clicked through ten minutes earlier.

Three of its readouts failed that rule at the repoint, in BOTH directions:

* **DET3-02 — the mark did not come back.** On a gating recording the probe proposes
  windows the estimator's own checks CONDEMN; the first Apply is refused and the second
  accepts, and the source line then reads "… — ACCEPTED OVER 2 FAILED HEALTH CHECKS
  (…). Depths and dI/I0 measured with these windows are not certifiable". Selecting
  another recording and coming back re-seeds *exactly those windows* out of the stored
  detection's own params (``AnalysisPanel._seed_form_from``) and re-stamps the source as
  ``'proposed'`` — but ``accepted_condemned()`` came back ``()`` and the line reverted
  to "These values: measured on this recording, and confirmed", the panel's most
  reassuring sentence, over windows the app itself refused. The health block below still
  listed the failing check as WARNING, so two surfaces disagreed about one fact, and
  clicking between recordings is the ordinary gesture of a multi-file session.

* **The mark stood over an EMPTY panel.** ``BaselinePanel.set_recording(fs, target)``
  documents ``target=None`` as "do not judge", and ``AnalysisPanel._show_empty`` passed
  ``None`` — so after the last document was CLOSED the form still held its windows and
  still said "measured on this recording, and confirmed — ACCEPTED OVER …", pointing at
  a "Baseline health below" that the same method had just emptied.

* **The REFUSED notice followed the panel around.** ``_post`` writes the refusal;
  nothing writes its absence (``set_gate(True)`` hides the label without emptying it,
  ``set_gate(False)`` touches neither). Measured: a refused first click left that banner
  VISIBLE, verbatim, after selecting a voltage-only recording where no baseline can be
  measured at all, and after every document was closed.

WHAT THESE TESTS PIN
====================
Both directions of the rule, because a fix that only CLEARS is as wrong as one that
only writes: the mark must come back with the windows it was about (and with THIS
recording's checks in it, never the other recording's), and it must not appear over any
other windows, over another recording, or over nothing at all.

Every assertion is on state a user can see — the two spins, the source line as
RENDERED, the gate banner's visibility, ``accepted_condemned()`` (the panel's own public
answer to the same question) — driven through a real :class:`MainWindow` with the same
gestures a user makes: the selection path the file tree uses, the panel's own Apply
button clicked twice, Detect, and closing a document.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtGui

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow


# ================================================================================================
# builders
# ================================================================================================

def _gating(n: int, fs: float, *, open_ms: float, blocked_ms: float,
            level: float, depth: float, noise: float, seed: int) -> np.ndarray:
    """A stepping/gating pore — the regime whose proposed windows the health checks
    condemn. Which check condemns depends on the duty cycle and the rate, which is how
    the two recordings in these tests come to carry DIFFERENT marks."""
    rng = np.random.default_rng(seed)
    x = np.full(n, level)
    o, b = int(open_ms * 1e-3 * fs), int(blocked_ms * 1e-3 * fs)
    i = 0
    while i + o + b <= n:
        x[i + o:i + o + b] = level - depth
        i += o + b
    return np.ascontiguousarray(x + rng.normal(0.0, noise, n), dtype=np.float32)


def _plain(n: int, fs: float, seed: int = 3) -> np.ndarray:
    """An ordinary recording: sparse short events, a proposal nothing condemns."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 95.0) + rng.normal(0.0, 1.5, n)
    for i in range(30):
        s = 2_000 + i * (n // 32)
        x[s:s + int(0.0008 * fs)] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str, arr: np.ndarray, fs: float,
         kind: ChannelKind = ChannelKind.CURRENT) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=kind)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(arr.size),
        sample_rate_hz=float(fs), rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Add + layer + select, the way MainWindow's own loader does."""
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _probe(window: MainWindow, qtbot) -> None:
    """Press "Measure this recording" and wait for the proposal to land."""
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        window._analysis_panel.baseline_panel.probe_button.click()
    qtbot.wait(50)


def _detect(window: MainWindow, qtbot) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        panel.trigger_detection()
    qtbot.wait(30)


def _source_line(window: MainWindow) -> str:
    """The source line AS RENDERED — the characters that reach the screen."""
    doc = QtGui.QTextDocument()
    doc.setHtml(window._analysis_panel.baseline_panel.source_label.text())
    return doc.toPlainText()


def _windows(window: MainWindow) -> tuple[float, float]:
    p = window._analysis_panel.baseline_panel.params()
    return float(p["baseline_rough_s"]), float(p["baseline_base_s"])


def _accept_condemned(window: MainWindow, qtbot) -> tuple[str, ...]:
    """Probe, then accept the condemned proposal THE WAY A USER DOES: two presses of
    the panel's own button — the first refused with the reason, the second accepting.
    Returns the mark that is now standing."""
    bp = window._analysis_panel.baseline_panel
    _probe(window, qtbot)
    assert bp.condemned_checks(), (
        "fixture drifted: this proposal is not condemned, so there is no mark for "
        "these tests to be about")
    bp.apply_button.click()
    qtbot.wait(20)
    assert not bp.accepted_condemned(), "the FIRST click must refuse, not accept"
    bp.apply_button.click()
    qtbot.wait(20)
    mark = bp.accepted_condemned()
    assert mark, "the second click did not accept the condemned windows"
    return mark


# ================================================================================================
# fixtures
# ================================================================================================

@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def gate_a(tmp_path):
    """60 ms open / 40 ms blocked at 50 kHz — condemned, and detected afterwards."""
    return _doc(tmp_path, "gate_a.dat",
                _gating(500_000, 50_000.0, open_ms=60, blocked_ms=40,
                        level=100.0, depth=40.0, noise=1.5, seed=0), 50_000.0)


@pytest.fixture
def gate_b(tmp_path):
    """20 ms open / 60 ms blocked — a pore blocked most of the time, so the proposal is
    condemned by ``mask_fraction_ceiling`` alone, a check that says nothing about
    `gate_a` (whose own three are base_vs_state, baseline_intrusion and overshoot).
    That difference is what lets a test tell one recording's mark from the other's."""
    return _doc(tmp_path, "gate_b.dat",
                _gating(500_000, 50_000.0, open_ms=20, blocked_ms=60,
                        level=120.0, depth=50.0, noise=2.0, seed=4), 50_000.0)


@pytest.fixture
def plain(tmp_path):
    return _doc(tmp_path, "plain.dat", _plain(200_000, 50_000.0), 50_000.0)


# ================================================================================================
# DET3-02 — the mark comes back with the windows it was about
# ================================================================================================

def test_the_mark_comes_back_when_the_condemned_windows_do(
        window, qtbot, gate_a, plain):
    """The defect itself: accept over the refusal, detect, click away, click back."""
    doc_a, ch_a = _install(window, gate_a)
    mark = _accept_condemned(window, qtbot)
    _detect(window, qtbot)
    accepted_windows = _windows(window)
    assert "ACCEPTED OVER" in _source_line(window), "setup: nothing was marked"

    doc_p, ch_p = _install(window, plain)
    qtbot.wait(20)
    window._on_selection(doc_a, ch_a)
    qtbot.wait(50)

    bp = window._analysis_panel.baseline_panel
    assert _windows(window) == accepted_windows, (
        "setup: the round trip did not put the condemned windows back, so this test "
        "is not about the mark")
    assert bp.accepted_condemned() == mark, (
        "the standing mark did not come back with the windows it describes: "
        f"{bp.accepted_condemned()!r}")
    assert "ACCEPTED OVER" in _source_line(window), _source_line(window)
    assert "not certifiable" in _source_line(window)
    assert "ffb454" in bp.source_label.styleSheet(), (
        "the mark came back without the warning colour it is written in")


def test_the_restored_mark_names_this_recordings_own_checks(
        window, qtbot, gate_a, gate_b):
    """Two recordings, both accepted over their OWN failing checks. The mark that comes
    back on A must be A's — a mark naming another file's checks is the
    surrogate-answer-on-the-wrong-recording defect, one field along."""
    doc_a, ch_a = _install(window, gate_a)
    mark_a = _accept_condemned(window, qtbot)
    _detect(window, qtbot)

    doc_b, ch_b = _install(window, gate_b)
    mark_b = _accept_condemned(window, qtbot)
    _detect(window, qtbot)
    assert set(mark_a) != set(mark_b), (
        "fixture drifted: the two recordings are condemned by the same checks, so "
        f"this test cannot tell them apart ({mark_a!r})")

    window._on_selection(doc_a, ch_a)
    qtbot.wait(50)
    bp = window._analysis_panel.baseline_panel
    assert bp.accepted_condemned() == mark_a, (
        f"recording A came back marked {bp.accepted_condemned()!r}; its own checks "
        f"are {mark_a!r} and recording B's are {mark_b!r}")
    for label in mark_b:
        if label not in mark_a:
            assert label not in _source_line(window), (
                "the source line on recording A names a check that condemned "
                f"recording B: {label!r}")


def test_the_replay_leaves_the_detections_own_health_block_on_screen(
        window, qtbot, gate_a, plain):
    """Restoring the mark re-offers the accepted proposal for an instant. What must be
    on screen when the dust settles is the DETECTION-scope block — the verdict on the
    table the user is reading depths off — never the proposal-scope one, and no
    proposal may be left on offer or the Apply button left armed."""
    doc_a, ch_a = _install(window, gate_a)
    _accept_condemned(window, qtbot)
    _detect(window, qtbot)

    _install(window, plain)
    qtbot.wait(20)
    window._on_selection(doc_a, ch_a)
    qtbot.wait(50)

    bp = window._analysis_panel.baseline_panel
    assert bp.accepted_condemned(), "setup: the mark was not restored"
    assert bp.health_scope() == "detection", (
        "the restored mark left a PROPOSAL-scope health block on screen; it describes "
        "windows on offer, not the table the user is reading")
    assert bp.proposal() is None, (
        "the replay left a proposal on offer that the user never asked for")
    assert bp.condemned_checks() == (), (
        "a proposal that is not on offer cannot be condemned")
    assert "anyway" not in bp.apply_button.text().lower(), (
        "the Apply button is left ARMED, so the next condemned proposal would be "
        f"accepted in one click: {bp.apply_button.text()!r}")


def test_no_mark_appears_over_windows_that_were_never_refused(
        window, qtbot, plain, gate_a):
    """The other direction. A recording whose proposal nothing condemned must never
    acquire the sentence, however many repoints it survives."""
    doc_p, ch_p = _install(window, plain)
    _probe(window, qtbot)
    bp = window._analysis_panel.baseline_panel
    assert bp.condemned_checks() == (), (
        "fixture drifted: this proposal IS condemned, so it cannot show that a clean "
        "one stays unmarked")
    bp.apply_button.click()          # one click, because nothing refuses it
    qtbot.wait(20)
    _detect(window, qtbot)
    assert "ACCEPTED OVER" not in _source_line(window)

    doc_a, ch_a = _install(window, gate_a)
    _accept_condemned(window, qtbot)
    window._on_selection(doc_p, ch_p)
    qtbot.wait(50)

    assert bp.accepted_condemned() == (), bp.accepted_condemned()
    assert "ACCEPTED OVER" not in _source_line(window), _source_line(window)
    assert "8a939c" in bp.source_label.styleSheet()


def test_the_mark_does_not_come_back_over_different_windows(
        window, qtbot, gate_a, plain):
    """The mark describes NUMBERS. Detect under hand-widened windows and the round trip
    re-seeds those — different numbers, nobody was refused them, no mark."""
    doc_a, ch_a = _install(window, gate_a)
    _accept_condemned(window, qtbot)
    bp = window._analysis_panel.baseline_panel

    bp.rough_s.setValue(2.0)         # a hand edit ends the mark: "these are not those"
    bp.base_s.setValue(0.5)
    qtbot.wait(20)
    assert bp.accepted_condemned() == (), "a hand edit must end the mark"
    _detect(window, qtbot)           # the table on screen is now measured with 2.0/0.5

    _install(window, plain)
    qtbot.wait(20)
    window._on_selection(doc_a, ch_a)
    qtbot.wait(50)

    assert _windows(window) == (2.0, 0.5), (
        f"setup: the round trip re-seeded {_windows(window)}, not the edited windows")
    assert bp.accepted_condemned() == (), (
        "the mark came back over windows that were never condemned or accepted: "
        f"{bp.accepted_condemned()!r}")
    assert "ACCEPTED OVER" not in _source_line(window), _source_line(window)


# ================================================================================================
# the same rule at the empty boundary — nothing selected at all
# ================================================================================================

def test_no_baseline_readout_survives_the_last_recording_being_closed(
        window, qtbot, gate_a):
    """Closing every document repoints the panel at nothing. The windows, the source
    line and the mark all described the recording that is gone."""
    doc_a, ch_a = _install(window, gate_a)
    _accept_condemned(window, qtbot)
    _detect(window, qtbot)
    assert "ACCEPTED OVER" in _source_line(window), "setup: nothing was marked"

    window._session.close_document(doc_a)
    qtbot.wait(50)

    bp = window._analysis_panel.baseline_panel
    assert window._analysis_panel._sel is None, "setup: something is still selected"
    assert bp.accepted_condemned() == (), (
        "the standing mark outlived the recording it was about: "
        f"{bp.accepted_condemned()!r}")
    assert "ACCEPTED OVER" not in _source_line(window), _source_line(window)
    assert "measured on this recording" not in _source_line(window), (
        "the source line still claims values measured on a recording that is closed: "
        f"{_source_line(window)!r}")


def test_the_mark_does_not_re_attach_to_a_re_opened_file(
        window, qtbot, gate_a, tmp_path):
    """A doc_id is per-LOAD. Re-opening the same path is a different recording and its
    user has been refused nothing — the same rule ``_on_document_closed`` already keeps
    for the FDR and recovery-efficiency state."""
    doc_a, ch_a = _install(window, gate_a)
    _accept_condemned(window, qtbot)
    _detect(window, qtbot)
    accepted = _windows(window)
    window._session.close_document(doc_a)
    qtbot.wait(30)

    again = _doc(tmp_path, "gate_a.dat",
                 _gating(500_000, 50_000.0, open_ms=60, blocked_ms=40,
                         level=100.0, depth=40.0, noise=1.5, seed=0), 50_000.0)
    doc_2, ch_2 = _install(window, again)
    qtbot.wait(30)
    bp = window._analysis_panel.baseline_panel
    bp.rough_s.setValue(accepted[0])   # the very windows, typed by hand this time
    bp.base_s.setValue(accepted[1])
    qtbot.wait(20)
    _detect(window, qtbot)
    qtbot.wait(30)

    assert bp.accepted_condemned() == (), (
        "a mark from a CLOSED load re-attached to a fresh open of the same file: "
        f"{bp.accepted_condemned()!r}")
    assert "ACCEPTED OVER" not in _source_line(window), _source_line(window)


# ================================================================================================
# the REFUSED notice — a control claiming a state it no longer has
# ================================================================================================

def test_the_refused_notice_does_not_follow_the_panel_to_another_recording(
        window, qtbot, gate_a, tmp_path):
    """The notice names a proposal and the checks that condemned it. A target change
    drops the proposal, so on the next recording there is nothing it could be about —
    and this one cannot even be measured, which is the case ``set_gate`` cannot hide."""
    _install(window, gate_a)
    _probe(window, qtbot)
    bp = window._analysis_panel.baseline_panel
    assert bp.condemned_checks(), "fixture drifted: nothing to refuse"
    bp.apply_button.click()
    qtbot.wait(20)
    assert not bp.gate_label.isHidden() and "REFUSED" in bp.gate_label.text(), (
        "setup: the refusal was never posted")

    volts = _doc(tmp_path, "volts.dat",
                 np.zeros(2_000, dtype=np.float32), 50_000.0,
                 kind=ChannelKind.VOLTAGE)
    _doc_id = window._session.add_document(volts)
    window._create_default_layers(volts)
    window._on_selection(_doc_id, 0)
    qtbot.wait(50)

    assert "REFUSED" not in bp.gate_label.text() or bp.gate_label.isHidden(), (
        "the refusal from the previous recording is still on screen, naming checks "
        f"that condemned nothing here: {bp.gate_label.text()[:90]!r}")


def test_the_refused_notice_does_not_outlive_the_last_document(
        window, qtbot, gate_a):
    """Same sentence, the empty boundary. ``set_gate(False, 'Select a CURRENT channel…')``
    neither empties the label nor hides it, so the banner stayed up over no recording."""
    doc_a, ch_a = _install(window, gate_a)
    _probe(window, qtbot)
    bp = window._analysis_panel.baseline_panel
    bp.apply_button.click()
    qtbot.wait(20)
    assert not bp.gate_label.isHidden(), "setup: the refusal was never posted"

    window._session.close_document(doc_a)
    qtbot.wait(50)

    assert window._analysis_panel._sel is None
    assert bp.gate_label.isHidden(), (
        "a REFUSED banner is still visible with no recording open: "
        f"{bp.gate_label.text()[:90]!r}")
    assert bp.gate_label.text() == "", bp.gate_label.text()
