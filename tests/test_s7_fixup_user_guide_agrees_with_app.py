"""Sweep-7 fix-up: docs/USER_GUIDE.md described two controls the sweep-7 fixes had rewritten.

The guide is a SURFACE, and after the fix batch two of its paragraphs contradicted the app
they describe (docs/sweep7/FINDINGS.md, fix-up defects E5, X-SURF-11 and CW-4):

* §5a's "A line under the windows always says where the current numbers came from" listed
  the readings as they were BEFORE sweep 7: it named *fell back -- the estimator could not
  run on this trace* as "the short-segment case", a stamp no fresh run produces since
  S7-DET3 (a clamped run keeps its caller's source -- measured on a 4,000-sample cut of the
  owner's -600 mV file: source 'default', label "app default (this recording was not
  measured)", plus the amber "actually ran at rough 0.08 s / base 0.016 s ... CLAMPED"
  line), and it did not list the 'batch' reading S7-DET1 added ("COPIED from the batch
  form ...").
* §3's **Treat this channel as** bullet said the override "decides which pane the channel
  gets and which overlay group it shares" -- the layout-only reading S7-CAL1 was about --
  after the D + C2 fixes made it the one resolver the pipeline follows. Measured live on a
  2 s CSV of the owner's own -700 mV recording with neutral headers: the gate triple
  (Detect button, can_run, Analysis > Run detection) went (False, False, False) ->
  (True, True, True) on Apply.

Written the way ``tests/io/test_s3_io_guide.py`` and
``tests/test_docs_merge_and_format_agree_with_app.py`` are: read the app's own behaviour
FIRST -- the live label table, a real clamped run through the real BaselinePanel, the real
Detect gate through a real MainWindow -- then check the document against THAT. A doc test
that hard-codes the sentence it is checking rots in step with the doc.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.baseline_panel import _SOURCE_LABELS, BaselinePanel
from nanoscope.gui.main_window import MainWindow

_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "USER_GUIDE.md"

FS = 50_000.0


# ================================================================================================
# helpers
# ================================================================================================

def _norm(text: str) -> str:
    """Prose and widget text on one footing: the guide sets ``--`` as an em dash, wraps
    quotes in ``*...*`` and code in backticks, and re-flows lines at 100 columns."""
    t = text.replace("—", "--").replace("–", "--").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"').replace("…", "...")
    t = re.sub(r"[*`]", "", t)
    return " ".join(t.split())


@pytest.fixture(scope="module")
def guide() -> str:
    return _GUIDE.read_text(encoding="utf-8")


def _paragraph(guide: str, opening: str) -> str:
    """The paragraph that starts with ``opening`` (up to the next blank line)."""
    m = re.search(re.escape(opening) + r".*?(?=\n\n)", guide, re.S)
    assert m, f"the guide no longer has a paragraph opening {opening!r}"
    return m.group(0)


def _bullet(guide: str, opening: str) -> str:
    """A list item that starts with ``opening`` -- ends at the next blank line or the next
    top-level bullet."""
    m = re.search(re.escape(opening) + r".*?(?=\n\n|\n- )", guide, re.S)
    assert m, f"the guide no longer has a bullet opening {opening!r}"
    return m.group(0)


# ================================================================================================
# §5a -- the readings of the "These values" line (E5 / X-SURF-11)
# ================================================================================================

_SOURCE_PARAGRAPH = "A line under the windows always says where the current numbers came from:"


def test_the_guide_lists_every_source_a_fresh_run_can_stamp(guide):
    """Every entry of ``PARAM_SOURCES`` the app stamps on a NEW run has its panel label in
    the paragraph, read off the live label table. 'fallback' is excluded by name because
    baseline.py keeps it only as a legacy stamp a pre-sweep-7 project can carry -- the
    guide mentions it as exactly that, and the assertion below is about fresh runs."""
    fresh = [s for s in B.PARAM_SOURCES if s != "fallback"]
    assert "default" in fresh and "proposed" in fresh
    para = _norm(_paragraph(guide, _SOURCE_PARAGRAPH))
    missing = [s for s in fresh if _norm(_SOURCE_LABELS[s]) not in para]
    assert not missing, (
        f"USER_GUIDE.md §5a does not list the panel's label for {missing}: "
        f"{[_SOURCE_LABELS[s] for s in missing]}")


def test_the_guide_does_not_present_the_retired_fell_back_stamp_as_a_reachable_reading(guide):
    """The pre-sweep-7 label "fell back -- the estimator could not run on this trace" is
    gone from the app (the panel's 'fallback' entry now reads as a legacy stamp); the
    guide must not name it, and must not name ANY reading as "the short-segment case"
    other than the clamp line, which is what a short segment actually shows."""
    para = _norm(_paragraph(guide, _SOURCE_PARAGRAPH))
    assert "fell back -- the estimator could not run on this trace" not in para
    assert "the estimator could not run on this trace" not in para
    for label in _SOURCE_LABELS.values():
        assert "could not run on this trace" not in label, (
            "the app re-grew the retired label; re-check this test's premise")


def test_a_clamped_run_keeps_its_source_and_the_guide_describes_what_the_box_shows(
        guide, qapp, qtbot):
    """THE APP FIRST: a trace shorter than the tracking window runs the configured
    estimator at clamped widths, keeps the caller's source ('default' -- nobody chose
    these windows) and records the clamp beside it; the real BaselinePanel then shows the
    unchanged "app default" line plus the amber "actually ran at ... CLAMPED" line.
    THEN THE GUIDE: §5a says exactly that of a short recording, and the short-segment
    section it points at says what the box shows."""
    rng = np.random.default_rng(7)
    x = (-35.0 + rng.normal(0.0, 1.0, 4000)).astype(np.float32)
    x[1500:2500] -= 88.0
    res = B.characterisation_baseline(x, FS, polarity=-1)
    prov = res.provenance()
    assert res.param_source == "default", res.param_source
    assert prov["baseline_fallback"] == B.FALLBACK_CLAMPED
    assert prov["baseline_rough_s"] == pytest.approx(4000 / FS)

    panel = BaselinePanel()
    qtbot.addWidget(panel)
    panel.set_recording(FS, ("doc", 0))
    panel.set_params({"baseline_rough_s": B.DEFAULT_BASELINE.rough_s,
                      "baseline_base_s": B.DEFAULT_BASELINE.base_s,
                      "baseline_param_source": res.param_source})
    panel.set_run_windows(prov)
    source_line = panel.source_label.text()
    assert source_line == "These values: " + _SOURCE_LABELS["default"], source_line
    assert panel.run_windows_label.isVisibleTo(panel)
    clamp_line = panel.run_windows_label.text()
    assert "actually ran at" in clamp_line and "CLAMPED" in clamp_line, clamp_line

    para = _norm(_paragraph(guide, _SOURCE_PARAGRAPH))
    # ...the guide's "short-segment" reading IS the clamp line, under an unchanged source
    assert "actually ran at" in para and "CLAMPED" in para, para
    assert "baseline_fallback" in para
    assert "short-segment case below" in para
    head, _, _tail = para.partition("actually ran at")
    assert _norm(_SOURCE_LABELS["default"]) in head
    # ...and the section the paragraph points at says what the box shows
    section = guide.split("### Short recordings and short segments", 1)[1]
    section = _norm(section.split("\n### ", 1)[0])
    assert "baseline_fallback = windows_clamped_to_recording" in section
    assert "actually ran at" in section and "CLAMPED" in section
    assert "These values" in section


# ================================================================================================
# §3 -- what "Treat this channel as" reaches (CW-4)
# ================================================================================================

N = 20_000


def _doc(path, channels) -> TraceDocument:
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    arrays = tuple(np.zeros(N, dtype=np.float32) for _ in channels)
    return TraceDocument(meta=meta, channels=arrays)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _open(window, doc, channel):
    did = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(did, channel)
    window._calibration_panel.set_selection(did, channel)
    return did


_TREAT_AS_BULLET = "- **Treat this channel as**"


def test_the_override_moves_the_detect_gate_and_the_guide_says_what_it_reaches(
        window, qtbot, tmp_path, guide):
    """THE APP FIRST, through the real combo + Apply on a synthetic undeclared recording
    under tmp_path (any sidecar lands there): the Detect button, ``can_run`` and Analysis >
    Run detection go from refused to runnable, and Batch detect targets the channel.
    THEN THE GUIDE: the bullet names detection, Run detection, Batch detect and the
    workflow strip as things the override reaches -- not a pane and an overlay group only."""
    doc = _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)))
    did = _open(window, doc, 0)
    panel = window._analysis_panel
    det = panel._detection
    qtbot.wait(20)
    before = (det.detect_button.isEnabled(), panel.can_run(),
              window._act_run_detection.isEnabled())
    assert before == (False, False, False), before

    cp = window._calibration_panel
    cp._treat_as.setCurrentIndex(cp._treat_as.findData(ChannelKind.CURRENT))
    cp._units.clear()
    qtbot.keyClicks(cp._units, "pA")
    cp._btn_apply.click()
    qtbot.wait(50)
    assert cp.status_text().startswith("Applied"), cp.status_text()
    assert window._session.effective_kind(did, 0) is ChannelKind.CURRENT
    window._sync_detection_action()
    after = (det.detect_button.isEnabled(), panel.can_run(),
             window._act_run_detection.isEnabled())
    assert after == (True, True, True), (
        f"the override no longer moves the gate ({before} -> {after}); the guide's "
        "sentence about it, and this test, need re-measuring")
    assert [t[1] for t in window._detectable_targets()] == [0]

    bullet = _norm(_bullet(guide, _TREAT_AS_BULLET))
    for claim in ("Detect", "Run detection", "Batch detect", "workflow strip"):
        assert claim.lower() in bullet.lower(), (
            f"USER_GUIDE.md §3 'Treat this channel as' does not say the override reaches "
            f"{claim!r}; measured gate {before} -> {after}")
    assert "pane" in bullet and "overlay group" in bullet, (
        "the pane / overlay-group half is still true and must stay")


def test_the_guide_quotes_the_refusal_an_override_causes_as_the_app_words_it(
        window, qtbot, tmp_path, guide):
    """The reverse override -- IN15, header current, marked VOLTAGE -- switches detection
    off and the refusal names the override rather than quoting the header. The guide's
    bullet carries that sentence, read off the live gate label."""
    doc = _doc(tmp_path / "declared.abf", (
        ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)))
    did = _open(window, doc, 1)
    panel = window._analysis_panel
    qtbot.wait(20)
    assert panel.can_run()
    window._session.set_calibration(
        did, 1, ChannelCalibration(1.0, 0.0, "mV", kind=ChannelKind.VOLTAGE),
        write_sidecar=False)
    qtbot.wait(20)
    assert not panel.can_run()
    refusal = panel._detection._gate_label.text()
    assert "Calibration-panel override" in refusal and "file says current" in refusal, refusal

    bullet = _norm(_bullet(guide, _TREAT_AS_BULLET))
    assert _norm(refusal) in bullet, (
        f"USER_GUIDE.md §3 does not quote the override refusal as the app words it: "
        f"{refusal!r}")
