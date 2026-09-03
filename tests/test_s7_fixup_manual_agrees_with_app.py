"""Sweep-7 fix-up: two hand-written paragraphs of the manual quoted surfaces sweep 7 rewrote.

`docs/manual-markdown/manual.md` is the PROSE source of truth for the manual (the Control
reference is rendered from `docs/inventory.json` at build time and dropped from the markdown --
see `docs/BUILDING-THE-MANUAL.md`), so a sentence that quotes a widget can only be checked
against the widget. After the fix batch two of them could not be (fix-up defect X-SURF-8):

* the `Measure this recording` walkthrough put a quotation-marked quote of the button's tooltip
  on the page -- *"usually a few seconds - measured 2 s on a 3 M-sample recording and 6 s on
  9 M samples"* -- that the tooltip does not contain: the live tooltip opens the clause with a
  capital *Usually*, sets its dash as two hyphens (``--``), and carries a parenthesis about the
  ``Measuring...`` label that the manual's version dropped. Quotation marks are a promise that
  the reader can find that string on screen.
* the Event-gallery figure caption called *the ``1-9 to label`` hint in the status line* the way
  labels are advertised. Since S7-POP6 the line names ONLY the digits the palette arms, and on a
  session with no palette -- every session until the user defines one -- it drops the digits and
  says where they come from instead. Measured here on a live 60-event gallery.

Both are new tonight: the pre-sweep-7 tree's manual agreed with the pre-sweep-7 tree's app (its
tooltip said "about a fifth of a second" and its status line always said "1-9 to label"), so the
snapshot at /Users/abrahamcampos/Desktop/nanoscope-reports/pre-sweep7-fixes-2026-09-01 is not the
oracle here -- the app is. These tests therefore READ THE APP FIRST, exactly the way
``tests/test_s7_fixup_user_guide_agrees_with_app.py`` and
``tests/test_docs_merge_and_format_agree_with_app.py`` do, and check the manual against what the
widgets actually say. A doc test that hard-codes the sentence it is checking rots in step with
the doc.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.main_window import MainWindow

_MANUAL = Path(__file__).resolve().parents[1] / "docs" / "manual-markdown" / "manual.md"

FS = 100_000.0


# ================================================================================================
# helpers
# ================================================================================================

def _flat(text: str) -> str:
    """One footing for prose and widget text: the markdown re-flows its lines and the status
    line pads its separators with three spaces, so runs of whitespace collapse. NOTHING ELSE
    is normalised -- not the dashes, not the ellipses, not the case. Half of this defect WAS
    the punctuation: a quote whose em dash the widget writes as ``--`` is not that widget's
    words, and a test that normalised them apart could not have caught it."""
    return " ".join(text.split())


@pytest.fixture(scope="module")
def manual() -> str:
    return _MANUAL.read_text(encoding="utf-8")


def _flat_manual(manual: str) -> str:
    """The manual with its emphasis markers and typographic quotes taken off -- an app string
    quoted on the page wears ``*...*`` and curly quotes around it, never inside it."""
    return _flat(re.sub(r"[*`“”]", "", manual))


# ================================================================================================
# the walkthrough's quote of the probe button's tooltip (X-SURF-8, first half)
# ================================================================================================

_TOOLTIP_LEAD = "The button's own tooltip says the same"


def test_the_manual_quotes_the_probe_tooltip_with_the_characters_the_button_uses(
        manual, qapp, qtbot):
    """THE APP FIRST: read the live tooltip off a real ``BaselinePanel``. THEN THE MANUAL: the
    run of text it presents in quotation marks as that tooltip must be a VERBATIM substring of
    it -- same words, same case, same dash. The pre-fix page quoted 'usually a few seconds -
    measured 2 s...' with a lower-case u and an em dash; the tooltip has neither."""
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    tip = panel.probe_button.toolTip()
    assert "propose a rough and base window" in tip, (
        f"the probe tooltip was rewritten again; re-read it before trusting this test: {tip!r}")

    flat = _flat(manual)
    assert _TOOLTIP_LEAD in flat, (
        "the manual no longer introduces the tooltip quote; this test needs re-aiming")
    after = flat.split(_TOOLTIP_LEAD, 1)[1]
    quoted = re.findall(r"“(.+?)”", after)
    assert quoted, "the manual introduces the tooltip and then quotes nothing"
    claim = _flat(re.sub(r"[*`]", "", quoted[0]))
    assert claim.rstrip(".") in _flat(tip), (
        "docs/manual-markdown/manual.md puts a string in quotation marks that the probe "
        f"button's tooltip does not contain.\n  manual : {claim!r}\n  tooltip: {tip!r}")


def test_the_manual_names_the_label_the_button_wears_while_the_probe_runs(manual, qapp, qtbot):
    """The same paragraph promises the button reads a particular thing while it works. Read
    that off ``set_busy`` rather than off the source line, and check the page says it."""
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    idle = panel.probe_button.text()
    panel.set_busy(True)
    busy = panel.probe_button.text()
    panel.set_busy(False)
    assert busy != idle, "the button no longer changes its label while measuring"

    para = _flat(manual).split(_TOOLTIP_LEAD, 1)[1][:400]
    assert busy in para, (
        f"the manual does not say the button reads {busy!r} while the probe runs: {para!r}")


# ================================================================================================
# the gallery caption's account of the digit hint (X-SURF-8, second half)
# ================================================================================================

def _trace(n_events: int = 60, seed: int = 3) -> np.ndarray:
    """The builder ``tests/gui/test_event_gallery.py`` uses: many events, different dwells and
    depths, so the gallery has a real page to state a count for."""
    rng = np.random.default_rng(seed)
    n = 400_000
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for i in range(n_events):
        s = 2_000 + i * 6_000
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def gallery(qapp, qtbot, tmp_path) -> EventGalleryWindow:
    """A real gallery over a real detection, opened from the menu action the user uses."""
    arr = _trace()
    csv = tmp_path / "many.csv"
    pd.DataFrame({"time": np.arange(arr.size) / FS,
                  "current": np.asarray(arr, dtype=np.float64)}).to_csv(csv, index=False)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=csv, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    w = MainWindow()
    qtbot.addWidget(w)
    doc_id = w._session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    w._create_default_layers(w._session.document(doc_id))
    w._on_selection(doc_id, 0)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    w._act_event_gallery.trigger()
    qtbot.wait(30)
    gal = w._event_gallery
    assert gal is not None, "the menu action opened no gallery"
    yield gal
    w._analysis_controller.abort()
    w._controller.abort()


def _palette(n: int) -> LabelVocabulary:
    v = LabelVocabulary()
    for i in range(n):
        v.add("molecule", f"m{i}")
    return v


def _status_lines(gal: EventGalleryWindow, qtbot) -> dict[str, str]:
    """The live status line in the palette states the manual describes."""
    lines = {}
    gal.set_vocabulary(LabelVocabulary())
    qtbot.wait(10)
    lines["empty"] = gal.status.text()
    gal.set_vocabulary(_palette(2))
    qtbot.wait(10)
    lines["two"] = gal.status.text()
    gal.apply_hotkey("5")            # a digit this palette does not arm
    qtbot.wait(10)
    lines["unarmed"] = gal.status.text()
    gal.set_vocabulary(_palette(9))
    qtbot.wait(10)
    lines["nine"] = gal.status.text()
    return lines


def test_the_gallery_status_line_advertises_only_the_digits_the_palette_arms(gallery, qtbot):
    """The premise of the caption, measured before it is checked: on an empty palette the line
    carries no digit hint at all, and the nine-digit form the manual's screenshot shows is
    reachable only with nine labels defined."""
    lines = _status_lines(gallery, qtbot)
    assert "to label" not in lines["empty"], lines["empty"]
    assert "no labels defined" in lines["empty"], lines["empty"]
    assert "1–2 to label" in lines["two"] and "1–9" not in lines["two"], lines["two"]
    assert "1–9 to label" in lines["nine"], lines["nine"]
    assert "5 is not armed" in lines["unarmed"], lines["unarmed"]


def test_every_status_line_the_gallery_caption_quotes_is_one_the_window_prints(
        gallery, qtbot, manual):
    """THE INVARIANT: each backticked run in the manual's *What that hint says today*
    paragraph is a claim about this window's status line, so each one must appear verbatim in
    a line the window actually printed in one of the palette states above. That is what the
    pre-fix caption failed -- it quoted `1–9 to label` as the hint of a session that has no
    palette, where the line says 'no labels defined - define some under Analysis > Event
    labels... to arm the digit keys' instead."""
    lines = [_flat(v) for v in _status_lines(gallery, qtbot).values()]
    para = re.search(r"\*\*What that hint says today\.\*\*.*?(?=\n\n)", manual, re.S)
    assert para, ("docs/manual-markdown/manual.md no longer has the paragraph that states what "
                  "the gallery's digit hint reads; the caption above the picture must not be "
                  "left as the only account of it")
    quotes = re.findall(r"`([^`]+)`", para.group(0))
    assert len(quotes) >= 4, f"the paragraph quotes almost nothing: {quotes}"
    unmatched = [q for q in quotes if not any(_flat(q) in line for line in lines)]
    assert not unmatched, (
        "the manual quotes the gallery status line saying things it does not say: "
        f"{unmatched}\n  measured lines: {lines}")


def test_the_manual_documents_the_empty_palette_reading_a_fresh_session_gets(gallery, qtbot,
                                                                             manual):
    """The reading EVERY session starts on -- LabelVocabulary() is empty until the user builds
    a palette -- has to be on the page, and the page must no longer present the nine-digit form
    as the hint the status line carries."""
    empty = _status_lines(gallery, qtbot)["empty"]
    clause = _flat(empty).split("·")[-3].strip()   # the palette clause, before the two gestures
    assert "no labels defined" in clause, clause
    flat = _flat_manual(manual)
    assert _flat(clause) in flat, (
        f"the manual does not carry the fresh-session reading of the status line: {clause!r}")
    assert "1–9 to label hint in the status line" not in flat, (
        "the manual still calls '1–9 to label' THE hint in the status line; it is the reading "
        "of a nine-entry palette only")
