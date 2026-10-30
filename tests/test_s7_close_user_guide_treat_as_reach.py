"""Sweep-7 close-out (D4): where §3's "Treat this channel as" bullet states the override's reach.

CW-4 gave the bullet the reach it had been missing — the pane/overlay-group reading was all it
had — but it stated that reach AFTER the phrase that ends the sentence describing what *Apply*
commits ("once *Apply* commits it, it decides … — which **pane** … *and* which channel the
analysis acts on. **Detect**, …"). Anything that reads the bullet up to that phrase — the D4
repro's own extraction, ``docs/sweep7/repros`` style, and a user who reads one sentence and
stops — therefore still met the layout-only sentence first. This module pins the ordering: the
clause that ends at *"once Apply commits it."* must ITSELF name what the override moves.

THE APP FIRST, then the document, the way ``tests/test_s7_fixup_user_guide_agrees_with_app.py``
and ``tests/io/test_s3_io_guide.py`` are written. Measured here on a real MainWindow, a real
Calibration-panel combo and a real *Apply*, on a synthetic recording whose header declares
channel 0 VOLTAGE and channel 1 nothing (sidecar lands under ``tmp_path``): treating channel 1
as *current* moves every surface the bullet credits — Detect False->True, Analysis ▸ Run
detection False->True, Batch detect False->True, Noise spectrum False->True, I-V False->True,
the "Measure this recording" gate False->True ("Detection runs on CURRENT channels; this is a
unknown channel." -> runnable), the status-bar chip "No current channel" -> "✓ Gain set: ×1 →
pA", and the workflow strip's DETECT stage locked -> available.

Note for anyone re-measuring: ``BaselinePanel`` never calls ``setEnabled(False)`` (a disabled
button cannot answer "why not?"), so ``probe_button.isEnabled()`` is True on both sides and is
NOT the gate — read ``BaselinePanel._gate_ok``. Reading the button instead is how a first pass
here concluded, wrongly, that the guide over-credited "Measure this recording".
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

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_catalog import ANALYSIS_LABELS
from nanoscope.gui.analysis_explorer import PAGE_NOISE
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.workflow_status import Stage

_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "USER_GUIDE.md"

#: The name the Analysis menu gives the PSD surface — ``main_window`` builds the menu item from
#: this very label, and the guide and the combo tooltip use it. Read, never copied: 3775a55
#: (2026-09-11) renamed it from "Power spectrum" to "Noise spectrum", and the hard-coded copy
#: this module used to carry went stale with it.
_NOISE = ANALYSIS_LABELS[PAGE_NOISE]

FS = 50_000.0
N = 20_000

#: The bullet, and the phrase that closes its "what Apply commits" sentence. The D4 repro
#: (docs/sweep7 scratch) extracts exactly this span with a NON-GREEDY match, so prose added
#: after the phrase is not part of what it — or a one-sentence reader — sees.
_BULLET_OPENING = "- **Treat this channel as**"
_COMMITS_CLAUSE = re.compile(
    r"\*\*Treat this channel as\*\*.*?once \*Apply\* commits it\.", re.S)


def _norm(text: str) -> str:
    """Prose and widget text on one footing: the guide sets ``--`` as an em dash, wraps quotes
    in ``*...*`` and code in backticks, and re-flows its lines at 100 columns."""
    t = text.replace("—", "--").replace("–", "--").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"').replace("…", "...")
    t = re.sub(r"[*`]", "", t)
    return " ".join(t.split())


@pytest.fixture(scope="module")
def guide() -> str:
    return _GUIDE.read_text(encoding="utf-8")


@pytest.fixture
def commits_clause(guide) -> str:
    m = _COMMITS_CLAUSE.search(guide)
    assert m, (
        "docs/USER_GUIDE.md §3 no longer has a 'Treat this channel as' clause ending "
        "'once *Apply* commits it.' — that phrase is what the D4 repro and this module "
        "anchor on; keep it at the END of the sentence that says what Apply commits")
    return _norm(m.group(0))


@pytest.fixture
def bullet(guide) -> str:
    m = re.search(re.escape(_BULLET_OPENING) + r".*?(?=\n\n|\n- )", guide, re.S)
    assert m, "the guide no longer has a bullet opening " + repr(_BULLET_OPENING)
    return _norm(m.group(0))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _half_declared(path: Path) -> TraceDocument:
    """Channel 0 declared VOLTAGE, channel 1 declared nothing — so the only thing standing
    between this recording and the whole pipeline is which channel is the current one."""
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(
                             ChannelInfo(index=0, name="IN14", raw_units="mV",
                                         kind=ChannelKind.VOLTAGE),
                             ChannelInfo(index=1, name="Untitled 1", raw_units="",
                                         kind=ChannelKind.UNKNOWN)))
    rng = np.random.default_rng(5)
    arrays = tuple((-35.0 + rng.normal(0.0, 1.0, N)).astype(np.float32) for _ in range(2))
    return TraceDocument(meta=meta, channels=arrays)


def _surfaces(window) -> dict[str, object]:
    """Every surface the bullet credits to the override, keyed by the PHRASE the guide uses
    for it — read off the live widgets, after running the syncs the app itself runs."""
    window._sync_detection_action()
    window._sync_batch_detect_action()
    window._sync_analysis_surfaces()
    window._refresh_workflow_status()
    panel = window._analysis_panel
    return {
        "Detect": panel._detection.detect_button.isEnabled(),
        "Run detection": window._act_run_detection.isEnabled(),
        "Batch detect": window._act_batch_detect.isEnabled(),
        _NOISE: window._act_psd.isEnabled(),
        "I-V": window._act_iv.isEnabled(),
        # NOT probe_button.isEnabled(): the baseline panel gates by styling, never by
        # setEnabled(False), so the button is enabled on both sides. This is the gate.
        "Measure this recording": panel._baseline._gate_ok,
        "calibration chip": window._calibration_chip._label.text(),
        "workflow strip": window._workflow_status.stage(Stage.DETECT).available,
    }


@pytest.fixture
def moved(window, qtbot, tmp_path) -> list[str]:
    """The surfaces that really move when *Apply* commits 'treat channel 1 as current' —
    measured through the real combo and the real Apply button, never read off connect()."""
    doc = _half_declared(tmp_path / "half_declared.csv")
    did = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(did, 1)
    window._calibration_panel.set_selection(did, 1)
    qtbot.wait(20)
    before = _surfaces(window)

    cp = window._calibration_panel
    cp._treat_as.setCurrentIndex(cp._treat_as.findData(ChannelKind.CURRENT))
    cp._units.clear()
    qtbot.keyClicks(cp._units, "pA")
    cp._btn_apply.click()
    qtbot.wait(50)
    assert cp.status_text().startswith("Applied"), cp.status_text()
    assert window._session.effective_kind(did, 1) is ChannelKind.CURRENT
    after = _surfaces(window)
    return [k for k in before if before[k] != after[k]]


def test_the_override_moves_every_surface_the_bullet_credits_it_with(moved):
    """The premise the guide's sentence rests on, measured first: one *Apply* on the combo
    turns the whole analysis on for a channel the file declared nothing about."""
    assert set(moved) == {"Detect", "Run detection", "Batch detect", _NOISE, "I-V",
                          "Measure this recording", "calibration chip", "workflow strip"}, moved


def test_the_clause_that_ends_at_apply_commits_it_names_what_apply_commits(
        commits_clause, moved):
    """D4: the sentence that says what *Apply* commits must name the analysis reach INSIDE
    itself. Before this fix it named the pane and the overlay group and stopped; everything
    measured in ``moved`` was described only in the sentences after it."""
    missing = [name for name in moved if name.lower() not in commits_clause.lower()]
    assert not missing, (
        f"docs/USER_GUIDE.md §3: the clause ending 'once *Apply* commits it.' does not name "
        f"{missing}, which one Apply measurably moves. It reads: {commits_clause!r}")


def test_the_clause_keeps_the_pane_and_overlay_group_half_that_is_still_true(commits_clause):
    """The layout reading was never wrong, only incomplete — the override still decides the
    pane and the overlay group, and S7-CAL1's whole point must survive the widening."""
    assert "pane" in commits_clause and "overlay group" in commits_clause, commits_clause


def test_the_bullet_and_the_combo_tooltip_name_the_same_surfaces(qapp, qtbot, bullet):
    """Two surfaces, one description: every surface the shipped tooltip on the real combo
    claims (calibration_panel.py) is a surface the guide's bullet claims too — so a user who
    hovers and a user who reads are told the same thing."""
    from nanoscope.core.session import Session

    panel = CalibrationPanel(Session())
    qtbot.addWidget(panel)
    tip = _norm(panel._treat_as.toolTip())
    vocabulary = ("Run detection", "Batch detect", _NOISE, "I-V",
                  "calibration chip", "workflow strip", "pane")
    claimed = [phrase for phrase in vocabulary if phrase.lower() in tip.lower()]
    assert len(claimed) >= 6, f"the tooltip stopped naming its surfaces: {tip!r}"
    missing = [phrase for phrase in claimed if phrase.lower() not in bullet.lower()]
    assert not missing, (
        f"the combo tooltip names {missing} and USER_GUIDE.md §3's bullet does not: "
        f"tooltip {tip!r}")
