"""S7-POP1 (inspector half) — ``EventInspectorWindow.set_stale_columns`` marks every readout
line that repeats a badged events-table column, carries the banner's sentence, marks the
plot title, follows navigation, and clears with ``()``.

Measured before it existed: after one knob in the Preprocess tab the events table badged
nine headers ⚠ and said "Depth, frac_depth, ECD, mean/max deviation and SNR are stale",
while the inspector went on printing "depth: 122.37 pA", "SNR: 21.4", "baseline: -61.079 pA"
with no mark.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N_EVENTS = 12
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000
_NOTE = "Depth, frac_depth, ECD, mean/max deviation and SNR are stale."


def _signal() -> np.ndarray:
    rng = np.random.default_rng(9)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pop1.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov),
        # Stamped with the live estimator, so the table -- and, now that MainWindow pushes
        # the table's badge set onto this window on every feed (S7-POP1 wiring), the
        # inspector -- start CLEAN. An unstamped result predates the estimator and is
        # judged stale on open, which is the owner's-project case, not this fixture's.
        baseline_fingerprint=w._analysis_panel.baseline_panel.config().fingerprint(FS)))
    w._open_event_inspector()
    qtbot.wait(20)
    insp = w._event_inspector
    insp.show()
    qtbot.wait(20)
    yield w, doc_id, insp
    w._analysis_controller.abort()


def _stale_body(insp) -> str:
    """The staleness sentence as the Event box carries it: its fixed-height note
    (``stale_note()``), NOT readout lines -- spliced into the readout as wrapped lines the
    sentence resized the waveform plot under the pointer on every flip (sweep-7 fix-up
    B3 / CW-2; see test_s7_fixup_inspector_stale_note_geometry.py)."""
    return insp.stale_note()


def _lines(insp) -> dict[str, str]:
    """``{label: rest}`` for the 'label:   value' lines of the readout."""
    out = {}
    for line in insp._readout.text().splitlines():
        head, sep, rest = line.partition(":")
        if sep and not head.startswith("Event"):
            out[head.strip()] = rest.strip()
    return out


def test_set_stale_columns_badges_each_affected_line_and_the_title_and_clears(wired, qtbot):
    w, doc_id, insp = wired
    assert insp.current_event_id() is not None
    text = insp._readout.text()
    assert "⚠" not in text, text         # 0.3 blockades: no dI/I0>1 glyph to confuse this
    assert "⚠" not in insp._plot.titleLabel.text

    insp.set_stale_columns(REMEASURED_COLUMNS, _NOTE)
    qtbot.wait(10)
    assert insp.stale_columns() == tuple(REMEASURED_COLUMNS)
    assert _stale_body(insp) == _NOTE, insp._readout.text().splitlines()[:3]
    got = _lines(insp)
    for label in ("depth", "dI/I0", "SNR", "rise / fall", "baseline"):
        assert got[label].endswith(" ⚠"), (label, got[label])
    for label in ("dwell", "accepted", "levels", "filter-limited"):
        assert not got[label].endswith("⚠"), (label, got[label])
    title = insp._plot.titleLabel.text
    assert "depth" in title and " ⚠" in title, title
    depth_part = next(p for p in title.split("   ·   ") if p.startswith("depth"))
    assert depth_part.endswith(" ⚠"), depth_part
    dwell_part = next(p for p in title.split("   ·   ") if p.startswith("dwell"))
    assert not dwell_part.endswith("⚠"), dwell_part

    insp.set_stale_columns(())
    qtbot.wait(10)
    assert "⚠" not in insp._readout.text(), insp._readout.text()
    assert "⚠" not in insp._plot.titleLabel.text


def test_the_mark_follows_navigation_and_a_subset_of_columns_marks_only_those(wired, qtbot):
    w, doc_id, insp = wired
    insp.set_stale_columns(("depth_pA", "baseline_pA"), _NOTE)
    insp.next_event()
    insp.next_event()
    qtbot.wait(10)
    got = _lines(insp)
    assert got["depth"].endswith(" ⚠") and got["baseline"].endswith(" ⚠"), got
    for label in ("dI/I0", "SNR", "rise / fall", "dwell"):
        assert not got[label].endswith("⚠"), (label, got[label])
    assert _stale_body(insp) == _NOTE


def test_without_a_sentence_the_readout_names_the_columns_itself(wired, qtbot):
    w, doc_id, insp = wired
    insp.set_stale_columns(("depth_pA", "snr"))
    qtbot.wait(10)
    body = _stale_body(insp)
    assert body.startswith("depth_pA, snr are stale"), body
    assert "Analysis panel" in body, body


def test_the_tuple_the_events_table_badges_is_the_tuple_the_seam_takes(wired, qtbot):
    """The finding's gesture — one knob in the Preprocess tab — badges the table with
    ``REMEASURED_COLUMNS`` and raises the banner; pushing exactly those through the seam is
    what the MainWindow wiring does."""
    w, doc_id, insp = wired
    ap = w._analysis_panel
    res = ap.current_detection()
    assert res is not None
    ap._baseline.rough_s.setValue(float(ap._baseline.rough_s.value()) * 4.0)
    qtbot.wait(30)
    ap._check_staleness(res)
    qtbot.wait(20)
    cols = tuple(ap._table.stale_columns)
    assert cols == tuple(REMEASURED_COLUMNS), cols
    note = ap._stale.message()
    assert len(note) > 150, note     # the banner names both estimators; it is long
    insp.set_stale_columns(cols, note)
    qtbot.wait(10)
    # The sentence is carried WHOLE, in the box's note and not as readout lines: a mono
    # readout does not wrap, one 300-character line would set the window's minimum width
    # past the 1512 px display the curation bar was measured against, and wrapped lines
    # spliced into the readout resized the plot above it (fix-up B3 / CW-2).
    assert _stale_body(insp) == " ".join(note.split())
    assert "⚠ " not in insp._readout.text(), insp._readout.text().splitlines()[:3]
    assert insp.minimumSizeHint().width() <= 1100, insp.minimumSizeHint().width()
    assert _lines(insp)["depth"].endswith(" ⚠")
