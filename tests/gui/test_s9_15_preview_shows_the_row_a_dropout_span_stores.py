"""S9-15 — A non-finite sample under the cursors no longer blanks the Measure tab's preview.

MEASURED DEFECT (sweep 9, a synthetic 1.2 s / 50 kHz recording, -1000 pA open pore, one 300-sample
blockade with ONE dropped sample, a NaN, in its middle). With Measure on and the region placed on the
blockade, the Measure tab read "Drag a region on the trace, or press C over an event, to see the event
Ctrl+M would store." under the region that was drawn, ``event_preview()`` was None and the Add event
button was greyed out. Ctrl+M on the very same mask then stored event 0 in the Manual table, rejected
automatically as non_finite (status: "Added event 0 to Manual, but it was rejected automatically
(non_finite)."): a row the preview never showed, added by the key while the button beside the preview
said there was nothing to add. An Inf sample did the same. ``MeasurePanel._install_stats`` kept the
owner's ``HandRowPreview`` only when the panel's OWN statistics came out
(``self._preview = preview if stats is not None else None``) and armed the button by those statistics;
``measure_selection`` refuses a span holding a non-finite sample (``NonFiniteSelection``), while the add
door measures the same span (the Manual contract interpolates non-finite samples on purpose) and stores
the row, flagged.

THE FIX. The preview is the owner's answer for the span and stands whatever the panel's statistics
made of it: the line shows the row Ctrl+M stores, "would be rejected: non_finite", and the red note
under the blank statistics still says why they are blank. With an owner, the Add button is armed by
the mask alone, as Ctrl+M is, so every refusal past that is the door's own, which says why (an
overlap or a stale table keeps the button live and refuses on the click, as it did); a panel with no
owner still arms it by its own readout.

These tests drive a real MainWindow, once with a NaN and once with an Inf: Measure turned on as the
Analysis menu turns it on, the region placed on the trace, then Analysis > Add event from cursors
(Ctrl+M) or a click on the Add event button, and compare what the tab showed before the gesture with
the row the gesture stored.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
SPAN = (20_000, 20_300)
DROPOUT = 20_150                   # one sample, in the middle of the event

#: The measured columns the preview prints and the stored row carries.
MEASURED = ("dwell_s", "depth_pA", "baseline_pA", "frac_depth", "t_start_s")


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._dirty = False


@pytest.fixture(params=[np.nan, np.inf], ids=["nan", "inf"])
def dropout(request, win, tmp_path):
    """The recording with a dropout, open and selected, Measure on, the region on the event."""
    rng = np.random.default_rng(0)
    x = np.full(N, I0) + rng.normal(0.0, 8.0, N)
    x[SPAN[0]:SPAN[1]] += 300.0
    x[DROPOUT] = request.param
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "dropout.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(x.astype(np.float32),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    xs = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(xs[SPAN[0]]), float(xs[SPAN[1]]))
    assert win._trace_view.measure_bounds() == SPAN            # a region IS selected
    return doc_id, mp


def _stored(win, doc_id):
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return None if res is None else res.payload.df


def _assert_is_the_previewed_row(stored, preview) -> None:
    assert stored is not None and len(stored) == 1
    row = stored.iloc[0]
    assert (int(row["start_idx"]), int(row["end_idx"])) == SPAN
    assert not bool(row["accepted"]) and "non_finite" in str(row["flag"])
    for key in MEASURED:
        assert float(row[key]) == pytest.approx(float(preview.row[key]), rel=1e-9), key
    assert str(row["flag"]) == str(preview.row["flag"])


def test_the_preview_shows_the_rejected_row_ctrl_m_stores(win, dropout):
    doc_id, mp = dropout
    preview, text = mp.event_preview(), mp.event_preview_text()

    assert preview is not None and preview.row is not None, preview      # was: None
    assert not preview.reason and not preview.refusal
    assert not bool(preview.row["accepted"]) and "non_finite" in str(preview.row["flag"])
    assert text.startswith("→ Manual: dwell "), text                     # was: "Drag a region..."
    assert "would be rejected: non_finite" in text.splitlines()[0], text
    # The statistics are still refused, and still say why: they are not the add.
    assert mp.current_measurement() is None
    assert "non-finite" in mp.refusal_note()

    win._act_add_event.trigger()                                         # Ctrl+M
    _assert_is_the_previewed_row(_stored(win, doc_id), preview)
    assert "rejected automatically (non_finite)" in win.statusBar().currentMessage()


def test_the_add_button_is_armed_for_the_span_ctrl_m_adds_and_stores_the_same_row(win, dropout):
    doc_id, mp = dropout
    preview = mp.event_preview()
    assert mp._add_button.isEnabled()                                   # was: greyed out
    assert _stored(win, doc_id) is None

    mp._add_button.click()
    _assert_is_the_previewed_row(_stored(win, doc_id), preview)
    assert "rejected automatically (non_finite)" in win.statusBar().currentMessage()
