"""The Measure panel must SAY why a readout is blank, not just go blank.

``measure_selection`` refuses a selection (or the open-pore window it is measured
against) that carries NaN/Inf rather than printing a robust number quietly computed on
poisoned data. That refusal is right — and it landed the user in a worse place than the
wrong number did: ``MeasurePanel._measure`` catches ``ValueError`` and returns ``None``,
so the whole readout went to "—" with no explanation anywhere on screen. "—" is also
exactly what the panel shows when nothing is selected, so the state was indistinguishable
from "you have not dragged the cursors yet".

The distinction that makes this safe to show is a TYPED one: only
``NonFiniteSelection`` — a defect in the data — raises the note. Every other
``ValueError`` from that call is an ordinary gesture (a collapsed drag, a one-sample
selection, a channel that went away mid-drag) and must stay silent, because a red banner
on every collapsed drag is how a user learns to ignore the banner that matters.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 300_000
CLEAN = (100_000, 130_000)          # a span with finite samples and finite flanks
POISONED_IN = (200_000, 230_000)    # contains the bad sample
POISONED_FLANK = (150_000, 180_000) # clean inside, bad sample in its left-hand flank

BAD_INSIDE = 210_000
BAD_FLANK = 149_950                 # 50 samples before POISONED_FLANK's start


def _trace(bad=np.inf) -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, -100.0) + rng.normal(0.0, 1.0, N)
    x[200_000:230_000] += 20.0
    x[150_000:180_000] += 20.0
    x[BAD_INSIDE] = bad
    x[BAD_FLANK] = bad
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "poisoned.dat",
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def panel(window, tmp_path):
    doc = _doc(tmp_path, _trace())
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    return mp, doc_id


def _place(window, doc_id: str, i0: int, i1: int) -> None:
    vals = window._session.x_axis(doc_id).values
    x1 = float(vals[i1]) if i1 < vals.size else float(vals[-1])
    window._trace_view.set_measure_region(float(vals[i0]), x1)


# ================================================================================================
# the note appears, and says something the user can act on
# ================================================================================================

def test_a_poisoned_selection_explains_itself_instead_of_going_silently_blank(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *POISONED_IN)

    assert mp.current_measurement() is None, "the refusal itself must stand"
    note = mp.refusal_note()
    assert note, "a blank readout with no reason is the defect this pins"
    assert "non-finite" in note
    assert f"{BAD_INSIDE:,}" in note, "the user needs the sample number to crop past"
    assert "localised taint" in note, "...and to know it is not a dead channel"


def test_a_poisoned_OPEN_PORE_WINDOW_explains_itself_too(panel, window):
    """The half nobody sees coming: every sample under the cursors is finite, and the
    reading is still refused because the reference beside them is not."""
    mp, doc_id = panel
    # The reference beside the cursors is only read when the panel measures on its own. With an
    # owner supplying the table's baseline (interpolated through bad samples) it is not, and the
    # readout stands (test_measure_tab_preview.py); detach the owner to pin the refusal itself.
    mp.set_basis_provider(None)
    _place(window, doc_id, *POISONED_FLANK)

    assert mp.current_measurement() is None
    note = mp.refusal_note()
    assert "open-pore window" in note
    assert f"{BAD_FLANK:,}" in note


def test_the_readout_reads_as_withheld_while_the_note_stands(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *POISONED_IN)

    assert mp.refusal_note()
    for key in ("mean", "median", "amplitude", "snr", "kurtosis"):
        assert mp.readout_value(key) in ("", "—"), key


def test_the_note_clears_when_the_cursors_move_off_the_poisoned_span(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *POISONED_IN)
    assert mp.refusal_note()

    _place(window, doc_id, *CLEAN)
    assert mp.current_measurement() is not None
    assert mp.refusal_note() == ""
    assert mp._refusal_label.isVisible() is False


def test_the_note_does_not_outlive_the_selection_it_describes(panel, window):
    """``self._refusal = None`` sat INSIDE ``_measure``, after its early return -- so
    every path that drops the target or the selection without measuring again left the
    red note standing under an empty readout, describing a selection that no longer
    exists on a channel that is no longer selected. Three doors, all of them ordinary:
    close the document, switch channel, switch Measure off."""
    mp, doc_id = panel
    _place(window, doc_id, *POISONED_IN)
    assert mp.refusal_note(), "the note has to be up before we can watch it not come down"

    mp.set_active(False)                      # the Measure toggle
    assert mp.refusal_note() == "", "the region is gone; the note describes nothing"

    mp.set_active(True)
    _place(window, doc_id, *POISONED_IN)
    assert mp.refusal_note()
    mp.clear()                                # document closed / no channel selected
    assert mp.refusal_note() == ""
    assert mp._refusal_label.isVisible() is False


def test_a_clean_selection_never_raises_the_note(panel, window):
    mp, doc_id = panel
    _place(window, doc_id, *CLEAN)

    stats = mp.current_measurement()
    assert stats is not None and np.isfinite(stats["amplitude"])
    assert mp.refusal_note() == ""


# ================================================================================================
# ...and an ordinary gesture stays silent
# ================================================================================================

def test_a_collapsed_selection_is_not_dressed_up_as_a_data_defect(panel, window):
    """The reason ``NonFiniteSelection`` had to be its own type."""
    mp, doc_id = panel
    _place(window, doc_id, 50_000, 50_001)

    assert mp.current_measurement() is None       # too short to measure
    assert mp.refusal_note() == "", "a one-sample drag is a gesture, not a poisoned trace"


def test_the_note_reads_the_text_not_the_widget_visibility(panel, window):
    """``approx_note`` had to do the same: a child of a window that was never shown
    reports itself invisible, so asking Qt would make this lie in every headless test."""
    mp, doc_id = panel
    _place(window, doc_id, *POISONED_IN)
    assert mp.refusal_note() == mp._refusal_label.text()
