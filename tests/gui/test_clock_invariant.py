"""High-6 — THREE SURFACES, TWO CLOCKS, NO LABELS.

The defect, measured on the owner's own apoferritin recordings:

* the trace x-axis is drawn on the **instrument clock** (``Session.x_axis`` =
  ``arange(n)/fs + meta.t0_s``; ``t0_s`` is 10.2526 / 61.3034 / 0 s on the three
  files),
* the event table's ``t_start_s`` and the PSD window's Start/End are seconds **from
  the start of the recording**,
* and all three were labelled ``s``.

Reading a quiet window off the trace at −500 mV and typing it into the PSD panel
therefore analysed a different stretch: total RMS **22.66 pA / α 1.822** instead of
**3.99 pA / α 0.985** — a 5.7× error in a headline noise number with nothing on
screen to say so. The −700 mV file has ``t0_s = 0``, so on one recording in three the
mistake is invisible.

THE DECISION, and why. The app cannot collapse to one clock: dropping ``t0_s`` throws
away the experimental time the file carries (``io/contract.py`` is explicit that this
is a bug, not a simplification), and rebasing the event table onto the instrument
clock puts an offset into 20 frozen schema columns and into every stored detection
ever written. So it is the second half of the invariant — **every control that shows
or takes a time says which clock it is in** — plus the corollary that actually
removes the arithmetic from the user's head: **a time crossing a surface boundary
travels as sample indices**, and MainWindow (the sole wirer) does the one conversion
in code.

Four carriers, each pinned below: the axis label, the crosshair read-out, the event
table's header, and the Analysis-menu hand-off of the Measure selection to the PSD
window.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 40_000
I0 = -1000.0
DEPTH = 300.0

#: The −600 mV apoferritin recording's own t₀. Used verbatim so the numbers in these
#: tests are the numbers in the report.
T0_APO = 61.3034
SPANS = [(10_000, 10_600), (20_000, 20_800)]


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in SPANS:
        x[s:e] -= DEPTH
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, t0_s: float, name="apo.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,),
        t0_s=float(t0_s))
    return TraceDocument(meta=meta, channels=(signal,))


def _open(win, tmp_path, *, t0_s, name="apo.dat"):
    signal = _signal()
    doc_id = win._session.add_document(_doc(tmp_path, signal, t0_s=t0_s, name=name))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return doc_id, signal


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


# ================================================================================================
# 2. the trace axis
# ================================================================================================
def test_the_trace_x_axis_says_it_is_on_the_instrument_clock(win, tmp_path):
    _open(win, tmp_path, t0_s=T0_APO)
    label = win._trace_view.x_axis_label()
    assert "instrument clock" in label, label
    assert "61.3034" in label, label


def test_the_trace_axis_ticks_are_in_the_unit_the_label_claims(win, tmp_path):
    """pyqtgraph's automatic SI prefix multiplies every tick by 1000 and appends
    "(x0.001)" once the visible span drops below 1 — measured on the trace view as
    ``s (×0.001)`` reading 884.3 for 0.8843 s. Zooming into a blockade is the normal
    way to use this view, so that was the normal state of its axis."""
    _open(win, tmp_path, t0_s=T0_APO)
    win._trace_view.set_x_range(T0_APO + 0.20, T0_APO + 0.21)   # 10 ms — well under 1 s
    for pane in win._trace_view.panes():
        axis = pane.plot.getAxis("bottom")
        assert axis.autoSIPrefix is False
        assert axis.autoSIPrefixScale == 1.0
    assert "×" not in win._trace_view.x_axis_label()
    assert "0.001" not in win._trace_view.x_axis_label()


def test_the_crosshair_states_the_instant_on_both_clocks(win, tmp_path):
    _open(win, tmp_path, t0_s=T0_APO)
    win._session.set_cursor(T0_APO + 0.2)
    texts = [item[2].textItem.toPlainText() for item in win._trace_view.crosshair._items]
    assert texts and "instrument" in texts[0], texts
    assert "0.2000 s in" in texts[0], texts


def test_the_crosshair_does_not_double_print_when_the_clocks_agree(win, tmp_path):
    """No clutter bought for a distinction that does not exist on this file."""
    _open(win, tmp_path, t0_s=0.0)
    win._session.set_cursor(0.2)
    texts = [item[2].textItem.toPlainText() for item in win._trace_view.crosshair._items]
    assert texts and "instrument" not in texts[0], texts


# ================================================================================================
# 3. the event table
# ================================================================================================
def _header(win, name: str, role):
    from PySide6.QtCore import Qt

    model = win._analysis_panel._table._model
    col = list(model.df.columns).index(name)
    return model.headerData(col, Qt.Horizontal, role)


def test_the_event_tables_time_column_says_which_clock_it_is_on(win, tmp_path):
    from PySide6.QtCore import Qt

    _open(win, tmp_path, t0_s=T0_APO)
    assert "t₀=0" in _header(win, "t_start_s", Qt.DisplayRole)
    tip = _header(win, "t_start_s", Qt.ToolTipRole)
    assert "from the start of this recording" in tip
    assert "61.3034" in tip and "add" in tip


def test_a_duration_column_is_left_alone(win, tmp_path):
    """``dwell_s`` is a LENGTH of time. It is on no clock, and annotating it would be
    noise pretending to be rigour."""
    from PySide6.QtCore import Qt

    _open(win, tmp_path, t0_s=T0_APO)
    assert _header(win, "dwell_s", Qt.DisplayRole) == "dwell_s"


def test_the_offset_follows_the_recording_that_is_shown(win, tmp_path):
    """10.25 / 61.30 / 0 s on the owner's three files: a stale offset is a WRONG
    label, not a missing one, so it is pushed on every result."""
    from PySide6.QtCore import Qt

    _open(win, tmp_path, t0_s=T0_APO, name="a.dat")
    assert "t₀=0" in _header(win, "t_start_s", Qt.DisplayRole)
    _open(win, tmp_path, t0_s=0.0, name="b.dat")
    assert _header(win, "t_start_s", Qt.DisplayRole) == "t_start_s"


# ================================================================================================
# 4. the hand-off — the half that removes the arithmetic instead of annotating it
# ================================================================================================
def test_the_measured_region_reaches_the_psd_window_on_the_right_clock(win, tmp_path):
    """THE 5.7x ERROR, and the path that cannot make it.

    The region is placed on the trace in INSTRUMENT seconds; the PSD segment lands in
    RECORDING seconds. Nobody types anything, and the numbers differ by exactly t₀.
    """
    _open(win, tmp_path, t0_s=T0_APO)
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(T0_APO + 0.20, T0_APO + 0.40)
    win._sync_create_segment_action()
    assert win._act_psd_selection.isEnabled()

    win._open_psd_for_selection()
    psd = win._psd_window
    assert psd is not None
    start, end = psd._start_spin.value(), psd._end_spin.value()
    assert start == pytest.approx(0.20, abs=1e-3), start
    assert end == pytest.approx(0.40, abs=1e-3), end
    # ...and the status line shows the pair of numbers that were never on screen together.
    msg = win.statusBar().currentMessage()
    assert "into the recording" in msg and "on the trace" in msg, msg
    assert "61.5034" in msg, msg     # 0.20 s in == 61.5034 s on the trace


def test_the_hand_off_uses_the_recording_the_region_is_on(win, tmp_path):
    """The Measure region does not have to be on the app's selected recording.
    Dividing one recording's indices by another's sample rate is exactly the class of
    error this action exists to remove, so the target travels with the bounds."""
    doc_a, _ = _open(win, tmp_path, t0_s=T0_APO, name="a.dat")
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(T0_APO + 0.20, T0_APO + 0.40)
    assert win._trace_view.measure_target() == (doc_a, 0)


def test_the_action_is_dead_with_no_selection_and_explains_itself(win, tmp_path):
    _open(win, tmp_path, t0_s=T0_APO)
    win._sync_create_segment_action()
    assert not win._act_psd_selection.isEnabled()
    win._open_psd_for_selection()      # a direct call must still refuse, out loud
    assert "Measure tool" in win.statusBar().currentMessage()
    assert win._psd_window is None


def test_the_selection_crosses_the_boundary_as_indices_not_seconds():
    """THE COROLLARY, pinned structurally: ``measure_bounds`` returns int sample
    indices, which are on neither clock. A version of this API that returned seconds
    would re-open the defect no matter how well the labels read."""
    import inspect

    from nanoscope.gui.trace_view import TraceView

    src = inspect.getsource(TraceView.measure_bounds)
    assert "tuple[int, int]" in src
    assert "indices" in src.lower()


def test_every_module_cross_reference_in_the_clock_surfaces_resolves():
    """A ``:mod:`` reference that names a module which does not exist is a dead
    pointer, and these four files are the ones a reader lands in when they are trying
    to work out which clock a number is on.

    The clock module lives at ``nanoscope.core.clock`` — it is imported by
    ``gui/trace_view.py``, ``gui/panels.py`` and ``gui/main_window.py`` as
    ``from ..core import clock`` — but eight comments and docstrings across those
    three files pointed at ``nanoscope.gui.clock``, which has never existed. The one
    module whose entire purpose is to be the app's single answer to "which clock?"
    was unreachable from every surface that cites it.

    Checked by IMPORTING each referenced module rather than by grepping for the wrong
    name, so a future reference to any other non-existent module fails here too.
    """
    import importlib
    import re

    import nanoscope.gui.main_window as mw
    import nanoscope.gui.panels as pn
    import nanoscope.gui.trace_view as tv
    import nanoscope.gui.baseline_panel as bp

    pattern = re.compile(r"(?::mod:`|``)(nanoscope(?:\.[a-z_]+)+)(?:`)")
    bad: list[str] = []
    for module in (mw, pn, tv, bp):
        text = pathlib.Path(module.__file__).read_text()
        for name in sorted(set(pattern.findall(text))):
            try:
                importlib.import_module(name)
            except ImportError:
                bad.append(f"{pathlib.Path(module.__file__).name} -> {name}")
    assert not bad, f"dead module cross-references: {bad}"


# ================================================================================================
# 6. THE THIRD SURFACE — the PSD window's Start/End spins
# ================================================================================================
# The finding names three surfaces. Two of them (the trace axis, the event table) were
# labelled in the first pass and pinned above; this one was not, and it is the one where
# the wrong number is TYPED rather than merely read. `psd_panel.py` contained zero
# occurrences of "clock", "instrument" or "t₀": the spins carried `setSuffix(" s")` and
# the tooltip "Start of the analysed segment, in seconds." The menu hand-off tested above
# routes around them with sample indices; a user typing is not routed around anything.
#
# Measured live on the −500 mV recording (t₀ = 10.2526 s): the segment read-out for a
# recording-clock window of 1.0–1.5 s now also states 11.2526–11.7526 s, which is the pair
# of numbers that was never on screen together.

def _psd_for(win, tmp_path, *, t0_s, name):
    doc_id, _ = _open(win, tmp_path, t0_s=t0_s, name=name)
    from nanoscope.gui.psd_panel import PSDWindow
    psd = PSDWindow(win._session)
    psd.set_target(doc_id, 0)
    return psd


def test_the_psd_segment_spins_say_which_clock_they_are_in(win, tmp_path):
    """Both spins, both halves: the general rule and this recording's own t₀."""
    psd = _psd_for(win, tmp_path, t0_s=T0_APO, name="psdclock.dat")
    try:
        for spin in (psd._start_spin, psd._end_spin):
            tip = spin.toolTip()
            assert "FROM THE START OF THIS RECORDING" in tip, tip
            # ...and the arithmetic, in numbers, not as an abstraction.
            assert "61.3034" in tip, tip
            assert "trace x-axis" in tip, tip
    finally:
        psd.deleteLater()


def test_the_psd_segment_readout_states_the_window_on_both_clocks(win, tmp_path):
    """The read-out under the spins is where a typed number is checked.

    0.20–0.40 s into the recording is 61.5034–61.7034 s on the trace. Before, the
    read-out said only ``20,000 samples · 0.4 s`` — a length, which is the one thing
    that is the same on both clocks and therefore cannot catch this mistake.
    """
    psd = _psd_for(win, tmp_path, t0_s=T0_APO, name="psdclock2.dat")
    try:
        psd.set_segment(0.20, 0.40)
        text = psd._seg_label.text()
        assert "into the recording" in text, text
        assert "61.5034" in text and "61.7034" in text, text
    finally:
        psd.deleteLater()


def test_a_recording_whose_clocks_agree_says_so_rather_than_printing_it_twice(win, tmp_path):
    """t₀ = 0 (the −700 mV file) is where this confusion is INVISIBLE, so the label
    still names its clock — it just also says the two read alike. The read-out does
    not double-print a number that is the same twice."""
    psd = _psd_for(win, tmp_path, t0_s=0.0, name="psdclock3.dat")
    try:
        psd.set_segment(0.20, 0.40)
        tip = psd._start_spin.toolTip()
        assert "FROM THE START OF THIS RECORDING" in tip, tip
        assert "t₀ is 0" in tip, tip
        assert "instrument clock" not in psd._seg_label.text(), psd._seg_label.text()
    finally:
        psd.deleteLater()


def test_the_psd_panel_names_a_clock_at_all():
    """The structural half. The file had ZERO occurrences of any of these three words,
    which is how a three-surface finding came to be closed on two surfaces."""
    import inspect

    from nanoscope.gui import psd_panel

    src = inspect.getsource(psd_panel)
    for word in ("clock", "instrument", "t₀"):
        assert word in src, f"psd_panel.py never mentions {word!r}"
