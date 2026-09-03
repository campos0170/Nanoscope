"""E2E-2 — the Measure basis hears about a filter the moment the user ticks it.

THE DEFECT. ``MeasurePanel`` talks to Session and to its TraceView and to no sibling
panel, so the Preprocess form can only reach it through ``set_conditioning``. That door
had exactly one caller — ``MainWindow._sync_processed_export``, which re-asks the form
when the Analysis menu opens — so ticking *enable* on the Preprocess tab's low-pass and
walking straight over to the Measure tab left the panel believing no chain existed. It
fell back to ``_stored_detection_chain``, and on a channel with no stored detection the
*Processed signal* item stayed greyed and a measurement on it was refused with "no filter
stage is enabled on the Preprocess tab" — while the orange processed curve computed from
that very form was drawn on the plot.

The arithmetic was never the problem: on the owner's 15 s -1000 mV apo recording,
samples 100,000-140,000, raw vs processed reads amplitude 2.823 -> 2.517 pA and std
27.18 -> 27.02 pA. The panel was simply never told.

WHAT THESE ASSERT. The user-visible effect only — a selectable combo item, a sentence
that stops denying the filter, and a number that moves to the conditioned one — reached
by real gestures on the FORM (``filter_enabled.setChecked``, the cutoff spin), never by
calling ``set_conditioning`` by hand. And the two states where the panel genuinely cannot
condition are pinned beside them, so "it always says yes now" cannot pass either.

WHAT FAILS BEFORE THE FIX. Every assertion that a form gesture reaches the panel:
``measure_panel.conditioning_params()`` was ``None`` for the life of the process (proved
by re-running this module against a copy of the pre-fix ``analysis_panel.py`` through an
out-of-tree plugin — 5 failed, 2 passed, the 2 being the refusal-accuracy guards that
must hold on both sides).
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
from nanoscope.analysis.conditioning import (
    BASIS_PROCESSED,
    conditioned_trace,
    conditioning_enabled,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 120_000
CUTOFF = 5_000.0


# ================================================================================================
# builders
# ================================================================================================
def _signal(seed: int = 11) -> np.ndarray:
    """A -600 pA open pore with real noise and six blockades.

    Noisy on purpose: the whole point of measuring on the conditioned trace is that a
    low-pass moves the spread, and a clean trace would let a dead wiring pass by
    returning nearly the same numbers.
    """
    rng = np.random.default_rng(seed)
    x = -600.0 + rng.normal(0.0, 12.0, N)
    for start in (10_000, 30_000, 50_000, 70_000, 90_000, 110_000):
        x[start:start + 400] -= 90.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, sig: np.ndarray, name: str = "pore.dat",
         *, rate: float | None = FS) -> TraceDocument:
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(sig.size),
        sample_rate_hz=rate, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture(autouse=True)
def _private_settings(qapp):
    """The Preprocess form persists through QSettings; never touch the owner's profile."""
    app = QtWidgets.QApplication.instance()
    org, name = app.organizationName(), app.applicationName()
    app.setOrganizationName("NanoScopeTest")
    app.setApplicationName("FinE2E2BasisPush")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        app.setOrganizationName(org)
        app.setApplicationName(name)


@pytest.fixture
def build(qapp, qtbot, tmp_path):
    """A real AnalysisPanel (so the real DetectionPanel and the real MeasurePanel) on one
    synthetic current channel. Returns ``(panel, session, doc_id, samples)``."""
    built: list[AnalysisController] = []

    def _build(*, rate: float | None = FS, name: str = "pore.dat", seed: int = 11):
        sig = _signal(seed)
        session = Session()
        tv = TraceView(session)
        qtbot.addWidget(tv)
        controller = AnalysisController(session)
        built.append(controller)
        p = AnalysisPanel(session, controller, tv)
        qtbot.addWidget(p)
        doc_id = session.add_document(_doc(tmp_path, sig, name, rate=rate))
        session.add_layer(doc_id, 0)
        p.set_selection(doc_id, 0)
        return p, session, doc_id, sig

    yield _build
    for c in built:
        c.abort()


def _tick_lowpass(p: AnalysisPanel, cutoff: float = CUTOFF) -> None:
    """The user's own gesture on the Preprocess tab: set a cutoff, tick *enable*."""
    form = p._detection
    form.filter_cutoff_hz.setValue(float(cutoff))
    form.filter_enabled.setChecked(True)


def _processed_item_enabled(p: AnalysisPanel) -> bool:
    """Is *Processed signal* selectable? — what the user sees when the combo drops open."""
    mp = p.measure_panel
    model = mp._basis_combo.model()
    return model.item(mp._basis_combo.findData(BASIS_PROCESSED)).isEnabled()


# ================================================================================================
# the push — a form gesture reaches the Measure panel with no menu in between
# ================================================================================================
def test_ticking_the_lowpass_makes_processed_selectable_with_no_menu_opened(build):
    """Before: greyed forever unless a menu happened to re-ask the form. After: live."""
    p, _session, _doc_id, _sig = build()

    assert not _processed_item_enabled(p), "nothing is enabled on the form yet"
    assert not conditioning_enabled(p.measure_panel.conditioning_params())

    _tick_lowpass(p)

    assert _processed_item_enabled(p)
    assert p.measure_panel.conditioning_params()["filter_cutoff_hz"] == CUTOFF
    assert p.measure_panel.processed_basis_available()


def test_the_basis_line_stops_denying_a_filter_that_is_ticked(build):
    """The sentence under the readout is the surface that lied; it names the chain now."""
    p, _session, _doc_id, _sig = build()
    p.measure_panel.set_measurement_basis(BASIS_PROCESSED)
    assert "no filter stage is enabled" in p.measure_panel.basis_note(), (
        "with nothing ticked the denial is TRUE — that is the state we start from")

    _tick_lowpass(p)
    note = p.measure_panel.basis_note()

    assert "no filter stage is enabled" not in note
    assert "low-pass" in note and "5000 Hz" in note
    # ...and it does NOT claim a stored detection: this is the LIVE form.
    assert "stored detection" not in note


def test_measuring_processed_after_the_tick_gives_the_conditioned_numbers(build):
    """The number moves, and moves to exactly what the pure kernel gives."""
    p, _session, doc_id, sig = build()
    mp = p.measure_panel
    mp.set_active(True)
    assert mp._place_cursors(doc_id, 0, 20_000, 60_000) is not None
    raw = mp.exact_measurement()

    _tick_lowpass(p)
    mp.set_measurement_basis(BASIS_PROCESSED)
    proc = mp.exact_measurement()

    assert raw is not None and proc is not None
    assert proc["n_samples"] == raw["n_samples"] == 40_000
    assert proc["std"] < raw["std"] * 0.9, "a 5 kHz low-pass on 50 kHz noise cuts the spread"
    expect = conditioned_trace(
        np.asarray(sig, dtype=np.float64), FS,
        {"filter_cutoff_hz": CUTOFF, "filter_kind": p._detection.params()["filter_kind"],
         "filter_order": p._detection.params()["filter_order"]},
    )[20_000:60_000]
    assert proc["std"] == pytest.approx(float(np.std(expect)), rel=1e-6)
    assert proc["mean"] == pytest.approx(float(np.mean(expect)), rel=1e-6)


def test_unticking_the_filter_takes_the_processed_basis_away_again(build):
    """The push is a live seam in both directions, not a one-way latch."""
    p, _session, _doc_id, _sig = build()
    _tick_lowpass(p)
    assert _processed_item_enabled(p)

    p._detection.filter_enabled.setChecked(False)

    assert not _processed_item_enabled(p)
    assert not conditioning_enabled(p.measure_panel.conditioning_params())
    assert p.measure_panel.conditioning_params()["filter_cutoff_hz"] is None


def test_selecting_a_channel_carries_a_form_that_was_configured_earlier(build):
    """A restored Preprocess form emits no ``paramsChanged``; the selection carries it.

    ``set_conditioning(None)`` here stands in for that never-pushed startup state — the
    panel is put back where a QSettings-restored form leaves it, and then the user does
    the one thing they always do next: pick a channel."""
    p, _session, doc_id, _sig = build()
    _tick_lowpass(p)
    p.measure_panel.set_conditioning(None)
    assert not _processed_item_enabled(p)

    p.set_selection(doc_id, 0)

    assert _processed_item_enabled(p)
    assert p.measure_panel.conditioning_params()["filter_cutoff_hz"] == CUTOFF


# ================================================================================================
# the refusals that must STAY — the two states where it genuinely cannot condition
# ================================================================================================
def test_with_nothing_ticked_the_refusal_still_names_the_preprocess_tab(build):
    """Accurate, because now it is true: no stage is on and no stored run had one."""
    p, _session, doc_id, _sig = build()
    mp = p.measure_panel
    mp.set_active(True)
    assert mp._place_cursors(doc_id, 0, 20_000, 60_000) is not None
    mp.set_measurement_basis(BASIS_PROCESSED)

    assert not mp.processed_basis_available()
    assert mp.exact_measurement() is None, "it refuses rather than measuring raw silently"
    note = mp.basis_note()
    assert "no filter stage is enabled on the Preprocess tab" in note
    assert "no sample rate" not in note


def test_with_a_filter_ticked_but_no_sample_rate_the_refusal_names_the_rate(build):
    """The other genuine state: the chain exists, the recording cannot carry it."""
    p, _session, doc_id, _sig = build(rate=None, name="norate.dat")
    _tick_lowpass(p)
    mp = p.measure_panel
    mp.set_active(True)
    mp._place_cursors(doc_id, 0, 20_000, 60_000)
    mp.set_measurement_basis(BASIS_PROCESSED)

    assert not mp.processed_basis_available()
    assert mp.exact_measurement() is None
    note = mp.basis_note()
    assert "no sample rate" in note
    assert "no filter stage is enabled" not in note
