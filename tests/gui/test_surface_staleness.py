"""C-1 — THE STANDALONE WINDOWS KEPT DRAWING THE PRE-CURATION FIT.

Measured on the owner's ``500 pM Apo ferritin…_0006_Process_-600 mV.abf`` (ch 2, 317
events) with Analysis ▸ Plots open on a ``dwell_s`` histogram, "use accepted events
only" ticked and an exponential fit:

===========================  ==========  =====
state                        tau (dwell)      n
===========================  ==========  =====
before curation               0.01429 s    316
after rejecting 159 events    0.01429 s    316
after clicking Refresh       0.008773 s    158
===========================  ==========  =====

**tau overstated by 63%, n wrong by 100%**, with no staleness marking in the legend, the
title or the Details pane — and the exported PNG byte-identical to the pre-curation
figure. The data was never stale: ``core.sources.discover_plot_sources`` passes
``res.payload.df`` BY REFERENCE, so curation writes the very frame the window holds. Only
the drawn scene and the cached fit were.

THE DEFECT IS THE WIRING SET, NOT THE WINDOW. ``_refresh_surface_windows`` was connected
to four signals — ``resultsChanged``, ``documentClosed``, ``measurementsChanged`` and the
measurement stack's ``changed`` — while six others changed what those windows draw:
``eventAcceptanceChanged``, ``eventFlagChanged``, ``curationBatchChanged``,
``eventEdited``, ``labelsChanged``, and (for PSD / I-V / V-dep) ``calibrationChanged`` and
``documentOpened``. ``labelsChanged`` is the proof that a hand-maintained connection list
cannot hold: it arrived with ``analysis/labels.py`` this week and the wiring did not
follow.

So the fix is a declared set (``main_window.SURFACE_MUTATION_SIGNALS``) that the wiring is
DERIVED from, plus its complement (``SURFACE_NEUTRAL_SIGNALS``) carrying a written reason
per signal. The guard tests below fail the day any signal declared on any of those classes
is in neither table — a seventh cannot be added silently, it has to be classified first.

The nastiest property of the original bug, and the reason it survived two hunts: any
unrelated nudge — changing bins, toggling log-x, re-ticking a source — calls ``_redraw()``
and silently repairs the number, so the wrong figure appeared and disappeared with
incidental clicking.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow


def _tables() -> tuple[dict, dict]:
    """The two classification tables, looked up LATE.

    Deliberately not a module-level import: against a reverted (pre-fix) package the whole
    file would fail to collect, and a collection error proves less than the behavioural
    tests below failing on their own assertions.
    """
    return (getattr(mw, "SURFACE_MUTATION_SIGNALS", {}),
            getattr(mw, "SURFACE_NEUTRAL_SIGNALS", {}))

FS = 100_000.0
#: Thirty events with a deliberate SPREAD of dwell times. Thirty because the Populations
#: window refuses to fit fewer than twenty ("a distribution fit to a handful of events is
#: decoration, not a measurement") and this test is about the FIT, not only the count —
#: the report's headline was tau overstated 63%, with n merely the visible half of it.
_SPANS = [(10_000 * (i + 1), 10_000 * (i + 1) + int(100 * 1.12 ** i)) for i in range(30)]
_N = 330_000


def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    for s, e in _SPANS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str = "pore_p1.dat") -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels if c.kind is ChannelKind.CURRENT)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot, *, min_dwell_s: float = 5e-4) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    return window, doc_id, channel


def _open_populations(window, qtbot):
    """Analysis ▸ Plots, dwell histogram, accepted-only, exponential fit — the exact
    configuration the report measured on."""
    window._act_plots.trigger()
    pw = window._plots_window
    pw.set_plot_type("Histogram")
    pw.set_histogram_column("dwell_s")
    pw._accepted_check.setChecked(True)
    pw.set_fit("exponential")
    qtbot.wait(10)
    return pw


_N_RE = re.compile(r"n\s*=\s*(\d+)")
_TAU_RE = re.compile(r"(?:tau|τ)\s*=?\s*([0-9.eE+-]+)")


def _drawn_n(pw) -> int:
    hit = _N_RE.search(pw.details_text())
    assert hit, f"the Details pane no longer states n:\n{pw.details_text()}"
    return int(hit.group(1))


def _drawn_tau(pw) -> float | None:
    hit = _TAU_RE.search(pw.details_text())
    return float(hit.group(1)) if hit else None


# ================================================================================================
# the measured defect
# ================================================================================================

def test_rejecting_events_moves_the_populations_fit_with_no_refresh_click(detected, qtbot):
    """THE headline. Reject the long events through the table — the commonest edit in the
    app — and the drawn n AND the fitted tau must follow, with nothing else clicked.

    Before the fix both were frozen at their pre-curation values and only the window's own
    Refresh button repaired them.
    """
    window, doc_id, channel = detected
    pw = _open_populations(window, qtbot)

    n_before = _drawn_n(pw)
    tau_before = _drawn_tau(pw)
    assert n_before >= 8, "need enough events for the curation to be a real edit"
    assert tau_before is not None, (
        "the exponential fit must state tau in the Details pane for this test to mean "
        f"anything:\n{pw.details_text()}")

    # Reject the six LONGEST events through the panel (the real curation path).
    df = window._analysis_panel.current_detection().payload.df
    longest = df.sort_values("dwell_s", ascending=False)["event_id"].head(6)
    for ev in longest:
        window._analysis_panel.set_event_accepted(int(ev), False)
    qtbot.wait(10)

    n_after = _drawn_n(pw)
    tau_after = _drawn_tau(pw)
    assert n_after == n_before - 6, (
        f"the Populations window is still drawing n = {n_after} after six rejections "
        f"(was {n_before}); it is showing the pre-curation figure")
    assert tau_after is not None and tau_after < tau_before, (
        f"tau did not follow the curation: {tau_before} -> {tau_after}. Rejecting the "
        "longest events must shorten the fitted dwell constant")

    # ...and the window's own Refresh now finds nothing left to repair. That is the whole
    # claim: the drawn scene already IS the current one.
    window._refresh_plots()
    qtbot.wait(5)
    assert _drawn_n(pw) == n_after
    assert _drawn_tau(pw) == pytest.approx(tau_after, rel=1e-12)


def test_the_event_plots_window_follows_curation_too(detected, qtbot):
    """The density / ensemble window is explicitly accepted-only and rode the same blind
    refresher. Affected set = {Populations, Event Plots}."""
    window, doc_id, channel = detected
    window._act_event_plots.trigger()
    epw = window._event_plot_window
    assert epw is not None

    seen: list[int] = []
    real = epw.set_source
    epw.set_source = lambda d, c, es, _r=real, _s=seen: (      # noqa: E731
        _s.append(0 if es is None else len(es.df)), _r(d, c, es))

    df = window._analysis_panel.current_detection().payload.df
    window._analysis_panel.set_event_accepted(int(df["event_id"].iloc[0]), False)
    qtbot.wait(10)
    assert seen, "a single accept/reject never reached the Event Plots window"


def test_a_mask_edit_and_a_batch_curation_and_a_label_all_reach_the_refresher(detected, qtbot):
    """``eventEdited``, ``curationBatchChanged`` and ``labelsChanged`` — the three the
    Inspector owns. All three change the shared frame; none of them reached the refresher."""
    window, doc_id, channel = detected
    window._open_event_inspector()
    insp = window._event_inspector

    hits: list[str] = []
    real = window._refresh_surface_windows
    window._refresh_surface_windows = lambda _r=real, _h=hits: (_h.append("x"), _r())[1]

    for signal in (insp.eventEdited, insp.curationBatchChanged, insp.labelsChanged):
        before = len(hits)
        signal.emit(*([0] if signal is insp.eventEdited else []))
        qtbot.wait(5)
        assert len(hits) > before, f"{signal} does not reach _refresh_surface_windows"


def test_a_label_written_in_the_gallery_reaches_the_open_inspector(detected, qtbot):
    """Two views of one frame, and the label a curator writes in one must appear in the
    other. Nothing re-fed the Inspector when the gallery wrote a label."""
    window, doc_id, channel = detected
    window._label_vocab.add("molecule", "apo-ferritin", "1")
    window._open_event_inspector()
    window._open_event_gallery()
    gal, insp = window._event_gallery, window._event_inspector

    df = window._analysis_panel.current_detection().payload.df
    ev = int(df["event_id"].iloc[0])
    insp.show_event(ev)
    assert "apo-ferritin" not in insp._readout.text()

    row = gal._row_for_id(ev)
    idx = gal._model.index_for_df_row(row)
    gal.view.selectionModel().select(idx, QtCore.QItemSelectionModel.ClearAndSelect)
    assert gal.apply_label("molecule", "apo-ferritin") == 1
    qtbot.wait(10)

    assert "apo-ferritin" in insp._readout.text(), (
        "a label written in the gallery never reached the open Event Inspector")


def test_the_refresher_does_not_destroy_the_gallerys_multi_tile_selection(detected, qtbot):
    """The cost of wiring more signals to one refresher, paid once, here.

    Re-feeding the gallery the SAME frame is a re-read, not an install: ``set_events``
    rebuilds the model and puts the cursor back on tile 0, which would throw away the
    multi-tile selection a curator labels with. Five tiles selected, an unrelated refresh,
    five tiles still selected.
    """
    window, doc_id, channel = detected
    window._open_event_gallery()
    gal = window._event_gallery
    sel = QtCore.QItemSelection(gal._model.index(0, 0), gal._model.index(4, 0))
    gal.view.selectionModel().select(sel, QtCore.QItemSelectionModel.ClearAndSelect)
    assert len(gal._selected_rows()) == 5

    window._refresh_surface_windows()
    qtbot.wait(5)
    assert len(gal._selected_rows()) == 5, (
        "an unrelated surface refresh cleared the gallery's selection")


# ================================================================================================
# the guard — a seventh signal cannot be added silently
# ================================================================================================

#: role -> the class whose declared signals that role covers.
_ROLE_CLASSES = {
    "session": "nanoscope.core.session:Session",
    "analysis_panel": "nanoscope.gui.analysis_panel:AnalysisPanel",
    "measure_panel": "nanoscope.gui.measure_panel:MeasurePanel",
    "measure_history": "nanoscope.core.history:MeasurementHistory",
    "event_inspector": "nanoscope.gui.event_inspector:EventInspectorWindow",
    "event_gallery": "nanoscope.gui.event_gallery:EventGalleryWindow",
}


def _resolve(spec: str):
    import importlib

    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def _declared_signals(cls) -> list[str]:
    """Signals this class declares ITSELF (Qt's own — destroyed, objectNameChanged,
    windowTitleChanged … — are inherited and are nobody's classification problem)."""
    return sorted(k for k, v in vars(cls).items()
                  if type(v).__name__ in ("Signal", "SignalInstance"))


def _unclassified(role: str, cls) -> list[str]:
    mutating, neutral = _tables()
    wired = set(mutating.get(role, ()))
    return [n for n in _declared_signals(cls)
            if n not in wired and f"{role}.{n}" not in neutral]


def test_every_signal_these_classes_declare_is_classified():
    """THE GUARD. Every signal on every object the refresher listens to is either wired
    or carries a written reason for not being.

    This is what ``labelsChanged`` slipped past. Adding a signal to any of these classes
    now fails here until somebody decides, in writing, whether it changes what a surface
    window draws.
    """
    mutating, _neutral = _tables()
    assert mutating, (
        "nanoscope.gui.main_window declares no SURFACE_MUTATION_SIGNALS: the set of "
        "signals that change what a surface window draws is not stated anywhere")
    missing = {role: _unclassified(role, _resolve(spec))
               for role, spec in _ROLE_CLASSES.items()}
    missing = {r: n for r, n in missing.items() if n}
    assert not missing, (
        "these signals are in NEITHER SURFACE_MUTATION_SIGNALS nor "
        f"SURFACE_NEUTRAL_SIGNALS (nanoscope/gui/main_window.py): {missing}. Decide "
        "whether each one changes what a surface window draws, and say so there.")


def test_the_guard_would_actually_catch_a_new_signal():
    """A test that passes both before and after pins nothing. This one pins the GUARD:
    an unclassified signal on one of these classes must be reported."""
    from PySide6.QtCore import QObject, Signal

    class _Grown(QObject):
        curationBatchChanged = Signal()      # already classified for this role
        somethingNewChanged = Signal()       # ...and this one is not

    assert _unclassified("event_gallery", _Grown) == ["somethingNewChanged"]


def test_every_neutral_claim_names_a_signal_that_exists():
    """The complement of the guard: a reason left behind after a signal is renamed or
    removed would silently re-open the hole it is standing in for."""
    _mutating, neutral = _tables()
    assert neutral, "no signal has been classified as leaving the surfaces alone"
    stale = []
    for qualified in neutral:
        role, _, name = qualified.partition(".")
        assert role in _ROLE_CLASSES, f"{qualified}: unknown role"
        if name not in _declared_signals(_resolve(_ROLE_CLASSES[role])):
            stale.append(qualified)
    assert not stale, f"SURFACE_NEUTRAL_SIGNALS names signals that no longer exist: {stale}"


def test_no_signal_is_in_both_tables():
    mutating, neutral = _tables()
    assert mutating and neutral, "both classification tables must exist to be checked"
    both = [f"{role}.{n}" for role, names in mutating.items()
            for n in names if f"{role}.{n}" in neutral]
    assert not both, f"classified twice, and contradictorily: {both}"


def test_the_registry_is_what_actually_wires_the_refresher(detected, qtbot):
    """The table is load-bearing, not documentation: EVERY name in it, on a live window,
    reaches ``_refresh_surface_windows``.

    Both lazily-built windows are opened first so their roles are wired.
    """
    window, doc_id, channel = detected
    window._open_event_inspector()
    window._open_event_gallery()
    sources = {
        "session": window._session,
        "analysis_panel": window._analysis_panel,
        "measure_panel": window._analysis_panel.measure_panel,
        "measure_history": window._measure_history,
        "event_inspector": window._event_inspector,
        "event_gallery": window._event_gallery,
    }
    mutating, _neutral = _tables()
    assert set(sources) == set(mutating), (
        "a role was added to SURFACE_MUTATION_SIGNALS without an object to wire it to")

    hits: list[str] = []
    real = window._refresh_surface_windows
    window._refresh_surface_windows = lambda _r=real, _h=hits: (_h.append("x"), _r())[1]

    # Signal signatures differ; emit each with the right arity, read off the object.
    args = {
        "resultsChanged": (doc_id, "detect.schmitt"), "documentOpened": (doc_id,),
        "documentClosed": (doc_id,), "calibrationChanged": (doc_id, channel),
        "sampleRateChanged": (doc_id,),
        "eventAcceptanceChanged": (0, True), "eventFlagChanged": (0, ""),
        "eventEdited": (0,), "amplitudeEdited": (0,),
    }
    for role, names in mutating.items():
        for name in names:
            before = len(hits)
            getattr(sources[role], name).emit(*args.get(name, ()))
            qtbot.wait(2)
            assert len(hits) > before, (
                f"SURFACE_MUTATION_SIGNALS[{role!r}] lists {name!r} but emitting it does "
                "not reach _refresh_surface_windows")


def test_wiring_an_unknown_role_or_a_missing_signal_is_a_loud_error(detected):
    """``_wire_surface_refresh`` never skips silently — skipping is the failure mode the
    whole registry replaces."""
    window, _doc_id, _channel = detected
    with pytest.raises(KeyError):
        window._wire_surface_refresh("no_such_role", window._session)
    with pytest.raises(AttributeError):
        window._wire_surface_refresh("event_inspector", window._session)


def test_the_module_still_states_the_set_in_one_place():
    """A reviewer must be able to read the whole answer in one object."""
    mutating, neutral = _tables()
    assert isinstance(mutating, dict) and mutating
    assert all(isinstance(v, tuple) and v for v in mutating.values())
    assert all(isinstance(v, str) and v.strip() for v in neutral.values()), (
        "every neutral claim must carry a reason a reviewer can disagree with")
    # ...and the six the report named are all in it.
    named = {"eventAcceptanceChanged", "curationBatchChanged", "eventFlagChanged",
             "labelsChanged", "eventEdited"}
    wired = {n for names in mutating.values() for n in names}
    assert named <= wired, f"still unwired: {sorted(named - wired)}"
    assert Path(mw.__file__).name == "main_window.py"
