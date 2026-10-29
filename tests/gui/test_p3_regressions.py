"""P3 — the ML-preset pair, the seeding invariant, and the resource/staleness tier.

Every test here reproduces a failure that was MEASURED on the tree as found. The three
groups are the three roots the batch fixed at:

**The seeding invariant.** "The settings on the tab are the settings that produced the
table on screen, and a value the user chose is never silently overwritten." Three defects
were three different violations of it — a model load rewriting seven hand-dialled
parameter groups (C25), a re-emitted selection reverting the form (C24), and a restored ML
result never re-seeding its own tab (C26).

**One action, one meaning.** Unticking a channel destroyed persisted view state while every
other path merely hid it (C15); closing a recording marked the session dirty while opening
one deliberately did not (C29).

**Nothing outlives the data it describes.** A closed recording left its rows in the Plots
window (C22); a rebuilt file dock left its widgets alive in the viewport (C39).

Plus the carried latents: the flag cell is a set and the curation keys are reachable; the
hand-measurement table is undoable; a save that drops work does not report success; and a
malformed manifest raises the one exception the API documents.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

from nanoscope.analysis.detect import detect_events  # noqa: E402
from nanoscope.analysis.events import (  # noqa: E402
    FLAG_CLIPPED,
    FLAG_CURATED,
    DERIVED_FLAGS,
    EventSet,
    flag_tokens,
)
from nanoscope.core.analysis import AnalysisResult  # noqa: E402
from nanoscope.core.document import TraceDocument  # noqa: E402
from nanoscope.core.project import ProjectError, load_project  # noqa: E402
from nanoscope.core.types import (  # noqa: E402
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.analysis_panel import _MODULE_ID, _ML_MODULE_ID  # noqa: E402
from nanoscope.gui.main_window import MainWindow  # noqa: E402

FS = 50_000.0
N = 20_000
_COL_NAME = 0


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0, dtype=np.float32) + rng.normal(0.0, 1.5, N).astype(np.float32)
    for k in range(5):
        s = 2_000 + k * 3_000
        x[s:s + 300] -= 40.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray, *, with_voltage: bool = False) -> TraceDocument:
    chans = [ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
    data = [arr]
    if with_voltage:
        chans.append(ChannelInfo(index=1, name="voltage", raw_units="mV",
                                 kind=ChannelKind.VOLTAGE))
        data.append(np.full(arr.size, -600.0, dtype=np.float32))
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=tuple(chans))
    return TraceDocument(meta=meta, channels=tuple(data))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _install(w: MainWindow, d: TraceDocument) -> tuple[str, int]:
    doc_id = w._session.add_document(d)
    w._create_default_layers(d)
    w._on_selection(doc_id, 0)
    return doc_id, 0


def _detect_into(w: MainWindow, doc_id: str, ch: int) -> AnalysisResult:
    """A stored Schmitt run whose params are the ones it ACTUALLY ran under (k_in 5.0),
    not whatever the live form happens to read — otherwise "did the seed run?" is
    unanswerable."""
    cc = w._session.calibrated(doc_id, ch)
    df, prov = detect_events(cc.data, cc.sample_rate_hz, k_in=5.0, k_out=1.0,
                             min_dwell_s=0.001)
    params = dict(w._analysis_panel._detection.params())
    params.update(k_in=5.0, k_out=1.0, min_dwell_s=0.001)
    res = AnalysisResult(
        module_id=_MODULE_ID, module_version="1", doc_id=doc_id, channel=ch,
        params=params,
        calibration_fingerprint=w._session.calibration(doc_id, ch).fingerprint(),
        payload=EventSet(df, prov))
    w._session.add_result(res)
    return res


def _channel_item(w: MainWindow, doc_id: str, ch: int):
    it = QtWidgets.QTreeWidgetItemIterator(w._file_dock._tree)
    while it.value():
        item = it.value()
        payload = item.data(_COL_NAME, QtCore.Qt.UserRole)
        if payload and payload[0] == "channel" and payload[1] == doc_id and payload[2] == ch:
            return item
        it += 1
    raise AssertionError(f"no row for channel {ch}")


# ================================================================================================
# C25 — a loaded model must not overrule a hand-dialled pipeline
# ================================================================================================

def test_c25_a_model_width_does_not_overwrite_hand_dialled_ml_parameters(win):
    """Measured before the fix: ``enter`` hand-set to 0.35 with the status reading CUSTOM,
    then a 2048-wide model loaded -> SEVEN parameter groups rewritten, ``enter`` back to
    0.5 and the status claiming "matches Ferritin — nb1". The latch that was supposed to
    stop this existed and was documented; nothing ever set it.

    The molecule makes it worse than a lost preference: the owner's practice recording is
    APO-ferritin and the ``notebook*`` presets are FERRITIN, and ``preset_for_window``'s
    own docstring says 2048 does not settle the molecule."""
    p = win._analysis_panel._detection
    p.ml_enter.setValue(0.35)
    p.ml_min_event_ms.setValue(9.0)
    before = (p.ml_preset_name(), p.ml_enter.value(), p.ml_exit.value(),
              p.ml_min_event_ms.value(), p.ml_preset_status.text())
    assert "CUSTOM" in before[4]
    assert p._ml_preset_chosen is True, "a hand-dialled value IS a choice"

    p.set_model_input_width(2048)

    after = (p.ml_preset_name(), p.ml_enter.value(), p.ml_exit.value(),
             p.ml_min_event_ms.value(), p.ml_preset_status.text())
    assert after == before


def test_c25_an_untouched_form_still_takes_the_width_hint_and_says_so(win):
    """The heuristic is not removed — it is only stopped from overruling a person. When it
    does fire it names both presets, because seven groups moving silently is the half of
    the defect that had nothing to do with the latch."""
    p = win._analysis_panel._detection
    assert p._ml_preset_chosen is False
    was = p.ml_preset_name()

    p.set_model_input_width(2048)

    assert p.ml_preset_name() != was
    note = p.ml_pipeline_note.text()
    assert "changed from" in note and "2048" in note


def test_c25_a_kept_choice_that_contradicts_the_model_s_molecule_is_named(win):
    """The state the latch CREATES, and the one it must not create silently.

    Refusing to overrule a choice makes "a ferritin preset in force, an apo-width model
    loaded" a state that can now persist — before the fix the width simply overwrote the
    preset, wrongly but visibly. Measured on the tree as found: preset ``notebook1``
    (FERRITIN), a 3072 model loaded, and the note said only that both apo pipelines are
    3072 — nothing about the pipeline actually about to run.

    ``preset_for_window``'s docstring is the authority for WHEN a width is evidence:
    3072 is both apo lineages and nothing else, so it settles the molecule."""
    p = win._analysis_panel._detection
    p.apply_ml_preset("notebook1", user_choice=True)       # FERRITIN, deliberately
    assert p.ml_preset_name() == "notebook1"

    p.set_model_input_width(3072)                          # an APO-only width

    assert p.ml_preset_name() == "notebook1", "the choice is still not overruled"
    note = p.ml_pipeline_note.text()
    assert "DISAGREE ABOUT THE MOLECULE" in note
    assert "APO-FERRITIN" in note and "FERRITIN" in note
    assert "Ferritin — nb1" in note                        # the pipeline about to run


def test_c25_a_width_that_does_not_settle_the_molecule_claims_nothing(win):
    """The mirror, and the reason the rule is a named set and not ``!=``.

    At 2048 there are the three ferritin notebooks AND an apo student, so an apo preset
    with a 2048 model loaded is not a contradiction — asserting one would be inventing
    evidence, which is the failure mode this whole tier is about."""
    p = win._analysis_panel._detection
    p.apply_ml_preset("apo_nb_600mV", user_choice=True)    # APO, deliberately

    p.set_model_input_width(2048)                          # a width shared by both

    assert p.ml_preset_name() == "apo_nb_600mV"
    assert "DISAGREE ABOUT THE MOLECULE" not in p.ml_pipeline_note.text()


def test_c25_the_width_that_was_allowed_to_set_the_preset_never_contradicts_it(win):
    """An auto-applied guess cannot disagree with itself. Guards the ordering: the
    disagreement is read AFTER the auto-apply, never before."""
    p = win._analysis_panel._detection
    assert p._ml_preset_chosen is False

    p.set_model_input_width(3072)

    assert "DISAGREE ABOUT THE MOLECULE" not in p.ml_pipeline_note.text()


# ================================================================================================
# C24 — re-pointing at the target already shown is navigation, and costs nothing
# ================================================================================================

def test_c24_reselecting_the_same_recording_does_not_revert_the_form(win, tmp_path):
    """Measured: edit ``k_in`` 5.0 -> 7.0, then click the VOLTAGE row of the SAME recording
    (which resolves back to this same current channel, deliberately) — ``k_in`` was 5.0
    again and the whole preprocessing chain had been re-applied, with no prompt."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace(), with_voltage=True))
    _detect_into(win, doc_id, ch)
    win._on_selection(doc_id, ch)          # seed the form from the stored run

    det = win._analysis_panel._detection
    det.k_in.setValue(7.0)
    det.ml_enter.setValue(0.42)

    win._on_selection(doc_id, 1)           # the voltage row of the same recording

    assert win._analysis_panel._sel == (doc_id, ch)
    assert det.k_in.value() == 7.0
    assert det.ml_enter.value() == 0.42


def test_c24_a_genuinely_new_table_on_the_same_target_still_seeds_the_form(win, tmp_path):
    """The guard is "same TABLE", not "same target" — a project restore installs the saved
    run into a session already pointed here, and the form has to follow it or the user is
    shown app defaults over somebody else's numbers."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    det = win._analysis_panel._detection
    det.k_in.setValue(7.0)

    res = _detect_into(win, doc_id, ch)    # k_in 5.0 in its params
    assert res.params["k_in"] == 5.0
    win._on_selection(doc_id, ch)          # same target, a table it has not seeded from

    assert det.k_in.value() == 5.0


def test_c24_selecting_another_recording_still_seeds_from_that_one(win, tmp_path):
    """The guard must not disable seeding across a real target change."""
    id_a, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, id_a, ch)
    win._on_selection(id_a, ch)
    win._analysis_panel._detection.k_in.setValue(7.0)

    id_b, _ = _install(win, _doc(tmp_path / "b.dat", _trace(seed=3)))
    win._analysis_panel._detection.k_in.setValue(6.5)
    win._on_selection(id_a, ch)            # back to A, whose stored run says 5.0

    assert win._analysis_panel._detection.k_in.value() == 5.0


# ================================================================================================
# C26 — a restored ML result seeds the ML tab
# ================================================================================================

def _store_ml_result(win, doc_id, ch, params) -> AnalysisResult:
    from nanoscope.analysis.events import empty_df
    res = AnalysisResult(
        module_id=_ML_MODULE_ID, module_version="1", doc_id=doc_id, channel=ch,
        params=params,
        calibration_fingerprint=win._session.calibration(doc_id, ch).fingerprint(),
        payload=EventSet(empty_df()))
    win._session.results.put(res)
    return res


def test_c26_a_stored_ml_run_puts_its_own_pipeline_back(win, tmp_path):
    """Measured: stored ``notebook3`` / ``enter=0.9`` came back onto a panel reading
    ``notebook1`` / ``enter=0.5`` / ``smoothing=9`` — verbatim the failure
    ``_load_stored_detection`` exists to prevent, on the detector whose preprocessing chain
    IS its contract with the trained model."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    panel, det = win._analysis_panel, win._analysis_panel._detection

    det.apply_ml_preset("notebook3", user_choice=True)
    det.ml_enter.setValue(0.9)
    stored = det.ml_params()
    _store_ml_result(win, doc_id, ch, stored)

    det.apply_ml_preset("notebook1", user_choice=False)
    det.ml_enter.setValue(0.5)
    panel._sel = None
    panel._seeded_from = None
    panel.set_selection(doc_id, ch)

    assert det.ml_preset_name() == "notebook3"
    assert det.ml_enter.value() == pytest.approx(0.9)
    assert det.ml_smoothing.value() == stored["ml_decode"]["smoothing_kernel"]


def test_c26_a_customised_ml_run_is_restored_as_customised_not_snapped_to_its_preset(
        win, tmp_path):
    """``ml_preprocess``/``ml_decode`` are the authority; the preset only names the baseline
    the deviations are reported against."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    panel, det = win._analysis_panel, win._analysis_panel._detection

    det.apply_ml_preset("notebook1", user_choice=True)
    det.ml_merge_ms.setValue(7.5)
    stored = det.ml_params()
    assert stored["ml_preset_customized"] is True
    _store_ml_result(win, doc_id, ch, stored)

    det.apply_ml_preset("notebook2", user_choice=False)
    panel._sel = None
    panel._seeded_from = None
    panel.set_selection(doc_id, ch)

    assert det.ml_preset_name() == "notebook1"
    assert det.ml_merge_ms.value() == pytest.approx(7.5)
    assert "CUSTOM" in det.ml_preset_status.text()


def test_c26_a_restored_ml_run_is_not_then_overruled_by_the_next_model_loaded(win, tmp_path):
    """The two halves meet: a run put back on screen is a deliberate configuration, so the
    width heuristic must not overwrite it either."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    panel, det = win._analysis_panel, win._analysis_panel._detection
    det.apply_ml_preset("notebook3", user_choice=True)
    _store_ml_result(win, doc_id, ch, det.ml_params())
    det.apply_ml_preset("apo_nb_600mV", user_choice=False)
    panel._sel = None
    panel._seeded_from = None
    panel.set_selection(doc_id, ch)
    assert det.ml_preset_name() == "notebook3"

    det.set_model_input_width(3072)        # an apo width

    assert det.ml_preset_name() == "notebook3"


def test_c26_a_stored_schmitt_run_still_leaves_the_ml_tab_alone(win, tmp_path):
    """Each detector seeds its OWN tab; neither writes over the other."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    det = win._analysis_panel._detection
    det.apply_ml_preset("notebook2", user_choice=True)
    det.ml_enter.setValue(0.77)

    _detect_into(win, doc_id, ch)
    win._analysis_panel._sel = None
    win._analysis_panel._seeded_from = None
    win._analysis_panel.set_selection(doc_id, ch)

    assert det.ml_preset_name() == "notebook2"
    assert det.ml_enter.value() == pytest.approx(0.77)


def test_c26_an_ml_result_with_no_stored_params_changes_nothing(win, tmp_path):
    """A v1/v2 project stored no params; a form showing values that at least came from the
    user beats one showing values invented here."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    panel, det = win._analysis_panel, win._analysis_panel._detection
    det.apply_ml_preset("notebook2", user_choice=True)
    det.ml_enter.setValue(0.77)
    _store_ml_result(win, doc_id, ch, {})

    panel._sel = None
    panel._seeded_from = None
    panel.set_selection(doc_id, ch)

    assert det.ml_preset_name() == "notebook2"
    assert det.ml_enter.value() == pytest.approx(0.77)


# ================================================================================================
# C15 / C29 — one action, one meaning
# ================================================================================================

def test_c29_closing_a_recording_without_a_project_does_not_arm_the_quit_guard(win, tmp_path):
    """``documentClosed`` is the mirror of ``documentOpened`` and gets the mirror rule: the
    set of open files is what a project stores as mere references."""
    _install(win, _doc(tmp_path / "a.dat", _trace()))
    id_b, _ = _install(win, _doc(tmp_path / "b.dat", _trace(seed=2)))
    win._mark_clean()

    win._session.close_document(id_b)

    assert win._dirty is False
    assert win.windowTitle() == "NanoScope"


# ================================================================================================
# C22 — no standalone window outlives the data it is showing
# ================================================================================================

def test_c22_closing_the_recording_empties_the_plots_window(win, tmp_path):
    """Measured: close the recording -> open documents ``()``, and the Plots window still
    holding its 5 rows, with a working Fit and a working Export CSV attributing numbers to
    a session that no longer contains the recording."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    win._open_plots()
    assert len(win._plots_window._sources) == 1

    win._session.close_document(doc_id)

    assert win._session.documents() == ()
    assert len(win._plots_window._sources) == 0


def test_c22_closing_an_unselected_recording_also_refreshes_the_plots_window(win, tmp_path):
    """The old partial refresh ran only when the closed document was the SELECTED one — and
    the Plots window is precisely the surface that reads across every open recording."""
    id_a, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, id_a, ch)
    id_b, _ = _install(win, _doc(tmp_path / "b.dat", _trace(seed=5)))
    _detect_into(win, id_b, 0)
    win._on_selection(id_b, 0)             # B is selected; A is not
    win._open_plots()
    assert len(win._plots_window._sources) == 2

    win._session.close_document(id_a)      # close the UNSELECTED one

    assert len(win._plots_window._sources) == 1


def test_c22_deleting_a_hand_measurement_empties_it_out_of_the_plots_window(win, tmp_path):
    """C22's rule on the OTHER table the Plots window reads.

    ``_plots_sources`` lists the manual-measurement table as a source in so many words,
    and ``_refresh_surface_windows`` was wired to ``resultsChanged`` and
    ``documentClosed`` only. Measured on the tree as found: three measurements, Plots
    open, delete one -> the session held 2 rows and the window still held 3, still
    carrying the deleted row's ``t_start_s``, with a live Fit and a live Export CSV."""
    doc_id, _ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _add_measurements(win, doc_id, [(2_000, 2_300), (5_000, 5_400), (8_000, 8_300)])
    mp = win._analysis_panel.measure_panel
    win._open_plots()
    source = lambda: next(s for s in win._plots_window._sources.values()
                          if s.label == "Manual measurements")
    assert len(source().df) == 3
    first_t = float(source().df.iloc[0]["t_start_s"])

    mp._confirm_remove = lambda _n: True
    mp._table_view.selectRow(0)
    mp.remove_selected()

    assert len(mp.measurements_df) == 2
    assert len(source().df) == 2
    assert float(source().df.iloc[0]["t_start_s"]) != first_t


def test_c22_undoing_that_delete_puts_the_row_back_in_the_window_too(win, tmp_path):
    """The undo does NOT go through ``measurementsChanged``: the stack installs its
    snapshot with ``set_measurements_df``, which is deliberately silent so a project
    restore does not look like an edit. So the stack's own ``changed`` is wired as well —
    a window left showing two rows after Ctrl+Z restored three is the same defect one
    keystroke later."""
    doc_id, _ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _add_measurements(win, doc_id, [(2_000, 2_300), (5_000, 5_400), (8_000, 8_300)])
    mp = win._analysis_panel.measure_panel
    mp._confirm_remove = lambda _n: True
    mp._table_view.selectRow(0)
    mp.remove_selected()
    win._open_plots()
    source = lambda: next(s for s in win._plots_window._sources.values()
                          if s.label == "Manual measurements")
    assert len(source().df) == 2

    win._undo_events()

    assert len(mp.measurements_df) == 3
    assert len(source().df) == 3


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the text reader reopens as one CURRENT channel at FS — so a
    restore genuinely reloads from disk rather than from anything this process holds."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _project_with_measurements(qtbot, csv: Path, arr: np.ndarray, proj: Path,
                               spans) -> None:
    """Save a project whose ONLY content is hand measurements — no detection."""
    from nanoscope.gui.main_window import MainWindow as _MW

    w = _MW()
    qtbot.addWidget(w)
    doc_id, _ch = _install(w, _doc(csv, arr))
    _add_measurements(w, doc_id, spans)
    assert w._write_project(str(proj)) is True


def test_c22_opening_another_project_re_feeds_the_plots_window(qapp, qtbot, tmp_path):
    """C22's other door, and the one hunt2 said would be masked: "Same hole on Open
    Project — masked whenever the incoming project carries a detection, so it appears
    exactly when the new project is clean."

    The per-signal wiring cannot cover it. ``_clear_session`` fires ``documentClosed`` at
    the START of the open, while the OUTGOING project's tables are still the live ones,
    and everything the incoming project installs afterwards either arrives through a
    deliberately silent door (``set_measurements_df``) or is not a document at all. So the
    restore re-feeds the windows itself, at the end, once.

    Measured on the tree as found, with two projects carrying only hand measurements:
    open A (3), open the Plots window, open B (1) -> the session held 1 row and the window
    still plotted A's 3, with a live Fit and a live Export CSV."""
    csv_a, csv_b = tmp_path / "a.csv", tmp_path / "b.csv"
    arr_a, arr_b = _trace(), _trace(seed=3)
    _write_csv(csv_a, arr_a)
    _write_csv(csv_b, arr_b)
    proj_a, proj_b = tmp_path / "A.nanoscope", tmp_path / "B.nanoscope"
    _project_with_measurements(qtbot, csv_a, arr_a, proj_a,
                               [(2_000, 2_300), (5_000, 5_400), (8_000, 8_300)])
    _project_with_measurements(qtbot, csv_b, arr_b, proj_b, [(1_000, 1_400)])

    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj_a))
    qtbot.wait(50)
    w._open_plots()
    source = lambda: next((s for s in w._plots_window._sources.values()
                           if s.label == "Manual measurements"), None)
    assert len(w._analysis_panel.measure_panel.measurements_df) == 3
    assert len(source().df) == 3

    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj_b))
    qtbot.wait(50)

    assert len(w._analysis_panel.measure_panel.measurements_df) == 1
    assert len(source().df) == 1


# ================================================================================================
# C39 — the file dock leaks no widgets
# ================================================================================================

def _viewport_widgets(win) -> tuple[int, int]:
    vp = win._file_dock._tree.viewport()
    return (len(vp.findChildren(QtWidgets.QFrame)),
            len(vp.findChildren(QtWidgets.QDoubleSpinBox)))


def test_c39_the_file_dock_leaves_no_widgets_behind(win, tmp_path, qapp):
    """``QTreeWidget.clear()`` deletes the items, not the widgets ``setItemWidget`` gave
    them: those are re-parented to the viewport and stay there. ``_rebuild`` runs on every
    documentOpened AND every documentClosed. Measured before the fix: +6 swatches and +6
    spin boxes per open/close cycle, and 18 of each still alive in a dock listing nothing.
    """
    assert _viewport_widgets(win) == (0, 0)
    ids = [_install(win, _doc(tmp_path / f"{k}.dat", _trace(k), with_voltage=True))[0]
           for k in range(3)]
    assert _viewport_widgets(win) == (6, 6), "one per channel row, and no residue"

    for doc_id in ids:
        win._session.close_document(doc_id)
    qapp.processEvents()
    qapp.sendPostedEvents(None, 0)
    qapp.processEvents()

    assert _viewport_widgets(win) == (0, 0)


# ================================================================================================
# The flag cell is a SET, and the curation keys reach it
# ================================================================================================

def test_the_f_key_adds_a_reason_instead_of_replacing_the_row_s_reasons(win, tmp_path):
    """The GUI wrote the curator's token straight over the ``flag`` cell, so marking an
    event the detector had flagged ``clipped_by_boundary`` would have erased that reason
    and left the row rejected with none — C21's failure, at the call site C21 did not
    reach. It survived review only because the keys were unreachable, which is why both
    halves had to land together."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    table = win._analysis_panel._table
    df = table.dataframe
    df.iat[0, list(df.columns).index("flag")] = FLAG_CLIPPED

    table._model.set_flag(0, FLAG_CURATED)

    tokens = flag_tokens(table.dataframe.iat[0, list(df.columns).index("flag")])
    assert FLAG_CLIPPED in tokens
    assert FLAG_CURATED in tokens


def test_the_documented_curation_keys_are_reachable_on_the_table(win, tmp_path, qapp):
    """C30's sibling: documented, shipped and dead.

    The handler was ``keyPressEvent`` on the CONTAINER; the focus is on the ``QTableView``
    inside it, and ``QAbstractItemView`` accepts a printable key for its own
    type-to-search, so the event never propagated up. Measured on a reverted tree with a
    REAL keystroke on the focused view: 'r' left ``accepted`` True and 'f' left the flag
    cell untouched. (A synthesised ``sendEvent`` on the view DOES propagate to the parent,
    which is why this test drives ``QTest.keyClick`` — the wrong probe would have passed
    both ways.)"""
    from PySide6.QtTest import QTest

    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    table = win._analysis_panel._table
    win.show()
    qapp.processEvents()
    table.view.setFocus()
    table.view.selectRow(0)
    qapp.processEvents()
    assert table.view.hasFocus()

    accepted_col = list(table.dataframe.columns).index("accepted")
    assert bool(table.dataframe.iat[0, accepted_col]) is True

    QTest.keyClick(table.view, QtCore.Qt.Key_R)
    qapp.processEvents()

    assert bool(table.dataframe.iat[0, accepted_col]) is False


def test_the_f_key_delivered_for_real_unions_instead_of_overwriting(win, tmp_path, qapp):
    """The two halves together, through the keyboard the user actually has: the key
    arrives, and what it writes is added to the row's reasons rather than replacing them.
    Measured on the reverted tree: ``'clipped_by_boundary'`` unchanged (the key never
    landed); here: ``'clipped_by_boundary;flagged'``."""
    from PySide6.QtTest import QTest

    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    table = win._analysis_panel._table
    flag_col = list(table.dataframe.columns).index("flag")
    table.dataframe.iat[0, flag_col] = FLAG_CLIPPED
    win.show()
    qapp.processEvents()
    table.view.setFocus()
    table.view.selectRow(0)
    qapp.processEvents()

    QTest.keyClick(table.view, QtCore.Qt.Key_F)
    qapp.processEvents()

    tokens = flag_tokens(table.dataframe.iat[0, flag_col])
    assert FLAG_CLIPPED in tokens and FLAG_CURATED in tokens


def test_the_curator_s_token_is_not_re_derived_by_a_rebuild(win):
    """It is a person's note, so it must be INHERITED by a merge or a resize, never
    re-derived away."""
    assert FLAG_CURATED not in DERIVED_FLAGS


# ================================================================================================
# Deleting a hand measurement is undoable
# ================================================================================================

def _add_measurements(win, doc_id, spans) -> None:
    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    vals = win._session.x_axis(doc_id).values
    for a, b in spans:
        win._trace_view.set_measure_region(float(vals[a]), float(vals[b]))
        QtWidgets.QApplication.processEvents()
        assert mp.add_current() is True


def test_deleting_a_hand_measurement_is_undoable(win, tmp_path):
    """The last destructive click in the application that nothing could take back. It
    warned — "nothing in the app can re-derive them" — which was true and no help at all to
    somebody who had just deleted the wrong four rows."""
    doc_id, _ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    mp = win._analysis_panel.measure_panel
    _add_measurements(win, doc_id, [(2_000, 2_300), (5_000, 5_300), (8_000, 8_300)])
    assert len(mp.measurements_df) == 3

    mp._table_view.selectAll()
    mp.remove_selected()
    assert len(mp.measurements_df) == 0

    win._on_edit_menu_about_to_show()
    assert win._act_undo.isEnabled()
    assert "Delete 3 measurements" in win._act_undo.text()
    win._undo_events()

    assert list(mp.measurements_df["measurement_id"]) == [0, 1, 2]


def test_undoing_a_measurement_delete_is_redoable(win, tmp_path):
    doc_id, _ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    mp = win._analysis_panel.measure_panel
    _add_measurements(win, doc_id, [(2_000, 2_300)])
    mp._table_view.selectAll()
    mp.remove_selected()
    win._undo_events()
    assert len(mp.measurements_df) == 1

    win._on_edit_menu_about_to_show()
    assert "Delete measurement" in win._act_redo.text()
    win._redo_events()

    assert len(mp.measurements_df) == 0


def test_one_ctrl_z_serves_both_tables_newest_first(win, tmp_path):
    """There is one Undo and there will not be a second: a user who has just deleted four
    measurement rows reaches for Ctrl+Z, not for a different Ctrl+Z. Which stack a press
    acts on is decided by recency."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    table = win._analysis_panel._table
    accepted_col = list(table.dataframe.columns).index("accepted")
    mp = win._analysis_panel.measure_panel
    _add_measurements(win, doc_id, [(2_000, 2_300)])

    table._model.set_accepted(0, False)          # step 1: the event table
    win._on_edit_menu_about_to_show()
    assert "Curate" in win._act_undo.text()

    mp._table_view.selectAll()
    mp.remove_selected()                          # step 2: the measurement table
    win._on_edit_menu_about_to_show()
    assert "Delete measurement" in win._act_undo.text()

    win._undo_events()                            # newest first: the measurement
    assert len(mp.measurements_df) == 1
    assert bool(table.dataframe.iat[0, accepted_col]) is False

    win._on_edit_menu_about_to_show()
    win._undo_events()                            # then the curation
    assert bool(table.dataframe.iat[0, accepted_col]) is True


def test_a_measurement_delete_marks_the_session_dirty(win, tmp_path):
    doc_id, _ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    mp = win._analysis_panel.measure_panel
    _add_measurements(win, doc_id, [(2_000, 2_300)])
    win._mark_clean()

    mp._table_view.selectAll()
    mp.remove_selected()

    assert win._dirty is True


# ================================================================================================
# The Edit menu names where the history IS
# ================================================================================================

def test_undo_says_which_other_detector_holds_this_channel_s_history(win, tmp_path):
    """Running the ML detector on a curated channel left Undo saying "nothing to undo on
    this event table" — honest, and indistinguishable from "your curation is gone". The
    Schmitt stack was intact the whole time, one detector switch away."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    table = win._analysis_panel._table
    table._model.set_accepted(0, False)
    win._on_edit_menu_about_to_show()
    assert win._history.can_undo((doc_id, ch, _MODULE_ID))

    _store_ml_result(win, doc_id, ch, {})
    win._analysis_panel._active_detection = _ML_MODULE_ID
    win._on_edit_menu_about_to_show()

    text = win._act_undo.text()
    assert "threshold detector" in text and "1 step" in text


# ================================================================================================
# C35 — a save that drops work does not report success
# ================================================================================================

def test_c35_a_save_that_drops_segments_leaves_the_session_dirty(win, tmp_path, monkeypatch):
    """The user was warned and chose "Save without them", so the file IS written — but the
    segments and their results are still live and still unrecoverable, and the file does
    not contain them. Clearing the dirty flag retired the quit guard on the one session
    that most needed it."""
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    seg_id = win._session.create_segment(doc_id, 1_000, 6_000, name="crop")
    _detect_into(win, seg_id, 0)
    win._session.close_document(doc_id)            # the segment's parent is gone
    monkeypatch.setattr(MainWindow, "_confirm_save_dropping_segments",
                        lambda self, n_seg, n_res: True)

    assert win._write_project(str(tmp_path / "p.nanoscope")) is True

    assert Path(tmp_path / "p.nanoscope").exists()
    assert win._dirty is True, "a save that knowingly dropped work is not a clean save"
    assert win.windowTitle().startswith("• ")


def test_an_ordinary_save_still_reports_success(win, tmp_path):
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    assert win._write_project(str(tmp_path / "p.nanoscope")) is True
    assert win._dirty is False


# ================================================================================================
# C34 — load_project raises the one exception it documents
# ================================================================================================

def _project_with(tmp_path: Path, **overrides) -> Path:
    from nanoscope.core.project import FORMAT, VERSION
    manifest = {
        "format": FORMAT, "version": VERSION, "app_version": "0.1.0",
        "documents": [], "segments": [], "calibrations": [], "layers": [],
        "view_mode": "overlay", "results": [], "measurements": None,
    }
    manifest.update(overrides)
    path = tmp_path / f"m{abs(hash(json.dumps(manifest, sort_keys=True))) % 10_000}.nanoscope"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    return path


@pytest.mark.parametrize("section,payload", [
    ("layers", {"layers": [{"doc_id": "/x", "channel": 0, "visible": True}]}),   # no colour
    ("documents", {"documents": [{"path": "/x", "loaded_range": "0:10"}]}),      # a string
    ("calibrations", {"calibrations": [{"doc_id": "/x", "channel": "two",
                                        "gain": 1.0, "offset": 0.0}]}),
    ("segments", {"segments": [{"segment_uid": "s", "parent_key": "/x", "i0": 0}]}),
    ("results", {"results": [{"doc_id": "/x", "channel": 0}]}),                  # no module_id
])
def test_c34_a_malformed_manifest_section_raises_projecterror(tmp_path, section, payload):
    """Six shapes escaped as bare ``KeyError`` / ``TypeError`` / ``ValueError`` through an
    API whose docstring promises exactly one exception type. The GUI's blanket
    ``except Exception`` is why nobody ever saw it — which is why it stayed wrong."""
    path = _project_with(tmp_path, **payload)
    with pytest.raises(ProjectError) as exc:
        load_project(str(path))
    assert section in str(exc.value)


def test_c34_a_well_formed_project_is_unaffected(win, tmp_path):
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", _trace()))
    _detect_into(win, doc_id, ch)
    proj = tmp_path / "good.nanoscope"
    assert win._write_project(str(proj)) is True
    state = load_project(str(proj))
    assert len(state.results) == 1


# ================================================================================================
# StagedSet leaves no empty directory behind
# ================================================================================================

def test_an_abandoned_write_set_removes_the_directory_it_had_to_create(tmp_path):
    """"Nothing on disk changes" is the promise; an empty folder named after an export that
    never happened is a change, and a confusing one."""
    from nanoscope.core.atomic_write import staged_write_set

    target = tmp_path / "fresh" / "deeper" / "a.csv"
    with pytest.raises(RuntimeError):
        with staged_write_set() as ss:
            ss.stage(target)
            raise RuntimeError("the family could not be built")

    assert not (tmp_path / "fresh").exists()


def test_a_committed_write_set_keeps_the_directory_it_created(tmp_path):
    from nanoscope.core.atomic_write import staged_write_set

    target = tmp_path / "fresh" / "a.csv"
    with staged_write_set() as ss:
        Path(ss.stage(target)).write_text("x")

    assert target.exists()


def test_an_abandoned_set_never_removes_a_directory_that_already_existed(tmp_path):
    from nanoscope.core.atomic_write import staged_write_set

    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "keep.txt").write_text("keep")
    with pytest.raises(RuntimeError):
        with staged_write_set() as ss:
            ss.stage(tmp_path / "mine" / "a.csv")
            raise RuntimeError("nope")

    assert (tmp_path / "mine" / "keep.txt").exists()


# ================================================================================================
# The FDR readout, and the sweep dialog's surrogate
# ================================================================================================

def test_a_refused_fdr_shows_the_number_it_refused_in_favour_of(qapp):
    """``null_control`` returns ``nan`` when it will not stand behind a rate, with the
    reason in ``notes`` and the honest count in ``n_event_free``. The panel tested
    ``fdr is None``, which ``nan`` is not — so on every real recording the readout said
    "estimated false-positive rate: nan%" and the number the refusal exists to produce
    never reached the screen."""
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr({"estimated_fdr": float("nan"), "n_real": 315, "n_event_free": 32,
                   "per_surrogate": [1354, 1350, 1360],
                   "notes": ("the AAFT surrogates produced 1354 detections where this "
                             "recording's own event-free noise produces 32",)})

    text = panel.fdr_label.text()
    assert "nan" not in text
    assert "NOT estimated" in text and "32" in text
    assert "AAFT surrogates produced" in panel.fdr_label.toolTip()


def test_a_real_fdr_is_still_shown_as_a_percentage(qapp):
    from nanoscope.gui.panels import StatsPanel

    panel = StatsPanel()
    panel.set_fdr({"estimated_fdr": 0.042, "n_real": 100, "n_event_free": 4,
                   "per_surrogate": [4, 4, 4], "notes": ()})
    assert "4.2%" in panel.fdr_label.text()


def test_the_sweep_dialog_does_not_call_a_non_null_surrogate_a_false_positive_count(qapp):
    """``sweep_threshold`` writes ``surrogate_is_a_null`` beside the ``k_recommended`` the
    dialog already reads. On the owner's apo file that curve sits ~88x ABOVE the real one
    and flat in k — drawn in red, labelled "surrogate (false +)", under a sentence
    promising the surrogate count "stays low"."""
    from nanoscope.gui.panels import SweepDialog

    df = pd.DataFrame({"k": [3.0, 4.0, 5.0, 6.0],
                       "n_events": [40.0, 30.0, 30.0, 20.0],
                       "n_surrogate": [2600.0, 2640.0, 2650.0, 2610.0],
                       "in_plateau": [False, True, True, False]})
    df.attrs["k_recommended"] = 4.5
    df.attrs["surrogate_is_a_null"] = False
    df.attrs["surrogate_note"] = "the AAFT surrogate produced a median of 2645 detections"

    dlg = SweepDialog(df)

    text = dlg._recommendation_text()
    assert "surrogate count stays low" not in text
    assert "median of 2645" in text
    names = [item.name() for item in dlg.plot.plotItem.listDataItems()]
    assert "surrogate (false +)" not in names
    assert any("NOT a null" in (n or "") for n in names)


def test_the_sweep_dialog_is_unchanged_when_the_surrogate_IS_a_null(qapp):
    from nanoscope.gui.panels import SweepDialog

    df = pd.DataFrame({"k": [3.0, 4.0, 5.0, 6.0],
                       "n_events": [40.0, 30.0, 30.0, 20.0],
                       "n_surrogate": [3.0, 2.0, 2.0, 1.0],
                       "in_plateau": [False, True, True, False]})
    df.attrs["k_recommended"] = 4.5
    df.attrs["surrogate_is_a_null"] = True
    df.attrs["surrogate_note"] = ""

    dlg = SweepDialog(df)

    assert "surrogate count stays low" in dlg._recommendation_text()
    names = [item.name() for item in dlg.plot.plotItem.listDataItems()]
    assert "surrogate (false +)" in names


# ================================================================================================
# The figure refusal names its rows
# ================================================================================================

def test_a_figure_refusal_names_the_rows_it_refused_over():
    """Item 2's own criterion was "every consumer, not one". ``stats._clean_1d`` names the
    offending rows in the user's own index; ``figures._column`` raised the same refusal and
    named none."""
    from nanoscope.plot import figures

    df = pd.DataFrame({"frac_depth": [0.1, np.nan, 0.3, 0.2]}, index=[7, 11, 13, 17])
    with pytest.raises(ValueError) as exc:
        figures._column(df, "frac_depth")
    msg = str(exc.value)
    assert "at rows 11" in msg
    assert "flag" in msg


def test_a_clean_figure_column_is_unaffected():
    from nanoscope.plot import figures

    df = pd.DataFrame({"frac_depth": [0.1, 0.2, 0.3]})
    assert figures._column(df, "frac_depth").tolist() == [0.1, 0.2, 0.3]


# ================================================================================================
# C30, measured the way the defect was: the KEY, not the action
# ================================================================================================

def test_ctrl_m_is_DELIVERED_with_the_measurement_page_hidden(qapp, qtbot, tmp_path):
    """C30 is a shortcut-DELIVERY defect, and every test covering it called ``.trigger()``.

    ``.trigger()`` bypasses Qt's shortcut map entirely: move the action back onto the panel
    with ``Qt.WindowShortcut`` — which is precisely the code hunt2 measured at 1 of 1 with the
    tab raised and **0 of 1 with it hidden** — and all three of those tests still pass. This one
    presses the key on a shown window, which is the only thing Qt will actually deliver to.

    It lives here rather than in ``test_measure_navigation.py`` because showing a window is not
    free of consequences for the tests that follow it in a file: the window is closed again at
    the end for the same reason.
    """
    from PySide6 import QtTest

    w = MainWindow()
    qtbot.addWidget(w)
    try:
        doc_id, _ch = _install(w, _doc(tmp_path / "a.dat", _trace()))
        mp = w._analysis_panel.measure_panel
        mp.set_active(True)
        x0 = w._trace_view.x_for_index(doc_id, 0, 2_000)
        x1 = w._trace_view.x_for_index(doc_id, 0, 3_000)
        w._trace_view.set_measure_region(x0, x1)

        w._analysis_panel._detection.show_detection_tab()
        assert not w._analysis_panel._detection._measure_page.isVisible()

        w.show()
        QtWidgets.QApplication.processEvents()
        QtTest.QTest.keyClick(w, QtCore.Qt.Key_M, QtCore.Qt.ControlModifier)
        # Ctrl+M files an EVENT now (here the Manual table: nothing was detected on this channel)
        manual = w._session.results.get(doc_id, 0, "detect.manual")
        assert manual is not None and len(manual.payload.df) == 1, (
            "Ctrl+M was not delivered with the page hidden")

        w._analysis_panel._detection.show_measure_tab()
        QtWidgets.QApplication.processEvents()
        # a second, different span: the same one would (rightly) be refused as an overlap
        w._trace_view.set_measure_region(w._trace_view.x_for_index(doc_id, 0, 5_000),
                                         w._trace_view.x_for_index(doc_id, 0, 6_000))
        QtTest.QTest.keyClick(w, QtCore.Qt.Key_M, QtCore.Qt.ControlModifier)
        assert len(w._session.results.get(doc_id, 0, "detect.manual").payload.df) == 2
    finally:
        w.close()
        QtWidgets.QApplication.processEvents()


# ================================================================================================
# "Visible view" over a view that is panned off the recording
# ================================================================================================

def test_a_view_panned_off_the_recording_is_refused_not_silently_widened(win, tmp_path):
    """The owner's own ABF starts at t0 = 61.303 s, so its x-axis does not contain 0.

    ``_visible_sample_range`` mapped a window lying entirely off the data to ``i1 <= i0`` and
    took the zero-width fallback, which returns the WHOLE RECORDING. Measured: a view framed
    on 10–20 s of a recording drawn at 61–121 s analysed all 2,993,798 samples while the
    control still read "Visible view". A disjoint window is not a degenerate one.
    """
    arr = _trace()
    doc_id, ch = _install(win, _doc(tmp_path / "a.dat", arr))
    panel = win._analysis_panel
    n = arr.size

    # On the data: still the honest whole-recording fallback for a ZERO-WIDTH view.
    axis = win._trace_view.x_axis_for(doc_id, ch).values
    mid = float(axis[len(axis) // 2])
    win._trace_view.anchor_plot().setXRange(mid, mid, padding=0)
    QtWidgets.QApplication.processEvents()
    assert panel._visible_sample_range(doc_id, ch, n) is not None

    # Entirely off the data: refused.
    far = float(axis[-1]) + 1_000.0
    win._trace_view.anchor_plot().setXRange(far, far + 10.0, padding=0)
    QtWidgets.QApplication.processEvents()
    assert panel._visible_sample_range(doc_id, ch, n) is None

    refusals: list[str] = []
    panel.runRefused.connect(refusals.append)
    panel.can_run_ml = lambda: True          # the gate is not what is under test here
    panel._on_ml_detect_requested({"range_source": "visible", "model_path": "nope.keras"})
    assert refusals, "a disjoint view must refuse out loud, not analyse the whole recording"
    assert "panned off" in refusals[0]
