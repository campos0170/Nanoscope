"""Usage examples of the Oct 18-19 NanoScope features on a real apo-ferritin recording.

Drives the actual NanoScope widgets offscreen (QT_QPA_PLATFORM=offscreen, QWidget.grab()),
so nothing opens on the desktop and no real preferences, recovery files or recordings are
touched: QSettings, recovery and the study library are redirected to WORK.

Run from the repository root:
    PYTHONPATH=. <python> examples/apo-ferritin-new-features/reproduction/capture_examples.py

Environment overrides: NANOSCOPE_EXAMPLES_OUT (output folder), NANOSCOPE_EXAMPLES_WORK
(private settings/recovery folder), NANOSCOPE_EXAMPLES_ABF (the -600 mV recording).
"""
import gc
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("NANOSCOPE_EXAMPLES_OUT", str(HERE.parent)))
WORK = Path(os.environ.get("NANOSCOPE_EXAMPLES_WORK",
                           str(Path(tempfile.gettempdir()) / "nanoscope-examples-work")))
ABF = Path(os.environ.get(
    "NANOSCOPE_EXAMPLES_ABF",
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf"))
SHOTS = OUT / "screenshots"
FILES = OUT / "outputs"
for p in (SHOTS, FILES, WORK / "settings", WORK / "recovery", WORK / "library"):
    p.mkdir(parents=True, exist_ok=True)

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLCONFIGDIR"] = str(WORK / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(WORK / "cache")
os.environ["NANOSCOPE_RECOVERY_ROOT"] = str(WORK / "recovery")
os.environ["NANOSCOPE_INCUBATION_LIBRARY_ROOT"] = str(WORK / "library")
gc.disable()  # the test suite's policy: PySide6 cyclic GC can deadlock on live wrappers

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
for scope in (QtCore.QSettings.UserScope, QtCore.QSettings.SystemScope):
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, scope, str(WORK / "settings"))

import nanoscope  # noqa: E402
from nanoscope.analysis.conditioning_recipes import complete_recipe  # noqa: E402
from nanoscope.analysis.curation_view import (  # noqa: E402
    FILTER_ALL, FILTER_UNREVIEWED, label_filter)
from nanoscope.analysis.labels import LabelVocabulary  # noqa: E402
from nanoscope.analysis.modules import register_all  # noqa: E402
from nanoscope.gui import analysis_panel as analysis_panel_mod  # noqa: E402
from nanoscope.gui import panels as panels_mod  # noqa: E402
from nanoscope.gui.app import _apply_theme, configure_pyqtgraph, create_app  # noqa: E402
from nanoscope.gui.main_window import MainWindow  # noqa: E402
from nanoscope.gui.overlays import register_all_renderers  # noqa: E402
from nanoscope.io.base import open_document  # noqa: E402

PROTOCOL_NAME = "Apo example - 10 kHz Bessel low-pass"
MANIFEST: dict = {
    "capture_method": "QT_QPA_PLATFORM=offscreen; QWidget.grab(); no desktop capture",
    "nanoscope_version": nanoscope.__version__,
    "recording": str(ABF),
    "steps": [], "screenshots": [], "files": [],
}


def save_manifest() -> None:
    (OUT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2, default=str))


def log(s: str) -> None:
    print(s, flush=True)


def pump(ms: int = 50) -> None:
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)


def until(fn, timeout: float = 120, name: str = "operation") -> None:
    end = time.monotonic() + timeout
    while not fn():
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
        if time.monotonic() > end:
            raise TimeoutError(name)
    pump(80)


def step(name: str, ok: bool, **details) -> None:
    MANIFEST["steps"].append({"name": name, "ok": bool(ok), **details})
    save_manifest()
    log(f"[{'ok' if ok else 'FAIL'}] {name}")


def shot(widget: QtWidgets.QWidget, name: str, caption: str) -> None:
    pump(220)
    path = SHOTS / f"{name}.png"
    if not widget.grab().save(str(path)):
        raise RuntimeError(f"could not save {path}")
    MANIFEST["screenshots"].append({"file": f"screenshots/{path.name}", "caption": caption,
                                    "size": [widget.width(), widget.height()]})
    save_manifest()
    log(f"  screenshot {path.name}")


def preview_idle(ws) -> None:
    until(lambda: ws.preview_state() != "pending" and not ws._debounce.isActive(),
          name="Signal Conditioning preview")


app = create_app([])
assert app.platformName() == "offscreen"
QtCore.QSettings().clear()
configure_pyqtgraph()
_apply_theme(app)
app.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
register_all()
register_all_renderers()

win = MainWindow(recovery_enabled=False)
win._confirm_close_choice = lambda: "discard"
win.resize(1600, 1000)
win.show()
pump(200)
win._inspector_tabs.setCurrentWidget(win._analysis_panel)
panel = win._analysis_panel
det = panel.detection_panel
status_bar = win.statusBar()
completed = False

try:
    # --- 0. Load the -600 mV recording and detect --------------------------------------
    log(f"Loading {ABF.name}")
    doc = open_document(ABF)
    did = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(did, 2)            # IN15, the current channel of these Process files
    pump(200)
    fs = float(doc.meta.sample_rate_hz)
    duration_s = doc.meta.n_samples / fs
    MANIFEST["recording_meta"] = {"n_samples": int(doc.meta.n_samples), "fs_hz": fs,
                                  "duration_s": duration_s}
    step("Load recording", True, fs_hz=fs, duration_s=round(duration_s, 2))

    det.show_open_pore_baseline()
    bp = panel.baseline_panel
    bp.probe_button.click()
    until(lambda: bp.proposal() is not None and panel._pending is None, 180,
          "baseline measurement")
    applied = bp.apply_proposal()
    step("Measure and apply the open-pore baseline", applied, proposal=bp.proposal_text())

    # CUSUM with a 5 kHz 8-pole Bessel low-pass: the chain the Results table will remember.
    det.set_params({**complete_recipe({}), "detector": "detect.cusum", "polarity": -1,
                    "cusum_k": 0.75, "cusum_h": 28.0, "min_dwell_s": 0.001,
                    "merge_gap_s": 0.0005, "filter_cutoff_hz": 5000.0, "filter_order": 8,
                    "filter_kind": "bessel"})
    params = panel.detection_params()
    MANIFEST["detection_params"] = params
    assert panel._run_detection(params), "detection was refused"
    until(lambda: panel._pending is None and panel.current_detection() is not None, 300,
          "CUSUM detection")
    df = panel.current_detection().payload.df
    n_events, n_accepted = len(df), int(df.accepted.sum())
    step("CUSUM detection", n_events > 0, events=n_events, accepted=n_accepted,
         summary=panel._results_summary.text())
    df.to_csv(FILES / "cusum-600mV-events.csv", index=False)
    MANIFEST["files"].append("outputs/cusum-600mV-events.csv")

    # --- 1. The Results dock -------------------------------------------------------------
    clock = win._trace_view.clock_offset_for(did, 2) or 0.0
    focus_row = df.iloc[len(df) // 3]
    t0 = float(focus_row.t_start_s)
    win._trace_view.zoom_x_to(clock + max(0.0, t0 - 0.2), clock + t0 + 0.8)
    det.mode_tabs.setCurrentIndex(1)
    panel.results_tabs.setCurrentIndex(0)
    shot(win, "01-results-dock-events",
         "Results dock under the trace: the CUSUM event table, with the Inspector keeping "
         "only the parameter forms.")
    panel.results_tabs.setCurrentIndex(1)
    shot(win, "02-results-dock-summary",
         "Results dock, Summary tab: event counts, dwell and depth summaries.")
    panel.results_tabs.setCurrentIndex(0)
    win._act_toggle_results.trigger()
    pump(200)
    hidden = not win._results_dock.isVisible()
    shot(win, "03-results-dock-hidden",
         "View > Panels > Results (Shift+Cmd+R) hides the dock; the trace takes the full "
         "height and nothing is lost.")
    win._act_toggle_results.trigger()
    pump(200)
    step("Toggle the Results dock", hidden and win._results_dock.isVisible())

    # --- 2. Signal Conditioning ----------------------------------------------------------
    a = min(30.0, max(0.0, duration_s - 12.0))
    win._trace_view.zoom_x_to(clock + a, clock + a + 10.0)   # previews follow this window
    win._open_signal_conditioning()
    ws = win._signal_conditioning
    ws.resize(1600, 1000)
    ws.set_follow_main(True)
    win._push_conditioning_source()
    preview_idle(ws)
    ws.set_time_mode("overlay")
    pump(150)
    step("Open Signal Conditioning (Time preview)", ws.preview_state() in ("exact", "window"),
         state=ws.preview_state(), status=ws.time_status.text(),
         stages=[str(r) for r in ws.stage_rows()])
    shot(ws, "04-sc-time-overlay",
         "Signal Conditioning, Time preview (Overlay): raw against the 5 kHz Bessel output "
         f"over the main trace's visible 10 s ({a:.0f}-{a + 10:.0f} s into the recording; "
         "the axis shows the instrument clock).")
    if ws.set_time_mode("difference"):
        pump(250)
        shot(ws, "05-sc-time-difference",
             "Time preview, Raw - processed: what the low-pass removes from the current.")
    ws.set_time_mode("overlay")

    ws.set_preview_tab("spectrum")
    preview_idle(ws)
    lines = ws.spectral_line_texts()
    step("Spectrum preview", ws.last_result() is not None, lines=lines)
    shot(ws, "06-sc-spectrum",
         "Spectrum preview: Welch PSD before and after the chain, with the peaks the line "
         "finder flags.")

    ws.set_preview_tab("response")
    ws.set_response_view("magnitude")
    pump(300)
    shot(ws, "07-sc-response-magnitude",
         "Response preview: magnitude of the 5 kHz Bessel low-pass.")
    if ws.set_response_view("step"):
        pump(300)
        shot(ws, "08-sc-response-step",
             "Response preview, Step: how the low-pass rounds an ideal blockade edge.")
    ws.set_response_view("magnitude")
    ws.set_preview_tab("time")

    # Change the chain after detection: raise the low-pass from 5 to 10 kHz. (No notch:
    # the line finder flags no mains line in this recording.)
    det.set_params({"filter_cutoff_hz": 10000.0})
    status = panel.refresh_chain_status()
    preview_idle(ws)
    step("Edit the chain after detection", status.kind == "differs", sentence=status.sentence)
    shot(ws, "09-sc-chain-edited",
         "The low-pass raised from 5 to 10 kHz after detection; the window names how the "
         "form now differs from the CUSUM table on screen.")
    win._trace_view.zoom_x_to(clock + max(0.0, t0 - 0.2), clock + t0 + 0.8)
    pump(200)
    shot(win, "10-results-chain-changed",
         "The Results toolbar shows '⚠ Chain changed' with Revert / Run / Open.")

    # Save the edited chain as a protocol, export it, and copy it as Python.
    orig_get_text = panels_mod.QInputDialog.getText
    panels_mod.QInputDialog.getText = staticmethod(lambda *a, **k: (PROTOCOL_NAME, True))
    try:
        det._save_protocol()
    finally:
        panels_mod.QInputDialog.getText = orig_get_text
    protocol_path = FILES / "conditioning-protocols.json"
    orig_ask = analysis_panel_mod.ask_export_path
    analysis_panel_mod.ask_export_path = lambda *a, **k: str(protocol_path)
    try:
        exported = panel.export_protocols(ws)
    finally:
        analysis_panel_mod.ask_export_path = orig_ask
    step("Save and export the chain as a protocol", exported and protocol_path.exists(),
         protocol=PROTOCOL_NAME)
    if protocol_path.exists():
        MANIFEST["files"].append("outputs/conditioning-protocols.json")
    script = panel.copy_chain_as_python(ws.current_window())
    (FILES / "rebuild_processed_trace.py").write_text(script)
    clip = QtGui.QGuiApplication.clipboard().text()
    step("Copy as Python", bool(script) and clip == script, chars=len(script))
    MANIFEST["files"].append("outputs/rebuild_processed_trace.py")

    reverted = panel.revert_chain_to_shown_detection()
    after = panel.refresh_chain_status()
    step("Revert the chain to the shown detection", reverted and after.kind != "differs",
         after=after.kind)
    ws.hide()

    # --- 3. Curation: labels, Unreviewed filter, Advance after decision -------------------
    vocab = LabelVocabulary()
    vocab.add("shape", "single-level", "1", "One flat blockade level")
    vocab.add("shape", "multi-level", "2", "Two or more distinct levels")
    vocab.add("quality", "artifact", "3", "Not a translocation (spike, drift, clipping)")
    win._label_vocab = vocab
    win._push_vocabulary()

    win._open_event_inspector()
    insp = win._event_inspector
    insp.resize(1200, 880)
    insp.set_filter(FILTER_UNREVIEWED)
    insp.set_advance_after_decision(True)
    # Start on an accepted event: event 0 is boundary-clipped, and the app refuses to
    # accept a clipped event (so A would correctly stay put there).
    accepted_rows = df.loc[df.accepted.astype(bool)]
    accepted_ids = (accepted_rows.event_id if "event_id" in df.columns
                    else accepted_rows.index).to_list()
    insp.show_event(int(accepted_ids[len(accepted_ids) // 3]))
    pump(200)
    decisions = []
    e0 = insp.current_event_id()
    insp.decide_accept(True); pump(150)
    decisions.append({"event_id": e0, "action": "accept (A)"})
    e1 = insp.current_event_id()
    insp.decide_label("1"); pump(150)
    decisions.append({"event_id": e1, "action": "label shape=single-level (1)"})
    e2 = insp.current_event_id()
    insp.decide_label("2"); pump(150)
    decisions.append({"event_id": e2, "action": "label shape=multi-level (2)"})
    e3 = insp.current_event_id()
    MANIFEST["demo_decisions"] = decisions
    step("Decide with Advance after decision", len({e0, e1, e2, e3}) == 4,
         visited=[e0, e1, e2, e3], progress=insp.progress_text())
    shot(insp, "11-inspector-advance",
         "Event inspector on the Unreviewed filter with Advance after decision: each A / R / "
         "label key moves to the next unreviewed event; the progress line counts them.")

    insp.set_filter(label_filter("shape=multi-level"))
    pump(200)
    step("Labelled filter", insp.current_event_id() == e2, shown=insp.current_event_id())
    shot(insp, "12-inspector-labelled-filter",
         "The Labelled: shape=multi-level filter shows only events carrying that label.")
    insp.set_filter(FILTER_UNREVIEWED)
    insp.hide()

    win._open_event_gallery()
    gal = win._event_gallery
    gal.resize(1600, 1000)
    gal.set_filter(FILTER_UNREVIEWED)
    pump(300)
    step("Event gallery on Unreviewed", True, progress=gal.progress_text())
    shot(gal, "13-gallery-unreviewed",
         "Event gallery on the Unreviewed filter, with the shared progress line and "
         "First unreviewed (Cmd+U).")
    gal.hide()

    # --- 4. The Event inspector docked beside the trace -----------------------------------
    win.set_curation_host("event_inspector", "dock", present=True)
    pump(300)
    insp.set_filter(FILTER_ALL)
    insp.show_event(e2)
    pump(200)
    step("Dock the Event inspector", win.curation_host("event_inspector") == "dock")
    shot(win, "14-inspector-docked",
         "Window > Dock Event inspector beside the trace: the same inspector as a tab in the "
         "right column, with Details collapsed.")
    win.set_curation_host("event_inspector", "window", present=False)
    insp.hide()
    pump(150)

    # --- 5. The Window menu ----------------------------------------------------------------
    # Never reach a menu through QAction.menu(): in this PySide6 the temporary wrapper takes
    # ownership and deletes the live menu (see MainWindow._build_actions_and_menus).
    window_menu = win.findChild(QtWidgets.QMenu, "windowMenu")
    if window_menu is not None:
        win._sync_window_menu()
        window_menu.popup(win.mapToGlobal(QtCore.QPoint(600, 30)))
        pump(250)
        step("Window menu", True,
             entries=[a.text() for a in window_menu.actions() if a.isVisible() and a.text()])
        shot(window_menu, "15-window-menu",
             "The new Window menu: open windows, Dock Event inspector beside the trace and "
             "Reset Window Layout…")
        window_menu.hide()
    else:
        step("Window menu", False, reason="no Window menu found")

    # --- 6. Save the example project ----------------------------------------------------
    project = FILES / "apo-ferritin-new-features.nanoscope"
    if project.exists():
        project.unlink()
    saved = win._write_project(str(project))
    step("Save the example project", saved and project.exists())
    MANIFEST["files"].append("outputs/apo-ferritin-new-features.nanoscope")
    completed = True
    MANIFEST["completed"] = True
    save_manifest()
except Exception as exc:  # noqa: BLE001 - record and exit non-zero
    MANIFEST["error"] = str(exc)
    MANIFEST["traceback"] = traceback.format_exc()
    save_manifest()
    traceback.print_exc()
finally:
    if win._signal_conditioning is not None:
        win._signal_conditioning.abort()
    win._analysis_controller.abort()
    pump(100)
    log("Finished")
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: the known PySide6 wrapper-GC shutdown hazard.
    os._exit(0 if completed else 1)
