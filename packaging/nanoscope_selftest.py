"""In-bundle smoke test.

Reached with `NanoScope.app/Contents/MacOS/NanoScope --self-test` on macOS, or
`NanoScope\\NanoScope.exe --self-test` on Windows. The checks are the same; the three that ask
about the BUNDLE LAYOUT resolve the platform's own answer rather than assuming a .app.

A frozen app can start, show a window, and still be broken in ways nobody notices until a labmate
tries to open a TDMS file six weeks later. Every check below corresponds to a specific way THIS
bundle can fail — a lazily imported reader that PyInstaller could not see, a scipy LAPACK call on
an undersized thread stack, TensorFlow grabbing the Metal device, matplotlib with no writable
config dir. It runs entirely headless (offscreen Qt), takes ~1 minute, and is what the build is
verified with.

    NanoScope --self-test                 # everything that needs no data files
    NanoScope --self-test FILE.abf        # also load a real recording end to end
    NanoScope --self-test FILE.abf M.keras  # also run the ML detector

Exit code is the number of failed checks — or 2, with nothing checked and BAD ARGUMENTS printed,
when a path on the command line does not exist (see _argument_problems).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

_results: list[tuple[str, bool, str]] = []

# Import time of this module, which the launcher imports only when --self-test is on the command
# line: near enough to process start to tell "matplotlib's font cache was written during THIS run"
# from "it was already on disk and got reused". See the matplotlib check.
_STARTED_AT = time.time()


def check(name: str):
    """Decorator: run the function, record pass/fail, never let it abort the run."""

    def wrap(fn):
        t0 = time.time()
        print(f"  [RUN] {name}", flush=True)
        try:
            detail = fn() or ""
            _results.append((name, True, f"{detail} [{time.time() - t0:.1f}s]"))
        except Exception as exc:  # noqa: BLE001 - a failed check is data, not a crash
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        _, passed, detail = _results[-1]
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
        return fn

    return wrap


def _argument_problems(argv: list[str]) -> list[str]:
    """Everything wrong with this command line, in plain words. An empty list means "go".

    "No file was given" and "the file you gave me is not there" used to be the same branch: three
    checks returned "SKIPPED (no file passed)" for both, @check records a RETURN as a PASS, and
    the run printed "18/18 checks passed" and exited 0. So one mistyped character, a stray flag,
    or — the overwhelmingly likely case for a recording named "500 pM Apo ferritin_1 x PBS_22
    nm_0004_process _ -1000 mV_ 385-400 s time trace.abf" — a path pasted into a shell without
    quotes certified a build whose ABF reader, Keras loader, TensorFlow and ML-through-the-
    controller path were never exercised. The summary line and the exit code were byte-identical
    to a run that did all of it. A harness that reports success for a run it never did is the
    worst failure mode there is, so a path that was NAMED and is not there stops the run.
    """
    problems: list[str] = []
    paths = [a for a in argv if not a.startswith("-")]
    for flag in (a for a in argv if a.startswith("-")):
        problems.append(f"unknown option {flag!r}: the only flag this understands is --self-test, "
                        "and everything else on the line must be a file path")
    for label, raw in zip(("data file", "model file"), paths):
        path = Path(raw).expanduser()
        if not path.exists():
            problems.append(f"the {label} you passed does not exist: {path}")
    if len(paths) > 2:
        problems.append(
            f"{len(paths) - 2} argument(s) after FILE.abf and MODEL.keras were ignored: "
            f"{paths[2:]} — a path containing spaces has to be quoted, or the shell splits it "
            "into several arguments and only the first word reaches this program")
    return problems


def _remove_scratch(tmp: Path) -> None:
    """Delete this run's scratch directory, and say on stdout what was in it.

    Every run used to abandon its scratch tree — ~19.5 MB, populations.h5 alone is 14.4 MB — and
    nothing ever collected it: 92 such directories totalling 1.66 GB had accumulated in this
    machine's TMPDIR, the oldest six days old. On Windows, where build_app.ps1 runs --self-test
    on every build and %TEMP% has no reaper at all, the growth is unbounded. The rest of the
    codebase already cleans up after itself (ml_detect.py's `finally: shutil.rmtree`,
    make_icon.py, build_app.sh); this module was the lone outlier.

    Printing the tally is the point of doing it here rather than silently: a run that failed a
    check now says exactly which artefacts were thrown away and where they had been.
    """
    import shutil

    files = [p for p in tmp.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    shutil.rmtree(tmp, ignore_errors=True)
    state = "still present" if tmp.exists() else "removed"
    print(f"scratch: {len(files)} file(s), {total / 1e6:.1f} MB in {tmp} — {state}")


def _acid_numerics_smoke() -> str:
    """Exercise compiled acid dependencies inside the frozen executable."""
    from copy import deepcopy
    from types import FunctionType
    import numpy as np
    import scipy
    import hdbscan
    import umap
    from diptest import diptest
    from nanoscope.analysis import acid_incubation_math as acid_math
    from nanoscope.analysis.acid_incubation_config import DEFAULTS
    from nanoscope.analysis.acid_incubation_numerics import median_filter_1d

    assert scipy.__version__.split('+')[0] not in {'1.15.0', '1.15.1'}, (
        'This SciPy version has a known one-dimensional median-filter defect')
    signal = np.array([3., 8., -2., 7., 1., 1., 9.], dtype=np.float32)
    padded = np.pad(signal, (2, 2), mode='symmetric')
    expected = np.array([np.median(padded[i:i+5]) for i in range(len(signal))])
    np.testing.assert_array_equal(median_filter_1d(signal, 5), expected)

    # The same static stage code runs with private recording globals in the app.
    namespace = dict(vars(acid_math))
    namespace.update(deepcopy(DEFAULTS))
    namespace['_acid_cancel'] = None
    stage = acid_math.stage_016
    FunctionType(stage.__code__, namespace, stage.__name__, stage.__defaults__,
                 stage.__closure__)()
    assert namespace['HAVE_NUMBA'], 'Acid CUSUM lacks its compiled Numba runtime'
    rng = np.random.default_rng(68)
    x = np.vstack([rng.normal(-4, .15, (40, 3)), rng.normal(4, .15, (40, 3))])
    labels = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5,
                            core_dist_n_jobs=1).fit_predict(x)
    assert len(set(labels) - {-1}) == 2, 'Compiled HDBSCAN lost the two separated groups'
    statistic, pvalue = diptest(x[:, 0])
    assert np.isfinite(statistic) and 0 <= pvalue <= 1, 'Dip-test extension failed'
    embedding = umap.UMAP(n_neighbors=8, n_components=2, random_state=68,
                          n_jobs=1, n_epochs=20).fit_transform(x)
    assert embedding.shape == (80, 2) and np.isfinite(embedding).all()
    return 'Acid CUSUM core checks, HDBSCAN, dip test and UMAP completed'


def _comparison_figures_smoke(tmp: Path) -> str:
    """Exercise the frozen Seaborn path using stored measurements, never detection."""
    import pandas as pd
    from PySide6 import QtCore, QtGui, QtWidgets

    from nanoscope.core.sources import PlotSource
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
    from nanoscope.gui.plot_export import write_live_plot_image
    from nanoscope.plot import comparison_figures

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    frame = pd.DataFrame({
        "event_id": [5, 5, 9, 11, 14, 17],
        "dwell_s": [.001, .002, .003, .004, .005, .006],
        "depth_pA": [10., 20., 30., 40., 50., 60.],
        "ecd_pA_s": [.01, .04, .09, .16, .25, .36],
        "snr": [5., 8., 10., 12., 15., 16.],
        "accepted": [True, True, False, True, True, True],
    }, index=[2, 2, 4, 9, 10, 10])
    sources = [PlotSource(key, label, frame.copy(deep=True), kind="manual", doc_key=key)
               for key, label in (("smoke-a", "Recording A"), ("smoke-b", "Recording B"))]
    originals = [source.df.copy(deep=True) for source in sources]
    signatures = {"png": b"\x89PNG\r\n\x1a\n", "pdf": b"%PDF-", "svg": b"<svg"}
    outputs = 0
    for kind, points in (("facets", "strip"), ("pairs", "strip"),
                         ("events", "strip"), ("events", "swarm")):
        page, restored = ComparisonFiguresWindow(), ComparisonFiguresWindow()
        try:
            page.resize(1000, 740)
            page.set_sources(sources, selected_keys=[source.key for source in sources])
            page.set_config({"kind": kind, "points": points,
                             "features": ["dwell_s", "depth_pA", "ecd_pA_s"]})
            if not page.analyze() or not page.has_data():
                raise AssertionError(f"{kind}/{points}: no generated figure")
            payload, tables = page.snapshot_state()
            if len(tables["observations"]) != 10:
                raise AssertionError("Comparison accepted-only selection lost or added rows")

            def no_prepare(*args, **kwargs):
                raise AssertionError("Snapshot restoration or appearance reran preparation")

            prepare = comparison_figures.prepare_comparison
            comparison_figures.prepare_comparison = no_prepare
            try:
                restored.restore_state(payload, tables)  # No sources offered to this page.
                restored.set_appearance_state({"title": "Stored comparison", "background": "light"})
                if not restored.has_data():
                    raise AssertionError("Source-free comparison snapshot has no figure")
                for name, expected in tables.items():
                    pd.testing.assert_frame_equal(restored._tables[name], expected, check_exact=True)
                for extension, signature in signatures.items():
                    path = tmp / f"comparison-{kind}-{points}.{extension}"
                    notes = write_live_plot_image(path, restored.figure_export_target())
                    data = path.read_bytes()
                    if notes or len(data) < 1000 or signature not in data[:500]:
                        raise AssertionError(f"Unusable comparison export: {path.name}; {notes}")
                    if extension == "png" and QtGui.QImage(str(path)).isNull():
                        raise AssertionError(f"Unreadable comparison PNG: {path.name}")
                    outputs += 1
            finally:
                comparison_figures.prepare_comparison = prepare
            for source, original in zip(sources, originals):
                pd.testing.assert_frame_equal(source.df, original, check_exact=True)
        finally:
            for window in (page, restored):
                window.close()
                window.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            app.processEvents()
    return f"facets + pairs + strip + swarm; {outputs} PNG/PDF/SVG exports; exact source-free snapshots"


def run(argv: list[str]) -> int:
    problems = _argument_problems(argv)
    if problems:
        # Loud and early, before a minute of checks whose result would be meaningless: the run
        # the operator asked for cannot be performed, so no run is reported at all.
        print("\n" + "=" * 78)
        print("NanoScope bundle self-test: BAD ARGUMENTS — nothing was checked")
        print("=" * 78)
        for problem in problems:
            print(f"  ! {problem}")
        print("=" * 78)
        print("0 of 0 checks ran. Usage:")
        print("    NanoScope --self-test                   # everything that needs no data files")
        print("    NanoScope --self-test FILE.abf          # also a real recording end to end")
        print("    NanoScope --self-test FILE.abf M.keras  # also the ML detector")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="nanoscope_selftest_"))
    try:
        return _run_checks(argv, tmp)
    finally:
        _remove_scratch(tmp)


def _project_restore_smoke(tmp):
    """Real reader, nested segment/layer restore, and Qt window teardown; no pytest.

    Caller owns QApplication and isolated settings/recovery paths. Synthetic data
    keeps this check portable. This exercises the restore/TraceView rebuild seam;
    it is not a reproduction of the earlier state-sensitive native fault.
    """
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from PySide6 import QtCore, QtWidgets
    from nanoscope.core.calibration import ChannelCalibration
    from nanoscope.core.layer import ViewMode
    from nanoscope.core.types import ChannelKind, Provenance
    from nanoscope.gui.app import configure_pyqtgraph, create_app
    from nanoscope.gui.main_window import MainWindow
    from nanoscope.gui.overlays import register_all_renderers

    app = create_app([])
    configure_pyqtgraph()
    register_all_renderers()
    tmp = Path(tmp)
    rec = tmp / "restore-smoke.csv"
    project = tmp / "restore-smoke.nanoscope"
    n, fs = 60_000, 50_000.0
    pd.DataFrame({"time": np.arange(n) / fs,
                  "voltage": np.full(n, -600.0),
                  "current": 100.0 + np.sin(np.arange(n) / 37.0)}).to_csv(rec, index=False)

    dismissed = []

    def answer_modal():
        dialog = QtWidgets.QApplication.activeModalWidget()
        if dialog is not None:
            detail = dialog.text() if isinstance(dialog, QtWidgets.QMessageBox) else ""
            dismissed.append(f"{dialog.windowTitle()}: {detail}")
            dialog.reject()

    # A warning enters a nested event loop before wait_for can check its clock.
    # Reject it using that event loop, retain the diagnostic, and fail the check.
    modal_pump = QtCore.QTimer()
    modal_pump.timeout.connect(answer_modal)
    modal_pump.start(100)

    def wait_for(predicate, label, timeout=60):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"{label} timed out after {timeout}s")
            app.processEvents()
            if dismissed:
                raise AssertionError(f"Unexpected restore dialog(s): {dismissed}")
            time.sleep(0.005)
        app.processEvents()

    def destroy(window):
        # Only disposable self-test work can be discarded if an assertion failed.
        window._confirm_close_choice = lambda: "discard"
        if not window.close():
            raise AssertionError("self-test MainWindow refused to close")
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()

    windows = []
    try:
        win = MainWindow(recovery_enabled=False)
        windows.append(win)
        win.show()
        win.open_paths([str(rec)])
        wait_for(lambda: win._session.doc_id_for_source(str(rec.resolve())) is not None
                 and win._active_open is None and not win._open_queue, "source load")
        session = win._session
        doc_id = session.doc_id_for_source(str(rec.resolve()))
        channel = next(c.index for c in session.document(doc_id).meta.channels
                       if c.kind is ChannelKind.CURRENT)
        session.set_calibration(doc_id, channel, ChannelCalibration(
            0.5, -1.0, "pA", Provenance.MEASURED, "restore smoke"), write_sidecar=False)
        cut_id = session.create_segment(doc_id, 5_000, 25_000, name="cut")
        win._create_default_layers(session.document(cut_id))
        nested_id = session.create_segment(cut_id, 2_000, 9_000, name="nested")
        win._create_default_layers(session.document(nested_id))
        uids = [session.document(i).origin.segment_uid for i in (cut_id, nested_id)]
        session.set_calibration(nested_id, 0, ChannelCalibration(
            0.037, 0.0, "nS", Provenance.UNKNOWN, "segment-local"), write_sidecar=False)
        session.set_view_mode(ViewMode.STACKED)
        win._on_selection(doc_id, channel)
        for layer in session.layers():
            session.set_layer_visible(layer.layer_id, True)
        session.set_layer_color(f"{doc_id}#{channel}", "#E57373")
        session.set_layer_time_offset(f"{doc_id}#{channel}", 0.25)
        visible_count = sum(layer.visible for layer in session.layers())
        assert visible_count >= 3, "nested segment layers are not visible"
        assert win._write_project(str(project)), f"project save failed; dialogs: {dismissed}"
        destroy(win)
        windows.remove(win)

        reopened = MainWindow(recovery_enabled=False)
        windows.append(reopened)
        reopened.show()
        restored = []
        reopened.projectRestored.connect(lambda: restored.append(True))
        reopened._open_project(str(project))
        wait_for(lambda: bool(restored), "project restore")
        session = reopened._session
        live = session.doc_id_for_source(str(rec.resolve()))
        assert live is not None and session.document(live).meta.n_samples == n
        segments = {d.origin.segment_uid: d for d in session.documents() if hasattr(d, "origin")}
        assert [segments[uid].meta.n_samples for uid in uids] == [20_000, 7_000]
        assert len(session.documents()) == 3
        cal = session.calibration(live, channel)
        assert (cal.gain, cal.offset, cal.target_units, cal.note) == (0.5, -1.0, "pA", "restore smoke")
        nested_cal = session.calibration(segments[uids[1]].doc_id, 0)
        assert (nested_cal.gain, nested_cal.target_units, nested_cal.note) == (0.037, "nS", "segment-local")
        layer = next(l for l in session.layers() if l.doc_id == live and l.channel == channel)
        assert layer.color == "#E57373" and abs(layer.time_offset_s - 0.25) < 1e-12
        assert session.view_mode is ViewMode.STACKED
        assert sum(l.visible for l in session.layers()) == visible_count
        assert reopened._carried_over is None
        assert not list(tmp.glob("*.npcal.json")), "self-test wrote a calibration sidecar"
        assert not dismissed, f"Unexpected restore dialog(s): {dismissed}"
        return f"60k samples; nested 20k/7k segments; {visible_count} visible layers; calibration + colour + offset survived close/reopen"
    finally:
        try:
            for window in windows:
                destroy(window)
        finally:
            modal_pump.stop()


def _run_checks(argv: list[str], tmp: Path) -> int:
    # _argument_problems has already run, so a Path here is a path that EXISTS; None means the
    # operator passed nothing, which is the only case a check may report as SKIPPED.
    data_file = Path(argv[0]).expanduser() if len(argv) > 0 else None
    model_file = Path(argv[1]).expanduser() if len(argv) > 1 else None

    # Offscreen: this must run over ssh, in CI, and from a plain terminal. The cocoa plugin is
    # proved separately by actually launching the .app.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # This harness creates/closes real MainWindows. Keep their settings and
    # recovery autosaves inside this run's disposable scratch tree, never in the
    # operator's normal macOS NativeFormat preference domain. This is test-only;
    # an ordinary application launch continues to use its existing settings.
    from PySide6 import QtCore

    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    for scope in (QtCore.QSettings.UserScope, QtCore.QSettings.SystemScope):
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, scope, str(tmp / "qsettings"))
    os.environ["NANOSCOPE_RECOVERY_ROOT"] = str(tmp / "recovery")
    os.environ["NANOSCOPE_INCUBATION_LIBRARY_ROOT"] = str(tmp / "study-library")
    # Handoff between checks (an EventSet built by one, exported by the next). A dict, not an
    # attribute on the function: @check runs each body at DECORATION time, so the function name
    # is not yet bound in this scope while its own body is executing.
    shared: dict[str, object] = {}

    @check("incubation study library survives a fresh instance")
    def _study_library():
        from nanoscope.analysis import manual_workspace as checkpoint
        from nanoscope.gui.incubation_library import StudyLibrary
        from nanoscope.gui.incubation_profiles import initialize_study_metadata
        from nanoscope.gui.manual_study_window import _empty_state

        root = tmp / "library-persistence"
        folder = tmp / "saved-library-study"
        state = initialize_study_metadata(_empty_state(), "cu", name="Saved copper study")
        checkpoint.save_workspace(state, folder)
        library = StudyLibrary(root)
        library.remember(folder, label="Copper incubation")
        restored = StudyLibrary(root)
        entries = restored.entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.available and entry.path == folder.resolve()
        assert entry.label == "Copper incubation" and entry.profile_id == "cu"
        assert Path(restored.picker_directory()).is_dir()
        return "named study folder persists; saved method and usable picker directory restored"

    # -- 1. the frozen environment itself ------------------------------------------------------
    @check("frozen bundle")
    def _frozen():
        if not getattr(sys, "frozen", False):
            raise AssertionError(
                "not running frozen — this is meant to run from inside the built bundle")
        import nanoscope

        return f"nanoscope {nanoscope.__version__}, python {sys.version.split()[0]}"

    @check("bootstrap stands down when frozen")
    def _bootstrap():
        # LANDMINE 1. bootstrap.py must NOT rewrite QT_PLUGIN_PATH inside the bundle, or it aims
        # Qt at Contents/Resources while PyInstaller aimed it at Contents/Frameworks.
        from nanoscope import bootstrap

        if not bootstrap._is_frozen():
            raise AssertionError("bootstrap does not recognise the frozen bundle")
        plugin_path = os.environ.get("QT_PLUGIN_PATH", "")
        # The question is "does QT_PLUGIN_PATH point INSIDE this bundle", and each platform
        # spells the inside differently: Contents/Frameworks in a .app, the onedir folder
        # itself (sys._MEIPASS) on Windows. Asking for the macOS spelling on Windows would
        # fail a correct build.
        root = os.path.realpath(getattr(sys, "_MEIPASS", ""))
        if not plugin_path or not os.path.realpath(plugin_path).startswith(root):
            raise AssertionError(
                f"QT_PLUGIN_PATH is not inside the bundle: {plugin_path!r} (bundle {root!r})")
        return os.path.relpath(plugin_path, root)

    @check("Qt loads + QApplication")
    def _qt():
        from PySide6 import QtCore, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        app.setApplicationName("NanoScope")
        return f"Qt {QtCore.qVersion()}"

    @check("version metadata (importlib.metadata)")
    def _meta():
        # analysis/detect.py stamps this onto every result's provenance.
        from importlib.metadata import version

        v = version("nanoscope")
        if v == "unknown":
            raise AssertionError("nanoscope dist-info missing from the bundle")
        return v

    # -- 2. readers ----------------------------------------------------------------------------
    @check("all four readers register")
    def _readers():
        # LANDMINE 6. io/base.py imports these through importlib at first use.
        from nanoscope.io import base

        base._load_readers()
        names = sorted(r.format_name for r in base.READERS)
        missing = {"Axon ABF"} - set(names)
        if missing or len(names) < 4:
            raise AssertionError(f"reader registry incomplete: {names}")
        return ", ".join(names)

    @check("h5py / npTDMS / neo / pyarrow import")
    def _libs():
        import h5py
        import neo
        import nptdms
        import pyarrow

        from neo.rawio import AxonRawIO  # noqa: F401 — the class the ABF reader actually uses

        return f"h5py {h5py.__version__}, nptdms {nptdms.__version__}, neo {neo.__version__}, pyarrow {pyarrow.__version__}"

    @check("acid CUSUM and compiled clustering runtime")
    def _acid_numerics():
        return _acid_numerics_smoke()

    @check("HDF5 write + read through the app's reader")
    def _hdf5_roundtrip():
        import h5py
        import numpy as np

        from nanoscope.io.base import open_document

        path = tmp / "synthetic.h5"
        n, fs = 200_000, 50_000.0
        rng = np.random.default_rng(0)
        sig = rng.normal(1000.0, 8.0, n)
        sig[50_000:50_400] -= 220.0  # one obvious blockade
        with h5py.File(path, "w") as fh:
            # Two things the reader insists on, both deliberate on its part: a lone /current is
            # rejected as "a data export, not a recording", and the sample rate is derived from
            # the /time base only — an attribute is not trusted. Write a real recording.
            fh.create_dataset("time", data=np.arange(n, dtype=np.float64) / fs)
            fh.create_dataset("current", data=sig)
            fh.create_dataset("voltage", data=np.full(n, -200.0))
        doc = open_document(path)
        if doc.meta.n_samples != n:
            raise AssertionError(f"read back {doc.meta.n_samples} samples, expected {n}")
        return f"{doc.meta.n_samples} samples, {len(doc.channels)} channel(s)"

    @check("pandas + pyarrow parquet round-trip")
    def _parquet():
        import pandas as pd

        df = pd.DataFrame({"dwell_s": [1e-4, 2e-4], "depth_pA": [100.0, 220.0]})
        path = tmp / "events.parquet"
        df.to_parquet(path, engine="pyarrow")
        back = pd.read_parquet(path, engine="pyarrow")
        if len(back) != 2:
            raise AssertionError("parquet round-trip lost rows")
        return f"{path.stat().st_size} bytes"

    # -- 3. analysis on a worker thread --------------------------------------------------------
    @check("detection on the QThread worker (16 MB stack / LAPACK)")
    def _worker_detect():
        # LANDMINE 7. scipy's zero-phase IIR filtering SIGBUSes on QThread's default 512 KB
        # stack; analysis_worker.py raises it to 16 MB. Drive the REAL controller, not the
        # bare function, so the thread setup is what is under test.
        import numpy as np
        from PySide6 import QtCore, QtWidgets

        from nanoscope.analysis import register_all
        from nanoscope.core.session import Session
        from nanoscope.gui.analysis_worker import AnalysisController
        from nanoscope.io.base import open_document

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        register_all()

        doc = open_document(tmp / "synthetic.h5")
        session = Session()
        doc_id = session.add_document(doc)

        controller = AnalysisController(session)
        outcome: dict = {}
        controller.finished.connect(lambda res: outcome.update(result=res))
        controller.failed.connect(lambda msg: outcome.update(error=msg))
        controller.start("detect.schmitt", doc_id, 0, {"k_in": 4.0, "k_out": 2.0})

        deadline = time.time() + 120
        while not outcome and time.time() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.02)
        controller.abort()
        if "error" in outcome:
            raise AssertionError(f"worker failed: {outcome['error']}")
        if "result" not in outcome:
            raise AssertionError("worker never finished (timeout)")

        # LANDMINE 3, part 1: disconnecting the worker is what aborts once TF is loaded.
        return f"{len(outcome['result'].payload.df)} event(s) detected"

    @check("matplotlib figure export (Agg -> PDF + PNG)")
    def _figures():
        # LANDMINE 5. MPLBACKEND=Agg and a writable MPLCONFIGDIR are set by the runtime hook.
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            raise AssertionError(f"backend is {matplotlib.get_backend()}, expected Agg")
        cfg = Path(matplotlib.get_configdir())
        if not os.access(cfg, os.W_OK):
            raise AssertionError(f"matplotlib config dir is not writable: {cfg}")

        # "The config dir is writable" was the old proof, and it is TRUE OF EVERY RUN — including
        # the year of runs where PyInstaller's own matplotlib hook handed matplotlib a throwaway
        # temp dir it deleted at exit, so the ~10 s font-cache build was paid on every launch (see
        # _restore_mpl_cache_dir in nanoscope_app.py). Assert the thing this check claims instead:
        # the font list must be PERSISTED, and persisted in the directory the app chose.
        import matplotlib.font_manager  # noqa: F401 - importing it is what builds/loads the cache

        cache = Path(matplotlib.get_cachedir())
        fontlists = sorted(cache.glob("fontlist-v*.json"))
        if not fontlists:
            raise AssertionError(
                f"matplotlib built its font list but persisted nothing in {cache} — every launch "
                "will pay the ~10 s rebuild")
        wanted = os.environ.get("NANOSCOPE_MPLCONFIGDIR")
        if wanted and Path(wanted).resolve() != cache.resolve():
            raise AssertionError(
                f"matplotlib is caching in {cache}, not the persistent cache directory the "
                f"runtime hook chose ({wanted}); that directory is thrown away at exit, so the "
                "font cache is rebuilt on every launch")
        # Honest about which run this is: on a genuine first launch the file is written NOW and
        # the 10 s is the cost the packaging promises to pay once. Only a later run can prove it
        # was reused, so report which one happened rather than implying the fast path.
        fontlist_age = _STARTED_AT - fontlists[0].stat().st_mtime
        cache_note = ("font cache BUILT this run (first launch)" if fontlist_age < 0
                      else f"font cache reused ({fontlist_age / 60:.0f} min old)")

        import numpy as np

        from nanoscope.analysis.detect import detect_events
        from nanoscope.analysis.events import EventSet
        from nanoscope.plot import figures

        # A real detection, so the EventSet carries every column the figures require rather than
        # a hand-rolled frame that drifts out of sync with MEASUREMENT_SCHEMA.
        rng = np.random.default_rng(1)
        fs = 50_000.0
        sig = rng.normal(1000.0, 8.0, 400_000)
        for start in range(20_000, 400_000, 20_000):
            sig[start:start + 200] -= 200.0
        df, prov = detect_events(sig, fs, k_in=4.0, k_out=2.0)
        if df.empty:
            raise AssertionError("synthetic trace produced no events to plot")
        fig, _ = figures.dwell_histogram(EventSet(df, prov), fit=False)
        pdf, png = tmp / "fig.pdf", tmp / "fig.png"
        fig.savefig(pdf)
        fig.savefig(png)
        if pdf.stat().st_size < 1000 or png.stat().st_size < 1000:
            raise AssertionError("figures wrote suspiciously small files")
        return (f"backend={matplotlib.get_backend()}, pdf={pdf.stat().st_size}B, "
                f"png={png.stat().st_size}B, {cache_note} in {cache}")

    # -- 4. TensorFlow -------------------------------------------------------------------------
    @check("TensorFlow imports with GPUs hidden")
    def _tf_import():
        # LANDMINE 2. The Metal pluggable device SIGABRTs on this mixed_float16 LSTM. The bundle
        # ships no libmetal_plugin.dylib at all, and _import_tf hides GPUs before the first op.
        from nanoscope.analysis import ml_detect

        if not ml_detect.TENSORFLOW_AVAILABLE:
            raise AssertionError("TENSORFLOW_AVAILABLE is False inside the bundle")
        tf = ml_detect._import_tf()
        gpus = tf.config.get_visible_devices("GPU")
        if gpus:
            raise AssertionError(f"GPUs are visible to TF: {gpus}")
        # No TF pluggable device may ship, whatever this platform calls a shared library.
        plugin = Path(sys._MEIPASS) / "tensorflow-plugins"
        _libs = ("*.dylib", "*.so", "*.dll")
        if plugin.exists() and any(any(plugin.glob(pat)) for pat in _libs):
            raise AssertionError(f"a TF pluggable device shipped in the bundle: {list(plugin.iterdir())}")
        return f"tf {tf.__version__}, visible GPUs: none"

    @check("google3 protobuf stub survives freezing")
    def _google3():
        # LANDMINE 3. Without the stub packages, the next QObject.disconnect() after TF is loaded
        # makes shiboken abort the process (exit 134). Prove the stubs exist AND that a real
        # disconnect completes.
        import sys as _sys

        from PySide6 import QtCore, QtWidgets

        leaves = [m for m in _sys.modules if m.startswith("google3.")]
        broken = [m for m in leaves for i in range(1, len(m.split(".")))
                  if _sys.modules.get(".".join(m.split(".")[:i])) is None]
        if broken:
            raise AssertionError(f"google3 ancestor packages missing: {sorted(set(broken))}")

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        obj = QtCore.QObject()
        timer = QtCore.QTimer(obj)
        timer.timeout.connect(obj.deleteLater)
        timer.timeout.disconnect(obj.deleteLater)  # this is the call that used to abort
        return f"{len(leaves)} google3 leaf module(s), disconnect() OK"

    @check("Keras model load + predict")
    def _keras():
        import numpy as np

        from nanoscope.analysis import ml_detect

        if model_file is None:  # not passed at all; a passed-but-missing path stopped the run
            return "SKIPPED (no model passed)"
        wrapper = ml_detect.load_keras_model(model_file)
        fs = 50_000.0
        sig = np.random.default_rng(2).normal(1000.0, 8.0, wrapper.window * 6).astype(np.float64)
        sig[wrapper.window * 2: wrapper.window * 2 + 300] -= 250.0
        regions, prob, prov = ml_detect.ml_detect_regions(sig, fs, wrapper)
        if prob.shape[0] != sig.shape[0]:
            raise AssertionError(f"probability length {prob.shape} != signal {sig.shape}")
        return f"W={wrapper.window} C={wrapper.channels}, {len(regions)} region(s) on synthetic noise"

    @check("ML detection through the GUI controller")
    def _ml_via_controller():
        # LANDMINE 3, the real reproduction. The controller disconnects its worker at the END of
        # every run; once TensorFlow is loaded that disconnect is what used to abort the process
        # with SIGABRT (exit 134). Nothing short of running an ML detection through
        # AnalysisController and letting it tear down proves the google3 stub survived freezing.
        import numpy as np
        from PySide6 import QtCore, QtWidgets

        from nanoscope.analysis import register_all
        from nanoscope.core.session import Session
        from nanoscope.core.types import ChannelKind
        from nanoscope.gui.analysis_worker import AnalysisController
        from nanoscope.io.base import open_document

        if model_file is None:  # not passed at all; a passed-but-missing path stopped the run
            return "SKIPPED (no model passed)"

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        register_all()

        # Prefer REAL signal. A model that fires once on Gaussian noise has proved only that
        # TensorFlow executed; running the owner's own network over the owner's own recording is
        # what shows the 716 MB of TensorFlow in this bundle does the job it was included for.
        # A bounded slice, not the whole file: this is a smoke test on a CPU, and the point is
        # that the graph runs on real data, not to time a full pass.
        source = "synthetic"
        # The guard at the bottom of this check accuses the MODEL of finding nothing in REAL
        # signal. Only a slice actually built from data_file may arm it — it used to be armed by
        # `source != "synthetic"`, which is True for the fallback string
        # "synthetic (real slice unavailable: ...)" too, so a reader refusal (a CPYA-variant ABF,
        # say) could end with the self-test blaming the owner's recording for a run that never
        # touched it and pointing them away from the refusal that actually happened.
        used_real = False
        ml_path = tmp / "synthetic.h5"
        if data_file is not None:
            try:
                import h5py

                real = open_document(data_file)
                idx = next((i for i, c in enumerate(real.meta.channels)
                            if c.kind is ChannelKind.CURRENT), 0)
                fs_real = float(real.meta.sample_rate_hz or 50_000.0)
                data = np.asarray(real.channels[idx][:500_000], dtype=np.float64)
                if data.size >= 20_000:
                    ml_path = tmp / "real_slice.h5"
                    with h5py.File(ml_path, "w") as fh:
                        fh.create_dataset(
                            "time", data=np.arange(data.size, dtype=np.float64) / fs_real)
                        fh.create_dataset("current", data=data)
                    source = (f"{data.size} real samples from {data_file.name} "
                              f"(ch{idx} {real.meta.channels[idx].name})")
                    used_real = True
                else:
                    # Say why the file the operator passed was not used, rather than reporting a
                    # synthetic run as though nothing had been asked for.
                    source = (f"synthetic (real slice too short: {data.size} samples from "
                              f"{data_file.name}, need 20000)")
            except Exception as exc:  # noqa: BLE001 - fall back rather than fail the ML check
                source = f"synthetic (real slice unavailable: {type(exc).__name__}: {exc})"

        session = Session()
        doc_id = session.add_document(open_document(ml_path))
        controller = AnalysisController(session)
        outcome: dict = {}
        controller.finished.connect(lambda res: outcome.update(result=res))
        controller.failed.connect(lambda msg: outcome.update(error=msg))
        controller.start("detect.keras", doc_id, 0, {"model_path": str(model_file)})

        deadline = time.time() + 300
        while not outcome and time.time() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.02)
        controller.abort()  # <- the disconnect that used to kill the process
        QtCore.QCoreApplication.processEvents()

        if "error" in outcome:
            raise AssertionError(f"ML worker failed: {outcome['error']}")
        if "result" not in outcome:
            raise AssertionError("ML worker never finished (timeout)")
        n_ml = len(outcome["result"].payload.df)
        if used_real and n_ml == 0:
            raise AssertionError(
                f"the ML detector found NOTHING in {source} — TensorFlow ran but the model "
                "produced no events on real signal")
        return f"{n_ml} event(s) on {source}; process survived teardown"

    @check("full MainWindow builds + opens a document")
    def _main_window():
        # Every panel, every registration, every icon and stylesheet the app touches at startup —
        # constructed for real. A missing hiddenimport in any panel shows up here as an ImportError
        # instead of as a blank window on a labmate's Mac.
        from PySide6 import QtCore, QtWidgets

        from nanoscope.gui.app import configure_pyqtgraph, create_app
        from nanoscope.gui.main_window import MainWindow
        from nanoscope.gui.overlays import register_all_renderers

        create_app([])
        configure_pyqtgraph()
        register_all_renderers()

        # A modal dialog inside a headless run is a deadlock: QDialog.exec() spins its own event
        # loop waiting for a click that can never come, and the whole self-test hangs forever
        # (observed — the close below raises the app's unsaved-work prompt once a document is
        # loaded). This dismisser answers anything modal that appears and records what it was, so
        # the check reports the prompt instead of hanging on it.
        dismissed: list[str] = []

        def answer_modals() -> None:
            w = QtWidgets.QApplication.activeModalWidget()
            if w is None:
                return
            dismissed.append(w.windowTitle() or type(w).__name__)
            w.reject()

        pump = QtCore.QTimer()
        pump.timeout.connect(answer_modals)
        pump.start(200)
        try:
            win = MainWindow()
            win.show()
            QtCore.QCoreApplication.processEvents()
            docks = win.findChildren(QtWidgets.QDockWidget)
            menus = [m.title() for m in win.menuBar().findChildren(QtWidgets.QMenu)]
            if not docks or not menus:
                raise AssertionError(f"window built empty: {len(docks)} docks, menus={menus}")

            # Exercise the actual lazy menu route: a main-window import alone does
            # not prove that the bundled acid workspace can be constructed.
            win._act_incubation_study.trigger()
            launcher = win._incubation_study_window
            assert launcher is not None
            acid = launcher.create_study(tmp / "acid-editor-study", profile="acid")
            QtCore.QCoreApplication.processEvents()
            acid = win._acid_study_window
            assert acid is not None and acid._stages.count() == 6, (
                "Acid incubation menu did not construct all six workflow stages")
            controls = {button.text() for button in acid.findChildren(QtWidgets.QPushButton)}
            for label in ("Accept selected events", "Reject selected events",
                          "Build event families for review", "Style selected figure",
                          "Curate in main workspace"):
                assert any(label in text for text in controls), (
                    f"Bundled acid workspace is missing {label!r}")
            # Run the actual shared-editor handoff and checkpoint return inside
            # the frozen app, including the lazy Arrow/bridge imports.
            import hashlib
            import numpy as np
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            from nanoscope.analysis import acid_incubation as acid_engine
            from nanoscope.analysis import manual_workspace as acid_checkpoint

            raw_path = tmp / "acid-editor-source.abf"
            raw_path.write_bytes(b"frozen-app synthetic prepared source identity")
            signal = np.random.default_rng(91).normal(0, .2, 5000).astype(np.float32)
            signal[1000:1040] -= 10
            signal[2000:2060] -= 12
            parquet = tmp / "acid-editor-prepared.parquet"
            pq.write_table(pa.table(dict(current_detr_pA=signal, current_raw_pA=signal-200)), parquet)
            source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            acid.state = acid_engine.new_state("Shared editor check", [dict(tag="acid_test",
                source_path=raw_path, source_info=dict(sha256=source_hash, fs_hz=50000.,
                    channels=[dict(units="pA")]), incubation_h=2., channel=0)])
            row = acid.state['records'][0]
            row['prepared'] = dict(parquet_path=parquet,
                sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(), source_sha256=source_hash,
                settings_sha256=acid_engine._digest({k: acid.state['settings'][k] for k in acid_engine.PREPARE_KEYS}),
                metadata=dict(fs_hz=50000., n_samples=5000, event_polarity=1))
            row['detection'] = dict(primary_set='cusum', catalog_detector='cusum', settings_sha256='smoke')
            row['catalog'] = pd.DataFrame(dict(event_id=['C000000', 'C000001'],
                start_idx=[1000, 2000], end_idx=[1039, 2059], accepted=[True, False],
                passes_gate=[True, False], automatic_accepted=[True, False], event_review=['unreviewed']*2))
            acid.refresh()
            errors = []
            acid.operationFailed.connect(errors.append)

            def complete_acid(action):
                assert action() is not False
                deadline = time.time() + 60
                while acid.is_busy() and time.time() < deadline:
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(.01)
                assert not acid.is_busy() and not errors, errors

            complete_acid(acid.open_main_curation)
            result = win._analysis_panel.current_detection()
            assert win._event_gallery.frame() is result.payload.df
            assert win._study_curation_bar.isVisible()
            event_id = int(result.payload.df.event_id.iloc[0])
            win._analysis_panel.set_event_accepted(event_id, False)
            complete_acid(lambda: acid.save_main_curation(win._study_curation_context))
            saved = acid_checkpoint.load_workspace(acid.folder)
            assert saved['records'][0]['catalog'].accepted.tolist() == [False, False]
            assert saved['records'][0]['catalog'].event_review.tolist() == ['reviewed', 'unreviewed']
            acid.close()
            QtCore.QCoreApplication.processEvents()

            # New profiles and both legacy/general routing modules must survive
            # freezing, and the same native Save button must commit Zn/Cu edits.
            from dataclasses import asdict
            from nanoscope.gui.incubation_profiles import peek_workspace, recording_defaults
            from nanoscope.gui.manual_study_window import _prep_defaults
            from nanoscope.analysis import manual_preparation as manual_prep
            from nanoscope.analysis import manual_detection as manual_detect
            from nanoscope.analysis.incubation_detection import RecordingAnalysis, _fingerprint
            for profile in ('general', 'zn', 'cu'):
                manual = launcher.create_study(tmp / (profile+'-editor-study'), profile=profile)
                assert peek_workspace(manual.folder).profile_id == profile
                tag = profile+'_test'
                raw = signal.astype(np.float64)-200
                metadata = dict(event_polarity=1, qc_mad_noise_pA=.2)
                prepared = manual_prep.RawPreparedRecording(tag, raw, signal.copy(), 50000., .1,
                    metadata, dict(raw_file=_fingerprint(raw_path), event_polarity=1),
                    raw.copy(), np.full(5000, -200., np.float32))
                catalog = pd.DataFrame(dict(event_id=[tag+':0', tag+':1'],
                    start_idx=[1000,2000], end_idx=[1039,2059], accepted=[True,False],
                    passes_gate=[True,True], depth_peak_pA=[10.,12.], I0_local_pA=[200.,200.]))
                manual.state['records'] = [dict(recording_id=profile+'-record', tag=tag,
                    source_path=raw_path, source_info=dict(channels=[dict(index=0,name='Current',units='pA')], fs_hz=50000., n_samples=5000),
                    incubation_h=2., pore_id='test-pore', pore_diameter_nm=18., concentration_pm=100.,
                    voltage_mv=-200., channel=0, event_polarity=1, reader_scaling='notebook_float32',
                    preparation_settings=_prep_defaults(), inference_settings=asdict(manual_detect.InferenceSettings()),
                    decode_settings=asdict(manual_detect.DecodeSettings()),
                    measurement_settings=asdict(manual_detect.MeasurementSettings()), prepared=prepared,
                    analysis=RecordingAnalysis(tag, raw, signal.copy(), 50000., .1, catalog, dict(prepared.provenance)),
                    **recording_defaults(profile))]
                manual.refresh()
                failures = []; manual.operationFailed.connect(failures.append)
                def complete_manual(action):
                    assert action() is not False
                    deadline = time.time()+60
                    while manual.is_busy() and time.time()<deadline:
                        QtCore.QCoreApplication.processEvents(); time.sleep(.01)
                    assert not manual.is_busy() and not failures, failures
                complete_manual(manual.open_main_analysis)
                win._analysis_panel.set_event_accepted(0, False)
                complete_manual(win._save_study_curation)
                restored = acid_checkpoint.load_workspace(manual.folder)
                assert restored['study_profile'] == profile
                assert restored['records'][0]['analysis'].catalog.accepted.tolist() == [False,False]
                assert restored['model_path'] == ''
                win._return_to_study()
                assert manual._stages.currentIndex() == 2
                manual.close()

            # The sink the macOS open-document relay in nanoscope_app.py feeds. Info.plist
            # advertises .abf/.h5/.tdms/.csv/.txt as document types, so this has to actually load
            # a file — an app that claims a file type and then opens empty is worse than one that
            # claims nothing at all.
            win.open_paths([str(tmp / "synthetic.h5")])
            deadline = time.time() + 60
            loaded = []
            while time.time() < deadline:
                loaded = [doc for doc in win._session.documents()
                          if doc.meta.path.resolve() == (tmp / "synthetic.h5").resolve()]
                if loaded: break
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.02)
            if not loaded:
                raise AssertionError("open_paths() never produced a document — a macOS "
                                     "double-click would open an empty window")
            win.close()
            QtCore.QCoreApplication.processEvents()
        finally:
            pump.stop()

        note = f", dismissed {dismissed}" if dismissed else ""
        return (f"{len(docks)} dock(s), {len(menus)} menus, opened "
                f"{loaded[0].meta.n_samples} samples via open_paths(); "
                f"unified launcher opens General, Zn, Cu and Acid profiles; "
                f"shared editor saves decisions back to the acid checkpoint; General/Zn/Cu curation checkpoints round-trip without a model{note}")

    @check("Project save + close + restore with nested visible segments")
    def _restore_project():
        return _project_restore_smoke(tmp)

    # -- 5. the standalone analysis surfaces ---------------------------------------------------
    #
    # Building MainWindow proves these windows IMPORT. It does not prove they COMPUTE: they are
    # created lazily, on the menu action, and each one reaches code nothing above it touches --
    # pyqtgraph's GraphicsScene and ImageItem, scipy.signal.welch, the 2-D density histogram, the
    # ensemble aligner. A bundle can pass every check above and still put up three empty windows.

    @check("Populations / density / PSD windows compute + draw")
    def _analysis_windows():
        import h5py
        import numpy as np
        from PySide6 import QtCore, QtWidgets

        from nanoscope.analysis import register_all
        from nanoscope.core.session import Session
        from nanoscope.core.sources import discover_plot_sources
        from nanoscope.gui.analysis_worker import AnalysisController
        from nanoscope.gui.app import configure_pyqtgraph
        from nanoscope.gui.event_plot_panel import EventPlotWindow
        from nanoscope.gui.plots_panel import PlotsWindow
        from nanoscope.gui.psd_panel import PSDWindow
        from nanoscope.io.base import open_document

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_pyqtgraph()
        register_all()

        # A denser trace than synthetic.h5: the density heatmap and the ensemble average are
        # only meaningful with a few dozen events, and a 1-event set would let a broken
        # binning path pass.
        path = tmp / "populations.h5"
        n, fs = 600_000, 50_000.0
        rng = np.random.default_rng(3)
        sig = rng.normal(1000.0, 8.0, n)
        for start in range(5_000, n - 5_000, 10_000):
            sig[start:start + 150] -= rng.normal(210.0, 15.0)
        with h5py.File(path, "w") as fh:
            fh.create_dataset("time", data=np.arange(n, dtype=np.float64) / fs)
            fh.create_dataset("current", data=sig)
            fh.create_dataset("voltage", data=np.full(n, -200.0))

        session = Session()
        doc_id = session.add_document(open_document(path))
        ch_index = 0

        # Through the real controller, then stored the way MainWindow stores it — because
        # discover_plot_sources() reads session.results, and that lookup is what the Plots
        # window is actually fed from.
        controller = AnalysisController(session)
        outcome: dict = {}
        controller.finished.connect(lambda res: outcome.update(result=res))
        controller.failed.connect(lambda msg: outcome.update(error=msg))
        controller.start("detect.schmitt", doc_id, ch_index, {"k_in": 4.0, "k_out": 2.0})
        deadline = time.time() + 120
        while not outcome and time.time() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.02)
        controller.abort()
        if "error" in outcome:
            raise AssertionError(f"detection for the analysis windows failed: {outcome['error']}")
        if "result" not in outcome:
            raise AssertionError("detection for the analysis windows never finished (timeout)")
        result = outcome["result"]
        session.add_result(result)
        events = result.payload
        n_events = len(events.df)
        if n_events < 10:
            raise AssertionError(f"only {n_events} events — too few to exercise the windows")

        # -- Populations (histogram + scatter + a fit) ---------------------------------------
        sources = discover_plot_sources(
            session.documents(), session.results.get,
            current=(doc_id, ch_index),
            calibration_fingerprint=lambda d, c: session.calibration(d, c).fingerprint(),
        )
        if not sources:
            raise AssertionError("discover_plot_sources() found nothing to plot")
        plots = PlotsWindow(session)
        plots.set_data(sources)
        plots.show()
        QtCore.QCoreApplication.processEvents()
        if plots._hist_item is None:
            raise AssertionError("Populations: histogram drew no bars")
        plots_png = tmp / "populations.png"
        if plots.save_image(str(plots_png)) is None or plots_png.stat().st_size < 1000:
            raise AssertionError("Populations: save_image wrote nothing usable")

        # -- density / ensemble ---------------------------------------------------------------
        evw = EventPlotWindow(session)
        evw.set_source(doc_id, ch_index, events)
        evw.show()
        QtCore.QCoreApplication.processEvents()
        if evw._density_counts is None or evw._density_counts.sum() == 0:
            raise AssertionError("Density: the 2-D histogram is empty")
        if evw._ensemble is None:
            raise AssertionError("Ensemble: no aligned event average was computed")
        density_png = tmp / "density.png"
        if evw.save_image(str(density_png)) is None or density_png.stat().st_size < 1000:
            raise AssertionError("Density: save_image wrote nothing usable")

        # -- PSD (scipy.signal.welch over the calibrated current) -----------------------------
        psd = PSDWindow(session)
        psd.set_target(doc_id, ch_index)
        psd.show()
        QtCore.QCoreApplication.processEvents()
        summary = psd.summary()
        if summary is None or psd._psd is None or not np.isfinite(psd._psd).any():
            raise AssertionError("PSD: no finite spectrum was computed")
        psd_png = tmp / "psd.png"
        if psd.save_image(str(psd_png)) is None or psd_png.stat().st_size < 1000:
            raise AssertionError("PSD: save_image wrote nothing usable")

        for w in (plots, evw, psd):
            w.close()
        QtCore.QCoreApplication.processEvents()

        shared["events"] = events   # handed to the export check below
        rms = summary.get("total_rms")
        if rms is None or not np.isfinite(rms) or rms <= 0.0:
            raise AssertionError(f"PSD: total_rms is not a usable number ({rms!r})")
        return (f"{n_events} events; histogram + {int(evw._density_counts.sum())}-count density "
                f"+ PSD ({len(psd._psd)} bins, total_rms={rms:.3g}); 3 PNGs written")

    @check("Comparison figures: Seaborn + frozen snapshots + PNG/PDF/SVG")
    def _comparisons():
        return _comparison_figures_smoke(tmp)

    @check("figure + events CSV export (the paths the owner ships from)")
    def _exports():
        # save_figure() is the app's real export: vector PDF, 300-dpi PNG *and* the numbers
        # behind the figure as a sibling CSV. The plain matplotlib check above calls savefig
        # directly and so never touches the sidecar logic. The events CSV is the other export
        # in the menu (MainWindow._export_events) — accepted-only, which is the curation
        # contract, and the file a labmate actually loads into their own analysis.
        import pandas as pd

        from nanoscope.analysis.events import EventSet
        from nanoscope.plot import figures

        events = shared.get("events")
        if events is None:
            raise AssertionError("the analysis-windows check did not produce an EventSet")

        fig, _ = figures.scatter_dwell_depth(events.accepted)
        out = figures.save_figure(fig, tmp / "export_scatter.pdf", data=events.accepted)
        for kind in ("pdf", "png", "csv"):
            if kind not in out:
                raise AssertionError(f"save_figure did not write a {kind} (got {sorted(out)})")
            if out[kind].stat().st_size < 200:
                raise AssertionError(f"save_figure wrote a suspiciously small {kind}")

        # The events CSV, exactly as MainWindow._export_events writes it.
        accepted = events.accepted.df
        csv = tmp / "events_export.csv"
        accepted.to_csv(csv, index=False)
        back = pd.read_csv(csv)
        if len(back) != len(accepted) or "dwell_s" not in back.columns:
            raise AssertionError(
                f"events CSV round-trip lost data: {len(back)}/{len(accepted)} rows, "
                f"columns={list(back.columns)[:8]}")
        if not isinstance(events, EventSet):
            raise AssertionError(f"payload is {type(events).__name__}, not an EventSet")

        return (f"figure pdf+png+csv ({out['csv'].stat().st_size}B of source data), "
                f"events CSV {len(back)} rows x {len(back.columns)} cols")

    # -- 6. a real recording, if one was supplied ----------------------------------------------
    @check("real recording end to end")
    def _real_file():
        if data_file is None:  # not passed at all; a passed-but-missing path stopped the run
            return "SKIPPED (no file passed)"
        import numpy as np

        from nanoscope.analysis.detect import detect_events
        from nanoscope.core.session import Session
        from nanoscope.io.base import open_document, select_reader

        from nanoscope.core.types import ChannelKind

        reader = select_reader(data_file)
        doc = open_document(data_file)
        session = Session()
        doc_id = session.add_document(doc)
        # Detection only means anything on the CURRENT channel; channel 0 of a real ABF is
        # typically the command-voltage trace, which has no measurable noise to threshold on.
        idx = next((i for i, c in enumerate(doc.meta.channels) if c.kind is ChannelKind.CURRENT), 0)
        ch = session.calibrated(doc_id, idx)
        data = np.asarray(ch.data[: min(len(ch.data), 2_000_000)], dtype=float)
        fs = doc.meta.sample_rate_hz or 50_000.0
        df, prov = detect_events(data, fs, k_in=5.0, k_out=2.0)
        return (f"{reader.format_name}: {doc.meta.n_samples} samples, {len(doc.channels)} ch, "
                f"fs={fs:g} Hz, ch{idx}={doc.meta.channels[idx].name} -> {len(df)} events "
                f"in the first {len(data)} samples")

    # -- report ---------------------------------------------------------------------------------
    failed = [r for r in _results if not r[1]]
    print("\n" + "=" * 78)
    print("NanoScope bundle self-test")
    print("=" * 78)
    for name, ok, detail in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
    print("=" * 78)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    return len(failed)
