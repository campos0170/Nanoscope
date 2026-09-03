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
        try:
            detail = fn() or ""
            _results.append((name, True, f"{detail} [{time.time() - t0:.1f}s]"))
        except Exception as exc:  # noqa: BLE001 - a failed check is data, not a crash
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
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


def _run_checks(argv: list[str], tmp: Path) -> int:
    # _argument_problems has already run, so a Path here is a path that EXISTS; None means the
    # operator passed nothing, which is the only case a check may report as SKIPPED.
    data_file = Path(argv[0]).expanduser() if len(argv) > 0 else None
    model_file = Path(argv[1]).expanduser() if len(argv) > 1 else None

    # Offscreen: this must run over ssh, in CI, and from a plain terminal. The cocoa plugin is
    # proved separately by actually launching the .app.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Handoff between checks (an EventSet built by one, exported by the next). A dict, not an
    # attribute on the function: @check runs each body at DECORATION time, so the function name
    # is not yet bound in this scope while its own body is executing.
    shared: dict[str, object] = {}

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

            # The sink the macOS open-document relay in nanoscope_app.py feeds. Info.plist
            # advertises .abf/.h5/.tdms/.csv/.txt as document types, so this has to actually load
            # a file — an app that claims a file type and then opens empty is worse than one that
            # claims nothing at all.
            win.open_paths([str(tmp / "synthetic.h5")])
            deadline = time.time() + 60
            while not win._session.documents() and time.time() < deadline:
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.02)
            loaded = win._session.documents()
            if not loaded:
                raise AssertionError("open_paths() never produced a document — a macOS "
                                     "double-click would open an empty window")
            win.close()
            QtCore.QCoreApplication.processEvents()
        finally:
            pump.stop()

        note = f", dismissed {dismissed}" if dismissed else ""
        return (f"{len(docks)} dock(s), {len(menus)} menus, opened "
                f"{loaded[0].meta.n_samples} samples via open_paths(){note}")

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
