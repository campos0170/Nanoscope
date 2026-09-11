# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for NanoScope.app (Apple Silicon, unsigned + ad-hoc signed).

Run it through packaging/build_app.sh, not by hand — that script generates the icon, cleans stale
output and does the ad-hoc signing + DMG afterwards.

WHY PyInstaller: it is the only one of the three realistic options (PyInstaller / py2app / Nuitka)
with maintained first-party hooks for *every* landmine in this dependency set at once — PySide6's
split framework layout, TensorFlow's lazy `_api` module tree, scipy/sklearn's compiled extensions
and h5py's plugins. py2app has no TensorFlow story; Nuitka would compile for 40+ minutes per
iteration and still need the same hook knowledge.

Layout is ONEDIR inside a .app (never onefile): the payload is >1 GB and a onefile build would
unpack all of it to a temp directory on every single launch.

The four things this spec is really doing:

  1. EXCLUDES. PySide6 ships 1.2 GB of Qt, of which this app imports QtCore/QtGui/QtWidgets. The
     module-graph excludes stop PyInstaller pulling the rest in; `_prune_toc` then removes what
     the hooks collect wholesale anyway (qml/, translations, unused plugin families). Biggest
     single win by far: QtWebEngineCore is 589 MB on its own.
  2. HIDDENIMPORTS. `nanoscope/io/base.py::_load_readers` imports the four reader modules through
     `importlib.import_module`, which the module graph cannot see. Missing them means the app
     starts fine and then claims it has no reader for .abf.
  3. TENSORFLOW, CPU-ONLY. tensorflow-metal's libmetal_plugin.dylib is deliberately NOT bundled —
     the Metal pluggable device SIGABRTs on this model's mixed_float16 LSTM. ml_detect already
     hides GPUs, and leaving the plugin out means there is nothing to hide.
  4. INFO.PLIST. Retina (NSHighResolutionCapable), dark-mode-aware, document types for the four
     formats the readers claim, and LSMinimumSystemVersion 15.0 — NOT the arm64 floor but the
     deployment target of the bundled PySide6/shiboken6 binaries, asserted post-build.
"""
import sys
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.osx import BUNDLE
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# SPECPATH is injected by PyInstaller; it is packaging/.
HERE = Path(SPECPATH).resolve()  # noqa: F821
ROOT = HERE.parent

APP_NAME = "NanoScope"
BUNDLE_ID = "org.nanoscope.NanoScope"

# MUST run before Analysis(): otherwise PyInstaller asks Anaconda's qt.conf where Qt lives and
# ships conda's Qt5 plugins inside this Qt6 app, which fails at launch. See qt_prefix_fix.py.
sys.path.insert(0, str(HERE))
from build_identity import write_build_record  # noqa: E402

BUILD_RECORD, BUILD_RECORD_PATH = write_build_record(ROOT)
APP_VERSION = BUILD_RECORD["version"]
from qt_prefix_fix import force_pyside6_qt_prefix  # noqa: E402

force_pyside6_qt_prefix()


# --------------------------------------------------------------------------------------------
# Hidden imports
# --------------------------------------------------------------------------------------------
hiddenimports = [
    # nanoscope.io.base imports these by name at first use — invisible to the module graph.
    "nanoscope.io.abf",
    "nanoscope.io.hdf5",
    "nanoscope.io.tdms",
    "nanoscope.io.text",
    # neo's ABF path. Imported statically by io/abf.py, but neo.rawio resolves its IO classes
    # through a registry, so name the one we actually use.
    "neo.rawio.axonrawio",
    # pandas' Parquet/Arrow engine is chosen at call time from a string.
    "pyarrow",
    "pyarrow.parquet",
    "pandas.io.parquet",
    # matplotlib file-export backends (MPLBACKEND=Agg is forced by the runtime hook; the PDF and
    # SVG writers are separate modules that pyplot loads on demand from savefig's suffix).
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
    # pyqtgraph's exporters are imported from the package root by five GUI panels.
    "pyqtgraph.exporters",
    "pyqtgraph.exporters.ImageExporter",
    # scipy pieces reached through lazy/dispatch layers rather than plain imports.
    "scipy.special.cython_special",
    "scipy._lib.array_api_compat.numpy.fft",
    # `NanoScope --self-test`; imported inside a function so the GUI path never pays for it.
    "nanoscope_selftest",
]

# Keras 3 picks its backend at import time from a config file / env var and imports it by string.
# No PyInstaller hook ships for Keras 3, so collect it whole; it is only 13 MB.
hiddenimports += collect_submodules("keras")

datas = [
    (str(BUILD_RECORD_PATH), "nanoscope"),
    # importlib.metadata.version("nanoscope") in analysis/detect.py stamps the app version onto
    # every detection result's provenance. Without the dist-info it silently degrades to
    # "unknown" and every exported CSV loses its version stamp.
    *copy_metadata("nanoscope"),
    # Keras' saving layer reads JSON schemas + the serialization registry from package data.
    *collect_data_files("keras"),
]

binaries = []


# --------------------------------------------------------------------------------------------
# Excludes — module graph
# --------------------------------------------------------------------------------------------
# Qt modules this app never touches. Verified against the source: the only PySide6 imports in
# nanoscope/ are QtCore, QtGui and QtWidgets. pyqtgraph probes for several of the others at
# import time inside try/except, which is exactly how they sneak into a build.
_QT_EXCLUDES = [
    f"PySide6.{m}"
    for m in (
        "Qt3DAnimation", "Qt3DCore", "Qt3DExtras", "Qt3DInput", "Qt3DLogic", "Qt3DRender",
        "QtBluetooth", "QtCharts", "QtDataVisualization", "QtDesigner", "QtGraphs",
        "QtGraphsWidgets", "QtHelp", "QtHttpServer", "QtLocation", "QtMultimedia",
        "QtMultimediaWidgets", "QtNfc", "QtPdf", "QtPdfWidgets", "QtPositioning", "QtQml",
        "QtQuick", "QtQuick3D", "QtQuickControls2", "QtQuickTest", "QtQuickWidgets",
        "QtRemoteObjects", "QtScxml", "QtSensors", "QtSerialBus", "QtSerialPort",
        "QtSpatialAudio", "QtSql", "QtStateMachine", "QtTest", "QtTextToSpeech", "QtUiTools",
        "QtWebChannel", "QtWebEngineCore", "QtWebEngineQuick", "QtWebEngineWidgets",
        "QtWebSockets",
    )
] + ["PySide6.scripts", "PySide6.support", "shiboken6_generator"]

# Two import chains found by inspecting the first build's cross-reference. Both are *optional*
# code paths that this app never executes, and both were dragging in hundreds of MB:
#
#   pyarrow -> pyarrow.fs -> fsspec -> fsspec.gui -> panel -> holoviews/hvplot/bokeh/vtk/intake
#       fsspec.gui is a Panel-based file browser widget. `import panel` reaches VTK (325 MB) and
#       the entire HoloViz stack. Nothing imports fsspec.gui; the module graph just walks into it.
#
#   neo.rawio -> neo.rawio.spikeglxrawio -> probeinterface -> zarr -> numcodecs
#       nanoscope only uses neo.rawio.AxonRawIO, and spikeglxrawio imports probeinterface INSIDE
#       a method (line ~309), not at module scope — verified, so cutting it cannot break the
#       import of neo.rawio.
_LEAF_EXCLUDES = [
    "fsspec.gui", "panel", "holoviews", "hvplot", "bokeh", "datashader", "param", "pyviz_comms",
    "intake", "altair", "vtk", "vtkmodules", "anaconda_catalogs", "colorcet",
    "probeinterface", "zarr", "numcodecs",
    # Round two, same method (xref -> shortest import path):
    #   keras.src.saving.saving_lib -> huggingface_hub -> pydantic -> mypy, and -> hf_xet
    #       Keras 3's "load a model from the Hugging Face Hub" path. nanoscope loads .keras files
    #       from local paths only. saving_lib wraps the import in try/except and sets the module
    #       to None (line ~41), so removing it degrades exactly the feature we do not use. 28 MB.
    #   pandas.io.xml -> lxml
    #       pandas' optional XML reader; it resolves lxml through import_optional_dependency at
    #       call time, never at import. The app reads ABF/HDF5/TDMS/CSV/parquet. 8 MB.
    "huggingface_hub", "pydantic", "mypy", "hf_xet", "lxml",
    #   tensorflow -> tensorflow.core.debug.debug_service_pb2_grpc -> grpc
    #       TF's remote debug service. `import tensorflow` does not touch it (checked: after a
    #       real import, no grpc module is in sys.modules); it arrives only because
    #       hook-tensorflow sweeps up submodules. 26 MB with libgrpc. Self-test verified.
    "grpc", "grpc_status",
]
# TRIED AND REVERTED — do not "optimize" this back in:
#   "tensorflow.lite" (17 MB). `import tensorflow` pulls tensorflow.lite in EAGERLY, so excluding
#   it makes the whole of TensorFlow unimportable and the ML detector silently degrades to
#   "TensorFlow is not available". The bundle still builds, signs and launches — only ML breaks.
#   This is exactly the failure `--self-test` exists to catch, and it did.

excludes = _QT_EXCLUDES + _LEAF_EXCLUDES + [
    # Other Qt bindings. Anaconda has PyQt5 installed and pyqtgraph will happily bind to it;
    # letting either into the graph drags a whole second Qt into the bundle.
    "PyQt5", "PyQt6", "PySide2", "tkinter", "wx",
    # Dev/test tooling.
    "pytest", "_pytest", "pytest_qt", "py", "pluggy", "nose", "unittest2",
    "IPython", "ipykernel", "jupyter", "jupyter_client", "jupyter_core", "notebook",
    "nbconvert", "nbformat", "qtconsole", "traitlets", "zmq", "tornado", "jedi",
    "sphinx", "numpydoc", "docutils", "setuptools._distutils",
    # Big scientific packages that live in this conda env but that nanoscope never imports.
    # Without these, a single transitive `import skimage` inside a dependency costs 300 MB.
    "torch", "torchvision", "jax", "jaxlib", "cv2", "skimage", "sympy", "numba", "llvmlite",
    "statsmodels", "plotly", "dask", "xarray", "sqlalchemy", "openpyxl",
    "xlsxwriter", "tables", "numexpr", "bottleneck", "PIL.ImageQt",
    # TensorFlow satellites present in the env and NOT used by ml_detect. tf_keras in particular
    # is a second, complete copy of Keras 2 that TF 2.15 will import if it is importable.
    "tensorflow_estimator", "tensorflow_addons", "tensorflow_datasets", "tensorflow_hub",
    "tensorflow_probability", "tensorflow_metadata", "tf_keras", "tensorboard",
    "tensorboard_data_server", "keras_preprocessing", "google.cloud", "googleapiclient",
]


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------
a = Analysis(
    [str(HERE / "nanoscope_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(HERE / "hooks")],
    hooksconfig={
        # Collect exactly the three file-writing backends instead of letting the hook autodetect
        # (which pulls in the macosx and Qt canvases). Qt translations are trimmed in _prune_toc
        # below — PyInstaller 6.21's Qt hook has no hooksconfig key for them.
        "matplotlib": {"backends": ["Agg", "pdf", "svg"]},
    },
    runtime_hooks=[str(HERE / "rthooks" / "rthook_nanoscope_env.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)


# --------------------------------------------------------------------------------------------
# TOC pruning — what the hooks collect wholesale but the app cannot use
# --------------------------------------------------------------------------------------------
# Path *prefixes*/fragments matched against the destination name of every collected file. This is
# the second half of the trimming story: `excludes` above works on the import graph, but hooks
# like hook-tensorflow and hook-PySide6 add data trees directly, and only a TOC filter reaches
# those. Every entry here was checked against a working build before being added.
_PRUNE_FRAGMENTS = (
    # --- Qt: whole subsystems the app has no code path into -----------------------------------
    "PySide6/Qt/qml/",
    "PySide6/Qt/translations/",
    "PySide6/Qt/lib/QtWebEngine",
    "PySide6/Qt/lib/QtQuick",
    "PySide6/Qt/lib/QtQml",
    "PySide6/Qt/lib/Qt3D",
    "PySide6/Qt/lib/QtCharts",
    "PySide6/Qt/lib/QtDataVisualization",
    "PySide6/Qt/lib/QtGraphs",
    "PySide6/Qt/lib/QtDesigner",
    "PySide6/Qt/lib/QtMultimedia",
    "PySide6/Qt/lib/QtPdf",
    "PySide6/Qt/lib/QtShaderTools",
    "PySide6/Qt/lib/QtLocation",
    "PySide6/Qt/lib/QtPositioning",
    "PySide6/Qt/lib/QtSpatialAudio",
    "PySide6/Qt/lib/QtRemoteObjects",
    "PySide6/Qt/lib/QtSensors",
    "PySide6/Qt/lib/QtLabs",
    "PySide6/Qt/libexec/",
    "PySide6/Qt/metatypes/",
    "PySide6/Qt/resources/",           # QtWebEngine's .pak blobs
    "PySide6/Qt/plugins/multimedia",
    "PySide6/Qt/plugins/position",
    "PySide6/Qt/plugins/sensors",
    "PySide6/Qt/plugins/sqldrivers",
    "PySide6/Qt/plugins/canbus",
    "PySide6/Qt/plugins/designer",
    "PySide6/Qt/plugins/qmltooling",
    "PySide6/Qt/plugins/scenegraph",
    "PySide6/Qt/plugins/renderers",
    "PySide6/Qt/plugins/geometryloaders",
    "PySide6/Qt/plugins/webview",
    "PySide6/Qt/plugins/texttospeech",
    "PySide6/Qt/plugins/help",
    # The only things that still referenced QtQml / QtQuick / QtPdf after the prunes above, found
    # by verify_bundle.py: the touchscreen virtual keyboard (macOS uses the cocoa plugin's own
    # input handling) and the PDF *image format* reader (the app writes PDFs through matplotlib,
    # it never asks Qt to rasterise one). Dropping these three makes the Qml/Quick/Pdf frameworks
    # genuinely unreferenced, which is what lets them be pruned at all.
    "PySide6/Qt/plugins/platforminputcontexts/",
    "PySide6/Qt/lib/QtVirtualKeyboard",
    "PySide6/Qt/plugins/imageformats/libqpdf",
    # --- TensorFlow: headers and sources, not runtime ------------------------------------------
    "tensorflow/include/",
    "tensorflow/xla_aot_runtime_src/",
    "tensorflow/python/tools/api/",
    "tensorflow-plugins/",             # libmetal_plugin.dylib — see landmine 2
    # --- generic dead weight -------------------------------------------------------------------
    "matplotlib/mpl-data/sample_data/",
    "matplotlib/mpl-data/fonts/pdfcorefonts/",
    "share/doc/",
    "/tests/",
    "/test/",
)


def _prune_toc(toc, fragments=_PRUNE_FRAGMENTS):
    """Drop TOC entries whose destination path contains any fragment. Returns (kept, dropped)."""
    kept, dropped = [], []
    for entry in toc:
        dest = entry[0].replace("\\", "/")
        (dropped if any(f in dest for f in fragments) else kept).append(entry)
    return kept, dropped


a.binaries, _dropped_bin = _prune_toc(a.binaries)
a.datas, _dropped_data = _prune_toc(a.datas)
print(f"[nanoscope.spec] pruned {len(_dropped_bin)} binaries + {len(_dropped_data)} data files")


# --------------------------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------------------------
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX corrupts signed/ad-hoc-signed Mach-O on macOS. Never enable.
    console=False,             # windowed .app; stdout/stderr are redirected by nanoscope_app.py
    disable_windowed_traceback=False,
    argv_emulation=False,      # we read argv ourselves; emulation breaks with a redirected stdio
    target_arch="arm64",       # Apple Silicon only, by decision. No universal2.
    codesign_identity=None,    # ad-hoc signing happens in build_app.sh, after the bundle exists
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=str(HERE / "resources" / "NanoScope.icns"),
    bundle_identifier=BUNDLE_ID,
    version=APP_VERSION,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": BUILD_RECORD["bundle_version"],
        "NanoScopeBuildID": BUILD_RECORD["build_id"],
        "NanoScopeSourceSHA256": BUILD_RECORD["source_sha256"],
        "CFBundlePackageType": "APPL",
        # NOT the arm64 floor (11.0) — the floor of what this env's wheels were COMPILED against.
        # PySide6 6.11.1 and shiboken6 ship binaries built for 15.0, and below that dyld
        # refuses them while macOS still permits the launch, so the app dies with no dialog
        # (its dialog is PySide6). verify_bundle.check_deployment_target derives the real
        # number from the Mach-O load commands and fails the build if this one understates it.
        "LSMinimumSystemVersion": "15.0",
        "LSApplicationCategoryType": "public.app-category.education",
        # Retina. Without this Qt renders into a 1x backing store and the trace view looks soft
        # on every Mac the app will actually run on.
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,  # the app ships its own dark palette
        "NSPrincipalClass": "NSApplication",
        "NSHumanReadableCopyright": "NanoScope",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Nanopore recording",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": ["public.data", "public.comma-separated-values-text",
                                       "public.plain-text"],
                "CFBundleTypeExtensions": ["abf", "h5", "hdf5", "tdms", "csv", "txt"],
            },
        ],
    },
)
