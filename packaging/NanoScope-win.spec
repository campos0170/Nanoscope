# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for NanoScope on WINDOWS (x86-64, onedir).

Run it through packaging\\build_app.ps1, not by hand.

WHY A SECOND SPEC AND NOT ONE WITH `if sys.platform`. The two builds disagree about more than a
few flags: macOS produces a code-signed .app via BUNDLE with an Info.plist, Windows produces a
directory with an .exe carrying an embedded icon and version resource; the Qt payload sits in
`PySide6/Qt/lib/*.framework` on one and `PySide6/Qt/bin/*.dll` on the other, so even the prune
list differs line for line. A single spec threaded with platform branches would be harder to read
than two, and the macOS one is load-bearing and already correct — it is not worth destabilising.

WHAT IS SHARED, and deliberately so: the hidden imports, the exclude list and the runtime hook.
Those encode facts about THIS APP's imports, not about an OS, and a second copy of them is how
the two builds would drift. They are imported from the macOS spec's own module below.

THE ONE BEHAVIOURAL DIFFERENCE WORTH KNOWING. `os.replace` — the atomic-save primitive in
nanoscope/core/atomic_write.py — fails on Windows when the destination file is open in another
process. Exporting a CSV over one that is open in Excel raises instead of succeeding. That is a
safer failure than the POSIX behaviour (no half-written file either way), but it IS a difference,
and the message the user sees comes from the OS.
"""
import sys
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

if sys.platform != "win32":
    raise SystemExit(
        "NanoScope-win.spec builds a Windows executable and PyInstaller does not "
        "cross-compile: run it on Windows. On macOS use packaging/NanoScope.spec."
    )

HERE = Path(SPECPATH).resolve()  # noqa: F821
ROOT = HERE.parent

APP_NAME = "NanoScope"

sys.path.insert(0, str(HERE))
from build_identity import write_build_record, write_windows_version_resource

BUILD_RECORD, BUILD_RECORD_PATH = write_build_record(ROOT)
APP_VERSION = BUILD_RECORD["version"]
VERSION_RESOURCE = write_windows_version_resource(BUILD_RECORD, BUILD_RECORD_PATH.parent)


# --------------------------------------------------------------------------------------------
# Hidden imports — the same facts about this app's imports as the macOS build
# --------------------------------------------------------------------------------------------
hiddenimports = [
    # nanoscope.io.base imports these by name at first use — invisible to the module graph.
    "nanoscope.io.abf",
    "nanoscope.io.hdf5",
    "nanoscope.io.tdms",
    "nanoscope.io.text",
    # neo resolves its IO classes through a registry; name the one io/abf.py actually uses.
    "neo.rawio.axonrawio",
    # pandas' Parquet engine is chosen at call time from a string.
    "pyarrow",
    "pyarrow.parquet",
    "pandas.io.parquet",
    # matplotlib's file-export backends (MPLBACKEND=Agg is forced by the runtime hook; the PDF
    # and SVG writers are separate modules pyplot loads on demand from savefig's suffix).
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
    # pyqtgraph's exporters are imported from the package root by five GUI panels.
    "pyqtgraph.exporters",
    "pyqtgraph.exporters.ImageExporter",
    # scipy pieces reached through lazy/dispatch layers rather than plain imports.
    "scipy.special.cython_special",
    "scipy._lib.array_api_compat.numpy.fft",
    # `NanoScope.exe --self-test`; imported inside a function so the GUI path never pays for it.
    "nanoscope_selftest",
]

datas = [
    (str(BUILD_RECORD_PATH), "nanoscope"),
    # importlib.metadata.version("nanoscope") stamps the app version onto every detection's
    # provenance; without the dist-info every exported CSV loses its version stamp.
    *copy_metadata("nanoscope"),
]
binaries = []

# --- OPTIONAL ML. TensorFlow/Keras are NOT in pyproject's dependencies: ml_detect imports them
# lazily and degrades to "TensorFlow is not available" when absent. The macOS build is made in an
# env that has them, so its spec collects them unconditionally. A Windows build made from a clean
# `pip install -e .` has neither, and an unguarded collect_submodules("keras") would abort the
# build with a bare import error. Guarded: ML ships when it is installed, and the app is built
# without it when it is not — which is a supported configuration, not a broken one.
try:
    hiddenimports += collect_submodules("keras")
    datas += collect_data_files("keras")
    HAVE_ML = True
except Exception as exc:  # noqa: BLE001 — absence is a supported configuration
    print(f"[NanoScope-win.spec] keras not collected ({exc.__class__.__name__}): "
          f"building WITHOUT the ML detector. Install tensorflow+keras and rebuild to include it.")
    HAVE_ML = False


# --------------------------------------------------------------------------------------------
# Excludes — identical reasoning to the macOS spec (see its comments for the why of each)
# --------------------------------------------------------------------------------------------
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

_LEAF_EXCLUDES = [
    "fsspec.gui", "panel", "holoviews", "hvplot", "bokeh", "datashader", "param", "pyviz_comms",
    "intake", "altair", "vtk", "vtkmodules", "anaconda_catalogs", "colorcet",
    "probeinterface", "zarr", "numcodecs",
    "huggingface_hub", "pydantic", "mypy", "hf_xet", "lxml",
    "grpc", "grpc_status",
]

excludes = _QT_EXCLUDES + _LEAF_EXCLUDES + [
    "PyQt5", "PyQt6", "PySide2", "tkinter", "wx",
    "pytest", "_pytest", "pytest_qt", "py", "pluggy", "nose", "unittest2",
    "IPython", "ipykernel", "jupyter", "jupyter_client", "jupyter_core", "notebook",
    "nbconvert", "nbformat", "qtconsole", "traitlets", "zmq", "tornado", "jedi",
    "sphinx", "numpydoc", "docutils", "setuptools._distutils",
    "torch", "torchvision", "jax", "jaxlib", "cv2", "skimage", "sympy", "numba", "llvmlite",
    "statsmodels", "plotly", "dask", "xarray", "sqlalchemy", "openpyxl",
    "xlsxwriter", "tables", "numexpr", "bottleneck", "PIL.ImageQt",
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
    hooksconfig={"matplotlib": {"backends": ["Agg", "pdf", "svg"]}},
    runtime_hooks=[str(HERE / "rthooks" / "rthook_nanoscope_env.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)


# --------------------------------------------------------------------------------------------
# TOC pruning — WINDOWS PATHS. Qt ships as DLLs in PySide6/ and PySide6/Qt/bin, not as
# .framework bundles, so every fragment below differs from the macOS spec's by more than a
# separator. Both spellings are matched where the PySide6 wheel layout has moved between
# versions (6.x has shipped Qt6Foo.dll both at PySide6/ and at PySide6/Qt/bin/).
# --------------------------------------------------------------------------------------------
_QT_DEAD = ("WebEngineCore", "WebEngineQuick", "WebEngineWidgets", "Quick", "Quick3D",
            "QuickControls2", "QuickWidgets", "Qml", "QmlModels", "QmlWorkerScript",
            "3DCore", "3DRender", "3DInput", "3DLogic", "3DAnimation", "3DExtras",
            "Charts", "DataVisualization", "Graphs", "Designer", "Multimedia",
            "MultimediaWidgets", "Pdf", "PdfWidgets", "ShaderTools", "Location",
            "Positioning", "SpatialAudio", "RemoteObjects", "Sensors", "VirtualKeyboard",
            "Bluetooth", "Nfc", "SerialBus", "SerialPort", "TextToSpeech", "WebChannel",
            "WebSockets", "Scxml", "StateMachine", "Sql", "Help", "UiTools", "Test")

_PRUNE_FRAGMENTS = tuple(
    [f"PySide6/Qt6{m}.dll" for m in _QT_DEAD]
    + [f"PySide6/Qt/bin/Qt6{m}.dll" for m in _QT_DEAD]
    + [f"PySide6/Qt{m}.pyd" for m in _QT_DEAD]
    + [
        # Whole subsystems, wherever the wheel puts them.
        "PySide6/Qt/qml/",
        "PySide6/qml/",
        "PySide6/Qt/translations/",
        "PySide6/translations/",
        "PySide6/Qt/resources/",          # QtWebEngine .pak blobs
        "PySide6/resources/",
        "PySide6/Qt/libexec/",
        "PySide6/Qt/metatypes/",
        "PySide6/metatypes/",
        # Plugin families with no code path into them. NOTE: platforms/ is NOT here — qwindows.dll
        # lives there and dropping it is the classic "could not find the Qt platform plugin
        # windows" build. styles/ stays too: qwindowsvistastyle.dll is the native look.
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
        "PySide6/Qt/plugins/platforminputcontexts/",
        "PySide6/Qt/plugins/imageformats/qpdf",
        # --- TensorFlow: headers and sources, not runtime -----------------------------------
        "tensorflow/include/",
        "tensorflow/xla_aot_runtime_src/",
        "tensorflow/python/tools/api/",
        # --- generic dead weight -------------------------------------------------------------
        "matplotlib/mpl-data/sample_data/",
        "matplotlib/mpl-data/fonts/pdfcorefonts/",
        "share/doc/",
        "/tests/",
        "/test/",
    ]
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
print(f"[NanoScope-win.spec] pruned {len(_dropped_bin)} binaries + {len(_dropped_data)} data files"
      f" | ML {'included' if HAVE_ML else 'NOT included'}")

# The platform plugin is the single point of failure for a Qt app on Windows: without
# plugins/platforms/qwindows.dll the exe dies with "could not find or load the Qt platform
# plugin windows" and no other diagnostic. Assert it survived the prune rather than discover
# it after shipping.
_dests = [e[0].replace("\\", "/") for e in a.binaries + a.datas]
if not any("plugins/platforms/qwindows" in d for d in _dests):
    raise SystemExit(
        "[NanoScope-win.spec] qwindows.dll is not in the bundle — the app would not start. "
        "A prune fragment is too broad, or PySide6 is not installed in this environment."
    )


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
    upx=False,                 # UPX mangles some Qt/numpy DLLs and trips SmartScreen. Never on.
    console=False,             # windowed; stdout/stderr are redirected by nanoscope_app.py
    disable_windowed_traceback=False,
    icon=str(HERE / "resources" / "NanoScope.ico"),
    version=str(VERSION_RESOURCE),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
