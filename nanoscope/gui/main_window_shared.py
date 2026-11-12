"""Module-level names shared by ``main_window`` and its ``main_window_<part>`` mixins: constants,
small helpers, and the ``_InfoBar`` and ``_RecoverySnapshot`` classes.

A leaf: it imports nothing from ``main_window`` and nothing from a part. ``main_window.py``
re-imports every name defined here, so each still resolves on ``nanoscope.gui.main_window``.
The definitions were moved verbatim out of ``main_window.py`` by a mechanical split; nothing in
them was edited.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from ..analysis.reconcile import CONSENSUS_MODULE_ID

_OPEN_FILTER = ("Recordings (*.abf *.h5 *.hdf5 *.tdms *.csv *.txt);;"
                "Axon ABF (*.abf);;HDF5 (*.h5 *.hdf5);;TDMS (*.tdms);;"
                "Text trace (*.csv *.txt);;All files (*)")
_MAX_RECENT = 10

#: How long :meth:`MainWindow._teardown` will wait for a detached worker on quit.
#: Every kernel polls the cooperative cancel token, so a job that CAN stop stops in
#: milliseconds and never sees this number; it only ever bites on an uninterruptible
#: C-level call (``ml_detect._import_tf`` measured at 4.1 s). 1000 ms was too short to
#: cover any of those and long enough to be felt, which is the worst of both — 2500 ms
#: covers the realistic ones, and ``gui.app.main`` handles the rest without a crash.
_TEARDOWN_DRAIN_MS = 2500

#: Why the ML actions are off on a machine without TensorFlow — VERBATIM the sentence
#: the DetectionPanel's gate label shows (panels.DetectionPanel._refresh_ml_enabled),
#: because the menu and the panel are two surfaces for the same fact and may never
#: disagree about it, down to the wording.
_ML_TF_MISSING = ("TensorFlow is not installed, so ML detection is unavailable. "
                  "Install tensorflow (2.15) to enable it.")

# The three answers to the unsaved-work close prompt (see MainWindow.closeEvent /
# _confirm_close_choice). Kept as plain strings so a test can select one without a modal.
_CLOSE_SAVE = "save"
_CLOSE_DISCARD = "discard"
_CLOSE_CANCEL = "cancel"

# The startup recovery prompt's three deliberate outcomes.  Plain strings keep the UI
# decision injectable: the integration suite can exercise every branch without opening a
# native modal, exactly as the close guard does above.
_RECOVERY_RECOVER = "recover"
_RECOVERY_DISCARD = "discard"
_RECOVERY_SHOW = "show"


@dataclass(frozen=True)
class _RecoverySnapshot:
    """One immutable hand-off from the GUI thread to the recovery writer."""

    revision: int
    state: Any
    source_project_path: str | None
    covered_explicit_save_revision: int


def _write_recovery_snapshot(store, snapshot: _RecoverySnapshot,
                             _revision: int | None = None):
    """Qt-free worker callable; it closes over the store, never a MainWindow."""
    return store.write(
        snapshot.state,
        revision=int(snapshot.revision),
        source_project_path=snapshot.source_project_path,
        covered_explicit_save_revision=int(snapshot.covered_explicit_save_revision),
    )

#: How a detector's ``module_id`` is spelled in a sentence the user reads. Used by the
#: Edit menu to say WHICH other table holds the undo history it does not have here.
_DETECTOR_LABELS = {
    "detect.schmitt": "the threshold detector",
    "detect.cusum": "the CUSUM detector",
    "detect.keras": "the ML detector",
    "detect.study": "the imported study candidate table",
    "detect.manual": "your Manual table",
    CONSENSUS_MODULE_ID: "the reconciled event table",
}

_RECONCILED_TRACE_ANALYSIS_REFUSAL = (
    "The Reconciled table can contain rows from different detector preprocessing "
    "chains, so one waveform cannot be reconstructed honestly for every row. Switch "
    "to a source detector table and run this analysis there."
)


# The two extra answers to the "this recording has segments cut from it" close prompt (see
# _confirm_close_with_segments); _CLOSE_CANCEL above is the third. Same plain-string discipline.
_CLOSE_SEGMENTS_TOO = "segments_too"
_CLOSE_PARENT_ONLY = "parent_only"

# Opening widths for the two docks on a FIRST launch (see _apply_default_dock_widths). They
# are deliberate defaults, NOT floors — both docks stay draggable well below and above these,
# and the moment the user settles on her own widths the saved windowState takes over.
# Left column: a little above the file manager's natural 293 px, so the tree's name column
# starts with room to grow rather than at its minimum. (It cannot start much wider without
# taking from the viewer; the tree's Δt column claims a fixed ~148 px whatever the dock's
# width, which is what really caps how much of a recording name is visible.)
_FILE_DOCK_DEFAULT_WIDTH = 320
# Right column: the Inspector's own children are all scroll areas now, so their combined
# sizeHint is an accident of whichever control stack happens to be widest rather than a
# considered default. This is the considered one — and it is the FLOOR of the opening
# width, not the width itself: see _apply_default_dock_widths, which opens the dock wide
# enough to SHOW its widest page (capped) rather than wide enough to hide half of it.
#
# WHY THIS STOPPED BEING ONE NUMBER. 460 opened the dock at a viewport of 420 px against
# an Analysis control column whose natural width is 836, so the page came up with a
# horizontal scrollbar and everything past x=420 — the protocol Save as…/Rename/Delete
# buttons, the ML preset buttons, Null control…, Recovery map…, Merge all — was off the
# right-hand edge of a 1512 px screen with no menu item and no shortcut behind it. A
# scrollbar is not a door: five features were reported missing that were only off-screen.
_INSPECTOR_DEFAULT_WIDTH = 460
# ...and the ceiling. The Inspector may not open so wide that the signal viewer — the thing
# this application is for — stops being the widest column. On the 1512 px display this is
# measured against, the three columns and their two separators are
# 320 (file) + 12 (separators) + inspector + central = 1512, so an Inspector of 560 leaves
# the viewer 620: wider than the Inspector, and wider than it than the file manager.
# MEASURED with a fresh profile at the width the dock actually opens at (502, below): the
# Analysis control column lays out in its 462 px viewport with NO horizontal scrollbar, and
# every control the audit found off-screen (protocol Save as…/Rename/Delete, the ML preset
# pair, Detect / Sweep threshold… / Null control… / Recovery map…, Merge all) sits whole
# inside the window.
# Above this the dock stays freely draggable — nothing here is a minimum or a maximum SIZE,
# both docks still drag from ~137 px to the whole window — it is only where they OPEN.
#
# THIS IS A CEILING, NOT THE WIDTH. The dock used to open AT it, because the derivation
# below asked each page for its *natural* width (the Analysis control column's sizeHint is
# 646) and every page wants more than the cap, so the cap always won. Whether a control is
# reachable is decided by the page's MINIMUM, not by its natural width, and the minimums are
# much smaller. Asking for the minimum instead (see _inspector_default_width) opens the dock
# at 502 px on this tree and hands the other 58 px back to the signal viewer — 678 px at
# 1512, the widest column by 176 px. The ceiling now only bites if a page's MINIMUM grows
# past it, which is the case it was written for.
_INSPECTOR_MAX_DEFAULT_WIDTH = 560
#: Frame + vertical scrollbar between a dock's own width and the viewport its page is
#: laid out in — measured (dock 460 -> viewport 420) and stable across the three pages.
_INSPECTOR_CHROME_WIDTH = 40
#: ...and one further nesting level of the same, for a page that lives inside a SECOND
#: scroll area rather than directly in the dock's own. Measured at dock 560, as
#: dock width minus viewport width: 6 px for a page sitting straight in the dock,
#: 40 for the Analysis control column (one scroll area in), 64 for the Manual-measurement
#: panel (two in). A nested page hides its minimum from its parent — that is why the dock
#: cannot simply ask the tab widget — so the deepest allowance is the one to reserve.
_INSPECTOR_NESTED_CHROME_WIDTH = 24
#: One dock separator. Two of them stand between the three columns: measured at 1512, where
#: 320 (file) + 620 (viewer) + 560 (inspector) = 1500 leaves exactly 12 px.
_DOCK_SEPARATOR_WIDTH = 6

#: saveState/restoreState version. 0 = every layout written before the Results dock.
#: A change that adds a dock bumps this and appends a step to _migrate_window_state.
_WINDOW_STATE_VERSION = 1
#: The Results dock's OPENING height (first launch, or a migrated old layout): tall enough
#: for this many table rows, capped at a fraction of the dock area so the trace keeps the
#: rest, and never below the minimum. Like the dock widths, a default and not a floor.
_RESULTS_DEFAULT_ROWS = 6
_RESULTS_MAX_HEIGHT_FRACTION = 1 / 3
_RESULTS_MIN_DEFAULT_HEIGHT = 180
#: Where the Results panel is reopened from, said ONCE: every hint that names it quotes this.
_RESULTS_PANEL_PATH = "View ▸ Panels ▸ Results (⇧⌘R)"

# -- the Event inspector's two hosts, and the child windows MainWindow remembers -------------
_ROLE_INSPECTOR = "event_inspector"
_ROLE_GALLERY = "event_gallery"
_HOST_WINDOW = "window"
_HOST_DOCK = "dock"
#: Opt-in: the inspector opens as its own window until the user docks it.
_INSPECTOR_DEFAULT_HOST = _HOST_WINDOW
_INSPECTOR_DOCK_NAME = "EventInspectorDock"
#: The side columns only. In the top or bottom area its minimum height would add to the
#: trace's and force the main window taller than a laptop screen.
_INSPECTOR_DOCK_AREAS = (QtCore.Qt.LeftDockWidgetArea, QtCore.Qt.RightDockWidgetArea)
#: Every child window whose size and place are remembered ("windows/<key>/geometry"),
#: in Window-menu order.
_CHILD_WINDOW_KEYS = ("event_inspector", "event_gallery", "analysis_explorer",
                      "figure_library", "first_analysis_guide", "about",
                      "incubation_launcher", "incubation_reference", "manual_study")
#: Window-menu names for the windows that are not always listed.
_CHILD_WINDOW_LABELS = {
    "first_analysis_guide": "Guided first analysis",
    "about": "About NanoScope",
    "incubation_launcher": "Incubation study",
    "incubation_reference": "Original Zn reference recipe",
    "manual_study": "Manual study",
}
_SETTING_INSPECTOR_HOST = "windows/event_inspector/host"
_SETTING_DETAILS = {_HOST_WINDOW: "windows/event_inspector/details_window",
                    _HOST_DOCK: "windows/event_inspector/details_dock"}


def _geometry_setting(key: str) -> str:
    return f"windows/{key}/geometry"


#: QWidget.saveGeometry()'s header: magic, major and minor version, the frame and the normal
#: rect (left, top, right, bottom each), the screen's index, the maximized / full-screen flags.
_GEOMETRY_HEADER = struct.Struct(">IHH4i4iiBB")
_GEOMETRY_MAGIC = 0x1D9D0CB
_GEOMETRY_MAJOR = 3


def _saved_normal_rect(blob) -> tuple[QtCore.QRect, bool] | None:
    """The normal rect and the maximized flag of a ``saveGeometry()`` blob; None for bytes
    Qt could never restore (empty, other magic, newer version, cut short, empty rect).

    Qt also answers False to a sound blob saved on a screen whose width is far from the
    current one (a laptop and an external monitor). That is not a damaged preference, so
    the two are told apart here, by the header Qt itself checks."""
    raw = bytes(blob)
    if len(raw) < _GEOMETRY_HEADER.size:
        return None
    magic, major, _minor, *numbers, _screen, maximized, _full = _GEOMETRY_HEADER.unpack_from(raw)
    if magic != _GEOMETRY_MAGIC or major > _GEOMETRY_MAJOR:
        return None
    left, top, right, bottom = numbers[4:8]
    rect = QtCore.QRect(QtCore.QPoint(left, top), QtCore.QPoint(right, bottom))
    return (rect, bool(maximized)) if rect.isValid() else None


def _with_project_suffix(path: str) -> str:
    """A chosen project path with the mandatory ``.nanoscope`` suffix.

    Shared by the overwrite check and the write, so the file the user is warned about
    is the file that gets replaced. Keeping the two spellings apart is how "Save As"
    over an existing ``study.nanoscope``, typed as ``study``, could destroy it with no
    warning at all.
    """
    return path if path.lower().endswith(".nanoscope") else path + ".nanoscope"


def _restorable_bytes(settings: QtCore.QSettings, key: str, ignored: list[str]) -> object | None:
    """A preferences value ``restoreGeometry``/``restoreState`` can actually be handed.

    *** THE PREFERENCES STORE IS EXTERNAL INPUT, LIKE ANY OTHER FILE ON DISK. ***

    ``QSettings.value`` returns whatever the backing store decodes to, and neither the
    plist (the shipping ``NativeFormat`` backend) nor an .ini can promise a QByteArray:
    a store the app wrote itself and that was later TRUNCATED — a crash or a full disk
    mid-quit, a half-synced or partially restored ``~/Library`` — hands back a plain
    ``str``, and so does ``defaults write … windowState -string x`` or any plist editor.
    Both Qt calls raise ``TypeError`` on that, inside ``MainWindow.__init__``, which
    ``gui.app.main`` does not guard — so the app did not start AT ALL, on every launch,
    with no message and no in-app way back: the user had to know to delete
    ``~/Library/Preferences/com.nanoscope-local.NanoScope.plist`` by hand.

    A damaged preference costs the user their saved layout. It must never cost them the
    program. So a value of the wrong kind is IGNORED and its key recorded in ``ignored``,
    which :meth:`MainWindow._restore_settings` says out loud rather than swallowing —
    the same reasoning ``recentFiles``/``recentProjects`` two lines below already got,
    and the same policy ``_open_project`` states for the other externally-mutable file
    ("a corrupt archive must not crash the app").
    """
    raw = settings.value(key)
    if raw is None:
        return None
    if isinstance(raw, (QtCore.QByteArray, bytes, bytearray, memoryview)):
        return raw
    ignored.append(key)
    return None


def _settings_store_damaged(s: QtCore.QSettings) -> bool:
    """Did the preferences store fail to parse? Asked so it can be SAID (see
    :meth:`MainWindow._restore_settings`) — a store that drops keys wholesale makes the
    app come up factory-fresh, and that is a change the user sees.

    ``status()`` alone is not a reliable witness: Qt parses a settings file once per
    process and serves every later ``QSettings`` from a cache that has forgotten the
    parse error (measured: first instance ``FormatError``, second ``NoError``, same
    damaged file). So when ``status()`` says nothing, the file is copied to a fresh path
    — which cannot be served from the cache — and parsed again; THAT instance's status
    is the file's real health. Best-effort in the honest direction: a copy that cannot
    be made (unreadable file, full temp disk) reports "not damaged", because a disclosure
    built on a guess is the false alarm that teaches the user to ignore the real one.
    """
    if s.status() != QtCore.QSettings.NoError:
        return True
    path = Path(s.fileName())
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / path.name
            shutil.copyfile(path, copy)
            fresh = QtCore.QSettings(str(copy), s.format())
            damaged = fresh.status() != QtCore.QSettings.NoError
            del fresh
        return damaged
    except OSError:
        return False


def _string_setting(settings: QtCore.QSettings, key: str, fallback: str,
                    ignored: list[str]) -> str:
    """A preferences value that is supposed to be a plain string, or ``fallback``.

    Same reasoning as :func:`_restorable_bytes`, one severity down: a ``lastDir`` that
    comes back as a list cannot crash anything, but ``str(["/a"])`` is not a directory
    and the next Open dialog would start in a place that does not exist, with nobody
    ever told why. Wrong kind -> the default, and the key is named.

    CONTENT is checked too, not only kind. Store damage does not always break the file's
    grammar: a truncated-then-scribbled ini parsed back a ``lastDir`` of
    ``'/Users/somebody/heal \\x00\\xff[broken'`` with ``status() == NoError`` — a str,
    happily adopted, silently pointing the next Open dialog at gibberish. This app never
    writes a control character into these keys (they hold real filesystem paths), so one
    coming BACK is the same statement as a wrong type: a value this app did not write.
    """
    raw = settings.value(key, fallback)
    if isinstance(raw, str) and not any(ord(ch) < 0x20 or ch == "\x7f" for ch in raw):
        return raw
    if raw is not None:
        ignored.append(key)
    return fallback


def _choice_setting(settings: QtCore.QSettings, key: str, allowed: tuple[str, ...],
                    ignored: list[str]) -> str | None:
    """One of ``allowed``, or ``None`` (absent, or a value this app did not write -- the
    key is then named, as :func:`_string_setting` does)."""
    if not settings.contains(key):
        return None
    raw = settings.value(key)
    if isinstance(raw, str) and raw in allowed:
        return raw
    ignored.append(key)
    return None


def _bool_setting(settings: QtCore.QSettings, key: str, ignored: list[str]) -> bool | None:
    """A bool this app wrote (an ini store hands it back as "true"/"false"), or ``None``."""
    if not settings.contains(key):
        return None
    raw = settings.value(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    ignored.append(key)
    return None


def _stored_baseline_fingerprint(provenance: object) -> str | None:
    """The open-pore estimator stamp a restored result carries, or ``None``.

    ``None`` for a table that never had one -- a pre-change project, or a detector
    handed a baseline array by its caller, which can name no estimator at all.
    Absence is the signal the staleness check reads, so a malformed value must
    become absence rather than a stamp that can never match anything.
    """
    if not isinstance(provenance, Mapping):
        return None
    fp = provenance.get("baseline_fingerprint")
    return str(fp) if isinstance(fp, str) and fp else None


class _InfoBar(QtWidgets.QFrame):
    """A thin, dismissible, non-blocking banner (used for the duplicate-representation advisory)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(
            "QFrame{background:#2a2f36; border:1px solid #3a4048; border-radius:4px;}"
            "QLabel{border:none;}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 6, 6)
        self._label = QtWidgets.QLabel()
        self._label.setWordWrap(True)
        lay.addWidget(self._label, 1)
        close = QtWidgets.QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.clicked.connect(self.hide)
        lay.addWidget(close)

    def show_message(self, text: str) -> None:
        self._label.setText(text)
        self.setVisible(True)
