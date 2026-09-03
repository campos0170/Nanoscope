"""A layer's view state is work: editing it arms the quit guard (offscreen, pytest-qt).

Colour, Δt alignment, and a channel's tick are all written into the project file. Until now
none of them marked the session dirty, so the sequence *save, untick the voltage channel,
quit* discarded the untick with no prompt — the same class of silent loss the close guard
exists to prevent, reached by the one door still left open.

C15's fix is what makes this worth closing. An untick used to DELETE the layer, taking its
colour and its Δt with it; it now records ``excluded=True`` and the layer survives. That
turned unticking from an act into a stored decision, which puts it on the work side of the
line rather than the decoration side.

THE LINE THIS DRAWS, and why the obvious wiring is wrong: the guard hangs off
``FileDock.viewStateEdited`` — announced by the three handlers a user can actually reach —
and NOT off ``Session.layersChanged``. That signal fires for every add/remove/visible/colour/
offset/reorder, including the layer churn of opening a recording, closing one, and switching
between them. Connecting the quit guard to it would rebuild, by a third door, exactly the
"open a file, look at it, quit, answer a Save/Discard/Cancel modal" nag that B3 removed from
``documentOpened`` and C29 removed from ``documentClosed``.

    edits are work        -> tick, untick, Δt, colour        -> dirty
    navigation is not     -> open, close, switch recording   -> unchanged

The regression guards below are therefore as load-bearing as the positive cases: three of
them assert that the nag did NOT come back.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.file_dock import _COL_NAME, _ROLE
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 20_000


# ================================================================================================
# builders
# ================================================================================================

def _doc(path: Path, seed: int = 0, n_channels: int = 2) -> TraceDocument:
    rng = np.random.default_rng(seed)
    arrs, infos = [], []
    for i in range(n_channels):
        arrs.append(np.ascontiguousarray(
            np.full(N, 100.0, dtype=np.float32) + rng.normal(0, 1.5, N).astype(np.float32)))
        infos.append(ChannelInfo(
            index=i,
            name="current" if i == 0 else "voltage",
            raw_units="pA" if i == 0 else "mV",
            kind=ChannelKind.CURRENT if i == 0 else ChannelKind.VOLTAGE))
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=tuple(infos))
    return TraceDocument(meta=meta, channels=tuple(arrs))


@pytest.fixture
def win(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _install(w: MainWindow, doc: TraceDocument) -> str:
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _channel_item(w: MainWindow, doc_id: str, ch: int):
    it = QtWidgets.QTreeWidgetItemIterator(w._file_dock._tree)
    while it.value():
        payload = it.value().data(_COL_NAME, _ROLE)
        if payload and tuple(payload) == ("channel", doc_id, ch):
            return it.value()
        it += 1
    return None


def _set_ticked(w: MainWindow, doc_id: str, ch: int, ticked: bool) -> None:
    """Drive the REAL check state, so ``_on_item_changed`` runs exactly as a click leaves it."""
    item = _channel_item(w, doc_id, ch)
    assert item is not None, f"no channel row for {doc_id}#{ch}"
    item.setCheckState(
        _COL_NAME, QtCore.Qt.Checked if ticked else QtCore.Qt.Unchecked)


# ================================================================================================
# an edit is work
# ================================================================================================

def test_unticking_a_channel_after_a_save_arms_the_quit_guard(win, tmp_path):
    """THE REPORTED FAILURE. Save, untick the voltage channel, and the untick used to be
    discarded on quit with no prompt."""
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    win._mark_clean()
    assert win._dirty is False

    _set_ticked(win, doc_id, 1, False)

    assert win._dirty is True
    assert win.windowTitle().startswith("• ")
    # and the decision really is stored, which is what makes losing it a loss
    assert win._session.layer(f"{doc_id}#1").excluded is True


def test_re_ticking_a_channel_arms_the_quit_guard(win, tmp_path):
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    _set_ticked(win, doc_id, 1, False)
    win._mark_clean()

    _set_ticked(win, doc_id, 1, True)

    assert win._dirty is True


def test_changing_a_layers_time_offset_arms_the_quit_guard(win, tmp_path):
    """Δt is persisted AND changes what the ML detector measures — the strongest case in the
    group for calling view state work."""
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    win._mark_clean()

    win._file_dock._on_offset_changed(f"{doc_id}#0", 1.5)

    assert win._dirty is True
    assert win._session.layer(f"{doc_id}#0").time_offset_s == pytest.approx(1.5)


def test_picking_a_layer_colour_arms_the_quit_guard(win, tmp_path, monkeypatch):
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    win._mark_clean()

    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor",
        staticmethod(lambda *a, **k: QtGui.QColor("#ff8800")))
    win._file_dock._pick_color(f"{doc_id}#0")

    assert win._dirty is True
    assert win._session.layer(f"{doc_id}#0").color == "#ff8800"


def test_a_cancelled_colour_dialog_changes_nothing(win, tmp_path, monkeypatch):
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    before = win._session.layer(f"{doc_id}#0").color
    win._mark_clean()

    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor",
        staticmethod(lambda *a, **k: QtGui.QColor()))   # invalid == user pressed Cancel
    win._file_dock._pick_color(f"{doc_id}#0")

    assert win._dirty is False
    assert win._session.layer(f"{doc_id}#0").color == before


def test_changing_the_persisted_view_mode_arms_the_quit_guard(win, tmp_path):
    """Overlay/Stacked is written into ProjectState, so changing it after a save is work.

    This signal is safe to track directly: unlike ``layersChanged`` it is never emitted by
    opening, closing, or selecting a recording, and project restoration runs behind the
    existing ``_restoring`` fence.
    """
    _install(win, _doc(tmp_path / "a.dat"))
    win._mark_clean()

    win._session.set_view_mode(ViewMode.STACKED)

    assert win._dirty is True
    assert win.windowTitle().startswith("• ")


# ================================================================================================
# navigation is not — the three regression guards (B3, C29, and the switch)
# ================================================================================================

def test_opening_a_recording_still_does_not_arm_the_quit_guard(win, tmp_path):
    """B3. Opening a file creates layers, so ``layersChanged`` fires — wiring the guard to
    that signal would arm it on the most ordinary action in the program."""
    _install(win, _doc(tmp_path / "a.dat"))

    assert win._dirty is False
    assert win.windowTitle() == "NanoScope"


def test_closing_a_second_recording_still_does_not_arm_the_quit_guard(win, tmp_path):
    """C29. Closing drops layers and emits ``layersChanged`` from ``close_document``."""
    a = _install(win, _doc(tmp_path / "a.dat", seed=1))
    _install(win, _doc(tmp_path / "b.dat", seed=2))
    win._mark_clean()

    win._session.close_document(a)

    assert win._dirty is False


def test_switching_between_recordings_does_not_arm_the_quit_guard(win, tmp_path):
    """The line: ``show_only`` rewrites every layer's visibility, and visibility IS persisted,
    but changing which recording is on screen is navigation, not an edit."""
    a = _install(win, _doc(tmp_path / "a.dat", seed=1))
    b = _install(win, _doc(tmp_path / "b.dat", seed=2))
    win._mark_clean()

    win._on_selection(a, 0)
    win._on_selection(b, 0)

    assert win._dirty is False


def test_the_docks_own_programmatic_sync_announces_nothing(win, tmp_path):
    """``_sync_layers`` rewrites every check state to match the session. It is guarded by
    ``_updating``; if that guard ever lapses, a repaint would mark the project dirty."""
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    win._mark_clean()

    seen = []
    win._file_dock.viewStateEdited.connect(lambda: seen.append(1))
    win._file_dock._sync_layers()

    assert seen == []
    assert win._dirty is False
    assert _channel_item(win, doc_id, 0) is not None    # the rows really were rebuilt


def test_a_restore_does_not_arm_the_quit_guard(win, tmp_path):
    """``_mark_dirty`` is a no-op while ``_restoring``; a restore reapplies every layer's
    colour, offset and exclusion, so this is the path that would announce most loudly."""
    doc_id = _install(win, _doc(tmp_path / "a.dat"))
    win._mark_clean()

    win._restoring = True
    try:
        _set_ticked(win, doc_id, 1, False)
        win._file_dock._on_offset_changed(f"{doc_id}#0", 2.0)
    finally:
        win._restoring = False

    assert win._dirty is False


# ================================================================================================
# structure — the wiring itself, so a refactor cannot quietly undo the decision
# ================================================================================================

def test_the_guard_is_not_wired_to_layers_changed(win):
    """If someone "simplifies" this onto ``Session.layersChanged``, the B3/C29 nag returns and
    only the three navigation tests above would catch it. Assert the wiring directly too:
    emitting the session signal by itself must not dirty anything."""
    win._mark_clean()

    win._session.layersChanged.emit()

    assert win._dirty is False
