"""Named templates for the analytical windows other than Distributions.

Distributions has had named templates for a long time (``PlotsWindow`` A3); the other
windows had only appearance presets, so a Welch setup or a comparison layout was re-made by hand
on every recording. :class:`WindowTemplateControls` is the one implementation the five windows
share, placed in each page's Export section. A template keeps the window's ANALYSIS SETTINGS and
its APPEARANCE, never its sources, its time span or its results: it is a recipe to put on another
recording, the same split ``PlotConfig.as_template`` makes. The window owns what its settings are
(``template_settings`` / ``apply_template_settings``); this module only names, stores and recalls
them. Recalling one never runs a gated analysis: where the window has Generate or Apply fit, the
settings arrive as the draft that button commits.

The project copy of these templates (and of the appearance presets and Distributions
templates) is ``gui.plot_preset_sync``'s. This module imports no panel: every analytical window
embeds it, so a reach from here into one would be a two-hop panel-to-panel import
(``tests/gui/test_measure_navigation.py``'s allowlist guard).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

from PySide6 import QtCore, QtWidgets

from ..core.plot_preset_state import NAME_LIMIT
from .plot_appearance import PlotAppearance, preset_store

__all__ = [
    "TEMPLATE_FORMAT", "TEMPLATE_VERSION", "WINDOW_KINDS", "TemplateError",
    "WindowTemplateControls", "decode_window_template", "encode_window_template",
    "load_window_templates", "store_window_templates",
]

TEMPLATE_FORMAT = "nanoscope-window-template"
TEMPLATE_VERSION = 1

#: The windows that keep templates here, by their stable store id, with the name the user sees.
#: The id is the QSettings key's last part and the project copy's key, so it never changes with
#: a window's title. I-V is absent on purpose: its only settings are appearance, which the
#: appearance presets already keep.
WINDOW_KINDS: dict[str, str] = {
    "noise-spectrum": "Noise spectrum",
    "voltage-dependence": "Voltage dependence",
    "event-waveforms": "Event waveforms",
    "distribution-summaries": "Distribution summaries",
    "comparison-figures": "Comparison figures",
}
_SETTINGS_PREFIX = "plot_templates/"
_PLACEHOLDER = "Choose a template…"


class TemplateError(ValueError):
    """A stored or carried template this build cannot recall, with a sentence saying why."""


# -- the store ------------------------------------------------------------------------------------

def _settings_key(kind: str) -> str:
    if kind not in WINDOW_KINDS:
        raise KeyError(f"no template store for window {kind!r}")
    return _SETTINGS_PREFIX + kind


def load_window_templates(kind: str) -> dict[str, dict]:
    """The saved templates for ``kind``, as raw payloads. A corrupt store reads as none.

    The Distributions store's rule (``PlotsWindow._load_templates``): a store this build cannot
    make sense of is as useful as an empty one, and refusing to open the window over it would be
    the worse failure. Each payload is decoded on recall, so one bad entry cannot hide the rest.
    """
    raw = QtCore.QSettings().value(_settings_key(kind), "")
    if not raw:
        return {}
    try:
        got = json.loads(raw) if isinstance(raw, str) else {}
    except (TypeError, ValueError):
        return {}
    if not isinstance(got, dict):
        return {}
    return {name: value for name, value in got.items()
            if isinstance(name, str) and name.strip() and isinstance(value, dict)}


def store_window_templates(kind: str, templates: Mapping[str, dict]) -> bool:
    settings = QtCore.QSettings()
    settings.setValue(_settings_key(kind),
                      json.dumps(dict(templates), sort_keys=True, allow_nan=False))
    settings.sync()
    return settings.status() == QtCore.QSettings.NoError


def encode_window_template(kind: str, settings: Mapping[str, Any],
                           appearance: Mapping[str, Any]) -> dict[str, Any]:
    from .. import __version__ as app_version   # local: the package imports the GUI lazily

    return {"format": TEMPLATE_FORMAT, "version": TEMPLATE_VERSION, "window": kind,
            "app_version": app_version,
            "settings": json.loads(json.dumps(dict(settings), allow_nan=False)),
            "appearance": PlotAppearance.from_mapping(appearance).to_dict()}


def decode_window_template(kind: str, payload: Any) -> tuple[dict, dict]:
    """``(settings, appearance)`` from a stored payload, or :class:`TemplateError`.

    A NEWER version is refused by number, not swallowed: a template that silently recalls as
    "nothing happened" is the bug that gets diagnosed as "the app is broken" (the Distributions
    reader's rule, ``PlotsWindow._template_config``).
    """
    if not isinstance(payload, Mapping) or payload.get("format") != TEMPLATE_FORMAT:
        raise TemplateError("It is not a NanoScope window template.")
    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise TemplateError("It declares no readable template version.")
    if version > TEMPLATE_VERSION:
        raise TemplateError(
            f"It was saved by a newer NanoScope (template version {version}; this version "
            f"reads up to {TEMPLATE_VERSION}).")
    if payload.get("window") != kind:
        label = WINDOW_KINDS.get(str(payload.get("window")), str(payload.get("window")))
        raise TemplateError(f"It is a template for {label}, not {WINDOW_KINDS[kind]}.")
    settings, appearance = payload.get("settings"), payload.get("appearance", {})
    if not isinstance(settings, Mapping) or not isinstance(appearance, Mapping):
        raise TemplateError("Its settings are damaged.")
    # STRICT JSON OR NOTHING. A project's manifest is read with plain ``json.loads``, so a
    # carried NaN or 1e400 arrives here as a float; the store writes with
    # ``allow_nan=False`` and would raise half-way through opening the project. Such a
    # template is unreadable (left in the project, named in the banner), never a crash.
    try:
        json.dumps(settings, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise TemplateError("Its settings hold a value that is not a plain number or "
                            "text (for example NaN or Infinity).") from None
    return dict(settings), PlotAppearance.from_mapping(appearance).to_dict()


# -- the shared controls --------------------------------------------------------------------------

class WindowTemplateControls(QtWidgets.QGroupBox):
    """Template combo, Save as template…, Rename and Delete, for ONE window kind.

    ``capture()`` returns the window's ``(settings, appearance)``; ``recall(settings,
    appearance)`` installs them. Both belong to the window, which alone knows which of its
    controls are a recipe and which are data. Picking a template uses ``activated``, a
    user-only signal, so rebuilding the combo can never recall one.
    """

    def __init__(self, kind: str, *, capture: Callable[[], tuple[Mapping, Mapping]],
                 recall: Callable[[dict, dict], None], note: str, parent=None) -> None:
        super().__init__("Templates", parent)
        _settings_key(kind)     # an unknown kind is a programming error: say so at build time
        self._kind = kind
        self._capture = capture
        self._recall = recall
        self.setObjectName("windowTemplateControls")
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Template:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.setAccessibleName(f"{WINDOW_KINDS[kind]} templates")
        self.combo.setToolTip("Your saved templates for this window. Picking one puts its "
                              "settings and appearance on the controls; your sources stay.")
        self.combo.activated.connect(self._activated)
        self.combo.currentIndexChanged.connect(lambda _i: self._sync_buttons())
        row.addWidget(self.combo, 1)
        layout.addLayout(row)
        buttons = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save as template…")
        self.save_button.setToolTip("Save this window's analysis settings and appearance under "
                                    "a name. Sources, time span and results are not saved.")
        self.save_button.clicked.connect(self._ask_save)
        self.rename_button = QtWidgets.QPushButton("Rename")
        self.rename_button.setToolTip("Rename the selected template. Its saved settings move "
                                      "unchanged — this is not a re-save of what is on screen.")
        self.rename_button.clicked.connect(self._ask_rename)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.setToolTip("Forget the selected template (asks first).")
        self.delete_button.clicked.connect(self._ask_delete)
        for button in (self.save_button, self.rename_button, self.delete_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        # THE SPLIT, ON SCREEN, as the Distributions template group states its own: what a
        # template carries and what pressing it does are short enough to say where it is used.
        self.note = QtWidgets.QLabel(note)
        self.note.setWordWrap(True)
        self.note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.note)
        self.refresh()
        # A project just opened may have added templates of this kind (plot_preset_sync).
        preset_store().changed.connect(self.refresh)

    # -- reading ---------------------------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(load_window_templates(self._kind), key=str.casefold)

    def selected(self) -> str | None:
        name = self.combo.currentData()
        return str(name) if name else None

    def refresh(self) -> None:
        """Rebuild the combo, keeping the selected name where it still exists."""
        selected = self.selected()
        with QtCore.QSignalBlocker(self.combo):
            self.combo.clear()
            self.combo.addItem(_PLACEHOLDER, None)
            for name in self.names():
                self.combo.addItem(name, name)
            self.combo.setCurrentIndex(max(0, self.combo.findData(selected)))
        self._sync_buttons()

    def showEvent(self, event) -> None:  # noqa: N802 -- Qt API
        # Another page of the same kind, or a project just opened, may have added one.
        self.refresh()
        super().showEvent(event)

    def _sync_buttons(self) -> None:
        on = self.selected() is not None
        self.rename_button.setEnabled(on)
        self.delete_button.setEnabled(on)

    def _select(self, name: str | None) -> None:
        with QtCore.QSignalBlocker(self.combo):
            self.combo.setCurrentIndex(max(0, self.combo.findData(name)))
        self._sync_buttons()

    # -- the operations (no dialogs: the dialog wrappers below call these) ------------------
    def save_template(self, name: str, *, overwrite: bool = False) -> bool:
        name = str(name).strip()[:NAME_LIMIT]
        templates = load_window_templates(self._kind)
        if not name or (name in templates and not overwrite):
            return False
        settings, appearance = self._capture()
        templates[name] = encode_window_template(self._kind, settings, appearance)
        ok = store_window_templates(self._kind, templates)
        self.refresh()
        self._select(name)
        return ok

    def recall_template(self, name: str) -> bool:
        """Install the template ``name``. Reports and returns False if it cannot be read."""
        payload = load_window_templates(self._kind).get(name)
        if payload is None:
            return False
        try:
            settings, appearance = decode_window_template(self._kind, payload)
            self._recall(settings, appearance)
        except (TemplateError, TypeError, ValueError, KeyError) as exc:
            self._report(f"“{name}” could not be recalled.\n\n{exc}")
            return False
        self._select(name)
        return True

    def rename_template(self, old: str, new: str, *, overwrite: bool = False) -> bool:
        """MOVE the stored settings to ``new``; never a re-save of what is on screen."""
        new = str(new).strip()[:NAME_LIMIT]
        templates = load_window_templates(self._kind)
        if old not in templates or not new or new == old or (new in templates and not overwrite):
            return False
        templates[new] = templates.pop(old)
        ok = store_window_templates(self._kind, templates)
        self.refresh()
        self._select(new)
        return ok

    def delete_template(self, name: str) -> bool:
        templates = load_window_templates(self._kind)
        if templates.pop(name, None) is None:
            return False
        ok = store_window_templates(self._kind, templates)
        self.refresh()
        self._select(None)
        return ok

    # -- the user's doors --------------------------------------------------------------------
    def _activated(self, index: int) -> None:
        name = self.combo.itemData(index)
        if name:
            self.recall_template(str(name))

    def _report(self, text: str) -> None:
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning,
                                    f"{WINDOW_KINDS[self._kind]} template", text,
                                    QtWidgets.QMessageBox.Ok, self)
        box.setTextFormat(QtCore.Qt.PlainText)
        box.exec()

    def _replace_ok(self, name: str, action: str) -> bool:
        answer = QtWidgets.QMessageBox.question(
            self, "Overwrite template?",
            f"A template named “{name}” already exists.\n\n{action} replaces its saved "
            f"settings. This cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return answer == QtWidgets.QMessageBox.Yes

    def _ask_save(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save template",
            "Name for this window's settings and appearance\n(sources and results are NOT saved):")
        name = (name or "").strip()[:NAME_LIMIT]
        if not ok or not name:
            return
        exists = name in load_window_templates(self._kind)
        if exists and not self._replace_ok(name, "Overwriting"):
            return
        self.save_template(name, overwrite=exists)

    def _ask_rename(self) -> None:
        old = self.selected()
        if old is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename template", f"New name for “{old}”:", text=old)
        new = (new or "").strip()[:NAME_LIMIT]
        if not ok or not new or new == old:
            return
        exists = new in load_window_templates(self._kind)
        if exists and not self._replace_ok(new, f"Renaming “{old}” onto it"):
            return
        self.rename_template(old, new, overwrite=exists)

    def confirm_delete(self, name: str) -> bool:
        """Ask before forgetting ``name``. True to delete. Cancel is the default.

        A template lives in the preferences: no Undo reaches it, so this question is the only
        thing between a stray click and the saved recipe (the Distributions rule,
        ``PlotsWindow._confirm_delete_template``). Separate so a test can answer it.
        """
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setTextFormat(QtCore.Qt.PlainText)   # the name is the user's own typing
        box.setWindowTitle("Delete template")
        box.setText(f"Delete the {WINDOW_KINDS[self._kind]} template “{name}”?")
        box.setInformativeText(
            "Its saved settings and appearance are forgotten on this computer. The plot on "
            "screen does not change. A project saved earlier keeps its own copy.")
        delete = box.addButton("Delete template", QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is delete

    def _ask_delete(self) -> None:
        name = self.selected()
        if name is not None and self.confirm_delete(name):
            self.delete_template(name)
