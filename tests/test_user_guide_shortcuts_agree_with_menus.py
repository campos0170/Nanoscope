"""Every shortcut on a menu action is in the user guide's section 13, which Help > Keyboard
shortcuts… opens, in the row that names that action.

That window shows the guide itself, so section 13 is the one table of shortcuts a user can reach
from the app. Written the way ``tests/test_s7_fixup_user_guide_agrees_with_app.py`` is: read the
live menus first, then check the document against what they say, so a shortcut added or changed on
a menu without a row in the guide fails here rather than going undocumented. The row is looked up
by the action it names, not by the key alone: a key that moves onto one another row already lists
for something else is a wrong guide, not a documented shortcut.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
from pathlib import Path

from PySide6 import QtGui, QtWidgets

from nanoscope.gui.main_window import MainWindow

_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "USER_GUIDE.md"

#: Close, Quit and Redo are bound to a platform STANDARD key, and which sequence that is depends on
#: the platform theme: the generic one the offscreen suite runs under answers Ctrl+F4, the key
#: named "Exit" and Ctrl+Y, where a Mac answers ⌘W, ⌘Q and ⇧⌘Z. The guide spells the reading its
#: readers have.
_STANDARD_KEYS_AS_THE_GUIDE_SPELLS_THEM = {
    QtGui.QKeySequence.StandardKey.Close: "Ctrl+W",
    QtGui.QKeySequence.StandardKey.Quit: "Ctrl+Q",
    QtGui.QKeySequence.StandardKey.Redo: "Ctrl+Shift+Z",
}


def _section_13() -> str:
    guide = _GUIDE.read_text(encoding="utf-8")
    found = re.search(r"^## 13\. Keyboard shortcuts$(.*?)^## ", guide, re.S | re.M)
    assert found, "the guide no longer has a '## 13. Keyboard shortcuts' section"
    text = found.group(1)
    # The guide writes Ctrl and ⌘ as "Ctrl/⌘" and the arrow keys as arrows; Qt writes Left, Right...
    for written, spelled in (("Ctrl/⌘", "Ctrl"), ("←", "Left"), ("→", "Right"), ("↑", "Up"), ("↓", "Down")):
        text = text.replace(written, spelled)
    return text


#: ``Menu > action`` -> what the Action cell of that action's row in section 13 begins with
#: (a regular expression, matched at the start of the cell with the markdown emphasis taken out).
#: A shortcut added to a menu needs its row in the guide AND its entry here, and an entry whose
#: shortcut is gone must go with its row: both are checked below.
_ROW_FOR = {
    "File > Open…": r"Open file$",
    "File > Save Project…": r"Save project$",
    "File > Save Project As…": r"Save project as",
    "File > Close document": r"Close current document",
    "File > Quit": r"Quit NanoScope",
    "Edit > Undo": r"Undo the last change",
    "Edit > Redo": r"Redo it$",
    "View > Zoom Back": r"Zoom Back / Forward",
    "View > Zoom Forward": r"Zoom Back / Forward",
    "View > Reset zoom": r"Reset zoom",
    "View > Centre selection in view": r"Centre the view on the current selection",
    "View > Go to event…": r"Go to event…",
    "View > Go to time…": r"Go to time…",
    "View > Annotate: drag adds an event": r"Annotate mode: a drag",
    "Panels > Results": r"Show / hide the Results panel",
    "Analysis > Run detection": r"Run detection$",
    "Analysis > Event inspector…": r"Open the Event inspector$",
    "Analysis > Measure event": r"Analysis ▸ Measure event on or off",
    "Analysis > Add event from cursors": r"Add the current cursor region as a hand-made event",
    "Analysis > Event gallery…": r"Open the Event gallery$",
    "Analysis > Preprocess workspace…": r"Open the Preprocess workspace$",
}


def _rows() -> list[tuple[str, str]]:
    """The (Shortcut cell, Action cell) of every row of section 13's table, emphasis taken out."""
    rows = []
    for line in _section_13().splitlines():
        if not line.startswith("|"):
            continue
        _, shortcut, action, _ = line.split("|")       # a row with a third cell fails here, loudly
        if set(shortcut.strip()) <= set("-: "):
            continue                                   # the |---|---| rule under the header
        rows.append((shortcut.replace("**", "").strip(), " ".join(action.replace("**", "").split())))
    return rows


def _holds(shortcut_cell: str, key: str) -> bool:
    return bool(re.search(r"(?<![\w+])" + re.escape(key) + r"(?!\w)", shortcut_cell))


def _menu_shortcuts(window: MainWindow) -> dict[str, list[str]]:
    """``Menu > action`` -> its shortcuts, as the guide spells them, for every menu and submenu.

    The menus are reached through ``findChildren``, never ``QAction.menu()``: that hands the QMenu
    to Python, which deletes it with the wrapper."""
    found = {}
    for menu in window.menuBar().findChildren(QtWidgets.QMenu):
        for action in menu.actions():
            spelled = [_as_the_guide_spells_it(s) for s in action.shortcuts()]
            if spelled:
                found[f"{menu.title().replace('&', '')} > {action.text().replace('&', '')}"] = spelled
    return found


def _as_the_guide_spells_it(sequence: QtGui.QKeySequence) -> str:
    for key, spelled in _STANDARD_KEYS_AS_THE_GUIDE_SPELLS_THEM.items():
        if sequence == QtGui.QKeySequence(key):
            return spelled
    return sequence.toString(QtGui.QKeySequence.SequenceFormat.PortableText)


def test_every_menu_shortcut_has_its_row_in_section_13(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    shortcuts = _menu_shortcuts(window)
    window._analysis_controller.abort()
    window._teardown()

    # Not vacuous: the menus carry a couple of dozen shortcuts, and the one the table once left
    # out (Quit) is among those checked.
    assert len(shortcuts) >= 20, sorted(shortcuts)
    assert shortcuts["File > Quit"] == ["Ctrl+Q"]
    assert shortcuts["File > Open…"] == ["Ctrl+O"]
    assert shortcuts["View > Zoom Back"] == ["Ctrl+Left", "Alt+Left"]

    new = sorted(set(shortcuts) - set(_ROW_FOR))
    assert not new, f"menu actions with a shortcut and no entry in _ROW_FOR (add their row to §13 too): {new}"
    gone = sorted(set(_ROW_FOR) - set(shortcuts))
    assert not gone, f"_ROW_FOR entries whose menu action has no shortcut now (drop the row from §13 too): {gone}"

    rows = _rows()
    assert len(rows) > 40, "the table was not read"
    missing = {}
    for name, spellings in shortcuts.items():
        action = re.compile(_ROW_FOR[name])
        absent = [s for s in spellings
                  if not any(_holds(key, s) and action.match(what) for key, what in rows)]
        if absent:
            missing[name] = absent
    assert not missing, f"menu shortcuts that section 13 does not give, in the row that names the action: {missing}"
