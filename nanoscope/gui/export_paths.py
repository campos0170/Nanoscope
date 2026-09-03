"""THE save-dialog seam: no widget in this package may ask for an output path any other way.

WHY THIS MODULE EXISTS
----------------------
``QFileDialog.getSaveFileName`` confirms an overwrite of **the one name the user typed**.
That confirmation is only truthful for an export that writes exactly that file, and
several of ours do not:

* *Analysis ▸ Export figure…* proposed ``figures.pdf`` — a file it **never writes** — and
  then wrote ``{stem}_{dwell,depth,scatter,trace}.{pdf,png,csv}``: up to **twelve** files,
  none of them mentioned, silently overwriting whatever was there. The ``.csv`` source-data
  files are the ones a user is most likely to have hand-edited.
* *Export events for training…* proposed ``events_training.csv`` and also wrote
  ``events_training.meta.json`` beside it, unmentioned.

Patching those two commands would leave the next export free to reintroduce the defect, so
the guarantee is moved to where every export must pass:

    the path a GUI export writes to is obtained from :func:`ask_export_path`, which is
    handed a callable that enumerates the REAL output set and confirms whatever part of
    that set already exists.

``tests/gui/test_export_overwrite.py`` holds an AST guard that fails if any module in
``nanoscope/gui`` calls ``getSaveFileName`` outside this file. That guard, not this
docstring, is what keeps the seam.

The native overwrite prompt is suppressed on purpose (``DontConfirmOverwrite``). Leaving it
on would ask about one file and then this module would ask about the real set — two prompts,
the first of them misleading. One prompt, naming what is actually at stake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

from PySide6 import QtWidgets

__all__ = ["ask_export_path", "existing_of", "confirm_overwrite"]

#: What a caller passes to describe its outputs: chosen path -> every path it may write.
OutputsFn = Callable[[Path], Sequence[Path]]


def existing_of(paths: Iterable[Path]) -> list[Path]:
    """The subset of ``paths`` that already exists on disk, de-duplicated, in order.

    Directories are excluded: a write to one fails outright rather than destroying it,
    so naming it in an overwrite prompt would be a false alarm.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)
        try:
            if p.is_file():
                out.append(p)
        except OSError:      # unreadable parent, weird mount — treat as "not there"
            continue
    return out


def confirm_overwrite(parent: QtWidgets.QWidget | None, title: str,
                      existing: Sequence[Path]) -> bool:
    """Ask before replacing ``existing``. True to proceed.

    Names every file, because "3 files will be replaced" is not something a user can
    check. Module-level (not a method) so a test can substitute it without a live modal.
    """
    if not existing:
        return True
    names = "\n".join(f"  {p.name}" for p in existing)
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(f"{len(existing)} existing file(s) will be replaced.")
    box.setInformativeText(
        f"This export writes more than the name you chose. Already in "
        f"{existing[0].parent}:\n\n{names}\n\nReplace them?")
    replace = box.addButton("Replace", QtWidgets.QMessageBox.DestructiveRole)
    cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
    # Name the buttons, never index them. ``buttons()`` returns LAYOUT order, not insertion
    # order, and on this platform layout order put the destructive button last — so
    # ``buttons()[-1]`` made Replace the Enter-key default on a prompt whose whole job is to
    # stop an accidental overwrite, and left no escape button at all. Return and Esc must both
    # keep the user's files.
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.exec()
    return box.clickedButton() is replace


def ask_export_path(
    parent: QtWidgets.QWidget | None,
    *,
    title: str,
    suggestion: str,
    filters: str,
    outputs: OutputsFn | None = None,
) -> Path | None:
    """Run the save dialog and confirm the export's REAL output set. None == user backed out.

    Args:
        parent: dialog parent.
        title: dialog + confirmation window title.
        suggestion: pre-filled name. For a multi-file export this is a BASE NAME, and
            should be phrased as one — the user is not choosing a single file.
        filters: Qt name filter string.
        outputs: chosen path -> every path the export may write. ``None`` means the
            export writes exactly the chosen path and nothing else; pass a real
            enumerator (``figures.figure_output_paths``,
            ``scoring.training_export_paths``, …) for anything that writes siblings.

    Returns:
        The path the user chose (verbatim — normalising it is the caller's job and must
        match what ``outputs`` assumed), or None when the dialog or the overwrite
        confirmation was declined.
    """
    chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, title, suggestion, filters,
        options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
    if not chosen:
        return None
    path = Path(chosen)
    targets = [Path(p) for p in (outputs(path) if outputs is not None else (path,))]
    if not confirm_overwrite(parent, title, existing_of(targets)):
        return None
    return path
