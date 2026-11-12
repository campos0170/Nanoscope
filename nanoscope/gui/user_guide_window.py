"""The user guide in a window of its own: Help ▸ User guide… and Help ▸ Keyboard shortcuts….

What it shows is ``docs/USER_GUIDE.md`` itself, as it is, so the shortcuts and everything else the
guide says have no second copy in the code to drift away from it.
"""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

#: The file's name: in the bundle it sits next to ``_build_info.json`` (both PyInstaller specs add
#: it to the ``nanoscope`` folder), in a source checkout it is in ``docs/``.
GUIDE_NAME = "USER_GUIDE.md"
#: The level-2 heading "Keyboard shortcuts…" opens the guide at.
SHORTCUTS_HEADING = "13. Keyboard shortcuts"
MISSING_SENTENCE = "This copy of NanoScope does not include the user guide."
OUTSIDE_LINK_SENTENCE = "That link points outside the guide, so it does not open here."


def _candidates():
    """Where the guide may be, best first: the bundled copy, then ``docs/`` of a source checkout."""
    try:
        yield resources.files("nanoscope").joinpath(GUIDE_NAME)
    except (ModuleNotFoundError, TypeError):
        pass
    yield Path(__file__).resolve().parents[2] / "docs" / GUIDE_NAME


def read_guide() -> str | None:
    """The guide's text, or None when this copy of the app has none."""
    for source in _candidates():
        try:
            return source.read_text(encoding="utf-8")
        except (OSError, ValueError):      # absent, unreadable, or not UTF-8 text
            continue
    return None


def guide_available() -> bool:
    return read_guide() is not None


def _slug(heading: str) -> str:
    """The name a "#..." link in the guide gives a heading (the GitHub rule the contents list
    was written to): lower case, punctuation dropped, each space a hyphen."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


class UserGuideWindow(QtWidgets.QDialog):
    """The guide, read-only. Modeless and owned by the main window, which builds it on the first
    Help press and keeps that one (``MainWindow._user_guide_window``). Never deleted on close,
    for the reason given in ``build_info_dialog.py``: a close is a hide, so the main window's
    quit closes it with everything else and its place is remembered."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NanoScope — User guide")
        self.setModal(False)
        self.resize(900, 720)
        self.setMinimumSize(480, 360)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setAccessibleName("User guide")
        self.browser.setReadOnly(True)
        # Links are followed here, not by Qt: the guide's contents list points at headings Qt gives
        # no anchors, and its other links name files and pages that are not in the app.
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self._follow_link)
        text = read_guide()
        if text is None:
            self.browser.setPlainText(MISSING_SENTENCE)
        else:
            self.browser.setMarkdown(text)
        layout.addWidget(self.browser)
        self._recolour_links()
        self._index_headings()

    def _recolour_links(self) -> None:
        """Paint the links in the accent colour. Qt's Markdown import bakes the application
        palette's Link colour into each one, and the app's theme leaves that at the default dark
        blue, which is close to unreadable on its dark background."""
        document = self.browser.document()
        spans = []
        block = document.begin()
        while block.isValid():
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if fragment.charFormat().isAnchor():
                    spans.append((fragment.position(), fragment.length()))
                fragments += 1
            block = block.next()
        link = QtGui.QTextCharFormat()
        link.setForeground(self.browser.palette().color(QtGui.QPalette.ColorRole.Highlight))
        document.setUndoRedoEnabled(False)
        for start, length in spans:
            cursor = QtGui.QTextCursor(document)
            cursor.setPosition(start)
            cursor.setPosition(start + length, QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(link)

    def _index_headings(self) -> None:
        """Find the headings by their block, not by searching the text: the contents list holds
        the same words as the heading it links to."""
        self._anchors: dict[str, int] = {}
        self._shortcuts_block: int | None = None
        block = self.browser.document().begin()
        while block.isValid():
            level = block.blockFormat().headingLevel()
            if level:
                heading = block.text().strip()
                self._anchors.setdefault(_slug(heading), block.blockNumber())
                if level == 2 and heading == SHORTCUTS_HEADING:
                    self._shortcuts_block = block.blockNumber()
            block = block.next()

    def show_guide(self) -> None:
        """Open, or bring forward, at the beginning."""
        self._present(0)

    def show_shortcuts(self) -> None:
        """Open, or bring forward, at the keyboard shortcuts."""
        self._present(self._shortcuts_block or 0)

    def _present(self, block_number: int) -> None:
        # Shown first: where a heading sits depends on the width the text is laid out at.
        self.show()
        self.raise_()
        self.activateWindow()
        self._scroll_to(block_number)

    def _scroll_to(self, block_number: int) -> None:
        document = self.browser.document()
        top = 0.0
        if block_number:
            block = document.findBlockByNumber(block_number)
            top = document.documentLayout().blockBoundingRect(block).top() - document.documentMargin()
        self.browser.verticalScrollBar().setValue(max(0, int(top)))

    def _follow_link(self, url: QtCore.QUrl) -> None:
        """A link to a heading of this guide scrolls to it; any other does nothing but say why."""
        number = None
        if not url.scheme() and not url.path() and url.hasFragment():
            number = self._anchors.get(url.fragment())
        if number is None:
            QtWidgets.QToolTip.showText(self._link_sentence_position(), OUTSIDE_LINK_SENTENCE, self.browser)
        else:
            self._scroll_to(number)

    def _link_sentence_position(self) -> QtCore.QPoint:
        """Where the sentence for an outside link is shown: at the pointer when it is over the text,
        which is where a click was. A link reached by Tab and pressed with Return leaves the pointer
        anywhere, perhaps outside the window, so the sentence goes to the text cursor, which the key
        moved onto the link."""
        viewport = self.browser.viewport()
        if viewport.underMouse():
            return QtGui.QCursor.pos()
        return viewport.mapToGlobal(self.browser.cursorRect().bottomLeft())
