"""Help > User guide… and Help > Keyboard shortcuts…: the bundled guide in one read-only window.

The window shows ``docs/USER_GUIDE.md`` itself (a bundled copy in the app, the checkout's in a
source tree), is built on the first press, kept, modeless, and closed by the app's quit. The
second entry opens it at the level-2 heading "13. Keyboard shortcuts".
"""
from __future__ import annotations

import re
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui import user_guide_window as UGW
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.user_guide_window import (
    GUIDE_NAME, MISSING_SENTENCE, OUTSIDE_LINK_SENTENCE, SHORTCUTS_HEADING, UserGuideWindow,
    guide_available, read_guide,
)

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, private_settings, settle)

_CHECKOUT_GUIDE = Path(nanoscope.__file__).resolve().parents[1] / "docs" / GUIDE_NAME


@pytest.fixture
def window(qtbot):
    widget = MainWindow()
    qtbot.addWidget(widget)
    yield widget
    widget._analysis_controller.abort()
    widget._teardown()


def _guide_windows(owner) -> list:
    """The user-guide windows among the application's top-level widgets that ``owner`` owns."""
    return [w for w in QtWidgets.QApplication.topLevelWidgets()
            if isinstance(w, UserGuideWindow) and w.parent() is owner]


def _block_at_the_top(guide: UserGuideWindow) -> str:
    """The block of text that is actually drawn at the top of the view: asked of the browser's
    own hit test, so it does not repeat the arithmetic the window scrolls by."""
    return guide.browser.cursorForPosition(QtCore.QPoint(30, 12)).block().text()


def _link_point(guide: UserGuideWindow, href: str) -> QtCore.QPoint:
    """A point (in the view) on the first link to ``href``, scrolled into sight."""
    document = guide.browser.document()
    block = document.begin()
    while block.isValid():
        fragments = block.begin()
        while not fragments.atEnd():
            fragment = fragments.fragment()
            if fragment.charFormat().anchorHref() == href:
                cursor = QtGui.QTextCursor(document)
                cursor.setPosition(fragment.position() + 1)
                guide.browser.setTextCursor(cursor)
                guide.browser.ensureCursorVisible()
                return guide.browser.cursorRect(cursor).center()
            fragments += 1
        block = block.next()
    raise AssertionError(f"the guide has no link to {href!r}")


# ------------------------------------------------------------------------------------------------
# Where the text comes from
# ------------------------------------------------------------------------------------------------
def test_the_loader_prefers_the_bundled_copy_then_the_checkout_then_says_there_is_none(
        tmp_path, monkeypatch):
    assert read_guide() == _CHECKOUT_GUIDE.read_text(encoding="utf-8"), "the source-checkout fallback"
    assert guide_available()

    (tmp_path / GUIDE_NAME).write_text("# A bundled copy\n", encoding="utf-8")
    monkeypatch.setattr(UGW.resources, "files", lambda name: tmp_path)
    assert read_guide() == "# A bundled copy\n", "importlib.resources comes first"

    (tmp_path / GUIDE_NAME).unlink()
    assert read_guide() == _CHECKOUT_GUIDE.read_text(encoding="utf-8"), "no bundled copy: the checkout's"

    monkeypatch.setattr(UGW, "GUIDE_NAME", "no-such-guide.md")
    assert read_guide() is None
    assert not guide_available()

    (tmp_path / "unreadable.md").write_bytes(b"\xff\xfe not UTF-8 text \x80")
    monkeypatch.setattr(UGW, "GUIDE_NAME", "unreadable.md")
    assert read_guide() is None, "a guide that cannot be read is an absent one, not a crash at launch"


def test_the_window_shows_whatever_the_loader_found_and_the_sentence_when_nothing(
        qtbot, tmp_path, monkeypatch):
    (tmp_path / GUIDE_NAME).write_text(
        f"# A bundled copy\n\n## {SHORTCUTS_HEADING}\n\n| Shortcut | Action |\n|---|---|\n| Ctrl+Q | Quit |\n",
        encoding="utf-8")
    monkeypatch.setattr(UGW.resources, "files", lambda name: tmp_path)
    guide = UserGuideWindow()
    qtbot.addWidget(guide)
    assert "A bundled copy" in guide.browser.toPlainText() and "Ctrl+Q" in guide.browser.toPlainText()
    assert guide._shortcuts_block is not None

    monkeypatch.setattr(UGW, "GUIDE_NAME", "no-such-guide.md")
    empty = UserGuideWindow()
    qtbot.addWidget(empty)
    assert empty.browser.toPlainText() == MISSING_SENTENCE


# ------------------------------------------------------------------------------------------------
# The two Help entries
# ------------------------------------------------------------------------------------------------
def _help_texts(window) -> list[str]:
    # findChildren, never QAction.menu(): that hands the QMenu to Python, which deletes it with the wrapper.
    menus = {m.title().replace("&", ""): m for m in window.menuBar().findChildren(QtWidgets.QMenu)}
    return [a.text() for a in menus["Help"].actions()]


def test_the_help_menu_leads_with_the_two_new_entries_and_keeps_the_rest(window):
    assert _help_texts(window) == [
        "User guide…", "Keyboard shortcuts…", "",
        "Guided first analysis…", "Analysis home…", "",
        "About NanoScope…"]
    for act in (window._act_user_guide, window._act_keyboard_shortcuts):
        assert act.isEnabled()
        tip = act.toolTip()
        assert tip[0].isupper() and tip.endswith("."), f"a written sentence, not a label echo: {tip!r}"


def test_both_entries_are_disabled_with_a_sentence_when_the_guide_is_absent(qtbot, monkeypatch):
    monkeypatch.setattr(UGW, "GUIDE_NAME", "no-such-guide.md")      # the loader now looks for a missing file
    missing = MainWindow()
    qtbot.addWidget(missing)
    try:
        for act, label in ((missing._act_user_guide, "User guide"),
                           (missing._act_keyboard_shortcuts, "Keyboard shortcuts")):
            assert not act.isEnabled()
            # Said where a native macOS menu shows it (the label) and where a Qt-drawn one does.
            assert act.text() == f"{label} — not included in this copy"
            assert act.toolTip() == MISSING_SENTENCE
        assert MISSING_SENTENCE[0].isupper() and MISSING_SENTENCE.endswith(".")
        assert _help_texts(missing)[:3] == [
            "User guide — not included in this copy", "Keyboard shortcuts — not included in this copy", ""]
        assert missing._user_guide_window is None
    finally:
        missing._analysis_controller.abort()
        missing._teardown()


def test_no_guide_window_is_built_until_the_first_press(window):
    assert window._user_guide_window is None
    assert _guide_windows(window) == []
    assert window.findChildren(UserGuideWindow) == []


# ------------------------------------------------------------------------------------------------
# Opening, reusing, closing
# ------------------------------------------------------------------------------------------------
def test_user_guide_opens_at_the_top_and_keyboard_shortcuts_at_section_13(window, qapp):
    window._act_keyboard_shortcuts.trigger()
    qapp.processEvents()
    guide = window._user_guide_window
    assert guide.isVisible() and guide.parent() is window
    block = guide.browser.document().findBlockByNumber(guide._shortcuts_block)
    assert block.text() == SHORTCUTS_HEADING and block.blockFormat().headingLevel() == 2
    assert guide.browser.verticalScrollBar().value() > 0
    assert _block_at_the_top(guide) == SHORTCUTS_HEADING

    window._act_user_guide.trigger()
    qapp.processEvents()
    assert window._user_guide_window is guide, "one window, reused"
    assert guide.browser.verticalScrollBar().value() == 0
    assert _block_at_the_top(guide) == "NanoScope — User Guide"

    window._act_keyboard_shortcuts.trigger()
    qapp.processEvents()
    assert window._user_guide_window is guide and _block_at_the_top(guide) == SHORTCUTS_HEADING
    assert len(window.findChildren(UserGuideWindow)) == 1 and _guide_windows(window) == [guide]


def test_it_is_one_modeless_read_only_window_that_a_close_only_hides(window, qtbot):
    window._act_user_guide.trigger()
    guide = window._user_guide_window
    assert not guide.isModal()
    assert not guide.testAttribute(QtCore.Qt.WA_DeleteOnClose)
    assert guide.browser.isReadOnly()
    before = guide.browser.toPlainText()
    qtbot.keyClicks(guide.browser, "typed into the guide")
    assert guide.browser.toPlainText() == before, "read-only"
    guide.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    QtCore.QCoreApplication.processEvents()
    assert shiboken6.isValid(guide) and guide.isHidden()
    window._act_keyboard_shortcuts.trigger()
    assert window._user_guide_window is guide and guide.isVisible()


def test_the_apps_quit_closes_it(qtbot):
    app_window = MainWindow()
    qtbot.addWidget(app_window)
    app_window._act_user_guide.trigger()
    guide = app_window._user_guide_window
    assert guide.isVisible()
    app_window._dirty = False
    app_window.close()
    assert guide.isHidden(), "the guide must not outlive the window that owns it"


def test_the_window_menu_lists_it_once_opened_and_its_place_is_remembered(private_settings, qtbot):
    window = build_window(qtbot, detection=False)

    def entries() -> dict[str, tuple[bool, bool]]:
        window._window_menu.aboutToShow.emit()
        return {a.text(): (a.isChecked(), a.isEnabled())
                for a in window._window_menu.actions() if a.isVisible() and not a.isSeparator()}

    assert "User guide" not in entries()
    window._open_user_guide()
    settle()
    assert entries()["User guide"] == (True, True)
    guide = window._user_guide_window
    guide.setGeometry(14, 90, 700, 500)
    settle()
    want = guide.geometry()
    guide.close()
    settle()
    assert entries()["User guide"] == (False, True)
    close_window(window)
    assert QtCore.QSettings().contains("windows/user_guide/geometry")

    again = build_window(qtbot, detection=False)
    again._open_keyboard_shortcuts()
    settle()
    got = again._user_guide_window.geometry()
    assert all(abs(a - b) <= 2 for a, b in zip(got.getRect(), want.getRect())), (got, want)
    again._user_guide_window.hide()
    close_window(again)


# ------------------------------------------------------------------------------------------------
# What it draws
# ------------------------------------------------------------------------------------------------
def test_every_link_to_a_heading_of_the_guide_resolves(window):
    window._act_user_guide.trigger()
    guide = window._user_guide_window
    slugs = re.findall(r"\]\(#([^)]+)\)", _CHECKOUT_GUIDE.read_text(encoding="utf-8"))
    assert len(slugs) > 15, "the contents list"
    assert [s for s in slugs if s not in guide._anchors] == []


def test_a_click_on_a_contents_entry_scrolls_to_its_heading_and_an_outside_link_only_says_why(
        window, qapp, qtbot):
    window._act_user_guide.trigger()
    guide = window._user_guide_window
    qapp.processEvents()
    viewport = guide.browser.viewport()

    qtbot.mouseClick(viewport, QtCore.Qt.LeftButton, pos=_link_point(guide, "#13-keyboard-shortcuts"))
    qapp.processEvents()
    assert _block_at_the_top(guide) == SHORTCUTS_HEADING

    window._act_user_guide.trigger()
    qapp.processEvents()
    before_text = guide.browser.toPlainText()
    point = _link_point(guide, "OPEN_PORE_REFERENCE.md")
    position = guide.browser.verticalScrollBar().value()
    QtWidgets.QToolTip.hideText()
    assert QtWidgets.QToolTip.text() == ""
    qtbot.mouseClick(viewport, QtCore.Qt.LeftButton, pos=point)
    qapp.processEvents()
    assert QtWidgets.QToolTip.text() == OUTSIDE_LINK_SENTENCE
    assert guide.browser.toPlainText() == before_text, "an outside link must not replace the guide"
    assert guide.browser.verticalScrollBar().value() == position


def test_a_link_pressed_from_the_keyboard_says_why_at_the_link_not_at_the_pointer(
        window, qapp, qtbot, monkeypatch):
    """Tab onto a link and press Return: the pointer can be anywhere, so the sentence must not be
    placed there. The first link of the guide (the illustrated manual) is one of these."""
    window._act_user_guide.trigger()
    guide = window._user_guide_window
    guide.move(400, 300)
    qapp.processEvents()
    browser, viewport = guide.browser, guide.browser.viewport()
    shown = []
    monkeypatch.setattr(QtWidgets.QToolTip, "showText",
                        lambda pos, text, widget=None, *rest: shown.append((pos, text)))
    monkeypatch.setattr(QtGui.QCursor, "pos", staticmethod(lambda: QtCore.QPoint(3, 3)))
    assert not viewport.underMouse(), "precondition: the pointer is not over the guide"
    browser.setFocus()
    qtbot.keyClick(browser, QtCore.Qt.Key_Tab)             # onto the first link, "illustrated manual"
    on_link = QtGui.QTextCursor(browser.textCursor())
    on_link.setPosition(on_link.selectionStart())
    on_link.movePosition(QtGui.QTextCursor.NextCharacter)
    assert on_link.charFormat().anchorHref() == "manual.html", "precondition: Tab reached a link"
    qtbot.keyClick(browser, QtCore.Qt.Key_Return)
    assert [text for _pos, text in shown] == [OUTSIDE_LINK_SENTENCE]
    (pos, _text), = shown
    shown_over = QtCore.QRect(viewport.mapToGlobal(QtCore.QPoint(0, 0)), viewport.size())
    assert shown_over.contains(pos), f"the sentence went to {pos}, outside the guide's text {shown_over}"


def test_links_are_drawn_in_the_accent_colour_not_qts_default_blue(window):
    window._act_user_guide.trigger()
    browser = window._user_guide_window.browser
    accent = browser.palette().color(QtGui.QPalette.ColorRole.Highlight)
    qt_default = QtGui.QGuiApplication.palette().color(QtGui.QPalette.ColorRole.Link)
    assert accent != qt_default, "precondition: the test palette must tell the two apart"
    colours = set()
    block = browser.document().begin()
    while block.isValid():
        fragments = block.begin()
        while not fragments.atEnd():
            fmt = fragments.fragment().charFormat()
            if fmt.isAnchor():
                colours.add(fmt.foreground().color().name())
            fragments += 1
        block = block.next()
    assert len(colours) > 0 and colours == {accent.name()}
