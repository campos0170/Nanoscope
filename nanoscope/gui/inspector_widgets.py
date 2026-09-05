"""Layout-only building blocks for the inspector; never own analysis parameters."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class InspectorScrollArea(QtWidgets.QScrollArea):
    """A bounded viewport: another page's content cannot dictate its height."""

    def __init__(self, content: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumSize(0, 0)
        self.setWidget(content)

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(240, 320)

    def minimumSizeHint(self):  # noqa: N802
        return QtCore.QSize(0, 0)


class Disclosure(QtWidgets.QWidget):
    """Show/hide an existing widget without disabling it or changing its values."""

    def __init__(self, title: str, content: QtWidgets.QWidget, *, expanded=False, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.toggle.setStyleSheet('QToolButton { text-align: left; padding: 4px; }')
        self.content = content
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool):
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.content.setVisible(expanded)


class PinnedActions(InspectorScrollArea):
    """Keep actions above vertical scrolling without imposing a dock-width floor.

    At exceptionally narrow saved widths the row can scroll horizontally. Its
    scrollbar gets its own height, so it never paints over the buttons.
    """

    def __init__(self, content: QtWidgets.QWidget, parent=None):
        super().__init__(content, parent)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        content.installEventFilter(self)
        self._sync_height()

    def _sync_height(self):
        content = self.widget()
        height = content.sizeHint().height()
        if content.minimumSizeHint().width() > self.viewport().width():
            height += self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(max(0, height))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._sync_height()

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QtCore.QEvent.LayoutRequest:
            self._sync_height()
        return super().eventFilter(watched, event)

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(0, self.height())


def host_layout(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    """Transfer an existing layout and its live controls to a plain container."""
    host = QtWidgets.QWidget()
    host.setLayout(layout)
    return host
