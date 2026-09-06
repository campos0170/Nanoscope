"""Presentation-only workflow and warning chrome shared by analytical pages."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6 import QtCore, QtWidgets


SECTIONS = ("Data", "Analysis", "Appearance", "Export")


class PlotWorkflowBar(QtWidgets.QWidget):
    """Switch control visibility without invoking any calculation or changing values."""

    sectionChanged = QtCore.Signal(str)

    def __init__(self, *, actions: Sequence[tuple[str, Callable]] = (),
                 exports: Sequence[tuple[str, Callable]] = (), parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sharedPlotWorkflow")
        self.setAccessibleName("Plot workflow")
        self._sections: dict[str, tuple[QtWidgets.QWidget, ...]] = {}
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Section"))
        self.sections = QtWidgets.QComboBox()
        self.sections.addItems(SECTIONS)
        self.sections.setAccessibleName("Plot control section")
        self.sections.setToolTip("Data → Analysis → Appearance → Export. Changing section only "
                                 "shows controls; it never analyzes or changes your data.")
        self.sections.currentTextChanged.connect(self._show_section)
        layout.addWidget(self.sections)
        layout.addStretch(1)
        self.primary_actions: list[QtWidgets.QPushButton] = []
        for title, callback in actions:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, fn=callback: fn())
            button.setAccessibleName(title)
            button.setToolTip("Explicit analysis action. Appearance and section changes do not run it.")
            layout.addWidget(button)
            self.primary_actions.append(button)
        self.export_button = QtWidgets.QToolButton()
        self.export_button.setText("Export ▾")
        self.export_button.setAccessibleName("Export current plot")
        self.export_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.export_menu = QtWidgets.QMenu(self.export_button)
        for title, callback in exports:
            self.export_menu.addAction(title, callback)
        self.export_button.setMenu(self.export_menu)
        self.export_button.setEnabled(bool(exports))
        layout.addWidget(self.export_button)

    def set_sections(self, mapping: Mapping[str, Sequence[QtWidgets.QWidget]]) -> None:
        self._sections = {name: tuple(mapping.get(name, ())) for name in SECTIONS}
        self._show_section(self.sections.currentText())

    def set_section(self, name: str) -> None:
        if name not in SECTIONS:
            raise ValueError(f"Unknown plot section: {name}")
        if self.sections.currentText() == name:
            self._show_section(name)
        else:
            self.sections.setCurrentText(name)

    def _show_section(self, name: str) -> None:
        shown = set(self._sections.get(name, ()))
        all_widgets = {w for widgets in self._sections.values() for w in widgets}
        for widget in all_widgets:
            widget.setVisible(widget in shown)
        self.sectionChanged.emit(name)


def install_plot_workflow(window: QtWidgets.QMainWindow, *, sections, actions=(), exports=()):
    """Attach the same fixed workflow bar to a standalone or Explorer-embedded page."""
    toolbar = QtWidgets.QToolBar("Plot workflow", window)
    toolbar.setObjectName("sharedPlotWorkflowToolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    bar = PlotWorkflowBar(actions=actions, exports=exports)
    bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    toolbar.addWidget(bar)
    window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
    bar.set_sections(sections)
    return bar


def control_group(title: str, text: str) -> QtWidgets.QGroupBox:
    """A small explanatory section when the page consumes a fixed analysis source."""
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(box)
    label = QtWidgets.QLabel(text)
    label.setTextFormat(QtCore.Qt.PlainText)
    label.setWordWrap(True)
    layout.addWidget(label)
    return box


def bounded_controls(widget: QtWidgets.QWidget, maximum_height=260) -> QtWidgets.QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setMinimumHeight(80)
    scroll.setMaximumHeight(maximum_height)
    return scroll


def ensure_item_legend(plot, entries) -> None:
    """Provide an opt-in legend while preserving the historical hidden default."""
    pairs = tuple((item, name) for item, name in entries if item is not None)
    legend = getattr(plot, "legend", None)
    if legend is None:
        legend = plot.addLegend()
        legend.hide()
    if getattr(plot, "_nanoscope_workflow_legend_pairs", None) == pairs:
        return
    legend.clear()
    for item, name in pairs:
        legend.addItem(item, name)
    plot._nanoscope_workflow_legend_pairs = pairs


class WarningSummary(QtWidgets.QWidget):
    """A persistent first warning/count, with every original word in bounded detail."""

    def __init__(self, parent=None, *, title="Warnings", maximum_height=150) -> None:
        super().__init__(parent)
        self._title = title
        self._full = ""
        self.setObjectName("plotWarningSummary")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        self.summary = QtWidgets.QLabel()
        self.summary.setTextFormat(QtCore.Qt.PlainText)
        self.summary.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.summary.setWordWrap(True)
        self.summary.setMaximumHeight(48)
        self.summary.setStyleSheet("color:#ffd479;")
        self.summary.setAccessibleName(title + " summary")
        row.addWidget(self.summary, 1)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText("Full details")
        self.toggle.setCheckable(True)
        self.toggle.setArrowType(QtCore.Qt.RightArrow)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        row.addWidget(self.toggle)
        layout.addLayout(row)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(maximum_height)
        self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.details.setAccessibleName("Full " + title.lower())
        self.details.hide()
        layout.addWidget(self.details)
        self.toggle.toggled.connect(self._expand)
        self.hide()

    def _expand(self, checked: bool) -> None:
        self.details.setVisible(checked)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def set_warnings(self, warnings: Sequence[str], *, full_text: str | None = None) -> None:
        blocks = [str(w) for w in warnings if str(w).strip()]
        self._full = "\n\n".join(blocks) if full_text is None else str(full_text)
        self.details.setPlainText(self._full)
        first = " ".join(blocks[0].split()) if blocks else ""
        # The count and explicit disclosure survive truncation; full text is never lost.
        if len(first) > 170:
            first = first[:167].rstrip() + "…"
        self.summary.setText(f"⚠ {self._title} ({len(blocks)}): {first}" if blocks else "")
        self.summary.setToolTip(self._full)
        self.setVisible(bool(self._full))

    def full_text(self) -> str:
        return self._full

    # Compatibility for existing warning-label read APIs: the retained text, not
    # the visually abbreviated summary, remains the authoritative warning string.
    def text(self) -> str:
        return self._full

    def setText(self, text: str) -> None:
        self.set_warnings((str(text),) if text else (), full_text=str(text))
