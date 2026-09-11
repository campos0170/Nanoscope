"""Reusable source-label/color editor with explicit figure and project scopes."""
from __future__ import annotations

from copy import deepcopy

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.source_style import MAX_LABEL_LENGTH, validate_source_appearance


class SourceAppearanceEditor(QtWidgets.QGroupBox):
    """Emit only committed presentation changes, never analysis requests.

    ``set_sources`` accepts PlotSources or snapshot mappings (key/label/color).
    ``changed`` carries the complete figure-local override mapping. Project
    defaults use Session's source-appearance mutator; saved snapshots stay pinned.
    """

    changed = QtCore.Signal(dict)

    def __init__(self, parent=None, *, session=None):
        super().__init__("Source names and colors", parent)
        self._session = session
        self._sources = {}
        self._state = {}
        self._color_override = None
        self._reset_project_color = False
        layout = QtWidgets.QGridLayout(self)
        self.source = QtWidgets.QComboBox()
        self.source.setAccessibleName("Source appearance identity")
        self.source.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(15)
        self.label = QtWidgets.QLineEdit()
        self.label.setMaxLength(MAX_LABEL_LENGTH)
        self.label.setAccessibleName("Source display label")
        self.label.setPlaceholderText("Captured source label (clear to restore)")
        self.label.setToolTip("Clear a figure override to restore the label captured with this figure. "
                              "Saving a blank project default clears the project alias for future figures.")
        self.color = QtWidgets.QPushButton("Choose color…")
        self.color.setAccessibleName("Source color")
        self.reset_color = QtWidgets.QPushButton("Captured color")
        self.reset_color.setToolTip("Apply to figure restores its captured source color. "
                                    "Use as project source default resets the project's palette color for future figures; "
                                    "the current frozen figure retains its captured color.")
        self.apply = QtWidgets.QPushButton("Apply to figure")
        self.project = QtWidgets.QPushButton("Use as project source default")
        self.project.setEnabled(callable(getattr(session, "set_source_appearance", None)))
        self.project.setToolTip("Save this name and color for this exact recording/segment, channel and detector. "
                                "Newly generated figures use it. Other saved figures are not rewritten.")
        note = QtWidgets.QLabel("Appearance only. Reset uses the captured source style. Project defaults apply to future figures for this exact source identity.")
        note.setWordWrap(True)
        layout.addWidget(QtWidgets.QLabel("Source"), 0, 0)
        layout.addWidget(self.source, 0, 1, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Display label"), 1, 0)
        layout.addWidget(self.label, 1, 1, 1, 3)
        layout.addWidget(self.color, 2, 0)
        layout.addWidget(self.reset_color, 2, 1)
        layout.addWidget(self.apply, 2, 2)
        layout.addWidget(self.project, 2, 3)
        layout.addWidget(note, 3, 0, 1, 4)
        self.source.currentIndexChanged.connect(self._load_source)
        self.color.clicked.connect(self._choose_color)
        self.reset_color.clicked.connect(self._reset_color)
        self.apply.clicked.connect(self._apply)
        self.project.clicked.connect(self._apply_project)
        self._load_source()

    def state(self):
        return deepcopy(self._state)

    def set_state(self, state):
        self._state = validate_source_appearance(state)
        self._load_source()

    def set_sources(self, sources):
        selected = self.source.currentData()
        offered = {}
        for source in sources:
            if isinstance(source, dict):
                key = str(source["key"])
                offered[key] = {"key": key, "label": str(source["label"]),
                                "color": str(source.get("color", "#0072b2"))}
            else:
                key = str(source.key)
                offered[key] = {"key": key, "label": str(source.label), "color": str(source.colour)}
        self._sources = offered
        blocker = QtCore.QSignalBlocker(self.source)
        self.source.clear()
        for key, source in offered.items():
            label = source["label"]
            self.source.addItem(label if len(label) <= 65 else label[:62] + "…", key)
            self.source.setItemData(self.source.count() - 1, label + "\n" + key, QtCore.Qt.ToolTipRole)
        if selected in offered:
            self.source.setCurrentIndex(self.source.findData(selected))
        del blocker
        self._load_source()

    def _load_source(self, *_args):
        key = self.source.currentData()
        entry = self._state.get(key, {})
        self.label.setText(entry.get("label", ""))
        self._color_override = entry.get("color")
        self._reset_project_color = False
        self.label.setEnabled(key in self._sources)
        for widget in (self.color, self.reset_color, self.apply):
            widget.setEnabled(key in self._sources)
        self.project.setEnabled(key in self._sources and callable(getattr(self._session, "set_source_appearance", None)))
        self._show_color()

    def _show_color(self):
        fallback = self._sources.get(self.source.currentData(), {}).get("color", "#0072b2")
        color = QtGui.QColor(self._color_override or fallback)
        pixmap = QtGui.QPixmap(18, 18)
        pixmap.fill(color)
        self.color.setIcon(QtGui.QIcon(pixmap))
        self.color.setToolTip(color.name() + (" · figure override" if self._color_override else " · source default"))

    def _choose_color(self):
        fallback = self._sources.get(self.source.currentData(), {}).get("color", "#0072b2")
        chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._color_override or fallback), self, "Source color")
        if chosen.isValid():
            self._color_override = chosen.name()
            self._reset_project_color = False
            self._show_color()

    def _reset_color(self):
        self._color_override = None
        self._reset_project_color = True
        self._show_color()

    def _entry(self):
        entry = {"label": self.label.text()}
        if self._color_override:
            entry["color"] = self._color_override
        key = self.source.currentData()
        return key, validate_source_appearance({key: entry}).get(key, {})

    def _apply(self):
        if self.source.currentData() not in self._sources:
            return
        key, entry = self._entry()
        state = self.state()
        if entry:
            state[key] = entry
        else:
            state.pop(key, None)
        self._state = state
        self._load_source()
        self.changed.emit(self.state())

    def _apply_project(self):
        if not callable(getattr(self._session, "set_source_appearance", None)):
            return
        key, entry = self._entry()
        self._session.set_source_appearance(key, label=entry.get("label", ""), color=entry.get("color"),
                                            reset_color=self._reset_project_color)
        # This is an explicit styling action on the current figure as well.
        # The other saved figures keep their pinned metadata.
        self._apply()
