"""Reusable source-label/color editor with explicit figure and project scopes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.source_style import MAX_LABEL_LENGTH, validate_source_appearance


def with_source_appearance_edit(appearance, state, *, view, source_key=None):
    """One legacy gesture updates local aliases/ink and the shared series cascade.

    The old editor remains an API for saved callers. An explicit source colour
    must replace a higher-priority colour for that source, preserving other series
    properties and every other source. Canvas-only renames leave series ink alone.
    """
    from ..core.figure_series import stable_series_id
    from ..core.figure_spec import thaw_json
    overrides = validate_source_appearance(state)
    previous = appearance.source_appearance or {}
    affected = {key for key in previous.keys() | overrides.keys()
                if previous.get(key, {}).get('color') != overrides.get(key, {}).get('color')}
    if source_key is not None:
        affected.add(source_key)
    series = thaw_json(appearance.series_styles or {})
    for key in affected:
        identity = stable_series_id(view, 'data', key)
        if identity in series:
            series[identity].pop('color', None)
            if not series[identity]:
                series.pop(identity)
    return replace(appearance, source_appearance=overrides, series_styles=series or None)


class SourceAppearanceEditor(QtWidgets.QGroupBox):
    """Emit only committed presentation changes, never analysis requests.

    ``set_sources`` accepts PlotSources or snapshot mappings (key/label/color).
    ``changed`` carries the complete figure-local override mapping. Project
    defaults use Session's source-appearance mutator; saved snapshots stay pinned.
    """

    changed = QtCore.Signal(dict)

    def __init__(self, parent=None, *, session=None, project_only=False):
        super().__init__("Source names and colors", parent)
        self._session = session
        self._project_only = bool(project_only)
        self._sources = {}
        self._state = {}
        self._color_override = None
        self._reset_project_color = False
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
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
        if self._project_only:
            self.setTitle("Project source defaults")
            self.apply.hide()
            self.project.setText("Save project source default")
            self.label.setPlaceholderText("Original source label (clear to restore)")
            self.reset_color.setText("Default palette color")
            self.reset_color.setToolTip("Restore this exact source's project palette color.")
            note.setText("Appearance only. These defaults are used by new or explicitly refreshed figures. "
                         "Existing figure overrides and saved snapshots are unchanged.")
        note.setWordWrap(True)
        # Label/field pairs may wrap in the narrow Format column. A fixed two-column
        # grid made the label width and the source combo's minimum width add together,
        # pushing its containing panel beyond the inspector's 290 px budget.
        self.setMinimumWidth(0)
        layout.addRow("Source", self.source)
        layout.addRow("Display label", self.label)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.color)
        buttons.addWidget(self.reset_color)
        layout.addRow(buttons)
        layout.addRow(self.apply)
        layout.addRow(self.project)
        layout.addRow(note)
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
        scope = "project override" if self._project_only else "figure override"
        self.color.setToolTip(color.name() + (" · " + scope if self._color_override else " · source default"))

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
        if (self.source.currentData() not in self._sources
                or not callable(getattr(self._session, "set_source_appearance", None))):
            return
        key, entry = self._entry()
        self._session.set_source_appearance(key, label=entry.get("label", ""), color=entry.get("color"),
                                            reset_color=self._reset_project_color)
        # This is an explicit styling action on the current figure as well.
        # The other saved figures keep their pinned metadata.
        if not self._project_only:
            self._apply()
