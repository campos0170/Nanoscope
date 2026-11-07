"""Editable, serializable background recipes; numerical work belongs to analysis."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from PySide6 import QtCore, QtWidgets


class _CurrentPageStack(QtWidgets.QWidget):
    """Only the visible form contributes to the layout's minimum size.

    QStackedLayout reserves the largest hidden page even when sizeHint is overridden;
    that would put a whole anchor table's blank height above the automatic controls.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages = []
        self._index = -1
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def addWidget(self, page):
        self._pages.append(page)
        self._layout.addWidget(page)
        page.hide()
        if self._index < 0:
            self.setCurrentIndex(0)

    def setCurrentIndex(self, index):
        self._index = int(index)
        for position, page in enumerate(self._pages):
            page.setVisible(position == self._index)
        self.updateGeometry()

    def currentIndex(self):
        return self._index

    def currentWidget(self):
        return self._pages[self._index] if 0 <= self._index < len(self._pages) else None


class BackgroundControls(QtWidgets.QGroupBox):
    """A recipe draft. Invalid text is retained until explicit calculate/export."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__('Background source', parent)
        self._writing = False
        self._units = 'pA'
        self._duration_s = 1.0
        self._reference_choice = None
        self._binding = None
        self._current_context = None
        layout = QtWidgets.QVBoxLayout(self)
        self.source = QtWidgets.QComboBox()
        self.source.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(14)
        for label, key in [('Automatic calculation', 'automatic'),
                           ('Editable time/current anchors', 'anchors'),
                           ('Current open-pore reference', 'intervals'),
                           ('Independent reference recording', 'recording')]:
            self.source.addItem(label, key)
        layout.addWidget(self.source)
        self.scope = QtWidgets.QLabel()
        self.scope.setWordWrap(True)
        layout.addWidget(self.scope)
        self.binding_note = self._note('')
        layout.addWidget(self.binding_note)
        self.confirm_source = QtWidgets.QPushButton('Confirm current source and units')
        self.confirm_source.setToolTip('Keep the entered numbers, deliberately assign them to the selected source, its current calibration, time base and units, then calculate again.')
        self.confirm_source.clicked.connect(self._bind_current)
        layout.addWidget(self.confirm_source)
        self.pages = _CurrentPageStack()
        self.pages.setMinimumWidth(0)
        self.pages.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        layout.addWidget(self.pages)
        self.pages.addWidget(self._note('Choose Event-reference residual or Drift-stage correction. These use separate estimators and settings.'))
        self.pages.addWidget(self._anchors_page())
        self.pages.addWidget(self._note('Uses the shared Known open-pore current or Selected open-pore intervals. Edit these in Event reference; those edits affect future measurements. This comparison does not replace stored results.'))
        self.pages.addWidget(self._reference_page())
        self.source.currentIndexChanged.connect(self._source_changed)
        self._source_changed()

    @staticmethod
    def _note(text):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        return label

    def _combo(self, options):
        combo = QtWidgets.QComboBox()
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
        for label, value in options:
            combo.addItem(label, value)
        combo.currentIndexChanged.connect(self._edited)
        return combo

    def _table(self, headers):
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setMinimumWidth(0)
        table.setMinimumHeight(100)
        table.setMaximumHeight(170)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.itemChanged.connect(self._edited)
        return table

    def _anchors_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._note('Times are milliseconds from this recording start. Regions cover the entire recording without gaps/overlaps. Each region is fitted separately; its boundary may contain a genuine jump.'))
        self.interpolation = self._combo([('Linear between anchors', 'linear'), ('PCHIP between anchors', 'pchip'), ('Constant (one anchor per region)', 'constant')])
        layout.addWidget(QtWidgets.QLabel('Interpolation within each region'))
        layout.addWidget(self.interpolation)
        self.edge_policy = self._combo([('Refuse outside anchor support', 'refuse'), ('Hold nearest endpoint within region', 'hold')])
        layout.addWidget(QtWidgets.QLabel('Outside anchor support'))
        layout.addWidget(self.edge_policy)
        self.segment_policy = self._combo([('Separate regions; preserve boundary jumps', 'separate'), ('One continuous region', 'continuous')])
        layout.addWidget(QtWidgets.QLabel('Breakpoint / jump policy'))
        layout.addWidget(self.segment_policy)
        self.regions = self._table(['Region', 'Start (ms)', 'End (ms)'])
        layout.addWidget(self.regions)
        row = QtWidgets.QHBoxLayout()
        self.add_region_button = QtWidgets.QPushButton('Add region')
        self.remove_region_button = QtWidgets.QPushButton('Remove region')
        self.add_region_button.clicked.connect(self.add_region)
        self.remove_region_button.clicked.connect(lambda: self._remove_rows(self.regions))
        row.addWidget(self.add_region_button); row.addWidget(self.remove_region_button)
        layout.addLayout(row)
        self.anchors = self._table(['Region', 'Time (ms)', 'Current'])
        layout.addWidget(self.anchors)
        row = QtWidgets.QHBoxLayout()
        self.add_anchor_button = QtWidgets.QPushButton('Add anchor')
        self.remove_anchor_button = QtWidgets.QPushButton('Remove anchor')
        self.add_anchor_button.clicked.connect(self.add_anchor)
        self.remove_anchor_button.clicked.connect(lambda: self._remove_rows(self.anchors))
        row.addWidget(self.add_anchor_button); row.addWidget(self.remove_anchor_button)
        layout.addLayout(row)
        return page

    def _reference_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._note('Subtract a separately measured current reference. The original recordings remain unchanged. Matching units and full time coverage are required. Positive shift moves the reference later.'))
        self.reference = self._combo([])
        self.reference.currentIndexChanged.disconnect(self._edited)
        self.reference.currentIndexChanged.connect(self._remember_reference)
        layout.addWidget(QtWidgets.QLabel('Open reference recording / current channel'))
        layout.addWidget(self.reference)
        self.shift = QtWidgets.QLineEdit('0')
        self.scale = QtWidgets.QLineEdit('1')
        self.offset = QtWidgets.QLineEdit('0')
        for label, control in [('Reference time shift (ms)', self.shift), ('Reference scale', self.scale), ('Reference current offset', self.offset)]:
            caption = QtWidgets.QLabel(label)
            if control is self.offset:
                self.offset_label = caption
            layout.addWidget(caption); layout.addWidget(control)
            control.textEdited.connect(self._edited)
        self.resampling = self._combo([('None — equal rates and aligned samples', 'none'), ('Linear resampling (explicit)', 'linear')])
        layout.addWidget(QtWidgets.QLabel('Reference resampling'))
        layout.addWidget(self.resampling)
        return page

    def _remember_reference(self, *_):
        if not self._writing:
            self._reference_choice = deepcopy(self.reference.currentData())
            self._edited()

    def _source_changed(self, *_):
        if (not self._writing and self.source_key() != 'automatic'
                and self._binding is None and self._current_context is not None):
            self._binding = deepcopy(self._current_context)
            self._units = self._binding['units']
        self.pages.setCurrentIndex(self.source.currentIndex())
        self.pages.updateGeometry()
        self.scope.setText('Comparison and export; automatic drift is enabled separately for detection.' if self.source_key() == 'automatic' else 'This custom recipe is for comparison/export. Existing automatic processing and main-trace display stay unchanged. Save the project to keep the draft.')
        self._refresh_binding()
        self._edited()

    def _edited(self, *_):
        if not self._writing:
            self.changed.emit()

    def source_key(self):
        return str(self.source.currentData() or 'automatic')

    def set_context(self, duration_s, units, binding=None):
        self._duration_s = float(duration_s or 1.0)
        self._current_context = deepcopy(binding)
        if self._binding is None:
            self._units = str(units or 'pA')
        self._refresh_binding()

    def _refresh_binding(self):
        self.anchors.setHorizontalHeaderLabels(['Region', 'Time (ms)', f'Current ({self._units})'])
        self.offset.setToolTip(f'Offset added to the scaled reference, in {self._units}.')
        self.offset_label.setText(f'Reference current offset ({self._units})')
        custom = self.source_key() != 'automatic'
        self.binding_note.setVisible(custom)
        matches = self._binding is not None and self._binding == self._current_context
        self.confirm_source.setVisible(custom and not matches)
        self.binding_note.setText(
            f'Values are bound to this source, its calibration and time base; current unit: {self._units}.'
            if matches else f'Values retain unit {self._units}. The source, calibration or time base is unconfirmed/changed. Review the numbers, then confirm the current source and units.')

    def _bind_current(self):
        if self._current_context is None:
            self.binding_note.setText('Select a current recording first.')
            return
        self._binding = deepcopy(self._current_context)
        self._units = self._binding['units']
        self._refresh_binding()
        self._edited()

    def require_current_context(self):
        if self._binding is None or self._binding != self._current_context:
            raise ValueError('Background values are bound to a different or unconfirmed source/calibration/time base. Review them and press Confirm current source and units.')

    def set_references(self, choices):
        selected = deepcopy(self._reference_choice)
        self._writing = True
        try:
            self.reference.clear()
            self.reference.addItem('Choose an open reference…', None)
            for label, identity in choices:
                self.reference.addItem(label, deepcopy(identity))
            index = next((i for i in range(self.reference.count()) if self.reference.itemData(i) == selected), -1)
            if selected is not None and index < 0:
                self.reference.addItem('Missing reference — reopen its recording', selected)
                index = self.reference.count() - 1
            self.reference.setCurrentIndex(max(0, index))
        finally:
            self._writing = False

    def _append(self, table, values):
        row = table.rowCount(); table.insertRow(row)
        for col, value in enumerate(values):
            table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))

    def add_region(self):
        used = {self.regions.item(r, 0).text() for r in range(self.regions.rowCount()) if self.regions.item(r, 0)}
        number = 1
        while str(number) in used: number += 1
        self._append(self.regions, [number, '0', f'{self._duration_s * 1000:.9g}'])
        self.regions.setCurrentCell(self.regions.rowCount() - 1, 0)
        self._edited()

    def add_anchor(self):
        row = self.regions.currentRow()
        region = self.regions.item(max(row, 0), 0) if self.regions.rowCount() else None
        self._append(self.anchors, [region.text() if region else '1', '0', '0'])
        self._edited()

    def _remove_rows(self, table):
        for row in sorted({index.row() for index in table.selectedIndexes()}, reverse=True):
            table.removeRow(row)
        self._edited()

    @staticmethod
    def _rows(table):
        return [[table.item(r, c).text() if table.item(r, c) else '' for c in range(table.columnCount())] for r in range(table.rowCount())]

    def state(self):
        return dict(version=1, source=self.source_key(), units=self._units, binding=deepcopy(self._binding), regions=self._rows(self.regions), anchors=self._rows(self.anchors), interpolation=self.interpolation.currentData(), edge_policy=self.edge_policy.currentData(), segment_policy=self.segment_policy.currentData(), reference=deepcopy(self._reference_choice), shift=self.shift.text(), scale=self.scale.text(), offset=self.offset.text(), resampling=self.resampling.currentData())

    def restore(self, state):
        if not isinstance(state, Mapping): raise ValueError('Background recipe draft must be an object.')
        if state:
            from ..core.form_draft_state import background_state_readable
            if not background_state_readable({'output_mode': 'zero_centered_display', 'draft': state}):
                raise ValueError('Background recipe draft is malformed or from an unsupported version.')
        self._writing = True
        try:
            for combo, value in [(self.source, state.get('source', 'automatic')), (self.interpolation, state.get('interpolation', 'linear')), (self.edge_policy, state.get('edge_policy', 'refuse')), (self.segment_policy, state.get('segment_policy', 'separate')), (self.resampling, state.get('resampling', 'none'))]:
                index = combo.findData(value)
                if index < 0: raise ValueError(f'Unknown background choice: {value}')
                combo.setCurrentIndex(index)
            for table, key in [(self.regions, 'regions'), (self.anchors, 'anchors')]:
                table.setRowCount(0)
                rows = state.get(key, [])
                if not isinstance(rows, list) or len(rows) > 10000: raise ValueError('Invalid background table.')
                for row in rows:
                    if not isinstance(row, (list, tuple)) or len(row) != 3: raise ValueError('Background table rows need three values.')
                    self._append(table, row)
            for control, key, default in [(self.shift, 'shift', '0'), (self.scale, 'scale', '1'), (self.offset, 'offset', '0')]:
                control.setText(str(state.get(key, default)))
            self._reference_choice = deepcopy(state.get('reference'))
            self._binding = deepcopy(state.get('binding'))
            self._units = str(state.get('units', 'pA'))
            self.pages.setCurrentIndex(self.source.currentIndex())
            self._source_changed()
        finally:
            self._writing = False
        self._edited()

    @staticmethod
    def _number(text, label):
        try: value = float(text)
        except (ValueError, TypeError): raise ValueError(f'{label} must be a finite number.') from None
        if not math.isfinite(value): raise ValueError(f'{label} must be a finite number.')
        return value

    def recipe(self, mode, interval_reference=None):
        source = self.source_key()
        recipe = dict(version=1, source=source, mode=mode, units=self._units)
        if source == 'anchors':
            rows = self._rows(self.regions)
            if not rows: raise ValueError('Add at least one region and its anchors.')
            if self.segment_policy.currentData() == 'continuous' and len(rows) != 1:
                raise ValueError('One continuous region requires exactly one region; use Separate regions to preserve jumps.')
            regions = {}
            for name, start, end in rows:
                if not name or name in regions: raise ValueError('Region names must be nonempty and unique.')
                regions[name] = dict(start_s=self._number(start, 'Region start') / 1000, end_s=self._number(end, 'Region end') / 1000, anchors=[])
            for name, time, current in self._rows(self.anchors):
                if name not in regions: raise ValueError(f'Anchor region {name!r} does not exist.')
                regions[name]['anchors'].append([self._number(time, 'Anchor time') / 1000, self._number(current, 'Anchor current')])
            recipe.update(regions=list(regions.values()), interpolation=self.interpolation.currentData(), edge_policy=self.edge_policy.currentData())
        elif source == 'intervals':
            if interval_reference is None: raise ValueError('Choose a known current or selected open-pore intervals below first.')
            recipe['interval_reference'] = interval_reference
        elif source == 'recording':
            if self._reference_choice is None: raise ValueError('Choose an independent reference recording and channel.')
            recipe.update(time_shift_s=self._number(self.shift.text(), 'Reference shift') / 1000, scale=self._number(self.scale.text(), 'Reference scale'), offset=self._number(self.offset.text(), 'Reference offset'), reference_resampling=self.resampling.currentData())
        else:
            raise ValueError('The automatic estimator uses its own controls below.')
        return recipe
