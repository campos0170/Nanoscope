"""Whole-cluster review over a saved, explicitly scoped assignment snapshot."""
from PySide6 import QtCore, QtWidgets

from ..analysis import study_cluster_review as review


def label_text(label, method):
    if label == review.UNASSIGNED:
        return 'Unassigned / no waveform features'
    if method == 'hdbscan' and label == -1:
        return 'Noise (-1)'
    return ('Population ' if method == 'subpop' else 'Cluster ') + str(label)


class StudyClusterControls(QtWidgets.QGroupBox):
    def __init__(self, owner):
        super().__init__('Review whole clusters', owner)
        self.owner = owner
        self._displayed = None
        box = QtWidgets.QVBoxLayout(self)
        self.source = QtWidgets.QComboBox()
        self.source.setAccessibleName('Cluster review source')
        self.source.addItem('Pooled clustering', 'pooled')
        self.source.addItem('Selected-recording mixture', 'mixture')
        box.addWidget(self.source)
        self.load = QtWidgets.QPushButton('Load saved clusters for review')
        self.load.clicked.connect(owner.load_cluster_review)
        box.addWidget(self.load)
        self.method = QtWidgets.QComboBox()
        self.method.setAccessibleName('Cluster review assignment method')
        box.addWidget(self.method)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        box.addWidget(self.status)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Cluster', 'Events', 'Accepted', 'Rejected', 'Decision'])
        self.table.setAccessibleName('Cluster acceptance summary')
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumHeight(160)
        self.table.setMaximumHeight(280)
        box.addWidget(self.table)
        self.actions = []
        for text, callback in (
            ('Show events in selected clusters', owner.show_cluster_members),
            ('Accept selected clusters', lambda: owner.review_clusters(True)),
            ('Reject selected clusters', lambda: owner.review_clusters(False)),
            ('Keep only selected clusters', lambda: owner.review_clusters(True, only=True)),
            ('Use cluster-reviewed events for analysis', owner.confirm_cluster_review),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            box.addWidget(button)
            self.actions.append(button)
        note = QtWidgets.QLabel(
            'Select one or more rows. Accept / Reject changes every event in those groups. '
            'Keep only selected clusters also rejects all other events in the listed recordings, '
            'including unassigned events. Individual event review remains available in 3 Detect / review. '
            'Confirm here, then recompute comparisons or extract features for the next analysis.')
        note.setWordWrap(True)
        box.addWidget(note)
        self.source.currentIndexChanged.connect(self.refresh)
        self.method.currentIndexChanged.connect(self._refresh_groups)

    def kind(self):
        return self.source.currentData()

    def snapshot(self):
        return self.owner.state.get('cluster_reviews', {}).get(self.kind())

    def selected_labels(self):
        return [self.table.item(index.row(), 0).data(QtCore.Qt.UserRole)
                for index in self.table.selectionModel().selectedRows()]

    def refresh(self, *args):
        method = self.method.currentData()
        snapshot = self.snapshot()
        self.method.blockSignals(True)
        self.method.clear()
        if snapshot is not None:
            for name in snapshot['methods']:
                title = ('HDBSCAN' if name == 'hdbscan' else 'Mixture populations' if name == 'subpop'
                         else 'K-means (' + name.removeprefix('kmeans') + ' clusters)')
                self.method.addItem(title, name)
        self.method.setCurrentIndex(max(0, self.method.findData(method)))
        self.method.blockSignals(False)
        result_key = 'clustering' if self.kind() == 'pooled' else 'mixture'
        self.load.setEnabled(self.owner.state.get(result_key) is not None)
        self._refresh_groups()

    def _refresh_groups(self, *args):
        snapshot = self.snapshot()
        displayed = (self.kind(), self.method.currentData(), snapshot.get('snapshot_id') if snapshot is not None else None)
        selected = self.selected_labels() if displayed == self._displayed else []
        self._displayed = displayed
        self.table.setRowCount(0)
        valid = False
        try:
            if snapshot is None:
                raise ValueError('Run clustering or a mixture fit, then load its saved clusters here. New runs load automatically.')
            view = self.owner.cluster_review_view(self.kind())
            if not view.get('current'):
                raise ValueError(view.get('error', 'The saved cluster review needs to be reloaded.'))
            method = self.method.currentData()
            frame = view['summaries'][method]
            self.table.setRowCount(len(frame))
            for i, row in enumerate(frame.itertuples(index=False)):
                values = [label_text(row.label, method), str(row.n_total), str(row.n_accepted),
                          str(row.n_rejected), str(row.status)]
                for j, text in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(text)
                    if j == 0:
                        item.setData(QtCore.Qt.UserRole, row.label)
                    self.table.setItem(i, j, item)
                if row.label in selected:
                    self.table.selectionModel().select(self.table.model().index(i, 0),
                        QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            tags = ', '.join(source['tag'] for source in snapshot['records'])
            self.status.setText('Applies to: ' + tags + '. Saved memberships stay fixed during review; counts show current acceptance.')
            valid = True
        except (ValueError, KeyError, TypeError) as exc:
            self.status.setText(str(exc))
        for button in self.actions:
            button.setEnabled(valid)
