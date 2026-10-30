"""An independent, saved incubation study: explicit inputs, execution and review."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis import incubation_study as study
from ..analysis.cancel import OperationCancelled, cancel_scope
from .export_paths import ask_export_path
from .numeric_text import parse_strict_number
from .worker_thread import JOB_NONE, retire_worker_thread, start_worker_thread


class _FrameModel(QtCore.QAbstractTableModel):
    """Expose a saved frame without creating a widget/item for every event."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = pd.DataFrame()

    def set_frame(self, frame):
        self.beginResetModel()
        self.frame = frame.copy(deep=False)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QtCore.QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or role not in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            return None
        value = self.frame.iat[index.row(), index.column()]
        if isinstance(value, float):
            return "" if math.isnan(value) else format(value, ".8g")
        return "" if value is None else str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):  # noqa: N802
        if role != QtCore.Qt.DisplayRole:
            return None
        return str(self.frame.columns[section]) if orientation == QtCore.Qt.Horizontal else str(section + 1)


class _FigureView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setAccessibleName("Study figure; drag to pan, use the wheel to zoom")

    def set_image(self, path):
        image = QtGui.QPixmap(str(path))
        if image.isNull():
            raise ValueError(f"Cannot display figure: {Path(path).name}")
        self.scene().clear()
        self.scene().addPixmap(image)
        self.scene().setSceneRect(QtCore.QRectF(image.rect()))
        self.fit()

    def fit(self):
        self.resetTransform()
        if not self.scene().itemsBoundingRect().isEmpty():
            self.fitInView(self.scene().itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if .02 <= scale <= 30:
            self.scale(factor, factor)
        event.accept()


class _StudyWorker(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, spec, output_dir):
        super().__init__()
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    @QtCore.Slot()
    def run(self):
        try:
            with cancel_scope(self.cancel_event.is_set):
                result = study.run_study(
                    self.spec, output_dir=self.output_dir, progress=self.progress.emit,
                    cancel=self.cancel_event.is_set)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface failures on the owning GUI thread
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)


class IncubationStudyWindow(QtWidgets.QMainWindow):
    """Reusable study setup and bundle viewer, independent of the live Session.

    A job receives a captured StudySpec, never widgets or the live Session. Closing
    retires its thread through the shared bounded lifetime helper. A job still
    winding down prevents this window from starting a replacement.
    """

    studyFinished = QtCore.Signal(object)
    studyFailed = QtCore.Signal(str)
    studyCancelled = QtCore.Signal()
    _COLUMNS = (
        "Tag", "Incubation (h)", "Pore ID", "Pore diameter (nm)",
        "Concentration (pM)", "Voltage (mV)", "Prepared CSV", "Metadata JSON",
    )

    def __init__(self, parent=None, *, spec_path=None, result_path=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Incubation study — NanoScope")
        self.resize(1120, 760)
        self._worker = None
        self._thread = None
        self._retiring_thread = None
        self._result = None
        self._recipe_version = "zn_student_incubation_v1"
        self._figure_paths = []
        self._retirement_timer = QtCore.QTimer(self)
        self._retirement_timer.setInterval(100)
        self._retirement_timer.timeout.connect(self._check_retired)
        self._build_ui()
        if spec_path is not None:
            self.load_recipe(spec_path)
        if result_path is not None:
            self.load_result(result_path)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        heading = QtWidgets.QLabel("Incubation study")
        font = heading.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        description = QtWidgets.QLabel(
            "Compare prepared recordings across incubation times. Save the recipe and "
            "review the resulting event tables, figures and report in one study bundle. "
            "This automated recipe runs detection and applies its automatic gate. To curate "
            "detections with the main analysis tools before comparisons and figures, open "
            "the manual workflow below.")
        description.setWordWrap(True)
        layout.addWidget(description)
        toolbar = QtWidgets.QHBoxLayout()
        self._load_recipe_btn = self._button("Open recipe…", self._choose_recipe, toolbar)
        self._save_recipe_btn = self._button("Save recipe…", self._choose_save_recipe, toolbar)
        self._open_result_btn = self._button("Open saved study…", self._choose_result, toolbar)
        self._manual_workflow_btn = self._button("Curate detections / manual workflow…", self._open_manual_workflow, toolbar)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, 1)
        self._build_setup()
        self._summary_view, self._summary_model = self._frame_tab("Summary")
        self._events_view, self._events_model = self._frame_tab("Events")
        figures = QtWidgets.QWidget()
        figures_layout = QtWidgets.QVBoxLayout(figures)
        figure_bar = QtWidgets.QHBoxLayout()
        self._figures_combo = QtWidgets.QComboBox()
        self._figures_combo.setAccessibleName("Study figure")
        self._figures_combo.currentIndexChanged.connect(self._show_figure)
        figure_bar.addWidget(self._figures_combo, 1)
        self._button("Fit", lambda: self._figure_view.fit(), figure_bar)
        self._button("−", lambda: self._figure_view.scale(1 / 1.2, 1 / 1.2), figure_bar).setToolTip("Zoom out")
        self._button("+", lambda: self._figure_view.scale(1.2, 1.2), figure_bar).setToolTip("Zoom in")
        figures_layout.addLayout(figure_bar)
        self._figure_view = _FigureView()
        figures_layout.addWidget(self._figure_view, 1)
        figures_layout.addWidget(QtWidgets.QLabel("Drag to pan · Use the wheel or + / − to zoom"))
        self._tabs.addTab(figures, "Figures")
        self._report = QtWidgets.QTextBrowser()
        self._report.setOpenExternalLinks(False)
        self._tabs.addTab(self._report, "Report")
        self._identity = QtWidgets.QLabel("No study result loaded.")
        self._identity.setTextFormat(QtCore.Qt.PlainText)
        self._identity.setWordWrap(True)
        self._identity.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self._identity)
        self._status = QtWidgets.QLabel("Choose a prepared recording folder or open a saved recipe.")
        self._status.setTextFormat(QtCore.Qt.PlainText)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        runbar = QtWidgets.QHBoxLayout()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        runbar.addWidget(self._progress, 1)
        self._run_btn = self._button("Run study", self.start_study, runbar)
        self._cancel_btn = self._button("Cancel", self.cancel, runbar)
        self._cancel_btn.setEnabled(False)
        layout.addLayout(runbar)

    @staticmethod
    def _button(text, callback, layout):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _open_manual_workflow(self):
        owner=self.parent()
        if callable(getattr(owner,'_open_manual_study',None)):
            owner._open_manual_study()
            return
        from .manual_study_window import ManualStudyWindow
        if getattr(self,'_manual_workspace',None) is None:
            self._manual_workspace=ManualStudyWindow(self)
        self._manual_workspace.show()
        self._manual_workspace.raise_()
        self._manual_workspace.activateWindow()

    def _build_setup(self):
        self._setup = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self._setup)
        form = QtWidgets.QFormLayout()
        self._name_edit = QtWidgets.QLineEdit("Incubation study")
        self._name_edit.setAccessibleName("Study name")
        form.addRow("Study name", self._name_edit)
        self._model_edit = QtWidgets.QLineEdit()
        self._model_edit.setAccessibleName("Student model path")
        self._model_edit.setPlaceholderText("Choose the trained .keras model used for this study")
        form.addRow("Student model", self._path_row(self._model_edit, self._choose_model))
        self._output_edit = QtWidgets.QLineEdit()
        self._output_edit.setAccessibleName("Study output folder")
        self._output_edit.setPlaceholderText("A new or empty folder for the saved study")
        form.addRow("Output folder", self._path_row(self._output_edit, self._choose_output))
        row = QtWidgets.QHBoxLayout()
        self._mixture_combo = QtWidgets.QComboBox()
        self._mixture_combo.setAccessibleName("Mixture target recording")
        row.addWidget(self._mixture_combo, 1)
        row.addWidget(QtWidgets.QLabel("Reference seed"))
        self._seed = QtWidgets.QSpinBox()
        self._seed.setRange(0, 0)
        self._seed.setReadOnly(True)
        self._seed.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self._seed.setToolTip("Recipe v1 fixes seed 0 to reproduce the reference notebook.")
        row.addWidget(self._seed)
        form.addRow("Mixture target", row)
        layout.addLayout(form)
        self._recipe_label = QtWidgets.QLabel("Recipe: zn_student_incubation_v1")
        self._recipe_label.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self._recipe_label)
        source_bar = QtWidgets.QHBoxLayout()
        self._button("Load prepared folder…", self._choose_folder, source_bar)
        self._button("Remove selected recordings", self._remove_rows, source_bar)
        source_bar.addStretch()
        layout.addLayout(source_bar)
        self._recordings = QtWidgets.QTableWidget(0, len(self._COLUMNS))
        self._recordings.setHorizontalHeaderLabels(self._COLUMNS)
        self._recordings.setAccessibleName("Study recordings and editable experiment metadata")
        self._recordings.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._recordings.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._recordings.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self._recordings.horizontalHeader().setStretchLastSection(True)
        self._recordings.itemChanged.connect(self._refresh_mixture)
        layout.addWidget(self._recordings, 1)
        note = QtWidgets.QLabel(
            "Inputs must already contain prepared time_s, current_detr_pA and current_raw_pA "
            "samples plus metadata. Review each incubation time and physical pore identity "
            "before running; different pores limit causal comparisons.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._tabs.addTab(self._setup, "Setup")

    def _path_row(self, edit, choose):
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        self._button("Choose…", choose, row)
        return widget

    def _frame_tab(self, label):
        view = QtWidgets.QTableView()
        model = _FrameModel(view)
        view.setModel(model)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        view.horizontalHeader().setDefaultSectionSize(160)
        self._tabs.addTab(view, label)
        return view, model

    def _text(self, row, col):
        item = self._recordings.item(row, col)
        return "" if item is None else item.text().strip()

    def set_spec(self, spec):
        if self.is_busy():
            raise ValueError("Wait for the current study to stop before changing its recipe.")
        if spec.seed != 0:
            raise ValueError("Recipe v1 fixes seed 0 to reproduce the reference notebook.")
        self._name_edit.setText(spec.name)
        self._model_edit.setText(str(spec.model_path))
        self._seed.setValue(spec.seed)
        self._recipe_version = spec.recipe_version
        self._recipe_label.setText(f"Recipe: {spec.recipe_version}")
        self._set_recordings(spec.recordings)
        if self._mixture_combo.findText(spec.mixture_tag) < 0:
            self._mixture_combo.addItem(spec.mixture_tag)
        self._mixture_combo.setCurrentText(spec.mixture_tag)

    def _set_recordings(self, recordings):
        self._recordings.blockSignals(True)
        try:
            self._recordings.setRowCount(len(recordings))
            for row, rec in enumerate(recordings):
                values = (rec.tag, rec.incubation_h, rec.pore_id, rec.pore_diameter_nm,
                          rec.concentration_pm, rec.voltage_mv, rec.trace_path, rec.metadata_path)
                for col, value in enumerate(values):
                    unknown = value is None or (isinstance(value, float) and not math.isfinite(value))
                    item = QtWidgets.QTableWidgetItem("" if unknown else str(value))
                    if col >= 6:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                        item.setToolTip(str(value))
                    self._recordings.setItem(row, col, item)
        finally:
            self._recordings.blockSignals(False)
        self._refresh_mixture()

    def _refresh_mixture(self, *_):
        previous = self._mixture_combo.currentText()
        tags = list(dict.fromkeys(self._text(row, 0) for row in range(self._recordings.rowCount()) if self._text(row, 0)))
        self._mixture_combo.clear()
        self._mixture_combo.addItems(tags)
        target = previous if previous in tags else "zn_30h" if "zn_30h" in tags else (tags[-1] if tags else "")
        self._mixture_combo.setCurrentText(target)

    def _remove_rows(self):
        for row in sorted({i.row() for i in self._recordings.selectionModel().selectedRows()}, reverse=True):
            self._recordings.removeRow(row)
        self._refresh_mixture()

    @staticmethod
    def _number(value, label, *, positive=False, nonnegative=False, nonzero=False):
        number = parse_strict_number(value, allow_unicode_minus=True)
        if number is None:
            raise ValueError(f"{label} must be a finite number using digits 0–9.")
        if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0) or (nonzero and number == 0):
            qualifier = "positive" if positive else "nonnegative" if nonnegative else "nonzero" if nonzero else "finite"
            raise ValueError(f"{label} must be {qualifier} and finite.")
        return number

    def capture_spec(self, *, check_files=True):
        """Validate and copy editable fields before starting any background work."""
        name = self._name_edit.text().strip()
        if not name:
            raise ValueError("Enter a study name.")
        if not self._recordings.rowCount():
            raise ValueError("Load at least one prepared recording.")
        records = []
        tags = set()
        for row in range(self._recordings.rowCount()):
            tag = self._text(row, 0)
            prefix = f"Recording {row + 1} ({tag or 'unnamed'})"
            if not tag or tag in tags:
                raise ValueError(f"{prefix}: each tag must be nonempty and unique.")
            tags.add(tag)
            pore = self._text(row, 2)
            if not pore:
                raise ValueError(f"{prefix}: enter the physical pore ID.")
            paths = [Path(self._text(row, col)).expanduser().resolve() for col in (6, 7)]
            if check_files:
                for path in paths:
                    if not path.is_file():
                        raise ValueError(f"{prefix}: input file does not exist: {path}")
            records.append(study.StudyRecording(
                tag=tag, trace_path=paths[0], metadata_path=paths[1],
                incubation_h=self._number(self._text(row, 1), f"{prefix}: incubation", nonnegative=True),
                pore_id=pore,
                pore_diameter_nm=self._number(self._text(row, 3), f"{prefix}: pore diameter", positive=True),
                concentration_pm=self._number(self._text(row, 4), f"{prefix}: concentration", positive=True),
                voltage_mv=self._number(self._text(row, 5), f"{prefix}: voltage", nonzero=True)))
        model_text = self._model_edit.text().strip()
        if not model_text:
            raise ValueError("Choose the trained Student model explicitly.")
        model = Path(model_text).expanduser().resolve()
        if check_files and not model.is_file():
            raise ValueError(f"Student model does not exist: {model}")
        mixture = self._mixture_combo.currentText()
        if mixture not in tags:
            raise ValueError("Choose a mixture target from the study recordings.")
        return study.StudySpec(name=name, recordings=tuple(records), model_path=model,
                               mixture_tag=mixture, recipe_version=self._recipe_version,
                               seed=self._seed.value())

    def load_folder(self, folder):
        if self.is_busy():
            raise ValueError("Wait for the current study to stop before replacing its recordings.")
        recordings = study.discover_recordings(Path(folder))
        if not recordings:
            raise ValueError("No prepared recordings found. Choose a folder with *_metadata.json and its prepared CSV files.")
        self._set_recordings(recordings)
        self._tabs.setCurrentWidget(self._setup)
        self._status.setText(f"Loaded {len(recordings)} prepared recordings. Review the study metadata and choose a model.")

    def load_recipe(self, path):
        self.set_spec(study.load_spec(Path(path)))
        self._tabs.setCurrentWidget(self._setup)
        self._status.setText(f"Recipe loaded: {Path(path).name}. Choose the output folder before running.")

    def save_recipe(self, path):
        study.save_spec(self.capture_spec(check_files=False), Path(path))
        self._status.setText(f"Recipe saved: {Path(path)}")

    def load_result(self, path):
        if self.is_busy():
            raise ValueError("Wait for the current study to stop before opening another result.")
        result = study.load_study_result(Path(path))
        recipe = Path(result.output_dir) / "recipe.json"
        if recipe.is_file():
            self.set_spec(study.load_spec(recipe))
        self._install_result(result)
        self._status.setText("Opened saved study; no analysis was rerun.")
        return result

    def _install_result(self, result):
        self._result = result
        self._summary_model.set_frame(result.summary)
        self._events_model.set_frame(result.events)
        self._figure_paths = [Path(path) for path in result.figure_paths if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".svg")]
        self._figures_combo.blockSignals(True)
        self._figures_combo.clear()
        self._figures_combo.addItems([path.stem.replace("_", " ") for path in self._figure_paths])
        self._figures_combo.blockSignals(False)
        self._figure_view.scene().clear()
        if self._figure_paths:
            self._show_figure(0)
        else:
            self._figure_view.scene().addText("No displayable figure in this study bundle.")
        folder = Path(result.output_dir)
        self._report.document().setBaseUrl(QtCore.QUrl.fromLocalFile(str(folder) + os.sep))
        report_path = next((folder / name for name in ("report.md", "REPORT.md", "report.txt") if (folder / name).is_file()), None)
        if report_path:
            text = report_path.read_text(encoding="utf-8")
            if report_path.suffix.lower() == ".md":
                self._report.setMarkdown(text)
            else:
                self._report.setPlainText(text)
        else:
            self._report.setPlainText(json.dumps(result.metadata, indent=2, default=str))
        self._identity.setText(f"Saved study: {folder}\n{len(result.summary):,} summary rows · {len(result.events):,} events")
        self._tabs.setCurrentIndex(1)

    def _show_figure(self, index):
        if 0 <= index < len(self._figure_paths):
            try:
                self._figure_view.set_image(self._figure_paths[index])
            except ValueError as exc:
                self._status.setText(str(exc))

    def is_busy(self):
        if self._worker is not None:
            return True
        if self._retiring_thread is not None:
            return self._retiring_thread.isRunning()
        return False

    def _set_busy(self, busy):
        self._setup.setEnabled(not busy)
        self._load_recipe_btn.setEnabled(not busy)
        self._save_recipe_btn.setEnabled(not busy)
        self._open_result_btn.setEnabled(not busy)
        self._run_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy and self._worker is not None)

    def start_study(self):
        if self.is_busy():
            return False
        try:
            spec = self.capture_spec()
            output_text = self._output_edit.text().strip()
            if not output_text:
                raise ValueError("Choose a new or empty output folder for this study.")
            output = Path(output_text).expanduser().resolve()
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("The output folder must be new or empty; choose another folder to preserve the saved study.")
        except (ValueError, OSError) as exc:
            self._status.setText(str(exc))
            self._tabs.setCurrentWidget(self._setup)
            return False
        worker = _StudyWorker(spec, output)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._progress.setRange(0, 0)
        self._status.setText(f"Starting {spec.name}…")
        self._set_busy(True)
        self._thread = start_worker_thread(worker, domain=JOB_NONE, label="Incubation study", owner=self)
        return True

    @QtCore.Slot(float, str)
    def _on_progress(self, fraction, message):
        if self.sender() is not self._worker or self._worker is None:
            return
        self._progress.setRange(0, 100)
        self._progress.setValue(max(0, min(100, round(fraction * 100))))
        if not self._worker.cancel_event.is_set():
            self._status.setText(message)

    def _retire_current(self):
        worker, thread = self._worker, self._thread
        self._worker = self._thread = None
        stopped = retire_worker_thread(worker, thread)
        if not stopped:
            self._retiring_thread = thread
            self._retirement_timer.start()
        self._progress.setRange(0, 100)
        self._set_busy(not stopped)

    @QtCore.Slot(object)
    def _on_finished(self, result):
        if self._worker is None or self.sender() is not self._worker:
            return
        self._retire_current()
        try:
            self._install_result(result)
        except Exception as exc:  # the written bundle remains available even if presentation fails
            self._status.setText(f"Study saved to {result.output_dir}, but its display failed: {exc}")
            self.studyFailed.emit(str(exc))
            return
        self._progress.setValue(100)
        self._status.setText("Study complete. The result bundle is saved and can be reopened without rerunning.")
        self.studyFinished.emit(result)

    @QtCore.Slot(str)
    def _on_failed(self, message):
        if self._worker is None or self.sender() is not self._worker:
            return
        self._retire_current()
        self._status.setText(f"Study failed: {message}")
        self.studyFailed.emit(message)

    @QtCore.Slot()
    def _on_cancelled(self):
        if self._worker is None or self.sender() is not self._worker:
            return
        self._retire_current()
        self._status.setText("Study cancelled. No completed result was installed.")
        self.studyCancelled.emit()

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancellation requested; waiting for the current operation to stop…")

    def abort(self):
        """Stop delivery on close/owner teardown; keep an unfinished worker alive."""
        if self._worker is not None:
            self._worker.cancel()
            self._retire_current()
            self._status.setText("Study closed; cancellation requested. Any unfinished operation is winding down." if self.is_busy() else "Study stopped.")
        workspace = getattr(self, '_manual_workspace', None)
        if workspace is not None:
            workspace_abort = getattr(workspace, 'abort', None)
            if callable(workspace_abort):
                workspace_abort()

    def _check_retired(self):
        if self._retiring_thread is None or not self._retiring_thread.isRunning():
            self._retiring_thread = None
            self._retirement_timer.stop()
            self._set_busy(False)

    def closeEvent(self, event):  # noqa: N802
        self.abort()
        super().closeEvent(event)

    def _dialog_action(self, action):
        try:
            action()
        except Exception as exc:
            self._status.setText(str(exc))

    def _choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Prepared recording folder")
        if path:
            self._dialog_action(lambda: self.load_folder(path))

    def _choose_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose trained Student model", "", "Keras model (*.keras *.h5 *.hdf5);;All files (*)")
        if path:
            self._model_edit.setText(path)

    def _choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a new or empty study output folder")
        if path:
            self._output_edit.setText(path)

    def _choose_recipe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open study recipe", "", "Study recipe (*.json)")
        if path:
            self._dialog_action(lambda: self.load_recipe(path))

    def _choose_save_recipe(self):
        # save_spec writes exactly the chosen file, so the seam's default output set (that
        # one path) is the honest declaration. None is a cancel.
        path = ask_export_path(self, title="Save study recipe", suggestion="incubation-study.json",
                               filters="Study recipe (*.json)")
        if path is not None:
            self._dialog_action(lambda: self.save_recipe(path))

    def _choose_result(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open saved study bundle")
        if path:
            self._dialog_action(lambda: self.load_result(path))
