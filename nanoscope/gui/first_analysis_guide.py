"""A navigation-only first-analysis lesson; no scientific parameters are set here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtWidgets

CONTROL_LEVEL_CHOICE_KEY = "onboarding/control_level_choice_v1"
EXISTING_CONTROL_LEVEL_KEY = "inspector/detection_control_level"


@dataclass(frozen=True)
class GuideStep:
    title: str
    instructions: str
    checkpoint: str
    actions: tuple[tuple[str, str], ...]


STEPS = (
    GuideStep("1 · Open a recording",
        "Open a recording and select its current channel in Files. Inspect the entire trace before zooming. "
        "You can instead create a clearly named SYNTHETIC practice CSV; it is added alongside existing "
        "recordings, never substituted for them. No detection or calibration is run automatically.",
        "I can identify the selected recording and current channel.",
        (("Open recording…", "open_recording"), ("Create synthetic practice recording…", "practice"))),
    GuideStep("2 · Verify acquisition and units",
        "Check the sample rate, current units, gain and offset against acquisition records. Entering a gain "
        "does not prove it is correct. For the generated practice CSV only: time is in seconds, current "
        "numbers are already pA, sampling is 10,000 Hz, and the intended gain/offset are 1/0 with target "
        "units pA. CSV units may need declaring explicitly. Never copy this calibration to real data.",
        "I understand the time base and have checked the meaning of the current values.",
        (("Show Calibration", "calibration"),)),
    GuideStep("3 · Inspect noise and baseline",
        "Open Baseline reference and choose a reference for THIS recording. Automatic retains the existing "
        "estimator: Measure this recording proposes windows; Use these values confirms them. The overlay "
        "before confirmation still uses the old windows. Known open-pore current accepts a signed current "
        "in calibrated pA, for example −120 pA, not its positive magnitude. Selected open-pore intervals "
        "uses trustworthy quiet regions: enter ordered start/end times in seconds from the recording start "
        "or copy Measure cursors. Use Constant for a stable level or Linear with at least two intervals "
        "for supported drift; intervening drift and held endpoints remain assumptions. Never copy another "
        "recording's reference blindly. In either user-reference mode, press Preview reference and detection "
        "to inspect the baseline, polarity, noise and candidates without changing stored results. Then run "
        "detection explicitly if appropriate. Changing a user reference needs fresh detection, not merely "
        "remeasuring old event depths. Warnings stay in Details, not on the plot. A health pass does not "
        "certify the physical open pore, and user-supplied information is not independently verified. "
        "Noise spectrum is available before detection. The practice trace has four downward rectangular "
        "blockades near 0.8, 1.8, 2.8 and 3.8 s, each 40 ms long and 25 pA deep, with noise and slow drift. "
        "Those are generator settings, not promises about detector output.",
        "I inspected a quiet interval and checked whether the baseline tracks the open pore rather than events.",
        (("Show Open-pore baseline", "baseline"), ("Open / compute Noise spectrum…", "noise"))),
    GuideStep("4 · Detect candidates",
        "Open Detection, inspect polarity, thresholds, dwell limits and the active processing chain, then "
        "press Run detection yourself. Existing settings are preserved. Essential controls simplify the "
        "view, not the analysis. Advanced controls remain available; enabled/nondefault settings are disclosed. "
        "A changed setting may make old results stale until you rerun.",
        "I checked the settings and compared the detected spans with the trace.",
        (("Show Detection", "detection"), ("Use Essential controls", "essential"),
         ("Use Advanced controls", "advanced"))),
    GuideStep("5 · Review the events",
        "Use Event inspector for one event and Event gallery for a collection. Inspect boundaries, baseline "
        "and shape; accept/reject deliberately. Automatically accepted candidates are not human-reviewed. "
        "Median depth and Selected amplitude are different columns; changing the red level does not repair "
        "missed events or incorrect boundaries. Return to the original trace when uncertain.",
        "I reviewed example events and understand acceptance versus human review.",
        (("Event inspector…", "event_inspector"), ("Event gallery…", "event_gallery"))),
    GuideStep("6 · Make a figure",
        "Open Distributions for histograms/scatter, Distribution summaries for ECDF/box/violin, or "
        "Comparison figures for cross-source panels. Use Data → Analysis → Appearance → Export. Check "
        "which sources and acceptance filter are selected. Opening a live tool can compute or update its "
        "view; appearance edits do not rerun analysis. A fitted curve is not evidence that its model is valid.",
        "I can name the source data, axis units and analysis behind my figure.",
        (("Distributions…", "distributions"), ("Analysis home…", "analysis_home"))),
    GuideStep("7 · Export and save",
        "Use the figure's Export controls for PNG/PDF/SVG, physical dimensions and print-readability details. "
        "Save to Figure Library freezes its picture and numbers; a bundle includes data/configuration/provenance. "
        "Save Project retains your workspace and library. An exported picture is not a resumable project, "
        "and a saved snapshot does not verify unavailable live recordings. Keep required source files too.",
        "I understand the difference between exporting a figure, saving a snapshot and saving the project.",
        (("Return to analysis", "analysis_current"), ("Save Project…", "save_project"))),
)


class FirstAnalysisGuide(QtWidgets.QDialog):
    """Self-paced, modeless guidance. Checkmarks are learning notes, not validation."""

    actionRequested = QtCore.Signal(str)
    controlLevelRequested = QtCore.Signal(str)

    def __init__(self, parent=None, *, context_provider=None, availability_provider=None):
        super().__init__(parent)
        self.setWindowTitle("NanoScope — Guided first analysis")
        self.setModal(False)
        self.resize(820, 570)
        self.setMinimumSize(600, 440)
        self._context_provider = context_provider
        self._availability_provider = availability_provider
        self._context_key = None
        self._checks = [False] * len(STEPS)
        root = QtWidgets.QVBoxLayout(self)
        self.context = QtWidgets.QLabel()
        self.context.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.context.setWordWrap(True)
        root.addWidget(self.context)
        note = QtWidgets.QLabel("Reading this guide does not change scientific settings or run analysis. "
                               "Clicking a tool opens its normal workflow and may compute that view. "
                               "Checkmarks mean ‘I checked this step’, not scientific certification.")
        note.setWordWrap(True)
        root.addWidget(note)
        self.first_use_choice = QtWidgets.QGroupBox("Choose your starting control view")
        choice_layout = QtWidgets.QVBoxLayout(self.first_use_choice)
        choice_note = QtWidgets.QLabel(
            "Essential shows the everyday controls; Advanced shows the full toolset. "
            "This changes visibility only—not filters, thresholds, calibration or results. You can switch later.")
        choice_note.setWordWrap(True)
        choice_layout.addWidget(choice_note)
        choices = QtWidgets.QHBoxLayout()
        self.essential_choice = QtWidgets.QPushButton("Start with Essential")
        self.advanced_choice = QtWidgets.QPushButton("Start with Advanced")
        for button, level in ((self.essential_choice, "essential"), (self.advanced_choice, "advanced")):
            button.clicked.connect(lambda _checked=False, value=level: self._choose_control_level(value))
            choices.addWidget(button)
        choice_layout.addLayout(choices)
        root.addWidget(self.first_use_choice)
        split = QtWidgets.QSplitter()
        self.steps = QtWidgets.QListWidget()
        self.steps.setAccessibleName("First-analysis steps")
        self.steps.addItems([step.title for step in STEPS])
        self.steps.setMaximumWidth(255)
        split.addWidget(self.steps)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        self.heading = QtWidgets.QLabel()
        self.heading.setStyleSheet("font-size:17px; font-weight:600")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.instructions = QtWidgets.QLabel()
        self.instructions.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.instructions.setWordWrap(True)
        self.instructions.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.instructions)
        self._buttons = []
        for _ in range(3):
            button = QtWidgets.QPushButton()
            button.clicked.connect(lambda _checked=False, btn=button: self._request_action(btn.property("route")))
            layout.addWidget(button)
            self._buttons.append(button)
        self.check = QtWidgets.QCheckBox("I checked this step")
        self.check.toggled.connect(self._checked)
        layout.addWidget(self.check)
        self.checkpoint = QtWidgets.QLabel()
        self.checkpoint.setWordWrap(True)
        layout.addWidget(self.checkpoint)
        layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)
        navigation = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QPushButton("Previous")
        self.next = QtWidgets.QPushButton("Next")
        close = QtWidgets.QPushButton("Close guide")
        navigation.addWidget(self.previous)
        navigation.addWidget(self.next)
        navigation.addStretch()
        navigation.addWidget(close)
        root.addLayout(navigation)
        self.previous.clicked.connect(lambda: self.steps.setCurrentRow(max(0, self.steps.currentRow() - 1)))
        self.next.clicked.connect(lambda: self.steps.setCurrentRow(min(len(STEPS) - 1, self.steps.currentRow() + 1)))
        close.clicked.connect(self.close)
        self.steps.currentRowChanged.connect(self._show_step)
        self.steps.setCurrentRow(0)
        self.refresh_context()

    def refresh_context(self):
        settings = QtCore.QSettings()
        self.first_use_choice.setVisible(not settings.contains(EXISTING_CONTROL_LEVEL_KEY)
                                          and not settings.contains(CONTROL_LEVEL_CHOICE_KEY))
        key, text = self._context_provider() if self._context_provider else (None, "No recording selected")
        if key != self._context_key:
            self._checks = [False] * len(STEPS)
            self._context_key = key
            self._update_step_names()
        self.context.setText(str(text))
        self._show_step(self.steps.currentRow())

    def _choose_control_level(self, level):
        if level not in {"essential", "advanced"}:
            raise ValueError("Unknown control level")
        # This explicit visibility preference is independent of a selected recording.
        # Existing expert preferences are never changed merely by opening the guide.
        QtCore.QSettings().setValue(CONTROL_LEVEL_CHOICE_KEY, level)
        self.controlLevelRequested.emit(level)
        self.first_use_choice.hide()

    def _show_step(self, index):
        if index < 0:
            return
        step = STEPS[index]
        self.heading.setText(step.title)
        self.instructions.setText(step.instructions)
        self.checkpoint.setText(step.checkpoint)
        blocker = QtCore.QSignalBlocker(self.check)
        self.check.setChecked(self._checks[index])
        del blocker
        for i, button in enumerate(self._buttons):
            button.setVisible(i < len(step.actions))
            if i < len(step.actions):
                label, route = step.actions[i]
                button.setText(label)
                button.setProperty("route", route)
                available, reason = self._availability_provider(route) if self._availability_provider else (True, "")
                button.setEnabled(bool(available))
                if available and route == "baseline":
                    reason = ("Inspect the open-pore reference for this recording. "
                              "Measure this recording proposes baseline windows; "
                              "Use these values confirms them.")
                button.setToolTip(reason)
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(index < len(STEPS) - 1)

    def _request_action(self, route):
        # Recheck live context rather than trusting the button's older display.
        available, _reason = self._availability_provider(route) if self._availability_provider else (True, "")
        if available:
            self.actionRequested.emit(route)

    def _checked(self, checked):
        index = self.steps.currentRow()
        if index >= 0:
            self._checks[index] = bool(checked)
            self._update_step_names()

    def _update_step_names(self):
        for index, step in enumerate(STEPS):
            self.steps.item(index).setText(("✓ " if self._checks[index] else "") + step.title)


def practice_trace():
    """Small reproducible pedagogical signal, not a detector benchmark or real data."""
    import numpy as np
    time = np.arange(50_000, dtype=float) / 10_000.0
    baseline = 100 + 4 * np.sin(2 * np.pi * time / 8)
    current = baseline + np.random.default_rng(20260911).normal(0, 1.2, time.size)
    for start in (.8, 1.8, 2.8, 3.8):
        current[int(start * 10_000):int(round((start + .04) * 10_000))] -= 25
    return time, current


def practice_path(path):
    """Keep the synthetic designation in the durable filename, even after Save As."""
    path = Path(path)
    stem = path.stem if path.suffix.lower() == ".csv" else path.name
    if not stem.upper().startswith("SYNTHETIC-"):
        stem = "SYNTHETIC-" + stem
    return path.with_name(stem + ".csv")


def write_practice_recording(path):
    from ..core.atomic_write import staged_write
    import numpy as np
    target = practice_path(path)
    time, current = practice_trace()
    with staged_write(target) as handle:
        np.savetxt(handle, np.column_stack((time, current)), delimiter=",", header="time,current",
                   comments="", fmt="%.17g")
    return target
