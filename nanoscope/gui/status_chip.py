"""The calibration chip — a small, clickable status-bar chip that makes the deferred
current-gain calibration IMPOSSIBLE to forget.

Until a real gain is dialled in, every blockage depth, ECD, conductance and noise
magnitude the app reports is ~2500x off and silently trusted. This chip renders the
:class:`~nanoscope.gui.workflow_status.CalibrationStatus` for the active recording so the
state is always on screen:

* UNCALIBRATED — amber, ``⚠ Current gain uncalibrated — magnitudes not physical``;
* CALIBRATED   — green,  ``✓ Gain set: <detail>``;
* DAMAGED      — amber,  ``⚠ Gain set: <detail> — channel non-finite, magnitudes not
  physical``, with ``Session.calibration_issues``' own sentence as the tooltip. The green
  tick was measured being drawn over a channel whose every sample was -inf (gain 1e38 on
  the owner's -1000 mV ABF); the chip may not certify what the File info panel is
  reporting as ruined;
* NO_CHANNEL   — neutral/dim, ``No current channel``; when the user's own "Treat this
  channel as" override is why there is no current channel, the tooltip is the workflow
  strip's CALIBRATE reason verbatim, so the chip and the strip never disagree about why.

Clicking it emits :attr:`clicked`; ``main_window`` maps that to revealing the Calibration
panel (this widget never reaches for another panel — it imports only Qt and the pure
status module).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .workflow_status import CalibrationState, CalibrationStatus

__all__ = ["CalibrationChip"]

# Colours consistent with the app's dark instrument theme (see gui/app.py). The status
# accents match the Calibration panel's provenance badges: amber = warning, green = good,
# dim = neutral. Exposed as constants so tests can assert the accent without parsing CSS.
AMBER = "#FFB74D"
GREEN = "#81C784"
NEUTRAL = "#8b9199"

_DARK_TEXT = "#16181c"        # legible on the solid amber / green chips
_AMBER_BORDER = "#c9822b"
_GREEN_BORDER = "#5c9a63"
_NEUTRAL_BG = "#2a2f36"
_NEUTRAL_BORDER = "#3a4048"

_UNCAL_TEXT = "⚠ Current gain uncalibrated — magnitudes not physical"

_TOOLTIPS = {
    CalibrationState.UNCALIBRATED: (
        "The current-channel gain is still at its default (uncalibrated). Blockage depth, "
        "ECD, conductance and noise magnitudes are NOT physical until you set the real gain.\n"
        "Click to open the Calibration panel."),
    CalibrationState.CALIBRATED: (
        "The current-channel gain has been set, so reported magnitudes are physical.\n"
        "Click to review the calibration."),
    CalibrationState.NO_CHANNEL: (
        "This recording has no live current channel to calibrate."),
    # DAMAGED: the tooltip is the issue sentence itself (see set_status); this is the
    # fallback for a status that somehow carries none.
    CalibrationState.DAMAGED: (
        "The current-channel gain made the channel non-finite, so reported magnitudes are "
        "NOT physical.\nClick to open the Calibration panel and lower the gain."),
}

_CLICK_TO_FIX = "\nClick to open the Calibration panel and lower the gain."

#: The neutral invitation, for a status whose ``issue`` is not about a ruined channel —
#: today the NO_CHANNEL state whose cause is the user's own "Treat this channel as"
#: override, where "lower the gain" would be nonsense advice.
_CLICK_TO_OPEN = "\nClick to open the Calibration panel."


class CalibrationChip(QtWidgets.QFrame):
    """A rounded, clickable calibration-status chip for the status bar."""

    clicked = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CalibrationChip")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._accent: str = NEUTRAL

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 10, 3)
        lay.setSpacing(0)
        self._label = QtWidgets.QLabel()
        # PLAIN TEXT, EXPLICITLY. The CALIBRATED chip reads "✓ Gain set: <detail>", and
        # ``detail`` ends in the channel's UNITS -- ``cal.target_units or info.raw_units``,
        # and ``raw_units`` comes verbatim out of the file header (see
        # ``workflow_status._calibration_detail``). A QLabel left at Qt::AutoText runs
        # mightBeRichText() over its text and parses it as HTML, so a unit string containing
        # a '<' followed by a tag name would be swallowed on its way to the screen -- on the
        # one chip whose whole job is saying whether reported magnitudes are physical, and in
        # exactly the part that says what they are physical IN.
        self._label.setTextFormat(QtCore.Qt.PlainText)
        self._label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        lay.addWidget(self._label)

        self._status = CalibrationStatus(state=CalibrationState.NO_CHANNEL,
                                         label="No current channel")
        self.set_status(self._status)

    # -- state -------------------------------------------------------------------------------
    def set_status(self, status: CalibrationStatus) -> None:
        """Re-render the chip from a :class:`CalibrationStatus`."""
        self._status = status
        text, self._accent, bg, fg, border = self._style_for(status)
        self._label.setText(text)
        self.setStyleSheet(
            f"#CalibrationChip {{ background:{bg}; border:1px solid {border}; "
            f"border-radius:9px; }}"
            f"#CalibrationChip QLabel {{ color:{fg}; background:transparent; "
            f"font-weight:600; }}")
        if status.issue:
            # THE STATUS'S OWN SENTENCE, VERBATIM, WHENEVER IT CARRIES ONE — not only for
            # DAMAGED. For DAMAGED it names the channel, the gain, the count of non-finite
            # samples and the remedy, and is the same text the File info panel prints. For
            # NO_CHANNEL it is the strip's own CALIBRATE reason, which says the user's
            # "Treat this channel as" override is why there is no current channel.
            #
            # WHY THE GUARD WENT (S7-D3 / S7-X8, measured): the reason is derived once in
            # ``workflow_status`` and reached three of its four consumers. After IN15
            # (header current) was re-declared a voltage, the strip's CALIBRATE tooltip read
            # "(IN15 is declared current by the file but your Calibration-panel override
            # treats it as voltage)" while the chip one glance away still said "This
            # recording has no live current channel to calibrate." — flatly denying a
            # channel the user had just told the app about, on the app's own trust
            # indicator. The producer already populated ``issue`` for this case; only this
            # consumer was gated on DAMAGED.
            self.setToolTip(status.issue + (
                _CLICK_TO_FIX if status.state is CalibrationState.DAMAGED
                else _CLICK_TO_OPEN))
        else:
            self.setToolTip(_TOOLTIPS.get(status.state, ""))
        # Accessible name for screen readers / test introspection.
        self.setAccessibleName(f"Calibration status: {text}")

    def status(self) -> CalibrationStatus:
        return self._status

    def text(self) -> str:
        """The rendered chip text (test/introspection helper)."""
        return self._label.text()

    def accent(self) -> str:
        """The active SEMANTIC status colour: :data:`AMBER` (uncalibrated), :data:`GREEN`
        (calibrated) or :data:`NEUTRAL` (no channel) — independent of the drawn background."""
        return self._accent

    @staticmethod
    def _style_for(status: CalibrationStatus) -> tuple[str, str, str, str, str]:
        """(text, accent, background, foreground, border) for a status. ``accent`` is the
        semantic status colour; the amber/green states fill the chip with it, the neutral
        state draws a dim chip whose accent is still :data:`NEUTRAL`."""
        if status.state is CalibrationState.UNCALIBRATED:
            return _UNCAL_TEXT, AMBER, AMBER, _DARK_TEXT, _AMBER_BORDER
        if status.state is CalibrationState.CALIBRATED:
            detail = status.detail or "current gain"
            return f"✓ Gain set: {detail}", GREEN, GREEN, _DARK_TEXT, _GREEN_BORDER
        if status.state is CalibrationState.DAMAGED:
            detail = status.detail or "current gain"
            return (f"⚠ Gain set: {detail} — channel non-finite, magnitudes not physical",
                    AMBER, AMBER, _DARK_TEXT, _AMBER_BORDER)
        return "No current channel", NEUTRAL, _NEUTRAL_BG, NEUTRAL, _NEUTRAL_BORDER

    # -- interaction -------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)
