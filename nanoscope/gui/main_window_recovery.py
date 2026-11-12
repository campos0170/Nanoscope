"""The crash-recovery half of ``MainWindow``: the autosave store and its scheduler, the snapshot
hand-off to the writer, the health line, the offer at start-up, and recover and discard.

``changeEvent`` stays in ``main_window.py``: it is a Qt override. So do ``_on_application_state_changed``,
``_on_recovery_saved``, ``_on_recovery_failed`` and ``_on_stale_recovery_saved``: they are registered
``@Slot`` methods, and no ``@Slot`` method leaves the class body (see the comment after the imports
in ``main_window.py``).

``_RecoveryMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
from collections.abc import Mapping
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .main_window_shared import (
    _RECOVERY_RECOVER,
    _RECOVERY_DISCARD,
    _RECOVERY_SHOW,
    _RecoverySnapshot,
    _write_recovery_snapshot,
    _DETECTOR_LABELS,
    _InfoBar,
)


class _RecoveryMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    # -- crash-recovery autosave ---------------------------------------------------------------
    def _init_recovery(self) -> None:
        """Create the Stage 2 recovery store and idle/deadline scheduler.

        Recovery is deliberately optional at this seam.  A read-only or otherwise broken
        application-support directory must cost the user crash recovery, never the ability
        to launch NanoScope and save a real project.  The controller reports later write
        failures through the status bar while leaving the user project untouched.
        """
        if not self._recovery_enabled:
            return
        try:
            from ..core.recovery import RecoveryStore
            from .recovery_controller import RecoveryController

            # ``hold_liveness``: this session's live.lock tells a second instance that
            # the lineage is a running window's, not a crash (sweep 11, s7).
            store = RecoveryStore(root=self._recovery_root, hold_liveness=True)
            controller = RecoveryController(
                self._capture_recovery_snapshot,
                lambda snapshot, revision, _store=store:
                    _write_recovery_snapshot(_store, snapshot, revision),
                parent=self,
            )
            controller.saved.connect(self._on_recovery_saved)
            controller.failed.connect(self._on_recovery_failed)
            controller.staleSaved.connect(self._on_stale_recovery_saved)
        except Exception as exc:  # noqa: BLE001 - recovery may not prevent application launch
            self._recovery_store = None
            self._recovery_controller = None
            self.statusBar().showMessage(
                f"Crash recovery is unavailable: {exc}. Save projects manually.", 12000)
            return
        self._recovery_store = store
        self._recovery_controller = controller

        # Recovery health, always on screen: when the last copy landed, or that it did
        # not.  A failure also raises its own banner, which stays until a copy lands.
        self._recovery_label = QtWidgets.QLabel("")
        self._recovery_label.setObjectName("recoveryHealthLabel")
        self.statusBar().addPermanentWidget(self._recovery_label)
        self._recovery_bar = _InfoBar()
        self._recovery_bar.setObjectName("recoveryFailureBar")
        holder = self._info_bar.parentWidget()
        lay = holder.layout() if holder is not None else None
        if lay is not None:
            lay.insertWidget(lay.indexOf(self._info_bar) + 1, self._recovery_bar)

        # Natural flush points (s4: an exported CSV was newer than the recovery copy).
        # A run is flushed as it starts, before its result can land; leaving the app
        # or the window is coalesced through _request_recovery_flush.
        self._analysis_controller.started.connect(self._flush_recovery_now)
        self._batch_controller.started.connect(self._flush_recovery_now)
        app = QtGui.QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_application_state_changed)

    def _flush_recovery_now(self, *_args) -> bool:
        """Capture the newest unsnapshotted revision now; the write runs on the worker."""
        controller = self._recovery_controller
        flush = getattr(controller, "flush", None)
        if not callable(flush):
            return False
        try:
            return bool(flush())
        except Exception:  # noqa: BLE001 - a flush point never interrupts the user's action
            return False

    def _request_recovery_flush(self) -> None:
        """Coalesce bursts (focus ping-pong, app switching) into one flush next turn."""
        if self._recovery_controller is None or self._recovery_flush_queued:
            return
        self._recovery_flush_queued = True
        QtCore.QTimer.singleShot(0, self, self._run_requested_recovery_flush)

    def _run_requested_recovery_flush(self) -> None:
        self._recovery_flush_queued = False
        self._flush_recovery_now()

    def _capture_recovery_snapshot(self, requested_revision: int):
        """Return an immutable snapshot for a recovery worker, on the GUI thread.

        The live project records intentionally share pandas frames with the editing
        surfaces.  They are safe for the synchronous explicit save, but handing those
        references to a worker would let a curation click mutate a table while Parquet is
        serialising it.  ``clone_project_state`` is therefore part of the thread boundary,
        not an implementation detail of the writer.
        """
        if not self._visible_project_dirty() or self._restoring:
            return None
        # Catch an in-place edit whose author forgot to emit its panel signal.  This can
        # advance the revision; the snapshot records the post-checkpoint value rather than
        # the timer's older request.
        self._checkpoint_everything()
        if not self._visible_project_dirty():
            return None
        from ..core.recovery import clone_project_state

        state = clone_project_state(self._gather_project_state())
        return _RecoverySnapshot(
            revision=int(self._session_revision),
            state=state,
            source_project_path=(self._project_path or
                                 self._recovery_suggested_project_path),
            covered_explicit_save_revision=int(self._explicit_saved_revision),
        )

    def _persist_recovery_snapshot(self, snapshot: _RecoverySnapshot,
                                   revision: int | None = None):
        """Direct test/support seam for writing an already-detached snapshot."""
        store = self._recovery_store
        if store is None:
            return None
        return _write_recovery_snapshot(store, snapshot, revision)

    def _reset_recovery_health(self, *, saved: bool = False) -> None:
        """Clear the failure banner; the label says ``saved HH:MM`` or nothing."""
        self._recovery_failing = False
        self._recovery_failed_revision = 0
        if not saved:
            self._recovery_saved_at = None
        if self._recovery_bar is not None:
            self._recovery_bar.hide()
        if saved:
            self._show_recovery_health()
        elif self._recovery_label is not None:
            self._recovery_label.setText("")
            self._recovery_label.setToolTip("")
            self._recovery_label.setStyleSheet("")

    @staticmethod
    def _recovery_time_text(stamp: float | None) -> str:
        if stamp is None:
            return ""
        from datetime import datetime

        return datetime.fromtimestamp(float(stamp)).strftime("%H:%M")

    def _show_recovery_health(self, error: str | None = None) -> None:
        label = self._recovery_label
        if label is None:
            return
        if self._recovery_failing:
            label.setText("Recovery copy not saved")
            label.setStyleSheet("color:#e0a040;")
            label.setToolTip(f"The last attempt failed: {error}" if error else "")
        elif self._recovery_saved_at is not None:
            label.setText(f"Recovery copy saved {self._recovery_time_text(self._recovery_saved_at)}")
            label.setStyleSheet("")
            label.setToolTip("A crash-recovery copy of the unsaved work was written then.")

    @staticmethod
    def _candidate_value(candidate, name: str, default=None):
        value = getattr(candidate, name, default)
        return default if value is None else value

    @staticmethod
    def _recovery_when_text(when) -> str:
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(float(when)).astimezone()
            return (f"{dt:%b} {dt.day}, {dt:%Y} at "
                    f"{dt.strftime('%I:%M:%S %p').lstrip('0')}")
        except (ValueError, TypeError, OSError):
            return str(when)

    @staticmethod
    def _recovery_writer_text(writer) -> str:
        """``NanoScope 1.2 (build …)``, plus who ran it when it was not the app."""
        if not isinstance(writer, Mapping):
            return ""
        text = f"NanoScope {writer.get('version') or '?'}"
        build = writer.get("build_id")
        if build:
            text += f" (build {build})"
        if writer.get("frozen") is False:
            text += f", run from Python ({writer.get('executable') or 'unknown interpreter'})"
        return text

    def _recovery_candidate_detail(self, candidate) -> str:
        """Name the work: recordings, tables with row counts, last edit and the build.

        The prompt used to say only "unsaved work from an unnamed analysis (revision N)",
        which no one could tell from a stray script's copy; the incident check found the
        owner discarding fourteen of them about a second apart.
        """
        source = self._candidate_value(candidate, "source_project_path", None)
        revision = self._candidate_value(candidate, "revision", "?")
        when = self._candidate_value(candidate, "timestamp", None)
        label = Path(source).name if source else "an unnamed analysis"
        detail = f"NanoScope found unsaved work from {label} (revision {revision})"
        if when:
            detail += f", last edited {self._recovery_when_text(when)}"
        detail += "."
        summary = getattr(candidate, "summary", None)
        if summary is not None:
            recordings = list(getattr(summary, "recordings", ()) or ())
            if recordings:
                shown = ", ".join(recordings[:4])
                more = f" and {len(recordings) - 4} more" if len(recordings) > 4 else ""
                detail += f"\nRecordings: {shown}{more}."
            tables = list(getattr(summary, "tables", ()) or ())
            if tables:
                lines = [f"{_DETECTOR_LABELS.get(module, module)} on {name}: "
                         f"{accepted} of {rows} event(s) accepted"
                         for module, name, rows, accepted in tables[:5]]
                if len(tables) > 5:
                    lines.append(f"and {len(tables) - 5} more table(s)")
                detail += "\nTables: " + "; ".join(lines) + "."
            elif not recordings:
                detail += "\nIt holds no recordings or event tables."
            n_measurements = int(getattr(summary, "measurements", 0) or 0)
            if n_measurements:
                detail += f"\nHand measurements: {n_measurements}."
        writer = self._recovery_writer_text(getattr(candidate, "writer", None))
        if writer:
            detail += f"\nWritten by {writer}."
        return detail

    def _choose_recovery_action(self, candidate) -> str | None:
        """Offer Recover / Discard / Show in Finder for one validated candidate."""
        from ..core.recovery import DEFAULT_DISCARD_KEEP_SECONDS

        detail = self._recovery_candidate_detail(candidate)
        keep_days = int(round(DEFAULT_DISCARD_KEEP_SECONDS / 86400))
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Recover unsaved NanoScope work")
        box.setText("NanoScope did not finish its previous session cleanly.")
        box.setInformativeText(
            detail + "\n\nRecover opens that snapshot as unsaved work. Discard moves it "
            f"to the Discarded folder beside it, where it is kept for {keep_days} days.")
        recover = box.addButton("Recover", QtWidgets.QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
        show = box.addButton("Show in Finder", QtWidgets.QMessageBox.ActionRole)
        not_now = box.addButton("Not now", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(recover)
        box.setEscapeButton(not_now)
        box.exec()
        clicked = box.clickedButton()
        if clicked is recover:
            return _RECOVERY_RECOVER
        if clicked is discard:
            return _RECOVERY_DISCARD
        if clicked is show:
            return _RECOVERY_SHOW
        return None

    def _unreadable_recovery_text(self, unreadable) -> str:
        count = len(unreadable)
        writers = sorted({text for text in (
            self._recovery_writer_text(getattr(item, "writer", None)) for item in unreadable)
            if text})
        text = (f"{count} recovery cop{'y' if count == 1 else 'ies'} could not be read by "
                f"this build")
        if writers:
            text += f" (written by {'; '.join(writers)})"
        text += (". They are kept, not deleted: a newer NanoScope may open them, and Show in "
                 "Finder shows where they are. Reason: "
                 + str(getattr(unreadable[0], "reason", "unknown")) + ".")
        return text

    def _report_unreadable_recoveries(self, unreadable) -> str | None:
        """Say how many copies this build cannot read; Show in Finder or OK."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle("Recover unsaved NanoScope work")
        box.setText(f"{len(unreadable)} recovery cop"
                    f"{'y' if len(unreadable) == 1 else 'ies'} could not be read by this build.")
        box.setInformativeText(self._unreadable_recovery_text(unreadable))
        show = box.addButton("Show in Finder", QtWidgets.QMessageBox.ActionRole)
        ok = box.addButton(QtWidgets.QMessageBox.Ok)
        box.setDefaultButton(ok)
        box.setEscapeButton(ok)
        box.exec()
        return _RECOVERY_SHOW if box.clickedButton() is show else None

    def _candidate_path(self, candidate) -> Path | None:
        for name in ("path", "envelope_path", "payload_path", "directory"):
            value = getattr(candidate, name, None)
            if value:
                return Path(value)
        return None

    def _show_recovery_in_finder(self, candidate) -> None:
        path = self._candidate_path(candidate)
        if path is None:
            return
        target = path if path.is_dir() else path.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))

    def _offer_startup_recovery(self) -> None:
        """Inspect completed prior-session generations once, after the window is visible."""
        if self._startup_recovery_checked:
            return
        self._startup_recovery_checked = True
        store = self._recovery_store
        if store is None:
            return
        try:
            scan = getattr(store, "scan", None)
            if callable(scan):
                found = scan()
                candidates, unreadable = list(found.candidates), list(found.unreadable)
            else:
                candidates, unreadable = store.list_candidates(), []
        except Exception as exc:  # noqa: BLE001 - discovery cannot prevent normal use
            self.statusBar().showMessage(
                f"Could not inspect crash-recovery copies: {exc}.", 12000)
            return
        # Never hidden in silence (s11c/d): a truncated copy, or one from a newer build
        # that this one cannot read, is still somebody's work.
        if unreadable:
            while self._report_unreadable_recoveries(unreadable) == _RECOVERY_SHOW:
                self._show_recovery_in_finder(unreadable[0])
        while candidates:
            candidate = candidates[0]
            action = self._choose_recovery_action(candidate)
            while action == _RECOVERY_SHOW:
                self._show_recovery_in_finder(candidate)
                action = self._choose_recovery_action(candidate)
            if action == _RECOVERY_RECOVER:
                self._recover_candidate(candidate)
                return
            if action != _RECOVERY_DISCARD:
                return
            self._discard_recovery(candidate)
            # More than one crashed process can leave valid work.  Discard is a decision
            # about the candidate just named, not permission to hide every older session.
            try:
                candidates = store.list_candidates()
            except Exception:
                return

    def _safe_recovery_target(self, value) -> str | None:
        if not value:
            return None
        try:
            path = Path(str(value)).expanduser()
        except (TypeError, ValueError, OSError):
            return None
        if not path.is_absolute() or path.suffix.lower() != ".nanoscope":
            return None
        return str(path)

    def _recover_candidate(self, candidate) -> bool:
        store = self._recovery_store
        if store is None:
            return False
        try:
            recovered = store.load(candidate)
        except Exception as exc:  # noqa: BLE001 - a bad generation falls back in the store
            QtWidgets.QMessageBox.warning(
                self, "Recover unsaved work",
                f"Could not recover the saved snapshot:\n{exc}")
            return False

        state = getattr(recovered, "state", None)
        if state is None:
            state = getattr(recovered, "project", None)
        if state is None:
            QtWidgets.QMessageBox.warning(
                self, "Recover unsaved work",
                "The recovery copy did not contain a NanoScope project snapshot.")
            return False

        self._clear_session()
        self._recovered_candidate = candidate
        source = getattr(recovered, "source_project_path", None)
        if source is None:
            source = getattr(candidate, "source_project_path", None)
        self._recovery_suggested_project_path = self._safe_recovery_target(source)
        revision = int(getattr(recovered, "revision",
                               getattr(candidate, "revision", 0)) or 0)
        covered = int(getattr(
            recovered, "covered_explicit_save_revision",
            getattr(candidate, "covered_explicit_save_revision", 0)) or 0)
        self._session_revision = max(self._session_revision, revision)
        self._explicit_saved_revision = max(self._explicit_saved_revision, covered)
        self._restore_is_recovery = True
        self._begin_restore(state)
        return True

    def _discard_recovery(self, candidate=None) -> None:
        """Retire the surfaced lineage and this session's queued copies, for good from
        discovery's point of view, but kept under ``Discarded/`` for seven days: every
        caller is a user's Discard (the startup prompt, the close guard, the guard before
        Open/New), and one click there must not be beyond undoing."""
        controller = self._recovery_controller
        if controller is not None:
            reset = getattr(controller, "reset_session", None)
            if callable(reset):
                reset()
        store = self._recovery_store
        if store is not None:
            discard_through = getattr(store, "discard_through", None)
            if not callable(discard_through):
                discard_through = store.retire_through
            try:
                if candidate is not None:
                    if getattr(candidate, "session_id", None) == store.session_id:
                        discard_through(int(getattr(candidate, "revision", 0)))
                    else:
                        # A surfaced prior-process lineage will never be reused; its
                        # discard tombstone makes the user's choice definitive.
                        store.discard(candidate)
                # This process DOES continue after Open Project.  Fence its outgoing
                # revisions, do not tombstone every future revision in the launch.
                discard_through(int(self._session_revision))
            except Exception as exc:  # noqa: BLE001 - report, but a close must remain possible
                self.statusBar().showMessage(
                    f"Could not discard the crash-recovery copy: {exc}.", 12000)
        # The copies went to Discarded/ and the work they guarded was let go: neither
        # "saved HH:MM" nor a failure banner describes the session that follows.
        self._reset_recovery_health()
        if candidate is None or candidate is self._recovered_candidate:
            self._recovered_candidate = None

    def _retire_recovery_through(self, revision: int) -> None:
        controller = self._recovery_controller
        if controller is not None:
            fence = getattr(controller, "retire_through", None)
            if callable(fence):
                fence(int(revision))
        if int(revision) >= self._session_revision and self._recovery_label is not None:
            # The project file now holds everything: neither "saved HH:MM" (that copy
            # is being retired) nor a failure banner describes anything any more.
            self._reset_recovery_health()
        store = self._recovery_store
        if store is None:
            return
        try:
            candidate = self._recovered_candidate
            store.retire_after_save(
                int(revision), recovered_candidate=candidate)
            if candidate is not None:
                self._recovered_candidate = None
        except Exception as exc:  # noqa: BLE001 - real project save already succeeded
            self.statusBar().showMessage(
                f"The project was saved, but its old recovery copy could not be removed: "
                f"{exc}.", 12000)
