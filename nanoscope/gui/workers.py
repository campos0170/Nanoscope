from __future__ import annotations

import threading
from pathlib import Path

from PySide6 import QtCore

from ..core.types import SampleRange
from ..io.base import open_document, probe_path
from ..io.errors import LoadCancelled, ReaderError
from .worker_thread import (
    DEFAULT_JOIN_MS,
    JOB_LOAD,
    retire_worker_thread,
    start_worker_thread,
)

__all__ = ["ProbeWorker", "LoadWorker", "LoadController"]


class ProbeWorker(QtCore.QObject):
    """probe_path() off the main thread. Cheap ([VERIFIED] 0.001–0.010 s) but it touches the disk,
    so it does not run on the GUI thread."""

    finished = QtCore.Signal(object)   # RecordingProbe
    failed = QtCore.Signal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.finished.emit(probe_path(self._path))
        except ReaderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # never let a neo/h5py/nptdms traceback escape the worker
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class LoadWorker(QtCore.QObject):
    """open_document() off the main thread, in chunks, reporting progress and polling cancel."""

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)   # Document — numpy payloads cross threads fine
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, path: Path, sample_range: SampleRange | None = None) -> None:
        super().__init__()
        self._path = path
        self._range = sample_range
        self._cancel = threading.Event()      # NOT a slot. See cancel().

    def cancel(self) -> None:
        """*** A PLAIN METHOD, CALLED DIRECTLY FROM THE GUI THREAD. ***
        Deliberately NOT a @Slot, and NEVER invoked via signal/slot or QMetaObject.invokeMethod.

        [VERIFIED] the canonical Qt tutorial pattern — a @Slot cancel() on a worker moved to a
        QThread — is delivered through THAT THREAD'S event loop, which is blocked inside run(). The
        call queues behind run() and arrives only after the load has already finished. A
        threading.Event set from the GUI thread is seen by the worker between chunks immediately.
        No Qt involved.
        """
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        """Reader polls `cancel` between chunks and raises LoadCancelled. Never lets an exception
        escape: emits cancelled() on LoadCancelled, failed(str) on ReaderError, and
        failed("Type: msg") on anything unexpected."""
        try:
            doc = open_document(
                self._path,
                self._range,
                progress=lambda frac, msg: self.progress.emit(frac, msg),
                cancel=self._cancel.is_set,
            )
            self.finished.emit(doc)
        except LoadCancelled:
            self.cancelled.emit()
        except ReaderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class LoadController(QtCore.QObject):
    """Owns thread lifetime. QThread + moveToThread — NOT QRunnable/QThreadPool: this is ONE
    long-lived task needing progress, cancel and deterministic teardown, which is exactly QThread's
    shape. ONE worker at a time; start() aborts any in-flight load first, so opening several files
    is SEQUENTIAL through this one controller.
    """

    probed = QtCore.Signal(object)          # RecordingProbe
    loaded = QtCore.Signal(object)          # Document
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(float, str)
    cancelled = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QtCore.QThread | None = None
        self._worker: QtCore.QObject | None = None

    # -- public API ------------------------------------------------------------------------------
    def probe(self, path: Path) -> None:
        self.abort()
        worker = ProbeWorker(path)
        worker.finished.connect(self._on_probed)
        worker.failed.connect(self._on_failed)
        self._spin_up(worker, "probe")

    def start(self, path: Path, sample_range: SampleRange | None = None) -> None:
        self.abort()
        worker = LoadWorker(path, sample_range)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_loaded)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._spin_up(worker, "load")

    def abort(self, wait_ms: int = DEFAULT_JOIN_MS) -> None:
        """Ask the in-flight load to stop and detach it. EMITS NOTHING; the caller resets
        its own UI. Called from start() before a new load, from the Cancel button, and from
        MainWindow.closeEvent().

        The reader polls the cancel Event between chunks, so a load stops promptly. What it
        does NOT do -- and must never do again -- is ``terminate()`` a thread that will not
        stop: that kills it wherever it happens to be, including inside the GIL, and the
        unbounded ``wait()`` that used to follow then hung the entire application (C19).
        Retirement is shared with the analysis controllers; see
        :mod:`nanoscope.gui.worker_thread`."""
        self.abort_stopped(wait_ms)

    def abort_stopped(self, wait_ms: int = DEFAULT_JOIN_MS) -> bool:
        """:meth:`abort`, reporting whether the load thread had actually stopped."""
        worker, thread = self._worker, self._thread
        self._worker, self._thread = None, None
        if isinstance(worker, LoadWorker):
            worker.cancel()
        return retire_worker_thread(worker, thread, wait_ms=wait_ms)

    def is_busy(self) -> bool:
        """Is a probe or load actually in flight?

        This used to answer True FOREVER after the first load: only ``abort()`` ever
        cleared ``_thread``, and the terminal slots did not, so a load that SUCCEEDED
        left the controller permanently 'busy'. It is what the Cancel button's
        ``_has_cancellable_job()`` consults and what the run gate consults, so a stuck
        True is a Cancel button offering to stop a job that finished minutes ago.
        :meth:`_retire` now runs on every terminal slot, exactly as the analysis
        controllers already did."""
        return self._thread is not None

    # -- internals -------------------------------------------------------------------------------
    def _spin_up(self, worker: QtCore.QObject, label: str) -> None:
        self._worker = worker
        self._thread = start_worker_thread(
            worker, domain=JOB_LOAD, label=label, owner=self)

    def _retire(self) -> None:
        """Join the finished thread and forget it, BEFORE announcing the outcome — so a
        slot reacting to ``loaded``/``failed`` sees ``is_busy() is False`` and may start
        the next queued open. Mirrors ``AnalysisController._stop``."""
        worker, thread = self._worker, self._thread
        self._worker, self._thread = None, None
        retire_worker_thread(worker, thread)

    def _is_active(self) -> bool:
        """A terminal signal is honoured only if it comes from the CURRENT worker. Guards against a
        stale signal queued by a worker that abort() already detached (e.g. an aborted load's
        cancelled() that was queued while the GUI thread was blocked in wait()).

        ``self._worker is not None`` matters: after abort() a detached worker whose C++ object
        has already been collected reports ``sender() is None``, and against a ``_worker`` of
        None that compared EQUAL — so the guard passed and the aborted load announced itself
        (C40)."""
        return self._worker is not None and self.sender() is self._worker

    @QtCore.Slot(object)
    def _on_probed(self, probe: object) -> None:
        if not self._is_active():
            return
        self._retire()
        self.probed.emit(probe)

    @QtCore.Slot(float, str)
    def _on_progress(self, frac: float, msg: str) -> None:
        if self._is_active():
            self.progressed.emit(frac, msg)

    @QtCore.Slot(object)
    def _on_loaded(self, doc: object) -> None:
        if not self._is_active():
            return
        self._retire()
        self.loaded.emit(doc)

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        if not self._is_active():
            return
        self._retire()
        self.failed.emit(msg)

    @QtCore.Slot()
    def _on_cancelled(self) -> None:
        if not self._is_active():
            return
        self._retire()
        self.cancelled.emit()
