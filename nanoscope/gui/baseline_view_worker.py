"""Cancellable latest-request delivery for main-trace baseline drawings."""
from __future__ import annotations

from PySide6 import QtCore

from .analysis_worker import CallableWorker
from .worker_thread import JOB_NONE, retire_worker_thread, start_worker_thread


class BaselineViewWorker(QtCore.QObject):
    ready = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = self._thread = None
        self._identity = None
        if parent is not None:
            parent.destroyed.connect(self.abort)

    def submit(self, identity, calculate):
        self.abort()
        self._identity = identity
        worker = CallableWorker(calculate)
        self._worker = worker
        worker.finished.connect(self._finished)
        worker.failed.connect(self._failed)
        worker.cancelled.connect(self._cancelled)
        self._thread = start_worker_thread(worker, domain=JOB_NONE,
                                           label='baseline display', owner=self)

    def abort(self, *_):
        worker, thread = self._worker, self._thread
        self._worker = self._thread = self._identity = None
        if worker is not None:
            worker.cancel()
        # The kernel observes cancel_scope; an indivisible numerical call may finish
        # later. Retire owns that lifetime and disconnects it from the current view.
        retire_worker_thread(worker, thread, wait_ms=0)

    @QtCore.Slot(object)
    def _finished(self, result):
        if self.sender() is not self._worker:
            return
        identity = self._identity
        self.abort()
        self.ready.emit(identity, result)

    @QtCore.Slot(str)
    def _failed(self, message):
        if self.sender() is not self._worker:
            return
        identity = self._identity
        self.abort()
        self.failed.emit(identity, message)

    @QtCore.Slot()
    def _cancelled(self):
        if self.sender() is self._worker:
            self.abort()
