"""Regression coverage for the WA_DeleteOnClose crash precondition (2026-09-13).

THE HAZARD. MainWindow's stop-holders teardown sweep walks every QObject in its tree
(``vars(self).values()`` plus ``findChildren(QtCore.QObject)``) and calls ``close()`` on
each window-like one as a stop mechanism -- documented at ``MainWindow`` as deliberately
uniform: ``close()`` on a surface window is a hide, never a delete, because
``WA_DeleteOnClose`` is set nowhere in ``nanoscope/gui``. Two dialogs added on 2026-09-11
(``BuildInfoDialog``, and the open-pore reference ``OpenPorePreviewDialog`` built in
``AnalysisPanel._finish_reference_preview``) broke that invariant: both set
``WA_DeleteOnClose``, both are parented into the tree the sweep walks, and a Python
wrapper the app still held (``MainWindow._build_info_dialog`` conceptually, and the local
``dialog`` reference before this fix) could then be silently deleted at the C++ level by
an ordinary Close click or by teardown -- a use-after-free waiting for the next signal
into that wrapper. ``tests/gui/test_fup_wiring_and_suite_green.py`` guards the general
AST-level rule (no ``WA_DeleteOnClose`` literal anywhere in ``nanoscope/gui``); this file
pins the two SPECIFIC dialogs by construction and verifies, via ``shiboken6.isValid``,
that ``close()`` really does leave the C++ object alive rather than merely trusting the
absence of the attribute literal.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6
from PySide6 import QtCore

from nanoscope.gui.build_info_dialog import BuildInfoDialog
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.open_pore_preview import OpenPorePreviewDialog


def _preview_payload(**overrides):
    payload = dict(
        time_s=[0.0, 0.1, 0.2], current_pA=[-120.0, -119.0, -121.0],
        reference_pA=[-120.0, -120.0, -120.0], event_start_s=[], event_end_s=[],
        event_level_pA=[], event_count=0, polarity=-1, noise_pA=1.0, warnings=[],
        reference_details={}, title="lifetime test",
    )
    payload.update(overrides)
    return payload


def test_build_info_dialog_never_sets_delete_on_close(qtbot):
    dialog = BuildInfoDialog()
    qtbot.addWidget(dialog)
    assert dialog.testAttribute(QtCore.Qt.WA_DeleteOnClose) is False


def test_build_info_dialog_survives_its_own_close(qtbot):
    """The exact failure mode: Close (or teardown's close()-sweep) must hide, not delete.

    ``WA_DeleteOnClose`` makes ``close()`` call ``deleteLater()``, a DEFERRED delete --
    the C++ object is still technically alive immediately after ``close()`` returns, and
    only actually destroyed once the event loop next processes posted events. A check
    right after ``close()`` with nothing pumped would pass whether or not the attribute
    is set, so the event loop is driven forward here exactly as it would be on the next
    return to it in the real app."""
    dialog = BuildInfoDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    QtCore.QCoreApplication.processEvents()
    assert shiboken6.isValid(dialog), (
        "close() destroyed the C++ object under a live Python wrapper once the event "
        "loop ran -- WA_DeleteOnClose is back, or something equivalent was reintroduced"
    )
    assert dialog.isHidden()
    # The wrapper is genuinely usable afterward, not merely a non-crashing husk.
    dialog.show()
    assert not dialog.isHidden()


def test_main_window_reuses_one_build_info_dialog_across_help_about_presses(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._open_build_info()
    first = window._build_info_dialog
    assert first is not None
    first.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    QtCore.QCoreApplication.processEvents()
    window._open_build_info()
    assert window._build_info_dialog is first, (
        "Help > About must reuse the retained dialog, not build a new one every press "
        "(the previous fix path relied on WA_DeleteOnClose to avoid accumulating them)"
    )
    assert shiboken6.isValid(first)
    assert not first.isHidden()
    window._analysis_controller.abort()


def test_open_pore_preview_dialog_never_sets_delete_on_close(qtbot):
    dialog = OpenPorePreviewDialog(_preview_payload())
    qtbot.addWidget(dialog)
    assert dialog.testAttribute(QtCore.Qt.WA_DeleteOnClose) is False


def test_open_pore_preview_survives_close_and_a_second_preview_disposes_the_first(qtbot, tmp_path):
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    import numpy as np

    window = MainWindow()
    qtbot.addWidget(window)
    n = 2_000
    data = np.random.default_rng(7).normal(-120, 1, n).astype(np.float32)
    meta = RecordingMeta(path=tmp_path / "lifetime.dat", format_name="synthetic", n_samples=n,
                         sample_rate_hz=10_000.0, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(index=0, name="current", raw_units="pA",
                                               kind=ChannelKind.CURRENT),), t0_s=0.0)
    doc = TraceDocument(meta=meta, channels=(data,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    panel = window._analysis_panel
    target = (doc_id, 0)
    assert panel._reference_preview_dialog is None

    payload1 = _preview_payload(settings=panel._form_params(),
                                 calibration=tuple(window._session.calibration(*target).fingerprint()))
    panel._finish_reference_preview(payload1, target)
    first = panel._reference_preview_dialog
    assert first is not None and shiboken6.isValid(first)
    assert first.testAttribute(QtCore.Qt.WA_DeleteOnClose) is False

    payload2 = _preview_payload(settings=panel._form_params(),
                                 calibration=tuple(window._session.calibration(*target).fingerprint()))
    panel._finish_reference_preview(payload2, target)
    second = panel._reference_preview_dialog
    assert second is not first, "a fresh preview must not silently reuse stale plotted data"
    # The previous instance is explicitly disposed (close()d immediately; deleteLater()
    # scheduled) rather than left to accumulate for the life of the session -- process
    # deferred deletion and the wrapper must not resurrect as a crash.
    assert first.isHidden()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert not shiboken6.isValid(first), (
        "the previous preview dialog was never disposed -- each preview run would "
        "otherwise leak one hidden dialog for the life of the session"
    )
    assert shiboken6.isValid(second)
    window._analysis_controller.abort()
