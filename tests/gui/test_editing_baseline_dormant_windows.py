"""Back to default is judged by what it moved: the table's half of design §6's "Baseline ▸
Back to default" row.

The panel's half is in ``test_editing_baseline_form.py``: Back to default is on screen in
every mode and moves only the estimator's two windows, never the reference. In Automatic
those windows measure every depth, and resetting them makes the table on screen stale --
that is the banner doing its job. Under a user-provided reference they are DORMANT,
"retained, not applied" (``BaselineConfig``), so the same press changes no number in the
table.

``BaselineConfig.fingerprint`` hashes the windows even while a reference is set, and
``AnalysisPanel._baseline_staleness`` compares fingerprints, so the press used to raise "The
open-pore estimator changed since this detection ran" with the same reference named as the
old estimator and the new one, over a "Re-measure depths" that changed nothing (item B
review, 2026-10-30). The fingerprint must keep the windows -- every stored reference table
would read stale without them -- so the cure is the judge's:
``AnalysisPanel._live_baseline_stamp`` judges a table measured under the reference the form
still holds with the table's own dormant windows, and so judges only the reference and the
rate.

The rig's own steps fail with ``pytest.fail``, so a broken rig reports itself rather than
reading as the judge's verdict.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nanoscope.analysis import baseline as B  # noqa: E402
from nanoscope.analysis.context import SimpleAnalysisContext  # noqa: E402
from nanoscope.analysis.modules import DetectSchmitt  # noqa: E402
from nanoscope.core.document import TraceDocument  # noqa: E402
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta  # noqa: E402
from nanoscope.gui.main_window import MainWindow  # noqa: E402

FS = 10_000.0
N = 20_000


def document(tmp_path):
    data = np.random.default_rng(45).normal(-120, 1, N).astype(np.float32)
    data[5_000:7_000] -= 30
    meta = RecordingMeta(path=tmp_path / "reference.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(index=0, name="current", raw_units="pA",
                                               kind=ChannelKind.CURRENT),), t0_s=0.0)
    return TraceDocument(meta=meta, channels=(data,))


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    doc = document(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    yield window, window._analysis_panel, (doc_id, 0)
    window._analysis_controller.abort()


@pytest.mark.parametrize("current_pA, stale_after", [
    pytest.param(None, True, id="automatic"),
    pytest.param(-120.0, False, id="constant"),
])
def test_back_to_default_stales_the_table_only_when_the_windows_measured_it(
        rig, current_pA, stale_after):
    window, panel, (doc_id, channel) = rig
    form = panel.baseline_panel
    form.rough_s.setValue(0.4)
    form.base_s.setValue(0.08)
    if current_pA is not None:
        editor = form.reference_editor
        editor.mode.setCurrentIndex(editor.mode.findData("constant"))
        editor.current.setText(f"{current_pA:g}")
    session = window._session
    result = DetectSchmitt().run(SimpleAnalysisContext(
        session.calibrated(doc_id, channel), session.document(doc_id).meta,
        panel._form_params()))
    session.add_result(result)
    panel._check_staleness(result)
    if not panel._stale.isHidden():
        pytest.fail("rig: the table read stale before the press: "
                    f"{panel._baseline_staleness(result)}")

    form.reset_button.click()
    if (form.rough_s.value(), form.base_s.value()) != (
            B.DEFAULT_BASELINE.rough_s, B.DEFAULT_BASELINE.base_s):
        pytest.fail("rig: Back to default did not move the windows")
    if panel._reference_redetection_needed(result):
        pytest.fail("rig: the press moved the reference itself")
    assert panel._stale.isHidden() is not stale_after, panel._baseline_staleness(result)
