"""EXP2-01 — A FIGURE THAT FAILS INSIDE ``fig.savefig`` STAYS PER-FIGURE.

``save_figure`` stages a member (``group.stage``) BEFORE ``fig.savefig`` writes it, so
a figure that BUILDS fine but raises at draw time — a genuine matplotlib failure mode,
e.g. invalid mathtext in a title carried from the data file — left a staged-but-
never-written temp in the SHARED StagedSet. ``commit()`` then refused over the orphan
(``FileNotFoundError: … were never written``), the three healthy figure families were
thrown away, and the user got the critical full-disk dialog ("could not be put in
place") about a perfectly writable folder — chasing a disk problem that does not exist
instead of the one bad label. The code's own promise is "We never let one figure sink
the whole export", and tests/gui/test_beta_export_figures_unwritable.py pins per-figure
best-effort for BUILD failures; the save-time leg was the hole.

THE FIX THESE TESTS PIN. ``save_figure`` withdraws its OWN family from a shared group
on any failure (``StagedSet.unstage`` — whole family, so a PDF can never commit beside
a previous export's PNG), the healthy figures commit, and the failed figure is
disclosed BY NAME with its real error. The environmental-OSError path (readonly
folder) keeps its beta wording — pinned in test_beta_export_figures_unwritable.py.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import io
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 40_000

#: A title matplotlib parses lazily: the figure BUILDS, and ``savefig`` raises at draw.
_POISON_TITLE = r"current $\badmathcommand{x}$"


def _poisoned_figure():
    """A figure that builds fine and raises inside ``fig.savefig`` — the EXP2-01 shape."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    import pandas as pd

    fig = Figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title(_POISON_TITLE)
    return fig, pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _poison_still_poisons() -> bool:
    """Self-check, exactly like the verifier's repro: if a future matplotlib learns to
    render the bad mathtext, these tests would be driving nothing."""
    fig, _df = _poisoned_figure()
    try:
        fig.savefig(io.BytesIO(), format="pdf")
    except Exception:
        return True
    return False


def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path: Path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def detected(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    doc = _synth_doc(tmp_path)
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    w._analysis_panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=60_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert w._analysis_panel.current_detection() is not None
    try:
        yield w
    finally:
        w._analysis_controller.abort()


@pytest.fixture
def dialogs(monkeypatch):
    """Capture every warning/critical instead of showing a modal offscreen."""
    shown: list[tuple[str, str, str]] = []
    for kind in ("warning", "critical"):
        monkeypatch.setattr(
            QtWidgets.QMessageBox, kind,
            staticmethod(lambda _p, title, text, *a, _k=kind, **kw:
                         shown.append((_k, title, text))))
    return shown


def _export_to(window, base: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(base), "PDF (*.pdf)")))
    window._export_figures()


@pytest.fixture
def poisoned_scatter(monkeypatch):
    """The scatter BUILDER succeeds; the figure it returns fails at savefig."""
    if not _poison_still_poisons():
        pytest.skip("this matplotlib renders the poison mathtext; the shape is gone")
    from nanoscope.plot import figures
    monkeypatch.setattr(figures, "scatter_dwell_depth",
                        lambda *a, **k: _poisoned_figure())


def test_a_render_failure_does_not_sink_the_healthy_figures(
        detected, monkeypatch, tmp_path, dialogs, poisoned_scatter):
    out = tmp_path / "figs"
    out.mkdir()
    _export_to(detected, out / "figures.pdf", monkeypatch)

    # The healthy families shipped, complete...
    for name in ("dwell", "depth"):
        for ext in ("pdf", "png", "csv"):
            assert (out / f"figures_{name}.{ext}").exists(), f"figures_{name}.{ext}"
    # ...the poisoned family shipped nothing (no PDF beside a missing PNG either)...
    assert not list(out.glob("figures_scatter.*")), list(out.iterdir())
    # ...and no dialog blamed the environment for a rendering error.
    assert not any(k == "critical" for k, _t, _x in dialogs), dialogs
    assert not any("could not be put in place" in x for _k, _t, x in dialogs), dialogs
    # The per-figure disclosure (status line — the leftover dialog needs a stale file,
    # tested below) replaced the "Wrote…" line and names the failed figure.
    assert "scatter" in detected.statusBar().currentMessage(), \
        detected.statusBar().currentMessage()


def test_the_failed_figure_is_named_with_its_real_error(
        detected, monkeypatch, tmp_path, dialogs, poisoned_scatter):
    out = tmp_path / "figs2"
    out.mkdir()
    _export_to(detected, out / "figures.pdf", monkeypatch)

    # Nothing of that name pre-existed, so the disclosure is the status line — and it
    # names the figure AND the error class, not a generic "could not be built".
    seen = detected.statusBar().currentMessage() + " ".join(x for _k, _t, x in dialogs)
    assert "scatter" in seen, seen
    assert "ValueError" in seen, (
        "the real render error is not named anywhere; the user is left to guess:\n"
        + seen)


def test_a_render_failure_with_a_previous_export_discloses_the_leftovers(
        detected, monkeypatch, tmp_path, dialogs, poisoned_scatter, qtbot):
    """The other half of the per-figure contract: a stale ``figures_scatter.*`` from an
    earlier export is named in the leftover dialog (the ``_report_unreplaced_figures``
    door the orphaned staging used to skip entirely via its early return)."""
    out = tmp_path / "figs3"
    out.mkdir()
    for ext in ("pdf", "png", "csv"):
        (out / f"figures_scatter.{ext}").write_bytes(b"previous export")

    # Consent to the overwrite (module-level exactly so a test can, see export_paths);
    # the exec patch below would otherwise swallow the consent box as a decline.
    from nanoscope.gui import export_paths
    monkeypatch.setattr(export_paths, "confirm_overwrite", lambda *a, **k: True)
    boxes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec",
        lambda self, *a, **k: boxes.append((self.text(), self.informativeText())))
    _export_to(detected, out / "figures.pdf", monkeypatch)

    assert (out / "figures_dwell.pdf").exists()
    assert (out / "figures_scatter.pdf").read_bytes() == b"previous export", \
        "the poisoned family overwrote (or half-overwrote) the previous export"
    assert boxes, "the stale leftovers were disclosed by nothing"
    text, informative = boxes[0]
    assert "earlier export" in text
    assert "figures_scatter.pdf" in informative
    assert "ValueError" in informative, (
        "the leftover dialog does not name the real error:\n" + informative)
