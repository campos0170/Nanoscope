"""B4DA-01 / -04 / -05 / -08 — the conditioning caveats have to reach a WIDGET.

Every fix in this group is worth exactly what a user can read off the screen, so these
tests drive the real gestures: an ``AnalysisPanel`` on a real ``Session``, the Preprocess
tab's own check boxes, ``trigger_detection()`` (what Analysis ▸ Run detection calls), and
then a structure-blind sweep of every text-bearing widget -- the same sweep
``tests/gui/test_warnings_surface.py`` uses, so a caveat that lands in some other panel
still counts and one that lands nowhere still fails.

What must be visible after a run:

* a railed event's row carries ``saturated`` in the events table's ``flag`` cell, and
  the Stats block says the depth is a lower bound (B4DA-05);
* a run with SavGol enabled says what the smoother cost the depth (B4DA-08);
* a run with the notch enabled on an ordinary-length recording says NOTHING new -- the
  filter now settles inside its own pad, so there is no artefact to caveat (B4DA-01).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
N = 400_000            # 8 s @ 50 kHz -- longer than the notch's own ring-down
I0 = 1000.0
DEPTH = 300.0
RAIL = 800.0


def _visible_text(root: QtWidgets.QWidget) -> str:
    """Every string a user could read off ``root``. Structure-blind on purpose."""
    out: list[str] = []
    for w in root.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(root):
            continue
        for attr in ("text", "toPlainText", "toolTip"):
            fn = getattr(w, attr, None)
            if callable(fn):
                try:
                    val = fn()
                except (TypeError, RuntimeError):
                    continue
                if isinstance(val, str) and val:
                    out.append(val)
        if isinstance(w, QtWidgets.QTableWidget):
            for r in range(w.rowCount()):
                for c in range(w.columnCount()):
                    item = w.item(r, c)
                    if item is not None:
                        out.append(item.text())
    return "\n".join(out)


def _signal(*, rail: float | None, seed: int = 3) -> np.ndarray:
    """A 1000 pA open pore with 40 blockades of 2 ms, optionally clipped at ``rail``."""
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 5.0, N)
    for i in range(40):
        s = 5_000 + i * 9_500
        x[s : s + 100] -= DEPTH
    if rail is not None:
        x = np.clip(x, rail, None)
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, sig: np.ndarray, name: str) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(sig.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture(autouse=True)
def _private_settings(qapp):
    app = QtWidgets.QApplication.instance()
    org, name = app.organizationName(), app.applicationName()
    app.setOrganizationName("NanoScopeTest")
    app.setApplicationName("S4Conditioning")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        app.setOrganizationName(org)
        app.setApplicationName(name)


@pytest.fixture
def panel(qapp, qtbot, tmp_path):
    def build(sig: np.ndarray, name: str = "conditioning.dat"):
        session = Session()
        tv = TraceView(session)
        qtbot.addWidget(tv)
        controller = AnalysisController(session)
        p = AnalysisPanel(session, controller, tv)
        qtbot.addWidget(p)
        doc_id = session.add_document(_doc(tmp_path, sig, name))
        session.add_layer(doc_id, 0)
        p.set_selection(doc_id, 0)
        built.append(controller)
        return p
    built: list[AnalysisController] = []
    yield build
    for c in built:
        c.abort()


def _detect(p: AnalysisPanel, qtbot):
    with qtbot.waitSignal(p._controller.finished, timeout=180_000):
        p.trigger_detection()
    qtbot.wait(50)
    res = p._stored_detection()
    assert res is not None, "the fixture must produce a detection"
    return res


# ---------------------------------------------------------------------------
def test_a_railed_event_says_saturated_where_the_user_reads_the_depth(panel, qtbot):
    """B4DA-05 through the shipped surface: the flag cell AND the caveat block."""
    p = panel(_signal(rail=RAIL))
    res = _detect(p, qtbot)
    df = res.payload.df
    assert len(df) > 0
    assert df["flag"].astype(str).str.contains("saturated").all(), df["flag"].unique()

    text = _visible_text(p)
    assert "saturated" in text, "the flag never reached a widget"
    assert "flat rail" in text and "LOWER BOUNDS" in text, (
        "the depth is 33% low and nothing on screen says it is a floor")


def test_an_unrailed_recording_shows_no_new_caveat(panel, qtbot):
    """The control: the same events without the rail must not gain a word."""
    p = panel(_signal(rail=None), name="clean.dat")
    res = _detect(p, qtbot)
    assert not res.payload.df["flag"].astype(str).str.contains("saturated").any()
    text = _visible_text(p)
    assert "flat rail" not in text
    assert "smoothed the trace before detection" not in text
    assert "startup transient" not in text


def test_enabling_savgol_puts_its_cost_on_screen(panel, qtbot):
    """B4DA-08 through the real check box on the Preprocess tab."""
    p = panel(_signal(rail=None), name="savgol.dat")
    p._detection.savgol_enabled.setChecked(True)
    p._detection.savgol_window_s.setValue(0.0030)
    qtbot.wait(10)
    res = _detect(p, qtbot)
    assert res.payload.provenance["savgol_enabled"] is True

    text = _visible_text(p)
    assert "smoothed the trace before detection" in text, (
        "a smoother that eats depth left no trace on screen")
    assert "attenuated" in text
    assert "Savitzky-Golay (3 ms window" in text


def test_enabling_the_notch_on_a_full_length_recording_adds_no_events_and_no_caveat(
        panel, qtbot):
    """B4DA-01 through the real check box: the head transient is gone at the filter,
    so the run is quiet -- no phantom rows in the first second, no new sentence."""
    p_off = panel(_signal(rail=None), name="notch_off.dat")
    off = _detect(p_off, qtbot)

    p_on = panel(_signal(rail=None), name="notch_on.dat")
    p_on._detection.notch_enabled.setChecked(True)
    qtbot.wait(10)
    on = _detect(p_on, qtbot)

    assert on.payload.provenance["notch_enabled"] is True
    assert on.payload.provenance["notch_unsettled_s"] == 0.0
    assert len(on.payload.df) == len(off.payload.df), (
        "the notch changed the event count on a hum-free recording",
        len(on.payload.df), len(off.payload.df))
    early = on.payload.df[on.payload.df["t_start_s"] < 1.0]
    assert len(early) == len(off.payload.df[off.payload.df["t_start_s"] < 1.0])
    assert "startup transient" not in _visible_text(p_on)
