"""Curation artwork stays clean while details preserve warnings and exact science."""
import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.event_inspector import EventInspectorWindow


@pytest.fixture
def curation(qapp, qtbot, tmp_path):
    fs = 50_000.0
    data = np.full(20_000, 100.0) + np.random.default_rng(18).normal(0, .5, 20_000)
    data[4_000:4_500] -= 230.0
    data[12_000:12_300] -= 30.0
    data = data.astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "curation-artwork.dat", format_name="synthetic",
                         n_samples=len(data), sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(data,)))
    frame, provenance = detect_events(data, fs, baseline=100.0, polarity=-1,
                                      min_dwell_s=0.00002)
    assert len(frame) == 2
    frame.loc[frame.index[0], "filter_limited"] = True
    inspector = EventInspectorWindow(session)
    gallery = EventGalleryWindow(session)
    for widget in (inspector, gallery):
        qtbot.addWidget(widget)
        widget.resize(1100, 850)
        widget.set_events(doc_id, 0, frame, provenance=provenance)
        widget.show()
    qtbot.wait(50)
    return session, inspector, gallery, frame


def test_inspector_artwork_keeps_numbers_amplitude_and_geometry_but_not_warnings(curation, qtbot):
    session, inspector, _gallery, frame = curation
    # A deliberate optional measurement establishes a valid amplitude field for
    # this test. Everything below is presentation only and must leave it exact.
    assert inspector._amplitude.apply_current(method="mean")
    before = frame.copy(deep=True)
    raw = session.calibrated(inspector._doc_id, inspector._channel).data.copy()
    title = inspector._plot.titleLabel.text
    assert "Event 0" in title and "dwell " in title
    assert "median depth " in title and "pA" in title
    assert "dI/I0 " in title and "amplitude " in title and "(mean)" in title
    assert "⚠" not in title and "filter-limited" not in title
    assert "⚠>1" in inspector._readout.text()
    assert "filter-limited:  yes" in inspector._readout.text()
    qtbot.wait(30)
    geometry = (inspector.size(), inspector._plot.sceneBoundingRect(),
                inspector._readout.geometry())
    ranges = np.asarray(inspector._plot.viewRange()).copy()
    inspector._plot.setTitle(inspector._event_title(frame.iloc[0]))
    qtbot.wait(20)
    assert (inspector.size(), inspector._plot.sceneBoundingRect(),
            inspector._readout.geometry()) == geometry

    note = "Calibration changed: depth, fractional depth and SNR are stale."
    inspector.set_stale_columns(REMEASURED_COLUMNS, note)
    qtbot.wait(30)
    assert inspector._plot.titleLabel.text == title
    assert inspector.stale_note() == note
    assert "⚠>1" in inspector._readout.text()
    assert "separate from depth/dI/I0/SNR" in inspector._readout.text()
    assert inspector._readout.text().split("amplitude:")[-1].rstrip().endswith("⚠")
    # The amplitude-editor gate has an existing note reflow on a staleness
    # transition (also reproduced with the original title implementation).
    # Check that title rendering itself moves neither this state nor the clean one.
    geometry = (inspector.size(), inspector._plot.sceneBoundingRect(),
                inspector._readout.geometry())
    inspector._plot.setTitle(inspector._event_title(frame.iloc[0]))
    qtbot.wait(20)
    assert (inspector.size(), inspector._plot.sceneBoundingRect(),
            inspector._readout.geometry()) == geometry
    np.testing.assert_array_equal(inspector._plot.viewRange(), ranges)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    np.testing.assert_array_equal(session.calibrated(inspector._doc_id, inspector._channel).data, raw)

    inspector.set_stale_columns(())
    qtbot.wait(20)
    assert inspector._plot.titleLabel.text == title
    assert inspector.stale_note() == ""
    # This independent condition survives clearing staleness.
    assert "⚠>1" in inspector._readout.text()


def test_gallery_moves_fraction_warning_to_footer_and_preserves_all_captions(curation, qtbot):
    _session, _inspector, gallery, frame = curation
    before = frame.copy(deep=True)
    gallery.set_sort("dI/I0 (deepest first)")
    qtbot.wait(20)
    captions = [gallery.caption_for(row) for _, row in frame.iterrows()]
    expected = [f"{int(row.event_id)} · {row.frac_depth:.3g}" for _, row in frame.iterrows()]
    assert captions == expected and all("⚠" not in value for value in captions)
    assert "1 event(s) on this page have dI/I0 > 1" in gallery.status.text()
    assert "not a blockade fraction" in gallery.status.text()
    geometry = (gallery.size(), gallery.view.geometry(), gallery.status.height())
    note = "The selected result uses an older baseline."
    gallery.set_stale_columns(REMEASURED_COLUMNS, note)
    qtbot.wait(20)
    assert [gallery.caption_for(row) for _, row in frame.iterrows()] == captions
    assert note in gallery.status.text()
    assert "dI/I0 > 1" in gallery.status.text()
    assert (gallery.size(), gallery.view.geometry(), gallery.status.height()) == geometry
    pd.testing.assert_frame_equal(frame, before, check_exact=True)

    gallery.set_sort("Depth (deepest first)")
    assert all(gallery.caption_for(row) == f"{int(row.event_id)} · {row.depth_pA:.3g}pA"
               for _, row in frame.iterrows())
    gallery.set_stale_columns(())
    assert note not in gallery.status.text()
    assert "not a blockade fraction" in gallery.status.text()

    # A different subset must not inherit warnings about tiles it does not show.
    clean = frame.iloc[[1]].copy(deep=True)
    gallery.set_events(gallery._doc_id, gallery._channel, clean)
    assert "dI/I0 > 1" not in gallery.status.text()
    gallery.set_events(gallery._doc_id, gallery._channel, frame.iloc[0:0].copy())
    assert "no events" in gallery.status.text().lower()
    assert "dI/I0 > 1" not in gallery.status.text()
