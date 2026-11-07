"""Narrow trace labels retain clock identity and the exact origin disclosure."""
import nanoscope  # noqa: F401
import numpy as np
import pytest

from nanoscope.core import clock
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.trace_view import TraceView, _trace_clock_text


@pytest.mark.parametrize('rebased,alignment', [(False, 0.), (False, .025), (True, 0.), (True, .025)])
def test_narrow_trace_clock_is_readable_and_full_origin_survives_without_rescaling(qtbot, tmp_path, rebased, alignment):
    raw = np.linspace(100., 120., 2000, dtype=np.float32)
    info = ChannelInfo(index=0, name='current', raw_units='pA', kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / 'synthetic.dat', format_name='synthetic',
        n_samples=raw.size, sample_rate_hz=1000., rate_provenance=Provenance.FILE_META,
        channels=(info,), t0_s=61.3034)
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(raw,)))
    if rebased:
        doc_id = session.create_segment(doc_id, 200, 1200, name='Rebased synthetic cut')
    layer = session.add_layer(doc_id, 0)
    session.set_layer_time_offset(layer.layer_id, alignment)
    view = TraceView(session); qtbot.addWidget(view)
    view.resize(360, 300); view.show(); qtbot.wait(20)
    pane = view.panes()[-1]
    axis = pane.plot.getAxis('bottom')
    offset = view.clock_offset_for(doc_id, 0)
    full = clock.axis_label('s', clock.INSTRUMENT, offset, display_unit='ms')
    assert view.x_axis_label() == axis.toolTip() == axis.label.toolTip() == full
    assert axis.labelText.startswith('ms · ')
    if rebased:
        assert 'cut clock (rebased)' in axis.labelText
        assert '61503.4 ms' in full and 'instrument clock' in full
        if alignment:
            assert 'aligned' in axis.labelText and 'shifted by 25.0 ms' in full
    else:
        assert axis.labelText == 'ms · instrument clock'
        assert clock.format_t0(offset, display_unit='ms') + ' ms' in full
    assert axis.label.boundingRect().width() <= axis.geometry().width()
    painted = view.mapFromScene(axis.label.sceneBoundingRect()).boundingRect()
    assert painted.left() >= 0 and painted.right() <= view.viewport().width()
    curve = next(iter(pane._curves.values()))
    expected = session.x_axis(doc_id, alignment).values
    np.testing.assert_array_equal(curve.xData, expected)
    assert axis.scale == 1000. and not axis.autoSIPrefix
    before = curve.xData.copy()
    if rebased and not alignment:
        # The existing ML strip temporarily owns the same bottom axis statement.
        assert view.set_confidence_track(doc_id, 0, None, note='No saved probabilities')
        carrier = view.confidence_plot().getAxis('bottom')
        assert carrier.labelText == 'ms · cut clock (rebased)'
        assert carrier.toolTip() == full == view.x_axis_label()
        view.clear_confidence_track()
        assert axis.labelText == 'ms · cut clock (rebased)'
        assert axis.toolTip() == full == view.x_axis_label()
    np.testing.assert_array_equal(curve.xData, before)


def test_unknown_and_sample_axes_never_gain_an_invented_clock():
    assert _trace_clock_text('sample', None) == ('sample', 'sample')
    assert _trace_clock_text('s', None) == ('ms', 'ms')
    short, full = _trace_clock_text('s', clock.ClockOffset(0.))
    assert short == 'ms · instrument clock'
    assert full == clock.axis_label('s', clock.INSTRUMENT, clock.ClockOffset(0.), display_unit='ms')
