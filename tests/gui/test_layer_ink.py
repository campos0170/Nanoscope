"""What colour the trace view draws a layer, and where that decision comes from.

D1 asked for ONE user-assignable colour per recording, shared by every surface. The
trace view's half of that is the concrete ``Layer.color`` value. New layers derive it
from the project's :class:`~nanoscope.core.layer.ColourRegistry`, and a user edit writes
both values, while project restore preserves the two independently persisted facts.

That distinction preserves the exact dark-pane ink saved by an older palette while the
Plots panel renders the same stable identity slot through its own light-background palette.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np

from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import LAYER_INKS, ColourRegistry, colour_key
from nanoscope.core.segment import persistent_key
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.trace_view import FILTER_PREVIEW_HEX, layer_ink

FS = 50_000.0
N = 4_000

_CHANNELS = (
    ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),
    ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
)


def _doc(tmp_path: Path, name: str) -> TraceDocument:
    rng = np.random.default_rng(0)
    data = tuple(rng.standard_normal(N).astype(np.float32) for _ in _CHANNELS)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=_CHANNELS,
    )
    return TraceDocument(meta=meta, channels=data)


def _session_with(tmp_path: Path, name: str = "rec.dat") -> tuple[Session, TraceDocument]:
    session = Session()
    doc = _doc(tmp_path, name)
    session.add_document(doc)
    for info in doc.meta.channels:
        session.add_layer(doc.doc_id, info.index)
    return session, doc


def test_a_real_session_starts_with_one_empty_colour_registry(qapp):
    session = Session()
    assert isinstance(session.colours, ColourRegistry)
    assert len(session.colours) == 0


def test_without_a_registry_the_layer_draws_its_own_colour(qapp, tmp_path):
    """The legacy/lightweight-session fallback remains byte-identical."""
    session, doc = _session_with(tmp_path)
    layer = session.layers()[0]
    layer.color = "#123456"
    del session.colours
    assert layer_ink(session, layer) == "#123456"


def test_the_concrete_layer_ink_wins_when_the_registry_differs(qapp, tmp_path):
    session, doc = _session_with(tmp_path)
    layer = session.layers()[0]
    layer.color = "#123456"

    reg = ColourRegistry()
    reg.set_ink(colour_key(persistent_key(doc), layer.channel), "#FF00FF")
    session.colours = reg

    assert layer_ink(session, layer) == "#123456"


def test_adding_a_layer_registers_its_persistent_identity(qapp, tmp_path):
    session = Session()
    doc = _doc(tmp_path, "rec.dat")
    session.add_document(doc)

    layer = session.add_layer(doc.doc_id, 0)
    key = colour_key(persistent_key(doc), 0)

    assert session.colours.keys() == (key,)
    assert layer.color == LAYER_INKS[0]
    assert layer_ink(session, layer) == LAYER_INKS[0]
    assert doc.doc_id not in key, "the registry key must survive a project reopen"


def test_setting_a_layer_colour_updates_the_shared_registry(qapp, tmp_path):
    session, doc = _session_with(tmp_path)
    layer = session.layers()[0]
    key = colour_key(persistent_key(doc), layer.channel)

    session.set_layer_color(layer.layer_id, "#FF00FF")

    assert layer.color == "#FF00FF"                 # concrete persisted fallback
    assert session.colours.is_user_set(key)
    assert session.colours.ink(key, LAYER_INKS) == "#FF00FF"
    assert layer_ink(session, layer) == "#FF00FF"   # live trace uses the same rule


def test_remove_and_readd_keeps_the_identity_slot_and_user_ink(qapp, tmp_path):
    session, doc = _session_with(tmp_path)
    first = session.layers()[0]
    key = colour_key(persistent_key(doc), first.channel)
    session.set_layer_color(first.layer_id, "#FF00FF")

    session.remove_layer(first.layer_id)
    again = session.add_layer(doc.doc_id, first.channel)

    assert session.colours.slot(key) == 0
    assert again.color == "#FF00FF"
    assert layer_ink(session, again) == "#FF00FF"


def test_adding_layers_stores_the_dark_pane_palette_not_the_plot_one(qapp, tmp_path):
    """The trace pane is ``#111316``; its concrete inks must remain legible there."""
    session, doc = _session_with(tmp_path)
    inks = [layer_ink(session, lay) for lay in session.layers()]
    assert inks == [LAYER_INKS[0], LAYER_INKS[1]]


def test_an_unrelated_registry_key_cannot_repaint_the_concrete_layer(qapp, tmp_path):
    session, doc = _session_with(tmp_path)
    layer = session.layers()[0]
    layer.color = "#123456"

    reg = ColourRegistry()
    reg.set_ink(colour_key(layer.doc_id, layer.channel), "#00FF00")   # the WRONG key
    session.colours = reg

    assert layer_ink(session, layer) == "#123456"


def test_a_layer_whose_document_has_gone_still_gets_an_ink(qapp, tmp_path):
    """A recording removed between the layer list and the draw is not a reason to draw
    nothing. Falls back to the ink the layer already carries."""
    session, doc = _session_with(tmp_path)
    layer = session.layers()[0]
    layer.color = "#123456"
    session.colours = ColourRegistry()
    layer.doc_id = "no-such-document"
    assert layer_ink(session, layer) == "#123456"


def test_the_filter_preview_is_not_one_of_the_layer_inks(qapp):
    """It is a ghost drawn ON a layer, and reading it as an overlaid recording is the
    one misreading that matters. Its distance from the palette is measured in
    ``tests/core/test_colour_identity.py``; this only pins that it is not literally one
    of them, which a future palette edit could otherwise make true by accident."""
    assert FILTER_PREVIEW_HEX.lower() not in {h.lower() for h in LAYER_INKS}
