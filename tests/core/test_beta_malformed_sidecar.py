"""A damaged optional sidecar must not prevent installing an intact recording."""
import json

import h5py
import numpy as np
import pytest

from nanoscope.core.calibration import sidecar_path
from nanoscope.core.session import Session
from nanoscope.io.base import open_document


@pytest.mark.parametrize("payload", [None, [], [1], "", "text", 1, True])
def test_nonobject_sidecar_is_ignored_and_disclosed_by_session(tmp_path, payload):
    path = tmp_path / "recording.h5"
    with h5py.File(path, "w") as h:
        h["time"] = np.arange(2000) / 50_000
        h["current"] = np.arange(2000, dtype=np.float32)
    sidecar = sidecar_path(path)
    original = json.dumps(payload)
    sidecar.write_text(original)
    session = Session()
    doc_id = session.add_document(open_document(path))
    assert session.has_document(doc_id)
    loaded = session.document(doc_id)
    assert any(sidecar.name in issue and "ignored" in issue for issue in loaded.meta.issues)
    np.testing.assert_array_equal(loaded.channels[0], np.arange(2000, dtype=np.float32))
    assert sidecar.read_text() == original
