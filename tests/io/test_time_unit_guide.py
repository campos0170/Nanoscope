"""The user guide describes the time-unit rule the app has — measured first, then read.

Until this change docs/USER_GUIDE.md §2 said a CSV time column in microseconds OR nanoseconds
was refused and "the unit is declared nowhere in the file", which left a reader believing a
non-seconds column could not load silently; a millisecond column did exactly that (see
tests/io/test_time_unit_declarations.py). Like tests/io/test_s3_io_guide.py, this checks the
app's own behaviour and then the document against it, rather than grepping prose.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from nanoscope.io.base import SUSPECT_UNDECLARED_RATE_HZ, probe_path

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "USER_GUIDE.md"


def _guide() -> str:
    return " ".join(_GUIDE.read_text(encoding="utf-8").split())


def _csv(path: Path, header: str, step: float, n: int = 3_000) -> Path:
    path.write_text("\n".join([header] + [f"{i * step!r},0.0" for i in range(n)]) + "\n",
                    encoding="utf-8")
    return path


def test_the_guide_describes_the_three_tiers_the_reader_has(tmp_path):
    asked = probe_path(_csv(tmp_path / "bare_ms.csv", "time,current", 0.01)).meta
    declared = probe_path(_csv(tmp_path / "time_ms.csv", "time_ms,current", 0.01)).meta
    refused = probe_path(_csv(tmp_path / "bare_us.csv", "time,current", 100.0)).meta
    guide = _guide()

    # tier 3 — asked about, rate kept: the row exists, names the bound and the remedies.
    assert asked.sample_rate_hz == pytest.approx(100.0)
    assert "kept as read" in " ".join(asked.issues)
    assert "**A time base that may not be in seconds**" in guide
    assert SUSPECT_UNDECLARED_RATE_HZ == 1000.0 and "below **1 kHz**" in guide
    assert "**kept as read**" in guide
    assert "Inspector ▸ File Info ▸ Sample rate / time unit…" in guide
    # tier 1 — a declared header is read as declared, and the Text row says which headers.
    assert declared.sample_rate_hz == pytest.approx(100_000.0) and declared.issues == ()
    assert "`time_ms`" in guide and "read in that unit" in guide
    # tier 2 — the refusal row still stands, and still says the unit is not guessed.
    assert refused.sample_rate_hz is None
    assert "**A time base that cannot be in seconds**" in guide and "**not guessed**" in guide
    # the sentence that was false is gone.
    assert "a **Note** says so when it happens" not in guide


def test_the_guide_quotes_the_surfaces_the_declaration_prints_on(qapp, tmp_path):
    """Measured first: what File Info prints after a declaration, the verdict the Results
    panel leads with, and that a rate-less recording gains a seconds axis from one."""
    import numpy as np
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.sample_rate import SampleRateDeclaration
    from nanoscope.core.session import Session
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    from nanoscope.gui.analysis_panel import _STALE_SAMPLE_RATE, _STALE_SUMMARIES
    from nanoscope.gui.file_info_panel import FileInfoPanel

    info = (ChannelInfo(0, "current", "", ChannelKind.CURRENT),)
    data = (np.zeros(2_000, dtype=np.float32),)
    session = Session()
    key = session.add_document(TraceDocument(
        RecordingMeta(tmp_path / "ms.csv", "Text trace", 2_000, 100.0, Provenance.MEASURED, info),
        data))
    session.set_sample_rate_declaration(key, SampleRateDeclaration("time_unit", time_unit="ms"))
    panel = FileInfoPanel(session)
    panel.set_document(key)
    printed = " ".join(panel._fields["Sample rate"].text().split())
    legacy = session.add_document(TraceDocument(
        RecordingMeta(tmp_path / "legacy.tdms", "TDMS (legacy)", 2_000, None, Provenance.UNKNOWN,
                      info), data))
    assert session.x_axis(legacy).unit == "sample"
    session.set_sample_rate_declaration(
        legacy, SampleRateDeclaration("sample_rate", sample_rate_hz=10_000.0))
    assert session.x_axis(legacy).unit == "s"

    guide = _guide()
    assert "**Declaring the time base yourself" in guide
    assert f"`{printed}`" in guide, printed
    assert f"*{_STALE_SUMMARIES[_STALE_SAMPLE_RATE]}*" in guide
    assert "the rate and the start time t₀ move together" in guide
    assert "the status bar prints the declared rate and its provenance" in guide
    assert "Segments cut from the recording follow its declaration" in guide
    assert "until you declare the rate under **Inspector ▸ File Info ▸ Sample rate / time unit…**" in guide
