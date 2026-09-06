"""Native cached comparison canvases retain exact geometry and vector export."""
import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import rc_context, rcParams
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter

from nanoscope.gui import figure_bundle as B
from nanoscope.gui.matplotlib_figure_item import MatplotlibFigureItem
from nanoscope.gui.plot_export import write_live_plot_image
from nanoscope.gui.plot_export_options import resolve_export_options


def representative_figure():
    figure = Figure(figsize=(8, 4), dpi=100, facecolor="white")
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.10, right=.975, bottom=.24, top=.85, wspace=.4)
    for marker, color, y in (("o", "#1765a0", .2), ("s", "#d94b32", .5),
                              ("^", "#279864", .8)):
        axes[0].scatter([-.3, 0, .25, .5, .75, 1, 1.3], [y] * 7,
                        s=120, marker=marker, color=color, edgecolor="black", alpha=.7)
    axes[0].plot([-.5, 1.5], [.9, .1], "--", color="#644099", lw=1.3)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Dwell (µs)", ylabel="Depth (pA)",
                title="Recording −500 mV")
    axes[0].grid(alpha=.3)
    axes[1].boxplot([[1, 1.5, 2, 2.5, 3], [1.5, 2, 2.5, 3, 4]],
                    tick_labels=["−500 mV", "−600 mV"])
    axes[1].scatter([1, 1, 1, 2, 2], [1, 2, 3, 2, 4], color="#b84631", s=20)
    axes[1].set(ylabel="Amplitude (pA)", title="Individual events", ylim=(0, 5))
    axes[1].tick_params(axis="x", labelrotation=35)
    return figure


def attach(qtbot, item):
    view = QtWidgets.QGraphicsView()
    scene = QtWidgets.QGraphicsScene(view)
    scene.addItem(item)
    scene.setSceneRect(item.boundingRect())
    view.setScene(scene)
    view.setBackgroundBrush(QtGui.QColor("white"))
    view.resize(850, 500)
    qtbot.addWidget(view)
    view.show()
    return view, scene


def rgba(image):
    converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    return np.frombuffer(converted.bits(), dtype=np.uint8).reshape(
        converted.height(), converted.width(), 4).copy()


def test_native_drawing_matches_agg_labels_markers_clipping_and_panels(qtbot):
    figure = representative_figure()
    item = MatplotlibFigureItem(figure)
    view, scene = attach(qtbot, item)
    png, svg, width, height = B.capture_figure_render(item)
    actual = rgba(QtGui.QImage.fromData(png))[:, :, :3].astype(float) / 255
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    reference = np.asarray(canvas.buffer_rgba())[:, :, :3].astype(float) / 255
    assert (width, height) == (800, 400)
    assert actual.shape == reference.shape
    # The renderers have different pixel antialiasing/hinting. Compare geometry
    # after a one-pixel blur, then require substantially overlapping ink masks.
    error = np.abs(gaussian_filter(actual, (1, 1, 0)) -
                   gaussian_filter(reference, (1, 1, 0)))
    assert float(error.mean()) < .025
    ink = np.min(reference, axis=2) < .6
    near_actual = gaussian_filter((np.min(actual, axis=2) < .6).astype(float), 1.2) > .07
    assert float(near_actual[ink].mean()) > .92
    # Explicitly cropped outside-axis markers must not escape into margins.
    margin = actual[100:280, 48:62]
    assert float(np.max(np.max(margin, axis=2) - np.min(margin, axis=2))) < .02
    root = ET.fromstring(svg)
    assert any(node.tag.endswith("clipPath") for node in root.iter())
    assert not any(node.tag.endswith("image") for node in root.iter())
    assert "Dwell (µs)" in svg.decode()


def test_capture_bundle_live_and_frozen_export_never_revisit_figure(qtbot, monkeypatch, tmp_path):
    figure = representative_figure()
    item = MatplotlibFigureItem(figure)
    view, scene = attach(qtbot, item)
    view.scale(1.4, 1.4)
    original_transform = view.transform()
    arrays = [collection.get_offsets().copy() for collection in figure.axes[0].collections]
    axes_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in figure.axes]
    monkeypatch.setattr(figure, "draw", lambda *a, **kw: pytest.fail("export redrew source Figure"))
    monkeypatch.setattr(figure, "savefig", lambda *a, **kw: pytest.fail("export revisited source Figure"))
    table = pd.DataFrame({"depth_pA": [np.pi, np.nextafter(1., 2.)],
                          "event_id": pd.array([1, 2], dtype="Int64")})
    snapshot = B.capture_figure_snapshot(item, data=table, config={"kind": "comparison"},
                                         provenance={"warnings": ["gain unverified"]})
    assert snapshot.svg_bytes == item.native_figure_svg()
    assert "gain unverified" not in snapshot.svg_bytes.decode()
    assert_frame_equal(snapshot.dataframe(), table)
    bundle = B.write_figure_bundle(tmp_path / "comparison.zip", snapshot)
    with zipfile.ZipFile(bundle) as archive:
        metadata = json.loads(archive.read("manifest.json"))
        restored = B.FigureSnapshot.from_storage(
            metadata, {key: archive.read(path) for key, path in metadata["files"].items()})
    assert_frame_equal(restored.dataframe(), table)
    assert dict(restored.additional_resources)["rendering/figure.qpic"]
    for suffix in ("png", "svg", "pdf"):
        destination = tmp_path / f"live.{suffix}"
        write_live_plot_image(destination, item)
        assert destination.stat().st_size > 100
        if suffix == "svg":
            assert destination.read_bytes() == snapshot.svg_bytes
        elif suffix == "pdf":
            assert b"/Subtype /Image" not in destination.read_bytes()
        options = resolve_export_options((800, 400), extension=suffix, native=False,
                                          width=180, height=90, dpi=150)
        frozen = tmp_path / f"frozen.{suffix}"
        B.write_frozen_figure_image(frozen, restored, options)
        assert frozen.stat().st_size > 100
    assert view.transform() == original_transform
    assert axes_limits == [(ax.get_xlim(), ax.get_ylim()) for ax in figure.axes]
    for collection, original in zip(figure.axes[0].collections, arrays):
        np.testing.assert_array_equal(collection.get_offsets(), original)


def test_update_preserves_view_and_unsupported_update_preserves_prior_drawing(qtbot):
    item = MatplotlibFigureItem(representative_figure())
    view, scene = attach(qtbot, item)
    view.scale(1.8, 1.8)
    view.horizontalScrollBar().setValue(20)
    transform = view.transform()
    scroll = view.horizontalScrollBar().value()
    styled = representative_figure()
    styled.axes[0].set_title("Restyled comparison")
    item.update_figure(styled)
    assert view.transform() == transform
    assert view.horizontalScrollBar().value() == scroll
    previous = item.native_figure_svg()
    bad = Figure(figsize=(3, 2))
    bad.subplots().add_patch(Rectangle((0, 0), 1, 1, hatch="/"))
    with pytest.raises(ValueError, match="Hatched"):
        item.update_figure(bad)
    assert item.native_figure_svg() == previous
    assert item.boundingRect() == QtCore.QRectF(0, 0, 800, 400)


@pytest.mark.parametrize("size", [(0, 2), (1000, 1000)])
def test_unsafe_canvas_rejected(qapp, size):
    figure = Figure(figsize=size, dpi=100)
    with pytest.raises(ValueError, match="finite, safely sized"):
        MatplotlibFigureItem(figure)


def test_nonrectangular_clip_is_preserved_in_native_picture(qtbot):
    figure = Figure(figsize=(3, 3), dpi=100)
    axes = figure.add_axes((0, 0, 1, 1), xlim=(0, 1), ylim=(0, 1))
    axes.set_axis_off()
    patch = Rectangle((0, 0), 1, 1, color="red")
    patch.set_clip_path(Circle((.5, .5), .25, transform=axes.transData))
    axes.add_patch(patch)
    item = MatplotlibFigureItem(figure)
    view, scene = attach(qtbot, item)
    png, _, _, _ = B.capture_figure_render(item)
    image = QtGui.QImage.fromData(png)
    assert image.pixelColor(150, 150).red() > 250
    assert image.pixelColor(150, 150).green() < 10
    assert image.pixelColor(80, 80) == QtGui.QColor("white")


def test_native_svg_uses_frozen_full_canvas_layout_and_isolated_defaults(qtbot, monkeypatch):
    figure = Figure(figsize=(8, 4), dpi=100, layout="constrained")
    axes = figure.subplots(1, 2)
    axes[0].plot([1, 2], [2, 3])
    axes[0].set(xlabel="Dwell (µs)", ylabel="Depth (pA)")
    axes[1].plot([1, 2], [3, 4])
    engine, canvas = figure.get_layout_engine(), figure.canvas
    original_execute = engine.execute
    calls = []

    def execute(fig):
        calls.append(True)
        return original_execute(fig)

    monkeypatch.setattr(engine, "execute", execute)
    with rc_context({"svg.fonttype": "none", "savefig.bbox": "tight"}):
        item = MatplotlibFigureItem(figure)
        assert rcParams["svg.fonttype"] == "none"
        assert rcParams["savefig.bbox"] == "tight"
    assert len(calls) == 1
    assert figure.get_layout_engine() is engine
    assert figure.canvas is canvas
    root = ET.fromstring(item.native_figure_svg())
    assert list(map(float, root.attrib["viewBox"].split())) == [0, 0, 576, 288]
    assert not any(node.tag.endswith("}text") for node in root.iter())
    assert not any(node.tag.endswith("}image") for node in root.iter())
