"""Publication previews/bundles are frozen pictures, never scientific re-runs."""
import io
import json
import math
from dataclasses import replace
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from PySide6 import QtCore, QtGui
import pyqtgraph as pg

from nanoscope.gui import figure_bundle as F
from nanoscope.gui.plot_export import choose_live_plot_export, write_live_plot_image
from nanoscope.gui.plot_export_options import ExportOptionsDialog, resolve_export_options


@pytest.fixture
def plots(qtbot):
    main = pg.PlotWidget(background="w")
    residual = pg.PlotWidget(background="w")
    for widget, size in ((main, (640, 400)), (residual, (640, 180))):
        qtbot.addWidget(widget)
        widget.resize(*size)
        widget.show()
    x = np.linspace(0.0, 4.0, 81)
    y = np.sin(x)
    curve = main.plot(x, y, pen=pg.mkPen("#124a90", width=2))
    resid_curve = residual.plot(x, 0.1 * np.cos(x), pen=pg.mkPen("#b92429", width=2))
    main.setLabel("left", "Signal", units="pA")
    residual.setLabel("left", "Residual", units="pA")
    main.setXRange(0.5, 3.5, padding=0)
    main.setYRange(-0.8, 1.1, padding=0)
    residual.setXRange(0.5, 3.5, padding=0)
    residual.setYRange(-0.15, 0.15, padding=0)
    qtbot.wait(20)
    return main, residual, curve, resid_curve, x, y


def test_preview_and_capture_preserve_arrays_item_identity_zoom_and_fit(plots, monkeypatch):
    main, residual, curve, resid_curve, x, y = plots
    ranges = (main.viewRange(), residual.viewRange())
    arrays = (curve.xData.copy(), curve.yData.copy(), resid_curve.yData.copy())
    items = (tuple(main.getPlotItem().items), tuple(residual.getPlotItem().items))
    for item in (curve, resid_curve):
        monkeypatch.setattr(item, "setData", lambda *a, **k: pytest.fail("capture recomputed plot data"))
    monkeypatch.setattr(main, "plot", lambda *a, **k: pytest.fail("capture recreated a fit or curve"))
    preview = F.capture_plot_preview(main.getPlotItem(), residual_item=residual.getPlotItem(),
                                     include_residual=True)
    snapshot = F.capture_figure_snapshot(
        main.getPlotItem(), data=pd.DataFrame({"x": x, "y": y}),
        config={"fit": {"model": "existing", "parameters": [1.0]}},
        provenance={"analysis": "cached"}, residual_item=residual.getPlotItem(),
        include_residual=True)
    assert not preview.isNull()
    assert snapshot.includes_residual
    assert ranges == (main.viewRange(), residual.viewRange())
    assert items == (tuple(main.getPlotItem().items), tuple(residual.getPlotItem().items))
    np.testing.assert_array_equal(curve.xData, arrays[0])
    np.testing.assert_array_equal(curve.yData, arrays[1])
    np.testing.assert_array_equal(resid_curve.yData, arrays[2])


def test_bundle_csv_metadata_and_lossless_table_roundtrip(plots, tmp_path):
    main, *_ = plots
    table = pd.DataFrame({"value": [math.pi, np.nextafter(1.0, 2.0)],
                          "count": pd.array([1, pd.NA], dtype="Int64"),
                          "label": pd.array(["first", pd.NA], dtype="string")})
    table.index = pd.Index([4, 9], name="source_row")
    config = {"metric": "amplitude_pA", "fit": {"parameters": [1.25, 2.5]}}
    provenance = {"source_ids": ["synthetic"], "fs": np.float64(50_000.0)}
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=table,
                                         config=config, provenance=provenance)
    expected = table.copy(deep=True)
    table.iloc[0, 0] = -999
    config["metric"] = "changed"
    path = F.write_figure_bundle(tmp_path / "figure.zip", snapshot)
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"figure.png", "figure.svg", "data.csv", "data.parquet",
                                          "config.json", "provenance.json", "manifest.json",
                                          "rendering/figure.qpic"}
        csv = pd.read_csv(io.BytesIO(archive.read("data.csv")), float_precision="round_trip")
        np.testing.assert_array_equal(csv["value"], expected["value"])
        assert_frame_equal(pd.read_parquet(io.BytesIO(archive.read("data.parquet"))), expected)
        assert json.loads(archive.read("config.json"))["metric"] == "amplitude_pA"
        assert json.loads(archive.read("provenance.json"))["fs"] == 50_000.0
        assert not json.loads(archive.read("manifest.json"))["includes_residual"]
    assert_frame_equal(snapshot.dataframe(), expected)


def test_snapshot_storage_is_json_safe_detached_and_checksummed(plots):
    main, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame({"x": [1.0]}))
    metadata, blobs = snapshot.to_storage()
    json.dumps(metadata, allow_nan=False)
    back = F.FigureSnapshot.from_storage(metadata, blobs)
    assert back == snapshot
    corrupted = {**blobs, "png": blobs["png"] + b"changed"}
    with pytest.raises(ValueError, match="checksum"):
        F.FigureSnapshot.from_storage(metadata, corrupted)
    with pytest.raises(ValueError, match="dimensions"):
        F.FigureSnapshot.from_storage({**metadata, "width": math.nan}, blobs)


def test_additional_analysis_resources_survive_snapshot_and_bundle(plots, tmp_path):
    main, *_ = plots
    resources = {"analysis_tables/source.parquet": b"exact table payload",
                 "analysis_blobs/model.bin": b"exact cached fit payload"}
    snapshot = F.capture_figure_snapshot(
        main.getPlotItem(), data=pd.DataFrame({"x": [1.0]}), additional_resources=resources,
        config={"table_resource": "analysis_tables/source.parquet"})
    metadata, blobs = snapshot.to_storage()
    restored = F.FigureSnapshot.from_storage(metadata, blobs)
    assert resources.items() <= dict(restored.additional_resources).items()
    path = F.write_figure_bundle(tmp_path / "complete.zip", restored)
    with zipfile.ZipFile(path) as archive:
        for name, value in resources.items():
            assert archive.read(name) == value
        manifest = json.loads(archive.read("manifest.json"))
        assert set(resources) <= set(manifest["files"].values())


@pytest.mark.parametrize("name", ["../secret", "/absolute", "analysis/../../secret",
                                  "analysis\\escape", "figure.png", "manifest.json", "a//b",
                                  "Figure.PNG", "figure.svg/child"])
def test_additional_resource_names_cannot_escape_or_shadow_bundle(plots, name, monkeypatch):
    main, *_ = plots
    monkeypatch.setattr(F, "capture_figure_render", lambda *a, **k: pytest.fail("validate before render"))
    with pytest.raises(ValueError, match="safe"):
        F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame(),
                                  additional_resources={name: b"bad"})


def test_nonfinite_metadata_refused_before_capturing_live_items(plots, monkeypatch):
    main, *_ = plots
    monkeypatch.setattr(F, "capture_figure_render", lambda *a, **k: pytest.fail("should validate first"))
    with pytest.raises(ValueError):
        F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame({"x": [1.0]}),
                                  config={"bad": math.nan})


def test_bundle_failure_preserves_previous_file(plots, tmp_path, monkeypatch):
    main, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame({"x": [1.0]}))
    path = tmp_path / "figure.zip"
    path.write_bytes(b"old bundle")
    monkeypatch.setattr(zipfile.ZipFile, "writestr",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        F.write_figure_bundle(path, snapshot)
    assert path.read_bytes() == b"old bundle"


def test_residual_composition_is_opt_in_and_svg_ids_are_unique(plots, tmp_path):
    main, residual, *_ = plots
    data = pd.DataFrame({"x": [1.0]})
    native = F.capture_figure_snapshot(main.getPlotItem(), data=data,
                                       residual_item=residual.getPlotItem())
    combined = F.capture_figure_snapshot(main.getPlotItem(), data=data,
                                         residual_item=residual.getPlotItem(), include_residual=True)
    assert not native.includes_residual
    assert combined.height > native.height
    root = ET.fromstring(combined.svg_bytes)
    assert len(root.findall("{http://www.w3.org/2000/svg}g")) == 2
    ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
    assert len(ids) == len(set(ids))
    before = (main.viewRange(), residual.viewRange())
    for extension in ("png", "svg", "pdf"):
        path = tmp_path / f"combined.{extension}"
        write_live_plot_image(path, main.getPlotItem(), residual_item=residual.getPlotItem(),
                              include_residual=True, width=480, height=400)
        assert path.stat().st_size > 100
        if extension == "png":
            image = QtGui.QImage(str(path))
            assert (image.width(), image.height()) == (480, 400)
    assert before == (main.viewRange(), residual.viewRange())


def test_explicit_path_native_defaults_ignore_unselected_residual(plots, tmp_path):
    main, residual, *_ = plots
    a, b = tmp_path / "native.png", tmp_path / "with_available_residual.png"
    write_live_plot_image(a, main.getPlotItem())
    write_live_plot_image(b, main.getPlotItem(), residual_item=residual.getPlotItem())
    assert a.read_bytes() == b.read_bytes()


def test_dialog_scales_frozen_preview_and_residual_is_explicit(plots, qtbot, monkeypatch):
    main, residual, *_ = plots
    preview = F.capture_plot_preview(main.getPlotItem())
    combined = F.capture_plot_preview(main.getPlotItem(), residual_item=residual.getPlotItem(),
                                      include_residual=True)
    dialog = ExportOptionsDialog(
        native_size=F.figure_native_size(main.getPlotItem()), preview_image=preview,
        residual_preview_image=combined, residual_native_size=F.figure_native_size(
            main.getPlotItem(), residual_item=residual.getPlotItem(), include_residual=True))
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(10)
    assert not dialog.options().include_residual
    assert not dialog.preview_label.pixmap().isNull()
    assert dialog.preview_note.geometry().top() > dialog.preview_label.geometry().bottom()
    assert dialog.preview_label.height() >= dialog.preview_label.pixmap().height()
    monkeypatch.setattr(F, "_capture_image", lambda *a, **k: pytest.fail("preview recaptured live plot"))
    dialog.residual_check.setChecked(True)
    assert dialog.options().include_residual
    dialog.preset_combo.setCurrentIndex(1)
    assert dialog.options().physical_width_mm == 85
    dialog.dpi_spin.setValue(600)
    dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(".svg"))
    assert not dialog.preview_label.pixmap().isNull()
    assert "not physical screen size" in dialog.preview_note.text()


def test_choose_live_export_passes_detached_preview_to_options(plots, monkeypatch, tmp_path):
    from nanoscope.gui import plot_export as P

    main, *_ = plots
    captured = {}
    def choose(parent, size, **kwargs):
        captured.update(kwargs)
        return resolve_export_options(size)
    monkeypatch.setattr(P, "choose_export_options", choose)
    monkeypatch.setattr(P, "ask_export_path", lambda *a, **k: tmp_path / "figure.png")
    out, options = choose_live_plot_export(None, main.getPlotItem(), "figure.png")
    assert out.name == "figure.png"
    assert isinstance(captured["preview_image"], QtGui.QImage)
    assert captured["residual_preview_image"] is None
    assert options.native and not options.include_residual


@pytest.mark.parametrize("include_residual", [False, True])
@pytest.mark.parametrize("extension", [".png", ".svg", ".pdf"])
def test_interactive_export_uses_preview_even_if_background_result_changes(
        plots, monkeypatch, tmp_path, extension, include_residual):
    from dataclasses import replace
    from nanoscope.gui import plot_export as P
    main, residual, curve, resid_curve, x, y = plots
    captured = {}
    def choose(_parent, size, **kwargs):
        # Equivalent to a queued worker completion while the modal dialog runs.
        curve.setData(x, y + 200)
        resid_curve.setData(x, y - 500)
        captured["preview"] = (kwargs["residual_preview_image"] if include_residual
                               else kwargs["preview_image"])
        if include_residual:
            size = kwargs["residual_native_size"]
        return replace(resolve_export_options(size, extension=extension),
                       include_residual=include_residual)
    monkeypatch.setattr(P, "choose_export_options", choose)
    monkeypatch.setattr(P, "ask_export_path", lambda *a, **k: tmp_path / ("approved" + extension))
    out, options = P.choose_live_plot_export(
        None, main.getPlotItem(), "figure.png", residual_item=residual.getPlotItem())
    snapshot = options.frozen_snapshot
    assert snapshot is not None and snapshot.includes_residual == include_residual
    assert captured["preview"] == QtGui.QImage.fromData(snapshot.png_bytes)
    monkeypatch.setattr(F, "capture_figure_snapshot", lambda *a, **k: pytest.fail("recaptured live picture"))
    monkeypatch.setattr(F, "capture_figure_render", lambda *a, **k: pytest.fail("recaptured live picture"))
    P.write_live_plot_image(out, main.getPlotItem(), frozen_snapshot=snapshot,
                            include_residual=options.include_residual)
    if extension == ".png":
        assert out.read_bytes() == snapshot.png_bytes
    elif extension == ".svg":
        assert out.read_bytes() == snapshot.svg_bytes
    else:
        assert out.read_bytes().startswith(b"%PDF")
    np.testing.assert_array_equal(curve.yData, y + 200)
    np.testing.assert_array_equal(resid_curve.yData, y - 500)


@pytest.mark.parametrize("extension", [".png", ".svg", ".pdf"])
@pytest.mark.parametrize("physical", [False, True])
def test_frozen_publication_image_has_exact_dimensions_without_live_access(
        plots, tmp_path, monkeypatch, extension, physical):
    main, residual, *_ = plots
    snapshot = F.capture_figure_snapshot(
        main.getPlotItem(), data=pd.DataFrame({"x": [math.pi]}),
        residual_item=residual.getPlotItem(), include_residual=True)
    before = snapshot.to_storage()
    monkeypatch.setattr(F, "capture_figure_render", lambda *a, **k: pytest.fail("live recapture"))
    monkeypatch.setattr(F, "_capture_image", lambda *a, **k: pytest.fail("live render"))
    options = resolve_export_options((snapshot.width, snapshot.height), extension=extension,
                                     native=not physical, width=50.8, height=25.4, dpi=300)
    out = tmp_path / ("figure" + extension)
    assert F.write_frozen_figure_image(out, snapshot, options) == []
    if extension == ".svg":
        svg = ET.fromstring(out.read_bytes())
        assert svg.attrib["viewBox"] == ET.fromstring(snapshot.svg_bytes).attrib["viewBox"]
        if physical:
            assert (svg.attrib["width"], svg.attrib["height"]) == ("50.8mm", "25.4mm")
        else:
            assert out.read_bytes() == snapshot.svg_bytes
    elif extension == ".png":
        image = QtGui.QImage(str(out))
        if physical:
            assert (image.width(), image.height()) == (600, 300)
            assert image.dotsPerMeterX() == image.dotsPerMeterY() == round(300 / 0.0254)
        else:
            assert out.read_bytes() == snapshot.png_bytes
    else:
        box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out.read_bytes())
        assert box
        dims = tuple(map(float, box.groups()))[2:]
        assert dims == pytest.approx((144, 72) if physical else
                                    (snapshot.width, snapshot.height), abs=1.0)
    assert snapshot.to_storage() == before


def test_frozen_sized_png_contains_both_main_and_residual_curve(plots, tmp_path):
    main, residual, *_ = plots
    snapshot = F.capture_figure_snapshot(
        main.getPlotItem(), data=pd.DataFrame(), residual_item=residual.getPlotItem(),
        include_residual=True)
    options = resolve_export_options((snapshot.width, snapshot.height), native=False,
                                     width=101.6, height=101.6, dpi=150)
    out = tmp_path / "combined.png"
    F.write_frozen_figure_image(out, snapshot, options)
    image = QtGui.QImage(str(out)).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
    blue = (pixels[:, :, 2] > 100) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 120)
    red = (pixels[:, :, 0] > 120) & (pixels[:, :, 1] < 90) & (pixels[:, :, 2] < 100)
    assert blue.sum() > 30
    assert red.sum() > 30
    assert np.where(red)[0].mean() > np.where(blue)[0].mean()


def test_frozen_render_preserves_geometry_aspect_ratio(qapp, tmp_path):
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    painter.fillRect(QtCore.QRectF(0, 0, 200, 100), QtGui.QColor("#b92429"))
    painter.end()
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    assert picture.save(buffer)
    out = tmp_path / "letterboxed.png"
    F.render_frozen_picture(bytes(buffer.data()), out, width=400, height=400, native_size=(200, 100))
    image = QtGui.QImage(str(out)).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
    rows, cols = np.where(pixels[:, :, 3] > 200)
    assert cols.ptp() + 1 == 400
    assert rows.ptp() + 1 == 200


@pytest.mark.parametrize("extension", [".png", ".svg", ".pdf"])
@pytest.mark.parametrize("physical", [False, True])
def test_available_but_unselected_residual_does_not_change_live_export(
        plots, tmp_path, extension, physical):
    main, residual, *_ = plots
    options = resolve_export_options(F.figure_native_size(main.getPlotItem()),
                                     extension=extension, native=not physical,
                                     width=50.8, height=25.4)
    kwargs = {"width": options.width, "height": options.height,
              "physical_width_mm": options.physical_width_mm,
              "physical_height_mm": options.physical_height_mm,
              "dpi": options.dpi if not options.native else None}
    first, second = tmp_path / ("a" + extension), tmp_path / ("b" + extension)
    ranges = main.viewRange(), residual.viewRange()
    write_live_plot_image(first, main.getPlotItem(), **kwargs)
    write_live_plot_image(second, main.getPlotItem(), residual_item=residual.getPlotItem(), **kwargs)
    if extension != ".pdf":
        assert first.read_bytes() == second.read_bytes()
    else:
        # Qt emits a new document UUID and clock metadata for each identical PDF.
        def geometry(payload):
            payload = re.sub(rb"/(?:CreationDate|ModDate|ID)\s*(?:\([^)]*\)|\[[^]]*\])", b"", payload)
            payload = re.sub(rb'uuid:[0-9a-f-]{36}', b'uuid:<same-document>', payload)
            return re.sub(rb'xmp:(?:CreateDate|ModifyDate|MetadataDate)="[^"]*"', b'', payload)
        assert geometry(first.read_bytes()) == geometry(second.read_bytes())
    assert ranges == (main.viewRange(), residual.viewRange())


def test_frozen_image_failure_does_not_replace_earlier_export(plots, tmp_path, monkeypatch):
    main, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame())
    target = tmp_path / "figure.pdf"
    target.write_bytes(b"earlier PDF")
    monkeypatch.setattr(F, "render_frozen_picture", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    options = resolve_export_options((snapshot.width, snapshot.height), extension="pdf")
    with pytest.raises(OSError, match="disk full"):
        F.write_frozen_figure_image(target, snapshot, options)
    assert target.read_bytes() == b"earlier PDF"


def test_frozen_image_rejects_mismatched_format_size_and_missing_residual(plots, tmp_path):
    main, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame())
    options = resolve_export_options((snapshot.width, snapshot.height))
    for target, bad, match in (
            ("wrong.pdf", options, "extension"),
            ("wrong.png", replace(options, width=500), "dimensions"),
            ("wrong.png", replace(options, include_residual=True), "residual")):
        with pytest.raises(ValueError, match=match):
            F.write_frozen_figure_image(tmp_path / target, snapshot, bad)
        assert not (tmp_path / target).exists()


@pytest.mark.parametrize("changes", [
    {"version": True}, {"width": True}, {"height": "400"}, {"includes_residual": 1},
    {"sha256": []}, {"sha256": {}}, {"resources": []},
    {"resources": {"safe.bin": []}}, {"resources": {"safe.bin": "missing"}},
])
def test_snapshot_rejects_malformed_storage_metadata(plots, changes):
    main, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame())
    metadata, blobs = snapshot.to_storage()
    with pytest.raises(ValueError):
        F.FigureSnapshot.from_storage({**metadata, **changes}, blobs)


def test_resource_checksum_failure_and_case_collisions_are_rejected(plots):
    main, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame(),
                                         additional_resources={"analysis/source.bin": b"good"})
    metadata, blobs = snapshot.to_storage()
    with pytest.raises(ValueError, match="checksum"):
        F.FigureSnapshot.from_storage(metadata, {**blobs, "resource_0": b"changed"})
    bad = replace(snapshot, additional_resources=(("analysis/A.bin", b"a"),
                                                   ("analysis/a.bin", b"b")))
    with pytest.raises(ValueError, match="unique"):
        bad.to_storage()


def test_mixed_event_row_ids_and_missing_object_scalars_roundtrip_losslessly(plots, tmp_path):
    main, *_ = plots
    values = ["", 17, True, None, pd.NA, float("nan"), -0.0, np.int64(23), np.float32(0.5), b"audit"]
    table = pd.DataFrame({"event_row_id": pd.Series(values, dtype=object),
                          "value": np.linspace(0.0, 1.0, len(values)),
                          "label": ["source"] * len(values)})
    table.index = pd.Index(["bin", 9, "event", 10, "audit", 11, "rest", 12, "scope", 13],
                            dtype=object, name="row_identity")
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=table)
    restored = snapshot.dataframe()
    assert_frame_equal(restored, table, check_exact=True)
    assert [type(value) for value in restored.event_row_id] == [type(value) for value in values]
    assert restored.event_row_id.iloc[4] is pd.NA
    assert np.signbit(restored.event_row_id.iloc[6])
    import pyarrow.parquet as pq
    arrow = pq.read_table(io.BytesIO(snapshot.data_parquet_bytes))
    codec = json.loads(arrow.schema.metadata[b"nanoscope.figure_table_codec"])
    assert codec["object_columns"] == [0]
    assert codec["object_index_levels"] == [0]
    assert str(arrow.schema.field("value").type) == "double"
    assert_frame_equal(restored[["value"]], table[["value"]], check_exact=True)
    path = F.write_figure_bundle(tmp_path / "mixed.zip", snapshot)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert "tagged JSON scalars" in manifest["data_contract"]
        assert archive.read("data.csv") == table.to_csv(index=False, float_format="%.17g").encode("utf-8")


def test_vector_picture_preserves_background_and_clips_curve_to_plot_area(plots, tmp_path):
    main, residual, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame(),
                                         residual_item=residual.getPlotItem(), include_residual=True)
    options = resolve_export_options((snapshot.width, snapshot.height), native=False,
                                     width=snapshot.width / 100 * 25.4,
                                     height=snapshot.height / 100 * 25.4, dpi=100)
    path = tmp_path / "faithful.png"
    F.write_frozen_figure_image(path, snapshot, options)
    image = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
    assert tuple(pixels[5, -5]) == (255, 255, 255, 255)
    blue = (pixels[:, :, 2] > 100) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 120)
    red = (pixels[:, :, 0] > 120) & (pixels[:, :, 1] < 90) & (pixels[:, :, 2] < 100)
    left = main.getPlotItem().getViewBox().sceneBoundingRect().left()
    assert np.where(blue)[1].min() >= left - 2
    assert np.where(red)[1].min() >= residual.getPlotItem().getViewBox().sceneBoundingRect().left() - 2
    assert np.where(red)[0].mean() > np.where(blue)[0].mean()


def test_frozen_pdf_and_png_keep_same_vector_geometry_at_publication_size(plots, tmp_path):
    QtPdf = pytest.importorskip("PySide6.QtPdf")
    main, residual, *_ = plots
    snapshot = F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame(),
                                         residual_item=residual.getPlotItem(), include_residual=True)
    images = []
    for extension in (".png", ".pdf"):
        options = resolve_export_options((snapshot.width, snapshot.height), extension=extension,
                                         native=False, width=101.6,
                                         height=101.6 * snapshot.height / snapshot.width, dpi=150)
        path = tmp_path / ("same" + extension)
        F.write_frozen_figure_image(path, snapshot, options)
        if extension == ".png":
            image = QtGui.QImage(str(path))
        else:
            assert b"/Subtype /Image" not in path.read_bytes(), "curve-only PDF must remain vector"
            document = QtPdf.QPdfDocument()
            assert document.load(str(path)) == QtPdf.QPdfDocument.Error.None_
            image = document.render(0, images[0].size())
            document.close()
        assert not image.isNull()
        images.append(image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888))
    bounds = []
    for image in images:
        pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
        blue = (pixels[:, :, 2] > 100) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 120)
        red = (pixels[:, :, 0] > 120) & (pixels[:, :, 1] < 90) & (pixels[:, :, 2] < 100)
        rows, cols = np.where(blue | red)
        bounds.append((cols.min(), cols.max(), rows.min(), rows.max()))
    assert bounds[0] == pytest.approx(bounds[1], abs=3)


def test_unserializable_object_refused_before_render_or_output(plots, monkeypatch):
    main, *_ = plots
    monkeypatch.setattr(F, "capture_figure_render", lambda *a, **k: pytest.fail("validate before capture"))
    with pytest.raises(TypeError, match="losslessly encode"):
        F.capture_figure_snapshot(main.getPlotItem(), data=pd.DataFrame({"unsafe": [object()]}))


@pytest.mark.parametrize("encoded", [
    ["numpy", "|i1", ["int", "500"]], ["numpy", "|b1", ["str", "false"]],
    ["numpy", "<f4", ["float", float(0.1).hex()]], ["numpy", "<U1", ["str", "long"]],
])
def test_numpy_scalar_codec_refuses_silent_loss_or_tag_coercion(encoded):
    with pytest.raises(ValueError):
        F._restore_object_scalar(encoded)


@pytest.mark.parametrize("extension", [".png", ".pdf"])
def test_live_residual_raster_pdf_skip_expensive_svg_serialization(plots, tmp_path, monkeypatch, extension):
    main, residual, *_ = plots
    monkeypatch.setattr(F, "_capture_svg", lambda *a, **k: pytest.fail("PNG/PDF serialized an SVG"))
    out = tmp_path / ("current" + extension)
    write_live_plot_image(out, main.getPlotItem(), residual_item=residual.getPlotItem(),
                          include_residual=True)
    assert out.stat().st_size > 100
