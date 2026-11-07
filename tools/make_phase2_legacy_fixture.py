"""Generate ONE bounded old-writer dialect fixture, with no QWidget/QApplication.

Run this script in a fresh process per revision. It imports the historical GUI module
only to construct its frozen PlotConfig value; it creates no windows and executes no
analysis. Nine seeded synthetic rows are explicit fixture data, not an experiment.

Example (root coordinator executes serially in its Qt-safe environment)::
    python tools/make_phase2_legacy_fixture.py --repo /path/to/historical/repo \
      --dialect D1 --output /path/to/fixture/output

The historical revision is archived into a temporary directory and read only. Nothing
is checked out into the user's working tree. SHA provenance pins code and output bytes.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REVISIONS = {
    "D1": ("54c14838834aec453a9ebc052611604ad04ffbb5", 12),
    "D2": ("56bdf1e60a8122b242c5e83cf983f2fddc8c56d6", 12),
    "v11": ("5eae42a9d74326980553848f3d934c921b00a9cb", 11),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dialect", choices=REVISIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision, expected_version = REVISIONS[args.dialect]
    resolved = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", revision], text=True).strip()
    if resolved != revision:
        raise RuntimeError("Historical revision identity differs")
    archive_bytes = subprocess.check_output(["git", "-C", str(args.repo), "archive", revision, "nanoscope"])
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanoscope-historical-fixture-") as temporary:
        root = Path(temporary).resolve()
        tree = root / "tree"
        tree.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if member.name.startswith("/") or ".." in parts or member.issym() or member.islnk():
                    raise RuntimeError("Historical archive contains an unsafe member")
            archive.extractall(tree)  # Member paths and links were validated above.
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[variable] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["NANOSCOPE_RECOVERY_ROOT"] = str(root / "recovery")
        os.environ["NANOSCOPE_INCUBATION_LIBRARY_ROOT"] = str(root / "study")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(tree))
        import nanoscope  # Historical bootstrap sets Qt plugin paths before PySide.
        # Isolate even incidental QSettings use before importing the old GUI module.
        from PySide6 import QtCore, QtWidgets
        settings = root / "settings"
        settings.mkdir()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        for scope in (QtCore.QSettings.Scope.UserScope, QtCore.QSettings.Scope.SystemScope):
            QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, scope, str(settings))
        assert QtWidgets.QApplication.instance() is None

        import numpy as np
        import pandas as pd
        from nanoscope.core import project, analysis_workspace
        from nanoscope.gui import plots_panel as plots
        assert Path(nanoscope.__file__).resolve().is_relative_to(tree)
        assert Path(project.__file__).resolve().is_relative_to(tree)
        assert project.VERSION == expected_version, (project.VERSION, expected_version)
        blank = plots.PlotConfig()
        supported = {f.name for f in fields(blank.style)}
        changes = {key: value for key, value in {
            "title": "Historical synthetic fixture", "x_label": "Own dwell label",
            "y_label": " ", "title_bold": True, "title_italic": True,
            "fit_values_on_plot": True, "show_r2": False, "show_gof": False,
        }.items() if key in supported}
        styled = replace(blank, style=replace(blank.style, **changes))
        if "fit_label" in supported:
            # D2 owns an explicit saved fit-label template in addition to old toggles.
            style = plots.FrozenMap({"visible": True, "fields": ("equation", "r2", "n", "ks"),
                                    "template": "Own r2={r2}; ks={ks}", "placement": "top-left"})
            styled = replace(styled, style=replace(styled.style, fit_label=style))
        rng = np.random.default_rng(20261103)
        frame = pd.DataFrame({"dwell_s": rng.uniform(.001, .02, 9),
                              "depth_pA": rng.normal(30., 2., 9), "accepted": [True] * 9})
        artifacts = []
        witnesses = {}
        for name, config in (("defaults", blank), ("legacy-look", styled)):
            resources = analysis_workspace._Resources()
            payload = {"config": analysis_workspace._encode(config, resources, "config"),
                       "selected_sources": [], "source_frame_fingerprint": None,
                       "cluster": None, "pca": None, "fits": [],
                       "fixture_seeded_table": "seeded.events"}
            tables = dict(resources.tables)
            tables["seeded.events"] = frame.copy(deep=True)
            artifact = project.WorkspaceArtifact("builtin.plots", "nanoscope.plots", 1,
                payload, tables, dict(resources.blobs),
                {"fixture_kind": "old-writer/frozen-config/seeded-synthetic-data"})
            path = args.output / f"{args.dialect.lower()}-{name}.nanoscope"
            if path.exists():
                raise RuntimeError(f"Refusing to replace an existing fixture: {path.name}")
            project.save_project(path, project.ProjectState(workspace=project.AnalyticalWorkspace(artifacts=[artifact])))
            witnesses[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                "plot_config": plots.plot_config_to_json(config), "rows": len(frame)}
            artifacts.append(path.name)
        assert QtWidgets.QApplication.instance() is None
        provenance = {"revision": revision, "dialect": args.dialect, "writer_project_version": project.VERSION,
            "writer_app_version": str(getattr(nanoscope, "__version__", "")),
            "source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "method": "Historical actual PlotConfig value and tagged metadata codec, actual historical save_project; no QWidget/QApplication or analysis execution.",
            "data": "Nine synthetic rows from numpy.default_rng(20261103); intentionally not experimental data.",
            "files": witnesses}
        (args.output / f"{args.dialect.lower()}-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps({"dialect": args.dialect, "revision": revision, "files": artifacts, "created_widgets": 0}))


if __name__ == "__main__":
    main()
