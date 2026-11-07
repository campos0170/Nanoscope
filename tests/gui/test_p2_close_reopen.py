"""Logical figures survive shell close and a real project save/open without eager pages."""
from PySide6 import QtWidgets

from nanoscope.gui.main_window import MainWindow


def test_multiple_same_kind_figures_save_reopen_frozen_and_keep_names(qapp, qtbot, tmp_path):
    first = MainWindow(recovery_enabled=False)
    qtbot.addWidget(first)
    owner = first._ensure_figure_workspace()
    ids = [owner.create("iv") for _ in range(3)]
    for i, key in enumerate(ids):
        owner.rename(key, f"Independent {i}")
        assert owner.activate(key)
        page = owner.page(key)
        edit = page._appearance_widget.title
        edit.selectAll()
        qtbot.keyClicks(edit, f"Title {i}")
        assert page._appearance_widget.state().title == f"Title {i}"
        assert owner.history(key).mark == 1  # one typing run, not one step per character
        assert owner.checkpoint(key)
        assert next(text.words.value for text in owner.get(key).spec.texts
                    if text.element_id == "title") == f"Title {i}"
    first._analysis_explorer.close()
    assert len(owner.records()) == 3 and len(owner.resident_ids()) == 3
    path = tmp_path / "independent.nanoscope"
    assert first._write_project(str(path))
    second = MainWindow(recovery_enabled=False)
    qtbot.addWidget(second)
    with qtbot.waitSignal(second.projectRestored, timeout=15000):
        second._open_project(str(path))
    restored = second._figure_workspace
    assert [r.spec.figure_id for r in restored.records()] == ids
    assert [r.spec.name for r in restored.records()] == [f"Independent {i}" for i in range(3)]
    assert not restored.resident_ids()
    assert all(r.spec.result.status == "empty" for r in restored.records())
    assert restored.activate(ids[1])
    assert restored.page(ids[1])._appearance_widget.state().title == "Title 1"
    assert len(restored.resident_ids()) == 1


def test_conversion_cancel_copy_and_verified_in_place_backup(qapp, qtbot, tmp_path, monkeypatch):
    import hashlib
    from nanoscope.core import project
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    path = tmp_path / "old.nanoscope"
    with monkeypatch.context() as old:
        old.setattr(project, "VERSION", 12)
        assert window._write_project(str(path))
    original = path.read_bytes()
    monkeypatch.setattr(project, "VERSION", 13)
    monkeypatch.setattr(window, "_confirm_figure_project_conversion", lambda target: "cancel")
    assert not window._write_project(str(path)) and path.read_bytes() == original
    copied = tmp_path / "copy.nanoscope"
    monkeypatch.setattr(window, "_confirm_figure_project_conversion", lambda target: "copy")
    monkeypatch.setattr(window, "_save_project_as", lambda: window._write_project(str(copied)))
    assert window._write_project(str(path)) and path.read_bytes() == original
    assert project.read_project_container_version(copied) == 13
    assert not list(tmp_path.glob("*.bak"))
    window._project_path = str(path)
    window._project_disk_stamp = window._disk_stamp_of(str(path))
    monkeypatch.setattr(window, "_confirm_figure_project_conversion", lambda target: "convert")
    assert window._write_project(str(path))
    backup = tmp_path / "old.nanoscope.v12.bak"
    assert backup.read_bytes() == original
    assert hashlib.sha256(backup.read_bytes()).digest() == hashlib.sha256(original).digest()
    assert project.read_project_container_version(path) == 13
    assert window._last_project_conversion_backup == str(backup)


def test_conversion_dialog_cannot_save_a_different_project_over_old_target(qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.core import project
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    path = tmp_path / "old.nanoscope"
    with monkeypatch.context() as old:
        old.setattr(project, "VERSION", 12)
        assert window._write_project(str(path))
    original = path.read_bytes()
    monkeypatch.setattr(project, "VERSION", 13)
    def changed_project(target):
        window._figure_project_epoch += 1
        return "convert"
    monkeypatch.setattr(window, "_confirm_figure_project_conversion", changed_project)
    assert not window._write_project(str(path))
    assert path.read_bytes() == original and not list(tmp_path.glob("*.bak"))
