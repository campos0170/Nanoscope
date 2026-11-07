"""Runtime figure callbacks must describe published, editable figure families."""
import json
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis import acid_incubation as acid
from nanoscope.plot import acid_figure_recipe as recipe


@pytest.fixture
def runtime(tmp_path):
    with acid._math_session(7):
        session = acid.AcidRuntime(acid.default_settings(), tmp_path, {
            'tag': 'test', 'source_path': str(tmp_path / 'unused.abf'), 'incubation_h': 1})
        session.namespace['FIG_DPI'] = 60
        yield session
        for fig in list(session.plot.owned):
            session.plot.close(fig)


@pytest.mark.parametrize('joint', [False, True])
def test_runtime_publishes_final_png_recipe_and_numeric_archive(runtime, joint):
    if joint:
        grid = runtime.plot.joint_grid(height=3)
        fig, axis = grid.figure, grid.ax_joint
    else:
        fig, axis = runtime.plot.subplots(figsize=(3, 2))
    axis.plot([0, 1, 2], [3, 7, 4])
    target = runtime.namespace['FIG_DIR'] / 'measured.png'
    if joint:
        runtime.plot.save_figure(fig, target, dpi=60)
    else:
        runtime._finish_figure('measured')
    assert runtime.figures == [target]
    assert target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    metadata = json.loads(target.with_suffix('.recipe.json').read_text())
    assert metadata['arrays'] == 'measured.artists.npz'
    assert target.with_suffix('.artists.npz').is_file()
    rebuilt, _, _ = recipe.reconstruct(target)
    np.testing.assert_array_equal(rebuilt.axes[0].lines[0].get_ydata(), [3, 7, 4])
    assert not list(target.parent.glob('.saving-*'))
    assert {p.name for p in target.parent.iterdir()} == {
        'measured.png', 'measured.recipe.json', 'measured.artists.npz'}
    rebuilt.clear()


@pytest.mark.parametrize('failure', ['save', 'capture'])
def test_runtime_keeps_previous_png_if_save_or_capture_fails(runtime, monkeypatch, failure):
    fig, axis = runtime.plot.subplots(figsize=(3, 2))
    axis.plot([0, 1], [3, 7])
    target = runtime.namespace['FIG_DIR'] / 'measured.png'
    target.write_bytes(b'previous published PNG')
    if failure == 'save':
        def broken_save(path, *args, **kwargs):
            Path(path).write_bytes(b'partial PNG')
            raise OSError('injected image failure')
        monkeypatch.setattr(fig, 'savefig', broken_save)
    else:
        def broken_capture(figure, path):
            assert path == target
            assert path.read_bytes() == b'previous published PNG'
            raise OSError('injected capture failure')
        monkeypatch.setattr(recipe, 'capture_figure', broken_capture)
    with pytest.raises(OSError, match='injected'):
        runtime._finish_figure('measured')
    assert target.read_bytes() == b'previous published PNG'
    assert runtime.figures == []
    assert list(target.parent.iterdir()) == [target]
