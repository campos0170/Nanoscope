"""A frozen TF import must share one native image across both loader paths."""
import importlib.util
from pathlib import Path

import pytest


def _module(name):
    path = Path(__file__).resolve().parents[2] / "packaging" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_symlinked_environment_dependency_becomes_one_image(tmp_path):
    runtime = _module("tensorflow_runtime")
    real = tmp_path / "actual-runtime.so"
    real.write_bytes(b"one native image")
    environment = tmp_path / "environment-runtime.so"
    environment.symlink_to(real.name)
    canonical = (runtime._EXTENSION, str(environment), "EXTENSION")
    dependency = (runtime._ALIAS, str(real), "BINARY")
    unrelated = ("libother.dylib", "/unrelated/library", "BINARY")
    original = [canonical, dependency, unrelated]
    result = runtime.deduplicate_tensorflow_runtime(original)
    assert result == [canonical, (runtime._ALIAS, runtime._EXTENSION, "SYMLINK"), unrelated]
    assert original[1] == dependency
    assert runtime.deduplicate_tensorflow_runtime(result) == result


def test_analysis_refuses_a_different_tensorflow_runtime(tmp_path):
    runtime = _module("tensorflow_runtime")
    one, two = tmp_path / "one.so", tmp_path / "two.so"
    one.write_bytes(b"same bytes still does not establish source identity")
    two.write_bytes(one.read_bytes())
    with pytest.raises(ValueError, match="extension differ"):
        runtime.deduplicate_tensorflow_runtime([
            (runtime._EXTENSION, str(one), "EXTENSION"),
            (runtime._ALIAS, str(two), "BINARY"),
        ])


@pytest.mark.parametrize("canonical", [[], [("tensorflow/python/_pywrap_tensorflow_internal.so", "unused", "BINARY")]])
def test_analysis_refuses_unresolved_extension_identity(canonical):
    runtime = _module("tensorflow_runtime")
    with pytest.raises(ValueError, match="Ambiguous"):
        runtime.deduplicate_tensorflow_runtime(canonical + [(runtime._ALIAS, "unused", "BINARY")])


def test_analysis_keeps_extension_when_no_top_level_copy_was_collected():
    runtime = _module("tensorflow_runtime")
    entries = [(runtime._EXTENSION, "unused", "EXTENSION")]
    assert runtime.deduplicate_tensorflow_runtime(entries) == entries


def test_bundle_invariant_rejects_copied_image_but_accepts_relative_alias(tmp_path):
    verifier = _module("verify_bundle")
    framework = tmp_path / "NanoScope.app/Contents/Frameworks"
    canonical = framework / "tensorflow/python/_pywrap_tensorflow_internal.so"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"same bytes, distinct loaded image")
    alias = framework / canonical.name
    alias.write_bytes(canonical.read_bytes())
    with pytest.raises(SystemExit, match="duplicate TensorFlow native images"):
        verifier.check_tensorflow_image_identity(tmp_path / "NanoScope.app")
    alias.unlink()
    alias.symlink_to(canonical.relative_to(framework))
    assert verifier.main(["--check-tensorflow-image", str(tmp_path / "NanoScope.app")]) == 0


def test_bundle_invariant_rejects_missing_runtime_and_dangling_alias(tmp_path):
    verifier = _module("verify_bundle")
    app = tmp_path / "NanoScope.app"
    with pytest.raises(SystemExit, match="extension is missing"):
        verifier.check_tensorflow_image_identity(app)
    canonical = app / "Contents/Frameworks/tensorflow/python/_pywrap_tensorflow_internal.so"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"native image")
    (app / "Contents/Frameworks/_pywrap_tensorflow_internal.so").symlink_to("missing.so")
    with pytest.raises(SystemExit, match="alias is dangling"):
        verifier.check_tensorflow_image_identity(app)
