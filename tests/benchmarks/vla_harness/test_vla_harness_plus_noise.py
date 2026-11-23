"""Compare against the installed official function, including its RNG state."""
import ast
import os
from pathlib import Path
import numpy as np
import pytest
gaussian = pytest.importorskip("skimage.filters").gaussian
pytest.importorskip("numba")
from benchmarks.vla_harness.libero_plus_noise import glass_blur

def reference(name="glass_blur"):
    root = Path(os.environ.get("LIBERO_PLUS_PATH", "/workspace/code/benchmarks/LIBERO-plus"))
    source = root / "libero/libero/envs/env_wrapper.py"
    if not source.is_file():
        pytest.skip("Requires the official LIBERO-plus checkout (LIBERO_PLUS_PATH)")
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"np": np, "gaussian": gaussian}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]

@pytest.mark.parametrize("severity", range(1, 11))
@pytest.mark.parametrize("shape", [(256, 256, 3), (19, 23, 3), (5, 7, 3)])
def test_pixels_and_rng_are_identical(severity, shape):
    original = reference()
    image = np.random.RandomState(137).randint(0, 256, shape, dtype=np.uint8)
    np.random.seed(10000 + severity)
    expected = original(image, severity)
    expected_rng = np.random.get_state()
    np.random.seed(10000 + severity)
    actual = glass_blur(image, severity)
    actual_rng = np.random.get_state()
    np.testing.assert_array_equal(actual, expected)
    assert all(np.array_equal(a, b) for a, b in zip(actual_rng, expected_rng))

@pytest.mark.parametrize("factor", [1.0, 1.01, 1.1, 1.25, 1.39, 1.5])
@pytest.mark.parametrize("size", [64, 256])
def test_zoom_pixels_and_rng_are_identical(factor, size):
    from scipy.ndimage import zoom as scizoom
    from benchmarks.vla_harness.libero_plus_noise import clipped_zoom
    original = reference("clipped_zoom")
    original.__globals__["scizoom"] = scizoom
    x = np.random.RandomState(83).rand(size, size, 3).astype(np.float32)
    state = np.random.get_state()
    np.testing.assert_array_equal(clipped_zoom(x, factor), original(x, factor))
    assert all(np.array_equal(a, b) for a,b in zip(state, np.random.get_state()))


@pytest.mark.parametrize("severity", range(1, 11))
def test_full_zoom_noise_uses_identical_severity_factors(severity):
    from scipy.ndimage import zoom as scizoom
    from benchmarks.vla_harness.libero_plus_noise import clipped_zoom
    original = reference("zoom_blur")
    original_zoom = reference("clipped_zoom")
    original_zoom.__globals__["scizoom"] = scizoom
    original.__globals__["clipped_zoom"] = original_zoom
    image = np.random.RandomState(31 + severity).randint(
        0, 256, (256, 256, 3), dtype=np.uint8
    )
    state = np.random.get_state()
    expected = original(image, severity)
    original.__globals__["clipped_zoom"] = clipped_zoom
    np.testing.assert_array_equal(original(image, severity), expected)
    assert all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state()))
