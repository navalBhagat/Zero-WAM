import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'rendering', Path(__file__).resolve().parents[1] / 'evaluation/robotwin/rendering.py',
)
rendering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rendering)


def test_default_and_hardcoded_scene_calls_respect_override(monkeypatch):
    monkeypatch.delenv('RENDER_SAMPLES_PER_PIXEL', raising=False)
    calls = []
    render = SimpleNamespace(set_ray_tracing_samples_per_pixel=lambda count: calls.append(count))
    assert rendering.configure_rendering(render)['samples_per_pixel_effective'] == 32
    monkeypatch.setenv('RENDER_SAMPLES_PER_PIXEL', '16')
    rendering.configure_rendering(render)
    render.set_ray_tracing_samples_per_pixel(32)  # actual RoboTwin setup's hardcoded call
    assert calls[-2:] == [16, 16]
    # Reconfiguration does not recursively wrap the setter.
    rendering.configure_rendering(render, samples=8)
    rendering.configure_rendering(render)  # preflight rereads the same effective setting
    assert calls[-2:] == [8, 8]
    assert rendering.render_settings(render)['samples_per_pixel_requested'] == 8


@pytest.mark.parametrize('value', ['0', '-1', 'abc', '1.5', True])
def test_invalid_render_settings_fail_early(value):
    with pytest.raises(ValueError):
        rendering.requested_samples(value)
