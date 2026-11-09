import pytest
import torch

# Load the pure helper without importing the inference stack in wan_va.__init__.
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location(
    "latent_guidance", Path(__file__).resolve().parents[1] / "wan_va/guidance/latent_guidance.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
guided_step = _module.guided_step
isolated_decode_cache = _module.isolated_decode_cache


def test_flow_estimate_and_gradient_under_no_grad():
    clean = torch.randn(1, 2, 2, 2, 3)
    velocity = torch.randn_like(clean)
    sample = clean + 0.4 * velocity
    base = sample - 0.1 * velocity
    seen = []

    def reward(x):
        seen.append(x.detach())
        return x.sum()

    with torch.no_grad():
        result, stats = guided_step(sample, velocity, 0.4, base, reward, 0.01,
                                    fixed_first_frame=True, max_norm=0)
    torch.testing.assert_close(seen[0], clean)
    torch.testing.assert_close(result[:, :, :1], base[:, :, :1])
    torch.testing.assert_close(result[:, :, 1:], base[:, :, 1:] + 0.01)
    assert not result.requires_grad
    assert stats['gradient_norm'] > 0


def test_zero_strength_skips_reward_and_preserves_identity():
    x = torch.ones(1)
    def fail(_):
        raise AssertionError('reward called')
    result, stats = guided_step(x, x, 1, x, fail, 0)
    assert result is x
    assert stats == {}


def test_directional_derivative_and_ascent():
    x = torch.randn(1, 1, 2, 2, 2, dtype=torch.float64)
    v = torch.randn_like(x)
    def reward(z):
        return -z.square().sum()
    result, _ = guided_step(x, v, 0.5, x, reward, 1e-5, max_norm=0)
    direction = (result - x) / 1e-5
    analytic = -2 * (x - 0.5 * v)
    torch.testing.assert_close(direction, analytic)
    eps = 1e-6
    difference = (reward(x + eps * direction - 0.5 * v) -
                  reward(x - eps * direction - 0.5 * v)) / (2 * eps)
    torch.testing.assert_close(difference, (analytic * direction).sum())
    assert reward(result - 0.5 * v) > reward(x - 0.5 * v)


def test_cache_restored_on_error():
    class VAE:
        def clear_cache(self):
            self._feat_map = [None]
            self._enc_feat_map = [None]
    vae = VAE()
    vae._feat_map, vae._enc_feat_map = [torch.ones(1)], [torch.zeros(1)]
    decoder, encoder = vae._feat_map, vae._enc_feat_map
    with pytest.raises(RuntimeError):
        with isolated_decode_cache(vae):
            assert vae._feat_map is not decoder
            raise RuntimeError('decode failed')
    assert vae._feat_map is decoder
    assert vae._enc_feat_map is encoder


def test_nonfinite_reward_rejected():
    x = torch.ones(1, 1, 2, 1, 1)
    with pytest.raises(FloatingPointError):
        guided_step(x, x, 0.5, x, lambda z: z.sum() * float('nan'), 1)


def test_fp32_state_preserves_small_correction_lost_in_bf16():
    x = torch.ones(1, 1, 2, 1, 1)
    result, stats = guided_step(x, torch.zeros_like(x), .5, x,
                                lambda clean: clean.sum(), .001,
                                fixed_first_frame=True, max_norm=0)
    assert result.dtype == torch.float32
    assert stats['applied_correction_norm'] > .0009
    assert stats['changed_fraction'] == 1
    assert (x.bfloat16() + .001).float().equal(x)
