"""Adapter identity initialization, gradients and portable checkpoints."""
from dataclasses import asdict

import pytest
import torch

from wan_va.noise_inversion.adapter import AdapterConfig, NoiseAdapter, load_adapter


@pytest.mark.parametrize("architecture", ["vit", "mlp"])
def test_zero_head_preserves_original_noise_and_receives_gradients(architecture, tmp_path):
    config = AdapterConfig(width=32, layers=1, heads=4, context_tokens=8, architecture=architecture)
    model = NoiseAdapter(config)
    noise = torch.randn(1, 48, 2, 4, 4)
    output = model(noise, noise, noise, torch.randn(1, 30, 2, 16, 1))
    assert torch.equal(output, noise)
    (output - 1).square().mean().backward()
    assert model.output.weight.grad is not None
    path = tmp_path / "adapter.pt"
    torch.save(dict(config=asdict(config), model=model.state_dict()), path)
    restored = load_adapter(path, "cpu")
    assert restored.config == config
