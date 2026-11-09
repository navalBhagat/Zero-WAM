"""CPU checks without importing the transformer/inference dependency stack."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_package = types.ModuleType('_reference_test')
_package.__path__ = [str(Path(__file__).resolve().parents[1] / 'wan_va/guidance')]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location(
    '_reference_test.reference', Path(_package.__path__[0]) / 'reference.py',
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1).to(torch.bfloat16)
        self.decoder = torch.nn.Linear(1, 1).double()
        self.post_quant_conv = torch.nn.Linear(1, 1).double()
        self.config = SimpleNamespace(latents_mean=[1., 2.], latents_std=[2., 3.],
                                      scale_factor_temporal=4)
        self.clear_cache()

    def clear_cache(self):
        self._feat_map = [None]
        self._enc_feat_map = [None]

    def decode(self, z, return_dict=False):
        self.input = z
        frames = 1 + 4 * (z.shape[2] - 1)
        return (z.new_zeros(1, 3, frames, z.shape[3], z.shape[4]),)


def test_layouts_match_server_conventions():
    x = torch.arange(48).reshape(2, 3, 2, 4)
    assert reference.reshape_latent(x, {}).shape == (1, 2, 3, 2, 4)
    fhwc = x.permute(1, 2, 3, 0)
    torch.testing.assert_close(reference.reshape_latent(fhwc, {'latent_layout': 'f_h_w_c'}), x[None])
    flat = fhwc.reshape(-1, 2)
    item = dict(latent_num_frames=3, latent_height=2, latent_width=4)
    torch.testing.assert_close(reference.reshape_latent(flat, item), x[None])
    with pytest.raises(ValueError):
        reference.reshape_latent(torch.zeros(2, 2, 3, 2, 4), {})


def test_reference_preserves_single_view_width_and_denormalizes(tmp_path):
    path = tmp_path / 'human.pth'
    torch.save(dict(latent=torch.ones(1, 2, 4, 2, 6)), path)
    vae = VAE()
    cache = vae._feat_map
    frames = reference.reconstructed_reference(vae, path, 3, 6)
    assert frames.shape == (3, 3, 2, 6)  # no three-camera crop
    assert vae.input.shape[2] == 3  # only necessary causal prefix
    assert vae.input.dtype == torch.float64  # decoder, not encoder dtype
    assert torch.all(vae.input[:, 0] == 3) and torch.all(vae.input[:, 1] == 5)
    assert torch.all(frames == .5) and not frames.requires_grad
    assert vae._feat_map is cache
    with pytest.raises(ValueError):
        reference.reconstructed_reference(vae, path, 0, 99)
    with pytest.raises(ValueError):
        reference.reconstructed_reference(vae, path, 4, 4)


def test_cpu_reference_restores_decoder_even_on_failure():
    vae = VAE()
    encoder = next(vae.encoder.parameters())
    with pytest.raises(RuntimeError):
        with reference.decoder_on_cpu(vae):
            assert next(vae.decoder.parameters()).dtype == torch.float32
            assert next(vae.post_quant_conv.parameters()).dtype == torch.float32
            assert next(vae.encoder.parameters()) is encoder
            assert encoder.dtype == torch.bfloat16
            raise RuntimeError('decode failure')
    assert next(vae.decoder.parameters()).dtype == torch.float64
    assert next(vae.post_quant_conv.parameters()).dtype == torch.float64
