"""HumanGen cache layouts and isolated, single-view reference reconstruction."""
from contextlib import contextmanager

import torch

from .latent_guidance import isolated_decode_cache


def reshape_latent(latent, item):
    """Match the released server's supported HumanGen latent layouts."""
    if not torch.is_tensor(latent):
        raise ValueError("ICL latent must be a tensor")
    layout = str(item.get("latent_layout", "")).lower().replace("_", " ")
    if latent.ndim == 5:
        if latent.shape[0] != 1:
            raise ValueError(f"ICL latent batch must be 1, got {latent.shape}")
        return latent.contiguous()
    if latent.ndim == 4:
        if layout in ("f h w c", "f h w d"):
            return latent.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        return latent.unsqueeze(0).contiguous()
    if latent.ndim == 2:
        f, h, w = (int(item[key]) for key in
                   ("latent_num_frames", "latent_height", "latent_width"))
        if latent.shape[0] != f * h * w:
            raise ValueError(f"Flattened ICL latent {latent.shape} does not match {(f, h, w)}")
        return latent.view(f, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0)
    raise ValueError(f"Unsupported ICL latent shape: {latent.shape}")


def decode_normalized(vae, latent):
    """Decode normalized latents using the decoder's precision, not the encoder's."""
    parameter = next(vae.post_quant_conv.parameters())
    latent = latent.to(device=parameter.device, dtype=parameter.dtype)
    mean = latent.new_tensor(vae.config.latents_mean)[None, :, None, None, None]
    std = latent.new_tensor(vae.config.latents_std)[None, :, None, None, None]
    with isolated_decode_cache(vae):
        return vae.decode(latent * std + mean, return_dict=False)[0]


@contextmanager
def decoder_on_cpu(vae):
    """Reference-only CPU decode; restore decoder placement without moving encoder."""
    parameter = next(vae.decoder.parameters())
    device, dtype = parameter.device, parameter.dtype
    try:
        vae.decoder.to(device='cpu', dtype=torch.float32)
        vae.post_quant_conv.to(device='cpu', dtype=torch.float32)
        yield
    finally:
        vae.decoder.to(device=device, dtype=dtype)
        vae.post_quant_conv.to(device=device, dtype=dtype)


@torch.no_grad()
def reconstructed_reference(vae, path, start=0, stop=None):
    """Select decoded RGB frames; HumanGen latents are already normalized.

    Decode the causal prefix required for the segment. Call decoder_on_cpu around
    this for long references when the WAM and simulator already occupy the GPU.
    """
    if start < 0 or (stop is not None and stop <= start):
        raise ValueError("Reference indices must satisfy 0 <= start < stop")
    item = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(item, dict) or 'latent' not in item:
        raise ValueError("Human cache must contain 'latent'")
    latent = reshape_latent(item['latent'], item)
    if latent.shape[1] != len(vae.config.latents_mean):
        raise ValueError("Human latent channels do not match this VAE")
    scale = getattr(vae.config, 'scale_factor_temporal', 4)
    available = 1 + scale * (latent.shape[2] - 1)
    stop = available if stop is None else stop
    if stop > available:
        raise ValueError(f"Reference stop {stop} exceeds {available} decoded frames")
    prefix = 1 + (stop - 1 + scale - 1) // scale
    video = decode_normalized(vae, latent[:, :, :prefix])
    # Clone the selected slice so the full decoded sequence can be released.
    return ((video[0, :, start:stop].permute(1, 0, 2, 3).float() + 1) / 2).clamp(0, 1).clone()
