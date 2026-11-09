"""Stop-gradient flow guidance. Transformer predictions are treated as constants."""
from contextlib import contextmanager

import torch


@contextmanager
def isolated_decode_cache(vae):
    """Wan decode clears internal caches. Restore their original objects even on error.

    The observation wrapper's feat_cache is separate and is never touched here.
    This assumes serial server inference (the existing server execution model).
    """
    names = ("_conv_num", "_conv_idx", "_feat_map", "_enc_conv_num",
             "_enc_conv_idx", "_enc_feat_map")
    saved = {name: getattr(vae, name) for name in names if hasattr(vae, name)}
    try:
        vae.clear_cache()
        yield
    finally:
        vae.clear_cache()
        for name, value in saved.items():
            setattr(vae, name, value)


def guided_step(sample, velocity, sigma, base_next, reward_fn, strength,
                fixed_first_frame=False, max_norm=1.0):
    """Return corrected Euler output and scalar diagnostics; never retain the graph.

    Raw gradient ascent, with a global L2 cap for stability. The finite-difference
    diagnostic must hold velocity fixed to match this stop-gradient approximation.
    """
    if strength == 0:
        return base_next, {}
    with torch.enable_grad():
        x = sample.detach().requires_grad_(True)
        clean = x - torch.as_tensor(sigma, device=x.device, dtype=x.dtype) * velocity.detach()
        reward = reward_fn(clean)
        grad, = torch.autograd.grad(reward, x)
    if fixed_first_frame:
        grad = grad.clone()  # autograd may return an expanded, aliased gradient
        grad[:, :, :1] = 0
    if not torch.isfinite(reward) or not torch.isfinite(grad).all():
        raise FloatingPointError("Non-finite external reward or gradient")
    norm = grad.float().norm()
    if max_norm > 0:
        grad = grad * (max_norm / norm.clamp_min(1e-12)).clamp(max=1).to(grad.dtype)
    correction = strength * grad
    stats = {"reward": reward.item(), "gradient_norm": norm.item(),
             "correction_norm": correction.float().norm().item(),
             "base_update_norm": (base_next - sample).float().norm().item()}
    return (base_next + correction).detach(), stats
