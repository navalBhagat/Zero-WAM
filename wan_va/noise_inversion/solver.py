"""Euler integration on the same shifted sigma grid as video generation."""
from collections.abc import Callable

import torch


def integrate(
    sample: torch.Tensor,
    sigmas: torch.Tensor,
    velocity: Callable[[torch.Tensor, float], torch.Tensor],
    observed_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate in either direction; velocity receives sigma, not model time."""
    if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all():
        raise ValueError("Expected a finite one-dimensional sigma grid")
    differences = sigmas[1:] - sigmas[:-1]
    if not ((differences > 0).all() or (differences < 0).all()):
        raise ValueError("Sigma grid must be strictly monotonic")
    state = sample.clone()
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        if observed_frame is not None:
            state[:, :, :1] = observed_frame
        prediction = velocity(state, float(sigma)).to(state)
        state = state + prediction * (next_sigma - sigma).to(state)
        if observed_frame is not None:
            state[:, :, :1] = observed_frame
        if not torch.isfinite(state).all():
            raise RuntimeError("Nonfinite video state during flow integration")
    return state


def relative_error(prediction: torch.Tensor, target: torch.Tensor, first_chunk: bool) -> float:
    """Exclude the observed frame, which is clamped rather than generated."""
    start = 1 if first_chunk else 0
    predicted = prediction[:, :, start:].float()
    expected = target[:, :, start:].float()
    return float(torch.linalg.vector_norm(predicted - expected) /
                 torch.linalg.vector_norm(expected).clamp_min(1e-8))
