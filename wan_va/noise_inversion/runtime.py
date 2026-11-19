"""Server adapter: rebuild robot-only history and invert recorded video chunks."""
import json
import time
from pathlib import Path

import torch

from .artifacts import ChunkStore
from .solver import integrate, relative_error


def invert_episode(server, directory: str) -> dict:
    from ..modules.icl_model import OBSERVATION_CACHE_TYPE

    root = Path(directory)
    source = ChunkStore(root / "chunks")
    count = source.count()
    if not count:
        raise ValueError("No source chunks recorded")
    output = ChunkStore(root / "inverted")
    if output.count():
        raise FileExistsError("Inverted chunks already exist; choose a fresh pilot directory")
    started = time.perf_counter()
    server._reset(prompt="", use_icl=False, use_text=False, use_guidance=False,
                  video_guidance_scale=1.0, icl_guidance_scale=1.0,
                  target_text_source="empty", icl_text_source="target")
    if server.use_text or server.use_icl or server.use_guidance:
        raise ValueError("Inversion requires language, ICL, and reward guidance disabled")
    server.scheduler.set_timesteps(server.job_config.num_inference_steps)
    descending = torch.cat([server.scheduler.sigmas, torch.zeros(1)])
    errors = []
    for index in range(count):
        chunk = source.load(index)
        server.chunk_idx = index
        server.init_latent = chunk["initial_frame"].to(server.device)
        target = chunk["video"].to(server.device).float()
        observed = server.init_latent if index == 0 else None
        def velocity(state: torch.Tensor, sigma: float) -> torch.Tensor:
            return server._video_velocity(state, sigma * server.scheduler.num_train_timesteps)

        noise = integrate(target, descending.flip(0), velocity, observed)
        reconstructed = integrate(noise, descending, velocity, observed)
        error = relative_error(reconstructed, target, index == 0)
        errors.append(error)
        output.save(index, video_noise=noise, action_noise=chunk["action_noise"])
        if index + 1 < count:
            history = ChunkStore(root / "history").load(index)
            # Recompute transformer representations without human context. Encoded
            # camera latents themselves are independent of ICL/text conditioning.
            server.transformer.clear_prediction_cache(-1, server.cache_name)
            for mode, key in (("video", "video"), ("action", "actions")):
                payload = server._prepare_icl_stream(
                    history[key].to(server.device), 0, mode, OBSERVATION_CACHE_TYPE)
                server.transformer(payload, update_cache=1, cache_name=server.cache_name,
                                   mode=f"forward_{'latent' if mode == 'video' else 'action'}_only",
                                   clean_window_cache=mode == "action")
    report = dict(use_text=False, use_icl=False, chunks=count, relative_errors=errors, seconds=time.perf_counter() - started)
    (root / "inversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
