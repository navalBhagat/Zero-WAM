"""Reference loading for offline CPU comparison and captured-step GPU replay."""

from pathlib import Path
from .runtime import guidance_modules


def load_reference(options, episode, reference_module):
    import torch

    paths = episode.get("server", {}).get("reference_paths", {}) or episode.get(
        "selected_reference", {}
    )
    kind = (episode.get("server", {}).get("guidance") or {}).get("reference_kind")
    if options.reference_video or (
        kind == "original_video" and not options.reference_latent
    ):
        import numpy as np
        import imageio.v2 as imageio
        import torch.nn.functional as F

        path = options.reference_video or Path(paths.get("video") or "")
        if not path.is_file():
            raise ValueError(f"Reference video missing: {path}; pass --reference-video")
        reader = imageio.get_reader(str(path))
        try:
            fps = float(reader.get_meta_data().get("fps", 0) or 0)
            frames = np.stack(list(reader))
        finally:
            reader.close()
        if fps > 0:
            count = max(1, int(round(len(frames) / fps * options.reference_fps)))
            indices = np.unique(
                np.clip(
                    np.floor(np.arange(count) * fps / options.reference_fps).astype(
                        int
                    ),
                    0,
                    len(frames) - 1,
                )
            )
            frames = frames[indices]
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        tensor = (
            F.interpolate(
                tensor,
                size=(options.reference_height, options.reference_width),
                mode="bilinear",
                align_corners=False,
            )
            / 255
        )
        return tensor, dict(
            kind="original_video",
            path=str(path.resolve()),
            fps=options.reference_fps,
            height=options.reference_height,
            width=options.reference_width,
            source_override=bool(options.reference_video and kind != "original_video"),
        )
    if options.reference_latent:
        kind = "reconstructed_latent"
    if kind not in (None, "reconstructed_latent"):
        raise ValueError(f"Unknown reference kind: {kind}")
    latent = options.reference_latent or Path(paths.get("latent") or "")
    model = options.model_path or Path(
        episode.get("server", {}).get("model_path") or ""
    )
    vae_path = options.vae_path or model / "vae"
    if not latent.is_file():
        raise ValueError(f"Reference latent missing: {latent}; pass --reference-latent")
    if not (vae_path / "config.json").is_file():
        raise ValueError(f"VAE missing: {vae_path}; pass --vae-path or --model-path")
    from diffusers import AutoencoderKLWan

    print(f"Loading VAE on CPU: {vae_path}", flush=True)
    vae = (
        AutoencoderKLWan.from_pretrained(
            str(vae_path), torch_dtype=torch.float32, local_files_only=True
        )
        .to("cpu")
        .eval()
    )
    print(f"Reconstructing full human reference on CPU: {latent}", flush=True)
    with torch.inference_mode():
        frames = reference_module.reconstructed_reference(vae, latent)
    del vae
    return frames, dict(
        kind="reconstructed_latent",
        path=str(latent.resolve()),
        vae_path=str(vae_path.resolve()),
    )


def load_reward(model_path: str, checkpoint: str, decoder_dtype: str):
    """Load only VAE+DINO on CUDA; persistent state and DINO stay FP32."""
    import torch
    from diffusers import AutoencoderKLWan

    _, dino, _ = guidance_modules()
    vae_path = Path(model_path) / "vae"
    if not vae_path.is_dir():
        raise FileNotFoundError(f"Missing model component 'vae' under {model_path}")
    vae = AutoencoderKLWan.from_pretrained(str(vae_path), torch_dtype=torch.float32).to(
        "cuda"
    )
    reward = dino.DinoReward(
        vae,
        checkpoint,
        torch.device("cuda"),
        decoder_dtype=getattr(torch, decoder_dtype),
    )
    return vae, reward


def prepare_reward_reference(reward, capture):
    """Prefer the exact captured RGB; decode a latent only for explicit overrides."""
    _, _, reference = guidance_modules()
    if capture.reference_rgb is not None:
        if capture.reference_rgb.ndim != 4 or len(capture.reference_rgb) != len(
            capture.reference_indices
        ):
            raise ValueError("Captured reference RGB and frame indices disagree")
        selected = list(capture.reference_indices)
        reward.set_reference(capture.reference_rgb.float())
    else:
        from importlib import import_module

        selection = import_module(reference.__package__ + ".selection")
        with reference.decoder_on_cpu(reward.vae):
            frames = reference.reconstructed_reference(
                reward.vae, str(capture.reference_path)
            )
        selected = selection.validate_frames(capture.reference_indices, len(frames))
        reward.set_reference(frames[selected])
    return selected
