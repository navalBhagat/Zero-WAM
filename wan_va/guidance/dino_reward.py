"""Frozen local DINOv2 reward on the external camera's final decoded frame."""
import torch
import torch.nn.functional as F

from .latent_guidance import isolated_decode_cache


class DinoReward:
    def __init__(self, vae, checkpoint, device, camera_count=3):
        from transformers import AutoModel

        self.vae = vae.eval().requires_grad_(False)
        self.encoder = AutoModel.from_pretrained(
            checkpoint, local_files_only=True
        ).to(device).eval().requires_grad_(False)
        self.device = device
        self.camera_count = camera_count
        self.reference = None

    def features(self, frames):
        # Differentiable preprocessing; frames are RGB in [0, 1].
        frames = F.interpolate(frames.float(), (224, 224), mode="bicubic",
                               align_corners=False).clamp(0, 1)
        mean = frames.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = frames.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        features = self.encoder(pixel_values=(frames - mean) / std).last_hidden_state[:, 0]
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def set_reference(self, frames):
        """frames: selected human segment, [T, C, H, W], RGB [0, 1]."""
        if not len(frames):
            raise ValueError("External guidance reference segment is empty")
        # One reference embedding; explicitly no temporal alignment in this prototype.
        self.reference = F.normalize(self.features(frames.to(self.device)).mean(0), dim=0)

    def decode(self, clean):
        if clean.shape[-1] % self.camera_count:
            raise ValueError("Expected equal camera latents concatenated along width")
        camera = clean.chunk(self.camera_count, dim=-1)[0]
        parameter = next(self.vae.parameters())
        camera = camera.to(device=parameter.device, dtype=parameter.dtype)
        mean = camera.new_tensor(self.vae.config.latents_mean)[None, :, None, None, None]
        std = camera.new_tensor(self.vae.config.latents_std)[None, :, None, None, None]
        with isolated_decode_cache(self.vae):
            video = self.vae.decode(camera * std + mean, return_dict=False)[0]
        return video

    def __call__(self, clean):
        if self.reference is None:
            raise RuntimeError("Set the human reference before sampling")
        video = self.decode(clean)
        frames = (video[:, :, -1].to(self.device).float() + 1) / 2
        return (self.features(frames) * self.reference).sum(-1).mean()
