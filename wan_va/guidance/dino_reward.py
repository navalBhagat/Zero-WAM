"""Frozen local DINOv2 reward on the external camera's final decoded frame."""
import torch
import torch.nn.functional as F

from .reference import decode_normalized
from .selection import validate_frames


class DinoReward:
    def __init__(self, vae, checkpoint, device, camera_count=3, decoder_dtype=None):
        from transformers import AutoModel

        self.vae = vae.eval().requires_grad_(False)
        if decoder_dtype is not None:
            # Encoder precision stays unchanged for streaming observations.
            self.vae.decoder.to(dtype=decoder_dtype)
            self.vae.post_quant_conv.to(dtype=decoder_dtype)
        self.encoder = AutoModel.from_pretrained(
            checkpoint, local_files_only=True
        ).to(device).eval().requires_grad_(False)
        self.device = device
        self.camera_count = camera_count
        self.reference = None
        self.reference_features = None
        self.reference_images = None
        self.selected_frames = None

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

    @torch.no_grad()
    def prepare_references(self, frames):
        """Cache one feature per human RGB frame, in small GPU batches."""
        if not len(frames):
            raise ValueError("Human reference sequence is empty")
        self.reference_images = frames.detach().cpu()
        self.reference_features = torch.cat([
            self.features(batch.to(self.device))
            for batch in self.reference_images.split(4)
        ])
        self.reference = None
        self.selected_frames = None

    def select_reference(self, frame_indices):
        """Choose a fixed target for the next chunk; multiple frames are averaged."""
        if self.reference_features is None:
            raise RuntimeError("Prepare the human reference sequence first")
        indices = validate_frames(frame_indices, len(self.reference_features))
        self.reference = F.normalize(self.reference_features[indices].mean(0), dim=0)
        self.selected_frames = indices

    def save_reference_frames(self, output):
        """Write each selected RGB frame once, for later visual inspection."""
        from pathlib import Path
        from PIL import Image
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in self.selected_frames:
            path = output / f'frame_{index:04d}.png'
            if not path.exists():
                rgb = (self.reference_images[index] * 255).clamp(0, 255).byte()
                Image.fromarray(rgb.permute(1, 2, 0).numpy()).save(path)
            paths.append(str(path.resolve()))
        return paths

    def decode(self, clean):
        if clean.shape[-1] % self.camera_count:
            raise ValueError("Expected equal camera latents concatenated along width")
        camera = clean.chunk(self.camera_count, dim=-1)[0]
        return decode_normalized(self.vae, camera)

    def __call__(self, clean):
        if self.reference is None:
            raise RuntimeError("Set the human reference before sampling")
        video = self.decode(clean)
        frames = (video[:, :, -1].to(self.device).float() + 1) / 2
        return (self.features(frames) * self.reference).sum(-1).mean()
