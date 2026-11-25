"""Residual patch transformer; frozen VAE latents supply demonstration/context."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AdapterConfig:
    channels: int = 48
    action_channels: int = 30
    patch: int = 2
    width: int = 512
    layers: int = 8
    heads: int = 8
    context_tokens: int = 256
    architecture: str = "vit"


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input = nn.Linear(width, width * 8)
        self.output = nn.Linear(width * 4, width)

    def forward(self, value):
        left, right = self.input(value).chunk(2, dim=-1)
        return self.output(F.silu(left) * right)


class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.width)
        self.feedforward = SwiGLU(config.width)

    def forward(self, noise, context):
        return noise + self.feedforward(self.norm(noise + context.mean(dim=1, keepdim=True)))


class AdapterBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.width)
        self.norm2 = nn.RMSNorm(config.width)
        self.norm3 = nn.RMSNorm(config.width)
        self.self_attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.feedforward = SwiGLU(config.width)

    def forward(self, noise, context):
        normalized = self.norm1(noise)
        noise = noise + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        noise = noise + self.cross_attention(self.norm2(noise), context, context, need_weights=False)[0]
        return noise + self.feedforward(self.norm3(noise))


class NoiseAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.width % config.heads or config.architecture not in ("vit", "mlp"):
            raise ValueError("Invalid adapter width/heads or architecture")
        size = config.channels * config.patch ** 2
        self.input = nn.Linear(size, config.width)
        self.position = nn.Linear(3, config.width, bias=False)
        self.modality = nn.Embedding(3, config.width)
        self.action = nn.Linear(config.action_channels, config.width)
        self.progress = nn.Linear(1, config.width)
        block = AdapterBlock if config.architecture == "vit" else MLPBlock
        self.blocks = nn.ModuleList(block(config) for _ in range(config.layers))
        self.output = nn.Linear(config.width, size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def tokenize(self, latent, modality):
        batch, channels, frames, height, width = latent.shape
        p = self.config.patch
        if channels != self.config.channels or height % p or width % p:
            raise ValueError(f"Invalid latent patch geometry: {latent.shape}")
        tokens = latent.reshape(batch, channels, frames, height // p, p, width // p, p)
        tokens = tokens.permute(0, 2, 3, 5, 1, 4, 6).reshape(batch, -1, channels * p * p)
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, frames, device=latent.device),
            torch.linspace(-1, 1, height // p, device=latent.device),
            torch.linspace(-1, 1, width // p, device=latent.device), indexing="ij"), dim=-1).reshape(-1, 3)
        return self.input(tokens) + self.position(grid.to(tokens)) + self.modality.weight[modality]

    def context_tokens(self, latent, modality):
        # Pool context spatially before attention; retain a bounded temporal grid.
        frames = min(latent.shape[2], 8)
        side = max(1, int(math.sqrt(self.config.context_tokens / frames)))
        p = self.config.patch
        pooled = F.adaptive_avg_pool3d(latent.float(), (frames, side * p, side * p))
        return self.tokenize(pooled.to(latent), modality)

    def forward(self, noise, demo, robot, actions, chunk=0):
        original_dtype = noise.dtype
        dtype = self.input.weight.dtype
        latent = noise.to(dtype)
        tokens = self.tokenize(latent, 0)
        context = torch.cat((self.context_tokens(demo.to(dtype), 1),
                             self.context_tokens(robot.to(dtype), 2)), dim=1)
        action_tokens = actions.to(dtype).mean(dim=(-1, -2)).transpose(1, 2)
        context = torch.cat((context, self.action(action_tokens)), dim=1)
        progress = torch.as_tensor(chunk, device=noise.device, dtype=dtype).reshape(-1, 1, 1)
        tokens = tokens + self.progress(progress / 100)
        for block in self.blocks:
            tokens = block(tokens, context)
        batch, channels, frames, height, width = noise.shape
        p = self.config.patch
        residual = self.output(tokens).reshape(batch, frames, height // p, width // p, channels, p, p)
        residual = residual.permute(0, 4, 1, 2, 5, 3, 6).reshape_as(noise)
        return noise + residual.to(original_dtype)


def load_adapter(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = NoiseAdapter(AdapterConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
