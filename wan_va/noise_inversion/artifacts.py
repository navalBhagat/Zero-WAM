"""Chunk tensors live on disk, never in a growing GPU-side episode buffer."""
from pathlib import Path

import torch


class ChunkStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, index: int, **tensors: torch.Tensor) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"chunk_{index:04d}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save({name: value.detach().cpu() for name, value in tensors.items()}, path)

    def load(self, index: int) -> dict[str, torch.Tensor]:
        return torch.load(self.directory / f"chunk_{index:04d}.pt", map_location="cpu", weights_only=True)

    def count(self) -> int:
        paths = sorted(self.directory.glob("chunk_*.pt"))
        expected = [f"chunk_{index:04d}.pt" for index in range(len(paths))]
        if [path.name for path in paths] != expected:
            raise ValueError("Chunk sequence is not contiguous")
        return len(paths)
