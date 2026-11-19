"""Read captures and preserve their exact target, dtype and observation semantics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from evaluation.robotwin.artifacts import find_artifact
from experiments.io import read_json


@dataclass
class CapturedStep:
    """One flow step; tensors are loaded lazily to the requested device."""

    step: dict[str, Any]
    observed: Any
    reference_rgb: Any
    reference_path: Path
    reference_indices: list[int]


def load_capture(
    run: Path, reference_override=None, frames_override=None, device="cuda"
) -> CapturedStep:
    import torch

    capture = torch.load(
        find_artifact(run, "capture_step.pt"), map_location=device, weights_only=True
    )
    config = read_json(run / "config.json")
    reference_rgb = (
        capture.get("reference_rgb")
        if not reference_override and not frames_override
        else None
    )
    reference_path = reference_override or Path(
        config["episodes"][0]["selected_reference"]["latent"] or ""
    )
    if reference_rgb is None and not reference_path.is_file():
        raise ValueError(
            "No captured reference RGB: pass --reference to the corresponding cached human latent"
        )
    indices = (
        [int(value) for value in frames_override.split(",")]
        if frames_override
        else capture.get("reference_frames")
    )
    if indices is None:
        raise ValueError(
            "Capture has no selected target metadata; pass --frames explicitly"
        )
    step = {
        key: capture[key]
        for key in ("sample", "velocity", "sigma", "base_next", "fixed_first_frame")
    }
    step = {
        key: value.float()
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in step.items()
    }
    observed = capture.get("observed")
    if observed is not None:
        observed = observed.float()
    return CapturedStep(step, observed, reference_rgb, reference_path, indices)


def observation_trace(run: Path, config: dict, episode: dict):
    """Resolve copied preview basenames; resumed episode IDs need not equal ordinals."""
    preview = (episode.get("server", {}).get("guidance") or {}).get(
        "reference_preview_dir"
    )
    ordinal = config["episodes"].index(episode)
    directory = run / "references" / f"episode_{ordinal:03d}"
    if preview and (run / "references" / Path(preview).name).is_dir():
        directory = run / "references" / Path(preview).name
    trace = directory / "selection.jsonl"
    if not trace.is_file():
        raise ValueError(f"Observation trace missing: {trace}")
    import json

    selected = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    chunks = [record["chunk"] for record in selected]
    if not chunks or chunks != sorted(set(chunks)):
        raise ValueError("Expected nonempty strictly increasing chunk IDs")
    images = [
        directory
        / Path(
            record.get("robot_preview_path") or f"robot_chunk_{record['chunk']:04d}.png"
        ).name
        for record in selected
    ]
    if any(not path.is_file() for path in images):
        raise ValueError(
            "Missing saved robot previews; copy the full references/episode_* folder"
        )
    return chunks, images
