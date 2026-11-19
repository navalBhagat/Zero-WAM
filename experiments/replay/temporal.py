"""CPU-only counterfactual target selection on one recorded robot trajectory.

This compares future targets, not policy success. Matching and preprocessing use
shared runtime guidance code; only the lookahead rule differs.
"""

import argparse
import json
import os
from pathlib import Path
import time
from experiments.paths import ROOT
from experiments.io import write_csv
from .runtime import guidance_modules
from .reference import load_reference
from .artifacts import observation_trace
from .temporal_plots import figures


def replay(
    reference_features,
    observation_features,
    chunks,
    window=24,
    lookahead=4,
    distance=None,
    smoothing=3,
    max_frames=24,
):
    temporal, _, _ = guidance_modules()
    fixed = temporal.WindowFrameSelector(len(reference_features), window, lookahead)
    adaptive = temporal.DistanceFrameSelector(
        reference_features, window, lookahead, distance, smoothing, max_frames
    )
    if len(chunks) != len(observation_features):
        raise ValueError("One chunk ID per observation is required")
    rows = []
    for chunk, query in zip(chunks, observation_features):
        scores = (reference_features @ query).tolist()
        frame_target = fixed.select(chunk, similarities=scores)[0]
        distance_target = adaptive.select(chunk, similarities=scores)[0]
        assert fixed.position == adaptive.position  # Only lookahead differs.
        rows.append(
            dict(
                chunk=chunk,
                matched_frame=fixed.position,
                matched_similarity=scores[fixed.position],
                frame_target=frame_target,
                distance_target=distance_target,
                **adaptive.details,
            )
        )
    return rows, adaptive


def parser() -> argparse.ArgumentParser:
    """Declare the offline replay's unchanged CLI and reference-format defaults."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--run",
        type=Path,
        default=ROOT
        / "experiments/stages/03_guidance_pilot/runs/place_object_scale_29326930",
    )
    argument_parser.add_argument(
        "--episode", type=int, default=0, help="config.json episode ID"
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/stages/03_temporal_guidance_test",
    )
    argument_parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=Path(
            os.environ.get("GUIDANCE_CHECKPOINT", ROOT / "checkpoints/dinov2-small")
        ),
    )
    argument_parser.add_argument("--reference-latent", type=Path)
    argument_parser.add_argument("--reference-video", type=Path)
    argument_parser.add_argument("--model-path", type=Path)
    argument_parser.add_argument("--vae-path", type=Path)
    argument_parser.add_argument("--reference-fps", type=float, default=12)
    argument_parser.add_argument("--reference-height", type=int, default=320)
    argument_parser.add_argument("--reference-width", type=int, default=480)
    argument_parser.add_argument("--window", type=int, default=24)
    argument_parser.add_argument("--lookahead", type=int, default=4)
    argument_parser.add_argument("--distance-increment", type=float)
    argument_parser.add_argument("--smoothing", type=int, default=3)
    argument_parser.add_argument("--max-lookahead-frames", type=int, default=24)
    argument_parser.add_argument("--threads", type=int, default=4)
    return argument_parser


def validate_options(options, argument_parser) -> None:
    """Reject invalid settings before importing model dependencies."""
    if options.reference_latent and options.reference_video:
        argument_parser.error("Choose one reference override")
    import math

    if (
        options.window < 1
        or options.lookahead < 0
        or options.max_lookahead_frames < 1
        or options.smoothing < 1
        or options.smoothing % 2 != 1
    ):
        argument_parser.error(
            "Window/max lookahead must be positive; lookahead nonnegative; smoothing positive odd"
        )
    if options.distance_increment is not None and (
        not math.isfinite(options.distance_increment) or options.distance_increment <= 0
    ):
        argument_parser.error("Distance increment must be finite and positive")
    if (
        options.threads < 1
        or options.reference_fps <= 0
        or min(options.reference_height, options.reference_width) < 1
    ):
        argument_parser.error("Invalid CPU/reference settings")
    if options.output.exists() and any(options.output.iterdir()):
        argument_parser.error("Output is not empty; choose a new --output")


def load_inputs(options, argument_parser):
    """Read one episode and locate its saved robot observations."""
    config_path = options.run / "config.json"
    if not config_path.is_file():
        argument_parser.error(f"Run config missing: {config_path}; pass --run")
    config = json.loads(config_path.read_text())
    episode = next(
        (e for e in config.get("episodes", []) if e["episode_id"] == options.episode),
        None,
    )
    if episode is None:
        argument_parser.error("Episode ID not found")
    if episode.get("task", config.get("TASK")) != "place_object_scale":
        argument_parser.error("This phase is for place_object_scale")
    try:
        override = getattr(options, "robot_observation_dir", None)
        if override is not None:
            image_paths = sorted(override.glob("robot_chunk_*.png"))
            chunks = [int(path.stem.rsplit("_", 1)[1]) for path in image_paths]
            if not image_paths or chunks != sorted(set(chunks)):
                raise ValueError(
                    "Expected nonempty uniquely numbered robot_chunk_*.png images"
                )
        else:
            chunks, image_paths = observation_trace(options.run, config, episode)
    except ValueError as error:
        argument_parser.error(str(error))
    if not (options.dino_checkpoint / "config.json").is_file():
        argument_parser.error(f"DINO checkpoint missing: {options.dino_checkpoint}")
    return episode, chunks, image_paths


def embed_observations(reference, image_paths, checkpoint, dino):
    """Embed complete reference and actual observations with runtime preprocessing."""
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoModel

    reward = dino.DinoReward.__new__(dino.DinoReward)
    reward.encoder = (
        AutoModel.from_pretrained(str(checkpoint), local_files_only=True)
        .to("cpu")
        .eval()
        .requires_grad_(False)
    )
    observations = []
    for path in image_paths:
        with Image.open(path) as image:
            observations.append(image.convert("RGB").copy())
    with torch.inference_mode():
        features = torch.cat([reward.features(batch) for batch in reference.split(4)])
        queries = torch.cat(
            [
                reward.features(
                    torch.from_numpy(np.asarray(img).copy())
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255
                )
                for img in observations
            ]
        )
    return features, queries, observations


def main(argv=None) -> None:
    """Prepare CPU inputs, compare targets, and save the compact diagnostic."""
    argument_parser = parser()
    options = argument_parser.parse_args(argv)
    validate_options(options, argument_parser)
    episode, chunks, image_paths = load_inputs(options, argument_parser)
    # Everything below runs on CPU, even on a host with visible GPUs.
    import torch

    torch.set_num_threads(options.threads)
    _, dino, reference_module = guidance_modules()
    started = time.perf_counter()
    print(f"CPU replay starting with {options.threads} threads", flush=True)
    try:
        reference, provenance = load_reference(options, episode, reference_module)
    except (ValueError, OSError) as error:
        argument_parser.error(str(error))
    print(
        f"CPU only: {len(reference)} human frames, {len(chunks)} saved robot observations.",
        flush=True,
    )
    reference_seconds = time.perf_counter() - started
    print(
        f"Reference preparation: {reference_seconds:.1f} seconds. Loading/embedding with DINO on CPU...",
        flush=True,
    )
    embedding_started = time.perf_counter()
    features, queries, observations = embed_observations(
        reference, image_paths, options.dino_checkpoint, dino
    )
    embedding_seconds = time.perf_counter() - embedding_started
    print(
        f"DINO load + embeddings: {embedding_seconds:.1f} seconds. Selecting targets and plotting...",
        flush=True,
    )
    rows, selector = replay(
        features,
        queries,
        chunks,
        options.window,
        options.lookahead,
        options.distance_increment,
        options.smoothing,
        options.max_lookahead_frames,
    )
    options.output.mkdir(parents=True, exist_ok=True)
    write_csv(options.output / "results.csv", rows)
    figures(options.output, rows, selector, reference, observations)
    provenance.update(
        run=str(options.run.resolve()),
        episode_id=options.episode,
        device="cpu",
        dino_checkpoint=str(options.dino_checkpoint.resolve()),
        window=options.window,
        frame_lookahead=options.lookahead,
        distance_increment=selector.distance,
        smoothing=options.smoothing,
        max_forward_frames=options.max_lookahead_frames,
        fallback_chunks=sum(bool(r["fallback"]) for r in rows),
        cap_or_end_limited_chunks=sum(r["horizon_limited"] for r in rows),
        timing_seconds=dict(
            reference_preparation=reference_seconds,
            dino_load_and_embedding=embedding_seconds,
            total=time.perf_counter() - started,
        ),
        interpretation="Counterfactual target selection on a recorded trajectory; does not measure policy success.",
    )
    (options.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        f"Threshold: {selector.distance:.6g}; bounded chunks: {provenance['cap_or_end_limited_chunks']}/{len(rows)}"
    )
    print(
        f"Outputs: {options.output}/results.csv, targets.png, comparison.png, provenance.json"
    )
