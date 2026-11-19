"""Captured-step strength/precision replay without a WAM or simulator.

The target RGB and observed first frame remain fixed. Reward gain is a local
stop-gradient surrogate, not evidence of action improvement or task success.
"""

import argparse
import json
import math
from pathlib import Path
import time

from experiments.io import write_csv
from .artifacts import CapturedStep, load_capture
from .reference import load_reward, prepare_reward_reference
from .runtime import guidance_modules


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Completed stage-3 run containing a saved prediction",
    )
    argument_parser.add_argument("--model", required=True)
    argument_parser.add_argument("--dino", required=True)
    argument_parser.add_argument(
        "--reference",
        type=Path,
        help="Override cached human latent path after moving results",
    )
    argument_parser.add_argument(
        "--frames", help="Override comma-separated zero-based reference RGB indices"
    )
    argument_parser.add_argument(
        "--strengths", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0]
    )
    argument_parser.add_argument(
        "--decoder-dtypes",
        nargs="+",
        choices=["float32", "float16"],
        default=["float32", "float16"],
    )
    argument_parser.add_argument("--repeats", type=int, default=3)
    argument_parser.add_argument("--output", type=Path, required=True)
    return argument_parser


def make_reward_function(reward, observed):
    """Match runtime masking: reward decoding cannot change the observed frame."""

    def reward_function(clean):
        if observed is not None:
            clean = clean.clone()
            clean[:, :, :1] = observed
        return reward(clean)

    return reward_function


def measure_strength(
    capture: CapturedStep,
    reward_function,
    strength: float,
    decoder_dtype: str,
    selected: list[int],
    repeat: int,
) -> dict:
    """Measure exactly one correction and then evaluate the same frozen target.

    Zero strength bypasses backward; its gradient is unmeasured, not reported as
    zero. Failures remain rows so a sweep cannot hide unstable strengths.
    """
    import torch
    from importlib import import_module

    _, _, reference = guidance_modules()
    guided_step = import_module(reference.__package__ + ".latent_guidance").guided_step
    step = capture.step
    clean = step["sample"] - step["sigma"] * step["velocity"]
    row = dict(
        decoder_dtype=decoder_dtype,
        strength=strength,
        repeat=repeat,
        reference_frames=selected,
        status="failed",
    )
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        corrected, stats = guided_step(
            **step, reward_fn=reward_function, strength=strength, max_norm=1.0
        )
        if strength == 0:
            stats = dict(
                reward=reward_function(clean).item(),
                gradient_norm=None,
                correction_norm=0.0,
                applied_correction_norm=0.0,
                changed_fraction=0.0,
                correction_base_percent=0.0,
                gradient_clipped=False,
                base_update_norm=(step["base_next"] - step["sample"]).norm().item(),
            )
        with torch.no_grad():
            after = reward_function(clean + (corrected - step["base_next"])).item()
        torch.cuda.synchronize()
        if not torch.isfinite(torch.tensor(after)):
            raise FloatingPointError("Non-finite post-correction reward")
        row.update(
            stats,
            reward_after=after,
            reward_gain=after - stats["reward"],
            seconds=time.perf_counter() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            status="ok",
        )
    except (FloatingPointError, RuntimeError) as error:
        row["error"] = str(error)
    return row


def save_measurements(options, rows: list[dict]) -> None:
    """Checkpoint the JSON after each measurement; CSV is written after the sweep."""
    payload = dict(
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(options).items()
        },
        definition="Only reward VAE decoder changes precision. Latents and DINO are FP32.",
        measurements=rows,
    )
    (options.output / "precision.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv=None) -> None:
    argument_parser = parser()
    options = argument_parser.parse_args(argv)
    if (
        options.repeats < 1
        or not options.strengths
        or any(not math.isfinite(value) or value < 0 for value in options.strengths)
    ):
        argument_parser.error("Use nonnegative finite strengths and positive repeats")
    if options.output.exists():
        argument_parser.error("Choose a new output directory")

    import torch

    if not torch.cuda.is_available():
        argument_parser.error("Run on a CUDA compute node with zerowam active")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        capture = load_capture(options.run, options.reference, options.frames)
    except ValueError as error:
        argument_parser.error(str(error))
    options.output.mkdir(parents=True)
    rows = []
    for decoder_dtype in options.decoder_dtypes:
        vae, reward = load_reward(options.model, options.dino, decoder_dtype)
        selected = prepare_reward_reference(reward, capture)
        reward_function = make_reward_function(reward, capture.observed)
        for strength in options.strengths:
            for repeat in range(options.repeats):
                rows.append(
                    measure_strength(
                        capture,
                        reward_function,
                        strength,
                        decoder_dtype,
                        selected,
                        repeat,
                    )
                )
                save_measurements(options, rows)
        # The closure owns the reward as well; drop it before loading another VAE.
        del reward_function, reward, vae
        torch.cuda.empty_cache()
    write_csv(options.output / "results.csv", rows)
    print(
        f"Results: {options.output}/results.csv and precision.json. Plot using experiments/analysis/plot_precision.py"
    )
