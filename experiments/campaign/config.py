"""Campaign CLI and defaults. No model imports, filesystem writes or submission."""

import argparse
from pathlib import Path
from experiments.paths import ROOT

MODES = {
    "text": (0, 1, 0),
    "icl": (1, 0, 0),
    "guidance": (0, 0, 1),
    "icl_guidance": (1, 0, 1),
}
# Do not let an earlier interactive experiment silently contaminate a new stage.
CONTROLLED = (
    "COUNT_POLICY_FLOPS",
    "RUN_DIAGNOSTICS",
    "SAVE_GUIDANCE_CAPTURE",
    "START_SEED",
    "EPISODE_OFFSET",
    "COMPACT_OBSERVATION_CACHE",
    "SAVE_ROLLOUT_SNAPSHOTS",
    "PHASE_COMPARE_ACTIONS",
    "PHASE_SEPARATION",
    "PHASE_OFFLOAD_TRANSFORMER",
    "STAGE_REQUIRE_SUCCESS",
    "GUIDANCE_CAPTURE",
    "GUIDANCE_SELECTION",
    "GUIDANCE_REFERENCE_FRAMES",
    "GUIDANCE_LOOKAHEAD_MODE",
    "GUIDANCE_DISTANCE_INCREMENT",
    "GUIDANCE_DISTANCE_SMOOTHING",
    "GUIDANCE_MAX_LOOKAHEAD_FRAMES",
    "GUIDANCE_MATCH_WINDOW",
    "GUIDANCE_LOOKAHEAD",
    "GUIDANCE_STRENGTH",
    "GUIDANCE_EVERY",
    "GUIDANCE_MAX_NORM",
    "GUIDANCE_REFERENCE_SOURCE",
    "GUIDANCE_DECODER_DTYPE",
    "GUIDANCE_STATE_DTYPE",
    "VIDEO_STATE_FP32",
    "USE_ICL",
    "USE_TEXT",
    "USE_GUIDANCE",
    "TEXT_GUIDANCE_SCALE",
    "ICL_GUIDANCE_SCALE",
    "TARGET_TEXT_SOURCE",
    "ICL_TEXT_SOURCE",
    "SEED",
    "SAMPLING_SEED",
    "ICL_SEED",
)


def parser() -> argparse.ArgumentParser:
    """Declare stable CLI flags grouped by the decisions a user needs to make."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    task_options = argument_parser.add_argument_group("Tasks and seeds")
    guidance_options = argument_parser.add_argument_group("Guidance and references")
    recording_options = argument_parser.add_argument_group("Output and profiling")
    runtime_options = argument_parser.add_argument_group("Memory controls")
    slurm_options = argument_parser.add_argument_group("Slurm resources and budget")

    argument_parser.add_argument(
        "stage", choices=["render", "baselines", "guidance", "precision"]
    )
    task_options.add_argument(
        "--tasks", nargs="+", choices=["place_object_scale", "place_bread_basket"]
    )
    guidance_options.add_argument(
        "--guidance-selectors", nargs="+", choices=["fixed", "window"]
    )
    task_options.add_argument("--sampling-seed", type=int, default=1234)
    guidance_options.add_argument(
        "--guidance-modes",
        nargs="+",
        choices=["guidance", "icl_guidance"],
        help="Guidance-stage conditions; defaults to both",
    )
    runtime_options.add_argument(
        "--compact-cache", action="store_true", help="Opt in after stage-00 validation"
    )
    task_options.add_argument("--seeds", nargs="+", type=int)
    task_options.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Desired total per task/mode/seed, including reused episodes",
    )
    task_options.add_argument(
        "--resume-from",
        type=Path,
        help="Reuse completed baseline episodes from this prior manifest",
    )
    task_options.add_argument("--repeats", type=int, default=1)
    task_options.add_argument("--spp", type=int, choices=[8, 16, 32], default=32)
    guidance_options.add_argument("--window", type=int, default=24)
    guidance_options.add_argument("--lookahead", type=int, default=4)
    guidance_options.add_argument(
        "--lookahead-mode", choices=["frames", "dino_distance"], default="frames"
    )
    guidance_options.add_argument(
        "--distance-increment",
        type=float,
        help="Positive cumulative cosine distance; omitted: median reference change over --lookahead frames",
    )
    guidance_options.add_argument(
        "--distance-smoothing",
        type=int,
        default=3,
        help="Odd reference embedding smoothing width",
    )
    guidance_options.add_argument("--max-lookahead-frames", type=int, default=24)
    guidance_options.add_argument(
        "--fixed-frame", default="last", help="Zero-based RGB index or last"
    )
    guidance_options.add_argument("--strength", type=float, default=0.01)
    guidance_options.add_argument("--guidance-every", type=int, default=10)
    recording_options.add_argument(
        "--diagnostics",
        action="store_true",
        help="Save package inventories, action hashes and cache/offload traces",
    )
    recording_options.add_argument(
        "--capture-guidance",
        action="store_true",
        help="Save one prediction for gradient/strength replay",
    )
    recording_options.add_argument(
        "--no-flop-count",
        action="store_true",
        default=True,
        help="Disable operator accounting; record missing FLOPs rather than zeros",
    )
    recording_options.add_argument(
        "--flop-count",
        dest="no_flop_count",
        action="store_false",
        help="Opt in to slow runtime operator accounting for calibration",
    )
    guidance_options.add_argument(
        "--reference-source", choices=["auto", "video", "latent"], default="auto"
    )
    slurm_options.add_argument("--partition", default="gpu_a100")
    slurm_options.add_argument(
        "--time", default=None, help="Wall limit per job; raise for multi-episode runs"
    )
    slurm_options.add_argument("--client-timeout", type=int, default=None)
    slurm_options.add_argument("--account")
    slurm_options.add_argument(
        "--job-script", type=Path, default=ROOT / "experiments/jobs/eval.sbatch"
    )
    recording_options.add_argument("--manifest", type=Path, required=True)
    slurm_options.add_argument("--minutes-per-episode", type=float, default=10)
    slurm_options.add_argument(
        "--startup-minutes",
        type=float,
        default=5,
        help="Planning allowance, not measured",
    )
    slurm_options.add_argument("--sbu-per-hour", type=float, default=128)
    slurm_options.add_argument("--budget-sbu", type=float, default=12202 + 52 / 60)
    slurm_options.add_argument("--submit", action="store_true")
    slurm_options.add_argument(
        "--parallel",
        action="store_true",
        help="Submit independent jobs without dependencies; one GPU per job",
    )
    runtime_options.add_argument(
        "--phase-separation",
        action="store_true",
        help="Opt into single-GPU observation-encoding offload after stage-00 validation",
    )
    return argument_parser
