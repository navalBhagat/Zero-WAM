"""Validate campaign inputs before snapshots, filesystem writes or Slurm calls."""

from argparse import ArgumentParser, Namespace
import math


def validate_options(argument_parser: ArgumentParser, options: Namespace) -> None:
    """Apply stage-specific defaults and preserve the existing CLI constraints."""
    options.time = options.time or (
        "01:00:00" if options.stage == "render" else "03:00:00"
    )
    options.client_timeout = options.client_timeout or (
        2400 if options.stage == "render" else 9000
    )
    if options.distance_increment is not None and (
        not math.isfinite(options.distance_increment) or options.distance_increment <= 0
    ):
        argument_parser.error("--distance-increment must be finite and positive")
    if (
        options.distance_smoothing < 1
        or options.distance_smoothing % 2 != 1
        or options.max_lookahead_frames < 1
    ):
        argument_parser.error(
            "Distance smoothing must be positive odd; maximum lookahead must be positive"
        )
    if options.guidance_every < 1:
        argument_parser.error("--guidance-every must be positive")
    if options.sampling_seed < 0:
        argument_parser.error("Sampling seed must be nonnegative")
    if (
        min(options.episodes, options.repeats, options.window, options.client_timeout)
        < 1
        or options.lookahead < 0
    ):
        argument_parser.error("Invalid counts/window/frame settings")
    if options.fixed_frame != "last":
        try:
            if int(options.fixed_frame) < 0:
                raise ValueError
        except ValueError:
            argument_parser.error("--fixed-frame must be last or a nonnegative integer")
    if (
        min(
            options.minutes_per_episode,
            options.sbu_per_hour,
            options.budget_sbu,
            options.strength,
        )
        <= 0
        or options.startup_minutes < 0
    ):
        argument_parser.error("Invalid timing, budget, or guidance strength")
    if options.manifest.exists():
        argument_parser.error(
            "Use a new manifest path; existing manifests are never overwritten"
        )
    if not options.job_script.is_file():
        argument_parser.error(f"Job script missing: {options.job_script}")
