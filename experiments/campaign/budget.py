"""Planning estimates in SBUs; actual billing uses allocated elapsed time."""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetEstimate:
    episodes: int
    estimated_sbu: float
    wall_hours: float
    reservation_ceiling_sbu: float


def estimated_gpu_hours(
    episodes: int, minutes: float, jobs: int, startup_minutes: float
) -> float:
    """Include per-job setup separately from per-episode execution."""
    return (episodes * minutes + jobs * startup_minutes) / 60


def plan_budget(options, specs, argument_parser) -> BudgetEstimate:
    """Reject impossible wall limits before reserving GPU time."""
    count = sum(int(r["environment"]["TEST_NUM"]) for r in specs)
    estimated = (
        estimated_gpu_hours(
            count, options.minutes_per_episode, len(specs), options.startup_minutes
        )
        * options.sbu_per_hour
    )
    try:
        parts = [int(v) for v in options.time.split(":")]
        if len(parts) != 3 or min(parts) < 0 or parts[1] >= 60 or parts[2] >= 60:
            raise ValueError
        wall_hours = parts[0] + parts[1] / 60 + parts[2] / 3600
        if wall_hours <= 0:
            raise ValueError
    except ValueError:
        argument_parser.error("--time must be positive HH:MM:SS")
    if (
        max(int(s["environment"]["TEST_NUM"]) for s in specs)
        * options.minutes_per_episode
        + options.startup_minutes
        > wall_hours * 60
    ):
        argument_parser.error(
            "Estimated per-job runtime exceeds --time; reduce episodes or increase the limit"
        )
    reservation_ceiling = len(specs) * wall_hours * options.sbu_per_hour
    return BudgetEstimate(count, estimated, wall_hours, reservation_ceiling)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--budget", type=float, default=12202 + 52 / 60)
    argument_parser.add_argument("--episodes", type=int, default=9)
    argument_parser.add_argument("--minutes", type=float, default=10)
    argument_parser.add_argument("--jobs", type=int, default=9)
    argument_parser.add_argument("--startup-minutes", type=float, default=5)
    argument_parser.add_argument("--rate", type=float, default=128)
    options = argument_parser.parse_args()
    if (
        min(options.budget, options.rate) <= 0
        or min(options.episodes, options.jobs, options.minutes, options.startup_minutes)
        < 0
    ):
        argument_parser.error("Use positive budget/rate and nonnegative counts/time")
    hours = estimated_gpu_hours(
        options.episodes, options.minutes, options.jobs, options.startup_minutes
    )
    cost = hours * options.rate
    print(
        f"Available: {options.budget:.2f} SBUs = {options.budget / options.rate:.2f} A100 GPU-hours"
    )
    print(
        f"Estimate: {hours:.2f} GPU-hours = {cost:.2f} SBUs ({cost / options.budget:.1%})"
    )
    print(f"Remaining after estimate: {options.budget - cost:.2f} SBUs")
    print(
        "Planning only. Actual billing uses allocated wall time, including setup and failures."
    )
