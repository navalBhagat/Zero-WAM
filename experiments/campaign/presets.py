"""Named, reproducible experiment presets. Commands remain dry-run by default."""

import argparse
from pathlib import Path
import subprocess
import sys
from experiments.paths import ROOT


def command(manifest, submit=False, account=None, budget=None, strength=0.1):
    cmd = [
        sys.executable,
        str(ROOT / "experiments/scripts/robotwin_suite.py"),
        "guidance",
        "--tasks",
        "place_object_scale",
        "--guidance-modes",
        "guidance",
        "--guidance-selectors",
        "window",
        "--seeds",
        "0",
        "--sampling-seed",
        "1234",
        "--episodes",
        "1",
        "--spp",
        "8",
        "--phase-separation",
        "--compact-cache",
        "--window",
        "24",
        "--lookahead",
        "4",
        "--lookahead-mode",
        "dino_distance",
        "--distance-smoothing",
        "3",
        "--max-lookahead-frames",
        "24",
        "--strength",
        str(strength),
        "--guidance-every",
        "10",
        "--reference-source",
        "latent",
        "--partition",
        "gpu_a100",
        "--time",
        "01:00:00",
        "--client-timeout",
        "3000",
        "--minutes-per-episode",
        "20",
        "--manifest",
        str(manifest),
    ]
    if submit:
        cmd.append("--submit")
    if account:
        cmd += ["--account", account]
    if budget is not None:
        cmd += ["--budget-sbu", str(budget)]
    return cmd


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--submit", action="store_true")
    argument_parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/stages/03_guidance_strength_010/manifest.json",
    )
    argument_parser.add_argument("--strength", type=float, default=0.1)
    argument_parser.add_argument("--account")
    argument_parser.add_argument("--budget-sbu", type=float)
    options = argument_parser.parse_args()
    print(
        "One A100 episode: scale, seed group 0 (expected simulator seed 100000), noise seed 1234.",
        flush=True,
    )
    print(
        "Compare with distance pilot job 29419404 (strength 0.01); verify actual simulator seed and reference path match in config.json.",
        flush=True,
    )
    subprocess.run(
        command(
            options.manifest,
            options.submit,
            options.account,
            options.budget_sbu,
            strength=options.strength,
        ),
        cwd=ROOT,
        check=True,
    )
