"""Slurm command construction and environment isolation. Submission is explicit."""

import json
import os
from pathlib import Path
import subprocess
from .config import CONTROLLED


def save_manifest(path: Path, manifest: dict) -> None:
    """Persist every state transition, including before the first submission."""
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_command(
    options, run: dict, previous_job: str | None, root: Path
) -> list[str]:
    """Use afterany for sequential campaigns; parallel runs have no dependency."""
    command = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--partition={options.partition}",
        f"--time={options.time}",
        "--output=" + str(options.manifest.resolve().parent / "slurm_%j.out"),
        f"--job-name=zw_{run['task']}_{run['label']}",
        f"--chdir={root}",
    ]
    if options.account:
        command.append(f"--account={options.account}")
    if previous_job and not options.parallel:
        command.append(f"--dependency=afterany:{previous_job}")
    command.append(str(options.job_script.resolve()))
    return command


def submit_run(command: list[str], environment: dict[str, str]) -> str:
    """Discard inherited experiment switches before exporting this run's settings."""
    child = {key: value for key, value in os.environ.items() if key not in CONTROLLED}
    child.update(environment)
    return subprocess.check_output(command, text=True, env=child).strip().split(";")[0]
