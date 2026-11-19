"""Campaign orchestration: validate, plan, snapshot, then optionally submit."""

import os
from pathlib import Path
import shlex
import subprocess
from experiments.paths import ROOT
from .config import parser
from .plans import specifications
from .resume import extend_baselines
from .provenance import snapshot
from .validation import validate_options
from .budget import plan_budget
from .slurm import build_command, save_manifest, submit_run


def main() -> None:
    argument_parser = parser()
    options = argument_parser.parse_args()
    validate_options(argument_parser, options)
    specs = specifications(options)
    reused = []
    if options.resume_from:
        if options.stage != "baselines":
            argument_parser.error("--resume-from currently supports baselines only")
        try:
            specs, reused = extend_baselines(specs, options.resume_from, ROOT)
        except (ValueError, OSError, KeyError) as error:
            argument_parser.error(str(error))
        print(
            f"Reusing {len(reused)} prior run segments; only remaining episodes will be submitted."
        )
        if not specs:
            print("All requested episodes are already complete; nothing submitted.")
            return
    budget = plan_budget(options, specs, argument_parser)
    count = budget.episodes
    estimated = budget.estimated_sbu
    reservation_ceiling = budget.reservation_ceiling_sbu
    manifest = dict(
        schema_version=1,
        stage=options.stage,
        submitted=options.submit,
        parallel=options.parallel,
        runs=reused + specs,
        resume_from=str(options.resume_from.resolve()) if options.resume_from else None,
        estimate=dict(
            episodes=count,
            sbu=estimated,
            budget_sbu=options.budget_sbu,
            sbu_per_hour=options.sbu_per_hour,
            startup_minutes=options.startup_minutes,
            requested_walltime_sbu_ceiling=reservation_ceiling,
            note="Guidance may cost more; estimate assumes supplied episode time.",
        ),
    )
    print(
        f"{len(specs)} jobs / {count} episodes; estimated {estimated:.1f} SBUs "
        f"({estimated / options.budget_sbu:.1%} of budget); includes assumed startup."
    )
    print(
        f"Requested wall-time ceiling: {reservation_ceiling:.1f} SBUs "
        "(submission may reserve this amount; actual charge uses elapsed allocation)."
    )
    if reservation_ceiling > options.budget_sbu:
        argument_parser.error(
            "Requested wall-time ceiling exceeds --budget-sbu; reduce jobs or --time"
        )
    if estimated > options.budget_sbu:
        argument_parser.error("Estimate exceeds --budget-sbu; reduce the experiment")
    if options.submit and options.stage in ("guidance", "precision"):
        checkpoint = Path(
            os.environ.get("GUIDANCE_CHECKPOINT", ROOT / "checkpoints/dinov2-small")
        )
        if not (checkpoint / "config.json").is_file():
            argument_parser.error(
                "Set GUIDANCE_CHECKPOINT to the local DINOv2 model directory"
            )
    options.manifest.parent.mkdir(parents=True, exist_ok=True)
    bundle = options.manifest.with_suffix(".source.zip")
    if bundle.exists():
        argument_parser.error(f"Source bundle already exists: {bundle}")
    digest = snapshot(bundle)
    manifest["source_bundle"] = str(bundle.resolve())
    manifest["source_digest"] = digest
    for spec in specs:
        spec["environment"]["EXPERIMENT_SOURCE_DIGEST"] = digest

    # Save complete planned set before submitting anything. Preserve provenance.
    def save():
        save_manifest(options.manifest, manifest)

    manifest["revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest["dirty_diff"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff"], text=True
    )
    save()
    previous_job = None
    for spec in specs:
        command = build_command(options, spec, previous_job, ROOT)
        spec["command"] = command
        print(
            " ".join(f"{k}={shlex.quote(v)}" for k, v in spec["environment"].items()),
            shlex.join(command),
        )
        if options.submit:
            try:
                spec["job_id"] = submit_run(command, spec["environment"])
                spec["status"] = "submitted"
                previous_job = spec["job_id"]
            except (OSError, subprocess.CalledProcessError) as error:
                spec["status"] = "submission_failed"
                spec["error"] = str(error)
                save()
                raise
        save()
    print(
        f"Manifest: {options.manifest}; "
        + (
            (
                "submitted independently"
                if options.parallel
                else "submitted sequentially"
            )
            if options.submit
            else "DRY RUN: nothing submitted"
        )
    )
