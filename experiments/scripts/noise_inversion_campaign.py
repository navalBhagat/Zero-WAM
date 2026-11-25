"""Plan, submit, run or summarize paired noise-inversion trials."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.noise_inversion import campaign
from experiments.noise_inversion.config import TASKS
from experiments.noise_inversion.report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Dry-run planning only; writes manifest/source snapshot")
    plan.add_argument("--cluster", choices=("snellius", "das6"), default="snellius")
    plan.add_argument("--directory", type=Path, required=True)
    plan.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    plan.add_argument("--scene-seeds", nargs="+", type=int, default=list(range(100000, 100005)))
    plan.add_argument("--sampling-seeds", nargs="+", type=int, default=[1234, 5678])
    plan.add_argument("--render-samples", type=int, default=16)
    plan.add_argument("--hours-per-job", type=float, default=3)
    plan.add_argument("--sbu-per-hour", type=float, default=128)
    plan.add_argument("--estimated-minutes", type=float, default=20)
    plan.add_argument("--max-scene-attempts", type=int, default=20,
                      help="Maximum feasibility checks per task/scene slot, shared across noises")
    submit = commands.add_parser("submit", help="Explicitly submit independent jobs, bounded by --limit")
    submit.add_argument("--directory", type=Path, required=True)
    submit.add_argument("--account", help="Optional Slurm account override; otherwise use the login default")
    submit.add_argument("--limit", type=int, default=7)
    submit.add_argument("--retry-incomplete", action="store_true")
    run = commands.add_parser("run", help="Internal job entry point")
    run.add_argument("--directory", type=Path, required=True)
    run.add_argument("--settings", type=Path, required=True)
    run.add_argument("--digest", required=True)
    run.add_argument("--retry-incomplete", action="store_true")
    run.add_argument("--port", type=int, default=int(os.environ.get("PORT", "29056")))
    analyze = commands.add_parser("report")
    analyze.add_argument("--directory", type=Path, required=True)
    analyze.add_argument("--no-plots", action="store_true", help="Write tables only; no matplotlib dependency")
    args = parser.parse_args()
    directory = args.directory.resolve()
    try:
        if args.command == "plan":
            data = campaign.plan(directory, args.tasks, args.scene_seeds, args.sampling_seeds,
                                 args.render_samples, args.hours_per_job, args.sbu_per_hour, args.estimated_minutes, args.max_scene_attempts, args.cluster)
            count = len(data["runs"])
            print(f"Planned {len(data['jobs'])} task/noise jobs: {count} paired trials / {3 * count} simulator rollouts. No jobs submitted.")
            print(f"Cluster: {args.cluster}; maximum {len(data['jobs']) * args.hours_per_job:g} GPU-hours.")
            if data["sbu_per_hour"] is not None:
                print(f"Estimated {count * args.estimated_minutes / 60 * data['sbu_per_hour']:.0f} SBUs.")
            report(directory, plots=False)
        elif args.command == "submit":
            jobs = campaign.submit(directory, args.account, args.limit, args.retry_incomplete)
            print("Submitted: " + ", ".join(jobs) if jobs else "No eligible trials to submit.")
        elif args.command == "run":
            campaign.execute(directory, args.settings, args.digest, args.retry_incomplete, args.port)
        else:
            report(directory, plots=not args.no_plots)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
