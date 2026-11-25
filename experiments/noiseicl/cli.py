"""Plan, submit and run five NoiseICL campaigns; collection input is explicit."""
import argparse
import json
from pathlib import Path

from .config import Settings, STAGES
from .dataset import index_contexts
from .planning import plan
from .reporting import report
from .submission import submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--stage", type=int, choices=STAGES, required=True)
    p.add_argument("--cluster", choices=("das6", "snellius"), default="das6")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--scenes", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hours", type=float, default=3)
    p.add_argument("--contexts", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--pairs-per-job", type=int, default=25)
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--architecture", choices=("vit", "mlp"), default="vit")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--overfit-pairs", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    s = commands.add_parser("submit")
    s.add_argument("--directory", type=Path, required=True)
    s.add_argument("--limit", type=int, default=1)
    s.add_argument("--retry", action="store_true")
    r = commands.add_parser("run")
    r.add_argument("--directory", type=Path, required=True)
    r.add_argument("--job", required=True)
    a = commands.add_parser("report")
    a.add_argument("--directory", type=Path, required=True)
    a.add_argument("--plots", action="store_true")
    i = commands.add_parser("index")
    i.add_argument("--trajectories", type=Path, required=True)
    i.add_argument("--output", type=Path, required=True)
    i.add_argument("--tasks", type=int, default=5)
    i.add_argument("--trajectories-per-task", type=int, default=20)
    i.add_argument("--points", type=int, default=10)
    commands.add_parser("schema", help="Print required recorded-trajectory JSONL fields")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            settings = Settings(stage=args.stage, cluster=args.cluster,
                scene_count=args.scenes if args.scenes is not None else (20 if args.stage in (2, 5) else 1),
                sampling_seed=args.seed, hours=args.hours, pairs_per_job=args.pairs_per_job,
                train_steps=args.train_steps, effective_batch=args.batch, architecture=args.architecture,
                width=args.width, layers=args.layers, heads=args.heads, overfit_pairs=args.overfit_pairs)
            manifest = plan(args.directory, settings, args.contexts, args.dataset)
            print(f"Stage {args.stage} ({STAGES[args.stage]}): {len(manifest['jobs'])} jobs planned. No submission.")
        elif args.command == "submit":
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            print(json.dumps(submit(args.directory, args.limit, args.retry)))
        elif args.command == "run":
            from .execution import execute
            execute(args.directory, args.job)
        elif args.command == "index":
            rows = index_contexts(args.trajectories, args.output, args.points, args.tasks, args.trajectories_per_task)
            print(f"Indexed {len(rows)} grouped history points; split assignment precedes Gaussian seeds.")
        elif args.command == "report":
            print(json.dumps(report(args.directory, args.plots), indent=2))
        else:
            print(json.dumps(dict(task="seen_task_name", trajectory_id="trajectory_001", scene_id="scene_001",
                simulator_seed=100000, demo_id="demo_001", demo_latent="/data/human.pth",
                initial="/data/trajectory/initial.pt", history="/data/trajectory/history", chunk_count=38), indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
