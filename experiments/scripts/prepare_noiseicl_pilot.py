"""Prepare a variable-size five-task pilot from HumanGen or recorded histories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.noiseicl.config import Settings, UNSEEN_TASKS
from wan_va.noise_inversion.dataset import index_contexts, read_lines
from experiments.noiseicl.planning import plan
from experiments.runtime.records import save_json


DEFAULT_TASKS = ("adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
                 "blocks_ranking_size", "click_alarmclock")


def prepare(trajectories, output, campaign, cluster, tasks=DEFAULT_TASKS, contexts_per_task=200):
    if len(tasks) != 5 or len(set(tasks)) != 5 or set(tasks) & set(UNSEEN_TASKS):
        raise ValueError("Pilot requires five distinct seen tasks")
    if contexts_per_task < 20:
        raise ValueError("Need at least 20 contexts per task to cover all human demos")
    source = Path(trajectories).resolve()
    rows = read_lines(source)
    selected = []
    for task in tasks:
        candidates = sorted((dict(r) for r in rows if r["task"] == task),
                            key=lambda r: str(r["trajectory_id"]))
        demos = {}
        for row in candidates:
            for key in ("demo_latent", "initial", "history"):
                row[key] = str((source.parent / Path(row[key]).expanduser()).resolve())
            if int(row["chunk_count"]) < 1:
                raise ValueError(f"Empty recorded trajectory: {task}/{row['trajectory_id']}")
            if not all(Path(row[key]).exists() for key in ("demo_latent", "initial", "history")):
                raise FileNotFoundError(f"Missing recorded assets for {task}/{row['trajectory_id']}")
            history = sorted(Path(row["history"]).glob("chunk_*.pt"))
            if (len(history) < int(row["chunk_count"]) - 1
                    or [p.name for p in history] != [f"chunk_{i:04d}.pt" for i in range(len(history))]):
                raise ValueError(f"Incomplete recorded history for {task}/{row['trajectory_id']}")
            if str(row["demo_id"]) in demos and demos[str(row["demo_id"])] != row["demo_latent"]:
                raise ValueError("One demo ID refers to multiple human latent files")
            demos[str(row["demo_id"])] = row["demo_latent"]
        if len(demos) != 20 or len(set(demos.values())) != 20:
            raise ValueError(f"{task}: need 20 distinct paired demonstrations; found {len(set(demos.values()))}")
        selected.extend(candidates)
    output, campaign = Path(output).resolve(), Path(campaign).resolve()
    if output.exists() or campaign.exists():
        raise FileExistsError("Choose fresh dataset and campaign directories")
    output.mkdir(parents=True)
    manifest = output / "trajectories.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
    contexts = index_contexts(manifest, output, points=max(int(r["chunk_count"]) for r in selected),
                              task_limit=5, trajectories_per_task=len(selected),
                              task_context_limit=contexts_per_task)
    counts = Counter(c["task"] for c in contexts)
    result = plan(campaign, Settings(stage=4, cluster=cluster, pairs_per_job=25), output / "index.json")
    summary = dict(target_contexts_per_task=contexts_per_task, seeds_per_context=2,
                   contexts=len(contexts), pairs=2 * len(contexts), jobs=len(result["jobs"]),
                   tasks={task: dict(contexts=counts[task], pairs=2 * counts[task],
                                    shortfall=max(0, contexts_per_task - counts[task])) for task in tasks})
    save_json(campaign / "pilot.json", summary)
    print(json.dumps(summary, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--trajectories", type=Path)
    sources.add_argument("--humangen-root", type=Path, help="Downloaded HumanGen root; convert latents/actions first")
    parser.add_argument("--histories", type=Path, default=Path("experiments/datasets/seen_pilot_histories"))
    parser.add_argument("--output", type=Path, default=Path("experiments/datasets/seen_pilot"))
    parser.add_argument("--directory", type=Path, default=Path("experiments/stages/04_collect_pilot"))
    parser.add_argument("--cluster", choices=("das6", "snellius"), default="das6")
    parser.add_argument("--tasks", nargs=5, default=DEFAULT_TASKS)
    parser.add_argument("--contexts-per-task", type=int, default=200)
    args = parser.parse_args()
    try:
        if len(set(args.tasks)) != 5 or set(args.tasks) & set(UNSEEN_TASKS):
            raise ValueError("Pilot requires five distinct seen tasks")
        if args.output.exists() or args.directory.exists():
            raise FileExistsError("Choose fresh dataset and campaign directories")
        source = args.trajectories
        if args.humangen_root:
            from wan_va.noise_inversion.humangen import convert
            source = convert(args.humangen_root, args.histories, args.tasks)
        result = prepare(source, args.output, args.directory, args.cluster, args.tasks, args.contexts_per_task)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))
    pairs = sum(len(job["pairs"]) for job in result["jobs"])
    print(f"Prepared {pairs} pairs / {len(result['jobs'])} jobs. No submission.")


if __name__ == "__main__":
    main()
