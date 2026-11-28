"""Prepare the research-plan pilot from recorded histories; submit separately."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.noiseicl.config import Settings, UNSEEN_TASKS
from experiments.noiseicl.dataset import index_contexts, read_lines
from experiments.noiseicl.planning import plan


DEFAULT_TASKS = ("adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
                 "blocks_ranking_size", "click_alarmclock")


def prepare(trajectories, output, campaign, cluster, tasks=DEFAULT_TASKS):
    if len(tasks) != 5 or len(set(tasks)) != 5 or set(tasks) & set(UNSEEN_TASKS):
        raise ValueError("Pilot requires five distinct seen tasks")
    source = Path(trajectories).resolve()
    rows = read_lines(source)
    selected = []
    for task in tasks:
        candidates = sorted((r for r in rows if r["task"] == task),
                            key=lambda r: str(r["trajectory_id"]))
        demos, latents, trajectories_seen = set(), set(), set()
        for row in candidates:
            row = dict(row)
            for key in ("demo_latent", "initial", "history"):
                row[key] = str((source.parent / Path(row[key]).expanduser()).resolve())
            if (row["demo_id"] in demos or row["demo_latent"] in latents
                    or row["trajectory_id"] in trajectories_seen or int(row["chunk_count"]) < 10):
                continue
            if not all(Path(row[key]).exists() for key in ("demo_latent", "initial", "history")):
                raise FileNotFoundError(f"Missing recorded assets for {task}/{row['trajectory_id']}")
            history = sorted(Path(row["history"]).glob("chunk_*.pt"))
            if (len(history) < int(row["chunk_count"]) - 1
                    or [p.name for p in history] != [f"chunk_{i:04d}.pt" for i in range(len(history))]):
                raise ValueError(f"Incomplete recorded history for {task}/{row['trajectory_id']}")
            demos.add(row["demo_id"])
            latents.add(row["demo_latent"])
            trajectories_seen.add(row["trajectory_id"])
            selected.append(row)
            if len(demos) == 20:
                break
        if len(demos) != 20:
            raise ValueError(f"{task}: need 20 distinct paired demonstrations with >=10 chunks; found {len(demos)}")
    output, campaign = Path(output).resolve(), Path(campaign).resolve()
    if output.exists() or campaign.exists():
        raise FileExistsError("Choose fresh dataset and campaign directories")
    output.mkdir(parents=True)
    manifest = output / "trajectories.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
    contexts = index_contexts(manifest, output, points=10, task_limit=5, trajectories_per_task=20)
    assert len(contexts) == 1000
    result = plan(campaign, Settings(stage=4, cluster=cluster, pairs_per_job=25), output / "index.json")
    assert sum(len(job["pairs"]) for job in result["jobs"]) == 2000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/datasets/seen_pilot"))
    parser.add_argument("--directory", type=Path, default=Path("experiments/stages/04_collect_pilot"))
    parser.add_argument("--cluster", choices=("das6", "snellius"), required=True)
    parser.add_argument("--tasks", nargs=5, default=DEFAULT_TASKS)
    args = parser.parse_args()
    try:
        result = prepare(args.trajectories, args.output, args.directory, args.cluster, args.tasks)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))
    print(f"Prepared 5 tasks / 100 demonstrations / 1000 contexts / 2000 pairs / {len(result['jobs'])} jobs. No submission.")


if __name__ == "__main__":
    main()
