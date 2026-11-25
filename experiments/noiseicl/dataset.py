"""Grouped trajectory indexing; seeds/chunks can never cross split boundaries."""
import hashlib
import json
from pathlib import Path

from experiments.runtime.records import save_json
from .config import UNSEEN_TASKS


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


def gaussian_seed(context_id, base_seed, replica):
    return int(stable_id(context_id, base_seed, replica)[:16], 16) & ((1 << 63) - 1)


def index_contexts(manifest, output, points=10, task_limit=5, trajectories_per_task=20):
    if min(points, task_limit, trajectories_per_task) < 1:
        raise ValueError("Dataset counts must be positive")
    if (Path(output) / "index.json").exists():
        raise FileExistsError("Context index already exists; choose a new output")
    source = Path(manifest).resolve()
    records = read_lines(source)
    required = ("task", "trajectory_id", "scene_id", "demo_id", "demo_latent", "initial", "history", "chunk_count", "simulator_seed")
    identities = set()
    for row in records:
        if any(key not in row for key in required):
            raise ValueError(f"Trajectory requires fields: {', '.join(required)}")
        if row["task"] in UNSEEN_TASKS:
            raise ValueError(f"Unseen task cannot enter training data: {row['task']}")
        key = (row["task"], str(row["trajectory_id"]))
        if key in identities:
            raise ValueError(f"Duplicate trajectory: {key}")
        identities.add(key)
        for name in ("demo_latent", "initial", "history"):
            path = Path(row[name]).expanduser()
            row[name] = str((source.parent / path).resolve() if not path.is_absolute() else path.resolve())
            if not Path(row[name]).exists():
                raise FileNotFoundError(row[name])
    tasks = sorted({row["task"] for row in records})[:task_limit]
    selected = [row for task in tasks for row in sorted(
        (r for r in records if r["task"] == task), key=lambda r: str(r["trajectory_id"]))[:trajectories_per_task]]
    if len(tasks) < task_limit:
        raise ValueError(f"Need {task_limit} distinct seen tasks, found {len(tasks)}")
    # Connected components prevent shared demos OR shared scenes crossing splits.
    parents = list(range(len(selected)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    owners = {}
    for index, row in enumerate(selected):
        for key in (("scene", row["task"], str(row["scene_id"])),
                    ("seed", row["task"], int(row["simulator_seed"])),
                    ("demo_id", row["task"], str(row["demo_id"])), ("demo", row["demo_latent"])):
            if key in owners:
                parents[find(index)] = find(owners[key])
            owners[key] = index
    groups = {}
    for index, row in enumerate(selected):
        groups.setdefault(find(index), []).append(row)
    ordered = sorted(groups.values(), key=lambda group: stable_id(*sorted(
        row["task"] + ":" + str(row["trajectory_id"]) for row in group)))
    if len(ordered) < 3:
        raise ValueError("Need at least three independent scene/demo groups for train/val/test")
    train_end = max(1, min(len(ordered) - 2, round(len(ordered) * .7)))
    val_end = max(train_end + 1, min(len(ordered) - 1, round(len(ordered) * .85)))
    contexts = []
    output = Path(output)
    for group_index, group in enumerate(ordered):
        split = "train" if group_index < train_end else "val" if group_index < val_end else "test"
        for row in group:
            history = sorted(Path(row["history"]).glob("chunk_*.pt"))
            if [p.name for p in history] != [f"chunk_{i:04d}.pt" for i in range(len(history))]:
                raise ValueError(f"Non-contiguous recorded history: {row['history']}")
            count = int(row["chunk_count"])
            if count < 1 or len(history) < count - 1:
                raise ValueError("Recorded history is too short for chunk_count")
            indices = sorted({round(i * (count - 1) / max(1, min(points, count) - 1))
                              for i in range(min(points, count))})
            for chunk in indices:
                identifier = stable_id(row["task"], row["trajectory_id"], chunk)
                context = dict(row, id=identifier, split=split, chunk=chunk,
                               history=[str(p) for p in history[:chunk]])
                path = output / "contexts" / f"{identifier}.json"
                save_json(path, context)
                contexts.append(dict(id=identifier, path=str(path.resolve()), split=split, task=row["task"],
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save_json(output / "index.json", dict(source=str(source), tasks=tasks, trajectories=len(selected),
              groups=len(ordered), contexts=contexts,
              split_counts={name: sum(c["split"] == name for c in contexts) for name in ("train", "val", "test")}))
    return contexts
