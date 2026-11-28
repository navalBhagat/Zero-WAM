"""Grouped trajectory indexing; seeds/chunks can never cross split boundaries."""

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from .records import save_json
from .tasks import UNSEEN_TASKS

SPLITS = ("train", "val", "test")


def context_group_keys(context: dict) -> list[tuple]:
    """Identities that must never cross training/validation/test boundaries."""
    task = context["task"]
    keys = [
        ("scene", task, str(context["scene_id"])),
        ("demo_id", task, str(context["demo_id"])),
        ("demo", str(Path(context["demo_latent"]).resolve())),
    ]
    if context.get("trajectory_id") is not None:
        keys.append(("trajectory", task, str(context["trajectory_id"])))
    if context.get("simulator_seed") is not None:
        keys.append(("seed", task, int(context["simulator_seed"])))
    return keys


def split_violations(contexts: list[dict]) -> list[str]:
    owners = {}
    errors = set()
    for context in contexts:
        split = context["split"]
        if split not in SPLITS:
            errors.add(f"Unknown dataset split: {split}")
        for key in context_group_keys(context):
            previous = owners.setdefault(key, split)
            if previous != split:
                errors.add(f"Dataset leaks {key} across {previous}/{split}")
    return sorted(errors)


def load_examples(dataset: str | Path) -> list[tuple[dict, dict]]:
    """Load eligible training records and check their context provenance/groups."""
    rows = []
    for path in sorted(Path(dataset).glob("pairs/*/record.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "completed" or row.get("fallback"):
            continue
        context_path = Path(row["context"])
        expected_hash = row.get("context_sha256")
        if (
            expected_hash
            and hashlib.sha256(context_path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError(
                f"Training context changed since collection: {context_path}"
            )
        context = json.loads(context_path.read_text())
        if context["task"] in UNSEEN_TASKS:
            raise ValueError("Unseen-task example found in training dataset")
        rows.append((row, context))
    errors = split_violations([context for _, context in rows])
    if errors:
        raise ValueError("; ".join(errors))
    return rows


def read_lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


def gaussian_seed(context_id, base_seed, replica):
    return int(stable_id(context_id, base_seed, replica)[:16], 16) & ((1 << 63) - 1)


def balanced_points(rows, points, task_budget):
    """Round-robin demos, then trajectories, then time; never repeat a point."""
    demos = defaultdict(list)
    for row in sorted(rows, key=lambda r: str(r["trajectory_id"])):
        count = int(row["chunk_count"])
        indices = sorted(
            {
                round(i * (count - 1) / max(1, min(points, count) - 1))
                for i in range(min(points, count))
            }
        )
        # Under a budget, visit both ends and then fill the largest temporal gap
        # rather than concentrating all selected contexts near the beginning.
        temporal = []
        if indices:
            temporal.append(indices.pop(0))
        if indices:
            temporal.append(indices.pop())
        while indices:
            next_point = max(
                indices, key=lambda point: min(abs(point - used) for used in temporal)
            )
            temporal.append(next_point)
            indices.remove(next_point)
        demos[str(row["demo_id"])].append(
            deque((row["trajectory_id"], i) for i in temporal)
        )
    queues = []
    for demo in sorted(demos):
        trajectories = deque(demos[demo])
        queue = deque()
        while trajectories:
            trajectory = trajectories.popleft()
            queue.append(trajectory.popleft())
            if trajectory:
                trajectories.append(trajectory)
        queues.append(queue)
    queues = deque(queues)
    selected = defaultdict(list)
    for _ in range(task_budget):
        if not queues:
            break
        queue = queues.popleft()
        trajectory, chunk = queue.popleft()
        selected[trajectory].append(chunk)
        if queue:
            queues.append(queue)
    return selected


def grouped_splits(
    records: list[dict], seed: int = 1234
) -> list[tuple[list[dict], str]]:
    """Split connected recording/demo groups 70/15/15 within each task."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, row in enumerate(records):
        for key in context_group_keys(row):
            if key in owners:
                parents[find(index)] = find(owners[key])
            owners[key] = index
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(find(index), []).append(row)
    task_groups = defaultdict(list)
    for group in groups.values():
        tasks = {row["task"] for row in group}
        if len(tasks) != 1:
            raise ValueError(
                "A shared recording/demo connects multiple tasks; independent per-task splitting is unsafe"
            )
        task_groups[tasks.pop()].append(group)

    assignments = []
    for task, groups in sorted(task_groups.items()):
        if len(groups) < 3:
            raise ValueError(
                f"{task}: need at least three independent recording/demo groups"
            )
        ordered = sorted(
            groups,
            key=lambda group: stable_id(
                seed, task, *sorted({str(row["trajectory_id"]) for row in group})
            ),
        )
        train_end = max(1, min(len(ordered) - 2, round(len(ordered) * 0.7)))
        val_end = max(train_end + 1, min(len(ordered) - 1, round(len(ordered) * 0.85)))
        for index, group in enumerate(ordered):
            split = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
            assignments.append((group, split))
    return assignments


def index_contexts(
    manifest,
    output,
    points=10,
    task_limit=5,
    trajectories_per_task=20,
    task_context_limit=None,
):
    if min(points, task_limit, trajectories_per_task) < 1:
        raise ValueError("Dataset counts must be positive")
    if task_context_limit is not None and task_context_limit < 1:
        raise ValueError("Task context limit must be positive")
    if (Path(output) / "index.json").exists():
        raise FileExistsError("Context index already exists; choose a new output")
    source = Path(manifest).resolve()
    records = read_lines(source)
    required = (
        "task",
        "trajectory_id",
        "scene_id",
        "demo_id",
        "demo_latent",
        "initial",
        "history",
        "chunk_count",
        "simulator_seed",
    )
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
            row[name] = str(
                (source.parent / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )
            if not Path(row[name]).exists():
                raise FileNotFoundError(row[name])
    tasks = sorted({row["task"] for row in records})[:task_limit]
    selected = [
        row
        for task in tasks
        for row in sorted(
            (r for r in records if r["task"] == task),
            key=lambda r: str(r["trajectory_id"]),
        )[:trajectories_per_task]
    ]
    if len(tasks) < task_limit:
        raise ValueError(f"Need {task_limit} distinct seen tasks, found {len(tasks)}")
    chosen = {
        task: balanced_points(
            [r for r in selected if r["task"] == task],
            points,
            (
                task_context_limit
                if task_context_limit is not None
                else sum(
                    min(points, int(r["chunk_count"]))
                    for r in selected
                    if r["task"] == task
                )
            ),
        )
        for task in tasks
    }
    assignments = grouped_splits(selected)
    contexts = []
    output = Path(output)
    for group, split in assignments:
        for row in group:
            history = sorted(Path(row["history"]).glob("chunk_*.pt"))
            if [p.name for p in history] != [
                f"chunk_{i:04d}.pt" for i in range(len(history))
            ]:
                raise ValueError(f"Non-contiguous recorded history: {row['history']}")
            count = int(row["chunk_count"])
            if count < 1 or len(history) < count - 1:
                raise ValueError("Recorded history is too short for chunk_count")
            indices = sorted(chosen[row["task"]][row["trajectory_id"]])
            for chunk in indices:
                identifier = stable_id(row["task"], row["trajectory_id"], chunk)
                context = dict(
                    row,
                    id=identifier,
                    split=split,
                    chunk=chunk,
                    history=[str(p) for p in history[:chunk]],
                )
                path = output / "contexts" / f"{identifier}.json"
                save_json(path, context)
                contexts.append(
                    dict(
                        id=identifier,
                        path=str(path.resolve()),
                        split=split,
                        task=row["task"],
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    save_json(
        output / "index.json",
        dict(
            source=str(source),
            tasks=tasks,
            trajectories=len(selected),
            groups=len(assignments),
            split_policy="per_task_recording_demo_groups",
            contexts=contexts,
            task_context_limit=task_context_limit,
            task_counts={
                task: sum(c["task"] == task for c in contexts) for task in tasks
            },
            split_counts={
                name: sum(c["split"] == name for c in contexts)
                for name in ("train", "val", "test")
            },
        ),
    )
    return contexts
