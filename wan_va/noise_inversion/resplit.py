"""Create a new split view of collected pairs without changing or copying tensors."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path

from .dataset import grouped_splits, stable_id
from .quality import inspect_dataset
from .records import read_json, save_json


def resplit_collection(
    source: str | Path, output: str | Path, seed: int = 1234
) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh split-view directory")
    if source in output.parents:
        raise ValueError(
            "Write the split view outside the original collection directory"
        )
    if seed < 0:
        raise ValueError("Split seed must be nonnegative")
    quality = inspect_dataset(source, check_tensors=False, require_task_coverage=False)
    if not quality["passed"]:
        raise ValueError(
            "Source dataset integrity failed: " + "; ".join(quality["errors"][:5])
        )
    manifest = read_json(source / "manifest.json")
    records, contexts = {}, {}
    for job in manifest["jobs"]:
        summary = read_json(source / "jobs" / job["name"] / "summary.json", {})
        if summary.get("status") != "completed":
            raise ValueError("Complete collection before changing its split view")
        for pair in job["pairs"]:
            record = read_json(source / "pairs" / pair["id"] / "record.json")
            records[pair["id"]] = record
            path = str(Path(record["context"]).resolve())
            if path not in contexts:
                contexts[path] = dict(read_json(path), source_context=path)

    assignments = grouped_splits(list(contexts.values()), seed)
    group_counts = defaultdict(Counter)
    context_counts = defaultdict(Counter)
    rewritten = {}
    for group, split in assignments:
        group_counts[group[0]["task"]][split] += 1
        for original in group:
            context = dict(original, split=split, source_split=original["split"])
            original_path = original["source_context"]
            context_path = output / "contexts" / f"{stable_id(original_path)}.json"
            context["source_context_sha256"] = hashlib.sha256(
                Path(original_path).read_bytes()
            ).hexdigest()
            save_json(context_path, context)
            rewritten[original_path] = dict(
                context=str(context_path),
                split=split,
                context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
            )
            context_counts[context["task"]][split] += 1

    # Keep the original collection digest: it identifies the code that generated
    # the shared tensors. The separate split-view receipt describes this rewrite.
    updated = deepcopy(manifest)
    updated["dataset_view"] = dict(
        source=str(source), seed=seed, policy="per_task_recording_demo_groups"
    )
    for job in updated["jobs"]:
        job["submissions"] = []
        for pair in job["pairs"]:
            record = records[pair["id"]]
            original_context = str(Path(record["context"]).resolve())
            replacement = rewritten[original_context]
            updated_record = dict(record, **replacement)
            updated_record.update(
                source_record=str(source / "pairs" / pair["id"] / "record.json"),
                source_context=original_context,
                source_split=record["split"],
            )
            save_json(output / "pairs" / pair["id"] / "record.json", updated_record)
            pair.update(replacement)
        save_json(output / "jobs" / job["name"] / "settings.json", job)
        save_json(
            output / "jobs" / job["name"] / "summary.json",
            dict(status="completed", reused_collection=str(source)),
        )
    save_json(output / "manifest.json", updated)
    checked = inspect_dataset(output, check_tensors=False)
    if not checked["passed"]:
        raise ValueError(
            "Split-view validation failed: " + "; ".join(checked["errors"][:5])
        )
    receipt = dict(
        source=str(source),
        output=str(output),
        seed=seed,
        policy=updated["dataset_view"]["policy"],
        pairs=checked["eligible_pairs"],
        groups={task: dict(counts) for task, counts in group_counts.items()},
        contexts={task: dict(counts) for task, counts in context_counts.items()},
        pair_counts={
            task: values["splits"] for task, values in checked["tasks"].items()
        },
        previous_pair_counts={
            task: values["splits"] for task, values in quality["tasks"].items()
        },
        source_manifest_sha256=hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
        note="Only metadata was rewritten. Keep the original tensors and shared histories available. "
        "Start fresh training; do not resume a model trained under the previous split.",
    )
    save_json(output / "resplit.json", receipt)
    return receipt
