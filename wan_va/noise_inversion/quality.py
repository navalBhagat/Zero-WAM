"""Read-only checks for a completed paired-noise dataset; no GPU/model required."""

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import statistics

from .dataset import SPLITS, context_group_keys, split_violations
from .records import read_json
from .tasks import UNSEEN_TASKS


def error_statistics(values: list[float]) -> dict:
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=p95,
        maximum=max(values),
    )


def check_pair_tensors(record: dict, context: dict, checked_files: set) -> None:
    """Check regression targets and shared inputs once, using CPU tensors only."""
    import torch
    from .latents import reshape_latent

    tensors = torch.load(record["pair"], map_location="cpu", weights_only=True)
    original = tensors["original"]
    if original.ndim != 5 or original.shape[:3] != (1, 48, 2):
        raise ValueError("Noise must have shape [1, 48, 2, height, width]")
    for key in ("original", "recovered", "target", "mask"):
        value = tensors[key]
        if value.shape != original.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid shape or nonfinite values in {key}")
    if "action_noise" in tensors and not torch.isfinite(tensors["action_noise"]).all():
        raise ValueError("Nonfinite action noise")
    expected_mask = torch.ones_like(original)
    if context["chunk"] == 0:
        expected_mask[:, :, :1] = 0
    if not torch.equal(tensors["mask"], expected_mask):
        raise ValueError("Noise mask does not match the initial/history context")
    if len(context["history"]) != context["chunk"]:
        raise ValueError("History length differs from context chunk index")

    paths = [(context["demo_latent"], "demo"), (context["initial"], "initial")]
    paths.extend((path, "history") for path in context["history"])
    for path, kind in paths:
        cache_key = (str(Path(path).resolve()), kind, tuple(original.shape[-2:]))
        if cache_key in checked_files:
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if kind == "demo":
            demo = reshape_latent(saved["latent"], saved)
            if demo.shape[1] != 48 or any(size < 1 for size in demo.shape):
                raise ValueError(f"Invalid human demonstration shape: {path}")
            values = [demo]
        else:
            video = saved["initial_frame"] if kind == "initial" else saved["video"]
            frames = 1 if kind == "initial" else 2
            if video.shape != (1, 48, frames, *original.shape[-2:]):
                raise ValueError(f"Invalid {kind} video shape: {path}")
            values = [video]
            if kind == "history":
                if saved["actions"].shape != (1, 30, 2, 16, 1):
                    raise ValueError(f"Invalid recorded action shape: {path}")
                values.append(saved["actions"])
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError(f"Nonfinite shared context tensor: {path}")
        checked_files.add(cache_key)


def inspect_dataset(
    directory: str | Path,
    check_tensors: bool = True,
    worst: int = 10,
    require_task_coverage: bool = True,
) -> dict:
    """Summarize errors and report integrity failures without modifying the data."""
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if not manifest or manifest.get("stage") not in ("collect", "benchmark"):
        raise ValueError("Quality checks require a collection campaign directory")
    planned = [pair for job in manifest["jobs"] for pair in job.get("pairs", [])]
    errors, warnings, eligible = [], [], []
    contexts, valid_contexts = {}, {}
    checked_files, targets, identities = set(), set(), set()
    statuses = Counter()
    seen_ids = set()

    for pair in planned:
        pair_id = pair["id"]
        if pair_id in seen_ids:
            errors.append(f"Duplicate planned pair ID: {pair_id}")
        seen_ids.add(pair_id)
        path = directory / "pairs" / pair_id / "record.json"
        try:
            record = read_json(path)
        except (OSError, ValueError) as error:
            errors.append(f"Unreadable pair record {pair_id}: {error}")
            statuses["unreadable"] += 1
            continue
        if record is None:
            statuses["missing"] += 1
            errors.append(f"Missing pair record: {pair_id}")
            continue
        if not isinstance(record, dict):
            errors.append(f"Pair record must be a JSON object: {pair_id}")
            statuses["unreadable"] += 1
            continue
        statuses[record.get("status", "unknown")] += 1
        if record.get("status") != "completed":
            errors.append(f"Incomplete pair: {pair_id}")
            continue
        if record.get("fallback"):
            statuses["fallback"] += 1
            warnings.append(f"Excluded fallback target: {pair_id}")
            continue
        try:
            context_path = Path(pair["context"])
            if Path(record["context"]).resolve() != context_path.resolve():
                raise ValueError("Recorded context differs from planned context")
            if record["seed"] != pair["seed"]:
                raise ValueError("Recorded seed differs from planned seed")
            expected_hash = pair.get("context_sha256")
            if (
                expected_hash
                and hashlib.sha256(context_path.read_bytes()).hexdigest()
                != expected_hash
            ):
                raise ValueError("Context changed since collection was planned")
            if record.get("source_digest") != manifest["source_digest"]:
                raise ValueError(
                    "Pair source digest differs from collection source digest"
                )
            if context_path not in contexts:
                contexts[context_path] = read_json(context_path)
            context = contexts[context_path]
            if context["task"] in UNSEEN_TASKS or context["split"] not in SPLITS:
                raise ValueError("Held-out task or unknown split in collection")
            if (
                context["split"] != pair["split"]
                or record.get("split") != pair["split"]
            ):
                raise ValueError("Pair and context split assignments differ")
            context_group_keys(context)
            valid_contexts[context_path] = context
            identity = (
                context["task"],
                context["trajectory_id"],
                context["chunk"],
                pair["seed"],
            )
            target = str(Path(record["pair"]).resolve())
            if identity in identities or target in targets:
                raise ValueError("Duplicate context/seed or shared target file")
            identities.add(identity)
            targets.add(target)
            if not Path(target).is_file():
                raise ValueError("Missing pair tensor file")
            value = record["reconstruction_error"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Invalid reconstruction error")
            if check_tensors:
                check_pair_tensors(record, context, checked_files)
            eligible.append(
                dict(
                    id=pair_id,
                    task=context["task"],
                    split=context["split"],
                    trajectory=context["trajectory_id"],
                    chunk=context["chunk"],
                    seed=record["seed"],
                    reconstruction_error=value,
                    context=str(context_path),
                    pair=target,
                )
            )
        except Exception as error:
            # An audit should report every bad example, rather than stop at the
            # first damaged JSON/tensor file and conceal the rest of the dataset.
            errors.append(f"{pair_id}: {error}")

    errors.extend(split_violations(list(valid_contexts.values())))
    unexpected = {
        path.parent.name for path in directory.glob("pairs/*/record.json")
    } - seen_ids
    if unexpected:
        errors.append(f"Unplanned pair records: {sorted(unexpected)}")
    split_counts = dict(Counter(row["split"] for row in eligible))
    for split in SPLITS:
        if not split_counts.get(split):
            errors.append(f"No eligible pairs in {split} split")
    tasks = defaultdict(list)
    for row in eligible:
        tasks[row["task"]].append(row)
    task_statistics = {}
    for task, rows in sorted(tasks.items()):
        task_statistics[task] = dict(
            **error_statistics([row["reconstruction_error"] for row in rows]),
            contexts=len({row["context"] for row in rows}),
            splits=dict(Counter(row["split"] for row in rows)),
        )
        if require_task_coverage:
            for split in SPLITS:
                if not task_statistics[task]["splits"].get(split):
                    errors.append(f"{task}: no eligible pairs in {split} split")
    if any(
        context.get("simulator_seed") is None for context in valid_contexts.values()
    ):
        warnings.append(
            "Simulator seeds are unavailable: recording/demo separation is checked, "
            "but distinct underlying scenes across recordings cannot be verified."
        )
    return dict(
        passed=not errors,
        planned_pairs=len(planned),
        statuses=dict(statuses),
        eligible_pairs=len(eligible),
        splits=split_counts,
        tasks=task_statistics,
        reconstruction_error=error_statistics(
            [row["reconstruction_error"] for row in eligible]
        ),
        worst_examples=sorted(
            eligible, key=lambda row: row["reconstruction_error"], reverse=True
        )[:worst],
        tensor_checks=check_tensors,
        errors=errors,
        warnings=warnings,
        note="Reconstruction errors are relative latent-space norms, not robot success rates. "
        "No reconstruction-error acceptance threshold is assumed.",
    )
