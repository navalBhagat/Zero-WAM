"""Compact supervisor-facing campaign output; raw experiment records stay intact."""

from collections import defaultdict
import statistics
from .plots import campaign_plots as plots

__all__ = ["results", "plots", "clean_legacy"]


LEGACY_FILES = (
    "runs.csv",
    "episodes.csv",
    "summary.csv",
    "paired.csv",
    "server_chunks.csv",
    "guidance.csv",
    "per_seed.csv",
    "README.md",
    "timing.png",
    "server_chunks.png",
    "guidance.png",
    "job_timing.png",
)


def results(runs, episodes, wilson):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["task"], run["label"])].append(run)
    rows = []
    for (task, condition), jobs in sorted(groups.items()):
        values = [e for e in episodes if (e["task"], e["label"]) == (task, condition)]
        # Resumed segments represent episode ranges, not independent new requests.
        targets = defaultdict(set)
        for run in jobs:
            targets[(run["seed"], run["repeat"])].update(
                range(run["episode_offset"], run["episode_offset"] + run["requested"])
            )
        requested = sum(map(len, targets.values()))
        n = len(values)
        successes = sum(e["success"] for e in values)
        low, high = wilson(successes, n)
        row = dict(
            task=task,
            condition=condition,
            requested_episodes=requested,
            completed_episodes=n,
            missing_episodes=max(0, requested - n),
            successes=successes,
            success_rate=successes / n if n else None,
            success_ci_low=low,
            success_ci_high=high,
            interval_method="pooled descriptive Wilson; episodes share seed groups",
            scene_seed_groups=len({e["seed"] for e in values}),
            sampling_seeds=",".join(sorted({str(e["sampling_seed"]) for e in values})),
            infrastructure_or_incomplete_episodes=sum(
                r["incomplete_records"] for r in jobs
            ),
            failed_or_incomplete_jobs=sum(r["status"] != "complete" for r in jobs),
            timed_jobs=sum(r.get("allocated_seconds") is not None for r in jobs),
            jobs=len(jobs),
            sbu_total=sum(
                r["estimated_sbu"] for r in jobs if r.get("estimated_sbu") is not None
            )
            if any(r.get("estimated_sbu") is not None for r in jobs)
            else None,
            preparation_definition="client server-reset roundtrip per episode; excludes process startup",
            rollout_definition="episode wall time minus server reset; includes scene setup and simulator",
            compute_scope="transformer block linear subtotal, NOT total policy compute",
            compute_omissions="VAE/DINO forward and backward, conditioning/MCP, attention execution and other operators",
            total_policy_tflops=None,
        )
        for key in (
            "wall_seconds",
            "preparation_seconds",
            "rollout_seconds",
            "actions",
            "chunks",
            "sampling_seconds_per_chunk",
            "policy_counted_tflops",
            "total_policy_tflops",
            "policy_backward_tflops",
            "correction_base_percent_median",
            "clipped_corrections",
            "transformer_core_linear_tflops",
            "transformer_dense_attention_tflops",
            "transformer_core_linear_tflops_per_chunk",
            "compute_sampling_log_coverage",
            "guidance_updates",
            "reward_decoder_forwards",
            "reward_dino_forwards",
            "reward_backward_passes",
            "reference_frames",
        ):
            xs = [e[key] for e in values if e.get(key) is not None]
            row[key + "_mean"] = statistics.mean(xs) if xs else None
            row[key + "_n"] = len(xs)
        peaks = [
            r["gpu_peak_used_mib"]
            for r in jobs
            if r.get("gpu_peak_used_mib") is not None
        ]
        row["gpu_peak_used_gib"] = max(peaks) / 1024 if peaks else None
        row["total_policy_tflops"] = (
            row.get("total_policy_tflops_mean")
            if row.get("total_policy_tflops_n") == n and n
            else None
        )
        row["compute_scope"] = "; ".join(
            sorted({e.get("compute_scope", "unavailable") for e in values})
        )
        row["compute_omissions"] = "; ".join(
            sorted({e.get("compute_omissions", "unavailable") for e in values})
        )
        row["compute_status"] = (
            "; ".join(sorted({e.get("compute_status", "unavailable") for e in values}))
            or "unavailable"
        )
        row["compute_assumptions"] = "; ".join(
            sorted(
                {
                    e["compute_assumptions"]
                    for e in values
                    if e.get("compute_assumptions")
                }
            )
        )
        row["reset_subtiming_episodes"] = sum(
            bool(e.get("reset_components_seconds")) for e in values
        )
        row["reset_components_seconds_mean"] = None
        import json

        components = defaultdict(list)
        for e in values:
            for k, v in e.get("reset_components_seconds", {}).items():
                components[k].append(v)
        if components:
            row["reset_components_seconds_mean"] = json.dumps(
                {k: statistics.mean(v) for k, v in components.items()}, sort_keys=True
            )
        comparable = [e for e in episodes if e["task"] == task]
        scene_sets = defaultdict(set)
        for e in comparable:
            if e.get("simulator_seed") is not None:
                scene_sets[e["label"]].add(
                    (
                        e["simulator_seed"],
                        e["sampling_seed"],
                        e["icl_seed"],
                        e["repeat"],
                    )
                )
        common = (
            set.intersection(*scene_sets.values()) if len(scene_sets) > 1 else set()
        )
        matched = [
            e
            for e in values
            if (e["simulator_seed"], e["sampling_seed"], e["icl_seed"], e["repeat"])
            in common
        ]
        row["matched_scene_episodes"] = len(matched)
        row["matched_scene_success_rate"] = (
            sum(e["success"] for e in matched) / len(matched) if matched else None
        )
        rows.append(row)
    return rows


def clean_legacy(output):
    # Retain accounting evidence without cluttering the presentation outputs.
    for name in ("accounting.csv", "sacct.txt"):
        source = output / name
        if source.is_file():
            detail = output / "diagnostics"
            detail.mkdir(exist_ok=True)
            target = detail / ("legacy-" + name)
            version = 1
            while target.exists() and target.read_bytes() != source.read_bytes():
                target = detail / (f"legacy-{version}-" + name)
                version += 1
            if target.exists():
                source.unlink()
            else:
                source.rename(target)
    for name in LEGACY_FILES:
        path = output / name
        if path.is_file():
            path.unlink()
