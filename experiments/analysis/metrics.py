"""Exclusive timing, descriptive success intervals and actual-scene pairing.

Missing observations stay missing. Wilson intervals are pooled descriptive
intervals, not a claim that episodes sharing scene/noise groups are independent.
"""

from collections import defaultdict
import csv
import math
import statistics

COMPONENTS = [
    "sampling",
    "server_reset",
    "cache_update",
    "camera_render",
    "planning",
    "physics",
    "action_other",
    "scene_setup",
    "video_encoding",
    "other_profiled",
    "unprofiled",
]


def breakdown(profile):
    result = dict.fromkeys(COMPONENTS, 0.0)
    for name, timing in profile.get("timings", {}).items():
        key = {
            "server_sampling_roundtrip": "sampling",
            "server_reset": "server_reset",
            "server_cache_update": "cache_update",
            "physics_step": "physics",
            "action_execution": "action_other",
            "scene_setup": "scene_setup",
            "video_encoding": "video_encoding",
        }.get(name)
        if name in ("camera_capture", "camera_readback", "render_state_update"):
            key = "camera_render"
        if name.startswith("planning_"):
            key = "planning"
        result[key or "other_profiled"] += timing.get("exclusive_seconds", 0.0)
    result["unprofiled"] = max(0.0, profile["wall_seconds"] - sum(result.values()))
    return result


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.96
    p = successes / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def summaries(episodes):
    groups = defaultdict(list)
    for row in episodes:
        groups[(row["task"], row["label"], row["spp"])].append(row)
    rows = []
    for (task, label, spp), values in sorted(groups.items()):
        n = len(values)
        successes = sum(v["success"] for v in values)
        low, high = wilson(successes, n)
        row = dict(
            task=task,
            label=label,
            spp=spp,
            n=n,
            successes=successes,
            success_rate=successes / n,
            wilson_low=low,
            wilson_high=high,
        )
        for key in [
            "wall_seconds",
            "seconds_per_action",
            "sampling_seconds_per_chunk",
            *COMPONENTS,
        ]:
            xs = [v[key] for v in values if v.get(key) is not None]
            row[key + "_mean"] = statistics.mean(xs) if xs else None
            row[key + "_sd"] = statistics.stdev(xs) if len(xs) > 1 else None
        rows.append(row)
    return rows


def pairs(episodes, stage):
    groups = defaultdict(list)
    for row in episodes:
        # Use actual accepted simulator seeds, not only nominal run seeds.
        key = (
            row["task"],
            row["simulator_seed"],
            row["sampling_seed"],
            row["icl_seed"],
            row["repeat"],
        )
        groups[key].append(row)
    result = []
    for key, values in groups.items():
        for left in values:
            for right in values:
                if stage == "render":
                    compare = left["spp"] == 32 and right["spp"] in (16, 8)
                elif stage == "textcheck":
                    compare = left["mode"] == "icl" and right["mode"] in (
                        "icl_legacy_text",
                        "icl_original_caption_only",
                    )
                elif stage == "baselines":
                    compare = left["mode"] == "text" and right["mode"] == "icl"
                elif stage == "guidance":
                    compare = (
                        left["mode"] == right["mode"]
                        and left["selector"] == "fixed"
                        and right["selector"] == "window"
                    )
                else:
                    compare = (
                        left["mode"] == right["mode"]
                        and left["decoder_dtype"] == "float32"
                        and right["decoder_dtype"] == "float16"
                    )
                if not compare:
                    continue
                row = dict(
                    task=key[0],
                    simulator_seed=key[1],
                    repeat=key[4],
                    left=left["label"],
                    right=right["label"],
                    success_delta=right["success"] - left["success"],
                    action_count_equal=left["actions"] == right["actions"],
                )
                for field in (
                    "wall_seconds",
                    "seconds_per_action",
                    "sampling_seconds_per_chunk",
                ):
                    a, b = left.get(field), right.get(field)
                    row[field + "_delta"] = (
                        b - a if a is not None and b is not None else None
                    )
                result.append(row)
    return result


def accounting_from_sacct(raw, rate):
    rows = []
    for record in csv.DictReader(raw.splitlines(), delimiter="|"):
        try:
            seconds = int(record["ElapsedRaw"])
        except (ValueError, KeyError, TypeError):
            continue
        resources = {}
        for item in record.get("AllocTRES", "").split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                resources[key] = value
        count = resources.get("gres/gpu")
        try:
            count = float(count)
        except (ValueError, TypeError):
            count = None
        rows.append(
            dict(
                job_id=record.get("JobIDRaw"),
                state=record.get("State"),
                elapsed_seconds=seconds,
                gpus=count,
                exit_code=record.get("ExitCode"),
                estimated_sbu=seconds / 3600 * count * rate
                if count is not None
                else None,
            )
        )
    return rows
