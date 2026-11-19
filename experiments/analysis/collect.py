"""Normalize historical/compact artifacts into run, episode and trace rows.

Only completed episodes enter success metrics. Duplicate actual-scene records
are counted once; conflicting duplicate outcomes are rejected. Missing timing
and compute measurements stay missing rather than becoming zero-cost runs.
"""

import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from evaluation.robotwin.artifacts import find_artifact
from .compute import estimate as estimate_compute, records as compute_records
from .io import read, log_records
from .metrics import breakdown


def run_folder(specification: dict, runs_root: Path | None) -> Path:
    """Find a copied resumed segment in its original stage when necessary."""
    root = runs_root or Path(specification["environment"]["EXPERIMENT_OUTPUT_ROOT"])
    folder = root / f"{specification['task']}_{specification.get('job_id')}"
    if runs_root and not folder.exists():
        candidates = list(runs_root.parent.parent.glob(f"*/runs/{folder.name}"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous local run copies: {folder.name}")
    return folder


def correction_ratios(log_path: Path) -> dict[int, list[float]]:
    """Recover applied/base percentages from legacy logs, indexed by reset ordinal."""
    ratios = defaultdict(list)
    ordinal = -1
    lines = (
        log_path.read_text(errors="replace").splitlines() if log_path.is_file() else []
    )
    for line in lines:
        if "[conditioning]" in line:
            ordinal += 1
        if "[external-guidance]" not in line:
            continue
        try:
            record = ast.literal_eval(line.split("[external-guidance]", 1)[1].strip())
            if isinstance(record, dict) and record.get("base_update_norm", 0) > 0:
                ratios[ordinal].append(
                    100 * record["applied_correction_norm"] / record["base_update_norm"]
                )
        except (ValueError, SyntaxError, KeyError):
            continue
    return ratios


def gpu_statistics(path: Path) -> dict:
    """Summarize device-level samples; PyTorch allocator peaks are a different metric."""
    if not path.is_file():
        return {}
    with path.open() as stream:
        samples = list(csv.DictReader(stream, skipinitialspace=True))
    result = {}
    columns = [
        ("memory.used [MiB]", "gpu_peak_used_mib", max),
        ("utilization.gpu [%]", "gpu_mean_utilization", statistics.mean),
    ]
    for column, target, operation in columns:
        numbers = []
        for sample in samples:
            try:
                numbers.append(float(str(sample.get(column, "")).split()[0]))
            except (ValueError, TypeError, IndexError):
                continue
        result[target] = operation(numbers) if numbers else None
    return result


def run_record(
    base: dict, folder: Path, config: dict, timing: dict, environment: dict, rate: float
) -> dict:
    """Job-level accounting includes failures and allocation outside completed episodes."""
    requested = int(environment["TEST_NUM"])
    completed = sum(
        episode.get("status") == "completed" for episode in config.get("episodes", [])
    )
    allocated = timing.get("allocated_seconds")
    return dict(
        base,
        episode_offset=int(environment.get("EPISODE_OFFSET", 0)),
        incomplete_records=sum(
            episode.get("status") != "completed"
            for episode in config.get("episodes", [])
        ),
        path=str(folder),
        requested=requested,
        completed=completed,
        missing=max(0, requested - completed),
        exit_code=timing.get("exit_code"),
        allocated_seconds=allocated,
        estimated_sbu=allocated / 3600 * rate if allocated is not None else None,
        startup_seconds=timing.get("server_startup_seconds"),
        pre_server_seconds=timing.get("pre_server_seconds"),
        client_seconds=timing.get("client_seconds"),
        status="complete"
        if completed == requested and timing.get("exit_code") == 0
        else "incomplete_or_failed",
    )


def episode_record(
    base: dict,
    episode: dict,
    folder: Path,
    environment: dict,
    ordinal: int,
    estimates: dict,
    ratios: list,
    reset_profiles: list,
) -> dict:
    """Combine measured timing, effective conditioning and coverage-labelled arithmetic."""
    profile_path = next(
        folder.glob(f"**/profile_episode_{episode['episode_id']}.json"), None
    )
    profile = read(profile_path, {}) if profile_path else {}
    row = dict(
        base,
        episode_id=episode["episode_id"],
        simulator_seed=episode.get("simulator_seed"),
        sampling_seed=environment["SAMPLING_SEED"],
        icl_seed=episode.get("icl_seed"),
        success=int(episode["success"]),
        actions=episode.get("policy_action_calls"),
        wall_seconds=profile.get("wall_seconds"),
        profile_available=bool(profile),
        step_limit_counter=episode.get("simulator_action_count"),
    )
    if profile:
        row.update(breakdown(profile))
        timings = profile.get("timings", {})
        calls = timings.get("server_sampling_roundtrip", {}).get("calls", 0)
        row["chunks"] = calls
        row["sampling_seconds_per_chunk"] = row["sampling"] / calls if calls else None
        row["seconds_per_action"] = (
            profile["wall_seconds"] / row["actions"] if row["actions"] else None
        )
        row["camera_hooks"] = json.dumps(
            {
                key: value
                for key, value in profile.get("hooks", {}).items()
                if "camera" in key or "physics" in key
            }
        )
    row.update(estimates.get(episode["episode_id"], {}))
    if ratios and row.get("correction_base_percent_median") is None:
        row["correction_base_percent_median"] = statistics.median(ratios)
    inference_steps = episode.get("server", {}).get("video_inference_steps")
    if row.get("chunks") and inference_steps is not None:
        expected = row["chunks"] * (inference_steps + 1)
        recorded = row.get("sampling_video_calls", 0)
        row["compute_sampling_log_coverage"] = min(1.0, recorded / expected)
        if recorded != expected and not row.get("operator_counting_available"):
            row["compute_status"] = (
                row.get("compute_status", "unavailable") + "; incomplete_sampling_logs"
            )
    if row.get("transformer_core_linear_tflops") is not None and row.get("chunks"):
        row["transformer_core_linear_tflops_per_chunk"] = (
            row["sampling_transformer_core_linear_tflops"] / row["chunks"]
        )
    reset = episode.get("server", {}).get("reset_profile")
    if not reset:
        reset = next(
            (record for record in reset_profiles if record.get("episode") == ordinal),
            None,
        )
    row["reset_components_seconds"] = (reset or {}).get("components_seconds", {})
    if profile and "server_reset" in profile.get("timings", {}):
        row["preparation_seconds"] = row["server_reset"]
        row["rollout_seconds"] = max(0.0, row["wall_seconds"] - row["server_reset"])
    return row


def collect(
    manifest: dict, runs_root: Path | None, model_config: Path | None = None
) -> tuple[list, list, list, list]:
    """Return (runs, completed episodes, inference chunks, guidance updates).

    This tuple and the row dictionaries remain compatible with old analysis
    callers. Actual accepted scene seeds, not nominal seeds, identify duplicates.
    """
    runs, episodes, chunks, guidance = [], [], [], []
    seen_jobs, seen_episodes = set(), {}
    for number, specification in enumerate(manifest["runs"]):
        environment = specification["environment"]
        job = specification.get("job_id")
        job_key = (specification["task"], job)
        if job and job_key in seen_jobs:
            continue
        if job:
            seen_jobs.add(job_key)
        folder = run_folder(specification, runs_root)
        base = dict(
            run=number,
            job_id=job,
            task=specification["task"],
            label=specification["label"].split("_r")[0]
            if manifest["stage"] == "render"
            else specification["label"],
            run_label=specification["label"],
            seed=specification["seed"],
            repeat=specification.get("repeat", 0),
            mode=specification["mode"],
            spp=int(environment["RENDER_SAMPLES_PER_PIXEL"]),
            selector=environment["GUIDANCE_SELECTION"],
            decoder_dtype=environment["GUIDANCE_DECODER_DTYPE"],
        )
        config = read(folder / "config.json", {})
        timing = read(find_artifact(folder, "job_timing.json"), {})
        estimates = estimate_compute(folder, config, model_config)
        log_path = find_artifact(folder, "server.log")
        reset_profiles = compute_records(log_path, "[reset-profile]")
        ratios = correction_ratios(log_path)
        run = run_record(
            base,
            folder,
            config,
            timing,
            environment,
            manifest["estimate"]["sbu_per_hour"],
        )
        runs.append(run)
        for episode in config.get("episodes", []):
            if episode.get("status") != "completed":
                continue
            identity = (
                base["task"],
                base["label"],
                episode.get("simulator_seed"),
                environment["SAMPLING_SEED"],
                episode.get("icl_seed"),
                base["repeat"],
            )
            if episode.get("simulator_seed") is not None and identity in seen_episodes:
                if seen_episodes[identity] != episode["success"]:
                    raise ValueError(f"Conflicting duplicate episode: {identity}")
                continue
            if episode.get("simulator_seed") is not None:
                seen_episodes[identity] = episode["success"]
            ordinal = config["episodes"].index(episode)
            episodes.append(
                episode_record(
                    base,
                    episode,
                    folder,
                    environment,
                    ordinal,
                    estimates,
                    ratios[ordinal],
                    reset_profiles,
                )
            )
        profiled = [
            episode["wall_seconds"]
            for episode in episodes
            if episode["run"] == number and episode.get("wall_seconds") is not None
        ]
        run["profiled_episode_seconds"] = sum(profiled)
        run["client_outside_episode_seconds"] = (
            max(0, timing["client_seconds"] - sum(profiled))
            if timing.get("client_seconds") is not None
            else None
        )
        run.update(gpu_statistics(find_artifact(folder, "gpu.csv")))
        for marker, target in [
            ("[inference-profile]", chunks),
            ("[external-guidance]", guidance),
        ]:
            for index, entry in enumerate(log_records(log_path, marker)):
                target.append(dict(base, call=index, **entry))
    return runs, episodes, chunks, guidance
