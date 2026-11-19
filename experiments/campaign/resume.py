"""Reuse completed baseline prefixes only when policy inputs and source agree.

A resumed episode starts after the last accepted simulator seed, which may differ
from the nominal seed after expert-feasibility skipping. Source checks are strict.
"""

import copy
import hashlib
import json
from pathlib import Path
import zipfile
from evaluation.robotwin.artifacts import find_artifact


def compatible_client_changes(previous: str, current: str) -> list[str]:
    """Allow only the documented exception handler and output relocation edits."""
    before = """            except Exception as e:
                args["render_freq"] = render_freq
                raise RuntimeError(
                    f"Expert setup failed for {task_name}, seed {now_seed}; "
                    "stopping instead of retrying a potentially broken environment."
                ) from e
"""
    after = """            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue
"""
    changes = []
    if previous.count(before) == 1:
        previous = previous.replace(before, after, 1)
        changes.append("restored_author_expert_scene_skipping")
    old_output = '    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")'
    new_output = "    save_dir = Path(save_root).resolve() / 'eval_result' / task_name / policy_name / task_config / str(ckpt_setting) / current_time"
    if previous.count(old_output) == 1:
        previous = previous.replace(old_output, new_output, 1)
        changes.append("job_local_auxiliary_outputs")
    return changes if previous == current else []


def extend_baselines(
    specs: list[dict], previous_path: Path, root: Path
) -> tuple[list[dict], list[dict]]:
    previous_path = previous_path.resolve()
    previous = json.loads(previous_path.read_text())
    if previous["stage"] != "baselines":
        raise ValueError("--resume-from requires a baseline manifest")
    # Timing/output controls may change; policy, rendering and seed settings may not.
    ignored = {
        "COUNT_POLICY_FLOPS",
        "RUN_DIAGNOSTICS",
        "SAVE_GUIDANCE_CAPTURE",
        "TEST_NUM",
        "CLIENT_TIMEOUT",
        "EXPERIMENT_OUTPUT_ROOT",
        "EXPERIMENT_SOURCE_DIGEST",
        "EXPERIMENT_LABEL",
        "EXPERIMENT_STAGE",
        "EXPERIMENT_REPEAT",
        "START_SEED",
        "EPISODE_OFFSET",
        "ZERO_WAM_ROOT",
    }
    reused, pending = [], []
    for planned_run in specs:
        environment = planned_run["environment"]
        target = int(environment["TEST_NUM"])
        completed = {}
        sources = []
        for previous_run in previous["runs"]:
            if (previous_run["task"], previous_run["seed"], previous_run["mode"]) != (
                planned_run["task"],
                planned_run["seed"],
                planned_run["mode"],
            ):
                continue
            if any(
                previous_run["environment"].get(k) != v
                for k, v in environment.items()
                if k not in ignored
            ):
                continue
            if not previous_run.get("job_id"):
                continue
            folder = (
                Path(previous_run["environment"]["EXPERIMENT_OUTPUT_ROOT"])
                / f"{previous_run['task']}_{previous_run['job_id']}"
            )
            config_path = folder / "config.json"
            if not config_path.exists():
                raise ValueError(
                    f"Missing prior results: {config_path}; resume on the machine holding the results"
                )
            config = json.loads(config_path.read_text())
            episodes = [
                e for e in config.get("episodes", []) if e.get("status") == "completed"
            ]
            if not episodes:
                continue
            if not find_artifact(folder, "job_timing.json").exists():
                raise ValueError(
                    f"Prior job may still be active: {folder}; wait for it to finish"
                )
            # A source archive is retained for every referenced segment, including extensions.
            bundle = Path(
                previous_run.get("resume_source_bundle") or previous["source_bundle"]
            )
            with zipfile.ZipFile(bundle) as archive:
                hashes = json.loads(archive.read("source_hashes.json"))
                old_client = archive.read(
                    "evaluation/robotwin/eval_policy_client_openpi.py"
                ).decode()
            accepted_changes = []
            for name, expected in hashes.items():
                if not (
                    name.startswith("wan_va/")
                    or name == "evaluation/robotwin/eval_policy_client_openpi.py"
                ):
                    continue
                path = root / name
                if (
                    not path.exists()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    changes = (
                        compatible_client_changes(old_client, path.read_text())
                        if name == "evaluation/robotwin/eval_policy_client_openpi.py"
                        and path.exists()
                        else []
                    )
                    if changes:
                        accepted_changes.extend(changes)
                        continue
                    raise ValueError(
                        f"Runtime changed since prior results: {name}; use a fresh campaign"
                    )
            for episode in episodes:
                index = int(episode["episode_id"])
                if index in completed:
                    raise ValueError(
                        f"Duplicate completed episode {index} for {planned_run['task']}/{planned_run['mode']}"
                    )
                completed[index] = episode
            item = copy.deepcopy(previous_run)
            item["resume_source_bundle"] = str(bundle)
            if accepted_changes:
                item["resume_accepted_runtime_changes"] = accepted_changes
            sources.append(item)
        prefix = 0
        while prefix in completed:
            prefix += 1
        if len(completed) != prefix:
            raise ValueError(
                "Prior completed episodes have gaps; automatic resume requires a contiguous prefix"
            )
        if prefix > target:
            raise ValueError(
                "--episodes is the desired total and is smaller than the completed prefix"
            )
        reused.extend(sources)
        if prefix == target:
            continue
        if prefix:
            environment["EPISODE_OFFSET"] = str(prefix)
            # Feasibility checks may have skipped seeds; do not assume start + episode count.
            environment["START_SEED"] = str(
                int(completed[prefix - 1]["simulator_seed"]) + 1
            )
            environment["TEST_NUM"] = str(target - prefix)
            planned_run["resumed_completed_episodes"] = prefix
        pending.append(planned_run)
    return pending, reused
