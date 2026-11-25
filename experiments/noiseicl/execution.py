"""Run a job against a live server, with durable outcomes and explicit retries."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.io import read_json
from experiments.runtime.records import read_episode, save_json
from experiments.paths import ROOT


def archive(directory):
    if directory.exists():
        target = directory.parent / "attempts" / f"{directory.name}_{time.time_ns()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(target)


def evaluate(campaign, manifest, job):
    settings = manifest["settings"]
    trial = campaign / "trials" / job["task"] / f"scene_{job['scene']}_noise_{settings['sampling_seed']}"
    trial.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    with (trial / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seeds = job.get("reserved_scenes", list(range(settings["scene_start"], settings["scene_start"] + settings["scene_count"])))
        accepted = None
        instruction = None
        for method in job["methods"]:
            directory = trial / method
            try:
                episode = read_episode(directory)
            except (OSError, ValueError, KeyError):
                archive(directory)
                directory.mkdir(parents=True)
                env = dict(os.environ, NOISE_METHOD="learned" if method == "learned_shuffled" else method, NOISE_OUTPUT=str(directory / "noise"),
                    TEST_NUM="1", EPISODE_OFFSET="0", START_SEED=str(job["scene"]), SEED="0",
                    SAMPLING_SEED=str(settings["sampling_seed"]), ICL_SEED="0",
                    USE_ICL="1", USE_TEXT="1" if method == "icl" else "0", TEXT_GUIDANCE_SCALE="1", ICL_GUIDANCE_SCALE="5",
                    NOISE_INVERSION_SCENE_REGISTRY=str(trial.parent / "scenes.json"),
                    NOISE_INVERSION_RESERVED_SCENES=",".join(map(str, seeds)), NOISE_INVERSION_MAX_SCENE_ATTEMPTS="20",
                    NOISE_INVERSION_STEP_TIMEOUT="90")
                if instruction is not None:
                    env["NOISE_INSTRUCTION"] = instruction
                if job.get("demo_latent"):
                    env["NOISE_DEMO_LATENT"] = job["demo_latent"]
                if method == "learned_shuffled":
                    env["NOISE_SHUFFLE_DEMO"] = "1"
                    if job.get("shuffled_demo_latent"):
                        env["NOISE_DEMO_LATENT"] = job["shuffled_demo_latent"]
                else:
                    env.pop("NOISE_SHUFFLE_DEMO", None)
                if method in ("learned", "learned_shuffled"):
                    env["NOISE_ADAPTER"] = str(campaign / "jobs/train/adapter.pt")
                try:
                    with (directory / "client.log").open("w") as log:
                        subprocess.run(["bash", str(ROOT / "evaluation/robotwin/launch_client.sh"),
                                        str(directory), job["task"]], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True, timeout=3600)
                    episode = read_episode(directory)
                except Exception as error:
                    outcomes[method] = dict(status="incomplete", error=str(error))
                    break
            if accepted is None:
                accepted, instruction = episode["simulator_seed"], episode["task_instruction"]
            if (episode["simulator_seed"], episode["task_instruction"]) != (accepted, instruction):
                raise ValueError("Comparison methods do not share the same scene/instruction")
            metadata = episode["server"]
            expected_method = "learned" if method == "learned_shuffled" else method
            if (metadata.get("noise_method") != expected_method or
                    metadata["use_text"] != (method == "icl") or
                    metadata.get("cached_caption_used", False) != (method == "icl") or
                    metadata.get("model_text") != "" or metadata.get("target_text_source") != "empty"):
                raise ValueError("Unexpected method/caption/instruction conditioning")
            if method != "icl" and metadata["use_icl"]:
                raise ValueError("Student received ICL context")
            records = [json.loads(line) for line in (directory / "noise/chunks.jsonl").read_text().splitlines()]
            errors = [row["reconstruction_error"] for row in records if row["reconstruction_error"] is not None]
            if len(records) != episode["rollout_chunks"]:
                raise ValueError("Chunk diagnostics do not cover the completed rollout")
            outcomes[method] = dict(status="completed", success=episode["success"],
                chunks=len(records), actions=episode.get("simulator_action_count"),
                mean_reconstruction_error=sum(errors) / len(errors) if errors else None,
                max_reconstruction_error=max(errors) if errors else None, fallback=False)
            save_json(trial / "summary.json", dict(task=job["task"], scene=accepted, methods=outcomes))
        complete = all(outcomes.get(method, {}).get("status") == "completed" for method in job["methods"])
        summary = dict(status="completed" if complete else "incomplete", task=job["task"], scene=accepted, methods=outcomes)
        save_json(trial / "summary.json", summary)
        return summary


def collect(campaign, job, client, source_digest):
    from evaluation.robotwin.watchdog import guarded_call
    os.environ["NOISE_INVERSION_STEP_TIMEOUT"] = os.environ.get("NOISE_RPC_TIMEOUT", "900")
    outcomes = []
    for pair in job["pairs"]:
        directory = campaign / "pairs" / pair["id"]
        record_path = directory / "record.json"
        if read_json(record_path, {}).get("status") == "completed":
            outcomes.append(read_json(record_path))
            continue
        archive(directory)
        directory.mkdir(parents=True)
        try:
            if hashlib.sha256(Path(pair["context"]).read_bytes()).hexdigest() != pair["context_sha256"]:
                raise ValueError("Recorded context changed after planning")
            result = guarded_call(f"collect {pair['id']}", client.infer,
                dict(collect_context=pair["context"], output=str(directory), seed=pair["seed"]))["pair"]
            row = dict(pair, **{key: value for key, value in result.items() if key not in pair}, status="completed", source_digest=source_digest)
        except Exception as error:
            row = dict(pair, status="incomplete", error=str(error))
        save_json(record_path, row)
        outcomes.append(row)
        if row["status"] != "completed":
            break
    return dict(status="completed" if all(row["status"] == "completed" for row in outcomes) else "incomplete",
                pairs=outcomes)


def execute(campaign, name):
    from experiments.runtime.provenance import fingerprint
    campaign = Path(campaign).resolve()
    manifest = read_json(campaign / "manifest.json")
    if manifest["source_digest"] != fingerprint()[0]:
        raise ValueError("Runtime differs from the planned snapshot")
    job = next(row for row in manifest["jobs"] if row["name"] == name)
    output = campaign / "jobs" / name
    started = time.perf_counter()
    if job["kind"] == "evaluate":
        summary = evaluate(campaign, manifest, job)
    elif job["kind"] == "collect":
        from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
        summary = collect(campaign, job, WebsocketClientPolicy(host="127.0.0.1", port=int(os.environ["PORT"])), manifest["source_digest"])
    else:
        from .training import train
        summary = train(Path(job["dataset"]), output, manifest["settings"])
    summary["seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    if summary["status"] != "completed":
        raise RuntimeError("Job has incomplete results; inspect summary.json")
