"""One server/job per task and noise seed; scene trials remain independently resumable."""
from __future__ import annotations
import fcntl
import getpass
import json
import os
import subprocess
from pathlib import Path

from experiments.campaign.config import CONTROLLED
from experiments.campaign.provenance import fingerprint, snapshot
from experiments.io import read_json
from experiments.paths import ROOT
from .config import Trial
from .records import save_json
from .scenes import SceneSearch


def plan(directory: Path, tasks: list[str], scenes: list[int], noises: list[int],
         render_samples: int = 16, hours: float = 3.0, rate: float = 128.0,
         estimated_minutes: float = 20.0, max_scene_attempts: int = 20) -> dict:
    """Write a plan and source snapshot only; never submit implicitly."""
    if max_scene_attempts < 1:
        raise ValueError("Scene attempt limit must be positive")
    if not tasks or not scenes or not noises or min(hours, rate, estimated_minutes) <= 0:
        raise ValueError("Nonempty grids and positive timing/rate estimates required")
    if hours * 60 < estimated_minutes or not float(hours * 60).is_integer():
        raise ValueError("Walltime must be whole minutes and cover the runtime estimate")
    if any(len(values) != len(set(values)) for values in (tasks, scenes, noises)):
        raise ValueError("Duplicate tasks/seeds would repeat trial identities")
    trials = [Trial(task, scene, noise, render_samples=render_samples)
              for scene in scenes for noise in noises for task in tasks]
    if hours * 60 < estimated_minutes * len(scenes):
        raise ValueError("Job walltime must cover the estimate for every scene in the group")
    directory.mkdir(parents=True, exist_ok=False)
    digest = snapshot(directory / "source.zip")
    runs = [dict(settings=trial.to_dict(), path=str(trial.directory(directory).relative_to(directory)))
            for trial in trials]
    jobs = [dict(name=f"{task}_noise_{noise}", task=task, sampling_seed=noise,
                 run_paths=[run["path"] for run in runs if run["settings"]["task"] == task
                            and run["settings"]["sampling_seed"] == noise], submissions=[])
            for noise in noises for task in tasks]
    manifest = dict(schema_version=3, source_digest=digest, runs=runs, jobs=jobs,
                    scene_search=dict(reserved_seeds=scenes, max_attempts=max_scene_attempts),
                    hours_per_job=hours, sbu_per_hour=rate, estimated_minutes=estimated_minutes,
                    protocol="paired_gaussian_continuation_v1")
    save_json(directory / "manifest.json", manifest)
    return manifest


def active_job(job_id: str) -> bool:
    result = subprocess.run(["squeue", "--noheader", "--jobs", job_id, "--format=%i"],
                            check=True, text=True, capture_output=True)
    return bool(result.stdout.strip())


def submit(directory: Path, account: str | None = None, limit: int = 7,
           retry: bool = False) -> list[str]:
    """Submit bounded batches; allocation enforcement is delegated to Slurm."""
    if limit < 1:
        raise ValueError("Positive batch limit required")
    account_key = account or f"default-login:{getpass.getuser()}"
    with (directory / ".submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "manifest.json"
        manifest = read_json(path)
        if fingerprint()[0] != manifest["source_digest"]:
            raise ValueError("Source changed since planning; create a new campaign")
        if manifest.get("schema_version") != 3:
            raise ValueError("Create a new grouped campaign plan before submitting")
        if any(attempt.get("status") == "submitting" for run in manifest["jobs"]
               for attempt in run["submissions"]):
            raise ValueError("An interrupted submission is unresolved; check squeue before editing its status")
        pending = []
        for run in manifest["jobs"]:
            if all(read_json(directory / path / "summary.json", {}).get("status") == "completed"
                   for path in run["run_paths"]):
                continue
            attempts = run["submissions"]
            if attempts:
                if not retry or any(active_job(a["job_id"]) for a in attempts if "job_id" in a):
                    continue
            pending.append(run)
        pending = pending[:limit]
        ceiling = manifest["hours_per_job"] * manifest["sbu_per_hour"]
        identifiers = []
        previous = None
        minutes = round(manifest["hours_per_job"] * 60)
        for run in pending:
            job_root = directory / "jobs" / run["name"]
            job_root.mkdir(parents=True, exist_ok=True)
            settings_file = job_root / "settings.json"
            save_json(settings_file, {key: run[key] for key in ("name", "task", "sampling_seed", "run_paths")})
            command = ["sbatch", "--parsable", "--export=ALL", "--partition=gpu_a100",
                       f"--time={minutes}", f"--chdir={ROOT}",
                       f"--output={job_root}/slurm_%j.out", "--job-name=zw_noise_" + run["task"]]
            if account:
                command.append(f"--account={account}")
            if previous:
                command.append(f"--dependency=afterany:{previous}")
            command.append(str(ROOT / "experiments/jobs/noise_inversion_campaign.sbatch"))
            excluded = set(CONTROLLED) | {k for k in os.environ if k.startswith("NOISE_INVERSION_")}
            environment = {k: v for k, v in os.environ.items() if k not in excluded}
            environment.update(ZERO_WAM_ROOT=str(ROOT), NOISE_INVERSION_JOB_ROOT=str(job_root),
                               NOISE_INVERSION_JOB_SETTINGS=str(settings_file),
                               NOISE_INVERSION_MANIFEST=str(path),
                               NOISE_INVERSION_SOURCE_DIGEST=manifest["source_digest"],
                               EXPERIMENT_SOURCE_DIGEST=manifest["source_digest"],
                               NOISE_INVERSION_RETRY=str(int(retry)),
                               CLIENT_TIMEOUT=str(max(60, minutes * 60 - 120)),
                               TASK=run["task"],
                               EXPERIMENT_LABEL="noise_inversion_campaign")
            attempt = dict(account=account_key, requested_account=account, sbu_ceiling=ceiling, status="submitting", command=command)
            run["submissions"].append(attempt)
            save_json(path, manifest)
            try:
                job_id = subprocess.check_output(command, env=environment, text=True).strip().split(";")[0]
            except (OSError, subprocess.CalledProcessError) as error:
                attempt.update(status="submission_failed", error=str(error))
                save_json(path, manifest)
                raise
            attempt.update(status="submitted", job_id=job_id)
            save_json(path, manifest)
            identifiers.append(job_id)
            previous = job_id
        return identifiers


def run_group(campaign_root: Path, group: dict, manifest: dict, infer, *,
              retry: bool = False, runner=None) -> dict:
    """Reuse one inference connection; skip completed trials and preserve partial ones."""
    if runner is None:
        from .pilot import run_trial
        runner = run_trial
    settings = {run["path"]: run["settings"] for run in manifest["runs"]}
    policy = manifest["scene_search"]
    result = dict(job=group["name"], completed=[], skipped=[], incomplete=[])
    for path in group["run_paths"]:
        root = campaign_root / path
        if read_json(root / "summary.json", {}).get("status") == "completed":
            result["skipped"].append(path)
            continue
        trial = Trial(**settings[path])
        if trial.task != group["task"] or trial.sampling_seed != group["sampling_seed"]:
            raise ValueError("Grouped job contains a different task or noise seed")
        search = SceneSearch(root.parent / "scenes.json", trial.scene_seed,
                             tuple(policy["reserved_seeds"]), policy["max_attempts"])
        try:
            runner(root, trial, infer, digest=manifest["source_digest"], retry=retry, scene_search=search)
            result["completed"].append(path)
        except Exception as error:
            result["incomplete"].append(dict(path=path, error=str(error)))
    return result


def execute(directory: Path, settings: Path, digest: str, retry: bool, port: int) -> None:
    """Connect to the model server once and run every scene in the grouped job."""
    if fingerprint()[0] != digest:
        raise ValueError("Source changed after submission")
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
    group = read_json(settings)
    manifest_path = Path(os.environ["NOISE_INVERSION_MANIFEST"])
    manifest = read_json(manifest_path)
    planned = next(job for job in manifest["jobs"] if job["name"] == group["name"])
    if any(group[key] != planned[key] for key in ("task", "sampling_seed", "run_paths")):
        raise ValueError("Job settings differ from the campaign manifest")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    result = run_group(manifest_path.parent, group, manifest, client.infer, retry=retry)
    save_json(directory / "summary.json", result)
    if result["incomplete"]:
        raise RuntimeError(f"{len(result['incomplete'])} scene trials incomplete; retry this grouped job")
