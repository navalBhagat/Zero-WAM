"""Cluster-aware independent submission; training dependencies are explicit."""
import fcntl
import os
from pathlib import Path
import subprocess

from experiments.io import read_json
from experiments.runtime.records import save_json
from experiments.paths import ROOT
from experiments.runtime.provenance import fingerprint


def active(identifier):
    result = subprocess.run(["squeue", "--noheader", "--jobs", str(identifier), "--format=%i"],
                            text=True, capture_output=True)
    if result.returncode and "invalid job id" not in result.stderr.lower():
        raise RuntimeError(f"Cannot determine job state: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def submit(directory, limit=1, retry=False):
    if limit < 1:
        raise ValueError("Submission limit must be positive")
    directory = Path(directory).resolve()
    with (directory / ".submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "manifest.json"
        manifest = read_json(path)
        if fingerprint()[0] != manifest["source_digest"]:
            raise ValueError("Code changed since planning; create a fresh campaign")
        identifiers = []
        for job in manifest["jobs"]:
            if len(identifiers) >= limit:
                break
            job_root = directory / "jobs" / job["name"]
            if read_json(job_root / "summary.json", {}).get("status") == "completed":
                continue
            attempts = job["submissions"]
            if attempts and (not retry or any(active(a["job_id"]) for a in attempts if a.get("job_id"))):
                continue
            if any(a.get("status") == "submitting" for a in attempts):
                raise ValueError("Interrupted submission: reconcile with squeue before retry")
            settings = manifest["settings"]
            command = ["sbatch", "--parsable", "--export=ALL", f"--chdir={ROOT}",
                       f"--time={round(settings['hours'] * 60)}",
                       f"--output={job_root}/slurm_%j.out", f"--job-name=noise_{job['name']}"]
            if job.get("depends_on"):
                parent = next(j for j in manifest["jobs"] if j["name"] == job["depends_on"])
                parent_done = read_json(directory / "jobs" / parent["name"] / "summary.json", {}).get("status") == "completed"
                if not parent_done:
                    submitted = [a["job_id"] for a in parent["submissions"] if a.get("job_id")]
                    if not submitted:
                        raise ValueError("Submit the training job before its evaluation jobs")
                    command.append(f"--dependency=afterok:{submitted[-1]}")
            command.append(str(ROOT / "experiments/jobs" / settings["cluster"] / "noiseicl.sbatch"))
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("NOISE_", "GUIDANCE_"))}
            environment.update(ZERO_WAM_ROOT=str(ROOT), NOISE_CAMPAIGN=str(directory),
                               NOISE_JOB=job["name"], EXPERIMENT_SOURCE_DIGEST=manifest["source_digest"])
            attempt = dict(status="submitting", command=command)
            attempts.append(attempt)
            save_json(path, manifest)
            try:
                identifier = subprocess.check_output(command, env=environment, text=True).strip().split(";")[0]
            except (OSError, subprocess.CalledProcessError) as error:
                attempt.update(status="submission_failed", error=str(error))
                save_json(path, manifest)
                raise
            attempt.update(status="submitted", job_id=identifier)
            save_json(path, manifest)
            identifiers.append(identifier)
        return identifiers
