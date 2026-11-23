"""Sequential, resumable capture/control/inversion/replay with explicit artifacts."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from dataclasses import replace
from .config import Trial
from .records import read_episode, save_json, summarize, validate_pair, validate_chunks, validate_source
from .scenes import SceneSearch


def launch_client(directory: Path, task: str, environment: dict) -> None:
    from experiments.paths import ROOT
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "client.log").open("a") as log:
        subprocess.run(["bash", str(ROOT / "evaluation/robotwin/launch_client.sh"),
                        str(directory), task], env=environment, cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=float(environment.get("NOISE_INVERSION_PHASE_TIMEOUT", "1800")))


def phase_environment(root: Path, trial: Trial, phase: str) -> dict[str, str]:
    """Override inherited experimental switches that could contaminate a pair."""
    capture = phase == "capture"
    return dict(os.environ, TEST_NUM="1", SEED="0", EPISODE_OFFSET="0",
                START_SEED=str(trial.scene_seed), SAMPLING_SEED=str(trial.sampling_seed),
                ICL_SEED=str(trial.icl_seed), USE_ICL=str(int(capture)), USE_TEXT=str(int(capture)),
                USE_GUIDANCE="0", TEXT_GUIDANCE_SCALE="1", ICL_GUIDANCE_SCALE="5" if capture else "1",
                RENDER_SAMPLES_PER_PIXEL=str(trial.render_samples),
                NOISE_INVERSION_MODE=phase, NOISE_INVERSION_DIRECTORY=str(root),
                NOISE_INVERSION_CONTINUATION="gaussian", NOISE_INVERSION_STRICT_SCENE="1",
                NOISE_INVERSION_STEP_TIMEOUT=os.environ.get("NOISE_INVERSION_STEP_TIMEOUT", "90"))


def archive_incomplete(root: Path, names: tuple[str, ...]) -> None:
    """Preserve failed attempts instead of mixing them with new tensors/logs."""
    present = [root / name for name in names if (root / name).exists()]
    if not present:
        return
    archive = root / "attempts" / f"retry_{time.time_ns()}"
    archive.mkdir(parents=True)
    for path in present:
        path.rename(archive / path.name)


def ensure_rollout(root: Path, trial: Trial, phase: str, launch, retry: bool,
                   scene_search: SceneSearch | None = None) -> dict:
    folder = root / phase
    if folder.exists():
        try:
            completed = read_episode(folder)
            if phase == "capture":
                validate_chunks(root / "chunks", completed["rollout_chunks"])
            return completed
        except (OSError, ValueError, KeyError):
            if not retry:
                raise ValueError(f"Incomplete {phase} in {root}; use --retry-incomplete")
            names = (phase, "control", "replay", "chunks", "history", "inverted", "inversion.json") if phase == "capture" else (phase,)
            archive_incomplete(root, names)
    environment = phase_environment(root, trial, phase)
    if scene_search is not None and phase == "capture":
        environment.update(scene_search.environment())
    launch(folder, trial.task, environment)
    completed = read_episode(folder)
    if phase == "capture":
        validate_chunks(root / "chunks", completed["rollout_chunks"])
    return completed


def ensure_inversion(root: Path, count: int, infer, retry: bool) -> None:
    """Reuse only complete inversions; archive partial numerical results on retry."""
    inversion_path = root / "inversion.json"
    if inversion_path.exists():
        try:
            validate_chunks(root / "inverted", count)
            inversion = json.loads(inversion_path.read_text())
            if inversion["chunks"] != count:
                raise ValueError("Inversion report chunk count mismatch")
        except (ValueError, KeyError):
            if not retry:
                raise ValueError("Incomplete inversion; use --retry-incomplete")
            archive_incomplete(root, ("inverted", "inversion.json"))
    if not inversion_path.exists():
        if (root / "inverted").exists():
            if not retry:
                raise ValueError("Partial inversion; use --retry-incomplete")
            archive_incomplete(root, ("inverted",))
        result = infer(dict(invert_episode=str(root)))
        validate_chunks(root / "inverted", count)
        save_json(inversion_path, result["inversion"])


def run_trial(root: Path, trial: Trial, infer, launch=launch_client, *, digest: str,
              retry: bool = False, scene_search: SceneSearch | None = None,
              capture_only: bool = False, repair: bool = False) -> dict:
    """Resume only compatible trials; never gate numerical inversion on success."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if scene_search is not None:
            scene_search = replace(scene_search, receipt=root / "capture" / "scene_selection.json")
        identity = dict(settings=trial.to_dict(), source_digest=digest, protocol="paired_gaussian_continuation_v1")
        if scene_search is not None:
            identity["scene_search"] = scene_search.environment()
        identity_path = root / "trial.json"
        if repair:
            summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else {}
            if summary.get("status") == "completed":
                raise ValueError("Repair refuses to overwrite a completed paired trial")
            archive_incomplete(root, ("trial.json", "summary.json", "capture", "control", "replay",
                                      "chunks", "history", "inverted", "inversion.json"))
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Trial settings/source changed; use a new campaign directory")
        save_json(identity_path, identity)
        started = time.perf_counter()
        try:
            source = ensure_rollout(root, trial, "capture", launch, retry, scene_search)
            accepted_seed = scene_search.accepted_seed() if scene_search is not None else trial.scene_seed
            paired = Trial(trial.task, accepted_seed, trial.sampling_seed, trial.icl_seed, trial.render_samples)
            validate_source(source, paired)
            if capture_only:
                summary = dict(status="incomplete", pilot_capture_complete=True, task=trial.task,
                               source_success=source["success"], simulator_seed=accepted_seed,
                               sampling_seed=trial.sampling_seed, requested_scene_seed=trial.scene_seed,
                               error="Capture-only safety pilot; control/inversion/replay not run")
                save_json(root / "summary.json", summary)
                return summary
            control = ensure_rollout(root, paired, "control", launch, retry)
            validate_pair(source, control)
            ensure_inversion(root, source["rollout_chunks"], infer, retry)
            ensure_rollout(root, paired, "replay", launch, retry)
            summary = summarize(root)
            summary.update(requested_scene_seed=trial.scene_seed,
                           last_attempt_seconds=time.perf_counter() - started)
            save_json(root / "summary.json", summary)
            return summary
        except Exception as error:
            save_json(root / "summary.json", dict(status="incomplete", task=trial.task,
                      requested_scene_seed=trial.scene_seed, sampling_seed=trial.sampling_seed,
                      error=str(error), last_attempt_seconds=time.perf_counter() - started))
            raise


def run_pilot(output: Path, tasks: list[str], sampling_seed: int, simulator_seed: int,
              infer, launch=launch_client) -> None:
    """Compatibility entry point; new campaigns use explicit actual scene seeds."""
    from experiments.campaign.provenance import fingerprint
    output.mkdir(parents=True, exist_ok=False)
    digest, _ = fingerprint()
    for task in tasks:
        trial = Trial(task, 100000 * (1 + simulator_seed), sampling_seed)
        search = SceneSearch(output / task / "scenes.json", trial.scene_seed, (trial.scene_seed,))
        run_trial(output / task, trial, infer, launch, digest=digest, scene_search=search)


def run_controls(source: Path, tasks: list[str], launch=launch_client) -> None:
    """Append controls to historical pilots without changing their source/replay."""
    for task in tasks:
        root = source / task
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        episode = read_episode(root / "capture")
        if (root / "control").exists():
            raise FileExistsError(root / "control")
        trial = Trial(task, episode["simulator_seed"], summary["sampling_seed"],
                      episode["icl_seed"], episode["rendering"]["samples_per_pixel_effective"])
        launch(root / "control", task, phase_environment(root, trial, "control"))
        control = read_episode(root / "control")
        validate_pair(episode, control)
        extended = control["rollout_chunks"] > episode["rollout_chunks"]
        summary.update(control_success=control["success"], control_chunks=control["rollout_chunks"],
                       control_success_within_source_horizon=control["success"] and not extended,
                       control_extended_beyond_saved_noise=extended)
        save_json(summary_path, summary)
