"""Run each phase sequentially using the existing RoboTwin client and server."""
import json
import os
import subprocess
from pathlib import Path

TASKS = ("place_bread_basket", "place_object_scale")


def read_episode(directory: Path) -> dict:
    episodes = json.loads((directory / "config.json").read_text())["episodes"]
    if len(episodes) != 1 or episodes[0].get("status") != "completed":
        raise RuntimeError("Expected exactly one completed pilot episode")
    return episodes[0]


def run_pilot(output: Path, tasks: list[str], sampling_seed: int, simulator_seed: int,
              infer, launch) -> None:
    """Dependencies are explicit so sequencing can be tested without a simulator."""
    output.mkdir(parents=True, exist_ok=False)
    for task in tasks:
        root = output / task
        root.mkdir()
        environment = dict(os.environ, TEST_NUM="1", SEED=str(simulator_seed), ICL_SEED="0",
                           EPISODE_OFFSET="0", START_SEED=str(100000 * (1 + simulator_seed)),
                           SAMPLING_SEED=str(sampling_seed), USE_GUIDANCE="0", USE_TEXT="1",
                           TEXT_GUIDANCE_SCALE="1", ICL_GUIDANCE_SCALE="5",
                           NOISE_INVERSION_DIRECTORY=str(root))
        launch(root / "capture", task, dict(environment, USE_ICL="1", NOISE_INVERSION_MODE="capture"))
        source = read_episode(root / "capture")
        summary = dict(task=task, replay_use_text=False, replay_use_icl=False, source_success=source["success"],
                       simulator_seed=source["simulator_seed"], sampling_seed=sampling_seed)
        if source["success"]:
            result = infer(dict(invert_episode=str(root)))
            summary["inversion"] = result["inversion"]
            launch(root / "replay", task, dict(environment, USE_ICL="0", USE_TEXT="0", NOISE_INVERSION_MODE="replay"))
            replay = read_episode(root / "replay")
            summary.update(replay_success=replay["success"],
                           noise_sequence_exhausted=replay.get("noise_sequence_exhausted", False),
                           replay_chunks=replay["rollout_chunks"], source_chunks=source["rollout_chunks"])
        else:
            summary["status"] = "source_failed_inversion_skipped"
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def launch_client(directory: Path, task: str, environment: dict) -> None:
    from experiments.paths import ROOT
    subprocess.run(["bash", str(ROOT / "evaluation/robotwin/launch_client.sh"),
                    str(directory), task], env=environment, cwd=ROOT, check=True)
