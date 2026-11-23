"""Three capture-only checks in existing incomplete trial folders; no submission."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.provenance import fingerprint, snapshot
from experiments.io import read_json
from experiments.noise_inversion.config import Trial
from experiments.noise_inversion.pilot import run_trial
from experiments.noise_inversion.records import save_json
from experiments.noise_inversion.scenes import SceneSearch

TARGETS = (("move_stapler_pad", 100002, 1234),
           ("stack_blocks_three", 100000, 1234),
           ("place_object_scale", 100001, 1234))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--job-output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "29056")))
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = read_json(root / "manifest.json")
    policy = manifest["scene_search"]
    output = args.job_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = snapshot(output / "repair_source.zip")
    if os.environ.get("EXPERIMENT_SOURCE_DIGEST") != digest:
        raise ValueError("Repair source changed while the server was starting")
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    results = []
    for task, scene, noise in TARGETS:
        trial = Trial(task, scene, noise)
        directory = trial.directory(root)
        if not directory.exists():
            raise ValueError(f"Expected original trial directory: {directory}")
        if read_json(directory / "summary.json", {}).get("status") == "completed":
            results.append(dict(task=task, status="skipped_completed"))
            continue
        search = SceneSearch(directory.parent / "scenes.json", scene,
                             tuple(policy["reserved_seeds"]), policy["max_attempts"])
        try:
            if fingerprint()[0] != digest:
                raise ValueError("Source changed during the repair pilot")
            summary = run_trial(directory, trial, client.infer, digest=digest,
                                retry=True, scene_search=search, capture_only=True, repair=True)
            results.append(dict(task=task, status="capture_completed", source_success=summary["source_success"]))
        except Exception as error:
            results.append(dict(task=task, status="error", error=str(error)))
        save_json(output / "repair_summary.json", dict(capture_only=True, results=results))
    save_json(output / "repair_summary.json", dict(capture_only=True, results=results))
    if any(result["status"] == "error" for result in results):
        raise RuntimeError("Repair pilot has errors; inspect repair_summary.json and trial capture/client.log")


if __name__ == "__main__":
    main()
