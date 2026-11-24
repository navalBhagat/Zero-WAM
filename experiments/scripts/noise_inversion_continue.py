"""Continue the completed stack diagnostic capture; never launch a new capture."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.provenance import fingerprint, snapshot
from experiments.io import read_json
from experiments.noise_inversion.config import Trial
from experiments.noise_inversion.pilot import ensure_rollout, ensure_inversion, launch_client
from experiments.noise_inversion.records import read_episode, save_json, summarize, validate_chunks, validate_pair, validate_source


def validate_runtime(source_snapshot, source_digest, current_hashes):
    """Allow launcher/diagnostic edits, but require identical capture/model code."""
    with zipfile.ZipFile(source_snapshot) as archive:
        recorded = json.loads(archive.read("source_hashes.json"))
    digest = hashlib.sha256(json.dumps(recorded, sort_keys=True).encode()).hexdigest()
    if digest != source_digest:
        raise ValueError("Snapshot does not belong to the saved capture")
    def scientific(name):
        return name.startswith(("wan_va/", "evaluation/", "experiments/noise_inversion/")) and name != "evaluation/robotwin/camera_diagnostics.py"
    changed = [name for name in sorted(recorded.keys() | current_hashes.keys())
               if scientific(name) and recorded.get(name) != current_hashes.get(name)]
    if changed:
        raise ValueError("Capture runtime changed; refusing mixed-code continuation: " + ", ".join(changed))


def continue_trial(root, infer, *, digest, launch=launch_client):
    """Keep the original identity/capture; record continuation provenance separately."""
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = read_episode(root / "capture")
        identity = read_json(root / "trial.json")
        requested = Trial(**identity["settings"])
        paired = Trial(requested.task, source["simulator_seed"], requested.sampling_seed,
                       requested.icl_seed, requested.render_samples)
        validate_source(source, paired)
        validate_chunks(root / "chunks", source["rollout_chunks"])
        validate_chunks(root / "history", source["rollout_chunks"])
        record_path = root / "continuation.json"
        record = dict(source_digest=identity["source_digest"], continuation_digest=digest,
                      allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                      capture_reused=True, settings=requested.to_dict())
        previous = read_json(record_path, {})
        if previous and previous != record:
            raise ValueError("Continuation settings/source changed; inspect the previous attempt first")
        save_json(record_path, record)
        started = time.perf_counter()
        try:
            control = ensure_rollout(root, paired, "control", launch, retry=True)
            validate_pair(source, control)
            ensure_inversion(root, source["rollout_chunks"], infer, retry=True)
            ensure_rollout(root, paired, "replay", launch, retry=True)
            summary = summarize(root)
            summary.update(requested_scene_seed=requested.scene_seed, capture_reused=True,
                           continuation_digest=digest, last_attempt_seconds=time.perf_counter() - started)
            save_json(root / "summary.json", summary)
            return summary
        except Exception as error:
            save_json(root / "summary.json", dict(status="incomplete", task=requested.task,
                      source_success=source["success"], capture_reused=True,
                      requested_scene_seed=requested.scene_seed, sampling_seed=requested.sampling_seed,
                      error=str(error), continuation_digest=digest))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--job-output", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Trial("stack_blocks_three", 100000, 1234).directory(args.campaign.resolve())
    identity = read_json(root / "trial.json")
    digest, hashes = fingerprint()
    if digest != os.environ.get("EXPERIMENT_SOURCE_DIGEST"):
        raise ValueError("Code changed while the continuation server was starting")
    validate_runtime(args.source_snapshot, identity["source_digest"], hashes)
    source = read_episode(root / "capture")
    for name in ("chunks", "history"):
        validate_chunks(root / name, source["rollout_chunks"])
    if args.check_only:
        print(f"Validated saved capture: {source['rollout_chunks']} source/history chunks; compatible runtime")
        return
    args.job_output.mkdir(parents=True, exist_ok=True)
    if snapshot(args.job_output / "continuation_source.zip") != digest:
        raise ValueError("Code changed during source snapshot")
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host="127.0.0.1", port=int(os.environ["PORT"]))
    summary = continue_trial(root, client.infer, digest=digest)
    save_json(args.job_output / "continuation_summary.json", summary)


if __name__ == "__main__":
    main()
