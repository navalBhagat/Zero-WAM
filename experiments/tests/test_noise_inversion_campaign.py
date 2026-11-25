"""Five-stage plans, leakage boundaries and bounded Slurm submission."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.noiseicl.config import Settings
from experiments.noiseicl.dataset import index_contexts
from experiments.noiseicl.planning import plan
from experiments.noiseicl.submission import submit
from experiments.noiseicl.reporting import report
from experiments.runtime.records import save_json


def trajectory_index(root):
    records = []
    for task in range(3):
        for number in range(3):
            source = root / f"source_{task}_{number}"
            (source / "history").mkdir(parents=True)
            for name in ("initial.pt", "demo.pt", "history/chunk_0000.pt"):
                (source / name).touch()
            records.append(dict(task=f"seen_{task}", trajectory_id=str(number), scene_id=str(number),
                simulator_seed=100000 + number, demo_id=str(number), demo_latent=str(source / "demo.pt"),
                initial=str(source / "initial.pt"), history=str(source / "history"), chunk_count=2))
    manifest = root / "trajectories.jsonl"
    manifest.write_text("\n".join(map(json.dumps, records)))
    index_contexts(manifest, root / "index", points=2, task_limit=3, trajectories_per_task=3)
    return root / "index/index.json"


def test_five_campaigns_and_training_dependencies(tmp_path):
    contexts = trajectory_index(tmp_path)
    with patch("experiments.noiseicl.planning.snapshot", side_effect=lambda path: path.write_text("snapshot") and "digest"):
        for stage, count in ((1, 7), (2, 140), (3, 2), (4, 2)):
            manifest = plan(tmp_path / f"stage{stage}", Settings(stage=stage,
                scene_count=20 if stage == 2 else 1), contexts=contexts if stage > 2 else None)
            assert len(manifest["jobs"]) == count
            if stage == 4:
                seeds = [pair["seed"] for item in manifest["jobs"] for pair in item["pairs"]]
                assert len(seeds) == len(set(seeds))
                for job in manifest["jobs"]:
                    save_json(tmp_path / "stage4/jobs" / job["name"] / "summary.json", dict(status="completed"))
                    for pair in job["pairs"]:
                        save_json(tmp_path / "stage4/pairs" / pair["id"] / "record.json",
                                  dict(pair, status="completed", reconstruction_error=.01, seconds=1, fallback=False))
            report(tmp_path / f"stage{stage}")
        manifest = plan(tmp_path / "stage5", Settings(stage=5), dataset=tmp_path / "stage4")
    assert manifest["jobs"][0]["kind"] == "train"
    assert any(job["name"].startswith("seen_") for job in manifest["jobs"])
    with patch("experiments.noiseicl.submission.fingerprint", return_value=("digest", {})), \
         patch("experiments.noiseicl.submission.subprocess.check_output", side_effect=["123", "124"]) as sbatch:
        assert submit(tmp_path / "stage5", limit=2) == ["123", "124"]
    assert "--dependency=afterok:123" in sbatch.call_args_list[1].args[0]


def test_unseen_tasks_cannot_enter_context_index(tmp_path):
    source = tmp_path / "bad.jsonl"
    row = dict(task="open_microwave", trajectory_id="1", scene_id="1", simulator_seed=1,
               demo_id="1", demo_latent="x", initial="x", history="x", chunk_count=2)
    source.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Unseen task"):
        index_contexts(source, tmp_path / "index", task_limit=1)
