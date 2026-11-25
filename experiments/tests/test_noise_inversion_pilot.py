"""Closed-loop campaign sequencing treats task failure as a completed trial."""
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from experiments.noiseicl.config import Settings
from experiments.noiseicl.execution import evaluate
from experiments.runtime.records import save_json


def trial_job():
    return dict(name="cup", task="place_empty_cup", scene=100000,
                methods=["icl", "control", "oracle"])


def completed_client(command, *, env, **kwargs):
    output = Path(command[-2])
    method = env["NOISE_METHOD"]
    save_json(output / "config.json", dict(episodes=[dict(status="completed", success=False,
        simulator_seed=100001, task_instruction=env.get("NOISE_INSTRUCTION", "Put the cup down"),
        rollout_chunks=2, simulator_action_count=32,
        server=dict(noise_method=method, use_text=method == "icl", use_icl=method == "icl",
                    cached_caption_used=method == "icl", model_text="", target_text_source="empty"))]))
    noise = output / "noise"
    noise.mkdir(exist_ok=True)
    (noise / "chunks.jsonl").write_text("\n".join(json.dumps(dict(reconstruction_error=.01 if method == "oracle" else None)) for _ in range(2)))


def test_failed_icl_episode_still_runs_control_and_online_oracle_and_resumes(tmp_path):
    manifest = dict(settings=Settings(stage=1).to_dict())
    with patch("experiments.noiseicl.execution.subprocess.run", side_effect=completed_client) as client:
        result = evaluate(tmp_path, manifest, trial_job())
        assert result["status"] == "completed"
        assert len(client.call_args_list) == 3
        for call in client.call_args_list[1:]:
            assert call.kwargs["env"]["NOISE_INSTRUCTION"] == "Put the cup down"
        evaluate(tmp_path, manifest, trial_job())
        assert len(client.call_args_list) == 3
    assert all(not row["success"] for row in result["methods"].values())


def test_runtime_error_stops_before_spending_on_remaining_methods(tmp_path):
    def launch(command, **kwargs):
        if kwargs["env"]["NOISE_METHOD"] == "control":
            raise subprocess.CalledProcessError(1, command)
        completed_client(command, **kwargs)
    with patch("experiments.noiseicl.execution.subprocess.run", side_effect=launch) as client:
        result = evaluate(tmp_path, dict(settings=Settings(stage=1).to_dict()), trial_job())
    assert result["status"] == "incomplete"
    assert len(client.call_args_list) == 2
    assert "oracle" not in result["methods"]
