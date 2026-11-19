import json
from pathlib import Path

import pytest

from experiments.scripts.provenance import ROOT, snapshot
from experiments.scripts.robotwin_suite import extend_baselines, parser, specifications
from experiments.analysis.analyze import collect


def test_extend_pilot_preserves_scene_and_demo_offsets_and_analysis(tmp_path):
    manifest = tmp_path / "pilot.json"
    a = parser().parse_args(
        [
            "baselines",
            "--seeds",
            "0",
            "--episodes",
            "2",
            "--spp",
            "8",
            "--manifest",
            str(manifest),
        ]
    )
    old = specifications(a)
    bundle = tmp_path / "pilot.source.zip"
    snapshot(bundle)
    for index, spec in enumerate(old):
        spec["job_id"] = str(index)
        folder = (
            Path(spec["environment"]["EXPERIMENT_OUTPUT_ROOT"])
            / f"{spec['task']}_{index}"
        )
        folder.mkdir(parents=True)
        # Include a skipped simulator seed and an unsuccessful completed episode.
        episodes = [
            dict(
                episode_id=i,
                simulator_seed=seed,
                status="completed",
                success=bool(i),
                icl_seed=0,
            )
            for i, seed in enumerate([100000, 100003])
        ]
        (folder / "config.json").write_text(json.dumps(dict(episodes=episodes)))
        (folder / "job_timing.json").write_text(
            json.dumps(dict(exit_code=0, allocated_seconds=10))
        )
    manifest.write_text(
        json.dumps(dict(stage="baselines", source_bundle=str(bundle), runs=old))
    )
    a.seeds = [0, 1, 2]
    a.episodes = 10
    pending, reused = extend_baselines(specifications(a), manifest, ROOT)
    assert len(pending) == 12 and len(reused) == 4
    assert sum(int(s["environment"]["TEST_NUM"]) for s in pending) == 112
    for spec in pending:
        env = spec["environment"]
        if spec["seed"] == 0:
            assert env["EPISODE_OFFSET"] == "2" and env["START_SEED"] == "100004"
            assert env["TEST_NUM"] == "8"
        else:
            assert "START_SEED" not in env and env["TEST_NUM"] == "10"
    _, episodes, _, _ = collect(
        dict(stage="baselines", runs=reused + pending, estimate=dict(sbu_per_hour=128)),
        None,
    )
    assert len(episodes) == 8 and sum(e["success"] for e in episodes) == 4
    # Changed rendering must not silently reuse a different experiment.
    a.spp = 16
    fresh, reused = extend_baselines(specifications(a), manifest, ROOT)
    assert not reused and sum(int(s["environment"]["TEST_NUM"]) for s in fresh) == 120
    # Gaps cannot be continued safely by a single episode offset.
    folder = (
        Path(old[0]["environment"]["EXPERIMENT_OUTPUT_ROOT"]) / f"{old[0]['task']}_0"
    )
    config = json.loads((folder / "config.json").read_text())
    config["episodes"][1]["episode_id"] = 2
    (folder / "config.json").write_text(json.dumps(config))
    a.spp = 8
    with pytest.raises(ValueError, match="gaps"):
        extend_baselines(specifications(a), manifest, ROOT)


def test_resume_allows_only_exact_scene_skip_change():
    import subprocess
    from experiments.scripts.robotwin_suite import compatible_client_changes

    old = subprocess.check_output(
        ["git", "show", "7085191:evaluation/robotwin/eval_policy_client_openpi.py"],
        text=True,
    )
    # Locate the historical handler in the current client without relying on surrounding edits.
    current = (ROOT / "evaluation/robotwin/eval_policy_client_openpi.py").read_text()
    start = current.index(
        "            except Exception as e:",
        current.index("while succ_seed < test_num:"),
    )
    end = current.index("\n        if (not expert_check)", start)
    new_handler = current[start:end].rstrip() + "\n"
    old_start = old.index(
        "            except Exception as e:", old.index("while succ_seed < test_num:")
    )
    old_end = old.index("\n        if (not expert_check)", old_start)
    old_handler = old[old_start:old_end].rstrip() + "\n"
    before = "unchanged prefix\n" + old_handler + "unchanged suffix\n"
    after = "unchanged prefix\n" + new_handler + "unchanged suffix\n"
    assert compatible_client_changes(before, after)
    assert not compatible_client_changes(
        before, after.replace("unchanged suffix", "changed policy")
    )
    assert not compatible_client_changes(
        before, after.replace("now_seed += 1", "now_seed += 2")
    )


def test_resume_allows_exact_output_relocation_but_not_extra_edits():
    from experiments.scripts.robotwin_suite import compatible_client_changes

    before = '    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")'
    after = "    save_dir = Path(save_root).resolve() / 'eval_result' / task_name / policy_name / task_config / str(ckpt_setting) / current_time"
    assert compatible_client_changes(before, after) == ["job_local_auxiliary_outputs"]
    assert not compatible_client_changes(before, after + "\nchanged_inference()")
