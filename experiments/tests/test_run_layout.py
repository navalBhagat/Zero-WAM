import csv
import json
from pathlib import Path

from evaluation.robotwin.artifacts import find_artifact
from evaluation.robotwin.run_metadata import record_episode
from experiments.analysis.compute import estimate
from experiments.scripts.robotwin_suite import parser, specifications


def test_results_csv_is_readable_and_includes_actual_enabled_paths(tmp_path):
    profile = tmp_path / "diagnostics/profiles/profile_episode_0.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            dict(
                wall_seconds=100,
                timings=dict(
                    server_reset=dict(exclusive_seconds=30),
                    server_sampling_roundtrip=dict(calls=2),
                ),
            )
        )
    )
    config = tmp_path / "config.json"
    record_episode(
        config,
        0,
        status="completed",
        task="scale",
        simulator_seed=100000,
        success=False,
        policy_action_calls=40,
        video_path=str(tmp_path / "videos/episode_000.mp4"),
        profile_path=str(profile),
        server=dict(
            use_text=False,
            use_icl=False,
            use_guidance=True,
            sampling_seed=1234,
            guidance=dict(strength=0.01, every=10),
        ),
        compute=dict(
            counted_flops=10**12,
            status="estimated_all_observed_ops",
            unsupported_ops={},
            components_flops=dict(guidance_backward=4 * 10**11),
            correction_base_percent_median=0.12,
            correction_updates=10,
        ),
    )
    (row,) = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert row["preparation_seconds"] == "30" and row["rollout_seconds"] == "70"
    assert (
        row["policy_counted_tflops"] == "1.0"
        and row["policy_estimated_tflops"] == "1.0"
    )
    assert (
        row["guidance_backward_tflops"] == "0.4" and row["text_encoder_tflops"] == "0.0"
    )
    assert (
        row["correction_base_percent_median"] == "0.12"
        and row["video"] == "videos/episode_000.mp4"
    )
    assert row["use_guidance"] == "True" and row["use_text"] == "False"
    assert not (tmp_path / "eval_result").exists()


def test_partial_counters_never_claim_complete_total(tmp_path):
    record_episode(
        tmp_path / "config.json",
        0,
        status="running",
        compute=dict(
            counted_flops=10**12,
            status="partial_unsupported_ops",
            unsupported_ops={"custom.op": 1},
        ),
    )
    (row,) = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert (
        row["policy_counted_tflops"] == "1.0" and row["policy_estimated_tflops"] == ""
    )
    assert row["unsupported_op_types"] == "1"


def test_new_and_old_layout_readers(tmp_path):
    old = tmp_path / "server.log"
    old.write_text("legacy")
    assert find_artifact(tmp_path, "server.log") == old
    new = tmp_path / "diagnostics/server.log"
    new.parent.mkdir()
    new.write_text("compact")
    assert find_artifact(tmp_path, "server.log") == new


def test_operator_totals_are_used_by_analysis(tmp_path):
    path = tmp_path / "diagnostics/flops.jsonl"
    path.parent.mkdir()
    full = dict(
        counted_flops=2 * 10**12,
        status="estimated_all_observed_ops",
        unsupported_ops={},
        components_flops={"guidance_backward": 10**12},
        correction_base_percent_median=0.11,
    )
    path.write_text(json.dumps(dict(episode=0, cumulative=full)) + "\n")
    row = estimate(tmp_path, dict(episodes=[dict(episode_id=3)]))[3]
    assert row["total_policy_tflops"] == 2 and row["policy_backward_tflops"] == 1
    assert row["operator_counting_available"]


def test_suite_defaults_to_counting_and_compact_output(tmp_path):
    args = parser().parse_args(["guidance", "--manifest", str(tmp_path / "a.json")])
    for run in specifications(args):
        env = run["environment"]
        assert (
            env["COUNT_POLICY_FLOPS"] == "1"
            and env["RUN_DIAGNOSTICS"] == "0"
            and env["SAVE_GUIDANCE_CAPTURE"] == "0"
        )
    args = parser().parse_args(
        [
            "guidance",
            "--manifest",
            str(tmp_path / "b.json"),
            "--guidance-every",
            "5",
            "--diagnostics",
            "--capture-guidance",
        ]
    )
    assert all(
        r["environment"]["GUIDANCE_EVERY"] == "5"
        and r["environment"]["RUN_DIAGNOSTICS"] == "1"
        and r["environment"]["SAVE_GUIDANCE_CAPTURE"] == "1"
        for r in specifications(args)
    )


def test_compact_collection_and_compute_plot_end_to_end(tmp_path):
    from experiments.analysis.analyze import collect, wilson
    from experiments.analysis import report

    args = parser().parse_args(
        [
            "guidance",
            "--tasks",
            "place_object_scale",
            "--seeds",
            "0",
            "--episodes",
            "1",
            "--guidance-modes",
            "guidance",
            "--guidance-selectors",
            "window",
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )
    (run,) = specifications(args)
    run["job_id"] = "42"
    root = Path(run["environment"]["EXPERIMENT_OUTPUT_ROOT"]) / "place_object_scale_42"
    profile = root / "diagnostics/profiles/profile_episode_0.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            dict(
                wall_seconds=120,
                timings=dict(
                    server_reset=dict(exclusive_seconds=30),
                    server_sampling_roundtrip=dict(exclusive_seconds=50, calls=1),
                ),
            )
        )
    )
    record_episode(
        root / "config.json",
        0,
        status="completed",
        task="place_object_scale",
        success=False,
        simulator_seed=100000,
        icl_seed=0,
        policy_action_calls=20,
        profile_path=str(profile),
        server=dict(
            sampling_seed=1234,
            use_icl=False,
            use_text=False,
            use_guidance=True,
            video_inference_steps=50,
        ),
        compute=dict(
            counted_flops=2 * 10**12,
            status="estimated_all_observed_ops",
            unsupported_ops={},
            components_flops={"guidance_backward": 10**12},
            correction_base_percent_median=0.1,
        ),
    )
    runs, episodes, _, _ = collect(
        dict(stage="guidance", runs=[run], estimate=dict(sbu_per_hour=128)), None
    )
    (row,) = episodes
    assert (
        row["total_policy_tflops"] == 2 and row["correction_base_percent_median"] == 0.1
    )
    assert row["preparation_seconds"] == 30
    summary = report.results(runs, episodes, wilson)
    assert summary[0]["total_policy_tflops"] == 2
    report.plots(tmp_path, summary)
    assert (tmp_path / "success.png").is_file() and (tmp_path / "cost.png").is_file()
