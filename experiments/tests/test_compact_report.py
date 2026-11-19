import json
from pathlib import Path


from experiments.analysis.compute import core_flops, estimate
from experiments.analysis.report import results, clean_legacy
from experiments.analysis.analyze import collect, wilson
from experiments.scripts.robotwin_suite import parser, specifications


def test_opt_in_labels_and_environment_preserve_baseline_defaults():
    a = parser().parse_args(["guidance", "--manifest", "/tmp/report.json"])
    runs = specifications(a)
    assert all("GUIDANCE_LOOKAHEAD_MODE" not in r["environment"] for r in runs)
    a.lookahead_mode = "dino_distance"
    a.distance_increment = 0.01
    runs = specifications(a)
    selected = [r for r in runs if r["environment"]["GUIDANCE_SELECTION"] == "window"]
    assert all(
        r["label"].endswith("_dino_distance")
        and r["environment"]["GUIDANCE_DISTANCE_INCREMENT"] == "0.01"
        for r in selected
    )


def test_core_formula_counts_calls_and_separates_attention():
    w = dict(
        query_tokens=2,
        padded_queries=128,
        key_tokens=256,
        text_tokens=3,
        width=4,
        inner=4,
        ffn_width=8,
        layers=2,
        calls=3,
    )
    linear, attention = core_flops(w)
    assert linear == 2 * 3 * (12 * 2 * 4 * 4 + 4 * 3 * 4 * 4 + 4 * 2 * 4 * 8)
    assert attention == 2 * 3 * 4 * 128 * (256 + 3) * 4
    assert core_flops(w | {"key_tokens": 512})[0] == linear
    assert core_flops(w | {"key_tokens": 512})[1] > attention


def test_historical_estimate_does_not_claim_full_compute(tmp_path):
    (tmp_path / "server.log").write_text(
        "[conditioning] reset\n[ICL] cache after infer: {0: 0, 1: 756, 2: 2240}\n"
    )
    config = dict(
        episodes=[
            dict(
                episode_id=2,
                server=dict(
                    video_inference_steps=50,
                    action_inference_steps=50,
                    frame_chunk_size=2,
                    use_icl=True,
                    icl_guidance_scale=5,
                ),
            )
        ]
    )
    row = estimate(tmp_path, config)[2]
    assert row["transformer_core_linear_tflops"] > 0
    assert row["total_policy_tflops"] is None
    assert row["transformer_dense_attention_tflops"] is None
    assert row["recorded_transformer_calls"] == 101
    assert row["compute_status"] == "partial_assumed_architecture"


def test_resumed_duplicate_jobs_and_episodes_are_not_double_counted(tmp_path):
    a = parser().parse_args(
        [
            "baselines",
            "--tasks",
            "place_object_scale",
            "--seeds",
            "0",
            "--episodes",
            "2",
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )
    spec = specifications(a)[0]
    spec["job_id"] = "1"
    folder = (
        Path(spec["environment"]["EXPERIMENT_OUTPUT_ROOT"]) / "place_object_scale_1"
    )
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        json.dumps(
            dict(
                episodes=[
                    dict(
                        episode_id=0,
                        simulator_seed=100000,
                        icl_seed=0,
                        status="completed",
                        success=True,
                    )
                ]
            )
        )
    )
    manifest = dict(
        stage="baselines", runs=[spec, spec], estimate=dict(sbu_per_hour=128)
    )
    runs, episodes, _, _ = collect(manifest, None)
    assert len(runs) == len(episodes) == 1
    rows = results(runs, episodes, wilson)
    assert rows[0]["requested_episodes"] == 2 and rows[0]["missing_episodes"] == 1
    assert rows[0]["wall_seconds_mean"] is None
    assert rows[0]["total_policy_tflops"] is None


def test_cleanup_only_known_generated_files(tmp_path):
    for name in ("summary.csv", "server_chunks.png", "raw.json", "accounting.csv"):
        (tmp_path / name).write_text("data")
    clean_legacy(tmp_path)
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "server_chunks.png").exists()
    assert (tmp_path / "raw.json").exists()
    assert (tmp_path / "diagnostics/legacy-accounting.csv").exists()
