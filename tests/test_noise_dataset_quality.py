"""Dataset integrity checks and training-only planning use actual small tensors."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from experiments.noiseicl.config import Settings
from experiments.noiseicl.planning import plan
from experiments.noiseicl.reporting import quality_report
from wan_va.noise_inversion.quality import inspect_dataset
from wan_va.noise_inversion.records import read_json, save_json


@pytest.fixture
def collection(tmp_path):
    root = tmp_path / "collection"
    planned = []
    for index, split in enumerate(("train", "val", "test")):
        demo = tmp_path / f"demo_{index}.pt"
        torch.save(dict(latent=torch.zeros(1, 48, 2, 4, 4)), demo)
        initial = tmp_path / f"initial_{index}.pt"
        torch.save(dict(initial_frame=torch.zeros(1, 48, 1, 4, 4)), initial)
        context = tmp_path / f"context_{index}.json"
        save_json(
            context,
            dict(
                task="adjust_bottle",
                trajectory_id=str(index),
                scene_id=str(index),
                simulator_seed=None,
                demo_id=str(index),
                demo_latent=str(demo),
                initial=str(initial),
                history=[],
                chunk=0,
                split=split,
            ),
        )
        for seed in (10, 20):
            pair_id = f"{index}_{seed}"
            metadata = dict(
                id=pair_id,
                context=str(context),
                split=split,
                seed=seed,
                context_sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
            )
            planned.append(metadata)
            destination = root / "pairs" / pair_id
            destination.mkdir(parents=True)
            original = torch.ones(1, 48, 2, 4, 4)
            mask = torch.ones_like(original)
            mask[:, :, :1] = 0
            pair_path = destination / "pair.pt"
            torch.save(
                dict(
                    original=original,
                    recovered=original * 0.9,
                    target=original,
                    mask=mask,
                ),
                pair_path,
            )
            save_json(
                destination / "record.json",
                dict(
                    metadata,
                    pair=str(pair_path),
                    status="completed",
                    fallback=False,
                    source_digest="collected-source",
                    reconstruction_error=(index + 1) / 100,
                ),
            )
    save_json(
        root / "manifest.json",
        dict(
            stage="collect",
            source_digest="collected-source",
            jobs=[dict(name="collect_000000", kind="collect", pairs=planned)],
        ),
    )
    save_json(root / "jobs/collect_000000/summary.json", dict(status="completed"))
    return root


def test_quality_reports_errors_splits_and_worst_examples(collection):
    result = quality_report(collection)
    assert result["passed"]
    assert result["eligible_pairs"] == 6
    assert result["splits"] == dict(train=2, val=2, test=2)
    assert result["reconstruction_error"]["mean"] == pytest.approx(0.02)
    assert result["reconstruction_error"]["p95"] == pytest.approx(0.03)
    assert result["worst_examples"][0]["reconstruction_error"] == 0.03
    assert result["warnings"]  # unknown simulator seeds are explained, not invented
    assert (collection / "quality/tasks.csv").is_file()
    assert (collection / "quality/worst_examples.csv").is_file()


def test_quality_detects_corrupted_targets_and_missing_records(collection):
    record = read_json(collection / "pairs/0_10/record.json")
    tensors = torch.load(record["pair"], weights_only=True)
    tensors["recovered"][0, 0, 1, 0, 0] = float("nan")
    torch.save(tensors, record["pair"])
    (collection / "pairs/1_10/record.json").unlink()
    result = inspect_dataset(collection)
    assert not result["passed"]
    assert any("nonfinite" in error for error in result["errors"])
    assert any("Missing pair record" in error for error in result["errors"])


def test_quality_checks_leakage_even_when_hashes_match(collection):
    manifest = read_json(collection / "manifest.json")
    context_path = Path(manifest["jobs"][0]["pairs"][2]["context"])
    context = read_json(context_path)
    context["demo_id"] = "0"  # same identity in train and validation
    save_json(context_path, context)
    for pair in manifest["jobs"][0]["pairs"]:
        if pair["context"] == str(context_path):
            pair["context_sha256"] = hashlib.sha256(
                context_path.read_bytes()
            ).hexdigest()
    save_json(collection / "manifest.json", manifest)
    result = inspect_dataset(collection, check_tensors=False)
    assert not result["passed"]
    assert any("leaks" in error for error in result["errors"])


def test_training_only_plans_one_job_without_simulator_seeds(collection, tmp_path):
    with patch(
        "experiments.noiseicl.planning.snapshot", return_value="new-training-source"
    ):
        manifest = plan(
            tmp_path / "training",
            Settings(stage=5, training_only=True, overfit_pairs=1),
            dataset=collection,
        )
    assert manifest["stage"] == "train"
    assert [(job["name"], job["kind"]) for job in manifest["jobs"]] == [
        ("train", "train")
    ]
    with pytest.raises(ValueError, match="training-only"):
        plan(tmp_path / "live", Settings(stage=5), dataset=collection)


def test_invalid_quality_blocks_training_only_plan(collection, tmp_path):
    (collection / "pairs/0_10/record.json").unlink()
    with pytest.raises(ValueError, match="quality checks failed"):
        plan(
            tmp_path / "training",
            Settings(stage=5, training_only=True),
            dataset=collection,
        )
    assert not (tmp_path / "training").exists()


def test_quality_excludes_fallback_targets(collection):
    path = collection / "pairs/0_10/record.json"
    record = read_json(path)
    record["fallback"] = True
    save_json(path, record)
    result = inspect_dataset(collection)
    assert result["passed"]
    assert result["eligible_pairs"] == 5
    assert result["statuses"]["fallback"] == 1
    assert all(row["id"] != "0_10" for row in result["worst_examples"])


def test_quality_detects_changed_context_and_duplicate_target(collection):
    context = collection.parent / "context_0.json"
    saved = read_json(context)
    saved["scene_id"] = "changed"
    save_json(context, saved)
    first = read_json(collection / "pairs/1_10/record.json")
    second_path = collection / "pairs/1_20/record.json"
    second = read_json(second_path)
    second["pair"] = first["pair"]
    save_json(second_path, second)
    result = inspect_dataset(collection)
    assert not result["passed"]
    assert any("Context changed" in error for error in result["errors"])
    assert any("shared target" in error for error in result["errors"])


def test_quality_cli_writes_report_and_exits_nonzero_on_bad_data(
    collection, monkeypatch
):
    from experiments.noiseicl.cli import main

    (collection / "pairs/0_10/record.json").unlink()
    monkeypatch.setattr(
        "sys.argv", ["noiseicl", "quality", "--directory", str(collection)]
    )
    with pytest.raises(SystemExit) as exit_status:
        main()
    assert exit_status.value.code == 1
    assert not read_json(collection / "quality/summary.json")["passed"]


def test_training_only_cli_creates_single_training_job(
    collection, tmp_path, monkeypatch
):
    from experiments.noiseicl.cli import main

    output = tmp_path / "overfit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "noiseicl",
            "plan",
            "--stage",
            "5",
            "--training-only",
            "--cluster",
            "das6",
            "--directory",
            str(output),
            "--dataset",
            str(collection),
            "--overfit-pairs",
            "1",
            "--train-steps",
            "2",
            "--batch",
            "1",
        ],
    )
    with patch(
        "experiments.noiseicl.planning.snapshot", return_value="training-source"
    ):
        main()
    manifest = read_json(output / "manifest.json")
    assert manifest["settings"]["training_only"]
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["kind"] == "train"
