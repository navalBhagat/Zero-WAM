"""Re-splitting changes metadata only and keeps connected demos together."""

import hashlib
from pathlib import Path

import pytest

from wan_va.noise_inversion.dataset import grouped_splits
from wan_va.noise_inversion.quality import inspect_dataset
from wan_va.noise_inversion.records import read_json, save_json
from wan_va.noise_inversion.resplit import resplit_collection


@pytest.fixture
def collection(tmp_path):
    root = tmp_path / "original"
    pairs = []
    for task in ("adjust_bottle", "beat_block_hammer"):
        for demo in range(20):
            # Deliberately reproduce missing task-level coverage, while the
            # collection as a whole still has all three splits.
            split = "train"
            if task == "beat_block_hammer":
                split = "train" if demo < 14 else "val" if demo < 17 else "test"
            for trajectory in range(2):
                context_path = root / "contexts" / f"{task}_{demo}_{trajectory}.json"
                save_json(
                    context_path,
                    dict(
                        task=task,
                        trajectory_id=f"{demo}_{trajectory}",
                        scene_id=f"{demo}_{trajectory}",
                        demo_id=str(demo),
                        demo_latent=str(tmp_path / task / f"demo_{demo}.pth"),
                        simulator_seed=None,
                        initial="unused.pt",
                        history=[],
                        chunk=0,
                        split=split,
                    ),
                )
                for seed in (11, 22):
                    pair_id = f"{task}_{demo}_{trajectory}_{seed}"
                    pair_path = root / "pairs" / pair_id / "pair.pt"
                    pair_path.parent.mkdir(parents=True)
                    pair_path.write_bytes(b"unchanged collected tensor")
                    pair = dict(
                        id=pair_id,
                        context=str(context_path),
                        split=split,
                        seed=seed,
                        context_sha256=hashlib.sha256(
                            context_path.read_bytes()
                        ).hexdigest(),
                    )
                    pairs.append(pair)
                    save_json(
                        pair_path.parent / "record.json",
                        dict(
                            pair,
                            pair=str(pair_path),
                            status="completed",
                            fallback=False,
                            source_digest="original-source",
                            reconstruction_error=0.04,
                        ),
                    )
    save_json(
        root / "manifest.json",
        dict(
            stage="collect",
            source_digest="original-source",
            jobs=[dict(name="collect", kind="collect", pairs=pairs, submissions=[])],
        ),
    )
    save_json(root / "jobs/collect/summary.json", dict(status="completed"))
    return root


def file_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_resplit_keeps_original_data_and_repairs_task_coverage(collection, tmp_path):
    original_files = file_hashes(collection)
    assert not inspect_dataset(collection, check_tensors=False)["passed"]
    output = tmp_path / "stratified"
    receipt = resplit_collection(collection, output)
    assert file_hashes(collection) == original_files
    assert receipt["pairs"] == 160
    assert receipt["pair_counts"] == {
        task: dict(train=56, val=12, test=12)
        for task in ("adjust_bottle", "beat_block_hammer")
    }
    assert inspect_dataset(output, check_tensors=False)["passed"]
    assert not list(output.rglob("*.pt"))
    source_pairs = {
        pair["id"]: pair
        for job in read_json(collection / "manifest.json")["jobs"]
        for pair in job["pairs"]
    }
    demo_splits = {}
    for record_path in output.glob("pairs/*/record.json"):
        record = read_json(record_path)
        original = read_json(collection / "pairs" / record["id"] / "record.json")
        assert record["pair"] == original["pair"]
        assert record["seed"] == original["seed"]
        assert record["source_digest"] == original["source_digest"]
        assert record["source_context"] == source_pairs[record["id"]]["context"]
        context = read_json(record["context"])
        key = (context["task"], context["demo_id"])
        assert demo_splits.setdefault(key, context["split"]) == context["split"]


def test_resplit_is_deterministic_and_rejects_overwrite(collection, tmp_path):
    first = resplit_collection(collection, tmp_path / "first", seed=1234)
    second = resplit_collection(collection, tmp_path / "second", seed=1234)
    assert first["pair_counts"] == second["pair_counts"]
    for path in (tmp_path / "first/pairs").glob("*/record.json"):
        other = tmp_path / "second/pairs" / path.parent.name / "record.json"
        assert read_json(path)["split"] == read_json(other)["split"]
    with pytest.raises(FileExistsError):
        resplit_collection(collection, tmp_path / "first")


def test_impossible_groups_fail_without_breaking_demo_separation():
    records = [
        dict(
            task="adjust_bottle",
            trajectory_id=str(i),
            scene_id=str(i),
            demo_id="shared",
            demo_latent="same.pth",
            simulator_seed=None,
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="three independent"):
        grouped_splits(records)
    records[0]["task"] = "beat_block_hammer"
    with pytest.raises(ValueError, match="multiple tasks"):
        grouped_splits(records)
