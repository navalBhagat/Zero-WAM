import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.scripts.prepare_noiseicl_pilot import DEFAULT_TASKS, prepare


@pytest.mark.parametrize("reuse_demo", [False, True])
def test_pilot_counts_and_distinct_demos(tmp_path, reuse_demo):
    rows = []
    for task in DEFAULT_TASKS:
        for number in range(20):
            root = tmp_path / task / str(number)
            (root / "history").mkdir(parents=True)
            (root / "initial.pt").touch()
            (root / "demo.pt").touch()
            for chunk in range(9):
                (root / "history" / f"chunk_{chunk:04d}.pt").touch()
            rows.append(dict(task=task, trajectory_id=str(number), scene_id=str(number),
                             simulator_seed=number, demo_id="shared" if reuse_demo else str(number),
                             demo_latent=str(root / "demo.pt"), initial=str(root / "initial.pt"),
                             history=str(root / "history"), chunk_count=10))
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(map(json.dumps, rows)))
    output, campaign = tmp_path / "dataset", tmp_path / "campaign"
    if reuse_demo:
        with pytest.raises(ValueError, match="demo ID|need 20 distinct"):
            prepare(source, output, campaign, "snellius")
        assert not campaign.exists()
        assert not output.exists()
        return
    with patch("experiments.noiseicl.planning.snapshot", return_value="digest"):
        result = prepare(source, output, campaign, "snellius")
    assert len(result["jobs"]) == 80
    pairs = [p for j in result["jobs"] for p in j["pairs"]]
    assert len(pairs) == 2000
    assert len({p["id"] for p in pairs}) == 2000
    contexts = json.loads((output / "index.json").read_text())
    assert contexts["trajectories"] == 100
    assert len(contexts["contexts"]) == 1000
    assert set(contexts["tasks"]) == set(DEFAULT_TASKS)


def test_short_recordings_keep_all_demos_without_duplicate_contexts(tmp_path):
    rows = []
    for task in DEFAULT_TASKS:
        for number in range(40):
            root = tmp_path / task / str(number)
            (root / "history").mkdir(parents=True)
            (root / "initial.pt").touch()
            for chunk in range(2):
                (root / "history" / f"chunk_{chunk:04d}.pt").touch()
            demo = tmp_path / task / f"demo_{number % 20}.pt"
            demo.touch()
            rows.append(dict(task=task, trajectory_id=str(number), scene_id=str(number),
                             simulator_seed=None, demo_id=str(number % 20), demo_latent=str(demo),
                             initial=str(root / "initial.pt"), history=str(root / "history"), chunk_count=3))
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(map(json.dumps, rows)))
    with patch("experiments.noiseicl.planning.snapshot", return_value="digest"):
        result = prepare(source, tmp_path / "index", tmp_path / "campaign", "das6")
    pairs = [p for job in result["jobs"] for p in job["pairs"]]
    assert len(pairs) == 5 * 40 * 3 * 2
    assert len({p["id"] for p in pairs}) == len(pairs)
    summary = json.loads((tmp_path / "campaign/pilot.json").read_text())
    assert all(row == dict(contexts=120, pairs=240, shortfall=80) for row in summary["tasks"].values())
    splits = {}
    for pair in pairs:
        context = json.loads(Path(pair["context"]).read_text())
        key = context["demo_latent"]
        assert splits.setdefault(key, context["split"]) == context["split"]
    assert len(splits) == 100


def test_context_budget_balances_demos_and_spans_time():
    from wan_va.noise_inversion.dataset import balanced_points
    rows = [dict(trajectory_id=f"{demo}_{trajectory}", demo_id=demo, chunk_count=15)
            for demo in range(20) for trajectory in range(3)]
    selected = balanced_points(rows, points=15, task_budget=200)
    assert sum(map(len, selected.values())) == 200
    for demo in range(20):
        chunks = [chunk for trajectory, points in selected.items()
                  if trajectory.startswith(f"{demo}_") for chunk in points]
        assert len(chunks) == 10
        assert 0 in chunks and 14 in chunks
