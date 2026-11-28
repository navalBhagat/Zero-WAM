import json
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
        with pytest.raises(ValueError, match="need 20 distinct"):
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
