import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from wan_va.noise_inversion.humangen import CAMERAS, convert, read_robot_latents

ROOT = Path(__file__).resolve().parents[1]


def source_fixture(tmp_path, frame_count=9):
    root = tmp_path / "HumanGen"
    task = root / "robotwin_data/adjust_bottle-demo_clean_collect_200-1000"
    (task / "meta").mkdir(parents=True)
    length = 16 * (frame_count - 1) + 15
    ids = list(range(0, 16 * (frame_count - 1) + 1, 4))
    for view, camera in enumerate(CAMERAS):
        path = task / "latents/chunk-000" / f"observation.images.{camera}"
        path.mkdir(parents=True)
        video = torch.stack([torch.full((14, 18, 48), view * 100 + f) for f in range(frame_count)])
        torch.save(dict(latent=video.reshape(-1, 48).to(torch.bfloat16), latent_num_frames=frame_count,
                        latent_height=14, latent_width=18, frame_ids=ids), path / f"episode_000042_0_{length}.pth")
    (task / "meta/info.json").write_text(json.dumps(dict(fps=50, chunks_size=1000,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")))
    (task / "meta/episodes.jsonl").write_text(json.dumps(dict(episode_index=42, length=length)))
    state = np.zeros((length, 16))
    state[:, [6, 14]] = 1  # valid xyzw quaternions
    action = state.copy()
    action[:, 0] = np.arange(length) * .001
    path = task / "data/chunk-000/episode_000042.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"action": action.tolist(), "observation.state": state.tolist(),
                            "frame_index": list(range(length))}), path)
    human = root / "human_latents/robotwin/run_test/samples/demo/generated_video_kling-v3.pth"
    human.parent.mkdir(parents=True)
    torch.save(dict(latent=torch.zeros(1), text_emb=torch.zeros(1)), human)
    (root / "icl_configs").mkdir()
    (root / "icl_configs/ICL_config_robotwin.json").write_text(json.dumps(dict(samples=[dict(
        status_info=dict(video_rel_path=f"{task.name}/videos/chunk-000/observation.images.cam_high/episode_000042.mp4"),
        human_video_path="icl_raw_data/human_data/robotwin/run_test/samples/demo/generated_video_kling-v3.mp4") ])))
    return root, task


@pytest.mark.parametrize("frame_count,contexts", [(9, 5), (8, 4), (5, 3)])
def test_conversion_uses_real_camera_frames_and_aligned_actions(tmp_path, frame_count, contexts):
    root, task = source_fixture(tmp_path, frame_count)
    manifest = convert(root, tmp_path / "histories", ["adjust_bottle"])
    row = json.loads(manifest.read_text())
    assert row["chunk_count"] == contexts
    assert row["simulator_seed"] is None  # source episode index is not a simulator seed
    assert row["scene_id"] == "episode_000042"
    initial = torch.load(row["initial"], weights_only=True)["initial_frame"]
    assert initial.shape == (1, 48, 1, 14, 54)
    assert initial[0, 0, 0, 0, [0, 18, 36]].tolist() == [0, 100, 200]
    histories = sorted(Path(row["history"]).glob("*.pt"))
    assert len(histories) == contexts - 1
    first = torch.load(histories[0], weights_only=True)
    assert first["video"].shape == (1, 48, 2, 14, 54)
    assert first["actions"].shape == (1, 30, 2, 16, 1)
    assert torch.count_nonzero(first["actions"][:, :, 0]) == 0
    for index, file in enumerate(histories):
        saved = torch.load(file, weights_only=True)
        assert saved["video"][0, 0, :, 0, 0].tolist() == [2 * index, 2 * index + 1]
    # The second history starts with action row 16, because the initial zero
    # slot occupies 16 entries. Check a changing position against its quantile.
    stats = json.loads((ROOT / "wan_va/assets/norm_stats/robotwin_icl.json").read_text())
    q01 = stats["norm_stats"]["action.hand.position"]["q01"][0]
    q99 = stats["norm_stats"]["action.hand.position"]["q99"][0]
    second = torch.load(histories[1], weights_only=True)
    expected = np.clip((.016 - q01) / (q99 - q01 + 1e-6) * 2 - 1, -2, 2)
    assert second["actions"][0, 0, 0, 0, 0].item() == pytest.approx(expected, abs=1e-6)


def test_misaligned_camera_fails_instead_of_silently_using_wrong_history(tmp_path):
    _, task = source_fixture(tmp_path)
    high = next(task.glob("latents/chunk-*/observation.images.cam_high/*.pth"))
    right = high.parent.parent / "observation.images.cam_right_wrist" / high.name
    data = torch.load(right, weights_only=True)
    data["frame_ids"] = [x + 1 for x in data["frame_ids"]]
    torch.save(data, right)
    with pytest.raises(ValueError, match="Camera layout/frame IDs differ"):
        read_robot_latents(high)


def test_training_group_check_accepts_unknown_simulator_seeds(tmp_path):
    from wan_va.noise_inversion.training import examples
    for index, split in enumerate(("train", "val", "test")):
        context = tmp_path / f"context_{index}.json"
        context.write_text(json.dumps(dict(task="adjust_bottle", scene_id=str(index),
            simulator_seed=None, demo_id=str(index), demo_latent=f"demo_{index}.pth", split=split)))
        pair = tmp_path / "pairs" / str(index)
        pair.mkdir(parents=True)
        (pair / "record.json").write_text(json.dumps(dict(status="completed", context=str(context))))
    assert len(examples(tmp_path)) == 3
