"""Convert released RoboTwin latent intervals to fixed, recorded robot histories.

No simulator, VAE decoding, or generated observations are used. Actions use the
existing training preprocessing: initial-state-relative poses, 30-channel layout,
quantile scaling, and a zero action slot for the initial observation.
"""
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
INTERVAL = re.compile(r"episode_(\d+)_(\d+)_(\d+)\.pth$")


def action_preprocessor():
    # Import this standalone helper without importing the dataset package's
    # optional LeRobot training dependencies into the inference environment.
    spec = importlib.util.spec_from_file_location("noiseicl_robotwin_action", PACKAGE_ROOT / "dataset/robotwin_action.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_robotwin_actions


def read_robot_latents(path):
    views, reference = [], None
    for camera in CAMERAS:
        file = path.parent.parent / f"observation.images.{camera}" / path.name
        data = torch.load(file, map_location="cpu", weights_only=True)
        frames, height, width = (int(data[k]) for k in ("latent_num_frames", "latent_height", "latent_width"))
        ids = np.asarray(data["frame_ids"], dtype=np.int64)
        signature = (frames, height, width, tuple(ids))
        if reference is not None and signature != reference:
            raise ValueError(f"Camera layout/frame IDs differ: {file}")
        reference = signature
        if (height, width) != (14, 18) or tuple(data["latent"].shape) != (frames * height * width, 48):
            raise ValueError(f"Unsupported checkpoint observation layout: {file}")
        if len(ids) < 5 or np.any(np.diff(ids) != 4) or frames != (len(ids) - 1) // 4 + 1:
            raise ValueError(f"Expected 12.5 Hz observations compressed 4x with 16 actions per latent: {file}")
        value = data["latent"].reshape(frames, height, width, 48).permute(3, 0, 1, 2).unsqueeze(0)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite robot latent: {file}")
        views.append(value)
    return torch.cat(views, dim=-1).contiguous(), ids


def normalized_actions(action, state, latent_frames, stats_path):
    stats = json.loads(Path(stats_path).read_text())
    if stats.get("method") != "abs":
        raise ValueError("Robot action statistics must use method=abs")
    hand = stats["norm_stats"]["action.hand.position"]
    effector = stats["norm_stats"]["action.effector.position"]
    used = list(range(7)) + [28] + list(range(7, 14)) + [29]
    inverse = [16] * 30
    for index, channel in enumerate(used):
        inverse[channel] = index
    # Actions passed here are already aligned to the first sampled observation.
    values, _ = action_preprocessor()(
        action=action, state=state,
        q01=hand["q01"] + [0.] * 14 + effector["q01"],
        q99=hand["q99"] + [0.] * 14 + effector["q99"],
        inverse_used_action_channel_ids=inverse, history_size=16,
        required_size=latent_frames * 16)
    result = torch.from_numpy(values).float().reshape(latent_frames, 16, 30).permute(2, 0, 1)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite normalized recorded actions")
    return result.unsqueeze(0).unsqueeze(-1).contiguous()


def convert(root, output, tasks):
    import pyarrow.parquet as pq

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh history output directory: {output}")
    manifest_path = root / "icl_configs/ICL_config_robotwin.json"
    manifest = json.loads(manifest_path.read_text())
    pairings = {}
    for sample in manifest["samples"]:
        video = (sample.get("status_info") or {}).get("video_rel_path", "")
        parts = Path(video).parts
        if not parts:
            continue
        match = re.match(r"episode_(\d+)", Path(video).stem)
        if match:
            key = (parts[0], int(match[1]))
            human = sample["human_video_path"].split("human_data/", 1)[1]
            latent = root / "human_latents" / Path(human).with_suffix(".pth")
            previous = pairings.get(key)
            if previous is not None and previous != latent:
                raise ValueError(f"Ambiguous paired human demonstration: {key}")
            pairings[key] = latent
    output.mkdir(parents=True)
    records, report, verified_demos = [], {}, set()
    stats_path = PACKAGE_ROOT / "assets/norm_stats/robotwin_icl.json"
    for task in tasks:
        folders = sorted(p for p in (root / "robotwin_data").iterdir() if p.is_dir() and p.name.split("-")[0] == task)
        if len(folders) != 1:
            raise ValueError(f"Expected one downloaded task directory for {task}")
        folder = folders[0]
        info = json.loads((folder / "meta/info.json").read_text())
        if info["fps"] != 50:
            raise ValueError(f"Expected 50 Hz recorded actions: {task}")
        episode_rows = (json.loads(line) for line in (folder / "meta/episodes.jsonl").read_text().splitlines()
                        if line.strip())
        episodes = {int(row["episode_index"]): row for row in episode_rows}
        task_rows = []
        for path in sorted(folder.glob("latents/chunk-*/observation.images.cam_high/*.pth")):
            match = INTERVAL.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unrecognized cached interval: {path}")
            episode, start, end = map(int, match.groups())
            demo = pairings.get((folder.name, episode))
            if demo is None or not demo.is_file():
                raise FileNotFoundError(f"Missing published human pairing/latent for {path}")
            if demo not in verified_demos:
                human = torch.load(demo, map_location="cpu", weights_only=True)
                if "latent" not in human or "text_emb" not in human or human["text_emb"] is None:
                    raise ValueError(f"Human latent requires latent and cached text_emb for this teacher: {demo}")
                verified_demos.add(demo)
            video, ids = read_robot_latents(path)
            if start < 0 or end > int(episodes[episode]["length"]) or ids[0] < start or ids[-1] >= end:
                raise ValueError(f"Frame IDs outside recorded episode interval: {path}")
            parquet = folder / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
            table = pq.read_table(parquet, columns=["action", "observation.state", "frame_index"])
            frame_index = np.asarray(table["frame_index"].to_pylist())
            if len(frame_index) < end or not np.array_equal(frame_index, np.arange(len(frame_index))):
                raise ValueError(f"Non-contiguous parquet frame indices: {parquet}")
            # Match the training loader: normalize all actions relative to the
            # first state of this interval and prepend the initial zero slot.
            first = int(ids[0])
            action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[first:end]
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[first:end]
            actions = normalized_actions(action, state, video.shape[2], stats_path)
            identifier = f"{episode:06d}_{start}_{end}"
            destination = output / task / identifier
            (destination / "history").mkdir(parents=True)
            torch.save(dict(initial_frame=video[:, :, :1].clone()), destination / "initial.pt")
            # First cached block includes the initial observation and zero-action
            # slot, as in training. Only complete two-frame blocks are used.
            # Leave at least one recorded latent after the final history boundary.
            history_count = (video.shape[2] - 1) // 2
            for chunk in range(history_count):
                section = slice(2 * chunk, 2 * chunk + 2)
                torch.save(dict(video=video[:, :, section].clone(), actions=actions[:, :, section].clone()),
                           destination / "history" / f"chunk_{chunk:04d}.pt")
            # Episode identity is a scene grouping proxy, not a simulator seed.
            # Never fabricate a simulator seed from an episode index.
            row = dict(task=task, trajectory_id=identifier, scene_id=f"episode_{episode:06d}",
                       simulator_seed=None, demo_id=demo.parent.name, demo_latent=str(demo),
                       initial=str(destination / "initial.pt"), history=str(destination / "history"),
                       chunk_count=history_count + 1, source_latent=str(path), source_parquet=str(parquet),
                       source_episode=episode, source_start=start, source_end=end,
                       frame_ids=ids.tolist(), action_stats=str(stats_path),
                       scene_id_source="recorded_episode", history_layout="initial_in_first_two_frame_block")
            records.append(row)
            task_rows.append(row)
        report[task] = dict(trajectories=len(task_rows), demos=len({r["demo_latent"] for r in task_rows}),
                            available_contexts=sum(r["chunk_count"] for r in task_rows))
        print(f"{task}: {report[task]}", flush=True)
    target = output / "trajectories.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "conversion.json").write_text(json.dumps(dict(tasks=report, source=str(manifest_path),
              action_stats=str(stats_path), simulator_seeds_available=False,
              note="Recorded episode IDs group related intervals; cross-episode scene identities are unavailable."), indent=2) + "\n")
    return target
