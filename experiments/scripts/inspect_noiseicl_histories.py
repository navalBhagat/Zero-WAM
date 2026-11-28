"""Read cached robot latents and report nominal two-frame history capacity."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

import torch


TASKS = ("adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
         "blocks_ranking_size", "click_alarmclock")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def inspect(root, tasks=TASKS):
    print("Read-only inspection; no videos, GPU, or model checkpoint required.", flush=True)
    print("Points below = initial state + completed two-frame observation blocks.")
    print("Includes the final recorded boundary; not a validated conversion or pair count.")
    print("Counts are per cached interval, not necessarily distinct trajectories or human demos.\n")
    for task in tasks:
        roots = sorted(p for p in root.iterdir() if p.is_dir() and p.name.split("-")[0] == task)
        if len(roots) != 1:
            print(f"{task}: expected one task directory, found {len(roots)}", flush=True)
            continue
        paths = sorted(roots[0].glob("latents/chunk-*/observation.images.cam_high/*.pth"))
        counts, errors = [], []
        for path in paths:
            try:
                frames = []
                ids = None
                for camera in CAMERAS:
                    camera_path = path.parent.parent / f"observation.images.{camera}" / path.name
                    data = torch.load(camera_path, map_location="cpu", weights_only=True)
                    f, h, w = (int(data[k]) for k in ("latent_num_frames", "latent_height", "latent_width"))
                    if tuple(data["latent"].shape) != (f * h * w, 48):
                        raise ValueError(f"Unexpected latent shape: {camera}")
                    current_ids = [int(x) for x in data["frame_ids"]]
                    if ids is not None and current_ids != ids:
                        raise ValueError("Camera frame IDs differ")
                    ids = current_ids
                    frames.append(f)
                if len(set(frames)) != 1:
                    raise ValueError("Camera temporal lengths differ")
                counts.append((path.name, frames[0], 1 + (frames[0] - 1) // 2))
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                errors.append(f"{path.name}: {error}")
        print(f"{task}: {len(paths)} cached intervals; {len(errors)} errors", flush=True)
        if counts:
            sizes = [f for _, f, _ in counts]
            points = [p for _, _, p in counts]
            print(f"  Latent frames min/median/max: {min(sizes)}/{statistics.median(sizes):g}/{max(sizes)}")
            print(f"  Nominal history-point counts: {dict(sorted(Counter(points).items()))}")
            print(f"  Intervals with >=10 points: {sum(p >= 10 for p in points)}/{len(points)}")
            print("  Shortest examples:", sorted(counts, key=lambda row: row[1])[:3], flush=True)
        for error in errors[:3]:
            print(f"  ERROR: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/HumanGen/robotwin_data"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()
    inspect(args.root, args.tasks)
