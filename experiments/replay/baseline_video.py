"""Recover the external-camera panel from the historical RoboTwin composite.

Only the known 960x624 layout is accepted: a 40-pixel title followed by three
320x240 real-camera panels. Lossy video compression differs from raw previews.
"""

import argparse
import json
from pathlib import Path
from experiments.paths import ROOT


def crop_external(frame):
    if frame.shape != (624, 960, 3):
        raise ValueError(
            f"Unexpected composite layout {frame.shape}; expected (624, 960, 3)"
        )
    return frame[40:280, :320].copy()


def extract(run, output):
    import imageio.v2 as imageio
    from PIL import Image

    config = json.loads((run / "config.json").read_text())
    episode = next(e for e in config["episodes"] if e["episode_id"] == 0)
    if not episode.get("success") or episode.get("simulator_seed") != 100000:
        raise ValueError("Expected the successful seed-100000 baseline episode")
    basename = Path(episode["video_path"]).name
    candidates = list(run.rglob(basename))
    if len(candidates) != 1:
        raise ValueError(f"Expected one recorded video named {basename}")
    reader = imageio.get_reader(str(candidates[0]))
    try:
        frames = [crop_external(frame) for frame in reader]
    finally:
        reader.close()
    # Historical client records every four actions: initial chunk executes 16
    # actions (4 images); subsequent chunks execute 32 actions (8 images).
    # This mapping is specific to the baseline preset, not video playback FPS.
    if len(frames) != 34 or episode.get("simulator_action_count") != 131:
        raise ValueError(
            "Expected 34 observations / 131 actions for baseline job 29286681"
        )
    indices = [0, 4, 12, 20, 28, 33]
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, video_frame in enumerate(indices):
        name = f"robot_chunk_{index:04d}.png"
        Image.fromarray(frames[video_frame]).save(output / name)
        records.append(
            dict(
                observation_id=index,
                video_frame=video_frame,
                action_count=min(video_frame * 4, 131),
                terminal=index == len(indices) - 1,
                image=name,
            )
        )
    (output / "source.json").write_text(
        json.dumps(
            dict(
                video=str(candidates[0].resolve()),
                crop_xyxy=[0, 40, 320, 280],
                compression="H264 recorded composite, not exact raw camera pixels",
                observations=records,
                note="IDs 0..4 are chunk-start observations; ID 5 is terminal, not another inference chunk.",
            ),
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--labels", type=Path)
    options = parser.parse_args()
    if options.output.exists():
        parser.error("Choose a fresh output directory")
    run = ROOT / "experiments/stages/02_baseline_pilot/runs/place_object_scale_29286681"
    observations = extract(run, options.output / "robot_frames")
    from .progress import main as diagnostic

    args = [
        "--run",
        str(run),
        "--output",
        str(options.output / "analysis"),
        "--robot-observation-dir",
        str(observations),
        "--threads",
        str(options.threads),
    ]
    if options.labels:
        args += ["--labels", str(options.labels)]
    diagnostic(args)


if __name__ == "__main__":
    main()
