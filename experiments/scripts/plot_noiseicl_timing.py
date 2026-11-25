"""Plot recorded client wall time without double-counting nested profiler spans."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(campaign, output):
    tasks = sorted((campaign / "trials").iterdir())
    tasks = [task for task in tasks if task.is_dir()]
    groups = {
        "Server sampling": ["server_sampling_roundtrip"],
        "Observation cache": ["server_cache_update"],
        "Simulation / planning": ["action_execution", "planning_left_plan_path", "planning_right_plan_path", "physics_step"],
        "Cameras / render": ["observation", "camera_capture", "camera_readback", "render_state_update"],
        "Reset / other": [],
    }
    methods = ("icl", "control", "oracle")
    fig, ax = plt.subplots(figsize=(13, 7))
    positions, labels, totals = [], [], []
    values = {group: [] for group in groups}
    for index, task in enumerate(tasks):
        trials = sorted(task.glob("scene_*_noise_*"))
        if len(trials) != 1:
            raise ValueError(f"Expected one matched trial for {task.name}")
        for offset, method in enumerate(methods):
            profile = json.loads((trials[0] / method / "diagnostics/profiles/profile_episode_0.json").read_text())
            timing = profile["timings"]
            wall = profile["wall_seconds"] / 60
            assigned = 0
            for group, spans in groups.items():
                minutes = sum(timing.get(span, {}).get("exclusive_seconds", 0) for span in spans) / 60
                if spans:
                    assigned += minutes
                else:
                    minutes = max(0, wall - assigned)
                values[group].append(minutes)
            positions.append(index * 4 + offset)
            labels.append(method.upper() if method == "icl" else method)
            totals.append(wall)
    bottom = np.zeros(len(positions))
    for group, minutes in values.items():
        ax.bar(positions, minutes, bottom=bottom, label=group)
        bottom += minutes
    for x, total in zip(positions, totals):
        ax.text(x, total + .4, f"{total:.1f}", ha="center", fontsize=8)
    ax.set_xticks(positions, labels, rotation=90)
    for index, task in enumerate(tasks):
        ax.text(index * 4 + 1, -.29, task.name.replace("_", "\n"), transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.set_ylabel("Minutes per episode")
    ax.set_title("NoiseICL smoke: inference dominates long episodes\nCampaign: " + campaign.name + "")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=.39)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.campaign, args.output)
