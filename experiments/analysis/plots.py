"""The shared plotting backend. Import matplotlib only when plots are requested."""

from collections import defaultdict
import statistics
from .metrics import COMPONENTS


def campaign_plots(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(7, len(tasks) * 5), 4.6),
        squeeze=False,
        layout="constrained",
    )
    for ax, task in zip(axes[0], tasks):
        group = [r for r in rows if r["task"] == task]
        for i, r in enumerate(group):
            if r["success_rate"] is None:
                continue
            p = r["success_rate"]
            ax.bar(i, p * 100, color="steelblue", alpha=0.85)
            ax.errorbar(
                i,
                p * 100,
                yerr=[
                    [(p - r["success_ci_low"]) * 100],
                    [(r["success_ci_high"] - p) * 100],
                ],
                fmt="none",
                color="black",
                capsize=4,
            )
            ax.text(
                i,
                r["success_ci_high"] * 100 + 3,
                f"{r['successes']}/{r['completed_episodes']}",
                ha="center",
            )
        ax.set_xticks(
            range(len(group)), [r["condition"].replace("_", "\n") for r in group]
        )
        ax.set(
            title=task.replace("_", " "),
            ylabel="Success (%)",
            ylim=(0, 115),
            yticks=[0, 20, 40, 60, 80, 100],
        )
    fig.suptitle("Completed episodes · counts and descriptive 95% intervals")
    fig.savefig(output / "success.png", dpi=170)
    plt.close(fig)
    compute_available = all(r.get("total_policy_tflops") is not None for r in rows)
    fig, axes = plt.subplots(
        2 if compute_available else 1,
        len(tasks),
        figsize=(max(7, len(tasks) * 5), 8 if compute_available else 4.8),
        squeeze=False,
        layout="constrained",
    )
    max_minutes = max(
        (
            (r.get("preparation_seconds_mean") or 0)
            + (r.get("rollout_seconds_mean") or 0)
        )
        / 60
        for r in rows
    )
    for ax, task in zip(axes[0], tasks):
        group = [r for r in rows if r["task"] == task]
        for i, r in enumerate(group):
            prep, roll = (
                r.get("preparation_seconds_mean"),
                r.get("rollout_seconds_mean"),
            )
            if prep is None or roll is None:
                ax.text(
                    i, 0, "Timing unavailable", rotation=90, ha="center", va="bottom"
                )
                continue
            ax.bar(
                i,
                prep / 60,
                color="darkorange",
                label="Episode preparation (reset)" if i == 0 else None,
            )
            ax.bar(
                i,
                roll / 60,
                bottom=prep / 60,
                color="steelblue",
                label="Rollout + simulator" if i == 0 else None,
            )
        ax.set_xticks(
            range(len(group)), [r["condition"].replace("_", "\n") for r in group]
        )
        ax.set(
            title=task.replace("_", " "),
            ylabel="Mean minutes per completed episode",
            ylim=(0, max(1, max_minutes * 1.3)),
        )
        ax.legend(fontsize=8, loc="upper left")
    if compute_available:
        for ax, task in zip(axes[1], tasks):
            group = [r for r in rows if r["task"] == task]
            ax.bar(
                range(len(group)),
                [r["total_policy_tflops"] for r in group],
                color="steelblue",
            )
            ax.set_xticks(
                range(len(group)), [r["condition"].replace("_", "\n") for r in group]
            )
            ax.set(
                ylabel="Estimated policy TFLOPs / episode",
                title="Operator accounting · dense attention convention",
            )
    fig.suptitle("Measured time · preparation separated from rollout")
    fig.savefig(output / "cost.png", dpi=170)
    plt.close(fig)


def diagnostic_plots(output, runs, episodes, chunks, guidance):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for e in episodes:
        if e.get("wall_seconds") is not None:
            groups[(e["task"], e["label"])].append(e)
    if groups:
        keys = sorted(groups)
        x = list(range(len(keys)))
        fig, axes = plt.subplots(
            3, 1, figsize=(max(9, len(keys) * 0.65), 13), layout="constrained"
        )
        bottom = [0.0] * len(keys)
        for component in COMPONENTS:
            heights = [
                statistics.mean(e.get(component, 0) for e in groups[k]) for k in keys
            ]
            axes[0].bar(x, heights, bottom=bottom, label=component)
            bottom = [b + h for b, h in zip(bottom, heights)]
        axes[0].set_ylabel("Mean episode seconds")
        axes[0].legend(fontsize=7, ncol=4)
        axes[0].set_title(
            "Exclusive wall time: camera costs can include GPU waits; physics hooks may be unavailable"
        )
        for axis, field, label in [
            (axes[1], "seconds_per_action", "Episode seconds / executed action"),
            (
                axes[2],
                "sampling_seconds_per_chunk",
                "Sampling roundtrip seconds / chunk",
            ),
        ]:
            for i, k in enumerate(keys):
                ys = [e[field] for e in groups[k] if e.get(field) is not None]
                axis.scatter([i] * len(ys), ys, alpha=0.7)
                if ys:
                    axis.plot(i, statistics.mean(ys), "k_", markersize=18)
            axis.set_ylabel(label)
        for axis in axes:
            axis.set_xticks(
                x, [f"{t}\n{l}" for t, l in keys], rotation=60, ha="right", fontsize=7
            )
        fig.savefig(output / "timing.png", dpi=170)
        plt.close(fig)
    if chunks:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        byrun = defaultdict(list)
        for c in chunks:
            byrun[c["run"]].append(c)
        for run, values in byrun.items():
            ax.plot(
                [v["call"] for v in values],
                [v["total_seconds"] for v in values],
                label=f"run {run}",
            )
        ax.set(
            xlabel="Inference call within job",
            ylabel="Server inference seconds",
            title="Server sampling (excludes client simulation)",
        )
        ax.legend(fontsize=6, ncol=3)
        fig.savefig(output / "server_chunks.png", dpi=170)
        plt.close(fig)
    if guidance:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
        byrun = defaultdict(list)
        for g in guidance:
            byrun[g["run"]].append(g)
        for run, values in byrun.items():
            axes[0].plot(
                [v["call"] for v in values],
                [v["reward"] for v in values],
                label=f"run {run}",
            )
            axes[1].plot([v["call"] for v in values], [v["seconds"] for v in values])
        axes[0].set(
            ylabel="DINO reward before correction",
            title="Descriptive only: targets and robot histories can differ across calls",
        )
        axes[1].set(xlabel="Guidance call within job", ylabel="Guidance seconds")
        axes[0].legend(fontsize=6, ncol=3)
        fig.savefig(output / "guidance.png", dpi=170)
        plt.close(fig)
    if runs:
        fig, ax = plt.subplots(
            figsize=(max(8, len(runs) * 0.4), 5), layout="constrained"
        )
        bottom = [0.0] * len(runs)
        for field, label in [
            ("pre_server_seconds", "Pre-server setup"),
            ("startup_seconds", "Server startup"),
            ("client_seconds", "Client total"),
        ]:
            heights = [r.get(field) or 0 for r in runs]
            ax.bar(range(len(runs)), heights, bottom=bottom, label=label)
            bottom = [b + h for b, h in zip(bottom, heights)]
        residual = [
            max(0, (r.get("allocated_seconds") or 0) - b) for r, b in zip(runs, bottom)
        ]
        ax.bar(
            range(len(runs)),
            residual,
            bottom=bottom,
            label="Cleanup / failed or unclassified time",
        )
        ax.set(
            xlabel="Manifest run number",
            ylabel="Job-script seconds",
            title="Startup and client costs; missing timings are not zero-cost runs",
        )
        ax.legend(fontsize=8)
        fig.savefig(output / "job_timing.png", dpi=170)
        plt.close(fig)
