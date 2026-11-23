"""Descriptive paired-trial plots; no independence assumption or model inference."""
from pathlib import Path

COLORS = {"source": "#4c78a8", "control": "#f2a541", "replay": "#2a9d8f"}
LABELS = {"source": "ICL", "control": "No ICL / no language",
          "replay": "No ICL / no language + reversal"}


def _task_labels(tables: list[dict]) -> list[str]:
    return [row["task"].replace("_", "\n", 1).replace("_", " ") for row in tables]


def success_figure(plt, tables: list[dict]):
    from matplotlib.patches import Patch

    figure, axis = plt.subplots(figsize=(max(9, len(tables) * 1.7), 6), layout="constrained")
    width = 0.24
    for offset, condition in enumerate(("source", "control", "replay")):
        for index, row in enumerate(tables):
            position = index + (offset - 1) * width
            total = row["valid_pairs"]
            if not total:
                axis.text(position, 2, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
                continue
            successes = row[condition + "_successes"]
            rate = 100 * successes / total
            axis.bar(position, rate, width, color=COLORS[condition])
            if condition == "replay":
                before = row["replay_success_within_noise_horizon"]
                axis.bar(position, 100 * (successes - before) / total, width,
                         bottom=100 * before / total, facecolor="none", hatch="///", edgecolor="#333333")
            axis.text(position, rate + 1.5, f"{successes}/{total}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(range(len(tables)), _task_labels(tables))
    axis.set(ylabel="Success (%)", ylim=(0, 115), yticks=range(0, 101, 20),
             title="Same scene/noise pairs across all three conditions")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    legend = [Patch(facecolor=COLORS[key], label=LABELS[key]) for key in COLORS]
    legend.append(Patch(facecolor="white", hatch="///", label="Reversal success after Gaussian fallback"))
    figure.legend(handles=legend, loc="outside upper center", ncol=2, frameon=False)
    incomplete = sum(row["incomplete"] for row in tables)
    pending = sum(row["pending"] for row in tables)
    figure.supxlabel(f"Counts use complete matched trials only; excluded: {incomplete} incomplete, {pending} pending.\n"
                     "Descriptive results: noises sharing a scene are correlated.", fontsize=9)
    return figure


def transfer_figure(plt, tables: list[dict]):
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(tables) * 1.9), 5.5), layout="constrained")
    for axis, count_key, total_key, color, title in (
        (axes[0], "rescues", "control_failures", "#2a9d8f", "Rescue: control failure → reversal success"),
        (axes[1], "regressions", "control_successes", "#d65f5f", "Regression: control success → reversal failure"),
    ):
        for index, row in enumerate(tables):
            count, total = row[count_key], row[total_key]
            if not total:
                axis.text(index, 2, "N/A", ha="center", fontsize=9)
                continue
            rate = 100 * count / total
            axis.bar(index, rate, color=color, width=0.65)
            axis.text(index, rate + 2, f"{count}/{total}", ha="center", fontsize=9)
        axis.set_xticks(range(len(tables)), _task_labels(tables), fontsize=8)
        axis.set(title=title, ylabel="Eligible trials (%)", ylim=(0, 115), yticks=range(0, 101, 20))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.supxlabel("Different denominators: control failures on the left, control successes on the right.\n"
                     "N/A means no eligible trials; full-episode reversal outcomes include any fallback.", fontsize=9)
    return figure


def write_plots(directory: Path, tables: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = directory / "plots"
    output.mkdir(exist_ok=True)
    for name, build in (("success", success_figure), ("rescue_regression", transfer_figure)):
        figure = build(plt, tables)
        try:
            figure.savefig(output / f"{name}.png", dpi=180)
        finally:
            plt.close(figure)
