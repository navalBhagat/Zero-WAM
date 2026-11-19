"""Label-assisted stage separability diagnostic from cached DINO embeddings.

Human stage labels define groups; robot labels only evaluate predictions. This
is a diagnostic on two existing trajectories, not a validated stage classifier.
"""

import argparse
import csv
from pathlib import Path

from experiments.io import write_csv
from experiments.replay.relative_matching import matching_scores

STAGES = ["pre_grasp", "holding", "lifted", "placed"]


def labels(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        kind: {
            int(r["index"]): r["stage"]
            for r in rows
            if r["kind"] == kind and r["stage"]
        }
        for kind in ["reference", "robot"]
    }


def analyze(embeddings, annotations):
    import numpy as np

    rows = []
    for trajectory in ["failed", "success"]:
        annotation = annotations[trajectory]
        scores, changed, _ = matching_scores(
            embeddings["reference"], embeddings[trajectory]
        )
        for method, matrix in scores.items():
            for aggregation in ["max", "mean"]:
                grouped = []
                for stage in STAGES:
                    ids = [i for i, s in annotation["reference"].items() if s == stage]
                    if not ids:
                        raise ValueError(f"No reference labels for stage {stage}")
                    values = matrix[:, ids].copy()
                    values[~np.isfinite(values)] = np.nan
                    grouped.append(
                        np.nanmax(values, axis=1)
                        if aggregation == "max"
                        else np.nanmean(values, axis=1)
                    )
                grouped = np.stack(grouped, axis=1)
                for observation in range(1, len(matrix)):
                    actual = annotation["robot"].get(observation, "")
                    if actual not in STAGES:
                        continue
                    valid = method == "absolute_cosine" or changed[observation]
                    values = grouped[observation]
                    truth = STAGES.index(actual)
                    prediction = STAGES[int(values.argmax())] if valid else ""
                    margin = (
                        float(values[truth] - np.delete(values, truth).max())
                        if valid
                        else float("nan")
                    )
                    rank = int(1 + sum(values > values[truth])) if valid else None
                    row = dict(
                        trajectory=trajectory,
                        observation_id=observation,
                        method=method,
                        aggregation=aggregation,
                        actual_stage=actual,
                        predicted_stage=prediction,
                        correct=prediction == actual,
                        actual_stage_rank=rank,
                        actual_vs_best_other_margin=margin,
                    )
                    row.update({f"{s}_score": float(v) for s, v in zip(STAGES, values)})
                    rows.append(row)
    return rows


def plot(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ["absolute_cosine", "relative_cosine", "relative_l2"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for axis_row, trajectory in zip(axes, ["failed", "success"]):
        for ax, method in zip(axis_row, methods):
            selected = [
                r
                for r in rows
                if r["trajectory"] == trajectory
                and r["method"] == method
                and r["aggregation"] == "mean"
            ]
            values = np.array([[r[f"{s}_score"] for s in STAGES] for r in selected])
            # Subtract each row's best score to expose stage preference, not image-wide offsets.
            values = values - values.max(axis=1, keepdims=True)
            image = ax.imshow(values, aspect="auto", origin="lower", vmax=0)
            ax.scatter(
                [STAGES.index(r["actual_stage"]) for r in selected],
                range(len(selected)),
                c="white",
                marker="x",
                label="Actual stage",
            )
            ax.set_xticks(range(4), STAGES, rotation=25)
            ax.set_yticks(range(len(selected)), [r["observation_id"] for r in selected])
            ax.set_ylabel("Observation ID")
            ax.set_title(f"{trajectory}: {method}")
            ax.legend(fontsize=7)
            fig.colorbar(image, ax=ax)
    fig.suptitle(
        "Mean reference-stage score minus best stage score; white crosses show labels"
    )
    fig.tight_layout()
    fig.savefig(output / "stage_scores.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--failed-labels", type=Path, required=True)
    parser.add_argument("--success-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        parser.error("Choose a fresh output directory")
    import numpy as np

    annotations = {
        name: labels(path)
        for name, path in [
            ("failed", options.failed_labels),
            ("success", options.success_labels),
        ]
    }
    if annotations["failed"]["reference"] != annotations["success"]["reference"]:
        parser.error("Reference labels differ")
    with np.load(options.cache, allow_pickle=False) as embeddings:
        rows = analyze(embeddings, annotations)
    options.output.mkdir(parents=True)
    write_csv(options.output / "results.csv", rows)
    plot(options.output, rows)
    for trajectory in ["failed", "success"]:
        for method in ["absolute_cosine", "relative_cosine", "relative_l2"]:
            for aggregation in ["max", "mean"]:
                rs = [
                    r
                    for r in rows
                    if r["trajectory"] == trajectory
                    and r["method"] == method
                    and r["aggregation"] == aggregation
                ]
                print(
                    trajectory,
                    method,
                    aggregation,
                    f"{sum(r['correct'] for r in rs)}/{len(rs)}",
                    "ranks",
                    [r["actual_stage_rank"] for r in rs],
                )
    print("Successful observations, mean-stage scoring:")
    for r in rows:
        if r["trajectory"] == "success" and r["aggregation"] == "mean":
            print(
                r["observation_id"],
                r["method"],
                r["actual_stage"],
                "->",
                r["predicted_stage"],
                "margin",
                round(r["actual_vs_best_other_margin"], 5),
            )


if __name__ == "__main__":
    main()
