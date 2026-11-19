"""Offline whole-image DINO matching and a causal reference-position diagnostic.

The posterior estimates correspondence to one reference, not physical completion.
Transition bounds and temperature are hypotheses to validate with stage labels.
No rollout selector is changed by this module.
"""

import csv
import json
import math
from pathlib import Path

from experiments.io import write_csv
from experiments.paths import ROOT
from .runtime import guidance_modules
from .reference import load_reference
from .temporal import parser, validate_options, load_inputs, embed_observations


def track(scores, max_advance=4, temperature=0.05, start_frame=0):
    """Causal Bayesian filter, uniform stay/advance transition, cosine likelihood.

    Start at a declared reference frame. Each robot chunk can stay or advance up
    to max_advance reference frames. Boundary transitions are renormalized.
    Equal visual evidence spreads uncertainty rather than proving progress.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not scores.size or not np.isfinite(scores).all():
        raise ValueError(
            "Expected finite nonempty [observations, reference frames] scores"
        )
    if max_advance < 0 or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Invalid transition bound or temperature")
    n = scores.shape[1]
    if not 0 <= start_frame < n:
        raise ValueError("Start frame is outside the reference")
    posterior = np.zeros(n)
    posterior[start_frame] = 1
    rows, distributions = [], []
    for step, similarity in enumerate(scores):
        prior = np.zeros(n)
        # The first observation is anchored, not advanced before seeing motion.
        if step == 0:
            prior = posterior.copy()
        else:
            for source, mass in enumerate(posterior):
                stop = min(n, source + max_advance + 1)
                prior[source:stop] += mass / (stop - source)
        logits = similarity / temperature
        reachable = prior > 0
        likelihood = np.exp(np.clip(logits - logits[reachable].max(), -745, 0))
        posterior = prior * likelihood
        posterior /= posterior.sum()
        cdf = posterior.cumsum()
        estimate = int(np.searchsorted(cdf, 0.5))
        low, high = (int(np.searchsorted(cdf, q)) for q in (0.1, 0.9))
        ranking = np.argsort(similarity)
        best = int(ranking[-1])
        runner_up = float(similarity[ranking[-2]]) if n > 1 else float(similarity[best])
        rows.append(
            dict(
                nearest_frame=best,
                nearest_similarity=float(similarity[best]),
                nearest_margin=float(similarity[best] - runner_up),
                progress_frame=estimate,
                interval_low=low,
                interval_high=high,
                progress_fraction=estimate / max(1, n - 1),
            )
        )
        distributions.append(posterior.copy())
    return rows, np.stack(distributions)


def read_labels(path, reference_count, chunks):
    """Optional CSV: kind,index,stage; kind is reference or robot, indices exact.

    Reference index is a frame; robot index is the recorded chunk ID. Labels are
    evaluation only and never enter the filter. Unlabelled frames remain unknown.
    """
    labels = {"reference": {}, "robot": {}}
    if path is None:
        return labels
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            kind, index, stage = row["kind"], int(row["index"]), row["stage"].strip()
            if not stage:
                continue
            if kind not in labels:
                raise ValueError(
                    "Labels require kind reference/robot and a nonempty stage"
                )
            if (kind == "reference" and not 0 <= index < reference_count) or (
                kind == "robot" and index not in chunks
            ):
                raise ValueError(f"Label index outside inputs: {kind} {index}")
            if index in labels[kind]:
                raise ValueError(f"Duplicate label: {kind} {index}")
            labels[kind][index] = stage
    return labels


def figures(output, scores, posterior, rows, reference, observations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, values, title in zip(
        axes,
        [scores, posterior],
        [
            "Whole-image DINO cosine similarity",
            "Causal position posterior (prior constrained)",
        ],
    ):
        image = ax.imshow(values, aspect="auto", origin="lower")
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Recorded observation order")
    axes[0].plot(
        [r["nearest_frame"] for r in rows],
        range(len(rows)),
        "w.-",
        label="Unconstrained nearest",
    )
    axes[1].plot(
        [r["progress_frame"] for r in rows],
        range(len(rows)),
        "w.-",
        label="Posterior median",
    )
    for ax in axes:
        ax.legend(loc="upper left")
    axes[1].set_xlabel("Human reference frame")
    fig.tight_layout()
    fig.savefig(output / "matching.png", dpi=160)
    plt.close(fig)
    indices = np.unique(np.linspace(0, len(rows) - 1, min(6, len(rows))).astype(int))
    fig, axes = plt.subplots(
        len(indices), 3, figsize=(10, 2.5 * len(indices)), squeeze=False
    )
    for axes_row, index in zip(axes, indices):
        row = rows[index]
        images = [
            observations[index],
            reference[row["nearest_frame"]].permute(1, 2, 0).numpy(),
            reference[row["recommended_target"]].permute(1, 2, 0).numpy(),
        ]
        titles = [
            f"Robot chunk {row['chunk']}",
            f"Nearest frame {row['nearest_frame']}",
            f"Candidate target {row['recommended_target']} (position {row['progress_frame']})",
        ]
        for ax, image, title in zip(axes_row, images, titles):
            ax.imshow(image)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    argument_parser = parser()
    argument_parser.description = __doc__
    argument_parser.set_defaults(
        output=ROOT / "experiments/progress_matching",
        run=ROOT
        / "experiments/stages/03_guidance_strength_100/runs/place_object_scale_29485194",
    )
    argument_parser.add_argument(
        "--task",
        choices=["place_object_scale", "place_bread_basket"],
        default="place_object_scale",
    )
    argument_parser.add_argument(
        "--max-progress-advance",
        type=int,
        default=4,
        help="Maximum reference-frame advance per recorded robot chunk; experimental prior",
    )
    argument_parser.add_argument(
        "--robot-observation-dir",
        type=Path,
        help="Explicit extracted external-camera observations",
    )
    argument_parser.add_argument("--temperature", type=float, default=0.05)
    argument_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Declared initial reference position; inspect frame 0 for compatibility",
    )
    argument_parser.add_argument(
        "--labels", type=Path, help="Optional evaluation CSV: kind,index,stage"
    )
    options = argument_parser.parse_args(argv)
    validate_options(options, argument_parser)
    if (
        options.max_progress_advance < 0
        or options.temperature <= 0
        or not math.isfinite(options.temperature)
    ):
        argument_parser.error("Invalid progress bound or temperature")
    episode, chunks, paths = load_inputs(options, argument_parser)
    import torch

    torch.set_num_threads(options.threads)
    temporal, dino, reference_module = guidance_modules()
    reference, provenance = load_reference(options, episode, reference_module)
    features, queries, observations = embed_observations(
        reference, paths, options.dino_checkpoint, dino
    )
    scores = (queries @ features.T).numpy()
    from .relative_matching import matching_scores

    relative_scores, changed, _ = matching_scores(features.numpy(), queries.numpy())
    try:
        rows, posterior = track(
            scores,
            options.max_progress_advance,
            options.temperature,
            options.start_frame,
        )
        labels = read_labels(options.labels, len(reference), chunks)
    except ValueError as error:
        argument_parser.error(str(error))
    # Reuse the existing automatic visual-distance horizon, changing only alignment.
    selector = temporal.DistanceFrameSelector(
        features,
        options.window,
        options.lookahead,
        options.distance_increment,
        options.smoothing,
        options.max_lookahead_frames,
    )
    source_records = {}
    if (
        options.robot_observation_dir
        and (options.robot_observation_dir / "source.json").is_file()
    ):
        source_records = {
            record["observation_id"]: record
            for record in json.loads(
                (options.robot_observation_dir / "source.json").read_text()
            )["observations"]
        }
    for i, (row, chunk) in enumerate(zip(rows, chunks)):
        if chunk in source_records:
            row.update(
                {
                    key: source_records[chunk][key]
                    for key in ("video_frame", "action_count", "terminal")
                }
            )
        baseline = selector.select(chunk, similarities=scores[i].tolist())[0]
        import bisect

        position = row["progress_frame"]
        target = (
            bisect.bisect_left(
                selector.coordinate,
                selector.coordinate[position] + selector.distance,
                lo=position,
            )
            if selector.distance > 1e-12
            else position + options.lookahead
        )
        target = min(
            len(reference) - 1, position + options.max_lookahead_frames, target
        )
        row.update(
            chunk=chunk,
            current_match=selector.position,
            current_target=baseline,
            frame_lookahead_target=min(
                len(reference) - 1, selector.position + options.lookahead
            ),
            initial_anchor=i == 0,
            recommended_target=target,
            robot_stage=labels["robot"].get(chunk, ""),
        )
        for method in ("relative_cosine", "relative_l2"):
            row[f"{method}_frame"] = (
                int(relative_scores[method][i].argmax()) if changed[i] else None
            )
        row["stay_at_start_frame"] = 0
        row["last_frame_frame"] = len(reference) - 1
        for method in (
            "nearest",
            "progress",
            "relative_cosine",
            "relative_l2",
            "stay_at_start",
            "last_frame",
        ):
            stage = labels["reference"].get(row[f"{method}_frame"], "")
            row[f"{method}_stage"] = stage
            row[f"{method}_stage_correct"] = (
                bool(stage and stage == row["robot_stage"])
                if row["robot_stage"]
                else ""
            )
    options.output.mkdir(parents=True, exist_ok=True)
    write_csv(options.output / "results.csv", rows)
    import numpy as np

    np.savez_compressed(
        options.output / "embeddings.npz",
        reference=features.numpy(),
        robot=queries.numpy(),
        chunks=chunks,
    )
    figures(options.output, scores, posterior, rows, reference, observations)
    # Emit every reference frame so manual stage annotation does not require decoding again.
    from PIL import Image

    preview = options.output / "reference_frames"
    preview.mkdir()
    for i, frame in enumerate(reference):
        Image.fromarray(
            frame.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
        ).save(preview / f"frame_{i:04d}.png")
    template = [
        dict(kind="reference", index=i, stage="") for i in range(len(reference))
    ]
    template += [dict(kind="robot", index=c, stage="") for c in chunks]
    write_csv(options.output / "labels_template.csv", template)
    provenance.update(
        robot_observation_dir=str(options.robot_observation_dir)
        if options.robot_observation_dir
        else None,
        observation_source=(
            json.loads((options.robot_observation_dir / "source.json").read_text())
            if options.robot_observation_dir
            and (options.robot_observation_dir / "source.json").is_file()
            else None
        ),
        run=str(options.run.resolve()),
        episode=options.episode,
        device="cpu",
        dino_checkpoint=str(options.dino_checkpoint.resolve()),
        max_progress_advance=options.max_progress_advance,
        temperature=options.temperature,
        start_frame=options.start_frame,
        distance=selector.distance,
        labels=str(options.labels) if options.labels else None,
        interpretation="Offline candidate correspondence, not semantic progress or rollout success; prior can impose apparent ordering.",
    )
    (options.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        f"Saved results.csv, matching.png, comparison.png and annotation frames to {options.output}"
    )
    for method in (
        "nearest",
        "progress",
        "relative_cosine",
        "relative_l2",
        "stay_at_start",
        "last_frame",
    ):
        evaluated = [
            r[f"{method}_stage_correct"]
            for r in rows[1:]
            if r[f"{method}_stage_correct"] != ""
        ]
        print(
            f"{method} stage agreement: {sum(evaluated)}/{len(evaluated)} labelled noninitial observations"
        )
