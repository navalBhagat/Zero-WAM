"""Compare absolute and initial-relative whole-image DINO correspondence.

Labels evaluate matches only. There is no time prior, learned classifier, or
lookahead: this isolates the visual evidence for current reference position.
"""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.paths import ROOT
from experiments.io import write_csv
from .progress import read_labels
from .runtime import guidance_modules
from .temporal import embed_observations


def matching_scores(reference, queries, epsilon=1e-6):
    """Unit-image embeddings -> absolute cosine, delta cosine and negative L2.

    Delta cosine ignores change magnitude. Delta L2 retains magnitude. An exact
    or near-zero query delta has no direction, so relative methods abstain.
    Reference frame zero is excluded from delta cosine for the same reason.
    """
    import numpy as np

    reference, queries = np.asarray(reference), np.asarray(queries)
    if (
        reference.ndim != 2
        or queries.ndim != 2
        or not len(reference)
        or not len(queries)
        or reference.shape[1] != queries.shape[1]
        or not np.isfinite(reference).all()
        or not np.isfinite(queries).all()
    ):
        raise ValueError("Expected compatible finite nonempty embedding matrices")
    dr, dq = reference - reference[0], queries - queries[0]
    nr, nq = np.linalg.norm(dr, axis=1), np.linalg.norm(dq, axis=1)
    cosine = (dq / np.maximum(nq[:, None], epsilon)) @ (
        dr / np.maximum(nr[:, None], epsilon)
    ).T
    cosine[:, nr <= epsilon] = -np.inf
    l2 = -np.linalg.norm(dq[:, None, :] - dr[None, :, :], axis=-1)
    return (
        {
            "absolute_cosine": queries @ reference.T,
            "relative_cosine": cosine,
            "relative_l2": l2,
        },
        nq > epsilon,
        nq,
    )


def evaluate(reference, queries, chunks, labels, trajectory):
    """Return per-observation outcomes; exclude the declared initial anchor fairly."""
    import numpy as np

    scores, changed, norms = matching_scores(reference, queries)
    rows = []
    for method, matrix in scores.items():
        for i, chunk in enumerate(chunks):
            valid = np.isfinite(matrix[i]).any() and (
                method == "absolute_cosine" or changed[i]
            )
            frame = int(np.argmax(matrix[i])) if valid else None
            actual = labels["robot"].get(chunk, "")
            predicted = labels["reference"].get(frame, "")
            evaluated = bool(
                i > 0 and actual
            )  # Count abstention as failure, not missing accuracy.
            row = dict(
                trajectory=trajectory,
                observation_id=chunk,
                method=method,
                predicted_frame=frame,
                robot_stage=actual,
                predicted_stage=predicted,
                initial_anchor=i == 0,
                abstained=not valid,
                query_change_norm=float(norms[i]),
                evaluated=evaluated,
                stage_correct=bool(valid and predicted and predicted == actual)
                if evaluated
                else "",
                score=float(matrix[i, frame]) if valid else "",
            )
            rows.append(row)
    # This control exposes accuracy obtainable by remaining in the initial stage.
    initial_stage = labels["reference"].get(0, "")
    for i, chunk in enumerate(chunks):
        actual = labels["robot"].get(chunk, "")
        evaluated = bool(i > 0 and actual)
        rows.append(
            dict(
                trajectory=trajectory,
                observation_id=chunk,
                method="stay_at_start",
                predicted_frame=0,
                robot_stage=actual,
                predicted_stage=initial_stage,
                initial_anchor=i == 0,
                abstained=False,
                query_change_norm=float(norms[i]),
                evaluated=evaluated,
                stage_correct=bool(initial_stage and initial_stage == actual)
                if evaluated
                else "",
                score="",
            )
        )
    return rows, scores


def figures(output, results, matrices):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ["absolute_cosine", "relative_cosine", "relative_l2"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    for axes_row, (name, scores) in zip(axes, matrices.items()):
        for ax, method in zip(axes_row, methods):
            values = scores[method].copy()
            values[~np.isfinite(values)] = np.nan
            if method.startswith("relative"):
                values[0] = np.nan
            im = ax.imshow(values, aspect="auto", origin="lower")
            selected = [
                r
                for r in results
                if r["trajectory"] == name
                and r["method"] == method
                and not r["initial_anchor"]
                and not r["abstained"]
            ]
            ax.plot(
                [r["predicted_frame"] for r in selected],
                [r["observation_id"] for r in selected],
                "w.-",
            )
            ax.set_title(f"{name}: {method}")
            ax.set_xlabel("Reference frame")
            ax.set_ylabel("Observation ID")
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "matching.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for ax, name in zip(axes[0], matrices):
        methods_all = methods + ["stay_at_start"]
        accuracy = []
        for method in methods_all:
            evaluated = [
                r
                for r in results
                if r["trajectory"] == name and r["method"] == method and r["evaluated"]
            ]
            accuracy.append(
                sum(r["stage_correct"] for r in evaluated) / len(evaluated)
                if evaluated
                else float("nan")
            )
        ax.bar(range(4), accuracy)
        ax.set_xticks(range(4), methods_all, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.set_ylabel("Stage agreement (initial anchor excluded)")
    fig.tight_layout()
    fig.savefig(output / "stage_agreement.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failed-diagnostic",
        type=Path,
        default=ROOT / "experiments/stages/progress_matching/job_29488053",
    )
    parser.add_argument(
        "--success-diagnostic",
        type=Path,
        default=ROOT / "experiments/stages/progress_matching_icl/job_29489163",
    )
    parser.add_argument(
        "--failed-run",
        type=Path,
        default=ROOT
        / "experiments/stages/03_guidance_strength_100/runs/place_object_scale_29485194",
    )
    parser.add_argument(
        "--failed-labels",
        type=Path,
        default=ROOT / "experiments/stages/progress_matching/labels.csv",
    )
    parser.add_argument(
        "--success-labels",
        type=Path,
        default=ROOT / "experiments/stages/progress_matching_icl/labels.csv",
    )
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=ROOT / "checkpoints/dinov2-small"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    options = parser.parse_args(argv)
    if options.output.exists() or options.threads < 1:
        parser.error("Choose a fresh output directory and positive thread count")
    refpaths = sorted(
        (options.failed_diagnostic / "reference_frames").glob("frame_*.png")
    )
    failedpaths = sorted(
        (options.failed_run / "references/episode_000").glob("robot_chunk_*.png")
    )
    successpaths = sorted(
        (options.success_diagnostic / "robot_frames").glob("robot_chunk_*.png")
    )
    if not refpaths or not failedpaths or not successpaths:
        parser.error(
            "Missing reference or robot images; check the diagnostic/run paths"
        )
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(options.threads)
    chunks = [
        [int(p.stem.rsplit("_", 1)[1]) for p in paths]
        for paths in [failedpaths, successpaths]
    ]
    labels = [
        read_labels(path, len(refpaths), ids)
        for path, ids in zip([options.failed_labels, options.success_labels], chunks)
    ]
    if labels[0]["reference"] != labels[1]["reference"]:
        parser.error(
            "Reference-stage labels differ between trajectories; use consistent annotations"
        )
    # Confirm both previous diagnostics decoded the same reference, not just the same frame count.
    a = json.loads((options.failed_diagnostic / "provenance.json").read_text())
    b = json.loads(
        (options.success_diagnostic / "analysis/provenance.json").read_text()
    )
    if (a["kind"], a["path"], a.get("vae_path")) != (
        b["kind"],
        b["path"],
        b.get("vae_path"),
    ):
        parser.error("The success/failure human references differ")
    reference = torch.stack(
        [
            torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy())
            .permute(2, 0, 1)
            .float()
            / 255
            for p in refpaths
        ]
    )
    _, dino, _ = guidance_modules()
    features, queries, _ = embed_observations(
        reference, failedpaths + successpaths, options.dino_checkpoint, dino
    )
    results, matrices = [], {}
    offset = 0
    for name, ids, annotation in zip(["failed", "success"], chunks, labels):
        rows, scores = evaluate(
            features.numpy(),
            queries[offset : offset + len(ids)].numpy(),
            ids,
            annotation,
            name,
        )
        results.extend(rows)
        matrices[name] = scores
        offset += len(ids)
    options.output.mkdir(parents=True)
    write_csv(options.output / "results.csv", results)
    figures(options.output, results, matrices)
    # Cached embeddings allow follow-up scientific analyses without model inference.
    np.savez_compressed(
        options.output / "embeddings.npz",
        reference=features.numpy(),
        failed=queries[: len(failedpaths)].numpy(),
        success=queries[len(failedpaths) :].numpy(),
    )
    paths = (
        refpaths
        + failedpaths
        + successpaths
        + [options.failed_labels, options.success_labels]
    )
    provenance = dict(
        device="cpu",
        arguments={k: str(v) for k, v in vars(options).items()},
        source_reference=a,
        inputs_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        note="Labels only score matches. Initial observation excluded for every method. Relative zero-change queries abstain; abstentions count as incorrect. Reference frames are PNG previews; success robot images are video-compressed. No policy or simulator run.",
    )
    (options.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for name in matrices:
        for method in list(matrices[name]) + ["stay_at_start"]:
            evaluated = [
                r
                for r in results
                if r["trajectory"] == name and r["method"] == method and r["evaluated"]
            ]
            print(
                f"{name} {method}: {sum(r['stage_correct'] for r in evaluated)}/{len(evaluated)} correct (initial excluded)"
            )
    print(f"Results: {options.output}")


if __name__ == "__main__":
    main()
