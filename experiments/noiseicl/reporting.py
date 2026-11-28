"""Common-denominator success tables, collection quality and measured GPU time."""

import csv
import json
import random
from pathlib import Path

from experiments.io import read_json, write_csv
from experiments.runtime.records import save_json


def quality_report(directory, output=None, check_tensors=True, worst=10):
    from wan_va.noise_inversion.quality import inspect_dataset

    result = inspect_dataset(directory, check_tensors=check_tensors, worst=worst)
    output = Path(output) if output else Path(directory) / "quality"
    save_json(output / "summary.json", result)
    write_csv(output / "worst_examples.csv", result["worst_examples"])
    task_rows = []
    for task, values in result["tasks"].items():
        row = dict(task=task, **values)
        row.pop("splits")
        row.update(
            {
                f"{split}_pairs": values["splits"].get(split, 0)
                for split in ("train", "val", "test")
            }
        )
        task_rows.append(row)
    write_csv(output / "tasks.csv", task_rows)
    return result


def campaign_rows(directory):
    manifest = read_json(directory / "manifest.json")
    rows, timings = [], []
    for job in manifest["jobs"]:
        root = directory / "jobs" / job["name"]
        summary = read_json(root / "summary.json", {})
        timing = read_json(root / "diagnostics/job_timing.json", {})
        job_status = summary.get(
            "status", "incomplete" if timing.get("exit_code", 0) != 0 else "pending"
        )
        timings.append(
            dict(
                job=job["name"],
                kind=job["kind"],
                status=job_status,
                gpu_hours=(timing.get("allocated_seconds") or 0) / 3600,
            )
        )
        if job["kind"] == "evaluate":
            methods = summary.get("methods", {})
            complete = all(
                methods.get(m, {}).get("status") == "completed" for m in job["methods"]
            )
            for method in job["methods"]:
                rows.append(
                    dict(
                        campaign=str(directory.resolve()),
                        sampling_seed=manifest["settings"]["sampling_seed"],
                        task=job["task"],
                        scene=job["scene"],
                        method=method,
                        matched_complete=complete,
                        **methods.get(method, dict(status=job_status)),
                    )
                )
    return manifest, rows, timings


def combined_rows(campaigns, output):
    sources = [Path(path).resolve() for path in campaigns]
    if len(set(sources)) != len(sources) or output.resolve() in sources:
        raise ValueError("Use distinct input campaigns and a separate output directory")
    rows, timings = [], []
    for source in sources:
        with (source / "trials.csv").open() as stream:
            for row in csv.DictReader(stream):
                row["campaign"] = str(source)
                for key in ("matched_complete", "success", "fallback"):
                    row[key] = row.get(key, "").lower() == "true"
                rows.append(row)
        with (source / "timing.csv").open() as stream:
            for row in csv.DictReader(stream):
                row.update(campaign=str(source), gpu_hours=float(row["gpu_hours"]))
                timings.append(row)
    return dict(jobs=[]), rows, timings


def report(directory, plots=False, campaigns=None):
    directory = Path(directory)
    if campaigns:
        manifest, rows, timings = combined_rows(campaigns, directory)
    else:
        manifest, rows, timings = campaign_rows(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "trials.csv", rows)
    write_csv(directory / "timing.csv", timings)
    tables = []
    for task in dict.fromkeys(row["task"] for row in rows):
        for method in dict.fromkeys(
            row["method"] for row in rows if row["task"] == task
        ):
            group = [r for r in rows if r["task"] == task and r["method"] == method]
            valid = [r for r in group if r["matched_complete"]]
            tables.append(
                dict(
                    task=task,
                    method=method,
                    planned=len(group),
                    valid=len(valid),
                    incomplete=sum(r.get("status") == "incomplete" for r in group),
                    successes=sum(r.get("success", False) for r in valid),
                    success_percent=(
                        100 * sum(r.get("success", False) for r in valid) / len(valid)
                        if valid
                        else None
                    ),
                )
            )
    for table in tables:
        group = [
            r
            for r in rows
            if r["task"] == table["task"]
            and r["method"] == table["method"]
            and r["matched_complete"]
        ]
        controls = {
            (r.get("campaign"), str(r["scene"])): r
            for r in rows
            if r["task"] == table["task"]
            and r["method"] == "control"
            and r["matched_complete"]
        }
        failures = [
            r
            for r in group
            if not controls[(r.get("campaign"), str(r["scene"]))]["success"]
        ]
        successes = [
            r
            for r in group
            if controls[(r.get("campaign"), str(r["scene"]))]["success"]
        ]
        differences = [
            int(r["success"])
            - int(controls[(r.get("campaign"), str(r["scene"]))]["success"])
            for r in group
        ]
        table.update(
            control_failures=len(failures),
            rescues=sum(r["success"] for r in failures),
            control_successes=len(successes),
            regressions=sum(not r["success"] for r in successes),
        )
        if len(differences) >= 2:
            rng = random.Random(1234)
            # Different noise seeds for the same scene are correlated. Resample
            # whole scene groups while retaining method pairing within campaigns.
            clusters = {}
            for row, difference in zip(group, differences):
                clusters.setdefault(str(row["scene"]), []).append(difference)
            scenes = list(clusters)
            bootstrap = []
            for _ in range(1000):
                sample = [
                    value
                    for scene in rng.choices(scenes, k=len(scenes))
                    for value in clusters[scene]
                ]
                bootstrap.append(100 * sum(sample) / len(sample))
            bootstrap.sort()
            table.update(
                gain_percent=100 * sum(differences) / len(differences),
                paired_scene_ci_low=bootstrap[25],
                paired_scene_ci_high=bootstrap[974],
            )
        else:
            table.update(
                gain_percent=100 * differences[0] if differences else None,
                paired_scene_ci_low=None,
                paired_scene_ci_high=None,
            )
    write_csv(directory / "success.csv", tables)
    pairs = [read_json(path) for path in directory.glob("pairs/*/record.json")]
    completed = [p for p in pairs if p["status"] == "completed"]
    errors = [p["reconstruction_error"] for p in completed]
    planned_pairs = sum(len(j.get("pairs", [])) for j in manifest["jobs"])
    quality = dict(
        planned_pairs=planned_pairs,
        pending_pairs=planned_pairs - len(pairs),
        completed_pairs=len(completed),
        incomplete_pairs=len(pairs) - len(completed),
        mean_reconstruction_error=sum(errors) / len(errors) if errors else None,
        mean_pair_seconds=(
            sum(p["seconds"] for p in completed) / len(completed) if completed else None
        ),
    )
    save_json(directory / "collection.json", quality)
    macros = {}
    for method in dict.fromkeys(row["method"] for row in tables):
        rates = [
            r["success_percent"]
            for r in tables
            if r["method"] == method and r["success_percent"] is not None
        ]
        macros[method] = sum(rates) / len(rates) if rates else None
    save_json(
        directory / "analysis.json",
        dict(
            macro_success=macros,
            collection=quality,
            measured_gpu_hours=sum(t["gpu_hours"] for t in timings),
            source_campaigns=(
                [str(Path(p).resolve()) for p in campaigns] if campaigns else None
            ),
        ),
    )
    if plots and tables:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tasks = list(dict.fromkeys(row["task"] for row in tables))
        methods = list(dict.fromkeys(row["method"] for row in tables))
        fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 1.3), 5))
        width = 0.8 / len(methods)
        for offset, method in enumerate(methods):
            values = [
                next(
                    r["success_percent"]
                    for r in tables
                    if r["task"] == task and r["method"] == method
                )
                for task in tasks
            ]
            ax.bar(
                [i + offset * width for i in range(len(tasks))],
                [float("nan") if v is None else v for v in values],
                width,
                label=method,
            )
        ax.set_xticks(
            [i + width * (len(methods) - 1) / 2 for i in range(len(tasks))],
            tasks,
            rotation=30,
            ha="right",
        )
        ax.set_ylabel("Success among completed matched trials (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "success.png", dpi=160)
        plt.close(fig)
    return dict(jobs=timings, success=tables, collection=quality)
