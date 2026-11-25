"""Common-denominator success tables, collection quality and measured GPU time."""
import json
import random
from pathlib import Path

from experiments.io import read_json, write_csv
from experiments.runtime.records import save_json


def report(directory, plots=False):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    rows, timings = [], []
    for job in manifest["jobs"]:
        root = directory / "jobs" / job["name"]
        summary = read_json(root / "summary.json", {})
        timing = read_json(root / "diagnostics/job_timing.json", {})
        job_status = summary.get("status", "incomplete" if timing.get("exit_code", 0) != 0 else "pending")
        timings.append(dict(job=job["name"], kind=job["kind"], status=job_status,
                            gpu_hours=(timing.get("allocated_seconds") or 0) / 3600))
        if job["kind"] == "evaluate":
            methods = summary.get("methods", {})
            complete = all(methods.get(m, {}).get("status") == "completed" for m in job["methods"])
            for method in job["methods"]:
                rows.append(dict(task=job["task"], scene=job["scene"], method=method,
                                 matched_complete=complete, **methods.get(method, dict(status=job_status))))
    write_csv(directory / "trials.csv", rows)
    write_csv(directory / "timing.csv", timings)
    tables = []
    for task in dict.fromkeys(row["task"] for row in rows):
        for method in dict.fromkeys(row["method"] for row in rows if row["task"] == task):
            group = [r for r in rows if r["task"] == task and r["method"] == method]
            valid = [r for r in group if r["matched_complete"]]
            tables.append(dict(task=task, method=method, planned=len(group), valid=len(valid),
                incomplete=sum(r.get("status") == "incomplete" for r in group),
                successes=sum(r.get("success", False) for r in valid),
                success_percent=100 * sum(r.get("success", False) for r in valid) / len(valid) if valid else None))
    for table in tables:
        group = [r for r in rows if r["task"] == table["task"] and r["method"] == table["method"] and r["matched_complete"]]
        controls = {r["scene"]: r for r in rows if r["task"] == table["task"] and r["method"] == "control" and r["matched_complete"]}
        failures = [r for r in group if not controls[r["scene"]]["success"]]
        successes = [r for r in group if controls[r["scene"]]["success"]]
        differences = [int(r["success"]) - int(controls[r["scene"]]["success"]) for r in group]
        table.update(control_failures=len(failures), rescues=sum(r["success"] for r in failures),
                     control_successes=len(successes), regressions=sum(not r["success"] for r in successes))
        if len(differences) >= 2:
            rng = random.Random(1234)
            bootstrap = sorted(100 * sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(1000))
            table.update(gain_percent=100 * sum(differences) / len(differences),
                         paired_scene_ci_low=bootstrap[25], paired_scene_ci_high=bootstrap[974])
        else:
            table.update(gain_percent=100 * differences[0] if differences else None,
                         paired_scene_ci_low=None, paired_scene_ci_high=None)
    write_csv(directory / "success.csv", tables)
    pairs = [read_json(path) for path in directory.glob("pairs/*/record.json")]
    completed = [p for p in pairs if p["status"] == "completed"]
    errors = [p["reconstruction_error"] for p in completed]
    planned_pairs = sum(len(j.get("pairs", [])) for j in manifest["jobs"])
    quality = dict(planned_pairs=planned_pairs, pending_pairs=planned_pairs - len(pairs),
                   completed_pairs=len(completed), incomplete_pairs=len(pairs) - len(completed),
                   mean_reconstruction_error=sum(errors) / len(errors) if errors else None,
                   mean_pair_seconds=sum(p["seconds"] for p in completed) / len(completed) if completed else None)
    save_json(directory / "collection.json", quality)
    macros = {}
    for method in dict.fromkeys(row["method"] for row in tables):
        rates = [r["success_percent"] for r in tables if r["method"] == method and r["success_percent"] is not None]
        macros[method] = sum(rates) / len(rates) if rates else None
    save_json(directory / "analysis.json", dict(macro_success=macros, collection=quality,
              measured_gpu_hours=sum(t["gpu_hours"] for t in timings)))
    if plots and tables:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tasks = list(dict.fromkeys(row["task"] for row in tables))
        methods = list(dict.fromkeys(row["method"] for row in tables))
        fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 1.3), 5))
        width = .8 / len(methods)
        for offset, method in enumerate(methods):
            values = [next(r["success_percent"] for r in tables if r["task"] == task and r["method"] == method) for task in tasks]
            ax.bar([i + offset * width for i in range(len(tasks))],
                   [float("nan") if v is None else v for v in values], width, label=method)
        ax.set_xticks([i + width * (len(methods) - 1) / 2 for i in range(len(tasks))], tasks, rotation=30, ha="right")
        ax.set_ylabel("Success among completed matched trials (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "success.png", dpi=160)
        plt.close(fig)
    return dict(jobs=timings, success=tables, collection=quality)
