"""Paired-trial tables with common denominators and explicit incomplete counts."""
from pathlib import Path

from experiments.io import read_json, write_csv


def collect(directory: Path) -> tuple[list[dict], list[dict]]:
    manifest = read_json(directory / "manifest.json")
    rows = []
    job_timings = {}
    for job in manifest.get("jobs", []):
        timing = read_json(directory / "jobs" / job["name"] / "diagnostics/job_timing.json", {})
        for path in job["run_paths"]:
            job_timings[path] = timing
    for run in manifest["runs"]:
        root = directory / run["path"]
        summary = read_json(root / "summary.json", {})
        timing = read_json(root / "diagnostics/job_timing.json", job_timings.get(run["path"], {}))
        if not summary and timing.get("exit_code", 0) != 0:
            summary = dict(status="incomplete", error="Job failed before trial completion; see diagnostics")
        errors = summary.get("inversion", {}).get("relative_errors", [])
        reconstruction = dict(mean_reconstruction_error=sum(errors) / len(errors) if errors else None,
                              max_reconstruction_error=max(errors) if errors else None)
        rows.append(dict(run["settings"], **reconstruction, **{k: v for k, v in summary.items()
                    if k not in run["settings"] and k != "inversion"},
                    trial_path=run["path"], result_status=summary.get("status", "pending")))
    tables = []
    for task in dict.fromkeys(row["task"] for row in rows):
        group = [row for row in rows if row["task"] == task]
        valid = [row for row in group if row["result_status"] == "completed"]
        failed = [row for row in valid if not row["control_success"]]
        succeeded = [row for row in valid if row["control_success"]]
        registry = read_json(directory / "trials" / task / "scenes.json", {})
        scene_attempts = [attempt for slot in registry.get("slots", {}).values() for attempt in slot["attempts"]]
        record = dict(task=task, planned=len(group), valid_pairs=len(valid),
                      infeasible_scene_checks=sum(a["status"] == "infeasible" for a in scene_attempts),
                      scene_check_errors=sum(a["status"] == "error" for a in scene_attempts),
                      incomplete=sum(row["result_status"] == "incomplete" for row in group),
                      pending=sum(row["result_status"] == "pending" for row in group),
                      distinct_accepted_scenes=len({row["simulator_seed"] for row in valid}))
        for key in ("source", "control", "replay"):
            count = sum(row[key + "_success"] for row in valid)
            record[key + "_successes"] = count
            record[key + "_success_pct"] = 100 * count / len(valid) if valid else None
        record.update(control_failures=len(failed), rescues=sum(row["replay_success"] for row in failed),
                      control_successes=len(succeeded), regressions=sum(not row["replay_success"] for row in succeeded))
        record["rescue_pct"] = 100 * record["rescues"] / len(failed) if failed else None
        record["regression_pct"] = 100 * record["regressions"] / len(succeeded) if succeeded else None
        for name in ("control", "replay"):
            record[name + "_success_within_noise_horizon"] = sum(row.get(name + "_success_within_source_horizon", False) for row in valid)
            record[name + "_extended"] = sum(row.get(name + "_extended_beyond_saved_noise", False) for row in valid)
        for outcome in (True, False):
            subgroup = [row for row in valid if row["source_success"] == outcome]
            label = "source_success" if outcome else "source_failure"
            record[label + "_pairs"] = len(subgroup)
            record[label + "_replay_successes"] = sum(row["replay_success"] for row in subgroup)
            record[label + "_control_failures"] = sum(not row["control_success"] for row in subgroup)
            record[label + "_rescues"] = sum(not row["control_success"] and row["replay_success"] for row in subgroup)
        tables.append(record)
    return rows, tables


def report(directory: Path, *, plots: bool = True) -> None:
    rows, tables = collect(directory)
    write_csv(directory / "trials.csv", rows)
    write_csv(directory / "results.csv", tables)
    lines = ["# Noise inversion campaign", "", "All success columns share the valid-pair denominator.",
             "Incomplete jobs are excluded from all three columns and listed separately.", "",
             "| Task | Valid/planned | ICL | No ICL | No ICL + reversal | Reversal before fallback | Rescues | Incomplete |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in tables:
        def cell(key):
            pct = row[key + "_success_pct"]
            return f"{row[key + '_successes']}/{row['valid_pairs']} ({pct:.1f}%)" if pct is not None else "—"
        rescues = f"{row['rescues']}/{row['control_failures']}" if row["control_failures"] else "—"
        before_fallback = f"{row['replay_success_within_noise_horizon']}/{row['valid_pairs']}" if row["valid_pairs"] else "—"
        lines.append(f"| {row['task']} | {row['valid_pairs']}/{row['planned']} | {cell('source')} | "
                     f"{cell('control')} | {cell('replay')} | {before_fallback} | {rescues} | {row['incomplete']} |")
    lines += ["", "Reversal may continue with seeded Gaussian noise after the saved sequence ends.",
              "See results.csv for within-horizon success, fallback counts, source-success/failure breakdowns,",
              "and rescue/regression denominators. Scene/noise repetitions are correlated."]
    lines += ["", "## Interpretation", ""]
    for row in tables:
        if not row["valid_pairs"]:
            lines.append(f"- {row['task']}: no complete matched trials yet.")
            continue
        gain = row["replay_success_pct"] - row["control_success_pct"]
        lines.append(f"- {row['task']}: reversal vs control {gain:+.1f} percentage points; "
                     f"{row['rescues']} rescues and {row['regressions']} regressions. "
                     f"{row['replay_success_within_noise_horizon']} reversal successes occurred before fallback; "
                     f"{row['replay_extended']} replays continued beyond the saved sequence.")
    lines += ["", "These paired observations do not establish unseen-task generalization or statistical significance."]
    if plots:
        from .plots import write_plots
        write_plots(directory, tables)
        lines += ["", "![Paired success rates](plots/success.png)", "",
                  "![Rescues and regressions](plots/rescue_regression.png)"]
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    manifest = read_json(directory / "manifest.json")
    accounting = []
    rate = manifest["sbu_per_hour"]
    units = manifest.get("jobs", manifest["runs"])
    for unit in units:
        grouped = "run_paths" in unit
        relative = str(Path("jobs") / unit["name"]) if grouped else unit["path"]
        root = directory / relative
        seconds = 0
        timings = list(root.glob("diagnostics/job_timing.json")) + list(root.glob("attempts/*/diagnostics/job_timing.json"))
        for path in timings:
            seconds += read_json(path).get("allocated_seconds", 0)
        submissions = unit.get("submissions", [])
        accounting.append(dict(task=unit["task"] if grouped else unit["settings"]["task"], path=relative,
            accounts=";".join(sorted({a["account"] for a in submissions})), measured_seconds=seconds,
            cluster=manifest.get("cluster", "snellius"), measured_gpu_hours=seconds / 3600,
            estimated_sbus=seconds / 3600 * rate if rate is not None else None,
            submitted_sbu_ceiling=sum(a["sbu_ceiling"] for a in submissions
                if a.get("status") != "submission_failed") if rate is not None else None))
    write_csv(directory / "costs.csv", accounting)
    attempts = []
    for task in dict.fromkeys(row["task"] for row in rows):
        registry = read_json(directory / "trials" / task / "scenes.json", {})
        for slot, entry in registry.get("slots", {}).items():
            for index, attempt in enumerate(entry["attempts"]):
                attempts.append(dict(task=task, requested_scene_seed=int(slot),
                                     accepted_scene_seed=entry["accepted_seed"],
                                     attempt=index + 1, **attempt))
    write_csv(directory / "scene_attempts.csv", attempts)

