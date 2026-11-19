"""Analyze a campaign into results.csv, success.png and cost.png. CPU-only."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.analysis.io import read, write_csv, log_records
from experiments.analysis.metrics import (
    COMPONENTS,
    breakdown,
    wilson,
    summaries,
    pairs,
    accounting_from_sacct,
)
from experiments.analysis.collect import collect
from experiments.analysis.plots import diagnostic_plots as plots
from experiments.analysis import report


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("manifest", type=Path)
    argument_parser.add_argument("--runs-root", type=Path)
    argument_parser.add_argument("--output", type=Path)
    argument_parser.add_argument("--sacct", action="store_true")
    argument_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write detailed legacy tables/plots into diagnostics/",
    )
    argument_parser.add_argument(
        "--compute-model-config",
        type=Path,
        help="Historical transformer checkpoint config.json; otherwise architecture assumptions are explicit",
    )
    options = argument_parser.parse_args()
    manifest = read(options.manifest)
    output = options.output or options.manifest.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    local_root = options.manifest.resolve().parent / "runs"
    root = options.runs_root or (local_root if local_root.is_dir() else None)
    runs, episodes, chunks, guidance = collect(
        manifest, root, options.compute_model_config
    )
    rows = report.results(runs, episodes, wilson)
    write_csv(output / "results.csv", rows)
    try:
        report.plots(output, rows)
    except ImportError as error:
        raise SystemExit(
            f"CSV written; plots need experiments/analysis/requirements.txt: {error}"
        )
    report.clean_legacy(output)
    if options.diagnostics or options.sacct:
        detail = output / "diagnostics"
        detail.mkdir(exist_ok=True)
        if options.diagnostics:
            for name, values in [
                ("runs", runs),
                ("episodes", episodes),
                ("summary", summaries(episodes)),
                ("paired", pairs(episodes, manifest["stage"])),
                ("server_chunks", chunks),
                ("guidance", guidance),
            ]:
                write_csv(detail / f"{name}.csv", values)
            plots(detail, runs, episodes, chunks, guidance)
        if options.sacct:
            ids = sorted({str(r["job_id"]) for r in runs if r["job_id"]})
            if ids:
                result = subprocess.run(
                    [
                        "sacct",
                        "-j",
                        ",".join(ids),
                        "-X",
                        "-P",
                        "--units=G",
                        "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode",
                    ],
                    text=True,
                    capture_output=True,
                )
                (detail / "sacct.txt").write_text(result.stdout + result.stderr)
                if result.returncode:
                    print("WARNING: sacct failed; see diagnostics/sacct.txt")
                else:
                    write_csv(
                        detail / "accounting.csv",
                        accounting_from_sacct(
                            result.stdout, manifest["estimate"]["sbu_per_hour"]
                        ),
                    )
    print(f"Outputs: {output}/results.csv, success.png, cost.png")
    if any(e.get("operator_counting_available") for e in episodes):
        print(
            "Operator FLOPs include executed forward/backward paths; inspect coverage and dense-attention assumptions."
        )
    else:
        print(
            "Historical FLOPs are a partial transformer subtotal; full operator counts were not recorded."
        )


# Compatibility exports for existing commands and analysis callers.
__all__ = [
    "read",
    "write_csv",
    "log_records",
    "COMPONENTS",
    "breakdown",
    "wilson",
    "summaries",
    "pairs",
    "accounting_from_sacct",
    "collect",
    "plots",
    "main",
]

if __name__ == "__main__":
    main()
