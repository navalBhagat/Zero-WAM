"""Job lifecycle helpers used by Slurm scripts; no model imports for reporting."""

import argparse
import json
import os
from pathlib import Path


def record_timing(values: list) -> dict:
    """Record wall seconds for completed and interrupted jobs, excluding queue wait."""
    status, start, server_start, ready, client_start, client_end, end = map(int, values)

    def delta(begin, finish):
        return finish - begin if begin > 0 and finish >= begin else None

    return dict(
        exit_code=status,
        allocated_seconds=delta(start, end),
        pre_server_seconds=delta(start, server_start),
        server_startup_seconds=delta(server_start, ready),
        client_seconds=delta(client_start, client_end),
        client_started=bool(client_start),
        client_completed=bool(client_end),
        start_epoch=start,
        end_epoch=end,
        note="Approximate job-script wall time; sacct ElapsedRaw is authoritative. "
        "Client includes expert checks, resets, rollout, and video export.",
    )


def check_completion(root: Path, expected: int, require_success: bool = False) -> None:
    """A successful process exit is insufficient if requested episodes are missing."""
    config = json.loads((root / "config.json").read_text())
    completed = [
        episode
        for episode in config.get("episodes", [])
        if episode.get("status") == "completed"
    ]
    print(root / "results.csv")
    print(
        f"Completed {len(completed)}/{expected}; successes {sum(bool(episode.get('success')) for episode in completed)}"
    )
    if len(completed) != expected:
        raise SystemExit(
            "Completed episode count differs from TEST_NUM; inspect diagnostics/client.log."
        )
    if require_success and not all(episode.get("success") for episode in completed):
        raise SystemExit(
            "Stage-00 cup control did not succeed; dependent tests must not start."
        )


def preflight_client() -> None:
    """Run only inside the Robotwin environment, before loading the GPU policy."""
    import importlib

    importlib.import_module("evaluation.robotwin.eval_policy_client_openpi")
    from evaluation.robotwin.render_probe import RenderProbe

    # __init__ catches errors; call the scene setup directly so failure is nonzero.
    probe = RenderProbe.__new__(RenderProbe)
    probe.setup_scene()
    print("SAPIEN renderer initialization passed; rollout will test camera rendering.")


def main(argv=None) -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    commands = argument_parser.add_subparsers(dest="command", required=True)
    timing = commands.add_parser(
        "timing", help="Write allocation/startup/client durations"
    )
    timing.add_argument("output", type=Path)
    timing.add_argument(
        "values",
        nargs=7,
        help="status, start, server-start, ready, client-start, client-end, end",
    )
    complete = commands.add_parser("complete", help="Check requested episodes finished")
    complete.add_argument("root", type=Path)
    complete.add_argument("expected", type=int)
    commands.add_parser("preflight", help="Validate client imports and renderer setup")
    options = argument_parser.parse_args(argv)
    if options.command == "timing":
        options.output.write_text(
            json.dumps(record_timing(options.values), indent=2) + "\n"
        )
    elif options.command == "complete":
        check_completion(
            options.root,
            options.expected,
            os.environ.get("STAGE_REQUIRE_SUCCESS") == "1",
        )
    else:
        preflight_client()
