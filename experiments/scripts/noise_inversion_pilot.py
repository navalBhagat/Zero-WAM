"""Run one ICL episode, invert its videos, and replay without ICL per task."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.noise_inversion.config import TASKS, PILOT_TASKS
from experiments.noise_inversion.pilot import launch_client, run_pilot, run_controls


def main() -> None:
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control-source", type=Path, help="Existing noise_inversion directory; append controls only")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(PILOT_TASKS))
    parser.add_argument("--sampling-seed", type=int, default=1234)
    parser.add_argument("--simulator-seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "29056")))
    args = parser.parse_args()
    if bool(args.output) == bool(args.control_source):
        parser.error("Choose either --output or --control-source")
    if args.control_source:
        run_controls(args.control_source.resolve(), args.tasks, launch_client)
        return
    if args.sampling_seed < 0 or args.simulator_seed < 0:
        parser.error("Seeds must be nonnegative")
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    run_pilot(args.output.resolve(), args.tasks, args.sampling_seed, args.simulator_seed,
              client.infer, launch_client)


if __name__ == "__main__":
    main()
