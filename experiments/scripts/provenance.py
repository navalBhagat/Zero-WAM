"""Compatible source snapshot API and queued-job digest validation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.provenance import ROOT, files, fingerprint, snapshot

# Compatibility exports for existing commands and analysis callers.
__all__ = ["ROOT", "files", "fingerprint", "snapshot"]

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--verify", required=True)
    options = argument_parser.parse_args()
    actual, _ = fingerprint()
    if actual != options.verify:
        raise SystemExit(
            "Source changed after planning: use a fresh manifest or restore the stage source snapshot."
        )
