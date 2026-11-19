"""Compatible CLI for render, baseline, guidance and precision campaigns."""

from pathlib import Path
import subprocess  # Shared module retained for callers that patch submission in tests.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.paths import ROOT
from experiments.campaign.config import CONTROLLED, MODES, parser
from experiments.campaign.plans import specifications
from experiments.campaign.resume import compatible_client_changes, extend_baselines
from experiments.campaign.cli import main

# Compatibility exports for existing commands and analysis callers.
__all__ = [
    "ROOT",
    "CONTROLLED",
    "MODES",
    "parser",
    "specifications",
    "compatible_client_changes",
    "extend_baselines",
    "main",
    "subprocess",
]

if __name__ == "__main__":
    main()
