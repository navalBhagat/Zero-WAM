"""Slurm job timing, completion and client preflight entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.job_report import main

if __name__ == "__main__":
    main()
