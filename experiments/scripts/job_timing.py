"""Compatibility entry point for historical job scripts; see job_report.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.job_report import record_timing as record, main

# Compatibility exports for existing commands and analysis callers.
__all__ = ["record", "main"]

if __name__ == "__main__":
    main(["timing", *sys.argv[1:]])
