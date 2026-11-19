"""Run the same-scene temporal guidance preset; preview unless --submit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.presets import command, main

# Compatibility exports for existing commands and analysis callers.
__all__ = ["command", "main"]

if __name__ == "__main__":
    main()
