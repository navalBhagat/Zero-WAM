"""Stage-00 cache/memory validation; preview unless --submit."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.campaign.stage00 import main

# Compatibility exports for existing commands and analysis callers.
__all__ = ["main", "subprocess"]

if __name__ == "__main__":
    main()
