"""Compatible CLI for the stage-00 memory/cache report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.analysis.stage00 import main, jsonlines

# Compatibility exports for existing commands and analysis callers.
__all__ = ["main", "jsonlines"]

if __name__ == "__main__":
    main()
