"""CPU-only temporal target replay. Existing phase command remains supported."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.replay.temporal import (
    main,
    replay,
    figures,
    load_reference,
    guidance_modules,
)

# Compatibility exports for existing commands and analysis callers.
__all__ = ["main", "replay", "figures", "load_reference", "guidance_modules"]

if __name__ == "__main__":
    main()
