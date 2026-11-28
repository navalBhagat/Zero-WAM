"""Atomic records and completed episode outcomes."""
import json
from pathlib import Path

from wan_va.noise_inversion.records import save_json


def read_episode(directory):
    records = json.loads((Path(directory) / "config.json").read_text())["episodes"]
    if len(records) != 1 or records[0].get("status") != "completed":
        raise ValueError(f"Episode incomplete: {directory}")
    if not isinstance(records[0].get("success"), bool):
        raise ValueError(f"Episode outcome missing: {directory}")
    return records[0]
