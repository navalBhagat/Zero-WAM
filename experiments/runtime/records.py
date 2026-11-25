"""Atomic records and completed episode outcomes."""
import json
import os
from pathlib import Path
from uuid import uuid4


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_episode(directory):
    records = json.loads((Path(directory) / "config.json").read_text())["episodes"]
    if len(records) != 1 or records[0].get("status") != "completed":
        raise ValueError(f"Episode incomplete: {directory}")
    if not isinstance(records[0].get("success"), bool):
        raise ValueError(f"Episode outcome missing: {directory}")
    return records[0]
