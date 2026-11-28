"""Atomic JSON records shared by dataset preparation and experiment runners."""
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



def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default

