"""JSON/CSV/log serialization shared by replay, launch and analysis."""

import ast
import csv
import json
from pathlib import Path


def read_json(path: Path, default=None):
    """Read JSON or return the caller's missing-artifact sentinel."""
    return json.loads(path.read_text()) if path.is_file() else default


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the union of row keys in first-seen order, preserving CSV schemas."""
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def log_records(path: Path, marker: str) -> list[dict]:
    """Read Python-dict logger records; ignore malformed/unrelated log lines."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            value = ast.literal_eval(line.split(marker, 1)[1].strip())
            if isinstance(value, dict):
                records.append(value)
        except (ValueError, SyntaxError):
            pass
    return records
