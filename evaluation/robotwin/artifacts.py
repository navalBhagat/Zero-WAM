"""Diagnostic output paths; no model dependencies."""
import os
from pathlib import Path


def diagnostic_path(root, name):
    path=Path(root)/'diagnostics'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    return path
