"""Compact output paths and backwards-compatible readers; no model dependencies."""
import os
from pathlib import Path


def diagnostic_path(root, name):
    path=Path(root)/'diagnostics'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def find_artifact(root, name):
    """Read new runs first; leave historical run directories unchanged."""
    root=Path(root)
    new=root/'diagnostics'/name
    return new if new.exists() else root/name


def detailed_outputs():
    return os.environ.get('RUN_DIAGNOSTICS','0')=='1'
