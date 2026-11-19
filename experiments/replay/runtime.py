"""Lazy access to guidance helpers without importing the WAM training stack.

CPU replay must not initialize distributed inference or import transformer models.
The isolated package loads the same guidance source files and preprocessing.
"""

import importlib
import sys
import types
from experiments.paths import ROOT


def guidance_modules():
    # Import the small shared modules without wan_va/__init__.py, which loads
    # training/distributed/transformer dependencies irrelevant to CPU replay.
    name = "_temporal_offline_guidance"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "wan_va/guidance")]
        sys.modules[name] = package
    return (
        importlib.import_module(name + ".temporal"),
        importlib.import_module(name + ".dino_reward"),
        importlib.import_module(name + ".reference"),
    )
