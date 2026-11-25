"""Generate one paired-noise target at a recorded, fixed robot history."""
import json
from pathlib import Path

import torch

from .online import OnlineSession


def collect_context(server, request):
    path = Path(request["collect_context"])
    context = json.loads(path.read_text())
    settings = dict(method="oracle", directory=request["output"], seed=int(request["seed"]), save_demo=False)
    session = OnlineSession(server, dict(prompt="", icl_video_path="",
        icl_latent_path=context["demo_latent"], noise_session=settings))
    session.initial_frame = torch.load(context["initial"], map_location="cpu", weights_only=True)["initial_frame"]
    session.history = [torch.load(item, map_location="cpu", weights_only=True)
                       for item in context["history"]]
    # Each fixed-history pair starts its own stream from request.seed, regardless
    # of the history's chunk index.
    session.sample({})
    record = json.loads((session.root / "chunks.jsonl").read_text().splitlines()[-1])
    record.update(context=str(path.resolve()), pair=str((session.root / "pairs" /
                  f"chunk_{len(session.history):04d}.pt").resolve()))
    return dict(pair=record)
