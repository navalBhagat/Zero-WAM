"""Residual-only training with grouped validation, fixed seeds and best checkpoints."""
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from experiments.io import read_json
from experiments.runtime.records import save_json
from wan_va.noise_inversion.adapter import AdapterConfig, NoiseAdapter
from wan_va.noise_inversion.latents import reshape_latent
from .config import UNSEEN_TASKS


def examples(dataset):
    rows = []
    for path in sorted(Path(dataset).glob("pairs/*/record.json")):
        row = read_json(path)
        if row.get("status") != "completed" or row.get("fallback"):
            continue
        if row.get("context_sha256") and hashlib.sha256(Path(row["context"]).read_bytes()).hexdigest() != row["context_sha256"]:
            raise ValueError("Training context changed since collection")
        context = read_json(Path(row["context"]))
        if context["task"] in UNSEEN_TASKS:
            raise ValueError("Unseen-task example found in training dataset")
        rows.append((row, context))
    owners = {}
    for _, context in rows:
        for key in (("scene", context["task"], str(context["scene_id"])),
                    ("seed", context["task"], int(context["simulator_seed"])),
                    ("demo_id", context["task"], str(context["demo_id"])),
                    ("demo", context["demo_latent"])):
            previous = owners.setdefault(key, context["split"])
            if previous != context["split"]:
                raise ValueError("Dataset leaks a scene/demo across splits")
    return rows


def load_example(row, context, device):
    tensors = torch.load(row["pair"], map_location="cpu", weights_only=True)
    demo_record = torch.load(context["demo_latent"], map_location="cpu", weights_only=True)
    demo = reshape_latent(demo_record["latent"], demo_record)
    history = [torch.load(path, map_location="cpu", weights_only=True) for path in context["history"]]
    robot = (torch.cat([item["video"] for item in history], dim=2) if history else
             torch.load(context["initial"], map_location="cpu", weights_only=True)["initial_frame"])
    actions = (torch.cat([item["actions"] for item in history], dim=2) if history else
               torch.zeros(1, 30, 1, 16, 1))
    values = [tensors[key] for key in ("original", "recovered", "mask")] + [demo, robot, actions]
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError(f"Nonfinite training example: {row['pair']}")
    return [value.to(device=device, dtype=torch.float32) for value in values]


def loss_for(model, row, context, device):
    original, target, mask, demo, robot, actions = load_example(row, context, device)
    prediction = model(original, demo, robot, actions, context["chunk"])
    denominator = mask.sum().clamp_min(1)
    loss = ((prediction - target).square() * mask).sum() / denominator
    baseline = ((original - target).square() * mask).sum() / denominator
    return loss, baseline


@torch.no_grad()
def evaluate(model, records, device):
    model.eval()
    losses, baselines = [], []
    for row, context in records:
        loss, baseline = loss_for(model, row, context, device)
        losses.append(float(loss))
        baselines.append(float(baseline))
    return dict(mse=sum(losses) / len(losses), zero_residual_mse=sum(baselines) / len(baselines), count=len(losses))


def train(dataset, output, settings):
    records = examples(dataset)
    partitions = {split: [record for record in records if record[1]["split"] == split]
                  for split in ("train", "val", "test")}
    if any(not rows for rows in partitions.values()):
        raise ValueError("Training requires nonempty grouped train/val/test partitions")
    if settings.get("overfit_pairs"):
        partitions["train"] = partitions["train"][:settings["overfit_pairs"]]
    signature = hashlib.sha256(json.dumps([(row["pair"], row.get("context_sha256"),
        row.get("source_digest")) for row, _ in records], sort_keys=True).encode()).hexdigest()
    seed = settings["sampling_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AdapterConfig(width=settings["width"], layers=settings["layers"],
                           heads=settings["heads"], architecture=settings["architecture"])
    model = NoiseAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = output / "latest.pt"
    start, best = 0, float("inf")
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["config"] != asdict(config) or saved["dataset_signature"] != signature:
            raise ValueError("Checkpoint architecture or dataset changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["step"], saved["best"]
        random.setstate(saved["random_state"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    for step in range(start, settings["train_steps"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(settings["effective_batch"]):
            row, context = random.choice(partitions["train"])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = loss_for(model, row, context, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / settings["effective_batch"]).backward()
            total += float(loss.detach()) / settings["effective_batch"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == settings["train_steps"]:
            metrics = evaluate(model, partitions["val"], device)
            if metrics["mse"] < best:
                best = metrics["mse"]
                temporary = output / "adapter.tmp"
                torch.save(dict(config=asdict(config), model=model.state_dict(), step=step + 1), temporary)
                temporary.replace(output / "adapter.pt")
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=step + 1, train_mse=total, validation=metrics)) + "\n")
            temporary = output / "latest.tmp"
            torch.save(dict(config=asdict(config), model=model.state_dict(), optimizer=optimizer.state_dict(),
                            step=step + 1, best=best, dataset_signature=signature, random_state=random.getstate(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                            if device.type == "cuda" else []), temporary)
            temporary.replace(checkpoint)
    saved = torch.load(output / "adapter.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    test = evaluate(model, partitions["test"], device)
    return dict(status="completed", best_validation_mse=best, test=test,
                checkpoint=str((output / "adapter.pt").resolve()), split_counts={k: len(v) for k, v in partitions.items()})
