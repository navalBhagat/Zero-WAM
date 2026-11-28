"""Train the residual adapter; preserve grouped splits and resumable checkpoints."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from .adapter import AdapterConfig, NoiseAdapter
from .dataset import SPLITS, load_examples as examples
from .latents import reshape_latent


def load_example(record, context, device):
    tensors = torch.load(record["pair"], map_location="cpu", weights_only=True)
    demo_record = torch.load(
        context["demo_latent"], map_location="cpu", weights_only=True
    )
    demo = reshape_latent(demo_record["latent"], demo_record)
    history = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in context["history"]
    ]
    if history:
        robot = torch.cat([item["video"] for item in history], dim=2)
        actions = torch.cat([item["actions"] for item in history], dim=2)
    else:
        robot = torch.load(context["initial"], map_location="cpu", weights_only=True)[
            "initial_frame"
        ]
        actions = torch.zeros(1, 30, 1, 16, 1)
    values = [tensors[key] for key in ("original", "recovered", "mask")]
    values += [demo, robot, actions]
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError(f"Nonfinite training example: {record['pair']}")
    return [value.to(device=device, dtype=torch.float32) for value in values]


def loss_for(model, record, context, device):
    original, target, mask, demo, robot, actions = load_example(record, context, device)
    prediction = model(original, demo, robot, actions, context["chunk"])
    denominator = mask.sum().clamp_min(1)
    loss = ((prediction - target).square() * mask).sum() / denominator
    baseline = ((original - target).square() * mask).sum() / denominator
    return loss, baseline


@torch.no_grad()
def evaluate(model, records, device):
    model.eval()
    losses, baselines = [], []
    for record, context in records:
        loss, baseline = loss_for(model, record, context, device)
        losses.append(float(loss))
        baselines.append(float(baseline))
    return dict(
        mse=sum(losses) / len(losses),
        zero_residual_mse=sum(baselines) / len(baselines),
        count=len(losses),
    )


def training_partitions(records, overfit_pairs, seed):
    partitions = {
        split: [(row, context) for row, context in records if context["split"] == split]
        for split in SPLITS
    }
    if any(not rows for rows in partitions.values()):
        raise ValueError("Training requires nonempty grouped train/val/test partitions")
    if overfit_pairs:
        if overfit_pairs > len(partitions["train"]):
            raise ValueError(
                "Requested overfit subset is larger than the training split"
            )
        partitions["train"] = random.Random(seed).sample(
            partitions["train"], overfit_pairs
        )
    return partitions


def run_training_step(model, optimizer, records, device, batch_size):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for _ in range(batch_size):
        record, context = random.choice(records)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss, _ = loss_for(model, record, context, device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        (loss / batch_size).backward()
        total_loss += float(loss.detach()) / batch_size
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total_loss


def save_training_state(path, model, optimizer, step, best, signature):
    state = dict(
        config=asdict(model.config),
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        best=best,
        signature=signature,
        random_state=random.getstate(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=(
            torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []
        ),
    )
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, signature, device):
    if not path.exists():
        return 0, float("inf")
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("signature") != signature or state["config"] != asdict(model.config):
        raise ValueError(
            "Checkpoint architecture, dataset, or training settings changed; use a new output directory"
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["random_state"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return state["step"], state["best"]


def train(dataset, output, settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = examples(dataset)
    seed = settings["sampling_seed"]
    overfit_pairs = settings.get("overfit_pairs", 0)
    partitions = training_partitions(records, overfit_pairs, seed)
    selection_split = "train" if overfit_pairs else "val"
    selection_records = partitions[selection_split]
    signature_data = dict(
        records=[
            (row["pair"], row.get("context_sha256"), row.get("source_digest"))
            for row, _ in records
        ],
        seed=seed,
        batch=settings["effective_batch"],
        overfit_pairs=overfit_pairs,
    )
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode()
    ).hexdigest()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AdapterConfig(
        width=settings["width"],
        layers=settings["layers"],
        heads=settings["heads"],
        architecture=settings["architecture"],
    )
    model = NoiseAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    start, best = restore_training_state(
        output / "latest.pt", model, optimizer, signature, device
    )
    for step in range(start, settings["train_steps"]):
        train_loss = run_training_step(
            model, optimizer, partitions["train"], device, settings["effective_batch"]
        )
        if (step + 1) % 100 != 0 and step + 1 != settings["train_steps"]:
            continue
        metrics = evaluate(model, selection_records, device)
        if metrics["mse"] < best:
            best = metrics["mse"]
            temporary = output / "adapter.tmp"
            torch.save(
                dict(config=asdict(config), model=model.state_dict(), step=step + 1),
                temporary,
            )
            temporary.replace(output / "adapter.pt")
        progress = dict(
            step=step + 1,
            train_mse=train_loss,
            selection_split=selection_split,
            selection=metrics,
        )
        print(json.dumps(progress), flush=True)
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(progress) + "\n")
        save_training_state(
            output / "latest.pt", model, optimizer, step + 1, best, signature
        )

    saved = torch.load(output / "adapter.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    # Overfit diagnostics measure memorization, without opening the test tensors.
    test = None if overfit_pairs else evaluate(model, partitions["test"], device)
    return dict(
        status="completed",
        mode="overfit" if overfit_pairs else "train",
        selection_split=selection_split,
        best_selection_mse=best,
        best_validation_mse=None if overfit_pairs else best,
        selected_metrics=evaluate(model, selection_records, device),
        test=test,
        checkpoint=str((output / "adapter.pt").resolve()),
        split_counts={split: len(rows) for split, rows in partitions.items()},
    )
