"""Small CPU runs cover resume state and the distinction between overfit/test."""

import pytest
import torch

from wan_va.noise_inversion.adapter import load_adapter
from wan_va.noise_inversion.records import save_json
from wan_va.noise_inversion.training import train


@pytest.fixture
def dataset(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        demo = tmp_path / f"{split}_demo.pt"
        torch.save(dict(latent=torch.randn(1, 48, 2, 4, 4)), demo)
        initial = tmp_path / f"{split}_initial.pt"
        torch.save(dict(initial_frame=torch.randn(1, 48, 1, 4, 4)), initial)
        context = tmp_path / f"{split}.json"
        save_json(
            context,
            dict(
                task="seen_example",
                scene_id=split,
                simulator_seed=None,
                demo_id=split,
                demo_latent=str(demo),
                initial=str(initial),
                history=[],
                split=split,
                chunk=0,
            ),
        )
        destination = dataset / "pairs" / split
        destination.mkdir(parents=True)
        original = torch.randn(1, 48, 2, 4, 4)
        mask = torch.ones_like(original)
        mask[:, :, :1] = 0
        pair = destination / "pair.pt"
        torch.save(dict(original=original, recovered=original * 0.5, mask=mask), pair)
        save_json(
            destination / "record.json",
            dict(
                status="completed", context=str(context), pair=str(pair), fallback=False
            ),
        )
    return dataset


def settings(**overrides):
    values = dict(
        sampling_seed=1234,
        train_steps=2,
        effective_batch=1,
        width=32,
        layers=1,
        heads=4,
        architecture="vit",
        overfit_pairs=0,
    )
    return dict(values, **overrides)


def test_training_checkpoint_restores_model_optimizer_and_rng(dataset, tmp_path):
    output = tmp_path / "resumed"
    assert train(dataset, output, settings())["status"] == "completed"
    assert load_adapter(output / "adapter.pt", "cpu").config.width == 32
    train(dataset, output, settings(train_steps=3))
    train(dataset, tmp_path / "continuous", settings(train_steps=3))
    resumed = torch.load(output / "latest.pt", weights_only=False)
    continuous = torch.load(tmp_path / "continuous/latest.pt", weights_only=False)
    assert resumed["step"] == 3
    for name, value in resumed["model"].items():
        assert torch.equal(value, continuous["model"][name])
    assert resumed["random_state"] == continuous["random_state"]
    assert torch.equal(resumed["torch_rng"], continuous["torch_rng"])


def test_overfit_measures_training_subset_without_reading_test_tensors(
    dataset, tmp_path, monkeypatch
):
    from wan_va.noise_inversion import training

    original_loader = training.load_example

    def training_only(record, context, device):
        assert context["split"] == "train"
        return original_loader(record, context, device)

    monkeypatch.setattr(training, "load_example", training_only)
    result = train(
        dataset, tmp_path / "overfit", settings(overfit_pairs=1, train_steps=5)
    )
    assert result["mode"] == "overfit"
    assert result["selection_split"] == "train"
    assert result["test"] is None
    assert result["best_validation_mse"] is None
    assert (
        result["selected_metrics"]["mse"]
        < result["selected_metrics"]["zero_residual_mse"]
    )


def test_resume_rejects_changed_batch_or_overfit_subset(dataset, tmp_path):
    output = tmp_path / "training"
    train(dataset, output, settings())
    with pytest.raises(ValueError, match="training settings changed"):
        train(dataset, output, settings(effective_batch=2))
    with pytest.raises(ValueError, match="larger than the training split"):
        train(dataset, tmp_path / "overfit", settings(overfit_pairs=100))
