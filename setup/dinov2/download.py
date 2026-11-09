"""Download and verify DINOv2 on CPU using the current Python environment.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(os.environ.get("DINO_CHECKPOINT", root / "checkpoints/dinov2-small")),
        help="Local checkpoint directory (default: DINO_CHECKPOINT or repository checkpoints/dinov2-small)",
    )
    parser.add_argument("--revision", default=os.environ.get("DINO_REVISION", "main"),
                        help="Hugging Face revision; use a commit hash for reproducibility")
    args = parser.parse_args()

    # Setup explicitly needs network access; offline inference is unchanged.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel

    repo = "facebook/dinov2-small"
    destination = args.output.expanduser().resolve()
    revision = HfApi().model_info(repo, revision=args.revision).sha
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=str(destination),
        allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
    )
    # Verify the offline loading route used by guidance. No GPU allocation.
    model = AutoModel.from_pretrained(str(destination), local_files_only=True)
    if model.config.model_type != "dinov2":
        raise RuntimeError(f"Expected DINOv2, got {model.config.model_type}")
    (destination / "download_manifest.json").write_text(
        json.dumps({"repo_id": repo, "revision": revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Verified local DINOv2 checkpoint: {destination}")
    print(f"Revision: {revision}")
    print("Set DINO_CHECKPOINT to the checkpoint directory when running guidance.")


if __name__ == "__main__":
    main()
