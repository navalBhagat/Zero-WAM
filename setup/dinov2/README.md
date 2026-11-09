# DINOv2 setup

Use a Python environment with `requirements.inference.txt` installed. Run from
any working directory; no GPU, Conda, or cluster modules are required.

```bash
python /path/to/Zero-WAM/setup/dinov2/download.py
```

Downloads `facebook/dinov2-small` to the repository's
`checkpoints/dinov2-small`, verifies CPU loading without network access, and
records the resolved commit in `download_manifest.json`.

Optional arguments:

```bash
python setup/dinov2/download.py --output /path/to/checkpoint --revision COMMIT_HASH
```

`DINO_CHECKPOINT` and `DINO_REVISION` also set defaults; command-line arguments
take precedence. Setup requires network access. Inference uses the downloaded
checkpoint locally: set `DINO_CHECKPOINT` to its absolute directory using your
shell or job configuration.
