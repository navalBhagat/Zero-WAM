# Zero-WAM: External Inference-Time Guidance

Research code for studying external guidance of a frozen Zero-WAM world-action
model using human demonstrations and RoboTwin evaluation.

Start with [INSTALL.md](INSTALL.md) for environment setup, downloads, verification,
and a one-episode evaluation.

The [original project README](old_readme.md) contains the upstream model overview,
training instructions, results, acknowledgments, and citation. External guidance
is under development; native human-video ICL remains the default.

## Conditioning and guidance modes

The three switches are independent. Choose a row and pass its settings to the
Snellius evaluation job; `0.01` is an initial experimental DINO strength.

| Mode | `USE_ICL` | `TARGET_TEXT_CFG` | `EXTERNAL_STRENGTH` |
|---|---:|---:|---:|
| Observation/history only | 0 | -1 | 0 |
| Text only | 0 | 1 | 0 |
| Human-video ICL (default) | 1 | -1 | 0 |
| Text + ICL | 1 | 1 | 0 |
| DINO guidance only | 0 | -1 | 0.01 |
| Text + DINO guidance | 0 | 1 | 0.01 |
| ICL + DINO guidance | 1 | -1 | 0.01 |
| Text + ICL + DINO guidance | 1 | 1 | 0.01 |

For example, run text plus DINO guidance without native ICL:

```bash
USE_ICL=0 TARGET_TEXT_CFG=1 EXTERNAL_STRENGTH=0.01 \
  sbatch --export=ALL implementation/eval_one_task.sbatch
```

Use the same settings with `bash evaluation/robotwin/launch_client.sh` when
launching manually; configure the server with the same `EXTERNAL_STRENGTH`.
All modes still condition on robot observations/history. `USE_ICL=0` removes the
human transformer context; `ICL_CFG=0` alone does not. `TARGET_TEXT_CFG=1` uses
text without text CFG; values above 1 enable text CFG only when ICL CFG is
inactive. Native ICL CFG defaults to `ICL_CFG=5`.

DINO rows require a local checkpoint (`python setup/dinov2/download.py`, then set
`DINO_CHECKPOINT`) and a human MP4 reward reference even when native ICL is off.
Select `REFERENCE_START`/`REFERENCE_STOP` (exclusive stop, at 12 FPS); their
defaults select the first human frame, not a task-completion target. Guidance runs
every `EXTERNAL_EVERY=10` denoising steps with `EXTERNAL_MAX_NORM=1` by default.
These modes are selectable; GPU feasibility and task performance still need
validation. See [the implementation guide](implementation/docs/external-guidance.md)
for staged Snellius checks. `implementation/` is currently Git-ignored, so its
job scripts must also be copied to the cluster checkout.
