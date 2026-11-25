# Zero-WAM experiments

Slurm entry points are grouped by cluster: `jobs/snellius/` and `jobs/das6/`.
The shared evaluation lifecycle is `scripts/run_evaluation.sh`; cluster-specific
module/Conda activation lives in `scripts/clusters/`. See the DAS6 quick start in
[noise inversion](docs/noise_inversion.md#das6-seven-task-pilot).
Existing guidance code and its relocated Snellius launchers remain unchanged in
behavior; its proposed cleanup is documented, not implemented.

Tools for planning RoboTwin runs, replaying saved predictions, and explaining results.
Run commands from the repository root. Historical results in `stages/` are preserved.

## Pick a workflow

| Goal | Entry point | Resources | Submits a job? |
| --- | --- | --- | --- |
| Validate cache/memory behavior | `scripts/run_stage00.py` | A100 | Only with `--submit` |
| Plan or launch a campaign | `scripts/robotwin_suite.py` | A100 | Only with `--submit` |
| Diagnose whole-image DINO correspondence | `jobs/snellius/progress_matching.sbatch` | CPU (Rome) | Yes, through `sbatch` |
| Compare temporal targets offline | `scripts/03_temporal_guidance_test.py` | CPU | No |
| Replay strengths/decoder precision | `scripts/precision_replay.py` | GPU, VAE+DINO only | No |
| Submit the small strength replay | `jobs/snellius/strength_replay.sbatch` | A100 | Yes, through `sbatch` |
| Summarize a campaign | `analysis/analyze.py` | CPU | No |

Paths in this table are relative to `experiments/`. Each entry point supports
`--help`, except Slurm job files. A dry run writes a manifest and source snapshot;
use a different manifest path for submission. Existing manifests are never overwritten.

## Quick start

First follow [setup and running](docs/running.md) to configure the existing Snellius
environments and local checkpoints. Stage 00 validates memory before larger runs.

```bash
# Preview one guidance episode; no GPU is allocated by this command.
python experiments/scripts/robotwin_suite.py guidance \
  --tasks place_object_scale --seeds 0 --episodes 1 \
  --guidance-modes guidance --guidance-selectors window \
  --spp 8 --phase-separation --compact-cache \
  --time 01:00:00 --client-timeout 3000 --minutes-per-episode 20 \
  --manifest experiments/stages/guidance_preview/manifest.json

# Submit the saved-prediction strength test from the repository root.
sbatch experiments/jobs/snellius/strength_replay.sbatch

# Analyze an existing campaign without rerunning it.
python experiments/analysis/analyze.py experiments/stages/03_guidance_pilot/manifest.json
```

Use your current `--budget-sbu` when submitting a campaign. The built-in balance is
an old planning default, not a live account balance.

## Read the results

```text
campaign/
  manifest.json + manifest.source.zip
  runs/TASK_JOBID/
    config.json       # Effective settings, seeds, outcomes and artifact paths
    results.csv       # One readable row per episode
    videos/           # Short episode filenames
    references/       # Selected targets, observations and alignment traces
    diagnostics/      # Logs, timings, FLOP coverage and optional deep diagnostics
  analysis/
    results.csv       # One row per task/condition
    success.png
    cost.png
```

The default analysis has one CSV and two plots. Extra tables/plots require
`--diagnostics`. See [outputs and compute](docs/outputs.md) for units, missing values,
FLOP assumptions, and compatibility with older runs.

## Guides

- [Setup, stages, submission and resume](docs/running.md)
- [Output files, timing, success and FLOPs](docs/outputs.md)
- [Temporal matching and strength/precision diagnostics](docs/diagnostics.md)
- [Code organization and regression checks](docs/development.md)

## Noise inversion campaign

See [noise inversion](docs/noise_inversion.md) for the seven-task paired campaign:
ICL, no ICL/language with original noise, and no ICL/language with recovered noise.
The planner creates 14 task/noise jobs (five scenes and one model server per job),
a flat trial layout, and a reproducible manifest without submitting:

```bash
python experiments/scripts/noise_inversion_campaign.py plan \
  --directory experiments/stages/noise_campaign_01
```

Submit bounded batches (Slurm uses your login default account); generate the supervisor
success table, bar charts and descriptive analysis with the `report` subcommand
(`--no-plots` for tables only). Historical pilot commands remain supported.
