# Setup and running experiments

## Environments and paths

Use the existing `zerowam` inference environment and `robotwin` simulator environment
on Snellius. The Slurm jobs activate both where needed; CPU analysis does not need
the simulator. Jobs use modules `2024`, `Anaconda3/2024.06-1`, and `CUDA/12.6.0`.

From the checkout root:

```bash
export ZERO_WAM_ROOT="$PWD"
export ROBOTWIN_ROOT="$HOME/Robotwin"  # Adjust to your installation.
export MODEL_PATH="$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin"
export ICL_LATENT_ROOT="$ZERO_WAM_ROOT/data/HumanGen/human_latents/robotwin"
export GUIDANCE_CHECKPOINT="$ZERO_WAM_ROOT/checkpoints/dinov2-small"
```

Checkpoints/data must already exist locally. Nothing downloads them automatically.
GPU replay must run inside an allocation, not on a login node. The CPU temporal
replay reconstructs a reference with the VAE; use the Rome launcher for this work.

## Campaign stages

| Stage | Purpose | Defaults |
| --- | --- | --- |
| `run_stage00.py` | Validate padding compaction and memory | Cup control/compaction, then microwave; `afterok` dependencies |
| `render` | Compare rendering cost | Scale, seed group 0, 32/16/8 SPP, one episode each |
| `baselines` | Text-only versus native ICL | Both tasks, seed groups 0/1/2, ten episodes per condition/group |
| `guidance` | Guidance-only and ICL+guidance | Both tasks/groups; fixed and window selectors |
| `precision` | Full-rollout decoder-precision comparison | Both guidance conditions, FP32/FP16, window selector |

Stage 00 is deliberately separate: no launcher automatically advances to another
stage. It tests observation/cache memory, not reward-backward memory. A short run
does not establish long-history robustness. Inspect its report before scaling up.

The default fixed reference is the last frame. Window matching uses a 24-frame
forward search and four-frame lookahead. Use `--lookahead-mode dino_distance` for
adaptive visual-distance lookahead. Detailed semantics are in [diagnostics](diagnostics.md).

## Preview versus submission

```bash
python experiments/scripts/robotwin_suite.py guidance \
  --tasks place_object_scale --seeds 0 --episodes 1 \
  --guidance-modes guidance --guidance-selectors window \
  --spp 8 --phase-separation --compact-cache \
  --time 01:00:00 --client-timeout 3000 --minutes-per-episode 20 \
  --manifest experiments/stages/my_guidance_run/manifest.json
```

Add `--submit` to submit; use a fresh manifest path if you already previewed.
`--parallel` submits independent jobs; otherwise jobs use `afterany` dependencies.
Each full rollout job requests one A100 and 18 CPU cores. `--account` is optional.
CLI settings override job defaults. Use `--help` for all flags grouped by purpose.

Use `--diagnostics` for package inventories, action hashes and cache/offload traces.
Use `--capture-guidance` to save a prediction for later strength/precision replay.
Runtime FLOP counting is off by default. Use `--flop-count` only for short
calibration runs; Python operator interception can substantially slow inference.
Lightweight component shapes/call counts remain logged for offline estimates.
`--no-flop-count` remains supported for existing commands.

## Seeds and resume

- Sampling seed defaults to **1234** and controls model noise.
- Seed group **0** starts simulator seeds at **100000**; group **1** at **200000**.
- Expert-feasibility checks can skip scenes; pair on the recorded actual seed.
- Human demonstration seed follows the seed group in the campaign environment.

For baseline extensions, pass `--resume-from PREVIOUS_MANIFEST --episodes TOTAL`.
The requested count is the desired total, not an additional count. Resume requires
completed contiguous episode IDs, finished jobs, matching settings, and compatible
runtime source. A changed policy requires a fresh campaign.

## Budget and provenance

Planning includes per-job startup and per-episode minutes. The wall-limit ceiling
is separate from estimated usage. Actual charges follow elapsed allocation,
including setup, failures and cleanup; queue wait is excluded.

Pass the current `--budget-sbu`; the default is not updated from the cluster.
The legacy `analysis/budget.py` command delegates to the shared campaign calculator.
Use analysis `--sacct` on Snellius for scheduler accounting.

Each campaign snapshots source and verifies its digest when a job starts. Snapshots
include nested experiment modules, runtime code and documentation, but not weights,
data or the external RoboTwin checkout. Do not change a checkout while its jobs
are queued/running; use another checkout if work must overlap.
