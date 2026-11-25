# NoiseICL campaigns

Initialize DAS6 with `source "$HOME/nbhagat-workspace-env.sh"`, load `prun`, activate
`robotwin`, and run commands from `$ZERO_WAM_ROOT`. Snellius uses its corresponding
cluster adapter. Install prerequisites are in [INSTALL.md](../INSTALL.md).

| Stage | Default workload | Required input |
| --- | --- | --- |
| 1 smoke | Seven unseen tasks, one scene each, ICL/control/online oracle | Model, RoboTwin assets, seven human references |
| 2 signal | Twenty distinct scene slots per unseen task | Same as stage 1 |
| 3 benchmark | First 100 noise pairs | Grouped seen-task context index |
| 4 collect | All pairs; 2000 for the five-task pilot | Same grouped context index |
| 5 train_evaluate | Residual adapter, seen/unseen rollouts, shuffled-demo ablation | Completed collection campaign |

```bash
python experiments/scripts/noiseicl.py plan --stage 1 --cluster das6 \
  --directory experiments/stages/01_smoke
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/01_smoke --limit 1
# After checking that trial, submit the remaining six with --limit 6.
python experiments/scripts/noiseicl.py report \
  --directory experiments/stages/01_smoke --plots
```

Plan the later campaigns explicitly:

```bash
python experiments/scripts/noiseicl.py plan --stage 2 --cluster das6 \
  --directory experiments/stages/02_signal --scenes 20
python experiments/scripts/noiseicl.py plan --stage 3 --cluster das6 \
  --directory experiments/stages/03_benchmark --contexts experiments/datasets/seen/index.json
python experiments/scripts/noiseicl.py plan --stage 4 --cluster das6 \
  --directory experiments/stages/04_collect --contexts experiments/datasets/seen/index.json
python experiments/scripts/noiseicl.py plan --stage 5 --cluster das6 \
  --directory experiments/stages/05_train_evaluate --dataset experiments/stages/04_collect
```

`submit --limit N` submits at most N eligible jobs. Evaluation/collection jobs are
independent; stage-5 evaluation depends on successful training. Completed jobs
are skipped. `submit --retry` retries inactive incomplete jobs, archiving partial
outputs. Resolve interrupted submissions and cancel failed-training dependent
jobs in Slurm before retrying those jobs. Code drift requires a new plan.

Resources: one GPU/job, DAS6 A6000 with four CPUs or Snellius A100 with 18 CPUs.
The default ceiling is three hours/job, not an expected duration. Online context
rebuilding is more expensive than saved-noise replay. Run long DAS6 jobs only in
a permitted window/site arrangement. Review stage-1 timing before increasing
submission concurrency. A 20-scene signal campaign plans 140 jobs / 420 rollouts.

## Recorded data

The collection stages do not generate fake histories or train on unseen tasks.
Download/prepare seen-task robot trajectories with normalized actions and frozen
Wan VAE observations. `noiseicl.py schema` prints the required JSONL record:
task, trajectory_id, scene_id, simulator_seed, demo_id, demo_latent, initial,
history, chunk_count. Initial files contain `initial_frame`; each contiguous
`history/chunk_NNNN.pt` contains `video` and normalized `actions` tensors.

```bash
python experiments/scripts/noiseicl.py index --trajectories seen_trajectories.jsonl \
  --output experiments/datasets/seen --tasks 5 --trajectories-per-task 20 --points 10
```

This chooses up to ten history points per trajectory and assigns scene/demo groups
to train/validation/test before deriving two reproducible, distinct Gaussian seeds per point. Seeds also
vary across contexts; collection does not reuse only two Gaussian tensors. A full five-task,
20-trajectory pilot has 1000 contexts / 2000 pairs. Shared scenes, simulator seeds,
demo IDs or demo files cannot cross splits. Actual grouped split sizes are recorded;
they may differ from nominal 70/15/15. The seven unseen tasks are rejected.

Stage 5 defaults to eight transformer blocks, width 512 and eight heads. Start
with `--overfit-pairs 100 --train-steps 500`; `--architecture mlp` selects a simple
patch-wise baseline. Effective batch size is controlled by gradient accumulation
through `--batch`. Validation selects the best checkpoint; test MSE is measured
only after selection. Dataset paths must stay available throughout the campaign.
