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

For the research-plan pilot, prepare exactly 2,000 pairs from recorded histories:

```bash
python experiments/scripts/prepare_noiseicl_pilot.py \
  --trajectories /data/seen_trajectories.jsonl --cluster snellius
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/04_collect_pilot --limit 1
# After checking the first job, submit the remaining 79 jobs:
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/04_collect_pilot --limit 79 --max-concurrent 2
```

Use `--cluster das6` on DAS6. Default seen tasks are `adjust_bottle`,
`beat_block_hammer`, `blocks_ranking_rgb`, `blocks_ranking_size`, and
`click_alarmclock`; override with `--tasks` followed by five seen task names.
The preparer requires 20 distinct paired human demonstrations and robot
trajectories per task, each with at least ten chunks. It freezes a selected
trajectory manifest, creates grouped splits, and plans 80 jobs of 25 pairs.
It does not submit jobs. Run preparation on the target cluster so absolute
asset paths are accessible there. Missing recorded assets must be prepared
first; the published HumanGen pairing JSON alone is not a history manifest.

The ICL mapping in `evaluation/robotwin/robotwin_icl_human_videos.py` covers all
50 tasks, with separate `ROBOTWIN_ICL_SEEN_HUMAN_VIDEOS` (43 tasks) and
`ROBOTWIN_ICL_UNSEEN_HUMAN_VIDEOS` (seven tasks) dictionaries and a combined
`ROBOTWIN_ICL_HUMAN_VIDEOS` dictionary used by evaluation. All five default pilot
tasks have 20 distinct references. Six seen tasks have only 19 distinct references
in the published pairing index: `grab_roller`, `lift_pot`, `place_a2b_left`,
`put_bottles_dustbin`, `rotate_qrcode`, and `stack_bowls_two`. The mapping retains
all available references for those tasks rather than duplicating one.
One reference is selected per episode. The `eval` download profile remains scoped
to the seven held-out tasks; use `robotwin` for seen-task assets.
Refresh the `eval` download profile on clusters
that previously downloaded only one reference per task.

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
