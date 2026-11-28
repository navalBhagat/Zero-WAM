# NoiseICL campaigns

Initialize DAS6 with `source "$HOME/nbhagat-workspace-env.sh"`, load `prun`, activate
`robotwin`, and run commands from `$ZERO_WAM_ROOT`. Snellius uses its corresponding
cluster adapter. Install prerequisites are in [INSTALL.md](../INSTALL.md).

| Stage | Default workload | Required input |
| --- | --- | --- |
| 1 smoke | Seven unseen tasks, one scene each, ICL/control/online oracle | Model, RoboTwin assets, seven human references |
| 2 signal | Twenty distinct scene slots per unseen task | Same as stage 1 |
| 3 benchmark | First 100 noise pairs | Grouped seen-task context index |
| 4 collect | All pairs; up to 2000 for the budgeted five-task pilot | Same grouped context index |
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

For the DAS6 pilot, convert the downloaded source data and plan collection:

```bash
source "$HOME/nbhagat-workspace-env.sh"
conda activate zerowam
cd "$ZERO_WAM_ROOT"
python -m pip install pyarrow  # also listed in requirements.inference.txt
python -u experiments/scripts/prepare_noiseicl_pilot.py \
  --humangen-root "$ZERO_WAM_ROOT/data/HumanGen" --cluster das6
cat experiments/stages/04_collect_pilot/pilot.json
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/04_collect_pilot --limit 1
# After checking the first job, submit eligible remaining jobs:
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/04_collect_pilot --limit 80 --max-concurrent 2
```

Collection runs the frozen model against recorded histories in `zerowam`; no
live RoboTwin simulator is started or patched. Use a permitted DAS6 GPU window.
Each job contains at most 25 pairs. The actual pair/job counts are in `pilot.json`.
No jobs are submitted by the preparer. Do not change source code after planning;
submission verifies the frozen source digest.

Default seen tasks are `adjust_bottle`, `beat_block_hammer`, `blocks_ranking_rgb`,
`blocks_ranking_size`, and `click_alarmclock`; override with `--tasks` followed by
five seen task names. The preparer requires 20 distinct paired human demos per
task and uses all downloaded paired robot intervals. It selects up to 200
contexts per task, balancing demos and recordings and covering early/late points,
then derives two distinct Gaussian seeds per context. Short recordings are not
padded, repeated, or rejected for having fewer than ten points. The dataset can
therefore contain fewer than 2,000 pairs; task shortfalls are explicitly recorded.

The converter checks three-camera layouts/frame IDs and recorded actions. It
reuses the training action preprocessing and checkpoint normalization statistics.
Reusable conversion code lives in `wan_va/noise_inversion/humangen.py`, alongside
the collection runtime in `wan_va/noise_inversion/collection.py`. Reusable context
indexing, sampling, and split grouping live in `wan_va/noise_inversion/dataset.py`;
`experiments/` contains command entry points, campaign settings, and Slurm orchestration.
Residual adapter training lives in `wan_va/noise_inversion/training.py`; experiment
runners call that implementation rather than defining model training themselves.
Initial observations and complete two-frame histories are written once and shared
by both seeds. The first history includes the initial observation and its zero
action slot. At least one recorded latent remains beyond each history boundary.
An unpaired interval, missing camera, or incompatible sampling layout fails
preparation. This validates storage/alignment; the first GPU job must still verify
model execution and reconstruction diagnostics.

HumanGen episode IDs group intervals from the same recording. Shared demo files
also stay in one split. Simulator seeds and cross-episode scene identities are
not supplied by these inputs: the converter records `simulator_seed: null` rather
than inventing a seed. Stage-5 seen-scene evaluation needs verified simulator
metadata and refuses to interpret episode indices as simulator seeds.

Use fresh output directories for reruns, or reuse a completed conversion with
`--trajectories experiments/datasets/seen_pilot_histories/trajectories.jsonl` and
fresh `--output` / `--directory` values. Do not delete a dataset referenced by jobs.

The ICL mapping in `evaluation/robotwin/robotwin_icl_human_videos.py` covers all
50 tasks, with separate `ROBOTWIN_ICL_SEEN_HUMAN_VIDEOS` (43 tasks) and
`ROBOTWIN_ICL_UNSEEN_HUMAN_VIDEOS` (seven tasks) dictionaries and a combined
`ROBOTWIN_ICL_HUMAN_VIDEOS` dictionary used by evaluation. All five default pilot
tasks have 20 distinct references. Six seen tasks have only 19 distinct references
in the published pairing index: `grab_roller`, `lift_pot`, `place_a2b_left`,
`put_bottles_dustbin`, `rotate_qrcode`, and `stack_bowls_two`. The mapping retains
all available references for those tasks rather than duplicating one.
One reference is selected per episode. The `eval` download profile remains scoped
to the seven held-out tasks; use `seen` for latent/action-only seen-task assets.
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
20-trajectory pilot would have 1000 contexts / 2000 pairs only when every trajectory
has ten valid points. The budgeted pilot above handles shorter trajectories. Shared scenes, simulator seeds,
demo IDs or demo files cannot cross splits. Actual grouped split sizes are recorded;
they may differ from nominal 70/15/15. The seven unseen tasks are rejected.

Stage 5 defaults to eight transformer blocks, width 512 and eight heads. Start
with `--overfit-pairs 100 --train-steps 500`; `--architecture mlp` selects a simple
patch-wise baseline. Effective batch size is controlled by gradient accumulation
through `--batch`. Dataset paths must stay available throughout the campaign.

## Check collection quality and train on DAS6

After syncing the updated code, use the completed collection directly. Quality
checks are read-only with respect to collected data and run on CPU:

```bash
source "$HOME/nbhagat-workspace-env.sh"
conda activate zerowam
cd "$ZERO_WAM_ROOT"
python -u experiments/scripts/noiseicl.py quality \
  --directory experiments/stages/04_collect_pilot
```

The command checks every planned pair, context hashes/source digests, duplicate
targets/seeds, finite tensors, shapes/masks, and recording/demo split separation.
Fallback targets are excluded from usable counts. It reports mean, median, 95th
percentile, and maximum reconstruction errors overall and per task, split counts,
and the ten highest-error examples. Errors exit with status 1. Missing simulator
seeds are explained as a limitation of scene verification, not fabricated.
There is no assumed reconstruction-error cutoff or claim of robot success.

Reports are saved in `experiments/stages/04_collect_pilot/quality/`:
`summary.json`, `tasks.csv`, and `worst_examples.csv`. Use `--worst N` for more
examples, `--output PATH` for a separate report directory, or `--metadata-only`
for a quick record/group check that skips tensor contents.

Start with a 100-pair overfit diagnostic. This uses the existing stage-5 planner
but creates only one training job, with no simulator evaluation or seed requirement:

```bash
python experiments/scripts/noiseicl.py plan --stage 5 --training-only \
  --cluster das6 --directory experiments/stages/05_overfit \
  --dataset experiments/stages/04_collect_pilot \
  --overfit-pairs 100 --train-steps 500 --batch 8
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/05_overfit --limit 1
```

The subset is selected reproducibly from training data only. Overfit mode selects
the checkpoint by MSE on that subset, reports the unchanged-noise baseline, and
does not read validation/test tensors. Check
`experiments/stages/05_overfit/jobs/train/summary.json` and `training.jsonl` there:
`selected_metrics.mse` should decrease below `selected_metrics.zero_residual_mse`.
This is a diagnostic, not evidence of generalization. Progress is logged every
100 steps and at the final step; the GPU job's `diagnostics/client.log` captures it.

Once that diagnostic works, create a fresh full-training campaign:

```bash
python experiments/scripts/noiseicl.py plan --stage 5 --training-only \
  --cluster das6 --directory experiments/stages/05_train \
  --dataset experiments/stages/04_collect_pilot --train-steps 5000 --batch 64
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/05_train --limit 1
```

Full training selects checkpoints using validation MSE and evaluates test MSE
only after selection. `adapter.pt` is the best model; `latest.pt` includes optimizer
and RNG state for resumption. `submit --retry` resumes an inactive unfinished
training job. Batch/seed/subset or architecture changes require a fresh run;
checkpoints from the earlier trainer format require a fresh output directory.
Training-only planning checks collection completion and metadata integrity. Run
the full quality command first to validate tensor contents as well.

The default job allocation ceiling is three hours, not an estimated training
duration. Use the overfit timing to choose `--hours` for the subsequent plan and
run within a permitted DAS6 GPU window. Do not change source after planning.

Core implementations live in `wan_va/noise_inversion/quality.py`, `dataset.py`,
and `training.py`. Experiment modules handle report files, planning, and submission.
