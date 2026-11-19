# Temporal and strength diagnostics

## CPU temporal replay

```bash
sbatch experiments/jobs/03_temporal_guidance_test.sbatch
```

This reconstructs the full human reference on CPU, embeds it and saved robot previews
with whole-image DINO, then compares +4 and distance-based targets on the **same
recorded trajectory**. It does not run a new policy or establish better task success.
Use `scripts/03_temporal_guidance_test.py --help` for paths and parameter overrides.
Results are `results.csv`, `targets.png`, `comparison.png` and `provenance.json`.

The matching frame is selected by unchanged forward-window DINO similarity.
Distance lookahead smooths reference embeddings across adjacent frames, L2-normalizes
them, then accumulates adjacent cosine changes. With automatic calibration, the
threshold is the median change across four-frame intervals. The target is the first
frame that reaches this increment, capped at 24 frames ahead. Flat calibration
falls back to bounded frame lookahead. This measures visual change, not verified
semantic progress; camera/hand motion can dominate it.

Inspect the actual images when curves differ. Increasing reference indices are not
proof that the robot is making progress. A physically unsuccessful grasp can still
look like a late human frame to whole-image DINO.

## Saved-step strength and precision replay

Submit the small FP32 strength sweep with:

```bash
sbatch experiments/jobs/strength_replay.sbatch

# Extend the sweep using arguments passed directly to the job script.
sbatch experiments/jobs/strength_replay.sbatch 0 0.1 0.3 1.0
```

Defaults are the saved scale prediction from job 29419404, strengths **0, 0.01,
0.03, 0.1**, and one measurement each. It loads VAE+DINO only on a GPU. Each job gets
a fresh `experiments/strength_replay/job_JOBID/` directory.

For direct execution **inside a GPU allocation**, in `zerowam`:

```bash
python experiments/scripts/precision_replay.py \
  --run "$CAPTURE_RUN" --model "$MODEL_PATH" --dino "$GUIDANCE_CHECKPOINT" \
  --decoder-dtypes float32 --strengths 0 0.01 0.03 0.1 --repeats 1 \
  --output experiments/strength_replay/manual_check
```

This Python command does not submit a job. `jobs/precision.sbatch` instead runs the
existing FP32/FP16 comparison with the replay CLI's default strengths/repetitions.

Captured RGB is preferred so the target is identical; pass `--reference`/`--frames`
only for deliberate overrides. Persistent latents and DINO stay FP32; only the
reward decoder dtype varies. The observed first frame is restored during reward
calculation. Zero strength bypasses backward, so its gradient is unmeasured.

`precision.json` checkpoints every measurement; `results.csv` is written after the
sweep. Inspect finite updates, correction magnitude, clipping, and same-target
reward gain. Failure rows are retained. Absolute rewards against changing targets
are not comparable evidence of improvement. Repeated identical captures test
numerics, not independent task trials.

## What to test next

First check that nonzero corrections survive numerical precision and improve the
same-target local surrogate. Then compare a small paired rollout with identical
scene/noise, conditioning, selector and frequency; change only strength.
`--guidance-every 5` changes frequency independently, so test it separately.

For a semantic alignment diagnostic, label a few saved frames as failed grasp,
grasp, lift and placement. Inspect unconstrained DINO matches against labelled
human-reference phases. Forward-only matching would force ordered indices and can
hide representation failures. Labels are for evaluation, not policy input.

## Whole-image matching and candidate progress

Submit from the repository root on Snellius:

```bash
sbatch experiments/jobs/progress_matching.sbatch
```

This submits a **CPU-only Rome job**, not a rollout. The default input is the
strength-1.0 scale run, job 29485194. It reconstructs the exact latent reference
with the VAE on CPU, then uses frozen whole-image DINO with runtime preprocessing.
Checkpoints and original reference latent must be available on Snellius.
No patch features, crops, or semantic models are introduced.

Each job writes `experiments/progress_matching/job_<JOB_ID>/`:

- `results.csv`: unconstrained nearest match, existing window/distance target,
  candidate position, 80% posterior interval, candidate distance-lookahead target,
  and optional stage agreement.
- `matching.png`: all robot/reference cosine similarities and causal posterior.
  The unconstrained heatmap is the test of visual evidence; a smooth filtered
  path alone does not demonstrate correct ordering.
- `comparison.png`: robot observation, unconstrained nearest human frame, and
  candidate target for up to six observations.
- `labels_template.csv`, `reference_frames/`: inputs for manual evaluation.
- `provenance.json`: reference source and filter assumptions.

First inspect all robot `robot_chunk_*.png` files in the original run and the
exported human frames. Copy the label template and label each available frame
and chunk using shared **visible state** names, for example `pre_grasp`, `holding`,
`lifted`, `placed`. A failed grasp remains `pre_grasp` if the gripper is empty;
it must not be labelled as holding just because the arm reached the object.
Use states that are visually distinguishable in both domains. Leave uncertain
rows blank; they are excluded from agreement calculations. Labels are evaluation
only and never supplied to the candidate tracker.

```bash
# Replace the path with your completed, manually edited CSV.
sbatch experiments/jobs/progress_matching.sbatch \
  --labels experiments/progress_matching/labels.csv
```

Stage agreement is printed with its evaluated sample count and recorded per
observation in the CSV. Annotate reference frames sufficiently densely; a predicted
frame with no label is excluded, so sparse labels can bias apparent accuracy.
If this failed trajectory never holds/lifts/places the object, it can test false
advancement but cannot establish recognition of successful stages. Include an
existing successful recorded trajectory for that test; no new rollout is required.
This CLI currently accepts scale runs with saved chunk observations.

The candidate uses a causal Bayesian filter over reference frame positions:
DINO cosine scores provide likelihoods; each hypothesis can stay or advance up to
four reference frames per robot chunk. The first observation is anchored to frame
0; verify the initial human state is compatible or provide `--start-frame`.
The reported position is the posterior median. The candidate target uses the
existing automatically calibrated visual-distance lookahead from that position.
No live selector is changed.

The transition bound (`--max-progress-advance`, default 4) and likelihood
`--temperature` (default 0.05) are experimental assumptions, not learned constants
or the existing +4 lookahead. Test bounds 2, 4 and 8 against the **same manual
labels**, not rollout success on this one example. A forward prior can still drift
through ambiguous images or hide a bad DINO representation. Broad intervals,
wrong-stage nearest matches, or late targets during failed grasps are reasons to
reject the candidate. Do not promote it to rollout guidance until labelled stage
agreement improves and failed-grasp observations remain in the correct stage.

## Successful same-seed ICL comparison

```bash
sbatch experiments/jobs/progress_matching_icl.sbatch
```

This is a CPU-only Rome job, using successful baseline job **29286681**, simulator
seed 100000, sampling seed 1234, 8 samples/pixel, and the same human reference 086.
It does not run the policy or simulator. It extracts the external camera from the
historical 960x624 composite video (rectangle x=0:320, y=40:280). Other layouts
are rejected. The extraction is video-compressed, unlike the failed-run PNGs.

Output: `experiments/progress_matching_icl/job_<JOB_ID>/` contains:

- `robot_frames/robot_chunk_*.png`: five chunk-start observations plus terminal
  success, ready to annotate.
- `robot_frames/source.json`: source video, crop, frame/action indices, terminal flag.
- `analysis/`: the same diagnostic CSV, two figures, reference frames and labels template.

Observation IDs 0..4 correspond to actions 0,16,48,80,112. ID 5 is terminal at
action 131, not a sixth inference chunk. The extraction validates this specific
baseline's 34-frame/131-action recording rather than guessing timing from FPS.

Copy `analysis/labels_template.csv` to a separate ICL labels file. Reuse the human
reference labels, but label these six robot images independently; **do not reuse
the failed trajectory's robot labels**. Then submit another fresh job:

```bash
sbatch experiments/jobs/progress_matching_icl.sbatch \
  --labels experiments/progress_matching_icl/labels.csv
```

Compare the labelled stage agreement and unconstrained DINO matches across success
and failure. The current prior still permits only four reference frames per
observation; over five transitions it cannot reach the placement stage (frame 41).
That is a known speed-prior limitation, not evidence DINO cannot recognize success.
Use the unconstrained scores to assess visual evidence first. Do not interpret
this short successful rollout's constrained-path accuracy as a fair test of the
prior without later calibrating the transition model on both trajectories.

## Initial-frame-relative matching: reuse both label sets

```bash
sbatch experiments/jobs/relative_matching.sbatch
```

This submits one CPU-only Rome job, without policy inference, VAE reconstruction,
or simulator execution. It reuses the exported human reference PNGs from
`stages/progress_matching/job_29488053`, the failed rollout's raw robot PNGs, and
successful ICL images from `stages/progress_matching_icl/job_29489163/robot_frames`.
Both existing `labels.csv` files in those stage directories are read unchanged.
Reference labels must agree. Override CLI paths after the script name if inputs
are moved. A local DINO checkpoint is required on Snellius.

For each trajectory separately, compare:

- Absolute cosine similarity of the normalized whole-image DINO embeddings.
- Relative cosine: subtract each domain's own initial embedding, normalize those
  change vectors, and compare directions. Reference frames with near-zero change
  have no direction and are excluded.
- Relative L2: compare unnormalized change vectors, preserving magnitude.
- Stay-at-start: always predict human frame 0, exposing class imbalance.

The matcher uses no labels or temporal prior. Initial observations are excluded
from every method's accuracy because they define the anchors. Relative methods
abstain for near-zero robot change (norm <= 1e-6); on labelled noninitial
observations, abstentions count as incorrect. Thus denominators are 12 for failure
and 5 for success when all current observations are labelled. Missing predicted
reference-stage labels also count as incorrect; blank robot labels are excluded.

Outputs in `experiments/stages/relative_matching/job_<JOB_ID>/`:
`results.csv`, `matching.png`, `stage_agreement.png`, cached `embeddings.npz`, and
`provenance.json` with input hashes. Recomputing absolute scores from exported
reference PNGs introduces quantization relative to the original floating-point
VAE outputs; all three visual methods use the same PNG reference in this test.
Success robot images additionally have video compression.

Look for improvement on **both** success and failure, rather than pooled accuracy.
A method that stays early may score well on failure but miss successful lifting.
A method that always selects late frames may match terminal success by chance.
Relative changes may still be dominated by arm motion or different camera geometry;
this experiment does not presume they encode semantic progress.
