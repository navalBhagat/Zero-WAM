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
