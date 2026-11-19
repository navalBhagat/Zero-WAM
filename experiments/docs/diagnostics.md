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
