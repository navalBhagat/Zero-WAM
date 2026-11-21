# Noise inversion pilot

One source episode per task (`place_bread_basket`, `place_object_scale`), followed
by offline inversion and a closed-loop replay without language or ICL if the source succeeds.
No training, DINO reward, or optimization through WAM is involved.

## Run

On Snellius, with the usual local checkpoints and RoboTwin environments installed:

```bash
sbatch experiments/jobs/noise_inversion.sbatch
```

This launches both tasks sequentially on one GPU, reusing the existing evaluation
job's startup, health check, logging, and cleanup. Defaults: model seed 1234,
evaluation seed 0 (scene seed 100000 before feasibility filtering), FP32 video
state, 16 render samples/pixel, compact observation cache enabled. Phase offload
is disabled by default; set `PHASE_SEPARATION=1` if needed for memory. Source,
inversion, and replay all use the same server/checkpoint/cache configuration.
A 3-hour allocation is requested; no jobs are submitted by importing Python code.

Alternatively start the usual server with `VIDEO_STATE_FP32=1`, then use the
RoboTwin client environment (with `ROBOTWIN_ROOT`, `ICL_LATENT_ROOT`, and
`PYTHONPATH` set as for normal evaluation):

```bash
python experiments/scripts/noise_inversion_pilot.py \
  --output experiments/runs/noise_pilot_01 \
  --tasks place_bread_basket place_object_scale \
  --sampling-seed 1234 --simulator-seed 0 --port 29056
```

The output directory must be new. Server and client must share its absolute
filesystem path. The simulator and model need CUDA; CPU tests do not validate
actual task success or GPU memory requirements.

## Phases and conditioning

1. Capture: ICL scale 5, empty target text, cached human caption; reward off.
   Save generated video latents and initial video/action noise for every chunk.
   Save encoded camera observations and normalized actions for history updates.
2. Inversion: reset without ICL or language, using empty text embeddings.
   Rebuild robot-only KV history from recorded observation latents/actions. Reverse
   Euler integration uses the forward sampler's shifted sigma grid plus sigma 0.
   Each chunk is forward-reconstructed under exactly the same robot-only history.
   No prediction cache is written during either numerical integration.
3. Replay: reset the simulator to the source's actual accepted scene seed and
   instruction for simulator metadata only; language input to the model is disabled.
   Use inverted video noise and original action noise per chunk;
   update history from this rollout's own observations and predicted actions.

The first chunk's observed first frame is clamped at every step and excluded from
relative reconstruction error. Reverse Euler is approximate. Inverted noise is
not guaranteed to follow a standard Gaussian; do not normalize it before replay.

An unsuccessful source is reported and inversion/replay are skipped. Replay stops
with `noise_sequence_exhausted=true` if it needs more chunks than were captured;
it never silently switches to fresh Gaussian noise. The terminal source chunk
may contain a full model prediction but only partially executed actions. Its
partial observations are not cached or needed for offline history reconstruction.

## Output

Under `OUTPUT/TASK/`:

- `capture/`, `replay/`: ordinary episode config, outcome, profiles and video.
- `chunks/chunk_XXXX.pt`: generated video, initial noise, initial observed frame,
  and normalized predicted actions, stored on CPU.
- `history/chunk_XXXX.pt`: encoded observed video and normalized action history.
- `inverted/chunk_XXXX.pt`: video noise to inject and original action noise.
- `inversion.json`: per-chunk relative reconstruction error and elapsed seconds.
- `summary.json`: source/replay success, chunk counts, seeds, exhaustion status.

Good reconstruction is a numerical check, not a guarantee of rollout success:
Actions without language/ICL can differ even given similar videos, and replay history can
then diverge. This pilot has no same-noise baseline without language/ICL; it tests feasibility,
not a measured improvement over ordinary sampling without language/ICL. A later experiment can add that control.

## Append original-noise controls to an existing pilot

Sync the updated code to Snellius, then submit from the repository root:

```bash
export NOISE_INVERSION_CONTROL_SOURCE="$PWD/experiments/stages/noise_inversion_pilot/runs/place_bread_basket_29492432/noise_inversion"
sbatch experiments/jobs/noise_inversion_control.sbatch
```

If the original pilot is still under `experiments/runs/`, point the variable there
instead. It must name the directory containing both task folders. This runs only
one control per task, reusing the saved initial video/action noise and actual
scene seed, with language, ICL and reward guidance disabled. The source capture
and inversion are not repeated. Configurations, videos and profiles are written
to `TASK/control/` beside `capture/` and `replay/`; `TASK/summary.json` is updated.
Existing control directories are not overwritten. Server/job diagnostics for the
new allocation are still stored under its own normal run directory.

The control may continue beyond the captured noise horizon to the normal task
limit, using the seeded Gaussian sequence after the saved chunks. The summary
reports eventual `control_success`, `control_success_within_source_horizon`,
and `control_extended_beyond_saved_noise` separately. Inverted-noise replay still
stops when its saved sequence is exhausted.
