# Zero-WAM experiments

Only two launch entry points are needed: `scripts/run_stage00.py` to validate memory,
and `scripts/robotwin_suite.py` for stages 1–4. Both preview by default; add `--submit`
to call Slurm. Existing manifests are never overwritten. Use a fresh directory for
retries. No launcher automatically advances to stages 1–4.

```text
experiments/
  README.md
  scripts/
    run_stage00.py       # cache-compaction checks
    robotwin_suite.py    # stages 1–4
    precision_replay.py  # captured-step precision diagnostic
    job_timing.py       # sbatch helper
    provenance.py       # source snapshots and drift checks
  jobs/
    eval.sbatch
    precision.sbatch
  analysis/
    analyze.py
    analyze_stage00.py
    plot_matching.py
    plot_precision.py
    budget.py
    requirements.txt
  tests/
  stages/               # generated outputs, Git-ignored
```

Old append/seed/caption scripts and one-off scale/phase-separation launchers have
been removed. Historical results, manifests and source ZIPs are untouched. The
shared analysis tool retains support for old manifests; source snapshots preserve
the old scripts. This cleanup does not move or delete results.

## Snellius setup

Run from the updated checkout on Snellius with the existing working Conda
installations (`zerowam`, `robotwin`) and model/data paths:

```bash
export ZERO_WAM_ROOT="$PWD"
export ROBOTWIN_ROOT="$HOME/Robotwin"  # use your actual existing path
export MODEL_PATH="$ZERO_WAM_ROOT/checkpoints/zero-wam-posttrain-robotwin"
export ICL_LATENT_ROOT="$ZERO_WAM_ROOT/data/HumanGen/human_latents/robotwin"
export GUIDANCE_CHECKPOINT="$ZERO_WAM_ROOT/checkpoints/dinov2-small"
```

The jobs use the existing modules 2024, Anaconda3/2024.06-1 and CUDA/12.6.0. Each
requests one A100 and 18 CPU cores. CLI runtime overrides take precedence over
sbatch defaults. Add `--account ACCOUNT` to a launcher if required.

Do not change source while jobs are queued: jobs verify the recorded source digest.
Source snapshots include runtime modules, scripts and configs, not weights/data or
the external Robotwin checkout. Preserve those external inputs. Each run records
settings, revisions, package versions, GPU measurements and logs. Distinguish model
sampling seed (`1234` by default), run seed (`SEED=0` starts simulator seeds at
100000), and human demonstration seed (`ICL_SEED`). Actual accepted simulator
seeds can differ after expert-feasibility filtering.

## 00 — padding compaction and bounded memory

```bash
# Preview under a separate directory if desired:
python3 experiments/scripts/run_stage00.py \
  --manifest experiments/stages/00_cache_compaction_preview/manifest.json

# Submit three one-episode jobs:
python3 experiments/scripts/run_stage00.py --submit
```

Default output: `experiments/stages/00_cache_compaction/`.

1. Cup control: current observation offload, compaction disabled.
2. Cup compaction: same settings, compaction enabled.
3. Microwave long-history stress test: compaction and observation offload enabled.
   Microwave is a capacity diagnostic here, **not** a stage 1–4 evaluation task.

All use sampling seed 1234, run/demo seeds 0, FP32 video state, 32 SPP, ICL CFG 5,
empty robot text plus cached human caption, external guidance off. Both cup arms
use the same single-GPU unsharded path required by offloading. Cup jobs must succeed;
`afterok` dependencies stop the chain if a prerequisite fails. A completed microwave
task failure is acceptable for the capacity test. Hashes remain diagnostic, not an
exact-equality gate: cache layout can affect floating-point reductions.

Use `--cup-only` for only the two cup jobs. The full test has an approximate
119.5-SBU planning estimate; three one-hour limits have a 384-SBU ceiling. Actual
transfer/runtime costs vary. Blocked jobs consume no GPU runtime; cancel them by
recorded job ID if a prerequisite fails. Do not submit duplicate batches.

### Completed pre-phase-1 diagnostics

All three diagnostic jobs completed successfully with model seed 1234, simulator
seed 100000, demo seed 0, 32 SPP, original ICL captions, FP32 video state, observation
offload and padding compaction:

| Task / job | Allocated minutes | Simulator actions | Success |
| --- | ---: | ---: | --- |
| open_microwave / 29284241 | 20.8 | 860 | Yes |
| place_object_scale / 29284242 | 7.5 | 131 | Yes |
| place_bread_basket / 29284243 | 6.6 | 139 | Yes |

Microwave reached opening fraction 0.6164 against the unchanged 0.6 threshold and
stopped at that action. Total approximate cost was 74.5 SBUs at 128 SBUs/hour.
This validates these episodes, not arbitrary long histories or guidance memory.
The temporary microwave instrumentation and three-task launcher mode have been
removed; saved results/source snapshots remain intact. The analyzer still supports
historical door logs.

The client retains early stopping on Robotwin's latched `eval_success`, captures
the terminal observation, and skips the terminal cache update. The next episode
resets the server. Stopping policy is recorded in episode metadata.

### What changes in memory

`COMPACT_OBSERVATION_CACHE=1` discards only observation entries whose sequence,
frame and modality IDs are all -1. Such alignment padding cannot satisfy the
attention mask's valid-sequence condition. The existing cleanup prunes metadata
and every layer's K/V tensors together, both before encoding and after updates.
All valid observations/actions, both CFG branches, the human prefix and the 64-phase
history window are preserved. Compaction defaults OFF outside explicit tests.

For the observed resolution: 788 valid robot entries/chunk were stored as 896
entries. Compaction removes the 108 padding entries instead of retaining them until
the window boundary. Human padding is unchanged. The 64 window counts alternating
video/action phase IDs, approximately 32 chunks. Retained entries should plateau
beyond that point. A bounded cache is necessary but does not prove total GPU memory
fits; prediction/update workspaces, simulator memory and guidance have their own peaks.

```bash
python experiments/analysis/analyze_stage00.py \
  experiments/stages/00_cache_compaction/manifest.json
python experiments/analysis/analyze.py \
  experiments/stages/00_cache_compaction/manifest.json --sacct
```

Inspect `analysis/stage00_summary.json`, `stage00_memory.png` and CSVs. Each run
also saves `cache_memory.jsonl` (valid counts, physical entries, padding count,
KV bytes, allocated/reserved memory), `phase_separation.jsonl` (encoding and transfer
memory/timing), `phase_actions.jsonl`, and per-chunk camera snapshots. A plateau
check requires at least 36 cache updates and five equal final stored-entry counts
with zero observation padding. Allocated-memory variation is reported separately;
this is not a promise of constant CUDA reserved memory. Short successful episodes,
missing data, OOMs and timeouts do not validate long history.

Check cup outcomes/videos and inspect remaining GPU headroom before proceeding.
Stage 00 validates unguided encoding/cache behavior; it does not establish enough
memory for external guidance backward. Runtime compaction code is in
`wan_va/modules/icl_model.py`; offloading is in `wan_va/modules/phase_separation.py`.

## 01 — rendering benchmark

New evaluation tasks: **place_object_scale and place_bread_basket**. Model sampling
seed defaults to **1234**, fixed across paired settings; use `--seeds` for run/demo
seed groups. A rendering preview/submission defaults to three episodes on scale:
one episode each at 32, 16 and 8 SPP, with the same seeds. This is a compute-saving
timing comparison without repeat-based uncertainty estimates.

After stage-00 review, explicitly opt into the validated memory configuration:

```bash
python3 experiments/scripts/robotwin_suite.py render \
  --phase-separation --compact-cache --repeats 1 \
  --manifest experiments/stages/01_render/manifest.json --submit
python experiments/analysis/analyze.py experiments/stages/01_render/manifest.json --sacct
```

`--tasks place_object_scale place_bread_basket` runs six episodes instead of three.
Alternatively validate the chosen SPP on bread separately. Inspect saved videos,
matched seconds/action and seconds/chunk, as well as total episode time. An early
failure is not a speedup. Episode profiles exclude expert feasibility checks;
whole-job/client timing includes more setup. Sum exclusive durations, not overlapping
inclusive durations. Camera readback can include GPU waits. Missing runs and
unavailable simulator hooks are reported; missing memory samples are not zero usage.

## 02 — baselines

Both tasks default on this stage. Start with a pilot, then expand based on measured
runtime and remaining budget. Replace SPP 32 below with your measured choice.

```bash
python3 experiments/scripts/robotwin_suite.py baselines --episodes 2 --spp 32 \
  --phase-separation --compact-cache --time 01:00:00 --client-timeout 2400 \
  --manifest experiments/stages/02_baseline_pilot/manifest.json --submit
python experiments/analysis/analyze.py experiments/stages/02_baseline_pilot/manifest.json --sacct
```

This gives 24 episodes (two tasks × three run/demo seeds × two modes × two episodes).
Text-only uses the task instruction, no human prefix, text CFG 5. Original ICL uses
empty robot text and the cached human caption with ICL CFG 5. This is deliberately
**not text-free human-context ICL**. External guidance is off. The same released
checkpoint is used; these are inference conditions, not separately trained models.
Actual conditioning and accepted scene/demo references are recorded per episode.
The model RNG restarts at the same sampling seed each episode.

Ten episodes per seed produces 120 baseline episodes. Pilot and expanded runs overlap;
report the expanded set, not both as independent samples. Pair on actual simulator,
sampling and demo seeds, not just nominal group numbers.

## 03 — temporal guidance

Before submitting, verify DINO weights and memory headroom with a small smoke test.
Both tasks default on; restrict tasks/seeds when debugging.

```bash
python3 experiments/scripts/robotwin_suite.py guidance \
  --tasks place_object_scale --seeds 0 --episodes 1 --spp 32 \
  --phase-separation --compact-cache --time 01:00:00 --client-timeout 2400 \
  --minutes-per-episode 20 \
  --manifest experiments/stages/03_guidance_smoke/manifest.json --submit
python experiments/analysis/analyze.py experiments/stages/03_guidance_smoke/manifest.json --sacct
python experiments/analysis/plot_matching.py \
  experiments/stages/03_guidance_smoke/runs/place_object_scale_JOBID
```

Four conditions: guidance-only and ICL+guidance, each with fixed final-human-frame or
window-matched targets. The matcher uses fresh observed external-camera RGB, cached
DINO features, a forward window of 24 human frames and lookahead 4 (configurable).
It may pause and can get stuck; inspect trajectory plots and robot/reference contact
sheets. No cropping is applied. A target stays fixed throughout one denoising chunk.
DINO reward over changing targets is descriptive, not an independent success metric.

A captured prediction and its exact selected reference RGB are saved for stage 04.
The current weight offloading covers observation encoding only, **not** reward
backward. Validate a single guided run before any expanded batch. If it fails for
memory, resolve that separately rather than consuming a full pilot allocation.

## 04 — precision

Use a completed guided run's capture to compare decoder precision without the full
WAM/simulator. This changes only the reward VAE decoder between FP32 and FP16;
DINO and persistent sampling state remain FP32. It is not full-model FP16.

```bash
export CAPTURE_RUN="$ZERO_WAM_ROOT/experiments/stages/03_guidance_smoke/runs/place_object_scale_JOBID"
sbatch --export=ALL --chdir="$ZERO_WAM_ROOT" \
  --output="$ZERO_WAM_ROOT/experiments/precision_%j.out" experiments/jobs/precision.sbatch
python experiments/analysis/plot_precision.py experiments/precision/replay_JOBID/precision.json
```

Replay uses captured reference RGB, velocity and Euler update, then sweeps guidance
strengths. It measures a local stop-gradient reward surrogate, not task success.
Failures/non-finite values remain in the results. Only after this and guided memory
validation, use the `precision` stage with `--episodes 1` for a small full-episode
comparison (24 episodes across two tasks, three seeds, two modes and two precisions).

## Reproducibility and budget

One A100 allocation costs 128 SBUs/GPU-hour. Startup, cleanup, errors and timeouts
consume allocation; wall-limit ceilings are shown separately from planning estimates.
`analysis/budget.py` is a planning calculator; `analyze.py --sacct` exports accounting.
Pass your **current** balance through `--budget-sbu`; the launcher's original balance
is not updated from the cluster automatically. No jobs are launched by analysis.

Analysis needs `analysis/requirements.txt`; runtime uses the existing installed
Zero-WAM/Robotwin dependencies. To analyze copied results elsewhere, use
`analyze.py MANIFEST --runs-root LOCAL_RUNS --output LOCAL_ANALYSIS`.

Compact CPU regression checks:

```bash
python -m pytest -q tests/test_cache_compaction.py experiments/tests \
  tests/test_phase_separation.py tests/test_conditioning.py tests/test_legacy_icl_text.py
```

The cache test uses the actual pruning functions over 45 chunks and checks exact
valid-KV/metadata preservation and bounded physical storage. It does not load a
checkpoint or certify CUDA numerics. CUDA transfer tests skip without a GPU. Stage
00 is the required full-workload validation. Do not infer paper reproduction or
success-rate improvement from these smoke tests.
