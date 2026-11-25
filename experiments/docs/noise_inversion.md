# Paired noise-inversion campaign

## DAS6 seven-task pilot

Cluster organization is now `experiments/jobs/snellius/` for the existing launchers
and `experiments/jobs/das6/` for the new DAS6 launchers. Server startup/health checks,
client execution, cleanup, GPU monitoring and timing share
`experiments/scripts/run_evaluation.sh`. The cluster adapters in
`scripts/clusters/` only initialize modules/Conda. The portable CPU download
template stays under `setup/zerowam/`; it is not a Snellius evaluation job.

The DAS6 adapter sources `$HOME/nbhagat-workspace-env.sh`; it never hardcodes a
username or substitutes a different workspace. Robotwin uses
`cuda12.1/toolkit/12.1` and the model subprocess uses `cuda12.6/toolkit/12.6`.
Both run in their existing named Conda environments. Optional module overrides
are `DAS6_ROBOTWIN_CUDA_MODULE` and `DAS6_ZEROWAM_CUDA_MODULE`.

Run these commands **on DAS6 after syncing the final code**, from its checkout:

```bash
source "$HOME/nbhagat-workspace-env.sh"
module load prun
conda activate robotwin
cd "$ZERO_WAM_ROOT"

python experiments/scripts/noise_inversion_campaign.py plan \
  --cluster das6 \
  --directory experiments/stages/noise_inversion_das6_pilot \
  --scene-seeds 100000 --sampling-seeds 1234 \
  --render-samples 16 --hours-per-job 1.5 --estimated-minutes 60

# First calibrate one task; the default first task is bread basket.
python experiments/scripts/noise_inversion_campaign.py submit \
  --directory experiments/stages/noise_inversion_das6_pilot --limit 1

# After checking it, submit the six remaining independent jobs.
python experiments/scripts/noise_inversion_campaign.py submit \
  --directory experiments/stages/noise_inversion_das6_pilot --limit 6

python experiments/scripts/noise_inversion_campaign.py report \
  --directory experiments/stages/noise_inversion_das6_pilot
```

To submit all seven immediately, use `--limit 7` instead of the two submit calls.
There are no dependencies. Each job requests `defq`, one `gpu:A6000`, four CPUs,
and a 90-minute ceiling. The complete pilot is seven paired trials / 21 simulator
rollouts, plus inversion/reconstruction, not 21 Slurm jobs. A feasible setup is
required, but teacher failure still proceeds to inversion and replay. Scene
replacement attempts are recorded and bounded, as before.

DAS6's general job policy limits working-hours jobs to 15 minutes; longer jobs
are for nights/weekends unless site permission applies. This pilot's full paired
trials do not fit the 15-minute template. Arrange the launch window accordingly;
no automatic scheduling, permission request or submission is performed here.
Source: https://www.cs.vu.nl/das/jobs.shtml

The model profile explicitly retains FP32 video state, BF16 transformer, compact
observation caching, phase separation/transformer offload, expandable allocation,
ray tracing with OIDN and depth 8, and 16 samples per pixel. No rasterization or
reward guidance is enabled. Native ICL cached-caption behavior is unchanged:
USE_TEXT=1 in capture preserves that context, while target text remains empty;
control/inversion/replay disable text and ICL. Do not reinterpret the flag alone
as a new instruction-conditioned experiment.

Per-trial files:
`noise_inversion_das6_pilot/trials/TASK/scene_100000_noise_1234/`.
Job/server/GPU logs:
`noise_inversion_das6_pilot/jobs/TASK_noise_1234/`, including `slurm_JOBID.out`.
Reports are generated explicitly with the report command. Completed trials are
skipped; retries use `submit --retry-incomplete`. Source changes require a fresh
plan; do not point this new runtime at the historical Snellius campaign manifest.
SBU fields are left blank for DAS6; wall time and GPU-hours are recorded.

The adapter is locally tested with stubbed modules/Slurm, not executed on DAS6.
Successful cup inference is a useful prerequisite, but does not establish that
all seven tasks, native rendering, inversion, or long cache growth are stable.


Run three conditions on the same task, scene, original Gaussian video noise and
original action noise: **ICL**, **no ICL/language**, and **no ICL/language with
recovered video noise**. Robot observations and action history remain enabled.
No adapter training or external reward guidance is involved.

## Plan, calibrate, then expand

Run from the checkout root on Snellius, after the usual environment/checkpoint
setup. The planner is CPU-only and never submits jobs. Its default grid is all
seven unseen tasks × five scene seeds × two model-noise seeds = 70 paired trials
(210 simulator rollouts plus numerical inversion/reconstruction), grouped into
**14 Slurm jobs**: one task × one model-noise seed per job. Each job starts the
inference server once and processes its five scene trials sequentially. Scene and
sampling changes use episode resets, not model-server restarts.

```bash
python experiments/scripts/noise_inversion_campaign.py plan \
  --directory experiments/stages/noise_campaign_01

# First batch: seven task/noise jobs, each covering five scenes.
# Use --limit 1 instead to calibrate one grouped job first.
# Submit while logged into the allocation you want charged.
python experiments/scripts/noise_inversion_campaign.py submit \
  --directory experiments/stages/noise_campaign_01 \
  --limit 7

# Once those finish, regenerate tables and inspect runtime before expanding.
python experiments/scripts/noise_inversion_campaign.py report \
  --directory experiments/stages/noise_campaign_01

# Continue with the next eligible trials, optionally using the second account.
python experiments/scripts/noise_inversion_campaign.py submit \
  --directory experiments/stages/noise_campaign_01 \
  --limit 7
```

Jobs are ordered noise → task, so the first seven cover all seven tasks at
sampling seed 1234. The next seven use sampling seed 5678. `--limit` counts jobs,
not scenes; `--limit 14` submits the entire default grid. Jobs within one submit
batch have no dependencies and can run in parallel, subject to Slurm availability. Available tasks: `place_bread_basket`, `place_object_scale`, `open_microwave`,
`place_empty_cup`, `stamp_seal`, `move_stapler_pad`, `stack_blocks_three`.
Use `plan --tasks ... --scene-seeds ... --sampling-seeds ...` for a smaller grid.
Scene seeds are initial scene-slot candidates (e.g. 100000), not the legacy
`SEED` offset. If a candidate fails expert feasibility, record it and automatically
try the next unused seed until a feasible scene is found. The other planned scene
seeds are reserved: replacements start after the largest planned seed, skipping
all candidates already used for this task. This preserves five distinct accepted
scenes per task, with both noises evaluated on each accepted scene.

Each task has a locked `scenes.json` registry shared by all of its noise trials.
Once a slot is accepted, later noises recheck that exact scene; they never replace
it independently. Unexpected exceptions/crashes are logged as errors, not skipped
as infeasible. Explicit instability, failed expert checks and the known expert
missing-grasp-pose assertion trigger replacement.
`plan --max-scene-attempts 20` bounds search per slot, cumulatively across retries.
Exhausting the limit or the job's walltime leaves the trial incomplete, rather than
spending without a bound. Allocation limits are left to Slurm.
Directories keep the requested slot seed in their name; episode configs and
summary.json contain the actual accepted simulator seed. `scene_attempts.csv`
exports every rejected/accepted/error/recheck event and its reason.

Planning defaults: one A100, three-hour wall limit per grouped job, 20 estimated
minutes per paired scene trial,
128 SBUs/hour, 16 render samples/pixel, FP32 video state, BF16 transformer, compact
observation cache, and phase offload disabled. The estimate is configurable and
not a measured seven-task runtime. Default jobs request three hours for all five scenes, not three hours per scene.
`--hours-per-job`, `--estimated-minutes`, and `--sbu-per-hour` are plan options.
A submit batch defaults to seven jobs; there is no automatic expansion.

`--account` is optional: without it, no account flag is passed to `sbatch`, so
Slurm uses your login/default submission settings. An explicit override is still
available. The launcher does not impose a local SBU budget; Slurm handles allocation
limits. Submissions retain an account/login label for reporting only.
`costs.csv` has one row per grouped job and estimates elapsed-time
cost from its shared job logs (without multiplying cost by the scene count); `sacct` remains authoritative. A timeout/crash is never a task
failure. A completed unsuccessful episode is a valid result.

## Output: one trial, one readable directory

```text
noise_campaign_01/
  manifest.json                 # Grid, settings, account/job submissions
  source.zip                    # Code snapshot and file hashes
  results.md                    # Supervisor's three-column success table
  results.csv                   # Counts, rates, rescues, regressions, subgroups
  trials.csv                    # One row per planned scene/noise pair
  costs.csv                     # Measured job time and estimated SBU usage
  scene_attempts.csv            # Feasibility attempts, accepted seeds and reasons
  jobs/
    place_object_scale_noise_1234/
      settings.json             # Task/noise group and its five trial paths
      summary.json              # Completed/skipped/incomplete scenes for the job
      diagnostics/              # Shared server/GPU logs and job timing
      slurm_JOBID.out
      attempts/                 # Archived diagnostics from earlier job attempts
  trials/
    place_object_scale/
      scenes.json               # Scene-slot registry shared across noise seeds
      scene_100000_noise_1234/
        trial.json              # Immutable source digest and protocol identity
        summary.json            # Paired outcomes, or incomplete status/error
        capture/                # ICL config, client.log, video and diagnostics
        control/                # Original-noise/no-language/no-ICL rollout
        replay/                 # Recovered-noise/no-language/no-ICL rollout
        chunks/                 # Original noise, generated video, predicted actions
        history/                # Encoded camera observations and normalized actions
        inverted/               # Recovered video noise + original action noise
        inversion.json          # Per-chunk reconstruction errors and timing
        attempts/               # Archived partial scene phases
```

Capture also saves camera snapshots under its diagnostics directory for later
adapter datasets. Tensor artifacts are CPU tensors loaded with `weights_only=True`.
No historical pilot results are moved or relabelled.

## Protocol

1. **Capture:** human demonstration + cached human caption, empty target text,
   ICL CFG 5, no reward. Save every generated chunk, including unsuccessful
   episodes. The last generated chunk may be only partially executed.
2. **Control:** original video/action noise, no ICL/language/reward; same actual
   scene and instruction metadata. Scene, rendering, effective model settings,
   action limit and sampling seed must match the source.
3. **Invert:** no ICL/language/reward. Recompute robot-history KV representations
   from recorded observation latents and normalized actions, without reusing
   ICL-conditioned KV values. Integrate clean video (sigma 0) → noise (sigma 1),
   then forward again to measure reconstruction error. This does not move the robot.
4. **Replay:** recovered video noise and original action noise at the beginning
   of each chunk. Normal MoT + scheduler sampling follows. History comes from
   this rollout's live observations and predicted actions.

Both control and replay receive the normal task action limit. When saved tensors
end, **both continue with their seeded Gaussian sequence**. Ordinary video/action
random draws are consumed even on overridden chunks, preserving the sequence
position. Metadata records exhaustion, the first fallback chunk and fallback
chunk count. Report full-episode success **and** success before saved-noise
exhaustion. A success after fallback is not evidence that the recovered sequence
alone completed the episode. The source horizon is measured in generated chunks,
not the number of executed actions in its partial terminal chunk.

The initial observed frame is clamped during integration and excluded from the
relative latent error `||reconstruction - target|| / ||target||`. Reverse Euler is
approximate. Recovered noise need not have unit Gaussian variance; do not normalize
it. Numerical reconstruction uses recorded history, not the replay's live history.

## Resume and reproducibility

Completed phases are reused only under the same trial settings, protocol and
runtime-source digest. Code drift between planning, queueing and execution is
rejected. Keep the checkout unchanged while jobs are queued; make a new plan for
new code. Source snapshots include the noise-inversion experiment package. Scene search
is recorded in the manifest and trial identity. Use a new plan after updating
this implementation; old source fingerprints intentionally cannot be resumed
with changed code. The grouped-job manifest uses schema version 3; make a new
plan after syncing this update. Historical campaign reports remain readable.
Checkpoint paths and effective settings are recorded and compared; model weights
must also remain unchanged at those paths (the snapshot does not archive weights).

After an incomplete job, inspect its logs. Explicitly retry stopped jobs with:

```bash
python experiments/scripts/noise_inversion_campaign.py submit \
  --directory experiments/stages/noise_campaign_01 \
  --limit 7 --retry-incomplete
```

Active Slurm jobs are not resubmitted. Groups whose scene trials are all complete
are skipped. In a retried group, completed scenes are skipped and completed phases
inside incomplete scenes are reused. Partial phase outputs are archived before
retry. A scene error is recorded and other scenes are attempted with the same
server; the grouped job exits nonzero if any scene remains incomplete. A capture retry invalidates downstream phase artifacts. Previous shared diagnostics
are preserved under `jobs/<task>_noise_<seed>/attempts/`. An interrupted `sbatch` call recorded as
`submitting` requires checking the queue before resolving that record: it is never
silently resubmitted. The runner locks each trial to prevent concurrent writes.

All three percentages use the same denominator: trials where all conditions and
inversion completed and paired settings passed validation. `results.csv` also
shows missing trials, distinct scenes, reversal success split by ICL-source
success/failure, rescues (control failure → replay success), and regressions
(control success → replay failure). Zero eligible rescue cases produce an empty
rate, not a misleading 0%. Repetitions sharing a scene are correlated.

## Existing pilot commands

The original `noise_inversion_pilot.py` and `noise_inversion.sbatch` remain small
compatibility entry points for the two-task pilot. New pilot runs now include the
control and invert completed failures too. `noise_inversion_control.sbatch` can
still append controls to the historical pilot at:

```text
$HOME/Zero-WAM/experiments/stages/noise_inversion_pilot/runs/place_bread_basket_29492432/noise_inversion
```

Override `NOISE_INVERSION_CONTROL_SOURCE` for another historical pilot. Historical
stopped-at-exhaustion replays keep their original meaning; new campaigns use the
explicit Gaussian-continuation protocol. Use new campaign outputs for the common
three-condition table rather than mixing historical protocols.

## Plot and analyze a campaign

After a batch (or while the campaign is still running), use the CPU-only report
command. It does not run inference or submit jobs:

```bash
python experiments/scripts/noise_inversion_campaign.py report \
  --directory experiments/stages/noise_campaign_01
```

This refreshes the CSV tables and `results.md`, adds a short per-task comparison,
and writes:

- `plots/success.png`: three-condition success percentages, annotated with exact
  success/trial counts. Hatched portions identify reversal successes after Gaussian
  fallback. Incomplete and pending trials are excluded and their counts shown.
- `plots/rescue_regression.png`: rescue rates among control failures and regression
  rates among control successes, each with its own numerator/denominator. No eligible
  trials are shown as N/A, not zero percent.

`results.md` embeds both figures. Rates are descriptive; repetitions sharing a
scene are correlated, and the report does not claim statistical significance.
Plotting needs the existing `experiments/analysis/requirements.txt` dependencies
(matplotlib). Use `report --no-plots` for tables/text without matplotlib. Planning
always writes tables only, so it retains no plotting dependency.

## RoboTwin success overrides

Before client imports, the job launcher installs the repository's released
`move_stapler_pad.py` or `stamp_seal.py` into the external RoboTwin checkout when
running that task, preserving an upstream backup and using a patch lock. The
criteria are unchanged: stapler placement requires XY proximity, physical pad
contact and both grippers open; seal placement requires XY error below 3 cm on
each axis and both grippers open. Setup also installs these two overrides.
The grouped launcher retains this step, so capture/control/replay use the same
success definition. These files were already present; no criteria restoration
or relaxation was needed for the grouped campaign.

## Small repair pilot after a stalled/OOM campaign

Stop the old stalled jobs first and sync the updated code. From the checkout:

```bash
sbatch experiments/jobs/snellius/noise_inversion_repair.sbatch
```

This requests **one GPU for at most 45 minutes** and runs only three source
captures, sequentially on one model server:

| Task | Requested scene slot | Model noise | Check |
| --- | ---: | ---: | --- |
| Stapler pad | 100002 | 1234 | Previously stalled action execution |
| Stack blocks three | 100000 | 1234 | Previously exhausted memory on a long rollout |
| Object scale | 100001 | 1234 | Missing-grasp-pose setup rejection and scene replacement |

The default campaign is `$HOME/Zero-WAM/experiments/stages/noise_inversion_full`.
Override `NOISE_INVERSION_REPAIR_SOURCE` if needed. New captures/noises/snapshots
are saved directly under the original `trials/TASK/scene_SLOT_noise_NOISE/` paths.
An explicit repair archives old incomplete trial artifacts and their original
identity under that trial's `attempts/` before starting fresh. Fully completed
paired trials are never overwritten; an active trial lock also prevents repair.

Shared logs and `repair_summary.json` are under
`noise_inversion_full/repair_jobs/job_JOBID/`. Each source's own log remains
`trials/TASK/scene_SLOT_noise_NOISE/capture/client.log`.

This pilot does **not** run control, inversion or replay. Its summary remains
`incomplete` for the paired experiment and includes `pilot_capture_complete=true`
when source capture finishes; failed policy execution still counts as a finished
capture. Do not interpret these diagnostic captures as complete paired results.

The pilot enables the existing `PHASE_SEPARATION=1` transformer offload around
VAE encoding. This is a memory mitigation requiring GPU validation, not a proved
numerically identical replacement for the old sharded path. Matched full trials
must use the same offload configuration for all conditions; do not mix old
completed phases with new offloaded phases. The main campaign default is left
unchanged pending this test.

Action execution, observations and expert execution have a **90-second watchdog**
that dumps a stack trace and exits instead of waiting hours. A source phase also
has a 20-minute process timeout; the whole pilot has a 40-minute client timeout.
The watchdog bounds/diagnoses a native stall; it does not claim to repair an
unknown planner/physics deadlock. Inspect its traceback before another run.

Only the exact observed expert assertion `target_pose cannot be None for move
action.` is treated as infeasibility; other assertions, GPU failures and crashes
remain errors. Scene selection also writes capture-local `scene_selection.json`
receipts, recovers accepted slots from these receipts, and can recover a missing
receipt from a completed historical capture. Conflicting accepted scenes are
rejected, not silently reconciled. Shared JSON records now use unique temporary
files and fsync before atomic replacement. These harden persistence without
claiming that the original registry inconsistency's underlying cause was proved.

## Staged camera and OOM diagnostics

Keep 16 samples/pixel, path depth 8 and OIDN fixed initially. Submit stage 1:

```bash
sbatch experiments/jobs/snellius/noise_inversion_camera.sbatch
```

This requests one GPU for at most 15 minutes, with no model server. It captures
200 observations per scene for stapler seed 100002 and scale seed 100006, with a
five-minute process limit per scene. Each individual take_picture has a 90-second
watchdog. Per-camera begin/end records are flushed to camera_timing.jsonl; the
last unmatched begin identifies the blocked camera. First and last RGB frames
are saved; every frame is checked for finite uint8 image data. These fixed-scene
captures include physics stepping but do not reproduce the original robot actions.
Passing does not rule out a pose-dependent failure.

Only after inspecting stage 1, submit the same captures with model weights loaded:

```bash
sbatch --export=ALL,CAMERA_LOAD_MODEL=1 experiments/jobs/snellius/noise_inversion_camera.sbatch
```

This stage also requests at most 15 minutes. It tests baseline model/renderer GPU
coexistence; it does not build ICL/history caches or run inference. If stage 1
fails, do not pay for stage 2 before inspecting the camera traceback. Outputs are
under noise_inversion_full/camera_tests/denoiser_DENOISER_model_STAGE_job_JOBID/TASK/, with shared
server/GPU logs in the job's diagnostics/. No paired trial results are overwritten.
A denoiser-off comparison is deliberately not automatic: choose it after reviewing
the unchanged-renderer results, and record it as a different rendering condition.

Test long-history OOM separately:

```bash
sbatch experiments/jobs/snellius/noise_inversion_oom.sbatch
```

This is a single stack_blocks_three source capture (slot 100000, noise 1234),
maximum 35 minutes (30-minute source deadline), using expandable_segments allocation plus existing VAE
transformer offload. It retains the normal 1200-action task limit and full KV
history. It explicitly repairs the original incomplete trial, preserving the old
attempt, and uses the same trial and repair_jobs paths as the repair pilot above.
Success means completed capture (policy success or reaching the normal action
limit), not merely getting past chunk 32. A timeout is inconclusive for full-horizon
memory safety. Compare cache_memory.jsonl, phase_separation.jsonl, GPU logs and
any allocation traceback. Expandable allocation may reduce fragmentation; it does
not bound live KV-cache growth. A repeat OOM requires another memory strategy
before a full campaign. No control/inversion/replay is run by these diagnostics.


### Findings and next comparison (23 November)

Camera-only job 29633317: stapler slot 100002 stalled on static[1] during the
second observation, without Zero-WAM. Scale seed 100006 passed all 200
observations. These are fixed-scene tests; the scale pass does not rule out
pose-dependent or intermittent failures in a rollout. Skip the model-loaded
camera stage for now and isolate denoising:

```bash
sbatch --export=ALL,CAMERA_LOAD_MODEL=0,CAMERA_DENOISER=none experiments/jobs/snellius/noise_inversion_camera.sbatch
```

This changes only denoising (16 samples and the external depth-8 setup remain).
The diagnostic intercepts RoboTwin's hardcoded denoiser setter before renderer
creation so setup cannot reenable OIDN. No external checkout is edited.
Requested settings are saved in each task's diagnostic_settings.json. If it
passes, repeat before treating it as reliable; if it stalls, inspect camera
labels and scene assets next. Do not adopt denoiser-free images for paired
research results based on this diagnostic alone.

OOM job 29633319 reached action 1072/chunk 33 without a recorded OOM, then the
1200-second source timeout stopped it. That timeout was too short to establish
full-horizon safety. The OOM launcher now allows a 1800-second source deadline
inside a 35-minute allocation; it retains the normal 1200-action limit and full
history. It still tests only capture, not inversion/replay, and archives the
previous incomplete attempt before rerunning.

Upstream context: RoboTwin issue 83 reports scene-dependent rendering slowdowns
at take_picture on A100/H100/V100 and discusses a pencil asset workaround:
https://github.com/RoboTwin-Platform/RoboTwin/issues/83
This is a lead, not evidence that our clean stapler scene contains that asset;
do not remove assets or skip scenes without checking the actual scene first.

### Rasterization comparison and stack continuation (24 November)

The denoiser-off camera job 29648541 still failed for stapler, with Vulkan
ErrorDeviceLost at image readback; scale passed 200 frames. Compare the same
scenes with the camera pipeline changed from rt to default (rasterization):

```bash
sbatch --export=ALL,CAMERA_LOAD_MODEL=0,CAMERA_DENOISER=none,CAMERA_SHADER=default experiments/jobs/snellius/noise_inversion_camera.sbatch
```

This is simulator-only, maximum 15 minutes. Denoising stays off to match the last
comparison; task, seed, resolution and sampling/depth configuration are retained.
Ray samples/depth do not apply to the raster pipeline. Outputs include shader in
the path: noise_inversion_full/camera_tests/shader_default_denoiser_none_model_0_job_JOBID/.
Each task writes scene_info.json (including static camera names) and requested
settings. This does not change the main campaign renderer or the external checkout.
A pass implicates the ray-tracing path but is not a production fix; repeat and
compare images before adopting a different observation distribution.

The stack capture from OOM job 29648542 completed 1200 actions/38 chunks without
OOM, with task success=false. To reuse it for the remaining paired phases:

```bash
sbatch experiments/jobs/snellius/noise_inversion_continue.sbatch
```

This requests one GPU for at most 90 minutes for control, inversion/reconstruction,
and replay. It never calls capture and preserves capture/, chunks/, history/ and
trial.json. All saved chunk/history files must be present; a pre-server check
fails if the capture is incomplete or its archived runtime differs. Model,
evaluation and noise-inversion runtime hashes must match the source snapshot;
only camera_diagnostics.py is exempt because it is not imported by the capture
or continuation path. Launcher and diagnostic script edits are tracked separately.
The snapshot must match the saved capture's digest; original identity is retained.

Baseline rendering stays rt/OIDN, 16 samples/depth 8. Serving remains FP32 video
state, BF16 transformer, compact observation cache, VAE phase offload and expandable
allocation. Source/control/replay settings are validated using episode metadata.
Both control and replay disable language, ICL and reward guidance. The normal
1200-action limit/full history remain; control/replay each have 30-minute phase
deadlines, with 85 minutes total client time. Inversion may still fail or OOM:
passing capture alone does not validate numerical inversion/replay.

Trial outputs stay under trials/stack_blocks_three/scene_100000_noise_1234/;
continuation.json records the new digest separately. Current source snapshot,
server/GPU logs and final continuation_summary.json go under
noise_inversion_full/continuation_jobs/job_JOBID/. Partial control/replay/inversion
artifacts are archived on retry; completed compatible phases are reused. Source
success=false remains a valid paired result, not evidence of successful ICL rescue.

## Proposed research and cleanup plan (not implemented)

This section interprets the supplied NoiseICL_research_plan.pdf (pp. 1–3).
No collector, adapter, new inversion method or guidance removal is implemented
as part of the cluster reorganization. The seven-task DAS6 pilot executes the
existing capture/control/invert/replay pipeline.

### Current implementation versus the proposal

| Question | Current implementation | Proposal | Consequence |
| --- | --- | --- | --- |
| When is the teacher queried? | A complete ICL source episode is captured first. | Teacher is queried at each chunk using the current rollout history. | Current approach is simpler, resumable and amortizes a saved source; online teacher is costlier but supplies an oracle at the actual visited state. |
| Which history is inverted? | The saved source history is rebuilt without ICL for each source chunk. | Teacher generation, inversion and reconstruction share the current chunk's fixed history. | Our numerical reconstruction is matched to source history, but the closed-loop replay history can diverge. |
| How is noise reused? | Recovered noise for source chunk i initializes replay chunk i. | Fresh teacher/inversion target is produced for each current chunk. | Failure of saved-noise replay can reflect a history mismatch, not necessarily bad inversion. |
| What is already measured? | Three closed-loop success outcomes, reconstruction errors, Gaussian fallback and incomplete phases. | Same outcomes plus an online matched-history oracle and training pairs from recorded seen-task trajectories. | Keep existing results as saved-noise transfer; do not relabel them as the proposed online oracle. |
| Are failed teachers used? | Inversion/replay run even when source success=false. | Data are generated across stages, not only successful episodes. | Keep failures, with explicit outcomes and numerical-quality flags; do not filter by success to inflate evaluation. |
| Is the teacher literally text-free? | Target text is empty; native ICL may retain cached caption embeddings. | The document describes the ICL teacher as no-language, with optional prompt input to the adapter. | Audit/report actual context. Preserve native behavior for comparability now; test a caption-free variant separately later. |

Recommendation: keep the current implementation through the seven-task pilot
and initial multi-seed evaluation. It directly asks whether a saved successful
behavior can be transferred via initial noise. Add the proposal's matched-history
collector before training: it supplies a well-defined target for the observation
that the deployed adapter will actually receive. An online oracle is an upper-bound
experiment requiring a teacher at deployment, not the final learned policy.
Neither method is universally better: saved-noise replay tests transfer under
closed-loop divergence; online reversal removes that confound at higher cost.

### Gates, data volumes and evaluation

1. Technical gate: one scene/noise pair on each of the seven unseen tasks (seven
   pairs, 21 rollouts). Verify conditioning/cache isolation, complete tensor/history
   sequences, finite inversion/reconstruction, task success criteria, output paths,
   renderer stability, peak memory, wall time and fallback accounting. This is a
   smoke test, not a meaningful success-rate estimate. Inspect videos for task
   progress and keep policy failure distinct from runtime failure/infeasibility.
2. Signal gate: 20 distinct feasible scenes per unseen task with one noise seed
   initially: 140 pairs / 420 rollouts. Expand to 30 scenes (210 pairs / 630
   rollouts) if useful and affordable. A second noise on five scenes per task adds
   35 pairs for seed sensitivity; those are not 35 new independent scenes.
   Estimate cost from the DAS6 pilot's per-task times/storage, not Snellius SBUs.
   Do not collect 100/task until this gate establishes a useful signal and stable
   failure handling. Keep all seven tasks excluded from adapter optimization.
3. Dataset audit: confirm access to the released seen-task human/robot pair IDs,
   frame/action trajectories, camera layout, VAE latents and normalization. Do not
   assume human videos alone contain usable robot action histories. Select five
   representative seen tasks covering arm choice, task length and manipulation
   types; choose them without tuning on the seven unseen test tasks.
4. Collector pilot: five seen tasks × 20 paired trajectories × 10 sampled history
   points × two Gaussian seeds = 2000 target pairs, if enough valid points exist.
   Sample across progress uniformly; shorter trajectories contribute fewer points
   rather than duplicate pseudo-independent data. Lock scene/trajectory groups
   before generating seeds. Use 14/3/3 trajectories per task for train/val/test:
   planned 1400/300/300 pairs. Keep shared human-demo and scene duplicates from
   leaking across splits. Keep the test partition untouched during training.
5. Scale only after learnability: 43 seen tasks × 20 trajectories × 10 points × two
   seeds = 17,200 pairs, nominally 12,040/2580/2580 under the same group split.
   Counts are targets, not promises: report invalid inversions, short trajectories
   and missing data. A planning estimate is N_pairs × measured seconds/pair plus
   evaluation/reset overhead, and N_pairs × measured bytes/pair plus shared
   contexts. Benchmark 100 pairs first before scheduling the full 2000.

Every evaluation table should show ICL, no-ICL/no-language control, saved-noise
replay, and later online oracle/learned adapter as separate columns. Report per-task
counts, macro average, paired gain, rescue conditional on control failure,
regression conditional on control success, source-success/source-failure strata,
fallback-free outcomes, failures/incompletes, and time/memory. Use scene-level
paired bootstrap intervals after enough scenes exist; chunk/seed observations
from one scene are correlated. Fix settings and success checks across conditions.
Do not silently replace a renderer in one condition. If RTX ray tracing fails,
complete rendering diagnostics before scaling.

### Collector and target schema to implement later

For a fixed robot history o and demo h, sample epsilon, query the frozen ICL
teacher for y, rebuild unconditioned robot-only context, invert y to epsilon_star,
and reconstruct y_hat under that same history. Save Delta=epsilon_star-epsilon.
Teacher and unconditioned KV/VAE states must be isolated; human-prefix tokens must
not leak into inversion or replay. Replay original prior actions when rebuilding
history. Repeat two seeds at each same history to learn the seed-dependent map.

Shared context records: task and split, scene/trajectory/demo IDs, actual seed,
human latent/reference path plus hash, robot history latents, prior normalized
actions, initial observation, camera geometry/order and masks. Do not repeat large
demo/history arrays for every seed if a stable context ID can reference them.

Pair records: context ID, chunk/progress index, Gaussian seed, original video and
action noise, recovered video noise/residual, teacher future video latent,
reconstruction relative L2, solver grid/steps/dtypes, actual conditioning sources,
source outcome, fallback/runtime/finite flags and checkpoint/code fingerprints.
Mask observed/clamped frames and padding in the regression loss: their recovered
values are constrained rather than unconstrained noise targets. Reject invalid
numerical targets and fallback noise, not every unsuccessful teacher episode.
Choose any reconstruction-quality filtering on seen-task validation data only;
compare with the unconditioned numerical round-trip control so solver error is not
mistaken for ICL information.

### Residual adapter experiment to implement later

Start from epsilon_pred=epsilon+f(epsilon,h,o), with a zero-initialized output head
and supervised masked residual MSE, without backbone gradients. Include original
Gaussian noise, frozen-VAE demo tokens, current robot/history latents and prior
actions. Use spatiotemporal token positions and modality IDs; include chunk/progress
as an optional feature/ablation. This network is called once at initialization
per chunk, so it does not inherently need the diffusion solver timestep as input.
Prompt embeddings and task IDs are optional ablations, not required default
inputs for a task-agnostic adapter. Cached teacher captions must still be logged.

First overfit about 100 training pairs. Compare a small residual MLP/coarse
convolutional baseline with the proposed patch transformer (eight blocks, width
512, eight heads, cross-attention to context). The transformer is attractive for
spatiotemporal structure and variable demo/history context, not required by the
fact that the target is noise. Start AdamW at 1e-4, BF16, effective batch 64, up to
5000 steps; choose capacity/stopping from grouped validation. Measure residual MSE
against the zero-residual baseline and reconstruction error, then evaluate
closed-loop success on held-out seen scenes and the seven unseen tasks. Add
shuffled-demo and history/seed ablations. No teacher/inversion runs at learned
inference; action generation remains unchanged. Frequency decomposition, RL and
backbone fine-tuning are later questions, not prerequisites.

### Guidance cleanup proposal to review before edits

1. Establish golden seeded inputs/outputs and conditioning metadata on one GPU,
   covering native ICL, empty-text/no-ICL sampling, inversion and replay. Preserve
   the current campaign/source snapshots and results; do not modify historical
   outputs or compare exact floating-point equality across different GPU stacks.
2. Move reusable FLOP/workload/reset timing from wan_va/guidance into a neutral
   diagnostics package. Preserve cache/GPU logging, inference timing, watchdogs,
   package/hardware/code metadata and the phase-separation diagnostics. Audit
   shared latent/reference utilities before deleting any guidance module.
3. Remove only external reward guidance: DINO reward/gradient steps, temporal
   reward matching/selectors, reward captures, DINO checkpoint/config options and
   guidance-only launch paths/tests/docs after confirming no remaining imports.
   This removal is not part of the current cluster changes.
4. Retain ICL classifier-free guidance and its scale, target/ICL caption selection,
   text-conditioning switches, empty embeddings and robot history caches. These
   belong to native model conditioning even though their names include guidance.
5. Rerun numerical/conditioning tests and one full paired episode; compare with
   the golden cases before collecting targets. Version the dataset by the final
   scientific runtime fingerprint and keep legacy result readers/metadata intact.
