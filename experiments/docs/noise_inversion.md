# Paired noise-inversion campaign

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
sbatch experiments/jobs/noise_inversion_repair.sbatch
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
