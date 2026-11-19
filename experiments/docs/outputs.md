# Outputs and compute accounting

## Run versus campaign results

`runs/TASK_JOBID/results.csv` has one row per episode. `analysis/results.csv` groups
completed episodes by task/condition. Do not combine the two tables as separate data.

| Run artifact | Meaning |
| --- | --- |
| `config.json` | Effective switches, accepted seeds, episode status, paths and compute metadata |
| `results.csv` | Status, success, actions/chunks, preparation/rollout seconds, FLOPs and corrections |
| `videos/episode_NNN.mp4` | Observed rollout; filename omits the long instruction |
| `references/episode_NNN/` | Robot previews, selected human frames and `selection.jsonl` |
| `diagnostics/client.log`, `server.log` | Client/server evidence and errors |
| `diagnostics/profiles/` | Per-episode exclusive/inclusive timings |
| `diagnostics/job_timing.json` | Allocation, startup and client time, including partial jobs |
| `diagnostics/gpu.csv` | Device memory/utilization samples when monitoring is enabled |
| `diagnostics/flops.jsonl` | Per-request operator estimates and cumulative coverage |

Captures are opt-in (`--capture-guidance`) and live in `diagnostics/capture_step.pt`.
Package inventories, action hashes and detailed cache/offload traces require
`--diagnostics`. The old `_result.txt` and `res.json` summaries are no longer written.

Historical run folders remain readable. Readers look under `diagnostics/` first,
then at the legacy root. Copied resumed runs can remain in their original stage.

## Timing and success

**Preparation** is the per-episode server-reset roundtrip: prompt/reference loading,
CPU reference reconstruction, DINO embedding and/or ICL cache preparation. It is not
restarting the WAM server. **Rollout** is episode wall time minus preparation and
includes scene setup, simulation, inference and video export. Process startup is
reported separately at job level.

Sum exclusive timing components only; inclusive timings overlap. Camera readback
can include GPU waits. A missing hook/measurement is not zero cost.

Success plots show completed episodes with descriptive 95% Wilson intervals.
Missing/infrastructure-failed episodes are reported separately. Episodes sharing
seed groups are not independent model-noise replicates. Use actual accepted scene
seeds for paired comparisons; one successful episode is not a success-rate study.

## FLOPs

`COUNT_POLICY_FLOPS=0` is the default; opt in with `--flop-count` for calibration.
Without counting, runtime operator totals remain missing, not zero. Workload logs
remain available to the existing offline estimator, whose coverage is partial.
A calibrated full-policy offline estimator is not yet implemented.

When enabled, The counter observes actual enabled text,
ICL and guidance paths, plus observation encoding, cache updates and video/action
inference. Guidance includes backward through frozen DINO/VAE weights to the latent;
it does not introduce transformer backward or parameter updates.

- Two operations per multiply-add; registered PyTorch matrix/conv/backward formulas.
- Scalar, normalization and interpolation costs use declared approximations.
- Compiled FlexAttention keeps its original kernel; accounting uses a dense
  attention-matmul bound, not measured sparse-kernel instructions.
- Unknown floating-point operators are listed, never silently treated as covered.
- Simulator/rendering, checkpoint initialization and non-Torch processing are
  outside this neural-policy accounting scope.

`policy_counted_tflops` is the covered arithmetic subtotal, which can contain bounds.
`policy_estimated_tflops` is filled only when no observed float operators are unmapped.
`flops_status` and the detailed trace state coverage; a nonempty total is still an
estimate, not a hardware measurement. Component columns distinguish text, transformer,
reference/observation processing, DINO, reward decode and guidance backward.

Old baselines have only retrospective partial estimates. Do not present them as
full operator counts. Validate coverage on a short new run before rerunning a
baseline campaign. Counting adds Python overhead: runtime comparisons need the
same counting setting. GPU utilization or elapsed seconds cannot reconstruct FLOPs.

## Correction magnitude

`correction_base_percent_median` is the median over updates of:

```text
100 × norm(applied latent correction) / norm(base denoising update)
```

It is not a percentage action change. `clipped_corrections` counts gradient clipping
before multiplication by strength; it is not clipping of the final correction.
