# Zero-WAM: External Inference-Time Guidance

Frozen Zero-WAM with optional human-video ICL, task text, and external DINOv2
guidance. [Installation](INSTALL.md) · [Upstream project](old_readme.md) ·
[Implementation notes](implementation/docs/external-guidance.md)

## Explicit conditioning switches

Each `USE_*` switch is independent. Parameters do not enable a switch.
Robot observations/history are always present.

| Mode | `USE_ICL` | `USE_TEXT` | `USE_GUIDANCE` |
|---|---:|---:|---:|
| Observation/history only | 0 | 0 | 0 |
| ICL only (default) | 1 | 0 | 0 |
| Text only | 0 | 1 | 0 |
| DINO guidance only | 0 | 0 | 1 |
| ICL + text | 1 | 1 | 0 |
| ICL + guidance | 1 | 0 | 1 |
| Text + guidance | 0 | 1 | 1 |
| All three | 1 | 1 | 1 |

- `USE_ICL=1`: cache the human video/latents in the transformer.
- `USE_TEXT=1`: use the current episode instruction in prediction and ICL context.
  When off, both receive empty text embeddings. Cached `.pth` captions are always
  ignored, so ICL cannot silently introduce task text.
- `USE_GUIDANCE=1`: apply DINO reward gradients. With ICL off, a human reference
  still supplies the reward but is not put in the transformer context cache.

**Behavior change:** old ICL runs could include cached caption embeddings even with
`TARGET_TEXT_CFG=-1`. They are not a pure-ICL baseline for the new implementation.
Legacy environment names (`ICL_CFG`, `TARGET_TEXT_CFG`, `EXTERNAL_*`,
`DINO_CHECKPOINT`, and old reference-range variables) no longer control these runs.

| Parameters | Defaults / meaning |
|---|---|
| `ICL_GUIDANCE_SCALE` | `5`; native ICL CFG strength |
| `TEXT_GUIDANCE_SCALE` | `1`; task text without text CFG. Above 1 enables text CFG |
| `GUIDANCE_STRENGTH`, `GUIDANCE_EVERY`, `GUIDANCE_MAX_NORM` | `0.01`, `10`, `1`; correction strength, interval, gradient-norm cap |
| `GUIDANCE_CHECKPOINT` | Local DINO directory; Slurm helper defaults to `checkpoints/dinov2-small` |
| `GUIDANCE_REFERENCE_SOURCE` | `auto`, `video`, or `latent`; auto prefers an existing MP4 |
| `GUIDANCE_REFERENCE_FRAMES` | `0`; comma-separated zero-based RGB indices, e.g. `65` or `58,59,60` |
| `GUIDANCE_DECODER_DTYPE` | `float32`; sampling state is also float32 with guidance |

ICL and text can be combined, but their separate CFG scales cannot both exceed 1
in the current two-branch sampler; unsupported settings raise an error. Model
weights stay BF16. No model weights are trained.

## Saved run outputs

```text
config.json                       # run settings, code revisions, episode metadata
client.log / server.log            # execution and model diagnostics
gpu.csv                           # optional whole-GPU memory/utilization samples
stseed-*/metrics/TASK/res.json     # aggregate success counts
stseed-*/metrics/TASK/profile_episode_N.json
stseed-*/visualization/TASK/*.mp4
```

`config.json` keeps the launch settings and adds an `episodes` list. Each episode
records its actual simulator/demo seed, task instruction, model text confirmed by
the server, selected human paths, effective switches/scales, reference type/frame
indices, sampling steps/precision, completion status, and video/profile paths.
`model_text=""` means no task text was supplied; `task_instruction` is still saved
as the task description. `metadata_verified=false` means an older server did not
confirm effective settings—do not infer them from the requested flags alone.
An unfinished episode remains `initializing` or `running` if interrupted.

`MONITOR_GPU=1` (default) writes gpu.csv every two seconds. Set `MONITOR_GPU=0` to
skip it. Timing profiles do not replace whole-device GPU measurements. Revision
information is now under `config.json.code`; new evaluation runs do not create
revision.txt or an empty generated/ directory. Existing results are not modified.
These structured JSON records can later feed W&B; W&B logging is not enabled yet.

## Explicit reference frames and rendering quality

Reference selection is now a separate, fixed policy. Human-frame DINO features
are cached once at reset; the selected frame(s) stay fixed throughout each robot
chunk. There is no similarity search or automatic progress estimate yet.

```bash
# Microwave: fixed frame 65; render with 16 rather than the default 32 samples/pixel.
TASK=open_microwave GUIDANCE_REFERENCE_SOURCE=latent GUIDANCE_REFERENCE_FRAMES=65 \
  RENDER_SAMPLES_PER_PIXEL=16 USE_ICL=1 USE_TEXT=0 \
  bash jobs/submit_guidance_checks.sh guided
```

`GUIDANCE_REFERENCE_FRAMES=65` selects frame 65; `58,59,60` averages those three
frames' features. Omitting it selects frame 0. Indices are zero-based and checked
against the decoded human sequence. The step diagnostic uses the same setting;
reference-range settings and CLI arguments have been removed.

For future selectors, each sampling request can supply explicit indices:
`{"obs": ..., "guidance_reference_frames": [65]}`. This overrides the fixed default
for that chunk only; invalid/empty indices fail before sampling. Responses return
`guidance_reference`, saved under each episode's `guidance_selections` in config.json.
Selected-frame PNGs are saved once under `references/episode_NNN/`; server logs use
`[guidance-reference]`. The full human feature bank is prepared in small batches;
reference RGB images stay on CPU. Preparing the full sequence can add startup time.

`RENDER_SAMPLES_PER_PIXEL` is a positive integer, default **32**. It is applied to
the preflight and actual RoboTwin scene setups, including subsequent hardcoded
calls to 32; no external checkout edits are required. Path depth/denoiser, camera
cadence and video FPS are unchanged. Config records the requested value and the
value applied through the native SAPIEN setter. Compare profiling and image quality
at 32/16/8 with all other settings held fixed; use the same quality for a paired
baseline and guided run. Sync the updated code and evaluation/step Slurm scripts.
