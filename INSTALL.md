# Installation

Run commands from the Zero-WAM repository root. Use separate Conda environments
for **Zero-WAM inference** and **RoboTwin simulation**.

## Requirements

- Linux x86-64, Conda, Git, and internet access.
- An NVIDIA GPU and working Vulkan support for simulator evaluation.
- A CUDA toolkit and compatible C++ compiler for source builds.
- Shared storage with sufficient personal quota. The checkpoint alone is about
  35 GB; allow additional space for environments, assets, data, and extraction.

Check your quota before downloading. Filesystem free space (`df -h`) is not your
personal allowance. Keep large files outside a small home directory. Initialize
Conda with `source /path/to/conda/etc/profile.d/conda.sh` if needed. On a managed
cluster, load its modules and use allocated compute resources for builds/runs.

## 1. Zero-WAM environment

Skip creation if your `zerowam` environment already works.

```bash
conda create -n zerowam python=3.10 -y
conda activate zerowam
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.inference.txt
python -m pip install -e . --no-deps
python -m pip check
```

Inference uses BF16 model inputs and FP32 video state for noise inversion.
Multiple GPUs do not automatically pool memory. The new residual adapter trains
in the existing zerowam environment without updating the backbone.

## 2. RoboTwin environment

The project installer pins RoboTwin and its source dependencies and applies the
required compatibility/evaluation patches. Do not run RoboTwin's upstream full
installer afterward.

```bash
export ROBOTWIN_ROOT="$HOME/Robotwin"
bash setup/robotwin/install.sh base

# Load a compatible CUDA toolkit/compiler before this stage.
# Use your GPU's compute capability: A100 = 8.0, RTX A6000 = 8.6, Titan RTX = 7.5.
export TORCH_CUDA_ARCH_LIST="8.0"
bash setup/robotwin/install.sh source
```

The default environment name is `robotwin`. To preserve an existing environment,
set `ROBOTWIN_ENV=robotwin-new` before **every** stage. To intentionally update
an existing one, rerun `base` with `ALLOW_EXISTING_ENV=1`. Failed source builds
can be retried with `source`; local Git changes are not discarded.

Pins: RoboTwin `2eeec322`, PyTorch 2.4.1/CUDA 12.1, CuRobo `v0.7.8`, and PyTorch3D
commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`. Python dependencies are in
`setup/robotwin/requirements.lock.txt`. This is a package snapshot and build
recipe, not a container image or a guarantee of compatibility with every driver.

If Warp 1.6.0 fails to load with `GLIBC_2.29 not found` (for example on DAS6
with GLIBC 2.28), build the same pinned version locally instead of replacing
system GLIBC. In an allocated build session with Conda initialized:

```bash
# DAS6: load the toolkit matching RoboTwin, not Zero-WAM's CUDA 12.6 runtime.
module load cuda12.1/toolkit/12.1
conda install -n robotwin -c conda-forge git-lfs -y
bash setup/robotwin/install.sh warp-source
```

This keeps an editable Warp checkout under `$ROBOTWIN_ROOT/third_party/warp`;
retain it with the installation. The stage can take longer than 15 minutes and
builds for multiple GPU architectures independently of `TORCH_CUDA_ARCH_LIST`.
Rerunning `base` may replace the source installation with the pinned wheel;
rerun `warp-source` afterward if necessary. A source build fixes the binary
GLIBC mismatch, but does not establish driver compatibility: Warp CUDA API errors
must still be investigated even if device detection/import checks pass.

## 3. Simulator assets

Assets are separate from HumanGen. No GPU is needed for downloading them.

```bash
bash setup/robotwin/install.sh assets
bash setup/robotwin/install.sh verify
```

Assets live under `$ROBOTWIN_ROOT/assets`. For external storage, move that
folder and symlink it back before downloading. Keep the repository and editable
CuRobo checkout in place. Extraction needs space for both archives and their
contents. If extraction fails, fix storage and rerun; imports do not verify assets.

## 4. Checkpoint and HumanGen downloads

Activate `zerowam` and authenticate once. Never put tokens in scripts.

```bash
conda activate zerowam
hf auth login
hf auth whoami
export STORAGE_ROOT=/path/to/large/shared/zero-wam

# Plan only: shows selected files, revisions, and required space.
bash setup/zerowam/download.sh all --storage-root "$STORAGE_ROOT"
```

After confirming sufficient personal quota, replace `REMAINING_GIB` with your
actual remaining allowance and download:

```bash
bash setup/zerowam/download.sh all --storage-root "$STORAGE_ROOT" \
  --quota-free-gib REMAINING_GIB --download --link
```

| Option | Purpose |
| --- | --- |
| `checkpoint`, `dataset`, or `all` | Select what to fetch |
| `--profile eval` (default) | 20 human VAE latents per held-out task (140 total), available reference MP4s, and ICL manifests |
| `--profile seen` | Separate collection download: human references and recorded robot files for the 43 seen tasks |
| `--profile seen --tasks TASK ...` | Limit the seen download to named tasks, e.g. the five-task pilot |
| `--profile robotwin` | Full simulation subset, including recorded robot videos/actions/latents |
| `--workers 1` | Reduce request concurrency; default is 2 |
| `--link` | Create repository paths expected by inference; refuse conflicting existing paths |

Live evaluation uses human references and simulator observations; recorded robot
trajectories are optional. Rerun the same command to resume, preserving destination
`.cache` metadata. Resolved revisions/completion are saved in `download-manifest.json`.
Use a new storage root for a different revision. Run one downloader per root.
On rate limiting, wait, verify authentication, and retry. The space estimate adds
2 GiB headroom but cannot reserve quota or account for other concurrent writes.

HumanGen may omit original human MP4s while providing their latents. The `eval`
profile accepts this and reports missing optional videos; missing required latents
still fail. The inference server loads cached human latents directly; the
downloader does not reconstruct videos.

If the repository itself is on scratch, download directly into the expected paths
without external-storage symlinks (add `--download --quota-free-gib REMAINING_GIB`
after inspecting each plan):

```bash
bash setup/zerowam/download.sh checkpoint --storage-root "$PWD/checkpoints"
bash setup/zerowam/download.sh dataset --storage-root "$PWD/data" --profile eval
```

For a **CPU-only Slurm job**, initialize required modules in your submission shell:

```bash
export ZERO_WAM_ROOT="$PWD"
export CONDA_BASE="$(conda info --base)"
sbatch setup/zerowam/download.sbatch all --storage-root "$STORAGE_ROOT" \
  --quota-free-gib REMAINING_GIB --download --link
```

Set partition, account, and an allowed walltime using your scheduler's options.
The template defaults to 15 minutes; transfers may need longer. Optional emails:
`sbatch --mail-type=BEGIN,END,FAIL --mail-user=you@example.org ...`.
Saved login credentials must be visible to the job; changing `HF_HOME` can hide
those credentials. Downloads default Xet chunk caching to the storage root.

### Prepare the recorded-history collection pilot on DAS6

After downloading the five seen tasks, prepare the histories and campaign in the
existing `zerowam` environment:

```bash
python -u experiments/scripts/prepare_noiseicl_pilot.py \
  --cluster das6 --humangen-root "$ZERO_WAM_ROOT/data/HumanGen"
cat experiments/stages/04_collect_pilot/pilot.json
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/04_collect_pilot --limit 1
```

The converter reads cached human/robot latents and recorded actions, checks camera
timing/layout, and applies training action normalization. It keeps 20 human demos
per task and uses their published robot pairings to select up to 200 distinct
history points per task. Two seeds per point give at most 2,000 pairs; shorter
tasks produce fewer pairs. Actual counts and shortfalls are in `pilot.json`.
It creates the histories, context index, source snapshot, and job manifests.
Use fresh output directories if preparing a new campaign.

Collection runs model inference on a GPU but does not start a live simulator;
the simulator camera checks below apply to live evaluation. After checking the
first job's timings and reconstruction diagnostics, submit remaining eligible
jobs in a permitted DAS6 GPU window:

```bash
python experiments/scripts/noiseicl.py report \
  --directory experiments/stages/04_collect_pilot
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/04_collect_pilot --limit 80 --max-concurrent 2
```

See [experiments/README.md](experiments/README.md) for grouping rules, output paths,
and limitations of the released simulator/scene metadata. Do not change source
code after planning: submission checks its frozen digest.

## 5. Verify GPU execution and task cameras

Inside an allocated GPU node:

```bash
bash setup/robotwin/install.sh gpu-check
conda activate robotwin
python setup/robotwin/check_camera.py --output results/camera-check
```

Use your custom environment name if applicable. The camera check loads the real
`place_empty_cup` scene and saves head/wrist images without Zero-WAM. It is not a
motion-planning or policy-success test. Stop on errors before launching evaluation.

Vulkan is needed even without a GUI. A missing Vulkan module or an initialization
warning alone does not establish failure; actual camera capture is the check.
Do not install system drivers on managed nodes. Warp 1.6.0 can emit CUDA errors on
some drivers; successful imports or camera capture do not validate planner CUDA.

## 6. Run one evaluation episode

Use two shells on the **same allocated GPU node**, each from this repository root.
The first starts Zero-WAM; the second runs RoboTwin. Text offload keeps the text
encoder on CPU and the VAE/transformer on GPU.

Server shell:

```bash
conda activate zerowam
export MODEL_PATH="$PWD/checkpoints/zero-wam-posttrain-robotwin"
export TEXT_ENCODER_OFFLOAD=1
bash evaluation/robotwin/launch_server.sh
```

Client shell, after the server reports ready:

```bash
conda activate robotwin
export ROBOTWIN_ROOT="$HOME/Robotwin"
TEST_NUM=1 SEED=0 USE_ICL=1 USE_TEXT=0 ICL_GUIDANCE_SCALE=5 \
  bash evaluation/robotwin/launch_client.sh "$PWD/results/smoke" place_empty_cup
```

The output contains `config.json`, `results.csv`, `videos/` and `diagnostics/`.
Inspect the episode's Boolean success and video, then stop the server. One episode
is a smoke test, not a success-rate estimate. Set TEST_NUM explicitly.

## 7. Five NoiseICL campaigns

On DAS6, initialize the existing workspace before planning:

```bash
source "$HOME/nbhagat-workspace-env.sh"
module load prun
conda activate robotwin
cd "$ZERO_WAM_ROOT"
python experiments/scripts/noiseicl.py plan --stage 1 --cluster das6 \
  --directory experiments/stages/01_smoke
python experiments/scripts/noiseicl.py submit \
  --directory experiments/stages/01_smoke --limit 1
```

Inspect the first full trial before submitting the remaining six with `--limit 6`.
Use `--cluster snellius` for the corresponding cluster launcher. Jobs are planned
without submission; the submit command is explicit. The default job ceiling is
three hours because online reversal generates a new teacher prediction at every
chunk. DAS6 long jobs belong in an allowed night/weekend window or an approved
site arrangement; the 15-minute daytime template cannot cover full paired trials.

| Stage | Purpose | Additional inputs |
| --- | --- | --- |
| 1 | One full matched comparison on each of seven unseen tasks | None beyond installed assets/model/human references |
| 2 | Twenty distinct scene slots per unseen task | Override with --scenes if needed |
| 3 | First 100 fixed-history noise pairs as a collection benchmark | --contexts /path/to/index.json |
| 4 | Full grouped dataset collection; budgeted pilot targets up to 2000 pairs | --contexts /path/to/index.json |
| 5 | Residual adapter training, seen/unseen evaluation and shuffled-demo control | --dataset /path/to/stage4 |

All stages use `noiseicl.py plan --stage N --directory PATH`. Stage 5 submits
training first; its evaluation jobs depend on successful training. Submission is
bounded by `--limit` (default one); `--retry` preserves completed work and archives
partial attempts. Existing old-guidance campaigns cannot resume under this runtime.
Results and source snapshots stay inside the chosen campaign directory. Run:

```bash
python experiments/scripts/noiseicl.py report --directory CAMPAIGN --plots
```

For collection, supply normalized recorded robot histories, not just human videos.
Print the required JSONL trajectory fields with `noiseicl.py schema`. Each history
file is a CPU tensor dictionary containing `video` and normalized `actions`; the
initial file contains `initial_frame`. Paths may be relative to the manifest.
The simulator seed and human demo IDs must be correct to prevent split leakage.
Build the grouped index before generating Gaussian seeds:

```bash
python experiments/scripts/noiseicl.py index --trajectories seen_trajectories.jsonl \
  --output experiments/datasets/seen_contexts --tasks 5 \
  --trajectories-per-task 20 --points 10
```

The seven unseen tasks are rejected from training data. Shared scenes/demos stay
in one train/validation/test split. Collection needs real seen-task inputs, which
are not created by running the seven-task unseen smoke campaign.

Stage 5 defaults to an eight-block, width-512 patch transformer. For the initial
100-pair overfit check, use --overfit-pairs 100; --architecture mlp selects the
smaller baseline. --width, --layers, --heads, --train-steps and --batch are explicit
planning options. A completed numerical run is not evidence that noise transfer
or the trained adapter improves task success.

## Reproducibility and troubleshooting

- Installer records are in `$ROBOTWIN_ROOT/.zero-wam-setup/`; keep source checkouts
  because CuRobo is editable. Retain checkpoint/data manifests with results.
- Modified sources have `.zerowam-upstream` backups. Evaluation patches change
  `move_stapler_pad` and `stamp_seal` criteria; document these in comparisons.
- `No module named msgpack`: verify the client is using the installed `robotwin`
  environment. `pkg_resources` failures: preserve the pinned setuptools below 81.
- Disk quota failures: check caches, archives, and personal quota before retrying;
  no dependency reinstall is needed. Temporary storage may expire and lack backups.
- GPU hangs: inspect both server and client logs and measure memory. Renderer
  initialization alone does not prove camera capture works alongside the model.
- Validate builds and execution on your target hardware before collecting results.
  General setup/model tests remain; only reward-guidance tests/docs were removed.
