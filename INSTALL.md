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

The validated inference configuration uses A100, BF16/FlexAttention, and text-only
CPU offload. Older GPUs may require precision/kernel changes; multiple GPUs do
not automatically pool their memory. Training dependencies are separate; see
[the archived README](old_readme.md#training).

## 2. RoboTwin environment

The project installer pins RoboTwin and its source dependencies and applies the
required compatibility/evaluation patches. Do not run RoboTwin's upstream full
installer afterward.

```bash
export ROBOTWIN_ROOT="$HOME/Robotwin"
bash setup/robotwin/install.sh base

# Load a compatible CUDA toolkit/compiler before this stage.
# Use your GPU's compute capability: A100 = 8.0, Titan RTX = 7.5.
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
| `--profile eval` (default) | Seven fixed human videos, their VAE latents, and ICL manifests |
| `--profile robotwin` | Full simulation subset, including recorded robot videos/actions/latents |
| `--workers 1` | Reduce request concurrency; default is 2 |
| `--link` | Create repository paths expected by inference; refuse conflicting existing paths |

Live evaluation uses human references and simulator observations; recorded robot
trajectories are optional. Rerun the same command to resume, preserving destination
`.cache` metadata. Resolved revisions/completion are saved in `download-manifest.json`.
Use a new storage root for a different revision. Run one downloader per root.
On rate limiting, wait, verify authentication, and retry. The space estimate adds
2 GiB headroom but cannot reserve quota or account for other concurrent writes.

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
TEST_NUM=1 SEED=0 ICL_CFG=5 TARGET_TEXT_CFG=-1 \
  bash evaluation/robotwin/launch_client.sh "$PWD/results/smoke" place_empty_cup
```

Metrics are saved under `results/smoke/stseed-*/metrics/place_empty_cup/res.json`;
rollout videos are under the corresponding `visualization` directory. Confirm
`total_num=1`, inspect the video, and stop the server when finished. One completed
episode is a smoke test, not a success-rate estimate. The client default is 100
episodes; set `TEST_NUM` explicitly. Native ICL + CFG is not language-only guidance.

For batch evaluation, include readiness checks, a time limit, GPU memory logging,
and server cleanup. The archived README describes the eight-GPU launcher: it runs
independent server/client pairs, not one model split across eight GPUs.

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
- Setup scripts have local syntax/offline tests. Validate builds and execution on
  your target hardware before collecting results.
