# Installation and evaluation setup

Use two separate environments: `zerowam` for model inference and `robotwin`
for the simulator/evaluation client. Keep source code in persistent storage and
large assets/checkpoints in storage with sufficient personal quota. None of the
download steps requires a GPU.

## 1. Plan storage before installation or downloads

```bash
quota -s
df -h "$HOME" /var/scratch/"$USER"
du -sh ~/.cache ~/Robotwin /var/scratch/"$USER" 2>/dev/null
```

Paths and quota tools vary by cluster. `df` reports shared filesystem capacity,
not your personal allowance. The RoboTwin checkpoint is approximately 35 GB;
allow additional space for environments, simulation assets, data, and temporary
files. Our DAS5 account had a 4 GiB home soft quota and a 40 GiB scratch soft
quota; with roughly 33 GiB already on scratch, the checkpoint did not fit.
Request more suitable storage before starting it. Do not rely on quota grace.

Pip caches can be purged without uninstalling packages:

```bash
python -m pip cache info
python -m pip cache purge
```

Set `PIP_CACHE_DIR` to suitable external storage in future installation jobs.
Do not change `HF_HOME` just to relocate downloads: it can hide an existing HF
login. Use the explicit storage-root option described below. Scratch may expire
and may not be backed up; preserve a restoration recipe and unique results.

## 2. Initialize Conda and prepare Zero-WAM

Initialize Conda using the installation on your cluster, for example on our DAS5
account:

```bash
source /var/scratch/nbhagat/miniforge3/etc/profile.d/conda.sh
```

On Snellius, load the available Conda/CUDA modules instead; exact versions are
cluster-specific. Do not reuse Snellius module names on DAS5.

The tested Zero-WAM stack is Python 3.10, PyTorch 2.9.0, CUDA 12.6. If `zerowam`
already works, activate it rather than reinstalling:

```bash
conda activate zerowam
python -m pip check
```

For a fresh inference environment:

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

Run these commands from the Zero-WAM checkout. For training, see README.md and
`requirements.txt`; training dependencies are not required for this baseline.
The Snellius environment exports are records of installed versions, not complete
portable installers. CUDA-suffixed PyTorch pins require the PyTorch CUDA index.

The successful baseline used an A100 with BF16/FlexAttention and text-only CPU
offloading. Installation success does not establish that this model works on
Titan RTX: precision, attention kernels, memory fit, and any model splitting need
separate validation. Two GPUs do not automatically pool their memory.

## 3. Prepare RoboTwin separately

Follow [setup/robotwin/README.md](setup/robotwin/README.md). From this checkout:

```bash
export ROBOTWIN_ROOT="$HOME/Robotwin"
bash setup/robotwin/install.sh base
```

In allocated build resources, load a compatible toolkit/compiler, set the actual
target GPU architecture, then build source dependencies:

```bash
# A100 = 8.0; Titan RTX = 7.5. This is a build target, not a model compatibility guarantee.
export TORCH_CUDA_ARCH_LIST="7.5"
bash setup/robotwin/install.sh source
```

This recipe uses RoboTwin `2eeec322`, PyTorch 2.4.1, pinned PyTorch3D and CuRobo
v0.7.8, setuptools below 81, msgpack, and our observed Warp pin. It applies the
SAPIEN/MPLib patches and the two evaluation success-condition patches. Do not
subsequently run the full upstream installer: it can overwrite these choices.
The detailed guide covers existing environments and resumable stages.

## 4. Obtain simulator assets, with enough space for extraction

RoboTwin simulation assets are separate from HumanGen and the model checkpoint.
They contain object/robot geometry and textures. Run the CPU asset stage:

```bash
bash setup/robotwin/install.sh assets
bash setup/robotwin/install.sh verify
```

Assets normally live under `$ROBOTWIN_ROOT/assets`. If home is small, move that
directory to suitable shared storage and symlink it back BEFORE downloading.
Keep the repository and editable CuRobo checkout in place. Check both bytes and
file quotas. Extraction needs room for archives and extracted content together.
A partially extracted directory is not a completed asset installation.

## 5. Authenticate and plan checkpoint / HumanGen downloads

Activate `zerowam` and log in once on the cluster:

```bash
conda activate zerowam
hf auth login
hf auth whoami
```

Never put tokens in scripts. Jobs normally see saved credentials through the
shared home directory; keep `HF_HOME`/token settings consistent. See
[setup/zerowam/README.md](setup/zerowam/README.md) for download and CPU Slurm
instructions.

The default only reads Hub metadata and prints a size plan:

```bash
bash setup/zerowam/download.sh all --storage-root /path/to/large/shared/zero-wam
```

After securing sufficient personal quota, execute the plan using the actual
remaining allowance from `quota -s` (in GiB):

```bash
bash setup/zerowam/download.sh all \
  --storage-root /path/to/large/shared/zero-wam \
  --quota-free-gib YOUR_REMAINING_GIB --download --link
```

The default `eval` profile selects only the fixed seven human references, their
VAE latents, and ICL manifests. It does NOT download recorded robot trajectories:
live evaluation uses simulator observations. Use `--profile robotwin` explicitly
for the full simulation subset, including recorded robot videos/actions/latents.
Neither profile downloads the external pretraining corpus.

The script records resolved Hub commits, resumes against those revisions, uses
two workers, checks personal-budget and filesystem headroom, and verifies file
sizes after download. Keep destination `.cache` files to reuse completed work.
If rate-limited, stop duplicate jobs, wait, verify login, and rerun the same
command. No forced redownload is needed. Run only one downloader per storage root.

`--link` creates the paths existing launchers expect, refusing to overwrite an
existing directory or conflicting link:

```text
Zero-WAM/checkpoints/zero-wam-posttrain-robotwin -> external checkpoint
Zero-WAM/data/HumanGen -> external HumanGen
```

## 6. GPU verification, then actual camera capture

Only in an allocated GPU node:

```bash
bash setup/robotwin/install.sh gpu-check
```

This checks GPU availability/planner imports, not a complete rollout. Then test
actual RoboTwin task camera images without loading Zero-WAM. Renderer
initialization alone is insufficient. A missing Vulkan module does not prove
Vulkan is absent; offscreen rendering still needs functioning graphics drivers.
Do not install drivers or use `sudo apt` on managed cluster nodes.

Warp CUDA initialization warnings remain unresolved in our recovered environment,
although one Snellius rollout completed. Investigate failed GPU checks rather
than silently treating them as success. Camera success alone does not establish
motion-planner correctness.

## 7. One closed-loop episode before scaling up

Server and simulator client use separate environments on the same allocated
node, communicating through a local port. Set:

```bash
export MODEL_PATH="$PWD/checkpoints/zero-wam-posttrain-robotwin"
export ROBOTWIN_ROOT="$HOME/Robotwin"
export TEXT_ENCODER_OFFLOAD=1
```

Launch `evaluation/robotwin/launch_server.sh` in `zerowam`, wait for `/healthz`,
then run in `robotwin`:

```bash
TEST_NUM=1 SEED=0 ICL_CFG=5 TARGET_TEXT_CFG=-1 \
  bash evaluation/robotwin/launch_client.sh "$PWD/results/smoke" place_empty_cup
```

The native baseline uses human-video ICL + CFG; it is not a language-only test.
Our successful A100 run used text-only CPU offload, peaked at approximately
34.53 GiB, and took 8 min 50 sec including startup. This is one observed run,
not a guarantee for another GPU/task. GPU memory monitoring, bounded startup/client
timeouts, readiness checks, and server cleanup should be included in batch jobs.
Personal diagnostic templates in ignored `implementation/` are not distributed
installation requirements. Adapt scheduler/module settings for your cluster.

Confirm completed metrics (`total_num=1`), inspect the rollout video, and record
runtime/memory before scheduling multiple episodes. A failed task with completed
metrics is a valid rollout; a crash/hang is not a task-success measurement.

## Evaluation criteria and records

The RoboTwin installer applies revised criteria for `move_stapler_pad` (released
stapler contacting and within the pad, independent of yaw) and `stamp_seal`
(position tolerance 3 cm rather than 1 cm). Document those criteria in results;
backups use `.zerowam-upstream` with the project installer.

Keep environment package records, source revisions, patches, checkpoint/data
manifest, task/reference, scene and sampling seeds, offload settings, and metrics
with experiments. The full paper protocol uses 100 episodes per task per seed;
one smoke episode is not a reproduction of the reported success rate.
