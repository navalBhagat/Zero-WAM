# Recreate the RoboTwin environment

Use this project-owned installer instead of modifying or rerunning RoboTwin's
`script/_install.sh`. It implements that revision's dependency and patch steps
with the versions recovered from our successful Snellius rollout.

This setup is separate from the `zerowam` environment. Source builds require a
Linux x86-64 host, a compatible CUDA toolkit/compiler, and allocated build
resources. A physical GPU is not requested by the installer. Cluster modules,
partitions, accounts, and driver installation remain cluster-specific.

## 1. Create the environment and checkout

From the Zero-WAM checkout on DAS5, initialize Conda using the cluster's supported
setup, then run:

```bash
export ROBOTWIN_ROOT="$HOME/Robotwin"
bash setup/robotwin/install.sh base
```

The installer creates `robotwin` using Python 3.10, installs PyTorch 2.4.1 and
torchvision 0.19.1 from the CUDA 12.1 index, then installs the Python package
snapshot. It clones RoboTwin and checks out `2eeec322`. Existing assets are not
downloaded or removed. Existing dirty repositories are never reset.

If `robotwin` already exists, use a fresh name to preserve it:

```bash
export ROBOTWIN_ENV=robotwin-rebuild
bash setup/robotwin/install.sh base
```

Alternatively, explicitly opt into updating the existing environment:

```bash
ALLOW_EXISTING_ENV=1 bash setup/robotwin/install.sh base
```

If the pip phase fails, fix the reported cause and rerun `base` with that opt-in;
do not delete the entire environment automatically. Use the same ROBOTWIN_ENV
and ROBOTWIN_ROOT for every stage.

## 2. Build source dependencies and apply patches

Inside a compute/build allocation, load a CUDA toolkit and host compiler
compatible with PyTorch's CUDA 12.1 build. Check `nvcc --version`; do not reuse
Snellius module names blindly. Set the GPU architecture explicitly:

```bash
# Titan RTX: 7.5; A100: 8.0. Confirm the actual target card first.
export TORCH_CUDA_ARCH_LIST="7.5"
bash setup/robotwin/install.sh source
```

This stage installs:

- PyTorch3D from commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`.
- CuRobo from tag `v0.7.8`, editable under `Robotwin/envs/curobo`.
- Both builds use `--no-build-isolation --no-deps`; their Python dependencies
  were installed in the base stage, so the build does not silently upgrade them.
- The upstream SAPIEN UTF-8 and MPLib collision-check patches.
- Zero-WAM's two task-success-condition patches.

Each modified Python source has a `.zerowam-upstream` backup. Rerunning `source`
rebuilds source packages and reapplies validated patches. Unrecognized package
source layouts fail rather than silently skipping a patch.

If all prerequisites are available in one build allocation, the default
`install` stage runs `base`, `source`, and `verify` sequentially:

```bash
TORCH_CUDA_ARCH_LIST="7.5" bash setup/robotwin/install.sh install
```

Do not run the upstream full installer afterward: it can overwrite pins and
clone a newer, incompatible CuRobo.

## 3. Download assets (no GPU required)

```bash
bash setup/robotwin/install.sh assets
```

Assets are written according to the pinned RoboTwin download script, normally
under `Robotwin/assets`. Check storage quota first. If authentication is needed:

```bash
conda run --no-capture-output -n robotwin huggingface-cli login
```

Use your custom environment name instead of `robotwin` if applicable. You may
also transfer the existing assets from Snellius rather than downloading again.
Assets, Hugging Face credentials, and model checkpoints are not environment
packages and are not bundled by this installer.

## 4. Verify in stages

Lightweight imports/package checks, without GPU execution:

```bash
bash setup/robotwin/install.sh verify
```

Inside an allocated GPU node:

```bash
bash setup/robotwin/install.sh gpu-check
```

The GPU stage checks both PyTorch and Warp CUDA availability and planner
imports. It does not count imports as proof of rendering or motion planning.
Next run the actual-task camera diagnostic and then a closed-loop episode.
The personal `implementation/test_camera.sbatch` must have its scheduler header
and module commands adapted for DAS5.

## Versions and limitations

`requirements.lock.txt` preserves the Python snapshot recovered from
`robotwin-snellius.yml`, excluding PyTorch and the two source-built packages,
which are installed explicitly. Important compatibility pins include
`setuptools==80.10.2`, `msgpack==1.2.4`, `sapien==3.0.0b1`, and
`warp-lang==1.6.0`. SciPy is pinned to the observed environment's 1.15.3, not the
older upstream requirements value.

Warp 1.6.0 emitted CUDA initialization warnings on Snellius even though an
episode completed. This is an unresolved compatibility issue, not a promise of
GPU portability. `gpu-check` treats unavailable Warp CUDA as a failure. Do not
blindly upgrade Warp: newer releases changed the API expected by CuRobo v1.

This is a repeatable installation recipe, not a byte-for-byte environment image:
the base Conda dependencies are not fully locked; source extensions are compiled
for the selected toolkit and GPU architecture. CuRobo is pinned to a release tag,
and its resolved commit is recorded. Driver/Vulkan support must be validated on
each cluster. Zero-WAM BF16/attention and multi-GPU support on Titan RTX remain
separate work; a working simulator environment does not resolve those issues.

Installed package lists, explicit Conda records, and Git revisions are written
under `Robotwin/.zero-wam-setup/`. Keep those records with experiment results.
Keep the source checkouts in place because CuRobo is installed editable.

Local verification of this recipe covers shell/Python syntax and patch logic.
The full download/build procedure still requires execution on the target cluster.
