#!/usr/bin/env bash
# Usage: bash setup/robotwin/install.sh [install|base|source|assets|verify|gpu-check]
set -euo pipefail

SETUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SETUP_DIR/../.." && pwd -P)
ENV_NAME=${ROBOTWIN_ENV:-robotwin}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$HOME/Robotwin}
export ROBOTWIN_ROOT
export MAX_JOBS=${MAX_JOBS:-4}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
STAGE=${1:-install}
ROBOTWIN_REV=2eeec322
CUROBO_REV=v0.7.8
PYTORCH3D_REV=75ebeeaea0908c5527e7b1e305fbc7681382db47

usage() {
    echo "Usage: $0 [install|base|source|assets|verify|gpu-check]"
    echo "Set ROBOTWIN_ROOT, ROBOTWIN_ENV, and TORCH_CUDA_ARCH_LIST as needed."
}
case "$STAGE" in install|base|source|assets|verify|gpu-check) ;; *) usage; exit 2 ;; esac
command -v conda >/dev/null || { echo "Initialize Conda first."; exit 1; }
command -v git >/dev/null || { echo "git is required."; exit 1; }
run_env() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

checkout() {
    local url=$1 directory=$2 revision=$3
    if [[ ! -d "$directory" ]]; then
        git clone "$url" "$directory"
    fi
    git -C "$directory" rev-parse --git-dir >/dev/null
    # Never reset a dirty checkout or silently replace a user's revision.
    if ! git -C "$directory" rev-parse --verify "$revision^{commit}" >/dev/null 2>&1; then
        git -C "$directory" fetch origin "$revision"
        if ! git -C "$directory" rev-parse --verify "$revision^{commit}" >/dev/null 2>&1; then
            git -C "$directory" fetch --tags origin
        fi
    fi
    local wanted current
    wanted=$(git -C "$directory" rev-parse "$revision^{commit}")
    current=$(git -C "$directory" rev-parse HEAD)
    if [[ "$current" != "$wanted" ]]; then
        if [[ -n "$(git -C "$directory" status --porcelain)" ]]; then
            echo "Refusing to change dirty checkout: $directory" >&2
            git -C "$directory" status --short
            exit 1
        fi
        git -C "$directory" checkout --detach "$wanted"
    fi
}

record() {
    local records="$ROBOTWIN_ROOT/.zero-wam-setup"
    mkdir -p "$records"
    run_env python -m pip freeze > "$records/pip-freeze.txt"
    conda list -n "$ENV_NAME" --explicit > "$records/conda-explicit.txt"
    git -C "$ROBOTWIN_ROOT" rev-parse HEAD > "$records/robotwin-revision.txt"
    if [[ -d "$ROBOTWIN_ROOT/envs/curobo/.git" ]]; then
        git -C "$ROBOTWIN_ROOT/envs/curobo" rev-parse HEAD > "$records/curobo-revision.txt"
    fi
}

base() {
    # Reuse an existing environment only when explicitly opting in.
    if conda run -n "$ENV_NAME" python -c 'pass' >/dev/null 2>&1; then
        if [[ "${ALLOW_EXISTING_ENV:-0}" != 1 ]]; then
            echo "Environment $ENV_NAME exists. Use a new ROBOTWIN_ENV name, or set ALLOW_EXISTING_ENV=1 to update it."
            exit 1
        fi
        conda env update -n "$ENV_NAME" -f "$SETUP_DIR/environment.yml"
    else
        conda env create -n "$ENV_NAME" -f "$SETUP_DIR/environment.yml"
    fi
    run_env python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
    checkout https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_ROOT" "$ROBOTWIN_REV"
    git -C "$ROBOTWIN_ROOT" submodule update --init --recursive
    # Install torch first, so later source builds can import it.
    run_env python -m pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121
    run_env python -m pip install -r "$SETUP_DIR/requirements.lock.txt"
    echo "Base packages installed. Next: source stage on allocated build resources."
}

source_builds() {
    command -v nvcc >/dev/null || { echo "Load the CUDA toolkit/compiler before source builds."; exit 1; }
    : "${TORCH_CUDA_ARCH_LIST:?Set TORCH_CUDA_ARCH_LIST explicitly: 7.5 for Titan RTX, 8.0 for A100}"
    export TORCH_CUDA_ARCH_LIST
    local revision
    revision=$(git -C "$ROBOTWIN_ROOT" rev-parse "$ROBOTWIN_REV^{commit}")
    [[ "$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)" == "$revision" ]] || {
        echo "RoboTwin must be at $ROBOTWIN_REV. Run base stage first."; exit 1;
    }
    nvcc --version
    run_env python -c 'import torch; print("Torch:", torch.__version__, "runtime:", torch.version.cuda)'
    mkdir -p "$ROBOTWIN_ROOT/third_party"
    checkout https://github.com/facebookresearch/pytorch3d.git \
        "$ROBOTWIN_ROOT/third_party/pytorch3d" "$PYTORCH3D_REV"
    run_env python -m pip install --no-build-isolation --no-deps \
        "$ROBOTWIN_ROOT/third_party/pytorch3d"
    checkout https://github.com/NVlabs/curobo.git "$ROBOTWIN_ROOT/envs/curobo" "$CUROBO_REV"
    run_env python -m pip install --no-build-isolation --no-deps -e "$ROBOTWIN_ROOT/envs/curobo"
    run_env python "$SETUP_DIR/patch.py" --robotwin-root "$ROBOTWIN_ROOT" --project-root "$PROJECT_ROOT"
    run_env python -m pip check
    record
    echo "Source builds and patches complete. Assets and GPU checks are separate stages."
}

verify() {
    run_env python -m pip check
    # Imports only; no CUDA execution or planner initialization here.
    run_env python -c 'import torch, sapien, mplib, pytorch3d, curobo, msgpack, websockets; print("Basic imports passed; CUDA execution untested.")'
    record
}

case "$STAGE" in
    install) base; source_builds; verify ;;
    base) base ;;
    source) source_builds ;;
    verify) verify ;;
    assets)
        cd "$ROBOTWIN_ROOT"
        # The upstream script otherwise continues after failed downloads/unzip.
        run_env bash -eo pipefail script/_download_assets.sh
        ;;
    gpu-check)
        cd "$ROBOTWIN_ROOT"
        run_env python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import warp as wp; wp.init(); assert wp.is_cuda_available(), "Warp CUDA is unavailable"; from curobo.types.math import Pose; from curobo.wrap.reacher.motion_gen import MotionGen; from envs.robot.planner import CuroboPlanner; print("GPU and planner imports passed; actual rendering/rollout still required.")'
        ;;
esac
