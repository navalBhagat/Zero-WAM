# Workspace paths and Conda initialization are owned by the user's DAS6 setup.
source "$HOME/nbhagat-workspace-env.sh"
: "${ZERO_WAM_ROOT:?Workspace helper must set ZERO_WAM_ROOT}"
: "${ROBOTWIN_ROOT:?Workspace helper must set ROBOTWIN_ROOT}"
: "${NBHAGAT_WORKSPACE:?Workspace helper must set NBHAGAT_WORKSPACE}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

activate_experiment_environment() {
    local environment="$1"
    local client_module="${DAS6_ROBOTWIN_CUDA_MODULE:-cuda12.1/toolkit/12.1}"
    local server_module="${DAS6_ZEROWAM_CUDA_MODULE:-cuda12.6/toolkit/12.6}"
    module unload "$client_module" "$server_module" 2>/dev/null || true
    case "$environment" in
        robotwin) module load "$client_module" ;;
        zerowam) module load "$server_module" ;;
        *) echo "Unknown DAS6 environment: $environment" >&2; return 1 ;;
    esac
    conda activate "$environment"
    export PATH="${CONDA_PREFIX:?Environment activation failed}/bin:$PATH"
    hash -r
}
