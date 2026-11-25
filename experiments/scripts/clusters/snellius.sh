# Sourced by run_evaluation.sh; retains the existing Snellius module stack.
module purge
module load 2024 Anaconda3/2024.06-1 CUDA/12.6.0
source "$(conda info --base)/etc/profile.d/conda.sh"

activate_experiment_environment() {
    conda activate "$1"
    export PATH="${CONDA_PREFIX:?Environment activation failed}/bin:$PATH"
    hash -r
}
