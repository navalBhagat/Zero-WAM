#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=progress_matching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=progress_matching_%j.out

set -euo pipefail
module purge
module load 2024 Anaconda3/2024.06-1 CUDA/12.6.0
source "$(conda info --base)/etc/profile.d/conda.sh"
source activate "${ZERO_WAM_ENV:-zerowam}"
export PATH="${CONDA_PREFIX:?}/bin:$PATH"
hash -r
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/temporal-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/temporal-cache-${SLURM_JOB_ID}"
cd "$ZERO_WAM_ROOT"
# Each submission gets a fresh output directory; prior results are never overwritten.
OUTPUT="${TEMPORAL_TEST_OUTPUT:-$ZERO_WAM_ROOT/experiments/progress_matching/job_${SLURM_JOB_ID}}"
python experiments/scripts/progress_matching.py \
    --threads "$OMP_NUM_THREADS" --output "$OUTPUT" "$@"
