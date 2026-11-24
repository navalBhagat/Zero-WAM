#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=noise_oom_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:35:00
#SBATCH --output=noise_oom_%j.out
set -euo pipefail
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
export NOISE_INVERSION_REPAIR_TASKS=stack_blocks_three
export NOISE_INVERSION_PHASE_TIMEOUT=1800 CLIENT_TIMEOUT=1900
# Allocation experiment: retains the full history and normal action limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "$ZERO_WAM_ROOT/experiments/jobs/noise_inversion_repair.sbatch"
