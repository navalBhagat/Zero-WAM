#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=zerowam_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=zerowam_smoke_%j.out

set -eo pipefail
export EXPERIMENT_CLUSTER=snellius
export ZERO_WAM_ROOT="${ZERO_WAM_ROOT:-$HOME/Zero-WAM}"
exec bash "$ZERO_WAM_ROOT/experiments/scripts/run_evaluation.sh"
